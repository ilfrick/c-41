//! Darkroom editing view — single-image editing with IOP module stack.
//!
//! Phase 3-ui-16/17: the live preview params live in their **module rows** in the
//! right panel (Exposure / Velvia / Split-toning render as `adw::ExpanderRow`s
//! whose enable switch gates the pipeline stage and whose sliders drive the
//! params), and a **live RGB histogram** under the image tracks the processed
//! output. Shared preview state is bundled in [`PreviewCtx`] (weak widget refs
//! with `Rc` data) so every callback re-runs `crate::preview::apply_pipeline`,
//! refreshes the histogram, and repaints. A header before/after toggle shows
//! the unprocessed image on demand (ui-17); a Reset action restores defaults and
//! rebuilds the panel (ui-19); clicking the image samples the processed pixel
//! into a colour-picker readout (ui-20). Remaining catalog modules stay as inert
//! toggle rows. Navigation back to the lighttable is via the NavigationView pop.

use adw::prelude::*;
use glib::clone;
use std::cell::RefCell;
use std::rc::Rc;
use crate::dialogs;
use crate::preview::{Histogram, PreviewParams};

/// Placeholder shown in the colour-picker readout before/after a sample.
const PICKER_PROMPT: &str = "Pick: click the image to sample a pixel";

/// Decoded preview image kept for live re-processing.
#[derive(Clone)]
struct BaseImage {
    bytes: Vec<u8>,
    width: i32,
    height: i32,
    rowstride: usize,
    nch: usize,
}

/// Shared live-preview state, cloned into every widget callback. Widgets are
/// held as `glib::WeakRef` to avoid widget→closure→widget reference cycles
/// (the page is dropped on navigation); the `Rc<RefCell<…>>` data is shared
/// strongly. Cloning is a handful of cheap refcount bumps.
#[derive(Clone)]
struct PreviewCtx {
    picture: glib::WeakRef<gtk4::Picture>,
    hist_area: glib::WeakRef<gtk4::DrawingArea>,
    /// Colour-picker readout; reset to its prompt on each re-render so a stale
    /// sample never lingers after an edit changes the displayed pixels.
    picker: glib::WeakRef<gtk4::Label>,
    base: Rc<RefCell<Option<BaseImage>>>,
    params: Rc<RefCell<PreviewParams>>,
    hist: Rc<RefCell<Histogram>>,
    /// While set, the preview shows the unprocessed image (before/after toggle).
    bypass: Rc<std::cell::Cell<bool>>,
    /// Debounced DB autosave of the current params (None when there's no db).
    autosave: Option<Rc<AutoSave>>,
}

/// Debounced writer that persists the current params a short time after the last
/// edit (so slider drags don't write per-tick), with an explicit flush on close.
struct AutoSave {
    db_path: String,
    file_path: String,
    params: Rc<RefCell<PreviewParams>>,
    pending: RefCell<Option<glib::SourceId>>,
}

impl AutoSave {
    /// (Re)arm the debounce timer; the last edit within the window wins.
    fn arm(self: &Rc<Self>) {
        if let Some(id) = self.pending.borrow_mut().take() {
            id.remove();
        }
        let this = self.clone();
        let id = glib::timeout_add_local_once(std::time::Duration::from_millis(800), move || {
            *this.pending.borrow_mut() = None;
            crate::persist::save_params(&this.db_path, &this.file_path, &this.params.borrow());
        });
        *self.pending.borrow_mut() = Some(id);
    }

    /// Cancel any pending timer and save immediately (final flush on page close).
    fn flush(&self) {
        if let Some(id) = self.pending.borrow_mut().take() {
            id.remove();
        }
        crate::persist::save_params(&self.db_path, &self.file_path, &self.params.borrow());
    }
}

/// The params the preview is currently showing: the live params, or their
/// bypassed (all-stages-off) form while the before/after toggle is active.
fn effective_params(ctx: &PreviewCtx) -> PreviewParams {
    let p = *ctx.params.borrow();
    if ctx.bypass.get() {
        p.bypassed()
    } else {
        p
    }
}

/// Re-run the pipeline over the base preview, refresh the histogram, and repaint
/// both the image and the histogram. Reads the current params (a `Copy`
/// snapshot, so no borrow is held across `apply_pipeline`). A no-op until the
/// image has decoded or if the page widgets have been dropped.
fn render_preview(ctx: &PreviewCtx) {
    let Some(picture) = ctx.picture.upgrade() else { return };
    let params = effective_params(ctx);
    if let Some(b) = ctx.base.borrow().as_ref() {
        let (w, h) = (b.width as usize, b.height as usize);
        let processed = crate::preview::apply_pipeline(&b.bytes, w, h, b.rowstride, b.nch, &params);

        *ctx.hist.borrow_mut() = crate::preview::compute_histogram(&processed, w, h, b.rowstride, b.nch);
        if let Some(area) = ctx.hist_area.upgrade() {
            area.queue_draw();
        }

        let fmt = if b.nch == 4 {
            gtk4::gdk::MemoryFormat::R8g8b8a8
        } else {
            gtk4::gdk::MemoryFormat::R8g8b8
        };
        let gbytes = glib::Bytes::from_owned(processed);
        let tex = gtk4::gdk::MemoryTexture::new(b.width, b.height, fmt, &gbytes, b.rowstride);
        picture.set_paintable(Some(&tex));

        // The displayed pixels just changed; any prior pick is now stale.
        if let Some(label) = ctx.picker.upgrade() {
            label.set_text(PICKER_PROMPT);
        }

        // Debounced persistence of the (live, not bypassed) params.
        if let Some(autosave) = &ctx.autosave {
            autosave.arm();
        }
    }
}

/// Build a NavigationPage for editing a single image at `file_path`.
///
/// The page title is set to the filename. The caller pushes this page onto
/// an `adw::NavigationView` and pops it to return to the lighttable. `db_path`
/// is the catalogue database used to restore the image's saved preview params
/// on open and persist them when the page is closed (empty string = no db).
pub fn darkroom_page(file_path: &str, db_path: &str) -> adw::NavigationPage {
    let filename = std::path::Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file_path)
        .to_string();

    // ── Main image area ────────────────────────────────────────────────────
    let picture = gtk4::Picture::builder()
        .hexpand(true)
        .vexpand(true)
        .content_fit(gtk4::ContentFit::Contain)
        .build();

    // ── Live RGB histogram strip under the image ───────────────────────────
    let hist_area = gtk4::DrawingArea::builder()
        .height_request(120)
        .hexpand(true)
        .build();

    // ── Colour-picker readout (populated by the click gesture below) ───────
    let picker_label = gtk4::Label::builder()
        .label(PICKER_PROMPT)
        .xalign(0.0)
        .margin_start(8)
        .margin_top(2)
        .margin_bottom(4)
        .build();
    picker_label.add_css_class("monospace");

    // Shared live-preview state. Params are seeded from the DB (saved on a
    // previous edit) before the panel/preview are built, so the sliders and the
    // first render reflect the restored values.
    let params = Rc::new(RefCell::new(initial_params(
        crate::persist::load_saved(db_path, file_path),
        crate::raw_preview::is_raw_path(file_path),
    )));
    let autosave = (!db_path.is_empty()).then(|| {
        Rc::new(AutoSave {
            db_path: db_path.to_string(),
            file_path: file_path.to_string(),
            params: params.clone(),
            pending: RefCell::new(None),
        })
    });
    let ctx = PreviewCtx {
        picture: picture.downgrade(),
        hist_area: hist_area.downgrade(),
        picker: picker_label.downgrade(),
        base: Rc::new(RefCell::new(None)),
        params,
        hist: Rc::new(RefCell::new([[0u32; 256]; 3])),
        bypass: Rc::new(std::cell::Cell::new(false)),
        autosave,
    };

    // The histogram paints from the shared `hist` buffer (no widget captured,
    // so no cycle with hist_area).
    let hist_for_draw = ctx.hist.clone();
    hist_area.set_draw_func(move |_, cr, w, h| {
        draw_histogram(cr, w, h, &hist_for_draw.borrow());
    });

    // Load + decode the image asynchronously so the page appears immediately.
    // Camera raws go through the Rust raw decoder (gdk-pixbuf can't read them);
    // everything else through gdk-pixbuf. Both yield an 8-bit `BaseImage`.
    let path_for_load = file_path.to_string();
    glib::spawn_future_local(clone!(@strong ctx => async move {
        let base = if crate::raw_preview::is_raw_path(&path_for_load) {
            // Decode + demosaic off the main thread (it's heavy); downscale to a
            // responsive preview size.
            let p = path_for_load.clone();
            gio::spawn_blocking(move || {
                crate::raw_preview::decode_raw_preview(&p, crate::raw_preview::PREVIEW_MAX_DIM)
                    .map(|rp| BaseImage {
                        bytes: rp.bytes,
                        width: rp.width,
                        height: rp.height,
                        rowstride: rp.rowstride,
                        nch: rp.nch,
                    })
            })
            .await
            .ok()
            .flatten()
        } else {
            let p = path_for_load.clone();
            let data = gio::spawn_blocking(move || std::fs::read(&p).ok())
                .await
                .ok()
                .flatten();
            data.and_then(|data| {
                let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
                let _ = loader.write(&data);
                let _ = loader.close();
                loader.pixbuf().map(|pb| BaseImage {
                    bytes: pb.read_pixel_bytes().to_vec(),
                    width: pb.width(),
                    height: pb.height(),
                    rowstride: pb.rowstride() as usize,
                    nch: pb.n_channels() as usize,
                })
            })
        };
        match base {
            Some(base) => {
                *ctx.base.borrow_mut() = Some(base);
                render_preview(&ctx);
            }
            // Don't make a failed decode (unsupported/corrupt raw, unreadable
            // file) an invisible blank — log it (no toast overlay here yet).
            None => eprintln!("darkroom preview: could not decode {path_for_load}"),
        }
    }));

    // ── Colour picker: click the image to read the processed pixel ─────────
    let click = gtk4::GestureClick::new();
    let pick_ctx = ctx.clone();
    let pick_pic = picture.downgrade();
    let pick_label = picker_label.downgrade();
    click.connect_pressed(move |_, _n, x, y| {
        let (Some(pic), Some(label)) = (pick_pic.upgrade(), pick_label.upgrade()) else {
            return;
        };
        if let Some(b) = pick_ctx.base.borrow().as_ref() {
            let (w, h) = (b.width as usize, b.height as usize);
            match crate::preview::map_widget_to_image(
                pic.width() as f64, pic.height() as f64, w, h, x, y,
            ) {
                Some((px, py)) => {
                    // Sample what's displayed: re-run the (bypass-aware) pipeline.
                    let processed = crate::preview::apply_pipeline(
                        &b.bytes, w, h, b.rowstride, b.nch, &effective_params(&pick_ctx),
                    );
                    if let Some((r, g, bl)) =
                        crate::preview::sample_pixel(&processed, w, h, b.rowstride, b.nch, px, py)
                    {
                        label.set_text(&format!(
                            "Pick ({px},{py}):  R {r}  G {g}  B {bl}   #{r:02X}{g:02X}{bl:02X}"
                        ));
                    }
                }
                None => label.set_text("Pick: (outside image)"),
            }
        }
    });
    picture.add_controller(click);

    // ── Left: image over histogram + picker readout ────────────────────────
    let image_area = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .hexpand(true)
        .build();
    image_area.append(&picture);
    image_area.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    image_area.append(&hist_area);
    image_area.append(&picker_label);

    // ── IOP module list (right panel) — hosts the live param widgets ───────
    let (modules_panel, panel_box) = build_modules_panel(&ctx);

    // ── Split view: image | modules ────────────────────────────────────────
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    content.append(&image_area);
    content.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    content.append(&modules_panel);

    // ── Header bar with Export button ─────────────────────────────────────
    let header = adw::HeaderBar::new();
    let title_widget = adw::WindowTitle::new(&filename, "Darkroom");
    header.set_title_widget(Some(&title_widget));

    // Before/after: while active, show the unprocessed image + its histogram.
    let before_after_btn = gtk4::ToggleButton::builder()
        .icon_name("view-reveal-symbolic")
        .tooltip_text("Show original (before/after)")
        .build();
    // Tooltips aren't reliably exposed as the accessible name for icon-only
    // buttons, so set it explicitly.
    before_after_btn.update_property(&[gtk4::accessible::Property::Label("Show original")]);
    let before_after_ctx = ctx.clone();
    before_after_btn.connect_toggled(move |b| {
        before_after_ctx.bypass.set(b.is_active());
        render_preview(&before_after_ctx);
    });
    header.pack_start(&before_after_btn);

    // Reset: restore default params and rebuild the panel so the sliders follow.
    let reset_btn = gtk4::Button::builder()
        .icon_name("edit-undo-symbolic")
        .tooltip_text("Reset all adjustments")
        .build();
    reset_btn.update_property(&[gtk4::accessible::Property::Label("Reset all adjustments")]);
    let reset_ctx = ctx.clone();
    let reset_panel = panel_box.downgrade();
    let reset_ba = before_after_btn.downgrade();
    reset_btn.connect_clicked(move |_| {
        *reset_ctx.params.borrow_mut() = PreviewParams::default();
        reset_ctx.bypass.set(false); // source of truth for bypass
        if let Some(ba) = reset_ba.upgrade() {
            ba.set_active(false); // sync the button visual (bypass already cleared)
        }
        if let Some(panel) = reset_panel.upgrade() {
            while let Some(child) = panel.first_child() {
                panel.remove(&child);
            }
            populate_modules(&panel, &reset_ctx);
        }
        render_preview(&reset_ctx);
    });
    header.pack_start(&reset_btn);

    let export_btn = gtk4::Button::builder()
        .label("Export")
        .tooltip_text("Export this image")
        .build();
    export_btn.add_css_class("suggested-action");
    let path_for_export = file_path.to_string();
    export_btn.connect_clicked(move |btn| {
        if let Some(root) = btn.root().and_downcast::<gtk4::Window>() {
            dialogs::show_export_dialog(
                root.upcast_ref::<gtk4::Window>(),
                vec![path_for_export.clone()],
                |msg| eprintln!("{msg}"), // darkroom view has no toast overlay yet
            );
        }
    });
    header.pack_end(&export_btn);

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&content));

    let page = adw::NavigationPage::builder()
        .title(&filename)
        .child(&toolbar_view)
        .build();

    // Flush any pending autosave when the page is popped back to the lighttable
    // (the debounce in render_preview covers edits before an abrupt app quit).
    let save_ctx = ctx.clone();
    page.connect_hidden(move |_| {
        if let Some(autosave) = &save_ctx.autosave {
            autosave.flush();
        }
    });
    page
}

/// A labelled horizontal slider row (used as an `ExpanderRow` child for a
/// single IOP parameter).
struct LabeledSlider {
    row: gtk4::Box,
    scale: gtk4::Scale,
}

/// Build a `[label] [────slider────]` row with a value read-out on the right.
fn labeled_slider(label: &str, min: f64, max: f64, step: f64, value: f64) -> LabeledSlider {
    let scale = gtk4::Scale::with_range(gtk4::Orientation::Horizontal, min, max, step);
    scale.set_value(value);
    scale.set_hexpand(true);
    scale.set_draw_value(true);
    scale.set_value_pos(gtk4::PositionType::Right);

    let lbl = gtk4::Label::new(Some(label));
    lbl.set_xalign(0.0);
    // Widest param label is 8 chars ("Compress", "Shad hue"); fixing the column
    // width keeps every slider track left-aligned across rows.
    lbl.set_width_chars(8);

    let row = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .margin_start(12).margin_end(12).margin_top(4).margin_bottom(4)
        .build();
    row.append(&lbl);
    row.append(&scale);
    LabeledSlider { row, scale }
}

/// Module-stack panel: the darktable module groups (base/tone/color/correct/
/// effect) from [`crate::catalog`]. Modules backed by a migrated `darkroom-core`
/// IOP (Exposure, Velvia) render as expandable rows with a live enable switch
/// and parameter sliders wired to the preview pipeline; the rest are inert
/// enable-toggle rows (history-stack wiring is a later milestone).
fn build_modules_panel(ctx: &PreviewCtx) -> (gtk4::Widget, gtk4::Box) {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .build();
    populate_modules(&panel, ctx);

    // Scrollable so the (long) module list never blows out the window height.
    let scrolled = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vexpand(true)
        .width_request(320)
        .child(&panel)
        .build();
    (scrolled.upcast(), panel)
}

/// (Re)build the module rows into `panel`, seeding each live module's widgets
/// from the current `ctx.params`. Called on first build and on Reset (after the
/// panel is cleared) so the sliders reflect the reset defaults.
///
/// Invariant for module builders: set each widget's *initial* value (slider
/// value, `enable_expansion`) **before** connecting its `value_changed` /
/// `*_notify` handler. Otherwise this rebuild — run inside the Reset handler —
/// would fire those handlers per row and re-enter `render_preview` mid-rebuild.
/// `module_expander`/`add_param_slider` already follow this (build then connect).
fn populate_modules(panel: &gtk4::Box, ctx: &PreviewCtx) {
    let header = gtk4::Label::builder()
        .label("Modules")
        .halign(gtk4::Align::Start)
        .build();
    header.add_css_class("title-4");
    panel.append(&header);

    for group in crate::catalog::module_catalog() {
        let pg = adw::PreferencesGroup::builder().title(group.name).build();
        for mi in group.modules {
            match mi.label {
                "Exposure" => pg.add(&exposure_module_row(ctx)),
                "Velvia" => pg.add(&velvia_module_row(ctx)),
                "Split-toning" => pg.add(&splittoning_module_row(ctx)),
                "Monochrome" => pg.add(&monochrome_module_row(ctx)),
                "Sigmoid" => pg.add(&sigmoid_module_row(ctx)),
                _ => pg.add(&inert_module_row(mi.label, mi.default_on)),
            }
        }
        panel.append(&pg);
    }
}

/// A placeholder module row: title + enable switch not yet wired to anything.
fn inert_module_row(label: &str, default_on: bool) -> adw::ActionRow {
    let row = adw::ActionRow::builder().title(label).build();
    let toggle = gtk4::Switch::builder()
        .active(default_on)
        .valign(gtk4::Align::Center)
        .build();
    row.add_suffix(&toggle);
    row.set_activatable_widget(Some(&toggle));
    row
}

/// Catalog labels that [`build_modules_panel`] dispatches to a *live* preview
/// module (everything else renders as an inert toggle). The match arms below
/// use these same literals; `catalog_has_live_modules` guards against a catalog
/// rename silently dropping a module back to inert. Test-only: it exists purely
/// as the contract checked by that test.
#[cfg(test)]
const LIVE_MODULE_LABELS: &[&str] = &["Exposure", "Velvia", "Split-toning", "Monochrome", "Sigmoid"];

// Borrow invariant for the closures below: GTK callbacks run on the main
// thread and never re-enter while a `params` borrow is held — each closure
// takes a short-lived `borrow_mut()` (dropped at the statement end) before
// `render_preview(&ctx)` snapshots params, so the two never overlap.

/// Build a live `ExpanderRow` for one IOP module: title/subtitle, a built-in
/// enable switch wired to `set_enabled`, and the param sliders added by
/// `add_params`. `enabled` seeds the switch from the current params.
fn module_expander(
    ctx: &PreviewCtx,
    title: &str,
    subtitle: &str,
    enabled: bool,
    set_enabled: fn(&mut PreviewParams, bool),
    add_params: impl FnOnce(&adw::ExpanderRow, &PreviewCtx),
) -> adw::ExpanderRow {
    let expander = adw::ExpanderRow::builder()
        .title(title)
        .subtitle(subtitle)
        .show_enable_switch(true)
        .enable_expansion(enabled)
        .build();
    let ctx_cl = ctx.clone();
    expander.connect_enable_expansion_notify(move |e| {
        set_enabled(&mut ctx_cl.params.borrow_mut(), e.enables_expansion());
        render_preview(&ctx_cl);
    });
    add_params(&expander, ctx);
    expander
}

/// Add one parameter slider to `expander`: `init` seeds it, `set` writes the
/// value into the shared params, then the preview re-renders. Arg-heavy by
/// nature (it is a slider builder); the explicit list keeps call sites readable.
#[allow(clippy::too_many_arguments)]
fn add_param_slider(
    expander: &adw::ExpanderRow,
    ctx: &PreviewCtx,
    label: &str,
    min: f64,
    max: f64,
    step: f64,
    init: f64,
    set: fn(&mut PreviewParams, f32),
) {
    let row = labeled_slider(label, min, max, step, init);
    let ctx_cl = ctx.clone();
    row.scale.connect_value_changed(move |s| {
        set(&mut ctx_cl.params.borrow_mut(), s.value() as f32);
        render_preview(&ctx_cl);
    });
    expander.add_row(&row.row);
}

/// Exposure module: enable switch gates `exposure_on`; EV / black-point sliders.
fn exposure_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Exposure", "EV + black point", p0.exposure_on,
        |p, on| p.exposure_on = on,
        |e, ctx| {
            // EV: scale = 2^ev.
            add_param_slider(e, ctx, "EV", -3.0, 3.0, 0.01, p0.ev as f64,
                |p, v| p.ev = v);
            // Black point: lifted before scaling (out = (in - black) * scale).
            add_param_slider(e, ctx, "Black", 0.0, 0.2, 0.001, p0.black as f64,
                |p, v| p.black = v);
        })
}

/// Velvia module: enable switch gates `velvia_on`; strength slider (0..100).
fn velvia_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Velvia", "saturation boost", p0.velvia_on,
        |p, on| p.velvia_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Strength", 0.0, 100.0, 1.0, p0.velvia_strength as f64,
                |p, v| p.velvia_strength = v);
        })
}

/// Split-toning module: enable switch gates `split_on`; shadow/highlight
/// hue+saturation, balance and compress sliders. Hue/sat/balance are 0..1;
/// compress is the C 0..100 slider (pre-scaled in apply_pipeline).
fn splittoning_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Split-toning", "shadow / highlight hues", p0.split_on,
        |p, on| p.split_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Shad hue", 0.0, 1.0, 0.001, p0.split_shadow_hue as f64,
                |p, v| p.split_shadow_hue = v);
            add_param_slider(e, ctx, "Shad sat", 0.0, 1.0, 0.01, p0.split_shadow_sat as f64,
                |p, v| p.split_shadow_sat = v);
            add_param_slider(e, ctx, "High hue", 0.0, 1.0, 0.001, p0.split_highlight_hue as f64,
                |p, v| p.split_highlight_hue = v);
            add_param_slider(e, ctx, "High sat", 0.0, 1.0, 0.01, p0.split_highlight_sat as f64,
                |p, v| p.split_highlight_sat = v);
            add_param_slider(e, ctx, "Balance", 0.0, 1.0, 0.01, p0.split_balance as f64,
                |p, v| p.split_balance = v);
            add_param_slider(e, ctx, "Compress", 0.0, 100.0, 1.0, p0.split_compress as f64,
                |p, v| p.split_compress = v);
        })
}

/// Monochrome module: enable switch gates `mono_on`; R/G/B grayscale mix weights
/// (channelmixer GRAY mode). Weights may go negative for dramatic B&W contrast.
fn monochrome_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Monochrome", "B&W channel mixer", p0.mono_on,
        |p, on| p.mono_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Red", -1.0, 2.0, 0.01, p0.mono_r as f64,
                |p, v| p.mono_r = v);
            add_param_slider(e, ctx, "Green", -1.0, 2.0, 0.01, p0.mono_g as f64,
                |p, v| p.mono_g = v);
            add_param_slider(e, ctx, "Blue", -1.0, 2.0, 0.01, p0.mono_b as f64,
                |p, v| p.mono_b = v);
        })
}

/// Choose the initial params for a freshly-opened image: the `saved` edit if
/// any, else defaults — with the sigmoid display tone-map defaulted ON for raws
/// that have **no saved edit** (scene-linear input; JPEGs are display-referred,
/// and a user who saved sigmoid off keeps it off). Pure, so the seeding policy
/// is unit-tested rather than buried in the GTK builder.
fn initial_params(saved: Option<PreviewParams>, is_raw: bool) -> PreviewParams {
    let mut p = saved.unwrap_or_default();
    if saved.is_none() && is_raw {
        p.sigmoid_on = true;
    }
    p
}

/// Sigmoid module: enable switch gates `sigmoid_on` (the scene-linear → display
/// tone map); contrast and skew sliders. Defaults on for raws (set at page load).
fn sigmoid_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Sigmoid", "tone mapping", p0.sigmoid_on,
        |p, on| p.sigmoid_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Contrast", 0.1, 10.0, 0.05, p0.sigmoid_contrast as f64,
                |p, v| p.sigmoid_contrast = v);
            add_param_slider(e, ctx, "Skew", -1.0, 1.0, 0.01, p0.sigmoid_skew as f64,
                |p, v| p.sigmoid_skew = v);
        })
}

/// Paint the RGB histogram: dark backdrop with one translucent filled curve per
/// channel, each normalised to the global max bin.
fn draw_histogram(cr: &gtk4::cairo::Context, w: i32, h: i32, hist: &Histogram) {
    let (w, h) = (w as f64, h as f64);
    cr.set_source_rgb(0.10, 0.10, 0.10);
    let _ = cr.paint();

    let max = hist
        .iter()
        .flat_map(|c| c.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1) as f64;

    let colours = [(0.90, 0.25, 0.25), (0.25, 0.85, 0.30), (0.35, 0.55, 0.95)];
    for (ch, &(r, g, b)) in colours.iter().enumerate() {
        cr.set_source_rgba(r, g, b, 0.55);
        cr.move_to(0.0, h);
        for (bin, &count) in hist[ch].iter().enumerate() {
            let x = bin as f64 / 255.0 * w;
            let y = h - (count as f64 / max) * h;
            cr.line_to(x, y);
        }
        cr.line_to(w, h);
        cr.close_path();
        let _ = cr.fill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_params_seeding_matrix() {
        use crate::preview::PreviewParams;
        // no saved edit + raw ⇒ sigmoid defaulted on
        assert!(initial_params(None, true).sigmoid_on);
        // no saved edit + JPEG ⇒ left off (default)
        assert!(!initial_params(None, false).sigmoid_on);
        // a saved edit with sigmoid OFF on a raw ⇒ respected (not re-enabled)
        let mut off = PreviewParams::default();
        off.sigmoid_on = false;
        off.ev = 0.3; // a real saved edit
        assert!(!initial_params(Some(off), true).sigmoid_on);
        // a saved edit is returned verbatim
        let mut on = PreviewParams::default();
        on.sigmoid_on = true;
        assert_eq!(initial_params(Some(on), false), on);
    }

    /// Every live-module label must still exist verbatim in the catalog, else
    /// `build_modules_panel`'s string dispatch silently renders it inert.
    #[test]
    fn catalog_has_live_modules() {
        let labels: Vec<&str> = crate::catalog::module_catalog()
            .iter()
            .flat_map(|g| g.modules.iter().map(|m| m.label))
            .collect();
        for live in LIVE_MODULE_LABELS {
            assert!(
                labels.contains(live),
                "live module {live:?} is no longer in the catalog — update \
                 LIVE_MODULE_LABELS and the build_modules_panel match arms"
            );
        }
    }
}
