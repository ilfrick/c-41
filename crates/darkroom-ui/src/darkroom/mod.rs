//! Darkroom editing view — single-image editing with IOP module stack.
//!
//! Phase 3-ui-14: the live preview params now live in their **module rows** in
//! the right panel (not a separate controls bar), converging the module-stack
//! UI with the preview pipeline. Modules backed by a migrated `darkroom-core`
//! IOP (Exposure, Velvia) render as `adw::ExpanderRow`s whose built-in enable
//! switch gates the corresponding pipeline stage (`exposure_on` / `velvia_on`)
//! and whose child sliders drive the stage params; every change re-runs
//! `crate::preview::apply_pipeline` over the preview. Remaining catalog modules
//! stay as inert enable-toggle rows. Navigation back to the lighttable is via
//! the NavigationView pop action.

use adw::prelude::*;
use glib::clone;
use std::cell::RefCell;
use std::rc::Rc;
use crate::dialogs;
use crate::preview::PreviewParams;

/// Decoded preview image kept for live re-processing.
#[derive(Clone)]
struct BaseImage {
    bytes: Vec<u8>,
    width: i32,
    height: i32,
    rowstride: usize,
    nch: usize,
}

/// Paint `picture` with the base preview run through the live `PreviewParams`
/// pipeline (exposure → velvia, via migrated `darkroom-core` IOPs), uploading
/// the result as a gdk::MemoryTexture.
fn render_preview(
    picture: &gtk4::Picture,
    base: &Rc<RefCell<Option<BaseImage>>>,
    params: &PreviewParams,
) {
    if let Some(b) = base.borrow().as_ref() {
        let processed = crate::preview::apply_pipeline(
            &b.bytes, b.width as usize, b.height as usize, b.rowstride, b.nch, params,
        );
        let fmt = if b.nch == 4 {
            gtk4::gdk::MemoryFormat::R8g8b8a8
        } else {
            gtk4::gdk::MemoryFormat::R8g8b8
        };
        let gbytes = glib::Bytes::from_owned(processed);
        let tex = gtk4::gdk::MemoryTexture::new(b.width, b.height, fmt, &gbytes, b.rowstride);
        picture.set_paintable(Some(&tex));
    }
}

/// Build a NavigationPage for editing a single image at `file_path`.
///
/// The page title is set to the filename. The caller pushes this page onto
/// an `adw::NavigationView` and pops it to return to the lighttable.
pub fn darkroom_page(file_path: &str) -> adw::NavigationPage {
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

    // Shared decoded preview + live pipeline params, re-read on every slider
    // change to re-process the preview.
    let base: Rc<RefCell<Option<BaseImage>>> = Rc::new(RefCell::new(None));
    let params: Rc<RefCell<PreviewParams>> = Rc::new(RefCell::new(PreviewParams::default()));

    // Load + decode the image asynchronously so the page appears immediately.
    let path_for_load = file_path.to_string();
    glib::spawn_future_local(clone!(@weak picture, @strong base, @strong params => async move {
        let data = gio::spawn_blocking(move || std::fs::read(&path_for_load).ok())
            .await
            .ok()
            .flatten();
        if let Some(data) = data {
            let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
            let _ = loader.write(&data);
            let _ = loader.close();
            if let Some(pb) = loader.pixbuf() {
                *base.borrow_mut() = Some(BaseImage {
                    bytes: pb.read_pixel_bytes().to_vec(),
                    width: pb.width(),
                    height: pb.height(),
                    rowstride: pb.rowstride() as usize,
                    nch: pb.n_channels() as usize,
                });
                render_preview(&picture, &base, &params.borrow());
            }
        }
    }));

    // ── IOP module list (right panel) — hosts the live param widgets ───────
    let modules_panel = build_modules_panel(&picture, &base, &params);

    // ── Split view: image | modules ────────────────────────────────────────
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    content.append(&picture);
    content.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    content.append(&modules_panel);

    // ── Header bar with Export button ─────────────────────────────────────
    let header = adw::HeaderBar::new();
    let title_widget = adw::WindowTitle::new(&filename, "Darkroom");
    header.set_title_widget(Some(&title_widget));

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

    adw::NavigationPage::builder()
        .title(&filename)
        .child(&toolbar_view)
        .build()
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
    lbl.set_width_chars(7);

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
fn build_modules_panel(
    picture: &gtk4::Picture,
    base: &Rc<RefCell<Option<BaseImage>>>,
    params: &Rc<RefCell<PreviewParams>>,
) -> gtk4::Widget {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .build();

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
                "Exposure" => pg.add(&exposure_module_row(picture, base, params)),
                "Velvia" => pg.add(&velvia_module_row(picture, base, params)),
                _ => pg.add(&inert_module_row(mi.label, mi.default_on)),
            }
        }
        panel.append(&pg);
    }

    // Scrollable so the (long) module list never blows out the window height.
    gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vexpand(true)
        .width_request(320)
        .child(&panel)
        .build()
        .upcast()
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
const LIVE_MODULE_LABELS: &[&str] = &["Exposure", "Velvia"];

// Borrow invariant for the closures below: GTK callbacks run on the main
// thread and never re-enter while a `params` borrow is held — each closure
// takes a short-lived `borrow_mut()` (dropped at the statement end) before the
// `render_preview(..., &params.borrow())` read, so the two never overlap.

/// Exposure module: an expander whose enable switch gates `exposure_on` and
/// whose EV / black-point sliders drive the exposure stage of the preview.
fn exposure_module_row(
    picture: &gtk4::Picture,
    base: &Rc<RefCell<Option<BaseImage>>>,
    params: &Rc<RefCell<PreviewParams>>,
) -> adw::ExpanderRow {
    let p0 = *params.borrow();
    let expander = adw::ExpanderRow::builder()
        .title("Exposure")
        .subtitle("EV + black point")
        .show_enable_switch(true)
        .enable_expansion(p0.exposure_on)
        .build();
    expander.connect_enable_expansion_notify(clone!(@weak picture, @strong base, @strong params => move |e| {
        params.borrow_mut().exposure_on = e.enables_expansion();
        render_preview(&picture, &base, &params.borrow());
    }));

    // EV: scale = 2^ev.
    let ev = labeled_slider("EV", -3.0, 3.0, 0.01, p0.ev as f64);
    ev.scale.connect_value_changed(clone!(@weak picture, @strong base, @strong params => move |s| {
        params.borrow_mut().ev = s.value() as f32;
        render_preview(&picture, &base, &params.borrow());
    }));
    expander.add_row(&ev.row);

    // Black point: lifted before scaling (out = (in - black) * scale).
    let black = labeled_slider("Black", 0.0, 0.2, 0.001, p0.black as f64);
    black.scale.connect_value_changed(clone!(@weak picture, @strong base, @strong params => move |s| {
        params.borrow_mut().black = s.value() as f32;
        render_preview(&picture, &base, &params.borrow());
    }));
    expander.add_row(&black.row);

    expander
}

/// Velvia module: an expander whose enable switch gates `velvia_on` and whose
/// strength slider drives the velvia stage of the preview.
fn velvia_module_row(
    picture: &gtk4::Picture,
    base: &Rc<RefCell<Option<BaseImage>>>,
    params: &Rc<RefCell<PreviewParams>>,
) -> adw::ExpanderRow {
    let p0 = *params.borrow();
    let expander = adw::ExpanderRow::builder()
        .title("Velvia")
        .subtitle("saturation boost")
        .show_enable_switch(true)
        .enable_expansion(p0.velvia_on)
        .build();
    expander.connect_enable_expansion_notify(clone!(@weak picture, @strong base, @strong params => move |e| {
        params.borrow_mut().velvia_on = e.enables_expansion();
        render_preview(&picture, &base, &params.borrow());
    }));

    // Strength (0..100); the C slider's scale, divided by 100 for the core IOP.
    let strength = labeled_slider("Strength", 0.0, 100.0, 1.0, p0.velvia_strength as f64);
    strength.scale.connect_value_changed(clone!(@weak picture, @strong base, @strong params => move |s| {
        params.borrow_mut().velvia_strength = s.value() as f32;
        render_preview(&picture, &base, &params.borrow());
    }));
    expander.add_row(&strength.row);

    expander
}

#[cfg(test)]
mod tests {
    use super::*;

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
