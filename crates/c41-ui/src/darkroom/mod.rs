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
use crate::history::{describe_change, HistoryStack};
use crate::preview::{Histogram, PreviewParams};
use c41_core::geometry::{Crop, Geometry};
use c41_core::rawimage::DemosaicMethod;
use crate::snapshots::{SnapshotStore, SNAPSHOT_CAP};

/// Shared snapshot store over the cached-render payload (the frozen pixels).
type SnapStore = Rc<RefCell<SnapshotStore<CachedRender>>>;

/// Placeholder shown in the colour-picker readout before/after a sample.
const PICKER_PROMPT: &str = "Pick: click the image to sample a pixel";

/// Decoded preview source kept for live re-processing — either an 8-bit sRGB
/// image (the JPEG / gdk-pixbuf path) or a **linear scene-referred RGBA f32**
/// image (the raw path). The raw variant skips the 8-bit round-trip so a
/// tone-map stage sees the unclipped highlights.
enum BaseImage {
    Srgb8 { bytes: Vec<u8>, width: i32, height: i32, rowstride: usize, nch: usize },
    Linear { width: usize, height: usize, pixels: Vec<f32> },
}

/// An 8-bit image produced by running the pipeline over a [`BaseImage`], ready
/// for texture upload / histogram / colour-picker (all of which want 8-bit).
struct Rendered {
    bytes: Vec<u8>,
    width: i32,
    height: i32,
    rowstride: usize,
    nch: usize,
}

/// The last displayed render, cached for the colour picker so a click samples
/// it instead of re-running the whole pipeline. `bytes` is the *same* refcounted
/// buffer uploaded to the texture (a `glib::Bytes` clone is a cheap refcount
/// bump — no pixel copy).
#[derive(Clone)]
struct CachedRender {
    bytes: glib::Bytes,
    width: i32,
    height: i32,
    rowstride: usize,
    nch: usize,
}

impl BaseImage {
    /// Run the live pipeline and return the 8-bit sRGB image to display.
    fn render(&self, params: &PreviewParams) -> Rendered {
        match self {
            BaseImage::Srgb8 { bytes, width, height, rowstride, nch } => {
                let (w, h) = (*width as usize, *height as usize);
                Rendered {
                    bytes: crate::preview::apply_pipeline(bytes, w, h, *rowstride, *nch, params),
                    width: *width,
                    height: *height,
                    rowstride: *rowstride,
                    nch: *nch,
                }
            }
            BaseImage::Linear { width, height, pixels } => Rendered {
                bytes: crate::preview::render_linear_to_srgb8(pixels, *width, *height, params),
                width: *width as i32,
                height: *height as i32,
                rowstride: width * 3,
                nch: 3,
            },
        }
    }
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
    /// The decoded raw preview BEFORE geometry (linear RGBA `f32` + dims), kept
    /// so a straighten/crop change re-applies [`Geometry`] to it without a full
    /// re-decode. `None` for the JPEG path (no geometry there yet).
    pristine: Rc<RefCell<Option<(usize, usize, Vec<f32>)>>>,
    /// Current per-image geometry (straighten + crop), applied to `pristine` to
    /// produce the displayed `base`.
    geometry: Rc<std::cell::Cell<Geometry>>,
    /// While set (crop-edit mode), the displayed `base` is rotated but UNcropped
    /// so the crop overlay can be dragged over the full frame; cleared, `base`
    /// shows the actual cropped result.
    crop_editing: Rc<std::cell::Cell<bool>>,
    /// Monotonic decode generation. Each [`spawn_decode`] bumps it and captures
    /// the value; a completed decode only paints if it is still the newest, so a
    /// slow earlier demosaic can't overwrite a newer one (stale-paint guard).
    decode_gen: Rc<std::cell::Cell<u64>>,
    params: Rc<RefCell<PreviewParams>>,
    hist: Rc<RefCell<Histogram>>,
    /// While set, the preview shows the unprocessed image (before/after toggle).
    bypass: Rc<std::cell::Cell<bool>>,
    /// Debounced DB autosave of the current params (None when there's no db).
    autosave: Option<Rc<AutoSave>>,
    /// Last displayed render, for the colour picker (see [`CachedRender`]).
    last_render: Rc<RefCell<Option<CachedRender>>>,
    /// Navigable edit history (undo/redo via the history panel).
    history: Rc<RefCell<HistoryStack>>,
    /// Debounced recorder that snapshots a settled edit into `history`.
    history_rec: Rc<HistoryRecorder>,
    /// The Bayer demosaic-method selector section (header + dropdown), hidden
    /// once a decode reveals the sensor is X-Trans (where the method is a no-op).
    /// Empty until the raw-only selector is built below.
    demosaic_row: glib::WeakRef<gtk4::Box>,
}

/// Debounced recorder that appends one [`HistoryStack`] entry per *settled* edit
/// (so a slider drag coalesces into a single history item, like [`AutoSave`]),
/// labelling it by which module changed, then refreshes the history list. The
/// dedup in [`HistoryStack::record`] means re-renders that don't change the
/// params (before/after toggle, a jump landing on its own entry) add nothing.
struct HistoryRecorder {
    params: Rc<RefCell<PreviewParams>>,
    history: Rc<RefCell<HistoryStack>>,
    list: glib::WeakRef<gtk4::ListBox>,
    pending: RefCell<Option<glib::SourceId>>,
}

impl HistoryRecorder {
    /// (Re)arm the debounce; the last edit within the window is the one recorded.
    fn arm(self: &Rc<Self>) {
        if let Some(id) = self.pending.borrow_mut().take() {
            id.remove();
        }
        let this = self.clone();
        let id = glib::timeout_add_local_once(std::time::Duration::from_millis(700), move || {
            *this.pending.borrow_mut() = None;
            // Reads the *live* params (`ctx.params`), never `effective_params`:
            // the before/after toggle paints the bypassed image but must not
            // record a history entry. Because the live params are unchanged by a
            // toggle, `record`'s dedup makes the toggle's render a no-op. Do NOT
            // switch this to `effective_params` — it would log a spurious entry
            // every time the user peeks at the original (and no test guards it).
            let params = *this.params.borrow();
            // Borrow the stack only briefly to read the current state, then to
            // record — never across the widget refresh.
            let label = {
                let h = this.history.borrow();
                describe_change(&h.current(), &params)
            };
            let changed = this.history.borrow_mut().record(label, params);
            if changed {
                if let Some(list) = this.list.upgrade() {
                    refresh_history_list(&list, &this.history.borrow());
                }
            }
        });
        *self.pending.borrow_mut() = Some(id);
    }
}

/// Rebuild the history panel's rows from the stack (oldest → newest) and select
/// the current cursor row. Programmatic `select_row` fires `row-selected`, not
/// `row-activated`, so it doesn't re-enter the click-to-jump handler.
fn refresh_history_list(list: &gtk4::ListBox, history: &HistoryStack) {
    while let Some(child) = list.first_child() {
        list.remove(&child);
    }
    for entry in history.entries() {
        let row = gtk4::ListBoxRow::new();
        let lbl = gtk4::Label::builder()
            .label(&entry.label)
            .halign(gtk4::Align::Start)
            .margin_start(10)
            .margin_end(10)
            .margin_top(3)
            .margin_bottom(3)
            .build();
        row.set_child(Some(&lbl));
        list.append(&row);
    }
    if let Some(row) = list.row_at_index(history.cursor() as i32) {
        list.select_row(Some(&row));
    }
}

/// Restore a history entry's params into the live state: write `p`, rebuild the
/// module sliders (the Reset path, so widgets follow), re-render, and select the
/// now-current row in the history list. Shared by the click-to-jump handler and
/// the Undo/Redo buttons — the caller has already moved the history cursor (via
/// `jump_to`/`undo`/`redo`) so the cursor is the entry being applied, and the
/// render's debounced record dedups (current == `p`, no spurious branch).
///
/// Restoring a snapshot also exits the before/after peek: otherwise `bypass`
/// would stay set and `render_preview` would paint the *bypassed* image (and a
/// mismatched histogram) while the params underneath had silently changed — the
/// viewport lying about the edit state. We clear `bypass` and re-sync the toggle
/// button, mirroring Reset, so the helper is the single source of truth.
fn apply_history_params(
    ctx: &PreviewCtx,
    panel: &glib::WeakRef<gtk4::Box>,
    list: &glib::WeakRef<gtk4::ListBox>,
    before_after: &glib::WeakRef<gtk4::ToggleButton>,
    p: PreviewParams,
) {
    *ctx.params.borrow_mut() = p;
    ctx.bypass.set(false); // restoring a snapshot exits the before/after peek
    if let Some(ba) = before_after.upgrade() {
        ba.set_active(false); // keep the toggle visual in sync with bypass
    }
    if let Some(panel) = panel.upgrade() {
        while let Some(child) = panel.first_child() {
            panel.remove(&child);
        }
        populate_modules(&panel, ctx);
    }
    render_preview(ctx);
    // Keep the list highlight on the cursor (Undo/Redo move it without a click).
    if let Some(list) = list.upgrade() {
        let cursor = ctx.history.borrow().cursor();
        if let Some(row) = list.row_at_index(cursor as i32) {
            list.select_row(Some(&row));
        }
    }
}

/// Rebuild the snapshots panel's rows from the store. Each row is a label plus a
/// remove button (per-row index closures, rebuilt here so indices stay valid);
/// clicking a row body activates the wipe overlay (one handler set in
/// [`darkroom_page`]). Removing a snapshot also clears the wipe overlay, since the
/// snapshot being shown may be the one removed.
fn refresh_snapshot_list(
    list: &gtk4::ListBox,
    store: &SnapStore,
    wipe: &WipeCompare,
) {
    // Tearing down old rows drops their remove-button closures (which hold an
    // `Rc` clone of the store, not a borrow) — so teardown never re-enters a
    // store borrow. Snapshot the labels into an owned Vec and drop the store
    // borrow before building rows, so the per-row closures can borrow freely.
    while let Some(child) = list.first_child() {
        list.remove(&child);
    }
    let labels: Vec<String> = store.borrow().entries().iter().map(|e| e.label.clone()).collect();
    if labels.is_empty() {
        let row = gtk4::ListBoxRow::new();
        row.set_selectable(false);
        row.set_activatable(false);
        let lbl = gtk4::Label::builder()
            .label("(no snapshots)")
            .halign(gtk4::Align::Start)
            .margin_start(10)
            .margin_end(10)
            .margin_top(3)
            .margin_bottom(3)
            .build();
        lbl.add_css_class("dim-label");
        row.set_child(Some(&lbl));
        list.append(&row);
        return;
    }
    for (i, label) in labels.iter().enumerate() {
        let row = gtk4::ListBoxRow::new();
        let hbox = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(6)
            .margin_start(10)
            .margin_end(6)
            .margin_top(2)
            .margin_bottom(2)
            .build();
        let name = gtk4::Label::builder()
            .label(label)
            .halign(gtk4::Align::Start)
            .hexpand(true)
            .ellipsize(gtk4::pango::EllipsizeMode::Middle)
            .build();
        let remove = gtk4::Button::builder()
            .icon_name("window-close-symbolic")
            .has_frame(false)
            .valign(gtk4::Align::Center)
            .tooltip_text("Remove snapshot")
            .build();
        let store_cl = store.clone();
        let list_cl = list.downgrade();
        let wipe_cl = wipe.clone();
        remove.connect_clicked(move |_| {
            // Drop the `borrow_mut` at the `;` *before* `refresh_snapshot_list`
            // re-borrows the store — that ordering is what keeps this panic-free.
            store_cl.borrow_mut().remove(i);
            wipe_cl.clear(); // stop showing a possibly-removed snapshot
            if let Some(l) = list_cl.upgrade() {
                l.unselect_all();
                refresh_snapshot_list(&l, &store_cl, &wipe_cl);
            }
        });
        hbox.append(&name);
        hbox.append(&remove);
        row.set_child(Some(&hbox));
        list.append(&row);
    }
}

/// Debounced writer that persists the current params **and the edit-history
/// stack** a short time after the last edit (so slider drags don't write
/// per-tick), with an explicit flush on close. Persisting the stack here covers
/// every state change — new entries (recorder fires at 700ms < this 800ms) and
/// cursor moves (undo/redo/jump re-render → re-arm) — without a second timer.
struct AutoSave {
    db_path: String,
    file_path: String,
    params: Rc<RefCell<PreviewParams>>,
    history: Rc<RefCell<HistoryStack>>,
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
            this.persist();
        });
        *self.pending.borrow_mut() = Some(id);
    }

    /// Cancel any pending timer and save immediately (final flush on page close).
    /// Force-records the current params first: a flush can pre-empt the 700ms
    /// history recorder, so the in-flight edit might not be in the stack yet
    /// (`record` dedups if it already is).
    fn flush(&self) {
        if let Some(id) = self.pending.borrow_mut().take() {
            id.remove();
        }
        // Capture an in-flight edit the 700ms recorder hasn't recorded yet (a
        // close can pre-empt it) — but ONLY when the live params differ from the
        // cursor entry. Skipping `record` entirely on a clean close (incl. right
        // after an undo/redo/jump, where params already == current) means we
        // never exercise the redo-tail-truncation path on a mid-stack cursor.
        let p = *self.params.borrow();
        let cur = self.history.borrow().current();
        if cur != p {
            self.history
                .borrow_mut()
                .record(describe_change(&cur, &p), p);
        }
        self.persist();
    }

    /// Write the params row and the history-stack row (best-effort).
    fn persist(&self) {
        crate::persist::save_params(&self.db_path, &self.file_path, &self.params.borrow());
        crate::persist::save_history(&self.db_path, &self.file_path, &self.history.borrow());
    }
}

/// Build a `gdk::MemoryTexture` from a cached render (shared by the live
/// preview and the snapshot comparison view).
fn cached_render_texture(c: &CachedRender) -> gtk4::gdk::MemoryTexture {
    // The render only ever emits 3- or 4-channel buffers; anything else would
    // mis-map the format (and could over-read in MemoryTexture::new).
    debug_assert!(c.nch == 3 || c.nch == 4, "unexpected channel count {}", c.nch);
    let fmt = if c.nch == 4 {
        gtk4::gdk::MemoryFormat::R8g8b8a8
    } else {
        gtk4::gdk::MemoryFormat::R8g8b8
    };
    gtk4::gdk::MemoryTexture::new(c.width, c.height, fmt, &c.bytes, c.rowstride)
}

/// A snapshot frozen for the **scale-locked wipe** overlay: the captured pixels
/// as a cairo surface plus their dimensions, so the wipe layer can paint them
/// into the *same* `Contain` rectangle the live image occupies — features line up
/// across the divider (unlike the old independent side-by-side letterboxing).
struct CompareState {
    surface: gtk4::cairo::ImageSurface,
    img_w: usize,
    img_h: usize,
}

/// Drives the snapshot wipe overlay: a transparent `DrawingArea` layered over the
/// live image that paints the selected snapshot to the left of a draggable
/// divider. Idle (no snapshot selected) it holds no state and is click-through
/// (`can_target(false)`), so the colour picker keeps working on the live image.
#[derive(Clone)]
struct WipeCompare {
    area: glib::WeakRef<gtk4::DrawingArea>,
    state: Rc<RefCell<Option<CompareState>>>,
    /// Divider position as a fraction [0,1] of the displayed image width.
    frac: Rc<std::cell::Cell<f64>>,
}

impl WipeCompare {
    /// Begin comparing against `c`: build the cairo surface, reset the divider to
    /// centre, take pointer input, and repaint. No-op if the surface can't be
    /// built (degenerate size).
    fn show(&self, c: &CachedRender) {
        let Some(surface) = cached_render_surface(c) else { return };
        *self.state.borrow_mut() = Some(CompareState {
            surface,
            img_w: c.width as usize,
            img_h: c.height as usize,
        });
        self.frac.set(0.5);
        if let Some(area) = self.area.upgrade() {
            area.set_can_target(true);
            area.queue_draw();
        }
    }

    /// Stop comparing: drop the snapshot, become click-through again, and repaint
    /// (the empty overlay reveals the live image in full).
    fn clear(&self) {
        self.state.borrow_mut().take();
        if let Some(area) = self.area.upgrade() {
            area.set_can_target(false);
            area.queue_draw();
        }
    }
}

/// Move the wipe divider to widget-space `x`, mapped to a fraction of the
/// displayed image width and repainting. No-op while not comparing.
fn set_wipe_from_x(wipe: &WipeCompare, x: f64) {
    let Some(area) = wipe.area.upgrade() else { return };
    // Scope the `state` borrow to exactly the geometry read, so nothing on this
    // path can ever sit under a live borrow (the draw func borrows `state` too).
    let rect = {
        let guard = wipe.state.borrow();
        let Some(state) = guard.as_ref() else { return };
        crate::preview::contain_rect(area.width() as f64, area.height() as f64, state.img_w, state.img_h)
    };
    if let Some(rect) = rect {
        wipe.frac.set(crate::preview::wipe_fraction(&rect, x));
        area.queue_draw();
    }
}

/// Convert a cached render to a cairo `Rgb24` surface for the wipe overlay. Thin
/// cairo wrapper over the pure [`crate::preview::pack_rgb24`] (the byte-swap and
/// greyscale logic is tested there). Runs once per snapshot selection (cached in
/// [`CompareState`]), not per draw. `None` on degenerate size or a cairo error.
fn cached_render_surface(c: &CachedRender) -> Option<gtk4::cairo::ImageSurface> {
    use gtk4::cairo::Format;
    if c.width <= 0 || c.height <= 0 {
        return None;
    }
    let stride = Format::Rgb24.stride_for_width(c.width as u32).ok()? as usize;
    let data = crate::preview::pack_rgb24(
        c.bytes.as_ref(),
        c.width as usize,
        c.height as usize,
        c.rowstride,
        c.nch,
        stride,
    );
    // `create_for_data` takes ownership of `data` and keeps a pointer into it for
    // the surface's lifetime — never refactor this to a borrow or a buffer mutated
    // afterwards.
    gtk4::cairo::ImageSurface::create_for_data(data, Format::Rgb24, c.width, c.height, stride as i32)
        .ok()
}

/// Paint the snapshot wipe overlay: the snapshot scaled into the live image's
/// `Contain` rect, clipped to the left of the divider, with a 1px divider line.
/// Untouched regions stay transparent, so the live image below shows through on
/// the right.
fn draw_wipe(cr: &gtk4::cairo::Context, w: i32, h: i32, state: &CompareState, frac: f64) {
    let Some(rect) = crate::preview::contain_rect(w as f64, h as f64, state.img_w, state.img_h)
    else {
        return;
    };
    let wipe_x = rect.off_x + rect.disp_w * frac;

    // Snapshot side: clip to [off_x, wipe_x] and paint the surface scaled into the
    // contain rect (same geometry as the live Picture, so the two align).
    let _ = cr.save();
    cr.rectangle(rect.off_x, rect.off_y, (wipe_x - rect.off_x).max(0.0), rect.disp_h);
    cr.clip();
    cr.translate(rect.off_x, rect.off_y);
    cr.scale(rect.disp_w / state.img_w as f64, rect.disp_h / state.img_h as f64);
    if cr.set_source_surface(&state.surface, 0.0, 0.0).is_ok() {
        let _ = cr.paint();
    } else {
        eprintln!("darkroom wipe: set_source_surface failed; snapshot pane left empty");
    }
    let _ = cr.restore();

    // Divider line down the displayed image height.
    cr.set_source_rgb(1.0, 1.0, 1.0);
    cr.set_line_width(1.0);
    cr.move_to(wipe_x, rect.off_y);
    cr.line_to(wipe_x, rect.off_y + rect.disp_h);
    let _ = cr.stroke();
}

/// Interactive crop-rectangle overlay (m4-48b). Layered over the `Picture` like
/// [`WipeCompare`]; idle it is click-through so the colour picker works, and it
/// is only interactive while crop-edit mode (`editing`) is on. Draws the crop
/// rect + 8 handles and dims outside; a `GestureDrag` grabs a handle (via
/// [`crate::crop_overlay::hit_test`]) and resizes/moves the crop in
/// [`PreviewCtx::geometry`]. While editing, the displayed image is rotated but
/// UNcropped (see [`apply_geometry_to_base`]) so the rect can be dragged over the
/// full frame; leaving crop mode shows the actual cropped result.
#[derive(Clone)]
struct CropOverlay {
    area: glib::WeakRef<gtk4::DrawingArea>,
    geometry: Rc<std::cell::Cell<Geometry>>,
    editing: Rc<std::cell::Cell<bool>>,
    /// Shared with the ctx: the current (rotated) base, for its display dims.
    base: Rc<RefCell<Option<BaseImage>>>,
    /// In-flight drag: the grabbed handle + the crop and pointer at drag start.
    drag: Rc<RefCell<Option<CropDrag>>>,
    /// Aspect-ratio lock applied to resize drags (`Free` = unconstrained).
    aspect: Rc<std::cell::Cell<crate::crop_overlay::AspectRatio>>,
}

/// The state captured when a crop drag begins, so each update recomputes from the
/// original crop (not the last frame) — no drift accumulation.
struct CropDrag {
    handle: crate::crop_overlay::CropHandle,
    start_crop: c41_core::geometry::Crop,
    start_fx: f32,
    start_fy: f32,
}

/// Display dimensions of the current base (`None` if not yet decoded).
fn base_dims(base: &Option<BaseImage>) -> Option<(usize, usize)> {
    match base.as_ref()? {
        BaseImage::Srgb8 { width, height, .. } => Some((*width as usize, *height as usize)),
        BaseImage::Linear { width, height, .. } => Some((*width, *height)),
    }
}

/// Paint the crop overlay: dim outside the crop rect, outline it, draw
/// rule-of-thirds guides and 8 grab handles. Coordinates come from
/// [`crate::preview::contain_rect`] over the current (rotated) base dims so the
/// rect lines up with the displayed image.
fn draw_crop(cr: &gtk4::cairo::Context, w: i32, h: i32, base_w: usize, base_h: usize, crop: Crop) {
    let Some(rect) = crate::preview::contain_rect(w as f64, h as f64, base_w, base_h) else {
        return;
    };
    let crop = crop.normalized();
    let x0 = rect.off_x + rect.disp_w * crop.left as f64;
    let x1 = rect.off_x + rect.disp_w * crop.right as f64;
    let y0 = rect.off_y + rect.disp_h * crop.top as f64;
    let y1 = rect.off_y + rect.disp_h * crop.bottom as f64;
    let (ix, iy, iw, ih) = (rect.off_x, rect.off_y, rect.disp_w, rect.disp_h);

    // Dim the four bands outside the crop rect (within the displayed image).
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.5);
    cr.rectangle(ix, iy, iw, y0 - iy); // top
    cr.rectangle(ix, y1, iw, iy + ih - y1); // bottom
    cr.rectangle(ix, y0, x0 - ix, y1 - y0); // left
    cr.rectangle(x1, y0, ix + iw - x1, y1 - y0); // right
    let _ = cr.fill();

    // Rule-of-thirds guides (faint).
    cr.set_source_rgba(1.0, 1.0, 1.0, 0.35);
    cr.set_line_width(1.0);
    for i in 1..3 {
        let gx = x0 + (x1 - x0) * i as f64 / 3.0;
        cr.move_to(gx, y0);
        cr.line_to(gx, y1);
        let gy = y0 + (y1 - y0) * i as f64 / 3.0;
        cr.move_to(x0, gy);
        cr.line_to(x1, gy);
    }
    let _ = cr.stroke();

    // Rect outline + 8 handles.
    cr.set_source_rgb(1.0, 1.0, 1.0);
    cr.rectangle(x0, y0, x1 - x0, y1 - y0);
    let _ = cr.stroke();
    let hs = 4.0;
    let (xm, ym) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    for &(hx, hy) in &[
        (x0, y0), (xm, y0), (x1, y0), (x1, ym), (x1, y1), (xm, y1), (x0, y1), (x0, ym),
    ] {
        cr.rectangle(hx - hs, hy - hs, hs * 2.0, hs * 2.0);
    }
    let _ = cr.fill();
}

/// The displayed-image rect for the crop overlay's current base, if editable.
/// Scopes the `base` borrow to exactly the dims read (the draw func borrows
/// `base` too), mirroring [`set_wipe_from_x`].
fn crop_contain_rect(ov: &CropOverlay) -> Option<(gtk4::DrawingArea, crate::preview::ContainRect)> {
    let area = ov.area.upgrade()?;
    let (bw, bh) = base_dims(&ov.base.borrow())?;
    let rect = crate::preview::contain_rect(area.width() as f64, area.height() as f64, bw, bh)?;
    Some((area, rect))
}

/// Grab a crop handle at widget point `(x, y)` — records the drag state so
/// updates recompute from the original crop.
fn crop_drag_begin(ov: &CropOverlay, x: f64, y: f64) {
    if !ov.editing.get() {
        return;
    }
    let Some((_, rect)) = crop_contain_rect(ov) else {
        return;
    };
    let (fx, fy) = crate::crop_overlay::widget_to_fraction(&rect, x, y);
    let crop = ov.geometry.get().crop.normalized();
    // ~12 px grab tolerance expressed as a fraction of the displayed width.
    let tol = if rect.disp_w > 0.0 { (12.0 / rect.disp_w) as f32 } else { 0.05 };
    let handle = crate::crop_overlay::hit_test(crop, fx, fy, tol);
    *ov.drag.borrow_mut() = Some(CropDrag { handle, start_crop: crop, start_fx: fx, start_fy: fy });
}

/// Update the crop from the grabbed handle and the current widget point
/// `(x, y)`, writing the new crop into `geometry` and repainting the overlay.
fn crop_drag_update(ov: &CropOverlay, x: f64, y: f64) {
    if !ov.editing.get() {
        return;
    }
    let drag = {
        let d = ov.drag.borrow();
        match d.as_ref() {
            Some(d) => (d.handle, d.start_crop, d.start_fx, d.start_fy),
            None => return,
        }
    };
    let (handle, start_crop, sfx, sfy) = drag;
    if handle == crate::crop_overlay::CropHandle::None {
        return;
    }
    let Some((area, rect)) = crop_contain_rect(ov) else {
        return;
    };
    let (fx, fy) = crate::crop_overlay::widget_to_fraction(&rect, x, y);
    let new_crop = if handle == crate::crop_overlay::CropHandle::Inside {
        crate::crop_overlay::translate(start_crop, fx - sfx, fy - sfy)
    } else {
        let resized = crate::crop_overlay::resize_to(start_crop, handle, fx, fy);
        // Lock the pixel aspect on resize (no-op for Free); needs the display
        // dims to convert the fraction crop to a pixel ratio.
        match base_dims(&ov.base.borrow()) {
            Some((bw, bh)) => {
                crate::crop_overlay::apply_aspect(resized, handle, ov.aspect.get(), bw, bh)
            }
            None => resized,
        }
    };
    let mut g = ov.geometry.get();
    g.crop = new_crop;
    ov.geometry.set(g);
    area.queue_draw();
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
        let r = b.render(&params);

        *ctx.hist.borrow_mut() = crate::preview::compute_histogram(
            &r.bytes, r.width as usize, r.height as usize, r.rowstride, r.nch,
        );
        if let Some(area) = ctx.hist_area.upgrade() {
            area.queue_draw();
        }

        let cr = CachedRender {
            bytes: glib::Bytes::from_owned(r.bytes),
            width: r.width,
            height: r.height,
            rowstride: r.rowstride,
            nch: r.nch,
        };
        picture.set_paintable(Some(&cached_render_texture(&cr)));

        // Cache the displayed pixels (same refcounted buffer, no copy) so the
        // colour picker samples them instead of re-rendering, and a "Take
        // snapshot" freezes exactly what's on screen.
        // INVARIANT: `render_preview` is the *only* writer of both
        // `picture.set_paintable` and `last_render`; keep them paired here so the
        // picker can never sample pixels that differ from what's on screen.
        *ctx.last_render.borrow_mut() = Some(cr);

        // The displayed pixels just changed; any prior pick is now stale.
        if let Some(label) = ctx.picker.upgrade() {
            label.set_text(PICKER_PROMPT);
        }

        // Debounced persistence of the (live, not bypassed) params.
        if let Some(autosave) = &ctx.autosave {
            autosave.arm();
        }
        // Debounced history snapshot of the same settled edit.
        ctx.history_rec.arm();
    }
}

/// Bayer demosaic selector option labels, ordered to match
/// [`DemosaicMethod::as_u8`] (0=RCD, 1=VNG, 2=PPG) so a `DropDown`'s selected
/// index round-trips through [`DemosaicMethod::from_u8`] with no extra mapping.
fn demosaic_method_labels() -> [&'static str; 3] {
    ["RCD — best quality", "VNG", "PPG — fastest"]
}

/// Straighten-slider degrees → [`Geometry::angle`] radians.
fn straighten_deg_to_rad(deg: f64) -> f32 {
    (deg * std::f64::consts::PI / 180.0) as f32
}

/// [`Geometry::angle`] radians → straighten-slider degrees (for seeding).
fn straighten_rad_to_deg(rad: f32) -> f64 {
    rad as f64 * 180.0 / std::f64::consts::PI
}

/// Apply the current [`PreviewCtx::geometry`] to the cached un-geometried raw
/// buffer ([`PreviewCtx::pristine`]) to produce the displayed `base`, then
/// re-render. No-op when there is no pristine buffer (the JPEG path, or before
/// the first decode). Cheap enough to re-run on every geometry change (a
/// downscaled-buffer resample), unlike a full re-decode.
fn apply_geometry_to_base(ctx: &PreviewCtx) {
    let geom = ctx.geometry.get();
    let editing = ctx.crop_editing.get();
    let base = {
        let pristine = ctx.pristine.borrow();
        let Some((w, h, pixels)) = pristine.as_ref() else {
            return;
        };
        // Crop-edit mode shows the rotated-but-uncropped frame (the overlay draws
        // the crop rect on top); otherwise the actual cropped result.
        let (gw, gh, gpixels) = if editing {
            c41_core::geometry::apply_rotate(pixels, *w, *h, geom.angle)
        } else {
            geom.apply(pixels, *w, *h)
        };
        BaseImage::Linear { width: gw, height: gh, pixels: gpixels }
    };
    *ctx.base.borrow_mut() = Some(base);
    render_preview(ctx);
}

/// Decode + downscale the image off the main thread, then display it. Camera
/// raws go through the Rust decoder with the chosen Bayer [`DemosaicMethod`]
/// (`method` is ignored for non-raw files and for X-Trans sensors), are stored
/// as the un-geometried [`PreviewCtx::pristine`] buffer, and get the current
/// [`Geometry`] applied to produce `base`; everything else goes through
/// gdk-pixbuf straight to `base` (no geometry). Re-run to change the demosaic
/// method — it re-decodes the full raw, unlike the geometry/pipeline changes
/// which reuse the already-downscaled buffer.
fn spawn_decode(ctx: &PreviewCtx, path: String, method: DemosaicMethod) {
    // Claim the newest generation; a stale (superseded) decode discards below.
    let generation = ctx.decode_gen.get().wrapping_add(1);
    ctx.decode_gen.set(generation);
    glib::spawn_future_local(clone!(@strong ctx => async move {
        if crate::raw_preview::is_raw_path(&path) {
            // Decode + demosaic off the main thread (it's heavy); downscale to a
            // responsive preview size. Keep the un-geometried buffer.
            let p = path.clone();
            let decoded = gio::spawn_blocking(move || {
                crate::raw_preview::decode_raw_preview_with(
                    &p,
                    crate::raw_preview::PREVIEW_MAX_DIM,
                    method,
                )
                .map(|rp| (rp.width, rp.height, rp.pixels, rp.is_xtrans))
            })
            .await
            .ok()
            .flatten();
            // A newer decode was requested while this ran (rapid method
            // switching) — drop this result so it can't paint over the newer one.
            if ctx.decode_gen.get() != generation {
                return;
            }
            match decoded {
                Some((w, h, px, is_xtrans)) => {
                    // The Bayer demosaic selector is a no-op for X-Trans
                    // (Markesteijn is fixed) — hide the section for those files.
                    // Runs on every decode, including Bayer method-change
                    // re-decodes; the redundant re-show there is intentional.
                    if let Some(row) = ctx.demosaic_row.upgrade() {
                        row.set_visible(!is_xtrans);
                    }
                    *ctx.pristine.borrow_mut() = Some((w, h, px));
                    apply_geometry_to_base(&ctx); // sets base + renders
                }
                // Don't make a failed decode an invisible blank — log it.
                // `base`/`pristine` and the display stay on the last successful
                // result; the selector shows the attempted (failed) method.
                None => eprintln!("darkroom preview: could not decode {path}"),
            }
            return;
        }

        // Non-raw (JPEG etc.): gdk-pixbuf straight to an 8-bit base, no geometry.
        let p = path.clone();
        let data = gio::spawn_blocking(move || std::fs::read(&p).ok())
            .await
            .ok()
            .flatten();
        if ctx.decode_gen.get() != generation {
            return;
        }
        let base = data.and_then(|data| {
            let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
            let _ = loader.write(&data);
            let _ = loader.close();
            loader.pixbuf().map(|pb| BaseImage::Srgb8 {
                bytes: pb.read_pixel_bytes().to_vec(),
                width: pb.width(),
                height: pb.height(),
                rowstride: pb.rowstride() as usize,
                nch: pb.n_channels() as usize,
            })
        });
        match base {
            Some(base) => {
                *ctx.pristine.borrow_mut() = None; // no geometry on the JPEG path
                *ctx.base.borrow_mut() = Some(base);
                render_preview(&ctx);
            }
            None => eprintln!("darkroom preview: could not decode {path}"),
        }
    }));
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
    // Middle-grey surround, as darktable does. Functional, not decorative: the
    // colour behind the image shifts how you judge its tone (see `crate::theme`).
    picture.add_css_class("c41-darkroom-canvas");

    // Scale-locked snapshot wipe: a transparent DrawingArea layered over the live
    // image (via the Overlay below) paints the selected snapshot to the left of a
    // draggable divider, sharing the live image's Contain geometry so features
    // line up across the wipe. Idle it's click-through, so the picker still works.
    let wipe_area = gtk4::DrawingArea::builder()
        .hexpand(true)
        .vexpand(true)
        .can_target(false)
        .build();
    let wipe = WipeCompare {
        area: wipe_area.downgrade(),
        state: Rc::new(RefCell::new(None)),
        frac: Rc::new(std::cell::Cell::new(0.5)),
    };
    {
        let draw_state = wipe.state.clone();
        let draw_frac = wipe.frac.clone();
        wipe_area.set_draw_func(move |_, cr, w, h| {
            if let Some(st) = draw_state.borrow().as_ref() {
                draw_wipe(cr, w, h, st, draw_frac.get());
            }
        });
    }
    // Drag (or click) anywhere on the overlay to move the divider.
    {
        let drag = gtk4::GestureDrag::new();
        let w_begin = wipe.clone();
        drag.connect_drag_begin(move |_, x, _| set_wipe_from_x(&w_begin, x));
        let w_update = wipe.clone();
        drag.connect_drag_update(move |g, ox, _| {
            if let Some((sx, _)) = g.start_point() {
                set_wipe_from_x(&w_update, sx + ox);
            }
        });
        wipe_area.add_controller(drag);
    }

    // Crop overlay: a second DrawingArea layered over the Picture. Idle it's
    // click-through (the picker still works); interactive only in crop-edit mode.
    // Its `CropOverlay` (draw func + gesture) is wired after `ctx` exists.
    let crop_area = gtk4::DrawingArea::builder()
        .hexpand(true)
        .vexpand(true)
        .can_target(false)
        .build();
    let snapshots: SnapStore = Rc::new(RefCell::new(SnapshotStore::new(SNAPSHOT_CAP)));

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

    // Edit history + live params are seeded together from the DB so the sliders,
    // the first render, and the history panel all reflect the restored state.
    // A saved history (its cursor = the last viewed state) wins; otherwise we
    // fall back to the backward-compatible params-only row (old dbs) through
    // `initial_params` (which applies raw-only defaults when there's no edit).
    let saved_history = crate::persist::load_history(db_path, file_path);
    let initial = match &saved_history {
        Some(h) => h.current(),
        None => initial_params(
            crate::persist::load_saved(db_path, file_path),
            crate::raw_preview::is_raw_path(file_path),
        ),
    };
    let params = Rc::new(RefCell::new(initial));
    // Fallback seed label is "Original" even when `initial` came from a restored
    // params-only row (an old db) rather than the true unedited state — it's the
    // root of *this* session's history timeline. A best-effort approximation: the
    // pre-feature db never stored the original, so this is the honest starting
    // point we have.
    let history = Rc::new(RefCell::new(
        saved_history.unwrap_or_else(|| HistoryStack::new("Original", *params.borrow())),
    ));

    let autosave = (!db_path.is_empty()).then(|| {
        Rc::new(AutoSave {
            db_path: db_path.to_string(),
            file_path: file_path.to_string(),
            params: params.clone(),
            history: history.clone(),
            pending: RefCell::new(None),
        })
    });

    // History list widget the recorder rebuilds as edits settle. It lives outside
    // the modules panel (which Reset/jump clear and repopulate).
    let history_list = gtk4::ListBox::builder()
        .selection_mode(gtk4::SelectionMode::Single)
        .build();
    history_list.add_css_class("navigation-sidebar");
    let history_rec = Rc::new(HistoryRecorder {
        params: params.clone(),
        history: history.clone(),
        list: history_list.downgrade(),
        pending: RefCell::new(None),
    });

    // Restore the saved per-image geometry (identity for JPEGs / no db) BEFORE
    // the first decode, so the initial render applies it.
    let geometry0 = crate::persist::load_geometry(db_path, file_path);
    let ctx = PreviewCtx {
        picture: picture.downgrade(),
        hist_area: hist_area.downgrade(),
        picker: picker_label.downgrade(),
        base: Rc::new(RefCell::new(None)),
        pristine: Rc::new(RefCell::new(None)),
        geometry: Rc::new(std::cell::Cell::new(geometry0)),
        crop_editing: Rc::new(std::cell::Cell::new(false)),
        decode_gen: Rc::new(std::cell::Cell::new(0)),
        params,
        hist: Rc::new(RefCell::new([[0u32; 256]; 3])),
        bypass: Rc::new(std::cell::Cell::new(false)),
        autosave,
        last_render: Rc::new(RefCell::new(None)),
        history,
        history_rec,
        demosaic_row: glib::WeakRef::new(),
    };
    // Show the seed entry immediately.
    refresh_history_list(&history_list, &ctx.history.borrow());

    // The histogram paints from the shared `hist` buffer (no widget captured,
    // so no cycle with hist_area).
    let hist_for_draw = ctx.hist.clone();
    hist_area.set_draw_func(move |_, cr, w, h| {
        draw_histogram(cr, w, h, &hist_for_draw.borrow());
    });

    // Load + decode the image asynchronously so the page appears immediately.
    // The saved Bayer demosaic method (raw only) seeds the first decode; the
    // selector below re-runs `spawn_decode` when the user changes it.
    let demosaic = crate::persist::load_demosaic(db_path, file_path);
    spawn_decode(&ctx, file_path.to_string(), demosaic);

    // ── Crop overlay wiring (draw func + drag gesture) ─────────────────────
    // Aspect lock shared with the selector in the geometry panel below.
    let crop_aspect = Rc::new(std::cell::Cell::new(crate::crop_overlay::AspectRatio::Free));
    let crop_overlay = CropOverlay {
        area: crop_area.downgrade(),
        geometry: ctx.geometry.clone(),
        editing: ctx.crop_editing.clone(),
        base: ctx.base.clone(),
        drag: Rc::new(RefCell::new(None)),
        aspect: crop_aspect.clone(),
    };
    {
        let ov = crop_overlay.clone();
        crop_area.set_draw_func(move |_, cr, w, h| {
            if !ov.editing.get() {
                return;
            }
            // Scope the `base` borrow to the dims read (apply_geometry_to_base
            // borrows it mutably on other turns; never nested with this).
            // GTK4 `queue_draw` is asynchronous — this draw func fires at the next
            // frame boundary, never reentrantly during a synchronous `base` borrow.
            if let Some((bw, bh)) = base_dims(&ov.base.borrow()) {
                draw_crop(cr, w, h, bw, bh, ov.geometry.get().crop);
            }
        });
    }
    {
        let drag = gtk4::GestureDrag::new();
        let ov_begin = crop_overlay.clone();
        drag.connect_drag_begin(move |_, x, y| crop_drag_begin(&ov_begin, x, y));
        let ov_update = crop_overlay.clone();
        drag.connect_drag_update(move |g, ox, oy| {
            if let Some((sx, sy)) = g.start_point() {
                crop_drag_update(&ov_update, sx + ox, sy + oy);
            }
        });
        // Persist the crop when the drag settles (cheap; not per-frame), but only
        // if a handle was actually grabbed — a click-through with no handle
        // (CropHandle::None) leaves the crop unchanged, so skip the DB write.
        let end_geom = ctx.geometry.clone();
        let end_drag = crop_overlay.drag.clone();
        let end_db = db_path.to_string();
        let end_path = file_path.to_string();
        drag.connect_drag_end(move |_, _, _| {
            let grabbed = end_drag
                .borrow()
                .as_ref()
                .is_some_and(|d| d.handle != crate::crop_overlay::CropHandle::None);
            if grabbed {
                crate::persist::save_geometry(&end_db, &end_path, &end_geom.get());
            }
        });
        crop_area.add_controller(drag);
    }

    // ── Colour picker: click the image to read the processed pixel ─────────
    let click = gtk4::GestureClick::new();
    let pick_ctx = ctx.clone();
    let pick_pic = picture.downgrade();
    let pick_label = picker_label.downgrade();
    click.connect_pressed(move |_, _n, x, y| {
        let (Some(pic), Some(label)) = (pick_pic.upgrade(), pick_label.upgrade()) else {
            return;
        };
        // Sample the last displayed render (no re-render).
        if let Some(c) = pick_ctx.last_render.borrow().as_ref() {
            let (w, h) = (c.width as usize, c.height as usize);
            match crate::preview::map_widget_to_image(
                pic.width() as f64, pic.height() as f64, w, h, x, y,
            ) {
                Some((px, py)) => {
                    if let Some((rr, gg, bb)) =
                        crate::preview::sample_pixel(&c.bytes, w, h, c.rowstride, c.nch, px, py)
                    {
                        label.set_text(&format!(
                            "Pick ({px},{py}):  R {rr}  G {gg}  B {bb}   #{rr:02X}{gg:02X}{bb:02X}"
                        ));
                    }
                }
                None => label.set_text("Pick: (outside image)"),
            }
        }
    });
    picture.add_controller(click);

    // ── Left: image (+ snapshot wipe overlay) over histogram + picker ──────
    // ALIGNMENT INVARIANT: the Overlay allocates `wipe_area` the *same* rect as
    // `picture`, and both letterbox via `preview::contain_rect` — that equal
    // allocation + shared geometry is what makes a feature land at the same panel
    // pixel on both sides of the wipe. (It does NOT rely on the snapshot and live
    // image sharing dimensions; `draw_wipe` uses the snapshot's own dims.)
    let image_overlay = gtk4::Overlay::builder()
        .hexpand(true)
        .vexpand(true)
        .build();
    image_overlay.set_child(Some(&picture));
    image_overlay.add_overlay(&wipe_area);
    image_overlay.add_overlay(&crop_area);

    let image_area = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .hexpand(true)
        .build();
    image_area.append(&image_overlay);
    image_area.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    image_area.append(&hist_area);
    image_area.append(&picker_label);

    // ── IOP module list (right panel) — hosts the live param widgets ───────
    let (modules_panel, panel_box) = build_modules_panel(&ctx);

    // Before/after toggle (created here so the history handlers below can clear
    // its bypass state on restore; packed into the header later).
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

    // ── History panel (above the modules) — click an entry to jump to it ───
    // One `row-activated` handler (set here, never rebuilt) restores that entry's
    // params and repopulates the module sliders, mirroring the Reset path. The
    // jump moves the history cursor *onto* the restored entry, so the render it
    // triggers records nothing new (the recorder dedups against the cursor).
    {
        let jump_ctx = ctx.clone();
        let jump_panel = panel_box.downgrade();
        let jump_list = history_list.downgrade();
        let jump_ba = before_after_btn.downgrade();
        history_list.connect_row_activated(move |_, row| {
            let idx = row.index();
            if idx < 0 {
                return;
            }
            let restored = jump_ctx.history.borrow_mut().jump_to(idx as usize);
            if let Some(p) = restored {
                apply_history_params(&jump_ctx, &jump_panel, &jump_list, &jump_ba, p);
            }
        });
    }

    let history_header = gtk4::Label::builder()
        .label("History")
        .halign(gtk4::Align::Start)
        .margin_top(12)
        .margin_bottom(6)
        .margin_start(12)
        .margin_end(12)
        .build();
    history_header.add_css_class("heading");
    let history_scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .min_content_height(120)
        .max_content_height(220)
        .child(&history_list)
        .build();
    let history_section = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .build();
    history_section.append(&history_header);
    history_section.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    history_section.append(&history_scroll);

    // ── Snapshots panel — take the current view + compare side-by-side ─────
    let snapshot_list = gtk4::ListBox::builder()
        .selection_mode(gtk4::SelectionMode::Single)
        .build();
    snapshot_list.add_css_class("navigation-sidebar");

    // Compare handler (set once): clicking a snapshot loads it into the wipe
    // overlay. Reads the frozen payload by row index; the row rebuild keeps
    // indices aligned with the store. Clone the payload out so the store borrow
    // is dropped before `show` builds the cairo surface.
    {
        let snap_store = snapshots.clone();
        let snap_wipe = wipe.clone();
        snapshot_list.connect_row_activated(move |_, row| {
            let idx = row.index();
            if idx < 0 {
                return;
            }
            let payload = snap_store.borrow().get(idx as usize).map(|s| s.payload.clone());
            if let Some(p) = payload {
                snap_wipe.show(&p);
            }
        });
    }

    // "Take snapshot": freeze the last displayed render (the cached pixels) into
    // the store. No-op until the first render has populated `last_render`.
    let take_btn = gtk4::Button::builder()
        .icon_name("list-add-symbolic")
        .has_frame(false)
        .tooltip_text("Take a snapshot of the current view")
        .build();
    take_btn.update_property(&[gtk4::accessible::Property::Label("Take snapshot")]);
    {
        let take_ctx = ctx.clone();
        let take_store = snapshots.clone();
        let take_list = snapshot_list.downgrade();
        let take_wipe = wipe.clone();
        take_btn.connect_clicked(move |_| {
            let current = take_ctx.last_render.borrow().clone();
            if let Some(cr) = current {
                take_store.borrow_mut().capture(cr);
                if let Some(l) = take_list.upgrade() {
                    refresh_snapshot_list(&l, &take_store, &take_wipe);
                }
            }
        });
    }

    // "Stop comparing": clear the wipe overlay and the row selection.
    let stop_btn = gtk4::Button::builder()
        .icon_name("view-restore-symbolic")
        .has_frame(false)
        .tooltip_text("Stop comparing")
        .build();
    stop_btn.update_property(&[gtk4::accessible::Property::Label("Stop comparing")]);
    {
        let stop_wipe = wipe.clone();
        let stop_list = snapshot_list.downgrade();
        stop_btn.connect_clicked(move |_| {
            stop_wipe.clear();
            if let Some(l) = stop_list.upgrade() {
                l.unselect_all();
            }
        });
    }

    let snapshot_header = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(6)
        .margin_top(12)
        .margin_bottom(6)
        .margin_start(12)
        .margin_end(6)
        .build();
    let snapshot_title = gtk4::Label::builder()
        .label("Snapshots")
        .halign(gtk4::Align::Start)
        .hexpand(true)
        .build();
    snapshot_title.add_css_class("heading");
    snapshot_header.append(&snapshot_title);
    snapshot_header.append(&take_btn);
    snapshot_header.append(&stop_btn);

    let snapshot_scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .min_content_height(90)
        .max_content_height(180)
        .child(&snapshot_list)
        .build();
    let snapshot_section = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .build();
    snapshot_section.append(&snapshot_header);
    snapshot_section.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    snapshot_section.append(&snapshot_scroll);
    // Seed the placeholder row.
    refresh_snapshot_list(&snapshot_list, &snapshots, &wipe);

    // Right column: history over snapshots over modules.
    let right_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .width_request(320)
        .build();

    // Bayer demosaic-method selector — raw images only (JPEGs aren't
    // demosaiced). Changing it re-decodes the raw with the chosen algorithm and
    // persists the choice per image. The DropDown's index is DemosaicMethod's
    // as_u8 code, so it round-trips through from_u8 with no extra mapping.
    if crate::raw_preview::is_raw_path(file_path) {
        // Header + dropdown live in one box so the whole section can be hidden
        // as a unit once a decode reveals an X-Trans sensor (Markesteijn-fixed).
        let demosaic_box = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
        let demosaic_header = gtk4::Label::builder()
            .label("Demosaic")
            .halign(gtk4::Align::Start)
            .margin_start(10)
            .margin_top(8)
            .margin_bottom(4)
            .build();
        demosaic_header.add_css_class("heading");
        let dropdown = gtk4::DropDown::from_strings(&demosaic_method_labels());
        dropdown.set_margin_start(10);
        dropdown.set_margin_end(10);
        dropdown.set_margin_bottom(8);
        // The method only affects Bayer sensors; X-Trans (.raf) always uses
        // Markesteijn, so the selector is hidden for X-Trans once decoded. The
        // tooltip still notes the caveat for the pre-decode window.
        dropdown.set_tooltip_text(Some(
            "Bayer demosaic algorithm (X-Trans files always use Markesteijn)",
        ));
        dropdown.update_property(&[gtk4::accessible::Property::Label("Demosaic algorithm")]);
        // Seed the current selection WITHOUT re-decoding: set before connecting
        // the handler (the initial `spawn_decode` already used this method).
        dropdown.set_selected(demosaic.as_u8() as u32);
        let dd_ctx = ctx.clone();
        let dd_path = file_path.to_string();
        let dd_db = db_path.to_string();
        dropdown.connect_selected_notify(move |dd| {
            let method = DemosaicMethod::from_u8(dd.selected() as u8);
            crate::persist::save_demosaic(&dd_db, &dd_path, method);
            spawn_decode(&dd_ctx, dd_path.clone(), method);
        });
        demosaic_box.append(&demosaic_header);
        demosaic_box.append(&dropdown);
        demosaic_box.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
        // Let the first decode toggle visibility for X-Trans (see spawn_decode).
        ctx.demosaic_row.set(Some(&demosaic_box));
        right_box.append(&demosaic_box);

        // Straighten (rotate) — a per-image geometry edit re-applied to the
        // cached pristine buffer (not a re-decode). The heavy resample + DB write
        // are debounced so a slider drag stays responsive.
        let geom_header = gtk4::Label::builder()
            .label("Geometry")
            .halign(gtk4::Align::Start)
            .margin_start(10)
            .margin_top(8)
            .margin_bottom(4)
            .build();
        geom_header.add_css_class("heading");
        let angle0_deg = straighten_rad_to_deg(ctx.geometry.get().angle);
        // labeled_slider sets the value internally, so the handler is connected
        // *after* the initial value → no spurious geometry apply on build.
        let straighten = labeled_slider("Straighten", -45.0, 45.0, 0.1, angle0_deg);
        straighten
            .scale
            .widget
            .set_tooltip_text(Some("Rotate the image, degrees"));
        let g_ctx = ctx.clone();
        let g_path = file_path.to_string();
        let g_db = db_path.to_string();
        let g_debounce: Rc<RefCell<Option<glib::SourceId>>> = Rc::new(RefCell::new(None));
        straighten.scale.connect_value_changed(move |v| {
            let mut geom = g_ctx.geometry.get();
            geom.angle = straighten_deg_to_rad(v);
            g_ctx.geometry.set(geom);
            // Debounce the resample + persist so a drag doesn't thrash them; the
            // last value within the window is the one applied.
            if let Some(id) = g_debounce.borrow_mut().take() {
                id.remove();
            }
            let d_ctx = g_ctx.clone();
            let d_path = g_path.clone();
            let d_db = g_db.clone();
            let d_deb = g_debounce.clone();
            let id =
                glib::timeout_add_local_once(std::time::Duration::from_millis(160), move || {
                    *d_deb.borrow_mut() = None;
                    crate::persist::save_geometry(&d_db, &d_path, &d_ctx.geometry.get());
                    apply_geometry_to_base(&d_ctx);
                });
            *g_debounce.borrow_mut() = Some(id);
        });
        // Reset crop + straighten to identity — the only way to undo a geometry
        // edit (the header Reset deliberately touches only the colour params).
        let reset_geom_btn = gtk4::Button::builder()
            .label("Reset crop & straighten")
            .tooltip_text("Clear the crop and straighten (geometry only)")
            .margin_start(10)
            .margin_end(10)
            .margin_bottom(6)
            .build();
        let rg_ctx = ctx.clone();
        // BauhausSlider is Rc-backed and Clone; no GObject weak ref needed.
        let rg_scale = straighten.scale.clone();
        let rg_crop = crop_area.downgrade();
        let rg_db = db_path.to_string();
        let rg_path = file_path.to_string();
        reset_geom_btn.connect_clicked(move |_| {
            rg_ctx.geometry.set(Geometry::default());
            // Reflect it on the slider (a no-op notify if already 0); the reset
            // itself applies + persists below, so it doesn't rely on the handler.
            rg_scale.set_value(0.0);
            crate::persist::save_geometry(&rg_db, &rg_path, &Geometry::default());
            apply_geometry_to_base(&rg_ctx); // re-render the (now un-cropped) frame
            if let Some(area) = rg_crop.upgrade() {
                area.queue_draw(); // redraw the overlay's identity rect (if editing)
            }
        });

        // Crop aspect-ratio lock — applies to crop-handle resizes (crop mode).
        let aspect_row = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(8)
            .margin_start(10)
            .margin_end(10)
            .margin_bottom(6)
            .build();
        let aspect_lbl = gtk4::Label::new(Some("Aspect"));
        aspect_lbl.set_xalign(0.0);
        let aspect_dd = gtk4::DropDown::from_strings(&crate::crop_overlay::aspect_labels());
        aspect_dd.set_hexpand(true);
        aspect_dd.set_tooltip_text(Some("Lock the crop aspect ratio (applies while dragging a crop handle)"));
        // The lock is a session tool (not persisted → resets to Free on reopen),
        // but selecting a ratio immediately reshapes the CURRENT crop to it (and
        // that crop IS persisted) so the displayed/exported crop matches the
        // selection without needing a drag — matching darktable/Lightroom.
        let a_cell = crop_aspect.clone();
        let as_ctx = ctx.clone();
        let as_crop = crop_area.downgrade();
        let as_db = db_path.to_string();
        let as_path = file_path.to_string();
        aspect_dd.connect_selected_notify(move |dd| {
            let ratio = crate::crop_overlay::aspect_from_index(dd.selected());
            a_cell.set(ratio);
            if let Some((bw, bh)) = base_dims(&as_ctx.base.borrow()) {
                let mut g = as_ctx.geometry.get();
                // Fit the ratio inside the current crop, centred (Free = no-op).
                g.crop = crate::crop_overlay::fit_aspect(g.crop, ratio, bw, bh);
                as_ctx.geometry.set(g);
                crate::persist::save_geometry(&as_db, &as_path, &g);
                if let Some(area) = as_crop.upgrade() {
                    area.queue_draw(); // redraw the overlay rect (crop-edit mode)
                }
                if !as_ctx.crop_editing.get() {
                    apply_geometry_to_base(&as_ctx); // re-render the cropped result
                }
            }
        });
        aspect_row.append(&aspect_lbl);
        aspect_row.append(&aspect_dd);

        right_box.append(&geom_header);
        right_box.append(&straighten.row);
        right_box.append(&aspect_row);
        right_box.append(&reset_geom_btn);
        right_box.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    }

    right_box.append(&history_section);
    right_box.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    right_box.append(&snapshot_section);
    right_box.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    right_box.append(&modules_panel);

    // ── Split view: image | (history / modules) ────────────────────────────
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    content.append(&image_area);
    content.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    content.append(&right_box);

    // Wrap the content so export (and future) status can surface as a toast —
    // the darkroom view previously logged export results only to stderr.
    let toast_overlay = adw::ToastOverlay::new();
    toast_overlay.set_child(Some(&content));

    // ── Header bar with Export button ─────────────────────────────────────
    let header = adw::HeaderBar::new();
    // View switcher (shared with the lighttable header) as the title, with the
    // filename below so the editor still shows which image is open. "Darkroom" is
    // the active view; "Lighttable" pops back to the grid via the NavigationView's
    // built-in `navigation.pop` action (installed on the page's descendants), so
    // no NavigationView handle is needed here.
    let sw = crate::build_view_switcher();
    sw.darkroom.set_active(true);
    sw.lighttable.connect_clicked(|b| {
        if b.is_active() {
            // Pop back to the lighttable via the NavigationView's built-in action
            // (this page holds no NavigationView handle). Only reachable while the
            // page is in the nav, so an Err would mean a future reparent broke that
            // invariant — log it rather than fail silently.
            if let Err(e) = b.activate_action("navigation.pop", None) {
                eprintln!("darkroom: view-switcher pop failed (page not in a NavigationView?): {e}");
            }
        }
    });
    header.set_title_widget(Some(&crate::view_switcher_title(&sw.container, &filename)));

    // Before/after toggle was created above (so the history restore can clear
    // its state); just place it in the header here.
    header.pack_start(&before_after_btn);

    // Crop-mode toggle (raw only). On: show the rotated-uncropped frame + the
    // interactive crop overlay. Off: show the cropped result and persist.
    if crate::raw_preview::is_raw_path(file_path) {
        let crop_btn = gtk4::ToggleButton::builder()
            .label("Crop")
            .tooltip_text("Crop / straighten overlay")
            .build();
        crop_btn.update_property(&[gtk4::accessible::Property::Label("Crop mode")]);
        let cb_ctx = ctx.clone();
        let cb_area = crop_area.downgrade();
        let cb_wipe = wipe.clone();
        let cb_db = db_path.to_string();
        let cb_path = file_path.to_string();
        crop_btn.connect_toggled(move |b| {
            let editing = b.is_active();
            cb_ctx.crop_editing.set(editing);
            if editing {
                // Only one overlay may be interactive: dismiss any active wipe
                // compare (it would show through the dim bands and its gesture
                // would fight the crop drag).
                cb_wipe.clear();
            }
            if let Some(area) = cb_area.upgrade() {
                area.set_can_target(editing); // interactive only while editing
                area.queue_draw();
            }
            // Re-render the base: rotated-uncropped while editing, cropped when done.
            apply_geometry_to_base(&cb_ctx);
            if !editing {
                crate::persist::save_geometry(&cb_db, &cb_path, &cb_ctx.geometry.get());
            }
        });
        header.pack_start(&crop_btn);
    }

    // Undo / Redo: step the history cursor and restore that entry. No-ops at the
    // ends (undo at the seed / redo at the newest return None), so the buttons
    // stay enabled without sensitivity tracking.
    let undo_btn = gtk4::Button::builder()
        .icon_name("edit-undo-symbolic")
        .tooltip_text("Undo")
        .build();
    undo_btn.update_property(&[gtk4::accessible::Property::Label("Undo")]);
    let undo_ctx = ctx.clone();
    let undo_panel = panel_box.downgrade();
    let undo_list = history_list.downgrade();
    let undo_ba = before_after_btn.downgrade();
    undo_btn.connect_clicked(move |_| {
        let restored = undo_ctx.history.borrow_mut().undo();
        if let Some(p) = restored {
            apply_history_params(&undo_ctx, &undo_panel, &undo_list, &undo_ba, p);
        }
    });
    header.pack_start(&undo_btn);

    let redo_btn = gtk4::Button::builder()
        .icon_name("edit-redo-symbolic")
        .tooltip_text("Redo")
        .build();
    redo_btn.update_property(&[gtk4::accessible::Property::Label("Redo")]);
    let redo_ctx = ctx.clone();
    let redo_panel = panel_box.downgrade();
    let redo_list = history_list.downgrade();
    let redo_ba = before_after_btn.downgrade();
    redo_btn.connect_clicked(move |_| {
        let restored = redo_ctx.history.borrow_mut().redo();
        if let Some(p) = restored {
            apply_history_params(&redo_ctx, &redo_panel, &redo_list, &redo_ba, p);
        }
    });
    header.pack_start(&redo_btn);

    // Keyboard shortcuts: Ctrl+Z = Undo, Ctrl+Shift+Z / Ctrl+Y = Redo. Each just
    // re-emits the matching button's `clicked`, so the history logic lives in one
    // place (and `undo`/`redo` bounds-check by returning None at the ends, so a
    // repeated key past the seed/tip is a safe no-op regardless of the button).
    // Attached to the page root (`toolbar_view`, below) with `Local` scope so it
    // covers the whole page — header buttons included — and dies cleanly when the
    // page is popped (no leak to the lighttable).
    let shortcuts = gtk4::ShortcutController::new();
    shortcuts.set_scope(gtk4::ShortcutScope::Local);
    let emit_click = |btn: &gtk4::Button| {
        let w = btn.downgrade();
        gtk4::CallbackAction::new(move |_, _| {
            if let Some(b) = w.upgrade() {
                b.emit_clicked();
            }
            glib::Propagation::Stop
        })
    };
    if let Some(t) = gtk4::ShortcutTrigger::parse_string("<Control>z") {
        shortcuts.add_shortcut(gtk4::Shortcut::new(Some(t), Some(emit_click(&undo_btn))));
    }
    for combo in ["<Control><Shift>z", "<Control>y"] {
        if let Some(t) = gtk4::ShortcutTrigger::parse_string(combo) {
            shortcuts.add_shortcut(gtk4::Shortcut::new(Some(t), Some(emit_click(&redo_btn))));
        }
    }

    // Reset: restore default params and rebuild the panel so the sliders follow.
    // (Distinct icon from Undo so the two aren't confused.)
    let reset_btn = gtk4::Button::builder()
        .icon_name("edit-clear-all-symbolic")
        .tooltip_text("Reset all adjustments")
        .build();
    reset_btn.update_property(&[gtk4::accessible::Property::Label("Reset all adjustments")]);
    let reset_ctx = ctx.clone();
    let reset_panel = panel_box.downgrade();
    let reset_ba = before_after_btn.downgrade();
    let reset_list = history_list.downgrade();
    reset_btn.connect_clicked(move |_| {
        *reset_ctx.params.borrow_mut() = PreviewParams::default();
        reset_ctx.bypass.set(false); // source of truth for bypass
        if let Some(ba) = reset_ba.upgrade() {
            ba.set_active(false); // sync the button visual (bypass already cleared)
        }
        // Record an explicit "Reset" entry now (the debounced recorder would
        // otherwise label it by which module changed); the later render's
        // recorder then dedups against this entry.
        let changed = reset_ctx
            .history
            .borrow_mut()
            .record("Reset", PreviewParams::default());
        if changed {
            if let Some(list) = reset_list.upgrade() {
                refresh_history_list(&list, &reset_ctx.history.borrow());
            }
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
    let ex_ctx = ctx.clone();
    let ex_db = db_path.to_string();
    let ex_overlay = toast_overlay.clone();
    export_btn.connect_clicked(move |btn| {
        if let Some(root) = btn.root().and_downcast::<gtk4::Window>() {
            // Bake the current c41-ui edit so a Rust-native export matches
            // the preview (geometry uses the committed crop even mid-crop-edit).
            // Raws render via our pipeline; a JPEG falls back to darktable-cli.
            let edit = crate::export::ExportEdit {
                method: crate::persist::load_demosaic(&ex_db, &path_for_export),
                geometry: ex_ctx.geometry.get(),
                params: *ex_ctx.params.borrow(),
            };
            // Surface the export result as a toast (the callback runs back on the
            // main thread after the export future resolves).
            let tf_overlay = ex_overlay.clone();
            dialogs::show_export_dialog(
                root.upcast_ref::<gtk4::Window>(),
                vec![path_for_export.clone()],
                Some(edit),
                None, // fixed edit above; no per-image catalog lookup needed
                move |msg| tf_overlay.add_toast(adw::Toast::new(&msg)),
            );
        }
    });
    header.pack_end(&export_btn);

    // ── Colour-label dot row (m4-24) ──────────────────────────────────────
    // Mirror the lighttable's 5-dot colour row in the header so the single-image
    // view shows (and can toggle) the same labels. Reuses the lighttable toolkit
    // for one source of truth on dot geometry, hues, DB read and toggle wiring.
    // The box is static (no cell recycling here), but `wire_color_clicks`' repaint
    // is guarded by `widget_name() == path`, so we stamp the box with `file_path`
    // for the guard to pass. The initial mask is read synchronously — consistent
    // with the history/params seeds already loaded sync at open just above.
    let colors_box = crate::lighttable::build_color_dots_box();
    colors_box.set_widget_name(file_path);
    colors_box.set_margin_end(8);
    colors_box.set_tooltip_text(Some("Colour labels (click to toggle)"));
    let initial_mask = crate::lighttable::query_color_labels(file_path, db_path);
    crate::lighttable::set_color_dots(&colors_box, initial_mask);
    crate::lighttable::wire_color_clicks(&colors_box, file_path.to_string(), db_path.to_string());
    header.pack_end(&colors_box);

    // ── Star-rating row (m4-28) ───────────────────────────────────────────
    // Mirror the lighttable's 5-star row, left of the colour dots. Reuses the
    // lighttable toolkit (build/set/wire, one source of truth). Unlike the colour
    // row, `wire_star_clicks` repaints synchronously and has no async read-back, so
    // it needs no `widget_name` stamp. Initial rating read synchronously, as above.
    let stars_box = crate::lighttable::build_stars_box();
    stars_box.set_margin_end(8);
    stars_box.set_tooltip_text(Some("Star rating (click a star to set)"));
    let initial_rating = crate::lighttable::query_rating(file_path, db_path).unwrap_or(0);
    crate::lighttable::set_stars(&stars_box, initial_rating);
    crate::lighttable::wire_star_clicks(&stars_box, file_path.to_string(), db_path.to_string());
    header.pack_end(&stars_box);

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&toast_overlay));
    // Page-root scope: covers header + content, scoped to this page.
    toolbar_view.add_controller(shortcuts);

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
    scale: crate::bauhaus::BauhausSlider,
}

/// Build one darktable-style parameter row.
///
/// The label and value are drawn *inside* the control by
/// [`crate::bauhaus::BauhausSlider`] rather than sitting in a separate label
/// widget beside a GTK `Scale` — that is what makes darktable's panels read as
/// dense rows of bars instead of a column of handles. The wrapping `Box` stays
/// so callers keep appending a single row widget.
fn labeled_slider(label: &str, min: f64, max: f64, step: f64, value: f64) -> LabeledSlider {
    let scale = crate::bauhaus::BauhausSlider::new(label, min, max, step, value);
    let row = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .margin_start(8).margin_end(8).margin_top(1).margin_bottom(1)
        .build();
    row.append(&scale.widget);
    LabeledSlider { row, scale }
}

/// Module-stack panel: the darktable module groups (base/tone/color/correct/
/// effect) from [`crate::catalog`]. Modules backed by a migrated `c41-core`
/// IOP (Exposure, Velvia) render as expandable rows with a live enable switch
/// and parameter sliders wired to the preview pipeline; the rest are inert
/// enable-toggle rows. The navigable history panel is built separately in
/// [`darkroom_page`] and lives above this (it survives the Reset/jump rebuilds).
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
    // Count first so the header can state how much of the catalogue is real.
    // Without this the panel is 44 near-identical rows and the ~30 placeholders
    // are indistinguishable from working modules — which made every shipped
    // increment invisible (see `is_live_module`).
    let total: usize = crate::catalog::module_catalog()
        .iter()
        .map(|g| g.modules.len())
        .sum();
    let live: usize = crate::catalog::module_catalog()
        .iter()
        .flat_map(|g| g.modules.iter())
        .filter(|mi| is_live_module(mi.label))
        .count();

    let header = gtk4::Label::builder()
        .label("Modules")
        .halign(gtk4::Align::Start)
        .build();
    header.add_css_class("title-4");
    panel.append(&header);

    let counter = gtk4::Label::builder()
        .label(format!("{live} of {total} active"))
        .halign(gtk4::Align::Start)
        .build();
    counter.add_css_class("dim-label");
    counter.add_css_class("caption");
    panel.append(&counter);

    for group in crate::catalog::module_catalog() {
        let pg = adw::PreferencesGroup::builder().title(group.name).build();
        // Live modules first: the catalogue is authored in darktable's
        // presentation order, which front-loads unported modules (Base opens
        // with three inert rows, Effect with nine), so a top-down scan hit
        // placeholders before anything that works. Stable within each half, so
        // the familiar relative order survives.
        let mut modules: Vec<&crate::catalog::ModuleInfo> = group.modules.iter().collect();
        modules.sort_by_key(|mi| !is_live_module(mi.label));
        for mi in modules {
            match mi.label {
                "Exposure" => pg.add(&exposure_module_row(ctx)),
                "Velvia" => pg.add(&velvia_module_row(ctx)),
                "Split-toning" => pg.add(&splittoning_module_row(ctx)),
                "Monochrome" => pg.add(&monochrome_module_row(ctx)),
                "Sigmoid" => pg.add(&sigmoid_module_row(ctx)),
                "Sharpen" => pg.add(&sharpen_module_row(ctx)),
                "Vibrance" => pg.add(&vibrance_module_row(ctx)),
                "Colorize" => pg.add(&colorize_module_row(ctx)),
                "Color correction" => pg.add(&colorcorrection_module_row(ctx)),
                "Color contrast" => pg.add(&colorcontrast_module_row(ctx)),
                "Primaries" => pg.add(&primaries_module_row(ctx)),
                "Negadoctor" => pg.add(&negadoctor_module_row(ctx)),
                "Tone equalizer" => pg.add(&toneequal_module_row(ctx)),
                "Color zones" => pg.add(&colorzones_module_row(ctx)),
                "Levels" => pg.add(&levels_module_row(ctx)),
                "Vignetting" => pg.add(&vignette_module_row(ctx)),
                "Lowlight vision" => pg.add(&lowlight_module_row(ctx)),
                "Graduated density" => pg.add(&gradnd_module_row(ctx)),
                "Contrast brightness saturation" => pg.add(&colisa_module_row(ctx)),
                "Basic adjustments" => pg.add(&basicadj_module_row(ctx)),
                "Shadows/Highlights" => pg.add(&shadhi_module_row(ctx)),
                "Lowpass" => pg.add(&lowpass_module_row(ctx)),
                "White balance" => pg.add(&whitebalance_module_row(ctx)),
                "Invert" => pg.add(&invert_module_row(ctx)),
                other => match elsewhere_hint(other) {
                    // Implemented, but driven from its own control — point at it
                    // rather than calling it unwired.
                    Some(hint) => pg.add(&elsewhere_module_row(other, hint)),
                    None => pg.add(&inert_module_row(other, mi.default_on)),
                },
            }
        }
        panel.append(&pg);
    }
}

/// A placeholder for a module whose processing exists in `c41-core` but has
/// no controls wired to it yet.
///
/// Deliberately **not** interactive. It previously rendered as a normal row with
/// a working switch, so ~30 of the 44 catalogue entries looked exactly like the
/// live ones and could be toggled with no effect — which reads as "the app is
/// broken" rather than "this module isn't wired yet", and buried each shipped
/// module in a crowd of lookalikes. Now it is dimmed, labelled, and inert.
fn inert_module_row(label: &str, _default_on: bool) -> adw::ActionRow {
    let row = adw::ActionRow::builder()
        .title(label)
        .subtitle("not yet wired")
        .build();
    row.add_css_class("dim-label");
    // No switch at all: a disabled switch still suggests "turn me on", while an
    // icon reads as a status. Sensitivity off so it can't take focus either.
    let marker = gtk4::Image::from_icon_name("content-loading-symbolic");
    marker.set_valign(gtk4::Align::Center);
    marker.set_tooltip_text(Some(
        "Processing is ported to c41-core; the panel controls are not built yet.",
    ));
    row.add_suffix(&marker);
    row.set_activatable(false);
    row.set_sensitive(false);
    row
}

/// A row for a module that works but is driven from a control elsewhere in the
/// panel. Not dimmed — it is functional — but non-activatable, since its switch
/// would have nothing to gate.
fn elsewhere_module_row(label: &str, hint: &str) -> adw::ActionRow {
    let row = adw::ActionRow::builder().title(label).subtitle(hint).build();
    let marker = gtk4::Image::from_icon_name("go-up-symbolic");
    marker.set_valign(gtk4::Align::Center);
    row.add_suffix(&marker);
    row.set_activatable(false);
    row
}

/// Whether `label` dispatches to a live preview module in [`populate_modules`].
///
/// Single source of truth for both the dispatch's presentation (ordering, the
/// "N of M active" count) and the drift test — previously this list existed only
/// under `#[cfg(test)]`, so the UI had no way to tell a working module from a
/// placeholder and rendered them identically.
pub(crate) fn is_live_module(label: &str) -> bool {
    LIVE_MODULE_LABELS.contains(&label) || ELSEWHERE_MODULE_LABELS.contains(&label)
}

/// Catalog labels whose functionality *is* implemented, but through dedicated
/// controls elsewhere in the darkroom panel rather than a row in this list —
/// Crop has its own mode toggle and overlay, Rotate & perspective the Straighten
/// slider. Marking them "not yet wired" would be simply false, so they count as
/// live and are labelled with where their controls actually are.
const ELSEWHERE_MODULE_LABELS: &[&str] = &["Crop", "Rotate & perspective"];

/// Where a module implemented outside this list keeps its controls.
fn elsewhere_hint(label: &str) -> Option<&'static str> {
    match label {
        "Crop" => Some("use the Crop button above"),
        "Rotate & perspective" => Some("use the Straighten slider above"),
        _ => None,
    }
}

/// Catalog labels that [`build_modules_panel`] dispatches to a *live* preview
/// module (everything else renders via [`inert_module_row`]). The match arms in
/// [`populate_modules`] use these same literals, and `catalog_has_live_modules`
/// guards against a catalog rename silently dropping a module back to inert.
///
/// Keep in sync with the match arms — adding a module means adding it here too,
/// or it will render live but be counted and sorted as a placeholder.
const LIVE_MODULE_LABELS: &[&str] = &["Exposure", "Velvia", "Split-toning", "Monochrome", "Sigmoid", "Sharpen", "Vibrance", "Colorize", "Color correction", "Color contrast", "Color zones", "Levels", "Vignetting", "Lowlight vision", "Graduated density", "Contrast brightness saturation", "Basic adjustments", "Shadows/Highlights", "Lowpass", "Primaries", "Negadoctor", "Tone equalizer", "Invert", "White balance"];

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
    row.scale.connect_value_changed(move |v| {
        set(&mut ctx_cl.params.borrow_mut(), v as f32);
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
pub(crate) fn initial_params(saved: Option<PreviewParams>, is_raw: bool) -> PreviewParams {
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

/// Sharpen module: enable switch gates `sharpen_on`; radius (0..100), amount
/// (0..5), and threshold (0..100) sliders. Runs scene-referred between channelmixer
/// and sigmoid (darktable iop_order.c); the Stage handles RGB↔Lab internally.
///
/// Radius is capped at 12 kernel taps (MAXR), matching darktable's sharpen.c —
/// values above ~12 produce diminishing returns because the Gaussian widens
/// while the kernel window stays fixed at 2·12+1 taps (see
/// `c41_core::pipeline::gaussian_kernel`).
fn sharpen_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Sharpen", "edge sharpening", p0.sharpen_on,
        |p, on| p.sharpen_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Radius", 0.0, 100.0, 0.5, p0.sharpen_radius as f64,
                |p, v| p.sharpen_radius = v);
            add_param_slider(e, ctx, "Amount", 0.0, 5.0, 0.01, p0.sharpen_amount as f64,
                |p, v| p.sharpen_amount = v);
            add_param_slider(e, ctx, "Threshold", 0.0, 100.0, 0.1, p0.sharpen_threshold as f64,
                |p, v| p.sharpen_threshold = v);
        })
}

/// Vibrance module: enable switch gates `vibrance_on`; an Amount slider (0..100).
/// Runs scene-referred between sharpen and sigmoid (darktable iop_order.c pos 39.1);
/// the Stage handles RGB↔Lab internally, using the same ColorSpace as Sharpen.
fn vibrance_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Vibrance", "saturation boost", p0.vibrance_on,
        |p, on| p.vibrance_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Amount", 0.0, 100.0, 1.0, p0.vibrance_amount as f64,
                |p, v| p.vibrance_amount = v);
        })
}

fn colorcontrast_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Color contrast", "chroma non-linearity", p0.color_contrast_on,
        |p, on| p.color_contrast_on = on,
        |e, ctx| {
            // Steepness 1.0 is identity (no chroma change); slider range 0.0..=5.0.
            add_param_slider(e, ctx, "A steepness", 0.0, 5.0, 1.0, p0.color_contrast_a_steepness as f64,
                |p, v| p.color_contrast_a_steepness = v);
            add_param_slider(e, ctx, "B steepness", 0.0, 5.0, 1.0, p0.color_contrast_b_steepness as f64,
                |p, v| p.color_contrast_b_steepness = v);
        })
}

fn colorcorrection_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Color correction", "shadow/highlight chroma recovery", p0.color_correction_on,
        |p, on| p.color_correction_on = on,
        |e, ctx| {
            // Shadow a-channel offset (loa). Range -100..100 covers the Lab a
            // axis (-128..127) with headroom.
            add_param_slider(e, ctx, "Shadow a", -100.0, 100.0, 1.0, p0.color_correction_loa as f64,
                |p, v| p.color_correction_loa = v);
            add_param_slider(e, ctx, "Highlight a", -100.0, 100.0, 1.0, p0.color_correction_hia as f64,
                |p, v| p.color_correction_hia = v);
            add_param_slider(e, ctx, "Shadow b", -100.0, 100.0, 1.0, p0.color_correction_lob as f64,
                |p, v| p.color_correction_lob = v);
            add_param_slider(e, ctx, "Highlight b", -100.0, 100.0, 1.0, p0.color_correction_hib as f64,
                |p, v| p.color_correction_hib = v);
            // Global saturation: -3..3, default 1.0 = no change.
            add_param_slider(e, ctx, "Saturation", -3.0, 3.0, 0.01, p0.color_correction_saturation as f64,
                |p, v| p.color_correction_saturation = v);
        })
}

fn colorzones_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Color zones", "LCH equaliser", p0.colorzones_on,
        |p, on| p.colorzones_on = on,
        |e, ctx| {
            // Channel: 0 = L, 1 = C, 2 = h
            add_param_slider(e, ctx, "Channel", 0.0, 2.0, 1.0, p0.colorzones_channel as f64,
                |p, v| p.colorzones_channel = v);
            // Mode: 0 = smooth (v3), 1 = strong (v1)
            add_param_slider(e, ctx, "Mode", 0.0, 1.0, 1.0, p0.colorzones_mode as f64,
                |p, v| p.colorzones_mode = v);
            // Strength: 0..100 on the C slider scale
            add_param_slider(e, ctx, "Strength", 0.0, 100.0, 1.0, p0.colorzones_strength as f64,
                |p, v| p.colorzones_strength = v);
        })
}

/// Levels module: enable switch gates `levels_on`; black / grey / white point
/// sliders on darktable's 0..100 scale. Grey centred between black and white is
/// gamma 1 (identity); moving it brightens or darkens the midtones. The stage
/// builds its 65536-entry LUT in `to_pipeline`, as the C `commit_params` does.
fn levels_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Levels", "black / grey / white points", p0.levels_on,
        |p, on| p.levels_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Black", 0.0, 100.0, 0.5, p0.levels_black as f64,
                |p, v| p.levels_black = v);
            add_param_slider(e, ctx, "Grey", 0.0, 100.0, 0.5, p0.levels_grey as f64,
                |p, v| p.levels_grey = v);
            add_param_slider(e, ctx, "White", 0.0, 100.0, 0.5, p0.levels_white as f64,
                |p, v| p.levels_white = v);
        })
}

/// Vignetting module: enable switch gates `vignette_on`; fall-off start/radius,
/// brightness and saturation strength, shape, and the centre offset.
///
/// This is the first *position-dependent* live module — its stage is not
/// pixel-local, so enabling it puts the whole pipeline on the serial path (see
/// `Stage::Vignette`). Nothing to do here, but it is why this module is more
/// expensive to have on than the others.
fn vignette_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Vignetting", "radial brightness falloff", p0.vignette_on,
        |p, on| p.vignette_on = on,
        |e, ctx| {
            // Both strengths at 0 is a no-op (to_pipeline skips the stage);
            // darktable's default is -0.5 for each, i.e. a darkening vignette.
            add_param_slider(e, ctx, "Brightness", -1.0, 1.0, 0.01, p0.vignette_brightness as f64,
                |p, v| p.vignette_brightness = v);
            add_param_slider(e, ctx, "Saturation", -1.0, 1.0, 0.01, p0.vignette_saturation as f64,
                |p, v| p.vignette_saturation = v);
            // Inner radius then falloff width, both % of the largest dimension.
            add_param_slider(e, ctx, "Fall-off start", 0.0, 200.0, 1.0, p0.vignette_scale as f64,
                |p, v| p.vignette_scale = v);
            add_param_slider(e, ctx, "Fall-off radius", 0.0, 200.0, 1.0, p0.vignette_falloff as f64,
                |p, v| p.vignette_falloff = v);
            // 1 = ellipse; higher squares the shape off.
            add_param_slider(e, ctx, "Shape", 0.0, 5.0, 0.01, p0.vignette_shape as f64,
                |p, v| p.vignette_shape = v);
            add_param_slider(e, ctx, "Centre X", -1.0, 1.0, 0.01, p0.vignette_center_x as f64,
                |p, v| p.vignette_center_x = v);
            add_param_slider(e, ctx, "Centre Y", -1.0, 1.0, 0.01, p0.vignette_center_y as f64,
                |p, v| p.vignette_center_y = v);
        })
}

/// Lowlight vision module: enable switch gates `lowlight_on`; a blue-shift
/// slider plus the six transition bands, which set how strongly the scotopic
/// (rod) response is mixed in at each luminance — band 0 is the darkest zone,
/// band 5 the brightest. darktable draws these as a curve widget; six sliders
/// carry the same parameters until a curve editor exists.
fn lowlight_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Lowlight vision", "scotopic night vision", p0.lowlight_on,
        |p, on| p.lowlight_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Blue shift", 0.0, 100.0, 1.0, p0.lowlight_blueness as f64,
                |p, v| p.lowlight_blueness = v);
            // 0.5 across all bands is darktable's default (an even blend).
            add_param_slider(e, ctx, "Zone 1 (dark)", 0.0, 1.0, 0.01, p0.lowlight_transition[0] as f64,
                |p, v| p.lowlight_transition[0] = v);
            add_param_slider(e, ctx, "Zone 2", 0.0, 1.0, 0.01, p0.lowlight_transition[1] as f64,
                |p, v| p.lowlight_transition[1] = v);
            add_param_slider(e, ctx, "Zone 3", 0.0, 1.0, 0.01, p0.lowlight_transition[2] as f64,
                |p, v| p.lowlight_transition[2] = v);
            add_param_slider(e, ctx, "Zone 4", 0.0, 1.0, 0.01, p0.lowlight_transition[3] as f64,
                |p, v| p.lowlight_transition[3] = v);
            add_param_slider(e, ctx, "Zone 5", 0.0, 1.0, 0.01, p0.lowlight_transition[4] as f64,
                |p, v| p.lowlight_transition[4] = v);
            add_param_slider(e, ctx, "Zone 6 (bright)", 0.0, 1.0, 0.01, p0.lowlight_transition[5] as f64,
                |p, v| p.lowlight_transition[5] = v);
        })
}

/// Graduated ND module: enable switch gates `gradnd_on`; density, hardness,
/// rotation, offset and an optional tint.
///
/// Like Vignetting this is a *position-dependent* stage, so enabling it puts the
/// pipeline on the serial path (see `Stage::GraduatedNd`). darktable also offers
/// an on-canvas line handle to set rotation/offset by dragging; the sliders here
/// carry the same parameters until that overlay exists.
fn gradnd_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Graduated density", "graduated ND filter", p0.gradnd_on,
        |p, on| p.gradnd_on = on,
        |e, ctx| {
            // Density in EV; negative brightens rather than darkens. 0 is a
            // no-op and to_pipeline skips the stage there.
            add_param_slider(e, ctx, "Density", -8.0, 8.0, 0.05, p0.gradnd_density as f64,
                |p, v| p.gradnd_density = v);
            add_param_slider(e, ctx, "Hardness", 0.0, 100.0, 1.0, p0.gradnd_hardness as f64,
                |p, v| p.gradnd_hardness = v);
            add_param_slider(e, ctx, "Rotation", -180.0, 180.0, 1.0, p0.gradnd_rotation as f64,
                |p, v| p.gradnd_rotation = v);
            // 50 = the line through the frame centre.
            add_param_slider(e, ctx, "Offset", 0.0, 100.0, 1.0, p0.gradnd_offset as f64,
                |p, v| p.gradnd_offset = v);
            // Saturation 0 keeps the filter neutral (a true ND); raise it for
            // the classic tinted-grad look.
            add_param_slider(e, ctx, "Hue", 0.0, 1.0, 0.01, p0.gradnd_hue as f64,
                |p, v| p.gradnd_hue = v);
            add_param_slider(e, ctx, "Saturation", 0.0, 1.0, 0.01, p0.gradnd_saturation as f64,
                |p, v| p.gradnd_saturation = v);
        })
}

/// Contrast/brightness/saturation (colisa): three -1..1 sliders, all neutral at
/// 0. darktable's own note on this module is "edit contrast while damaging
/// colour" — it is the blunt instrument, which is why it sits in the
/// display-referred cluster rather than among the scene-referred tools.
fn colisa_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Contrast brightness saturation", "quick tone + colour", p0.colisa_on,
        |p, on| p.colisa_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Contrast", -1.0, 1.0, 0.01, p0.colisa_contrast as f64,
                |p, v| p.colisa_contrast = v);
            add_param_slider(e, ctx, "Brightness", -1.0, 1.0, 0.01, p0.colisa_brightness as f64,
                |p, v| p.colisa_brightness = v);
            add_param_slider(e, ctx, "Saturation", -1.0, 1.0, 0.01, p0.colisa_saturation as f64,
                |p, v| p.colisa_saturation = v);
        })
}

/// Basic adjustments (basicadj): black point, exposure, highlight compression,
/// brightness, contrast, saturation and vibrance in one module.
///
/// Ranges are darktable's own `$MIN`/`$MAX` from `dt_iop_basicadj_params_t`,
/// except exposure: upstream allows -18..18 EV but its slider soft-range is far
/// narrower, and a full-width -18..18 control makes every useful adjustment
/// sub-pixel. -3..3 matches what the exposure module already exposes here.
///
/// Two of darktable's params are deliberately not surfaced. `clip` has no
/// implementation in the migrated kernel, so a slider for it would do nothing.
/// `preserve_colors` is an enum, not a slider, and defaults to LUMINANCE; it
/// wants a dropdown, which is a separate increment rather than a fake slider.
///
/// Upstream's own iop_order note on this module is "mixing view/model/control at
/// once, usage should be discouraged" — it overlaps exposure, filmic and
/// colorbalancergb. It is here because the processing is ported and darktable
/// still ships it, not as the recommended path.
fn basicadj_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Basic adjustments", "black, exposure, tone", p0.basicadj_on,
        |p, on| p.basicadj_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Black level", -1.0, 1.0, 0.001, p0.basicadj_black_point as f64,
                |p, v| p.basicadj_black_point = v);
            add_param_slider(e, ctx, "Exposure", -3.0, 3.0, 0.01, p0.basicadj_exposure as f64,
                |p, v| p.basicadj_exposure = v);
            // `hlcompr` has a slider (darktable exposes it at 0..100 soft-max 500);
            // `hlcomprthresh` does NOT — darktable only sets it via auto-exposure,
            // so it stays at its 0.0 default here. Exposing it would be a control
            // that does nothing useful from the user's perspective.
            add_param_slider(e, ctx, "Highlight compression", 0.0, 500.0, 1.0, p0.basicadj_hlcompr as f64,
                |p, v| p.basicadj_hlcompr = v);
            add_param_slider(e, ctx, "Contrast", -1.0, 5.0, 0.01, p0.basicadj_contrast as f64,
                |p, v| p.basicadj_contrast = v);
            add_param_slider(e, ctx, "Middle gray", 0.05, 100.0, 0.01, p0.basicadj_middle_grey as f64,
                |p, v| p.basicadj_middle_grey = v);
            add_param_slider(e, ctx, "Brightness", -4.0, 4.0, 0.01, p0.basicadj_brightness as f64,
                |p, v| p.basicadj_brightness = v);
            add_param_slider(e, ctx, "Saturation", -1.0, 1.0, 0.01, p0.basicadj_saturation as f64,
                |p, v| p.basicadj_saturation = v);
            add_param_slider(e, ctx, "Vibrance", -1.0, 1.0, 0.01, p0.basicadj_vibrance as f64,
                |p, v| p.basicadj_vibrance = v);
        })
}

/// Shadows/Highlights (shadhi.c): a Gaussian-blurred base layer is merged with
/// the original Lab pixels to lift shadows and recover highlights. Not exposed
/// here: the C bilateral algorithm (the GUI default) — we hardcode Gaussian
/// because `crate::gaussian` only implements that, and the shadow/highlight math
/// is identical regardless of blur kernel.
///
/// Ranges mirror `dt_iop_shadhi_params_v5_t` from `src/iop/shadhi.c`:
/// shadows/highlights -100..100, whitepoint -10..10, radius 0.1..500,
/// compress 0..100, ccorrect 0..100. `flags` (UNBOUND_DEFAULT) and
/// `low_approximation` (0.000001) are hardcoded in the Stage apply arm.
fn shadhi_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Shadows/Highlights", "local contrast recovery", p0.shadhi_on,
        |p, on| p.shadhi_on = on,
        |e, ctx| {
            // The darktable slider order is shadows, highlights, whitepoint,
            // radius, compress, shadows_ccorrect, highlights_ccorrect — matching
            // the params struct field order in shadhi.c.
            add_param_slider(e, ctx, "Shadows", -100.0, 100.0, 1.0, p0.shadhi_shadows as f64,
                |p, v| p.shadhi_shadows = v);
            add_param_slider(e, ctx, "Highlights", -100.0, 100.0, 1.0, p0.shadhi_highlights as f64,
                |p, v| p.shadhi_highlights = v);
            add_param_slider(e, ctx, "Whitepoint", -10.0, 10.0, 0.1, p0.shadhi_whitepoint as f64,
                |p, v| p.shadhi_whitepoint = v);
            add_param_slider(e, ctx, "Radius", 0.1, 500.0, 1.0, p0.shadhi_radius as f64,
                |p, v| p.shadhi_radius = v);
            add_param_slider(e, ctx, "Compress", 0.0, 100.0, 1.0, p0.shadhi_compress as f64,
                |p, v| p.shadhi_compress = v);
            add_param_slider(e, ctx, "Shadows color adj.", 0.0, 100.0, 1.0, p0.shadhi_shadows_ccorrect as f64,
                |p, v| p.shadhi_shadows_ccorrect = v);
            add_param_slider(e, ctx, "Highlights color adj.", 0.0, 100.0, 1.0, p0.shadhi_highlights_ccorrect as f64,
                |p, v| p.shadhi_highlights_ccorrect = v);
        })
}

/// Lowpass (local contrast enhancement): a Gaussian blur of the image, then
/// contrast/brightness LUTs + a/b saturation applied to the blurred copy.
///
/// Ranges mirror `dt_iop_lowpass_params_t` from `src/iop/lowpass.c`: radius
/// 0.1..500 (default 10), contrast/brightness/saturation all -3..3 (defaults
/// 1.0 / 0.0 / 1.0). `unbound` is not surfaced — darktable's default is 1 (true),
/// and the GUI checkbox only appears in the scene-referred safety path we don't
/// expose here.
fn lowpass_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Lowpass", "local contrast boost", p0.lowpass_on,
        |p, on| p.lowpass_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Radius", 0.1, 500.0, 1.0, p0.lowpass_radius as f64,
                |p, v| p.lowpass_radius = v);
            add_param_slider(e, ctx, "Contrast", -3.0, 3.0, 0.01, p0.lowpass_contrast as f64,
                |p, v| p.lowpass_contrast = v);
            add_param_slider(e, ctx, "Brightness", -3.0, 3.0, 0.01, p0.lowpass_brightness as f64,
                |p, v| p.lowpass_brightness = v);
            add_param_slider(e, ctx, "Saturation", -3.0, 3.0, 0.01, p0.lowpass_saturation as f64,
                |p, v| p.lowpass_saturation = v);
        })
}

/// Primaries (primaries.c): rotate and scale each working-space primary
/// around the white point. Two controls per channel — hue (degrees) and purity
/// (multiplier, 1.0 = unchanged). Achromatic tint shifts the white point itself.
///
/// Ranges are darktable's *soft* ranges (from primaries.c `_setup_*_slider`),
/// NOT the introspection hard range (hue ±180°, purity 0.01..5.0). The hard hue
/// range is unusable: `rotate_and_scale_primary` projects primaries onto the
/// gamut hull, so |hue| ≳ 112° lands them on the opposite triangle edge, making
/// all three primaries collinear and the matrix singular (coefficients ~1e16).
/// darktable hides this behind the soft range; we do too. Tint hue keeps its
/// full ±180° — darktable sets no soft range on it, and it is safe (moves only
/// the white point, leaving the primaries triangle intact).
fn primaries_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Primaries", "primary hue & purity", p0.primaries_on,
        |p, on| p.primaries_on = on,
        |e, ctx| {
            // Tint hue: full ±180° is safe (moves only the white point).
            add_param_slider(e, ctx, "Achromatic tint hue", -180.0, 180.0, 1.0, p0.primaries_achromatic_tint_hue as f64,
                |p, v| p.primaries_achromatic_tint_hue = v);
            // Tint purity: darktable's soft range is 0..0.2.
            add_param_slider(e, ctx, "Achromatic tint purity", 0.0, 0.99, 0.01, p0.primaries_achromatic_tint_purity as f64,
                |p, v| p.primaries_achromatic_tint_purity = v);
            // RGB hue: darktable's soft range is ±20° (hard is ±180°).
            add_param_slider(e, ctx, "Red hue", -20.0, 20.0, 1.0, p0.primaries_red_hue as f64,
                |p, v| p.primaries_red_hue = v);
            // RGB purity: darktable's soft range is 0.5..1.5 (hard is 0.01..5.0).
            add_param_slider(e, ctx, "Red purity", 0.5, 1.5, 0.01, p0.primaries_red_purity as f64,
                |p, v| p.primaries_red_purity = v);
            add_param_slider(e, ctx, "Green hue", -20.0, 20.0, 1.0, p0.primaries_green_hue as f64,
                |p, v| p.primaries_green_hue = v);
            add_param_slider(e, ctx, "Green purity", 0.5, 1.5, 0.01, p0.primaries_green_purity as f64,
                |p, v| p.primaries_green_purity = v);
            add_param_slider(e, ctx, "Blue hue", -20.0, 20.0, 1.0, p0.primaries_blue_hue as f64,
                |p, v| p.primaries_blue_hue = v);
            add_param_slider(e, ctx, "Blue purity", 0.5, 1.5, 0.01, p0.primaries_blue_purity as f64,
                |p, v| p.primaries_blue_purity = v);
        })
}

/// Negadoctor (negadoctor.c): Cinéon-style log-density film negative inversion
/// with print-on-paper simulation. Display-referred, iop_order.c pos 28.5.
fn negadoctor_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    let is_bw = (p0.negadoctor_film_stock as i32) == 0;

    module_expander(ctx, "Negadoctor", "film negative inversion", p0.negadoctor_on,
        |p, on| p.negadoctor_on = on,
        |e, ctx| {
            // Dmin G/B: created first so the film-stock callback can toggle their
            // visibility (darktable toggle_stock_controls:negadoctor.c:388-410
            // hides them when film_stock == B&W). Added to the expander after
            // Dmin R, keeping the same visual order darktable uses.
            let dmin_g = Rc::new(labeled_slider("Dmin G", 0.00001, 1.5, 0.001,
                p0.negadoctor_dmin_g as f64));
            let dmin_b = Rc::new(labeled_slider("Dmin B", 0.00001, 1.5, 0.001,
                p0.negadoctor_dmin_b as f64));
            dmin_g.row.set_visible(!is_bw);
            dmin_b.row.set_visible(!is_bw);

            let ctx_g = ctx.clone();
            dmin_g.scale.connect_value_changed(move |v| {
                ctx_g.params.borrow_mut().negadoctor_dmin_g = v as f32;
                render_preview(&ctx_g);
            });
            let ctx_b = ctx.clone();
            dmin_b.scale.connect_value_changed(move |v| {
                ctx_b.params.borrow_mut().negadoctor_dmin_b = v as f32;
                render_preview(&ctx_b);
            });

            // Film stock: 0 = B&W, 1 = colour (darktable DT_FILMSTOCK_NB / COLOR).
            // Uses labeled_slider directly (not add_param_slider) so the
            // callback can toggle Dmin G/B visibility — add_param_slider takes
            // a bare fn pointer and can't capture widget handles.
            let dg = dmin_g.clone();
            let db = dmin_b.clone();
            let filmstock = labeled_slider("Film stock", 0.0, 1.0, 1.0,
                p0.negadoctor_film_stock as f64);
            let ctx_fs = ctx.clone();
            filmstock.scale.connect_value_changed(move |v| {
                let bw = (v as i32) == 0;
                {
                    let mut p = ctx_fs.params.borrow_mut();
                    p.negadoctor_film_stock = v as u8 as f32;
                    // Mirror Dmin R → G/B in B&W mode (gui_changed:negadoctor.c:953-957).
                    if bw {
                        p.negadoctor_dmin_g = p.negadoctor_dmin_r;
                        p.negadoctor_dmin_b = p.negadoctor_dmin_r;
                    }
                }
                dg.row.set_visible(!bw);
                db.row.set_visible(!bw);
                render_preview(&ctx_fs);
            });
            e.add_row(&filmstock.row);

            // Dmin R: per-channel minimum film density. Mono-collapse to R in
            // to_pipeline when film_stock is B&W. Range 0.00001..1.5.
            // In B&W mode the callback also mirrors to G/B params (gui_changed:negadoctor.c:953-957);
            // the G/B sliders are hidden, so only params need updating.
            add_param_slider(e, ctx, "Dmin R", 0.00001, 1.5, 0.001, p0.negadoctor_dmin_r as f64,
                |p, v| {
                    p.negadoctor_dmin_r = v;
                    if (p.negadoctor_film_stock as i32) == 0 {
                        p.negadoctor_dmin_g = v;
                        p.negadoctor_dmin_b = v;
                    }
                });
            e.add_row(&dmin_g.row);
            e.add_row(&dmin_b.row);
            // White balance high (illuminant multipliers). Range 0.25..2.
            add_param_slider(e, ctx, "WB high R", 0.25, 2.0, 0.001, p0.negadoctor_wb_high_r as f64,
                |p, v| p.negadoctor_wb_high_r = v);
            add_param_slider(e, ctx, "WB high G", 0.25, 2.0, 0.001, p0.negadoctor_wb_high_g as f64,
                |p, v| p.negadoctor_wb_high_g = v);
            add_param_slider(e, ctx, "WB high B", 0.25, 2.0, 0.001, p0.negadoctor_wb_high_b as f64,
                |p, v| p.negadoctor_wb_high_b = v);
            // White balance low (base light offsets). Range 0.25..2.
            add_param_slider(e, ctx, "WB low R", 0.25, 2.0, 0.001, p0.negadoctor_wb_low_r as f64,
                |p, v| p.negadoctor_wb_low_r = v);
            add_param_slider(e, ctx, "WB low G", 0.25, 2.0, 0.001, p0.negadoctor_wb_low_g as f64,
                |p, v| p.negadoctor_wb_low_g = v);
            add_param_slider(e, ctx, "WB low B", 0.25, 2.0, 0.001, p0.negadoctor_wb_low_b as f64,
                |p, v| p.negadoctor_wb_low_b = v);
            // D_max: maximum film density. Range 0.1..6.
            add_param_slider(e, ctx, "D max", 0.1, 6.0, 0.001, p0.negadoctor_d_max as f64,
                |p, v| p.negadoctor_d_max = v);
            // Offset: inversion offset. Range -1..1.
            add_param_slider(e, ctx, "Offset", -1.0, 1.0, 0.001, p0.negadoctor_offset as f64,
                |p, v| p.negadoctor_offset = v);
            // Black point: affects the print-linear to display mapping.
            add_param_slider(e, ctx, "Black", -0.5, 0.5, 0.001, p0.negadoctor_black as f64,
                |p, v| p.negadoctor_black = v);
            // Gamma: display gamma (Cinéon 2.2 power, darktable default 4.0 = 1/0.222).
            add_param_slider(e, ctx, "Gamma", 1.0, 8.0, 0.01, p0.negadoctor_gamma as f64,
                |p, v| p.negadoctor_gamma = v);
            // Soft clip: highlight compression threshold in [0,1].
            add_param_slider(e, ctx, "Soft clip", 0.0001, 1.0, 0.001, p0.negadoctor_soft_clip as f64,
                |p, v| p.negadoctor_soft_clip = v);
            // Exposure: slider in EV (−1..=1), param stores the linear multiplier
            // 2^EV. Mirrors darktable negadoctor.c gui_init:925-929 (slider in EV,
            // range −1..+1, format "EV"), gui_changed:964 (param = 2^slider) and
            // gui_update:988 (slider = log2(param)).
            let exposure_ev = (p0.negadoctor_exposure as f64).log2();
            add_param_slider(e, ctx, "Exposure (EV)", -1.0, 1.0, 0.01, exposure_ev,
                |p, v| p.negadoctor_exposure = 2.0f32.powf(v));
        })
}

/// Tone equalizer (toneequal.c): nine per-exposure-channel gain sliders, one
/// per EV band from −8 EV to 0 EV. Labels pair darktable's params-struct
/// descriptions with the EV positions its GUI shows
/// (`dt_bauhaus_widget_set_label`, toneequal.c:3205-3213). All sliders are
/// −2..+2 EV, step 0.01 ($MIN/$MAX/$DEFAULT in the params struct comments,
/// toneequal.c:172-180); all zero = flat unity correction.
///
/// Scope note: this runs the `details == DT_TONEEQ_NONE` configuration
/// ("preserve details: no") — darktable's default is the guided-filter mode,
/// which is not ported; the smoothing/feathering/blending controls only affect
/// those modes or the mask display, so they are not surfaced.
fn toneequal_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Tone equalizer", "exposure channel tone mapping", p0.toneeq_on,
        |p, on| p.toneeq_on = on,
        |e, ctx| {
            add_param_slider(e, ctx, "Blacks (−8 EV)", -2.0, 2.0, 0.01, p0.toneeq_noise as f64,
                |p, v| p.toneeq_noise = v);
            add_param_slider(e, ctx, "Deep shadows (−7 EV)", -2.0, 2.0, 0.01,
                p0.toneeq_ultra_deep_blacks as f64,
                |p, v| p.toneeq_ultra_deep_blacks = v);
            add_param_slider(e, ctx, "Shadows (−6 EV)", -2.0, 2.0, 0.01,
                p0.toneeq_deep_blacks as f64,
                |p, v| p.toneeq_deep_blacks = v);
            add_param_slider(e, ctx, "Light shadows (−5 EV)", -2.0, 2.0, 0.01,
                p0.toneeq_blacks as f64,
                |p, v| p.toneeq_blacks = v);
            add_param_slider(e, ctx, "Mid-tones (−4 EV)", -2.0, 2.0, 0.01,
                p0.toneeq_shadows as f64,
                |p, v| p.toneeq_shadows = v);
            add_param_slider(e, ctx, "Dark highlights (−3 EV)", -2.0, 2.0, 0.01,
                p0.toneeq_midtones as f64,
                |p, v| p.toneeq_midtones = v);
            add_param_slider(e, ctx, "Highlights (−2 EV)", -2.0, 2.0, 0.01,
                p0.toneeq_highlights as f64,
                |p, v| p.toneeq_highlights = v);
            add_param_slider(e, ctx, "Whites (−1 EV)", -2.0, 2.0, 0.01, p0.toneeq_whites as f64,
                |p, v| p.toneeq_whites = v);
            add_param_slider(e, ctx, "Speculars (+0 EV)", -2.0, 2.0, 0.01,
                p0.toneeq_speculars as f64,
                |p, v| p.toneeq_speculars = v);
        })
}

fn whitebalance_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "White balance", "channel multipliers", p0.temperature_on,
        |p, on| p.temperature_on = on,
        |e, ctx| {
            // Per-channel RGB multipliers (1.0 = no change). Range 0..=4 to cover
            // extreme WB shifts while staying bounded.
            add_param_slider(e, ctx, "Red", 0.0, 4.0, 0.01, p0.temperature_r as f64,
                |p, v| p.temperature_r = v);
            add_param_slider(e, ctx, "Green", 0.0, 4.0, 0.01, p0.temperature_g as f64,
                |p, v| p.temperature_g = v);
            add_param_slider(e, ctx, "Blue", 0.0, 4.0, 0.01, p0.temperature_b as f64,
                |p, v| p.temperature_b = v);
        })
}

fn invert_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Invert", "film negative inversion", p0.invert_on,
        |p, on| p.invert_on = on,
        |e, ctx| {
            // Per-channel film-back colour: out = color - in (default 1.0 = negate).
            // Range 0..=4 covers extreme inversions while staying bounded.
            add_param_slider(e, ctx, "Red", 0.0, 4.0, 0.01, p0.invert_r as f64,
                |p, v| p.invert_r = v);
            add_param_slider(e, ctx, "Green", 0.0, 4.0, 0.01, p0.invert_g as f64,
                |p, v| p.invert_g = v);
            add_param_slider(e, ctx, "Blue", 0.0, 4.0, 0.01, p0.invert_b as f64,
                |p, v| p.invert_b = v);
        })
}

fn colorize_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    module_expander(ctx, "Colorize", "HSL colour replacement", p0.colorize_on,
        |p, on| p.colorize_on = on,
        |e, ctx| {
            // Hue 0..1 (normalised colour wheel), sat 0..1, lightness 0..100.
            add_param_slider(e, ctx, "Hue", 0.0, 1.0, 0.01, p0.colorize_hue as f64,
                |p, v| p.colorize_hue = v);
            add_param_slider(e, ctx, "Saturation", 0.0, 1.0, 0.01, p0.colorize_sat as f64,
                |p, v| p.colorize_sat = v);
            add_param_slider(e, ctx, "Lightness", 0.0, 100.0, 1.0, p0.colorize_lightness as f64,
                |p, v| p.colorize_lightness = v);
            // Source lightness mix: how much of the input L is kept (0 = full colour,
            // 100 = full input luminance). Core gets ×0.01.
            add_param_slider(e, ctx, "Source mix", 0.0, 100.0, 1.0, p0.colorize_lightness_mix as f64,
                |p, v| p.colorize_lightness_mix = v);
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

    #[test]
    fn demosaic_labels_align_with_method_codes() {
        // The selector relies on DropDown index == DemosaicMethod::as_u8, so a
        // change handler can `from_u8(dd.selected())`. Pin one label per method,
        // each code in range, and no duplicate labels.
        let labels = demosaic_method_labels();
        assert_eq!(labels.len(), 3);
        for (i, m) in [DemosaicMethod::Rcd, DemosaicMethod::Vng, DemosaicMethod::Ppg]
            .into_iter()
            .enumerate()
        {
            assert_eq!(m.as_u8() as usize, i, "{m:?} code must equal its label index");
            assert_eq!(DemosaicMethod::from_u8(i as u8), m);
        }
        assert!(!labels[0].is_empty() && labels[0] != labels[1] && labels[1] != labels[2]);
    }

    #[test]
    fn straighten_deg_rad_round_trip() {
        // 0 maps to 0; a mid value round-trips within float tolerance; the
        // slider extremes map to the expected radian magnitude (±π/4).
        assert_eq!(straighten_deg_to_rad(0.0), 0.0);
        for deg in [-45.0, -12.3, 7.5, 45.0] {
            let back = straighten_rad_to_deg(straighten_deg_to_rad(deg));
            assert!((back - deg).abs() < 1e-4, "deg {deg} -> {back}");
        }
        assert!((straighten_deg_to_rad(45.0) - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    }

    #[test]
    fn linear_base_render_shape() {
        // BaseImage::Linear renders to tightly-packed RGB8 (nch 3, rowstride w*3)
        // — the texture-upload contract the Linear arm relies on.
        let b = BaseImage::Linear {
            width: 2,
            height: 1,
            pixels: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        };
        let r = b.render(&PreviewParams::default());
        assert_eq!((r.width, r.height), (2, 1));
        assert_eq!(r.nch, 3);
        assert_eq!(r.rowstride, 2 * 3);
        assert_eq!(r.bytes.len(), 2 * 1 * 3);
    }

    /// `LIVE_MODULE_LABELS` now drives presentation (ordering + the "N of M
    /// active" count), not just this test, so a module present in the dispatch
    /// but missing from the list would render live yet be sorted and counted as
    /// a placeholder. Pin the two together by parsing the match arms from source.
    #[test]
    fn live_module_labels_match_the_dispatch_arms() {
        let src = include_str!("mod.rs");
        let dispatched: std::collections::BTreeSet<&str> = src
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                let rest = l.strip_prefix('"')?;
                let (label, tail) = rest.split_once('"')?;
                tail.trim_start().starts_with("=> pg.add").then_some(label)
            })
            .collect();
        let declared: std::collections::BTreeSet<&str> =
            LIVE_MODULE_LABELS.iter().copied().collect();
        assert_eq!(
            dispatched, declared,
            "LIVE_MODULE_LABELS and the populate_modules match arms disagree — \
             a module would be rendered live but counted/sorted as inert (or vice versa)"
        );
        // The "implemented elsewhere" labels are deliberately NOT dispatch arms;
        // they must not overlap, or a module would get two rows.
        for l in ELSEWHERE_MODULE_LABELS {
            assert!(
                !declared.contains(l),
                "{l} is both a dispatch arm and an ELSEWHERE label"
            );
            assert!(elsewhere_hint(l).is_some(), "{l} has no hint text");
            assert!(is_live_module(l), "{l} should count as live");
        }
        assert!(
            declared.iter().all(|l| is_live_module(l)),
            "is_live_module disagrees with its own backing list"
        );
        assert!(!is_live_module("Bloom"), "an unported module must not read as live");
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
