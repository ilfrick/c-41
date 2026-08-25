//! Lighttable **zoomable** view mode (m4-139, parity audit 3.1) — darktable's
//! infinite pan plane of thumbnails, laid out in a grid whose cell size varies
//! continuously with the zoom.
//!
//! A [`gtk4::GridView`] cannot express this layout (integer column counts, no
//! continuous sizing, a model-driven plane rather than an infinite one), which is
//! why the mode shipped greyed out for so long. It is a hand-drawn
//! [`gtk4::DrawingArea`] instead: the plane geometry comes from [`plane_of`], the
//! visible band is painted every frame from a bounded texture cache, and the
//! ScrolledWindow's native scrolling provides the pan. All geometry lives in pure
//! functions below (tested display-free); the widget side only wires them to
//! allocations, adjustments and gestures.
//!
//! **Interactions**, mirroring darktable's zoomable view as far as the existing
//! controls allow: plain wheel zooms continuously (one multiplicative stop per
//! notch, anchored at the cursor — the m4-133 adjustment discipline: immediate
//! write plus a generation-guarded idle re-assert once GTK has reallocated at the
//! new request), primary-button drag pans, single click selects through the SAME
//! `SingleSelection` the grid uses (so the metadata panel, rating shortcuts and
//! export follow automatically), double-click opens the darkroom page exactly
//! like a grid-cell activation, and the bottom bar's thumb-size stepper is
//! repurposed as an images-per-row control — the precedent is culling, which
//! repurposes it as the comparison-set size. One canonical zoom state (a cell
//! size in px, [`ZOOM_CELL`]); wheel and stepper are two writers of it.
//!
//! **Thumbnails come from gdk-pixbuf** — the same source the file-manager grid
//! uses, with the same known limitation: camera raws get no placeholder here
//! either (the full preview is where raws become visible; sharing its decode
//! pipeline with this canvas is a recorded follow-up, not part of this slice).
//! Decodes are keyed `(path, bucket)` where the bucket quantises the target size
//! to a power of two ([`texture_bucket`]) — continuous zooming would otherwise
//! fire a fresh decode per pixel of cell growth. At most one decode per path is
//! ever in flight, and a newer larger request supersedes an older smaller one.
//! The cache is an LRU under a byte budget ([`TEXTURE_BUDGET_BYTES`],
//! [`PixbufCache`]), because a high-zoom thumbnail can be megapixels and an
//! unbounded map would grow with every visit.
//!
//! **Sentinel rows never reach the canvas**: the grid model carries placeholder
//! entries (empty-state / truncation notices, no `/`) that a cell bind classifies
//! away; here the item list simply filters them out and the empty case draws its
//! own message.
//!
//! The zoom level is session-only state, unlike the *mode* itself which persists
//! through the ordinary view-mode token — restoring a session lands at the
//! default zoom, like darktable's per-view zoom reset.

use adw::prelude::*;
use glib::signal::Propagation;
use gtk4::{
    gdk, gio, glib, DrawingArea, EventControllerMotion, EventControllerScroll,
    EventControllerScrollFlags, GestureClick, GestureDrag, ScrolledWindow, SingleSelection,
};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

/// Smallest cell, in px. Below this thumbnails stop being informative even at a
/// glance — darktable's zoomable bottoms out around the same visual density.
pub(crate) const CELL_MIN_PX: i32 = 48;

/// Gap between neighbouring cells, in px. Part of the pitch, not the cell: two
/// cells touch edge-to-edge at `2·cell + GAP`, and the plane maths treats the
/// pitch as atomic so a gap can never be clipped off by rounding.
const CELL_GAP_PX: i32 = 8;

/// Caption strip INSIDE the cell's bottom edge, in px. Filenames draw over this
/// band (darktable draws them below the frame; the band keeps the pitch square
/// and the hit-test trivial).
const CAPTION_PX: i32 = 16;

/// Default cell size — the zoom a fresh session starts at. Near the
/// file-manager's THUMB_SIZE so switching modes doesn't lurch.
const DEFAULT_CELL_PX: f64 = 192.0;

/// One wheel notch scales the cell by this much.
const WHEEL_FACTOR: f64 = 1.25;

/// Texture-cache byte budget. A viewport-sized texture at high zoom is ~8 MB;
/// 96 MB holds a screenful of those with slack, and eviction keeps worst-case
/// memory flat no matter how large the collection grows.
const TEXTURE_BUDGET_BYTES: u64 = 96 * 1024 * 1024;

// ── Pure core ───────────────────────────────────────────────────────────────
//
// Every function in this section is display-free and unit-tested below; the
// widget code calls them and nothing else does geometry.

/// How many whole cells fit across `viewport_w` at cell size `cell`. Always ≥ 1:
/// a cell wider than the viewport still lays out one per row (and clips), rather
/// than dividing down to zero columns and vanishing.
fn zoom_columns(viewport_w: i32, cell: i32) -> u32 {
    if viewport_w <= 0 || cell <= 0 {
        return 1;
    }
    (viewport_w / cell).max(1) as u32
}

/// Largest useful cell size for a viewport: one image filling the longer axis.
/// Clamped into `[CELL_MIN_PX, 4096]` so degenerate or enormous viewports can't
/// push the zoom range anywhere silly.
fn zoom_cell_max(viewport_w: i32, viewport_h: i32) -> i32 {
    f64::from(viewport_w.max(viewport_h).max(CELL_MIN_PX))
        .clamp(f64::from(CELL_MIN_PX), 4096.0) as i32
}

/// The plane geometry for `n_items` at cell size `cell`: integer column count
/// from the viewport width, rows ceiled from the count, extent from the pitch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plane {
    cols: u32,
    rows: u32,
    /// Total painted extent (cols·pitch / rows·pitch).
    plane_w: i32,
    plane_h: i32,
}

/// Cell pitch (cell + one gap). Atomic everywhere: rects derive from it, so a
/// gap is never lost to rounding.
fn pitch(cell_px: i32) -> i32 {
    cell_px.max(1) + CELL_GAP_PX
}

/// The cell size this plane was computed at, back-derived from the extent. Both
/// painters and the hit-test need it; deriving it keeps [`Plane`] the single
/// carrier of geometry so no caller can mix two different cell sizes.
fn plane_cell(p: &Plane) -> i32 {
    ((p.plane_w / p.cols.max(1) as i32) - CELL_GAP_PX).max(1)
}

/// Compute the plane. `n_items == 0` yields a zero-extent plane (the empty-state
/// message is drawn centred in the viewport instead, not on the plane).
fn plane_of(viewport_w: i32, n_items: usize, cell: f64) -> Plane {
    let cell_i = cell.round().max(1.0) as i32;
    let cols = zoom_columns(viewport_w, cell_i);
    let rows: u32 =
        if n_items == 0 { 0 } else { n_items.div_ceil(cols as usize) as u32 };
    // Empty collections get a fully zero extent: nothing is laid out, the
    // empty-state message paints in the viewport instead.
    let width = if n_items == 0 { 0 } else { cols as i32 * pitch(cell_i) };
    Plane {
        cols,
        rows,
        plane_w: width,
        plane_h: rows as i32 * pitch(cell_i),
    }
}

/// Top-left of item `index`'s cell on the plane. Caller guarantees
/// `index < n_items` (both enforced by [`plane_of`]).
fn cell_origin(p: &Plane, index: u32) -> (f64, f64) {
    let cols = p.cols.max(1);
    let col = index % cols;
    let row = index / cols;
    let pch = pitch(plane_cell(p));
    (f64::from(col) * f64::from(pch), f64::from(row) * f64::from(pch))
}

/// Which item sits at plane point `(x, y)`, or `None` outside the plane / in a
/// gap. Inverse of [`cell_origin`] over the valid index range — the click
/// handler and the paint loop therefore cannot disagree about what is where.
fn index_at_point(p: &Plane, n_items: usize, x: f64, y: f64) -> Option<u32> {
    if n_items == 0 || p.cols == 0 || x < 0.0 || y < 0.0 {
        return None;
    }
    let pch = pitch(plane_cell(p));
    let col = (x / f64::from(pch)).floor();
    let row = (y / f64::from(pch)).floor();
    if col < 0.0 || row < 0.0 || col >= f64::from(p.cols) || row >= f64::from(p.rows.max(1)) {
        return None;
    }
    // Inside the pitch but past the cell proper = the gap between cells.
    let fx = x - col * f64::from(pch);
    let fy = y - row * f64::from(pch);
    if fx >= f64::from(plane_cell(p)) || fy >= f64::from(plane_cell(p)) {
        return None;
    }
    let index = row as u32 * p.cols + col as u32;
    (index < n_items as u32).then_some(index)
}

/// One wheel step. Multiplicative both ways so the zoom feels symmetric around
/// the current value, clamped to `[CELL_MIN_PX, max]`, and pinned exactly at the
/// ends once close (a limit approached asymptotically reads as a dead wheel).
fn zoom_step_cell(cell: f64, zoom_in: bool, max_px: i32) -> f64 {
    let lo = f64::from(CELL_MIN_PX);
    let hi = f64::from(max_px.max(CELL_MIN_PX));
    let next = if zoom_in { cell * WHEEL_FACTOR } else { cell / WHEEL_FACTOR };
    let clamped = next.clamp(lo, hi);
    if (clamped - lo).abs() < 2.0 {
        lo
    } else if (hi - clamped).abs() < 2.0 {
        hi
    } else {
        clamped
    }
}

/// The cell size that fits `columns` across the viewport, clamped into the legal
/// zoom range. This is the stepper's write path — it speaks columns because that
/// is what the control means everywhere else; the canvas stores cell pixels.
fn zoom_columns_to_cell(viewport_w: i32, viewport_h: i32, columns: u32) -> f64 {
    let raw = f64::from(viewport_w.max(1)) / f64::from(columns.max(1));
    raw.clamp(f64::from(CELL_MIN_PX), f64::from(zoom_cell_max(viewport_w, viewport_h)))
}

/// Scrollbar value that keeps plane point `old_content` under the same viewport
/// position after the plane scaled by `k`: solve `new_value + cursor_viewport ==
/// old_content · k`. The caller assigns it; GTK clamps to the adjustment range.
fn anchored_scroll_value(old_content: f64, cursor_viewport: f64, k: f64) -> f64 {
    old_content * k - cursor_viewport
}

/// The cache key for one `(path, bucket)` pair — painter and producer share it,
/// so the two sides cannot drift into different spellings.
fn texture_key(path: &str, bucket: u32) -> String {
    format!("{}\u{0}{}", path, bucket)
}

/// Decode target for a needed pixel size: the smallest power-of-two bucket that
/// contains it, clamped to sane thumbnail limits. Continuous zoom then fires a
/// decode at most once per doubling instead of once per pixel.
fn texture_bucket(px: i32) -> u32 {
    for b in [128u32, 256, 512, 1024, 2048] {
        if px <= b as i32 {
            return b;
        }
    }
    2048
}

/// Rectangle of the contained image inside a box, centred. `None` for degenerate
/// inputs (undecoded dimensions arrive as 0; so do pre-allocation boxes).
fn contain_rect(src_w: i32, src_h: i32, box_w: i32, box_h: i32) -> Option<(i32, i32, i32, i32)> {
    let (fw, fh) = super::fit_inside(src_w, src_h, box_w, box_h)?;
    Some(((box_w - fw) / 2, (box_h - fh) / 2, fw, fh))
}

/// Keep-set of an LRU under a byte budget: walk `entries` newest-first and keep
/// until the budget is exhausted; everything older is evictable. The first entry
/// is always kept even when oversized — evicting everything would turn every
/// frame into a reload storm, and going over budget by one entry beats that.
fn evict_keep_set(entries: &[(String, u64)], budget: u64) -> Vec<String> {
    let mut kept = Vec::new();
    let mut used = 0u64;
    for (key, bytes) in entries {
        if used == 0 || used + bytes <= budget {
            kept.push(key.clone());
            used += bytes;
        }
    }
    kept
}

// ── Pixbuf cache ────────────────────────────────────────────────────────────

/// One decoded thumbnail. Cairo paints pixbufs directly
/// (`set_source_pixbuf`), so the cache holds them rather than GPU textures —
/// no second copy per entry.
type Pix = gtk4::gdk_pixbuf::Pixbuf;

struct PixbufCache {
    map: HashMap<String, (Pix, u64)>,
    /// Keys, most-recently-used first. Touched keys move to the front; inserts
    /// prepend.
    order: VecDeque<String>,
    bytes: u64,
}

impl PixbufCache {
    fn new() -> Self {
        Self { map: HashMap::new(), order: VecDeque::new(), bytes: 0 }
    }

    fn get(&mut self, key: &str) -> Option<Pix> {
        let (pix, _) = self.map.get(key)?;
        let pix = pix.clone();
        self.touch(key);
        Some(pix)
    }

    /// Mark `key` most-recently-used, if present.
    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
            self.order.push_front(key.to_string());
        }
    }

    fn insert(&mut self, key: String, pix: Pix, bytes: u64) {
        if let Some((_, old)) = self.map.remove(&key) {
            self.bytes -= old;
            self.order.retain(|k| k != &key);
        }
        self.map.insert(key.clone(), (pix, bytes));
        self.order.push_front(key);
        self.bytes += bytes;
        self.evict();
    }

    /// Drop least-recently-used entries until back under budget, via the shared
    /// pure policy ([`evict_keep_set`], whose never-empty rule is pinned by its
    /// test).
    fn evict(&mut self) {
        if self.bytes <= TEXTURE_BUDGET_BYTES {
            return;
        }
        let entries: Vec<(String, u64)> = self
            .order
            .iter()
            .filter_map(|k| self.map.get(k).map(|(_, b)| (k.clone(), *b)))
            .collect();
        let keep: HashSet<String> =
            evict_keep_set(&entries, TEXTURE_BUDGET_BYTES).into_iter().collect();
        let doomed: Vec<String> =
            self.order.iter().filter(|k| !keep.contains(*k)).cloned().collect();
        for key in doomed {
            if let Some((_, b)) = self.map.remove(&key) {
                self.bytes -= b;
            }
            self.order.retain(|k| k != &key);
        }
    }
}

// ── Widget side ─────────────────────────────────────────────────────────────

thread_local! {
    /// Current cell size in px (continuous). Session-only zoom state; see the
    /// module doc for why this isn't persisted.
    static ZOOM_CELL: Cell<f64> = const { Cell::new(DEFAULT_CELL_PX) };
    /// Last known viewport `(w, h)` of the canvas, from the resize callback.
    static ZOOM_VP: Cell<(i32, i32)> = const { Cell::new((0, 0)) };
    /// Weak handle to the drawing area, so the bottom-bar stepper — which holds
    /// no canvas reference — can trigger relayout + repaint. Every upgrade
    /// failure simply means "not on screen", mirroring the culling thread-local
    /// pattern.
    static ZOOM_AREA: glib::WeakRef<DrawingArea> = glib::WeakRef::new();
    static ZOOM_SCROLLER: glib::WeakRef<ScrolledWindow> = glib::WeakRef::new();
    /// Item-count mirror for [`relayout_only`], which runs outside the
    /// `CanvasState` closures (the stepper path). Written by the canvas whenever
    /// items change; a stale value costs at most one extra/missing row of
    /// size-request until the next sync — never a wrong paint (the draw func
    /// recomputes from the true list).
    static CURRENT_COUNT: Cell<usize> = const { Cell::new(0) };
    /// Generation guard for deferred adjustment writes (one per zoom step),
    /// mirroring the full preview's `zoom_gen`.
    static ZOOM_GEN: Cell<u64> = const { Cell::new(0) };
}

fn current_item_count() -> usize {
    CURRENT_COUNT.with(Cell::get)
}

/// Strong handle to the canvas's drawing area, if it still exists. The `.with`
/// dance is confined here so call sites read like plain upgrades.
fn zoom_area() -> Option<DrawingArea> {
    ZOOM_AREA.with(|w| w.upgrade())
}

fn zoom_scroller() -> Option<ScrolledWindow> {
    ZOOM_SCROLLER.with(|w| w.upgrade())
}

/// The double-click callback type. An alias because the bare form trips the
/// type-complexity lint at the field site.
type ActivateCb = Rc<dyn Fn(String)>;

/// One canvas item: the path plus its index in the FULL collection model, so a
/// click can select through the shared `SingleSelection` even though sentinel
/// rows were filtered out of the canvas.
struct ZoomItem {
    path: String,
    base_index: u32,
}

/// Shared state behind the canvas's closures.
struct CanvasState {
    items: RefCell<Vec<ZoomItem>>,
    textures: RefCell<PixbufCache>,
    /// Paths whose decode has failed once (raw formats pixbuf can't parse,
    /// unreadable files). Without this, every paint re-spawns the decode and a
    /// screenful of camera raws becomes a permanent read-parse-fail-redraw
    /// loop. Reset by [`CanvasState::sync_items`], so a genuine fix to the file
    /// (or collection change) retries.
    failed: RefCell<HashSet<String>>,
    /// Decodes in flight, `path → bucket`. A newer request supersedes an entry
    /// with a smaller bucket, so at most one decode per path ever runs.
    inflight: RefCell<HashMap<String, u32>>,
    /// Double-click callback (opens the darkroom page), wired by lib.rs.
    activate: RefCell<Option<ActivateCb>>,
    selection_w: glib::WeakRef<SingleSelection>,
}

impl CanvasState {
    /// Re-read the item list from the shared selection's model. Sentinel rows
    /// (no `/`) are skipped — see the module doc. Also refreshes the
    /// [`CURRENT_COUNT`] mirror the stepper-side relayout reads.
    fn sync_items(&self) {
        let paths = self
            .selection_w
            .upgrade()
            .and_then(|sel| sel.model())
            .map(|m| super::model_paths(&m))
            .unwrap_or_default();
        let items: Vec<ZoomItem> = paths
            .into_iter()
            .enumerate()
            .filter(|(_, p)| p.contains('/'))
            .map(|(i, path)| ZoomItem { path, base_index: i as u32 })
            .collect();
        CURRENT_COUNT.with(|c| c.set(items.len()));
        // A resync is also the retry opportunity for previously failed decodes:
        // the list changed, so "this path can't be decoded" is re-examined.
        self.failed.borrow_mut().clear();
        *self.items.borrow_mut() = items;
    }

    fn selected_base_index(&self) -> Option<u32> {
        self.selection_w.upgrade().map(|sel| sel.selected())
    }
}

/// The zoomable canvas: an overlay child covering the lighttable's centre slot
/// while the mode is active, plus the thread-local accessors the bottom-bar
/// stepper needs. Built once by [`super::lighttable_page`]; shown/hidden by
/// [`super::reconfigure_grid_for`].
pub struct ZoomableCanvas {
    pub(crate) layer: ScrolledWindow,
    /// The painted canvas itself — exposed so lib.rs can hang the lighttable
    /// key controller on it (it is a sibling of the grid, never its ancestor).
    area: DrawingArea,
    state: Rc<CanvasState>,
}

impl ZoomableCanvas {
    /// Build the canvas watching `selection`. Hidden until a mode switch reveals
    /// it; the caller parents `layer` as an overlay over the grid's scroller.
    pub(crate) fn new(selection: &SingleSelection) -> Self {
        let area = DrawingArea::new();
        area.set_hexpand(true);
        area.set_vexpand(true);
        // Focusable so clicks land keyboard focus here and the shared lighttable
        // key shortcuts keep working in this mode (the grid holds them for the
        // other layouts).
        area.set_focusable(true);
        // Same canvas greys as the grid (darktable's lighttable_bg_color).
        area.add_css_class("c41-lighttable-canvas");

        let scroller = ScrolledWindow::builder()
            .hscrollbar_policy(gtk4::PolicyType::Automatic)
            .vscrollbar_policy(gtk4::PolicyType::Automatic)
            .hexpand(true)
            .vexpand(true)
            .child(&area)
            .build();

        let state = Rc::new(CanvasState {
            items: RefCell::new(Vec::new()),
            textures: RefCell::new(PixbufCache::new()),
            failed: RefCell::new(HashSet::new()),
            inflight: RefCell::new(HashMap::new()),
            activate: RefCell::new(None),
            selection_w: selection.downgrade(),
        });
        state.sync_items();

        ZOOM_AREA.with(|w| w.set(Some(&area)));
        ZOOM_SCROLLER.with(|w| w.set(Some(&scroller)));

        let paint_state = Rc::clone(&state);
        area.set_draw_func(move |area, cr, _w, _h| {
            paint(&paint_state, area, cr);
        });

        wire_gestures(&area, &scroller, &state);
        wire_model_watch(&state, selection);

        Self { layer: scroller, area, state }
    }

    /// Wire the callback invoked on double-click. lib.rs passes the same
    /// open-the-darkroom-page body the grid's `activate` signal runs, so both
    /// paths stay behaviourally identical by construction.
    pub(crate) fn set_activate_callback(&self, cb: ActivateCb) {
        *self.state.activate.borrow_mut() = Some(cb);
    }

    /// Mode entry: resync from the model, size the plane, repaint. Cached
    /// pixbufs persist across mode switches deliberately — the cache outliving a
    /// visit is exactly what makes returning cheap.
    pub(crate) fn on_enter(&self) {
        self.state.sync_items();
        // A stale idle re-assert from a previous visit must not move this one.
        cancel_pending_zoom_assert();
        relayout_and_repaint();
        // Keyboard shortcuts live on the focused widget; entering the mode via
        // the switcher button would otherwise leave focus there and every
        // lighttable key inert until the first click (review finding).
        self.area.grab_focus();
    }
}

impl ZoomableCanvas {
    /// The drawing area, for attaching controllers beside the grid's.
    pub(crate) fn area(&self) -> &DrawingArea {
        &self.area
    }
}

/// Cheap handle copy: the layer is a GObject refcount and the state an Rc. The
/// switcher handlers each capture a clone so none of them can move the canvas
/// out from under `LighttablePage`.
impl Clone for ZoomableCanvas {
    fn clone(&self) -> Self {
        Self {
            layer: self.layer.clone(),
            area: self.area.clone(),
            state: Rc::clone(&self.state),
        }
    }
}

/// The frame painter. Geometry from the pure core; every pixbuf miss spawns at
/// most one decode. Runs on the whole visible band each frame.
fn paint(state: &Rc<CanvasState>, area: &DrawingArea, cr: &gtk4::cairo::Context) {
    let vp = ZOOM_VP.get();
    let cell = ZOOM_CELL.get();
    let n_items = state.items.borrow().len();
    let plane = plane_of(vp.0, n_items, cell);

    // Empty collection: centre a quiet message, mirroring the grid sentinel's
    // job in the other layouts.
    if n_items == 0 {
        draw_centered_message(area, cr, vp, "No images in this view");
        return;
    }

    let v_scroll =
        zoom_scroller().map(|s| s.vadjustment().value()).unwrap_or(0.0);

    let cell_i = cell.round().max(1.0) as i32;
    let pch = pitch(cell_i);
    let img_box_h = (cell_i - CAPTION_PX).max(1);
    let selected = state.selected_base_index();
    let fg = area.style_context().color();

    // Closed-form visible band (uniform grid ⇒ no scan): start at the row under
    // the scroll top and stop one row past the bottom edge. The per-item bounds
    // check stays as float-drift insurance at the two edges. Horizontal is laid
    // out fully by construction (cols derives from the viewport width); note the
    // pitch's +GAP can push plane_w slightly past the viewport at some zooms —
    // that just shows the h-scrollbar, nothing mispaints.
    let items = state.items.borrow();
    let n_items = items.len();
    let mut missing: Vec<(String, u32)> = Vec::new();
    if n_items > 0 {
        let pch_f = f64::from(pch);
        let cols = plane.cols;
        let first_row = (v_scroll / pch_f).floor().max(0.0) as u32;
        let mut idx = first_row.saturating_mul(cols);
        while idx < n_items as u32 {
            let (cx, cy) = cell_origin(&plane, idx);
            if cy > v_scroll + f64::from(vp.1) {
                break;
            }
            let item = &items[idx as usize];
            if cy + pch_f >= v_scroll {
                let bucket = texture_bucket(cell_i);
                match state.textures.borrow_mut().get(&texture_key(&item.path, bucket)) {
                    Some(pb) => blit_pixbuf(cr, &pb, cx, cy, cell_i, img_box_h),
                    None => missing.push((item.path.clone(), bucket)),
                }

                // Caption honours the overlay mode's filename switch — Hidden
                // hides captions here exactly as it does on the grid cells.
                if super::current_overlay_mode().shows_filenames() {
                    draw_caption(&fg, cr, cx, cy, cell_i, &item.path);
                }

                // The selection frame paints OVER the thumbnail: an image that
                // fills its contain-rect would otherwise hide three of the four
                // bars, and "what is selected" is this mode's core affordance.
                if selected == Some(item.base_index) {
                    draw_frame(cr, cx, cy, cell_i);
                }
            }
            idx += 1;
        }
    }
    drop(items);

    for (path, bucket) in missing {
        spawn_decode(state, area, &path, bucket);
    }
}

/// Paint `pix` aspect-fit inside the cell's image box (everything above the
/// caption band), centred. The clip guards the scale-blit from bleeding into
/// neighbouring cells at fractional scales.
fn blit_pixbuf(
    cr: &gtk4::cairo::Context,
    pix: &gtk4::gdk_pixbuf::Pixbuf,
    cx: f64,
    cy: f64,
    cell_i: i32,
    img_box_h: i32,
) {
    let Some((ix, iy, iw, ih)) =
        contain_rect(pix.width(), pix.height(), cell_i, img_box_h)
    else {
        return;
    };
    let dx = cx + f64::from(ix);
    let dy = cy + f64::from(iy);
    let sx = f64::from(iw).max(1.0) / f64::from(pix.width().max(1));
    let sy = f64::from(ih).max(1.0) / f64::from(pix.height().max(1));
    let _ = cr.save();
    cr.rectangle(dx, dy, f64::from(iw), f64::from(ih));
    cr.clip();
    cr.scale(sx.max(f64::MIN_POSITIVE), sy.max(f64::MIN_POSITIVE));
    cr.set_source_pixbuf(pix, dx / sx, dy / sy);
    let _ = cr.paint();
    let _ = cr.restore();
}

/// Click-to-select / double-click-to-open / drag-to-pan / wheel-to-zoom, all on
/// the area so their coordinates are plane(content) space directly.
fn wire_gestures(area: &DrawingArea, scroller: &ScrolledWindow, state: &Rc<CanvasState>) {
    // Pointer tracking for cursor-anchored zoom (the scroll controller carries
    // no coordinates; same sidecar pattern as the full preview).
    let last_pos: Rc<Cell<Option<(f64, f64)>>> = Rc::new(Cell::new(None));
    {
        let motion = EventControllerMotion::new();
        let lp = last_pos.clone();
        motion.connect_enter(move |_, x, y| lp.set(Some((x, y))));
        let lp = last_pos.clone();
        motion.connect_motion(move |_, x, y| lp.set(Some((x, y))));
        area.add_controller(motion);
    }

    // Wheel: one multiplicative stop per notch, anchored at the pointer.
    // DISCRETE keeps touchpad flicks out (the m4-133 lesson); Propagation::Stop
    // starves the scroller's native wheel handling — while over the canvas the
    // wheel belongs to the zoomer entirely. Zoom state is the thread-locals; no
    // CanvasState capture needed here.
    {
        let lp = last_pos.clone();
        let wheel = EventControllerScroll::new(
            EventControllerScrollFlags::VERTICAL | EventControllerScrollFlags::DISCRETE,
        );
        wheel.connect_scroll(move |_, _dx, dy| {
            let vp = ZOOM_VP.get();
            if vp.0 <= 0 {
                return Propagation::Stop;
            }
            let old = ZOOM_CELL.get();
            let next = zoom_step_cell(old, dy < 0.0, zoom_cell_max(vp.0, vp.1));
            if next != old {
                ZOOM_CELL.set(next);
                zoom_anchored_apply(old, next, lp.get());
            }
            Propagation::Stop
        });
        area.add_controller(wheel);
    }

    // Drag pans: adjustments track minus the drag delta, seeded from the values
    // at drag-begin.
    {
        let drag = GestureDrag::new();
        let begin_adj: Rc<RefCell<Option<(f64, f64)>>> = Rc::new(RefCell::new(None));
        {
            let begin_adj = begin_adj.clone();
            let sw = scroller.downgrade();
            drag.connect_drag_begin(move |_d, _x, _y| {
                *begin_adj.borrow_mut() = sw.upgrade()
                    .map(|s| (s.hadjustment().value(), s.vadjustment().value()));
            });
        }
        let begin_adj2 = begin_adj;
        let sw2 = scroller.downgrade();
        drag.connect_drag_update(move |_d, dx, dy| {
            if let (Some(s), Some((h0, v0))) = (sw2.upgrade(), begin_adj2.borrow().as_ref().copied()) {
                s.hadjustment().set_value(h0 - dx);
                s.vadjustment().set_value(v0 - dy);
            }
        });
        area.add_controller(drag);
    }

    // Click: single selects through the shared selection; double opens.
    {
        let st = Rc::clone(state);
        let click = GestureClick::new();
        click.set_button(gdk::BUTTON_PRIMARY);
        // `released` carries the click count directly — no need to query the
        // gesture mid-press.
        click.connect_released(move |_g, n_clicks, x, y| {
            let vp = ZOOM_VP.get();
            let items = st.items.borrow();
            let plane = plane_of(vp.0, items.len(), ZOOM_CELL.get());
            let Some(idx) = index_at_point(&plane, items.len(), x, y) else {
                return;
            };
            let item_path = items[idx as usize].path.clone();
            let base_index = items[idx as usize].base_index;
            drop(items);
            if n_clicks >= 2 {
                if let Some(cb) = st.activate.borrow().as_ref() {
                    cb(item_path);
                }
                return;
            }
            if let Some(sel) = st.selection_w.upgrade() {
                sel.set_selected(base_index);
            }
        });
        area.add_controller(click);
    }

    // Resize: remember the viewport, reflow the plane. Scroll position stays
    // clamped (GTK's own behaviour) rather than re-anchored — a window resize
    // keeping the top-left corner stable is viewer-normal.
    area.connect_resize(move |area, w, h| {
        let prev = ZOOM_VP.get();
        ZOOM_VP.set((w, h));
        if prev != (w, h) {
            relayout_only();
            area.queue_draw();
        }
    });
}

/// Follow collection changes while visible (import, filter, folder switch) and
/// repaint the selection border whenever the selection moves. Neither event may
/// cancel in-flight decodes: a membership change re-syncs the item LIST (decodes
/// keyed by path stay valid — stale arrivals just warm the cache), and a
/// selection change touches nothing but the frame.
fn wire_model_watch(state: &Rc<CanvasState>, selection: &SingleSelection) {
    // Selection moves touch only the frame — nothing but a repaint.
    selection.connect_selection_changed(move |_sel, _pos, _n| {
        if let Some(area) = zoom_area() {
            area.queue_draw();
        }
    });
    if let Some(model) = selection.model() {
        let st = Rc::clone(state);
        model.connect_items_changed(move |_m, _pos, _removed, _added| {
            st.sync_items();
            // `inflight` is deliberately NOT cleared here: entries are path-keyed
            // and removed by their own completions, so clearing would only let a
            // still-running decode respawn alongside itself on the next frame.
            relayout_only();
            if let Some(area) = zoom_area() {
                area.queue_draw();
            }
        });
    }
}

/// Write the plane extent into the area's size request (that IS the pan range)
/// and repaint. Used wherever the cell size or item count changed.
fn relayout_and_repaint() {
    relayout_only();
    if let Some(area) = zoom_area() {
        area.queue_draw();
    }
}

fn relayout_only() {
    let Some(area) = zoom_area() else { return };
    let plane = plane_of(ZOOM_VP.get().0, current_item_count(), ZOOM_CELL.get());
    area.set_size_request(plane.plane_w.max(1), plane.plane_h.max(1));
}

/// Invalidate any pending generation-guarded adjustment re-assert. Every writer
/// of the canonical zoom state calls this — otherwise a wheel-step's idle could
/// fire after a later stepper/enter relayout and drag the view to an offset
/// computed for geometry that no longer exists.
fn cancel_pending_zoom_assert() {
    let gen = ZOOM_GEN.with(|g| g.get().wrapping_add(1));
    ZOOM_GEN.with(|g| g.set(gen));
}

/// Apply a zoom step around the cursor: resize the plane and write both
/// adjustments so the plane point under the pointer stays put. Immediate write
/// plus one generation-guarded idle re-assert — the exact discipline the full
/// preview documents (`ZoomState::apply`), for the identical reason: uppers
/// describe the new range only after GTK reallocates.
fn zoom_anchored_apply(old_cell: f64, new_cell: f64, anchor: Option<(f64, f64)>) {
    let Some(scroller) = zoom_scroller() else { return };
    let Some(area) = zoom_area() else { return };
    relayout_only();

    let h = scroller.hadjustment();
    let v = scroller.vadjustment();
    let k = new_cell / old_cell;
    // Anchor defaults to the viewport centre (first wheel before any pointer
    // motion landed). Anchor coordinates are area-local == content space (the
    // area IS the plane-sized child); viewport space subtracts the scroll.
    let vp = ZOOM_VP.get();
    let (ax, ay) = anchor.unwrap_or((f64::from(vp.0) / 2.0, f64::from(vp.1) / 2.0));
    let cur_vp = (ax - h.value(), ay - v.value());
    let nh = anchored_scroll_value(ax, cur_vp.0, k);
    let nv = anchored_scroll_value(ay, cur_vp.1, k);
    h.set_value(nh);
    v.set_value(nv);

    cancel_pending_zoom_assert();
    let gen = ZOOM_GEN.with(|g| g.get());
    let sw = scroller.downgrade();
    glib::idle_add_local_once(move || {
        if ZOOM_GEN.with(|g| g.get()) != gen {
            return;
        }
        if let Some(s) = sw.upgrade() {
            s.hadjustment().set_value(nh);
            s.vadjustment().set_value(nv);
        }
    });
    area.queue_draw();
}

/// Async decode for one `(path, bucket)` miss. At most one decode per
/// `(path, bucket)` ever runs; a newer LARGER bucket request supersedes a
/// still-running smaller one rather than waiting for it (the loser's result is
/// still cached when it lands — both keys are useful at different zooms).
/// Completions are NOT generation-gated: a decode that lands after a resync is
/// still a valid cache entry — worst case it warms a thumbnail nobody currently
/// shows, which the byte-budget eviction absorbs.
///
/// Only the file READ runs on a worker thread — the loader/scale stay on the
/// main thread, the same split the file-manager cells use (GObjects are not
/// Send, and this keeps it that way). The loader is told the target size up
/// front (`connect_size_prepared`, the m4-132 full-preview lesson) so a large
/// JPEG decodes once at thumbnail scale instead of materialising full size just
/// to be scaled down — and because misses spawn concurrently, per-decode spikes
/// would otherwise multiply across a screenful of large images.
fn spawn_decode(state: &Rc<CanvasState>, area: &DrawingArea, path: &str, bucket: u32) {
    {
        let mut inflight = state.inflight.borrow_mut();
        if state.failed.borrow().contains(path) {
            return; // known undecodable until the next resync
        }
        if let Some(prev) = inflight.get(path) {
            if *prev >= bucket {
                return; // equal-or-bigger decode already running
            }
        }
        inflight.insert(path.to_string(), bucket);
    }

    let path_owned = path.to_string();
    let st = Rc::clone(state);
    let area_w = area.downgrade();
    glib::spawn_future_local(async move {
        let p = path_owned.clone();
        let bytes = gio::spawn_blocking(move || std::fs::read(&p).ok())
            .await
            .ok()
            .flatten();

        let mut decoded = None;
        if let Some(data) = &bytes {
            let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
            loader.connect_size_prepared(move |loader, w, h| {
                let longest = w.max(h);
                if longest > bucket as i32 {
                    // One scale factor on both axes: `set_size` does NOT
                    // preserve aspect ratio for you.
                    let scale = f64::from(bucket as i32) / f64::from(longest);
                    loader.set_size(
                        ((f64::from(w) * scale) as i32).max(1),
                        ((f64::from(h) * scale) as i32).max(1),
                    );
                }
            });
            // Both unconditional: a loader finalized without `close()` emits a
            // g_warning, so an early return on a rejected header would print one
            // per retry.
            let _ = loader.write(data);
            let _ = loader.close();
            decoded = loader.pixbuf();
        }

        match decoded {
            Some(raw) => {
                if let Some((fw, fh)) =
                    super::fit_inside(raw.width(), raw.height(), bucket as i32, bucket as i32)
                {
                    if let Some(pb) =
                        raw.scale_simple(fw, fh, gtk4::gdk_pixbuf::InterpType::Bilinear)
                    {
                        let n_bytes = i64::from(pb.width()) * i64::from(pb.height())
                            * i64::from(pb.n_channels());
                        st.textures
                            .borrow_mut()
                            .insert(texture_key(&path_owned, bucket), pb, n_bytes as u64);
                    }
                }
            }
            None => {
                // Negative-cache the failure (review CRITICAL): without it every
                // frame re-reads and re-fails on undecodable files forever.
                st.failed.borrow_mut().insert(path_owned.clone());
            }
        }
        // Superseded bookkeeping: remove only if THIS decode is still the
        // registered one — an in-flight larger request owns the slot now.
        let mut inflight = st.inflight.borrow_mut();
        if inflight.get(&path_owned).copied() == Some(bucket) {
            inflight.remove(&path_owned);
        }
        drop(inflight);
        if let Some(a) = area_w.upgrade() {
            a.queue_draw();
        }
    });
}

// ── Bottom-bar stepper bridge ───────────────────────────────────────────────

/// What the thumb-size stepper shows and allows while zoomable is active:
/// `(images per row right now, lowest, highest)`. The range matches what the
/// wheel itself can reach — `CELL_MIN_PX` sets the true upper column count — so
/// the buttons stay sensitive exactly as long as a step can change something,
/// even after wheeling out past the file-manager's 12-column habit. `None`
/// before the first allocation (no meaningful column count yet) — the caller
/// falls back to its own range, exactly like the culling arm does
/// pre-allocation.
pub(crate) fn stepper_state() -> Option<(u32, u32, u32)> {
    let vp = ZOOM_VP.get();
    if vp.0 <= 0 {
        return None;
    }
    let cols = zoom_columns(vp.0, ZOOM_CELL.get().round() as i32);
    Some((cols, 1, zoom_columns(vp.0, CELL_MIN_PX)))
}

/// One stepper click: `up` means more images per row (`zoom_in`, "smaller
/// thumbnails" — the SAME convention the file-manager and culling arms follow),
/// `down` fewer. Writes the same thread-local cell size the wheel drives — one
/// canonical zoom state, two controls. At either end the target cell equals the
/// current one and the write degenerates to a no-op.
pub(crate) fn stepper_step(up: bool) {
    let vp = ZOOM_VP.get();
    if vp.0 <= 0 {
        return;
    }
    let cols_now = zoom_columns(vp.0, ZOOM_CELL.get().round() as i32);
    let target = if up {
        cols_now.saturating_add(1)
    } else {
        cols_now.saturating_sub(1)
    };
    let next = zoom_columns_to_cell(vp.0, vp.1, target);
    if next != ZOOM_CELL.get() {
        ZOOM_CELL.set(next);
        cancel_pending_zoom_assert();
        relayout_and_repaint();
    }
}

// ── Small draw helpers ──────────────────────────────────────────────────────

/// Text on the canvas uses cairo's toy font API — the same idiom as the bauhaus
/// sliders — because pango-layout-on-cairo would drag in pangocairo for two
/// strings a frame.
fn set_canvas_font(cr: &gtk4::cairo::Context, size: f64) {
    cr.select_font_face(
        "sans-serif",
        gtk4::cairo::FontSlant::Normal,
        gtk4::cairo::FontWeight::Normal,
    );
    cr.set_font_size(size);
}

/// Truncate `text` by chars until it measures within `max_w`, appending an
/// ellipsis when anything was cut. UTF-8 safe (char boundaries only).
fn fit_text(cr: &gtk4::cairo::Context, text: &str, max_w: f64) -> String {
    let advance = |t: &str| cr.text_extents(t).map(|e| e.x_advance()).unwrap_or(0.0);
    if advance(text) <= max_w {
        return text.to_string();
    }
    let mut n = text.chars().count();
    while n > 1 {
        n -= 1;
        let acc = text.chars().take(n).collect::<String>() + "\u{2026}";
        if advance(&acc) <= max_w {
            return acc;
        }
    }
    // One char still too wide: give up on the ellipsis and clip.
    text.chars().take(1).collect()
}

fn draw_centered_message(
    area: &DrawingArea,
    cr: &gtk4::cairo::Context,
    vp: (i32, i32),
    text: &str,
) {
    if vp.0 <= 0 || vp.1 <= 0 {
        return;
    }
    set_canvas_font(cr, 14.0);
    // Theme foreground, not a hardcoded grey — same policy as bauhaus labels.
    let fg = area.style_context().color();
    cr.set_source_rgba(fg.red() as f64, fg.green() as f64, fg.blue() as f64, 0.75);
    if let Ok(ext) = cr.text_extents(text) {
        let x = (f64::from(vp.0) - ext.width()) / 2.0;
        let y = (f64::from(vp.1) - ext.height()) / 2.0;
        cr.move_to(x - ext.x_bearing(), y - ext.y_bearing());
        let _ = cr.show_text(text);
    }
}

/// Accent selection frame: four thin bars along the cell's edges.
fn draw_frame(cr: &gtk4::cairo::Context, x: f64, y: f64, cell: i32) {
    let t = 2.0;
    cr.set_source_rgba(0.85, 0.55, 0.10, 1.0);
    let outer = f64::from(cell);
    for (rx, ry, rw, rh) in [
        (x, y, outer, t),
        (x, y + outer - t, outer, t),
        (x, y, t, outer),
        (x + outer - t, y, t, outer),
    ] {
        cr.rectangle(rx, ry, rw, rh);
    }
    let _ = cr.fill();
}

/// Filename over the cell's bottom band, dimmed backdrop for legibility.
fn draw_caption(
    fg: &gdk::RGBA,
    cr: &gtk4::cairo::Context,
    x: f64,
    y: f64,
    cell: i32,
    path: &str,
) {
    let name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);
    let band_y = y + f64::from(cell - CAPTION_PX);
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.55);
    cr.rectangle(x, band_y, f64::from(cell), f64::from(CAPTION_PX));
    let _ = cr.fill();

    set_canvas_font(cr, 10.0);
    cr.set_source_rgba(fg.red() as f64, fg.green() as f64, fg.blue() as f64, 0.92);
    let shown = fit_text(cr, name, f64::from(cell - 8));
    if let Ok(ext) = cr.text_extents(&shown) {
        let tx = x + (f64::from(cell) - ext.x_advance()) / 2.0;
        let ty =
            band_y + f64::from(CAPTION_PX / 2) - ext.height() / 2.0 - ext.y_bearing();
        cr.move_to(tx, ty);
        let _ = cr.show_text(&shown);
    }
}

// ── Tests (display-free, per repo discipline) ───────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_never_reach_zero_even_when_the_cell_overflows_the_viewport() {
        assert_eq!(zoom_columns(909, 192), 4);
        assert_eq!(zoom_columns(400, 192), 2);
        // Cell wider than the viewport: one per row, clipped, never zero.
        assert_eq!(zoom_columns(100, 192), 1);
        // Degenerate inputs collapse to the same safe answer.
        assert_eq!(zoom_columns(0, 192), 1);
        assert_eq!(zoom_columns(-50, 192), 1);
        assert_eq!(zoom_columns(909, 0), 1);
        assert_eq!(zoom_columns(909, -3), 1);
    }

    #[test]
    fn plane_rows_are_ceiled_and_empty_collections_yield_a_zero_plane() {
        let p = plane_of(1000, 10, 200.0);
        assert_eq!(p.cols, 5);
        assert_eq!(p.rows, 2);
        assert_eq!(p.plane_w, 5 * (200 + CELL_GAP_PX));
        assert_eq!(p.plane_h, 2 * (200 + CELL_GAP_PX));

        // A remainder row exists: 11 items over 5 columns is 3 rows.
        assert_eq!(plane_of(1000, 11, 200.0).rows, 3);

        // Zero items: zero extent — the empty message paints in the viewport.
        let e = plane_of(1000, 0, 200.0);
        assert_eq!(e.rows, 0);
        assert_eq!(e.plane_h, 0);
        assert_eq!(e.plane_w, 0);

        // Degenerate viewport still produces a valid one-column plane.
        let d = plane_of(0, 5, 200.0);
        assert_eq!(d.cols, 1);
        assert_eq!(d.rows, 5);
    }

    #[test]
    fn index_at_point_round_trips_through_cell_origin_and_skips_gaps() {
        let p = plane_of(1000, 12, 200.0);
        for idx in [0u32, 4, 5, 11] {
            let (x, y) = cell_origin(&p, idx);
            assert_eq!(index_at_point(&p, 12, x + 1.0, y + 1.0), Some(idx));
        }
        // Past the last item (row 2 holds 2 of 5): inside the plane rect, no item.
        assert_eq!(index_at_point(&p, 12, 10.0, 900.0), None);
        // Negative coordinates and inter-cell gaps answer None.
        assert_eq!(index_at_point(&p, 12, -1.0, 10.0), None);
        assert_eq!(index_at_point(&p, 12, 205.0, 205.0), None); // 200..208 is gap
        // Empty collection: nothing anywhere.
        let e = plane_of(1000, 0, 200.0);
        assert_eq!(index_at_point(&e, 0, 10.0, 10.0), None);
    }

    #[test]
    fn wheel_steps_are_multiplicative_clamped_and_pinned_at_the_ends() {
        let max = zoom_cell_max(1200, 900);
        assert_eq!(max, 1200);
        let mut c = 200.0;
        for _ in 0..40 {
            c = zoom_step_cell(c, true, max);
        }
        assert_eq!(c, f64::from(max), "zoom-in converges to exactly the cap");
        for _ in 0..60 {
            c = zoom_step_cell(c, false, max);
        }
        assert_eq!(c, f64::from(CELL_MIN_PX), "zoom-out converges to exactly the floor");
        // Symmetric around the middle: one in then one out restores (within fp).
        let mid = zoom_step_cell(zoom_step_cell(300.0, true, max), false, max);
        assert!((mid - 300.0).abs() < 1e-9);
        // Degenerate max collapses onto the floor rather than inverting the range.
        assert_eq!(zoom_step_cell(200.0, true, 0), f64::from(CELL_MIN_PX));
    }

    #[test]
    fn columns_to_cell_maps_back_through_zoom_columns() {
        // 909px viewport, want 4 columns ⇒ ~227px cells ⇒ zoom_columns agrees.
        let cell = zoom_columns_to_cell(909, 700, 4);
        assert_eq!(zoom_columns(909, cell.round() as i32), 4);
        // Column counts beyond what CELL_MIN affords clamp onto it.
        let tiny = zoom_columns_to_cell(909, 700, 10_000);
        assert!((tiny - f64::from(CELL_MIN_PX)).abs() < 1e-9);
        // Degenerate width (pre-allocation): raw cell 1px clamps onto CELL_MIN.
        let big = zoom_columns_to_cell(0, 500, 1);
        assert!((big - f64::from(CELL_MIN_PX)).abs() < 1e-9);
    }

    #[test]
    fn texture_buckets_are_powers_of_two_with_hard_limits() {
        assert_eq!(texture_bucket(1), 128);
        assert_eq!(texture_bucket(127), 128);
        assert_eq!(texture_bucket(128), 128);
        assert_eq!(texture_bucket(129), 256);
        assert_eq!(texture_bucket(600), 1024);
        assert_eq!(texture_bucket(2048), 2048);
        assert_eq!(texture_bucket(100_000), 2048, "huge requests cap at the top bucket");
        assert_eq!(texture_bucket(0), 128);
        assert_eq!(texture_bucket(-5), 128);
    }

    #[test]
    fn anchored_scroll_keeps_the_cursor_point_stationary() {
        // Content point 300 under a cursor at viewport 100 ⇒ the current scroll
        // value IS 200 (value + cursor == content point). Identity zoom keeps it.
        assert!((anchored_scroll_value(300.0, 100.0, 1.0) - 200.0).abs() < 1e-9);
        // Doubling: content point 300 under a cursor at viewport 100 ⇒ the new
        // value puts 600 back under the same screen spot.
        assert!((anchored_scroll_value(300.0, 100.0, 2.0) - 500.0).abs() < 1e-9);
        // Shrinking to a quarter clamps to the top-left corner region.
        assert!((anchored_scroll_value(800.0, 200.0, 0.25) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn contain_rect_centres_and_rejects_degenerate_sources() {
        // Square source scales to the limiting height (184) and centres in both
        // axes: x=(200−184)/2=8, y=(184−184)/2=0.
        assert_eq!(contain_rect(100, 100, 200, 184), Some((8, 0, 184, 184)));
        // Landscape centres horizontally within the box.
        let (x, _, w, _) = contain_rect(300, 200, 200, 184).unwrap();
        assert_eq!((x + w) / 2, 100);
        assert_eq!(contain_rect(0, 100, 200, 200), None);
        assert_eq!(contain_rect(100, 0, 200, 200), None);
        assert_eq!(contain_rect(100, 100, 0, 200), None);
    }

    #[test]
    fn evict_keep_set_respects_the_budget_but_never_empties() {
        let k = |n: &str| n.to_string();
        let entries = [(k("newest"), 40u64), (k("mid"), 40), (k("oldest"), 40)];
        // Budget fits two: the oldest goes.
        assert_eq!(evict_keep_set(&entries, 80), vec!["newest", "mid"]);
        // Budget fits everything.
        assert_eq!(evict_keep_set(&entries, 120).len(), 3);
        // Budget fits nothing: keep exactly the MRU entry anyway — an emptied
        // cache would turn every frame into a reload storm.
        assert_eq!(evict_keep_set(&entries, 10), vec!["newest"]);
        assert!(evict_keep_set(&[], 100).is_empty());
    }

    #[test]
    fn zoom_cell_max_tracks_the_longer_axis_within_sane_bounds() {
        assert_eq!(zoom_cell_max(1200, 900), 1200);
        assert_eq!(zoom_cell_max(900, 1200), 1200);
        assert_eq!(zoom_cell_max(0, 0), CELL_MIN_PX, "degenerate viewport floors");
        assert_eq!(zoom_cell_max(10_000_000, 10_000_000), 4096, "absurd viewports cap");
    }

    #[test]
    fn plane_pitch_is_atomic_across_origin_and_hit_test_math() {
        // The pitch behind cell_origin must equal the one index_at_point derives
        // from the plane, or clicks land one gap off from what paint drew. Dense
        // round-trip sweep pins the invariant.
        let p = plane_of(1000, 25, 150.0);
        for idx in 0..25u32 {
            let (x, y) = cell_origin(&p, idx);
            assert_eq!(index_at_point(&p, 25, x + 3.0, y + 3.0), Some(idx), "at {idx}");
        }
    }
}
