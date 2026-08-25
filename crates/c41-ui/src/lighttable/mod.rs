//! Lighttable view -- async thumbnail grid + star ratings + colour labels.
//!
//! Phase 3-ui-8: each cell has a 5-star rating row that reads/writes
//! the rating from/to c41-db asynchronously.
//! Phase 3-m4-20: each cell also has a 5-dot colour-label row (red/yellow/
//! green/blue/purple); clicking a dot toggles that label via the
//! `c41_db::colorlabels` DAO, resolving the image id by path.

pub mod full_preview;
pub mod rule_stack;
pub(crate) mod selection;
pub mod timeline;
pub mod zoomable;

mod thumbs;

use adw::prelude::*;
use gtk4::{GridView, ListItem, ScrolledWindow, SignalListItemFactory, SingleSelection};
use glib::clone;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

pub const THUMB_SIZE: i32 = 160;

/// The grid's lower column bound in the file-manager layout — low enough that a
/// narrow framebuffer fits a row instead of clipping it. Culling raises it to pin
/// exactly one row of the comparison window, and restores this on the way out, so
/// both places name the same constant rather than repeating a literal.
const GRID_MIN_COLUMNS: u32 = 2;

/// The thumb-size stepper's starting value (m4-98a). Declared here, not in
/// `lib.rs`, because [`leave_culling`] needs it as the restore fallback — one
/// definition can't drift into two meanings.
pub const THUMB_COLS_DEFAULT: u32 = 6;

pub type LighttableModel = gtk4::StringList;

/// The lighttable's widgets, as built by [`lighttable_page`].
///
/// The `grid` is handed out rather than left to be re-derived with
/// `scroll.child().and_downcast::<GridView>()`: every such downcast is a control
/// that silently goes inert (no error, no log) the day the scroller's child
/// changes. Returning it makes "the bottom-bar controls always find the grid" hold
/// by construction — see the m4-98c constraint on [`ViewMode`].
pub struct LighttablePage {
    /// The centre slot's root: an `Overlay` holding the grid scroller as its
    /// child and the zoomable canvas as a hidden overlay child above it (m4-139).
    /// lib.rs hands THIS to [`full_preview::FullPreview::wrap`], so the preview
    /// layer still covers every layout; the grid scroller itself never changes
    /// child — the m4-98c constraint is untouched, the canvas simply sits beside
    /// it inside one overlay.
    pub overlay: gtk4::Overlay,
    pub scroll: ScrolledWindow,
    pub grid: GridView,
    pub model: LighttableModel,
    pub selection: SingleSelection,
    /// The zoomable-mode canvas. Hidden until [`set_view_mode`] enters
    /// [`ViewMode::Zoomable`]; the bottom-bar stepper reaches its zoom through
    /// the thread-locals in `lighttable/zoomable.rs`.
    pub zoom: zoomable::ZoomableCanvas,
}

/// Build the lighttable widget — see [`LighttablePage`] for what comes back.
///
/// `db_path` is stored in each cell's gesture handler for rating updates.
pub fn lighttable_page(db_path: String) -> LighttablePage {
    let model     = gtk4::StringList::new(&[]);
    let selection = SingleSelection::new(Some(model.clone()));
    let factory   = SignalListItemFactory::new();

    // ── Setup: widget tree for each visible cell ───────────────────────────
    factory.connect_setup(|_, list_item| {
        let item = list_item.downcast_ref::<ListItem>().unwrap();

        let vbox = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(2)
            .build();

        let thumb = gtk4::Picture::builder()
            .width_request(THUMB_SIZE)
            .height_request(THUMB_SIZE)
            .content_fit(gtk4::ContentFit::Cover)
            .build();
        thumb.add_css_class("frame");

        let label = gtk4::Label::builder()
            .max_width_chars(16)
            .ellipsize(gtk4::pango::EllipsizeMode::End)
            .build();
        label.add_css_class("caption");

        // Star rating row: 5 Labels "★"
        let stars_box = build_stars_box();

        // Colour-label row: 5 filled-circle Labels, coloured via Pango markup.
        let colors_box = build_color_dots_box();

        vbox.append(&thumb);
        vbox.append(&label);
        vbox.append(&stars_box);
        vbox.append(&colors_box);
        item.set_child(Some(&vbox));

        // Selection clicks (m4-144): plain = exclusive, ctrl = toggle, shift =
        // range from the anchor. The handler reads the thumb's stamped widget
        // name at event time, so a recycled cell always acts on its CURRENT
        // image — no stale-path closure to strip on rebind. The gesture only
        // observes: it never claims the sequence, so GridView's own handling
        // (cursor move; double-click activation into darkroom) proceeds.
        //
        // Primary button ONLY (senior review MINOR-5): GridView's native
        // gestures are primary-only, so without this a right-click would
        // silently rewrite the selection; darktable reserves it for a context
        // menu. Acted on RELEASE, like the zoomable canvas's click handler
        // (MINOR-7): a touch/touchpad pan starting on a cell is recognised as
        // a drag before release and never mutates the selection at press time.
        let sel_click = gtk4::GestureClick::new();
        sel_click.set_button(gtk4::gdk::BUTTON_PRIMARY);
        sel_click.connect_released(|gesture, n_press, _x, _y| {
            if n_press != 1 {
                return; // multi-press belongs to activate (open darkroom)
            }
            let Some(w) = gesture.widget() else { return };
            let Some(thumb) = w.downcast::<gtk4::Picture>().ok() else { return };
            let path = thumb.widget_name().to_string();
            if !path.contains('/') {
                return; // placeholder rows are not selectable
            }
            let state = gesture.current_event_state();
            let ctrl = state.contains(gtk4::gdk::ModifierType::CONTROL_MASK);
            let shift = state.contains(gtk4::gdk::ModifierType::SHIFT_MASK);
            if shift {
                selection::range_from_anchor(&selection::current_view_paths(), &path);
            } else if ctrl {
                selection::toggle(&path);
            } else {
                selection::select_exclusive(&path);
            }
            selection::notify_changed();
        });
        thumb.add_controller(sel_click);
    });

    // ── Bind: fill data + start async loads ────────────────────────────────
    let db_for_bind = db_path.clone();
    factory.connect_bind(move |_, list_item| {
        let item       = list_item.downcast_ref::<ListItem>().unwrap();
        let vbox       = item.child().and_downcast::<gtk4::Box>().unwrap();
        let thumb      = vbox.first_child().and_downcast::<gtk4::Picture>().unwrap();
        let label      = nth_child(&vbox, 1).and_downcast::<gtk4::Label>()
            .unwrap_or_else(|| gtk4::Label::new(None));
        let stars_box  = nth_child(&vbox, 2).and_downcast::<gtk4::Box>()
            .unwrap_or_else(|| gtk4::Box::new(gtk4::Orientation::Horizontal, 0));
        let colors_box = nth_child(&vbox, 3).and_downcast::<gtk4::Box>()
            .unwrap_or_else(|| gtk4::Box::new(gtk4::Orientation::Horizontal, 0));

        let string_obj = item.item().and_downcast::<gtk4::StringObject>().unwrap();
        let full_path  = string_obj.string().to_string();

        let filename = std::path::Path::new(&full_path)
            .file_name().and_then(|n| n.to_str()).unwrap_or(&full_path).to_string();
        label.set_label(&filename);
        // Size + content fit BEFORE painting: cells are recycled, so the requests
        // a previous bind left behind may belong to the other layout (THUMB_SIZE
        // square in the file manager vs viewport-filling box in culling).
        apply_cell_size(&thumb);
        thumb.set_paintable(gtk4::gdk::Paintable::NONE);

        // Stamp each async-painted widget with the identity it's now bound to,
        // UNCONDITIONALLY — including placeholder rows. GTK recycles cells: a cell
        // can be rebound (even to a placeholder) while an earlier async read for
        // the PREVIOUS path is still in flight. Each task re-checks the widget's
        // name on resolve and bails if it no longer matches, so a slow read can't
        // smear image A onto image B — nor re-wire an A-path gesture onto a cell
        // that has since become a placeholder. Stamping must precede the
        // placeholder early-return below, or the stale name would survive and the
        // in-flight read would still match and paint/wire.
        thumb.set_widget_name(&full_path);
        stars_box.set_widget_name(&full_path);
        colors_box.set_widget_name(&full_path);

        // Honour the overlay mode (m4-98e) on EVERY bind path, before any early
        // return: cells are recycled, so a mode changed while this cell was
        // off-screen (or while it showed a placeholder) must take effect as it
        // comes back. `set_overlay_mode` only covers the cells realized at the time
        // it ran, so this unconditional call is what makes the pair exhaustive.
        let is_placeholder = !full_path.contains('/');
        apply_overlay_visibility(
            &vbox,
            effective_overlay_visibility(is_placeholder, current_overlay_mode()),
        );
        // Selection frame (m4-144): re-applied on EVERY bind — cells recycle,
        // so a class a previous image left behind must be re-decided here, and
        // this is also what repaints off-screen-selected cells when they
        // scroll back in.
        apply_selection_frame(&thumb, selection::contains(&full_path));

        if is_placeholder {
            set_stars(&stars_box, 0);
            set_color_dots(&colors_box, 0);
            return;
        }

        // The rating/colour-label reads below run REGARDLESS of the overlay mode,
        // and that is load-bearing: [`set_overlay_mode`] only toggles `visible`, it
        // never populates. Skipping the queries in `Hidden`/`Normal` as an
        // "optimisation" would make a later switch back to `Extended` reveal
        // permanently empty rows on every already-bound cell.
        //
        // Async thumbnail load (m4-140): through the shared thumbnail service —
        // one decoder + caches behind both this grid and the zoomable canvas.
        // Camera raws decode now instead of leaving blank cells, cache hits
        // paint synchronously with no thread hop, and undecodable files cost
        // one attempt per collection load (the shared negative cache).
        ensure_grid_thumb(thumb.downgrade(), full_path.clone());

        // Async rating load
        let db = db_for_bind.clone();
        let fp = string_obj.string().to_string();
        glib::spawn_future_local(clone!(@weak stars_box => async move {
            let path = fp.clone();
            let db2  = db.clone();
            let rating = gio::spawn_blocking(move || query_rating(&path, &db2))
                .await.ok().flatten().unwrap_or(0);
            if stars_box.widget_name() != fp { return; } // cell rebound mid-read
            set_stars(&stars_box, rating);

            // Re-wire the star click handlers for THIS bind. bind() fires on every
            // cell recycle, so wire_star_clicks strips the prior gestures first —
            // otherwise a recycled cell accumulates one stale-path gesture per bind.
            wire_star_clicks(&stars_box, fp, db);
        }));

        // Async colour-label load + click wiring (mirrors the rating block).
        let db_c = db_for_bind.clone();
        let fp_c = string_obj.string().to_string();
        glib::spawn_future_local(clone!(@weak colors_box => async move {
            let path = fp_c.clone();
            let db2  = db_c.clone();
            let mask = gio::spawn_blocking(move || query_color_labels(&path, &db2))
                .await.unwrap_or(0);
            if colors_box.widget_name() != fp_c { return; } // cell rebound mid-read
            set_color_dots(&colors_box, mask);
            wire_color_clicks(&colors_box, fp_c, db_c);
        }));
    });

    // ── Unbind ─────────────────────────────────────────────────────────────
    // Deliberately does NOT reset row visibility (m4-98e): bind re-establishes it
    // on every path, and leaving it (plus the stamped widget names) means an
    // unbound-but-still-parented cell is still classified correctly by
    // [`set_overlay_mode`]'s walk.
    factory.connect_unbind(|_, list_item| {
        let item = list_item.downcast_ref::<ListItem>().unwrap();
        if let Some(vbox) = item.child().and_downcast::<gtk4::Box>() {
            if let Some(thumb) = vbox.first_child().and_downcast::<gtk4::Picture>() {
                thumb.set_paintable(gtk4::gdk::Paintable::NONE);
            }
            if let Some(stars) = nth_child(&vbox, 2).and_downcast::<gtk4::Box>() {
                set_stars(&stars, 0);
            }
            if let Some(colors) = nth_child(&vbox, 3).and_downcast::<gtk4::Box>() {
                set_color_dots(&colors, 0);
            }
        }
    });

    let grid = GridView::builder()
        .model(&selection)
        .factory(&factory)
        // The *max*-column bound (thumbnail size) is owned by the bottom toolbar's
        // thumb-size stepper (see lib.rs THUMB_COLS_*); it calls set_max_columns
        // on this grid at startup and on every ± click. Only the min bound lives
        // here, and culling raises it to pin one row (see `enter_culling`).
        // `GRID_MIN_COLUMNS` keeps a narrow framebuffer from clipping the row.
        .min_columns(GRID_MIN_COLUMNS)
        .build();
    // Slightly darker than the darkroom canvas, matching darktable's
    // lighttable_bg_color (grey_40) — see `crate::theme`.
    grid.add_css_class("c41-lighttable-canvas");
    grid.add_css_class("lighttable-grid");
    // Multi-selection state's handle on the grid (m4-144): lets the parameter-
    // free `selection::notify_changed()` repaint realized frames from anywhere.
    selection::set_grid(&grid);

    let scroll = ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .child(&grid)
        .vexpand(true)
        .build();

    // Zoomable mode's canvas (m4-139): a hidden overlay sibling of the grid
    // scroller. The overlay is the centre slot; lib.rs wraps it with the full
    // preview exactly as it used to wrap the bare scroller.
    let zoom = zoomable::ZoomableCanvas::new(&selection);
    // The canvas paints selection frames lazily from the shared set (m4-144),
    // so set-only mutations — Esc, ctrl+A, a click in the OTHER surface — must
    // queue it a redraw or stale frames linger until the next scroll/zoom/
    // click (senior review MINOR-6).
    selection::set_canvas(zoom.area());
    let overlay = gtk4::Overlay::new();
    overlay.set_child(Some(&scroll));
    overlay.add_overlay(&zoom.layer);
    zoom.layer.set_visible(false);
    overlay.set_hexpand(true);
    overlay.set_vexpand(true);

    LighttablePage { overlay, scroll, grid, model, selection, zoom }
}

// ── Rating helpers ────────────────────────────────────────────────────────

fn nth_child(b: &gtk4::Box, n: usize) -> Option<gtk4::Widget> {
    let mut child = b.first_child();
    for _ in 0..n {
        child = child.and_then(|w| w.next_sibling());
    }
    child
}

/// Build a fresh star-rating row: 5 unlit `★` `Label`s in a centred horizontal
/// box. One source of truth for the star-row layout shared by the lighttable grid
/// cells and the darkroom header (m4-28); the caller seeds the lit count via
/// [`set_stars`] and wires click-to-rate via [`wire_star_clicks`]. `pub(crate)`
/// so the darkroom view reuses the exact construction (mirrors `build_color_dots_box`).
pub(crate) fn build_stars_box() -> gtk4::Box {
    let stars_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(1)
        .halign(gtk4::Align::Center)
        .build();
    // The stars are pure glyphs; give the row a group role + name so assistive
    // tech announces it in both the grid cells and the darkroom header.
    stars_box.set_accessible_role(gtk4::AccessibleRole::Group);
    stars_box.update_property(&[gtk4::accessible::Property::Label("Star rating")]);
    for _ in 0..5 {
        let star = gtk4::Label::new(Some("\u{2605}")); // ★
        star.add_css_class("dim-label");
        stars_box.append(&star);
    }
    stars_box
}

/// Paint the first `rating` stars lit (accent) and the rest dim. `pub(crate)` so
/// the darkroom header (m4-28) seeds its star row from the same rating.
pub(crate) fn set_stars(stars_box: &gtk4::Box, rating: u8) {
    let mut child = stars_box.first_child();
    let mut i = 0u8;
    while let Some(w) = child {
        if let Some(lbl) = w.downcast_ref::<gtk4::Label>() {
            i += 1;
            if i <= rating {
                lbl.remove_css_class("dim-label");
                lbl.add_css_class("accent");
            } else {
                lbl.remove_css_class("accent");
                lbl.add_css_class("dim-label");
            }
        }
        child = w.next_sibling();
    }
}

/// Remove every `GestureClick` previously attached to `w`. `connect_bind` fires
/// on every cell recycle and the wire_* helpers run inside it, so without this a
/// single recycled label accumulates one gesture per bind — each carrying a
/// now-stale path — and one user click fans out into N writes. Stripping first
/// keeps exactly one live gesture (the current bind's) per label.
fn clear_click_gestures(w: &gtk4::Widget) {
    let controllers = w.observe_controllers();
    let mut stale = Vec::new();
    for i in 0..controllers.n_items() {
        if let Some(g) = controllers.item(i).and_downcast::<gtk4::GestureClick>() {
            stale.push(g);
        }
    }
    for g in stale {
        w.remove_controller(&g);
    }
}

/// Attach GestureClick to each star so clicking star k sets rating k. Strips the
/// prior bind's gestures first (see `clear_click_gestures`). Repaints synchronously
/// then persists off-thread — no async read-back, so (unlike `wire_color_clicks`)
/// it needs no `widget_name` recycle guard and works on a static box too.
/// `pub(crate)` so the darkroom header (m4-28) reuses the exact click-to-rate wiring.
pub(crate) fn wire_star_clicks(stars_box: &gtk4::Box, full_path: String, db_path: String) {
    let mut child = stars_box.first_child();
    let mut k = 0u8;
    while let Some(w) = child {
        k += 1;
        let pos = k;
        if let Some(lbl) = w.downcast_ref::<gtk4::Label>() {
            clear_click_gestures(lbl.upcast_ref());
            let gesture = gtk4::GestureClick::new();
            let sb  = stars_box.clone();
            let fp  = full_path.clone();
            let db  = db_path.clone();
            gesture.connect_pressed(move |_, _, _, _| {
                let new_rating = pos;
                set_stars(&sb, new_rating);
                let fp2 = fp.clone();
                let db2 = db.clone();
                glib::spawn_future_local(async move {
                    serialized_write(fp2.clone(), move || {
                        if let Err(e) = save_rating(&fp2, &db2, new_rating) {
                            eprintln!("darkroom: save rating failed for {fp2}: {e}");
                        }
                    }).await;
                });
            });
            lbl.add_controller(gesture);
        }
        child = w.next_sibling();
    }
}

/// darktable's `images.flags` bit layout for ratings (src/common/ratings.h +
/// src/common/image.c `dt_image_get_xmp_rating_from_flags`): the 0–5 star rating
/// lives in bits 0–2 (`DT_VIEW_RATINGS_MASK = 0x7`); "rejected" is a *separate*
/// bit 3 (`DT_IMAGE_REJECTED = 8`), orthogonal to the star value. Keeping every
/// Rust read/write on this exact convention means a rating set here is read
/// identically by the C app AND by our own grid sort/filter ([`SortOrder::Rating`],
/// [`rating_predicate`]). (An earlier bits-1..3 scheme here silently disagreed with both
/// — a 3-star image landed on `flags & 7 == 6`, i.e. read as *rejected*.)
const DT_VIEW_RATINGS_MASK: i64 = 0x7;
const DT_IMAGE_REJECTED: i64 = 0x8;

/// The 0–5 star value stored in `flags` (bits 0–2), clamped for safety (a legacy
/// `flags & 7` of 6/7 from pre-migration darktable can't over-fill the star row).
fn flags_star_rating(flags: i64) -> u8 {
    (flags & DT_VIEW_RATINGS_MASK).min(5) as u8
}

/// `flags` with its star rating replaced by `rating` (0–5), preserving every other
/// bit — the reject bit, LDR/RAW/HDR, local-copy, etc. A Rust mirror of
/// `save_rating`'s `(flags & ~7) | r` SQL, so the write bit-maths (and its
/// composition with the filter reader) is unit-testable under the display-free
/// discipline. Test-only — production writes go through `save_rating`'s SQL.
#[cfg(test)]
fn flags_with_star_rating(flags: i64, rating: u8) -> i64 {
    (flags & !DT_VIEW_RATINGS_MASK) | (rating as i64 & DT_VIEW_RATINGS_MASK)
}

/// Read an image's 0–5 star rating (from `images.flags` bits 0–2) by path, or
/// `None` for the empty/absent db or an unresolvable path. `pub(crate)` so the
/// darkroom header (m4-28) seeds its star row on image open.
/// Open the rating DB connection with a 3s `busy_timeout` so a rating read/write
/// waits out a transient lock instead of failing immediately. Matters now that the
/// darkroom header (m4-28) can set a rating while the same view's debounced
/// autosave/history writer holds the DB — default SQLite returns `SQLITE_BUSY` at
/// once, silently dropping the write; the timeout lets it retry. Rating sibling of
/// [`open_colorlabels_conn`]; returns a `Result` (not `Option`) so `save_rating`
/// can `?`-propagate the real open error while `query_rating` maps it with `.ok()`.
fn open_rating_conn(db_path: &str) -> rusqlite::Result<rusqlite::Connection> {
    let conn = rusqlite::Connection::open(db_path)?;
    let _ = conn.busy_timeout(std::time::Duration::from_secs(3));
    Ok(conn)
}

pub(crate) fn query_rating(full_path: &str, db_path: &str) -> Option<u8> {
    if db_path.is_empty() { return None; }
    let conn = open_rating_conn(db_path).ok()?;
    let p    = std::path::Path::new(full_path);
    let filename = p.file_name()?.to_str()?;
    let folder   = p.parent()?.to_str()?;
    let flags: i64 = conn.query_row(
        "SELECT i.flags FROM main.images i \
         JOIN main.film_rolls f ON f.id = i.film_id \
         WHERE f.folder = ?1 AND i.filename = ?2",
        rusqlite::params![folder, filename],
        |row| row.get(0),
    ).ok()?;
    Some(flags_star_rating(flags))
}

fn save_rating(full_path: &str, db_path: &str, rating: u8) -> rusqlite::Result<()> {
    if db_path.is_empty() { return Ok(()); }
    let conn     = open_rating_conn(db_path)?;
    let p        = std::path::Path::new(full_path);
    let filename = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let folder   = p.parent().and_then(|d| d.to_str()).unwrap_or("");
    // Write the star value into bits 0–2 only (`& ~7`), preserving the reject bit
    // and every other flag — matches `flags_with_star_rating` and darktable.
    let bits     = rating as i64 & DT_VIEW_RATINGS_MASK;
    conn.execute(
        "UPDATE main.images SET flags = (flags & ~7) | ?1 \
         WHERE id = (SELECT i.id FROM main.images i \
                     JOIN main.film_rolls f ON f.id = i.film_id \
                     WHERE f.folder = ?2 AND i.filename = ?3 LIMIT 1)",
        rusqlite::params![bits, folder, filename],
    )?;
    Ok(())
}

// ── Off-thread metadata-write serialization ───────────────────────────────

/// Per-image lock making off-thread metadata writes (star rating + colour label)
/// to the *same* image mutually exclusive, so rapid inputs can't interleave a
/// read-modify-write. What it guarantees: each write runs atomically, and since
/// completion order equals lock-release (== DB-commit) order, the *last* repaint
/// continuation always reflects the *last committed* value — the UI never
/// diverges from the DB. What it does NOT guarantee: input order. `std::sync::
/// Mutex` is unfair and `spawn_blocking` dispatch isn't FIFO, so two truly
/// simultaneous same-image writes can commit in either order. That's harmless
/// for colour labels (bit-flips commute) and acceptable for ratings (a same-image
/// double-set is a race the user can't perceive; both DB and UI stay consistent,
/// just not necessarily on the later keystroke). Keyed by path string; the lock
/// is only ever held inside a `spawn_blocking` worker thread (never on the GTK
/// main loop), so it can't stall the UI. The registry grows one small entry per
/// distinct image written this session — bounded by library size, not worth
/// evicting. (A single dedicated DB-writer thread fed by a channel would also
/// bound blocking-pool use and coalesce writes — deferred, not needed at input
/// rates.)
fn path_write_lock(path: &str) -> Arc<Mutex<()>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
    map.entry(path.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Run metadata-write closure `f` for `path` off the main thread, serialized
/// against other writes to the same image (via [`path_write_lock`]) and with the
/// `spawn_blocking` join failure logged instead of silently swallowed. Returns
/// the closure's value, or `None` if the worker thread panicked so callers can
/// fall back (e.g. repaint from 0). The single choke point all four write sites
/// (star click, rating key, colour click, colour key) route through, so the
/// serialize-and-log guarantee stays symmetric across them.
async fn serialized_write<T, F>(path: String, f: F) -> Option<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let lock = path_write_lock(&path);
    match gio::spawn_blocking(move || {
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        f()
    })
    .await
    {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("darkroom: metadata-write worker failed for {path}: {e:?}");
            None
        }
    }
}

// ── Colour-label helpers ──────────────────────────────────────────────────

/// Number of colour labels, mirroring `c41_db::colorlabels::COLOR_COUNT`.
/// `pub(crate)` so the left-panel colour filter (`panels`) can iterate the same
/// colour domain and render matching swatches without redefining it.
pub(crate) const COLOR_COUNT: u8 = c41_db::colorlabels::COLOR_COUNT;

/// Display hex per colour index (0 red, 1 yellow, 2 green, 3 blue, 4 purple).
const COLOR_HEX: [&str; COLOR_COUNT as usize] =
    ["#e74c3c", "#f1c40f", "#27ae60", "#3498db", "#9b59b6"];

/// Grey used for an unassigned (unlit) dot.
const COLOR_DIM_HEX: &str = "#777777";

/// Build a fresh colour-label dot row: `COLOR_COUNT` filled-circle `Label`s, all
/// unlit, in a centred horizontal box. One source of truth for the dot-row layout
/// shared by the lighttable grid cells and the darkroom header (m4-24); the caller
/// then seeds the lit state via [`set_color_dots`] and wires toggling via
/// [`wire_color_clicks`]. `pub(crate)` so the darkroom view reuses the exact same
/// construction without redefining the dot geometry.
pub(crate) fn build_color_dots_box() -> gtk4::Box {
    let colors_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(2)
        .halign(gtk4::Align::Center)
        .build();
    // The dots are pure Pango-markup glyphs, invisible to assistive tech; give the
    // row a group role + accessible name so screen readers announce it in both the
    // grid cells and the darkroom header (one fix, both call sites inherit it).
    colors_box.set_accessible_role(gtk4::AccessibleRole::Group);
    colors_box.update_property(&[gtk4::accessible::Property::Label("Colour labels")]);
    for idx in 0..COLOR_COUNT {
        let dot = gtk4::Label::new(None);
        dot.set_markup(&color_dot_markup(idx, false));
        colors_box.append(&dot);
    }
    colors_box
}

/// Pango markup for one colour dot: its own hue when `lit`, dim grey otherwise.
/// Pure (no GTK) so it's unit-testable; an out-of-range `idx` falls back to grey.
/// `pub(crate)` so the left-panel colour filter reuses the exact same hues as the
/// grid cells (one source of truth for colour rendering across `c41-ui`).
pub(crate) fn color_dot_markup(idx: u8, lit: bool) -> String {
    let hex = match COLOR_HEX.get(idx as usize) {
        Some(h) if lit => h,
        _ => COLOR_DIM_HEX,
    };
    format!("<span foreground=\"{hex}\">\u{25cf}</span>") // ●
}

/// Repaint the dot row from a 5-bit colour-label mask (bit `c` = colour `c`).
/// `pub(crate)` so the darkroom header (m4-24) seeds its dot row from the same path.
pub(crate) fn set_color_dots(colors_box: &gtk4::Box, mask: u8) {
    let mut child = colors_box.first_child();
    let mut idx = 0u8;
    while let Some(w) = child {
        if let Some(lbl) = w.downcast_ref::<gtk4::Label>() {
            lbl.set_markup(&color_dot_markup(idx, mask & (1 << idx) != 0));
            idx += 1;
        }
        child = w.next_sibling();
    }
}

/// Where the colour quick-filter token persists (same store as
/// `lib.rs`'s `RATING_FILTER_PREF_KEY`). `pub(crate)` because `lib.rs` owns the db
/// path: it restores from this key at startup and runs the single persist observer
/// that writes it.
pub(crate) const COLOUR_FILTER_PREF_KEY: &str = "colour_filter";

/// Where the aspect-ratio quick-filter token persists (m4-128; same store and
/// same restore/persist split as [`COLOUR_FILTER_PREF_KEY`] — `lib.rs` restores
/// from this key at startup and runs the persist observer that writes it).
pub(crate) const ASPECT_FILTER_PREF_KEY: &str = "aspect_filter";

/// The lighttable quick-filter's colour circles (m4-126) — darktable's bar-mounted
/// colour filter. Five buttons drawn with the SAME Pango dot glyphs as the grid
/// cells ([`color_dot_markup`], one source of truth for hues), lit per the live
/// [`current_colour_mask`]; clicking one toggles that colour's bit through
/// [`set_colour_filter`]. The left panel's checks and this row are two mirrors of
/// one state — [`add_filter_observer`] keeps them in step in both directions, and
/// persists the compact token (`db_path`) so the filter survives a restart.
///
/// Built twice (top bar + bottom bar); each instance registers its own
/// display-refresh observer, so a change through any control repaints both rows.
/// Persistence is deliberately NOT here — it lives in one app-level observer in
/// `lib.rs` (`COLOUR_FILTER_PREF_KEY`), so N mirrors still mean one write per
/// change. Main-thread only.
pub fn colour_circles_row() -> gtk4::Box {
    let row = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(0)
        .build();
    row.add_css_class("linked");
    row.set_tooltip_text(Some("Filter by colour label"));
    // Accessibility: same treatment as the grid cells' dot rows above.
    row.set_accessible_role(gtk4::AccessibleRole::Group);
    row.update_property(&[gtk4::accessible::Property::Label("Colour label filter")]);

    let circles: std::rc::Rc<Vec<gtk4::Button>> = std::rc::Rc::new(
        (0..COLOR_COUNT)
            .map(|idx| {
                let b = gtk4::Button::new();
                let lbl = gtk4::Label::new(None);
                lbl.set_markup(&color_dot_markup(idx, false));
                b.set_child(Some(&lbl));
                let c = format!("colour {idx}");
                b.set_tooltip_text(Some(&c));
                // The only text content is the Pango dot glyph, so give assistive
                // tech a real name (the tooltip alone isn't one).
                let name = format!("Filter colour {idx}");
                b.update_property(&[gtk4::accessible::Property::Label(name.as_str())]);
                b
            })
            .collect(),
    );
    for b in circles.iter() {
        row.append(b);
    }

    // Lit-state repaint from the live mask. Registered as an observer so changes
    // made anywhere else (left panel checks, the OTHER bar's circles) land here.
    let refresh: std::rc::Rc<dyn Fn()> = {
        let circles = circles.clone();
        Rc::new(move || {
            let mask = current_colour_mask();
            for (i, b) in circles.iter().enumerate() {
                let Some(lbl) = b.child().and_downcast::<gtk4::Label>() else { continue };
                lbl.set_markup(&color_dot_markup(i as u8, mask & (1 << i) != 0));
            }
        })
    };

    for (i, b) in circles.iter().enumerate() {
        b.connect_clicked(move |_| {
            // Consult the sync guard like every other filter control: inert today
            // (plain Buttons only emit `clicked` on real input), but it stops being
            // inert the moment these become toggles an observer sets — see the
            // bottom-bar stars' identical comment. No explicit repaint needed:
            // `set_colour_filter` fans out through the observer bus, which
            // includes this row's own refresh.
            if filter_sync_in_progress() {
                return;
            }
            set_colour_filter(current_colour_mask() ^ (1 << i), current_colour_all());
        });
    }
    refresh(); // sync to whatever was restored at startup

    {
        let refresh = refresh.clone();
        add_filter_observer(move || {
            refresh();
        });
    }
    row
}

/// Attach a GestureClick to each dot so clicking colour `c` toggles that label
/// and repaints the row from the resulting mask (see `wire_star_clicks` for the
/// once-per-bind-cycle rationale).
///
/// The repaint is guarded by `cb2.widget_name() == path`, the lighttable's
/// cell-recycle check. Callers with a static (non-recycled) row — e.g. the
/// darkroom header (m4-24) — must therefore stamp the box's `widget_name` with
/// `full_path` so the guard passes; an unstamped box would silently skip repaints.
/// `pub(crate)` so the darkroom view reuses the exact toggle/repaint wiring.
pub(crate) fn wire_color_clicks(colors_box: &gtk4::Box, full_path: String, db_path: String) {
    let mut child = colors_box.first_child();
    let mut idx = 0u8;
    while let Some(w) = child {
        if let Some(lbl) = w.downcast_ref::<gtk4::Label>() {
            clear_click_gestures(lbl.upcast_ref());
            let color = idx;
            let gesture = gtk4::GestureClick::new();
            let cb = colors_box.clone();
            let fp = full_path.clone();
            let db = db_path.clone();
            gesture.connect_pressed(move |_, _, _, _| {
                let cb2 = cb.clone();
                let fp2 = fp.clone();
                let db2 = db.clone();
                glib::spawn_future_local(async move {
                    let path = fp2.clone();
                    // On a worker panic the mask is unknown, so skip the repaint
                    // rather than clear labels still in the DB (the next rebind
                    // paints from the DB); likewise skip if the cell was recycled.
                    if let Some(mask) =
                        serialized_write(fp2.clone(), move || toggle_color_label(&fp2, &db2, color)).await
                    {
                        if cb2.widget_name() == path {
                            set_color_dots(&cb2, mask);
                        }
                    }
                });
            });
            lbl.add_controller(gesture);
            idx += 1;
        }
        child = w.next_sibling();
    }
}

/// Open the colour-labels DB connection with a 3s `busy_timeout` so a colour-label
/// read/write waits out a transient lock instead of failing immediately. Matters
/// now that the darkroom header (m4-24) can toggle a label while the same view's
/// debounced autosave/history writer holds the DB — default SQLite returns
/// `SQLITE_BUSY` at once, silently dropping the toggle; the timeout lets it retry.
fn open_colorlabels_conn(db_path: &str) -> Option<rusqlite::Connection> {
    let conn = rusqlite::Connection::open(db_path).ok()?;
    let _ = conn.busy_timeout(std::time::Duration::from_secs(3));
    Some(conn)
}

/// Read an image's colour-label mask by path. Returns 0 for the demo/empty db, an
/// unresolvable path, or any DB error (the dots simply show unlit).
/// `pub(crate)` so the darkroom header (m4-24) seeds its dot row on image open.
pub(crate) fn query_color_labels(full_path: &str, db_path: &str) -> u8 {
    if db_path.is_empty() { return 0; }
    let conn = match open_colorlabels_conn(db_path) {
        Some(c) => c,
        None => return 0,
    };
    let imgid = match c41_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        _ => return 0,
    };
    c41_db::colorlabels::color_labels_get(&conn, imgid).unwrap_or(0)
}

/// Map a function key F1–F5 to its colour-label index (0–4, matching darktable's
/// lighttable accelerators: red/yellow/green/blue/purple), or `None` for any other
/// key. Pure (no GTK realization needed — `gdk::Key` constants are plain keyvals)
/// so the keyboard handler's mapping is unit-testable under the display-free
/// discipline; the toggle + repaint it drives is GTK-bound and tested by Docker.
pub(crate) fn fkey_to_color(keyval: gtk4::gdk::Key) -> Option<u8> {
    match keyval {
        gtk4::gdk::Key::F1 => Some(0),
        gtk4::gdk::Key::F2 => Some(1),
        gtk4::gdk::Key::F3 => Some(2),
        gtk4::gdk::Key::F4 => Some(3),
        gtk4::gdk::Key::F5 => Some(4),
        _ => None,
    }
}

/// Toggle colour label `color` on every selected image and repaint each
/// realized cell's dot row in place (no full reload, so scroll position and the
/// other cells' in-flight async loads are untouched). Since m4-144 the action
/// fans out over the WHOLE multi-selection — upstream toggles the label bit per
/// selected image (`dt_colorlabels_toggle_label` per id), which can leave mixed
/// masks when the selection's labels disagree; that is faithful, not a bug.
/// Falls back to the cursor image alone when the set is empty.
///
/// The DB writes run off the main thread as one serialised write PER IMAGE,
/// each under that image's own [`path_write_lock`] (senior review MAJOR-3: a
/// whole-batch-under-the-anchor's-lock shape would run concurrently with a
/// dot-click on any other selected cell, and the toggle is read-modify-write
/// in the DB — exactly the race the per-image lock exists to prevent). Each
/// in-place repaint then targets the realized cell still bound to that path —
/// matched exactly like `wire_color_clicks`, except the keyboard path holds no
/// `colors_box` reference, so it must *find* rows among the grid's cells
/// ([`repaint_color_dots_for_path`]). Cells recycled or scrolled off-screen by
/// the time the toggle resolves simply miss the repaint; their next bind paints
/// the new mask from the DB.
pub fn toggle_selected_color(
    grid: &GridView,
    selection: &SingleSelection,
    db_path: &str,
    color: u8,
) {
    let mut targets = selection::paths_snapshot();
    if targets.is_empty() {
        // darktable parity: with nothing multi-selected, keys act on the
        // active image.
        if let Some(path) = selected_path(selection) {
            targets.push(path);
        }
    }
    if targets.is_empty() {
        return;
    }
    let db = db_path.to_string();
    glib::spawn_future_local(clone!(@weak grid => async move {
        // Sequential per-target writes: N blocking-pool hops is nothing at
        // input rates, and a worker panic now costs only the remaining
        // images' repaints instead of the whole batch.
        let mut masks = Vec::with_capacity(targets.len());
        for path in &targets {
            let p   = path.clone();
            let db2 = db.clone();
            let Some(mask) =
                serialized_write(path.clone(), move || toggle_color_label(&p, &db2, color))
                    .await
            else {
                break; // worker panic: later masks unknown, stop writing AND repainting
            };
            masks.push(mask);
        }
        for (path, mask) in targets.into_iter().zip(masks) {
            repaint_color_dots_for_path(&grid, &path, mask);
        }
    }));
}

/// Re-read an image's colour-label mask from the DB and repaint its realized grid
/// cell in place (m4-25). Used to sync the lighttable after the darkroom
/// single-image view may have toggled labels: on `NavigationView` pop we don't
/// know *whether* anything changed, so we just re-query and repaint — visually a
/// no-op when unchanged. Same off-thread-read → in-place-repaint shape as
/// [`toggle_selected_color`], minus the write; a no-op if the cell isn't realized
/// (scrolled off / never on-screen), since the next bind paints it from the DB.
pub fn refresh_color_dots_for_path(grid: &GridView, db_path: &str, path: &str) {
    let db = db_path.to_string();
    let path = path.to_string();
    glib::spawn_future_local(clone!(@weak grid => async move {
        let p   = path.clone();
        let db2 = db.clone();
        let mask = gio::spawn_blocking(move || query_color_labels(&p, &db2))
            .await.unwrap_or(0);
        repaint_color_dots_for_path(&grid, &path, mask);
    }));
}

/// Re-read an image's star rating from the DB and repaint its realized grid cell's
/// star row in place (m4-28). Star sibling of [`refresh_color_dots_for_path`], used
/// by the same `NavigationView` pop handler so a rating changed in the darkroom
/// single-image view shows in the lighttable on return. No-op if the cell isn't
/// realized; the next bind paints it from the DB.
pub fn refresh_stars_for_path(grid: &GridView, db_path: &str, path: &str) {
    let db = db_path.to_string();
    let path = path.to_string();
    glib::spawn_future_local(clone!(@weak grid => async move {
        let p   = path.clone();
        let db2 = db.clone();
        let rating = gio::spawn_blocking(move || query_rating(&p, &db2))
            .await.ok().flatten().unwrap_or(0);
        repaint_stars_for_path(&grid, &path, rating);
    }));
}

/// Repaint the star row bound to `path` among the grid's *realized* cells, to
/// `rating`. Star sibling of [`repaint_color_dots_for_path`]; no-op if the image
/// isn't realized (off-screen), the next bind painting it from the DB.
fn repaint_stars_for_path(grid: &GridView, path: &str, rating: u8) {
    if let Some(stars) = find_stars_box_for_path(grid.upcast_ref::<gtk4::Widget>(), path) {
        set_stars(&stars, rating);
    }
}

/// Map a top-row or keypad digit `0`–`5` to its star rating (darktable's lighttable
/// rating accelerators), or `None` for any other key. Pure (`gdk::Key` constants are
/// plain keyvals) so it's unit-testable under the display-free discipline, like
/// [`fkey_to_color`]; the set + repaint it drives is GTK-bound and Docker-tested.
///
/// The keypad arm assumes NumLock is ON (with it off the keypad emits `KP_Insert`/
/// `KP_End`/… instead of `KP_0`/`KP_1`/…, so keypad rating simply won't fire — the
/// top-row digits work regardless, matching darktable and near-universal behaviour).
pub(crate) fn digit_to_rating(keyval: gtk4::gdk::Key) -> Option<u8> {
    use gtk4::gdk::Key;
    match keyval {
        Key::_0 | Key::KP_0 => Some(0),
        Key::_1 | Key::KP_1 => Some(1),
        Key::_2 | Key::KP_2 => Some(2),
        Key::_3 | Key::KP_3 => Some(3),
        Key::_4 | Key::KP_4 => Some(4),
        Key::_5 | Key::KP_5 => Some(5),
        _ => None,
    }
}

/// Set the star `rating` on every selected image and repaint each realized
/// cell's star row in place (m4-29). Star sibling of [`toggle_selected_color`],
/// except ratings are an ABSOLUTE set (digit `k` → rating `k`, matching the star
/// click handler and darktable) rather than a toggle — and since m4-144 the set
/// fans out over the WHOLE multi-selection, falling back to the cursor image
/// when nothing is multi-selected. No-op when neither yields a real target.
/// One serialised write PER IMAGE under its own [`path_write_lock`] — same
/// shape as [`toggle_selected_color`] (senior review MAJOR-3); per-path
/// in-place repaints only where the write committed, a no-op for any cell
/// recycled or scrolled off-screen (its next bind paints from the DB).
pub fn set_selected_rating(
    grid: &GridView,
    selection: &SingleSelection,
    db_path: &str,
    rating: u8,
) {
    let mut targets = selection::paths_snapshot();
    if targets.is_empty() {
        if let Some(path) = selected_path(selection) {
            targets.push(path);
        }
    }
    if targets.is_empty() {
        return;
    }
    let db = db_path.to_string();
    glib::spawn_future_local(clone!(@weak grid => async move {
        let mut written = Vec::with_capacity(targets.len());
        for path in &targets {
            let p   = path.clone();
            let db2 = db.clone();
            match serialized_write(path.clone(), move || save_rating(&p, &db2, rating)).await {
                Some(Ok(())) => written.push(path.clone()),
                Some(Err(e)) => eprintln!("darkroom: save rating failed for {path}: {e}"),
                None => break, // worker panic: stop writing and repainting
            }
        }
        // Repaint only what committed — stars must never show a value the DB
        // rejected; the next bind would correct it, but why show it at all.
        for path in &written {
            repaint_stars_for_path(&grid, path, rating);
        }
    }));
}

/// Repaint the colour-dot row bound to `path` among the grid's *realized* cells,
/// from `mask`. The keyboard toggle (unlike the per-dot click handlers) holds no
/// reference to the cell's `colors_box`, so it locates the row here. Cells are
/// recycled, so we match BOTH the bound identity (`widget_name == path`) and the
/// structural slot (4th child of a `Picture`-led cell vbox) — `stars_box` carries
/// the same name, so name alone is ambiguous. No-op if the image isn't realized
/// (off-screen); the next bind paints it from the DB.
fn repaint_color_dots_for_path(grid: &GridView, path: &str, mask: u8) {
    if let Some(colors) = find_color_box_for_path(grid.upcast_ref::<gtk4::Widget>(), path) {
        set_color_dots(&colors, mask);
    }
}

/// Depth-first search of `root`'s descendants for a per-cell metadata row `Box`
/// (`child_index` within the cell vbox: 2 = stars, 3 = colour dots) of the cell
/// currently bound to `path`. A cell vbox is recognised by its first child being
/// the thumbnail `Picture`. Returns the first match, or `None` if no realized cell
/// is showing `path`.
///
/// `widget_name` is the bind-time stamp and is shared by the cell's thumb/stars/
/// colour widgets, so we require BOTH the thumb AND the target row to carry `path`
/// — a single stale stamp (a cell mid-recycle, where the bind hasn't re-stamped
/// every child yet) can't then mis-target a neighbouring cell. Cross-*cell*
/// uniqueness still rests on grid paths being distinct, the same assumption
/// [`index_of_path`] documents; the loaders' joins yield distinct `folder/filename`
/// rows today, so at most one realized cell carries a given `path`. Worst case if
/// that ever breaks is a transient repaint of a duplicate's twin that self-heals
/// on its next bind — the DB write (the source of truth) is unaffected.
fn find_cell_row_for_path(root: &gtk4::Widget, path: &str, child_index: usize) -> Option<gtk4::Box> {
    let mut child = root.first_child();
    while let Some(w) = child {
        if let Some(vbox) = w.downcast_ref::<gtk4::Box>() {
            if let Some(thumb) = vbox.first_child().and_downcast::<gtk4::Picture>() {
                if thumb.widget_name().as_str() == path {
                    if let Some(row) = nth_child(vbox, child_index).and_downcast::<gtk4::Box>() {
                        if row.widget_name().as_str() == path {
                            return Some(row);
                        }
                    }
                }
            }
        }
        if let Some(found) = find_cell_row_for_path(&w, path, child_index) {
            return Some(found);
        }
        child = w.next_sibling();
    }
    None
}

/// Colour-dot row (4th child, index 3) of the realized cell bound to `path`.
fn find_color_box_for_path(root: &gtk4::Widget, path: &str) -> Option<gtk4::Box> {
    find_cell_row_for_path(root, path, 3)
}

// ── Multi-image selection frames (m4-144) ──────────────────────────────────

/// CSS class marking a selected cell's thumbnail; styled as an inset outline
/// in `theme.rs`, mirroring darktable's selection border without shifting the
/// layout.
const SEL_CSS_CLASS: &str = "c41-cell-selected";

fn apply_selection_frame(thumb: &gtk4::Picture, selected: bool) {
    if selected {
        thumb.add_css_class(SEL_CSS_CLASS);
    } else {
        thumb.remove_css_class(SEL_CSS_CLASS);
    }
}

/// Set every REALIZED cell's frame to match the live selection set. Off-screen
/// cells aren't realized; their next bind applies the frame straight from
/// [`selection::contains`], which is what makes this walk safe to skip them.
fn refresh_selection_frames(grid: &GridView) {
    for_each_realized_cell(grid.upcast_ref::<gtk4::Widget>(), &mut |thumb, path| {
        apply_selection_frame(thumb, selection::contains(&path));
    });
}

/// Visit every realized cell's stamped thumbnail `(thumb, path)` among root's
/// descendants. A cell is recognised exactly like [`find_cell_row_for_path`]:
/// a `Box` whose first child is the bind-stamped thumb `Picture`; placeholders
/// (no `/`) never visit.
fn for_each_realized_cell(root: &gtk4::Widget, f: &mut dyn FnMut(&gtk4::Picture, String)) {
    let mut child = root.first_child();
    while let Some(w) = child {
        if let Some(vbox) = w.downcast_ref::<gtk4::Box>() {
            if let Some(thumb) = vbox.first_child().and_downcast::<gtk4::Picture>() {
                let name = thumb.widget_name().to_string();
                if name.contains('/') {
                    f(&thumb, name);
                }
            }
        }
        for_each_realized_cell(&w, f);
        child = w.next_sibling();
    }
}

/// Star-rating row (3rd child, index 2) of the realized cell bound to `path`.
fn find_stars_box_for_path(root: &gtk4::Widget, path: &str) -> Option<gtk4::Box> {
    find_cell_row_for_path(root, path, 2)
}

/// Toggle one colour label for an image (by path) and return the resulting mask
/// so the caller can repaint. A no-op returning 0 if the path can't be resolved.
fn toggle_color_label(full_path: &str, db_path: &str, color: u8) -> u8 {
    if db_path.is_empty() { return 0; }
    let conn = match open_colorlabels_conn(db_path) {
        Some(c) => c,
        None => return 0,
    };
    let imgid = match c41_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        _ => return 0,
    };
    if let Err(e) = c41_db::colorlabels::color_label_toggle(&conn, imgid, color) {
        eprintln!("darkroom: colour-label toggle failed: {e}");
    }
    c41_db::colorlabels::color_labels_get(&conn, imgid).unwrap_or(0)
}

// ── DB-backed load functions ──────────────────────────────────────────────

/// Hard cap on grid rows per load. Every loader queries `LIMIT GRID_CAP + 1` so a
/// full page (exactly `GRID_CAP` images) can be told apart from an over-full one
/// and flagged, rather than silently truncating.
const GRID_CAP: usize = 2000;

/// Grid sort order, chosen from the lighttable's "sort by" dropdown. Applied
/// uniformly by every loader's `ORDER BY` (see [`SortOrder::order_clause`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SortOrder {
    #[default]
    Filename,
    DateTaken,
    Rating,
}

impl SortOrder {
    /// The ordered `(expr, ascending, reversible)` terms of this sort's *natural*
    /// order. `expr` is static text — never user input — so it's injection-safe to
    /// interpolate; all loaders alias the images table as `i`, so these column refs
    /// are valid everywhere. `ascending` is the term's natural direction;
    /// `reversible` marks whether the "reverse sort" toggle flips it (an
    /// undated-last guard stays put so undated images never bubble to the top).
    fn terms(self) -> &'static [(&'static str, bool, bool)] {
        match self {
            // Filename groups naturally by folder (dates are foldered YYYY_MM_DD).
            SortOrder::Filename => &[("f.folder", true, true), ("i.filename", true, true)],
            // Undated images (NULL or 0) sort LAST: the leading boolean is 0 for
            // dated, 1 for undated, so ASC keeps dated photos in date order up top
            // and dumps undated at the end. It's NON-reversible so undated stays at
            // the bottom in both directions; only the datetime/name terms flip.
            SortOrder::DateTaken => &[
                ("(i.datetime_taken IS NULL OR i.datetime_taken = 0)", true, false),
                ("i.datetime_taken", true, true),
                ("i.filename", true, true),
            ],
            // darktable packs the 0..5 star rating in bits 0–2 of flags and the
            // "rejected" state in the SEPARATE bit 3 (= 8), orthogonal to the
            // stars (src/common/collection.c: `CASE WHEN flags & 8 = 8 THEN -1
            // ELSE flags & 7 END`). Highest rating first (DESC); rejected maps to
            // -1 so it sinks below 0-star. Reversed, ascending puts rejected first.
            // A legacy `flags & 7` of 6/7 (a pre-migration bits-1..3 value) is also
            // sunk to -1 so it can't out-rank a real 5-star under DESC — mirroring
            // the >5 clamp in [`flags_star_rating`] and the `BETWEEN … AND 5` in
            // [`rating_predicate`], so all three sites agree on the 0..5 domain.
            SortOrder::Rating => &[
                ("CASE WHEN (i.flags & 8) = 8 OR (i.flags & 7) > 5 THEN -1 ELSE (i.flags & 7) END", false, true),
                ("i.filename", true, true),
            ],
        }
    }

    /// The `ORDER BY` expression (without the `ORDER BY` keyword) for this sort,
    /// optionally reversed. Reversing flips every *reversible* term's direction.
    fn order_clause(self, reverse: bool) -> String {
        self.terms()
            .iter()
            .map(|&(expr, ascending, reversible)| {
                let ascending = if reversible && reverse { !ascending } else { ascending };
                format!("{expr} {}", if ascending { "ASC" } else { "DESC" })
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// How the bottom-bar rating filter compares an image's star value against the
/// chosen star count (m4-98d) — darktable's rating-filter comparator dropdown.
/// `AtLeast(0)` is the canonical "no filter" state (≥ 0 stars = everything).
/// `Rejected` matches darktable's reject bit and ignores the star count entirely.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RatingCompare {
    /// `≥ N` stars (excludes rejected). `N == 0` ⇒ no filter.
    AtLeast,
    /// `= N` stars exactly (excludes rejected; `= 0` ⇒ unrated only).
    Exactly,
    /// `≤ N` stars (excludes rejected; `≤ 0` ⇒ unrated only).
    AtMost,
    /// Rejected images only (the star count is irrelevant in this mode).
    Rejected,
}

impl RatingCompare {
    /// Comparators in dropdown-row order, so the UI index ↔ variant mapping and
    /// the persisted-token order have one source of truth.
    const ALL: [RatingCompare; 4] =
        [Self::AtLeast, Self::Exactly, Self::AtMost, Self::Rejected];

    /// Map a DropDown selection index back to a comparator (out-of-range ⇒ the
    /// default `AtLeast`, so a corrupt index can never panic).
    pub fn from_index(i: u32) -> RatingCompare {
        *Self::ALL.get(i as usize).unwrap_or(&Self::AtLeast)
    }

    /// The comparator's dropdown-row index (for seeding the DropDown selection).
    pub fn to_index(self) -> u32 {
        Self::ALL.iter().position(|&c| c == self).unwrap_or(0) as u32
    }
}

/// A one-click quick-filter preset for the top bar's `filter [all images ▾]`
/// dropdown (m4-97c) — darktable's collection quick-filter. A preset is **not** a
/// parallel filter implementation: each one is just a named `(comparator, stars)`
/// pair applied to the *same* rating-filter state the bottom bar drives, so the
/// two controls can never disagree about what the grid is showing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterPreset {
    /// No filter at all (`≥ 0`).
    AllImages,
    /// Unrated images only (`= 0`).
    UnstarredOnly,
    /// `≥ N` stars, for N in 1..=5.
    AtLeastStars(u8),
    /// Rejected images only.
    RejectedOnly,
}

impl FilterPreset {
    /// Presets in dropdown-row order. One source of truth for the row labels, the
    /// index ↔ variant mapping, and [`FilterPreset::for_state`]'s reverse lookup.
    pub const ALL: [FilterPreset; 8] = [
        Self::AllImages,
        Self::UnstarredOnly,
        Self::AtLeastStars(1),
        Self::AtLeastStars(2),
        Self::AtLeastStars(3),
        Self::AtLeastStars(4),
        Self::AtLeastStars(5),
        Self::RejectedOnly,
    ];

    /// The rating-filter state this preset stands for. The single mapping point
    /// between a preset and the filter primitives, so a preset can never express
    /// something the bottom bar can't also show.
    pub fn state(self) -> (RatingCompare, u8) {
        match self {
            Self::AllImages => (RatingCompare::AtLeast, 0),
            Self::UnstarredOnly => (RatingCompare::Exactly, 0),
            Self::AtLeastStars(n) => (RatingCompare::AtLeast, n.clamp(1, 5)),
            Self::RejectedOnly => (RatingCompare::Rejected, 0),
        }
    }

    /// The preset matching a `(comparator, stars)` state, or `None` when the state
    /// isn't expressible as one (e.g. `≤ 3`, which only the bottom bar can set).
    /// Lets the dropdown reflect filter changes made elsewhere instead of lying.
    ///
    /// The state is **canonicalised first**, exactly as [`rating_predicate`] and
    /// [`rating_filter_token_for`] do, so equal filters compare equal: `Rejected`
    /// ignores the star count (so `(Rejected, 3)` *is* "rejected only" — reachable
    /// by picking 3 stars then switching the comparator to ⚑, since the bottom bar
    /// deliberately retains the count), and counts are clamped to the 0..=5 domain.
    /// Skipping this would report `custom` for a state a preset does name.
    ///
    /// Not canonicalised: `(AtMost, 0)` yields the same rows as `UnstarredOnly`
    /// (`BETWEEN 0 AND 0` ≡ `= 0`) but stays `custom` on purpose — `≤` is a
    /// deliberate bottom-bar choice, so echoing it back as `=` would misreport
    /// which control the user is driving.
    pub fn for_state(cmp: RatingCompare, stars: u8) -> Option<FilterPreset> {
        let stars = if cmp == RatingCompare::Rejected { 0 } else { stars.min(5) };
        Self::ALL.into_iter().find(|p| p.state() == (cmp, stars))
    }

    /// Map a DropDown selection index back to a preset (out-of-range ⇒ `AllImages`,
    /// so a corrupt index can never panic).
    pub fn from_index(i: u32) -> FilterPreset {
        *Self::ALL.get(i as usize).unwrap_or(&Self::AllImages)
    }

    /// The preset's dropdown row index — the inverse of [`Self::from_index`], so
    /// `lib.rs` doesn't open-code the reverse lookup over [`Self::ALL`].
    pub fn to_index(self) -> u32 {
        Self::ALL.iter().position(|&p| p == self).unwrap_or(0) as u32
    }

    /// Row index of the dropdown's trailing **display-only** `custom` row, shown
    /// when the live filter isn't expressible as a preset. Never applied as a
    /// filter — selecting it snaps the control back to the real state.
    pub const CUSTOM_INDEX: u32 = Self::ALL.len() as u32;

    /// The preset's dropdown row label, kept beside the variants so the control
    /// built from [`Self::ALL`] can't drift out of sync with them.
    pub fn label(self) -> String {
        match self {
            Self::AllImages => "all images".to_string(),
            Self::UnstarredOnly => "unstarred only".to_string(),
            Self::AtLeastStars(n) => format!("★ {n} and higher"),
            Self::RejectedOnly => "rejected only".to_string(),
        }
    }
}

/// The aspect-ratio quick-filter (m4-128) — the first slice of darktable's
/// "collection filters" expander (`src/libs/filtering.c`). Its three stock
/// presets are range rules on the stored `aspect_ratio` column — `square` =
/// `[1;1]`, `landscape` = `>=1.01`, `portrait` = `<=0.99`. That column is a
/// float snapped by `dt_usable_aspect` (which is why the thresholds have slop);
/// we don't materialise one, so the same intent is expressed directly against
/// the import-time `width`/`height` integers — exact, no snapping thresholds,
/// and an image with unknown dimensions (a failed probe leaves 0×0) matches
/// nothing instead of wrongly counting as square.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AspectFilter {
    /// No aspect restriction.
    Off,
    /// Wider than tall (darktable's `landscape` preset).
    Landscape,
    /// Taller than wide (darktable's `portrait` preset).
    Portrait,
    /// Width equals height, non-zero (darktable's `square` preset).
    Square,
}

impl AspectFilter {
    /// Variants in dropdown-row order — one source of truth for the labels, the
    /// index ↔ variant mapping and the token roundtrip test.
    pub const ALL: [AspectFilter; 4] =
        [Self::Off, Self::Landscape, Self::Portrait, Self::Square];

    /// Map a DropDown selection index back to a filter (out-of-range ⇒ `Off`,
    /// so a corrupt index can never panic).
    pub fn from_index(i: u32) -> AspectFilter {
        *Self::ALL.get(i as usize).unwrap_or(&Self::Off)
    }

    /// The dropdown-row index (for seeding the DropDown selection).
    pub fn to_index(self) -> u32 {
        Self::ALL.iter().position(|&f| f == self).unwrap_or(0) as u32
    }

    /// Row label, kept beside the variants so the control built from
    /// [`Self::ALL`] can't drift out of sync with them.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "All aspects",
            Self::Landscape => "Landscape",
            Self::Portrait => "Portrait",
            Self::Square => "Square",
        }
    }
}

/// Which per-thumbnail overlays the grid draws (m4-98e) — our port of darktable's
/// thumbnail "overlays" setting. Ordered as the bottom-bar dropdown lists them.
/// `Hidden` is darktable's "no overlays"; named so it can't be confused with
/// `Option::None` at a match site.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OverlayMode {
    /// Thumbnail only — no filename, no stars, no colour dots.
    Hidden,
    /// Stars + colour labels, but no filename (darktable's "normal" overlays).
    Normal,
    /// Filename + stars + colour labels — the original cell layout, and the
    /// default so the out-of-box look is unchanged.
    Extended,
}

impl OverlayMode {
    /// Modes in dropdown-row order, so the UI index ↔ variant mapping and the
    /// persisted-token order have one source of truth. [`OverlayMode::LABELS`] is
    /// built from this, so the dropdown can't drift out of sync with the variants.
    pub const ALL: [OverlayMode; 3] = [Self::Hidden, Self::Normal, Self::Extended];

    /// Map a DropDown selection index back to a mode (out-of-range ⇒ `Extended`,
    /// the default, so a corrupt index can never panic).
    pub fn from_index(i: u32) -> OverlayMode {
        *Self::ALL.get(i as usize).unwrap_or(&Self::Extended)
    }

    /// The mode's dropdown-row index (for seeding the DropDown selection). An
    /// exhaustive match rather than a lookup in [`Self::ALL`]: the compiler then
    /// forces this to be updated when a variant is added, and there's no silent
    /// `unwrap_or(0)` fallback that would disagree with [`Self::from_index`].
    pub fn to_index(self) -> u32 {
        match self {
            Self::Hidden => 0,
            Self::Normal => 1,
            Self::Extended => 2,
        }
    }

    /// Whether this mode draws filename captions. Shared decision between the
    /// grid cells (bind-time visibility) and the zoomable canvas (paint-time),
    /// so the overlay dropdown can never mean different things per layout.
    pub fn shows_filenames(self) -> bool {
        !matches!(self, Self::Hidden)
    }

    /// The mode's dropdown row label. Kept next to the variants (not in `lib.rs`)
    /// so the control is built from [`Self::ALL`] and the two can't diverge. Kept
    /// terse — the bottom bar's minimum width is contended (see the ~915px
    /// lighttable overflow note in the plan).
    pub fn label(self) -> &'static str {
        match self {
            Self::Hidden => "None",
            Self::Normal => "Stars",
            Self::Extended => "Full",
        }
    }
}

/// Which of a cell's three metadata rows a mode shows: `(filename, stars,
/// colours)`. Pure, so the mapping is unit-testable under the display-free
/// discipline.
fn overlay_visibility(mode: OverlayMode) -> (bool, bool, bool) {
    match mode {
        OverlayMode::Hidden => (false, false, false),
        OverlayMode::Normal => (false, true, true),
        OverlayMode::Extended => (true, true, true),
    }
}

/// Row visibility for a cell, accounting for **placeholder** rows ("(No images…)",
/// the truncation notice). A placeholder speaks through its label, so it always
/// shows it whatever the mode — hiding it would leave an unexplained empty grid —
/// but never its stars/colour dots, which are meaningless there. The single
/// decision point for both appliers (bind and [`set_overlay_mode`]'s walk), so the
/// carve-out can't be implemented two subtly different ways.
fn effective_overlay_visibility(is_placeholder: bool, mode: OverlayMode) -> (bool, bool, bool) {
    if is_placeholder {
        (true, false, false)
    } else {
        overlay_visibility(mode)
    }
}

/// Persisted overlay-mode token pieces — shared by the encoder and decoder so
/// they can't silently disagree (same discipline as the rating-filter token).
const OVERLAY_TOK_HIDDEN: &str = "none";
const OVERLAY_TOK_NORMAL: &str = "normal";
const OVERLAY_TOK_EXTENDED: &str = "extended";

/// Pure encoder for the persisted overlay-mode token. `pub` so the bottom bar can
/// persist exactly the mode it just applied, rather than re-reading the global.
pub fn overlay_mode_token_for(mode: OverlayMode) -> &'static str {
    match mode {
        OverlayMode::Hidden => OVERLAY_TOK_HIDDEN,
        OverlayMode::Normal => OVERLAY_TOK_NORMAL,
        OverlayMode::Extended => OVERLAY_TOK_EXTENDED,
    }
}

/// Parse a persisted overlay-mode token, falling back to `Extended` (the default
/// look) on anything unrecognised. Derived by *inverting the encoder* over
/// [`OverlayMode::ALL`] rather than restating the mapping, so encoder and decoder
/// cannot drift apart. Pure, so it's unit-testable.
fn parse_overlay_mode_token(tok: &str) -> OverlayMode {
    OverlayMode::ALL
        .iter()
        .find(|&&m| overlay_mode_token_for(m) == tok)
        .copied()
        .unwrap_or(OverlayMode::Extended)
}

/// Seed the overlay mode from a persisted token *without* touching any widget —
/// called at startup before the grid has realized cells, so the first bind
/// already applies it. Main-thread only.
pub fn apply_overlay_mode_token(tok: &str) {
    OVERLAY_MODE.with(|m| m.set(parse_overlay_mode_token(tok)));
}

/// Which lighttable layout the grid is drawn in (m4-98c) — our port of darktable's
/// file manager / zoomable / culling layouts. Ordered as darktable's switcher lists
/// them.
///
/// **Modes reconfigure the one `GridView`** (its model and column bounds) — see
/// [`reconfigure_grid_for`], which windows the model for culling and unwinds it
/// again for the file manager. They must never swap the
/// `ScrolledWindow`'s child: a control that re-derives the grid from the scroller
/// would silently go inert (no error, no log) the moment the child changed, which
/// is why [`lighttable_page`] hands the grid out instead. See the m4-98c design
/// note in `RUST_MIGRATION_PLAN.md`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    /// Scrolling grid of thumbnails — the layout the lighttable has always had,
    /// and the default.
    FileManager,
    /// darktable's infinite zoom plane: a hand-drawn canvas of continuously
    /// zoomable thumbnails (`lighttable/zoomable.rs`, m4-139). Not a `GridView`
    /// reconfiguration — the canvas covers the centre slot as an overlay child
    /// while this mode is active, which is why it's the one mode [`LighttablePage`]
    /// hands out extra widgets for.
    Zoomable,
    /// A fixed window of N images at a time, paged with ← / →, for side-by-side
    /// comparison — the same grid with a `SliceListModel` over its model.
    Culling,
}

impl ViewMode {
    /// Modes in switcher-button order — the switcher and the token table are both
    /// built by iterating this, so neither can drift from the other. Note the
    /// completeness of this list is **convention, not compiler-checked**: a variant
    /// added to the enum and forgotten here would have no button and no token, and
    /// nothing would say so. Adding a variant means touching this line.
    pub const ALL: [ViewMode; 3] = [Self::FileManager, Self::Zoomable, Self::Culling];

    /// Whether *this build* can actually render the mode. The switcher greys out
    /// the rest, and [`apply_view_mode_token`] refuses to restore one — so a pref
    /// written by a later build (or hand-edited) can never leave the lighttable in
    /// a mode that draws nothing. Flip a variant to `true` in the increment that
    /// implements it; that single edit lights up the button and the restore path
    /// together.
    ///
    /// `const` on purpose: availability is a property of the build, not runtime
    /// state — nothing should ever make a mode available conditionally at runtime.
    pub const fn is_available(self) -> bool {
        match self {
            Self::FileManager | Self::Culling | Self::Zoomable => true,
        }
    }

    /// Icon for the mode's switcher button. The bar is icon-only here: the bottom
    /// `CenterBox`'s minimum width is contended (the ~915px lighttable overflow
    /// gotcha), and this control shares the centre slot with the overlay dropdown.
    /// All three are core Adwaita symbolics — a name absent from the theme renders
    /// as the broken-image glyph rather than failing loudly, so they were checked
    /// against the container's icon theme
    /// (`/usr/share/icons/Adwaita/symbolic/actions/`).
    pub const fn icon_name(self) -> &'static str {
        match self {
            Self::FileManager => "view-grid-symbolic",
            Self::Zoomable => "view-paged-symbolic",
            Self::Culling => "view-dual-symbolic",
        }
    }

    /// One line describing the mode — the only place it is spelled out, since its
    /// button carries just an icon. Rendered into the switcher's *container*
    /// tooltip, never onto the buttons: see [`view_mode_switcher_tooltip`].
    pub const fn tooltip(self) -> &'static str {
        match self {
            Self::FileManager => "File manager: scrolling grid of thumbnails",
            Self::Zoomable => "Zoomable: infinite pannable plane — wheel zooms, drag pans",
            // m4-132: the cells now fill the viewport, so this can promise what
            // darktable's culling does — keep the wording in step with the layout.
            Self::Culling => "Culling: compare a screenful side by side (← →)",
        }
    }
}

/// Tooltip for the switcher **container**, listing every mode.
///
/// It goes on the container and not on the buttons because **GTK4 never emits
/// `query-tooltip` for an insensitive widget** — this repo already learned that on
/// the header's disabled "Other" view (see `build_view_switcher`), where the fix
/// was to drop the tooltip entirely. Here the modes that aren't ported yet are
/// precisely the ones needing explanation, so instead the sensitive parent carries
/// one string covering all of them; a greyed icon then has somewhere to explain
/// itself. Built from [`ViewMode::ALL`] and pure, so it's unit-testable and can't
/// list a mode the switcher doesn't show.
pub fn view_mode_switcher_tooltip() -> String {
    let mut s = String::from("Lighttable layout:");
    for mode in ViewMode::ALL {
        s.push_str("\n• ");
        s.push_str(mode.tooltip());
    }
    s
}

/// Persisted view-mode token pieces — shared by the encoder and decoder so they
/// can't silently disagree (same discipline as the overlay-mode token).
const VIEW_TOK_FILEMANAGER: &str = "filemanager";
const VIEW_TOK_ZOOMABLE: &str = "zoomable";
const VIEW_TOK_CULLING: &str = "culling";

/// Pure encoder for the persisted view-mode token. `pub` so the bottom bar can
/// persist exactly the mode it just applied, rather than re-reading the global.
pub const fn view_mode_token_for(mode: ViewMode) -> &'static str {
    match mode {
        ViewMode::FileManager => VIEW_TOK_FILEMANAGER,
        ViewMode::Zoomable => VIEW_TOK_ZOOMABLE,
        ViewMode::Culling => VIEW_TOK_CULLING,
    }
}

/// Parse a persisted view-mode token, falling back to `FileManager` on anything
/// unrecognised **or not available in this build** — restoring a mode that draws
/// nothing would look like a hung lighttable. Derived by *inverting the encoder*
/// over [`ViewMode::ALL`] rather than restating the mapping, so encoder and decoder
/// cannot drift apart. Pure, so it's unit-testable.
fn parse_view_mode_token(tok: &str) -> ViewMode {
    ViewMode::ALL
        .iter()
        .find(|&&m| view_mode_token_for(m) == tok && m.is_available())
        .copied()
        .unwrap_or(ViewMode::FileManager)
}

/// Seed the view mode from a persisted token *without* touching any widget —
/// called at startup before the grid is configured. Main-thread only.
pub fn apply_view_mode_token(tok: &str) {
    // Through the single writer, so the "current mode is always one we can render"
    // rule lives in exactly one place. The parser already filters unavailable
    // modes, so the write cannot be refused here — and if it ever were, the mode
    // would stay at the renderable default rather than at something blank.
    store_view_mode(parse_view_mode_token(tok));
}

thread_local! {
    /// The current grid sort order (main-thread-only UI state).
    static SORT_ORDER: Cell<SortOrder> = const { Cell::new(SortOrder::Filename) };
    /// Whether the current sort is reversed (the "sort direction" toggle).
    static SORT_REVERSE: Cell<bool> = const { Cell::new(false) };
    /// The star count the bottom-bar rating filter compares against (0..=5). Its
    /// meaning depends on [`RATING_COMPARE`]; with the default `AtLeast` a value of
    /// 0 means "no filter". Applied on top of whatever collection is active (folder
    /// / tag / colour / search), like the sort — see [`rating_predicate`].
    static MIN_RATING: Cell<u8> = const { Cell::new(0) };
    /// How [`MIN_RATING`] is compared (the comparator dropdown). Default `AtLeast`
    /// so the out-of-box state (`AtLeast` + 0 stars) is "show everything".
    static RATING_COMPARE: Cell<RatingCompare> = const { Cell::new(RatingCompare::AtLeast) };
    /// Which per-thumbnail overlays the grid draws. Default `Extended` (filename +
    /// stars + colour dots) so the out-of-box look matches the pre-m4-98e cell.
    static OVERLAY_MODE: Cell<OverlayMode> = const { Cell::new(OverlayMode::Extended) };
    /// Which layout the grid is drawn in (m4-98c). Default `FileManager` — the
    /// scrolling grid the lighttable has always shown.
    static VIEW_MODE: Cell<ViewMode> = const { Cell::new(ViewMode::FileManager) };
    /// Index of the first image in the culling window. Kept across mode switches so
    /// leaving culling and coming back resumes where the user was, not at image 0.
    static CULL_OFFSET: Cell<u32> = const { Cell::new(0) };
    /// The base model whose `items-changed` re-clamps the culling offset, and the
    /// handler doing it — held so it can be disconnected instead of stacking.
    static CULL_BASE_WATCH: RefCell<Option<(gtk4::gio::ListModel, glib::SignalHandlerId)>> =
        const { RefCell::new(None) };
    /// Per-cell thumbnail box size while culling (m4-132): `(width, height)` the
    /// cells request so `window` images fill the viewport instead of sitting in
    /// `THUMB_SIZE` squares. `(0, 0)` = not computed (no allocation yet) — binds
    /// fall back to [`THUMB_SIZE`] until the first resize tick fills this in.
    static CULL_CELL_PX: Cell<(i32, i32)> = const { Cell::new((0, 0)) };
    /// The file-manager `max_columns` (the user's chosen thumbnail size) saved
    /// while culling pins the grid to exactly one row. `0` = nothing saved; a real
    /// value is always ≥ 1, so the sentinel can't collide.
    static CULL_SAVED_MAX_COLS: Cell<u32> = const { Cell::new(0) };
    /// Display-refresh closures for the filter controls (m4-97c) — see
    /// [`add_filter_observer`]. Several controls now drive one filter state, so
    /// each change has to push back out to all of them.
    static FILTER_OBSERVERS: RefCell<Vec<Rc<dyn Fn()>>> = const { RefCell::new(Vec::new()) };
    /// Inclusive year range the timeline is filtering to (m4-99), or `None` for all
    /// years. Composes with the rating filter and the active collection — see
    /// [`current_filters_sql`].
    static YEAR_RANGE: Cell<Option<(i32, i32)>> = const { Cell::new(None) };
    /// Nesting depth of the observer pass ([`sync_filter_controls`]), so the widget
    /// writes those observers make aren't mistaken for user edits — see
    /// [`filter_sync_in_progress`] and [`FilterSyncGuard`].
    static FILTER_SYNC_DEPTH: Cell<u32> = const { Cell::new(0) };
    /// The colour-label quick-filter bitmask (m4-126): bit `c` = colour `c`, 0 =
    /// no colour filter. Composes ON TOP of whatever collection is active (folder /
    /// tag / search / all), like [`MIN_RATING`] and [`YEAR_RANGE`] — it is no longer
    /// a collection selector itself; the left panel's checks and both bars' circles
    /// are three mirrors of this one state.
    static COLOUR_MASK: Cell<u8> = const { Cell::new(0) };
    /// How [`COLOUR_MASK`] combines when several bits are set: `false` = any (OR),
    /// `true` = all (AND). Sticky across clears.
    static COLOUR_ALL: Cell<bool> = const { Cell::new(false) };
    /// The aspect-ratio quick-filter (m4-128): darktable's square/landscape/
    /// portrait collection presets. Composes ON TOP of whatever collection is
    /// active, like [`MIN_RATING`], [`YEAR_RANGE`] and [`COLOUR_MASK`] — see
    /// [`current_filters_sql`].
    static ASPECT_FILTER: Cell<AspectFilter> = const { Cell::new(AspectFilter::Off) };
    /// The collection rule stack (m4-134, parity 2.6 slice 2): arbitrary
    /// AND/OR/AND-NOT rules composing on top of the active collection like the
    /// single-state filters around it — see [`current_filters_sql`] and
    /// [`rule_stack`]. `RefCell<Vec>` because rules carry a value string.
    static RULE_STACK: RefCell<Vec<rule_stack::Rule>> = const { RefCell::new(Vec::new()) };
    /// A closure that re-runs the *current* view's loader. Each loader registers
    /// itself here on every call (capturing its own args), so the sort dropdown
    /// can re-apply the view under a new order without the trigger sites (folder
    /// clicks, search, tag/colour filters) knowing anything about sorting.
    static RELOAD_CURRENT: RefCell<Option<Rc<dyn Fn()>>> = const { RefCell::new(None) };
}

/// The order every loader should apply right now.
fn current_sort() -> SortOrder {
    SORT_ORDER.with(|s| s.get())
}

/// Whether the current view should be sorted in reverse right now.
fn current_reverse() -> bool {
    SORT_REVERSE.with(|r| r.get())
}

/// The star count the rating filter is comparing against right now (0..=5). Its
/// meaning depends on [`current_rating_compare`]. `pub` so the bottom bar can lit
/// the right number of stars.
pub fn current_min_rating() -> u8 {
    MIN_RATING.with(|r| r.get())
}

/// How the rating filter compares [`current_min_rating`] right now. `pub` so the
/// bottom bar can seed its comparator dropdown and grey the stars out in the
/// `Rejected` mode where they're irrelevant.
pub fn current_rating_compare() -> RatingCompare {
    RATING_COMPARE.with(|c| c.get())
}

/// The overlay mode the grid should draw right now. `pub` so the bottom bar can
/// seed its dropdown from the restored value.
pub fn current_overlay_mode() -> OverlayMode {
    OVERLAY_MODE.with(|m| m.get())
}

/// The layout the grid is drawn in right now. `pub` so the bottom bar can seed its
/// switcher from the restored value.
pub fn current_view_mode() -> ViewMode {
    VIEW_MODE.with(|m| m.get())
}

/// The one writer of [`VIEW_MODE`], so "the current mode is always one this build
/// can render" is enforced in a single place rather than re-checked by every
/// caller. Returns `false` — leaving the current mode **untouched** — if the mode
/// is unavailable. Private and widget-free: it is the availability gate the persist
/// path depends on, so it stays testable with no display. Main-thread only.
fn store_view_mode(mode: ViewMode) -> bool {
    if !mode.is_available() {
        return false;
    }
    VIEW_MODE.with(|m| m.set(mode));
    true
}

/// Apply `mode` to the lighttable — reconfiguring the grid in place for the
/// GridView-backed layouts (never swapping the `ScrolledWindow`'s child, see
/// [`ViewMode`]), or swapping the overlay's visible surface for zoomable. Separate
/// from the state write so the startup path can re-apply the restored mode to a
/// freshly built lighttable without going through the switcher's handlers.
///
/// `FileManager` is how the grid is built, so entering it means undoing culling.
/// Returns whether the layout was actually applied — a mode that can't configure
/// the view must be *refused*, not half-applied with its button lit (see
/// [`set_view_mode`]).
fn reconfigure_grid_for(grid: &GridView, zoom: &zoomable::ZoomableCanvas, mode: ViewMode) -> bool {
    match mode {
        ViewMode::FileManager | ViewMode::Culling => {
            zoom.layer.set_visible(false);
            if mode == ViewMode::Culling {
                enter_culling(grid)
            } else {
                leave_culling(grid)
            }
        }
        ViewMode::Zoomable => {
            // Leave culling FIRST so the selection model is the plain full
            // collection the canvas indexes into (its items carry base-model
            // indices; clicking through a window would select the wrong image).
            leave_culling(grid);
            zoom.on_enter();
            zoom.layer.set_visible(true);
            true
        }
    }
}

// ── Culling (m4-98c b) ─────────────────────────────────────────────────────
//
// darktable's culling layout: instead of scrolling thumbnails, one screenful of
// images at a time, paged with ← / →. Implemented as **the same `GridView` with a
// `SliceListModel` window over the same base model** — so the entire cell factory
// (thumbnail, filename, stars, colour dots, overlay modes) and every gesture on it
// keep working, which a hand-rolled comparison widget would not.
//
// The window is swapped *inside the existing `SingleSelection`* rather than by
// installing a new selection model on the grid. That matters: `selected_path` /
// `reselect_path` and the keyboard shortcuts all read `selection.model()`, so they
// follow the window automatically and keep acting on the image the user can
// actually see. Installing a second selection object would leave every one of them
// reading a stale one — silently, and only for culling.

/// How many images culling shows at once, as a function of the thumb-size stepper
/// (`max_columns`) — so the control the user already has for thumbnail size also
/// sets the comparison-set size, as in darktable. Clamped: below `MIN` there is
/// nothing to compare, and above `MAX` the cells fall under their natural width and
/// the row overflows.
const CULL_MIN_IMAGES: u32 = 2;
const CULL_MAX_IMAGES: u32 = 8;

fn cull_window_size(max_columns: u32) -> u32 {
    max_columns.clamp(CULL_MIN_IMAGES, CULL_MAX_IMAGES)
}

/// Vertical space inside a culling cell that is *not* image: the filename caption,
/// the star row, the colour-dot row, their spacings and the cell's CSS padding.
/// Empirical (m4-132), same spirit as the old `CULL_CELL_WIDTH_PX` estimate: it
/// only decides how tall the image box is, and [`gtk4::ContentFit::Contain`]
/// letterboxes whatever doesn't fit, so being off by a few pixels costs letterbox
/// slack, not correctness.
const CULL_CELL_CHROME_PX: i32 = 84;

/// The thumbnail box size for a culling cell: `window` columns across the
/// viewport, each as tall as the viewport affords after [`CULL_CELL_CHROME_PX`].
/// `None` when any input isn't usable yet — the grid reports 0 until its first
/// allocation, and at startup the mode is restored before that first layout (callers
/// re-run on the resize tick, which is why this may legitimately say "not yet").
/// Pure.
fn cull_cell_pixels(viewport_w: i32, viewport_h: i32, window: u32) -> Option<(i32, i32)> {
    if viewport_w <= 0 || viewport_h <= 0 || window == 0 {
        return None;
    }
    let w = (viewport_w / window as i32).max(1);
    // Never collapse below thumb size: a degenerate viewport gets THUMB_SIZE cells
    // (which then wrap/clip like the pre-m4-132 grid did) rather than slivers.
    let h = (viewport_h - CULL_CELL_CHROME_PX).max(THUMB_SIZE);
    Some((w, h))
}

/// The thumbnail box size a *bind* should request right now: the culling cell
/// size when culling is live and measured, else the plain file-manager square.
/// Cells are recycled, so every bind re-applies this rather than trusting the
/// requests a previous bind left behind. Delegates to [`cell_px_for`], which
/// carries the whole decision and is unit-tested; this wrapper only supplies the
/// live thread-local inputs.
fn current_cell_px() -> (i32, i32) {
    cell_px_for(current_view_mode(), CULL_CELL_PX.with(Cell::get))
}

/// Pure core of [`current_cell_px`]: `(mode, stored culling size)` → the box a
/// cell should request. The stored size counts only while both dimensions are
/// usable — `(0, 0)` means "not measured yet" (no allocation), which falls back
/// to the file-manager square rather than collapsing every cell to nothing. Pure.
fn cell_px_for(mode: ViewMode, stored: (i32, i32)) -> (i32, i32) {
    if mode == ViewMode::Culling && stored.0 > 0 && stored.1 > 0 {
        stored
    } else {
        (THUMB_SIZE, THUMB_SIZE)
    }
}

/// Apply [`current_cell_px`] to one cell's thumbnail `Picture`, including the
/// content fit: `Cover` in the file manager (the familiar square-cropped thumbs)
/// but `Contain` in culling — comparison exists to judge whole frames, so cropping
/// edges off would defeat the point.
fn apply_cell_size(thumb: &gtk4::Picture) {
    let (w, h) = current_cell_px();
    thumb.set_width_request(w);
    thumb.set_height_request(h);
    thumb.set_content_fit(if current_view_mode() == ViewMode::Culling {
        gtk4::ContentFit::Contain
    } else {
        gtk4::ContentFit::Cover
    });
}

/// Re-apply the current cell size to every realized cell of `grid`. Called after
/// any change to that size — mode switches, window resizes, stepper steps — since
/// already-bound cells don't get a fresh bind just because the viewport moved.
/// Same walk [`set_overlay_mode`] uses.
fn refresh_cull_cells(grid: &GridView) {
    for_each_cell_vbox(grid.upcast_ref::<gtk4::Widget>(), &mut |vbox| {
        if let Some(thumb) = vbox.first_child().and_downcast::<gtk4::Picture>() {
            apply_cell_size(&thumb);
        }
    });
}

/// Scale `(src_w, src_h)` to the largest size that fits inside `(box_w, box_h)`
/// preserving aspect ratio (each dimension ≥ 1). Upscaling is allowed — a small
/// source shown in a big culling cell should fill it, blurred, rather than sit in
/// a corner. `None` when any dimension isn't positive. Pure. `pub(crate)` so the
/// zoomable canvas reuses the exact same contain math (m4-139).
pub(crate) fn fit_inside(src_w: i32, src_h: i32, box_w: i32, box_h: i32) -> Option<(i32, i32)> {
    if src_w <= 0 || src_h <= 0 || box_w <= 0 || box_h <= 0 {
        return None;
    }
    let scale = f64::from(box_w) / f64::from(src_w);
    let scale = scale.min(f64::from(box_h) / f64::from(src_h));
    // Round, not truncate: a near-exact fit (box - ε) would otherwise land 1px
    // under the box and read as a sliver of letterbox for no reason.
    Some((
        ((f64::from(src_w) * scale).round() as i32).clamp(1, box_w),
        ((f64::from(src_h) * scale).round() as i32).clamp(1, box_h),
    ))
}

/// Paint a decoded thumbnail into a grid cell's Picture: packed RGB8 straight
/// into an R8G8B8 texture — no pixbuf intermediate. The pixel copy is the price
/// of painting from the shared cache's `Rc` without moving bytes out from under
/// it; at thumbnail sizes it's noise next to any decode.
fn paint_thumb(thumb: &gtk4::Picture, img: &thumbs::ThumbImage) {
    let bytes = glib::Bytes::from_owned(img.rgb.clone());
    thumb.set_paintable(Some(&gtk4::gdk::MemoryTexture::new(
        img.width,
        img.height,
        gtk4::gdk::MemoryFormat::R8g8b8,
        &bytes,
        img.width as usize * 3,
    )));
}

/// Make sure `thumb` (a grid cell's Picture) shows `path`. Cache hit paints
/// immediately; a known-failed path stays blank until the next collection load
/// ([`fill_grid`] resets the negative cache); anything else needs a bounded
/// decode. Before touching the gate this registers `(path, bucket)` in the
/// shared in-flight registry ([`thumbs::inflight_register`], m4-143): an
/// equal-or-bigger decode already running for this file — a sibling cell's
/// rebind, a scroll bounce, or the zoomable canvas — makes this call a no-op,
/// so a seconds-long raw demosaic is paid once. The gate slot itself is still
/// claimed BEFORE any task spawns; when it is busy this schedules one timed
/// retry instead of queueing, so gio's blocking pool never holds a parked
/// worker and the rating/colour-label queries sharing it can't stall behind
/// thumbnail decodes (review MAJOR, m4-140). Every entry re-checks the stamped
/// widget name: cells recycle, so a retry landing on a rebound cell is a
/// no-op — that cell's own bind runs a fresh chain.
fn ensure_grid_thumb(thumb_w: glib::WeakRef<gtk4::Picture>, path: String) {
    let Some(thumb) = thumb_w.upgrade() else { return };
    if thumb.widget_name() != path {
        return; // cell rebound mid-chain; its own bind owns the load now
    }
    // Bucket = the CURRENT cell box, quantised exactly like every other
    // consumer ([`thumbs::bucket_for`]) so pixel-cache entries cross-hit between
    // the file manager, culling and the zoomable canvas. Re-read per attempt:
    // a resize landing mid-chain just retargets the next attempt.
    let (bw, bh) = current_cell_px();
    let bucket = thumbs::bucket_for(bw.max(bh).max(THUMB_SIZE));
    if let Some(img) = thumbs::lookup(&path, bucket) {
        paint_thumb(&thumb, &img);
        return;
    }
    if thumbs::is_failed(&path) {
        return;
    }
    if !thumbs::inflight_register(&path, bucket) {
        // An equal-or-bigger decode for this path is already running. Unlike
        // the busy gate there is no completion of OURS to paint the cell
        // afterwards — the running decode may belong to the other surface,
        // whose result never reaches this cache — so without a retry the cell
        // would stay blank until some unrelated rebind (review MAJOR-1,
        // m4-143). Retry exactly like the busy gate: self-terminating once the
        // owner finishes, then this chain either cache-hits or registers.
        glib::timeout_add_local_once(
            std::time::Duration::from_millis(150),
            move || ensure_grid_thumb(thumb_w, path),
        );
        return;
    }
    let Some(permit) = thumbs::DecodePermit::try_acquire() else {
        // Both decoders busy: hand back the unused registration and try again
        // shortly instead of waiting anywhere.
        thumbs::inflight_unregister(&path, bucket);
        glib::timeout_add_local_once(
            std::time::Duration::from_millis(150),
            move || ensure_grid_thumb(thumb_w, path),
        );
        return;
    };
    glib::spawn_future_local(async move {
        let p = path.clone();
        let decoded =
            gio::spawn_blocking(move || thumbs::decode_with_permit(permit, &p, bucket)).await;
        // This decode's registration ends here unless a superseding bigger
        // request took over the path meanwhile (the owner check is inside).
        thumbs::inflight_unregister(&path, bucket);
        match decoded {
            Ok(Some(img)) => {
                let cached = thumbs::store(&path, bucket, img);
                if let Some(t) = thumb_w.upgrade() {
                    if t.widget_name() == path {
                        paint_thumb(&t, &cached);
                    }
                }
            }
            Ok(None) => thumbs::mark_failed(&path),
            Err(_) => {
                // A panicked decoder is a bug, not a corrupt file — leave the
                // path retryable instead of negative-caching it blank.
                eprintln!("c41-thumbs: grid decode task panicked for {path}");
            }
        }
    });
}

/// The culling window for `grid` right now: what the stepper asks for. Since
/// m4-132 the cells resize to fit the viewport, so every clamped window size fits
/// by construction — the old "cap the window to what 180px cells could fill" logic
/// is gone along with its wrapping failure mode.
fn cull_effective_window(grid: &GridView) -> u32 {
    cull_window_size(grid.max_columns())
}

/// Where a page step lands. Paging forward stops on the last whole page rather than
/// walking off the end: an offset at or past `n_items` renders as an **empty grid**
/// with no error, which is the failure this repo keeps meeting. Pure.
fn cull_page_offset(offset: u32, n_items: u32, window: u32, forward: bool) -> u32 {
    let window = window.max(1);
    if forward {
        let next = offset.saturating_add(window);
        // Only step if the next page has something in it; otherwise stay put.
        if next < n_items {
            next
        } else {
            offset
        }
    } else {
        offset.saturating_sub(window)
    }
}

/// Pull an offset back into a collection that may have shrunk under it (a filter,
/// a folder switch, an import). An offset still inside the collection is passed
/// through untouched — the window slides, it is not re-aligned to a page grid —
/// and only one that fell off the end snaps back to the last whole page start.
/// Pure.
fn cull_clamp_offset(offset: u32, n_items: u32, window: u32) -> u32 {
    if n_items == 0 || offset < n_items {
        return if n_items == 0 { 0 } else { offset };
    }
    let window = window.max(1);
    ((n_items - 1) / window) * window
}

/// The grid's *unwindowed* model: the slice's base while culling, the selection's
/// own model otherwise. Entering culling twice must not stack a window on a window.
pub(crate) fn cull_base_model(selection: &SingleSelection) -> Option<gtk4::gio::ListModel> {
    let model = selection.model()?;
    match model.downcast::<gtk4::SliceListModel>() {
        Ok(slice) => slice.model(),
        Err(model) => Some(model),
    }
}

/// The page start that brings image `index` on screen — culling enters on the page
/// holding the selected image, as darktable does, rather than wherever the user
/// last left the window. Pure.
fn cull_entry_offset(index: u32, window: u32) -> u32 {
    let window = window.max(1);
    (index / window) * window
}

/// Every path in `model`, in order. Used to locate an image by path when the index
/// spaces differ (the window's vs the collection's), and by the zoomable canvas
/// to read the collection it paints (m4-139).
pub(crate) fn model_paths(model: &gtk4::gio::ListModel) -> Vec<String> {
    (0..model.n_items())
        .filter_map(|i| {
            model.item(i).and_downcast::<gtk4::StringObject>().map(|o| o.string().to_string())
        })
        .collect()
}

fn enter_culling(grid: &GridView) -> bool {
    let Some(selection) = grid.model().and_downcast::<SingleSelection>() else {
        tracing::warn!("culling: grid has no SingleSelection; staying in file manager");
        return false;
    };
    let Some(base) = cull_base_model(&selection) else {
        tracing::warn!("culling: grid selection has no model; staying in file manager");
        return false;
    };

    // Carry the selection across the model swap: `set_model` resets a
    // SingleSelection to index 0, so without this, entering culling would silently
    // drop whatever the user had picked.
    let previous = selected_path(&selection);

    let window = cull_effective_window(grid);
    // Open on the selected image's page when there is one; otherwise resume where
    // the user last left the window.
    let offset = match previous
        .as_deref()
        .and_then(|p| index_of_path(&model_paths(&base), p))
    {
        Some(idx) => cull_entry_offset(idx, window),
        None => cull_clamp_offset(CULL_OFFSET.with(|o| o.get()), base.n_items(), window),
    };
    CULL_OFFSET.with(|o| o.set(offset));

    let slice = gtk4::SliceListModel::new(Some(base), offset, window);
    // A reload under the window (filter, folder, import) can leave the offset past
    // the end, which renders as an empty grid and looks like a hung lighttable.
    // Re-clamp on every change of the base model; the handler is tracked so
    // re-entering culling can't stack a second one.
    watch_base_for_cull_clamp(&slice);

    selection.set_model(Some(&slice));
    reselect_path(&selection, previous.as_deref());

    // Pin exactly one row of `window` cells sized to fill the viewport (m4-132).
    // Pinning min=max is what makes the fill layout safe: with both bounds at the
    // window size, the grid always lays out that single row and the *cells* absorb
    // narrow viewports by shrinking — which is only possible because the cell size
    // below is derived from the viewport instead of being a fixed THUMB_SIZE. The
    // pre-m4-132 layout couldn't pin (fixed-size cells + no horizontal scrollbar
    // turned "one row" into "clipped"), which is why it capped the window instead.
    let had_saved = CULL_SAVED_MAX_COLS.with(|m| m.get()) != 0;
    if !had_saved {
        CULL_SAVED_MAX_COLS.with(|m| m.set(grid.max_columns()));
    }
    // Max first so the pair is never transiently inverted (min > max).
    grid.set_max_columns(window);
    grid.set_min_columns(window);

    // Cell size from the current allocation. Before the first allocation this
    // stores nothing (`cull_cell_pixels` → None) and binds fall back to
    // THUMB_SIZE until the resize tick fills it in.
    if let Some(px) = cull_cell_pixels(grid.width(), grid.height(), window) {
        CULL_CELL_PX.with(|c| c.set(px));
    }
    refresh_cull_cells(grid);
    true
}

fn leave_culling(grid: &GridView) -> bool {
    // Unconditionally, and first: a watch left connected would clamp an offset for
    // a base nobody is showing and keep it alive in a thread-local forever.
    unwatch_base_for_cull_clamp();
    // Drop the fill sizing FIRST so the walk below restores the realized cells to
    // their file-manager square rather than re-applying culling sizes.
    CULL_CELL_PX.with(|c| c.set((0, 0)));
    let Some(selection) = grid.model().and_downcast::<SingleSelection>() else {
        return true; // nothing windowed, so nothing to unwind
    };
    // Consumed only once the unwind is actually proceeding — bailing above must
    // not throw away the manager bound the eventual real leave will restore.
    let saved_max = CULL_SAVED_MAX_COLS.with(Cell::take);
    if let Some(slice) = selection.model().and_downcast::<gtk4::SliceListModel>() {
        if let Some(base) = slice.model() {
            let previous = selected_path(&selection);
            selection.set_model(Some(&base));
            reselect_path(&selection, previous.as_deref());
        }
    }
    // Restore both column bounds: enter pinned max too (the file manager keeps its
    // own chosen thumb size across a culling visit, like darktable). The `.max`
    // covers unwinding without a matching enter (nothing saved): fall back to the
    // startup default rather than a degenerate bound.
    grid.set_min_columns(GRID_MIN_COLUMNS);
    grid.set_max_columns(saved_max.max(THUMB_COLS_DEFAULT));
    refresh_cull_cells(grid);
    true
}

/// Keep the culling window inside the collection as the collection changes under
/// it. Replaces any previous watch, so entering culling repeatedly can't leave a
/// pile of handlers clamping the same offset.
fn watch_base_for_cull_clamp(slice: &gtk4::SliceListModel) {
    unwatch_base_for_cull_clamp();
    let Some(base) = slice.model() else { return };
    let id = base.connect_items_changed({
        let slice = slice.clone();
        move |base, _, _, _| {
            let window = slice.size();
            let clamped = cull_clamp_offset(slice.offset(), base.n_items(), window);
            if clamped != slice.offset() {
                slice.set_offset(clamped);
            }
            CULL_OFFSET.with(|o| o.set(clamped));
        }
    });
    CULL_BASE_WATCH.with(|w| *w.borrow_mut() = Some((base, id)));
}

fn unwatch_base_for_cull_clamp() {
    if let Some((base, id)) = CULL_BASE_WATCH.with(|w| w.borrow_mut().take()) {
        base.disconnect(id);
    }
}

/// Page the culling window by one screenful. Returns whether the key belonged to
/// culling at all — `false` means "not culling, handle this normally", so the
/// caller must not swallow arrow keys in the file manager.
pub fn cull_step(grid: &GridView, forward: bool) -> bool {
    if current_view_mode() != ViewMode::Culling {
        return false;
    }
    let Some(slice) = grid
        .model()
        .and_downcast::<SingleSelection>()
        .and_then(|s| s.model())
        .and_downcast::<gtk4::SliceListModel>()
    else {
        // Culling is the current mode but no window is installed: paging would be a
        // silent no-op, so say so rather than swallowing the key.
        tracing::warn!("culling: no window installed; arrow keys left to the grid");
        return false;
    };
    let n_items = slice.model().map_or(0, |m| m.n_items());
    let next = cull_page_offset(slice.offset(), n_items, slice.size(), forward);
    if next != slice.offset() {
        slice.set_offset(next);
        CULL_OFFSET.with(|o| o.set(next));
    }
    true
}

/// Which way (if either) a key pages the culling window. Pure, so the key mapping
/// is testable without a display. Both the arrows and Page Up/Down page by a whole
/// screenful — there is no within-page cursor in culling.
pub fn cull_key_direction(keyval: gtk4::gdk::Key) -> Option<bool> {
    use gtk4::gdk::Key;
    match keyval {
        Key::Right | Key::Page_Down => Some(true),
        Key::Left | Key::Page_Up => Some(false),
        _ => None,
    }
}

/// Re-fit culling to the current viewport and stepper value: re-pins the column
/// bounds, recomputes the per-cell fill size (`cull_cell_pixels`) onto realized
/// cells, and resizes the installed window when the count changed. Runs on every
/// viewport resize tick (both page-size notifies) as well as after stepper steps,
/// so most invocations only refresh cell pixels. A no-op outside culling.
///
/// Resizes the installed `SliceListModel` in place rather than building a new one:
/// `SingleSelection::set_model` resets the selection to index 0, so rebuilding on
/// every ± click would throw away the user's pick.
pub fn cull_resync(grid: &GridView) {
    if current_view_mode() != ViewMode::Culling {
        return;
    }
    let Some(slice) = grid
        .model()
        .and_downcast::<SingleSelection>()
        .and_then(|s| s.model())
        .and_downcast::<gtk4::SliceListModel>()
    else {
        // Culling with no window installed: enter properly rather than no-op.
        enter_culling(grid);
        return;
    };
    let window = cull_effective_window(grid);

    // Keep the fill sizing in sync BEFORE the early return below: this runs on
    // every viewport resize (the hadjustment/vadjustment page-size ticks), and a
    // resize that doesn't change the window count still changes the cell pixels.
    // (No save of `max_columns` here — by this point `enter_culling` has always
    // run and captured the file-manager bound; reading it now would read the
    // pinned value instead.)
    // Max first so the pair is never transiently inverted (min > max).
    grid.set_max_columns(window);
    grid.set_min_columns(window);
    if let Some(px) = cull_cell_pixels(grid.width(), grid.height(), window) {
        CULL_CELL_PX.with(|c| c.set(px));
    }
    refresh_cull_cells(grid);

    if window == slice.size() {
        return;
    }
    slice.set_size(window);
    let n_items = slice.model().map_or(0, |m| m.n_items());
    let offset = cull_clamp_offset(slice.offset(), n_items, window);
    if offset != slice.offset() {
        slice.set_offset(offset);
    }
    CULL_OFFSET.with(|o| o.set(offset));
}

/// What the thumb-size stepper should show and allow while culling:
/// `(images on screen, lowest, highest)`. `None` outside culling, meaning "use the
/// stepper's own range".
///
/// Since m4-132 every clamped window size is fully visible — the cells resize to
/// fill the viewport instead of wrapping — so the window always equals what the
/// stepper asked for and the range is simply the comparison-set bounds. The old
/// viewport-cap dead zone (steps that counted up on the label while nothing moved)
/// is gone by construction; a control that looks live and isn't is this repo's
/// recurring bug shape.
pub fn cull_stepper_state(grid: &GridView) -> Option<(u32, u32, u32)> {
    (current_view_mode() == ViewMode::Culling)
        .then(|| (cull_effective_window(grid), CULL_MIN_IMAGES, CULL_MAX_IMAGES))
}

/// Switch the lighttable layout: write the mode, then reconfigure the view for
/// it. An unavailable mode is refused rather than half-applied — and the caller
/// learns that from the `false` return, so it neither persists nor keeps
/// displaying a mode that was never entered. Main-thread only.
#[must_use]
pub fn set_view_mode(
    grid: &GridView,
    zoom: &zoomable::ZoomableCanvas,
    mode: ViewMode,
) -> bool {
    let previous = current_view_mode();
    if !store_view_mode(mode) {
        return false;
    }
    if !reconfigure_grid_for(grid, zoom, mode) {
        // The layout didn't take (a view we can't configure). Put the mode back
        // rather than leaving the state claiming a layout that isn't on screen —
        // the caller then rolls its button back and persists nothing.
        store_view_mode(previous);
        // Undo any layer flip the failed arm already performed, so the widgets
        // match the restored mode exactly.
        zoom.layer.set_visible(previous == ViewMode::Zoomable);
        return false;
    }
    true
}

/// Show/hide a cell's three metadata rows — `(filename, stars, colours)`, as
/// returned by [`effective_overlay_visibility`]. The thumbnail (child 0) is never
/// touched. Applied at bind time so every newly-bound *and* recycled cell honours
/// the mode, and by [`set_overlay_mode`] to the cells already on screen.
fn apply_overlay_visibility(vbox: &gtk4::Box, (filename, stars, colors): (bool, bool, bool)) {
    for (idx, visible) in [(1usize, filename), (2, stars), (3, colors)] {
        if let Some(w) = nth_child(vbox, idx) {
            w.set_visible(visible);
        }
    }
}

/// If `w` is a realized grid cell, its vbox. A cell is recognised the same way
/// [`find_cell_row_for_path`] does it — its first child is the thumbnail
/// `Picture` — which also can't match the cell's own metadata rows (those are
/// `Box`es leading with `Label`s).
fn cell_vbox_of(w: &gtk4::Widget) -> Option<gtk4::Box> {
    let vbox = w.downcast_ref::<gtk4::Box>()?;
    vbox.first_child().and_downcast::<gtk4::Picture>()?;
    Some(vbox.clone())
}

/// Depth-first walk of `root`'s descendants applying `f` to every realized cell
/// vbox. Identified cells are not descended into — their children are the
/// metadata rows, never nested cells.
fn for_each_cell_vbox(root: &gtk4::Widget, f: &mut dyn FnMut(&gtk4::Box)) {
    let mut child = root.first_child();
    while let Some(w) = child {
        // NOTE: the call to `f` is the point of this walk — keep it in a plain
        // `if let` so it stays visibly load-bearing. (An `Option` combinator chain
        // here reads as a no-op and invites a "simplification" that would silently
        // turn `set_overlay_mode` into a state-setter that repaints nothing.)
        if let Some(vbox) = cell_vbox_of(&w) {
            f(&vbox);
        } else {
            for_each_cell_vbox(&w, f);
        }
        child = w.next_sibling();
    }
}

/// Whether a realized cell is showing a placeholder row, judged by the path
/// stamped on its thumbnail at bind time (placeholders carry no `/`). A cell that
/// has never been bound reports GTK's type-name fallback (`"GtkPicture"`), which
/// likewise has no `/` — so it too counts as a placeholder and keeps its label
/// until its first bind. That's the fail-open direction.
fn cell_is_placeholder(vbox: &gtk4::Box) -> bool {
    vbox.first_child()
        .and_downcast::<gtk4::Picture>()
        .is_some_and(|t| !t.widget_name().contains('/'))
}

/// Set the thumbnail overlay mode and apply it immediately to the cells already
/// realized in `grid` (new/recycled cells pick it up at bind time). Main-thread
/// only.
pub fn set_overlay_mode(grid: &GridView, mode: OverlayMode) {
    OVERLAY_MODE.with(|m| m.set(mode));
    for_each_cell_vbox(grid.upcast_ref::<gtk4::Widget>(), &mut |vbox| {
        let vis = effective_overlay_visibility(cell_is_placeholder(vbox), mode);
        apply_overlay_visibility(vbox, vis);
    });
}

/// A trailing ` AND (...)` SQL fragment implementing the rating filter, or `""`
/// when no filter is active (`AtLeast` + 0 stars). darktable keeps the 0..5 star
/// value in bits 0–2 of `flags` and the reject state in the separate bit 3 (= 8),
/// so each non-`Rejected` comparator excludes rejected images first, then bounds
/// the star value; `Rejected` matches the reject bit and ignores the star count.
/// `stars` is clamped to 0..=5 so the interpolated integer is never user text
/// (injection-safe); every loader aliases images as `i`, so `i.flags` is valid
/// wherever this splices in. The `= N`/`BETWEEN 0 AND N` bounds (N ≤ 5) also drop
/// any legacy `flags & 7` of 6/7, keeping the whole path on the 0..5 domain.
fn rating_predicate(stars: u8, cmp: RatingCompare) -> String {
    // Derive the masks from the named layout constants (they render as the literals
    // 8 and 7) so the SQL can't drift from the documented bit scheme.
    let rej = DT_IMAGE_REJECTED;
    let mask = DT_VIEW_RATINGS_MASK;
    let s = stars.min(5);
    match cmp {
        RatingCompare::Rejected => format!(" AND (i.flags & {rej}) = {rej}"),
        // ≥ 0 stars is "everything" — the canonical no-filter state.
        RatingCompare::AtLeast if s == 0 => String::new(),
        RatingCompare::AtLeast => {
            format!(" AND (i.flags & {rej}) = 0 AND (i.flags & {mask}) BETWEEN {s} AND 5")
        }
        RatingCompare::Exactly => {
            format!(" AND (i.flags & {rej}) = 0 AND (i.flags & {mask}) = {s}")
        }
        RatingCompare::AtMost => {
            format!(" AND (i.flags & {rej}) = 0 AND (i.flags & {mask}) BETWEEN 0 AND {s}")
        }
    }
}

/// The rating-filter SQL fragment for the *current* comparator + star count —
/// what the loaders splice into their WHERE.
fn current_rating_sql() -> String {
    rating_predicate(current_min_rating(), current_rating_compare())
}

/// The year range the timeline is filtering to now (`None` = all years).
pub fn current_year_range() -> Option<(i32, i32)> {
    YEAR_RANGE.with(|r| r.get())
}

/// The colour quick-filter mask live right now (bit `c` = colour `c`, 0 = none).
/// `pub` so the left panel and both bars' circles can seed/refresh their displays.
pub fn current_colour_mask() -> u8 {
    COLOUR_MASK.with(|m| m.get())
}

/// Whether the colour filter combines several selected colours with AND (`true`)
/// or OR (`false`). `pub` so the left panel's Any/All toggle can seed itself.
pub fn current_colour_all() -> bool {
    COLOUR_ALL.with(|a| a.get())
}

/// Set the colour-label quick filter (`mask` bit `c` = colour `c`, 0 = no colour
/// restriction; `match_all` = require every selected colour) and re-render the
/// current view under it. Composes with whatever collection is active, like the
/// rating filter. Main-thread only.
pub fn set_colour_filter(mask: u8, match_all: bool) {
    COLOUR_MASK.with(|m| m.set(mask & 0x1F));
    COLOUR_ALL.with(|a| a.set(match_all));
    filter_changed();
}

/// The colour quick-filter as one trailing ` AND …` fragment over the row's `id`
/// column — a correlated subquery counting which of the selected colours the image
/// carries. `>= 1` is OR semantics ("any"), `= N` is AND ("all"). Empty when
/// `mask` is 0 (the no-filter state), so splicing it is always safe. Pure, so the
/// fragment shape is unit-testable without a display.
///
/// References `main.color_labels` unconditionally, so it must only splice into
/// queries run against a real catalogue (which always has the table — darktable's
/// schema); the in-memory demo db does not, but it also has no prefs store, so a
/// persisted filter can never be restored there.
fn colour_predicate(mask: u8, match_all: bool) -> String {
    let colors = colors_from_mask(mask);
    if colors.is_empty() {
        return String::new();
    }
    let in_list = colors.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
    let n = colors.len();
    // ANY = at least one of the chosen colours; ALL = every one of them.
    let (op, bound) = if match_all { ("=", n) } else { (">=", 1) };
    format!(
        " AND ((SELECT COUNT(DISTINCT cl.color) FROM main.color_labels cl \
         WHERE cl.imgid = i.id AND cl.color IN ({in_list})) {op} {bound})"
    )
}

/// The aspect-ratio quick-filter live right now. `pub` so the left panel can seed
/// its dropdown from the restored value.
pub fn current_aspect_filter() -> AspectFilter {
    ASPECT_FILTER.with(|f| f.get())
}

/// Set the aspect-ratio quick filter and re-render the current view under it.
/// Composes with whatever collection is active, like the rating filter.
/// Main-thread only.
pub fn set_aspect_filter(f: AspectFilter) {
    ASPECT_FILTER.with(|a| a.set(f));
    filter_changed();
}

/// The aspect quick-filter as one trailing ` AND …` fragment over the row's
/// import-time `width`/`height` columns. Empty for [`AspectFilter::Off`] (the
/// no-filter state), so splicing it is always safe. Pure, so the fragment shape
/// is unit-testable without a display.
///
/// Integer comparisons stand in for darktable's float-range presets: `w > h` ⇔
/// ratio > 1 (landscape), `h > w` ⇔ portrait, `w == h && w > 0` ⇔ square. The
/// `w > 0` guard matters — a failed dimension probe leaves 0×0, which would
/// otherwise satisfy `=` and silently match every broken row as square; with it,
/// unknown-dimension images match no aspect at all, mirroring how they carry no
/// usable ratio in darktable either.
///
/// Two documented deviations vs darktable: (1) its stored ratio is snapped by
/// `dt_usable_aspect` with a ±0.5 % tolerance band around each canonical value,
/// so e.g. a true 1.004 ratio stores as exactly 1.0 and its `square` preset
/// (`[1;1]`) catches it, while our exact comparison classes that row landscape —
/// the divergence window is only that 0.5 % band around each boundary; we trade
/// the tolerance for not materialising and snapping a float column at all.
/// (2) its ratio uses the *developed* preview dimensions (`p_width`/
/// `p_height`), so an image cropped across the sensor orientation still counts
/// by its crop there and by its raw sensor here — we have no materialised
/// post-crop ratio column yet.
fn aspect_predicate(f: AspectFilter) -> &'static str {
    match f {
        AspectFilter::Off => "",
        AspectFilter::Landscape => " AND i.width > i.height",
        AspectFilter::Portrait => " AND i.height > i.width",
        AspectFilter::Square => " AND i.width = i.height AND i.width > 0",
    }
}

/// The rule stack (m4-134) as one fragment, via [`rule_stack::rule_stack_sql`].
/// Empty when no rule has a value; the combinators stay parenthesized inside,
/// so splicing is safe wherever the other filters' fragments go.
fn rule_stack_fragment() -> String {
    let stack = RULE_STACK.with(|s| s.borrow().clone());
    rule_stack::rule_stack_sql(&stack)
}

/// **Every** compose-on-top quick filter as one trailing ` AND …` fragment: the
/// rating filter (m4-98b/d), the timeline's year range (m4-99), the colour
/// quick filter (m4-126), and the aspect-ratio rule (m4-128). Loaders splice this
/// single string, so adding a filter never means touching them again — and they
/// can't accidentally apply one but not another.
fn current_filters_sql() -> String {
    format!(
        "{}{}{}{}{}",
        current_rating_sql(),
        timeline::year_range_and(current_year_range()),
        colour_predicate(current_colour_mask(), current_colour_all()),
        aspect_predicate(current_aspect_filter()),
        rule_stack_fragment(),
    )
}

/// The rule stack's live state — a clone, so callers can compare against what
/// their widgets show without holding the [`RULE_STACK`] borrow.
pub fn current_rule_stack() -> Vec<rule_stack::Rule> {
    RULE_STACK.with(|s| s.borrow().clone())
}

/// Replace the whole stack and re-render the current view under it, through the
/// same observer bus as every other filter (so the persistence observer in
/// `lib.rs` fires and all controls re-sync). Main-thread only. Blank-valued
/// rules are legal state — [`rule_stack::rule_stack_sql`] skips them.
pub fn set_rule_stack(stack: Vec<rule_stack::Rule>) {
    RULE_STACK.with(|s| *s.borrow_mut() = stack);
    filter_changed();
}

/// Persisted token pieces for the rule stack — same one-source-of-truth shape
/// as the rating/aspect tokens.
pub const RULE_STACK_PREF_KEY: &str = "rule_stack";

/// Encode the live stack for persistence. Pure passthrough to
/// [`rule_stack::rule_stack_token_for`].
pub fn rule_stack_token() -> String {
    rule_stack::rule_stack_token_for(&current_rule_stack())
}

/// Restore the stack from a persisted token (lenient; garbage ⇒ empty). Called
/// once at startup BEFORE any control is built, like the other tokens — a
/// programmatic write with no handler connected fires nothing.
pub fn apply_rule_stack_token(tok: &str) {
    let parsed = rule_stack::parse_rule_stack_token(tok);
    if !parsed.is_empty() {
        RULE_STACK.with(|s| *s.borrow_mut() = parsed);
    }
}

/// Set the timeline's year-range filter (`None` clears it) and re-render the
/// current view. Goes through the same observer bus as the rating filter, so the
/// timeline's own highlight and any other control stay in sync. Main-thread only.
pub fn set_year_range(range: Option<(i32, i32)>) {
    YEAR_RANGE.with(|r| r.set(range));
    filter_changed();
}

// ── Named collection presets (parity 2.6 close, m4-136) ─────────────────────
//
// A preset captures the WHOLE filter set as one payload string: version tag,
// then the five persisted tokens in fixed order — rating, colour, aspect,
// year-range, rule stack — whitespace-separated. Fixed positions keep the
// format self-describing without inventing a key=value grammar; every
// component token is already space-free (the rule token is pct-encoded by
// construction), and each component decoder is lenient on its own, so a
// corrupt field degrades to no-filter rather than poisoning the rest.
//
// The year range has NO persistence token of its own (session-only state), so
// `off` / `<from>:<to>` here is its first codec.

/// Sentinel for the rules field when the stack is empty. An empty STRING would
/// vanish under the payload's whitespace splitting (five fields instead of
/// six); a real stack's token always carries its `:`/`/` separators, so a bare
/// `-` can never collide with one.
const COLLECTION_RULES_EMPTY: &str = "-";

/// Encode the current filter state as a collection-preset payload (see the
/// module-group comment above for the format).
pub fn collection_filter_payload() -> String {
    let year = match current_year_range() {
        Some((a, b)) => format!("{a}:{b}"),
        None => "off".to_string(),
    };
    let rules = rule_stack_token();
    let rules = if rules.is_empty() { COLLECTION_RULES_EMPTY } else { &rules };
    format!(
        "v1 {} {} {} {} {}",
        rating_filter_token(),
        colour_filter_token(),
        aspect_filter_token(),
        year,
        rules,
    )
}

/// One parsed preset payload: the five component tokens plus the decoded year
/// range. Component tokens stay strings — their own decoders own the leniency.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectionFilterState {
    pub rating: String,
    pub colour: String,
    pub aspect: String,
    pub year: Option<(i32, i32)>,
    pub rules: String,
}

/// Pure decoder for a payload written by [`collection_filter_payload`].
/// Strict about STRUCTURE (version tag, six fields, ordered years) because a
/// structurally different payload means a writer we can't reason about;
/// lenient about CONTENT, because every component token's own parser already
/// falls back to no-filter on garbage.
pub fn parse_collection_payload(payload: &str) -> Option<CollectionFilterState> {
    let fields: Vec<&str> = payload.split_whitespace().collect();
    if fields.len() != 6 || fields[0] != "v1" {
        return None;
    }
    let year = if fields[4] == "off" {
        None
    } else {
        let (a, b) = fields[4].split_once(':')?;
        let (a, b) = (a.parse::<i32>().ok()?, b.parse::<i32>().ok()?);
        if a > b {
            return None;
        }
        Some((a, b))
    };
    Some(CollectionFilterState {
        rating: fields[1].to_string(),
        colour: fields[2].to_string(),
        aspect: fields[3].to_string(),
        year,
        // Normalise the empty-stack sentinel so callers see "" — the same
        // representation [`current_rule_stack`] implies for no rules.
        rules: {
            let r = fields[5].to_string();
            if r == COLLECTION_RULES_EMPTY { String::new() } else { r }
        },
    })
}

/// Apply a saved preset's payload, landing the whole filter set as ONE view
/// reload: end state is exactly what hand-clicking each control would produce
/// (controls re-sync via the bus, the grid reloads, the per-key persistence
/// observers rewrite the live pref tokens — an applied preset BECOMES current
/// state), but without firing the bus once per setter. Rating/colour/aspect
/// already apply silently; year and the rule stack are written through their
/// silent in-module paths too (their pub setters each fire [`filter_changed`],
/// which would otherwise show an intermediate half-applied grid — senior-review
/// MINOR-2, m4-136). Returns false (applying nothing) when the payload doesn't
/// parse at all.
pub fn apply_collection_payload(payload: &str) -> bool {
    let Some(state) = parse_collection_payload(payload) else {
        return false;
    };
    apply_rating_filter_token(&state.rating);
    apply_colour_filter_token(&state.colour);
    apply_aspect_filter_token(&state.aspect);
    YEAR_RANGE.with(|r| r.set(state.year));
    if state.rules.is_empty() {
        // `apply_rule_stack_token` deliberately ignores empty parses ("garbage
        // must not clobber"), but a preset whose stack is EMPTY must clear the
        // live one — recalling a no-rules preset while rules are active.
        RULE_STACK.with(|s| *s.borrow_mut() = Vec::new());
    } else {
        apply_rule_stack_token(&state.rules);
    }
    filter_changed();
    true
}

/// Persisted rating-filter token pieces — one source of truth shared by the
/// encoder ([`rating_filter_token_for`]) and decoder ([`parse_rating_filter_token`])
/// so a prefix typo can't make them silently disagree.
const RATING_TOK_OFF: &str = "off";
const RATING_TOK_REJ: &str = "rej";
const RATING_TOK_GE: &str = "ge";
const RATING_TOK_EQ: &str = "eq";
const RATING_TOK_LE: &str = "le";

/// Pure encoder: a compact, stable token for `(comparator, star count)` — `off`,
/// `ge:N`, `eq:N`, `le:N`, or `rej`. `Rejected` encodes to `rej` and **drops** the
/// star count (it's irrelevant in that mode); consequently a session left in
/// `Rejected` restores as no-filter (`off`) rather than restoring the pre-reject
/// count — the retained in-session count is intentionally not persisted.
fn rating_filter_token_for(cmp: RatingCompare, stars: u8) -> String {
    let s = stars.min(5);
    match cmp {
        RatingCompare::Rejected => RATING_TOK_REJ.to_string(),
        RatingCompare::AtLeast if s == 0 => RATING_TOK_OFF.to_string(),
        RatingCompare::AtLeast => format!("{RATING_TOK_GE}:{s}"),
        RatingCompare::Exactly => format!("{RATING_TOK_EQ}:{s}"),
        RatingCompare::AtMost => format!("{RATING_TOK_LE}:{s}"),
    }
}

/// Encode the *current* rating filter as a persistence token (see
/// [`apply_rating_filter_token`]). `pub` so `lib.rs` — which holds the db path —
/// can store it.
pub fn rating_filter_token() -> String {
    rating_filter_token_for(current_rating_compare(), current_min_rating())
}

/// Parse a persisted rating-filter token back into `(comparator, star count)`,
/// clamping the count to 0..=5 and falling back to the no-filter state on any
/// unrecognised/corrupt token. Pure, so it's unit-testable.
fn parse_rating_filter_token(tok: &str) -> (RatingCompare, u8) {
    if tok == RATING_TOK_REJ {
        return (RatingCompare::Rejected, 0);
    }
    if let Some((pfx, val)) = tok.split_once(':') {
        if let Ok(n) = val.parse::<u8>() {
            let n = n.min(5);
            if pfx == RATING_TOK_GE {
                return (RatingCompare::AtLeast, n);
            } else if pfx == RATING_TOK_EQ {
                return (RatingCompare::Exactly, n);
            } else if pfx == RATING_TOK_LE {
                return (RatingCompare::AtMost, n);
            }
        }
    }
    (RatingCompare::AtLeast, 0) // "off" and sibling tokens' fallback ⇒ no filter
}

/// Colour quick-filter token pieces — same compact scheme as the rating filter's
/// ([`RATING_TOK_OFF`]): `off`, or `any:M` / `all:M` with the 5-bit mask in decimal.
const COLOUR_TOK_OFF: &str = "off";
const COLOUR_TOK_ANY: &str = "any:";
const COLOUR_TOK_ALL: &str = "all:";

/// Pure encoder for `(mask, match_all)` — `off`, `any:M`, or `all:M`. Mask 0
/// canonicalises to `off` regardless of mode (mode with no colours selected is not
/// a state worth persisting).
fn colour_filter_token_for(mask: u8, all: bool) -> String {
    let mask = mask & 0x1F;
    if mask == 0 {
        return COLOUR_TOK_OFF.to_string();
    }
    format!("{}{mask}", if all { COLOUR_TOK_ALL } else { COLOUR_TOK_ANY })
}

/// Parse a persisted colour-filter token back into `(mask, all)`, clamping to the
/// 5-bit domain and falling back to the no-filter state on anything unrecognised.
/// Pure, so it's unit-testable.
fn parse_colour_filter_token(tok: &str) -> (u8, bool) {
    if tok == COLOUR_TOK_OFF {
        return (0, false);
    }
    if let Some(rest) = tok.strip_prefix(COLOUR_TOK_ANY) {
        if let Ok(m) = rest.parse::<u8>() {
            return (m & 0x1F, false);
        }
    }
    if let Some(rest) = tok.strip_prefix(COLOUR_TOK_ALL) {
        if let Ok(m) = rest.parse::<u8>() {
            return (m & 0x1F, true);
        }
    }
    (0, false)
}

/// Encode the *current* colour quick-filter as its persistence token. `pub` so
/// `lib.rs` — which holds the db path — can store it.
pub fn colour_filter_token() -> String {
    colour_filter_token_for(current_colour_mask(), current_colour_all())
}

/// Seed the rating filter from a persisted token *without* reloading — called at
/// startup before any view loader has registered (so the first load already
/// reflects the restored filter). Main-thread only.
pub fn apply_rating_filter_token(tok: &str) {
    let (cmp, stars) = parse_rating_filter_token(tok);
    MIN_RATING.with(|r| r.set(stars));
    RATING_COMPARE.with(|c| c.set(cmp));
}

/// Seed the colour quick-filter from a persisted token *without* reloading —
/// same startup contract as [`apply_rating_filter_token`]. Main-thread only.
pub fn apply_colour_filter_token(tok: &str) {
    let (mask, all) = parse_colour_filter_token(tok);
    COLOUR_MASK.with(|m| m.set(mask));
    COLOUR_ALL.with(|a| a.set(all));
}

/// Persisted aspect-filter tokens — the variant names themselves; there is no
/// parameterised state to compact away (unlike the rating/colour masks), so a
/// bare word per row is the whole codec. One source of truth shared by
/// [`aspect_filter_token_for`] and [`parse_aspect_filter_token`].
const ASPECT_TOK_OFF: &str = "off";
const ASPECT_TOK_LANDSCAPE: &str = "landscape";
const ASPECT_TOK_PORTRAIT: &str = "portrait";
const ASPECT_TOK_SQUARE: &str = "square";

/// Pure encoder for an [`AspectFilter`] — `off`, `landscape`, `portrait`, or
/// `square`. Private; [`aspect_filter_token`] is the only external encoding path,
/// so callers can't encode a stale copy of the state.
fn aspect_filter_token_for(f: AspectFilter) -> String {
    match f {
        AspectFilter::Off => ASPECT_TOK_OFF,
        AspectFilter::Landscape => ASPECT_TOK_LANDSCAPE,
        AspectFilter::Portrait => ASPECT_TOK_PORTRAIT,
        AspectFilter::Square => ASPECT_TOK_SQUARE,
    }
    .to_string()
}

/// Parse a persisted aspect-filter token, falling back to the no-filter state on
/// anything unrecognised (including `off` itself — same path, one exit). Pure,
/// so it's unit-testable.
fn parse_aspect_filter_token(tok: &str) -> AspectFilter {
    match tok {
        ASPECT_TOK_LANDSCAPE => AspectFilter::Landscape,
        ASPECT_TOK_PORTRAIT => AspectFilter::Portrait,
        ASPECT_TOK_SQUARE => AspectFilter::Square,
        _ => AspectFilter::Off,
    }
}

/// Encode the *current* aspect filter as its persistence token. `pub` so `lib.rs`
/// — which holds the db path — can store it.
pub fn aspect_filter_token() -> String {
    aspect_filter_token_for(current_aspect_filter())
}

/// Seed the aspect quick-filter from a persisted token *without* reloading —
/// same startup contract as [`apply_rating_filter_token`]. Main-thread only.
pub fn apply_aspect_filter_token(tok: &str) {
    ASPECT_FILTER.with(|a| a.set(parse_aspect_filter_token(tok)));
}

/// Record how to re-run the current view (called by each loader with a closure
/// that re-invokes it with the same args). Stored as `Rc` so [`set_sort_order`]
/// can clone it out and call it *without* holding the `RefCell` borrow — the
/// loader it invokes re-registers here (`borrow_mut`), which would otherwise
/// panic on the outstanding borrow.
fn register_reload(f: impl Fn() + 'static) {
    RELOAD_CURRENT.with(|r| *r.borrow_mut() = Some(Rc::new(f)));
}

/// Change the grid sort order and re-render the current view under it. No-op if
/// nothing has been loaded yet (no registered reload). Main-thread only (reads
/// thread-local state and touches the GTK model).
pub fn set_sort_order(order: SortOrder) {
    SORT_ORDER.with(|s| s.set(order));
    reload_current_view();
}

/// Change the sort direction (reversed or natural) and re-render the current
/// view under it. No-op if nothing has been loaded yet. Main-thread only.
pub fn set_sort_reverse(reverse: bool) {
    SORT_REVERSE.with(|r| r.set(reverse));
    reload_current_view();
}

/// Set the rating filter's star count (0..=5) and re-render the current view
/// under it, interpreted through the active comparator. Composes with whatever
/// collection is active. No-op if nothing has been loaded yet. Main-thread only.
pub fn set_min_rating(min: u8) {
    MIN_RATING.with(|r| r.set(min.min(5)));
    filter_changed();
}

/// Set the rating filter's comparator (the dropdown) and re-render the current
/// view under it. No-op if nothing has been loaded yet. Main-thread only.
pub fn set_rating_compare(cmp: RatingCompare) {
    RATING_COMPARE.with(|c| c.set(cmp));
    filter_changed();
}

/// Apply a quick-filter [`FilterPreset`] — comparator *and* star count in one
/// step, so the intermediate state never reaches a loader (one reload, not two).
/// Main-thread only.
pub fn set_filter_preset(preset: FilterPreset) {
    let (cmp, stars) = preset.state();
    RATING_COMPARE.with(|c| c.set(cmp));
    MIN_RATING.with(|r| r.set(stars));
    filter_changed();
}

/// The preset matching the live filter state, or `None` if the state isn't
/// expressible as one (so a dropdown can show "custom" rather than lie).
pub fn current_filter_preset() -> Option<FilterPreset> {
    FilterPreset::for_state(current_rating_compare(), current_min_rating())
}

/// Register a closure that re-syncs a filter control's *display* from the live
/// filter state. Every control that both reads and writes the filter (the top-bar
/// preset dropdown, the bottom bar's comparator + stars) registers one, so a
/// change made through any of them refreshes all the others — without any control
/// knowing the others exist. Main-thread only.
pub fn add_filter_observer(f: impl Fn() + 'static) {
    FILTER_OBSERVERS.with(|o| o.borrow_mut().push(Rc::new(f)));
}

/// Whether we're currently inside an observer pass ([`sync_filter_controls`]). UI
/// handlers consult this to distinguish a *user* edit from the programmatic widget
/// update an observer is making, and skip re-applying the latter (which would
/// recurse and could clobber state mid-sync — e.g. a `DropDown::set_selected` from
/// an observer re-emits `selected-notify`). `pub` because the handlers live in
/// `lib.rs`. Every filter control's handler must consult this, including ones that
/// look inert today: a control that gains a programmatic setter later (e.g. the
/// stars becoming toggles) turns "safe by accident" into a state clobber.
pub fn filter_sync_in_progress() -> bool {
    FILTER_SYNC_DEPTH.with(|d| d.get()) > 0
}

/// RAII depth counter for the observer pass. A **depth counter, not a flag**: if an
/// observer triggers a further filter change, the inner pass's exit must not
/// release the guard while the outer pass is still running — every observer after
/// that point would mistake its own widget write for a user edit. `Drop` releases
/// it, so a panicking observer can't wedge the guard on either.
struct FilterSyncGuard;

impl FilterSyncGuard {
    fn enter() -> Self {
        FILTER_SYNC_DEPTH.with(|d| d.set(d.get().saturating_add(1)));
        FilterSyncGuard
    }
}

impl Drop for FilterSyncGuard {
    fn drop(&mut self) {
        FILTER_SYNC_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Re-sync every registered filter control's display from the live filter state,
/// **without** reloading the grid. Used by a control that has to snap back after
/// rejecting its own input (the top bar's display-only `custom` row) — calling its
/// own sync closure directly from a signal handler would instead re-enter and
/// apply a filter the user never picked. `pub` for that call site.
pub fn sync_filter_controls() {
    // Clone the observers OUT of the RefCell before invoking them: an observer may
    // register another (or otherwise re-enter), and holding the borrow across the
    // calls would panic — the same discipline `reload_current_view` documents.
    let observers = FILTER_OBSERVERS.with(|o| o.borrow().clone());
    let _guard = FilterSyncGuard::enter();
    for f in observers {
        f();
    }
}

/// Re-render the current view for a filter change, then re-sync every registered
/// filter control so they all agree on what's showing.
fn filter_changed() {
    reload_current_view();
    sync_filter_controls();
}

/// Re-run the currently-registered view loader (used after a sort-order or
/// sort-direction change).
fn reload_current_view() {
    // Clone the Rc OUT of the cell before invoking it — this is load-bearing for
    // two independent reasons, so don't "optimize" it into a direct call:
    //  1. the reload closure re-enters `register_reload` (`borrow_mut`); holding
    //     a `borrow()` across the call would panic;
    //  2. `register_reload` overwrites the slot, dropping the very closure that's
    //     currently executing on the stack — the extra refcount here keeps its
    //     environment (incl. the grid model) alive until this returns, so it's
    //     not a use-after-free.
    let reload = RELOAD_CURRENT.with(|r| r.borrow().clone());
    if let Some(reload) = reload {
        reload();
    }
}

/// Split a loader's fetched rows (queried with `LIMIT GRID_CAP + 1`) into the
/// rows to display (capped at `GRID_CAP`) and an optional trailing notice shown
/// when the result was truncated. The notice carries no `/`, so the grid's
/// selection/activation guards skip it exactly like the empty-state placeholders.
/// Pure (no GTK) so it can be unit-tested under the display-free discipline.
fn cap_rows(mut rows: Vec<String>) -> (Vec<String>, Option<String>) {
    if rows.len() > GRID_CAP {
        rows.truncate(GRID_CAP);
        (rows, Some(format!("(showing first {GRID_CAP} — refine your filter)")))
    } else {
        (rows, None)
    }
}

/// Replace the grid model's contents with a loader's rows: cap them, add the
/// truncation notice if any, and fall back to `empty_placeholder` when nothing
/// matched. The single place the cap/notice tail lives, shared by every loader,
/// and the single place the model is (re)filled — callers no longer clear first.
///
/// Uses one `splice()` to swap old contents for new: O(N) and a *single*
/// `items-changed` emission, versus the old O(N^2) `remove(0)` loop plus per-row
/// `append` (~2N emissions). This matters because the header's image-count label
/// (and any future model observer) is bound to `items-changed` — one update per
/// load instead of thousands.
fn fill_grid(model: &LighttableModel, rows: Vec<String>, empty_placeholder: &str) {
    let (mut rows, notice) = cap_rows(rows);
    if let Some(notice) = notice {
        rows.push(notice);
    }
    if rows.is_empty() {
        rows.push(empty_placeholder.to_string());
    }
    // A collection load is also the retry opportunity for undecodable files
    // (m4-140): the negative cache lives in the shared thumbnail service and
    // serves both this grid and the zoomable canvas, so it resets exactly where
    // "the collection changed" happens — here, not per consumer.
    thumbs::clear_failed();
    // Multi-selection reload sweep (m4-144): every collection change funnels
    // through this single fill — search, rating/colour/sort filter changes,
    // folder switches, imports (senior review MAJOR-2: reselect_path only sees
    // the tag-filter and culling paths). Intersecting here keeps the count
    // label honest on EVERY reload surface and stops stale entries from
    // receiving keyboard metadata writes while invisible. Placeholder and
    // notice rows carry no `/`, so they can never match a selected path.
    selection::prune_to(&rows);
    selection::sync_count_label();
    let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
    model.splice(0, model.n_items(), &refs);
}

pub fn lighttable_load_from_db(model: &LighttableModel, db_path: &str) {
    lighttable_load_by_folder(model, db_path, None);
}

pub fn lighttable_load_by_folder(model: &LighttableModel, db_path: &str, folder: Option<&str>) {
    register_reload({
        let m = model.clone();
        let db = db_path.to_string();
        let f = folder.map(str::to_string);
        move || lighttable_load_by_folder(&m, &db, f.as_deref())
    });

    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        // open_catalog (m4-148): same busy_timeout rationale as every other
        // catalogue read — a held off-thread write lock must wait out, not
        // read as an empty result.
        crate::persist::open_catalog(db_path).unwrap_or_else(|_| open_demo_db())
    };

    let order = current_sort().order_clause(current_reverse());
    // Rating filter (bottom bar) composes on top of the folder selection. The
    // no-folder branch uses `WHERE 1=1` so the ` AND (...)` fragment splices in
    // uniformly (SQLite folds the constant away); empty when no rating filter.
    let rating = current_filters_sql();
    let rows: Vec<String> = match folder {
        Some(f) => {
            conn.prepare(&format!(
                "SELECT f.folder || '/' || i.filename \
                 FROM main.images i \
                 JOIN main.film_rolls f ON f.id = i.film_id \
                 WHERE f.folder = ?1{rating} \
                 ORDER BY {order} LIMIT {}", GRID_CAP + 1),
            )
            .and_then(|mut s| s.query_map([f], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect()))
            .unwrap_or_default()
        }
        None => {
            conn.prepare(&format!(
                "SELECT f.folder || '/' || i.filename \
                 FROM main.images i \
                 JOIN main.film_rolls f ON f.id = i.film_id \
                 WHERE 1=1{rating} \
                 ORDER BY {order} LIMIT {}", GRID_CAP + 1),
            )
            .and_then(|mut s| s.query_map([], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect()))
            .unwrap_or_default()
        }
    };

    fill_grid(model, rows, "(No images in this collection)");
}

/// Filter the lighttable to images whose filename contains `query` (case-insensitive).
/// Pass an empty query to show all images (same as load_from_db).
pub fn lighttable_filter_by_name(model: &LighttableModel, db_path: &str, query: &str) {
    if query.is_empty() {
        lighttable_load_by_folder(model, db_path, None);
        return;
    }
    register_reload({
        let m = model.clone();
        let db = db_path.to_string();
        let q = query.to_string();
        move || lighttable_filter_by_name(&m, &db, &q)
    });
    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        // open_catalog (m4-148): same busy_timeout rationale as every other
        // catalogue read — a held off-thread write lock must wait out, not
        // read as an empty result.
        crate::persist::open_catalog(db_path).unwrap_or_else(|_| open_demo_db())
    };
    let order = current_sort().order_clause(current_reverse());
    let rating = current_filters_sql();
    let pattern = format!("%{query}%");
    let rows: Vec<String> = conn
        .prepare(&format!(
            "SELECT f.folder || '/' || i.filename \
             FROM main.images i JOIN main.film_rolls f ON f.id = i.film_id \
             WHERE i.filename LIKE ?1{rating} \
             ORDER BY {order} LIMIT {}", GRID_CAP + 1),
        )
        .and_then(|mut s| {
            s.query_map([pattern.as_str()], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect())
        })
        .unwrap_or_default();
    fill_grid(model, rows, "(No results)");
}

/// Filter the lighttable to images tagged with `prefix` **or any hierarchical
/// descendant** of it (`prefix|child`, `prefix|child|grandchild`, …). darktable
/// stores hierarchy as `parent|child` in `data.tags.name`, so a leaf tag's
/// prefix matches only itself (subsuming the old id-based filter) while a parent
/// or virtual node gathers its whole subtree. Empty result, or a db without the
/// tag tables (e.g. the demo db), shows a placeholder.
///
/// Matching `prefix` requires `data.tags` (tag names live only there); that is
/// the same connection-reachability contract the left-panel tag tree already
/// relies on — the clicked node was produced by `tag_list_with_counts`
/// (panels::load_tags_with_counts), which queries `data.tags` through
/// `c41_db::schema::open_catalog_session`, an attached-catalog connection, so
/// whenever a clickable node exists this JOIN is reachable too. If the tag tree
/// is ever re-sourced (e.g. cached), revisit this. `prefix` is
/// matched literally: its LIKE metacharacters are escaped (see [`escape_like`])
/// so a tag containing `%`/`_` can't widen the descendant match.
pub fn lighttable_load_by_tag_prefix(model: &LighttableModel, db_path: &str, prefix: &str) {
    register_reload({
        let m = model.clone();
        let db = db_path.to_string();
        let p = prefix.to_string();
        move || lighttable_load_by_tag_prefix(&m, &db, &p)
    });
    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        // open_catalog attaches data.db, where tag names live (data.tags).
        c41_db::schema::open_catalog(db_path).unwrap_or_else(|_| open_demo_db())
    };
    let order = current_sort().order_clause(current_reverse());
    let rating = current_filters_sql();
    // `prefix` itself (the exact tag) OR `prefix|…` (any descendant). DISTINCT
    // because an image carrying several tags under `prefix` would otherwise
    // appear once per matching tag. The OR is parenthesised so the rating filter
    // (` AND …`) applies to BOTH the exact and descendant branches, not just the
    // last one (AND binds tighter than OR).
    let descendants = format!("{}|%", escape_like(prefix));
    let rows: Vec<String> = match conn
        .prepare(&format!(
            "SELECT DISTINCT f.folder || '/' || i.filename \
             FROM main.images i \
             JOIN main.film_rolls f ON f.id = i.film_id \
             JOIN main.tagged_images ti ON ti.imgid = i.id \
             JOIN data.tags t ON t.id = ti.tagid \
             WHERE (t.name = ?1 OR t.name LIKE ?2 ESCAPE '\\'){rating} \
             ORDER BY {order} LIMIT {}", GRID_CAP + 1),
        )
        .and_then(|mut s| {
            s.query_map(rusqlite::params![prefix, descendants], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect())
        }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("darkroom: tag-prefix filter query failed (prefix {prefix:?}): {e}");
            Vec::new()
        }
    };
    fill_grid(model, rows, "(No images with this tag)");
}

/// Colour indices (ascending) selected by a 5-bit `mask` (bit `c` set = colour
/// `c`). Out-of-range bits can't be set (`mask` is `u8` but only `0..COLOR_COUNT`
/// are ever produced). Pure (no GTK / DB) so the mask→indices step is unit-tested.
fn colors_from_mask(mask: u8) -> Vec<u8> {
    (0..COLOR_COUNT).filter(|c| mask & (1 << c) != 0).collect()
}

/// Build the SQL that lists images by a colour-label `mask`. `match_all` true =
/// the image must carry EVERY selected colour (AND); false = ANY of them (OR).
/// Returns `None` for an empty mask (the caller then shows all images), so the
/// "nothing selected" state never reaches the DB. The colour ints are derived
/// from the mask (`0..COLOR_COUNT`), never user text, so inlining them in the
/// `IN (…)` list is injection-safe. AND uses `GROUP BY … HAVING COUNT(DISTINCT
/// cl.color) = N` over the selected colours (`UNIQUE(imgid,color)` makes that
/// count exactly how many of the selected labels the image has); OR uses `SELECT
/// DISTINCT` (an image with several selected colours would otherwise repeat).
/// `ORDER BY`/`LIMIT` mirror the other loaders. Pure (returns a string) so the
/// AND/OR shape is unit-testable under the display-free discipline.
/// Escape the SQL `LIKE` metacharacters (`%`, `_`) and the escape char itself
/// (`\`) in `s`, for use as a literal segment in a `LIKE … ESCAPE '\'` pattern.
/// Backslash is escaped first so the escapes we add for `%`/`_` aren't re-escaped.
fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

// ── Selection preservation across reloads ──────────────────────────────────

/// The full image path currently selected in the grid, or `None` if nothing
/// real is selected (e.g. a placeholder row, which carries no `/`). Capture this
/// BEFORE a model reload so the selection can be restored afterwards.
pub fn selected_path(selection: &SingleSelection) -> Option<String> {
    let model = selection.model()?;
    let s = model
        .item(selection.selected())
        .and_downcast::<gtk4::StringObject>()?
        .string()
        .to_string();
    s.contains('/').then_some(s)
}

/// Re-select `prev` in the grid if it survived a reload; otherwise leave the
/// model's default (autoselected index 0) in place. No-op when `prev` is `None`.
/// Under culling the selection's model is only a *window* over the collection, so
/// the lookup runs against the base and the window is moved to the image's page —
/// otherwise a reload would silently lose the selection whenever the image sits off
/// the current page.
pub fn reselect_path(selection: &SingleSelection, prev: Option<&str>) {
    let Some(model) = selection.model() else { return };
    let slice = model.clone().downcast::<gtk4::SliceListModel>().ok();
    let Some(base) = (match &slice {
        Some(s) => s.model(),
        None => Some(model),
    }) else {
        return;
    };
    let base_paths = model_paths(&base);
    let Some(prev) = prev else { return };
    let Some(idx) = index_of_path(&base_paths, prev) else { return };
    match &slice {
        Some(slice) => {
            let offset = cull_entry_offset(idx, slice.size());
            if offset != slice.offset() {
                slice.set_offset(offset);
                CULL_OFFSET.with(|o| o.set(offset));
            }
            selection.set_selected(idx - offset);
        }
        None => selection.set_selected(idx),
    }
}

/// Pure core of [`reselect_path`]: the index of `target` in `paths`, if present.
/// Kept separate so the (display-bound) reselect logic has a unit-testable seam.
/// Returns the FIRST match; assumes grid paths are unique (the loaders' joins
/// yield distinct `folder/filename` rows today — revisit if that ever changes).
fn index_of_path(paths: &[String], target: &str) -> Option<u32> {
    paths.iter().position(|p| p == target).map(|i| i as u32)
}

fn open_demo_db() -> rusqlite::Connection {
    use rusqlite::Connection;
    let conn = Connection::open_in_memory().expect("in-memory db");
    // `color_labels` is present even though the demo seeds no labels: the m4-126
    // quick filter's fragment references the table unconditionally, and without it
    // a live click would make every subsequent load fail its query into an empty
    // grid (senior-review MAJOR-1). Same class as the m4-135 EXIF columns: a live
    // numeric rule-stack row splices `(i.exposure<…)` into every loader query, and
    // without the columns that prepare fails into an empty grid for ALL rows —
    // including ones a text rule composed via OR should keep (senior-review
    // MAJOR-1, second occurrence). NULLs seed nothing: unprobed is the truthful
    // demo state, and NULL never satisfies a comparison. The demo never restores
    // persisted filters (empty prefs path), so only deliberate clicks can arm the
    // fragments here.
    conn.execute_batch(
        "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder VARCHAR, access_timestamp INTEGER);
         CREATE TABLE images    (id INTEGER PRIMARY KEY, film_id INTEGER, filename VARCHAR,
                                 width INTEGER, height INTEGER, flags INTEGER, datetime_taken INTEGER,
                                 exposure REAL, aperture REAL, iso REAL, focal_length REAL);
         CREATE TABLE color_labels (imgid INTEGER, color INTEGER);
         INSERT INTO film_rolls VALUES (1, '/photos/demo', 0);
         INSERT INTO images VALUES (1, 1, 'DSC_0001.jpg', 6000, 4000, 0, 100,
                                    NULL, NULL, NULL, NULL);
         INSERT INTO images VALUES (2, 1, 'DSC_0002.jpg', 6000, 4000, 0, 200,
                                    NULL, NULL, NULL, NULL);
         INSERT INTO images VALUES (3, 1, 'DSC_0003.jpg', 6000, 4000, 0, 300,
                                    NULL, NULL, NULL, NULL);",
    )
    .expect("demo data");
    conn
}

#[cfg(test)]
mod tests {
    use super::{cap_rows, color_dot_markup, colors_from_mask,
                digit_to_rating, escape_like, fkey_to_color, flags_star_rating,
                add_filter_observer, apply_overlay_mode_token, current_overlay_mode,
                current_filters_sql, current_rating_compare, effective_overlay_visibility,
                filter_sync_in_progress,
                colour_filter_token_for, colour_predicate, parse_colour_filter_token,
                aspect_predicate, aspect_filter_token_for, apply_aspect_filter_token,
                current_aspect_filter, parse_aspect_filter_token, set_aspect_filter,
                current_rule_stack, rule_stack_token, apply_rule_stack_token, set_rule_stack,
                collection_filter_payload, parse_collection_payload, apply_collection_payload,
                current_min_rating, current_colour_mask, current_colour_all, current_year_range,
                set_colour_filter, set_filter_preset, set_min_rating, set_rating_compare,
                set_year_range, FilterPreset,
                flags_with_star_rating, index_of_path, overlay_mode_token_for,
                overlay_visibility, parse_overlay_mode_token, parse_rating_filter_token,
                path_write_lock, rating_filter_token_for, rating_predicate, OverlayMode,
                apply_view_mode_token, current_view_mode, parse_view_mode_token,
                store_view_mode, view_mode_switcher_tooltip, view_mode_token_for, ViewMode,
                cull_cell_pixels, cull_clamp_offset, cull_entry_offset, cull_key_direction,
                cull_page_offset, cull_window_size, fit_inside, CULL_CELL_CHROME_PX,
                THUMB_SIZE,
                CULL_MAX_IMAGES, CULL_MIN_IMAGES,
                RatingCompare, SortOrder, AspectFilter, COLOR_COUNT, COLOR_DIM_HEX, COLOR_HEX,
                GRID_CAP};
    use std::sync::Arc;

    fn n_rows(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("/a/{i}.jpg")).collect()
    }

    #[test]
    fn path_write_lock_is_stable_per_path_and_distinct_across_paths() {
        // Same path → same lock (so writes to one image serialize against each
        // other); different paths → different locks (so they run concurrently).
        let a1 = path_write_lock("/lib/a.raw");
        let a2 = path_write_lock("/lib/a.raw");
        let b  = path_write_lock("/lib/b.raw");
        assert!(Arc::ptr_eq(&a1, &a2), "same path must map to the same lock");
        assert!(!Arc::ptr_eq(&a1, &b), "distinct paths must map to distinct locks");
    }

    #[test]
    fn colors_from_mask_extracts_set_bits_ascending() {
        assert_eq!(colors_from_mask(0), Vec::<u8>::new());
        assert_eq!(colors_from_mask(0b00001), vec![0]);
        assert_eq!(colors_from_mask(0b10000), vec![4]);
        assert_eq!(colors_from_mask(0b10101), vec![0, 2, 4]);
        assert_eq!(colors_from_mask(0b11111), (0..COLOR_COUNT).collect::<Vec<_>>());
    }

    #[test]
    fn colors_from_mask_ignores_bits_above_the_colour_domain() {
        // Only 0..COLOR_COUNT are scanned, so a stray high bit can't widen the set.
        assert_eq!(colors_from_mask(0b1010_0001), vec![0]);
    }

    #[test]
    fn colour_predicate_empty_for_no_filter() {
        assert_eq!(colour_predicate(0, false), "");
        assert_eq!(colour_predicate(0, true), "");
    }

    #[test]
    fn colour_predicate_any_requires_at_least_one_selected_colour() {
        let sql = colour_predicate(0b10101, false);
        // Correlated subquery over the row's id; >= 1 is OR ("any") semantics.
        assert!(sql.contains("FROM main.color_labels cl WHERE cl.imgid = i.id"), "{sql}");
        assert!(sql.contains("cl.color IN (0,2,4)"), "{sql}");
        assert!(sql.contains(">= 1"), "{sql}");
    }

    #[test]
    fn colour_predicate_all_counts_every_selected_colour() {
        let sql = colour_predicate(0b01010, true);
        assert!(sql.contains("cl.color IN (1,3)"), "{sql}");
        // AND => image must carry ALL N selected colours.
        assert!(sql.contains("= 2"), "{sql}");
    }

    #[test]
    fn colour_predicate_single_bit_collapses_to_count_one() {
        let sql = colour_predicate(0b00100, true);
        assert!(sql.contains("cl.color IN (2)"), "{sql}");
        assert!(sql.contains("= 1"), "{sql}");
    }

    #[test]
    fn colour_filter_token_roundtrips_with_canonicalisations() {
        // encode∘decode identity for every mask the UI can produce, in both modes,
        // modulo the codec's one canonicalisation: mask 0 is always "off".
        for m in [0u8, 1, 0b00100, 0b10101, 0x1F] {
            for all in [false, true] {
                let tok = colour_filter_token_for(m, all);
                let want = if m == 0 { (0, false) } else { (m & 0x1F, all) };
                assert_eq!(parse_colour_filter_token(&tok), want, "tok={tok}");
            }
        }
    }

    #[test]
    fn colour_token_parse_falls_back_to_off_on_corruption() {
        assert_eq!(parse_colour_filter_token(""), (0, false));
        assert_eq!(parse_colour_filter_token("sometimes"), (0, false));
        assert_eq!(parse_colour_filter_token("any:"), (0, false));
        assert_eq!(parse_colour_filter_token("any:zz"), (0, false));
        // A stray high bit can't widen the colour domain.
        assert_eq!(parse_colour_filter_token("any:255"), (255 & 0x1F, false));
        assert_eq!(parse_colour_filter_token("all:31"), (0b11111, true));
    }

    #[test]
    fn aspect_predicate_matches_darktables_stock_presets() {
        // darktable's three stock collection presets as SQL fragments — exact
        // strings, since the loaders splice them straight after a WHERE term.
        assert_eq!(aspect_predicate(AspectFilter::Off), "");
        assert_eq!(aspect_predicate(AspectFilter::Landscape), " AND i.width > i.height");
        assert_eq!(aspect_predicate(AspectFilter::Portrait), " AND i.height > i.width");
        assert_eq!(
            aspect_predicate(AspectFilter::Square),
            " AND i.width = i.height AND i.width > 0",
            "the w>0 guard keeps a failed probe's 0×0 row from counting as square"
        );
    }

    #[test]
    fn aspect_filter_token_roundtrips_all_variants() {
        for f in AspectFilter::ALL {
            let tok = aspect_filter_token_for(f);
            if f == AspectFilter::Off {
                assert_eq!(tok, "off");
            }
            assert_eq!(parse_aspect_filter_token(&tok), f, "tok={tok}");
            assert!(!f.label().is_empty(), "label for {f:?}");
            // Index mapping is the dropdown's seed/apply path; it must agree
            // with ALL for every variant and stay panic-free out of range.
            assert_eq!(AspectFilter::from_index(f.to_index()), f);
        }
        assert_eq!(AspectFilter::from_index(99), AspectFilter::Off);
    }

    #[test]
    fn aspect_state_seeds_from_persisted_token_without_reload() {
        // The startup contract: apply_* writes state only — no reload is possible
        // here anyway (no registered loader on this thread), but `set_*` would go
        // through filter_changed; apply must not need one to be correct.
        apply_aspect_filter_token("portrait");
        assert_eq!(current_aspect_filter(), AspectFilter::Portrait);
        apply_aspect_filter_token("");
        assert_eq!(current_aspect_filter(), AspectFilter::Off, "corrupt ⇒ no filter");
        apply_aspect_filter_token("square");
        assert_eq!(current_aspect_filter(), AspectFilter::Square);
        apply_aspect_filter_token("off");
        assert_eq!(current_aspect_filter(), AspectFilter::Off);
    }

    #[test]
    fn quick_filters_include_the_aspect_rule_in_composition_order() {
        // Splice order is rating → year → colour → aspect, each fragment with its
        // own leading " AND ". Pinned exactly like quick_filters_compose… above,
        // because the loaders splice straight after a WHERE term.
        set_filter_preset(FilterPreset::AllImages);
        set_year_range(None);
        set_aspect_filter(AspectFilter::Off);
        assert_eq!(current_filters_sql(), "", "no filters ⇒ nothing spliced");

        set_aspect_filter(AspectFilter::Square);
        assert_eq!(current_filters_sql(), " AND i.width = i.height AND i.width > 0");

        set_filter_preset(FilterPreset::AtLeastStars(2));
        set_colour_filter(0b00001, false); // ANY of {red}
        assert_eq!(
            current_filters_sql(),
            concat!(
                " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 2 AND 5",
                " AND ((SELECT COUNT(DISTINCT cl.color) FROM main.color_labels cl",
                " WHERE cl.imgid = i.id AND cl.color IN (0)) >= 1)",
                " AND i.width = i.height AND i.width > 0",
            )
        );
        // Clearing the aspect rule alone drops just its fragment.
        set_aspect_filter(AspectFilter::Off);
        let rest = current_filters_sql();
        assert!(rest.contains("color_labels") && !rest.contains("i.width"), "{rest}");
        set_colour_filter(0, false);
        set_filter_preset(FilterPreset::AllImages);
    }

    #[test]
    fn rule_stack_splices_into_composition_order() {
        // The stack composes AFTER the single-state filters (m4-134 splice
        // order: rating → year → colour → aspect → rules), and an empty/blank
        // stack splices nothing — pinned here because the loaders append this
        // string straight after a WHERE term.
        use crate::lighttable::rule_stack::{Combinator, Rule, RuleProperty, RuleCmp};
        set_filter_preset(FilterPreset::AllImages);
        set_rule_stack(Vec::new());
        assert_eq!(current_filters_sql(), "", "empty stack ⇒ nothing spliced");

        // A blank-valued row is state but not SQL: still nothing.
        set_rule_stack(vec![Rule::new(
            Combinator::And, RuleProperty::FileName, RuleCmp::Contains, " ",
        )]);
        assert_eq!(current_filters_sql(), "", "all-blank stack ⇒ nothing spliced");

        // One real rule lands as the leading-AND fragment…
        set_rule_stack(vec![Rule::new(
            Combinator::And, RuleProperty::FileName, RuleCmp::Contains, "img",
        )]);
        assert_eq!(
            current_filters_sql(),
            " AND (i.filename LIKE '%img%' ESCAPE '\\')"
        );

        // …and combines with the quick filters in declared order (the rating
        // fragment carries its own not-rejected term).
        set_min_rating(2);
        assert_eq!(
            current_filters_sql(),
            concat!(
                " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 2 AND 5",
                " AND (i.filename LIKE '%img%' ESCAPE '\\')",
            )
        );
        // Restore: the thread-local must not leak into other tests.
        set_rule_stack(Vec::new());
        set_filter_preset(FilterPreset::AllImages);
        assert_eq!(current_filters_sql(), "");
    }

    #[test]
    fn numeric_rule_prepares_against_the_demo_schema() {
        // Senior-review MAJOR-1 (m4-135): a live numeric rule splices
        // `(i.exposure<…)` into every loader query, and the demo catalogue's DDL
        // must carry those columns — without them `prepare` fails and the grid
        // degrades to empty for ALL rows, including ones an OR-composed text rule
        // should keep. Same failure class as m4-126's missing `color_labels`.
        // Prepared against the real demo connection, so a dropped column fails
        // here rather than at user time; NULL rows match no comparison, so zero
        // survivors is also the CORRECT answer, not just a non-crashing one.
        use crate::lighttable::rule_stack::{Combinator, Rule, RuleProperty, RuleCmp};
        set_filter_preset(FilterPreset::AllImages);
        set_rule_stack(vec![Rule::new(
            Combinator::And,
            RuleProperty::Iso,
            RuleCmp::Gte,
            "100",
        )]);
        let sql = format!(
            "SELECT i.filename FROM images i \
             JOIN film_rolls f ON f.id = i.film_id WHERE 1=1{}",
            current_filters_sql()
        );
        let n = super::open_demo_db()
            .prepare(&sql)
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .flatten()
            .count();
        assert_eq!(n, 0, "all-NULL demo rows satisfy no comparison");
        set_rule_stack(Vec::new());
    }

    #[test]
    fn rule_stack_token_roundtrips_through_the_live_state() {
        use crate::lighttable::rule_stack::{Combinator, Rule, RuleProperty, RuleCmp};
        let original = vec![
            Rule::new(Combinator::And, RuleProperty::FileName, RuleCmp::Contains, "keep"),
            Rule::new(Combinator::AndNot, RuleProperty::FilmRoll, RuleCmp::Excludes, "raw"),
        ];
        set_rule_stack(original.clone());
        let tok = rule_stack_token();
        apply_rule_stack_token(&tok);
        assert_eq!(current_rule_stack(), original);
        set_rule_stack(Vec::new());
    }

    #[test]
    fn collection_payload_round_trips_the_whole_filter_set() {
        // m4-136: a preset payload captures ALL five filters; applying one lands
        // the same end state as hand-clicking each control, with ONE view
        // reload at the end (see apply_collection_payload). Reset first, then
        // set every filter, capture, wipe, apply, compare state.
        use crate::lighttable::rule_stack::{Combinator, Rule, RuleProperty, RuleCmp};
        set_filter_preset(FilterPreset::AllImages);
        set_min_rating(0);
        set_colour_filter(0, false);
        set_aspect_filter(AspectFilter::Off);
        set_year_range(None);
        set_rule_stack(Vec::new());
        let base = parse_collection_payload(&collection_filter_payload()).unwrap();
        assert_eq!(
            (base.rating.as_str(), base.colour.as_str(), base.aspect.as_str()),
            ("off", "off", "off"),
            "no-filter payload carries the no-filter tokens"
        );
        assert_eq!(base.year, None);
        assert_eq!(base.rules, "");

        set_min_rating(3);
        set_colour_filter(0b0_0011, false);
        set_aspect_filter(AspectFilter::Landscape);
        set_year_range(Some((2020, 2023)));
        set_rule_stack(vec![
            Rule::new(Combinator::And, RuleProperty::Iso, RuleCmp::Gte, "800"),
        ]);
        let payload = collection_filter_payload();

        // Wipe everything back to no-filter…
        set_min_rating(0);
        set_colour_filter(0, false);
        set_aspect_filter(AspectFilter::Off);
        set_year_range(None);
        set_rule_stack(Vec::new());

        // …apply the captured payload, and every filter comes back.
        assert!(apply_collection_payload(&payload));
        assert_eq!(current_min_rating(), 3);
        assert_eq!(current_colour_mask(), 0b0_0011);
        assert!(!current_colour_all());
        assert_eq!(current_aspect_filter(), AspectFilter::Landscape);
        assert_eq!(current_year_range(), Some((2020, 2023)));
        assert_eq!(
            current_rule_stack(),
            vec![Rule::new(Combinator::And, RuleProperty::Iso, RuleCmp::Gte, "800")]
        );

        // A space inside a rule VALUE must survive the whitespace-separated
        // payload: pct-encoding maps it to %20 before split_whitespace ever
        // sees it (senior-review m3a, m4-136).
        let spaced = vec![Rule::new(
            Combinator::And,
            RuleProperty::FileName,
            RuleCmp::Contains,
            "new year",
        )];
        set_rule_stack(spaced.clone());
        assert!(apply_collection_payload(&collection_filter_payload()));
        assert_eq!(current_rule_stack(), spaced);

        // A preset saved with NO rules must CLEAR active rules when applied —
        // the generic token applier deliberately ignores empty parses ("garbage
        // must not clobber"), so the explicit-clear path is pinned separately.
        let no_rules_payload = {
            set_rule_stack(Vec::new());
            let p = collection_filter_payload();
            set_rule_stack(vec![
                Rule::new(Combinator::And, RuleProperty::Iso, RuleCmp::Gte, "800"),
            ]);
            p
        };
        assert!(apply_collection_payload(&no_rules_payload));
        assert!(
            current_rule_stack().is_empty(),
            "a no-rules preset clears the active stack"
        );

        // Restore ALL five thread-locals — each test runs on its own thread, so
        // nothing actually leaks, but leaving this block honest beats leaving a
        // comment implying more than it does (senior-review m3b, m4-136).
        set_min_rating(0);
        set_colour_filter(0, false);
        set_aspect_filter(AspectFilter::Off);
        set_year_range(None);
        set_rule_stack(Vec::new());
    }

    #[test]
    fn collection_payload_parser_is_strict_on_structure_lenient_on_content() {
        // Structure errors reject the WHOLE payload: wrong arity, unknown
        // version (a future writer we can't reason about), missing/malformed
        // year pair, years out of order.
        assert!(parse_collection_payload("v1 off off off off").is_none());
        assert!(parse_collection_payload("v2 off off off off off").is_none());
        assert!(parse_collection_payload("").is_none());
        assert!(parse_collection_payload("v1 off off off 2035:1990 off").is_none());
        assert!(parse_collection_payload("v1 off off off 2000:x off").is_none());
        // Content garbage still parses: each component's own decoder falls back
        // to no-filter when it's applied, so one corrupt field can't poison a
        // preset by making it unusable wholesale.
        let st =
            parse_collection_payload("v1 junk any:99 nosuch 1:2 pct%3Aok").unwrap();
        assert_eq!(st.rating, "junk");
        assert_eq!(st.colour, "any:99");
        assert_eq!(st.aspect, "nosuch");
        assert_eq!(st.year, Some((1, 2)));
        assert_eq!(st.rules, "pct%3Aok");
        // The empty-stack sentinel normalises to "" so callers see the same
        // representation `current_rule_stack` implies for no rules.
        let empty = parse_collection_payload("v1 off off off off -").unwrap();
        assert_eq!(empty.rules, "");
        // An unparsable payload applies NOTHING (not even the fallbacks) — the
        // caller surfaces it instead of silently clearing the user's filters.
        assert!(!apply_collection_payload("nonsense"));
    }

    #[test]
    fn sort_order_clauses_are_stable() {
        assert_eq!(SortOrder::Filename.order_clause(false), "f.folder ASC, i.filename ASC");
        // Undated (NULL/0) sorts last via the leading boolean key.
        assert_eq!(
            SortOrder::DateTaken.order_clause(false),
            "(i.datetime_taken IS NULL OR i.datetime_taken = 0) ASC, i.datetime_taken ASC, i.filename ASC"
        );
        // Rejected (flags bit 3 = 8) or a legacy >5 star value is remapped below 0
        // so it sorts under 0-star.
        assert_eq!(
            SortOrder::Rating.order_clause(false),
            "CASE WHEN (i.flags & 8) = 8 OR (i.flags & 7) > 5 THEN -1 ELSE (i.flags & 7) END DESC, i.filename ASC"
        );
    }

    #[test]
    fn reversing_flips_reversible_terms_only() {
        // Every reversible term flips ASC<->DESC.
        assert_eq!(SortOrder::Filename.order_clause(true), "f.folder DESC, i.filename DESC");
        assert_eq!(
            SortOrder::Rating.order_clause(true),
            "CASE WHEN (i.flags & 8) = 8 OR (i.flags & 7) > 5 THEN -1 ELSE (i.flags & 7) END ASC, i.filename DESC"
        );
        // The undated-last guard is NON-reversible: it stays ASC so undated images
        // remain at the bottom even when the date sort is reversed.
        assert_eq!(
            SortOrder::DateTaken.order_clause(true),
            "(i.datetime_taken IS NULL OR i.datetime_taken = 0) ASC, i.datetime_taken DESC, i.filename DESC"
        );
    }

    #[test]
    fn colour_predicate_composes_with_rating_in_where() {
        // The loaders splice rating + year + colour fragments in sequence onto
        // their WHERE; pin the concatenation shape for two of them (the same seam
        // current_filters_sql assembles).
        let composed = format!(
            "{}{}",
            rating_predicate(3, RatingCompare::AtLeast),
            colour_predicate(0b00100, true),
        );
        assert!(
            composed.contains("AND (i.flags & 7) BETWEEN 3 AND 5 AND ((SELECT COUNT(DISTINCT cl.color)"),
            "{composed}"
        );
    }

    #[test]
    fn colour_filter_keeps_only_labelled_matches_end_to_end() {
        // End-to-end over a real SQLite catalogue: the composed fragment must keep
        // exactly the images whose colour labels satisfy OR/AND semantics, on top
        // of the collection's own WHERE (here: everything). Seeded with the same
        // table shape ensure_base_schema creates.
        use rusqlite::Connection;
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT);
             CREATE TABLE color_labels (imgid INTEGER, color INTEGER);
             INSERT INTO film_rolls VALUES (1, '/f');
             INSERT INTO images VALUES (1, 1, 'red.raw');
             INSERT INTO images VALUES (2, 1, 'red_green.raw');
             INSERT INTO images VALUES (3, 1, 'unlabelled.raw');
             INSERT INTO color_labels VALUES (1, 0);
             INSERT INTO color_labels VALUES (2, 0);
             INSERT INTO color_labels VALUES (2, 2);",
        )
        .unwrap();
        // NOTE: the real loaders run against `main.images`; the in-memory db here
        // IS main, so the fragment's `main.color_labels` reference resolves the
        // same way it does in the app (same unqualified-table semantics).
        let base = "SELECT i.filename FROM images i \
                    JOIN film_rolls f ON f.id = i.film_id WHERE 1=1";
        let run = |frag: String| -> Vec<String> {
            let sql = format!("{base}{frag} ORDER BY i.filename");
            let mut s = conn.prepare(&sql).unwrap();
            s.query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .flatten()
                .collect()
        };

        // ANY of {red} → both red-labelled images; unlabelled drops out.
        assert_eq!(run(colour_predicate(0b00001, false)), vec!["red.raw", "red_green.raw"]);
        // ALL of {red (bit 0), green (bit 2)} → only the image carrying both.
        assert_eq!(run(colour_predicate(0b00101, true)), vec!["red_green.raw"]);
        // Empty mask → no fragment → the whole collection.
        assert_eq!(run(colour_predicate(0, false)).len(), 3);
    }

    #[test]
    fn aspect_filter_keeps_only_matching_orientations_end_to_end() {
        // Same end-to-end discipline as the colour test above, for the aspect
        // fragment: run it through real SQLite so alias/syntax/NULL surprises
        // surface as wrong survivors rather than a passing string comparison.
        // Seeded with every orientation plus the failed-probe row (0×0) and a
        // NULL-dims row — together they pin the guard semantics end to end.
        use rusqlite::Connection;
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, filename TEXT,
                                   width INTEGER, height INTEGER);
             INSERT INTO images VALUES (1, 'landscape.raw', 6000, 4000);
             INSERT INTO images VALUES (2, 'portrait.raw',  4000, 6000);
             INSERT INTO images VALUES (3, 'square.raw',    3000, 3000);
             INSERT INTO images VALUES (4, 'broken.raw',    0,    0);
             INSERT INTO images VALUES (5, 'unknown.raw',   NULL, NULL);",
        )
        .unwrap();
        let base = "SELECT i.filename FROM images i WHERE 1=1";
        let run = |frag: &str| -> Vec<String> {
            let sql = format!("{base}{frag} ORDER BY i.filename");
            conn.prepare(&sql)
                .unwrap()
                .query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .flatten()
                .collect()
        };

        assert_eq!(run(aspect_predicate(AspectFilter::Landscape)), vec!["landscape.raw"]);
        assert_eq!(run(aspect_predicate(AspectFilter::Portrait)), vec!["portrait.raw"]);
        // Square must NOT catch the 0×0 probe failure (the w>0 guard)…
        assert_eq!(run(aspect_predicate(AspectFilter::Square)), vec!["square.raw"]);
        // …and Off keeps everything, broken rows included (no filter ≠ all-match).
        assert_eq!(run(aspect_predicate(AspectFilter::Off)).len(), 5);
    }

    #[test]
    fn rule_stack_keeps_only_matching_filenames_end_to_end() {
        // Same end-to-end discipline as the colour/aspect tests above, for the
        // rule-stack fragment: real SQLite, so a wrong property column (the
        // hand-written `i.filename`/`f.folder` strings), invalid ESCAPE syntax,
        // or an unneutralised quote/wildcard shows up as wrong survivors or a
        // prepare error rather than a passing string comparison.
        use crate::lighttable::rule_stack::{self, Combinator, Rule, RuleProperty, RuleCmp};
        use rusqlite::Connection;
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT);
             INSERT INTO film_rolls VALUES (1, '/rolls/2024');
             INSERT INTO film_rolls VALUES (2, '/rolls/raw');
             INSERT INTO images VALUES (1, 1, 'img_0001.raw');
             INSERT INTO images VALUES (2, 1, 'img_0002.raw');
             INSERT INTO images VALUES (3, 2, 'other.jpg');",
        )
        .unwrap();
        let base = "SELECT i.filename FROM images i \
                    JOIN film_rolls f ON f.id = i.film_id WHERE 1=1";
        let run = |stack: &[Rule]| -> Vec<String> {
            let sql = format!("{base}{}", rule_stack::rule_stack_sql(stack));
            conn.prepare(&sql)
                .unwrap()
                .query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .flatten()
                .collect()
        };
        use Combinator::{And, AndNot, Or};
        use RuleProperty::{FilmRoll, FileName};
        use RuleCmp::{Contains, Excludes, Gte, Lt};

        // A plain filename rule hits exactly its match…
        assert_eq!(
            run(&[Rule::new(And, FileName, Contains, "0001")]),
            vec!["img_0001.raw"]
        );
        // …and the film-roll property goes through ITS column, not filename.
        assert_eq!(run(&[Rule::new(And, FilmRoll, Contains, "raw")]), vec!["other.jpg"]);
        // Excludes flips to NOT LIKE over the same pattern shape.
        assert_eq!(
            run(&[Rule::new(And, FileName, Excludes, "0001")]),
            vec!["img_0002.raw", "other.jpg"]
        );
        // A % in the value is a LITERAL: nothing seeded contains one. Broken
        // escaping would leave it a wildcard and every row would survive.
        assert_eq!(run(&[Rule::new(And, FileName, Contains, "%")]).len(), 0);
        // The classic injection stays an inert literal (quote doubled): zero
        // matches, not the whole table.
        assert_eq!(run(&[Rule::new(And, FileName, Contains, "x' OR 1=1 --")]).len(), 0);
        // Left-assoc composition survives the database too: (has-img AND NOT
        // raw-roll) OR named-other — the trailing OR binds to the whole group.
        assert_eq!(
            run(&[
                Rule::new(And, FileName, Contains, "img"),
                Rule::new(AndNot, FilmRoll, Contains, "raw"),
                Rule::new(Or, FileName, Contains, "other"),
            ]),
            vec!["img_0001.raw", "img_0002.raw", "other.jpg"]
        );

        // ── Numeric properties over the m4-135 EXIF columns ──
        // Same in-memory db gains the four nullable columns: two probed rows,
        // one row WITHOUT exif (NULL) — pinning that NULLs never satisfy a
        // comparison (unknowns stay out of numeric-rule results).
        conn.execute_batch(
            "ALTER TABLE images ADD COLUMN exposure REAL;
             ALTER TABLE images ADD COLUMN aperture REAL;
             ALTER TABLE images ADD COLUMN iso REAL;
             ALTER TABLE images ADD COLUMN focal_length REAL;
             UPDATE images SET exposure = 0.016, iso = 1600, aperture = 4.0 WHERE id = 1;
             UPDATE images SET exposure = 2.5,  iso = 100,  aperture = 2.8 WHERE id = 2;",
        )
        .unwrap();
        use crate::lighttable::rule_stack::PropertyKind;
        use RuleProperty::{Exposure as PExp, Iso as PIso};
        assert_eq!(
            run(&[Rule::new(And, PIso, Gte, "800")]),
            vec!["img_0001.raw"]
        );
        // Fraction syntax parses before it reaches SQL.
        assert_eq!(
            run(&[Rule::new(And, PExp, Lt, "1/60")]),
            vec!["img_0001.raw"]
        );
        // Aperture goes through ITS column too (senior-review m1: a typo in the
        // stringly-typed column() would otherwise only surface as an M1-style
        // empty grid at user time).
        assert_eq!(
            run(&[Rule::new(And, RuleProperty::Aperture, Gte, "2.8")]),
            vec!["img_0001.raw", "img_0002.raw"]
        );
        assert_eq!(
            run(&[Rule::new(And, RuleProperty::Aperture, Gte, "3")]),
            vec!["img_0001.raw"]
        );
        // NULL never satisfies ANY comparison: no row has focal_length, so
        // even the most generous bound keeps nothing.
        let anyone_wide = [Rule::new(And, RuleProperty::FocalLength, Lt, "100000")];
        assert_eq!(run(&anyone_wide), Vec::<String>::new());
        // Probed rows still answer numeric rules normally (0.016 < 1 < 2.5).
        assert_eq!(run(&[Rule::new(And, PExp, Lt, "1")]), vec!["img_0001.raw"]);
        // …and the same rows still answer TEXT rules: exclusion is per-rule,
        // not per-image.
        assert_eq!(
            run(&[Rule::new(And, FileName, Contains, ".")]).len(),
            3
        );
        // Kind/comparator mismatch is inert against the real engine too: the
        // composer SKIPS the rule, so no fragment lands and the whole table
        // answers — an unusable rule never narrows or errors the query.
        assert_eq!(run(&[Rule::new(And, PIso, Excludes, "100")]).len(), 3);
        assert_eq!(PIso.kind(), PropertyKind::Numeric);
    }

    #[test]
    fn rating_predicate_fragment_shape_and_clamp() {
        use RatingCompare::*;
        // AtLeast 0 = no filter → empty fragment (nothing spliced into any WHERE).
        assert_eq!(rating_predicate(0, AtLeast), "");
        // AtLeast N (N≥1) → exclude rejected (bit 3) THEN keep N..5 stars (bits 0–2).
        assert_eq!(rating_predicate(1, AtLeast), " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 1 AND 5");
        assert_eq!(rating_predicate(5, AtLeast), " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 5 AND 5");
        // Out-of-range clamps to 5 (never emits a >5 bound that would match nothing).
        assert_eq!(rating_predicate(9, AtLeast), " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 5 AND 5");
        // Exactly N → `= N` (Exactly 0 is a real filter: unrated only).
        assert_eq!(rating_predicate(0, Exactly), " AND (i.flags & 8) = 0 AND (i.flags & 7) = 0");
        assert_eq!(rating_predicate(3, Exactly), " AND (i.flags & 8) = 0 AND (i.flags & 7) = 3");
        // AtMost N → `BETWEEN 0 AND N` (AtMost 0 is a real filter: unrated only).
        assert_eq!(rating_predicate(0, AtMost), " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 0 AND 0");
        assert_eq!(rating_predicate(2, AtMost), " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 0 AND 2");
        // Rejected → the reject bit only, star count ignored (even a nonzero one).
        assert_eq!(rating_predicate(4, Rejected), " AND (i.flags & 8) = 8");
    }

    #[test]
    fn rating_filter_token_roundtrips_and_falls_back() {
        use RatingCompare::*;
        // Every (comparator, stars) the UI can produce round-trips through the token.
        for (cmp, stars, tok) in [
            (AtLeast, 0u8, "off"),
            (AtLeast, 3, "ge:3"),
            (Exactly, 0, "eq:0"),
            (Exactly, 5, "eq:5"),
            (AtMost, 2, "le:2"),
            (Rejected, 0, "rej"),
        ] {
            assert_eq!(parse_rating_filter_token(tok), (cmp, stars), "decode {tok}");
        }
        // Corrupt/unknown tokens and out-of-range stars fall back safely.
        assert_eq!(parse_rating_filter_token("garbage"), (AtLeast, 0));
        assert_eq!(parse_rating_filter_token("ge:99"), (AtLeast, 5), "clamp to 5");
        assert_eq!(parse_rating_filter_token("xx:2"), (AtLeast, 0), "unknown prefix");
        assert_eq!(parse_rating_filter_token("ge:x"), (AtLeast, 0), "non-numeric");
    }

    #[test]
    fn filter_presets_map_to_states_and_back() {
        use RatingCompare::*;
        // Each preset names a state the bottom bar can also express...
        assert_eq!(FilterPreset::AllImages.state(), (AtLeast, 0));
        assert_eq!(FilterPreset::UnstarredOnly.state(), (Exactly, 0));
        assert_eq!(FilterPreset::AtLeastStars(3).state(), (AtLeast, 3));
        assert_eq!(FilterPreset::RejectedOnly.state(), (Rejected, 0));
        // ...and the reverse lookup round-trips every row, so a preset applied by
        // the top bar always reads back as that same row (never "custom").
        for (i, p) in FilterPreset::ALL.into_iter().enumerate() {
            let (cmp, stars) = p.state();
            assert_eq!(FilterPreset::for_state(cmp, stars), Some(p), "roundtrip {p:?}");
            assert_eq!(FilterPreset::from_index(i as u32), p, "index {i}");
            assert!(!p.label().is_empty(), "label {p:?}");
        }
        // Out-of-range index falls back to the no-filter preset, never panics.
        assert_eq!(FilterPreset::from_index(99), FilterPreset::AllImages);
        // Star clamping keeps a preset inside the 1..=5 domain.
        assert_eq!(FilterPreset::AtLeastStars(9).state(), (AtLeast, 5));
    }

    #[test]
    fn quick_filters_compose_into_one_fragment() {
        // The loaders splice exactly one string, so the rating filter and the
        // timeline's year range must both land in it — a regression here would
        // silently drop one filter from every view at once.
        set_filter_preset(FilterPreset::AllImages);
        set_year_range(None);
        assert_eq!(current_filters_sql(), "", "no filters ⇒ nothing spliced");

        set_filter_preset(FilterPreset::AtLeastStars(2));
        let rating_only = current_filters_sql();
        assert!(rating_only.contains("(i.flags & 7) BETWEEN 2 AND 5"), "{rating_only}");
        assert!(!rating_only.contains("strftime"), "{rating_only}");

        set_year_range(Some((2018, 2020)));
        // Assert the EXACT composed string, not just `contains`: the loaders splice
        // this straight after a WHERE term, so a stray leading `WHERE`/`AND` or a
        // missing leading space would produce invalid SQL that `contains` misses.
        assert_eq!(
            current_filters_sql(),
            concat!(
                " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 2 AND 5",
                " AND i.datetime_taken > 0 AND CAST(strftime('%Y',",
                " (i.datetime_taken / 1000000 - 62135596800), 'unixepoch') AS INTEGER)",
                " BETWEEN 2018 AND 2020",
            )
        );

        // Clearing one leaves the others intact (they're independent).
        set_filter_preset(FilterPreset::AllImages);
        let year_only = current_filters_sql();
        assert!(!year_only.contains("i.flags"), "{year_only}");
        assert!(year_only.contains("BETWEEN 2018 AND 2020"), "{year_only}");
        set_year_range(None);

        // All three at once, pinned exactly (m4-126): splice order is rating →
        // year → colour, each fragment carrying its own leading " AND ".
        set_filter_preset(FilterPreset::AtLeastStars(2));
        set_year_range(Some((2018, 2020)));
        set_colour_filter(0b00101, true); // ALL of {red, green}
        assert_eq!(
            current_filters_sql(),
            concat!(
                " AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN 2 AND 5",
                " AND i.datetime_taken > 0 AND CAST(strftime('%Y',",
                " (i.datetime_taken / 1000000 - 62135596800), 'unixepoch') AS INTEGER)",
                " BETWEEN 2018 AND 2020",
                " AND ((SELECT COUNT(DISTINCT cl.color) FROM main.color_labels cl",
                " WHERE cl.imgid = i.id AND cl.color IN (0,2)) = 2)",
            )
        );
        // Disarming the colour filter alone drops just its fragment.
        set_colour_filter(0, false);
        let two_way = current_filters_sql();
        assert!(two_way.contains("BETWEEN 2018 AND 2020") && !two_way.contains("color_labels"),
                "{two_way}");
        set_year_range(None);
    }

    #[test]
    fn rejected_state_canonicalises_to_the_rejected_preset() {
        use RatingCompare::*;
        // `rating_predicate` ignores the star count in Rejected mode, so (Rejected, N)
        // IS "rejected only" — reachable by picking N stars then switching to ⚑ (the
        // bottom bar retains the count). Reporting "custom" there would be a display
        // lie about a filter a preset does name.
        for n in 0..=5u8 {
            assert_eq!(
                FilterPreset::for_state(Rejected, n),
                Some(FilterPreset::RejectedOnly),
                "(Rejected, {n})"
            );
        }
        // Counts are clamped into the 0..=5 domain the predicate uses, so an
        // out-of-range count still resolves to its preset instead of "custom".
        assert_eq!(
            FilterPreset::for_state(AtLeast, 9),
            Some(FilterPreset::AtLeastStars(5))
        );
    }

    #[test]
    fn filter_observers_run_under_a_guard_that_survives_nesting() {
        use std::cell::Cell as StdCell;
        use std::rc::Rc;
        // The observer bus is the risky part of m4-97c and is display-free testable:
        // the thread-locals need no GTK, and reload_current_view no-ops with no
        // loader registered. (Each #[test] gets its own thread ⇒ its own locals.)
        let runs: Rc<StdCell<u32>> = Rc::new(StdCell::new(0));
        let guarded_inside: Rc<StdCell<bool>> = Rc::new(StdCell::new(false));
        {
            let runs = runs.clone();
            let guarded_inside = guarded_inside.clone();
            add_filter_observer(move || {
                runs.set(runs.get() + 1);
                // Must read `true` INSIDE the pass — that's what stops an observer's
                // widget write being re-applied as a user edit.
                guarded_inside.set(filter_sync_in_progress());
            });
        }
        set_min_rating(3);
        assert_eq!(runs.get(), 1, "observer ran once");
        assert!(guarded_inside.get(), "guard set during the pass");
        assert!(!filter_sync_in_progress(), "guard cleared after the pass");

        // Every filter setter must notify, not just set_min_rating.
        set_rating_compare(RatingCompare::Exactly);
        assert_eq!(runs.get(), 2);
        set_filter_preset(FilterPreset::RejectedOnly);
        assert_eq!(runs.get(), 3);
        assert_eq!(current_rating_compare(), RatingCompare::Rejected);
        // The timeline's year range rides the same bus — that's what keeps its
        // highlight in step with a filter cleared from another control.
        set_year_range(Some((2019, 2019)));
        assert_eq!(runs.get(), 4);
        set_year_range(None);
        assert_eq!(runs.get(), 5);

        // A nested change from inside an observer must NOT release the outer pass's
        // guard (the reason the guard is a depth counter, not a bool).
        let nested_saw_guard: Rc<StdCell<bool>> = Rc::new(StdCell::new(false));
        {
            let nested_saw_guard = nested_saw_guard.clone();
            let armed = Rc::new(StdCell::new(true));
            add_filter_observer(move || {
                if armed.replace(false) {
                    set_min_rating(1); // re-entrant filter change
                }
                // After the inner pass returns, the outer pass is still running.
                nested_saw_guard.set(filter_sync_in_progress());
            });
        }
        set_min_rating(2);
        assert!(nested_saw_guard.get(), "outer pass still guarded after a nested change");
        assert!(!filter_sync_in_progress(), "guard fully released at the end");
    }

    #[test]
    fn filter_states_outside_the_presets_have_no_preset() {
        use RatingCompare::*;
        // The bottom bar can express filters no preset names — `for_state` must
        // return None for those so the dropdown shows "custom" instead of lying.
        assert_eq!(FilterPreset::for_state(AtMost, 3), None, "≤ 3 is not a preset");
        assert_eq!(FilterPreset::for_state(Exactly, 4), None, "= 4 is not a preset");
        // But every preset's own state must be recognised (guards a stale ALL).
        for p in FilterPreset::ALL {
            let (cmp, stars) = p.state();
            assert!(FilterPreset::for_state(cmp, stars).is_some(), "{p:?}");
        }
    }

    #[test]
    fn overlay_visibility_maps_each_mode_to_its_rows() {
        // (filename, stars, colours) — Extended is the default/original layout.
        assert_eq!(overlay_visibility(OverlayMode::Hidden), (false, false, false));
        assert_eq!(overlay_visibility(OverlayMode::Normal), (false, true, true));
        assert_eq!(overlay_visibility(OverlayMode::Extended), (true, true, true));
    }

    #[test]
    fn placeholder_cells_keep_their_label_in_every_mode() {
        // The user-hostile failure this carve-out exists to prevent: a placeholder
        // ("(No images…)") whose label is hidden leaves an unexplained empty grid.
        // Its stars/dots stay hidden — they're meaningless on a sentinel row.
        for mode in OverlayMode::ALL {
            assert_eq!(
                effective_overlay_visibility(true, mode),
                (true, false, false),
                "placeholder under {mode:?}"
            );
        }
        // Real image cells are unaffected by the carve-out.
        for mode in OverlayMode::ALL {
            assert_eq!(effective_overlay_visibility(false, mode), overlay_visibility(mode));
        }
    }

    #[test]
    fn overlay_mode_index_and_token_roundtrip() {
        // Iterate ALL (not a hardcoded list) so adding a variant forces a test touch.
        assert_eq!(OverlayMode::ALL.len(), 3);
        for (i, mode) in OverlayMode::ALL.into_iter().enumerate() {
            // Dropdown index ↔ variant is a bijection over the rows...
            assert_eq!(OverlayMode::from_index(i as u32), mode, "index {i}");
            assert_eq!(mode.to_index(), i as u32, "to_index {mode:?}");
            // ...and encode∘decode is identity (guards encoder AND decoder).
            assert_eq!(parse_overlay_mode_token(overlay_mode_token_for(mode)), mode);
            // Every row has a non-empty label, so the dropdown can't render blanks.
            assert!(!mode.label().is_empty(), "label {mode:?}");
        }
        // Out-of-range index and corrupt tokens fall back to the default look.
        assert_eq!(OverlayMode::from_index(99), OverlayMode::Extended);
        assert_eq!(parse_overlay_mode_token("garbage"), OverlayMode::Extended);
        assert_eq!(parse_overlay_mode_token(""), OverlayMode::Extended);
    }

    #[test]
    fn apply_overlay_mode_token_seeds_the_current_mode() {
        // The startup restore path end-to-end (thread-locals are per-test-thread).
        for mode in OverlayMode::ALL {
            apply_overlay_mode_token(overlay_mode_token_for(mode));
            assert_eq!(current_overlay_mode(), mode);
        }
        // A corrupt persisted value restores the default look rather than panicking.
        apply_overlay_mode_token("nonsense");
        assert_eq!(current_overlay_mode(), OverlayMode::Extended);
    }

    #[test]
    fn view_mode_buttons_are_labelled_and_tokens_are_distinct() {
        // Iterate ALL (not a hardcoded list) so adding a variant forces a test touch.
        assert_eq!(ViewMode::ALL.len(), 3);
        let mut tokens: Vec<&str> = Vec::new();
        for mode in ViewMode::ALL {
            // Every button has an icon and a description, since it carries no label:
            // an empty icon name would render as the broken-image glyph, and an
            // empty description would leave the mode entirely unnamed in the UI.
            assert!(!mode.icon_name().is_empty(), "icon {mode:?}");
            assert!(!mode.tooltip().is_empty(), "tooltip {mode:?}");
            // The switcher's one tooltip is the only place a greyed-out mode can
            // explain itself, so every mode must actually appear in it.
            assert!(
                view_mode_switcher_tooltip().contains(mode.tooltip()),
                "switcher tooltip omits {mode:?}"
            );
            tokens.push(view_mode_token_for(mode));
        }
        // Tokens are distinct, so no two modes can collide in the prefs table (the
        // decoder inverts the encoder, so a duplicate would silently alias a mode).
        tokens.sort_unstable();
        let distinct = tokens.len();
        tokens.dedup();
        assert_eq!(tokens.len(), distinct, "view-mode tokens must be distinct");
        // Corrupt tokens fall back to the default layout rather than panicking.
        assert_eq!(parse_view_mode_token("garbage"), ViewMode::FileManager);
        assert_eq!(parse_view_mode_token(""), ViewMode::FileManager);
    }

    #[test]
    fn view_mode_token_decodes_only_available_modes() {
        // encode∘decode is identity for a mode this build can render; a mode it
        // cannot render decodes to FileManager even though its token is perfectly
        // well-formed — restoring it would open the lighttable onto a layout that
        // draws nothing. Both halves iterate ALL, so implementing a mode (flipping
        // `is_available`) moves it from one arm to the other with no test edit.
        for mode in ViewMode::ALL {
            let decoded = parse_view_mode_token(view_mode_token_for(mode));
            if mode.is_available() {
                assert_eq!(decoded, mode, "available mode must round-trip: {mode:?}");
            } else {
                assert_eq!(
                    decoded,
                    ViewMode::FileManager,
                    "unavailable mode must not restore: {mode:?}"
                );
            }
        }
        // The default layout is always available — otherwise the fallback above
        // would itself be unrenderable.
        assert!(ViewMode::FileManager.is_available());
    }

    #[test]
    fn apply_view_mode_token_seeds_the_current_mode() {
        // The startup restore path end-to-end (thread-locals are per-test-thread).
        for mode in ViewMode::ALL.into_iter().filter(|m| m.is_available()) {
            apply_view_mode_token(view_mode_token_for(mode));
            assert_eq!(current_view_mode(), mode);
        }
        // A corrupt persisted value restores the default layout rather than
        // panicking. Deliberately last: it also leaves VIEW_MODE back at the
        // default for anything else sharing this test thread.
        apply_view_mode_token("nonsense");
        assert_eq!(current_view_mode(), ViewMode::FileManager);
    }

    #[test]
    fn cull_window_size_tracks_the_stepper_within_bounds() {
        // The thumb stepper's whole range maps into the comparison-set bounds.
        assert_eq!(cull_window_size(1), CULL_MIN_IMAGES);
        assert_eq!(cull_window_size(0), CULL_MIN_IMAGES);
        assert_eq!(cull_window_size(4), 4);
        assert_eq!(cull_window_size(u32::MAX), CULL_MAX_IMAGES);
        // Culling needs at least two images to compare, and a range to move in.
        const _: () = assert!(CULL_MIN_IMAGES >= 2 && CULL_MAX_IMAGES > CULL_MIN_IMAGES);
    }

    #[test]
    fn cull_paging_never_walks_off_the_end() {
        // 10 images, 4 per page ⇒ pages start at 0, 4, 8; the last is short (2).
        assert_eq!(cull_page_offset(0, 10, 4, true), 4);
        assert_eq!(cull_page_offset(4, 10, 4, true), 8);
        // The property that matters: an offset at/past n_items shows NOTHING, with
        // no error — so forward must stop rather than produce an empty grid.
        assert_eq!(cull_page_offset(8, 10, 4, true), 8);
        assert_eq!(cull_page_offset(8, 10, 4, false), 4);
        assert_eq!(cull_page_offset(0, 10, 4, false), 0);
        // Degenerate inputs can't panic or hang the window: fewer images than a
        // page, an empty collection, a zero window (would divide/step by nothing).
        assert_eq!(cull_page_offset(0, 3, 4, true), 0);
        assert_eq!(cull_page_offset(0, 0, 4, true), 0);
        assert_eq!(cull_page_offset(0, 10, 0, true), 1);
        // An offset already past the end can only be produced by skipping the
        // clamp, and paging forward from there must not invent a further step;
        // `cull_clamp_offset` is what brings it back into view.
        assert_eq!(cull_page_offset(u32::MAX, 10, 4, true), u32::MAX);
        assert!(cull_clamp_offset(u32::MAX, 10, 4) < 10);
    }

    #[test]
    fn cull_cell_pixels_fill_the_viewport() {
        // An unallocated grid reports 0 on either axis: "not yet known", not a
        // degenerate size — callers re-run on the resize tick.
        assert_eq!(cull_cell_pixels(0, 600, 4), None);
        assert_eq!(cull_cell_pixels(909, 0, 4), None);
        assert_eq!(cull_cell_pixels(-1, 600, 4), None);
        assert_eq!(cull_cell_pixels(909, 600, 0), None);
        // The real container viewport: four images across 909px ⇒ ~227px columns,
        // and the height is what's left after the metadata rows' chrome.
        let (w, h) = cull_cell_pixels(909, 700, 4).expect("allocated");
        assert_eq!(w, 227); // 909 / 4, truncated
        assert_eq!(h, 700 - CULL_CELL_CHROME_PX);
        // Width always divides down to at least a pixel per cell...
        let (w2, _) = cull_cell_pixels(3, 600, CULL_MAX_IMAGES).expect("allocated");
        assert_eq!(w2, 1); // 3 / 8 truncates to 0 — clamped so a cell still draws
        // ...and a degenerate viewport never collapses below thumb size.
        let (_, h3) = cull_cell_pixels(400, 20, 2).expect("allocated");
        assert_eq!(h3, THUMB_SIZE);
    }

    #[test]
    fn fit_inside_preserves_aspect_ratio() {
        // Landscape into a square-ish box: width is the binding side.
        assert_eq!(fit_inside(3000, 2000, 200, 200), Some((200, 133)));
        // Portrait: height binds.
        assert_eq!(fit_inside(2000, 3000, 200, 200), Some((133, 200)));
        // Rectangular box, exact fit along one axis.
        assert_eq!(fit_inside(900, 600, 450, 500), Some((450, 300)));
        // Upscaling is allowed — a small source fills a big culling cell.
        assert_eq!(fit_inside(100, 50, 400, 400), Some((400, 200)));
        // Degenerate inputs are rejected rather than dividing by zero.
        assert_eq!(fit_inside(0, 10, 10, 10), None);
        assert_eq!(fit_inside(10, 10, 0, 10), None);
        assert_eq!(fit_inside(-1, 10, 10, 10), None);
    }

    #[test]
    fn cell_px_follows_the_mode_and_falls_back_until_measured() {
        use super::cell_px_for;
        let measured = (454, 616);
        // Live culling with a real measurement: fill the viewport.
        assert_eq!(cell_px_for(ViewMode::Culling, measured), measured);
        // Culling before the first allocation — the (0,0) "not yet" sentinel
        // must NOT become the requests of every cell; fall back to the square.
        assert_eq!(cell_px_for(ViewMode::Culling, (0, 0)), (THUMB_SIZE, THUMB_SIZE));
        // A half-measured state is just as unusable as an unmeasured one.
        assert_eq!(cell_px_for(ViewMode::Culling, (454, 0)), (THUMB_SIZE, THUMB_SIZE));
        // Any other layout ignores the stored size entirely — the recycled cells
        // go back to their file-manager square on leave.
        assert_eq!(cell_px_for(ViewMode::FileManager, measured), (THUMB_SIZE, THUMB_SIZE));
        // Zoomable included: it doesn't use grid cells at all (hand-drawn canvas),
        // so the decision keys off "is culling", nothing else.
        assert_eq!(cell_px_for(ViewMode::Zoomable, measured), (THUMB_SIZE, THUMB_SIZE));
    }

    // Note: `cull_effective_window` has no dedicated test since m4-132 — its body
    // is a one-line delegation to `cull_window_size(max_columns)` (the viewport cap
    // is gone), and it takes a display-bound `GridView`, so there is nothing to
    // assert beyond what `cull_window_size_tracks_the_stepper_within_bounds`
    // already covers.

    #[test]
    fn cull_entry_offset_opens_on_the_selected_image_page() {
        // Entering culling shows the page holding the selection, not page 1 —
        // the offset is always a page start, and the image is inside that page.
        for window in [1u32, 2, 4, 8] {
            for index in [0u32, 1, 7, 8, 9, 1000] {
                let offset = cull_entry_offset(index, window);
                assert_eq!(offset % window, 0, "not page-aligned: {index}/{window}");
                assert!(offset <= index, "page start past the image: {index}/{window}");
                assert!(index - offset < window, "image outside its page: {index}/{window}");
            }
        }
        // A zero window would divide by zero; it is treated as one image per page.
        assert_eq!(cull_entry_offset(5, 0), 5);
    }

    #[test]
    fn cull_clamp_pulls_a_shrunken_collection_back_into_view() {
        // A filter cutting 100 images to 10 must not leave the window past the end
        // (an empty grid that looks like a hang). It lands on a whole-page start.
        assert_eq!(cull_clamp_offset(96, 10, 4), 8);
        assert_eq!(cull_clamp_offset(10, 10, 4), 8);
        // Offsets already inside the collection are left exactly where they are.
        assert_eq!(cull_clamp_offset(4, 10, 4), 4);
        assert_eq!(cull_clamp_offset(0, 10, 4), 0);
        // Empty collection and zero window are both defined, not panics.
        assert_eq!(cull_clamp_offset(96, 0, 4), 0);
        assert_eq!(cull_clamp_offset(96, 10, 0), 9);
        // Whatever it returns is always a *visible* index (or 0 when nothing is).
        for n_items in [0u32, 1, 3, 10, 97] {
            for offset in [0u32, 1, 9, 96, u32::MAX] {
                for window in [1u32, 2, 4, 8] {
                    let c = cull_clamp_offset(offset, n_items, window);
                    assert!(
                        c < n_items || n_items == 0,
                        "clamp({offset},{n_items},{window}) = {c} is not visible"
                    );
                }
            }
        }
    }

    #[test]
    fn cull_key_direction_maps_only_paging_keys() {
        use gtk4::gdk::Key;
        assert_eq!(cull_key_direction(Key::Right), Some(true));
        assert_eq!(cull_key_direction(Key::Page_Down), Some(true));
        assert_eq!(cull_key_direction(Key::Left), Some(false));
        assert_eq!(cull_key_direction(Key::Page_Up), Some(false));
        // Keys the grid and the metadata shortcuts own must fall through: mapping
        // one here would swallow it in culling and break that shortcut silently.
        for key in [Key::Up, Key::Down, Key::Return, Key::F1, Key::_0, Key::_5, Key::Escape] {
            assert_eq!(cull_key_direction(key), None, "{key:?} must not page");
        }
    }

    #[test]
    fn refused_view_mode_leaves_the_current_mode_untouched() {
        // The property the switcher's persist path depends on: a refused switch is
        // a no-op, not a half-applied one. `store_view_mode` is the pure half of
        // `set_view_mode` precisely so this needs no display.
        apply_view_mode_token(view_mode_token_for(ViewMode::FileManager));
        for mode in ViewMode::ALL.into_iter().filter(|m| !m.is_available()) {
            assert!(!store_view_mode(mode), "unavailable mode must be refused: {mode:?}");
            assert_eq!(
                current_view_mode(),
                ViewMode::FileManager,
                "a refused switch must not mutate the current mode: {mode:?}"
            );
        }
    }

    #[test]
    fn current_view_mode_is_always_renderable() {
        // The invariant the switcher silently depends on when it seeds its buttons:
        // GTK's `set_active` ignores sensitivity, so a current mode whose button is
        // insensitive would pin the group on a button the user can never click off.
        // Every writer of VIEW_MODE must preserve this.
        for tok in ["", "garbage", "culling", "zoomable", "filemanager", "FileManager"] {
            apply_view_mode_token(tok);
            assert!(
                current_view_mode().is_available(),
                "token {tok:?} left the lighttable in an unrenderable mode"
            );
        }
        for mode in ViewMode::ALL {
            store_view_mode(mode);
            assert!(current_view_mode().is_available(), "after storing {mode:?}");
        }
    }

    #[test]
    fn rating_filter_token_encode_decode_roundtrips_full_matrix() {
        use RatingCompare::*;
        // Guard the ENCODER too (not just the decoder): encode∘decode is identity
        // for every (comparator, star) the UI can produce, modulo the codec's two
        // intentional canonicalisations — AtLeast 0 ⇒ "off", and Rejected drops the
        // star count. A prefix typo on either side would break this.
        for cmp in [AtLeast, Exactly, AtMost, Rejected] {
            for s in 0..=5u8 {
                let tok = rating_filter_token_for(cmp, s);
                let want = match cmp {
                    Rejected => (Rejected, 0),
                    _ => (cmp, s),
                };
                assert_eq!(parse_rating_filter_token(&tok), want, "tok={tok}");
            }
        }
    }

    #[test]
    fn flags_rating_bits_roundtrip_and_preserve_other_bits() {
        // Star value lives in bits 0–2; write/read must round-trip and NEVER touch
        // the reject bit (8) or any high flag (e.g. RAW/LDR at bits 6/10/16 — the
        // shape of Nicola's real darktable-written catalog).
        let base = 0x10440; // bits 6, 10, 16 set (no rating, no reject)
        for r in 0..=5u8 {
            let f = flags_with_star_rating(base, r);
            assert_eq!(flags_star_rating(f), r, "roundtrip r={r}");
            assert_eq!(f & !0x7, base, "high/reject bits preserved r={r}");
        }
        // Re-rating an image that carries the reject bit keeps reject set.
        let rejected = base | 0x8;
        let re = flags_with_star_rating(rejected, 4);
        assert_eq!(flags_star_rating(re), 4);
        assert_eq!(re & 0x8, 0x8, "reject bit survives a re-rate");
        // A legacy `flags & 7` of 6/7 clamps to 5 stars, never over-fills the row.
        assert_eq!(flags_star_rating(6), 5);
        assert_eq!(flags_star_rating(7), 5);
    }

    #[test]
    fn rating_filter_keeps_only_n_to_5_stars() {
        // End-to-end over the DARKTABLE bit convention (rating in bits 0–2, reject
        // in bit 3): the filter excludes unrated(0) and rejected — INCLUDING a
        // rejected image that still carries stars — while keeping N..5-star images.
        // Seeded via the exact write bit-maths save_rating uses (`(flags & ~7)|r`)
        // plus the reject bit, so this chains the writer to the filter reader (the
        // seam the bit-offset bug lived in). StringList can't be built headlessly.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT, flags INTEGER);
             INSERT INTO film_rolls VALUES (1, '/f');",
        )
        .unwrap();
        // (filename, star rating, rejected?) written the way the app writes them.
        let seed = [
            ("unrated.raw", 0u8, false),
            ("two.raw", 2, false),
            ("five.raw", 5, false),
            ("rejected.raw", 0, true),
            ("rej_five.raw", 5, true), // rejected AND 5-star — reject must win
        ];
        for (id, (name, rating, rejected)) in seed.iter().enumerate() {
            let base: i64 = 0x10440 | if *rejected { 0x8 } else { 0 }; // high bits + maybe reject
            let flags = flags_with_star_rating(base, *rating);
            conn.execute(
                "INSERT INTO images VALUES (?1, 1, ?2, ?3)",
                rusqlite::params![id as i64 + 1, name, flags],
            )
            .unwrap();
        }
        let run = |stars: u8, cmp: RatingCompare| -> Vec<String> {
            let sql = format!(
                "SELECT i.filename FROM images i JOIN film_rolls f ON f.id = i.film_id \
                 WHERE 1=1{} ORDER BY i.filename",
                rating_predicate(stars, cmp)
            );
            conn.prepare(&sql)
                .unwrap()
                .query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .flatten()
                .collect()
        };
        use RatingCompare::*;
        // No filter (AtLeast 0): all five (unrated, rejected, rejected-5-star included).
        assert_eq!(run(0, AtLeast), ["five.raw", "rej_five.raw", "rejected.raw", "two.raw", "unrated.raw"]);
        // ≥3 stars: only the non-rejected 5-star (rej_five is excluded by reject).
        assert_eq!(run(3, AtLeast), ["five.raw"]);
        // ≥1 star: 2- and 5-star; excludes unrated AND both rejected images.
        assert_eq!(run(1, AtLeast), ["five.raw", "two.raw"]);
        // = 5 stars: only the non-rejected five (reject wins over the rejected 5-star).
        assert_eq!(run(5, Exactly), ["five.raw"]);
        // = 0 stars: the unrated one only (rejected images are excluded, not "0-star").
        assert_eq!(run(0, Exactly), ["unrated.raw"]);
        // ≤ 2 stars: unrated + 2-star (excludes 5-star and both rejected).
        assert_eq!(run(2, AtMost), ["two.raw", "unrated.raw"]);
        // Rejected: both rejected images regardless of their star value; nothing else.
        assert_eq!(run(0, Rejected), ["rej_five.raw", "rejected.raw"]);
    }

    #[test]
    fn order_clause_valid_sql_with_undated_and_rejected() {
        // Run every SortOrder's clause end-to-end against a realistic fixture
        // (the columns/alias the loaders reference). Catches a clause that names
        // a column a loader's query doesn't provide, and locks the undated-last
        // (Q3) and rejected-bottom (Q4) placement. StringList can't be built
        // headlessly, so this exercises the shared clause directly via rusqlite.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT,
                                  datetime_taken INTEGER, flags INTEGER);
             INSERT INTO film_rolls VALUES (1, '/f');
             INSERT INTO images VALUES (1, 1, 'a.raw', 100, 8);  -- dated,   rejected (bit 3)
             INSERT INTO images VALUES (2, 1, 'b.raw', 200, 5);  -- dated,   5-star
             INSERT INTO images VALUES (3, 1, 'c.raw', 0,   3);  -- undated, 3-star",
        )
        .unwrap();
        let run = |order: SortOrder, reverse: bool| -> Vec<String> {
            let sql = format!(
                "SELECT i.filename FROM images i JOIN film_rolls f ON f.id = i.film_id \
                 ORDER BY {}",
                order.order_clause(reverse)
            );
            conn.prepare(&sql)
                .unwrap()
                .query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .flatten()
                .collect()
        };
        assert_eq!(run(SortOrder::Filename, false), ["a.raw", "b.raw", "c.raw"]);
        // Ascending date order for dated images, undated (c) pushed last.
        assert_eq!(run(SortOrder::DateTaken, false), ["a.raw", "b.raw", "c.raw"]);
        // Highest rating first; rejected (a) sorts below the 3-star (c).
        assert_eq!(run(SortOrder::Rating, false), ["b.raw", "c.raw", "a.raw"]);

        // Reversed: filename Z→A; date newest-first for dated images but undated
        // (c) STILL last (non-reversible guard); rating lowest-first with rejected
        // (a) now at the very top.
        assert_eq!(run(SortOrder::Filename, true), ["c.raw", "b.raw", "a.raw"]);
        assert_eq!(run(SortOrder::DateTaken, true), ["b.raw", "a.raw", "c.raw"]);
        assert_eq!(run(SortOrder::Rating, true), ["a.raw", "c.raw", "b.raw"]);
    }

    #[test]
    fn cap_rows_under_cap_passes_through_without_notice() {
        let (rows, notice) = cap_rows(n_rows(3));
        assert_eq!(rows.len(), 3);
        assert!(notice.is_none());
    }

    #[test]
    fn cap_rows_exactly_at_cap_has_no_notice() {
        // A full page is NOT truncation — the +1 fetch is what disambiguates.
        let (rows, notice) = cap_rows(n_rows(GRID_CAP));
        assert_eq!(rows.len(), GRID_CAP);
        assert!(notice.is_none());
    }

    #[test]
    fn cap_rows_over_cap_truncates_and_notices() {
        // Loaders fetch GRID_CAP + 1; that extra row triggers the notice and is
        // dropped so the grid still shows exactly GRID_CAP real items.
        let (rows, notice) = cap_rows(n_rows(GRID_CAP + 1));
        assert_eq!(rows.len(), GRID_CAP);
        let notice = notice.expect("over-cap result must carry a notice");
        assert!(!notice.contains('/'), "notice must be inert (no `/`)");
    }

    #[test]
    fn escape_like_passes_plain_text_through() {
        assert_eq!(escape_like("places"), "places");
        // The hierarchy separator is not a LIKE metachar, so it survives verbatim
        // (the `|%` descendant wildcard is appended by the caller, not here).
        assert_eq!(escape_like("places|Italy"), "places|Italy");
    }

    #[test]
    fn escape_like_escapes_metacharacters() {
        assert_eq!(escape_like("50%"), "50\\%");
        assert_eq!(escape_like("a_b"), "a\\_b");
    }

    #[test]
    fn escape_like_escapes_backslash_first() {
        // A literal backslash must become `\\` and must not double-escape the
        // escapes we add for `%`/`_` — order matters.
        assert_eq!(escape_like("a\\b"), "a\\\\b");
        assert_eq!(escape_like("a\\%b"), "a\\\\\\%b");
    }

    fn paths() -> Vec<String> {
        ["/a/1.jpg", "/a/2.jpg", "/b/3.jpg"].iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn index_of_path_finds_surviving_selection() {
        assert_eq!(index_of_path(&paths(), "/a/2.jpg"), Some(1));
        assert_eq!(index_of_path(&paths(), "/b/3.jpg"), Some(2));
    }

    #[test]
    fn index_of_path_none_when_dropped() {
        // e.g. the filtered-on tag was detached, so the image left the grid
        assert_eq!(index_of_path(&paths(), "/a/9.jpg"), None);
    }

    #[test]
    fn index_of_path_none_in_empty_grid() {
        assert_eq!(index_of_path(&[], "/a/1.jpg"), None);
    }

    #[test]
    fn color_dot_lit_uses_its_own_hue() {
        for idx in 0..COLOR_COUNT {
            let m = color_dot_markup(idx, true);
            assert!(m.contains(COLOR_HEX[idx as usize]), "idx {idx}: {m}");
            assert!(m.contains('\u{25cf}'));
        }
    }

    #[test]
    fn color_dot_unlit_is_dim_grey() {
        for idx in 0..COLOR_COUNT {
            let m = color_dot_markup(idx, false);
            assert!(m.contains(COLOR_DIM_HEX), "idx {idx}: {m}");
        }
    }

    #[test]
    fn fkey_maps_f1_through_f5_to_colour_indices() {
        use gtk4::gdk::Key;
        // F1..F5 → red/yellow/green/blue/purple, matching COLOR_HEX order and the
        // darktable accelerators. Off-by-one here would silently mislabel images.
        assert_eq!(fkey_to_color(Key::F1), Some(0));
        assert_eq!(fkey_to_color(Key::F2), Some(1));
        assert_eq!(fkey_to_color(Key::F3), Some(2));
        assert_eq!(fkey_to_color(Key::F4), Some(3));
        assert_eq!(fkey_to_color(Key::F5), Some(4));
        // Every mapped index stays inside the colour domain.
        for k in [Key::F1, Key::F2, Key::F3, Key::F4, Key::F5] {
            assert!(fkey_to_color(k).unwrap() < COLOR_COUNT);
        }
    }

    #[test]
    fn fkey_other_keys_are_ignored() {
        use gtk4::gdk::Key;
        // Keys just past the range and unrelated keys must not toggle anything.
        assert_eq!(fkey_to_color(Key::F6), None);
        assert_eq!(fkey_to_color(Key::F12), None);
        assert_eq!(fkey_to_color(Key::a), None);
        assert_eq!(fkey_to_color(Key::Return), None);
    }

    #[test]
    fn digit_maps_0_through_5_to_ratings_top_row_and_keypad() {
        use gtk4::gdk::Key;
        // 0..5 → rating 0..5 (darktable), identity mapping — an off-by-one would
        // silently misrate images. Both the top-row and keypad digits map the same.
        for (k, kp, r) in [
            (Key::_0, Key::KP_0, 0u8),
            (Key::_1, Key::KP_1, 1),
            (Key::_2, Key::KP_2, 2),
            (Key::_3, Key::KP_3, 3),
            (Key::_4, Key::KP_4, 4),
            (Key::_5, Key::KP_5, 5),
        ] {
            assert_eq!(digit_to_rating(k), Some(r));
            assert_eq!(digit_to_rating(kp), Some(r));
        }
    }

    #[test]
    fn digit_other_keys_are_ignored() {
        use gtk4::gdk::Key;
        // 6..9 are out of the 0–5 rating range; unrelated keys map to None too.
        assert_eq!(digit_to_rating(Key::_6), None);
        assert_eq!(digit_to_rating(Key::_9), None);
        assert_eq!(digit_to_rating(Key::KP_9), None);
        assert_eq!(digit_to_rating(Key::a), None);
        assert_eq!(digit_to_rating(Key::Return), None);
    }

    #[test]
    fn color_dot_out_of_range_falls_back_to_grey() {
        // Even "lit", an index past the palette can't panic and shows grey.
        assert!(color_dot_markup(COLOR_COUNT, true).contains(COLOR_DIM_HEX));
        assert!(color_dot_markup(99, true).contains(COLOR_DIM_HEX));
    }
}
