//! Lighttable view -- async thumbnail grid + star ratings + colour labels.
//!
//! Phase 3-ui-8: each cell has a 5-star rating row that reads/writes
//! the rating from/to darkroom-db asynchronously.
//! Phase 3-m4-20: each cell also has a 5-dot colour-label row (red/yellow/
//! green/blue/purple); clicking a dot toggles that label via the
//! `darkroom_db::colorlabels` DAO, resolving the image id by path.

use adw::prelude::*;
use gtk4::{GridView, ListItem, ScrolledWindow, SignalListItemFactory, SingleSelection};
use glib::clone;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

pub const THUMB_SIZE: i32 = 160;

pub type LighttableModel = gtk4::StringList;

/// Build the lighttable widget. Returns (NavigationPage, model, selection).
///
/// `db_path` is stored in each cell's gesture handler for rating updates.
pub fn lighttable_page(db_path: String) -> (ScrolledWindow, LighttableModel, SingleSelection) {
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

        if !full_path.contains('/') {
            set_stars(&stars_box, 0);
            set_color_dots(&colors_box, 0);
            return;
        }

        // Async thumbnail load
        glib::spawn_future_local(clone!(@weak thumb => async move {
            let path = full_path.clone();
            let bytes = gio::spawn_blocking(move || std::fs::read(&path).ok())
                .await.ok().flatten();
            if thumb.widget_name() != full_path { return; } // cell rebound mid-read
            if let Some(data) = bytes {
                let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
                let _ = loader.write(&data);
                let _ = loader.close();
                if let Some(raw) = loader.pixbuf() {
                    if let Some(pb) = raw.scale_simple(
                        THUMB_SIZE, THUMB_SIZE, gtk4::gdk_pixbuf::InterpType::Bilinear,
                    ) {
                        thumb.set_paintable(Some(&gtk4::gdk::Texture::for_pixbuf(&pb)));
                    }
                }
            }
        }));

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
        .max_columns(12)
        .min_columns(2)
        .build();
    grid.add_css_class("lighttable-grid");

    let scroll = ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .child(&grid)
        .vexpand(true)
        .build();

    (scroll, model, selection)
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

/// Read an image's 0–5 star rating (from `images.flags` bits 1–3) by path, or
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
    Some(((flags >> 1) & 7) as u8)
}

fn save_rating(full_path: &str, db_path: &str, rating: u8) -> rusqlite::Result<()> {
    if db_path.is_empty() { return Ok(()); }
    let conn     = open_rating_conn(db_path)?;
    let p        = std::path::Path::new(full_path);
    let filename = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let folder   = p.parent().and_then(|d| d.to_str()).unwrap_or("");
    let bits     = (rating as i64 & 7) << 1;
    conn.execute(
        "UPDATE main.images SET flags = (flags & ~14) | ?1 \
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

/// Number of colour labels, mirroring `darkroom_db::colorlabels::COLOR_COUNT`.
/// `pub(crate)` so the left-panel colour filter (`panels`) can iterate the same
/// colour domain and render matching swatches without redefining it.
pub(crate) const COLOR_COUNT: u8 = darkroom_db::colorlabels::COLOR_COUNT;

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
/// grid cells (one source of truth for colour rendering across `darkroom-ui`).
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
    let imgid = match darkroom_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        _ => return 0,
    };
    darkroom_db::colorlabels::color_labels_get(&conn, imgid).unwrap_or(0)
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

/// Toggle colour label `color` on the grid's currently-selected image and repaint
/// that one cell's dot row in place (no full reload, so scroll position and the
/// other cells' in-flight async loads are untouched). No-op when nothing real is
/// selected (a placeholder row carries no `/`, so `selected_path` returns `None`).
///
/// The DB write runs off the main thread; the in-place repaint then targets the
/// realized cell still bound to `path` — matched exactly like `wire_color_clicks`,
/// except the keyboard path holds no `colors_box` reference, so it must *find* the
/// row among the grid's cells (see [`repaint_color_dots_for_path`]). If the cell
/// was recycled or scrolled off-screen by the time the toggle resolves, the
/// repaint is a no-op and the next bind paints the new mask from the DB.
pub fn toggle_selected_color(
    grid: &GridView,
    selection: &SingleSelection,
    db_path: &str,
    color: u8,
) {
    let Some(path) = selected_path(selection) else { return };
    let db = db_path.to_string();
    glib::spawn_future_local(clone!(@weak grid => async move {
        let p   = path.clone();
        let db2 = db.clone();
        // Skip the repaint on a worker panic (mask unknown — don't clear labels
        // still in the DB); the next rebind paints the row from the DB.
        if let Some(mask) =
            serialized_write(path.clone(), move || toggle_color_label(&p, &db2, color)).await
        {
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

/// Set the star `rating` on the grid's currently-selected image and repaint that
/// cell's star row in place (m4-29). Star sibling of [`toggle_selected_color`],
/// except ratings are an ABSOLUTE set (digit `k` → rating `k`, matching the star
/// click handler and darktable) rather than a toggle. No-op when nothing real is
/// selected. The DB write runs off the main thread; the in-place repaint then
/// targets the realized cell still bound to `path` (a no-op if it was recycled /
/// scrolled off, the next bind painting from the DB).
pub fn set_selected_rating(
    grid: &GridView,
    selection: &SingleSelection,
    db_path: &str,
    rating: u8,
) {
    let Some(path) = selected_path(selection) else { return };
    let db = db_path.to_string();
    glib::spawn_future_local(clone!(@weak grid => async move {
        let p   = path.clone();
        let db2 = db.clone();
        serialized_write(path.clone(), move || {
            if let Err(e) = save_rating(&p, &db2, rating) {
                eprintln!("darkroom: save rating failed for {p}: {e}");
            }
        }).await;
        repaint_stars_for_path(&grid, &path, rating);
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
    let imgid = match darkroom_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        _ => return 0,
    };
    if let Err(e) = darkroom_db::colorlabels::color_label_toggle(&conn, imgid, color) {
        eprintln!("darkroom: colour-label toggle failed: {e}");
    }
    darkroom_db::colorlabels::color_labels_get(&conn, imgid).unwrap_or(0)
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
    /// The `ORDER BY` expression (without the `ORDER BY` keyword). Static text —
    /// never user input — so it's injection-safe to interpolate. All loaders
    /// alias the images table as `i`, so these column refs are valid everywhere.
    fn order_clause(self) -> &'static str {
        match self {
            // Filename groups naturally by folder (dates are foldered YYYY_MM_DD).
            SortOrder::Filename => "f.folder, i.filename",
            // Undated images (NULL or 0) sort LAST: the leading boolean is 0 for
            // dated, 1 for undated, so ASC keeps dated photos in date order up top
            // and dumps undated at the end. Tie-break by name for a stable order.
            SortOrder::DateTaken => {
                "(i.datetime_taken IS NULL OR i.datetime_taken = 0), i.datetime_taken, i.filename"
            }
            // darktable packs the star rating in the low 3 bits of flags (0..5,
            // 6 = rejected). Highest rating first, but rejected is the *bottom*,
            // not the top — map it below 0 so DESC doesn't rank it above 5 stars.
            SortOrder::Rating => {
                "CASE (i.flags & 7) WHEN 6 THEN -1 ELSE (i.flags & 7) END DESC, i.filename"
            }
        }
    }
}

thread_local! {
    /// The current grid sort order (main-thread-only UI state).
    static SORT_ORDER: Cell<SortOrder> = const { Cell::new(SortOrder::Filename) };
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
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };

    let order = current_sort().order_clause();
    let rows: Vec<String> = match folder {
        Some(f) => {
            conn.prepare(&format!(
                "SELECT f.folder || '/' || i.filename \
                 FROM main.images i \
                 JOIN main.film_rolls f ON f.id = i.film_id \
                 WHERE f.folder = ?1 \
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
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };
    let order = current_sort().order_clause();
    let pattern = format!("%{query}%");
    let rows: Vec<String> = conn
        .prepare(&format!(
            "SELECT f.folder || '/' || i.filename \
             FROM main.images i JOIN main.film_rolls f ON f.id = i.film_id \
             WHERE i.filename LIKE ?1 \
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
/// (panels::load_tags_with_counts), which queries `data.tags` over the same bare
/// `Connection::open`, so whenever a clickable node exists this JOIN is reachable
/// too. If the tag tree is ever re-sourced (e.g. cached), revisit this. `prefix` is
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
        darkroom_db::schema::open_catalog(db_path).unwrap_or_else(|_| open_demo_db())
    };
    let order = current_sort().order_clause();
    // `prefix` itself (the exact tag) OR `prefix|…` (any descendant). DISTINCT
    // because an image carrying several tags under `prefix` would otherwise
    // appear once per matching tag.
    let descendants = format!("{}|%", escape_like(prefix));
    let rows: Vec<String> = match conn
        .prepare(&format!(
            "SELECT DISTINCT f.folder || '/' || i.filename \
             FROM main.images i \
             JOIN main.film_rolls f ON f.id = i.film_id \
             JOIN main.tagged_images ti ON ti.imgid = i.id \
             JOIN data.tags t ON t.id = ti.tagid \
             WHERE t.name = ?1 OR t.name LIKE ?2 ESCAPE '\\' \
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
fn build_color_mask_query(mask: u8, match_all: bool, sort: SortOrder) -> Option<String> {
    let colors = colors_from_mask(mask);
    if colors.is_empty() {
        return None;
    }
    let in_list = colors.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
    let limit = GRID_CAP + 1;
    let order = sort.order_clause();
    let sql = if match_all {
        format!(
            "SELECT f.folder || '/' || i.filename \
             FROM main.images i \
             JOIN main.film_rolls f ON f.id = i.film_id \
             JOIN main.color_labels cl ON cl.imgid = i.id \
             WHERE cl.color IN ({in_list}) \
             GROUP BY i.id HAVING COUNT(DISTINCT cl.color) = {n} \
             ORDER BY {order} LIMIT {limit}",
            n = colors.len(),
        )
    } else {
        format!(
            "SELECT DISTINCT f.folder || '/' || i.filename \
             FROM main.images i \
             JOIN main.film_rolls f ON f.id = i.film_id \
             JOIN main.color_labels cl ON cl.imgid = i.id \
             WHERE cl.color IN ({in_list}) \
             ORDER BY {order} LIMIT {limit}",
        )
    };
    Some(sql)
}

/// Reload the grid to show images matching a colour-label `mask` under AND
/// (`match_all`) / OR semantics — see [`build_color_mask_query`]. An **empty mask
/// shows all images** (the no-colour-filter state), so the left panel can route
/// every colour-filter change here without special-casing "nothing selected".
/// The single-colour case is just a one-bit mask, so this is the sole colour
/// loader the panel needs (m4-26).
pub fn lighttable_load_by_color_mask(
    model: &LighttableModel,
    db_path: &str,
    mask: u8,
    match_all: bool,
) {
    register_reload({
        let m = model.clone();
        let db = db_path.to_string();
        move || lighttable_load_by_color_mask(&m, &db, mask, match_all)
    });
    let Some(sql) = build_color_mask_query(mask, match_all, current_sort()) else {
        lighttable_load_from_db(model, db_path);
        return;
    };
    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };
    let rows: Vec<String> = match conn
        .prepare(&sql)
        .and_then(|mut s| {
            s.query_map([], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect())
        }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!(
                "darkroom: colour-mask filter query failed (mask {mask:05b}, all={match_all}): {e}"
            );
            Vec::new()
        }
    };
    fill_grid(model, rows, "(No images with these colour labels)");
}

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
pub fn reselect_path(selection: &SingleSelection, prev: Option<&str>) {
    let Some(prev) = prev else { return };
    let Some(model) = selection.model() else { return };
    let paths: Vec<String> = (0..model.n_items())
        .filter_map(|i| {
            model.item(i).and_downcast::<gtk4::StringObject>().map(|o| o.string().to_string())
        })
        .collect();
    if let Some(idx) = index_of_path(&paths, prev) {
        selection.set_selected(idx);
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
    conn.execute_batch(
        "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder VARCHAR, access_timestamp INTEGER);
         CREATE TABLE images    (id INTEGER PRIMARY KEY, film_id INTEGER, filename VARCHAR,
                                 width INTEGER, height INTEGER, flags INTEGER, datetime_taken INTEGER);
         INSERT INTO film_rolls VALUES (1, '/photos/demo', 0);
         INSERT INTO images VALUES (1, 1, 'DSC_0001.jpg', 6000, 4000, 0, 100);
         INSERT INTO images VALUES (2, 1, 'DSC_0002.jpg', 6000, 4000, 0, 200);
         INSERT INTO images VALUES (3, 1, 'DSC_0003.jpg', 6000, 4000, 0, 300);",
    )
    .expect("demo data");
    conn
}

#[cfg(test)]
mod tests {
    use super::{build_color_mask_query, cap_rows, color_dot_markup, colors_from_mask,
                digit_to_rating, escape_like, fkey_to_color, index_of_path, path_write_lock,
                SortOrder, COLOR_COUNT, COLOR_DIM_HEX, COLOR_HEX, GRID_CAP};
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
    fn build_color_mask_query_none_for_empty_mask() {
        assert!(build_color_mask_query(0, false, SortOrder::Filename).is_none());
        assert!(build_color_mask_query(0, true, SortOrder::Filename).is_none());
    }

    #[test]
    fn build_color_mask_query_or_uses_distinct_no_having() {
        let sql = build_color_mask_query(0b10101, false, SortOrder::Filename).expect("non-empty mask");
        assert!(sql.contains("SELECT DISTINCT"), "{sql}");
        assert!(sql.contains("cl.color IN (0,2,4)"), "{sql}");
        assert!(!sql.contains("HAVING"), "OR must not group/having: {sql}");
        assert!(sql.contains(&format!("LIMIT {}", GRID_CAP + 1)), "{sql}");
    }

    #[test]
    fn build_color_mask_query_and_groups_and_counts_selected() {
        let sql = build_color_mask_query(0b01010, true, SortOrder::Filename).expect("non-empty mask");
        assert!(sql.contains("cl.color IN (1,3)"), "{sql}");
        // AND => image must carry both selected colours; N = popcount(mask).
        assert!(sql.contains("HAVING COUNT(DISTINCT cl.color) = 2"), "{sql}");
        assert!(!sql.contains("SELECT DISTINCT"), "AND path must not DISTINCT: {sql}");
    }

    #[test]
    fn build_color_mask_query_single_colour_counts_one() {
        // The single-colour case (one-bit mask) collapses to N=1 under AND.
        let sql = build_color_mask_query(0b00100, true, SortOrder::Filename).expect("non-empty mask");
        assert!(sql.contains("cl.color IN (2)"), "{sql}");
        assert!(sql.contains("HAVING COUNT(DISTINCT cl.color) = 1"), "{sql}");
    }

    #[test]
    fn sort_order_clauses_are_stable() {
        assert_eq!(SortOrder::Filename.order_clause(), "f.folder, i.filename");
        // Undated (NULL/0) sorts last via the leading boolean key.
        assert_eq!(
            SortOrder::DateTaken.order_clause(),
            "(i.datetime_taken IS NULL OR i.datetime_taken = 0), i.datetime_taken, i.filename"
        );
        // Rejected (rating 6) is remapped below 0 so it sorts under 5-star.
        assert_eq!(
            SortOrder::Rating.order_clause(),
            "CASE (i.flags & 7) WHEN 6 THEN -1 ELSE (i.flags & 7) END DESC, i.filename"
        );
    }

    #[test]
    fn color_mask_query_applies_sort_order() {
        let sql = build_color_mask_query(0b00100, false, SortOrder::Rating).expect("mask");
        assert!(
            sql.contains("ORDER BY CASE (i.flags & 7) WHEN 6 THEN -1 ELSE (i.flags & 7) END DESC"),
            "{sql}"
        );
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
             INSERT INTO images VALUES (1, 1, 'a.raw', 100, 6);  -- dated,   rejected
             INSERT INTO images VALUES (2, 1, 'b.raw', 200, 5);  -- dated,   5-star
             INSERT INTO images VALUES (3, 1, 'c.raw', 0,   3);  -- undated, 3-star",
        )
        .unwrap();
        let run = |order: SortOrder| -> Vec<String> {
            let sql = format!(
                "SELECT i.filename FROM images i JOIN film_rolls f ON f.id = i.film_id \
                 ORDER BY {}",
                order.order_clause()
            );
            conn.prepare(&sql)
                .unwrap()
                .query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .flatten()
                .collect()
        };
        assert_eq!(run(SortOrder::Filename), ["a.raw", "b.raw", "c.raw"]);
        // Ascending date order for dated images, undated (c) pushed last.
        assert_eq!(run(SortOrder::DateTaken), ["a.raw", "b.raw", "c.raw"]);
        // Highest rating first; rejected (a) sorts below the 3-star (c).
        assert_eq!(run(SortOrder::Rating), ["b.raw", "c.raw", "a.raw"]);
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
