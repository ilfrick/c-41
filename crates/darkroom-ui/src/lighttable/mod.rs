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

pub const THUMB_SIZE: i32 = 160;

pub type LighttableModel = gtk4::StringList;

/// Build the lighttable widget. Returns (NavigationPage, model, selection).
///
/// `db_path` is stored in each cell's gesture handler for rating updates.
pub fn lighttable_page(db_path: String) -> (adw::NavigationPage, LighttableModel, SingleSelection) {
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
        let stars_box = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(1)
            .halign(gtk4::Align::Center)
            .build();
        for _ in 0..5 {
            let star = gtk4::Label::new(Some("\u{2605}"));  // ★
            star.add_css_class("dim-label");
            stars_box.append(&star);
        }

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

    let page = adw::NavigationPage::builder()
        .title("Lighttable")
        .child(&scroll)
        .build();

    (page, model, selection)
}

// ── Rating helpers ────────────────────────────────────────────────────────

fn nth_child(b: &gtk4::Box, n: usize) -> Option<gtk4::Widget> {
    let mut child = b.first_child();
    for _ in 0..n {
        child = child.and_then(|w| w.next_sibling());
    }
    child
}

fn set_stars(stars_box: &gtk4::Box, rating: u8) {
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
/// prior bind's gestures first (see `clear_click_gestures`).
fn wire_star_clicks(stars_box: &gtk4::Box, full_path: String, db_path: String) {
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
                    let _ = gio::spawn_blocking(move || {
                        save_rating(&fp2, &db2, new_rating)
                    }).await;
                });
            });
            lbl.add_controller(gesture);
        }
        child = w.next_sibling();
    }
}

fn query_rating(full_path: &str, db_path: &str) -> Option<u8> {
    if db_path.is_empty() { return None; }
    let conn = rusqlite::Connection::open(db_path).ok()?;
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
    let conn     = rusqlite::Connection::open(db_path)?;
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
                    let mask = gio::spawn_blocking(move || toggle_color_label(&fp2, &db2, color))
                        .await.unwrap_or(0);
                    // Skip the repaint if the cell was recycled while the toggle ran.
                    if cb2.widget_name() == path {
                        set_color_dots(&cb2, mask);
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
        let mask = gio::spawn_blocking(move || toggle_color_label(&p, &db2, color))
            .await.unwrap_or(0);
        repaint_color_dots_for_path(&grid, &path, mask);
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

/// Depth-first search of `root`'s descendants for the colour-dot `Box` of the cell
/// currently bound to `path`. A cell vbox is recognised by its first child being
/// the thumbnail `Picture`; its 4th child (index 3) is the colour row. Returns the
/// first match, or `None` if no realized cell is showing `path`.
///
/// `widget_name` is the bind-time stamp and is shared by the cell's thumb/stars/
/// colour widgets, so we require BOTH the thumb AND the colour row to carry `path`
/// — a single stale stamp (a cell mid-recycle, where the bind hasn't re-stamped
/// every child yet) can't then mis-target a neighbouring cell. Cross-*cell*
/// uniqueness still rests on grid paths being distinct, the same assumption
/// [`index_of_path`] documents; the loaders' joins yield distinct `folder/filename`
/// rows today, so at most one realized cell carries a given `path`. Worst case if
/// that ever breaks is a transient repaint of a duplicate's twin that self-heals
/// on its next bind — the DB write (the source of truth) is unaffected.
fn find_color_box_for_path(root: &gtk4::Widget, path: &str) -> Option<gtk4::Box> {
    let mut child = root.first_child();
    while let Some(w) = child {
        if let Some(vbox) = w.downcast_ref::<gtk4::Box>() {
            if let Some(thumb) = vbox.first_child().and_downcast::<gtk4::Picture>() {
                if thumb.widget_name().as_str() == path {
                    if let Some(colors) = nth_child(vbox, 3).and_downcast::<gtk4::Box>() {
                        if colors.widget_name().as_str() == path {
                            return Some(colors);
                        }
                    }
                }
            }
        }
        if let Some(found) = find_color_box_for_path(&w, path) {
            return Some(found);
        }
        child = w.next_sibling();
    }
    None
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

/// Append a loader's rows to the (freshly-cleared) grid model: cap them, add the
/// truncation notice if any, and fall back to `empty_placeholder` when nothing
/// matched. The single place the cap/notice tail lives, shared by every loader.
fn fill_grid(model: &LighttableModel, rows: Vec<String>, empty_placeholder: &str) {
    let (rows, notice) = cap_rows(rows);
    for path in rows {
        model.append(&path);
    }
    if let Some(notice) = notice {
        model.append(&notice);
    }
    if model.n_items() == 0 {
        model.append(empty_placeholder);
    }
}

pub fn lighttable_load_from_db(model: &LighttableModel, db_path: &str) {
    lighttable_load_by_folder(model, db_path, None);
}

pub fn lighttable_load_by_folder(model: &LighttableModel, db_path: &str, folder: Option<&str>) {
    while model.n_items() > 0 {
        model.remove(0);
    }

    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };

    let rows: Vec<String> = match folder {
        Some(f) => {
            conn.prepare(&format!(
                "SELECT f.folder || '/' || i.filename \
                 FROM main.images i \
                 JOIN main.film_rolls f ON f.id = i.film_id \
                 WHERE f.folder = ?1 \
                 ORDER BY i.filename LIMIT {}", GRID_CAP + 1),
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
                 ORDER BY f.folder, i.filename LIMIT {}", GRID_CAP + 1),
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
    while model.n_items() > 0 {
        model.remove(0);
    }
    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };
    let pattern = format!("%{query}%");
    let rows: Vec<String> = conn
        .prepare(&format!(
            "SELECT f.folder || '/' || i.filename \
             FROM main.images i JOIN main.film_rolls f ON f.id = i.film_id \
             WHERE i.filename LIKE ?1 \
             ORDER BY f.folder, i.filename LIMIT {}", GRID_CAP + 1),
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
    while model.n_items() > 0 {
        model.remove(0);
    }
    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };
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
             ORDER BY f.folder, i.filename LIMIT {}", GRID_CAP + 1),
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

/// Reload the grid to show only images carrying colour label `color` (0 red,
/// 1 yellow, 2 green, 3 blue, 4 purple). Mirrors [`lighttable_load_by_tag_prefix`]
/// but keyed on `main.color_labels` instead of the tag tree. `color` is the same
/// `0..COLOR_COUNT` domain the DAO and grid dots use; an out-of-range value simply
/// matches no rows (the table never holds such a `color` — see
/// `darkroom_db::colorlabels`). The `color_labels` `UNIQUE(imgid, color)` index
/// means an image matches at most once, so no `DISTINCT` is needed.
pub fn lighttable_load_by_color(model: &LighttableModel, db_path: &str, color: u8) {
    while model.n_items() > 0 {
        model.remove(0);
    }
    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };
    let rows: Vec<String> = match conn
        .prepare(&format!(
            "SELECT f.folder || '/' || i.filename \
             FROM main.images i \
             JOIN main.film_rolls f ON f.id = i.film_id \
             JOIN main.color_labels cl ON cl.imgid = i.id \
             WHERE cl.color = ?1 \
             ORDER BY f.folder, i.filename LIMIT {}", GRID_CAP + 1),
        )
        .and_then(|mut s| {
            s.query_map(rusqlite::params![color], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect())
        }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("darkroom: colour-label filter query failed (color {color}): {e}");
            Vec::new()
        }
    };
    fill_grid(model, rows, "(No images with this colour label)");
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
                                 width INTEGER, height INTEGER, flags INTEGER);
         INSERT INTO film_rolls VALUES (1, '/photos/demo', 0);
         INSERT INTO images VALUES (1, 1, 'DSC_0001.jpg', 6000, 4000, 0);
         INSERT INTO images VALUES (2, 1, 'DSC_0002.jpg', 6000, 4000, 0);
         INSERT INTO images VALUES (3, 1, 'DSC_0003.jpg', 6000, 4000, 0);",
    )
    .expect("demo data");
    conn
}

#[cfg(test)]
mod tests {
    use super::{cap_rows, color_dot_markup, escape_like, fkey_to_color, index_of_path,
                COLOR_COUNT, COLOR_DIM_HEX, COLOR_HEX, GRID_CAP};

    fn n_rows(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("/a/{i}.jpg")).collect()
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
    fn color_dot_out_of_range_falls_back_to_grey() {
        // Even "lit", an index past the palette can't panic and shows grey.
        assert!(color_dot_markup(COLOR_COUNT, true).contains(COLOR_DIM_HEX));
        assert!(color_dot_markup(99, true).contains(COLOR_DIM_HEX));
    }
}
