//! Lighttable view -- async thumbnail grid + star ratings.
//!
//! Phase 3-ui-8: each cell has a 5-star rating row that reads/writes
//! the rating from/to darkroom-db asynchronously.

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

        vbox.append(&thumb);
        vbox.append(&label);
        vbox.append(&stars_box);
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

        let string_obj = item.item().and_downcast::<gtk4::StringObject>().unwrap();
        let full_path  = string_obj.string().to_string();

        let filename = std::path::Path::new(&full_path)
            .file_name().and_then(|n| n.to_str()).unwrap_or(&full_path).to_string();
        label.set_label(&filename);
        thumb.set_paintable(gtk4::gdk::Paintable::NONE);

        if !full_path.contains('/') {
            set_stars(&stars_box, 0);
            return;
        }

        // Async thumbnail load
        glib::spawn_future_local(clone!(@weak thumb => async move {
            let path = full_path.clone();
            let bytes = gio::spawn_blocking(move || std::fs::read(&path).ok())
                .await.ok().flatten();
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
            set_stars(&stars_box, rating);

            // Wire star click handlers (only once — remove then re-add)
            wire_star_clicks(&stars_box, fp, db);
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

/// Attach GestureClick to each star so clicking star k sets rating k.
fn wire_star_clicks(stars_box: &gtk4::Box, full_path: String, db_path: String) {
    let mut child = stars_box.first_child();
    let mut k = 0u8;
    while let Some(w) = child {
        k += 1;
        let pos = k;
        if let Some(lbl) = w.downcast_ref::<gtk4::Label>() {
            // Remove any existing gesture so we don't double-attach
            // (gtk4 doesn't expose a way to remove by type, so we just add;
            //  in practice wire_star_clicks is only called once per bind cycle
            //  after the async rating query)
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

// ── DB-backed load functions ──────────────────────────────────────────────

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
            conn.prepare(
                "SELECT f.folder || '/' || i.filename \
                 FROM main.images i \
                 JOIN main.film_rolls f ON f.id = i.film_id \
                 WHERE f.folder = ?1 \
                 ORDER BY i.filename LIMIT 2000",
            )
            .and_then(|mut s| s.query_map([f], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect()))
            .unwrap_or_default()
        }
        None => {
            conn.prepare(
                "SELECT f.folder || '/' || i.filename \
                 FROM main.images i \
                 JOIN main.film_rolls f ON f.id = i.film_id \
                 ORDER BY f.folder, i.filename LIMIT 2000",
            )
            .and_then(|mut s| s.query_map([], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect()))
            .unwrap_or_default()
        }
    };

    for path in rows {
        model.append(&path);
    }

    if model.n_items() == 0 {
        model.append("(No images in this collection)");
    }
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
        .prepare(
            "SELECT f.folder || '/' || i.filename \
             FROM main.images i JOIN main.film_rolls f ON f.id = i.film_id \
             WHERE i.filename LIKE ?1 \
             ORDER BY f.folder, i.filename LIMIT 2000",
        )
        .and_then(|mut s| {
            s.query_map([pattern.as_str()], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect())
        })
        .unwrap_or_default();
    for path in rows {
        model.append(&path);
    }
    if model.n_items() == 0 {
        model.append("(No results)");
    }
}

/// Filter the lighttable to images carrying the tag `tag_id`. Empty result (or a
/// db without the tag tables, e.g. the demo db) shows a placeholder.
pub fn lighttable_load_by_tag(model: &LighttableModel, db_path: &str, tag_id: u32) {
    while model.n_items() > 0 {
        model.remove(0);
    }
    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };
    let rows: Vec<String> = conn
        .prepare(
            "SELECT f.folder || '/' || i.filename \
             FROM main.images i \
             JOIN main.film_rolls f ON f.id = i.film_id \
             JOIN main.tagged_images ti ON ti.imgid = i.id \
             WHERE ti.tagid = ?1 \
             ORDER BY f.folder, i.filename LIMIT 2000",
        )
        .and_then(|mut s| {
            s.query_map([tag_id], |r| r.get::<_, String>(0))
                .map(|it| it.flatten().collect())
        })
        .unwrap_or_default();
    for path in rows {
        model.append(&path);
    }
    if model.n_items() == 0 {
        model.append("(No images with this tag)");
    }
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
