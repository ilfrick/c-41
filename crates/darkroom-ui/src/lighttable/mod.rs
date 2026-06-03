//! Lighttable view — async thumbnail grid for browsing the image collection.
//!
//! Phase 3-ui-3: GtkGridView backed by a StringList of full file paths.
//! Each cell loads its thumbnail asynchronously via gio::spawn_blocking so
//! the UI stays responsive while decoding large RAW/JPEG files.

use adw::prelude::*;
use gtk4::{GridView, ListItem, ScrolledWindow, SignalListItemFactory, SingleSelection};
use glib::clone;

pub const THUMB_SIZE: i32 = 160;

/// The shared model: each string is the absolute path to an image file.
pub type LighttableModel = gtk4::StringList;

/// Build the lighttable widget. Returns (NavigationPage, shared StringList).
pub fn lighttable_page() -> (adw::NavigationPage, LighttableModel) {
    let model     = gtk4::StringList::new(&[]);
    let selection = SingleSelection::new(Some(model.clone()));
    let factory   = SignalListItemFactory::new();

    // ── Setup: create widget tree for each visible cell ────────────────────
    factory.connect_setup(|_, list_item| {
        let item = list_item.downcast_ref::<ListItem>().unwrap();
        let vbox = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(4)
            .build();

        // Thumbnail image
        let thumb = gtk4::Picture::builder()
            .width_request(THUMB_SIZE)
            .height_request(THUMB_SIZE)
            .content_fit(gtk4::ContentFit::Cover)
            .build();
        thumb.add_css_class("frame");

        // Filename label
        let label = gtk4::Label::builder()
            .max_width_chars(16)
            .ellipsize(gtk4::pango::EllipsizeMode::End)
            .build();
        label.add_css_class("caption");

        vbox.append(&thumb);
        vbox.append(&label);
        item.set_child(Some(&vbox));
    });

    // ── Bind: fill widgets with item data + start async thumbnail load ─────
    factory.connect_bind(|_, list_item| {
        let item  = list_item.downcast_ref::<ListItem>().unwrap();
        let vbox  = item.child().and_downcast::<gtk4::Box>().unwrap();
        let thumb = vbox.first_child().and_downcast::<gtk4::Picture>().unwrap();
        let label = vbox.last_child().and_downcast::<gtk4::Label>().unwrap();

        let string_obj = item.item().and_downcast::<gtk4::StringObject>().unwrap();
        let full_path  = string_obj.string().to_string();

        // Set filename label
        let filename = std::path::Path::new(&full_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&full_path)
            .to_string();
        label.set_label(&filename);

        // Clear stale thumbnail from recycled cell
        thumb.set_paintable(gtk4::gdk::Paintable::NONE);

        // Skip placeholder strings (no '/')
        if !full_path.contains('/') {
            return;
        }

        // Async thumbnail decode:
        //   1. Read raw bytes on a thread pool (Vec<u8> is Send).
        //   2. Decode into a Pixbuf on the main thread (GObject, not Send).
        glib::spawn_future_local(clone!(@weak thumb => async move {
            let path = full_path.clone();

            // Step 1 — I/O on a thread pool
            let bytes = gio::spawn_blocking(move || std::fs::read(&path).ok())
                .await
                .ok()
                .flatten();

            // Step 2 — decode on the main thread
            if let Some(data) = bytes {
                let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
                let _ = loader.write(&data);
                let _ = loader.close();
                if let Some(raw) = loader.pixbuf() {
                    // Scale down to thumbnail size
                    if let Some(pb) = raw.scale_simple(
                        THUMB_SIZE, THUMB_SIZE,
                        gtk4::gdk_pixbuf::InterpType::Bilinear,
                    ) {
                        thumb.set_paintable(Some(&gtk4::gdk::Texture::for_pixbuf(&pb)));
                    }
                }
            }
        }));
    });

    // ── Unbind: clear thumbnail when cell scrolls off-screen ──────────────
    factory.connect_unbind(|_, list_item| {
        let item = list_item.downcast_ref::<ListItem>().unwrap();
        if let Some(vbox) = item.child().and_downcast::<gtk4::Box>() {
            if let Some(thumb) = vbox.first_child().and_downcast::<gtk4::Picture>() {
                thumb.set_paintable(gtk4::gdk::Paintable::NONE);
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

    (page, model)
}

/// Reload the lighttable from the database at `db_path`.
///
/// Stores absolute file paths (folder/filename) in the model so the thumbnail
/// loader can open them directly. Falls back to in-memory demo data when
/// `db_path` is empty or the file doesn't exist.
pub fn lighttable_load_from_db(model: &LighttableModel, db_path: &str) {
    while model.n_items() > 0 {
        model.remove(0);
    }

    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };

    // Return the full absolute path so the thumbnail loader can open it.
    if let Ok(mut stmt) = conn.prepare(
        "SELECT f.folder || '/' || i.filename \
         FROM main.images i \
         JOIN main.film_rolls f ON f.id = i.film_id \
         ORDER BY f.folder, i.filename \
         LIMIT 2000",
    ) {
        let _ = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map(|rows| {
                for path in rows.flatten() {
                    model.append(&path);
                }
            });
    }

    if model.n_items() == 0 {
        model.append("(No images — import a folder to begin)");
    }
}

/// Open a minimal in-memory demo database with sample data.
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
