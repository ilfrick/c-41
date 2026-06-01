//! Lighttable view — thumbnail grid for browsing the image collection.
//!
//! Phase 3-ui-1: GtkGridView backed by a GtkStringList. Items show
//! image filenames as placeholders; Phase 3-ui-2 adds real thumbnail
//! pixbufs via a background load queue.

use adw::prelude::*;
use gtk4::{GridView, ListItem, ScrolledWindow, SignalListItemFactory, SingleSelection};

pub const THUMB_SIZE: i32 = 160;

/// The shared model that drives the grid.
pub type LighttableModel = gtk4::StringList;

/// Build the lighttable widget. Returns (NavigationPage, shared StringList).
/// The caller owns the StringList and can append/remove items at any time.
pub fn lighttable_page() -> (adw::NavigationPage, LighttableModel) {
    let model = gtk4::StringList::new(&[]);
    let selection = SingleSelection::new(Some(model.clone()));

    let factory = SignalListItemFactory::new();

    factory.connect_setup(|_, list_item| {
        let item = list_item.downcast_ref::<ListItem>().unwrap();
        let vbox = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(4)
            .build();
        // Thumbnail placeholder (grey box until real pixbuf loads)
        let thumb = gtk4::Picture::builder()
            .width_request(THUMB_SIZE)
            .height_request(THUMB_SIZE)
            .build();
        thumb.add_css_class("frame");
        // Filename label below thumbnail
        let label = gtk4::Label::builder()
            .max_width_chars(16)
            .ellipsize(gtk4::pango::EllipsizeMode::End)
            .build();
        label.add_css_class("caption");
        vbox.append(&thumb);
        vbox.append(&label);
        item.set_child(Some(&vbox));
    });

    factory.connect_bind(|_, list_item| {
        let item = list_item.downcast_ref::<ListItem>().unwrap();
        let vbox = item.child().and_downcast::<gtk4::Box>().unwrap();
        // Label is the second child
        let label = vbox.last_child().and_downcast::<gtk4::Label>().unwrap();
        let string_obj = item.item().and_downcast::<gtk4::StringObject>().unwrap();
        label.set_label(&string_obj.string());
    });

    let grid = GridView::builder()
        .model(&selection)
        .factory(&factory)
        .max_columns(12)
        .min_columns(2)
        .build();

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

/// Load filenames from the darkroom-db layer into the lighttable model.
///
/// Opens the library database at `db_path` (or uses an in-memory demo if
/// the path is empty / the file doesn't exist), queries all film rolls, and
/// populates the model with `folder/filename` strings.
pub fn lighttable_load_from_db(model: &LighttableModel, db_path: &str) {
    // Clear existing items
    while model.n_items() > 0 {
        model.remove(0);
    }

    let conn = if db_path.is_empty() {
        open_demo_db()
    } else {
        rusqlite::Connection::open(db_path).unwrap_or_else(|_| open_demo_db())
    };

    // Query all images with their folder path
    if let Ok(mut stmt) = conn.prepare(
        "SELECT f.folder || '/' || i.filename \
         FROM main.images i \
         JOIN main.film_rolls f ON f.id = i.film_id \
         ORDER BY f.folder, i.filename \
         LIMIT 500"
    ) {
        let _ = stmt.query_map([], |row| {
            let path: String = row.get(0)?;
            // Strip leading path — just show filename
            let filename = path.rsplit('/').next().unwrap_or(&path).to_string();
            Ok(filename)
        }).map(|rows| {
            for row in rows.flatten() {
                model.append(&row);
            }
        });
    }

    // If the DB was empty, show a friendly hint
    if model.n_items() == 0 {
        model.append("(No images — import a folder to begin)");
    }
}

/// Open a minimal in-memory demo database with sample data.
fn open_demo_db() -> rusqlite::Connection {
    use rusqlite::Connection;
    let conn = Connection::open_in_memory().expect("in-memory db");
    conn.execute_batch("
        CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder VARCHAR, access_timestamp INTEGER);
        CREATE TABLE images    (id INTEGER PRIMARY KEY, film_id INTEGER, filename VARCHAR,
                                width INTEGER, height INTEGER, flags INTEGER);
        INSERT INTO film_rolls VALUES (1, '/photos/demo', 0);
        INSERT INTO images VALUES (1, 1, 'DSC_0001.jpg', 6000, 4000, 0);
        INSERT INTO images VALUES (2, 1, 'DSC_0002.jpg', 6000, 4000, 0);
        INSERT INTO images VALUES (3, 1, 'DSC_0003.jpg', 6000, 4000, 0);
    ").expect("demo data");
    conn
}
