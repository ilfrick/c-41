//! Side-panel widgets — collections list and metadata inspector.
//!
//! Phase 3-ui-3: left panel shows film rolls with image counts from DB;
//! right panel shows a metadata stub (to be wired to image selection).

use adw::prelude::*;

/// Build the collections (left) panel.
///
/// Lists all film rolls from the database with their image counts. Pass the
/// same `db_path` that was given to `lighttable_load_from_db`; an empty string
/// falls back to the in-memory demo database.
pub fn left_panel(db_path: &str) -> gtk4::Box {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .width_request(210)
        .build();

    // Section header
    let header = gtk4::Label::builder()
        .label("Collections")
        .halign(gtk4::Align::Start)
        .margin_top(12)
        .margin_bottom(6)
        .margin_start(12)
        .margin_end(12)
        .build();
    header.add_css_class("heading");
    panel.append(&header);
    panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

    // Scrollable list of film rolls
    let list_box = gtk4::ListBox::builder()
        .selection_mode(gtk4::SelectionMode::Single)
        .build();
    list_box.add_css_class("navigation-sidebar");

    let rolls = load_film_rolls(db_path);
    if rolls.is_empty() {
        let row = gtk4::ListBoxRow::new();
        let lbl = gtk4::Label::builder()
            .label("No collections yet")
            .halign(gtk4::Align::Start)
            .margin_start(12)
            .margin_top(6)
            .margin_bottom(6)
            .build();
        lbl.add_css_class("dim-label");
        row.set_child(Some(&lbl));
        list_box.append(&row);
    } else {
        for (folder, count) in &rolls {
            let row = gtk4::ListBoxRow::new();
            let hbox = gtk4::Box::builder()
                .orientation(gtk4::Orientation::Horizontal)
                .spacing(8)
                .margin_start(12).margin_end(8)
                .margin_top(6).margin_bottom(6)
                .build();

            // Show only the last path component as the collection name
            let name = folder.rsplit('/').next().unwrap_or(folder);
            let name_lbl = gtk4::Label::builder()
                .label(name)
                .halign(gtk4::Align::Start)
                .hexpand(true)
                .ellipsize(gtk4::pango::EllipsizeMode::Middle)
                .build();

            let count_lbl = gtk4::Label::builder()
                .label(&count.to_string())
                .halign(gtk4::Align::End)
                .build();
            count_lbl.add_css_class("dim-label");
            count_lbl.add_css_class("numeric");

            hbox.append(&name_lbl);
            hbox.append(&count_lbl);
            row.set_child(Some(&hbox));
            row.set_tooltip_text(Some(folder));
            list_box.append(&row);
        }
    }

    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .child(&list_box)
        .vexpand(true)
        .build();
    panel.append(&scroll);

    panel
}

/// Build the metadata (right) panel.
pub fn right_panel() -> gtk4::Box {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .width_request(210)
        .build();

    let header = gtk4::Label::builder()
        .label("Metadata")
        .halign(gtk4::Align::Start)
        .margin_top(12)
        .margin_bottom(6)
        .margin_start(12)
        .margin_end(12)
        .build();
    header.add_css_class("heading");
    panel.append(&header);
    panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

    let placeholder = gtk4::Label::builder()
        .label("Select an image\nto view metadata")
        .halign(gtk4::Align::Center)
        .valign(gtk4::Align::Center)
        .vexpand(true)
        .justify(gtk4::Justification::Center)
        .build();
    placeholder.add_css_class("dim-label");
    panel.append(&placeholder);

    panel
}

/// Query the database for film rolls and their image counts.
fn load_film_rolls(db_path: &str) -> Vec<(String, i64)> {
    let conn = if db_path.is_empty() {
        match rusqlite::Connection::open_in_memory() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        }
    } else {
        match rusqlite::Connection::open(db_path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        }
    };

    let mut rolls = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT f.folder, COUNT(i.id) \
         FROM main.film_rolls f \
         LEFT JOIN main.images i ON i.film_id = f.id \
         GROUP BY f.id, f.folder \
         ORDER BY f.folder",
    ) {
        let _ = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))
            .map(|rows| rolls.extend(rows.flatten()));
    }
    rolls
}
