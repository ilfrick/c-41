//! Side-panel widgets -- collections list and metadata inspector.
//!
//! Phase 3-ui-6:
//!   left_panel  -- film rolls with image counts; clicking filters the grid
//!   MetadataPanel -- right panel that updates when an image is selected

use adw::prelude::*;
use glib::clone;
use crate::lighttable::{LighttableModel, lighttable_load_by_folder};

// ── Left panel (collections) ──────────────────────────────────────────────

/// Build the collections (left) panel.
///
/// Clicking a film roll reloads `lt_model` to show only images from that
/// folder. The first row ("All images") clears the filter.
pub fn left_panel(db_path: &str, lt_model: &LighttableModel) -> gtk4::Box {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .width_request(210)
        .build();

    let header = gtk4::Label::builder()
        .label("Collections")
        .halign(gtk4::Align::Start)
        .margin_top(12).margin_bottom(6)
        .margin_start(12).margin_end(12)
        .build();
    header.add_css_class("heading");
    panel.append(&header);
    panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

    let list_box = gtk4::ListBox::builder()
        .selection_mode(gtk4::SelectionMode::Single)
        .build();
    list_box.add_css_class("navigation-sidebar");

    // "All images" row
    append_roll_row(&list_box, "All images", -1, None);

    // Film roll rows from DB
    let rolls = load_film_rolls(db_path);
    for (folder, count) in &rolls {
        append_roll_row(&list_box, folder, *count, Some(folder.as_str()));
    }

    // Activate: reload lighttable with folder filter
    let db = db_path.to_string();
    list_box.connect_row_activated(clone!(@weak lt_model => move |_, row| {
        let folder_filter: Option<String> = row
            .widget_name()
            .as_str()
            .ne("all")
            .then(|| row.widget_name().to_string());
        lighttable_load_by_folder(
            &lt_model,
            &db,
            folder_filter.as_deref(),
        );
    }));

    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .child(&list_box)
        .vexpand(true)
        .build();
    panel.append(&scroll);
    panel
}

fn append_roll_row(list_box: &gtk4::ListBox, label: &str, count: i64, folder: Option<&str>) {
    let row = gtk4::ListBoxRow::new();
    // Encode the folder in the widget name; "all" means no filter
    row.set_widget_name(folder.unwrap_or("all"));

    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .margin_start(12).margin_end(8)
        .margin_top(6).margin_bottom(6)
        .build();

    let name_str = if folder.is_none() {
        label.to_string()
    } else {
        label.rsplit('/').next().unwrap_or(label).to_string()
    };
    let name_lbl = gtk4::Label::builder()
        .label(&name_str)
        .halign(gtk4::Align::Start)
        .hexpand(true)
        .ellipsize(gtk4::pango::EllipsizeMode::Middle)
        .build();

    hbox.append(&name_lbl);

    if count >= 0 {
        let count_lbl = gtk4::Label::builder()
            .label(&count.to_string())
            .halign(gtk4::Align::End)
            .build();
        count_lbl.add_css_class("dim-label");
        count_lbl.add_css_class("numeric");
        hbox.append(&count_lbl);
    }

    if folder.is_some() {
        row.set_tooltip_text(Some(label));
    }
    row.set_child(Some(&hbox));
    list_box.append(&row);
}

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

// ── Right panel (metadata) ────────────────────────────────────────────────

/// Metadata inspector widget with an `update` method.
///
/// All label references are GTK GObject ref-counts so `MetadataPanel` is
/// cheaply cloneable -- just clone it to share with a selection callback.
#[derive(Clone)]
pub struct MetadataPanel {
    pub widget: gtk4::Box,
    filename_lbl: gtk4::Label,
    folder_lbl:   gtk4::Label,
    dims_lbl:     gtk4::Label,
    size_lbl:     gtk4::Label,
}

impl MetadataPanel {
    pub fn new() -> Self {
        let panel = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(0)
            .width_request(210)
            .build();

        let header = gtk4::Label::builder()
            .label("Metadata")
            .halign(gtk4::Align::Start)
            .margin_top(12).margin_bottom(6)
            .margin_start(12).margin_end(12)
            .build();
        header.add_css_class("heading");
        panel.append(&header);
        panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

        let grid = gtk4::Grid::builder()
            .row_spacing(4)
            .column_spacing(8)
            .margin_start(12).margin_end(12)
            .margin_top(10)
            .build();

        let mk_key = |text: &str| {
            let l = gtk4::Label::builder().label(text).halign(gtk4::Align::End).build();
            l.add_css_class("dim-label");
            l
        };
        let mk_val = || {
            gtk4::Label::builder()
                .halign(gtk4::Align::Start)
                .hexpand(true)
                .max_width_chars(20)
                .ellipsize(gtk4::pango::EllipsizeMode::Middle)
                .build()
        };

        let filename_lbl = mk_val();
        let folder_lbl   = mk_val();
        let dims_lbl     = mk_val();
        let size_lbl     = mk_val();

        let rows: [(&str, &gtk4::Label); 4] = [
            ("File",   &filename_lbl),
            ("Folder", &folder_lbl),
            ("Size",   &dims_lbl),
            ("Disk",   &size_lbl),
        ];
        for (i, (key, val)) in rows.iter().enumerate() {
            grid.attach(&mk_key(key), 0, i as i32, 1, 1);
            grid.attach(*val, 1, i as i32, 1, 1);
        }

        let placeholder = gtk4::Label::builder()
            .label("Select an image\nto view metadata")
            .halign(gtk4::Align::Center)
            .valign(gtk4::Align::Center)
            .vexpand(true)
            .justify(gtk4::Justification::Center)
            .build();
        placeholder.add_css_class("dim-label");

        panel.append(&grid);
        panel.append(&placeholder);

        Self { widget: panel, filename_lbl, folder_lbl, dims_lbl, size_lbl }
    }

    /// Update the panel with metadata for the image at `full_path`.
    pub fn update(&self, full_path: &str, db_path: &str) {
        use std::path::Path;

        let p = Path::new(full_path);
        let filename = p.file_name().and_then(|n| n.to_str()).unwrap_or(full_path);
        let folder   = p.parent().and_then(|d| d.to_str()).unwrap_or("");

        self.filename_lbl.set_label(filename);
        self.folder_lbl.set_label(
            folder.rsplit('/').next().unwrap_or(folder)
        );

        // Dimensions from DB
        let dims = query_dims(full_path, db_path)
            .map(|(w, h)| format!("{w} × {h}"))
            .unwrap_or_else(|| "—".to_string());
        self.dims_lbl.set_label(&dims);

        // File size on disk
        let disk = std::fs::metadata(full_path)
            .map(|m| format_bytes(m.len()))
            .unwrap_or_else(|_| "—".to_string());
        self.size_lbl.set_label(&disk);
    }
}

impl Default for MetadataPanel {
    fn default() -> Self { Self::new() }
}

fn query_dims(full_path: &str, db_path: &str) -> Option<(i64, i64)> {
    let conn = rusqlite::Connection::open(db_path).ok()?;
    let p = std::path::Path::new(full_path);
    let filename = p.file_name()?.to_str()?;
    let folder   = p.parent()?.to_str()?;
    conn.query_row(
        "SELECT i.width, i.height \
         FROM main.images i JOIN main.film_rolls f ON f.id = i.film_id \
         WHERE f.folder = ?1 AND i.filename = ?2",
        rusqlite::params![folder, filename],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    ).ok()
}

fn format_bytes(n: u64) -> String {
    if n < 1024 { format!("{n} B") }
    else if n < 1024 * 1024 { format!("{:.1} KB", n as f64 / 1024.0) }
    else { format!("{:.1} MB", n as f64 / (1024.0 * 1024.0)) }
}
