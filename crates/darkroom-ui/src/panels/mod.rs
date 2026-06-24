//! Side-panel widgets -- collections list and metadata inspector.
//!
//! Phase 3-ui-6:
//!   left_panel  -- film rolls with image counts; clicking filters the grid
//!   MetadataPanel -- right panel that updates when an image is selected

use adw::prelude::*;
use glib::clone;
use crate::lighttable::{LighttableModel, lighttable_load_by_folder, lighttable_load_by_tag};
use darkroom_db;

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

    // Both the Collections and Tags sections scroll together inside one content
    // box so neither steals the other's vertical space.
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .build();

    // ── Collections (film rolls) ──────────────────────────────────────────
    content.append(&section_header("Collections"));
    content.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

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
    content.append(&list_box);

    // ── Tags ──────────────────────────────────────────────────────────────
    // Only shown when the library has user tags. Clicking one filters the grid.
    let tags = load_tags_with_counts(db_path);
    if !tags.is_empty() {
        content.append(&section_header("Tags"));
        content.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

        let tag_box = gtk4::ListBox::builder()
            .selection_mode(gtk4::SelectionMode::Single)
            .build();
        tag_box.add_css_class("navigation-sidebar");
        for (id, name, count) in &tags {
            append_tag_row(&tag_box, *id, name, *count);
        }

        let db_tags = db_path.to_string();
        tag_box.connect_row_activated(clone!(@weak lt_model => move |_, row| {
            // The tag id is encoded in the row's widget name (see append_tag_row).
            if let Ok(tag_id) = row.widget_name().as_str().parse::<u32>() {
                lighttable_load_by_tag(&lt_model, &db_tags, tag_id);
            }
        }));
        content.append(&tag_box);
    }

    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .child(&content)
        .vexpand(true)
        .build();
    panel.append(&scroll);
    panel
}

/// A section heading label styled like the panel headers.
fn section_header(text: &str) -> gtk4::Label {
    let header = gtk4::Label::builder()
        .label(text)
        .halign(gtk4::Align::Start)
        .margin_top(12).margin_bottom(6)
        .margin_start(12).margin_end(12)
        .build();
    header.add_css_class("heading");
    header
}

/// Append a tag row carrying its attached-image count; the tag id is stashed in
/// the row's widget name so the activation handler can recover it.
fn append_tag_row(list_box: &gtk4::ListBox, id: u32, name: &str, count: i64) {
    let row = gtk4::ListBoxRow::new();
    row.set_widget_name(&id.to_string());

    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .margin_start(12).margin_end(8)
        .margin_top(6).margin_bottom(6)
        .build();

    let name_lbl = gtk4::Label::builder()
        .label(name)
        .halign(gtk4::Align::Start)
        .hexpand(true)
        .ellipsize(gtk4::pango::EllipsizeMode::Middle)
        .build();
    hbox.append(&name_lbl);

    let count_lbl = gtk4::Label::builder()
        .label(&count.to_string())
        .halign(gtk4::Align::End)
        .build();
    count_lbl.add_css_class("dim-label");
    count_lbl.add_css_class("numeric");
    hbox.append(&count_lbl);

    row.set_tooltip_text(Some(name));
    row.set_child(Some(&hbox));
    list_box.append(&row);
}

fn load_tags_with_counts(db_path: &str) -> Vec<(u32, String, i64)> {
    if db_path.is_empty() {
        return Vec::new();
    }
    match rusqlite::Connection::open(db_path) {
        Ok(conn) => darkroom_db::tags::tag_list_with_counts(&conn).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
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

// ── Right panel (metadata + tags) ────────────────────────────────────────

/// Metadata inspector widget with an `update` method.
///
/// All GTK fields are GObject ref-counts so `MetadataPanel` is Clone.
#[derive(Clone)]
pub struct MetadataPanel {
    pub widget:   gtk4::Box,
    filename_lbl: gtk4::Label,
    folder_lbl:   gtk4::Label,
    dims_lbl:     gtk4::Label,
    size_lbl:     gtk4::Label,
    tags_flow:    gtk4::FlowBox,
    tag_entry:    gtk4::Entry,
    /// Shared (path, db_path) for the add-tag handler
    ctx: std::rc::Rc<std::cell::RefCell<(String, String)>>,
}

impl MetadataPanel {
    pub fn new() -> Self {
        let panel = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(0)
            .width_request(210)
            .build();

        // ── Header ────────────────────────────────────────────────────────
        let header = gtk4::Label::builder()
            .label("Metadata")
            .halign(gtk4::Align::Start)
            .margin_top(12).margin_bottom(6)
            .margin_start(12).margin_end(12)
            .build();
        header.add_css_class("heading");
        panel.append(&header);
        panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

        // ── Info grid ─────────────────────────────────────────────────────
        let mk_key = |text: &str| {
            let l = gtk4::Label::builder().label(text).halign(gtk4::Align::End).build();
            l.add_css_class("dim-label");
            l
        };
        let mk_val = || gtk4::Label::builder()
            .halign(gtk4::Align::Start).hexpand(true)
            .max_width_chars(20).ellipsize(gtk4::pango::EllipsizeMode::Middle)
            .build();

        let filename_lbl = mk_val();
        let folder_lbl   = mk_val();
        let dims_lbl     = mk_val();
        let size_lbl     = mk_val();

        let grid = gtk4::Grid::builder()
            .row_spacing(4).column_spacing(8)
            .margin_start(12).margin_end(12).margin_top(10)
            .build();
        for (i, (key, val)) in [
            ("File", &filename_lbl), ("Folder", &folder_lbl),
            ("Size", &dims_lbl),    ("Disk",   &size_lbl),
        ].iter().enumerate() {
            grid.attach(&mk_key(key), 0, i as i32, 1, 1);
            grid.attach(*val, 1, i as i32, 1, 1);
        }
        panel.append(&grid);

        // ── Tags section ──────────────────────────────────────────────────
        let tags_header = gtk4::Label::builder()
            .label("Tags")
            .halign(gtk4::Align::Start)
            .margin_top(12).margin_bottom(4)
            .margin_start(12).margin_end(12)
            .build();
        tags_header.add_css_class("heading");
        panel.append(&tags_header);

        let tags_flow = gtk4::FlowBox::builder()
            .selection_mode(gtk4::SelectionMode::None)
            .homogeneous(false)
            .max_children_per_line(10)
            .margin_start(10).margin_end(10).margin_bottom(6)
            .build();
        panel.append(&tags_flow);

        // Add-tag entry
        let tag_entry = gtk4::Entry::builder()
            .placeholder_text("Add tag…")
            .margin_start(10).margin_end(10).margin_bottom(8)
            .build();
        panel.append(&tag_entry);

        // ── Placeholder ───────────────────────────────────────────────────
        let placeholder = gtk4::Label::builder()
            .label("Select an image\nto view metadata")
            .halign(gtk4::Align::Center).valign(gtk4::Align::Center)
            .vexpand(true).justify(gtk4::Justification::Center)
            .build();
        placeholder.add_css_class("dim-label");
        panel.append(&placeholder);

        let ctx = std::rc::Rc::new(std::cell::RefCell::new((String::new(), String::new())));

        // Wire add-tag on Enter key
        {
            let ctx2      = ctx.clone();
            let flow_ref  = tags_flow.clone();
            tag_entry.connect_activate(move |entry| {
                let tag_name = entry.text().trim().to_string();
                if tag_name.is_empty() { return; }
                let (ref path, ref db) = *ctx2.borrow();
                if !db.is_empty() {
                    add_tag_to_image(path, db, &tag_name);
                    rebuild_tags_flow(&flow_ref, path, db);
                }
                entry.set_text("");
            });
        }

        Self { widget: panel, filename_lbl, folder_lbl, dims_lbl, size_lbl,
               tags_flow, tag_entry, ctx }
    }

    /// Refresh the panel for the image at `full_path`.
    pub fn update(&self, full_path: &str, db_path: &str) {
        use std::path::Path;
        let p        = Path::new(full_path);
        let filename = p.file_name().and_then(|n| n.to_str()).unwrap_or(full_path);
        let folder   = p.parent().and_then(|d| d.to_str()).unwrap_or("");

        self.filename_lbl.set_label(filename);
        self.folder_lbl.set_label(folder.rsplit('/').next().unwrap_or(folder));

        let dims = query_dims(full_path, db_path)
            .map(|(w, h)| format!("{w} \u{00d7} {h}"))
            .unwrap_or_else(|| "\u{2014}".into());
        self.dims_lbl.set_label(&dims);

        let disk = std::fs::metadata(full_path)
            .map(|m| format_bytes(m.len()))
            .unwrap_or_else(|_| "\u{2014}".into());
        self.size_lbl.set_label(&disk);

        // Store context for the tag-entry handler
        *self.ctx.borrow_mut() = (full_path.to_string(), db_path.to_string());

        // Rebuild tag chips
        rebuild_tags_flow(&self.tags_flow, full_path, db_path);
        self.tag_entry.set_text("");
    }
}

impl Default for MetadataPanel {
    fn default() -> Self { Self::new() }
}

// ── Tag helpers ───────────────────────────────────────────────────────────

fn rebuild_tags_flow(flow: &gtk4::FlowBox, full_path: &str, db_path: &str) {
    // Clear existing chips
    while let Some(child) = flow.first_child() {
        flow.remove(&child);
    }

    let tags = load_tags(full_path, db_path);
    if tags.is_empty() {
        let lbl = gtk4::Label::builder().label("(none)").build();
        lbl.add_css_class("dim-label");
        flow.insert(&lbl, -1);
        return;
    }

    for tag in tags {
        let chip = gtk4::Label::builder()
            .label(&tag)
            .margin_start(4).margin_end(4)
            .margin_top(2).margin_bottom(2)
            .build();
        chip.add_css_class("tag");
        flow.insert(&chip, -1);
    }
}

fn load_tags(full_path: &str, db_path: &str) -> Vec<String> {
    if db_path.is_empty() { return Vec::new(); }
    let conn = match rusqlite::Connection::open(db_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let imgid = match darkroom_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        _ => return Vec::new(),
    };
    darkroom_db::tags::tag_get_attached(&conn, imgid)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|id| darkroom_db::tags::tag_get_name(&conn, id).ok().flatten())
        .collect()
}

fn add_tag_to_image(full_path: &str, db_path: &str, tag_name: &str) {
    let Ok(conn) = rusqlite::Connection::open(db_path) else { return };
    let Ok(Some(imgid)) = darkroom_db::image::image_get_id_by_path(&conn, full_path) else { return };
    // Create tag if it doesn't exist, then attach it
    if let Ok(Some(tag_id)) = darkroom_db::tags::tag_new(&conn, tag_name) {
        let _ = darkroom_db::tags::tag_attach(&conn, tag_id, imgid);
    }
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
