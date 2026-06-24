//! Side-panel widgets -- collections list and metadata inspector.
//!
//! Phase 3-ui-6:
//!   LeftPanel  -- film rolls with image counts; clicking filters the grid
//!   MetadataPanel -- right panel that updates when an image is selected

use adw::prelude::*;
use glib::clone;
use crate::lighttable::{
    LighttableModel, lighttable_load_by_folder, lighttable_load_by_tag, lighttable_load_from_db,
};
use darkroom_db;

// ── Left panel (collections) ──────────────────────────────────────────────

/// The collections (left) panel: film rolls plus a live Tags section.
///
/// Clicking a film roll reloads the lighttable to show only that folder; the
/// first row ("All images") clears the filter. The Tags section can be rebuilt
/// in place via [`LeftPanel::refresh_tags`] after a tag is attached elsewhere
/// (e.g. from the metadata panel), so newly-created tags and changed counts
/// appear without restarting the app.
///
/// All fields are GObject ref-counts (plus a `String`), so `LeftPanel` is Clone
/// and can be handed to the metadata panel's change callback cheaply.
#[derive(Clone)]
pub struct LeftPanel {
    pub widget:  gtk4::Box,
    /// Stable tag list box; only its rows are rebuilt on refresh so the
    /// folder↔tag selection-coordination handlers (bound once) stay valid.
    tag_box:     gtk4::ListBox,
    /// Section chrome whose visibility tracks whether any user tags exist.
    tags_header: gtk4::Label,
    tags_sep:    gtk4::Separator,
    db_path:     String,
    /// Shared "currently-filtering tag id" (None = no tag filter). Kept current
    /// by the folder/tag click handlers; read by `delete_tag` so deleting the
    /// filtered-on tag can drop the now-dangling filter and revert to All.
    active_tag:  std::rc::Rc<std::cell::Cell<Option<u32>>>,
    /// The lighttable grid model, so `delete_tag` can reload it to "All images"
    /// when the active tag filter is invalidated.
    lt_model:    LighttableModel,
    /// Optional notify fired after a library-wide tag mutation here (rename /
    /// delete), so the metadata panel can re-render the current image's chips.
    /// Mirror of `MetadataPanel::on_tags_changed`. Set via `set_on_tags_changed`.
    on_tags_changed: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>>,
}

impl LeftPanel {
    /// `active_tag` is the shared "currently-filtering tag id" (None = no tag
    /// filter). The folder/tag click handlers keep it current so a later tag
    /// mutation can decide whether to re-run the grid filter.
    pub fn new(
        db_path: &str,
        lt_model: &LighttableModel,
        active_tag: &std::rc::Rc<std::cell::Cell<Option<u32>>>,
    ) -> Self {
        let panel = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(0)
            .width_request(210)
            .build();

        // Both the Collections and Tags sections scroll together inside one
        // content box so neither steals the other's vertical space.
        let content = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(0)
            .build();

        // ── Collections (film rolls) ──────────────────────────────────────
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

        // The Tags list box is built up-front (even if empty) so the folder
        // handler can clear its selection — the two SelectionMode::Single boxes
        // are mutually exclusive, so a folder/tag filter never leaves a stale
        // highlight in the other list implying an AND that isn't running.
        // Clicking "All images" (which clears the tag highlight too) is the way
        // out of a tag filter. The box is stable across refreshes; only its rows
        // are rebuilt, so the handlers bound below never go stale.
        let tag_box = gtk4::ListBox::builder()
            .selection_mode(gtk4::SelectionMode::Single)
            .build();
        tag_box.add_css_class("navigation-sidebar");

        // Activate: reload lighttable with folder filter, dropping any tag filter.
        let db = db_path.to_string();
        let at_folder = active_tag.clone();
        list_box.connect_row_activated(clone!(@weak lt_model, @weak tag_box => move |_, row| {
            tag_box.unselect_all();
            at_folder.set(None);   // a folder/all view is not a tag filter
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

        // ── Tags ──────────────────────────────────────────────────────────
        // The header/separator/box are always present; their visibility tracks
        // whether the library has any user tags (toggled in `refresh_tags`).
        let tags_header = section_header("Tags");
        let tags_sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);
        content.append(&tags_header);
        content.append(&tags_sep);

        let db_tags = db_path.to_string();
        let at_tag = active_tag.clone();
        tag_box.connect_row_activated(clone!(@weak lt_model, @weak list_box => move |_, row| {
            list_box.unselect_all();
            // The tag id is encoded in the row's widget name (see append_tag_row).
            if let Ok(tag_id) = row.widget_name().as_str().parse::<u32>() {
                at_tag.set(Some(tag_id));   // remember the active tag filter
                lighttable_load_by_tag(&lt_model, &db_tags, tag_id);
            }
        }));
        content.append(&tag_box);

        let scroll = gtk4::ScrolledWindow::builder()
            .hscrollbar_policy(gtk4::PolicyType::Never)
            .vscrollbar_policy(gtk4::PolicyType::Automatic)
            .child(&content)
            .vexpand(true)
            .build();
        panel.append(&scroll);

        let lp = Self {
            widget: panel,
            tag_box,
            tags_header,
            tags_sep,
            db_path: db_path.to_string(),
            active_tag: active_tag.clone(),
            lt_model: lt_model.clone(),
            on_tags_changed: std::rc::Rc::new(std::cell::RefCell::new(None)),
        };
        lp.refresh_tags();
        lp
    }

    /// Register a callback fired after a tag is renamed or deleted here, so the
    /// metadata panel can re-render the current image's chips. Mirror of
    /// `MetadataPanel::set_on_tags_changed`; replaces any previous callback. The
    /// callback must not re-enter a left-panel tag mutation, or it would loop.
    pub fn set_on_tags_changed<F: Fn() + 'static>(&self, f: F) {
        *self.on_tags_changed.borrow_mut() = Some(std::rc::Rc::new(f));
    }

    /// Fire the tags-changed notify, if set (clone out of the cell first so it
    /// isn't borrowed while the callback runs).
    fn fire_tags_changed(&self) {
        let cb = self.on_tags_changed.borrow().clone();
        if let Some(cb) = cb { cb(); }
    }

    /// Rebuild the Tags section in place from the current library state.
    ///
    /// Clears and repopulates only the tag rows (the box itself is stable, so
    /// the activation handler bound in `new` keeps working), then shows or hides
    /// the section depending on whether any user tags exist. Safe to call after
    /// a tag is attached from the metadata panel to surface new tags / counts.
    pub fn refresh_tags(&self) {
        while let Some(child) = self.tag_box.first_child() {
            self.tag_box.remove(&child);
        }
        let tags = load_tags_with_counts(&self.db_path);
        for (id, name, count) in &tags {
            self.append_tag_row(*id, name, *count);
        }
        let has_tags = !tags.is_empty();
        self.tags_header.set_visible(has_tags);
        self.tags_sep.set_visible(has_tags);
        self.tag_box.set_visible(has_tags);
    }

    /// Append a tag row carrying its attached-image count; the tag id is stashed
    /// in the row's widget name so the activation handler can recover it. A
    /// secondary (right) click opens a rename/delete context popover.
    fn append_tag_row(&self, id: u32, name: &str, count: i64) {
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

        // Secondary-click → rename/delete popover. The gesture lives as long as
        // the row (so for the app lifetime while the row is in `tag_box`); to
        // avoid a strong-ref cycle (tag_box→row→gesture→LeftPanel→tag_box) it
        // captures only weak refs + the db path and reconstructs a transient
        // `LeftPanel` on demand. Removed rows then free cleanly on refresh.
        let gesture = gtk4::GestureClick::new();
        gesture.set_button(gtk4::gdk::BUTTON_SECONDARY);
        let widget_w  = self.widget.downgrade();
        let tag_box_w = self.tag_box.downgrade();
        let header_w  = self.tags_header.downgrade();
        let sep_w     = self.tags_sep.downgrade();
        let db        = self.db_path.clone();
        // The notify Rc, active-tag cell and grid model never reference back at
        // the left-panel widgets, so capturing them strongly introduces no cycle.
        let notify     = self.on_tags_changed.clone();
        let active_tag = self.active_tag.clone();
        let lt_model   = self.lt_model.clone();
        let name_owned = name.to_string();
        let row_w     = row.downgrade();
        gesture.connect_pressed(move |g, _, x, y| {
            g.set_state(gtk4::EventSequenceState::Claimed);
            if let (Some(widget), Some(tag_box), Some(tags_header), Some(tags_sep), Some(row)) = (
                widget_w.upgrade(), tag_box_w.upgrade(), header_w.upgrade(),
                sep_w.upgrade(), row_w.upgrade(),
            ) {
                let lp = LeftPanel {
                    widget, tag_box, tags_header, tags_sep, db_path: db.clone(),
                    active_tag: active_tag.clone(), lt_model: lt_model.clone(),
                    on_tags_changed: notify.clone(),
                };
                lp.show_tag_menu(&row, id, &name_owned, count, x, y);
            }
        });
        row.add_controller(gesture);

        self.tag_box.append(&row);
    }

    /// Pop up the rename/delete menu for a tag row at the click point.
    fn show_tag_menu(&self, row: &gtk4::ListBoxRow, id: u32, name: &str, count: i64, x: f64, y: f64) {
        let popover = gtk4::Popover::builder().build();
        popover.set_parent(row);
        popover.set_pointing_to(Some(&gtk4::gdk::Rectangle::new(x as i32, y as i32, 1, 1)));

        let vbox = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(6)
            .margin_start(6).margin_end(6).margin_top(6).margin_bottom(6)
            .build();

        let entry = gtk4::Entry::builder().text(name).build();
        vbox.append(&entry);

        let rename_btn = gtk4::Button::with_label("Rename");
        rename_btn.add_css_class("suggested-action");
        vbox.append(&rename_btn);

        vbox.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

        let plural = if count == 1 { "" } else { "s" };
        let delete_btn = gtk4::Button::with_label(&format!("Delete (from {count} image{plural})"));
        delete_btn.add_css_class("destructive-action");
        vbox.append(&delete_btn);

        popover.set_child(Some(&vbox));

        // Rename on button click or Enter in the entry (skipped if blank or
        // unchanged — no needless write/refresh).
        let do_rename = {
            let lp = self.clone();
            let pop = popover.clone();
            let entry = entry.clone();
            let original = name.to_string();
            move || {
                // Read the entry, then dismiss the popover BEFORE refresh_tags
                // removes its parent row — so no orphaned subtree exists mid-call.
                let new_name = entry.text().trim().to_string();
                pop.popdown();
                if !new_name.is_empty() && new_name != original {
                    lp.rename_tag(id, &new_name);
                }
            }
        };
        rename_btn.connect_clicked({
            let f = do_rename.clone();
            move |_| f()
        });
        entry.connect_activate({
            let f = do_rename.clone();
            move |_| f()
        });

        // Delete (confirmed) on button click.
        {
            let lp = self.clone();
            let pop = popover.clone();
            let name_owned = name.to_string();
            delete_btn.connect_clicked(move |_| {
                pop.popdown();
                lp.confirm_delete_tag(id, &name_owned, count);
            });
        }

        // The popover is parented to the row; unparent it when dismissed so it
        // doesn't outlive (and leak against) the row it points at.
        popover.connect_closed(|p| p.unparent());
        popover.popup();
    }

    /// Rename a tag library-wide (best-effort; a UNIQUE-name clash logs and
    /// leaves the tag unchanged), then refresh the list.
    fn rename_tag(&self, id: u32, new_name: &str) {
        if self.db_path.is_empty() { return; }
        match rusqlite::Connection::open(&self.db_path) {
            Ok(conn) => {
                if let Err(e) = darkroom_db::tags::tag_rename(&conn, id, new_name) {
                    eprintln!("darkroom: tag rename failed (duplicate name?): {e}");
                }
            }
            Err(e) => eprintln!("darkroom: cannot open library db to rename tag: {e}"),
        }
        self.refresh_tags();
        self.fire_tags_changed();
    }

    /// Confirm, then delete a tag and all its image associations.
    fn confirm_delete_tag(&self, id: u32, name: &str, count: i64) {
        let plural = if count == 1 { "" } else { "s" };
        let dialog = adw::AlertDialog::new(
            Some("Delete tag?"),
            Some(&format!(
                "\u{201c}{name}\u{201d} will be removed from {count} image{plural}. \
                 This cannot be undone."
            )),
        );
        dialog.add_responses(&[("cancel", "Cancel"), ("delete", "Delete")]);
        dialog.set_response_appearance("delete", adw::ResponseAppearance::Destructive);
        dialog.set_default_response(Some("cancel"));
        dialog.set_close_response("cancel");
        let lp = self.clone();
        dialog.connect_response(None, move |_, resp| {
            if resp == "delete" { lp.delete_tag(id); }
        });
        dialog.present(Some(&self.widget));
    }

    /// Delete a tag and its associations (best-effort; logs faults), then refresh.
    ///
    /// If the deleted tag is the one currently filtering the grid, the filter is
    /// now dangling (and its id could later be reused by SQLite) — so we drop it
    /// and revert the grid to "All images" rather than leave a stale/empty view.
    fn delete_tag(&self, id: u32) {
        if self.db_path.is_empty() { return; }
        match rusqlite::Connection::open(&self.db_path) {
            Ok(conn) => {
                if let Err(e) = darkroom_db::tags::tag_delete(&conn, id) {
                    eprintln!("darkroom: tag delete failed: {e}");
                }
            }
            Err(e) => eprintln!("darkroom: cannot open library db to delete tag: {e}"),
        }
        let was_active = self.active_tag.get() == Some(id);
        self.active_tag.set(next_active_tag(self.active_tag.get(), id));
        self.refresh_tags();
        if was_active {
            // active_tag is now None, so the on_tags_changed `reapply` below is a
            // no-op; reload the grid here to surface the full library again.
            lighttable_load_from_db(&self.lt_model, &self.db_path);
        }
        self.fire_tags_changed();
    }
}

/// Resolve the active tag filter after a tag is deleted: clears it when it
/// pointed at the just-deleted tag, otherwise leaves it untouched. Kept as a
/// free function so the (display-bound) `delete_tag` logic has a unit-testable core.
fn next_active_tag(current: Option<u32>, deleted: u32) -> Option<u32> {
    match current {
        Some(id) if id == deleted => None,
        other => other,
    }
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

fn load_tags_with_counts(db_path: &str) -> Vec<(u32, String, i64)> {
    if db_path.is_empty() {
        return Vec::new();
    }
    // Log faults so a corrupt/locked catalog reads differently from "no tags"
    // (the Tags section is simply hidden either way, but the cause is recoverable
    // from the logs — the established read-path discipline).
    match rusqlite::Connection::open(db_path) {
        Ok(conn) => match darkroom_db::tags::tag_list_with_counts(&conn) {
            Ok(tags) => tags,
            Err(e) => {
                eprintln!("darkroom: tag list query failed: {e}");
                Vec::new()
            }
        },
        Err(e) => {
            eprintln!("darkroom: cannot open library db for tags: {e}");
            Vec::new()
        }
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
    /// Optional notify fired after a tag is attached, so other panels (the
    /// left-panel Tags section) can refresh. Set via [`set_on_tags_changed`].
    on_tags_changed: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>>,
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
        let on_tags_changed: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));

        // Wire add-tag on Enter key
        {
            let ctx2      = ctx.clone();
            let flow_ref  = tags_flow.clone();
            let notify    = on_tags_changed.clone();
            tag_entry.connect_activate(move |entry| {
                let tag_name = entry.text().trim().to_string();
                if tag_name.is_empty() { return; }
                // Clone the (path, db) out so the cell isn't borrowed while
                // rebuild_tags_flow re-borrows it.
                let (path, db) = ctx2.borrow().clone();
                if !db.is_empty() {
                    add_tag_to_image(&path, &db, &tag_name);
                    rebuild_tags_flow(&flow_ref, &ctx2, &notify);
                    // Notify other panels (clone the callback out before invoking
                    // so the cell isn't borrowed while it runs).
                    let cb = notify.borrow().clone();
                    if let Some(cb) = cb { cb(); }
                }
                entry.set_text("");
            });
        }

        Self { widget: panel, filename_lbl, folder_lbl, dims_lbl, size_lbl,
               tags_flow, tag_entry, ctx, on_tags_changed }
    }

    /// Register a callback fired whenever a tag is attached from this panel.
    ///
    /// Used to keep the left-panel Tags list in sync with newly-created tags
    /// and changed image counts. Replaces any previously-set callback. This is
    /// the canonical "tags mutated" hook: future tag *detach* / *rename* paths
    /// should route through it too, so the left-panel count refresh stays
    /// single-sourced. The callback must not re-enter a metadata-panel tag
    /// mutation, or it would loop.
    pub fn set_on_tags_changed<F: Fn() + 'static>(&self, f: F) {
        *self.on_tags_changed.borrow_mut() = Some(std::rc::Rc::new(f));
    }

    /// Re-render the current image's tag chips from the DB without changing the
    /// selected image. Used as the left-panel's "tags mutated" callback so a
    /// rename/delete there updates chips shown here immediately.
    pub fn refresh_tags_display(&self) {
        // Nothing selected yet → leave the placeholder; skip a pointless rebuild.
        if self.ctx.borrow().0.is_empty() { return; }
        rebuild_tags_flow(&self.tags_flow, &self.ctx, &self.on_tags_changed);
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

        // Store context for the tag-entry / detach handlers
        *self.ctx.borrow_mut() = (full_path.to_string(), db_path.to_string());

        // Rebuild tag chips (display only — no notify on a mere selection change)
        rebuild_tags_flow(&self.tags_flow, &self.ctx, &self.on_tags_changed);
        self.tag_entry.set_text("");
    }
}

impl Default for MetadataPanel {
    fn default() -> Self { Self::new() }
}

// ── Tag helpers ───────────────────────────────────────────────────────────

/// Rebuild the per-image tag chips. Each chip carries an inline ✕ button that
/// detaches the tag and refreshes; `ctx` (live path/db) and `notify` (the
/// "tags mutated" hook) are threaded through so the detach handler can re-read
/// the current image and fan the change out to the left-panel Tags list.
///
/// Rebuilding alone never fires `notify` — only an actual detach does — so a
/// mere selection change repaints chips without spuriously refreshing siblings.
fn rebuild_tags_flow(
    flow: &gtk4::FlowBox,
    ctx: &std::rc::Rc<std::cell::RefCell<(String, String)>>,
    notify: &std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>>,
) {
    // Clear existing chips
    while let Some(child) = flow.first_child() {
        flow.remove(&child);
    }

    let (full_path, db_path) = ctx.borrow().clone();
    let tags = load_tags(&full_path, &db_path);
    if tags.is_empty() {
        let lbl = gtk4::Label::builder().label("(none)").build();
        lbl.add_css_class("dim-label");
        flow.insert(&lbl, -1);
        return;
    }

    for (tag_id, name) in tags {
        let chip = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(2)
            .margin_start(4).margin_end(4)
            .margin_top(2).margin_bottom(2)
            .build();
        chip.add_css_class("tag");

        let lbl = gtk4::Label::builder().label(&name).build();
        chip.append(&lbl);

        let close = gtk4::Button::builder()
            .icon_name("window-close-symbolic")
            .tooltip_text("Remove tag")
            .build();
        close.add_css_class("flat");
        close.add_css_class("circular");

        // Detach reads the *current* image from ctx at click time (the chip may
        // outlive a selection change in flight), removes the tag, rebuilds the
        // chips, then fires the shared notify so the left-panel counts update.
        let flow_w   = flow.downgrade();
        let ctx_c    = ctx.clone();
        let notify_c = notify.clone();
        close.connect_clicked(move |_| {
            let (p, d) = ctx_c.borrow().clone();
            if d.is_empty() { return; }
            detach_tag_from_image(&p, &d, tag_id);
            if let Some(flow) = flow_w.upgrade() {
                rebuild_tags_flow(&flow, &ctx_c, &notify_c);
            }
            let cb = notify_c.borrow().clone();
            if let Some(cb) = cb { cb(); }
        });
        chip.append(&close);

        flow.insert(&chip, -1);
    }
}

/// Tags attached to an image, as `(tag_id, name)` so chips can offer detach.
fn load_tags(full_path: &str, db_path: &str) -> Vec<(u32, String)> {
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
        .filter_map(|id| {
            darkroom_db::tags::tag_get_name(&conn, id).ok().flatten().map(|n| (id, n))
        })
        .collect()
}

/// Detach a tag from the image at `full_path` (best-effort; logs faults).
fn detach_tag_from_image(full_path: &str, db_path: &str, tag_id: u32) {
    let conn = match rusqlite::Connection::open(db_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("darkroom: cannot open library db to detach tag: {e}"); return; }
    };
    let imgid = match darkroom_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        Ok(None) => return,            // image not in catalog — nothing to detach
        Err(e) => { eprintln!("darkroom: image lookup failed on tag detach: {e}"); return; }
    };
    if let Err(e) = darkroom_db::tags::tag_detach(&conn, tag_id, imgid) {
        eprintln!("darkroom: tag detach failed: {e}");
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_active_tag_clears_when_deleted_tag_is_active() {
        assert_eq!(next_active_tag(Some(5), 5), None);
    }

    #[test]
    fn next_active_tag_keeps_unrelated_filter() {
        assert_eq!(next_active_tag(Some(3), 5), Some(3));
    }

    #[test]
    fn next_active_tag_noop_when_no_filter() {
        assert_eq!(next_active_tag(None, 5), None);
    }
}
