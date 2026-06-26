//! Side-panel widgets -- collections list and metadata inspector.
//!
//! Phase 3-ui-6:
//!   LeftPanel  -- film rolls with image counts; clicking filters the grid
//!   MetadataPanel -- right panel that updates when an image is selected

use adw::prelude::*;
use glib::clone;
use crate::lighttable::{
    LighttableModel, lighttable_load_by_folder, lighttable_load_by_tag_prefix,
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
    /// Optional notify fired after a library-wide tag mutation here (rename /
    /// delete), so the metadata panel can re-render the current image's chips.
    /// Mirror of `MetadataPanel::on_tags_changed`. Set via `set_on_tags_changed`.
    on_tags_changed: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>>,
}

impl LeftPanel {
    /// `active_tag` is the shared "currently-filtering tag prefix" (None = no tag
    /// filter). It holds a tag's full `parent|child` path; the folder/tag click
    /// handlers keep it current so a later tag mutation can re-run the same
    /// hierarchical-prefix grid filter.
    pub fn new(
        db_path: &str,
        lt_model: &LighttableModel,
        active_tag: &std::rc::Rc<std::cell::RefCell<Option<String>>>,
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
            *at_folder.borrow_mut() = None;   // a folder/all view is not a tag filter
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
            // The full `parent|child` path is encoded in the row's widget name
            // (see append_tag_tree_row) for both real and virtual nodes. Clicking
            // either filters to that tag plus its whole hierarchical subtree.
            let prefix = row.widget_name().to_string();
            if !prefix.is_empty() {
                *at_tag.borrow_mut() = Some(prefix.clone());   // remember the filter
                lighttable_load_by_tag_prefix(&lt_model, &db_tags, &prefix);
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
        let rows = flatten_tag_tree(&tags);
        for r in &rows {
            self.append_tag_tree_row(r);
        }
        let has_tags = !rows.is_empty();
        self.tags_header.set_visible(has_tags);
        self.tags_sep.set_visible(has_tags);
        self.tag_box.set_visible(has_tags);
    }

    /// Append one hierarchical tag row, indented by its depth in the `|`-tree.
    ///
    /// Every row stashes its full `parent|child` path in the widget name so the
    /// activation handler can filter to that path plus its subtree (see the
    /// `tag_box` row-activated handler in `new`). A **real** tag (`row.id` is
    /// `Some`) additionally shows its count and gets a secondary-click
    /// rename/delete popover. A **virtual** parent (a path prefix with no tag of
    /// its own) is rendered dim and stays count-less and menu-less, but is still
    /// clickable — activating it filters the grid to its whole subtree.
    fn append_tag_tree_row(&self, row_data: &TagTreeRow) {
        let row = gtk4::ListBoxRow::new();

        // Indent by depth; the base margin matches the old flat rows (12px).
        let indent = 12 + (row_data.depth as i32) * 16;
        let hbox = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(8)
            .margin_start(indent).margin_end(8)
            .margin_top(6).margin_bottom(6)
            .build();

        let name_lbl = gtk4::Label::builder()
            .label(&row_data.label)
            .halign(gtk4::Align::Start)
            .hexpand(true)
            .ellipsize(gtk4::pango::EllipsizeMode::Middle)
            .build();
        hbox.append(&name_lbl);

        // The full `parent|child` path drives prefix filtering on click (real and
        // virtual rows alike); stash it in the widget name for the activation
        // handler. flatten_tag_tree never emits an empty `full_name`, so the
        // handler's non-empty guard always admits a genuine tag row.
        row.set_widget_name(&row_data.full_name);

        let Some(id) = row_data.id else {
            // Virtual parent: dim and count-less (a path prefix with no tag of its
            // own, so nothing to rename/delete), but still clickable — activating
            // it filters the grid to the whole subtree under this prefix.
            name_lbl.add_css_class("dim-label");
            row.set_tooltip_text(Some(&row_data.full_name));
            row.set_child(Some(&hbox));
            self.tag_box.append(&row);
            return;
        };

        let count_lbl = gtk4::Label::builder()
            .label(&row_data.count.to_string())
            .halign(gtk4::Align::End)
            .build();
        count_lbl.add_css_class("dim-label");
        count_lbl.add_css_class("numeric");
        hbox.append(&count_lbl);

        row.set_tooltip_text(Some(&row_data.full_name));
        row.set_child(Some(&hbox));

        // Secondary-click → rename/delete popover. The gesture lives as long as
        // the row (so for the app lifetime while the row is in `tag_box`); to
        // avoid a strong-ref cycle (tag_box→row→gesture→LeftPanel→tag_box) it
        // captures only weak refs + the db path and reconstructs a transient
        // `LeftPanel` on demand. Removed rows then free cleanly on refresh.
        // The popover operates on the FULL tag path (rename/delete are per-tag,
        // hierarchy-unaware in this slice — segment-only rename is a later step).
        let gesture = gtk4::GestureClick::new();
        gesture.set_button(gtk4::gdk::BUTTON_SECONDARY);
        let widget_w  = self.widget.downgrade();
        let tag_box_w = self.tag_box.downgrade();
        let header_w  = self.tags_header.downgrade();
        let sep_w     = self.tags_sep.downgrade();
        let db        = self.db_path.clone();
        // The notify Rc never references back at the left-panel widgets, so
        // capturing it strongly introduces no cycle.
        let notify     = self.on_tags_changed.clone();
        let name_owned = row_data.full_name.clone();
        let count      = row_data.count;
        let row_w     = row.downgrade();
        gesture.connect_pressed(move |g, _, x, y| {
            g.set_state(gtk4::EventSequenceState::Claimed);
            if let (Some(widget), Some(tag_box), Some(tags_header), Some(tags_sep), Some(row)) = (
                widget_w.upgrade(), tag_box_w.upgrade(), header_w.upgrade(),
                sep_w.upgrade(), row_w.upgrade(),
            ) {
                let lp = LeftPanel {
                    widget, tag_box, tags_header, tags_sep, db_path: db.clone(),
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

        // Edit only this node's own segment (the last `|` component); the parent
        // prefix is fixed and the whole subtree moves with it (see do_rename).
        let segment = name.rsplit_once('|').map(|(_, s)| s).unwrap_or(name);
        let entry = gtk4::Entry::builder().text(segment).build();
        vbox.append(&entry);

        let hint = gtk4::Label::builder()
            .label("Renames this tag and any sub-tags")
            .halign(gtk4::Align::Start)
            .build();
        hint.add_css_class("dim-label");
        hint.add_css_class("caption");
        vbox.append(&hint);

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
            let old_full = name.to_string();
            move || {
                // Read the entry, then dismiss the popover BEFORE refresh_tags
                // removes its parent row — so no orphaned subtree exists mid-call.
                // respliced_tag_path re-attaches the fixed parent prefix and
                // returns None for a blank/unchanged segment (no needless write).
                let new_segment = entry.text().to_string();
                pop.popdown();
                if let Some(new_full) = respliced_tag_path(&old_full, &new_segment) {
                    lp.rename_tag_subtree(&old_full, &new_full);
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

    /// Rename a tag's segment library-wide, rewriting the whole `old_full`→
    /// `new_full` subtree so descendants move with it (best-effort; a UNIQUE-name
    /// clash with an existing destination path logs and — because the underlying
    /// UPDATE is atomic — leaves every tag unchanged), then refresh the list.
    fn rename_tag_subtree(&self, old_full: &str, new_full: &str) {
        if self.db_path.is_empty() { return; }
        match rusqlite::Connection::open(&self.db_path) {
            Ok(conn) => {
                if let Err(e) = darkroom_db::tags::tag_rename_subtree(&conn, old_full, new_full) {
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
    /// The active filter is a tag **path** (not an id), so a delete can't leave a
    /// dangling id-reuse hazard the way the old id-based filter could. We simply
    /// refresh the tree and fire the change notify: the wired `reapply` re-runs
    /// the current hierarchical-prefix filter, so deleting a parent keeps its
    /// surviving descendants on screen, while deleting the exact filtered-on leaf
    /// collapses to the empty-result placeholder (the user clicks a folder/All to
    /// leave it).
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
        self.refresh_tags();
        self.fire_tags_changed();
    }
}

/// Compute the new full tag path when a node's own segment is renamed in place.
/// `full_name` is the node's `parent|child` path; `new_segment` is the user's
/// replacement for the LAST segment (trimmed here). Returns the rewritten full
/// path, or `None` when the edit is a no-op — blank input or the segment is
/// unchanged — so the caller skips the DB write. A `new_segment` containing the
/// `|` hierarchy separator is rejected (also `None`): this popover renames a node
/// in place, so re-parenting/deepening the tree via a typed `|` is out of scope —
/// and forbidding it also rules out a rewrite that could self-collide against a
/// row it is itself moving. The parent prefix is preserved verbatim; a top-level
/// tag (no `|`) just becomes `new_segment`. Kept as a free function so the
/// (display-bound) rename popover has a unit-testable core.
fn respliced_tag_path(full_name: &str, new_segment: &str) -> Option<String> {
    let new_segment = new_segment.trim();
    if new_segment.is_empty() || new_segment.contains('|') {
        return None;
    }
    let (parent, segment) = match full_name.rsplit_once('|') {
        Some((p, s)) => (Some(p), s),
        None => (None, full_name),
    };
    if new_segment == segment {
        return None;
    }
    Some(match parent {
        Some(p) => format!("{p}|{new_segment}"),
        None => new_segment.to_string(),
    })
}

/// One row of the hierarchical tag display, in pre-order render order.
/// `depth` is the nesting level (0 = top), `label` the path segment shown, and
/// `full_name` the cumulative `a|b|c` path (used by the rename/delete popover and
/// later by prefix filtering). `id`/`count` are set only for a **real** tag; a
/// **virtual** parent (a path prefix with no tag of its own) has `id: None` and
/// `count: 0` — descendant counts are deliberately NOT summed, since an image
/// carrying two sibling tags would be double-counted.
#[derive(Debug, Clone, PartialEq)]
struct TagTreeRow {
    depth:     usize,
    label:     String,
    full_name: String,
    id:        Option<u32>,
    count:     i64,
}

/// Build the pre-order hierarchical display rows from the flat `(id, name, count)`
/// tag list. Names use `|` as the hierarchy separator (darktable convention);
/// children are alphabetised, and intermediate path segments with no tag of their
/// own become virtual parent rows so a `places|Italy` tag still shows a `places`
/// group even when `places` itself is untagged.
fn flatten_tag_tree(tags: &[(u32, String, i64)]) -> Vec<TagTreeRow> {
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Node {
        children: BTreeMap<String, Node>,
        tag:      Option<(u32, i64)>,   // (id, count) when a real tag ends here
    }

    let mut root = Node::default();
    for (id, name, count) in tags {
        // Skip empty segments so a malformed name ("a|", "a||b", "|a") collapses
        // to its meaningful ancestry rather than rendering a blank-label row —
        // which would otherwise carry a working count + rename/delete popover on
        // a tag the user can't see. A name that is all separators yields nothing.
        let mut segs = name.split('|').filter(|s| !s.is_empty()).peekable();
        if segs.peek().is_none() {
            continue;
        }
        let mut cur = &mut root;
        while let Some(seg) = segs.next() {
            cur = cur.children.entry(seg.to_string()).or_default();
            if segs.peek().is_none() {
                // Safe last-write: `data.tags.name` is UNIQUE, so no two distinct
                // ids ever resolve to the same full path (no real-tag clobber).
                cur.tag = Some((*id, *count));
            }
        }
    }

    fn dfs(node: &Node, depth: usize, prefix: &str, out: &mut Vec<TagTreeRow>) {
        for (seg, child) in &node.children {
            let full_name = if prefix.is_empty() {
                seg.clone()
            } else {
                format!("{prefix}|{seg}")
            };
            let (id, count) = match child.tag {
                Some((id, count)) => (Some(id), count),
                None => (None, 0),
            };
            out.push(TagTreeRow { depth, label: seg.clone(), full_name: full_name.clone(), id, count });
            dfs(child, depth + 1, &full_name, out);
        }
    }

    let mut out = Vec::new();
    dfs(&root, 0, "", &mut out);
    out
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
/// Best-effort: structural faults log (parity with `detach_tag_from_image`); an
/// uncatalogued image is a silent empty (nothing attached), as is a single tag
/// whose name can't be resolved (best-effort display, not worth spamming).
/// NOTE: this runs on every lighttable selection change, so the `Err` logs fire
/// per-repaint under a genuinely broken db — kept loud on purpose (the healthy
/// `Ok(None)` path is silent); if this ever ships to users, latch "db unhealthy"
/// once per session rather than making this log quieter.
fn load_tags(full_path: &str, db_path: &str) -> Vec<(u32, String)> {
    if db_path.is_empty() { return Vec::new(); }
    let conn = match rusqlite::Connection::open(db_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("darkroom: cannot open library db to load tags: {e}"); return Vec::new(); }
    };
    let imgid = match darkroom_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        Ok(None) => return Vec::new(),   // image not in catalog — no tags
        Err(e) => { eprintln!("darkroom: image lookup failed on load tags: {e}"); return Vec::new(); }
    };
    let attached = match darkroom_db::tags::tag_get_attached(&conn, imgid) {
        Ok(ids) => ids,
        Err(e) => { eprintln!("darkroom: cannot read attached tags: {e}"); return Vec::new(); }
    };
    attached
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

/// Create the tag if needed and attach it to the image at `full_path`
/// (best-effort; logs faults — parity with `detach_tag_from_image`). An
/// uncatalogued image is a silent no-op (nothing to attach to).
fn add_tag_to_image(full_path: &str, db_path: &str, tag_name: &str) {
    let conn = match rusqlite::Connection::open(db_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("darkroom: cannot open library db to add tag: {e}"); return; }
    };
    let imgid = match darkroom_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        Ok(None) => return,            // image not in catalog — nothing to attach
        Err(e) => { eprintln!("darkroom: image lookup failed on tag add: {e}"); return; }
    };
    // Create tag if it doesn't exist, then attach it.
    match darkroom_db::tags::tag_new(&conn, tag_name) {
        Ok(Some(tag_id)) => {
            if let Err(e) = darkroom_db::tags::tag_attach(&conn, tag_id, imgid) {
                eprintln!("darkroom: tag attach failed: {e}");
            }
        }
        Ok(None) => eprintln!("darkroom: could not create or find tag \u{201c}{tag_name}\u{201d}"),
        Err(e) => eprintln!("darkroom: tag create failed: {e}"),
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

    fn t(id: u32, name: &str, count: i64) -> (u32, String, i64) {
        (id, name.to_string(), count)
    }

    #[test]
    fn resplice_rewrites_last_segment_keeping_parent() {
        assert_eq!(
            respliced_tag_path("places|Italy", "Italia").as_deref(),
            Some("places|Italia"),
        );
        assert_eq!(
            respliced_tag_path("a|b|c", "z").as_deref(),
            Some("a|b|z"),
        );
    }

    #[test]
    fn resplice_top_level_tag_has_no_parent() {
        assert_eq!(respliced_tag_path("landscape", "scenery").as_deref(), Some("scenery"));
    }

    #[test]
    fn resplice_is_noop_when_blank_or_unchanged() {
        assert_eq!(respliced_tag_path("places|Italy", "Italy"), None);
        assert_eq!(respliced_tag_path("places|Italy", "   "), None);
        assert_eq!(respliced_tag_path("places|Italy", ""), None);
        // Trimmed input equal to the current segment is still a no-op.
        assert_eq!(respliced_tag_path("places|Italy", "  Italy  "), None);
    }

    #[test]
    fn resplice_trims_input() {
        assert_eq!(
            respliced_tag_path("places|Italy", "  Italia  ").as_deref(),
            Some("places|Italia"),
        );
    }

    #[test]
    fn resplice_rejects_pipe_in_segment() {
        // A typed `|` would re-parent/deepen the tree (out of scope for an
        // in-place segment rename) and could let the rewrite self-collide.
        assert_eq!(respliced_tag_path("places|Italy", "Italy|north"), None);
        assert_eq!(respliced_tag_path("places|Italy", "a|b"), None);
        assert_eq!(respliced_tag_path("landscape", "a|b"), None);
    }

    #[test]
    fn flatten_flat_tags_are_depth_zero_reals() {
        let rows = flatten_tag_tree(&[t(1, "landscape", 4), t(2, "portrait", 2)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], TagTreeRow { depth: 0, label: "landscape".into(), full_name: "landscape".into(), id: Some(1), count: 4 });
        assert_eq!(rows[1], TagTreeRow { depth: 0, label: "portrait".into(), full_name: "portrait".into(), id: Some(2), count: 2 });
    }

    #[test]
    fn flatten_synthesises_virtual_parent() {
        // "places" is not itself a tag — it must appear as a virtual group.
        let rows = flatten_tag_tree(&[t(7, "places|Italy", 3)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], TagTreeRow { depth: 0, label: "places".into(), full_name: "places".into(), id: None, count: 0 });
        assert_eq!(rows[1], TagTreeRow { depth: 1, label: "Italy".into(), full_name: "places|Italy".into(), id: Some(7), count: 3 });
    }

    #[test]
    fn flatten_real_parent_keeps_its_id_and_count() {
        // "places" is BOTH a tag and a parent.
        let rows = flatten_tag_tree(&[t(1, "places", 5), t(2, "places|Italy", 3)]);
        assert_eq!(rows[0], TagTreeRow { depth: 0, label: "places".into(), full_name: "places".into(), id: Some(1), count: 5 });
        assert_eq!(rows[1], TagTreeRow { depth: 1, label: "Italy".into(), full_name: "places|Italy".into(), id: Some(2), count: 3 });
    }

    #[test]
    fn flatten_orders_children_alphabetically_per_level() {
        // Input order is irrelevant; siblings sort, subtrees stay grouped.
        let rows = flatten_tag_tree(&[t(3, "b", 1), t(1, "a|z", 1), t(2, "a|m", 1)]);
        let shape: Vec<(usize, &str)> = rows.iter().map(|r| (r.depth, r.label.as_str())).collect();
        assert_eq!(shape, vec![(0, "a"), (1, "m"), (1, "z"), (0, "b")]);
    }

    #[test]
    fn flatten_handles_three_levels() {
        let rows = flatten_tag_tree(&[t(1, "a|b|c", 9)]);
        let shape: Vec<(usize, &str, Option<u32>)> =
            rows.iter().map(|r| (r.depth, r.label.as_str(), r.id)).collect();
        assert_eq!(shape, vec![(0, "a", None), (1, "b", None), (2, "c", Some(1))]);
    }

    #[test]
    fn flatten_collapses_trailing_separator() {
        // "a|" must render as a normal `a` tag, not `a > (blank)`.
        let rows = flatten_tag_tree(&[t(1, "a|", 2)]);
        assert_eq!(rows, vec![TagTreeRow { depth: 0, label: "a".into(), full_name: "a".into(), id: Some(1), count: 2 }]);
    }

    #[test]
    fn flatten_collapses_double_and_leading_separator() {
        let rows = flatten_tag_tree(&[t(1, "a||b", 1), t(2, "|c", 1)]);
        let shape: Vec<(usize, &str, Option<u32>)> =
            rows.iter().map(|r| (r.depth, r.label.as_str(), r.id)).collect();
        assert_eq!(shape, vec![(0, "a", None), (1, "b", Some(1)), (0, "c", Some(2))]);
    }

    #[test]
    fn flatten_drops_all_separator_name() {
        // A name with no representable segment contributes nothing.
        assert!(flatten_tag_tree(&[t(1, "|", 1), t(2, "", 1)]).is_empty());
    }
}
