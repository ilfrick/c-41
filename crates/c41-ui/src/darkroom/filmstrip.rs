//! Bottom filmstrip (w5) — the darkroom's horizontal band of the current
//! image's film-roll thumbnails, darktable's `src/libs/tools/filmstrip.c`.
//!
//! The strip is deliberately thin: it reuses the lighttable's thumbnail service
//! and paint path ([`crate::lighttable::ensure_grid_thumb`] /
//! [`crate::lighttable::apply_selection_frame`]), so there is no second image
//! decoder and a thumbnail the grid already decoded is a cache hit here (same
//! `(path, bucket)` key). Decoding stays off the UI thread exactly as in the
//! grid. Clicking a thumbnail hands its path to the caller-supplied `on_open`
//! callback; the darkroom module itself knows nothing about navigation, so the
//! same strip works wherever the page is hosted.
//!
//! Cells are virtualised: the roll backs a [`gtk4::ListView`] over a
//! [`gtk4::StringList`], so only the cells in view are realized and only those
//! start a decode chain. Building one button per image instead would arm one
//! 150 ms retry timer per cell behind `thumbs::MAX_CONCURRENT_DECODES = 2`
//! (~3.3k wake-ups/s on a 500-image roll) and realize 500 widgets on the UI
//! thread — the same hazard `CULL_MAX_IMAGES` bounds in the grid.
//!
//! Scope: the filmstrip is the current image's *film roll* (its folder), not
//! the lighttable's possibly-filtered collection — matching darktable's
//! per-roll strip. Ordering is by file name, which within one roll (constant
//! folder) is exactly the lighttable's default `SortOrder::Filename` order; the
//! alternate `DateTaken`/`Rating` sorts are intentionally not mirrored.

use gtk4::prelude::*;
use std::rc::Rc;

/// Strip height in pixels (spec: ~90–110). Sized to clear the cell's own
/// natural height — 82 px thumbnail + 8 px row margins + the button's padding
/// and border — plus a horizontal scrollbar when the roll does not fit, so
/// nothing is clipped. `height_request` is a floor, so over-reserving only
/// costs a few pixels of preview.
const STRIP_HEIGHT: i32 = 120;
/// Square thumbnail box inside a strip cell. Note this is *display* size only:
/// [`crate::lighttable::ensure_grid_thumb`] buckets its decode on the grid's
/// own cell geometry (160 px), so a strip thumb is decoded at ~4x the pixels it
/// shows. That is the deliberate cache-sharing tradeoff above — one decode
/// serves both surfaces — and it means the strip is coupled to the grid's
/// thumbnail bucket, not to its layout.
const THUMB_BOX: i32 = 82;

/// Sort a film roll's sibling paths into the strip's display order: by file
/// name ascending, with the full path as a stable tie-break. Pure.
pub(crate) fn filmstrip_order(mut paths: Vec<String>) -> Vec<String> {
    paths.sort_by(|a, b| {
        let an = std::path::Path::new(a).file_name();
        let bn = std::path::Path::new(b).file_name();
        an.cmp(&bn).then_with(|| a.cmp(b))
    });
    paths
}

/// Index of `current` in the display-ordered `paths`, or `None` when the current
/// image is not in its film roll (an uncatalogued image). Pure.
pub(crate) fn filmstrip_current_index(paths: &[String], current: &str) -> Option<usize> {
    paths.iter().position(|p| p == current)
}

/// Whether a roll of `n` images is worth a strip. A lone thumbnail is noise,
/// and as a button it would rebuild the same page for no visible effect — so a
/// 1-image (or empty) roll hides the strip entirely. Pure.
pub(crate) fn filmstrip_visible(n: usize) -> bool {
    n > 1
}

/// Display label for a cell: the file name, not the full path.
fn file_label(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string()
}

/// Every catalogue image in the same film roll as `file_path`, in catalogue
/// filename order. Empty when there is no catalogue, the image is
/// uncatalogued, or its folder has no film-roll row — callers then render no
/// strip.
pub(crate) fn film_roll_paths(file_path: &str, db_path: &str) -> Vec<String> {
    if db_path.is_empty() {
        return Vec::new();
    }
    let Some(folder) = std::path::Path::new(file_path)
        .parent()
        .and_then(|p| p.to_str())
    else {
        return Vec::new();
    };
    let Ok(conn) = crate::persist::open_catalog(db_path) else {
        return Vec::new();
    };
    let Ok(Some(film_id)) = c41_db::film::film_get_id(&conn, folder) else {
        return Vec::new();
    };
    c41_db::film::film_image_paths(&conn, film_id).unwrap_or_default()
}

/// Build the bottom strip for `paths` (display-ordered), marking `current` and
/// calling `on_open(path)` on click. `None` when the roll is too small to be
/// worth showing (see [`filmstrip_visible`]).
pub(crate) fn filmstrip(
    paths: Vec<String>,
    current: &str,
    on_open: Rc<dyn Fn(String)>,
) -> Option<gtk4::ScrolledWindow> {
    if !filmstrip_visible(paths.len()) {
        return None;
    }
    let current_index = filmstrip_current_index(&paths, current);
    // Resolved once: the bind closure only needs "is this cell the current
    // image", and the index is spent on the initial scroll below.
    let current_path = current_index.map(|i| paths[i].clone());
    let model = gtk4::StringList::new(
        &paths
            .iter()
            .map(|p| p.as_str())
            .collect::<Vec<_>>(),
    );
    let factory = gtk4::SignalListItemFactory::new();

    // ── Setup: one reusable cell — a picture-only button ────────────────────
    let on_open_setup = on_open.clone();
    factory.connect_setup(move |_, list_item| {
        let item = list_item.downcast_ref::<gtk4::ListItem>().unwrap();
        let picture = gtk4::Picture::builder()
            .content_fit(gtk4::ContentFit::Contain)
            .width_request(THUMB_BOX)
            .height_request(THUMB_BOX)
            .build();
        let btn = gtk4::Button::builder()
            .child(&picture)
            .has_frame(false)
            .build();
        // The click reads the picture's stamped widget name rather than a
        // captured path: cells are recycled, so a closure captured at setup
        // time would act on whatever the cell showed first (same protocol as
        // the grid's selection gesture).
        let open = on_open_setup.clone();
        btn.connect_clicked(move |b| {
            let Some(picture) = b.child().and_downcast::<gtk4::Picture>() else {
                return;
            };
            let path = picture.widget_name().to_string();
            if path.is_empty() {
                return; // never bound, or bound to a placeholder
            }
            open(path);
        });
        item.set_child(Some(&btn));
    });

    // ── Bind: stamp the identity, mark the current frame, start the load ────
    factory.connect_bind(move |_, list_item| {
        let item = list_item.downcast_ref::<gtk4::ListItem>().unwrap();
        let Some(btn) = item.child().and_downcast::<gtk4::Button>() else {
            return;
        };
        let Some(picture) = btn.child().and_downcast::<gtk4::Picture>() else {
            return;
        };
        let Some(string_obj) = item.item().and_downcast::<gtk4::StringObject>() else {
            return;
        };
        let path = string_obj.string().to_string();
        // Size and paintable are reset per bind: a recycled cell keeps the
        // requests and texture of the path it showed before. Stamping the name
        // unconditionally is what makes `ensure_grid_thumb`'s own re-check
        // authoritative — an in-flight decode for the previous path bails.
        picture.set_widget_name(&path);
        picture.set_paintable(gtk4::gdk::Paintable::NONE);
        let label = file_label(&path);
        btn.set_tooltip_text(Some(&label));
        // A picture-only button has no accessible name of its own; the file
        // name is the meaningful one, same as the tooltip.
        btn.update_property(&[gtk4::accessible::Property::Label(label.as_str())]);
        crate::lighttable::apply_selection_frame(&picture, current_path.as_deref() == Some(&path));
        // Cache hit paints synchronously; a miss decodes on a worker thread and
        // paints on completion — never blocking the UI thread. Only realized
        // cells ever get here, which is what bounds the decode fan-out.
        crate::lighttable::ensure_grid_thumb(picture.downgrade(), path);
    });

    // `NoSelection` (not `SingleSelection`): the strip marks the current frame
    // itself, so a selection model would only add GTK's own "selected"
    // styling on top and move the frame out from under the user. `FOCUS` alone
    // is what centres the current image.
    let selection = gtk4::NoSelection::new(Some(model));
    let list = gtk4::ListView::builder()
        .model(&selection)
        .factory(&factory)
        .build();
    list.set_orientation(gtk4::Orientation::Horizontal);

    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Automatic)
        .vscrollbar_policy(gtk4::PolicyType::Never)
        .height_request(STRIP_HEIGHT)
        .hexpand(true)
        .child(&list)
        .build();
    if let Some(i) = current_index {
        scroll_to_once(&list, i as u32);
    }
    Some(scroll)
}

/// Bring item `pos` into view the first time the strip is shown. Issued from
/// `connect_map` rather than at build time: a `ListView` has no allocation
/// before its first map, so an earlier `scroll_to` would be a no-op. Once-only,
/// so a later re-map (returning from the lighttable) does not yank the strip
/// back to the frame the user had scrolled to.
fn scroll_to_once(list: &gtk4::ListView, pos: u32) {
    let done = Rc::new(std::cell::Cell::new(false));
    list.connect_map(move |l| {
        if done.replace(true) {
            return;
        }
        // `ListScrollFlags::FOCUS` scrolls and focuses without selecting, which
        // is all a `NoSelection` view can act on.
        l.scroll_to(pos, gtk4::ListScrollFlags::FOCUS, Some(gtk4::ScrollInfo::new()));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn order_sorts_by_file_name_then_path() {
        let ordered = filmstrip_order(vec![p("/r/c.jpg"), p("/r/a.jpg"), p("/r/b.jpg")]);
        assert_eq!(ordered, vec![p("/r/a.jpg"), p("/r/b.jpg"), p("/r/c.jpg")]);
    }

    #[test]
    fn order_is_stable_across_same_name_in_different_folders() {
        // A film roll shares a folder, so this is defensive; the full path is the
        // tie-break and must be deterministic.
        let ordered = filmstrip_order(vec![p("/z/same.jpg"), p("/a/same.jpg")]);
        assert_eq!(ordered, vec![p("/a/same.jpg"), p("/z/same.jpg")]);
        // Idempotent: re-ordering an ordered list changes nothing.
        assert_eq!(filmstrip_order(ordered.clone()), ordered);
    }

    #[test]
    fn current_index_locates_the_image_or_reports_absent() {
        let paths = vec![p("/r/a.jpg"), p("/r/b.jpg"), p("/r/c.jpg")];
        assert_eq!(filmstrip_current_index(&paths, "/r/b.jpg"), Some(1));
        assert_eq!(filmstrip_current_index(&paths, "/r/first.jpg"), None);
        assert_eq!(filmstrip_current_index(&[], "/r/a.jpg"), None);
    }

    #[test]
    fn visibility_hides_empty_and_single_image_rolls() {
        assert!(!filmstrip_visible(0));
        assert!(
            !filmstrip_visible(1),
            "a lone thumbnail is noise + a no-op button"
        );
        assert!(filmstrip_visible(2));
        assert!(filmstrip_visible(50));
    }

    #[test]
    fn film_roll_paths_is_empty_without_a_catalogue() {
        assert!(film_roll_paths("/photos/x/a.jpg", "").is_empty());
    }
}
