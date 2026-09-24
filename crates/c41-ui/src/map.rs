//! Map view, list-only first slice (u3; parity audit 3.4, map leg).
//!
//! Scope is deliberately narrow: a list of every image carrying a full
//! lat+lon fix, each row showing a thumbnail plus filename plus coordinates,
//! with row activation opening the image in the darkroom. There are no
//! slippy-map tiles, no network calls, no WebKit, and no new dependencies —
//! the header note on the page says exactly that, so the placeholder reads as
//! honest rather than as a disabled dead end. The list IS the feature.
//!
//! The file follows the established pure-model-then-widget discipline (see
//! [`crate::print`]): the row text ([`geo_row_text`], [`map_row_title`],
//! [`build_map_rows`]) is GTK-free and unit-tested headless; the page widget
//! over it is just the GTK control surface.
//!
//! Thumbnails reuse the lighttable thumbnail service
//! ([`crate::lighttable::thumbs`]): the same cache lookup, in-flight dedupe,
//! decode gate, and packed-RGB8 paint path the grid cells use, at a fixed
//! small bucket. Row activation calls the SAME open function the grid's
//! `activate` signal uses — [`map_page`] takes it as a parameter and the
//! caller in `lib.rs` hands it the hoisted `open_in_darkroom` closure —
//! rather than reimplementing the darkroom push.
//!
//! Refresh: the DAO query runs at page build, and every `win.open-map`
//! activation builds a fresh page, so leaving and re-entering the map
//! refreshes the list. A query failure (no catalogue open, or a catalog that
//! predates the geo migration) renders as the empty state, never as an error
//! page — the page always opens.

use adw::prelude::*;
use gtk4::{gio, glib};
use std::rc::Rc;

use c41_db::image::GeoImage;

/// Navigation-page tag for the map list. Deliberately slash-free, so the
/// lighttable's `popped` cell re-sync (which only handles tags containing
/// `/`) and the view-switcher mirror (which lights Darkroom only for slash
/// tags) both ignore it — the same contract as the print page's tag.
pub const MAP_PAGE_TAG: &str = "map";

/// Display size of a map row's thumbnail in pixels. The decode bucket is the
/// shared quantisation of this size, so entries cross-hit with any other
/// surface that decoded the same file at the same bucket.
const MAP_THUMB_PX: i32 = 96;

// ── Pure row model (GTK-free, headless-tested) ───────────────────────────────

/// One map-list row's text: the full catalogue path (kept for activation),
/// the bare filename for the title line, and the coordinate subtitle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapRow {
    pub path: String,
    pub filename: String,
    pub coord_text: String,
}

/// The file's bare name for the row title, falling back to the full path when
/// it has no file-name component. Mirrors the print composer's display name.
pub fn map_row_title(path: &str) -> &str {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
}

/// One fix as `lat {..}, lon {..}`. Rendering reuses the Geotagging panel's
/// trimmed-decimal rule, so the map list and the panel never disagree about
/// how a fix reads — including negative fixes, which keep their signs.
pub fn geo_row_text(latitude: f64, longitude: f64) -> String {
    format!(
        "lat {}, lon {}",
        crate::panels::format_geo_num(Some(latitude)),
        crate::panels::format_geo_num(Some(longitude)),
    )
}

/// Build the row model for one DAO result, preserving its order (the DAO
/// owns the filename ordering; this mapping never re-sorts). An empty input
/// yields an empty vec, which is exactly the page's empty-state condition.
pub fn build_map_rows(items: &[GeoImage]) -> Vec<MapRow> {
    items
        .iter()
        .map(|item| MapRow {
            path: item.path.clone(),
            filename: map_row_title(&item.path).to_string(),
            coord_text: geo_row_text(item.latitude, item.longitude),
        })
        .collect()
}

// ── Catalogue read (house guard shape, after the tag loader in panels) ──────

/// Read the geotagged list, or empty when there is nothing to show. Empty db
/// path, an unopenable catalogue, and a query failure all converge here: the
/// page shows its empty state rather than refusing to open.
fn load_geotagged(db_path: &str) -> Vec<GeoImage> {
    if db_path.is_empty() {
        return Vec::new();
    }
    // Session-only open: this is a read on every map entry, so skip the
    // durable-schema DDL (bootstrapped once at startup).
    let conn = match c41_db::schema::open_catalog_session(db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("c41-map: cannot open library db: {e}");
            return Vec::new();
        }
    };
    match c41_db::image::image_list_geotagged(&conn) {
        Ok(items) => items,
        Err(e) => {
            eprintln!("c41-map: cannot list geotagged images: {e}");
            Vec::new()
        }
    }
}

// ── Thumbnails (the grid's paint path, shared for stable rows) ──────────


/// Make sure `thumb` (a map row's Picture, stamped with its path in the
/// widget name) shows `path`. Same chain as the grid's `ensure_grid_thumb`:
/// shared pixel-cache hit paints immediately, known-failed paths stay blank
/// until the next collection load, and anything else goes through the shared
/// in-flight registry plus the decode gate with a timed retry instead of any
/// wait. Rows never recycle, but the name is still re-checked on completion
/// so a late decode can never paint the wrong row.
fn ensure_map_thumb(thumb_w: glib::WeakRef<gtk4::Picture>, path: String) {
    use crate::lighttable::thumbs;
    let Some(thumb) = thumb_w.upgrade() else { return };
    if thumb.widget_name() != path {
        return;
    }
    let bucket = thumbs::bucket_for(MAP_THUMB_PX);
    if let Some(img) = thumbs::lookup(&path, bucket) {
        crate::lighttable::paint_thumb(&thumb, &img);
        return;
    }
    if thumbs::is_failed(&path) {
        return;
    }
    if !thumbs::inflight_register(&path, bucket) {
        glib::timeout_add_local_once(
            std::time::Duration::from_millis(150),
            move || ensure_map_thumb(thumb_w, path),
        );
        return;
    }
    let Some(permit) = thumbs::DecodePermit::try_acquire() else {
        thumbs::inflight_unregister(&path, bucket);
        glib::timeout_add_local_once(
            std::time::Duration::from_millis(150),
            move || ensure_map_thumb(thumb_w, path),
        );
        return;
    };
    glib::spawn_future_local(async move {
        let p = path.clone();
        let decoded =
            gio::spawn_blocking(move || thumbs::decode_with_permit(permit, &p, bucket)).await;
        thumbs::inflight_unregister(&path, bucket);
        match decoded {
            Ok(Some(img)) => {
                let cached = thumbs::store(&path, bucket, img);
                if let Some(t) = thumb_w.upgrade() {
                    if t.widget_name() == path {
                        crate::lighttable::paint_thumb(&t, &cached);
                    }
                }
            }
            Ok(None) => thumbs::mark_failed(&path),
            Err(_) => {
                eprintln!("c41-map: thumbnail decode task panicked for {path}");
            }
        }
    });
}

// ── Page widget ─────────────────────────────────────────────────────────────

/// Build the map list page. `open` is the grid's darkroom opener (hoisted
/// `open_in_darkroom` in `lib.rs`): row activation calls it with the row's
/// full catalogue path, so the map and the grid open the editor by the same
/// code path — including closing the full preview and tagging the pushed page
/// with the image path for the pop re-sync.
pub fn map_page(db_path: String, open: Rc<dyn Fn(String)>) -> adw::NavigationPage {
    let rows = build_map_rows(&load_geotagged(&db_path));

    let content = gtk4::Box::new(gtk4::Orientation::Vertical, 12);
    content.set_margin_start(12);
    content.set_margin_end(12);
    content.set_margin_top(12);
    content.set_margin_bottom(12);

    // Header line: honest placeholder text — the list below IS the feature,
    // tiles are what is missing.
    let note = gtk4::Label::new(Some(
        "Map tiles are not rendered yet — every geotagged image is listed below. \
         Activating a row opens it in the darkroom.",
    ));
    note.set_wrap(true);
    note.set_halign(gtk4::Align::Start);
    note.add_css_class("dim-label");
    content.append(&note);

    // The list (house ListBox idiom, as in the panel sections).
    let list = gtk4::ListBox::new();
    list.set_selection_mode(gtk4::SelectionMode::Single);
    for row in &rows {
        let item = gtk4::ListBoxRow::new();
        // The full path drives activation (see the handler below); stash it in
        // the widget name, the same carrier the tag-tree rows use.
        item.set_widget_name(&row.path);
        let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 12);
        hbox.set_margin_start(8);
        hbox.set_margin_end(8);
        hbox.set_margin_top(6);
        hbox.set_margin_bottom(6);
        let thumb = gtk4::Picture::new();
        thumb.set_width_request(MAP_THUMB_PX);
        thumb.set_height_request(MAP_THUMB_PX);
        thumb.set_widget_name(&row.path);
        hbox.append(&thumb);
        let texts = gtk4::Box::new(gtk4::Orientation::Vertical, 2);
        texts.set_valign(gtk4::Align::Center);
        texts.set_hexpand(true);
        let title = gtk4::Label::new(Some(&row.filename));
        title.set_halign(gtk4::Align::Start);
        title.set_ellipsize(gtk4::pango::EllipsizeMode::Middle);
        texts.append(&title);
        let coords = gtk4::Label::new(Some(&row.coord_text));
        coords.set_halign(gtk4::Align::Start);
        coords.add_css_class("dim-label");
        texts.append(&coords);
        hbox.append(&texts);
        item.set_child(Some(&hbox));
        list.append(&item);
        ensure_map_thumb(thumb.downgrade(), row.path.clone());
    }
    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .child(&list)
        .vexpand(true)
        .hexpand(true)
        .build();
    scroll.set_visible(!rows.is_empty());
    content.append(&scroll);

    // Empty state: the page still opens with no geotagged images (or no
    // catalogue at all) and says where fixes come from.
    let empty = gtk4::Box::new(gtk4::Orientation::Vertical, 6);
    empty.set_halign(gtk4::Align::Center);
    empty.set_valign(gtk4::Align::Center);
    empty.set_vexpand(true);
    let empty_title = gtk4::Label::new(Some("No geotagged images"));
    empty.append(&empty_title);
    let empty_hint = gtk4::Label::new(Some(
        "Add GPS coordinates in the Geotagging section of the right panel.",
    ));
    empty_hint.set_wrap(true);
    empty_hint.add_css_class("dim-label");
    empty.append(&empty_hint);
    empty.set_visible(rows.is_empty());
    content.append(&empty);

    list.connect_row_activated(move |_, row| {
        let path = row.widget_name().to_string();
        if !path.is_empty() {
            open(path);
        }
    });

    adw::NavigationPage::builder()
        .title("Map")
        .tag(MAP_PAGE_TAG)
        .child(&content)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo_item(path: &str, lat: f64, lon: f64) -> GeoImage {
        GeoImage { id: 1, path: path.to_string(), latitude: lat, longitude: lon, altitude: None }
    }

    #[test]
    fn row_text_names_both_axes_as_trimmed_decimals() {
        assert_eq!(geo_row_text(48.8581, 2.3525), "lat 48.8581, lon 2.3525");
        assert_eq!(geo_row_text(0.0, -0.5), "lat 0, lon -0.5");
        assert_eq!(geo_row_text(1.5, 100.0), "lat 1.5, lon 100");
    }

    #[test]
    fn row_text_keeps_negative_signs() {
        // Southern/western-hemisphere fixes must never lose their sign: a
        // dropped minus would plot the image on the wrong side of the planet.
        assert_eq!(geo_row_text(-33.85, 151.2), "lat -33.85, lon 151.2");
        assert_eq!(geo_row_text(40.71, -74.0), "lat 40.71, lon -74");
    }

    #[test]
    fn row_title_uses_the_bare_filename() {
        assert_eq!(map_row_title("/photos/test/IMG_0001.dng"), "IMG_0001.dng");
        assert_eq!(map_row_title("IMG_0001.dng"), "IMG_0001.dng");
        // No file-name component: show the path rather than nothing.
        assert_eq!(map_row_title(""), "");
        assert_eq!(map_row_title("/"), "/");
    }

    #[test]
    fn build_rows_maps_every_item_preserving_order() {
        let items = vec![
            geo_item("/b/second.dng", 1.0, 2.0),
            geo_item("/a/first.dng", -3.0, -4.0),
        ];
        let rows = build_map_rows(&items);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            MapRow {
                path: "/b/second.dng".to_string(),
                filename: "second.dng".to_string(),
                coord_text: "lat 1, lon 2".to_string(),
            }
        );
        assert_eq!(rows[1].filename, "first.dng");
        assert_eq!(rows[1].coord_text, "lat -3, lon -4");
    }

    #[test]
    fn build_rows_empty_is_the_empty_state_condition() {
        // The page shows its empty state exactly when the row model is empty —
        // no catalogue, no fixes, and query failure all converge here.
        assert!(build_map_rows(&[]).is_empty());
    }

    #[test]
    fn map_tag_is_slash_free() {
        // The popped cell re-sync and the switcher mirror only handle slash
        // tags (image paths); an auxiliary tag containing `/` would be
        // mistaken for a darkroom page.
        assert!(!MAP_PAGE_TAG.contains('/'), "tag {MAP_PAGE_TAG:?}");
    }
}
