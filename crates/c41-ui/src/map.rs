//! Map view: slippy-map canvas over the geotagged list (u5; parity audit 3.4,
//! map leg — u3 shipped the list-only first slice).
//!
//! The page shows a pannable, zoomable raster map ([`build_map_canvas`]) above
//! the full list of every image carrying a lat+lon fix, each row showing a
//! thumbnail plus filename plus coordinates, with row activation opening the
//! image in the darkroom. Tiles come from the OpenStreetMap tile server over
//! plain HTTPS through [`crate::tiles`] — integer zoom only (wheel steps ±1,
//! clamped `0..=19`), drag-pan by centre lat/lon, memory + disk cached, with
//! the required "© OpenStreetMap contributors" attribution label under the
//! canvas. There is no WebKit and no system location library.
//!
//! The file follows the established pure-model-then-widget discipline (see
//! [`crate::print`]): the row text ([`geo_row_text`], [`map_row_title`],
//! [`build_map_rows`]) and the marker model ([`build_map_markers`]) are
//! GTK-free and unit-tested headless; the tile maths lives in
//! [`crate::tiles`] under the same discipline. The page widget over both is
//! just the GTK control surface.
//!
//! Thumbnails reuse the lighttable thumbnail service
//! ([`crate::lighttable::thumbs`]): the same cache lookup, in-flight dedupe,
//! decode gate, and packed-RGB8 paint path the grid cells use, at a fixed
//! small bucket. Row activation calls the SAME open function the grid's
//! `activate` signal uses — [`map_page`] takes it as a parameter and the
//! caller in `lib.rs` hands it the hoisted `open_in_darkroom` closure —
//! rather than reimplementing the darkroom push. Marker clicks call it too,
//! with the clicked marker's full catalogue path.
//!
//! Refresh: the DAO query runs at page build, and every `win.open-map`
//! activation builds a fresh page, so leaving and re-entering the map
//! refreshes the list. A query failure (no catalogue open, or a catalog that
//! predates the geo migration) renders as the empty state, never as an error
//! page — the page always opens.

use adw::prelude::*;
use gtk4::{gdk, gio, glib};
use std::cell::Cell;
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

/// One canvas marker: the full catalogue path (for the click-to-open) plus
/// the fix the canvas projects. Same order as the DAO result, so marker `i`
/// and row `i` are the same image.
#[derive(Debug, Clone, PartialEq)]
pub struct MapMarker {
    pub path: String,
    pub latitude: f64,
    pub longitude: f64,
}

/// Build the marker model for one DAO result, preserving its order like
/// [`build_map_rows`]. An empty input yields no markers and the canvas opens
/// on the default world view.
pub fn build_map_markers(items: &[GeoImage]) -> Vec<MapMarker> {
    items
        .iter()
        .map(|item| MapMarker {
            path: item.path.clone(),
            latitude: item.latitude,
            longitude: item.longitude,
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

// ── Slippy-map canvas (u5; geometry in `crate::tiles`) ─────────────────────

/// Canvas height in pixels. Wide and short: a map strip above the list, not a
/// full takeover — the list below stays the page's dense index.
const MAP_CANVAS_H: i32 = 280;

/// Click radius around a marker in pixels. Markers draw at radius 6; the hit
/// area is deliberately a little larger so touchpad clicks land.
const MARKER_HIT_PX: f64 = 10.0;

/// Nominal viewport the initial fit is computed against. The real allocation
/// is not known at build time; the fit only sets the opening view (the user
/// pans and zooms from there), so a typical window-centre size is close
/// enough — and it keeps the initial view a pure function of the fixes.
const MAP_FIT_W: f64 = 800.0;
const MAP_FIT_H: f64 = 360.0;

/// Fetch one tile off the GTK loop: at most one fetch per tile ever runs
/// (the shared in-flight registry refuses duplicates from sibling frames),
/// the network leg counts against the tile gate (claimed BEFORE spawning, so
/// no worker parks waiting for a slot), and completion stores into the memory
/// cache plus repaints. Returns true when the gate was busy: the caller owes
/// one timed retry redraw, so the tile converges on a later frame. A busy
/// gate is NOT a failure — nothing is negative-cached for it. Genuine misses
/// negative-cache session-only on completion; a panicked worker logs and
/// stays retryable.
fn spawn_tile_fetch(area_w: glib::WeakRef<gtk4::DrawingArea>, z: i32, x: i32, y: i32) -> bool {
    use crate::tiles;
    if !tiles::tile_inflight_register(z, x, y) {
        return false;
    }
    let Some(permit) = tiles::TileNetPermit::try_acquire() else {
        tiles::tile_inflight_unregister(z, x, y);
        return true;
    };
    glib::spawn_future_local(async move {
        let fetched =
            gio::spawn_blocking(move || tiles::fetch_tile_blocking(permit, z, x, y)).await;
        tiles::tile_inflight_unregister(z, x, y);
        match fetched {
            Ok(Some(t)) => {
                tiles::tile_store(z, x, y, tiles::tile_pixbuf(&t));
                if let Some(a) = area_w.upgrade() {
                    a.queue_draw();
                }
            }
            // Genuine miss (HTTP error, timeout, corrupt body): negative-cache
            // session-only HERE on the main thread — the fetch itself runs on a
            // worker, where the session sets do not live. Next launch retries.
            Ok(None) => {
                tiles::tile_mark_failed(z, x, y);
            }
            Err(_) => {
                eprintln!("c41-map: tile fetch task panicked for {z}/{x}/{y}");
            }
        }
    });
    false
}

/// Build the slippy-map canvas for `markers`. `open` is the same darkroom
/// opener the list rows use: a marker click within [`MARKER_HIT_PX`] opens
/// that image. Returns the canvas plus the attribution label stacked in one
/// vertical box, ready to sit above the list.
///
/// Interactions mirror the zoomable canvas's idioms where they apply: wheel
/// steps the integer zoom ±1 (DISCRETE, so touchpad flicks do not zoom; the
/// wheel belongs to the zoomer entirely), primary-button drag pans by centre
/// lat/lon, single click on a marker opens it. There is no cursor-anchored
/// zoom — at integer tile zooms the centre-pinned step is the honest mapping.
fn build_map_canvas(markers: Vec<MapMarker>, open: Rc<dyn Fn(String)>) -> gtk4::Box {
    use crate::tiles;
    tiles::prune_tiles_at_startup();

    let pts: Vec<(f64, f64)> = markers.iter().map(|m| (m.latitude, m.longitude)).collect();
    let (init_lat, init_lon, init_zoom) = tiles::fit_view(&pts, MAP_FIT_W, MAP_FIT_H);
    let center: Rc<Cell<(f64, f64)>> = Rc::new(Cell::new((init_lat, init_lon)));
    let zoom: Rc<Cell<i32>> = Rc::new(Cell::new(init_zoom));
    let markers: Rc<Vec<MapMarker>> = Rc::new(markers);

    let area = gtk4::DrawingArea::new();
    area.set_hexpand(true);
    area.set_size_request(-1, MAP_CANVAS_H);

    // Paint: background, cached tiles (placeholder + async fetch for misses),
    // then markers. Missing spawns are collected and fired after the paint so
    // the draw closure itself never awaits anything.
    {
        let center = center.clone();
        let zoom = zoom.clone();
        let markers = markers.clone();
        area.set_draw_func(move |area, cr, w, h| {
            let (clat, clon) = center.get();
            let z = zoom.get();
            let vp_w = f64::from(w);
            let vp_h = f64::from(h);
            cr.set_source_rgb(0.10, 0.10, 0.11);
            let _ = cr.paint();
            let mut missing: Vec<(i32, i32)> = Vec::new();
            if vp_w > 0.0 && vp_h > 0.0 {
                let ((x0, x1), (y0, y1)) = tiles::visible_tile_range(clat, clon, z, vp_w, vp_h);
                for ty in y0..=y1 {
                    for tx in x0..=x1 {
                        let (sx, sy) = tiles::tile_screen_origin(tx, ty, clat, clon, z, vp_w, vp_h);
                        match tiles::tile_lookup(z, tx, ty) {
                            Some(pix) => {
                                cr.set_source_pixbuf(&pix, sx, sy);
                                let _ = cr.paint();
                            }
                            None => {
                                cr.set_source_rgb(0.16, 0.16, 0.17);
                                cr.rectangle(
                                    sx,
                                    sy,
                                    f64::from(tiles::TILE_PX),
                                    f64::from(tiles::TILE_PX),
                                );
                                let _ = cr.fill();
                                missing.push((tx, ty));
                            }
                        }
                    }
                }
            }
            for m in markers.iter() {
                let (mx, my) =
                    tiles::geo_to_screen(m.latitude, m.longitude, clat, clon, z, vp_w, vp_h);
                if mx < -12.0 || my < -12.0 || mx > vp_w + 12.0 || my > vp_h + 12.0 {
                    continue;
                }
                cr.set_source_rgb(0.85, 0.2, 0.2);
                cr.arc(mx, my, 6.0, 0.0, std::f64::consts::TAU);
                let _ = cr.fill_preserve();
                cr.set_source_rgb(1.0, 1.0, 1.0);
                cr.set_line_width(1.5);
                let _ = cr.stroke();
            }
            let area_w = area.downgrade();
            // One retry flag for the whole frame (not one timer per tile): a
            // busy gate repaints once shortly, and the next frame re-attempts
            // every still-missing tile. Tiles the negative cache owns are left
            // as placeholders until the next launch.
            let mut retry = false;
            for (tx, ty) in missing {
                if tiles::tile_is_failed(z, tx, ty) {
                    continue;
                }
                retry |= spawn_tile_fetch(area_w.clone(), z, tx, ty);
            }
            if retry {
                let retry_w = area.downgrade();
                glib::timeout_add_local_once(std::time::Duration::from_millis(300), move || {
                    if let Some(a) = retry_w.upgrade() {
                        a.queue_draw();
                    }
                });
            }
        });
    }

    // Wheel: one integer step per notch, pinned at the ends. DISCRETE keeps
    // touchpad flicks out (the zoomable lesson); Propagation::Stop starves any
    // parent scroller — while over the canvas the wheel belongs to the zoomer.
    {
        let zoom = zoom.clone();
        let area_w = area.downgrade();
        let wheel = gtk4::EventControllerScroll::new(
            gtk4::EventControllerScrollFlags::VERTICAL | gtk4::EventControllerScrollFlags::DISCRETE,
        );
        wheel.connect_scroll(move |_, _dx, dy| {
            let z = zoom.get();
            let nz = tiles::clamp_zoom(z + if dy < 0.0 { 1 } else { -1 }) as i32;
            if nz != z {
                zoom.set(nz);
                if let Some(a) = area_w.upgrade() {
                    a.queue_draw();
                }
            }
            glib::Propagation::Stop
        });
        area.add_controller(wheel);
    }

    // Drag pans by centre lat/lon through world pixels (the exact plane both
    // screen projections offset into): the content follows the pointer, and a
    // drag past either pole pins at the limit instead of producing NaN.
    {
        let center = center.clone();
        let zoom = zoom.clone();
        let area_w = area.downgrade();
        let drag = gtk4::GestureDrag::new();
        let begin_world: Rc<Cell<Option<(f64, f64)>>> = Rc::new(Cell::new(None));
        {
            let center = center.clone();
            let zoom = zoom.clone();
            let begin_world = begin_world.clone();
            drag.connect_drag_begin(move |_, _, _| {
                let (clat, clon) = center.get();
                begin_world.set(Some(tiles::geo_to_world(clat, clon, zoom.get())));
            });
        }
        {
            let zoom = zoom.clone();
            drag.connect_drag_update(move |_, dx, dy| {
                if let Some((wx, wy)) = begin_world.get() {
                    let (lat, lon) = tiles::world_to_geo(wx - dx, wy - dy, zoom.get());
                    center.set((lat, lon));
                    if let Some(a) = area_w.upgrade() {
                        a.queue_draw();
                    }
                }
            });
        }
        area.add_controller(drag);
    }

    // Click: a marker within the hit radius opens that image through the same
    // `open` the list rows use. Drags never reach here — the drag gesture
    // claims the sequence past its threshold, the click only fires on a tap.
    {
        let center = center.clone();
        let zoom = zoom.clone();
        let click = gtk4::GestureClick::new();
        click.set_button(gdk::BUTTON_PRIMARY);
        click.connect_released(move |gesture, n_clicks, x, y| {
            if n_clicks != 1 {
                return;
            }
            let Some(area) = gesture.widget().and_downcast::<gtk4::DrawingArea>() else {
                return;
            };
            let (clat, clon) = center.get();
            let z = zoom.get();
            let vp_w = f64::from(area.width());
            let vp_h = f64::from(area.height());
            let pts: Vec<(f64, f64)> = markers
                .iter()
                .map(|m| tiles::geo_to_screen(m.latitude, m.longitude, clat, clon, z, vp_w, vp_h))
                .collect();
            if let Some(idx) = tiles::hit_marker(&pts, x, y, MARKER_HIT_PX) {
                open(markers[idx].path.clone());
            }
        });
        area.add_controller(click);
    }

    // Attribution is a policy requirement (OSM tile usage policy): always
    // visible while the canvas is on screen, which covers every frame tiles
    // render in.
    let attribution = gtk4::Label::new(Some(tiles::OSM_ATTRIBUTION));
    attribution.set_halign(gtk4::Align::End);
    attribution.add_css_class("dim-label");
    attribution.add_css_class("caption");

    let wrap = gtk4::Box::new(gtk4::Orientation::Vertical, 2);
    wrap.append(&area);
    wrap.append(&attribution);
    wrap
}

// ── Page widget ─────────────────────────────────────────────────────────────

/// Build the map list page. `open` is the grid's darkroom opener (hoisted
/// `open_in_darkroom` in `lib.rs`): row activation calls it with the row's
/// full catalogue path, so the map and the grid open the editor by the same
/// code path — including closing the full preview and tagging the pushed page
/// with the image path for the pop re-sync. Marker clicks call the same
/// function with the clicked marker's path.
pub fn map_page(db_path: String, open: Rc<dyn Fn(String)>) -> adw::NavigationPage {
    let geo = load_geotagged(&db_path);
    let rows = build_map_rows(&geo);
    let markers = build_map_markers(&geo);

    let content = gtk4::Box::new(gtk4::Orientation::Vertical, 12);
    content.set_margin_start(12);
    content.set_margin_end(12);
    content.set_margin_top(12);
    content.set_margin_bottom(12);

    // Header line: what the canvas does and how a row opens.
    let note = gtk4::Label::new(Some(
        "Drag to pan, scroll to zoom. Activating a list row or clicking a map \
         marker opens it in the darkroom.",
    ));
    note.set_wrap(true);
    note.set_halign(gtk4::Align::Start);
    note.add_css_class("dim-label");
    content.append(&note);

    // The slippy-map canvas with its attribution label, above the list.
    content.append(&build_map_canvas(markers, open.clone()));

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
    fn build_markers_mirrors_rows_item_for_item() {
        let items = vec![
            geo_item("/b/second.dng", 1.0, 2.0),
            geo_item("/a/first.dng", -3.0, -4.0),
        ];
        let markers = build_map_markers(&items);
        assert_eq!(markers.len(), 2);
        assert_eq!(
            markers[0],
            MapMarker {
                path: "/b/second.dng".to_string(),
                latitude: 1.0,
                longitude: 2.0,
            }
        );
        // Marker i and row i are the same image: same path, same order.
        let rows = build_map_rows(&items);
        for (m, r) in markers.iter().zip(rows.iter()) {
            assert_eq!(m.path, r.path);
        }
        assert!(build_map_markers(&[]).is_empty(), "no fixes, no markers");
    }

    #[test]
    fn map_tag_is_slash_free() {
        // The popped cell re-sync and the switcher mirror only handle slash
        // tags (image paths); an auxiliary tag containing `/` would be
        // mistaken for a darkroom page.
        assert!(!MAP_PAGE_TAG.contains('/'), "tag {MAP_PAGE_TAG:?}");
    }
}
