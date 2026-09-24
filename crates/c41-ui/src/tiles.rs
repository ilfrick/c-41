//! Slippy-map tiles for the Map page (u5; parity audit 3.4, map leg).
//!
//! The map list (u3) proved which images carry a full lat+lon fix; this module
//! puts them on a pannable, zoomable raster map. Tiles come from the
//! OpenStreetMap tile server over plain HTTPS — no WebKit, no system
//! location library, and exactly one new dependency ([`ureq`], pure-Rust
//! rustls TLS, used only inside `gio::spawn_blocking`).
//!
//! The file follows the established pure-model-then-widget discipline (see
//! [`crate::map`]): every geometric rule below — Web-Mercator tile maths,
//! cache keys, fetch URLs, screen projections, initial fit, marker hit-testing
//! — is GTK-free and unit-tested headless. The canvas in [`crate::map`] is
//! just the GTK control surface: it calls these functions and paints pixbufs.
//!
//! Tile policy, all in one place so the audit can check it:
//!
//! * Source is `https://tile.openstreetmap.org/{z}/{x}/{y}.png`, the only host
//!   ever built ([`tile_url`]). Coordinates are normalised first
//!   ([`normalize_tile`]), so no invalid URL can ever be constructed.
//! * Integer zoom only, clamped to `0..=19` ([`ZOOM_MIN`], [`ZOOM_MAX`]); the
//!   OSM operations team asks that clients cap zoom at 19, and the canvas
//!   wheel-steps ±1 through [`clamp_zoom`].
//! * Every request carries `User-Agent: c41-darkroom/<crate version>`
//!   ([`tile_user_agent`]), per the OSM tile usage policy — a stock library
//!   default would be impolite and indistinguishable from a scraper.
//! * At most [`MAX_CONCURRENT_TILE_FETCH`] network fetches run at once
//!   ([`TileNetPermit`], the same claim-before-spawn gate shape as the
//!   thumbnail service's `DecodePermit`); disk reads never take a slot.
//! * Memory holds at most [`TILE_MEM_CAP`] decoded tiles (count-based LRU);
//!   disk persists raw PNG bytes under `$XDG_CACHE_HOME/c41/tiles` inside a
//!   [`TILE_DISK_BUDGET_BYTES`] budget, pruned once per process at first map
//!   open via the shared never-empty keep policy
//!   ([`crate::lighttable::thumbs::evict_keep_set`]) — the same keep/evict
//!   concepts as the thumbnail disk store.
//! * Failures are negative-cached session-only ([`tile_mark_failed`]): a tile
//!   that 404s or times out stays a placeholder until the next launch, when a
//!   transient outage gets its retry. Nothing about tile failure is persisted.
//! * Attribution is a policy requirement, not a nicety: the map page shows a
//!   visible [`OSM_ATTRIBUTION`] label whenever tiles can render.

use gtk4::gdk_pixbuf::{Colorspace, Pixbuf};
use gtk4::glib;
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

// ── Policy constants ─────────────────────────────────────────────────────────

/// Tile edge in pixels. OSM serves 256px PNGs and the canvas paints them 1:1
/// at every integer zoom — no resampling, no fractional scales.
pub const TILE_PX: u32 = 256;

/// Lowest/highest integer zoom the canvas can reach. 19 is the cap the OSM
/// tile usage policy asks clients to respect; 0 is the whole world on one tile.
pub const ZOOM_MIN: u32 = 0;
pub const ZOOM_MAX: u32 = 19;

/// Web-Mercator latitude limit in degrees. Past this the projection runs to
/// infinity, so every latitude input is clamped here before any maths.
pub const MAX_LAT: f64 = 85.05112878;

/// How many decoded tiles the session memory cache holds. One 256² RGB tile
/// is ~192 KB, so 256 entries cost ~50 MB in the common case (the fixed OSM
/// host serves 256² tiles). The 1024px decode cap bounds the theoretical
/// worst case at ~768 MB against a hostile source; unreachable in practice,
/// and eviction still keeps the entry count flat no matter how far the user
/// pans.
pub const TILE_MEM_CAP: usize = 256;

/// Byte budget for the on-disk tile store. Tiles are regenerable from the
/// network, so the budget is modest next to the thumbnail store's: enough for
/// a well-explored city at mid zooms, pruned back at startup when exceeded.
pub const TILE_DISK_BUDGET_BYTES: u64 = 128 * 1024 * 1024;

/// At most this many tile HTTP fetches run concurrently (see the module doc).
/// Disk hits never take a slot — only bytes off the network do.
pub const MAX_CONCURRENT_TILE_FETCH: usize = 4;

/// Per-request network timeout in seconds. OSM tiles are tens of KB; anything
/// slower than this is treated as a failure and negative-cached.
pub const TILE_FETCH_TIMEOUT_SECS: u64 = 15;

/// Largest tile body ever accepted, in bytes. A real 256px PNG is tens of KB;
/// anything past this is not a tile (a captive portal, an error page) and is
/// dropped rather than decoded or cached.
pub const TILE_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Attribution text the map page must show whenever tiles render (OSM tile
/// usage policy: credit "© OpenStreetMap contributors").
pub const OSM_ATTRIBUTION: &str = "© OpenStreetMap contributors";

/// Disk extension for cached tiles (raw server PNG bytes, stored verbatim).
const TILE_DISK_EXT: &str = "png";

// ── Pure Web-Mercator maths (GTK-free, headless-tested) ─────────────────────

/// Clamp an integer zoom into the legal `[ZOOM_MIN, ZOOM_MAX]` range. Every
/// public entry point funnels through here, so out-of-range zooms can never
/// reach URL building, world-size maths, or the canvas.
pub fn clamp_zoom(z: i32) -> u32 {
    z.clamp(ZOOM_MIN as i32, ZOOM_MAX as i32) as u32
}

/// Clamp a latitude into `[-MAX_LAT, MAX_LAT]`. Non-finite inputs (a corrupt
/// catalogue row that slipped past the DAO range filter) map to the equator
/// rather than poisoning the projection with NaN.
pub fn clamp_lat(lat: f64) -> f64 {
    if !lat.is_finite() {
        return 0.0;
    }
    lat.clamp(-MAX_LAT, MAX_LAT)
}

/// Wrap a longitude into `[-180, 180)`. Non-finite inputs map to 0 for the
/// same reason as [`clamp_lat`].
pub fn wrap_lon(lon: f64) -> f64 {
    if !lon.is_finite() {
        return 0.0;
    }
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

/// Number of tiles per axis at `zoom` (already clamped): `2^zoom`.
fn tiles_per_axis(zoom: u32) -> f64 {
    2u32.pow(zoom.min(ZOOM_MAX)) as f64
}

/// Fractional tile coordinates of one fix: `x` in `[0, 2^z)`, `y` likewise.
/// The standard slippy-map formulae; the integer tile is the floor of each.
pub fn lon_lat_to_tile_f(lon: f64, lat: f64, zoom: i32) -> (f64, f64) {
    let n = tiles_per_axis(clamp_zoom(zoom));
    let lon = wrap_lon(lon);
    let lat = clamp_lat(lat).to_radians();
    let x = (lon + 180.0) / 360.0 * n;
    let y = (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / std::f64::consts::PI) / 2.0 * n;
    (x, y)
}

/// Normalise a tile reference: zoom clamped, `x` wrapped around the
/// antimeridian into `[0, 2^z)`, `y` clamped into `[0, 2^z - 1]` (there is no
/// tile past either pole). The single funnel every cache key, disk path, and
/// URL is built through — callers can pass viewport-derived indices without
/// range-checking first.
pub fn normalize_tile(z: i32, x: i32, y: i32) -> (u32, u32, u32) {
    let z = clamp_zoom(z);
    let n = 2u32.pow(z) as i32;
    (z, x.rem_euclid(n) as u32, y.clamp(0, n - 1) as u32)
}

/// The memory-cache / in-flight / negative-cache key for one tile: the
/// normalised triple, so an `x` of `-1` and an `x` of `2^z - 1` address the
/// same entry instead of fetching twice.
pub fn tile_cache_key(z: i32, x: i32, y: i32) -> (u32, u32, u32) {
    normalize_tile(z, x, y)
}

/// The tile URL. Normalises first, so no invalid URL is ever built: `x` wraps
/// at the antimeridian, `y` clamps at the poles, `z` clamps to `0..=19`.
pub fn tile_url(z: i32, x: i32, y: i32) -> String {
    let (z, x, y) = normalize_tile(z, x, y);
    format!("https://tile.openstreetmap.org/{z}/{x}/{y}.png")
}

/// Latitude of tile row `y`'s top edge at a world of `n` tiles across: the
/// inverse Gudermannian of the Mercator fraction.
fn tile_y_to_lat(y: f64, n: f64) -> f64 {
    let t = std::f64::consts::PI * (1.0 - 2.0 * y / n);
    t.sinh().atan().to_degrees()
}

/// Geographic bounds of one tile as `(west, south, east, north)` in degrees.
/// Normalises first, mirroring [`tile_url`].
pub fn tile_xy_bounds(z: i32, x: i32, y: i32) -> (f64, f64, f64, f64) {
    let (z, x, y) = normalize_tile(z, x, y);
    let n = 2u32.pow(z) as f64;
    let west = f64::from(x) / n * 360.0 - 180.0;
    let east = f64::from(x + 1) / n * 360.0 - 180.0;
    let north = tile_y_to_lat(f64::from(y), n);
    let south = tile_y_to_lat(f64::from(y + 1), n);
    (west, south, east, north)
}

/// Mercator `y` fraction (`0` at the north limit, `1` at the south) of a
/// clamped latitude. The shared core of [`geo_to_world`] and [`fit_view`].
fn mercator_y_frac(lat: f64) -> f64 {
    let rad = clamp_lat(lat).to_radians();
    (1.0 - (rad.tan() + 1.0 / rad.cos()).ln() / std::f64::consts::PI) / 2.0
}

/// World size in pixels at `zoom`: one 256px tile per tile per axis.
fn world_size_px(zoom: u32) -> f64 {
    f64::from(TILE_PX) * 2u32.pow(zoom.min(ZOOM_MAX)) as f64
}

/// World-pixel coordinates of one fix at `zoom`. The continuous plane both
/// screen projections are offsets into.
pub fn geo_to_world(lat: f64, lon: f64, zoom: i32) -> (f64, f64) {
    let world = world_size_px(clamp_zoom(zoom));
    let wx = (wrap_lon(lon) + 180.0) / 360.0 * world;
    let wy = mercator_y_frac(lat) * world;
    (wx, wy)
}

/// Inverse of [`geo_to_world`]: world pixels back to `(lat, lon)`. The `y`
/// input clamps to the world (a drag past either pole pins at the limit
/// instead of producing NaN through `asinh`), and longitude wraps.
pub fn world_to_geo(wx: f64, wy: f64, zoom: i32) -> (f64, f64) {
    let world = world_size_px(clamp_zoom(zoom));
    let lon = wrap_lon(wx / world * 360.0 - 180.0);
    let t = std::f64::consts::PI * (1.0 - 2.0 * wy.clamp(0.0, world) / world);
    (t.sinh().atan().to_degrees(), lon)
}

/// Screen pixels of one fix for a viewport of `vp_w × vp_h` centred on
/// `(center_lat, center_lon)` at `zoom`. The `x` delta takes the short way
/// around the planet (wrapped into half a world), so a marker just across
/// the antimeridian from the centre draws beside it rather than a world away.
pub fn geo_to_screen(
    lat: f64,
    lon: f64,
    center_lat: f64,
    center_lon: f64,
    zoom: i32,
    vp_w: f64,
    vp_h: f64,
) -> (f64, f64) {
    let z = clamp_zoom(zoom);
    let world = world_size_px(z);
    let (wx, wy) = geo_to_world(lat, lon, z as i32);
    let (cwx, cwy) = geo_to_world(center_lat, center_lon, z as i32);
    let mut dx = wx - cwx;
    dx -= (dx / world).round() * world;
    (dx + vp_w / 2.0, (wy - cwy) + vp_h / 2.0)
}

/// Inverse of [`geo_to_screen`] within half a world of the centre (the only
/// range a viewport can show): screen pixels back to `(lat, lon)`. The canvas
/// drag-pan goes through [`geo_to_world`]/[`world_to_geo`] instead, which are
/// exact everywhere.
pub fn screen_to_geo(
    px: f64,
    py: f64,
    center_lat: f64,
    center_lon: f64,
    zoom: i32,
    vp_w: f64,
    vp_h: f64,
) -> (f64, f64) {
    let z = clamp_zoom(zoom);
    let (cwx, cwy) = geo_to_world(center_lat, center_lon, z as i32);
    world_to_geo(cwx + px - vp_w / 2.0, cwy + py - vp_h / 2.0, z as i32)
}

/// Screen origin (top-left) of the tile at raw indices `(x_raw, y)`. `x_raw`
/// is deliberately UNwrapped — the visible range runs past `2^z - 1` near the
/// antimeridian so panning stays continuous, and each tile is fetched by its
/// normalised form while being painted at its continuous position. `y` comes
/// from [`visible_tile_range`], already clamped.
pub fn tile_screen_origin(
    x_raw: i32,
    y: i32,
    center_lat: f64,
    center_lon: f64,
    zoom: i32,
    vp_w: f64,
    vp_h: f64,
) -> (f64, f64) {
    let z = clamp_zoom(zoom);
    let (cwx, cwy) = geo_to_world(center_lat, center_lon, z as i32);
    (
        f64::from(x_raw) * f64::from(TILE_PX) - cwx + vp_w / 2.0,
        f64::from(y) * f64::from(TILE_PX) - cwy + vp_h / 2.0,
    )
}

/// Tile index ranges covering a `vp_w × vp_h` viewport centred on
/// `(center_lat, center_lon)` at `zoom`, as `((x_min, x_max), (y_min, y_max))`.
/// `x` is raw (possibly negative or past `2^z - 1`; paint wraps per tile via
/// [`normalize_tile`]), `y` is clamped into `[0, 2^z - 1]`. Degenerate
/// viewports answer the single centre tile rather than an empty range.
pub fn visible_tile_range(
    center_lat: f64,
    center_lon: f64,
    zoom: i32,
    vp_w: f64,
    vp_h: f64,
) -> ((i32, i32), (i32, i32)) {
    let z = clamp_zoom(zoom);
    let n = 2u32.pow(z) as i32;
    let (cwx, cwy) = geo_to_world(center_lat, center_lon, z as i32);
    // NaN dimensions take the degenerate branch too (`<=` alone is false
    // for NaN), spelled explicitly per clippy::neg_cmp_op_on_partial_ord.
    if vp_w.is_nan() || vp_h.is_nan() || vp_w <= 0.0 || vp_h <= 0.0 {
        let cx = (cwx / f64::from(TILE_PX)).floor() as i32;
        let cy = ((cwy / f64::from(TILE_PX)).floor() as i32).clamp(0, n - 1);
        return ((cx, cx), (cy, cy));
    }
    let tile = f64::from(TILE_PX);
    let x0 = ((cwx - vp_w / 2.0) / tile).floor() as i32;
    // The viewport covers pixels `[left, left + vp)`: the last covered pixel is
    // one short of the right edge, so an edge landing exactly on a tile
    // boundary does not pull in a zero-width sliver of the next tile.
    let x1 = ((cwx + vp_w / 2.0 - 1.0) / tile).floor() as i32;
    let y0 = (((cwy - vp_h / 2.0) / tile).floor() as i32).clamp(0, n - 1);
    let y1 = (((cwy + vp_h / 2.0 - 1.0) / tile).floor() as i32).clamp(0, n - 1);
    ((x0, x0.max(x1)), (y0, y1.max(y0)))
}

/// Initial `(center_lat, center_lon, zoom)` for a set of `(lat, lon)` fixes on
/// a `vp_w × vp_h` viewport: the bounding-box centre at the highest integer
/// zoom that still fits the box (with a marker margin kept clear), or
/// `(0, 0)@2` when there is nothing to fit. A single fix (or duplicates)
/// centres exactly on it at zoom 12, street-level-ish rather than a whole
/// continent for one photo. The longitude box is naive (no antimeridian
/// split): a set straddling the date line fits conservatively, showing ocean
/// — correct, merely wide.
pub fn fit_view(points: &[(f64, f64)], vp_w: f64, vp_h: f64) -> (f64, f64, i32) {
    if points.is_empty() || vp_w.is_nan() || vp_h.is_nan() || vp_w <= 0.0 || vp_h <= 0.0 {
        return (0.0, 0.0, 2);
    }
    let mut lat_min = f64::INFINITY;
    let mut lat_max = f64::NEG_INFINITY;
    let mut lon_min = f64::INFINITY;
    let mut lon_max = f64::NEG_INFINITY;
    for (lat, lon) in points {
        if !lat.is_finite() || !lon.is_finite() {
            continue;
        }
        let lat = clamp_lat(*lat);
        let lon = wrap_lon(*lon);
        lat_min = lat_min.min(lat);
        lat_max = lat_max.max(lat);
        lon_min = lon_min.min(lon);
        lon_max = lon_max.max(lon);
    }
    if lat_min == f64::INFINITY {
        return (0.0, 0.0, 2);
    }
    let center_lat = (lat_min + lat_max) / 2.0;
    let center_lon = wrap_lon((lon_min + lon_max) / 2.0);
    let lon_span = (lon_max - lon_min).max(0.0);
    let lat_span_frac = (mercator_y_frac(lat_max) - mercator_y_frac(lat_min)).abs();
    if lon_span <= f64::EPSILON && lat_span_frac <= 1e-12 {
        return (center_lat, center_lon, 12);
    }
    // Viewport room minus a margin so edge markers are not half-clipped; a
    // viewport smaller than the margin still fits against at least 1px.
    let avail_w = (vp_w - 48.0).max(1.0);
    let avail_h = (vp_h - 48.0).max(1.0);
    let dx = (lon_span / 360.0 * f64::from(TILE_PX)).max(f64::MIN_POSITIVE);
    let dy = (lat_span_frac * f64::from(TILE_PX)).max(f64::MIN_POSITIVE);
    let z = (avail_w / dx).min(avail_h / dy).log2().floor() as i32;
    (center_lat, center_lon, clamp_zoom(z) as i32)
}

/// Which marker (screen positions in `markers`) sits under a click at
/// `(x, y)`: the nearest one within `radius_px`, or `None`. A non-positive
/// radius never hits — the canvas passes its marker constant.
pub fn hit_marker(markers: &[(f64, f64)], x: f64, y: f64, radius_px: f64) -> Option<usize> {
    // NaN radius (or coordinates) never hit; spelled explicitly per
    // clippy::neg_cmp_op_on_partial_ord.
    if radius_px.is_nan() || radius_px <= 0.0 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for (i, (mx, my)) in markers.iter().enumerate() {
        let d = (mx - x).hypot(my - y);
        if d <= radius_px && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

// ── Fetch identity ───────────────────────────────────────────────────────────

/// One memory-cache entry: tile key plus its paintable.
type TileMemCacheEntry = ((u32, u32, u32), Pixbuf);
/// The session memory cache container (factored out for clippy::type_complexity).
type TileMemCache = VecDeque<TileMemCacheEntry>;

/// The `User-Agent` every tile request carries (OSM tile usage policy: identify
/// the application; a stock library default would read as a scraper).
pub fn tile_user_agent() -> String {
    format!("c41-darkroom/{}", env!("CARGO_PKG_VERSION"))
}

// ── Session memory cache (count-based LRU, main-thread only) ────────────────

thread_local! {
    /// Decoded tiles, most-recently-used first. Main-thread only: paints,
    /// lookups and fetch completions all run on the GTK loop.
    static TILE_CACHE: RefCell<TileMemCache> =
        const { RefCell::new(VecDeque::new()) };
    /// Tiles that failed to fetch this session (HTTP error, timeout, corrupt
    /// body). Session-only by design: the next launch retries, so a transient
    /// outage heals itself.
    static TILE_FAILED: RefCell<HashSet<(u32, u32, u32)>> = RefCell::new(HashSet::new());
    /// Tiles with a fetch currently running. Same dedupe shape as the
    /// thumbnail service's in-flight map: one map shared by every frame, so a
    /// repaint can never stack a second fetch for a tile already in flight.
    static TILE_INFLIGHT: RefCell<HashSet<(u32, u32, u32)>> = RefCell::new(HashSet::new());
}

/// The cached pixbuf for one tile, if present. Touches the entry to MRU.
pub fn tile_lookup(z: i32, x: i32, y: i32) -> Option<Pixbuf> {
    let key = tile_cache_key(z, x, y);
    TILE_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let pos = cache.iter().position(|(k, _)| *k == key)?;
        let (_, pix) = cache.remove(pos).unwrap();
        let out = pix.clone();
        cache.push_front((key, pix));
        Some(out)
    })
}

/// Insert (or replace) one decoded tile, evicting least-recently-used entries
/// past [`TILE_MEM_CAP`]. Replacement keeps a single entry per key.
pub fn tile_store(z: i32, x: i32, y: i32, pix: Pixbuf) {
    let key = tile_cache_key(z, x, y);
    TILE_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        cache.retain(|(k, _)| *k != key);
        cache.push_front((key, pix));
        while cache.len() > TILE_MEM_CAP {
            cache.pop_back();
        }
    });
}

/// True if this tile already failed to fetch since launch.
pub fn tile_is_failed(z: i32, x: i32, y: i32) -> bool {
    let key = tile_cache_key(z, x, y);
    TILE_FAILED.with(|f| f.borrow().contains(&key))
}

/// Record a fetch failure — session-only (see the `TILE_FAILED` doc). The
/// caller decides when a miss really is one: a busy net gate is not.
pub fn tile_mark_failed(z: i32, x: i32, y: i32) {
    TILE_FAILED.with(|f| f.borrow_mut().insert(tile_cache_key(z, x, y)));
}

/// Register the fetch about to spawn. Returns `false` — having registered
/// nothing — when this tile is already in flight, meaning the caller must not
/// spawn. Returns `true` having registered, which obliges the caller to
/// [`tile_inflight_unregister`] on every exit path.
pub fn tile_inflight_register(z: i32, x: i32, y: i32) -> bool {
    let key = tile_cache_key(z, x, y);
    TILE_INFLIGHT.with(|m| {
        if m.borrow().contains(&key) {
            return false;
        }
        m.borrow_mut().insert(key);
        true
    })
}

/// Release an in-flight registration. Unconditional (unlike the thumbnail
/// service's owner-checked release): one tile has one owner — a superseding
/// larger request cannot exist for an exact tile the way it can for a
/// decode bucket.
pub fn tile_inflight_unregister(z: i32, x: i32, y: i32) {
    TILE_INFLIGHT.with(|m| {
        m.borrow_mut().remove(&tile_cache_key(z, x, y));
    });
}

// ── Network gate ─────────────────────────────────────────────────────────────

/// Live network-fetch count. `Mutex::new` is const, so this needs no lazy init.
static ACTIVE_TILE_FETCH: Mutex<usize> = Mutex::new(0);

/// RAII slot in the tile network gate. Held across one whole fetch; released
/// on every return path including panics. Never blocks: slots are claimed by
/// the caller before spawning anything (the thumbnail gate's discipline), so
/// no worker thread ever parks waiting for one.
#[must_use = "dropping the permit immediately releases the slot"]
pub struct TileNetPermit;

impl TileNetPermit {
    /// Claim a free slot, or `None` when [`MAX_CONCURRENT_TILE_FETCH`] fetches
    /// are live. A refusal is not a failure — the canvas retries on a later
    /// frame — so the caller must NOT negative-cache it.
    pub fn try_acquire() -> Option<Self> {
        let mut n = ACTIVE_TILE_FETCH.lock().unwrap();
        if *n >= MAX_CONCURRENT_TILE_FETCH {
            return None;
        }
        *n += 1;
        Some(Self)
    }
}

impl Drop for TileNetPermit {
    fn drop(&mut self) {
        *ACTIVE_TILE_FETCH.lock().unwrap() -= 1;
    }
}

// ── Disk store (raw PNG bytes, budget-pruned at startup) ────────────────────

#[cfg(test)]
thread_local! {
    /// Test seam: when set, [`tile_disk_dir`] returns this instead of the
    /// user-wide directory, so tests never read or pollute a real cache.
    static TEST_TILE_DIR: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Where cached tile PNGs persist. `C41_TILE_CACHE_DIR` overrides wholesale
/// (container experiments), otherwise the XDG cache dir — the same
/// `~/.cache/c41/...` convention as the thumbnail disk store.
fn tile_disk_dir() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(d) = TEST_TILE_DIR.with(|t| t.borrow().clone()) {
        return Some(d);
    }
    if let Some(d) = std::env::var_os("C41_TILE_CACHE_DIR").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("c41").join("tiles"))
}

/// Full path of one tile's disk entry: `{dir}/{z}/{x}/{y}.png`. Normalised,
/// so the path can never escape the store (no `..`, no negative components).
pub fn tile_disk_path(z: i32, x: i32, y: i32) -> Option<PathBuf> {
    let dir = tile_disk_dir()?;
    let (z, x, y) = normalize_tile(z, x, y);
    Some(
        dir.join(z.to_string())
            .join(x.to_string())
            .join(format!("{y}.{TILE_DISK_EXT}")),
    )
}

/// Read one tile's raw PNG bytes off disk. Any IO problem is just a miss.
fn tile_disk_load(z: i32, x: i32, y: i32) -> Option<Vec<u8>> {
    let path = tile_disk_path(z, x, y)?;
    std::fs::read(&path).ok()
}

/// Uniqueness for tmp files during atomic writes: pid separates concurrent
/// processes, the counter separates threads within one (same shape as the
/// thumbnail store's writer).
static TILE_WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Persist one tile's raw bytes atomically — write a tmp file in the SAME
/// directory, then rename over any existing entry, so readers never observe a
/// half-written file. Every failure path is silently skipped: a full disk
/// degrades to "no caching", never breaks the map.
fn tile_disk_store(z: i32, x: i32, y: i32, bytes: &[u8]) {
    let Some(path) = tile_disk_path(z, x, y) else {
        return;
    };
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let seq = TILE_WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(
        ".tmp-{}-{seq:x}.{TILE_DISK_EXT}",
        std::process::id()
    ));
    if std::fs::write(&tmp, bytes).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Delete oldest-mtime entries until the store fits `budget`, via the shared
/// never-empty keep policy ([`crate::lighttable::thumbs::evict_keep_set`],
/// newest survive, regenerable tail goes) — the thumbnail store's keep/evict
/// concepts applied to the `{z}/{x}/{y}.png` tree. Entries whose metadata
/// cannot even be read are deleted outright rather than exempted from every
/// future sweep. Pure selection math is the shared policy itself, pinned by
/// its own tests in the thumbnail service.
fn prune_tiles_dir(dir: &Path, budget: u64) {
    let mut unstatable: Vec<PathBuf> = Vec::new();
    let mut entries: Vec<(PathBuf, u64, SystemTime)> = Vec::new();
    for entry in walkdir::WalkDir::new(dir)
        .min_depth(1)
        .into_iter()
        .flatten()
    {
        let p = entry.path().to_path_buf();
        if p.extension().and_then(|x| x.to_str()) != Some(TILE_DISK_EXT) {
            continue;
        }
        match std::fs::metadata(&p) {
            Ok(md) => {
                if let Ok(mt) = md.modified() {
                    entries.push((p, md.len(), mt));
                } else {
                    unstatable.push(p);
                }
            }
            Err(_) => unstatable.push(p),
        }
    }
    for p in unstatable {
        let _ = std::fs::remove_file(p);
    }
    if entries.is_empty() {
        return;
    }
    let mut newest_first = entries.clone();
    newest_first.sort_by_key(|(_, _, m)| std::cmp::Reverse(*m));
    let sized: Vec<(&Path, u64)> = newest_first
        .iter()
        .map(|(p, s, _)| (p.as_path(), *s))
        .collect();
    let keep: HashSet<&Path> = crate::lighttable::thumbs::evict_keep_set(&sized, budget)
        .into_iter()
        .collect();
    for (p, _, _) in entries {
        if !keep.contains(p.as_path()) {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Prune the tile disk store back toward its budget, once per process. Called
/// when the first map canvas builds — the store only grows while the map is
/// open, so pruning anywhere else would sweep a directory nobody is filling.
pub fn prune_tiles_at_startup() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        if let Some(dir) = tile_disk_dir() {
            prune_tiles_dir(&dir, TILE_DISK_BUDGET_BYTES);
        }
    });
}

// ── Fetch + decode (blocking: run inside `gio::spawn_blocking`) ─────────────

/// One fetched tile: decoded RGB bytes plus dimensions. Deliberately NOT a
/// `Pixbuf`: gtk pixbufs are `!Send` and this value crosses `spawn_blocking`
/// thread boundaries, so the pixbuf is built on the GTK loop at completion
/// (see [`tile_pixbuf`]) while everything here stays plain `Send` data.
pub struct FetchedTile {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub byte_len: u64,
}

/// Decode raw tile PNG bytes (off disk or off the wire) into RGB bytes via
/// the already-vendored `image` crate. Oversized or absurd bodies are
/// rejected before decoding: a real tile is 256px square, and the
/// `TILE_MAX_BYTES` cap keeps a captive-portal page from becoming an
/// allocation. Thread-safe by construction (no GTK types involved).
pub fn decode_tile_png(bytes: &[u8]) -> Option<FetchedTile> {
    if bytes.is_empty() || bytes.len() > TILE_MAX_BYTES {
        return None;
    }
    let rgb = image::load_from_memory(bytes).ok()?.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w == 0 || h == 0 || w > 1024 || h > 1024 {
        return None;
    }
    let byte_len = w as u64 * h as u64 * 3;
    Some(FetchedTile {
        rgb: rgb.into_raw(),
        width: w,
        height: h,
        byte_len,
    })
}

/// Build the paintable pixbuf for a fetched tile. Main thread only: it
/// constructs GTK types, so callers must invoke it on the GTK loop (the
/// fetch completion does, right before storing into the memory cache).
pub fn tile_pixbuf(t: &FetchedTile) -> Pixbuf {
    let buf = glib::Bytes::from_owned(t.rgb.clone());
    Pixbuf::from_bytes(
        &buf,
        Colorspace::Rgb,
        false,
        8,
        t.width as i32,
        t.height as i32,
        t.width as i32 * 3,
    )
}

/// One tile HTTP GET with the policy `User-Agent` and a bounded global
/// timeout. `None` covers every failure shape (DNS, connect, timeout, HTTP
/// error — `ureq` surfaces non-2xx as an error, so a 404 for a just-rotated
/// tile is a miss here, never an exception): the caller negative-caches it.
fn tile_http_get(url: &str) -> Option<Vec<u8>> {
    let resp = ureq::get(url)
        .header("User-Agent", tile_user_agent())
        .config()
        .timeout_global(Some(std::time::Duration::from_secs(
            TILE_FETCH_TIMEOUT_SECS,
        )))
        .build()
        .call()
        .ok()?;
    let mut body = resp.into_body();
    let bytes = body.read_to_vec().ok()?;
    if bytes.len() > TILE_MAX_BYTES {
        return None;
    }
    Some(bytes)
}

/// Fetch one tile, blocking: memory callers must not run this on the GTK
/// loop — it does disk IO and up to one HTTPS round trip. Order is disk
/// first (a hit skips the network entirely), then the network with the
/// result persisted for every future session. Holds `_permit` across the
/// whole call so the network leg counts against the concurrency gate; the
/// binding still owns it (lives to scope end, releases on every exit path).
/// `None` means "stay a placeholder". Threading note: the session
/// negative-cache and in-flight sets are main-thread `thread_local!`s, so
/// they are owned OUTSIDE this function — the draw loop skips failed tiles
/// before spawning, and the completion marks failures back on the loop.
/// Calling those sets from in here would touch the worker's own empty copies.
pub fn fetch_tile_blocking(_permit: TileNetPermit, z: i32, x: i32, y: i32) -> Option<FetchedTile> {
    if let Some(bytes) = tile_disk_load(z, x, y) {
        if let Some(t) = decode_tile_png(&bytes) {
            return Some(t);
        }
        // Corrupt disk entry: drop it so the network leg below can heal the
        // slot instead of missing forever on bad bytes.
        if let Some(p) = tile_disk_path(z, x, y) {
            let _ = std::fs::remove_file(p);
        }
    }
    let bytes = tile_http_get(&tile_url(z, x, y))?;
    tile_disk_store(z, x, y, &bytes);
    decode_tile_png(&bytes)
}

// ── Tests (display-free, per repo discipline) ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_at_zoom_zero_is_the_centre_of_the_single_tile() {
        // (0, 0)@0 is the middle of the world: tile (0.5, 0.5)-ish fractions.
        let (x, y) = lon_lat_to_tile_f(0.0, 0.0, 0);
        assert!((x - 0.5).abs() < 1e-9, "x {x}");
        assert!((y - 0.5).abs() < 1e-9, "y {y}");
    }

    #[test]
    fn latitudes_clamp_at_the_mercator_limit() {
        assert_eq!(clamp_lat(90.0), MAX_LAT);
        assert_eq!(clamp_lat(-90.0), -MAX_LAT);
        assert_eq!(clamp_lat(48.85), 48.85);
        assert_eq!(
            clamp_lat(f64::NAN),
            0.0,
            "NaN maps to the equator, never poisons"
        );
        assert_eq!(clamp_lat(f64::INFINITY), 0.0);
        // A clamped pole still projects to a finite tile row (within float
        // dust of the top edge; the normaliser clamps the rest of the way).
        let (_, y) = lon_lat_to_tile_f(0.0, 90.0, 4);
        assert!(y.is_finite() && y > -1e-6, "y {y}");
    }

    #[test]
    fn x_wraps_at_the_antimeridian_and_y_clamps_at_the_poles() {
        // z2 has 4 tiles per axis: x -1 wraps to 3, x 4 wraps to 0.
        assert_eq!(normalize_tile(2, -1, 1), (2, 3, 1));
        assert_eq!(normalize_tile(2, 4, 1), (2, 0, 1));
        assert_eq!(
            normalize_tile(2, 1, 1),
            (2, 1, 1),
            "in-range passes through"
        );
        // y clamps instead of wrapping: there is no tile past either pole.
        assert_eq!(normalize_tile(3, 0, 100), (3, 0, 7));
        assert_eq!(normalize_tile(3, 0, -5), (3, 0, 0));
        // ±180 are the same meridian: same tile column.
        let (xa, _) = lon_lat_to_tile_f(180.0, 0.0, 3);
        let (xb, _) = lon_lat_to_tile_f(-180.0, 0.0, 3);
        assert!((xa - xb).abs() < 1e-9, "{xa} vs {xb}");
        // Cache keys agree across the wrap: one entry, one fetch.
        assert_eq!(tile_cache_key(2, -1, 1), tile_cache_key(2, 3, 1));
    }

    #[test]
    fn zooms_clamp_into_zero_to_nineteen() {
        assert_eq!(clamp_zoom(-3), 0);
        assert_eq!(clamp_zoom(0), 0);
        assert_eq!(clamp_zoom(19), 19);
        assert_eq!(clamp_zoom(99), 19);
        assert_eq!(clamp_zoom(i32::MIN), 0, "no wrap on extreme input");
        // Clamping reaches the URL: no zoom ever builds an invalid path.
        assert_eq!(normalize_tile(99, 0, 0).0, 19);
        assert_eq!(normalize_tile(-1, 0, 0).0, 0);
    }

    #[test]
    fn tile_url_shape_and_normalisation() {
        assert_eq!(
            tile_url(3, 4, 2),
            "https://tile.openstreetmap.org/3/4/2.png"
        );
        assert_eq!(
            tile_url(0, 0, 0),
            "https://tile.openstreetmap.org/0/0/0.png"
        );
        // Out-of-range inputs normalise instead of building invalid URLs.
        assert_eq!(
            tile_url(99, 0, 0),
            "https://tile.openstreetmap.org/19/0/0.png"
        );
        assert_eq!(
            tile_url(2, 4, 99),
            "https://tile.openstreetmap.org/2/0/3.png"
        );
    }

    #[test]
    fn zoom_zero_tile_spans_the_whole_world() {
        let (west, south, east, north) = tile_xy_bounds(0, 0, 0);
        assert!((west + 180.0).abs() < 1e-9, "west {west}");
        assert!((east - 180.0).abs() < 1e-9, "east {east}");
        assert!((north - MAX_LAT).abs() < 1e-6, "north {north}");
        assert!((south + MAX_LAT).abs() < 1e-6, "south {south}");
    }

    #[test]
    fn tile_bounds_tile_the_plane_without_gaps() {
        // z1's four tiles meet at the equator and the prime meridian.
        let (_, _, e0, _) = tile_xy_bounds(1, 0, 0);
        let (w1, _, _, _) = tile_xy_bounds(1, 1, 0);
        assert!((e0 - w1).abs() < 1e-9, "{e0} vs {w1}");
        let (_, s0, _, _) = tile_xy_bounds(1, 0, 0);
        let (_, _, _, n1) = tile_xy_bounds(1, 0, 1);
        assert!((s0 - n1).abs() < 1e-9, "{s0} vs {n1}");
    }

    #[test]
    fn screen_and_geo_round_trip() {
        // Away from the antimeridian the two projections are inverses.
        let (lat, lon) = (48.8581, 2.3525);
        for zoom in [0, 2, 10, 19] {
            let (px, py) = geo_to_screen(lat, lon, 48.0, 2.0, zoom, 800.0, 600.0);
            let (blat, blon) = screen_to_geo(px, py, 48.0, 2.0, zoom, 800.0, 600.0);
            assert!((blat - lat).abs() < 1e-9, "z{zoom} lat {blat}");
            assert!((blon - lon).abs() < 1e-9, "z{zoom} lon {blon}");
        }
        // The centre maps to the viewport middle exactly.
        assert_eq!(
            geo_to_screen(10.0, 20.0, 10.0, 20.0, 5, 800.0, 600.0),
            (400.0, 300.0)
        );
    }

    #[test]
    fn markers_take_the_short_way_around_the_antimeridian() {
        // Centre at +179, marker at -179: two degrees apart, not 358.
        let (mx, _) = geo_to_screen(0.0, -179.0, 0.0, 179.0, 2, 800.0, 600.0);
        assert!((mx - 400.0).abs() < 30.0, "mx {mx}");
    }

    #[test]
    fn world_and_geo_round_trip_including_pole_pin() {
        let (wx, wy) = geo_to_world(48.85, 2.35, 10);
        let (lat, lon) = world_to_geo(wx, wy, 10);
        assert!((lat - 48.85).abs() < 1e-9);
        assert!((lon - 2.35).abs() < 1e-9);
        // A drag past the pole pins at the limit instead of NaN.
        let (plat, _) = world_to_geo(wx, -1e9, 10);
        assert!(plat.is_finite() && (plat - MAX_LAT).abs() < 1e-9, "{plat}");
    }

    #[test]
    fn visible_range_covers_exactly_one_tile_for_a_tile_sized_viewport() {
        // (0,0)@0 on a 256x256 viewport: the single world tile, no sliver.
        assert_eq!(
            visible_tile_range(0.0, 0.0, 0, 256.0, 256.0),
            ((0, 0), (0, 0))
        );
        // A 257px-wide viewport overhangs the tile's left edge by half a
        // pixel, so the (wrapped) neighbour column is correctly included.
        assert_eq!(
            visible_tile_range(0.0, 0.0, 0, 257.0, 256.0),
            ((-1, 0), (0, 0))
        );
        // Rows clamp at the poles: centring past MAX_LAT still yields valid rows.
        let ((_, _), (y0, y1)) = visible_tile_range(85.0, 0.0, 2, 800.0, 600.0);
        assert!(y0 >= 0 && y1 <= 3 && y0 <= y1, "{y0}..={y1}");
        // A tall viewport spans multiple tile ROWS, not just columns: (0,0)
        // at z1 on 256x512 covers x 0..=1 and y 0..=1 (regression: the row
        // range once collapsed to a single row for every viewport).
        assert_eq!(
            visible_tile_range(0.0, 0.0, 1, 256.0, 512.0),
            ((0, 1), (0, 1))
        );
    }

    #[test]
    fn tile_origin_places_tile_zero_at_the_viewport_corner() {
        // (0,0)@0 centred on a 256x256 viewport: tile (0,0) fills it exactly.
        let (sx, sy) = tile_screen_origin(0, 0, 0.0, 0.0, 0, 256.0, 256.0);
        assert!((sx).abs() < 1e-9 && (sy).abs() < 1e-9, "{sx},{sy}");
    }

    #[test]
    fn fit_view_defaults_and_degenerates() {
        assert_eq!(fit_view(&[], 800.0, 600.0), (0.0, 0.0, 2));
        assert_eq!(fit_view(&[(1.0, 2.0)], 0.0, 600.0), (0.0, 0.0, 2));
        assert_eq!(
            fit_view(&[(f64::NAN, 0.0)], 800.0, 600.0),
            (0.0, 0.0, 2),
            "all-non-finite is nothing to fit"
        );
        // One fix centres (near-)exactly on it at street zoom. Near, not
        // bit-exact: the longitude midpoint round-trips through the
        // antimeridian wrap, which costs a float ulp or two.
        let (slat, slon, sz) = fit_view(&[(48.85, 2.35)], 800.0, 600.0);
        assert!((slat - 48.85).abs() < 1e-9 && (slon - 2.35).abs() < 1e-9 && sz == 12);
        let (dlat, dlon, dz) = fit_view(&[(48.85, 2.35), (48.85, 2.35)], 800.0, 600.0);
        assert!((dlat - slat).abs() < 1e-12 && (dlon - slon).abs() < 1e-12 && dz == sz);
    }

    #[test]
    fn fit_view_frames_a_spread_pair() {
        // Paris + London on 800x600: centre between them, zoomed to fit.
        let (clat, clon, z) = fit_view(&[(48.85, 2.35), (51.50, -0.12)], 800.0, 600.0);
        assert!((clat - 50.175).abs() < 1e-9, "{clat}");
        assert!((clon - 1.115).abs() < 1e-9, "{clon}");
        assert!((0..=19).contains(&z), "{z}");
        // Both fixes land inside the viewport at the chosen zoom.
        for (lat, lon) in [(48.85, 2.35), (51.50, -0.12)] {
            let (px, py) = geo_to_screen(lat, lon, clat, clon, z, 800.0, 600.0);
            assert!(
                (0.0..=800.0).contains(&px) && (0.0..=600.0).contains(&py),
                "{px},{py}"
            );
        }
    }

    #[test]
    fn hit_marker_picks_the_nearest_within_radius() {
        let ms = vec![(100.0, 100.0), (200.0, 200.0)];
        assert_eq!(hit_marker(&ms, 103.0, 104.0, 10.0), Some(0));
        assert_eq!(hit_marker(&ms, 500.0, 500.0, 10.0), None);
        assert_eq!(
            hit_marker(&ms, 100.0, 100.0, 0.0),
            None,
            "zero radius never hits"
        );
        // Between the two, the nearer wins.
        assert_eq!(hit_marker(&ms, 140.0, 140.0, 100.0), Some(0));
        assert!(hit_marker(&[], 0.0, 0.0, 10.0).is_none());
    }

    #[test]
    fn user_agent_identifies_the_application() {
        let ua = tile_user_agent();
        assert!(ua.starts_with("c41-darkroom/"), "{ua}");
        assert!(!ua.contains(' '), "no spaces: {ua}");
    }

    #[test]
    fn disk_path_shape_is_store_relative_and_normalised() {
        with_test_tile_dir("shape", |dir| {
            let p = tile_disk_path(2, -1, 99).expect("seam dir set");
            assert_eq!(p, dir.join("2").join("3").join("3.png"));
            assert!(!p.to_str().unwrap().contains(".."), "no escape: {p:?}");
        });
    }

    #[test]
    fn tile_png_round_trips_through_the_image_crate() {
        // Encode a real PNG in-memory (no network), then decode it the way a
        // fetch completion would: dims survive, bytes are RGB tripled.
        let rgb = image::RgbImage::from_fn(64, 32, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(rgb)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let t = decode_tile_png(&bytes).expect("valid PNG decodes");
        assert_eq!((t.width, t.height), (64, 32));
        assert_eq!(t.rgb.len() as u64, 64 * 32 * 3);
        assert_eq!(t.byte_len, 64 * 32 * 3);
        assert!(decode_tile_png(b"not a png").is_none());
        assert!(decode_tile_png(&[]).is_none());
    }

    #[test]
    fn prune_enforces_the_budget_on_real_files() {
        with_test_tile_dir("prune", |dir| {
            for name in ["a", "b", "c"] {
                let p = dir.join("1").join("0").join(format!("{name}.png"));
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(&p, vec![7u8; 120_000]).unwrap();
            }
            prune_tiles_dir(dir, 150_000);
            let left: Vec<_> = walkdir::WalkDir::new(dir)
                .into_iter()
                .flatten()
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some(TILE_DISK_EXT))
                .collect();
            assert_eq!(left.len(), 1, "two entries pruned, newest survives");
        });
    }

    /// Run `f` with [`TEST_TILE_DIR`] pointed at a fresh per-test temp dir.
    /// Mirrors the thumbnail store's seam: per-test isolation without a
    /// process-global env var, removed even when `f` panics.
    fn with_test_tile_dir<T>(name: &str, f: impl FnOnce(&std::path::Path) -> T) -> T {
        struct Cleanup<'a>(&'a std::path::Path);
        impl Drop for Cleanup<'_> {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(self.0);
            }
        }
        let dir = std::env::temp_dir().join(format!("c41_tiles_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let _guard = Cleanup(&dir);
        TEST_TILE_DIR.with(|t| *t.borrow_mut() = Some(dir.clone()));
        let out = f(&dir);
        TEST_TILE_DIR.with(|t| *t.borrow_mut() = None);
        out
    }
}
