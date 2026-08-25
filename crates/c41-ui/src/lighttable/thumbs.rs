//! Session-wide **thumbnail service** (m4-140) — the one place that turns a
//! file path into bounded RGB8 pixels, plus the session caches in front of it.
//!
//! Before this module the lighttable had two independent thumbnail decoders
//! (the grid cell bind and, since m4-139, the zoomable canvas), both
//! gdk-pixbuf-only: camera raws — most of any real catalogue — decoded to
//! nothing, and each retry re-read megabytes. Both surfaces now go through
//! this module, which owns three shared things:
//!
//! * **The decoder** ([`decode_with_permit`]): raws route through the same
//!   pipeline the full preview uses ([`crate::raw_preview::decode_raw_preview`]
//!   plus sRGB encode); everything else goes through gdk-pixbuf scaled *during*
//!   decode (`connect_size_prepared`, the m4-132 lesson). Output is owned
//!   packed RGB8, `Send` by construction, so decodes run on worker threads and
//!   only bytes ever cross back.
//!
//! * **A session negative cache** ([`is_failed`] / [`mark_failed`] /
//!   [`clear_failed`]): an undecodable file costs one attempt per collection
//!   load, not one per painted frame or cell recycle. Cleared at
//!   [`super::fill_grid`], the single point where "the collection changed",
//!   which is also the retry opportunity.
//!
//! * **One pixel cache** ([`lookup`] / [`store`]), keyed `(path, bucket)` with
//!   the SAME power-of-two quantisation for every consumer ([`bucket_for`]).
//!   Honest scope note (review MINOR-2, m4-143): today only the grid reads and
//!   fills it; the zoomable canvas keeps its own pixbuf cache because it blits
//!   through cairo rather than a Picture. So a canvas decode is NOT visible
//!   here yet, and a sequential re-request from the other surface still pays a
//!   second demosaic — publishing completions into one shared cache is the
//!   recorded follow-up. LRU under a byte budget, never-empty eviction
//!   ([`evict_keep_set`]).
//!
//! **Concurrency**: at most [`MAX_CONCURRENT_DECODES`] decodes run at once. A
//! raw decode materialises a full sensor-resolution linear buffer before
//! downscaling (tens of MB), and one scroll of a fresh folder wants a decode
//! per visible cell — unbounded, that's a memory spike plus gio thread-pool
//! thrash. Slots are claimed BEFORE any task is spawned
//! (`DecodePermit::try_acquire`, non-blocking): a caller that finds the gate
//! busy simply doesn't spawn (the next bind/paint retries naturally), so no
//! worker thread ever parks waiting for a slot and unrelated blocking work
//! (rating / colour-label queries) can't queue behind parked decoders.
//!
//! **One decode per path** (m4-143, m4-141 review N4): before touching the
//! gate every consumer registers `(path, bucket)` in one shared in-flight map
//! ([`inflight_register`]) and does nothing while an equal-or-bigger decode for
//! that path is already running — so a rebound cell, a scroll bounce, or the
//! canvas racing the grid can never run two demosaics of the same raw at the
//! same time. Refusals retry on the same bounded 150ms timer as a busy gate,
//! so the pending request converges once the owner finishes. Sequential
//! re-decodes across surfaces are a separate gap — see the pixel-cache note
//! above.
//!
//! **Cost honesty**: a raw thumbnail pays one full-resolution demosaic —
//! seconds of CPU each, serialised two-at-a-time through the gate. Since
//! m4-141 that cost is paid once per (file, bucket) per *machine*, not per
//! session: finished decodes persist to a darktable-mipmap-style on-disk
//! store (see the "Persistent disk cache" section) keyed by path + bucket,
//! with the source's mtime sealed inside every entry — a replaced or re-saved
//! raw invalidates itself, while a DELETED file keeps showing its last known
//! render (darktable behaves the same until a refresh). Only the very first view of
//! each file stays expensive; serving that from the camera's own embedded
//! preview JPEG is still the recorded follow-up (rawloader exposes no
//! thumbnail parser, and large-preview placement is vendor-specific).

use gtk4::gdk_pixbuf::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

/// Cache byte budget. A 512² RGB8 thumbnail is ~0.75 MB, so this holds tens of
/// thousands of grid thumbs or hundreds of culling-sized ones; eviction keeps
/// worst-case memory flat no matter how large the catalogue grows.
const BUDGET_BYTES: u64 = 128 * 1024 * 1024;

/// Max simultaneous decodes — see the module doc for why the cap exists and
/// why slots are pre-claimed rather than waited on.
const MAX_CONCURRENT_DECODES: usize = 2;

// ── Decode target sizing ────────────────────────────────────────────────────

/// The decode target for a needed pixel size: the smallest power-of-two bucket
/// that contains it, clamped to sane thumbnail limits. Shared by every
/// consumer so the pixel cache's keys agree across surfaces; continuous
/// zooming fires a decode at most once per doubling instead of once per pixel.
pub(crate) fn bucket_for(px: i32) -> u32 {
    for b in [128u32, 256, 512, 1024, 2048] {
        if px <= b as i32 {
            return b;
        }
    }
    2048
}

// ── Concurrency gate ────────────────────────────────────────────────────────

/// Live decode count. `Mutex::new` is const, so this needs no lazy init.
static ACTIVE_DECODES: Mutex<usize> = Mutex::new(0);

/// RAII slot in the decode gate. Held across one whole decode; released on
/// every return path including panics. Never blocks: slots are *claimed* by
/// the caller before spawning anything (see the module doc).
#[must_use = "dropping the permit immediately releases the slot"]
pub(crate) struct DecodePermit;

impl DecodePermit {
    /// Claim a free slot, or `None` when [`MAX_CONCURRENT_DECODES`] are live.
    pub(crate) fn try_acquire() -> Option<Self> {
        let mut n = ACTIVE_DECODES.lock().unwrap();
        if *n >= MAX_CONCURRENT_DECODES {
            return None;
        }
        *n += 1;
        Some(Self)
    }
}

impl Drop for DecodePermit {
    fn drop(&mut self) {
        *ACTIVE_DECODES.lock().unwrap() -= 1;
    }
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// One decoded thumbnail: tightly-packed RGB8 (`rowstride == width * 3`). The
/// result of dropping a real [`DecodePermit`] into [`decode_with_permit`].
pub struct ThumbImage {
    pub width: i32,
    pub height: i32,
    pub rgb: Vec<u8>,
}

impl ThumbImage {
    /// Bytes actually held — the cache's accounting unit.
    fn byte_len(&self) -> u64 {
        self.rgb.len() as u64
    }
}

/// Run one decode while holding `permit`. Call on a worker thread (`spawn_blocking`)
/// with the permit claimed beforehand on the main thread — the pairing keeps
/// pool threads free of gate waits. `None` = unsupported or unreadable; callers
/// pair that with [`mark_failed`]. Panics inside a decoder surface as a join
/// error to the awaiting side, NOT as a failure mark (a crashing decoder is a
/// bug, not a corrupt file).
pub(crate) fn decode_with_permit(
    _permit: DecodePermit,
    path: &str,
    max_dim: u32,
) -> Option<ThumbImage> {
    // The underscore-prefixed binding still owns the permit: it lives to the
    // end of this scope and releases the slot on every exit path.
    decode_uncapped(path, max_dim)
}

/// The decode itself, gate-free — tests bypass the gate through this, and
/// [`decode_with_permit`] wraps it.
fn decode_uncapped(path: &str, max_dim: u32) -> Option<ThumbImage> {
    // Clamp once so no downstream i32 cast can wrap, whatever a future caller
    // passes (current callers: ≤ 2048 buckets, ≤ viewport-sized cells).
    let max = max_dim.clamp(1, 8192) as usize;
    if crate::raw_preview::is_raw_path(path) {
        return decode_raw_thumb(path, max);
    }
    {
        let data = std::fs::read(path).ok()?;
        let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
        loader.connect_size_prepared(move |loader, w, h| {
            let longest = w.max(h);
            if longest > max as i32 {
                // One scale factor on both axes: `set_size` does NOT preserve
                // aspect ratio for you.
                let scale = f64::from(max as i32) / f64::from(longest);
                loader.set_size(
                    ((f64::from(w) * scale) as i32).max(1),
                    ((f64::from(h) * scale) as i32).max(1),
                );
            }
        });
        // Both unconditional: a loader finalized without `close()` emits a
        // g_warning, so an early return on a rejected header would print one
        // per retry.
        let _ = loader.write(&data);
        let _ = loader.close();
        let raw = loader.pixbuf()?;
        let Some((fw, fh)) = fit_inside_box(raw.width(), raw.height(), max as i32) else {
            return None; // undecodable header reports 0×0
        };
        // Degenerate target keeps the unscaled pixels; identical dims skip the
        // pointless identity resample (a full Bilinear copy of every thumb).
        let pb = if fw == raw.width() && fh == raw.height() {
            raw
        } else {
            raw.scale_simple(fw, fh, gtk4::gdk_pixbuf::InterpType::Bilinear)
                .unwrap_or(raw)
        };
        Some(ThumbImage {
            width: pb.width(),
            height: pb.height(),
            rgb: pixbuf_to_rgb8(&pb),
        })
    }
}

/// `super::fit_inside` re-exported shape: longest-side fit, `None` when either
/// side is zero (undecodable headers report 0×0).
fn fit_inside_box(src_w: i32, src_h: i32, max: i32) -> Option<(i32, i32)> {
    if src_w <= 0 || src_h <= 0 || max <= 0 {
        return None;
    }
    let scale = f64::from(max) / f64::from(src_w.max(src_h));
    if scale >= 1.0 {
        return Some((src_w, src_h));
    }
    Some((
        ((f64::from(src_w) * scale) as i32).max(1),
        ((f64::from(src_h) * scale) as i32).max(1),
    ))
}

/// Flatten a pixbuf of any channel count / rowstride padding into packed RGB8.
/// Alpha IS honoured: straight-alpha pixels are composited over black
/// (`rgb·a/255`), matching what the old `Texture::for_pixbuf` path showed over
/// the dark cells — a transparent-background PNG keeps its background black
/// instead of exposing whatever under-colour its stored RGB carries.
fn pixbuf_to_rgb8(pb: &gtk4::gdk_pixbuf::Pixbuf) -> Vec<u8> {
    let (w, h) = (pb.width().max(0) as usize, pb.height().max(0) as usize);
    let nc = pb.n_channels() as usize;
    let rs = pb.rowstride() as usize;
    // `read_pixel_bytes` hands back an immutable snapshot; `Bytes` derefs to
    // `[u8]`, rowstride padding included.
    let src_bytes = pb.read_pixel_bytes();
    let src: &[u8] = &src_bytes;
    let mut out = vec![0u8; w * h * 3];
    if nc < 3 {
        return out; // exotic monochrome pixbuf: stays black rather than panicking
    }
    let has_alpha = nc >= 4;
    for y in 0..h {
        let row = &src[y * rs..];
        for x in 0..w {
            let px = &row[x * nc..x * nc + 3];
            let dst = &mut out[(y * w + x) * 3..(y * w + x) * 3 + 3];
            if has_alpha {
                let a = u32::from(row[x * nc + 3]);
                dst[0] = ((u32::from(px[0]) * a) / 255) as u8;
                dst[1] = ((u32::from(px[1]) * a) / 255) as u8;
                dst[2] = ((u32::from(px[2]) * a) / 255) as u8;
            } else {
                dst.copy_from_slice(px);
            }
        }
    }
    out
}

// ── Persistent disk cache (raw decodes survive sessions) ───────────────────

/// Byte budget for the on-disk raw-thumbnail store. A raw decode costs
/// seconds of CPU; paying that once per machine instead of once per session
/// is what makes re-opening a large catalogue instant. [`prune_disk_dir`]
/// keeps the total under this regardless of catalogue age.
const DISK_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// Entry file format: magic + version + packed-RGB dims + source mtime (all
/// little-endian), then exactly `w*h*3` payload bytes. No encoder sits in the
/// loop, so what loads back is bit-identical to what was decoded — the
/// thumbnail a cache hit paints is the thumbnail the one expensive decode
/// produced.
const DISK_MAGIC: &[u8; 4] = b"C41T";
const DISK_VERSION: u32 = 1;
const DISK_HEADER_LEN: usize = 32;
const DISK_EXT: &str = "c41thumb";

#[cfg(test)]
thread_local! {
    /// Test seam: when set, [`disk_cache_dir`] returns this instead of the
    /// user-wide directory, so tests never read or pollute a real cache.
    /// Each test runs on its own thread, which makes this per-test isolation
    /// (a process-global env var could not be set safely from parallel tests).
    static TEST_DISK_DIR: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Where decoded raw thumbnails persist. `C41_THUMB_CACHE_DIR` overrides
/// wholesale (tests use the thread-local above; this env var exists for
/// container experiments), otherwise the XDG cache dir — the same
/// `~/.cache/...` convention darktable's mipmap store uses.
fn disk_cache_dir() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(d) = TEST_DISK_DIR.with(|t| t.borrow().clone()) {
        return Some(d);
    }
    if let Some(d) = std::env::var_os("C41_THUMB_CACHE_DIR").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("c41").join("thumbs"))
}

/// FNV-1a 64-bit — deterministic across processes, which std's `DefaultHasher`
/// is NOT (it is keyed randomly per process, so hashed filenames would orphan
/// every cached entry at every launch). Not cryptographic: the input is our
/// own path string, and a collision can at worst serve another file's render
/// when their sealed mtimes also coincide — any mtime difference rejects the
/// hit and the next decode overwrites the entry.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Source-file mtime in nanoseconds since the epoch, `None` when it cannot be
/// read (file vanished between listing and decode). Sealed into every entry:
/// a load whose stored mtime differs from a readable current one is stale and
/// is re-decoded; an unreadable current mtime serves the last known render
/// (the collection still lists the image — a blank cell would be worse).
/// Known boundary, shared with darktable's own mtime invalidation: a
/// replacement that deliberately PRESERVES the mtime (`cp -p`,
/// restore-from-backup) defeats staleness — the seal has no way to see it.
fn source_mtime_nanos(path: &str) -> Option<u128> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
}

/// Deterministic filename for `(path, bucket)` — short and filesystem-safe
/// regardless of how deep the source path is. The mtime is deliberately NOT
/// part of the name: it lives inside the entry so that a vanished source can
/// still resolve to (and be served from) its old file.
fn entry_file_name(hash: u64, bucket: u32) -> String {
    format!("{hash:016x}-{bucket}.{DISK_EXT}")
}

/// Full path of the cache entry for one raw decode request.
fn disk_key(dir: &Path, path: &str, bucket: u32) -> PathBuf {
    dir.join(entry_file_name(fnv1a(path.as_bytes()), bucket))
}

/// Serialise a thumbnail plus its source mtime into the on-disk format (see
/// [`DISK_MAGIC`]). An unreadable mtime seals as 0 — consistent because loads
/// compare `Some(current)` against the sealed value only when a current value
/// exists at all.
fn encode_disk_entry(img: &ThumbImage, mtime_nanos: Option<u128>) -> Vec<u8> {
    let mut out = Vec::with_capacity(DISK_HEADER_LEN + img.rgb.len());
    out.extend_from_slice(DISK_MAGIC);
    out.extend_from_slice(&DISK_VERSION.to_le_bytes());
    out.extend_from_slice(&img.width.to_le_bytes());
    out.extend_from_slice(&img.height.to_le_bytes());
    out.extend_from_slice(&mtime_nanos.unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&img.rgb);
    out
}

/// Parse back what [`encode_disk_entry`] wrote, returning the payload and its
/// sealed mtime. Anything off — magic, version, non-positive dims, byte count
/// not matching the declared dims (truncated or corrupt file) — is a miss,
/// never a panic: the next successful decode overwrites the entry via rename
/// anyway. Dims are trusted only after the length check pins them to the
/// payload size; this is the user's own cache directory, same trust level as
/// darktable's mipmap store.
fn decode_disk_entry(bytes: &[u8]) -> Option<(ThumbImage, u128)> {
    if bytes.len() < DISK_HEADER_LEN || &bytes[0..4] != DISK_MAGIC {
        return None;
    }
    let ver = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    if ver != DISK_VERSION {
        return None;
    }
    let w = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let h = i32::from_le_bytes(bytes[12..16].try_into().ok()?);
    if w <= 0 || h <= 0 {
        return None;
    }
    let mtime = u128::from_le_bytes(bytes[16..32].try_into().ok()?);
    let n = (w as usize).checked_mul(h as usize)?.checked_mul(3)?;
    if bytes.len() != DISK_HEADER_LEN + n {
        return None;
    }
    Some((ThumbImage { width: w, height: h, rgb: bytes[DISK_HEADER_LEN..].to_vec() }, mtime))
}

/// Load one validated entry. `current_mtime` is the SOURCE file's mtime now:
/// `Some(x)` with a differently-sealed x means the raw was replaced or
/// re-saved → miss; `None` (source unreadable/absent) serves the last known
/// render. Any IO problem is likewise just a miss — never surfaced as an
/// error.
fn disk_load(entry: &Path, current_mtime: Option<u128>) -> Option<ThumbImage> {
    let bytes = std::fs::read(entry).ok()?;
    let (img, sealed) = decode_disk_entry(&bytes)?;
    if let Some(cur) = current_mtime {
        if sealed != cur {
            return None;
        }
    }
    Some(img)
}

/// Uniqueness for tmp files during atomic writes: pid separates concurrent
/// processes, the counter separates threads within one, and a nanosecond
/// timestamp covers even separate containers sharing a volume with distinct
/// pid namespaces (both would otherwise be "pid 42, counter 0").
static DISK_WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Bytes appended since the last prune sweep. Pruning is a full directory
/// stat-and-sort — measurable churn at five-figure entry counts — so it runs
/// only after this much NEW data has landed, not on every store. At most
/// 1/16th of the budget overruns between sweeps; regenerable cache, fine.
static DISK_BYTES_SINCE_PRUNE: AtomicU64 = AtomicU64::new(0);

/// Persist one entry atomically — write a tmp file in the SAME directory,
/// then rename over any existing entry (same-directory rename is atomic), so
/// concurrent writers of one key race to identical bytes and readers never
/// observe a half-written file — then prune the store back toward its budget
/// once enough new data accumulated ([`DISK_BYTES_SINCE_PRUNE`]). Every
/// failure path is silently skipped: a full disk must degrade to "no
/// caching", never break thumbnails.
fn disk_store(dir: &Path, entry: &Path, img: &ThumbImage, mtime_nanos: Option<u128>) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let payload = encode_disk_entry(img, mtime_nanos);
    let seq = DISK_WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = dir.join(format!(".tmp-{}-{seq:x}-{now_nanos:x}.{DISK_EXT}", std::process::id()));
    if std::fs::write(&tmp, &payload).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    if std::fs::rename(&tmp, entry).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    let added = payload.len() as u64;
    if DISK_BYTES_SINCE_PRUNE.fetch_add(added, Ordering::Relaxed) + added
        > DISK_BUDGET_BYTES / 16
    {
        DISK_BYTES_SINCE_PRUNE.store(0, Ordering::Relaxed);
        prune_disk_dir(dir, DISK_BUDGET_BYTES, Some(entry));
    }
}

/// Delete oldest-mtime entries until the directory fits `budget`, never
/// deleting `keep` (the entry just written; needed because equal mtimes tie
/// in arbitrary `read_dir` order and could push the fresh entry out of the
/// greedy keep-set). Newest-first keep policy is the shared [`evict_keep_set`]
/// walk, same as the session LRU: newest entries survive, the regenerable
/// tail goes. Entries whose metadata cannot even be read are deleted outright
/// rather than silently exempting them from every future sweep. Pure
/// selection math lives in [`prune_plan`], pinned by tests there.
fn prune_disk_dir(dir: &Path, budget: u64, keep: Option<&Path>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut unstatable: Vec<PathBuf> = Vec::new();
    let entries: Vec<(PathBuf, u64, SystemTime)> = rd
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some(DISK_EXT))
        .filter_map(|e| {
            let p = e.path();
            match e.metadata() {
                Ok(md) => md.modified().ok().map(|mt| (p.clone(), md.len(), mt)),
                // A transiently unstatable entry costs one regenerate later;
                // keeping it around would put it outside the budget forever.
                Err(_) => {
                    unstatable.push(p);
                    None
                }
            }
        })
        .collect();
    for p in unstatable {
        let _ = std::fs::remove_file(p);
    }
    for victim in prune_plan(entries, budget) {
        if keep.is_some_and(|k| k == victim) {
            continue;
        }
        let _ = std::fs::remove_file(victim);
    }
}

/// Which entries exceed `budget`: walk newest-first keeping greedily until the
/// next entry no longer fits (the first always survives, mirroring
/// [`evict_keep_set`]'s never-empty rule — here it merely avoids deleting the
/// freshest work); everything older is a victim.
fn prune_plan(entries: Vec<(PathBuf, u64, SystemTime)>, budget: u64) -> Vec<PathBuf> {
    let mut newest_first = entries.clone();
    newest_first.sort_by_key(|(_, _, m)| std::cmp::Reverse(*m));
    let sized: Vec<(&Path, u64)> =
        newest_first.iter().map(|(p, s, _)| (p.as_path(), *s)).collect();
    let keep: HashSet<&Path> = evict_keep_set(&sized, budget).into_iter().collect();
    entries
        .into_iter()
        .filter(|(p, _, _)| !keep.contains(p.as_path()))
        .map(|(p, _, _)| p)
        .collect()
}

/// The raw branch of [`decode_uncapped`]: the persistent store is consulted
/// first (a hit skips demosaicing entirely — this is what makes the second
/// session over a folder instant), then the full preview decode runs and its
/// result is persisted for every future session. Both surfaces' requests land
/// here through their buckets, so one visit warms every bucket it touched.
fn decode_raw_thumb(path: &str, max: usize) -> Option<ThumbImage> {
    let dir = disk_cache_dir();
    let mtime = source_mtime_nanos(path);
    let entry = dir.as_deref().map(|d| disk_key(d, path, max as u32));
    if let Some(e) = &entry {
        if let Some(hit) = disk_load(e, mtime) {
            return Some(hit);
        }
    }
    // Demosaicing runs at sensor resolution regardless — `max` only caps the
    // integer-factor box-average downscale afterwards.
    let rp = crate::raw_preview::decode_raw_preview(path, max)?;
    let rgb = crate::preview::render_linear_to_srgb8(
        &rp.pixels,
        rp.width,
        rp.height,
        &crate::preview::PreviewParams::default(),
    );
    let img = ThumbImage { width: rp.width as i32, height: rp.height as i32, rgb };
    if let (Some(d), Some(e)) = (&dir, &entry) {
        disk_store(d, e, &img, mtime);
    }
    Some(img)
}

// ── Session pixel cache ─────────────────────────────────────────────────────

struct Entry {
    key: (String, u32),
    img: Rc<ThumbImage>,
}

thread_local! {
    /// MRU-first entry list. A `VecDeque` scanned linearly is deliberate: hit
    /// rates are high and the list is bounded by the byte budget, so the scan
    /// is a few hundred pointer compares against a HashMap key-build per hit.
    /// Main-thread only (binds, paints and completions all run on the loop).
    static CACHE: RefCell<VecDeque<Entry>> = const { RefCell::new(VecDeque::new()) };
    /// Paths whose decode failed this session. Reset by [`clear_failed`] on
    /// every collection load, which is also the retry opportunity.
    static FAILED: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

/// Cached pixels for `(path, max_dim)`, if present. Moves the entry to MRU.
pub(crate) fn lookup(path: &str, max_dim: u32) -> Option<Rc<ThumbImage>> {
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let pos = cache.iter().position(|e| e.key.0 == path && e.key.1 == max_dim)?;
        let entry = cache.remove(pos).unwrap();
        let img = Rc::clone(&entry.img);
        cache.push_front(entry);
        Some(img)
    })
}

/// True if this path already failed to decode since the last collection load.
pub(crate) fn is_failed(path: &str) -> bool {
    FAILED.with(|f| f.borrow().contains(path))
}

/// Record a decode failure — the caller decides when a None really is one (a
/// busy gate or a panicked worker is not).
pub(crate) fn mark_failed(path: &str) {
    FAILED.with(|f| f.borrow_mut().insert(path.to_string()));
}

/// Forget all failures. Called from [`super::fill_grid`]: the collection
/// changed, so "this file can't be decoded" deserves re-examination.
pub(crate) fn clear_failed() {
    FAILED.with(|f| f.borrow_mut().clear());
}

// ── In-flight dedupe ────────────────────────────────────────────────────────

thread_local! {
    /// Decodes currently running, `path → requested bucket`. ONE map shared by
    /// every consumer (grid cell bind AND zoomable frame, m4-143): an
    /// equal-or-bigger request for a path that already has a decode running is
    /// refused, so a cell rebind, a scroll bounce, or the canvas opening while
    /// the grid fills can never start a second demosaic of the same fresh raw.
    /// A strictly bigger bucket supersedes the entry (its output dominates for
    /// cache purposes); the superseded decode keeps running and simply no
    /// longer owns the slot at completion. Main-thread only, like every other
    /// piece of session state here; entries are never bulk-cleared — they are
    /// transient by construction, removed by their own completions.
    static INFLIGHT: RefCell<HashMap<String, u32>> = RefCell::new(HashMap::new());
}

/// Register `(path, bucket)` as the decode about to spawn. Returns `false` —
/// having registered nothing — when an equal-or-bigger decode for `path` is
/// already in flight, meaning the caller must not spawn. Returns `true` having
/// registered, which obliges the caller to [`inflight_unregister`] with the
/// same bucket on EVERY exit path (including bailing out before spawning).
pub(crate) fn inflight_register(path: &str, bucket: u32) -> bool {
    INFLIGHT.with(|m| {
        let mut inflight = m.borrow_mut();
        if inflight.get(path).is_some_and(|prev| *prev >= bucket) {
            return false;
        }
        inflight.insert(path.to_string(), bucket);
        true
    })
}

/// Release a registration — but only if this exact `(path, bucket)` still owns
/// it: a superseding bigger decode registered after ours must survive ours.
pub(crate) fn inflight_unregister(path: &str, bucket: u32) {
    INFLIGHT.with(|m| {
        let mut inflight = m.borrow_mut();
        if inflight.get(path).copied() == Some(bucket) {
            inflight.remove(path);
        }
    });
}

/// Insert/replace under the byte budget, MRU-first. Evicts via the shared
/// never-empty policy ([`evict_keep_set`]).
pub(crate) fn store(path: &str, max_dim: u32, img: ThumbImage) -> Rc<ThumbImage> {
    let key = (path.to_string(), max_dim);
    let rc = Rc::new(img);
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        cache.retain(|e| e.key != key);
        cache.push_front(Entry { key, img: Rc::clone(&rc) });
        let bytes: u64 = cache.iter().map(|e| e.img.byte_len()).sum();
        if bytes > BUDGET_BYTES {
            let entries: Vec<((String, u32), u64)> = cache
                .iter()
                .map(|e| (e.key.clone(), e.img.byte_len()))
                .collect();
            let keep: HashSet<(String, u32)> =
                evict_keep_set(&entries, BUDGET_BYTES).into_iter().collect();
            cache.retain(|e| keep.contains(&e.key));
        }
    });
    rc
}

/// Keep-set of an LRU under a byte budget: walk `entries` newest-first and keep
/// until the budget is exhausted; everything older is evictable. The first
/// entry is always kept even when oversized — evicting everything would turn
/// every frame into a reload storm, and going over budget by one entry beats
/// that. Generic over the key so both this module's `(path, size)` keys and
/// the zoomable texture cache's flat string keys share one policy. (Moved here
/// from zoomable.rs in m4-140, pinned by the tests below.)
pub(crate) fn evict_keep_set<K: Clone + Eq + std::hash::Hash>(
    entries: &[(K, u64)],
    budget: u64,
) -> Vec<K> {
    let mut kept = Vec::new();
    let mut used = 0u64;
    for (key, bytes) in entries {
        if used == 0 || used + bytes <= budget {
            kept.push(key.clone());
            used += bytes;
        }
    }
    kept
}

// ── Tests (display-free, per repo discipline) ───────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a small solid-colour image so the pixbuf branch has a real file to
    /// chew on. Names carry the pid so parallel CI jobs on one machine can't
    /// collide in the shared temp dir.
    fn write_test_image(dir: &std::path::Path, name: &str, rgba: bool, w: u32, h: u32) -> String {
        let p = dir.join(format!("c41_{}_{name}", std::process::id()));
        if rgba {
            image::RgbaImage::from_fn(w, h, |x, y| {
                image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255])
            })
            .save_with_format(&p, image::ImageFormat::Png)
            .unwrap();
        } else {
            image::RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
            })
            .save_with_format(&p, image::ImageFormat::Jpeg)
            .unwrap();
        }
        p.to_str().unwrap().to_string()
    }

    #[test]
    fn jpeg_decode_bounds_dims_and_packs_rgb() {
        let dir = std::env::temp_dir();
        let p = write_test_image(&dir, "small.jpg", false, 320, 200);
        let img = decode_uncapped(&p, 128).expect("jpeg must decode");
        assert_eq!(img.width.max(img.height), 128, "longest side hits the bound");
        assert!(img.width <= 128 && img.height <= 128);
        assert_eq!(
            img.rgb.len(),
            img.width as usize * img.height as usize * 3,
            "packed RGB8, rowstride == width*3"
        );
        assert!(img.rgb.chunks(3).any(|px| px != [0, 0, 0]), "not black");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn jpeg_at_or_under_bound_is_not_upscaled() {
        let dir = std::env::temp_dir();
        let p = write_test_image(&dir, "tiny.jpg", false, 96, 64);
        let img = decode_uncapped(&p, 256).expect("jpeg must decode");
        assert_eq!((img.width, img.height), (96, 64));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn transparent_png_composites_over_black_not_stored_rgb() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("c41_{}_alpha.png", std::process::id()));
        // Left half opaque red, right half fully transparent white — the
        // transparent half must come out BLACK after the flatten.
        image::RgbaImage::from_fn(64, 32, |x, _| {
            if x < 32 {
                image::Rgba([255, 0, 0, 255])
            } else {
                image::Rgba([255, 255, 255, 0])
            }
        })
        .save_with_format(&p, image::ImageFormat::Png)
        .unwrap();
        let img = decode_uncapped(p.to_str().unwrap(), 128).expect("png must decode");
        let mid_row = |x: usize| {
            let o = (16 * img.width as usize + x) * 3;
            [img.rgb[o], img.rgb[o + 1], img.rgb[o + 2]]
        };
        assert_eq!(mid_row(4), [255, 0, 0], "opaque red survives exactly");
        assert_eq!(mid_row(img.width as usize - 4), [0, 0, 0], "transparent → black");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn missing_file_decodes_to_none_without_panicking() {
        assert!(decode_uncapped("/nonexistent/c41/nope.jpg", 128).is_none());
    }

    #[test]
    fn failed_set_round_trips_and_clears() {
        clear_failed();
        assert!(!is_failed("/x/y.orf"));
        mark_failed("/x/y.orf");
        assert!(is_failed("/x/y.orf"));
        clear_failed();
        assert!(!is_failed("/x/y.orf"));
    }

    #[test]
    fn inflight_blocks_equal_and_smaller_until_the_owner_releases() {
        // Under --test-threads=1 every test shares one thread, so leave the
        // map as found on every exit path; the unique paths here also keep the
        // test correct when tests run on parallel threads.
        assert!(inflight_register("/a/raw.nef", 256), "first claim wins");
        assert!(
            !inflight_register("/a/raw.nef", 256),
            "equal bucket is refused — the running decode covers it"
        );
        assert!(
            !inflight_register("/a/raw.nef", 128),
            "smaller bucket is refused too: its pixels will be cached anyway"
        );
        // Owner-checked release: a stale (superseded-never-happened) bucket
        // must NOT free the slot.
        inflight_unregister("/a/raw.nef", 128);
        assert!(!inflight_register("/a/raw.nef", 128), "stale release is a no-op");
        inflight_unregister("/a/raw.nef", 256);
        assert!(inflight_register("/a/raw.nef", 128), "real release frees the path");
        inflight_unregister("/a/raw.nef", 128);
    }

    #[test]
    fn inflight_supersede_takes_over_then_blocks() {
        assert!(inflight_register("/b/raw.nef", 128));
        // A strictly bigger bucket supersedes while the smaller keeps running:
        // its output dominates for cache purposes, so it must be spawnable.
        assert!(inflight_register("/b/raw.nef", 512));
        assert!(!inflight_register("/b/raw.nef", 512), "equal to the new owner");
        // The superseded decode finishes first: releasing ITS bucket must not
        // evict the entry the bigger decode now owns.
        inflight_unregister("/b/raw.nef", 128);
        assert!(!inflight_register("/b/raw.nef", 256), "still owned by the bigger");
        inflight_unregister("/b/raw.nef", 512);
        assert!(inflight_register("/b/raw.nef", 256), "path free again after both");
        inflight_unregister("/b/raw.nef", 256);
        // Reverse interleaving (review NIT-1): the BIGGER decode completes
        // first and the stale smaller release lands afterwards — still ends
        // empty, and the late stale release is a no-op.
        assert!(inflight_register("/b/raw.nef", 128));
        assert!(inflight_register("/b/raw.nef", 512));
        inflight_unregister("/b/raw.nef", 512);
        assert!(inflight_register("/b/raw.nef", 256), "empty once the owner releases");
        inflight_unregister("/b/raw.nef", 256);
        inflight_unregister("/b/raw.nef", 128); // stale latecomer
    }

    #[test]
    fn inflight_is_per_path() {
        // The whole point of m4-143: two surfaces asking for DIFFERENT files
        // never block each other; only same-path duplicates dedupe.
        assert!(inflight_register("/c/one.nef", 512));
        assert!(inflight_register("/d/two.nef", 512));
        assert!(inflight_register("/e/three.jpg", 256));
        assert!(!inflight_register("/d/two.nef", 512));
        inflight_unregister("/c/one.nef", 512);
        assert!(!inflight_register("/d/two.nef", 256), "unrelated release changed nothing");
        inflight_unregister("/d/two.nef", 512);
        inflight_unregister("/e/three.jpg", 256);
    }

    #[test]
    fn gate_admits_two_then_blocks_until_a_slot_is_dropped() {
        // This test is the gate's only user (decode tests go through
        // decode_uncapped), so the process-wide counter starts at zero here
        // regardless of which tests run concurrently.
        let a = DecodePermit::try_acquire();
        let b = DecodePermit::try_acquire();
        assert!(a.is_some() && b.is_some(), "the cap admits two");
        assert!(DecodePermit::try_acquire().is_none(), "a third is refused");
        drop(b);
        assert!(
            DecodePermit::try_acquire().is_some(),
            "a dropped slot admits exactly one more"
        );
        // Remaining permits drop at scope end, restoring the empty gate.
    }

    #[test]
    fn cache_stores_lookups_and_evicts_mru_first_never_empty() {
        // Isolated keys so parallel test threads can't collide.
        let tag = format!("t{}", line!());
        let mk = |n: usize| ThumbImage { width: n as i32, height: 1, rgb: vec![7; n] };
        let a = store(&format!("/{tag}/a"), 100, mk(70_000_000)); // ~70 MB each:
        let b = store(&format!("/{tag}/b"), 100, mk(70_000_000)); // two exceed the budget
        // Never-empty: the newest oversized entry survives even over budget…
        assert!(Rc::ptr_eq(&lookup(&format!("/{tag}/b"), 100).unwrap(), &b));
        // …and the older one was the eviction victim.
        assert!(lookup(&format!("/{tag}/a"), 100).is_none());
        drop(a);
        drop(b);

        // Replacement: storing the same key twice leaves ONE entry, updated.
        let v1 = store(&format!("/{tag}/r"), 50, mk(10));
        let v2 = store(&format!("/{tag}/r"), 50, mk(20));
        let hit = lookup(&format!("/{tag}/r"), 50).unwrap();
        assert_eq!(hit.rgb.len(), 20);
        assert!(Rc::ptr_eq(&hit, &v2) || Rc::ptr_eq(&hit, &v1));
        assert_eq!(
            CACHE.with(|c| c.borrow().iter().filter(|e| e.key.0 == format!("/{tag}/r")).count()),
            1
        );

        // Bucket separation: same path, different size = different entries.
        store(&format!("/{tag}/multi"), 64, mk(4));
        store(&format!("/{tag}/multi"), 256, mk(8));
        assert!(lookup(&format!("/{tag}/multi"), 64).is_some());
        assert!(lookup(&format!("/{tag}/multi"), 256).is_some());
    }

    #[test]
    fn keep_set_walks_newest_first_and_keeps_the_first_entry_even_overweight() {
        let e = |n: &str, b: u64| (n.to_string(), b);
        // Newest-first input: big(90) is kept by the never-empty rule alone
        // (90 > 40); after it `used` already exceeds the budget so nothing
        // else fits.
        assert_eq!(
            evict_keep_set(&[e("big", 90), e("mid", 30), e("small", 5)], 40),
            vec!["big"]
        );
        // Normal case: fill greedily newest-first until the next entry no
        // longer fits.
        assert_eq!(
            evict_keep_set(&[e("a", 10), e("b", 30), e("c", 5)], 40),
            vec!["a", "b"]
        );
    }

    #[test]
    fn buckets_are_powers_of_two_with_hard_limits() {
        assert_eq!(bucket_for(1), 128);
        assert_eq!(bucket_for(127), 128);
        assert_eq!(bucket_for(128), 128);
        assert_eq!(bucket_for(129), 256);
        assert_eq!(bucket_for(600), 1024);
        assert_eq!(bucket_for(2048), 2048);
        assert_eq!(bucket_for(100_000), 2048, "huge requests cap at the top bucket");
        assert_eq!(bucket_for(0), 128);
        assert_eq!(bucket_for(-5), 128);
    }

    /// Run `f` with [`TEST_DISK_DIR`] pointed at a fresh per-test temp dir, so
    /// raw-branch decodes never touch the real user cache. The name keeps
    /// parallel tests on one machine from colliding; the guard removes the
    /// dir even when `f` panics.
    fn with_test_disk_dir<T>(name: &str, f: impl FnOnce(&std::path::Path) -> T) -> T {
        struct Cleanup<'a>(&'a std::path::Path);
        impl Drop for Cleanup<'_> {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(self.0);
            }
        }
        let dir = std::env::temp_dir()
            .join(format!("c41_thumbcache_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let _guard = Cleanup(&dir);
        TEST_DISK_DIR.with(|t| *t.borrow_mut() = Some(dir.clone()));
        let out = f(&dir);
        // Normal-path seam reset; a panicking thread takes its thread-local
        // with it, so the seam can never leak across tests either way.
        TEST_DISK_DIR.with(|t| *t.borrow_mut() = None);
        out
    }

    #[test]
    fn raw_decode_produces_bounded_nontrivial_pixels() {
        // testdata/ is untracked (local + Docker-gate only); skip cleanly where
        // absent so CI machines without the fixture still pass. Resolved from
        // the manifest dir because cargo runs tests with the CRATE root as CWD,
        // not the repo root — a bare relative path silently skips everywhere.
        let raw = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/portrait.orf");
        if !raw.exists() {
            return;
        }
        with_test_disk_dir("bounded", |_| {
            let raw = raw.to_str().unwrap();
            let img = decode_uncapped(raw, 256).expect("real raw must decode");
            // The raw branch downscales by an integer box-average factor, so
            // the bound is a ceiling, not a target — only "fits within" is
            // contractual.
            assert!(img.width.max(img.height) <= 256, "longest side bounded");
            assert!(img.width > 0 && img.height > 0);
            assert!(img.rgb.chunks(3).any(|px| px != [0, 0, 0]));
        });
    }

    #[test]
    fn fnv1a_matches_published_vectors() {
        // Canonical FNV-1a 64 test vectors (empty = offset basis) pin the
        // exact variant; determinism across processes is the property the
        // cache filenames depend on.
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn entry_name_is_deterministic_and_sensitive_to_every_key_part() {
        let h = fnv1a(b"/x/img.orf");
        assert_eq!(
            entry_file_name(h, 256),
            entry_file_name(h, 256),
            "same inputs, same filename"
        );
        assert_ne!(entry_file_name(h, 256), entry_file_name(h, 512), "bucket in key");
        assert_ne!(
            entry_file_name(h, 256),
            entry_file_name(fnv1a(b"/y/img.orf"), 256),
            "path in key"
        );
        assert!(
            entry_file_name(h, 2048).len() < 64,
            "filename stays short whatever the inputs"
        );
    }

    #[test]
    fn disk_entry_round_trips_and_load_validates_the_sealed_mtime() {
        let img = ThumbImage { width: 244, height: 71, rgb: (0..244 * 71 * 3).map(|i| i as u8).collect() };
        let bytes = encode_disk_entry(&img, Some(12345));
        let (back, sealed) = decode_disk_entry(&bytes).expect("valid entry parses");
        assert_eq!((back.width, back.height), (img.width, img.height));
        assert_eq!(sealed, 12345, "mtime survives the round trip");
        assert_eq!(back.rgb, img.rgb, "bit-identical payload");

        // Corrupt shapes are misses, never panics: truncated tail, appended
        // byte (length check), bad magic, unknown version (a future format
        // change invalidates old files wholesale), zero dims, header alone.
        assert!(decode_disk_entry(&bytes[..bytes.len() - 1]).is_none());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(decode_disk_entry(&extra).is_none());
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert!(decode_disk_entry(&magic).is_none());
        let mut ver = bytes.clone();
        ver[4] = 99;
        assert!(decode_disk_entry(&ver).is_none());
        let mut zero = bytes.clone();
        zero[8..12].copy_from_slice(&0i32.to_le_bytes());
        assert!(decode_disk_entry(&zero).is_none());
        assert!(decode_disk_entry(&bytes[..DISK_HEADER_LEN]).is_none());

        // Load-time staleness contract: matching mtime hits, a different one
        // (replaced/re-saved source) misses, an unreadable current mtime
        // (source vanished) serves the last known render.
        let entry_path = std::env::temp_dir()
            .join(format!("c41_{}_mtcheck.{DISK_EXT}", std::process::id()));
        std::fs::write(&entry_path, &bytes).unwrap();
        assert!(disk_load(&entry_path, Some(12345)).is_some(), "matching mtime hits");
        assert!(disk_load(&entry_path, Some(54321)).is_none(), "changed mtime is stale");
        assert!(disk_load(&entry_path, None).is_some(), "vanished source still serves");
        let _ = std::fs::remove_file(&entry_path);
    }

    #[test]
    fn raw_decode_survives_source_deletion_via_disk_cache() {
        // testdata/ untracked — skip where absent (see bounded test).
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/portrait.orf");
        if !src.exists() {
            return;
        }
        with_test_disk_dir("survives", |cache| {
            // Work on a copy so deleting the "source" can't hurt other tests
            // that read the fixture concurrently.
            let copy = std::env::temp_dir()
                .join(format!("c41_{}_delme.orf", std::process::id()));
            std::fs::copy(&src, &copy).unwrap();
            let path = copy.to_str().unwrap().to_string();

            let first = decode_uncapped(&path, 256).expect("first view decodes for real");
            // Exactly one entry persisted for this request.
            let entries: Vec<_> = std::fs::read_dir(cache)
                .unwrap()
                .flatten()
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some(DISK_EXT))
                .collect();
            assert_eq!(entries.len(), 1, "one bucket, one file");

            // Delete the SOURCE. The second request can now only be served by
            // the disk store — this is the proof that hits skip the decoder.
            std::fs::remove_file(&copy).unwrap();
            let second = decode_uncapped(&path, 256).expect("disk cache serves it");
            assert_eq!((second.width, second.height), (first.width, first.height));
            assert_eq!(second.rgb, first.rgb, "bit-identical to the original decode");
        });
    }

    #[test]
    fn prune_plan_keeps_the_newest_tail_and_victims_are_the_rest() {
        let t = |s: u64| SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(s);
        let e = |n: &str, sz: u64, m: u64| (PathBuf::from(n), sz, t(m));
        // Budget 25, sizes 10/20/30: newest-first keep walk holds 30 (first,
        // always kept), then nothing else fits — both older entries go, even
        // though keeping {10} or none would fit strictly. Mirrors the session
        // LRU's over-budget-by-one-entry trade-off on the freshest work.
        let victims = prune_plan(
            vec![e("old", 10, 1), e("mid", 20, 2), e("new", 30, 3)],
            25,
        );
        assert_eq!(victims.len(), 2);
        assert!(victims.contains(&PathBuf::from("old")));
        assert!(victims.contains(&PathBuf::from("mid")));

        // Under budget: nobody is a victim regardless of order.
        assert!(prune_plan(vec![e("a", 5, 1), e("b", 5, 2)], 100).is_empty());
    }

    #[test]
    fn prune_disk_dir_enforces_budget_on_real_files() {
        with_test_disk_dir("prune", |dir| {
            let mk = |n: u8| {
                ThumbImage { width: 200, height: 200, rgb: vec![n; 200 * 200 * 3] } // 120 KB each
            };
            let key = |name: &str| disk_key(dir, name, 256);
            disk_store(dir, &key("a.orf"), &mk(1), Some(1));
            disk_store(dir, &key("b.orf"), &mk(2), Some(2));
            disk_store(dir, &key("c.orf"), &mk(3), Some(3)); // ~360 KB total
            // 150 KB budget fits exactly one 120 KB entry.
            prune_disk_dir(dir, 150_000, None);
            let left: Vec<_> =
                std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
            assert_eq!(left.len(), 1, "two entries pruned, newest survives");
            assert_eq!(
                std::fs::metadata(&left[0]).unwrap().len(),
                (DISK_HEADER_LEN + 200 * 200 * 3) as u64
            );
            // A hit after pruning still validates end-to-end.
            assert!(disk_load(&left[0], None).is_some());
        });
    }

    #[test]
    fn decoded_results_are_cached_by_key_and_shared_via_rc() {
        // Store-then-lookup is the contract both consumers rely on (decoding
        // itself is cache-free; callers own the store call).
        let tag = format!("t{}", line!());
        let img = ThumbImage { width: 2, height: 1, rgb: vec![1, 2, 3, 4, 5, 6] };
        let stored = store(&format!("/{tag}/x"), 64, img);
        let hit = lookup(&format!("/{tag}/x"), 64).expect("same key hits");
        assert!(Rc::ptr_eq(&stored, &hit));
        assert!(lookup(&format!("/{tag}/x"), 65).is_none(), "other bucket misses");
    }
}
