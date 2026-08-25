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
//!   the SAME power-of-two quantisation for every consumer ([`bucket_for`]), so
//!   an image decoded while scrolling the grid is a cache hit when the zoomable
//!   canvas opens the same folder, and vice versa. LRU under a byte budget,
//!   never-empty eviction ([`evict_keep_set`]).
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
//! **Cost honesty**: a raw thumbnail pays one full-resolution demosaic per
//! bucket per session — seconds of CPU each, serialised two-at-a-time through
//! the gate. Most raws carry an embedded preview JPEG that would decode orders
//! of magnitude faster (that is darktable's mipmap approach); routing raws to
//! embedded-previews-first-with-demosaic-fallback is the recorded follow-up,
//! alongside persistent on-disk thumbnails.

use gtk4::gdk_pixbuf::prelude::*;
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Mutex;

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
        // Raw branch (m4-140): the full preview's decode at thumbnail scale.
        // Demosaicing runs at sensor resolution regardless — the bound only
        // caps the integer-factor box-average downscale afterwards — but the
        // session caches mean that cost is paid once per (path, bucket).
        let rp = crate::raw_preview::decode_raw_preview(path, max)?;
        let rgb = crate::preview::render_linear_to_srgb8(
            &rp.pixels,
            rp.width,
            rp.height,
            &crate::preview::PreviewParams::default(),
        );
        Some(ThumbImage {
            width: rp.width as i32,
            height: rp.height as i32,
            rgb,
        })
    } else {
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
        let raw = raw.to_str().unwrap();
        let img = decode_uncapped(raw, 256).expect("real raw must decode");
        // The raw branch downscales by an integer box-average factor, so the
        // bound is a ceiling, not a target — only "fits within" is contractual.
        assert!(img.width.max(img.height) <= 256, "longest side bounded");
        assert!(img.width > 0 && img.height > 0);
        assert!(img.rgb.chunks(3).any(|px| px != [0, 0, 0]));
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
