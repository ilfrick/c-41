//! Grayscale-detection scan ported from `src/imageio/format/tiff.c`
//! (`write_image`, the 8bpp `shortfile` branch, m4-219). The C loop walks
//! the 1-pixel-border-excluded interior of the 4-channel u8 export buffer
//! and flips the shared `layers` header field from 1 (grayscale) back to 3
//! (RGB) at the first pixel whose R/G/B lanes differ.
//!
//! What the C loop does, per interior pixel (`x` in `1..width - 1`,
//! `y` in `1..height - 1`):
//! - read lanes 0..2 of the RGBA quad (the alpha lane is never read);
//! - if `abs(r - g) > 2 || abs(r - b) > 2 || abs(g - b) > 2`, set the
//!   shared flag to 3 (otherwise leave it).
//!
//! Bit-exactness notes:
//! - The C flag is written idempotently (1 -> 3 only, never back), so
//!   although the loop ran under OpenMP with a shared flag, the final
//!   flag is schedule-independent: 3 iff some interior pixel differs. A
//!   serial scan returning false at the first differing pixel is
//!   result-identical by construction.
//! - `u8::abs_diff` is exactly the C `abs((int)a - (int)b)` on u8 inputs,
//!   with the same `> 2` threshold spelling (a lane pair differing by
//!   exactly 2 still counts as grey).
//! - The 1-pixel border is never read, as in C: pipeline edge artefacts
//!   are excluded from the decision on both sides.
//! - The `shortfile` / dims-greater-than-4 / bpp gates stay in C; the
//!   kernel treats an empty interior (width or height below 3) as
//!   grayscale, matching the C behaviour of leaving the flag untouched
//!   when there is nothing to scan.
//!
//! The 16-bit (threshold 165) and float (ratio 1.01) sibling scans stay
//! in C for a follow-up increment; only the 8bpp branch is replaced.
//!
//! The Rust kernel is single-threaded sequential; see the first note for
//! why thread scheduling cannot change the result.

/// True when one pixel's R/G/B lanes differ pairwise by more than the C
/// threshold: `abs(r-g) > 2 || abs(r-b) > 2 || abs(g-b) > 2`. Shared by
/// the kernel and the divergent reference below so the two cannot drift
/// on the spelling while still differing structurally.
fn rgb_triple_differs(r: u8, g: u8, b: u8) -> bool {
    r.abs_diff(g) > 2 || r.abs_diff(b) > 2 || g.abs_diff(b) > 2
}

/// Scan the interior of an 8-bit RGBA buffer for colour.
///
/// Port of the former element-wise loop in the `else // 8bpp` branch of
/// `write_image()` (`src/imageio/format/tiff.c`): returns false at the
/// first interior pixel whose R/G/B lanes trip `rgb_triple_differs`,
/// true when the whole interior is grey. See the module docs for the
/// schedule-independence argument and the threshold spelling.
///
/// `width`/`height` are the full-frame dims; only rows `1..height - 1`
/// and columns `1..width - 1` are read. Degenerate dims (either below 3)
/// are a grayscale no-op; short buffers scan the addressable row prefix
/// (no panic, no out-of-bounds access). For the well-formed buffers the
/// C caller passes the clamps never engage and the behaviour is exactly
/// the C loop's.
pub fn tiff_u8_is_grayscale(inp: &[u8], width: usize, height: usize) -> bool {
    if width <= 2 || height <= 2 {
        return true;
    }
    let Some(stride) = width.checked_mul(4) else {
        return true;
    };
    for y in 1..height - 1 {
        let Some(row) = y.checked_mul(stride) else {
            break;
        };
        let Some(row_end) = row.checked_add(stride) else {
            break;
        };
        // The whole row must be addressable: every interior read below
        // stays under row_end, so one check per row keeps all indexing
        // panic-free without per-pixel bounds tests.
        if row_end > inp.len() {
            break;
        }
        for x in 1..width - 1 {
            // x < width and width * 4 did not overflow, so x * 4 cannot
            // overflow; row + x * 4 + 2 stays under row_end.
            let b = row + x * 4;
            if rgb_triple_differs(inp[b], inp[b + 1], inp[b + 2]) {
                return false;
            }
        }
    }
    true
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `tiff_u8_is_grayscale`: walks
/// whole rows as `chunks_exact` and skips the border rows and columns
/// with iterator combinators (the kernel walks the interior with index
/// math and one bounds check per row), so the sweep test cross-checks
/// traversal as well as values. The lane predicate is the shared
/// `rgb_triple_differs` helper by construction (see the module docs: an
/// averaging or max-diff respelling must not be used). Same
/// well-formed-buffers precondition for the sweep; short buffers scan
/// only the whole rows present.
#[cfg(test)]
fn ref_tiff_u8_is_grayscale(inp: &[u8], width: usize, height: usize) -> bool {
    if width <= 2 || height <= 2 {
        return true;
    }
    let Some(stride) = width.checked_mul(4) else {
        return true;
    };
    if stride == 0 {
        return true;
    }
    let rows = (inp.len() / stride).min(height);
    if rows <= 2 {
        return true;
    }
    for row in inp
        .chunks_exact(stride)
        .take(rows)
        .skip(1)
        .take(rows - 2)
    {
        for q in row.chunks_exact(4).skip(1).take(width - 2) {
            if rgb_triple_differs(q[0], q[1], q[2]) {
                return false;
            }
        }
    }
    true
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `inp` must hold at least `4 * width * height` bytes (the C caller
/// passes the 8-bit RGBA export buffer for a `width * height` image).
/// Returns 1 when every interior pixel is grey (lanes pairwise within
/// 2), 0 as soon as one pixel differs. Null pointers, degenerate dims,
/// and overflowing dim products are guarded no-ops returning 1 (no
/// evidence of colour, so the C header keeps its grayscale assumption —
/// the same value the C flag holds when the loop never trips).
#[no_mangle]
pub unsafe extern "C" fn darkroom_tiff_u8_is_grayscale(
    inp: *const u8,
    width: usize,
    height: usize,
) -> i32 {
    if inp.is_null() {
        return 1;
    }
    // validate the products BEFORE building the slice below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(len) = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(4))
    else {
        return 1;
    };
    if len > isize::MAX as usize {
        return 1;
    }
    let buf = std::slice::from_raw_parts(inp, len);
    if tiff_u8_is_grayscale(buf, width, height) {
        1
    } else {
        0
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn grey_frame(w: usize, h: usize, v: u8, alpha: u8) -> Vec<u8> {
        let mut buf = vec![0u8; 4 * w * h];
        for q in buf.chunks_exact_mut(4) {
            q[0] = v;
            q[1] = v;
            q[2] = v;
            q[3] = alpha;
        }
        buf
    }

    // Solid grey frames stay grayscale on every rail (0/1/128/254/255);
    // the alpha lane is never read, so garbage there must not matter.
    #[test]
    fn solid_grey_is_grayscale() {
        for v in [0u8, 1, 128, 254, 255] {
            for alpha in [0u8, 7, 128, 255] {
                let buf = grey_frame(6, 5, v, alpha);
                assert!(tiff_u8_is_grayscale(&buf, 6, 5), "v {v} alpha {alpha}");
                assert!(ref_tiff_u8_is_grayscale(&buf, 6, 5), "v {v} alpha {alpha}");
            }
        }
        // a 3x3 frame has exactly one interior pixel; grey there is grey
        let buf = grey_frame(3, 3, 200, 0);
        assert!(tiff_u8_is_grayscale(&buf, 3, 3));
    }

    // Each of the three lane pairs trips the scan on its own, and the
    // threshold is strictly greater-than: a pair differing by exactly 2
    // still counts as grey, 3 does not.
    #[test]
    fn each_pair_trips_and_threshold_is_strict() {
        let plant = |w: usize, h: usize, x: usize, y: usize, px: [u8; 4]| {
            let mut buf = grey_frame(w, h, 100, 255);
            let b = 4 * (y * w + x);
            buf[b..b + 4].copy_from_slice(&px);
            buf
        };
        // r-g pair only
        assert!(!tiff_u8_is_grayscale(&plant(5, 5, 2, 2, [103, 100, 100, 255]), 5, 5));
        // r-b pair only
        assert!(!tiff_u8_is_grayscale(&plant(5, 5, 2, 2, [100, 100, 103, 255]), 5, 5));
        // g-b pair only
        assert!(!tiff_u8_is_grayscale(&plant(5, 5, 1, 3, [100, 103, 100, 255]), 5, 5));
        // the (0,2,4) shape: neighbours agree within 2 but the outer
        // pair does not, so only the third comparison can catch it
        assert!(!tiff_u8_is_grayscale(&plant(5, 5, 2, 2, [0, 2, 4, 255]), 5, 5));
        // boundary: diff of exactly 2 on every pair stays grey
        assert!(tiff_u8_is_grayscale(&plant(5, 5, 2, 2, [100, 102, 100, 255]), 5, 5));
        assert!(tiff_u8_is_grayscale(&plant(5, 5, 2, 2, [100, 102, 102, 255]), 5, 5));
    }

    // The 1-pixel border is never read: colour anywhere on the outer
    // ring must not flip the verdict (pipeline edge artefacts are
    // excluded by construction, as in C).
    #[test]
    fn border_pixels_ignored() {
        let (w, h) = (7, 6);
        let mut buf = grey_frame(w, h, 90, 255);
        let paint = |buf: &mut [u8], x: usize, y: usize| {
            let b = 4 * (y * w + x);
            buf[b] = 0;
            buf[b + 1] = 255;
            buf[b + 2] = 17;
        };
        paint(&mut buf, 0, 0);
        paint(&mut buf, 3, 0);
        paint(&mut buf, w - 1, 2);
        paint(&mut buf, 0, h - 1);
        paint(&mut buf, 5, h - 1);
        paint(&mut buf, 2, h - 1);
        assert!(tiff_u8_is_grayscale(&buf, w, h));
        assert!(ref_tiff_u8_is_grayscale(&buf, w, h));
        // one step inside the border the same colour trips immediately
        paint(&mut buf, 1, 1);
        assert!(!tiff_u8_is_grayscale(&buf, w, h));
        assert!(!ref_tiff_u8_is_grayscale(&buf, w, h));
    }

    // Kernel and reference must agree exactly over several shapes, both
    // on near-grey noise (threshold neighbourhood) and on frames with
    // sparse planted colour pixels (both verdicts exercised).
    #[test]
    fn matches_reference_over_sweep() {
        let shapes = [
            (1usize, 1usize),
            (2, 5),
            (5, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (7, 9),
            (16, 16),
            (32, 21),
        ];
        for (si, (w, h)) in shapes.iter().enumerate() {
            let (w, h) = (*w, *h);
            // near-grey noise: lanes within +-2 of a per-pixel base
            let mut buf = vec![0u8; 4 * w * h];
            for (i, q) in buf.chunks_exact_mut(4).enumerate() {
                let base = ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(si as u64)) % 256;
                let base8 = base as u8;
                let tweak = (i % 5) as u8; // 0..4: some quads trip, most do not
                q[0] = base8;
                q[1] = base8.saturating_add(tweak.min(3));
                q[2] = base8.saturating_sub((tweak + 1) / 2);
                q[3] = (i % 251) as u8;
            }
            assert_eq!(
                tiff_u8_is_grayscale(&buf, w, h),
                ref_tiff_u8_is_grayscale(&buf, w, h),
                "shape {w}x{h} noise"
            );
            // mostly grey with a sparse planted colour pixel inside the
            // interior (when the frame has one)
            let mut buf2 = grey_frame(w, h, 128, 255);
            if w > 2 && h > 2 {
                let b = 4 * (1 * w + 1);
                buf2[b] = 200;
            }
            assert_eq!(
                tiff_u8_is_grayscale(&buf2, w, h),
                ref_tiff_u8_is_grayscale(&buf2, w, h),
                "shape {w}x{h} planted"
            );
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        // empty buffer with live dims: no panic, no evidence of colour
        assert!(tiff_u8_is_grayscale(&[], 6, 5));
        // zero / sub-interior dims: nothing to scan
        let buf = grey_frame(6, 5, 0, 0);
        assert!(tiff_u8_is_grayscale(&buf, 0, 0));
        assert!(tiff_u8_is_grayscale(&buf, 6, 0));
        assert!(tiff_u8_is_grayscale(&buf, 0, 5));
        assert!(tiff_u8_is_grayscale(&buf, 2, 5));
        assert!(tiff_u8_is_grayscale(&buf, 6, 2));
        // truncated buffer: the addressable row prefix decides; colour
        // past the cut is not read
        let mut full = grey_frame(6, 6, 77, 255);
        let b = 4 * (4 * 6 + 4); // interior pixel in the last interior row
        full[b] = 250;
        let cut = full.len() - 2 * 4 * 6; // drop rows 4..5 entirely
        assert!(tiff_u8_is_grayscale(&full[..cut], 6, 6));
        assert!(!tiff_u8_is_grayscale(&full, 6, 6));
        // colour inside the addressable prefix still trips
        let b2 = 4 * (1 * 6 + 1);
        full[b2] = 250;
        assert!(!tiff_u8_is_grayscale(&full[..cut], 6, 6));
    }

    #[test]
    fn ffi_round_trip() {
        let (w, h) = (9, 7);
        let grey = grey_frame(w, h, 140, 255);
        let mut colour = grey.clone();
        let b = 4 * (3 * w + 4);
        colour[b] = 10;
        colour[b + 1] = 250;
        unsafe {
            assert_eq!(darkroom_tiff_u8_is_grayscale(grey.as_ptr(), w, h), 1);
            assert_eq!(darkroom_tiff_u8_is_grayscale(colour.as_ptr(), w, h), 0);
        }
        assert!(tiff_u8_is_grayscale(&grey, w, h));
        assert!(!tiff_u8_is_grayscale(&colour, w, h));
    }

    #[test]
    fn ffi_guards() {
        let buf = grey_frame(4, 4, 10, 255);
        unsafe {
            // null pointer: no evidence of colour, grayscale assumed
            assert_eq!(darkroom_tiff_u8_is_grayscale(std::ptr::null(), 4, 4), 1);
            // zero dims
            assert_eq!(darkroom_tiff_u8_is_grayscale(buf.as_ptr(), 0, 4), 1);
            assert_eq!(darkroom_tiff_u8_is_grayscale(buf.as_ptr(), 4, 0), 1);
            // overflowing dim product: guarded before any slice is built
            assert_eq!(
                darkroom_tiff_u8_is_grayscale(buf.as_ptr(), usize::MAX, 2),
                1
            );
            assert_eq!(
                darkroom_tiff_u8_is_grayscale(buf.as_ptr(), usize::MAX, usize::MAX),
                1
            );
        }
    }
}
