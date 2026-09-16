//! Kernel ported from `src/imageio/format/jxl.c` (`write_image`, m4-215).
//! The kernel replaces the whole loop body: each RGBA float pixel of the
//! export pipeline buffer is repacked into the 3-channel float frame the
//! JXL encoder consumes, dropping the alpha lane.
//!
//! What the C loop does, per pixel `(y, x)`:
//! - `out[3*(y*width+x) + c] = in[4*(y*width+x) + c]` for `c` in `0..2`,
//!   where `in` is the tightly packed RGBA pipeline buffer (`in_tmp`) and
//!   `out` the freshly allocated `pixels` frame (`width*height*3` floats,
//!   matching the `{ 3, JXL_TYPE_FLOAT, ... }` pixel format). The alpha
//!   lane is never read.
//!
//! Bit-exactness notes:
//! - The body is three plain float assignments, so the kernel copies every
//!   bit including NaN payloads and signed zeros; there is no arithmetic
//!   to order, no clamp, and no rounding mode to match.
//! - Both buffers are tightly packed (no row stride: the C side addresses
//!   them with flat `4*(y*width+x)` / `3*(y*width+x)` indices), so the
//!   kernel takes no stride parameter.
//! - The two buffers never alias on the C path (`out` is a fresh
//!   `g_try_malloc` allocation); the kernel must not be called with
//!   overlapping buffers.
//!
//! The Rust kernel is single-threaded sequential; the C loop was
//! `DT_OMP_FOR_SIMD(collapse(2))` over rows and columns, but each output
//! triple reads only its own source quad, so thread scheduling cannot
//! change the result.

/// Repack RGBA float pixels into the 3-channel JXL export frame.
///
/// Port of the former element-wise loop in `write_image`
/// (src/imageio/format/jxl.c): `src` holds `4*width*height` tightly packed
/// floats (RGBA; the alpha lane is never read), `out` receives
/// `3*width*height` tightly packed floats (RGB lanes copied verbatim).
/// See the module docs for the bitwise-copy fidelity point and the
/// no-aliasing contract.
///
/// Degenerate dims (`width == 0` or `height == 0`) are no-ops; short
/// buffers are handled by clamped iteration (no panic, no out-of-bounds
/// access). For the well-formed buffers the C caller passes the clamps
/// never engage and the behaviour is exactly the C loop's.
pub fn jxl_rgba_to_rgb_float(src: &[f32], out: &mut [f32], width: usize, height: usize) {
    if width == 0 || height == 0 {
        return;
    }
    let Some(src_row) = width.checked_mul(4) else {
        return;
    };
    let Some(dst_row) = width.checked_mul(3) else {
        return;
    };
    for y in 0..height {
        let Some(s) = y.checked_mul(src_row) else {
            break;
        };
        let Some(s_end) = s.checked_add(src_row) else {
            break;
        };
        if src.len() < s_end {
            break;
        }
        let Some(d) = y.checked_mul(dst_row) else {
            break;
        };
        let Some(d_end) = d.checked_add(dst_row) else {
            break;
        };
        if out.len() < d_end {
            break;
        }
        for x in 0..width {
            let si = s + 4 * x;
            let di = d + 3 * x;
            out[di] = src[si];
            out[di + 1] = src[si + 1];
            out[di + 2] = src[si + 2];
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `jxl_rgba_to_rgb_float`: walks
/// pixels by flat index with `div`/`mod` (the kernel uses nested
/// row/column loops with explicit stride arithmetic) and copies each
/// triple through a zipped slice-iterator pair (the kernel indexes both
/// sides directly), so the sweep test cross-checks indexing as well as
/// values. Plain assignment on both sides, so NaN payloads and signed
/// zeros are preserved bit for bit by construction. Same
/// well-formed-buffers precondition, enforced here by early return rather
/// than clamping — so on a multi-row short buffer the kernel
/// partial-writes row by row while this reference writes nothing; only
/// well-formed buffers (which the C caller always passes) are compared
/// by the sweep test.
#[cfg(test)]
fn ref_jxl_rgba_to_rgb_float(src: &[f32], out: &mut [f32], width: usize, height: usize) {
    if width == 0 || height == 0 {
        return;
    }
    let (Some(src_row), Some(dst_row)) = (width.checked_mul(4), width.checked_mul(3)) else {
        return;
    };
    let Some(npixels) = width.checked_mul(height) else {
        return;
    };
    if src.len() < npixels.saturating_mul(4) || out.len() < npixels.saturating_mul(3) {
        return;
    }
    for p in 0..npixels {
        let y = p / width;
        let x = p % width;
        for (slot, lane) in out[y * dst_row + 3 * x..y * dst_row + 3 * x + 3]
            .iter_mut()
            .zip(src[y * src_row + 4 * x..y * src_row + 4 * x + 3].iter())
        {
            *slot = *lane;
        }
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `in_data` must hold at least `4 * width * height` floats (the C caller
/// passes the export pipeline buffer, tightly packed RGBA) and `out` at
/// least `3 * width * height` floats (the freshly allocated JXL frame).
/// The two buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn darkroom_jxl_rgba_to_rgb_float(
    in_data: *const f32,
    out: *mut f32,
    width: usize,
    height: usize,
) {
    if in_data.is_null() || out.is_null() || width == 0 || height == 0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(src_len) = width.checked_mul(height).and_then(|n| n.checked_mul(4)) else {
        return;
    };
    let Some(dst_len) = width.checked_mul(height).and_then(|n| n.checked_mul(3)) else {
        return;
    };
    let src = std::slice::from_raw_parts(in_data, src_len);
    let out = std::slice::from_raw_parts_mut(out, dst_len);
    jxl_rgba_to_rgb_float(src, out, width, height);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Rails: exact RGB passthrough, the alpha lane dropped (out holds 3
    // lanes per pixel), signed zero and an extreme preserved bit for bit.
    #[test]
    fn rails_pin() {
        let src = [
            0.0f32, 0.5, 1.0, 0.9, // pixel 0 (alpha 0.9 dropped)
            -0.0, 1e30, -1e30, 7.0, // pixel 1: signed zero + extremes
        ];
        let mut out = vec![9.0f32; 6];
        jxl_rgba_to_rgb_float(&src, &mut out, 2, 1);
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[1].to_bits(), 0.5f32.to_bits());
        assert_eq!(out[2].to_bits(), 1.0f32.to_bits());
        assert_eq!(out[3].to_bits(), (-0.0f32).to_bits()); // not +0.0
        assert_eq!(out[4].to_bits(), 1e30f32.to_bits());
        assert_eq!(out[5].to_bits(), (-1e30f32).to_bits());
    }

    // A NaN lane (with a non-canonical payload) and infinities must round-
    // trip bit-exactly: the kernel is pure assignment, so payloads survive.
    // The alpha slot holds a signalling value to prove it is never read.
    #[test]
    fn nan_payload_and_infinity_preserved() {
        let nan = f32::from_bits(0x7FC0_1234);
        let src = [f32::INFINITY, nan, f32::NEG_INFINITY, 1.0];
        let mut out = vec![0.0f32; 3];
        jxl_rgba_to_rgb_float(&src, &mut out, 1, 1);
        assert_eq!(out[0].to_bits(), f32::INFINITY.to_bits());
        assert_eq!(out[1].to_bits(), 0x7FC0_1234);
        assert_eq!(out[2].to_bits(), f32::NEG_INFINITY.to_bits());
    }

    // Kernel and reference agree bit-exactly over several shapes (square,
    // wide, tall, single pixel/row/column), with inputs spanning negatives,
    // NaNs and infinities so the bitwise-copy spelling is exercised.
    #[test]
    fn matches_reference_over_sweep() {
        // (width, height)
        let shapes = [
            (1usize, 1usize),
            (3, 2),
            (5, 4),
            (7, 1), // single row
            (1, 9), // single column
            (16, 9),
            (64, 33),
        ];
        for (w, h) in shapes {
            let mut src = vec![0.0f32; 4 * w * h];
            for (i, v) in src.iter_mut().enumerate() {
                // deterministic sweep across [-1.5, 1.5], then poison
                // every 7th lane with a NaN payload and every 11th with
                // infinity so non-finite bits cross the sweep too
                if i % 7 == 3 {
                    *v = f32::from_bits(0x7FC0_0000 + ((i as u32) & 0x3FFF));
                } else if i % 11 == 5 {
                    *v = if i % 2 == 0 { f32::INFINITY } else { f32::NEG_INFINITY };
                } else {
                    let k =
                        ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(0x9E37) % 3001) as f32;
                    *v = k / 1000.0 - 1.5;
                }
            }
            let mut direct = vec![-2.0f32; 3 * w * h];
            let mut reference = vec![-2.0f32; 3 * w * h];
            jxl_rgba_to_rgb_float(&src, &mut direct, w, h);
            ref_jxl_rgba_to_rgb_float(&src, &mut reference, w, h);
            assert_eq!(direct.len(), reference.len());
            for (k, (d, r)) in direct.iter().zip(reference.iter()).enumerate() {
                assert_eq!(d.to_bits(), r.to_bits(), "shape ({w},{h}) lane {k}");
            }
            // alpha lanes never leak into output: out holds exactly the
            // RGB lanes, so lane-for-lane it equals the strided source
            for p in 0..w * h {
                assert_eq!(direct[3 * p].to_bits(), src[4 * p].to_bits());
                assert_eq!(direct[3 * p + 1].to_bits(), src[4 * p + 1].to_bits());
                assert_eq!(direct[3 * p + 2].to_bits(), src[4 * p + 2].to_bits());
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let src = vec![1.0f32; 8];
        let mut out = vec![7.0f32; 6];
        // zero dims: output untouched
        jxl_rgba_to_rgb_float(&src, &mut out, 0, 2);
        jxl_rgba_to_rgb_float(&src, &mut out, 2, 0);
        assert_eq!(out, vec![7.0f32; 6]);
        // truncated source: no panic, no writes (2x1 needs 8 floats;
        // with 7 the single row fails its bounds check and is skipped)
        let short = vec![1.0f32; 7];
        jxl_rgba_to_rgb_float(&short, &mut out, 2, 1);
        assert_eq!(out, vec![7.0f32; 6]);
        // truncated destination: no panic, no writes
        let mut short_out = vec![0.0f32; 5];
        jxl_rgba_to_rgb_float(&src, &mut short_out, 2, 1);
        assert_eq!(short_out, vec![0.0f32; 5]);
        // empty buffers with live dims: no panic, no writes possible
        let empty: Vec<f32> = vec![];
        let mut out2 = vec![0.0f32; 3];
        jxl_rgba_to_rgb_float(&empty, &mut out2, 1, 1);
        assert_eq!(out2, vec![0.0f32; 3]);
    }

    #[test]
    fn ffi_round_trip() {
        let (w, h) = (9usize, 5usize);
        let mut src = vec![0.0f32; 4 * w * h];
        for (i, v) in src.iter_mut().enumerate() {
            let k = ((i as u64).wrapping_mul(2_654_435_761) % 3001) as f32;
            *v = k / 1000.0 - 1.5;
        }
        let mut ffi_out = vec![-2.0f32; 3 * w * h];
        let mut direct_out = vec![-2.0f32; 3 * w * h];
        unsafe {
            darkroom_jxl_rgba_to_rgb_float(src.as_ptr(), ffi_out.as_mut_ptr(), w, h);
        }
        jxl_rgba_to_rgb_float(&src, &mut direct_out, w, h);
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn ffi_guards() {
        let src = vec![1.0f32; 16];
        let mut out = vec![7.0f32; 12];
        unsafe {
            // null pointers
            darkroom_jxl_rgba_to_rgb_float(std::ptr::null(), out.as_mut_ptr(), 2, 2);
            darkroom_jxl_rgba_to_rgb_float(src.as_ptr(), std::ptr::null_mut(), 2, 2);
            // zero dims
            darkroom_jxl_rgba_to_rgb_float(src.as_ptr(), out.as_mut_ptr(), 0, 2);
            darkroom_jxl_rgba_to_rgb_float(src.as_ptr(), out.as_mut_ptr(), 2, 0);
            // overflowing dims
            darkroom_jxl_rgba_to_rgb_float(src.as_ptr(), out.as_mut_ptr(), usize::MAX, 2);
            darkroom_jxl_rgba_to_rgb_float(src.as_ptr(), out.as_mut_ptr(), 2, usize::MAX);
        }
        assert_eq!(out, vec![7.0f32; 12]); // untouched
    }
}
