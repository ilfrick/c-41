//! Separable 5-tap B-spline &agrave;-trous blur (m4-192).
//!
//! Ports `blur_2D_Bspline` (`src/common/bspline.h:142`, loop ~152-172): the
//! separable `[1 4 6 4 1] / 16` blur applied first vertically into a row
//! scratch buffer and then horizontally into the output row, with the tap
//! stride `mult` (&agrave;-trous: taps at `row +/- mult` and `row +/- 2*mult`,
//! clamped to the image edges) and optional `MAX(0, ...)` negative clipping.
//! Only the `#else` (non-nontemporal) path is ported — `USE_NONTEMPORAL` is
//! `FALSE`, so the nontemporal branch is dead.
//!
//! Row visiting order reuses [`crate::math::dwt_interleave_rows`]: it is a
//! pure cache optimisation over independent output rows, so the result is
//! identical to natural order. The C per-thread `tempbuf`/`padded_size`
//! plumbing stays in the C signature (callers unchanged); the Rust side owns
//! a private row scratch buffer instead.
//!
//! Fidelity notes:
//! - Tap values are the exact C literals (`1/16`, `4/16`, `6/16` — all
//!   exactly representable in `f32`).
//! - Each 5-tap dot product keeps the C association order
//!   `f0*x0 + f1*x1 + f2*x2 + f3*x3 + f4*x4` (left-associative, no leading
//!   zero accumulator), so results agree with the C bit-exactly.
//! - Clipping replicates glib `MAX(0.0f, v)` = `(0.0f > v) ? 0.0f : v`,
//!   which propagates NaN and preserves `-0.0`. It is deliberately *not*
//!   `f32::max`, which would return `0.0` for NaN inputs.

/// B-spline taps, exactly as the C `static const float filter[5]`.
const FILTER: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// glib `MAX(0.0f, v)` when `clip` is set, identity otherwise.
///
/// Written as an explicit comparison (not `f32::max`) so NaN inputs stay NaN
/// and `-0.0` stays `-0.0`, exactly like the C macro.
#[inline(always)]
fn maybe_clip(v: f32, clip: bool) -> f32 {
    if clip {
        if 0.0f32 > v {
            0.0
        } else {
            v
        }
    } else {
        v
    }
}

/// One 5-tap dot product in the C association order:
/// `f[0]*x0 + f[1]*x1 + f[2]*x2 + f[3]*x3 + f[4]*x4`, left-associative.
#[inline(always)]
fn tap5(f: &[f32; 5], x0: f32, x1: f32, x2: f32, x3: f32, x4: f32) -> f32 {
    f[0] * x0 + f[1] * x1 + f[2] * x2 + f[3] * x3 + f[4] * x4
}

/// Safe separable B-spline blur kernel.
///
/// `out`/`inp` hold `4 * width * height` packed-RGBA floats each and must not
/// overlap (matching the C `restrict` qualifiers); `mult >= 1` is the
/// &agrave;-trous tap stride. Zero dims or `mult == 0` are no-ops. Buffer
/// lengths are a caller contract (`debug_assert`ed).
pub fn blur_2d_bspline_kernel(
    out: &mut [f32],
    inp: &[f32],
    width: usize,
    height: usize,
    mult: usize,
    clip_negatives: bool,
) {
    if width == 0 || height == 0 || mult == 0 {
        return;
    }
    debug_assert_eq!(out.len(), width * height * 4);
    debug_assert_eq!(inp.len(), width * height * 4);
    let h = height as i64;
    let w = width as i64;
    let m = mult as i64;
    // One-row scratch buffer (the C uses a per-thread row out of `tempbuf`).
    let mut temp = vec![0.0f32; width * 4];
    for row in 0..height {
        // Cache-friendly visiting order; output rows are independent, so the
        // order does not affect the result.
        let i = crate::math::dwt_interleave_rows(row, height, mult) as i64;
        let r0 = (i - 2 * m).clamp(0, h - 1) as usize;
        let r1 = (i - m).clamp(0, h - 1) as usize;
        let r2 = i as usize;
        let r3 = (i + m).clamp(0, h - 1) as usize;
        let r4 = (i + 2 * m).clamp(0, h - 1) as usize;
        // Vertical pass into the row scratch buffer.
        for j in 0..width {
            let b0 = (r0 * width + j) * 4;
            let b1 = (r1 * width + j) * 4;
            let b2 = (r2 * width + j) * 4;
            let b3 = (r3 * width + j) * 4;
            let b4 = (r4 * width + j) * 4;
            for c in 0..4 {
                temp[j * 4 + c] = maybe_clip(
                    tap5(
                        &FILTER, inp[b0 + c], inp[b1 + c], inp[b2 + c], inp[b3 + c], inp[b4 + c],
                    ),
                    clip_negatives,
                );
            }
        }
        // Horizontal pass into the output row.
        let dst_row = (i as usize) * width;
        for j in 0..width {
            let jj = j as i64;
            let c0 = (jj - 2 * m).clamp(0, w - 1) as usize;
            let c1 = (jj - m).clamp(0, w - 1) as usize;
            let c3 = (jj + m).clamp(0, w - 1) as usize;
            let c4 = (jj + 2 * m).clamp(0, w - 1) as usize;
            let dst = (dst_row + j) * 4;
            for c in 0..4 {
                out[dst + c] = maybe_clip(
                    tap5(
                        &FILTER,
                        temp[c0 * 4 + c],
                        temp[c1 * 4 + c],
                        temp[j * 4 + c],
                        temp[c3 * 4 + c],
                        temp[c4 * 4 + c],
                    ),
                    clip_negatives,
                );
            }
        }
    }
}

/// Structurally divergent reference for [`blur_2d_bspline_kernel`].
///
/// Computes the same clamped two-pass blur but through a full-image scratch
/// buffer (vertical pass over every row first, then the horizontal pass),
/// visiting rows in natural order with flat per-float inner loops and
/// closure-based `i64` index math — instead of the kernel's fused row
/// scratch, interleaved row order, and pixel/channel loop nest with inline
/// `usize` math. The per-pixel scalar expressions keep the C association
/// order (that order is load-bearing for bit-exact agreement, so it is
/// shared on purpose); everything around it differs.
#[cfg(test)]
fn blur_2d_bspline_reference(
    out: &mut [f32],
    inp: &[f32],
    width: usize,
    height: usize,
    mult: usize,
    clip_negatives: bool,
) {
    if width == 0 || height == 0 || mult == 0 {
        return;
    }
    debug_assert_eq!(out.len(), width * height * 4);
    debug_assert_eq!(inp.len(), width * height * 4);
    let h = height as i64;
    let w = width as i64;
    let m = mult as i64;
    let clamp_row = |r: i64| r.clamp(0, h - 1) as usize;
    let clamp_col = |c: i64| c.clamp(0, w - 1) as usize;
    // Vertical pass over the whole image into a full scratch copy.
    let mut tmp = vec![0.0f32; width * height * 4];
    for i in 0..height {
        let ii = i as i64;
        let rows = [
            clamp_row(ii - 2 * m),
            clamp_row(ii - m),
            i,
            clamp_row(ii + m),
            clamp_row(ii + 2 * m),
        ];
        for k in 0..width * 4 {
            let col = k / 4;
            let ch = k % 4;
            let at = |r: usize| inp[(r * width + col) * 4 + ch];
            tmp[i * width * 4 + k] = maybe_clip(
                tap5(&FILTER, at(rows[0]), at(rows[1]), at(rows[2]), at(rows[3]), at(rows[4])),
                clip_negatives,
            );
        }
    }
    // Horizontal pass in natural row order.
    for i in 0..height {
        for k in 0..width * 4 {
            let col = (k / 4) as i64;
            let ch = k % 4;
            let cols = [
                clamp_col(col - 2 * m),
                clamp_col(col - m),
                col as usize,
                clamp_col(col + m),
                clamp_col(col + 2 * m),
            ];
            let at = |c: usize| tmp[(i * width + c) * 4 + ch];
            out[i * width * 4 + k] = maybe_clip(
                tap5(&FILTER, at(cols[0]), at(cols[1]), at(cols[2]), at(cols[3]), at(cols[4])),
                clip_negatives,
            );
        }
    }
}

/// Separable 5-tap B-spline &agrave;-trous blur over packed-RGBA floats.
///
/// Replaces the loop body of `blur_2D_Bspline()` in `src/common/bspline.h`
/// (non-nontemporal path only). `in_buf`/`out_buf` each hold
/// `4 * width * height` floats and must not overlap; `mult >= 1` is the
/// &agrave;-trous tap stride and `clip_negatives != 0` enables the
/// `MAX(0, ...)` clamp. Null pointers, zero dims, `mult <= 0`, and
/// overflowing dim products are guarded no-ops; the stated buffer lengths
/// remain a caller contract.
///
/// Named `darkroom_blur_2d_bspline` — distinct from the unrelated
/// single-channel `darkroom_blurs_bspline_2d` (blurs.c's own copy).
///
/// # Safety
/// `in_buf`/`out_buf` must be valid for `4 * width * height` floats each.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blur_2d_bspline(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    mult: i32,
    clip_negatives: i32,
) {
    if in_buf.is_null() || out_buf.is_null() {
        return;
    }
    if width == 0 || height == 0 {
        return;
    }
    if mult <= 0 {
        return;
    }
    let n = match width.checked_mul(height).and_then(|p| p.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    let inp = std::slice::from_raw_parts(in_buf, n);
    let out = std::slice::from_raw_parts_mut(out_buf, n);
    blur_2d_bspline_kernel(out, inp, width, height, mult as usize, clip_negatives != 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random fill with negatives and a periodic NaN.
    fn test_image(width: usize, height: usize) -> Vec<f32> {
        (0..width * height * 4)
            .map(|k| {
                if k % 37 == 13 {
                    f32::NAN
                } else {
                    let u = ((k as u64).wrapping_mul(2654435761) % 1000) as f32 / 1000.0;
                    (u - 0.5) * 6.0
                }
            })
            .collect()
    }

    fn run_kernel(inp: &[f32], width: usize, height: usize, mult: usize, clip: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; inp.len()];
        blur_2d_bspline_kernel(&mut out, inp, width, height, mult, clip);
        out
    }

    fn run_reference(inp: &[f32], width: usize, height: usize, mult: usize, clip: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; inp.len()];
        blur_2d_bspline_reference(&mut out, inp, width, height, mult, clip);
        out
    }

    fn assert_bits_eq(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "k={k} kernel={x} ref={y}");
        }
    }

    #[test]
    fn golden_single_row_impulse_mult1() {
        // 5x1 image, impulse 1.0 at pixel 2: the height-1 vertical pass is an
        // identity (taps sum to 1), so the output is the hand-computed 5-tap
        // [1 4 6 4 1] / 16 spread, exact in f32.
        let (w, h) = (5usize, 1usize);
        let mut inp = vec![0.0f32; w * h * 4];
        for c in 0..4 {
            inp[2 * 4 + c] = 1.0;
        }
        let out = run_kernel(&inp, w, h, 1, false);
        let taps = [1.0f32, 4.0, 6.0, 4.0, 1.0];
        for j in 0..w {
            let expected = taps[j] / 16.0;
            for c in 0..4 {
                assert_eq!(out[j * 4 + c].to_bits(), expected.to_bits(), "j={j} c={c}");
            }
        }
    }

    #[test]
    fn golden_single_row_impulse_mult2() {
        // 9x1 image, impulse 1.0 at pixel 4, stride 2: taps land at pixels
        // 0, 2, 4, 6, 8 with weights [1 4 6 4 1] / 16, zeros elsewhere.
        let (w, h) = (9usize, 1usize);
        let mut inp = vec![0.0f32; w * h * 4];
        for c in 0..4 {
            inp[4 * 4 + c] = 1.0;
        }
        let out = run_kernel(&inp, w, h, 2, false);
        let expected = [1.0f32, 0.0, 4.0, 0.0, 6.0, 0.0, 4.0, 0.0, 1.0];
        for j in 0..w {
            let want = expected[j] / 16.0;
            for c in 0..4 {
                assert_eq!(out[j * 4 + c].to_bits(), want.to_bits(), "j={j} c={c}");
            }
        }
    }

    #[test]
    fn golden_center_impulse_3x3() {
        // 3x3 image, impulse 1.0 at the center pixel: edge clamping only
        // replicates zero pixels, so the output is the exact outer product
        // of [1 4 6 4 1] / 16 with itself (all values k / 256, exact in f32).
        let (w, h) = (3usize, 3usize);
        let mut inp = vec![0.0f32; w * h * 4];
        for c in 0..4 {
            inp[(1 * w + 1) * 4 + c] = 1.0;
        }
        let out = run_kernel(&inp, w, h, 1, false);
        let taps = [1.0f32, 4.0, 6.0, 4.0, 1.0];
        // impulse at (1,1) reads tap index (3 - di, 3 - dj) for output (di,dj):
        // row 0 sees the impulse through F3, row 1 through F2, row 2 through F1.
        let tap_index = [3usize, 2, 1];
        for i in 0..h {
            for j in 0..w {
                let want = taps[tap_index[i]] * taps[tap_index[j]] / 256.0;
                for c in 0..4 {
                    assert_eq!(
                        out[(i * w + j) * 4 + c].to_bits(),
                        want.to_bits(),
                        "i={i} j={j} c={c}"
                    );
                }
            }
        }
    }

    #[test]
    fn kernel_matches_reference_bits() {
        // Includes NaN pixels: kernel and reference must agree bit-exactly,
        // proving identical association order through both passes.
        let cases = [
            (1usize, 1usize, 1usize),
            (5, 1, 1),
            (1, 5, 2),
            (3, 3, 1),
            (4, 3, 2),
            (7, 5, 3),
            (8, 8, 5),
            (9, 2, 4),
        ];
        for (w, h, mult) in cases {
            let inp = test_image(w, h);
            for clip in [false, true] {
                let k = run_kernel(&inp, w, h, mult, clip);
                let r = run_reference(&inp, w, h, mult, clip);
                assert_bits_eq(&k, &r);
            }
        }
    }

    #[test]
    fn clip_negatives_clamps_while_off_preserves() {
        // Constant -2.0 field: the blur of a constant is the constant itself
        // (taps sum to 1, exact here), so clip-off preserves -2.0 bit-exactly
        // and clip-on yields +0.0 bit-exactly.
        let (w, h) = (4usize, 4usize);
        let inp = vec![-2.0f32; w * h * 4];
        let off = run_kernel(&inp, w, h, 1, false);
        let on = run_kernel(&inp, w, h, 1, true);
        for k in 0..w * h * 4 {
            assert_eq!(off[k].to_bits(), (-2.0f32).to_bits(), "k={k}");
            assert_eq!(on[k].to_bits(), 0.0f32.to_bits(), "k={k}");
        }
    }

    #[test]
    fn clip_matches_c_max_nan_and_negzero_semantics() {
        // glib MAX(0.0f, v): NaN propagates (unlike f32::max) and -0.0 is
        // preserved (0.0 > -0.0 is false, so the second operand wins).
        assert!(maybe_clip(f32::NAN, true).is_nan());
        assert_eq!(maybe_clip(-0.0, true).to_bits(), (-0.0f32).to_bits());
        assert_eq!(maybe_clip(-1.5, true).to_bits(), 0.0f32.to_bits());
        assert_eq!(maybe_clip(2.5, true).to_bits(), 2.5f32.to_bits());
        assert!(maybe_clip(f32::NAN, false).is_nan());
        assert_eq!(maybe_clip(-1.5, false).to_bits(), (-1.5f32).to_bits());
    }

    #[test]
    fn nan_blur_center_is_nan_and_matches_reference() {
        // Single NaN in a 3x3 zero field, clip on: the center output's tap
        // window covers the NaN, and MAX(0, NaN) is NaN in the C semantics.
        let (w, h) = (3usize, 3usize);
        let mut inp = vec![0.0f32; w * h * 4];
        inp[(1 * w + 1) * 4] = f32::NAN;
        let k = run_kernel(&inp, w, h, 1, true);
        let r = run_reference(&inp, w, h, 1, true);
        assert!(k[(1 * w + 1) * 4].is_nan());
        assert_bits_eq(&k, &r);
    }

    #[test]
    fn constant_field_is_near_identity() {
        // Flat positive field passes through (taps sum to 1) up to rounding.
        for (w, h, mult) in [(6usize, 5usize, 1usize), (6, 5, 2), (7, 7, 3)] {
            let inp = vec![0.3f32; w * h * 4];
            for clip in [false, true] {
                let out = run_kernel(&inp, w, h, mult, clip);
                for (k, v) in out.iter().enumerate() {
                    assert!((v - 0.3).abs() < 1e-6, "k={k} v={v}");
                }
            }
        }
    }

    #[test]
    fn ffi_matches_kernel() {
        let (w, h) = (7usize, 5usize);
        let inp = test_image(w, h);
        let mut via_ffi = vec![0.0f32; inp.len()];
        unsafe {
            darkroom_blur_2d_bspline(inp.as_ptr(), via_ffi.as_mut_ptr(), w, h, 2, 1);
        }
        let expected = run_kernel(&inp, w, h, 2, true);
        assert_bits_eq(&via_ffi, &expected);
    }

    #[test]
    fn ffi_guards_null_dims_and_mult() {
        let inp = vec![1.0f32; 16];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // Null pointers: must not crash.
            darkroom_blur_2d_bspline(std::ptr::null(), out.as_mut_ptr(), 2, 2, 1, 0);
            darkroom_blur_2d_bspline(inp.as_ptr(), std::ptr::null_mut(), 2, 2, 1, 0);
            darkroom_blur_2d_bspline(std::ptr::null(), std::ptr::null_mut(), 2, 2, 1, 0);
            // Degenerate dims and mult: no-op, output untouched.
            darkroom_blur_2d_bspline(inp.as_ptr(), out.as_mut_ptr(), 0, 2, 1, 0);
            darkroom_blur_2d_bspline(inp.as_ptr(), out.as_mut_ptr(), 2, 0, 1, 0);
            darkroom_blur_2d_bspline(inp.as_ptr(), out.as_mut_ptr(), 2, 2, 0, 0);
            darkroom_blur_2d_bspline(inp.as_ptr(), out.as_mut_ptr(), 2, 2, -3, 0);
            // Overflowing dim product: no-op without touching memory.
            darkroom_blur_2d_bspline(inp.as_ptr(), out.as_mut_ptr(), usize::MAX, 2, 1, 0);
            darkroom_blur_2d_bspline(inp.as_ptr(), out.as_mut_ptr(), usize::MAX, usize::MAX, 1, 0);
        }
        assert!(out.iter().all(|&v| v == 7.0));
    }
}
