//! Edge-avoiding à-trous wavelets (eaw.c) — the decompose/synthesize kernels
//! the denoise (profiled) wavelets path drives.
//!
//! Port of `eaw_dn_decompose` + `eaw_synthesize` (+ the `dn_weight` and
//! `fast_mexp2f` helpers from `denoiseprofile.c:226` / `math.h:452`). Deviations
//! from the C, all deliberate:
//!
//! - **Natural row order.** The C visits rows via `dwt_interleave_rows`
//!   (cache-conflict reduction only); we visit rows 0..height in order. Only
//!   the float summation order of the per-scale `sum_y2` statistic changes (a
//!   threshold *input*), never pixel values.
//! - **Clamped indexing everywhere** instead of the C's three-phase
//!   boundary/interior split. The interior fast path exists in C purely for
//!   vectorisation; the clamped form computes identical pixels — the C's own
//!   edge phases clamp exactly like this (`CLAMP(y,0,height-1)`; left phase
//!   `if(x<0) x=0`, right phase full clamp, interior phase unguarded because
//!   the indices are provably inside).
//!
//! The 5×5 B-spline filter is the outer product of [1,4,6,4,1]/16 with itself,
//! applied at stride `2^scale` ("à trous").
//!
//! One more corner: on images narrower than 4·stride the C's left-edge phase
//! (which only guards `x < 0`) walks past the row end and silently reads into
//! the *next* row (in-allocation, so no crash — but garbage taps). Our unified
//! clamp refuses to bleed across rows; pixel values differ from the C only on
//! those degenerate sizes, and stay sane there.

/// Bit-exact port of `fast_mexp2f` (math.h:452) — the *float* variant used by
/// eaw/denoise (not `dt_fast_memp2f`, which does its arithmetic in int space).
/// Fast approximation of `exp2(-x)` for 0<x<126 via float→int bit punning:
/// builds the float whose bit pattern is `0x3f800000 + x·(0x3f000000−0x3f800000)`
/// clamped at zero, which reads back as an exponential decay.
#[inline]
pub fn fast_mexp2f(x: f32) -> f32 {
    const I1: f32 = 0x3f800000u32 as f32; // 2^0
    const I2: f32 = 0x3f000000u32 as f32; // 2^-1
    let k0 = I1 + x * (I2 - I1);
    // k.i = k0 >= 0x800000 ? k0 : 0, punned straight back to float.
    let bits = if k0 >= 0x800000u32 as f32 { k0 } else { 0.0 };
    f32::from_bits(bits as u32)
}

/// Edge-avoiding weight between two RGBA pixels (`dn_weight`, eaw.c:226):
/// `fast_mexp2f(max(0, |c1-c2|²·inv_sigma2·0.02 − 9))`; the 9 = (3σ)² offset
/// makes weights saturate near 1 for colours within ~3σ of each other.
#[inline]
pub fn dn_weight(px: &[f32], px2: &[f32], inv_sigma2: f32) -> f32 {
    let d0 = px[0] - px2[0];
    let d1 = px[1] - px2[1];
    let d2 = px[2] - px2[2];
    let dot = (d0 * d0 + d1 * d1 + d2 * d2) * inv_sigma2;
    const VAR: f32 = 0.02; // FIXME carried from C: should depend on pre-VST noise!
    const OFF2: f32 = 9.0; // (3 sigma)^2
    fast_mexp2f((dot * VAR - OFF2).max(0.0))
}

/// Filter taps, row-major [jj][ii]: outer product of [1,4,6,4,1]/256 with
/// itself — identical table to `eaw_dn_decompose`'s local `filter[25]`.
const FILTER: [f32; 25] = [
    1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0, //
    4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0, //
    6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0, //
    4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0, //
    1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
];

/// One à-trous decompose step (eaw.c `eaw_dn_decompose`): B-spline smooth of
/// `input` at stride `2^scale` into `out_coarse`, detail = input − coarse into
/// `detail`. Returns the accumulated squared detail per channel (`sum_y2`,
/// summed over every pixel) that the caller feeds into the BayesShrink
/// threshold; `inv_sigma2` is the edge-stopping sharpness (1/σ_band²).
///
/// Buffers are packed RGBA f32 of `width*height*4`; all three must be distinct
/// (the kernel reads `input` while writing the other two).
pub fn dn_decompose(
    out_coarse: &mut [f32],
    input: &[f32],
    detail: &mut [f32],
    scale: u32,
    inv_sigma2: f32,
    width: usize,
    height: usize,
) -> [f32; 4] {
    let mult = 1u64 << scale;
    let mut sum_sq = [0.0f32; 4];
    for rowid in 0..height {
        // dwt_interleave_rows(rowid, height, mult) in the C reorders rows for
        // cache friendliness only; natural order here (see module doc).
        let j = rowid;
        let row = &input[j * width * 4..][..width * 4];
        for i in 0..width {
            let mut sum = [0.0f32; 4];
            let mut wgt = [0.0f32; 4];
            let mut fi = 0usize;
            for jj in -2i64..=2 {
                let y = (j as i64 + mult as i64 * jj).clamp(0, height as i64 - 1) as usize;
                for ii in -2i64..=2 {
                    let x = (i as i64 + mult as i64 * ii).clamp(0, width as i64 - 1) as usize;
                    let px2 = &input[y * width * 4 + x * 4..][..4];
                    let f = FILTER[fi];
                    fi += 1;
                    // Edge weights are measured against the center pixel,
                    // like the C's per-pixel `px` (not the row start).
                    let center = &row[i * 4..][..4];
                    let w = f * dn_weight(center, px2, inv_sigma2);
                    for c in 0..4 {
                        wgt[c] += w;
                        sum[c] += w * px2[c];
                    }
                }
            }
            let o = j * width * 4 + i * 4;
            let mut det = [0.0f32; 4];
            for c in 0..4 {
                sum[c] /= wgt[c];
                out_coarse[o + c] = sum[c];
                det[c] = row[i * 4 + c] - sum[c];
                detail[o + c] = det[c];
                sum_sq[c] += det[c] * det[c];
            }
        }
    }
    sum_sq
}

/// Soft-threshold accumulate (eaw.c `accumulate` + `eaw_synthesize`): adds
/// `boost · soft(detail, thresh)` into `accum`, elementwise over the whole
/// packed buffer. Soft thresholding: `amount = max(detail−thresh, 0) +
/// min(detail+thresh, 0)` — shrinks detail magnitudes below `thresh` toward
/// zero without shifting their sign.
///
/// `accum` may alias `detail`? No — distinct buffers (the C calls it both ways,
/// but our driver always passes distinct slices; aliasing would also be fine
/// since each index is read once before written).
pub fn synthesize(
    accum: &mut [f32],
    detail: &[f32],
    threshold: &[f32; 4],
    boost: &[f32; 4],
    npixels: usize,
) {
    for k in 0..npixels {
        let o = k * 4;
        for c in 0..4 {
            let d = detail[o + c];
            let amount =
                f32::max(d - threshold[c], 0.0) + f32::min(d + threshold[c], 0.0);
            accum[o + c] += boost[c] * amount;
        }
    }
}

/// FFI wrapper for C `eaw_synthesize` (eaw.c:207).
///
/// Replaces only the `DT_OMP_FOR` loop body: reads `threshold`/`boost` as
/// 4-float vectors and accumulates `boost · soft(detail, thresh)` into `out`
/// via [`synthesize`] — identical soft-threshold semantics, no duplicated math.
///
/// `in_buf` is unused (the C body never reads it either) and kept solely for
/// `eaw_synthesize_t` function-pointer compatibility; it may even alias `out`
/// (the denoiseprofile caller passes `out, out, ...`) or be NULL. The stated
/// buffer lengths (`width·height·4` floats for `out`/`detail`, 4 floats for
/// `threshold`/`boost`) remain a caller contract; null pointers, non-positive
/// dims, and overflowing dim products are guarded no-ops.
///
/// # Safety
/// `out`/`detail` must each point to `width·height·4` valid floats (they must
/// not overlap each other; either may alias `in_buf`), `threshold`/`boost` to
/// 4 valid floats each, with `width > 0` and `height > 0`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_eaw_synthesize(
    out: *mut f32,
    _in_buf: *const f32,
    detail: *const f32,
    threshold: *const f32,
    boost: *const f32,
    width: i32,
    height: i32,
) {
    if out.is_null() || detail.is_null() || threshold.is_null() || boost.is_null() {
        return;
    }
    if width <= 0 || height <= 0 {
        return;
    }
    let (w, h) = (width as usize, height as usize);
    let npixels = match w.checked_mul(h) {
        Some(n) => n,
        None => return,
    };
    let len = match npixels.checked_mul(4) {
        Some(n) => n,
        None => return,
    };
    let accum = std::slice::from_raw_parts_mut(out, len);
    let det = std::slice::from_raw_parts(detail, len);
    let thresh = std::slice::from_raw_parts(threshold, 4);
    let boostv = std::slice::from_raw_parts(boost, 4);
    let t = [thresh[0], thresh[1], thresh[2], thresh[3]];
    let b = [boostv[0], boostv[1], boostv[2], boostv[3]];
    synthesize(accum, det, &t, &b, npixels);
}

/// FFI wrapper for C `eaw_dn_decompose` (eaw.c:269).
///
/// Replaces only the `DT_OMP_FOR` loop body: runs the à-trous B-spline smooth
/// at stride `2^scale` with edge-avoiding `dn_weight` taps via [`dn_decompose`]
/// — same kernel, no duplication — and writes the returned per-channel sum of
/// squared details back through `sum_squared`. Pixels match the C in range;
/// the `sum_sq` statistic follows natural row order (the C summed it in OpenMP
/// reduction order), and degenerate narrow images use unified clamping instead
/// of the C's row-bleeding left-edge phase.
///
/// Buffers are packed RGBA f32 of `width·height·4`; `out`/`in_buf`/`detail`
/// must be non-null, with `out`/`detail` distinct from each other and from
/// `in_buf` (the kernel reads `in_buf` while writing the other two).
/// `sum_squared` must point to 4 writable floats. Null pointers,
/// non-positive dims, out-of-range `scale`, and overflowing dim products are
/// guarded no-ops (buffers and `sum_squared` left untouched); the stated
/// buffer lengths remain a caller contract.
///
/// # Safety
/// `out`/`detail` must each point to `width·height·4` valid floats (distinct
/// from each other and from `in_buf`), `in_buf` to `width·height·4` valid
/// floats, `sum_squared` to 4 valid floats, with `width > 0`, `height > 0`
/// and `0 <= scale < 32`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_eaw_dn_decompose(
    out: *mut f32,
    in_buf: *const f32,
    detail: *mut f32,
    sum_squared: *mut f32,
    scale: i32,
    inv_sigma2: f32,
    width: i32,
    height: i32,
) {
    if out.is_null() || in_buf.is_null() || detail.is_null() || sum_squared.is_null() {
        return;
    }
    if width <= 0 || height <= 0 {
        return;
    }
    // The C computes `1u << scale` (32-bit); negative or huge scales are
    // undefined there and would panic the `1u64 << scale` shift here.
    if !(0..32).contains(&scale) {
        return;
    }
    let (w, h) = (width as usize, height as usize);
    let npixels = match w.checked_mul(h) {
        Some(n) => n,
        None => return,
    };
    let len = match npixels.checked_mul(4) {
        Some(n) => n,
        None => return,
    };
    // Slice lengths must also fit isize (a from_raw_parts requirement):
    // i32 dims can still produce products past that on 64-bit (e.g.
    // i32::MAX²·4), where checked_mul above stays Some.
    if len > isize::MAX as usize {
        return;
    };
    let coarse = std::slice::from_raw_parts_mut(out, len);
    let input = std::slice::from_raw_parts(in_buf, len);
    let det = std::slice::from_raw_parts_mut(detail, len);
    let sum_sq = dn_decompose(coarse, input, det, scale as u32, inv_sigma2, w, h);
    let sums = std::slice::from_raw_parts_mut(sum_squared, 4);
    sums.copy_from_slice(&sum_sq);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_mexp2f_tracks_exp2_and_is_monotone() {
        // The punning trick is exact at integer arguments (it lands the float
        // bit pattern of an exact power of two).
        for k in 0..10 {
            let got = fast_mexp2f(k as f32);
            let want = 2.0f32.powf(-(k as f32));
            assert_eq!(got, want, "fast_mexp2f({k}) must be exactly 2^-{k}");
        }
        // Between integers it linearly ramps the *value* across each octave
        // (uniform mantissa ⇒ chord interpolation), so the worst case vs the
        // true exponential sits at the octave midpoint: 0.75 vs √½ ≈ +6.07%.
        // Pin that envelope.
        for s in 1..100 {
            let x = s as f32 / 10.0;
            let got = fast_mexp2f(x);
            let want = 2.0f32.powf(-x);
            let rel = (got - want).abs() / want;
            assert!(
                rel < 0.065,
                "fast_mexp2f({x}) = {got}, want ~{want} (rel {rel})"
            );
        }
        // Monotonically non-increasing, saturates at 0.
        let mut prev = f32::INFINITY;
        for s in 0..200 {
            let v = fast_mexp2f(s as f32 * 0.25);
            assert!(v <= prev, "not monotone at {s}: {v} > {prev}");
            prev = v;
        }
        assert_eq!(fast_mexp2f(130.0), 0.0, "beyond the exponent range → 0");
    }

    #[test]
    fn decompose_flat_field_has_zero_detail_and_identity_coarse() {
        // A flat field has no edges and no detail: every tap sees the same
        // value, so coarse == input and detail == 0 — up to f32 rounding of
        // the 25-tap weighted mean (identical in the C).
        let (w, h) = (24usize, 16usize);
        let input = vec![0.37f32; w * h * 4];
        let mut coarse = vec![0.0f32; w * h * 4];
        let mut detail = vec![1.0f32; w * h * 4];
        let sum_sq = dn_decompose(&mut coarse, &input, &mut detail, 0, 1.0, w, h);
        for k in 0..input.len() {
            assert!(
                (coarse[k] - input[k]).abs() < 1e-6,
                "flat field: coarse must equal input at {k}: {} vs {}",
                coarse[k],
                input[k]
            );
            assert!(
                detail[k].abs() < 1e-6,
                "flat field has (near-)zero detail at {k}: {}",
                detail[k]
            );
        }
        for c in 0..4 {
            assert!(sum_sq[c] < 1e-9, "sum_sq[{c}] = {}", sum_sq[c]);
        }
    }

    #[test]
    fn decompose_telescopes_back_to_input() {
        // By construction detail_s = input_s − coarse(input_s) at every scale,
        // so input = Σ_s detail_s + final coarse, up to float rounding.
        let (w, h) = (32usize, 32usize);
        let mut input: Vec<f32> = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let v =
                    0.3 + 0.02 * ((x as f32) / 7.0).sin() + 0.05 * ((y as f32) / 5.0).cos();
                input.extend_from_slice(&[v, v * 1.01, v * 0.98, 1.0]);
            }
        }
        let scales = [0u32, 1, 2];
        // Accumulate Σ detail_s; keep coarse as the next scale's input.
        let mut total = vec![0.0f32; w * h * 4]; // ends up holding final coarse
        let mut detail = vec![0.0f32; w * h * 4];
        let mut c2 = vec![0.0f32; w * h * 4];
        let mut run = input.clone();
        for &s in &scales {
            dn_decompose(&mut c2, &run, &mut detail, s, 1.0, w, h);
            for k in 0..total.len() {
                total[k] += detail[k];
            }
            std::mem::swap(&mut run, &mut c2);
        }
        // total = final coarse (in `run` after the last swap) + Σ details.
        for k in 0..total.len() {
            total[k] += run[k];
        }
        for k in 0..input.len() {
            assert!(
                (total[k] - input[k]).abs() < 1e-4,
                "telescoping broke at {k}: {} vs {}",
                total[k],
                input[k]
            );
        }
    }

    #[test]
    fn dn_decompose_weights_use_center_pixel() {
        // Regression oracle for the m4-185 review: edge weights must be
        // measured against the center pixel, not the row start. A lone bright
        // pixel under strong edge-stopping keeps its own value (dark taps get
        // ~zero weight); measuring against pixel (j, 0) instead would drag it
        // toward dark.
        let (w, h) = (5usize, 1usize);
        let mut input = vec![0.0f32; w * h * 4];
        input[(w - 1) * 4] = 100.0;
        input[(w - 1) * 4 + 1] = 100.0;
        input[(w - 1) * 4 + 2] = 100.0;
        let mut coarse = vec![0.0f32; w * h * 4];
        let mut detail = vec![0.0f32; w * h * 4];
        dn_decompose(&mut coarse, &input, &mut detail, 0, 1e6, w, h);
        assert!(
            coarse[(w - 1) * 4] > 50.0,
            "bright pixel lost its value: coarse={coarse:?}"
        );
    }

    #[test]
    fn synthesize_soft_threshold_matches_c_semantics() {
        // amount = max(d−t,0)+min(d+t,0): zero threshold passes detail through,
        // a threshold above |d| annihilates it, and small thresholds shrink
        // toward zero without flipping sign.
        let npixels = 3;
        let detail = [1.0f32, -1.0, 0.25, 0.0, -0.5, 0.9, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let boost = [1.0f32; 4];

        let mut accum = vec![0.5f32; npixels * 4];
        synthesize(&mut accum, &detail, &[0.0; 4], &boost, npixels);
        for c in 0..detail.len() {
            assert!((accum[c] - (0.5 + detail[c])).abs() < 1e-6);
        }

        let mut accum = vec![0.5f32; npixels * 4];
        synthesize(&mut accum, &detail, &[10.0; 4], &boost, npixels);
        assert!(accum.iter().all(|&a| a == 0.5), "huge threshold kills all detail");

        let mut accum = vec![0.0f32; npixels * 4];
        synthesize(&mut accum, &detail, &[0.5; 4], &boost, npixels);
        // d=1 → 0.5; d=−1 → −0.5; d=0.25 → 0; d=−0.5 → 0; d=0.9 → 0.4;
        // d=−2 → −1.5; ...
        let expect = [
            0.5f32, -0.5, 0.0, 0.0, 0.0, 0.4, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0,
        ];
        for (got, want) in accum.iter().zip(expect.iter()) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }

        // Boost scales the surviving amount.
        let mut accum = vec![0.0f32; npixels * 4];
        synthesize(&mut accum, &detail, &[0.5; 4], &[2.0; 4], npixels);
        assert!((accum[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ffi_synthesize_agrees_with_safe_kernel() {
        // Round-trip: the FFI export must produce bit-identical output to a
        // direct `synthesize` call on non-trivial threshold/boost/detail.
        let (w, h) = (7i32, 5i32);
        let npixels = (w as usize) * (h as usize);
        let mut detail = vec![0.0f32; npixels * 4];
        for (k, d) in detail.iter_mut().enumerate() {
            // Deterministic pseudo-signal spanning both sides of the threshold.
            *d = (((k as u32).wrapping_mul(2654435761u32) >> 8) % 2000) as f32 / 1000.0 - 1.0;
        }
        let threshold = [0.15f32, 0.3, 0.05, 0.5];
        let boost = [1.0f32, 2.0, 0.5, 1.5];
        let mut want = vec![0.25f32; npixels * 4];
        synthesize(&mut want, &detail, &threshold, &boost, npixels);
        let mut got = vec![0.25f32; npixels * 4];
        unsafe {
            darkroom_eaw_synthesize(
                got.as_mut_ptr(),
                got.as_ptr(), // in_buf aliases out: ignored, must be harmless
                detail.as_ptr(),
                threshold.as_ptr(),
                boost.as_ptr(),
                w,
                h,
            );
        }
        assert_eq!(got, want, "FFI must agree with synthesize bit-for-bit");
    }

    #[test]
    fn ffi_synthesize_threshold_and_boost_semantics() {
        // Zero threshold passes detail through scaled by boost; huge threshold
        // annihilates; per-channel boost scales survivors independently.
        let (w, h) = (2i32, 1i32);
        let detail = [1.0f32, -1.0, 0.25, -0.25, 0.6, -0.6, 2.0, -2.0];
        let zero = [0.0f32; 4];
        let one = [1.0f32; 4];

        let mut out = vec![0.0f32; 8];
        unsafe {
            darkroom_eaw_synthesize(
                out.as_mut_ptr(),
                std::ptr::null(),
                detail.as_ptr(),
                zero.as_ptr(),
                one.as_ptr(),
                w,
                h,
            );
        }
        for k in 0..8 {
            assert!((out[k] - detail[k]).abs() < 1e-6, "passthrough at {k}");
        }

        let huge = [10.0f32; 4];
        let mut out = vec![0.5f32; 8];
        unsafe {
            darkroom_eaw_synthesize(
                out.as_mut_ptr(),
                std::ptr::null(),
                detail.as_ptr(),
                huge.as_ptr(),
                one.as_ptr(),
                w,
                h,
            );
        }
        assert!(out.iter().all(|&a| a == 0.5), "huge threshold kills all detail");

        // Per-channel: thresh 0.5, boost [2,1,0,3].
        let thresh = [0.5f32; 4];
        let boost = [2.0f32, 1.0, 0.0, 3.0];
        let mut out = vec![0.0f32; 8];
        unsafe {
            darkroom_eaw_synthesize(
                out.as_mut_ptr(),
                std::ptr::null(),
                detail.as_ptr(),
                thresh.as_ptr(),
                boost.as_ptr(),
                w,
                h,
            );
        }
        // px0: d=1 -> 0.5*boost; -1 -> -0.5*boost; 0.25 -> 0; -0.25 -> 0.
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - -0.5).abs() < 1e-6);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
        // px1: 0.6 -> 0.1*2 = 0.2; -0.6 -> -0.1; 2.0 -> 1.5*0 = 0; -2.0 -> -1.5*3.
        assert!((out[4] - 0.2).abs() < 1e-6);
        assert!((out[5] - -0.1).abs() < 1e-6);
        assert_eq!(out[6], 0.0);
        assert!((out[7] - -4.5).abs() < 1e-6);
    }

    #[test]
    fn ffi_synthesize_guards_are_noops() {
        // Null pointers, degenerate dims, and in==NULL must not crash and must
        // leave the output untouched. (Short-buffer overruns remain a caller
        // contract — the guards only cover null/dim/overflow.)
        let detail = [1.0f32; 16];
        let t = [0.0f32; 4];
        let b = [1.0f32; 4];

        // Null out.
        unsafe {
            darkroom_eaw_synthesize(
                std::ptr::null_mut(),
                std::ptr::null(),
                detail.as_ptr(),
                t.as_ptr(),
                b.as_ptr(),
                2,
                1,
            );
        }
        // Null detail / threshold / boost with a live out: untouched.
        let mut out = vec![0.5f32; 8];
        let before = out.clone();
        unsafe {
            darkroom_eaw_synthesize(out.as_mut_ptr(), std::ptr::null(), std::ptr::null(), t.as_ptr(), b.as_ptr(), 2, 1);
            darkroom_eaw_synthesize(out.as_mut_ptr(), std::ptr::null(), detail.as_ptr(), std::ptr::null(), b.as_ptr(), 2, 1);
            darkroom_eaw_synthesize(out.as_mut_ptr(), std::ptr::null(), detail.as_ptr(), t.as_ptr(), std::ptr::null(), 2, 1);
        }
        assert_eq!(out, before);

        // Degenerate dims: zero / negative width or height.
        for (w, h) in [(0, 1), (1, 0), (-3, 4), (4, -2)] {
            unsafe {
                darkroom_eaw_synthesize(
                    out.as_mut_ptr(),
                    std::ptr::null(),
                    detail.as_ptr(),
                    t.as_ptr(),
                    b.as_ptr(),
                    w,
                    h,
                );
            }
        }
        assert_eq!(out, before, "degenerate dims must be no-ops");

        // NULL in_buf is explicitly allowed (parameter is unused).
        let mut out = vec![0.0f32; 8];
        unsafe {
            darkroom_eaw_synthesize(
                out.as_mut_ptr(),
                std::ptr::null(),
                detail.as_ptr(),
                t.as_ptr(),
                b.as_ptr(),
                2,
                1,
            );
        }
        assert_eq!(out, vec![1.0f32; 8], "NULL in_buf must not affect the result");
    }

    fn dn_test_image(w: usize, h: usize) -> Vec<f32> {
        // Deterministic pseudo-signal with edges in every channel plus a
        // hard vertical step, so both the smooth taps and the edge-avoiding
        // weights are exercised.
        let mut v = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let s = 0.3 + 0.02 * ((x as f32) / 7.0).sin() + 0.05 * ((y as f32) / 5.0).cos();
                let step = if x * 2 >= w { 0.4 } else { 0.0 };
                v.extend_from_slice(&[s + step, s * 1.01 + step, s * 0.98, 1.0]);
            }
        }
        v
    }

    #[test]
    fn ffi_dn_decompose_agrees_with_safe_kernel() {
        // Bit-for-bit agreement with `dn_decompose` on coarse, detail and the
        // sum-of-squared-details write-back, across the scales the denoise
        // caller actually uses (0..7) plus a mid inv_sigma2.
        let (w, h) = (24usize, 18usize);
        let input = dn_test_image(w, h);
        for scale in 0..7 {
            let mut want_coarse = vec![0.0f32; w * h * 4];
            let mut want_detail = vec![0.0f32; w * h * 4];
            let want_sum =
                dn_decompose(&mut want_coarse, &input, &mut want_detail, scale, 1.0, w, h);
            let mut got_coarse = vec![0.0f32; w * h * 4];
            let mut got_detail = vec![0.0f32; w * h * 4];
            let mut got_sum = [99.0f32; 4];
            unsafe {
                darkroom_eaw_dn_decompose(
                    got_coarse.as_mut_ptr(),
                    input.as_ptr(),
                    got_detail.as_mut_ptr(),
                    got_sum.as_mut_ptr(),
                    scale as i32,
                    1.0,
                    w as i32,
                    h as i32,
                );
            }
            assert_eq!(got_coarse, want_coarse, "coarse differs at scale {scale}");
            assert_eq!(got_detail, want_detail, "detail differs at scale {scale}");
            assert_eq!(got_sum, want_sum, "sum_squared differs at scale {scale}");
        }
    }

    #[test]
    fn ffi_dn_decompose_sum_matches_squared_detail() {
        // The write-back must equal the per-channel Σ detail² the caller feeds
        // into the BayesShrink threshold — recomputed independently here.
        let (w, h) = (16usize, 12usize);
        let input = dn_test_image(w, h);
        let mut coarse = vec![0.0f32; w * h * 4];
        let mut detail = vec![0.0f32; w * h * 4];
        let mut sum = [0.0f32; 4];
        unsafe {
            darkroom_eaw_dn_decompose(
                coarse.as_mut_ptr(),
                input.as_ptr(),
                detail.as_mut_ptr(),
                sum.as_mut_ptr(),
                1,
                2.5,
                w as i32,
                h as i32,
            );
        }
        // detail = input − coarse by construction; re-derive and accumulate.
        let mut expect = [0.0f32; 4];
        for k in 0..w * h {
            for c in 0..4 {
                let d = input[k * 4 + c] - coarse[k * 4 + c];
                assert!(
                    (detail[k * 4 + c] - d).abs() < 1e-6,
                    "detail != input-coarse at pixel {k} ch {c}"
                );
                expect[c] += detail[k * 4 + c] * detail[k * 4 + c];
            }
        }
        for c in 0..4 {
            assert!(
                (sum[c] - expect[c]).abs() / expect[c].max(1e-9) < 1e-5,
                "sum_squared[{c}] = {}, recomputed {}",
                sum[c],
                expect[c]
            );
        }
    }

    #[test]
    fn ffi_dn_decompose_edge_avoidance_threshold_behavior() {
        // A hard 0→1 vertical step at scale 0: with inv_sigma2 == 0 every tap
        // weighs fast_mexp2f(0) == 1, so the step is blurred across the edge;
        // with a huge inv_sigma2, off-edge taps weigh ~0 and the edge pixel
        // keeps (near-)its own value.
        let (w, h) = (16usize, 4usize);
        let mut input = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = if x >= w / 2 { 1.0 } else { 0.0 };
                let o = (y * w + x) * 4;
                input[o] = v;
                input[o + 1] = v;
                input[o + 2] = v;
                input[o + 3] = 1.0;
            }
        }
        let run = |inv: f32| {
            let mut coarse = vec![0.0f32; w * h * 4];
            let mut detail = vec![0.0f32; w * h * 4];
            let mut sum = [0.0f32; 4];
            unsafe {
                darkroom_eaw_dn_decompose(
                    coarse.as_mut_ptr(),
                    input.as_ptr(),
                    detail.as_mut_ptr(),
                    sum.as_mut_ptr(),
                    0,
                    inv,
                    w as i32,
                    h as i32,
                );
            }
            (coarse, detail, sum)
        };
        // Edge pixels: last dark column (x = w/2-1) and first bright column.
        let edge = |coarse: &[f32], x: usize| coarse[(2 * w + x) * 4];
        let (blurred, _, sum_blur) = run(0.0);
        let dark_blur = edge(&blurred, w / 2 - 1);
        let bright_blur = edge(&blurred, w / 2);
        assert!(
            dark_blur > 0.05 && dark_blur < 0.5,
            "inv_sigma2=0 must bleed bright into the dark edge pixel, got {dark_blur}"
        );
        assert!(
            bright_blur > 0.5 && bright_blur < 0.95,
            "inv_sigma2=0 must bleed dark into the bright edge pixel, got {bright_blur}"
        );
        let (sharp, _, sum_sharp) = run(1.0e12);
        let dark_sharp = edge(&sharp, w / 2 - 1);
        let bright_sharp = edge(&sharp, w / 2);
        assert!(
            dark_sharp.abs() < 1e-4,
            "huge inv_sigma2 must preserve the dark edge pixel, got {dark_sharp}"
        );
        assert!(
            (bright_sharp - 1.0).abs() < 1e-4,
            "huge inv_sigma2 must preserve the bright edge pixel, got {bright_sharp}"
        );
        // Interior pixels far from the step are unaffected by the weights in
        // both regimes, so both runs agree there.
        for x in [0usize, 2, w - 3, w - 1] {
            assert!(
                (edge(&blurred, x) - edge(&sharp, x)).abs() < 1e-5,
                "far-from-edge pixel {x} must agree across regimes"
            );
        }
        // Blurring across the edge leaves larger residuals than preserving it.
        assert!(
            sum_blur[0] > sum_sharp[0],
            "blurred sum {} must exceed edge-preserving sum {}",
            sum_blur[0],
            sum_sharp[0]
        );
    }

    #[test]
    fn ffi_dn_decompose_degenerate_sizes_stay_sane() {
        // 1×1, single-row and single-column images only ever read clamped
        // taps: outputs must be finite with detail == input − coarse, and the
        // write-back must equal Σ detail².
        for (w, h) in [(1usize, 1usize), (9, 1), (1, 7), (3, 2)] {
            let input = dn_test_image(w, h);
            for scale in [0i32, 1, 3] {
                let mut coarse = vec![f32::NAN; w * h * 4];
                let mut detail = vec![f32::NAN; w * h * 4];
                let mut sum = [f32::NAN; 4];
                unsafe {
                    darkroom_eaw_dn_decompose(
                        coarse.as_mut_ptr(),
                        input.as_ptr(),
                        detail.as_mut_ptr(),
                        sum.as_mut_ptr(),
                        scale,
                        1.0,
                        w as i32,
                        h as i32,
                    );
                }
                for k in 0..w * h * 4 {
                    assert!(
                        coarse[k].is_finite(),
                        "{w}x{h}s{scale}: coarse[{k}] not finite"
                    );
                    assert!(
                        detail[k].is_finite(),
                        "{w}x{h}s{scale}: detail[{k}] not finite"
                    );
                }
                let mut expect = [0.0f32; 4];
                for k in 0..w * h {
                    for c in 0..4 {
                        assert!(
                            (detail[k * 4 + c] - (input[k * 4 + c] - coarse[k * 4 + c])).abs()
                                < 1e-5,
                            "{w}x{h}s{scale}: detail != input-coarse at {k}:{c}"
                        );
                        expect[c] += detail[k * 4 + c] * detail[k * 4 + c];
                    }
                }
                for c in 0..4 {
                    assert!(sum[c].is_finite(), "{w}x{h}s{scale}: sum[{c}] not finite");
                    assert!(
                        (sum[c] - expect[c]).abs() < 1e-4,
                        "{w}x{h}s{scale}: sum[{c}] = {}, recomputed {}",
                        sum[c],
                        expect[c]
                    );
                }
            }
        }
        // Flat 1×1: every tap sees the same pixel, so coarse == input and the
        // statistic is (near-)zero — the degenerate analogue of the existing
        // flat-field kernel test, through the FFI.
        let input = [0.5f32, 0.25, 0.75, 1.0];
        let mut coarse = [0.0f32; 4];
        let mut detail = [0.0f32; 4];
        let mut sum = [0.0f32; 4];
        unsafe {
            darkroom_eaw_dn_decompose(
                coarse.as_mut_ptr(),
                input.as_ptr(),
                detail.as_mut_ptr(),
                sum.as_mut_ptr(),
                0,
                1.0,
                1,
                1,
            );
        }
        for c in 0..4 {
            assert!((coarse[c] - input[c]).abs() < 1e-6, "1x1 coarse[{c}]");
            assert!(detail[c].abs() < 1e-6, "1x1 detail[{c}]");
            assert!(sum[c] < 1e-9, "1x1 sum[{c}] = {}", sum[c]);
        }
    }

    #[test]
    fn ffi_dn_decompose_scale_guards_are_noops() {
        // Negative scales are UB in the C (`1u << scale`); scales >= 32 leave
        // the C's 32-bit shift domain (and would panic the Rust shift). Both
        // are guarded no-ops: buffers and sum_squared untouched.
        let (w, h) = (8usize, 6usize);
        let input = dn_test_image(w, h);
        for scale in [-1i32, -100, 32, 33, 100, i32::MAX] {
            let mut coarse = vec![7.0f32; w * h * 4];
            let mut detail = vec![7.0f32; w * h * 4];
            let mut sum = [7.0f32; 4];
            unsafe {
                darkroom_eaw_dn_decompose(
                    coarse.as_mut_ptr(),
                    input.as_ptr(),
                    detail.as_mut_ptr(),
                    sum.as_mut_ptr(),
                    scale,
                    1.0,
                    w as i32,
                    h as i32,
                );
            }
            assert!(
                coarse.iter().all(|&v| v == 7.0),
                "scale {scale} touched coarse"
            );
            assert!(
                detail.iter().all(|&v| v == 7.0),
                "scale {scale} touched detail"
            );
            assert_eq!(sum, [7.0; 4], "scale {scale} touched sum_squared");
        }
        // Boundary scales 0 and 31 still run and stay finite.
        for scale in [0i32, 31] {
            let mut coarse = vec![0.0f32; w * h * 4];
            let mut detail = vec![0.0f32; w * h * 4];
            let mut sum = [0.0f32; 4];
            unsafe {
                darkroom_eaw_dn_decompose(
                    coarse.as_mut_ptr(),
                    input.as_ptr(),
                    detail.as_mut_ptr(),
                    sum.as_mut_ptr(),
                    scale,
                    1.0,
                    w as i32,
                    h as i32,
                );
            }
            assert!(
                coarse.iter().all(|v| v.is_finite()),
                "scale {scale} produced non-finite coarse"
            );
            assert!(
                sum.iter().all(|v| v.is_finite()),
                "scale {scale} sum not finite"
            );
        }
    }

    #[test]
    fn ffi_dn_decompose_guards_are_noops() {
        // Null pointers, degenerate dims, and overflowing dim products must
        // not crash and must leave live buffers untouched. (Short — but
        // non-overflowing — buffers remain a caller contract, as with
        // darkroom_eaw_synthesize.)
        let (w, h) = (6usize, 4usize);
        let input = dn_test_image(w, h);
        let mut coarse = vec![0.5f32; w * h * 4];
        let mut detail = vec![0.5f32; w * h * 4];
        let mut sum = [0.5f32; 4];
        let (before_c, before_d, before_s) = (coarse.clone(), detail.clone(), sum);
        unsafe {
            // Each null in turn (one live-pointer arm each so the call is
            // otherwise well-formed).
            darkroom_eaw_dn_decompose(
                std::ptr::null_mut(),
                input.as_ptr(),
                detail.as_mut_ptr(),
                sum.as_mut_ptr(),
                0,
                1.0,
                w as i32,
                h as i32,
            );
            darkroom_eaw_dn_decompose(
                coarse.as_mut_ptr(),
                std::ptr::null(),
                detail.as_mut_ptr(),
                sum.as_mut_ptr(),
                0,
                1.0,
                w as i32,
                h as i32,
            );
            darkroom_eaw_dn_decompose(
                coarse.as_mut_ptr(),
                input.as_ptr(),
                std::ptr::null_mut(),
                sum.as_mut_ptr(),
                0,
                1.0,
                w as i32,
                h as i32,
            );
            darkroom_eaw_dn_decompose(
                coarse.as_mut_ptr(),
                input.as_ptr(),
                detail.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
                1.0,
                w as i32,
                h as i32,
            );
        }
        assert_eq!(coarse, before_c, "null guards must not touch coarse");
        assert_eq!(detail, before_d, "null guards must not touch detail");
        assert_eq!(sum, before_s, "null guards must not touch sum_squared");

        // Degenerate dims: zero / negative width or height.
        for (dw, dh) in [(0, 4), (6, 0), (0, 0), (-3, 4), (6, -2)] {
            unsafe {
                darkroom_eaw_dn_decompose(
                    coarse.as_mut_ptr(),
                    input.as_ptr(),
                    detail.as_mut_ptr(),
                    sum.as_mut_ptr(),
                    0,
                    1.0,
                    dw,
                    dh,
                );
            }
        }
        assert_eq!(coarse, before_c, "degenerate dims must be no-ops");
        assert_eq!(detail, before_d, "degenerate dims must be no-ops");
        assert_eq!(sum, before_s, "degenerate dims must be no-ops");

        // Overflowing dim product (i32::MAX²·4 exceeds u64): guarded before
        // any slice is built, so dangling-but-non-null pointers are safe here
        // and the live buffers stay untouched.
        let dangling = std::ptr::NonNull::<f32>::dangling().as_ptr();
        let dangling_mut = std::ptr::NonNull::<f32>::dangling().as_ptr() as *mut f32;
        unsafe {
            darkroom_eaw_dn_decompose(
                dangling_mut,
                dangling,
                dangling_mut,
                sum.as_mut_ptr(),
                0,
                1.0,
                i32::MAX,
                i32::MAX,
            );
        }
        assert_eq!(sum, before_s, "overflow guard must not touch sum_squared");
    }
}
