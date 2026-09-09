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
                    let w = f * dn_weight(row, px2, inv_sigma2);
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
}
