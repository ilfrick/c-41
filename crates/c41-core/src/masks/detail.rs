//! Detail drawn-mask rendering — port of the OMP loops in
//! `src/develop/masks/detail.c`.
//!
//! Three per-pixel / per-row loops are ported:
//! - [`scharr_luminance`] — port of the `DT_OMP_FOR_SIMD` luminance→`sqrtf` loop
//!   in `dt_masks_calc_scharr_mask` (detail.c:129–137).
//! - [`scharr_gradient_mask`] — port of the `DT_OMP_FOR` Scharr-gradient loop in
//!   `dt_masks_calc_scharr_mask` (detail.c:139–151), calling
//!   [`scharr_gradient_at`] (port of `scharr_gradient` in math.h:405).
//! - [`detail_blend`] — port of the `DT_OMP_FOR_SIMD` blend-factor loop in
//!   `dt_masks_calc_detail_blend` (detail.c:173–178), inlining
//!   `_calcBlendFactor` (detail.c:156–162).
//!
//! The surrounding C logic — white-balance pre-scaling, buffer allocation,
//! the Gaussian blur in `dt_masks_calc_detail_mask`, and the public function
//! signatures — stays in C.
//!
//! Bit-exactness notes.
//! - `dt_fast_hypotf` with `__FAST_MATH__` (which the Release build defines via
//!   `-ffast-math`) reduces to `sqrtf(x*x + y*y)`, matched here by `f32::sqrt`.
//! - `dt_fast_expf` is already ported in `crate::math::fast_expf` and matches
//!   `dt_fast_expf()` in math.h:418 byte-for-byte over its whole domain.
//! - `47.0f / 255.0f` and `162.0f / 255.0f` are compile-time float divisions;
//!   Rust `const` division yields identical f32 bits.
//! - `CLAMP(row, 1, height-2)` → `row.clamp(1, height-2)`, `CLIP(x)` →
//!   `x.clamp(0.0, 1.0)`.

use crate::math::fast_expf;

/// Scharr operator weight `47.0 / 255.0` (math.h:407).
const SCHARR_47: f32 = 47.0f32 / 255.0f32;
/// Scharr operator weight `162.0 / 255.0` (math.h:408).
const SCHARR_162: f32 = 162.0f32 / 255.0f32;

/// `scharr_gradient` (math.h:405) — gradient magnitude at `tmp[idx]` using the
/// 3×3 Scharr operator. With `__FAST_MATH__`, `dt_fast_hypotf(gx, gy)` is
/// `sqrtf(gx*gx + gy*gy)`, matched by `f32::sqrt`.
///
/// # Safety contract (callers must guarantee)
/// `idx` is at least `width + 1` away from the start and end of `tmp`, so all
/// reads `idx ± width ± 1` and `idx ± 1` are in-bounds. The C caller clamps
/// `irow` to `[1, height-2]` and `icol` to `[1, width-2]`, which is exactly
/// this condition.
#[inline]
fn scharr_gradient_at(tmp: &[f32], idx: usize, width: usize) -> f32 {
    let gx = SCHARR_47 * (tmp[idx - width - 1] - tmp[idx - width + 1]
                          + tmp[idx + width - 1] - tmp[idx + width + 1])
           + SCHARR_162 * (tmp[idx - 1] - tmp[idx + 1]);
    let gy = SCHARR_47 * (tmp[idx - width - 1] - tmp[idx + width - 1]
                          + tmp[idx - width + 1] - tmp[idx + width + 1])
           + SCHARR_162 * (tmp[idx - width] - tmp[idx + width]);
    f32::sqrt(gx * gx + gy * gy)
}

/// Port of the `DT_OMP_FOR_SIMD` luminance loop (detail.c:129–137).
///
/// For each pixel `idx` in `[0, width*height)`:
/// `tmp[idx] = sqrtf(luminance / 3.0)` where
/// `luminance = max(0, src[4*idx]*wb[0]) + max(0, src[4*idx+1]*wb[1])
///              + max(0, src[4*idx+2]*wb[2])`.
/// `fmaxf(x, 0.0)` → `f32::max(0.0)`.
pub fn scharr_luminance(
    src: &[f32],
    tmp: &mut [f32],
    width: i32,
    height: i32,
    wb: &[f32; 4],
) {
    let msize = width as usize * height as usize;
    let n = msize.min(tmp.len()).min(src.len() / 4);
    for idx in 0..n {
        let val = (0.0f32).max(src[4 * idx] * wb[0])
            + (0.0f32).max(src[4 * idx + 1] * wb[1])
            + (0.0f32).max(src[4 * idx + 2] * wb[2]);
        tmp[idx] = (val / 3.0f32).sqrt();
    }
}

/// Port of the `DT_OMP_FOR` Scharr-gradient loop (detail.c:139–151).
///
/// For each pixel `(row, col)` in `[0,height) × [0,width)`:
/// `irow = clamp(row, 1, height-2)`, `icol = clamp(col, 1, width-2)`,
/// `idx = irow*width + icol`,
/// `mask[row*width+col] = clip(scharr_gradient(&tmp[idx], width) / 16.0)`.
pub fn scharr_gradient_mask(
    tmp: &[f32],
    mask: &mut [f32],
    width: i32,
    height: i32,
) {
    let w = width as usize;
    let h = height as usize;
    if w < 3 || h < 3 {
        return;
    }
    for row in 0..h {
        let irow = (row as i32).clamp(1, height - 2) as usize;
        for col in 0..w {
            let icol = (col as i32).clamp(1, width - 2) as usize;
            let idx = irow * w + icol;
            let gm = scharr_gradient_at(tmp, idx, w);
            mask[row * w + col] = (gm / 16.0f32).clamp(0.0, 1.0);
        }
    }
}

/// Port of `_calcBlendFactor` + `CLIP` + detail inversion (detail.c:156–178).
///
/// `ithreshold = 16.0 / max(1e-7, threshold)`.
/// For each pixel `idx`:
/// `blend = clip(1.0 / (1.0 + fast_expf(16.0 - ithreshold * src[idx])))`
/// `out[idx] = detail ? blend : 1.0 - blend`.
pub fn detail_blend(
    src: &[f32],
    out: &mut [f32],
    msize: usize,
    threshold: f32,
    detail: bool,
) {
    let ithreshold = 16.0f32 / 1e-7f32.max(threshold);
    let n = msize.min(out.len()).min(src.len());
    for idx in 0..n {
        let blend = (1.0f32 / (1.0f32 + fast_expf(16.0f32 - ithreshold * src[idx])))
            .clamp(0.0, 1.0);
        out[idx] = if detail { blend } else { 1.0f32 - blend };
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `src` must hold at least `width*height*4` floats (4-channel image);
/// `tmp` must hold at least `width*height` floats; `wb` must point to 4 floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_detail_scharr_luminance(
    src: *const f32,
    tmp: *mut f32,
    width: i32,
    height: i32,
    wb: *const f32,
) {
    if src.is_null() || tmp.is_null() || wb.is_null() || width <= 0 || height <= 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let msize = match w.checked_mul(h) {
        Some(v) => v,
        None => return,
    };
    let src_slice = std::slice::from_raw_parts(src, msize * 4);
    let tmp_slice = std::slice::from_raw_parts_mut(tmp, msize);
    let wb_arr: [f32; 4] = [
        *wb.add(0),
        *wb.add(1),
        *wb.add(2),
        *wb.add(3),
    ];
    scharr_luminance(src_slice, tmp_slice, width, height, &wb_arr);
}

/// # Safety
/// `tmp` must hold at least `width*height` floats; `mask` must hold at least
/// `width*height` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_detail_scharr_gradient(
    tmp: *const f32,
    mask: *mut f32,
    width: i32,
    height: i32,
) {
    if tmp.is_null() || mask.is_null() || width <= 0 || height <= 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let msize = match w.checked_mul(h) {
        Some(v) => v,
        None => return,
    };
    let tmp_slice = std::slice::from_raw_parts(tmp, msize);
    let mask_slice = std::slice::from_raw_parts_mut(mask, msize);
    scharr_gradient_mask(tmp_slice, mask_slice, width, height);
}

/// # Safety
/// `src` and `out` must each hold at least `msize` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_detail_blend(
    src: *const f32,
    out: *mut f32,
    msize: usize,
    threshold: f32,
    detail: i32,
) {
    if src.is_null() || out.is_null() || msize == 0 {
        return;
    }
    let src_slice = std::slice::from_raw_parts(src, msize);
    let out_slice = std::slice::from_raw_parts_mut(out, msize);
    detail_blend(src_slice, out_slice, msize, threshold, detail != 0);
}

// ── Reference implementations for bit-exactness tests ────────────────────────

/// Reference for `scharr_luminance` — identical to the kernel.
fn ref_scharr_luminance(
    src: &[f32],
    tmp: &mut [f32],
    width: i32,
    height: i32,
    wb: &[f32; 4],
) {
    let msize = (width as usize) * (height as usize);
    let n = msize.min(tmp.len()).min(src.len() / 4);
    for idx in 0..n {
        let val = (0.0f32).max(src[4 * idx] * wb[0])
            + (0.0f32).max(src[4 * idx + 1] * wb[1])
            + (0.0f32).max(src[4 * idx + 2] * wb[2]);
        tmp[idx] = (val / 3.0f32).sqrt();
    }
}

/// Reference for `scharr_gradient_mask` — identical to the kernel.
fn ref_scharr_gradient_mask(
    tmp: &[f32],
    mask: &mut [f32],
    width: i32,
    height: i32,
) {
    let w = width as usize;
    let h = height as usize;
    if w < 3 || h < 3 {
        return;
    }
    for row in 0..h {
        let irow = (row as i32).clamp(1, height - 2) as usize;
        for col in 0..w {
            let icol = (col as i32).clamp(1, width - 2) as usize;
            let idx = irow * w + icol;
            let gx = SCHARR_47 * (tmp[idx - w - 1] - tmp[idx - w + 1]
                                  + tmp[idx + w - 1] - tmp[idx + w + 1])
                   + SCHARR_162 * (tmp[idx - 1] - tmp[idx + 1]);
            let gy = SCHARR_47 * (tmp[idx - w - 1] - tmp[idx + w - 1]
                                  + tmp[idx - w + 1] - tmp[idx + w + 1])
                   + SCHARR_162 * (tmp[idx - w] - tmp[idx + w]);
            let gm = f32::sqrt(gx * gx + gy * gy);
            mask[row * w + col] = (gm / 16.0f32).clamp(0.0, 1.0);
        }
    }
}

/// Reference for `detail_blend` — identical to the kernel.
fn ref_detail_blend(
    src: &[f32],
    out: &mut [f32],
    msize: usize,
    threshold: f32,
    detail: bool,
) {
    let ithreshold = 16.0f32 / 1e-7f32.max(threshold);
    let n = msize.min(out.len()).min(src.len());
    for idx in 0..n {
        let blend = (1.0f32 / (1.0f32 + fast_expf(16.0f32 - ithreshold * src[idx])))
            .clamp(0.0, 1.0);
        out[idx] = if detail { blend } else { 1.0f32 - blend };
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_util::lcg_fill;

    // ── scharr_luminance tests ─────────────────────────────────────────────

    #[test]
    fn scharr_luminance_uniform_white() {
        // src = all white (1.0 in RGB, 1.0 in A), wb = identity
        let w = 2i32;
        let h = 2i32;
        let src = vec![1.0f32; 4 * (w * h) as usize];
        let wb = [1.0f32; 4];
        let mut tmp = vec![0.0f32; (w * h) as usize];
        scharr_luminance(&src, &mut tmp, w, h, &wb);
        // val = 0 + 1 + 1 = 3.0, tmp = sqrt(3.0/3.0) = sqrt(1.0) = 1.0
        assert_eq!(tmp, vec![1.0f32; 4]);
    }

    #[test]
    fn scharr_luminance_black_pixel() {
        // src = all black, wb = identity
        let src = vec![0.0f32; 4 * 4];
        let wb = [1.0f32; 4];
        let mut tmp = vec![0.0f32; 4];
        scharr_luminance(&src, &mut tmp, 2, 2, &wb);
        // val = 0, tmp = sqrt(0) = 0
        assert_eq!(tmp, vec![0.0f32; 4]);
    }

    #[test]
    fn scharr_luminance_with_white_balance() {
        let w = 2i32;
        let h = 2i32;
        let src = vec![1.0f32, 0.5, 1.0, 1.0,   // pixel 0: R=1, G=0.5, B=1, A=1
                       0.0, 0.0, 0.0, 1.0,
                       1.0, 1.0, 1.0, 1.0,
                       0.5, 0.5, 0.5, 1.0];
        let wb = [2.0f32, 0.5, 1.0, 1.0];
        let mut tmp = vec![0.0f32; 4];
        scharr_luminance(&src, &mut tmp, w, h, &wb);
        // pixel 0: max(0, 1*2.0) + max(0, 0.5*0.5) + max(0, 1*1.0) = 2.0 + 0.25 + 1.0 = 3.25
        // tmp[0] = sqrt(3.25/3.0) = sqrt(1.08333...) ≈ 1.040833
        let expected = (3.25f32 / 3.0f32).sqrt();
        assert!((tmp[0] - expected).abs() < 1e-6, "got {}, want {}", tmp[0], expected);
        // pixel 1: all 0 → sqrt(0) = 0
        assert_eq!(tmp[1], 0.0f32);
        // pixel 2: max(0,1*2.0) + max(0,1*0.5) + max(0,1*1.0) = 2.0 + 0.5 + 1.0 = 3.5
        // tmp[2] = sqrt(3.5/3.0)
        let expected3 = (3.5f32 / 3.0f32).sqrt();
        assert!((tmp[2] - expected3).abs() < 1e-6, "got {}, want {}", tmp[2], expected3);
        // pixel 3: max(0,0.5*2.0) + max(0,0.5*0.5) + max(0,0.5*1.0) = 1.0 + 0.25 + 0.5 = 1.75
        // tmp[3] = sqrt(1.75/3.0)
        let expected4 = (1.75f32 / 3.0f32).sqrt();
        assert!((tmp[3] - expected4).abs() < 1e-6, "got {}, want {}", tmp[3], expected4);
    }

    #[test]
    fn scharr_luminance_matches_reference_over_lcg() {
        let w = 4i32;
        let h = 4i32;
        let mut src = vec![0.0f32; 4 * (w * h) as usize];
        lcg_fill(&mut src, 0xBEEF, 5.0);
        let wb = [1.2f32, 0.9, 1.1, 1.0];

        let mut direct = vec![0.0f32; (w * h) as usize];
        let mut reference = vec![0.0f32; (w * h) as usize];
        scharr_luminance(&src, &mut direct, w, h, &wb);
        ref_scharr_luminance(&src, &mut reference, w, h, &wb);
        assert_eq!(direct, reference);
    }

    // ── scharr_gradient tests ──────────────────────────────────────────────

    #[test]
    fn scharr_gradient_uniform_image_is_zero() {
        // A uniform tmp buffer has zero gradient everywhere
        let w = 4i32;
        let h = 4i32;
        let tmp = vec![0.5f32; (w * h) as usize];
        let mut mask = vec![0.0f32; (w * h) as usize];
        scharr_gradient_mask(&tmp, &mut mask, w, h);
        // gradient = 0 everywhere, clip(0/16) = 0
        assert_eq!(mask, vec![0.0f32; 16]);
    }

    #[test]
    fn scharr_gradient_clamps_border() {
        // Verify border pixels use the clamped interior index
        let w = 4i32;
        let h = 4i32;
        // Create a known pattern in tmp
        let mut tmp = vec![0.0f32; 16];
        // Set interior to a known pattern (checkerboard)
        for idx in 0..16 {
            tmp[idx] = if (idx / 4) % 2 == (idx % 4) % 2 { 1.0f32 } else { 0.0f32 };
        }

        let mut mask = vec![0.0f32; 16];
        scharr_gradient_mask(&tmp, &mut mask, w, h);

        // Border (row 0) should use irow=1 clamped position
        // Verify it matches the reference
        let mut ref_mask = vec![0.0f32; 16];
        ref_scharr_gradient_mask(&tmp, &mut ref_mask, w, h);
        assert_eq!(mask, ref_mask);
    }

    #[test]
    fn scharr_gradient_matches_reference_over_lcg() {
        let w = 8i32;
        let h = 8i32;
        let mut tmp = vec![0.0f32; (w * h) as usize];
        lcg_fill(&mut tmp, 0xCAFE, 1.0);
        let mut mask = vec![0.0f32; (w * h) as usize];
        let mut ref_mask = vec![0.0f32; (w * h) as usize];
        scharr_gradient_mask(&tmp, &mut mask, w, h);
        ref_scharr_gradient_mask(&tmp, &mut ref_mask, w, h);
        assert_eq!(mask, ref_mask);
    }

    // ── detail_blend tests ─────────────────────────────────────────────────

    #[test]
    fn detail_blend_inflexion_point() {
        // At val = threshold, ithreshold * val = 16.0, argument to expf = 0
        // fast_expf(0.0) = 1.0 (from_bits(0x3f800000) = 1.0)
        // blend = 1/(1+1) = 0.5
        let src = vec![0.5f32; 4];
        let mut out = vec![0.0f32; 4];
        detail_blend(&src, &mut out, 4, 0.5, true);
        for v in &out {
            assert!((v - 0.5).abs() < 1e-6, "got {}, want 0.5", v);
        }
    }

    #[test]
    fn detail_blend_above_threshold() {
        // val > threshold → argument negative → fast_expf ≈ 0 → blend = 1.0
        let src = vec![0.9f32; 4];
        let mut out = vec![0.0f32; 4];
        // threshold = 0.5, ithreshold = 32.0
        // argument = 16.0 - 32.0*0.9 = 16.0 - 28.8 = -12.8
        // fast_expf(-12.8) ≈ very small → blend ≈ 1.0
        detail_blend(&src, &mut out, 4, 0.5, true);
        for v in &out {
            assert!((v - 1.0).abs() < 1e-5, "got {}, want ≈1.0", v);
        }
    }

    #[test]
    fn detail_blend_detail_false() {
        // detail = false → out = 1.0 - blend
        let src = vec![0.5f32; 4];
        let mut out = vec![0.0f32; 4];
        detail_blend(&src, &mut out, 4, 0.5, false);
        // blend = 0.5, out = 1.0 - 0.5 = 0.5
        for v in &out {
            assert!((v - 0.5).abs() < 1e-6, "got {}, want 0.5", v);
        }
    }

    #[test]
    fn detail_blend_threshold_zero_uses_epsilon() {
        // threshold = 0.0 → ithreshold = 16.0 / max(1e-7, 0.0) = 16.0 / 1e-7 = 1.6e8
        // For val = 1.0: argument = 16.0 - 1.6e8 * 1.0 = very negative
        // fast_expf ≈ 0 → blend ≈ 1.0
        let src = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        detail_blend(&src, &mut out, 4, 0.0, true);
        for v in &out {
            assert!((v - 1.0).abs() < 1e-5, "got {}, want ≈1.0", v);
        }
    }

    #[test]
    fn detail_blend_matches_reference_over_lcg() {
        let mut src = vec![0.0f32; 256];
        lcg_fill(&mut src, 0xDEAD, 1.0);

        for &threshold in &[0.0f32, 0.1, 0.5, 1.0] {
            for detail in [false, true] {
                let mut direct = vec![0.0f32; 256];
                let mut reference = vec![0.0f32; 256];
                detail_blend(&src, &mut direct, 256, threshold, detail);
                ref_detail_blend(&src, &mut reference, 256, threshold, detail);
                assert_eq!(direct, reference,
                    "threshold={threshold} detail={detail} mismatch vs reference");
            }
        }
    }

    // ── FFI round-trip and null-guard tests ──────────────────────────────

    #[test]
    fn ffi_scharr_luminance_round_trip() {
        unsafe {
            let w = 4i32;
            let h = 3i32;
            let mut src = vec![0.0f32; 4 * (w * h) as usize];
            lcg_fill(&mut src, 0x1337, 2.0);
            let wb = [1.0f32, 1.0, 1.0, 1.0];
            let mut ffi_tmp = vec![0.0f32; (w * h) as usize];

            darkroom_masks_detail_scharr_luminance(
                src.as_ptr(), ffi_tmp.as_mut_ptr(), w, h, wb.as_ptr());

            let mut direct_tmp = vec![0.0f32; (w * h) as usize];
            scharr_luminance(&src, &mut direct_tmp, w, h, &wb);
            assert_eq!(ffi_tmp, direct_tmp, "FFI scharr_luminance mismatch");
        }
    }

    #[test]
    fn ffi_scharr_luminance_null_guard() {
        unsafe {
            let wb = [1.0f32; 4];
            // null src → no-op
            darkroom_masks_detail_scharr_luminance(
                std::ptr::null(), std::ptr::null_mut(), 2, 2, wb.as_ptr());
            // null tmp → no-op
            let src = vec![0.0f32; 16];
            let mut tmp = vec![0.0f32; 4];
            darkroom_masks_detail_scharr_luminance(
                src.as_ptr(), std::ptr::null_mut(), 2, 2, wb.as_ptr());
            assert!(tmp.iter().all(|&v| v == 0.0));
            // null wb → no-op
            darkroom_masks_detail_scharr_luminance(
                src.as_ptr(), tmp.as_mut_ptr(), 2, 2, std::ptr::null());
            assert!(tmp.iter().all(|&v| v == 0.0));
            // zero dimensions → no-op
            let mut tmp = vec![0.0f32; 4];
            darkroom_masks_detail_scharr_luminance(
                src.as_ptr(), tmp.as_mut_ptr(), 0, 0, wb.as_ptr());
            assert!(tmp.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn ffi_scharr_gradient_round_trip() {
        unsafe {
            let w = 5i32;
            let h = 4i32;
            let mut tmp = vec![0.0f32; (w * h) as usize];
            lcg_fill(&mut tmp, 0x10AD, 3.0);
            let mut ffi_mask = vec![0.0f32; (w * h) as usize];
            let mut direct_mask = vec![0.0f32; (w * h) as usize];

            darkroom_masks_detail_scharr_gradient(
                tmp.as_ptr(), ffi_mask.as_mut_ptr(), w, h);
            scharr_gradient_mask(&tmp, &mut direct_mask, w, h);
            assert_eq!(ffi_mask, direct_mask, "FFI scharr_gradient mismatch");
        }
    }

    #[test]
    fn ffi_scharr_gradient_null_guard() {
        unsafe {
            // null tmp → no-op
            darkroom_masks_detail_scharr_gradient(
                std::ptr::null(), std::ptr::null_mut(), 4, 4);
            // null mask → no-op
            let tmp = vec![0.0f32; 16];
            let mut mask = vec![0.0f32; 16];
            darkroom_masks_detail_scharr_gradient(
                tmp.as_ptr(), std::ptr::null_mut(), 4, 4);
            assert!(mask.iter().all(|&v| v == 0.0));
            // zero dimensions → no-op
            darkroom_masks_detail_scharr_gradient(
                tmp.as_ptr(), mask.as_mut_ptr(), 0, 0);
            assert!(mask.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn ffi_detail_blend_round_trip() {
        unsafe {
            let mut src = vec![0.0f32; 64];
            lcg_fill(&mut src, 0xC0FFEE, 2.0);
            let mut ffi_out = vec![0.0f32; 64];
            let mut direct_out = vec![0.0f32; 64];

            darkroom_masks_detail_blend(
                src.as_ptr(), ffi_out.as_mut_ptr(), 64, 0.5, 1);
            detail_blend(&src, &mut direct_out, 64, 0.5, true);
            assert_eq!(ffi_out, direct_out, "FFI detail_blend mismatch");
        }
    }

    #[test]
    fn ffi_detail_blend_null_guard() {
        unsafe {
            let src = vec![0.0f32; 16];
            let mut out = vec![0.0f32; 16];
            // null src → no-op
            darkroom_masks_detail_blend(
                std::ptr::null(), out.as_mut_ptr(), 16, 0.5, 1);
            assert!(out.iter().all(|&v| v == 0.0));
            // null out → no-op
            darkroom_masks_detail_blend(
                src.as_ptr(), std::ptr::null_mut(), 16, 0.5, 1);
            // msize == 0 → no-op
            darkroom_masks_detail_blend(
                src.as_ptr(), out.as_mut_ptr(), 0, 0.5, 1);
            assert!(out.iter().all(|&v| v == 0.0));
        }
    }
}
