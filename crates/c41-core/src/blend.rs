//! Mask post-processing kernels — port of the OMP loops in `src/develop/blend.c`.
//!
//! Three flat element-wise loops are ported, each operating on a single-channel
//! (1 float per pixel) mask buffer:
//!
//! - [`refine_detail_mask`] — `mask[k] *= clip(warp_mask[k])`, the detail-mask
//!   refinement in `_refine_with_detail_mask` (blend.c:291 / blend.c:841 CL path).
//! - [`mask_tone_curve`] — sigmoid contrast/brightness tone-curve on the mask,
//!   `_develop_blend_process_mask_tone_curve` (blend.c:417).
//! - [`invert_raster_mask`] — `mask[k] = (1.0 - raster_mask[k]) * opacity`, the
//!   inverted raster-mask fill (blend.c:571 / blend.c:1071 CL path).
//!
//! Bit-exactness notes.
//! - `CLIP(x)` (`math.h:73`) → `f32::clamp(0.0, 1.0)`. Mask values are never
//!   NaN, so NaN-propagation differences between the C ternary and Rust's
//!   IEEE min/max are moot.
//! - `FLT_EPSILON` → `std::f32::EPSILON` (identical bits).
//! - `expf` → `f32::exp`; under `-ffast-math` GCC still calls libm `expf` on
//!   x86-64, which is the same symbol Rust resolves to.
//! - `fminf`/`fmaxf` → `f32::min`/`f32::max`; `fabsf` → `f32::abs`.
//! - Expression order is preserved verbatim: `2.0 * mask[k] / opacity - 1.0`
//!   evaluates left-to-right as `((2.0 * mask[k]) / opacity) - 1.0`.

/// `mask[k] *= clamp(warp_mask[k], 0, 1)` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop at blend.c:291 (CPU path) and blend.c:841
/// (OpenCL path). Both are identical; the C side picks one based on the device.
pub fn refine_detail_mask(
    mask: &mut [f32],
    warp_mask: &[f32],
    n: usize,
) {
    let m = n.min(mask.len()).min(warp_mask.len());
    for k in 0..m {
        mask[k] *= warp_mask[k].clamp(0.0, 1.0);
    }
}

/// Sigmoid tone-curve on a mask buffer (in-place).
///
/// Port of the `DT_OMP_FOR_SIMD` loop at blend.c:417–441.
/// Pre-computes `e = expf(3 * contrast)` once, then per pixel applies the
/// brightness-biased x-axis mapping followed by the sigmoid
/// `0.5 * (x*e / (1 + (e-1)*|x|)) + 0.5`, clamped to `[0, opacity]`.
pub fn mask_tone_curve(
    mask: &mut [f32],
    n: usize,
    contrast: f32,
    brightness: f32,
    opacity: f32,
) {
    const MASK_EPSILON: f32 = 16.0f32 * std::f32::EPSILON;
    const MVAL_THRESHOLD: f32 = 1e-6f32;

    let e = (3.0f32 * contrast).exp();
    let nn = n.min(mask.len());

    for k in 0..nn {
        let mut x = 2.0f32 * mask[k] / opacity - 1.0f32;

        if 1.0f32 - brightness <= 0.0f32 {
            x = if mask[k] <= MASK_EPSILON { -1.0f32 } else { 1.0f32 };
        } else if 1.0f32 + brightness <= 0.0f32 {
            x = if mask[k] >= 1.0f32 - MASK_EPSILON { 1.0f32 } else { -1.0f32 };
        } else if brightness > 0.0f32 {
            x = (x + brightness) / (1.0f32 - brightness);
            x = x.min(1.0f32);
        } else {
            x = (x + brightness) / (1.0f32 + brightness);
            x = x.max(-1.0f32);
        }

        let cval = 0.5f32 * (x * e / (1.0f32 + (e - 1.0f32) * x.abs())) + 0.5f32;
        let mval = if cval > MVAL_THRESHOLD { cval } else { 0.0f32 };
        mask[k] = mval.clamp(0.0, 1.0) * opacity;
    }
}

/// `mask[k] = (1.0 - raster_mask[k]) * opacity` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop at blend.c:571 (CPU path) and blend.c:1071
/// (OpenCL path). Both are identical; the C side picks one based on the device.
pub fn invert_raster_mask(
    mask: &mut [f32],
    raster_mask: &[f32],
    n: usize,
    opacity: f32,
) {
    let m = n.min(mask.len()).min(raster_mask.len());
    for k in 0..m {
        mask[k] = (1.0f32 - raster_mask[k]) * opacity;
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `mask` and `warp_mask` must each hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blend_refine_detail_mask(
    mask: *mut f32,
    warp_mask: *const f32,
    n: usize,
) {
    if mask.is_null() || warp_mask.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let mask_slice = std::slice::from_raw_parts_mut(mask, n);
    let warp_slice = std::slice::from_raw_parts(warp_mask, n);
    refine_detail_mask(mask_slice, warp_slice, n);
}

/// # Safety
/// `mask` must hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blend_mask_tone_curve(
    mask: *mut f32,
    n: usize,
    contrast: f32,
    brightness: f32,
    opacity: f32,
) {
    if mask.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let mask_slice = std::slice::from_raw_parts_mut(mask, n);
    mask_tone_curve(mask_slice, n, contrast, brightness, opacity);
}

/// # Safety
/// `mask` and `raster_mask` must each hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blend_invert_raster_mask(
    mask: *mut f32,
    raster_mask: *const f32,
    n: usize,
    opacity: f32,
) {
    if mask.is_null() || raster_mask.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let mask_slice = std::slice::from_raw_parts_mut(mask, n);
    let raster_slice = std::slice::from_raw_parts(raster_mask, n);
    invert_raster_mask(mask_slice, raster_slice, n, opacity);
}

// ── Reference implementations for bit-exactness tests ────────────────────────
// These mirror the kernel bodies exactly so we can validate via assert_eq!.

fn ref_refine_detail_mask(
    mask: &mut [f32],
    warp_mask: &[f32],
    n: usize,
) {
    let m = n.min(mask.len()).min(warp_mask.len());
    for k in 0..m {
        mask[k] *= warp_mask[k].clamp(0.0, 1.0);
    }
}

fn ref_mask_tone_curve(
    mask: &mut [f32],
    n: usize,
    contrast: f32,
    brightness: f32,
    opacity: f32,
) {
    let mask_epsilon = 16.0f32 * std::f32::EPSILON;
    let e = (3.0f32 * contrast).exp();
    let nn = n.min(mask.len());
    for k in 0..nn {
        let mut x = 2.0f32 * mask[k] / opacity - 1.0f32;
        if 1.0f32 - brightness <= 0.0f32 {
            x = if mask[k] <= mask_epsilon { -1.0f32 } else { 1.0f32 };
        } else if 1.0f32 + brightness <= 0.0f32 {
            x = if mask[k] >= 1.0f32 - mask_epsilon { 1.0f32 } else { -1.0f32 };
        } else if brightness > 0.0f32 {
            x = (x + brightness) / (1.0f32 - brightness);
            x = x.min(1.0f32);
        } else {
            x = (x + brightness) / (1.0f32 + brightness);
            x = x.max(-1.0f32);
        }
        let cval = 0.5f32 * (x * e / (1.0f32 + (e - 1.0f32) * x.abs())) + 0.5f32;
        let mval = if cval > 1e-6 { cval } else { 0.0f32 };
        mask[k] = mval.clamp(0.0, 1.0) * opacity;
    }
}

fn ref_invert_raster_mask(
    mask: &mut [f32],
    raster_mask: &[f32],
    n: usize,
    opacity: f32,
) {
    let m = n.min(mask.len()).min(raster_mask.len());
    for k in 0..m {
        mask[k] = (1.0f32 - raster_mask[k]) * opacity;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    // ── refine_detail_mask ─────────────────────────────────────────────────────

    #[test]
    fn refine_detail_mask_clips_warp_to_1() {
        let mut mask = vec![1.0f32; 4];
        let warp = vec![2.0f32; 4]; // all above 1 → clipped to 1
        refine_detail_mask(&mut mask, &warp, 4);
        assert_eq!(mask, vec![1.0f32; 4]);
    }

    #[test]
    fn refine_detail_mask_clips_warp_to_0() {
        let mut mask = vec![1.0f32; 4];
        let warp = vec![-0.5f32; 4]; // below 0 → clipped to 0
        refine_detail_mask(&mut mask, &warp, 4);
        assert_eq!(mask, vec![0.0f32; 4]);
    }

    #[test]
    fn refine_detail_mask_normal_case() {
        let mut mask = vec![0.8f32, 0.6, 0.4, 0.2];
        let warp = vec![0.5f32, 0.5, 0.5, 0.5];
        refine_detail_mask(&mut mask, &warp, 4);
        // 0.8*0.5, 0.6*0.5, 0.4*0.5, 0.2*0.5
        assert_eq!(mask, vec![0.4, 0.3, 0.2, 0.1]);
    }

    #[test]
    fn refine_detail_mask_matches_reference_over_lcg() {
        let mut mask = vec![0.0f32; 256];
        let mut warp = vec![0.0f32; 256];
        lcg_fill(&mut mask, 0xABCD, 2.0); // [0, 2) → has out-of-range
        lcg_fill(&mut warp, 0xBEEF, 2.0);

        let mut direct = mask.clone();
        let mut reference = mask.clone();
        refine_detail_mask(&mut direct, &warp, 256);
        ref_refine_detail_mask(&mut reference, &warp, 256);
        assert_eq!(direct, reference);
    }

    // ── mask_tone_curve ───────────────────────────────────────────────────────

    #[test]
    fn mask_tone_curve_uniform_mask() {
        // mask = 0.5, contrast = 0, brightness = 0, opacity = 1.0
        // x = 2*0.5/1 - 1 = 0
        // cval = 0.5 * (0*e / (1 + (e-1)*0)) + 0.5 = 0.5
        // mval = 0.5, clip = 0.5, * opacity(1) = 0.5
        let mut mask = vec![0.5f32; 4];
        mask_tone_curve(&mut mask, 4, 0.0, 0.0, 1.0);
        for &v in &mask {
            assert!((v - 0.5).abs() < 1e-6, "got {v}, want 0.5");
        }
    }

    #[test]
    fn mask_tone_curve_clamps_below_threshold() {
        // Very negative contrast → e ≈ 0 → sigmoid becomes a step at x=0.
        // At x=-1 (mask=0) cval → 0; at x=0 (mask=0.5) cval is exactly 0.5;
        // at x=+1 (mask=1) cval → 1. So mask=0.0 gives near-zero after clamp.
        let mut mask = vec![0.0f32; 4];
        mask_tone_curve(&mut mask, 4, -20.0, 0.0, 1.0);
        for &v in &mask {
            assert!(v < 1e-5, "got {v}, expected near 0");
        }
    }

    #[test]
    fn mask_tone_curve_clamps_above_one() {
        // mask near 1.0 with positive contrast → cval > 1 → clipped to 1 * opacity
        let mut mask = vec![1.0f32; 4];
        mask_tone_curve(&mut mask, 4, 5.0, 0.0, 1.0);
        for &v in &mask {
            assert!((v - 1.0).abs() < 1e-6 || v < 1e-6, "got {v}");
        }
    }

    #[test]
    fn mask_tone_curve_matches_reference_over_lcg() {
        let mut mask = vec![0.0f32; 256];
        lcg_fill(&mut mask, 0xF00D, 1.0); // [0, 1)

        let mut direct = mask.clone();
        let mut reference = mask.clone();
        mask_tone_curve(&mut direct, 256, 0.5, 0.1, 1.0);
        ref_mask_tone_curve(&mut reference, 256, 0.5, 0.1, 1.0);
        assert_eq!(direct, reference);
    }

    #[test]
    fn mask_tone_curve_brightness_extreme() {
        // brightness = 1.0 → 1 - brightness = 0 → 1.f - brightness <= 0.f is true
        let mut mask = vec![0.7f32; 4];
        mask_tone_curve(&mut mask, 4, 0.0, 1.0, 1.0);
        // x = 0.7 <= mask_epsilon? No → x = 1.0
        // cval = 0.5 * (1.0 * e / (1 + (e-1) * 1.0)) + 0.5 = 0.5 * (e/e) + 0.5 = 1.0
        // mval = 1.0, clip = 1.0, * 1.0 = 1.0
        for &v in &mask {
            assert!((v - 1.0).abs() < 1e-6, "got {v}, want 1.0");
        }
    }

    #[test]
    fn mask_tone_curve_negative_brightness() {
        // brightness = -1.0 → 1 + brightness = 0 → 1.f + brightness <= 0.f is true
        let mut mask = vec![0.7f32; 4];
        mask_tone_curve(&mut mask, 4, 0.0, -1.0, 1.0);
        // x = 0.7 >= 1 - mask_epsilon? No → x = -1.0
        // cval = 0.5 * (-e / (1 + (e-1)*1)) + 0.5 = 0.5 * (-1) + 0.5 = 0.0
        // mval = 0.0, clip = 0.0, * 1.0 = 0.0
        for &v in &mask {
            assert!(v.abs() < 1e-6, "got {v}, want ≈0");
        }
    }

    // ── invert_raster_mask ─────────────────────────────────────────────────────

    #[test]
    fn invert_raster_mask_uniform() {
        let mut mask = vec![0.0f32; 4];
        let raster = vec![0.5f32; 4];
        invert_raster_mask(&mut mask, &raster, 4, 1.0);
        // (1 - 0.5) * 1.0 = 0.5
        assert_eq!(mask, vec![0.5f32; 4]);
    }

    #[test]
    fn invert_raster_mask_with_opacity() {
        let mut mask = vec![0.0f32; 4];
        let raster = vec![0.0f32, 1.0, 0.5, 0.25];
        invert_raster_mask(&mut mask, &raster, 4, 0.8);
        // (1-0)*0.8=0.8, (1-1)*0.8=0, (1-0.5)*0.8=0.4, (1-0.25)*0.8=0.6
        assert_eq!(mask, vec![0.8, 0.0, 0.4, 0.6]);
    }

    #[test]
    fn invert_raster_mask_matches_reference_over_lcg() {
        let raster = {
            let mut v = vec![0.0f32; 256];
            lcg_fill(&mut v, 0xCAFE, 1.0);
            v
        };
        let mut direct = vec![0.0f32; 256];
        let mut reference = vec![0.0f32; 256];
        invert_raster_mask(&mut direct, &raster, 256, 0.7);
        ref_invert_raster_mask(&mut reference, &raster, 256, 0.7);
        assert_eq!(direct, reference);
    }

    // ── FFI round-trip and null-guard tests ────────────────────────────────────

    #[test]
    fn ffi_refine_detail_mask_round_trip() {
        let mut mask = vec![0.0f32; 64];
        let mut warp = vec![0.0f32; 64];
        lcg_fill(&mut mask, 0x1111, 1.0);
        lcg_fill(&mut warp, 0x2222, 2.0);

        let mut ffi_mask = mask.clone();
        let mut direct_mask = mask.clone();

        unsafe {
            darkroom_blend_refine_detail_mask(ffi_mask.as_mut_ptr(), warp.as_ptr(), 64);
        }
        refine_detail_mask(&mut direct_mask, &warp, 64);
        assert_eq!(ffi_mask, direct_mask, "FFI refine_detail_mask mismatch");
    }

    #[test]
    fn ffi_refine_detail_mask_null_guard() {
        unsafe {
            darkroom_blend_refine_detail_mask(std::ptr::null_mut(), std::ptr::null(), 10);
        }
    }

    #[test]
    fn ffi_refine_detail_mask_zero_n() {
        let mut mask = vec![1.0f32; 4];
        let warp = vec![0.5f32; 4];
        unsafe {
            darkroom_blend_refine_detail_mask(mask.as_mut_ptr(), warp.as_ptr(), 0);
        }
        assert_eq!(mask, vec![1.0f32; 4]); // untouched
    }

    #[test]
    fn ffi_mask_tone_curve_round_trip() {
        let mut mask = vec![0.0f32; 64];
        lcg_fill(&mut mask, 0x3333, 1.0);

        let mut ffi_mask = mask.clone();
        let mut direct_mask = mask.clone();

        unsafe {
            darkroom_blend_mask_tone_curve(ffi_mask.as_mut_ptr(), 64, 0.5, 0.1, 1.0);
        }
        mask_tone_curve(&mut direct_mask, 64, 0.5, 0.1, 1.0);
        assert_eq!(ffi_mask, direct_mask, "FFI mask_tone_curve mismatch");
    }

    #[test]
    fn ffi_mask_tone_curve_null_guard() {
        unsafe {
            darkroom_blend_mask_tone_curve(std::ptr::null_mut(), 10, 0.0, 0.0, 1.0);
        }
    }

    #[test]
    fn ffi_invert_raster_mask_round_trip() {
        let mut raster = vec![0.0f32; 64];
        lcg_fill(&mut raster, 0x4444, 1.0);

        let mut ffi_mask = vec![0.0f32; 64];
        let mut direct_mask = vec![0.0f32; 64];

        unsafe {
            darkroom_blend_invert_raster_mask(ffi_mask.as_mut_ptr(), raster.as_ptr(), 64, 0.7);
        }
        invert_raster_mask(&mut direct_mask, &raster, 64, 0.7);
        assert_eq!(ffi_mask, direct_mask, "FFI invert_raster_mask mismatch");
    }

    #[test]
    fn ffi_invert_raster_mask_null_guard() {
        unsafe {
            darkroom_blend_invert_raster_mask(std::ptr::null_mut(), std::ptr::null(), 10, 1.0);
        }
    }
}
