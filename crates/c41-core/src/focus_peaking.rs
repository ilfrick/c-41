//! Element-wise loops ported from `src/common/focus_peaking.h`.
//!
//! Three loops are ported here, all from `dt_focuspeaking`:
//! - The luma computation loop (focus_peaking.h:86) that converts a 4-channel
//!   uint8 sRGB image to a single-channel float luma buffer using
//!   `sqrt(pow(c0, 4.4) + pow(c1, 4.4) + pow(c2, 4.4))`.
//! - The TV_sum reduction (focus_peaking.h:136) that sums luma values over the
//!   interior pixel region.
//! - The sigma reduction (focus_peaking.h:147) that sums `|luma[k] - TV_sum|`
//!   over the interior pixel region.
//!
//! The original loops used `DT_OMP_FOR_SIMD` (parallel+SIMD). The Rust kernels
//! are single-threaded sequential; LLVM's auto-vectorizer provides SIMD at `-O3`,
//! but multi-threaded parallelism is no longer used. This matches the m4-161
//! `blend.rs` and m4-162 `imagebuf.rs` pattern.
//!
//! Bit-exactness notes:
//! - The luma loop computes `powf(c, 4.4)` — a single transcendental call
//!   with no `a*b + c` pattern, so there is no FMA contraction risk. (The
//!   C `const float exponent = 2.0f * 2.2f` is compile-time folded to 4.4f.)
//! - No `fmaxf`/`fminf` in these loops — only `sqrtf`, `powf`, `fabsf`, and
//!   reductions.
//! - The reductions are sequential left-to-right sums, matching the C
//!   sequential fallback's order. The C parallel path (OpenMP `reduction`)
//!   reassociates, so its summation order differs from any fixed order; the
//!   magnitude of the difference grows with element count rather than being
//!   bounded by 1 ULP. Accepted here because the sums feed focus-peaking
//!   overlay thresholds only, per the `imagebuf.rs::linear_blend` precedent.
//! - `fabsf(x)` → `x.abs()`. Bit-identical for all finite and NaN inputs.

use std::os::raw::c_uchar;

/// Compute luma from a 4-channel uint8 sRGB image.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_focuspeaking`
/// (focus_peaking.h:86). For each pixel, computes:
/// `sqrt(pow(c0/255, 4.4) + pow(c1/255, 4.4) + pow(c2/255, 4.4))`
///
/// # Arguments
/// - `image`: 4-channel uint8 image data (BGRA or RGBA depending on caller).
/// - `luma`: single-channel float output buffer, must hold at least `n_pixels` floats.
/// - `n_pixels`: number of pixels to process.
pub fn compute_luma(image: &[u8], luma: &mut [f32], n_pixels: usize) {
    let m = n_pixels.min(luma.len()).min(image.len() / 4);
    let exponent = 2.0f32 * 2.2f32;
    for k in 0..m {
        let index_rgb = k * 4;
        let c0 = image[index_rgb] as f32 / 255.0f32;
        let c1 = image[index_rgb + 1] as f32 / 255.0f32;
        let c2 = image[index_rgb + 2] as f32 / 255.0f32;
        luma[k] = (c0.powf(exponent) + c1.powf(exponent) + c2.powf(exponent)).sqrt();
    }
}

/// Sum luma values over the interior region of an image.
///
/// Port of the `DT_OMP_FOR_SIMD` reduction in `dt_focuspeaking`
/// (focus_peaking.h:136). Sums `luma_ds[i * width + j]` for
/// `i` in `2..height-2` and `j` in `2..width-2`.
///
/// # Arguments
/// - `luma_ds`: float buffer of size `width * height`.
/// - `width`: image width in pixels.
/// - `height`: image height in pixels.
/// - `border`: border exclusion (must be >= 2; the C code uses 2).
///
/// Returns the raw sum (before the final division).
pub fn sum_interior(luma_ds: &[f32], width: usize, height: usize, border: usize) -> f32 {
    // Clamp to the slice the same way sum_abs_deviation does, so a safe-Rust
    // caller with a short slice degrades instead of panicking on index.
    let h = height.min(luma_ds.len() / width.max(1));
    // Guard against underflow when height or width is <= 2*border
    if h <= 2 * border || width <= 2 * border {
        return 0.0f32;
    }
    let mut sum = 0.0f32;
    for i in border..h - border {
        for j in border..width - border {
            sum += luma_ds[i * width + j];
        }
    }
    sum
}

/// Sum `|luma_ds[i * width + j] - tv_sum|` over the interior region.
///
/// Port of the `DT_OMP_FOR_SIMD` reduction in `dt_focuspeaking`
/// (focus_peaking.h:147). Sums `fabsf(luma_ds[i * width + j] - TV_sum)` for
/// `i` in `2..height-2` and `j` in `2..width-2`.
///
/// # Arguments
/// - `luma_ds`: float buffer of size `width * height`.
/// - `width`: image width in pixels.
/// - `height`: image height in pixels.
/// - `tv_sum`: the mean luma value (already divided by count).
/// - `border`: border exclusion (must be >= 2; the C code uses 2).
///
/// Returns the raw sum (before the final division).
pub fn sum_abs_deviation(luma_ds: &[f32], width: usize, height: usize, tv_sum: f32, border: usize) -> f32 {
    let h = height.min(luma_ds.len() / width.max(1));
    if h <= 2 * border || width <= 2 * border {
        return 0.0f32;
    }
    let mut sum = 0.0f32;
    for i in border..h - border {
        for j in border..width - border {
            sum += (luma_ds[i * width + j] - tv_sum).abs();
        }
    }
    sum
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `image` must hold at least `n_pixels * 4` bytes. `luma` must hold at
/// least `n_pixels` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_focuspeaking_compute_luma(
    image: *const c_uchar,
    luma: *mut f32,
    n_pixels: usize,
) {
    if image.is_null() || luma.is_null() || n_pixels == 0 || n_pixels > i32::MAX as usize {
        return;
    }
    let image_slice = std::slice::from_raw_parts(image, n_pixels * 4);
    let luma_slice = std::slice::from_raw_parts_mut(luma, n_pixels);
    compute_luma(image_slice, luma_slice, n_pixels);
}

/// # Safety
/// `luma_ds` must hold at least `width * height` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_focuspeaking_sum_interior(
    luma_ds: *const f32,
    width: usize,
    height: usize,
) -> f32 {
    if luma_ds.is_null() || width == 0 || height == 0
        || width > i32::MAX as usize || height > i32::MAX as usize
    {
        return 0.0f32;
    }
    let luma_slice = std::slice::from_raw_parts(luma_ds, width * height);
    sum_interior(luma_slice, width, height, 2)
}

/// # Safety
/// `luma_ds` must hold at least `width * height` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_focuspeaking_sum_abs_deviation(
    luma_ds: *const f32,
    width: usize,
    height: usize,
    tv_sum: f32,
) -> f32 {
    if luma_ds.is_null() || width == 0 || height == 0
        || width > i32::MAX as usize || height > i32::MAX as usize
    {
        return 0.0f32;
    }
    let luma_slice = std::slice::from_raw_parts(luma_ds, width * height);
    sum_abs_deviation(luma_slice, width, height, tv_sum, 2)
}

// ── Independent reference implementations ──────────────────────────────────
//
// These recompute the same results with a slightly different code shape
// (named temporaries, different iteration form). The structural difference
// is modest — the real validation weight is carried by the known-value
// basic tests and the FFI round-trips below. This matches the established
// repo-wide reference-implementation pattern.

#[allow(dead_code)]
fn ref_compute_luma(image: &[u8], luma: &mut [f32], n_pixels: usize) {
    let m = n_pixels.min(luma.len()).min(image.len() / 4);
    let exponent = 2.0f32 * 2.2f32;
    for k in 0..m {
        let index_rgb = k * 4;
        let c0 = image[index_rgb] as f32 / 255.0f32;
        let c1 = image[index_rgb + 1] as f32 / 255.0f32;
        let c2 = image[index_rgb + 2] as f32 / 255.0f32;
        let p0 = c0.powf(exponent);
        let p1 = c1.powf(exponent);
        let p2 = c2.powf(exponent);
        let sum = p0 + p1 + p2;
        luma[k] = sum.sqrt();
    }
}

#[allow(dead_code)]
fn ref_sum_interior(luma_ds: &[f32], width: usize, height: usize, border: usize) -> f32 {
    let h = height.min(luma_ds.len() / width.max(1));
    if h <= 2 * border || width <= 2 * border {
        return 0.0f32;
    }
    let mut total = 0.0f32;
    for i in border..h - border {
        for j in border..width - border {
            total += luma_ds[i * width + j];
        }
    }
    total
}

#[allow(dead_code)]
fn ref_sum_abs_deviation(luma_ds: &[f32], width: usize, height: usize, tv_sum: f32, border: usize) -> f32 {
    let h = height.min(luma_ds.len() / width.max(1));
    if h <= 2 * border || width <= 2 * border {
        return 0.0f32;
    }
    let mut total = 0.0f32;
    for i in border..h - border {
        for j in border..width - border {
            let diff = luma_ds[i * width + j] - tv_sum;
            total += diff.abs();
        }
    }
    total
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_luma ───────────────────────────────────────────────────────────

    #[test]
    fn compute_luma_basic() {
        // All-black pixel (BGRA = 0,0,0,255) → luma = 0
        let image = vec![0u8, 0, 0, 255];
        let mut luma = vec![-1.0f32; 1];
        compute_luma(&image, &mut luma, 1);
        assert_eq!(luma[0], 0.0);

        // All-white pixel (255,255,255,255) → luma = sqrt(3 * 1.0^4.4) = sqrt(3)
        let image = vec![255u8, 255, 255, 255];
        let mut luma = vec![-1.0f32; 1];
        compute_luma(&image, &mut luma, 1);
        assert!((luma[0] - 3.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn compute_luma_matches_reference_over_lcg() {
        let mut image = vec![0u8; 256 * 4];
        // Fill with pseudo-random bytes (deterministic)
        let mut seed = 0xABu32;
        for b in &mut image {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = ((seed >> 16) % 256) as u8;
        }

        let mut direct = vec![0.0f32; 256];
        let mut reference = vec![0.0f32; 256];

        compute_luma(&image, &mut direct, 256);
        ref_compute_luma(&image, &mut reference, 256);

        assert_eq!(direct, reference);
    }

    // ── sum_interior ───────────────────────────────────────────────────────────

    #[test]
    fn sum_interior_basic() {
        // 5x5 image, all ones, border=2 → interior is 1x1 (the center pixel)
        let luma = vec![1.0f32; 25];
        let sum = sum_interior(&luma, 5, 5, 2);
        assert_eq!(sum, 1.0); // only pixel (2,2) is interior

        // 6x6 all ones, border=2 → interior is 2x2
        let luma = vec![1.0f32; 36];
        let sum = sum_interior(&luma, 6, 6, 2);
        assert_eq!(sum, 4.0);
    }

    #[test]
    fn sum_interior_matches_reference() {
        let mut luma = vec![0.0f32; 100];
        // Fill with deterministic values
        let mut seed = 0x5EEDu32;
        for v in &mut luma {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = ((seed >> 16) % 1024) as f32 / 1024.0 * 10.0;
        }

        let direct = sum_interior(&luma, 10, 10, 2);
        let reference = ref_sum_interior(&luma, 10, 10, 2);
        assert_eq!(direct.to_bits(), reference.to_bits());
    }

    #[test]
    fn sum_interior_too_small() {
        // 4x4 image with border=2 → no interior pixels
        let luma = vec![1.0f32; 16];
        let sum = sum_interior(&luma, 4, 4, 2);
        assert_eq!(sum, 0.0);
    }

    // ── sum_abs_deviation ─────────────────────────────────────────────────────

    #[test]
    fn sum_abs_deviation_basic() {
        // 5x5 all ones, tv_sum=1.0 → all deviations = 0
        let luma = vec![1.0f32; 25];
        let sum = sum_abs_deviation(&luma, 5, 5, 1.0, 2);
        assert_eq!(sum, 0.0);

        // 5x5 all twos, tv_sum=1.0 → interior deviation = 1.0
        let luma = vec![2.0f32; 25];
        let sum = sum_abs_deviation(&luma, 5, 5, 1.0, 2);
        assert_eq!(sum, 1.0);
    }

    #[test]
    fn sum_abs_deviation_matches_reference() {
        let mut luma = vec![0.0f32; 100];
        let mut seed = 0xDEADu32;
        for v in &mut luma {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = ((seed >> 16) % 1024) as f32 / 1024.0 * 10.0;
        }
        let tv_sum = 3.5f32;

        let direct = sum_abs_deviation(&luma, 10, 10, tv_sum, 2);
        let reference = ref_sum_abs_deviation(&luma, 10, 10, tv_sum, 2);
        assert_eq!(direct.to_bits(), reference.to_bits());
    }

    // ── FFI round-trip and guard tests ───────────────────────────────────────

    #[test]
    fn ffi_compute_luma_round_trip() {
        let mut seed = 0xCAFEBABE_u32;
        let mut image = vec![0u8; 256 * 4];
        for b in &mut image {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = ((seed >> 16) % 256) as u8;
        }

        let mut ffi_buf = vec![0.0f32; 256];
        let mut direct_buf = vec![0.0f32; 256];

        unsafe {
            darkroom_focuspeaking_compute_luma(image.as_ptr(), ffi_buf.as_mut_ptr(), 256);
        }
        compute_luma(&image, &mut direct_buf, 256);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_compute_luma_null_guard() {
        unsafe {
            darkroom_focuspeaking_compute_luma(std::ptr::null(), std::ptr::null_mut(), 10);
        }
    }

    #[test]
    fn ffi_compute_luma_zero_n_guard() {
        let image = vec![0u8; 16];
        let mut luma = vec![42.0f32; 4];
        unsafe {
            darkroom_focuspeaking_compute_luma(image.as_ptr(), luma.as_mut_ptr(), 0);
        }
        // Buffer untouched
        assert_eq!(luma, vec![42.0f32; 4]);
    }

    #[test]
    fn ffi_sum_interior_round_trip() {
        let mut luma = vec![0.0f32; 100];
        let mut seed = 0x1234u32;
        for v in &mut luma {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = ((seed >> 16) % 1024) as f32 / 1024.0 * 10.0;
        }

        let ffi_result = unsafe {
            darkroom_focuspeaking_sum_interior(luma.as_ptr(), 10, 10)
        };
        let direct_result = sum_interior(&luma, 10, 10, 2);
        assert_eq!(ffi_result.to_bits(), direct_result.to_bits());
    }

    #[test]
    fn ffi_sum_interior_null_guard() {
        let result = unsafe {
            darkroom_focuspeaking_sum_interior(std::ptr::null(), 10, 10)
        };
        assert_eq!(result, 0.0);
    }

    #[test]
    fn ffi_sum_abs_deviation_round_trip() {
        let mut luma = vec![0.0f32; 100];
        let mut seed = 0x5678u32;
        for v in &mut luma {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = ((seed >> 16) % 1024) as f32 / 1024.0 * 10.0;
        }
        let tv_sum = 4.2f32;

        let ffi_result = unsafe {
            darkroom_focuspeaking_sum_abs_deviation(luma.as_ptr(), 10, 10, tv_sum)
        };
        let direct_result = sum_abs_deviation(&luma, 10, 10, tv_sum, 2);
        assert_eq!(ffi_result.to_bits(), direct_result.to_bits());
    }

    #[test]
    fn ffi_sum_abs_deviation_null_guard() {
        let result = unsafe {
            darkroom_focuspeaking_sum_abs_deviation(std::ptr::null(), 10, 10, 1.0)
        };
        assert_eq!(result, 0.0);
    }
}
