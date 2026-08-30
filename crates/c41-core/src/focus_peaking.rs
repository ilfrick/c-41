//! Element-wise loops ported from `src/common/focus_peaking.h`.
//!
//! Five loops are ported here (m4-167 and m4-170):
//! - The luma computation loop (formerly at focus_peaking.h:86,
//!   `DT_OMP_FOR_SIMD`) that converts a 4-channel uint8 sRGB image to a
//!   single-channel float luma buffer using
//!   `sqrt(pow(c0, 4.4) + pow(c1, 4.4) + pow(c2, 4.4))`.
//! - The TV_sum reduction (formerly at :136) that sums luma values over the
//!   interior pixel region.
//! - The sigma reduction (formerly at :147) that sums
//!   `|luma[k] - TV_sum|` over the interior pixel region.
//! - The close/far laplacian gradient loop (formerly at :95, collapse(2)),
//!   including the `_get_indices` ring helper and `_laplacian` (whose
//!   `dt_fast_hypotf` is `sqrtf(x*x + y*y)` under the project's
//!   `-ffast-math` build, math.h:391, sqrtf branch at :393).
//! - The BGRA overlay painting loop (formerly at :144, collapse(2)) that
//!   thresholds the blurred gradient into yellow/green/blue/transparent.
//!
//! The original loops used `DT_OMP_FOR`/`DT_OMP_FOR_SIMD` (parallel+SIMD).
//! The Rust kernels are single-threaded sequential; LLVM's auto-vectorizer
//! provides SIMD at `-O3`, but multi-threaded parallelism is no longer used.
//! This matches the m4-161 `blend.rs` and m4-162 `imagebuf.rs` pattern.
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
//! - The gradients kernel has two FMA contraction surfaces vs the
//!   historical C binary: `x*x + y*y` inside `fast_hypot` (an `a*b+c`
//!   pattern, which GCC's `-ffp-contract=fast` may fuse into an FMA) and
//!   the `lap - 0.67f*(far - eps)` combination in `gradients`. Rust does
//!   not contract by default, so results may differ by a small (order-ULP)
//!   amount from the *historical* C reference — the C loop is deleted and
//!   the kernel is now the definition.
//! - The overlay kernel is pure integer/threshold work — no FP contraction
//!   surface; NaN gradients compare false against all three thresholds and
//!   paint transparent, exactly like C.
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

/// Build the 8-neighbour index ring at `(i, j)` for the given delta.
///
/// Port of `_get_indices` (focus_peaking.h:54). Layout:
/// `[NW, N, NE, W, E, SW, S, SE]`. `height` is unused exactly as in C.
fn get_indices(i: usize, j: usize, width: usize, delta: usize) -> [usize; 8] {
    let upper_line = (i - delta) * width;
    let center_line = i * width;
    let lower_line = (i + delta) * width;
    let left_row = j - delta;
    let right_row = j + delta;
    [
        upper_line + left_row, // north west
        upper_line + j,        // north
        upper_line + right_row, // north east
        center_line + left_row, // west
        center_line + right_row, // east
        lower_line + left_row, // south west
        lower_line + j,        // south
        lower_line + right_row, // south east
    ]
}

/// `dt_fast_hypotf` as compiled in this project (math.h:391, sqrtf branch at :393, under
/// `__FAST_MATH__`, which the Release build defines): `sqrtf(x*x + y*y)`.
fn fast_hypot(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// Gradient magnitude over the principal and diagonal neighbour pairs.
///
/// Port of `_laplacian` (focus_peaking.h:38):
/// `(hypot(E-W, S-N) + hypot(SE-NW, SW-NE)) / 2`. Reads go through
/// `read_clamped` (see its doc — C's bottom-right far-corner read is one
/// element past the allocation).
fn laplacian(luma: &[f32], index: &[usize; 8]) -> f32 {
    let r = |k: usize| read_clamped(luma, k);
    let l1 = fast_hypot(r(index[4]) - r(index[3]), r(index[6]) - r(index[1]));
    let l2 = fast_hypot(r(index[7]) - r(index[0]), r(index[5]) - r(index[2]));
    (l1 + l2) / 2.0f32
}

/// Read `luma[k]`, clamping the index to the last element.
///
/// C's far-ring read for the single interior corner pixel `(h-3, w-2)`
/// computes index `h*w` — one float past the allocation (a pre-existing
/// upstream quirk; the value read is indeterminate malloc padding). The
/// kernel substitutes the last element (`h*w - 1`, one element *before*
/// that address) there: a determinate value in place of C's garbage read.
/// The perturbed gradient does not stay confined to that one pixel — it
/// spreads through the `dt_box_mean` and `fast_surface_blur` passes that
/// follow — but its magnitude is tiny and the result is deterministic,
/// which C's was not. All in-bounds (including row-wrap) indices are
/// untouched.
#[inline]
fn read_clamped(luma: &[f32], k: usize) -> f32 {
    luma[k.min(luma.len() - 1)]
}

/// Compute the focus-peaking gradient buffer in place on `luma_ds`.
///
/// Port of the `DT_OMP_FOR(collapse(2))` loop at focus_peaking.h:95.
/// Border pixels (i < 2, i+2 >= height, j < 2, j+2 > width — behaviorally
/// equivalent to C's `i >= height-2` / `j > width-2` for all dimensions:
/// for dims ≤ 3 C's int wrap disables the other clause exactly where the
/// `i < 2`/`j < 2` clauses already fire) are zeroed; interior pixels get
/// `laplacian(close) - 0.67 * (laplacian(far) - 2^-8)`.
pub fn gradients(luma: &[f32], luma_ds: &mut [f32], width: usize, height: usize) {
    if width == 0 || height == 0 || luma.is_empty() {
        return;
    }
    let n = (width * height).min(luma_ds.len());
    for p in 0..n {
        let i = p / width;
        let j = p % width;
        let border = i < 2 || i + 2 >= height || j < 2 || j + 2 > width;
        if border {
            // ensure defined value for borders
            luma_ds[p] = 0.0f32;
        } else {
            let index_close = get_indices(i, j, width, 1);
            let index_far = get_indices(i, j, width, 2);
            luma_ds[p] = laplacian(luma, &index_close)
                - 0.67f32 * (laplacian(luma, &index_far) - 0.00390625f32);
        }
    }
}

/// Paint the focus-peaking BGRA overlay from the blurred gradient buffer.
///
/// Port of the `DT_OMP_FOR(collapse(2))` loop at focus_peaking.h:144.
/// `TV > six_sigma` → yellow, `> four_sigma` → green, `> two_sigma` → blue,
/// else transparent; a NaN gradient compares false against all three and
/// paints transparent, exactly like C.
pub fn paint_overlay(
    luma_ds: &[f32],
    focus_peaking: &mut [u8],
    width: usize,
    height: usize,
    six_sigma: f32,
    four_sigma: f32,
    two_sigma: f32,
) {
    if width == 0 || height == 0 || luma_ds.is_empty() {
        return;
    }
    const YELLOW: [u8; 4] = [0, 255, 255, 255]; // B, G, R, A
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [255, 0, 0, 255];
    let n = (width * height).min(luma_ds.len()).min(focus_peaking.len() / 4);
    for p in 0..n {
        let tv = luma_ds[p];
        let color = if tv > six_sigma {
            YELLOW
        } else if tv > four_sigma {
            GREEN
        } else if tv > two_sigma {
            BLUE
        } else {
            [0, 0, 0, 0]
        };
        focus_peaking[p * 4..p * 4 + 4].copy_from_slice(&color);
    }
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

/// # Safety
/// `luma` and `luma_ds` must each hold at least `width * height` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_focuspeaking_gradients(
    luma: *const f32,
    luma_ds: *mut f32,
    width: usize,
    height: usize,
) {
    if luma.is_null() || luma_ds.is_null() || width == 0 || height == 0
        || width > i32::MAX as usize || height > i32::MAX as usize
    {
        return;
    }
    let luma_slice = std::slice::from_raw_parts(luma, width * height);
    let ds_slice = std::slice::from_raw_parts_mut(luma_ds, width * height);
    gradients(luma_slice, ds_slice, width, height);
}

/// # Safety
/// `luma_ds` must hold at least `width * height` floats; `focus_peaking`
/// must hold at least `width * height * 4` bytes.
#[no_mangle]
pub unsafe extern "C" fn darkroom_focuspeaking_paint_overlay(
    luma_ds: *const f32,
    focus_peaking: *mut u8,
    width: usize,
    height: usize,
    six_sigma: f32,
    four_sigma: f32,
    two_sigma: f32,
) {
    if luma_ds.is_null() || focus_peaking.is_null() || width == 0 || height == 0
        || width > i32::MAX as usize || height > i32::MAX as usize
    {
        return;
    }
    let ds_slice = std::slice::from_raw_parts(luma_ds, width * height);
    let fp_slice = std::slice::from_raw_parts_mut(focus_peaking, width * height * 4);
    paint_overlay(
        ds_slice,
        fp_slice,
        width,
        height,
        six_sigma,
        four_sigma,
        two_sigma,
    );
}

// ── Independent reference implementations ──────────────────────────────────
//
// These recompute the same results with a slightly different code shape
// (named temporaries, different iteration form). The structural difference
// is modest — the real validation weight is carried by the known-value
// basic tests and the FFI round-trips below. This matches the established
// repo-wide reference-implementation pattern.

#[allow(dead_code)]
fn ref_gradients(luma: &[f32], luma_ds: &mut [f32], width: usize, height: usize) {
    if width == 0 || height == 0 || luma.is_empty() {
        return;
    }
    let n = (width * height).min(luma_ds.len());
    let read = |k: usize| luma[k.min(luma.len() - 1)];
    for p in 0..n {
        let i = p / width;
        let j = p % width;
        let is_border = (i < 2) || (i + 2 >= height) || (j < 2) || (j + 2 > width);
        if is_border {
            luma_ds[p] = 0.0;
            continue;
        }
        // Inline the ring construction with scalar arithmetic (row/col
        // offsets spelled out) instead of building index arrays.
        let c_up = (i - 1) * width;
        let c_mid = i * width;
        let c_lo = (i + 1) * width;
        let l = j - 1;
        let r = j + 1;
        let close = [c_up + l, c_up + j, c_up + r, c_mid + l, c_mid + r, c_lo + l, c_lo + j, c_lo + r];
        let f_up = (i - 2) * width;
        let f_mid = i * width;
        let f_lo = (i + 2) * width;
        let fl = j - 2;
        let fr = j + 2;
        let far = [f_up + fl, f_up + j, f_up + fr, f_mid + fl, f_mid + fr, f_lo + fl, f_lo + j, f_lo + fr];
        let close_l1 = fast_hypot(read(close[4]) - read(close[3]), read(close[6]) - read(close[1]));
        let close_l2 = fast_hypot(read(close[7]) - read(close[0]), read(close[5]) - read(close[2]));
        let far_l1 = fast_hypot(read(far[4]) - read(far[3]), read(far[6]) - read(far[1]));
        let far_l2 = fast_hypot(read(far[7]) - read(far[0]), read(far[5]) - read(far[2]));
        luma_ds[p] = (close_l1 + close_l2) / 2.0 - 0.67 * ((far_l1 + far_l2) / 2.0 - 0.00390625);
    }
}

#[allow(dead_code)]
fn ref_paint_overlay(
    luma_ds: &[f32],
    focus_peaking: &mut [u8],
    width: usize,
    height: usize,
    six_sigma: f32,
    four_sigma: f32,
    two_sigma: f32,
) {
    if width == 0 || height == 0 || luma_ds.is_empty() {
        return;
    }
    let n = (width * height).min(luma_ds.len()).min(focus_peaking.len() / 4);
    for p in 0..n {
        let tv = luma_ds[p];
        let bgra: [u8; 4] = if !(tv > two_sigma) {
            [0, 0, 0, 0]
        } else if !(tv > four_sigma) {
            [255, 0, 0, 255]
        } else if !(tv > six_sigma) {
            [0, 255, 0, 255]
        } else {
            [0, 255, 255, 255]
        };
        focus_peaking[p * 4..p * 4 + 4].copy_from_slice(&bgra);
    }
}

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
    use crate::masks::test_util::lcg_fill;

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

    // ── gradients ──────────────────────────────────────────────────────────────

    #[test]
    fn gradients_constant_image_uniform_interior() {
        // Constant luma: every neighbour difference is 0, so each interior
        // pixel is 0 - 0.67*(0 - 2^-8) = 0.67 * 2^-8; borders are 0.
        let luma = vec![0.5f32; 6 * 6];
        let mut ds = vec![f32::NAN; 6 * 6];
        gradients(&luma, &mut ds, 6, 6);
        let expected = 0.0f32 - 0.67f32 * (0.0f32 - 0.00390625f32);
        for i in 0..6 {
            for j in 0..6 {
                let v = ds[i * 6 + j];
                if i < 2 || i + 2 >= 6 || j < 2 || j + 2 > 6 {
                    assert_eq!(v, 0.0, "border at {i},{j}");
                } else {
                    assert_eq!(v, expected, "interior at {i},{j}");
                }
            }
        }
    }

    #[test]
    fn gradients_border_condition_matches_c_quirks() {
        // Interior is i in [2, h-3], j in [2, w-2] (C: `i >= h-2` border,
        // `j > w-2` border — note the inclusive j bound). For a 6x6 image
        // that is i in 2..=3, j in 2..=4. Constant luma gives interior
        // pixels the uniform 0.67*2^-8 value, borders 0.
        let luma = vec![0.5f32; 6 * 6];
        let mut ds = vec![-1.0f32; 6 * 6];
        gradients(&luma, &mut ds, 6, 6);
        for i in 0..6 {
            for j in 0..6 {
                let interior = (2..=3).contains(&i) && (2..=4).contains(&j);
                if interior {
                    assert!(ds[i * 6 + j] != 0.0, "interior at {i},{j} must be painted");
                } else {
                    assert_eq!(ds[i * 6 + j], 0.0, "border at {i},{j} must be zeroed");
                }
            }
        }
    }

    #[test]
    fn gradients_corner_far_index_read_is_clamped_not_oob() {
        // The (h-3, w-2) interior corner computes far SE index h*w in C —
        // one element past the allocation. The kernel must substitute the
        // last element (no panic, no OOB) and stay deterministic.
        let mut luma = vec![0.0f32; 6 * 6];
        lcg_fill(&mut luma, 0xF00D, 1.0);
        let mut ds = vec![f32::NAN; 6 * 6];
        gradients(&luma, &mut ds, 6, 6);
        assert!(ds.iter().all(|v| v.is_finite()));
        // Cross-check the corner pixel against the reference
        let mut ds_ref = vec![f32::NAN; 6 * 6];
        ref_gradients(&luma, &mut ds_ref, 6, 6);
        assert_eq!(ds[3 * 6 + 4].to_bits(), ds_ref[3 * 6 + 4].to_bits());
    }

    #[test]
    fn gradients_matches_reference_over_lcg() {
        let mut luma = vec![0.0f32; 16 * 16];
        lcg_fill(&mut luma, 0xAB12, 1.0);
        let mut direct = vec![0.0f32; 16 * 16];
        let mut reference = vec![0.0f32; 16 * 16];
        gradients(&luma, &mut direct, 16, 16);
        ref_gradients(&luma, &mut reference, 16, 16);
        assert_eq!(direct, reference);
    }

    #[test]
    fn gradients_odd_nonsquare_matches_reference() {
        // Odd, non-square dims pin the asymmetric border bounds
        // (i+2 >= h vs j+2 > w) and exercise the far-ring row-wrap at
        // j = w-2 against the reference.
        let mut luma = vec![0.0f32; 7 * 5];
        lcg_fill(&mut luma, 0x5E17, 1.0);
        let mut direct = vec![0.0f32; 7 * 5];
        let mut reference = vec![0.0f32; 7 * 5];
        gradients(&luma, &mut direct, 7, 5);
        ref_gradients(&luma, &mut reference, 7, 5);
        assert_eq!(direct, reference);
        // Interior for 7x5: i in 2..=2, j in 2..=5 — exactly one row
        for j in 0..7 {
            let interior = (2..=5).contains(&j);
            assert_eq!(direct[2 * 7 + j] != 0.0, interior, "at 2,{j}");
        }
    }

    // ── paint_overlay ──────────────────────────────────────────────────────────

    #[test]
    fn paint_overlay_threshold_bands() {
        // six=10, four=7, two=4: pick one TV per band plus the boundaries
        let luma_ds = vec![11.0f32, 10.0, 8.0, 7.0, 5.0, 4.0, 0.0, f32::NAN];
        let mut fp = vec![9u8; 8 * 4];
        paint_overlay(&luma_ds, &mut fp, 8, 1, 10.0, 7.0, 4.0);
        // strict > comparisons: 10.0 is NOT > six → green; 7.0 → blue; 4.0 → off
        let expected: [[u8; 4]; 8] = [
            [0, 255, 255, 255], // yellow (11 > 10)
            [0, 255, 0, 255],   // green (10 ≤ 10)
            [0, 255, 0, 255],   // green (8 > 7)
            [255, 0, 0, 255],   // blue (7 ≤ 7)
            [255, 0, 0, 255],   // blue (5 > 4)
            [0, 0, 0, 0],       // off (4 ≤ 4)
            [0, 0, 0, 0],       // off
            [0, 0, 0, 0],       // NaN compares false → off
        ];
        for p in 0..8 {
            assert_eq!(&fp[p * 4..p * 4 + 4], &expected[p], "pixel {p}");
        }
    }

    #[test]
    fn paint_overlay_matches_reference_over_lcg() {
        let mut luma_ds = vec![0.0f32; 256];
        lcg_fill(&mut luma_ds, 0x77AA, 1.0);
        // A few NaNs to pin the NaN path against the reference too
        luma_ds[3] = f32::NAN;
        luma_ds[100] = f32::NAN;
        let mut direct = vec![9u8; 256 * 4];
        let mut reference = vec![9u8; 256 * 4];
        paint_overlay(&luma_ds, &mut direct, 256, 1, 0.6, 0.4, 0.2);
        ref_paint_overlay(&luma_ds, &mut reference, 256, 1, 0.6, 0.4, 0.2);
        assert_eq!(direct, reference);
    }

    // ── FFI round-trip and guard tests (m4-170 kernels) ─────────────────────────

    #[test]
    fn ffi_gradients_round_trip() {
        let mut luma = vec![0.0f32; 12 * 12];
        lcg_fill(&mut luma, 0x9911, 1.0);
        let mut ffi_buf = vec![f32::NAN; 12 * 12];
        let mut direct_buf = vec![f32::NAN; 12 * 12];
        unsafe {
            darkroom_focuspeaking_gradients(luma.as_ptr(), ffi_buf.as_mut_ptr(), 12, 12);
        }
        gradients(&luma, &mut direct_buf, 12, 12);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_gradients_guards() {
        unsafe {
            darkroom_focuspeaking_gradients(std::ptr::null(), std::ptr::null_mut(), 4, 4);
        }
        let luma = vec![1.0f32; 4];
        let mut ds = vec![1.0f32; 4];
        unsafe {
            darkroom_focuspeaking_gradients(luma.as_ptr(), ds.as_mut_ptr(), 0, 4);
            darkroom_focuspeaking_gradients(
                luma.as_ptr(),
                ds.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                4,
            );
        }
        assert_eq!(ds, vec![1.0f32; 4]); // untouched
    }

    #[test]
    fn ffi_paint_overlay_round_trip() {
        let mut luma_ds = vec![0.0f32; 64];
        lcg_fill(&mut luma_ds, 0x88BB, 1.0);
        let mut ffi_buf = vec![9u8; 64 * 4];
        let mut direct_buf = vec![9u8; 64 * 4];
        unsafe {
            darkroom_focuspeaking_paint_overlay(
                luma_ds.as_ptr(),
                ffi_buf.as_mut_ptr(),
                64,
                1,
                0.6,
                0.4,
                0.2,
            );
        }
        paint_overlay(&luma_ds, &mut direct_buf, 64, 1, 0.6, 0.4, 0.2);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_paint_overlay_guards() {
        unsafe {
            darkroom_focuspeaking_paint_overlay(
                std::ptr::null(),
                std::ptr::null_mut(),
                4,
                1,
                1.0,
                1.0,
                1.0,
            );
        }
        let luma_ds = vec![1.0f32; 4];
        let mut fp = vec![9u8; 16];
        unsafe {
            // zero-dim and oversized-dim guards must leave the buffer alone
            darkroom_focuspeaking_paint_overlay(
                luma_ds.as_ptr(),
                fp.as_mut_ptr(),
                0,
                1,
                1.0,
                1.0,
                1.0,
            );
            darkroom_focuspeaking_paint_overlay(
                luma_ds.as_ptr(),
                fp.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                1,
                1.0,
                1.0,
                1.0,
            );
        }
        assert_eq!(fp, vec![9u8; 16]); // untouched by guards
        // A valid call paints through the FFI: tv=1.0 is not > any of the
        // 1.0 thresholds → every pixel transparent
        unsafe {
            darkroom_focuspeaking_paint_overlay(
                luma_ds.as_ptr(),
                fp.as_mut_ptr(),
                4,
                1,
                1.0,
                1.0,
                1.0,
            );
        }
        assert_eq!(fp, vec![0u8; 16]);
    }
}
