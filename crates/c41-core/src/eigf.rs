//! Kernels ported from `src/common/eigf.h` (the Exposure-Independent
//! Guided Filter). Six loops are ported here (m4-165 and m4-171):
//! - The variance/covariance correction loops of `eigf_variance_analysis`
//!   and `eigf_variance_analysis_no_mask` (m4-165, formerly at :115/:160).
//! - The pack + min/max reduction loops of the same two functions
//!   (m4-171, formerly at :88/:137) — packing `{g, g², m, m·g}` (4ch) or
//!   `{g, g²}` (2ch) while tracking per-channel ranges for
//!   `dt_gaussian_init`.
//! - The `eigf_blending` and `eigf_blending_no_mask` element-wise loops
//!   (m4-171, formerly at :168/:201).
//!
//! The original C code used `DT_OMP_FOR`/`DT_OMP_FOR_SIMD` (parallel+SIMD).
//! The Rust kernels are single-threaded sequential loops; LLVM's auto-vectorizer
//! provides SIMD at `-O3`, but multi-threaded parallelism is no longer used.
//! This matches the m4-161 `blend.rs` and m4-162 `imagebuf.rs` pattern.
//!
//! Bit-exactness notes:
//! - The expressions `ch -= avg * avg` and `ch -= avg * other` are multiply-subtract
//!   patterns susceptible to FMA contraction. The C source is compiled with GCC 9's
//!   default `-ffp-contract=fast` for C99+ at `-O3`, so the original loop may contract
//!   `a*b + c` to FMA. The Rust kernels use separate multiply and subtract
//!   operations (no `-fp-contract=fast` in the release profile). On targets with
//!   FMA hardware this can produce ≤1 ULP difference between the old C path and
//!   the new Rust FFI path on some pixels. This is the same trade-off accepted for
//!   `linear_blend` in `imagebuf.rs` and is not visually significant for variance
//!   estimation in a guided filter.
//! - The min/max reductions are exact and **order-independent for NaN-free
//!   data** (min/max are associative and commutative there), so the
//!   sequential port is bit-identical to the C loop at any OpenMP thread
//!   count — unlike sum reductions. With NaN data the ternary is neither
//!   commutative nor associative (see the next bullet), so the parallel
//!   result was partitioning-dependent — and under `-ffinite-math-only`
//!   (pragma TUs) GCC may legally compile the ternary into a commutative
//!   min, so the historical binary's NaN behaviour was compiler-defined
//!   anyway. Signed zeros are a second-order caveat: the ternary returns
//!   the second operand when `+0 < -0` compares false.
//! - The C code's MIN/MAX are **GLib ternary macros** (`((a) < (b) ? (a) : (b))`
//!   etc., gmacros.h:933/936), NOT `fminf`/`fmaxf`. Their NaN behaviour differs
//!   from both fminf and Rust's `f32::min`: `MIN(acc, NaN)` yields NaN (the
//!   comparison is false so the macro takes `b`), and once the accumulator is
//!   NaN, `MIN(NaN, next)` yields `next` — a mid-stream NaN is absorbed and the
//!   accumulator resumes from the next value. The kernels replicate the
//!   ternaries exactly (`glib_min`/`glib_max`); the quirk is pinned by tests.
//!   NaN guide/mask data is anomalous for this filter (luminance masks).
//! - `eigf.h` has no `#pragma GCC optimize("fast-math")`. Only `finite-math-only`
//!   (via `extra_optimizations.h` in the caller) is active. `finite-math-only`
//!   does NOT enable FMA contraction; the residual risk comes solely from GCC's
//!   default `-ffp-contract` policy.
//! - OOM path (both variance-analysis functions): if `dt_alloc_align_float`
//!   returns NULL, the old C crashed inside the pack loop; the new C skips
//!   packing silently (FFI null guard) and crashes later inside the Gaussian
//!   blur instead. Both are undefined behaviour; noted, not fixed — matching
//!   the upstream posture.
//! - The blending loops are dense multiply-add chains (`avg_g * image[k]`,
//!   `image[k] * a + b`, `b = avg_m - a * avg_g`) — the same FMA-contraction
//!   class as above applies vs the historical C binary. Their `fmaxf(x, MIN_FLOAT)`
//!   and `fmaxf(x, 1E-6)` map to the NaN-ignoring `f32::max` (matching standard
//!   libm; the TU-dependent vectorized-max caveat documented in
//!   `fast_guided_filter.rs` applies to the removed C paths).

/// Variance and covariance correction for the 4-channel eigf variance analysis.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `eigf_variance_analysis` (eigf.h:115).
/// After Gaussian blur, `buf` holds 4 channels per element:
/// ch0 = E[guide], ch1 = E[guide²], ch2 = E[mask], ch3 = E[guide·mask].
///
/// Correction:
/// - ch1 -= ch0² → variance = E[g²] - E[g]²
/// - ch3 -= ch0*ch2 → covariance = E[mg] - E[m]·E[g]
pub fn eigf_variance_correct_4c(buf: &mut [f32], n_elements: usize) {
    let m = n_elements.min(buf.len() / 4);
    for k in 0..m {
        let avg = buf[k * 4];
        buf[k * 4 + 1] -= avg * avg;
        buf[k * 4 + 3] -= avg * buf[k * 4 + 2];
    }
}

/// Variance correction for the 2-channel eigf variance analysis (no mask).
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `eigf_variance_analysis_no_mask`
/// (eigf.h:160). After Gaussian blur, `buf` holds 2 channels per element:
/// ch0 = E[guide], ch1 = E[guide²].
///
/// Correction:
/// - ch1 -= ch0² → variance = E[g²] - E[g]²
pub fn eigf_variance_correct_2c(buf: &mut [f32], n_elements: usize) {
    let m = n_elements.min(buf.len() / 2);
    for k in 0..m {
        let avg = buf[k * 2];
        buf[k * 2 + 1] -= avg * avg;
    }
}

/// `MIN_FLOAT` from `fast_guided_filter.h` (`exp2f(-16.0f)` = 2^-16, exactly
/// representable in f32). `eigf.h` includes that header.
const MIN_FLOAT: f32 = 1.52587890625e-5_f32;

/// GLib's `MIN(a, b)` macro: `((a) < (b) ? (a) : (b))` — a ternary, NOT
/// `fminf`. Replicated exactly (see the module docs for the NaN semantics).
#[inline]
fn glib_min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// GLib's `MAX(a, b)` macro: `((a) > (b) ? (a) : (b))` — a ternary, NOT
/// `fmaxf`.
#[inline]
fn glib_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Pack the eigf variance-analysis input and track its value ranges.
///
/// Port of the `DT_OMP_FOR` pack + min/max reduction loop in
/// `eigf_variance_analysis` (formerly at eigf.h:88). Per element, `in` gets
/// `{g, g², m, m·g}` (NOTE: a different channel order than
/// `fast_guided_filter::pack_variance_4c`'s `{g, m, g², g·m}` — the blur and
/// the variance-correct kernel read it in this order). Simultaneously
/// accumulates the per-channel min/max used to configure `dt_gaussian_init`.
///
/// `min_out`/`max_out` receive `{g, g², m, m·g}` ranges with C's initial
/// values (mins start at 1e7, maxes at 0.0). Min/max reductions are exact
/// and order-independent for NaN-free data, so unlike sum reductions the
/// sequential port is bit-identical to any OpenMP thread count (the NaN
/// behaviour of the GLib ternaries — see `glib_min` — breaks that, but NaN
/// guide/mask data is anomalous for this filter).
pub fn pack_variance_minmax_4c(
    input: &mut [f32],
    guide: &[f32],
    mask: &[f32],
    n_elements: usize,
    min_out: &mut [f32],
    max_out: &mut [f32],
) {
    let m = n_elements.min(guide.len()).min(mask.len()).min(input.len() / 4);
    let (mut ming, mut ming2, mut minm, mut minmg) =
        (10000000.0f32, 10000000.0f32, 10000000.0f32, 10000000.0f32);
    let (mut maxg, mut maxg2, mut maxm, mut maxmg) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for k in 0..m {
        let pixelg = guide[k];
        let pixelm = mask[k];
        let pixelg2 = pixelg * pixelg;
        let pixelmg = pixelm * pixelg;
        input[k * 4] = pixelg;
        input[k * 4 + 1] = pixelg2;
        input[k * 4 + 2] = pixelm;
        input[k * 4 + 3] = pixelmg;
        ming = glib_min(ming, pixelg);
        maxg = glib_max(maxg, pixelg);
        minm = glib_min(minm, pixelm);
        maxm = glib_max(maxm, pixelm);
        ming2 = glib_min(ming2, pixelg2);
        maxg2 = glib_max(maxg2, pixelg2);
        minmg = glib_min(minmg, pixelmg);
        maxmg = glib_max(maxmg, pixelmg);
    }
    if min_out.len() >= 4 {
        min_out[0] = ming;
        min_out[1] = ming2;
        min_out[2] = minm;
        min_out[3] = minmg;
    }
    if max_out.len() >= 4 {
        max_out[0] = maxg;
        max_out[1] = maxg2;
        max_out[2] = maxm;
        max_out[3] = maxmg;
    }
}

/// 2-channel (guide == mask) variant of the pack + min/max loop.
///
/// Port of the `DT_OMP_FOR` loop in `eigf_variance_analysis_no_mask`
/// (formerly at eigf.h:137): `in` gets `{g, g²}`, ranges likewise.
pub fn pack_variance_minmax_2c(
    input: &mut [f32],
    guide: &[f32],
    n_elements: usize,
    min_out: &mut [f32],
    max_out: &mut [f32],
) {
    let m = n_elements.min(guide.len()).min(input.len() / 2);
    let (mut ming, mut ming2) = (10000000.0f32, 10000000.0f32);
    let (mut maxg, mut maxg2) = (0.0f32, 0.0f32);
    for k in 0..m {
        let pixelg = guide[k];
        let pixelg2 = pixelg * pixelg;
        input[2 * k] = pixelg;
        input[2 * k + 1] = pixelg2;
        ming = glib_min(ming, pixelg);
        maxg = glib_max(maxg, pixelg);
        ming2 = glib_min(ming2, pixelg2);
        maxg2 = glib_max(maxg2, pixelg2);
    }
    if min_out.len() >= 2 {
        min_out[0] = ming;
        min_out[1] = ming2;
    }
    if max_out.len() >= 2 {
        max_out[0] = maxg;
        max_out[1] = maxg2;
    }
}

/// Exposure-independent guided-filter blending (guide ≠ mask).
///
/// Port of the `DT_OMP_FOR` loop in `eigf_blending` (formerly at eigf.h:168).
/// `av` holds the blurred/corrected moments `{E[g], var_g, E[m], covar_mg}`
/// (4 floats per element); `filter` is `DT_GF_BLENDING_LINEAR` (0) or
/// `DT_GF_BLENDING_GEOMEAN` (anything else, matching C's `else`).
pub fn eigf_blending(
    image: &mut [f32],
    mask: &[f32],
    av: &[f32],
    n_elements: usize,
    filter: i32,
    feathering: f32,
) {
    let m = n_elements.min(image.len()).min(mask.len()).min(av.len() / 4);
    for k in 0..m {
        let avg_g = av[k * 4];
        let avg_m = av[k * 4 + 2];
        let var_g = av[k * 4 + 1];
        let covar_mg = av[k * 4 + 3];
        let norm_g = (avg_g * image[k]).max(1e-6_f32);
        let norm_m = (avg_m * mask[k]).max(1e-6_f32);
        let normalized_var_guide = var_g / norm_g;
        let normalized_covar = covar_mg / (norm_g * norm_m).sqrt();
        let a = normalized_covar / (normalized_var_guide + feathering);
        let b = avg_m - a * avg_g;
        if filter == 0 {
            // DT_GF_BLENDING_LINEAR
            image[k] = (image[k] * a + b).max(MIN_FLOAT);
        } else {
            // DT_GF_BLENDING_GEOMEAN
            image[k] *= (image[k] * a + b).max(MIN_FLOAT);
            image[k] = image[k].sqrt();
        }
    }
}

/// Exposure-independent guided-filter blending, guide == mask variant.
///
/// Port of the `DT_OMP_FOR` loop in `eigf_blending_no_mask` (formerly at
/// eigf.h:201). `av` holds `{E[g], var_g}` (2 floats per element).
pub fn eigf_blending_no_mask(
    image: &mut [f32],
    av: &[f32],
    n_elements: usize,
    filter: i32,
    feathering: f32,
) {
    let m = n_elements.min(image.len()).min(av.len() / 2);
    for k in 0..m {
        let avg_g = av[k * 2];
        let var_g = av[k * 2 + 1];
        let norm_g = (avg_g * image[k]).max(1e-6_f32);
        let normalized_var_guide = var_g / norm_g;
        let a = normalized_var_guide / (normalized_var_guide + feathering);
        let b = avg_g - a * avg_g;
        if filter == 0 {
            // DT_GF_BLENDING_LINEAR
            image[k] = (image[k] * a + b).max(MIN_FLOAT);
        } else {
            // DT_GF_BLENDING_GEOMEAN
            image[k] *= (image[k] * a + b).max(MIN_FLOAT);
            image[k] = image[k].sqrt();
        }
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `buf` must hold at least `n_elements * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_eigf_variance_correct_4c(
    buf: *mut f32,
    n_elements: usize,
) {
    if buf.is_null() || n_elements == 0 || n_elements > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n_elements * 4);
    eigf_variance_correct_4c(buf_slice, n_elements);
}

/// # Safety
/// `buf` must hold at least `n_elements * 2` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_eigf_variance_correct_2c(
    buf: *mut f32,
    n_elements: usize,
) {
    if buf.is_null() || n_elements == 0 || n_elements > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n_elements * 2);
    eigf_variance_correct_2c(buf_slice, n_elements);
}

/// # Safety
/// `input` must hold at least `n_elements * 4` floats; `guide` and `mask`
/// at least `n_elements` each; `min` and `max` at least 4 floats each.
#[no_mangle]
pub unsafe extern "C" fn darkroom_eigf_pack_variance_minmax_4c(
    input: *mut f32,
    guide: *const f32,
    mask: *const f32,
    n_elements: usize,
    min: *mut f32,
    max: *mut f32,
) {
    if input.is_null() || guide.is_null() || mask.is_null() || min.is_null() || max.is_null()
        || n_elements == 0 || n_elements > i32::MAX as usize
    {
        return;
    }
    let input_slice = std::slice::from_raw_parts_mut(input, n_elements * 4);
    let guide_slice = std::slice::from_raw_parts(guide, n_elements);
    let mask_slice = std::slice::from_raw_parts(mask, n_elements);
    let min_slice = std::slice::from_raw_parts_mut(min, 4);
    let max_slice = std::slice::from_raw_parts_mut(max, 4);
    pack_variance_minmax_4c(input_slice, guide_slice, mask_slice, n_elements, min_slice, max_slice);
}

/// # Safety
/// `input` must hold at least `n_elements * 2` floats; `guide` at least
/// `n_elements`; `min` and `max` at least 2 floats each.
#[no_mangle]
pub unsafe extern "C" fn darkroom_eigf_pack_variance_minmax_2c(
    input: *mut f32,
    guide: *const f32,
    n_elements: usize,
    min: *mut f32,
    max: *mut f32,
) {
    if input.is_null() || guide.is_null() || min.is_null() || max.is_null()
        || n_elements == 0 || n_elements > i32::MAX as usize
    {
        return;
    }
    let input_slice = std::slice::from_raw_parts_mut(input, n_elements * 2);
    let guide_slice = std::slice::from_raw_parts(guide, n_elements);
    let min_slice = std::slice::from_raw_parts_mut(min, 2);
    let max_slice = std::slice::from_raw_parts_mut(max, 2);
    pack_variance_minmax_2c(input_slice, guide_slice, n_elements, min_slice, max_slice);
}

/// # Safety
/// `image` and `mask` must hold at least `n_elements` floats each; `av` at
/// least `n_elements * 4` floats. `filter` is the
/// `dt_iop_guided_filter_blending_t` enum value (0 = linear).
#[no_mangle]
pub unsafe extern "C" fn darkroom_eigf_blending(
    image: *mut f32,
    mask: *const f32,
    av: *const f32,
    n_elements: usize,
    filter: i32,
    feathering: f32,
) {
    if image.is_null() || mask.is_null() || av.is_null()
        || n_elements == 0 || n_elements > i32::MAX as usize
    {
        return;
    }
    let image_slice = std::slice::from_raw_parts_mut(image, n_elements);
    let mask_slice = std::slice::from_raw_parts(mask, n_elements);
    let av_slice = std::slice::from_raw_parts(av, n_elements * 4);
    eigf_blending(image_slice, mask_slice, av_slice, n_elements, filter, feathering);
}

/// # Safety
/// `image` must hold at least `n_elements` floats; `av` at least
/// `n_elements * 2` floats. `filter` is the
/// `dt_iop_guided_filter_blending_t` enum value (0 = linear).
#[no_mangle]
pub unsafe extern "C" fn darkroom_eigf_blending_no_mask(
    image: *mut f32,
    av: *const f32,
    n_elements: usize,
    filter: i32,
    feathering: f32,
) {
    if image.is_null() || av.is_null()
        || n_elements == 0 || n_elements > i32::MAX as usize
    {
        return;
    }
    let image_slice = std::slice::from_raw_parts_mut(image, n_elements);
    let av_slice = std::slice::from_raw_parts(av, n_elements * 2);
    eigf_blending_no_mask(image_slice, av_slice, n_elements, filter, feathering);
}

// ── Independent reference implementations for bit-exactness tests ─────────────
//
// Structural divergence varies by family (all mathematically identical, and
// identical in FP evaluation order where the kernel's order is load-bearing):
// - The variance-correct refs compute products into temporaries first, then
//   subtract — a genuinely different evaluation order.
// - The pack refs use f32::min/f32::max folds instead of the kernel's GLib
//   ternary replicas (identical on the non-NaN LCG data; the ternary's NaN
//   quirk is pinned by dedicated known-value tests instead).
// - The blending refs keep the kernel's evaluation order and restructure
//   only the control flow (if/else clamps in place of `.max`).

#[allow(dead_code)]
fn ref_eigf_variance_correct_4c(buf: &mut [f32], n_elements: usize) {
    let m = n_elements.min(buf.len() / 4);
    for k in 0..m {
        let avg = buf[k * 4];
        let mask_avg = buf[k * 4 + 2];
        let var_term = avg * avg;
        let covar_term = avg * mask_avg;
        buf[k * 4 + 1] -= var_term;
        buf[k * 4 + 3] -= covar_term;
    }
}

#[allow(dead_code)]
fn ref_eigf_variance_correct_2c(buf: &mut [f32], n_elements: usize) {
    let m = n_elements.min(buf.len() / 2);
    for k in 0..m {
        let avg = buf[k * 2];
        let var_term = avg * avg;
        buf[k * 2 + 1] -= var_term;
    }
}

#[allow(dead_code)]
fn ref_pack_variance_minmax_4c(
    input: &mut [f32],
    guide: &[f32],
    mask: &[f32],
    n_elements: usize,
    min_out: &mut [f32],
    max_out: &mut [f32],
) {
    // Uses f32::min/f32::max (NaN-ignoring) instead of the kernel's GLib
    // ternary replicas — identical on the non-NaN LCG data these refs run
    // over; the NaN quirk of the ternary is pinned by dedicated tests.
    let m = n_elements.min(guide.len()).min(mask.len()).min(input.len() / 4);
    let gs: Vec<f32> = guide[..m].to_vec();
    let ms: Vec<f32> = mask[..m].to_vec();
    for k in 0..m {
        input[k * 4] = gs[k];
        input[k * 4 + 1] = gs[k] * gs[k];
        input[k * 4 + 2] = ms[k];
        input[k * 4 + 3] = ms[k] * gs[k];
    }
    let g2s: Vec<f32> = gs.iter().map(|g| g * g).collect();
    let mgs: Vec<f32> = gs.iter().zip(ms.iter()).map(|(g, m)| m * g).collect();
    let mins = [
        gs.iter().copied().fold(10000000.0f32, f32::min),
        g2s.iter().copied().fold(10000000.0f32, f32::min),
        ms.iter().copied().fold(10000000.0f32, f32::min),
        mgs.iter().copied().fold(10000000.0f32, f32::min),
    ];
    let maxs = [
        gs.iter().copied().fold(0.0f32, f32::max),
        g2s.iter().copied().fold(0.0f32, f32::max),
        ms.iter().copied().fold(0.0f32, f32::max),
        mgs.iter().copied().fold(0.0f32, f32::max),
    ];
    min_out[..4].copy_from_slice(&mins);
    max_out[..4].copy_from_slice(&maxs);
}

#[allow(dead_code)]
fn ref_pack_variance_minmax_2c(
    input: &mut [f32],
    guide: &[f32],
    n_elements: usize,
    min_out: &mut [f32],
    max_out: &mut [f32],
) {
    let m = n_elements.min(guide.len()).min(input.len() / 2);
    let gs: Vec<f32> = guide[..m].to_vec();
    for k in 0..m {
        input[2 * k] = gs[k];
        input[2 * k + 1] = gs[k] * gs[k];
    }
    let g2s: Vec<f32> = gs.iter().map(|g| g * g).collect();
    min_out[0] = gs.iter().copied().fold(10000000.0f32, f32::min);
    min_out[1] = g2s.iter().copied().fold(10000000.0f32, f32::min);
    max_out[0] = gs.iter().copied().fold(0.0f32, f32::max);
    max_out[1] = g2s.iter().copied().fold(0.0f32, f32::max);
}

#[allow(dead_code)]
fn ref_eigf_blending(
    image: &mut [f32],
    mask: &[f32],
    av: &[f32],
    n_elements: usize,
    filter: i32,
    feathering: f32,
) {
    // Same FP evaluation order as the kernel, restructured: the fmax clamps
    // become if/else (NaN-propagating, but identical on the non-NaN LCG
    // data this ref runs over — the kernel's NaN-ignoring `.max` matches
    // C's `fmaxf` and is pinned by dedicated tests).
    let m = n_elements.min(image.len()).min(mask.len()).min(av.len() / 4);
    for k in 0..m {
        let avg_g = av[k * 4];
        let avg_m = av[k * 4 + 2];
        let var_g = av[k * 4 + 1];
        let covar_mg = av[k * 4 + 3];
        let raw_g = avg_g * image[k];
        let raw_m = avg_m * mask[k];
        let norm_g = if raw_g < 1e-6_f32 { 1e-6_f32 } else { raw_g };
        let norm_m = if raw_m < 1e-6_f32 { 1e-6_f32 } else { raw_m };
        let normalized_var_guide = var_g / norm_g;
        let normalized_covar = covar_mg / (norm_g * norm_m).sqrt();
        let a = normalized_covar / (normalized_var_guide + feathering);
        let b = avg_m - a * avg_g;
        let blended = image[k] * a + b;
        let floored = if blended < MIN_FLOAT { MIN_FLOAT } else { blended };
        if filter == 0 {
            image[k] = floored;
        } else {
            let product = image[k] * floored;
            image[k] = product.sqrt();
        }
    }
}

#[allow(dead_code)]
fn ref_eigf_blending_no_mask(
    image: &mut [f32],
    av: &[f32],
    n_elements: usize,
    filter: i32,
    feathering: f32,
) {
    let m = n_elements.min(image.len()).min(av.len() / 2);
    for k in 0..m {
        let avg_g = av[k * 2];
        let var_g = av[k * 2 + 1];
        let raw_g = avg_g * image[k];
        let norm_g = if raw_g < 1e-6_f32 { 1e-6_f32 } else { raw_g };
        let normalized_var_guide = var_g / norm_g;
        let a = normalized_var_guide / (normalized_var_guide + feathering);
        let b = avg_g - a * avg_g;
        let blended = image[k] * a + b;
        let floored = if blended < MIN_FLOAT { MIN_FLOAT } else { blended };
        if filter == 0 {
            image[k] = floored;
        } else {
            let product = image[k] * floored;
            image[k] = product.sqrt();
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    // ── eigf_variance_correct_4c ─────────────────────────────────────────────────

    #[test]
    fn variance_correct_4c_basic() {
        // ch0 = avg_guide, ch1 = E[g²], ch2 = avg_mask, ch3 = E[mg]
        let mut buf = vec![0.0f32; 8]; // 2 elements × 4 channels
        // element 0: guide=2.0, guide²=5.0, mask=3.0, mg=7.0
        buf[0] = 2.0;
        buf[1] = 5.0;
        buf[2] = 3.0;
        buf[3] = 7.0;
        // element 1: guide=1.0, guide²=2.0, mask=4.0, mg=6.0
        buf[4] = 1.0;
        buf[5] = 2.0;
        buf[6] = 4.0;
        buf[7] = 6.0;

        eigf_variance_correct_4c(&mut buf, 2);

        // var = E[g²] - E[g]²: 5 - 4 = 1, 2 - 1 = 1
        assert_eq!(buf[1], 1.0);
        assert_eq!(buf[5], 1.0);
        // covar = E[mg] - E[m]*E[g]: 7 - 6 = 1, 6 - 4 = 2
        assert_eq!(buf[3], 1.0);
        assert_eq!(buf[7], 2.0);
        // guide and mask unchanged
        assert_eq!(buf[0], 2.0);
        assert_eq!(buf[2], 3.0);
        assert_eq!(buf[4], 1.0);
        assert_eq!(buf[6], 4.0);
    }

    #[test]
    fn variance_correct_4c_matches_reference_over_lcg() {
        let mut buf = vec![0.0f32; 256 * 4];
        lcg_fill(&mut buf, 0xABCD, 10.0);

        let mut direct = buf.clone();
        let mut reference = buf.clone();

        eigf_variance_correct_4c(&mut direct, 256);
        ref_eigf_variance_correct_4c(&mut reference, 256);

        assert_eq!(direct, reference);
    }

    #[test]
    fn variance_correct_4c_zero_variance() {
        // When avg_guide = 0, variance = E[g²] - 0 = E[g²]
        let mut buf = vec![0.0f32; 4];
        buf[0] = 0.0; // avg_guide
        buf[1] = 5.0; // E[g²]
        buf[2] = 1.0; // avg_mask
        buf[3] = 3.0; // E[mg]

        eigf_variance_correct_4c(&mut buf, 1);

        assert_eq!(buf[1], 5.0); // var = 5 - 0 = 5
        assert_eq!(buf[3], 3.0); // covar = 3 - 0*1 = 3
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[2], 1.0);
    }

    // ── eigf_variance_correct_2c ─────────────────────────────────────────────────

    #[test]
    fn variance_correct_2c_basic() {
        let mut buf = vec![0.0f32; 4]; // 2 elements × 2 channels
        // element 0: guide=2.0, guide²=5.0
        buf[0] = 2.0;
        buf[1] = 5.0;
        // element 1: guide=1.0, guide²=2.0
        buf[2] = 1.0;
        buf[3] = 2.0;

        eigf_variance_correct_2c(&mut buf, 2);

        // var = E[g²] - E[g]²: 5 - 4 = 1, 2 - 1 = 1
        assert_eq!(buf[1], 1.0);
        assert_eq!(buf[3], 1.0);
        // guide unchanged
        assert_eq!(buf[0], 2.0);
        assert_eq!(buf[2], 1.0);
    }

    #[test]
    fn variance_correct_2c_matches_reference_over_lcg() {
        let mut buf = vec![0.0f32; 256 * 2];
        lcg_fill(&mut buf, 0xBEEF, 10.0);

        let mut direct = buf.clone();
        let mut reference = buf.clone();

        eigf_variance_correct_2c(&mut direct, 256);
        ref_eigf_variance_correct_2c(&mut reference, 256);

        assert_eq!(direct, reference);
    }

    #[test]
    fn variance_correct_2c_zero_variance() {
        let mut buf = vec![0.0f32; 2];
        buf[0] = 0.0; // avg_guide
        buf[1] = 7.0; // E[g²]

        eigf_variance_correct_2c(&mut buf, 1);

        assert_eq!(buf[1], 7.0); // var = 7 - 0 = 7
        assert_eq!(buf[0], 0.0);
    }

    // ── FFI round-trip and guard tests ──────────────────────────────────────────

    #[test]
    fn ffi_variance_correct_4c_round_trip() {
        let mut buf = vec![0.0f32; 256 * 4];
        lcg_fill(&mut buf, 0x1234, 10.0);

        let mut ffi_buf = buf.clone();
        let mut direct_buf = buf.clone();

        unsafe {
            darkroom_eigf_variance_correct_4c(ffi_buf.as_mut_ptr(), 256);
        }
        eigf_variance_correct_4c(&mut direct_buf, 256);

        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_variance_correct_4c_null_guard() {
        unsafe {
            darkroom_eigf_variance_correct_4c(std::ptr::null_mut(), 10);
        }
    }

    #[test]
    fn ffi_variance_correct_4c_zero_n_guard() {
        let mut buf = vec![1.0f32; 4];
        unsafe {
            darkroom_eigf_variance_correct_4c(buf.as_mut_ptr(), 0);
        }
        assert_eq!(buf, vec![1.0; 4]); // untouched
    }

    #[test]
    fn ffi_variance_correct_2c_round_trip() {
        let mut buf = vec![0.0f32; 256 * 2];
        lcg_fill(&mut buf, 0x5678, 10.0);

        let mut ffi_buf = buf.clone();
        let mut direct_buf = buf.clone();

        unsafe {
            darkroom_eigf_variance_correct_2c(ffi_buf.as_mut_ptr(), 256);
        }
        eigf_variance_correct_2c(&mut direct_buf, 256);

        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_variance_correct_2c_null_guard() {
        unsafe {
            darkroom_eigf_variance_correct_2c(std::ptr::null_mut(), 10);
        }
    }

    #[test]
    fn ffi_variance_correct_2c_zero_n_guard() {
        let mut buf = vec![1.0f32; 2];
        unsafe {
            darkroom_eigf_variance_correct_2c(buf.as_mut_ptr(), 0);
        }
        assert_eq!(buf, vec![1.0; 2]); // untouched
    }

    #[test]
    fn ffi_zero_n_guard_both() {
        let mut buf4 = vec![1.0f32; 8];
        let mut buf2 = vec![1.0f32; 4];
        unsafe {
            darkroom_eigf_variance_correct_4c(buf4.as_mut_ptr(), 0);
            darkroom_eigf_variance_correct_2c(buf2.as_mut_ptr(), 0);
        }
        assert_eq!(buf4, vec![1.0; 8]);
        assert_eq!(buf2, vec![1.0; 4]);
    }

    // ── pack_variance_minmax_4c ────────────────────────────────────────────────

    #[test]
    fn pack_minmax_4c_basic() {
        let guide = vec![2.0f32, 1.0];
        let mask = vec![3.0f32, 4.0];
        let mut input = vec![f32::NAN; 8];
        let mut min = vec![f32::NAN; 4];
        let mut max = vec![f32::NAN; 4];
        pack_variance_minmax_4c(&mut input, &guide, &mask, 2, &mut min, &mut max);
        // channel order {g, g², m, m·g}
        assert_eq!(input, vec![2.0, 4.0, 3.0, 6.0, 1.0, 1.0, 4.0, 4.0]);
        assert_eq!(min, vec![1.0, 1.0, 3.0, 4.0]);
        assert_eq!(max, vec![2.0, 4.0, 4.0, 6.0]);
    }

    #[test]
    fn pack_minmax_4c_seed_values() {
        // Mixed data around the seeds: guide [2e7, -5] → min[0] = -5 (beats
        // both 2e7 and the 1e7 seed), max[0] = 2e7 (beats the 0.0 seed);
        // g²/m·g stay positive so their mins beat the 1e7 seed
        let guide = vec![2e7f32, -5.0];
        let mask = vec![3e7f32, -3.0];
        let mut input = vec![0.0f32; 8];
        let mut min = vec![f32::NAN; 4];
        let mut max = vec![f32::NAN; 4];
        pack_variance_minmax_4c(&mut input, &guide, &mask, 2, &mut min, &mut max);
        assert_eq!(min[0], -5.0);
        assert_eq!(min[2], -3.0);
        assert_eq!(max[0], 2e7);
        assert_eq!(max[2], 3e7);
        assert_eq!(min[1], 25.0);
        assert_eq!(min[3], 15.0);
        assert_eq!(max[1], 4e14);
        assert_eq!(max[3], 6e14);

        // All values above the 1e7 min seed → mins stay AT the seed,
        // and the 0.0 max seed is beaten by every value
        let guide = vec![2e7f32];
        let mask = vec![3e7f32];
        let mut min = vec![f32::NAN; 4];
        let mut max = vec![f32::NAN; 4];
        pack_variance_minmax_4c(&mut input, &guide, &mask, 1, &mut min, &mut max);
        assert_eq!(min[0], 10000000.0f32);
        assert_eq!(min[2], 10000000.0f32);
        assert_eq!(min[1], 10000000.0f32); // g² = 4e14 > 1e7 seed
        assert_eq!(min[3], 10000000.0f32);
        assert_eq!(max[0], 2e7);
    }

    #[test]
    fn pack_minmax_4c_glib_ternary_nan_quirk() {
        // GLib MIN/MAX are ternaries, not fminf/fmaxf:
        //   MIN(ming, NaN) → NaN (ming < NaN is false → takes b)
        //   MIN(NaN, next) → next (NaN < next is false → takes b)
        // so a NaN mid-stream is absorbed and the accumulator continues from
        // the NEXT value; a trailing NaN stays.
        let guide = vec![1.0f32, f32::NAN, 3.0];
        let mask = vec![1.0f32, 1.0, 1.0];
        let mut input = vec![0.0f32; 12];
        let mut min = vec![f32::NAN; 4];
        let mut max = vec![f32::NAN; 4];
        pack_variance_minmax_4c(&mut input, &guide, &mask, 3, &mut min, &mut max);
        assert_eq!(min[0], 3.0); // 1 → NaN → 3
        assert_eq!(max[0], 3.0); // 1 → NaN → 3

        let guide = vec![1.0f32, f32::NAN];
        let mut min = vec![f32::NAN; 4];
        let mut max = vec![f32::NAN; 4];
        pack_variance_minmax_4c(&mut input, &guide, &mask, 2, &mut min, &mut max);
        assert!(min[0].is_nan()); // trailing NaN stays
        assert!(max[0].is_nan());
    }

    #[test]
    fn pack_minmax_4c_matches_reference_over_lcg() {
        let mut guide = vec![0.0f32; 256];
        let mut mask = vec![0.0f32; 256];
        lcg_fill(&mut guide, 0xE1E1, 10.0);
        lcg_fill(&mut mask, 0xE2E2, 10.0);

        let mut direct_in = vec![0.0f32; 256 * 4];
        let mut ref_in = vec![0.0f32; 256 * 4];
        let mut dmin = vec![0.0f32; 4];
        let mut dmax = vec![0.0f32; 4];
        let mut rmin = vec![0.0f32; 4];
        let mut rmax = vec![0.0f32; 4];

        pack_variance_minmax_4c(&mut direct_in, &guide, &mask, 256, &mut dmin, &mut dmax);
        ref_pack_variance_minmax_4c(&mut ref_in, &guide, &mask, 256, &mut rmin, &mut rmax);

        assert_eq!(direct_in, ref_in);
        assert_eq!(dmin, rmin);
        assert_eq!(dmax, rmax);
    }

    // ── pack_variance_minmax_2c ────────────────────────────────────────────────

    #[test]
    fn pack_minmax_2c_basic() {
        let guide = vec![2.0f32, 1.0];
        let mut input = vec![f32::NAN; 4];
        let mut min = vec![f32::NAN; 2];
        let mut max = vec![f32::NAN; 2];
        pack_variance_minmax_2c(&mut input, &guide, 2, &mut min, &mut max);
        assert_eq!(input, vec![2.0, 4.0, 1.0, 1.0]);
        assert_eq!(min, vec![1.0, 1.0]);
        assert_eq!(max, vec![2.0, 4.0]);
    }

    #[test]
    fn pack_minmax_2c_seed_and_nan_quirks() {
        // Above-seed values: mins stay at the 1e7 seed
        let guide = vec![2e7f32];
        let mut input = vec![0.0f32; 2];
        let mut min = vec![f32::NAN; 2];
        let mut max = vec![f32::NAN; 2];
        pack_variance_minmax_2c(&mut input, &guide, 1, &mut min, &mut max);
        assert_eq!(min[0], 10000000.0f32);
        assert_eq!(min[1], 10000000.0f32); // g² = 4e14 > seed
        assert_eq!(max[0], 2e7);

        // GLib ternary NaN chain, mirroring the 4c pin: mid-stream NaN is
        // absorbed and the accumulator resumes at the next value
        let guide = vec![1.0f32, f32::NAN, 3.0];
        let mut input = vec![0.0f32; 6];
        let mut min = vec![f32::NAN; 2];
        let mut max = vec![f32::NAN; 2];
        pack_variance_minmax_2c(&mut input, &guide, 3, &mut min, &mut max);
        assert_eq!(min[0], 3.0);
        assert_eq!(max[0], 3.0);
    }

    #[test]
    fn pack_minmax_2c_matches_reference_over_lcg() {
        let mut guide = vec![0.0f32; 256];
        lcg_fill(&mut guide, 0xE3E3, 10.0);
        let mut direct_in = vec![0.0f32; 256 * 2];
        let mut ref_in = vec![0.0f32; 256 * 2];
        let mut dmin = vec![0.0f32; 2];
        let mut dmax = vec![0.0f32; 2];
        let mut rmin = vec![0.0f32; 2];
        let mut rmax = vec![0.0f32; 2];

        pack_variance_minmax_2c(&mut direct_in, &guide, 256, &mut dmin, &mut dmax);
        ref_pack_variance_minmax_2c(&mut ref_in, &guide, 256, &mut rmin, &mut rmax);

        assert_eq!(direct_in, ref_in);
        assert_eq!(dmin, rmin);
        assert_eq!(dmax, rmax);
    }

    // ── eigf_blending ──────────────────────────────────────────────────────────

    #[test]
    fn blending_4c_linear_identity() {
        // av = {avg_g=1, var_g=1, avg_m=1, covar=1}, image=mask=1, feather=1
        // → norm_g=norm_m=1, nv=1, nc=1, a=1/2, b=1-1/2 → blend = 1
        let mut image = vec![1.0f32];
        let mask = vec![1.0f32];
        let av = vec![1.0f32, 1.0, 1.0, 1.0];
        eigf_blending(&mut image, &mask, &av, 1, 0, 1.0);
        assert_eq!(image[0], 1.0);
    }

    #[test]
    fn blending_4c_geomean_identity() {
        let mut image = vec![1.0f32];
        let mask = vec![1.0f32];
        let av = vec![1.0f32, 1.0, 1.0, 1.0];
        eigf_blending(&mut image, &mask, &av, 1, 1, 1.0);
        assert_eq!(image[0], 1.0); // sqrt(1 * 1)
    }

    #[test]
    fn blending_4c_linear_clamps_to_min_float() {
        // avg_m=-4 drives b very negative → blend < MIN_FLOAT → clamped
        let mut image = vec![1.0f32];
        let mask = vec![1.0f32];
        let av = vec![1.0f32, 1.0, -4.0, 1.0];
        eigf_blending(&mut image, &mask, &av, 1, 0, 1.0);
        assert_eq!(image[0], MIN_FLOAT);
    }

    #[test]
    fn blending_4c_geomean_negative_blend_clamps() {
        let mut image = vec![1.0f32];
        let mask = vec![1.0f32];
        let av = vec![1.0f32, 1.0, -4.0, 1.0];
        eigf_blending(&mut image, &mask, &av, 1, 1, 1.0);
        // image *= MIN_FLOAT then sqrt
        assert_eq!(image[0], MIN_FLOAT.sqrt());
    }

    #[test]
    fn blending_4c_nan_blend_is_nan_ignoring_like_fmaxf() {
        // covar = NaN → a = NaN → blend = NaN; the kernel's `.max(MIN_FLOAT)`
        // is NaN-ignoring (IEEE maximumNum, matching C's fmaxf) → MIN_FLOAT
        let mut image = vec![1.0f32];
        let mask = vec![1.0f32];
        let av = vec![1.0f32, 1.0, 1.0, f32::NAN];
        eigf_blending(&mut image, &mask, &av, 1, 0, 1.0);
        assert_eq!(image[0], MIN_FLOAT);
    }

    #[test]
    fn blending_4c_matches_reference_over_lcg() {
        let mut image = vec![0.0f32; 256];
        let mut mask = vec![0.0f32; 256];
        let mut av = vec![0.0f32; 256 * 4];
        lcg_fill(&mut image, 0xE4E4, 5.0);
        lcg_fill(&mut mask, 0xE5E5, 5.0);
        lcg_fill(&mut av, 0xE6E6, 5.0);
        for v in image.iter_mut() {
            *v = v.abs() + 0.5;
        }
        for v in av.iter_mut() {
            *v = v.abs() + 0.5;
        }

        for filter in [0i32, 1] {
            let mut direct = image.clone();
            let mut reference = image.clone();
            eigf_blending(&mut direct, &mask, &av, 256, filter, 0.05);
            ref_eigf_blending(&mut reference, &mask, &av, 256, filter, 0.05);
            assert_eq!(direct, reference, "filter={filter}");
        }
    }

    // ── eigf_blending_no_mask ─────────────────────────────────────────────────

    #[test]
    fn blending_2c_linear_identity() {
        // av = {avg_g=1, var_g=1}, image=1, feather=1 → a=1/2, b=1-1/2 → 1
        let mut image = vec![1.0f32];
        let av = vec![1.0f32, 1.0];
        eigf_blending_no_mask(&mut image, &av, 1, 0, 1.0);
        assert_eq!(image[0], 1.0);
    }

    #[test]
    fn blending_2c_geomean_identity() {
        let mut image = vec![1.0f32];
        let av = vec![1.0f32, 1.0];
        eigf_blending_no_mask(&mut image, &av, 1, 1, 1.0);
        assert_eq!(image[0], 1.0);
    }

    #[test]
    fn blending_2c_zero_image_uses_norm_floor() {
        // image=0 → norm_g = max(avg_g*0, 1e-6) = 1e-6; with avg_g=1,
        // var_g=1e-6: nv = 1; a = 1/(1+1) = 0.5; b = 1-0.5 = 0.5;
        // blend = 0*0.5+0.5 = 0.5
        let mut image = vec![0.0f32];
        let av = vec![1.0f32, 1e-6];
        eigf_blending_no_mask(&mut image, &av, 1, 0, 1.0);
        assert_eq!(image[0], 0.5);
    }

    #[test]
    fn blending_2c_matches_reference_over_lcg() {
        let mut image = vec![0.0f32; 256];
        let mut av = vec![0.0f32; 256 * 2];
        lcg_fill(&mut image, 0xE7E7, 5.0);
        lcg_fill(&mut av, 0xE8E8, 5.0);
        for v in image.iter_mut() {
            *v = v.abs() + 0.5;
        }
        for v in av.iter_mut() {
            *v = v.abs() + 0.5;
        }

        for filter in [0i32, 1] {
            let mut direct = image.clone();
            let mut reference = image.clone();
            eigf_blending_no_mask(&mut direct, &av, 256, filter, 0.05);
            ref_eigf_blending_no_mask(&mut reference, &av, 256, filter, 0.05);
            assert_eq!(direct, reference, "filter={filter}");
        }
    }

    // ── FFI round-trip and guard tests (m4-171 kernels) ───────────────────────

    #[test]
    fn ffi_pack_minmax_4c_round_trip() {
        let mut guide = vec![0.0f32; 128];
        let mut mask = vec![0.0f32; 128];
        lcg_fill(&mut guide, 0xE9E9, 10.0);
        lcg_fill(&mut mask, 0xEAEA, 10.0);
        let mut ffi_in = vec![0.0f32; 128 * 4];
        let mut direct_in = vec![0.0f32; 128 * 4];
        let mut fmin = vec![0.0f32; 4];
        let mut fmax = vec![0.0f32; 4];
        let mut dmin = vec![0.0f32; 4];
        let mut dmax = vec![0.0f32; 4];

        unsafe {
            darkroom_eigf_pack_variance_minmax_4c(
                ffi_in.as_mut_ptr(),
                guide.as_ptr(),
                mask.as_ptr(),
                128,
                fmin.as_mut_ptr(),
                fmax.as_mut_ptr(),
            );
        }
        pack_variance_minmax_4c(&mut direct_in, &guide, &mask, 128, &mut dmin, &mut dmax);
        assert_eq!(ffi_in, direct_in);
        assert_eq!(fmin, dmin);
        assert_eq!(fmax, dmax);
    }

    #[test]
    fn ffi_pack_minmax_4c_guards() {
        unsafe {
            darkroom_eigf_pack_variance_minmax_4c(
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                10,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
        }
        let guide = vec![1.0f32; 4];
        let mask = vec![1.0f32; 4];
        let mut input = vec![1.0f32; 16];
        let mut min = vec![1.0f32; 4];
        let mut max = vec![1.0f32; 4];
        unsafe {
            darkroom_eigf_pack_variance_minmax_4c(
                input.as_mut_ptr(),
                guide.as_ptr(),
                mask.as_ptr(),
                0,
                min.as_mut_ptr(),
                max.as_mut_ptr(),
            );
            darkroom_eigf_pack_variance_minmax_4c(
                input.as_mut_ptr(),
                guide.as_ptr(),
                mask.as_ptr(),
                (i32::MAX as usize) + 1,
                min.as_mut_ptr(),
                max.as_mut_ptr(),
            );
        }
        assert_eq!(input, vec![1.0f32; 16]); // untouched
        assert_eq!(min, vec![1.0f32; 4]); // untouched
        assert_eq!(max, vec![1.0f32; 4]);
    }

    #[test]
    fn ffi_pack_minmax_2c_round_trip_and_guards() {
        let mut guide = vec![0.0f32; 128];
        lcg_fill(&mut guide, 0xEBEB, 10.0);
        let mut ffi_in = vec![0.0f32; 128 * 2];
        let mut direct_in = vec![0.0f32; 128 * 2];
        let mut fmin = vec![0.0f32; 2];
        let mut fmax = vec![0.0f32; 2];
        let mut dmin = vec![0.0f32; 2];
        let mut dmax = vec![0.0f32; 2];

        unsafe {
            darkroom_eigf_pack_variance_minmax_2c(
                ffi_in.as_mut_ptr(),
                guide.as_ptr(),
                128,
                fmin.as_mut_ptr(),
                fmax.as_mut_ptr(),
            );
        }
        pack_variance_minmax_2c(&mut direct_in, &guide, 128, &mut dmin, &mut dmax);
        assert_eq!(ffi_in, direct_in);
        assert_eq!(fmin, dmin);
        assert_eq!(fmax, dmax);

        unsafe {
            darkroom_eigf_pack_variance_minmax_2c(
                std::ptr::null_mut(),
                std::ptr::null(),
                10,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            darkroom_eigf_pack_variance_minmax_2c(
                ffi_in.as_mut_ptr(),
                guide.as_ptr(),
                0,
                fmin.as_mut_ptr(),
                fmax.as_mut_ptr(),
            );
            darkroom_eigf_pack_variance_minmax_2c(
                ffi_in.as_mut_ptr(),
                guide.as_ptr(),
                (i32::MAX as usize) + 1,
                fmin.as_mut_ptr(),
                fmax.as_mut_ptr(),
            );
        }
        assert_eq!(fmin, dmin); // untouched by the zero-n and overflow calls
        assert_eq!(fmax, dmax);
    }

    #[test]
    fn ffi_blending_round_trips() {
        let mut image = vec![0.0f32; 128];
        let mut mask = vec![0.0f32; 128];
        let mut av4 = vec![0.0f32; 128 * 4];
        let mut av2 = vec![0.0f32; 128 * 2];
        lcg_fill(&mut image, 0xECEC, 5.0);
        lcg_fill(&mut mask, 0xEDED, 5.0);
        lcg_fill(&mut av4, 0xEEEE, 5.0);
        lcg_fill(&mut av2, 0xEFEF, 5.0);
        for v in image.iter_mut() {
            *v = v.abs() + 0.5;
        }
        for v in av4.iter_mut() {
            *v = v.abs() + 0.5;
        }
        for v in av2.iter_mut() {
            *v = v.abs() + 0.5;
        }

        let mut ffi_img = image.clone();
        let mut direct_img = image.clone();
        unsafe {
            darkroom_eigf_blending(
                ffi_img.as_mut_ptr(),
                mask.as_ptr(),
                av4.as_ptr(),
                128,
                0,
                0.05,
            );
        }
        eigf_blending(&mut direct_img, &mask, &av4, 128, 0, 0.05);
        assert_eq!(ffi_img, direct_img);

        let mut ffi_img = image.clone();
        let mut direct_img = image.clone();
        unsafe {
            darkroom_eigf_blending_no_mask(ffi_img.as_mut_ptr(), av2.as_ptr(), 128, 1, 0.05);
        }
        eigf_blending_no_mask(&mut direct_img, &av2, 128, 1, 0.05);
        assert_eq!(ffi_img, direct_img);
    }

    #[test]
    fn ffi_blending_guards() {
        unsafe {
            darkroom_eigf_blending(
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                10,
                0,
                1.0,
            );
            darkroom_eigf_blending_no_mask(std::ptr::null_mut(), std::ptr::null(), 10, 0, 1.0);
        }
        let av2 = vec![1.0f32; 8];
        let mut image = vec![1.0f32; 4];
        unsafe {
            darkroom_eigf_blending_no_mask(image.as_mut_ptr(), av2.as_ptr(), 0, 0, 1.0);
            darkroom_eigf_blending_no_mask(
                image.as_mut_ptr(),
                av2.as_ptr(),
                (i32::MAX as usize) + 1,
                0,
                1.0,
            );
        }
        assert_eq!(image, vec![1.0f32; 4]); // untouched
    }
}
