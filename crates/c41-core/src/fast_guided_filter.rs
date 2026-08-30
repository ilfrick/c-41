//! Element-wise loops ported from `src/common/fast_guided_filter.h`.
//!
//! Six C loops are ported here (m4-166 and m4-168/169):
//! - The variance-analyse pack loop (formerly at fast_guided_filter.h:178,
//!   `DT_OMP_FOR_SIMD`) that packs `{guide, mask, guide², guide·mask}` into a
//!   4-channel buffer before the box-mean blur.
//! - The apply-linear-blending loop (formerly at :211, `DT_OMP_FOR_SIMD`) that
//!   applies the `a*image + b` blend and clamps to `MIN_FLOAT`.
//! - The a/b solve at the end of `variance_analyse` (formerly at :186,
//!   plain `DT_OMP_FOR`).
//! - The `apply_linear_blending_w_geomean` loop (formerly at :215, plain
//!   `DT_OMP_FOR`).
//! - The `quantize` fast and general tracks (formerly at :242 and :250, plain
//!   `DT_OMP_FOR`; the `sampling == 0.0f` copy branch stays in C).
//! - The `interpolate_bilinear` gather loop (formerly at :104, collapse(2)),
//!   used by `fast_surface_blur` and many IOPs.
//!
//! The Rust kernels are single-threaded sequential; LLVM's auto-vectorizer
//! provides SIMD at `-O3`, but multi-threaded parallelism is no longer used.
//! This matches the m4-161 `blend.rs` and m4-162 `imagebuf.rs` pattern.
//!
//! Bit-exactness notes:
//! - `apply_linear_blending` computes `image[k] * ab[k*2] + ab[k*2+1]` — a
//!   multiply-add pattern susceptible to FMA contraction. GCC 9's default
//!   `-ffp-contract=fast` for C99+ may contract this at `-O3`, while the Rust
//!   release profile does not. This produces ≤1 ULP difference on some pixels,
//!   same trade-off as `linear_blend` in `imagebuf.rs` and documented for
//!   `mask_tone_curve` in `blend.rs`.
//! - The pack loop is pure multiplies and assignments — no add-after-multiply,
//!   so there is no FMA contraction risk.
//! - `fmaxf(x, MIN_FLOAT)` → `x.max(MIN_FLOAT)`. `MIN_FLOAT` is
//!   `exp2f(-16.0f)` in C = 2^(-16), which is exactly representable in f32.
//!   Rust's `f32::max` is the NaN-ignoring maximum (IEEE 754 `maximumNum`),
//!   matching standard libm `fmaxf` — both return the non-NaN operand when
//!   one is NaN. The divergence is TU-dependent on the C side:
//!   the global Release flags are `-ffast-math -fno-finite-math-only`
//!   (src/CMakeLists.txt:720), which preserve `fmaxf` NaN semantics, but
//!   TUs that `#include extra_optimizations.h` re-apply
//!   `#pragma GCC optimize("finite-math-only")`, and there GCC lowers the
//!   **vectorized** max to a SIMD max whose source order makes NaN
//!   propagate (the computed value is the second source operand, and SSE max
//!   returns the second operand when either input is NaN — verified
//!   empirically in the CI image; the scalar `maxss` inlining of
//!   `fmaxf(x, MIN_FLOAT)` happens to be NaN-ignoring because the constant
//!   ends up as the second operand). The removed C loop was vectorized
//!   (`DT_OMP_FOR_SIMD`), so in pragma TUs the old C path returned NaN on NaN
//!   input where the Rust kernel returns `MIN_FLOAT`. NaN reaching
//!   `image[k]*ab[k*2] + ab[k*2+1]` is rare (requires upstream inf-inf or
//!   sqrt-of-negative in the luminance-mask pipeline).
//! - Note: the reference implementation `ref_apply_linear_blending`
//!   deliberately diverges using `if blended < MIN_FLOAT { MIN_FLOAT } else { blended }`,
//!   which is NaN-propagating (NaN < MIN_FLOAT is false, so NaN is returned).
//!   This structural divergence is intentional for independent validation.
//! - `solve_ab` (variance_analyse's a/b solve) computes `ch2 - ch0*ch0` and
//!   `ch1 - a*ch0` — both multiply-subtract patterns susceptible to GCC FMA
//!   contraction, so the C-vs-Rust result may differ by ≤1 ULP per element
//!   (same accepted class as `apply_linear_blending` above). The
//!   `fmaxf(d, 1e-15f)` division guard is NaN-ignoring in standard libm and
//!   in Rust (`f32::max`); the TU-dependent vectorized-max caveat above
//!   applies to it unchanged.
//! - `apply_linear_blending_w_geomean` has the same multiply-add inside its
//!   `fmaxf` (contraction risk, ≤1 ULP) plus a `sqrtf`, which has no
//!   additional contraction surface. NaN input propagates through the
//!   multiply into `sqrtf` in both languages (the `.max(MIN_FLOAT)` is
//!   NaN-ignoring in both); only the pragma-TU vectorized-max caveat above
//!   changes that on the C side.
//! - `quantize` chains `log2f → floorf → (÷) → (×) → exp2f` then
//!   `fmaxf(fminf(v, top), bottom)` (fast_clamp). There is no add-after-
//!   multiply pattern, so no FMA contraction risk. `f32::min`/`f32::max`
//!   match standard `fminf`/`fmaxf` (NaN-ignoring), so a NaN input
//!   quantizes to `clip_max` in both languages — under standard libm
//!   semantics; the same TU-dependent vectorized-max caveat above applies
//!   (whether GCC vectorizes these log2/exp2 chains into diverging libmvec
//!   variants in pragma TUs is unverified — the old loops were plain
//!   `DT_OMP_FOR`, so vectorization was opportunistic).
//! - `interpolate_bilinear` is the first gather-style port: per output pixel
//!   it reads four corner pixels and blends them with
//!   `Dy*(Q_SW*Dx_n + Q_SE*Dx_p) + Dy_n*(Q_NW*Dx_n + Q_NE*Dx_p)` — a dense
//!   multiply-add chain, so the C-vs-Rust difference from GCC's FMA
//!   contraction (and, more broadly, `-ffast-math` reassociation, which has
//!   no clean ULP bound) applies here, unlike the single-site cases above. The C quirks preserved: `Dx_next`/`Dy_next` are computed from
//!   the *clamped* neighbour indices (negative near the right/bottom border,
//!   weights still summing to 1), and the coordinate chain is
//!   `(float)j / width_out * width_in` in that order.
//! - `fast_guided_filter.h` has no `#pragma GCC optimize("fast-math")`.
//!   The global CMakeLists.txt `-ffast-math` flag is TU-wide and may enable
//!   FMA contraction (`-ffp-contract=fast`) at `-O3`. The kernel-vs-reference
//!   FMA difference is Rust-internal (both use the same FP evaluation order).
//!   The C-vs-Rust ≤1 ULP FMA contraction difference is documented above.

/// `MIN_FLOAT` constant from `luminance_mask.h`: `exp2f(-16.0f)` = 2^(-16).
/// Exactly representable in f32, so no rounding discrepancy vs C.
const MIN_FLOAT: f32 = 1.52587890625e-5_f32;

/// Pack guide and mask into a 4-channel buffer with their products.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `variance_analyse`
/// (formerly at fast_guided_filter.h:178). For each element:
/// - ch0 = guide[k]
/// - ch1 = mask[k]
/// - ch2 = guide[k] * guide[k]
/// - ch3 = guide[k] * mask[k]
pub fn pack_variance_4c(input: &mut [f32], guide: &[f32], mask: &[f32], n_elements: usize) {
    let m = n_elements.min(guide.len()).min(mask.len()).min(input.len() / 4);
    for k in 0..m {
        let index = k * 4;
        let g = guide[k];
        let mg = mask[k];
        input[index] = g;
        input[index + 1] = mg;
        input[index + 2] = g * g;
        input[index + 3] = g * mg;
    }
}

/// Apply linear blending `image[k] = max(image[k]*a + b, MIN_FLOAT)`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `apply_linear_blending`
/// (formerly at fast_guided_filter.h:211). `ab[k*2]` is the blend coefficient `a`,
/// `ab[k*2+1]` is the offset `b`.
pub fn apply_linear_blending(image: &mut [f32], ab: &[f32], n_elements: usize) {
    let m = n_elements.min(image.len()).min(ab.len() / 2);
    for k in 0..m {
        let val = image[k] * ab[k * 2] + ab[k * 2 + 1];
        image[k] = val.max(MIN_FLOAT);
    }
}

/// Solve the per-pixel guided-filter coefficients from the blurred moments.
///
/// Port of the `DT_OMP_FOR` loop in `variance_analyse`
/// (formerly at fast_guided_filter.h:186). For each element, `input` holds the box-blurred
/// 4-channel pack `{I, p, I², I·p}` (ch0..ch3); computes
/// `d = fmaxf((ch2 - ch0²) + feathering, 1e-15f)` (division guard),
/// `a = (ch3 - ch0·ch1) / d`, `b = ch1 - a·ch0` and writes them to
/// `ab[2k]`/`ab[2k+1]`.
pub fn solve_ab(input: &[f32], ab: &mut [f32], n_elements: usize, feathering: f32) {
    let m = n_elements.min(ab.len() / 2).min(input.len() / 4);
    for k in 0..m {
        let g = input[4 * k];
        let p = input[4 * k + 1];
        let gg = input[4 * k + 2];
        let gp = input[4 * k + 3];
        let d = ((gg - g * g) + feathering).max(1e-15_f32);
        let a = (gp - g * p) / d;
        let b = p - a * g;
        ab[2 * k] = a;
        ab[2 * k + 1] = b;
    }
}

/// Blending with geometric mean: `image[k] = sqrt(image[k] * max(image[k]*a + b, MIN_FLOAT))`.
///
/// Port of the `DT_OMP_FOR` loop in `apply_linear_blending_w_geomean`
/// (formerly at fast_guided_filter.h:215). `image[k]` is positive outside the luminance
/// mask, so the square-root argument is expected non-negative there; a NaN
/// argument propagates through the sqrt.
pub fn apply_linear_blending_w_geomean(image: &mut [f32], ab: &[f32], n_elements: usize) {
    let m = n_elements.min(image.len()).min(ab.len() / 2);
    for k in 0..m {
        let blended = (image[k] * ab[k * 2] + ab[k * 2 + 1]).max(MIN_FLOAT);
        image[k] = (image[k] * blended).sqrt();
    }
}

/// Quantize in exposure levels evenly spaced in log by `sampling`.
///
/// Port of the two `DT_OMP_FOR` loops in `quantize`
/// (formerly at fast_guided_filter.h:242 fast track, :250 general track):
/// `fast_clamp(exp2f(floorf(log2f(image[k]) / sampling) * sampling),
/// clip_min, clip_max)`, where the fast track (`sampling == 1.0f`) skips the
/// divide/multiply by 1. The `sampling == 0.0f` copy branch stays in C
/// (`dt_iop_image_copy`) — this kernel must not be called with 0.
/// `fast_clamp` is `fmaxf(fminf(v, top), bottom)`; `f32::min`/`f32::max`
/// are NaN-ignoring exactly like standard `fminf`/`fmaxf`, so a NaN input
/// quantizes to `clip_max` (NaN.min(top) = top), matching standard libm.
pub fn quantize(
    image: &[f32],
    out: &mut [f32],
    n_elements: usize,
    sampling: f32,
    clip_min: f32,
    clip_max: f32,
) {
    let m = n_elements.min(image.len()).min(out.len());
    for k in 0..m {
        let v = if sampling == 1.0f32 {
            image[k].log2().floor().exp2()
        } else {
            ((image[k].log2() / sampling).floor() * sampling).exp2()
        };
        // fast_clamp: fmaxf(fminf(v, top), bottom)
        out[k] = v.min(clip_max).max(clip_min);
    }
}

/// Fast bilinear interpolation of a `ch`-channel image to a new size.
///
/// Port of the `DT_OMP_FOR(collapse(2))` loop in `interpolate_bilinear`
/// (formerly at fast_guided_filter.h:104). For each output pixel, maps back
/// to input coordinates, clamps the four neighbours into the input image,
/// and blends: `out = Dy_prev*(Q_SW*Dx_next + Q_SE*Dx_prev) +
/// Dy_next*(Q_NW*Dx_next + Q_NE*Dx_prev)` per channel.
///
/// Note the C quirks preserved exactly: `Dx_next`/`Dy_next` are computed
/// from the *clamped* neighbour indices (so both can go slightly negative
/// near the right/bottom border, with the weights still summing to 1), and
/// the coordinate chain is `(float)j / width_out * width_in`, in that order.
pub fn interpolate_bilinear(
    src: &[f32],
    width_in: usize,
    height_in: usize,
    out: &mut [f32],
    width_out: usize,
    height_out: usize,
    ch: usize,
) {
    if width_in == 0 || height_in == 0 || ch == 0 || width_out == 0 || height_out == 0 {
        return;
    }
    if src.len() < width_in * height_in * ch {
        return;
    }
    // Clamp the total pixel count, not the row stride: rows keep the
    // width_out stride so a defensively short `out` truncates the tail
    // instead of corrupting the layout (direct Rust callers only — via the
    // FFI a short buffer is already UB at slice construction).
    let n_pixels = (width_out * height_out).min(out.len() / ch);
    for p in 0..n_pixels {
        let i = p / width_out;
        let j = p % width_out;
        // Relative coordinates of the pixel in output space
        let x_out = j as f32 / width_out as f32;
        let y_out = i as f32 / height_out as f32;

        // Corresponding absolute coordinates of the pixel in input space
        let x_in = x_out * width_in as f32;
        let y_in = y_out * height_in as f32;

        // Nearest neighbours coordinates in input space
        let x_prev = (x_in.floor() as usize).min(width_in - 1);
        let x_next = (x_prev + 1).min(width_in - 1);
        let y_prev = (y_in.floor() as usize).min(height_in - 1);
        let y_next = (y_prev + 1).min(height_in - 1);

        // Nearest pixels in input array (nodes in grid)
        let y_prev_row = y_prev * width_in;
        let y_next_row = y_next * width_in;
        let q_nw = &src[(y_prev_row + x_prev) * ch..];
        let q_ne = &src[(y_prev_row + x_next) * ch..];
        let q_se = &src[(y_next_row + x_next) * ch..];
        let q_sw = &src[(y_next_row + x_prev) * ch..];

        // Spatial differences between nodes
        let dy_next = y_next as f32 - y_in;
        let dy_prev = 1.0f32 - dy_next;
        let dx_next = x_next as f32 - x_in;
        let dx_prev = 1.0f32 - dx_next;

        // Interpolate over ch layers
        let pixel_out = &mut out[p * ch..];
        for c in 0..ch {
            pixel_out[c] = dy_prev * (q_sw[c] * dx_next + q_se[c] * dx_prev)
                + dy_next * (q_nw[c] * dx_next + q_ne[c] * dx_prev);
        }
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `input` must hold at least `n_elements * 4` floats. `guide` and `mask`
/// must each hold at least `n_elements` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_fgf_pack_variance_4c(
    input: *mut f32,
    guide: *const f32,
    mask: *const f32,
    n_elements: usize,
) {
    if input.is_null() || guide.is_null() || mask.is_null()
        || n_elements == 0 || n_elements > i32::MAX as usize
    {
        return;
    }
    let input_slice = std::slice::from_raw_parts_mut(input, n_elements * 4);
    let guide_slice = std::slice::from_raw_parts(guide, n_elements);
    let mask_slice = std::slice::from_raw_parts(mask, n_elements);
    pack_variance_4c(input_slice, guide_slice, mask_slice, n_elements);
}

/// # Safety
/// `image` must hold at least `n_elements` floats. `ab` must hold at least
/// `n_elements * 2` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_fgf_apply_linear_blending(
    image: *mut f32,
    ab: *const f32,
    n_elements: usize,
) {
    if image.is_null() || ab.is_null() || n_elements == 0 || n_elements > i32::MAX as usize {
        return;
    }
    let image_slice = std::slice::from_raw_parts_mut(image, n_elements);
    let ab_slice = std::slice::from_raw_parts(ab, n_elements * 2);
    apply_linear_blending(image_slice, ab_slice, n_elements);
}

/// # Safety
/// `input` must hold at least `n_elements * 4` floats. `ab` must hold at
/// least `n_elements * 2` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_fgf_solve_ab(
    input: *const f32,
    ab: *mut f32,
    n_elements: usize,
    feathering: f32,
) {
    if input.is_null() || ab.is_null() || n_elements == 0 || n_elements > i32::MAX as usize {
        return;
    }
    let input_slice = std::slice::from_raw_parts(input, n_elements * 4);
    let ab_slice = std::slice::from_raw_parts_mut(ab, n_elements * 2);
    solve_ab(input_slice, ab_slice, n_elements, feathering);
}

/// # Safety
/// `image` must hold at least `n_elements` floats. `ab` must hold at least
/// `n_elements * 2` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_fgf_apply_linear_blending_w_geomean(
    image: *mut f32,
    ab: *const f32,
    n_elements: usize,
) {
    if image.is_null() || ab.is_null() || n_elements == 0 || n_elements > i32::MAX as usize {
        return;
    }
    let image_slice = std::slice::from_raw_parts_mut(image, n_elements);
    let ab_slice = std::slice::from_raw_parts(ab, n_elements * 2);
    apply_linear_blending_w_geomean(image_slice, ab_slice, n_elements);
}

/// # Safety
/// `image` and `out` must each hold at least `n_elements` floats. `sampling`
/// must not be 0.0 (the C caller handles that branch itself).
#[no_mangle]
pub unsafe extern "C" fn darkroom_fgf_quantize(
    image: *const f32,
    out: *mut f32,
    n_elements: usize,
    sampling: f32,
    clip_min: f32,
    clip_max: f32,
) {
    if image.is_null() || out.is_null() || n_elements == 0 || n_elements > i32::MAX as usize {
        return;
    }
    let image_slice = std::slice::from_raw_parts(image, n_elements);
    let out_slice = std::slice::from_raw_parts_mut(out, n_elements);
    quantize(
        image_slice,
        out_slice,
        n_elements,
        sampling,
        clip_min,
        clip_max,
    );
}

/// # Safety
/// `src` must hold at least `width_in * height_in * ch` floats. `out` must
/// hold at least `width_out * height_out * ch` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_fgf_interpolate_bilinear(
    src: *const f32,
    width_in: usize,
    height_in: usize,
    out: *mut f32,
    width_out: usize,
    height_out: usize,
    ch: usize,
) {
    if src.is_null()
        || out.is_null()
        || width_in == 0
        || height_in == 0
        || width_out == 0
        || height_out == 0
        || ch == 0
        || width_in.max(height_in).max(width_out).max(height_out) > i32::MAX as usize
        || ch > i32::MAX as usize
    {
        return;
    }
    let src_slice = std::slice::from_raw_parts(src, width_in * height_in * ch);
    let out_slice = std::slice::from_raw_parts_mut(out, width_out * height_out * ch);
    interpolate_bilinear(
        src_slice,
        width_in,
        height_in,
        out_slice,
        width_out,
        height_out,
        ch,
    );
}

// ── Independent reference implementations ────────────────────────────────────
//
// These compute the same results via a different evaluation order:
// products are stored in named temporaries before assignment, rather than
// inlined into the write expression. This provides genuine bit-exactness
// validation without being a copy-paste of the kernel.

#[allow(dead_code)]
fn ref_pack_variance_4c(input: &mut [f32], guide: &[f32], mask: &[f32], n_elements: usize) {
    let m = n_elements.min(guide.len()).min(mask.len()).min(input.len() / 4);
    for k in 0..m {
        let g = guide[k];
        let mg = mask[k];
        let gg = g * g;
        let gmg = g * mg;
        input[k * 4] = g;
        input[k * 4 + 1] = mg;
        input[k * 4 + 2] = gg;
        input[k * 4 + 3] = gmg;
    }
}

#[allow(dead_code)]
fn ref_apply_linear_blending(image: &mut [f32], ab: &[f32], n_elements: usize) {
    let m = n_elements.min(image.len()).min(ab.len() / 2);
    for k in 0..m {
        let a = ab[k * 2];
        let b = ab[k * 2 + 1];
        let blended = image[k] * a + b;
        let floored = if blended < MIN_FLOAT { MIN_FLOAT } else { blended };
        image[k] = floored;
    }
}

#[allow(dead_code)]
fn ref_solve_ab(input: &[f32], ab: &mut [f32], n_elements: usize, feathering: f32) {
    let m = n_elements.min(ab.len() / 2).min(input.len() / 4);
    for k in 0..m {
        let g = input[4 * k];
        let p = input[4 * k + 1];
        let gg = input[4 * k + 2];
        let gp = input[4 * k + 3];
        let raw_d = (gg - g * g) + feathering;
        let d = if raw_d > 1e-15_f32 { raw_d } else { 1e-15_f32 };
        let num = gp - g * p;
        let a = num / d;
        let b = p - a * g;
        ab[2 * k] = a;
        ab[2 * k + 1] = b;
    }
}

#[allow(dead_code)]
fn ref_apply_linear_blending_w_geomean(image: &mut [f32], ab: &[f32], n_elements: usize) {
    let m = n_elements.min(image.len()).min(ab.len() / 2);
    for k in 0..m {
        let a = ab[k * 2];
        let b = ab[k * 2 + 1];
        let blended = image[k] * a + b;
        let floored = if blended < MIN_FLOAT { MIN_FLOAT } else { blended };
        let product = image[k] * floored;
        image[k] = product.sqrt();
    }
}

#[allow(dead_code)]
fn ref_quantize(
    image: &[f32],
    out: &mut [f32],
    n_elements: usize,
    sampling: f32,
    clip_min: f32,
    clip_max: f32,
) {
    let m = n_elements.min(image.len()).min(out.len());
    for k in 0..m {
        let l = image[k].log2();
        let v = if sampling == 1.0f32 {
            l.floor().exp2()
        } else {
            let scaled = l / sampling;
            (scaled.floor() * sampling).exp2()
        };
        let top = if v < clip_max { v } else { clip_max };
        out[k] = if top > clip_min { top } else { clip_min };
    }
}

#[allow(dead_code)]
fn ref_interpolate_bilinear(
    src: &[f32],
    width_in: usize,
    height_in: usize,
    out: &mut [f32],
    width_out: usize,
    height_out: usize,
    ch: usize,
) {
    // Same maths and FP evaluation order, different code shape: the four
    // corner reads go through explicit index temporaries and the two blend
    // terms are computed as named values.
    let m = (width_out * height_out).min(out.len() / ch.max(1));
    for p in 0..m {
        let i = p / width_out;
        let j = p % width_out;
        let x_in = (j as f32 / width_out as f32) * width_in as f32;
        let y_in = (i as f32 / height_out as f32) * height_in as f32;
        let xp = (x_in.floor() as usize).min(width_in - 1);
        let xn = (xp + 1).min(width_in - 1);
        let yp = (y_in.floor() as usize).min(height_in - 1);
        let yn = (yp + 1).min(height_in - 1);
        let dx_n = xn as f32 - x_in;
        let dx_p = 1.0f32 - dx_n;
        let dy_n = yn as f32 - y_in;
        let dy_p = 1.0f32 - dy_n;
        let i_nw = (yp * width_in + xp) * ch;
        let i_ne = (yp * width_in + xn) * ch;
        let i_se = (yn * width_in + xn) * ch;
        let i_sw = (yn * width_in + xp) * ch;
        for c in 0..ch {
            let low = src[i_sw + c] * dx_n + src[i_se + c] * dx_p;
            let high = src[i_nw + c] * dx_n + src[i_ne + c] * dx_p;
            out[p * ch + c] = dy_p * low + dy_n * high;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    // ── pack_variance_4c ───────────────────────────────────────────────────────

    #[test]
    fn pack_variance_4c_basic() {
        let guide = vec![1.0f32, 2.0, 3.0];
        let mask = vec![0.5f32, 0.25, 0.75];
        let mut input = vec![0.0f32; 12];

        pack_variance_4c(&mut input, &guide, &mask, 3);

        // element 0: g=1, m=0.5, g²=1, gm=0.5
        assert_eq!(input[0], 1.0);
        assert_eq!(input[1], 0.5);
        assert_eq!(input[2], 1.0);
        assert_eq!(input[3], 0.5);
        // element 1: g=2, m=0.25, g²=4, gm=0.5
        assert_eq!(input[4], 2.0);
        assert_eq!(input[5], 0.25);
        assert_eq!(input[6], 4.0);
        assert_eq!(input[7], 0.5);
        // element 2: g=3, m=0.75, g²=9, gm=2.25
        assert_eq!(input[8], 3.0);
        assert_eq!(input[9], 0.75);
        assert_eq!(input[10], 9.0);
        assert_eq!(input[11], 2.25);
    }

    #[test]
    fn pack_variance_4c_zeros() {
        let guide = vec![0.0f32; 4];
        let mask = vec![0.0f32; 4];
        let mut input = vec![1.0f32; 16];

        pack_variance_4c(&mut input, &guide, &mask, 4);

        for v in &input {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn pack_variance_4c_matches_reference_over_lcg() {
        let mut guide = vec![0.0f32; 256];
        let mut mask = vec![0.0f32; 256];
        lcg_fill(&mut guide, 0x1111, 10.0);
        lcg_fill(&mut mask, 0x2222, 10.0);

        let mut direct = vec![0.0f32; 256 * 4];
        let mut reference = vec![0.0f32; 256 * 4];

        pack_variance_4c(&mut direct, &guide, &mask, 256);
        ref_pack_variance_4c(&mut reference, &guide, &mask, 256);

        assert_eq!(direct, reference);
    }

    // ── apply_linear_blending ──────────────────────────────────────────────────

    #[test]
    fn apply_linear_blending_basic() {
        let mut image = vec![2.0f32, 4.0, 6.0];
        let ab = vec![0.5f32, 1.0, 1.0, 0.0, 2.0, -1.0];
        // k=0: 2.0*0.5 + 1.0 = 2.0 (>= MIN_FLOAT)
        // k=1: 4.0*1.0 + 0.0 = 4.0
        // k=2: 6.0*2.0 + (-1.0) = 11.0
        apply_linear_blending(&mut image, &ab, 3);
        assert_eq!(image, vec![2.0, 4.0, 11.0]);
    }

    #[test]
    fn apply_linear_blending_clamps_to_min_float() {
        let mut image = vec![0.0f32, 0.0, 0.0];
        let ab = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        // All results: 0*0 + 0 = 0.0, clamped to MIN_FLOAT
        apply_linear_blending(&mut image, &ab, 3);
        for v in &image {
            assert_eq!(*v, MIN_FLOAT);
        }
    }

    #[test]
    fn apply_linear_blending_negative_clamped() {
        let mut image = vec![-5.0f32];
        let ab = vec![1.0f32, -1.0];
        // result: -5.0 * 1.0 + (-1.0) = -6.0 → clamped to MIN_FLOAT
        apply_linear_blending(&mut image, &ab, 1);
        assert_eq!(image[0], MIN_FLOAT);
    }

    #[test]
    fn apply_linear_blending_matches_reference_over_lcg() {
        // LCG data is finite and non-NaN, so the kernel's `.max(MIN_FLOAT)`
        // (NaN-ignoring, matching C's `fmaxf`) and the reference's
        // `if blended < MIN_FLOAT` produce identical results on all test inputs.
        // The FMA vs no-FMA difference is C-vs-Rust only (documented in module
        // docs), not kernel-vs-reference, since both are Rust with the same FP
        // evaluation order.
        let mut image = vec![0.0f32; 256];
        let mut ab = vec![0.0f32; 512];
        lcg_fill(&mut image, 0x3333, 5.0);
        lcg_fill(&mut ab, 0x4444, 5.0);

        let mut direct = image.clone();
        let mut reference = image.clone();

        apply_linear_blending(&mut direct, &ab, 256);
        ref_apply_linear_blending(&mut reference, &ab, 256);

        assert_eq!(direct, reference);
    }

    // ── FFI round-trip and guard tests ─────────────────────────────────────────

    #[test]
    fn ffi_pack_variance_4c_round_trip() {
        let mut guide = vec![0.0f32; 256];
        let mut mask = vec![0.0f32; 256];
        lcg_fill(&mut guide, 0x5555, 5.0);
        lcg_fill(&mut mask, 0x6666, 5.0);

        let mut ffi_buf = vec![0.0f32; 256 * 4];
        let mut direct_buf = vec![0.0f32; 256 * 4];

        unsafe {
            darkroom_fgf_pack_variance_4c(
                ffi_buf.as_mut_ptr(),
                guide.as_ptr(),
                mask.as_ptr(),
                256,
            );
        }
        pack_variance_4c(&mut direct_buf, &guide, &mask, 256);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_pack_variance_4c_null_guard() {
        unsafe {
            darkroom_fgf_pack_variance_4c(std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), 10);
        }
    }

    #[test]
    fn ffi_pack_variance_4c_zero_n_guard() {
        let guide = vec![1.0f32; 4];
        let mask = vec![1.0f32; 4];
        let mut input = vec![1.0f32; 16];
        unsafe {
            darkroom_fgf_pack_variance_4c(input.as_mut_ptr(), guide.as_ptr(), mask.as_ptr(), 0);
        }
        assert_eq!(input, vec![1.0; 16]); // untouched
    }

    #[test]
    fn ffi_apply_linear_blending_round_trip() {
        // Verifies FFI call path matches the direct Rust kernel call (same FP
        // evaluation, no C-vs-Rust FMA divergence here). The C-vs-Rust FMA
        // difference (≤1 ULP) is documented in the module-level docs and is
        // not exercised by this FFI-to-Rust comparison.
        let mut image = vec![0.0f32; 256];
        let mut ab = vec![0.0f32; 512];
        lcg_fill(&mut image, 0x7777, 5.0);
        lcg_fill(&mut ab, 0x8888, 5.0);

        let mut ffi_buf = image.clone();
        let mut direct_buf = image.clone();

        unsafe {
            darkroom_fgf_apply_linear_blending(ffi_buf.as_mut_ptr(), ab.as_ptr(), 256);
        }
        apply_linear_blending(&mut direct_buf, &ab, 256);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_apply_linear_blending_null_guard() {
        unsafe {
            darkroom_fgf_apply_linear_blending(std::ptr::null_mut(), std::ptr::null(), 10);
        }
    }

    #[test]
    fn ffi_apply_linear_blending_zero_n_guard() {
        let mut image = vec![1.0f32; 4];
        let ab = vec![1.0f32; 8];
        unsafe {
            darkroom_fgf_apply_linear_blending(image.as_mut_ptr(), ab.as_ptr(), 0);
        }
        assert_eq!(image, vec![1.0; 4]); // untouched
    }

    #[test]
    fn ffi_zero_n_guard_both() {
        let mut input = vec![1.0f32; 16];
        let mut image = vec![1.0f32; 4];
        let guide = vec![1.0f32; 4];
        let mask = vec![1.0f32; 4];
        let ab = vec![1.0f32; 8];
        unsafe {
            darkroom_fgf_pack_variance_4c(input.as_mut_ptr(), guide.as_ptr(), mask.as_ptr(), 0);
            darkroom_fgf_apply_linear_blending(image.as_mut_ptr(), ab.as_ptr(), 0);
        }
        assert_eq!(input, vec![1.0; 16]);
        assert_eq!(image, vec![1.0; 4]);
    }

    #[test]
    fn ffi_overflow_guard_pack_variance_4c() {
        // n > i32::MAX should return early without touching the buffer
        let mut input = vec![1.0f32; 4];
        let guide = vec![1.0f32; 4];
        let mask = vec![1.0f32; 4];
        let big_n = (i32::MAX as usize) + 1;
        unsafe {
            darkroom_fgf_pack_variance_4c(input.as_mut_ptr(), guide.as_ptr(), mask.as_ptr(), big_n);
        }
        assert_eq!(input, vec![1.0; 4]); // untouched
    }

    #[test]
    fn ffi_overflow_guard_apply_linear_blending() {
        let mut image = vec![1.0f32; 4];
        let ab = vec![1.0f32; 8];
        let big_n = (i32::MAX as usize) + 1;
        unsafe {
            darkroom_fgf_apply_linear_blending(image.as_mut_ptr(), ab.as_ptr(), big_n);
        }
        assert_eq!(image, vec![1.0; 4]); // untouched
    }

    #[test]
    fn apply_linear_blending_nan_matches_fmaxf() {
        // Kernel uses `.max(MIN_FLOAT)` which is NaN-ignoring (matches C's `fmaxf`):
        // NaN.max(MIN_FLOAT) returns MIN_FLOAT, not NaN.
        // Reference uses `if blended < MIN_FLOAT` which is NaN-propagating:
        // NaN < MIN_FLOAT is false, so reference returns NaN.
        // This test confirms the kernel matches C's `fmaxf` behavior.
        let mut image = vec![f32::NAN, 0.0, -1.0];
        let ab = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        apply_linear_blending(&mut image, &ab, 3);
        // k=0: NaN * 1.0 + 0.0 = NaN → .max(MIN_FLOAT) returns MIN_FLOAT (not NaN)
        assert_eq!(image[0], MIN_FLOAT);
        // k=1: 0.0 * 1.0 + 0.0 = 0.0 → .max(MIN_FLOAT) returns MIN_FLOAT
        assert_eq!(image[1], MIN_FLOAT);
        // k=2: -1.0 * 1.0 + 0.0 = -1.0 → .max(MIN_FLOAT) returns MIN_FLOAT
        assert_eq!(image[2], MIN_FLOAT);
    }

    // ── solve_ab ───────────────────────────────────────────────────────────────

    #[test]
    fn solve_ab_basic_exact() {
        // input = {g=2, p=3, gg=6, gp=9}, feathering = 1
        // d = (6 - 4) + 1 = 3; a = (9 - 2*3)/3 = 1; b = 3 - 1*2 = 1
        let input = vec![2.0f32, 3.0, 6.0, 9.0];
        let mut ab = vec![0.0f32; 2];
        solve_ab(&input, &mut ab, 1, 1.0);
        assert_eq!(ab[0], 1.0);
        assert_eq!(ab[1], 1.0);
    }

    #[test]
    fn solve_ab_constant_guide_zero_slope() {
        // Constant guide: variance zero → d = feathering; covariance zero → a = 0, b = p
        let input = vec![2.0f32, 0.5, 4.0, 1.0];
        let mut ab = vec![f32::NAN; 2];
        solve_ab(&input, &mut ab, 1, 0.1);
        assert_eq!(ab[0], 0.0);
        assert_eq!(ab[1], 0.5);
    }

    #[test]
    fn solve_ab_division_guard_clamps_to_1e_15() {
        // gg - g² = 0 - 1 = -1, feathering = 0 → d clamps to 1e-15;
        // gp - g·p = 0.5 - 0.5 = 0 → a = 0, b = 0.5
        let input = vec![1.0f32, 0.5, 0.0, 0.5];
        let mut ab = vec![f32::NAN; 2];
        solve_ab(&input, &mut ab, 1, 0.0);
        assert_eq!(ab[0], 0.0);
        assert_eq!(ab[1], 0.5);
        // Negative numerator over the tiny guard: still finite (no panic, no inf/NaN)
        let input = vec![1.0f32, 0.5, 0.0, 0.25];
        let mut ab = vec![f32::NAN; 2];
        solve_ab(&input, &mut ab, 1, 0.0);
        assert!(ab[0].is_finite());
        assert!(ab[1].is_finite());
    }

    #[test]
    fn solve_ab_matches_reference_over_lcg() {
        let mut input = vec![0.0f32; 256 * 4];
        lcg_fill(&mut input, 0x9999, 10.0);
        let mut direct = vec![0.0f32; 256 * 2];
        let mut reference = vec![0.0f32; 256 * 2];

        solve_ab(&input, &mut direct, 256, 0.05);
        ref_solve_ab(&input, &mut reference, 256, 0.05);

        assert_eq!(direct, reference);
    }

    // ── apply_linear_blending_w_geomean ────────────────────────────────────────

    #[test]
    fn geomean_basic() {
        // k=0: blend = 4*1 + 0 = 4 → sqrt(4 * 4) = 4
        // k=1: blend = 8*0.5 + 2 = 6 → sqrt(8 * 6) = sqrt(48)
        // k=2: blend = 16*0 + (-1) = -1 → clamped to MIN_FLOAT → sqrt(16 * MIN_FLOAT)
        let mut image = vec![4.0f32, 8.0, 16.0];
        let ab = vec![1.0f32, 0.0, 0.5, 2.0, 0.0, -1.0];
        apply_linear_blending_w_geomean(&mut image, &ab, 3);
        assert_eq!(image[0], 4.0);
        assert_eq!(image[1], 48.0f32.sqrt());
        assert_eq!(image[2], (16.0f32 * MIN_FLOAT).sqrt());
    }

    #[test]
    fn geomean_clamps_negative_blend_to_min_float() {
        let mut image = vec![2.0f32];
        let ab = vec![1.0f32, -8.0];
        // blend = 2*1 - 8 = -6 → clamped to MIN_FLOAT → sqrt(2 * MIN_FLOAT)
        apply_linear_blending_w_geomean(&mut image, &ab, 1);
        assert_eq!(image[0], (2.0f32 * MIN_FLOAT).sqrt());
    }

    #[test]
    fn geomean_matches_reference_over_lcg() {
        let mut image = vec![0.0f32; 256];
        let mut ab = vec![0.0f32; 512];
        lcg_fill(&mut image, 0xAAAA, 5.0);
        lcg_fill(&mut ab, 0xBBBB, 5.0);
        // Keep image positive (its documented domain outside the luminance mask)
        for v in image.iter_mut() {
            *v = v.abs() + 0.5;
        }

        let mut direct = image.clone();
        let mut reference = image.clone();

        apply_linear_blending_w_geomean(&mut direct, &ab, 256);
        ref_apply_linear_blending_w_geomean(&mut reference, &ab, 256);

        assert_eq!(direct, reference);
    }

    // ── quantize ───────────────────────────────────────────────────────────────

    #[test]
    fn quantize_fast_track_basic() {
        // 5.0 → log2 ≈ 2.32 → floor 2 → exp2 = 4; clamped into [0, 10] unchanged
        let image = vec![5.0f32];
        let mut out = vec![f32::NAN; 1];
        quantize(&image, &mut out, 1, 1.0, 0.0, 10.0);
        assert_eq!(out[0], 4.0);
    }

    #[test]
    fn quantize_slow_track_basic() {
        // 3.0, sampling 0.5: log2 ≈ 1.585 → /0.5 = 3.17 → floor 3 → ×0.5 = 1.5 → exp2 ≈ 2.828
        let image = vec![3.0f32];
        let mut out = vec![f32::NAN; 1];
        quantize(&image, &mut out, 1, 0.5, 0.0, 10.0);
        assert_eq!(out[0], (1.5f32).exp2());
    }

    #[test]
    fn quantize_clamps_to_bounds() {
        // 1e9 → log2 ≈ 29.9 → floor 29 → exp2 ≈ 5.4e8 > clip_max → clamps to clip_max
        // 1e-9 → log2 ≈ -29.9 → floor -30 → exp2 ≈ 9.3e-10 < clip_min = 1e-3 → clamps up
        let image = vec![1e9f32, 1e-9];
        let mut out = vec![f32::NAN; 2];
        quantize(&image, &mut out, 2, 1.0, 1e-3, 1e3);
        assert_eq!(out[0], 1e3);
        assert_eq!(out[1], 1e-3);
    }

    #[test]
    fn quantize_nan_input_yields_clip_max() {
        // fminf(NaN, top) = top (NaN-ignoring), then fmaxf(top, bottom) = top
        // → a NaN input quantizes to clip_max, matching standard libm semantics.
        let image = vec![f32::NAN, -1.0]; // log2 of both is NaN
        let mut out = vec![f32::NAN; 2];
        quantize(&image, &mut out, 2, 1.0, 0.1, 7.0);
        assert_eq!(out[0], 7.0);
        assert_eq!(out[1], 7.0);
    }

    #[test]
    fn quantize_matches_reference_over_lcg() {
        let mut image = vec![0.0f32; 256];
        lcg_fill(&mut image, 0xCCCC, 5.0);
        // Keep strictly positive (log2 domain), spanning several octaves
        for v in image.iter_mut() {
            *v = v.abs() * 8.0 + 1e-4;
        }

        let mut direct = vec![0.0f32; 256];
        let mut reference = vec![0.0f32; 256];

        quantize(&image, &mut direct, 256, 0.25, 1e-6, 1e6);
        ref_quantize(&image, &mut reference, 256, 0.25, 1e-6, 1e6);

        assert_eq!(direct, reference);

        // Fast track over the same data
        quantize(&image, &mut direct, 256, 1.0, 1e-6, 1e6);
        ref_quantize(&image, &mut reference, 256, 1.0, 1e-6, 1e6);
        assert_eq!(direct, reference);
    }

    // ── FFI round-trip and guard tests (m4-168 kernels) ────────────────────────

    #[test]
    fn ffi_solve_ab_round_trip() {
        let mut input = vec![0.0f32; 256 * 4];
        lcg_fill(&mut input, 0xDDDD, 10.0);
        let mut ffi_buf = vec![0.0f32; 256 * 2];
        let mut direct_buf = vec![0.0f32; 256 * 2];

        unsafe {
            darkroom_fgf_solve_ab(input.as_ptr(), ffi_buf.as_mut_ptr(), 256, 0.05);
        }
        solve_ab(&input, &mut direct_buf, 256, 0.05);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_solve_ab_guards() {
        unsafe {
            darkroom_fgf_solve_ab(std::ptr::null(), std::ptr::null_mut(), 10, 1.0);
        }
        let input = vec![1.0f32; 4];
        let mut ab = vec![1.0f32; 2];
        unsafe {
            darkroom_fgf_solve_ab(input.as_ptr(), ab.as_mut_ptr(), 0, 1.0);
            darkroom_fgf_solve_ab(
                input.as_ptr(),
                ab.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                1.0,
            );
        }
        assert_eq!(ab, vec![1.0f32; 2]); // untouched
    }

    #[test]
    fn ffi_geomean_round_trip() {
        let mut image = vec![0.0f32; 256];
        let mut ab = vec![0.0f32; 512];
        lcg_fill(&mut image, 0xEEEE, 5.0);
        lcg_fill(&mut ab, 0xFFFF, 5.0);
        for v in image.iter_mut() {
            *v = v.abs() + 0.5;
        }

        let mut ffi_buf = image.clone();
        let mut direct_buf = image.clone();

        unsafe {
            darkroom_fgf_apply_linear_blending_w_geomean(ffi_buf.as_mut_ptr(), ab.as_ptr(), 256);
        }
        apply_linear_blending_w_geomean(&mut direct_buf, &ab, 256);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_geomean_guards() {
        unsafe {
            darkroom_fgf_apply_linear_blending_w_geomean(std::ptr::null_mut(), std::ptr::null(), 10);
        }
        let mut image = vec![1.0f32; 4];
        let ab = vec![1.0f32; 8];
        unsafe {
            darkroom_fgf_apply_linear_blending_w_geomean(image.as_mut_ptr(), ab.as_ptr(), 0);
            darkroom_fgf_apply_linear_blending_w_geomean(
                image.as_mut_ptr(),
                ab.as_ptr(),
                (i32::MAX as usize) + 1,
            );
        }
        assert_eq!(image, vec![1.0f32; 4]); // untouched
    }

    #[test]
    fn ffi_quantize_round_trip() {
        let mut image = vec![0.0f32; 256];
        lcg_fill(&mut image, 0x12F3, 5.0);
        for v in image.iter_mut() {
            *v = v.abs() * 8.0 + 1e-4;
        }
        let mut ffi_buf = vec![0.0f32; 256];
        let mut direct_buf = vec![0.0f32; 256];

        unsafe {
            darkroom_fgf_quantize(image.as_ptr(), ffi_buf.as_mut_ptr(), 256, 0.25, 1e-6, 1e6);
        }
        quantize(&image, &mut direct_buf, 256, 0.25, 1e-6, 1e6);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_quantize_guards() {
        unsafe {
            darkroom_fgf_quantize(std::ptr::null(), std::ptr::null_mut(), 10, 1.0, 0.0, 1.0);
        }
        let image = vec![1.0f32; 4];
        let mut out = vec![1.0f32; 4];
        unsafe {
            darkroom_fgf_quantize(image.as_ptr(), out.as_mut_ptr(), 0, 1.0, 0.0, 1.0);
            darkroom_fgf_quantize(
                image.as_ptr(),
                out.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                1.0,
                0.0,
                1.0,
            );
        }
        assert_eq!(out, vec![1.0f32; 4]); // untouched
    }

    // ── interpolate_bilinear ───────────────────────────────────────────────────

    #[test]
    fn bilinear_identity_single_pixel() {
        // 1x1 → 1x1: all four corners are the same pixel → out == in exactly
        let src = vec![7.5f32];
        let mut out = vec![f32::NAN; 1];
        interpolate_bilinear(&src, 1, 1, &mut out, 1, 1, 1);
        assert_eq!(out[0], 7.5);
    }

    #[test]
    fn bilinear_upscale_weights() {
        // 2x1 → 4x1, ch=1: interior samples are exact midpoints at j=1 (0.5, 0.5)
        let src = vec![1.0f32, 3.0];
        let mut out = vec![f32::NAN; 4];
        interpolate_bilinear(&src, 2, 1, &mut out, 4, 1, 1);
        // j=0: x_in=0 → in[0]; j=2: x_in=1.0 → floor 1, next clamped → in[1]
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0); // (1 + 3)/2
        assert_eq!(out[2], 3.0);
        // j=3: x_out=0.75 → x_in=1.5 → prev=1, next clamped to 1 → in[1]
        assert_eq!(out[3], 3.0);
    }

    #[test]
    fn bilinear_border_clamp_weights_sum_to_one() {
        // 1x1 → 2x2: the clamped-neighbour quirk yields negative Dx_next/Dy_next
        // with weights still summing to 1, so the constant image reproduces exactly
        let src = vec![4.0f32];
        let mut out = vec![f32::NAN; 4];
        interpolate_bilinear(&src, 1, 1, &mut out, 2, 2, 1);
        assert_eq!(out, vec![4.0f32; 4]);
    }

    #[test]
    fn bilinear_multichannel() {
        // 1x2 → 1x2 with ch=3: rows interpolate independently per channel
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![f32::NAN; 6];
        interpolate_bilinear(&src, 1, 2, &mut out, 1, 2, 3);
        // i=0: y_in=0 → row 0; i=1: y_in=1.0 → prev=0? floor(1.0)=1 → row 1
        assert_eq!(out[0..3], src[0..3]);
        assert_eq!(out[3..6], src[3..6]);
    }

    #[test]
    fn bilinear_upscale_samples_and_blends() {
        // 4x1 → 8x1 (upscale): j=0 lands exactly on node 0; j=3 (x_in=1.5) is
        // an exact midpoint; j=7 (x_in=3.5) has its right neighbour clamped
        // to node 3. True downscale coverage lives in the LCG test below.
        let src = vec![0.0f32, 4.0, 8.0, 12.0];
        let mut out = vec![f32::NAN; 8];
        interpolate_bilinear(&src, 4, 1, &mut out, 8, 1, 1);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[3], 6.0); // 0.5*4 + 0.5*8
        assert_eq!(out[7], 12.0);
    }

    #[test]
    fn bilinear_matches_reference_over_lcg() {
        let mut src = vec![0.0f32; 16 * 8 * 4];
        lcg_fill(&mut src, 0x1B1B, 10.0);
        let mut direct = vec![0.0f32; 32 * 16 * 4];
        let mut reference = vec![0.0f32; 32 * 16 * 4];

        interpolate_bilinear(&src, 16, 8, &mut direct, 32, 16, 4);
        ref_interpolate_bilinear(&src, 16, 8, &mut reference, 32, 16, 4);
        assert_eq!(direct, reference);

        // Downscale direction too, and a non-power-of-two pair so the
        // coordinate-division rounding path is actually compared (the
        // power-of-two dims divide exactly and skip it)
        let mut direct_ds = vec![0.0f32; 8 * 4 * 4];
        let mut reference_ds = vec![0.0f32; 8 * 4 * 4];
        interpolate_bilinear(&src, 16, 8, &mut direct_ds, 8, 4, 4);
        ref_interpolate_bilinear(&src, 16, 8, &mut reference_ds, 8, 4, 4);
        assert_eq!(direct_ds, reference_ds);

        let mut src_npo = vec![0.0f32; 15 * 9 * 4];
        lcg_fill(&mut src_npo, 0x3D3D, 10.0);
        let mut direct_npo = vec![0.0f32; 23 * 13 * 4];
        let mut reference_npo = vec![0.0f32; 23 * 13 * 4];
        interpolate_bilinear(&src_npo, 15, 9, &mut direct_npo, 23, 13, 4);
        ref_interpolate_bilinear(&src_npo, 15, 9, &mut reference_npo, 23, 13, 4);
        assert_eq!(direct_npo, reference_npo);
    }

    #[test]
    fn ffi_bilinear_round_trip() {
        let mut src = vec![0.0f32; 16 * 8 * 2];
        lcg_fill(&mut src, 0x2C2C, 10.0);
        let mut ffi_buf = vec![0.0f32; 24 * 12 * 2];
        let mut direct_buf = vec![0.0f32; 24 * 12 * 2];

        unsafe {
            darkroom_fgf_interpolate_bilinear(
                src.as_ptr(),
                16,
                8,
                ffi_buf.as_mut_ptr(),
                24,
                12,
                2,
            );
        }
        interpolate_bilinear(&src, 16, 8, &mut direct_buf, 24, 12, 2);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_bilinear_guards() {
        unsafe {
            darkroom_fgf_interpolate_bilinear(
                std::ptr::null(),
                4,
                4,
                std::ptr::null_mut(),
                2,
                2,
                1,
            );
        }
        let src = vec![1.0f32; 4];
        let mut out = vec![1.0f32; 4];
        // Zero dims, zero ch, oversized dim
        unsafe {
            darkroom_fgf_interpolate_bilinear(src.as_ptr(), 0, 4, out.as_mut_ptr(), 2, 2, 1);
            darkroom_fgf_interpolate_bilinear(src.as_ptr(), 4, 0, out.as_mut_ptr(), 2, 2, 1);
            darkroom_fgf_interpolate_bilinear(
                src.as_ptr(),
                4,
                4,
                out.as_mut_ptr(),
                2,
                2,
                0,
            );
            darkroom_fgf_interpolate_bilinear(
                src.as_ptr(),
                4,
                4,
                out.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                2,
                1,
            );
        }
        assert_eq!(out, vec![1.0f32; 4]); // untouched
    }
}
