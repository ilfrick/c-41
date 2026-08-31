//! Kernels ported from `src/common/guided_filter.c` (`_guided_filter_tiling`,
//! the CPU tile pipeline of the guided filter IOP). Two loops are ported
//! here (m4-173):
//! - The Cramer-rule 3x3 solve loop (formerly at guided_filter.c:163,
//!   `DT_OMP_FOR` over the flat tile): per tile pixel it solves
//!   `Sigma · a = cov` for the colour guide and derives the offset `b`,
//!   overwriting the packed mean buffer in place (the C code recycles the
//!   `mean` allocation as `a_b`, so the output aliases the input means).
//! - The tile apply loop (formerly at guided_filter.c:221, `DT_OMP_FOR` over
//!   rows of the target rectangle): per output pixel it evaluates
//!   `guide_weight * (a_r·g_r + a_g·g_g + a_b·g_b) + b` against the guide
//!   and clamps to `[min, max]` with GLib's `CLAMP`.
//!
//! The Rust kernels are single-threaded sequential; LLVM's auto-vectorizer
//! provides SIMD at `-O3`, but multi-threaded parallelism is no longer used.
//! This matches the m4-161 `blend.rs` and m4-162 `imagebuf.rs` pattern.
//!
//! Bit-exactness notes:
//! - The C file is compiled with the repo-wide Release flags
//!   `-O3 -ffast-math -fno-finite-math-only` and GCC's default
//!   `-ffp-contract=fast` for C99+. Both ported loops are dense
//!   multiply-subtract / multiply-add chains (the Sigma entries
//!   `var - guide*guide`, the Cramer determinants, `b = inp - a_r*guide_r - ...`,
//!   and `guide_weight * (a_r*px_r + a_g*px_g + a_b*px_b) + b`), so the C
//!   binary may contract several `a*b ± c` sites into FMAs where Rust (no
//!   contraction in the release profile) keeps separate mul/sub ops. The
//!   C-vs-Rust difference is the order-ULP class accepted repo-wide
//!   (cf. `eigf.rs`, `fast_guided_filter.rs`); it can additionally flip the
//!   `fabsf(det0) > 4.f * FLT_EPSILON` singularity branch for determinants
//!   within a few ULP of the threshold — a knife-edge the C code itself
//!   sits on across compilers. `-ffast-math` reassociation has no clean ULP
//!   bound; GCC did not appear to reassociate these expressions at -O3
//!   (verified against a standalone compile of the C loop), but that is a
//!   compiler-version observation, not a guarantee.
//! - The branch and the clamps are exact: `fabsf` maps to `f32::abs`, and
//!   `4.f * FLT_EPSILON` is `4.0f32 * f32::EPSILON` (exactly representable).
//! - The apply pass clamps with GLib's `CLAMP(x, low, high)` macro
//!   (gmacros.h:942), a double ternary
//!   `((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x))` — NOT
//!   `fminf`/`fmaxf` and NOT Rust's `f32::clamp`. A NaN `res` fails both
//!   comparisons and passes through unchanged. The kernel replicates the
//!   ternaries exactly (`glib_clamp`); the NaN pass-through is pinned by a
//!   test. Using `.min()/.max()/.clamp()` would be wrong (NaN-ignoring or
//!   panicking on NaN).
//! - The singular branch (`|det0| <= 4*FLT_EPSILON`) re-reads the input
//!   mean via `_get_color_pixel(mean, i)[INP_MEAN]`; since all four result
//!   writes happen after every read of the element, that re-read equals the
//!   `inp_mean` loaded at the top of the iteration — the kernel keeps the
//!   read-before-write ordering and uses the single load.
//! - The solve output aliases the input means (C: `color_image a_b = mean`),
//!   so the kernel takes a single `&mut [f32]` for mean/a_b. Per element all
//!   reads precede all writes, so in-place operation is race-free even
//!   though the C loop was OpenMP-parallel (each iteration touches only its
//!   own 4-float slot).
//! - The apply pass reads guide channels 0..3 only (`guide_stride = ch` can
//!   exceed 3; higher channels are ignored, matching C).
//! - `guide_weight` appears twice by upstream design, as two halves of one
//!   model: the solve loop folds it into the packed guide means
//!   (`pixel[k] *= guide_weight` at pack time), and the apply pass
//!   multiplies the raw guide pixels by it again when reconstructing
//!   (`w·(a·g) + b = a·(w·g) + b ≈ inp`). Dropping either factor would
//!   break the reconstruction — do NOT "fix" one side.
//! - The C `#define`s `INP_MEAN`/`GUIDE_MEAN_*`/`COV_*`/`VAR_*` (and the
//!   now kernel-only `A_RED`/`A_GREEN`/`A_BLUE`/`B`) survive in
//!   guided_filter.c to document the channel layout; the constants below
//!   mirror them.

/// Channel offsets in the packed 4-float mean/a_b pixel (C `#define`s in
/// `_guided_filter_tiling`; the same slots hold `{inp_mean, guide_r, guide_g,
/// guide_b}` on input and `{a_r, a_g, a_b, b}` after the solve).
const INP_MEAN: usize = 0;
const GUIDE_MEAN_R: usize = 1;
const GUIDE_MEAN_G: usize = 2;
const GUIDE_MEAN_B: usize = 3;
const A_RED: usize = 0;
const A_GREEN: usize = 1;
const A_BLUE: usize = 2;
const B: usize = 3;

/// Channel offsets in the packed 9-float variance pixel (C `#define`s; note
/// the C file defines `VAR_BB 8` before `VAR_GB 7` — the values, not the
/// definition order, are what matters).
const COV_R: usize = 0;
const COV_G: usize = 1;
const COV_B: usize = 2;
const VAR_RR: usize = 3;
const VAR_RG: usize = 4;
const VAR_RB: usize = 5;
const VAR_GG: usize = 6;
const VAR_GB: usize = 7;
const VAR_BB: usize = 8;

/// GLib's `CLAMP(x, low, high)` macro (gmacros.h:942): a double ternary,
/// not `fminf`/`fmaxf` and not `f32::clamp`. NaN passes through unchanged
/// (both comparisons are false). Replicated exactly (see module docs).
#[inline]
fn glib_clamp(x: f32, low: f32, high: f32) -> f32 {
    if x > high {
        high
    } else if x < low {
        low
    } else {
        x
    }
}

/// Cramer-rule 3x3 solve for the guided filter tile.
///
/// Port of the `DT_OMP_FOR` loop at guided_filter.c:163 (the former
/// element-wise solve loop). `buf` holds `size` packed 4-float pixels
/// `{inp_mean, guide_r, guide_g, guide_b}` and receives the solve results
/// in place as `{a_r, a_g, a_b, b}` (the C code recycles the mean buffer as
/// `a_b`). `variance` holds `size` packed 9-float pixels
/// `{cov_r, cov_g, cov_b, var_rr, var_rg, var_rb, var_gg, var_gb, var_bb}`.
/// `eps` is the regulariser added to the variance diagonal.
///
/// Per element: builds the symmetric 3x3 `Sigma` (variance minus outer
/// product of the guide means, `+eps` on the diagonal), computes `det0`; if
/// `|det0| > 4*FLT_EPSILON` solves `a` via Cramer's rule (det1..det3) and
/// `b = inp_mean − a_r·guide_r − a_g·guide_g − a_b·guide_b`, else the
/// system is singular and `a = 0, b = inp_mean`. Every read of the element
/// precedes every write, preserving the C in-place aliasing semantics.
pub fn guided_filter_solve(buf: &mut [f32], variance: &[f32], size: usize, eps: f32) {
    let m = size.min(buf.len() / 4).min(variance.len() / 9);
    for i in 0..m {
        let inp_mean = buf[4 * i + INP_MEAN];
        let guide_r = buf[4 * i + GUIDE_MEAN_R];
        let guide_g = buf[4 * i + GUIDE_MEAN_G];
        let guide_b = buf[4 * i + GUIDE_MEAN_B];
        // solve linear system of equations of size 3x3 via Cramer's rule
        // symmetric coefficient matrix (expression structure transcribed
        // op-for-op from the C loop)
        let sigma_0_0 = variance[9 * i + VAR_RR] - (guide_r * guide_r) + eps;
        let sigma_0_1 = variance[9 * i + VAR_RG] - (guide_r * guide_g);
        let sigma_0_2 = variance[9 * i + VAR_RB] - (guide_r * guide_b);
        let sigma_1_1 = variance[9 * i + VAR_GG] - (guide_g * guide_g) + eps;
        let sigma_1_2 = variance[9 * i + VAR_GB] - (guide_g * guide_b);
        let sigma_2_2 = variance[9 * i + VAR_BB] - (guide_b * guide_b) + eps;
        let det0 = sigma_0_0 * (sigma_1_1 * sigma_2_2 - sigma_1_2 * sigma_1_2)
            - sigma_0_1 * (sigma_0_1 * sigma_2_2 - sigma_0_2 * sigma_1_2)
            + sigma_0_2 * (sigma_0_1 * sigma_1_2 - sigma_0_2 * sigma_1_1);
        let (a_r_, a_g_, a_b_, b_);
        if f32::abs(det0) > 4.0f32 * f32::EPSILON {
            let cov_r = variance[9 * i + COV_R] - guide_r * inp_mean;
            let cov_g = variance[9 * i + COV_G] - guide_g * inp_mean;
            let cov_b = variance[9 * i + COV_B] - guide_b * inp_mean;
            let det1 = cov_r * (sigma_1_1 * sigma_2_2 - sigma_1_2 * sigma_1_2)
                - sigma_0_1 * (cov_g * sigma_2_2 - cov_b * sigma_1_2)
                + sigma_0_2 * (cov_g * sigma_1_2 - cov_b * sigma_1_1);
            let det2 = sigma_0_0 * (cov_g * sigma_2_2 - cov_b * sigma_1_2)
                - cov_r * (sigma_0_1 * sigma_2_2 - sigma_0_2 * sigma_1_2)
                + sigma_0_2 * (sigma_0_1 * cov_b - sigma_0_2 * cov_g);
            let det3 = sigma_0_0 * (sigma_1_1 * cov_b - sigma_1_2 * cov_g)
                - sigma_0_1 * (sigma_0_1 * cov_b - sigma_0_2 * cov_g)
                + cov_r * (sigma_0_1 * sigma_1_2 - sigma_0_2 * sigma_1_1);
            a_r_ = det1 / det0;
            a_g_ = det2 / det0;
            a_b_ = det3 / det0;
            b_ = inp_mean - a_r_ * guide_r - a_g_ * guide_g - a_b_ * guide_b;
        } else {
            // linear system is singular
            a_r_ = 0.0f32;
            a_g_ = 0.0f32;
            a_b_ = 0.0f32;
            b_ = inp_mean;
        }
        // now data of imgg_mean_? is no longer needed, we can safely overwrite aliasing arrays
        buf[4 * i + A_RED] = a_r_;
        buf[4 * i + A_GREEN] = a_g_;
        buf[4 * i + A_BLUE] = a_b_;
        buf[4 * i + B] = b_;
    }
}

/// Apply pass of the guided filter tile.
///
/// Port of the `DT_OMP_FOR` loop at guided_filter.c:221 (the former
/// row-loop over the target rectangle). `guide` is the colour guide image
/// with `guide_stride` floats per pixel (ch >= 3; channels 0..3 are read);
/// `ab` holds the blurred `{a_r, a_g, a_b, b}` coefficients, 4 floats per
/// tile pixel; `out` receives one float per pixel at the guide-sample
/// positions.
///
/// Geometry (matching the C index math): for each guide row
/// `j in [target_lower, target_upper)` the guide pixel run starts at
/// `target_left + j*guide_width` and the ab pixel run starts at
/// `(target_left − source_left) + (j − source_lower)*tile_width`, where
/// `tile_width` is the source-tile width. Per column:
/// `out = CLAMP(guide_weight * (a_r·g_r + a_g·g_g + a_b·g_b) + b, min, max)`
/// with GLib's ternary `CLAMP` (NaN passes through — see module docs).
///
/// Degenerate or inconsistent geometry (empty rect, zero widths, source
/// origin outside the target — the C caller always passes `source ⊇ target`)
/// is a no-op; out-of-range columns are skipped by clamped iteration, as in
/// the other ported modules.
#[allow(clippy::too_many_arguments)]
pub fn guided_filter_apply(
    guide: &[f32],
    guide_stride: usize,
    ab: &[f32],
    out: &mut [f32],
    guide_width: usize,
    target_left: usize,
    target_right: usize,
    target_lower: usize,
    target_upper: usize,
    source_left: usize,
    source_lower: usize,
    tile_width: usize,
    guide_weight: f32,
    min: f32,
    max: f32,
) {
    if target_right <= target_left
        || target_upper <= target_lower
        || guide_width == 0
        || tile_width == 0
        || guide_stride == 0
        || source_left > target_left
        || source_lower > target_lower
    {
        return;
    }
    let cols = target_right - target_left;
    // guide and out share their pixel indexing (pixel p lives at
    // p*guide_stride in guide and at p in out), like the C loop's l cursor
    let guide_pixels = guide.len() / guide_stride;
    let ab_pixels = ab.len() / 4;
    for j in target_lower..target_upper {
        let l = target_left.saturating_add(j.saturating_mul(guide_width));
        let k = (target_left - source_left)
            .saturating_add(j.saturating_sub(source_lower).saturating_mul(tile_width));
        // clamped iteration: cap the row at what fits in each buffer
        let cols = cols
            .min(guide_pixels.saturating_sub(l))
            .min(ab_pixels.saturating_sub(k))
            .min(out.len().saturating_sub(l));
        for i in 0..cols {
            let pixel = &guide[(l + i).saturating_mul(guide_stride)..];
            let px_ab = &ab[(k + i) * 4..];
            let mut res = guide_weight
                * (px_ab[A_RED] * pixel[0]
                    + px_ab[A_GREEN] * pixel[1]
                    + px_ab[A_BLUE] * pixel[2]);
            res += px_ab[B];
            out[l + i] = glib_clamp(res, min, max);
        }
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `mean_ab` must hold at least `size * 4` floats (packed means on entry,
/// solve results on return — the buffers alias, matching C); `variance` at
/// least `size * 9` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_guided_filter_solve(
    mean_ab: *mut f32,
    variance: *const f32,
    size: usize,
    eps: f32,
) {
    if mean_ab.is_null() || variance.is_null() || size == 0 || size > i32::MAX as usize {
        return;
    }
    let mean_ab = std::slice::from_raw_parts_mut(mean_ab, size * 4);
    let variance = std::slice::from_raw_parts(variance, size * 9);
    guided_filter_solve(mean_ab, variance, size, eps);
}

/// # Safety
/// `guide` must hold at least
/// `((target_upper - 1) * guide_width + target_right) * guide_stride`
/// floats, `ab` at least
/// `((target_left - source_left) + (target_upper - 1 - source_lower) *
/// tile_width + (target_right - target_left)) * 4` floats and `out` at
/// least `(target_upper - 1) * guide_width + target_right` floats — i.e.
/// exactly the elements the target rectangle addresses. `guide_stride` is
/// the guide channel count (>= 3; only channels 0..3 are read). The caller
/// guarantees `source ⊇ target` (as `_guided_filter_tiling` does).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_guided_filter_apply(
    guide: *const f32,
    guide_stride: usize,
    ab: *const f32,
    out: *mut f32,
    guide_width: usize,
    target_left: usize,
    target_right: usize,
    target_lower: usize,
    target_upper: usize,
    source_left: usize,
    source_lower: usize,
    tile_width: usize,
    guide_weight: f32,
    min: f32,
    max: f32,
) {
    if guide.is_null()
        || ab.is_null()
        || out.is_null()
        || target_right <= target_left
        || target_upper <= target_lower
        || guide_width == 0
        || tile_width == 0
        || guide_stride < 3
        || source_left > target_left
        || source_lower > target_lower
        || guide_width > i32::MAX as usize
        || tile_width > i32::MAX as usize
        || guide_stride > i32::MAX as usize
        || target_right > i32::MAX as usize
        || target_upper > i32::MAX as usize
    {
        return;
    }
    // exact slice lengths for the addressed rectangle (all dims are
    // <= i32::MAX and the subtractions above are guarded, so these fit)
    let guide_pixels = (target_upper - 1) * guide_width + target_right;
    let ab_pixels = (target_left - source_left)
        + (target_upper - 1 - source_lower) * tile_width
        + (target_right - target_left);
    let guide = std::slice::from_raw_parts(guide, guide_pixels * guide_stride);
    let ab = std::slice::from_raw_parts(ab, ab_pixels * 4);
    let out = std::slice::from_raw_parts_mut(out, guide_pixels);
    guided_filter_apply(
        guide,
        guide_stride,
        ab,
        out,
        guide_width,
        target_left,
        target_right,
        target_lower,
        target_upper,
        source_left,
        source_lower,
        tile_width,
        guide_weight,
        min,
        max,
    );
}

// ── Independent reference implementations for bit-exactness tests ─────────────
//
// Both refs are structurally divergent but keep the kernel's FP evaluation
// order: the solve ref runs a general 3x3 column-replacement Cramer helper
// on 9-element matrices (instead of the kernel's inline per-determinant
// expansions — for the symmetric inputs here the helper expands to exactly
// the same multiply/subtract/add sequence), and the apply ref walks the
// rectangle with C-style incrementing cursors instead of recomputed indices.

#[allow(dead_code)]
fn ref_cramer_det(m: &[f32; 9]) -> f32 {
    // det of the 3x3 row-major matrix, expanded along the first row
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

#[allow(dead_code)]
fn ref_cramer_replace_col(sigma: &[f32; 9], cov: &[f32; 3], col: usize) -> [f32; 9] {
    let mut m = *sigma;
    for r in 0..3 {
        m[r * 3 + col] = cov[r];
    }
    m
}

#[allow(dead_code)]
fn ref_guided_filter_solve(buf: &mut [f32], variance: &[f32], size: usize, eps: f32) {
    let m = size.min(buf.len() / 4).min(variance.len() / 9);
    for i in 0..m {
        let inp_mean = buf[4 * i + INP_MEAN];
        let g_r = buf[4 * i + GUIDE_MEAN_R];
        let g_g = buf[4 * i + GUIDE_MEAN_G];
        let g_b = buf[4 * i + GUIDE_MEAN_B];
        // symmetric Sigma as a 9-element row-major matrix
        let sigma: [f32; 9] = [
            variance[9 * i + VAR_RR] - (g_r * g_r) + eps,
            variance[9 * i + VAR_RG] - (g_r * g_g),
            variance[9 * i + VAR_RB] - (g_r * g_b),
            variance[9 * i + VAR_RG] - (g_r * g_g),
            variance[9 * i + VAR_GG] - (g_g * g_g) + eps,
            variance[9 * i + VAR_GB] - (g_g * g_b),
            variance[9 * i + VAR_RB] - (g_r * g_b),
            variance[9 * i + VAR_GB] - (g_g * g_b),
            variance[9 * i + VAR_BB] - (g_b * g_b) + eps,
        ];
        let det0 = ref_cramer_det(&sigma);
        let (a_r_, a_g_, a_b_, b_);
        if f32::abs(det0) > 4.0f32 * f32::EPSILON {
            let cov: [f32; 3] = [
                variance[9 * i + COV_R] - g_r * inp_mean,
                variance[9 * i + COV_G] - g_g * inp_mean,
                variance[9 * i + COV_B] - g_b * inp_mean,
            ];
            // solve by replacing one Sigma column with the cov vector each time
            let m1 = ref_cramer_replace_col(&sigma, &cov, 0);
            let m2 = ref_cramer_replace_col(&sigma, &cov, 1);
            let m3 = ref_cramer_replace_col(&sigma, &cov, 2);
            a_r_ = ref_cramer_det(&m1) / det0;
            a_g_ = ref_cramer_det(&m2) / det0;
            a_b_ = ref_cramer_det(&m3) / det0;
            b_ = inp_mean - a_r_ * g_r - a_g_ * g_g - a_b_ * g_b;
        } else {
            a_r_ = 0.0f32;
            a_g_ = 0.0f32;
            a_b_ = 0.0f32;
            b_ = inp_mean;
        }
        buf[4 * i + A_RED] = a_r_;
        buf[4 * i + A_GREEN] = a_g_;
        buf[4 * i + A_BLUE] = a_b_;
        buf[4 * i + B] = b_;
    }
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ref_guided_filter_apply(
    guide: &[f32],
    guide_stride: usize,
    ab: &[f32],
    out: &mut [f32],
    guide_width: usize,
    target_left: usize,
    target_right: usize,
    target_lower: usize,
    target_upper: usize,
    source_left: usize,
    source_lower: usize,
    tile_width: usize,
    guide_weight: f32,
    min: f32,
    max: f32,
) {
    if target_right <= target_left
        || target_upper <= target_lower
        || guide_width == 0
        || tile_width == 0
        || guide_stride == 0
    {
        return;
    }
    // C-style incrementing row/column cursors (l and k advance by one per
    // column, exactly like the removed loop)
    let mut l = target_left + target_lower * guide_width;
    let mut k = (target_left - source_left) + (target_lower - source_lower) * tile_width;
    let mut o = l;
    for _j in target_lower..target_upper {
        for _i in target_left..target_right {
            if o < out.len() && (k + 1) * 4 <= ab.len() && (l + 1) * guide_stride <= guide.len() {
                let mut res = guide_weight
                    * (ab[4 * k] * guide[guide_stride * l]
                        + ab[4 * k + 1] * guide[guide_stride * l + 1]
                        + ab[4 * k + 2] * guide[guide_stride * l + 2]);
                res += ab[4 * k + 3];
                let clamped = if res > max {
                    max
                } else if res < min {
                    min
                } else {
                    res
                };
                out[o] = clamped;
            }
            k += 1;
            l += 1;
            o += 1;
        }
        l += guide_width - (target_right - target_left);
        o += guide_width - (target_right - target_left);
        k += tile_width - (target_right - target_left);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    // ── guided_filter_solve ─────────────────────────────────────────────────────

    #[test]
    fn solve_nonsingular_pin() {
        // Hand-computed pin, verified against a standalone -O3 compile of
        // the C loop (gcc -O3 -ffast-math): mean {inp=2, g_r=g_g=g_b=0},
        // variance {cov_r=15, cov_g=7, cov_b=9, var_*=0}, eps=1 →
        // Sigma = diag(1, 1, 1) (var_* - 0² + eps), det0 = 1,
        // a_r = 15/1 = 15, a_g = 7, a_b = 9, b = 2 - 0 = 2.
        let mut buf = vec![2.0f32, 0.0, 0.0, 0.0];
        let variance = vec![15.0f32, 7.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        guided_filter_solve(&mut buf, &variance, 1, 1.0);
        assert_eq!(buf, vec![15.0f32, 7.0, 9.0, 2.0]);
    }

    #[test]
    fn solve_nonsingular_with_guide_pin() {
        // Non-zero guide mean: Sigma00 = var_rr - g_r² + eps.
        // mean {inp=2, g=(1,0,0)}, var {cov=(15,7,9), rr=4, others 0},
        // eps=1 → Sigma = diag(4, 1, 1) (S01 = 0 - 1*0 = 0), det0 = 4;
        // cov_r = 15 - 1*2 = 13, cov_g = 7, cov_b = 9.
        // det1 = 13*(1*1) = 13 → a_r = 13/4; det2 = 4*(7*1) = 28 → a_g = 7;
        // det3 = 4*(1*9) = 36 → a_b = 9; b = 2 - 13/4*1 = -5/4.
        let mut buf = vec![2.0f32, 1.0, 0.0, 0.0];
        let variance = vec![15.0f32, 7.0, 9.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        guided_filter_solve(&mut buf, &variance, 1, 1.0);
        assert_eq!(buf[0], 13.0f32 / 4.0);
        assert_eq!(buf[1], 7.0f32);
        assert_eq!(buf[2], 9.0f32);
        assert_eq!(buf[3], -1.25f32);
    }

    #[test]
    fn solve_singular_pin() {
        // variance all 0 and eps = 0 → Sigma = 0, det0 = 0, which fails the
        // |det0| > 4*FLT_EPSILON test → a's 0, b = inp_mean.
        let mut buf = vec![2.0f32, 0.5, -0.25, 0.125];
        let variance = vec![0.0f32; 9];
        guided_filter_solve(&mut buf, &variance, 1, 0.0);
        assert_eq!(buf, vec![0.0f32, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn solve_singular_threshold_boundary() {
        // Exactly at the threshold the strict > is false → singular branch.
        // Diagonal Sigma with guides 0, eps 0, var_rr = var_gg = var_bb =
        // 2^-7 → det0 = (2^-7)³ = 2^-21 = 4*FLT_EPSILON exactly; the strict
        // > fails and the singular branch runs.
        let t = 2.0f32.powi(-7);
        let mut buf = vec![0.75f32, 0.0, 0.0, 0.0];
        let variance = vec![0.0f32, 0.0, 0.0, t, 0.0, 0.0, t, 0.0, t];
        guided_filter_solve(&mut buf, &variance, 1, 0.0);
        assert_eq!(buf, vec![0.0f32, 0.0, 0.0, 0.75]);

        // Above the threshold the solve runs: var_rr = 2^-6 (all off-diag
        // and other guide means 0) → det0 = 2^-6 * (2^-7 * 2^-7) = 2^-20,
        // every step exact in f32; a_r = cov_r*(t*t) / det0 = 50*64 = 3200.
        let mut buf2 = vec![0.75f32, 0.0, 0.0, 0.0];
        let variance2 = vec![50.0f32, 0.0, 0.0, 2.0f32.powi(-6), 0.0, 0.0, t, 0.0, t];
        guided_filter_solve(&mut buf2, &variance2, 1, 0.0);
        assert_eq!(buf2[0], 3200.0f32);
        assert_eq!(buf2[1], 0.0);
        assert_eq!(buf2[2], 0.0);
        assert_eq!(buf2[3], 0.75);
    }

    #[test]
    fn solve_in_place_aliasing() {
        // The results overwrite the mean slots; a later element's inputs
        // must not be corrupted by an earlier element's writes (the C loop
        // was parallel over disjoint 4-float slots).
        let mut buf = vec![2.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
        let variance = vec![
            15.0f32, 7.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ];
        guided_filter_solve(&mut buf, &variance, 2, 1.0);
        assert_eq!(buf[..4], vec![15.0f32, 7.0, 9.0, 2.0]);
        assert_eq!(buf[4..], vec![30.0f32, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn solve_matches_reference_over_lcg() {
        let mut buf = vec![0.0f32; 512 * 4];
        let mut variance = vec![0.0f32; 512 * 9];
        lcg_fill(&mut buf, 0x6F17, 2.0);
        lcg_fill(&mut variance, 0x6F18, 0.5);
        // keep the guide means away from the singular knife-edge: give the
        // diagonal a strong eps so det0 stays well above the threshold
        let mut direct = buf.clone();
        let mut reference = buf.clone();

        guided_filter_solve(&mut direct, &variance, 512, 4.0);
        ref_guided_filter_solve(&mut reference, &variance, 512, 4.0);

        assert_eq!(direct, reference);
    }

    #[test]
    fn solve_matches_reference_over_lcg_mixed() {
        // Without the eps boost: some elements fall through to the singular
        // branch (both paths exercised against the reference).
        let mut buf = vec![0.0f32; 512 * 4];
        let mut variance = vec![0.0f32; 512 * 9];
        lcg_fill(&mut buf, 0x6F19, 0.05);
        lcg_fill(&mut variance, 0x6F1A, 0.01);
        let mut direct = buf.clone();
        let mut reference = buf.clone();

        guided_filter_solve(&mut direct, &variance, 512, 1e-4);
        ref_guided_filter_solve(&mut reference, &variance, 512, 1e-4);

        assert_eq!(direct, reference);
        // sanity: the run actually contained both branches
        let solved = direct
            .chunks_exact(4)
            .filter(|p| p[0] != 0.0 || p[1] != 0.0 || p[2] != 0.0)
            .count();
        assert!(solved > 0 && solved < 512, "solved = {solved}");
    }

    // ── guided_filter_apply ─────────────────────────────────────────────────────

    // 4x4 tile, guide ch=3; ab pixel 5 overridden to isolate one column.
    fn apply_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut guide = Vec::new();
        for _ in 0..16 {
            guide.extend_from_slice(&[2.0f32, 3.0, 4.0]);
        }
        let mut ab = Vec::new();
        for _ in 0..16 {
            ab.extend_from_slice(&[1.0f32, 0.0, 0.0, 10.0]);
        }
        ab[5 * 4] = 0.0;
        ab[5 * 4 + 1] = 1.0;
        ab[5 * 4 + 2] = 0.0;
        ab[5 * 4 + 3] = 0.0;
        (guide, ab, vec![0.0f32; 16])
    }

    #[test]
    fn apply_no_clamp_pin() {
        // res = guide_weight * (a_r*2 + a_g*3 + a_b*4) + b, weight 1,
        // min/max far outside → no clamping. Pixel 5 (row 1, col 1) uses
        // {0,1,0,0} → res = 3; every other pixel {1,0,0,10} → res = 12.
        let (guide, ab, mut out) = apply_fixture();
        guided_filter_apply(
            &guide, 3, &ab, &mut out, 4, 0, 4, 0, 4, 0, 0, 4, 1.0, -1.0e30, 1.0e30,
        );
        for (i, v) in out.iter().enumerate() {
            let expect = if i == 5 { 3.0f32 } else { 12.0f32 };
            assert_eq!(*v, expect, "pixel {i}");
        }
    }

    #[test]
    fn apply_nontrivial_geometry_pin() {
        // target = rows 1..3, cols 1..3 of the 4-wide guide; source tile is
        // the full 4x4 (source_left = source_lower = 0, tile_width = 4) so
        // the ab cursors are offset from the target cursors.
        let (guide, ab, mut out) = apply_fixture();
        guided_filter_apply(
            &guide, 3, &ab, &mut out, 4, 1, 3, 1, 3, 0, 0, 4, 1.0, -1.0e30, 1.0e30,
        );
        // only the 2x2 interior is written; ab pixel for (row j, col i) is
        // (j - 0)*4 + (i - 0)
        let mut expect = vec![0.0f32; 16];
        for j in 1..3 {
            for i in 1..3 {
                let p = j * 4 + i;
                expect[p] = if p == 5 { 3.0 } else { 12.0 };
            }
        }
        assert_eq!(out, expect);
    }

    #[test]
    fn apply_clamp_semantics() {
        // GLib CLAMP double ternary: res > max → max; res < min → min;
        // NaN fails both comparisons and passes through.
        let mut guide = Vec::new();
        for _ in 0..4 {
            guide.extend_from_slice(&[0.25f32, 0.0, 0.0]);
        }
        let ab = vec![
            1.0, 0.0, 0.0, -0.5, // res = 0.25 - 0.5 = -0.25 → min
            1.0, 0.0, 0.0, 2.0,  // res = 0.25 + 2.0 = 2.25 → max
            1.0, 0.0, 0.0, 0.5,  // res = 0.75 → untouched (inside)
            f32::NAN, 0.0, 0.0, 0.0, // res = NaN → NaN passes through
        ];
        let mut out = vec![0.0f32; 4];
        guided_filter_apply(&guide, 3, &ab, &mut out, 2, 0, 2, 0, 2, 0, 0, 2, 1.0, 0.0, 1.0);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 0.75);
        assert!(out[3].is_nan());
    }

    #[test]
    fn apply_guide_weight_scales_rgb_sum_only() {
        // weight multiplies the a·g sum, then b is added (not scaled).
        let guide = vec![2.0f32, 3.0, 4.0];
        let ab = vec![1.0f32, 1.0, 1.0, 100.0];
        let mut out = vec![0.0f32];
        guided_filter_apply(&guide, 3, &ab, &mut out, 1, 0, 1, 0, 1, 0, 0, 1, 0.5, -1e30, 1e30);
        assert_eq!(out[0], 0.5 * (2.0 + 3.0 + 4.0) + 100.0);
    }

    #[test]
    fn apply_matches_reference_over_lcg() {
        let w = 17usize;
        let h = 13usize;
        let mut guide = vec![0.0f32; w * h * 4];
        let mut ab = vec![0.0f32; w * h * 4];
        lcg_fill(&mut guide, 0x6F1B, 2.0);
        lcg_fill(&mut ab, 0x6F1C, 2.0);
        let mut direct = vec![0.0f32; w * h];
        let mut reference = vec![0.0f32; w * h];

        guided_filter_apply(
            &guide, 4, &ab, &mut direct, w, 2, w - 1, 3, h - 2, 0, 1, w, 1.7, -50.0, 50.0,
        );
        ref_guided_filter_apply(
            &guide, 4, &ab, &mut reference, w, 2, w - 1, 3, h - 2, 0, 1, w, 1.7, -50.0, 50.0,
        );
        assert_eq!(direct, reference);
    }

    // ── FFI round-trip and guard tests ──────────────────────────────────────────

    #[test]
    fn ffi_solve_round_trip() {
        let mut buf = vec![0.0f32; 256 * 4];
        let mut variance = vec![0.0f32; 256 * 9];
        lcg_fill(&mut buf, 0x6F1D, 0.1);
        lcg_fill(&mut variance, 0x6F1E, 0.1);
        let mut ffi_buf = buf.clone();
        let mut direct_buf = buf.clone();

        unsafe {
            darkroom_guided_filter_solve(ffi_buf.as_mut_ptr(), variance.as_ptr(), 256, 0.01);
        }
        guided_filter_solve(&mut direct_buf, &variance, 256, 0.01);

        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_solve_guards() {
        unsafe {
            darkroom_guided_filter_solve(std::ptr::null_mut(), std::ptr::null(), 10, 1.0);
        }
        let variance = vec![0.0f32; 9];
        let mut buf = vec![1.0f32; 4];
        unsafe {
            darkroom_guided_filter_solve(buf.as_mut_ptr(), variance.as_ptr(), 0, 1.0);
            darkroom_guided_filter_solve(
                buf.as_mut_ptr(),
                variance.as_ptr(),
                (i32::MAX as usize) + 1,
                1.0,
            );
        }
        assert_eq!(buf, vec![1.0f32; 4]); // untouched
    }

    #[test]
    fn ffi_apply_round_trip() {
        let w = 11usize;
        let h = 9usize;
        let mut guide = vec![0.0f32; w * h * 3];
        let mut ab = vec![0.0f32; w * h * 4];
        lcg_fill(&mut guide, 0x6F1F, 2.0);
        lcg_fill(&mut ab, 0x6F20, 2.0);
        let mut ffi_out = vec![0.0f32; w * h];
        let mut direct_out = vec![0.0f32; w * h];

        unsafe {
            darkroom_guided_filter_apply(
                guide.as_ptr(),
                3,
                ab.as_ptr(),
                ffi_out.as_mut_ptr(),
                w,
                1,
                w - 1,
                2,
                h - 1,
                0,
                1,
                w,
                1.3,
                -10.0,
                10.0,
            );
        }
        guided_filter_apply(
            &guide, 3, &ab, &mut direct_out, w, 1, w - 1, 2, h - 1, 0, 1, w, 1.3, -10.0, 10.0,
        );
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn ffi_apply_guards() {
        let guide = vec![1.0f32; 16 * 3];
        let ab = vec![1.0f32; 16 * 4];
        let mut out = vec![1.0f32; 16];
        unsafe {
            darkroom_guided_filter_apply(
                std::ptr::null(),
                3,
                std::ptr::null(),
                std::ptr::null_mut(),
                4,
                0,
                4,
                0,
                4,
                0,
                0,
                4,
                1.0,
                0.0,
                1.0,
            );
            // degenerate rects
            darkroom_guided_filter_apply(
                guide.as_ptr(),
                3,
                ab.as_ptr(),
                out.as_mut_ptr(),
                4,
                2,
                2, // right == left
                0,
                4,
                0,
                0,
                4,
                1.0,
                0.0,
                1.0,
            );
            darkroom_guided_filter_apply(
                guide.as_ptr(),
                3,
                ab.as_ptr(),
                out.as_mut_ptr(),
                4,
                0,
                4,
                3,
                3, // upper == lower
                0,
                0,
                4,
                1.0,
                0.0,
                1.0,
            );
            // zero widths
            darkroom_guided_filter_apply(
                guide.as_ptr(),
                3,
                ab.as_ptr(),
                out.as_mut_ptr(),
                0,
                0,
                4,
                0,
                4,
                0,
                0,
                4,
                1.0,
                0.0,
                1.0,
            );
            darkroom_guided_filter_apply(
                guide.as_ptr(),
                3,
                ab.as_ptr(),
                out.as_mut_ptr(),
                4,
                0,
                4,
                0,
                4,
                0,
                0,
                0, // tile_width == 0
                1.0,
                0.0,
                1.0,
            );
            // guide_stride below the documented minimum (would panic on
            // the pixel[1]/[2] reads without this guard)
            darkroom_guided_filter_apply(
                guide.as_ptr(),
                2,
                ab.as_ptr(),
                out.as_mut_ptr(),
                4,
                0,
                4,
                0,
                4,
                0,
                0,
                4,
                1.0,
                0.0,
                1.0,
            );
            // dim cap
            darkroom_guided_filter_apply(
                guide.as_ptr(),
                3,
                ab.as_ptr(),
                out.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                0,
                4,
                0,
                4,
                0,
                0,
                4,
                1.0,
                0.0,
                1.0,
            );
        }
        assert_eq!(out, vec![1.0f32; 16]); // untouched
    }
}
