//! Element-wise loops ported from `src/common/fast_guided_filter.h`.
//!
//! Two `DT_OMP_FOR_SIMD` loops are ported here:
//! - The variance-analyse pack loop (fast_guided_filter.h:178) that packs
//!   `{guide, mask, guide², guide·mask}` into a 4-channel buffer before the
//!   box-mean blur.
//! - The apply-linear-blending loop (fast_guided_filter.h:211) that applies
//!   the `a*image + b` blend and clamps to `MIN_FLOAT`.
//!
//! Both original loops used `DT_OMP_FOR_SIMD` (parallel+SIMD). The Rust kernels
//! are single-threaded sequential; LLVM's auto-vectorizer provides SIMD at `-O3`,
//! but multi-threaded parallelism is no longer used. This matches the m4-161
//! `blend.rs` and m4-162 `imagebuf.rs` pattern.
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
//!   ends up as the second operand). The ported loop was `DT_OMP_FOR_SIMD`,
//!   i.e. vectorized, so in pragma TUs the old C path returned NaN on NaN
//!   input where the Rust kernel returns `MIN_FLOAT`. NaN reaching
//!   `image[k]*ab[k*2] + ab[k*2+1]` is rare (requires upstream inf-inf or
//!   sqrt-of-negative in the luminance-mask pipeline), and the
//!   still-unported `apply_linear_blending_w_geomean` (still C, still uses
//!   `fmaxf`) carries this same inconsistency live.
//! - Note: the reference implementation `ref_apply_linear_blending`
//!   deliberately diverges using `if blended < MIN_FLOAT { MIN_FLOAT } else { blended }`,
//!   which is NaN-propagating (NaN < MIN_FLOAT is false, so NaN is returned).
//!   This structural divergence is intentional for independent validation.
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
/// (fast_guided_filter.h:178). For each element:
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
/// (fast_guided_filter.h:211). `ab[k*2]` is the blend coefficient `a`,
/// `ab[k*2+1]` is the offset `b`.
pub fn apply_linear_blending(image: &mut [f32], ab: &[f32], n_elements: usize) {
    let m = n_elements.min(image.len()).min(ab.len() / 2);
    for k in 0..m {
        let val = image[k] * ab[k * 2] + ab[k * 2 + 1];
        image[k] = val.max(MIN_FLOAT);
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
}
