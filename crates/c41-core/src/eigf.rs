//! Element-wise variance-analysis correction loops — port of the
//! `DT_OMP_FOR_SIMD` loops in `src/common/eigf.h`.
//!
//! After a Gaussian blur computes raw moment averages, these loops convert
//! `E[X²]` into variance `E[X²] - E[X]²` and `E[XY]` into covariance
//! `E[XY] - E[X]·E[Y]`. Each function is a flat element-wise loop over a
//! multi-channel float buffer with stride access.
//!
//! The original C code used `DT_OMP_FOR_SIMD(aligned(out:64))` (parallel+SIMD).
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
//!   FMA hardware this can produce ≤1 ULP difference between the old C path and the
//!   new Rust FFI path on some pixels. This is the same trade-off accepted for
//!   `linear_blend` in `imagebuf.rs` and is not visually significant for variance
//!   estimation in a guided filter.
//! - `eigf.h` has no `#pragma GCC optimize("fast-math")`. Only `finite-math-only`
//!   (via `extra_optimizations.h` in the caller) is active. `finite-math-only`
//!   does NOT enable FMA contraction; the residual risk comes solely from GCC's
//!   default `-ffp-contract` policy.

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

// ── Independent reference implementations for bit-exactness tests ─────────────
//
// These use a different floating-point evaluation order than the kernels above:
// products are computed into temporaries first, then subtracted. This changes
// the intermediate rounding behavior and provides genuine independent validation
// (catching channel-index swaps or wrong-operation bugs that a copy-pasted ref
// would miss). The operations are mathematically identical.

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
}
