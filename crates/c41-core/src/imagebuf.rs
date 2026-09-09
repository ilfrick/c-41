//! Element-wise image buffer operations — port of the `DT_OMP_FOR_SIMD` loops
//! in `src/common/imagebuf.c`.
//!
//! Each function is a flat element-wise loop over a `width*height*ch` float
//! buffer. The original C code used `#ifdef _OPENMP` blocks with
//! `DT_OMP_FOR_SIMD(num_threads(n))` (parallel+SIMD) and `DT_OMP_SIMD`
//! (sequential SIMD fallback). The Rust kernels are single-threaded sequential
//! loops; LLVM's auto-vectorizer provides SIMD at `-O3`, but multi-threaded
//! parallelism is no longer used. This matches the m4-161 `blend.rs` pattern.
//!
//! Bit-exactness notes:
//! - The six arithmetic-only kernels (`scaled_copy`, `add_const`, `add_image`,
//!   `sub_image`, `invert`, `mul_const`) are single FP operations with no
//!   function calls. Under `-ffast-math` GCC evaluates them left-to-right
//!   with the same rounding as Rust's strict IEEE evaluation.
//! - `linear_blend` (`lambda*buf[k] + lambda_1*other[k]`) is a multiply-add
//!   pattern susceptible to FMA contraction under GCC `-ffast-math` (`-O3`
//!   enables `-ffp-contract=fast` for C99+). The Rust kernel uses two separate
//!   operations (multiply then add), matching the C sequential fallback but
//!   potentially differing from the C parallel path by ≤1 ULP on some pixels.
//! - `fmaxf`/`fminf` → `f32::max`/`f32::min` (not needed in these loops — no
//!   min/max calls in the ported imagebuf functions).
//! - Expression order preserved verbatim (e.g. `lambda*buf[k] + lambda_1*other[k]`
//!   evaluates as `((lambda*buf[k]) + (lambda_1*other[k]))`).

/// `buf[k] = scale * src[k]` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_iop_image_scaled_copy`
/// (imagebuf.c:257).
pub fn scaled_copy(buf: &mut [f32], src: &[f32], n: usize, scale: f32) {
    let m = n.min(buf.len()).min(src.len());
    for k in 0..m {
        buf[k] = scale * src[k];
    }
}

/// `buf[k] += value` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_iop_image_add_const`
/// (imagebuf.c:327).
pub fn add_const(buf: &mut [f32], n: usize, value: f32) {
    let m = n.min(buf.len());
    for k in 0..m {
        buf[k] += value;
    }
}

/// `buf[k] += other[k]` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_iop_image_add_image`
/// (imagebuf.c:355).
pub fn add_image(buf: &mut [f32], other: &[f32], n: usize) {
    let m = n.min(buf.len()).min(other.len());
    for k in 0..m {
        buf[k] += other[k];
    }
}

/// `buf[k] -= other[k]` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_iop_image_sub_image`
/// (imagebuf.c:383).
pub fn sub_image(buf: &mut [f32], other: &[f32], n: usize) {
    let m = n.min(buf.len()).min(other.len());
    for k in 0..m {
        buf[k] -= other[k];
    }
}

/// `buf[k] = max_value - buf[k]` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_iop_image_invert`
/// (imagebuf.c:411).
pub fn invert(buf: &mut [f32], n: usize, max_value: f32) {
    let m = n.min(buf.len());
    for k in 0..m {
        buf[k] = max_value - buf[k];
    }
}

/// `buf[k] *= value` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_iop_image_mul_const`
/// (imagebuf.c:439).
pub fn mul_const(buf: &mut [f32], n: usize, value: f32) {
    let m = n.min(buf.len());
    for k in 0..m {
        buf[k] *= value;
    }
}

/// `buf[k] = lambda*buf[k] + (1-lambda)*other[k]` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_iop_image_linear_blend`
/// (imagebuf.c:470). The C pre-computes `lambda_1 = 1.0f - lambda` once;
/// the Rust mirrors that to preserve expression order.
pub fn linear_blend(buf: &mut [f32], other: &[f32], n: usize, lambda: f32) {
    let lambda_1 = 1.0f32 - lambda;
    let m = n.min(buf.len()).min(other.len());
    for k in 0..m {
        buf[k] = lambda * buf[k] + lambda_1 * other[k];
    }
}

/// `buf[k] = value` for each `k` in `0..n`.
///
/// Port of `dt_iop_image_fill` (imagebuf.c:253). The C has two paths: a
/// data-parallel chunked fill for large buffers and a sequential fallback
/// (`memset` when `value == 0.0f`, plain loop otherwise). All paths write
/// the same value to every element, so the sequential loop matches both.
/// The `value == 0.0` branch writes canonical `+0.0`, matching the C
/// `memset(0)` (which also normalizes `-0.0` to `+0.0`).
pub fn fill(buf: &mut [f32], n: usize, value: f32) {
    let m = n.min(buf.len());
    if value == 0.0 {
        for k in 0..m {
            buf[k] = 0.0;
        }
    } else {
        for k in 0..m {
            buf[k] = value;
        }
    }
}

/// `out[k] = in[k]` for each `k` in `0..n`.
///
/// Port of the `DT_OMP_FOR_SIMD` loop in `dt_simd_memcpy`
/// (imagebuf.h:70). Simple element-wise copy.
pub fn simd_memcpy(buf: &mut [f32], src: &[f32], n: usize) {
    let m = n.min(buf.len()).min(src.len());
    for k in 0..m {
        buf[k] = src[k];
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `buf` and `src` must each hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_scaled_copy(
    buf: *mut f32,
    src: *const f32,
    n: usize,
    scale: f32,
) {
    if buf.is_null() || src.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    let src_slice = std::slice::from_raw_parts(src, n);
    scaled_copy(buf_slice, src_slice, n, scale);
}

/// # Safety
/// `buf` must hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_add_const(
    buf: *mut f32,
    n: usize,
    value: f32,
) {
    if buf.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    add_const(buf_slice, n, value);
}

/// # Safety
/// `buf` and `other` must each hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_add_image(
    buf: *mut f32,
    other: *const f32,
    n: usize,
) {
    if buf.is_null() || other.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    let other_slice = std::slice::from_raw_parts(other, n);
    add_image(buf_slice, other_slice, n);
}

/// # Safety
/// `buf` and `other` must each hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_sub_image(
    buf: *mut f32,
    other: *const f32,
    n: usize,
) {
    if buf.is_null() || other.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    let other_slice = std::slice::from_raw_parts(other, n);
    sub_image(buf_slice, other_slice, n);
}

/// # Safety
/// `buf` must hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_invert(
    buf: *mut f32,
    n: usize,
    max_value: f32,
) {
    if buf.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    invert(buf_slice, n, max_value);
}

/// # Safety
/// `buf` must hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_mul_const(
    buf: *mut f32,
    n: usize,
    value: f32,
) {
    if buf.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    mul_const(buf_slice, n, value);
}

/// # Safety
/// `buf` must hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_fill(
    buf: *mut f32,
    n: usize,
    value: f32,
) {
    if buf.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    fill(buf_slice, n, value);
}

/// # Safety
/// `buf` and `other` must each hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_linear_blend(
    buf: *mut f32,
    other: *const f32,
    n: usize,
    lambda: f32,
) {
    if buf.is_null() || other.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    let other_slice = std::slice::from_raw_parts(other, n);
    linear_blend(buf_slice, other_slice, n, lambda);
}

/// # Safety
/// `buf` and `src` must each hold at least `n` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_simd_memcpy(
    buf: *mut f32,
    src: *const f32,
    n: usize,
) {
    if buf.is_null() || src.is_null() || n == 0 || n > i32::MAX as usize {
        return;
    }
    let buf_slice = std::slice::from_raw_parts_mut(buf, n);
    let src_slice = std::slice::from_raw_parts(src, n);
    simd_memcpy(buf_slice, src_slice, n);
}

// ── Reference implementations for bit-exactness tests ────────────────────────

#[allow(dead_code)]
fn ref_scaled_copy(buf: &mut [f32], src: &[f32], n: usize, scale: f32) {
    let m = n.min(buf.len()).min(src.len());
    for k in 0..m {
        buf[k] = scale * src[k];
    }
}

#[allow(dead_code)]
fn ref_add_const(buf: &mut [f32], n: usize, value: f32) {
    let m = n.min(buf.len());
    for k in 0..m {
        buf[k] += value;
    }
}

#[allow(dead_code)]
fn ref_add_image(buf: &mut [f32], other: &[f32], n: usize) {
    let m = n.min(buf.len()).min(other.len());
    for k in 0..m {
        buf[k] += other[k];
    }
}

#[allow(dead_code)]
fn ref_sub_image(buf: &mut [f32], other: &[f32], n: usize) {
    let m = n.min(buf.len()).min(other.len());
    for k in 0..m {
        buf[k] -= other[k];
    }
}

#[allow(dead_code)]
fn ref_invert(buf: &mut [f32], n: usize, max_value: f32) {
    let m = n.min(buf.len());
    for k in 0..m {
        buf[k] = max_value - buf[k];
    }
}

#[allow(dead_code)]
fn ref_mul_const(buf: &mut [f32], n: usize, value: f32) {
    let m = n.min(buf.len());
    for k in 0..m {
        buf[k] *= value;
    }
}

#[allow(dead_code)]
fn ref_linear_blend(buf: &mut [f32], other: &[f32], n: usize, lambda: f32) {
    let lambda_1 = 1.0f32 - lambda;
    let m = n.min(buf.len()).min(other.len());
    for k in 0..m {
        buf[k] = lambda * buf[k] + lambda_1 * other[k];
    }
}

#[allow(dead_code)]
fn ref_fill(buf: &mut [f32], n: usize, value: f32) {
    // Deliberately different: while-loop with canonical +0.0 on the zero
    // path (mirrors memset), versus the for-loop kernel. Same result.
    let m = n.min(buf.len());
    let v = if value == 0.0 { 0.0 } else { value };
    let mut k = 0;
    while k < m {
        buf[k] = v;
        k += 1;
    }
}

#[allow(dead_code)]
fn ref_simd_memcpy(buf: &mut [f32], src: &[f32], n: usize) {
    // Deliberately different: iterate using index arithmetic rather than
    // letting the for-loop iterator handle it. Same result, different shape.
    let m = n.min(buf.len()).min(src.len());
    let mut k = 0;
    while k < m {
        buf[k] = src[k];
        k += 1;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    // ── scaled_copy ────────────────────────────────────────────────────────────

    #[test]
    fn scaled_copy_basic() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut buf = vec![0.0f32; 4];
        scaled_copy(&mut buf, &src, 4, 2.0);
        assert_eq!(buf, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn scaled_copy_negative_scale() {
        let src = vec![1.0f32, -2.0, 3.0];
        let mut buf = vec![0.0f32; 3];
        scaled_copy(&mut buf, &src, 3, -1.0);
        assert_eq!(buf, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn scaled_copy_matches_reference_over_lcg() {
        let mut src = vec![0.0f32; 256];
        lcg_fill(&mut src, 0xABCD, 1.0);

        let mut direct = vec![0.0f32; 256];
        let mut reference = vec![0.0f32; 256];
        scaled_copy(&mut direct, &src, 256, 0.5);
        ref_scaled_copy(&mut reference, &src, 256, 0.5);
        assert_eq!(direct, reference);
    }

    // ── add_const ─────────────────────────────────────────────────────────────

    #[test]
    fn add_const_basic() {
        let mut buf = vec![0.1f32, 0.2, 0.3, 0.4];
        add_const(&mut buf, 4, 0.5);
        assert_eq!(buf, vec![0.6, 0.7, 0.8, 0.9]);
    }

    #[test]
    fn add_const_negative() {
        let mut buf = vec![0.5f32, 1.0, 0.0];
        add_const(&mut buf, 3, -0.3);
        assert!((buf[0] - 0.2f32).abs() < 1e-6);
        assert!((buf[1] - 0.7f32).abs() < 1e-6);
        assert!((buf[2] - (-0.3f32)).abs() < 1e-6);
    }

    #[test]
    fn add_const_matches_reference_over_lcg() {
        let mut buf = vec![0.0f32; 256];
        lcg_fill(&mut buf, 0xBEEF, 1.0);

        let mut direct = buf.clone();
        let mut reference = buf.clone();
        add_const(&mut direct, 256, 0.25);
        ref_add_const(&mut reference, 256, 0.25);
        assert_eq!(direct, reference);
    }

    // ── add_image ─────────────────────────────────────────────────────────────

    #[test]
    fn add_image_basic() {
        let mut buf = vec![0.1f32, 0.2, 0.3];
        let other = vec![0.9f32, 0.8, 0.7];
        add_image(&mut buf, &other, 3);
        assert_eq!(buf, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn add_image_matches_reference_over_lcg() {
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        lcg_fill(&mut a, 0x1111, 1.0);
        lcg_fill(&mut b, 0x2222, 1.0);

        let mut direct = a.clone();
        let mut reference = a.clone();
        add_image(&mut direct, &b, 256);
        ref_add_image(&mut reference, &b, 256);
        assert_eq!(direct, reference);
    }

    // ── sub_image ─────────────────────────────────────────────────────────────

    #[test]
    fn sub_image_basic() {
        let mut buf = vec![1.0f32, 0.8, 0.6];
        let other = vec![0.1f32, 0.2, 0.3];
        sub_image(&mut buf, &other, 3);
        assert_eq!(buf, vec![0.9, 0.6, 0.3]);
    }

    #[test]
    fn sub_image_matches_reference_over_lcg() {
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        lcg_fill(&mut a, 0x3333, 1.0);
        lcg_fill(&mut b, 0x4444, 1.0);

        let mut direct = a.clone();
        let mut reference = a.clone();
        sub_image(&mut direct, &b, 256);
        ref_sub_image(&mut reference, &b, 256);
        assert_eq!(direct, reference);
    }

    // ── invert ─────────────────────────────────────────────────────────────────

    #[test]
    fn invert_basic() {
        let mut buf = vec![0.0f32, 0.5, 1.0];
        invert(&mut buf, 3, 1.0);
        assert_eq!(buf, vec![1.0, 0.5, 0.0]);
    }

    #[test]
    fn invert_max_value_255() {
        let mut buf = vec![0.0f32, 128.0, 255.0];
        invert(&mut buf, 3, 255.0);
        assert_eq!(buf, vec![255.0, 127.0, 0.0]);
    }

    #[test]
    fn invert_matches_reference_over_lcg() {
        let mut buf = vec![0.0f32; 256];
        lcg_fill(&mut buf, 0x5555, 1.0);

        let mut direct = buf.clone();
        let mut reference = buf.clone();
        invert(&mut direct, 256, 1.0);
        ref_invert(&mut reference, 256, 1.0);
        assert_eq!(direct, reference);
    }

    // ── mul_const ──────────────────────────────────────────────────────────────

    #[test]
    fn mul_const_basic() {
        let mut buf = vec![0.5f32, 1.0, 2.0];
        mul_const(&mut buf, 3, 2.0);
        assert_eq!(buf, vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn mul_const_zero() {
        let mut buf = vec![1.0f32, 2.0, 3.0];
        mul_const(&mut buf, 3, 0.0);
        assert_eq!(buf, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn mul_const_matches_reference_over_lcg() {
        let mut buf = vec![0.0f32; 256];
        lcg_fill(&mut buf, 0x6666, 1.0);

        let mut direct = buf.clone();
        let mut reference = buf.clone();
        mul_const(&mut direct, 256, 0.75);
        ref_mul_const(&mut reference, 256, 0.75);
        assert_eq!(direct, reference);
    }

    // ── fill ───────────────────────────────────────────────────────────────────

    #[test]
    fn fill_zero() {
        let mut buf = vec![1.0f32, 2.0, 3.0, 4.0];
        fill(&mut buf, 4, 0.0);
        assert_eq!(buf, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn fill_negative_zero_normalizes_to_positive_zero() {
        // Matches the C `memset(buf, 0, ...)` taken when `fill_value == 0.0f`.
        let mut buf = vec![1.0f32; 4];
        fill(&mut buf, 4, -0.0);
        assert!(buf.iter().all(|&v| v.to_bits() == 0u32));
    }

    #[test]
    fn fill_reference_zero_path_matches_bit_exact() {
        // Exercise ref_fill's zero branch too: both implementations must take
        // the canonical +0.0 path for either float zero.
        for value in [0.0f32, -0.0] {
            let mut direct = vec![1.0f32; 16];
            let mut reference = vec![1.0f32; 16];
            fill(&mut direct, 16, value);
            ref_fill(&mut reference, 16, value);
            assert_eq!(direct, reference, "value bits: {:x}", value.to_bits());
            assert!(direct.iter().all(|&v| v.to_bits() == 0u32));
        }
    }

    #[test]
    fn fill_nonzero() {
        let mut buf = vec![0.0f32; 4];
        fill(&mut buf, 4, 0.5);
        assert_eq!(buf, vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn fill_non_multiple_of_4_tails() {
        for n in [1usize, 2, 3, 5, 6, 7, 9, 13, 1001] {
            let mut direct = vec![0.0f32; n];
            let mut reference = vec![0.0f32; n];
            fill(&mut direct, n, 1.25);
            ref_fill(&mut reference, n, 1.25);
            assert_eq!(direct, reference, "n={n}");
            assert!(direct.iter().all(|&v| v == 1.25));
        }
    }

    #[test]
    fn fill_nan_inf_bit_exact() {
        let nan_bits = 0x7FC0_1234u32;
        let nan = f32::from_bits(nan_bits);
        let mut buf = vec![0.0f32; 8];
        fill(&mut buf, 8, nan);
        assert!(buf.iter().all(|&v| v.to_bits() == nan_bits));

        let mut buf = vec![0.0f32; 8];
        fill(&mut buf, 8, f32::INFINITY);
        assert!(buf.iter().all(|&v| v == f32::INFINITY));
        let mut buf = vec![0.0f32; 8];
        fill(&mut buf, 8, f32::NEG_INFINITY);
        assert!(buf.iter().all(|&v| v == f32::NEG_INFINITY));
    }

    #[test]
    fn fill_matches_reference_over_lcg() {
        let mut buf = vec![0.0f32; 256];
        lcg_fill(&mut buf, 0xF111, 1.0);

        let mut direct = buf.clone();
        let mut reference = buf.clone();
        fill(&mut direct, 256, -2.5);
        ref_fill(&mut reference, 256, -2.5);
        assert_eq!(direct, reference);
    }

    #[test]
    fn fill_short_inputs_clamp() {
        // n larger than the buffer: kernel clamps to buf.len().
        let mut buf = vec![0.0f32; 3];
        fill(&mut buf, 100, 7.0);
        assert_eq!(buf, vec![7.0, 7.0, 7.0]);

        // n smaller than the buffer: tail untouched.
        let mut buf = vec![0.0f32; 8];
        fill(&mut buf, 3, 7.0);
        assert_eq!(&buf[..3], &[7.0, 7.0, 7.0]);
        assert_eq!(&buf[3..], &[0.0; 5]);
    }

    #[test]
    fn fill_degenerate_empty() {
        let mut buf: Vec<f32> = vec![];
        fill(&mut buf, 0, 1.0); // no-op, no panic
        assert!(buf.is_empty());
        let mut buf = vec![1.0f32; 4];
        fill(&mut buf, 0, 2.0); // n == 0: untouched
        assert_eq!(buf, vec![1.0; 4]);
    }

    // ── linear_blend ───────────────────────────────────────────────────────────

    #[test]
    fn linear_blend_basic() {
        let mut buf = vec![1.0f32, 0.0, 0.5];
        let other = vec![0.0f32, 1.0, 0.5];
        // lambda=0.5: buf = 0.5*buf + 0.5*other
        linear_blend(&mut buf, &other, 3, 0.5);
        assert_eq!(buf, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn linear_blend_lambda_0_is_passthrough() {
        let mut buf = vec![0.3f32, 0.7, 0.9];
        let other = vec![0.0f32, 0.0, 0.0];
        linear_blend(&mut buf, &other, 3, 0.0);
        // lambda=0: buf = 0*buf + 1*other = other
        assert_eq!(buf, other);
    }

    #[test]
    fn linear_blend_lambda_1_is_identity() {
        let original = vec![0.3f32, 0.7, 0.9];
        let mut buf = original.clone();
        let other = vec![1.0f32, 1.0, 1.0];
        linear_blend(&mut buf, &other, 3, 1.0);
        // lambda=1: buf = 1*buf + 0*other = buf (unchanged)
        assert_eq!(buf, original);
    }

    #[test]
    fn linear_blend_matches_reference_over_lcg() {
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        lcg_fill(&mut a, 0x7777, 1.0);
        lcg_fill(&mut b, 0x8888, 1.0);

        let mut direct = a.clone();
        let mut reference = a.clone();
        linear_blend(&mut direct, &b, 256, 0.3);
        ref_linear_blend(&mut reference, &b, 256, 0.3);
        assert_eq!(direct, reference);
    }

    // ── simd_memcpy ─────────────────────────────────────────────────────────────

    #[test]
    fn simd_memcpy_basic() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut buf = vec![0.0f32; 4];
        simd_memcpy(&mut buf, &src, 4);
        assert_eq!(buf, src);
    }

    #[test]
    fn simd_memcpy_matches_reference_over_lcg() {
        let mut src = vec![0.0f32; 256];
        lcg_fill(&mut src, 0xB0B0, 1.0);

        let mut direct = vec![0.0f32; 256];
        let mut reference = vec![0.0f32; 256];
        simd_memcpy(&mut direct, &src, 256);
        ref_simd_memcpy(&mut reference, &src, 256);
        assert_eq!(direct, reference);
    }

    // ── FFI round-trip and null-guard tests ────────────────────────────────────

    #[test]
    fn ffi_scaled_copy_round_trip() {
        let mut src = vec![0.0f32; 64];
        lcg_fill(&mut src, 0x9999, 1.0);

        let mut ffi_buf = vec![0.0f32; 64];
        let mut direct_buf = vec![0.0f32; 64];

        unsafe {
            darkroom_imagebuf_scaled_copy(ffi_buf.as_mut_ptr(), src.as_ptr(), 64, 2.0);
        }
        scaled_copy(&mut direct_buf, &src, 64, 2.0);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_scaled_copy_null_guard() {
        unsafe {
            darkroom_imagebuf_scaled_copy(std::ptr::null_mut(), std::ptr::null(), 10, 1.0);
        }
    }

    #[test]
    fn ffi_add_const_round_trip() {
        let mut src = vec![0.0f32; 64];
        lcg_fill(&mut src, 0xAAAA, 1.0);

        let mut ffi_buf = src.clone();
        let mut direct_buf = src.clone();

        unsafe {
            darkroom_imagebuf_add_const(ffi_buf.as_mut_ptr(), 64, 0.5);
        }
        add_const(&mut direct_buf, 64, 0.5);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_add_const_null_guard() {
        unsafe {
            darkroom_imagebuf_add_const(std::ptr::null_mut(), 10, 0.5);
        }
    }

    #[test]
    fn ffi_add_image_round_trip() {
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        lcg_fill(&mut a, 0xBBBB, 1.0);
        lcg_fill(&mut b, 0xCCCC, 1.0);

        let mut ffi_buf = a.clone();
        let mut direct_buf = a.clone();

        unsafe {
            darkroom_imagebuf_add_image(ffi_buf.as_mut_ptr(), b.as_ptr(), 64);
        }
        add_image(&mut direct_buf, &b, 64);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_add_image_null_guard() {
        unsafe {
            darkroom_imagebuf_add_image(std::ptr::null_mut(), std::ptr::null(), 10);
        }
    }

    #[test]
    fn ffi_sub_image_round_trip() {
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        lcg_fill(&mut a, 0xDDDD, 1.0);
        lcg_fill(&mut b, 0xEEEE, 1.0);

        let mut ffi_buf = a.clone();
        let mut direct_buf = a.clone();

        unsafe {
            darkroom_imagebuf_sub_image(ffi_buf.as_mut_ptr(), b.as_ptr(), 64);
        }
        sub_image(&mut direct_buf, &b, 64);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_sub_image_null_guard() {
        unsafe {
            darkroom_imagebuf_sub_image(std::ptr::null_mut(), std::ptr::null(), 10);
        }
    }

    #[test]
    fn ffi_invert_round_trip() {
        let mut src = vec![0.0f32; 64];
        lcg_fill(&mut src, 0xFFFF, 1.0);

        let mut ffi_buf = src.clone();
        let mut direct_buf = src.clone();

        unsafe {
            darkroom_imagebuf_invert(ffi_buf.as_mut_ptr(), 64, 1.0);
        }
        invert(&mut direct_buf, 64, 1.0);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_invert_null_guard() {
        unsafe {
            darkroom_imagebuf_invert(std::ptr::null_mut(), 10, 1.0);
        }
    }

    #[test]
    fn ffi_mul_const_round_trip() {
        let mut src = vec![0.0f32; 64];
        lcg_fill(&mut src, 0x1234, 1.0);

        let mut ffi_buf = src.clone();
        let mut direct_buf = src.clone();

        unsafe {
            darkroom_imagebuf_mul_const(ffi_buf.as_mut_ptr(), 64, 3.0);
        }
        mul_const(&mut direct_buf, 64, 3.0);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_mul_const_null_guard() {
        unsafe {
            darkroom_imagebuf_mul_const(std::ptr::null_mut(), 10, 2.0);
        }
    }

    #[test]
    fn ffi_fill_round_trip() {
        let mut src = vec![0.0f32; 64];
        lcg_fill(&mut src, 0xF00D, 1.0);

        let mut ffi_buf = src.clone();
        let mut direct_buf = src.clone();

        unsafe {
            darkroom_imagebuf_fill(ffi_buf.as_mut_ptr(), 64, 0.5);
        }
        fill(&mut direct_buf, 64, 0.5);
        assert_eq!(ffi_buf, direct_buf);
        // Pin the memset-equivalent zero path through FFI as well.
        for value in [0.0f32, -0.0] {
            let mut ffi_zero = vec![1.0f32; 64];
            let mut direct_zero = vec![1.0f32; 64];
            unsafe {
                darkroom_imagebuf_fill(ffi_zero.as_mut_ptr(), 64, value);
            }
            fill(&mut direct_zero, 64, value);
            assert_eq!(ffi_zero, direct_zero, "value bits: {:x}", value.to_bits());
            assert!(ffi_zero.iter().all(|&v| v.to_bits() == 0u32));
        }
    }

    #[test]
    fn ffi_fill_null_guard() {
        unsafe {
            darkroom_imagebuf_fill(std::ptr::null_mut(), 10, 1.0);
        }
    }

    #[test]
    fn ffi_fill_zero_n_guard() {
        let mut buf = vec![1.0f32; 4];
        unsafe {
            darkroom_imagebuf_fill(buf.as_mut_ptr(), 0, 2.0);
        }
        assert_eq!(buf, vec![1.0; 4]); // untouched
    }

    #[test]
    fn ffi_fill_overflow_guard() {
        let mut buf = vec![1.0f32; 4];
        let big_n = (i32::MAX as usize) + 1;
        unsafe {
            darkroom_imagebuf_fill(buf.as_mut_ptr(), big_n, 2.0);
        }
        assert_eq!(buf, vec![1.0; 4]); // untouched
    }

    #[test]
    fn ffi_linear_blend_round_trip() {
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        lcg_fill(&mut a, 0x5678, 1.0);
        lcg_fill(&mut b, 0x9ABC, 1.0);

        let mut ffi_buf = a.clone();
        let mut direct_buf = a.clone();

        unsafe {
            darkroom_imagebuf_linear_blend(ffi_buf.as_mut_ptr(), b.as_ptr(), 64, 0.3);
        }
        linear_blend(&mut direct_buf, &b, 64, 0.3);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_linear_blend_null_guard() {
        unsafe {
            darkroom_imagebuf_linear_blend(std::ptr::null_mut(), std::ptr::null(), 10, 0.5);
        }
    }

    #[test]
    fn ffi_zero_n_guard() {
        let mut buf = vec![1.0f32; 4];
        let other = vec![0.5f32; 4];
        unsafe {
            darkroom_imagebuf_add_const(buf.as_mut_ptr(), 0, 0.5);
            darkroom_imagebuf_add_image(buf.as_mut_ptr(), other.as_ptr(), 0);
            darkroom_imagebuf_sub_image(buf.as_mut_ptr(), other.as_ptr(), 0);
            darkroom_imagebuf_invert(buf.as_mut_ptr(), 0, 1.0);
            darkroom_imagebuf_mul_const(buf.as_mut_ptr(), 0, 2.0);
            darkroom_imagebuf_linear_blend(buf.as_mut_ptr(), other.as_ptr(), 0, 0.5);
            darkroom_imagebuf_scaled_copy(buf.as_mut_ptr(), other.as_ptr(), 0, 2.0);
        }
        assert_eq!(buf, vec![1.0; 4]); // untouched
    }

    #[test]
    fn ffi_simd_memcpy_round_trip() {
        let mut src = vec![0.0f32; 64];
        lcg_fill(&mut src, 0xDADA, 1.0);

        let mut ffi_buf = vec![0.0f32; 64];
        let mut direct_buf = vec![0.0f32; 64];

        unsafe {
            darkroom_imagebuf_simd_memcpy(ffi_buf.as_mut_ptr(), src.as_ptr(), 64);
        }
        simd_memcpy(&mut direct_buf, &src, 64);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_simd_memcpy_null_guard() {
        unsafe {
            darkroom_imagebuf_simd_memcpy(std::ptr::null_mut(), std::ptr::null(), 10);
        }
    }

    #[test]
    fn ffi_simd_memcpy_zero_n_guard() {
        let mut buf = vec![1.0f32; 4];
        let src = vec![0.5f32; 4];
        unsafe {
            darkroom_imagebuf_simd_memcpy(buf.as_mut_ptr(), src.as_ptr(), 0);
        }
        assert_eq!(buf, vec![1.0; 4]); // untouched
    }

    #[test]
    fn ffi_simd_memcpy_overflow_guard() {
        let mut buf = vec![1.0f32; 4];
        let src = vec![0.5f32; 4];
        let big_n = (i32::MAX as usize) + 1;
        unsafe {
            darkroom_imagebuf_simd_memcpy(buf.as_mut_ptr(), src.as_ptr(), big_n);
        }
        assert_eq!(buf, vec![1.0; 4]); // untouched
    }
}
