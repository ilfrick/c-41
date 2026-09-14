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
//! `copy_alpha` is the one exception to the imagebuf.c source: it ports the
//! strided alpha-lane loop of `dt_iop_alpha_copy` in
//! `src/develop/imageop_math.h`, not a flat element-wise loop.
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

/// `out[k] = src[k]` for each alpha lane `k` in `3, 7, 11, ... < width*height*4`.
///
/// Port of the strided loop in `dt_iop_alpha_copy` (imageop_math.h:141).
/// Unlike `simd_memcpy` this is NOT a bulk copy: only channel 3 (alpha) of
/// each RGBA pixel is copied and the RGB lanes of `out` are left untouched.
/// Each lane is read before its own write, so aliasing (`out` is `src`)
/// is benign. The element count is checked (`width*height*4` overflow or
/// empty dims are a no-op) and clamped to the shorter slice.
pub fn copy_alpha(out: &mut [f32], src: &[f32], width: usize, height: usize) {
    let Some(n) = width.checked_mul(height).and_then(|p| p.checked_mul(4)) else {
        return;
    };
    let m = n.min(out.len()).min(src.len());
    for k in (3..m).step_by(4) {
        out[k] = src[k];
    }
}

/// ROI copy with zero-fill for out-of-range pixels.
///
/// Port of the `DT_OMP_FOR(collapse(2))` fallback loop in
/// `dt_iop_copy_image_roi` (imagebuf.c:223). The C fast paths — whole-buffer
/// copy and per-row `memcpy` — stay in C; only this branch, taken when the
/// ROIs are inconsistent, is ported. For each output pixel `(row, col)`,
/// `(irow, icol) = (row + dy, col + dx)`; when inside the input ROI the `ch`
/// channels are copied, otherwise the pixel is zeroed. This is a
/// bounds-conditional kernel, not a bulk memcpy.
///
/// `dx`/`dy` are signed (`roi_out - roi_in` offsets can be negative), so the
/// index math runs in `i64` and an out-of-range source coordinate takes the
/// zero branch without ever forming an out-of-bounds index. Undersized
/// buffers or a `ch*w*h` overflow are a no-op (the FFI caller guarantees
/// exact lengths, so production always runs full). Like the C `restrict`
/// contract, `out` and `src` must not overlap.
#[allow(clippy::too_many_arguments)]
pub fn copy_roi(
    out: &mut [f32],
    src: &[f32],
    ch: usize,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    dx: i64,
    dy: i64,
) {
    let Some(need_out) = ch.checked_mul(out_w).and_then(|v| v.checked_mul(out_h)) else {
        return;
    };
    let Some(need_in) = ch.checked_mul(in_w).and_then(|v| v.checked_mul(in_h)) else {
        return;
    };
    if need_out == 0 || need_in == 0 || out.len() < need_out || src.len() < need_in {
        return;
    }
    let in_w_i = in_w as i64;
    let in_h_i = in_h as i64;
    for row in 0..out_h {
        for col in 0..out_w {
            let irow = row as i64 + dy;
            let icol = col as i64 + dx;
            let ox = ch * (row * out_w + col);
            if irow >= 0 && irow < in_h_i && icol >= 0 && icol < in_w_i {
                let ix = ch * ((irow as usize) * in_w + (icol as usize));
                out[ox..ox + ch].copy_from_slice(&src[ix..ix + ch]);
            } else {
                for c in 0..ch {
                    out[ox + c] = 0.0;
                }
            }
        }
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

/// Copy the alpha channel 1:1 from `src` to `out`.
///
/// Port of `dt_iop_alpha_copy` (imageop_math.h:141): for each RGBA pixel
/// `p` in `0..width*height`, `out[p*4+3] = src[p*4+3]`; RGB lanes are
/// untouched. This is a strided channel copy, not a bulk memcpy.
/// Aliasing (`src == out`) is safe: every lane is read before its own
/// write. NULL pointers, empty dims, or a `width*height*4` overflow are a
/// no-op; short buffers clamp instead of panicking.
///
/// # Safety
/// `src` and `out` must each hold at least `width*height*4` floats (or be
/// NULL, in which case this is a no-op). A `width*height*4` overflow is also
/// a no-op.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_copy_alpha(
    src: *const f32,
    out: *mut f32,
    width: usize,
    height: usize,
) {
    if src.is_null() || out.is_null() {
        return;
    }
    let Some(n) = width.checked_mul(height).and_then(|v| v.checked_mul(4)) else {
        return;
    };
    if n == 0 {
        return;
    }
    let src_slice = std::slice::from_raw_parts(src, n);
    let out_slice = std::slice::from_raw_parts_mut(out, n);
    copy_alpha(out_slice, src_slice, width, height);
}

/// Copy the inconsistent-ROI fallback of `dt_iop_copy_image_roi`.
///
/// `dx`/`dy` are the C `int` ROI offsets (`roi_out - roi_in`, may be
/// negative); the remaining sizes are element counts. `out_len`/`in_len`
/// are validated against the checked `ch*w*h` products — the slices are
/// built from the recomputed products, never from the caller lengths, so a
/// wrapped C-side product cannot size a slice. NULL pointers, empty dims,
/// arithmetic overflow, oversized (`> i32::MAX`) products, or short buffers
/// are a no-op. `out` and `src` must not overlap (C `restrict` contract).
///
/// # Safety
/// `out` must hold at least `out_len` floats and `src` at least `in_len`
/// floats (or be NULL, in which case this is a no-op).
#[no_mangle]
pub unsafe extern "C" fn darkroom_imagebuf_copy_roi(
    out: *mut f32,
    src: *const f32,
    ch: usize,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    dx: i32,
    dy: i32,
    out_len: usize,
    in_len: usize,
) {
    if out.is_null() || src.is_null() {
        return;
    }
    let Some(need_out) = ch.checked_mul(out_w).and_then(|v| v.checked_mul(out_h)) else {
        return;
    };
    let Some(need_in) = ch.checked_mul(in_w).and_then(|v| v.checked_mul(in_h)) else {
        return;
    };
    if need_out == 0 || need_in == 0 || out_len < need_out || in_len < need_in {
        return;
    }
    if need_out > i32::MAX as usize || need_in > i32::MAX as usize {
        return;
    }
    let out_slice = std::slice::from_raw_parts_mut(out, need_out);
    let src_slice = std::slice::from_raw_parts(src, need_in);
    copy_roi(
        out_slice,
        src_slice,
        ch,
        in_w,
        in_h,
        out_w,
        out_h,
        dx as i64,
        dy as i64,
    );
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

#[allow(dead_code)]
fn ref_copy_alpha(out: &mut [f32], src: &[f32], width: usize, height: usize) {
    // Deliberately different: per-pixel base indexing (pixel p lives at
    // p*4..p*4+4) instead of the flat strided lane loop of the kernel.
    // Same result, different shape.
    let Some(npix) = width.checked_mul(height) else {
        return;
    };
    let m = npix.min(out.len() / 4).min(src.len() / 4);
    let mut p = 0;
    while p < m {
        out[p * 4 + 3] = src[p * 4 + 3];
        p += 1;
    }
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ref_copy_roi(
    out: &mut [f32],
    src: &[f32],
    ch: usize,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    dx: i64,
    dy: i64,
) {
    // Deliberately different: single flat pass over the output pixels with
    // div/mod coordinate recovery and a while-loop channel copy, versus the
    // kernel's nested row/col/for-channel loops. Same result.
    let Some(need_out) = ch.checked_mul(out_w).and_then(|v| v.checked_mul(out_h)) else {
        return;
    };
    let Some(need_in) = ch.checked_mul(in_w).and_then(|v| v.checked_mul(in_h)) else {
        return;
    };
    if need_out == 0 || need_in == 0 || out.len() < need_out || src.len() < need_in {
        return;
    }
    let npix = out_w * out_h;
    let mut p = 0;
    while p < npix {
        let row = p / out_w;
        let col = p % out_w;
        let irow = row as i64 + dy;
        let icol = col as i64 + dx;
        let ox = p * ch;
        let inside = irow >= 0 && irow < in_h as i64 && icol >= 0 && icol < in_w as i64;
        let mut c = 0;
        while c < ch {
            out[ox + c] = if inside {
                src[ch * ((irow as usize) * in_w + (icol as usize)) + c]
            } else {
                0.0
            };
            c += 1;
        }
        p += 1;
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

    // ── copy_alpha ────────────────────────────────────────────────────────────

    fn to_bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn copy_alpha_basic() {
        // 2x1 RGBA: alphas copied, RGB lanes untouched.
        let src = vec![1.0f32, 2.0, 3.0, 10.0, 5.0, 6.0, 7.0, 20.0];
        let mut out = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        copy_alpha(&mut out, &src, 2, 1);
        assert_eq!(
            to_bits(&out),
            to_bits(&[-1.0, -2.0, -3.0, 10.0, -5.0, -6.0, -7.0, 20.0])
        );
    }

    #[test]
    fn copy_alpha_golden_vector() {
        // Fixed 1x3 golden vector, compared bit-exact.
        let src = vec![
            0.0f32, 0.0, 0.0, 0.25, //
            1.0, 0.5, 0.125, 0.5, //
            0.3, 0.6, 0.9, 1.0,
        ];
        let mut out = vec![9.0f32; 12];
        copy_alpha(&mut out, &src, 3, 1);
        let expected = vec![
            9.0f32, 9.0, 9.0, 0.25, //
            9.0, 9.0, 9.0, 0.5, //
            9.0, 9.0, 9.0, 1.0,
        ];
        assert_eq!(to_bits(&out), to_bits(&expected));
    }

    #[test]
    fn copy_alpha_matches_reference_over_lcg() {
        let mut src = vec![0.0f32; 4 * 64];
        let mut dst = vec![0.0f32; 4 * 64];
        lcg_fill(&mut src, 0xA1FA, 1.0);
        lcg_fill(&mut dst, 0xC0FF, 1.0);

        let mut direct = dst.clone();
        let mut reference = dst.clone();
        copy_alpha(&mut direct, &src, 8, 8);
        ref_copy_alpha(&mut reference, &src, 8, 8);
        assert_eq!(to_bits(&direct), to_bits(&reference));

        // RGB lanes preserved from dst, alpha lanes taken from src.
        for p in 0..64 {
            assert_eq!(direct[p * 4 + 3].to_bits(), src[p * 4 + 3].to_bits());
            assert_eq!(
                to_bits(&direct[p * 4..p * 4 + 3]),
                to_bits(&dst[p * 4..p * 4 + 3])
            );
        }
    }

    #[test]
    fn copy_alpha_degenerate_empty() {
        let src = vec![1.0f32; 8];
        let mut out = vec![2.0f32; 8];
        copy_alpha(&mut out, &src, 0, 4); // width == 0: untouched
        copy_alpha(&mut out, &src, 2, 0); // height == 0: untouched
        copy_alpha(&mut out, &src, 0, 0);
        assert_eq!(out, vec![2.0; 8]);
        let mut empty: Vec<f32> = vec![];
        copy_alpha(&mut empty, &[], 0, 0); // no-op, no panic
        assert!(empty.is_empty());
    }

    #[test]
    fn copy_alpha_short_inputs_clamp() {
        // Buffers shorter than width*height*4: copy what fits, no panic.
        let src = vec![1.0f32, 2.0, 3.0, 10.0, 5.0];
        let mut direct = vec![0.0f32; 5];
        let mut reference = vec![0.0f32; 5];
        copy_alpha(&mut direct, &src, 2, 1);
        ref_copy_alpha(&mut reference, &src, 2, 1);
        assert_eq!(to_bits(&direct), to_bits(&reference));
        assert_eq!(direct[3].to_bits(), 10.0f32.to_bits());
        assert_eq!(&direct[..3], &[0.0, 0.0, 0.0][..]); // RGB untouched
        assert_eq!(direct[4].to_bits(), 0.0f32.to_bits()); // tail untouched
    }

    #[test]
    fn copy_alpha_overflow_noop() {
        // Overflowing width*height*4: no-op, no panic.
        let src = vec![1.0f32; 8];
        let mut out = vec![2.0f32; 8];
        copy_alpha(&mut out, &src, usize::MAX, 2);
        copy_alpha(&mut out, &src, usize::MAX, usize::MAX);
        assert_eq!(out, vec![2.0; 8]);
    }

    #[test]
    fn ffi_copy_alpha_round_trip() {
        let mut src = vec![0.0f32; 4 * 64];
        let mut dst = vec![0.0f32; 4 * 64];
        lcg_fill(&mut src, 0x5EED, 1.0);
        lcg_fill(&mut dst, 0xCAFE, 1.0);

        let mut ffi_out = dst.clone();
        let mut direct_out = dst.clone();
        unsafe {
            darkroom_imagebuf_copy_alpha(src.as_ptr(), ffi_out.as_mut_ptr(), 8, 8);
        }
        copy_alpha(&mut direct_out, &src, 8, 8);
        assert_eq!(to_bits(&ffi_out), to_bits(&direct_out));
    }

    #[test]
    fn ffi_copy_alpha_null_guard() {
        let mut buf = vec![1.0f32; 8];
        unsafe {
            darkroom_imagebuf_copy_alpha(std::ptr::null(), buf.as_mut_ptr(), 2, 1);
            darkroom_imagebuf_copy_alpha(buf.as_ptr(), std::ptr::null_mut(), 2, 1);
            darkroom_imagebuf_copy_alpha(std::ptr::null(), std::ptr::null_mut(), 2, 1);
        }
        assert_eq!(buf, vec![1.0; 8]); // untouched
    }

    #[test]
    fn ffi_copy_alpha_zero_dims_guard() {
        let src = vec![3.0f32; 8];
        let mut out = vec![1.0f32; 8];
        unsafe {
            darkroom_imagebuf_copy_alpha(src.as_ptr(), out.as_mut_ptr(), 0, 2);
            darkroom_imagebuf_copy_alpha(src.as_ptr(), out.as_mut_ptr(), 2, 0);
        }
        assert_eq!(out, vec![1.0; 8]); // untouched
    }

    #[test]
    fn ffi_copy_alpha_overflow_guard() {
        // Overflow must return before any slice is built: no panic, untouched.
        let src = vec![3.0f32; 8];
        let mut out = vec![1.0f32; 8];
        unsafe {
            darkroom_imagebuf_copy_alpha(src.as_ptr(), out.as_mut_ptr(), usize::MAX, 2);
        }
        assert_eq!(out, vec![1.0; 8]); // untouched
    }

    #[test]
    fn ffi_copy_alpha_alias_in_place() {
        // in == out: each alpha lane reads itself; buffer bit-identical.
        let mut buf = vec![0.0f32; 4 * 16];
        lcg_fill(&mut buf, 0xA11A5, 1.0);
        let before = to_bits(&buf);
        unsafe {
            darkroom_imagebuf_copy_alpha(buf.as_ptr(), buf.as_mut_ptr(), 4, 4);
        }
        assert_eq!(to_bits(&buf), before);
    }

    // ── copy_roi ─────────────────────────────────────────────────────────────

    #[test]
    fn copy_roi_full_overlap_is_identity() {
        // Same dims, zero offset: out == in, bit-exact.
        let mut src = vec![0.0f32; 4 * 6];
        lcg_fill(&mut src, 0xC01, 1.0);
        let mut out = vec![-7.0f32; 4 * 6];
        copy_roi(&mut out, &src, 4, 3, 2, 3, 2, 0, 0);
        assert_eq!(to_bits(&out), to_bits(&src));
    }

    #[test]
    fn copy_roi_shifted_window() {
        // 4x4 single-channel input 0..16; 2x2 output at dx=1, dy=1.
        let src: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let mut out = vec![-1.0f32; 4];
        copy_roi(&mut out, &src, 1, 4, 4, 2, 2, 1, 1);
        assert_eq!(to_bits(&out), to_bits(&[5.0, 6.0, 9.0, 10.0]));
    }

    #[test]
    fn copy_roi_zero_pad() {
        // 2x2 input, 3x3 output at dx=-1, dy=-1: the input lands in the
        // bottom-right corner, everything else is +0.0.
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![-1.0f32; 9];
        copy_roi(&mut out, &src, 1, 2, 2, 3, 3, -1, -1);
        assert_eq!(
            to_bits(&out),
            to_bits(&[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
        );
    }

    #[test]
    fn copy_roi_multichannel_shift_and_pad() {
        // 2x1 RGBA input; 2x2 RGBA output at dx=0, dy=-1: first row zeros,
        // second row carries the input row.
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![-1.0f32; 16];
        copy_roi(&mut out, &src, 4, 2, 1, 2, 2, 0, -1);
        assert_eq!(
            to_bits(&out),
            to_bits(&[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ])
        );
    }

    #[test]
    fn copy_roi_matches_reference_over_lcg() {
        // Aligned overlap plus a shifted window with zero padding, incl. ch=1.
        for (ch, in_w, in_h, out_w, out_h, dx, dy) in [
            (4usize, 8, 6, 8, 6, 0i64, 0i64),
            (4, 8, 6, 5, 4, 2, 1),
            (4, 5, 4, 8, 6, -2, -1),
            (1, 7, 5, 4, 9, 3, -2),
        ] {
            let mut src = vec![0.0f32; ch * in_w * in_h];
            lcg_fill(&mut src, 0xC0E1, 1.0);
            let mut direct = vec![-3.0f32; ch * out_w * out_h];
            let mut reference = vec![-3.0f32; ch * out_w * out_h];
            copy_roi(&mut direct, &src, ch, in_w, in_h, out_w, out_h, dx, dy);
            ref_copy_roi(&mut reference, &src, ch, in_w, in_h, out_w, out_h, dx, dy);
            assert_eq!(to_bits(&direct), to_bits(&reference), "dx={dx} dy={dy}");
        }
    }

    #[test]
    fn copy_roi_nan_payload_bit_exact() {
        // NaN payloads and infinities survive the copy bit-identical.
        let nan = f32::from_bits(0x7FC0_1234);
        let src = vec![nan, f32::INFINITY, f32::NEG_INFINITY, 1.5];
        let mut out = vec![0.0f32; 4];
        copy_roi(&mut out, &src, 1, 2, 2, 2, 2, 0, 0);
        assert_eq!(to_bits(&out), to_bits(&src));
    }

    #[test]
    fn copy_roi_degenerate_noop() {
        let src = vec![1.0f32; 16];
        let mut out = vec![2.0f32; 16];
        copy_roi(&mut out, &src, 4, 2, 2, 2, 2, 0, 0); // sanity: runs
        copy_roi(&mut out, &src, 0, 2, 2, 2, 2, 0, 0); // ch == 0
        copy_roi(&mut out, &src, 4, 0, 2, 2, 2, 0, 0); // in_w == 0
        copy_roi(&mut out, &src, 4, 2, 2, 0, 2, 0, 0); // out_w == 0
        copy_roi(&mut out, &src, usize::MAX, 2, 2, 2, 0, 0, 0); // overflow
        let mut empty: Vec<f32> = vec![];
        copy_roi(&mut empty, &[], 4, 0, 0, 0, 0, 0, 0); // no panic
        assert!(empty.is_empty());
    }

    #[test]
    fn copy_roi_short_buffers_noop() {
        // Buffers shorter than ch*w*h: untouched, no panic.
        let src = vec![1.0f32; 8];
        let mut out = vec![2.0f32; 8];
        copy_roi(&mut out, &src, 4, 2, 2, 2, 2, 0, 0); // needs 16 each
        assert_eq!(out, vec![2.0; 8]);
    }

    #[test]
    fn ffi_copy_roi_round_trip() {
        let mut src = vec![0.0f32; 4 * 8 * 6];
        lcg_fill(&mut src, 0xBEEF, 1.0);
        let (in_w, in_h) = (8usize, 6usize);
        let (out_w, out_h) = (5usize, 4usize);
        let (dx, dy) = (2i32, 1i32);
        let mut ffi_out = vec![-3.0f32; 4 * out_w * out_h];
        let mut direct_out = vec![-3.0f32; 4 * out_w * out_h];
        unsafe {
            darkroom_imagebuf_copy_roi(
                ffi_out.as_mut_ptr(),
                src.as_ptr(),
                4,
                in_w,
                in_h,
                out_w,
                out_h,
                dx,
                dy,
                ffi_out.len(),
                src.len(),
            );
        }
        copy_roi(
            &mut direct_out,
            &src,
            4,
            in_w,
            in_h,
            out_w,
            out_h,
            dx as i64,
            dy as i64,
        );
        assert_eq!(to_bits(&ffi_out), to_bits(&direct_out));
    }

    #[test]
    fn ffi_copy_roi_guards() {
        let src = vec![1.0f32; 16];
        let mut out = vec![2.0f32; 16];
        unsafe {
            // NULL pointers.
            darkroom_imagebuf_copy_roi(
                std::ptr::null_mut(),
                src.as_ptr(),
                4,
                2,
                2,
                2,
                2,
                0,
                0,
                16,
                16,
            );
            darkroom_imagebuf_copy_roi(
                out.as_mut_ptr(),
                std::ptr::null(),
                4,
                2,
                2,
                2,
                2,
                0,
                0,
                16,
                16,
            );
            // Empty dims.
            darkroom_imagebuf_copy_roi(
                out.as_mut_ptr(),
                src.as_ptr(),
                4,
                2,
                2,
                0,
                2,
                0,
                0,
                16,
                16,
            );
            // Short caller lengths.
            darkroom_imagebuf_copy_roi(
                out.as_mut_ptr(),
                src.as_ptr(),
                4,
                2,
                2,
                2,
                2,
                0,
                0,
                15,
                16,
            );
            darkroom_imagebuf_copy_roi(
                out.as_mut_ptr(),
                src.as_ptr(),
                4,
                2,
                2,
                2,
                2,
                0,
                0,
                16,
                15,
            );
        }
        assert_eq!(out, vec![2.0; 16]); // untouched
    }
}
