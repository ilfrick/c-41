use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Blurs;

impl IopProcess for Blurs {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "blurs" }
}

/// Restore the alpha channel of `out` from `in` after a Gaussian blur that
/// overwrote it.
///
/// For each pixel k:  out[k*4+3] = in[k*4+3]
///
/// Matches the DT_OMP_FOR_SIMD at src/iop/blurs.c:601.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_alpha_restore(
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    if npixels == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out  = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in 0..npixels {
        out[k * 4 + 3] = inp[k * 4 + 3];
    }
}

/// Sparse spatial convolution for the lens/motion blur paths.
///
/// Matches the DT_OMP_FOR(collapse(2)) at src/iop/blurs.c:652.
///
/// For each output pixel (i, j):
///   ci = i + oy;  cj = j + ox    (position within the input roi)
///   if ci/cj are in the interior (margin = radius on all sides):
///     acc += values[s] * in[center + offsets[s]]   for s in 0..n_nonzero
///   else (edge):
///     acc += kernel[(l+radius)*kw + (m+radius)] * in[clamp_pixel]
///          for l in -radius..=radius, m in -radius..=radius, if kernel > 1e-6
///   out[idx + 0..2] = acc[0..2]
///   out[idx + 3]    = in[ci*in_width + cj].alpha   (original alpha preserved)
///
/// Parameters:
///   `offsets` — `n_nonzero` signed word-offsets (in f32 elements from the
///               centre pixel pointer) for the interior fast path.
///               C type: `ptrdiff_t[]`; Rust maps to `isize`.
///   `values`  — corresponding kernel weights for the interior path.
///   `kernel`  — full `(2*radius+1)²` kernel matrix for the edge path.
///   `ox`, `oy` — output-in-input origin: `roi_out.x - roi_in.x` etc.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_sparse_convolve(
    in_buf:   *const f32,
    out_buf:  *mut f32,
    out_width:  usize,
    out_height: usize,
    in_width:   usize,
    in_height:  usize,
    radius: i32,
    ox: i32,
    oy: i32,
    offsets:    *const isize,  // n_nonzero entries
    values:     *const f32,    // n_nonzero entries
    n_nonzero:  usize,
    kernel:     *const f32,    // (2*radius+1)^2 entries
) {
    if out_width == 0 || out_height == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf,  in_width  * in_height * 4);
    let out  = std::slice::from_raw_parts_mut(out_buf, out_width * out_height * 4);
    let off  = std::slice::from_raw_parts(offsets, n_nonzero);
    let val  = std::slice::from_raw_parts(values,  n_nonzero);
    let kw   = (2 * radius + 1) as usize;
    let ker  = std::slice::from_raw_parts(kernel, kw * kw);

    let r    = radius as i32;
    let inh  = in_height as i32;
    let inw  = in_width  as i32;

    for i in 0..out_height {
        for j in 0..out_width {
            let idx_out = (i * out_width + j) * 4;
            let mut acc = [0.0_f32; 4];
            let ci = i as i32 + oy;
            let cj = j as i32 + ox;

            if ci >= r && cj >= r && ci < inh - r && cj < inw - r {
                // Interior: use precomputed sparse offsets
                let center = (ci as usize * in_width + cj as usize) * 4;
                for s in 0..n_nonzero {
                    let src_base = (center as isize + off[s]) as usize;
                    let w = val[s];
                    acc[0] += w * inp[src_base];
                    acc[1] += w * inp[src_base + 1];
                    acc[2] += w * inp[src_base + 2];
                    acc[3] += w * inp[src_base + 3];
                }
            } else {
                // Edge: clamp neighbours to input bounds
                for l in -r..=r {
                    for m in -r..=r {
                        let k = ker[((l + r) as usize) * kw + ((m + r) as usize)];
                        if k > 1e-6 {
                            let ii = (ci + l).clamp(0, inh - 1) as usize;
                            let jj = (cj + m).clamp(0, inw - 1) as usize;
                            let src_base = (ii * in_width + jj) * 4;
                            acc[0] += k * inp[src_base];
                            acc[1] += k * inp[src_base + 1];
                            acc[2] += k * inp[src_base + 2];
                            acc[3] += k * inp[src_base + 3];
                        }
                    }
                }
            }

            out[idx_out]     = acc[0];
            out[idx_out + 1] = acc[1];
            out[idx_out + 2] = acc[2];
            // Preserve original alpha from the centre input pixel
            out[idx_out + 3] = inp[(ci as usize * in_width + cj as usize) * 4 + 3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_restore_copies_from_input() {
        let inp = vec![0.1_f32, 0.2, 0.3, 0.42];
        let mut out = vec![0.5_f32, 0.5, 0.5, 0.99];
        unsafe { darkroom_blurs_alpha_restore(inp.as_ptr(), out.as_mut_ptr(), 1); }
        // RGB unchanged, alpha from in
        assert_eq!(out[3], 0.42);
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn sparse_convolve_identity_kernel() {
        // 3×3 interior pixel, radius=1; kernel = identity (1.0 at centre, 0 elsewhere).
        // offsets: just [0] (center), value 1.0 → out = in.
        let w = 5; let h = 5;
        let mut inp = vec![0.0_f32; w * h * 4];
        // Set the centre pixel's RGB
        let cx = 2; let cy = 2;
        inp[(cy * w + cx) * 4]     = 0.3;
        inp[(cy * w + cx) * 4 + 1] = 0.5;
        inp[(cy * w + cx) * 4 + 2] = 0.7;
        inp[(cy * w + cx) * 4 + 3] = 1.0;

        let offsets: Vec<isize> = vec![0isize]; // centre only
        let values  = vec![1.0_f32];
        let kw = 3usize;
        let mut kernel = vec![0.0_f32; kw * kw];
        kernel[1 * kw + 1] = 1.0; // centre of 3×3

        let mut out = vec![0.0_f32; w * h * 4];
        unsafe {
            darkroom_blurs_sparse_convolve(
                inp.as_ptr(), out.as_mut_ptr(),
                w, h, w, h,
                1, 0, 0,
                offsets.as_ptr(), values.as_ptr(), 1,
                kernel.as_ptr(),
            );
        }
        // Output centre pixel should equal input
        let oi = (cy * w + cx) * 4;
        assert!((out[oi]     - 0.3).abs() < 1e-5);
        assert!((out[oi + 1] - 0.5).abs() < 1e-5);
        assert!((out[oi + 2] - 0.7).abs() < 1e-5);
        assert_eq!(out[oi + 3], 1.0); // alpha preserved
    }

    #[test]
    fn sparse_convolve_edge_pixel_does_not_panic() {
        // 3×3 image, radius=2. Interior check requires ci >= 2 AND ci < h-2 = 1 — impossible for
        // any pixel, so ALL pixels go through the edge path. The sparse list is unused.
        let w = 3; let h = 3;
        let inp = vec![0.5_f32; w * h * 4];
        let kw = 5usize; // radius=2 → 5×5 kernel
        let kernel = vec![1.0_f32 / 25.0; kw * kw]; // uniform box blur
        let offsets: Vec<isize> = vec![];
        let values: Vec<f32> = vec![];
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe {
            darkroom_blurs_sparse_convolve(
                inp.as_ptr(), out.as_mut_ptr(),
                w, h, w, h, 2, 0, 0, // radius=2
                offsets.as_ptr(), values.as_ptr(), 0,
                kernel.as_ptr(),
            );
        }
        // Uniform input → uniform output (box blur of constant field = same constant)
        for i in 0..(w * h) {
            assert!((out[i * 4] - 0.5).abs() < 1e-5, "k={i} out={}", out[i * 4]);
        }
    }
}

/// 2D B-spline (5-tap per axis) single-channel convolution with clamp-to-edge.
/// filter = [1/16, 4/16, 6/16, 4/16, 1/16] applied non-separably.
/// Matches _blur_2D_Bspline DT_OMP_FOR(collapse(2)) at blurs.c:152.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_bspline_2d(
    in_buf:  *const f32,
    out_buf: *mut f32,
    width:   usize,
    height:  usize,
) {
    const FILTER: [f32; 5] = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0];
    let inp = std::slice::from_raw_parts(in_buf, width * height);
    let out = std::slice::from_raw_parts_mut(out_buf, width * height);
    let hi = height as i32;
    let wi = width as i32;
    for i in 0..height {
        for j in 0..width {
            let mut acc = 0.0f32;
            for ii in 0usize..5 {
                let row = (i as i32 + ii as i32 - 2).clamp(0, hi - 1) as usize;
                for jj in 0usize..5 {
                    let col = (j as i32 + jj as i32 - 2).clamp(0, wi - 1) as usize;
                    acc += FILTER[ii] * FILTER[jj] * inp[row * width + col];
                }
            }
            out[i * width + j] = acc;
        }
    }
}

// ── Kernel builders (Phase 2z+57) ────────────────────────────────────────

/// Zero-initialise a float kernel buffer. Matches blurs.c::_init_kernel DT_OMP_FOR_SIMD.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_init_kernel(buf: *mut f32, n: usize) {
    std::slice::from_raw_parts_mut(buf, n).fill(0.0);
}

/// 2D Gaussian kernel: buf[i,j] = exp(-4*(x²+y²)) for (x,y) in [-1,1]².
/// Matches blurs.c::_create_gauss_kernel DT_OMP_FOR(collapse(2)).
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_gauss_kernel(
    buf: *mut f32, width: usize, height: usize,
) {
    let b = std::slice::from_raw_parts_mut(buf, width * height);
    let radius = (width - 1) as f32 / 2.0 - 1.0;
    for i in 0..height {
        for j in 0..width {
            let x = (i as f32 - 1.0) / radius - 1.0;
            let y = (j as f32 - 1.0) / radius - 1.0;
            b[i * width + j] = (-4.0 * (x*x + y*y)).exp();
        }
    }
}

/// Lens diaphragm kernel: 1.0 inside envelope, 0.0 outside.
/// n=blades, m=concavity, k=roundness, rotation in radians.
/// Matches blurs.c::_create_lens_kernel DT_OMP_FOR(collapse(2)).
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_lens_kernel(
    buf: *mut f32, width: usize, height: usize,
    n: f32, m: f32, k: f32, rotation: f32,
) {
    let b = std::slice::from_raw_parts_mut(buf, width * height);
    let eps = 1.0 / width as f32;
    let radius = (width - 1) as f32 / 2.0 - 1.0;
    for i in 0..height {
        for j in 0..width {
            let x = (i as f32 - 1.0) / radius - 1.0;
            let y = (j as f32 - 1.0) / radius - 1.0;
            let r = x.hypot(y);
            let angle = y.atan2(x);
            let num   = ((2.0 * k.asin() + std::f32::consts::PI * m) / (2.0 * n)).cos();
            let denom = ((2.0 * (k * (n * (angle + rotation)).cos()).asin()
                          + std::f32::consts::PI * m) / (2.0 * n))
                .cos()
                .max(1e-9);
            let big_m = num / denom;
            b[i * width + j] = if big_m >= r + eps { 1.0 } else { 0.0 };
        }
    }
}

/// Motion blur kernel: paints the polynomial arc into buf.
/// a=curvature/2, offset=user offset param; rot_m is the 2×2 rotation matrix.
/// Internally computes B=1, C = -a*offset^2 + offset, then for each i:
///   x = (i/8 - 1)/radius - 1; X = x - offset; y = X^2*a + X + C
/// Matches blurs.c::_create_motion_kernel DT_OMP_FOR_SIMD.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_motion_kernel(
    buf: *mut f32, width: usize,
    a: f32, offset: f32,
    rot_m: *const f32,   // 4 floats: [M00, M01, M10, M11]
    radius: f32, eps: f32,
) {
    let b = std::slice::from_raw_parts_mut(buf, width * width);
    let m = std::slice::from_raw_parts(rot_m, 4);
    let c = -a * offset * offset + offset; // C = -A*offset^2 + B*offset, B=1
    let w = width as i32;
    for i in 0..8 * width {
        let x  = (i as f32 / 8.0 - 1.0) / radius - 1.0;
        let bx = x - offset;                        // X = x - offset
        let y  = bx * bx * a + bx + c;             // y = X^2*A + X*B + C
        let rx = x * m[0] + y * m[1];
        let ry = x * m[2] + y * m[3];
        let yf = [((ry + 1.0) * radius - eps).round() as i32,
                  ((ry + 1.0) * radius + eps).round() as i32];
        let xf = [((rx + 1.0) * radius - eps).round() as i32,
                  ((rx + 1.0) * radius + eps).round() as i32];
        for &xi in &xf {
            for &yi in &yf {
                if xi > 0 && xi < w-1 && yi > 0 && yi < w-1 {
                    b[yi as usize * width + xi as usize] = 1.0;
                }
            }
        }
    }
}

/// Convert float kernel to RGBA u8 display buffer (for GUI).
/// rgba[k*4+0..3] = round(255*kernel[k]) for all channels.
/// Matches blurs.c:345 DT_OMP_FOR.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_gui_rgba(
    kernel: *const f32, rgba: *mut u8, n: usize,
) {
    let k = std::slice::from_raw_parts(kernel, n);
    let r = std::slice::from_raw_parts_mut(rgba, n * 4);
    for i in 0..n {
        let v = (255.0 * k[i]).round() as u8;
        r[i*4] = v; r[i*4+1] = v; r[i*4+2] = v; r[i*4+3] = v;
    }
}

/// Sum of all elements in a float buffer. Matches blurs.c::_compute_norm.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_compute_norm(buf: *const f32, n: usize) -> f32 {
    std::slice::from_raw_parts(buf, n).iter().sum()
}

/// Divide each element by norm in-place. Matches blurs.c::_normalize.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_normalize(buf: *mut f32, n: usize, norm: f32) {
    let b = std::slice::from_raw_parts_mut(buf, n);
    if norm.abs() > 1e-10 { b.iter_mut().for_each(|v| *v /= norm); }
}

/// Copy RGBA image rows into the padded buffer (interior pixels).
/// Matches blurs.c:454 DT_OMP_FOR_SIMD.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_pad_image(
    in_buf: *const f32, padded: *mut f32,
    in_height: usize, in_width: usize, padded_width: usize,
) {
    let inp  = std::slice::from_raw_parts(in_buf, in_width * in_height * 4);
    let outp = std::slice::from_raw_parts_mut(padded, padded_width * in_height * 4);
    for i in 0..in_height {
        for j in 0..in_width {
            for c in 0..4 {
                outp[(i * padded_width + j) * 4 + c] = inp[(i * in_width + j) * 4 + c];
            }
        }
    }
}

/// Fill right-edge column of padded buffer from last input column.
/// Matches blurs.c:466 DT_OMP_FOR_SIMD.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_pad_right_edge(
    in_buf: *const f32, padded: *mut f32,
    in_height: usize, in_width: usize, padded_width: usize,
) {
    let inp  = std::slice::from_raw_parts(in_buf, in_width * in_height * 4);
    let outp = std::slice::from_raw_parts_mut(padded, padded_width * in_height * 4);
    for i in 0..in_height {
        let si = (i * (in_width - 1)) * 4;
        let di = (i * (padded_width - 1)) * 4;
        for c in 0..4 { outp[di + c] = inp[si + c]; }
    }
}

/// Fill bottom-edge row of padded buffer from last input row.
/// Matches blurs.c:477 DT_OMP_FOR_SIMD.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_pad_bottom_edge(
    in_buf: *const f32, padded: *mut f32,
    in_height: usize, in_width: usize,
    padded_width: usize, _padded_height: usize,
) {
    let inp  = std::slice::from_raw_parts(in_buf, in_width * in_height * 4);
    let outp = std::slice::from_raw_parts_mut(padded, padded_width * (in_height + 1) * 4);
    for j in 0..in_width {
        let si = ((in_height - 1) * in_width + j) * 4;
        let di = ((_padded_height - 1) * padded_width + j) * 4;
        for c in 0..4 { outp[di + c] = inp[si + c]; }
    }
}

/// Place a kernel in the centre of a zero-padded kernel buffer.
/// Matches blurs.c:500 DT_OMP_FOR_SIMD.
#[no_mangle]
pub unsafe extern "C" fn darkroom_blurs_pad_kernel(
    kernel: *const f32, padded_kernel: *mut f32,
    kernel_width: usize, padded_width: usize,
    offset_i: usize, offset_j: usize,
) {
    let k  = std::slice::from_raw_parts(kernel, kernel_width * kernel_width);
    let pk = std::slice::from_raw_parts_mut(padded_kernel, padded_width * padded_width);
    pk.fill(0.0);
    let i_reach = offset_i + kernel_width;
    let j_reach = offset_j + kernel_width;
    for i in 0..padded_width {
        for j in 0..padded_width {
            if i >= offset_i && i < i_reach && j >= offset_j && j < j_reach {
                pk[i * padded_width + j] = k[(i - offset_i) * kernel_width + (j - offset_j)];
            }
        }
    }
}
