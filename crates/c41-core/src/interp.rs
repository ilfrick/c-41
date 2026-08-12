//! 2D separable pixel interpolation for RGBA images.
//!
//! Implements bilinear, bicubic, Lanczos-2, and Lanczos-3 upsampling kernels
//! as faithful ports of darktable's interpolation.c. Used to replace
//! dt_interpolation_compute_pixel4c() calls in IOP process loops.

use std::f32::consts::PI;

// ── Mirror-clamp helper ───────────────────────────────────────────────────

#[inline(always)]
fn mirror(i: i32, max: i32) -> i32 {
    if i < 0 { -i }
    else if i > max { max - (i - max) }
    else { i }
}

// ── Kernel tap generators ─────────────────────────────────────────────────

/// Bilinear tap: 1 - |t|  (defined for |t| <= 1)
#[inline(always)]
fn bilinear_tap(t: f32) -> f32 {
    1.0 - t.abs()
}

/// Bicubic tap with a = -0.5 (Mitchell-Netravali).
/// Matches _maketaps_bicubic in interpolation.c.
#[inline(always)]
fn bicubic_tap(t: f32) -> f32 {
    let t_abs = t.abs();
    let t2    = t * t;
    if t_abs <= 1.0 {
        (3.0 * t_abs * t2 - 5.0 * t2 + 2.0) * 0.5
    } else if t_abs < 2.0 {
        (-t_abs * t2 + 5.0 * t2 - 8.0 * t_abs + 4.0) * 0.5
    } else {
        0.0
    }
}

/// Lanczos tap with `n` lobes. Matches _maketaps_lanczos in interpolation.c.
#[inline(always)]
fn lanczos_tap(t: f32, n: f32) -> f32 {
    const EPS: f32 = 1e-9;
    if t.abs() >= n { return 0.0; }
    if t.abs() < EPS { return 1.0; }
    // sin(pi*t) = sign * sin(pi*r)  where a = int(t), r = t - a
    let a      = t as i32;
    let r      = t - a as f32;
    let sign   = if a & 1 != 0 { -1.0f32 } else { 1.0 };
    let sine1  = (PI * r).sin();
    let sine2  = (PI * t / n).sin();
    let num    = sign * sine1 * n * sine2;
    let denom  = PI * PI * t * t + EPS;
    num / denom
}

// ── Kernel builder ────────────────────────────────────────────────────────

/// Half-filter width for each interpolation type.
const fn half_width(interp: u32) -> usize {
    match interp {
        0 => 1, // bilinear
        1 | 2 => 2, // bicubic, lanczos2
        _ => 3, // lanczos3
    }
}

/// Fill `2*hw` taps for fractional coordinate offset `t_rel`.
/// `t_rel = coord - first_pixel`, `interval = -1.0`.
/// Returns the kernel norm (1.0 for bilinear/bicubic, computed for Lanczos).
fn make_kernel(taps: &mut [f32], t_rel: f32, hw: usize, interp: u32) -> f32 {
    let n = 2 * hw;
    for j in 0..n {
        let vt = t_rel - j as f32;
        taps[j] = match interp {
            0 => bilinear_tap(vt),
            1 => bicubic_tap(vt),
            2 => lanczos_tap(vt, 2.0),
            _ => lanczos_tap(vt, 3.0),
        };
    }
    // Bilinear and bicubic have unit norm by construction; Lanczos must be summed.
    if interp >= 2 { taps[..n].iter().sum() } else { 1.0 }
}

// ── Public interpolation API ──────────────────────────────────────────────

/// Compute one interpolated RGBA pixel at fractional coordinate (x, y).
///
/// Uses mirror-clamp at image boundaries (matching darktable's slow path).
/// `linestride` is the number of **floats** per row (normally `width * 4`).
pub fn compute_pixel4c(
    in_buf: &[f32],
    x: f32,
    y: f32,
    img_width: i32,
    img_height: i32,
    linestride: i32,
    interp: u32,
) -> [f32; 4] {
    let hw = half_width(interp);
    let ksz = 2 * hw;

    let mut kh = [0.0f32; 6]; // max Lanczos3 = 6 taps
    let mut kv = [0.0f32; 6];

    // first contributing pixel index for each axis
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let t_rel_x = x - (ix - hw as i32 + 1) as f32;
    let t_rel_y = y - (iy - hw as i32 + 1) as f32;

    let norm_h = make_kernel(&mut kh[..ksz], t_rel_x, hw, interp);
    let norm_v = make_kernel(&mut kv[..ksz], t_rel_y, hw, interp);
    let oonorm  = 1.0 / (norm_h * norm_v);

    let start_x = ix - hw as i32 + 1;
    let start_y = iy - hw as i32 + 1;
    let max_x   = img_width  - 1;
    let max_y   = img_height - 1;

    let mut pixel = [0.0f32; 4];
    for i in 0..ksz {
        let py   = mirror(start_y + i as i32, max_y) as usize;
        let base = py * linestride as usize;
        let mut h = [0.0f32; 4];
        for j in 0..ksz {
            let px  = mirror(start_x + j as i32, max_x) as usize;
            let off = base + px * 4;
            let k   = kh[j];
            h[0] += k * in_buf[off];
            h[1] += k * in_buf[off + 1];
            h[2] += k * in_buf[off + 2];
            h[3] += k * in_buf[off + 3];
        }
        pixel[0] += kv[i] * h[0];
        pixel[1] += kv[i] * h[1];
        pixel[2] += kv[i] * h[2];
        pixel[3] += kv[i] * h[3];
    }
    [pixel[0] * oonorm, pixel[1] * oonorm, pixel[2] * oonorm, pixel[3] * oonorm]
}

/// Compute one interpolated single-channel sample at fractional (x, y).
///
/// `linestride`: floats per image row (= width for single-channel).
/// Equivalent to dt_interpolation_compute_sample with samplestride=1.
pub fn compute_sample_1c(
    in_buf: &[f32],
    x: f32,
    y: f32,
    img_width: i32,
    img_height: i32,
    linestride: i32,
    interp: u32,
) -> f32 {
    let hw  = half_width(interp);
    let ksz = 2 * hw;
    let mut kh = [0.0f32; 6];
    let mut kv = [0.0f32; 6];
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let norm_h = make_kernel(&mut kh[..ksz], x - (ix - hw as i32 + 1) as f32, hw, interp);
    let norm_v = make_kernel(&mut kv[..ksz], y - (iy - hw as i32 + 1) as f32, hw, interp);
    let oonorm  = 1.0 / (norm_h * norm_v);
    let start_x = ix - hw as i32 + 1;
    let start_y = iy - hw as i32 + 1;
    let max_x   = img_width  - 1;
    let max_y   = img_height - 1;
    let mut acc = 0.0f32;
    for i in 0..ksz {
        let py   = mirror(start_y + i as i32, max_y) as usize;
        let mut h = 0.0f32;
        for j in 0..ksz {
            let px  = mirror(start_x + j as i32, max_x) as usize;
            h += kh[j] * in_buf[py * linestride as usize + px];
        }
        acc += kv[i] * h;
    }
    acc * oonorm
}

/// 4-channel scale resampling: maps each output pixel (i, j) to (i*x_scale, j*y_scale)
/// in the input and interpolates. Matches the DT_OMP_FOR in scalepixels.c::process().
#[no_mangle]
pub unsafe extern "C" fn darkroom_scalepixels_process(
    in_buf:      *const f32,
    out_buf:     *mut f32,
    out_width:   i32,
    out_height:  i32,
    in_width:    i32,
    in_height:   i32,
    x_scale:     f32,
    y_scale:     f32,
    interp_type: u32,
) {
    if out_width <= 0 || out_height <= 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf, (in_width * in_height * 4) as usize);
    let outp = std::slice::from_raw_parts_mut(out_buf, (out_width * out_height * 4) as usize);
    let ls   = in_width * 4;
    for j in 0..out_height as usize {
        for i in 0..out_width as usize {
            let x = i as f32 * x_scale;
            let y = j as f32 * y_scale;
            if x >= 0.0 && y >= 0.0 && x < in_width as f32 && y < in_height as f32 {
                let px = compute_pixel4c(inp, x, y, in_width, in_height, ls, interp_type);
                let b  = (j * out_width as usize + i) * 4;
                outp[b] = px[0]; outp[b+1] = px[1]; outp[b+2] = px[2]; outp[b+3] = px[3];
            } else {
                let b = (j * out_width as usize + i) * 4;
                outp[b] = 0.0; outp[b+1] = 0.0; outp[b+2] = 0.0; outp[b+3] = 0.0;
            }
        }
    }
}

/// 1-channel scale resampling with CLIP: maps (i,j) -> (i*x_scale, j*y_scale).
/// Matches the DT_OMP_FOR(collapse(2)) in scalepixels.c::distort_mask().
#[no_mangle]
pub unsafe extern "C" fn darkroom_scalepixels_distort_mask(
    in_buf:      *const f32,
    out_buf:     *mut f32,
    out_width:   i32,
    out_height:  i32,
    in_width:    i32,
    in_height:   i32,
    x_scale:     f32,
    y_scale:     f32,
    interp_type: u32,
) {
    if out_width <= 0 || out_height <= 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf, (in_width * in_height) as usize);
    let outp = std::slice::from_raw_parts_mut(out_buf, (out_width * out_height) as usize);
    for row in 0..out_height as usize {
        for col in 0..out_width as usize {
            let x = col as f32 * x_scale;
            let y = row as f32 * y_scale;
            let v = compute_sample_1c(inp, x, y, in_width, in_height, in_width, interp_type);
            outp[row * out_width as usize + col] = v.clamp(0.0, 1.0);
        }
    }
}

// ── Homographic projection helper ────────────────────────────────────────

/// Apply a 3×3 homography matrix (row-major) to a 2D point + perspective divide.
/// Returns (x, y) in input-image space, or None if pin[2] ≈ 0.
#[inline(always)]
fn homographic_project(
    out_x: f32, out_y: f32,
    roi_out_scale: f32,
    roi_in_scale: f32,
    roi_in_x: f32, roi_in_y: f32,
    cx: f32, cy: f32,
    h: &[f32; 9],
) -> Option<(f32, f32)> {
    let px = (out_x + cx) / roi_out_scale;
    let py = (out_y + cy) / roi_out_scale;
    // mat3mulv: pin = h * [px, py, 1]
    let pin0 = h[0]*px + h[1]*py + h[2];
    let pin1 = h[3]*px + h[4]*py + h[5];
    let pin2 = h[6]*px + h[7]*py + h[8];
    if pin2.abs() < 1e-9 { return None; }
    let x = pin0 / pin2 * roi_in_scale - roi_in_x;
    let y = pin1 / pin2 * roi_in_scale - roi_in_y;
    Some((x, y))
}

/// Ashift distort_mask: 1-channel homographic resampling with CLIP [0,1].
/// Matches the DT_OMP_FOR(shared(ihomograph)) in ashift.c:1106.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_distort_mask(
    in_buf:         *const f32,
    out_buf:        *mut f32,
    out_width:      i32,
    out_height:     i32,
    roi_out_x:      f32,
    roi_out_y:      f32,
    roi_out_scale:  f32,
    in_width:       i32,
    in_height:      i32,
    roi_in_x:       f32,
    roi_in_y:       f32,
    roi_in_scale:   f32,
    cx:             f32,
    cy:             f32,
    ihomograph:     *const f32,   // 9 floats, row-major
    interp_type:    u32,
) {
    if out_width <= 0 || out_height <= 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf, (in_width * in_height) as usize);
    let outp = std::slice::from_raw_parts_mut(out_buf, (out_width * out_height) as usize);
    let h: &[f32; 9] = &*(ihomograph as *const [f32; 9]);

    for j in 0..out_height as usize {
        for i in 0..out_width as usize {
            let coord = homographic_project(
                roi_out_x + i as f32, roi_out_y + j as f32,
                roi_out_scale, roi_in_scale, roi_in_x, roi_in_y, cx, cy, h,
            );
            outp[j * out_width as usize + i] = match coord {
                Some((x, y)) => compute_sample_1c(
                    inp, x, y, in_width, in_height, in_width, interp_type,
                ).clamp(0.0, 1.0),
                None => 0.0,
            };
        }
    }
}

/// Ashift process: 4-channel RGBA homographic resampling.
/// Matches the DT_OMP_FOR(shared(ihomograph)) in ashift.c:3535.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_process(
    in_buf:         *const f32,
    out_buf:        *mut f32,
    out_width:      i32,
    out_height:     i32,
    roi_out_x:      f32,
    roi_out_y:      f32,
    roi_out_scale:  f32,
    in_width:       i32,
    in_height:      i32,
    roi_in_x:       f32,
    roi_in_y:       f32,
    roi_in_scale:   f32,
    cx:             f32,
    cy:             f32,
    ihomograph:     *const f32,
    interp_type:    u32,
) {
    if out_width <= 0 || out_height <= 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf, (in_width * in_height * 4) as usize);
    let outp = std::slice::from_raw_parts_mut(out_buf, (out_width * out_height * 4) as usize);
    let h: &[f32; 9] = &*(ihomograph as *const [f32; 9]);
    let ls = in_width * 4;

    for j in 0..out_height as usize {
        for i in 0..out_width as usize {
            let b = (j * out_width as usize + i) * 4;
            let coord = homographic_project(
                roi_out_x + i as f32, roi_out_y + j as f32,
                roi_out_scale, roi_in_scale, roi_in_x, roi_in_y, cx, cy, h,
            );
            match coord {
                Some((x, y)) if x >= 0.0 && y >= 0.0 && x < in_width as f32 && y < in_height as f32 => {
                    let px = compute_pixel4c(inp, x, y, in_width, in_height, ls, interp_type);
                    outp[b] = px[0]; outp[b+1] = px[1]; outp[b+2] = px[2]; outp[b+3] = px[3];
                }
                _ => { outp[b] = 0.0; outp[b+1] = 0.0; outp[b+2] = 0.0; outp[b+3] = 0.0; }
            }
        }
    }
}

/// FFI wrapper for `compute_pixel4c`.
///
/// `interp_type`: 0=bilinear, 1=bicubic, 2=lanczos2, 3=lanczos3
/// `linestride`:  floats per row (usually `width * 4`)
///
/// Matches dt_interpolation_compute_pixel4c() from src/common/interpolation.c.
#[no_mangle]
pub unsafe extern "C" fn darkroom_interpolate_pixel4c(
    in_buf:      *const f32,
    out:         *mut f32,
    x:           f32,
    y:           f32,
    img_width:   i32,
    img_height:  i32,
    linestride:  i32,
    interp_type: u32,
) {
    if x < 0.0 || y < 0.0 || x >= img_width as f32 || y >= img_height as f32 {
        let o = std::slice::from_raw_parts_mut(out, 4);
        o.fill(0.0);
        return;
    }
    let n = (img_height * linestride) as usize;
    let input = std::slice::from_raw_parts(in_buf, n);
    let result = compute_pixel4c(input, x, y, img_width, img_height, linestride, interp_type);
    let o = std::slice::from_raw_parts_mut(out, 4);
    o[0] = result[0];
    o[1] = result[1];
    o[2] = result[2];
    o[3] = result[3];
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: i32, h: i32, rgba: [f32; 4]) -> Vec<f32> {
        let mut v = vec![0.0f32; (w * h * 4) as usize];
        for px in 0..(w * h) as usize {
            v[px * 4..px * 4 + 4].copy_from_slice(&rgba);
        }
        v
    }

    #[test]
    fn bilinear_solid_returns_same_color() {
        let img = solid_image(8, 8, [0.5, 0.3, 0.7, 1.0]);
        let result = compute_pixel4c(&img, 3.5, 3.5, 8, 8, 32, 0);
        for (a, b) in result.iter().zip([0.5, 0.3, 0.7, 1.0].iter()) {
            assert!((a - b).abs() < 1e-5, "bilinear solid: {result:?}");
        }
    }

    #[test]
    fn bicubic_solid_returns_same_color() {
        let img = solid_image(10, 10, [0.2, 0.8, 0.4, 1.0]);
        let result = compute_pixel4c(&img, 4.7, 3.2, 10, 10, 40, 1);
        for (a, b) in result.iter().zip([0.2, 0.8, 0.4, 1.0].iter()) {
            assert!((a - b).abs() < 1e-4, "bicubic solid: {result:?}");
        }
    }

    #[test]
    fn lanczos2_solid_returns_same_color() {
        let img = solid_image(12, 12, [1.0, 0.0, 0.5, 1.0]);
        let result = compute_pixel4c(&img, 5.3, 5.7, 12, 12, 48, 2);
        for (a, b) in result.iter().zip([1.0, 0.0, 0.5, 1.0].iter()) {
            assert!((a - b).abs() < 1e-3, "lanczos2 solid: {result:?}");
        }
    }

    #[test]
    fn bilinear_at_integer_coord_returns_exact_pixel() {
        // 2×2 image: (0,0)=red, (1,0)=green, (0,1)=blue, (1,1)=white
        let img: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0,   // (0,0) red
            0.0, 1.0, 0.0, 1.0,   // (1,0) green
            0.0, 0.0, 1.0, 1.0,   // (0,1) blue
            1.0, 1.0, 1.0, 1.0,   // (1,1) white
        ];
        let r = compute_pixel4c(&img, 0.0, 0.0, 2, 2, 8, 0);
        assert!((r[0] - 1.0).abs() < 1e-4, "expected red R=1");
        assert!((r[1] - 0.0).abs() < 1e-4, "expected red G=0");
    }

    #[test]
    fn bilinear_midpoint_averages_corners() {
        let img: Vec<f32> = vec![
            0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 1.0,
        ];
        let r = compute_pixel4c(&img, 0.5, 0.5, 2, 2, 8, 0);
        assert!((r[0] - 0.5).abs() < 1e-4, "R midpoint={}", r[0]);
        assert!((r[1] - 0.5).abs() < 1e-4, "G midpoint={}", r[1]);
    }

    #[test]
    fn mirror_clamp_works() {
        assert_eq!(mirror(-1, 5), 1);
        assert_eq!(mirror(6,  5), 4);
        assert_eq!(mirror(3,  5), 3);
    }
}
