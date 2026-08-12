//! Denoiseprofile IOP -- Anscombe/power-law variance-stabilising transforms.
//!
//! All 6 DT_OMP_FOR loops in denoiseprofile.c are pure per-pixel math with
//! no ICC/work_profile dependency.

use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Denoiseprofile;
impl IopProcess for Denoiseprofile {
    fn name(&self) -> &'static str { "denoiseprofile" }
    fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("denoiseprofile: use C FFI path".into()))
    }
    fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("denoiseprofile: no OpenCL path".into()))
    }
}

// ── Row-major 3x4 transposed matrix multiply (for Y0U0V0 paths) ──────────

#[inline(always)]
fn mat3x4_mul(m: &[f32; 12], v: &[f32]) -> [f32; 4] {
    // dt_apply_transposed_color_matrix: out[r] = sum_c m[c*4+r] * v[c]
    let mut out = [0.0f32; 4];
    for r in 0..4usize {
        out[r] = m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2];
    }
    out
}

// ── Anscombe VST (forward / inverse) ─────────────────────────────────────

/// Forward Anscombe VST: buf[c] = 2*sqrt(max(0, in[c]/a[c] + (b[c]/a[c])^2 + 3/8)).
/// Matches denoiseprofile.c::precondition DT_OMP_FOR at line 924.
#[no_mangle]
pub unsafe extern "C" fn darkroom_denoise_precondition(
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
    a:       *const f32,  // 4 floats
    b:       *const f32,  // 4 floats
) {
    let inp = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let a = std::slice::from_raw_parts(a, 4);
    let b = std::slice::from_raw_parts(b, 4);
    let s: [f32; 4] = std::array::from_fn(|c|
        if c < 3 { (b[c]/a[c]).powi(2) + 3.0/8.0 } else { 0.0 }
    );
    for j in (0..npixels * 4).step_by(4) {
        for c in 0..4 {
            let d = (inp[j+c] / a[c] + s[c]).max(0.0);
            out[j+c] = 2.0 * d.sqrt();
        }
    }
}

/// Inverse Anscombe VST (closed-form low-bias approximation).
/// Matches denoiseprofile.c::backtransform DT_OMP_FOR at line 949.
#[no_mangle]
pub unsafe extern "C" fn darkroom_denoise_backtransform(
    buf:     *mut f32,
    npixels: usize,
    a:       *const f32,
    b:       *const f32,
) {
    let buf = std::slice::from_raw_parts_mut(buf, npixels * 4);
    let a = std::slice::from_raw_parts(a, 4);
    let b = std::slice::from_raw_parts(b, 4);
    let s: [f32; 4] = std::array::from_fn(|c|
        if c < 3 { (b[c]/a[c]).powi(2) + 1.0/8.0 } else { 0.0 }
    );
    let sqrt32 = (3.0_f32 / 2.0).sqrt();
    for j in (0..npixels * 4).step_by(4) {
        for c in 0..4 {
            let x = buf[j+c];
            let x2 = x * x;
            buf[j+c] = if x < 0.5 { 0.0 } else {
                a[c] * (0.25*x2 + 0.25*sqrt32/x - 11.0/8.0/x2
                        + 5.0/8.0*sqrt32/(x*x2) - s[c])
            };
        }
    }
}

// ── Generalized (v2) power-law VST ───────────────────────────────────────

/// Forward generalized VST: buf[c] = 2*(max(in[c]/wb[c]+b,0))^(1-p[c]/2) / ((2-p[c])*sqrt(a)).
/// Matches denoiseprofile.c::precondition_v2 DT_OMP_FOR at line 1005.
#[no_mangle]
pub unsafe extern "C" fn darkroom_denoise_precondition_v2(
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
    a: f32, p: *const f32, b: f32, wb: *const f32,
) {
    let inp = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let p  = std::slice::from_raw_parts(p,  4);
    let wb = std::slice::from_raw_parts(wb, 4);
    let exp: [f32; 4] = std::array::from_fn(|c| -p[c]/2.0 + 1.0);
    let den: [f32; 4] = std::array::from_fn(|c|
        if c < 3 { (-p[c]+2.0) * a.sqrt() } else { 1.0 });
    for j in (0..npixels * 4).step_by(4) {
        for c in 0..4 {
            let scaled = (inp[j+c] / wb[c] + b).max(0.0);
            out[j+c] = 2.0 * scaled.powf(exp[c]) / den[c];
        }
    }
}

/// Inverse generalized VST (quadratic formula + power-law).
/// Matches denoiseprofile.c::backtransform_v2 DT_OMP_FOR at line 1103.
#[no_mangle]
pub unsafe extern "C" fn darkroom_denoise_backtransform_v2(
    buf:     *mut f32,
    npixels: usize,
    a: f32, p: *const f32, b: f32, bias: f32, wb: *const f32,
) {
    let buf = std::slice::from_raw_parts_mut(buf, npixels * 4);
    let p  = std::slice::from_raw_parts(p,  4);
    let wb = std::slice::from_raw_parts(wb, 4);
    let exp: [f32; 4] = std::array::from_fn(|c| 1.0/(1.0 - p[c]/2.0));
    let den: [f32; 4] = std::array::from_fn(|c|
        if c < 3 { 4.0 / (a.sqrt() * (2.0 - p[c])) } else { 1.0 });
    for j in (0..npixels * 4).step_by(4) {
        let mut z1 = [0.0f32; 4];
        for c in 0..4 {
            let x = buf[j+c].max(0.0);
            let delta = (x*x + bias).max(0.0);
            z1[c] = (x + delta.sqrt()) / den[c];
        }
        for c in 0..4 {
            buf[j+c] = wb[c] * (z1[c].powf(exp[c]) - b);
        }
    }
}

// ── Y0U0V0 color-space variants ───────────────────────────────────────────

/// Forward VST in Y0U0V0: power-law + color-matrix rotation.
/// toY0U0V0_trans: 12-float row-major 3x4 transposed matrix.
/// Matches denoiseprofile.c::precondition_Y0U0V0 DT_OMP_FOR at line 1139.
#[no_mangle]
pub unsafe extern "C" fn darkroom_denoise_precondition_yuv(
    in_buf:    *const f32,
    out_buf:   *mut f32,
    npixels:   usize,
    a: f32, p: *const f32, b: f32,
    to_yuv:    *const f32,  // 12 floats 3x4 transposed
) {
    let inp    = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out    = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let p      = std::slice::from_raw_parts(p, 4);
    let m: &[f32; 12] = &*(to_yuv as *const [f32; 12]);
    let exp: [f32; 4] = std::array::from_fn(|c| -p[c]/2.0 + 1.0);
    let scale: [f32; 4] = std::array::from_fn(|c|
        if c < 3 { 2.0 / ((-p[c]+2.0) * a.sqrt()) } else { 1.0 });
    for j in (0..npixels * 4).step_by(4) {
        let mut tmp = [0.0f32; 4];
        for c in 0..4 {
            tmp[c] = (inp[j+c] + b).max(0.0).powf(exp[c]) * scale[c];
        }
        let yuv = mat3x4_mul(m, &tmp);
        for c in 0..4 { out[j+c] = yuv[c]; }
    }
}

/// Inverse VST from Y0U0V0: inverse color-matrix + inverse power-law.
/// toRGB_trans: 12-float row-major 3x4 transposed matrix.
/// Matches denoiseprofile.c::backtransform_Y0U0V0 DT_OMP_FOR at line 1182.
#[no_mangle]
pub unsafe extern "C" fn darkroom_denoise_backtransform_yuv(
    buf:      *mut f32,
    npixels:  usize,
    a: f32, p: *const f32, b: f32, bias: f32, wb: *const f32,
    to_rgb:   *const f32,  // 12 floats 3x4 transposed
) {
    let buf    = std::slice::from_raw_parts_mut(buf, npixels * 4);
    let p      = std::slice::from_raw_parts(p, 4);
    let wb     = std::slice::from_raw_parts(wb, 4);
    let m: &[f32; 12] = &*(to_rgb as *const [f32; 12]);
    let exp: [f32; 4] = std::array::from_fn(|c| 1.0/(1.0 - p[c]/2.0));
    let scale: [f32; 4] = std::array::from_fn(|c|
        if c < 3 { (a.sqrt() * (2.0 - p[c])) / 4.0 } else { 1.0 });
    let bias_wb: [f32; 4] = std::array::from_fn(|c|
        if c < 3 { bias * wb[c] } else { 0.0 });
    for j in (0..npixels * 4).step_by(4) {
        let rgb = mat3x4_mul(m, &buf[j..j+4]);
        let mut z1 = [0.0f32; 4];
        for c in 0..4 {
            let x = rgb[c].max(0.0);
            let delta = (x*x + bias_wb[c]).max(0.0);
            z1[c] = (x + delta.sqrt()) * scale[c];
        }
        for c in 0..4 {
            buf[j+c] = z1[c].powf(exp[c]) - b;
        }
    }
}
