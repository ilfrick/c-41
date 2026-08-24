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

// ── Wavelets driver (denoiseprofile.c::process_wavelets) ─────────────────
//
// Port of darktable's profiled-denoise à-trous path. Scope decisions (all
// deliberate, documented for parity):
//
// - **Wavelets mode only** (the C default). The nlmeans patch-search modes and
//   the OpenCL paths are not ported.
// - **use_new_vst = TRUE** (the C default since 3.4) — the old plain-Anscombe
//   precondition/backtransform pair is not driven (the kernels above stay for
//   their FFI consumers).
// - **Generic Poissonian noise profile**: a = 1e-4, b = 0 per channel
//   (`dt_noiseprofile_generic` in noiseprofiles.c). We have no
//   noiseprofiles.json database wired, so there is no per-camera/ISO
//   interpolation; BayesShrink thresholds are data-driven anyway.
// - **wb = [1, 1, 1]**: the C derives VST white-balance factors from the
//   raw temperature coeffs (falling back to exactly these ones for
//   already-RGB input); our preview buffer arrives post-WB linear RGB.
// - **in_scale = 1**: we always process the whole (already downscaled)
//   buffer, never a zoomed ROI of a larger pipe, so the C's
//   roi->scale/iscale ratio is 1. The scale-count heuristic therefore uses
//   this buffer's own dimensions where the C uses the full-resolution ones —
//   identical behaviour whenever the processed image is the reference image,
//   which is the only case we ship.
// - **Flat force curve**: the C derives per-band/per-channel threshold
//   multipliers from a Catmull-Rom curve through the y params; its defaults
//   are flat 0.5, which makes every multiplier (0.5²·4 = 1.0) — i.e. adjt is
//   exactly 8.0. We ship only that default (no curve editor), so the 8.0 is
//   inlined with the derivation noted.

/// User-facing parameters of the wavelets denoise. Defaults mirror
/// `reload_defaults` in denoiseprofile.c (strength 1, shadows 1, bias 0,
/// Y0U0V0 colour mode).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveletsParams {
    /// Noise strength after equalisation (C slider 0.001..1000, default 1).
    pub strength: f32,
    /// Shadow preservation 0..1.8 (default 1) — shifts the power-law p.
    pub shadows: f32,
    /// Backtransform bias correction (C slider −1000..100, default 0).
    pub bias: f32,
    /// Colour mode: true = Y0U0V0 (C default), false = RGB.
    pub mode_y0u0v0: bool,
}

impl Default for WaveletsParams {
    fn default() -> Self {
        WaveletsParams { strength: 1.0, shadows: 1.0, bias: 0.0, mode_y0u0v0: true }
    }
}

/// `DT_IOP_DENOISE_PROFILE_BANDS` (denoiseprofile.c:59): hard cap on scales.
const BANDS: u32 = 7;
/// `DT_IOP_DENOISE_PROFILE_P_FULCRUM` (denoiseprofile.c:66).
const P_FULCRUM: f32 = 0.05;
/// `dt_noiseprofile_generic`'s green-channel a (noiseprofiles.c) — the one
/// channel the wavelets path reads (`d->a[1]`); b is 0 there too.
const GENERIC_A: f32 = 0.0001;

/// dt_colormatrix_t: row-major 4×4 (row 3 unused here).
type ColorMatrix = [[f32; 4]; 4];

/// Faithful port of `invert_matrix` (denoiseprofile.c:1108): 3×3 adjugate
/// inverse of the top-left block; `None` on zero determinant.
fn invert_matrix(m: &ColorMatrix) -> Option<ColorMatrix> {
    let biga = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let bigb = -m[1][0] * m[2][2] + m[1][2] * m[2][0];
    let bigc = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let bigd = -m[0][1] * m[2][2] + m[0][2] * m[2][1];
    let bige = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let bigf = -m[0][0] * m[2][1] + m[0][1] * m[2][0];
    let bigg = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let bigh = -m[0][0] * m[1][2] + m[0][2] * m[1][0];
    let bigi = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    let det = m[0][0] * biga + m[0][1] * bigb + m[0][2] * bigc;
    if det == 0.0 {
        return None;
    }
    let mut out = [[0.0f32; 4]; 4];
    out[0][0] = 1.0 / det * biga;
    out[0][1] = 1.0 / det * bigd;
    out[0][2] = 1.0 / det * bigg;
    out[1][0] = 1.0 / det * bigb;
    out[1][1] = 1.0 / det * bige;
    out[1][2] = 1.0 / det * bigh;
    out[2][0] = 1.0 / det * bigc;
    out[2][1] = 1.0 / det * bigf;
    out[2][2] = 1.0 / det * bigi;
    Some(out)
}

/// Faithful port of `set_up_conversion_matrices` (denoiseprofile.c:1146):
/// adapts the base Y0U0V0 matrix to the (unit) white balance so each output
/// channel has unit variance under Poissonian noise, then derives the RGB
/// inverse. Mutates `to_y0u0v0` in place like the C.
fn set_up_conversion_matrices(
    to_y0u0v0: &mut ColorMatrix,
    to_rgb: &mut ColorMatrix,
    wb: &[f32; 4],
) {
    // Weighted mean making SNR maximal; normalised to unit variance (√3).
    let mut sum_invwb = 1.0 / wb[0] + 1.0 / wb[1] + 1.0 / wb[2];
    sum_invwb *= 3.0f32.sqrt();
    for c in 0..3 {
        to_y0u0v0[0][c] = sum_invwb / wb[c];
    }
    to_y0u0v0[0][3] = 0.0;
    // U0/V0 rows keep their difference coefficients, rescaled to unit variance.
    let stddev_u0 =
        (0.5 * 0.5 * wb[0] * wb[0] + 0.5 * 0.5 * wb[2] * wb[2]).sqrt();
    let stddev_v0 = (0.25 * 0.25 * wb[0] * wb[0]
        + 0.5 * 0.5 * wb[1] * wb[1]
        + 0.25 * 0.25 * wb[2] * wb[2])
        .sqrt();
    for c in 0..3 {
        to_y0u0v0[1][c] /= stddev_u0;
        to_y0u0v0[2][c] /= stddev_v0;
    }
    to_y0u0v0[1][3] = 0.0;
    to_y0u0v0[2][3] = 0.0;
    match invert_matrix(to_y0u0v0) {
        Some(inv) => *to_rgb = inv,
        None => {
            // Standard-form fallback if the WB-adapted matrix is singular.
            let stddev_y0 =
                (1.0 / 9.0 * (wb[0] * wb[0] + wb[1] * wb[1] + wb[2] * wb[2])).sqrt();
            for c in 0..3 {
                to_y0u0v0[0][c] = 1.0 / (3.0 * stddev_y0);
            }
            to_y0u0v0[0][3] = 0.0;
            if let Some(inv) = invert_matrix(to_y0u0v0) {
                *to_rgb = inv;
            }
        }
    }
}

/// Flat 3×4 transposed copy for the FFI kernels: element [c*4 + r] holds
/// `m[r][c]` (see `mat3x4_mul`'s layout contract above).
#[inline]
fn transpose_flat12(m: &ColorMatrix) -> [f32; 12] {
    let mut t = [0.0f32; 12];
    for r in 0..3 {
        for c in 0..3 {
            t[c * 4 + r] = m[r][c];
        }
    }
    t
}

/// Profiled wavelet denoise — port of `process_wavelets` (denoiseprofile.c:1281).
/// Reads packed RGBA f32 `input` (width×height×4), writes the denoised result
/// into `output`. See the module-doc block above for scope deviations.
pub fn wavelets_denoise(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    p: &WaveletsParams,
) {
    // ── scale-count heuristic (mirrors the C's loop shape) ────────────────
    // Largest desired filter on this buffer: hard cap 257 taps, else 20% of
    // the longest side.
    let supp0 = (2.0 * (2u32 << (BANDS - 1)) as f32 + 1.0)
        .min(width.max(height) as f32 * 0.2);
    let i0 = ((supp0 - 1.0) * 0.5).log2();
    let mut max_scale: u32 = 0;
    while max_scale < BANDS {
        let supp = 2.0 * (2u32 << max_scale) as f32 + 1.0;
        // in_scale == 1 ⇒ filter size on the reference image is supp itself.
        let i_in = ((supp - 1.0) * 0.5).log2() - 1.0;
        if 1.0 - (i_in + 0.5) / i0 < 0.0 {
            break;
        }
        max_scale += 1;
    }

    let npixels = width * height;
    let n = npixels * 4;

    // max_scale can legitimately end at 0 on degenerate sizes (supp0 < ~2.4
    // ⇒ i0 < 0.5 rejects even scale 0); the C then shifts by −1 (UB). We pass
    // through instead — nothing sensible to smooth at that size anyway.
    if max_scale == 0 || n == 0 {
        output.copy_from_slice(input);
        return;
    }

    let max_mult = 1usize << (max_scale - 1);
    // Corner case of an extremely small image: would read out of bounds.
    if width < 2 * max_mult || height < 2 * max_mult {
        output.copy_from_slice(input);
        return;
    }

    // ── VST parameterisation ──────────────────────────────────────────────
    // wb = [1,1,1], in_scale = 1 ⇒ p collapses to `shadows` and the bias
    // correction's log term vanishes (documented deviations).
    let wb = [1.0f32; 4];
    let p_vec = [
        (p.shadows + 0.1 * (1.0f32 / wb[0]).ln()).max(0.0),
        (p.shadows + 0.1 * (1.0f32 / wb[1]).ln()).max(0.0),
        (p.shadows + 0.1 * (1.0f32 / wb[2]).ln()).max(0.0),
        0.0,
    ];
    let compensate_p = P_FULCRUM / P_FULCRUM.powf(p.shadows);
    let a_arg = GENERIC_A * compensate_p;
    let b_arg = 0.0f32;

    // Base Y0U0V0 ("secrets of image denoising cuisine") conversion.
    let mut to_y0u0v0: ColorMatrix = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        [0.5, 0.0, -0.5, 0.0],
        [0.25, -0.5, 0.25, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let mut to_rgb_m: ColorMatrix = [[0.0; 4]; 4];
    set_up_conversion_matrices(&mut to_y0u0v0, &mut to_rgb_m, &wb);

    // More strength in Y0U0V0 for similar smoothing as RGB mode.
    let compensate_strength = if p.mode_y0u0v0 { 2.5 } else { 1.0 };
    let s = p.strength * compensate_strength; // × in_scale (=1)
    for k in 0..3 {
        for c in 0..3 {
            to_y0u0v0[k][c] /= s;
            to_rgb_m[k][c] *= s;
        }
    }
    let to_yuv_t = transpose_flat12(&to_y0u0v0);
    let to_rgb_t = transpose_flat12(&to_rgb_m);

    // The C then scales wb by the same factor (denoiseprofile.c:1385). That
    // scaled wb is the *only* strength carrier in RGB mode — the v2 kernels
    // take no colour matrix, so without it Strength would do nothing there —
    // and in Y0U0V0 mode it scales the backtransform's bias term
    // (bias_wb[c] = bias·wb[c]). Note the matrices above are deliberately
    // built from the *unscaled* wb (C builds them before this line too).
    let wb_s = [s, s, s, 1.0];

    // ── variance-stabilising transform of the whole frame ─────────────────
    let mut precond = vec![0.0f32; n];
    unsafe {
        if !p.mode_y0u0v0 {
            darkroom_denoise_precondition_v2(
                input.as_ptr(), precond.as_mut_ptr(), npixels,
                a_arg, p_vec.as_ptr(), b_arg, wb_s.as_ptr(),
            );
        } else {
            darkroom_denoise_precondition_yuv(
                input.as_ptr(), precond.as_mut_ptr(), npixels,
                a_arg, p_vec.as_ptr(), b_arg, to_yuv_t.as_ptr(),
            );
        }
    }

    // ── à-trous scale loop: decompose → BayesShrink → soft-threshold ──────
    let varf = (2.0f32 + 2.0 * 4.0 * 4.0 + 6.0 * 6.0).sqrt() / 16.0;
    let mut buf1 = precond;
    let mut buf2 = vec![0.0f32; n];
    let mut detail = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];
    let boost = [1.0f32; 4];

    for scale in 0..max_scale {
        let sigma_band = varf.powi(scale as i32);
        let sb2 = sigma_band * sigma_band;
        let sum_y2 = crate::iop::eaw::dn_decompose(
            &mut buf2, &buf1, &mut detail, scale, 1.0 / sb2, width, height,
        );

        // BayesShrink: thrs = adjt·σ_band²/√(var(detail)−σ_band²), clamped at
        // 1e-6 inside the sqrt. adjt = 8.0 under the flat default force curve
        // (derivation in the module doc above).
        let npf = npixels as f32;
        let mut thrs = [0.0f32; 4];
        for c in 0..3 {
            let std_x = (f32::max(1e-6, sum_y2[c] / (npf - 1.0) - sb2)).sqrt();
            thrs[c] = 8.0 * sb2 / std_x;
        }

        eaw::synthesize(&mut out, &detail, &thrs, &boost, npixels);

        std::mem::swap(&mut buf1, &mut buf2);
    }

    // Add in the final residue.
    for k in 0..n {
        out[k] += buf1[k];
    }

    // ── inverse VST back to linear RGB ────────────────────────────────────
    unsafe {
        if !p.mode_y0u0v0 {
            darkroom_denoise_backtransform_v2(
                out.as_mut_ptr(), npixels,
                a_arg, p_vec.as_ptr(), b_arg, p.bias, wb_s.as_ptr(),
            );
        } else {
            darkroom_denoise_backtransform_yuv(
                out.as_mut_ptr(), npixels,
                a_arg, p_vec.as_ptr(), b_arg, p.bias, wb_s.as_ptr(), to_rgb_t.as_ptr(),
            );
        }
    }

    // The VST kernels run over all four lanes; lane 3 comes back as
    // computational garbage (darktable's callers never read alpha either, but
    // our pipeline contract carries the source alpha through). Restore it.
    for k in (3..n).step_by(4) {
        out[k] = input[k];
    }

    output.copy_from_slice(&out);
}

use crate::iop::eaw;

#[cfg(test)]
mod wavelets_tests {
    use super::*;

    #[test]
    fn conversion_matrices_are_inverses() {
        // set_up_conversion_matrices must produce a Y0U0V0 whose inverse is
        // toRGB — for the unit wb we ship, and the strength scaling cancels
        // (rows divided by s, inverse multiplied by s).
        let wb = [1.0f32; 4];
        let mut m: ColorMatrix = [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, -0.5, 0.0],
            [0.25, -0.5, 0.25, 0.0],
            [0.0; 4],
        ];
        let mut inv = [[0.0f32; 4]; 4];
        set_up_conversion_matrices(&mut m, &mut inv, &wb);
        let s = 2.5f32; // compensate_strength × default strength
        for k in 0..3 {
            for c in 0..3 {
                m[k][c] /= s;
                inv[k][c] *= s;
            }
        }
        for r in 0..3 {
            for c in 0..3 {
                let dot = m[r][0] * inv[0][c]
                    + m[r][1] * inv[1][c]
                    + m[r][2] * inv[2][c];
                assert!(
                    (dot - (if r == c { 1.0 } else { 0.0 })).abs() < 1e-5,
                    "M·M⁻¹[{r}][{c}] = {dot}"
                );
            }
        }
    }

    #[test]
    fn wavelets_flat_field_is_invariant_in_both_modes() {
        // A noise-free flat field has zero detail at every scale, so the
        // whole pipeline reduces to VST → constant → inverse VST, which is an
        // exact pair (the v2 backtransform solves the forward's quadratic).
        for mode in [true, false] {
            let (w, h) = (48usize, 40usize);
            let input = vec![0.42f32; w * h * 4];
            let mut out = vec![0.0f32; w * h * 4];
            wavelets_denoise(
                &input, &mut out, w, h,
                &WaveletsParams { mode_y0u0v0: mode, ..Default::default() },
            );
            for k in 0..input.len() {
                assert!(
                    (out[k] - input[k]).abs() < 1e-5,
                    "flat field moved in mode {mode} at {k}: {} vs {}",
                    out[k],
                    input[k]
                );
            }
        }
    }

    #[test]
    fn wavelets_small_image_is_passthrough() {
        // Below 2·max_mult pixels per side the C memcpy's the input rather
        // than risk out-of-bounds reads — pin that passthrough.
        let (w, h) = (8usize, 8usize);
        let mut input = vec![0.0f32; w * h * 4];
        for (k, v) in input.iter_mut().enumerate() {
            *v = (k % 7) as f32 * 0.13 + 0.05;
        }
        let mut out = vec![0.0f32; w * h * 4];
        wavelets_denoise(&input, &mut out, w, h, &WaveletsParams::default());
        assert_eq!(out, input, "tiny image must pass through untouched");
    }

    #[test]
    fn wavelets_reduces_noise_variance_chroma_hard_luma_gently() {
        // Inject noise whose variance matches the generic Poissonian profile
        // (var = a·E[x] ⇒ σ ≈ sqrt(1e-4 · 0.2)) into a flat field and check
        // the two signatures of the Y0U0V0 mode at the C's default settings:
        //
        // - **chroma detail is wiped**: U0/V0 rows carry small variance, so
        //   their BayesShrink threshold clamps at the 1e-6 floor and every
        //   chroma detail is soft-thresholded away — R−G noise collapses.
        // - **luma is reduced but conservatively**: with the flat force curve
        //   the luma row ends up over-normalised (norm ≈ 3.6 after the
        //   strength division), so its threshold lands around half a σ —
        //   darktable's own "a little weak" default. Assert a real but
        //   moderate cut, not a wipe.
        let (w, h) = (96usize, 72usize);
        const FLAT: f32 = 0.2;
        let mut input = vec![0.0f32; w * h * 4];
        // Deterministic LCG, roughly gaussian via 4-uniform sum.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u: f32 = ((state >> 40) as f32) / 16777216.0;
            u - 0.5
        };
        let sigma = (GENERIC_A * FLAT).sqrt();
        // Independent noise per channel so the field carries a real chroma
        // component (identical-per-channel noise is pure luma).
        for k in (0..input.len()).step_by(4) {
            for c in 0..3 {
                let n: f32 =
                    (noise() + noise() + noise() + noise()) * 2.0 * sigma;
                input[k + c] = FLAT + n;
            }
            input[k + 3] = 1.0;
        }

        let std_of = |buf: &[f32], f: &dyn Fn([f32; 3]) -> f32| {
            let vals: Vec<f32> =
                buf.chunks_exact(4).map(|p| f([p[0], p[1], p[2]])).collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32)
                .sqrt()
        };
        let lum = |p: [f32; 3]| (p[0] + p[1] + p[2]) / 3.0;
        let chroma = |p: [f32; 3]| p[0] - p[1];

        let mut out = vec![0.0f32; w * h * 4];
        wavelets_denoise(&input, &mut out, w, h, &WaveletsParams::default());

        let in_luma_std = std_of(&input, &lum);
        let out_luma_std = std_of(&out, &lum);
        let in_chroma_std = std_of(&input, &chroma);
        let out_chroma_std = std_of(&out, &chroma);

        assert!(
            out_luma_std < 0.85 * in_luma_std,
            "luma noise must drop measurably: {out_luma_std} vs {in_luma_std}"
        );
        assert!(
            out_luma_std > 0.3 * in_luma_std,
            "luma reduction blew past darktable's conservative default: {out_luma_std} vs {in_luma_std}"
        );
        assert!(
            out_chroma_std < 0.35 * in_chroma_std,
            "chroma noise must be nearly wiped: {out_chroma_std} vs {in_chroma_std}"
        );
        // Alpha passes through untouched.
        assert!(out.chunks_exact(4).all(|p| p[3] == 1.0));
        // …and the mean must not drift much (bias-corrected backtransform).
        let in_mean = input.chunks_exact(4).map(|p| p[0]).sum::<f32>() / (w * h) as f32;
        let out_mean = out.chunks_exact(4).map(|p| p[0]).sum::<f32>() / (w * h) as f32;
        assert!((in_mean - out_mean).abs() < sigma, "mean drifted");
    }

    #[test]
    fn wavelets_strength_scaled_wb_preserves_exact_inverse_on_flat_fields() {
        // Regression (senior review, m4-120): in RGB mode the v2 VST kernels
        // take no colour matrix, so `strength` reaches them only through the
        // strength-scaled wb vector (denoiseprofile.c:1385). Whatever the
        // carrier, the forward/backward pair must keep cancelling exactly —
        // here at off-default strengths in BOTH modes, which is what exercises
        // the RGB wb_s path (the Y0U0V0 path cancels via the ÷s/×s matrices).
        //
        // Note strength does NOT otherwise modulate these runs: with the
        // generic a=1e-4 profile the transformed-space noise variance sits far
        // below σ_band² (=1 at scale 0), so the BayesShrink denominator clamps
        // at its 1e-6 floor and thresholds saturate for any strength — detail
        // is wiped wholesale and the result is the coarse pyramid alone.
        // Identical in the C with the same profile; strength only bites once a
        // real per-camera profile stabilises variance towards 1.
        for mode in [true, false] {
            for strength in [0.4f32, 2.5] {
                let (w, h) = (48usize, 40usize);
                let input = vec![0.42f32; w * h * 4];
                let mut out = vec![0.0f32; w * h * 4];
                wavelets_denoise(
                    &input, &mut out, w, h,
                    &WaveletsParams { strength, mode_y0u0v0: mode, ..Default::default() },
                );
                for k in 0..input.len() {
                    assert!(
                        (out[k] - input[k]).abs() < 1e-5,
                        "flat field moved (mode {mode}, strength {strength}) at {k}: \
                         {} vs {}",
                        out[k],
                        input[k]
                    );
                }
            }
        }
    }

    #[test]
    fn wavelets_bias_is_scaled_by_strength_through_wb() {
        // Second half of the same finding: the backtransform's bias term is
        // bias·wb[c] with wb scaled by strength (backtransform_Y0U0V0's
        // bias_wb, C :1453; same in backtransform_v2). Under the pre-fix unit
        // wb, a non-zero bias produced a strength-INDEPENDENT shift — the two
        // means below came out equal. The scaled wb makes the shift grow with
        // strength.
        let (w, h) = (96usize, 72usize);
        const FLAT: f32 = 0.2;
        let sigma = (GENERIC_A * FLAT).sqrt();
        let mut input = vec![0.0f32; w * h * 4];
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u: f32 = ((state >> 40) as f32) / 16777216.0;
            u - 0.5
        };
        for k in (0..input.len()).step_by(4) {
            for c in 0..3 {
                let n: f32 =
                    (noise() + noise() + noise() + noise()) * 2.0 * sigma;
                input[k + c] = FLAT + n;
            }
            input[k + 3] = 1.0;
        }

        let mean_for = |strength: f32| {
            let mut out = vec![0.0f32; w * h * 4];
            wavelets_denoise(
                &input, &mut out, w, h,
                &WaveletsParams {
                    strength,
                    bias: 0.6,
                    mode_y0u0v0: false,
                    ..Default::default()
                },
            );
            out.chunks_exact(4).map(|p| p[0]).sum::<f32>() / (w * h) as f32
        };

        let low = mean_for(0.5);
        let high = mean_for(3.0);
        // Measured drift ≈ 1.9e-5 at these settings (second-order in bias·wb):
        // orders above the ~1e-9 mean-quantisation floor, far below any
        // physical level shift. Pre-fix the two means were exactly equal.
        let drift = (high - low).abs();
        assert!(
            drift > 4e-6 && drift < 1e-3,
            "bias shift must scale with strength via wb: {high} vs {low}"
        );
        // …and stay bounded: the bias correction may not wreck the level.
        assert!((low - FLAT).abs() < 4.0 * sigma && (high - FLAT).abs() < 4.0 * sigma);
    }
}
