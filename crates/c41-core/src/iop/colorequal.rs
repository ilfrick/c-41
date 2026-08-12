use crate::{color, params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Colorequal;

impl IopProcess for Colorequal {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colorequal" }
}

/// Initialise the per-pixel UV covariance matrix from raw UV values.
///
///   cov[k*4 + 0] = U*U
///   cov[k*4 + 1] = cov[k*4 + 2] = U*V
///   cov[k*4 + 3] = V*V
///
/// Matches `_init_covariance()` in src/iop/colorequal.c:482.
/// `uv_buf` is `pixels * 2` floats (interleaved U, V per pixel).
/// `cov_buf` must be `pixels * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_init_covariance(
    uv_buf: *const f32,
    cov_buf: *mut f32,
    pixels: usize,
) {
    if pixels == 0 { return; }
    let uv  = std::slice::from_raw_parts(uv_buf,  pixels * 2);
    let cov = std::slice::from_raw_parts_mut(cov_buf, pixels * 4);
    for k in 0..pixels {
        let u = uv[2 * k];
        let v = uv[2 * k + 1];
        cov[4 * k]     = u * u;
        cov[4 * k + 1] = u * v;
        cov[4 * k + 2] = u * v;
        cov[4 * k + 3] = v * v;
    }
}

/// Finalise the covariance matrix by subtracting avg(x)*avg(y).
///
///   cov[k*4 + 0] -= U*U
///   cov[k*4 + 1] -= U*V
///   cov[k*4 + 2] -= U*V
///   cov[k*4 + 3] -= V*V
///
/// Matches `_finish_covariance()` in src/iop/colorequal.c:502.
/// `uv_buf` here contains the **blurred** averages of U and V (the output
/// of the guided-filter blur step performed by the caller before this call).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_finish_covariance(
    uv_buf: *const f32,
    cov_buf: *mut f32,
    pixels: usize,
) {
    if pixels == 0 { return; }
    let uv  = std::slice::from_raw_parts(uv_buf,  pixels * 2);
    let cov = std::slice::from_raw_parts_mut(cov_buf, pixels * 4);
    for k in 0..pixels {
        let u = uv[2 * k];
        let v = uv[2 * k + 1];
        cov[4 * k]     -= u * u;
        cov[4 * k + 1] -= u * v;
        cov[4 * k + 2] -= u * v;
        cov[4 * k + 3] -= v * v;
    }
}

/// Compute the per-pixel guided-filter regression coefficients (a, b) for
/// the 2D UV space.
///
/// For each pixel k:
///   Σ = cov + ε * I₂     (2×2 regularised covariance)
///   if |det(Σ)| > 4*FLT_EPSILON:
///     Σ⁻¹ computed analytically
///     a[k*4 .. k*4+4] = 2×2 regression matrix from cov and Σ⁻¹
///   else:
///     a[k*4 .. k*4+4] = 0
///   b[k*2 + 0] = U - a[k*4+0]*U - a[k*4+1]*V
///   b[k*2 + 1] = V - a[k*4+2]*U - a[k*4+3]*V
///
/// Matches `_prepare_prefilter()` in src/iop/colorequal.c:523.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_prepare_prefilter(
    uv_buf: *const f32,
    cov_buf: *const f32,
    a_buf: *mut f32,
    b_buf: *mut f32,
    pixels: usize,
    eps: f32,
) {
    if pixels == 0 { return; }
    let uv  = std::slice::from_raw_parts(uv_buf,  pixels * 2);
    let cov = std::slice::from_raw_parts(cov_buf, pixels * 4);
    let a   = std::slice::from_raw_parts_mut(a_buf, pixels * 4);
    let b   = std::slice::from_raw_parts_mut(b_buf, pixels * 2);

    for k in 0..pixels {
        let sigma = [
            cov[4 * k]     + eps,
            cov[4 * k + 1],
            cov[4 * k + 2],
            cov[4 * k + 3] + eps,
        ];
        let det = sigma[0] * sigma[3] - sigma[1] * sigma[2];

        if det.abs() > 4.0 * f32::EPSILON {
            let sigma_inv = [
                 sigma[3] / det,
                -sigma[1] / det,
                -sigma[2] / det,
                 sigma[0] / det,
            ];
            a[4 * k]     = cov[4 * k]     * sigma_inv[0] + cov[4 * k + 1] * sigma_inv[1];
            a[4 * k + 1] = cov[4 * k]     * sigma_inv[2] + cov[4 * k + 1] * sigma_inv[3];
            a[4 * k + 2] = cov[4 * k + 2] * sigma_inv[0] + cov[4 * k + 3] * sigma_inv[1];
            a[4 * k + 3] = cov[4 * k + 2] * sigma_inv[2] + cov[4 * k + 3] * sigma_inv[3];
        } else {
            a[4 * k] = 0.0; a[4 * k + 1] = 0.0;
            a[4 * k + 2] = 0.0; a[4 * k + 3] = 0.0;
        }

        let u = uv[2 * k];
        let v = uv[2 * k + 1];
        b[2 * k]     = u - a[4 * k]     * u - a[4 * k + 1] * v;
        b[2 * k + 1] = v - a[4 * k + 2] * u - a[4 * k + 3] * v;
    }
}

/// Linearly-interpolated lookup in the precomputed sigmoid saturation-weight
/// table. Mirrors `_get_satweight()` in colorequal.c:461.
///
/// `satweights` has `2 * satsize + 1` entries, initialised by
/// `_init_satweights(contrast)` in C. The argument `sat` is the raw
/// difference `saturation[k] - sat_shift`; values outside `[-1, 1)` are
/// clamped before indexing.
#[inline(always)]
fn get_satweight(sat: f32, satweights: &[f32], satsize: usize) -> f32 {
    // CLAMP(sat, -1, 1 - 1/SATSIZE) then map to [0, 2*SATSIZE]
    let sat_clamp = sat.clamp(-1.0, 1.0 - (1.0 / satsize as f32));
    let isat = satsize as f32 * (1.0 + sat_clamp);
    let base = isat.floor();
    let i = base as usize;
    satweights[i] + (isat - base) * (satweights[i + 1] - satweights[i])
}

/// Apply the guided-filter regression to correct UV, blending with the
/// original based on a sigmoid saturation weight.
///
/// For each pixel k:
///   u_corr = a[k*4+0]*U + a[k*4+1]*V + b[k*2+0]
///   v_corr = a[k*4+2]*U + a[k*4+3]*V + b[k*2+1]
///   w = get_satweight(saturation[k] - sat_shift, satweights, satsize)
///   UV[k*2+0] = U + w * (u_corr - U)      ← lerp toward correction
///   UV[k*2+1] = V + w * (v_corr - V)
///
/// `satweights` is the precomputed logistic table (length `2*satsize+1`),
/// filled by `_init_satweights(contrast)` in C. The Rust port does not
/// recompute it; the caller passes the live C static array pointer.
///
/// Matches `_apply_prefilter()` in src/iop/colorequal.c:573.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_apply_prefilter(
    uv_buf: *mut f32,
    saturation: *const f32,
    a_buf: *const f32,
    b_buf: *const f32,
    npixels: usize,
    sat_shift: f32,
    satweights: *const f32,
    satsize: usize,
) {
    if npixels == 0 || satsize == 0 { return; }
    let uv  = std::slice::from_raw_parts_mut(uv_buf, npixels * 2);
    let sat = std::slice::from_raw_parts(saturation, npixels);
    let a   = std::slice::from_raw_parts(a_buf, npixels * 4);
    let b   = std::slice::from_raw_parts(b_buf, npixels * 2);
    let sw  = std::slice::from_raw_parts(satweights, 2 * satsize + 1);

    for k in 0..npixels {
        let u = uv[2 * k];
        let v = uv[2 * k + 1];
        let u_corr = a[4 * k]     * u + a[4 * k + 1] * v + b[2 * k];
        let v_corr = a[4 * k + 2] * u + a[4 * k + 3] * v + b[2 * k + 1];
        let w = get_satweight(sat[k] - sat_shift, sw, satsize);
        uv[2 * k]     = u + w * (u_corr - u);
        uv[2 * k + 1] = v + w * (v_corr - v);
    }
}

// ── Gamut LUT helpers (from darktable_ucs_22_helpers.h) ──────────────────────

const LUT_ELEM: usize = 512;

/// Linear interpolation in a hue-ring LUT.
/// `hue` is in [-π, π]. Matches `lookup_gamut()` in darktable_ucs_22_helpers.h:132.
#[inline(always)]
pub fn lookup_gamut(lut: &[f32; LUT_ELEM], hue: f32) -> f32 {
    use std::f32::consts::PI;
    let x_test = LUT_ELEM as f32 * (hue + PI) / (2.0 * PI);
    let x_prev = x_test.floor();
    let xi  = (x_prev as i32) & (LUT_ELEM as i32 - 1);
    let xii = (x_prev as i32 + 1) & (LUT_ELEM as i32 - 1);
    let y_prev = lut[xi as usize];
    if xi != xii {
        y_prev + (x_test - x_prev) * (lut[xii as usize] - y_prev)
    } else {
        y_prev
    }
}

/// Soft-clip HSB saturation inside the gamut boundary.
/// Matches `gamut_map_HSB()` in darktable_ucs_22_helpers.h:167.
#[inline(always)]
pub fn gamut_map_hsb(hsb: &mut [f32; 4], gamut_lut: &[f32; LUT_ELEM], l_white: f32) {
    let jch = color::dt_ucs_hsb_to_jch(hsb);
    let max_colorfulness = lookup_gamut(gamut_lut, jch[2]);
    let max_chroma = 15.932993652962535
        * (jch[0] * l_white).powf(0.6523997524738018)
        * max_colorfulness.powf(0.6007557017508491)
        / l_white;
    let jch_boundary = [jch[0], max_chroma, jch[2], 0.0];
    let hsb_boundary = color::dt_ucs_jch_to_hsb(&jch_boundary);
    // soft_clip(x, 0.8*upper, upper)
    let upper = hsb_boundary[1];
    let soft  = 0.8 * upper;
    let norm  = upper - soft;
    hsb[1] = if hsb[1] > soft && norm > 0.0 {
        soft + (1.0 - (-(hsb[1] - soft) / norm).exp()) * norm
    } else {
        hsb[1]
    };
}

/// Scharr gradient magnitude at pixel `p` (stride `w`).
/// Matches `scharr_gradient()` in common/math.h:405.
#[inline(always)]
pub fn scharr_gradient(p: &[f32], w: usize, idx: usize) -> f32 {
    let gx = (47.0 / 255.0) * (p[idx-w-1] - p[idx-w+1] + p[idx+w-1] - p[idx+w+1])
           + (162.0 / 255.0) * (p[idx-1] - p[idx+1]);
    let gy = (47.0 / 255.0) * (p[idx-w-1] - p[idx+w-1] + p[idx-w+1] - p[idx+w+1])
           + (162.0 / 255.0) * (p[idx-w] - p[idx+w]);
    (gx * gx + gy * gy).sqrt()
}

// ── Step 3: finish conversion to HSB + compute corrections ───────────────────

/// dt UCS HSB → XYZ D65 (quick path via JCH → xyY → XYZ).
#[inline(always)]
pub fn dt_ucs_hsb_to_xyz(hsb: &[f32; 4], l_white: f32) -> [f32; 4] {
    let jch = color::dt_ucs_hsb_to_jch(hsb);
    let xyy = color::dt_ucs_jch_to_xyy(&jch, l_white);
    color::xyy_to_xyz(&xyy)
}

/// STEP 3: convert L*/UV → JCH → HSB and compute hue/sat/brightness corrections.
///
/// For each pixel k:
///   JCH = dt_UCS_LUV_to_JCH(Lscharr[k], white, UV[k])
///   pix_out[k] = dt_UCS_JCH_to_HSB(JCH)
///   if use_filter:
///     Lscharr[k] = gradient_amp * sqrf(max(0, scharr(saturation, width) - 0.02))
///   if JCH[1] > NORM_MIN:
///     corrections[k*2+0] = LUT_hue[hue]
///     corrections[k*2+1] = LUT_saturation[hue]
///     b_corrections[k]   = sat * (LUT_brightness[hue] - 1)
///   else: corrections = {0, 1}, b_corrections = 0
///
/// `lut_hue`, `lut_saturation`, `lut_brightness` are each 512-float arrays.
/// `saturation` is shared with the Scharr gradient: index [kk] is clamped to [1, h-2].
/// Matches the DT_OMP_FOR at src/iop/colorequal.c:974.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_compute_hsb_corrections(
    uv_buf:       *const f32,
    lscharr:      *mut f32,
    saturation:   *const f32,
    pix_out_buf:  *mut f32,
    corrections:  *mut f32,
    b_corrections: *mut f32,
    npixels: usize,
    width: usize,
    height: usize,
    white: f32,
    gradient_amp: f32,
    use_filter: i32,
    lut_hue:        *const f32,
    lut_saturation: *const f32,
    lut_brightness: *const f32,
) {
    if npixels == 0 { return; }
    const NORM_MIN: f32 = 1.52587890625e-05;
    let uv    = std::slice::from_raw_parts(uv_buf, npixels * 2);
    let ls    = std::slice::from_raw_parts_mut(lscharr, npixels);
    let sat   = std::slice::from_raw_parts(saturation, npixels);
    let pout  = std::slice::from_raw_parts_mut(pix_out_buf, npixels * 4);
    let corr  = std::slice::from_raw_parts_mut(corrections, npixels * 2);
    let bcorr = std::slice::from_raw_parts_mut(b_corrections, npixels);
    let lh = std::slice::from_raw_parts(lut_hue,        LUT_ELEM);
    let ls2 = std::slice::from_raw_parts(lut_saturation, LUT_ELEM);
    let lb = std::slice::from_raw_parts(lut_brightness,  LUT_ELEM);

    // Cast the LUT slices to fixed-size arrays for lookup_gamut
    let lh_arr: &[f32; LUT_ELEM] = lh.try_into().expect("LUT_HUE size");
    let ls_arr: &[f32; LUT_ELEM] = ls2.try_into().expect("LUT_SAT size");
    let lb_arr: &[f32; LUT_ELEM] = lb.try_into().expect("LUT_BRI size");

    for k in 0..npixels {
        let row = k / width;
        let col = k - row * width;
        let uv_pair = [uv[2*k], uv[2*k+1], 0.0, 0.0];
        let jch = color::dt_ucs_luv_to_jch(ls[k], white, &[uv_pair[0], uv_pair[1]]);
        let mut hsb = color::dt_ucs_jch_to_hsb(&jch);
        pout[k*4]   = hsb[0];
        pout[k*4+1] = hsb[1];
        pout[k*4+2] = hsb[2];
        // alpha was already set by Step 1

        if use_filter != 0 {
            let vrow = row.clamp(1, height.saturating_sub(2));
            let vcol = col.clamp(1, width.saturating_sub(2));
            let kk   = vrow * width + vcol;
            let g = scharr_gradient(sat, width, kk);
            ls[k] = gradient_amp * (0.0_f32.max(g - 0.02)).powi(2);
        }

        if jch[1] > NORM_MIN {
            let hue_rad = pout[k*4]; // HSB[0] = hue in radians
            corr[2*k]   = lookup_gamut(lh_arr, hue_rad);
            corr[2*k+1] = lookup_gamut(ls_arr, hue_rad);
            bcorr[k]    = hsb[1] * (lookup_gamut(lb_arr, hue_rad) - 1.0);
        } else {
            corr[2*k]   = 0.0;
            corr[2*k+1] = 1.0;
            bcorr[k]    = 0.0;
        }
        let _ = hsb; // suppress move warning
    }
}

/// STEP 5: apply corrections and convert back to RGB.
///
/// For each pixel k:
///   pix_out[k] += corrections[k*2+0]   (hue offset)
///   pix_out[k*4+1] = max(0, sat * (1 + SAT_EFFECT * (corrections[k*2+1] - 1)))
///   pix_out[k*4+2] = max(0, bri * (1 + BRIGHT_EFFECT * b_corrections[k]))
///   gamut_map_HSB(pix_out)
///   XYZ_D65 = dt_UCS_HSB_to_XYZ(pix_out, white)
///   pix_out = dot_product(XYZ_D65, output_matrix)
///
/// Matches the DT_OMP_FOR at src/iop/colorequal.c:1035.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_apply_corrections(
    pix_out_buf:  *mut f32,
    corrections:  *const f32,
    b_corrections: *const f32,
    npixels: usize,
    white: f32,
    gamut_lut: *const f32,   // LUT_ELEM floats
    output_matrix: *const f32, // flat 16-float (4×4)
) {
    if npixels == 0 { return; }
    const SAT_EFFECT:   f32 = 2.0;
    const BRIGHT_EFFECT: f32 = 8.0;
    let pout  = std::slice::from_raw_parts_mut(pix_out_buf, npixels * 4);
    let corr  = std::slice::from_raw_parts(corrections,   npixels * 2);
    let bcorr = std::slice::from_raw_parts(b_corrections, npixels);
    let glut  = std::slice::from_raw_parts(gamut_lut, LUT_ELEM);
    let glut_arr: &[f32; LUT_ELEM] = glut.try_into().expect("gamut LUT size");

    let m_sl = std::slice::from_raw_parts(output_matrix, 16);
    let mut m_out = [[0.0_f32; 4]; 4];
    for r in 0..4 { for c in 0..4 { m_out[r][c] = m_sl[r*4+c]; } }

    for k in 0..npixels {
        let b = k * 4;
        pout[b]   += corr[2*k];   // hue offset
        pout[b+1]  = (pout[b+1] * (1.0 + SAT_EFFECT * (corr[2*k+1] - 1.0))).max(0.0);
        pout[b+2]  = (pout[b+2] * (1.0 + BRIGHT_EFFECT * bcorr[k])).max(0.0);

        // gamut_map_HSB
        let mut hsb = [pout[b], pout[b+1], pout[b+2], pout[b+3]];
        gamut_map_hsb(&mut hsb, glut_arr, white);
        pout[b]   = hsb[0]; pout[b+1] = hsb[1]; pout[b+2] = hsb[2];

        // HSB → XYZ D65
        let xyz_d65 = dt_ucs_hsb_to_xyz(&hsb, white);

        // XYZ D65 → pipe RGB (dot_product: out[i] = M[i] · xyz)
        let rgb = color::dot_product(&xyz_d65, &m_out);
        pout[b]   = rgb[0]; pout[b+1] = rgb[1]; pout[b+2] = rgb[2];
        // alpha preserved in pout[b+3]
    }
}

/// Mask-display visualization: convert HSB brightness + correction into grey overlay.
///
/// For each pixel k:
///   val = sqrt(pix_out[k*4+2] * white)
///   corr based on mode:
///     0=BRIGHTNESS:      BRIGHT_EFFECT * b_corrections[k]
///     1=SATURATION:      SAT_EFFECT * (corrections[k*2+1] - 1)
///     2=BRIGHTNESS_GRAD: _get_satweight(saturation[k]-bright_shift) - 0.5
///     3=SATURATION_GRAD: _get_satweight(saturation[k]-sat_shift)    - 0.5
///     4=HUE (default):   0.2 * corrections[k*2+0]
///   neg = corr < 0
///   corr = |corr| < 0.002 ? 0 : |corr|
///   pix_out[0] = max(0, neg ? val-corr : val)
///   pix_out[1] = max(0, val-corr)
///   pix_out[2] = max(0, neg ? val : val-corr)
///   if mode==BRIGHTNESS && Lscharr[k]>0.1: pix_out={0, Lscharr[k], 0}
///
/// Matches the DT_OMP_FOR at src/iop/colorequal.c:930.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_mask_display(
    pix_out_buf:  *mut f32,
    corrections:  *const f32,
    b_corrections: *const f32,
    saturation:   *const f32,
    lscharr:      *const f32,
    npixels: usize,
    mode: i32,
    white: f32,
    sat_shift: f32,
    bright_shift: f32,
    satweights: *const f32,
    satsize: usize,
) {
    if npixels == 0 || satsize == 0 { return; }
    const SAT_EFFECT:   f32 = 2.0;
    const BRIGHT_EFFECT: f32 = 8.0;
    let pout  = std::slice::from_raw_parts_mut(pix_out_buf, npixels * 4);
    let corr  = std::slice::from_raw_parts(corrections,   npixels * 2);
    let bcorr = std::slice::from_raw_parts(b_corrections, npixels);
    let sat   = std::slice::from_raw_parts(saturation,    npixels);
    let ls    = std::slice::from_raw_parts(lscharr,       npixels);
    let sw    = std::slice::from_raw_parts(satweights, 2 * satsize + 1);

    for k in 0..npixels {
        let b   = k * 4;
        let val = (pout[b + 2] * white).sqrt();
        let c   = match mode {
            0 => BRIGHT_EFFECT * bcorr[k],
            1 => SAT_EFFECT * (corr[2*k+1] - 1.0),
            2 => get_satweight(sat[k] - bright_shift, sw, satsize) - 0.5,
            3 => get_satweight(sat[k] - sat_shift,    sw, satsize) - 0.5,
            _ => 0.2 * corr[2*k],  // HUE
        };
        let neg  = c < 0.0;
        let c    = if c.abs() < 2e-3 { 0.0 } else { c.abs() };
        pout[b]   = if neg { (val - c).max(0.0) } else { val.max(0.0) };
        pout[b+1] = (val - c).max(0.0);
        pout[b+2] = if neg { val.max(0.0) } else { (val - c).max(0.0) };
        if mode == 0 && ls[k] > 0.1 {
            pout[b] = 0.0; pout[b+2] = 0.0; pout[b+1] = ls[k];
        }
    }
}

// ── _guide_with_chromaticity helpers ─────────────────────────────────────────

/// Build the guide×corrections correlation matrix.
///
///   corr[k*4+0] = UV[k*2+0] * corrections[k*2+1]   corr(U, sat)
///   corr[k*4+1] = UV[k*2+1] * corrections[k*2+1]   corr(V, sat)
///   corr[k*4+2] = UV[k*2+0] * b_corrections[k]      corr(U, bright)
///   corr[k*4+3] = UV[k*2+1] * b_corrections[k]      corr(V, bright)
///
/// Matches the DT_OMP_FOR_SIMD at src/iop/colorequal.c:698.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_init_correlations(
    uv_buf:           *const f32,
    corrections_buf:  *const f32,
    b_corrections:    *const f32,
    corr_buf:         *mut f32,
    pixels: usize,
) {
    if pixels == 0 { return; }
    let uv   = std::slice::from_raw_parts(uv_buf,          pixels * 2);
    let corr_in = std::slice::from_raw_parts(corrections_buf, pixels * 2);
    let bcorr   = std::slice::from_raw_parts(b_corrections,   pixels);
    let corr    = std::slice::from_raw_parts_mut(corr_buf,    pixels * 4);
    for k in 0..pixels {
        let u  = uv[2 * k];
        let v  = uv[2 * k + 1];
        let cs = corr_in[2 * k + 1];
        let cb = bcorr[k];
        corr[4 * k]     = u * cs;
        corr[4 * k + 1] = v * cs;
        corr[4 * k + 2] = u * cb;
        corr[4 * k + 3] = v * cb;
    }
}

/// Finish the correlations by subtracting avg(UV) × avg(corrections).
///
/// Matches the DT_OMP_FOR_SIMD at src/iop/colorequal.c:727.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_finish_correlations(
    uv_buf:           *const f32,
    corrections_buf:  *const f32,
    b_corrections:    *const f32,
    corr_buf:         *mut f32,
    pixels: usize,
) {
    if pixels == 0 { return; }
    let uv      = std::slice::from_raw_parts(uv_buf,          pixels * 2);
    let corr_in = std::slice::from_raw_parts(corrections_buf, pixels * 2);
    let bcorr   = std::slice::from_raw_parts(b_corrections,   pixels);
    let corr    = std::slice::from_raw_parts_mut(corr_buf,    pixels * 4);
    for k in 0..pixels {
        let u  = uv[2 * k];
        let v  = uv[2 * k + 1];
        let cs = corr_in[2 * k + 1];
        let cb = bcorr[k];
        corr[4 * k]     -= u * cs;
        corr[4 * k + 1] -= v * cs;
        corr[4 * k + 2] -= u * cb;
        corr[4 * k + 3] -= v * cb;
    }
}

/// Compute guided-filter regression params (a, b) from covariance + correlations.
///
/// Same 2×2 matrix inversion as `darkroom_colorequal_prepare_prefilter` but:
///   - numerator is `correlations` (not `covariance`)
///   - `b[k*2+0] = corrections[k*2+1] - a·UV`
///   - `b[k*2+1] = b_corrections[k]  - a·UV`
///
/// Matches the DT_OMP_FOR_SIMD at src/iop/colorequal.c:755.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_compute_guided_params(
    uv_buf:          *const f32,
    covariance_buf:  *const f32,
    correlations:    *const f32,
    corrections_buf: *const f32,
    b_corrections:   *const f32,
    a_buf: *mut f32,
    b_buf: *mut f32,
    pixels: usize,
    eps: f32,
) {
    if pixels == 0 { return; }
    let uv    = std::slice::from_raw_parts(uv_buf,          pixels * 2);
    let cov   = std::slice::from_raw_parts(covariance_buf,  pixels * 4);
    let cor   = std::slice::from_raw_parts(correlations,    pixels * 4);
    let corr  = std::slice::from_raw_parts(corrections_buf, pixels * 2);
    let bcorr = std::slice::from_raw_parts(b_corrections,   pixels);
    let a     = std::slice::from_raw_parts_mut(a_buf, pixels * 4);
    let b     = std::slice::from_raw_parts_mut(b_buf, pixels * 2);

    for k in 0..pixels {
        let sigma = [
            cov[4 * k]     + eps,
            cov[4 * k + 1],
            cov[4 * k + 2],
            cov[4 * k + 3] + eps,
        ];
        let det = sigma[0] * sigma[3] - sigma[1] * sigma[2];
        if det.abs() > 4.0 * f32::EPSILON {
            let si = [sigma[3]/det, -sigma[1]/det, -sigma[2]/det, sigma[0]/det];
            a[4*k]   = cor[4*k]*si[0] + cor[4*k+1]*si[1];
            a[4*k+1] = cor[4*k]*si[2] + cor[4*k+1]*si[3];
            a[4*k+2] = cor[4*k+2]*si[0] + cor[4*k+3]*si[1];
            a[4*k+3] = cor[4*k+2]*si[2] + cor[4*k+3]*si[3];
        } else {
            a[4*k] = 0.0; a[4*k+1] = 0.0; a[4*k+2] = 0.0; a[4*k+3] = 0.0;
        }
        let u = uv[2*k]; let v = uv[2*k+1];
        b[2*k]   = corr[2*k+1]  - a[4*k]*u - a[4*k+1]*v;
        b[2*k+1] = bcorr[k]     - a[4*k+2]*u - a[4*k+3]*v;
    }
}

/// Apply the guided-filter to corrections using the sigmoid saturation weighting.
///
/// For each pixel k:
///   cv[0] = a[k*4+0]*U + a[k*4+1]*V + b[k*2+0]
///   cv[1] = a[k*4+2]*U + a[k*4+3]*V + b[k*2+1]
///   corrections[k*2+1] = 1 + (cv[0]-1) * get_satweight(sat[k] - sat_shift)
///   gradient_weight    = 1 - CLIP(gradients[k])
///   b_corrections[k]   = cv[1] * gradient_weight * get_satweight(sat[k] - bright_shift)
///
/// Matches the DT_OMP_FOR_SIMD at src/iop/colorequal.c:823.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_apply_guided_filter(
    uv_buf:       *const f32,
    saturation:   *const f32,
    gradients:    *const f32,
    a_buf:        *const f32,
    b_buf:        *const f32,
    corrections:  *mut f32,
    b_corrections: *mut f32,
    npixels: usize,
    sat_shift: f32,
    bright_shift: f32,
    satweights: *const f32,
    satsize: usize,
) {
    if npixels == 0 || satsize == 0 { return; }
    let uv   = std::slice::from_raw_parts(uv_buf,     npixels * 2);
    let sat  = std::slice::from_raw_parts(saturation, npixels);
    let grad = std::slice::from_raw_parts(gradients,  npixels);
    let a    = std::slice::from_raw_parts(a_buf,      npixels * 4);
    let b    = std::slice::from_raw_parts(b_buf,      npixels * 2);
    let corr = std::slice::from_raw_parts_mut(corrections,   npixels * 2);
    let bcorr = std::slice::from_raw_parts_mut(b_corrections, npixels);
    let sw   = std::slice::from_raw_parts(satweights, 2 * satsize + 1);

    for k in 0..npixels {
        let u = uv[2*k]; let v = uv[2*k+1];
        let cv0 = a[4*k]*u + a[4*k+1]*v + b[2*k];
        let cv1 = a[4*k+2]*u + a[4*k+3]*v + b[2*k+1];
        let w_sat  = get_satweight(sat[k] - sat_shift,   sw, satsize);
        let w_bri  = get_satweight(sat[k] - bright_shift, sw, satsize);
        let gradient_weight = (1.0 - grad[k]).clamp(0.0, 1.0);
        corr[2*k+1]  = 1.0 + (cv0 - 1.0) * w_sat;
        bcorr[k]     = cv1 * gradient_weight * w_bri;
    }
}

// ── Step 1: RGB → dt UCS UV ───────────────────────────────────────────────────

const NORM_MIN: f32 = 1.52587890625e-05; // 2^-16, from common/math.h

/// STEP 1 of colorequal process(): convert RGB to dt UCS UV + saturation + L*.
///
/// For each pixel k:
///   XYZ_D65 = dot_product(pix_in, input_matrix)   (non-transposed)
///   xyY     = d65_xyz_to_xyy(XYZ_D65)
///   sat[k]  = delta / dmax  (0 if dmax or delta < NORM_MIN)
///   UV[k*2..] = xyY_to_dt_UCS_UV(xyY)
///   Lscharr[k] = Y_to_dt_UCS_L_star(xyY[2])
///
/// `input_matrix` is a flat 16-float row-major 4×4 matrix:
///   input_matrix = XYZ_D50_to_D65_CAT16 × work_profile->matrix_in
///
/// Matches the DT_OMP_FOR at src/iop/colorequal.c:944.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorequal_rgb_to_ucs_uv(
    in_buf:       *const f32,
    uv_buf:       *mut f32,
    saturation:   *mut f32,
    lscharr:      *mut f32,
    npixels: usize,
    ch: usize,
    input_matrix: *const f32,   // flat 16-float (4×4)
) {
    if npixels == 0 || ch == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf, npixels * ch);
    let uv  = std::slice::from_raw_parts_mut(uv_buf,    npixels * 2);
    let sat = std::slice::from_raw_parts_mut(saturation, npixels);
    let ls  = std::slice::from_raw_parts_mut(lscharr,    npixels);

    let m_slice = std::slice::from_raw_parts(input_matrix, 16);
    let mut m = [[0.0_f32; 4]; 4];
    for r in 0..4 { for c in 0..4 { m[r][c] = m_slice[r * 4 + c]; } }

    for k in 0..npixels {
        let b = k * ch;
        let pix = [inp[b], inp[b+1], inp[b+2], inp[b+3].max(0.0)];

        // dot_product: out[i] = M[i] · pix (row-major, non-transposed)
        let xyz_d65 = color::dot_product(&pix, &m);

        // XYZ D65 → xyY with D65 white-point fallback
        let xyy = color::d65_xyz_to_xyy(&xyz_d65);

        // Saturation from input RGB
        let dmax = pix[0].max(pix[1]).max(pix[2]);
        let dmin = pix[0].min(pix[1]).min(pix[2]);
        let delta = dmax - dmin;
        sat[k] = if dmax > NORM_MIN && delta > NORM_MIN { delta / dmax } else { 0.0 };

        // UV in dt UCS space
        let uv_pair = color::xyy_to_dt_ucs_uv(&xyy);
        uv[2*k]   = uv_pair[0];
        uv[2*k+1] = uv_pair[1];

        // L* for later JCH conversion
        ls[k] = color::y_to_dt_ucs_l_star(xyy[2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_covariance_computes_outer_product() {
        // UV = [(2, 3)] → cov = [4, 6, 6, 9]
        let uv  = vec![2.0_f32, 3.0];
        let mut cov = vec![0.0_f32; 4];
        unsafe { darkroom_colorequal_init_covariance(uv.as_ptr(), cov.as_mut_ptr(), 1); }
        assert_eq!(cov, vec![4.0, 6.0, 6.0, 9.0]);
    }

    #[test]
    fn finish_covariance_subtracts_product() {
        // cov was [10, 8, 8, 12]; avg = (2, 3) → subtract [4,6,6,9] → [6,2,2,3]
        let uv  = vec![2.0_f32, 3.0];
        let mut cov = vec![10.0_f32, 8.0, 8.0, 12.0];
        unsafe { darkroom_colorequal_finish_covariance(uv.as_ptr(), cov.as_mut_ptr(), 1); }
        assert_eq!(cov, vec![6.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn prepare_prefilter_identity_when_cov_is_eps_times_identity() {
        // cov = 0; σ = ε*I; σ⁻¹ = (1/ε)*I; a = cov * σ⁻¹ = 0; b = UV
        let uv  = vec![0.5_f32, 0.7];
        let cov = vec![0.0_f32; 4];
        let mut a = vec![99.0_f32; 4];
        let mut b = vec![99.0_f32; 2];
        unsafe {
            darkroom_colorequal_prepare_prefilter(
                uv.as_ptr(), cov.as_ptr(), a.as_mut_ptr(), b.as_mut_ptr(), 1, 1e-4,
            );
        }
        // a should be 0
        for v in &a { assert!(v.abs() < 1e-6, "a={v}"); }
        // b = uv since a is 0
        assert!((b[0] - 0.5).abs() < 1e-6);
        assert!((b[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn prepare_prefilter_singular_matrix_zeroes_a() {
        // All-zero cov + tiny eps → near-singular matrix
        let uv  = vec![1.0_f32, 1.0];
        let cov = vec![0.0_f32; 4];
        let mut a = vec![1.0_f32; 4];
        let mut b = vec![0.0_f32; 2];
        // eps = 0 → det = 0 → singular path
        unsafe {
            darkroom_colorequal_prepare_prefilter(
                uv.as_ptr(), cov.as_ptr(), a.as_mut_ptr(), b.as_mut_ptr(), 1, 0.0,
            );
        }
        for v in &a { assert_eq!(*v, 0.0, "a={v}"); }
    }

    /// Build a satweights table with the same formula as C _init_satweights.
    fn make_satweights(satsize: usize, contrast: f64) -> Vec<f32> {
        let factor = -60.0 - 40.0 * contrast;
        let n = 2 * satsize + 1;
        (0..n).map(|idx| {
            let i = idx as i64 - satsize as i64;
            let val = 0.5 / satsize as f64 * i as f64;
            (1.0 / (1.0 + (factor * val).exp())) as f32
        }).collect()
    }

    #[test]
    fn apply_prefilter_identity_correction_is_noop() {
        // a = identity, b = 0 → u_corr = u, v_corr = v → no change regardless of satweight
        const SATSIZE: usize = 4096;
        let sw = make_satweights(SATSIZE, 0.0);
        let mut uv = vec![0.3_f32, 0.5];
        let sat = vec![0.5_f32];
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![0.0_f32, 0.0];
        unsafe {
            darkroom_colorequal_apply_prefilter(
                uv.as_mut_ptr(), sat.as_ptr(), a.as_ptr(), b.as_ptr(),
                1, 0.0, sw.as_ptr(), SATSIZE,
            );
        }
        assert!((uv[0] - 0.3).abs() < 1e-5);
        assert!((uv[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn apply_prefilter_uses_sigmoid_not_linear_ramp() {
        // With contrast=0 the sigmoid at sat-sat_shift=0.0 should be 0.5 (midpoint)
        // not 1.0 (which the old linear ramp would give at sat_shift=0.0, sat=0.0).
        const SATSIZE: usize = 4096;
        let sw = make_satweights(SATSIZE, 0.0);
        let weight = get_satweight(0.0, &sw, SATSIZE);
        // Logistic at 0 → 0.5 (regardless of contrast)
        assert!((weight - 0.5).abs() < 0.01, "sigmoid midpoint should be ~0.5, got {weight}");
    }
}
