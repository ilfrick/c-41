use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Colorin;

impl IopProcess for Colorin {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colorin" }
}

const EPSILON: f32 = 216.0 / 24389.0;
const KAPPA: f32 = 24389.0 / 27.0;
const D50_INV: [f32; 3] = [1.0 / 0.9642, 1.0, 1.0 / 0.8249];
/// Input-profile tone-curve LUT resolution (`#define LUT_SAMPLES 0x10000`).
const LUT_SAMPLES: usize = 0x10000;

#[inline(always)]
fn xyz_to_lab_f(x: f32) -> f32 {
    if x > EPSILON { x.cbrt() } else { (KAPPA * x + 16.0) / 116.0 }
}

/// D50 XYZ → Lab. `xyz` is the top-3 tristimulus.
#[inline(always)]
fn xyz_to_lab(xyz: [f32; 3]) -> [f32; 4] {
    let f = [
        xyz_to_lab_f(xyz[0] * D50_INV[0]),
        xyz_to_lab_f(xyz[1] * D50_INV[1]),
        xyz_to_lab_f(xyz[2] * D50_INV[2]),
    ];
    [116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), -200.0 * (f[2] - f[1]), 0.0]
}

/// `cmatrix` (16 floats, row-major 4×4 `dt_colormatrix_t`) applied to a 3-vector,
/// standard row form `out[r] = Σ_c cm[r*4+c]·v[c]`. This equals the C bm path's
/// `transpose_3xSSE` + `dt_apply_transposed_color_matrix` (a transpose then a
/// transposed apply is a plain apply), matching the fastpath convention.
#[inline(always)]
fn cmatrix_apply(cm: &[f32], v: [f32; 3]) -> [f32; 3] {
    [
        cm[0] * v[0] + cm[1] * v[1] + cm[2] * v[2],
        cm[4] * v[0] + cm[5] * v[1] + cm[6] * v[2],
        cm[8] * v[0] + cm[9] * v[1] + cm[10] * v[2],
    ]
}

/// Tone-curve LUT lerp (`_lerp_lut`); caller guarantees `v < 1.0`, so the index
/// stays `<= LUT_SAMPLES-2` and needs no clamp.
#[inline(always)]
fn lerp_lut(lut: &[f32], v: f32) -> f32 {
    let z = v.max(0.0);
    let ft = z * (LUT_SAMPLES - 1) as f32;
    let t = ft as usize;
    let f = ft - t as f32;
    lut[t] * (1.0 - f) + lut[t + 1] * f
}

/// Unbounded extrapolation `coeff[1]·pow(v·coeff[0], coeff[2])`
/// (`dt_iop_eval_exp`).
#[inline(always)]
fn eval_exp(coeff: &[f32], v: f32) -> f32 {
    coeff[1] * (v * coeff[0]).powf(coeff[2])
}

/// colorin's blue-tint mapping (`_apply_blue_mapping`): nudge very-blue,
/// bright-enough pixels slightly toward green. In place on channels 1,2.
#[inline(always)]
fn apply_blue_mapping(p: &mut [f32; 4]) {
    let yy = p[0] + p[1] + p[2];
    if yy > 0.0 {
        let zz = p[2] / yy;
        let (bound_z, bound_y, amount) = (0.5f32, 0.5f32, 0.11f32);
        if zz > bound_z {
            let t = (zz - bound_z) / (1.0 - bound_z) * (yy / bound_y).min(1.0);
            p[1] += t * amount;
            p[2] -= t * amount;
        }
    }
}

/// Camera-RGB → Lab via the input colour matrix, the tone-curve ("shaper") path.
/// Replaces the per-pixel loop in `_process_cmatrix_bm()` in colorin.c (the
/// matrix codepath that is NOT the linear fastpath — it applies the input
/// profile's tone curves and the blue mapping first).
///
/// Per channel: if the profile is non-linear (`lut[c][0] >= 0`), map `in[c]`
/// through the LUT (`in[c] < 1`) or the unbounded exp fit; else pass through.
/// Then `_apply_blue_mapping`, then either `cmatrix→XYZ→Lab` (no clipping) or,
/// when `clipping != 0`, `nmatrix→clamp[0,1]→lmatrix→XYZ→Lab` (gamut clip).
///
/// `cmatrix`/`nmatrix`/`lmatrix`: 16 floats each, row-major 4×4 (`dt_colormatrix_t`,
/// untransposed as stored). `lut`: 3×`LUT_SAMPLES` floats (channel c at
/// `c*LUT_SAMPLES`). `unbounded_coeffs`: 3×3 floats (channel c at `c*3`).
/// Output alpha is 0.
///
/// # Safety
/// All pointers must be valid for the stated lengths; `in_buf`/`out_buf` cover
/// `npixels*4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorin_cmatrix_bm(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    cmatrix: *const f32,
    nmatrix: *const f32,
    lmatrix: *const f32,
    lut: *const f32,
    unbounded_coeffs: *const f32,
    clipping: i32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let cm = std::slice::from_raw_parts(cmatrix, 16);
    let nm = std::slice::from_raw_parts(nmatrix, 16);
    let lm = std::slice::from_raw_parts(lmatrix, 16);
    let lut = std::slice::from_raw_parts(lut, 3 * LUT_SAMPLES);
    let uc = std::slice::from_raw_parts(unbounded_coeffs, 3 * 3);
    let clipping = clipping != 0;

    for k in 0..npixels {
        let b = k * 4;
        let mut cam = [0.0f32; 4];
        for c in 0..3 {
            let ch_lut = &lut[c * LUT_SAMPLES..(c + 1) * LUT_SAMPLES];
            let v = input[b + c];
            cam[c] = if ch_lut[0] >= 0.0 {
                if v < 1.0 {
                    lerp_lut(ch_lut, v)
                } else {
                    eval_exp(&uc[c * 3..c * 3 + 3], v)
                }
            } else {
                v
            };
        }

        apply_blue_mapping(&mut cam);
        let cam3 = [cam[0], cam[1], cam[2]];

        let xyz = if !clipping {
            cmatrix_apply(cm, cam3)
        } else {
            let nrgb = cmatrix_apply(nm, cam3);
            let crgb = [nrgb[0].clamp(0.0, 1.0), nrgb[1].clamp(0.0, 1.0), nrgb[2].clamp(0.0, 1.0)];
            cmatrix_apply(lm, crgb)
        };

        let lab = xyz_to_lab(xyz);
        output[b..b + 4].copy_from_slice(&lab);
    }
}

/// Camera-RGB → Lab via a 4×4 colour matrix (cam→XYZ) and D50 XYZ→Lab.
///
/// Replaces the per-pixel loop inside _cmatrix_fastpath_simple() in colorin.c.
///
/// `corr`:    4 white-balance correction coefficients
/// `cmatrix`: 16 floats, row-major 4×4 (dt_colormatrix_t); only the 3×3
///            top-left is used.
/// Output alpha is always 0.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorin_cmatrix_fastpath_simple(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    corr: *const f32,
    cmatrix: *const f32,
) {
    let input  = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let cr = std::slice::from_raw_parts(corr, 4);
    // dt_colormatrix_t is float[4][4]: row stride = 4
    let cm = std::slice::from_raw_parts(cmatrix, 16);

    for k in 0..npixels {
        let b = k * 4;
        let cam = [
            input[b]     * cr[0],
            input[b + 1] * cr[1],
            input[b + 2] * cr[2],
        ];

        // XYZ[r] = cm[r*4+0]*cam[0] + cm[r*4+1]*cam[1] + cm[r*4+2]*cam[2]
        let xyz = [
            cm[0] * cam[0] + cm[1] * cam[1] + cm[2] * cam[2],
            cm[4] * cam[0] + cm[5] * cam[1] + cm[6] * cam[2],
            cm[8] * cam[0] + cm[9] * cam[1] + cm[10] * cam[2],
        ];

        // XYZ → Lab (D50 white point)
        let f = [
            xyz_to_lab_f(xyz[0] * D50_INV[0]),
            xyz_to_lab_f(xyz[1] * D50_INV[1]),
            xyz_to_lab_f(xyz[2] * D50_INV[2]),
        ];

        output[b]     = 116.0 * f[1] - 16.0;
        output[b + 1] = 500.0 * (f[0] - f[1]);
        output[b + 2] = -200.0 * (f[2] - f[1]);
        output[b + 3] = 0.0;
    }
}

/// Shared per-pixel tone-curve application (`_apply_tone_curves` in colorin.c).
/// For each channel `c`: if the profile is non-linear (`lut[c][0] >= 0`), map
/// `cam[c]` through the LUT (`cam[c] < 1`) or the unbounded exp fit; else pass
/// through. Operates on lanes 0..2 in place; lane 3 untouched.
#[inline(always)]
fn apply_tone_curves(cam: &mut [f32; 4], lut: &[f32], uc: &[f32]) {
    for c in 0..3 {
        let ch_lut = &lut[c * LUT_SAMPLES..(c + 1) * LUT_SAMPLES];
        if ch_lut[0] >= 0.0 {
            let v = cam[c];
            cam[c] = if v < 1.0 {
                lerp_lut(ch_lut, v)
            } else {
                eval_exp(&uc[c * 3..c * 3 + 3], v)
            };
        }
    }
}

/// Camera-RGB → Lab via the normalising matrix + gamut clip (linear path).
///
/// Replaces the per-pixel loop in `_cmatrix_fastpath_clipping()` in colorin.c:
/// per pixel, `cam = in·corr`, `nRGB = nmatrix·cam`, clamp to [0,1]
/// (`dt_vector_clip`), then `lmatrix·nRGB → XYZ → Lab` (`dt_RGB_to_Lab`).
/// Output alpha is 0.
///
/// `nmatrix`/`lmatrix`: 16 floats each, row-major 4×4 (`dt_colormatrix_t`,
/// untransposed as stored — the C builds row vectors out of the columns and
/// applies by row, which is a plain row-major apply). `corr`: 4 white-balance
/// coefficients. Parameter order mirrors the C wrapper (`nmatrix, lmatrix,
/// corr`) with `in_buf`/`out_buf` first per file convention.
///
/// # Safety
/// All pointers must be non-null and valid for the stated lengths;
/// `in_buf`/`out_buf` cover `npixels*4` floats. Null pointers, `npixels == 0`
/// and `npixels*4` overflow are guarded (no-op return).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorin_cmatrix_fastpath_clipping(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    nmatrix: *const f32,
    lmatrix: *const f32,
    corr: *const f32,
) {
    if npixels == 0 { return; }
    if in_buf.is_null() || out_buf.is_null() || nmatrix.is_null() || lmatrix.is_null() || corr.is_null() {
        return;
    }
    let Some(len) = npixels.checked_mul(4) else { return; };
    let input = std::slice::from_raw_parts(in_buf, len);
    let output = std::slice::from_raw_parts_mut(out_buf, len);
    let nm = std::slice::from_raw_parts(nmatrix, 16);
    let lm = std::slice::from_raw_parts(lmatrix, 16);
    let cr = std::slice::from_raw_parts(corr, 4);

    for k in 0..npixels {
        let b = k * 4;
        let cam = [input[b] * cr[0], input[b + 1] * cr[1], input[b + 2] * cr[2]];
        let nrgb = cmatrix_apply(nm, cam);
        // dt_vector_clip clamps all 4 lanes; lane 3 of nRGB is 0 (matrix rows
        // carry 0 in lane 3) and is never read downstream, so clamping lanes
        // 0..2 here is exactly equivalent.
        let crgb = [
            nrgb[0].max(0.0).min(1.0),
            nrgb[1].max(0.0).min(1.0),
            nrgb[2].max(0.0).min(1.0),
        ];
        let lab = xyz_to_lab(cmatrix_apply(lm, crgb));
        output[b..b + 4].copy_from_slice(&lab);
    }
}

/// Camera-RGB → Lab via the input colour matrix + tone curves, no gamut clip.
///
/// Replaces the per-pixel loop in `_cmatrix_proper_simple()` in colorin.c:
/// per pixel, `cam = in·corr`, `_apply_tone_curves`, then
/// `cmatrix·cam → XYZ → Lab` (`dt_RGB_to_Lab`). Output alpha is 0.
///
/// `lut`: 3×`LUT_SAMPLES` floats (channel c at `c*LUT_SAMPLES`; `lut[c][0] < 0`
/// marks a linear channel = passthrough). `unbounded_coeffs`: 3×3 floats
/// (channel c at `c*3`, `dt_iop_eval_exp` semantics). `cmatrix`: 16 floats,
/// row-major 4×4. `corr`: 4 floats. `d`'s fields sit where `d` sat in the C
/// wrapper (`d, cmatrix, corr` → `lut, unbounded_coeffs, cmatrix, corr`).
///
/// # Safety
/// Same contract as [`darkroom_colorin_cmatrix_fastpath_clipping`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorin_cmatrix_proper_simple(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    lut: *const f32,
    unbounded_coeffs: *const f32,
    cmatrix: *const f32,
    corr: *const f32,
) {
    if npixels == 0 { return; }
    if in_buf.is_null()
        || out_buf.is_null()
        || lut.is_null()
        || unbounded_coeffs.is_null()
        || cmatrix.is_null()
        || corr.is_null()
    {
        return;
    }
    let Some(len) = npixels.checked_mul(4) else { return; };
    let input = std::slice::from_raw_parts(in_buf, len);
    let output = std::slice::from_raw_parts_mut(out_buf, len);
    let lut = std::slice::from_raw_parts(lut, 3 * LUT_SAMPLES);
    let uc = std::slice::from_raw_parts(unbounded_coeffs, 3 * 3);
    let cm = std::slice::from_raw_parts(cmatrix, 16);
    let cr = std::slice::from_raw_parts(corr, 4);

    for k in 0..npixels {
        let b = k * 4;
        let mut cam = [input[b] * cr[0], input[b + 1] * cr[1], input[b + 2] * cr[2], 1.0];
        apply_tone_curves(&mut cam, lut, uc);
        let lab = xyz_to_lab(cmatrix_apply(cm, [cam[0], cam[1], cam[2]]));
        output[b..b + 4].copy_from_slice(&lab);
    }
}

/// Camera-RGB → Lab via tone curves + normalising matrix + gamut clip.
///
/// Replaces the per-pixel loop in `_cmatrix_proper_clipping()` in colorin.c:
/// per pixel, `cam = in·corr`, `_apply_tone_curves`, `nRGB = nmatrix·cam`,
/// clamp to [0,1], then `lmatrix·nRGB → XYZ → Lab`. Output alpha is 0.
/// Pointer layout follows the C wrapper (`d, nmatrix, lmatrix, corr`) with
/// `d` flattened to `lut, unbounded_coeffs`.
///
/// # Safety
/// Same contract as [`darkroom_colorin_cmatrix_fastpath_clipping`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorin_cmatrix_proper_clipping(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    lut: *const f32,
    unbounded_coeffs: *const f32,
    nmatrix: *const f32,
    lmatrix: *const f32,
    corr: *const f32,
) {
    if npixels == 0 { return; }
    if in_buf.is_null()
        || out_buf.is_null()
        || lut.is_null()
        || unbounded_coeffs.is_null()
        || nmatrix.is_null()
        || lmatrix.is_null()
        || corr.is_null()
    {
        return;
    }
    let Some(len) = npixels.checked_mul(4) else { return; };
    let input = std::slice::from_raw_parts(in_buf, len);
    let output = std::slice::from_raw_parts_mut(out_buf, len);
    let lut = std::slice::from_raw_parts(lut, 3 * LUT_SAMPLES);
    let uc = std::slice::from_raw_parts(unbounded_coeffs, 3 * 3);
    let nm = std::slice::from_raw_parts(nmatrix, 16);
    let lm = std::slice::from_raw_parts(lmatrix, 16);
    let cr = std::slice::from_raw_parts(corr, 4);

    for k in 0..npixels {
        let b = k * 4;
        let mut cam = [input[b] * cr[0], input[b + 1] * cr[1], input[b + 2] * cr[2], 1.0];
        apply_tone_curves(&mut cam, lut, uc);
        let nrgb = cmatrix_apply(nm, [cam[0], cam[1], cam[2]]);
        let crgb = [
            nrgb[0].max(0.0).min(1.0),
            nrgb[1].max(0.0).min(1.0),
            nrgb[2].max(0.0).min(1.0),
        ];
        let lab = xyz_to_lab(cmatrix_apply(lm, crgb));
        output[b..b + 4].copy_from_slice(&lab);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_cmatrix() -> Vec<f32> {
        // Identity 4×4 matrix (cam == XYZ for testing)
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn flat_corr() -> Vec<f32> { vec![1.0, 1.0, 1.0, 1.0] }

    #[test]
    fn black_pixel_maps_to_zero_lab() {
        let cm = identity_cmatrix();
        let corr = flat_corr();
        let input = vec![0.0f32, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 4];
        unsafe {
            darkroom_colorin_cmatrix_fastpath_simple(
                input.as_ptr(), out.as_mut_ptr(), 1, corr.as_ptr(), cm.as_ptr(),
            );
        }
        // XYZ=(0,0,0) → f=(16/116, 16/116, 16/116) → L=0, a=0, b=0
        assert!((out[0]).abs() < 1e-4, "L={}", out[0]);
        assert!((out[1]).abs() < 1e-4, "a={}", out[1]);
        assert!((out[2]).abs() < 1e-4, "b={}", out[2]);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn d50_white_maps_to_l100() {
        // D50 white point: XYZ = (0.9642, 1.0, 0.8249)
        // With identity cmatrix, cam == XYZ, so input that white point should give L≈100
        let cm = identity_cmatrix();
        let corr = flat_corr();
        let input = vec![0.9642f32, 1.0, 0.8249, 1.0];
        let mut out = vec![0.0f32; 4];
        unsafe {
            darkroom_colorin_cmatrix_fastpath_simple(
                input.as_ptr(), out.as_mut_ptr(), 1, corr.as_ptr(), cm.as_ptr(),
            );
        }
        assert!((out[0] - 100.0).abs() < 0.01, "L={}", out[0]);
        assert!(out[1].abs() < 0.01, "a={}", out[1]);
        assert!(out[2].abs() < 0.01, "b={}", out[2]);
    }

    #[test]
    fn corr_scales_input() {
        let cm = identity_cmatrix();
        // Double the green channel
        let corr = vec![1.0f32, 2.0, 1.0, 1.0];
        let input = vec![0.1f32, 0.1, 0.1, 1.0];
        let mut out_scaled = vec![0.0f32; 4];
        let mut out_ref    = vec![0.0f32; 4];
        let corr_ref = flat_corr();
        let input_ref = vec![0.1f32, 0.2, 0.1, 1.0];
        unsafe {
            darkroom_colorin_cmatrix_fastpath_simple(
                input.as_ptr(), out_scaled.as_mut_ptr(), 1, corr.as_ptr(), cm.as_ptr(),
            );
            darkroom_colorin_cmatrix_fastpath_simple(
                input_ref.as_ptr(), out_ref.as_mut_ptr(), 1, corr_ref.as_ptr(), cm.as_ptr(),
            );
        }
        for c in 0..3 {
            assert!((out_scaled[c] - out_ref[c]).abs() < 1e-4, "c={} scaled={} ref={}", c, out_scaled[c], out_ref[c]);
        }
    }

    // ── cmatrix-bm loop (m4-86) ──

    const IDENTITY_CM: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    /// A 3×LUT_SAMPLES LUT: linear-profile sentinel (`lut[c][0] = -1`) or an
    /// identity ramp (`lut[c][i] = i/(N-1)`).
    fn make_lut(linear: bool) -> Vec<f32> {
        let mut lut = vec![0.0f32; 3 * LUT_SAMPLES];
        if linear {
            for c in 0..3 {
                lut[c * LUT_SAMPLES] = -1.0;
            }
        } else {
            for c in 0..3 {
                for i in 0..LUT_SAMPLES {
                    lut[c * LUT_SAMPLES + i] = i as f32 / (LUT_SAMPLES - 1) as f32;
                }
            }
        }
        lut
    }

    fn run_bm(input: &[f32], cm: &[f32; 16], nm: &[f32; 16], lm: &[f32; 16], lut: &[f32], clip: i32) -> Vec<f32> {
        let n = input.len() / 4;
        let uc = [0.0f32; 9];
        let mut out = vec![0.0f32; input.len()];
        unsafe {
            darkroom_colorin_cmatrix_bm(
                input.as_ptr(), out.as_mut_ptr(), n, cm.as_ptr(), nm.as_ptr(), lm.as_ptr(),
                lut.as_ptr(), uc.as_ptr(), clip,
            );
        }
        out
    }

    #[test]
    fn bm_linear_identity_d50_white_is_lab_100() {
        // D50 white through an identity cmatrix (XYZ = cam) → Lab (100, 0, 0).
        let input = [0.9642, 1.0, 0.8249, 1.0];
        let lut = make_lut(true);
        let out = run_bm(&input, &IDENTITY_CM, &IDENTITY_CM, &IDENTITY_CM, &lut, 0);
        assert!((out[0] - 100.0).abs() < 1e-2, "L={}", out[0]);
        assert!(out[1].abs() < 1e-2 && out[2].abs() < 1e-2, "a={} b={}", out[1], out[2]);
    }

    #[test]
    fn bm_identity_ramp_lut_matches_linear() {
        // an identity ramp LUT must produce (within lerp precision) the same Lab
        // as the linear-profile path for the same input.
        let input = [0.3, 0.5, 0.2, 1.0];
        let ramp = run_bm(&input, &IDENTITY_CM, &IDENTITY_CM, &IDENTITY_CM, &make_lut(false), 0);
        let lin = run_bm(&input, &IDENTITY_CM, &IDENTITY_CM, &IDENTITY_CM, &make_lut(true), 0);
        for c in 0..3 {
            assert!((ramp[c] - lin[c]).abs() < 1e-3, "c={c} ramp={} lin={}", ramp[c], lin[c]);
        }
    }

    #[test]
    fn bm_clipping_clamps_to_unit_before_lmatrix() {
        // clipping with identity n/l matrices: input 2.0 → nrgb 2.0 → clamped to
        // 1.0 → lmatrix → Lab. Must equal the non-clipping result for input 1.0.
        let lut = make_lut(true);
        let clipped = run_bm(&[2.0, 2.0, 2.0, 1.0], &IDENTITY_CM, &IDENTITY_CM, &IDENTITY_CM, &lut, 1);
        let unit = run_bm(&[1.0, 1.0, 1.0, 1.0], &IDENTITY_CM, &IDENTITY_CM, &IDENTITY_CM, &lut, 0);
        for c in 0..3 {
            assert!((clipped[c] - unit[c]).abs() < 1e-4, "c={c} clipped={} unit={}", clipped[c], unit[c]);
        }
    }

    #[test]
    fn bm_clipping_nonidentity_matrices_index_correctly() {
        // Catches a matrix-index transposition in the clipping branch specifically:
        // non-identity nmatrix (clamps ch0) + non-identity lmatrix, linear profile.
        // input [0.6,0.3,0.2] (blue map is a no-op: zz=0.18<0.5).
        // nmatrix = diag(2,1,1) → nRGB [1.2,0.3,0.2] → clamp → [1.0,0.3,0.2]
        // lmatrix row1 has a 0.1 cross term → XYZ [1.0, 0.1·1+0.3, 0.2] = [1.0,0.4,0.2]
        let nm = [
            2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let lm = [
            1.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let out = run_bm(&[0.6, 0.3, 0.2, 1.0], &IDENTITY_CM, &nm, &lm, &make_lut(true), 1);
        let expected = xyz_to_lab([1.0, 0.4, 0.2]);
        for c in 0..3 {
            assert!((out[c] - expected[c]).abs() < 1e-4, "c={c} out={} exp={}", out[c], expected[c]);
        }
    }

    #[test]
    fn apply_blue_mapping_nudges_very_blue_pixels() {
        // zz = 0.8 > 0.5 → green up, blue down by t·0.11.
        let mut p = [0.1, 0.1, 0.8, 1.0];
        apply_blue_mapping(&mut p);
        // t = (0.8-0.5)/0.5 · min(1.0/0.5, 1.0) = 0.6
        assert!((p[1] - (0.1 + 0.6 * 0.11)).abs() < 1e-6, "g={}", p[1]);
        assert!((p[2] - (0.8 - 0.6 * 0.11)).abs() < 1e-6, "b={}", p[2]);
        assert_eq!(p[0], 0.1, "R untouched");
        // a neutral pixel (zz < 0.5) is untouched
        let mut q = [0.5, 0.5, 0.5, 1.0];
        apply_blue_mapping(&mut q);
        assert_eq!(q, [0.5, 0.5, 0.5, 1.0]);
    }

    // ── clipping variants (m4-181) ──

    fn run_fastpath_clipping(input: &[f32], nm: &[f32; 16], lm: &[f32; 16], corr: &[f32]) -> Vec<f32> {
        let n = input.len() / 4;
        let mut out = vec![0.0f32; input.len()];
        unsafe {
            darkroom_colorin_cmatrix_fastpath_clipping(
                input.as_ptr(), out.as_mut_ptr(), n, nm.as_ptr(), lm.as_ptr(), corr.as_ptr(),
            );
        }
        out
    }

    fn run_proper_simple(input: &[f32], lut: &[f32], uc: &[f32], cm: &[f32; 16], corr: &[f32]) -> Vec<f32> {
        let n = input.len() / 4;
        let mut out = vec![0.0f32; input.len()];
        unsafe {
            darkroom_colorin_cmatrix_proper_simple(
                input.as_ptr(), out.as_mut_ptr(), n, lut.as_ptr(), uc.as_ptr(), cm.as_ptr(),
                corr.as_ptr(),
            );
        }
        out
    }

    fn run_proper_clipping(
        input: &[f32], lut: &[f32], uc: &[f32], nm: &[f32; 16], lm: &[f32; 16], corr: &[f32],
    ) -> Vec<f32> {
        let n = input.len() / 4;
        let mut out = vec![0.0f32; input.len()];
        unsafe {
            darkroom_colorin_cmatrix_proper_clipping(
                input.as_ptr(), out.as_mut_ptr(), n, lut.as_ptr(), uc.as_ptr(), nm.as_ptr(),
                lm.as_ptr(), corr.as_ptr(),
            );
        }
        out
    }

    fn run_fastpath_simple(input: &[f32], corr: &[f32], cm: &[f32; 16]) -> Vec<f32> {
        let n = input.len() / 4;
        let mut out = vec![0.0f32; input.len()];
        unsafe {
            darkroom_colorin_cmatrix_fastpath_simple(
                input.as_ptr(), out.as_mut_ptr(), n, corr.as_ptr(), cm.as_ptr(),
            );
        }
        out
    }

    /// Independent f64 reference for the fastpath-clipping pixel pipeline
    /// (corr scale → nmatrix → [0,1] clip → lmatrix → D50 XYZ→Lab).
    fn ref_fastpath_clipping_f64(cam: [f64; 3], nm: &[f32; 16], lm: &[f32; 16]) -> [f64; 4] {
        let apply = |m: &[f32; 16], v: [f64; 3]| {
            [
                m[0] as f64 * v[0] + m[1] as f64 * v[1] + m[2] as f64 * v[2],
                m[4] as f64 * v[0] + m[5] as f64 * v[1] + m[6] as f64 * v[2],
                m[8] as f64 * v[0] + m[9] as f64 * v[1] + m[10] as f64 * v[2],
            ]
        };
        let n = apply(nm, cam);
        let c = apply(lm, [n[0].clamp(0.0, 1.0), n[1].clamp(0.0, 1.0), n[2].clamp(0.0, 1.0)]);
        let (eps, kap) = (216.0f64 / 24389.0, 24389.0f64 / 27.0);
        let dinv = [1.0 / 0.9642f64, 1.0, 1.0 / 0.8249f64];
        let f: Vec<f64> = c
            .iter()
            .zip(dinv)
            .map(|(&v, d)| {
                let x = v * d;
                if x > eps { x.cbrt() } else { (kap * x + 16.0) / 116.0 }
            })
            .collect();
        [116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), -200.0 * (f[2] - f[1]), 0.0]
    }

    #[test]
    fn fastpath_clipping_identity_matches_fastpath_simple_bit_exact() {
        // In-gamut input through identity n/l matrices: the clamp is a no-op,
        // so the loop must be bit-identical to _cmatrix_fastpath_simple.
        let corr = flat_corr();
        let px = [0.25, 0.5, 0.125, 1.0, 0.9, 0.1, 0.7, 0.0];
        let a = run_fastpath_clipping(&px, &IDENTITY_CM, &IDENTITY_CM, &corr);
        let b = run_fastpath_simple(&px, &corr, &IDENTITY_CM);
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "lane {i}: {x} vs {y}");
        }
    }

    #[test]
    fn fastpath_clipping_matches_f64_reference() {
        // Non-trivial matrices + out-of-gamut values exercise the clip branch.
        let nm: [f32; 16] = [
            2.0, 0.5, -0.25, 0.0, 0.1, 1.0, 0.3, 0.0, -0.5, 0.2, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let lm: [f32; 16] = [
            0.8, 0.1, 0.1, 0.0, 0.05, 0.9, 0.05, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let corr = [1.0f32, 0.5, 2.0, 1.0];
        let px = [1.5, -0.25, 0.75, 1.0, 0.2, 0.4, 0.6, 1.0];
        let out = run_fastpath_clipping(&px, &nm, &lm, &corr);
        for (k, chunk) in px.chunks_exact(4).enumerate() {
            let cam = [
                chunk[0] as f64 * corr[0] as f64,
                chunk[1] as f64 * corr[1] as f64,
                chunk[2] as f64 * corr[2] as f64,
            ];
            let exp = ref_fastpath_clipping_f64(cam, &nm, &lm);
            for c in 0..4 {
                assert!(
                    (out[k * 4 + c] as f64 - exp[c]).abs() < 1e-4,
                    "px{k} c{c}: {} vs {}",
                    out[k * 4 + c],
                    exp[c]
                );
            }
        }
        // Negative nrgb lanes clamp to 0, lanes > 1 clamp to 1: feeding the
        // all-clamped corner must equal feeding [1,0,1] extremes through
        // the lmatrix only.
        let hard = run_fastpath_clipping(&[10.0, -5.0, 0.5, 1.0], &IDENTITY_CM, &IDENTITY_CM, &corr);
        let corner = xyz_to_lab([1.0, 0.0, 0.5 * corr[2]]);
        for c in 0..3 {
            assert!((hard[c] - corner[c]).abs() < 1e-5, "c={c} {} vs {}", hard[c], corner[c]);
        }
    }

    #[test]
    fn proper_simple_linear_matches_fastpath_simple_bit_exact() {
        // Linear profile (sentinel LUT) + identity matrix: tone curves are a
        // passthrough, so this must equal the linear fastpath bit-exactly.
        let lut = make_lut(true);
        let uc = [0.0f32; 9];
        let corr = flat_corr();
        let px = [0.3, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.8, 0.7, 1.0];
        let a = run_proper_simple(&px, &lut, &uc, &IDENTITY_CM, &corr);
        let b = run_fastpath_simple(&px, &corr, &IDENTITY_CM);
        for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "lane {i}: {x} vs {y}");
        }
    }

    #[test]
    fn proper_simple_tone_curve_on_off_and_exp() {
        // Identity ramp LUT ≈ linear path (lerp precision only).
        let lin = make_lut(true);
        let ramp = make_lut(false);
        let uc = [0.0f32; 9];
        let corr = flat_corr();
        let px = [0.3, 0.5, 0.2, 1.0];
        let a = run_proper_simple(&px, &ramp, &uc, &IDENTITY_CM, &corr);
        let b = run_proper_simple(&px, &lin, &uc, &IDENTITY_CM, &corr);
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() < 1e-3, "c={c} {} vs {}", a[c], b[c]);
        }
        // v >= 1 takes the unbounded exp fit: coeff (1,2,1) doubles the value.
        let uc2 = [1.0f32, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0];
        let out = run_proper_simple(&[2.0, 2.0, 2.0, 1.0], &ramp, &uc2, &IDENTITY_CM, &corr);
        let exp = xyz_to_lab([4.0, 4.0, 4.0]);
        for c in 0..3 {
            assert!((out[c] - exp[c]).abs() < 1e-4, "c={c} {} vs {}", out[c], exp[c]);
        }
        // Negative input through an active LUT clips to lut[0] (C MAX(v,0)):
        // must equal input 0.0 bit-exactly for a ramp starting at 0.
        let neg = run_proper_simple(&[-0.5, -0.5, -0.5, 1.0], &ramp, &uc, &IDENTITY_CM, &corr);
        let zero = run_proper_simple(&[0.0, 0.0, 0.0, 1.0], &ramp, &uc, &IDENTITY_CM, &corr);
        for (i, (&x, &y)) in neg.iter().zip(&zero).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "lane {i}: {x} vs {y}");
        }
        // ... while a linear (sentinel) channel passes negatives through.
        let neg_lin = run_proper_simple(&[-0.5, 0.25, 0.25, 1.0], &lin, &uc, &IDENTITY_CM, &corr);
        let exp_lin = xyz_to_lab([-0.5, 0.25, 0.25]);
        for c in 0..3 {
            assert!((neg_lin[c] - exp_lin[c]).abs() < 1e-5, "c={c} {}", neg_lin[c]);
        }
    }

    #[test]
    fn apply_tone_curves_boundary_and_mixed_sentinel() {
        // v == 1.0 takes the unbounded exp branch, not the LUT branch.
        // Channels are mixed: ch0 active, ch1 linear sentinel, ch2 active.
        let mut lut = make_lut(false);
        lut[LUT_SAMPLES] = -1.0;
        let uc = [1.0f32, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0];
        let mut cam = [1.0f32, 0.5, -0.25, 9.0];
        apply_tone_curves(&mut cam, &lut, &uc);
        assert_eq!(cam[0].to_bits(), 2.0f32.to_bits());
        assert_eq!(cam[1].to_bits(), 0.5f32.to_bits());
        assert_eq!(cam[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(cam[3].to_bits(), 9.0f32.to_bits());
    }

    #[test]
    fn proper_clipping_index_clip_and_agreement() {
        // Same non-identity setup as the bm index test: linear profile,
        // nmatrix = diag(2,1,1), lmatrix row1 with a 0.1 cross term.
        let nm: [f32; 16] = [
            2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let lm: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let lin = make_lut(true);
        let uc = [0.0f32; 9];
        let corr = flat_corr();
        let out = run_proper_clipping(&[0.6, 0.3, 0.2, 1.0], &lin, &uc, &nm, &lm, &corr);
        let expected = xyz_to_lab([1.0, 0.4, 0.2]);
        for c in 0..3 {
            assert!((out[c] - expected[c]).abs() < 1e-4, "c={c} {} vs {}", out[c], expected[c]);
        }
        // Clipping on/off: input 2.0 clipped (identity n/l) == input 1.0 plain.
        let clipped = run_proper_clipping(&[2.0, 2.0, 2.0, 1.0], &lin, &uc, &IDENTITY_CM, &IDENTITY_CM, &corr);
        let plain = run_proper_simple(&[1.0, 1.0, 1.0, 1.0], &lin, &uc, &IDENTITY_CM, &corr);
        for c in 0..3 {
            assert!((clipped[c] - plain[c]).abs() < 1e-4, "c={c} {} vs {}", clipped[c], plain[c]);
        }
        // No-clip agreement: identity n/l, in-gamut input → bit-exact vs simple.
        let px = [0.3, 0.5, 0.2, 1.0];
        let c = run_proper_clipping(&px, &lin, &uc, &IDENTITY_CM, &IDENTITY_CM, &corr);
        let s = run_proper_simple(&px, &lin, &uc, &IDENTITY_CM, &corr);
        for (i, (&x, &y)) in c.iter().zip(&s).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "lane {i}: {x} vs {y}");
        }
        // Tone curves apply before the clip: ramp LUT ≈ linear here.
        let ramp = make_lut(false);
        let c_ramp = run_proper_clipping(&px, &ramp, &uc, &IDENTITY_CM, &IDENTITY_CM, &corr);
        for ch in 0..3 {
            assert!((c_ramp[ch] - s[ch]).abs() < 1e-3, "ch={ch} {} vs {}", c_ramp[ch], s[ch]);
        }
    }

    #[test]
    fn clipping_variants_ffi_guards() {
        // npixels == 0 with null pointers: must return, not trap.
        unsafe {
            darkroom_colorin_cmatrix_fastpath_clipping(
                std::ptr::null(), std::ptr::null_mut(), 0,
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
            );
            darkroom_colorin_cmatrix_proper_simple(
                std::ptr::null(), std::ptr::null_mut(), 0,
                std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
            );
            darkroom_colorin_cmatrix_proper_clipping(
                std::ptr::null(), std::ptr::null_mut(), 0,
                std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(),
            );
        }
        // Null input with npixels > 0: no-op, output buffer untouched.
        let nm = IDENTITY_CM;
        let lut = make_lut(true);
        let uc = [0.0f32; 9];
        let corr = flat_corr();
        let mut out = vec![7.0f32; 4];
        unsafe {
            darkroom_colorin_cmatrix_fastpath_clipping(
                std::ptr::null(), out.as_mut_ptr(), 1, nm.as_ptr(), nm.as_ptr(), corr.as_ptr(),
            );
            assert_eq!(out, vec![7.0; 4]);
            darkroom_colorin_cmatrix_proper_simple(
                std::ptr::null(), out.as_mut_ptr(), 1, lut.as_ptr(), uc.as_ptr(), nm.as_ptr(),
                corr.as_ptr(),
            );
            assert_eq!(out, vec![7.0; 4]);
            darkroom_colorin_cmatrix_proper_clipping(
                std::ptr::null(), out.as_mut_ptr(), 1, lut.as_ptr(), uc.as_ptr(), nm.as_ptr(),
                nm.as_ptr(), corr.as_ptr(),
            );
            assert_eq!(out, vec![7.0; 4]);
            // Null matrix / lut / corr guards likewise leave output alone.
            let input = vec![0.5f32; 4];
            darkroom_colorin_cmatrix_fastpath_clipping(
                input.as_ptr(), out.as_mut_ptr(), 1, std::ptr::null(), nm.as_ptr(), corr.as_ptr(),
            );
            assert_eq!(out, vec![7.0; 4]);
            darkroom_colorin_cmatrix_proper_simple(
                input.as_ptr(), out.as_mut_ptr(), 1, std::ptr::null(), uc.as_ptr(), nm.as_ptr(),
                corr.as_ptr(),
            );
            assert_eq!(out, vec![7.0; 4]);
            darkroom_colorin_cmatrix_proper_clipping(
                input.as_ptr(), out.as_mut_ptr(), 1, lut.as_ptr(), uc.as_ptr(), nm.as_ptr(),
                nm.as_ptr(), std::ptr::null(),
            );
            assert_eq!(out, vec![7.0; 4]);
            // Dimension-product overflow is rejected before any slice exists;
            // these dangling pointers are never dereferenced.
            let dangling = std::ptr::NonNull::<f32>::dangling().as_ptr();
            darkroom_colorin_cmatrix_fastpath_clipping(
                dangling,
                dangling as *mut f32,
                usize::MAX,
                dangling,
                dangling,
                dangling,
            );
        }
    }
}
