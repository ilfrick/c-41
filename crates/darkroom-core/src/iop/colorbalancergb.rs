//! `colorbalancergb` — the gamut-boundary LUT builders (m4-83), first slice of
//! porting darktable's most complex IOP (`src/iop/colorbalancergb.c`).
//!
//! Both `saturation_formula` paths need a `hue → max-saturation/colorfulness` LUT
//! (`LUT_ELEM` entries over `[-π, π)`) that the process loop's `lookup_gamut`
//! reads to gamut-map. There are two builders, chosen by the formula:
//! - [`build_gamut_lut_jzazbz`] — samples a `STEPS³` RGB cube through the working
//!   profile into JzAzBz, keeping the max saturation per hue bin, then a 5-tap box
//!   anti-alias (port of the `commit_params` JzAzBz branch, colorbalancergb.c:1196).
//! - [`build_gamut_lut_ucs`] — marches the RGB gamut *boundary* in CIE xyY and
//!   records the dt-UCS colourfulness² per hue bin (port of
//!   `dt_UCS_22_build_gamut_LUT`, darktable_ucs_22_helpers.h).
//!
//! Both take the **RGB → XYZ D65** matrix in transposed form (the C premultiplies
//! `XYZ_D50→D65_CAT16 · work_profile->matrix_in`; for the Rust pipeline the
//! working space is known Rec.2020, giving Rec.2020→XYZ D65 directly). `commit_params`
//! (m4-84) will derive that matrix and pick the builder; the per-pixel process
//! loop (m4-85) will be wired as a `pipeline::Stage`.

use crate::color::{
    apply_transposed_color_matrix, d65_xyz_to_xyy, dt_ucs_hcb_to_jch, dt_ucs_hsb_to_jch,
    dt_ucs_jch_to_hcb, dt_ucs_jch_to_hsb, dt_ucs_jch_to_xyy, gamut_check_yrg, grading_rgb_to_lms,
    jzazbz_to_xyz_d65, lms_2006_to_xyz, lms_to_grading_rgb, lms_to_yrg, lookup_gamut, make_ych,
    opacity_masks, rec2020_to_xyz_d65, soft_clip, xyy_to_dt_ucs_jch, xyy_to_dt_ucs_uv, xyy_to_xyz,
    xyz_d65_to_rec2020, xyz_to_jzazbz, xyz_to_lms_2006, y_to_dt_ucs_l_star, ych_to_grading_rgb,
    ych_to_yrg, yrg_to_lms, yrg_to_ych, LUT_ELEM,
};

/// RGB-cube sampling resolution for the JzAzBz gamut LUT (`#define STEPS 92`).
const STEPS: usize = 92;

/// Hue-wheel shift: Filmlight Yrg puts red at 330° vs the usual 360/0°
/// (`#define ANGLE_SHIFT -30.f`).
const ANGLE_SHIFT: f32 = -30.0;

/// `deg2radf`.
#[inline]
fn deg2rad(x: f32) -> f32 {
    x * core::f32::consts::PI / 180.0
}

/// `CONVENTIONAL_DEG_TO_YRG_RAD`: shift a conventional-wheel hue (degrees) onto
/// the Filmlight Yrg wheel and convert to radians.
#[inline]
fn conv_deg_to_yrg_rad(x: f32) -> f32 {
    deg2rad(x + ANGLE_SHIFT)
}

/// The saturation/gamut-mapping formula (`dt_iop_colorbalancrgb_saturation_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaturationFormula {
    /// JzAzBz (2021).
    Jzazbz,
    /// darktable UCS (2022) — the default.
    DtUcs,
}

/// User-facing colorbalancergb parameters (`dt_iop_colorbalancergb_params_t`,
/// v5). Hue fields are in degrees `[0, 360)`, chroma `[0, 1]`, the ±1 sliders as
/// stored. GUI-only fields (checker colours/size) are omitted.
#[derive(Clone, Copy, Debug)]
pub struct CbRgbParams {
    pub shadows_y: f32,
    pub shadows_c: f32,
    pub shadows_h: f32,
    pub midtones_y: f32,
    pub midtones_c: f32,
    pub midtones_h: f32,
    pub highlights_y: f32,
    pub highlights_c: f32,
    pub highlights_h: f32,
    pub global_y: f32,
    pub global_c: f32,
    pub global_h: f32,
    pub shadows_weight: f32,
    pub white_fulcrum: f32,
    pub highlights_weight: f32,
    pub chroma_shadows: f32,
    pub chroma_highlights: f32,
    pub chroma_global: f32,
    pub chroma_midtones: f32,
    pub saturation_global: f32,
    pub saturation_highlights: f32,
    pub saturation_midtones: f32,
    pub saturation_shadows: f32,
    pub hue_angle: f32,
    pub brilliance_global: f32,
    pub brilliance_highlights: f32,
    pub brilliance_midtones: f32,
    pub brilliance_shadows: f32,
    pub mask_grey_fulcrum: f32,
    pub vibrance: f32,
    pub grey_fulcrum: f32,
    pub contrast: f32,
    pub saturation_formula: SaturationFormula,
}

impl Default for CbRgbParams {
    /// The darktable `$DEFAULT`s (a neutral, no-op edit).
    fn default() -> Self {
        Self {
            shadows_y: 0.0, shadows_c: 0.0, shadows_h: 0.0,
            midtones_y: 0.0, midtones_c: 0.0, midtones_h: 0.0,
            highlights_y: 0.0, highlights_c: 0.0, highlights_h: 0.0,
            global_y: 0.0, global_c: 0.0, global_h: 0.0,
            shadows_weight: 1.0, white_fulcrum: 0.0, highlights_weight: 1.0,
            chroma_shadows: 0.0, chroma_highlights: 0.0, chroma_global: 0.0, chroma_midtones: 0.0,
            saturation_global: 0.0, saturation_highlights: 0.0, saturation_midtones: 0.0,
            saturation_shadows: 0.0,
            hue_angle: 0.0,
            brilliance_global: 0.0, brilliance_highlights: 0.0, brilliance_midtones: 0.0,
            brilliance_shadows: 0.0,
            mask_grey_fulcrum: 0.1845,
            vibrance: 0.0, grey_fulcrum: 0.1845, contrast: 0.0,
            saturation_formula: SaturationFormula::DtUcs,
        }
    }
}

/// Derived per-commit data (`dt_iop_colorbalancergb_data_t`), ready for the
/// per-pixel process loop. GUI-only fields (checker, `lut_inited`, `work_profile`)
/// are omitted; `max_chroma` is unused by the process loop.
#[derive(Clone)]
pub struct CbRgbData {
    pub global: [f32; 4],
    pub shadows: [f32; 4],
    pub highlights: [f32; 4],
    pub midtones: [f32; 4],
    pub midtones_y: f32,
    pub chroma_global: f32,
    pub chroma: [f32; 4],
    pub vibrance: f32,
    pub contrast: f32,
    pub saturation_global: f32,
    pub saturation: [f32; 4],
    pub brilliance_global: f32,
    pub brilliance: [f32; 4],
    pub hue_angle: f32,
    pub shadows_weight: f32,
    pub highlights_weight: f32,
    pub midtones_weight: f32,
    pub mask_grey_fulcrum: f32,
    pub white_fulcrum: f32,
    pub grey_fulcrum: f32,
    pub gamut_lut: [f32; LUT_ELEM],
    pub saturation_formula: SaturationFormula,
}

#[inline]
fn sqf(x: f32) -> f32 {
    x * x
}

impl CbRgbData {
    /// Derive the process-ready data from user params (port of `commit_params`,
    /// colorbalancergb.c:1087). `rgb_to_xyz_d65_t` is the working profile's
    /// **transposed** RGB→XYZ-D65 matrix (pass `color::REC2020_TO_XYZ_D65_T4` for
    /// the pipeline's Rec.2020 working space) — used only to build the gamut LUT.
    pub fn from_params(p: &CbRgbParams, rgb_to_xyz_d65_t: &[[f32; 4]; 4]) -> Self {
        // measure the grading RGB of a pure white (achromatic reference)
        let rgb_norm = ych_to_grading_rgb([1.0, 0.0, 1.0, 0.0]);

        // global: offset around the achromatic reference, scaled by global_Y
        let mut global = ych_to_grading_rgb(make_ych(1.0, p.global_c, conv_deg_to_yrg_rad(p.global_h)));
        for (g, n) in global.iter_mut().zip(rgb_norm.iter()) {
            *g = (*g - n) + n * p.global_y;
        }

        // shadows / highlights: 1 + offset + luminance; weight = 2 + w·2
        let mut shadows =
            ych_to_grading_rgb(make_ych(1.0, p.shadows_c, conv_deg_to_yrg_rad(p.shadows_h)));
        for (s, n) in shadows.iter_mut().zip(rgb_norm.iter()) {
            *s = 1.0 + (*s - n) + p.shadows_y;
        }
        let shadows_weight = 2.0 + p.shadows_weight * 2.0;

        let mut highlights =
            ych_to_grading_rgb(make_ych(1.0, p.highlights_c, conv_deg_to_yrg_rad(p.highlights_h)));
        for (h, n) in highlights.iter_mut().zip(rgb_norm.iter()) {
            *h = 1.0 + (*h - n) + p.highlights_y;
        }
        let highlights_weight = 2.0 + p.highlights_weight * 2.0;

        // midtones: reciprocal power base 1/(1+offset)
        let mut midtones =
            ych_to_grading_rgb(make_ych(1.0, p.midtones_c, conv_deg_to_yrg_rad(p.midtones_h)));
        for (m, n) in midtones.iter_mut().zip(rgb_norm.iter()) {
            *m = 1.0 / (1.0 + (*m - n));
        }
        let midtones_y = 1.0 / (1.0 + p.midtones_y);
        let white_fulcrum = p.white_fulcrum.exp2();
        let midtones_weight = sqf(shadows_weight) * sqf(highlights_weight)
            / (sqf(shadows_weight) + sqf(highlights_weight));
        let mask_grey_fulcrum = p.mask_grey_fulcrum.powf(0.4101205819200422);

        // gamut LUT, per the selected saturation formula
        let gamut_lut = match p.saturation_formula {
            SaturationFormula::Jzazbz => build_gamut_lut_jzazbz(rgb_to_xyz_d65_t),
            SaturationFormula::DtUcs => build_gamut_lut_ucs(rgb_to_xyz_d65_t),
        };

        Self {
            global,
            shadows,
            highlights,
            midtones,
            midtones_y,
            chroma_global: p.chroma_global,
            chroma: [p.chroma_shadows, p.chroma_midtones, p.chroma_highlights, 0.0],
            vibrance: p.vibrance,
            contrast: 1.0 + p.contrast,
            saturation_global: p.saturation_global,
            saturation: [p.saturation_shadows, p.saturation_midtones, p.saturation_highlights, 0.0],
            brilliance_global: p.brilliance_global,
            brilliance: [p.brilliance_shadows, p.brilliance_midtones, p.brilliance_highlights, 0.0],
            hue_angle: deg2rad(p.hue_angle),
            shadows_weight,
            highlights_weight,
            midtones_weight,
            mask_grey_fulcrum,
            white_fulcrum,
            grey_fulcrum: p.grey_fulcrum,
            gamut_lut,
            saturation_formula: p.saturation_formula,
        }
    }
}

/// D65 white point in xyY (`D65xyY = {0.31271, 0.32902, 1.0}`, colorspaces.h:38).
const D65_X: f32 = 0.31271;
const D65_Y: f32 = 0.32902;

/// Hue-bin index for `hue ∈ [-π, π)` with the C's `roundf` + cyclic wrap.
#[inline]
fn hue_index(hue: f32) -> usize {
    let mut index = ((LUT_ELEM - 1) as f32 * (hue + core::f32::consts::PI)
        / core::f32::consts::TAU)
        .round() as i32;
    index += if index < 0 { LUT_ELEM as i32 } else { 0 };
    index -= if index >= LUT_ELEM as i32 { LUT_ELEM as i32 } else { 0 };
    index as usize
}

/// Angle difference `h_1 - h_2` folded into `[-π, π]`. Port of `Delta_H()`.
#[inline]
fn delta_h(h_1: f32, h_2: f32) -> f32 {
    let mut diff = h_1 - h_2;
    diff += if diff < -core::f32::consts::PI { core::f32::consts::TAU } else { 0.0 };
    diff -= if diff > core::f32::consts::PI { core::f32::consts::TAU } else { 0.0 };
    diff
}

/// Build the gamut LUT for the JzAzBz saturation formula. `input_matrix_t` is the
/// **transposed** RGB → XYZ D65 matrix. Port of the JzAzBz branch of
/// `commit_params` (colorbalancergb.c:1196–1234).
pub fn build_gamut_lut_jzazbz(input_matrix_t: &[[f32; 4]; 4]) -> [f32; LUT_ELEM] {
    let mut sampler = [0.0f32; LUT_ELEM];
    let denom = (STEPS - 1) as f32;
    for r in 0..STEPS {
        for g in 0..STEPS {
            for b in 0..STEPS {
                let rgb = [r as f32 / denom, g as f32 / denom, b as f32 / denom, 0.0];
                let xyz = apply_transposed_color_matrix(&rgb, input_matrix_t);
                let jab = xyz_to_jzazbz(xyz);
                // JCh: [J, chroma, hue]
                let j = jab[0];
                // dt_fast_hypotf: under darktable's release `-ffast-math` build this
                // is `sqrtf(x²+y²)` (not libm `hypotf`); parity assumes that variant.
                let chroma = (jab[2] * jab[2] + jab[1] * jab[1]).sqrt();
                let hue = jab[2].atan2(jab[1]);
                let saturation = if j > 0.0 { chroma / j } else { 0.0 };
                let idx = hue_index(hue);
                sampler[idx] = sampler[idx].max(saturation);
            }
        }
    }

    // 5-tap box anti-alias, with cyclic bounds.
    let mut lut = [0.0f32; LUT_ELEM];
    let n = LUT_ELEM;
    for k in 2..n - 2 {
        lut[k] = (sampler[k - 2] + sampler[k - 1] + sampler[k] + sampler[k + 1] + sampler[k + 2]) / 5.0;
    }
    lut[0] = (sampler[n - 2] + sampler[n - 1] + sampler[0] + sampler[1] + sampler[2]) / 5.0;
    lut[1] = (sampler[n - 1] + sampler[0] + sampler[1] + sampler[2] + sampler[3]) / 5.0;
    lut[n - 1] = (sampler[n - 3] + sampler[n - 2] + sampler[n - 1] + sampler[0] + sampler[1]) / 5.0;
    lut[n - 2] = (sampler[n - 4] + sampler[n - 3] + sampler[n - 2] + sampler[n - 1] + sampler[0]) / 5.0;
    lut
}

/// Build the gamut LUT for the dt-UCS saturation formula by marching the RGB gamut
/// boundary in CIE xyY. `input_matrix_t` is the **transposed** RGB → XYZ D65
/// matrix. Port of `dt_UCS_22_build_gamut_LUT()`. Stores **colourfulness²** (M²).
pub fn build_gamut_lut_ucs(input_matrix_t: &[[f32; 4]; 4]) -> [f32; LUT_ELEM] {
    let mut gamut = [0.0f32; LUT_ELEM];
    let mut sampler = [0.0f32; LUT_ELEM];

    let d65 = [D65_X, D65_Y];

    // RGB primaries → xyY
    let xyz_red = apply_transposed_color_matrix(&[1.0, 0.0, 0.0, 0.0], input_matrix_t);
    let xyz_green = apply_transposed_color_matrix(&[0.0, 1.0, 0.0, 0.0], input_matrix_t);
    let xyz_blue = apply_transposed_color_matrix(&[0.0, 0.0, 1.0, 0.0], input_matrix_t);
    let xyy_red = d65_xyz_to_xyy(&xyz_red);
    let xyy_green = d65_xyz_to_xyy(&xyz_green);
    let xyy_blue = d65_xyz_to_xyy(&xyz_blue);

    // primary "hue" angles in xy relative to D65
    let h_red = (xyy_red[1] - d65[1]).atan2(xyy_red[0] - d65[0]);
    let h_green = (xyy_green[1] - d65[1]).atan2(xyy_green[0] - d65[0]);
    let h_blue = (xyy_blue[1] - d65[1]).atan2(xyy_blue[0] - d65[0]);

    // march the gamut boundary by angular steps of 0.02°
    let steps = 50 * LUT_ELEM;
    for i in 0..steps {
        let angle = -core::f32::consts::PI + (i as f32) / (steps as f32) * core::f32::consts::TAU;
        let tan_angle = angle.tan();

        let t_1 = delta_h(angle, h_blue) / delta_h(h_red, h_blue);
        let t_2 = delta_h(angle, h_red) / delta_h(h_green, h_red);
        let t_3 = delta_h(angle, h_green) / delta_h(h_blue, h_green);

        let (mut x_t, mut y_t) = (0.0f32, 0.0f32);

        // pick the boundary edge whose barycentric parameter lands in [0, 1]
        if t_1 == t_1.clamp(0.0, 1.0) {
            let t = (d65[1] - xyy_blue[1] + tan_angle * (xyy_blue[0] - d65[0]))
                / (xyy_red[1] - xyy_blue[1] + tan_angle * (xyy_blue[0] - xyy_red[0]));
            x_t = xyy_blue[0] + t * (xyy_red[0] - xyy_blue[0]);
            y_t = xyy_blue[1] + t * (xyy_red[1] - xyy_blue[1]);
        } else if t_2 == t_2.clamp(0.0, 1.0) {
            let t = (d65[1] - xyy_red[1] + tan_angle * (xyy_red[0] - d65[0]))
                / (xyy_green[1] - xyy_red[1] + tan_angle * (xyy_red[0] - xyy_green[0]));
            x_t = xyy_red[0] + t * (xyy_green[0] - xyy_red[0]);
            y_t = xyy_red[1] + t * (xyy_green[1] - xyy_red[1]);
        } else if t_3 == t_3.clamp(0.0, 1.0) {
            let t = (d65[1] - xyy_green[1] + tan_angle * (xyy_green[0] - d65[0]))
                / (xyy_blue[1] - xyy_green[1] + tan_angle * (xyy_green[0] - xyy_blue[0]));
            x_t = xyy_green[0] + t * (xyy_blue[0] - xyy_green[0]);
            y_t = xyy_green[1] + t * (xyy_blue[1] - xyy_green[1]);
        }

        // to dt UCS UV*'
        let uv = xyy_to_dt_ucs_uv(&[x_t, y_t, 1.0, 0.0]);
        let hue = uv[1].atan2(uv[0]);
        let idx = hue_index(hue);
        // store M² (colourfulness squared), averaged over the bin's samples
        gamut[idx] += uv[0] * uv[0] + uv[1] * uv[1];
        sampler[idx] += 1.0;
    }

    // NB: the C marches this with an OMP `reduction(+:)`, whose FP add order is
    // thread-count-dependent — so darktable's own UCS LUT is not bit-reproducible
    // across thread counts. This serial port matches single-threaded C exactly;
    // any future golden dump must be captured with OMP_NUM_THREADS=1, and a rayon
    // parallelisation here would not stay bit-identical to this serial version.
    for k in 0..LUT_ELEM {
        gamut[k] /= sampler[k].max(1.0);
    }
    gamut
}

/// Dot product over the 4 lanes (`scalar_product`); the mask/zone vectors always
/// carry 0 in lane 3, so this equals the C's 3-wide sum.
#[inline]
fn dot4(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// The "center middle grey in 50 %" exponent shared by the opacity masks and the
/// mask fulcrum (colorbalancergb.c:699, 1167).
const GREY_CENTER_EXP: f32 = 0.4101205819200422;

/// JzAzBz perceptual saturation/brilliance branch (colorbalancergb.c:768–850).
/// Takes and returns XYZ D65.
fn saturation_jzazbz(xyz_d65: [f32; 4], opacities: &[f32; 4], d: &CbRgbData) -> [f32; 4] {
    use core::f32::consts::FRAC_PI_2;

    let jab = xyz_to_jzazbz(xyz_d65);
    // JCh: brightness/chroma vector + hue angle
    let mut jc = [jab[0], (jab[1] * jab[1] + jab[2] * jab[2]).sqrt()]; // dt_fast_hypotf(Jab[1],Jab[2])
    let h = jab[2].atan2(jab[1]);

    // rotate onto the saturation eigenvector S (angle T over the hue plane)
    let t = jc[1].atan2(jc[0]);
    let (sin_t, cos_t) = (t.sin(), t.cos());
    let boosts = [
        1.0 + d.brilliance_global + dot4(opacities, &d.brilliance), // S direction
        d.saturation_global + dot4(opacities, &d.saturation),       // O direction
    ];
    // M_rot_dir[0] = {cos_T, sin_T}
    let so0 = jc[0] * cos_t + jc[1] * sin_t;
    let so1 = so0 * (t * boosts[1]).clamp(-t, FRAC_PI_2 - t); // MIN(MAX(T·b,-T), π/2-T)
    let so0 = (so0 * boosts[0]).max(0.0);
    // rotate back by -T (M_rot_inv rows {cos_T,-sin_T},{sin_T,cos_T})
    jc[0] = (so0 * cos_t - so1 * sin_t).max(0.0);
    jc[1] = (so0 * sin_t + so1 * cos_t).max(0.0);

    // gamut mapping toward the boundary at this hue
    let out_max_sat_h = lookup_gamut(&d.gamut_lut, h);
    let sat = if jc[0] > 0.0 {
        soft_clip(jc[1] / jc[0], 0.8 * out_max_sat_h, out_max_sat_h)
    } else {
        out_max_sat_h
    };
    let max_c_at_sat = jc[0] * sat;
    let max_j_at_sat = if sat > 0.0 { jc[1] / sat } else { jc[0] };
    jc[0] = (jc[0] + max_j_at_sat) / 2.0;
    jc[1] = (jc[1] + max_c_at_sat) / 2.0;

    // gamut-clip in Jch at constant hue+lightness (avoid negative L'M'S')
    let (cos_hh, sin_hh) = (h.cos(), h.sin());
    let d0 = 1.6295499532821566e-11f32;
    let dd = -0.56f32;
    let mut iz = jc[0] + d0;
    iz /= 1.0 + dd - dd * iz;
    iz = iz.max(0.0);

    // Izazbz → L'M'S' test matrix (transposed), colorbalancergb.c:824–827
    const AI_TRANS: [[f32; 4]; 4] = [
        [1.0, 1.0, 1.0, 0.0],
        [0.1386050432715393, -0.1386050432715393, -0.0960192420263190, 0.0],
        [0.0580473161561189, -0.0580473161561189, -0.8118918960560390, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let izazbz = [iz, jc[1] * cos_hh, jc[1] * sin_hh, 0.0];
    let lms = apply_transposed_color_matrix(&izazbz, &AI_TRANS);

    let mut max_c = jc[1];
    if lms[0] < 0.0 {
        max_c = (-iz / (AI_TRANS[1][0] * cos_hh + AI_TRANS[2][0] * sin_hh)).min(max_c);
    }
    if lms[1] < 0.0 {
        max_c = (-iz / (AI_TRANS[1][1] * cos_hh + AI_TRANS[2][1] * sin_hh)).min(max_c);
    }
    if lms[2] < 0.0 {
        max_c = (-iz / (AI_TRANS[1][2] * cos_hh + AI_TRANS[2][2] * sin_hh)).min(max_c);
    }

    let jab_out = [jc[0], max_c * cos_hh, max_c * sin_hh, 0.0];
    jzazbz_to_xyz_d65(jab_out)
}

/// darktable-UCS perceptual saturation/brilliance branch (colorbalancergb.c:851–897).
/// Takes and returns XYZ D65. `l_white` is `Y_to_dt_UCS_L_star(white_fulcrum)`.
fn saturation_dtucs(xyz_d65: [f32; 4], opacities: &[f32; 4], d: &CbRgbData, l_white: f32) -> [f32; 4] {
    let xyy = d65_xyz_to_xyy(&xyz_d65);
    let jch = xyy_to_dt_ucs_jch(&xyy, l_white);
    let mut hcb = dt_ucs_jch_to_hcb(jch);

    let radius = (hcb[1] * hcb[1] + hcb[2] * hcb[2]).sqrt(); // dt_fast_hypotf(HCB[1],HCB[2])
    let sin_t = if radius > 0.0 { hcb[1] / radius } else { 0.0 };
    let cos_t = if radius > 0.0 { hcb[2] / radius } else { 0.0 };

    let p = f32::MIN_POSITIVE.max(hcb[1]); // MAX(FLT_MIN, HCB[1])
    let w = sin_t * hcb[1] + cos_t * hcb[2];

    let mut a = (1.0 + d.saturation_global + dot4(opacities, &d.saturation)).max(0.0);
    let b = (1.0 + d.brilliance_global + dot4(opacities, &d.brilliance)).max(0.0);
    let max_a = (p * p + w * w).sqrt() / p; // dt_fast_hypotf(P,W)/P
    a = soft_clip(a, 0.5 * max_a, max_a);

    let p_prime = (a - 1.0) * p;
    let w_prime = (p * p * (1.0 - a * a) + w * w).sqrt() * b;
    // M_rot_inv rows {cos_T, sin_T}, {-sin_T, cos_T}
    hcb[1] = (cos_t * p_prime + sin_t * w_prime).max(0.0);
    hcb[2] = (-sin_t * p_prime + cos_t * w_prime).max(0.0);

    let jch = dt_ucs_hcb_to_jch(hcb);

    // gamut mapping (max_colorfulness is M²)
    let max_colorfulness = lookup_gamut(&d.gamut_lut, jch[2]);
    let max_chroma = 15.932993652962535
        * (jch[0] * l_white).powf(0.6523997524738018)
        * max_colorfulness.powf(0.6007557017508491)
        / l_white;
    let hsb_boundary = dt_ucs_jch_to_hsb(&[jch[0], max_chroma, jch[2], 0.0]);

    // clip saturation at constant brightness
    let mut hsb = [hcb[0], if hcb[2] > 0.0 { hcb[1] / hcb[2] } else { 0.0 }, hcb[2], 0.0];
    hsb[1] = soft_clip(hsb[1], 0.8 * hsb_boundary[1], hsb_boundary[1]);

    let jch = dt_ucs_hsb_to_jch(&hsb);
    let xyy = dt_ucs_jch_to_xyy(&jch, l_white);
    xyy_to_xyz(&xyy)
}

/// Apply colorbalancergb to a packed-RGBA scene-linear **Rec.2020** buffer
/// (`input`/`output` are `n·4` floats). Faithful port of the process loop
/// (colorbalancergb.c:662–943), sans the GUI mask-display checkerboard.
///
/// The C's premultiplied pipeline↔LMS matrices become direct compositions of the
/// pipeline's fixed Rec.2020↔XYZ-D65 conversions. Alpha (lane 3) is preserved
/// from the input (the color chain drops it via the matrices, as in the C, but the
/// pipeline keeps it rather than emitting 0).
pub fn process(input: &[f32], output: &mut [f32], d: &CbRgbData) {
    let (hue_cos, hue_sin) = (d.hue_angle.cos(), d.hue_angle.sin());
    let l_white = y_to_dt_ucs_l_star(d.white_fulcrum);

    for (i_px, o_px) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        // clip pipeline RGB
        let mut rgb = [i_px[0].max(0.0), i_px[1].max(0.0), i_px[2].max(0.0), 0.0];

        // → CIE 2006 LMS D65 → Filmlight Yrg → Ych
        let mut lms = xyz_to_lms_2006(rec2020_to_xyz_d65(rgb));
        let mut yrg = lms_to_yrg(lms);
        let mut ych = yrg_to_ych(yrg);
        ych[0] = ych[0].max(0.0); // no negative luminance

        // luma opacity masks
        let (opacities, opacities_comp) = opacity_masks(
            ych[0].powf(GREY_CENTER_EXP),
            d.shadows_weight,
            d.highlights_weight,
            d.midtones_weight,
            d.mask_grey_fulcrum,
        );

        // hue shift (2×2 rotation of the (cos h, sin h) vector)
        let (cos_h, sin_h) = (ych[2], ych[3]);
        ych[2] = hue_cos * cos_h - hue_sin * sin_h;
        ych[3] = hue_sin * cos_h + hue_cos * sin_h;

        // linear chroma: boost + vibrance at constant luminance
        let chroma_boost = d.chroma_global + dot4(&opacities, &d.chroma);
        let vibrance = d.vibrance * (1.0 - ych[1].powf(d.vibrance.abs()));
        let chroma_factor = (1.0 + chroma_boost + vibrance).max(0.0);
        ych[1] *= chroma_factor;

        // clip chroma to the Yrg/LMS cone, then → Filmlight grading RGB
        ych = gamut_check_yrg(ych);
        yrg = ych_to_yrg(ych);
        lms = yrg_to_lms(yrg);
        rgb = lms_to_grading_rgb(lms);

        // colour balance: global offset
        for (v, g) in rgb.iter_mut().zip(d.global.iter()) {
            *v += g;
        }
        // shadows/highlights: 2 masked slopes
        for c in 0..4 {
            rgb[c] *= opacities_comp[2] * (opacities_comp[0] + opacities[0] * d.shadows[c])
                + opacities[2] * d.highlights[c];
        }
        // midtones power (per-channel), sign-preserving around the white fulcrum
        for c in 0..4 {
            let sign = if rgb[c] < 0.0 { -1.0 } else { 1.0 };
            let scaled = rgb[c].abs() / d.white_fulcrum;
            rgb[c] = scaled.powf(d.midtones[c]) * sign * d.white_fulcrum;
        }

        // back to Yrg for the non-linear luma ops (RGB doesn't preserve colour)
        lms = grading_rgb_to_lms(rgb);
        yrg = lms_to_yrg(lms);
        // Y midtones power (gamma)
        yrg[0] = (yrg[0] / d.white_fulcrum).max(0.0).powf(d.midtones_y) * d.white_fulcrum;
        // Y fulcrumed contrast
        yrg[0] = d.grey_fulcrum * (yrg[0] / d.grey_fulcrum).powf(d.contrast);

        // → XYZ D65 → perceptual saturation adjustments
        lms = yrg_to_lms(yrg);
        let xyz_d65 = lms_2006_to_xyz(lms);
        let xyz_d65 = match d.saturation_formula {
            SaturationFormula::Jzazbz => saturation_jzazbz(xyz_d65, &opacities, d),
            SaturationFormula::DtUcs => saturation_dtucs(xyz_d65, &opacities, d, l_white),
        };

        // back to pipeline RGB, clip negatives, preserve alpha
        let pix = xyz_d65_to_rec2020(xyz_d65);
        o_px[0] = pix[0].max(0.0);
        o_px[1] = pix[1].max(0.0);
        o_px[2] = pix[2].max(0.0);
        o_px[3] = i_px[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // sRGB → XYZ (D65), transposed for `apply_transposed_color_matrix`.
    const SRGB_TO_XYZ_D65_T: [[f32; 4]; 4] = [
        [0.4124, 0.2126, 0.0193, 0.0],
        [0.3576, 0.7152, 0.1192, 0.0],
        [0.1805, 0.0722, 0.9505, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];

    fn finite_nonneg(lut: &[f32; LUT_ELEM]) -> bool {
        lut.iter().all(|v| v.is_finite() && *v >= 0.0)
    }

    fn has_variation(lut: &[f32; LUT_ELEM]) -> bool {
        let max = lut.iter().cloned().fold(f32::MIN, f32::max);
        let min = lut.iter().cloned().fold(f32::MAX, f32::min);
        max - min > 1e-4
    }

    #[test]
    fn jzazbz_lut_is_finite_nonneg_and_varies() {
        let lut = build_gamut_lut_jzazbz(&SRGB_TO_XYZ_D65_T);
        assert!(finite_nonneg(&lut), "jzazbz LUT has non-finite/negative entries");
        assert!(has_variation(&lut), "jzazbz LUT is flat — hue binning likely broken");
    }

    #[test]
    fn ucs_lut_is_finite_nonneg_and_varies() {
        let lut = build_gamut_lut_ucs(&SRGB_TO_XYZ_D65_T);
        assert!(finite_nonneg(&lut), "ucs LUT has non-finite/negative entries");
        assert!(has_variation(&lut), "ucs LUT is flat — boundary march likely broken");
    }

    #[test]
    fn builders_are_deterministic() {
        // pure functions of the input matrix.
        assert_eq!(
            build_gamut_lut_jzazbz(&SRGB_TO_XYZ_D65_T),
            build_gamut_lut_jzazbz(&SRGB_TO_XYZ_D65_T)
        );
        assert_eq!(
            build_gamut_lut_ucs(&SRGB_TO_XYZ_D65_T),
            build_gamut_lut_ucs(&SRGB_TO_XYZ_D65_T)
        );
    }

    #[test]
    fn hue_index_wraps_within_bounds() {
        // every angle (incl. the ±π endpoints and out-of-range) maps in-bounds.
        for k in 0..2000 {
            let a = -4.0 + (k as f32) * 8.0 / 2000.0;
            assert!(hue_index(a) < LUT_ELEM);
        }
    }

    #[test]
    fn delta_h_folds_into_pi_range() {
        use core::f32::consts::PI;
        // near +2π apart folds toward 0; result always in [-π, π].
        for &(a, b) in &[(3.0f32, -3.0f32), (-3.0, 3.0), (0.5, 0.2), (PI, -PI)] {
            let d = delta_h(a, b);
            assert!(d >= -PI - 1e-6 && d <= PI + 1e-6, "delta_h({a},{b})={d} out of range");
        }
    }

    // ── commit_params (m4-84) ──

    use crate::color::REC2020_TO_XYZ_D65_T4;

    #[test]
    fn neutral_params_derive_a_no_op_balance() {
        // default params = a no-op edit: offset 0, slope 1, power 1 on all zones.
        let d = CbRgbData::from_params(&CbRgbParams::default(), &REC2020_TO_XYZ_D65_T4);
        for c in 0..4 {
            assert!(d.global[c].abs() < 1e-5, "global[{c}]={} != 0", d.global[c]);
            assert!((d.shadows[c] - 1.0).abs() < 1e-5, "shadows[{c}]={}", d.shadows[c]);
            assert!((d.highlights[c] - 1.0).abs() < 1e-5, "highlights[{c}]={}", d.highlights[c]);
            assert!((d.midtones[c] - 1.0).abs() < 1e-5, "midtones[{c}]={}", d.midtones[c]);
        }
        // scalar derivations for the defaults
        assert!((d.contrast - 1.0).abs() < 1e-6); // 1 + 0
        assert!((d.midtones_y - 1.0).abs() < 1e-6); // 1/(1+0)
        assert!((d.white_fulcrum - 1.0).abs() < 1e-6); // 2^0
        assert!((d.shadows_weight - 4.0).abs() < 1e-6); // 2 + 1·2
        assert!((d.highlights_weight - 4.0).abs() < 1e-6);
        assert!((d.midtones_weight - 8.0).abs() < 1e-5); // 16·16/(16+16)
        assert!(d.gamut_lut.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn scalar_derivations_track_params() {
        let mut p = CbRgbParams::default();
        p.contrast = 0.5;
        p.midtones_y = 0.25;
        p.white_fulcrum = 2.0;
        p.shadows_weight = 0.5;
        p.highlights_weight = 3.0;
        let d = CbRgbData::from_params(&p, &REC2020_TO_XYZ_D65_T4);
        assert!((d.contrast - 1.5).abs() < 1e-6);
        assert!((d.midtones_y - 1.0 / 1.25).abs() < 1e-6);
        assert!((d.white_fulcrum - 4.0).abs() < 1e-6); // 2^2
        assert!((d.shadows_weight - 3.0).abs() < 1e-6); // 2 + 0.5·2
        assert!((d.highlights_weight - 8.0).abs() < 1e-6); // 2 + 3·2
        let sw2 = 3.0f32 * 3.0;
        let hw2 = 8.0f32 * 8.0;
        assert!((d.midtones_weight - sw2 * hw2 / (sw2 + hw2)).abs() < 1e-4);
    }

    #[test]
    fn saturation_formula_selects_the_builder() {
        let mut pj = CbRgbParams::default();
        pj.saturation_formula = SaturationFormula::Jzazbz;
        let dj = CbRgbData::from_params(&pj, &REC2020_TO_XYZ_D65_T4);
        let du = CbRgbData::from_params(&CbRgbParams::default(), &REC2020_TO_XYZ_D65_T4);
        // the two formulas build genuinely different LUTs
        let differ = dj.gamut_lut.iter().zip(du.gamut_lut.iter()).any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(differ, "JzAzBz and dt-UCS produced identical LUTs");
    }

    #[test]
    fn gamut_lut_actually_uses_the_input_matrix() {
        // a genuinely different gamut (sRGB primaries) must change the LUT — proves
        // the matrix argument is honoured (guards the transpose-contract wiring).
        // NB: the dt-UCS builder is chromaticity-based, so a *uniform scale* of the
        // matrix would NOT change it (same xyY) — the primaries must actually move.
        let p = CbRgbParams::default();
        let rec = CbRgbData::from_params(&p, &REC2020_TO_XYZ_D65_T4);
        let srgb = CbRgbData::from_params(&p, &SRGB_TO_XYZ_D65_T);
        let differ = rec.gamut_lut.iter().zip(srgb.gamut_lut.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differ, "LUT ignored the input matrix (Rec.2020 vs sRGB identical)");
    }

    // ── process loop (m4-85) ──

    fn neutral() -> CbRgbData {
        CbRgbData::from_params(&CbRgbParams::default(), &REC2020_TO_XYZ_D65_T4)
    }

    fn run_one(rgb: [f32; 3], d: &CbRgbData) -> [f32; 4] {
        let inp = [rgb[0], rgb[1], rgb[2], 1.0];
        let mut out = [0.0f32; 4];
        process(&inp, &mut out, d);
        out
    }

    #[test]
    fn neutral_preserves_grey_both_formulas() {
        // colorbalancergb always gamut-maps, but a neutral edit on an achromatic
        // pixel must keep it achromatic and (very nearly) the same luminance.
        for formula in [SaturationFormula::DtUcs, SaturationFormula::Jzazbz] {
            let mut p = CbRgbParams::default();
            p.saturation_formula = formula;
            let d = CbRgbData::from_params(&p, &REC2020_TO_XYZ_D65_T4);
            for &v in &[0.05f32, 0.18, 0.5, 0.9] {
                let o = run_one([v, v, v], &d);
                assert!(
                    (o[0] - o[1]).abs() < 1e-3 && (o[1] - o[2]).abs() < 1e-3,
                    "{formula:?}: grey drifted to colour: {o:?}"
                );
                assert!((o[0] - v).abs() < 3e-2, "{formula:?}: grey {v} -> {}", o[0]);
            }
        }
    }

    #[test]
    fn output_finite_nonneg_for_colours() {
        let du = neutral();
        let mut pj = CbRgbParams::default();
        pj.saturation_formula = SaturationFormula::Jzazbz;
        let dj = CbRgbData::from_params(&pj, &REC2020_TO_XYZ_D65_T4);
        for rgb in [[0.6, 0.2, 0.1], [0.1, 0.5, 0.3], [0.02, 0.03, 0.4], [0.9, 0.85, 0.1]] {
            for d in [&du, &dj] {
                let o = run_one(rgb, d);
                assert!(o.iter().all(|v| v.is_finite() && *v >= 0.0), "bad output {o:?} for {rgb:?}");
            }
        }
    }

    #[test]
    fn alpha_is_preserved() {
        let inp = [0.3, 0.4, 0.5, 0.42];
        let mut out = [0.0f32; 4];
        process(&inp, &mut out, &neutral());
        assert_eq!(out[3], 0.42);
    }

    #[test]
    fn parameters_change_the_output() {
        let base = run_one([0.3, 0.25, 0.2], &neutral());
        // brighten via global luminance
        let mut p = CbRgbParams::default();
        p.global_y = 0.5;
        let bright = run_one([0.3, 0.25, 0.2], &CbRgbData::from_params(&p, &REC2020_TO_XYZ_D65_T4));
        assert!(
            base.iter().zip(bright.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
            "global_Y had no effect"
        );
        // add global saturation
        let mut p2 = CbRgbParams::default();
        p2.saturation_global = 0.5;
        let sat = run_one([0.3, 0.25, 0.2], &CbRgbData::from_params(&p2, &REC2020_TO_XYZ_D65_T4));
        assert!(
            base.iter().zip(sat.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
            "saturation_global had no effect"
        );
    }

    #[test]
    fn processes_a_multi_pixel_buffer() {
        let d = neutral();
        let inp: Vec<f32> = (0..16)
            .flat_map(|i| {
                let v = i as f32 / 16.0;
                [v, v * 0.8, v * 0.6, 1.0]
            })
            .collect();
        let mut out = vec![0.0f32; inp.len()];
        process(&inp, &mut out, &d);
        assert_eq!(out.len(), inp.len());
        assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
}
