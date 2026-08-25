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
//! `XYZ_D50→D65_CAT16 · work_profile->matrix_in`; the Rust pipeline passes its
//! working space's matrix — Rec.2020 on the raw path, linear sRGB on the non-raw
//! path). `commit_params` (m4-84) derives that matrix and picks the builder; the
//! per-pixel process loop (m4-85) is wired as a `pipeline::Stage`.
//!
//! m4-137 adds the C-binary side: `darkroom_colorbalancergb_*` FFI exports that
//! replace the four remaining OMP loops in the C file — the process loop
//! (colorbalancergb.c:662), the JzAzBz gamut-LUT sampler (:1197), the GUI
//! checkerboard fill (:1511) and the GUI opacity-mask LUT build (:1555). The
//! process export mirrors the C's *premultiplied-matrix* form (one transposed
//! apply in/out, arbitrary pipe work profile), which is deliberately separate
//! from [`process_in_space`]'s fixed-space fn-pointer pair so the shipped Rust
//! pipeline keeps its exact current arithmetic. All four are serial scalar,
//! matching single-threaded C; the LUT sampler's `reduction(max:)` is
//! order-independent so serial ≡ parallel.

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
///
/// `PartialEq` backs the pipeline's identity gate: a params set equal to
/// [`CbRgbParams::default`] is darktable's neutral edit and emits no stage
/// (`to_pipeline` in c41-ui compares against exactly this value).
#[derive(Clone, Copy, Debug, PartialEq)]
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
///
/// `Debug`/`PartialEq` exist because the value travels inside a
/// `pipeline::Stage` (which derives both). The builders are pure and
/// deterministic (pinned by `builders_are_deterministic`), so two data sets
/// derived from equal params compare equal field-for-field.
#[derive(Clone, Debug, PartialEq)]
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

/// Working-space RGB↔XYZ(D65) converter pair for [`process_in_space`] — the
/// alpha-preserving `[f32; 4] -> [f32; 4]` colour transforms `pipeline::Stage`
/// picks per buffer space (`rec2020_to_xyz_d65`/`xyz_d65_to_rec2020` on the raw
/// path, `srgb_to_xyz_d65`/`xyz_d65_to_srgb` on the non-raw path).
pub type RgbXyzConv = fn([f32; 4]) -> [f32; 4];

/// Apply colorbalancergb to a packed-RGBA scene-linear buffer (`input`/
/// `output` are `n·4` floats) in the pipeline's **raw-path** Rec.2020 working
/// space. Faithful port of the process loop (colorbalancergb.c:662–943), sans
/// the GUI mask-display checkerboard. See [`process_in_space`] for the
/// space-general form.
pub fn process(input: &[f32], output: &mut [f32], d: &CbRgbData) {
    process_in_space(input, output, d, rec2020_to_xyz_d65, xyz_d65_to_rec2020);
}

/// The same grading over an arbitrary working space: pass the matching
/// RGB→XYZ-D65 / XYZ-D65→RGB pair and build `d`'s gamut LUT with that space's
/// [`crate::color`]-matrix twin ([`crate::color::REC2020_TO_XYZ_D65_T4`] /
/// [`crate::color::SRGB_TO_XYZ_D65_T4`]) so saturation gamut-mapping clips at
/// the right primaries — how the C gets its LUT from `work_profile->matrix_in`.
/// Between the two conversions (Yrg chain, masks, offsets) everything depends
/// only on XYZ, so colours grading to results inside *both* spaces' gamuts come
/// out identical; near/over the working-space boundary the per-space LUT
/// correctly clips harder in the smaller space (sRGB ⊂ Rec.2020).
///
/// The C's premultiplied pipeline↔LMS matrices become direct compositions of
/// the working-space conversions. Alpha (lane 3) is preserved from the input
/// (the color chain drops it via the matrices, as in the C, but the pipeline
/// keeps it rather than emitting 0).
pub fn process_in_space(
    input: &[f32],
    output: &mut [f32],
    d: &CbRgbData,
    rgb_to_xyz_d65: RgbXyzConv,
    xyz_d65_to_rgb: RgbXyzConv,
) {
    let (hue_cos, hue_sin) = (d.hue_angle.cos(), d.hue_angle.sin());
    let l_white = y_to_dt_ucs_l_star(d.white_fulcrum);

    for (i_px, o_px) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        // clip pipeline RGB
        let mut rgb = [i_px[0].max(0.0), i_px[1].max(0.0), i_px[2].max(0.0), 0.0];

        // → CIE 2006 LMS D65 → Filmlight Yrg → Ych
        let mut lms = xyz_to_lms_2006(rgb_to_xyz_d65(rgb));
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
        let pix = xyz_d65_to_rgb(xyz_d65);
        o_px[0] = pix[0].max(0.0);
        o_px[1] = pix[1].max(0.0);
        o_px[2] = pix[2].max(0.0);
        o_px[3] = i_px[3];
    }
}

// ── C FFI (m4-137) ────────────────────────────────────────────────────────────
//
// Loop-replacement exports for the C binary, per the m4-86 `colorin` convention:
// the C keeps its orchestration (matrix premultiplication, mask-display gating)
// and the OMP loop bodies move here as serial scalar code.

/// Per-pixel grading over caller-supplied premultiplied matrices — the exact
/// arithmetic of the C process loop (colorbalancergb.c:662–943): one transposed
/// matrix apply in (`RGB → LMS 2006 D65`) and one out (`XYZ D65 → pipeline RGB`),
/// with the GUI mask-display checkerboard branch included. `d` carries the
/// commit_params-derived fields; its `gamut_lut` must be the one built for the
/// same work profile.
fn process_premultiplied(
    input: &[f32],
    output: &mut [f32],
    d: &CbRgbData,
    input_matrix_t: &[[f32; 4]; 4],
    output_matrix_t: &[[f32; 4]; 4],
    out_width: usize,
    mask_display: bool,
    mask_type: usize,
    checker_1: usize,
    checker_color_1: [f32; 4],
    checker_color_2: [f32; 4],
) {
    let npixels = input.len() / 4;
    let l_white = y_to_dt_ucs_l_star(d.white_fulcrum);
    let (hue_cos, hue_sin) = (d.hue_angle.cos(), d.hue_angle.sin());

    for k in 0..npixels {
        let b = k * 4;
        // clip pipeline RGB
        let mut rgb = [
            input[b].max(0.0),
            input[b + 1].max(0.0),
            input[b + 2].max(0.0),
            input[b + 3].max(0.0),
        ];

        // → CIE 2006 LMS D65 (single premultiplied apply) → Filmlight Yrg → Ych
        let mut lms = apply_transposed_color_matrix(&rgb, input_matrix_t);
        let mut yrg = lms_to_yrg(lms);
        let mut ych = yrg_to_ych(yrg);
        ych[0] = ych[0].max(0.0); // sanitise: no negative luminance

        // luma opacity masks
        let (opacities, opacities_comp) = opacity_masks(
            ych[0].powf(GREY_CENTER_EXP),
            d.shadows_weight,
            d.highlights_weight,
            d.midtones_weight,
            d.mask_grey_fulcrum,
        );

        // hue shift — 2×2 rotation of the (cos h, sin h) vector
        let (cos_h, sin_h) = (ych[2], ych[3]);
        ych[2] = hue_cos * cos_h - hue_sin * sin_h;
        ych[3] = hue_sin * cos_h + hue_cos * sin_h;

        // linear chroma boost + vibrance at constant luminance
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
        // shadows/highlights: two masked slopes
        for c in 0..4 {
            rgb[c] *= opacities_comp[2] * (opacities_comp[0] + opacities[0] * d.shadows[c])
                + opacities[2] * d.highlights[c];
        }
        // midtones power (per channel), sign-preserving around the white fulcrum
        for c in 0..4 {
            let sign = if rgb[c] < 0.0 { -1.0 } else { 1.0 };
            let scaled = rgb[c].abs() / d.white_fulcrum;
            rgb[c] = scaled.powf(d.midtones[c]) * sign * d.white_fulcrum;
        }

        // back to Yrg for the non-linear luma ops
        lms = grading_rgb_to_lms(rgb);
        yrg = lms_to_yrg(lms);
        // Y midtones gamma, then fulcrumed contrast
        yrg[0] = (yrg[0] / d.white_fulcrum).max(0.0).powf(d.midtones_y) * d.white_fulcrum;
        yrg[0] = d.grey_fulcrum * (yrg[0] / d.grey_fulcrum).powf(d.contrast);

        // → XYZ D65 → perceptual saturation/brilliance adjustments
        lms = yrg_to_lms(yrg);
        let xyz_d65 = lms_2006_to_xyz(lms);
        let xyz_d65 = match d.saturation_formula {
            SaturationFormula::Jzazbz => saturation_jzazbz(xyz_d65, &opacities, d),
            SaturationFormula::DtUcs => saturation_dtucs(xyz_d65, &opacities, d, l_white),
        };

        // project back to pipeline RGB
        let mut pix_out = apply_transposed_color_matrix(&xyz_d65, output_matrix_t);

        if mask_display {
            // draw the checkerboard behind the masked edit (C: :908–936)
            let i = k / out_width;
            let j = k % out_width;
            let checker_2 = checker_1 * 2;
            // a zero cell size would panic on % 0 where C hits UB/SIGFPE;
            // degrade to a solid first-colour field instead
            let color = if checker_1 == 0 {
                checker_color_1
            } else if i % checker_1 < i % checker_2 {
                if j % checker_1 < j % checker_2 { checker_color_2 } else { checker_color_1 }
            } else if j % checker_1 < j % checker_2 {
                checker_color_1
            } else {
                checker_color_2
            };
            let opacity = opacities[mask_type];
            let opacity_comp = 1.0 - opacity;
            for c in 0..4 {
                let v = pix_out[c].max(0.0); // dt_vector_clipneg before blending
                pix_out[c] = opacity_comp * color[c] + opacity * v;
            }
            pix_out[3] = 1.0; // alpha opaque so the preview shows
        } else {
            for v in pix_out.iter_mut() {
                *v = v.max(0.0); // dt_vector_clipneg
            }
        }

        output[b] = pix_out[0];
        output[b + 1] = pix_out[1];
        output[b + 2] = pix_out[2];
        output[b + 3] = pix_out[3];
    }
}

/// # Safety
/// `in_buf` holds `4·npixels` floats; `out_buf` accepts `4·npixels`; both
/// matrices hold 16 floats (row-major, passed exactly as the C stores its
/// premultiplied `dt_colormatrix_t`); the seven zone vectors hold 4 floats
/// each; `gamut_lut` holds [`LUT_ELEM`] floats; both checker colours hold 4.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_colorbalancergb_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    out_width: usize,
    input_matrix_trans: *const f32,
    output_matrix_trans: *const f32,
    global: *const f32,
    shadows: *const f32,
    highlights: *const f32,
    midtones: *const f32,
    chroma: *const f32,
    saturation_v: *const f32,
    brilliance: *const f32,
    chroma_global: f32,
    vibrance: f32,
    contrast: f32,
    saturation_global: f32,
    brilliance_global: f32,
    midtones_y: f32,
    hue_angle: f32,
    shadows_weight: f32,
    highlights_weight: f32,
    midtones_weight: f32,
    mask_grey_fulcrum: f32,
    white_fulcrum: f32,
    grey_fulcrum: f32,
    saturation_formula: i32,
    gamut_lut: *const f32,
    mask_display: i32,
    mask_type: i32,
    checker_1: usize,
    checker_color_1: *const f32,
    checker_color_2: *const f32,
) {
    if npixels == 0 {
        return;
    }
    let load4 = |p: *const f32| {
        let s = std::slice::from_raw_parts(p, 4);
        [s[0], s[1], s[2], s[3]]
    };
    let load_matrix = |p: *const f32| {
        let s = std::slice::from_raw_parts(p, 16);
        [
            [s[0], s[1], s[2], s[3]],
            [s[4], s[5], s[6], s[7]],
            [s[8], s[9], s[10], s[11]],
            [s[12], s[13], s[14], s[15]],
        ]
    };

    // Checker colours are read only under mask display — the caller may pass
    // NULL otherwise (the C always holds them in `d`, but don't demand it).
    let (checker_color_1_v, checker_color_2_v) = if mask_display != 0 {
        (load4(checker_color_1), load4(checker_color_2))
    } else {
        ([0.0f32; 4], [0.0f32; 4])
    };

    // Assemble the commit_params-derived data so the perceptual branches and
    // helper conversions can be reused verbatim. The gamut LUT is copied once
    // per call (2 KiB) — negligible next to a full-band process pass.
    let d = CbRgbData {
        global: load4(global),
        shadows: load4(shadows),
        highlights: load4(highlights),
        midtones: load4(midtones),
        midtones_y,
        chroma_global,
        chroma: load4(chroma),
        vibrance,
        contrast,
        saturation_global,
        saturation: load4(saturation_v),
        brilliance_global,
        brilliance: load4(brilliance),
        hue_angle,
        shadows_weight,
        highlights_weight,
        midtones_weight,
        mask_grey_fulcrum,
        white_fulcrum,
        grey_fulcrum,
        gamut_lut: std::slice::from_raw_parts(gamut_lut, LUT_ELEM).try_into().unwrap(),
        // C enum: DT_COLORBALANCE_SATURATION_JZAZBZ = 0, DTUCS = 1; any other
        // value takes the UCS branch, as in the C's if/else shape.
        saturation_formula: if saturation_formula == 0 {
            SaturationFormula::Jzazbz
        } else {
            SaturationFormula::DtUcs
        },
    };

    process_premultiplied(
        std::slice::from_raw_parts(in_buf, npixels * 4),
        std::slice::from_raw_parts_mut(out_buf, npixels * 4),
        &d,
        &load_matrix(input_matrix_trans),
        &load_matrix(output_matrix_trans),
        out_width,
        mask_display != 0,
        // valid GUI values are 0..=3 (MASK_SHADOWS/MIDTONES/HIGHLIGHTS/NONE);
        // lane 3 of `opacities` is 0.0, so a clamped garbage index degrades to
        // "checker only" instead of panicking.
        mask_type.clamp(0, 3) as usize,
        checker_1,
        checker_color_1_v,
        checker_color_2_v,
    );
}

/// Build the JzAzBz gamut LUT into `gamut_lut` ([`LUT_ELEM`] floats) by sampling
/// the STEPS³ RGB cube through `input_matrix` — the premultiplied RGB → XYZ D65
/// matrix **exactly as the C stores it** (row-major `dt_colormatrix_t`; our
/// transposed-apply convention reproduces the C's `dot_product` on it). Serial:
/// the C's `reduction(max: sampler[:LUT_ELEM])` is order-independent, so this is
/// bit-identical to any OpenMP thread count. Replaces colorbalancergb.c:1197.
///
/// # Safety
/// `input_matrix` holds 16 floats; `gamut_lut` accepts [`LUT_ELEM`] floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorbalancergb_build_gamut_lut_jzazbz(
    input_matrix: *const f32,
    gamut_lut: *mut f32,
) {
    let s = std::slice::from_raw_parts(input_matrix, 16);
    let mut m = [[0.0f32; 4]; 4];
    for r in 0..4 {
        m[r].copy_from_slice(&s[r * 4..r * 4 + 4]);
    }
    let lut = build_gamut_lut_jzazbz(&m);
    std::slice::from_raw_parts_mut(gamut_lut, LUT_ELEM).copy_from_slice(&lut);
}

/// Fill the GUI mask-preview checkerboard gradient strip (`data`, an ARGB32
/// byte buffer of `graph_height · line_height · 4` bytes laid out packed, as
/// cairo guarantees `stride == width·4` for ARGB32). Replaces the
/// `DT_OMP_FOR(collapse(2))` fill at colorbalancergb.c:1511.
///
/// Returns early when `checker_1` is 0 (the C would divide by zero there; the
/// GUI always passes ≥ DPI-scaled 6).
///
/// # Safety
/// `data` accepts `graph_height · line_height · 4` bytes.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorbalancergb_checkerboard_fill(
    data: *mut u8,
    graph_height: usize,
    line_height: usize,
    checker_1: usize,
) {
    if graph_height == 0 || line_height == 0 || checker_1 == 0 {
        return;
    }
    let buf = std::slice::from_raw_parts_mut(data, graph_height * line_height * 4);
    let checker_2 = checker_1 * 2;
    for i in 0..graph_height {
        for j in 0..line_height {
            let k = (i * line_height + j) * 4;
            let alpha = i as f32 / graph_height as f32;
            let color = if i % checker_1 < i % checker_2 {
                if j % checker_1 < j % checker_2 { 150.0f32 } else { 100.0 }
            } else if j % checker_1 < j % checker_2 {
                100.0
            } else {
                150.0
            };
            let c_byte = (color * alpha) as u8;
            buf[k] = c_byte;
            buf[k + 1] = c_byte;
            buf[k + 2] = c_byte;
            buf[k + 3] = (alpha * 255.0) as u8;
        }
    }
}

/// Fill the three opacity-mask curve LUTs ([`LUT_ELEM`] entries each) shown
/// under the shadows/midtones/highlights sliders. Derives `midtones_weight`
/// and the powered `mask_grey_fulcrum` from the raw slider weights exactly as
/// the C draw callback does (:1548–1550). Replaces the `DT_OMP_FOR` at
/// colorbalancergb.c:1555.
///
/// # Safety
/// Each of `lut_shadows`/`lut_midtones`/`lut_highlights` accepts [`LUT_ELEM`]
/// floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorbalancergb_opacity_luts(
    lut_shadows: *mut f32,
    lut_midtones: *mut f32,
    lut_highlights: *mut f32,
    shadows_weight: f32,
    highlights_weight: f32,
    mask_grey_fulcrum_param: f32,
) {
    let sw2 = shadows_weight * shadows_weight;
    let hw2 = highlights_weight * highlights_weight;
    let midtones_weight = sw2 * hw2 / (sw2 + hw2);
    let mask_grey_fulcrum = mask_grey_fulcrum_param.powf(GREY_CENTER_EXP);

    let luts = [
        std::slice::from_raw_parts_mut(lut_shadows, LUT_ELEM),
        std::slice::from_raw_parts_mut(lut_midtones, LUT_ELEM),
        std::slice::from_raw_parts_mut(lut_highlights, LUT_ELEM),
    ];
    for k in 0..LUT_ELEM {
        let y = k as f32 / (LUT_ELEM - 1) as f32;
        let (out, _) = opacity_masks(y, shadows_weight, highlights_weight, midtones_weight, mask_grey_fulcrum);
        luts[0][k] = out[0];
        luts[1][k] = out[1];
        luts[2][k] = out[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::SRGB_TO_XYZ_D65_T4;

    // (the sRGB → XYZ D65 test matrix moved to `color::SRGB_TO_XYZ_D65_T4` —
    // one source of truth now that the pipeline grades in both spaces)

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
        let lut = build_gamut_lut_jzazbz(&SRGB_TO_XYZ_D65_T4);
        assert!(finite_nonneg(&lut), "jzazbz LUT has non-finite/negative entries");
        assert!(has_variation(&lut), "jzazbz LUT is flat — hue binning likely broken");
    }

    #[test]
    fn ucs_lut_is_finite_nonneg_and_varies() {
        let lut = build_gamut_lut_ucs(&SRGB_TO_XYZ_D65_T4);
        assert!(finite_nonneg(&lut), "ucs LUT has non-finite/negative entries");
        assert!(has_variation(&lut), "ucs LUT is flat — boundary march likely broken");
    }

    #[test]
    fn builders_are_deterministic() {
        // pure functions of the input matrix.
        assert_eq!(
            build_gamut_lut_jzazbz(&SRGB_TO_XYZ_D65_T4),
            build_gamut_lut_jzazbz(&SRGB_TO_XYZ_D65_T4)
        );
        assert_eq!(
            build_gamut_lut_ucs(&SRGB_TO_XYZ_D65_T4),
            build_gamut_lut_ucs(&SRGB_TO_XYZ_D65_T4)
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
    fn params_equality_gate_tracks_the_neutral_edit() {
        // The pipeline's identity gate compares params against `default()` with
        // derived `PartialEq`. Pin that the default IS equal to itself (trivial,
        // but guards an accidental NaN creeping into a default — NaN != NaN
        // would make every enabled instance emit a stage) and that each slider's
        // neutral position is exactly representable, so "moved away and back"
        // re-derives an equal struct.
        assert_eq!(CbRgbParams::default(), CbRgbParams::default());
        for moved in [
            CbRgbParams { shadows_y: 0.1, ..CbRgbParams::default() },
            CbRgbParams { global_c: 0.5, ..CbRgbParams::default() },
            CbRgbParams { white_fulcrum: -2.0, ..CbRgbParams::default() },
            CbRgbParams { contrast: 0.25, ..CbRgbParams::default() },
            CbRgbParams { grey_fulcrum: 0.3, ..CbRgbParams::default() },
            CbRgbParams {
                saturation_formula: SaturationFormula::Jzazbz,
                ..CbRgbParams::default()
            },
        ] {
            assert_ne!(moved, CbRgbParams::default());
        }
    }

    #[test]
    fn gamut_lut_actually_uses_the_input_matrix() {
        // a genuinely different gamut (sRGB primaries) must change the LUT — proves
        // the matrix argument is honoured (guards the transpose-contract wiring).
        // NB: the dt-UCS builder is chromaticity-based, so a *uniform scale* of the
        // matrix would NOT change it (same xyY) — the primaries must actually move.
        let p = CbRgbParams::default();
        let rec = CbRgbData::from_params(&p, &REC2020_TO_XYZ_D65_T4);
        let srgb = CbRgbData::from_params(&p, &SRGB_TO_XYZ_D65_T4);
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
    fn process_in_space_produces_valid_pixels_in_both_working_spaces() {
        // Each pipeline path (raw Rec.2020 / non-raw linear sRGB) must emit
        // finite, non-negative pixels in its OWN space for mild and heavy
        // edits alike. Crossed wiring (converters from one space, gamut LUT
        // from the other) pushes results outside the buffer's hull and shows
        // up here as negatives; legitimate per-space gamut-mapping differences
        // do NOT (each LUT matches its converters).
        let edits = [
            CbRgbParams { saturation_global: 0.2, ..CbRgbParams::default() },
            CbRgbParams {
                saturation_global: 0.9,
                global_y: 0.03,
                global_h: 210.0,
                shadows_h: 30.0,
                contrast: 0.3,
                vibrance: 0.4,
                ..CbRgbParams::default()
            },
        ];
        let colours = [[0.55_f32, 0.25, 0.08], [0.5, 0.5, 0.6], [0.2, 0.45, 0.3], [0.05, 0.09, 0.4]];
        for p in &edits {
            for (matrix_t, to_xyz, from_xyz) in [
                (
                    &REC2020_TO_XYZ_D65_T4,
                    rec2020_to_xyz_d65 as RgbXyzConv,
                    xyz_d65_to_rec2020 as RgbXyzConv,
                ),
                (
                    &SRGB_TO_XYZ_D65_T4,
                    crate::color::srgb_to_xyz_d65 as RgbXyzConv,
                    crate::color::xyz_d65_to_srgb as RgbXyzConv,
                ),
            ] {
                let d = CbRgbData::from_params(p, matrix_t);
                for colour in colours {
                    let inp = [colour[0], colour[1], colour[2], 1.0];
                    let mut out = [0.0f32; 4];
                    process_in_space(&inp, &mut out, &d, to_xyz, from_xyz);
                    assert!(
                        out[..3].iter().all(|v| v.is_finite() && *v >= 0.0),
                        "invalid output {out:?} for {colour:?} with edit sat={}",
                        p.saturation_global
                    );
                }
            }
        }
    }

    #[test]
    fn process_delegates_to_process_in_space_with_the_rec2020_pair() {
        // process() is a thin alias — pin it to the Rec2020 converters so the
        // raw-path entry point can't drift from the general one.
        let p = CbRgbParams {
            saturation_global: 0.6,
            contrast: 0.3,
            ..CbRgbParams::default()
        };
        let d = CbRgbData::from_params(&p, &REC2020_TO_XYZ_D65_T4);
        let inp = [0.42_f32, 0.19, 0.07, 1.0];

        let mut via_alias = [0.0f32; 4];
        process(&inp, &mut via_alias, &d);
        let mut via_general = [0.0f32; 4];
        process_in_space(
            &inp,
            &mut via_general,
            &d,
            rec2020_to_xyz_d65,
            xyz_d65_to_rec2020,
        );
        assert_eq!(via_alias, via_general);
    }

    #[test]
    fn srgb_path_clips_into_its_own_gamut_while_rec2020_keeps_chroma() {
        // A heavy saturation push sends the pixel past the sRGB boundary but
        // not past Rec.2020's (its triangle strictly contains sRGB's): the
        // sRGB path must map chroma back toward its boundary while the Rec.2020
        // path keeps the boost — the per-space behaviour the C gets from
        // building the LUT off the working profile.
        let p = CbRgbParams {
            saturation_global: 0.9,
            ..CbRgbParams::default()
        };
        let d_rec = CbRgbData::from_params(&p, &REC2020_TO_XYZ_D65_T4);
        let d_srgb = CbRgbData::from_params(&p, &SRGB_TO_XYZ_D65_T4);

        let srgb_in = [0.55_f32, 0.25, 0.08, 1.0]; // saturated orange
        let rec_in = xyz_d65_to_rec2020(crate::color::srgb_to_xyz_d65(srgb_in));

        let mut out_srgb = [0.0f32; 4];
        process_in_space(
            &srgb_in,
            &mut out_srgb,
            &d_srgb,
            crate::color::srgb_to_xyz_d65,
            crate::color::xyz_d65_to_srgb,
        );
        let out_rec = run_one(rec_in[..3].try_into().unwrap(), &d_rec);

        // Chroma proxy in XYZ: distance from the equal-chroma axis, scaled by
        // luminance — (X+Z)/(Y) style spread. Simpler and robust here: the
        // sRGB result converted back into Rec.2020 must stay non-negative
        // (inside the Rec.2020 unit cube ⇒ representable), while BOTH results
        // are finite/non-negative in their own spaces.
        let back = xyz_d65_to_rec2020(crate::color::srgb_to_xyz_d65(out_srgb));
        for c in 0..3 {
            assert!(out_srgb[c].is_finite() && out_srgb[c] >= 0.0);
            assert!(out_rec[c].is_finite() && out_rec[c] >= 0.0);
            assert!(
                back[c] >= -1e-4,
                "sRGB-path result left the Rec.2020 hull on channel {c}: {back:?}"
            );
        }
        // And the sRGB path must end up LESS chromatic than the unclipped
        // Rec.2020 path at the same hue: compare UCS-style x-spread from D65.
        let xyy_r = crate::color::d65_xyz_to_xyy(&rec2020_to_xyz_d65(out_rec));
        let xyy_s = crate::color::d65_xyz_to_xyy(&crate::color::srgb_to_xyz_d65(out_srgb));
        let d65 = [crate::color::D65_X, crate::color::D65_Y];
        let dr = ((xyy_r[0] - d65[0]).powi(2) + (xyy_r[1] - d65[1]).powi(2)).sqrt();
        let ds = ((xyy_s[0] - d65[0]).powi(2) + (xyy_s[1] - d65[1]).powi(2)).sqrt();
        assert!(
            ds < dr,
            "expected the sRGB path to clip harder than Rec.2020: ds={ds} dr={dr}"
        );
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

    // ── C FFI (m4-137) ──

    use crate::color::{XYZ_D65_TO_LMS_2006_T, XYZ_D65_TO_REC2020_T4};

    /// Row-wise product `A·B` — composing two transposed-apply conversions into
    /// the single premultiplied matrix the C passes over FFI.
    fn compose(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut m = [[0.0f32; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                m[r][c] = (0..4).map(|k| a[r][k] * b[k][c]).sum();
            }
        }
        m
    }

    #[test]
    fn ffi_process_matches_the_pure_pipeline_path() {
        // The premultiplied-matrix FFI form and the pipeline's two-step fn-pointer
        // form differ only in FP associativity (one apply vs RGB→XYZ→LMS); with
        // the same Rec.2020 working space and the same derived data they must
        // agree to well under any visible epsilon.
        let p = CbRgbParams {
            saturation_global: 0.35,
            contrast: 0.2,
            global_h: 30.0,
            ..CbRgbParams::default()
        };
        let d = CbRgbData::from_params(&p, &REC2020_TO_XYZ_D65_T4);
        // compose(A, B) applies A then B: Rec2020→XYZ D65 first, then XYZ D65→LMS.
        let input_m = compose(&REC2020_TO_XYZ_D65_T4, &XYZ_D65_TO_LMS_2006_T);
        let output_m = XYZ_D65_TO_REC2020_T4;

        let inp: Vec<f32> = (0..24)
            .flat_map(|i| {
                let v = i as f32 / 24.0;
                [v, 1.0 - v, v * 0.5, 1.0]
            })
            .collect();
        let mut out_ffi = vec![0.0f32; inp.len()];
        unsafe {
            darkroom_colorbalancergb_process(
                inp.as_ptr(), out_ffi.as_mut_ptr(), 24, 24,
                input_m.as_ptr() as *const f32, output_m.as_ptr() as *const f32,
                d.global.as_ptr(), d.shadows.as_ptr(), d.highlights.as_ptr(),
                d.midtones.as_ptr(), d.chroma.as_ptr(), d.saturation.as_ptr(),
                d.brilliance.as_ptr(),
                d.chroma_global, d.vibrance, d.contrast,
                d.saturation_global, d.brilliance_global, d.midtones_y, d.hue_angle,
                d.shadows_weight, d.highlights_weight, d.midtones_weight,
                d.mask_grey_fulcrum, d.white_fulcrum, d.grey_fulcrum,
                1, // DT_COLORBALANCE_SATURATION_DTUCS
                d.gamut_lut.as_ptr(),
                0, 0, 0, std::ptr::null(), std::ptr::null(),
            );
        }

        let mut out_pure = vec![0.0f32; inp.len()];
        process_in_space(&inp, &mut out_pure, &d, rec2020_to_xyz_d65, xyz_d65_to_rec2020);

        for (a, b) in out_ffi.chunks_exact(4).zip(out_pure.chunks_exact(4)) {
            for c in 0..3 {
                assert!(
                    (a[c] - b[c]).abs() < 1e-4,
                    "FFI {a:?} vs pure {b:?} diverged on channel {c}"
                );
            }
            // C-parity detail: the FFI path writes the computed alpha lane (the
            // colour chain zeroes it), unlike process_in_space's preserved alpha.
            assert_eq!(a[3], 0.0, "FFI must emit alpha 0 like the C");
        }
    }

    #[test]
    fn ffi_process_mask_none_blends_to_pure_checkerboard() {
        // With mask_type = MASK_NONE (3) the opacity slot reads 0, so every pixel
        // collapses to its checker colour with opaque alpha — pins the blend
        // arithmetic, the i/j cell selection and the clip-before-blend order.
        let d = neutral();
        let c1 = [0.25f32, 0.5, 0.75, 1.0];
        let c2 = [0.9f32, 0.1, 0.2, 1.0];
        let width = 4usize;
        let inp: Vec<f32> = vec![0.42; width * 4 * 2]; // 2 rows × 4 cols, alpha 1
        let mut out = vec![0.0f32; inp.len()];
        // matrices are irrelevant here (opacity 0 ⇒ output is pure checker
        // colour); pass a valid one for the entry apply.
        unsafe {
            darkroom_colorbalancergb_process(
                inp.as_ptr(), out.as_mut_ptr(), 8, width,
                REC2020_TO_XYZ_D65_T4.as_ptr() as *const f32,
                XYZ_D65_TO_REC2020_T4.as_ptr() as *const f32,
                d.global.as_ptr(), d.shadows.as_ptr(), d.highlights.as_ptr(),
                d.midtones.as_ptr(), d.chroma.as_ptr(), d.saturation.as_ptr(),
                d.brilliance.as_ptr(),
                d.chroma_global, d.vibrance, d.contrast,
                d.saturation_global, d.brilliance_global, d.midtones_y, d.hue_angle,
                d.shadows_weight, d.highlights_weight, d.midtones_weight,
                d.mask_grey_fulcrum, d.white_fulcrum, d.grey_fulcrum,
                1, d.gamut_lut.as_ptr(),
                1, 3 /* MASK_NONE */, 1 /* checker_1 */,
                c1.as_ptr(), c2.as_ptr(),
            );
        }
        // checker_1 = 1 ⇒ checker_2 = 2 ⇒ i%1<i%2 ⇔ i odd; same for j. Cell
        // colour: (odd i,odd j)→c1·pattern… evaluated straight from the C text:
        // i odd → outer true: j odd → c2, j even → c1; i even → outer false:
        // j odd → c1, j even → c2.
        for k in 0..8 {
            let (i, j) = (k / width, k % width);
            let want = match (i % 2 == 1, j % 2 == 1) {
                (true, true) | (false, false) => c2,
                _ => c1,
            };
            let o = &out[k * 4..k * 4 + 4];
            for c in 0..4 {
                assert!((o[c] - want[c]).abs() < 1e-6, "cell ({i},{j}) ch{c}: {} != {}", o[c], want[c]);
            }
            assert_eq!(o[3], 1.0, "mask preview must be opaque");
        }
    }

    #[test]
    fn ffi_process_mask_blend_recovers_a_single_intermediate_opacity() {
        // Review gap: pin the masked blend at an opacity strictly between 0
        // and 1. The plain run's clipped output IS the v blended in the masked
        // run (the branch only differs after pix_out), so the shared opacity
        // can be recovered from the data, o = (m−colour)/(p−colour), and is
        // then required to agree across every lane and cell — a per-lane or
        // pre-clip blend cannot satisfy that.
        let d = neutral();
        let c1 = [0.15f32, 0.65, 0.35, 1.0];
        let c2 = [0.85f32, 0.25, 0.55, 1.0];
        let width = 4usize;
        // dark neutral frame + default shadows weight ⇒ shadows opacity
        // ≈ sigmoid(−0.6·4) ≈ 0.08 — strictly intermediate.
        let inp: Vec<f32> = [0.02f32, 0.02, 0.02, 1.0].repeat(width * 2);
        let mut plain = vec![0.0f32; inp.len()];
        let mut masked = vec![0.0f32; inp.len()];
        let run = |out: &mut Vec<f32>, mask_display: i32| unsafe {
            darkroom_colorbalancergb_process(
                inp.as_ptr(), out.as_mut_ptr(), 8, width,
                REC2020_TO_XYZ_D65_T4.as_ptr() as *const f32,
                XYZ_D65_TO_REC2020_T4.as_ptr() as *const f32,
                d.global.as_ptr(), d.shadows.as_ptr(), d.highlights.as_ptr(),
                d.midtones.as_ptr(), d.chroma.as_ptr(), d.saturation.as_ptr(),
                d.brilliance.as_ptr(),
                d.chroma_global, d.vibrance, d.contrast,
                d.saturation_global, d.brilliance_global, d.midtones_y, d.hue_angle,
                d.shadows_weight, d.highlights_weight, d.midtones_weight,
                d.mask_grey_fulcrum, d.white_fulcrum, d.grey_fulcrum,
                1, d.gamut_lut.as_ptr(),
                mask_display, 0 /* MASK_SHADOWS */, 1 /* checker_1 */,
                c1.as_ptr(), c2.as_ptr(),
            );
        };
        run(&mut plain, 0);
        run(&mut masked, 1);

        let mut opacity: Option<f32> = None;
        for k in 0..8 {
            let (i, j) = (k / width, k % width);
            let want = match (i % 2 == 1, j % 2 == 1) {
                (true, true) | (false, false) => c2,
                _ => c1,
            };
            let m = &masked[k * 4..k * 4 + 4];
            let p = &plain[k * 4..k * 4 + 4];
            assert_eq!(m[3], 1.0, "mask preview must be opaque");
            let o = match opacity {
                Some(o) => o,
                None => {
                    assert!((p[0] - want[0]).abs() > 1e-3, "lane 0 not identifiable");
                    let o = (m[0] - want[0]) / (p[0] - want[0]);
                    assert!(o > 1e-3 && o < 1.0 - 1e-3, "opacity {o} not intermediate");
                    opacity = Some(o);
                    o
                }
            };
            for c in 0..3 {
                let expect = (1.0 - o) * want[c] + o * p[c];
                assert!(
                    (m[c] - expect).abs() < 1e-5,
                    "cell ({i},{j}) ch{c}: {} != blend({o}, colour, {})",
                    m[c],
                    expect
                );
            }
        }
    }

    #[test]
    fn ffi_process_mask_display_zero_cell_size_degrades_not_panics() {
        // pins the % 0 guard: the C would SIGFPE on i % checker_1 with a zero
        // cell; the export must instead render a solid first-colour field.
        let d = neutral();
        let c1 = [0.15f32, 0.65, 0.35, 1.0];
        let c2 = [0.85f32, 0.25, 0.55, 1.0];
        let width = 4usize;
        let inp: Vec<f32> = [0.02f32, 0.02, 0.02, 1.0].repeat(width * 2);
        let mut plain = vec![0.0f32; inp.len()];
        let mut masked = vec![0.0f32; inp.len()];
        let run = |out: &mut Vec<f32>, md: i32, checker: usize| unsafe {
            darkroom_colorbalancergb_process(
                inp.as_ptr(), out.as_mut_ptr(), 8, width,
                REC2020_TO_XYZ_D65_T4.as_ptr() as *const f32,
                XYZ_D65_TO_REC2020_T4.as_ptr() as *const f32,
                d.global.as_ptr(), d.shadows.as_ptr(), d.highlights.as_ptr(),
                d.midtones.as_ptr(), d.chroma.as_ptr(), d.saturation.as_ptr(),
                d.brilliance.as_ptr(),
                d.chroma_global, d.vibrance, d.contrast,
                d.saturation_global, d.brilliance_global, d.midtones_y, d.hue_angle,
                d.shadows_weight, d.highlights_weight, d.midtones_weight,
                d.mask_grey_fulcrum, d.white_fulcrum, d.grey_fulcrum,
                1, d.gamut_lut.as_ptr(),
                md, 0 /* MASK_SHADOWS */, checker,
                c1.as_ptr(), c2.as_ptr(),
            );
        };
        run(&mut plain, 0, 1);
        run(&mut masked, 1, 0);
        let o = (masked[0] - c1[0]) / (plain[0] - c1[0]);
        for k in 0..8 {
            let m = &masked[k * 4..k * 4 + 4];
            assert_eq!(m[3], 1.0);
            for c in 0..3 {
                let expect = (1.0 - o) * c1[c] + o * plain[k * 4 + c]; // solid c1 field
                assert!((m[c] - expect).abs() < 1e-5, "cell {k} ch{c} diverged");
            }
        }
    }

    #[test]
    fn ffi_gamut_lut_builder_writes_the_same_values_as_the_rust_entry() {
        let m = compose(&XYZ_D65_TO_LMS_2006_T, &SRGB_TO_XYZ_D65_T4);
        let mut lut = [0.0f32; LUT_ELEM];
        unsafe {
            darkroom_colorbalancergb_build_gamut_lut_jzazbz(m.as_ptr() as *const f32, lut.as_mut_ptr());
        }
        assert_eq!(lut, build_gamut_lut_jzazbz(&m));
    }

    #[test]
    fn ffi_checkerboard_fill_matches_hand_computed_cells() {
        // 5 rows × 5 cols, checker_1 = 2 (checker_2 = 4). Hand-derived from the
        // C's nested condition at :1518–1527.
        let mut buf = [0u8; 5 * 5 * 4];
        unsafe {
            darkroom_colorbalancergb_checkerboard_fill(buf.as_mut_ptr(), 5, 5, 2);
        }
        let px = |i: usize, j: usize| -> [u8; 4] {
            let k = (i * 5 + j) * 4;
            [buf[k], buf[k + 1], buf[k + 2], buf[k + 3]]
        };
        // (0,0): 0%2<0%4 F, 0%2<0%4 F → 150 · alpha 0
        assert_eq!(px(0, 0), [0, 0, 0, 0]);
        // (0,3): outer F, 3%2=1<3%4=3 T → 100 · alpha 0
        assert_eq!(px(0, 3), [0, 0, 0, 0]);
        // (3,0): 3%2=1<3%4=3 T, inner j F → 100 · alpha 3/5 → 153.0→153, 60.0→60
        assert_eq!(px(3, 0), [60, 60, 60, 153]);
        // (3,3): outer T, inner T → 150 · alpha 3/5 → 90
        assert_eq!(px(3, 3), [90, 90, 90, 153]);
        // (4,4): 4%2=0<4%4=0 F, 4%2=0<4%4=0 F → 150 · alpha 4/5 → 120, 204
        assert_eq!(px(4, 4), [120, 120, 120, 204]);
    }

    #[test]
    fn ffi_checkerboard_fill_rejects_degenerate_dims() {
        // modulo-by-zero would panic where the C would be UB; the export must
        // refuse instead.
        let mut buf = [0u8; 16];
        unsafe {
            darkroom_colorbalancergb_checkerboard_fill(buf.as_mut_ptr(), 1, 4, 0);
            darkroom_colorbalancergb_checkerboard_fill(buf.as_mut_ptr(), 0, 4, 2);
            darkroom_colorbalancergb_checkerboard_fill(buf.as_mut_ptr(), 1, 0, 2);
        }
        assert!(buf.iter().all(|b| *b == 0));
    }

    #[test]
    fn ffi_opacity_luts_match_direct_mask_evaluation() {
        let (sw, hw, mgf_param) = (4.0f32, 6.0f32, 0.18f32);
        let mut l0 = [0.0f32; LUT_ELEM];
        let mut l1 = [0.0f32; LUT_ELEM];
        let mut l2 = [0.0f32; LUT_ELEM];
        unsafe {
            darkroom_colorbalancergb_opacity_luts(
                l0.as_mut_ptr(), l1.as_mut_ptr(), l2.as_mut_ptr(), sw, hw, mgf_param,
            );
        }
        let mw = {
            let (a, b) = (sw * sw, hw * hw);
            a * b / (a + b)
        };
        let mgf = mgf_param.powf(GREY_CENTER_EXP);
        for &k in &[0usize, 1, LUT_ELEM / 3, LUT_ELEM / 2, LUT_ELEM - 2, LUT_ELEM - 1] {
            let y = k as f32 / (LUT_ELEM - 1) as f32;
            let (out, _) = opacity_masks(y, sw, hw, mw, mgf);
            assert_eq!(l0[k], out[0]);
            assert_eq!(l1[k], out[1]);
            assert_eq!(l2[k], out[2]);
        }
    }
}
