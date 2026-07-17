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
    apply_transposed_color_matrix, d65_xyz_to_xyy, xyy_to_dt_ucs_uv, xyz_to_jzazbz, LUT_ELEM,
};

/// RGB-cube sampling resolution for the JzAzBz gamut LUT (`#define STEPS 92`).
const STEPS: usize = 92;

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
}
