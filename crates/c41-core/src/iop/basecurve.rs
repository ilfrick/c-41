use crate::{params::IopParams, roi::RoiIn, Result};
use crate::color::eval_exp; // shared unbounded LUT extrapolation (dt_iop_eval_exp)
use super::{ClBuffer, IopProcess};

pub struct Basecurve;

impl IopProcess for Basecurve {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "basecurve" }
}

const LUT_SIZE: usize = 0x10000; // 65536

/// Integer-truncation LUT lookup matching table[CLAMP((int)(f*0x10000), 0, 0xffff)].
/// Output is floored at 0.
#[inline(always)]
fn lut_lookup(table: &[f32], f: f32) -> f32 {
    let idx = ((f * LUT_SIZE as f32) as i32).clamp(0, (LUT_SIZE - 1) as i32) as usize;
    table[idx].max(0.0)
}


/// Fast exp approximation matching dt_fast_expf() in common/math.h.
/// Valid for x in [-100, 0]; behaviour outside that range is intentionally imprecise.
#[inline(always)]
fn fast_expf(x: f32) -> f32 {
    const I1: f32 = 0x3f800000_u32 as f32;
    const SCALE: f32 = (0x402DF854_u32 - 0x3f800000_u32) as f32;
    let k0 = (I1 + x * SCALE) as i32;
    f32::from_bits(if k0 > 0 { k0 as u32 } else { 0 })
}

/// Per-channel tone curve (integer-truncation LUT) for the legacy no-preserve-colors path.
///
/// Matches apply_legacy_curve() in src/iop/basecurve.c.
/// table:            65536 floats — single shared LUT for all RGB channels.
/// unbounded_coeffs: 3 floats — [coeff0, coeff1, coeff2] for eval_exp extrapolation.
/// mul:              pre-scalar applied to every channel value before the LUT lookup.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_apply_legacy_curve(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mul: f32,
    table: *const f32,
    unbounded_coeffs: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lut = std::slice::from_raw_parts(table, LUT_SIZE);
    let coeffs = std::slice::from_raw_parts(unbounded_coeffs, 3);

    for k in 0..npixels {
        let base = k * 4;
        for i in 0..3 {
            let f = input[base + i] * mul;
            output[base + i] = if f < 1.0 {
                lut_lookup(lut, f)
            } else {
                eval_exp(coeffs, f).max(0.0)
            };
        }
        output[base + 3] = input[base + 3];
    }
}

/// Tone curve with colour preservation (the `preserve_colors != NONE` path).
///
/// Matches apply_curve() in src/iop/basecurve.c. Per pixel it computes
/// `lum = mul * dt_rgb_norm(rgb, preserve_colors, work_profile)`, maps it through
/// the shared curve (`table` for `lum < 1`, else `eval_exp(unbounded_coeffs, lum)`),
/// and scales all three channels by `mul * curve_lum / lum`. Alpha passes through.
///
/// Only the LUMINANCE norm (mode 1) consults the working ICC profile; the other
/// norms are profile-independent. Work-profile fields are passed flat and may be
/// NULL when `has_work_profile == 0` (LUMINANCE then falls back to the camera
/// primaries, matching dt_camera_rgb_luminance):
///   * `matrix_in`     — 16 floats (`[4][4]`; only the Y row is read)
///   * `lut0/lut1/lut2` — `lutsize` floats each (only read when `nonlinearlut != 0`)
///   * `unbounded_in`  — 9 floats (`[3][3]`; per-channel TRC extrapolation)
///
/// # Safety
/// Pointers must be valid for the documented lengths. The work-profile pointers
/// may be NULL only when `has_work_profile == 0` (and the luts only when
/// `nonlinearlut == 0`).
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_apply_curve(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mul: f32,
    preserve_colors: i32,
    table: *const f32,
    unbounded_coeffs: *const f32,
    has_work_profile: i32,
    matrix_in: *const f32,
    lut0: *const f32,
    lut1: *const f32,
    lut2: *const f32,
    unbounded_in: *const f32,
    lutsize: i32,
    nonlinearlut: i32,
) {
    use crate::color;
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tbl = std::slice::from_raw_parts(table, LUT_SIZE);
    let curve_ub = std::slice::from_raw_parts(unbounded_coeffs, 3);

    let has_wp = has_work_profile != 0;
    let nonlinear = nonlinearlut != 0;
    let lutsize = lutsize as usize;

    // Hoist work-profile views — constant across all pixels.
    let matrix = if has_wp {
        Some(&*(matrix_in as *const [[f32; 4]; 4]))
    } else {
        None
    };
    let trc = if has_wp && nonlinear {
        let l0 = std::slice::from_raw_parts(lut0, lutsize);
        let l1 = std::slice::from_raw_parts(lut1, lutsize);
        let l2 = std::slice::from_raw_parts(lut2, lutsize);
        let ub = std::slice::from_raw_parts(unbounded_in, 9);
        Some(([l0, l1, l2], [&ub[0..3], &ub[3..6], &ub[6..9]]))
    } else {
        None
    };

    for k in 0..npixels {
        let base = k * 4;
        let r = input[base];
        let g = input[base + 1];
        let b = input[base + 2];

        // dt_rgb_norm: only LUMINANCE (mode 1) depends on the work profile.
        let norm = if preserve_colors == 1 {
            match matrix {
                Some(m) => match trc {
                    Some((luts, ubc)) => {
                        color::get_rgb_matrix_luminance([r, g, b, 0.0], m, luts, ubc, lutsize, true)
                    }
                    // linear profile: Y-row dot product, no TRC.
                    None => m[1][0] * r + m[1][1] * g + m[1][2] * b,
                },
                // no work profile -> camera primaries (dt_camera_rgb_luminance).
                None => r * 0.2225045 + g * 0.7168786 + b * 0.0606169,
            }
        } else {
            color::rgb_norm(r, g, b, preserve_colors)
        };

        let lum = mul * norm;
        let mut ratio = 1.0f32;
        if lum > 0.0 {
            let curve_lum = if lum < 1.0 {
                // table[CLAMP((int)(lum * 0x10000), 0, 0xffff)]
                let idx = ((lum * LUT_SIZE as f32) as i32).clamp(0, (LUT_SIZE - 1) as i32) as usize;
                tbl[idx]
            } else {
                eval_exp(curve_ub, lum)
            };
            ratio = mul * curve_lum / lum;
        }
        output[base] = (ratio * r).max(0.0);
        output[base + 1] = (ratio * g).max(0.0);
        output[base + 2] = (ratio * b).max(0.0);
        output[base + 3] = input[base + 3];
    }
}

/// Compute per-pixel exposure-fusion features into the alpha channel in-place.
///
/// Matches compute_features() in src/iop/basecurve.c.
/// Writes sat * well_exposedness into buf[k*4+3] for every pixel k.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_compute_features(
    buf: *mut f32,
    npixels: usize,
) {
    let buf = std::slice::from_raw_parts_mut(buf, npixels * 4);

    for k in 0..npixels {
        let x = k * 4;
        let r = buf[x];
        let g = buf[x + 1];
        let b = buf[x + 2];

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let sat = 0.1_f32 + 0.1_f32 * (max - min) / max.max(1e-4_f32);

        const C: f32 = 0.54;
        let v = (r - C).abs().max((g - C).abs()).max((b - C).abs());
        const VAR_SQ: f32 = 0.5 * 0.5; // var = 0.5
        let exp_val = 0.2_f32 + fast_expf(-v * v / VAR_SQ);

        buf[x + 3] = sat * exp_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_lookup_identity_lut() {
        let lut: Vec<f32> = (0..LUT_SIZE).map(|k| k as f32 / LUT_SIZE as f32).collect();
        // f=0.5 → index = int(0.5 * 65536) = 32768 → lut[32768] = 32768/65536 = 0.5
        let out = lut_lookup(&lut, 0.5);
        assert!((out - 0.5).abs() < 1e-4, "out={out}");
    }

    #[test]
    fn lut_lookup_negative_clips_to_zero() {
        let lut = vec![0.42_f32; LUT_SIZE];
        assert_eq!(lut_lookup(&lut, -1.0), 0.42);
    }

    #[test]
    fn lut_lookup_floors_negative_lut_values() {
        let mut lut = vec![0.0_f32; LUT_SIZE];
        lut[0] = -0.5;
        assert_eq!(lut_lookup(&lut, 0.0), 0.0);
    }

    #[test]
    fn eval_exp_matches_formula() {
        let coeff = [2.0_f32, 3.0, 0.5];
        let v = 0.25_f32;
        let expected = 3.0 * (0.25 * 2.0_f32).powf(0.5);
        assert!((eval_exp(&coeff, v) - expected).abs() < 1e-5);
    }

    #[test]
    fn apply_legacy_curve_alpha_passthrough() {
        let lut: Vec<f32> = (0..LUT_SIZE).map(|k| k as f32 / LUT_SIZE as f32).collect();
        let coeffs = [1.0_f32, 1.0, 1.0];
        let input = vec![0.25_f32, 0.5, 0.75, 0.9999];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_basecurve_apply_legacy_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                lut.as_ptr(), coeffs.as_ptr(),
            );
        }
        assert!((out[0] - 0.25).abs() < 1e-3, "R={}", out[0]);
        assert!((out[1] - 0.5 ).abs() < 1e-3, "G={}", out[1]);
        assert!((out[2] - 0.75).abs() < 1e-3, "B={}", out[2]);
        assert_eq!(out[3], 0.9999); // alpha unchanged
    }

    #[test]
    fn apply_legacy_curve_unbounded_path() {
        // f >= 1.0 triggers eval_exp, result floored at 0
        let lut = vec![0.0_f32; LUT_SIZE];
        // coeff: coeff[1]*pow(v*coeff[0], coeff[2]) = 1*pow(2*1, 1) = 2
        let coeffs = [1.0_f32, 1.0, 1.0];
        let input = vec![2.0_f32, 2.0, 2.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_basecurve_apply_legacy_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                lut.as_ptr(), coeffs.as_ptr(),
            );
        }
        assert!((out[0] - 2.0).abs() < 1e-5, "R={}", out[0]);
    }

    #[test]
    fn compute_features_grey_pixel() {
        // For a grey pixel r=g=b=0.54: max==min → sat=0.1, v=0 → exp_val=0.2+fast_expf(0)≈1.2
        let mut buf = vec![0.54_f32, 0.54, 0.54, 0.0];
        unsafe { darkroom_basecurve_compute_features(buf.as_mut_ptr(), 1); }
        // sat = 0.1, exp_val ≈ 0.2 + 1.0 = 1.2 (fast_expf(0) ≈ 1.0)
        let alpha = buf[3];
        assert!(alpha > 0.0 && alpha < 1.0, "alpha={alpha}");
    }

    #[test]
    fn compute_features_does_not_touch_rgb() {
        let mut buf = vec![0.3_f32, 0.5, 0.7, 0.0];
        let orig = [buf[0], buf[1], buf[2]];
        unsafe { darkroom_basecurve_compute_features(buf.as_mut_ptr(), 1); }
        assert_eq!(buf[0], orig[0]);
        assert_eq!(buf[1], orig[1]);
        assert_eq!(buf[2], orig[2]);
    }

    // Identity curve table: table[i] = i / 65536, so curve_lum ≈ lum for lum < 1.
    fn identity_table() -> Vec<f32> {
        (0..LUT_SIZE).map(|i| i as f32 / LUT_SIZE as f32).collect()
    }

    #[test]
    fn apply_curve_identity_max_norm_is_passthrough() {
        let table = identity_table();
        let ub = [1.0f32, 1.0, 1.0];
        let input = [0.5f32, 0.25, 0.1, 0.9];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_basecurve_apply_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                2, // DT_RGB_NORM_MAX -> norm = 0.5; identity curve -> ratio 1
                table.as_ptr(), ub.as_ptr(),
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), 0, 0,
            );
        }
        assert!((out[0] - 0.5).abs() < 1e-3, "{out:?}");
        assert!((out[1] - 0.25).abs() < 1e-3);
        assert!((out[2] - 0.1).abs() < 1e-3);
        assert_eq!(out[3], 0.9);
    }

    #[test]
    fn apply_curve_luminance_no_profile_uses_camera_primaries() {
        let table = identity_table();
        let ub = [1.0f32, 1.0, 1.0];
        // grey 0.5: camera luminance = 0.5*(0.2225045+0.7168786+0.0606169) = 0.5
        let input = [0.5f32, 0.5, 0.5, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_basecurve_apply_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                1, // DT_RGB_NORM_LUMINANCE, no work profile -> camera primaries
                table.as_ptr(), ub.as_ptr(),
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), 0, 0,
            );
        }
        for c in 0..3 {
            assert!((out[c] - 0.5).abs() < 1e-3, "c={c} {out:?}");
        }
    }

    #[test]
    fn apply_curve_luminance_linear_profile_uses_matrix_y_row() {
        let table = identity_table();
        let ub = [1.0f32, 1.0, 1.0];
        // 4x4 matrix; Y row (row 1) = [0.2, 0.7, 0.1]. nonlinearlut = 0 (linear).
        let matrix: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.2, 0.7, 0.1, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let input = [0.5f32, 0.5, 0.5, 1.0]; // lum = 0.5*(0.2+0.7+0.1) = 0.5
        let mut out = [0f32; 4];
        unsafe {
            darkroom_basecurve_apply_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                1, table.as_ptr(), ub.as_ptr(),
                1, matrix.as_ptr(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), 0, 0,
            );
        }
        for c in 0..3 {
            assert!((out[c] - 0.5).abs() < 1e-3, "c={c} {out:?}");
        }
    }

    // ── m4-124: commit_params port + orchestration ──────────────────────────

    #[test]
    fn build_table_identity_is_ramp_with_linear_tail() {
        let t = build_table(&[(0.0, 0.0), (1.0, 1.0)], crate::curve_tools::MONOTONE_HERMITE);
        assert!(
            t.table
                .iter()
                .enumerate()
                .all(|(k, &v)| (v - k as f32 / 0xffff as f32).abs() < 1e-4),
            "identity anchors must sample to a ramp"
        );
        // A straight segment recovers y = x (coeffs ≈ [1, 1, ·]).
        assert!((t.unbounded_coeffs[0] - 1.0).abs() < 1e-3);
        assert!((t.unbounded_coeffs[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn build_table_matches_rgbcurve_lut_for_same_nodes() {
        // Both commit_params ports sample the same V1 sampler the same way;
        // pin that they agree for a non-trivial curve.
        let nodes = vec![(0.0f32, 0.0f32), (0.5, 0.35), (1.0, 1.0)];
        let ty = crate::curve_tools::CUBIC_SPLINE;
        let bc = build_table(&nodes, ty);
        let rc = super::super::rgbcurve::build_luts(
            &nodes, ty,
            &[(0.0, 0.0), (1.0, 1.0)], crate::curve_tools::MONOTONE_HERMITE,
            &[(0.0, 0.0), (1.0, 1.0)], crate::curve_tools::MONOTONE_HERMITE,
        );
        assert_eq!(&bc.table[..], &rc.table_r[..], "R tables must be byte-equal");
        assert_eq!(bc.unbounded_coeffs, rc.unbounded_coeffs[0]);
    }

    #[test]
    fn exposure_increment_matches_c_formula() {
        // offset = stops·fusion·(bias−1)/2; inc = 2^(stops·e + offset)
        // bias=1 → offset 0: exposures at 2^0, 2^1.
        assert!((exposure_increment(1.0, 0, 1, 1.0) - 1.0).abs() < 1e-6);
        assert!((exposure_increment(1.0, 1, 1, 1.0) - 2.0).abs() < 1e-6);
        // bias=0, fusion=2, stops=1 → offset −1: e=0 sits at half exposure.
        assert!((exposure_increment(1.0, 0, 2, 0.0) - 0.5).abs() < 1e-6);
        assert!((exposure_increment(1.0, 2, 2, 0.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn apply_curve_pixels_dispatches_both_branches() {
        // preserve=0 (legacy): same table per channel — a coloured pixel keeps
        // its ratios only because the table is shared; preserve=1 scales all
        // channels by one ratio. With a midtone-down curve both must darken,
        // but the legacy result is NOT chroma-preserving on an off-grey input.
        let nodes = vec![(0.0f32, 0.0f32), (0.5, 0.4), (1.0, 1.0)];
        let t = build_table(&nodes, crate::curve_tools::CATMULL_ROM);
        let inp = [0.6f32, 0.3, 0.15, 1.0];
        let mut legacy = [0f32; 4];
        apply_curve_pixels(&inp, &mut legacy, 1.0, &t.table[..], &t.unbounded_coeffs, 0, None);
        let mut preserve = [0f32; 4];
        apply_curve_pixels(&inp, &mut preserve, 1.0, &t.table[..], &t.unbounded_coeffs, 1, Some(crate::color::SRGB_TO_XYZ_D65_Y_ROW));
        // Legacy: each channel looked up independently → different ratios.
        let r_legacy = legacy[0] / inp[0];
        assert!(
            (r_legacy - legacy[1] / inp[1]).abs() > 1e-4 || (r_legacy - legacy[2] / inp[2]).abs() > 1e-4,
            "legacy path must not be chroma-preserving here"
        );
        // Preserve: single ratio across channels.
        let r_pres = preserve[0] / inp[0];
        assert!((r_pres - preserve[1] / inp[1]).abs() < 1e-4);
        assert!((r_pres - preserve[2] / inp[2]).abs() < 1e-4);
    }

    #[test]
    fn apply_curve_pixels_y_row_drives_luminance_norm() {
        // With LUMINANCE preservation and a supplied Y row, the norm must be
        // exactly dot(rgb, y_row): pick a Y row making lum computable by hand
        // and verify the ratio equals curve(lum)/lum.
        let nodes = vec![(0.0f32, 0.0f32), (0.5, 0.4), (1.0, 1.0)];
        let t = build_table(&nodes, crate::curve_tools::CATMULL_ROM);
        let y = [0.25f32, 0.5, 0.25];
        let inp = [0.8f32, 0.4, 0.4, 1.0]; // lum = 0.5
        let mut out = [0f32; 4];
        apply_curve_pixels(&inp, &mut out, 1.0, &t.table[..], &t.unbounded_coeffs, 1, Some(y));
        // curve at 0.5 for this table ≈ 0.4 → ratio < 1, all channels scaled
        // equally; spot-check equality of ratios rather than the exact curve
        // value (pinned elsewhere).
        let r = out[0] / inp[0];
        assert!((out[1] / inp[1] - r).abs() < 1e-5);
        assert!((out[2] / inp[2] - r).abs() < 1e-5);
        assert!(r < 1.0, "midtone-down curve must darken: {r}");
        // And the ratio must match an explicit norm computation through the
        // kernel's own formula: lum = dot(inp, y) = 0.5.
        let lum: f32 = 0.25 * 0.8 + 0.5 * 0.4 + 0.25 * 0.4;
        assert!((lum - 0.5).abs() < 1e-6);
        let idx = ((lum * 65536.0) as i32).clamp(0, 0xffff) as usize;
        let expect_ratio = t.table[idx] / lum;
        assert!((r - expect_ratio).abs() < 1e-4, "{r} vs {expect_ratio}");
    }

    #[test]
    fn fusion_two_exposures_pulls_grey_up_toward_brighter_rendition() {
        // Constant grey frame, identity curve, two exposures at mul ∈ {1, 2}
        // (stops=1, bias=1 → offset 0). The well-exposedness weight favours the
        // copy near 0.54, so the blend must land strictly between the plain
        // curve output (0.5) and the pushed copy (1.0) — analytically ≈ 0.67.
        const W: usize = 8;
        const H: usize = 8;
        let npx = W * H;
        let mut input = Vec::with_capacity(npx * 4);
        for _ in 0..npx {
            input.extend_from_slice(&[0.5f32, 0.5, 0.5, 0.75]);
        }
        let mut output = vec![0.0f32; npx * 4];
        let t = build_table(&[(0.0, 0.0), (1.0, 1.0)], crate::curve_tools::MONOTONE_HERMITE);
        process_fusion(
            &input,
            &mut output,
            W,
            H,
            &t.table[..],
            &t.unbounded_coeffs,
            1, // LUMINANCE
            1.0,
            1, // two exposures
            1.0,
            Some(crate::color::SRGB_TO_XYZ_D65_Y_ROW),
        );
        for k in 0..npx {
            let v = output[k * 4];
            assert!(
                v > 0.60 && v < 0.75,
                "grey must blend toward the brighter exposure, got {v}"
            );
            assert_eq!(output[k * 4 + 1], v, "constant grey must stay neutral");
            assert_eq!(output[k * 4 + 3], 0.75, "alpha comes from the input");
        }
    }

    #[test]
    fn fusion_with_identity_curve_and_one_level_matches_plain_curve_on_flat_input() {
        // For a CONSTANT image every pyramid detail vanishes, so with weights
        // equal across exposures the fusion reduces to the weighted mean of
        // curve(mul·v). With a single exposure (fusion=1 but identical muls —
        // use stops so small that 2^stops ≈ 1) it must match the LUT path.
        const W: usize = 8;
        const H: usize = 8;
        let npx = W * H;
        let mut input = Vec::with_capacity(npx * 4);
        for _ in 0..npx {
            input.extend_from_slice(&[0.42f32, 0.42, 0.42, 1.0]);
        }
        let mut out_fusion = vec![0.0f32; npx * 4];
        let mut out_lut = vec![0.0f32; npx * 4];
        // A darkening curve so the value is non-trivial.
        let t = build_table(
            &[(0.0f32, 0.0f32), (0.5, 0.3), (1.0, 1.0)],
            crate::curve_tools::MONOTONE_HERMITE,
        );
        let eps_stops = 1e-6; // 2^eps − 1 below float resolution of the blend
        process_fusion(
            &input,
            &mut out_fusion,
            W,
            H,
            &t.table[..],
            &t.unbounded_coeffs,
            1,
            eps_stops,
            1,
            1.0,
            Some(crate::color::SRGB_TO_XYZ_D65_Y_ROW),
        );
        apply_curve_pixels(&input, &mut out_lut, 1.0, &t.table[..], &t.unbounded_coeffs, 1, Some(crate::color::SRGB_TO_XYZ_D65_Y_ROW));
        for k in 0..npx {
            for c in 0..3 {
                let a = out_fusion[k * 4 + c];
                let b = out_lut[k * 4 + c];
                assert!(
                    (a - b).abs() < 2e-3,
                    "flat-input single-exposure fusion must collapse to the LUT path: {a} vs {b}"
                );
            }
        }
    }
}

// ── Gaussian pyramid (Phase 2z+56) ───────────────────────────────────────

/// Mirror-reflect index outside [0, max] (half-sample boundary convention).
/// Left: -x;  Right: 2*max+1-x.  Matches basecurve.c gauss_blur borders.
#[inline(always)]
fn bc_mirror(x: i32, max: i32) -> usize {
    if x < 0 { (-x) as usize }
    else if x > max { (2 * max + 1 - x) as usize }
    else { x as usize }
}

/// 5-tap separable Gaussian blur for a 4-channel RGBA image.
/// Kernel: [1/16, 4/16, 6/16, 4/16, 1/16]. In-place safe.
/// Matches the two DT_OMP_FOR loops in basecurve.c::gauss_blur().
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_gauss_blur(
    input:  *const f32,
    output: *mut f32,
    wd:     usize,
    ht:     usize,
) {
    const W: [f32; 5] = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0];
    let n   = wd * ht * 4;
    let inp = std::slice::from_raw_parts(input, n);
    let out = std::slice::from_raw_parts_mut(output, n);
    let wdi = wd as i32;
    let hti = ht as i32;

    // Horizontal pass into temp buffer
    let mut tmp = vec![0.0f32; n];
    for j in 0..ht {
        for i in 0..wd {
            let b = (j * wd + i) * 4;
            for c in 0..4 {
                let mut s = 0.0f32;
                for ii in -2i32..=2 {
                    let si = bc_mirror(i as i32 + ii, wdi - 1);
                    s += inp[(j * wd + si) * 4 + c] * W[(ii + 2) as usize];
                }
                tmp[b + c] = s;
            }
        }
    }

    // Vertical pass into output
    for j in 0..ht {
        for i in 0..wd {
            let b = (j * wd + i) * 4;
            for c in 0..4 {
                let mut s = 0.0f32;
                for jj in -2i32..=2 {
                    let sj = bc_mirror(j as i32 + jj, hti - 1);
                    s += tmp[(sj * wd + i) * 4 + c] * W[(jj + 2) as usize];
                }
                out[b + c] = s;
            }
        }
    }
}

/// Gaussian pyramid upsampling: fill even pixels from coarse (×4), blur.
/// Matches gauss_expand() in basecurve.c (DT_OMP_FOR(collapse(2)) + gauss_blur).
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_gauss_expand(
    coarse: *const f32,
    fine:   *mut f32,
    wd:     usize,
    ht:     usize,
) {
    let cw  = (wd - 1) / 2 + 1;
    let ch  = (ht - 1) / 2 + 1;
    let n   = wd * ht * 4;
    let fin = std::slice::from_raw_parts_mut(fine, n);
    let crs = std::slice::from_raw_parts(coarse, cw * ch * 4);

    fin.fill(0.0);
    for j in (0..ht).step_by(2) {
        for i in (0..wd).step_by(2) {
            let cb = (j / 2 * cw + i / 2) * 4;
            let fb = (j * wd + i) * 4;
            for c in 0..4 { fin[fb + c] = 4.0 * crs[cb + c]; }
        }
    }
    darkroom_basecurve_gauss_blur(fine as *const f32, fine, wd, ht);
}

/// Weight update: col0_alpha *= 0.1 + ||out_rgb||. Matches basecurve.c:1193.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_weight_update(
    col0:    *mut f32,
    out_buf: *const f32,
    npixels: usize,
) {
    let col = std::slice::from_raw_parts_mut(col0, npixels * 4);
    let out = std::slice::from_raw_parts(out_buf, npixels * 4);
    for k in (0..npixels * 4).step_by(4) {
        let mag = (out[k]*out[k] + out[k+1]*out[k+1] + out[k+2]*out[k+2]).sqrt();
        col[k + 3] *= 0.1 + mag;
    }
}

/// Blend one pyramid level into comb_k.
/// is_base!=0: comb[c] += w*col[c];  is_base==0: comb[c] += w*(col[c]-out[c]).
/// comb[3] += w always. Matches basecurve.c:1229.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_pyramid_blend(
    comb_k:  *mut f32,
    col_k:   *const f32,
    out_buf: *const f32,
    npixels: usize,
    is_base: i32,
) {
    let comb = std::slice::from_raw_parts_mut(comb_k, npixels * 4);
    let col  = std::slice::from_raw_parts(col_k,  npixels * 4);
    let out  = std::slice::from_raw_parts(out_buf, npixels * 4);
    for x in (0..npixels * 4).step_by(4) {
        let w = col[x + 3];
        if is_base != 0 {
            for c in 0..3 { comb[x+c] += w * col[x+c]; }
        } else {
            for c in 0..3 { comb[x+c] += w * (col[x+c] - out[x+c]); }
        }
        comb[x + 3] += w;
    }
}

/// Normalize RGB by alpha: if comb[x+3] > 1e-8, divide comb[x..x+3] by it.
/// Matches basecurve.c:1265.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_normalize_alpha(
    comb_k:  *mut f32,
    npixels: usize,
) {
    let comb = std::slice::from_raw_parts_mut(comb_k, npixels * 4);
    for x in (0..npixels * 4).step_by(4) {
        let a = comb[x + 3];
        if a > 1e-8 { for c in 0..3 { comb[x+c] /= a; } }
    }
}

/// Add expanded coarser level to comb_k RGB. Matches basecurve.c:1273.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_add_layers(
    comb_k:  *mut f32,
    out_buf: *const f32,
    npixels: usize,
) {
    let comb = std::slice::from_raw_parts_mut(comb_k, npixels * 4);
    let out  = std::slice::from_raw_parts(out_buf, npixels * 4);
    for x in (0..npixels * 4).step_by(4) {
        for c in 0..3 { comb[x+c] += out[x+c]; }
    }
}

/// Copy comb0 RGB (clamped ≥ 0) + in alpha to output. Matches basecurve.c:1283.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_copy_output(
    comb0:   *const f32,
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    let comb = std::slice::from_raw_parts(comb0,   npixels * 4);
    let inp  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out  = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in (0..npixels * 4).step_by(4) {
        for c in 0..3 { out[k+c] = comb[k+c].max(0.0); }
        out[k + 3] = inp[k + 3];
    }
}

// ── m4-124: pipeline-facing layer (commit_params port + orchestration) ──────

/// Port of the sampling half of `commit_params` (basecurve.c:1277-1301):
/// channel 0's anchors are sampled at 0x10000 resolution through the V1 sampler
/// (`dt_draw_curve_calc_values`), then an exponential tail is fitted over
/// x∈{0.7,0.8,0.9,1.0}·xm with lookups mirrored from the table. Basecurve is
/// single-channel — only `basecurve[0]` is ever read by the C (`const int
/// ch = 0`, :1265); channels 1/2 exist in the params struct but are reserved
/// ("maybe we'll have cam rgb at some point") — so this is exactly the
/// per-channel construction `_generate_curve_lut` performs for rgbcurve, and
/// the sampler lives once in [`super::rgbcurve::build_single_lut`], shared by
/// both modules.
///
/// The node slice must be non-empty with its last anchor at x ≤ 1 (the pipeline
/// pins endpoints before calling, as the editor guarantees).
pub fn build_table(nodes: &[(f32, f32)], curve_type: u32) -> super::rgbcurve::CurveLut {
    super::rgbcurve::build_single_lut(nodes, curve_type)
}

/// Port of `exposure_increment` (basecurve.c:566-569): the per-exposure
/// pre-scalar used by the exposure-fusion path.
pub fn exposure_increment(stops: f32, e: i32, fusion: i32, bias: f32) -> f32 {
    let offset = stops * fusion as f32 * (bias - 1.0) / 2.0;
    (2.0f32).powf(stops * e as f32 + offset)
}

/// Whole-buffer application of the curve with pre-scalar `mul` — the shared
/// entry for `process_lut` (mul = 1) and each fused exposure. `preserve_colors`
/// 0 routes to [`darkroom_basecurve_apply_legacy_curve`], anything else to
/// [`darkroom_basecurve_apply_curve`].
///
/// `luminance_y_row` plays `work_profile->matrix_in`'s Y row: only the
/// LUMINANCE norm (mode 1) consults it, as a linear profile (Y-row dot
/// product, no TRC luts — C41's working spaces are linear). The pipeline
/// passes the working space's RGB→XYZ Y row, which is what C receives from
/// `dt_ioppr_get_iop_work_profile_info` in practice (the Y row is invariant
/// under Bradford adaptation, so D65 stands in for darktable's D50-derived ICC
/// row); `None` falls back to the camera-primary weights
/// (`dt_camera_rgb_luminance`) exactly like a NULL work profile does in C.
///
/// Preview/export call these very kernels that production C calls, so the two
/// paths cannot drift.
pub fn apply_curve_pixels(
    input: &[f32],
    output: &mut [f32],
    mul: f32,
    table: &[f32],
    unbounded_coeffs: &[f32; 3],
    preserve_colors: i32,
    luminance_y_row: Option<[f32; 3]>,
) {
    debug_assert_eq!(input.len(), output.len());
    let npx = input.len() / 4;
    // The kernel reads only row 1 of matrix_in (16 floats); zero-pad the rest.
    let mut matrix = [[0.0f32; 4]; 4];
    if let Some(y) = luminance_y_row {
        matrix[1][0] = y[0];
        matrix[1][1] = y[1];
        matrix[1][2] = y[2];
    }
    unsafe {
        if preserve_colors == 0 {
            darkroom_basecurve_apply_legacy_curve(
                input.as_ptr(),
                output.as_mut_ptr(),
                npx,
                mul,
                table.as_ptr(),
                unbounded_coeffs.as_ptr(),
            );
        } else {
            darkroom_basecurve_apply_curve(
                input.as_ptr(),
                output.as_mut_ptr(),
                npx,
                mul,
                preserve_colors,
                table.as_ptr(),
                unbounded_coeffs.as_ptr(),
                luminance_y_row.is_some() as i32,
                matrix.as_ptr() as *const f32,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0, // lutsize: unused without the TRC luts
                0, // nonlinearlut: linear profile — Y-row path only
            );
        }
    }
}

/// Port of the `gauss_reduce` static inline (basecurve.c:1051-1087): blur at
/// full resolution, keep every second pixel into `coarse`, and optionally write
/// the laplacian detail `input − expand(coarse)` into `detail`.
fn gauss_reduce(
    input: &[f32],
    coarse: &mut [f32],
    detail: Option<&mut [f32]>,
    wd: usize,
    ht: usize,
) {
    let cw = (wd - 1) / 2 + 1;
    let chh = (ht - 1) / 2 + 1;
    // C allocates a scratch buffer for the blur result; on OOM it degrades to
    // blurring in place — Rust allocation failure aborts instead.
    let mut blurred = vec![0.0f32; wd * ht * 4];
    unsafe { darkroom_basecurve_gauss_blur(input.as_ptr(), blurred.as_mut_ptr(), wd, ht) };
    for j in 0..chh {
        for i in 0..cw {
            for c in 0..4 {
                coarse[4 * (j * cw + i) + c] = blurred[4 * (2 * j * wd + 2 * i) + c];
            }
        }
    }
    if let Some(detail) = detail {
        unsafe { darkroom_basecurve_gauss_expand(coarse.as_ptr(), detail.as_mut_ptr(), wd, ht) };
        for k in 0..wd * ht * 4 {
            detail[k] = input[k] - detail[k];
        }
    }
}

/// Dimensions of pyramid level `k` for a whole frame of `wd × ht`.
fn level_dims(wd: usize, ht: usize, k: usize) -> (usize, usize) {
    let mut w = wd;
    let mut h = ht;
    for _ in 0..k {
        w = (w - 1) / 2 + 1;
        h = (h - 1) / 2 + 1;
    }
    (w, h)
}

/// Port of `process_fusion` (basecurve.c:1090-1219): blend `fusion + 1`
/// exposure-shifted, curve-graded copies of the frame through a laplacian
/// pyramid, weighting each pixel toward its best-exposed rendition.
/// `output` doubles as the C's scratch buffer (it abuses the output for the
/// detail/expand temporaries), which is safe because the final
/// [`darkroom_basecurve_copy_output`] overwrites every pixel.
///
/// `rad` is `MIN(wd, ceil(256 · scale/iscale))` in the C; this stage never
/// tiles, so scale ≡ iscale ≡ 1 and rad collapses to `min(wd, 256)`. Levels are
/// capped at 8 like the C, stopping early once a level drops below 4×4 or the
/// step exceeds rad.
#[allow(clippy::too_many_arguments)]
pub fn process_fusion(
    input: &[f32],
    output: &mut [f32],
    wd: usize,
    ht: usize,
    table: &[f32],
    unbounded_coeffs: &[f32; 3],
    preserve_colors: i32,
    stops: f32,
    fusion: i32,
    bias: f32,
    luminance_y_row: Option<[f32; 3]>,
) {
    debug_assert!(fusion >= 1, "fusion==0 must take the process_lut path");
    let npx = wd * ht;
    // Pyramid allocation loop (:1107-1121): allocate level k at its own dims,
    // then shrink for k+1; stop once step > rad or the level is < 4 wide/tall.
    let rad = wd.min(256usize);
    let mut col: Vec<Vec<f32>> = Vec::new();
    let mut comb: Vec<Vec<f32>> = Vec::new();
    let mut num_levels = 8usize;
    let (mut w, mut h) = (wd, ht);
    let mut step = 1usize;
    for k in 0..8 {
        col.push(vec![0.0f32; 4 * w * h]);
        comb.push(vec![0.0f32; 4 * w * h]);
        w = (w - 1) / 2 + 1;
        h = (h - 1) / 2 + 1;
        step *= 2;
        if step > rad || w < 4 || h < 4 {
            num_levels = k + 1;
            break;
        }
    }

    for e in 0..=fusion {
        // Curve-grade the full-res input into col[0], push features into its
        // alpha lane (:1123-1131), then build the gaussian pyramid while
        // harvesting the level-0 laplacian detail for the weight update.
        let mul = exposure_increment(stops, e, fusion, bias);
        // C's process_fusion fetches the same work profile process_lut does and
        // hands it to apply_curve (:1127) — so the Y row flows through here too.
        apply_curve_pixels(input, &mut col[0], mul, table, unbounded_coeffs, preserve_colors, luminance_y_row);
        unsafe { darkroom_basecurve_compute_features(col[0].as_mut_ptr(), npx) };
        let (mut w, mut h) = (wd, ht);
        {
            // gauss_reduce(col[0], col[1], out, wd, ht): level-0 detail lands
            // in the output scratch, then feeds the weight update below.
            let (head, tail) = col.split_at_mut(1);
            gauss_reduce(&head[0], &mut tail[0], Some(&mut output[..]), w, h);
        }
        unsafe { darkroom_basecurve_weight_update(col[0].as_mut_ptr(), output.as_ptr(), npx) };
        for k in 1..num_levels {
            let (lower, upper) = col.split_at_mut(k);
            gauss_reduce(&lower[k - 1], &mut upper[0], None, w, h);
            w = (w - 1) / 2 + 1;
            h = (h - 1) / 2 + 1;
        }

        // Blend coarse → fine (:1160-1182): comb[k] += weight·(level or detail),
        // expanding the next-coarser colour buffer through the output scratch.
        for k in (0..num_levels).rev() {
            let (w, h) = level_dims(wd, ht, k);
            if k != num_levels - 1 {
                let (_coarser, same_level_plus_one) = col.split_at_mut(k + 1);
                unsafe {
                    darkroom_basecurve_gauss_expand(
                        same_level_plus_one[0].as_ptr(),
                        output.as_mut_ptr(),
                        w,
                        h,
                    )
                };
            }
            let npixels = w * h;
            let is_base = i32::from(k == num_levels - 1);
            unsafe {
                darkroom_basecurve_pyramid_blend(
                    comb[k].as_mut_ptr(),
                    col[k].as_ptr(),
                    output.as_ptr(),
                    npixels,
                    is_base,
                )
            };
        }
    }

    // Normalise weights and reconstruct coarse → fine (:1184-1207).
    for k in (0..num_levels).rev() {
        let (w, h) = level_dims(wd, ht, k);
        unsafe { darkroom_basecurve_normalize_alpha(comb[k].as_mut_ptr(), w * h) };
        if k < num_levels - 1 {
            unsafe {
                darkroom_basecurve_gauss_expand(comb[k + 1].as_ptr(), output.as_mut_ptr(), w, h)
            };
            unsafe { darkroom_basecurve_add_layers(comb[k].as_mut_ptr(), output.as_ptr(), w * h) };
        }
    }
    unsafe { darkroom_basecurve_copy_output(comb[0].as_ptr(), input.as_ptr(), output.as_mut_ptr(), npx) };
}
