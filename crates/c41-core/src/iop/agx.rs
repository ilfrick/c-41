use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Agx;

impl IopProcess for Agx {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "agx" }
}

const EPSILON: f32 = 1e-6;

/// Mirror of tone_mapping_params_t from src/iop/agx.c.
/// gboolean (= gint = int32) fields are mapped to i32.
/// Layout must match the C struct exactly.
#[repr(C)]
pub struct AgxToneMappingParams {
    pub black_relative_ev: f32,
    pub white_relative_ev: f32,
    pub range_in_ev: f32,
    pub curve_gamma: f32,
    pub pivot_x: f32,
    pub pivot_y: f32,
    pub target_black: f32,
    pub toe_power: f32,
    pub toe_transition_x: f32,
    pub toe_transition_y: f32,
    pub toe_scale: f32,
    pub need_convex_toe: i32,
    pub toe_fallback_coefficient: f32,
    pub toe_fallback_power: f32,
    pub slope: f32,
    pub intercept: f32,
    pub target_white: f32,
    pub shoulder_power: f32,
    pub shoulder_transition_x: f32,
    pub shoulder_transition_y: f32,
    pub shoulder_scale: f32,
    pub need_concave_shoulder: i32,
    pub shoulder_fallback_coefficient: f32,
    pub shoulder_fallback_power: f32,
    pub look_lift: f32,
    pub look_slope: f32,
    pub look_power: f32,
    pub look_saturation: f32,
    pub look_original_hue_mix_ratio: f32,
    pub look_tuned: i32,
    pub restore_hue: i32,
}

// --- matrix helpers (dt_colormatrix_t = float[4][4] row-major, m[c][r] = flat[c*4+r]) ---

#[inline(always)]
fn apply_transposed_3(pix: [f32; 3], m: &[f32]) -> [f32; 3] {
    [
        m[0]*pix[0] + m[4]*pix[1] + m[8]*pix[2],
        m[1]*pix[0] + m[5]*pix[1] + m[9]*pix[2],
        m[2]*pix[0] + m[6]*pix[1] + m[10]*pix[2],
    ]
}

#[inline(always)]
fn luminance_from_matrix(pix: [f32; 3], m: &[f32]) -> f32 {
    // xyz[1] = Y
    m[1]*pix[0] + m[5]*pix[1] + m[9]*pix[2]
}

// --- HSV conversions (ported from colorspaces_inline_conversions.h) ---

#[inline(always)]
fn rgb_2_hue(r: f32, g: f32, b: f32, max: f32, delta: f32) -> f32 {
    let hue = if r == max {
        (g - b) / delta
    } else if g == max {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };
    let h = hue / 6.0;
    h - h.floor()
}

#[inline(always)]
fn hue_2_rgb(h: f32, c: f32, min: f32) -> [f32; 3] {
    let h6 = h * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let fc = f * c;
    let top = c + min;
    let inc = fc + min;
    let dec = top - fc;
    match i as usize {
        0 => [top, inc, min],
        1 => [dec, top, min],
        2 => [min, top, inc],
        3 => [min, dec, top],
        4 => [inc, min, top],
        _ => [top, min, dec],
    }
}

#[inline(always)]
fn rgb_2_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let (s, hue) = if max.abs() > 1e-6 && delta.abs() > 1e-6 {
        (delta / max, rgb_2_hue(r, g, b, max, delta))
    } else {
        (0.0, 0.0)
    };
    [hue, s, max]
}

#[inline(always)]
fn hsv_2_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let c = hsv[1] * hsv[2]; // S * V = chroma
    let m = hsv[2] - c;      // min
    hue_2_rgb(hsv[0], c, m)
}

// --- tone curve helpers ---

#[inline(always)]
fn sigmoid(x: f32, power: f32) -> f32 {
    x / (1.0 + x.powf(power)).powf(1.0 / power)
}

#[inline(always)]
fn scaled_sigmoid(x: f32, scale: f32, slope: f32, power: f32, tx: f32, ty: f32) -> f32 {
    scale * sigmoid(slope * (x - tx) / scale, power) + ty
}

#[inline(always)]
fn fallback_toe(x: f32, p: &AgxToneMappingParams) -> f32 {
    if x < 0.0 {
        p.target_black
    } else {
        p.target_black + (p.toe_fallback_coefficient * x.powf(p.toe_fallback_power)).max(0.0)
    }
}

#[inline(always)]
fn fallback_shoulder(x: f32, p: &AgxToneMappingParams) -> f32 {
    if x >= 1.0 {
        p.target_white
    } else {
        p.target_white
            - (p.shoulder_fallback_coefficient * (1.0 - x).powf(p.shoulder_fallback_power)).max(0.0)
    }
}

#[inline(always)]
fn apply_curve(x: f32, p: &AgxToneMappingParams) -> f32 {
    let result = if x < p.toe_transition_x {
        if p.need_convex_toe != 0 {
            fallback_toe(x, p)
        } else {
            scaled_sigmoid(x, p.toe_scale, p.slope, p.toe_power, p.toe_transition_x, p.toe_transition_y)
        }
    } else if x <= p.shoulder_transition_x {
        p.slope * x + p.intercept
    } else if p.need_concave_shoulder != 0 {
        fallback_shoulder(x, p)
    } else {
        scaled_sigmoid(x, p.shoulder_scale, p.slope, p.shoulder_power, p.shoulder_transition_x, p.shoulder_transition_y)
    };
    result.clamp(p.target_black, p.target_white)
}

#[inline(always)]
fn apply_log_encoding(x: f32, range_in_ev: f32, black_relative_ev: f32) -> f32 {
    let x_rel = (x / 0.18).max(EPSILON);
    let mapped = (x_rel.log2() - black_relative_ev) / range_in_ev;
    mapped.clamp(0.0, 1.0)
}

// f_ss/f_ts from https://www.desmos.com/calculator/yrysofmx8h
#[inline(always)]
fn apply_slope_lift(x: f32, slope: f32, lift: f32) -> f32 {
    let m = slope / (1.0 + lift);
    m * x + lift * m
}

#[inline(always)]
fn lerp_hue(original: f32, processed: f32, mix: f32) -> f32 {
    // shortest signed distance on hue circle via IEEE remainder
    let d = processed - original;
    let shortest = d - d.round(); // remainderf(d, 1.0) for divisor=1
    let mixed = (1.0 - mix) * shortest + original;
    mixed - mixed.floor()
}

// --- gamut compression (Blender AgX luminance compensation) ---

fn compress_into_gamut(rgb: &mut [f32; 3]) {
    const LUM: [f32; 3] = [0.2658180370250449, 0.59846986045365, 0.1357121025213052];

    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let input_y = r * LUM[0] + g * LUM[1] + b * LUM[2];
    let max_rgb = r.max(g).max(b);

    let opp = [max_rgb - r, max_rgb - g, max_rgb - b];
    let opponent_y = opp[0]*LUM[0] + opp[1]*LUM[1] + opp[2]*LUM[2];
    let max_opponent = opp[0].max(opp[1]).max(opp[2]);
    let y_compensate_negative = max_opponent - opponent_y + input_y;

    let min_rgb = r.min(g).min(b);
    let offset = (-min_rgb).max(0.0);
    let rgb_off = [r + offset, g + offset, b + offset];
    let max_off = rgb_off[0].max(rgb_off[1]).max(rgb_off[2]);

    let opp_off = [max_off - rgb_off[0], max_off - rgb_off[1], max_off - rgb_off[2]];
    let max_inv_off = opp_off[0].max(opp_off[1]).max(opp_off[2]);
    let y_inv_off = opp_off[0]*LUM[0] + opp_off[1]*LUM[1] + opp_off[2]*LUM[2];
    let y_new_base = rgb_off[0]*LUM[0] + rgb_off[1]*LUM[1] + rgb_off[2]*LUM[2];
    let y_new = max_inv_off - y_inv_off + y_new_base;

    let ratio = if y_new > y_compensate_negative && y_new > EPSILON {
        y_compensate_negative / y_new
    } else {
        1.0
    };

    rgb[0] = ratio * rgb_off[0];
    rgb[1] = ratio * rgb_off[1];
    rgb[2] = ratio * rgb_off[2];
}

// --- look adjustments (slope/lift/power + saturation) ---

fn agx_look(rgb: &mut [f32; 3], p: &AgxToneMappingParams, m_rxyz: &[f32]) {
    for k in 0..3 {
        let v = apply_slope_lift(rgb[k], p.look_slope, p.look_lift);
        rgb[k] = if v > 0.0 { v.powf(p.look_power) } else { v };
    }
    let luma = luminance_from_matrix(*rgb, m_rxyz);
    for k in 0..3 {
        rgb[k] = luma + p.look_saturation * (rgb[k] - luma);
    }
}

// --- full per-pixel tone mapping pipeline ---

fn agx_tone_mapping(rgb: &mut [f32; 3], p: &AgxToneMappingParams, m_rxyz: &[f32]) {
    let h_before = if p.restore_hue != 0 { rgb_2_hsv(*rgb)[0] } else { 0.0 };

    let mut transformed = [0.0f32; 3];
    for k in 0..3 {
        let log_val = apply_log_encoding(rgb[k], p.range_in_ev, p.black_relative_ev);
        transformed[k] = apply_curve(log_val, p);
    }

    if p.look_tuned != 0 {
        agx_look(&mut transformed, p, m_rxyz);
    }

    for k in 0..3 {
        transformed[k] = transformed[k].max(0.0).powf(p.curve_gamma);
    }

    if p.restore_hue != 0 {
        let hsv_after = rgb_2_hsv(transformed);
        let h_after = lerp_hue(h_before, hsv_after[0], p.look_original_hue_mix_ratio);
        let result = hsv_2_rgb([h_after, hsv_after[1], hsv_after[2]]);
        *rgb = result;
    } else {
        *rgb = transformed;
    }
}

/// AgX IOP — full tone mapping pipeline including gamut compression and look.
///
/// Replaces the DT_OMP_FOR loop in src/iop/agx.c::process().
/// pipe_to_base / base_to_rendering / rendering_to_pipe / rendering_to_xyz:
///   each 16 floats (dt_colormatrix_t = float[4][4] row-major, transposed form).
/// base_working_same_profile: non-zero skips pipe_to_base matrix.
/// params: pointer to tone_mapping_params_t (same ABI as AgxToneMappingParams).
#[no_mangle]
pub unsafe extern "C" fn darkroom_agx_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    pipe_to_base: *const f32,
    base_to_rendering: *const f32,
    rendering_to_pipe: *const f32,
    rendering_to_xyz: *const f32,
    base_working_same_profile: i32,
    params: *const AgxToneMappingParams,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let m_pb = std::slice::from_raw_parts(pipe_to_base, 16);
    let m_br = std::slice::from_raw_parts(base_to_rendering, 16);
    let m_rp = std::slice::from_raw_parts(rendering_to_pipe, 16);
    let m_rxyz = std::slice::from_raw_parts(rendering_to_xyz, 16);
    let p = &*params;

    for k in 0..npixels {
        let pix = &input[k * 4..k * 4 + 4];
        let alpha = pix[3];

        let sanitised = [
            if pix[0].is_nan() { 0.0 } else { pix[0].clamp(-1e6, 1e6) },
            if pix[1].is_nan() { 0.0 } else { pix[1].clamp(-1e6, 1e6) },
            if pix[2].is_nan() { 0.0 } else { pix[2].clamp(-1e6, 1e6) },
        ];

        let mut base = if base_working_same_profile != 0 {
            sanitised
        } else {
            apply_transposed_3(sanitised, m_pb)
        };

        compress_into_gamut(&mut base);

        let mut rendering = apply_transposed_3(base, m_br);

        agx_tone_mapping(&mut rendering, p, m_rxyz);

        let result = apply_transposed_3(rendering, m_rp);

        output[k * 4]     = result[0];
        output[k * 4 + 1] = result[1];
        output[k * 4 + 2] = result[2];
        output[k * 4 + 3] = alpha;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_encoding_midgray_maps_to_half() {
        // 0.18 mid-gray with range=24, black=-12 → (log2(1) + 12) / 24 = 0.5
        let result = apply_log_encoding(0.18, 24.0, -12.0);
        assert!((result - 0.5).abs() < 1e-5, "got {result}");
    }

    #[test]
    fn log_encoding_clips_to_zero_one() {
        assert_eq!(apply_log_encoding(0.0, 24.0, -12.0), 0.0);
        assert_eq!(apply_log_encoding(1e10, 24.0, -12.0), 1.0);
    }

    #[test]
    fn compress_gamut_gray_unchanged() {
        let mut rgb = [0.5f32, 0.5, 0.5];
        compress_into_gamut(&mut rgb);
        assert!((rgb[0] - 0.5).abs() < 1e-5);
        assert!((rgb[1] - 0.5).abs() < 1e-5);
        assert!((rgb[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn compress_gamut_no_negatives_in_output() {
        // out-of-gamut pixel with a negative component
        let mut rgb = [-0.3f32, 0.8, 0.4];
        compress_into_gamut(&mut rgb);
        assert!(rgb[0] >= 0.0, "R should be non-negative after compression");
        assert!(rgb[1] >= 0.0);
        assert!(rgb[2] >= 0.0);
    }

    #[test]
    fn hsv_round_trip() {
        let original = [0.2f32, 0.7, 0.4];
        let hsv = rgb_2_hsv(original);
        let back = hsv_2_rgb(hsv);
        for c in 0..3 {
            assert!((back[c] - original[c]).abs() < 1e-5, "channel {c}: {} vs {}", back[c], original[c]);
        }
    }

    #[test]
    fn lerp_hue_zero_mix_returns_original() {
        let h = lerp_hue(0.3, 0.7, 1.0); // mix=1 → original
        assert!((h - 0.3).abs() < 1e-5, "got {h}");
    }

    #[test]
    fn struct_size_matches_c() {
        // tone_mapping_params_t has 31 x 4-byte fields = 124 bytes
        assert_eq!(std::mem::size_of::<AgxToneMappingParams>(), 124);
    }
}
