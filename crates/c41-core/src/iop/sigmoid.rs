use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct Sigmoid;

impl IopProcess for Sigmoid {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "sigmoid" }
}

/// film_response = (film_fog + max(value,0))^film_power
/// paper_response = magnitude * (film_response / (paper_exp + film_response))^paper_power
#[inline(always)]
fn loglogistic_sigmoid(value: f32, magnitude: f32, paper_exp: f32, film_fog: f32, film_power: f32, paper_power: f32) -> f32 {
    let clamped = value.max(0.0);
    let film_response = (film_fog + clamped).powf(film_power);
    let paper_response = magnitude * (film_response / (paper_exp + film_response)).powf(paper_power);
    if paper_response.is_nan() { magnitude } else { paper_response }
}

/// Desaturate any negative channels toward the achromatic axis.
#[inline(always)]
fn desaturate_negative(pix: [f32; 4]) -> [f32; 4] {
    let avg = ((pix[0] + pix[1] + pix[2]) / 3.0).max(0.0);
    let min_v = pix[0].min(pix[1]).min(pix[2]);
    let sf = if min_v < 0.0 { -avg / (min_v - avg) } else { 1.0 };
    [
        avg + sf * (pix[0] - avg),
        avg + sf * (pix[1] - avg),
        avg + sf * (pix[2] - avg),
        pix[3],
    ]
}

/// Returns (min_idx, mid_idx, max_idx) sorted by RGB value.
#[inline(always)]
fn channel_order(p: [f32; 3]) -> (usize, usize, usize) {
    if p[0] >= p[1] {
        if p[1] > p[2]      { (2, 1, 0) }
        else if p[2] > p[0] { (1, 0, 2) }
        else if p[2] > p[1] { (1, 2, 0) }
        else                { (2, 1, 0) }
    } else if p[0] >= p[2]  { (2, 0, 1) }
    else if p[2] > p[1]     { (0, 1, 2) }
    else                    { (0, 2, 1) }
}

/// Middle-grey reference (matches sigmoid.c `MIDDLE_GREY`).
const MIDDLE_GREY: f32 = 0.1845;

/// Derive the rgb-ratio sigmoid process parameters from the user-facing controls
/// — `contrast` (middle_grey_contrast), `skew` (contrast_skewness), and the
/// display white/black targets as percentages — mirroring `commit_params` in
/// sigmoid.c. Returns `[white_target, black_target, paper_exposure, film_fog,
/// film_power, paper_power]`: the trailing args of
/// [`darkroom_sigmoid_rgb_ratio_process`] (its `contrast_power` = `film_power`,
/// `skew_power` = `paper_power`). The curve is built so f(0)=black_target,
/// f(MIDDLE_GREY)=MIDDLE_GREY, f(∞)=white_target, with slope at grey set only by
/// `contrast`.
pub fn rgb_ratio_params(contrast: f32, skew: f32, white_pct: f32, black_pct: f32) -> [f32; 6] {
    let mg = MIDDLE_GREY;
    let delta = 1e-6f32;

    // Reference slope at grey for no skew on a normalised display.
    let ref_film_power = contrast;
    let (ref_paper_power, ref_magnitude, ref_film_fog) = (1.0f32, 1.0f32, 0.0f32);
    let ref_paper_exposure = (ref_film_fog + mg).powf(ref_film_power) * (ref_magnitude / mg - 1.0);
    let ref_slope = (loglogistic_sigmoid(mg + delta, ref_magnitude, ref_paper_exposure, ref_film_fog, ref_film_power, ref_paper_power)
        - loglogistic_sigmoid(mg - delta, ref_magnitude, ref_paper_exposure, ref_film_fog, ref_film_power, ref_paper_power))
        / 2.0 / delta;

    // Skew maps to paper_power; find the film_power that restores the ref slope.
    let paper_power = 5.0f32.powf(-skew);
    let temp_film_power = 1.0f32;
    let temp_white_target = 0.01 * white_pct;
    let temp_white_grey_relation = (temp_white_target / mg).powf(1.0 / paper_power) - 1.0;
    let temp_paper_exposure = mg.powf(temp_film_power) * temp_white_grey_relation;
    let temp_slope = (loglogistic_sigmoid(mg + delta, temp_white_target, temp_paper_exposure, ref_film_fog, temp_film_power, paper_power)
        - loglogistic_sigmoid(mg - delta, temp_white_target, temp_paper_exposure, ref_film_fog, temp_film_power, paper_power))
        / 2.0 / delta;
    let film_power = ref_slope / temp_slope;

    // Remaining parameters now that film/paper power are known.
    let white_target = 0.01 * white_pct;
    let black_target = 0.01 * black_pct;
    let white_grey_relation = (white_target / mg).powf(1.0 / paper_power) - 1.0;
    let white_black_relation = (black_target / white_target).powf(-1.0 / paper_power) - 1.0;
    let film_fog = mg * white_grey_relation.powf(1.0 / film_power)
        / (white_black_relation.powf(1.0 / film_power) - white_grey_relation.powf(1.0 / film_power));
    let paper_exposure = (film_fog + mg).powf(film_power) * white_grey_relation;

    [white_target, black_target, paper_exposure, film_fog, film_power, paper_power]
}

/// Hue + energy preserving correction: blends per-channel result toward hue-correct result.
#[inline(always)]
fn preserve_hue_and_energy(
    pix_in: [f32; 4],
    pc: [f32; 4],
    order: (usize, usize, usize),
    hue_preservation: f32,
) -> [f32; 4] {
    let (imin, imid, imax) = order;
    let chroma = pix_in[imax] - pix_in[imin];
    let midscale = if chroma != 0.0 { (pix_in[imid] - pix_in[imin]) / chroma } else { 0.0 };
    let full_hue_correction = pc[imin] + (pc[imax] - pc[imin]) * midscale;
    let naive_hue_mid = (1.0 - hue_preservation) * pc[imid] + hue_preservation * full_hue_correction;

    let pc_energy = pc[0] + pc[1] + pc[2];
    let naive_energy = pc[imin] + naive_hue_mid + pc[imax];
    let sum_min_mid = pix_in[imin] + pix_in[imid];
    let blend = if sum_min_mid != 0.0 { 2.0 * pix_in[imin] / sum_min_mid } else { 0.0 };
    let energy_target = blend * pc_energy + (1.0 - blend) * naive_energy;

    let mut out = pc;
    if naive_hue_mid <= pc[imid] {
        let c_mid = ((1.0 - hue_preservation) * pc[imid]
            + hue_preservation * (midscale * pc[imax] + (1.0 - midscale) * (energy_target - pc[imax])))
            / (1.0 + hue_preservation * (1.0 - midscale));
        out[imin] = energy_target - pc[imax] - c_mid;
        out[imid] = c_mid;
        out[imax] = pc[imax];
    } else {
        let c_mid = ((1.0 - hue_preservation) * pc[imid]
            + hue_preservation * (pc[imin] * (1.0 - midscale) + midscale * (energy_target - pc[imin])))
            / (1.0 + hue_preservation * midscale);
        out[imin] = pc[imin];
        out[imid] = c_mid;
        out[imax] = energy_target - pc[imin] - c_mid;
    }
    out
}

/// dt_apply_transposed_color_matrix: out[r] = Σ_c m[c][r] * pix[c]
/// m is 16 floats (float[4][4] row-major), so m[c][r] = flat[c*4 + r].
#[inline(always)]
fn apply_transposed(pix: [f32; 4], m: &[f32]) -> [f32; 4] {
    [
        m[0]*pix[0] + m[4]*pix[1] + m[8]*pix[2]  + m[12]*pix[3],
        m[1]*pix[0] + m[5]*pix[1] + m[9]*pix[2]  + m[13]*pix[3],
        m[2]*pix[0] + m[6]*pix[1] + m[10]*pix[2] + m[14]*pix[3],
        m[3]*pix[0] + m[7]*pix[1] + m[11]*pix[2] + m[15]*pix[3],
    ]
}

/// Sigmoid IOP — RGB-ratio path: tone curve applied on average luma, ratio-scales chroma.
///
/// black_target is used for hyperbolic gamut compression bounds.
#[no_mangle]
pub unsafe extern "C" fn darkroom_sigmoid_rgb_ratio_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    white_target: f32,
    black_target: f32,
    paper_exp: f32,
    film_fog: f32,
    contrast_power: f32,
    skew_power: f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    for px in 0..npixels {
        let base = px * 4;
        let pix_in = [inp[base], inp[base+1], inp[base+2], inp[base+3]];
        let sp = desaturate_negative(pix_in);
        let luma = (sp[0] + sp[1] + sp[2]) / 3.0;
        let mapped_luma = loglogistic_sigmoid(luma, white_target, paper_exp, film_fog, contrast_power, skew_power);

        let pre_out = if luma > 1e-9 {
            let sf = mapped_luma / luma;
            [sf * sp[0], sf * sp[1], sf * sp[2], pix_in[3]]
        } else {
            [mapped_luma, mapped_luma, mapped_luma, pix_in[3]]
        };

        // Hyperbolic gamut compression
        let (imin, _imid, imax) = channel_order([pre_out[0], pre_out[1], pre_out[2]]);
        let pmin = pre_out[imin];
        let pmax = pre_out[imax];
        let eps = 1e-6_f32;
        let dbvc_w = (white_target - mapped_luma) / (pmax - mapped_luma + eps);
        let dbvc_b = (black_target - mapped_luma) / (pmin - mapped_luma - eps);
        let dbvc = dbvc_w.min(dbvc_b);
        let cvm = (mapped_luma - pmin) / (mapped_luma + eps);
        let pca = 1.0 / (cvm * dbvc + eps);
        let hyp_c = 2.0 * cvm / (1.0 - cvm * cvm + eps) * pca;
        let hyp_z = (hyp_c * hyp_c + 1.0).sqrt();
        let cf = hyp_c / (1.0 + hyp_z) * dbvc;

        let o = &mut out[base..base+4];
        o[0] = mapped_luma + cf * (pre_out[0] - mapped_luma);
        o[1] = mapped_luma + cf * (pre_out[1] - mapped_luma);
        o[2] = mapped_luma + cf * (pre_out[2] - mapped_luma);
        o[3] = pix_in[3];
    }
}

/// Sigmoid IOP — per-channel path: applies sigmoid per channel with hue correction.
///
/// Geometry matrices pipe_to_base / base_to_rendering / rendering_to_pipe must be
/// pre-computed by the C caller via _calculate_adjusted_primaries. Each is 16 floats
/// (dt_colormatrix_t = float[4][4]).
#[no_mangle]
pub unsafe extern "C" fn darkroom_sigmoid_per_channel_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    white_target: f32,
    paper_exp: f32,
    film_fog: f32,
    contrast_power: f32,
    skew_power: f32,
    hue_preservation: f32,
    pipe_to_base: *const f32,
    base_to_rendering: *const f32,
    rendering_to_pipe: *const f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let m_pb = std::slice::from_raw_parts(pipe_to_base, 16);
    let m_br = std::slice::from_raw_parts(base_to_rendering, 16);
    let m_rp = std::slice::from_raw_parts(rendering_to_pipe, 16);

    for px in 0..npixels {
        let base = px * 4;
        let pix_in = [inp[base], inp[base+1], inp[base+2], inp[base+3]];

        let pix_base = apply_transposed(pix_in, m_pb);
        let pix_sp = desaturate_negative(pix_base);
        let rendering = apply_transposed(pix_sp, m_br);

        let per_channel = [
            loglogistic_sigmoid(rendering[0], white_target, paper_exp, film_fog, contrast_power, skew_power),
            loglogistic_sigmoid(rendering[1], white_target, paper_exp, film_fog, contrast_power, skew_power),
            loglogistic_sigmoid(rendering[2], white_target, paper_exp, film_fog, contrast_power, skew_power),
            rendering[3],
        ];

        let order = channel_order([rendering[0], rendering[1], rendering[2]]);
        let hue_corrected = preserve_hue_and_energy(rendering, per_channel, order, hue_preservation);
        let pix_out = apply_transposed(hue_corrected, m_rp);

        let o = &mut out[base..base+4];
        o[0] = pix_out[0];
        o[1] = pix_out[1];
        o[2] = pix_out[2];
        o[3] = pix_in[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix() -> Vec<f32> {
        // float[4][4] row-major, identity: m[r][c] = if r==c {1} else {0}
        // flat[r*4+c] = identity
        // But apply_transposed reads m[c*4+r], so for identity: m[c*4+r] = if c==r {1} else {0}
        // Both are the same for identity matrix.
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    #[test]
    fn loglogistic_at_film_fog_is_zero() {
        // At value=0, film_response = film_fog^film_power; for film_fog=0 → 0
        let v = loglogistic_sigmoid(0.0, 1.0, 1.0, 0.0, 1.0, 1.0);
        assert!(v >= 0.0);
        assert!(v < 0.1);
    }

    #[test]
    fn rgb_ratio_grey_maps_deterministically() {
        let inp = [0.5f32, 0.5, 0.5, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_sigmoid_rgb_ratio_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1.0, 0.0,  // white_target=1, black_target=0
                1.0, 0.01, 1.5, 1.0,  // paper_exp, film_fog, contrast, skew
            )
        };
        // Grey in → grey out (equal channels)
        assert!((out[0] - out[1]).abs() < 1e-6);
        assert!((out[1] - out[2]).abs() < 1e-6);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn per_channel_grey_identity_matrices() {
        let inp = [0.5f32, 0.5, 0.5, 1.0];
        let mut out = [0f32; 4];
        let id = identity_matrix();
        unsafe {
            darkroom_sigmoid_per_channel_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1.0, 1.0, 0.01, 1.5, 1.0, // white, paper_exp, film_fog, contrast, skew
                1.0,  // hue_preservation
                id.as_ptr(), id.as_ptr(), id.as_ptr(),
            )
        };
        // Grey in → equal channels out
        assert!((out[0] - out[1]).abs() < 1e-5);
        assert!((out[1] - out[2]).abs() < 1e-5);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn per_channel_alpha_preserved() {
        let inp = [0.3f32, 0.4, 0.5, 0.7];
        let mut out = [0f32; 4];
        let id = identity_matrix();
        unsafe {
            darkroom_sigmoid_per_channel_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1.0, 1.0, 0.01, 1.5, 1.0, 0.5,
                id.as_ptr(), id.as_ptr(), id.as_ptr(),
            )
        };
        assert_eq!(out[3], 0.7);
    }
}
