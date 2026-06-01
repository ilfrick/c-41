use crate::{color, params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Colorharmonizer;

impl IopProcess for Colorharmonizer {
    fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colorharmonizer" }
}

// ── inline helpers mirroring the C statics ───────────────────────────────────

/// Gaussian-weighted nearest-node hue-shift lookup.
/// Matches `get_weighted_hue_shift()` in colorharmonizer.c:170.
#[inline(always)]
fn get_weighted_hue_shift(
    px_hue: f32,
    nodes: &[f32],
    pull_width_factor: f32,
    winning_idx: &mut usize,
    max_weight:  &mut f32,
) -> f32 {
    let n = nodes.len();
    if n == 0 {
        *winning_idx = 0;
        *max_weight  = 0.0;
        return 0.0;
    }
    let sigma = pull_width_factor * 0.5 / n as f32;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

    let mut w_max      = 0.0_f32;
    let mut winner     = 0;
    let mut diff_win   = 0.0_f32;

    for (i, &node) in nodes.iter().enumerate() {
        let mut d = (px_hue - node).abs();
        if d > 0.5 { d = 1.0 - d; }
        let w = (- d * d * inv_2sigma2).exp();
        let mut diff = node - px_hue;
        if diff >  0.5 { diff -= 1.0; }
        else if diff < -0.5 { diff += 1.0; }
        if w > w_max {
            w_max    = w;
            winner   = i;
            diff_win = diff;
        }
    }
    *winning_idx = winner;
    *max_weight  = w_max;
    diff_win * w_max
}

/// Wrap hue to [0, 1). Matches `wrap_hue()` in colorharmonizer.c:213.
#[inline(always)]
fn wrap_hue(h: f32) -> f32 {
    let h = h.rem_euclid(1.0);
    if h < 0.0 { h + 1.0 } else { h }
}

// Hue conversion constants
const INV_2PI: f32 = 1.0 / (2.0 * std::f32::consts::PI);
const TWO_PI:  f32 = 2.0 * std::f32::consts::PI;

// ── FFI helpers ───────────────────────────────────────────────────────────────

/// Reconstruct a [[f32;4];4] matrix from a flat 16-float C pointer.
#[inline(always)]
unsafe fn matrix_from_ptr(p: *const f32) -> [[f32; 4]; 4] {
    let s = std::slice::from_raw_parts(p, 16);
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}

// ── Public FFI functions ──────────────────────────────────────────────────────

/// Fused single-pass colorharmonizer: RGB → JCH → hue/sat correction → RGB.
///
/// Matches the `smoothing <= 0` DT_OMP_FOR loop in colorharmonizer.c:344.
///
/// `matrix_in_transposed` and `matrix_out_transposed` are flat 16-float
/// (4×4 row-major) arrays from `work_profile->matrix_*_transposed`.
/// `nodes` is `num_nodes` floats of node hue values in [0, 1).
/// `node_saturation` is `num_nodes` floats of per-node saturation factors.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorharmonizer_fused(
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
    ch: usize,
    matrix_in_transposed:  *const f32,
    matrix_out_transposed: *const f32,
    l_white: f32,
    nodes: *const f32, num_nodes: i32,
    pull_width: f32, pull_strength: f32, cutoff: f32,
    node_saturation: *const f32,
) {
    if npixels == 0 || ch == 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf,   npixels * ch);
    let out  = std::slice::from_raw_parts_mut(out_buf, npixels * ch);
    let m_in  = matrix_from_ptr(matrix_in_transposed);
    let m_out = matrix_from_ptr(matrix_out_transposed);
    let n_nodes = num_nodes.max(0) as usize;
    let nv: &[f32] = if n_nodes == 0 || nodes.is_null() { &[] }
                     else { std::slice::from_raw_parts(nodes, n_nodes) };
    let ns: &[f32] = if n_nodes == 0 || node_saturation.is_null() { &[] }
                     else { std::slice::from_raw_parts(node_saturation, n_nodes) };

    for k in 0..npixels {
        let b = k * ch;
        let px_rgb = [inp[b].max(0.0), inp[b+1].max(0.0), inp[b+2].max(0.0), 0.0];
        let px_jch = color::rgb_to_dt_ucs_jch(&px_rgb, &m_in, l_white);

        let hue    = (px_jch[2] + std::f32::consts::PI) * INV_2PI;
        let chroma = px_jch[1];

        let mut winning_idx = 0;
        let mut max_weight  = 0.0;
        let hue_shift = get_weighted_hue_shift(hue, nv, pull_width, &mut winning_idx, &mut max_weight);
        let sat_delta = if ns.is_empty() { 0.0 } else { (ns[winning_idx] - 1.0) * max_weight };
        let chroma_weight = chroma / (chroma + cutoff + 1e-5);

        let new_hue = wrap_hue(hue + hue_shift * pull_strength * chroma_weight) * TWO_PI - std::f32::consts::PI;
        let new_ch  = (chroma * (1.0 + sat_delta * chroma_weight)).max(0.0);
        let new_jch = [px_jch[0], new_ch, new_hue, 0.0];

        let rgb_out = color::dt_ucs_jch_to_rgb(&new_jch, &m_out, l_white);
        for c in 0..ch.min(3) { out[b + c] = rgb_out[c]; }
        out[b + 3] = inp[b + 3]; // passthrough alpha
    }
}

/// Pass 1: RGB → JCH cache + per-pixel corrections.
///
/// Matches the first DT_OMP_FOR in the smoothing path (colorharmonizer.c:399).
/// `jch_cache` must be `npixels * 3` floats (J, chroma, normalised-hue).
/// `corrections` must be `npixels * 2` floats (hue_shift, sat_delta).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorharmonizer_cache_pass(
    in_buf:  *const f32,
    jch_cache:   *mut f32,
    corrections: *mut f32,
    npixels: usize, ch: usize,
    matrix_in_transposed: *const f32,
    l_white: f32,
    nodes: *const f32, num_nodes: i32,
    pull_width: f32,
    node_saturation: *const f32,
) {
    if npixels == 0 || ch == 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf, npixels * ch);
    let jchl = std::slice::from_raw_parts_mut(jch_cache, npixels * 3);
    let corr = std::slice::from_raw_parts_mut(corrections, npixels * 2);
    let m_in = matrix_from_ptr(matrix_in_transposed);
    let n = num_nodes.max(0) as usize;
    let nv: &[f32] = if n == 0 || nodes.is_null() { &[] } else { std::slice::from_raw_parts(nodes, n) };
    let ns: &[f32] = if n == 0 || node_saturation.is_null() { &[] } else { std::slice::from_raw_parts(node_saturation, n) };

    for k in 0..npixels {
        let b   = k * ch;
        let px  = [inp[b].max(0.0), inp[b+1].max(0.0), inp[b+2].max(0.0), 0.0];
        let jch = color::rgb_to_dt_ucs_jch(&px, &m_in, l_white);
        let hue = (jch[2] + std::f32::consts::PI) * INV_2PI;
        jchl[k*3]   = jch[0];
        jchl[k*3+1] = jch[1];
        jchl[k*3+2] = hue;
        let mut wi = 0; let mut mw = 0.0;
        let hs = get_weighted_hue_shift(hue, nv, pull_width, &mut wi, &mut mw);
        corr[k*2]   = hs;
        corr[k*2+1] = if ns.is_empty() { 0.0 } else { (ns[wi] - 1.0) * mw };
    }
}

/// Pass 2: apply (Gaussian-blurred) corrections using cached JCH.
///
/// Matches the second DT_OMP_FOR in the smoothing path (colorharmonizer.c:437).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorharmonizer_apply_pass(
    in_buf:  *const f32,
    out_buf: *mut f32,
    jch_cache:   *const f32,
    corrections: *const f32,
    npixels: usize, ch: usize,
    matrix_out_transposed: *const f32,
    l_white: f32,
    cutoff: f32, pull_strength: f32,
) {
    if npixels == 0 || ch == 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf, npixels * ch);
    let out  = std::slice::from_raw_parts_mut(out_buf, npixels * ch);
    let jchl = std::slice::from_raw_parts(jch_cache, npixels * 3);
    let corr = std::slice::from_raw_parts(corrections, npixels * 2);
    let m_out = matrix_from_ptr(matrix_out_transposed);

    for k in 0..npixels {
        let b      = k * ch;
        let j      = jchl[k*3];
        let chroma = jchl[k*3+1];
        let hue    = jchl[k*3+2];
        let cw     = chroma / (chroma + cutoff + 1e-5);
        let new_hue = wrap_hue(hue + corr[k*2] * pull_strength * cw) * TWO_PI - std::f32::consts::PI;
        let new_ch  = (chroma * (1.0 + corr[k*2+1] * cw)).max(0.0);
        let jch = [j, new_ch, new_hue, 0.0];
        let rgb = color::dt_ucs_jch_to_rgb(&jch, &m_out, l_white);
        for c in 0..ch.min(3) { out[b + c] = rgb[c]; }
        out[b + 3] = inp[b + 3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // sRGB→XYZ D50 transposed (approximate)
    const M_IN: [f32; 16] = [
        0.4360747, 0.2225045, 0.0139322, 0.0,
        0.3850649, 0.7168786, 0.0971045, 0.0,
        0.1430804, 0.0606169, 0.7141733, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    // XYZ D50→sRGB transposed (approximate inverse)
    const M_OUT: [f32; 16] = [
        3.1338561, -0.9787684,  0.0719453, 0.0,
       -1.6168667,  1.9161415, -0.2289914, 0.0,
       -0.4906146,  0.0334540,  1.4052427, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    const L_WHITE: f32 = 0.988; // approx Y_to_dt_UCS_L_star(1.0)

    #[test]
    fn fused_grey_pixel_is_neutral() {
        // Pure grey has zero chroma → chroma_weight=0 → no correction → passthrough
        let inp  = vec![0.5_f32, 0.5, 0.5, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let nodes = [0.5_f32]; // one node at hue=0.5
        let ns    = [1.0_f32]; // no saturation change
        unsafe {
            darkroom_colorharmonizer_fused(
                inp.as_ptr(), out.as_mut_ptr(), 1, 4,
                M_IN.as_ptr(), M_OUT.as_ptr(), L_WHITE,
                nodes.as_ptr(), 1, 0.3, 0.5, 0.0, ns.as_ptr(),
            );
        }
        for c in 0..3 { assert!((out[c] - inp[c]).abs() < 0.05, "c={c}: out={}", out[c]); }
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn cache_and_apply_match_fused() {
        // With pull_strength and no blur, cache+apply should equal fused
        let inp  = vec![0.8_f32, 0.2, 0.1, 0.9];
        let mut out_fused = vec![0.0_f32; 4];
        let mut out_split = vec![0.0_f32; 4];
        let nodes = [0.1_f32, 0.5, 0.9];
        let ns    = [1.0_f32, 1.0, 1.0];
        let cutoff = 0.002_f32;
        let ps     = 0.8_f32;
        let pw     = 0.5_f32;

        unsafe {
            // Fused pass
            darkroom_colorharmonizer_fused(
                inp.as_ptr(), out_fused.as_mut_ptr(), 1, 4,
                M_IN.as_ptr(), M_OUT.as_ptr(), L_WHITE,
                nodes.as_ptr(), 3, pw, ps, cutoff, ns.as_ptr(),
            );
            // Split pass (no blur)
            let mut jchl = vec![0.0_f32; 3];
            let mut corr = vec![0.0_f32; 2];
            darkroom_colorharmonizer_cache_pass(
                inp.as_ptr(), jchl.as_mut_ptr(), corr.as_mut_ptr(), 1, 4,
                M_IN.as_ptr(), L_WHITE,
                nodes.as_ptr(), 3, pw, ns.as_ptr(),
            );
            darkroom_colorharmonizer_apply_pass(
                inp.as_ptr(), out_split.as_mut_ptr(),
                jchl.as_ptr(), corr.as_ptr(), 1, 4,
                M_OUT.as_ptr(), L_WHITE, cutoff, ps,
            );
        }
        for c in 0..4 {
            assert!((out_split[c] - out_fused[c]).abs() < 1e-4, "c={c}: fused={} split={}", out_fused[c], out_split[c]);
        }
    }

    #[test]
    fn alpha_passthrough() {
        let inp = vec![0.4_f32, 0.6, 0.2, 0.77];
        let mut out = vec![-1.0_f32; 4];
        let ns = [1.0_f32];
        unsafe {
            darkroom_colorharmonizer_fused(
                inp.as_ptr(), out.as_mut_ptr(), 1, 4,
                M_IN.as_ptr(), M_OUT.as_ptr(), L_WHITE,
                std::ptr::null(), 0, 0.3, 0.5, 0.001, ns.as_ptr(),
            );
        }
        assert_eq!(out[3], 0.77);
    }
}
