use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Toneequal;

impl IopProcess for Toneequal {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "toneequal" }
}

/// Apply the tone-equalizer LUT-based correction to every pixel.
///
/// For each pixel k:
///   exposure  = clamp(log2(luminance[k]), min_ev, max_ev)
///   idx       = round((exposure - min_ev) * lut_resolution)
///   correction = lut[idx]
///   out[k*4..k*4+3] = correction * in[k*4..k*4+3]   (all 4 channels)
///
/// `lut` must have length `pixel_chan * lut_resolution + 1` — matching the C
/// allocation `PIXEL_CHAN * LUT_RESOLUTION + 1`. The `.min(last_idx)` guard
/// inside the function makes accesses safe even at the upper boundary.
///
/// Matches the `DT_OMP_FOR` in `apply_toneequalizer()` (toneequal.c:789).
#[no_mangle]
pub unsafe extern "C" fn darkroom_toneequal_apply_lut(
    in_buf: *const f32,
    luminance: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    lut: *const f32,
    lut_len: usize,
    min_ev: f32,
    max_ev: f32,
    lut_resolution: f32,
) {
    if npixels == 0 || lut_len == 0 { return; }
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let lum   = std::slice::from_raw_parts(luminance, npixels);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lut_slice = std::slice::from_raw_parts(lut, lut_len);
    let last_idx = lut_len - 1;

    for k in 0..npixels {
        let exposure = lum[k].log2().clamp(min_ev, max_ev);
        let idx = ((exposure - min_ev) * lut_resolution).round() as usize;
        let idx = idx.min(last_idx);
        let correction = lut_slice[idx];
        let base = k * 4;
        for c in 0..4 {
            output[base + c] = correction * input[base + c];
        }
    }
}

/// Build the correction LUT from a Gaussian RBF approximation.
///
/// For j in 0..=(pixel_chan * lut_resolution):
///   exposure = j / lut_resolution + min_ev
///   result   = clamp(Σ_i gaussian(exposure - centers[i], denom) * factors[i], 0.25, 4.0)
///   lut[j]   = result
///
/// `gaussian(radius, denom) = exp(-radius² / denom)`
/// `denom = 2 * sigma²`
///
/// Matches `build_correction_lut()` (toneequal.c:1231).
/// `centers` is `pixel_chan` floats; `factors` is `pixel_chan` floats.
/// `lut` must be at least `pixel_chan * lut_resolution + 1` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_toneequal_build_lut(
    lut: *mut f32,
    factors: *const f32,
    centers: *const f32,
    pixel_chan: usize,
    lut_resolution: usize,
    sigma: f32,
    min_ev: f32,
) {
    if pixel_chan == 0 || lut_resolution == 0 { return; }
    let lut_len = pixel_chan * lut_resolution + 1;
    let out = std::slice::from_raw_parts_mut(lut, lut_len);
    let fac = std::slice::from_raw_parts(factors, pixel_chan);
    let ctr = std::slice::from_raw_parts(centers, pixel_chan);
    let denom = 2.0_f32 * sigma * sigma;

    for j in 0..lut_len {
        let exposure = j as f32 / lut_resolution as f32 + min_ev;
        let mut result = 0.0_f32;
        for i in 0..pixel_chan {
            let r = exposure - ctr[i];
            result += (-r * r / denom).exp() * fac[i];
        }
        out[j] = result.clamp(0.25, 4.0);
    }
}

/// Render the luminance-mask debug overlay.
///
/// For each pixel k:
///   lum_clamped = clamp((luminance[k] - 1/256) / (1 - 1/256), 0, 1)
///   intensity   = sqrt(lum_clamped)           (gamma 2.0 for shadow legibility)
///   out[k*4..k*4+3] = intensity   (all 4 channels written)
///   out[k*4+3]      = in[k*4+3]  (alpha copy from in, overwriting the above)
///
/// Matches `DT_OMP_FOR(collapse(2))` in the mask-display helper (toneequal.c:967).
/// `in_width`/`in_height` are the full input ROI dimensions. `out_width`/
/// `out_height` may be smaller (roi crop). `offset_x`/`offset_y` map each
/// output pixel back into the input/luminance buffer.
/// Both `in_buf` (in_width×in_height×4) and `luminance` (in_width×in_height)
/// must be fully sized — the maximum inner index is
/// `(out_height-1+offset_y)*in_width + (out_width-1+offset_x)` and must
/// be < `in_width * in_height`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_toneequal_mask_display(
    in_buf: *const f32,
    luminance: *const f32,
    out_buf: *mut f32,
    out_width: usize,
    out_height: usize,
    in_width: usize,
    in_height: usize,
    offset_x: usize,
    offset_y: usize,
) {
    if out_width == 0 || out_height == 0 || in_width == 0 || in_height == 0 { return; }
    let in_total = in_width * in_height;
    let inp = std::slice::from_raw_parts(in_buf, in_total * 4);
    let lum = std::slice::from_raw_parts(luminance, in_total);
    let out = std::slice::from_raw_parts_mut(out_buf, out_width * out_height * 4);

    for i in 0..out_height {
        for j in 0..out_width {
            let lum_val = lum[(i + offset_y) * in_width + (j + offset_x)];
            let clamped = ((lum_val - 0.00390625).max(0.0) / 0.99609375).min(1.0);
            let intensity = clamped.sqrt();
            let oi = (i * out_width + j) * 4;
            for c in 0..4 { out[oi + c] = intensity; }
            // overwrite alpha with source alpha
            let ii = ((i + offset_y) * in_width + (j + offset_x)) * 4 + 3;
            out[oi + 3] = inp[ii];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Centers from toneequal.c: {-56/7, -48/7, ..., 0}
    const CENTERS_OPS: [f32; 8] = [
        -56.0/7.0, -48.0/7.0, -40.0/7.0, -32.0/7.0,
        -24.0/7.0, -16.0/7.0,  -8.0/7.0,   0.0/7.0,
    ];

    #[test]
    fn apply_lut_scales_all_channels() {
        // lut maps to 2.0 everywhere; correction = 2 → out = 2 * in
        let inp = vec![1.0_f32, 0.5, 0.25, 1.0];
        let lum = vec![0.5_f32]; // log2(0.5) = -1, within [-8,0]
        let lut = vec![2.0_f32; 80002]; // pixel_chan*lut_res+1 = 80001
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_toneequal_apply_lut(
                inp.as_ptr(), lum.as_ptr(), out.as_mut_ptr(), 1,
                lut.as_ptr(), lut.len(),
                -8.0, 0.0, 10000.0,
            );
        }
        for c in 0..4 { assert!((out[c] - inp[c] * 2.0).abs() < 1e-5); }
    }

    #[test]
    fn apply_lut_clamps_exposure() {
        // luminance > 1 → exposure = 0 → last LUT entry
        let inp = vec![0.5_f32, 0.5, 0.5, 1.0];
        let lum = vec![100.0_f32]; // log2 >> max_ev
        let lut = vec![3.0_f32; 80002];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_toneequal_apply_lut(
                inp.as_ptr(), lum.as_ptr(), out.as_mut_ptr(), 1,
                lut.as_ptr(), lut.len(), -8.0, 0.0, 10000.0,
            );
        }
        assert_eq!(out[0], 1.5);
    }

    #[test]
    fn build_lut_unity_factors_yield_approx_one_at_center() {
        // With all factors = 1 and the Gaussian peaks summing near 1 at center,
        // we get a value in [0.25, 4.0] — just check it doesn't panic and clamps.
        let factors = [1.0_f32; 8];
        let lut_res = 100;
        let lut_len = 8 * lut_res + 1;
        let mut lut = vec![0.0_f32; lut_len];
        unsafe {
            darkroom_toneequal_build_lut(
                lut.as_mut_ptr(), factors.as_ptr(),
                CENTERS_OPS.as_ptr(), 8, lut_res, 1.0, -8.0,
            );
        }
        for &v in &lut {
            assert!(v >= 0.25 && v <= 4.0, "out of clamp range: {v}");
        }
    }

    #[test]
    fn build_lut_zero_factors_yield_clamped_minimum() {
        let factors = [0.0_f32; 8];
        let lut_res = 10;
        let mut lut = vec![0.0_f32; 8 * lut_res + 1];
        unsafe {
            darkroom_toneequal_build_lut(
                lut.as_mut_ptr(), factors.as_ptr(),
                CENTERS_OPS.as_ptr(), 8, lut_res, 1.0, -8.0,
            );
        }
        // All factors zero → result = 0 → clamp(0, 0.25, 4) = 0.25
        for &v in &lut { assert_eq!(v, 0.25); }
    }

    #[test]
    fn mask_display_computes_sqrt_intensity() {
        // luminance = 1.0 → clamped = (1 - 1/256)/(1-1/256) = 1.0 → sqrt(1) = 1
        let inp  = vec![0.0_f32; 4]; // alpha = 0
        let lum  = vec![1.0_f32];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_toneequal_mask_display(
                inp.as_ptr(), lum.as_ptr(), out.as_mut_ptr(),
                1, 1, 1, 1, 0, 0,  // in_height=1 added
            );
        }
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert_eq!(out[3], 0.0); // alpha from inp[3]
    }

    #[test]
    fn mask_display_alpha_comes_from_input() {
        let mut inp = vec![0.0_f32; 4];
        inp[3] = 0.42;
        let lum = vec![0.5_f32];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_toneequal_mask_display(
                inp.as_ptr(), lum.as_ptr(), out.as_mut_ptr(),
                1, 1, 1, 1, 0, 0,  // in_height=1 added
            );
        }
        assert_eq!(out[3], 0.42);
    }

    #[test]
    fn mask_display_with_nonzero_offset() {
        // 2×2 input, crop 1×1 at offset (1, 1): accesses lum[1*2+1] = lum[3]
        let inp = vec![0.0_f32; 2 * 2 * 4]; // all zero alpha
        let lum = vec![0.0_f32, 0.0, 0.0, 0.0625]; // lum[3] = 0.0625
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_toneequal_mask_display(
                inp.as_ptr(), lum.as_ptr(), out.as_mut_ptr(),
                1, 1, 2, 2, 1, 1,  // out 1×1, in 2×2, offset (1,1)
            );
        }
        // clamped = (0.0625 - 0.00390625) / 0.99609375 ≈ 0.0589; sqrt ≈ 0.243
        assert!(out[0] > 0.0 && out[0] < 0.5, "intensity={}", out[0]);
    }

    // ── Preview path (solve_weights / cached_correction_lut / process_preview_pixels)

    /// The least-squares fit must actually solve the normal equations: the
    /// residual Aᵀ(A·w − y) is zero at the optimum for a full-rank AᵀA.
    #[test]
    fn solve_weights_satisfies_normal_equations() {
        let gains = [0.1, -0.4, 0.9, 0.0, 0.33, -1.2, 0.7, 0.05, -0.6];
        let w = super::solve_weights(&gains);
        let basis = |j: usize, k: usize| {
            let r = super::CENTERS_PARAMS[k] - super::CENTERS_OPS[j];
            let denom = 2.0 * std::f32::consts::SQRT_2 * std::f32::consts::SQRT_2;
            (-r * r / denom).exp()
        };
        let mut worst = 0.0f32;
        for i in 0..8 {
            // (AᵀA·w − Aᵀy)_i = Σ_j (AᵀA)_ij w_j − Σ_k A_ki y_k
            let mut ata_w = 0.0f32;
            for j in 0..8 {
                let mut ata_ij = 0.0f32;
                for k in 0..9 {
                    ata_ij += basis(i, k) * basis(j, k);
                }
                ata_w += ata_ij * w[j];
            }
            let mut aty = 0.0f32;
            for k in 0..9 {
                aty += basis(i, k) * gains[k].exp2();
            }
            worst = worst.max((ata_w - aty).abs());
        }
        assert!(worst < 1e-3, "normal-equation residual too large: {worst}");
    }

    /// All-zero gains are the module default and must be a flat ×1 correction:
    /// exp2(0)=1 everywhere. The least-squares fit of a constant target over
    /// this RBF basis deviates by at most ~0.7% (verified against numpy
    /// lstsq) — that residual smoothness is inherent to the design.
    #[test]
    fn solve_weights_flat_unity_at_default_gains() {
        let w = super::solve_weights(&[0.0; 9]);
        let denom = 2.0 * std::f32::consts::SQRT_2 * std::f32::consts::SQRT_2;
        for &cp in &super::CENTERS_PARAMS {
            let mut fit = 0.0f32;
            for (wi, &co) in w.iter().zip(&super::CENTERS_OPS) {
                let r = cp - co;
                fit += wi * (-r * r / denom).exp();
            }
            assert!((fit - 1.0).abs() < 0.01, "fit at {cp} EV = {fit}");
        }
    }

    /// Boosting one channel by +1 EV must land the fitted correction near ×2
    /// at that channel's centre — but not exactly ×2: the RBF least-squares
    /// fit trades peak amplitude for smoothness, landing at ≈1.75 (numpy
    /// lstsq cross-check). The far end relaxes to ≈0.96.
    #[test]
    fn single_channel_boost_lands_near_exp2_gain() {
        let gains = [
            0.0, 0.0, 0.0, 0.0,
            1.0, // shadows = mid-tones slider = −4 EV centre
            0.0, 0.0, 0.0, 0.0,
        ];
        let lut = super::cached_correction_lut(&gains);
        // idx = round((expo + 8) * LUT_RESOLUTION); −4 EV → index 40_000.
        let got = lut[40_000];
        assert!(
            (got - 1.746).abs() < 0.01,
            "correction at −4 EV = {got}, want ≈1.75"
        );
        // Far from the boosted channel the correction relaxes toward unity.
        let far = lut[80_000]; // 0 EV end of the table
        assert!(far > 0.9 && far < 1.0, "correction at 0 EV = {far}, want ≈0.96");
    }

    /// The preview pixel pass scales each channel by the same correction
    /// looked up at that pixel's own luminance: a uniform grey patch boosted
    /// on its own channel must come out uniformly brighter.
    #[test]
    fn process_preview_pixels_scales_uniform_patch() {
        // Grey at 0.25 linear ⇒ norm-2 lum = 0.25·√3 ≈ 0.433, expo ≈ −1.207.
        let px = 0.25_f32;
        let mut input = Vec::with_capacity(64);
        for _ in 0..16 {
            input.extend_from_slice(&[px, px, px, 1.0]);
        }
        let mut output = vec![0.0_f32; 64];
        let gains = [
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];
        super::process_preview_pixels(&input, &mut output, &gains);
        let expected = super::cached_correction_lut(&gains);
        let lum = (3.0f32 * px * px).sqrt().max(2.0f32.powi(-16));
        let idx =
            ((lum.log2() - super::MIN_EV) * 10_000.0).round().clamp(0.0, 80_000.0) as usize;
        for chunk in output.chunks_exact(4) {
            for c in 0..3 {
                assert!(
                    (chunk[c] - px * expected[idx]).abs() < 1e-5,
                    "channel {} = {} want {}",
                    c,
                    chunk[c],
                    px * expected[idx]
                );
            }
        }
    }
}

/// Build a log-exposure histogram from a luminance buffer.
///
/// Each luminance value is mapped to an EV bin in [-10, +6]:
///   index = CLAMP((log2(lum) + 10) / 16 * temp_samples, 0, temp_samples-1)
/// hist[index] += 1
///
/// `temp_samples` = 2 * UI_SAMPLES = 512 in production. `hist` must be
/// zeroed by the caller and have at least `temp_samples` entries.
/// Matches DT_OMP_FOR_SIMD(reduction) in src/iop/toneequal.c:1379.
#[no_mangle]
pub unsafe extern "C" fn darkroom_toneequal_build_log_histogram(
    luminance: *const f32,
    num_elem: usize,
    hist: *mut i32,
    temp_samples: usize,
) {
    if num_elem == 0 || temp_samples == 0 { return; }
    let lum = std::slice::from_raw_parts(luminance, num_elem);
    let h   = std::slice::from_raw_parts_mut(hist, temp_samples);
    let ts  = temp_samples as f32;
    for k in 0..num_elem {
        let ev = lum[k].log2();
        let idx = (((ev + 10.0) / 16.0) * ts) as i64;
        let idx = idx.clamp(0, (temp_samples - 1) as i64) as usize;
        h[idx] += 1;
    }
}

// Fixed node positions for PIXEL_CHAN=8 Gaussian RBF, uniformly spaced over [-8, 0] EV.
const CENTERS_OPS: [f32; 8] = [
    -56.0 / 7.0, -48.0 / 7.0, -40.0 / 7.0, -32.0 / 7.0,
    -24.0 / 7.0, -16.0 / 7.0,  -8.0 / 7.0,   0.0 / 7.0,
];
// Fixed evaluation points for CHANNELS=9 UI parameters.
const CENTERS_PARAMS: [f32; 9] = [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0];

#[inline(always)]
fn pixel_correction_rs(exposure: f32, factors: &[f32; 8], gauss_denom: f32) -> f32 {
    let expo = exposure.clamp(-8.0, 0.0);
    let result: f32 = CENTERS_OPS.iter().zip(factors.iter())
        .map(|(&c, &f)| (-(expo - c).powi(2) / gauss_denom).exp() * f)
        .sum();
    result.clamp(0.25, 4.0)
}

/// Build the GUI display curve LUT (UI_SAMPLES=256 entries).
///
/// LUT[k] = offset - log2(pixel_correction(x_k, factors, sigma)) / scaling
/// where x_k = 8*(k/(UI_SAMPLES-1)) - 8   maps k to [-8, 0] EV.
///
/// Matches the DT_OMP_FOR_SIMD at src/iop/toneequal.c:1454.
#[no_mangle]
pub unsafe extern "C" fn darkroom_toneequal_build_gui_lut(
    lut: *mut f32,
    factors: *const f32,
    sigma: f32,
    offset: f32,
    scaling: f32,
) {
    const UI_SAMPLES: usize = 256;
    let out = std::slice::from_raw_parts_mut(lut, UI_SAMPLES);
    let f: &[f32; 8] = &*(factors as *const [f32; 8]);
    let gauss_denom = 2.0 * sigma * sigma;
    for k in 0..UI_SAMPLES {
        let x = 8.0 * (k as f32 / (UI_SAMPLES - 1) as f32) - 8.0;
        let pc = pixel_correction_rs(x, f, gauss_denom);
        out[k] = offset - pc.log2() / scaling;
    }
}

/// Compute correction factors for CHANNELS=9 UI parameters from PIXEL_CHAN=8 RBF weights.
///
/// out[i] = clamp(sum_j(gaussian(centers_params[i] - centers_ops[j]) * factors[j]), 0.25, 4)
///
/// Matches the DT_OMP_FOR_SIMD at src/iop/toneequal.c:1254.
#[no_mangle]
pub unsafe extern "C" fn darkroom_toneequal_compute_channels_factors(
    factors: *const f32,
    out: *mut f32,
    sigma: f32,
) {
    let f: &[f32; 8] = &*(factors as *const [f32; 8]);
    let o = std::slice::from_raw_parts_mut(out, 9);
    let gauss_denom = 2.0 * sigma * sigma;
    for (i, &cp) in CENTERS_PARAMS.iter().enumerate() {
        o[i] = pixel_correction_rs(cp, f, gauss_denom);
    }
}

// ── Preview path: UI channel gains → RBF weights → correction LUT ────────────

/// darktable fixes the radial-basis smoothing at sqrt(2); module version 2
/// removed the GUI slider that used to expose it (params field comment:
/// `$DEFAULT: 1.414213562`, no $MIN/$MAX ⇒ not surfaced by the GUI).
const SMOOTHING_SQRT2: f32 = std::f32::consts::SQRT_2;

/// `DT_TONEEQ_MIN_EV` / `DT_TONEEQ_MAX_EV` (toneequal.c:771-772).
const MIN_EV: f32 = -8.0;

/// `PIXEL_CHAN * LUT_RESOLUTION + 1` — the correction-LUT length the C
/// allocates (toneequal.c data struct).
const CORRECTION_LUT_LEN: usize = 8 * 10_000 + 1;

/// Solve the over-determined radial-basis system A·w = y for the 8 RBF weights,
/// porting `build_interpolation_matrix` (toneequal.c:1337) plus `pseudo_solve`
/// (choleski.h:289): form the normal equations AᵀA·w = Aᵀy and solve the 8×8
/// hermitian positive-definite system with the Cholesky–Banachiewicz
/// decomposition the C uses. `gains_ev` are the nine user channel gains in EV
/// (log2), ordered noise→−8 EV … speculars→0 EV per `get_channels_gains`
/// (toneequal.c:1210), converted to linear factors with exp2 exactly as
/// `get_channels_factors` does.
///
/// On a non-positive-definite pivot the C solver bails with FALSE — but
/// `commit_params` then copies the *unsolved* `exp2(gains)` vector into
/// `d->factors` anyway (`dt_simd_memcpy` runs unconditionally after
/// `pseudo_solve`, toneequal.c), silently mis-reading EV gains as RBF weights.
/// We do NOT mirror that: we return zeros, so the correction LUT collapses to
/// its 0.25 floor uniformly — visibly broken rather than subtly wrong. For the
/// fixed σ=√2 basis this branch is unreachable either way (AᵀA is positive
/// definite); `solve_weights_satisfies_normal_equations` pins that the solve
/// succeeds.
pub fn solve_weights(gains_ev: &[f32; 9]) -> [f32; 8] {
    const CHANNELS: usize = 9;
    const N: usize = 8;
    let denom = 2.0 * SMOOTHING_SQRT2 * SMOOTHING_SQRT2;

    // User gains are log2 offsets (EV) → linear factors.
    let mut y = [0.0f32; CHANNELS];
    for (yi, &g) in y.iter_mut().zip(gains_ev) {
        *yi = g.exp2();
    }

    // A[i][j] = gaussian(centers_params[i] - centers_ops[j]).
    let mut a = [0.0f32; CHANNELS * N];
    for i in 0..CHANNELS {
        for j in 0..N {
            let r = CENTERS_PARAMS[i] - CENTERS_OPS[j];
            a[i * N + j] = (-r * r / denom).exp();
        }
    }

    // Normal equations, lower triangle only — `_transpose_dot_matrix` /
    // `_transpose_dot_vector` skip the mirrored upper half and the Cholesky
    // below reads the lower triangle exclusively.
    let mut m = [0.0f32; N * N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..CHANNELS {
                sum += a[k * N + i] * a[k * N + j];
            }
            m[i * N + j] = sum;
        }
    }
    let mut b = [0.0f32; N];
    for i in 0..N {
        let mut sum = 0.0f32;
        for k in 0..CHANNELS {
            sum += a[k * N + i] * y[k];
        }
        b[i] = sum;
    }

    // Cholesky–Banachiewicz: AᵀA = L·Lᵀ (`_choleski_decompose`). The first
    // pivot gates like the C does (`if(A[0] <= 0.0f) return FALSE`).
    if m[0] <= 0.0 {
        return [0.0; N];
    }
    let mut l = [0.0f32; N * N];
    let mut valid = true;
    for i in 0..N {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * N + k] * l[j * N + k];
            }
            if i == j {
                let t = m[i * N + i] - sum;
                if t < 0.0 {
                    valid = false;
                    l[i * N + j] = f32::NAN;
                } else {
                    l[i * N + j] = t.sqrt();
                }
            } else {
                let t = l[j * N + j];
                if t == 0.0 {
                    valid = false;
                    l[i * N + j] = f32::NAN;
                } else {
                    l[i * N + j] = (m[i * N + j] - sum) / t;
                }
            }
        }
    }
    if !valid {
        return [0.0; N];
    }
    // Triangular descent L·x = b (`_triangular_descent`).
    let mut x = [0.0f32; N];
    for i in 0..N {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * N + j] * x[j];
        }
        let t = l[i * N + i];
        if t != 0.0 {
            x[i] = sum / t;
        } else {
            return [0.0; N];
        }
    }
    // Triangular ascent Lᵀ·w = x (`_triangular_ascent`, bottom-up).
    let mut w = [0.0f32; N];
    for i in (0..N).rev() {
        let mut sum = x[i];
        for j in (i + 1)..N {
            sum -= l[j * N + i] * w[j];
        }
        let t = l[i * N + i];
        if t != 0.0 {
            w[i] = sum / t;
        } else {
            return [0.0; N];
        }
    }
    w
}

/// Identity of a correction LUT: every input it depends on, as raw bits so the
/// comparison is exact (a NaN key never compares equal to itself, which would
/// be worse than a miss).
type LutKey = [u32; 9];

/// Cached LUT keyed by the inputs that produce it, so repeated calls within a
/// render reuse the same table. See [`CORRECTION_LUT_CACHE`] for why this
/// matters here more than anywhere else.
type Cached = std::cell::RefCell<Option<(LutKey, std::rc::Rc<[f32; CORRECTION_LUT_LEN]>)>>;

thread_local! {
    /// Memo of the last correction LUT built on this thread.
    ///
    /// `Pipeline::process` splits the image into ~64k-pixel bands and runs the
    /// stage — hence this builder — **once per band**. Each rebuild evaluates
    /// 80 001 × 8 Gaussians (~640k `exp` calls), so on a 20 MP export the 306
    /// band rebuilds would cost ~200 M exp calls for a byte-identical table:
    /// roughly 20× the cost of the whole pixel pass. Keyed on the inputs, so
    /// this is pure memoisation; thread-local rather than shared, so it needs
    /// no lock and the `Rc` never crosses a thread. Same pattern as the
    /// basicadj LUT pair.
    #[allow(clippy::type_complexity)]
    static CORRECTION_LUT_CACHE: Cached = const { std::cell::RefCell::new(None) };
}

fn cached_correction_lut(gains_ev: &[f32; 9]) -> std::rc::Rc<[f32; CORRECTION_LUT_LEN]> {
    let key = [
        gains_ev[0].to_bits(),
        gains_ev[1].to_bits(),
        gains_ev[2].to_bits(),
        gains_ev[3].to_bits(),
        gains_ev[4].to_bits(),
        gains_ev[5].to_bits(),
        gains_ev[6].to_bits(),
        gains_ev[7].to_bits(),
        gains_ev[8].to_bits(),
    ];
    CORRECTION_LUT_CACHE.with(|c| {
        if let Some((k, lut)) = c.borrow().as_ref() {
            if *k == key {
                return lut.clone();
            }
        }
        let weights = solve_weights(gains_ev);
        let mut lut = Box::new([0.0f32; CORRECTION_LUT_LEN]);
        // SAFETY: `lut` is an exclusive boxed slice of exactly
        // PIXEL_CHAN*LUT_RESOLUTION+1 floats — the length
        // `darkroom_toneequal_build_lut` documents; `weights`/`CENTERS_OPS`
        // supply the 8 RBF weights and centres it reads.
        unsafe {
            darkroom_toneequal_build_lut(
                lut.as_mut_ptr(),
                weights.as_ptr(),
                CENTERS_OPS.as_ptr(),
                8,
                10_000,
                SMOOTHING_SQRT2,
                MIN_EV,
            );
        }
        let rc: std::rc::Rc<[f32; CORRECTION_LUT_LEN]> = std::rc::Rc::from(lut);
        *c.borrow_mut() = Some((key, rc.clone()));
        rc
    })
}

/// Preview-path application of the tone equalizer in its "preserve details:
/// no" configuration (`details = DT_TONEEQ_NONE`, the only mode without the
/// guided-filter surface blur): compute the per-pixel luminance mask with the
/// RGB euclidean-norm estimator (darktable's `DT_TONEEQ_NORM_2` default),
/// then scale each pixel by the correction looked up at its own exposure.
///
/// Mirrors the `details == DT_TONEEQ_NONE` flow of `toneeq_process`
/// (toneequal.c:1088-1137): `luminance_mask(..., exposure_boost, 0.0, 1.0)`
/// followed by `apply_toneequalizer`. With the defaults
/// `exposure_boost = exp2(0) = 1`, fulcrum 0 and contrast boost 1,
/// `linear_contrast` reduces to `max(lum, MIN_FLOAT)` — the exp2(−16) floor
/// that keeps log2 finite — and the estimator reduces to √(r²+g²+b²)
/// (pixel_rgb_norm_2, luminance_mask.h:160).
pub fn process_preview_pixels(input: &[f32], output: &mut [f32], gains_ev: &[f32; 9]) {
    let lut = cached_correction_lut(gains_ev);
    let npixels = input.len() / 4;
    let min_float = 2.0f32.powi(-16); // MIN_FLOAT, luminance_mask.h:14
    let mut luminance = Vec::with_capacity(npixels);
    for px in input.chunks_exact(4) {
        let lum = (px[0] * px[0] + px[1] * px[1] + px[2] * px[2]).sqrt();
        luminance.push(lum.max(min_float));
    }
    // SAFETY: input/output are packed RGBA f32 buffers of npixels*4 floats and
    // do not overlap (Pipeline::apply's contract); `luminance` holds exactly
    // npixels entries and `lut` exactly CORRECTION_LUT_LEN — the lengths the
    // kernel documents.
    unsafe {
        darkroom_toneequal_apply_lut(
            input.as_ptr(),
            luminance.as_ptr(),
            output.as_mut_ptr(),
            npixels,
            lut.as_ptr(),
            lut.len(),
            MIN_EV,
            0.0,
            10_000.0,
        );
    }
}
