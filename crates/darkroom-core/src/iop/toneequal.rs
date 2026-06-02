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
