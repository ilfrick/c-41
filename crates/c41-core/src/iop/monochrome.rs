use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Monochrome;

impl IopProcess for Monochrome {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "monochrome" }
}

/// Gaussian color filter: exp(-clamp(((ai-a)^2 + (bi-b)^2) / dbl_size, 0, 1))
#[inline(always)]
fn color_filter(ai: f32, bi: f32, a: f32, b: f32, dbl_size: f32) -> f32 {
    let v = ((ai - a) * (ai - a) + (bi - b) * (bi - b)) / dbl_size;
    (-v.clamp(0.0, 1.0)).exp()
}

/// Smooth envelope on L for highlight blending.
/// Returns a weight in [0,1] that is 1 at L=60 (beta=0.6) and tapers to 0 at edges.
#[inline(always)]
fn envelope(l: f32) -> f32 {
    let x = (l / 100.0).clamp(0.0, 1.0);
    const BETA: f32 = 0.6;
    if x < BETA {
        let tmp = x / BETA - 1.0;
        1.0 - tmp * tmp
    } else {
        let tmp1 = (1.0 - x) / (1.0 - BETA);
        let tmp2 = tmp1 * tmp1;
        3.0 * tmp2 - 2.0 * tmp2 * tmp1
    }
}

/// First monochrome pass: convert Lab pixel to monochrome using a Gaussian color filter.
///
/// out[k]   = 100 * color_filter(in[k+1], in[k+2], a, b, sigma2)  (L channel)
/// out[k+1] = out[k+2] = 0  (desaturate a/b)
/// Alpha channel is not written.
///
/// sigma2 = 2 * (d->size * 128)^2  pre-computed by caller.
/// a/b are the Lab a*/b* center of the color filter from d->a / d->b.
#[no_mangle]
pub unsafe extern "C" fn darkroom_monochrome_colorfilter(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    a: f32,
    b: f32,
    sigma2: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in (0..npixels * 4).step_by(4) {
        output[k]     = 100.0 * color_filter(input[k + 1], input[k + 2], a, b, sigma2);
        output[k + 1] = 0.0;
        output[k + 2] = 0.0;
    }
}

/// Second monochrome pass: blend bilateral-filtered result with original input.
///
/// out already contains the bilateral-blurred version of the color-filtered L values.
/// For each pixel k:
///   t  = envelope(in[k]) + (1 - envelope(in[k])) * (1 - highlights)
///   out[k] = (1-t)*in[k]  +  t * out[k] * in[k] / 100
/// (out[k+1..3] unchanged by the blur and not touched here)
///
/// highlights = d->highlights (range 0..1 in C, passed directly).
#[no_mangle]
pub unsafe extern "C" fn darkroom_monochrome_blend(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    highlights: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in (0..npixels * 4).step_by(4) {
        let tt = envelope(input[k]);
        let t = tt + (1.0 - tt) * (1.0 - highlights);
        output[k] = (1.0 - t) * input[k] + t * output[k] * (1.0 / 100.0) * input[k];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colorfilter_neutral_grey_at_center() {
        // a=b=0 pixel at color filter center: filter returns exp(0)=1 → L=100
        let input = vec![50.0f32, 0.0, 0.0, 0.5];
        let mut out = vec![0.0f32; 4];
        let sigma2 = 2.0 * (0.5 * 128.0) * (0.5 * 128.0);
        unsafe { darkroom_monochrome_colorfilter(input.as_ptr(), out.as_mut_ptr(), 1, 0.0, 0.0, sigma2); }
        assert!((out[0] - 100.0).abs() < 0.001, "L={}", out[0]);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn colorfilter_far_from_center_near_zero() {
        // large distance from filter center → filter → 0
        let input = vec![50.0f32, 100.0, 100.0, 0.5];
        let mut out = vec![0.0f32; 4];
        let sigma2 = 1.0; // small sigma: exp(-clamp(20000,0,1)) = exp(-1) ≈ 0.37
        unsafe { darkroom_monochrome_colorfilter(input.as_ptr(), out.as_mut_ptr(), 1, 0.0, 0.0, sigma2); }
        // distance^2 = 100^2 + 100^2 = 20000, /1.0 > 1 → clamped to 1 → exp(-1) ≈ 0.368
        assert!((out[0] - 100.0 * (-1.0f32).exp()).abs() < 0.5, "L={}", out[0]);
    }

    #[test]
    fn blend_zero_highlights_passes_through_input() {
        // highlights=0: t = tt + (1-tt)*(1-0) = tt + 1 - tt = 1 always
        // out[k] = 0 * in + 1 * out * in / 100
        let input  = vec![80.0f32, 0.0, 0.0, 0.0];
        let mut out = vec![60.0f32, 0.0, 0.0, 0.0]; // blurred L = 60
        unsafe { darkroom_monochrome_blend(input.as_ptr(), out.as_mut_ptr(), 1, 0.0); }
        let expected = 60.0 * 80.0 / 100.0; // = 48
        assert!((out[0] - expected).abs() < 1e-4, "out={}", out[0]);
    }

    #[test]
    fn blend_full_highlights_blends_with_envelope() {
        // highlights=1: t = envelope(L)
        let input = vec![0.0f32, 0.0, 0.0, 0.0]; // L=0 → envelope(0)=0 → t=0
        let mut out = vec![50.0f32, 0.0, 0.0, 0.0];
        unsafe { darkroom_monochrome_blend(input.as_ptr(), out.as_mut_ptr(), 1, 1.0); }
        // t=0 → out = (1-0)*0 + 0*... = 0
        assert_eq!(out[0], 0.0);
    }
}
