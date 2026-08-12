//! Split-toning IOP — shadow/highlight colorization via HSL.
//!
//! Replaces the OMP loop in src/iop/splittoning.c::process().
//!
//! Per-pixel algorithm (RGB input, 4 channels):
//!   rgb2hsl(pixel → h, s, l)
//!   if l < balance - compress:
//!     hsl2rgb(shadow_hue, shadow_sat, l → mix)
//!     ra = CLIP((balance - compress - l) * 2)
//!     out[c] = CLIP(in[c] * (1-ra) + mix[c] * ra)  for c in 0..4
//!   elif l > balance + compress:
//!     hsl2rgb(highlight_hue, highlight_sat, l → mix)
//!     ra = CLIP((l - (balance+compress)) * 2)
//!     out[c] = CLIP(in[c] * (1-ra) + mix[c] * ra)  for c in 0..4
//!   else: passthrough
//!
//! compress must be pre-scaled by the C caller: (data->compress / 110.0f) / 2.0f

use crate::{
    color::{hsl2rgb, rgb2hsl},
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

pub struct SplitToning;

impl IopProcess for SplitToning {
    fn name(&self) -> &'static str {
        "splittoning"
    }
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(Error::Pipeline("splittoning: use the C FFI entry point".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("splittoning: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    shadow_hue: f32,
    shadow_saturation: f32,
    highlight_hue: f32,
    highlight_saturation: f32,
    balance: f32,
    compress: f32, // pre-scaled: (data->compress / 110.0) / 2.0
) {
    for (ci, co) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let (_, _, l) = rgb2hsl(ci[0], ci[1], ci[2]);

        if l < balance - compress {
            let (mr, mg, mb, ma) = hsl2rgb(shadow_hue, shadow_saturation, l);
            let ra = ((balance - compress - l) * 2.0).clamp(0.0, 1.0);
            let la = 1.0 - ra;
            co[0] = (ci[0] * la + mr * ra).clamp(0.0, 1.0);
            co[1] = (ci[1] * la + mg * ra).clamp(0.0, 1.0);
            co[2] = (ci[2] * la + mb * ra).clamp(0.0, 1.0);
            co[3] = (ci[3] * la + ma * ra).clamp(0.0, 1.0);
        } else if l > balance + compress {
            let (mr, mg, mb, ma) = hsl2rgb(highlight_hue, highlight_saturation, l);
            let ra = ((l - (balance + compress)) * 2.0).clamp(0.0, 1.0);
            let la = 1.0 - ra;
            co[0] = (ci[0] * la + mr * ra).clamp(0.0, 1.0);
            co[1] = (ci[1] * la + mg * ra).clamp(0.0, 1.0);
            co[2] = (ci[2] * la + mb * ra).clamp(0.0, 1.0);
            co[3] = (ci[3] * la + ma * ra).clamp(0.0, 1.0);
        } else {
            co.copy_from_slice(ci);
        }
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// # Safety
/// All pointer arguments must be valid for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn darkroom_splittoning_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    shadow_hue: f32,
    shadow_saturation: f32,
    highlight_hue: f32,
    highlight_saturation: f32,
    balance: f32,
    compress: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    process_pixels(
        input, output,
        shadow_hue, shadow_saturation,
        highlight_hue, highlight_saturation,
        balance, compress,
    );
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_in_midtone_range() {
        // balance=0.5, compress=0.1 → range [0.4, 0.6]; gray pixel l≈0.5 passes through
        let input = vec![0.5f32, 0.5, 0.5, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0, 1.0, 0.2, 1.0, 0.5, 0.1);
        for i in 0..4 {
            assert!((output[i] - input[i]).abs() < 1e-5, "passthrough ch{i}: {}", output[i]);
        }
    }

    #[test]
    fn shadow_toning_applied_to_dark_pixel() {
        // dark pixel l=0.1, balance=0.5, compress=0.1 → shadow toning
        let input = vec![0.1f32, 0.1, 0.1, 1.0];
        let mut output = vec![0.0f32; 4];
        // shadow_hue=0 (red), saturation=1.0 → mixrgb will be red-ish
        process_pixels(&input, &mut output, 0.0, 1.0, 0.5, 1.0, 0.5, 0.1);
        // ra = clip((0.5-0.1-0.1)*2) = clip(0.6) = 0.6 → strong shadow toning
        assert!(output[0] > output[2], "red channel should dominate for hue=0");
    }

    #[test]
    fn output_clamped_to_0_1() {
        let input = vec![0.9f32, 0.9, 0.9, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.1, 1.0, 0.1, 1.0, 0.5, 0.1);
        for &v in &output {
            assert!(v >= 0.0 && v <= 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn alpha_blended_with_mix_alpha_zero() {
        // In shadow zone, mix alpha = 0, so out[3] = in[3] * la
        let input = vec![0.1f32, 0.1, 0.1, 0.8];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0, 0.5, 0.5, 0.5, 0.5, 0.1);
        // ra = clip((0.5-0.1-0.1)*2) = 0.6, la=0.4 → out[3]=0.8*0.4+0*0.6=0.32
        assert!((output[3] - 0.32).abs() < 0.01, "alpha: {}", output[3]);
    }
}
