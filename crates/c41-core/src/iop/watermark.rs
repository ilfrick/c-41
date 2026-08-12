use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Watermark;

impl IopProcess for Watermark {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "watermark" }
}

/// Alpha-composite a Cairo BGRA watermark over a float RGBA image.
///
/// Replaces the DT_OMP_FOR loop in watermark.c::process().
/// watermark is Cairo-rendered 8-bit BGRA (byte order: B=0, G=1, R=2, A=3).
/// Output: o[c] = (1 - alpha)*in[c] + opacity*(watermark[c]/255)
/// Alpha channel: o[3] = in[3] (pass-through).
#[no_mangle]
pub unsafe extern "C" fn darkroom_watermark_blend(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    watermark: *const u8,
    opacity: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let wm = std::slice::from_raw_parts(watermark, npixels * 4);
    for j in 0..npixels {
        let alpha = (wm[j * 4 + 3] as f32 / 255.0) * opacity;
        let one_minus = 1.0 - alpha;
        // Cairo BGRA: byte 0=B, 1=G, 2=R, 3=A — maps to RGB as [2,1,0]
        output[j * 4]     = one_minus * input[j * 4]     + opacity * (wm[j * 4 + 2] as f32 / 255.0);
        output[j * 4 + 1] = one_minus * input[j * 4 + 1] + opacity * (wm[j * 4 + 1] as f32 / 255.0);
        output[j * 4 + 2] = one_minus * input[j * 4 + 2] + opacity * (wm[j * 4]     as f32 / 255.0);
        output[j * 4 + 3] = input[j * 4 + 3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fully_transparent_watermark_passes_input_through() {
        let input = vec![0.5f32, 0.25, 0.1, 0.8];
        let mut out = vec![0.0f32; 4];
        // alpha=0 → no watermark contribution
        let wm: Vec<u8> = vec![0, 0, 0, 0]; // BGRA, A=0
        unsafe {
            darkroom_watermark_blend(input.as_ptr(), out.as_mut_ptr(), 1, wm.as_ptr(), 1.0);
        }
        assert!((out[0] - 0.5).abs()  < 1e-6);
        assert!((out[1] - 0.25).abs() < 1e-6);
        assert!((out[2] - 0.1).abs()  < 1e-6);
        assert!((out[3] - 0.8).abs()  < 1e-6);
    }

    #[test]
    fn fully_opaque_white_watermark_produces_opacity() {
        let input = vec![0.0f32, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 4];
        // BGRA: B=255, G=255, R=255, A=255 → white, fully opaque
        let wm: Vec<u8> = vec![255, 255, 255, 255];
        let opacity = 0.5f32;
        unsafe {
            darkroom_watermark_blend(input.as_ptr(), out.as_mut_ptr(), 1, wm.as_ptr(), opacity);
        }
        // alpha = (255/255) * 0.5 = 0.5
        // o[c] = (1 - 0.5)*0.0 + 0.5*(255/255) = 0.5
        assert!((out[0] - 0.5).abs() < 1e-6, "R={}", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-6, "G={}", out[1]);
        assert!((out[2] - 0.5).abs() < 1e-6, "B={}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-6, "A={}", out[3]); // alpha pass-through
    }

    #[test]
    fn alpha_channel_always_passes_through() {
        let input = vec![0.3f32, 0.3, 0.3, 0.7];
        let mut out = vec![0.0f32; 4];
        let wm: Vec<u8> = vec![128, 128, 128, 255];
        unsafe {
            darkroom_watermark_blend(input.as_ptr(), out.as_mut_ptr(), 1, wm.as_ptr(), 1.0);
        }
        assert!((out[3] - 0.7).abs() < 1e-6);
    }
}
