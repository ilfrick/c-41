use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Bloom;

impl IopProcess for Bloom {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "bloom" }
}

/// First bloom pass: threshold-filter input L channel into a packed 1-channel buffer.
/// blur_buf must be npixels floats (1 channel, not 4).
/// scale = 1.0 / exp2f(-1.0 * (min(100, strength+1) / 100))  pre-computed by caller.
#[no_mangle]
pub unsafe extern "C" fn darkroom_bloom_gather(
    in_buf: *const f32,
    blur_buf: *mut f32,
    npixels: usize,
    threshold: f32,
    scale: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let blur  = std::slice::from_raw_parts_mut(blur_buf, npixels);
    for k in 0..npixels {
        let l = input[k * 4] * scale;
        blur[k] = if l > threshold { l } else { 0.0 };
    }
}

/// Second bloom pass: screen-blend blurred lightness into the 4-channel output.
/// blur_buf is the 1-channel result of dt_box_mean on the gather output (npixels floats).
/// Screen blend: L_out = 100 - (100 - L_in) * (100 - L_blur) / 100; a/b/alpha copied.
#[no_mangle]
pub unsafe extern "C" fn darkroom_bloom_blend(
    in_buf: *const f32,
    out_buf: *mut f32,
    blur_buf: *const f32,
    npixels: usize,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let blur   = std::slice::from_raw_parts(blur_buf, npixels);
    for k in 0..npixels {
        output[k * 4]     = 100.0 - ((100.0 - input[k * 4]) * (100.0 - blur[k])) / 100.0;
        output[k * 4 + 1] = input[k * 4 + 1];
        output[k * 4 + 2] = input[k * 4 + 2];
        output[k * 4 + 3] = input[k * 4 + 3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_zeros_below_threshold() {
        let input = vec![50.0f32, 10.0, -5.0, 1.0,
                         80.0f32, 20.0,  5.0, 1.0];
        let mut blur = vec![0.0f32; 2];
        unsafe { darkroom_bloom_gather(input.as_ptr(), blur.as_mut_ptr(), 2, 60.0, 1.0); }
        assert_eq!(blur[0], 0.0);  // 50 < 60 → 0
        assert_eq!(blur[1], 80.0); // 80 > 60 → 80
    }

    #[test]
    fn gather_scale_applied() {
        let input = vec![50.0f32, 0.0, 0.0, 0.0];
        let mut blur = vec![0.0f32; 1];
        unsafe { darkroom_bloom_gather(input.as_ptr(), blur.as_mut_ptr(), 1, 60.0, 2.0); }
        assert_eq!(blur[0], 100.0); // 50*2=100 > 60
    }

    #[test]
    fn blend_screen_formula() {
        // Screen: 100 - (100-80)*(100-40)/100 = 100 - 20*60/100 = 100 - 12 = 88
        let input  = vec![80.0f32, 5.0, -3.0, 0.5];
        let blur   = vec![40.0f32];
        let mut output = vec![0.0f32; 4];
        unsafe { darkroom_bloom_blend(input.as_ptr(), output.as_mut_ptr(), blur.as_ptr(), 1); }
        assert!((output[0] - 88.0).abs() < 1e-4, "L={}", output[0]);
        assert_eq!(output[1],  5.0);
        assert_eq!(output[2], -3.0);
        assert_eq!(output[3],  0.5);
    }

    #[test]
    fn blend_zero_blur_is_passthrough() {
        let input  = vec![70.0f32, 1.0, 2.0, 0.8];
        let blur   = vec![0.0f32];
        let mut output = vec![0.0f32; 4];
        unsafe { darkroom_bloom_blend(input.as_ptr(), output.as_mut_ptr(), blur.as_ptr(), 1); }
        assert!((output[0] - 70.0).abs() < 1e-4);
    }
}
