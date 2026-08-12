use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Highpass;

impl IopProcess for Highpass {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "highpass" }
}

/// First highpass pass: invert and pack the L channel into a 1-channel buffer.
///
/// out_buf is treated as a packed 1-channel buffer (npixels floats).
/// out[k] = 100 - clamp(in[4*k], 0, 100)
/// The caller then blurs out_buf with dt_box_mean (1 channel).
#[no_mangle]
pub unsafe extern "C" fn darkroom_highpass_invert(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    let input  = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels);
    for k in 0..npixels {
        output[k] = 100.0 - input[k * 4].clamp(0.0, 100.0);
    }
}

/// Second highpass pass: blend packed 1-channel blurred L with original 4-channel input.
///
/// After dt_box_mean, out_buf contains packed 1-channel data at out[0..npixels).
/// This function reads out[k] (packed blurred L) and writes out + 4*k (4-channel pixel)
/// going in REVERSE ORDER (k from npixels-1 down to 0) so that writes never clobber
/// packed values still needed in future iterations.
///
/// pixel_L = clamp((out[k] + in[4*k] - 100) * contrast_scale + 50, 0, 100)
/// Written pixel: {pixel_L, 0, 0, 0} (desaturated Lab).
///
/// contrast_scale = ((data->contrast / 100) * 7.5) * 0.5  pre-computed by caller.
/// This single call replaces the two _blend() OMP calls plus the final sequential loop in C.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highpass_blend(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    contrast_scale: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    // out_buf is first used as 1-ch packed (npixels floats), then expanded to 4-ch (4*npixels)
    let out_packed = std::slice::from_raw_parts(out_buf, npixels);
    // read all packed blurred values before any writes touch the overlap region
    let blurred: Vec<f32> = out_packed.to_vec();

    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in (0..npixels).rev() {
        let l = (blurred[k] + input[k * 4] - 100.0) * contrast_scale + 50.0;
        let l = l.clamp(0.0, 100.0);
        output[k * 4]     = l;
        output[k * 4 + 1] = 0.0;
        output[k * 4 + 2] = 0.0;
        output[k * 4 + 3] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_clamps_and_inverts() {
        let input = vec![
            80.0f32, 5.0, -3.0, 1.0,   // L=80 → 20
            0.0f32,  0.0,  0.0, 0.0,   // L=0  → 100
            110.0f32, 0.0, 0.0, 0.0,   // L=110 → clamp to 100 → 0
            -5.0f32,  0.0, 0.0, 0.0,   // L=-5  → clamp to 0 → 100
        ];
        let mut out = vec![0.0f32; 4];
        unsafe { darkroom_highpass_invert(input.as_ptr(), out.as_mut_ptr(), 4); }
        assert!((out[0] - 20.0).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1] - 100.0).abs() < 1e-4);
        assert!((out[2] - 0.0).abs() < 1e-4);
        assert!((out[3] - 100.0).abs() < 1e-4);
    }

    #[test]
    fn blend_desaturates_output() {
        let n = 2usize;
        let input  = vec![50.0f32, 10.0, -5.0, 0.5,  // pixel 0
                          70.0f32, 20.0,  3.0, 0.8]; // pixel 1
        // out initially contains packed blurred L: [40.0, 60.0] at indices [0,1]
        // after blend, out is 4*n = 8 floats
        let mut out = vec![0.0f32; n * 4];
        out[0] = 40.0; // packed blurred L for pixel 0
        out[1] = 60.0; // packed blurred L for pixel 1

        unsafe { darkroom_highpass_blend(input.as_ptr(), out.as_mut_ptr(), n, 1.0); }

        // pixel 0: L = (40 + 50 - 100)*1 + 50 = -10 + 50 = 40 → clamp 40
        assert!((out[0] - 40.0).abs() < 1e-4, "pixel0 L={}", out[0]);
        assert_eq!(out[1], 0.0); // desaturated a
        assert_eq!(out[2], 0.0); // desaturated b

        // pixel 1: L = (60 + 70 - 100)*1 + 50 = 30 + 50 = 80
        assert!((out[4] - 80.0).abs() < 1e-4, "pixel1 L={}", out[4]);
        assert_eq!(out[5], 0.0);
    }
}
