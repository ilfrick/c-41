use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Overlay;

impl IopProcess for Overlay {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "overlay" }
}

/// Alpha-blend a Cairo ARGB32 overlay (byte order [B, G, R, A]) onto an RGBA f32 image.
#[no_mangle]
pub unsafe extern "C" fn darkroom_overlay_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    image: *const u8,   // Cairo ARGB32, stride bytes per row
    stride: usize,
    opacity: f32,
) {
    let npixels = width * height;
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let img    = std::slice::from_raw_parts(image, height * stride);

    for y in 0..height {
        for x in 0..width {
            let pix = (y * width + x) * 4;
            let src = y * stride + x * 4;

            // Cairo ARGB32 little-endian: [B, G, R, A]
            let alpha = (img[src + 3] as f32 / 255.0) * opacity;
            let inv   = 1.0 - alpha;

            output[pix]     = inv * input[pix]     + opacity * img[src + 2] as f32 / 255.0;
            output[pix + 1] = inv * input[pix + 1] + opacity * img[src + 1] as f32 / 255.0;
            output[pix + 2] = inv * input[pix + 2] + opacity * img[src]     as f32 / 255.0;
            output[pix + 3] = input[pix + 3];
        }
    }
}
