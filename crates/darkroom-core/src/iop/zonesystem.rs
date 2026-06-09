use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Zonesystem;

impl IopProcess for Zonesystem {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "zonesystem" }
}

#[no_mangle]
pub unsafe extern "C" fn darkroom_zonesystem_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    rzscale: f32,
    zonemap_offset: *const f32, // [size] floats
    zonemap_scale: *const f32,  // [size] floats
    size: usize,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let offsets = std::slice::from_raw_parts(zonemap_offset, size);
    let scales = std::slice::from_raw_parts(zonemap_scale, size);

    let max_rz = (size as i32) - 2;

    for k in (0..npixels * 4).step_by(4) {
        let luma = input[k];
        let rz = (luma * rzscale) as i32;
        let rz = rz.clamp(0, max_rz) as usize;

        let zs = if rz > 0 && luma != 0.0 {
            offsets[rz] / luma
        } else {
            0.0
        } + scales[rz];

        output[k]     = input[k]     * zs;
        output[k + 1] = input[k + 1] * zs;
        output[k + 2] = input[k + 2] * zs;
        output[k + 3] = input[k + 3] * zs;
    }
}

/// Extract one interleaved channel (channel 0 / luma) from an `npixels * ch`
/// buffer into a contiguous single-channel buffer. Ports the strided copy
/// loops feeding the GUI zone-map preview in zonesystem.c::process_common_cleanup.
///
/// # Safety
/// `in_buf` must hold `npixels * ch` floats, `out_buf` `npixels` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_zonesystem_extract_channel(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    ch: usize,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * ch);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels);
    for k in 0..npixels {
        output[k] = input[ch * k];
    }
}

/// Quantise a blurred luma buffer (0..100) into zone indices clamped to
/// `[0, size - 2]`. Ports the CLAMPS fill loops that build the GUI preview
/// zone-map (guchar) in zonesystem.c. Matches the C `(guchar)` truncation.
///
/// # Safety
/// `blurred` must hold `npixels` floats, `zonemap` `npixels` bytes; `size >= 2`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_zonesystem_build_zonemap(
    blurred: *const f32,
    zonemap: *mut u8,
    npixels: usize,
    size: usize,
) {
    let input = std::slice::from_raw_parts(blurred, npixels);
    let output = std::slice::from_raw_parts_mut(zonemap, npixels);
    let sm1 = (size as i32 - 1) as f32;
    let sm2 = (size as i32 - 2) as f32;
    for k in 0..npixels {
        // CLAMPS(tmp * (size-1)/100, 0, size-2); float -> guchar truncates.
        let v = (input[k] * sm1 / 100.0).max(0.0).min(sm2);
        output[k] = v as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_channel_pulls_luma_with_stride() {
        // 3 RGBA pixels; channel 0 = 1.0, 2.0, 3.0
        let input = [1.0, 9.0, 9.0, 9.0, 2.0, 9.0, 9.0, 9.0, 3.0, 9.0, 9.0, 9.0];
        let mut out = [0f32; 3];
        unsafe { darkroom_zonesystem_extract_channel(input.as_ptr(), out.as_mut_ptr(), 3, 4); }
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn build_zonemap_quantises_and_clamps() {
        // size = 10 -> sm1 = 9, sm2 = 8. v = luma * 9 / 100, clamped to [0, 8].
        let input = [0.0_f32, 50.0, 100.0, 200.0, -5.0];
        let mut out = [0u8; 5];
        unsafe { darkroom_zonesystem_build_zonemap(input.as_ptr(), out.as_mut_ptr(), 5, 10); }
        // 0 -> 0; 50*0.09=4.5 -> 4; 100*0.09=9 -> clamp 8; 200 -> clamp 8; -5 -> 0
        assert_eq!(out, [0, 4, 8, 8, 0]);
    }
}
