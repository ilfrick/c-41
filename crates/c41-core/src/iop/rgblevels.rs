use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;
use crate::color::rgb_norm;

pub struct RgbLevels;

impl IopProcess for RgbLevels {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "rgblevels" }
}

/// Apply black/white-point + gamma to a single value.
/// Assumes value > min (caller must check).
#[inline(always)]
fn levels_curve(lut: &[f32], min: f32, max: f32, inv_gamma: f32, v: f32) -> f32 {
    let pct = (v - min) / (max - min);
    if v >= max {
        pct.powf(inv_gamma)
    } else {
        lut[((pct * 0x1_0000_u32 as f32) as usize).clamp(0, 0xffff)]
    }
}

/// RGB-levels IOP — per-channel or linked black/white-point + gamma correction.
///
/// mode: 0 = independent channels (autoscale == INDEPENDENT or preserve_colors == NONE)
///       1 = linked via rgb_norm (luma-based)
/// preserve_colors: dt_rgb_norm mode for the linked path (ignored when mode == 0)
/// min_levels / max_levels / inv_gamma: 3 floats each (R, G, B)
/// lut_r/g/b: 65536 floats each (d->lut[0..2])
#[no_mangle]
pub unsafe extern "C" fn darkroom_rgblevels_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mode: i32,
    preserve_colors: i32,
    min_levels: *const f32,
    max_levels: *const f32,
    inv_gamma: *const f32,
    lut_r: *const f32,
    lut_g: *const f32,
    lut_b: *const f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let mins = std::slice::from_raw_parts(min_levels, 3);
    let maxs = std::slice::from_raw_parts(max_levels, 3);
    let inv_g = std::slice::from_raw_parts(inv_gamma, 3);
    let luts = [
        std::slice::from_raw_parts(lut_r, 0x10000),
        std::slice::from_raw_parts(lut_g, 0x10000),
        std::slice::from_raw_parts(lut_b, 0x10000),
    ];

    if mode == 0 {
        for px in 0..npixels {
            let base = px * 4;
            let o = &mut out[base..base+4];
            for c in 0..3 {
                let v = inp[base+c];
                o[c] = if v <= mins[c] { 0.0 } else { levels_curve(luts[c], mins[c], maxs[c], inv_g[c], v) };
            }
            o[3] = inp[base+3];
        }
    } else {
        // Linked: apply channel-0 curve on rgb_norm luma, ratio-scale all channels.
        let min0 = mins[0];
        let max0 = maxs[0];
        let ig0 = inv_g[0];
        for px in 0..npixels {
            let base = px * 4;
            let i = &inp[base..base+4];
            let o = &mut out[base..base+4];
            let lum = rgb_norm(i[0], i[1], i[2], preserve_colors);
            if lum > min0 {
                let curve_lum = levels_curve(luts[0], min0, max0, ig0, lum);
                let ratio = curve_lum / lum;
                o[0] = ratio * i[0];
                o[1] = ratio * i[1];
                o[2] = ratio * i[2];
            } else {
                o[0] = 0.0;
                o[1] = 0.0;
                o[2] = 0.0;
            }
            o[3] = i[3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_lut() -> Vec<f32> {
        (0..0x10000usize).map(|i| i as f32 / 0xffff as f32).collect()
    }

    #[test]
    fn independent_full_range_is_identity() {
        let lut = linear_lut();
        let mins = [0.0f32, 0.0, 0.0];
        let maxs = [1.0f32, 1.0, 1.0];
        let inv_g = [1.0f32, 1.0, 1.0];
        let inp = [0.3f32, 0.5, 0.7, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgblevels_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                0, 0,
                mins.as_ptr(), maxs.as_ptr(), inv_g.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(),
            )
        };
        assert!((out[0] - 0.3).abs() < 1e-3);
        assert!((out[1] - 0.5).abs() < 1e-3);
        assert!((out[2] - 0.7).abs() < 1e-3);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn below_black_point_clips_to_zero() {
        let lut = linear_lut();
        let mins = [0.5f32, 0.5, 0.5];
        let maxs = [1.0f32, 1.0, 1.0];
        let inv_g = [1.0f32, 1.0, 1.0];
        let inp = [0.2f32, 0.2, 0.2, 1.0]; // below black point 0.5
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgblevels_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                0, 0,
                mins.as_ptr(), maxs.as_ptr(), inv_g.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(),
            )
        };
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn linked_grey_is_symmetric() {
        // Grey input with mode=1 (linked) should stay grey (equal channels)
        let lut = linear_lut();
        let mins = [0.0f32, 0.0, 0.0];
        let maxs = [1.0f32, 1.0, 1.0];
        let inv_g = [1.0f32, 1.0, 1.0];
        let inp = [0.5f32, 0.5, 0.5, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgblevels_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1, 0, // linked, NORM_NONE... but mode=1 means we use linked path
                mins.as_ptr(), maxs.as_ptr(), inv_g.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(),
            )
        };
        assert!((out[0] - out[1]).abs() < 1e-5);
        assert!((out[1] - out[2]).abs() < 1e-5);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn alpha_passes_through() {
        let lut = linear_lut();
        let mins = [0.0f32; 3];
        let maxs = [1.0f32; 3];
        let inv_g = [1.0f32; 3];
        let inp = [0.5f32, 0.5, 0.5, 0.42];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgblevels_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                0, 0,
                mins.as_ptr(), maxs.as_ptr(), inv_g.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(),
            )
        };
        assert_eq!(out[3], 0.42);
    }
}
