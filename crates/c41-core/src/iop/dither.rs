use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Dither;

impl IopProcess for Dither {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "dither" }
}

/// Posterize path of the dither IOP.
///
/// Replaces the DT_OMP_FOR loop in _process_posterize() in src/iop/dither.c.
/// f = levels - 1, rf = 1.0 / f.
/// _quantize(x, f, rf) = rf * ceil(x*f - 0.5) — rounds up only if frac > 0.5.
/// All 4 channels (including alpha) are quantized identically.
#[no_mangle]
pub unsafe extern "C" fn darkroom_dither_posterize(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    f: f32,
    rf: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in 0..npixels {
        for c in 0..4 {
            output[k * 4 + c] = rf * (input[k * 4 + c] * f - 0.5).ceil();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterize_two_levels_maps_to_zero_or_one() {
        // f=1, rf=1: quantize(x) = ceil(x - 0.5) → 0 for x<=0.5, 1 for x>0.5
        let input = vec![0.0f32, 0.4, 0.5, 0.6,
                         1.0f32, 0.5, 0.0, 0.0];
        let mut out = vec![0.0f32; 8];
        unsafe { darkroom_dither_posterize(input.as_ptr(), out.as_mut_ptr(), 2, 1.0, 1.0); }
        assert_eq!(out[0], 0.0); // ceil(-0.5) = 0
        assert_eq!(out[1], 0.0); // ceil(-0.1) = 0
        assert_eq!(out[2], 0.0); // ceil(0.0)  = 0  (rounds up ONLY if frac > 0.5)
        assert_eq!(out[4], 1.0); // ceil(0.5)  = 1
    }

    #[test]
    fn posterize_four_levels() {
        // f=3, rf=1/3: quantize(0.5) = (1/3)*ceil(0.5*3 - 0.5) = (1/3)*ceil(1.0) = 1/3
        let input = vec![0.5f32, 0.5, 0.5, 0.5];
        let mut out = vec![0.0f32; 4];
        let f = 3.0f32;
        let rf = 1.0 / f;
        unsafe { darkroom_dither_posterize(input.as_ptr(), out.as_mut_ptr(), 1, f, rf); }
        let expected = rf * (0.5 * f - 0.5f32).ceil();
        assert!((out[0] - expected).abs() < 1e-6);
    }
}
