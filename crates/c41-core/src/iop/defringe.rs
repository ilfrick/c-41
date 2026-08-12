use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Defringe;

impl IopProcess for Defringe {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "defringe" }
}

/// Build the per-pixel edge-chroma map and (optionally) sum it for the
/// global-average mode of defringe.
///
/// For every pixel:
///   edge = (in.a - out.a)^2 + (in.b - out.b)^2
///   out.alpha = edge
///   sum    += edge   (only when use_global_average != 0)
///
/// `in_buf` is the original Lab RGBA buffer; `out_buf` is the gaussian-blurred
/// copy of `in_buf` produced by the preceding `dt_gaussian_blur_4c` call —
/// the function reads the a/b channels of both and writes the chroma value
/// into the alpha channel of `out_buf` (overwriting whatever the blur put
/// there, matching the C contract exactly).
///
/// Returns the accumulated chroma sum; the caller normalises by pixel count
/// to obtain `avg_edge_chroma` as in the original C code.
///
/// Matches the DT_OMP_FOR_SIMD loop at line 271 of src/iop/defringe.c.
#[no_mangle]
pub unsafe extern "C" fn darkroom_defringe_edge_chroma_pass(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    use_global_average: i32,
) -> f32 {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let weight = if use_global_average != 0 { 1.0_f32 } else { 0.0_f32 };

    let mut sum = 0.0_f32;
    for k in 0..npixels {
        let j = k * 4;
        let a = input[j + 1] - output[j + 1];
        let b = input[j + 2] - output[j + 2];
        let edge = a * a + b * b;
        output[j + 3] = edge;
        sum += edge * weight;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_zero_when_in_equals_out() {
        // If gauss(in) == in, then a = b = 0 → edge = 0 everywhere.
        let n = 16;
        let input = vec![0.5_f32; n * 4];
        let mut output = input.clone();
        let sum = unsafe {
            darkroom_defringe_edge_chroma_pass(input.as_ptr(), output.as_mut_ptr(), n, 1)
        };
        assert_eq!(sum, 0.0);
        for k in 0..n {
            assert_eq!(output[k * 4 + 3], 0.0);
        }
    }

    #[test]
    fn writes_edge_into_alpha_channel() {
        let n = 1;
        // in = (L=0, a=2, b=3, alpha=0); out = (L=0, a=1, b=2, alpha=0)
        // a_diff = 1, b_diff = 1 → edge = 2
        let input = vec![0.0_f32, 2.0, 3.0, 0.0];
        let mut output = vec![0.0_f32, 1.0, 2.0, 0.0];
        let sum = unsafe {
            darkroom_defringe_edge_chroma_pass(input.as_ptr(), output.as_mut_ptr(), n, 1)
        };
        assert!((sum - 2.0).abs() < 1e-6);
        assert!((output[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ignores_sum_when_global_average_off() {
        let n = 2;
        let input = vec![0.0_f32, 1.0, 0.0, 0.0,
                         0.0,     2.0, 0.0, 0.0];
        let mut output = vec![0.0_f32; n * 4];
        let sum = unsafe {
            darkroom_defringe_edge_chroma_pass(input.as_ptr(), output.as_mut_ptr(), n, 0)
        };
        // Sum should be zero (weighted by 0) even though the per-pixel
        // chroma was non-zero.
        assert_eq!(sum, 0.0);
        // But the per-pixel edge value was still written into alpha.
        assert_eq!(output[3], 1.0); // 1^2 + 0
        assert_eq!(output[7], 4.0); // 2^2 + 0
    }

    #[test]
    fn l_channel_does_not_affect_chroma() {
        // Edge is computed only from a/b channels (indices 1, 2). The L
        // channel (index 0) must be ignored.
        let n = 1;
        let input = vec![100.0_f32, 0.0, 0.0, 0.0];
        let mut output = vec![0.0_f32, 0.0, 0.0, 0.0];
        let sum = unsafe {
            darkroom_defringe_edge_chroma_pass(input.as_ptr(), output.as_mut_ptr(), n, 1)
        };
        assert_eq!(sum, 0.0);
        assert_eq!(output[3], 0.0);
    }
}
