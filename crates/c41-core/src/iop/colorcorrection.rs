//! Color-correction IOP — luminance-dependent Lab a/b scaling with saturation.
//!
//! Replaces the OMP loop in src/iop/colorcorrection.c::process().
//!
//! Per-pixel algorithm (Lab input, 4 channels):
//!   out.L = in.L
//!   out.a = saturation * (in.a + in.L * a_scale + a_base)
//!   out.b = saturation * (in.b + in.L * b_scale + b_base)
//!   out.α = in.α

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct ColorCorrection;

impl IopProcess for ColorCorrection {
    fn name(&self) -> &'static str {
        "colorcorrection"
    }

    fn process(
        &self,
        input: &[f32],
        output: &mut [f32],
        params: &IopParams,
        _roi: &RoiIn,
    ) -> Result<()> {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct ColorCorrectionData {
            a_scale: f32,
            a_base: f32,
            b_scale: f32,
            b_base: f32,
            saturation: f32,
        }
        let d = unsafe {
            params
                .cast::<ColorCorrectionData>()
                .ok_or_else(|| Error::Pipeline("colorcorrection: params too short".into()))?
        };
        process_pixels(
            input,
            output,
            d.a_scale,
            d.a_base,
            d.b_scale,
            d.b_base,
            d.saturation,
        );
        Ok(())
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl(
            "colorcorrection: OpenCL path not yet ported".into(),
        ))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    a_scale: f32,
    a_base: f32,
    b_scale: f32,
    b_base: f32,
    saturation: f32,
) {
    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let l = chunk_in[0];
        chunk_out[0] = l;
        chunk_out[1] = saturation * (chunk_in[1] + l * a_scale + a_base);
        chunk_out[2] = saturation * (chunk_in[2] + l * b_scale + b_base);
        chunk_out[3] = chunk_in[3];
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/colorcorrection.c in place of the OMP loop.
///
/// # Safety
/// `in_buf` and `out_buf` must be non-overlapping arrays of `npixels * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorcorrection_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    a_scale: f32,
    a_base: f32,
    b_scale: f32,
    b_base: f32,
    saturation: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    process_pixels(input, output, a_scale, a_base, b_scale, b_base, saturation);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_scale0_base0_sat1() {
        // a_scale=0, a_base=0 → out.a = 1 * (in.a + 0 + 0) = in.a
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 16];
        process_pixels(&input, &mut output, 0.0, 0.0, 0.0, 0.0, 1.0);
        for (i, o) in input.iter().zip(output.iter()) {
            assert!((i - o).abs() < 1e-6, "identity failed: {i} vs {o}");
        }
    }

    #[test]
    fn saturation_zero_zeroes_ab() {
        let input = vec![50.0f32, 20.0, -10.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 1.0, 5.0, 2.0, -3.0, 0.0);
        assert!((output[0] - 50.0).abs() < 1e-6);
        assert!(output[1].abs() < 1e-6, "a should be 0 at sat=0: {}", output[1]);
        assert!(output[2].abs() < 1e-6, "b should be 0 at sat=0: {}", output[2]);
        assert!((output[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l_dependent_ab_shift() {
        // L=100, a_scale=0.5, a_base=10, sat=1 → out.a = in.a + 100*0.5 + 10 = in.a + 60
        let input = vec![100.0f32, 5.0, -5.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.5, 10.0, 0.5, 10.0, 1.0);
        // out.a = 5 + 100*0.5 + 10 = 65; out.b = -5 + 100*0.5 + 10 = 55
        assert!((output[1] - 65.0).abs() < 1e-5, "a: expected 65, got {}", output[1]);
        assert!((output[2] - 55.0).abs() < 1e-5, "b: expected 55, got {}", output[2]);
    }

    #[test]
    fn l_and_alpha_pass_through() {
        let input = vec![75.0f32, 10.0, -10.0, 0.5];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 2.0, 3.0, 0.5, -5.0, 2.0);
        assert!((output[0] - 75.0).abs() < 1e-6, "L must pass through");
        assert!((output[3] - 0.5).abs() < 1e-6, "alpha must pass through");
    }
}
