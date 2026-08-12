//! Colorize IOP — replace a/b channels with a fixed Lab color, blend L from input.
//!
//! Replaces the OMP loop in src/iop/colorize.c::process().
//!
//! Per-pixel algorithm (Lab input, 4 channels):
//!   lmlmix = color_L - (mix * 100) / 2
//!   out.L  = lmlmix + in.L * mix
//!   out.a  = color_a
//!   out.b  = color_b
//!   out.α  = 0   (matches C: copy_pixel(out, {0,a,b,0}) zeroes alpha)

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct Colorize;

impl IopProcess for Colorize {
    fn name(&self) -> &'static str {
        "colorize"
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
        struct ColorizeData {
            l: f32,
            a: f32,
            b: f32,
            mix: f32,
        }
        let d = unsafe {
            params
                .cast::<ColorizeData>()
                .ok_or_else(|| Error::Pipeline("colorize: params too short".into()))?
        };
        process_pixels(input, output, d.l, d.a, d.b, d.mix);
        Ok(())
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("colorize: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    color_l: f32,
    color_a: f32,
    color_b: f32,
    mix: f32,
) {
    let lmlmix = color_l - (mix * 100.0) / 2.0;
    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        chunk_out[0] = lmlmix + chunk_in[0] * mix;
        chunk_out[1] = color_a;
        chunk_out[2] = color_b;
        chunk_out[3] = 0.0; // C writes {0,a,b,0} via copy_pixel — alpha zeroed
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/colorize.c in place of the OMP loop.
///
/// # Safety
/// `in_buf` and `out_buf` must be non-overlapping arrays of `npixels * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorize_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    color_l: f32,
    color_a: f32,
    color_b: f32,
    mix: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    process_pixels(input, output, color_l, color_a, color_b, mix);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ab_replaced_by_params() {
        let input = vec![50.0f32, 20.0, -10.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 60.0, 5.0, -15.0, 1.0);
        assert!((output[1] - 5.0).abs() < 1e-6);
        assert!((output[2] - (-15.0)).abs() < 1e-6);
    }

    #[test]
    fn alpha_always_zero() {
        let input = vec![50.0f32, 20.0, -10.0, 0.75];
        let mut output = vec![1.0f32; 4];
        process_pixels(&input, &mut output, 60.0, 5.0, -15.0, 1.0);
        assert!(output[3].abs() < 1e-7, "alpha must always be 0: {}", output[3]);
    }

    #[test]
    fn l_blend_at_mix_zero() {
        // mix=0 → lmlmix=color_l, out.L = color_l (input L ignored)
        let input = vec![30.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 60.0, 0.0, 0.0, 0.0);
        assert!((output[0] - 60.0).abs() < 1e-5, "mix=0: out.L should equal color_l: {}", output[0]);
    }

    #[test]
    fn l_blend_at_mix_one() {
        // mix=1 → lmlmix=color_l-50, out.L = (color_l-50) + in.L
        let input = vec![70.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 60.0, 0.0, 0.0, 1.0);
        let expected = 60.0_f32 - 50.0 + 70.0; // = 80
        assert!((output[0] - expected).abs() < 1e-5, "mix=1: expected {expected}, got {}", output[0]);
    }

    #[test]
    fn multiple_pixels_consistent() {
        let input = vec![40.0f32, 5.0, -5.0, 1.0, 80.0, -10.0, 10.0, 1.0];
        let mut output = vec![0.0f32; 8];
        process_pixels(&input, &mut output, 50.0, 3.0, -3.0, 0.5);
        // lmlmix = 50 - 25 = 25; out.L[0] = 25 + 40*0.5 = 45; out.L[1] = 25 + 80*0.5 = 65
        assert!((output[0] - 45.0).abs() < 1e-5);
        assert!((output[4] - 65.0).abs() < 1e-5);
        assert!((output[1] - 3.0).abs() < 1e-6);
        assert!((output[5] - 3.0).abs() < 1e-6);
    }
}
