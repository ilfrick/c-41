//! Color-contrast IOP — affine transform on Lab a/b channels.
//!
//! Replaces the inner loops of src/iop/colorcontrast.c::process().
//! Two modes:
//!   unbound=true  → out[c] = in[c] * slope[c] + offset[c]  (no clamping)
//!   unbound=false → same, but a/b clamped to [-128, 128]
//!
//! slope  = [1, a_steepness, b_steepness, 1]
//! offset = [0, a_offset,    b_offset,    0]

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

// ── Data types ────────────────────────────────────────────────────────────────

/// Computed runtime params (mirrors dt_iop_colorcontrast_params_t, used directly).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ColorContrastData {
    pub a_steepness: f32,
    pub a_offset: f32,
    pub b_steepness: f32,
    pub b_offset: f32,
    pub unbound: i32, // non-zero = no clamping
}

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct ColorContrast;

impl IopProcess for ColorContrast {
    fn name(&self) -> &'static str {
        "colorcontrast"
    }

    fn process(
        &self,
        input: &[f32],
        output: &mut [f32],
        params: &IopParams,
        _roi: &RoiIn,
    ) -> Result<()> {
        let d = unsafe {
            params
                .cast::<ColorContrastData>()
                .ok_or_else(|| Error::Pipeline("colorcontrast: params too short".into()))?
        };
        process_pixels(
            input,
            output,
            d.a_steepness,
            d.a_offset,
            d.b_steepness,
            d.b_offset,
            d.unbound != 0,
        );
        Ok(())
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl(
            "colorcontrast: OpenCL path not yet ported".into(),
        ))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// Lab-channel affine scale/offset with optional clamping.
///
/// Pixels are 4-channel (L, a, b, alpha). L and alpha pass through unchanged.
/// a is scaled by `a_steepness` and shifted by `a_offset`;
/// b is scaled by `b_steepness` and shifted by `b_offset`.
#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    a_steepness: f32,
    a_offset: f32,
    b_steepness: f32,
    b_offset: f32,
    unbound: bool,
) {
    let slope = [1.0f32, a_steepness, b_steepness, 1.0f32];
    let offset = [0.0f32, a_offset, b_offset, 0.0f32];

    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        if unbound {
            for c in 0..4 {
                chunk_out[c] = chunk_in[c] * slope[c] + offset[c];
            }
        } else {
            chunk_out[0] = chunk_in[0]; // L: passthrough
            chunk_out[1] = (chunk_in[1] * slope[1] + offset[1]).clamp(-128.0, 128.0);
            chunk_out[2] = (chunk_in[2] * slope[2] + offset[2]).clamp(-128.0, 128.0);
            chunk_out[3] = chunk_in[3]; // alpha: passthrough
        }
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/colorcontrast.c in place of the two OMP loops.
///
/// # Safety
/// `in_buf` and `out_buf` must be non-overlapping arrays of `npixels * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorcontrast_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    a_steepness: f32,
    a_offset: f32,
    b_steepness: f32,
    b_offset: f32,
    unbound: i32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    process_pixels(
        input,
        output,
        a_steepness,
        a_offset,
        b_steepness,
        b_offset,
        unbound != 0,
    );
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn unbound_identity_when_slope1_offset0() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 16];
        process_pixels(&input, &mut output, 1.0, 0.0, 1.0, 0.0, true);
        assert!(max_delta(&input, &output) < 1e-7);
    }

    #[test]
    fn unbound_scales_all_channels() {
        let input = vec![50.0f32, 10.0, -20.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 2.0, 5.0, 3.0, -10.0, true);
        // L/alpha unchanged (slope=1, offset=0), a: 10*2+5=25, b: -20*3-10=-70
        assert!((output[0] - 50.0).abs() < 1e-6);
        assert!((output[1] - 25.0).abs() < 1e-6);
        assert!((output[2] - (-70.0)).abs() < 1e-6);
        assert!((output[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bounded_clamps_ab_but_not_luma() {
        // a/b pushed far out of range; L/alpha should pass through
        let input = vec![150.0f32, 200.0, -200.0, 0.5];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 1.0, 0.0, 1.0, 0.0, false);
        assert!((output[0] - 150.0).abs() < 1e-6); // L unchanged
        assert_eq!(output[1], 128.0); // a clamped to 128
        assert_eq!(output[2], -128.0); // b clamped to -128
        assert!((output[3] - 0.5).abs() < 1e-6); // alpha unchanged
    }
}
