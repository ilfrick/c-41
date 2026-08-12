//! Relight IOP — center-weighted L-channel boost in Lab space.
//!
//! Replaces the OMP loop in src/iop/relight.c::process().
//!
//! Per-pixel algorithm (Lab input, 4 channels):
//!   lightness = L / 100
//!   x         = -1 + lightness * 2
//!   gauss     = exp(-(x-b)² / c²)       where b = -1+center*2, c = width/10/2
//!   relight   = exp2(ev * clamp(gauss, 0, 1))
//!   out.L     = 100 * clamp(lightness * relight, 0, 1)
//!   out.a/b/α = in.a/b/α

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct Relight;

impl IopProcess for Relight {
    fn name(&self) -> &'static str {
        "relight"
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
        struct RelightData {
            ev: f32,
            center: f32,
            width: f32,
        }
        let d = unsafe {
            params
                .cast::<RelightData>()
                .ok_or_else(|| Error::Pipeline("relight: params too short".into()))?
        };
        process_pixels(input, output, d.ev, d.center, d.width);
        Ok(())
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("relight: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// `GAUSS(a=1, b, c, x) = expf(-(x-b)²/c²)` — matches relight.c macro exactly (no 2× in denominator).
#[inline(always)]
fn gauss(b: f32, c2: f32, x: f32) -> f32 {
    (-(x - b) * (x - b) / c2).exp()
}

#[inline]
pub fn process_pixels(input: &[f32], output: &mut [f32], ev: f32, center: f32, width: f32) {
    let b = -1.0_f32 + center * 2.0;
    let c = (width / 10.0) / 2.0;
    let c2 = c * c;

    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let lightness = chunk_in[0] / 100.0;
        let x = -1.0 + lightness * 2.0;
        let g = gauss(b, c2, x).clamp(0.0, 1.0);
        let relight = (ev * g).exp2();
        chunk_out[0] = 100.0 * (lightness * relight).clamp(0.0, 1.0);
        chunk_out[1] = chunk_in[1];
        chunk_out[2] = chunk_in[2];
        chunk_out[3] = chunk_in[3];
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/relight.c in place of the OMP loop.
///
/// # Safety
/// `in_buf` and `out_buf` must be non-overlapping arrays of `npixels * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_relight_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    ev: f32,
    center: f32,
    width: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    process_pixels(input, output, ev, center, width);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_ev_passthrough_l() {
        // ev=0 → relight=exp2(0)=1 → out.L = 100*clip(L/100) = L for L in [0,100]
        let input = vec![50.0f32, 10.0, -5.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0, 0.5, 1.0);
        assert!((output[0] - 50.0).abs() < 0.01, "L passthrough: {}", output[0]);
        assert!((output[1] - 10.0).abs() < 1e-6);
        assert!((output[2] - (-5.0)).abs() < 1e-6);
        assert!((output[3] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn ab_alpha_always_pass_through() {
        let input = vec![70.0f32, 30.0, -20.0, 0.8];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 2.0, 0.3, 0.5);
        assert!((output[1] - 30.0).abs() < 1e-6);
        assert!((output[2] - (-20.0)).abs() < 1e-6);
        assert!((output[3] - 0.8).abs() < 1e-7);
    }

    #[test]
    fn l_clamped_to_100() {
        // High ev at center L → clip to 100
        let input = vec![100.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 10.0, 1.0, 1.0);
        assert!(output[0] <= 100.0 + 1e-5, "L must not exceed 100: {}", output[0]);
        assert!(output[0] >= 0.0);
    }

    #[test]
    fn l_clamped_to_zero() {
        // Negative ev, high gauss response → lightness*relight → 0, clips to 0
        let input = vec![50.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, -100.0, 0.5, 10.0);
        assert!(output[0] >= 0.0, "L must not be negative: {}", output[0]);
    }
}
