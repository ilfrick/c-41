//! Negadoctor IOP — film negative scan inversion.
//!
//! Replaces the OMP loop in src/iop/negadoctor.c::process().
//!
//! Per-pixel algorithm (RGB, 4 channels), per channel c:
//!   clamped  = max(in[c], THRESHOLD)           // −32 EV floor
//!   density  = Dmin[c] / clamped
//!   log_dens = −log10(density) = −log2(density) * LOG2_TO_LOG10
//!   corrected = wb_high[c] * log_dens + offset[c]
//!   ten_to_x = 10^corrected
//!   print_linear = max(−(exposure * ten_to_x + black), 0)
//!   print_gamma  = print_linear ^ gamma
//!   // highlight soft-clip (OpenEXR formula):
//!   if print_gamma > soft_clip:
//!     out[c] = soft_clip + (1 − exp(−(print_gamma−soft_clip)/soft_clip_comp)) * soft_clip_comp
//!   else:
//!     out[c] = print_gamma

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

const THRESHOLD: f32 = 2.3283064365386963e-10; // 2^−32
const LOG2_TO_LOG10: f32 = std::f32::consts::LOG10_2;

pub struct Negadoctor;

impl IopProcess for Negadoctor {
    fn name(&self) -> &'static str {
        "negadoctor"
    }
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(Error::Pipeline("negadoctor: use the C FFI entry point".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("negadoctor: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

#[inline(always)]
fn process_channel(v: f32, dmin: f32, wb_high: f32, offset: f32,
                   black: f32, exposure: f32, gamma: f32,
                   soft_clip: f32, soft_clip_comp: f32) -> f32 {
    let clamped = v.max(THRESHOLD);
    let density = dmin / clamped;
    // log10(density) = log2(density) * LOG2_TO_LOG10; negate for positive log_density
    let log_density = -density.log2() * LOG2_TO_LOG10;
    let corrected = wb_high * log_density + offset;
    let ten_to_x = (corrected * std::f32::consts::LN_10).exp();
    let print_linear = (-(exposure * ten_to_x + black)).max(0.0);
    let print_gamma = print_linear.powf(gamma);
    if print_gamma > soft_clip {
        let exponent = -(print_gamma - soft_clip) / soft_clip_comp;
        soft_clip + (1.0 - exponent.exp()) * soft_clip_comp
    } else {
        print_gamma
    }
}

/// `dmin`, `wb_high`, `offset` must each point to at least 4 floats.
#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    dmin: &[f32],
    wb_high: &[f32],
    offset: &[f32],
    black: f32,
    gamma: f32,
    soft_clip: f32,
    soft_clip_comp: f32,
    exposure: f32,
) {
    for (ci, co) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        for c in 0..3 {
            co[c] = process_channel(
                ci[c], dmin[c], wb_high[c], offset[c],
                black, exposure, gamma, soft_clip, soft_clip_comp,
            );
        }
        co[3] = ci[3];
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// # Safety
/// `dmin`, `wb_high`, `offset` must each be valid arrays of at least 4 floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_negadoctor_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    dmin: *const f32,
    wb_high: *const f32,
    offset: *const f32,
    black: f32,
    gamma: f32,
    soft_clip: f32,
    soft_clip_comp: f32,
    exposure: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let dmin = std::slice::from_raw_parts(dmin, 4);
    let wb_high = std::slice::from_raw_parts(wb_high, 4);
    let offset = std::slice::from_raw_parts(offset, 4);
    process_pixels(input, output, dmin, wb_high, offset,
                   black, gamma, soft_clip, soft_clip_comp, exposure);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> ([f32; 4], [f32; 4], [f32; 4], f32, f32, f32, f32, f32) {
        let dmin = [0.1f32, 0.1, 0.1, 0.0];
        let wb_high = [1.0f32, 1.0, 1.0, 0.0];
        let offset = [0.0f32; 4];
        let black = 0.0f32;
        let gamma = 1.0f32;
        let soft_clip = 1.0f32;
        let soft_clip_comp = 1.0f32; // 1 - soft_clip
        let exposure = -1.0f32;
        (dmin, wb_high, offset, black, gamma, soft_clip, soft_clip_comp, exposure)
    }

    #[test]
    fn threshold_prevents_zero_division() {
        let (dmin, wb_high, offset, black, gamma, soft_clip, sc_comp, exposure) = default_params();
        let input = vec![0.0f32, 0.0, 0.0, 1.0]; // zero pixel → uses THRESHOLD
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, &dmin, &wb_high, &offset,
                       black, gamma, soft_clip, sc_comp, exposure);
        assert!(output[0].is_finite(), "output should be finite: {}", output[0]);
    }

    #[test]
    fn alpha_passes_through() {
        let (dmin, wb_high, offset, black, gamma, soft_clip, sc_comp, exposure) = default_params();
        let input = vec![0.5f32, 0.5, 0.5, 0.75];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, &dmin, &wb_high, &offset,
                       black, gamma, soft_clip, sc_comp, exposure);
        assert!((output[3] - 0.75).abs() < 1e-7, "alpha: {}", output[3]);
    }

    #[test]
    fn output_is_finite_for_normal_input() {
        let (dmin, wb_high, offset, black, gamma, soft_clip, sc_comp, exposure) = default_params();
        let input = vec![0.3f32, 0.4, 0.5, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, &dmin, &wb_high, &offset,
                       black, gamma, soft_clip, sc_comp, exposure);
        for (i, &v) in output[0..3].iter().enumerate() {
            assert!(v.is_finite(), "ch{i}: {v}");
            assert!(v >= 0.0, "ch{i} negative: {v}");
        }
    }
}
