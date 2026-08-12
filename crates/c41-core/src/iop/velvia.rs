//! Velvia IOP — film-emulation saturation boost in RGB colorspace.
//!
//! Replaces the OMP loop in src/iop/velvia.c::process().
//!
//! Per-pixel algorithm (RGB input, values nominally in [0, 1]):
//!   pmax = max(R, G, B)
//!   pmin = min(R, G, B)
//!   plum = (pmax + pmin) / 2
//!   psat = plum ≤ 0.5 ? (pmax-pmin)/(ε+pmax+pmin) : (pmax-pmin)/(ε+max(0,2-pmax-pmin))
//!   pweight = clamp((1 − 1.5·psat + (1 + |plum−0.5|·2)·(1−bias)) / (1+(1−bias)), 0, 1)
//!   saturation = strength · pweight
//!   out[c] = clamp(in[c] + saturation·(in[c] − 0.5·(sum of other two channels)), 0, 1)
//!
//! When strength ≤ 0 the input is copied to output unchanged (matching C early-return).

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct Velvia;

impl IopProcess for Velvia {
    fn name(&self) -> &'static str {
        "velvia"
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
        struct VelviaData {
            strength: f32,
            bias: f32,
        }
        let d = unsafe {
            params
                .cast::<VelviaData>()
                .ok_or_else(|| Error::Pipeline("velvia: params too short".into()))?
        };
        process_pixels(input, output, d.strength / 100.0, d.bias);
        Ok(())
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("velvia: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// `strength` must already be divided by 100 (matching the C `data->strength / 100.0f`).
#[inline]
pub fn process_pixels(input: &[f32], output: &mut [f32], strength: f32, bias: f32) {
    if strength <= 0.0 {
        output.copy_from_slice(input);
        return;
    }

    const EPS: f32 = 1e-5;

    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let r = chunk_in[0];
        let g = chunk_in[1];
        let b = chunk_in[2];

        let pmax = r.max(g).max(b);
        let pmin = r.min(g).min(b);
        let plum = (pmax + pmin) * 0.5;

        let psat = if plum <= 0.5 {
            (pmax - pmin) / (EPS + pmax + pmin)
        } else {
            (pmax - pmin) / (EPS + (2.0 - pmax - pmin).max(0.0))
        };

        let one_minus_bias = 1.0 - bias;
        let pweight = ((1.0 - 1.5 * psat
            + (1.0 + (plum - 0.5).abs() * 2.0) * one_minus_bias)
            / (1.0 + one_minus_bias))
            .clamp(0.0, 1.0);

        let sat = strength * pweight;

        // othersum[c] = sum of the other two RGB channels
        let other = [g + b, b + r, r + g];
        chunk_out[0] = (r + sat * (r - 0.5 * other[0])).clamp(0.0, 1.0);
        chunk_out[1] = (g + sat * (g - 0.5 * other[1])).clamp(0.0, 1.0);
        chunk_out[2] = (b + sat * (b - 0.5 * other[2])).clamp(0.0, 1.0);
        chunk_out[3] = chunk_in[3];
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/velvia.c.
///
/// `strength` is `data->strength / 100.0f` (pre-scaled in the C wrapper).
///
/// # Safety
/// `in_buf` and `out_buf` must be non-overlapping arrays of `npixels * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_velvia_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    strength: f32,
    bias: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    process_pixels(input, output, strength, bias);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_strength_is_identity() {
        let input: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let mut output = vec![0.0f32; 8];
        process_pixels(&input, &mut output, 0.0, 1.0);
        for (i, o) in input.iter().zip(output.iter()) {
            assert!((i - o).abs() < 1e-7);
        }
    }

    #[test]
    fn negative_strength_is_identity() {
        let input = vec![0.2f32, 0.5, 0.8, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, -0.5, 1.0);
        for (i, o) in input.iter().zip(output.iter()) {
            assert!((i - o).abs() < 1e-7);
        }
    }

    #[test]
    fn grey_pixel_unchanged_by_saturation() {
        // R=G=B → pmax=pmin=plum, psat=0 → saturation=strength → but in[c]=0.5*(G+B)
        // so (in[c] - 0.5*other) = R - 0.5*(G+B) = 0 for equal channels
        let input = vec![0.5f32, 0.5, 0.5, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.5, 1.0);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn alpha_passes_through() {
        let input = vec![0.3f32, 0.6, 0.1, 0.75];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.5, 1.0);
        assert!((output[3] - 0.75).abs() < 1e-7);
    }

    #[test]
    fn output_clamped_to_unit_range() {
        // Force extreme values that might overflow before clamping
        let input = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 1.0, 1.0);
        for &v in output[..3].iter() {
            assert!(v >= 0.0 && v <= 1.0, "out of range: {v}");
        }
    }
}
