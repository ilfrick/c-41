//! Vibrance IOP — saturation-weighted chroma boost in Lab colorspace.
//!
//! Replaces the OMP loop in src/iop/vibrance.c::process().
//!
//! Per-pixel formula (Lab input, channels L/a/b/alpha):
//!   sw = sqrt(a² + b²) / 256          — saturation weight ∈ [0,1]
//!   ls = 1 − (amount × sw × 0.25)     — luma scaling
//!   ss = 1 + (amount × sw)            — chroma scaling
//!   out = [L×ls, a×ss, b×ss, alpha]

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct Vibrance;

impl IopProcess for Vibrance {
    fn name(&self) -> &'static str {
        "vibrance"
    }

    fn process(
        &self,
        input: &[f32],
        output: &mut [f32],
        params: &IopParams,
        _roi: &RoiIn,
    ) -> Result<()> {
        // dt_iop_vibrance_data_t has a single f32 already scaled by 0.01.
        let amount = unsafe {
            params
                .cast::<f32>()
                .ok_or_else(|| Error::Pipeline("vibrance: params too short".into()))?
        };
        process_pixels(input, output, amount);
        Ok(())
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("vibrance: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// Apply vibrance (saturation-weighted chroma boost) to a flat RGBA pixel buffer.
///
/// `amount` must be pre-scaled by 0.01 (matching the C commit_params step).
#[inline]
pub fn process_pixels(input: &[f32], output: &mut [f32], amount: f32) {
    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let a = chunk_in[1];
        let b = chunk_in[2];
        let sw = (a * a + b * b).sqrt() / 256.0;
        let ls = 1.0 - amount * sw * 0.25;
        let ss = 1.0 + amount * sw;
        chunk_out[0] = chunk_in[0] * ls;
        chunk_out[1] = a * ss;
        chunk_out[2] = b * ss;
        chunk_out[3] = chunk_in[3];
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/vibrance.c in place of the OMP loop.
///
/// `amount` is `d->amount * 0.01` (already scaled in commit_params).
///
/// # Safety
/// `in_buf` and `out_buf` must be non-overlapping arrays of `npixels * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_vibrance_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    amount: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    process_pixels(input, output, amount);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_amount_is_identity() {
        let input = vec![50.0f32, 20.0, -30.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0);
        for (i, o) in input.iter().zip(output.iter()) {
            assert!((i - o).abs() < 1e-7, "identity failed: {i} != {o}");
        }
    }

    #[test]
    fn alpha_always_passes_through() {
        let input = vec![50.0f32, 10.0, 10.0, 0.42];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.5);
        assert!((output[3] - 0.42).abs() < 1e-7, "alpha changed");
    }

    #[test]
    fn grey_pixel_unchanged_luma() {
        // Grey pixel: a=b=0, sw=0 → ls=1, ss=1, no change
        let input = vec![60.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 1.0);
        assert!((output[0] - 60.0).abs() < 1e-6, "grey luma changed");
        assert!(output[1].abs() < 1e-7);
        assert!(output[2].abs() < 1e-7);
    }

    #[test]
    fn known_values_match_c_formula() {
        let amount = 0.5f32;
        let input = vec![50.0f32, 128.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, amount);

        let a = 128.0f32;
        let b = 0.0f32;
        let sw = (a * a + b * b).sqrt() / 256.0;
        let ls = 1.0 - amount * sw * 0.25;
        let ss = 1.0 + amount * sw;
        assert!((output[0] - 50.0 * ls).abs() < 1e-5);
        assert!((output[1] - a * ss).abs() < 1e-5);
        assert!((output[2] - b * ss).abs() < 1e-5);
    }
}
