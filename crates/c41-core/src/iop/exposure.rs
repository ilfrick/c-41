//! Exposure IOP — first Rust replacement for src/iop/exposure.c.
//!
//! The C pixelpipe still orchestrates the module (commit_params, GUI, etc.).
//! Only the inner pixel loop is replaced:
//!   out[k] = (in[k] - black) * scale          ← exposure.c:562
//!
//! Phase 1 goal: identical numerical output to the C loop, verified by tests.

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

// ── Data types ────────────────────────────────────────────────────────────────

/// Computed runtime params that drive the pixel loop.
/// Layout mirrors dt_iop_exposure_data_t.{black, scale} (exposure.c:104–110).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExposureData {
    pub black: f32,
    pub scale: f32,
}

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct Exposure;

impl IopProcess for Exposure {
    fn name(&self) -> &'static str {
        "exposure"
    }

    fn process(
        &self,
        input: &[f32],
        output: &mut [f32],
        params: &IopParams,
        _roi: &RoiIn,
    ) -> Result<()> {
        // Safety: ExposureData is repr(C); bytes come from C's commit_params.
        let d = unsafe {
            params
                .cast::<ExposureData>()
                .ok_or_else(|| Error::Pipeline("exposure: params buffer too short".into()))?
        };
        process_pixels(input, output, d.black, d.scale);
        Ok(())
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("exposure: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// Apply black-level correction and exposure scaling to a flat pixel buffer.
///
/// Equivalent to the C loop in exposure.c:559–563:
///   for(size_t k = 0; k < ch * npixels; k++)
///       out[k] = (in[k] - black) * scale;
///
/// LLVM auto-vectorises this to AVX2/SSE4 on x86_64 at opt-level ≥ 2.
#[inline]
pub fn process_pixels(input: &[f32], output: &mut [f32], black: f32, scale: f32) {
    for (o, &i) in output.iter_mut().zip(input.iter()) {
        *o = (i - black) * scale;
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/exposure.c instead of the hand-written C loop.
///
/// # Safety
/// - `in_buf` and `out_buf` must be valid, non-overlapping float arrays of
///   length `npixels * channels`, aligned to at least 4 bytes.
/// - Caller (the darktable pixelpipe) owns both buffers for the duration of
///   this call.
#[no_mangle]
pub unsafe extern "C" fn darkroom_exposure_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    channels: usize,
    black: f32,
    scale: f32,
) {
    let len = npixels * channels;
    let input = std::slice::from_raw_parts(in_buf, len);
    let output = std::slice::from_raw_parts_mut(out_buf, len);
    process_pixels(input, output, black, scale);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn zero_black_unit_scale_is_identity() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let mut output = vec![0.0_f32; 16];
        process_pixels(&input, &mut output, 0.0, 1.0);
        assert!(max_delta(&input, &output) < 1e-7, "identity failed");
    }

    #[test]
    fn known_values_match_c_formula() {
        // Replicates the C loop: out[k] = (in[k] - black) * scale
        let black = 0.02f32;
        let scale = 2.0f32;
        let input = vec![0.0f32, 0.5, 1.0, -0.1];
        let expected: Vec<f32> = input.iter().map(|&v| (v - black) * scale).collect();

        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, black, scale);
        assert!(
            max_delta(&output, &expected) < 1e-6,
            "mismatch: got {output:?}, expected {expected:?}"
        );
    }

    #[test]
    fn ffi_entry_point_matches_safe_fn() {
        let input = vec![0.1f32, 0.4, 0.7, 1.0, 0.25, 0.5, 0.75, 0.9];
        let black = 0.01f32;
        let scale = 1.5f32;

        let mut safe_out = vec![0.0f32; 8];
        process_pixels(&input, &mut safe_out, black, scale);

        let mut ffi_out = vec![0.0f32; 8];
        unsafe {
            darkroom_exposure_process(
                input.as_ptr(),
                ffi_out.as_mut_ptr(),
                4, // npixels
                2, // channels
                black,
                scale,
            );
        }
        assert!(
            max_delta(&safe_out, &ffi_out) < 1e-7,
            "FFI mismatch: safe={safe_out:?}, ffi={ffi_out:?}"
        );
    }
}
