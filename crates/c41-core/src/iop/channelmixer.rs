//! Channel-mixer IOP — linear RGB and HSL channel remapping.
//!
//! Replaces the four sub-process functions in src/iop/channelmixer.c::process().
//!
//! operation_mode values:
//!   0 = OPERATION_MODE_RGB  — rgb_matrix only, out = max(M*in, 0)
//!   1 = OPERATION_MODE_GRAY — same but broadcast result to R=G=B
//!   2 = OPERATION_MODE_HSL_V1 — hsl_matrix replaces H/S/L individually if non-zero, then rgb_matrix
//!   3 = OPERATION_MODE_HSL_V2 — hsl_matrix maps RGB→HSL mix, then rgb_matrix
//!
//! Alpha channel: follows C behaviour (not written for modes 0/1/2/3).
//! We copy input alpha for safety.

use crate::{
    color::{hsl2rgb, rgb2hsl},
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

pub struct ChannelMixer;

impl IopProcess for ChannelMixer {
    fn name(&self) -> &'static str {
        "channelmixer"
    }
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(Error::Pipeline("channelmixer: use the C FFI entry point".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("channelmixer: OpenCL path not yet ported".into()))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline(always)]
fn mat3_row_dot(m: &[f32], row: usize, r: f32, g: f32, b: f32) -> f32 {
    let j = row * 3;
    m[j] * r + m[j + 1] * g + m[j + 2] * b
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// `hsl_matrix` and `rgb_matrix` must each point to exactly 9 floats (3×3, row-major).
#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    hsl_matrix: &[f32; 9],
    rgb_matrix: &[f32; 9],
    operation_mode: i32,
) {
    match operation_mode {
        0 => {
            // OPERATION_MODE_RGB
            for (ci, co) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
                for i in 0..3 {
                    co[i] = mat3_row_dot(rgb_matrix, i, ci[0], ci[1], ci[2]).max(0.0);
                }
                co[3] = ci[3];
            }
        }
        1 => {
            // OPERATION_MODE_GRAY
            for (ci, co) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
                let gray = mat3_row_dot(rgb_matrix, 0, ci[0], ci[1], ci[2]).max(0.0);
                co[0] = gray;
                co[1] = gray;
                co[2] = gray;
                co[3] = ci[3];
            }
        }
        2 => {
            // OPERATION_MODE_HSL_V1
            for (ci, co) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
                let hmix = mat3_row_dot(hsl_matrix, 0, ci[0], ci[1], ci[2]).clamp(0.0, 1.0);
                let smix = mat3_row_dot(hsl_matrix, 1, ci[0], ci[1], ci[2]).clamp(0.0, 1.0);
                let lmix = mat3_row_dot(hsl_matrix, 2, ci[0], ci[1], ci[2]).clamp(0.0, 1.0);

                let (r, g, b) = if hmix != 0.0 || smix != 0.0 || lmix != 0.0 {
                    let (h, s, l) = rgb2hsl(ci[0], ci[1], ci[2]);
                    let h2 = if hmix != 0.0 { hmix } else { h };
                    let s2 = if smix != 0.0 { smix } else { s };
                    let l2 = if lmix != 0.0 { lmix } else { l };
                    let (r, g, b, _) = hsl2rgb(h2, s2, l2);
                    (r, g, b)
                } else {
                    (ci[0], ci[1], ci[2])
                };

                for i in 0..3 {
                    co[i] = mat3_row_dot(rgb_matrix, i, r, g, b).clamp(0.0, 1.0);
                }
                co[3] = ci[3];
            }
        }
        3 | _ => {
            // OPERATION_MODE_HSL_V2
            for (ci, co) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
                let (mut r, mut g, mut b) = (ci[0], ci[1], ci[2]);
                let hm = mat3_row_dot(hsl_matrix, 0, r, g, b).clamp(0.0, 1.0);
                let sm = mat3_row_dot(hsl_matrix, 1, r, g, b).clamp(0.0, 1.0);
                let lm = mat3_row_dot(hsl_matrix, 2, r, g, b).clamp(0.0, 1.0);

                if hm != 0.0 || sm != 0.0 || lm != 0.0 {
                    // rgb2hsl expects clipped values
                    let rc = r.clamp(0.0, 1.0);
                    let gc = g.clamp(0.0, 1.0);
                    let bc = b.clamp(0.0, 1.0);
                    let (h, s, l) = rgb2hsl(rc, gc, bc);
                    let h2 = if hm != 0.0 { hm } else { h };
                    let s2 = if sm != 0.0 { sm } else { s };
                    let l2 = if lm != 0.0 { lm } else { l };
                    let (rn, gn, bn, _) = hsl2rgb(h2, s2, l2);
                    r = rn;
                    g = gn;
                    b = bn;
                }

                for i in 0..3 {
                    co[i] = mat3_row_dot(rgb_matrix, i, r, g, b).max(0.0);
                }
                co[3] = ci[3];
            }
        }
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// # Safety
/// `hsl_matrix` and `rgb_matrix` must each be valid arrays of exactly 9 floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_channelmixer_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    hsl_matrix: *const f32,
    rgb_matrix: *const f32,
    operation_mode: i32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let hsl = &*(hsl_matrix as *const [f32; 9]);
    let rgb = &*(rgb_matrix as *const [f32; 9]);
    process_pixels(input, output, hsl, rgb, operation_mode);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_rgb_matrix() -> [f32; 9] {
        [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]
    }
    fn zero_hsl_matrix() -> [f32; 9] { [0.0; 9] }

    #[test]
    fn rgb_identity_passthrough() {
        let input = vec![0.3f32, 0.5, 0.7, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, &zero_hsl_matrix(), &identity_rgb_matrix(), 0);
        assert!((output[0] - 0.3).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn gray_mode_broadcasts() {
        // rgb_matrix row 0: [0.2126, 0.7152, 0.0722] ≈ luminance
        let rgb_m = [0.2126f32, 0.7152, 0.0722,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0];
        let input = vec![1.0f32, 0.0, 0.0, 1.0]; // pure red
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, &zero_hsl_matrix(), &rgb_m, 1);
        // all 3 channels should equal 0.2126
        assert!((output[0] - 0.2126).abs() < 1e-5);
        assert_eq!(output[0], output[1]);
        assert_eq!(output[1], output[2]);
    }

    #[test]
    fn hsl_v1_no_mix_passthrough() {
        // zero hsl_matrix → no HSL modification; identity rgb_matrix
        let input = vec![0.6f32, 0.3, 0.1, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, &zero_hsl_matrix(), &identity_rgb_matrix(), 2);
        assert!((output[0] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn alpha_passes_through_all_modes() {
        let input = vec![0.5f32, 0.5, 0.5, 0.42];
        for mode in 0..4 {
            let mut output = vec![0.0f32; 4];
            process_pixels(&input, &mut output, &zero_hsl_matrix(), &identity_rgb_matrix(), mode);
            assert!((output[3] - 0.42).abs() < 1e-7, "mode {mode}: alpha={}", output[3]);
        }
    }
}
