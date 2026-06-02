use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct ProfileGamma;

impl IopProcess for ProfileGamma {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "profile_gamma" }
}

/// IEEE 754 bit-manipulation fast log2 approximation (matches src/common/math.h).
#[inline(always)]
fn fastlog2(x: f32) -> f32 {
    let bits = x.to_bits();
    let mx = f32::from_bits((bits & 0x007F_FFFF) | 0x3F00_0000);
    let y = bits as f32 * 1.192_092_9e-7_f32;
    y - 124.225_515f32 - 1.498_030_3f32 * mx - 1.725_88f32 / (0.352_088_7f32 + mx)
}

/// `coeff[1] * (x * coeff[0]).powf(coeff[2])` — matches dt_iop_eval_exp.
#[inline(always)]
fn eval_exp(coeffs: &[f32], x: f32) -> f32 {
    coeffs[1] * (x * coeffs[0]).powf(coeffs[2])
}

/// Profile-gamma IOP: logarithmic or gamma LUT tone mapping.
///
/// mode 0 (LOG): applies fastlog2 normalization to every element including alpha.
/// mode 1 (GAMMA): applies LUT/eval_exp to channels 0..2 only (ch=4, alpha not touched).
///
/// For GAMMA mode the caller must ensure alpha is handled separately (dt_iop_alpha_copy
/// is called by C process() when mask_display is set — we skip that here).
#[no_mangle]
pub unsafe extern "C" fn darkroom_profile_gamma_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mode: i32,
    grey: f32,          // grey_point / 100.0
    dynamic_range: f32,
    shadows_range: f32,
    table: *const f32,          // 65536 floats (used in GAMMA mode)
    unbounded_coeffs: *const f32, // 3 floats (used in GAMMA mode)
) {
    const NOISE: f32 = 5.960_464_5e-8_f32; // 2^-24 ≈ powf(2,-16) from C

    if mode == 0 {
        // LOG mode: process all ch*npixels elements
        let total = npixels * 4;
        let inp = std::slice::from_raw_parts(in_buf, total);
        let out = std::slice::from_raw_parts_mut(out_buf, total);
        for k in 0..total {
            let mut tmp = inp[k] / grey;
            if tmp < NOISE { tmp = NOISE; }
            tmp = (fastlog2(tmp) - shadows_range) / dynamic_range;
            out[k] = if tmp < NOISE { NOISE } else { tmp };
        }
    } else {
        // GAMMA mode: process channels 0..2 per pixel, skip channel 3
        let tbl = std::slice::from_raw_parts(table, 0x10000);
        let coeffs = std::slice::from_raw_parts(unbounded_coeffs, 3);
        let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
        let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
        for px in 0..npixels {
            let base = px * 4;
            for i in 0..3 {
                let v = inp[base + i];
                out[base + i] = if v < 1.0 {
                    tbl[((v * 0x1_0000_u32 as f32) as usize).min(0xffff)]
                } else {
                    eval_exp(coeffs, v)
                };
            }
            // alpha not touched in GAMMA mode (C handles separately)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastlog2_near_one() {
        // log2(1.0) = 0.0; our approximation should be close
        assert!((fastlog2(1.0)).abs() < 0.01);
    }

    #[test]
    fn fastlog2_near_two() {
        assert!((fastlog2(2.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn log_mode_grey_normalizes_to_zero() {
        // input == grey → fastlog2(1.0) ≈ 0; result = (0 - shadows_range)/dynamic_range
        let grey = 0.18f32;
        let dynamic_range = 10.0f32;
        let shadows_range = -5.0f32;
        let inp = [grey, grey, grey, grey];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_profile_gamma_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                0, grey, dynamic_range, shadows_range,
                std::ptr::null(), std::ptr::null(),
            )
        };
        let expected = (0.0 - shadows_range) / dynamic_range;
        for &v in &out {
            assert!((v - expected).abs() < 0.05);
        }
    }

    #[test]
    fn gamma_mode_lut_passthrough() {
        // LUT = identity (index/65535), coeffs for linear extrapolation
        let tbl: Vec<f32> = (0..0x10000usize).map(|i| i as f32 / 0xffff as f32).collect();
        let coeffs = [1.0f32, 1.0, 1.0]; // eval_exp(1,x) = x^1
        let inp = [0.5f32, 0.25, 0.75, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_profile_gamma_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1, 0.18, 10.0, -5.0,
                tbl.as_ptr(), coeffs.as_ptr(),
            )
        };
        assert!((out[0] - 0.5).abs() < 1e-4);
        assert!((out[1] - 0.25).abs() < 1e-4);
        assert!((out[2] - 0.75).abs() < 1e-4);
        // alpha not written
    }
}

/// Build the tone curve LUT (65536 entries) for profile_gamma commit_params.
///
/// Three variants, all dispatched internally:
///   gamma == 1.0          → linear passthrough  table[k] = k/0x10000
///   linear == 0.0         → pure power curve    table[k] = (k/0x10000)^gamma
///   otherwise             → linear segment + power curve (sRGB-like)
///
/// Matches the three DT_OMP_FOR loops in src/iop/profile_gamma.c:438-472.
#[no_mangle]
pub unsafe extern "C" fn darkroom_profile_gamma_build_lut(
    table: *mut f32,
    gamma: f32,
    linear: f32,
) {
    let lut = std::slice::from_raw_parts_mut(table, 0x10000);
    const N: f32 = 0x10000 as f32;

    if (gamma - 1.0).abs() < f32::EPSILON {
        for k in 0..0x10000usize {
            lut[k] = k as f32 / N;
        }
    } else if linear == 0.0 {
        for k in 0..0x10000usize {
            lut[k] = (k as f32 / N).powf(gamma);
        }
    } else {
        let (a, b, c, g) = if linear < 1.0 {
            let g = gamma * (1.0 - linear) / (1.0 - gamma * linear);
            let a = 1.0 / (1.0 + linear * (g - 1.0));
            let b = linear * (g - 1.0) * a;
            let c = (a * linear + b).powf(g) / linear;
            (a, b, c, g)
        } else {
            (0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32)
        };
        let linear_end = (N * linear) as usize;
        for k in 0..0x10000usize {
            lut[k] = if k < linear_end {
                c * k as f32 / N
            } else {
                (a * k as f32 / N + b).powf(g)
            };
        }
    }
}
