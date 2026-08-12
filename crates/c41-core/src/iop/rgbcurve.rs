use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;
use crate::color::rgb_norm;

pub struct RgbCurve;

impl IopProcess for RgbCurve {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "rgbcurve" }
}

/// `coeff[1] * (x * coeff[0]).powf(coeff[2])` — matches dt_iop_eval_exp.
#[inline(always)]
fn eval_exp(coeffs: &[f32], x: f32) -> f32 {
    coeffs[1] * (x * coeffs[0]).powf(coeffs[2])
}

#[inline(always)]
fn lut_or_exp(tbl: &[f32], coeffs: &[f32], xm: f32, v: f32) -> f32 {
    if v < xm {
        tbl[((v * 0x1_0000_u32 as f32) as usize).clamp(0, 0xffff)]
    } else {
        eval_exp(coeffs, v)
    }
}

/// RGB-curve IOP: per-channel or linked-channel LUT tone mapping.
///
/// autoscale: 0 = AUTOMATIC_RGB (linked), 1 = MANUAL_RGB (independent channels)
/// preserve_colors: 0 = NONE, non-zero = luminance-norm mode (uses rgb_norm from color.rs)
///
/// Each table is 65536 floats; each unbounded_coeffs group is 3 floats.
/// unbounded_coeffs layout: [R*3 | G*3 | B*3] = 9 floats total.
/// xm_r/g/b = 1.0 / unbounded_coeffs[channel][0] (pre-computed by C caller).
#[no_mangle]
pub unsafe extern "C" fn darkroom_rgbcurve_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    table_r: *const f32,        // 65536 floats
    table_g: *const f32,        // 65536 floats
    table_b: *const f32,        // 65536 floats
    unbounded_r: *const f32,    // 3 floats
    unbounded_g: *const f32,    // 3 floats
    unbounded_b: *const f32,    // 3 floats
    xm_r: f32,
    xm_g: f32,
    xm_b: f32,
    autoscale: i32,     // 0 = AUTOMATIC_RGB, 1 = MANUAL_RGB
    preserve_colors: i32, // 0 = NONE
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tr = std::slice::from_raw_parts(table_r, 0x10000);
    let tg = std::slice::from_raw_parts(table_g, 0x10000);
    let tb = std::slice::from_raw_parts(table_b, 0x10000);
    let ur = std::slice::from_raw_parts(unbounded_r, 3);
    let ug = std::slice::from_raw_parts(unbounded_g, 3);
    let ub = std::slice::from_raw_parts(unbounded_b, 3);

    for px in 0..npixels {
        let base = px * 4;
        let i = &inp[base..base + 4];
        let o = &mut out[base..base + 4];

        if autoscale == 1 {
            // MANUAL_RGB: independent per-channel curves
            o[0] = lut_or_exp(tr, ur, xm_r, i[0]);
            o[1] = lut_or_exp(tg, ug, xm_g, i[1]);
            o[2] = lut_or_exp(tb, ub, xm_b, i[2]);
        } else if preserve_colors == 0 {
            // AUTOMATIC_RGB, no luminance norm: apply R curve to all channels
            o[0] = lut_or_exp(tr, ur, xm_r, i[0]);
            o[1] = lut_or_exp(tr, ur, xm_r, i[1]);
            o[2] = lut_or_exp(tr, ur, xm_r, i[2]);
        } else {
            // AUTOMATIC_RGB + preserve_colors: apply R curve to luminance, ratio-scale RGB
            let lum = rgb_norm(i[0], i[1], i[2], preserve_colors);
            if lum > 0.0 {
                let curve_lum = lut_or_exp(tr, ur, xm_r, lum);
                let ratio = curve_lum / lum;
                o[0] = ratio * i[0];
                o[1] = ratio * i[1];
                o[2] = ratio * i[2];
            } else {
                o[0] = i[0];
                o[1] = i[1];
                o[2] = i[2];
            }
        }
        o[3] = i[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut() -> Vec<f32> {
        (0..0x10000usize).map(|i| i as f32 / 0xffff as f32).collect()
    }
    fn linear_coeffs() -> [f32; 3] { [1.0, 1.0, 1.0] }

    #[test]
    fn manual_mode_identity() {
        let tbl = identity_lut();
        let c = linear_coeffs();
        let inp = [0.3f32, 0.5, 0.8, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgbcurve_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tbl.as_ptr(), tbl.as_ptr(), tbl.as_ptr(),
                c.as_ptr(), c.as_ptr(), c.as_ptr(),
                1.0, 1.0, 1.0,
                1, 0,
            )
        };
        assert!((out[0] - 0.3).abs() < 1e-4);
        assert!((out[1] - 0.5).abs() < 1e-4);
        assert!((out[2] - 0.8).abs() < 1e-4);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn automatic_none_applies_r_curve_to_all() {
        let mut tbl = identity_lut();
        // Double all values
        for v in &mut tbl { *v = (*v * 2.0).min(1.0); }
        let c = linear_coeffs();
        let inp = [0.2f32, 0.4, 0.6, 0.5];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgbcurve_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tbl.as_ptr(), tbl.as_ptr(), tbl.as_ptr(),
                c.as_ptr(), c.as_ptr(), c.as_ptr(),
                1.0, 1.0, 1.0,
                0, 0,
            )
        };
        // All channels should be doubled (clamped to 1.0 where necessary)
        assert!((out[0] - (0.2f32 * 2.0).min(1.0)).abs() < 1e-3);
        assert!((out[1] - (0.4f32 * 2.0).min(1.0)).abs() < 1e-3);
        assert!((out[2] - (0.6f32 * 2.0).min(1.0)).abs() < 1e-3);
    }

    #[test]
    fn alpha_always_passes_through() {
        let tbl = identity_lut();
        let c = linear_coeffs();
        let inp = [0.5f32, 0.5, 0.5, 0.75];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgbcurve_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tbl.as_ptr(), tbl.as_ptr(), tbl.as_ptr(),
                c.as_ptr(), c.as_ptr(), c.as_ptr(),
                1.0, 1.0, 1.0,
                1, 0,
            )
        };
        assert_eq!(out[3], 0.75);
    }
}
