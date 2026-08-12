use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};
use crate::color::rgb_norm;

pub struct Basicadj;

impl IopProcess for Basicadj {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "basicadj" }
}

fn hlcurve(level: f32, hlcomp: f32, hlrange: f32) -> f32 {
    if hlcomp > 0.0 {
        let mut val = level + (hlrange - 1.0);
        if val == 0.0 { val = 0.000001; }
        let mut y = (val / hlrange) * hlcomp;
        if y <= -1.0 { y = -0.999999; }
        let r = hlrange / (val * hlcomp);
        y.ln_1p() * r
    } else {
        1.0
    }
}

fn lut_gamma(x: f32, gamma: f32, lut: &[f32]) -> f32 {
    if x > 1.0 {
        x.powf(gamma)
    } else {
        lut[((x * 65536.0) as i32).clamp(0, 65535) as usize]
    }
}

fn lut_contrast(x: f32, contrast: f32, mg: f32, inv_mg: f32, lut: &[f32]) -> f32 {
    if x > 1.0 {
        (x * inv_mg).powf(contrast) * mg
    } else {
        lut[((x * 65536.0) as i32).clamp(0, 65535) as usize]
    }
}

#[no_mangle]
pub unsafe extern "C" fn darkroom_basicadj_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    // exposure
    black_point: f32,
    scale: f32,
    // highlight compression
    process_hlcompr: i32,
    hlcomp: f32,
    hlrange: f32,
    lum_r: f32,
    lum_g: f32,
    lum_b: f32,
    // gamma LUT
    process_gamma: i32,
    gamma: f32,
    lut_gamma_ptr: *const f32,
    // contrast
    plain_contrast: i32,
    preserve_colors: i32,
    contrast: f32,
    middle_grey: f32,
    inv_middle_grey: f32,
    lut_contrast_ptr: *const f32,
    // saturation / vibrance
    process_saturation_vibrance: i32,
    saturation: f32,
    vibrance: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lg = std::slice::from_raw_parts(lut_gamma_ptr,   65536);
    let lc = std::slice::from_raw_parts(lut_contrast_ptr, 65536);

    for k in (0..npixels * 4).step_by(4) {
        // 1. Exposure
        output[k]     = (input[k]     - black_point) * scale;
        output[k + 1] = (input[k + 1] - black_point) * scale;
        output[k + 2] = (input[k + 2] - black_point) * scale;

        // 2. Highlight compression
        if process_hlcompr != 0 {
            let lum = output[k] * lum_r + output[k + 1] * lum_g + output[k + 2] * lum_b;
            if lum > 0.0 {
                let ratio = hlcurve(lum, hlcomp, hlrange);
                output[k]     *= ratio;
                output[k + 1] *= ratio;
                output[k + 2] *= ratio;
            }
        }

        // 3. Gamma (per channel, values > 0 only)
        if process_gamma != 0 {
            for c in 0..3 {
                if output[k + c] > 0.0 {
                    output[k + c] = lut_gamma(output[k + c], gamma, lg);
                }
            }
        }

        // 4. Plain contrast (per channel, mutually exclusive with preserve_colors)
        if plain_contrast != 0 {
            for c in 0..3 {
                if output[k + c] > 0.0 {
                    output[k + c] = lut_contrast(output[k + c], contrast, middle_grey, inv_middle_grey, lc);
                }
            }
        }

        // 5. Contrast with preserve colors (luminance-based ratio)
        if preserve_colors != 0 {
            let lum = rgb_norm(output[k], output[k+1], output[k+2], preserve_colors);
            if lum > 0.0 {
                let contrast_lum = (lum * inv_middle_grey).powf(contrast) * middle_grey;
                let ratio = contrast_lum / lum;
                output[k]     *= ratio;
                output[k + 1] *= ratio;
                output[k + 2] *= ratio;
            }
        }

        // 6. Saturation / vibrance
        if process_saturation_vibrance != 0 {
            let avg = (output[k] + output[k + 1] + output[k + 2]) / 3.0;
            let d0 = avg - output[k];
            let d1 = avg - output[k + 1];
            let d2 = avg - output[k + 2];
            let delta = (d0 * d0 + d1 * d1 + d2 * d2).sqrt();
            let p = vibrance * (1.0 - delta.powf(vibrance.abs()));
            let factor = saturation + p;
            output[k]     = avg + factor * (output[k]     - avg);
            output[k + 1] = avg + factor * (output[k + 1] - avg);
            output[k + 2] = avg + factor * (output[k + 2] - avg);
        }

        // 7. Alpha passthrough
        output[k + 3] = input[k + 3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(
        input: &[f32],
        black: f32, scale: f32,
        hlcompr: i32, hlcomp: f32, hlrange: f32,
        pg: i32, gamma: f32,
        pc: i32, pv: i32, contrast: f32, mg: f32,
        psv: i32, sat: f32, vib: f32,
    ) -> Vec<f32> {
        let n = input.len() / 4;
        let mut out = vec![0f32; input.len()];
        let lut_g = vec![0f32; 65536];
        let lut_c = vec![0f32; 65536];
        unsafe {
            darkroom_basicadj_process(
                input.as_ptr(), out.as_mut_ptr(), n,
                black, scale,
                hlcompr, hlcomp, hlrange, 0.2126, 0.7152, 0.0722,
                pg, gamma, lut_g.as_ptr(),
                pc, pv, contrast, mg, 1.0 / mg, lut_c.as_ptr(),
                psv, sat, vib,
            );
        }
        out
    }

    #[test]
    fn exposure_identity() {
        let input = vec![0.5, 0.4, 0.3, 1.0];
        let out = call(&input, 0.0, 1.0, 0, 0.0, 1.0, 0, 1.0, 0, 0, 1.0, 0.1842, 0, 1.0, 0.0);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.4).abs() < 1e-6);
        assert!((out[2] - 0.3).abs() < 1e-6);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn exposure_black_and_scale() {
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let out = call(&input, 0.1, 2.0, 0, 0.0, 1.0, 0, 1.0, 0, 0, 1.0, 0.1842, 0, 1.0, 0.0);
        assert!((out[0] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn alpha_passes_through() {
        let input = vec![0.5, 0.5, 0.5, 0.75];
        let out = call(&input, 0.0, 1.0, 0, 0.0, 1.0, 0, 1.0, 0, 0, 1.0, 0.1842, 0, 1.0, 0.0);
        assert_eq!(out[3], 0.75);
    }

    #[test]
    fn hlcurve_zero_hlcomp_returns_one() {
        assert_eq!(hlcurve(0.8, 0.0, 0.8), 1.0);
    }

    #[test]
    fn saturation_zero_is_grey() {
        // saturation=0 → all channels collapse to average (p=0 when vibrance=0)
        let input = vec![0.8, 0.4, 0.2, 1.0];
        let out = call(&input, 0.0, 1.0, 0, 0.0, 1.0, 0, 1.0, 0, 0, 1.0, 0.1842, 1, 0.0, 0.0);
        let avg = (0.8 + 0.4 + 0.2) / 3.0;
        assert!((out[0] - avg).abs() < 1e-5);
        assert!((out[1] - avg).abs() < 1e-5);
        assert!((out[2] - avg).abs() < 1e-5);
    }
}
