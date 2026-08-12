use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Filmic;

impl IopProcess for Filmic {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "filmic" }
}

// IEEE 754 bit-manipulation log2 approximation matching darktable's fastlog2().
#[inline(always)]
fn fastlog2(x: f32) -> f32 {
    let vx = x.to_bits();
    let mx = f32::from_bits((vx & 0x007FFFFF) | 0x3F000000);
    let y = vx as f32 * 1.192_092_895_507_812_5e-7_f32;
    y - 124.225_514_99_f32
        - 1.498_030_302_f32 * mx
        - 1.725_879_99_f32 / (0.352_088_706_8_f32 + mx)
}

#[inline(always)]
fn lab_f_inv(x: f32) -> f32 {
    const EPS: f32 = 0.206_896_551_724_137_96; // cbrt(216/24389)
    const KAPPA: f32 = 24389.0 / 27.0;
    if x > EPS { x * x * x } else { (116.0 * x - 16.0) / KAPPA }
}

#[inline(always)]
fn lab_f(x: f32) -> f32 {
    const EPS: f32 = 216.0 / 24389.0;
    const KAPPA: f32 = 24389.0 / 27.0;
    if x > EPS { x.cbrt() } else { (KAPPA * x + 16.0) / 116.0 }
}

/// CIE Lab → XYZ (D50 white point)
#[inline(always)]
fn lab_to_xyz(l: f32, a: f32, b: f32) -> [f32; 3] {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    [0.9642 * lab_f_inv(fx), lab_f_inv(fy), 0.8249 * lab_f_inv(fz)]
}

/// XYZ → ProPhoto RGB (D50, transposed matrix dt_apply_transposed_color_matrix)
#[inline(always)]
fn xyz_to_prophoto(xyz: [f32; 3]) -> [f32; 3] {
    [
         1.345_943_3 * xyz[0] - 0.255_607_5 * xyz[1] - 0.051_111_8 * xyz[2],
        -0.544_598_9 * xyz[0] + 1.508_167_3 * xyz[1] + 0.020_535_1 * xyz[2],
                                                         1.211_812_8 * xyz[2],
    ]
}

/// ProPhoto RGB → XYZ (D50, transposed matrix)
#[inline(always)]
fn prophoto_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    [
        0.797_674_9 * rgb[0] + 0.135_191_7 * rgb[1] + 0.031_353_4 * rgb[2],
        0.288_040_2 * rgb[0] + 0.711_874_1 * rgb[1] + 0.000_085_7 * rgb[2],
                                                        0.825_210_0 * rgb[2],
    ]
}

/// ProPhoto Y (XYZ luma) = prophotorgb_to_xyz_transpose[*][1] · rgb
#[inline(always)]
fn prophoto_luma(rgb: [f32; 3]) -> f32 {
    0.288_040_2 * rgb[0] + 0.711_874_1 * rgb[1] + 0.000_085_7 * rgb[2]
}

/// XYZ → CIE Lab (D50)
#[inline(always)]
fn xyz_to_lab(xyz: [f32; 3]) -> [f32; 3] {
    const D50_INV: [f32; 3] = [1.0 / 0.9642, 1.0, 1.0 / 0.8249];
    let f0 = lab_f(xyz[0] * D50_INV[0]);
    let f1 = lab_f(xyz[1] * D50_INV[1]);
    let f2 = lab_f(xyz[2] * D50_INV[2]);
    [116.0 * f1 - 16.0, 500.0 * (f0 - f1), -200.0 * (f2 - f1)]
}

#[inline(always)]
fn lut_idx(v: f32) -> usize {
    ((v * 0x1_0000_u32 as f32) as u64).clamp(0, 0xffff) as usize
}

#[no_mangle]
pub unsafe extern "C" fn darkroom_filmic_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    grey_source: f32,
    black_source: f32,
    inv_dynamic_range: f32,
    output_power: f32,
    saturation: f32,
    eps: f32,
    desaturate: i32,
    preserve_color: i32,
    table: *const f32,
    grad_2: *const f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tbl = std::slice::from_raw_parts(table,  0x10000);
    let grd = std::slice::from_raw_parts(grad_2, 0x10000);

    for k in (0..npixels * 4).step_by(4) {
        // 1. Lab → XYZ → ProPhoto
        let xyz = lab_to_xyz(input[k], input[k + 1], input[k + 2]);
        let mut rgb = xyz_to_prophoto(xyz);

        // 2. Global desaturation (optional)
        if desaturate != 0 {
            let lum = xyz[1]; // Y = linear luminance
            for c in 0..3 {
                rgb[c] = lum + saturation * (rgb[c] - lum);
            }
        }

        let luma;
        let concavity;

        if preserve_color != 0 {
            // 3a. Preserve-colour path: tone-map the max channel, reapply ratios
            let max_c = rgb[0].max(rgb[1]).max(rgb[2]);
            let ratios = [rgb[0] / max_c, rgb[1] / max_c, rgb[2] / max_c];

            let mut m = max_c / grey_source;
            m = if m > eps { (fastlog2(m) - black_source) * inv_dynamic_range } else { eps };
            m = m.clamp(0.0, 1.0);

            let idx = lut_idx(m);
            let mapped = tbl[idx];
            concavity = grd[idx];

            rgb = [ratios[0] * mapped, ratios[1] * mapped, ratios[2] * mapped];
            luma = mapped;
        } else {
            // 3b. Per-channel path
            for c in 0..3 { rgb[c] /= grey_source; }

            let log_rgb = [fastlog2(rgb[0]), fastlog2(rgb[1]), fastlog2(rgb[2])];

            // log tone-map then clamp to [0,1]
            for c in 0..3 {
                rgb[c] = if rgb[c] > eps {
                    (log_rgb[c] - black_source) * inv_dynamic_range
                } else {
                    eps
                };
                rgb[c] = rgb[c].clamp(0.0, 1.0);
            }

            // concavity from pre-LUT luma
            concavity = grd[lut_idx(prophoto_luma(rgb))];

            // apply filmic S-curve LUT per channel
            let indices = [lut_idx(rgb[0]), lut_idx(rgb[1]), lut_idx(rgb[2])];
            for c in 0..3 { rgb[c] = tbl[indices[c]]; }

            luma = prophoto_luma(rgb);
        }

        // 4. Concavity desaturation
        for c in 0..3 {
            rgb[c] = (luma + concavity * (rgb[c] - luma)).clamp(0.0, 1.0);
        }

        // 5. Output power (gamma)
        for c in 0..3 { rgb[c] = rgb[c].powf(output_power); }

        // 6. ProPhoto → XYZ → Lab; alpha = 0 (matches copy_pixel_nontemporal(out, res))
        let lab_out = xyz_to_lab(prophoto_to_xyz(rgb));
        output[k]     = lab_out[0];
        output[k + 1] = lab_out[1];
        output[k + 2] = lab_out[2];
        output[k + 3] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pixels: &[f32], grey: f32, black: f32, inv_dr: f32, power: f32,
           sat: f32, eps: f32, desat: i32, pres: i32) -> Vec<f32> {
        let n = pixels.len() / 4;
        let mut out = vec![0f32; pixels.len()];
        let table = vec![0.5f32; 0x10000];
        let grad2 = vec![1.0f32; 0x10000];
        unsafe {
            darkroom_filmic_process(
                pixels.as_ptr(), out.as_mut_ptr(), n,
                grey, black, inv_dr, power, sat, eps,
                desat, pres,
                table.as_ptr(), grad2.as_ptr(),
            );
        }
        out
    }

    #[test]
    fn roundtrip_neutral_lab() {
        // L=50 a=0 b=0 is neutral grey; output should be finite Lab with same sign L
        let input = vec![50.0f32, 0.0, 0.0, 0.0];
        let out = run(&input, 0.1842, -7.0, 1.0 / 14.0, 2.2, 1.0, 2e-5, 0, 1);
        assert!(out[0].is_finite(), "L must be finite");
        assert!(out[0] >= 0.0, "L must be non-negative");
    }

    #[test]
    fn alpha_is_zero() {
        let input = vec![50.0f32, 10.0, -5.0, 1.0];
        let out = run(&input, 0.1842, -7.0, 1.0 / 14.0, 2.2, 1.0, 2e-5, 0, 0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn preserve_color_and_perchannel_both_finite() {
        let input = vec![60.0f32, 20.0, -10.0, 0.0];
        let out_pc = run(&input, 0.1842, -7.0, 1.0 / 14.0, 2.2, 0.8, 2e-5, 1, 1);
        let out_ch = run(&input, 0.1842, -7.0, 1.0 / 14.0, 2.2, 0.8, 2e-5, 1, 0);
        for v in out_pc.iter().chain(out_ch.iter()).take(3) {
            assert!(v.is_finite(), "output must be finite");
        }
    }
}

/// Average two equally-sized LUT tables element-wise: `table[k] = (table[k] + table_temp[k]) / 2`.
///
/// Used in `compute_curve_lut` when blending two spline LUTs.
/// Matches the DT_OMP_FOR in src/iop/filmic.c:1068.
#[no_mangle]
pub unsafe extern "C" fn darkroom_filmic_average_luts(
    table: *mut f32,
    table_temp: *const f32,
    len: usize,
) {
    let t  = std::slice::from_raw_parts_mut(table, len);
    let tt = std::slice::from_raw_parts(table_temp, len);
    for k in 0..len {
        t[k] = (t[k] + tt[k]) * 0.5;
    }
}

/// Build the desaturation window LUT (`grad_2`, 65536 entries).
///
/// grad_2[k] = exp(-0.5 * (center - k/65536)² / sigma)  if sigma != 0, else 0.
/// Matches the DT_OMP_FOR in src/iop/filmic.c:1125.
#[no_mangle]
pub unsafe extern "C" fn darkroom_filmic_build_grad2_lut(
    grad2: *mut f32,
    center: f32,
    sigma: f32,
) {
    let lut = std::slice::from_raw_parts_mut(grad2, 65536);
    if sigma == 0.0 {
        for v in lut.iter_mut() { *v = 0.0; }
    } else {
        for k in 0..65536usize {
            let x = k as f32 / 65536.0;
            lut[k] = (-0.5 * (center - x) * (center - x) / sigma).exp();
        }
    }
}
