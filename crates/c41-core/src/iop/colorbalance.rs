use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Colorbalance;

impl IopProcess for Colorbalance {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colorbalance" }
}

// Mode constants matching dt_iop_colorbalance_mode_t in colorbalance.c
const LEGACY:           i32 = 0;
const LIFT_GAMMA_GAIN:  i32 = 1;
const SLOPE_OFFSET_POWER: i32 = 2;

// ── Color conversion helpers ────────────────────────────────────────────────

#[inline(always)]
fn lab_f_inv(x: f32) -> f32 {
    const EPS: f32 = 0.206_896_551_724_137_96;
    const KAPPA: f32 = 24389.0 / 27.0;
    if x > EPS { x * x * x } else { (116.0 * x - 16.0) / KAPPA }
}

#[inline(always)]
fn lab_to_xyz(l: f32, a: f32, b: f32) -> [f32; 3] {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    [0.9642 * lab_f_inv(fx), lab_f_inv(fy), 0.8249 * lab_f_inv(fz)]
}

#[inline(always)]
fn xyz_to_prophoto(xyz: [f32; 3]) -> [f32; 3] {
    [
         1.345_943_3 * xyz[0] - 0.255_607_5 * xyz[1] - 0.051_111_8 * xyz[2],
        -0.544_598_9 * xyz[0] + 1.508_167_3 * xyz[1] + 0.020_535_1 * xyz[2],
                                                         1.211_812_8 * xyz[2],
    ]
}

#[inline(always)]
fn prophoto_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    [
        0.797_674_9 * rgb[0] + 0.135_191_7 * rgb[1] + 0.031_353_4 * rgb[2],
        0.288_040_2 * rgb[0] + 0.711_874_1 * rgb[1] + 0.000_085_7 * rgb[2],
                                                        0.825_210_0 * rgb[2],
    ]
}

#[inline(always)]
fn prophoto_luma(rgb: [f32; 3]) -> f32 {
    0.288_040_2 * rgb[0] + 0.711_874_1 * rgb[1] + 0.000_085_7 * rgb[2]
}

/// XYZ (D50) → linear sRGB — transposed matrix from colorspaces_inline_conversions.h
#[inline(always)]
fn xyz_to_linear_srgb(xyz: [f32; 3]) -> [f32; 3] {
    [
         3.133_856_1 * xyz[0] - 1.616_866_7 * xyz[1] - 0.490_614_6 * xyz[2],
        -0.978_768_4 * xyz[0] + 1.916_141_5 * xyz[1] + 0.033_454_0 * xyz[2],
         0.071_945_3 * xyz[0] - 0.228_991_4 * xyz[1] + 1.405_242_7 * xyz[2],
    ]
}

/// linear sRGB → XYZ (D50) — transposed matrix
#[inline(always)]
fn linear_srgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    [
        0.436_074_7 * rgb[0] + 0.385_064_9 * rgb[1] + 0.143_080_4 * rgb[2],
        0.222_504_5 * rgb[0] + 0.716_878_6 * rgb[1] + 0.060_616_9 * rgb[2],
        0.013_932_2 * rgb[0] + 0.097_104_5 * rgb[1] + 0.714_173_3 * rgb[2],
    ]
}

/// gamma sRGB → linear sRGB
#[inline(always)]
fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}

/// linear sRGB → gamma sRGB
#[inline(always)]
fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn lab_f(x: f32) -> f32 {
    const EPS: f32 = 216.0 / 24389.0;
    const KAPPA: f32 = 24389.0 / 27.0;
    if x > EPS { x.cbrt() } else { (KAPPA * x + 16.0) / 116.0 }
}

fn xyz_to_lab(xyz: [f32; 3]) -> [f32; 3] {
    const D50_INV: [f32; 3] = [1.0 / 0.9642, 1.0, 1.0 / 0.8249];
    let f0 = lab_f(xyz[0] * D50_INV[0]);
    let f1 = lab_f(xyz[1] * D50_INV[1]);
    let f2 = lab_f(xyz[2] * D50_INV[2]);
    [116.0 * f1 - 16.0, 500.0 * (f0 - f1), -200.0 * (f2 - f1)]
}

// ── Per-mode pixel helpers ──────────────────────────────────────────────────

#[inline(always)]
fn process_legacy_pixel(
    l: f32, a: f32, b: f32,
    lift: &[f32], gamma_inv: &[f32], gain: &[f32],
) -> [f32; 3] {
    let xyz = lab_to_xyz(l, a, b);
    let lin = xyz_to_linear_srgb(xyz);
    // linear → gamma sRGB
    let mut rgb = [linear_to_srgb(lin[0]), linear_to_srgb(lin[1]), linear_to_srgb(lin[2])];
    // lift gamma gain
    for c in 0..3 {
        rgb[c] = ((rgb[c] - 1.0) * lift[c] + 1.0) * gain[c];
        rgb[c] = rgb[c].max(0.0);
    }
    // apply gamma_inv power
    for c in 0..3 { if rgb[c] > 0.0 { rgb[c] = rgb[c].powf(gamma_inv[c]); } }
    // gamma sRGB → linear → XYZ → Lab
    let lin2 = [srgb_to_linear(rgb[0]), srgb_to_linear(rgb[1]), srgb_to_linear(rgb[2])];
    let xyz2 = linear_srgb_to_xyz(lin2);
    xyz_to_lab(xyz2)
}

#[inline(always)]
fn process_lgg_pixel(
    l: f32, a: f32, b: f32,
    lift: &[f32], gamma_inv: &[f32], gain: &[f32],
    grey: f32, saturation: f32, saturation_out: f32,
    contrast_power: &[f32],
) -> [f32; 3] {
    let xyz = lab_to_xyz(l, a, b);
    let mut rgb = xyz_to_prophoto(xyz);

    // optional input saturation
    if (saturation - 1.0).abs() > 1e-6 {
        let luma = xyz[1]; // XYZ Y
        for c in 0..3 { rgb[c] = luma + saturation * (rgb[c] - luma); }
    }

    // RGB gamma 1/2.2 pre-correction, then lift+gain, then gamma_inv
    for c in 0..3 { rgb[c] = rgb[c].max(0.0).powf(1.0 / 2.2); }
    for c in 0..3 { rgb[c] = ((rgb[c] - 1.0) * lift[c] + 1.0) * gain[c]; }
    for c in 0..3 { rgb[c] = rgb[c].max(0.0); }
    for c in 0..3 { if rgb[c] > 0.0 { rgb[c] = rgb[c].powf(gamma_inv[c]); } }

    // optional output saturation
    if (saturation_out - 1.0).abs() > 1e-6 {
        let luma = prophoto_luma(rgb);
        for c in 0..3 { rgb[c] = luma + saturation_out * (rgb[c] - luma); }
    }

    // optional fulcrum contrast
    if (contrast_power[0] - 1.0).abs() > 1e-6 {
        for c in 0..3 { rgb[c] = rgb[c].max(0.0); }
        for c in 0..3 { rgb[c] = (rgb[c] / grey).powf(contrast_power[c]) * grey; }
    }

    let lab = xyz_to_lab(prophoto_to_xyz(rgb));
    lab
}

#[inline(always)]
fn process_sop_pixel(
    l: f32, a: f32, b: f32,
    lift: &[f32], gamma: &[f32], gain: &[f32],
    grey: f32, saturation: f32, saturation_out: f32,
    contrast_power: &[f32],
) -> [f32; 3] {
    let xyz = lab_to_xyz(l, a, b);
    let mut rgb = xyz_to_prophoto(xyz);

    // optional input saturation
    if (saturation - 1.0).abs() > 1e-6 {
        let luma = xyz[1];
        for c in 0..3 { rgb[c] = luma + saturation * (rgb[c] - luma); }
    }

    // CDL: slope*x + offset, clip, power
    for c in 0..3 { rgb[c] = rgb[c] * gain[c] + lift[c]; }
    for c in 0..3 {
        rgb[c] = rgb[c].max(0.0);
        if rgb[c] > 0.0 { rgb[c] = rgb[c].powf(gamma[c]); }
    }

    // optional output saturation
    if (saturation_out - 1.0).abs() > 1e-6 {
        let luma = prophoto_luma(rgb);
        for c in 0..3 { rgb[c] = luma + saturation_out * (rgb[c] - luma); }
    }

    // optional fulcrum contrast
    if (contrast_power[0] - 1.0).abs() > 1e-6 {
        for c in 0..3 { rgb[c] = rgb[c].max(0.0); }
        for c in 0..3 { rgb[c] = (rgb[c] / grey).powf(contrast_power[c]) * grey; }
    }

    xyz_to_lab(prophoto_to_xyz(rgb))
}

/// Color Balance IOP pixel loop.
///
/// Replaces the DT_OMP_FOR block in src/iop/colorbalance.c::process().
///
/// mode: 0=LEGACY, 1=LIFT_GAMMA_GAIN, 2=SLOPE_OFFSET_POWER
///
/// param1[4]: lift (LEGACY/LGG) or lift_sop (SOP) — pre-computed in process()
/// param2[4]: gamma_inv_legacy/gamma_inv_lgg (LEGACY/LGG) or gamma_sop (SOP)
/// gain[4]:   pre-computed gain vector
/// contrast_power[4]: { contrast, contrast, contrast, contrast } where
///   contrast = (d->contrast != 0) ? 1/d->contrast : 1e6
///
/// grey = d->grey / 100.0f; saturation = d->saturation; saturation_out = d->saturation_out
/// (grey/saturation/saturation_out are ignored in LEGACY mode)
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorbalance_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mode: i32,
    param1: *const f32,
    param2: *const f32,
    gain: *const f32,
    grey: f32,
    saturation: f32,
    saturation_out: f32,
    contrast_power: *const f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let p1 = std::slice::from_raw_parts(param1, 4);
    let p2 = std::slice::from_raw_parts(param2, 4);
    let gn = std::slice::from_raw_parts(gain,   4);
    let cp = std::slice::from_raw_parts(contrast_power, 4);

    for k in (0..npixels * 4).step_by(4) {
        let l = input[k];
        let a = input[k + 1];
        let b = input[k + 2];
        let alpha = input[k + 3];

        let lab_out = match mode {
            LEGACY => process_legacy_pixel(l, a, b, p1, p2, gn),
            LIFT_GAMMA_GAIN => process_lgg_pixel(l, a, b, p1, p2, gn, grey, saturation, saturation_out, cp),
            SLOPE_OFFSET_POWER => process_sop_pixel(l, a, b, p1, p2, gn, grey, saturation, saturation_out, cp),
            _ => [l, a, b],
        };

        output[k]     = lab_out[0];
        output[k + 1] = lab_out[1];
        output[k + 2] = lab_out[2];
        output[k + 3] = alpha; // alpha passthrough (matches copy_pixel_nontemporal(out, res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_legacy(pixels: &[f32]) -> Vec<f32> {
        let n = pixels.len() / 4;
        let mut out = vec![0f32; pixels.len()];
        // identity: lift=2, gamma_inv=1, gain=1 → (x-1)*2+1 = 2x-1, powf(1) = 2x-1
        // Actually neutral lift=2-1*1=2... hmm, let's use "no-op" lift: lift[c]=2-(1*1)=1
        // lift[c] = 2 - lift_red*lift_factor = 2 - 1*1 = 1
        // ((rgb-1)*1 + 1)*1 = rgb → identity
        let lift = vec![1.0f32; 4];
        let gamma_inv = vec![1.0f32; 4];
        let gain = vec![1.0f32; 4];
        let cp = vec![1.0f32; 4];
        unsafe {
            darkroom_colorbalance_process(
                pixels.as_ptr(), out.as_mut_ptr(), n,
                LEGACY, lift.as_ptr(), gamma_inv.as_ptr(), gain.as_ptr(),
                0.18, 1.0, 1.0, cp.as_ptr(),
            );
        }
        out
    }

    #[test]
    fn legacy_output_finite() {
        let input = vec![50.0f32, 10.0, -5.0, 0.0];
        let out = run_legacy(&input);
        assert!(out[0].is_finite());
        assert!(out[1].is_finite());
        assert!(out[2].is_finite());
    }

    #[test]
    fn alpha_preserved() {
        let input = vec![50.0f32, 0.0, 0.0, 0.77];
        let out = run_legacy(&input);
        assert_eq!(out[3], 0.77);
    }

    #[test]
    fn lgg_output_finite() {
        let n = 1usize;
        let input = vec![60.0f32, 5.0, -10.0, 0.0];
        let mut out = vec![0f32; 4];
        let lift  = vec![1.0f32; 4];
        let ginv  = vec![1.0f32; 4];
        let gain  = vec![1.0f32; 4];
        let cp    = vec![1.0f32; 4];
        unsafe {
            darkroom_colorbalance_process(
                input.as_ptr(), out.as_mut_ptr(), n,
                LIFT_GAMMA_GAIN, lift.as_ptr(), ginv.as_ptr(), gain.as_ptr(),
                0.18, 1.0, 1.0, cp.as_ptr(),
            );
        }
        assert!(out[0].is_finite());
    }

    #[test]
    fn sop_output_finite() {
        let n = 1usize;
        let input = vec![50.0f32, 0.0, 0.0, 0.0];
        let mut out = vec![0f32; 4];
        // neutral SOP: slope=1, offset=0, power=1 → x
        let lift_sop = vec![0.0f32; 4]; // offset=0
        let gamma_sop = vec![1.0f32; 4]; // power=1
        let gain = vec![1.0f32; 4]; // slope=1
        let cp = vec![1.0f32; 4];
        unsafe {
            darkroom_colorbalance_process(
                input.as_ptr(), out.as_mut_ptr(), n,
                SLOPE_OFFSET_POWER, lift_sop.as_ptr(), gamma_sop.as_ptr(), gain.as_ptr(),
                0.18, 1.0, 1.0, cp.as_ptr(),
            );
        }
        assert!(out[0].is_finite());
    }

    #[test]
    fn srgb_roundtrip() {
        // Test gamma ↔ linear roundtrip
        let v = 0.5f32;
        let lin = srgb_to_linear(v);
        let back = linear_to_srgb(lin);
        assert!((v - back).abs() < 1e-5, "sRGB roundtrip: {v} → {lin} → {back}");
    }
}
