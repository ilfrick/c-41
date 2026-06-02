use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct ColorZones;

impl IopProcess for ColorZones {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colorzones" }
}

const DT_IOP_COLORZONES_LUT_RES: usize = 0x10000;
const DT_2PI: f32 = std::f32::consts::TAU;

/// Linear interpolation in a 65536-entry LUT, index in [0,1].
#[inline(always)]
fn lut_lookup(lut: &[f32], i: f32) -> f32 {
    let bin0 = ((DT_IOP_COLORZONES_LUT_RES as f32 * i) as usize).clamp(0, 0xffff);
    let bin1 = (bin0 + 1).min(0xffff);
    let f = DT_IOP_COLORZONES_LUT_RES as f32 * i - bin0 as f32;
    lut[bin1] * f + lut[bin0] * (1.0 - f)
}

/// Lab → LCH: L unchanged, C = hypot(a,b), h = atan2(b,a)/(2π) in [0,1).
#[inline(always)]
fn lab_to_lch(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let var_h = b.atan2(a);
    let h = if var_h > 0.0 {
        var_h / DT_2PI
    } else {
        1.0 - var_h.abs() / DT_2PI
    };
    (l, a.hypot(b), h)
}

/// LCH → Lab: L unchanged, a = C*cos(h*2π), b = C*sin(h*2π).
#[inline(always)]
fn lch_to_lab(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    let (sin_h, cos_h) = (DT_2PI * h).sin_cos();
    (l, cos_h * c, sin_h * c)
}

/// Color-zones IOP — luminance/chroma/hue equalizer in LCH space.
///
/// mode: 0 = v3 smooth (DT_IOP_COLORZONES_MODE_SMOOTH), non-zero = v1 legacy
/// channel: 0 = L, 1 = C, 2 = h (which dimension drives selection)
/// lut_l/a/b: each 65536 floats (d->lut[0..2]).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorzones_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mode: i32,    // 0 = smooth/v3 (DT_IOP_COLORZONES_MODE_SMOOTH), non-zero = flat/v1
    channel: i32, // 0=L, 1=C, 2=h
    lut_l: *const f32,
    lut_a: *const f32,
    lut_b: *const f32,
) {
    const NORMALIZE_C: f32 = 1.0 / (128.0 * std::f32::consts::SQRT_2);

    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let ll = std::slice::from_raw_parts(lut_l, DT_IOP_COLORZONES_LUT_RES);
    let la = std::slice::from_raw_parts(lut_a, DT_IOP_COLORZONES_LUT_RES);
    let lb = std::slice::from_raw_parts(lut_b, DT_IOP_COLORZONES_LUT_RES);

    for px in 0..npixels {
        let base = px * 4;
        let i = &inp[base..base + 4];
        let o = &mut out[base..base + 4];
        let (in_l, in_a, in_b) = (i[0], i[1], i[2]);

        if mode != 0 {
            // v1: legacy flat mode (DT_IOP_COLORZONES_MODE_FLAT)
            let (l, c, h) = lab_to_lch(in_l, in_a, in_b);
            let select = (match channel {
                0 => l * 0.01,
                1 => c * NORMALIZE_C,
                _ => h,
            }).clamp(0.0, 1.0);

            let out_l = l * 2.0f32.powf(4.0 * (lut_lookup(ll, select) - 0.5));
            let out_c = c * 2.0 * lut_lookup(la, select);
            let out_h = h + lut_lookup(lb, select) - 0.5;
            let (rl, ra, rb) = lch_to_lab(out_l, out_c, out_h);
            o[0] = rl;
            o[1] = ra;
            o[2] = rb;
        } else {
            // v3: smooth mode (DT_IOP_COLORZONES_MODE_SMOOTH = 0) — edit in a/b space directly
            let a = in_a;
            let b = in_b;
            let h = (b.atan2(a) + DT_2PI).rem_euclid(DT_2PI) / DT_2PI;
            let c = (b * b + a * a).sqrt();
            let (select, blend) = match channel {
                0 => (((in_l / 100.0).min(1.0)), 0.0f32),
                1 => ((c / 128.0).min(1.0), 0.0f32),
                _ => (h, (1.0 - c / 128.0) * (1.0 - c / 128.0)),
            };
            let lm = (blend * 0.5 + (1.0 - blend) * lut_lookup(ll, select)) - 0.5;
            let hm = (blend * 0.5 + (1.0 - blend) * lut_lookup(lb, select)) - 0.5;
            let cm = 2.0 * lut_lookup(la, select);
            let out_l = in_l * 2.0f32.powf(4.0 * lm);
            o[0] = out_l;
            let new_h = h + hm;
            let (sin_h, cos_h) = (DT_2PI * new_h).sin_cos();
            o[1] = cos_h * cm * c;
            o[2] = sin_h * cm * c;
        }
        o[3] = i[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half_lut() -> Vec<f32> {
        vec![0.5f32; DT_IOP_COLORZONES_LUT_RES]
    }

    #[test]
    fn v1_flat_lut_zeroes_lch_offsets() {
        // lut_a = 0.5 → C *= 2*0.5 = 1 (no change)
        // lut_l = 0.5 → L *= 2^(4*(0.5-0.5)) = 1 (no change)
        // lut_b = 0.5 → h += 0.5 - 0.5 = 0 (no change)
        let half = half_lut();
        let inp = [50.0f32, 30.0, 10.0, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_colorzones_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1, 2, // v1 (non-zero = legacy), h channel
                half.as_ptr(), half.as_ptr(), half.as_ptr(),
            )
        };
        assert!((out[0] - 50.0).abs() < 0.5);
        assert!((out[1] - 30.0).abs() < 0.5);
        assert!((out[2] - 10.0).abs() < 0.5);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn v3_flat_lut_neutral() {
        // lut_l = 0.5 → Lm = 0 → L *= 2^0 = 1
        // lut_a = 0.5 → Cm = 1 → chroma scaled by 1
        // lut_b = 0.5 → hm = 0 → hue unchanged
        let half = half_lut();
        let inp = [60.0f32, 20.0, 10.0, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_colorzones_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                0, 2, // v3/smooth (mode=0), h channel
                half.as_ptr(), half.as_ptr(), half.as_ptr(),
            )
        };
        assert!((out[0] - 60.0).abs() < 0.5);
        // a/b may shift slightly due to float precision but should be close
        let c_in  = (inp[1]*inp[1] + inp[2]*inp[2]).sqrt();
        let c_out = (out[1]*out[1] + out[2]*out[2]).sqrt();
        assert!((c_out - c_in).abs() < 0.5);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn alpha_passes_through() {
        let half = half_lut();
        let inp = [50.0f32, 0.0, 0.0, 0.42];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_colorzones_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1, 0, // v1 (non-zero = legacy), L channel
                half.as_ptr(), half.as_ptr(), half.as_ptr(),
            )
        };
        assert_eq!(out[3], 0.42);
    }
}

#[inline(always)]
fn lab_2_lch(lab: &[f32]) -> [f32; 3] {
    let h_raw = lab[2].atan2(lab[1]);
    let h = if h_raw > 0.0 {
        h_raw / std::f32::consts::TAU
    } else {
        1.0 - h_raw.abs() / std::f32::consts::TAU
    };
    [lab[0], lab[1].hypot(lab[2]), h]
}

/// Mask-display pass for colorzones process_display().
///
/// For each pixel: Lab → LCh, select L/C/h, look up the display-channel LUT,
/// write mask alpha into out[3]. Input and output are RGBA (4 floats/pixel).
/// The caller copies ivoid → ovoid first; this function only writes alpha.
///
/// `channel`: 0=L, 1=C, 2=h (DT_IOP_COLORZONES_{L,C,h} enum values).
/// `lut`: pointer to `d->lut[display_channel]` — exactly 65536 floats.
///
/// Matches the DT_OMP_FOR in src/iop/colorzones.c:444.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorzones_display(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    channel: i32,
    lut: *const f32,
) {
    if npixels == 0 { return; }
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lut_s = std::slice::from_raw_parts(lut, DT_IOP_COLORZONES_LUT_RES);
    const NORM_C: f32 = 1.0 / (128.0 * std::f32::consts::SQRT_2);

    for k in 0..npixels {
        let px = &input[k * 4..k * 4 + 4];
        let lch = lab_2_lch(px);
        let select = match channel {
            0 => lch[0] * 0.01,
            1 => lch[1] * NORM_C,
            _ => lch[2],
        };
        let select = select.clamp(0.0, 1.0);
        let alpha = (lut_lookup(lut_s, select) - 0.5).abs() * 4.0;
        output[k * 4 + 3] = alpha.clamp(0.0, 1.0);
    }
}
