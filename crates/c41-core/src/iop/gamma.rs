use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Gamma;

impl IopProcess for Gamma {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "gamma" }
}

// "Yellow" overlay colour used for all channel-display modes.
const MASK_COLOR: [f32; 3] = [1.0, 1.0, 0.0];

/// Standard sRGB gamma transfer function.
/// Matches the linearisation/gamma in `_write_pixel()` (gamma.c:72-75).
#[inline(always)]
fn srgb_gamma(v: f32) -> f32 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Write a 3-channel linear RGB pixel to a BGR uint8 output slot with
/// sRGB gamma + alpha blending with `MASK_COLOR`.
///
/// Matches `_write_pixel()` in gamma.c:63. Output layout is BGR, so
/// `out[base + 2 - c]` for c in 0..3 (R→slot 2, G→slot 1, B→slot 0).
#[inline(always)]
fn write_pixel(pixel: [f32; 3], alpha: f32, out: &mut [u8], base: usize) {
    let one_minus_alpha = 1.0 - alpha;
    for c in 0..3 {
        let g = srgb_gamma(pixel[c]);
        let value = (255.0 * (g * one_minus_alpha + MASK_COLOR[c] * alpha)).round();
        out[base + 2 - c] = value.clamp(0.0, 255.0) as u8;
    }
}

/// Convert a linear float RGBA buffer to uint8 BGR (no sRGB gamma).
///
/// For each pixel:
///   out[j + 2 - c] = clamp(round(255 * max(in[j + c], 0)), 0, 255)  for c in 0..3
///
/// Note: alpha channel (index 3) is NOT written. Output is tightly packed 4-byte
/// pixels with BGR bytes in slots 0-2 (slot 3 is not touched).
///
/// **`buffsize` is the f32 element count = `width * height * 4`, NOT a byte count.**
/// This equals the u8 output byte count only because RGBA has 4 channels and each
/// u8 output pixel occupies 4 bytes. Passing the true byte count (× sizeof(float))
/// would over-size the input slice and is UB.
///
/// Matches `_copy_output()` in src/iop/gamma.c:269.
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_copy_output(
    in_buf: *const f32,
    out_buf: *mut u8,
    buffsize: usize, // f32 element count = width * height * 4
) {
    if buffsize == 0 { return; }
    debug_assert!(buffsize % 4 == 0, "buffsize must be width*height*4 f32 elements");
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);

    let mut j = 0;
    while j < buffsize {
        for c in 0..3 {
            let v = inp[j + c].max(0.0);
            let byte = (255.0 * v).round().clamp(0.0, 255.0) as u8;
            out[j + 2 - c] = byte;
        }
        j += 4;
    }
}

/// Monochrome channel display with sRGB gamma and mask overlay.
///
/// For each pixel: grey = in[j+1] (second channel) used for all 3 RGB slots.
/// Then `_write_pixel({grey, grey, grey}, MASK_COLOR, in[j+3]*alpha, out)`.
///
/// Matches `_channel_display_monochrome()` in src/iop/gamma.c:114.
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_display_monochrome(
    in_buf: *const f32,
    out_buf: *mut u8,
    buffsize: usize,
    alpha: f32,
) {
    if buffsize == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);

    let mut j = 0;
    while j < buffsize {
        let grey = inp[j + 1];
        write_pixel([grey, grey, grey], inp[j + 3] * alpha, out, j);
        j += 4;
    }
}

/// False-colour single-channel display for R, G, B, and saturation modes.
///
/// `mode` selects the pixel construction:
///   0 (R)          → {in[j+1], 0.0,   0.0,               0.0}
///   1 (G)          → {0.0,     in[j+1], 0.0,              0.0}
///   2 (B)          → {0.0,     0.0,   in[j+1],             0.0}
///   3 (saturation) → {0.5,     0.5*(1-in[j+1]), 0.5,       0.0}
///
/// Applies `_write_pixel` with `MASK_COLOR` and `alpha = in[j+3] * alpha`.
///
/// Matches the DT_OMP_FOR_SIMD loops for DISPLAY_R, G, B, LCH_C/HSL_S/Cz
/// in `_channel_display_false_color()` (gamma.c:163, 172, 180, 190).
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_display_false_color_simple(
    in_buf: *const f32,
    out_buf: *mut u8,
    buffsize: usize,
    alpha: f32,
    mode: u32,
) {
    if buffsize == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);

    let mut j = 0;
    while j < buffsize {
        let v = inp[j + 1];
        let pixel = match mode {
            0 => [v, 0.0, 0.0],
            1 => [0.0, v, 0.0],
            2 => [0.0, 0.0, v],
            _ => [0.5, 0.5 * (1.0 - v), 0.5], // saturation modes (mode 3+)
        };
        write_pixel(pixel, inp[j + 3] * alpha, out, j);
        j += 4;
    }
}

/// Luminance-mask display overlay.
///
/// `mix` comes from `dt_conf_get_float("darkroom/ui/develop_mask_mix")`.
/// `interpolatef(mix, in[j+3], luma) = mix * (in[j+3] - luma) + luma`
/// where `luma = 0.3*R + 0.59*G + 0.11*B`.
///
/// Then `_write_pixel({grey, grey, grey}, MASK_COLOR, in[j+3]*alpha, out)`.
///
/// Matches `_mask_display()` in src/iop/gamma.c:255.
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_mask_display(
    in_buf: *const f32,
    out_buf: *mut u8,
    buffsize: usize,
    alpha: f32,
    mix: f32,
) {
    if buffsize == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);

    let mut j = 0;
    while j < buffsize {
        let luma = 0.3 * inp[j] + 0.59 * inp[j + 1] + 0.11 * inp[j + 2];
        // interpolatef(a, b, c) = a * (b - c) + c
        let grey = mix * (inp[j + 3] - luma) + luma;
        write_pixel([grey, grey, grey], inp[j + 3] * alpha, out, j);
        j += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_output_bgr_swap() {
        // R=1.0, G=0.0, B=0.0 → out BGR: [0, 0, 255] (B=0 slot 0, G=0 slot 1, R=255 slot 2)
        let inp = vec![1.0_f32, 0.0, 0.0, 1.0];
        let mut out = vec![0_u8; 4];
        unsafe { darkroom_gamma_copy_output(inp.as_ptr(), out.as_mut_ptr(), 4); }
        assert_eq!(out[2], 255); // R in slot 2
        assert_eq!(out[1], 0);   // G in slot 1
        assert_eq!(out[0], 0);   // B in slot 0
    }

    #[test]
    fn copy_output_clamps_over_one() {
        let inp = vec![2.0_f32, 0.5, 0.0, 1.0];
        let mut out = vec![0_u8; 4];
        unsafe { darkroom_gamma_copy_output(inp.as_ptr(), out.as_mut_ptr(), 4); }
        assert_eq!(out[2], 255); // R clamped
    }

    #[test]
    fn copy_output_no_srgb_gamma() {
        // 0.5 linear → 128 (no gamma). With gamma it would be ~188.
        let inp = vec![0.0_f32, 0.0, 0.5, 1.0];
        let mut out = vec![0_u8; 4];
        unsafe { darkroom_gamma_copy_output(inp.as_ptr(), out.as_mut_ptr(), 4); }
        assert_eq!(out[0], 128); // B in slot 0
    }

    #[test]
    fn monochrome_uses_second_channel_for_grey() {
        // in[j+1] = 0.0; no mask (alpha=0) → grey goes through sRGB gamma
        // srgb_gamma(0) = 0 → all output bytes = 0
        let inp = vec![0.5_f32, 0.0, 0.3, 1.0];
        let mut out = vec![0xff_u8; 4];
        unsafe { darkroom_gamma_display_monochrome(inp.as_ptr(), out.as_mut_ptr(), 4, 0.0); }
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
    }

    #[test]
    fn false_color_r_puts_value_in_red_slot() {
        // mode 0 = R: pixel = {in[j+1], 0, 0} → after sRGB+mask, B and G ≈ 0 (from mask_color)
        // alpha=0 means no mask overlay; gamma(0)=0 so G=0, B=0; gamma(in[j+1]) in R slot.
        let inp = vec![0.0_f32, 0.5, 0.0, 1.0];
        let mut out = vec![0xff_u8; 4];
        unsafe { darkroom_gamma_display_false_color_simple(inp.as_ptr(), out.as_mut_ptr(), 4, 0.0, 0); }
        // G and B are 0, R = sRGB(0.5) ≈ 0.735 → ~188
        assert!(out[2] > 100, "R slot = {}", out[2]);
        assert_eq!(out[1], 0, "G slot");
        assert_eq!(out[0], 0, "B slot");
    }

    #[test]
    fn mask_display_grey_from_luma() {
        // mix=0: grey = luma = 0.3*R + 0.59*G + 0.11*B, alpha=0
        let inp = vec![1.0_f32, 0.0, 0.0, 0.5]; // R=1, G=0, B=0, alpha=0.5
        let mut out = vec![0_u8; 4];
        unsafe { darkroom_gamma_mask_display(inp.as_ptr(), out.as_mut_ptr(), 4, 0.0, 0.0); }
        // luma = 0.3*1 + 0 + 0 = 0.3; grey = 0.3; sRGB(0.3) ≈ 0.607 → ~155
        assert!(out[0] > 100 && out[0] < 200, "grey byte = {}", out[0]);
        // all three slots should be equal (grey)
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }
}

// ── Channel display helpers (gamma.c Phase 2z+64) ────────────────────────

#[inline(always)]
fn write_pixel_bgr(pixel: [f32; 4], out: &mut [u8], offset: usize, alpha: f32) {
    for c in 0..3usize {
        let v = srgb_gamma(pixel[c]) * (1.0 - alpha) + MASK_COLOR[c] * alpha;
        out[offset + 2 - c] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

/// Channel display: synthetic Lab from 'a' channel → XYZ → sRGB (with overlay).
/// Matches _channel_display_false_color DT_OMP_FOR_SIMD at gamma.c:128.
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_display_a_channel(
    in_buf: *const f32, out_buf: *mut u8, buffsize: usize, alpha: f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);
    for j in (0..buffsize).step_by(4) {
        let value = (inp[j+1]*256.0 - 128.0).clamp(-56.0, 56.0);
        let lab = [79.0 - value*(11.0/56.0), value, 0.0, 0.0];
        let xyz  = crate::color::lab_to_xyz(lab);
        let srgb = crate::color::xyz_d50_to_srgb(xyz);
        let px   = crate::color::normalize_color(srgb, 0.75);
        write_pixel_bgr(px, out, j, inp[j+3]*alpha);
    }
}

/// Channel display: synthetic Lab from 'b' channel → XYZ → sRGB.
/// Matches _channel_display_false_color DT_OMP_FOR_SIMD at gamma.c:143.
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_display_b_channel(
    in_buf: *const f32, out_buf: *mut u8, buffsize: usize, alpha: f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);
    for j in (0..buffsize).step_by(4) {
        let value = (inp[j+1]*256.0 - 128.0).clamp(-65.0, 65.0);
        let lab = [60.0 + value*(2.0/65.0), 0.0, value, 0.0];
        let xyz  = crate::color::lab_to_xyz(lab);
        let srgb = crate::color::xyz_d50_to_srgb(xyz);
        let px   = crate::color::normalize_color(srgb, 0.75);
        write_pixel_bgr(px, out, j, inp[j+3]*alpha);
    }
}

/// Channel display: LCH hue → Lab → XYZ → sRGB.
/// Matches _channel_display_false_color DT_OMP_FOR_SIMD at gamma.c:171.
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_display_lch_h(
    in_buf: *const f32, out_buf: *mut u8, buffsize: usize, alpha: f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);
    for j in (0..buffsize).step_by(4) {
        let lch  = [65.0f32, 37.0, inp[j+1], 0.0];
        let lab  = crate::color::lch_to_lab(lch);
        let xyz  = crate::color::lab_to_xyz(lab);
        let srgb = crate::color::xyz_d50_to_srgb(xyz);
        let px   = crate::color::normalize_color(srgb, 0.75);
        write_pixel_bgr(px, out, j, inp[j+3]*alpha);
    }
}

/// Channel display: HSL hue → RGB (normalized).
/// Matches _channel_display_false_color DT_OMP_FOR at gamma.c:184.
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_display_hsl_h(
    in_buf: *const f32, out_buf: *mut u8, buffsize: usize, alpha: f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);
    for j in (0..buffsize).step_by(4) {
        let (r,g,b,_) = crate::color::hsl2rgb(inp[j+1], 0.5, 0.5);
        let px = crate::color::normalize_color([r,g,b,0.0], 0.75);
        write_pixel_bgr(px, out, j, inp[j+3]*alpha);
    }
}

/// Channel display: JzCzhz hue → JzAzBz → XYZ D65 → sRGB.
/// Matches _channel_display_false_color DT_OMP_FOR at gamma.c:195.
#[no_mangle]
pub unsafe extern "C" fn darkroom_gamma_display_jz_hz(
    in_buf: *const f32, out_buf: *mut u8, buffsize: usize, alpha: f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, buffsize);
    let out = std::slice::from_raw_parts_mut(out_buf, buffsize);
    for j in (0..buffsize).step_by(4) {
        let jzczhz  = [0.011f32, 0.01, inp[j+1], 0.0];
        let jzazbz  = crate::color::jzczhz_to_jzazbz(jzczhz);
        let xyz_d65 = crate::color::jzazbz_to_xyz_d65(jzazbz);
        let srgb    = crate::color::xyz_d65_to_srgb(xyz_d65);
        let px      = crate::color::normalize_color(srgb, 0.75);
        write_pixel_bgr(px, out, j, inp[j+3]*alpha);
    }
}
