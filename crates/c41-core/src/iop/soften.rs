use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Soften;

impl IopProcess for Soften {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "soften" }
}

/// Converts linear RGB to HSL.  Ports darktable rgb2hsl() from colorspaces.h.
#[inline(always)]
fn rgb2hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    const EPS: f32 = 1.525_878_906_25e-5_f32;
    let pmax = r.max(g).max(b);
    let pmin = r.min(g).min(b);
    let delta = pmax - pmin;
    let l = (pmin + pmax) / 2.0;

    if delta == 0.0 {
        return (0.0, 0.0, l);
    }

    let s = if l < 0.5 {
        delta / (pmax + pmin).max(EPS)
    } else {
        delta / (2.0 - pmax - pmin).max(EPS)
    };

    let mut h = if pmax == r {
        (g - b) / delta
    } else if pmax == g {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };
    h /= 6.0;
    if h < 0.0 { h += 1.0; } else if h > 1.0 { h -= 1.0; }

    (h, s, l)
}

/// Converts one HSL channel to RGB.  Ports darktable hue2rgb() — hue is pre-scaled to [0, 6).
#[inline(always)]
fn hue2rgb(m1: f32, m2: f32, hue: f32) -> f32 {
    if hue < 1.0 { m1 + (m2 - m1) * hue }
    else if hue < 3.0 { m2 }
    else if hue < 4.0 { m1 + (m2 - m1) * (4.0 - hue) }
    else { m1 }
}

/// Converts HSL back to RGB.  Ports darktable hsl2rgb() from colorspaces.h.
#[inline(always)]
fn hsl2rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    let m2 = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let m1 = 2.0 * l - m2;
    let h6 = h * 6.0;
    let r = hue2rgb(m1, m2, if h6 < 4.0 { h6 + 2.0 } else { h6 - 4.0 });
    let g = hue2rgb(m1, m2, h6);
    let b = hue2rgb(m1, m2, if h6 > 2.0 { h6 - 2.0 } else { h6 + 4.0 });
    (r, g, b)
}

/// Soften IOP initial pixel loop.
///
/// Converts each pixel to HSL, scales saturation and lightness, writes back RGB.
/// Matches the DT_OMP_FOR loop in src/iop/soften.c::process() (before the box-mean blur).
///
/// `brightness` = 1.0 / exp2f(-d->brightness)
/// `saturation` = d->saturation / 100.0
#[no_mangle]
pub unsafe extern "C" fn darkroom_soften_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    brightness: f32,
    saturation: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    for k in (0..npixels * 4).step_by(4) {
        let (h, s, l) = rgb2hsl(input[k], input[k + 1], input[k + 2]);
        let s = (s * saturation).clamp(0.0, 1.0);
        let l = (l * brightness).clamp(0.0, 1.0);
        let (r, g, b) = hsl2rgb(h, s, l);
        output[k]     = r;
        output[k + 1] = g;
        output[k + 2] = b;
        output[k + 3] = 0.0; // hsl2rgb sets alpha=0 in C
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pixels: &[f32], brightness: f32, saturation: f32) -> Vec<f32> {
        let n = pixels.len() / 4;
        let mut out = vec![0f32; pixels.len()];
        unsafe { darkroom_soften_process(pixels.as_ptr(), out.as_mut_ptr(), n, brightness, saturation); }
        out
    }

    #[test]
    fn grey_pixel_stays_grey() {
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let out = run(&input, 1.0, 1.0);
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
        assert!((out[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn zero_saturation_produces_grey() {
        let input = vec![0.8, 0.3, 0.1, 1.0];
        let out = run(&input, 1.0, 0.0);
        // s=0 → all channels equal to L
        assert!((out[0] - out[1]).abs() < 1e-5);
        assert!((out[1] - out[2]).abs() < 1e-5);
    }

    #[test]
    fn alpha_is_zero() {
        let input = vec![0.6, 0.4, 0.2, 0.9];
        let out = run(&input, 1.0, 1.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn roundtrip_hsl() {
        let r = 0.7_f32;
        let g = 0.3_f32;
        let b = 0.5_f32;
        let (h, s, l) = rgb2hsl(r, g, b);
        let (rr, gg, bb) = hsl2rgb(h, s, l);
        assert!((r - rr).abs() < 1e-5, "R roundtrip");
        assert!((g - gg).abs() < 1e-5, "G roundtrip");
        assert!((b - bb).abs() < 1e-5, "B roundtrip");
    }
}
