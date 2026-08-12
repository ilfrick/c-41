use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Globaltonemap;

impl IopProcess for Globaltonemap {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "globaltonemap" }
}

/// Global tonemap — Reinhard operator: L_out = 100 * (l / (1 + l)), a/b copied.
/// ch is piece->colors (stride in floats per pixel, normally 4).
#[no_mangle]
pub unsafe extern "C" fn darkroom_globaltonemap_reinhard(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    ch: usize,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * ch);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * ch);
    for k in 0..npixels {
        let l = input[ch * k] / 100.0;
        output[ch * k]     = 100.0 * (l / (1.0 + l));
        output[ch * k + 1] = input[ch * k + 1];
        output[ch * k + 2] = input[ch * k + 2];
    }
}

/// Global tonemap — filmic (Hable) operator.
/// L_out = 100 * ((x*(6.2*x+0.5)) / (x*(6.2*x+1.7)+0.06)) where x = max(0, L/100 - 0.004).
/// a/b copied, alpha not written (matches C behaviour — only L/a/b accessed).
#[no_mangle]
pub unsafe extern "C" fn darkroom_globaltonemap_filmic(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    ch: usize,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * ch);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * ch);
    for k in 0..npixels {
        let l = input[ch * k] / 100.0;
        let x = (l - 0.004_f32).max(0.0);
        output[ch * k]     = 100.0 * ((x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06));
        output[ch * k + 1] = input[ch * k + 1];
        output[ch * k + 2] = input[ch * k + 2];
    }
}

/// Compute the maximum luminance (first channel * 0.01) across all pixels.
///
/// For each pixel k: lwmax = max(lwmax, in[k*ch] * 0.01)
/// `initial` is the starting value (typically -FLT_MAX or a cached previous max).
///
/// Matches the `DT_OMP_FOR(reduction(max : lwmax))` in globaltonemap.c:221.
#[no_mangle]
pub unsafe extern "C" fn darkroom_globaltonemap_luma_max(
    in_buf: *const f32,
    npixels: usize,
    ch: usize,
    initial: f32,
) -> f32 {
    if npixels == 0 || ch == 0 { return initial; }
    let inp = std::slice::from_raw_parts(in_buf, npixels * ch);
    let mut max_val = initial;
    for k in 0..npixels {
        let l = inp[ch * k] * 0.01;
        if l > max_val { max_val = l; }
    }
    max_val
}

/// Global tonemap — Drago operator (pre-computed ldc/bl/eps from caller).
///
/// ldc = data->drago.max_light * 0.01 / log10f(lwmax + 1)
/// bl  = logf(max(eps, data->drago.bias)) / logf(0.5)
/// eps = 0.0001 (constant in C)
/// L_out = 100 * ldc * log(max(eps, Lw+1)) / log(max(eps, 2 + (Lw/lwmax)^bl * 8))
/// where Lw = L_in * 0.01.
#[no_mangle]
pub unsafe extern "C" fn darkroom_globaltonemap_drago(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    ch: usize,
    ldc: f32,
    bl: f32,
    lwmax: f32,
    eps: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * ch);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * ch);
    for k in 0..npixels {
        let lw = input[ch * k] * 0.01;
        let numer = ldc * (lw + 1.0_f32).max(eps).ln();
        let denom = (2.0 + (lw / lwmax).powf(bl) * 8.0_f32).max(eps).ln();
        output[ch * k]     = 100.0 * numer / denom;
        output[ch * k + 1] = input[ch * k + 1];
        output[ch * k + 2] = input[ch * k + 2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_reinhard(pixels: &[f32], ch: usize) -> Vec<f32> {
        let n = pixels.len() / ch;
        let mut out = pixels.to_vec();
        unsafe { darkroom_globaltonemap_reinhard(pixels.as_ptr(), out.as_mut_ptr(), n, ch); }
        out
    }

    fn run_filmic(pixels: &[f32], ch: usize) -> Vec<f32> {
        let n = pixels.len() / ch;
        let mut out = pixels.to_vec();
        unsafe { darkroom_globaltonemap_filmic(pixels.as_ptr(), out.as_mut_ptr(), n, ch); }
        out
    }

    fn run_drago(pixels: &[f32], ch: usize, ldc: f32, bl: f32, lwmax: f32, eps: f32) -> Vec<f32> {
        let n = pixels.len() / ch;
        let mut out = pixels.to_vec();
        unsafe { darkroom_globaltonemap_drago(pixels.as_ptr(), out.as_mut_ptr(), n, ch, ldc, bl, lwmax, eps); }
        out
    }

    #[test]
    fn reinhard_zero_l() {
        let pix = vec![0.0f32, 5.0, -3.0, 1.0];
        let out = run_reinhard(&pix, 4);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 5.0);
        assert_eq!(out[2], -3.0);
    }

    #[test]
    fn reinhard_known_value() {
        // L=100 → l=1 → 100*(1/2) = 50
        let pix = vec![100.0f32, 0.0, 0.0, 0.0];
        let out = run_reinhard(&pix, 4);
        assert!((out[0] - 50.0).abs() < 1e-4, "L={}", out[0]);
    }

    #[test]
    fn filmic_below_cutoff_is_zero() {
        // L=0.4 → l=0.004 → x=0 → out=0
        let pix = vec![0.4f32, 1.0, 2.0, 0.5];
        let out = run_filmic(&pix, 4);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 2.0);
    }

    #[test]
    fn luma_max_finds_max_first_channel() {
        let inp = vec![100.0_f32, 0.0, 0.0, 1.0,   // L=100 → 1.0 after *0.01
                        50.0_f32, 0.0, 0.0, 1.0];
        let max = unsafe {
            darkroom_globaltonemap_luma_max(inp.as_ptr(), 2, 4, -f32::MAX)
        };
        assert!((max - 1.0).abs() < 1e-5);
    }

    #[test]
    fn luma_max_respects_initial_value() {
        let inp = vec![1.0_f32, 0.0, 0.0, 1.0];
        let max = unsafe {
            darkroom_globaltonemap_luma_max(inp.as_ptr(), 1, 4, 999.0)
        };
        // initial value 999 > 0.01 → max stays 999
        assert!((max - 999.0).abs() < 1e-5);
    }

    #[test]
    fn drago_ab_passthrough() {
        let pix = vec![50.0f32, 10.0, -5.0, 1.0];
        let out = run_drago(&pix, 4, 1.0, 1.0, 1.0, 0.0001);
        assert_eq!(out[1], 10.0);
        assert_eq!(out[2], -5.0);
    }
}
