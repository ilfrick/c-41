use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Colormapping;

impl IopProcess for Colormapping {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colormapping" }
}

/// Scan an RGBA Lab image to find the minimum and maximum of the a and b
/// channels.
///
/// Returns (a_min, a_max, b_min, b_max) via four out-pointers. Matches the
/// `DT_OMP_FOR(reduction(min: a_min, b_min) reduction(max: a_max, b_max))`
/// loop in `kmeans()` at src/iop/colormapping.c:298.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colormapping_ab_range(
    col: *const f32,
    npixels: usize,
    out_a_min: *mut f32,
    out_a_max: *mut f32,
    out_b_min: *mut f32,
    out_b_max: *mut f32,
) {
    if npixels == 0 {
        *out_a_min = f32::MAX;
        *out_a_max = -f32::MAX;
        *out_b_min = f32::MAX;
        *out_b_max = -f32::MAX;
        return;
    }
    let pixels = std::slice::from_raw_parts(col, npixels * 4);

    let mut a_min = f32::MAX;
    let mut a_max = -f32::MAX;
    let mut b_min = f32::MAX;
    let mut b_max = -f32::MAX;

    for k in 0..npixels {
        let a = pixels[k * 4 + 1];
        let b = pixels[k * 4 + 2];
        if a < a_min { a_min = a; }
        if a > a_max { a_max = a; }
        if b < b_min { b_min = b; }
        if b > b_max { b_max = b; }
    }

    *out_a_min = a_min;
    *out_a_max = a_max;
    *out_b_min = b_min;
    *out_b_max = b_max;
}

/// Compute the blended/equalised L-delta for every pixel and store it into
/// the L channel of `out`.
///
/// For each pixel k:
///   L         = in[k*4]
///   bin       = clamp(HISTN * L / 100, 0, HISTN - 1)
///   hist_val  = source_ihist[target_hist[bin]]
///   out[k*4]  = clamp(0.5 * ((L * (1 - eq) + hist_val * eq) - L) + 50, 0, 100)
///
/// `target_hist` is an integer array of length `histn`; `source_ihist` is a
/// float array of length `histn`.
///
/// Matches the `DT_OMP_FOR()` loop in `process()` at
/// src/iop/colormapping.c:492.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colormapping_l_delta(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    target_hist:  *const i32,
    source_ihist: *const f32,
    histn: usize,
    equalization: f32,
) {
    if npixels == 0 || histn == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tgt  = std::slice::from_raw_parts(target_hist,  histn);
    let src  = std::slice::from_raw_parts(source_ihist, histn);
    // Keep `last` as usize throughout to avoid a `histn > i32::MAX` wrap.
    // The C version is immune because it uses the compile-time constant HISTN;
    // here histn is caller-supplied, so we stay in native usize arithmetic.
    let last = histn - 1;

    for k in 0..npixels {
        let j = k * 4;
        let l = inp[j];
        // Clamp the float index to [0, last] then convert directly to usize.
        let bin = ((histn as f32 * l / 100.0).clamp(0.0, last as f32) as usize).min(last);
        // target_hist values come from invert_histogram which normalises them
        // into [0, HISTN-1]. Guard defensively with .min(last).
        let tbin = (tgt[bin].max(0) as usize).min(last);
        let hist_val = src[tbin];
        let delta = 0.5 * ((l * (1.0 - equalization) + hist_val * equalization) - l) + 50.0;
        out[j] = delta.clamp(0.0, 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ab_range_finds_extremes() {
        // 3 pixels: a = {1, -2, 5}, b = {0, 3, -1}
        let pixels = vec![
            0.0, 1.0, 0.0, 1.0,
            0.0, -2.0, 3.0, 1.0,
            0.0, 5.0, -1.0, 1.0,
        ];
        let (mut a_min, mut a_max, mut b_min, mut b_max) = (0.0_f32, 0.0, 0.0, 0.0);
        unsafe {
            darkroom_colormapping_ab_range(
                pixels.as_ptr(), 3,
                &mut a_min, &mut a_max, &mut b_min, &mut b_max,
            );
        }
        assert_eq!(a_min, -2.0);
        assert_eq!(a_max,  5.0);
        assert_eq!(b_min, -1.0);
        assert_eq!(b_max,  3.0);
    }

    #[test]
    fn ab_range_single_pixel() {
        let pixels = vec![0.0, 7.0, -3.0, 1.0];
        let (mut a_min, mut a_max, mut b_min, mut b_max) = (0.0_f32, 0.0, 0.0, 0.0);
        unsafe {
            darkroom_colormapping_ab_range(
                pixels.as_ptr(), 1,
                &mut a_min, &mut a_max, &mut b_min, &mut b_max,
            );
        }
        assert_eq!(a_min, 7.0); assert_eq!(a_max, 7.0);
        assert_eq!(b_min, -3.0); assert_eq!(b_max, -3.0);
    }

    #[test]
    fn ab_range_zero_pixels_returns_extremal_sentinels() {
        let (mut a_min, mut a_max, mut b_min, mut b_max) = (0.0_f32, 0.0, 0.0, 0.0);
        unsafe {
            darkroom_colormapping_ab_range(
                std::ptr::null(), 0,
                &mut a_min, &mut a_max, &mut b_min, &mut b_max,
            );
        }
        assert_eq!(a_min, f32::MAX);
        assert_eq!(a_max, -f32::MAX);
    }

    const HISTN: usize = 2048;

    fn identity_hist() -> (Vec<i32>, Vec<f32>) {
        let tgt: Vec<i32> = (0..HISTN as i32).collect();
        let src: Vec<f32> = (0..HISTN).map(|i| 100.0 * i as f32 / HISTN as f32).collect();
        (tgt, src)
    }

    #[test]
    fn l_delta_equalization_zero_produces_fifty() {
        // eq=0: formula = 0.5 * ((L - L)) + 50 = 50 for any L
        let (tgt, src) = identity_hist();
        let inp = vec![60.0_f32, 0.0, 0.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_colormapping_l_delta(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tgt.as_ptr(), src.as_ptr(), HISTN, 0.0,
            );
        }
        assert!((out[0] - 50.0).abs() < 1e-5, "out[0]={}", out[0]);
    }

    #[test]
    fn l_delta_equalization_one_uses_hist_val() {
        // eq=1: formula = 0.5 * (source_ihist[bin] - L) + 50
        // With identity histogram source_ihist[bin] ≈ L → delta ≈ 50
        let (tgt, src) = identity_hist();
        let inp = vec![50.0_f32, 0.0, 0.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_colormapping_l_delta(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tgt.as_ptr(), src.as_ptr(), HISTN, 1.0,
            );
        }
        // identity: source_ihist ≈ L → 0.5*(L-L)+50 = 50
        assert!((out[0] - 50.0).abs() < 0.1, "out[0]={}", out[0]);
    }

    #[test]
    fn l_delta_does_not_touch_ab_alpha() {
        let (tgt, src) = identity_hist();
        let inp = vec![50.0_f32, -20.0, 30.0, 0.42];
        let mut out = vec![-7.0_f32; 4];
        unsafe {
            darkroom_colormapping_l_delta(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tgt.as_ptr(), src.as_ptr(), HISTN, 0.5,
            );
        }
        // Only index 0 should change
        assert_ne!(out[0], -7.0);
        assert_eq!(out[1], -7.0);
        assert_eq!(out[2], -7.0);
        assert_eq!(out[3], -7.0);
    }
}
