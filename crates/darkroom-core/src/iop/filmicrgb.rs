use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Filmicrgb;

impl IopProcess for Filmicrgb {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "filmicrgb" }
}

/// Build the per-pixel highlight-reconstruction mask.
///
/// For each pixel k:
///   pix_max = sqrt(R² + G² + B²)    (Euclidean norm of RGB)
///   argument = -pix_max * normalize + feathering
///   weight   = clamp(1 / (1 + 2^argument), 0, 1)  (soft sigmoid)
///   mask[k]  = weight
///   clipped += (4.0 > argument)      (count pixels near transition)
///
/// Returns the number of pixels that are "clipped" (argument < 4.0).
/// If this count is <= 9, the caller skips the expensive reconstruction.
///
/// Matches the `DT_OMP_FOR(reduction(+:clipped))` loop in
/// `reconstruct_highlights_build_mask()` (filmicrgb.c:1050).
#[no_mangle]
pub unsafe extern "C" fn darkroom_filmicrgb_build_reconstruction_mask(
    in_buf: *const f32,
    mask_buf: *mut f32,
    npixels: usize,
    normalize: f32,
    feathering: f32,
) -> i32 {
    if npixels == 0 { return 0; }
    let inp  = std::slice::from_raw_parts(in_buf, npixels * 4);
    let mask = std::slice::from_raw_parts_mut(mask_buf, npixels);
    let mut clipped: i32 = 0;

    for k in 0..npixels {
        let b = k * 4;
        let r = inp[b];
        let g = inp[b + 1];
        let bi = inp[b + 2];
        let pix_max = (r * r + g * g + bi * bi).sqrt();
        let argument = -pix_max * normalize + feathering;
        let weight = (1.0 / (1.0 + (argument).exp2())).clamp(0.0, 1.0);
        mask[k] = weight;
        if 4.0_f32 > argument { clipped += 1; }
    }
    clipped
}

/// Copy a single-channel mask scalar to all 4 channels of every output pixel.
///
/// For each pixel k:
///   out[k*4 + c] = mask[k]   for c in 0..4
///
/// Matches `display_mask()` in filmicrgb.c:2012.
#[no_mangle]
pub unsafe extern "C" fn darkroom_filmicrgb_display_mask(
    mask_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    if npixels == 0 { return; }
    let mask = std::slice::from_raw_parts(mask_buf, npixels);
    let out  = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in 0..npixels {
        let v = mask[k];
        out[k * 4]     = v;
        out[k * 4 + 1] = v;
        out[k * 4 + 2] = v;
        out[k * 4 + 3] = v;
    }
}

/// Restore per-channel pixel values from their ratios and norms.
///
/// For each pixel k and channel c:
///   ratios[k*4 + c] = clamp(ratios[k*4 + c], 0, 1) * norms[k]
///
/// Matches `restore_ratios()` in filmicrgb.c:2051.
#[no_mangle]
pub unsafe extern "C" fn darkroom_filmicrgb_restore_ratios(
    ratios_buf: *mut f32,
    norms_buf: *const f32,
    npixels: usize,
) {
    if npixels == 0 { return; }
    let ratios = std::slice::from_raw_parts_mut(ratios_buf, npixels * 4);
    let norms  = std::slice::from_raw_parts(norms_buf, npixels);
    for k in 0..npixels {
        let n = norms[k];
        for c in 0..4 {
            ratios[k * 4 + c] = ratios[k * 4 + c].clamp(0.0, 1.0) * n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_is_one_when_argument_is_very_negative() {
        // pix_max large, argument = -large*norm + feather ≈ -∞ → 1/(1+2^-∞) → 1
        let inp = vec![1000.0_f32, 0.0, 0.0, 1.0];
        let mut mask = vec![0.0_f32; 1];
        let n = unsafe {
            darkroom_filmicrgb_build_reconstruction_mask(
                inp.as_ptr(), mask.as_mut_ptr(), 1, 1.0, 1.0,
            )
        };
        assert!((mask[0] - 1.0).abs() < 1e-4, "mask={}", mask[0]);
        assert_eq!(n, 1); // argument = -999, 4.0 > -999 = true
    }

    #[test]
    fn mask_is_half_at_zero_argument() {
        // pix_max = feather/normalize → argument = 0 → 1/(1+1) = 0.5
        let feathering = 0.5_f32;
        let normalize  = 0.5_f32;
        // want pix_max * normalize = feathering → pix_max = feathering/normalize = 1
        let inp = vec![1.0_f32, 0.0, 0.0, 0.0]; // pix_max = 1
        let mut mask = vec![0.0_f32; 1];
        unsafe {
            darkroom_filmicrgb_build_reconstruction_mask(
                inp.as_ptr(), mask.as_mut_ptr(), 1, normalize, feathering,
            );
        }
        assert!((mask[0] - 0.5).abs() < 1e-4, "mask={}", mask[0]);
    }

    #[test]
    fn display_mask_broadcasts_to_all_channels() {
        let mask = vec![0.42_f32; 3];
        let mut out = vec![-1.0_f32; 12];
        unsafe { darkroom_filmicrgb_display_mask(mask.as_ptr(), out.as_mut_ptr(), 3); }
        for v in &out { assert!((v - 0.42).abs() < 1e-6); }
    }

    #[test]
    fn restore_ratios_clamps_then_multiplies() {
        // ratio = 1.5 → clamp(1.5, 0, 1) = 1.0 → * norm=2 = 2.0
        let mut ratios = vec![1.5_f32, -0.5, 0.5, 0.25];
        let norms = vec![2.0_f32];
        unsafe {
            darkroom_filmicrgb_restore_ratios(ratios.as_mut_ptr(), norms.as_ptr(), 1);
        }
        assert!((ratios[0] - 2.0).abs() < 1e-6); // 1.5 → 1.0 * 2.0
        assert!((ratios[1] - 0.0).abs() < 1e-6); // -0.5 → 0.0 * 2.0
        assert!((ratios[2] - 1.0).abs() < 1e-6); // 0.5 → 0.5 * 2.0
        assert!((ratios[3] - 0.5).abs() < 1e-6); // 0.25 → 0.25 * 2.0
    }
}
