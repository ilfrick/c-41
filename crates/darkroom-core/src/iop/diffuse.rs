use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Diffuse;

impl IopProcess for Diffuse {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "diffuse" }
}

/// Compute the diffuse-reconstruction mask: 1 for every pixel where at least
/// one of R, G, B exceeds the threshold, 0 otherwise.
///
/// Matches `build_mask()` in src/iop/diffuse.c. `in_buf` is an RGBA float
/// buffer of length `npixels * 4`; `mask` is a single-byte mask buffer of
/// length `npixels`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_diffuse_build_mask(
    in_buf: *const f32,
    mask: *mut u8,
    npixels: usize,
    threshold: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let mask  = std::slice::from_raw_parts_mut(mask, npixels);

    for k in 0..npixels {
        let i = k * 4;
        let hit = input[i] > threshold
               || input[i + 1] > threshold
               || input[i + 2] > threshold;
        mask[k] = if hit { 1 } else { 0 };
    }
}

/// Initialise the inpaint buffer with per-pixel Gaussian noise inside masked
/// areas, and copy original pixel values outside them.
///
/// For each pixel k (stride 4):
///   if mask[k/4]:
///     seed state from (i, j) where i = k/width, j = k - i
///     warm-up state ×4
///     for c in 0..4: inpainted[k+c] = abs(gaussian_noise(orig[k+c], orig[k+c], i%2||j%2))
///   else:
///     inpainted[k..k+4] = original[k..k+4]
///
/// Matches `inpaint_mask()` in src/iop/diffuse.c:1302.
#[no_mangle]
pub unsafe extern "C" fn darkroom_diffuse_inpaint_mask(
    inpainted_buf: *mut f32,
    original_buf: *const f32,
    mask_buf: *const u8,
    width: usize,
    height: usize,
) {
    let npx = width * height;
    if npx == 0 { return; }
    let inpainted = std::slice::from_raw_parts_mut(inpainted_buf, npx * 4);
    let original  = std::slice::from_raw_parts(original_buf, npx * 4);
    let mask      = std::slice::from_raw_parts(mask_buf, npx);

    for px in 0..npx {
        let k  = px * 4;         // byte-like offset in the C sense (k strides by 4)
        let kc = k as u32;       // C uses uint32_t arithmetic
        // C: i = k / width;  j = k - i  (where k is 4*pixel_idx)
        let i  = kc / width as u32;
        let j  = kc - i;

        if mask[px] != 0 {
            // Per-pixel deterministic state
            let mut state = [
                crate::math::splitmix32((j + 1) as u64),
                crate::math::splitmix32(((j + 1) as u64) * ((i + 3) as u64)),
                crate::math::splitmix32(1337),
                crate::math::splitmix32(666),
            ];
            // 4 warm-up rounds (matches C)
            for _ in 0..4 { crate::math::xoshiro128plus(&mut state); }

            let flip = (i % 2 != 0) || (j % 2 != 0);
            for c in 0..4 {
                let mu    = original[k + c];
                let sigma = original[k + c];
                let noise = crate::math::gaussian_noise(mu, sigma, flip, &mut state);
                inpainted[k + c] = noise.abs();
            }
        } else {
            inpainted[k..k + 4].copy_from_slice(&original[k..k + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_below_threshold_produces_zero_mask() {
        let input = vec![0.1_f32, 0.2, 0.3, 1.0];
        let mut mask = vec![0xff_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn any_channel_above_threshold_sets_mask() {
        // R alone exceeds the threshold
        let input = vec![0.6_f32, 0.0, 0.0, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 1);

        // G alone
        let input = vec![0.0_f32, 0.6, 0.0, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 1);

        // B alone
        let input = vec![0.0_f32, 0.0, 0.6, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 1);
    }

    #[test]
    fn equal_to_threshold_is_not_a_hit() {
        // Predicate is strict `>` so equality fails
        let input = vec![0.5_f32, 0.5, 0.5, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn alpha_channel_does_not_influence_mask() {
        // High alpha but RGB below threshold → mask 0
        let input = vec![0.1_f32, 0.1, 0.1, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn inpaint_mask_passes_through_unmasked_pixels() {
        let orig = vec![0.3_f32, 0.5, 0.7, 1.0];
        let mut inp = vec![-1.0_f32; 4];
        let mask = vec![0_u8; 1]; // pixel not masked
        unsafe {
            darkroom_diffuse_inpaint_mask(inp.as_mut_ptr(), orig.as_ptr(), mask.as_ptr(), 1, 1);
        }
        assert_eq!(inp, orig);
    }

    #[test]
    fn inpaint_mask_fills_masked_pixel_with_nonneg_noise() {
        // When masked, the pixel is filled with |gaussian_noise(v, v)| → always ≥ 0
        let orig = vec![0.5_f32, 0.3, 0.8, 0.9];
        let mut inp = vec![-1.0_f32; 4];
        let mask = vec![1_u8]; // masked
        unsafe {
            darkroom_diffuse_inpaint_mask(inp.as_mut_ptr(), orig.as_ptr(), mask.as_ptr(), 1, 1);
        }
        // All channels non-negative (abs applied)
        for c in 0..4 { assert!(inp[c] >= 0.0, "c={c} val={}", inp[c]); }
        // Result differs from original (noise was added)
        // (with high probability — pathological exact-zero noise would be a 1-in-2^24 event)
    }

    #[test]
    fn inpaint_mask_is_deterministic() {
        // Same seed = same output every time
        let orig = vec![0.4_f32, 0.6, 0.2, 1.0];
        let mask = vec![1_u8];
        let mut out1 = vec![0.0_f32; 4];
        let mut out2 = vec![0.0_f32; 4];
        unsafe {
            darkroom_diffuse_inpaint_mask(out1.as_mut_ptr(), orig.as_ptr(), mask.as_ptr(), 1, 1);
            darkroom_diffuse_inpaint_mask(out2.as_mut_ptr(), orig.as_ptr(), mask.as_ptr(), 1, 1);
        }
        assert_eq!(out1, out2);
    }

    #[test]
    fn multi_pixel_mixed_result() {
        let input = vec![
            0.1, 0.1, 0.1, 1.0,  // below
            0.9, 0.1, 0.1, 1.0,  // R above
            0.1, 0.9, 0.1, 1.0,  // G above
            0.1, 0.1, 0.1, 1.0,  // below
        ];
        let mut mask = vec![0_u8; 4];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 4, 0.5); }
        assert_eq!(mask, vec![0, 1, 1, 0]);
    }
}
