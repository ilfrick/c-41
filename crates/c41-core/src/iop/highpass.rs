use crate::{params::IopParams, roi::RoiIn, Result};
use super::{box_filters::box_mean_1ch, ClBuffer, IopProcess};

pub struct Highpass;

impl IopProcess for Highpass {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "highpass" }
}

/// First highpass pass: invert and pack the L channel into a 1-channel buffer.
///
/// out_buf is treated as a packed 1-channel buffer (npixels floats).
/// out[k] = 100 - clamp(in[4*k], 0, 100)
/// The caller then blurs out_buf with dt_box_mean (1 channel).
#[no_mangle]
pub unsafe extern "C" fn darkroom_highpass_invert(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    let input  = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels);
    for k in 0..npixels {
        output[k] = 100.0 - input[k * 4].clamp(0.0, 100.0);
    }
}

/// Second highpass pass: blend packed 1-channel blurred L with original 4-channel input.
///
/// After dt_box_mean, out_buf contains packed 1-channel data at out[0..npixels).
/// This function reads out[k] (packed blurred L) and writes out + 4*k (4-channel pixel)
/// going in REVERSE ORDER (k from npixels-1 down to 0) so that writes never clobber
/// packed values still needed in future iterations.
///
/// pixel_L = clamp((out[k] + in[4*k] - 100) * contrast_scale + 50, 0, 100)
/// Written pixel: {pixel_L, 0, 0, 0} (desaturated Lab).
///
/// contrast_scale = ((data->contrast / 100) * 7.5) * 0.5  pre-computed by caller.
/// This single call replaces the two _blend() OMP calls plus the final sequential loop in C.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highpass_blend(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    contrast_scale: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    // out_buf is first used as 1-ch packed (npixels floats), then expanded to 4-ch (4*npixels)
    let out_packed = std::slice::from_raw_parts(out_buf, npixels);
    // read all packed blurred values before any writes touch the overlap region
    let blurred: Vec<f32> = out_packed.to_vec();

    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in (0..npixels).rev() {
        let l = (blurred[k] + input[k * 4] - 100.0) * contrast_scale + 50.0;
        let l = l.clamp(0.0, 100.0);
        output[k * 4]     = l;
        output[k * 4 + 1] = 0.0;
        output[k * 4 + 2] = 0.0;
        output[k * 4 + 3] = 0.0;
    }
}

/// Apply the highpass operation to a packed Lab frame using darktable's
/// sharpness-derived box radius and eight-pass box mean.
pub fn process_lab(input: &[f32], output: &mut [f32], width: usize, height: usize, sharpness: f32, contrast: f32, scale: f32) -> bool {
    let Some(n) = width.checked_mul(height) else { return false; };
    let Some(len) = n.checked_mul(4) else { return false; };
    if width == 0 || height == 0 || input.len() < len || output.len() < len { return false; }
    // Oversized/invalid scale values cannot produce a valid C radius conversion.
    // Normal pipeline callers provide a finite non-negative ROI scale.
    if !scale.is_finite() || scale < 0.0 || !sharpness.is_finite() || !contrast.is_finite() { return false; }
    let mut packed = vec![0.0; len];
    unsafe { darkroom_highpass_invert(input.as_ptr(), packed.as_mut_ptr(), n); }
    // C computes `sharpness + 1` as float before promotion to double for fmin.
    let sharpness_plus_one = sharpness + 1.0f32;
    let rad = (16.0 * (100.0f64.min(sharpness_plus_one as f64) / 100.0)) as usize;
    let radius = ((rad as f32 * scale).ceil() as usize).min(16);
    box_mean_1ch(&mut packed[..n], height, width, (2 * radius + 1) / 2, 8);
    unsafe { darkroom_highpass_blend(input.as_ptr(), packed.as_mut_ptr(), n, ((contrast / 100.0) * 7.5) * 0.5); }
    output[..len].copy_from_slice(&packed[..len]);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_clamps_and_inverts() {
        let input = vec![
            80.0f32, 5.0, -3.0, 1.0,   // L=80 → 20
            0.0f32,  0.0,  0.0, 0.0,   // L=0  → 100
            110.0f32, 0.0, 0.0, 0.0,   // L=110 → clamp to 100 → 0
            -5.0f32,  0.0, 0.0, 0.0,   // L=-5  → clamp to 0 → 100
        ];
        let mut out = vec![0.0f32; 4];
        unsafe { darkroom_highpass_invert(input.as_ptr(), out.as_mut_ptr(), 4); }
        assert!((out[0] - 20.0).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1] - 100.0).abs() < 1e-4);
        assert!((out[2] - 0.0).abs() < 1e-4);
        assert!((out[3] - 100.0).abs() < 1e-4);
    }

    #[test]
    fn driver_matches_eight_pass_box_mean_and_keeps_rgba_shape() {
        let input = vec![
            0.2, 0.0, 0.0, 0.25, 0.4, 0.0, 0.0, 0.5,
            0.6, 0.0, 0.0, 0.75, 0.8, 0.0, 0.0, 1.0,
        ];
        let mut expected = vec![0.0; 16];
        unsafe { darkroom_highpass_invert(input.as_ptr(), expected.as_mut_ptr(), 4); }
        // sharpness=50 gives rad=8; scale=0.25 gives radius=2. C uses
        // hr=(2*radius+1)/2, i.e. 2, and exactly eight iterations.
        box_mean_1ch(&mut expected[..4], 2, 2, 2, 8);
        unsafe { darkroom_highpass_blend(input.as_ptr(), expected.as_mut_ptr(), 4, 1.875); }
        let mut output = vec![-1.0; 16];
        assert!(process_lab(&input, &mut output, 2, 2, 50.0, 50.0, 0.25));
        assert_eq!(output, expected);
        assert!(output.chunks_exact(4).all(|p| p[1..] == [0.0; 3]));
    }

    #[test]
    fn driver_border_windows_keep_constant_plane_constant() {
        // The box filter shortens its window at edges (rather than clamping or
        // replicating); a constant plane must nevertheless remain constant.
        let mut input = Vec::new();
        for _ in 0..6 {
            input.extend_from_slice(&[37.0, 12.0, -8.0, 1.0]);
        }
        let mut output = vec![0.0; input.len()];
        assert!(process_lab(&input, &mut output, 3, 2, 100.0, 50.0, 1.0));
        for px in output.chunks_exact(4) {
            assert!((px[0] - 50.0).abs() < 1e-5);
            assert_eq!(&px[1..], &[0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn driver_rejects_invalid_shapes_without_panicking() {
        assert!(!process_lab(&[0.0; 4], &mut [0.0; 4], 2, 1, 50.0, 50.0, 1.0));
        assert!(!process_lab(&[0.0; 8], &mut [0.0; 8], 2, 1, 50.0, 50.0, f32::NAN));
    }

    fn scalar_c_reference(input: &[f32], width: usize, height: usize, sharpness: f32, contrast: f32, scale: f32) -> Vec<f32> {
        let mut work: Vec<f32> = input.chunks_exact(4).map(|p| 100.0 - p[0].clamp(0.0, 100.0)).collect();
        let rad = (16.0 * (100.0f64.min((sharpness + 1.0f32) as f64) / 100.0)) as usize;
        let radius = ((rad as f32 * scale).ceil() as usize).min(16);
        let half = (2 * radius + 1) / 2;
        for _ in 0..8 {
            let mut tmp = vec![0.0; work.len()];
            for y in 0..height { for x in 0..width {
                let lo = x.saturating_sub(half); let hi = (x + half).min(width - 1);
                let sum: f32 = (lo..=hi).map(|xx| work[y * width + xx]).sum();
                tmp[y * width + x] = sum / (hi - lo + 1) as f32;
            }}
            let mut next = vec![0.0; work.len()];
            for y in 0..height { for x in 0..width {
                let lo = y.saturating_sub(half); let hi = (y + half).min(height - 1);
                let sum: f32 = (lo..=hi).map(|yy| tmp[yy * width + x]).sum();
                next[y * width + x] = sum / (hi - lo + 1) as f32;
            }}
            work = next;
        }
        let factor = ((contrast / 100.0) * 7.5) * 0.5;
        let mut out = vec![0.0; input.len()];
        for i in 0..work.len() { out[i * 4] = ((work[i] + input[i * 4] - 100.0) * factor + 50.0).clamp(0.0, 100.0); }
        out
    }

    #[test]
    fn driver_matches_independent_scalar_c_reference() {
        let (w, h) = (5, 4);
        let input: Vec<f32> = (0..w*h).flat_map(|i| [((i * 17 + 11) % 101) as f32, i as f32, -(i as f32), 0.7]).collect();
        let below_first_radius_boundary = f32::from_bits(5.25f32.to_bits() - 1);
        for (sharpness, scale, contrast) in [
            (0.0, 1.0, 0.0),
            (below_first_radius_boundary, 1.0, 100.0),
            (5.25, 1.0, 100.0),
            (50.0, 0.25, 50.0),
            (99.0, 0.5, 100.0),
            (100.0, 1.0, 37.0),
        ] {
            let mut actual = vec![0.0; input.len()];
            assert!(process_lab(&input, &mut actual, w, h, sharpness, contrast, scale));
            let expected = scalar_c_reference(&input, w, h, sharpness, contrast, scale);
            for (a, e) in actual.iter().zip(expected) { assert!((a-e).abs() < 2e-4, "{a} != {e}"); }
            assert!(actual.chunks_exact(4).all(|p| p[1..] == [0.0; 3]));
        }
    }

    #[test]
    fn rejects_non_finite_parameters_without_writing_output() {
        for (sharpness, contrast, scale) in [(f32::NAN, 50.0, 1.0), (50.0, f32::INFINITY, 1.0), (50.0, 50.0, f32::NAN)] {
            let mut out = [9.0; 4];
            assert!(!process_lab(&[50.0; 4], &mut out, 1, 1, sharpness, contrast, scale));
            assert_eq!(out, [9.0; 4]);
        }
    }

    #[test]
    fn blend_desaturates_output() {
        let n = 2usize;
        let input  = vec![50.0f32, 10.0, -5.0, 0.5,  // pixel 0
                          70.0f32, 20.0,  3.0, 0.8]; // pixel 1
        // out initially contains packed blurred L: [40.0, 60.0] at indices [0,1]
        // after blend, out is 4*n = 8 floats
        let mut out = vec![0.0f32; n * 4];
        out[0] = 40.0; // packed blurred L for pixel 0
        out[1] = 60.0; // packed blurred L for pixel 1

        unsafe { darkroom_highpass_blend(input.as_ptr(), out.as_mut_ptr(), n, 1.0); }

        // pixel 0: L = (40 + 50 - 100)*1 + 50 = -10 + 50 = 40 → clamp 40
        assert!((out[0] - 40.0).abs() < 1e-4, "pixel0 L={}", out[0]);
        assert_eq!(out[1], 0.0); // desaturated a
        assert_eq!(out[2], 0.0); // desaturated b

        // pixel 1: L = (60 + 70 - 100)*1 + 50 = 30 + 50 = 80
        assert!((out[4] - 80.0).abs() < 1e-4, "pixel1 L={}", out[4]);
        assert_eq!(out[5], 0.0);
    }
}
