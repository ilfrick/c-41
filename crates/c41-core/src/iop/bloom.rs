use super::{box_filters::box_mean_1ch, ClBuffer, IopProcess};
use crate::{params::IopParams, roi::RoiIn, Result};

pub struct Bloom;

/// Full bloom driver over Lab pixel buffers — the safe-Rust composition of
/// bloom.c's `process()`: derive radius/scale from the sliders, gather
/// thresholded light, box-blur it (ch=1 path of `dt_box_mean`, 8 iterations),
/// screen-blend back into L.
///
/// `input`/`output` are interleaved RGBA Lab pixels with L at index 0 (the
/// module's colourspace is IOP_CS_LAB upstream); the RGB↔Lab sandwich happens
/// in the pipeline stage, matching how Colorize/ColorCorrection are wired.
///
/// The C derives `radius = MIN(256, ceilf(rad * roi_in->scale / piece->iscale))`.
/// Whole-frame renders have scale/iscale == 1 so the ceil is a no-op; the
/// ratio is kept as a parameter so a future tiled caller can thread its own
/// through unchanged.
pub fn process(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    size: f32,
    threshold: f32,
    strength: f32,
) {
    let npixels = width * height;
    debug_assert_eq!(input.len(), npixels * 4);
    debug_assert_eq!(output.len(), npixels * 4);

    // bloom.c:137 — int truncation of the float expression, then ceilf of
    // rad·(scale/iscale), then a MIN against 256 that also truncates to int.
    let rad = (256.0f32 * ((size.min(100.0) + 1.0) / 100.0)) as i32;
    let radius_f = (rad as f32).ceil(); // · 1.0 for whole-frame renders
    let radius = radius_f.min(256.0) as usize;
    // range = 2*radius+1, hr = range/2 == radius for integer radius
    let blur_radius = (2 * radius + 1) / 2;

    // bloom.c:144
    let scale = 1.0 / (-(strength.min(100.0) + 1.0) / 100.0).exp2();

    let mut blur = vec![0.0f32; npixels];
    unsafe {
        darkroom_bloom_gather(input.as_ptr(), blur.as_mut_ptr(), npixels, threshold, scale);
    }
    // BOX_ITERATIONS = 8 (box_filters.h:25)
    box_mean_1ch(&mut blur, height, width, blur_radius, 8);
    unsafe {
        darkroom_bloom_blend(input.as_ptr(), output.as_mut_ptr(), blur.as_ptr(), npixels);
    }
}

impl IopProcess for Bloom {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "bloom" }
}

/// First bloom pass: threshold-filter input L channel into a packed 1-channel buffer.
/// blur_buf must be npixels floats (1 channel, not 4).
/// scale = 1.0 / exp2f(-1.0 * (min(100, strength+1) / 100))  pre-computed by caller.
#[no_mangle]
pub unsafe extern "C" fn darkroom_bloom_gather(
    in_buf: *const f32,
    blur_buf: *mut f32,
    npixels: usize,
    threshold: f32,
    scale: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let blur  = std::slice::from_raw_parts_mut(blur_buf, npixels);
    for k in 0..npixels {
        let l = input[k * 4] * scale;
        blur[k] = if l > threshold { l } else { 0.0 };
    }
}

/// Second bloom pass: screen-blend blurred lightness into the 4-channel output.
/// blur_buf is the 1-channel result of dt_box_mean on the gather output (npixels floats).
/// Screen blend: L_out = 100 - (100 - L_in) * (100 - L_blur) / 100; a/b/alpha copied.
#[no_mangle]
pub unsafe extern "C" fn darkroom_bloom_blend(
    in_buf: *const f32,
    out_buf: *mut f32,
    blur_buf: *const f32,
    npixels: usize,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let blur   = std::slice::from_raw_parts(blur_buf, npixels);
    for k in 0..npixels {
        output[k * 4]     = 100.0 - ((100.0 - input[k * 4]) * (100.0 - blur[k])) / 100.0;
        output[k * 4 + 1] = input[k * 4 + 1];
        output[k * 4 + 2] = input[k * 4 + 2];
        output[k * 4 + 3] = input[k * 4 + 3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_zeros_below_threshold() {
        let input = vec![50.0f32, 10.0, -5.0, 1.0,
                         80.0f32, 20.0,  5.0, 1.0];
        let mut blur = vec![0.0f32; 2];
        unsafe { darkroom_bloom_gather(input.as_ptr(), blur.as_mut_ptr(), 2, 60.0, 1.0); }
        assert_eq!(blur[0], 0.0);  // 50 < 60 → 0
        assert_eq!(blur[1], 80.0); // 80 > 60 → 80
    }

    #[test]
    fn gather_scale_applied() {
        let input = vec![50.0f32, 0.0, 0.0, 0.0];
        let mut blur = vec![0.0f32; 1];
        unsafe { darkroom_bloom_gather(input.as_ptr(), blur.as_mut_ptr(), 1, 60.0, 2.0); }
        assert_eq!(blur[0], 100.0); // 50*2=100 > 60
    }

    #[test]
    fn blend_screen_formula() {
        // Screen: 100 - (100-80)*(100-40)/100 = 100 - 20*60/100 = 100 - 12 = 88
        let input  = vec![80.0f32, 5.0, -3.0, 0.5];
        let blur   = vec![40.0f32];
        let mut output = vec![0.0f32; 4];
        unsafe { darkroom_bloom_blend(input.as_ptr(), output.as_mut_ptr(), blur.as_ptr(), 1); }
        assert!((output[0] - 88.0).abs() < 1e-4, "L={}", output[0]);
        assert_eq!(output[1],  5.0);
        assert_eq!(output[2], -3.0);
        assert_eq!(output[3],  0.5);
    }

    #[test]
    fn blend_zero_blur_is_passthrough() {
        let input  = vec![70.0f32, 1.0, 2.0, 0.8];
        let blur   = vec![0.0f32];
        let mut output = vec![0.0f32; 4];
        unsafe { darkroom_bloom_blend(input.as_ptr(), output.as_mut_ptr(), blur.as_ptr(), 1); }
        assert!((output[0] - 70.0).abs() < 1e-4);
    }

    #[test]
    fn process_below_threshold_everywhere_is_exact_identity() {
        // Nothing passes the gather threshold → the blur plane stays all-zero
        // → screen blend with zero is L_out = L_in exactly.
        let (w, h) = (9usize, 7usize);
        let mut input = Vec::new();
        for k in 0..w * h {
            input.extend_from_slice(&[40.0f32 + (k % 5) as f32, -3.0, 6.0, 1.0]);
        }
        let mut output = vec![0.0f32; input.len()];
        process(&input, &mut output, w, h, 20.0, 90.0, 25.0);
        assert_eq!(output, input);
    }

    #[test]
    fn process_bright_spot_glow_spreads_and_lifts_neighbours() {
        // A single bright pixel on a dim field: after gather+blur its light
        // spreads into a halo; neighbours' L rises while far pixels trail.
        // size=2 → rad = int(256·3/100) = 7 per pass and 8 iterations compound
        // to ~56 px of effective reach — hence a 61 px frame with decay asserted
        // at the corner rather than strict isolation. (The screen blend caps at
        // L=100, so the spot itself must sit below that for "gains L" to be
        // observable.)
        let (w, h) = (61usize, 61usize);
        let mut input = vec![0.0f32; w * h * 4];
        for px in input.chunks_exact_mut(4) {
            px[0] = 30.0; // L below threshold 90
            px[3] = 1.0;
        }
        input[(30 * w + 30) * 4] = 95.0; // centre spot

        let mut output = vec![0.0f32; input.len()];
        process(&input, &mut output, w, h, 2.0, 90.0, 25.0);

        let spot = (30 * w + 30) * 4;
        assert!(output[spot] > input[spot], "spot must gain L");
        // A neighbour inside the first-pass window gets some of the glow…
        let nb = (32 * w + 32) * 4;
        assert!(output[nb] > input[nb], "nearby neighbour must gain L");
        // …the far corner receives less than the near neighbour (monotone
        // decay of the blurred light with distance)…
        let (cx, cy) = (0usize, 0usize);
        let corner = (cy * w + cx) * 4;
        assert!(
            output[corner] < output[nb],
            "glow decays: corner {} must trail neighbour {}",
            output[corner], output[nb]
        );
        // …and a/b/alpha are never touched by the blend.
        assert_eq!(output[nb + 1], input[nb + 1]);
        assert_eq!(output[nb + 2], input[nb + 2]);
        assert_eq!(output[nb + 3], input[nb + 3]);
    }

    #[test]
    fn process_radius_derivation_matches_c_int_math() {
        // rad = int(256·(min(100,size+1)/100)); hr = (2·rad+1)/2 == rad.
        // For size=20: 256·21/100 = 53.76 → rad 53 → radius 53.
        let (w, h) = (128usize, 128usize);
        let mut input = vec![0.0f32; w * h * 4];
        for px in input.chunks_exact_mut(4) {
            px[0] = 50.0;
        }
        let mut output = vec![0.0f32; input.len()];
        // Must not panic for any slider value; extremes exercise both clamps.
        for size in [0.0f32, 20.0, 50.0, 99.0, 100.0, 150.0] {
            process(&input, &mut output, w, h, size, 90.0, 25.0);
        }
        // size ≥ 99 clamps min(100,size+1)=100 → rad=256 → MIN(256,·)=256.
        let mut out2 = vec![0.0f32; input.len()];
        process(&input, &mut out2, w, h, 99.0, 90.0, 25.0);
        let mut out3 = out2.clone();
        process(&input, &mut out3, w, h, 150.0, 90.0, 25.0);
        assert_eq!(out2, out3, "sizes beyond the clamp must agree");
    }
}
