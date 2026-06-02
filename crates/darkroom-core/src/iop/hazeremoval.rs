use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Hazeremoval;

impl IopProcess for Hazeremoval {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "hazeremoval" }
}

/// Compute the per-pixel dark channel: min(R,G,B) over an RGBA input,
/// writing one gray scalar per pixel.
///
/// Matches the inner loop of _dark_channel() in src/iop/hazeremoval.c.
/// `npixels` is the total pixel count (height * width).
#[no_mangle]
pub unsafe extern "C" fn darkroom_hazeremoval_dark_channel(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels);

    for i in 0..npixels {
        let base = i * 4;
        let r = input[base];
        let g = input[base + 1];
        let b = input[base + 2];
        output[i] = r.min(g).min(b);
    }
}

/// Compute the per-pixel transition map:
///   out[i] = 1 - min(min(R*A0_inv[0], G*A0_inv[1]), B*A0_inv[2]) * strength
///
/// Matches the inner loop of _transition_map() in src/iop/hazeremoval.c.
/// `a0_inv` is a 3-float array of reciprocal ambient-light values.
#[no_mangle]
pub unsafe extern "C" fn darkroom_hazeremoval_transition_map(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    a0_inv: *const f32,
    strength: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels);
    let a0i = std::slice::from_raw_parts(a0_inv, 3);

    for i in 0..npixels {
        let base = i * 4;
        let r = input[base] * a0i[0];
        let g = input[base + 1] * a0i[1];
        let b = input[base + 2] * a0i[2];
        let m = r.min(g).min(b);
        output[i] = 1.0 - m * strength;
    }
}

/// Apply the haze-removal formula to every channel of every pixel:
///   t = max(trans_map[i], t_min)
///   out[4i + c] = (in[4i + c] - A0[c]) / t + A0[c]   for c in 0..4
///
/// Matches the final dehazing loop in `process()` (hazeremoval.c ~line 683).
/// `a0` is a 4-float ambient-light array (RGB + alpha pad).
#[no_mangle]
pub unsafe extern "C" fn darkroom_hazeremoval_dehaze(
    in_buf: *const f32,
    out_buf: *mut f32,
    trans_map: *const f32,
    npixels: usize,
    a0: *const f32,
    t_min: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let trans = std::slice::from_raw_parts(trans_map, npixels);
    let a0s = std::slice::from_raw_parts(a0, 4);

    for i in 0..npixels {
        let t = trans[i].max(t_min);
        let inv_t = 1.0 / t;
        let base = i * 4;
        for c in 0..4 {
            output[base + c] = (input[base + c] - a0s[c]) * inv_t + a0s[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_channel_takes_min_of_rgb() {
        let input = vec![0.7_f32, 0.2, 0.5, 0.9, 0.4, 0.4, 0.4, 1.0];
        let mut out = vec![0.0_f32; 2];
        unsafe { darkroom_hazeremoval_dark_channel(input.as_ptr(), out.as_mut_ptr(), 2); }
        assert_eq!(out[0], 0.2);
        assert_eq!(out[1], 0.4);
    }

    #[test]
    fn dark_channel_ignores_alpha() {
        let input = vec![0.5_f32, 0.5, 0.5, 0.01];
        let mut out = vec![0.0_f32; 1];
        unsafe { darkroom_hazeremoval_dark_channel(input.as_ptr(), out.as_mut_ptr(), 1); }
        assert_eq!(out[0], 0.5); // alpha 0.01 must NOT win the min
    }

    #[test]
    fn transition_map_formula() {
        // a0_inv = (1/0.5, 1/0.5, 1/0.5) = (2,2,2); strength = 1
        let input = vec![0.1_f32, 0.2, 0.3, 1.0];
        let a0_inv = [2.0_f32, 2.0, 2.0];
        let mut out = vec![0.0_f32; 1];
        unsafe {
            darkroom_hazeremoval_transition_map(
                input.as_ptr(), out.as_mut_ptr(), 1, a0_inv.as_ptr(), 1.0,
            );
        }
        // min(0.2, 0.4, 0.6) = 0.2 → 1 - 0.2 = 0.8
        assert!((out[0] - 0.8).abs() < 1e-5, "out={}", out[0]);
    }

    #[test]
    fn transition_map_strength_scales_haze_term() {
        let input = vec![0.1_f32, 0.1, 0.1, 1.0];
        let a0_inv = [1.0_f32, 1.0, 1.0];
        let mut out = vec![0.0_f32; 1];
        unsafe {
            darkroom_hazeremoval_transition_map(
                input.as_ptr(), out.as_mut_ptr(), 1, a0_inv.as_ptr(), 0.5,
            );
        }
        // min = 0.1 → 1 - 0.1*0.5 = 0.95
        assert!((out[0] - 0.95).abs() < 1e-5, "out={}", out[0]);
    }

    #[test]
    fn dehaze_identity_when_t_one_and_a0_zero() {
        let input = vec![0.3_f32, 0.5, 0.7, 1.0];
        let trans = vec![1.0_f32];
        let a0 = [0.0_f32, 0.0, 0.0, 0.0];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_hazeremoval_dehaze(
                input.as_ptr(), out.as_mut_ptr(), trans.as_ptr(),
                1, a0.as_ptr(), 0.0,
            );
        }
        assert_eq!(out, input);
    }

    #[test]
    fn dehaze_applies_t_min_floor() {
        // trans_map below t_min must be clamped
        let input = vec![1.0_f32, 1.0, 1.0, 1.0];
        let trans = vec![0.001_f32];
        let a0 = [0.0_f32, 0.0, 0.0, 0.0];
        let t_min = 0.5_f32;
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_hazeremoval_dehaze(
                input.as_ptr(), out.as_mut_ptr(), trans.as_ptr(),
                1, a0.as_ptr(), t_min,
            );
        }
        // (1 - 0)/0.5 + 0 = 2.0 (uses t_min, not 0.001)
        assert!((out[0] - 2.0).abs() < 1e-5, "out={}", out[0]);
    }

    #[test]
    fn dehaze_subtracts_and_adds_a0_per_channel() {
        // For in[c]=0.6, A0[c]=0.4, t=2 → (0.6-0.4)/2 + 0.4 = 0.5
        let input = vec![0.6_f32, 0.6, 0.6, 0.6];
        let trans = vec![2.0_f32];
        let a0 = [0.4_f32, 0.4, 0.4, 0.4];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_hazeremoval_dehaze(
                input.as_ptr(), out.as_mut_ptr(), trans.as_ptr(),
                1, a0.as_ptr(), 0.0,
            );
        }
        for c in 0..4 {
            assert!((out[c] - 0.5).abs() < 1e-5, "c={} out={}", c, out[c]);
        }
    }
}

/// Estimate ambient light A0 by summing RGBA over bright-hazy pixels.
///
/// Accumulates rgba into a0_out (4 floats, caller-zeroed) and increments
/// *count_out for each pixel where dark_channel[i] >= crit_haze_level AND
/// sum(R+G+B) >= crit_brightness. Caller divides by count if count > 0.
///
/// Matches DT_OMP_FOR reduction in src/iop/hazeremoval.c:510.
#[no_mangle]
pub unsafe extern "C" fn darkroom_hazeremoval_ambient_light(
    dark_channel: *const f32,
    in_rgba: *const f32,
    size: usize,
    crit_haze_level: f32,
    crit_brightness: f32,
    a0_out: *mut f32,
    count_out: *mut usize,
) {
    if size == 0 { return; }
    let data = std::slice::from_raw_parts(dark_channel, size);
    let inp  = std::slice::from_raw_parts(in_rgba, size * 4);
    let a0   = std::slice::from_raw_parts_mut(a0_out, 4);
    a0.fill(0.0);
    *count_out = 0;
    for i in 0..size {
        if data[i] >= crit_haze_level {
            let r = inp[i * 4];
            let g = inp[i * 4 + 1];
            let b = inp[i * 4 + 2];
            if r + g + b >= crit_brightness {
                a0[0] += r;
                a0[1] += g;
                a0[2] += b;
                a0[3] += inp[i * 4 + 3];
                *count_out += 1;
            }
        }
    }
}
