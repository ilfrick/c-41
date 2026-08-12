use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Shadhi;

impl IopProcess for Shadhi {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "shadhi" }
}

// UNBOUND flag bits — must match #defines in src/iop/shadhi.c
const UNBOUND_SHADOWS_L:    u32 = 1;
const UNBOUND_SHADOWS_A:    u32 = 2;
const UNBOUND_SHADOWS_B:    u32 = 4;
const UNBOUND_HIGHLIGHTS_L: u32 = 8;
const UNBOUND_HIGHLIGHTS_A: u32 = 16;
const UNBOUND_HIGHLIGHTS_B: u32 = 32;

#[inline(always)]
fn sign(x: f32) -> f32 { if x < 0.0 { -1.0 } else { 1.0 } }

/// Lab scale: divide L by 100, a/b by 128 (maps Lab to [0,1]/[-1,1] range).
#[inline(always)]
fn lab_scale(i: [f32; 3]) -> [f32; 3] {
    [i[0] / 100.0, i[1] / 128.0, i[2] / 128.0]
}

/// Undo lab_scale.
#[inline(always)]
fn lab_rescale(i: [f32; 3]) -> [f32; 3] {
    [i[0] * 100.0, i[1] * 128.0, i[2] * 128.0]
}

/// Shadows/Highlights IOP pixel loop.
///
/// Replaces the DT_OMP_FOR loop in src/iop/shadhi.c::process().
/// The caller (C) must first run the gaussian/bilateral blur so that
/// `out_buf` already contains the blurred version of the input when
/// this function is called.
///
/// All scalar parameters are pre-computed from dt_iop_shadhi_data_t:
///   shadows    = 2 * clamp(data->shadows / 100, -1, 1)
///   highlights = 2 * clamp(data->highlights / 100, -1, 1)
///   whitepoint = max(1 - data->whitepoint / 100, 0.01)
///   compress   = clamp(data->compress / 100, 0, 0.99)
///   shadows_ccorrect / highlights_ccorrect: as computed in process()
///   low_approximation = data->low_approximation
///   flags      = data->flags  (UNBOUND_* bitmask)
///   unbound_mask = computed from shadhi_algo and flags
#[no_mangle]
pub unsafe extern "C" fn darkroom_shadhi_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    shadows: f32,
    highlights: f32,
    whitepoint: f32,
    compress: f32,
    shadows_ccorrect: f32,
    highlights_ccorrect: f32,
    low_approximation: f32,
    flags: u32,
    unbound_mask: i32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    const HALF: f32 = 0.5;
    const LMIN: f32 = 0.0;
    const LMAX: f32 = 1.0;
    const DBMAX: f32 = 2.0; // 2*lmax
    const MIN_AB: f32 = -1.0;
    const MAX_AB: f32 =  1.0;

    for j in (0..npixels * 4).step_by(4) {
        let mut ta = lab_scale([input[j], input[j + 1], input[j + 2]]);

        // out_buf already holds the blurred pixel; invert and desaturate it
        let blurred_l = 100.0 - output[j];
        let mut tb = lab_scale([blurred_l, 0.0, 0.0]);

        ta[0] = if ta[0] > 0.0 { ta[0] / whitepoint } else { ta[0] };
        tb[0] = if tb[0] > 0.0 { tb[0] / whitepoint } else { tb[0] };

        // --- Highlight overlay ---
        let mut highlights2 = highlights * highlights; // 0..4
        let highlights_xform = (1.0 - tb[0] / (1.0 - compress)).clamp(0.0, 1.0);

        while highlights2 > 0.0 {
            let la = if (flags & UNBOUND_HIGHLIGHTS_L) != 0 { ta[0] } else { ta[0].clamp(LMIN, LMAX) };
            let mut lb = (tb[0] - HALF) * sign(-highlights) * sign(LMAX - la) + HALF;
            if unbound_mask == 0 { lb = lb.clamp(LMIN, LMAX); }

            let lref = {
                let v = la.abs().max(low_approximation);
                la.signum() / v
            };
            let href = {
                let v = (1.0 - la).abs().max(low_approximation);
                (1.0 - la).signum() / v
            };

            let chunk = if highlights2 > 1.0 { 1.0 } else { highlights2 };
            let optrans = chunk * highlights_xform;
            highlights2 -= 1.0;

            ta[0] = la * (1.0 - optrans)
                + (if la > HALF {
                    LMAX - (LMAX - DBMAX * (la - HALF)) * (LMAX - lb)
                } else {
                    DBMAX * la * lb
                }) * optrans;
            if (flags & UNBOUND_HIGHLIGHTS_L) == 0 { ta[0] = ta[0].clamp(LMIN, LMAX); }

            let chroma = ta[0] * lref * (1.0 - highlights_ccorrect)
                + (1.0 - ta[0]) * href * highlights_ccorrect;
            ta[1] = ta[1] * (1.0 - optrans) + (ta[1] + tb[1]) * chroma * optrans;
            if (flags & UNBOUND_HIGHLIGHTS_A) == 0 { ta[1] = ta[1].clamp(MIN_AB, MAX_AB); }
            ta[2] = ta[2] * (1.0 - optrans) + (ta[2] + tb[2]) * chroma * optrans;
            if (flags & UNBOUND_HIGHLIGHTS_B) == 0 { ta[2] = ta[2].clamp(MIN_AB, MAX_AB); }
        }

        // --- Shadow overlay ---
        let mut shadows2 = shadows * shadows; // 0..4
        let shadows_xform = (tb[0] / (1.0 - compress) - compress / (1.0 - compress)).clamp(0.0, 1.0);

        while shadows2 > 0.0 {
            let la = if (flags & UNBOUND_SHADOWS_L) != 0 { ta[0] } else { ta[0].clamp(LMIN, LMAX) };
            let mut lb = (tb[0] - HALF) * sign(shadows) * sign(LMAX - la) + HALF;
            if unbound_mask == 0 { lb = lb.clamp(LMIN, LMAX); }

            let lref = {
                let v = la.abs().max(low_approximation);
                la.signum() / v
            };
            let href = {
                let v = (1.0 - la).abs().max(low_approximation);
                (1.0 - la).signum() / v
            };

            let chunk = if shadows2 > 1.0 { 1.0 } else { shadows2 };
            let optrans = chunk * shadows_xform;
            shadows2 -= 1.0;

            ta[0] = la * (1.0 - optrans)
                + (if la > HALF {
                    LMAX - (LMAX - DBMAX * (la - HALF)) * (LMAX - lb)
                } else {
                    DBMAX * la * lb
                }) * optrans;
            if (flags & UNBOUND_SHADOWS_L) == 0 { ta[0] = ta[0].clamp(LMIN, LMAX); }

            let chroma = ta[0] * lref * shadows_ccorrect
                + (1.0 - ta[0]) * href * (1.0 - shadows_ccorrect);
            ta[1] = ta[1] * (1.0 - optrans) + (ta[1] + tb[1]) * chroma * optrans;
            if (flags & UNBOUND_SHADOWS_A) == 0 { ta[1] = ta[1].clamp(MIN_AB, MAX_AB); }
            ta[2] = ta[2] * (1.0 - optrans) + (ta[2] + tb[2]) * chroma * optrans;
            if (flags & UNBOUND_SHADOWS_B) == 0 { ta[2] = ta[2].clamp(MIN_AB, MAX_AB); }
        }

        let out = lab_rescale(ta);
        output[j]     = out[0];
        output[j + 1] = out[1];
        output[j + 2] = out[2];
        // alpha unchanged (not written in C either — only L/a/b are written via _Lab_rescale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_identity(pixels: &[f32], blurred: &[f32]) -> Vec<f32> {
        let n = pixels.len() / 4;
        let mut out = blurred.to_vec();
        // UNBOUND_DEFAULT = 1|2|4|8|16|32|64 = 127
        unsafe {
            darkroom_shadhi_process(
                pixels.as_ptr(), out.as_mut_ptr(), n,
                0.0, 0.0, // shadows=0, highlights=0 → no-op overlay
                1.0, 0.0, // whitepoint=1, compress=0
                0.5, 0.5, // cc corrections
                0.01,
                127, 1, // UNBOUND_DEFAULT, unbound_mask=1
            );
        }
        out
    }

    #[test]
    fn zero_shadows_highlights_is_near_identity() {
        // With shadows=0 and highlights=0 the while loops never execute (0*0=0).
        // ta = lab_scale(in), tb = lab_scale([100-blurred_L, 0, 0])
        // Then we just rescale ta back. So output should match input.
        let input  = vec![50.0, 10.0, -20.0, 0.0];
        let blurred = vec![45.0,  0.0,   0.0, 0.0];
        let out = run_identity(&input, &blurred);
        // L channel: ta[0] = 50/100 / 1.0 = 0.5 → rescaled = 50
        assert!((out[0] - 50.0).abs() < 1e-4, "L: {}", out[0]);
        assert!((out[1] - 10.0).abs() < 1e-4, "a: {}", out[1]);
        assert!((out[2] + 20.0).abs() < 1e-4, "b: {}", out[2]);
    }

    #[test]
    fn output_is_finite() {
        let input   = vec![60.0, 5.0, -5.0, 0.0, 30.0, -10.0, 8.0, 0.0];
        let blurred = vec![55.0, 0.0,  0.0, 0.0, 25.0,   0.0, 0.0, 0.0];
        let n = input.len() / 4;
        let mut out = blurred.clone();
        unsafe {
            darkroom_shadhi_process(
                input.as_ptr(), out.as_mut_ptr(), n,
                0.5, -0.5, 1.0, 0.2, 0.5, 0.5, 0.01, 127, 1,
            );
        }
        for v in &out[..6] { assert!(v.is_finite(), "NaN/Inf in output: {v}"); }
    }
}
