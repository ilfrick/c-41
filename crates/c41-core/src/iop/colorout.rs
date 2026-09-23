use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct ColorOut;

impl IopProcess for ColorOut {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colorout" }
}

// Matches lab_f_inv() in colorspaces_inline_conversions.h.
// epsilon = cbrt(216/24389), kappa = 24389/27.
#[inline(always)]
fn lab_f_inv(x: f32) -> f32 {
    const EPSILON: f32 = 0.20689655172413796;
    const KAPPA: f32 = 24389.0 / 27.0;
    if x > EPSILON { x * x * x } else { (116.0 * x - 16.0) / KAPPA }
}

// Matches dt_Lab_to_XYZ() — D50 white point, Lab→XYZ per CIE standard.
#[inline(always)]
fn lab_to_xyz(lab: &[f32]) -> [f32; 3] {
    // D50 = { 0.9642, 1.0, 0.8249 }
    const D50: [f32; 3] = [0.9642, 1.0, 0.8249];
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;
    [D50[0] * lab_f_inv(fx), D50[1] * lab_f_inv(fy), D50[2] * lab_f_inv(fz)]
}

// Matches _transform_cmatrix_linear() — Lab→XYZ then transposed-matrix multiply.
// cmatrix: pre-transposed 3×4 colormatrix (12 floats, row-major).
//   cmatrix[row*4 + out_ch] so rgb[c] = cmatrix[0][c]*X + cmatrix[1][c]*Y + cmatrix[2][c]*Z
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorout_cmatrix_linear(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    cmatrix: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let cm = std::slice::from_raw_parts(cmatrix, 12);
    for k in 0..npixels {
        let xyz = lab_to_xyz(&input[k * 4..]);
        // rgb[c] = cm[0+c]*X + cm[4+c]*Y + cm[8+c]*Z  (transposed multiply)
        output[k * 4]     = cm[0] * xyz[0] + cm[4] * xyz[1] + cm[8]  * xyz[2];
        output[k * 4 + 1] = cm[1] * xyz[0] + cm[5] * xyz[1] + cm[9]  * xyz[2];
        output[k * 4 + 2] = cm[2] * xyz[0] + cm[6] * xyz[1] + cm[10] * xyz[2];
        output[k * 4 + 3] = 0.0;
    }
}

const LUT_SAMPLES: usize = 0x10000;

/// Linear interpolation into a 65536-entry float LUT.
/// Matches _lerp_lut() in colorout.c: clips v to [0, +∞), then interpolates.
/// Caller guarantees v < 1.0 so the index stays within [0, LUT_SAMPLES-2].
#[inline(always)]
fn lerp_lut(lut: &[f32], v: f32) -> f32 {
    let z = v.max(0.0);
    let ft = z * (LUT_SAMPLES - 1) as f32;
    let t = (ft as usize).min(LUT_SAMPLES - 2);
    let f = ft - t as f32;
    lut[t] * (1.0 - f) + lut[t + 1] * f
}

/// Unbounded extrapolation: coeff[1] * pow(v * coeff[0], coeff[2]).
/// Matches dt_iop_eval_exp() in imageop_math.h.
#[inline(always)]
fn eval_exp(coeff: &[f32], v: f32) -> f32 {
    coeff[1] * (v * coeff[0]).powf(coeff[2])
}

/// Apply per-channel tone curves (LUT + unbounded exp) in-place.
///
/// Replaces both DT_OMP_FOR loops in process_fastpath_apply_tonecurves() in colorout.c.
/// lut:              3 × 65536 floats, row-major (channel c at offset c*65536).
/// unbounded_coeffs: 3 × 3 floats, row-major (channel c at offset c*3).
/// lut_active:       3 ints — non-zero means the LUT for that channel is active.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorout_apply_tonecurves(
    buf: *mut f32,
    npixels: usize,
    lut: *const f32,
    unbounded_coeffs: *const f32,
    lut_active: *const i32,
) {
    let buf = std::slice::from_raw_parts_mut(buf, npixels * 4);
    let lut = std::slice::from_raw_parts(lut, 3 * LUT_SAMPLES);
    let coeffs = std::slice::from_raw_parts(unbounded_coeffs, 9);
    let active = std::slice::from_raw_parts(lut_active, 3);

    for k in 0..npixels {
        let base = k * 4;
        for c in 0..3 {
            if active[c] != 0 {
                let v = buf[base + c];
                buf[base + c] = if v < 1.0 {
                    lerp_lut(&lut[c * LUT_SAMPLES..(c + 1) * LUT_SAMPLES], v)
                } else {
                    eval_exp(&coeffs[c * 3..(c + 1) * 3], v)
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_f_inv_identity_at_epsilon() {
        // For x > epsilon, should be x^3
        let x = 0.5f32;
        assert!((lab_f_inv(x) - x * x * x).abs() < 1e-6);
    }

    #[test]
    fn lab_f_inv_linear_below_epsilon() {
        let x = 0.1f32;
        let expected = (116.0 * x - 16.0) / (24389.0 / 27.0);
        assert!((lab_f_inv(x) - expected).abs() < 1e-6);
    }

    #[test]
    fn d65_white_in_lab_gives_d50_xyz() {
        // L=100, a=0, b=0 is D50 white in Lab
        let lab = [100.0f32, 0.0, 0.0, 0.0];
        let xyz = lab_to_xyz(&lab);
        // Should be close to D50 = [0.9642, 1.0, 0.8249]
        assert!((xyz[0] - 0.9642).abs() < 1e-4, "X={}", xyz[0]);
        assert!((xyz[1] - 1.0).abs() < 1e-4,    "Y={}", xyz[1]);
        assert!((xyz[2] - 0.8249).abs() < 1e-4,  "Z={}", xyz[2]);
    }

    #[test]
    fn lerp_lut_identity_lut() {
        // A LUT where lut[k] = k/(N-1) is the identity mapping.
        let n = LUT_SAMPLES;
        let lut: Vec<f32> = (0..n).map(|k| k as f32 / (n - 1) as f32).collect();
        let v = 0.5f32;
        let out = lerp_lut(&lut, v);
        assert!((out - v).abs() < 1e-4, "out={out}");
    }

    #[test]
    fn lerp_lut_clips_negative() {
        let lut: Vec<f32> = vec![0.0f32; LUT_SAMPLES];
        // negative input clips to 0 → lut[0] = 0
        assert_eq!(lerp_lut(&lut, -0.5), 0.0);
    }

    #[test]
    fn eval_exp_matches_formula() {
        let coeff = [2.0f32, 3.0, 0.5];
        let v = 0.25f32;
        let expected = 3.0 * (0.25 * 2.0f32).powf(0.5);
        assert!((eval_exp(&coeff, v) - expected).abs() < 1e-5);
    }

    #[test]
    fn apply_tonecurves_inactive_channel_unchanged() {
        let lut = vec![0.0f32; 3 * LUT_SAMPLES]; // all-zero LUT
        let coeffs = [1.0f32, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0];
        let active = [0i32, 0, 0]; // all inactive
        let mut buf = vec![0.3f32, 0.6, 0.9, 1.0];
        unsafe {
            darkroom_colorout_apply_tonecurves(
                buf.as_mut_ptr(), 1,
                lut.as_ptr(), coeffs.as_ptr(), active.as_ptr(),
            );
        }
        assert_eq!(buf, vec![0.3, 0.6, 0.9, 1.0]); // unchanged
    }

    #[test]
    fn apply_tonecurves_identity_lut_passthrough() {
        // Identity LUT: maps v → v for v in [0,1)
        let n = LUT_SAMPLES;
        let single_lut: Vec<f32> = (0..n).map(|k| k as f32 / (n - 1) as f32).collect();
        let lut: Vec<f32> = single_lut.iter().chain(single_lut.iter()).chain(single_lut.iter()).copied().collect();
        let coeffs = [1.0f32; 9];
        let active = [1i32, 1, 1];
        let input = [0.25f32, 0.5, 0.75, 1.0];
        let mut buf = input.to_vec();
        unsafe {
            darkroom_colorout_apply_tonecurves(
                buf.as_mut_ptr(), 1,
                lut.as_ptr(), coeffs.as_ptr(), active.as_ptr(),
            );
        }
        assert!((buf[0] - 0.25).abs() < 1e-4, "R={}", buf[0]);
        assert!((buf[1] - 0.5 ).abs() < 1e-4, "G={}", buf[1]);
        assert!((buf[2] - 0.75).abs() < 1e-4, "B={}", buf[2]);
        assert_eq!(buf[3], 1.0); // alpha unchanged
    }

    #[test]
    fn identity_cmatrix_passes_xyz_through() {
        // Use a 3×4 identity-like cmatrix (transposed form):
        // row 0 = [1,0,0,0], row 1 = [0,1,0,0], row 2 = [0,0,1,0]
        // rgb[0] = 1*X + 0*Y + 0*Z = X, etc.
        let cm = [
            1.0f32, 0.0, 0.0, 0.0,   // row 0
            0.0f32, 1.0, 0.0, 0.0,   // row 1
            0.0f32, 0.0, 1.0, 0.0,   // row 2
        ];
        // L=100, a=0, b=0 → XYZ = D50
        let input = vec![100.0f32, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 4];
        unsafe {
            darkroom_colorout_cmatrix_linear(
                input.as_ptr(), out.as_mut_ptr(), 1, cm.as_ptr()
            );
        }
        assert!((out[0] - 0.9642).abs() < 1e-4, "R={}", out[0]);
        assert!((out[1] - 1.0).abs()    < 1e-4,  "G={}", out[1]);
        assert!((out[2] - 0.8249).abs() < 1e-4,  "B={}", out[2]);
        assert_eq!(out[3], 0.0); // alpha always zeroed
    }

    #[test]
    fn gamutcheck_trigger_rails_and_alpha() {
        let cyan_bits: Vec<u32> = [0.0f32, 1.0, 1.0, 0.0].iter().map(|v| v.to_bits()).collect();
        let mut buf = vec![
            0.5f32, 0.25, 0.125, 0.7, // clean: untouched incl. alpha
            -0.0, 0.25, 0.125, 0.7,   // -0.0 is NOT < 0: untouched
            -1e-45, 0.25, 0.125, 0.7, // tiny negative lane 0: triggers
            0.5, -2.0, 0.125, 0.7,    // lane 1 triggers
            0.5, 0.25, -3.0, 0.7,     // lane 2 triggers
            -1.0, -1.0, -1.0, 0.9,    // all lanes negative: triggers
        ];
        let before = buf.clone();
        colorout_gamutcheck_fill(&mut buf, 6);
        for (k, px) in buf.as_chunks::<4>().0.iter().enumerate() {
            let bits: Vec<u32> = px.iter().map(|v| v.to_bits()).collect();
            if k < 2 {
                let want: Vec<u32> = before[k * 4..k * 4 + 4].iter().map(|v| v.to_bits()).collect();
                assert_eq!(bits, want, "pixel {k} must be untouched");
            } else {
                assert_eq!(bits, cyan_bits, "pixel {k} must be cyan");
            }
        }
        assert_eq!(buf[3].to_bits(), 0.7f32.to_bits(), "clean alpha preserved");
        assert_eq!(buf[23].to_bits(), 0.0f32.to_bits(), "triggered alpha overwritten");
    }

    #[test]
    fn gamutcheck_nan_lane_does_not_trigger() {
        let nan = f32::NAN;
        // NaN in each lane with the other two non-negative: untouched.
        for lane in 0..3 {
            let mut px = [0.5f32, 0.25, 0.125, 0.7];
            px[lane] = nan;
            let mut buf = px.to_vec();
            colorout_gamutcheck_fill(&mut buf, 1);
            assert_eq!(buf[0].to_bits(), px[0].to_bits(), "lane {lane}");
            assert_eq!(buf[1].to_bits(), px[1].to_bits(), "lane {lane}");
            assert_eq!(buf[2].to_bits(), px[2].to_bits(), "lane {lane}");
            assert_eq!(buf[3].to_bits(), px[3].to_bits(), "lane {lane}");
        }
        // NaN alongside a genuinely negative lane: the negative still fires.
        let mut buf = vec![nan, -0.5, 0.125, 0.7];
        colorout_gamutcheck_fill(&mut buf, 1);
        assert_eq!(buf, vec![0.0f32, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn gamutcheck_matches_reference_lcg_sweep() {
        // LCG over raw u32 bits -> f32: negatives, subnormals, zeros,
        // infinities and NaNs all appear; kernel must equal the divergent
        // reference bit-exactly, and the sentinel tail must stay untouched.
        const N: usize = 512;
        let mut state: u32 = 0x12345678;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            f32::from_bits(state)
        };
        let mut input = vec![0.0f32; N * 4 + 8];
        for v in input[..N * 4].iter_mut() {
            *v = next();
        }
        for v in input[N * 4..].iter_mut() {
            *v = 42.0;
        }
        let mut a = input.clone();
        let mut b = input.clone();
        colorout_gamutcheck_fill(&mut a, N);
        colorout_gamutcheck_fill_ref(&mut b, N);
        for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "lane {k}");
        }
        assert!(a[N * 4..].iter().all(|&v| v == 42.0), "sentinel tail untouched");
    }

    #[test]
    fn gamutcheck_ffi_maybeuninit_matches_kernel() {
        use std::mem::MaybeUninit;
        const N: usize = 64;
        let mut state: u32 = 0xdeadbeef;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            f32::from_bits(state)
        };
        let src: Vec<f32> = (0..N * 4).map(|_| next()).collect();
        let mut ffi_buf: Box<[MaybeUninit<f32>]> = Box::new_uninit_slice(N * 4);
        for (slot, &v) in ffi_buf.iter_mut().zip(src.iter()) {
            slot.write(v);
        }
        unsafe {
            darkroom_colorout_gamutcheck_fill(ffi_buf.as_mut_ptr() as *mut f32, N);
        }
        let ffi_out = unsafe { Box::<[f32]>::from_raw(Box::into_raw(ffi_buf) as *mut [f32]) };
        let mut direct = src.clone();
        colorout_gamutcheck_fill(&mut direct, N);
        assert_eq!(ffi_out.len(), direct.len());
        for (k, (f, d)) in ffi_out.iter().zip(direct.iter()).enumerate() {
            assert_eq!(f.to_bits(), d.to_bits(), "lane {k}");
        }
    }

    #[test]
    fn gamutcheck_ffi_guards_and_degenerate() {
        // Null pointer and zero count are no-ops (must not trap).
        unsafe {
            darkroom_colorout_gamutcheck_fill(std::ptr::null_mut(), 4);
            darkroom_colorout_gamutcheck_fill(std::ptr::null_mut(), 0);
            let mut z = [1.0f32, 2.0, 3.0, 4.0];
            darkroom_colorout_gamutcheck_fill(z.as_mut_ptr(), 0);
            assert_eq!(z, [1.0, 2.0, 3.0, 4.0], "zero count leaves buffer alone");
            // Checked-product overflow: returns before dereferencing.
            let mut small = [0.5f32, 0.5, 0.5, 0.5];
            darkroom_colorout_gamutcheck_fill(small.as_mut_ptr(), usize::MAX);
            assert_eq!(small, [0.5, 0.5, 0.5, 0.5]);
            // isize byte-span cap: products valid, byte span over the cap.
            darkroom_colorout_gamutcheck_fill(small.as_mut_ptr(), isize::MAX as usize / 16 + 1);
            assert_eq!(small, [0.5, 0.5, 0.5, 0.5]);
        }
        // Degenerate first-quad contents through the FFI.
        let mut clean = [0.0f32, 0.0, 0.0, 0.0];
        unsafe {
            darkroom_colorout_gamutcheck_fill(clean.as_mut_ptr(), 1);
        }
        assert_eq!(clean, [0.0, 0.0, 0.0, 0.0]);
        let mut dirty = [0.0f32, 0.0, -1e-30, 5.0];
        unsafe {
            darkroom_colorout_gamutcheck_fill(dirty.as_mut_ptr(), 1);
        }
        assert_eq!(dirty, [0.0, 1.0, 1.0, 0.0]);
    }
}

/// Fused Lab->linearRGB (via pre-transposed cmatrix) + per-channel tone curve.
///
/// For each pixel:
///   1. Lab -> XYZ (D50) via CIE formula
///   2. rgb[r] = sum_c(cmatrix_row_c[r] * XYZ[c])   (transposed matrix multiply)
///   3. For each channel c: if lut[c*LUT_SAMPLES] >= 0, apply LUT+exp extrapolation
///
/// cmatrix:          12 floats (3 rows x 4 cols, row-major); same layout as cmatrix_linear
/// lut:              3 x LUT_SAMPLES floats; channel active iff lut[c*LUT_SAMPLES] >= 0
/// unbounded_coeffs: 9 floats (3 sets of 3); eval_exp(c*3, v) = coeff[1]*pow(v*coeff[0], coeff[2])
///
/// Matches _transform_cmatrix_tonecurve() DT_OMP_FOR in src/iop/colorout.c:413.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorout_cmatrix_tonecurve(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    cmatrix: *const f32,
    lut: *const f32,
    unbounded_coeffs: *const f32,
) {
    let input   = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output  = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let cm      = std::slice::from_raw_parts(cmatrix, 12);
    let lut_s   = std::slice::from_raw_parts(lut, 3 * LUT_SAMPLES);
    let coeffs  = std::slice::from_raw_parts(unbounded_coeffs, 9);

    for k in 0..npixels {
        let xyz = lab_to_xyz(&input[k * 4..]);
        let mut rgb = [0.0f32; 4];
        for r in 0..3usize {
            rgb[r] = cm[r]         * xyz[0]
                   + cm[4 + r]     * xyz[1]
                   + cm[8 + r]     * xyz[2];
        }
        for c in 0..3usize {
            let lut_c = &lut_s[c * LUT_SAMPLES..(c + 1) * LUT_SAMPLES];
            if lut_c[0] >= 0.0 {
                rgb[c] = if rgb[c] < 1.0 {
                    lerp_lut(lut_c, rgb[c])
                } else {
                    eval_exp(&coeffs[c * 3..(c + 1) * 3], rgb[c])
                };
            }
        }
        output[k * 4]     = rgb[0];
        output[k * 4 + 1] = rgb[1];
        output[k * 4 + 2] = rgb[2];
        output[k * 4 + 3] = input[k * 4 + 3];
    }
}

// Gamut-check cyan fill (m4-244).
//
// Matches the `if(gamutcheck)` block in `_transform_lcms()` in
// src/iop/colorout.c: per 4-lane pixel, when ANY of lanes 0-2 is strictly
// less than 0.0, all 4 lanes are overwritten with cyan
// { 0.0, 1.0, 1.0, 0.0 } (hardcoded here to match the C `cyan` constant);
// otherwise the pixel is left fully untouched, alpha included. The trigger
// comparison is plain `<`, so a NaN lane compares false and never triggers
// on its own -- replicate exactly, do NOT add `is_sign_negative` or NaN
// checks. `copy_pixel_nontemporal` in C is the same bytes on all standard
// (vectorized/SSE/aarch64) paths — a full 4-lane copy with a nontemporal
// store hint (throughput only); plain stores write the same bytes, and the
// `dt_omploop_sfence()` after the outer loop stays in C.
const COLOROUT_GAMUTCHECK_CYAN: [f32; 4] = [0.0, 1.0, 1.0, 0.0];

// In-place cyan fill over the first `npixels` quads of `out`.
// Defensive guards (length/overflow) return without touching `out`; the FFI
// wrapper below already validates, so these only fire on direct misuse.
pub fn colorout_gamutcheck_fill(out: &mut [f32], npixels: usize) {
    let Some(len) = npixels.checked_mul(4) else { return; };
    if out.len() < len {
        return;
    }
    for j in 0..npixels {
        let b = j * 4;
        if out[b] < 0.0 || out[b + 1] < 0.0 || out[b + 2] < 0.0 {
            out[b..b + 4].copy_from_slice(&COLOROUT_GAMUTCHECK_CYAN);
        }
    }
}

#[cfg(test)]
fn colorout_gamutcheck_fill_ref(out: &mut [f32], npixels: usize) {
    // Divergent traversal (chunk iterator plus `any()` over lanes 0-2)
    // against the kernel's stride indexing plus `||` chain; same strict-`<`
    // trigger, same cyan value, same leave-clean-pixels-alone rule.
    let (quads, _) = out.as_chunks_mut::<4>();
    quads.iter_mut().take(npixels).for_each(|px| {
        if px[0..3].iter().any(|&v| v < 0.0) {
            px.copy_from_slice(&COLOROUT_GAMUTCHECK_CYAN);
        }
    });
}

/// # Safety
/// `out` must be non-null and valid for `npixels*4` floats. Null pointer,
/// `npixels == 0`, `4*npixels` overflow and `isize::MAX` byte-span overflow
/// are guarded (no-op return).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorout_gamutcheck_fill(out: *mut f32, npixels: usize) {
    if out.is_null() || npixels == 0 {
        return;
    }
    let Some(len) = npixels.checked_mul(4) else { return; };
    let Some(nbytes) = len.checked_mul(4) else { return; };
    if nbytes > isize::MAX as usize {
        return;
    }
    let out = std::slice::from_raw_parts_mut(out, len);
    colorout_gamutcheck_fill(out, npixels);
}
