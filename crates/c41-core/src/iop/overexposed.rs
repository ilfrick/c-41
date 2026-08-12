use crate::{color, params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Overexposed;

impl IopProcess for Overexposed {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "overexposed" }
}

/// Highlight clipped pixels with a tint colour, for the per-channel
/// "any RGB" preview mode of the overexposed IOP.
///
/// For each pixel k:
///   if any of R, G, B in img_tmp >= upper        → out[k] = upper_color
///   else if R, G, B in img_tmp all <= lower      → out[k] = lower_color
///   else                                         → out[k] = in[k]
///
/// `in`, `out`, `img_tmp` are tightly-packed RGBA float buffers of length
/// `npixels * 4`. `upper_color` and `lower_color` are 4-float arrays.
///
/// Matches the DT_CLIPPING_PREVIEW_ANYRGB branch in
/// src/iop/overexposed.c (process()).
#[no_mangle]
pub unsafe extern "C" fn darkroom_overexposed_anyrgb(
    in_buf: *const f32,
    out_buf: *mut f32,
    img_tmp: *const f32,
    npixels: usize,
    upper: f32,
    lower: f32,
    upper_color: *const f32,
    lower_color: *const f32,
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tmp = std::slice::from_raw_parts(img_tmp, npixels * 4);
    let upc = std::slice::from_raw_parts(upper_color, 4);
    let loc = std::slice::from_raw_parts(lower_color, 4);

    for k in 0..npixels {
        let i = k * 4;
        let r = tmp[i];
        let g = tmp[i + 1];
        let b = tmp[i + 2];

        if r >= upper || g >= upper || b >= upper {
            out[i..i + 4].copy_from_slice(upc);
        } else if r <= lower && g <= lower && b <= lower {
            out[i..i + 4].copy_from_slice(loc);
        } else {
            out[i..i + 4].copy_from_slice(&inp[i..i + 4]);
        }
    }
}

/// Highlight pixels whose work-profile luminance falls outside `[lower, upper]`.
///
/// Matches `DT_CLIPPING_PREVIEW_LUMINANCE` in src/iop/overexposed.c.
///
/// For each pixel `k`:
///   y = work_profile_luminance(img_tmp[k])
///   if y >= upper          → out[k] = upper_color
///   else if y <= lower     → out[k] = lower_color
///   else                   → out[k] = in[k]
///
/// `matrix_in` is the working profile's 4x4 colour matrix (only row 1 is read).
/// `lut0/1/2` are the three per-channel TRC LUTs, each `lutsize` floats long.
/// `unbounded_coeffs` is `[3][3]` flattened to 9 floats.
/// `nonlinear_lut` toggles the linearisation pre-step.
///
/// # Safety
/// All buffer pointers must be valid for the indicated extents and non-aliasing.
#[no_mangle]
pub unsafe extern "C" fn darkroom_overexposed_luminance(
    in_buf: *const f32,
    out_buf: *mut f32,
    img_tmp: *const f32,
    npixels: usize,
    upper: f32,
    lower: f32,
    upper_color: *const f32,
    lower_color: *const f32,
    matrix_in: *const f32,            // 4*4 = 16 floats
    lut0: *const f32,
    lut1: *const f32,
    lut2: *const f32,
    lutsize: usize,
    unbounded_coeffs: *const f32,      // 3*3 = 9 floats
    nonlinear_lut: i32,
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tmp = std::slice::from_raw_parts(img_tmp, npixels * 4);
    let upc = std::slice::from_raw_parts(upper_color, 4);
    let loc = std::slice::from_raw_parts(lower_color, 4);

    // Rebuild the 4x4 matrix (kept as native [[f32;4];4] for clarity).
    let m_slice = std::slice::from_raw_parts(matrix_in, 16);
    let mut m = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = m_slice[r * 4 + c];
        }
    }
    let luts = [
        std::slice::from_raw_parts(lut0, lutsize),
        std::slice::from_raw_parts(lut1, lutsize),
        std::slice::from_raw_parts(lut2, lutsize),
    ];
    let coeffs_slice = std::slice::from_raw_parts(unbounded_coeffs, 9);
    let coeffs = [
        &coeffs_slice[0..3],
        &coeffs_slice[3..6],
        &coeffs_slice[6..9],
    ];
    let nonlinear = nonlinear_lut != 0;

    for k in 0..npixels {
        let i = k * 4;
        let pixel = [tmp[i], tmp[i + 1], tmp[i + 2], tmp[i + 3]];
        let y = color::get_rgb_matrix_luminance(
            pixel, &m, [luts[0], luts[1], luts[2]],
            [coeffs[0], coeffs[1], coeffs[2]],
            lutsize, nonlinear,
        );

        if y >= upper {
            out[i..i + 4].copy_from_slice(upc);
        } else if y <= lower {
            out[i..i + 4].copy_from_slice(loc);
        } else {
            out[i..i + 4].copy_from_slice(&inp[i..i + 4]);
        }
    }
}

/// Per-channel "relative saturation" used by the GAMUT and SATURATION modes.
///
/// For each colour channel c:
///   d = pixel[c] - luminance
///   sat[c] = sqrt(d^2 / (luminance^2 + pixel[c]^2))
///
/// Matches the per-pixel inner block in src/iop/overexposed.c (process()).
#[inline(always)]
fn saturation3(pixel: [f32; 4], luminance: f32) -> [f32; 3] {
    let mut s = [0.0_f32; 3];
    for c in 0..3 {
        let d = pixel[c] - luminance;
        let denom = luminance * luminance + pixel[c] * pixel[c];
        s[c] = (d * d / denom).sqrt();
    }
    s
}

#[inline(always)]
fn any_above(rgb: [f32; 4], thr: f32) -> bool {
    rgb[0] >= thr || rgb[1] >= thr || rgb[2] >= thr
}

#[inline(always)]
fn all_below_or_eq(rgb: [f32; 4], thr: f32) -> bool {
    rgb[0] <= thr && rgb[1] <= thr && rgb[2] <= thr
}

/// Gamut-out-of-bounds clipping preview.
///
/// Matches `DT_CLIPPING_PREVIEW_GAMUT` in src/iop/overexposed.c. For each pixel:
///   y = work_profile_luminance(tmp[k])
///   if y >= upper    → upper_color
///   else if y <= lower → lower_color
///   else
///     sat = saturation3(tmp[k], y)
///     if any sat[c] > upper OR any tmp[c] >= upper  → upper_color
///     else if all tmp[c] <= lower                   → lower_color
///     else                                          → in[k]
#[no_mangle]
pub unsafe extern "C" fn darkroom_overexposed_gamut(
    in_buf: *const f32,
    out_buf: *mut f32,
    img_tmp: *const f32,
    npixels: usize,
    upper: f32,
    lower: f32,
    upper_color: *const f32,
    lower_color: *const f32,
    matrix_in: *const f32,
    lut0: *const f32,
    lut1: *const f32,
    lut2: *const f32,
    lutsize: usize,
    unbounded_coeffs: *const f32,
    nonlinear_lut: i32,
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tmp = std::slice::from_raw_parts(img_tmp, npixels * 4);
    let upc = std::slice::from_raw_parts(upper_color, 4);
    let loc = std::slice::from_raw_parts(lower_color, 4);

    let m_slice = std::slice::from_raw_parts(matrix_in, 16);
    let mut m = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 { m[r][c] = m_slice[r * 4 + c]; }
    }
    let luts = [
        std::slice::from_raw_parts(lut0, lutsize),
        std::slice::from_raw_parts(lut1, lutsize),
        std::slice::from_raw_parts(lut2, lutsize),
    ];
    let coeffs_slice = std::slice::from_raw_parts(unbounded_coeffs, 9);
    let coeffs = [
        &coeffs_slice[0..3], &coeffs_slice[3..6], &coeffs_slice[6..9],
    ];
    let nonlinear = nonlinear_lut != 0;

    for k in 0..npixels {
        let i = k * 4;
        let pixel = [tmp[i], tmp[i + 1], tmp[i + 2], tmp[i + 3]];
        let y = color::get_rgb_matrix_luminance(
            pixel, &m, [luts[0], luts[1], luts[2]],
            [coeffs[0], coeffs[1], coeffs[2]], lutsize, nonlinear,
        );

        if y >= upper {
            out[i..i + 4].copy_from_slice(upc);
        } else if y <= lower {
            out[i..i + 4].copy_from_slice(loc);
        } else {
            let sat = saturation3(pixel, y);
            let sat_out = sat[0] > upper || sat[1] > upper || sat[2] > upper;
            if sat_out || any_above(pixel, upper) {
                out[i..i + 4].copy_from_slice(upc);
            } else if all_below_or_eq(pixel, lower) {
                out[i..i + 4].copy_from_slice(loc);
            } else {
                out[i..i + 4].copy_from_slice(&inp[i..i + 4]);
            }
        }
    }
}

/// Saturation-out-of-bounds clipping preview.
///
/// Matches `DT_CLIPPING_PREVIEW_SATURATION` in src/iop/overexposed.c. For each
/// pixel, runs the same saturation/RGB clipping test as GAMUT — but only
/// when the luminance falls inside (lower, upper). When the pixel is too
/// dark or too bright by luminance, the original input is passed through.
#[no_mangle]
pub unsafe extern "C" fn darkroom_overexposed_saturation(
    in_buf: *const f32,
    out_buf: *mut f32,
    img_tmp: *const f32,
    npixels: usize,
    upper: f32,
    lower: f32,
    upper_color: *const f32,
    lower_color: *const f32,
    matrix_in: *const f32,
    lut0: *const f32,
    lut1: *const f32,
    lut2: *const f32,
    lutsize: usize,
    unbounded_coeffs: *const f32,
    nonlinear_lut: i32,
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tmp = std::slice::from_raw_parts(img_tmp, npixels * 4);
    let upc = std::slice::from_raw_parts(upper_color, 4);
    let loc = std::slice::from_raw_parts(lower_color, 4);

    let m_slice = std::slice::from_raw_parts(matrix_in, 16);
    let mut m = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 { m[r][c] = m_slice[r * 4 + c]; }
    }
    let luts = [
        std::slice::from_raw_parts(lut0, lutsize),
        std::slice::from_raw_parts(lut1, lutsize),
        std::slice::from_raw_parts(lut2, lutsize),
    ];
    let coeffs_slice = std::slice::from_raw_parts(unbounded_coeffs, 9);
    let coeffs = [
        &coeffs_slice[0..3], &coeffs_slice[3..6], &coeffs_slice[6..9],
    ];
    let nonlinear = nonlinear_lut != 0;

    for k in 0..npixels {
        let i = k * 4;
        let pixel = [tmp[i], tmp[i + 1], tmp[i + 2], tmp[i + 3]];
        let y = color::get_rgb_matrix_luminance(
            pixel, &m, [luts[0], luts[1], luts[2]],
            [coeffs[0], coeffs[1], coeffs[2]], lutsize, nonlinear,
        );

        if y < upper && y > lower {
            let sat = saturation3(pixel, y);
            let sat_out = sat[0] > upper || sat[1] > upper || sat[2] > upper;
            if sat_out || any_above(pixel, upper) {
                out[i..i + 4].copy_from_slice(upc);
            } else if all_below_or_eq(pixel, lower) {
                out[i..i + 4].copy_from_slice(loc);
            } else {
                out[i..i + 4].copy_from_slice(&inp[i..i + 4]);
            }
        } else {
            out[i..i + 4].copy_from_slice(&inp[i..i + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upper_clip_paints_upper_color() {
        let inp = vec![0.0_f32; 4];
        let tmp = vec![1.5_f32, 0.0, 0.0, 1.0]; // R out of bounds
        let mut out = vec![-1.0_f32; 4];
        let upc = [0.9_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 0.9, 1.0];
        unsafe {
            darkroom_overexposed_anyrgb(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                1.0, 0.01, upc.as_ptr(), loc.as_ptr(),
            );
        }
        assert_eq!(&out[..], &upc[..]);
    }

    #[test]
    fn lower_clip_paints_lower_color() {
        let inp = vec![0.5_f32, 0.5, 0.5, 1.0];
        let tmp = vec![0.001_f32, 0.001, 0.001, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let upc = [0.9_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 0.9, 1.0];
        unsafe {
            darkroom_overexposed_anyrgb(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                1.0, 0.01, upc.as_ptr(), loc.as_ptr(),
            );
        }
        assert_eq!(&out[..], &loc[..]);
    }

    #[test]
    fn unclipped_pixel_uses_in_buf() {
        let inp = vec![0.3_f32, 0.4, 0.5, 0.9];
        let tmp = vec![0.5_f32, 0.5, 0.5, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let upc = [0.9_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 0.9, 1.0];
        unsafe {
            darkroom_overexposed_anyrgb(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                1.0, 0.01, upc.as_ptr(), loc.as_ptr(),
            );
        }
        assert_eq!(&out[..], &inp[..]);
    }

    #[test]
    fn upper_takes_priority_over_lower() {
        // R is huge, G & B are tiny — upper triggers first (matches C ordering)
        let inp = vec![0.5_f32; 4];
        let tmp = vec![2.0_f32, 0.0, 0.0, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let upc = [1.0_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 1.0, 1.0];
        unsafe {
            darkroom_overexposed_anyrgb(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                1.0, 0.5, upc.as_ptr(), loc.as_ptr(),
            );
        }
        assert_eq!(&out[..], &upc[..]);
    }

    fn ident_matrix() -> [f32; 16] {
        let mut m = [0.0_f32; 16];
        for i in 0..4 { m[i * 4 + i] = 1.0; }
        m
    }

    #[test]
    fn luminance_mode_linear_passthrough_unclipped() {
        // Identity matrix → luminance = G channel. Mid-grey: y=0.5, between
        // [0.1, 0.9] thresholds, so pixel should pass through from `inp`.
        let inp = vec![0.3_f32, 0.5, 0.7, 1.0];
        let tmp = vec![0.5_f32, 0.5, 0.5, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let m = ident_matrix();
        let lut = vec![-1.0_f32; 4]; // disabled
        let coeffs = [0.0_f32; 9];
        unsafe {
            darkroom_overexposed_luminance(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                0.9, 0.1,
                [0.9_f32, 0.0, 0.0, 1.0].as_ptr(),
                [0.0_f32, 0.0, 0.9, 1.0].as_ptr(),
                m.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(), 4,
                coeffs.as_ptr(), 0,
            );
        }
        assert_eq!(&out[..], &inp[..]);
    }

    #[test]
    fn luminance_mode_upper_clip() {
        let inp = vec![0.0_f32; 4];
        let tmp = vec![1.0_f32, 1.0, 1.0, 1.0]; // y=1 > 0.9 → upper
        let mut out = vec![-1.0_f32; 4];
        let m = ident_matrix();
        let lut = vec![-1.0_f32; 4];
        let coeffs = [0.0_f32; 9];
        let upc = [0.9_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 0.9, 1.0];
        unsafe {
            darkroom_overexposed_luminance(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                0.9, 0.1, upc.as_ptr(), loc.as_ptr(),
                m.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(), 4,
                coeffs.as_ptr(), 0,
            );
        }
        assert_eq!(&out[..], &upc[..]);
    }

    #[test]
    fn luminance_mode_lower_clip() {
        let inp = vec![0.5_f32; 4];
        let tmp = vec![0.0_f32, 0.0, 0.0, 1.0]; // y=0 < 0.1 → lower
        let mut out = vec![-1.0_f32; 4];
        let m = ident_matrix();
        let lut = vec![-1.0_f32; 4];
        let coeffs = [0.0_f32; 9];
        let upc = [0.9_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 0.9, 1.0];
        unsafe {
            darkroom_overexposed_luminance(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                0.9, 0.1, upc.as_ptr(), loc.as_ptr(),
                m.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(), 4,
                coeffs.as_ptr(), 0,
            );
        }
        assert_eq!(&out[..], &loc[..]);
    }

    #[test]
    fn gamut_mode_paints_upper_when_rgb_clips_through_saturation_path() {
        // Mid-luminance pixel with one channel >= upper triggers the inner
        // "any_above(pixel, upper)" branch → upper_color.
        let inp = vec![0.5_f32; 4];
        // y(matrix=ident, lut disabled) = G = 0.5; R = 1.2 (>= upper=1.0)
        let tmp = vec![1.2_f32, 0.5, 0.5, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let m = ident_matrix();
        let lut = vec![-1.0_f32; 4];
        let coeffs = [0.0_f32; 9];
        let upc = [1.0_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 1.0, 1.0];
        unsafe {
            darkroom_overexposed_gamut(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                1.0, 0.01, upc.as_ptr(), loc.as_ptr(),
                m.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(), 4,
                coeffs.as_ptr(), 0,
            );
        }
        assert_eq!(&out[..], &upc[..]);
    }

    #[test]
    fn gamut_mode_luminance_clip_short_circuits() {
        // luminance >= upper → upper_color directly, saturation branch skipped
        let inp = vec![0.0_f32; 4];
        let tmp = vec![1.5_f32, 1.5, 1.5, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let m = ident_matrix();
        let lut = vec![-1.0_f32; 4];
        let coeffs = [0.0_f32; 9];
        let upc = [0.9_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 0.9, 1.0];
        unsafe {
            darkroom_overexposed_gamut(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                1.0, 0.01, upc.as_ptr(), loc.as_ptr(),
                m.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(), 4,
                coeffs.as_ptr(), 0,
            );
        }
        assert_eq!(&out[..], &upc[..]);
    }

    #[test]
    fn saturation_mode_passes_through_when_luminance_out_of_range() {
        // y = 1.5 > upper=1.0 → bypass saturation test, copy inp
        let inp = vec![0.3_f32, 0.5, 0.7, 1.0];
        let tmp = vec![1.5_f32, 1.5, 1.5, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let m = ident_matrix();
        let lut = vec![-1.0_f32; 4];
        let coeffs = [0.0_f32; 9];
        let upc = [0.9_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 0.9, 1.0];
        unsafe {
            darkroom_overexposed_saturation(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                1.0, 0.01, upc.as_ptr(), loc.as_ptr(),
                m.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(), 4,
                coeffs.as_ptr(), 0,
            );
        }
        assert_eq!(&out[..], &inp[..]);
    }

    #[test]
    fn saturation_mode_triggers_upper_on_high_relative_saturation() {
        // luminance is in range, but R deviates a lot from luminance → sat[0]
        // is large → exceeds upper.
        // y = G = 0.5. R = 0.5, B = 0.0 → sat[2] = sqrt(0.25 / (0.25 + 0)) = 1
        let inp = vec![0.2_f32, 0.3, 0.4, 1.0];
        let tmp = vec![0.5_f32, 0.5, 0.0, 1.0];
        let mut out = vec![-1.0_f32; 4];
        let m = ident_matrix();
        let lut = vec![-1.0_f32; 4];
        let coeffs = [0.0_f32; 9];
        let upc = [0.9_f32, 0.0, 0.0, 1.0];
        let loc = [0.0_f32, 0.0, 0.9, 1.0];
        unsafe {
            darkroom_overexposed_saturation(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 1,
                0.95, 0.01, upc.as_ptr(), loc.as_ptr(),
                m.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), lut.as_ptr(), 4,
                coeffs.as_ptr(), 0,
            );
        }
        assert_eq!(&out[..], &upc[..]);
    }

    #[test]
    fn multi_pixel_mixed_outcomes() {
        // 3 pixels: clip-upper, unclipped, clip-lower
        let inp = vec![
            0.0, 0.0, 0.0, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.5, 0.5, 0.5, 1.0,
        ];
        let tmp = vec![
            1.5, 0.0, 0.0, 1.0,
            0.5, 0.5, 0.5, 1.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut out = vec![-1.0_f32; 12];
        let upc = [1.0_f32, 0.1, 0.1, 1.0];
        let loc = [0.1_f32, 0.1, 1.0, 1.0];
        unsafe {
            darkroom_overexposed_anyrgb(
                inp.as_ptr(), out.as_mut_ptr(), tmp.as_ptr(), 3,
                1.0, 0.01, upc.as_ptr(), loc.as_ptr(),
            );
        }
        assert_eq!(&out[0..4], &upc[..]);
        assert_eq!(&out[4..8], &inp[4..8]);
        assert_eq!(&out[8..12], &loc[..]);
    }
}
