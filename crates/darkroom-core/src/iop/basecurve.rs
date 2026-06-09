use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Basecurve;

impl IopProcess for Basecurve {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "basecurve" }
}

const LUT_SIZE: usize = 0x10000; // 65536

/// Integer-truncation LUT lookup matching table[CLAMP((int)(f*0x10000), 0, 0xffff)].
/// Output is floored at 0.
#[inline(always)]
fn lut_lookup(table: &[f32], f: f32) -> f32 {
    let idx = ((f * LUT_SIZE as f32) as i32).clamp(0, (LUT_SIZE - 1) as i32) as usize;
    table[idx].max(0.0)
}

/// Unbounded extrapolation: coeff[1] * pow(v * coeff[0], coeff[2]).
/// Matches dt_iop_eval_exp() in imageop_math.h.
#[inline(always)]
fn eval_exp(coeff: &[f32], v: f32) -> f32 {
    coeff[1] * (v * coeff[0]).powf(coeff[2])
}

/// Fast exp approximation matching dt_fast_expf() in common/math.h.
/// Valid for x in [-100, 0]; behaviour outside that range is intentionally imprecise.
#[inline(always)]
fn fast_expf(x: f32) -> f32 {
    const I1: f32 = 0x3f800000_u32 as f32;
    const SCALE: f32 = (0x402DF854_u32 - 0x3f800000_u32) as f32;
    let k0 = (I1 + x * SCALE) as i32;
    f32::from_bits(if k0 > 0 { k0 as u32 } else { 0 })
}

/// Per-channel tone curve (integer-truncation LUT) for the legacy no-preserve-colors path.
///
/// Matches apply_legacy_curve() in src/iop/basecurve.c.
/// table:            65536 floats — single shared LUT for all RGB channels.
/// unbounded_coeffs: 3 floats — [coeff0, coeff1, coeff2] for eval_exp extrapolation.
/// mul:              pre-scalar applied to every channel value before the LUT lookup.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_apply_legacy_curve(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mul: f32,
    table: *const f32,
    unbounded_coeffs: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lut = std::slice::from_raw_parts(table, LUT_SIZE);
    let coeffs = std::slice::from_raw_parts(unbounded_coeffs, 3);

    for k in 0..npixels {
        let base = k * 4;
        for i in 0..3 {
            let f = input[base + i] * mul;
            output[base + i] = if f < 1.0 {
                lut_lookup(lut, f)
            } else {
                eval_exp(coeffs, f).max(0.0)
            };
        }
        output[base + 3] = input[base + 3];
    }
}

/// Tone curve with colour preservation (the `preserve_colors != NONE` path).
///
/// Matches apply_curve() in src/iop/basecurve.c. Per pixel it computes
/// `lum = mul * dt_rgb_norm(rgb, preserve_colors, work_profile)`, maps it through
/// the shared curve (`table` for `lum < 1`, else `eval_exp(unbounded_coeffs, lum)`),
/// and scales all three channels by `mul * curve_lum / lum`. Alpha passes through.
///
/// Only the LUMINANCE norm (mode 1) consults the working ICC profile; the other
/// norms are profile-independent. Work-profile fields are passed flat and may be
/// NULL when `has_work_profile == 0` (LUMINANCE then falls back to the camera
/// primaries, matching dt_camera_rgb_luminance):
///   * `matrix_in`     — 16 floats (`[4][4]`; only the Y row is read)
///   * `lut0/lut1/lut2` — `lutsize` floats each (only read when `nonlinearlut != 0`)
///   * `unbounded_in`  — 9 floats (`[3][3]`; per-channel TRC extrapolation)
///
/// # Safety
/// Pointers must be valid for the documented lengths. The work-profile pointers
/// may be NULL only when `has_work_profile == 0` (and the luts only when
/// `nonlinearlut == 0`).
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_apply_curve(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mul: f32,
    preserve_colors: i32,
    table: *const f32,
    unbounded_coeffs: *const f32,
    has_work_profile: i32,
    matrix_in: *const f32,
    lut0: *const f32,
    lut1: *const f32,
    lut2: *const f32,
    unbounded_in: *const f32,
    lutsize: i32,
    nonlinearlut: i32,
) {
    use crate::color;
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tbl = std::slice::from_raw_parts(table, LUT_SIZE);
    let curve_ub = std::slice::from_raw_parts(unbounded_coeffs, 3);

    let has_wp = has_work_profile != 0;
    let nonlinear = nonlinearlut != 0;
    let lutsize = lutsize as usize;

    // Hoist work-profile views — constant across all pixels.
    let matrix = if has_wp {
        Some(&*(matrix_in as *const [[f32; 4]; 4]))
    } else {
        None
    };
    let trc = if has_wp && nonlinear {
        let l0 = std::slice::from_raw_parts(lut0, lutsize);
        let l1 = std::slice::from_raw_parts(lut1, lutsize);
        let l2 = std::slice::from_raw_parts(lut2, lutsize);
        let ub = std::slice::from_raw_parts(unbounded_in, 9);
        Some(([l0, l1, l2], [&ub[0..3], &ub[3..6], &ub[6..9]]))
    } else {
        None
    };

    for k in 0..npixels {
        let base = k * 4;
        let r = input[base];
        let g = input[base + 1];
        let b = input[base + 2];

        // dt_rgb_norm: only LUMINANCE (mode 1) depends on the work profile.
        let norm = if preserve_colors == 1 {
            match matrix {
                Some(m) => match trc {
                    Some((luts, ubc)) => {
                        color::get_rgb_matrix_luminance([r, g, b, 0.0], m, luts, ubc, lutsize, true)
                    }
                    // linear profile: Y-row dot product, no TRC.
                    None => m[1][0] * r + m[1][1] * g + m[1][2] * b,
                },
                // no work profile -> camera primaries (dt_camera_rgb_luminance).
                None => r * 0.2225045 + g * 0.7168786 + b * 0.0606169,
            }
        } else {
            color::rgb_norm(r, g, b, preserve_colors)
        };

        let lum = mul * norm;
        let mut ratio = 1.0f32;
        if lum > 0.0 {
            let curve_lum = if lum < 1.0 {
                // table[CLAMP((int)(lum * 0x10000), 0, 0xffff)]
                let idx = ((lum * LUT_SIZE as f32) as i32).clamp(0, (LUT_SIZE - 1) as i32) as usize;
                tbl[idx]
            } else {
                eval_exp(curve_ub, lum)
            };
            ratio = mul * curve_lum / lum;
        }
        output[base] = (ratio * r).max(0.0);
        output[base + 1] = (ratio * g).max(0.0);
        output[base + 2] = (ratio * b).max(0.0);
        output[base + 3] = input[base + 3];
    }
}

/// Compute per-pixel exposure-fusion features into the alpha channel in-place.
///
/// Matches compute_features() in src/iop/basecurve.c.
/// Writes sat * well_exposedness into buf[k*4+3] for every pixel k.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_compute_features(
    buf: *mut f32,
    npixels: usize,
) {
    let buf = std::slice::from_raw_parts_mut(buf, npixels * 4);

    for k in 0..npixels {
        let x = k * 4;
        let r = buf[x];
        let g = buf[x + 1];
        let b = buf[x + 2];

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let sat = 0.1_f32 + 0.1_f32 * (max - min) / max.max(1e-4_f32);

        const C: f32 = 0.54;
        let v = (r - C).abs().max((g - C).abs()).max((b - C).abs());
        const VAR_SQ: f32 = 0.5 * 0.5; // var = 0.5
        let exp_val = 0.2_f32 + fast_expf(-v * v / VAR_SQ);

        buf[x + 3] = sat * exp_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_lookup_identity_lut() {
        let lut: Vec<f32> = (0..LUT_SIZE).map(|k| k as f32 / LUT_SIZE as f32).collect();
        // f=0.5 → index = int(0.5 * 65536) = 32768 → lut[32768] = 32768/65536 = 0.5
        let out = lut_lookup(&lut, 0.5);
        assert!((out - 0.5).abs() < 1e-4, "out={out}");
    }

    #[test]
    fn lut_lookup_negative_clips_to_zero() {
        let lut = vec![0.42_f32; LUT_SIZE];
        assert_eq!(lut_lookup(&lut, -1.0), 0.42);
    }

    #[test]
    fn lut_lookup_floors_negative_lut_values() {
        let mut lut = vec![0.0_f32; LUT_SIZE];
        lut[0] = -0.5;
        assert_eq!(lut_lookup(&lut, 0.0), 0.0);
    }

    #[test]
    fn eval_exp_matches_formula() {
        let coeff = [2.0_f32, 3.0, 0.5];
        let v = 0.25_f32;
        let expected = 3.0 * (0.25 * 2.0_f32).powf(0.5);
        assert!((eval_exp(&coeff, v) - expected).abs() < 1e-5);
    }

    #[test]
    fn apply_legacy_curve_alpha_passthrough() {
        let lut: Vec<f32> = (0..LUT_SIZE).map(|k| k as f32 / LUT_SIZE as f32).collect();
        let coeffs = [1.0_f32, 1.0, 1.0];
        let input = vec![0.25_f32, 0.5, 0.75, 0.9999];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_basecurve_apply_legacy_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                lut.as_ptr(), coeffs.as_ptr(),
            );
        }
        assert!((out[0] - 0.25).abs() < 1e-3, "R={}", out[0]);
        assert!((out[1] - 0.5 ).abs() < 1e-3, "G={}", out[1]);
        assert!((out[2] - 0.75).abs() < 1e-3, "B={}", out[2]);
        assert_eq!(out[3], 0.9999); // alpha unchanged
    }

    #[test]
    fn apply_legacy_curve_unbounded_path() {
        // f >= 1.0 triggers eval_exp, result floored at 0
        let lut = vec![0.0_f32; LUT_SIZE];
        // coeff: coeff[1]*pow(v*coeff[0], coeff[2]) = 1*pow(2*1, 1) = 2
        let coeffs = [1.0_f32, 1.0, 1.0];
        let input = vec![2.0_f32, 2.0, 2.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_basecurve_apply_legacy_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                lut.as_ptr(), coeffs.as_ptr(),
            );
        }
        assert!((out[0] - 2.0).abs() < 1e-5, "R={}", out[0]);
    }

    #[test]
    fn compute_features_grey_pixel() {
        // For a grey pixel r=g=b=0.54: max==min → sat=0.1, v=0 → exp_val=0.2+fast_expf(0)≈1.2
        let mut buf = vec![0.54_f32, 0.54, 0.54, 0.0];
        unsafe { darkroom_basecurve_compute_features(buf.as_mut_ptr(), 1); }
        // sat = 0.1, exp_val ≈ 0.2 + 1.0 = 1.2 (fast_expf(0) ≈ 1.0)
        let alpha = buf[3];
        assert!(alpha > 0.0 && alpha < 1.0, "alpha={alpha}");
    }

    #[test]
    fn compute_features_does_not_touch_rgb() {
        let mut buf = vec![0.3_f32, 0.5, 0.7, 0.0];
        let orig = [buf[0], buf[1], buf[2]];
        unsafe { darkroom_basecurve_compute_features(buf.as_mut_ptr(), 1); }
        assert_eq!(buf[0], orig[0]);
        assert_eq!(buf[1], orig[1]);
        assert_eq!(buf[2], orig[2]);
    }

    // Identity curve table: table[i] = i / 65536, so curve_lum ≈ lum for lum < 1.
    fn identity_table() -> Vec<f32> {
        (0..LUT_SIZE).map(|i| i as f32 / LUT_SIZE as f32).collect()
    }

    #[test]
    fn apply_curve_identity_max_norm_is_passthrough() {
        let table = identity_table();
        let ub = [1.0f32, 1.0, 1.0];
        let input = [0.5f32, 0.25, 0.1, 0.9];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_basecurve_apply_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                2, // DT_RGB_NORM_MAX -> norm = 0.5; identity curve -> ratio 1
                table.as_ptr(), ub.as_ptr(),
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), 0, 0,
            );
        }
        assert!((out[0] - 0.5).abs() < 1e-3, "{out:?}");
        assert!((out[1] - 0.25).abs() < 1e-3);
        assert!((out[2] - 0.1).abs() < 1e-3);
        assert_eq!(out[3], 0.9);
    }

    #[test]
    fn apply_curve_luminance_no_profile_uses_camera_primaries() {
        let table = identity_table();
        let ub = [1.0f32, 1.0, 1.0];
        // grey 0.5: camera luminance = 0.5*(0.2225045+0.7168786+0.0606169) = 0.5
        let input = [0.5f32, 0.5, 0.5, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_basecurve_apply_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                1, // DT_RGB_NORM_LUMINANCE, no work profile -> camera primaries
                table.as_ptr(), ub.as_ptr(),
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), 0, 0,
            );
        }
        for c in 0..3 {
            assert!((out[c] - 0.5).abs() < 1e-3, "c={c} {out:?}");
        }
    }

    #[test]
    fn apply_curve_luminance_linear_profile_uses_matrix_y_row() {
        let table = identity_table();
        let ub = [1.0f32, 1.0, 1.0];
        // 4x4 matrix; Y row (row 1) = [0.2, 0.7, 0.1]. nonlinearlut = 0 (linear).
        let matrix: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.2, 0.7, 0.1, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let input = [0.5f32, 0.5, 0.5, 1.0]; // lum = 0.5*(0.2+0.7+0.1) = 0.5
        let mut out = [0f32; 4];
        unsafe {
            darkroom_basecurve_apply_curve(
                input.as_ptr(), out.as_mut_ptr(), 1, 1.0,
                1, table.as_ptr(), ub.as_ptr(),
                1, matrix.as_ptr(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), 0, 0,
            );
        }
        for c in 0..3 {
            assert!((out[c] - 0.5).abs() < 1e-3, "c={c} {out:?}");
        }
    }
}

// ── Gaussian pyramid (Phase 2z+56) ───────────────────────────────────────

/// Mirror-reflect index outside [0, max] (half-sample boundary convention).
/// Left: -x;  Right: 2*max+1-x.  Matches basecurve.c gauss_blur borders.
#[inline(always)]
fn bc_mirror(x: i32, max: i32) -> usize {
    if x < 0 { (-x) as usize }
    else if x > max { (2 * max + 1 - x) as usize }
    else { x as usize }
}

/// 5-tap separable Gaussian blur for a 4-channel RGBA image.
/// Kernel: [1/16, 4/16, 6/16, 4/16, 1/16]. In-place safe.
/// Matches the two DT_OMP_FOR loops in basecurve.c::gauss_blur().
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_gauss_blur(
    input:  *const f32,
    output: *mut f32,
    wd:     usize,
    ht:     usize,
) {
    const W: [f32; 5] = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0];
    let n   = wd * ht * 4;
    let inp = std::slice::from_raw_parts(input, n);
    let out = std::slice::from_raw_parts_mut(output, n);
    let wdi = wd as i32;
    let hti = ht as i32;

    // Horizontal pass into temp buffer
    let mut tmp = vec![0.0f32; n];
    for j in 0..ht {
        for i in 0..wd {
            let b = (j * wd + i) * 4;
            for c in 0..4 {
                let mut s = 0.0f32;
                for ii in -2i32..=2 {
                    let si = bc_mirror(i as i32 + ii, wdi - 1);
                    s += inp[(j * wd + si) * 4 + c] * W[(ii + 2) as usize];
                }
                tmp[b + c] = s;
            }
        }
    }

    // Vertical pass into output
    for j in 0..ht {
        for i in 0..wd {
            let b = (j * wd + i) * 4;
            for c in 0..4 {
                let mut s = 0.0f32;
                for jj in -2i32..=2 {
                    let sj = bc_mirror(j as i32 + jj, hti - 1);
                    s += tmp[(sj * wd + i) * 4 + c] * W[(jj + 2) as usize];
                }
                out[b + c] = s;
            }
        }
    }
}

/// Gaussian pyramid upsampling: fill even pixels from coarse (×4), blur.
/// Matches gauss_expand() in basecurve.c (DT_OMP_FOR(collapse(2)) + gauss_blur).
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_gauss_expand(
    coarse: *const f32,
    fine:   *mut f32,
    wd:     usize,
    ht:     usize,
) {
    let cw  = (wd - 1) / 2 + 1;
    let ch  = (ht - 1) / 2 + 1;
    let n   = wd * ht * 4;
    let fin = std::slice::from_raw_parts_mut(fine, n);
    let crs = std::slice::from_raw_parts(coarse, cw * ch * 4);

    fin.fill(0.0);
    for j in (0..ht).step_by(2) {
        for i in (0..wd).step_by(2) {
            let cb = (j / 2 * cw + i / 2) * 4;
            let fb = (j * wd + i) * 4;
            for c in 0..4 { fin[fb + c] = 4.0 * crs[cb + c]; }
        }
    }
    darkroom_basecurve_gauss_blur(fine as *const f32, fine, wd, ht);
}

/// Weight update: col0_alpha *= 0.1 + ||out_rgb||. Matches basecurve.c:1193.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_weight_update(
    col0:    *mut f32,
    out_buf: *const f32,
    npixels: usize,
) {
    let col = std::slice::from_raw_parts_mut(col0, npixels * 4);
    let out = std::slice::from_raw_parts(out_buf, npixels * 4);
    for k in (0..npixels * 4).step_by(4) {
        let mag = (out[k]*out[k] + out[k+1]*out[k+1] + out[k+2]*out[k+2]).sqrt();
        col[k + 3] *= 0.1 + mag;
    }
}

/// Blend one pyramid level into comb_k.
/// is_base!=0: comb[c] += w*col[c];  is_base==0: comb[c] += w*(col[c]-out[c]).
/// comb[3] += w always. Matches basecurve.c:1229.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_pyramid_blend(
    comb_k:  *mut f32,
    col_k:   *const f32,
    out_buf: *const f32,
    npixels: usize,
    is_base: i32,
) {
    let comb = std::slice::from_raw_parts_mut(comb_k, npixels * 4);
    let col  = std::slice::from_raw_parts(col_k,  npixels * 4);
    let out  = std::slice::from_raw_parts(out_buf, npixels * 4);
    for x in (0..npixels * 4).step_by(4) {
        let w = col[x + 3];
        if is_base != 0 {
            for c in 0..3 { comb[x+c] += w * col[x+c]; }
        } else {
            for c in 0..3 { comb[x+c] += w * (col[x+c] - out[x+c]); }
        }
        comb[x + 3] += w;
    }
}

/// Normalize RGB by alpha: if comb[x+3] > 1e-8, divide comb[x..x+3] by it.
/// Matches basecurve.c:1265.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_normalize_alpha(
    comb_k:  *mut f32,
    npixels: usize,
) {
    let comb = std::slice::from_raw_parts_mut(comb_k, npixels * 4);
    for x in (0..npixels * 4).step_by(4) {
        let a = comb[x + 3];
        if a > 1e-8 { for c in 0..3 { comb[x+c] /= a; } }
    }
}

/// Add expanded coarser level to comb_k RGB. Matches basecurve.c:1273.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_add_layers(
    comb_k:  *mut f32,
    out_buf: *const f32,
    npixels: usize,
) {
    let comb = std::slice::from_raw_parts_mut(comb_k, npixels * 4);
    let out  = std::slice::from_raw_parts(out_buf, npixels * 4);
    for x in (0..npixels * 4).step_by(4) {
        for c in 0..3 { comb[x+c] += out[x+c]; }
    }
}

/// Copy comb0 RGB (clamped ≥ 0) + in alpha to output. Matches basecurve.c:1283.
#[no_mangle]
pub unsafe extern "C" fn darkroom_basecurve_copy_output(
    comb0:   *const f32,
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    let comb = std::slice::from_raw_parts(comb0,   npixels * 4);
    let inp  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out  = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in (0..npixels * 4).step_by(4) {
        for c in 0..3 { out[k+c] = comb[k+c].max(0.0); }
        out[k + 3] = inp[k + 3];
    }
}
