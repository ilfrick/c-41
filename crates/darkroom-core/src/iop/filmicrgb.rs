use crate::{params::IopParams, roi::RoiIn, Result};
use crate::color;
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

/// Add statistical noise to highlights to seed the wavelet reconstruction.
///
/// For each pixel (i, j):
///   seed xoshiro128+ from (i, j) → 4 warm-up rounds
///   sigma[c] = pix_in[c] * noise_level / threshold
///   noise = dt_noise_generator_simd(dist, pix_in, sigma, flip=[T,F,T,F])
///   pix_out[c] = max(pix_in[c]*(1-weight) + weight*noise[c], 0)
///
/// `noise_distribution`: 0 = uniform, 1 = gaussian, 2 = poissonian.
/// Matches `inpaint_noise()` in src/iop/filmicrgb.c:1062.
#[no_mangle]
pub unsafe extern "C" fn darkroom_filmicrgb_inpaint_noise(
    in_buf:     *const f32,
    mask_buf:   *const f32,
    inpainted:  *mut f32,
    noise_level: f32,
    threshold:   f32,
    noise_distribution: u32,
    width:  usize,
    height: usize,
) {
    let npx = width * height;
    if npx == 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf,   npx * 4);
    let mask = std::slice::from_raw_parts(mask_buf, npx);
    let out  = std::slice::from_raw_parts_mut(inpainted, npx * 4);

    const FLIP: [bool; 4] = [true, false, true, false];

    for i in 0..height {
        for j in 0..width {
            let mut state = [
                crate::math::splitmix32((j + 1) as u64),
                crate::math::splitmix32(((j + 1) * (i + 3)) as u64),
                crate::math::splitmix32(1337),
                crate::math::splitmix32(666),
            ];
            // 4 warm-up rounds
            for _ in 0..4 { crate::math::xoshiro128plus(&mut state); }

            let idx   = i * width + j;
            let index = idx * 4;
            let weight = mask[idx];
            let mu: [f32; 4] = [inp[index], inp[index+1], inp[index+2], inp[index+3]];

            let thr = threshold.max(1e-6); // avoid division by zero
            let sigma: [f32; 4] = [
                mu[0] * noise_level / thr,
                mu[1] * noise_level / thr,
                mu[2] * noise_level / thr,
                mu[3] * noise_level / thr,
            ];

            let noise = crate::math::dt_noise_generator_4ch(
                noise_distribution, &mu, &sigma, &FLIP, &mut state,
            );

            for c in 0..4 {
                out[index + c] = (mu[c] * (1.0 - weight) + weight * noise[c]).max(0.0);
            }
        }
    }
}

// ── Tone-mapping primitives (ports of filmicrgb.c scalar helpers) ─────────────

/// norm can't be < 2^(-16) — matches NORM_MIN in src/common/math.h.
pub const NORM_MIN: f32 = 1.52587890625e-05;

/// Clamp to [0, 1] — matches clamp_simd() in src/develop/openmp_maths.h.
#[inline(always)]
fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// Per-channel `x**p` via `2^(log2(x)*p)`, matching dt_vector_powf()
/// (src/common/math.h). For `x == 0` this yields 0 for any `p > 0`.
#[inline(always)]
fn powf_log2(x: f32, p: f32) -> f32 {
    (x.log2() * p).exp2()
}

/// log tone-mapping v1: `CLAMP((log2(x/grey) - black)/range, NORM_MIN, 1)`.
#[inline(always)]
fn log_tonemapping_v1(x: f32, grey: f32, black: f32, dynamic_range: f32) -> f32 {
    let temp = ((x / grey).log2() - black) / dynamic_range;
    temp.max(NORM_MIN).min(1.0)
}

/// log tone-mapping v2 per-channel: `clamp01((log2(x/grey) - black)/range)`.
#[inline(always)]
fn log_tonemapping_v2(x: f32, grey: f32, black: f32, dynamic_range: f32) -> f32 {
    clamp01(((x / grey).log2() - black) / dynamic_range)
}

/// Desaturation coefficient v1 — matches filmic_desaturate_v1().
#[inline(always)]
fn filmic_desaturate_v1(x: f32, sigma_toe: f32, sigma_shoulder: f32, saturation: f32) -> f32 {
    let radius_toe = x;
    let radius_shoulder = 1.0 - x;
    let key_toe = (-0.5 * radius_toe * radius_toe / sigma_toe).exp();
    let key_shoulder = (-0.5 * radius_shoulder * radius_shoulder / sigma_shoulder).exp();
    1.0 - clamp01((key_toe + key_shoulder) / saturation)
}

/// Desaturation coefficient v2 — matches filmic_desaturate_v2().
#[inline(always)]
fn filmic_desaturate_v2(x: f32, sigma_toe: f32, sigma_shoulder: f32, saturation: f32) -> f32 {
    let radius_toe = x;
    let radius_shoulder = 1.0 - x;
    let sat2 = 0.5 / saturation.sqrt();
    let key_toe = (-radius_toe * radius_toe / sigma_toe * sat2).exp();
    let key_shoulder = (-radius_shoulder * radius_shoulder / sigma_shoulder * sat2).exp();
    saturation - (key_toe + key_shoulder) * saturation
}

/// Linear interpolation toward luminance — matches linear_saturation().
#[inline(always)]
fn linear_saturation(x: f32, luminance: f32, saturation: f32) -> f32 {
    luminance + saturation * (x - luminance)
}

// Curve types (dt_iop_filmicrgb_curve_type_t): 0 = POLY_4, 1 = POLY_3, 2 = RATIONAL.
const CURVE_POLY_4: i32 = 0;
const CURVE_POLY_3: i32 = 1;

/// Evaluate the filmic spline at `x` — matches filmic_spline() in filmicrgb.c.
/// `m1..m5` hold the factor vectors (index 0 = toe, 1 = shoulder, 2 = latitude).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn filmic_spline(
    x: f32,
    m1: &[f32], m2: &[f32], m3: &[f32], m4: &[f32], m5: &[f32],
    latitude_min: f32, latitude_max: f32, type0: i32, type1: i32,
) -> f32 {
    if x < latitude_min {
        if type0 == CURVE_POLY_4 {
            m1[0] + x * (m2[0] + x * (m3[0] + x * (m4[0] + x * m5[0])))
        } else if type0 == CURVE_POLY_3 {
            m1[0] + x * (m2[0] + x * (m3[0] + x * m4[0]))
        } else {
            let xi = latitude_min - x;
            let rat = xi * (xi * m2[0] + 1.0);
            m4[0] - m1[0] * rat / (rat + m3[0])
        }
    } else if x > latitude_max {
        if type1 == CURVE_POLY_4 {
            m1[1] + x * (m2[1] + x * (m3[1] + x * (m4[1] + x * m5[1])))
        } else if type1 == CURVE_POLY_3 {
            m1[1] + x * (m2[1] + x * (m3[1] + x * m4[1]))
        } else {
            let xi = x - latitude_max;
            let rat = xi * (xi * m2[1] + 1.0);
            m4[1] + m1[1] * rat / (rat + m3[1])
        }
    } else {
        m1[2] + x * m2[2]
    }
}

/// Borrowed view of the work-profile fields needed for luminance.
struct WorkProfile<'a> {
    matrix: &'a [[f32; 4]; 4],
    trc: Option<([&'a [f32]; 3], [&'a [f32]; 3])>,
    lutsize: usize,
}

/// Relative luminance under the working profile, with camera-primary fallback
/// when absent — matches dt_ioppr_get_rgb_matrix_luminance / dt_camera_rgb_luminance.
#[inline(always)]
fn luminance(rgb: [f32; 4], wp: &Option<WorkProfile>) -> f32 {
    match wp {
        Some(p) => match &p.trc {
            Some((luts, ubc)) => color::get_rgb_matrix_luminance(
                rgb, p.matrix, [luts[0], luts[1], luts[2]], [ubc[0], ubc[1], ubc[2]], p.lutsize, true,
            ),
            None => p.matrix[1][0] * rgb[0] + p.matrix[1][1] * rgb[1] + p.matrix[1][2] * rgb[2],
        },
        None => rgb[0] * 0.2225045 + rgb[1] * 0.7168786 + rgb[2] * 0.0606169,
    }
}

/// Build a `WorkProfile` view from raw FFI pointers (or `None`). The lut/coeff
/// slices are only materialised when `nonlinear`, mirroring the C guarantee.
///
/// # Safety
/// When `has_wp`, `matrix_in` must point to 16 contiguous floats (`[4][4]`).
/// When `nonlinear`, `lut0/lut1/lut2` must each point to `lutsize` floats
/// (`lutsize >= 2`, required by `extrapolate_lut`) and `unbounded_in` to 9
/// contiguous floats in row-major `[c][k]` order (per-channel `eval_exp` coeffs).
#[allow(clippy::too_many_arguments)]
unsafe fn make_work_profile<'a>(
    has_wp: bool, matrix_in: *const f32,
    lut0: *const f32, lut1: *const f32, lut2: *const f32, unbounded_in: *const f32,
    lutsize: usize, nonlinear: bool,
) -> Option<WorkProfile<'a>> {
    if !has_wp {
        return None;
    }
    let matrix = &*(matrix_in as *const [[f32; 4]; 4]);
    let trc = if nonlinear {
        // extrapolate_lut indexes lut[lutsize-1]; guard the future footgun.
        debug_assert!(lutsize >= 2, "nonlinear work profile needs lutsize >= 2");
        let l0 = std::slice::from_raw_parts(lut0, lutsize);
        let l1 = std::slice::from_raw_parts(lut1, lutsize);
        let l2 = std::slice::from_raw_parts(lut2, lutsize);
        let ub = std::slice::from_raw_parts(unbounded_in, 9);
        Some(([l0, l1, l2], [&ub[0..3], &ub[3..6], &ub[6..9]]))
    } else {
        None
    };
    Some(WorkProfile { matrix, trc, lutsize })
}

/// Shared body for the chroma-free split path. `v2` selects the v2/v3 log-mapping
/// clamp and desaturation formula; otherwise the v1 forms are used.
#[allow(clippy::too_many_arguments)]
unsafe fn split_impl(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize, wp: &Option<WorkProfile>,
    grey: f32, black: f32, dynamic_range: f32,
    sigma_toe: f32, sigma_shoulder: f32, saturation: f32, output_power: f32,
    m1: &[f32], m2: &[f32], m3: &[f32], m4: &[f32], m5: &[f32],
    latitude_min: f32, latitude_max: f32, type0: i32, type1: i32, v2: bool,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    for k in 0..npixels {
        let base = k * 4;
        let mut temp = [0.0f32; 4];
        for c in 0..3 {
            let x = input[base + c].max(NORM_MIN);
            temp[c] = if v2 {
                log_tonemapping_v2(x, grey, black, dynamic_range)
            } else {
                log_tonemapping_v1(x, grey, black, dynamic_range)
            };
        }

        let lum = luminance(temp, wp);
        let desaturation = if v2 {
            filmic_desaturate_v2(lum, sigma_toe, sigma_shoulder, saturation)
        } else {
            filmic_desaturate_v1(lum, sigma_toe, sigma_shoulder, saturation)
        };

        let mut pix_out = [0.0f32; 3];
        for c in 0..3 {
            let xs = linear_saturation(temp[c], lum, desaturation);
            let s = filmic_spline(xs, m1, m2, m3, m4, m5, latitude_min, latitude_max, type0, type1);
            pix_out[c] = powf_log2(clamp01(s), output_power);
        }
        output[base] = pix_out[0];
        output[base + 1] = pix_out[1];
        output[base + 2] = pix_out[2];
        output[base + 3] = 0.0; // clip(0)^power == 0, matches the C path
    }
}

/// Filmic chroma-free tone mapping, colour-science v1.
/// Replaces the DT_OMP_FOR loop in filmic_split_v1() (filmicrgb.c).
///
/// # Safety
/// Buffers hold `npixels*4` floats; `m1..m5` hold >= 3 floats each; work-profile
/// pointers valid per `make_work_profile`'s contract.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_filmicrgb_split_v1(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize,
    has_work_profile: i32, matrix_in: *const f32,
    lut0: *const f32, lut1: *const f32, lut2: *const f32, unbounded_in: *const f32,
    lutsize: i32, nonlinearlut: i32,
    grey_source: f32, black_source: f32, dynamic_range: f32,
    sigma_toe: f32, sigma_shoulder: f32, saturation: f32, output_power: f32,
    m1: *const f32, m2: *const f32, m3: *const f32, m4: *const f32, m5: *const f32,
    latitude_min: f32, latitude_max: f32, type0: i32, type1: i32,
) {
    let wp = make_work_profile(
        has_work_profile != 0, matrix_in, lut0, lut1, lut2, unbounded_in,
        lutsize as usize, nonlinearlut != 0,
    );
    let (m1, m2, m3, m4, m5) = (
        std::slice::from_raw_parts(m1, 4), std::slice::from_raw_parts(m2, 4),
        std::slice::from_raw_parts(m3, 4), std::slice::from_raw_parts(m4, 4),
        std::slice::from_raw_parts(m5, 4),
    );
    split_impl(
        in_buf, out_buf, npixels, &wp, grey_source, black_source, dynamic_range,
        sigma_toe, sigma_shoulder, saturation, output_power,
        m1, m2, m3, m4, m5, latitude_min, latitude_max, type0, type1, false,
    );
}

/// Filmic chroma-free tone mapping, colour-science v2/v3.
/// Replaces the DT_OMP_FOR loop in filmic_split_v2_v3() (filmicrgb.c).
///
/// # Safety
/// Same contract as `darkroom_filmicrgb_split_v1`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_filmicrgb_split_v2_v3(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize,
    has_work_profile: i32, matrix_in: *const f32,
    lut0: *const f32, lut1: *const f32, lut2: *const f32, unbounded_in: *const f32,
    lutsize: i32, nonlinearlut: i32,
    grey_source: f32, black_source: f32, dynamic_range: f32,
    sigma_toe: f32, sigma_shoulder: f32, saturation: f32, output_power: f32,
    m1: *const f32, m2: *const f32, m3: *const f32, m4: *const f32, m5: *const f32,
    latitude_min: f32, latitude_max: f32, type0: i32, type1: i32,
) {
    let wp = make_work_profile(
        has_work_profile != 0, matrix_in, lut0, lut1, lut2, unbounded_in,
        lutsize as usize, nonlinearlut != 0,
    );
    let (m1, m2, m3, m4, m5) = (
        std::slice::from_raw_parts(m1, 4), std::slice::from_raw_parts(m2, 4),
        std::slice::from_raw_parts(m3, 4), std::slice::from_raw_parts(m4, 4),
        std::slice::from_raw_parts(m5, 4),
    );
    split_impl(
        in_buf, out_buf, npixels, &wp, grey_source, black_source, dynamic_range,
        sigma_toe, sigma_shoulder, saturation, output_power,
        m1, m2, m3, m4, m5, latitude_min, latitude_max, type0, type1, true,
    );
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
    fn inpaint_noise_zero_weight_passes_through_rgb() {
        // weight=0 → out[c] = max(inp[c] * 1 + 0 * noise[c], 0) for c in 0..3.
        // Channel 3 (alpha) is left as max(NaN,0)=0 in both C and Rust since
        // gaussian_noise_simd fills u1/u2 for channels 0..2 only, making u1[3]=0
        // → log(0)=-∞ → noise[3]=inf → 0*inf=NaN → max(NaN,0)=0.
        let inp  = vec![0.3_f32, 0.5, 0.7, 1.0];
        let mask = vec![0.0_f32; 1];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_filmicrgb_inpaint_noise(
                inp.as_ptr(), mask.as_ptr(), out.as_mut_ptr(),
                0.1, 0.9, 1, 1, 1,
            );
        }
        // Only check RGB channels; alpha behaviour matches C (becomes 0 via NaN path)
        for c in 0..3 { assert!((out[c] - inp[c]).abs() < 1e-5, "c={c}: out={}", out[c]); }
    }

    #[test]
    fn inpaint_noise_output_is_nonneg() {
        // Gaussian noise * some weight; abs ensures non-negative output
        let inp  = vec![0.5_f32, 0.5, 0.5, 1.0];
        let mask = vec![1.0_f32];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_filmicrgb_inpaint_noise(
                inp.as_ptr(), mask.as_ptr(), out.as_mut_ptr(),
                0.5, 0.5, 1, 1, 1,
            );
        }
        for c in 0..4 { assert!(out[c] >= 0.0, "c={c}: out={}", out[c]); }
    }

    #[test]
    fn inpaint_noise_is_deterministic() {
        let inp  = vec![0.4_f32, 0.6, 0.2, 1.0];
        let mask = vec![0.8_f32];
        let mut o1 = vec![0.0_f32; 4];
        let mut o2 = vec![0.0_f32; 4];
        unsafe {
            darkroom_filmicrgb_inpaint_noise(inp.as_ptr(), mask.as_ptr(), o1.as_mut_ptr(), 0.3, 0.7, 1, 1, 1);
            darkroom_filmicrgb_inpaint_noise(inp.as_ptr(), mask.as_ptr(), o2.as_mut_ptr(), 0.3, 0.7, 1, 1, 1);
        }
        assert_eq!(o1, o2);
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

    #[test]
    fn log_tonemapping_v1_clamps_to_norm_min_and_one() {
        assert_eq!(log_tonemapping_v1(1e-9, 0.18, -5.0, 8.0), NORM_MIN);
        assert_eq!(log_tonemapping_v1(1e6, 0.18, -5.0, 8.0), 1.0);
    }

    #[test]
    fn spline_latitude_is_linear() {
        let m1 = [0.0, 0.0, 0.1, 0.0];
        let m2 = [0.0, 0.0, 0.5, 0.0];
        let z = [0.0f32; 4];
        let x = 0.4;
        let y = filmic_spline(x, &m1, &m2, &z, &z, &z, 0.2, 0.8, 0, 0);
        assert!((y - (0.1 + 0.5 * x)).abs() < 1e-6, "{y}");
    }

    #[test]
    fn spline_poly4_toe_matches_horner() {
        let m1 = [0.1, 0.0, 0.0, 0.0];
        let m2 = [0.2, 0.0, 0.0, 0.0];
        let m3 = [0.3, 0.0, 0.0, 0.0];
        let m4 = [0.4, 0.0, 0.0, 0.0];
        let m5 = [0.5, 0.0, 0.0, 0.0];
        let x = 0.1f32;
        let expect = 0.1 + x * (0.2 + x * (0.3 + x * (0.4 + x * 0.5)));
        let y = filmic_spline(x, &m1, &m2, &m3, &m4, &m5, 0.2, 0.8, 0, 0);
        assert!((y - expect).abs() < 1e-6, "{y} vs {expect}");
    }

    #[test]
    fn split_v1_runs_finite_and_zeroes_alpha() {
        let input = [0.18f32, 0.18, 0.18, 1.0];
        let mut out = [0f32; 4];
        let m1 = [0.0, 0.0, 0.0, 0.0];
        let m2 = [0.0, 0.0, 1.0, 0.0]; // latitude slope 1
        let z = [0.0f32; 4];
        unsafe {
            darkroom_filmicrgb_split_v1(
                input.as_ptr(), out.as_mut_ptr(), 1,
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), 0, 0,
                0.18, -5.0, 8.0, 0.2, 0.2, 1.0, 1.0,
                m1.as_ptr(), m2.as_ptr(), z.as_ptr(), z.as_ptr(), z.as_ptr(),
                0.0, 1.0, 0, 0,
            );
        }
        for c in 0..3 {
            assert!(out[c].is_finite() && (0.0..=1.0).contains(&out[c]), "c={c} {out:?}");
        }
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn split_v1_v2_agree_on_neutral_saturation() {
        // With saturation=1 and a symmetric grey pixel, both versions should
        // produce finite, bounded output (exact equality not expected — the
        // desaturation formulas differ — but both must be well-behaved).
        let input = [0.05f32, 0.2, 0.6, 1.0];
        let m1 = [0.0, 0.0, 0.0, 0.0];
        let m2 = [0.0, 0.0, 1.0, 0.0];
        let z = [0.0f32; 4];
        let mut o1 = [0f32; 4];
        let mut o2 = [0f32; 4];
        unsafe {
            darkroom_filmicrgb_split_v1(
                input.as_ptr(), o1.as_mut_ptr(), 1, 0,
                std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), 0, 0, 0.18, -5.0, 8.0, 0.2, 0.2, 1.0, 1.0,
                m1.as_ptr(), m2.as_ptr(), z.as_ptr(), z.as_ptr(), z.as_ptr(), 0.0, 1.0, 0, 0,
            );
            darkroom_filmicrgb_split_v2_v3(
                input.as_ptr(), o2.as_mut_ptr(), 1, 0,
                std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), 0, 0, 0.18, -5.0, 8.0, 0.2, 0.2, 1.0, 1.0,
                m1.as_ptr(), m2.as_ptr(), z.as_ptr(), z.as_ptr(), z.as_ptr(), 0.0, 1.0, 0, 0,
            );
        }
        for c in 0..3 {
            assert!(o1[c].is_finite() && (0.0..=1.0).contains(&o1[c]), "v1 c={c} {o1:?}");
            assert!(o2[c].is_finite() && (0.0..=1.0).contains(&o2[c]), "v2 c={c} {o2:?}");
        }
    }
}
