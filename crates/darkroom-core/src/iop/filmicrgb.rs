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

// ── Chroma-preservation path (ratio-preserving norm tone mapping) ─────────────

const INVERSE_SQRT_3: f32 = 0.5773502691896258;

/// `(|R|³+|G|³+|B|³) / max(|R|²+|G|²+|B|², 1e-12)` — matches pixel_rgb_norm_power().
#[inline(always)]
fn pixel_rgb_norm_power(p: [f32; 4]) -> f32 {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for c in 0..3 {
        let v = p[c].abs();
        let sq = v * v;
        num += sq * v;
        den += sq;
    }
    num / den.max(1e-12)
}

/// Pixel norm dispatch — matches get_pixel_norm(). Variant values mirror
/// dt_iop_filmicrgb_methods_type_t (1=max,2=luminance,3=power,4/5=euclidean).
#[inline(always)]
fn get_pixel_norm(p: [f32; 4], variant: i32, wp: &Option<WorkProfile>) -> f32 {
    match variant {
        1 => p[0].max(p[1]).max(p[2]),
        3 => pixel_rgb_norm_power(p),
        4 => (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt(),
        5 => (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() * INVERSE_SQRT_3,
        // 2 (LUMINANCE) and the default both use the profile luminance.
        _ => luminance(p, wp),
    }
}

/// Chroma-preserving tone mapping, colour-science v1.
/// Replaces the DT_OMP_FOR loop in filmic_chroma_v1() (filmicrgb.c).
///
/// # Safety
/// Same buffer/pointer contract as `darkroom_filmicrgb_split_v1`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_filmicrgb_chroma_v1(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize, variant: i32,
    has_work_profile: i32, matrix_in: *const f32,
    lut0: *const f32, lut1: *const f32, lut2: *const f32, unbounded_in: *const f32,
    lutsize: i32, nonlinearlut: i32,
    grey: f32, black: f32, dynamic_range: f32,
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
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    for k in 0..npixels {
        let base = k * 4;
        let pix_in = [input[base], input[base + 1], input[base + 2], input[base + 3]];
        let mut norm = get_pixel_norm(pix_in, variant, &wp).max(NORM_MIN);

        let mut ratios = [0.0f32; 4];
        for c in 0..4 {
            ratios[c] = pix_in[c] / norm;
        }
        let min_ratios = ratios[0].min(ratios[1]).min(ratios[2]);
        if min_ratios < 0.0 {
            for r in ratios.iter_mut() {
                *r -= min_ratios;
            }
        }

        norm = log_tonemapping_v1(norm, grey, black, dynamic_range);
        let desaturation = filmic_desaturate_v1(norm, sigma_toe, sigma_shoulder, saturation);

        for r in ratios.iter_mut() {
            *r *= norm;
        }
        let lum = luminance(ratios, &wp);
        for r in ratios.iter_mut() {
            *r = linear_saturation(*r, lum, desaturation) / norm;
        }

        let s = clamp01(filmic_spline(norm, m1, m2, m3, m4, m5, latitude_min, latitude_max, type0, type1));
        norm = s.powf(output_power); // scalar libm powf, matching the C chroma path
        // channel 3 (alpha) is carried through the ratio math, not zeroed, to
        // match the C `for_each_channel` 4-wide default (alpha is unused downstream).
        for c in 0..4 {
            output[base + c] = ratios[c] * norm;
        }
    }
}

/// Chroma-preserving tone mapping, colour-science v2/v3.
/// Replaces the DT_OMP_FOR loop in filmic_chroma_v2_v3() (filmicrgb.c).
/// `colorscience_version == 2` selects the v3 re-normalisation branch.
///
/// # Safety
/// Same buffer/pointer contract as `darkroom_filmicrgb_split_v1`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_filmicrgb_chroma_v2_v3(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize, variant: i32,
    colorscience_version: i32,
    has_work_profile: i32, matrix_in: *const f32,
    lut0: *const f32, lut1: *const f32, lut2: *const f32, unbounded_in: *const f32,
    lutsize: i32, nonlinearlut: i32,
    grey: f32, black: f32, dynamic_range: f32,
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
    let is_v3 = colorscience_version == 2; // DT_FILMIC_COLORSCIENCE_V3
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    for k in 0..npixels {
        let base = k * 4;
        let pix_in = [input[base], input[base + 1], input[base + 2], input[base + 3]];
        let mut norm = get_pixel_norm(pix_in, variant, &wp).max(NORM_MIN);

        let mut ratios = [0.0f32; 4];
        for c in 0..4 {
            ratios[c] = pix_in[c] / norm;
        }
        let min_ratios = ratios[0].min(ratios[1]).min(ratios[2]);
        if min_ratios < 0.0 {
            for r in ratios.iter_mut() {
                *r -= min_ratios;
            }
        }

        norm = log_tonemapping_v2(norm, grey, black, dynamic_range); // == log_tonemapping_v2_1ch
        let desaturation = filmic_desaturate_v2(norm, sigma_toe, sigma_shoulder, saturation);

        let s = clamp01(filmic_spline(norm, m1, m2, m3, m4, m5, latitude_min, latitude_max, type0, type1));
        norm = s.powf(output_power);

        for r in ratios.iter_mut() {
            *r = (*r + (1.0 - *r) * (1.0 - desaturation)).max(0.0);
        }
        if is_v3 {
            norm /= get_pixel_norm(ratios, variant, &wp).max(NORM_MIN);
        }

        // channel 3 (alpha) carried through the ratio math (not zeroed), matching
        // the C `for_each_channel` 4-wide default; alpha is unused downstream.
        let mut pix_out = [0.0f32; 4];
        for c in 0..4 {
            pix_out[c] = ratios[c] * norm;
        }
        let max_pix = pix_out[0].max(pix_out[1]).max(pix_out[2]);
        if max_pix > 1.0 {
            for c in 0..4 {
                ratios[c] = (ratios[c] + (1.0 - max_pix)).max(0.0);
                pix_out[c] = clamp01(ratios[c] * norm);
            }
        }
        output[base..base + 4].copy_from_slice(&pix_out);
    }
}

// ── Colour-science v4/v5 gamut-mapped path (Filmlight Yrg) ────────────────────

/// CLAMP(x, lo, hi) == MAX(lo, MIN(x, hi)). Mirrors the C macro and, unlike
/// `f32::clamp`, never panics when `lo > hi` (matches darktable's behaviour).
#[inline(always)]
fn clamp_lh(x: f32, lo: f32, hi: f32) -> f32 {
    x.min(hi).max(lo)
}

/// The six prepared colour matrices `filmic_v4_prepare_matrices` builds on the C
/// side and passes flat across the FFI. Borrowed views over `[[f32;4];4]`.
struct V4Matrices<'a> {
    input_matrix_trans: &'a [[f32; 4]; 4],
    output_matrix: &'a [[f32; 4]; 4],
    output_matrix_trans: &'a [[f32; 4]; 4],
    export_input_matrix_trans: &'a [[f32; 4]; 4],
    export_output_matrix: &'a [[f32; 4]; 4],
    export_output_matrix_trans: &'a [[f32; 4]; 4],
}

/// Massage Ych_final's chroma towards/away from the original chroma per the
/// saturation control. Mutates `ych_final[1]`. Matches `filmic_desaturate_v4()`
/// in filmicrgb.c:1456.
#[inline(always)]
fn filmic_desaturate_v4(ych_original: [f32; 4], ych_final: &mut [f32; 4], saturation: f32) {
    // Ych is normalised, so c is a saturation; chroma = c * Y.
    let chroma_original = ych_original[1] * ych_original[0]; // c2
    let mut chroma_final = ych_final[1] * ych_final[0]; // c1
    let delta_chroma = saturation * (chroma_original - chroma_final);

    let filmic_brightens = ych_final[0] > ych_original[0];
    let filmic_resat = chroma_original < chroma_final;
    let filmic_desat = chroma_original > chroma_final;
    let user_resat = saturation > 0.0;
    let user_desat = saturation < 0.0;

    chroma_final = if filmic_brightens && filmic_resat {
        (chroma_original + chroma_final) / 2.0
    } else if (user_resat && filmic_desat) || user_desat {
        chroma_final + delta_chroma
    } else {
        chroma_final
    };

    ych_final[1] = (chroma_final / ych_final[0]).max(0.0);
}

/// Bring a possibly-out-of-gamut Ych pixel back into the target RGB gamut by
/// estimating an in-gamut luminance and clipping chroma, returning the clamped
/// RGB. Matches `gamut_check_RGB()` in filmicrgb.c:1494.
#[inline(always)]
fn gamut_check_rgb(
    matrix_in_trans: &[[f32; 4]; 4],
    matrix_out: &[[f32; 4]; 4],
    matrix_out_trans: &[[f32; 4]; 4],
    display_black: f32,
    display_white: f32,
    ych_in: [f32; 4],
) -> [f32; 4] {
    // How much white light to add to bring the brightened pixel back in gamut.
    let mut rgb_brightened = color::ych_to_rgb(ych_in, matrix_out_trans);
    let min_pix = rgb_brightened[0].min(rgb_brightened[1]).min(rgb_brightened[2]);
    let black_offset = (-min_pix).max(0.0);
    for v in rgb_brightened.iter_mut() {
        *v += black_offset;
    }
    let ych_brightened = color::rgb_to_ych(rgb_brightened, matrix_in_trans);

    let y = clamp_lh(
        (ych_in[0] + ych_brightened[0]) / 2.0,
        color::CIE_Y_1931_TO_2006 * display_black,
        color::CIE_Y_1931_TO_2006 * display_white,
    );

    let cos_h = ych_in[2];
    let sin_h = ych_in[3];
    let new_chroma = ych_in[1].min(color::ych_max_chroma(matrix_out, display_white, y, cos_h, sin_h));

    let ych = [y, new_chroma, cos_h, sin_h];
    let mut rgb_out = color::ych_to_rgb(ych, matrix_out_trans);
    for v in rgb_out.iter_mut() {
        *v = clamp_lh(*v, 0.0, display_white);
    }
    rgb_out
}

/// Force hue to the original, clip luminance, massage chroma, gamut-check in Yrg
/// and in target RGB. Returns the gamut-mapped pixel. Matches `gamut_mapping()`
/// in filmicrgb.c:1538.
#[inline(always)]
fn gamut_mapping_v4(
    mut ych_final: [f32; 4],
    ych_original: [f32; 4],
    m: &V4Matrices,
    display_black: f32,
    display_white: f32,
    saturation: f32,
    use_output_profile: bool,
) -> [f32; 4] {
    // Force final hue to original
    ych_final[2] = ych_original[2];
    ych_final[3] = ych_original[3];

    // Clip luminance
    ych_final[0] = clamp_lh(
        ych_final[0],
        color::CIE_Y_1931_TO_2006 * display_black,
        color::CIE_Y_1931_TO_2006 * display_white,
    );

    // Massage chroma, then gamut-clip chroma in Yrg/LMS cone space
    filmic_desaturate_v4(ych_original, &mut ych_final, saturation);
    ych_final = color::gamut_check_yrg(ych_final);

    if !use_output_profile {
        gamut_check_rgb(
            m.input_matrix_trans, m.output_matrix, m.output_matrix_trans,
            display_black, display_white, ych_final,
        )
    } else {
        let pix_out = gamut_check_rgb(
            m.export_input_matrix_trans, m.export_output_matrix, m.export_output_matrix_trans,
            display_black, display_white, ych_final,
        );
        // export RGB -> CIE LMS 2006 D65 -> pipeline RGB D50
        let lms = color::apply_transposed_color_matrix(&pix_out, m.export_input_matrix_trans);
        color::apply_transposed_color_matrix(&lms, m.output_matrix_trans)
    }
}

/// Spline factor vectors + latitudes + curve types, grouped for the v4 helpers.
struct Spline<'a> {
    m1: &'a [f32], m2: &'a [f32], m3: &'a [f32], m4: &'a [f32], m5: &'a [f32],
    latitude_min: f32, latitude_max: f32, type0: i32, type1: i32,
}

/// Chroma-preserving (norm) tone mapping for v4/v5. Matches
/// `norm_tone_mapping_v4()` in filmicrgb.c:1619. The output alpha is unused —
/// `gamut_mapping_v4` overwrites the whole pixel downstream.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn norm_tone_mapping_v4(
    pix_in: [f32; 4], variant: i32, wp: &Option<WorkProfile>,
    grey: f32, black: f32, dynamic_range: f32, output_power: f32,
    sp: &Spline, norm_min: f32, norm_max: f32, display_black: f32, display_white: f32,
) -> [f32; 4] {
    let mut norm = clamp_lh(get_pixel_norm(pix_in, variant, wp), norm_min, norm_max);
    let ratios = [pix_in[0] / norm, pix_in[1] / norm, pix_in[2] / norm, pix_in[3] / norm];

    norm = log_tonemapping_v2(norm, grey, black, dynamic_range);
    let s = clamp_lh(
        filmic_spline(norm, sp.m1, sp.m2, sp.m3, sp.m4, sp.m5,
                      sp.latitude_min, sp.latitude_max, sp.type0, sp.type1),
        display_black, display_white,
    );
    norm = s.powf(output_power); // scalar libm powf, matching the C norm path

    [ratios[0] * norm, ratios[1] * norm, ratios[2] * norm, ratios[3] * norm]
}

/// Per-channel ("naive") RGB tone mapping for v4/v5. Matches
/// `RGB_tone_mapping_v4()` in filmicrgb.c:1658. Alpha is unused downstream.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn rgb_tone_mapping_v4(
    pix_in: [f32; 4], grey: f32, black: f32, dynamic_range: f32, output_power: f32,
    sp: &Spline, display_white: f32,
) -> [f32; 4] {
    // log_tonemapping_v2 (4-wide, clamped to [0,1] via dt_vector_clip)
    let mut mapped = [0.0f32; 4];
    for c in 0..3 {
        mapped[c] = log_tonemapping_v2(pix_in[c], grey, black, dynamic_range);
    }
    // spline on RGB only — matches the C `for(c = 0; c < 3; c++)` loop
    for c in 0..3 {
        mapped[c] = filmic_spline(mapped[c], sp.m1, sp.m2, sp.m3, sp.m4, sp.m5,
                                  sp.latitude_min, sp.latitude_max, sp.type0, sp.type1);
    }
    // individual components can always go to zero; luminance is clamped later
    let mut out = [0.0f32; 4];
    for c in 0..3 {
        // dt_vector_pow1 == 2^(log2(x)*p); use the same libm form as the split path
        out[c] = powf_log2(clamp_lh(mapped[c], 0.0, display_white), output_power);
    }
    out
}

/// Read a flat 16-float FFI pointer as a `&[[f32;4];4]` colour matrix.
/// # Safety: `p` must point to 16 contiguous floats.
#[inline(always)]
unsafe fn mat4(p: *const f32) -> &'static [[f32; 4]; 4] {
    &*(p as *const [[f32; 4]; 4])
}

/// Chroma-preserving (norm) tone mapping + gamut mapping, colour-science v4.
/// Replaces the DT_OMP_FOR loop in filmic_chroma_v4() (filmicrgb.c:1710).
///
/// # Safety
/// Buffers hold `npixels*4` floats; `m1..m5` hold >= 4 floats each; the six
/// matrix pointers each hold 16 floats; work-profile pointers valid per
/// `make_work_profile`'s contract.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_filmicrgb_chroma_v4(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize, variant: i32,
    has_work_profile: i32, matrix_in: *const f32,
    lut0: *const f32, lut1: *const f32, lut2: *const f32, unbounded_in: *const f32,
    lutsize: i32, nonlinearlut: i32,
    grey: f32, black: f32, dynamic_range: f32, output_power: f32, saturation: f32,
    m1: *const f32, m2: *const f32, m3: *const f32, m4: *const f32, m5: *const f32,
    latitude_min: f32, latitude_max: f32, type0: i32, type1: i32,
    input_matrix_trans: *const f32, output_matrix: *const f32, output_matrix_trans: *const f32,
    export_input_matrix_trans: *const f32, export_output_matrix: *const f32, export_output_matrix_trans: *const f32,
    use_output_profile: i32, norm_min: f32, norm_max: f32, display_black: f32, display_white: f32,
) {
    let wp = make_work_profile(
        has_work_profile != 0, matrix_in, lut0, lut1, lut2, unbounded_in,
        lutsize as usize, nonlinearlut != 0,
    );
    let sp = Spline {
        m1: std::slice::from_raw_parts(m1, 4), m2: std::slice::from_raw_parts(m2, 4),
        m3: std::slice::from_raw_parts(m3, 4), m4: std::slice::from_raw_parts(m4, 4),
        m5: std::slice::from_raw_parts(m5, 4),
        latitude_min, latitude_max, type0, type1,
    };
    let mats = V4Matrices {
        input_matrix_trans: mat4(input_matrix_trans),
        output_matrix: mat4(output_matrix),
        output_matrix_trans: mat4(output_matrix_trans),
        export_input_matrix_trans: mat4(export_input_matrix_trans),
        export_output_matrix: mat4(export_output_matrix),
        export_output_matrix_trans: mat4(export_output_matrix_trans),
    };
    let use_op = use_output_profile != 0;
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    for k in 0..npixels {
        let base = k * 4;
        let pix_in = [input[base], input[base + 1], input[base + 2], input[base + 3]];
        let pix_out = norm_tone_mapping_v4(
            pix_in, variant, &wp, grey, black, dynamic_range, output_power,
            &sp, norm_min, norm_max, display_black, display_white,
        );
        let ych_original = color::rgb_to_ych(pix_in, mats.input_matrix_trans);
        let ych_final = color::rgb_to_ych(pix_out, mats.input_matrix_trans);
        let mapped = gamut_mapping_v4(
            ych_final, ych_original, &mats, display_black, display_white, saturation, use_op,
        );
        output[base..base + 4].copy_from_slice(&mapped);
    }
}

/// Per-channel ("split") tone mapping + gamut mapping, colour-science v4.
/// Replaces the DT_OMP_FOR loop in filmic_split_v4() (filmicrgb.c:1761).
///
/// # Safety
/// Same contract as `darkroom_filmicrgb_chroma_v4`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_filmicrgb_split_v4(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize,
    grey: f32, black: f32, dynamic_range: f32, output_power: f32, saturation: f32,
    m1: *const f32, m2: *const f32, m3: *const f32, m4: *const f32, m5: *const f32,
    latitude_min: f32, latitude_max: f32, type0: i32, type1: i32,
    input_matrix_trans: *const f32, output_matrix: *const f32, output_matrix_trans: *const f32,
    export_input_matrix_trans: *const f32, export_output_matrix: *const f32, export_output_matrix_trans: *const f32,
    use_output_profile: i32, display_black: f32, display_white: f32,
) {
    let sp = Spline {
        m1: std::slice::from_raw_parts(m1, 4), m2: std::slice::from_raw_parts(m2, 4),
        m3: std::slice::from_raw_parts(m3, 4), m4: std::slice::from_raw_parts(m4, 4),
        m5: std::slice::from_raw_parts(m5, 4),
        latitude_min, latitude_max, type0, type1,
    };
    let mats = V4Matrices {
        input_matrix_trans: mat4(input_matrix_trans),
        output_matrix: mat4(output_matrix),
        output_matrix_trans: mat4(output_matrix_trans),
        export_input_matrix_trans: mat4(export_input_matrix_trans),
        export_output_matrix: mat4(export_output_matrix),
        export_output_matrix_trans: mat4(export_output_matrix_trans),
    };
    let use_op = use_output_profile != 0;
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    for k in 0..npixels {
        let base = k * 4;
        let pix_in = [input[base], input[base + 1], input[base + 2], input[base + 3]];
        let pix_out = rgb_tone_mapping_v4(
            pix_in, grey, black, dynamic_range, output_power, &sp, display_white,
        );
        let ych_original = color::rgb_to_ych(pix_in, mats.input_matrix_trans);
        let mut ych_final = color::rgb_to_ych(pix_out, mats.input_matrix_trans);
        ych_final[1] = ych_original[1].min(ych_final[1]);
        let mapped = gamut_mapping_v4(
            ych_final, ych_original, &mats, display_black, display_white, saturation, use_op,
        );
        output[base..base + 4].copy_from_slice(&mapped);
    }
}

/// Default colour-science v5: blend of naive (per-channel) and max-RGB (norm)
/// tone mapping, then gamut mapping with saturation forced to 0.
/// Replaces the DT_OMP_FOR loop in filmic_v5() (filmicrgb.c:1813).
///
/// # Safety
/// Same contract as `darkroom_filmicrgb_chroma_v4`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_filmicrgb_v5(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize,
    has_work_profile: i32, matrix_in: *const f32,
    lut0: *const f32, lut1: *const f32, lut2: *const f32, unbounded_in: *const f32,
    lutsize: i32, nonlinearlut: i32,
    grey: f32, black: f32, dynamic_range: f32, output_power: f32, saturation: f32,
    m1: *const f32, m2: *const f32, m3: *const f32, m4: *const f32, m5: *const f32,
    latitude_min: f32, latitude_max: f32, type0: i32, type1: i32,
    input_matrix_trans: *const f32, output_matrix: *const f32, output_matrix_trans: *const f32,
    export_input_matrix_trans: *const f32, export_output_matrix: *const f32, export_output_matrix_trans: *const f32,
    use_output_profile: i32, norm_min: f32, norm_max: f32, display_black: f32, display_white: f32,
) {
    let wp = make_work_profile(
        has_work_profile != 0, matrix_in, lut0, lut1, lut2, unbounded_in,
        lutsize as usize, nonlinearlut != 0,
    );
    let sp = Spline {
        m1: std::slice::from_raw_parts(m1, 4), m2: std::slice::from_raw_parts(m2, 4),
        m3: std::slice::from_raw_parts(m3, 4), m4: std::slice::from_raw_parts(m4, 4),
        m5: std::slice::from_raw_parts(m5, 4),
        latitude_min, latitude_max, type0, type1,
    };
    let mats = V4Matrices {
        input_matrix_trans: mat4(input_matrix_trans),
        output_matrix: mat4(output_matrix),
        output_matrix_trans: mat4(output_matrix_trans),
        export_input_matrix_trans: mat4(export_input_matrix_trans),
        export_output_matrix: mat4(export_output_matrix),
        export_output_matrix_trans: mat4(export_output_matrix_trans),
    };
    let use_op = use_output_profile != 0;
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    const METHOD_MAX_RGB: i32 = 1; // DT_FILMIC_METHOD_MAX_RGB
    for k in 0..npixels {
        let base = k * 4;
        let pix_in = [input[base], input[base + 1], input[base + 2], input[base + 3]];
        let naive_rgb = rgb_tone_mapping_v4(
            pix_in, grey, black, dynamic_range, output_power, &sp, display_white,
        );
        let max_rgb = norm_tone_mapping_v4(
            pix_in, METHOD_MAX_RGB, &wp, grey, black, dynamic_range, output_power,
            &sp, norm_min, norm_max, display_black, display_white,
        );
        // Mix max RGB with naive RGB. The C `for_each_channel` blend also writes
        // alpha = (0.5+sat)*pix_in[3]; we leave alpha 0 here because gamut_mapping_v4
        // overwrites the whole pixel below (its final Ych_to_RGB zeroes channel 3),
        // so the output alpha is 0 in both the C and Rust paths.
        let mut pix_out = [0.0f32; 4];
        for c in 0..3 {
            pix_out[c] = (0.5 - saturation) * naive_rgb[c] + (0.5 + saturation) * max_rgb[c];
        }
        let ych_original = color::rgb_to_ych(pix_in, mats.input_matrix_trans);
        let mut ych_final = color::rgb_to_ych(pix_out, mats.input_matrix_trans);
        ych_final[1] = ych_original[1].min(ych_final[1]);
        let mapped = gamut_mapping_v4(
            ych_final, ych_original, &mats, display_black, display_white, 0.0, use_op,
        );
        output[base..base + 4].copy_from_slice(&mapped);
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

    #[test]
    fn get_pixel_norm_variants() {
        let p = [0.2f32, 0.5, 0.1, 1.0];
        let none = None;
        assert!((get_pixel_norm(p, 1, &none) - 0.5).abs() < 1e-6); // max
        // power: (.2³+.5³+.1³)/(.2²+.5²+.1²)
        let num = 0.2f32.powi(3) + 0.5f32.powi(3) + 0.1f32.powi(3);
        let den = 0.2f32.powi(2) + 0.5f32.powi(2) + 0.1f32.powi(2);
        assert!((get_pixel_norm(p, 3, &none) - num / den).abs() < 1e-6);
        // euclidean v1 vs v2 differ by 1/sqrt(3)
        let e1 = get_pixel_norm(p, 4, &none);
        let e2 = get_pixel_norm(p, 5, &none);
        assert!((e2 - e1 * INVERSE_SQRT_3).abs() < 1e-6);
    }

    #[test]
    fn chroma_v1_grey_pixel_preserves_neutrality() {
        // grey input -> ratios all 1 -> output channels equal (still neutral).
        let input = [0.18f32, 0.18, 0.18, 1.0];
        let mut out = [0f32; 4];
        let m1 = [0.0, 0.0, 0.0, 0.0];
        let m2 = [0.0, 0.0, 1.0, 0.0];
        let z = [0.0f32; 4];
        unsafe {
            darkroom_filmicrgb_chroma_v1(
                input.as_ptr(), out.as_mut_ptr(), 1, 3, // POWER_NORM
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), 0, 0, 0.18, -5.0, 8.0, 0.2, 0.2, 1.0, 1.0,
                m1.as_ptr(), m2.as_ptr(), z.as_ptr(), z.as_ptr(), z.as_ptr(), 0.0, 1.0, 0, 0,
            );
        }
        assert!((out[0] - out[1]).abs() < 1e-5 && (out[1] - out[2]).abs() < 1e-5, "{out:?}");
        for c in 0..3 {
            assert!(out[c].is_finite(), "c={c} {out:?}");
        }
    }

    #[test]
    fn chroma_v2_v3_gamut_maps_and_stays_bounded() {
        // bright saturated input should be gamut-mapped to <= 1 by the penalty.
        let input = [2.0f32, 0.1, 0.05, 1.0];
        let mut out = [0f32; 4];
        let m1 = [0.0, 0.0, 0.0, 0.0];
        let m2 = [0.0, 0.0, 1.0, 0.0];
        let z = [0.0f32; 4];
        unsafe {
            darkroom_filmicrgb_chroma_v2_v3(
                input.as_ptr(), out.as_mut_ptr(), 1, 1, 2, // MAX_RGB, colorscience V3
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), 0, 0, 0.18, -5.0, 8.0, 0.2, 0.2, 1.0, 1.0,
                m1.as_ptr(), m2.as_ptr(), z.as_ptr(), z.as_ptr(), z.as_ptr(), 0.0, 1.0, 0, 0,
            );
        }
        for c in 0..3 {
            assert!(out[c].is_finite() && out[c] <= 1.0 + 1e-6 && out[c] >= 0.0, "c={c} {out:?}");
        }
    }

    // ── v4/v5 gamut-mapped path ───────────────────────────────────────────────

    // Flat 4x4 identity (last row/col padding zero), used as stand-in colour
    // matrices: RGB == LMS so the Yrg conversions are well-defined. The gamut
    // path's final clamp guarantees output in [0, display_white] when finite.
    const IDENTITY_FLAT: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];

    fn linear_spline() -> ([f32; 4], [f32; 4], [f32; 4]) {
        // M1 = 0, M2 latitude slope = 1, rest zero → identity-ish S-curve.
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0f32; 4])
    }

    #[test]
    fn split_v4_runs_finite_and_bounded() {
        let input = [0.18f32, 0.2, 0.15, 1.0];
        let mut out = [0f32; 4];
        let (m1, m2, z) = linear_spline();
        let id = IDENTITY_FLAT;
        unsafe {
            darkroom_filmicrgb_split_v4(
                input.as_ptr(), out.as_mut_ptr(), 1,
                0.18, -5.0, 8.0, 1.0, 0.0,
                m1.as_ptr(), m2.as_ptr(), z.as_ptr(), z.as_ptr(), z.as_ptr(),
                0.0, 1.0, 0, 0,
                id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(),
                0, 0.0, 1.0,
            );
        }
        for c in 0..3 {
            assert!(out[c].is_finite() && out[c] >= 0.0 && out[c] <= 1.0 + 1e-5, "c={c} {out:?}");
        }
    }

    #[test]
    fn chroma_v4_runs_finite_and_bounded() {
        // bright saturated input must be gamut-mapped into [0, display_white].
        let input = [2.0f32, 0.1, 0.05, 1.0];
        let mut out = [0f32; 4];
        let (m1, m2, z) = linear_spline();
        let id = IDENTITY_FLAT;
        // norm_min/norm_max from exp_tonemapping_v2(0/1): grey*2^(dr*x+black)
        let grey = 0.18f32; let black = -5.0f32; let dr = 8.0f32;
        let norm_min = grey * (dr * 0.0 + black).exp2();
        let norm_max = grey * (dr * 1.0 + black).exp2();
        unsafe {
            darkroom_filmicrgb_chroma_v4(
                input.as_ptr(), out.as_mut_ptr(), 1, 1, // MAX_RGB
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                0, 0,
                grey, black, dr, 1.0, 0.0,
                m1.as_ptr(), m2.as_ptr(), z.as_ptr(), z.as_ptr(), z.as_ptr(),
                0.0, 1.0, 0, 0,
                id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(),
                0, norm_min, norm_max, 0.0, 1.0,
            );
        }
        for c in 0..3 {
            assert!(out[c].is_finite() && out[c] >= 0.0 && out[c] <= 1.0 + 1e-5, "c={c} {out:?}");
        }
    }

    #[test]
    fn v5_runs_finite_and_bounded() {
        let input = [0.3f32, 0.5, 0.9, 1.0];
        let mut out = [0f32; 4];
        let (m1, m2, z) = linear_spline();
        let id = IDENTITY_FLAT;
        let grey = 0.18f32; let black = -5.0f32; let dr = 8.0f32;
        let norm_min = grey * (dr * 0.0 + black).exp2();
        let norm_max = grey * (dr * 1.0 + black).exp2();
        unsafe {
            darkroom_filmicrgb_v5(
                input.as_ptr(), out.as_mut_ptr(), 1,
                0, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                0, 0,
                grey, black, dr, 1.0, 0.0,
                m1.as_ptr(), m2.as_ptr(), z.as_ptr(), z.as_ptr(), z.as_ptr(),
                0.0, 1.0, 0, 0,
                id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(),
                0, norm_min, norm_max, 0.0, 1.0,
            );
        }
        for c in 0..3 {
            assert!(out[c].is_finite() && out[c] >= 0.0 && out[c] <= 1.0 + 1e-5, "c={c} {out:?}");
        }
    }

    #[test]
    fn filmic_desaturate_v4_no_user_sat_keeps_final() {
        // saturation=0, filmic darkens (final Y < original): chroma_final kept.
        let original = [0.5f32, 0.4, 1.0, 0.0]; // Y, c, cos, sin
        let mut final_ych = [0.3f32, 0.2, 1.0, 0.0];
        filmic_desaturate_v4(original, &mut final_ych, 0.0);
        // chroma_final unchanged → c stays 0.2
        assert!((final_ych[1] - 0.2).abs() < 1e-6, "{final_ych:?}");
    }
}
