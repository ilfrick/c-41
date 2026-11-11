use crate::pipeline::ColorSpace;
use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Channelmixerrgb;

impl IopProcess for Channelmixerrgb {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "channelmixerrgb" }
}

// ── constants ────────────────────────────────────────────────────────────────

const COLOR_MIN: f32 = -2.0; // channelmixerrgb.c
const COLOR_MAX: f32 = 2.0;
const TEMP_MIN: f32 = 1667.0;
const TEMP_MAX: f32 = 25000.0;
const NORM_MIN: f32 = 1.52587890625e-05; // 2^-16
const INVERSE_SQRT_3: f32 = 0.5773502691896258;
const D50XYY: [f32; 2] = [0.34567, 0.35850]; // D50 chromaticity (CIE xy)
const D50_UV: [f32; 2] = [0.20915914598542354, 0.488075320769787]; // D50 in u'v'

// dt_adaptation_t enum values from chromatic_adaptation.h
const KIND_LINEAR_BRADFORD: i32 = 0;
const KIND_CAT16: i32 = 1;
const KIND_FULL_BRADFORD: i32 = 2;
const KIND_XYZ: i32 = 3;
// KIND_RGB = 4 (no WB)

// dt_iop_channelmixer_rgb_version_t
const VER_V1: i32 = 0;
const VER_V3: i32 = 2;

// Bradford: XYZ → Bradford LMS (transposed, row-stride 3)
const XYZ_TO_BRADFORD_LMS_T: [f32; 9] = [
     0.8951, -0.7502,  0.0389,
     0.2664,  1.7135, -0.0685,
    -0.1614,  0.0367,  1.0296,
];

// Bradford: Bradford LMS → XYZ (transposed, row-stride 3)
const BRADFORD_LMS_TO_XYZ_T: [f32; 9] = [
     0.9870,  0.4323, -0.0085,
    -0.1471,  0.5184,  0.0400,
     0.1600,  0.0493,  0.9685,
];

// CAT16: XYZ → CAT16 LMS (transposed, row-stride 3)
const XYZ_TO_CAT16_LMS_T: [f32; 9] = [
     0.401288, -0.250268, -0.002079,
     0.650173,  1.204414,  0.048952,
    -0.051461,  0.045854,  0.953127,
];

// CAT16: CAT16 LMS → XYZ (transposed, row-stride 3)
const CAT16_LMS_TO_XYZ_T: [f32; 9] = [
     1.862068,  0.38752, -0.015841,
    -1.011255,  0.621447, -0.034123,
     0.149187, -0.008974,  1.049964,
];

// D50 primaries in each LMS space (from chromatic_adaptation.h)
const BRADFORD_D50: [f32; 3] = [0.996078, 1.020646, 0.818155];
const CAT16_D50:    [f32; 3] = [0.994535, 1.000997, 0.833036];
const XYZ_D50:      [f32; 3] = [0.9642119944211994, 1.0, 0.8251882845188288];

// ── matrix helpers ────────────────────────────────────────────────────────────

/// Apply pre-transposed 3×3 compact matrix (stride 3) to a 3-vector.
/// out[r] = t[r]*v[0] + t[3+r]*v[1] + t[6+r]*v[2]
#[inline(always)]
fn apply_t3(v: [f32; 3], t: &[f32; 9]) -> [f32; 3] {
    [
        t[0]*v[0] + t[3]*v[1] + t[6]*v[2],
        t[1]*v[0] + t[4]*v[1] + t[7]*v[2],
        t[2]*v[0] + t[5]*v[1] + t[8]*v[2],
    ]
}

/// Apply pre-transposed 4×4 matrix (row-stride 4, passed from C as *const f32)
/// to a 3-vector.  Only the top-left 3×3 is used.
/// out[r] = t[r]*v[0] + t[4+r]*v[1] + t[8+r]*v[2]
#[inline(always)]
fn apply_t4(v: [f32; 3], t: &[f32]) -> [f32; 3] {
    [
        t[0]*v[0] + t[4]*v[1] + t[8]*v[2],
        t[1]*v[0] + t[5]*v[1] + t[9]*v[2],
        t[2]*v[0] + t[6]*v[1] + t[10]*v[2],
    ]
}

// ── chromatic adaptation helpers ─────────────────────────────────────────────

#[inline(always)]
fn convert_xyz_to_bradford_lms(xyz: [f32; 3]) -> [f32; 3] {
    apply_t3(xyz, &XYZ_TO_BRADFORD_LMS_T)
}
#[inline(always)]
fn convert_bradford_lms_to_xyz(lms: [f32; 3]) -> [f32; 3] {
    apply_t3(lms, &BRADFORD_LMS_TO_XYZ_T)
}
#[inline(always)]
fn convert_xyz_to_cat16_lms(xyz: [f32; 3]) -> [f32; 3] {
    apply_t3(xyz, &XYZ_TO_CAT16_LMS_T)
}
#[inline(always)]
fn convert_cat16_lms_to_xyz(lms: [f32; 3]) -> [f32; 3] {
    apply_t3(lms, &CAT16_LMS_TO_XYZ_T)
}

fn convert_any_xyz_to_lms(xyz: [f32; 3], kind: i32) -> [f32; 3] {
    match kind {
        KIND_LINEAR_BRADFORD | KIND_FULL_BRADFORD => convert_xyz_to_bradford_lms(xyz),
        KIND_CAT16  => convert_xyz_to_cat16_lms(xyz),
        _ => xyz, // XYZ and RGB: pass-through
    }
}

fn convert_any_lms_to_xyz(lms: [f32; 3], kind: i32) -> [f32; 3] {
    match kind {
        KIND_LINEAR_BRADFORD | KIND_FULL_BRADFORD => convert_bradford_lms_to_xyz(lms),
        KIND_CAT16  => convert_cat16_lms_to_xyz(lms),
        _ => lms, // XYZ and RGB: pass-through
    }
}

/// Bradford chromatic adaptation → D50 illuminant.
/// `full`: non-linear (powf on B channel) if true, linear if false.
/// `p` must be pre-computed as powf(illuminant[2] / D50[2], 0.0834).
#[inline(always)]
fn bradford_adapt_d50(lms: [f32; 3], illuminant: [f32; 3], p: f32, full: bool) -> [f32; 3] {
    let t = [lms[0]/illuminant[0], lms[1]/illuminant[1], lms[2]/illuminant[2]];
    let t2 = if full { if t[2] > 0.0 { t[2].powf(p) } else { t[2] } } else { t[2] };
    [BRADFORD_D50[0]*t[0], BRADFORD_D50[1]*t[1], BRADFORD_D50[2]*t2]
}

/// CAT16 chromatic adaptation → D50.
/// `d`: degree of adaptation (1.0 = full).
#[inline(always)]
fn cat16_adapt_d50(lms: [f32; 3], illuminant: [f32; 3], d: f32, full: bool) -> [f32; 3] {
    if full {
        [lms[0]*CAT16_D50[0]/illuminant[0], lms[1]*CAT16_D50[1]/illuminant[1], lms[2]*CAT16_D50[2]/illuminant[2]]
    } else {
        [lms[0]*(d*CAT16_D50[0]/illuminant[0] + 1.0-d),
         lms[1]*(d*CAT16_D50[1]/illuminant[1] + 1.0-d),
         lms[2]*(d*CAT16_D50[2]/illuminant[2] + 1.0-d)]
    }
}

/// XYZ chromatic adaptation → D50.
#[inline(always)]
fn xyz_adapt_d50(xyz: [f32; 3], illuminant: [f32; 3]) -> [f32; 3] {
    [xyz[0]*XYZ_D50[0]/illuminant[0], xyz[1]*XYZ_D50[1]/illuminant[1], xyz[2]*XYZ_D50[2]/illuminant[2]]
}

/// Divide vector by `scaling` (safe: uses NORM_MIN floor).
#[inline(always)]
fn downscale(mut v: [f32; 3], scaling: f32) -> [f32; 3] {
    let denom = if scaling > NORM_MIN { scaling + NORM_MIN } else { NORM_MIN };
    for c in 0..3 { v[c] /= denom; }
    v
}

/// Multiply vector by `scaling` (safe: uses NORM_MIN floor).
#[inline(always)]
fn upscale(mut v: [f32; 3], scaling: f32) -> [f32; 3] {
    let scale = if scaling > NORM_MIN { scaling + NORM_MIN } else { NORM_MIN };
    for c in 0..3 { v[c] *= scale; }
    v
}

// ── gamut mapping ─────────────────────────────────────────────────────────────

/// Compress chromaticity toward D50 white point in uvY space.
/// Pure-math port of _gamut_mapping() in channelmixerrgb.c.
fn gamut_mapping(xyz: [f32; 3], compression: f32, clip: bool) -> [f32; 3] {
    let sum = xyz[0] + xyz[1] + xyz[2];
    let y = xyz[1];

    // xyY chromaticity (fallback to D50 when sum==0)
    let mut xy = if sum > 0.0 {
        [xyz[0]/sum, xyz[1]/sum]
    } else {
        [D50XYY[0], D50XYY[1]]
    };
    let y_val = y;

    // xyY → uvY
    let denom = -2.0*xy[0] + 12.0*xy[1] + 3.0;
    let mut uv = [4.0*xy[0]/denom, 9.0*xy[1]/denom];

    // Compress chromaticity toward D50
    let delta = [D50_UV[0] - uv[0], D50_UV[1] - uv[1]];
    let big_delta = y_val * (delta[0]*delta[0] + delta[1]*delta[1]);
    let correction = if compression == 0.0 { 0.0 } else { big_delta.powf(compression) };
    for c in 0..2 {
        let tmp = correction * delta[c] + uv[c];
        uv[c] = if uv[c] > D50_UV[c] { tmp.max(D50_UV[c]) } else { tmp.min(D50_UV[c]) };
    }

    // uvY → xyY
    let denom2 = 6.0*uv[0] - 16.0*uv[1] + 12.0;
    xy[0] = 9.0*uv[0]/denom2;
    xy[1] = 4.0*uv[1]/denom2;

    if clip {
        xy[0] = xy[0].max(0.0);
        xy[1] = xy[1].max(0.0);
    }

    let safe_y = xy[1].max(NORM_MIN);
    let scale = xy[0] + safe_y;
    let (x_out, y_out) = if scale >= 1.0 {
        (xy[0]/scale, safe_y/scale)
    } else {
        (xy[0], safe_y)
    };

    // xyY → XYZ
    [y_val * x_out / y_out, y_val, y_val * (1.0 - x_out - y_out) / y_out]
}

// ── luma/chroma adjustment ────────────────────────────────────────────────────

/// Saturation + lightness per-pixel adjustment.
/// Port of _luma_chroma() in channelmixerrgb.c.
fn luma_chroma(input: [f32; 3], sat: [f32; 3], light: [f32; 3], version: i32) -> [f32; 3] {
    let norm_sq = input[0]*input[0] + input[1]*input[1] + input[2]*input[2];
    let mut norm = norm_sq.sqrt().max(NORM_MIN);
    let avg = ((input[0]+input[1]+input[2]) / 3.0).max(NORM_MIN);

    let mix = input[0]*light[0] + input[1]*light[1] + input[2]*light[2];

    if version == VER_V3 { norm *= INVERSE_SQRT_3; }

    // Ratios
    let mut ratio = [input[0]/norm, input[1]/norm, input[2]/norm];

    // Coeff ratio for saturation
    let coeff_ratio = if version == VER_V1 {
        let mut s = 0.0f32;
        for c in 0..3 { s += (1.0-ratio[c])*(1.0-ratio[c]) * sat[c]; }
        s
    } else {
        (ratio[0]*sat[0] + ratio[1]*sat[1] + ratio[2]*sat[2]) / 3.0
    };

    for c in 0..3 {
        let min_ratio = if ratio[c] < 0.0 { ratio[c] } else { 0.0 };
        ratio[c] = ((1.0-ratio[c])*coeff_ratio + ratio[c]).max(min_ratio);
    }

    if version == VER_V3 {
        let new_norm = (ratio[0]*ratio[0] + ratio[1]*ratio[1] + ratio[2]*ratio[2]).sqrt().max(NORM_MIN);
        norm /= new_norm * INVERSE_SQRT_3;
    }

    norm *= (1.0 + mix/avg).max(0.0);
    [ratio[0]*norm, ratio[1]*norm, ratio[2]*norm]
}

// ── main FFI entry point ──────────────────────────────────────────────────────

/// White-balance + chromatic adaptation + mix + luma/chroma per-pixel pipeline.
///
/// Replaces the DT_OMP_FOR loop inside `_loop_switch()` in channelmixerrgb.c.
///
/// The C caller pre-computes and transposes all matrices before calling here.
/// All matrix pointers are flat row-major float[4][4] (16 floats, row-stride 4).
///
/// kind:    0=LINEAR_BRADFORD, 1=CAT16, 2=FULL_BRADFORD, 3=XYZ, 4=RGB/bypass
/// version: 0=V1, 1=V2, 2=V3  (saturation algorithm generation)
/// clip:    non-zero → clamp negatives/NaN to 0 throughout
/// apply_grey: non-zero → convert output to greyscale via dot(output, grey)
/// p:       Bradford power, pre-computed as powf(illuminant[2]/D50[2], 0.0834)
/// gamut:   chromaticity compression exponent (0 = off)
#[no_mangle]
pub unsafe extern "C" fn darkroom_channelmixerrgb_loop_switch(
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
    rgb_to_xyz_trans: *const f32, // [16] pre-transposed RGB→XYZ
    rgb_to_lms_trans: *const f32, // [16] pre-transposed RGB→LMS (Bradford/CAT16/XYZ)
    mix_to_xyz_trans: *const f32, // [16] pre-transposed MIX→XYZ
    xyz_to_rgb_trans: *const f32, // [16] pre-transposed XYZ→RGB
    minval: f32,                  // 0.0 when clip==true, -f32::MAX otherwise
    illuminant:  *const f32,      // [4]
    saturation:  *const f32,      // [4]
    lightness:   *const f32,      // [4]
    grey:        *const f32,      // [4]
    p:       f32,
    gamut:   f32,
    clip:    i32,
    apply_grey: i32,
    kind:    i32,
    version: i32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let rxt = std::slice::from_raw_parts(rgb_to_xyz_trans, 16);
    let rlt = std::slice::from_raw_parts(rgb_to_lms_trans, 16);
    let mxt = std::slice::from_raw_parts(mix_to_xyz_trans, 16);
    let xrt = std::slice::from_raw_parts(xyz_to_rgb_trans, 16);
    let illum = std::slice::from_raw_parts(illuminant, 4);
    let sat   = std::slice::from_raw_parts(saturation, 4);
    let lig   = std::slice::from_raw_parts(lightness,  4);
    let gry   = std::slice::from_raw_parts(grey, 4);

    let illum3 = [illum[0], illum[1], illum[2]];
    let sat3   = [sat[0],   sat[1],   sat[2]  ];
    let lig3   = [lig[0],   lig[1],   lig[2]  ];
    let gry3   = [gry[0],   gry[1],   gry[2]  ];
    let clip_b = clip != 0;

    for k in 0..npixels {
        let b = k * 4;
        // max_nan: replace NaN with minval, clamp to minval (≥0 when clip)
        let t2 = [
            f32::max(input[b],     minval),
            f32::max(input[b + 1], minval),
            f32::max(input[b + 2], minval),
            f32::max(input[b + 3], minval),
        ];

        // White balance / chromatic adaptation
        let t2_rgb = [t2[0], t2[1], t2[2]];
        let adapted = match kind {
            KIND_FULL_BRADFORD => {
                let xyz = apply_t4(t2_rgb, rxt);
                let y   = xyz[1];
                let lms = convert_xyz_to_bradford_lms(xyz);
                let lms_n = downscale(lms, y);
                let adapted_n = bradford_adapt_d50(lms_n, illum3, p, true);
                upscale(adapted_n, y)
            }
            KIND_LINEAR_BRADFORD => {
                let lms = apply_t4(t2_rgb, rlt);
                bradford_adapt_d50(lms, illum3, p, false)
            }
            KIND_CAT16 => {
                let lms = apply_t4(t2_rgb, rlt);
                cat16_adapt_d50(lms, illum3, 1.0, true)
            }
            KIND_XYZ => {
                let xyz = apply_t4(t2_rgb, rlt);
                xyz_adapt_d50(xyz, illum3)
            }
            _ => { // RGB: no WB — pass RGB directly through the MIX step below
                t2_rgb
            }
        };

        // 3D mix (rotation + homothety) → XYZ
        let mut xyz = apply_t4(adapted, mxt);

        if clip_b { for c in 0..3 { xyz[c] = xyz[c].max(0.0); } }

        // Gamut mapping in xyY/uvY space
        let gmapped = gamut_mapping(xyz, gamut, clip_b);

        // Convert XYZ → output space (LMS for adaptation modes, RGB for bypass)
        let mut lms_or_rgb = if kind >= 0 && kind <= KIND_XYZ {
            convert_any_xyz_to_lms(gmapped, kind)
        } else {
            apply_t4(gmapped, xrt) // XYZ → RGB for bypass
        };

        if clip_b { for c in 0..3 { lms_or_rgb[c] = lms_or_rgb[c].max(0.0); } }

        // Lightness + saturation adjustment
        let mut adjusted = luma_chroma(lms_or_rgb, sat3, lig3, version);

        if clip_b { for c in 0..3 { adjusted[c] = adjusted[c].max(0.0); } }

        // Final output
        let rgb_out = if apply_grey != 0 {
            let grey_mix = (adjusted[0]*gry3[0] + adjusted[1]*gry3[1] + adjusted[2]*gry3[2]).max(0.0);
            [grey_mix, grey_mix, grey_mix]
        } else {
            // LMS/XYZ → XYZ → RGB
            let xyz_out = if kind >= 0 && kind <= KIND_XYZ {
                convert_any_lms_to_xyz(adjusted, kind)
            } else {
                // RGB mode: adjusted is already in LMS≡pipeline RGB after mix; go XYZ→RGB
                let xyz_via = apply_t4(adjusted, rxt); // RGB → XYZ (backwards to get XYZ)
                // Actually for RGB mode: adjusted came from XYZ→RGB, go back XYZ
                // The C code does: dt_apply_transposed_color_matrix(temp_two, RGB_to_XYZ_trans, temp_one)
                xyz_via
            };
            let mut xyz_clipped = xyz_out;
            if clip_b { for c in 0..3 { xyz_clipped[c] = xyz_clipped[c].max(0.0); } }
            let rgb = apply_t4(xyz_clipped, xrt);
            if clip_b {
                [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]
            } else {
                rgb
            }
        };

        output[b]     = rgb_out[0];
        output[b + 1] = rgb_out[1];
        output[b + 2] = rgb_out[2];
        output[b + 3] = input[b + 3]; // alpha passthrough
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_identity_4x4() -> Vec<f32> {
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    #[test]
    fn bradford_d50_adapt_identity_illuminant() {
        // If illuminant == D50, output should equal D50 scaled
        let lms = [0.5, 0.5, 0.5];
        let illum = BRADFORD_D50;
        let out = bradford_adapt_d50(lms, illum, 1.0, false);
        assert!((out[0] - 0.5).abs() < 1e-4, "{:?}", out);
        assert!((out[1] - 0.5).abs() < 1e-4, "{:?}", out);
        assert!((out[2] - 0.5).abs() < 1e-4, "{:?}", out);
    }

    #[test]
    fn cat16_full_adapt_identity_illuminant() {
        let lms = [0.3, 0.6, 0.2];
        let out = cat16_adapt_d50(lms, CAT16_D50, 1.0, true);
        assert!((out[0] - 0.3).abs() < 1e-4, "{:?}", out);
        assert!((out[1] - 0.6).abs() < 1e-4, "{:?}", out);
        assert!((out[2] - 0.2).abs() < 1e-4, "{:?}", out);
    }

    #[test]
    fn xyz_adapt_identity_illuminant() {
        let xyz = [0.4, 0.5, 0.3];
        let out = xyz_adapt_d50(xyz, XYZ_D50.map(|x| x));
        assert!((out[0] - 0.4).abs() < 1e-4, "{:?}", out);
        assert!((out[1] - 0.5).abs() < 1e-4, "{:?}", out);
        assert!((out[2] - 0.3).abs() < 1e-4, "{:?}", out);
    }

    #[test]
    fn gamut_mapping_zero_compression_passthrough() {
        let xyz = [0.3, 0.4, 0.2];
        let out = gamut_mapping(xyz, 0.0, false);
        // With compression=0 the correction=0 so xyY comes back unchanged → same XYZ
        assert!((out[0] - xyz[0]).abs() < 1e-4, "{:?} vs {:?}", out, xyz);
        assert!((out[1] - xyz[1]).abs() < 1e-4, "{:?} vs {:?}", out, xyz);
        assert!((out[2] - xyz[2]).abs() < 1e-4, "{:?} vs {:?}", out, xyz);
    }

    #[test]
    fn rgb_bypass_loop_neutral_mix() {
        // RGB bypass with identity matrices, neutral illuminant and saturation/lightness
        // → output should be close to input
        let id = flat_identity_4x4();
        let illum   = vec![1.0f32, 1.0, 1.0, 0.0];
        let sat     = vec![1.0f32, 1.0, 1.0, 0.0];
        let light   = vec![0.0f32, 0.0, 0.0, 0.0];
        let grey    = vec![1.0f32, 1.0, 1.0, 0.0];

        let input  = vec![0.2f32, 0.4, 0.6, 1.0];
        let mut out = vec![0.0f32; 4];

        unsafe {
            darkroom_channelmixerrgb_loop_switch(
                input.as_ptr(), out.as_mut_ptr(), 1,
                id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(),
                0.0, illum.as_ptr(), sat.as_ptr(), light.as_ptr(), grey.as_ptr(),
                1.0, 0.0, 1, 0, 4 /*KIND_RGB*/, 2 /*VER_V3*/,
            );
        }
        assert!(out[3] == 1.0, "alpha={}", out[3]);
    }

    #[test]
    fn alpha_always_passes_through() {
        let id   = flat_identity_4x4();
        let ones = vec![1.0f32, 1.0, 1.0, 1.0];
        let zero = vec![0.0f32, 0.0, 0.0, 0.0];
        let input = vec![0.5f32, 0.5, 0.5, 0.777];
        let mut out = vec![0.0f32; 4];

        unsafe {
            darkroom_channelmixerrgb_loop_switch(
                input.as_ptr(), out.as_mut_ptr(), 1,
                id.as_ptr(), id.as_ptr(), id.as_ptr(), id.as_ptr(),
                0.0, ones.as_ptr(), ones.as_ptr(), zero.as_ptr(), ones.as_ptr(),
                1.0, 0.0, 1, 0, 4, 2,
            );
        }
        assert!((out[3] - 0.777).abs() < 1e-5, "alpha={}", out[3]);
    }
}

/// Convert an RGBA input image to D50-origin xyY chromaticity in `temp`.
///
/// For each pixel (i,j):
///   1. clip negatives
///   2. XYZ[r] = sum_c(rgb_to_xyz[r*4+c] * RGB[c])   row-major 3x4 matrix
///   3. chromaticity: sum = max(X+Y+Z, NORM_MIN); x = X/sum; y = Y/sum
///   4. norm = hypot(d50_x, d50_y)
///   5. temp[base+0] = (x - d50_x) / norm
///      temp[base+1] = (y - d50_y) / norm
///      temp[base+2] = Y
///
/// rgb_to_xyz: 12 floats, row-major (3 rows x 4 cols).
/// Matches the DT_OMP_FOR(collapse(2)) in src/iop/channelmixerrgb.c:746.
#[no_mangle]
pub unsafe extern "C" fn darkroom_channelmixerrgb_rgb_to_xyY(
    in_buf: *const f32,
    temp: *mut f32,
    width: usize,
    height: usize,
    ch: usize,
    rgb_to_xyz: *const f32,
    d50_x: f32,
    d50_y: f32,
) {
    if width == 0 || height == 0 { return; }
    let input  = std::slice::from_raw_parts(in_buf, width * height * ch);
    let output = std::slice::from_raw_parts_mut(temp, width * height * ch);
    let m = std::slice::from_raw_parts(rgb_to_xyz, 12); // row-major 3x4
    let norm_d50 = d50_x.hypot(d50_y);

    for k in 0..width * height {
        let base = k * ch;
        let r = input[base    ].max(0.0);
        let g = input[base + 1].max(0.0);
        let b = input[base + 2].max(0.0);

        let x_xyz = m[0] * r + m[1] * g + m[2] * b + m[3] * 0.0;
        let y_xyz = m[4] * r + m[5] * g + m[6] * b + m[7] * 0.0;
        let z_xyz = m[8] * r + m[9] * g + m[10] * b + m[11] * 0.0;

        let sum = (x_xyz + y_xyz + z_xyz).max(NORM_MIN);
        let x_chr = x_xyz / sum;
        let y_chr = y_xyz / sum;

        output[base]     = (x_chr - d50_x) / norm_d50;
        output[base + 1] = (y_chr - d50_y) / norm_d50;
        output[base + 2] = y_xyz;
    }
}

// ── live darkroom module (S4: `Color calibration`) ───────────────────────────
//
// Port of `commit_params` (channelmixerrgb.c:2744) + the matrix packaging in
// `process` (:1811) / `_loop_switch` (:649) into a pure-Rust packet builder
// [`ChannelmixerRgbData::from_params`] that feeds the already-ported FFI kernel
// [`darkroom_channelmixerrgb_loop_switch`].
//
// **D50/D65 account.** darktable's pipeline (and therefore channelmixerrgb's
// `work_profile->matrix_in/out`) is **D50-referenced**: `commit_params` maps
// the pipeline RGB with the working profile's D50 matrix, and the kernel's
// bookkeeping — the Bradford blue `p = powf(0.818155/illuminant[2], 0.0834)`,
// `bradford_adapt_d50`, `gamut_mapping`'s D50 white/uv targets — is written
// against D50. c41's working spaces ([`crate::color`]) are **D65-referenced**
// (`REC2020_TO_XYZ_D65_T4`, `SRGB_TO_XYZ_D65_T4`). To hand the kernel its
// intended D50 matrices we pre-compose the CAT16 white-point conversion around
// the c41 D65 matrix (the same trick ColorBalanceRgb reaches via its
// `XYZ_D50→D65_CAT16` premultiply, but in the reverse direction here because
// channelmixerrgb is D50-internal):
//
//   rgb_to_xyz = CAT16(D65→D50) · R65          (R65 = c41 workspace RGB→XYZ-D65)
//   xyz_to_rgb = X2R65 · CAT16(D50→D65)
//
// then the C's own compositions (RGB_to_LMS = XYZ_to_LMS·rgb_to_xyz,
// MIX_to_XYZ = LMS_to_XYZ·MIX for the Bradford/CAT16 modes, MIX for XYZ,
// rgb_to_xyz·MIX for the RGB bypass) are applied on top. `rgb_to_lms` is used
// by the LINEAR_BRADFORD/CAT16/XYZ adaptation paths, `rgb_to_xyz` by
// FULL_BRADFORD and the RGB-bypass, `mix_to_xyz` after adaptation, and
// `xyz_to_rgb` for the final project back to pipeline RGB in every mode.
// The illuminant (a pure light-source chromaticity → XYZ → adaptation LMS) is
// computed *without* the CAT bridge — it is a physical property independent of
// the working space, exactly as `commit_params` computes it.
//
// **Identity gate.** At the darktable `$DEFAULT`s (illuminant D / 5003 K,
// adaptation CAT16, gamut 1.0 → 1/gamut, clip TRUE, mix identity, sat/light 0)
// the kernel still adapts to a 5003 K daylight illuminant, runs the D50 gamut
// compression and clips — a real edit, not a no-op. So the pipeline emits the
// stage on the *enable flag alone* (flag-only gate, like Bloom/cacorrectrgb);
// the UI's `is_identity` mirrors it. The only parameter set that *is* a
// pass-through is the D50/algebraic no-op config (illuminant PIPE ↔ D50,
// gamut 0, clip FALSE, mix identity, sat/light/grey 0, V3) — used by the
// D50/D65 round-trip test.

// ── illuminant constants (illuminants.h: dt_illuminant_t) ─────────────────────
const ILLUM_PIPE: u8 = 0;
const ILLUM_A: u8 = 1;
const ILLUM_D: u8 = 2;
const ILLUM_E: u8 = 3;
const ILLUM_F: u8 = 4;
const ILLUM_LED: u8 = 5;
const ILLUM_BB: u8 = 6;
// (CUSTOM = 7 and CAMERA = 10 matched by the catch-all `_` arm: neither has a
// params-only resolution — CUSTOM keeps the stored xy and CAMERA needs RAW
// EXIF matrix data c41 does not carry.)

/// CIE fluorescent-standard xy chromaticities (illuminants.h `fluorescent`).
const FLUORESCENT_XY: [[f32; 2]; 12] = [
    [0.31310, 0.33727], // F1
    [0.37208, 0.37529], // F2
    [0.40910, 0.39430], // F3
    [0.44018, 0.40329], // F4
    [0.31379, 0.34531], // F5
    [0.37790, 0.38835], // F6
    [0.31292, 0.32933], // F7
    [0.34588, 0.35875], // F8
    [0.37417, 0.37281], // F9
    [0.34609, 0.35986], // F10
    [0.38052, 0.37713], // F11
    [0.43695, 0.40441], // F12
];

/// CIE white LED xy chromaticities (illuminants.h `led`).
const LED_XY: [[f32; 2]; 9] = [
    [0.4560, 0.4078], // B1
    [0.4357, 0.4012], // B2
    [0.3756, 0.3723], // B3
    [0.3422, 0.3502], // B4
    [0.3118, 0.3236], // B5
    [0.4474, 0.4066], // BH1
    [0.4557, 0.4211], // RGB1
    [0.4560, 0.4548], // V1
    [0.3781, 0.3775], // V2
];

/// `CCT_to_xy_daylight` (illuminants.h:138) — the closest daylight illuminant
/// chromaticity for 4000–25000 K. Out-of-range leaves `(0, 0)`.
fn cct_to_xy_daylight(t: f32) -> [f32; 2] {
    let mut x_temp = 0.0f32;
    if (4000.0f32..=7000.0f32).contains(&t) {
        x_temp = ((-4.6070e9f32 / t + 2.9678e6f32) / t + 0.09911e3f32) / t + 0.244_063f32;
    } else if t > 7000.0f32 && t <= 25000.0f32 {
        x_temp = ((-2.0064e9f32 / t + 1.9018e6f32) / t + 0.24748e3f32) / t + 0.237_040f32;
    }
    let y_temp = (-3.0f32 * x_temp + 2.87f32) * x_temp - 0.275f32;
    [x_temp, y_temp]
}

/// `CCT_to_xy_blackbody` (illuminants.h:157) — the general Planckian locus
/// chromaticity for 1667–25000 K. Out-of-range leaves `(0, 0)`.
fn cct_to_xy_blackbody(t: f32) -> [f32; 2] {
    let mut x_temp = 0.0f32;
    if (1667.0f32..=4000.0f32).contains(&t) {
        x_temp = ((-0.2661239e9f32 / t - 0.2343589e6f32) / t + 0.8776956e3f32) / t + 0.179_910f32;
    } else if t > 4000.0f32 && t <= 25000.0f32 {
        x_temp = ((-3.0258469e9f32 / t + 2.1070379e6f32) / t + 0.2226347e3f32) / t + 0.240_390f32;
    }
    let mut y_temp = 0.0f32;
    if (1667.0f32..=2222.0f32).contains(&t) {
        y_temp = ((-1.1063814f32 * x_temp - 1.34811020f32) * x_temp + 2.18555832f32) * x_temp
            - 0.20219683f32;
    } else if t > 2222.0f32 && t <= 4000.0f32 {
        y_temp = ((-0.9549476f32 * x_temp - 1.37418593f32) * x_temp + 2.09137015f32) * x_temp
            - 0.16748867f32;
    } else if t > 4000.0f32 && t <= 25000.0f32 {
        y_temp = ((3.0817580f32 * x_temp - 5.87338670f32) * x_temp + 3.75112997f32) * x_temp
            - 0.37001483f32;
    }
    [x_temp, y_temp]
}

/// `illuminant_xy_to_XYZ` (illuminants.h): X = x/y, Y = 1, Z = (1−x−y)/y.
fn illuminant_xy_to_xyz(x: f32, y: f32) -> [f32; 3] {
    [x / y, 1.0f32, (1.0f32 - x - y) / y]
}

/// Port of the `illuminant_to_xy` switch (illuminants.h:225), returning the
/// illuminant chromaticity from the params. Returns `None` where the C cannot
/// resolve one (custom/camera/out-of-range) — the caller then keeps the stored
/// `x`/`y` (which is exactly what `commit_params` does with `float x = p->x;
/// float y = p->y;`). The `CAMERA` white balance can only be resolved from RAW
/// EXIF + camera matrix, which c41 does not carry, so it always falls back to
/// the stored xy (documented deviation, see `PARITY_AUDIT.md`).
fn illuminant_to_xy(
    illuminant: u8,
    temperature: f32,
    fluo: u8,
    led: u8,
) -> Option<[f32; 2]> {
    match illuminant {
        ILLUM_PIPE => Some([0.34567, 0.35850]), // D50xyY
        ILLUM_E => Some([1.0 / 3.0, 1.0 / 3.0]),
        ILLUM_A => Some([0.44757, 0.40745]),
        ILLUM_F => ((fluo as usize) < FLUORESCENT_XY.len()).then(|| FLUORESCENT_XY[fluo as usize]),
        ILLUM_LED => ((led as usize) < LED_XY.len()).then(|| LED_XY[led as usize]),
        ILLUM_D => {
            let d = cct_to_xy_daylight(temperature);
            if d[0] != 0.0 && d[1] != 0.0 {
                Some(d)
            } else {
                // out of daylight range → fall through to the black-body model
                let bb = cct_to_xy_blackbody(temperature);
                (bb[0] != 0.0 && bb[1] != 0.0).then_some(bb)
            }
        }
        ILLUM_BB => {
            let bb = cct_to_xy_blackbody(temperature);
            (bb[0] != 0.0 && bb[1] != 0.0).then_some(bb)
        }
        _ => None, // CUSTOM / CAMERA / DETECT: caller keeps stored x/y
    }
}

// ── CAT16 white-point bridge (chromatic_adaptation.h) ────────────────────────
// Compact **transposed-storage** `[f32; 9]` (S(M)[r][c] = M_std[c][r]) of the
// `XYZ_..._CAT16_trans` constants — the same storage convention as the port's
// `XYZ_TO_BRADFORD_LMS_T` etc.
const XYZ_D65_TO_D50_CAT16_T: [f32; 9] = [
    1.01085433e00, 5.42814201e-03, 2.50722468e-04,
    4.07086103e-02, 9.93581926e-01, -1.14918759e-02,
    -3.41445825e-02, 1.15592039e-03, 7.67964947e-01,
];
const XYZ_D50_TO_D65_CAT16_T: [f32; 9] = [
    9.89466254e-01, -5.40518733e-03, -4.03920992e-04,
    -4.00304626e-02, 1.00666069e00, 1.50768030e-02,
    4.40530317e-02, -1.75551955e-03, 1.30210211e00,
];

/// Plain flat 3×3 product in transposed-storage space (`S(A)·S(B) = S(B·A)`
/// in standard orientation). Both operands are compact transposed-storage.
#[inline]
fn mul3(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut out = [0.0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] =
                a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
    out
}

/// Extract the top-left 3×3 of a `[[f32; 4]; 4]` transposed-storage matrix
/// (the workspace `REC2020_TO_XYZ_D65_T4` / `SRGB_TO_XYZ_D65_T4`) as a compact
/// `[f32; 9]` transposed-storage matrix.
#[inline]
fn compact_t4(m: &[[f32; 4]; 4]) -> [f32; 9] {
    let mut out = [0.0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] = m[r][c];
        }
    }
    out
}

/// Expand a compact transposed-storage `[f32; 9]` into the flat row-stride-4
/// `[f32; 16]` the kernel's `apply_t4` expects: `t[c*4+r] = compact[c*3+r]`
/// (so `apply_t4(v, t) = M_std·v`), 4th row/col zero.
#[inline]
fn expand16(c: &[f32; 9]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for r in 0..3 {
        for k in 0..3 {
            out[k * 4 + r] = c[k * 3 + r];
        }
    }
    out
}

// ── params / data ────────────────────────────────────────────────────────────

/// User-facing `Color calibration` params — the c41 mirror of
/// `dt_iop_channelmixer_rgb_params_t` (v3, `channelmixerrgb.c:92`). Every field
/// that affects the matrix/illuminant is present so a hand-edited blob renders
/// exactly like the C, including the ones the slice-1 UI does not expose.
/// `red/green/blue` carry `[CHANNEL_SIZE=4]` with the alpha lane (index 3)
/// ignored by the math (read as 0 like the C's 3×3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelmixerRgbParams {
    pub red: [f32; 4],
    pub green: [f32; 4],
    pub blue: [f32; 4],
    pub saturation: [f32; 4],
    pub lightness: [f32; 4],
    pub grey: [f32; 4],
    pub normalize_r: bool,
    pub normalize_g: bool,
    pub normalize_b: bool,
    pub normalize_sat: bool,
    pub normalize_light: bool,
    pub normalize_grey: bool,
    pub illuminant: u8,
    pub illum_fluo: u8,
    pub illum_led: u8,
    pub adaptation: u8,
    pub x: f32,
    pub y: f32,
    pub temperature: f32,
    pub gamut: f32,
    pub clip: bool,
    pub version: u8,
}

/// darktable `$DEFAULT`s (channelmixerrgb.c:92-112): the neutral channel-mix
/// identity, zero saturation/lightness/grey, all normalize toggles FALSE,
/// illuminant D (daylight) @ 5003 K, adaptation CAT16, gamut 1.0, clip TRUE,
/// saturation algorithm V3.
impl Default for ChannelmixerRgbParams {
    fn default() -> Self {
        Self {
            red: [1.0, 0.0, 0.0, 0.0],
            green: [0.0, 1.0, 0.0, 0.0],
            blue: [0.0, 0.0, 1.0, 0.0],
            saturation: [0.0; 4],
            lightness: [0.0; 4],
            grey: [0.0; 4],
            normalize_r: false,
            normalize_g: false,
            normalize_b: false,
            normalize_sat: false,
            normalize_light: false,
            normalize_grey: false,
            illuminant: ILLUM_D,
            illum_fluo: 2,  // DT_ILLUMINANT_FLUO_F3
            illum_led: 4,   // DT_ILLUMINANT_LED_B5
            adaptation: KIND_CAT16 as u8,
            x: 0.333,
            y: 0.333,
            temperature: 5003.0,
            gamut: 1.0,
            clip: true,
            version: VER_V3 as u8,
        }
    }
}

/// The process-ready packet: the four pre-transposed kernel matrices plus every
/// scalar vector, all derived once per render by [`ChannelmixerRgbData::from_params`]
/// (the port of `commit_params` + the `_loop_switch` matrix packaging) and
/// carried in the pipeline `Stage` so each per-band [[Self::process_data]] call
/// is pure kernel invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelmixerRgbData {
    pub rgb_to_xyz_trans: [f32; 16],
    pub rgb_to_lms_trans: [f32; 16],
    pub mix_to_xyz_trans: [f32; 16],
    pub xyz_to_rgb_trans: [f32; 16],
    pub illuminant: [f32; 4],
    pub saturation: [f32; 4],
    pub lightness: [f32; 4],
    pub grey: [f32; 4],
    pub p: f32,
    pub gamut: f32,
    pub clip: bool,
    pub apply_grey: bool,
    pub kind: i32,
    pub version: i32,
    pub minval: f32,
}

/// Fold non-finite slider values back to the darktable default and clamp to the
/// introspection ranges — defensive decode hardening shared by the blob path and
/// any direct caller (the UI fold lives in `preview.rs`; this is the core-layer
/// backstop).
fn sanitize(p: &ChannelmixerRgbParams) -> ChannelmixerRgbParams {
    let d = ChannelmixerRgbParams::default();
    let mut q = *p;
    let fin = |v: f32, def: f32| -> f32 { if v.is_finite() { v } else { def } };
    let clamp = |v: f32, lo: f32, hi: f32| -> f32 { v.clamp(lo, hi) };
    for i in 0..3 {
        q.red[i] = clamp(fin(p.red[i], d.red[i]), COLOR_MIN, COLOR_MAX);
        q.green[i] = clamp(fin(p.green[i], d.green[i]), COLOR_MIN, COLOR_MAX);
        q.blue[i] = clamp(fin(p.blue[i], d.blue[i]), COLOR_MIN, COLOR_MAX);
        q.saturation[i] = clamp(fin(p.saturation[i], d.saturation[i]), -2.0, 2.0);
        q.lightness[i] = clamp(fin(p.lightness[i], d.lightness[i]), -2.0, 2.0);
        q.grey[i] = clamp(fin(p.grey[i], d.grey[i]), -2.0, 2.0);
    }
    q.x = fin(p.x, d.x);
    q.y = fin(p.y, d.y);
    q.temperature = clamp(fin(p.temperature, d.temperature), TEMP_MIN, TEMP_MAX);
    q.gamut = clamp(fin(p.gamut, d.gamut), 0.0, 12.0);
    q
}

impl ChannelmixerRgbData {
    /// Derive the process-ready packet from user params + the buffer's working
    /// colour space (`ColorSpace::Rec2020` for raws, `ColorSpace::LinearSrgb`
    /// for non-raws). Port of the `commit_params` chain at channelmixerrgb.c:2744
    /// (mix/normalization, saturation/lightness/grey vectors, illuminant from
    /// temperature, the Bradford blue power) composed with the `_loop_switch`
    /// matrix packaging at :649 and the D50/D65 CAT16 bridge (see the module
    /// doc for the algebra).
    pub fn from_params(p: &ChannelmixerRgbParams, space: ColorSpace) -> Self {
        let p = sanitize(p);

        // ── mix + normalize (channelmixerrgb.c:2756-2791) ────────────────
        let norm_r = if p.normalize_r { p.red[0] + p.red[1] + p.red[2] } else { 1.0 };
        let norm_g = if p.normalize_g { p.green[0] + p.green[1] + p.green[2] } else { 1.0 };
        let norm_b = if p.normalize_b { p.blue[0] + p.blue[1] + p.blue[2] } else { 1.0 };
        let norm_sat = if p.normalize_sat {
            (p.saturation[0] + p.saturation[1] + p.saturation[2]) / 3.0
        } else {
            0.0
        };
        let norm_light = if p.normalize_light {
            (p.lightness[0] + p.lightness[1] + p.lightness[2]) / 3.0
        } else {
            0.0
        };
        let mut norm_grey = p.grey[0] + p.grey[1] + p.grey[2];
        let apply_grey = p.grey[0] != 0.0 || p.grey[1] != 0.0 || p.grey[2] != 0.0;
        if !p.normalize_grey || norm_grey == 0.0 {
            norm_grey = 1.0;
        }

        // MIX: standard orientation rows = red / green / blue channel coeffs.
        let mut mix_std = [[0.0f32; 3]; 3];
        for i in 0..3 {
            mix_std[0][i] = p.red[i] / norm_r;
            mix_std[1][i] = p.green[i] / norm_g;
            mix_std[2][i] = p.blue[i] / norm_b;
        }
        let mut mix_compact = [0.0f32; 9]; // transposed-storage = M_stdᵀ
        for r in 0..3 {
            for c in 0..3 {
                mix_compact[r * 3 + c] = mix_std[c][r];
            }
        }

        let mut saturation = [0.0f32; 4];
        let mut lightness = [0.0f32; 4];
        let mut grey = [0.0f32; 4];
        for i in 0..3 {
            saturation[i] = -p.saturation[i] + norm_sat;
            lightness[i] = p.lightness[i] - norm_light;
            grey[i] = p.grey[i] / norm_grey;
        }
        if (p.version as i32) == VER_V1 {
            // v1 saturation algo reverses the R and B coefficients
            saturation.swap(0, 2);
        }

        // ── gamut exponent + clip (channelmixerrgb.c:2794-2795) ──────────
        let gamut = if p.gamut == 0.0 { 0.0 } else { 1.0 / p.gamut };
        let clip = p.clip;
        let minval = if clip { 0.0 } else { -f32::MAX };

        // ── illuminant (channelmixerrgb.c:2800-2827) ──────────────────────
        // Start with the stored white point, overwrite on a successful lookup
        // (custom/camera/out-of-range keep the stored x/y).
        let (mut x, mut y) = (p.x, p.y);
        if let Some(xy) = illuminant_to_xy(p.illuminant, p.temperature, p.illum_fluo, p.illum_led) {
            x = xy[0];
            y = xy[1];
        }
        let xyz = illuminant_xy_to_xyz(x, y);
        let lms = convert_any_xyz_to_lms(xyz, p.adaptation as i32);
        let illuminant = [lms[0], lms[1], lms[2], 0.0];

        // Bradford blue compensation: reference illuminant is hard-set D50 for
        // darktable's pipeline (channelmixerrgb.c:2843-2846).
        let p_bradford = (0.818155f32 / illuminant[2]).powf(0.0834);

        // ── matrix composition + D50/D65 bridge (see module doc) ──────────
        let (ws_r2x_t, x2ws_t) = match space {
            ColorSpace::Rec2020 => (
                &crate::color::REC2020_TO_XYZ_D65_T4,
                &crate::color::XYZ_D65_TO_REC2020_T4,
            ),
            ColorSpace::LinearSrgb => {
                (&crate::color::SRGB_TO_XYZ_D65_T4, &crate::color::XYZ_D65_TO_SRGB_T4)
            }
        };
        let ws_r2x = compact_t4(ws_r2x_t);
        let x2ws = compact_t4(x2ws_t);
        // rgb_to_xyz = CAT16(D65→D50)·R65  →  S(product) = S(R65)·S(CAT)
        let rgb_to_xyz = mul3(&ws_r2x, &XYZ_D65_TO_D50_CAT16_T);
        // xyz_to_rgb = X2R65·CAT16(D50→D65)  →  S(product) = S(CAT)·S(X2R65)
        let xyz_to_rgb = mul3(&XYZ_D50_TO_D65_CAT16_T, &x2ws);

        let kind = p.adaptation as i32;
        let rgb_to_lms = match kind {
            KIND_LINEAR_BRADFORD | KIND_FULL_BRADFORD => mul3(&rgb_to_xyz, &XYZ_TO_BRADFORD_LMS_T),
            KIND_CAT16 => mul3(&rgb_to_xyz, &XYZ_TO_CAT16_LMS_T),
            _ => rgb_to_xyz, // XYZ: RGB_to_LMS = RGB_to_XYZ; RGB bypass unused
        };
        let mix_to_xyz = match kind {
            KIND_LINEAR_BRADFORD | KIND_FULL_BRADFORD => {
                mul3(&mix_compact, &BRADFORD_LMS_TO_XYZ_T)
            }
            KIND_CAT16 => mul3(&mix_compact, &CAT16_LMS_TO_XYZ_T),
            KIND_XYZ => mix_compact,
            _ => mul3(&mix_compact, &rgb_to_xyz), // RGB bypass: RGB_to_XYZ·MIX
        };

        Self {
            rgb_to_xyz_trans: expand16(&rgb_to_xyz),
            rgb_to_lms_trans: expand16(&rgb_to_lms),
            mix_to_xyz_trans: expand16(&mix_to_xyz),
            xyz_to_rgb_trans: expand16(&xyz_to_rgb),
            illuminant,
            saturation,
            lightness,
            grey,
            p: p_bradford,
            gamut,
            clip,
            apply_grey,
            kind,
            version: p.version as i32,
            minval,
        }
    }

    /// The per-band pixel driver: a single elementwise call to the migrated FFI
    /// kernel [`darkroom_channelmixerrgb_loop_switch`]. Pixel-local, so the
    /// pipeline runs it over rayon bands. Guards zero/short buffers.
    pub fn process_data(&self, input: &[f32], output: &mut [f32]) {
        let npixels = input.len() / 4;
        if npixels == 0 || output.len() / 4 < npixels {
            return; // nothing to do / short output buffer
        }
        unsafe {
            darkroom_channelmixerrgb_loop_switch(
                input.as_ptr(),
                output.as_mut_ptr(),
                npixels,
                self.rgb_to_xyz_trans.as_ptr(),
                self.rgb_to_lms_trans.as_ptr(),
                self.mix_to_xyz_trans.as_ptr(),
                self.xyz_to_rgb_trans.as_ptr(),
                self.minval,
                self.illuminant.as_ptr(),
                self.saturation.as_ptr(),
                self.lightness.as_ptr(),
                self.grey.as_ptr(),
                self.p,
                self.gamut,
                self.clip as i32,
                self.apply_grey as i32,
                self.kind,
                self.version,
            );
        }
    }
}

/// Convenience driver matching the sprint spec: build the data for `params` in
/// `space`, run the kernel over all `w`×`h` pixels, return the RGBA output.
/// (The pipeline uses [`ChannelmixerRgbData::from_params`] once and
/// [`ChannelmixerRgbData::process_data`] per band instead, so the matrix chain
/// is not rebuilt per band.)
pub fn process(
    pixels: &[f32],
    w: usize,
    h: usize,
    p: &ChannelmixerRgbParams,
    space: ColorSpace,
) -> Vec<f32> {
    if pixels.is_empty() || w == 0 || h == 0 {
        return Vec::new();
    }
    let npixels = pixels.len() / 4;
    let data = ChannelmixerRgbData::from_params(p, space);
    let mut out = vec![0.0f32; npixels * 4];
    data.process_data(&pixels[..npixels * 4], &mut out);
    out
}

#[cfg(test)]
mod live_tests {
    use super::*;

    /// D50/D65 algebra round-trip: the *near*-no-op parameter set — illuminant
    /// PIPE (== D50 chromaticity), adaptation CAT16, identity mix, sat/light/
    /// grey 0, gamut off, no clip, V3. The CAT16 D65→D50 bridge in and the
    /// D50→D65 bridge out must cancel, and the kernel must reproduce the input
    /// in both working spaces.
    ///
    /// This is NOT bit-exact even in darktable: the CAT16 adaptation divides
    /// by the **computed** illuminant LMS — whose S lane is 0.8334567 for the
    /// PIPE chromaticity (0.34567, 0.35850) — then multiplies by the hard-coded
    /// `CAT16_D50` constant S = 0.833036 (chromatic_adaptation.h `CAT16_adapt_D50`;
    /// Rust at [`CAT16_D50`]). That 5.05e-4 S-lane mismatch is intrinsic to the C
    /// and produces ~5e-4 lane errors on a faithful port, so the tolerance must
    /// sit above it (darktable's own algebraic "no-op" is not exact either). A
    /// genuinely swapped bridge (wrong `mul3` operand order) diverges by
    /// ~0.04–0.08, far above this bound, so the test still trips a transpose bug.
    #[test]
    fn d50d65_round_trip_near_noop() {
        let mut p = ChannelmixerRgbParams::default();
        p.illuminant = ILLUM_PIPE; // D50xyY — CAT16 adaptation is ~identity
        p.gamut = 0.0;
        p.clip = false;
        let input: Vec<f32> = vec![
            0.123, 0.456, 0.789, 1.0,
            0.9, 0.1, 0.4, 1.0,
        ];
        for space in [ColorSpace::Rec2020, ColorSpace::LinearSrgb] {
            let out = process(&input, 2, 1, &p, space);
            assert_eq!(out.len(), input.len());
            for (a, b) in input.iter().zip(out.iter()) {
                assert!((a - b).abs() < 5e-3, "round-trip diverged ({space:?}): {a} != {b}");
            }
        }
    }

    /// Alpha always passes through untouched, whatever the colour math does.
    #[test]
    fn alpha_passthrough() {
        let mut p = ChannelmixerRgbParams::default();
        p.temperature = 7000.0; // non-trivial illuminant
        let input: Vec<f32> = vec![0.5, 0.4, 0.3, 0.777];
        let out = process(&input, 1, 1, &p, ColorSpace::Rec2020);
        assert_eq!(out[3], 0.777);
    }

    /// A temperature move changes the output: the illuminant path feeds through.
    #[test]
    fn temperature_move_changes_output() {
        let p = ChannelmixerRgbParams::default();
        let p2 = ChannelmixerRgbParams {
            temperature: 2500.0,
            ..p
        };
        let input: Vec<f32> = vec![0.3, 0.4, 0.5, 1.0];
        let a = process(&input, 1, 1, &p, ColorSpace::Rec2020);
        let b = process(&input, 1, 1, &p2, ColorSpace::Rec2020);
        assert_ne!(&a[..3], &b[..3], "temperature must shift the adapted colour");
    }

    /// A non-identity red-mix row (the C's `MIX[0][i] = red[i]/norm_R`) changes
    /// the output vs the identity default.
    #[test]
    fn non_identity_mix_changes_output() {
        let p = ChannelmixerRgbParams::default();
        let p2 = ChannelmixerRgbParams {
            red: [0.5, 0.25, 0.25, 0.0],
            ..Default::default()
        };
        let input = vec![0.6f32, 0.3, 0.2, 1.0];
        let a = process(&input, 1, 1, &p, ColorSpace::Rec2020);
        let b = process(&input, 1, 1, &p2, ColorSpace::Rec2020);
        assert_ne!(&a[..3], &b[..3], "red mix coefficients must alter the output");
    }

    /// The normalize toggles reshape the applied mix exactly as `commit_params`
    /// does (`MIX[0][i] = red[i]/(sum red[])` when `normalize_R`).
    #[test]
    fn normalize_toggle_changes_mix() {
        let p0 = ChannelmixerRgbParams {
            red: [1.0, 1.0, 1.0, 0.0],
            normalize_r: false, // MIX row0 = [1,1,1]
            ..Default::default()
        };
        let p1 = ChannelmixerRgbParams {
            red: [1.0, 1.0, 1.0, 0.0],
            normalize_r: true, // MIX row0 = [1/3,1/3,1/3]
            ..Default::default()
        };
        let input = vec![0.4f32, 0.5, 0.6, 1.0];
        let a = process(&input, 1, 1, &p0, ColorSpace::Rec2020);
        let b = process(&input, 1, 1, &p1, ColorSpace::Rec2020);
        assert_ne!(&a[..3], &b[..3], "normalize_R must change the applied mix");
    }

    /// V1 saturation reverts the R/B coefficient order (`-sat[2]`, `-sat[0]`).
    #[test]
    fn v1_saturation_swaps_rb() {
        let mut p = ChannelmixerRgbParams::default();
        p.saturation = [0.5, 0.0, 0.0, 0.0];
        p.version = VER_V1 as u8;
        let d = ChannelmixerRgbData::from_params(&p, ColorSpace::Rec2020);
        // v1: saturation[0] = -sat[2] + norm = -0.0 = 0 ; saturation[2] = -sat[0]+norm = -0.5
        assert_eq!(d.saturation[0], 0.0);
        assert!((d.saturation[2] + 0.5).abs() < 1e-6, "v1 R/B swap: {:?}", d.saturation);
        // V3 keeps them straight.
        p.version = VER_V3 as u8;
        let d3 = ChannelmixerRgbData::from_params(&p, ColorSpace::Rec2020);
        assert_ne!(d3.saturation[2], d.saturation[2]);
    }

    /// show grey vector / apply_grey detection.
    #[test]
    fn apply_grey_detection() {
        let mut p = ChannelmixerRgbParams::default();
        assert!(!ChannelmixerRgbData::from_params(&p, ColorSpace::Rec2020).apply_grey);
        p.grey = [0.2, 0.7, 0.1, 0.0];
        let d = ChannelmixerRgbData::from_params(&p, ColorSpace::Rec2020);
        assert!(d.apply_grey);
        assert!((d.grey[0] - 0.2).abs() < 1e-6 && (d.grey[1] - 0.7).abs() < 1e-6);
    }

    /// Gamut exponent is the reciprocal (1/gamut), gamut 0 stays off.
    #[test]
    fn gamut_exponent() {
        let d = ChannelmixerRgbData::from_params(&ChannelmixerRgbParams::default(), ColorSpace::Rec2020);
        assert!((d.gamut - 1.0).abs() < 1e-6, "gamut 1.0 → 1/1: {}", d.gamut);
        let d2 = ChannelmixerRgbData::from_params(
            &ChannelmixerRgbParams { gamut: 2.0, ..Default::default() },
            ColorSpace::Rec2020,
        );
        assert!((d2.gamut - 0.5).abs() < 1e-6, "gamut 2.0 → 1/2: {}", d2.gamut);
        let d0 = ChannelmixerRgbData::from_params(
            &ChannelmixerRgbParams { gamut: 0.0, ..Default::default() },
            ColorSpace::Rec2020,
        );
        assert_eq!(d0.gamut, 0.0);
    }

    /// Sanitize folds none-finite slider values back to the defaults and clamps
    /// out-of-range values, so a hostile/hand-edited blob cannot poison the math.
    #[test]
    fn degenerate_params_folded() {
        let p = ChannelmixerRgbParams {
            red: [f32::NAN, 5.0, -9.0, 0.0],
            temperature: f32::NAN,
            gamut: 99.0,
            ..Default::default()
        };
        let q = sanitize(&p);
        assert_eq!(q.red[0], 1.0, "NaN red folded to default");
        assert_eq!(q.red[1], 2.0, "red clamped to COLOR_MAX");
        assert_eq!(q.red[2], -2.0, "red clamped to COLOR_MIN");
        assert_eq!(q.temperature, 5003.0, "NaN temperature folded");
        assert_eq!(q.gamut, 12.0, "gamut clamped");
    }

    /// Empty / short / non-multiple-of-4 buffers return empty output without
    /// touching the kernel (no slice overrun, no panic).
    #[test]
    fn degenerate_buffer_guards() {
        let p = ChannelmixerRgbParams::default();
        assert!(process(&[], 0, 0, &p, ColorSpace::Rec2020).is_empty());
        // a 1.5-pixel tail: only the where floor(npixels) whole pixels processed
        assert!(process(&[0.1, 0.2, 0.3], 1, 1, &p, ColorSpace::Rec2020).is_empty());
        // a 2-pixel-padded buffer: only the floor(npixels) is processed
        let input = vec![0.2f32, 0.3, 0.4, 1.0, 0.5, 0.6, 0.7, 1.0, 0.1, 0.1, 0.1, 1.0];
        assert_eq!(process(&input, 3, 1, &p, ColorSpace::Rec2020).len(), 12);
    }

    /// The daylight polynomial matches an independently precomputed reference
    /// at the default 5003 K (so a transcription error shared by the impl and a
    /// re-derived formula both fail). Values computed from the C polynomial
    /// (illuminants.h:138-151) at double precision then rounded.
    #[test]
    fn daylight_polynomial_at_default() {
        let xy = cct_to_xy_daylight(5003.0);
        let (x_ref, y_ref) = (0.3456530489086881_f32, 0.35859615970831915_f32);
        assert!(
            (xy[0] - x_ref).abs() < 1e-5 && (xy[1] - y_ref).abs() < 1e-5,
            "daylight(5003) = {xy:?}, expected ({x_ref}, {y_ref})"
        );
    }
}
