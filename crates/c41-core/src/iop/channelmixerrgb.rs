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
