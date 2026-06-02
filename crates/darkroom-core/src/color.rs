//! Shared color-space utilities used across multiple IOP modules.

// ── HSL ↔ RGB (from src/common/colorspaces.h) ────────────────────────────────

fn hue2rgb(m1: f32, m2: f32, hue: f32) -> f32 {
    if hue < 1.0 {
        m1 + (m2 - m1) * hue
    } else if hue < 3.0 {
        m2
    } else if hue < 4.0 {
        m1 + (m2 - m1) * (4.0 - hue)
    } else {
        m1
    }
}

/// Returns (h, s, l) in [0,1]. Matches C rgb2hsl().
pub fn rgb2hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let pmax = r.max(g).max(b);
    let pmin = r.min(g).min(b);
    let lv = (pmin + pmax) * 0.5;
    let delta = pmax - pmin;
    if delta == 0.0 {
        return (0.0, 0.0, lv);
    }
    const EPS: f32 = 1.52587890625e-05;
    let sv = if lv < 0.5 {
        delta / (pmax + pmin).max(EPS)
    } else {
        delta / (2.0 - pmax - pmin).max(EPS)
    };
    let mut hv = if pmax == r {
        (g - b) / delta
    } else if pmax == g {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };
    hv /= 6.0;
    if hv < 0.0 {
        hv += 1.0;
    } else if hv > 1.0 {
        hv -= 1.0;
    }
    (hv, sv, lv)
}

/// Returns (r, g, b, 0.0). Matches C hsl2rgb() — alpha always 0.
pub fn hsl2rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l, 0.0);
    }
    let m2 = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let m1 = 2.0 * l - m2;
    let h6 = h * 6.0;
    let r = hue2rgb(m1, m2, if h6 < 4.0 { h6 + 2.0 } else { h6 - 4.0 });
    let g = hue2rgb(m1, m2, h6);
    let b = hue2rgb(m1, m2, if h6 > 2.0 { h6 - 2.0 } else { h6 + 4.0 });
    (r, g, b, 0.0)
}

// ── Lab ↔ XYZ (D50 white point) ───────────────────────────────────────────────

const D50: [f32; 3] = [0.9642, 1.0, 0.8249];
const D50_INV: [f32; 3] = [1.0 / 0.9642, 1.0, 1.0 / 0.8249];
const LAB_EPSILON: f32 = 216.0 / 24389.0;
const LAB_KAPPA: f32 = 24389.0 / 27.0;
// cbrt(216/24389) — threshold for lab_f_inv
const LAB_CBRT_EPSILON: f32 = 0.20689655172413796;

fn lab_f_inv(x: f32) -> f32 {
    if x > LAB_CBRT_EPSILON {
        x * x * x
    } else {
        (116.0 * x - 16.0) / LAB_KAPPA
    }
}

/// Matches C dt_Lab_to_XYZ().
pub fn lab_to_xyz(lab: [f32; 4]) -> [f32; 4] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;
    [
        D50[0] * lab_f_inv(fx),
        D50[1] * lab_f_inv(fy),
        D50[2] * lab_f_inv(fz),
        lab[3],
    ]
}

/// Matches C dt_XYZ_to_Lab().
pub fn xyz_to_lab(xyz: [f32; 4]) -> [f32; 4] {
    let f: [f32; 3] = std::array::from_fn(|i| {
        let x = xyz[i] * D50_INV[i];
        if x > LAB_EPSILON {
            x.cbrt()
        } else {
            (LAB_KAPPA * x + 16.0) / 116.0
        }
    });
    [
        116.0 * f[1] - 16.0,
        500.0 * (f[0] - f[1]),
        -200.0 * (f[2] - f[1]),
        xyz[3],
    ]
}

// ── Lab ↔ ProPhoto RGB ────────────────────────────────────────────────────────

// Transposed matrices from colorspaces_inline_conversions.h:439-462.
// applied as: out[i] = sum_j M_T[j][i] * in[j]
//
// xyz_to_prophotorgb_transpose row j, col i:
//   j=0: [1.3459433, -0.5445989, 0.0,       0.0]
//   j=1: [-0.2556075, 1.5081673, 0.0,       0.0]
//   j=2: [-0.0511118, 0.0205351, 1.2118128, 0.0]
//
// rgb[0] = 1.3459433*X - 0.2556075*Y - 0.0511118*Z
// rgb[1] = -0.5445989*X + 1.5081673*Y + 0.0205351*Z
// rgb[2] = 0.0*X + 0.0*Y + 1.2118128*Z

pub fn xyz_to_prophotorgb(xyz: [f32; 4]) -> [f32; 4] {
    [
        1.3459433 * xyz[0] - 0.2556075 * xyz[1] - 0.0511118 * xyz[2],
        -0.5445989 * xyz[0] + 1.5081673 * xyz[1] + 0.0205351 * xyz[2],
        1.2118128 * xyz[2],
        xyz[3],
    ]
}

// prophotorgb_to_xyz_transpose:
//   j=0: [0.7976749, 0.2880402, 0.0, 0.0]
//   j=1: [0.1351917, 0.7118741, 0.0, 0.0]
//   j=2: [0.0313534, 0.0000857, 0.8252100, 0.0]
//
// XYZ[0] = 0.7976749*r + 0.1351917*g + 0.0313534*b
// XYZ[1] = 0.2880402*r + 0.7118741*g + 0.0000857*b
// XYZ[2] = 0.0*r + 0.0*g + 0.8252100*b

pub fn prophotorgb_to_xyz(rgb: [f32; 4]) -> [f32; 4] {
    [
        0.7976749 * rgb[0] + 0.1351917 * rgb[1] + 0.0313534 * rgb[2],
        0.2880402 * rgb[0] + 0.7118741 * rgb[1] + 0.0000857 * rgb[2],
        0.8252100 * rgb[2],
        rgb[3],
    ]
}

pub fn lab_to_prophotorgb(lab: [f32; 4]) -> [f32; 4] {
    xyz_to_prophotorgb(lab_to_xyz(lab))
}

pub fn prophotorgb_to_lab(rgb: [f32; 4]) -> [f32; 4] {
    xyz_to_lab(prophotorgb_to_xyz(rgb))
}

// ── rgb_norm (from src/common/rgb_norms.h) ────────────────────────────────────

/// ProPhoto RGB luminance = Y from prophotorgb_to_XYZ.
pub fn prophoto_luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2880402 * r + 0.7118741 * g + 0.0000857 * b
}

/// Matches dt_rgb_norm() using hardcoded ProPhoto profile (tonecurve always uses ProPhoto).
pub fn rgb_norm(r: f32, g: f32, b: f32, mode: i32) -> f32 {
    match mode {
        1 => prophoto_luminance(r, g, b),
        2 => r.max(g).max(b),
        3 => (r + g + b) / 3.0,
        4 => r + g + b,
        5 => (r * r + g * g + b * b).sqrt(),
        6 => {
            let r2 = r * r;
            let g2 = g * g;
            let b2 = b * b;
            let den = r2 + g2 + b2;
            if den > 0.0 {
                (r * r2 + g * g2 + b * b2) / den
            } else {
                (r + g + b) / 3.0
            }
        }
        _ => (r + g + b) / 3.0,
    }
}

// ── dt UCS 2.2 color space ────────────────────────────────────────────────────
// Mirrors src/common/colorspaces_inline_conversions.h starting at line 1270.
// This color model is used by colorequal, colorbalancergb, colorharmonizer.

const DT_UCS_L_STAR_RANGE:       f32 = 2.098883786377;
const DT_UCS_L_STAR_UPPER_LIMIT: f32 = 2.09885;

/// D65 white point chromaticity fallback (from colorspaces.h D65xyY).
pub const D65_X: f32 = 0.31271;
pub const D65_Y: f32 = 0.32902;

/// D50 white point chromaticity fallback (from colorspaces.h D50xyY).
pub const D50_X: f32 = 0.34567;
pub const D50_Y: f32 = 0.35850;

/// `Y_to_dt_UCS_L_star(Y)` — convert absolute luminance to dt UCS lightness.
/// Matches Y_to_dt_UCS_L_star() in colorspaces_inline_conversions.h:1274.
#[inline(always)]
pub fn y_to_dt_ucs_l_star(y: f32) -> f32 {
    let y_hat = y.powf(0.631651345306265);
    DT_UCS_L_STAR_RANGE * y_hat / (y_hat + 1.12426773749357)
}

/// Inverse of `y_to_dt_ucs_l_star`.
/// Matches dt_UCS_L_star_to_Y() in colorspaces_inline_conversions.h:1280.
#[inline(always)]
pub fn dt_ucs_l_star_to_y(l_star: f32) -> f32 {
    ((1.12426773749357 * l_star / (DT_UCS_L_STAR_RANGE - l_star)).max(0.0)).powf(1.5831518565279648)
}

/// Convert xyY to dt UCS UV_star_prime[2].
/// Matches xyY_to_dt_UCS_UV() in colorspaces_inline_conversions.h:1286.
#[inline(always)]
pub fn xyy_to_dt_ucs_uv(xyy: &[f32; 4]) -> [f32; 2] {
    const X_FACTORS: [f32; 3] = [-0.783941002840055,  0.745273540913283,  0.318707282433486];
    const Y_FACTORS: [f32; 3] = [ 0.277512987809202, -0.205375866083878,  2.16743692732158];
    const OFFSETS:   [f32; 3] = [ 0.153836578598858, -0.165478376301988,  0.291320554395942];

    let mut uvd = [0.0_f32; 3];
    for c in 0..3 {
        uvd[c] = X_FACTORS[c] * xyy[0] + Y_FACTORS[c] * xyy[1] + OFFSETS[c];
    }

    let div = if uvd[2] >= 0.0 { uvd[2].max(f32::MIN_POSITIVE) }
              else { uvd[2].min(-f32::MIN_POSITIVE) };
    uvd[0] /= div;
    uvd[1] /= div;

    let factors     = [1.39656225667, 1.4513954287];
    let half_values = [1.49217352929, 1.52488637914];
    let mut uv_star = [0.0_f32; 2];
    for c in 0..2 {
        uv_star[c] = factors[c] * uvd[c] / (uvd[c].abs() + half_values[c]);
    }

    // 2D matrix product
    [
        -1.124983854323892 * uv_star[0] - 0.980483721769325 * uv_star[1],
         1.86323315098672  * uv_star[0] + 1.971853092390862 * uv_star[1],
    ]
}

/// Convert (L_star, L_white, UV_star_prime) to JCH.
/// Matches dt_UCS_LUV_to_JCH() in colorspaces_inline_conversions.h:1314.
#[inline(always)]
pub fn dt_ucs_luv_to_jch(l_star: f32, l_white: f32, uv: &[f32; 2]) -> [f32; 4] {
    let m2 = uv[0] * uv[0] + uv[1] * uv[1];
    let j  = l_star / l_white;
    let c  = 15.932993652962535 * l_star.powf(0.6523997524738018)
           * m2.powf(0.6007557017508491) / l_white;
    let h  = uv[1].atan2(uv[0]);
    [j, c, h, 0.0]
}

/// Convert xyY (normalized D65 CIE 2°) to dt UCS JCH.
/// Matches xyY_to_dt_UCS_JCH() in colorspaces_inline_conversions.h:1325.
#[inline(always)]
pub fn xyy_to_dt_ucs_jch(xyy: &[f32; 4], l_white: f32) -> [f32; 4] {
    let uv = xyy_to_dt_ucs_uv(xyy);
    dt_ucs_luv_to_jch(y_to_dt_ucs_l_star(xyy[2]), l_white, &uv)
}

/// Convert dt UCS JCH back to xyY.
/// Matches dt_UCS_JCH_to_xyY() in colorspaces_inline_conversions.h:1343.
#[inline(always)]
pub fn dt_ucs_jch_to_xyy(jch: &[f32; 4], l_white: f32) -> [f32; 4] {
    let l_star = (jch[0] * l_white).clamp(0.0, DT_UCS_L_STAR_UPPER_LIMIT);
    let m = if l_star != 0.0 {
        (jch[1] * l_white / (15.932993652962535 * l_star.powf(0.6523997524738018)))
            .max(0.0)
            .powf(0.8322850678616855)
    } else {
        0.0
    };
    let u_star_prime = m * jch[2].cos();
    let v_star_prime = m * jch[2].sin();

    // inverse 2D matrix
    let uv_star = [
        -5.037522385190711 * u_star_prime - 2.504856328185843 * v_star_prime,
         4.760029407436461 * u_star_prime + 2.874012963239247 * v_star_prime,
    ];

    let factors     = [1.39656225667, 1.4513954287];
    let half_values = [1.49217352929, 1.52488637914];
    let mut uv = [0.0_f32; 2];
    for c in 0..2 {
        uv[c] = -half_values[c] * uv_star[c] / (uv_star[c].abs() - factors[c]);
    }

    const U_FACTORS: [f32; 3] = [ 0.167171472114775, -0.150959086409163,  0.940254742367256];
    const V_FACTORS: [f32; 3] = [ 0.141299802443708, -0.155185060382272,  1.000000000000000];
    const OFFSETS2:  [f32; 3] = [-0.00801531300850582, -0.00843312433578007, -0.0256325967652889];

    let mut xyd = [0.0_f32; 3];
    for c in 0..3 {
        xyd[c] = U_FACTORS[c] * uv[0] + V_FACTORS[c] * uv[1] + OFFSETS2[c];
    }
    let div = if xyd[2] >= 0.0 { xyd[2].max(f32::MIN_POSITIVE) }
              else { xyd[2].min(-f32::MIN_POSITIVE) };

    [xyd[0] / div, xyd[1] / div, dt_ucs_l_star_to_y(l_star), 0.0]
}

/// JCH → HSB (Hue/Saturation/Brightness).
/// Matches dt_UCS_JCH_to_HSB() in colorspaces_inline_conversions.h:1389.
#[inline(always)]
pub fn dt_ucs_jch_to_hsb(jch: &[f32; 4]) -> [f32; 4] {
    let b = jch[0] * (jch[1].powf(1.33654221029386) + 1.0);
    let s = if b > 0.0 { jch[1] / b } else { 0.0 };
    [jch[2], s, b, jch[3]]
}

/// HSB → JCH.
/// Matches dt_UCS_HSB_to_JCH() in colorspaces_inline_conversions.h:1397.
#[inline(always)]
pub fn dt_ucs_hsb_to_jch(hsb: &[f32; 4]) -> [f32; 4] {
    let c = hsb[1] * hsb[2];
    let j = hsb[2] / (c.powf(1.33654221029386) + 1.0);
    [j, c, hsb[0], hsb[3]]
}

/// XYZ D65 → xyY with D65 white-point fallback on zero-sum.
/// Matches dt_D65_XYZ_to_xyY() in colorspaces_inline_conversions.h:246.
#[inline(always)]
pub fn d65_xyz_to_xyy(xyz: &[f32; 4]) -> [f32; 4] {
    let x = xyz[0].max(0.0);
    let y = xyz[1].max(0.0);
    let z = xyz[2].max(0.0);
    let sum = x + y + z;
    if sum > 0.0 {
        [x / sum, y / sum, y, 0.0]
    } else {
        [D65_X, D65_Y, 0.0, 0.0]
    }
}

/// xyY → XYZ (Bruce Lindbloom formula).
/// Matches dt_xyY_to_XYZ() in colorspaces_inline_conversions.h:272.
#[inline(always)]
pub fn xyy_to_xyz(xyy: &[f32; 4]) -> [f32; 4] {
    if xyy[1] == 0.0 {
        [0.0, 0.0, 0.0, 0.0]
    } else {
        let big_y = xyy[2];
        [
            big_y * xyy[0] / xyy[1],
            big_y,
            big_y * (1.0 - xyy[0] - xyy[1]) / xyy[1],
            0.0,
        ]
    }
}

// ── Matrix operations + chromatic adaptation ─────────────────────────────────

/// Standard (non-transposed) row-major 3×4 matrix product.
///
/// `out[i] = M[i] · in`  (i in 0..3)
///
/// Matches `dot_product()` in src/common/math.h:279.
#[inline(always)]
pub fn dot_product(inp: &[f32; 4], m: &[[f32; 4]; 4]) -> [f32; 4] {
    let mut out = [0.0_f32; 4];
    for i in 0..4 {
        out[i] = m[i][0]*inp[0] + m[i][1]*inp[1] + m[i][2]*inp[2] + m[i][3]*inp[3];
    }
    out
}

/// Apply a transposed 4×4 colour matrix (stored row-major with padding).
///
/// out[r] = M[0][r]*in[0] + M[1][r]*in[1] + M[2][r]*in[2]
///
/// `matrix` is `[[f32; 4]; 4]` (C `dt_colormatrix_t`), only the first 3
/// rows and first 3 columns are used. Matches `dt_apply_transposed_color_matrix()`
/// in colorspaces_inline_conversions.h:121.
#[inline(always)]
pub fn apply_transposed_color_matrix(inp: &[f32; 4], m: &[[f32; 4]; 4]) -> [f32; 4] {
    let mut out = [0.0_f32; 4];
    for r in 0..4 {
        out[r] = m[0][r] * inp[0] + m[1][r] * inp[1] + m[2][r] * inp[2];
    }
    out
}

// CAT16 D50↔D65 transposed matrices (from chromatic_adaptation.h).
pub const XYZ_D50_TO_D65_CAT16_TRANS: [[f32; 4]; 4] = [
    [ 9.89466254e-01, -5.40518733e-03, -4.03920992e-04, 0.0],
    [-4.00304626e-02,  1.00666069e+00,  1.50768030e-02, 0.0],
    [ 4.40530317e-02, -1.75551955e-03,  1.30210211e+00, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub const XYZ_D65_TO_D50_CAT16_TRANS: [[f32; 4]; 4] = [
    [ 1.01085433e+00,  5.42814201e-03,  2.50722468e-04, 0.0],
    [ 4.07086103e-02,  9.93581926e-01, -1.14918759e-02, 0.0],
    [-3.41445825e-02,  1.15592039e-03,  7.67964947e-01, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// XYZ D50 → XYZ D65 (CAT16). Matches XYZ_D50_to_D65() in chromatic_adaptation.h:406.
#[inline(always)]
pub fn xyz_d50_to_d65(xyz: &[f32; 4]) -> [f32; 4] {
    apply_transposed_color_matrix(xyz, &XYZ_D50_TO_D65_CAT16_TRANS)
}

/// XYZ D65 → XYZ D50 (CAT16). Matches XYZ_D65_to_D50() in chromatic_adaptation.h:412.
#[inline(always)]
pub fn xyz_d65_to_d50(xyz: &[f32; 4]) -> [f32; 4] {
    apply_transposed_color_matrix(xyz, &XYZ_D65_TO_D50_CAT16_TRANS)
}

/// Convert pipeline RGB to dt UCS JCH using a pre-transposed working-space→XYZ
/// D50 matrix.  Matches `dt_ioppr_rgb_matrix_to_dt_UCS_JCH()` in iop_profile.h:410.
///
/// The matrix is `work_profile->matrix_in_transposed` (a 4×4 array of [f32;4]).
/// `l_white` = `Y_to_dt_UCS_L_star(1.0f)`.
#[inline(always)]
pub fn rgb_to_dt_ucs_jch(rgb: &[f32; 4], matrix_in_transposed: &[[f32; 4]; 4], l_white: f32) -> [f32; 4] {
    let xyz_d50  = apply_transposed_color_matrix(rgb, matrix_in_transposed);
    let xyz_d65  = xyz_d50_to_d65(&xyz_d50);
    let xyy      = d65_xyz_to_xyy(&xyz_d65);
    xyy_to_dt_ucs_jch(&xyy, l_white)
}

/// Convert dt UCS JCH back to pipeline RGB.
/// Inverse of `rgb_to_dt_ucs_jch`: JCH → xyY → XYZ D65 → XYZ D50 → RGB.
#[inline(always)]
pub fn dt_ucs_jch_to_rgb(jch: &[f32; 4], matrix_out_transposed: &[[f32; 4]; 4], l_white: f32) -> [f32; 4] {
    let xyy     = dt_ucs_jch_to_xyy(jch, l_white);
    let xyz_d65 = xyy_to_xyz(&xyy);
    let xyz_d50 = xyz_d65_to_d50(&xyz_d65);
    apply_transposed_color_matrix(&xyz_d50, matrix_out_transposed)
}

// ── eval_exp (unbounded LUT extrapolation) ────────────────────────────────────

/// coeff[1] * (x * coeff[0])^coeff[2] — darktable's eval_exp for LUT tails.
pub fn eval_exp(coeff: &[f32], x: f32) -> f32 {
    coeff[1] * (x * coeff[0]).powf(coeff[2])
}

// ── ICC profile primitives (mirrors src/common/iop_profile.h inline helpers) ─

/// Linearly interpolate a per-channel LUT.
///
/// Matches `extrapolate_lut()` in src/common/iop_profile.h: clamps the input
/// position to [0, lutsize-1], picks the floor index (capped at lutsize-2 so
/// `t+1` is in bounds), and interpolates between the two nearest LUT entries.
#[inline(always)]
pub fn extrapolate_lut(lut: &[f32], v: f32, lutsize: usize) -> f32 {
    let ft = (v * (lutsize - 1) as f32).clamp(0.0, (lutsize - 1) as f32);
    let t = if (ft as usize) < lutsize - 2 { ft as usize } else { lutsize - 2 };
    let f = ft - t as f32;
    lut[t] * (1.0 - f) + lut[t + 1] * f
}

/// Apply the per-channel tone response curve to the three RGB components.
///
/// Matches `dt_ioppr_apply_trc()`. For each channel:
/// * if `lut[c][0] < 0` the LUT is disabled (no-op);
/// * else if `rgb_in[c] < 1.0`, look it up via `extrapolate_lut`;
/// * else extrapolate with `eval_exp(unbounded_coeffs[c], rgb_in[c])`.
///
/// `luts[c]` is a slice of length `lutsize`; `unbounded_coeffs[c]` is a 3-float
/// slice as the C side stores per-channel `eval_exp` parameters.
#[inline(always)]
pub fn apply_trc(
    rgb_in: [f32; 4],
    luts: [&[f32]; 3],
    unbounded_coeffs: [&[f32]; 3],
    lutsize: usize,
) -> [f32; 4] {
    let mut out = rgb_in;
    for c in 0..3 {
        out[c] = if luts[c][0] >= 0.0 {
            if rgb_in[c] < 1.0 {
                extrapolate_lut(luts[c], rgb_in[c], lutsize)
            } else {
                eval_exp(unbounded_coeffs[c], rgb_in[c])
            }
        } else {
            rgb_in[c]
        };
    }
    out
}

/// Compute the relative luminance Y of an RGB pixel under a working-space ICC
/// profile.
///
/// Matches `dt_ioppr_get_rgb_matrix_luminance()`:
/// * if `nonlinear_lut` is true, first linearise the pixel via `apply_trc`;
/// * then return the row-1 dot product with the input matrix (the Y row of
///   the 3x4 colour matrix laid out as a 4x4 padded array).
///
/// `matrix_in` is the 4x4 colour-matrix-to-XYZ array (we only read row 1).
#[inline(always)]
pub fn get_rgb_matrix_luminance(
    rgb: [f32; 4],
    matrix_in: &[[f32; 4]; 4],
    luts: [&[f32]; 3],
    unbounded_coeffs: [&[f32]; 3],
    lutsize: usize,
    nonlinear_lut: bool,
) -> f32 {
    let r = if nonlinear_lut {
        apply_trc(rgb, luts, unbounded_coeffs, lutsize)
    } else {
        rgb
    };
    matrix_in[1][0] * r[0] + matrix_in[1][1] * r[1] + matrix_in[1][2] * r[2]
}

#[cfg(test)]
mod ucs_tests {
    use super::*;

    const L_WHITE: f32 = 2.098883786377; // Y_to_dt_UCS_L_star(1.0)

    #[test]
    fn l_star_round_trips() {
        for y in [0.01, 0.1, 0.5, 1.0, 2.0_f32] {
            let l = y_to_dt_ucs_l_star(y);
            let y2 = dt_ucs_l_star_to_y(l);
            assert!((y2 - y).abs() < 1e-5, "y={y} → l={l} → y2={y2}");
        }
    }

    #[test]
    fn xyy_ucs_jch_round_trips() {
        // Mid-grey D65 (x=0.31, y=0.33, Y=0.5)
        let xyy: [f32; 4] = [0.31271, 0.32902, 0.5, 0.0];
        let jch = xyy_to_dt_ucs_jch(&xyy, L_WHITE);
        let xyy2 = dt_ucs_jch_to_xyy(&jch, L_WHITE);
        for c in 0..3 {
            assert!((xyy2[c] - xyy[c]).abs() < 1e-4, "c={c}: {}->{}", xyy[c], xyy2[c]);
        }
    }

    #[test]
    fn jch_hsb_round_trips() {
        let jch: [f32; 4] = [0.5, 0.3, 1.0, 0.0];
        let hsb = dt_ucs_jch_to_hsb(&jch);
        let jch2 = dt_ucs_hsb_to_jch(&hsb);
        for c in 0..3 {
            assert!((jch2[c] - jch[c]).abs() < 1e-5, "c={c}: {}->{}", jch[c], jch2[c]);
        }
    }

    #[test]
    fn apply_transposed_matrix_identity() {
        // Identity 4×4 matrix → output = input
        let m: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inp = [0.3, 0.5, 0.7, 0.0];
        let out = apply_transposed_color_matrix(&inp, &m);
        for c in 0..3 { assert!((out[c] - inp[c]).abs() < 1e-6); }
    }

    #[test]
    fn d50_d65_round_trip() {
        let xyz = [0.2, 0.4, 0.3, 0.0_f32];
        let d65 = xyz_d50_to_d65(&xyz);
        let d50 = xyz_d65_to_d50(&d65);
        for c in 0..3 { assert!((d50[c] - xyz[c]).abs() < 1e-5, "c={c}"); }
    }

    #[test]
    fn rgb_ucs_jch_pipeline_produces_finite_values() {
        // Use sRGB D65 matrix (identity-ish) — just verify no NaN/inf
        let m: [[f32; 4]; 4] = [
            [0.4124564, 0.2126729, 0.0193339, 0.0],
            [0.3575761, 0.7151522, 0.1191920, 0.0],
            [0.1804375, 0.0721750, 0.9503041, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let rgb = [0.5_f32, 0.3, 0.8, 0.0];
        let jch = rgb_to_dt_ucs_jch(&rgb, &m, L_WHITE);
        for c in 0..3 { assert!(jch[c].is_finite(), "c={c}: {}", jch[c]); }
    }

    #[test]
    fn xyy_to_xyz_and_back() {
        let xyy: [f32; 4] = [0.3, 0.4, 0.5, 0.0];
        let xyz = xyy_to_xyz(&xyy);
        let sum = xyz[0] + xyz[1] + xyz[2];
        assert!((xyz[0] / sum - 0.3).abs() < 1e-5);
        assert!((xyz[1] / sum - 0.4).abs() < 1e-5);
        assert!((xyz[1] - 0.5).abs() < 1e-5);
    }
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn extrapolate_lut_identity() {
        // LUT[i] = i/(N-1) maps v → v
        let n = 1024;
        let lut: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        for &v in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let r = extrapolate_lut(&lut, v, n);
            assert!((r - v).abs() < 1e-4, "v={v} got={r}");
        }
    }

    #[test]
    fn extrapolate_lut_clamps_above_one() {
        let lut: Vec<f32> = vec![0.5; 16];
        assert_eq!(extrapolate_lut(&lut, 5.0, 16), 0.5); // saturates at last entry
    }

    #[test]
    fn apply_trc_disabled_lut_is_passthrough() {
        let lut = vec![-1.0_f32; 4]; // negative sentinel → no-op
        let coeffs = [1.0_f32, 1.0, 1.0];
        let rgb = [0.3, 0.5, 0.7, 1.0];
        let out = apply_trc(rgb,
            [&lut[..], &lut[..], &lut[..]],
            [&coeffs[..], &coeffs[..], &coeffs[..]],
            lut.len());
        assert_eq!(out, rgb);
    }

    #[test]
    fn luminance_linear_path_takes_y_row() {
        // matrix[1] = [0.25, 0.5, 0.25, 0]; rgb = [1,1,1] → 1.0
        let m: [[f32; 4]; 4] = [
            [0.0; 4],
            [0.25, 0.5, 0.25, 0.0],
            [0.0; 4],
            [0.0; 4],
        ];
        let lut = vec![0.0_f32; 4];
        let coeffs = [0.0_f32; 3];
        let y = get_rgb_matrix_luminance(
            [1.0, 1.0, 1.0, 0.0], &m,
            [&lut[..], &lut[..], &lut[..]],
            [&coeffs[..], &coeffs[..], &coeffs[..]],
            lut.len(), false,
        );
        assert!((y - 1.0).abs() < 1e-6);
    }
}
