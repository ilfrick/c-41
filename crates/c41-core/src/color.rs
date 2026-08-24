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

// ── sRGB (D50) ↔ XYZ ─────────────────────────────────────────────────────────
//
// Matrices from src/common/colorspaces_inline_conversions.h:505-513.
// Applied via dt_apply_transposed_color_matrix convention:
//   out[r] = sum_c(matrix[c][r] * in[c])

const SRGB_TO_XYZ_T: [[f32; 3]; 3] = [
    [0.4360747, 0.2225045, 0.0139322],  // row 0 = R coefficients for XYZ[0..2]
    [0.3850649, 0.7168786, 0.0971045],  // row 1 = G coefficients
    [0.1430804, 0.0606169, 0.7141733],  // row 2 = B coefficients
];

const XYZ_TO_SRGB_T: [[f32; 3]; 3] = [
    [ 3.1338561, -0.9787684,  0.0719453],
    [-1.6168667,  1.9161415, -0.2289914],
    [-0.4906146,  0.0334540,  1.4052427],
];

/// Linear sRGB → XYZ D50 (matches dt_linearRGB_to_XYZ / dt_Rec709_to_XYZ_D50).
pub fn srgb_to_xyz_d50(rgb: [f32; 4]) -> [f32; 4] {
    let xyz: [f32; 3] = std::array::from_fn(|r|
        SRGB_TO_XYZ_T[0][r]*rgb[0] + SRGB_TO_XYZ_T[1][r]*rgb[1] + SRGB_TO_XYZ_T[2][r]*rgb[2]
    );
    [xyz[0], xyz[1], xyz[2], rgb[3]]
}

/// XYZ D50 → linear sRGB (matches dt_XYZ_to_linearRGB / dt_XYZ_to_Rec709_D50).
pub fn xyz_d50_to_srgb(xyz: [f32; 4]) -> [f32; 4] {
    let rgb: [f32; 3] = std::array::from_fn(|r|
        XYZ_TO_SRGB_T[0][r]*xyz[0] + XYZ_TO_SRGB_T[1][r]*xyz[1] + XYZ_TO_SRGB_T[2][r]*xyz[2]
    );
    [rgb[0], rgb[1], rgb[2], xyz[3]]
}

/// Linear **sRGB** → **Lab** (D50): sRGB→XYZ(D50)→Lab. `srgb_to_xyz_d50`'s matrix
/// is Bradford-adapted to D50 **by design** (dt_Rec709_to_XYZ_D50), so — unlike
/// the Rec.2020 path (D65 primaries → explicit CAT16) — no CAT16 step belongs
/// here, and alpha survives (no CAT16 helper to zero ch 3). The linear-sRGB
/// working-space twin of [`rec2020_to_lab`].
pub fn srgb_to_lab(rgb: [f32; 4]) -> [f32; 4] {
    xyz_to_lab(srgb_to_xyz_d50(rgb))
}

/// **Lab** (D50) → linear **sRGB**: the inverse of [`srgb_to_lab`].
pub fn lab_to_srgb(lab: [f32; 4]) -> [f32; 4] {
    xyz_d50_to_srgb(lab_to_xyz(lab))
}

/// XYZ D65 → linear sRGB (matches dt_XYZ_to_Rec709_D65).
pub fn xyz_d65_to_srgb(xyz: [f32; 4]) -> [f32; 4] {
    const M: [[f32; 3]; 3] = [
        [ 3.2404542, -0.9692660,  0.0556434],
        [-1.5371385,  1.8760108, -0.2040259],
        [-0.4985314,  0.0415560,  1.0572252],
    ];
    let rgb: [f32; 3] = std::array::from_fn(|r|
        M[0][r]*xyz[0] + M[1][r]*xyz[1] + M[2][r]*xyz[2]
    );
    [rgb[0], rgb[1], rgb[2], xyz[3]]
}

// ── linear sRGB (D65) → XYZ D65 ──────────────────────────────────────────────
// The exact inverse of the matrix in `xyz_d65_to_srgb` above (same transposed
// M[in][out] storage convention). Completes the sRGB↔XYZ-D65 pair so space-
// aware IOPs can grade in linear sRGB the way they already do in Rec.2020.

const SRGB_TO_XYZ_D65_T: [[f32; 3]; 3] = [
    [0.4124564, 0.2126729, 0.0193339],  // R → X,Y,Z
    [0.3575761, 0.7151522, 0.0721750],  // G → X,Y,Z
    [0.1804375, 0.0721750, 0.9503041],  // B → X,Y,Z
];

/// [`SRGB_TO_XYZ_D65_T`] padded to `[[f32; 4]; 4]` for
/// [`apply_transposed_color_matrix`] — the RGB→XYZ-D65 input matrix for a
/// **linear sRGB** working profile (the non-raw pipeline's space), derived
/// from the 3×3 so the two can't drift.
pub const SRGB_TO_XYZ_D65_T4: [[f32; 4]; 4] = [
    [SRGB_TO_XYZ_D65_T[0][0], SRGB_TO_XYZ_D65_T[0][1], SRGB_TO_XYZ_D65_T[0][2], 0.0],
    [SRGB_TO_XYZ_D65_T[1][0], SRGB_TO_XYZ_D65_T[1][1], SRGB_TO_XYZ_D65_T[1][2], 0.0],
    [SRGB_TO_XYZ_D65_T[2][0], SRGB_TO_XYZ_D65_T[2][1], SRGB_TO_XYZ_D65_T[2][2], 0.0],
    [0.0, 0.0, 0.0, 0.0],
];

/// Linear **sRGB** (D65) → XYZ (D65). Alpha (ch 3) passes through. The
/// linear-sRGB working-space twin of [`rec2020_to_xyz_d65`].
pub fn srgb_to_xyz_d65(rgb: [f32; 4]) -> [f32; 4] {
    let xyz: [f32; 3] = std::array::from_fn(|o|
        SRGB_TO_XYZ_D65_T[0][o]*rgb[0] + SRGB_TO_XYZ_D65_T[1][o]*rgb[1] + SRGB_TO_XYZ_D65_T[2][o]*rgb[2]
    );
    [xyz[0], xyz[1], xyz[2], rgb[3]]
}

// ── Rec.2020 (D65) ↔ XYZ ─────────────────────────────────────────────────────
// BT.2020 primaries, D65 white. Stored transposed (M[in][out]) like the sRGB
// matrices above: out[o] = sum_c M[c][o] * in[c]. The Rec.2020 luma row
// (0.2627, 0.6780, 0.0593) matches `pipeline::REC2020_LUMA`.

const REC2020_TO_XYZ_D65_T: [[f32; 3]; 3] = [
    [0.6369580, 0.2627002, 0.0000000],  // R → X,Y,Z
    [0.1446169, 0.6779981, 0.0280727],  // G → X,Y,Z
    [0.1688810, 0.0593017, 1.0609851],  // B → X,Y,Z
];

const XYZ_D65_TO_REC2020_T: [[f32; 3]; 3] = [
    [ 1.7166512, -0.6666844,  0.0176399],  // X → R,G,B
    [-0.3556708,  1.6164812, -0.0427706],  // Y → R,G,B
    [-0.2533663,  0.0157685,  0.9421031],  // Z → R,G,B
];

/// [`REC2020_TO_XYZ_D65_T`] padded to `[[f32; 4]; 4]` for
/// [`apply_transposed_color_matrix`] — the RGB→XYZ-D65 input matrix the
/// `colorbalancergb` gamut-LUT builders expect (the pipeline's fixed Rec.2020
/// working space is what darktable would reach via `CAT16 · matrix_in`). Derived
/// from the 3×3 so the two can't drift.
pub const REC2020_TO_XYZ_D65_T4: [[f32; 4]; 4] = [
    [REC2020_TO_XYZ_D65_T[0][0], REC2020_TO_XYZ_D65_T[0][1], REC2020_TO_XYZ_D65_T[0][2], 0.0],
    [REC2020_TO_XYZ_D65_T[1][0], REC2020_TO_XYZ_D65_T[1][1], REC2020_TO_XYZ_D65_T[1][2], 0.0],
    [REC2020_TO_XYZ_D65_T[2][0], REC2020_TO_XYZ_D65_T[2][1], REC2020_TO_XYZ_D65_T[2][2], 0.0],
    [0.0, 0.0, 0.0, 0.0],
];

/// Linear **Rec.2020** (D65) → XYZ (D65). Alpha (ch 3) passes through.
pub fn rec2020_to_xyz_d65(rgb: [f32; 4]) -> [f32; 4] {
    let xyz: [f32; 3] = std::array::from_fn(|o|
        REC2020_TO_XYZ_D65_T[0][o]*rgb[0] + REC2020_TO_XYZ_D65_T[1][o]*rgb[1] + REC2020_TO_XYZ_D65_T[2][o]*rgb[2]
    );
    [xyz[0], xyz[1], xyz[2], rgb[3]]
}

/// XYZ (D65) → linear **Rec.2020** (D65). Alpha passes through.
pub fn xyz_d65_to_rec2020(xyz: [f32; 4]) -> [f32; 4] {
    let rgb: [f32; 3] = std::array::from_fn(|o|
        XYZ_D65_TO_REC2020_T[0][o]*xyz[0] + XYZ_D65_TO_REC2020_T[1][o]*xyz[1] + XYZ_D65_TO_REC2020_T[2][o]*xyz[2]
    );
    [rgb[0], rgb[1], rgb[2], xyz[3]]
}

/// Linear **Rec.2020** (D65) → **Lab** (D50): Rec.2020→XYZ(D65)→XYZ(D50, CAT16)→
/// Lab. The composition the pipeline's Rec.2020 working space needs to run Lab-
/// domain IOPs (e.g. the sharpen L channel). Alpha passes through.
pub fn rec2020_to_lab(rgb: [f32; 4]) -> [f32; 4] {
    let xyz65 = rec2020_to_xyz_d65(rgb);
    let mut lab = xyz_to_lab(xyz_d65_to_d50(&xyz65));
    lab[3] = rgb[3]; // the CAT16 helper zeroes ch 3 — carry the original alpha
    lab
}

/// **Lab** (D50) → linear **Rec.2020** (D65): the inverse of [`rec2020_to_lab`].
pub fn lab_to_rec2020(lab: [f32; 4]) -> [f32; 4] {
    let xyz50 = lab_to_xyz(lab);
    let mut rgb = xyz_d65_to_rec2020(xyz_d50_to_d65(&xyz50));
    rgb[3] = lab[3]; // carry alpha (CAT16 zeroes ch 3)
    rgb
}

// ── sRGB transfer function (encoded ↔ linear) ────────────────────────────────
// The matrices above operate on *linear* sRGB; these convert between the
// gamma-encoded sRGB a display/8-bit image stores and linear light. Standard
// IEC 61966-2-1 piecewise curve.

/// Gamma-encoded sRGB component [0,1] → linear light. Inputs ≤ 0 pass through
/// the linear segment (no `powf` on negatives), so it's safe for any value.
#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light → gamma-encoded sRGB component. Values ≤ 0.0031308 (incl.
/// negatives from an over-aggressive edit) take the linear segment, so `powf`
/// only ever sees positive inputs; the caller clamps the result to [0,1].
#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Polar LCh → Cartesian Lab (h is normalized 0..1, same as dt_LCH_2_Lab).
pub fn lch_to_lab(lch: [f32; 4]) -> [f32; 4] {
    let h = lch[2] * std::f32::consts::TAU;
    [lch[0], lch[1] * h.cos(), lch[1] * h.sin(), lch[3]]
}

/// Polar JzCzhz → Cartesian JzAzBz (matches dt_JzCzhz_2_JzAzBz).
pub fn jzczhz_to_jzazbz(jzczhz: [f32; 4]) -> [f32; 4] {
    let h = jzczhz[2] * std::f32::consts::TAU;
    [jzczhz[0], h.cos()*jzczhz[1], h.sin()*jzczhz[1], jzczhz[3]]
}

/// JzAzBz → XYZ D65 via ICtCp/PQ inverse (matches dt_JzAzBz_2_XYZ).
pub fn jzazbz_to_xyz_d65(v: [f32; 4]) -> [f32; 4] {
    const B: f32 = 1.15; const G: f32 = 0.66;
    const C1: f32 = 0.8359375; const C2: f32 = 18.8515625; const C3: f32 = 18.6875;
    const N_INV: f32 = 1.0 / 0.159301758;
    const P_INV: f32 = 1.0 / 134.034375;
    const D: f32 = -0.56; const D0: f32 = 1.6295499532821566e-11;
    const AI: [[f32; 3]; 3] = [
        [1.0,                 1.0,                  1.0               ],
        [0.1386050432715393, -0.1386050432715393,  -0.0960192420263190],
        [0.0580473161561189, -0.0580473161561189,  -0.8118918960560390],
    ];
    const MI: [[f32; 3]; 3] = [
        [ 1.9242264357876067,  0.3503167620949991, -0.0909828109828475],
        [-1.0047923125953657,  0.7264811939316552, -0.3127282905230739],
        [ 0.0376514040306180, -0.0653844229480850,  1.5227665613052603],
    ];
    let mut iz = v[0] + D0;
    iz = (iz / (1.0 + D - D * iz)).max(0.0);
    let iaz = [iz, v[1], v[2]];
    // iaz → LMS via AI (transposed convention: lms[r] = sum_c AI[c][r]*iaz[c])
    let mut lms: [f32; 3] = std::array::from_fn(|r|
        AI[0][r]*iaz[0] + AI[1][r]*iaz[1] + AI[2][r]*iaz[2]
    );
    for l in &mut lms { *l = l.max(0.0).powf(P_INV); }
    for l in &mut lms { *l = (C1 - *l) / (C3 * *l - C2); }
    for l in &mut lms { *l = (10000.0 * l.max(0.0).powf(N_INV)); }
    // lms → X'Y'Z via MI
    let xyz_p: [f32; 3] = std::array::from_fn(|r|
        MI[0][r]*lms[0] + MI[1][r]*lms[1] + MI[2][r]*lms[2]
    );
    let x = (xyz_p[0] + (B-1.0)*xyz_p[2]) / B;
    let y = (xyz_p[1] + (G-1.0)*x) / G;
    [x, y, xyz_p[2], v[3]]
}

/// Normalize an RGB pixel so max(R,G,B) = norm (matches _normalize_color).
pub fn normalize_color(pixel: [f32; 4], norm: f32) -> [f32; 4] {
    let m = pixel[0].max(pixel[1]).max(pixel[2]);
    if m > 0.0 {
        let f = norm / m;
        [pixel[0]*f, pixel[1]*f, pixel[2]*f, pixel[3]]
    } else { pixel }
}

/// Batch linear sRGB → Lab. Matches dt_Rec709_to_XYZ_D50 + dt_XYZ_to_Lab per pixel.
/// Used by ashift.c:1317 and retouch.c:3053.
#[no_mangle]
pub unsafe extern "C" fn darkroom_color_rgb_to_lab(
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    let inp = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in 0..npixels {
        let rgb = [inp[k*4], inp[k*4+1], inp[k*4+2], inp[k*4+3]];
        let xyz = srgb_to_xyz_d50(rgb);
        let lab = xyz_to_lab(xyz);
        out[k*4] = lab[0]; out[k*4+1] = lab[1]; out[k*4+2] = lab[2]; out[k*4+3] = lab[3];
    }
}

/// Batch Lab → linear sRGB in-place. Matches dt_Lab_to_XYZ + dt_XYZ_to_linearRGB per pixel.
/// Used by ashift.c:1339 and retouch.c:3068.
#[no_mangle]
pub unsafe extern "C" fn darkroom_color_lab_to_rgb(
    buf:     *mut f32,
    npixels: usize,
) {
    let b = std::slice::from_raw_parts_mut(buf, npixels * 4);
    for k in 0..npixels {
        let lab = [b[k*4], b[k*4+1], b[k*4+2], b[k*4+3]];
        let xyz = lab_to_xyz(lab);
        let rgb = xyz_d50_to_srgb(xyz);
        b[k*4] = rgb[0]; b[k*4+1] = rgb[1]; b[k*4+2] = rgb[2]; b[k*4+3] = rgb[3];
    }
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
            // Intentional divergence from C dt_rgb_norm(): the C POWER branch
            // divides unconditionally, yielding NaN for an all-zero pixel. We
            // guard it (return the average, i.e. 0). End-to-end output is
            // identical because every caller treats the result via `lum > 0`,
            // which is false for both NaN and 0 → pixel passes through. Do not
            // "restore" the unguarded division in a later match-C pass.
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

// ── Filmlight Yrg / CIE-2006-LMS colour space (src/common/*.h, gamut_mapping.h) ─
// Used by filmicrgb's colour-science v4 chroma path.

/// Filmlight grading-RGB -> CIE 2006 LMS D65 (transposed), `filmlightRGB_D65_to_LMS_D65_trans`.
const FILMLIGHT_RGB_TO_LMS_T: [[f32; 4]; 4] = [
    [0.95, 0.05, 0.00, 0.0],
    [0.38, 0.62, 0.00, 0.0],
    [0.00, 0.03, 0.97, 0.0],
    [0.0, 0.0, 0.0, 0.0],
];

/// CIE 2006 LMS D65 -> Filmlight grading-RGB (transposed), `LMS_D65_to_filmlightRGB_D65_trans`.
const LMS_TO_FILMLIGHT_RGB_T: [[f32; 4]; 4] = [
    [1.08771930, -0.0877193, 0.00, 0.0],
    [-0.66666667, 1.66666667, 0.00, 0.0],
    [0.02061856, -0.05154639, 1.03092784, 0.0],
    [0.0, 0.0, 0.0, 0.0],
];

// Yrg white point (r, g of D50 white through the conversion chain).
const YRG_WP_R: f32 = 0.21902143;
const YRG_WP_G: f32 = 0.54371398;

/// CIE 2006 LMS D65 -> normalised Filmlight Yrg luminance/chromaticity.
/// Matches LMS_to_Yrg().
#[inline(always)]
pub fn lms_to_yrg(lms: [f32; 4]) -> [f32; 4] {
    let y = 0.68990272 * lms[0] + 0.34832189 * lms[1];
    let a = lms[0] + lms[1] + lms[2];
    let norm = if a == 0.0 {
        [0.0; 4]
    } else {
        [lms[0] / a, lms[1] / a, lms[2] / a, 0.0]
    };
    let rgb = apply_transposed_color_matrix(&norm, &LMS_TO_FILMLIGHT_RGB_T);
    [y, rgb[0], rgb[1], 0.0]
}

/// Filmlight Yrg -> CIE 2006 LMS D65. Matches Yrg_to_LMS().
#[inline(always)]
pub fn yrg_to_lms(yrg: [f32; 4]) -> [f32; 4] {
    let y = yrg[0];
    let r = yrg[1];
    let g = yrg[2];
    let b = 1.0 - r - g;
    let lms = apply_transposed_color_matrix(&[r, g, b, 0.0], &FILMLIGHT_RGB_TO_LMS_T);
    let denom = 0.68990272 * lms[0] + 0.34832189 * lms[1];
    let a = if denom == 0.0 { 0.0 } else { y / denom };
    [lms[0] * a, lms[1] * a, lms[2] * a, 0.0]
}

/// Filmlight Yrg -> polar Ych. Stores [Y, c, cos_h, sin_h]. Matches Yrg_to_Ych().
#[inline(always)]
pub fn yrg_to_ych(yrg: [f32; 4]) -> [f32; 4] {
    let y = yrg[0];
    let r = yrg[1] - YRG_WP_R;
    let g = yrg[2] - YRG_WP_G;
    let c = (r * r + g * g).sqrt(); // dt_fast_hypotf (matches its __FAST_MATH__ branch; symmetric in r,g)
    let (cos_h, sin_h) = if c != 0.0 { (r / c, g / c) } else { (1.0, 0.0) };
    [y, c, cos_h, sin_h]
}

/// Polar Ych [Y, c, cos_h, sin_h] -> Filmlight Yrg. Matches Ych_to_Yrg().
#[inline(always)]
pub fn ych_to_yrg(ych: [f32; 4]) -> [f32; 4] {
    let y = ych[0];
    let c = ych[1];
    let r = c * ych[2] + YRG_WP_R;
    let g = c * ych[3] + YRG_WP_G;
    [y, r, g, 0.0]
}

/// Pipeline RGB -> Filmlight Ych. `matrix_trans` is the RGB->LMS-2006 transposed
/// matrix (from prepare_RGB_Yrg_matrices). Matches RGB_to_Ych().
#[inline(always)]
pub fn rgb_to_ych(rgb: [f32; 4], matrix_trans: &[[f32; 4]; 4]) -> [f32; 4] {
    let lms = apply_transposed_color_matrix(&rgb, matrix_trans);
    yrg_to_ych(lms_to_yrg(lms))
}

/// Filmlight Ych -> pipeline RGB. `matrix_trans` is the LMS-2006->RGB transposed
/// matrix. Matches Ych_to_RGB().
#[inline(always)]
pub fn ych_to_rgb(ych: [f32; 4], matrix_trans: &[[f32; 4]; 4]) -> [f32; 4] {
    let lms = yrg_to_lms(ych_to_yrg(ych));
    apply_transposed_color_matrix(&lms, matrix_trans)
}

// ── colorbalancergb helpers (Filmlight grading-RGB, CIE-2006 LMS<->XYZ, UCS HCB, ──
// ── JzAzBz forward, soft_clip). Fixed-matrix conversions from
// colorspaces_inline_conversions.h + darktable_ucs_22_helpers.h. Infra for the
// colorbalancergb process loop (its own conversions are all fixed matrices).

/// CIE 2006 LMS D65 -> CIE 1931 XYZ D65 (transposed), `LMS_2006_D65_to_XYZ_D65_trans`.
const LMS_2006_TO_XYZ_D65_T: [[f32; 4]; 4] = [
    [1.80794659, 0.61783960, -0.12546960, 0.0],
    [-1.29971660, 0.39595453, 0.20478038, 0.0],
    [0.34785879, -0.04104687, 1.74274183, 0.0],
    [0.0, 0.0, 0.0, 0.0],
];

/// CIE 1931 XYZ D65 -> CIE 2006 LMS D65 (transposed), `XYZ_D65_to_LMS_2006_D65_trans`.
const XYZ_D65_TO_LMS_2006_T: [[f32; 4]; 4] = [
    [0.257085, -0.394427, 0.064856, 0.0],
    [0.859943, 1.175800, -0.076250, 0.0],
    [-0.031061, 0.106423, 0.559067, 0.0],
    [0.0, 0.0, 0.0, 0.0],
];

/// CIE 2006 LMS D65 -> CIE 1931 XYZ D65. Matches `LMS_to_XYZ()`.
#[inline(always)]
pub fn lms_2006_to_xyz(lms: [f32; 4]) -> [f32; 4] {
    apply_transposed_color_matrix(&lms, &LMS_2006_TO_XYZ_D65_T)
}

/// CIE 1931 XYZ D65 -> CIE 2006 LMS D65. Matches `XYZ_to_LMS()`.
#[inline(always)]
pub fn xyz_to_lms_2006(xyz: [f32; 4]) -> [f32; 4] {
    apply_transposed_color_matrix(&xyz, &XYZ_D65_TO_LMS_2006_T)
}

/// CIE 2006 LMS D65 -> Filmlight grading-RGB. Matches `LMS_to_gradingRGB()`.
#[inline(always)]
pub fn lms_to_grading_rgb(lms: [f32; 4]) -> [f32; 4] {
    apply_transposed_color_matrix(&lms, &LMS_TO_FILMLIGHT_RGB_T)
}

/// Filmlight grading-RGB -> CIE 2006 LMS D65. Matches `gradingRGB_to_LMS()`.
#[inline(always)]
pub fn grading_rgb_to_lms(rgb: [f32; 4]) -> [f32; 4] {
    apply_transposed_color_matrix(&rgb, &FILMLIGHT_RGB_TO_LMS_T)
}

/// Exponential soft clip above `soft`, hard-limited toward `hard` (must be
/// `> soft`); below `soft` it is the identity. Matches `soft_clip()`.
#[inline(always)]
pub fn soft_clip(x: f32, soft: f32, hard: f32) -> f32 {
    let norm = hard - soft;
    if x > soft {
        soft + (1.0 - (-(x - soft) / norm).exp()) * norm
    } else {
        x
    }
}

/// The dt-UCS JCH<->HCB brightness exponent (`darktable_ucs_22_helpers.h`).
const UCS_HCB_EXP: f32 = 1.33654221029386;

/// dt UCS 22 JCH -> HCB (hue / colourfulness / brightness). Matches
/// `dt_UCS_JCH_to_HCB()`. `HCB = [J·(C^k+1) as brightness in slot 2, C, H]` →
/// stored `[H, C, B, 0]`.
#[inline(always)]
pub fn dt_ucs_jch_to_hcb(jch: [f32; 4]) -> [f32; 4] {
    [jch[2], jch[1], jch[0] * (jch[1].powf(UCS_HCB_EXP) + 1.0), 0.0]
}

/// dt UCS 22 HCB -> JCH. Matches `dt_UCS_HCB_to_JCH()`.
#[inline(always)]
pub fn dt_ucs_hcb_to_jch(hcb: [f32; 4]) -> [f32; 4] {
    [hcb[2] / (hcb[1].powf(UCS_HCB_EXP) + 1.0), hcb[1], hcb[0], 0.0]
}

/// CIE 1931 XYZ D65 -> JzAzBz. Faithful port of `dt_XYZ_2_JzAzBz()` (the reverse
/// [`jzazbz_to_xyz_d65`] already exists). Self-contained: an X'Y'Z pre-adaptation,
/// a fixed matrix, the PQ (SMPTE ST 2084) forward transfer, a fixed matrix, and
/// the Iz->Jz correction.
pub fn xyz_to_jzazbz(xyz_d65: [f32; 4]) -> [f32; 4] {
    const B: f32 = 1.15;
    const G: f32 = 0.66;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    const N: f32 = 0.159301758;
    const P: f32 = 134.034375;
    const D: f32 = -0.56;
    const D0: f32 = 1.6295499532821566e-11;
    // XYZ' -> L'M'S' (transposed), `M_transposed`.
    const M_T: [[f32; 4]; 4] = [
        [0.41478972, -0.2015100, -0.0166008, 0.0],
        [0.57999900, 1.1206490, 0.2648000, 0.0],
        [0.01464800, 0.0531008, 0.6684799, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    // L'M'S' -> Izazbz (transposed), `A_transposed`.
    const A_T: [[f32; 4]; 4] = [
        [0.5, 3.524000, 0.199076, 0.0],
        [0.5, -4.066708, 1.096799, 0.0],
        [0.0, 0.542708, -1.295875, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];

    // XYZ -> X'Y'Z (pre-adaptation)
    let xyz = [
        B * xyz_d65[0] - (B - 1.0) * xyz_d65[2],
        G * xyz_d65[1] - (G - 1.0) * xyz_d65[0],
        xyz_d65[2],
        0.0,
    ];

    // X'Y'Z -> L'M'S' then PQ forward
    let mut lms = apply_transposed_color_matrix(&xyz, &M_T);
    for v in lms.iter_mut().take(3) {
        let t = (*v / 10000.0).max(0.0).powf(N);
        *v = ((C1 + C2 * t) / (1.0 + C3 * t)).powf(P);
    }

    // L'M'S' -> Izazbz, then Iz -> Jz
    let mut jab = apply_transposed_color_matrix(&lms, &A_T);
    jab[0] = (((1.0 + D) * jab[0]) / (1.0 + D * jab[0]) - D0).max(0.0);
    jab
}

/// Number of elements in the gamut LUT (`LUT_ELEM` in `src/common/math.h`). A
/// power of two, so [`lookup_gamut`] can wrap the hue ring with a bitmask.
pub const LUT_ELEM: usize = 512;

/// Build a polar Ych pixel `[Y, c, cos(h), sin(h)]` from luminance `Y`,
/// chroma `c`, and hue `h` (radians). Matches `make_Ych()`.
#[inline(always)]
pub fn make_ych(y: f32, c: f32, h: f32) -> [f32; 4] {
    [y, c, h.cos(), h.sin()]
}

/// Filmlight Ych -> Filmlight grading-RGB (Ych → Yrg → LMS → grading-RGB).
/// Matches `Ych_to_gradingRGB()`.
#[inline(always)]
pub fn ych_to_grading_rgb(ych: [f32; 4]) -> [f32; 4] {
    lms_to_grading_rgb(yrg_to_lms(ych_to_yrg(ych)))
}

/// Sigmoidal shadow/highlight/midtone opacity masks for luma `x`, returning
/// `(output, output_comp)` where `output = [alpha (shadows), gamma (midtones),
/// beta (highlights), 0]` and `output_comp = 1 - output` (componentwise, slot 3
/// left 0). Matches `opacity_masks()` (colorbalancergb.c:551).
pub fn opacity_masks(
    x: f32,
    shadows_weight: f32,
    highlights_weight: f32,
    midtones_weight: f32,
    mask_grey_fulcrum: f32,
) -> ([f32; 4], [f32; 4]) {
    let x_offset = x - mask_grey_fulcrum;
    let x_offset_norm = x_offset / mask_grey_fulcrum;
    let alpha = 1.0 / (1.0 + (x_offset_norm * shadows_weight).exp()); // shadows opacity
    let beta = 1.0 / (1.0 + (-x_offset_norm * highlights_weight).exp()); // highlights opacity
    let alpha_comp = 1.0 - alpha;
    let beta_comp = 1.0 - beta;
    // midtones opacity
    let gamma = (-(x_offset * x_offset) * midtones_weight / 4.0).exp()
        * (alpha_comp * alpha_comp)
        * (beta_comp * beta_comp)
        * 8.0;
    let gamma_comp = 1.0 - gamma;
    ([alpha, gamma, beta, 0.0], [alpha_comp, gamma_comp, beta_comp, 0.0])
}

/// Linearly interpolate the gamut LUT (`LUT_ELEM` entries indexed by hue) at
/// `hue` (radians, `[-π, π)`), wrapping cyclically. Matches `lookup_gamut()`
/// (darktable_ucs_22_helpers.h:132).
#[inline(always)]
pub fn lookup_gamut(gamut_lut: &[f32], hue: f32) -> f32 {
    // hue -> fractional LUT index
    let x_test = LUT_ELEM as f32 * (hue + core::f32::consts::PI) / core::f32::consts::TAU;
    let x_prev = x_test.floor();
    let x_next = x_test.ceil();
    // two closest integer indices, wrapped on the hue ring (LUT_ELEM is 2^k)
    let xi = (x_prev as i32) as usize & (LUT_ELEM - 1);
    let xii = (x_next as i32) as usize & (LUT_ELEM - 1);
    let y_prev = gamut_lut[xi];
    if xi != xii {
        y_prev + (x_test - x_prev) * (gamut_lut[xii] - y_prev)
    } else {
        y_prev
    }
}

// ── filmicrgb v4 gamut mapping (src/common/gamut_mapping.h + gamut_check_Yrg) ──

/// CIE Y 1931 -> CIE Y 2006 scale (achromatic). Matches the
/// `CIE_Y_1931_to_CIE_Y_2006` macro in gamut_mapping.h:31.
pub const CIE_Y_1931_TO_2006: f32 = 1.05785528;

/// Gamut-clip an Ych pixel's chroma at constant hue and luminance so it fits the
/// Yrg / LMS cone space. Returns the sanitised `[Y, c, cos_h, sin_h]`.
/// Matches `gamut_check_Yrg()` in colorspaces_inline_conversions.h:1200.
#[inline(always)]
pub fn gamut_check_yrg(ych: [f32; 4]) -> [f32; 4] {
    let yrg = ych_to_yrg(ych);
    let mut max_c = ych[1];
    let cos_h = ych[2];
    let sin_h = ych[3];
    if yrg[1] < 0.0 {
        max_c = max_c.min(-YRG_WP_R / cos_h);
    }
    if yrg[2] < 0.0 {
        max_c = max_c.min(-YRG_WP_G / sin_h);
    }
    if yrg[1] + yrg[2] > 1.0 {
        max_c = max_c.min((1.0 - YRG_WP_R - YRG_WP_G) / (cos_h + sin_h));
    }
    [ych[0], max_c, cos_h, sin_h]
}

/// Chroma that brings one RGB component to `target_white`, before the near-white
/// numerical correction. `coeffs` is a row of the LMS->RGB matrix. Returns
/// `f32::MAX` when this channel can't limit chroma (matches the C `FLT_MAX`).
/// Matches `_clip_chroma_white_raw()` in gamut_mapping.h:34.
#[inline(always)]
fn clip_chroma_white_raw(coeffs: &[f32; 4], target_white: f32, y: f32, cos_h: f32, sin_h: f32) -> f32 {
    let denominator_y_coeff = coeffs[0] * (0.979381443298969 * cos_h + 0.391752577319588 * sin_h)
        + coeffs[1] * (0.0206185567010309 * cos_h + 0.608247422680412 * sin_h)
        - coeffs[2] * (cos_h + sin_h);
    let denominator_target_term = target_white * (0.68285981628866 * cos_h + 0.482137060515464 * sin_h);

    // this channel won't limit the chroma
    if denominator_y_coeff == 0.0 {
        return f32::MAX;
    }

    // asymptote of the max-chroma equation; below it the upper bound is meaningless
    let y_asymptote = denominator_target_term / denominator_y_coeff;
    if y <= y_asymptote {
        return f32::MAX;
    }

    let denominator = y * denominator_y_coeff - denominator_target_term;
    let numerator = -0.427506877216495
        * (y * (coeffs[0] + 0.856492345150334 * coeffs[1] + 0.554995960637719 * coeffs[2])
            - 0.988237752433297 * target_white);
    numerator / denominator
}

/// Max chroma to keep one channel <= `target_white`, with the near-max-luminance
/// linear feather and negative->FLT_MAX guard. Matches `_clip_chroma_white()`
/// in gamut_mapping.h:64.
#[inline(always)]
fn clip_chroma_white(coeffs: &[f32; 4], target_white: f32, y: f32, cos_h: f32, sin_h: f32) -> f32 {
    let eps = 1e-3;
    let max_y = CIE_Y_1931_TO_2006 * target_white;
    let delta_y = (max_y - y).max(0.0);
    let max_chroma = if delta_y < eps {
        delta_y / (eps * max_y) * clip_chroma_white_raw(coeffs, target_white, (1.0 - eps) * max_y, cos_h, sin_h)
    } else {
        clip_chroma_white_raw(coeffs, target_white, y, cos_h, sin_h)
    };
    if max_chroma >= 0.0 { max_chroma } else { f32::MAX }
}

/// Max chroma to keep one channel >= 0 (target value 0). Matches
/// `_clip_chroma_black()` in gamut_mapping.h:87.
#[inline(always)]
fn clip_chroma_black(coeffs: &[f32; 4], cos_h: f32, sin_h: f32) -> f32 {
    let denominator = coeffs[0] * (0.979381443298969 * cos_h + 0.391752577319588 * sin_h)
        + coeffs[1] * (0.0206185567010309 * cos_h + 0.608247422680412 * sin_h)
        - coeffs[2] * (cos_h + sin_h);

    if denominator == 0.0 {
        return f32::MAX;
    }
    let numerator =
        -0.427506877216495 * (coeffs[0] + 0.856492345150334 * coeffs[1] + 0.554995960637719 * coeffs[2]);
    let max_chroma = numerator / denominator;
    if max_chroma >= 0.0 { max_chroma } else { f32::MAX }
}

/// Min over R/G/B of the chroma that keeps each channel non-negative.
/// `matrix_out` rows are the LMS->RGB coeffs. Matches
/// `Ych_max_chroma_without_negatives()` in gamut_mapping.h:109.
#[inline(always)]
pub fn ych_max_chroma_without_negatives(matrix_out: &[[f32; 4]; 4], cos_h: f32, sin_h: f32) -> f32 {
    let cr = clip_chroma_black(&matrix_out[0], cos_h, sin_h);
    let cg = clip_chroma_black(&matrix_out[1], cos_h, sin_h);
    let cb = clip_chroma_black(&matrix_out[2], cos_h, sin_h);
    cr.min(cg).min(cb)
}

/// Max in-gamut chroma at the given luminance/hue: the tighter of the white
/// (over-max) and black (negative) clipping bounds. Matches `Ych_max_chroma()`
/// in gamut_mapping.h:172.
#[inline(always)]
pub fn ych_max_chroma(matrix_out: &[[f32; 4]; 4], target_white: f32, y: f32, cos_h: f32, sin_h: f32) -> f32 {
    let cr = clip_chroma_white(&matrix_out[0], target_white, y, cos_h, sin_h);
    let cg = clip_chroma_white(&matrix_out[1], target_white, y, cos_h, sin_h);
    let cb = clip_chroma_white(&matrix_out[2], target_white, y, cos_h, sin_h);
    let max_chroma_white = cr.min(cg).min(cb);
    let max_chroma_black = ych_max_chroma_without_negatives(matrix_out, cos_h, sin_h);
    max_chroma_black.min(max_chroma_white)
}

#[cfg(test)]
mod ucs_tests {
    use super::*;

    const L_WHITE: f32 = 2.098883786377; // Y_to_dt_UCS_L_star(1.0)

    const IDENTITY_T: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];

    #[test]
    fn srgb_transfer_known_points_and_roundtrip() {
        // endpoints exact
        assert!((srgb_to_linear(0.0)).abs() < 1e-7);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-5);
        assert!((linear_to_srgb(0.0)).abs() < 1e-7);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-5);
        // sRGB 0.5 encoded ≈ 0.214 linear (well-known)
        assert!((srgb_to_linear(0.5) - 0.214).abs() < 1e-3);
        // round-trips across the range
        for i in 0..=20 {
            let c = i as f32 / 20.0;
            assert!((linear_to_srgb(srgb_to_linear(c)) - c).abs() < 1e-4, "rt {c}");
        }
        // negative / out-of-range stay finite (linear segment, no NaN)
        assert!(linear_to_srgb(-0.1).is_finite());
        assert!(srgb_to_linear(-0.1).is_finite());
        assert!(linear_to_srgb(1.5) > 1.0); // overshoot encodes >1, caller clamps
    }

    #[test]
    fn lms_yrg_round_trips() {
        let lms = [0.5f32, 0.3, 0.2, 0.0];
        let back = yrg_to_lms(lms_to_yrg(lms));
        for c in 0..3 {
            assert!((back[c] - lms[c]).abs() < 1e-4, "c={c}: {} -> {}", lms[c], back[c]);
        }
    }

    #[test]
    fn yrg_ych_round_trips() {
        let yrg = [0.45f32, 0.30, 0.55, 0.0];
        let back = ych_to_yrg(yrg_to_ych(yrg));
        for c in 0..3 {
            assert!((back[c] - yrg[c]).abs() < 1e-5, "c={c}: {} -> {}", yrg[c], back[c]);
        }
    }

    #[test]
    fn rgb_ych_round_trips_identity_matrix() {
        // With identity RGB<->LMS matrices, RGB->Ych->RGB must recover the input.
        let rgb = [0.5f32, 0.3, 0.2, 0.0];
        let ych = rgb_to_ych(rgb, &IDENTITY_T);
        let back = ych_to_rgb(ych, &IDENTITY_T);
        for c in 0..3 {
            assert!((back[c] - rgb[c]).abs() < 1e-4, "c={c}: {} -> {}", rgb[c], back[c]);
        }
    }

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

#[cfg(test)]
mod gamut_tests {
    use super::*;

    // Representative LMS->RGB output matrix (rows = R, G, B coeffs from LMS),
    // and an RGB->LMS transposed matrix. Both are also used by the C reference
    // generator (tools/goldgen) that produced the golden values below.
    const MATRIX_OUT: [[f32; 4]; 4] = [
        [1.80, -1.30, 0.35, 0.0],
        [0.62, 0.40, -0.04, 0.0],
        [-0.13, 0.20, 1.74, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    const MT: [[f32; 4]; 4] = [
        [0.95, 0.38, 0.00, 0.0],
        [0.05, 0.62, 0.03, 0.0],
        [0.00, 0.00, 0.97, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
    }

    // Golden vectors: values printed by a self-contained C program that copies
    // the verbatim functions from colorspaces_inline_conversions.h and
    // gamut_mapping.h (FLT_MAX == 3.40282347e+38).
    const FLT_MAX_C: f32 = 3.40282347e+38;

    #[test]
    fn rgb_to_ych_matches_c_reference() {
        let cases: [([f32; 4], [f32; 4]); 3] = [
            ([0.18, 0.20, 0.15, 0.0], [0.191889524, 0.0852646381, -0.965918958, 0.258844584]),
            ([0.8, 0.1, 0.05, 0.0], [0.655261397, 0.292412996, 0.954872489, -0.297016054]),
            ([0.4, 0.5, 0.9, 0.0], [0.440335333, 0.201490209, -0.660455108, -0.750865519]),
        ];
        for (rgb, expect) in cases {
            let got = rgb_to_ych(rgb, &MT);
            for c in 0..4 {
                assert!(close(got[c], expect[c], 1e-5), "rgb={rgb:?} c={c}: got={} want={}", got[c], expect[c]);
            }
        }
    }

    #[test]
    fn gamut_check_yrg_matches_c_reference() {
        // in0 clips chroma (r+g>1); in1/in2 pass through unchanged.
        let cases: [([f32; 4], [f32; 4]); 3] = [
            ([0.2, 0.6, 0.9, -0.4], [0.2, 0.474529177, 0.9, -0.4]),
            ([0.5, 0.3, -0.7, 0.71], [0.5, 0.3, -0.7, 0.71]),
            ([0.3, 0.05, 0.5, 0.5], [0.3, 0.05, 0.5, 0.5]),
        ];
        for (ych, expect) in cases {
            let got = gamut_check_yrg(ych);
            for c in 0..4 {
                assert!(close(got[c], expect[c], 1e-5), "ych={ych:?} c={c}: got={} want={}", got[c], expect[c]);
            }
        }
    }

    #[test]
    fn clip_chroma_black_matches_c_reference() {
        // cos_h=0.6, sin_h=0.8. R,G hit the FLT_MAX guard; B is the limiter.
        let (cos_h, sin_h) = (0.6, 0.8);
        assert!(close(clip_chroma_black(&MATRIX_OUT[0], cos_h, sin_h), FLT_MAX_C, 1.0));
        assert!(close(clip_chroma_black(&MATRIX_OUT[1], cos_h, sin_h), FLT_MAX_C, 1.0));
        assert!(close(clip_chroma_black(&MATRIX_OUT[2], cos_h, sin_h), 0.175473303, 1e-6));
        assert!(close(ych_max_chroma_without_negatives(&MATRIX_OUT, cos_h, sin_h), 0.175473303, 1e-6));
    }

    #[test]
    fn clip_chroma_white_raw_matches_c_reference() {
        // target_white=1.0, Y=0.5, cos_h=0.6, sin_h=0.8.
        let (cos_h, sin_h) = (0.6, 0.8);
        assert!(close(clip_chroma_white_raw(&MATRIX_OUT[0], 1.0, 0.5, cos_h, sin_h), FLT_MAX_C, 1.0));
        assert!(close(clip_chroma_white_raw(&MATRIX_OUT[1], 1.0, 0.5, cos_h, sin_h), FLT_MAX_C, 1.0));
        // raw can be negative (only _white clamps it); B = -0.102483056.
        assert!(close(clip_chroma_white_raw(&MATRIX_OUT[2], 1.0, 0.5, cos_h, sin_h), -0.102483056, 1e-6));
    }

    #[test]
    fn clip_chroma_white_eps_branch_matches_c_reference() {
        // Y just below max_Y triggers the linear-feather branch.
        let max_y = CIE_Y_1931_TO_2006 * 1.0;
        let got = clip_chroma_white(&MATRIX_OUT[0], 1.0, max_y - 1e-4, 0.6, 0.8);
        assert!(close(got, 3.21725286e37, 3.2e32), "got={got}");
    }

    #[test]
    fn ych_max_chroma_matches_c_reference() {
        let got = ych_max_chroma(&MATRIX_OUT, 1.0, 0.5, 0.6, 0.8);
        assert!(close(got, 0.175473303, 1e-6), "got={got}");
    }
}

#[cfg(test)]
mod rec2020_tests {
    use super::*;

    fn close4(a: [f32; 4], b: [f32; 4], eps: f32) -> bool {
        (0..4).all(|i| (a[i] - b[i]).abs() < eps)
    }

    #[test]
    fn rec2020_xyz_round_trips() {
        // Real f32 round-trip is ~5e-8; 1e-5 stays far below it yet ~600× under the
        // transpose bug this caught (~6e-2), so it's a sensitive asymmetry guard.
        for rgb in [[0.3, 0.5, 0.7, 1.0], [0.9, 0.1, 0.4, 0.5], [1.0, 1.0, 1.0, 1.0]] {
            let back = xyz_d65_to_rec2020(rec2020_to_xyz_d65(rgb));
            assert!(close4(rgb, back, 1e-5), "rgb={rgb:?} back={back:?}");
        }
    }

    #[test]
    fn rec2020_white_is_neutral_lab_l100() {
        // Rec.2020 [1,1,1] is the D65 white; through CAT16→D50→Lab it must land on
        // neutral a≈0, b≈0 (L=100 holds even WITHOUT adaptation — the a/b≈0 is what
        // proves the CAT16 D65→D50 chain; without it b≈-19).
        let lab = rec2020_to_lab([1.0, 1.0, 1.0, 1.0]);
        assert!((lab[0] - 100.0).abs() < 0.05, "L={}", lab[0]);
        assert!(lab[1].abs() < 0.05 && lab[2].abs() < 0.05, "a={} b={}", lab[1], lab[2]);
    }

    #[test]
    fn rec2020_lab_round_trips() {
        for rgb in [[0.2, 0.4, 0.6, 1.0], [0.8, 0.2, 0.5, 1.0], [0.05, 0.9, 0.3, 1.0]] {
            let back = lab_to_rec2020(rec2020_to_lab(rgb));
            assert!(close4(rgb, back, 1e-5), "rgb={rgb:?} back={back:?}");
        }
    }

    #[test]
    fn rec2020_and_srgb_lab_agree_on_neutral_grey() {
        // Cross-check the two independent Lab entry points: a spectrally neutral
        // grey must be neutral (a≈0,b≈0) in both, with equal L (both spaces
        // normalise white to Y=1, so a grey's luminance — hence L — matches). Pins
        // the Rec.2020 and sRGB matrix sets together against future drift.
        for v in [0.1f32, 0.4, 0.8] {
            let lab_rec = rec2020_to_lab([v, v, v, 1.0]);
            let lab_srgb = xyz_to_lab(srgb_to_xyz_d50([v, v, v, 1.0]));
            assert!(lab_rec[1].abs() < 0.1 && lab_rec[2].abs() < 0.1, "rec2020 grey: {lab_rec:?}");
            assert!(lab_srgb[1].abs() < 0.1 && lab_srgb[2].abs() < 0.1, "srgb grey: {lab_srgb:?}");
            assert!((lab_rec[0] - lab_srgb[0]).abs() < 0.2, "L mismatch: {} vs {}", lab_rec[0], lab_srgb[0]);
        }
    }

    #[test]
    fn rec2020_y_row_matches_luma_weights() {
        // The Y (luminance) row of Rec.2020→XYZ must equal the luma weights the
        // pipeline's Sharpen used (0.2627, 0.6780, 0.0593), to ~4 decimals.
        assert!((REC2020_TO_XYZ_D65_T[0][1] - 0.2627).abs() < 1e-3);
        assert!((REC2020_TO_XYZ_D65_T[1][1] - 0.6780).abs() < 1e-3);
        assert!((REC2020_TO_XYZ_D65_T[2][1] - 0.0593).abs() < 1e-3);
    }

    #[test]
    fn srgb_lab_round_trips_and_white_is_neutral() {
        for rgb in [[0.2, 0.4, 0.6, 1.0], [0.8, 0.2, 0.5, 1.0], [0.05, 0.9, 0.3, 1.0]] {
            let back = lab_to_srgb(srgb_to_lab(rgb));
            assert!(close4(rgb, back, 1e-4), "rgb={rgb:?} back={back:?}");
        }
        // sRGB white is already D50-referenced → neutral Lab, no CAT16 involved.
        let w = srgb_to_lab([1.0, 1.0, 1.0, 1.0]);
        assert!((w[0] - 100.0).abs() < 0.05 && w[1].abs() < 0.05 && w[2].abs() < 0.05, "{w:?}");
    }

    // ── colorbalancergb helpers (m4-81) ──

    #[test]
    fn lms_2006_xyz_round_trips() {
        // the two matrices are documented exact inverses.
        for xyz in [[0.3, 0.35, 0.4, 0.0], [0.9, 0.8, 0.7, 0.0], [0.1, 0.05, 0.2, 0.0]] {
            let back = lms_2006_to_xyz(xyz_to_lms_2006(xyz));
            assert!(close4(xyz, back, 1e-4), "xyz={xyz:?} back={back:?}");
        }
    }

    #[test]
    fn grading_rgb_lms_round_trips() {
        for rgb in [[0.3, 0.35, 0.4, 0.0], [0.9, 0.1, 0.5, 0.0]] {
            let back = lms_to_grading_rgb(grading_rgb_to_lms(rgb));
            assert!(close4(rgb, back, 1e-4), "rgb={rgb:?} back={back:?}");
        }
    }

    #[test]
    fn soft_clip_identity_below_and_bounded_above() {
        // below the soft threshold: identity
        assert_eq!(soft_clip(0.3, 0.5, 1.0), 0.3);
        // above: strictly between soft and hard, monotone increasing toward hard
        // (inputs chosen below the f32-saturation-to-hard region).
        let a = soft_clip(0.7, 0.5, 1.0);
        let b = soft_clip(1.0, 0.5, 1.0);
        let c = soft_clip(1.5, 0.5, 1.0);
        assert!(0.5 < a && a < b && b < c && c < 1.0, "a={a} b={b} c={c}");
        // and it does saturate to the hard limit for very large inputs
        assert!((soft_clip(1e6, 0.5, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ucs_jch_hcb_round_trips() {
        for jch in [[0.5, 0.2, 1.3, 0.0], [0.8, 0.0, 0.1, 0.0], [0.2, 0.4, -2.0, 0.0]] {
            let back = dt_ucs_hcb_to_jch(dt_ucs_jch_to_hcb(jch));
            assert!(close4(jch, back, 1e-5), "jch={jch:?} back={back:?}");
        }
    }

    #[test]
    fn xyz_jzazbz_round_trips_against_reverse() {
        // strong faithfulness anchor: forward (new) then the existing reverse must
        // recover the input XYZ for realistic positive tristimulus values.
        for xyz in [[0.3, 0.35, 0.4, 0.0], [0.6, 0.5, 0.45, 0.0], [0.05, 0.06, 0.07, 0.0]] {
            let jab = xyz_to_jzazbz(xyz);
            let back = jzazbz_to_xyz_d65(jab);
            assert!(close4(xyz, back, 2e-3), "xyz={xyz:?} jab={jab:?} back={back:?}");
        }
    }

    // ── colorbalancergb per-pixel helpers (m4-82) ──

    #[test]
    fn make_ych_stores_polar_and_recovers_hue() {
        let ych = make_ych(0.5, 0.3, 0.7);
        assert!((ych[0] - 0.5).abs() < 1e-6 && (ych[1] - 0.3).abs() < 1e-6);
        assert!((ych[2] - 0.7f32.cos()).abs() < 1e-6 && (ych[3] - 0.7f32.sin()).abs() < 1e-6);
        // get_hue_angle_from_Ych == atan2(sin, cos)
        assert!((ych[3].atan2(ych[2]) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn opacity_masks_symmetric_at_fulcrum_and_complement() {
        // at x == mask_grey_fulcrum the offset is 0 → alpha = beta = 0.5, and
        // gamma = e^0 · 0.25 · 0.25 · 8 = 0.5.
        let (out, comp) = opacity_masks(0.5, 3.0, 3.0, 2.0, 0.5);
        assert!((out[0] - 0.5).abs() < 1e-5 && (out[1] - 0.5).abs() < 1e-5 && (out[2] - 0.5).abs() < 1e-5);
        // output_comp is 1 - output on the three active slots
        for c in 0..3 {
            assert!((comp[c] - (1.0 - out[c])).abs() < 1e-6, "comp[{c}] not complement");
        }
        // opacities stay in [0, 1] for a sub/over-fulcrum sample
        let (o2, _) = opacity_masks(0.9, 4.0, 4.0, 2.0, 0.5);
        assert!(o2[0] >= 0.0 && o2[0] <= 1.0 && o2[2] >= 0.0 && o2[2] <= 1.0);
    }

    #[test]
    fn ych_to_grading_rgb_is_hue_independent_when_achromatic() {
        // chroma 0 → Yrg loses the hue, so grading RGB is the same for any hue.
        let a = ych_to_grading_rgb(make_ych(0.6, 0.0, 0.3));
        let b = ych_to_grading_rgb(make_ych(0.6, 0.0, 2.9));
        assert!(close4(a, b, 1e-6), "achromatic Ych depended on hue: {a:?} vs {b:?}");
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lookup_gamut_maps_index_and_interpolates() {
        // a linear LUT: lut[k] = k. Linear interpolation of a linear function is
        // exact, so lookup(hue) must equal the fractional index x_test =
        // LUT_ELEM·(hue+π)/2π — away from the wrap boundary.
        let lut: Vec<f32> = (0..LUT_ELEM).map(|k| k as f32).collect();
        for hue in [-2.0f32, -1.0, 0.0, 1.0, 2.0] {
            let x_test = LUT_ELEM as f32 * (hue + core::f32::consts::PI) / core::f32::consts::TAU;
            let got = lookup_gamut(&lut, hue);
            assert!((got - x_test).abs() < 1e-2, "hue={hue} got={got} want={x_test}");
        }
        // hue = 0 lands exactly on integer index 256 (π/2π = 0.5)
        assert!((lookup_gamut(&lut, 0.0) - 256.0).abs() < 1e-3);
        // a constant LUT returns the constant regardless of hue (incl. wrap)
        let flat = vec![7.0f32; LUT_ELEM];
        assert!((lookup_gamut(&flat, 3.0) - 7.0).abs() < 1e-6);
    }
}
