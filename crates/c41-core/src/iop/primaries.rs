use crate::{params::IopParams, pipeline::ColorSpace, roi::RoiIn, Result};
use super::IopProcess;

/// Working-space primaries and whitepoint for the colour spaces the preview
/// pipeline uses. In darktable these come from the ICC profile (`profile->*`),
/// but C41's pipeline only ever works in linear Rec.2020 (raws) or linear sRGB
/// (non-raws), so the standard xy chromaticities are hardcoded here.
///
/// Each entry bundles the D65 white point and the working RGB primaries so that
/// [`compute_matrix`] can derive the working XYZ→RGB matrix by inverting the
/// RGB→XYZ matrix that [`make_transposed_matrices_from_primaries_and_whitepoint`]
/// builds from those same primaries.

// --- Working-space constants ------------------------------------------------
// sRGB primaries (IEC 61966-2-1 / Rec. 709), D65 white.
const SRGB_PRIMARIES: [[f32; 2]; 3] = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]];
// Rec. 2020 primaries (ITU-R BT.2020), D65 white.
const REC2020_PRIMARIES: [[f32; 2]; 3] = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]];
// D65 white point in xy (CIE 1931 2°).
const D65_WHITEPOINT: [f32; 2] = [0.3127, 0.3290];

pub struct Primaries;

impl IopProcess for Primaries {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "primaries" }
}

/// Apply a transposed 3×3 color matrix: `out[r] = Σ matrix[c][r] * in[c]` for
/// c in 0..3. Only the first 3 rows/cols are read (the 4th row/col is for alpha,
/// handled separately by the caller).
fn apply_transposed_color_matrix_3(inp: [f32; 3], m: &[[f32; 3]; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for r in 0..3 {
        out[r] = m[0][r] * inp[0] + m[1][r] * inp[1] + m[2][r] * inp[2];
    }
    out
}

/// Clamp |y| away from zero (mirrors `_sanitizeY` in custom_primaries.c).
fn sanitize_y(y: f32) -> f32 {
    let eps = f32::EPSILON;
    if y >= 0.0 && y < eps {
        eps
    } else if y < 0.0 && y > -eps {
        -eps
    } else {
        y
    }
}

/// 2×2 determinant: | a b | = a*d - b*c.
fn determinant(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a * d - b * c
}

/// Line-segment intersection parameter `t` for the ray (x1,y1)→(x2,y2) hitting
/// the segment (x3,y3)→(x4,y4). Returns `t ≥ 0` (the distance along the ray) or
/// `f32::MAX` if there is no valid intersection (parallel or behind the origin).
///
/// Mirrors `_intersect_line_segments` (static inline in custom_primaries.c).
fn intersect_line_segments(
    x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32, x4: f32, y4: f32,
) -> f32 {
    let denominator = determinant(x1 - x2, x3 - x4, y1 - y2, y3 - y4);
    if denominator == 0.0 {
        return f32::MAX;
    }
    let t = determinant(x1 - x3, x3 - x4, y1 - y3, y3 - y4) / denominator;
    if t >= 0.0 {
        t
    } else {
        f32::MAX
    }
}

/// Find the distance from the white point to the gamut edge along a ray
/// defined by a unit vector `(cos_angle, sin_angle)`. The gamut edge is the
/// triangle formed by the working-space primaries.
///
/// Mirrors `_find_distance_to_edge` (static inline in custom_primaries.c).
fn find_distance_to_edge(
    primaries: &[[f32; 2]; 3],
    whitepoint: &[f32; 2],
    cos_angle: f32,
    sin_angle: f32,
) -> f32 {
    let x1 = whitepoint[0];
    let y1 = whitepoint[1];
    let x2 = x1 + cos_angle;
    let y2 = y1 + sin_angle;
    let mut distance_to_edge = f32::MAX;
    for i in 0..3 {
        let next_i = if i == 2 { 0 } else { i + 1 };
        let x3 = primaries[i][0];
        let y3 = primaries[i][1];
        let x4 = primaries[next_i][0];
        let y4 = primaries[next_i][1];
        let distance = intersect_line_segments(x1, y1, x2, y2, x3, y3, x4, y4);
        if distance < distance_to_edge {
            distance_to_edge = distance;
        }
    }
    distance_to_edge
}

/// Rotate a working-space primary by `rotation` (radians) and scale its
/// distance from the white point by `scaling` (1.0 = unchanged). The result is
/// the new xy coordinate of that primary.
///
/// Mirrors `dt_rotate_and_scale_primary` (custom_primaries.c).
fn rotate_and_scale_primary(
    primaries: &[[f32; 2]; 3],
    whitepoint: &[f32; 2],
    scaling: f32,
    rotation: f32,
    primary_index: usize,
) -> [f32; 2] {
    let dx = primaries[primary_index][0] - whitepoint[0];
    let dy = primaries[primary_index][1] - whitepoint[1];
    let angle = dy.atan2(dx) + rotation;
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();
    let distance_to_edge = find_distance_to_edge(primaries, whitepoint, cos_angle, sin_angle);
    let dx_new = scaling * distance_to_edge * cos_angle;
    let dy_new = scaling * distance_to_edge * sin_angle;
    [dx_new + whitepoint[0], dy_new + whitepoint[1]]
}

/// The 4×4 identity matrix in darktable's transposed row-major storage,
/// used as the no-op fallback when the primaries configuration is degenerate.
pub const IDENTITY_4X4: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Invert a 3×3 matrix stored in darktable's "transposed" storage convention
/// (i.e. the stored array is M^T; the inverse is (M^T)^{-1} = (M^{-1})^T, which
/// is the XYZ→RGB transposed storage we need).
///
/// Returns `None` on a near-singular input, letting the caller decide the
/// fail-safe behaviour.
///
/// Faithful port of `mat3SSEinv` (matrices.c).
fn mat3_inv(src: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let a11 = src[0][0]; let a12 = src[0][1]; let a13 = src[0][2];
    let a21 = src[1][0]; let a22 = src[1][1]; let a23 = src[1][2];
    let a31 = src[2][0]; let a32 = src[2][1]; let a33 = src[2][2];

    let det = a11 * (a33 * a22 - a32 * a23)
        - a12 * (a33 * a21 - a31 * a23)
        + a13 * (a32 * a21 - a31 * a22);

    if det.abs() < 1e-7 {
        return None;
    }
    let inv_det = 1.0 / det;

    // In C: B(y,x) = src[(y-1)*4 + (x-1)] for the 4-wide rows of dt_colormatrix_t,
    // but only the first 3×3 (indices 0..2) are used. The formula below mirrors
    // mat3SSEinv's B(1..3, 1..3) cofactor expansion.
    Some([
        [
            inv_det * (a33 * a22 - a32 * a23),
            -inv_det * (a33 * a12 - a32 * a13),
            inv_det * (a23 * a12 - a22 * a13),
        ],
        [
            -inv_det * (a33 * a21 - a31 * a23),
            inv_det * (a33 * a11 - a31 * a13),
            -inv_det * (a23 * a11 - a21 * a13),
        ],
        [
            inv_det * (a32 * a21 - a31 * a22),
            -inv_det * (a32 * a11 - a31 * a12),
            inv_det * (a22 * a11 - a21 * a12),
        ],
    ])
}

/// Build the RGB→XYZ matrix (in darktable's transposed storage) from primaries
/// xy coordinates and a white-point xy, using Bruce Lindbloom's standard method.
///
/// Mirrors `dt_make_transposed_matrices_from_primaries_and_whitepoint`
/// (colorspaces.c).
fn make_transposed_matrices_from_primaries_and_whitepoint(
    primaries: &[[f32; 2]; 3],
    whitepoint: &[f32; 2],
) -> Option<[[f32; 3]; 3]> {
    // P = [ X_R/Y_R 1 Y_R/Z_R ;  ... ] per-row for each primary (unnormalised)
    let mut p = [[0.0f32; 3]; 3];
    for i in 0..3 {
        let y = sanitize_y(primaries[i][1]);
        p[i][0] = primaries[i][0] / y;
        p[i][1] = 1.0;
        p[i][2] = (1.0 - primaries[i][0] - y) / y;
    }
    // Invert P → P^{-1} (still transposed storage), then multiply by the white
    // point's XYZ to get the per-primary scale factors. If P is singular
    // (degenerate primaries), the caller handles the fallback.
    let Some(p_inv) = mat3_inv(&p) else { return None };
    let y_w = sanitize_y(whitepoint[1]);
    let xyz_white = [whitepoint[0] / y_w, 1.0, (1.0 - whitepoint[0] - y_w) / y_w];
    let scale = apply_transposed_color_matrix_3(xyz_white, &p_inv);
    // RGB_to_XYZ_transposed[i][j] = scale[i] * P[i][j]
    let mut rgb_to_xyz = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rgb_to_xyz[i][j] = scale[i] * p[i][j];
        }
    }
    Some(rgb_to_xyz)
}

/// Matrix multiply in darktable's transposed storage: `dst[k][i] = Σ_j m1[k][j] * m2[j][i]`.
///
/// In logical terms this computes `M_dst = M_m2 * M_m1` (note the swapped order),
/// which is how `dt_colormatrix_mul` works.
fn colormatrix_mul(
    m1: &[[f32; 3]; 3],
    m2: &[[f32; 3]; 3],
) -> [[f32; 3]; 3] {
    let mut dst = [[0.0f32; 3]; 3];
    for k in 0..3 {
        for i in 0..3 {
            let mut sum = 0.0f32;
            for j in 0..3 {
                sum += m1[k][j] * m2[j][i];
            }
            dst[k][i] = sum;
        }
    }
    dst
}

/// Look up the working-space primaries and white point for the given colour
/// space. The working XYZ→RGB matrix is derived at call time from these
/// primaries via [`make_transposed_matrices_from_primaries_and_whitepoint`] +
/// [`mat3_inv`].
fn working_profile(space: ColorSpace) -> (&'static [[f32; 2]; 3], &'static [f32; 2]) {
    match space {
        ColorSpace::Rec2020 => (&REC2020_PRIMARIES, &D65_WHITEPOINT),
        ColorSpace::LinearSrgb => (&SRGB_PRIMARIES, &D65_WHITEPOINT),
    }
}

/// Compute the 4×4 transposed color matrix that the primaries IOP applies per
/// pixel. Given the 8 user-facing parameters (4 hue rotation in radians, 4
/// purity scale) and the buffer's working colour space, this reproduces
/// darktable's `_calculate_adjustment_matrix` exactly.
///
/// The matrix is `matrix_out * RGB_to_XYZ(custom)`, where `RGB_to_XYZ(custom)`
/// is built from the rotated/scaled primaries and `matrix_out` is the working
/// space's XYZ→RGB. When all params are at their defaults the custom primaries
/// match the working primaries, the matrix is identity, and the stage is a no-op
/// (guarded by `is_identity` / the `to_pipeline` gate).
///
/// The result is padded to 4×4 (alpha passthrough) in darktable's transposed
/// row-major storage, ready for [`darkroom_primaries_process`].
pub fn compute_matrix(
    space: ColorSpace,
    achromatic_tint_hue: f32,
    achromatic_tint_purity: f32,
    red_hue: f32,
    red_purity: f32,
    green_hue: f32,
    green_purity: f32,
    blue_hue: f32,
    blue_purity: f32,
) -> [f32; 16] {
    let (primaries, whitepoint) = working_profile(space);

    // Rotate + scale each primary. achromatic_tint rotates+scaled the whitepoint
    // (darktable's `setup` anchors it at primary index 0 and the C `process` code
    // uses it to compute the whitepoint adjustment).
    let mut custom_primaries = [[0.0f32; 2]; 3];
    custom_primaries[0] = rotate_and_scale_primary(primaries, whitepoint, red_purity, red_hue, 0);
    custom_primaries[1] = rotate_and_scale_primary(primaries, whitepoint, green_purity, green_hue, 1);
    custom_primaries[2] = rotate_and_scale_primary(primaries, whitepoint, blue_purity, blue_hue, 2);
    let custom_whitepoint = rotate_and_scale_primary(primaries, whitepoint, achromatic_tint_purity, achromatic_tint_hue, 0);

    // Precondition: the three custom primaries must span a non-degenerate triangle.
    // The determinant epsilon inside mat3_inv is absolute and gets straddled by
    // f32 rounding noise precisely in the degenerate zone (where all three
    // primaries become collinear). 1e-9 sits ~4 orders below the smallest
    // legitimate area (2.1e-5, all purities at minimum) and ~9 orders above the
    // collinear case (~2.6e-18), so it cannot reject a usable config.
    let area = ((custom_primaries[1][0] - custom_primaries[0][0])
        * (custom_primaries[2][1] - custom_primaries[0][1])
        - (custom_primaries[2][0] - custom_primaries[0][0])
        * (custom_primaries[1][1] - custom_primaries[0][1])).abs() * 0.5;
    if area < 1e-9 { return IDENTITY_4X4; }

    // Custom RGB→XYZ (transposed storage).
    let Some(rgb_to_xyz) = make_transposed_matrices_from_primaries_and_whitepoint(&custom_primaries, &custom_whitepoint) else {
        return IDENTITY_4X4;
    };

    // Working profile's XYZ→RGB (matrix_out_transposed): invert the working
    // RGB→XYZ (which is just `rgb_to_xyz` when custom == working, i.e. at
    // defaults). At runtime we always recompute from the working primaries.
    let Some(working_rgb_to_xyz) = make_transposed_matrices_from_primaries_and_whitepoint(primaries, whitepoint) else {
        return IDENTITY_4X4;
    };
    let Some(matrix_out) = mat3_inv(&working_rgb_to_xyz) else {
        return IDENTITY_4X4;
    };

    // matrix = matrix_out * rgb_to_xyz  (logical M_out * M_custom)
    let m = colormatrix_mul(&rgb_to_xyz, &matrix_out);

    // Pad to 4×4 (row-major, darktable transposed storage): top-left 3×3 from
    // the multiplication, alpha on the diagonal.
    let mut matrix_4x4 = [0.0f32; 16];
    for i in 0..3 {
        for j in 0..3 {
            matrix_4x4[i * 4 + j] = m[i][j];
        }
    }
    matrix_4x4[15] = 1.0;

    // Output reasonableness backstop: never hand a non-finite or absurd matrix to
    // the pixel loop. Worst legitimate value across the proposed slider ranges is
    // ~1.82; 1e3 is a generous backstop that catches the 1e15+ failure mode.
    if matrix_4x4.iter().any(|v| !v.is_finite() || v.abs() > 1e3) {
        return IDENTITY_4X4;
    }

    matrix_4x4
}

/// Apply a transposed 4×4 color matrix (only first 3 rows/cols active).
///
/// `out[r] = Σ matrix[c][r] * in[c]` for c in 0..3, with alpha passed through
/// unchanged. Matches `dt_apply_transposed_color_matrix` + `darkroom_primaries_process`
/// (primaries.c `process_rgb`).
///
/// # Safety
///
/// The caller must ensure:
/// - `in_buf` points to at least `npixels * 4` valid `f32` values.
/// - `out_buf` points to at least `npixels * 4` writable `f32` slots.
/// - `in_buf` and `out_buf` do **not** alias (no overlap). The kernel reads and
///   writes in lockstep per pixel, so in-place operation is undefined behaviour.
/// - `matrix` points to at least 16 valid `f32` values.
///
/// These preconditions are guaranteed by `Pipeline::process_band`, which
/// ping-pongs between distinct `output`/`scratch` buffers.
#[no_mangle]
pub unsafe fn darkroom_primaries_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    matrix: *const f32, // float[4][4] = 16 floats, row-major
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    // matrix[row][col] stored row-major as 16 floats
    let m = std::slice::from_raw_parts(matrix, 16);
    let m_arr: &[f32; 16] = m.try_into().expect("matrix must be 16 floats");
    process_pixels(inp, out, m_arr);
}

/// Safe wrapper: applies the 4×4 matrix to each pixel's RGB channels, passing
/// alpha through unchanged. Bounds-checked slice arithmetic; no `unsafe`.
pub fn process_pixels(input: &[f32], output: &mut [f32], m: &[f32; 16]) {
    let npixels = input.len() / 4;
    debug_assert_eq!(input.len(), npixels * 4, "input must be 4-aligned");
    debug_assert_eq!(output.len(), npixels * 4, "output must be 4-aligned");
    debug_assert_ne!(input.as_ptr(), output.as_ptr(), "buffers must not alias");
    for px in 0..npixels {
        let i = &input[px * 4..px * 4 + 4];
        let o = &mut output[px * 4..px * 4 + 4];
        o[0] = m[0] * i[0] + m[4] * i[1] + m[8]  * i[2];
        o[1] = m[1] * i[0] + m[5] * i[1] + m[9]  * i[2];
        o[2] = m[2] * i[0] + m[6] * i[1] + m[10] * i[2];
        o[3] = i[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matrix_passes_through() {
        #[rustfmt::skip]
        let matrix: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let inp = [0.2f32, 0.5, 0.8, 1.0];
        let mut out = [0f32; 4];
        let m: [f32; 16] = matrix;
        process_pixels(&inp, &mut out, &m);
        assert!((out[0] - 0.2).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.8).abs() < 1e-6);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn swap_channels() {
        // swap R and G: matrix[0] = (0,1,0,0), matrix[1] = (1,0,0,0)
        #[rustfmt::skip]
        let matrix: [f32; 16] = [
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let inp = [0.1f32, 0.9, 0.5, 0.0];
        let mut out = [0f32; 4];
        let m: [f32; 16] = matrix;
        process_pixels(&inp, &mut out, &m);
        assert!((out[0] - 0.9).abs() < 1e-6);
        assert!((out[1] - 0.1).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
    }

    /// A zero-rotation / unity-purity parameter set must produce the identity
    /// matrix — the stage is then a true no-op (and `is_identity` skips it).
    #[test]
    fn default_params_yield_identity_matrix() {
        for space in [ColorSpace::Rec2020, ColorSpace::LinearSrgb] {
            let m = compute_matrix(space, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
            // Top-left 3×3 (transposed storage) should be the identity.
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0f32 } else { 0.0 };
                    assert!(
                        (m[i * 4 + j] - expected).abs() < 1e-5,
                        "{space:?}: matrix[{i}][{j}] = {} ≠ {}",
                        m[i * 4 + j],
                        expected
                    );
                }
            }
            // Alpha passthrough.
            assert!((m[15] - 1.0).abs() < 1e-6);
        }
    }

    /// Non-default params must shift at least one matrix element, confirming the
    /// hue/purity inputs actually drive the computation.
    #[test]
    fn red_hue_shifts_red_primary() {
        let identity = compute_matrix(ColorSpace::Rec2020, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let rotated = compute_matrix(ColorSpace::Rec2020, 0.0, 0.0, 0.1, 1.0, 0.0, 1.0, 0.0, 1.0);
        let changed = identity
            .iter()
            .zip(rotated.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "rotating red hue must change the matrix");
    }

    /// The matrix must be deterministic — same inputs, same output.
    #[test]
    fn matrix_is_deterministic() {
        let a = compute_matrix(ColorSpace::Rec2020, 0.1, 0.2, 0.3, 1.5, -0.2, 0.8, 0.15, 1.2);
        let b = compute_matrix(ColorSpace::Rec2020, 0.1, 0.2, 0.3, 1.5, -0.2, 0.8, 0.15, 1.2);
        assert_eq!(a, b);
    }

    /// At red_hue=120° the primaries are collinear (triangle area ~2.6e-18),
    /// so the matrix is structurally singular. compute_matrix must fall back to
    /// the identity rather than emitting a 1e16 matrix.
    #[test]
    fn degenerate_hue_returns_identity() {
        // red_hue = 120 deg (in radians) makes the red primary land on the G-B edge.
        let m = compute_matrix(ColorSpace::Rec2020, 0.0, 0.0,
            120f32.to_radians(), 1.0, 0.0, 1.0, 0.0, 1.0);
        assert_eq!(m, IDENTITY_4X4,
            "degenerate primaries must yield identity fallback, got max|A| = {}",
            m[..15].iter().map(|v| v.abs()).fold(0.0f32, f32::max));
    }

    /// Over the proposed slider ranges (hue ±20°, purity 0.5..1.5, tint hue ±180°,
    /// tint purity 0..0.2), every coefficient must stay reasonable. The worst
    /// measured value is ~1.82; we assert < 10.0 to catch any future range
    /// widening that re-enters the degenerate zone.
    #[test]
    fn matrix_bounded_over_proposed_slider_ranges() {
        for th in [-20.0_f32, 0.0, 20.0] {
            for tp in [0.0_f32, 0.1, 0.2] {
                for rh in [-20.0_f32, 0.0, 20.0] {
                    for rp in [0.5_f32, 1.0, 1.5] {
                        for gh in [-20.0_f32, 0.0, 20.0] {
                            for gp in [0.5_f32, 1.0, 1.5] {
                                for bh in [-20.0_f32, 0.0, 20.0] {
                                    for bp in [0.5_f32, 1.0, 1.5] {
                                        let m = compute_matrix(ColorSpace::Rec2020,
                                            th.to_radians(), tp,
                                            rh.to_radians(), rp,
                                            gh.to_radians(), gp,
                                            bh.to_radians(), bp);
                                        for v in &m {
                                            assert!(v.is_finite(),
                                                "non-finite coefficient with th={th} tp={tp} rh={rh} rp={rp}");
                                            assert!(v.abs() < 10.0,
                                                "coefficient {v} out of bounds with th={th} tp={tp} rh={rh} rp={rp}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
