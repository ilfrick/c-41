//! Input-profile tone curves ported from `src/common/iop_profile.c`
//! (`_apply_tonecurves`, iop_profile.c:364-420, m4-187).
//!
//! What the loops do: each RGB pixel of a 4-lane (RGBA) image is mapped
//! through its per-channel tone curve — `extrapolate_lut` while the input
//! value is below 1.0, `eval_exp` (the unbounded exponential fit) at and
//! above 1.0. The C function holds two `DT_OMP_FOR(collapse(2))` loops: an
//! all-channels fast path taken when every LUT is active, and a
//! per-channel-sentinel path taken when only some LUTs are active. A LUT
//! whose first entry is negative (`lut[c][0] < 0.0`) marks a linear channel
//! and is skipped; when all three are linear neither branch runs.
//!
//! The single-threaded Rust port is identical for any OpenMP thread count:
//! pixels are independent (each output lane is written exactly once from its
//! own input lane), so the merge order is irrelevant.
//!
//! Semantic edge cases (all pinned by tests):
//! - Inactive (linear) channels are left **untouched** in `image_out` —
//!   they are NOT copied from `image_in`. Callers pre-fill or reuse the
//!   buffer (notably `_transform_lab_to_rgb_matrix` runs fully in place).
//!   This differs from `crate::color::apply_trc`, which passes inactive
//!   channels through from the input pixel.
//! - All three channels linear is a complete no-op: `image_out` is not
//!   written at all, including alpha.
//! - The alpha lane (`k + 3`) is never read or written in any branch; the C
//!   comment calls this out explicitly ("some code needs image_out[3]
//!   preserved").
//! - `v < 1.0` selects the LUT path (negative inputs clamp to `lut[0]` inside
//!   `extrapolate_lut`); `v >= 1.0` — including exactly 1.0 — selects
//!   `eval_exp`.
//! - The LUT/eval_exp arithmetic itself lives in the shared
//!   `crate::color::{extrapolate_lut, eval_exp}` helpers and is reused here,
//!   not duplicated.
//!
//! Degenerate/short buffers: zero `width`/`height`, `lutsize < 2` (below
//! which both the C loop and the shared helper index out of bounds — real
//! profiles carry `lutsize` floats per LUT, default `0x10000`), a dimension
//! arithmetic overflow, a LUT shorter than `lutsize`, or a coefficient slice
//! shorter than 3 floats is a no-op — no panic, no out-of-bounds access.
//! Well-formed callers (the two `_transform_*_matrix` wrappers, which forward
//! the profile's own `lutsize`-long LUTs over exactly `4*width*height`
//! floats) never hit this path.
//!
//! Aliasing: the split kernel requires non-overlapping buffers (like the C
//! `restrict` qualifiers). The in-place call
//! (`_transform_lab_to_rgb_matrix` invokes the C function with
//! `image_out, image_out`) is served by [`apply_tonecurves_inplace`]; the
//! FFI export dispatches on pointer equality.

use crate::color::{
    apply_transposed_color_matrix, apply_trc, eval_exp, extrapolate_lut, lab_to_xyz, xyz_to_lab,
};

/// Map one channel value through its tone curve.
///
/// Returns `None` for inactive (linear, `lut[0] < 0.0`) channels so the
/// caller leaves that output lane untouched. Active channels take the LUT
/// below 1.0 and the unbounded exponential fit at and above 1.0.
#[inline(always)]
fn map_channel(v: f32, lut: &[f32], coeff: &[f32], lutsize: usize) -> Option<f32> {
    if lut[0] < 0.0 {
        return None;
    }
    Some(if v < 1.0 {
        extrapolate_lut(lut, v, lutsize)
    } else {
        eval_exp(coeff, v)
    })
}

/// Validate dimensions and buffer lengths for the tone-curve kernels.
///
/// Returns the pixel count (`width * height`) on success. Fails on zero
/// dimensions, `lutsize < 2`, dimension arithmetic overflow, image buffers
/// shorter than `4*width*height` floats, LUTs shorter than `lutsize`, or
/// coefficient slices shorter than 3 floats.
fn checked_len(
    image_in_len: usize,
    image_out_len: usize,
    luts: [&[f32]; 3],
    coeffs: [&[f32]; 3],
    width: usize,
    height: usize,
    lutsize: usize,
) -> Option<usize> {
    if width == 0 || height == 0 || lutsize < 2 {
        return None;
    }
    let npixels = width.checked_mul(height)?;
    let pix_len = npixels.checked_mul(4)?;
    if image_in_len < pix_len || image_out_len < pix_len {
        return None;
    }
    for lut in luts {
        if lut.len() < lutsize {
            return None;
        }
    }
    for coeff in coeffs {
        if coeff.len() < 3 {
            return None;
        }
    }
    Some(npixels)
}

/// Apply the input-profile tone curves (`_apply_tonecurves`, split buffers).
///
/// Port of the two `DT_OMP_FOR(collapse(2))` loops at iop_profile.c:387-398
/// (all LUTs active) and iop_profile.c:404-418 (per-channel sentinel), with
/// the branch structure preserved: the fast path writes all three channels
/// per pixel, the sentinel path writes only active channels, and the
/// all-linear case writes nothing. Channels 0..2 are mapped per
/// [`map_channel`]; channel 3 (alpha) is never touched. `lutr/lutg/lutb`
/// each hold `lutsize` floats; each coefficient slice holds 3 floats.
///
/// `image_in` and `image_out` must not overlap (matching the C `restrict`
/// qualifiers); use [`apply_tonecurves_inplace`] for the in-place call.
/// Degenerate/short inputs are a no-op as documented above.
#[allow(clippy::too_many_arguments)]
pub fn apply_tonecurves(
    image_in: &[f32],
    image_out: &mut [f32],
    width: usize,
    height: usize,
    lutr: &[f32],
    lutg: &[f32],
    lutb: &[f32],
    unbounded_coeffsr: &[f32],
    unbounded_coeffsg: &[f32],
    unbounded_coeffsb: &[f32],
    lutsize: usize,
) {
    let luts = [lutr, lutg, lutb];
    let coeffs = [unbounded_coeffsr, unbounded_coeffsg, unbounded_coeffsb];
    let Some(npixels) = checked_len(
        image_in.len(),
        image_out.len(),
        luts,
        coeffs,
        width,
        height,
        lutsize,
    ) else {
        return;
    };
    let active = [
        luts[0][0] >= 0.0,
        luts[1][0] >= 0.0,
        luts[2][0] >= 0.0,
    ];
    if active[0] && active[1] && active[2] {
        for px in 0..npixels {
            let b = px * 4;
            for c in 0..3 {
                let v = image_in[b + c];
                image_out[b + c] = if v < 1.0 {
                    extrapolate_lut(luts[c], v, lutsize)
                } else {
                    eval_exp(coeffs[c], v)
                };
            }
        }
    } else if active[0] || active[1] || active[2] {
        for px in 0..npixels {
            let b = px * 4;
            for c in 0..3 {
                if active[c] {
                    let v = image_in[b + c];
                    image_out[b + c] = if v < 1.0 {
                        extrapolate_lut(luts[c], v, lutsize)
                    } else {
                        eval_exp(coeffs[c], v)
                    };
                }
            }
        }
    }
    // all channels linear: neither C branch runs — output fully untouched.
}

/// Apply the input-profile tone curves in place.
///
/// Same semantics as [`apply_tonecurves`] on a single buffer (each lane is
/// read before it is written, so per-lane in-place mapping is exact). Serves
/// the `_transform_lab_to_rgb_matrix` call site, which invokes the C
/// function with `image_out, image_out`. Degenerate/short inputs are a
/// no-op as documented above.
#[allow(clippy::too_many_arguments)]
pub fn apply_tonecurves_inplace(
    image: &mut [f32],
    width: usize,
    height: usize,
    lutr: &[f32],
    lutg: &[f32],
    lutb: &[f32],
    unbounded_coeffsr: &[f32],
    unbounded_coeffsg: &[f32],
    unbounded_coeffsb: &[f32],
    lutsize: usize,
) {
    let luts = [lutr, lutg, lutb];
    let coeffs = [unbounded_coeffsr, unbounded_coeffsg, unbounded_coeffsb];
    let Some(npixels) = checked_len(image.len(), image.len(), luts, coeffs, width, height, lutsize)
    else {
        return;
    };
    let active = [
        luts[0][0] >= 0.0,
        luts[1][0] >= 0.0,
        luts[2][0] >= 0.0,
    ];
    if !(active[0] || active[1] || active[2]) {
        return;
    }
    for px in 0..npixels {
        let b = px * 4;
        for c in 0..3 {
            if active[c] {
                let v = image[b + c];
                image[b + c] = if v < 1.0 {
                    extrapolate_lut(luts[c], v, lutsize)
                } else {
                    eval_exp(coeffs[c], v)
                };
            }
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for [`apply_tonecurves`]: channel-outer
/// looping (for each channel, sweep every pixel) through the shared
/// [`map_channel`] helper — writing a lane only when the helper returns
/// `Some` — where the kernel loops pixel-outer with an inlined branch.
/// Same no-op preconditions: well-formed buffers only.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ref_apply_tonecurves(
    image_in: &[f32],
    image_out: &mut [f32],
    width: usize,
    height: usize,
    lutr: &[f32],
    lutg: &[f32],
    lutb: &[f32],
    unbounded_coeffsr: &[f32],
    unbounded_coeffsg: &[f32],
    unbounded_coeffsb: &[f32],
    lutsize: usize,
) {
    let luts = [lutr, lutg, lutb];
    let coeffs = [unbounded_coeffsr, unbounded_coeffsg, unbounded_coeffsb];
    let Some(npixels) = checked_len(
        image_in.len(),
        image_out.len(),
        luts,
        coeffs,
        width,
        height,
        lutsize,
    ) else {
        return;
    };
    for (c, (&lut, &coeff)) in luts.iter().zip(coeffs.iter()).enumerate() {
        if lut[0] < 0.0 {
            continue;
        }
        for px in 0..npixels {
            let idx = px * 4 + c;
            if let Some(mapped) = map_channel(image_in[idx], lut, coeff, lutsize) {
                image_out[idx] = mapped;
            }
        }
    }
}

// ── RGB↔Lab matrix transforms (m4-188) ─────────────────────────────────────────
//
// Ports of `_transform_rgb_to_lab_matrix` (iop_profile.c:403-448) and
// `_transform_lab_to_rgb_matrix` (iop_profile.c:451-488): the non-LCMS
// matrix paths. Each pixel is mapped RGB→XYZ via the profile's pre-transposed
// `matrix_in_transposed` (resp. Lab→XYZ then `matrix_out_transposed`→RGB)
// with the shared D50 [`xyz_to_lab`]/[`lab_to_xyz`] conversions.
//
// The C wrappers keep their tone-curve orchestration (`_apply_tonecurves`
// before the RGB→Lab loops when `nonlinearlut`, after the Lab→RGB loop when
// `nonlinearlut`); only the three `DT_OMP_FOR` loop bodies move here:
//   - RGB→Lab nonlinear branch: in-place over `image_out` (tone curves
//     already ran `image_in`→`image_out`), served by
//     [`rgb_to_lab_matrix_inplace`];
//   - RGB→Lab linear branch: split `image_in`→`image_out`, [`rgb_to_lab_matrix`];
//   - Lab→RGB: split `image_in`→`image_out` with alpha preservation,
//     [`lab_to_rgb_matrix`] (alias-tolerant via [`lab_to_rgb_matrix_inplace`]
//     — the C comment notes callers rely on in-place conversion).
//
// Semantic edge cases (all pinned by tests):
// - RGB→Lab output alpha is forced to **0.0**: `dt_XYZ_to_Lab` writes
//   `Lab[3] = 0*(f[3]-0)-0` under the normal 4-channel vector build (only
//   under the rare `DT_NO_VECTORIZATION` build would it loop 3 channels and
//   leave alpha untouched — not a target). The shared [`xyz_to_lab`] instead
//   preserves `xyz[3]`, so the kernels overwrite lane 3 with 0.0 after the
//   call. `Lab[0..2]` never depend on `xyz[3]` (only `xyz[0..2]` feed `f`).
// - Lab→RGB output alpha is the **input** alpha (`in[3]` read before the
//   write; `dt_Lab_to_XYZ` zeroes `XYZ[3]` and the matrix write to `out[3]`
//   is then overwritten with the saved alpha). The shared [`lab_to_xyz`]
//   carries `lab[3]` into `xyz[3]`, but that lane is never read by
//   [`apply_transposed_color_matrix`] (only `in[0..2]`), so reusing the
//   helper is bit-exact here.
// - Matrix order is the transposed convention
//   (`out[r] = m[0][r]*in[0] + m[1][r]*in[1] + m[2][r]*in[2]`, `r` over all
//   4 lanes); the profile hands over its already-transposed matrices and the
//   kernels must NOT transpose again.
//
// Degenerate/short buffers: zero `width`/`height`, dimension arithmetic
// overflow, image buffers shorter than `4*width*height` floats, or a matrix
// shorter than 16 floats is a no-op — no panic, no out-of-bounds access.
// Well-formed callers forward the profile's own 16-float matrices over
// exactly `4*width*height` floats and never hit this path.
//
// Aliasing: the split kernels require non-overlapping buffers. In-place
// traffic is served by [`rgb_to_lab_matrix_inplace`]/
// [`lab_to_rgb_matrix_inplace`]; the FFI exports dispatch on pointer
// equality. Partial overlap is unsupported (stricter than C).

/// Validate dimensions and buffer lengths for the matrix kernels.
///
/// Returns the pixel count (`width * height`) on success. Fails on zero
/// dimensions, dimension arithmetic overflow, image buffers shorter than
/// `4*width*height` floats, or a matrix shorter than 16 floats
/// (`dt_colormatrix_t`).
fn checked_matrix_len(
    image_in_len: usize,
    image_out_len: usize,
    matrix_len: usize,
    width: usize,
    height: usize,
) -> Option<usize> {
    if width == 0 || height == 0 {
        return None;
    }
    let npixels = width.checked_mul(height)?;
    let pix_len = npixels.checked_mul(4)?;
    if image_in_len < pix_len || image_out_len < pix_len {
        return None;
    }
    if matrix_len < 16 {
        return None;
    }
    Some(npixels)
}

/// Reinterpret a 16-float `dt_colormatrix_t` slice as `[[f32; 4]; 4]`.
///
/// The caller guarantees `matrix.len() >= 16` (checked by
/// [`checked_matrix_len`]); row-major layout matches the C type exactly, and
/// `[[f32; 4]; 4]` has the same alignment as `f32`, so the cast is sound.
#[inline(always)]
fn as_colormatrix(matrix: &[f32]) -> &[[f32; 4]; 4] {
    // SAFETY: `matrix` holds at least 16 contiguous `f32`s with alignment 4,
    // which satisfies `[[f32; 4]; 4]` (size 64, alignment 4).
    unsafe { &*(matrix.as_ptr() as *const [[f32; 4]; 4]) }
}

/// RGB→Lab via the profile matrix, split buffers.
///
/// Port of the linear `DT_OMP_FOR` loop in `_transform_rgb_to_lab_matrix`:
/// `xyz = matrix_in_transposed * rgb; lab = XYZ_to_Lab(xyz)` with output
/// alpha forced to 0.0 (see above). `matrix` is the 16-float transposed
/// profile matrix. Buffers must not overlap; degenerate/short inputs are a
/// no-op.
pub fn rgb_to_lab_matrix(
    image_in: &[f32],
    image_out: &mut [f32],
    matrix: &[f32],
    width: usize,
    height: usize,
) {
    let Some(npixels) =
        checked_matrix_len(image_in.len(), image_out.len(), matrix.len(), width, height)
    else {
        return;
    };
    let m = as_colormatrix(matrix);
    for px in 0..npixels {
        let b = px * 4;
        let inp = [
            image_in[b],
            image_in[b + 1],
            image_in[b + 2],
            image_in[b + 3],
        ];
        let xyz = apply_transposed_color_matrix(&inp, m);
        let lab = xyz_to_lab(xyz);
        image_out[b] = lab[0];
        image_out[b + 1] = lab[1];
        image_out[b + 2] = lab[2];
        // dt_XYZ_to_Lab zeroes the fourth lane (4-channel vector build);
        // xyz_to_lab preserves xyz[3] instead, so force it here.
        image_out[b + 3] = 0.0;
    }
}

/// RGB→Lab via the profile matrix, in place.
///
/// Same per-pixel semantics as [`rgb_to_lab_matrix`] on a single buffer (each
/// lane is read into a local before it is written, so in-place mapping is
/// exact). Serves the nonlinear branch, whose C loop runs over `image_out`
/// after the tone curves. Degenerate/short inputs are a no-op.
pub fn rgb_to_lab_matrix_inplace(
    image: &mut [f32],
    matrix: &[f32],
    width: usize,
    height: usize,
) {
    let Some(npixels) = checked_matrix_len(image.len(), image.len(), matrix.len(), width, height)
    else {
        return;
    };
    let m = as_colormatrix(matrix);
    for px in 0..npixels {
        let b = px * 4;
        let inp = [image[b], image[b + 1], image[b + 2], image[b + 3]];
        let xyz = apply_transposed_color_matrix(&inp, m);
        let lab = xyz_to_lab(xyz);
        image[b] = lab[0];
        image[b + 1] = lab[1];
        image[b + 2] = lab[2];
        image[b + 3] = 0.0;
    }
}

/// Lab→RGB via the profile matrix, split buffers.
///
/// Port of the `DT_OMP_FOR` loop in `_transform_lab_to_rgb_matrix`:
/// `xyz = Lab_to_XYZ(lab); rgb = matrix_out_transposed * xyz` with the input
/// alpha saved and restored afterwards. `matrix` is the 16-float transposed
/// profile matrix. Buffers must not overlap; degenerate/short inputs are a
/// no-op.
pub fn lab_to_rgb_matrix(
    image_in: &[f32],
    image_out: &mut [f32],
    matrix: &[f32],
    width: usize,
    height: usize,
) {
    let Some(npixels) =
        checked_matrix_len(image_in.len(), image_out.len(), matrix.len(), width, height)
    else {
        return;
    };
    let m = as_colormatrix(matrix);
    for px in 0..npixels {
        let b = px * 4;
        let alpha = image_in[b + 3];
        let lab = [image_in[b], image_in[b + 1], image_in[b + 2], alpha];
        // lab[3] never affects xyz[0..2] (dt_Lab_to_XYZ zeroes XYZ[3]
        // regardless) and xyz[3] is never read by the matrix multiply.
        let xyz = lab_to_xyz(lab);
        let rgb = apply_transposed_color_matrix(&xyz, m);
        image_out[b] = rgb[0];
        image_out[b + 1] = rgb[1];
        image_out[b + 2] = rgb[2];
        image_out[b + 3] = alpha;
    }
}

/// Lab→RGB via the profile matrix, in place.
///
/// Same per-pixel semantics as [`lab_to_rgb_matrix`] on a single buffer.
/// Serves callers that convert in place (the C comment: "some code does
/// in-place conversions and relies on alpha being preserved").
/// Degenerate/short inputs are a no-op.
pub fn lab_to_rgb_matrix_inplace(
    image: &mut [f32],
    matrix: &[f32],
    width: usize,
    height: usize,
) {
    let Some(npixels) = checked_matrix_len(image.len(), image.len(), matrix.len(), width, height)
    else {
        return;
    };
    let m = as_colormatrix(matrix);
    for px in 0..npixels {
        let b = px * 4;
        let alpha = image[b + 3];
        let lab = [image[b], image[b + 1], image[b + 2], alpha];
        let xyz = lab_to_xyz(lab);
        let rgb = apply_transposed_color_matrix(&xyz, m);
        image[b] = rgb[0];
        image[b + 1] = rgb[1];
        image[b + 2] = rgb[2];
        image[b + 3] = alpha;
    }
}

// ── Independent reference implementations for bit-exactness tests ─────────────

// Local copies of the D50 Lab constants for the divergent references below,
// so they do not call into the shared-helper path. Values mirror
// `crate::color::{D50, D50_INV, LAB_EPSILON, LAB_KAPPA, LAB_CBRT_EPSILON}`
// (colorspaces_inline_conversions.h:144-145, dt_XYZ_to_Lab/dt_Lab_to_XYZ).
#[allow(dead_code)]
const REF_D50: [f32; 3] = [0.9642, 1.0, 0.8249];
#[allow(dead_code)]
const REF_D50_INV: [f32; 3] = [1.0 / 0.9642, 1.0, 1.0 / 0.8249];
#[allow(dead_code)]
const REF_LAB_EPSILON: f32 = 216.0 / 24389.0;
#[allow(dead_code)]
const REF_LAB_KAPPA: f32 = 24389.0 / 27.0;
#[allow(dead_code)]
const REF_LAB_CBRT_EPSILON: f32 = 0.20689655172413796;

/// Structurally divergent reference for [`rgb_to_lab_matrix`]: channel-outer
/// accumulation (for each output lane, dot the matrix column against the
/// pixel, then run an inlined XYZ→Lab conversion) where the kernel loops
/// pixel-outer through the shared helpers. Arithmetic association order is
/// kept identical so the comparison is bit-exact.
#[allow(dead_code)]
fn ref_rgb_to_lab_matrix(
    image_in: &[f32],
    image_out: &mut [f32],
    matrix: &[f32],
    width: usize,
    height: usize,
) {
    let Some(npixels) =
        checked_matrix_len(image_in.len(), image_out.len(), matrix.len(), width, height)
    else {
        return;
    };
    for px in 0..npixels {
        let b = px * 4;
        // transposed application, lane by lane: out[r] = m[0][r]*in0 + ...
        // (row-major flat layout: m[c*4+r]).
        let mut xyz = [0.0f32; 4];
        for r in 0..4 {
            xyz[r] = matrix[r] * image_in[b] + matrix[4 + r] * image_in[b + 1]
                + matrix[8 + r] * image_in[b + 2];
        }
        // inlined dt_XYZ_to_Lab (D50): same ops as xyz_to_lab, written out.
        let mut f = [0.0f32; 3];
        for i in 0..3 {
            let x = xyz[i] * REF_D50_INV[i];
            f[i] = if x > REF_LAB_EPSILON {
                x.cbrt()
            } else {
                (REF_LAB_KAPPA * x + 16.0) / 116.0
            };
        }
        image_out[b] = 116.0 * f[1] - 16.0;
        image_out[b + 1] = 500.0 * (f[0] - f[1]);
        image_out[b + 2] = -200.0 * (f[2] - f[1]);
        image_out[b + 3] = 0.0;
    }
}

/// Structurally divergent reference for [`lab_to_rgb_matrix`]: per-pixel
/// scalar staging through an inlined Lab→XYZ conversion (explicit fy/fx/fz
/// temporaries and a direct `lab_f_inv` copy) followed by a lane-by-lane
/// transposed matrix multiply, where the kernel threads whole `[f32; 4]`
/// arrays through the shared helpers. Arithmetic association order is kept
/// identical so the comparison is bit-exact.
#[allow(dead_code)]
fn ref_lab_to_rgb_matrix(
    image_in: &[f32],
    image_out: &mut [f32],
    matrix: &[f32],
    width: usize,
    height: usize,
) {
    let Some(npixels) =
        checked_matrix_len(image_in.len(), image_out.len(), matrix.len(), width, height)
    else {
        return;
    };
    for px in 0..npixels {
        let b = px * 4;
        let (l, a, bb) = (image_in[b], image_in[b + 1], image_in[b + 2]);
        let alpha = image_in[b + 3];
        let fy = (l + 16.0) / 116.0;
        let fx = a / 500.0 + fy;
        let fz = fy - bb / 200.0;
        let xyz = [
            REF_D50[0] * ref_lab_f_inv(fx),
            REF_D50[1] * ref_lab_f_inv(fy),
            REF_D50[2] * ref_lab_f_inv(fz),
        ];
        for r in 0..3 {
            image_out[b + r] =
                matrix[r] * xyz[0] + matrix[4 + r] * xyz[1] + matrix[8 + r] * xyz[2];
        }
        image_out[b + 3] = alpha;
    }
}

/// Local copy of the `lab_f_inv` step so the Lab→RGB reference does not call
/// the shared helper path.
#[allow(dead_code)]
fn ref_lab_f_inv(x: f32) -> f32 {
    if x > REF_LAB_CBRT_EPSILON {
        x * x * x
    } else {
        (116.0 * x - 16.0) / REF_LAB_KAPPA
    }
}

// ── FFI exports (matrix transforms) ────────────────────────────────────────────

/// Validate the raw FFI arguments shared by both matrix-transform exports.
///
/// Returns `(npixels, pix_len)` on success: all pointers non-null, `width`
/// and `height` nonzero and within the C `int` domain (negative values arrive
/// as huge `size_t` and must not reach slice construction), and the
/// `4*width*height` product non-overflowing. (The safe kernels re-check
/// buffer lengths defensively.)
fn checked_ffi_matrix_args(
    image_in: *const f32,
    image_out: *mut f32,
    matrix: *const f32,
    width: usize,
    height: usize,
) -> Option<usize> {
    if image_in.is_null() || image_out.is_null() || matrix.is_null() {
        return None;
    }
    if width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
    {
        return None;
    }
    let npixels = width.checked_mul(height)?;
    let pix_len = npixels.checked_mul(4)?;
    Some(pix_len)
}

/// RGB→Lab via the profile matrix (`_transform_rgb_to_lab_matrix` loops).
///
/// `image_in`/`image_out` each hold `4*width*height` floats (RGBA); only used
/// by the linear branch split — the nonlinear branch calls with
/// `image_in == image_out` (tone curves already ran into `image_out`) and is
/// dispatched to [`rgb_to_lab_matrix_inplace`]. `matrix` holds 16 floats: the
/// profile's `matrix_in_transposed` exactly as stored (already transposed;
/// applied without further transposition). Output alpha is forced to 0.0,
/// matching `dt_XYZ_to_Lab`. Otherwise the buffers must not overlap (partial
/// overlap is unsupported, stricter than C).
///
/// # Safety
/// All pointers must be valid for the stated lengths; the C caller forwards
/// the profile's own 16-float matrix over exactly `4*width*height` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_iop_profile_rgb_to_lab_matrix(
    image_in: *const f32,
    image_out: *mut f32,
    width: usize,
    height: usize,
    matrix: *const f32,
) {
    let Some(pix_len) =
        checked_ffi_matrix_args(image_in, image_out, matrix, width, height)
    else {
        return;
    };
    let matrix = std::slice::from_raw_parts(matrix, 16);
    if std::ptr::eq(image_in, image_out) {
        // nonlinear branch (runs over image_out after the tone curves):
        // a single mutable slice avoids shared+mutable aliasing.
        let image = std::slice::from_raw_parts_mut(image_out, pix_len);
        rgb_to_lab_matrix_inplace(image, matrix, width, height);
    } else {
        let image_in = std::slice::from_raw_parts(image_in, pix_len);
        let image_out = std::slice::from_raw_parts_mut(image_out, pix_len);
        rgb_to_lab_matrix(image_in, image_out, matrix, width, height);
    }
}

/// Lab→RGB via the profile matrix (`_transform_lab_to_rgb_matrix` loop).
///
/// `image_in`/`image_out` each hold `4*width*height` floats (RGBA); input
/// alpha is preserved into the output. `matrix` holds 16 floats: the
/// profile's `matrix_out_transposed` exactly as stored. May be called in
/// place (`image_in == image_out`, which some callers rely on); otherwise the
/// buffers must not overlap (partial overlap is unsupported, stricter
/// than C).
///
/// # Safety
/// All pointers must be valid for the stated lengths; the C caller forwards
/// the profile's own 16-float matrix over exactly `4*width*height` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_iop_profile_lab_to_rgb_matrix(
    image_in: *const f32,
    image_out: *mut f32,
    width: usize,
    height: usize,
    matrix: *const f32,
) {
    let Some(pix_len) =
        checked_ffi_matrix_args(image_in, image_out, matrix, width, height)
    else {
        return;
    };
    let matrix = std::slice::from_raw_parts(matrix, 16);
    if std::ptr::eq(image_in, image_out) {
        let image = std::slice::from_raw_parts_mut(image_out, pix_len);
        lab_to_rgb_matrix_inplace(image, matrix, width, height);
    } else {
        let image_in = std::slice::from_raw_parts(image_in, pix_len);
        let image_out = std::slice::from_raw_parts_mut(image_out, pix_len);
        lab_to_rgb_matrix(image_in, image_out, matrix, width, height);
    }
}

// ── RGB→RGB matrix transform (m4-189) ──────────────────────────────────────────
//
// Port of `_transform_matrix_rgb` (iop_profile.c:470-558): the non-LCMS
// RGB→RGB path between two matrix profiles. The C wrapper premultiplies the
// two 3x3 profile matrices once per image
// (`_matrix = to->matrix_out * from->matrix_in`, then `transpose_3xSSE`) and
// hands the kernel the already-transposed 16-float product; the kernel never
// sees the two factors and must NOT transpose again. Two `DT_OMP_FOR` loops
// move here:
//   - nonlinear branch (either profile `nonlinearlut`): per-pixel
//     linearize (input TRC) → premultiplied matrix → delinearize (output
//     TRC), [`matrix_rgb`] with either flag set;
//   - linear branch (both linear): premultiplied matrix only.
// The C wrapper keeps the premultiplication, the transpose, and the outer
// `nonlinearlut` branch; both branches call the single FFI export (the linear
// branch with both flags 0).
//
// Per-pixel semantics, mirroring the C loop bodies exactly:
// - Linearize (only when `nonlinear_from`): each channel through the shared
//   [`apply_trc`] (LUT below 1.0, `eval_exp` at and above, passthrough for
//   channels whose LUT is marked linear via `lut[0] < 0.0`). Otherwise the
//   pixel is copied unchanged (the C `for_each_channel` copy, all 4 lanes).
// - Matrix: shared [`apply_transposed_color_matrix`] over lanes 0..2 (lane 3
//   of the input is never read, matching C).
// - Delinearize (only when `nonlinear_to`): same TRC shape over the matrix
//   result; otherwise the matrix result is stored directly.
//
// Semantic edge cases (all pinned by tests):
// - Each side carries its OWN `lutsize` (`from->lutsize` for the input TRCs,
//   `to->lutsize` for the output TRCs, as in C — the two profiles may differ).
// - Alpha (lane 3) is written only on matrix-direct-to-output paths, i.e.
//   when `nonlinear_to` is false (the linear branch, and the
//   from-nonlinear/to-linear sub-case): `out[3] = m[0][3]*in0 + m[1][3]*in1 +
//   m[2][3]*in2` through the shared helper. The transposed product carries
//   zero padding there (`transpose_3xSSE` writes 0.0), so finite inputs yield
//   exactly 0.0 — but an infinite input yields NaN (`0.0 * inf`), and the
//   helper reproduces that bit-exactly, which is why lane 3 is computed
//   rather than forced to 0.0. When `nonlinear_to` is true, lane 3 is NEVER
//   written (preserved as-is in `image_out`), as in C.
// - `v < 1.0` selects the LUT path (negatives clamp to `lut[0]` inside
//   `extrapolate_lut`); `v >= 1.0` — including exactly 1.0 — selects
//   `eval_exp`. Inactive (linear-marked) channels pass the value through on
//   BOTH sides (unlike [`apply_tonecurves`'s leave-untouched rule — here the
//   C loop assigns `rgb[c] = in[c]` / `out[c] = temp[c]`).
//
// Degenerate/short buffers: zero `width`/`height`, dimension arithmetic
// overflow, image buffers shorter than `4*width*height` floats, or a matrix
// shorter than 16 floats is a no-op — no panic, no out-of-bounds access.
// TRC tables are validated PER SIDE and only when that side is nonlinear: a
// linear side accepts empty LUT/coefficient slices (never touched); a
// nonlinear side requires `lutsize >= 2`, LUTs of at least `lutsize` floats
// and coefficient slices of at least 3 floats. Well-formed callers forward
// the profiles' own `lutsize`-long LUTs and never hit these paths.
//
// Aliasing: split buffers must not overlap (matching the C `restrict`
// qualifiers); no in-place kernel is provided (unlike the m4-188 pair —
// `_transform_matrix_rgb` is `restrict`-qualified and has no in-place
// callers, so the FFI export offers no pointer-equality dispatch).

/// Validate one side's TRC tables: `lutsize >= 2` (below which both the C
/// loop and `extrapolate_lut` index out of bounds), every LUT at least
/// `lutsize` floats, every coefficient slice at least 3 floats.
fn check_trc_side(luts: [&[f32]; 3], coeffs: [&[f32]; 3], lutsize: usize) -> bool {
    if lutsize < 2 {
        return false;
    }
    for lut in luts {
        if lut.len() < lutsize {
            return false;
        }
    }
    for coeff in coeffs {
        if coeff.len() < 3 {
            return false;
        }
    }
    true
}

/// Validate dimensions and buffer lengths for the RGB→RGB matrix kernel.
///
/// Returns the pixel count (`width * height`) on success. Always fails on
/// zero dimensions, dimension arithmetic overflow, image buffers shorter
/// than `4*width*height` floats, or a matrix shorter than 16 floats
/// (`dt_colormatrix_t`). TRC tables are checked per side, only for nonlinear
/// sides (see [`check_trc_side`]).
#[allow(clippy::too_many_arguments)]
fn checked_matrix_rgb_len(
    image_in_len: usize,
    image_out_len: usize,
    matrix_len: usize,
    luts_in: [&[f32]; 3],
    coeffs_in: [&[f32]; 3],
    lutsize_in: usize,
    nonlinear_from: bool,
    luts_out: [&[f32]; 3],
    coeffs_out: [&[f32]; 3],
    lutsize_out: usize,
    nonlinear_to: bool,
    width: usize,
    height: usize,
) -> Option<usize> {
    if width == 0 || height == 0 {
        return None;
    }
    let npixels = width.checked_mul(height)?;
    let pix_len = npixels.checked_mul(4)?;
    if image_in_len < pix_len || image_out_len < pix_len {
        return None;
    }
    if matrix_len < 16 {
        return None;
    }
    if nonlinear_from && !check_trc_side(luts_in, coeffs_in, lutsize_in) {
        return None;
    }
    if nonlinear_to && !check_trc_side(luts_out, coeffs_out, lutsize_out) {
        return None;
    }
    Some(npixels)
}

/// RGB→RGB via the premultiplied profile matrix, with optional TRCs.
///
/// Port of the two `DT_OMP_FOR` loops in `_transform_matrix_rgb`: the
/// nonlinear loop (iop_profile.c:497-545) when either flag is set, the linear
/// loop (iop_profile.c:549-556) otherwise. `matrix` is the 16-float
/// premultiplied TRANSPOSED product exactly as the C wrapper builds it
/// (already transposed; applied without further transposition). `luts_in` /
/// `coeffs_in` are the source profile's `lut_in` / `unbounded_coeffs_in`
/// (`lutsize_in` floats per LUT, 3 floats per coefficient slice, gated on
/// `nonlinear_from`); `luts_out` / `coeffs_out` / `lutsize_out` are the
/// destination profile's `lut_out` / `unbounded_coeffs_out` (gated on
/// `nonlinear_to`). Linear sides may pass empty slices. Buffers must not
/// overlap; degenerate/short inputs are a no-op as documented above.
#[allow(clippy::too_many_arguments)]
pub fn matrix_rgb(
    image_in: &[f32],
    image_out: &mut [f32],
    matrix: &[f32],
    luts_in: [&[f32]; 3],
    coeffs_in: [&[f32]; 3],
    lutsize_in: usize,
    nonlinear_from: bool,
    luts_out: [&[f32]; 3],
    coeffs_out: [&[f32]; 3],
    lutsize_out: usize,
    nonlinear_to: bool,
    width: usize,
    height: usize,
) {
    let Some(npixels) = checked_matrix_rgb_len(
        image_in.len(),
        image_out.len(),
        matrix.len(),
        luts_in,
        coeffs_in,
        lutsize_in,
        nonlinear_from,
        luts_out,
        coeffs_out,
        lutsize_out,
        nonlinear_to,
        width,
        height,
    ) else {
        return;
    };
    let m = as_colormatrix(matrix);
    if nonlinear_from || nonlinear_to {
        for px in 0..npixels {
            let b = px * 4;
            let inp = [
                image_in[b],
                image_in[b + 1],
                image_in[b + 2],
                image_in[b + 3],
            ];
            // linearize (apply_trc passes linear-marked channels through and
            // carries lane 3; the matrix multiply below never reads lane 3,
            // so the C garbage-`rgb[3]` versus our `in[3]` is unobservable).
            let lin = if nonlinear_from {
                apply_trc(inp, luts_in, coeffs_in, lutsize_in)
            } else {
                inp
            };
            let temp = apply_transposed_color_matrix(&lin, m);
            if nonlinear_to {
                let dl = apply_trc(temp, luts_out, coeffs_out, lutsize_out);
                image_out[b] = dl[0];
                image_out[b + 1] = dl[1];
                image_out[b + 2] = dl[2];
                // lane 3 untouched (preserved), as in C.
            } else {
                image_out[b] = temp[0];
                image_out[b + 1] = temp[1];
                image_out[b + 2] = temp[2];
                image_out[b + 3] = temp[3];
            }
        }
    } else {
        for px in 0..npixels {
            let b = px * 4;
            let inp = [
                image_in[b],
                image_in[b + 1],
                image_in[b + 2],
                image_in[b + 3],
            ];
            let out4 = apply_transposed_color_matrix(&inp, m);
            image_out[b] = out4[0];
            image_out[b + 1] = out4[1];
            image_out[b + 2] = out4[2];
            image_out[b + 3] = out4[3];
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Inlined LUT sample so the RGB→RGB reference does not call the shared
/// helper path. Same ops as [`extrapolate_lut`], written out.
#[allow(dead_code)]
fn ref_lut_sample(lut: &[f32], v: f32, lutsize: usize) -> f32 {
    let ft = (v * (lutsize - 1) as f32).clamp(0.0, (lutsize - 1) as f32);
    let t = if (ft as usize) < lutsize - 2 {
        ft as usize
    } else {
        lutsize - 2
    };
    let f = ft - t as f32;
    lut[t] * (1.0 - f) + lut[t + 1] * f
}

/// Inlined exponential tail so the reference does not call [`eval_exp`].
/// Same ops: `coeff[1] * (x * coeff[0])^coeff[2]`.
#[allow(dead_code)]
fn ref_exp_sample(coeff: &[f32], x: f32) -> f32 {
    coeff[1] * (x * coeff[0]).powf(coeff[2])
}

/// Structurally divergent reference for [`matrix_rgb`]: a single unified
/// pixel loop (where the kernel mirrors C's two loop nests), per-pixel
/// scalar TRC staging through the inlined [`ref_lut_sample`] /
/// [`ref_exp_sample`] (where the kernel threads whole `[f32; 4]` arrays
/// through the shared [`apply_trc`]), and a lane-by-lane flat-index matrix
/// dot (`matrix[r]*v0 + matrix[4+r]*v1 + matrix[8+r]*v2`, i.e. `m[c][r]` at
/// flat `c*4+r`) where the kernel uses [`apply_transposed_color_matrix`].
/// Arithmetic association order is kept identical so the comparison is
/// bit-exact. Same no-op preconditions: well-formed buffers only.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ref_matrix_rgb(
    image_in: &[f32],
    image_out: &mut [f32],
    matrix: &[f32],
    luts_in: [&[f32]; 3],
    coeffs_in: [&[f32]; 3],
    lutsize_in: usize,
    nonlinear_from: bool,
    luts_out: [&[f32]; 3],
    coeffs_out: [&[f32]; 3],
    lutsize_out: usize,
    nonlinear_to: bool,
    width: usize,
    height: usize,
) {
    let Some(npixels) = checked_matrix_rgb_len(
        image_in.len(),
        image_out.len(),
        matrix.len(),
        luts_in,
        coeffs_in,
        lutsize_in,
        nonlinear_from,
        luts_out,
        coeffs_out,
        lutsize_out,
        nonlinear_to,
        width,
        height,
    ) else {
        return;
    };
    for px in 0..npixels {
        let b = px * 4;
        let mut rgb = [image_in[b], image_in[b + 1], image_in[b + 2]];
        if nonlinear_from {
            for c in 0..3 {
                if luts_in[c][0] >= 0.0 {
                    let v = rgb[c];
                    rgb[c] = if v < 1.0 {
                        ref_lut_sample(luts_in[c], v, lutsize_in)
                    } else {
                        ref_exp_sample(coeffs_in[c], v)
                    };
                }
            }
        }
        let mut tmp = [0.0f32; 4];
        for r in 0..4 {
            tmp[r] = matrix[r] * rgb[0] + matrix[4 + r] * rgb[1] + matrix[8 + r] * rgb[2];
        }
        if nonlinear_to {
            for c in 0..3 {
                image_out[b + c] = if luts_out[c][0] >= 0.0 {
                    if tmp[c] < 1.0 {
                        ref_lut_sample(luts_out[c], tmp[c], lutsize_out)
                    } else {
                        ref_exp_sample(coeffs_out[c], tmp[c])
                    }
                } else {
                    tmp[c]
                };
            }
        } else {
            image_out[b] = tmp[0];
            image_out[b + 1] = tmp[1];
            image_out[b + 2] = tmp[2];
            image_out[b + 3] = tmp[3];
        }
    }
}

// ── FFI export (RGB→RGB matrix transform, m4-189) ────────────────────────────

/// Materialize one side's TRC tables from raw pointers.
///
/// Returns `None` on any null pointer, `lutsize < 2`, `lutsize` above the C
/// `int` domain (negative values arrive as huge `size_t` and must not reach
/// slice construction), or the LUT/coefficient slices below. The caller only
/// invokes this for nonlinear sides (mirroring this wrapper's `nonlinearlut`
/// flags); linear sides bind empty tables and additionally tolerate null
/// pointers and any `lutsize`.
#[allow(clippy::type_complexity)]
unsafe fn trc_side<'a>(
    lut0: *const f32,
    lut1: *const f32,
    lut2: *const f32,
    coeff0: *const f32,
    coeff1: *const f32,
    coeff2: *const f32,
    lutsize: usize,
) -> Option<([&'a [f32]; 3], [&'a [f32]; 3])> {
    if lut0.is_null()
        || lut1.is_null()
        || lut2.is_null()
        || coeff0.is_null()
        || coeff1.is_null()
        || coeff2.is_null()
        || lutsize < 2
        || lutsize > i32::MAX as usize
    {
        return None;
    }
    Some((
        [
            std::slice::from_raw_parts(lut0, lutsize),
            std::slice::from_raw_parts(lut1, lutsize),
            std::slice::from_raw_parts(lut2, lutsize),
        ],
        [
            std::slice::from_raw_parts(coeff0, 3),
            std::slice::from_raw_parts(coeff1, 3),
            std::slice::from_raw_parts(coeff2, 3),
        ],
    ))
}

/// RGB→RGB via the premultiplied profile matrix (`_transform_matrix_rgb`
/// loops).
///
/// `image_in`/`image_out` each hold `4*width*height` floats (RGBA) and must
/// not overlap (matching the C `restrict` qualifiers; no in-place support).
/// `matrix` holds 16 floats: the C wrapper's premultiplied TRANSPOSED product
/// (`transpose_3xSSE(to->matrix_out * from->matrix_in)`) exactly as stored.
/// The `lut_in_*` / `coeff_in_*` triple is the source profile's `lut_in` /
/// `unbounded_coeffs_in` (`lutsize_in` floats per LUT — the source profile's
/// own `lutsize` — 3 floats per coefficient slice), used only when
/// `nonlinear_from != 0`; the `lut_out_*` / `coeff_out_*` triple is the
/// destination profile's `lut_out` / `unbounded_coeffs_out` (`lutsize_out`
/// floats per LUT), used only when `nonlinear_to != 0`. A LUT whose first
/// entry is negative marks a linear channel (passed through). Unused sides
/// tolerate null pointers and any `lutsize`. Output lane 3 follows the C
/// loops exactly: preserved when `nonlinear_to != 0`, otherwise the matrix
/// zero-padding product (0.0 for finite inputs, NaN for infinite ones).
/// Lane-3 behavior assumes the normal 4-channel vector build
/// (`DT_NO_VECTORIZATION` is commented out in `src/common/darktable.h`);
/// not a target otherwise.
///
/// # Safety
/// All pointers dereferenced on the taken paths must be valid for the stated
/// lengths; the C caller forwards the profiles' own tables over exactly
/// `4*width*height` floats.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn darkroom_iop_profile_matrix_rgb(
    image_in: *const f32,
    image_out: *mut f32,
    width: usize,
    height: usize,
    matrix: *const f32,
    lut_in_r: *const f32,
    lut_in_g: *const f32,
    lut_in_b: *const f32,
    coeff_in_r: *const f32,
    coeff_in_g: *const f32,
    coeff_in_b: *const f32,
    lut_out_r: *const f32,
    lut_out_g: *const f32,
    lut_out_b: *const f32,
    coeff_out_r: *const f32,
    coeff_out_g: *const f32,
    coeff_out_b: *const f32,
    lutsize_in: usize,
    lutsize_out: usize,
    nonlinear_from: i32,
    nonlinear_to: i32,
) {
    if image_in.is_null() || image_out.is_null() || matrix.is_null() {
        return;
    }
    // No in-place callers exist and the C marks these pointers `restrict`:
    // reject exact aliasing up front rather than building overlapping slices.
    if std::ptr::eq(image_in, image_out) {
        return;
    }
    if width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
    {
        return;
    }
    // with width, height <= i32::MAX the pixel product still needs checking
    // (4*width*height can overflow usize in theory); validate BEFORE
    // constructing the slices (the safe kernel re-checks defensively)
    let Some(npixels) = width.checked_mul(height) else {
        return;
    };
    let Some(pix_len) = npixels.checked_mul(4) else {
        return;
    };
    let nonlinear_from = nonlinear_from != 0;
    let nonlinear_to = nonlinear_to != 0;
    // TRC tables materialize only for nonlinear sides; linear sides bind
    // empty tables (never touched by the kernel) and skip pointer/lutsize
    // validation entirely.
    let empty: &[f32] = &[];
    let (luts_in, coeffs_in) = if nonlinear_from {
        let Some(t) = trc_side(
            lut_in_r, lut_in_g, lut_in_b, coeff_in_r, coeff_in_g, coeff_in_b, lutsize_in,
        ) else {
            return;
        };
        t
    } else {
        ([empty, empty, empty], [empty, empty, empty])
    };
    let (luts_out, coeffs_out) = if nonlinear_to {
        let Some(t) = trc_side(
            lut_out_r, lut_out_g, lut_out_b, coeff_out_r, coeff_out_g, coeff_out_b,
            lutsize_out,
        ) else {
            return;
        };
        t
    } else {
        ([empty, empty, empty], [empty, empty, empty])
    };
    let image_in = std::slice::from_raw_parts(image_in, pix_len);
    let image_out = std::slice::from_raw_parts_mut(image_out, pix_len);
    let matrix = std::slice::from_raw_parts(matrix, 16);
    matrix_rgb(
        image_in,
        image_out,
        matrix,
        luts_in,
        coeffs_in,
        lutsize_in,
        nonlinear_from,
        luts_out,
        coeffs_out,
        lutsize_out,
        nonlinear_to,
        width,
        height,
    )
}

// ── FFI export (tone curves, m4-187) ───────────────────────────────────────────

/// Apply the input-profile tone curves (`_apply_tonecurves`).
///
/// `image_in`/`image_out` each hold `4*width*height` floats (RGBA; only
/// channels 0..2 are written, alpha preserved). `lutr/lutg/lutb` each hold
/// `lutsize` floats; each `unbounded_coeffs*` holds 3 floats. A LUT whose
/// first entry is negative marks a linear channel (left untouched). May be
/// called in place (`image_in == image_out`, as
/// `_transform_lab_to_rgb_matrix` does); otherwise the buffers must not
/// overlap (matching the C `restrict` qualifiers).
///
/// # Safety
/// All pointers must be valid for the stated lengths; the C caller forwards
/// the profile's own `lutsize`-long LUTs over exactly `4*width*height`
/// floats.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn darkroom_apply_tonecurves(
    image_in: *const f32,
    image_out: *mut f32,
    width: usize,
    height: usize,
    lutr: *const f32,
    lutg: *const f32,
    lutb: *const f32,
    unbounded_coeffsr: *const f32,
    unbounded_coeffsg: *const f32,
    unbounded_coeffsb: *const f32,
    lutsize: usize,
) {
    if image_in.is_null()
        || image_out.is_null()
        || lutr.is_null()
        || lutg.is_null()
        || lutb.is_null()
        || unbounded_coeffsr.is_null()
        || unbounded_coeffsg.is_null()
        || unbounded_coeffsb.is_null()
        || width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
        // lutsize mirrors the C `int` (negative values arrive as huge
        // size_t); below 2 both the C loop and `extrapolate_lut` index out
        // of bounds, so reject before constructing the slices below (the
        // safe kernels re-check defensively)
        || lutsize < 2
        || lutsize > i32::MAX as usize
    {
        return;
    }
    // with width, height <= i32::MAX the pixel product still needs checking
    // (4*width*height can overflow usize in theory); validate BEFORE
    // constructing the slices (the safe kernels re-check defensively)
    let Some(npixels) = width.checked_mul(height) else {
        return;
    };
    let Some(pix_len) = npixels.checked_mul(4) else {
        return;
    };
    let lutr = std::slice::from_raw_parts(lutr, lutsize);
    let lutg = std::slice::from_raw_parts(lutg, lutsize);
    let lutb = std::slice::from_raw_parts(lutb, lutsize);
    let unbounded_coeffsr = std::slice::from_raw_parts(unbounded_coeffsr, 3);
    let unbounded_coeffsg = std::slice::from_raw_parts(unbounded_coeffsg, 3);
    let unbounded_coeffsb = std::slice::from_raw_parts(unbounded_coeffsb, 3);
    if std::ptr::eq(image_in, image_out) {
        // in-place call (`_transform_lab_to_rgb_matrix` passes image_out
        // twice): a single mutable slice avoids shared+mutable aliasing.
        let image = std::slice::from_raw_parts_mut(image_out, pix_len);
        apply_tonecurves_inplace(
            image,
            width,
            height,
            lutr,
            lutg,
            lutb,
            unbounded_coeffsr,
            unbounded_coeffsg,
            unbounded_coeffsb,
            lutsize,
        );
    } else {
        let image_in = std::slice::from_raw_parts(image_in, pix_len);
        let image_out = std::slice::from_raw_parts_mut(image_out, pix_len);
        apply_tonecurves(
            image_in,
            image_out,
            width,
            height,
            lutr,
            lutg,
            lutb,
            unbounded_coeffsr,
            unbounded_coeffsg,
            unbounded_coeffsb,
            lutsize,
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    /// Two-entry identity LUT: `extrapolate_lut([0,1], v) == v` exactly for
    /// `v in [0,1)` (`0*(1-v) + 1*v`), `lut[0]` for clamped negatives.
    fn identity_lut() -> Vec<f32> {
        vec![0.0, 1.0]
    }

    /// Identity exponential fit: `eval_exp([1,1,1], v) == v` exactly.
    fn identity_coeff() -> [f32; 3] {
        [1.0, 1.0, 1.0]
    }

    /// Linear-channel marker LUT: first entry negative, matching the C
    /// "omit luts marked as linear (negative as marker)" convention.
    fn linear_lut() -> Vec<f32> {
        vec![-1.0, 0.5]
    }

    #[test]
    fn all_active_identity_is_exact_and_preserves_alpha() {
        // 2x2 image, values in [0,1): identity LUT + identity exp fit maps
        // every RGB lane to itself bit-exactly; alpha prefilled with sentinel
        // must survive untouched.
        let (w, h) = (2usize, 2usize);
        let lr = identity_lut();
        let lg = identity_lut();
        let lb = identity_lut();
        let (cr, cg, cb) = (identity_coeff(), identity_coeff(), identity_coeff());
        let image_in: Vec<f32> = vec![
            0.0, 0.25, 0.5, 7.0, //
            0.75, 0.1, 0.9, 8.0, //
            0.33, 0.66, 0.99, 9.0, //
            0.125, 0.375, 0.625, 10.0,
        ];
        let mut image_out = vec![-99.0f32; 4 * w * h];
        apply_tonecurves(
            &image_in, &mut image_out, w, h, &lr, &lg, &lb, &cr, &cg, &cb, 2,
        );
        for px in 0..w * h {
            for c in 0..3 {
                assert_eq!(image_out[px * 4 + c], image_in[px * 4 + c]);
                assert_eq!(
                    image_out[px * 4 + c].to_bits(),
                    image_in[px * 4 + c].to_bits(),
                    "px={px} c={c}"
                );
            }
            // alpha preserved (was sentinel, never the input alpha)
            assert_eq!(image_out[px * 4 + 3].to_bits(), (-99.0f32).to_bits());
        }
    }

    #[test]
    fn mixed_sentinel_writes_only_active_channels() {
        // green marked linear: R and B map through identity, G and alpha
        // lanes keep their prefill sentinel (NOT copied from input).
        let (w, h) = (2usize, 1usize);
        let lr = identity_lut();
        let lg = linear_lut();
        let lb = identity_lut();
        let (cr, cg, cb) = (identity_coeff(), identity_coeff(), identity_coeff());
        let image_in: Vec<f32> = vec![0.2, 0.4, 0.6, 1.0, 0.8, 0.3, 0.1, 2.0];
        let mut image_out = vec![-77.0f32; 4 * w * h];
        apply_tonecurves(
            &image_in, &mut image_out, w, h, &lr, &lg, &lb, &cr, &cg, &cb, 2,
        );
        assert_eq!(image_out, vec![0.2, -77.0, 0.6, -77.0, 0.8, -77.0, 0.1, -77.0]);
        // single active channel (only blue): R/G untouched as well.
        let lr2 = linear_lut();
        let lg2 = linear_lut();
        let mut image_out2 = vec![-77.0f32; 4 * w * h];
        apply_tonecurves(
            &image_in, &mut image_out2, w, h, &lr2, &lg2, &lb, &cr, &cg, &cb, 2,
        );
        assert_eq!(
            image_out2,
            vec![-77.0, -77.0, 0.6, -77.0, -77.0, -77.0, 0.1, -77.0]
        );
    }

    #[test]
    fn all_linear_is_full_no_op() {
        // every LUT marked linear: neither C branch runs, output (including
        // alpha) keeps every prefill bit.
        let (w, h) = (3usize, 2usize);
        let l = linear_lut();
        let (cr, cg, cb) = (identity_coeff(), identity_coeff(), identity_coeff());
        let mut image_in = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut image_in, 0x1A9, 2.0);
        let mut image_out = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut image_out, 0x2B4, 3.0);
        let before = image_out.clone();
        apply_tonecurves(&image_in, &mut image_out, w, h, &l, &l, &l, &cr, &cg, &cb, 2);
        assert_eq!(image_out, before);
        // reference agrees on the no-op
        let mut ref_out = before.clone();
        ref_apply_tonecurves(&image_in, &mut ref_out, w, h, &l, &l, &l, &cr, &cg, &cb, 2);
        assert_eq!(ref_out, before);
    }

    #[test]
    fn above_one_takes_eval_exp_including_exactly_one() {
        // coeffs [1,2,0]: eval_exp = 2*(v)^0 = exactly 2.0 for every v
        // (powf(_, 0.0) == 1.0 is C99-mandated, unlike powf(x, 1.0)). v >= 1
        // — including exactly 1.0, which the C `< 1.0f` test sends to the
        // exp path — yields 2.0; v < 1 stays on the identity LUT.
        let (w, h) = (1usize, 4usize);
        let lr = identity_lut();
        let lg = identity_lut();
        let lb = identity_lut();
        let c2 = [1.0f32, 2.0, 0.0];
        let image_in: Vec<f32> = vec![
            1.0, 1.5, 0.5, 3.0, //
            2.0, 1.0, 1.25, 4.0, //
            0.0, 3.0, 0.999, 5.0, //
            10.0, 0.001, 1.0, 6.0,
        ];
        let mut image_out = vec![-1.0f32; 4 * w * h];
        apply_tonecurves(&image_in, &mut image_out, w, h, &lr, &lg, &lb, &c2, &c2, &c2, 2);
        assert_eq!(
            image_out,
            vec![
                2.0, 2.0, 0.5, -1.0, //
                2.0, 2.0, 2.0, -1.0, //
                0.0, 2.0, 0.999, -1.0, //
                2.0, 0.001, 2.0, -1.0,
            ]
        );
    }

    #[test]
    fn negative_inputs_clamp_to_lut_first_entry() {
        // extrapolate_lut clamps the position to [0, lutsize-1]: negatives
        // yield exactly lut[0]. Uses a non-identity LUT so the clamp value
        // (0.25) is distinguishable from the input.
        let lut = vec![0.25f32, 0.75];
        let c = identity_coeff();
        let image_in: Vec<f32> = vec![-0.5, -3.0, -0.0, 9.0, -100.0, 0.5, -1.0, 8.0];
        let mut image_out = vec![-1.0f32; 8];
        apply_tonecurves(&image_in, &mut image_out, 2, 1, &lut, &lut, &lut, &c, &c, &c, 2);
        // -0.0 clamps to lut[0] as well; 0.5 on [0.25,0.75] lerps to 0.5.
        assert_eq!(image_out, vec![0.25, 0.25, 0.25, -1.0, 0.25, 0.5, 0.25, -1.0]);
    }

    #[test]
    fn matches_reference_over_lcg() {
        // all-active, each-single-active, and mixed-sentinel LUT sets over
        // pseudo-random inputs spanning negatives, [0,1), 1.0 and above —
        // exact and to_bits equality against the divergent reference.
        let luts_all: Vec<Vec<f32>> = vec![
            vec![0.0, 0.2, 0.5, 0.9, 1.0],
            vec![0.1, 0.3, 0.4, 0.8, 1.0],
            vec![0.0, 0.0, 0.6, 0.7, 1.0],
        ];
        let coeff_sets: Vec<[f32; 3]> = vec![[1.0, 1.2, 0.9], [0.8, 1.0, 1.1], [1.1, 0.9, 1.0]];
        // sentinel variants: bit patterns for which channels are linear
        let sentinel_masks: Vec<[bool; 3]> = vec![
            [true, true, true],
            [true, false, true],
            [false, true, false],
            [true, false, false],
            [false, false, true],
        ];
        for mask in &sentinel_masks {
            let lr;
            let lg;
            let lb;
            // build per-mask LUT views without reallocating the active ones
            let lin = linear_lut();
            lr = if mask[0] { &luts_all[0] } else { &lin };
            lg = if mask[1] { &luts_all[1] } else { &lin };
            lb = if mask[2] { &luts_all[2] } else { &lin };
            for &(w, h) in &[(1usize, 1usize), (2, 1), (3, 2), (5, 4), (8, 7)] {
                let mut image_in = vec![0.0f32; 4 * w * h];
                // LCG in [0,2) shifted to [-0.5,1.5): covers clamp, LUT
                // and exp regions, including values near 1.0.
                lcg_fill(&mut image_in, 0x70C0 + w as u32 * 17 + h as u32, 2.0);
                for v in image_in.iter_mut().step_by(4) {
                    *v -= 0.5;
                }
                // force exact-boundary values into the stream
                if !image_in.is_empty() {
                    image_in[0] = 1.0;
                    if image_in.len() > 4 {
                        image_in[4] = -0.0;
                    }
                    if image_in.len() > 8 {
                        image_in[8] = 0.999_999_9;
                    }
                }
                let mut direct = vec![-33.0f32; 4 * w * h];
                let mut refr = vec![-33.0f32; 4 * w * h];
                apply_tonecurves(
                    &image_in, &mut direct, w, h, lr, lg, lb, &coeff_sets[0],
                    &coeff_sets[1], &coeff_sets[2], 5,
                );
                ref_apply_tonecurves(
                    &image_in, &mut refr, w, h, lr, lg, lb, &coeff_sets[0],
                    &coeff_sets[1], &coeff_sets[2], 5,
                );
                assert_eq!(direct, refr, "mask={mask:?} w={w} h={h}");
                for (d, r) in direct.iter().zip(refr.iter()) {
                    assert_eq!(d.to_bits(), r.to_bits(), "mask={mask:?} w={w} h={h}");
                }
                // alpha lanes never written in either implementation
                for px in 0..w * h {
                    assert_eq!(direct[px * 4 + 3].to_bits(), (-33.0f32).to_bits());
                }
            }
        }
    }

    #[test]
    fn inplace_matches_split() {
        // the in-place kernel must equal the split kernel with a copied
        // input, for both all-active and mixed-sentinel LUT sets.
        let luts_all: Vec<Vec<f32>> = vec![
            vec![0.0, 0.2, 0.5, 0.9, 1.0],
            vec![0.1, 0.3, 0.4, 0.8, 1.0],
            vec![0.0, 0.0, 0.6, 0.7, 1.0],
        ];
        let lin = linear_lut();
        let coeff_sets: Vec<[f32; 3]> = vec![[1.0, 1.2, 0.9], [0.8, 1.0, 1.1], [1.1, 0.9, 1.0]];
        let cases: Vec<(&[f32], &[f32], &[f32])> = vec![
            (&luts_all[0], &luts_all[1], &luts_all[2]),
            (&luts_all[0], &lin, &luts_all[2]),
            (&lin, &lin, &lin),
        ];
        for (lr, lg, lb) in cases {
            for &(w, h) in &[(1usize, 1usize), (4, 3), (7, 5)] {
                let mut base = vec![0.0f32; 4 * w * h];
                lcg_fill(&mut base, 0x10C4, 2.0);
                let mut split_out = vec![-5.0f32; 4 * w * h];
                // split kernel needs a defined output for untouched lanes:
                // seed with the input (what the in-place caller would hold).
                split_out.copy_from_slice(&base);
                apply_tonecurves(
                    &base, &mut split_out, w, h, lr, lg, lb, &coeff_sets[0],
                    &coeff_sets[1], &coeff_sets[2], 5,
                );
                let mut inplace = base.clone();
                apply_tonecurves_inplace(
                    &mut inplace, w, h, lr, lg, lb, &coeff_sets[0], &coeff_sets[1],
                    &coeff_sets[2], 5,
                );
                assert_eq!(inplace, split_out, "w={w} h={h}");
                for (a, b) in inplace.iter().zip(split_out.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "w={w} h={h}");
                }
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let lr = identity_lut();
        let c = identity_coeff();
        // zero dims leave buffers untouched
        for &(w, h) in &[(0usize, 4usize), (4usize, 0usize), (0usize, 0usize)] {
            let inn: Vec<f32> = vec![0.5; 64];
            let mut out = vec![9.0f32; 64];
            apply_tonecurves(&inn, &mut out, w, h, &lr, &lr, &lr, &c, &c, &c, 2);
            assert!(out.iter().all(|&v| v == 9.0), "w={w} h={h}");
            let mut io = vec![9.0f32; 64];
            apply_tonecurves_inplace(&mut io, w, h, &lr, &lr, &lr, &c, &c, &c, 2);
            assert!(io.iter().all(|&v| v == 9.0), "inplace w={w} h={h}");
        }
        // lutsize < 2 (both the C loop and extrapolate_lut index out of
        // bounds there) is a no-op, no panic
        for &lutsize in &[0usize, 1usize] {
            let inn: Vec<f32> = vec![0.5; 16];
            let mut out = vec![9.0f32; 16];
            apply_tonecurves(&inn, &mut out, 2, 2, &lr, &lr, &lr, &c, &c, &c, lutsize);
            assert_eq!(out, vec![9.0f32; 16], "lutsize={lutsize}");
        }
        // short buffers: length preconditions fail -> no-op, no panic
        let (w, h) = (4usize, 4usize);
        let mut inn = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut inn, 0x5E07, 2.0);
        let mut out = vec![9.0f32; 4 * w * h];
        apply_tonecurves(&inn[..10], &mut out, w, h, &lr, &lr, &lr, &c, &c, &c, 2);
        apply_tonecurves(&inn, &mut out[..10], w, h, &lr, &lr, &lr, &c, &c, &c, 2);
        apply_tonecurves(&inn, &mut out, w, h, &lr[..1], &lr, &lr, &c, &c, &c, 2);
        apply_tonecurves(&inn, &mut out, w, h, &lr, &lr, &lr, &c[..2], &c, &c, 2);
        assert!(out.iter().all(|&v| v == 9.0));
        // boundary dimension arithmetic must no-op without panicking
        apply_tonecurves(&[], &mut [], usize::MAX, 1, &[], &[], &[], &[], &[], &[], 2);
        ref_apply_tonecurves(&[], &mut [], usize::MAX, 1, &[], &[], &[], &[], &[], &[], 2);
        apply_tonecurves_inplace(&mut [], usize::MAX, 1, &[], &[], &[], &[], &[], &[], 2);
    }

    #[test]
    fn ffi_round_trip() {
        // split and in-place FFI calls agree with the safe kernels.
        let luts_all: Vec<Vec<f32>> = vec![
            vec![0.0, 0.2, 0.5, 0.9, 1.0],
            vec![0.1, 0.3, 0.4, 0.8, 1.0],
            vec![0.0, 0.0, 0.6, 0.7, 1.0],
        ];
        let coeff_sets: Vec<[f32; 3]> = vec![[1.0, 1.2, 0.9], [0.8, 1.0, 1.1], [1.1, 0.9, 1.0]];
        for &(w, h) in &[(4usize, 2usize), (5usize, 3usize)] {
            let mut image_in = vec![0.0f32; 4 * w * h];
            lcg_fill(&mut image_in, 0xFF11 + w as u32, 2.0);
            let mut ffi_out = vec![-3.0f32; 4 * w * h];
            let mut direct = vec![-3.0f32; 4 * w * h];
            unsafe {
                darkroom_apply_tonecurves(
                    image_in.as_ptr(),
                    ffi_out.as_mut_ptr(),
                    w,
                    h,
                    luts_all[0].as_ptr(),
                    luts_all[1].as_ptr(),
                    luts_all[2].as_ptr(),
                    coeff_sets[0].as_ptr(),
                    coeff_sets[1].as_ptr(),
                    coeff_sets[2].as_ptr(),
                    5,
                );
            }
            apply_tonecurves(
                &image_in, &mut direct, w, h, &luts_all[0], &luts_all[1], &luts_all[2],
                &coeff_sets[0], &coeff_sets[1], &coeff_sets[2], 5,
            );
            assert_eq!(ffi_out, direct, "w={w} h={h}");
            // in-place FFI (same pointer twice, like
            // _transform_lab_to_rgb_matrix) matches the inplace kernel.
            let mut ffi_io = image_in.clone();
            let mut direct_io = image_in.clone();
            unsafe {
                darkroom_apply_tonecurves(
                    ffi_io.as_ptr(),
                    ffi_io.as_mut_ptr(),
                    w,
                    h,
                    luts_all[0].as_ptr(),
                    luts_all[1].as_ptr(),
                    luts_all[2].as_ptr(),
                    coeff_sets[0].as_ptr(),
                    coeff_sets[1].as_ptr(),
                    coeff_sets[2].as_ptr(),
                    5,
                );
            }
            apply_tonecurves_inplace(
                &mut direct_io, w, h, &luts_all[0], &luts_all[1], &luts_all[2],
                &coeff_sets[0], &coeff_sets[1], &coeff_sets[2], 5,
            );
            assert_eq!(ffi_io, direct_io, "inplace w={w} h={h}");
        }
    }

    #[test]
    fn ffi_guards() {
        let lr = identity_lut();
        let c = identity_coeff();
        let inn: Vec<f32> = vec![0.5; 64];
        let mut out = vec![7.0f32; 64];
        unsafe {
            // null pointers (each of the eight, one at a time)
            darkroom_apply_tonecurves(
                std::ptr::null(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), std::ptr::null_mut(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, std::ptr::null(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), std::ptr::null(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                std::ptr::null(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), std::ptr::null(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), std::ptr::null(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), std::ptr::null(), 2,
            );
            // zero dims and degenerate lutsize
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 0, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 0, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 1,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 0,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), (i32::MAX as usize) + 1,
            );
            // i32::MAX caps (C ints arrive non-negative; negatives would
            // wrap to huge size_t and must not reach slice construction)
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), (i32::MAX as usize) + 1, 4, lr.as_ptr(),
                lr.as_ptr(), lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, (i32::MAX as usize) + 1, lr.as_ptr(),
                lr.as_ptr(), lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
        }
        assert!(out.iter().all(|&v| v == 7.0)); // untouched
    }

    // ── m4-188 matrix-transform tests ─────────────────────────────────────────

    /// 16-float identity 3x3 embedded in a 4x4 (flat row-major, as stored):
    /// `out[r] = in[r]` for `r = 0..2` (and `out[3] = 0` from the multiply).
    fn identity_matrix() -> Vec<f32> {
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// Cyclic lane permute (flat row-major of the *transposed* matrix):
    /// `out[0] = in[1]`, `out[1] = in[2]`, `out[2] = in[0]`. A naive
    /// non-transposed application (`out[r] = m[r*4+0]*in0 + ...`) would give
    /// the inverse rotation instead, so this distinguishes the order.
    fn permute_matrix() -> Vec<f32> {
        vec![
            0.0, 0.0, 1.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// Non-trivial realistic matrix (sRGB→XYZ D50 coefficients, flat
    /// row-major with padding) for reference-agreement fuzzing.
    fn srgb_like_matrix() -> Vec<f32> {
        vec![
            0.4360747, 0.2225045, 0.0139322, 0.0, //
            0.3850649, 0.7168786, 0.0971045, 0.0, //
            0.1430804, 0.0606169, 0.7141733, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    #[test]
    fn matrix_order_is_transposed() {
        use crate::color::{apply_transposed_color_matrix, lab_to_xyz, xyz_to_lab};
        // RGB→Lab: permute-then-convert must equal hand-staged
        // permute (flat column-dot) + shared xyz_to_lab with alpha forced 0.
        let m = permute_matrix();
        let rgb = [0.1f32, 0.5, 0.9, 0.7];
        let image_in = rgb.to_vec();
        let mut out = vec![-1.0f32; 4];
        rgb_to_lab_matrix(&image_in, &mut out, &m, 1, 1);
        // hand-staged: xyz = [in1, in2, in0] (column dots of the flat rows).
        let xyz = [rgb[1], rgb[2], rgb[0], 0.0];
        let mut expected = xyz_to_lab(xyz);
        expected[3] = 0.0;
        assert_eq!(out, expected);
        // sanity: the wrong (non-transposed) order gives the inverse rotation
        // [in2, in0, in1] — the test above is not vacuous.
        let wrong_xyz = [rgb[2], rgb[0], rgb[1], 0.0];
        assert_ne!(xyz_to_lab(xyz), xyz_to_lab(wrong_xyz));
        // cross-check against the [[f32;4];4] helper path too.
        let marr: [[f32; 4]; 4] = [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let xyz2 = apply_transposed_color_matrix(&rgb, &marr);
        assert_eq!(xyz2, xyz);

        // Lab→RGB: hand-staged lab_to_xyz + permute, alpha preserved.
        let lab = [50.0f32, 20.0, -30.0, 0.42];
        let image_lab = lab.to_vec();
        let mut rgb_out = vec![-1.0f32; 4];
        lab_to_rgb_matrix(&image_lab, &mut rgb_out, &m, 1, 1);
        let xyz = lab_to_xyz(lab);
        assert_eq!(rgb_out, vec![xyz[1], xyz[2], xyz[0], 0.42]);
        // wrong order would give [xyz2, xyz0, xyz1] instead.
        assert_ne!((rgb_out[0], rgb_out[1], rgb_out[2]), (xyz[2], xyz[0], xyz[1]));
    }

    #[test]
    fn xyz_lab_round_trip_through_kernels() {
        // RGB→Lab (diag(2,3,0.5)) then Lab→RGB (inverse diag) recovers the
        // input up to f32 cbrt/pow round-trip error. Alpha goes in as 0.0
        // (RGB→Lab forces it) so the return leg restores 0.0 as well.
        let fwd = vec![
            2.0, 0.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, //
            0.0, 0.0, 0.5, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let inv = vec![
            0.5, 0.0, 0.0, 0.0, //
            0.0, 1.0 / 3.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let (w, h) = (3usize, 2usize);
        let mut rgb = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut rgb, 0xBE57, 1.5);
        for v in rgb.iter_mut().skip(3).step_by(4) {
            *v = 0.0; // input alpha irrelevant: RGB→Lab forces 0.0
        }
        // keep XYZ safely positive and away from the epsilon kink
        for v in rgb.iter_mut().step_by(4) {
            *v = 0.05 + (*v % 1.0).abs() * 0.5;
        }
        let mut lab = vec![0.0f32; 4 * w * h];
        rgb_to_lab_matrix(&rgb, &mut lab, &fwd, w, h);
        let mut back = vec![0.0f32; 4 * w * h];
        lab_to_rgb_matrix(&lab, &mut back, &inv, w, h);
        for px in 0..w * h {
            for c in 0..3 {
                let (a, b) = (rgb[px * 4 + c], back[px * 4 + c]);
                assert!(
                    (a - b).abs() <= 1e-4 * a.abs().max(1.0),
                    "px={px} c={c}: {a} vs {b}"
                );
            }
            assert_eq!(back[px * 4 + 3].to_bits(), 0.0f32.to_bits());
        }
    }

    #[test]
    fn alpha_semantics() {
        let m = identity_matrix();
        // RGB→Lab forces alpha to exactly 0.0 whatever the input held.
        for &alpha in &[0.0f32, 0.7, 1.0, 123.5] {
            let inn = vec![0.4f32, 0.2, 0.6, alpha];
            let mut out = vec![-1.0f32; 4];
            rgb_to_lab_matrix(&inn, &mut out, &m, 1, 1);
            assert_eq!(out[3].to_bits(), 0.0f32.to_bits(), "alpha={alpha}");
            let mut io = inn.clone();
            rgb_to_lab_matrix_inplace(&mut io, &m, 1, 1);
            assert_eq!(io, out, "alpha={alpha}");
            assert_eq!(io[3].to_bits(), 0.0f32.to_bits());
        }
        // Lab→RGB restores the input alpha bit-exactly.
        for &alpha in &[0.0f32, 1.0, 0.33, -2.5] {
            let inn = vec![50.0f32, 10.0, -10.0, alpha];
            let mut out = vec![-1.0f32; 4];
            lab_to_rgb_matrix(&inn, &mut out, &m, 1, 1);
            assert_eq!(out[3].to_bits(), alpha.to_bits(), "alpha={alpha}");
            let mut io = inn.clone();
            lab_to_rgb_matrix_inplace(&mut io, &m, 1, 1);
            assert_eq!(io, out, "alpha={alpha}");
        }
    }

    #[test]
    fn matrix_matches_reference_over_lcg() {
        // kernel vs structurally divergent reference: bit-exact (to_bits)
        // over pseudo-random RGB and Lab inputs, several image shapes.
        let matrices = [identity_matrix(), permute_matrix(), srgb_like_matrix()];
        for m in &matrices {
            for &(w, h) in &[(1usize, 1usize), (2, 1), (3, 2), (5, 4), (8, 7)] {
                // RGB inputs in [0, 1.5): covers the Lab epsilon kink region
                // after the sRGB-like matrix as well as plain positives.
                let mut rgb = vec![0.0f32; 4 * w * h];
                lcg_fill(&mut rgb, 0x1880 + w as u32 * 17 + h as u32, 1.5);
                let mut direct = vec![-33.0f32; 4 * w * h];
                let mut refr = vec![-33.0f32; 4 * w * h];
                rgb_to_lab_matrix(&rgb, &mut direct, m, w, h);
                ref_rgb_to_lab_matrix(&rgb, &mut refr, m, w, h);
                assert_eq!(direct, refr, "rgb2lab w={w} h={h}");
                for (d, r) in direct.iter().zip(refr.iter()) {
                    assert_eq!(d.to_bits(), r.to_bits(), "rgb2lab w={w} h={h}");
                }
                // Lab inputs: L in [0,100), a/b shifted to [-50,50).
                let mut lab = vec![0.0f32; 4 * w * h];
                lcg_fill(&mut lab, 0x1881 + w as u32 * 31 + h as u32, 100.0);
                for px in 0..w * h {
                    lab[px * 4 + 1] -= 50.0;
                    lab[px * 4 + 2] -= 50.0;
                }
                let mut direct = vec![-33.0f32; 4 * w * h];
                let mut refr = vec![-33.0f32; 4 * w * h];
                lab_to_rgb_matrix(&lab, &mut direct, m, w, h);
                ref_lab_to_rgb_matrix(&lab, &mut refr, m, w, h);
                assert_eq!(direct, refr, "lab2rgb w={w} h={h}");
                for (d, r) in direct.iter().zip(refr.iter()) {
                    assert_eq!(d.to_bits(), r.to_bits(), "lab2rgb w={w} h={h}");
                }
            }
        }
    }

    #[test]
    fn matrix_inplace_matches_split() {
        // in-place kernels equal the split kernels with a copied input, for
        // identity, permute and realistic matrices.
        let matrices = [identity_matrix(), permute_matrix(), srgb_like_matrix()];
        for m in &matrices {
            for &(w, h) in &[(1usize, 1usize), (4, 3), (7, 5)] {
                let mut rgb = vec![0.0f32; 4 * w * h];
                lcg_fill(&mut rgb, 0x1882, 1.5);
                let mut split = vec![0.0f32; 4 * w * h];
                rgb_to_lab_matrix(&rgb, &mut split, m, w, h);
                let mut io = rgb.clone();
                rgb_to_lab_matrix_inplace(&mut io, m, w, h);
                assert_eq!(io, split, "rgb2lab w={w} h={h}");

                let mut lab = vec![0.0f32; 4 * w * h];
                lcg_fill(&mut lab, 0x1883, 100.0);
                for px in 0..w * h {
                    lab[px * 4 + 1] -= 50.0;
                    lab[px * 4 + 2] -= 50.0;
                }
                let mut split = vec![0.0f32; 4 * w * h];
                lab_to_rgb_matrix(&lab, &mut split, m, w, h);
                let mut io = lab.clone();
                lab_to_rgb_matrix_inplace(&mut io, m, w, h);
                assert_eq!(io, split, "lab2rgb w={w} h={h}");
            }
        }
    }

    #[test]
    fn matrix_degenerate_guards_no_op() {
        let m = identity_matrix();
        let short_m = vec![1.0f32; 15];
        // zero dims leave buffers untouched (all four kernels + references)
        for &(w, h) in &[(0usize, 4usize), (4usize, 0usize), (0usize, 0usize)] {
            let inn: Vec<f32> = vec![0.5; 64];
            let mut out = vec![9.0f32; 64];
            rgb_to_lab_matrix(&inn, &mut out, &m, w, h);
            lab_to_rgb_matrix(&inn, &mut out, &m, w, h);
            assert!(out.iter().all(|&v| v == 9.0), "w={w} h={h}");
            let mut io = vec![9.0f32; 64];
            rgb_to_lab_matrix_inplace(&mut io, &m, w, h);
            lab_to_rgb_matrix_inplace(&mut io, &m, w, h);
            assert!(io.iter().all(|&v| v == 9.0), "inplace w={w} h={h}");
            let mut ref_out = vec![9.0f32; 64];
            ref_rgb_to_lab_matrix(&inn, &mut ref_out, &m, w, h);
            ref_lab_to_rgb_matrix(&inn, &mut ref_out, &m, w, h);
            assert!(ref_out.iter().all(|&v| v == 9.0), "ref w={w} h={h}");
        }
        // short buffers / short matrix: no-op, no panic
        let (w, h) = (4usize, 4usize);
        let mut inn = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut inn, 0x5E88, 1.5);
        let mut out = vec![9.0f32; 4 * w * h];
        rgb_to_lab_matrix(&inn[..10], &mut out, &m, w, h);
        rgb_to_lab_matrix(&inn, &mut out[..10], &m, w, h);
        lab_to_rgb_matrix(&inn[..10], &mut out, &m, w, h);
        lab_to_rgb_matrix(&inn, &mut out[..10], &m, w, h);
        rgb_to_lab_matrix(&inn, &mut out, &short_m, w, h);
        lab_to_rgb_matrix(&inn, &mut out, &short_m, w, h);
        rgb_to_lab_matrix(&inn, &mut out, &[], w, h);
        assert!(out.iter().all(|&v| v == 9.0));
        let mut io = vec![9.0f32; 10];
        rgb_to_lab_matrix_inplace(&mut io, &m, w, h);
        lab_to_rgb_matrix_inplace(&mut io, &m, w, h);
        assert!(io.iter().all(|&v| v == 9.0));
        // boundary dimension arithmetic must no-op without panicking
        rgb_to_lab_matrix(&[], &mut [], &m, usize::MAX, 1);
        lab_to_rgb_matrix(&[], &mut [], &m, usize::MAX, 1);
        ref_rgb_to_lab_matrix(&[], &mut [], &m, usize::MAX, 1);
        ref_lab_to_rgb_matrix(&[], &mut [], &m, usize::MAX, 1);
        rgb_to_lab_matrix_inplace(&mut [], &m, usize::MAX, 1);
        lab_to_rgb_matrix_inplace(&mut [], &m, usize::MAX, 1);
    }

    #[test]
    fn matrix_ffi_round_trip() {
        // split and in-place FFI calls agree with the safe kernels, both ways.
        let matrices = [identity_matrix(), permute_matrix(), srgb_like_matrix()];
        for m in &matrices {
            for &(w, h) in &[(4usize, 2usize), (5usize, 3usize)] {
                let mut rgb = vec![0.0f32; 4 * w * h];
                lcg_fill(&mut rgb, 0xFF88 + w as u32, 1.5);
                let mut ffi_out = vec![-3.0f32; 4 * w * h];
                let mut direct = vec![-3.0f32; 4 * w * h];
                unsafe {
                    darkroom_iop_profile_rgb_to_lab_matrix(
                        rgb.as_ptr(), ffi_out.as_mut_ptr(), w, h, m.as_ptr(),
                    );
                }
                rgb_to_lab_matrix(&rgb, &mut direct, m, w, h);
                assert_eq!(ffi_out, direct, "rgb2lab w={w} h={h}");
                // in-place FFI (same pointer twice, like the nonlinear
                // RGB→Lab branch) matches the inplace kernel.
                let mut ffi_io = rgb.clone();
                let mut direct_io = rgb.clone();
                unsafe {
                    darkroom_iop_profile_rgb_to_lab_matrix(
                        ffi_io.as_ptr(), ffi_io.as_mut_ptr(), w, h, m.as_ptr(),
                    );
                }
                rgb_to_lab_matrix_inplace(&mut direct_io, m, w, h);
                assert_eq!(ffi_io, direct_io, "rgb2lab inplace w={w} h={h}");

                let mut lab = vec![0.0f32; 4 * w * h];
                lcg_fill(&mut lab, 0xFF89 + w as u32, 100.0);
                for px in 0..w * h {
                    lab[px * 4 + 1] -= 50.0;
                    lab[px * 4 + 2] -= 50.0;
                }
                let mut ffi_out = vec![-3.0f32; 4 * w * h];
                let mut direct = vec![-3.0f32; 4 * w * h];
                unsafe {
                    darkroom_iop_profile_lab_to_rgb_matrix(
                        lab.as_ptr(), ffi_out.as_mut_ptr(), w, h, m.as_ptr(),
                    );
                }
                lab_to_rgb_matrix(&lab, &mut direct, m, w, h);
                assert_eq!(ffi_out, direct, "lab2rgb w={w} h={h}");
                let mut ffi_io = lab.clone();
                let mut direct_io = lab.clone();
                unsafe {
                    darkroom_iop_profile_lab_to_rgb_matrix(
                        ffi_io.as_ptr(), ffi_io.as_mut_ptr(), w, h, m.as_ptr(),
                    );
                }
                lab_to_rgb_matrix_inplace(&mut direct_io, m, w, h);
                assert_eq!(ffi_io, direct_io, "lab2rgb inplace w={w} h={h}");
            }
        }
    }

    #[test]
    fn matrix_ffi_guards() {
        let m = identity_matrix();
        let inn: Vec<f32> = vec![0.5; 64];
        let mut out = vec![7.0f32; 64];
        unsafe {
            for export in [
                darkroom_iop_profile_rgb_to_lab_matrix,
                darkroom_iop_profile_lab_to_rgb_matrix,
            ] {
                // null pointers (each of the three, one at a time)
                export(std::ptr::null(), out.as_mut_ptr(), 4, 4, m.as_ptr());
                export(inn.as_ptr(), std::ptr::null_mut(), 4, 4, m.as_ptr());
                export(inn.as_ptr(), out.as_mut_ptr(), 4, 4, std::ptr::null());
                // zero dims
                export(inn.as_ptr(), out.as_mut_ptr(), 0, 4, m.as_ptr());
                export(inn.as_ptr(), out.as_mut_ptr(), 4, 0, m.as_ptr());
                // i32::MAX caps (C ints arrive non-negative; negatives would
                // wrap to huge size_t and must not reach slice construction)
                export(
                    inn.as_ptr(),
                    out.as_mut_ptr(),
                    (i32::MAX as usize) + 1,
                    4,
                    m.as_ptr(),
                );
                export(
                    inn.as_ptr(),
                    out.as_mut_ptr(),
                    4,
                    (i32::MAX as usize) + 1,
                    m.as_ptr(),
                );
            }
        }
        assert!(out.iter().all(|&v| v == 7.0)); // untouched
    }

    // ── m4-189 RGB→RGB matrix-transform tests ────────────────────────────────

    /// 2-entry LUT mapping `v` in `[0,1)` to `0.5 + 0.5*v` (lerp between the
    /// entries): `0.5 -> 0.75`, `0.25 -> 0.625`, `0.75 -> 0.875` — all dyadic
    /// and hence bit-exact — distinguishing mapped channels from passthrough
    /// ones. `lut[0]` is non-negative, so the channel is active.
    fn scale_lut() -> Vec<f32> {
        vec![0.5, 1.0]
    }

    /// Constant-2 exponential fit: `eval_exp([1,2,0], v) == 2.0` for every
    /// `v` (`powf(_, 0.0) == 1.0` is C99-mandated), pinning the `v >= 1.0`
    /// tail including exactly 1.0.
    fn const2_coeff() -> [f32; 3] {
        [1.0, 2.0, 0.0]
    }

    #[test]
    fn matrix_rgb_flag_paths_and_alpha() {
        // identity matrix + scale LUTs + const-2 tails: every (nf, nt) combo
        // is bit-exact by hand computation, pinning which lanes the C loops
        // write. Input lanes: 0.5 (LUT region), 1.5 (exp region -> 2.0),
        // 0.25 (LUT region); output prefilled with -7.0 to detect untouched
        // lanes (preserved alpha shows the PREFILL, not the input alpha).
        let m = identity_matrix();
        let sc = scale_lut();
        let c2 = const2_coeff();
        let image_in = vec![0.5f32, 1.5, 0.25, 0.9];
        // linearize(0.5)=0.75, linearize(1.5)=2.0, linearize(0.25)=0.625;
        // delinearize(0.75)=0.875, delinearize(2.0)=2.0, delinearize(0.625)=0.8125.
        let cases: [((bool, bool), [f32; 4]); 4] = [
            ((false, false), [0.5, 1.5, 0.25, 0.0]),
            ((true, false), [0.75, 2.0, 0.625, 0.0]),
            ((false, true), [0.75, 2.0, 0.625, -7.0]),
            ((true, true), [0.875, 2.0, 0.8125, -7.0]),
        ];
        for ((nf, nt), expected) in cases {
            let mut out = vec![-7.0f32; 4];
            matrix_rgb(
                &image_in, &mut out, &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, nf,
                [&sc, &sc, &sc], [&c2, &c2, &c2], 2, nt, 1, 1,
            );
            assert_eq!(out, expected.to_vec(), "nf={nf} nt={nt}");
            for (o, e) in out.iter().zip(expected.iter()) {
                assert_eq!(o.to_bits(), e.to_bits(), "nf={nf} nt={nt}");
            }
            // the divergent reference agrees bit-exactly on every path
            let mut refr = vec![-7.0f32; 4];
            ref_matrix_rgb(
                &image_in, &mut refr, &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, nf,
                [&sc, &sc, &sc], [&c2, &c2, &c2], 2, nt, 1, 1,
            );
            assert_eq!(refr, expected.to_vec(), "ref nf={nf} nt={nt}");
        }
    }

    #[test]
    fn matrix_rgb_linear_alpha_zero_product() {
        // linear path writes lane 3 through the matrix (zero padding), not by
        // preserving the input alpha: permute matrix moves lanes 0..2 and
        // zeroes lane 3 whatever it held.
        let m = permute_matrix();
        let empty: Vec<f32> = vec![];
        let e = empty.as_slice();
        for &alpha in &[0.0f32, 0.7, 1.0, -3.25] {
            let image_in = vec![0.1f32, 0.5, 0.9, alpha];
            let mut out = vec![-1.0f32; 4];
            matrix_rgb(
                &image_in, &mut out, &m, [e, e, e], [e, e, e], 0, false,
                [e, e, e], [e, e, e], 0, false, 1, 1,
            );
            assert_eq!(out, vec![0.5, 0.9, 0.1, 0.0], "alpha={alpha}");
            assert_eq!(out[3].to_bits(), 0.0f32.to_bits());
        }
        // infinite RGB input: 0.0 * inf is NaN, and the C loop (like the
        // shared helper) computes lane 3 rather than forcing 0.0.
        let image_in = vec![f32::INFINITY, 0.5, 0.25, 1.0];
        let mut out = vec![-1.0f32; 4];
        matrix_rgb(
            &image_in, &mut out, &m, [e, e, e], [e, e, e], 0, false,
            [e, e, e], [e, e, e], 0, false, 1, 1,
        );
        assert!(out[3].is_nan(), "out={out:?}");
    }

    #[test]
    fn matrix_rgb_linear_nan_input_propagates() {
        // Linear path, identity matrix: NaN in lane 0 contaminates every lane
        // through the multiply (0.0 * NaN is NaN); nothing on this path forces
        // lanes, unlike the nonlinear alpha-preserve path.
        let m = identity_matrix();
        let empty: Vec<f32> = vec![];
        let e = empty.as_slice();
        let image_in = vec![f32::NAN, 0.5, 0.25, 1.0];
        let mut out = vec![-1.0f32; 4];
        matrix_rgb(
            &image_in, &mut out, &m, [e, e, e], [e, e, e], 0, false,
            [e, e, e], [e, e, e], 0, false, 1, 1,
        );
        for c in 0..4 {
            assert!(out[c].is_nan(), "lane {c} must propagate NaN: {out:?}");
        }
    }

    #[test]
    fn matrix_rgb_sentinel_passthrough() {
        // green TRCs marked linear on both sides: R/B map through the scale
        // LUT on the way in and out, G passes the value through untouched —
        // on the way in (`rgb[c] = in[c]`) AND on the way out
        // (`out[c] = temp[c]`).
        let m = identity_matrix();
        let sc = scale_lut();
        let lin = linear_lut();
        let c = identity_coeff();
        let image_in = vec![0.5f32, 0.5, 0.5, 0.9];
        let mut out = vec![-7.0f32; 4];
        matrix_rgb(
            &image_in, &mut out, &m, [&sc, &lin, &sc], [&c, &c, &c], 2, true,
            [&sc, &lin, &sc], [&c, &c, &c], 2, true, 1, 1,
        );
        // lin: [0.75, 0.5, 0.75]; dl: [0.875, 0.5, 0.875]; alpha prefill kept.
        assert_eq!(out, vec![0.875, 0.5, 0.875, -7.0]);
        let mut refr = vec![-7.0f32; 4];
        ref_matrix_rgb(
            &image_in, &mut refr, &m, [&sc, &lin, &sc], [&c, &c, &c], 2, true,
            [&sc, &lin, &sc], [&c, &c, &c], 2, true, 1, 1,
        );
        assert_eq!(refr, out);
    }

    #[test]
    fn matrix_rgb_premultiplied_equivalence() {
        // the C wrapper premultiplies once per image; one kernel call with
        // the standard 4x4 product C = A*B must equal the chained A-then-B
        // application up to f32 rounding (different association order), while
        // the kernel-vs-reference comparison on C itself is bit-exact.
        let a = vec![
            0.9, 0.1, 0.05, 0.0, //
            0.2, 0.8, 0.1, 0.0, //
            0.05, 0.15, 0.7, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        let b = vec![
            1.1, 0.0, 0.1, 0.0, //
            0.05, 0.9, 0.0, 0.0, //
            0.1, 0.05, 1.2, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        let mut c = vec![0.0f32; 16];
        for row in 0..4 {
            for col in 0..4 {
                let mut s = 0.0f32;
                for k in 0..4 {
                    s += a[row * 4 + k] * b[k * 4 + col];
                }
                c[row * 4 + col] = s;
            }
        }
        let empty: Vec<f32> = vec![];
        let e = empty.as_slice();
        for &(w, h) in &[(1usize, 1usize), (3, 2), (5, 4)] {
            let mut image_in = vec![0.0f32; 4 * w * h];
            lcg_fill(&mut image_in, 0x1890 + w as u32 * 17 + h as u32, 1.5);
            let mut chained = vec![0.0f32; 4 * w * h];
            let mut step = vec![0.0f32; 4 * w * h];
            matrix_rgb(
                &image_in, &mut step, &a, [e, e, e], [e, e, e], 0, false,
                [e, e, e], [e, e, e], 0, false, w, h,
            );
            matrix_rgb(
                &step, &mut chained, &b, [e, e, e], [e, e, e], 0, false,
                [e, e, e], [e, e, e], 0, false, w, h,
            );
            let mut combined = vec![0.0f32; 4 * w * h];
            matrix_rgb(
                &image_in, &mut combined, &c, [e, e, e], [e, e, e], 0, false,
                [e, e, e], [e, e, e], 0, false, w, h,
            );
            // non-vacuous: the product is not the identity
            assert!(
                combined
                    .iter()
                    .zip(image_in.iter())
                    .any(|(o, i)| (o - i).abs() > 1e-3),
                "w={w} h={h}"
            );
            for (o, r) in combined.iter().zip(chained.iter()) {
                // NaN never appears for these finite inputs; compare relative
                assert!(
                    (o - r).abs() <= 1e-5 * r.abs().max(1.0),
                    "w={w} h={h}: {o} vs {r}"
                );
            }
            // ... while the reference on the product itself is bit-exact
            let mut refr = vec![0.0f32; 4 * w * h];
            ref_matrix_rgb(
                &image_in, &mut refr, &c, [e, e, e], [e, e, e], 0, false,
                [e, e, e], [e, e, e], 0, false, w, h,
            );
            assert_eq!(combined, refr, "w={w} h={h}");
            for (o, r) in combined.iter().zip(refr.iter()) {
                assert_eq!(o.to_bits(), r.to_bits(), "w={w} h={h}");
            }
        }
    }

    #[test]
    fn matrix_rgb_matches_reference_over_lcg() {
        // kernel vs structurally divergent reference: bit-exact (to_bits)
        // over all four (nonlinear_from, nonlinear_to) combos, mixed
        // per-side sentinel LUT sets, and — unlike m4-187/188 — DIFFERENT
        // input/output lutsizes (the two profiles carry their own).
        let luts5: Vec<Vec<f32>> = vec![
            vec![0.0, 0.2, 0.5, 0.9, 1.0],
            vec![0.1, 0.3, 0.4, 0.8, 1.0],
            vec![0.0, 0.0, 0.6, 0.7, 1.0],
        ];
        let lin5 = vec![-1.0f32, 0.2, 0.5, 0.9, 1.0];
        let coeff_sets: Vec<[f32; 3]> = vec![[1.0, 1.2, 0.9], [0.8, 1.0, 1.1], [1.1, 0.9, 1.0]];
        let out_lut = scale_lut();
        let out_lin = linear_lut();
        let c2 = const2_coeff();
        let matrices = [identity_matrix(), permute_matrix(), srgb_like_matrix()];
        // (in-mask, out-mask): which channels carry active TRCs per side
        let masks: Vec<([bool; 3], [bool; 3])> = vec![
            ([true, true, true], [true, true, true]),
            ([true, false, true], [false, true, false]),
            ([false, false, false], [true, true, true]),
            ([true, true, true], [false, false, false]),
        ];
        for m in &matrices {
            for (in_mask, out_mask) in &masks {
                let li: [&[f32]; 3] = [
                    if in_mask[0] { &luts5[0] } else { &lin5 },
                    if in_mask[1] { &luts5[1] } else { &lin5 },
                    if in_mask[2] { &luts5[2] } else { &lin5 },
                ];
                let lo: [&[f32]; 3] = [
                    if out_mask[0] { &out_lut } else { &out_lin },
                    if out_mask[1] { &out_lut } else { &out_lin },
                    if out_mask[2] { &out_lut } else { &out_lin },
                ];
                for &(nf, nt) in &[(false, false), (true, false), (false, true), (true, true)]
                {
                    for &(w, h) in &[(1usize, 1usize), (2, 1), (3, 2), (5, 4)] {
                        let mut image_in = vec![0.0f32; 4 * w * h];
                        // [-0.5, 1.5): clamp, LUT and exp regions, near 1.0
                        lcg_fill(&mut image_in, 0x1891 + w as u32 * 17 + h as u32, 2.0);
                        for v in image_in.iter_mut().step_by(4) {
                            *v -= 0.5;
                        }
                        if !image_in.is_empty() {
                            image_in[0] = 1.0;
                            if image_in.len() > 4 {
                                image_in[4] = -0.0;
                            }
                            if image_in.len() > 8 {
                                image_in[8] = 0.999_999_9;
                            }
                        }
                        let mut direct = vec![-33.0f32; 4 * w * h];
                        let mut refr = vec![-33.0f32; 4 * w * h];
                        matrix_rgb(
                            &image_in, &mut direct, m, li,
                            [&coeff_sets[0], &coeff_sets[1], &coeff_sets[2]], 5, nf, lo,
                            [&c2, &c2, &c2], 2, nt, w, h,
                        );
                        ref_matrix_rgb(
                            &image_in, &mut refr, m, li,
                            [&coeff_sets[0], &coeff_sets[1], &coeff_sets[2]], 5, nf, lo,
                            [&c2, &c2, &c2], 2, nt, w, h,
                        );
                        assert_eq!(direct, refr, "masks={in_mask:?}/{out_mask:?} nf={nf} nt={nt} w={w} h={h}");
                        for (d, r) in direct.iter().zip(refr.iter()) {
                            assert_eq!(
                                d.to_bits(),
                                r.to_bits(),
                                "masks={in_mask:?}/{out_mask:?} nf={nf} nt={nt} w={w} h={h}"
                            );
                        }
                        // alpha lanes: written on matrix-direct paths only
                        for px in 0..w * h {
                            if nt {
                                assert_eq!(
                                    direct[px * 4 + 3].to_bits(),
                                    (-33.0f32).to_bits(),
                                    "alpha preserved nf={nf} nt={nt}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn matrix_rgb_degenerate_guards_no_op() {
        let m = identity_matrix();
        let short_m = vec![1.0f32; 15];
        let sc = scale_lut();
        let c2 = const2_coeff();
        let empty: Vec<f32> = vec![];
        let e = empty.as_slice();
        // zero dims leave buffers untouched (kernel + reference, all paths)
        for &(w, h) in &[(0usize, 4usize), (4usize, 0usize), (0usize, 0usize)] {
            for &(nf, nt) in &[(false, false), (true, true)] {
                let inn: Vec<f32> = vec![0.5; 64];
                let mut out = vec![9.0f32; 64];
                matrix_rgb(
                    &inn, &mut out, &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, nf,
                    [&sc, &sc, &sc], [&c2, &c2, &c2], 2, nt, w, h,
                );
                assert!(out.iter().all(|&v| v == 9.0), "w={w} h={h}");
                let mut ref_out = vec![9.0f32; 64];
                ref_matrix_rgb(
                    &inn, &mut ref_out, &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, nf,
                    [&sc, &sc, &sc], [&c2, &c2, &c2], 2, nt, w, h,
                );
                assert!(ref_out.iter().all(|&v| v == 9.0), "ref w={w} h={h}");
            }
        }
        // short images / short matrix: no-op, no panic
        let (w, h) = (4usize, 4usize);
        let mut inn = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut inn, 0x5E90, 2.0);
        let mut out = vec![9.0f32; 4 * w * h];
        matrix_rgb(
            &inn[..10], &mut out, &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        matrix_rgb(
            &inn, &mut out[..10], &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        matrix_rgb(
            &inn, &mut out, &short_m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        matrix_rgb(
            &inn, &mut out, &[], [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        assert!(out.iter().all(|&v| v == 9.0));
        // nonlinear side with short tables / degenerate lutsize: no-op
        let short_lut = vec![0.5f32; 1];
        let short_c = vec![1.0f32; 2];
        let mut o = vec![9.0f32; 4 * w * h];
        // short input LUT (len 1 < lutsize 2)
        matrix_rgb(
            &inn, &mut o, &m, [&short_lut, &sc, &sc], [&c2, &c2, &c2], 2, true,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        // short input coeffs (len 2 < 3)
        matrix_rgb(
            &inn, &mut o, &m, [&sc, &sc, &sc], [&short_c, &c2, &c2], 2, true,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        // degenerate input lutsize
        matrix_rgb(
            &inn, &mut o, &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 1, true,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        // short output LUT / degenerate output lutsize
        matrix_rgb(
            &inn, &mut o, &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true,
            [&short_lut, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        matrix_rgb(
            &inn, &mut o, &m, [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 0, true, w, h,
        );
        assert!(o.iter().all(|&v| v == 9.0));
        // linear sides skip table validation entirely: EMPTY tables still run
        let mut o = vec![9.0f32; 4 * w * h];
        matrix_rgb(
            &inn, &mut o, &m, [e, e, e], [e, e, e], 0, false, [e, e, e], [e, e, e],
            0, false, w, h,
        );
        assert!(!o.iter().all(|&v| v == 9.0)); // ran: identity matrix copied RGB
        // one side linear + other nonlinear: only the nonlinear side is gated
        let mut o = vec![9.0f32; 4 * w * h];
        matrix_rgb(
            &inn, &mut o, &m, [e, e, e], [e, e, e], 0, false,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        assert!(!o.iter().all(|&v| v == 9.0));
        let mut o = vec![9.0f32; 4 * w * h];
        ref_matrix_rgb(
            &inn, &mut o, &m, [e, e, e], [e, e, e], 0, false,
            [&sc, &sc, &sc], [&c2, &c2, &c2], 2, true, w, h,
        );
        assert!(!o.iter().all(|&v| v == 9.0));
        // boundary dimension arithmetic must no-op without panicking
        matrix_rgb(
            &[], &mut [], &m, [e, e, e], [e, e, e], 0, false, [e, e, e], [e, e, e],
            0, false, usize::MAX, 1,
        );
        ref_matrix_rgb(
            &[], &mut [], &m, [e, e, e], [e, e, e], 0, false, [e, e, e], [e, e, e],
            0, false, usize::MAX, 1,
        );
    }

    #[test]
    fn matrix_rgb_ffi_round_trip() {
        // FFI calls agree with the safe kernel on every flag combo, with
        // distinct per-side lutsizes.
        let luts5: Vec<Vec<f32>> = vec![
            vec![0.0, 0.2, 0.5, 0.9, 1.0],
            vec![0.1, 0.3, 0.4, 0.8, 1.0],
            vec![0.0, 0.0, 0.6, 0.7, 1.0],
        ];
        let coeff_sets: Vec<[f32; 3]> = vec![[1.0, 1.2, 0.9], [0.8, 1.0, 1.1], [1.1, 0.9, 1.0]];
        let out_lut = scale_lut();
        let c2 = const2_coeff();
        let matrices = [identity_matrix(), permute_matrix(), srgb_like_matrix()];
        for m in &matrices {
            for &(nf, nt) in &[(false, false), (true, false), (false, true), (true, true)] {
                for &(w, h) in &[(4usize, 2usize), (5usize, 3usize)] {
                    let mut image_in = vec![0.0f32; 4 * w * h];
                    lcg_fill(&mut image_in, 0xFF90 + w as u32, 2.0);
                    let mut ffi_out = vec![-3.0f32; 4 * w * h];
                    let mut direct = vec![-3.0f32; 4 * w * h];
                    unsafe {
                        darkroom_iop_profile_matrix_rgb(
                            image_in.as_ptr(),
                            ffi_out.as_mut_ptr(),
                            w,
                            h,
                            m.as_ptr(),
                            luts5[0].as_ptr(),
                            luts5[1].as_ptr(),
                            luts5[2].as_ptr(),
                            coeff_sets[0].as_ptr(),
                            coeff_sets[1].as_ptr(),
                            coeff_sets[2].as_ptr(),
                            out_lut.as_ptr(),
                            out_lut.as_ptr(),
                            out_lut.as_ptr(),
                            c2.as_ptr(),
                            c2.as_ptr(),
                            c2.as_ptr(),
                            5,
                            2,
                            nf as i32,
                            nt as i32,
                        );
                    }
                    matrix_rgb(
                        &image_in, &mut direct, m,
                        [&luts5[0], &luts5[1], &luts5[2]],
                        [&coeff_sets[0], &coeff_sets[1], &coeff_sets[2]],
                        5, nf,
                        [&out_lut, &out_lut, &out_lut],
                        [&c2, &c2, &c2],
                        2, nt, w, h,
                    );
                    assert_eq!(ffi_out, direct, "nf={nf} nt={nt} w={w} h={h}");
                    for (f, d) in ffi_out.iter().zip(direct.iter()) {
                        assert_eq!(f.to_bits(), d.to_bits(), "nf={nf} nt={nt}");
                    }
                }
            }
        }
    }

    #[test]
    fn matrix_rgb_ffi_guards() {
        let m = identity_matrix();
        let sc = scale_lut();
        let c2 = const2_coeff();
        let inn: Vec<f32> = vec![0.5; 64];
        let mut out = vec![7.0f32; 64];
        unsafe {
            // null image/matrix pointers (each, one at a time)
            darkroom_iop_profile_matrix_rgb(
                std::ptr::null(), out.as_mut_ptr(), 4, 4, m.as_ptr(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), 2, 2, 1, 1,
            );
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), std::ptr::null_mut(), 4, 4, m.as_ptr(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), 2, 2, 1, 1,
            );
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, std::ptr::null(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), 2, 2, 1, 1,
            );
            // null input LUT with nonlinear_from: guarded no-op ...
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, m.as_ptr(), std::ptr::null(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), 2, 2, 1, 1,
            );
            // null output LUT / coeff with nonlinear_to: guarded no-op ...
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, m.as_ptr(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                std::ptr::null(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), 2, 2, 1, 1,
            );
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, m.as_ptr(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                std::ptr::null(), 2, 2, 1, 1,
            );
            // ... but degenerate input lutsize with linear_from: tables
            // untouched, so the call below (flags 0,0 with nulls) must RUN.
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, m.as_ptr(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), 1, 0, 1, 1,
            );
            // zero dims
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 0, 4, m.as_ptr(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), 2, 2, 1, 1,
            );
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, 0, m.as_ptr(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), 2, 2, 1, 1,
            );
            // huge lutsize on a nonlinear side: no-op, no slice-construction UB
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, m.as_ptr(), sc.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), usize::MAX, 2, 1, 1,
            );
            // i32::MAX caps (C ints arrive non-negative; negatives would wrap
            // to huge size_t and must not reach slice construction)
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), (i32::MAX as usize) + 1, 4, m.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), c2.as_ptr(), 2, 2, 1, 1,
            );
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, (i32::MAX as usize) + 1, m.as_ptr(),
                sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), sc.as_ptr(), sc.as_ptr(), sc.as_ptr(), c2.as_ptr(),
                c2.as_ptr(), c2.as_ptr(), 2, 2, 1, 1,
            );
        }
        assert!(out.iter().all(|&v| v == 7.0)); // untouched
        // linear sides tolerate null LUTs and any lutsize: runs (identity
        // matrix copies RGB, zeroes alpha).
        let mut out = vec![7.0f32; 64];
        unsafe {
            darkroom_iop_profile_matrix_rgb(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, m.as_ptr(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), std::ptr::null(), usize::MAX,
                usize::MAX, 0, 0,
            );
        }
        for px in 0..16 {
            assert_eq!(out[px * 4], 0.5);
            assert_eq!(out[px * 4 + 1], 0.5);
            assert_eq!(out[px * 4 + 2], 0.5);
            assert_eq!(out[px * 4 + 3].to_bits(), 0.0f32.to_bits());
        }
    }
}
