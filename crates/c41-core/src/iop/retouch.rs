//! Retouch IOP helpers -- portable OMP loops from retouch.c.
//!
//! In addition to the mask/copy helpers below, this module ports the two
//! preview-levels loops (m4-190):
//! - `rt_process_stats` (retouch.c:3066): per-pixel working-RGB→Lab (profile
//!   TRC + matrix, or sRGB fallback) with an L min/max/sum reduction;
//! - `rt_adjust_levels` (retouch.c:3111): in-place working-RGB→Lab, L-channel
//!   level scaling, Lab→working-RGB.
//!
//! Both reuse the shared [`crate::color`] TRC / transposed-matrix / XYZ↔Lab
//! helpers and the [`crate::iop_profile`] table conventions (flattened
//! 16-float matrices, per-channel LUT + 3-float coefficient slices).

use crate::{params::IopParams, roi::RoiIn, Result};
use crate::color::{
    apply_trc, apply_transposed_color_matrix, lab_to_xyz, srgb_to_xyz_d50, xyz_d50_to_srgb,
    xyz_to_lab,
};
use super::{ClBuffer, IopProcess};

pub struct Retouch;
impl IopProcess for Retouch {
    fn name(&self) -> &'static str { "retouch" }
    fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("retouch: use C FFI path".into()))
    }
    fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("retouch: no OpenCL path".into()))
    }
}

/// Copy y_to rows from `in` (with offset xoffs,yoffs) into `out`.
/// rowsize is in bytes. Matches rt_copy_in_to_out DT_OMP_FOR at retouch.c:3250.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_copy_rows(
    in_buf:    *const f32,
    out_buf:   *mut f32,
    y_to:      i32,
    xoffs:     i32,
    yoffs:     i32,
    in_width:  i32,
    out_width: i32,
    ch:        i32,
    rowsize:   usize,
) {
    let bytes_in  = ((yoffs + y_to) * in_width * ch) as usize * 4;
    let bytes_out = (y_to * out_width * ch) as usize * 4;
    let inp  = std::slice::from_raw_parts(in_buf  as *const u8, bytes_in);
    let outp = std::slice::from_raw_parts_mut(out_buf as *mut u8, bytes_out);
    for y in 0..y_to as usize {
        let si = ((y as i32 + yoffs) * in_width + xoffs) as usize * ch as usize * 4;
        let di = y * out_width as usize * ch as usize * 4;
        outp[di..di + rowsize].copy_from_slice(&inp[si..si + rowsize]);
    }
}

/// Nearest-neighbour mask scaling into `mask_tmp`.
/// Matches rt_build_scaled_mask DT_OMP_FOR at retouch.c:3300.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_build_mask(
    mask:       *const f32,
    mask_tmp:   *mut f32,
    roi_mask_x: i32, roi_mask_y: i32,
    roi_mask_w: i32, roi_mask_h: i32,
    roi_ms_x:   i32, roi_ms_y:   i32,
    roi_ms_w:   i32, roi_ms_h:   i32,
    x_to:       i32, y_to:       i32,
    scale:      f32,
) {
    let m  = std::slice::from_raw_parts(mask,     (roi_mask_w * roi_mask_h) as usize);
    let ms = std::slice::from_raw_parts_mut(mask_tmp, (roi_ms_w * roi_ms_h) as usize);
    for yy in roi_ms_y..y_to {
        let mi = (yy as f32 / scale) as i32 - roi_mask_y;
        if mi < 0 || mi >= roi_mask_h { continue; }
        let ms_row = (yy - roi_ms_y) * roi_ms_w;
        for xx in roi_ms_x..x_to {
            let mx = (xx as f32 / scale) as i32 - roi_mask_x;
            if mx < 0 || mx >= roi_mask_w { continue; }
            ms[(ms_row + xx - roi_ms_x) as usize] = m[(mi * roi_mask_w + mx) as usize];
        }
    }
}

/// Masked alpha blend: dest = dest*(1-f) + src*f  where f = mask*opacity.
/// dest_npixels is roi_dest->width * roi_dest->height.
/// Matches rt_copy_image_masked DT_OMP_FOR at retouch.c:3333.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_copy_masked(
    src:          *const f32,
    dest:         *mut f32,
    dest_roi_x:   i32, dest_roi_y: i32, dest_roi_w: i32,
    dest_npixels: usize,
    mask:         *const f32,
    mask_roi_x:   i32, mask_roi_y: i32,
    mask_w:       i32, mask_h:     i32,
    opacity:      f32,
) {
    let s = std::slice::from_raw_parts(src,  (mask_w * mask_h * 4) as usize);
    let d = std::slice::from_raw_parts_mut(dest, dest_npixels * 4);
    let m = std::slice::from_raw_parts(mask, (mask_w * mask_h) as usize);
    for yy in 0..mask_h as usize {
        let mi = yy * mask_w as usize;
        let si = mi * 4;
        let di = ((yy as i32 + mask_roi_y - dest_roi_y) * dest_roi_w
                  + (mask_roi_x - dest_roi_x)) as usize * 4;
        for xx in 0..mask_w as usize {
            let f  = m[mi + xx] * opacity;
            let f1 = 1.0 - f;
            for c in 0..4 {
                d[di + xx*4 + c] = d[di + xx*4 + c] * f1 + s[si + xx*4 + c] * f;
            }
        }
    }
}

/// Update dest alpha: d[3] = max(d[3], mask*opacity).
/// img_npixels = roi_img->width * roi_img->height.
/// Matches rt_copy_mask_to_alpha DT_OMP_FOR at retouch.c:3366.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_copy_mask_to_alpha(
    img:           *mut f32,
    roi_img_x:     i32, roi_img_y: i32, roi_img_w: i32,
    img_npixels:   usize,
    ch:            i32,
    mask:          *const f32,
    mask_roi_x:    i32, mask_roi_y: i32,
    mask_w:        i32, mask_h:     i32,
    opacity:       f32,
) {
    let d = std::slice::from_raw_parts_mut(img, img_npixels * ch as usize);
    let m = std::slice::from_raw_parts(mask, (mask_w * mask_h) as usize);
    for yy in 0..mask_h as usize {
        let mi = yy * mask_w as usize;
        let di = ((yy as i32 + mask_roi_y - roi_img_y) * roi_img_w
                  + (mask_roi_x - roi_img_x)) as usize * ch as usize;
        for xx in 0..mask_w as usize {
            let f = m[mi + xx] * opacity;
            let alpha = &mut d[di + xx * ch as usize + 3];
            if f > *alpha { *alpha = f; }
        }
    }
}

/// Fill masked region: dest = dest*(1-f) + fill_color*f where f = mask*opacity.
/// dest_npixels = roi_in->width * roi_in->height.
/// Matches _retouch_fill DT_OMP_FOR at retouch.c:3392.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_fill(
    dest:          *mut f32,
    roi_in_x:      i32, roi_in_y: i32, roi_in_w: i32,
    dest_npixels:  usize,
    mask:          *const f32,
    mask_roi_x:    i32, mask_roi_y: i32,
    mask_w:        i32, mask_h:     i32,
    opacity:       f32,
    fill_color:    *const f32,   // 4 floats
) {
    let d    = std::slice::from_raw_parts_mut(dest, dest_npixels * 4);
    let m    = std::slice::from_raw_parts(mask, (mask_w * mask_h) as usize);
    let fill = std::slice::from_raw_parts(fill_color, 4);
    for yy in 0..mask_h as usize {
        let mi = yy * mask_w as usize;
        let di = ((yy as i32 + mask_roi_y - roi_in_y) * roi_in_w
                  + (mask_roi_x - roi_in_x)) as usize * 4;
        for xx in 0..mask_w as usize {
            let f  = m[mi + xx] * opacity;
            let f1 = 1.0 - f;
            for c in 0..4 {
                d[di + xx*4 + c] = d[di + xx*4 + c] * f1 + fill[c] * f;
            }
        }
    }
}

// ── Preview auto-levels (m4-190) ─────────────────────────────────────────────
//
// Ports of `rt_process_stats` (retouch.c:3066-3109) and `rt_adjust_levels`
// (retouch.c:3111-3187): the two `DT_OMP_FOR` loops that bracket the wavelet
// preview path. Each pixel is converted working-RGB→Lab through the pipe work
// profile (`dt_ioppr_rgb_matrix_to_lab`: per-channel TRC when `nonlinearlut`,
// then the pre-transposed `matrix_in_transposed`, then `dt_XYZ_to_Lab`), or —
// when the C caller passes no work profile — through the sRGB fallback
// (`dt_linearRGB_to_XYZ` + `dt_XYZ_to_Lab`). `rt_adjust_levels` scales the L
// lane and converts back (`dt_ioppr_lab_to_rgb_matrix` / `dt_Lab_to_XYZ` +
// `dt_XYZ_to_linearRGB`).
//
// Semantic edge cases (all pinned by tests):
// - Lane 3 converges to **+0.0** in every path for finite inputs, exactly as
//   in C: the forward `dt_XYZ_to_Lab` zeroes it (4-channel vector build:
//   `0*(f[3]-0)-0`), and the backward conversion never restores it (the TRC
//   branch leaves lane 3 untouched; the linear/matrix branches recompute the
//   zero-padding product, `+0.0` for finite inputs). For non-finite inputs the
//   padding product can yield NaN (`0.0 * Inf`, same as C — neither side reads
//   `xyz[3]` downstream), so the `+0.0` claim is finite-inputs-only. The
//   shared [`xyz_to_lab`] instead preserves `xyz[3]`, so the kernels force
//   lane 3 to 0.0 after the forward call; the fallback matrix helpers likewise
//   preserve the input alpha where C writes 0.0, hence the same forcing.
//   Callers must not expect alpha preservation here (the conditional C
//   `dt_iop_alpha_copy` after the levels call — retouch.c:3772-3778, only when
//   mask display is active — handles display).
// - `Lab[0..2]` never depend on `xyz[3]` (only `xyz[0..2]` feed `f`), so the
//   lane-3 forcing is unobservable in the stats reduction and the L scaling.
// - Stats reduction mirrors the glib `MIN`/`MAX` macros literally
//   (`a<b?a:b`), including NaN propagation (`MIN(x,NaN)=NaN` but
//   `MIN(NaN,x)=x`, so a trailing NaN L poisons the extrema while a later
//   finite L rescues them — unlike `f32::min`/`max`, which ignore NaN
//   entirely; the sum, once NaN, stays NaN). The sum folds
//   sequentially in pixel order (deterministic; the old OpenMP
//   `reduction(+:l_sum)` / `reduction(max/min:)` partitioned work across
//   threads with unspecified combine order).
// - Levels scaling mirrors the C lane loop (`for c in 0..1`: L only; a/b
//   pass through the round trip untouched): `L_in <= left` maps to exactly
//   0.0, otherwise `100 * ((L_in-left)/(right-left))^in_inv_gamma`.
// - The default-triple early return (`left == -3, middle == 0, right == 3`)
//   and the `in_inv_gamma = 10^((middle-mid)/delta)` derivation stay in the
//   C wrapper (orchestration, following the m4-187/188/189 precedent of
//   C pre-computing scalars); the kernel takes `left`, `right` and
//   `in_inv_gamma` directly.
// - `v < 1.0` selects the LUT path, `v >= 1.0` the `eval_exp` tail, via the
//   shared [`apply_trc`] (linear-marked channels, `lut[0] < 0.0`, pass
//   through on both sides).
//
// Degenerate/short buffers: zero `width`/`height`, `ch < 4` (both kernels
// touch 4 lanes per pixel — matching what the C loops safely address when
// `ch == 4`, which well-formed callers always forward), dimension arithmetic
// `ch*width*height` floats, matrices shorter than 16 floats, or — on a
// nonlinear profile — `lutsize < 2`, LUTs shorter than `lutsize`, or
// coefficient slices shorter than 3 floats is a no-op (stats leaves `levels`
// untouched, levels leaves the image untouched). Empty input (zero
// `width`/`height`) likewise leaves `levels` untouched, where the old C loop
// fell through to `(l_sum/count)/100 = 0/0 = NaN` with `±FLT_MAX/100`
// extrema; well-formed callers always forward the non-empty `roi_rt`
// dimensions, so only direct kernel/FFI misuse can observe the difference.
// Well-formed callers forward
// the profile's own tables over exactly `ch*width*height` floats (always
// `ch == 4`) and never hit this path.
//
// Aliasing: stats reads its input only; levels runs fully in place on a
// single buffer (each lane is staged through locals before it is written, so
// the forward/scale/backward sequence is exact), matching the C reliance on
// in-place conversion. No split-buffer levels kernel is provided.

/// Borrowed view of the pipe work-profile tables for the levels kernels.
///
/// `matrix_in`/`matrix_out` are the 16-float pre-transposed profile matrices
/// (`matrix_in_transposed` / `matrix_out_transposed`, applied without further
/// transposition). Each `lut_in`/`lut_out` entry holds `lutsize` floats, each
/// coefficient entry 3 floats. `nonlinear` is the profile's `nonlinearlut`.
/// `None` (no work profile) selects the sRGB fallback path.
pub struct RetouchProfile<'a> {
    pub matrix_in: &'a [f32],
    pub matrix_out: &'a [f32],
    pub lut_in: [&'a [f32]; 3],
    pub coeff_in: [&'a [f32]; 3],
    pub lut_out: [&'a [f32]; 3],
    pub coeff_out: [&'a [f32]; 3],
    pub lutsize: usize,
    pub nonlinear: bool,
}

/// Reinterpret a 16-float `dt_colormatrix_t` slice as `[[f32; 4]; 4]`.
///
/// The caller guarantees `matrix.len() >= 16` (checked by
/// [`checked_levels_len`]); row-major layout matches the C type exactly, and
/// `[[f32; 4]; 4]` has the same alignment as `f32`, so the cast is sound.
#[inline(always)]
fn as_colormatrix(matrix: &[f32]) -> &[[f32; 4]; 4] {
    // SAFETY: `matrix` holds at least 16 contiguous `f32`s with alignment 4,
    // which satisfies `[[f32; 4]; 4]` (size 64, alignment 4).
    unsafe { &*(matrix.as_ptr() as *const [[f32; 4]; 4]) }
}

/// Validate one side's TRC tables: `lutsize >= 2`, every LUT at least
/// `lutsize` floats, every coefficient slice at least 3 floats.
fn check_trc_tables(luts: [&[f32]; 3], coeffs: [&[f32]; 3], lutsize: usize) -> bool {
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

/// Validate dimensions and buffer lengths for the levels kernels.
///
/// Returns the pixel count (`width * height`) on success. Always fails on
/// zero dimensions, `ch < min_ch`, dimension arithmetic overflow, or image
/// buffers shorter than `ch*width*height` floats. With a profile, requires
/// both matrices to hold at least 16 floats and checks the TRC tables only
/// when `nonlinear` (a linear profile's tables are never touched).
fn checked_levels_len(
    image_len: usize,
    profile: Option<&RetouchProfile>,
    width: usize,
    height: usize,
    ch: usize,
    min_ch: usize,
) -> Option<usize> {
    if width == 0 || height == 0 || ch < min_ch {
        return None;
    }
    let npixels = width.checked_mul(height)?;
    let pix_len = npixels.checked_mul(ch)?;
    if image_len < pix_len {
        return None;
    }
    if let Some(p) = profile {
        if p.matrix_in.len() < 16 || p.matrix_out.len() < 16 {
            return None;
        }
        if p.nonlinear
            && (!check_trc_tables(p.lut_in, p.coeff_in, p.lutsize)
                || !check_trc_tables(p.lut_out, p.coeff_out, p.lutsize))
        {
            return None;
        }
    }
    Some(npixels)
}

/// Forward pixel conversion: working-RGB → Lab with lane 3 forced to 0.0.
///
/// Mirrors `dt_ioppr_rgb_matrix_to_lab` (TRC when nonlinear, transposed
/// matrix, `dt_XYZ_to_Lab` with its lane-3 zeroing) or, without a profile,
/// the `dt_linearRGB_to_XYZ` + `dt_XYZ_to_Lab` fallback.
fn retouch_rgb_to_lab(px: [f32; 4], profile: Option<&RetouchProfile>) -> [f32; 4] {
    match profile {
        Some(p) => {
            let lin = if p.nonlinear {
                apply_trc(px, p.lut_in, p.coeff_in, p.lutsize)
            } else {
                px
            };
            let xyz = apply_transposed_color_matrix(&lin, as_colormatrix(p.matrix_in));
            let mut lab = xyz_to_lab(xyz);
            // dt_XYZ_to_Lab zeroes lane 3 (vector build); xyz_to_lab
            // preserves xyz[3] instead, so force it here. Lab[0..2] never
            // depend on xyz[3].
            lab[3] = 0.0;
            lab
        }
        None => {
            let mut lab = xyz_to_lab(srgb_to_xyz_d50(px));
            lab[3] = 0.0;
            lab
        }
    }
}

/// Backward pixel conversion: Lab (lane 3 must be 0.0) → working-RGB.
///
/// Mirrors `dt_ioppr_lab_to_rgb_matrix` (or the `dt_Lab_to_XYZ` +
/// `dt_XYZ_to_linearRGB` fallback). The nonlinear TRC branch writes lanes
/// 0..2 only; lane 3 carries the matrix zero-padding product (`+0.0` for
/// finite inputs), matching the C buffer value left by the forward pass.
fn retouch_lab_to_rgb(lab: [f32; 4], profile: Option<&RetouchProfile>) -> [f32; 4] {
    match profile {
        Some(p) => {
            // lab[3] == 0.0, so lab_to_xyz yields xyz[3] == 0.0 exactly as
            // C's `d50[3] * inv[3]` (whose -0.0 is never read downstream).
            let xyz = lab_to_xyz(lab);
            let rgb = apply_transposed_color_matrix(&xyz, as_colormatrix(p.matrix_out));
            if p.nonlinear {
                let dl = apply_trc(rgb, p.lut_out, p.coeff_out, p.lutsize);
                [dl[0], dl[1], dl[2], rgb[3]]
            } else {
                rgb
            }
        }
        None => xyz_d50_to_srgb(lab_to_xyz(lab)),
    }
}

/// L min/max/sum reduction (`rt_process_stats`, retouch.c:3082-3108).
///
/// `levels` receives `[l_min/100, mean/100, l_max/100]`. Degenerate/short
/// inputs leave `levels` untouched.
/// NOTE: empty input leaves `levels` untouched, where C produced NaN/±FLT_MAX/100.
pub fn retouch_process_stats(
    img: &[f32],
    width: usize,
    height: usize,
    ch: usize,
    levels: &mut [f32; 3],
    profile: Option<&RetouchProfile>,
) {
    let Some(npixels) = checked_levels_len(img.len(), profile, width, height, ch, 4)
    else {
        return;
    };
    // C seeds: l_max = -FLT_MAX, l_min = FLT_MAX (== f32::MAX).
    let mut l_min = f32::MAX;
    let mut l_max = -f32::MAX;
    let mut l_sum = 0.0f32;
    for px in 0..npixels {
        let b = px * ch;
        let lab = retouch_rgb_to_lab([img[b], img[b + 1], img[b + 2], img[b + 3]], profile);
        // glib MIN/MAX order (a<b?a:b / a>b?a:b), NaN-poisoning included —
        // NOT f32::min/max, which ignore NaN.
        l_min = if l_min < lab[0] { l_min } else { lab[0] };
        l_max = if l_max > lab[0] { l_max } else { lab[0] };
        l_sum += lab[0];
    }
    let n = npixels as f32;
    levels[0] = l_min / 100.0;
    levels[2] = l_max / 100.0;
    levels[1] = (l_sum / n) / 100.0;
}

/// In-place L level scaling (`rt_adjust_levels`, retouch.c:3135-3187).
///
/// `left`/`right` are the level bounds (L/100 domain), `in_inv_gamma` the
/// C pre-computed `10^((middle-mid)/delta)`. Pixels with `L/100 <= left` go
/// to exactly 0.0, otherwise `100 * ((L/100-left)/(right-left))^in_inv_gamma`.
/// Degenerate/short inputs leave the image untouched.
#[allow(clippy::too_many_arguments)]
pub fn retouch_adjust_levels(
    img: &mut [f32],
    width: usize,
    height: usize,
    ch: usize,
    left: f32,
    right: f32,
    in_inv_gamma: f32,
    profile: Option<&RetouchProfile>,
) {
    let Some(npixels) = checked_levels_len(img.len(), profile, width, height, ch, 4)
    else {
        return;
    };
    for px in 0..npixels {
        let b = px * ch;
        // Stage through locals: the C loop converts in place (forward into
        // the buffer, scale lane 0, convert back), which is exact lane-wise.
        let mut lab = retouch_rgb_to_lab([img[b], img[b + 1], img[b + 2], img[b + 3]], profile);
        // The C `for(c = 0; c < 1; c++)` touches the L lane only.
        let l_in = lab[0] / 100.0;
        lab[0] = if l_in <= left {
            0.0
        } else {
            100.0 * ((l_in - left) / (right - left)).powf(in_inv_gamma)
        };
        let rgb = retouch_lab_to_rgb(lab, profile);
        img[b] = rgb[0];
        img[b + 1] = rgb[1];
        img[b + 2] = rgb[2];
        img[b + 3] = rgb[3];
    }
}

// ── Independent reference implementations for bit-exactness tests ─────────────

// Local copies of the D50 Lab constants so the references do not call into
// the shared-helper path. Values mirror
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

// sRGB(D50)→XYZ matrix from colorspaces_inline_conversions.h:505-508,
// row-major M[in][out] (transposed application).
#[allow(dead_code)]
const REF_SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4360747, 0.2225045, 0.0139322],
    [0.3850649, 0.7168786, 0.0971045],
    [0.1430804, 0.0606169, 0.7141733],
];
#[allow(dead_code)]
const REF_XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [3.1338561, -0.9787684, 0.0719453],
    [-1.6168667, 1.9161415, -0.2289914],
    [-0.4906146, 0.0334540, 1.4052427],
];

/// Inlined LUT sample so the references do not call the shared helper path.
/// Same ops as `extrapolate_lut`, written out.
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

/// Inlined exponential tail so the references do not call `eval_exp`.
/// Same ops: `coeff[1] * (x * coeff[0])^coeff[2]`.
#[allow(dead_code)]
fn ref_exp_sample(coeff: &[f32], x: f32) -> f32 {
    coeff[1] * (x * coeff[0]).powf(coeff[2])
}

/// Inlined per-channel TRC (linear-marked channels pass through).
#[allow(dead_code)]
fn ref_apply_trc(lut: &[f32], coeff: &[f32], lutsize: usize, v: f32) -> f32 {
    if lut[0] < 0.0 {
        v
    } else if v < 1.0 {
        ref_lut_sample(lut, v, lutsize)
    } else {
        ref_exp_sample(coeff, v)
    }
}

/// Structurally divergent reference for [`retouch_rgb_to_lab`]: per-pixel
/// scalar staging through lane-by-lane flat-index matrix dots
/// (`matrix[r]*v0 + matrix[4+r]*v1 + matrix[8+r]*v2`) and an inlined XYZ→Lab,
/// where the kernel threads whole `[f32; 4]` arrays through the shared
/// helpers. Arithmetic association order is kept identical so the comparison
/// is bit-exact on finite inputs.
#[allow(dead_code)]
fn ref_retouch_rgb_to_lab(px: [f32; 4], profile: Option<&RetouchProfile>) -> [f32; 4] {
    let (lin0, lin1, lin2, m): (f32, f32, f32, &[f32]) = match profile {
        Some(p) if p.nonlinear => (
            ref_apply_trc(p.lut_in[0], p.coeff_in[0], p.lutsize, px[0]),
            ref_apply_trc(p.lut_in[1], p.coeff_in[1], p.lutsize, px[1]),
            ref_apply_trc(p.lut_in[2], p.coeff_in[2], p.lutsize, px[2]),
            p.matrix_in,
        ),
        Some(p) => (px[0], px[1], px[2], p.matrix_in),
        None => {
            // sRGB fallback: lane dot against REF_SRGB_TO_XYZ, then inlined
            // dt_XYZ_to_Lab below.
            let x = REF_SRGB_TO_XYZ[0][0] * px[0]
                + REF_SRGB_TO_XYZ[1][0] * px[1]
                + REF_SRGB_TO_XYZ[2][0] * px[2];
            let y = REF_SRGB_TO_XYZ[0][1] * px[0]
                + REF_SRGB_TO_XYZ[1][1] * px[1]
                + REF_SRGB_TO_XYZ[2][1] * px[2];
            let z = REF_SRGB_TO_XYZ[0][2] * px[0]
                + REF_SRGB_TO_XYZ[1][2] * px[1]
                + REF_SRGB_TO_XYZ[2][2] * px[2];
            let mut f = [0.0f32; 3];
            for (i, v) in [x, y, z].iter().enumerate() {
                let t = *v * REF_D50_INV[i];
                f[i] = if t > REF_LAB_EPSILON {
                    t.cbrt()
                } else {
                    (REF_LAB_KAPPA * t + 16.0) / 116.0
                };
            }
            return [
                116.0 * f[1] - 16.0,
                500.0 * (f[0] - f[1]),
                -200.0 * (f[2] - f[1]),
                0.0,
            ];
        }
    };
    let mut xyz = [0.0f32; 3];
    for r in 0..3 {
        xyz[r] = m[r] * lin0 + m[4 + r] * lin1 + m[8 + r] * lin2;
    }
    let mut f = [0.0f32; 3];
    for i in 0..3 {
        let t = xyz[i] * REF_D50_INV[i];
        f[i] = if t > REF_LAB_EPSILON {
            t.cbrt()
        } else {
            (REF_LAB_KAPPA * t + 16.0) / 116.0
        };
    }
    [
        116.0 * f[1] - 16.0,
        500.0 * (f[0] - f[1]),
        -200.0 * (f[2] - f[1]),
        0.0,
    ]
}

/// Local copy of the `lab_f_inv` step so the backward reference does not call
/// the shared helper path.
#[allow(dead_code)]
fn ref_lab_f_inv(x: f32) -> f32 {
    if x > REF_LAB_CBRT_EPSILON {
        x * x * x
    } else {
        (116.0 * x - 16.0) / REF_LAB_KAPPA
    }
}

/// Structurally divergent reference for [`retouch_lab_to_rgb`]: explicit
/// fy/fx/fz temporaries, an inlined `lab_f_inv`, a lane-by-lane matrix dot,
/// and per-lane inlined TRC staging, where the kernel threads whole arrays
/// through the shared helpers. Same association order → bit-exact on finite
/// inputs.
#[allow(dead_code)]
fn ref_retouch_lab_to_rgb(lab: [f32; 4], profile: Option<&RetouchProfile>) -> [f32; 4] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;
    let xyz = [
        REF_D50[0] * ref_lab_f_inv(fx),
        REF_D50[1] * ref_lab_f_inv(fy),
        REF_D50[2] * ref_lab_f_inv(fz),
    ];
    match profile {
        Some(p) => {
            let mut tmp = [0.0f32; 4];
            for (r, t) in tmp.iter_mut().enumerate() {
                *t = p.matrix_out[r] * xyz[0]
                    + p.matrix_out[4 + r] * xyz[1]
                    + p.matrix_out[8 + r] * xyz[2];
            }
            if p.nonlinear {
                [
                    ref_apply_trc(p.lut_out[0], p.coeff_out[0], p.lutsize, tmp[0]),
                    ref_apply_trc(p.lut_out[1], p.coeff_out[1], p.lutsize, tmp[1]),
                    ref_apply_trc(p.lut_out[2], p.coeff_out[2], p.lutsize, tmp[2]),
                    tmp[3],
                ]
            } else {
                tmp
            }
        }
        None => {
            let m = REF_XYZ_TO_SRGB;
            [
                m[0][0] * xyz[0] + m[1][0] * xyz[1] + m[2][0] * xyz[2],
                m[0][1] * xyz[0] + m[1][1] * xyz[1] + m[2][1] * xyz[2],
                m[0][2] * xyz[0] + m[1][2] * xyz[1] + m[2][2] * xyz[2],
                0.0,
            ]
        }
    }
}

/// Structurally divergent reference for [`retouch_process_stats`]: two-pass
/// form (convert every pixel's L into a temp vec, then fold the reduction —
/// same pixel order, same MIN/MAX-macro comparisons) through the inlined
/// [`ref_retouch_rgb_to_lab`], where the kernel folds single-pass through the
/// shared helpers.
#[allow(dead_code)]
fn ref_retouch_process_stats(
    img: &[f32],
    width: usize,
    height: usize,
    ch: usize,
    levels: &mut [f32; 3],
    profile: Option<&RetouchProfile>,
) {
    let Some(npixels) = checked_levels_len(img.len(), profile, width, height, ch, 4)
    else {
        return;
    };
    let mut ls = Vec::with_capacity(npixels);
    for px in 0..npixels {
        let b = px * ch;
        ls.push(ref_retouch_rgb_to_lab([img[b], img[b + 1], img[b + 2], img[b + 3]], profile)[0]);
    }
    let mut l_min = f32::MAX;
    let mut l_max = -f32::MAX;
    let mut l_sum = 0.0f32;
    for l in ls {
        l_min = if l_min < l { l_min } else { l };
        l_max = if l_max > l { l_max } else { l };
        l_sum += l;
    }
    let n = npixels as f32;
    levels[0] = l_min / 100.0;
    levels[2] = l_max / 100.0;
    levels[1] = (l_sum / n) / 100.0;
}

/// Structurally divergent reference for [`retouch_adjust_levels`]: same
/// single-pass shape but through the inlined
/// [`ref_retouch_rgb_to_lab`]/[`ref_retouch_lab_to_rgb`], where the kernel
/// uses the shared helpers.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ref_retouch_adjust_levels(
    img: &mut [f32],
    width: usize,
    height: usize,
    ch: usize,
    left: f32,
    right: f32,
    in_inv_gamma: f32,
    profile: Option<&RetouchProfile>,
) {
    let Some(npixels) = checked_levels_len(img.len(), profile, width, height, ch, 4)
    else {
        return;
    };
    for px in 0..npixels {
        let b = px * ch;
        let mut lab =
            ref_retouch_rgb_to_lab([img[b], img[b + 1], img[b + 2], img[b + 3]], profile);
        let l_in = lab[0] / 100.0;
        lab[0] = if l_in <= left {
            0.0
        } else {
            100.0 * ((l_in - left) / (right - left)).powf(in_inv_gamma)
        };
        let rgb = ref_retouch_lab_to_rgb(lab, profile);
        img[b] = rgb[0];
        img[b + 1] = rgb[1];
        img[b + 2] = rgb[2];
        img[b + 3] = rgb[3];
    }
}

// ── FFI exports ──────────────────────────────────────────────────────────────

/// Materialize the borrowed profile view from raw pointers.
///
/// `have_profile == 0` selects the sRGB fallback (all table pointers ignored
/// and allowed null). Otherwise `matrix_in`/`matrix_out` must be non-null
/// 16-float transposed matrices; when `nonlinear != 0` the six LUT pointers
/// (each `lutsize` floats) and six coefficient pointers (each 3 floats) must
/// be non-null with `lutsize >= 2`, else `None` (→ FFI no-op). A linear
/// profile binds empty tables and tolerates null LUT/coefficient pointers.
#[allow(clippy::too_many_arguments)]
unsafe fn retouch_profile_view<'a>(
    matrix_in: *const f32,
    matrix_out: *const f32,
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
    lutsize: usize,
    nonlinear: bool,
    have_profile: bool,
) -> Option<Option<RetouchProfile<'a>>> {
    if !have_profile {
        return Some(None);
    }
    if matrix_in.is_null() || matrix_out.is_null() {
        return None;
    }
    let empty: &[f32] = &[];
    let (lut_in, coeff_in, lut_out, coeff_out) = if nonlinear {
        if lut_in_r.is_null()
            || lut_in_g.is_null()
            || lut_in_b.is_null()
            || coeff_in_r.is_null()
            || coeff_in_g.is_null()
            || coeff_in_b.is_null()
            || lut_out_r.is_null()
            || lut_out_g.is_null()
            || lut_out_b.is_null()
            || coeff_out_r.is_null()
            || coeff_out_g.is_null()
            || coeff_out_b.is_null()
            || lutsize < 2
        {
            return None;
        }
        (
            [
                std::slice::from_raw_parts(lut_in_r, lutsize),
                std::slice::from_raw_parts(lut_in_g, lutsize),
                std::slice::from_raw_parts(lut_in_b, lutsize),
            ],
            [
                std::slice::from_raw_parts(coeff_in_r, 3),
                std::slice::from_raw_parts(coeff_in_g, 3),
                std::slice::from_raw_parts(coeff_in_b, 3),
            ],
            [
                std::slice::from_raw_parts(lut_out_r, lutsize),
                std::slice::from_raw_parts(lut_out_g, lutsize),
                std::slice::from_raw_parts(lut_out_b, lutsize),
            ],
            [
                std::slice::from_raw_parts(coeff_out_r, 3),
                std::slice::from_raw_parts(coeff_out_g, 3),
                std::slice::from_raw_parts(coeff_out_b, 3),
            ],
        )
    } else {
        (
            [empty, empty, empty],
            [empty, empty, empty],
            [empty, empty, empty],
            [empty, empty, empty],
        )
    };
    Some(Some(RetouchProfile {
        matrix_in: std::slice::from_raw_parts(matrix_in, 16),
        matrix_out: std::slice::from_raw_parts(matrix_out, 16),
        lut_in,
        coeff_in,
        lut_out,
        coeff_out,
        lutsize,
        nonlinear,
    }))
}

/// Validate the shared FFI dimensions: positive `int`-domain width/height/ch
/// with a non-overflowing `ch*width*height` product. Returns the pixel length.
fn checked_ffi_dims(width: i32, height: i32, ch: i32) -> Option<usize> {
    if width <= 0 || height <= 0 || ch <= 0 {
        return None;
    }
    let (w, h, c) = (width as usize, height as usize, ch as usize);
    w.checked_mul(h)?.checked_mul(c)
}

/// L min/max/sum reduction (`rt_process_stats` loop, retouch.c:3082-3108).
///
/// `img` holds `ch*width*height` working-RGB floats (read only); `levels`
/// receives 3 floats `[min/100, mean/100, max/100]`. Flattened profile tables
/// follow the [`RetouchProfile`] layout (forward direction only):
/// `matrix_in` is the 16-float `matrix_in_transposed` exactly as stored; each
/// `lut_in_*` holds `lutsize` floats, each `coeff_in_*` 3 floats; `nonlinear`
/// is the profile's `nonlinearlut`. With `have_profile == 0` the sRGB
/// fallback runs and all table pointers are ignored (may be null). Null
/// `img`/`levels`, non-positive dimensions, dimension overflow, or
/// missing/invalid required tables are a no-op (the safe kernel additionally
/// guards short buffers).
///
/// # Safety
/// Non-null pointers must be valid for the stated lengths; the C caller
/// forwards the work profile's own tables over exactly `ch*width*height`
/// floats.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_process_stats(
    img: *const f32,
    width: i32,
    height: i32,
    ch: i32,
    levels: *mut f32,
    matrix_in: *const f32,
    lut_in_r: *const f32,
    lut_in_g: *const f32,
    lut_in_b: *const f32,
    coeff_in_r: *const f32,
    coeff_in_g: *const f32,
    coeff_in_b: *const f32,
    lutsize: i32,
    nonlinear: i32,
    have_profile: i32,
) {
    if img.is_null() || levels.is_null() {
        return;
    }
    let Some(pix_len) = checked_ffi_dims(width, height, ch) else {
        return;
    };
    if lutsize < 0 {
        return;
    }
    let have_profile = have_profile != 0;
    let nonlinear = nonlinear != 0;
    let lutsize = lutsize as usize;
    // Materialize the forward tables. The stats path never converts back, so
    // the backward slots reuse the forward slices (length-validated, never
    // read) to satisfy the shared validator.
    let profile = if !have_profile {
        None
    } else {
        if matrix_in.is_null() {
            return;
        }
        let empty: &[f32] = &[];
        let (lut_in, coeff_in) = if nonlinear {
            if lut_in_r.is_null()
                || lut_in_g.is_null()
                || lut_in_b.is_null()
                || coeff_in_r.is_null()
                || coeff_in_g.is_null()
                || coeff_in_b.is_null()
                || lutsize < 2
            {
                return;
            }
            (
                [
                    std::slice::from_raw_parts(lut_in_r, lutsize),
                    std::slice::from_raw_parts(lut_in_g, lutsize),
                    std::slice::from_raw_parts(lut_in_b, lutsize),
                ],
                [
                    std::slice::from_raw_parts(coeff_in_r, 3),
                    std::slice::from_raw_parts(coeff_in_g, 3),
                    std::slice::from_raw_parts(coeff_in_b, 3),
                ],
            )
        } else {
            ([empty, empty, empty], [empty, empty, empty])
        };
        let matrix = std::slice::from_raw_parts(matrix_in, 16);
        Some(RetouchProfile {
            matrix_in: matrix,
            matrix_out: matrix,
            lut_in,
            coeff_in,
            lut_out: lut_in,
            coeff_out: coeff_in,
            lutsize,
            nonlinear,
        })
    };
    let img = std::slice::from_raw_parts(img, pix_len);
    let levels = std::slice::from_raw_parts_mut(levels, 3);
    let levels_arr: &mut [f32; 3] = (&mut levels[..3]).try_into().unwrap();
    retouch_process_stats(
        img,
        width as usize,
        height as usize,
        ch as usize,
        levels_arr,
        profile.as_ref(),
    );
}

/// In-place L level scaling (`rt_adjust_levels` loop, retouch.c:3135-3187).
///
/// `img` holds `ch*width*height` working-RGB floats, converted in place
/// (each lane staged through locals, so aliasing is exact). `left`/`right`
/// are the level bounds and `in_inv_gamma` the C pre-computed
/// `10^((middle-mid)/delta)`; the default-triple early return stays in the C
/// wrapper. Flattened profile tables follow the [`RetouchProfile`] layout
/// (both directions); with `have_profile == 0` the sRGB fallback runs and
/// all table pointers are ignored (may be null). Guard behavior mirrors
/// [`darkroom_retouch_process_stats`].
///
/// # Safety
/// Non-null pointers must be valid for the stated lengths; the C caller
/// forwards the work profile's own tables over exactly `ch*width*height`
/// floats.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_adjust_levels(
    img: *mut f32,
    width: i32,
    height: i32,
    ch: i32,
    left: f32,
    right: f32,
    in_inv_gamma: f32,
    matrix_in: *const f32,
    matrix_out: *const f32,
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
    lutsize: i32,
    nonlinear: i32,
    have_profile: i32,
) {
    if img.is_null() {
        return;
    }
    let Some(pix_len) = checked_ffi_dims(width, height, ch) else {
        return;
    };
    if lutsize < 0 {
        return;
    }
    let Some(profile) = retouch_profile_view(
        matrix_in,
        matrix_out,
        lut_in_r,
        lut_in_g,
        lut_in_b,
        coeff_in_r,
        coeff_in_g,
        coeff_in_b,
        lut_out_r,
        lut_out_g,
        lut_out_b,
        coeff_out_r,
        coeff_out_g,
        coeff_out_b,
        lutsize as usize,
        nonlinear != 0,
        have_profile != 0,
    ) else {
        return;
    };
    let img = std::slice::from_raw_parts_mut(img, pix_len);
    retouch_adjust_levels(
        img,
        width as usize,
        height as usize,
        ch as usize,
        left,
        right,
        in_inv_gamma,
        profile.as_ref(),
    );
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod levels_tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    /// 4x4 identity-ish profile matrices (row-major `dt_colormatrix_t`).
    fn identity_matrix() -> Vec<f32> {
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ]
    }

    /// Two-entry identity LUT: `lut[0] >= 0` (active) and identity on [0,1).
    fn identity_lut() -> Vec<f32> {
        vec![0.0, 1.0]
    }

    /// Identity exponential fit: `eval_exp([1,1,1], v) == v`.
    fn identity_coeff() -> Vec<f32> {
        vec![1.0, 1.0, 1.0]
    }

    /// Nonlinear profile view over owned tables (identity TRCs, identity
    /// matrices): forward = matrix + XYZ→Lab, backward = inverse.
    fn nonlinear_profile<'a>(
        m_in: &'a [f32],
        m_out: &'a [f32],
        luts_in: [&'a [f32]; 3],
        coeffs_in: [&'a [f32]; 3],
        luts_out: [&'a [f32]; 3],
        coeffs_out: [&'a [f32]; 3],
    ) -> RetouchProfile<'a> {
        RetouchProfile {
            matrix_in: m_in,
            matrix_out: m_out,
            lut_in: luts_in,
            coeff_in: coeffs_in,
            lut_out: luts_out,
            coeff_out: coeffs_out,
            lutsize: 2,
            nonlinear: true,
        }
    }

    fn linear_profile<'a>(m_in: &'a [f32], m_out: &'a [f32]) -> RetouchProfile<'a> {
        RetouchProfile {
            matrix_in: m_in,
            matrix_out: m_out,
            lut_in: [&[], &[], &[]],
            coeff_in: [&[], &[], &[]],
            lut_out: [&[], &[], &[]],
            coeff_out: [&[], &[], &[]],
            lutsize: 0,
            nonlinear: false,
        }
    }

    /// C-order level derivation: `delta/mid/tmp` then `10^tmp`.
    fn gamma_for(left: f32, middle: f32, right: f32) -> f32 {
        let delta = (right - left) / 2.0;
        let mid = left + delta;
        10.0f32.powf((middle - mid) / delta)
    }

    /// Element-wise bit comparison for output vectors: catches signed-zero
    /// and NaN differences that `assert_eq!` on `f32` would miss
    /// (`-0.0 == 0.0`, and `NaN != NaN` fails even when both sides are
    /// identically poisoned).
    fn assert_bits_eq(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "lane {i}: {x:?} vs {y:?}");
        }
    }

    #[test]
    fn stats_matches_reference_nonlinear_profile() {
        let (w, h, ch) = (7usize, 5usize, 4usize);
        let m_in = identity_matrix();
        let m_out = identity_matrix();
        let (lr, lg, lb) = (identity_lut(), identity_lut(), identity_lut());
        let (cr, cg, cb) = (identity_coeff(), identity_coeff(), identity_coeff());
        let p = nonlinear_profile(
            &m_in, &m_out, [&lr, &lg, &lb], [&cr, &cg, &cb], [&lr, &lg, &lb],
            [&cr, &cg, &cb],
        );
        let mut img = vec![0.0f32; w * h * ch];
        lcg_fill(&mut img, 0xC4_190, 1.5);
        let mut a = [0.0f32; 3];
        let mut b = [-99.0f32; 3];
        retouch_process_stats(&img, w, h, ch, &mut a, Some(&p));
        ref_retouch_process_stats(&img, w, h, ch, &mut b, Some(&p));
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
        // sanity: min <= mean <= max, all finite for finite input.
        assert!(a[0] <= a[1] && a[1] <= a[2]);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stats_matches_reference_linear_and_fallback() {
        let (w, h, ch) = (6usize, 4usize, 4usize);
        let m_in = identity_matrix();
        let m_out = identity_matrix();
        let lin = linear_profile(&m_in, &m_out);
        let mut img = vec![0.0f32; w * h * ch];
        lcg_fill(&mut img, 77, 2.0);
        for profile in [Some(&lin), None] {
            let mut a = [0.0f32; 3];
            let mut b = [-99.0f32; 3];
            retouch_process_stats(&img, w, h, ch, &mut a, profile);
            ref_retouch_process_stats(&img, w, h, ch, &mut b, profile);
            assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
        }
    }

    #[test]
    fn stats_constant_grey_pins_reduction_semantics() {
        // Uniform grey: min == mean == max == L/100 exactly.
        let (w, h, ch) = (3usize, 2usize, 4usize);
        let px = [0.18f32, 0.18, 0.18, 0.7];
        let img = vec![px; w * h].concat();
        let expected = xyz_to_lab(srgb_to_xyz_d50(px))[0] / 100.0;
        let mut levels = [-1.0f32; 3];
        retouch_process_stats(&img, w, h, ch, &mut levels, None);
        assert_eq!(levels[0].to_bits(), expected.to_bits());
        assert_eq!(levels[1].to_bits(), expected.to_bits());
        assert_eq!(levels[2].to_bits(), expected.to_bits());
    }

    #[test]
    fn stats_and_levels_black_golden_vector() {
        // C-independent golden: all-black fallback image. `dt_linearRGB_to_XYZ`
        // (colorspaces_inline_conversions.h:515-518) maps zero RGB through the
        // sRGB→XYZ matrix (sRGB_to_xyz_transposed, :505-508) to exactly zero
        // XYZ (sums of `c * 0.0`); `dt_XYZ_to_Lab` (:148-180) takes the linear
        // branch (`0 < epsilon`, epsilon/kappa at :150-151), giving
        // `f = 16/116` per lane and `L = 116*(16/116)-16 = +0.0` in f32
        // arithmetic, with lane 3 zeroed by the vector build (:161-179). The
        // `rt_process_stats` reduction (retouch.c:3082-3108) then yields
        // `[0/100, 0/100, 0/100]`. Expected bits are hard-coded here — not
        // derived from the shared Rust helpers under test.
        let (w, h, ch) = (2usize, 2usize, 4usize);
        let img = vec![0.0f32; w * h * ch];
        let mut levels = [-1.0f32; 3];
        retouch_process_stats(&img, w, h, ch, &mut levels, None);
        assert_bits_eq(&levels, &[0.0, 0.0, 0.0]);
        // Round trip golden: `dt_Lab_to_XYZ` (:192-206, via `lab_f_inv`
        // :183-188) maps Lab(0,0,0,0) back to zero XYZ
        // (`(116*(16/116)-16)/kappa = 0`, times the D50 white point :144),
        // and `dt_XYZ_to_linearRGB` (:535-538, xyz_to_srgb_transposed :510-513)
        // maps that to zero RGB, so `rt_adjust_levels` (retouch.c:3135-3187)
        // with a below-range clamp leaves black — alpha included — at exactly
        // `+0.0` per lane.
        let mut black = vec![0.0f32, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25];
        retouch_adjust_levels(&mut black, 2, 1, 4, 0.1, 0.9, 1.0, None);
        assert_bits_eq(&black, &[0.0; 8]);
    }

    #[test]
    fn stats_nan_poisoning_matches_glib_ordering() {
        // The reduction mirrors the glib MIN/MAX macros (`a<b ? a : b`), not
        // `f32::min`/`max`: `MIN(x, NaN)` is NaN (comparison false, take `b`),
        // but `MIN(NaN, x)` is `x`. So a NaN L poisons the running extrema,
        // yet a later finite L rescues them — while the sum, once NaN, stays
        // NaN. `f32::min`/`max` would instead ignore the NaN entirely.
        // (retouch.c:3082-3108).
        let px0 = [0.2f32, 0.2, 0.2, 1.0];
        let pxn = [f32::NAN, 0.2, 0.2, 1.0];
        let px2 = [0.4f32, 0.4, 0.4, 1.0];
        // Anchors: single-pixel levels pin each finite pixel's L/100.
        let mut l0 = [0.0f32; 3];
        let mut l2 = [0.0f32; 3];
        retouch_process_stats(&px0, 1, 1, 4, &mut l0, None);
        retouch_process_stats(&px2, 1, 1, 4, &mut l2, None);
        assert!(l0[0].is_finite() && l2[0].is_finite());
        assert_ne!(l0[0].to_bits(), l2[0].to_bits());

        // NaN last: min, mean and max are all NaN.
        let img = [px0, px2, pxn].concat();
        let mut levels = [0.0f32; 3];
        let mut reference = [0.0f32; 3];
        retouch_process_stats(&img, 3, 1, 4, &mut levels, None);
        ref_retouch_process_stats(&img, 3, 1, 4, &mut reference, None);
        assert_bits_eq(&levels, &reference);
        assert!(levels.iter().all(|v| v.is_nan()), "{levels:?}");

        // NaN in the middle: the trailing finite L rescues min/max to its own
        // value (both lanes equal L(px2)), while the mean stays NaN via the
        // poisoned sum. `f32::min` would instead report L(px0) as the min.
        let img = [px0, pxn, px2].concat();
        let mut levels = [0.0f32; 3];
        let mut reference = [0.0f32; 3];
        retouch_process_stats(&img, 3, 1, 4, &mut levels, None);
        ref_retouch_process_stats(&img, 3, 1, 4, &mut reference, None);
        assert_bits_eq(&levels, &reference);
        assert_bits_eq(&levels[0..1], &l2[0..1]);
        assert_bits_eq(&levels[2..3], &l2[2..3]);
        assert_ne!(levels[0].to_bits(), l0[0].to_bits());
        assert!(levels[1].is_nan(), "{levels:?}");
    }

    #[test]
    fn stats_degenerate_inputs_leave_levels_untouched() {
        let m_in = identity_matrix();
        let m_out = identity_matrix();
        let lin = linear_profile(&m_in, &m_out);
        let img = vec![0.5f32; 16];
        // zero width, ch below the 3 readable lanes, short buffer.
        for (w, h, ch, len) in [(0, 2, 4, 16), (2, 2, 2, 16), (4, 4, 4, 16)] {
            let mut levels = [7.0f32, 8.0, 9.0];
            retouch_process_stats(&img[..len.min(img.len())], w, h, ch, &mut levels, Some(&lin));
            assert_eq!(
                levels.map(f32::to_bits),
                [7.0f32, 8.0, 9.0].map(f32::to_bits),
                "w={w} h={h} ch={ch}"
            );
        }
        // short profile tables on a nonlinear profile.
        let short_lut = [0.0f32];
        let coeff = [1.0f32, 1.0, 1.0];
        let bad = RetouchProfile {
            matrix_in: &m_in,
            matrix_out: &m_out,
            lut_in: [&short_lut, &short_lut, &short_lut],
            coeff_in: [&coeff, &coeff, &coeff],
            lut_out: [&short_lut, &short_lut, &short_lut],
            coeff_out: [&coeff, &coeff, &coeff],
            lutsize: 2,
            nonlinear: true,
        };
        let mut levels = [7.0f32, 8.0, 9.0];
        retouch_process_stats(&img, 2, 2, 4, &mut levels, Some(&bad));
        assert_eq!(levels.map(f32::to_bits), [7.0f32, 8.0, 9.0].map(f32::to_bits));
        // short matrix (< 16 floats) is a no-op even on a linear profile.
        let short_matrix = [1.0f32; 15];
        let bad_matrix = RetouchProfile {
            matrix_in: &short_matrix,
            matrix_out: &m_out,
            lut_in: [&[], &[], &[]],
            coeff_in: [&[], &[], &[]],
            lut_out: [&[], &[], &[]],
            coeff_out: [&[], &[], &[]],
            lutsize: 0,
            nonlinear: false,
        };
        let mut levels = [7.0f32, 8.0, 9.0];
        retouch_process_stats(&img, 2, 2, 4, &mut levels, Some(&bad_matrix));
        assert_eq!(levels.map(f32::to_bits), [7.0f32, 8.0, 9.0].map(f32::to_bits));
        // short coefficient slice (< 3 floats) on a nonlinear profile.
        let lut2 = [0.0f32, 1.0];
        let short_coeff = [1.0f32, 1.0];
        let bad_coeff = RetouchProfile {
            matrix_in: &m_in,
            matrix_out: &m_out,
            lut_in: [&lut2, &lut2, &lut2],
            coeff_in: [&short_coeff, &short_coeff, &short_coeff],
            lut_out: [&lut2, &lut2, &lut2],
            coeff_out: [&short_coeff, &short_coeff, &short_coeff],
            lutsize: 2,
            nonlinear: true,
        };
        let mut levels = [7.0f32, 8.0, 9.0];
        retouch_process_stats(&img, 2, 2, 4, &mut levels, Some(&bad_coeff));
        assert_eq!(levels.map(f32::to_bits), [7.0f32, 8.0, 9.0].map(f32::to_bits));
        // nonlinear `lutsize` 0/1 can never index a LUT segment: no-op.
        for lutsize in [0, 1] {
            let bad_lutsize = RetouchProfile {
                matrix_in: &m_in,
                matrix_out: &m_out,
                lut_in: [&lut2, &lut2, &lut2],
                coeff_in: [&coeff, &coeff, &coeff],
                lut_out: [&lut2, &lut2, &lut2],
                coeff_out: [&coeff, &coeff, &coeff],
                lutsize,
                nonlinear: true,
            };
            let mut levels = [7.0f32, 8.0, 9.0];
            retouch_process_stats(&img, 2, 2, 4, &mut levels, Some(&bad_lutsize));
            assert_eq!(
                levels.map(f32::to_bits),
                [7.0f32, 8.0, 9.0].map(f32::to_bits),
                "lutsize={lutsize}"
            );
        }
    }

    #[test]
    fn levels_matches_reference_profile_and_fallback() {
        let (w, h, ch) = (7usize, 5usize, 4usize);
        let m_in = identity_matrix();
        let m_out = identity_matrix();
        let (lr, lg, lb) = (identity_lut(), identity_lut(), identity_lut());
        let (cr, cg, cb) = (identity_coeff(), identity_coeff(), identity_coeff());
        let nl = nonlinear_profile(
            &m_in, &m_out, [&lr, &lg, &lb], [&cr, &cg, &cb], [&lr, &lg, &lb],
            [&cr, &cg, &cb],
        );
        let lin = linear_profile(&m_in, &m_out);
        let (left, right) = (-1.0f32, 1.5f32);
        let gamma = gamma_for(left, 0.2, right);
        let mut base = vec![0.0f32; w * h * ch];
        lcg_fill(&mut base, 0x1E_90, 1.2);
        for profile in [Some(&nl), Some(&lin), None] {
            let mut a = base.clone();
            let mut b = base.clone();
            retouch_adjust_levels(&mut a, w, h, ch, left, right, gamma, profile);
            ref_retouch_adjust_levels(&mut b, w, h, ch, left, right, gamma, profile);
            assert_bits_eq(&a, &b);
            assert!(a.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn levels_black_clamps_to_zero_and_zeroes_alpha() {
        // Fallback path: black converts to Lab(0,0,0,0); with left = 0.1 the
        // L lane clamps to exactly 0 and the round trip yields black with
        // alpha 0 — pinning the C behavior that the forward conversion
        // destroys the input alpha (it is NOT preserved here).
        let (w, h, ch) = (2usize, 1usize, 4usize);
        let mut img = vec![0.0f32, 0.0, 0.0, 0.5, 0.6, 0.6, 0.6, 0.9];
        retouch_adjust_levels(&mut img, w, h, ch, 0.1, 0.9, 1.0, None);
        assert_bits_eq(&img[0..4], &[0.0, 0.0, 0.0, 0.0]);
        // second pixel: alpha forced to 0 even though its L survives.
        assert_eq!(img[7].to_bits(), 0.0f32.to_bits());
        assert!(img[4..7].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn levels_degenerate_inputs_leave_image_untouched() {
        let m_in = identity_matrix();
        let m_out = identity_matrix();
        let lin = linear_profile(&m_in, &m_out);
        let base = vec![0.3f32; 32];
        for (w, h, ch, len) in [(0, 2, 4, 32), (4, 2, 3, 32), (8, 8, 4, 32)] {
            let mut img = base.clone();
            let n = len.min(img.len());
            retouch_adjust_levels(&mut img[..n], w, h, ch, -1.0, 1.0, 1.0, Some(&lin));
            assert_bits_eq(&img, &base);
        }
        // short matrix (< 16 floats) is a no-op even on a linear profile.
        let short_matrix = [1.0f32; 15];
        let bad_matrix = RetouchProfile {
            matrix_in: &short_matrix,
            matrix_out: &m_out,
            lut_in: [&[], &[], &[]],
            coeff_in: [&[], &[], &[]],
            lut_out: [&[], &[], &[]],
            coeff_out: [&[], &[], &[]],
            lutsize: 0,
            nonlinear: false,
        };
        let mut img = base.clone();
        retouch_adjust_levels(&mut img, 4, 2, 4, -1.0, 1.0, 1.0, Some(&bad_matrix));
        assert_bits_eq(&img, &base);
        // short coefficient slice and nonlinear `lutsize` 0/1: no-op.
        let lut2 = [0.0f32, 1.0];
        let coeff3 = [1.0f32, 1.0, 1.0];
        let short_coeff = [1.0f32, 1.0];
        let bad_coeff = RetouchProfile {
            matrix_in: &m_in,
            matrix_out: &m_out,
            lut_in: [&lut2, &lut2, &lut2],
            coeff_in: [&short_coeff, &short_coeff, &short_coeff],
            lut_out: [&lut2, &lut2, &lut2],
            coeff_out: [&short_coeff, &short_coeff, &short_coeff],
            lutsize: 2,
            nonlinear: true,
        };
        let mut img = base.clone();
        retouch_adjust_levels(&mut img, 4, 2, 4, -1.0, 1.0, 1.0, Some(&bad_coeff));
        assert_bits_eq(&img, &base);
        for lutsize in [0, 1] {
            let bad_lutsize = RetouchProfile {
                matrix_in: &m_in,
                matrix_out: &m_out,
                lut_in: [&lut2, &lut2, &lut2],
                coeff_in: [&coeff3, &coeff3, &coeff3],
                lut_out: [&lut2, &lut2, &lut2],
                coeff_out: [&coeff3, &coeff3, &coeff3],
                lutsize,
                nonlinear: true,
            };
            let mut img = base.clone();
            retouch_adjust_levels(&mut img, 4, 2, 4, -1.0, 1.0, 1.0, Some(&bad_lutsize));
            assert_bits_eq(&img, &base);
        }
    }

    #[test]
    fn ffi_guards_reject_null_and_degenerate_dims() {
        let m_in = identity_matrix();
        let m_out = identity_matrix();
        let (lut, coeff) = (identity_lut(), identity_coeff());
        let mut img = vec![0.4f32; 64];
        let mut levels = [1.0f32, 2.0, 3.0];
        let img_ptr = img.as_ptr();
        let lv_ptr = levels.as_mut_ptr();
        let mi = m_in.as_ptr();
        let mo = m_out.as_ptr();
        let lp = lut.as_ptr();
        let cp = coeff.as_ptr();
        unsafe {
            // stats: null image, null levels, zero/negative dims.
            darkroom_retouch_process_stats(
                std::ptr::null(), 4, 4, 4, lv_ptr, mi, lp, lp, lp, cp, cp, cp, 2, 1, 1,
            );
            darkroom_retouch_process_stats(
                img_ptr, 4, 4, 4, std::ptr::null_mut(), mi, lp, lp, lp, cp, cp, cp, 2,
                1, 1,
            );
            darkroom_retouch_process_stats(
                img_ptr, 0, 4, 4, lv_ptr, mi, lp, lp, lp, cp, cp, cp, 2, 1, 1,
            );
            darkroom_retouch_process_stats(
                img_ptr, 4, -3, 4, lv_ptr, mi, lp, lp, lp, cp, cp, cp, 2, 1, 1,
            );
            // stats: profile present but matrix null; nonlinear with null LUT.
            darkroom_retouch_process_stats(
                img_ptr, 4, 4, 4, lv_ptr, std::ptr::null(), lp, lp, lp, cp, cp, cp, 2,
                1, 1,
            );
            darkroom_retouch_process_stats(
                img_ptr, 4, 4, 4, lv_ptr, mi, std::ptr::null(), lp, lp, cp, cp, cp, 2,
                1, 1,
            );
            // levels: null image, zero dims, null matrix_in on a profile path.
            darkroom_retouch_adjust_levels(
                std::ptr::null_mut(), 4, 4, 4, -1.0, 1.0, 1.0, mi, mo, lp, lp, lp,
                cp, cp, cp, lp, lp, lp, cp, cp, cp, 2, 1, 1,
            );
            darkroom_retouch_adjust_levels(
                img.as_mut_ptr(), 0, 4, 4, -1.0, 1.0, 1.0, mi, mo, lp, lp, lp,
                cp, cp, cp, lp, lp, lp, cp, cp, cp, 2, 1, 1,
            );
            darkroom_retouch_adjust_levels(
                img.as_mut_ptr(), 4, 4, 4, -1.0, 1.0, 1.0, std::ptr::null(), mo, lp,
                lp, lp, cp, cp, cp, lp, lp, lp, cp, cp, cp, 2, 1, 1,
            );
            // negative lutsize through FFI: rejected before any slice is built.
            darkroom_retouch_process_stats(
                img_ptr, 4, 4, 4, lv_ptr, mi, lp, lp, lp, cp, cp, cp, -1, 1, 1,
            );
            darkroom_retouch_adjust_levels(
                img.as_mut_ptr(), 4, 4, 4, -1.0, 1.0, 1.0, mi, mo, lp, lp, lp,
                cp, cp, cp, lp, lp, lp, cp, cp, cp, -1, 1, 1,
            );
            // nonlinear `lutsize` 0/1: no LUT segment can be indexed, no-op.
            for bad_lutsize in [0, 1] {
                darkroom_retouch_process_stats(
                    img_ptr, 4, 4, 4, lv_ptr, mi, lp, lp, lp, cp, cp, cp, bad_lutsize,
                    1, 1,
                );
                darkroom_retouch_adjust_levels(
                    img.as_mut_ptr(), 4, 4, 4, -1.0, 1.0, 1.0, mi, mo, lp, lp, lp,
                    cp, cp, cp, lp, lp, lp, cp, cp, cp, bad_lutsize, 1, 1,
                );
            }
        }
        // Every call above was rejected: levels and image bit-identical.
        assert_eq!(levels.map(f32::to_bits), [1.0f32, 2.0, 3.0].map(f32::to_bits));
        assert!(img.iter().all(|v| v.to_bits() == 0.4f32.to_bits()));
        unsafe {
            // A valid fallback call (null tables tolerated) writes finite levels.
            darkroom_retouch_process_stats(
                img_ptr, 4, 4, 4, lv_ptr, std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null(), 0, 0, 0,
            );
        }
        assert!(levels.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ffi_valid_calls_agree_with_safe_kernels() {
        let (w, h, ch) = (5usize, 3usize, 4usize);
        let m_in = identity_matrix();
        let m_out = identity_matrix();
        let (lut, coeff) = (identity_lut(), identity_coeff());
        let mut base = vec![0.0f32; w * h * ch];
        lcg_fill(&mut base, 4242, 1.1);
        let (left, right) = (-0.5f32, 1.25f32);
        let gamma = gamma_for(left, 0.1, right);
        unsafe {
            // stats, nonlinear profile.
            let mut lv_ffi = [0.0f32; 3];
            darkroom_retouch_process_stats(
                base.as_ptr(), w as i32, h as i32, ch as i32, lv_ffi.as_mut_ptr(),
                m_in.as_ptr(), lut.as_ptr(), lut.as_ptr(), lut.as_ptr(), coeff.as_ptr(),
                coeff.as_ptr(), coeff.as_ptr(), 2, 1, 1,
            );
            let p = nonlinear_profile(
                &m_in, &m_out, [&lut, &lut, &lut], [&coeff, &coeff, &coeff],
                [&lut, &lut, &lut], [&coeff, &coeff, &coeff],
            );
            let mut lv_safe = [0.0f32; 3];
            retouch_process_stats(&base, w, h, ch, &mut lv_safe, Some(&p));
            assert_eq!(lv_ffi.map(f32::to_bits), lv_safe.map(f32::to_bits));
            // levels, nonlinear profile.
            let mut a = base.clone();
            let mut b = base.clone();
            darkroom_retouch_adjust_levels(
                a.as_mut_ptr(), w as i32, h as i32, ch as i32, left, right, gamma,
                m_in.as_ptr(), m_out.as_ptr(), lut.as_ptr(), lut.as_ptr(), lut.as_ptr(),
                coeff.as_ptr(), coeff.as_ptr(), coeff.as_ptr(), lut.as_ptr(),
                lut.as_ptr(), lut.as_ptr(), coeff.as_ptr(), coeff.as_ptr(),
                coeff.as_ptr(), 2, 1, 1,
            );
            retouch_adjust_levels(&mut b, w, h, ch, left, right, gamma, Some(&p));
            assert_bits_eq(&a, &b);
        }
    }

    #[test]
    fn ffi_linear_profile_null_tables_is_valid_path() {
        // A linear profile never touches the LUT/coefficient tables, so NULL
        // table pointers with `nonlinear == 0` are a valid path (not a guard
        // rejection): both FFI entries must agree bit-exactly with the safe
        // kernels. `lutsize` is ignored on this path.
        let (w, h, ch) = (4usize, 3usize, 4usize);
        let m_in = identity_matrix();
        let m_out = identity_matrix();
        let mut base = vec![0.0f32; w * h * ch];
        lcg_fill(&mut base, 0x11E, 1.3);
        let (left, right) = (-0.5f32, 1.25f32);
        let gamma = gamma_for(left, 0.1, right);
        let null: *const f32 = std::ptr::null();
        unsafe {
            let mut lv_ffi = [0.0f32; 3];
            darkroom_retouch_process_stats(
                base.as_ptr(), w as i32, h as i32, ch as i32, lv_ffi.as_mut_ptr(),
                m_in.as_ptr(), null, null, null, null, null, null, 0, 0, 1,
            );
            let lin = linear_profile(&m_in, &m_out);
            let mut lv_safe = [0.0f32; 3];
            retouch_process_stats(&base, w, h, ch, &mut lv_safe, Some(&lin));
            assert_bits_eq(&lv_ffi, &lv_safe);
            assert!(lv_ffi.iter().all(|v| v.is_finite()));
            let mut a = base.clone();
            let mut b = base.clone();
            darkroom_retouch_adjust_levels(
                a.as_mut_ptr(), w as i32, h as i32, ch as i32, left, right, gamma,
                m_in.as_ptr(), m_out.as_ptr(), null, null, null, null, null, null,
                null, null, null, null, null, null, 0, 0, 1,
            );
            retouch_adjust_levels(&mut b, w, h, ch, left, right, gamma, Some(&lin));
            assert_bits_eq(&a, &b);
        }
    }
}
