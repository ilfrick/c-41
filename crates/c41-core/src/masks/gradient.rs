//! Gradient drawn-mask rendering — port of the OMP loops in
//! `src/develop/masks/gradient.c`.
//!
//! Both `_gradient_get_mask` (whole-pipe) and `_gradient_get_mask_roi` share
//! the same four-loop structure: grid-points fill → LUT build → in-place value
//! evaluation → bilinear splat. Loop 1 (grid fill) delegates to the shared
//! [`fill_grid_points`] helper (with `iscale=1.0` for the whole-pipe path and
//! the ROI's `iscale` for the ROI path). Loop 4 (interpolation) delegates to
//! [`interpolate_into_buffer`]. Only the LUT builder and the values evaluator
//! are gradient-specific.
//!
//! Bit-exactness: the LUT uses the C `erff` via FFI (glibc `erff` on the
//! Linux target — same symbol the C code calls), so sigmoidal gradients are
//! bit-identical. The values loop replicates the C rotation formula
//! `x0 = (cosv·x + sinv·y − xoffset)·hwscale`,
//! `distance = y0 − curvature·x0²`, with the same inclusive `<=` / `>=`
//! clamping before the LUT lookup.

use super::{fill_grid_points, interpolate_into_buffer, GRADIENT_STATE_LINEAR};

// `erff` from glibc — same symbol the C gradient code calls via `<math.h>`,
// guaranteeing bit-identical sigmoidal LUT values. On Linux it lives in
// `libc.so.6`; the Rust linker resolves it from the system libc.
extern "C" {
    fn erff(x: f32) -> f32;
}

/// Build the sigmoidal/linear LUT (loop 2 of both `_gradient_get_mask` and
/// `_gradient_get_mask_roi`).
///
/// For each index `n`, `distance = (n - lutmax) * hwscale`; the LUT value is
/// `0.5 + 0.5 * (normf * distance)` for linear state,
/// `0.5 + 0.5 * erff(distance / compression)` for sigmoidal state,
/// clamped to [0, 1].
pub fn fill_lut(
    lut: &mut [f32],
    lutmax: i32,
    hwscale: f32,
    normf: f32,
    compression: f32,
    is_linear: bool,
) {
    for (n, slot) in lut.iter_mut().enumerate() {
        let distance = (n as i32 - lutmax) as f32 * hwscale;
        let value = if is_linear {
            0.5f32 + 0.5f32 * (normf * distance)
        } else {
            let e = unsafe { erff(distance / compression) };
            0.5f32 + 0.5f32 * e
        };
        *slot = value.clamp(0.0f32, 1.0f32);
    }
}

/// Bilinear LUT lookup, matching `dt_gradient_lookup` from gradient.c:1102.
/// `i` is the fractional bin index; `lutmax` offsets into the centred LUT
/// (`lut[lutmax]` is the zero-distance sample). The caller's `<= ±4·compression`
/// clamping guarantees both indices are in range.
#[inline]
fn gradient_lookup(lut: &[f32], lutmax: i32, i: f32) -> f32 {
    let bin0 = i as i32; // truncates toward zero, matching C's `(int)i`
    let bin1 = bin0 + 1;
    let f = i - bin0 as f32; // C: `i - bin0`
    let idx0 = (lutmax + bin0) as usize;
    let idx1 = (lutmax + bin1) as usize;
    // C order: lut[bin1]*f + lut[bin0]*(1-f)
    lut[idx1] * f + lut[idx0] * (1.0f32 - f)
}

/// In-place mask-value evaluation at back-transformed grid points (loop 3 of
/// both paths). Reads (x, y) from even/odd lanes, writes the masked value to
/// the even lane — exactly as the C re-uses the `points` array.
pub fn fill_values_in_place(
    points: &mut [f32],
    count: usize,
    lut: &[f32],
    lutmax: i32,
    cosv: f32,
    sinv: f32,
    xoffset: f32,
    yoffset: f32,
    hwscale: f32,
    ihwscale: f32,
    curvature: f32,
    compression: f32,
) {
    assert!(count * 2 <= points.len(), "need 2*count lanes for count grid points");
    let neg_clip = -4.0f32 * compression;
    let pos_clip = 4.0f32 * compression;
    for idx in 0..count {
        let x = points[2 * idx];
        let y = points[2 * idx + 1];
        let x0 = (cosv * x + sinv * y - xoffset) * hwscale;
        let y0 = (sinv * x - cosv * y - yoffset) * hwscale;
        let distance = y0 - curvature * x0 * x0;
        let value = if distance <= neg_clip {
            0.0f32
        } else if distance >= pos_clip {
            1.0f32
        } else {
            gradient_lookup(lut, lutmax, distance * ihwscale)
        };
        points[2 * idx] = value;
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `points` must hold `2·bbw·bbh` floats; wraps [`fill_grid_points`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_gradient_grid(
    points: *mut f32,
    bbw: usize,
    bbh: usize,
    bbxm: i32,
    bbym: i32,
    px: i32,
    py: i32,
    iscale: f32,
    grid: i32,
) {
    if points.is_null() || bbw == 0 || bbh == 0 || grid < 1 {
        return;
    }
    let Some(len) = bbw.checked_mul(bbh) else { return };
    if len > i32::MAX as usize {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(points, 2 * len);
    fill_grid_points(slice, bbw, bbh, bbxm, bbym, px, py, iscale, grid);
}

/// # Safety
/// `lut` must hold `lutsize` floats; wraps [`fill_lut`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_gradient_lut(
    lut: *mut f32,
    lutsize: usize,
    lutmax: i32,
    hwscale: f32,
    normf: f32,
    compression: f32,
    state: i32,
) {
    if lut.is_null() || lutsize == 0 || compression < 0.001f32 {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(lut, lutsize);
    fill_lut(slice, lutmax, hwscale, normf, compression, state == GRADIENT_STATE_LINEAR);
}

/// # Safety
/// `points` must hold `2·count` writable floats; `lut` must hold
/// `2·lutmax + 2` floats (the C-side `lutsize`). All scalars must be the
/// pre-computed gradient parameters from `gradient.c`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_masks_gradient_values(
    points: *mut f32,
    count: usize,
    lut: *const f32,
    lutmax: i32,
    cosv: f32,
    sinv: f32,
    xoffset: f32,
    yoffset: f32,
    hwscale: f32,
    ihwscale: f32,
    curvature: f32,
    compression: f32,
) {
    if points.is_null() || lut.is_null() || count == 0 || lutmax < 0 {
        return;
    }
    if count > i32::MAX as usize / 2 {
        return;
    }
    // lutsize = 2*lutmax + 2 (from the C alloc)
    let lutsize = (2 * lutmax as usize).saturating_add(2);
    if lutsize > i32::MAX as usize {
        return;
    }
    let points = std::slice::from_raw_parts_mut(points, count * 2);
    let lut = std::slice::from_raw_parts(lut, lutsize);
    fill_values_in_place(
        points, count, lut, lutmax, cosv, sinv, xoffset, yoffset,
        hwscale, ihwscale, curvature, compression,
    );
}

/// # Safety
/// `buffer` must hold `w·height` floats (only rows in `[start_j,end_j) ×
/// cols [start_i,end_i)` are written); `points` must hold `2·bbw·bbh` floats.
/// Same caller invariant as circle/ellipse interp.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_masks_gradient_interp(
    buffer: *mut f32,
    w: usize,
    height: usize,
    points: *const f32,
    bbw: usize,
    bbh: usize,
    start_i: i32,
    end_i: i32,
    start_j: i32,
    end_j: i32,
    grid: i32,
) {
    if buffer.is_null()
        || points.is_null()
        || w == 0
        || height == 0
        || bbw == 0
        || bbh == 0
        || grid < 1
        || start_i < 0
        || start_j < 0
        || end_i < start_i
        || end_j < start_j
        || end_i > w as i32
        || end_j > height as i32
    {
        return;
    }
    // Caller invariant: neighbour lookups stay inside the bbox.
    let max_mi = (end_i - 1).div_euclid(grid) - start_i.div_euclid(grid);
    let max_mj = (end_j - 1).div_euclid(grid) - start_j.div_euclid(grid);
    if max_mi + 1 >= bbw as i32 || max_mj + 1 >= bbh as i32 {
        return;
    }
    let Some(len) = w.checked_mul(height) else { return };
    let Some(plen) = bbw.checked_mul(bbh) else { return };
    let buffer = std::slice::from_raw_parts_mut(buffer, len);
    let points = std::slice::from_raw_parts(points, plen * 2);
    interpolate_into_buffer(buffer, w, points, bbw, start_i, end_i, start_j, end_j, grid);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference LUT builder straight from the C text, using the same `erff`
    /// via FFI so it serves as a structural cross-check.
    fn ref_lut(
        lut: &mut [f32],
        lutmax: i32,
        hwscale: f32,
        normf: f32,
        compression: f32,
        is_linear: bool,
    ) {
        for (n, slot) in lut.iter_mut().enumerate() {
            let distance = (n as i32 - lutmax) as f32 * hwscale;
            let value = if is_linear {
                0.5f32 + 0.5f32 * (normf * distance)
            } else {
                let e = unsafe { erff(distance / compression) };
                0.5f32 + 0.5f32 * e
            };
            *slot = value.clamp(0.0f32, 1.0f32);
        }
    }

    /// Reference lookup matching `dt_gradient_lookup` in gradient.c:1102.
    fn ref_lookup(lut: &[f32], lutmax: i32, i: f32) -> f32 {
        let bin0 = i as i32;
        let bin1 = bin0 + 1;
        let f = i - bin0 as f32;
        let idx0 = (lutmax + bin0) as usize;
        let idx1 = (lutmax + bin1) as usize;
        lut[idx1] * f + lut[idx0] * (1.0f32 - f)
    }

    /// Reference values-in-place straight from the C text (gradient.c:1208).
    fn ref_values_in_place(
        points: &mut [f32],
        count: usize,
        lut: &[f32],
        lutmax: i32,
        cosv: f32,
        sinv: f32,
        xoffset: f32,
        yoffset: f32,
        hwscale: f32,
        ihwscale: f32,
        curvature: f32,
        compression: f32,
    ) {
        let neg = -4.0f32 * compression;
        let pos = 4.0f32 * compression;
        for idx in 0..count {
            let x = points[2 * idx];
            let y = points[2 * idx + 1];
            let x0 = (cosv * x + sinv * y - xoffset) * hwscale;
            let y0 = (sinv * x - cosv * y - yoffset) * hwscale;
            let distance = y0 - curvature * x0 * x0;
            let val = if distance <= neg {
                0.0f32
            } else if distance >= pos {
                1.0f32
            } else {
                ref_lookup(lut, lutmax, distance * ihwscale)
            };
            points[2 * idx] = val;
        }
    }

    #[test]
    fn erff_matches_known_values() {
        assert_eq!(unsafe { erff(0.0) }, 0.0);
        let v = unsafe { erff(0.5) };
        assert!((v - 0.5204999).abs() < 1e-6, "erff(0.5) = {v}");
        assert!((unsafe { erff(-0.5) } - (-v)).abs() < 1e-6);
    }

    #[test]
    fn fill_lut_linear_matches_reference() {
        let lutmax = 10i32;
        let lutsize = (2 * lutmax + 2) as usize;
        let mut lut = vec![0f32; lutsize];
        let mut ref_lutbuf = vec![0f32; lutsize];
        let (hwscale, normf, compression) = (0.01f32, 100.0, 0.5f32);
        fill_lut(&mut lut, lutmax, hwscale, normf, compression, true);
        ref_lut(&mut ref_lutbuf, lutmax, hwscale, normf, compression, true);
        for (a, b) in lut.iter().zip(ref_lutbuf.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "linear LUT mismatch");
        }
    }

    #[test]
    fn fill_lut_sigmoidal_matches_reference() {
        let lutmax = 17i32;
        let lutsize = (2 * lutmax + 2) as usize;
        let mut lut = vec![0f32; lutsize];
        let mut ref_lutbuf = vec![0f32; lutsize];
        let (hwscale, normf, compression) = (0.02f32, 50.0, 1.0f32);
        fill_lut(&mut lut, lutmax, hwscale, normf, compression, false);
        ref_lut(&mut ref_lutbuf, lutmax, hwscale, normf, compression, false);
        for (a, b) in lut.iter().zip(ref_lutbuf.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "sigmoidal LUT mismatch");
        }
    }

    #[test]
    fn fill_lut_clamps_extremes() {
        let lutmax = 5i32;
        let lutsize = (2 * lutmax + 2) as usize;
        let mut lut = vec![0f32; lutsize];
        let (hwscale, normf, compression) = (10.0f32, 1.0, 1.0f32);
        fill_lut(&mut lut, lutmax, hwscale, normf, compression, false);
        // n=0 → distance = -lutmax * hwscale = -50 → erff(-50)≈-1 → value≈0
        assert!(lut[0] <= 1e-6);
        // n=lutsize-1 → distance = +50 → erff(50)≈1 → value≈1
        assert!(lut[lutsize - 1] >= 1.0 - 1e-6);
        // centre → distance = 0 → erff(0)=0 → value = 0.5
        assert!((lut[lutmax as usize] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gradient_lookup_matches_reference() {
        let lutmax = 8i32;
        let lutsize = (2 * lutmax + 2) as usize;
        let mut lut = vec![0f32; lutsize];
        super::super::test_util::lcg_fill(&mut lut, 0xE13CE1, 1.0);
        for i in [-3.7f32, -1.5, 0.0, 0.25, 1.9, 4.5, 7.99] {
            let result = gradient_lookup(&lut, lutmax, i);
            let expect = ref_lookup(&lut, lutmax, i);
            assert_eq!(result.to_bits(), expect.to_bits(), "lookup mismatch at i={i}");
        }
    }

    #[test]
    fn fill_values_in_place_matches_reference_over_lcg_points() {
        let n = 256usize;
        let mut points = vec![0f32; 2 * n];
        super::super::test_util::lcg_fill(&mut points, 0xDEADBEEF, 800.0);
        let snapshot = points.clone();
        // Parameters must be self-consistent: lutmax = ceil(4 * compression * ihwscale),
        // and hwscale/compression must match between fill_lut and fill_values_in_place.
        let compression = 1.0f32;
        let hwscale = 0.02f32;
        let ihwscale = 1.0f32 / hwscale; // 50.0
        let lutmax = (4.0f32 * compression * ihwscale).ceil() as i32; // 200
        let lutsize = (2 * lutmax + 2) as usize;
        let mut lut = vec![0f32; lutsize];
        let normf = 1.0f32 / compression;
        fill_lut(&mut lut, lutmax, hwscale, normf, compression, false);
        let (cosv, sinv, xoffset, yoffset) = (0.8f32, 0.6, 100.0, 50.0);
        let curvature = 0.001f32;

        fill_values_in_place(
            &mut points, n, &lut, lutmax, cosv, sinv, xoffset, yoffset,
            hwscale, ihwscale, curvature, compression,
        );

        let mut ref_points = snapshot.clone();
        ref_values_in_place(
            &mut ref_points, n, &lut, lutmax, cosv, sinv, xoffset, yoffset,
            hwscale, ihwscale, curvature, compression,
        );
        for k in 0..n {
            assert_eq!(points[2 * k].to_bits(), ref_points[2 * k].to_bits(), "even lane {k}");
            // odd lanes (y coords) are untouched
            assert_eq!(points[2 * k + 1].to_bits(), ref_points[2 * k + 1].to_bits(), "odd lane {k}");
        }
    }

    #[test]
    fn fill_values_in_place_clamps_outside() {
        // Points far from centre → distance exceeds ±4·compression → value clamped
        let n = 4usize;
        let mut points = vec![0f32; 2 * n];
        // Use cosv=0, sinv=1 so y0 = x * hwscale (positive for positive x) → clamped to 1.0
        for k in 0..n {
            points[2 * k] = 100.0;
            points[2 * k + 1] = 100.0;
        }
        let lutmax = 5i32;
        let lutsize = (2 * lutmax + 2) as usize;
        let mut lut = vec![0f32; lutsize];
        fill_lut(&mut lut, lutmax, 0.01, 100.0, 1.0, false);

        let compression = 1.0f32;
        let cosv = 0.0f32;
        let sinv = 1.0f32;
        let xoffset = 0.0f32;
        let yoffset = 0.0f32;
        let hwscale = 1.0f32;
        let ihwscale = 1.0f32;
        let curvature = 0.0f32;

        fill_values_in_place(
            &mut points, n, &lut, lutmax, cosv, sinv, xoffset, yoffset,
            hwscale, ihwscale, curvature, compression,
        );
        // x0 = (0*100 + 1*100)*1 = 100, y0 = (1*100 - 0*100)*1 = 100
        // distance = 100 - 0 = 100 → >= 4*compression=4 → value = 1.0
        for k in 0..n {
            assert_eq!(points[2 * k], 1.0, "point {k} should be clamped to 1");
        }
    }

    #[test]
    fn fill_grid_points_matches_c_indexing_without_scale() {
        // Whole-pipe path: iscale=1.0, bbxm=0, bbym=0
        let (gw, gh) = (5usize, 3usize);
        let mut pts = vec![0f32; 2 * gw * gh];
        fill_grid_points(&mut pts, gw, gh, 0, 0, 200, 150, 1.0, 8);
        for j in 0..gh as i32 {
            for i in 0..gw as i32 {
                let index = (j * gw as i32 + i) as usize;
                assert_eq!(pts[2 * index], ((8 * i + 200) as f32) * 1.0);
                assert_eq!(pts[2 * index + 1], ((8 * j + 150) as f32) * 1.0);
            }
        }
    }

    #[test]
    fn ffi_exports_round_trip_through_c_abi() {
        unsafe {
            // grid fill
            let (gw, gh) = (7usize, 5usize);
            let mut pts = vec![0f32; 2 * gw * gh];
            darkroom_masks_gradient_grid(pts.as_mut_ptr(), gw, gh, 0, 0, 30, 20, 0.5, 8);
            let mut ref_pts = vec![0f32; 2 * gw * gh];
            fill_grid_points(&mut ref_pts, gw, gh, 0, 0, 30, 20, 0.5, 8);
            assert_eq!(pts, ref_pts);

            // LUT + values: single consistent parameter set — in the C code
            // hwscale, ihwscale=1/hwscale, compression, and lutmax=ceil(4*compression*ihwscale)
            // are shared between LUT build and values evaluation.
            let hwscale = 0.01f32;
            let ihwscale = 100.0f32;
            let compression = 1.0f32;
            let lutmax = (4.0f32 * compression * ihwscale).ceil() as i32; // 400
            let lutsize = (2 * lutmax + 2) as usize;
            let normf = 1.0f32 / compression;

            // LUT fill (sigmoidal, state=2)
            let mut lut = vec![0f32; lutsize];
            darkroom_masks_gradient_lut(lut.as_mut_ptr(), lutsize, lutmax, hwscale, normf, compression, 2);
            let mut ref_lutbuf = vec![0f32; lutsize];
            fill_lut(&mut ref_lutbuf, lutmax, hwscale, normf, compression, false);
            for (a, b) in lut.iter().zip(ref_lutbuf.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }

            // LUT fill (linear, state=1) should match is_linear=true
            let mut lut_lin = vec![0f32; lutsize];
            darkroom_masks_gradient_lut(lut_lin.as_mut_ptr(), lutsize, lutmax, hwscale, normf, compression, 1);
            let mut ref_lut_lin = vec![0f32; lutsize];
            fill_lut(&mut ref_lut_lin, lutmax, hwscale, normf, compression, true);
            for (a, b) in lut_lin.iter().zip(ref_lut_lin.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }

            // values in-place — same hwscale/compression/lutmax as the LUT above
            let n = 96usize;
            let mut vpts = vec![0f32; 2 * n];
            super::super::test_util::lcg_fill(&mut vpts, 0x1337, 2000.0);
            let snap = vpts.clone();
            darkroom_masks_gradient_values(
                vpts.as_mut_ptr(), n, lut.as_ptr(), lutmax,
                0.6f32, 0.8f32, 120.0, 80.0, hwscale, ihwscale, 0.0005, compression,
            );
            let mut ref_vpts = snap.clone();
            fill_values_in_place(
                &mut ref_vpts, n, &lut, lutmax,
                0.6f32, 0.8f32, 120.0, 80.0, hwscale, ihwscale, 0.0005, compression,
            );
            assert_eq!(vpts, ref_vpts);

            // null guards refuse without panicking
            darkroom_masks_gradient_grid(std::ptr::null_mut(), 4, 4, 0, 0, 0, 0, 1.0, 1);
            darkroom_masks_gradient_lut(std::ptr::null_mut(), 4, 2, 0.1, 10.0, 1.0, 1);
            darkroom_masks_gradient_values(
                std::ptr::null_mut(), 4, lut.as_ptr(), lutmax,
                1.0, 0.0, 0.0, 0.0, 0.001, 1000.0, 0.0, 1.0,
            );
            darkroom_masks_gradient_interp(
                std::ptr::null_mut(), 4, 4, std::ptr::null(), 4, 4, 0, 4, 0, 4, 2,
            );
        }
    }
}
