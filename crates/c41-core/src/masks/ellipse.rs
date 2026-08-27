//! Ellipse drawn-mask rendering — port of the OMP loops in
//! `src/develop/masks/ellipse.c`.
//!
//! Whole-pipe path (`_ellipse_get_mask`): coord-grid fill + `_fill_mask` with
//! `out_scale=0`.
//! ROI path (`_ellipse_get_mask_roi`): outline point-count formula + outline
//! loop + grid-points fill + `_fill_mask` with `out_scale=1` (in-place even
//! lanes) + bilinear splat into the output buffer.
//!
//! The rotation/axis-swap branch (`if radius[0]>=radius[1] ... else ...`) and
//! the `deg2radf`/`cosf`/`sinf` pre-computation live in C — the Rust kernels
//! receive the resolved `a, b, ta, tb, alpha, cosa, sina` scalars.

use super::{fill_coord_grid, fill_grid_points, interpolate_into_buffer, DT_2PI_F};

/// Whole-pipe form, `_fill_mask(.., out_scale=0)`: write mask values to a
/// separate output buffer (one value per point pair).
///
/// For each point the unit direction is rotation-corrected (`-alpha`) then
/// projected against the inner `(a,b)` and outer `(ta,tb)` ellipses to get
/// `radius2`/`total2`; the value is `sqf(CLIP((total2-l2)/(total2-radius2)))`.
pub fn fill_mask(
    bufptr: &mut [f32],
    points: &[f32],
    center_x: f32,
    center_y: f32,
    a: f32,
    b: f32,
    ta: f32,
    tb: f32,
    alpha: f32,
) {
    assert_eq!(points.len(), bufptr.len() * 2, "one point pair per output");
    fill_mask_into(bufptr, points, 0, center_x, center_y, a, b, ta, tb, alpha);
}

/// ROI in-place form, `_fill_mask(.., out_scale=1)`: read both lanes
/// (x,y coordinates) and write the mask value to the even lane (`i << 1`),
/// reusing the points array exactly like the C does. Since `points[2*i]` is
/// read and Copy'd before `points[i << 1]` is written in the next statement,
/// NLL permits the in-place aliasing.
pub fn fill_mask_in_place(
    points: &mut [f32],
    count: usize,
    center_x: f32,
    center_y: f32,
    a: f32,
    b: f32,
    ta: f32,
    tb: f32,
    alpha: f32,
) {
    assert!(count * 2 <= points.len(), "need 2*count lanes for count points");
    let a2 = a * a;
    let b2 = b * b;
    let ta2 = ta * ta;
    let tb2 = tb * tb;
    let cos_alpha = alpha.cos();
    let sin_alpha = alpha.sin();
    for i in 0..count {
        let x = points[2 * i] - center_x;
        let y = points[2 * i + 1] - center_y;
        let l2 = x * x + y * y;
        let l = l2.sqrt();
        // C's `l ? x/l : 0` — l != 0.0 is true for NaN too, matching C truthiness.
        let x_norm = if l != 0.0 { x / l } else { 0.0 };
        let y_norm = if l != 0.0 { y / l } else { 1.0 };
        let x_rot = x_norm * cos_alpha + y_norm * sin_alpha;
        let y_rot = -x_norm * sin_alpha + y_norm * cos_alpha;
        let cosv2 = x_rot * x_rot;
        let sinv2 = y_rot * y_rot;
        let radius2 = a2 * b2 / (a2 * sinv2 + b2 * cosv2);
        let total2 = ta2 * tb2 / (ta2 * sinv2 + tb2 * cosv2);
        let ratio = (total2 - l2) / (total2 - radius2);
        let f = ratio.clamp(0.0, 1.0);
        points[i << 1] = f * f;
    }
}

/// Shared inner kernel — writes to `bufptr[i << out_scale]`. Factored out of
/// [`fill_mask`] and [`fill_mask_in_place`] for bit-exactness against the C
/// `_fill_mask` body. The aliasing case (in-place) is handled by the caller
/// copying inputs first (see [`fill_mask_in_place`]).
fn fill_mask_into(
    bufptr: &mut [f32],
    points: &[f32],
    out_scale: u32,
    center_x: f32,
    center_y: f32,
    a: f32,
    b: f32,
    ta: f32,
    tb: f32,
    alpha: f32,
) {
    let a2 = a * a;
    let b2 = b * b;
    let ta2 = ta * ta;
    let tb2 = tb * tb;
    let cos_alpha = alpha.cos();
    let sin_alpha = alpha.sin();
    let n = bufptr.len() >> out_scale;
    assert_eq!(points.len(), n * 2);
    for i in 0..n {
        let x = points[2 * i] - center_x;
        let y = points[2 * i + 1] - center_y;
        let l2 = x * x + y * y;
        let l = l2.sqrt();
        // C's `l ? x/l : 0` — l != 0.0 is true for NaN too, matching C truthiness.
        let x_norm = if l != 0.0 { x / l } else { 0.0 };
        let y_norm = if l != 0.0 { y / l } else { 1.0 };
        let x_rot = x_norm * cos_alpha + y_norm * sin_alpha;
        let y_rot = -x_norm * sin_alpha + y_norm * cos_alpha;
        let cosv2 = x_rot * x_rot;
        let sinv2 = y_rot * y_rot;
        let radius2 = a2 * b2 / (a2 * sinv2 + b2 * cosv2);
        let total2 = ta2 * tb2 / (ta2 * sinv2 + tb2 * cosv2);
        let ratio = (total2 - l2) / (total2 - radius2);
        let f = ratio.clamp(0.0, 1.0);
        bufptr[i << out_scale] = f * f;
    }
}

/// Parametric ellipse outline (ROI path, no eight-fold symmetry — the ellipse
/// can be sheared by the pixelpipe so symmetry doesn't hold). Writes `ellpts`
/// (x,y) pairs into `ell`.
pub fn fill_outline(ell: &mut [f32], center_x: f32, center_y: f32, ta: f32, tb: f32,
                    cosa: f32, sina: f32) {
    let ellpts = ell.len() / 2;
    assert_eq!(ell.len(), 2 * ellpts);
    for n in 0..ellpts {
        let phi = DT_2PI_F * n as f32 / ellpts as f32;
        let cosp = phi.cos();
        let sinp = phi.sin();
        ell[2 * n] = center_x + ta * cosa * cosp - tb * sina * sinp;
        ell[2 * n + 1] = center_y + ta * sina * cosp + tb * cosa * sinp;
    }
}

/// Point count for the outline loop — the Ramanujan arc-length approximation
/// from `_ellipse_get_mask_roi`. Uses `M_PI` (double) exactly as the C does:
/// `(int)(M_PI * (ta+tb) * (1 + 3λ²/(10+√(4-3λ²))))`, then `MIN(360, l)`.
pub fn outline_point_count(ta: f32, tb: f32) -> usize {
    let lambda = (ta - tb) / (ta + tb);
    let inner = 3.0f32 * lambda * lambda;
    let denom = 10.0f32 + (4.0f32 - inner).sqrt();
    let scale = 1.0f32 + inner / denom;
    // M_PI is double in C; promote the float operands at the multiply.
    let l = (std::f64::consts::PI * (ta + tb) as f64 * scale as f64) as i32;
    let ellpts = (360i32).min(l);
    if ellpts <= 0 { 0 } else { ellpts as usize }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `points` must hold `2·w·h` floats; see [`fill_coord_grid`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_ellipse_coord_grid(
    points: *mut f32,
    w: usize,
    h: usize,
    pos_x: f32,
    pos_y: f32,
) {
    if points.is_null() || w == 0 || h == 0 || w > i32::MAX as usize || h > i32::MAX as usize {
        return;
    }
    let Some(len) = w.checked_mul(h) else { return };
    let slice = std::slice::from_raw_parts_mut(points, 2 * len);
    fill_coord_grid(slice, w, h, pos_x, pos_y);
}

/// # Safety
/// `bufptr` must hold `n` floats, `points` `2·n` floats; see [`fill_mask`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_ellipse_fill(
    bufptr: *mut f32,
    points: *const f32,
    n: usize,
    center_x: f32,
    center_y: f32,
    a: f32,
    b: f32,
    ta: f32,
    tb: f32,
    alpha: f32,
) {
    if bufptr.is_null() || points.is_null() || n == 0 {
        return;
    }
    let buffer = std::slice::from_raw_parts_mut(bufptr, n);
    let points = std::slice::from_raw_parts(points, n * 2);
    fill_mask(buffer, points, center_x, center_y, a, b, ta, tb, alpha);
}

/// # Safety
/// `points` must hold `2·npoints` writable floats; see [`fill_mask_in_place`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_ellipse_values(
    points: *mut f32,
    npoints: usize,
    center_x: f32,
    center_y: f32,
    a: f32,
    b: f32,
    ta: f32,
    tb: f32,
    alpha: f32,
) {
    if points.is_null() || npoints == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(points, npoints * 2);
    fill_mask_in_place(slice, npoints, center_x, center_y, a, b, ta, tb, alpha);
}

/// # Safety
/// `ell` must hold `2·ellpts` floats; see [`fill_outline`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_ellipse_outline(
    ell: *mut f32,
    ellpts: usize,
    center_x: f32,
    center_y: f32,
    ta: f32,
    tb: f32,
    cosa: f32,
    sina: f32,
) {
    if ell.is_null() || ellpts == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(ell, ellpts * 2);
    fill_outline(slice, center_x, center_y, ta, tb, cosa, sina);
}

/// # Safety
/// `points` must hold `2·bbw·bbh` floats; see [`fill_grid_points`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_ellipse_grid(
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
/// `buffer` must hold `w·height` floats (only rows in `[start_j,end_j)` are
/// written); `points` must hold `2·bbw·bbh` floats; see
/// [`interpolate_into_buffer`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_masks_ellipse_interp(
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
    // Caller invariant from _ellipse_get_mask_roi: the strict `<` ends keep
    // both neighbour columns/rows inside the bbox (mi ≤ bbw-2, mj ≤ bbh-2).
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

    /// Naive reference straight from the C `_fill_mask` text, for bit-exact
    /// comparison. Handles both `out_scale` values.
    fn ref_mask(
        points: &[f32],
        n: usize,
        out_scale: u32,
        center_x: f32,
        center_y: f32,
        a: f32,
        b: f32,
        ta: f32,
        tb: f32,
        alpha: f32,
    ) -> Vec<f32> {
        let a2 = a * a;
        let b2 = b * b;
        let ta2 = ta * ta;
        let tb2 = tb * tb;
        let cos_alpha = alpha.cos();
        let sin_alpha = alpha.sin();
        let mut out = vec![0f32; n << out_scale];
        for i in 0..n {
            let x = points[2 * i] - center_x;
            let y = points[2 * i + 1] - center_y;
            let l2 = x * x + y * y;
            let l = l2.sqrt();
            let x_norm = if l != 0.0 { x / l } else { 0.0 };
            let y_norm = if l != 0.0 { y / l } else { 1.0 };
            let x_rot = x_norm * cos_alpha + y_norm * sin_alpha;
            let y_rot = -x_norm * sin_alpha + y_norm * cos_alpha;
            let cosv2 = x_rot * x_rot;
            let sinv2 = y_rot * y_rot;
            let radius2 = a2 * b2 / (a2 * sinv2 + b2 * cosv2);
            let total2 = ta2 * tb2 / (ta2 * sinv2 + tb2 * cosv2);
            let ratio = (total2 - l2) / (total2 - radius2);
            let f = ratio.clamp(0.0, 1.0);
            out[i << out_scale] = f * f;
        }
        out
    }

    #[test]
    fn fill_mask_matches_reference_over_lcg_points() {
        let n = 1024usize;
        let mut pts = vec![0f32; 2 * n];
        super::super::test_util::lcg_fill(&mut pts, 0xE13CE1, 2000.0);
        let (cx, cy, a, b, ta, tb, alpha) = (512.0, 480.0, 300.0, 150.0, 400.0, 250.0, 0.349);
        let mut out = vec![0f32; n];
        fill_mask(&mut out, &pts, cx, cy, a, b, ta, tb, alpha);
        let expect = ref_mask(&pts, n, 0, cx, cy, a, b, ta, tb, alpha);
        assert_eq!(out, expect);
    }

    #[test]
    fn fill_mask_in_place_preserves_odd_lanes() {
        let n = 512usize;
        let mut pts = vec![0f32; 2 * n];
        super::super::test_util::lcg_fill(&mut pts, 0xBEEFCAFE, 900.0);
        let snapshot = pts.clone();
        let (cx, cy, a, b, ta, tb, alpha) = (100.0, 100.0, 80.0, 40.0, 120.0, 60.0, 1.234);
        fill_mask_in_place(&mut pts, n, cx, cy, a, b, ta, tb, alpha);
        // odd lanes untouched
        for k in 0..n {
            assert_eq!(pts[2 * k + 1], snapshot[2 * k + 1], "lane {}", k);
        }
        // even lanes match reference with out_scale=1
        let expect = ref_mask(&snapshot, n, 1, cx, cy, a, b, ta, tb, alpha);
        for k in 0..n {
            assert_eq!(pts[2 * k], expect[2 * k], "even lane {}", k);
        }
    }

    #[test]
    fn fill_mask_clamps_inside_and_outside() {
        // centre point → l2 = 0 → inside → ratio ≥ 1 → clamped to 1 → value 1
        let pts = vec![0f32, 0.0f32, 1000.0f32, 1000.0];
        let mut out = vec![0f32; 2];
        let (a, b, ta, tb, alpha) = (10.0, 5.0, 12.0, 6.0, 0.0);
        fill_mask(&mut out, &pts, 0.0, 0.0, a, b, ta, tb, alpha);
        assert_eq!(out[0], 1.0, "centre should be fully masked");
        assert_eq!(out[1], 0.0, "point far outside should be unmasked");
    }

    #[test]
    fn outline_point_count_matches_c_ramanujan() {
        // C: l = (int)(M_PI * (ta+tb) * (1 + 3λ²/(10+√(4-3λ²)))), ellpts = MIN(360,l)
        let cases = [(400.0_f32, 250.0), (300.0, 300.0), (1.0, 1.0), (1000.0, 10.0)];
        for (ta, tb) in cases {
            let pts = outline_point_count(ta, tb);
            // replicate the C formula in f64 for cross-check
            let lambda: f32 = (ta - tb) / (ta + tb);
            let inner = 3.0f32 * lambda * lambda;
            let denom = 10.0f32 + (4.0f32 - inner).sqrt();
            let l = (std::f64::consts::PI * (ta + tb) as f64 * (1.0 + (inner / denom) as f64)) as i32;
            let expect = (360i32).min(l).max(0) as usize;
            assert_eq!(pts, expect, "ta={ta} tb={tb}");
        }
    }

    #[test]
    fn outline_points_are_parametric_ellipse() {
        let (ta, tb, cosa, sina) = (300.0_f32, 150.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2);
        let ellpts = outline_point_count(ta, tb);
        assert!(ellpts > 0);
        let mut ell = vec![0f32; 2 * ellpts];
        fill_outline(&mut ell, 500.0, 400.0, ta, tb, cosa, sina);
        // n=0 should land at (cx + ta*cosa, cy + ta*sina) — phi=0 → cosp=1, sinp=0
        assert!((ell[0] - (500.0 + ta * cosa)).abs() < 1e-3);
        assert!((ell[1] - (400.0 + ta * sina)).abs() < 1e-3);
        // all distances from center should be bounded by √(ta²+tb²)
        let max_d = (ta * ta + tb * tb).sqrt();
        for n in 0..ellpts {
            let dx = ell[2 * n] - 500.0;
            let dy = ell[2 * n + 1] - 400.0;
            let d = (dx * dx + dy * dy).sqrt();
            assert!(d <= max_d + 1e-3, "n={n}: d={d} > max={max_d}");
        }
    }

    #[test]
    fn coord_grid_matches_c_indexing() {
        let (w, h) = (7usize, 4usize);
        let mut pts = vec![0f32; 2 * w * h];
        fill_coord_grid(&mut pts, w, h, 100.5, -50.25);
        for i in 0..h {
            for j in 0..w {
                assert_eq!(pts[2 * (i * w + j)], 100.5 + j as f32);
                assert_eq!(pts[2 * (i * w + j) + 1], i as f32 - 50.25);
            }
        }
    }

    #[test]
    fn grid_points_use_integer_arithmetic_before_float() {
        // (grid*i + px) evaluated in i32 first — pin against a float drift.
        let bbw = 6usize;
        let bbh = 3usize;
        let mut pts = vec![0f32; 2 * bbw * bbh];
        fill_grid_points(&mut pts, bbw, bbh, 5, 8, 12, 3, 0.25, 3);
        for j in 0..bbh as i32 {
            for i in 0..bbw as i32 {
                let index = (j * bbw as i32 + i) as usize;
                assert_eq!(
                    pts[2 * index],
                    ((3 * (5 + i) + 12) as f32) * 0.25
                );
                assert_eq!(
                    pts[2 * index + 1],
                    ((3 * (8 + j) + 3) as f32) * 0.25
                );
            }
        }
    }

    #[test]
    fn interp_grid_of_one_is_a_copy() {
        // grid=1 collapses the bilinear weights onto the sample itself.
        // The bbox grid carries one extra cell beyond the written range
        // (caller invariant): 5×5 samples feed a 4×4 written region.
        let (w, h) = (5usize, 5usize);
        let mut buffer = vec![0f32; w * h];
        let mut pts = vec![0f32; 2 * 25];
        super::super::test_util::lcg_fill(&mut pts, 11, 1.0);
        interpolate_into_buffer(&mut buffer, w, &pts, 5, 0, 4, 0, 4, 1);
        for j in 0..4 {
            for i in 0..4 {
                let idx = j * w + i;
                assert_eq!(buffer[idx], pts[2 * idx]);
            }
        }
        // last row/col outside [start,end) stay untouched (lcg values > 0)
        assert_eq!(buffer[4], 0.0);
        assert_eq!(buffer[4 * w + 3], 0.0);
        assert_eq!(buffer[4 * w + 4], 0.0);
    }

    #[test]
    fn interp_matches_hand_computed_bilinear_cell() {
        // One 2×2 cell (grid=2), samples 1,2 / 5,6 — write exactly that cell
        // ([0,2)×[0,2)) and verify all four pixels against the C expression
        // evaluated by hand.
        let (w, h) = (4usize, 4usize);
        let mut buffer = vec![7f32; w * h]; // outside stays caller-initialised
        let mut pts = vec![0f32; 2 * 16];
        for s in 0..16 {
            pts[2 * s] = (s + 1) as f32;
        }
        interpolate_into_buffer(&mut buffer, w, &pts, 4, 0, 2, 0, 2, 2);
        // cell (mj,mi)=(0,0): corners p0,p1 / p4,p5 → values 1,2 / 5,6
        let g = 2.0_f32;
        let expect = |ii: f32, jj: f32| {
            (1.0 * (g - ii) * (g - jj) + 2.0 * ii * (g - jj)
                + 5.0 * (g - ii) * jj + 6.0 * ii * jj)
                / (g * g)
        };
        assert_eq!(buffer[0], expect(0.0, 0.0)); // = 1
        assert_eq!(buffer[1], expect(1.0, 0.0)); // = 1.5
        assert_eq!(buffer[w], expect(0.0, 1.0)); // = 3
        assert_eq!(buffer[w + 1], expect(1.0, 1.0)); // = 3.5
        assert_eq!(buffer[2], 7.0, "outside the written range untouched");
        assert_eq!(buffer[2 * w], 7.0, "row below the cell untouched");
    }

    #[test]
    fn ffi_exports_round_trip_through_c_abi() {
        unsafe {
            // coord grid + fill over the FFI boundary
            let (w, h) = (11usize, 7usize);
            let mut gridbuf = vec![0f32; 2 * w * h];
            darkroom_masks_ellipse_coord_grid(gridbuf.as_mut_ptr(), w, h, 5.0, -3.0);
            let mut out = vec![0f32; w * h];
            darkroom_masks_ellipse_fill(
                out.as_mut_ptr(),
                gridbuf.as_ptr(),
                w * h,
                100.0, 200.0, 300.0, 150.0, 330.0, 160.0, std::f32::consts::FRAC_PI_6,
            );
            let mut safe_out = vec![0f32; w * h];
            fill_mask(&mut safe_out, &gridbuf, 100.0, 200.0, 300.0, 150.0, 330.0, 160.0, std::f32::consts::FRAC_PI_6);
            assert_eq!(out, safe_out);

            // in-place values path
            let n = 256usize;
            let mut pts = vec![0f32; 2 * n];
            super::super::test_util::lcg_fill(&mut pts, 0x11ce11, 500.0);
            let snap = pts.clone();
            darkroom_masks_ellipse_values(
                pts.as_mut_ptr(), n, 250.0, 250.0, 100.0, 50.0, 130.0, 65.0, std::f32::consts::FRAC_PI_4,
            );
            let expect = ref_mask(&snap, n, 1, 250.0, 250.0, 100.0, 50.0, 130.0, 65.0, std::f32::consts::FRAC_PI_4);
            for k in 0..n {
                assert_eq!(pts[2 * k], expect[2 * k]);
                assert_eq!(pts[2 * k + 1], snap[2 * k + 1]);
            }

            // null guards refuse without panicking
            darkroom_masks_ellipse_coord_grid(std::ptr::null_mut(), 4, 4, 0.0, 0.0);
            darkroom_masks_ellipse_fill(std::ptr::null_mut(), gridbuf.as_ptr(), 4, 0., 0., 1., 1., 1., 1., 0.);
            darkroom_masks_ellipse_outline(std::ptr::null_mut(), 8, 0., 0., 1., 1., 1., 0.);
            darkroom_masks_ellipse_grid(std::ptr::null_mut(), 4, 4, 0, 0, 0, 0, 1.0, 1);
            darkroom_masks_ellipse_values(std::ptr::null_mut(), 4, 0., 0., 1., 1., 1., 1., 0.);
            darkroom_masks_ellipse_interp(std::ptr::null_mut(), 4, 4, std::ptr::null(), 4, 4, 0, 4, 0, 4, 2);
        }
    }
}
