//! Circle drawn-mask rendering — port of the OMP loops in
//! `src/develop/masks/circle.c` `_circle_get_mask` (whole-pipe form mask) and
//! `_circle_get_mask_roi` (ROI grid-sampled mask).
//!
//! The pixelpipe callbacks between the loops (`dt_dev_distort_backtransform_
//! plus`, `dt_dev_distort_transform_plus`) stay in C; everything pure moved
//! here behind `darkroom_masks_circle_*` exports.
//!
//! Mask semantics: with `radius2 = (r·mindim)²`, `total2 = ((r+border)·mindim)²`,
//! the value is 1 inside the hard circle, a quadratic (`f²`) falloff across the
//! feather band, 0 outside — `sqf(CLIP((total2 - l2) / border2))`.

use super::{circle_feather, masks_roundup, DT_2PI_F};

/// Full-form path, loop 1 of `_circle_get_mask` (circle.c:1100): fill
/// `points` (2 floats per pixel) with the pipe-area coordinate grid,
/// `(posx + j, posy + i)`.
pub fn fill_coord_grid(points: &mut [f32], w: usize, h: usize, pos_x: f32, pos_y: f32) {
    assert_eq!(points.len(), 2 * w * h, "grid buffer must hold w*h pairs");
    let slots = points.as_chunks_mut::<2>().0;
    for i in 0..h {
        let y = i as f32 + pos_y;
        for (j, slot) in slots[i * w..(i + 1) * w].iter_mut().enumerate() {
            slot[0] = pos_x + j as f32;
            slot[1] = y;
        }
    }
}

/// Full-form path, loop 2 of `_circle_get_mask` (circle.c:1147): evaluate the
/// feathered-circle value at every back-transformed point into `buffer`.
pub fn fill_values(
    buffer: &mut [f32],
    points: &[f32],
    center_x: f32,
    center_y: f32,
    total2: f32,
    border2: f32,
) {
    assert_eq!(buffer.len() * 2, points.len(), "one point pair per value");
    for (i, v) in buffer.iter_mut().enumerate() {
        let dx = points[2 * i] - center_x;
        let dy = points[2 * i + 1] - center_y;
        *v = circle_feather(dx * dx + dy * dy, total2, border2);
    }
}

/// ROI path, loop 1 of `_circle_get_mask_roi` (circle.c:1217): write the
/// outer-circle outline around `(center_x, center_y)` with radius `total`
/// into `circ` (2 floats per point), using the C's eight-fold symmetry
/// expansion — `n < circpts/8` base angles each produce 8 mirrored points.
pub fn fill_outline(circ: &mut [f32], center_x: f32, center_y: f32, total: f32) {
    let circpts = circ.len() / 2;
    assert!(
        circpts.is_multiple_of(8),
        "outline buffer must hold a multiple of 8 points"
    );
    for n in 0..circpts / 8 {
        let phi = DT_2PI_F * n as f32 / circpts as f32;
        let x = total * phi.cos();
        let y = total * phi.sin();
        // take advantage of symmetry
        let ix = 16 * n;
        circ[ix] = center_x + x;
        circ[ix + 1] = center_y + y;
        circ[ix + 2] = center_x + x;
        circ[ix + 3] = center_y - y;
        circ[ix + 4] = center_x - x;
        circ[ix + 5] = center_y + y;
        circ[ix + 6] = center_x - x;
        circ[ix + 7] = center_y - y;
        circ[ix + 8] = center_x + y;
        circ[ix + 9] = center_y + x;
        circ[ix + 10] = center_x + y;
        circ[ix + 11] = center_y - x;
        circ[ix + 12] = center_x - y;
        circ[ix + 13] = center_y + x;
        circ[ix + 14] = center_x - y;
        circ[ix + 15] = center_y - x;
    }
}

/// Number of outline points for outer radius squared `total2` — the C's
/// `dt_masks_roundup(MIN(360, DT_2PI_F * total2), 8)` with its float→int
/// truncation (`MIN` promotes to float, then `size_t` conversion truncates
/// toward zero).
pub fn outline_point_count(total2: f32) -> usize {
    let raw = 360.0_f32.min(DT_2PI_F * total2) as i32;
    masks_roundup(raw, 8) as usize
}

/// ROI path, grid populate loop (circle.c:1307): fill `points` (bbw×bbh
/// pairs) with the sampled grid coordinates in module coordinates. The C
/// computes `(grid*i + px)` in INTEGER arithmetic before the float
/// conversion — replicated here.
#[allow(clippy::too_many_arguments)] // mirrors the C loop's parameter list
pub fn fill_grid_points(
    points: &mut [f32],
    bbw: usize,
    bbh: usize,
    bbxm: i32,
    bbym: i32,
    px: i32,
    py: i32,
    iscale: f32,
    grid: i32,
) {
    assert_eq!(points.len(), 2 * bbw * bbh);
    for j in 0..bbh as i32 {
        let gy = (grid * (bbym + j) + py) as f32 * iscale;
        for i in 0..bbw as i32 {
            let index = (j * bbw as i32 + i) as usize;
            points[2 * index] = (grid * (bbxm + i) + px) as f32 * iscale;
            points[2 * index + 1] = gy;
        }
    }
}

/// ROI path, mask-value loop (circle.c:1334): evaluate the feathered circle
/// at the back-transformed grid points, writing results into the even lanes
/// IN PLACE exactly like the C re-use of the `points` array.
pub fn values_in_place(
    points: &mut [f32],
    count: usize,
    center_x: f32,
    center_y: f32,
    total2: f32,
    border2: f32,
) {
    assert!(count <= points.len() / 2);
    for idx in 0..count {
        let dx = points[2 * idx] - center_x;
        let dy = points[2 * idx + 1] - center_y;
        points[2 * idx] = circle_feather(dx * dx + dy * dy, total2, border2);
    }
}

/// ROI path, interpolation loop (circle.c:1358): splat the bbw×bbh sampled
/// values (even lanes of `points`) over the ROI `buffer` (`w` wide) by
/// bilinear weighting within each `grid × grid` cell. Only the bounding-box
/// rows/cols are touched, matching the C's `[start_i, end_i) × [start_j,
/// end_j)` ranges where the caller pre-initialised the rest to zero.
///
/// The sequential float multiplications mirror the C expression order
/// (`v*(g-ii)*(g-jj)` evaluates left-to-right, converting each int operand at
/// its own multiply), and the denominator is the INT product `grid*grid`
/// converted once at the division.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_into_buffer(
    buffer: &mut [f32],
    w: usize,
    points: &[f32],
    bbw: usize,
    start_i: i32,
    end_i: i32,
    start_j: i32,
    end_j: i32,
    grid: i32,
) {
    let g = grid;
    let denom = (g * g) as f32;
    for j in start_j..end_j {
        let jj = j.rem_euclid(g);
        let mj = j.div_euclid(g) - start_j.div_euclid(g);
        for i in start_i..end_i {
            let ii = i.rem_euclid(g);
            let mi = i.div_euclid(g) - start_i.div_euclid(g);
            let mindex = (mj * bbw as i32 + mi) as usize;
            buffer[j as usize * w + i as usize] =
                (points[mindex * 2] * (g - ii) as f32 * (g - jj) as f32
                    + points[(mindex + 1) * 2] * ii as f32 * (g - jj) as f32
                    + points[(mindex + bbw) * 2] * (g - ii) as f32 * jj as f32
                    + points[(mindex + bbw + 1) * 2] * ii as f32 * jj as f32)
                    / denom;
        }
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `points` must hold `2·w·h` floats; see [`fill_coord_grid`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_circle_coord_grid(
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
/// `buffer` must hold `n` floats and `points` `2·n`; see [`fill_values`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_circle_fill(
    buffer: *mut f32,
    points: *const f32,
    n: usize,
    center_x: f32,
    center_y: f32,
    total2: f32,
    border2: f32,
) {
    if buffer.is_null() || points.is_null() || n == 0 {
        return;
    }
    let buffer = std::slice::from_raw_parts_mut(buffer, n);
    let points = std::slice::from_raw_parts(points, n * 2);
    fill_values(buffer, points, center_x, center_y, total2, border2);
}

/// # Safety
/// `circ` must hold `2·circpts` floats with `circpts % 8 == 0`; see
/// [`fill_outline`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_circle_outline(
    circ: *mut f32,
    circpts: usize,
    center_x: f32,
    center_y: f32,
    total: f32,
) {
    if circ.is_null() || circpts == 0 || !circpts.is_multiple_of(8) || circpts > i32::MAX as usize
    {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(circ, circpts * 2);
    fill_outline(slice, center_x, center_y, total);
}

/// # Safety
/// `points` must hold `2·bbw·bbh` floats; see [`fill_grid_points`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_circle_grid(
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
/// `points` must hold `2·npoints` writable floats; see [`values_in_place`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_circle_values(
    points: *mut f32,
    npoints: usize,
    center_x: f32,
    center_y: f32,
    total2: f32,
    border2: f32,
) {
    if points.is_null() || npoints == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(points, npoints * 2);
    values_in_place(slice, npoints, center_x, center_y, total2, border2);
}

/// # Safety
/// `buffer` must hold `w·height` floats (only rows in `[start_j,end_j)` are
/// written); `points` must hold `2·bbw·bbh` floats with the caller-side
/// invariants `end_i ≤ min(w, (bbxm+bbw-1)·grid)` and the same for `j`, which
/// keep the neighbour lookups in bounds exactly as they do in C; see
/// [`interpolate_into_buffer`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_masks_circle_interp(
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
    // Caller invariant from _circle_get_mask_roi: the strict `<` ends keep
    // both neighbour columns/rows inside the bbox (mi ≤ bbw-2, mj ≤ bbh-2).
    // Refuse rather than panic if a corrupt caller breaks it.
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

    /// Naive reference straight from the C text, for bit-exact comparison.
    fn ref_feather(l2: f32, total2: f32, border2: f32) -> f32 {
        let ratio = (total2 - l2) / border2;
        let f = if ratio < 0.0 {
            0.0
        } else if ratio > 1.0 {
            1.0
        } else {
            ratio
        };
        f * f
    }

    #[test]
    fn feather_matches_c_reference_and_clamps() {
        // C-consistent geometry: radius 5, border 5 → total 10;
        // border2 = total2 − radius2 = 100 − 25 = 75.
        let (t2, b2) = (100.0_f32, 75.0_f32);
        // centre and hard-radius edge are both fully opaque (ratio ≥ 1 clamps)
        assert_eq!(circle_feather(0.0, t2, b2), 1.0);
        assert_eq!(circle_feather(25.0, t2, b2), 1.0);
        // outer ring and beyond → both clamped arms give 0
        assert_eq!(circle_feather(100.0, t2, b2), 0.0);
        assert_eq!(circle_feather(1e9, t2, b2), 0.0);
        // mid-band: l2 = 7.5² → ratio (100−56.25)/75 = 7/12 → f² = 49/144
        let mid = ref_feather(56.25, t2, b2);
        assert_eq!(circle_feather(56.25, t2, b2), mid);
        assert!((mid - 49.0 / 144.0).abs() < 1e-6, "{mid}");
    }

    #[test]
    fn coord_grid_matches_c_indexing() {
        let (w, h) = (5usize, 3usize);
        let mut pts = vec![0f32; 2 * w * h];
        fill_coord_grid(&mut pts, w, h, 11.5, -7.25);
        for i in 0..h {
            for j in 0..w {
                assert_eq!(pts[2 * (i * w + j)], 11.5 + j as f32);
                assert_eq!(pts[2 * (i * w + j) + 1], i as f32 - 7.25);
            }
        }
    }

    #[test]
    fn fill_values_equals_reference_over_random_points() {
        let n = 997usize; // deliberately not a multiple of anything
        let mut points = vec![0f32; 2 * n];
        super::super::test_util::lcg_fill(&mut points, 0xC1FC1, 2000.0);
        let mut buffer = vec![0f32; n];
        fill_values(&mut buffer, &points, 512.25, 480.75, 640.0, 1234.5);
        for i in 0..n {
            let dx = points[2 * i] - 512.25;
            let dy = points[2 * i + 1] - 480.75;
            let expect = ref_feather(dx * dx + dy * dy, 640.0, 1234.5);
            assert_eq!(buffer[i], expect, "point {i}");
        }
    }

    #[test]
    fn outline_has_eightfold_symmetry_and_radius() {
        for total in [1.0_f32, 12.5, 300.0] {
            let total2 = total * total;
            let circpts = outline_point_count(total2);
            assert!(circpts % 8 == 0 && circpts <= 360);
            let mut circ = vec![0f32; 2 * circpts];
            fill_outline(&mut circ, 3.0, 4.0, total);
            for n in 0..circpts / 8 {
                let ix = 16 * n;
                // every mirrored point sits at distance `total` from the centre
                for k in 0..8 {
                    let dx = circ[ix + 2 * k] - 3.0;
                    let dy = circ[ix + 2 * k + 1] - 4.0;
                    let d = (dx * dx + dy * dy).sqrt();
                    assert!((d - total).abs() < 1e-4 * total.max(1.0), "n{n} k{k}: {d} vs {total}");
                }
                // and the mirror set covers ±x/±y/±y/±x exactly (approx:
                // round-tripping cx±y through f32 addition isn't bit-stable)
                assert_eq!(circ[ix + 2], circ[ix]); // same x, mirrored y — same var
                let mirror_y = circ[ix + 3] - (8.0 - circ[ix + 1]);
                assert!(mirror_y.abs() < 1e-3, "cy−y arm: {}", circ[ix + 3]);
                assert!((circ[ix + 8] - 3.0 - (circ[ix + 1] - 4.0)).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn outline_point_count_matches_c_formula() {
        // roundup(MIN(360, 2π·total2), 8)
        assert_eq!(outline_point_count(0.0), 0);
        assert_eq!(outline_point_count(1.0), masks_roundup(6, 8) as usize); // 2π→6
        assert_eq!(outline_point_count(16.0), masks_roundup(100, 8) as usize); // 2π·16≈100.5
        assert_eq!(outline_point_count(1e6), 360);
    }

    #[test]
    fn grid_points_use_integer_arithmetic_before_float() {
        // (grid*i + px) evaluated in i32 first — pin against a float drift.
        let bbw = 4usize;
        let bbh = 3usize;
        let mut pts = vec![0f32; 2 * bbw * bbh];
        fill_grid_points(&mut pts, bbw, bbh, 10, 20, 3, 7, 0.5, 4);
        for j in 0..bbh as i32 {
            for i in 0..bbw as i32 {
                let index = (j * bbw as i32 + i) as usize;
                assert_eq!(pts[2 * index], ((4 * (10 + i) + 3) as f32) * 0.5);
                assert_eq!(pts[2 * index + 1], ((4 * (20 + j) + 7) as f32) * 0.5);
            }
        }
    }

    #[test]
    fn values_in_place_writes_even_lanes_only() {
        let n = 33usize;
        let mut pts = vec![0f32; 2 * n];
        super::super::test_util::lcg_fill(&mut pts, 42, 100.0);
        let odd_before: Vec<f32> = pts.iter().skip(1).step_by(2).copied().collect();
        values_in_place(&mut pts, n, 50.0, 50.0, 400.0, 600.0);
        // odd lanes untouched (still the y coordinates)
        for (k, v) in pts.iter().skip(1).step_by(2).copied().enumerate() {
            assert_eq!(v, odd_before[k], "lane {k}");
        }
        // even lanes hold values — covered bit-exactly by the snapshot test below.
    }

    #[test]
    fn values_in_place_match_reference_from_snapshot() {
        let n = 64usize;
        let mut pts = vec![0f32; 2 * n];
        super::super::test_util::lcg_fill(&mut pts, 777, 90.0);
        let snapshot = pts.clone();
        values_in_place(&mut pts, n, 45.5, 44.5, 500.0, 700.0);
        for i in 0..n {
            let dx = snapshot[2 * i] - 45.5;
            let dy = snapshot[2 * i + 1] - 44.5;
            assert_eq!(pts[2 * i], ref_feather(dx * dx + dy * dy, 500.0, 700.0));
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
        super::super::test_util::lcg_fill(&mut pts, 9, 1.0);
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
            pts[2 * s] = (s + 1) as f32; // values 1..=16 in bbox order
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
            let (w, h) = (13usize, 9usize);
            let mut gridbuf = vec![0f32; 2 * w * h];
            darkroom_masks_circle_coord_grid(gridbuf.as_mut_ptr(), w, h, 2.0, 3.0);
            let mut out = vec![0f32; w * h];
            darkroom_masks_circle_fill(
                out.as_mut_ptr(),
                gridbuf.as_ptr(),
                (w * h) as usize,
                6.0,
                7.0,
                36.0,
                108.0,
            );
            let mut safe_out = vec![0f32; w * h];
            fill_values(&mut safe_out, &gridbuf, 6.0, 7.0, 36.0, 108.0);
            assert_eq!(out, safe_out);

            // null guards refuse without panicking
            darkroom_masks_circle_coord_grid(std::ptr::null_mut(), 4, 4, 0.0, 0.0);
            darkroom_masks_circle_fill(std::ptr::null_mut(), gridbuf.as_ptr(), 4, 0., 0., 1., 1.);
            darkroom_masks_circle_outline(std::ptr::null_mut(), 8, 0., 0., 1.);
            darkroom_masks_circle_values(std::ptr::null_mut(), 4, 0., 0., 1., 1.);
            darkroom_masks_circle_interp(
                std::ptr::null_mut(), 4, 4, std::ptr::null(), 4, 4, 0, 4, 0, 4, 2,
            );
        }
    }
}
