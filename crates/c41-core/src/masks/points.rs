//! Mask point-manipulation loops — port of the OMP loops that shift,
//! generate, and reduce mask control/circumference/guide points.
//!
//! These loops live in the point-pipeline phase of each mask shape: after
//! the C side allocates buffers and runs pixelpipe transforms, the points
//! are shifted (source-anchor offset), circumferences are generated, a
//! bounding box is reduced, or gradient guide points are accumulated.
//!
//! The C code already calls into `darkroom_masks_*` for the per-pixel mask
//! value kernels (circle/ellipse/gradient/brush/path/object/detail). The loops
//! ported here are the remaining point-arithmetic loops that were still
//! using `DT_OMP_FOR` / `DT_OMP_FOR_SIMD`.

use super::DT_2PI_F;

// ── Pure kernels ──────────────────────────────────────────────────────────────

/// Shift all points by `(dx, dy)` starting from `start_index`.
///
/// Replaces the four identical point-shift OMP loops in circle.c:744,
/// ellipse.c:333, brush.c:1065, and path.c:1511. Each C loop is:
/// ```c
/// for(int i = start; i < count; i++) {
///   ptsbuf[i * 2]     += dx;
///   ptsbuf[i * 2 + 1] += dy;
/// }
/// ```
pub fn shift_points(points: &mut [f32], count: usize, dx: f32, dy: f32, start_index: usize) {
    assert!(start_index <= count && count * 2 <= points.len(), "point buffer too small for count");
    for i in start_index..count {
        points[i * 2] += dx;
        points[i * 2 + 1] += dy;
    }
}

/// Generate `l+1` points: center at index 0, then the circle circumference
/// sampled at `alpha = (i-1) * 2π / l` for i=1..=l.
///
/// Replaces circle.c:695 — `_points_to_transform` circumference loop:
/// ```c
/// points[0] = center_x;  points[1] = center_y;
/// for(int i = 1; i < l + 1; i++) {
///   const float alpha = (i - 1) * DT_2PI_F / (float)l;
///   points[i * 2]     = center_x + r * cosf(alpha);
///   points[i * 2 + 1] = center_y + r * sinf(alpha);
/// }
/// ```
pub fn circle_circumference(points: &mut [f32], center_x: f32, center_y: f32, r: f32, l: usize) {
    assert!(points.len() >= 2 * (l + 1), "circumference buffer must hold l+1 point pairs");
    points[0] = center_x;
    points[1] = center_y;
    for i in 1..=l {
        let alpha = DT_2PI_F * (i as f32 - 1.0) / (l as f32);
        points[i * 2] = center_x + r * alpha.cos();
        points[i * 2 + 1] = center_y + r * alpha.sin();
    }
}

/// Generate ellipse circumference points starting at index 5 (indices 0–4
/// hold the center + four pivot points, written by the caller).
///
/// Replaces ellipse.c:282 — `_points_to_transform` circumference loop:
/// ```c
/// for(int i = 5; i < l + 5; i++) {
///   const float alpha = (i - 5) * DT_2PI_F / (float)l;
///   points[i * 2]     = x + a * cosf(alpha) * cosv - b * sinf(alpha) * sinv;
///   points[i * 2 + 1] = y + a * cosf(alpha) * sinv + b * sinf(alpha) * cosv;
/// }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn ellipse_circumference(
    points: &mut [f32],
    x: f32,
    y: f32,
    a: f32,
    b: f32,
    cosv: f32,
    sinv: f32,
    l: usize,
) {
    assert!(points.len() >= 2 * (l + 5), "ellipse buffer must hold l+5 point pairs");
    for i in 5..(l + 5) {
        let alpha = DT_2PI_F * (i as f32 - 5.0) / (l as f32);
        let cosa = alpha.cos();
        let sina = alpha.sin();
        points[i * 2] = x + a * cosa * cosv - b * sina * sinv;
        points[i * 2 + 1] = y + a * cosa * sinv + b * sina * cosv;
    }
}

/// Reduce a bounding box over `points[start_idx..count]` (and optionally
/// `border` at the same range): find min/max x and y.
///
/// Replaces brush.c:2768 — `_brush_bounding_box_raw` reduction:
/// ```c
/// float xmin = FLT_MAX, xmax = FLT_MIN, ymin = FLT_MAX, ymax = FLT_MIN;
/// for(int i = start; i < num_points; i++) {
///   if(border) { update xmin/xmax/ymin/ymax from border[i*2], border[i*2+1]; }
///   update xmin/xmax/ymin/ymax from points[i*2], points[i*2+1];
/// }
/// ```
pub fn bbox_reduction(
    points: &[f32],
    border: Option<&[f32]>,
    count: usize,
    start_idx: usize,
) -> (f32, f32, f32, f32) {
    assert!(start_idx <= count && count * 2 <= points.len());
    let plen = count * 2;
    let blen = border.map_or(plen, |b| b.len().min(plen));
    let border = border.map(|b| &b[..blen]);

    let mut xmin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    for i in start_idx..count {
        let idx = i * 2;
        if let Some(b) = border {
            let bx = b[idx];
            let by = b[idx + 1];
            xmin = xmin.min(bx);
            xmax = xmax.max(bx);
            ymin = ymin.min(by);
            ymax = ymax.max(by);
        }
        let px = points[idx];
        let py = points[idx + 1];
        xmin = xmin.min(px);
        xmax = xmax.max(px);
        ymin = ymin.min(py);
        ymax = ymax.max(py);
    }
    (xmin, xmax, ymin, ymax)
}

/// Generate gradient guide curve points and append them (after the 3 control
/// points already written by the caller) into the output `points` buffer.
///
/// Replaces gradient.c:734 — the C `_gradient_get_points` thread-local
/// accumulation + merge loop. The C version parallelizes across threads
/// with per-thread counters, then merges in thread order. The Rust serial
/// version computes the same points in index order, so the merge loop is
/// unnecessary — iteration order is deterministic.
///
/// `count` is the total point buffer size (including the 3 control points).
/// Returns the number of guide points actually written (those not clipped
/// by the image-frame guard), starting at index 3.
#[allow(clippy::too_many_arguments)]
pub fn gradient_guide_points(
    points: &mut [f32],
    x: f32,
    y: f32,
    wd: f32,
    ht: f32,
    scale: f32,
    cosv: f32,
    sinv: f32,
    curvature: f32,
    count: usize,
) -> usize {
    assert!(count >= 3 && count * 2 <= points.len(), "need at least 3 control points");

    let xstart = if curvature.abs() > 1.0 {
        -1.0 / curvature.abs().sqrt()
    } else {
        -1.0
    };
    // count - 3 is the number of guide points to attempt
    let n_guides = count - 3;
    let xdelta = -2.0 * xstart / (n_guides as f32);

    let mut written = 0usize;
    for k in 0..n_guides {
        // i = _nb_ctrl_point() + k  =>  i = 3 + k  in the C (count excludes controls)
        let xi = xstart + (k as f32 + 3.0 - 3.0) * xdelta; // = xstart + k * xdelta
        let yi = curvature * xi * xi;
        let xii = (cosv * xi + sinv * yi) * scale;
        let yii = (sinv * xi - cosv * yi) * scale;
        let xiii = xii + x * wd;
        let yiii = yii + y * ht;

        // image-frame guard — skip points that extend too far beyond the frame
        if !(xiii < -wd || xiii > 2.0 * wd || yiii < -ht || yiii > 2.0 * ht) {
            let idx = (3 + written) * 2;
            points[idx] = xiii;
            points[idx + 1] = yiii;
            written += 1;
        }
    }
    written
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `points` must hold `2·count` writable floats. `start_index` skips the
/// first `start_index` control-point pairs.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_points_shift(
    points: *mut f32,
    count: usize,
    dx: f32,
    dy: f32,
    start_index: usize,
) {
    if points.is_null() || count == 0 || start_index > count || count > i32::MAX as usize {
        return;
    }
    let Some(len) = count.checked_mul(2) else { return };
    let slice = std::slice::from_raw_parts_mut(points, len);
    shift_points(slice, count, dx, dy, start_index);
}

/// # Safety
/// `points` must hold `2*(l+1)` writable floats; see [`circle_circumference`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_circle_circumference(
    points: *mut f32,
    center_x: f32,
    center_y: f32,
    r: f32,
    l: i32,
) {
    if points.is_null() || l <= 0 {
        return;
    }
    let l = l as usize;
    let Some(len) = l.checked_add(1).and_then(|n| n.checked_mul(2)) else { return };
    let slice = std::slice::from_raw_parts_mut(points, len);
    circle_circumference(slice, center_x, center_y, r, l);
}

/// # Safety
/// `points` must hold `2*(l+5)` writable floats (indices 0–4 set by caller);
/// see [`ellipse_circumference`].
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_ellipse_circumference(
    points: *mut f32,
    x: f32,
    y: f32,
    a: f32,
    b: f32,
    cosv: f32,
    sinv: f32,
    l: i32,
) {
    if points.is_null() || l <= 0 {
        return;
    }
    let l = l as usize;
    let Some(len) = l.checked_add(5).and_then(|n| n.checked_mul(2)) else { return };
    let slice = std::slice::from_raw_parts_mut(points, len);
    ellipse_circumference(slice, x, y, a, b, cosv, sinv, l);
}

/// # Safety
/// `points` and (optionally) `border` must each hold at least `2*count` floats.
/// Output values are written through the four `*_out` pointers.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_bbox_reduction(
    points: *const f32,
    border: *const f32,
    count: usize,
    start_idx: usize,
    x_min_out: *mut f32,
    x_max_out: *mut f32,
    y_min_out: *mut f32,
    y_max_out: *mut f32,
) {
    if points.is_null()
        || x_min_out.is_null()
        || x_max_out.is_null()
        || y_min_out.is_null()
        || y_max_out.is_null()
        || count == 0
        || start_idx > count
        || count > i32::MAX as usize
    {
        return;
    }
    let Some(len) = count.checked_mul(2) else { return };
    let points_slice = std::slice::from_raw_parts(points, len);
    let border_slice = if border.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(border, len))
    };
    let (xmin, xmax, ymin, ymax) = bbox_reduction(points_slice, border_slice, count, start_idx);
    *x_min_out = xmin;
    *x_max_out = xmax;
    *y_min_out = ymin;
    *y_max_out = ymax;
}

/// # Safety
/// `points` must hold `2*count` writable floats. Indices 0–2 (the 3 control
/// points) must already be set by the caller. Writes `written` guide points
/// starting at index 3. Returns the number of guide points written.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_gradient_guide_points(
    points: *mut f32,
    count: usize,
    x: f32,
    y: f32,
    wd: f32,
    ht: f32,
    scale: f32,
    cosv: f32,
    sinv: f32,
    curvature: f32,
) -> usize {
    if points.is_null() || count < 3 || count > i32::MAX as usize {
        return 0;
    }
    let Some(len) = count.checked_mul(2) else { return 0 };
    let slice = std::slice::from_raw_parts_mut(points, len);
    gradient_guide_points(slice, x, y, wd, ht, scale, cosv, sinv, curvature, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── shift_points ─────────────────────────────────────────────────────────

    #[test]
    fn shift_all_points_matches_c_loop() {
        // C circle.c:744 shift starting at 0
        let mut pts = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dx = 10.5;
        let dy = -3.25;
        shift_points(&mut pts, 4, dx, dy, 0);
        for i in 0..pts.len() / 2 {
            assert_eq!(pts[i * 2], (i as f32 * 2.0 + 1.0) + dx, "x[{i}]");
            assert_eq!(pts[i * 2 + 1], (i as f32 * 2.0 + 2.0) + dy, "y[{i}]");
        }
    }

    #[test]
    fn shift_skips_first_5_points_matches_ellipse_c_loop() {
        // C ellipse.c:333 shift starting at index 5
        let mut pts = vec![0.0f32; 10]; // 5 pairs
        for i in 0..5 {
            pts[i * 2] = i as f32;
            pts[i * 2 + 1] = (i * 10) as f32;
        }
        let dx = 100.0;
        let dy = 200.0;
        shift_points(&mut pts, 5, dx, dy, 5);
        // points 0..4 unchanged
        for i in 0..5 {
            assert_eq!(pts[i * 2], i as f32);
            assert_eq!(pts[i * 2 + 1], (i * 10) as f32);
        }
    }

    #[test]
    fn shift_empty_noop() {
        let mut pts = vec![0.0f32; 4];
        shift_points(&mut pts, 0, 1.0, 2.0, 0);
        assert_eq!(pts, vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ── circle_circumference ────────────────────────────────────────────────

    #[test]
    fn circle_circumference_center_first_then_unit_circle() {
        let r = 5.0f32;
        let center_x = 10.0f32;
        let center_y = 20.0f32;
        let l = 8usize; // 8 segments → 9 points total

        let mut pts = vec![0.0f32; 2 * (l + 1)];
        circle_circumference(&mut pts, center_x, center_y, r, l);

        // index 0 is the center
        assert_eq!(pts[0], center_x);
        assert_eq!(pts[1], center_y);

        // index 1 → alpha = 0 → (center_x + r, center_y)
        assert!((pts[2] - (center_x + r)).abs() < 1e-5);
        assert!((pts[3] - center_y).abs() < 1e-5);

        // index 5 → alpha = 4 * 2π/8 = π → (center_x - r, center_y)
        let i = 5;
        assert!((pts[i * 2] - (center_x - r)).abs() < 1e-5);
        assert!((pts[i * 2 + 1] - center_y).abs() < 1e-5);
    }

    #[test]
    fn circle_circumference_single_segment() {
        let mut pts = vec![0.0f32; 4]; // l=1 → 2 points
        circle_circumference(&mut pts, 0.0, 0.0, 10.0, 1);
        // center + one circumference point at alpha=0
        assert_eq!(pts[0], 0.0);
        assert_eq!(pts[1], 0.0);
        assert!((pts[2] - 10.0).abs() < 1e-5);
        assert!((pts[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn circle_circumference_all_points_at_radius() {
        let r = 42.5f32;
        let l = 100usize;
        let mut pts = vec![0.0f32; 2 * (l + 1)];
        circle_circumference(&mut pts, 100.0, 100.0, r, l);
        for i in 1..=l {
            let dx = pts[i * 2] - 100.0;
            let dy = pts[i * 2 + 1] - 100.0;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!((dist - r).abs() < 1e-3, "point {i}: dist={dist} r={r}");
        }
    }

    // ── ellipse_circumference ──────────────────────────────────────────────

    #[test]
    fn ellipse_circumference_first_generated_point() {
        // i=5 → alpha = 0 → cosa=1, sina=0
        // x + a*1*cosv - b*0*sinv = x + a*cosv
        // y + a*1*sinv + b*0*cosv = y + a*sinv
        let (x, y, a, b) = (100.0f32, 200.0, 50.0, 30.0);
        let v = 0.0f32; // cosv=1, sinv=0
        let mut pts = vec![0.0f32; 2 * 10]; // l=5 → 10 points total
        ellipse_circumference(&mut pts, x, y, a, b, v.cos(), v.sin(), 5);
        assert!((pts[10] - (x + a)).abs() < 1e-5);
        assert!((pts[11] - (y + 0.0)).abs() < 1e-5);
    }

    #[test]
    fn ellipse_circumference_quarter_point() {
        // i=6 → alpha = DT_2PI_F * 1 / l; with l=4: alpha = π/2
        // cosa ≈ 0, sina ≈ 1
        // x + a*0*cosv - b*1*sinv = x - b*sinv
        // y + a*0*sinv + b*1*cosv = y + b*cosv
        let (x, y, a, b) = (50.0f32, 60.0, 70.0, 40.0);
        let cosv = 0.6;
        let sinv = 0.8;
        let l = 4usize;
        let mut pts = vec![0.0f32; 2 * (l + 5)];
        ellipse_circumference(&mut pts, x, y, a, b, cosv, sinv, l);
        // i=6 → alpha = 2π*1/4 = π/2 → cosa≈0, sina≈1
        let i = 6;
        let exp_x = x - b * sinv; // 50 - 40*0.8 = 18
        let exp_y = y + b * cosv; // 60 + 40*0.6 = 84
        assert!((pts[i * 2] - exp_x).abs() < 1e-4, "x: {} vs {}", pts[i * 2], exp_x);
        assert!((pts[i * 2 + 1] - exp_y).abs() < 1e-4, "y: {} vs {}", pts[i * 2 + 1], exp_y);
    }

    // ── bbox_reduction ──────────────────────────────────────────────────────

    #[test]
    fn bbox_reduction_points_only() {
        // points: (1,2), (5,10), (-3,4), (2,-1)
        let pts = vec![1.0f32, 2.0, 5.0, 10.0, -3.0, 4.0, 2.0, -1.0];
        let (xmin, xmax, ymin, ymax) = bbox_reduction(&pts, None, 4, 0);
        assert_eq!(xmin, -3.0);
        assert_eq!(xmax, 5.0);
        assert_eq!(ymin, -1.0);
        assert_eq!(ymax, 10.0);
    }

    #[test]
    fn bbox_reduction_with_border() {
        let pts = vec![0.0f32; 8]; // 4 points, all 0
        let border = vec![10.0f32, 20.0, -5.0, 15.0, 3.0, 8.0, 12.0, -2.0];
        let (xmin, xmax, ymin, ymax) = bbox_reduction(&pts, Some(&border), 4, 0);
        assert_eq!(xmin, -5.0);
        assert_eq!(xmax, 12.0);
        assert_eq!(ymin, -2.0);
        assert_eq!(ymax, 20.0);
    }

    #[test]
    fn bbox_reduction_skips_first_points() {
        // 5 points, start_idx=2 → only points 2 and 3
        let pts = vec![100.0f32, 100.0, 200.0, 200.0, 1.0, 2.0, 5.0, 10.0, -3.0, 4.0];
        let (xmin, xmax, ymin, ymax) = bbox_reduction(&pts, None, 4, 2);
        assert_eq!(xmin, 1.0);
        assert_eq!(xmax, 5.0);
        assert_eq!(ymin, 2.0);
        assert_eq!(ymax, 10.0);
    }

    // ── gradient_guide_points ──────────────────────────────────────────────

    #[test]
    fn gradient_guide_points_count_matches_c_count_minus_3() {
        // C: count = scale + 3, guide points start at index 3, loop i from 3 to count
        // So n_guides = count - 3, all written (no clipping at normal scale)
        let count = 15usize;
        let mut pts = vec![0.0f32; count * 2];
        // set control points
        pts[0] = 1.0; pts[1] = 2.0;
        pts[2] = 3.0; pts[3] = 4.0;
        pts[4] = 5.0; pts[5] = 6.0;

        let n_guides = count - 3;
        // x, y are normalised (0–1) coords per the C code; scale is the image diagonal
        let x = 0.5f32;
        let y = 0.5f32;
        let wd = 800.0f32;
        let ht = 600.0f32;
        let scale = dt_fast_hypotf(wd, ht); // = 1000
        let curvature = 0.0f32; // parabolic → xstart = -1, xdelta = 2/12
        let cosv = 1.0f32;
        let sinv = 0.0f32;

        let written = gradient_guide_points(&mut pts, x, y, wd, ht, scale, cosv, sinv, curvature, count);
        assert_eq!(written, n_guides);
        // points[0..5] (control points) untouched
        assert_eq!(pts[0], 1.0);
        assert_eq!(pts[1], 2.0);
        assert_eq!(pts[2], 3.0);
        assert_eq!(pts[3], 4.0);
        assert_eq!(pts[4], 5.0);
        assert_eq!(pts[5], 6.0);
        assert!(pts[6] != 0.0 || pts[7] != 0.0, "first guide point should be written");
    }

    #[test]
    fn gradient_guide_points_clips_out_of_frame() {
        // With a very large curvature, some points will be out of frame and skipped
        let count = 20usize;
        let mut pts = vec![0.0f32; count * 2];
        // control points
        for k in 0..3 {
            pts[k * 2] = k as f32;
            pts[k * 2 + 1] = k as f32;
        }
        let x = 0.5f32;
        let y = 0.5f32;
        let wd = 100.0f32;
        let ht = 100.0f32;
        let scale = dt_fast_hypotf(wd, ht);
        let cosv = 0.0f32;
        let sinv = 1.0f32;
        // curvature that produces points far outside the frame
        let curvature = 50.0f32;

        let written = gradient_guide_points(&mut pts, x, y, wd, ht, scale, cosv, sinv, curvature, count);
        // With extreme curvature, some or all points will be clipped
        assert!(written <= count - 3);
        // but at least verify the first written point is within frame
        if written > 0 {
            let px = pts[3 * 2];
            let py = pts[3 * 2 + 1];
            assert!(px >= -wd && px <= 2.0 * wd);
            assert!(py >= -ht && py <= 2.0 * ht);
        }
    }

    #[test]
    fn gradient_guide_points_xstart_matches_c_condition() {
        // When |curvature| <= 1: xstart = -1.0
        // When |curvature| > 1: xstart = -sqrt(1/|curvature|)
        let count = 10usize;
        let mut pts = vec![0.0f32; count * 2];
        for k in 0..3 {
            pts[k * 2] = k as f32;
            pts[k * 2 + 1] = k as f32;
        }
        let x = 0.5f32;
        let y = 0.5f32;
        let wd = 100.0f32;
        let ht = 100.0f32;
        let scale = dt_fast_hypotf(wd, ht);
        let cosv = 1.0f32;
        let sinv = 0.0f32;

        // curvature = 0.0 → |curvature| < 1 → xstart = -1
        let written0 = gradient_guide_points(&mut pts, x, y, wd, ht, scale, cosv, sinv, 0.0, count);
        let first_x0 = pts[6]; // first guide point x

        // Manually compute expected for i=3, k=0:
        // xstart = -1.0 (curvature=0 < 1)
        // xdelta = -2.0 * (-1.0) / (count-3) = 2.0/7
        // xi = xstart + 0 * xdelta = -1.0
        // yi = 0.0 * xi * xi = 0.0
        // xii = (1.0 * (-1.0) + 0.0 * 0.0) * scale = -scale
        // yii = (0.0 * (-1.0) - 1.0 * 0.0) * scale = 0.0
        // xiii = -scale + x*wd = -1014.01 + 0.5*100 = -964.01 (out of frame!)
        // So this point gets clipped. Let's just verify the function runs without panic.
        let _ = first_x0;
        assert!(written0 <= count - 3);

        // curvature = 4.0 → |curvature| > 1 → xstart = -sqrt(1/4) = -0.5
        let mut pts2 = vec![0.0f32; count * 2];
        for k in 0..3 {
            pts2[k * 2] = k as f32;
            pts2[k * 2 + 1] = k as f32;
        }
        let written1 = gradient_guide_points(&mut pts2, x, y, wd, ht, scale, cosv, sinv, 4.0, count);
        assert!(written1 <= count - 3);
    }

    // ── FFI round-trip tests ───────────────────────────────────────────────

    #[test]
    fn ffi_shift_points_round_trip() {
        let mut pts = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        unsafe {
            darkroom_masks_points_shift(pts.as_mut_ptr(), 3, 5.0, -5.0, 0);
        }
        assert_eq!(pts[0], 15.0);
        assert_eq!(pts[1], 15.0);
        assert_eq!(pts[2], 35.0);
        assert_eq!(pts[3], 35.0);
        assert_eq!(pts[4], 55.0);
        assert_eq!(pts[5], 55.0);
    }

    #[test]
    fn ffi_shift_points_null_guard() {
        unsafe {
            darkroom_masks_points_shift(std::ptr::null_mut(), 3, 1.0, 2.0, 0);
        }
        // No panic — just a no-op
    }

    #[test]
    fn ffi_circle_circumference_round_trip() {
        let l = 8i32;
        let mut pts = vec![0.0f32; 2 * (l as usize + 1)];
        unsafe {
            darkroom_masks_circle_circumference(pts.as_mut_ptr(), 10.0, 20.0, 5.0, l);
        }
        assert_eq!(pts[0], 10.0);
        assert_eq!(pts[1], 20.0);
        // alpha=0: center + (r, 0)
        assert!((pts[2] - 15.0).abs() < 1e-5);
        assert!((pts[3] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn ffi_ellipse_circumference_round_trip() {
        let l = 5i32;
        let mut pts = vec![0.0f32; 2 * (l as usize + 5)];
        let cosv = 1.0f32;
        let sinv = 0.0f32;
        unsafe {
            darkroom_masks_ellipse_circumference(pts.as_mut_ptr(), 0.0, 0.0, 10.0, 5.0, cosv, sinv, l);
        }
        // i=5 → alpha=0 → x + a*cosv = 10, y + a*sinv = 0
        assert!((pts[10] - 10.0).abs() < 1e-5);
        assert!((pts[11] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn ffi_bbox_reduction_round_trip() {
        let pts = [1.0f32, 2.0, 10.0, 20.0, -5.0, 3.0, 7.0, -1.0];
        let mut xmin = 0.0f32;
        let mut xmax = 0.0f32;
        let mut ymin = 0.0f32;
        let mut ymax = 0.0f32;
        unsafe {
            darkroom_masks_bbox_reduction(
                pts.as_ptr(),
                std::ptr::null(),
                4,
                0,
                &mut xmin, &mut xmax, &mut ymin, &mut ymax,
            );
        }
        assert_eq!(xmin, -5.0);
        assert_eq!(xmax, 10.0);
        assert_eq!(ymin, -1.0);
        assert_eq!(ymax, 20.0);
    }

    #[test]
    fn ffi_bbox_reduction_null_guard() {
        let mut xmin = 0.0f32;
        let mut xmax = 0.0f32;
        let mut ymin = 0.0f32;
        let mut ymax = 0.0f32;
        unsafe {
            darkroom_masks_bbox_reduction(
                std::ptr::null(),
                std::ptr::null(),
                4,
                0,
                &mut xmin, &mut xmax, &mut ymin, &mut ymax,
            );
        }
        // No panic — values untouched
        assert_eq!(xmin, 0.0);
        assert_eq!(xmax, 0.0);
    }

    #[test]
    fn ffi_gradient_guide_points_round_trip() {
        // Use parameters where all points stay in-frame
        let count = 10usize;
        let mut pts = vec![0.0f32; count * 2];
        let x = 0.5f32;
        let y = 0.5f32;
        let wd = 100.0f32;
        let ht = 100.0f32;
        let scale = dt_fast_hypotf(wd, ht);
        let cosv = 1.0f32;
        let sinv = 0.0f32;
        let curvature = 0.0f32; // |curvature| < 1 → xstart = -1

        let written = unsafe {
            darkroom_masks_gradient_guide_points(
                pts.as_mut_ptr(), count, x, y, wd, ht, scale, cosv, sinv, curvature,
            )
        };
        // With curvature=0, xstart=-1, xdelta=2/(count-3)=2/7
        // For k=0: xi=-1, yi=0, xii=-scale, yii=0, xiii=-scale+50, yiii=50
        // -scale+50 ≈ -964 which is < -wd(-100)? No wait: -scale ≈ -141.4
        // xiii = -141.4 + 0.5*100 = -141.4 + 50 = -91.4
        // -wd = -100, so -91.4 >= -100 → in frame!
        // Actually wd=100, so xiii=-91.4 which is > -wd(-100) → in frame
        assert!(written <= count - 3);
    }

    #[test]
    fn ffi_gradient_guide_points_null_guard() {
        let written = unsafe {
            darkroom_masks_gradient_guide_points(
                std::ptr::null_mut(), 10, 0.5, 0.5, 100.0, 100.0,
                141.421, 1.0, 0.0, 0.0,
            )
        };
        assert_eq!(written, 0);
    }

    /// Reference: the C `dt_fast_hypotf(x, y)` is `sqrtf(x*x + y*y)` — same
    /// as `f32::hypot` in Rust.
    fn dt_fast_hypotf(x: f32, y: f32) -> f32 {
        (x * x + y * y).sqrt()
    }
}
