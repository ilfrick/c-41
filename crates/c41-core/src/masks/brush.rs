//! Brush drawn-mask rendering — port of the OMP loops in
//! `src/develop/masks/brush.c` `_brush_get_mask` (whole-pipe form mask) and
//! `_brush_get_mask_roi` (ROI grid-sampled mask).
//!
//! The pixelpipe callbacks that transform stroke points (`_brush_get_pts_border`
//! and `_brush_bounding_box_raw`) stay in C; only the pure per-segment falloff
//! arithmetic moves here, exposed through `darkroom_masks_*` FFI exports.
//!
//! Two falloff kernels are ported because their opacity formulas differ:
//! - [`brush_falloff`] (whole-pipe) computes `density * (1 - k/soft)` as a single
//!   expression per pixel.
//! - [`brush_falloff_roi`] (ROI) accumulates opacity via repeated `op -= dop`
//!   subtraction, which is **not** bit-equivalent but must be matched exactly.
//!
//! The adjacent-pixel writes exist to fill gaps from integer truncation of the
//! float-stepped position (`// this one is to avoid gap due to int rounding`).
//! The whole-pipe kernel guards with `x > 0` / `y > 0` (no upper bound); the
//! ROI kernel guards with `x + dx >= 0 && x + dx < bw` / `y + dy >= 0 && y + dy < bh`.

/// Whole-pipe falloff for a single brush segment (port of `_brush_falloff`,
/// brush.c:2867).
///
/// `p0` and `p1` are integer segment endpoints (already truncated from the float
/// `points`/`border` arrays by the caller). `posx`/`posy` offset the buffer
/// origin in image coordinates; `bw` is the buffer stride. Writes use `MAX()`
/// so overlapping segments accumulate the maximum opacity.
pub fn brush_falloff(
    buffer: &mut [f32],
    bw: i32,
    p0: [i32; 2],
    p1: [i32; 2],
    posx: i32,
    posy: i32,
    hardness: f32,
    density: f32,
) {
    // segment length — int arithmetic matches C; use i64 to avoid overflow
    let dx = (p1[0] - p0[0]) as i64;
    let dy = (p1[1] - p0[1]) as i64;
    let l = ((dx * dx + dy * dy) as f64).sqrt() as i32 + 1;
    let solid = (l as f32 * hardness) as i32;
    let soft = l - solid;

    let lx = (p1[0] - p0[0]) as f32;
    let ly = (p1[1] - p0[1]) as f32;
    let l_f = l as f32;

    for i in 0..l {
        let i_f = i as f32;
        // integer arithmetic before float conversion, matching C:
        //   (int)((float)i * lx / (float)l) + p0[0] - posx
        let x = (i_f * lx / l_f) as i32 + p0[0] - posx;
        let y = (i_f * ly / l_f) as i32 + p0[1] - posy;

        // op = density * (i <= solid ? 1.0f : 1.0f - (float)(i-solid)/(float)soft)
        let op = density
            * if i <= solid {
                1.0f32
            } else {
                1.0f32 - (i - solid) as f32 / soft as f32
            };

        // buffer[y*bw + x] = MAX(buffer[y*bw+x], op)  — C has no bounds check,
        // but the Rust slice must stay in range.
        if x >= 0 && y >= 0 {
            let idx = (y as usize) * (bw as usize) + (x as usize);
            if idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
        // adjacent pixel (left) — C: if(x > 0)
        if x > 0 {
            let idx = (y as usize) * (bw as usize) + (x - 1) as usize;
            if y >= 0 && idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
        // adjacent pixel (above) — C: if(y > 0)
        if y > 0 {
            let idx = (y - 1) as usize * (bw as usize) + (x as usize);
            if x >= 0 && idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
    }
}

/// ROI-path falloff for a single brush segment (port of `_brush_falloff_roi`,
/// brush.c:2985).
///
/// Unlike [`brush_falloff`], this variant steps with normalised floats (`fx += lx`,
/// `fy += ly`), clamps each pixel to `[0, bw) × [0, bh)`, and accumulates
/// opacity via repeated `op -= dop` subtraction rather than a single expression.
pub fn brush_falloff_roi(
    buffer: &mut [f32],
    bw: i32,
    bh: i32,
    p0: [i32; 2],
    p1: [i32; 2],
    hardness: f32,
    density: f32,
) {
    // segment length — same as brush_falloff
    let dx = (p1[0] - p0[0]) as i64;
    let dy = (p1[1] - p0[1]) as i64;
    let l = ((dx * dx + dy * dy) as f64).sqrt() as i32 + 1;
    let solid = (hardness * l as f32) as i32;

    let lx = (p1[0] - p0[0]) as f32 / l as f32;
    let ly = (p1[1] - p0[1]) as f32 / l as f32;

    let dx_step = if lx <= 0.0 { -1 } else { 1 };
    let dy_step = if ly <= 0.0 { -1 } else { 1 };

    let mut fx = p0[0] as f32;
    let mut fy = p0[1] as f32;

    let mut op = density;
    let dop = density / (l - solid) as f32;

    let bw_usize = bw as usize;

    for i in 0..l {
        // C: const int x = fx;  — float→int truncates toward zero
        let x = fx as i32;
        let y = fy as i32;

        fx += lx;
        fy += ly;
        if i > solid {
            op -= dop;
        }

        // C: if(x < 0 || x >= bw || y < 0 || y >= bh) continue;
        if x < 0 || x >= bw || y < 0 || y >= bh {
            continue;
        }

        let idx = (y as usize) * bw_usize + (x as usize);
        // *buf = MAX(*buf, op)
        buffer[idx] = buffer[idx].max(op);

        // adjacent pixel in primary direction: buf[dpx] where dpx = dx
        let adj_x = x + dx_step;
        if adj_x >= 0 && adj_x < bw {
            let adj_idx = (y as usize) * bw_usize + (adj_x as usize);
            buffer[adj_idx] = buffer[adj_idx].max(op);
        }
        // adjacent pixel in secondary direction: buf[dpy] where dpy = dy * bw
        let adj_y = y + dy_step;
        if adj_y >= 0 && adj_y < bh {
            let adj_idx = (adj_y as usize) * bw_usize + (x as usize);
            buffer[adj_idx] = buffer[adj_idx].max(op);
        }
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `buffer` must hold at least `bw*bh` floats; `points`, `border`, and
/// `payload` must each hold at least `2*end_idx` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_brush_falloff(
    buffer: *mut f32,
    bw: i32,
    bh: i32,
    points: *const f32,
    border: *const f32,
    payload: *const f32,
    start_idx: i32,
    end_idx: i32,
    posx: i32,
    posy: i32,
) {
    if buffer.is_null() || points.is_null() || border.is_null() || payload.is_null() {
        return;
    }
    if bw <= 0 || bh <= 0 || end_idx <= start_idx || start_idx < 0 {
        return;
    }
    let n = end_idx as usize;
    let points = std::slice::from_raw_parts(points, 2 * n);
    let border = std::slice::from_raw_parts(border, 2 * n);
    let payload = std::slice::from_raw_parts(payload, 2 * n);
    let Some(total) = bw.checked_mul(bh) else { return };
    let buffer = std::slice::from_raw_parts_mut(buffer, total as usize);

    for i in start_idx..end_idx {
        let idx = i as usize;
        // float→int truncation matches C: p0[0] = points[i*2]
        let p0 = [points[2 * idx] as i32, points[2 * idx + 1] as i32];
        let p1 = [border[2 * idx] as i32, border[2 * idx + 1] as i32];
        brush_falloff(
            buffer, bw, p0, p1, posx, posy,
            payload[2 * idx], payload[2 * idx + 1],
        );
    }
}

/// # Safety
/// `buffer` must hold `bw*bh` floats; `points`, `border`, and `payload` must
/// each hold at least `2*end_idx` floats. The skip check (segment entirely
/// outside the ROI) is performed inside the Rust loop.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_brush_falloff_roi(
    buffer: *mut f32,
    bw: i32,
    bh: i32,
    points: *const f32,
    border: *const f32,
    payload: *const f32,
    start_idx: i32,
    end_idx: i32,
) {
    if buffer.is_null() || points.is_null() || border.is_null() || payload.is_null() {
        return;
    }
    if bw <= 0 || bh <= 0 || end_idx <= start_idx || start_idx < 0 {
        return;
    }

    let n = end_idx as usize;
    let points = std::slice::from_raw_parts(points, 2 * n);
    let border = std::slice::from_raw_parts(border, 2 * n);
    let payload = std::slice::from_raw_parts(payload, 2 * n);
    let Some(total) = bw.checked_mul(bh) else { return };
    let buffer = std::slice::from_raw_parts_mut(buffer, total as usize);

    for i in start_idx..end_idx {
        let idx = i as usize;
        // float→int truncation matches C: const int p0[] = { points[i*2], ... }
        let p0 = [points[2 * idx] as i32, points[2 * idx + 1] as i32];
        let p1 = [border[2 * idx] as i32, border[2 * idx + 1] as i32];

        // skip if segment is entirely outside ROI
        // C: if(MAX(p0[0],p1[0])<0 || MIN(p0[0],p1[0])>=width || MAX(p0[1],p1[1])<0 || MIN(p0[1],p1[1])>=height)
        if std::cmp::max(p0[0], p1[0]) < 0
            || std::cmp::min(p0[0], p1[0]) >= bw
            || std::cmp::max(p0[1], p1[1]) < 0
            || std::cmp::min(p0[1], p1[1]) >= bh
        {
            continue;
        }

        brush_falloff_roi(buffer, bw, bh, p0, p1, payload[2 * idx], payload[2 * idx + 1]);
    }
}

// ── Reference implementations for bit-exactness tests ───────────────────────

/// Reference for `brush_falloff` — mirrors `_brush_falloff` in brush.c:2867.
fn ref_brush_falloff(
    buffer: &mut [f32],
    bw: i32,
    p0: [i32; 2],
    p1: [i32; 2],
    posx: i32,
    posy: i32,
    hardness: f32,
    density: f32,
) {
    let dx = (p1[0] - p0[0]) as i64;
    let dy = (p1[1] - p0[1]) as i64;
    let l = ((dx * dx + dy * dy) as f64).sqrt() as i32 + 1;
    let solid = (l as f32 * hardness) as i32;
    let soft = l - solid;
    let lx = (p1[0] - p0[0]) as f32;
    let ly = (p1[1] - p0[1]) as f32;
    let l_f = l as f32;

    for i in 0..l {
        let i_f = i as f32;
        let x = (i_f * lx / l_f) as i32 + p0[0] - posx;
        let y = (i_f * ly / l_f) as i32 + p0[1] - posy;
        let op = density
            * if i <= solid {
                1.0f32
            } else {
                1.0f32 - (i - solid) as f32 / soft as f32
            };

        if x >= 0 && y >= 0 {
            let idx = (y as usize) * (bw as usize) + (x as usize);
            if idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
        if x > 0 {
            let idx = (y as usize) * (bw as usize) + (x - 1) as usize;
            if y >= 0 && idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
        if y > 0 {
            let idx = (y - 1) as usize * (bw as usize) + (x as usize);
            if x >= 0 && idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
    }
}

/// Reference for `brush_falloff_roi` — mirrors `_brush_falloff_roi` in brush.c:2985.
fn ref_brush_falloff_roi(
    buffer: &mut [f32],
    bw: i32,
    bh: i32,
    p0: [i32; 2],
    p1: [i32; 2],
    hardness: f32,
    density: f32,
) {
    let dx = (p1[0] - p0[0]) as i64;
    let dy = (p1[1] - p0[1]) as i64;
    let l = ((dx * dx + dy * dy) as f64).sqrt() as i32 + 1;
    let solid = (hardness * l as f32) as i32;
    let lx = (p1[0] - p0[0]) as f32 / l as f32;
    let ly = (p1[1] - p0[1]) as f32 / l as f32;
    let dx_step = if lx <= 0.0 { -1 } else { 1 };
    let dy_step = if ly <= 0.0 { -1 } else { 1 };
    let mut fx = p0[0] as f32;
    let mut fy = p0[1] as f32;
    let mut op = density;
    let dop = density / (l - solid) as f32;
    let bw_usize = bw as usize;

    for i in 0..l {
        let x = fx as i32;
        let y = fy as i32;
        fx += lx;
        fy += ly;
        if i > solid {
            op -= dop;
        }
        if x < 0 || x >= bw || y < 0 || y >= bh {
            continue;
        }
        let idx = (y as usize) * bw_usize + (x as usize);
        buffer[idx] = buffer[idx].max(op);
        let adj_x = x + dx_step;
        if adj_x >= 0 && adj_x < bw {
            let adj_idx = (y as usize) * bw_usize + (adj_x as usize);
            buffer[adj_idx] = buffer[adj_idx].max(op);
        }
        let adj_y = y + dy_step;
        if adj_y >= 0 && adj_y < bh {
            let adj_idx = (adj_y as usize) * bw_usize + (x as usize);
            buffer[adj_idx] = buffer[adj_idx].max(op);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_util::lcg_fill;

    #[test]
    fn brush_falloff_horizontal_segment() {
        // Horizontal segment from (3,5) to (8,5): dx=5, dy=0, l=sqrt(25)+1=6
        // solid=3 (hardness 0.5), soft=3
        let mut buf = vec![0f32; 10 * 10];
        brush_falloff(&mut buf, 10, [3, 5], [8, 5], 0, 0, 0.5, 1.0);

        // i=0: x=3, y=5, op=1.0 (i<=solid) → buf[55]=1.0, buf[54]=1.0 (adj left)
        assert_eq!(buf[55], 1.0);
        assert_eq!(buf[54], 1.0);

        // All solid pixels should be 1.0
        for i in 0..4 {
            let x = (i as f32 * 5.0 / 6.0) as i32 + 3;
            assert_eq!(buf[5 * 10 + x as usize], 1.0, "solid pixel at x={x}");
        }

        // Verify reference matches
        let mut ref_buf = vec![0f32; 10 * 10];
        ref_brush_falloff(&mut ref_buf, 10, [3, 5], [8, 5], 0, 0, 0.5, 1.0);
        assert_eq!(buf, ref_buf, "whole-pipe mismatch vs reference");
    }

    #[test]
    fn brush_falloff_vertical_segment() {
        // Vertical segment from (5,2) to (5,7): dx=0, dy=5, l=sqrt(25)+1=6
        let mut buf = vec![0f32; 10 * 10];
        brush_falloff(&mut buf, 10, [5, 2], [5, 7], 0, 0, 1.0, 0.8);

        let mut ref_buf = vec![0f32; 10 * 10];
        ref_brush_falloff(&mut ref_buf, 10, [5, 2], [5, 7], 0, 0, 1.0, 0.8);
        assert_eq!(buf, ref_buf, "vertical mismatch vs reference");
    }

    #[test]
    fn brush_falloff_with_offset() {
        // Test with posx/posy offset (buffer origin != image origin)
        let mut buf = vec![0f32; 10 * 10];
        brush_falloff(&mut buf, 10, [10, 10], [15, 10], 0, 0, 0.7, 0.9);
        let mut ref_buf = vec![0f32; 10 * 10];
        ref_brush_falloff(&mut ref_buf, 10, [10, 10], [15, 10], 0, 0, 0.7, 0.9);
        assert_eq!(buf, ref_buf, "offset mismatch vs reference");
    }

    #[test]
    fn brush_falloff_single_pixel_segment() {
        // p0 == p1: dx=0, dy=0, l=1, solid=(1*0.5)as i32=0, soft=1
        // i=0: op = density (i <= solid is 0<=0 true)
        let mut buf = vec![0f32; 5 * 5];
        brush_falloff(&mut buf, 5, [2, 2], [2, 2], 0, 0, 0.5, 0.7);
        assert_eq!(buf[2 * 5 + 2], 0.7); // main pixel
        assert_eq!(buf[2 * 5 + 1], 0.7); // left adjacent (x>0)
        assert_eq!(buf[1 * 5 + 2], 0.7); // top adjacent (y>0)
        assert_eq!(buf[2 * 5 + 3], 0.0); // right (not written)
        assert_eq!(buf[3 * 5 + 2], 0.0); // bottom (not written)

        let mut ref_buf = vec![0f32; 5 * 5];
        ref_brush_falloff(&mut ref_buf, 5, [2, 2], [2, 2], 0, 0, 0.5, 0.7);
        assert_eq!(buf, ref_buf);
    }

    #[test]
    fn brush_falloff_hardness_one_full_density() {
        // hardness=1.0 → solid=l, soft=0, op=density for all pixels
        let mut buf = vec![0f32; 10 * 10];
        brush_falloff(&mut buf, 10, [3, 5], [8, 5], 0, 0, 1.0, 0.5);
        let mut ref_buf = vec![0f32; 10 * 10];
        ref_brush_falloff(&mut ref_buf, 10, [3, 5], [8, 5], 0, 0, 1.0, 0.5);
        assert_eq!(buf, ref_buf);
    }

    #[test]
    fn brush_falloff_overlapping_segments() {
        // Two segments sharing the same path — second has higher density
        let mut buf = vec![0f32; 10 * 10];
        brush_falloff(&mut buf, 10, [3, 5], [8, 5], 0, 0, 1.0, 0.5);
        brush_falloff(&mut buf, 10, [3, 5], [8, 5], 0, 0, 1.0, 0.8);

        let mut ref_buf = vec![0f32; 10 * 10];
        ref_brush_falloff(&mut ref_buf, 10, [3, 5], [8, 5], 0, 0, 1.0, 0.5);
        ref_brush_falloff(&mut ref_buf, 10, [3, 5], [8, 5], 0, 0, 1.0, 0.8);
        assert_eq!(buf, ref_buf);

        // Overlapping with MAX: density 0.8 should dominate 0.5
        assert_eq!(buf[5 * 10 + 3], 0.8);
    }

    #[test]
    fn brush_falloff_roi_inside_bounds() {
        let mut buf = vec![0f32; 20 * 20];
        brush_falloff_roi(&mut buf, 20, 20, [2, 2], [15, 15], 0.5, 1.0);
        let mut ref_buf = vec![0f32; 20 * 20];
        ref_brush_falloff_roi(&mut ref_buf, 20, 20, [2, 2], [15, 15], 0.5, 1.0);
        assert_eq!(buf, ref_buf, "ROI inside bounds mismatch");
    }

    #[test]
    fn brush_falloff_roi_different_directions() {
        // Test segments going in different directions
        for (p0, p1) in [
            ([2, 2], [15, 15]), // SW→NE
            ([15, 15], [2, 2]),  // NE→SW
            ([2, 15], [15, 2]),  // NW→SE
            ([15, 2], [2, 15]),  // SE→NW
            ([5, 5], [5, 15]),   // vertical
            ([5, 15], [5, 5]),   // vertical reversed
            ([5, 5], [15, 5]),   // horizontal
            ([15, 5], [5, 5]),   // horizontal reversed
        ] {
            let mut buf = vec![0f32; 20 * 20];
            brush_falloff_roi(&mut buf, 20, 20, p0, p1, 0.3, 0.9);
            let mut ref_buf = vec![0f32; 20 * 20];
            ref_brush_falloff_roi(&mut ref_buf, 20, 20, p0, p1, 0.3, 0.9);
            assert_eq!(buf, ref_buf, "ROI direction mismatch for {p0:?}→{p1:?}");
        }
    }

    #[test]
    fn brush_falloff_roi_skip_check_excludes_outside() {
        // Segment entirely outside ROI (all negative x)
        let mut buf = vec![0.0f32; 10 * 10];
        brush_falloff_roi(&mut buf, 10, 10, [-5, 5], [-1, 5], 0.5, 1.0);
        assert!(buf.iter().all(|&v| v == 0.0), "should skip segment entirely outside");

        // Segment entirely beyond width
        let mut buf2 = vec![0.0f32; 10 * 10];
        brush_falloff_roi(&mut buf2, 10, 10, [20, 5], [25, 5], 0.5, 1.0);
        assert!(buf2.iter().all(|&v| v == 0.0), "should skip segment beyond width");

        // Segment partially inside — should write
        let mut buf3 = vec![0.0f32; 10 * 10];
        brush_falloff_roi(&mut buf3, 10, 10, [-2, 5], [5, 5], 1.0, 1.0);
        assert!(buf3.iter().any(|&v| v > 0.0), "should write for partially inside segment");
    }

    #[test]
    fn brush_falloff_roi_single_pixel() {
        // p0 == p1: l=1, solid=0
        let mut buf = vec![0f32; 5 * 5];
        brush_falloff_roi(&mut buf, 5, 5, [2, 2], [2, 2], 0.5, 0.7);
        // i=0: x=2, y=2 (in bounds), op=0.7 (i<=solid is 0<=0 true, no op-=)
        assert_eq!(buf[2 * 5 + 2], 0.7);
        // adjacent: dx=1 (lx=0/1=0, lx<=0 → dx=-1... wait, lx = 0/1 = 0.0, lx <= 0 → dx=-1
        // dy: ly = 0/1 = 0.0, ly <= 0 → dy=-1
        // So adj pixels are at (1,2) and (2,1)
        assert_eq!(buf[2 * 5 + 1], 0.7); // x+dx = 2-1 = 1, in bounds [0,5)
        assert_eq!(buf[1 * 5 + 2], 0.7); // y+dy = 2-1 = 1, in bounds [0,5)
        assert_eq!(buf[2 * 5 + 3], 0.0); // not written (dx=-1)
        assert_eq!(buf[3 * 5 + 2], 0.0); // not written (dy=-1)

        let mut ref_buf = vec![0f32; 5 * 5];
        ref_brush_falloff_roi(&mut ref_buf, 5, 5, [2, 2], [2, 2], 0.5, 0.7);
        assert_eq!(buf, ref_buf);
    }

    #[test]
    fn brush_falloff_matches_reference_over_lcg_segments() {
        // Test many random-looking segments to verify bit-exactness
        let mut points = vec![0f32; 200];
        lcg_fill(&mut points, 0xC0FFEE, 100.0);
        let mut border = vec![0f32; 200];
        lcg_fill(&mut border, 0xDEAD, 100.0);
        let mut payload = vec![0f32; 200];
        lcg_fill(&mut payload, 0xBEEF, 1.0);

        for k in 0..10 {
            let p0 = [points[2*k] as i32, points[2*k+1] as i32];
            let p1 = [border[2*k] as i32, border[2*k+1] as i32];
            let hardness = 0.3 + 0.05 * (k as f32);
            let density = 0.5 + 0.04 * (k as f32);

            let mut buf = vec![0f32; 50 * 50];
            brush_falloff(&mut buf, 50, p0, p1, 0, 0, hardness, density);
            let mut ref_buf = vec![0f32; 50 * 50];
            ref_brush_falloff(&mut ref_buf, 50, p0, p1, 0, 0, hardness, density);
            assert_eq!(buf, ref_buf, "whole-pipe LCG mismatch at k={k}, p0={p0:?}, p1={p1:?}");

            let mut buf = vec![0f32; 50 * 50];
            brush_falloff_roi(&mut buf, 50, 50, p0, p1, hardness, density);
            let mut ref_buf = vec![0f32; 50 * 50];
            ref_brush_falloff_roi(&mut ref_buf, 50, 50, p0, p1, hardness, density);
            assert_eq!(buf, ref_buf, "ROI LCG mismatch at k={k}, p0={p0:?}, p1={p1:?}");
        }
    }

    #[test]
    fn brush_falloff_does_not_panic_on_out_of_bounds() {
        // Extreme coordinates entirely outside the buffer — should not panic,
        // should not write anything.
        let mut buf = vec![0f32; 5 * 5];
        brush_falloff(&mut buf, 5, [100, 0], [200, 0], 0, 0, 0.5, 1.0);
        assert!(buf.iter().all(|&v| v == 0.0));

        // ROI path with a segment crossing through the buffer — should write
        // but not panic.
        let mut buf2 = vec![0f32; 5 * 5];
        brush_falloff_roi(&mut buf2, 5, 5, [-100, 2], [200, 2], 0.5, 1.0);
        assert!(buf2.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn ffi_falloff_round_trip_matches_direct_call() {
        unsafe {
            let mut points = vec![0f32; 10];
            points[0] = 3.7; points[1] = 5.2;  // p0 → [3, 5]
            points[2] = 8.9; points[3] = 5.1;  // p1 → [8, 5]
            points[4] = 3.5; points[5] = 3.5;  // p0 → [3, 3]
            points[6] = 9.5; points[7] = 3.5;  // p1 → [9, 3]
            let border = points.clone();
            let mut payload = vec![0f32; 10];
            payload[0] = 0.5; payload[1] = 1.0;  // segment 0: h=0.5, d=1.0
            payload[2] = 1.0; payload[3] = 0.8;  // segment 1: h=1.0, d=0.8
            payload[4] = 0.7; payload[5] = 0.6;  // segment 2: h=0.7, d=0.6
            payload[6] = 0.3; payload[7] = 0.9;  // segment 3: h=0.3, d=0.9
            payload[8] = 0.0; payload[9] = 0.5;  // segment 4: h=0.0, d=0.5

            let mut ffi_buf = vec![0f32; 10 * 10];
            darkroom_masks_brush_falloff(
                ffi_buf.as_mut_ptr(), 10, 10,
                points.as_ptr(), border.as_ptr(), payload.as_ptr(),
                2, 5,  // start_idx=2, end_idx=5
                0, 0,  // posx=0, posy=0
            );

            // Direct per-segment calls with the same truncation
            let mut direct_buf = vec![0f32; 10 * 10];
            for i in 2..5 {
                let p0 = [points[2*i] as i32, points[2*i+1] as i32];
                let p1 = [border[2*i] as i32, border[2*i+1] as i32];
                brush_falloff(&mut direct_buf, 10, p0, p1, 0, 0,
                    payload[2*i], payload[2*i+1]);
            }

            assert_eq!(ffi_buf, direct_buf, "FFI mismatch vs direct call");
        }
    }

    #[test]
    fn ffi_falloff_roi_round_trip_matches_direct_call() {
        unsafe {
            let mut points = vec![0f32; 10];
            points[0] = 3.7; points[1] = 5.2;
            points[2] = 8.9; points[3] = 5.1;
            points[4] = -5.0; points[5] = 5.0;  // outside ROI → skipped
            points[6] = 20.0; points[7] = 5.0;  // outside ROI → skipped
            let border = points.clone();
            let mut payload = vec![0f32; 10];
            payload[0] = 0.5; payload[1] = 1.0;
            payload[2] = 1.0; payload[3] = 0.8;
            payload[4] = 0.7; payload[5] = 0.6;
            payload[6] = 0.3; payload[7] = 0.9;

            let mut ffi_buf = vec![0f32; 10 * 10];
            darkroom_masks_brush_falloff_roi(
                ffi_buf.as_mut_ptr(), 10, 10,
                points.as_ptr(), border.as_ptr(), payload.as_ptr(),
                0, 4,  // start_idx=0, end_idx=4
            );

            // Direct per-segment calls with skip check
            let mut direct_buf = vec![0f32; 10 * 10];
            for i in 0..4 {
                let p0 = [points[2*i] as i32, points[2*i+1] as i32];
                let p1 = [border[2*i] as i32, border[2*i+1] as i32];
                if std::cmp::max(p0[0], p1[0]) < 0
                    || std::cmp::min(p0[0], p1[0]) >= 10
                    || std::cmp::max(p0[1], p1[1]) < 0
                    || std::cmp::min(p0[1], p1[1]) >= 10
                {
                    continue;
                }
                brush_falloff_roi(&mut direct_buf, 10, 10, p0, p1,
                    payload[2*i], payload[2*i+1]);
            }

            assert_eq!(ffi_buf, direct_buf, "FFI ROI mismatch vs direct call");
            // Segments 2 and 3 were outside ROI → no writes there
            // Only segments 0 and 1 contribute
        }
    }

    #[test]
    fn ffi_null_guards() {
        unsafe {
            darkroom_masks_brush_falloff(
                std::ptr::null_mut(), 10, 10,
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
                0, 5, 0, 0,
            );
            darkroom_masks_brush_falloff_roi(
                std::ptr::null_mut(), 10, 10,
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
                0, 5,
            );
        }
    }
}
