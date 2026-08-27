//! Path drawn-mask rendering — port of the OMP loops in `src/develop/masks/path.c`.
//!
//! The pixelpipe callbacks that transform stroke points (`dt_dev_distort_*`) and
//! the `DT_INVALID_COORDINATE` deduplication logic stay in C — only the pure
//! per-segment falloff arithmetic and the even-odd fill loops move here, exposed
//! through `darkroom_masks_*` FFI exports exactly like every other replaced C loop.
//!
//! Three kernels are ported:
//! - [`path_falloff`] (whole-pipe), port of `_path_falloff` (path.c:3132).
//! - [`path_falloff_roi`], port of `_path_falloff_roi` (path.c:3558).
//! - [`fill_plain`] / [`fill_plain_roi`], ports of the even-odd fill loops
//!   (path.c:3327 and path.c:3835).
//!
//! Key differences from brush:
//! - Uses `sqrtf` (float) not `sqrt` (double) for segment length.
//! - No hardness/density parameters — opacity is always `1.0 - i/l`.
//! - Whole-pipe uses `fmaxf`; ROI uses the `MAX` macro — both are `f32::max`
//!   for non-NaN values.
//! - ROI direction uses `lx < 0 ? -1 : 1` (strict `<`, not brush's `<=`).
//! - Adjacent-pixel guards differ: whole-pipe uses `x > 0` / `y > 0`; ROI
//!   uses combined `x + dx >= 0 && x + dx < bw` / `y + dy >= 0 && y + dy < bh`.

// ── Per-segment kernels ────────────────────────────────────────────────────────

/// Whole-pipe falloff for a single path segment (port of `_path_falloff`,
/// path.c:3132).
///
/// `p0`/`p1` are integer segment endpoints (already truncated from the float
/// `points`/`border` arrays by the C caller). `posx`/`posy` offset the buffer
/// origin in image coordinates; `bw` is the buffer stride. Writes use `MAX()`
/// so overlapping segments accumulate the maximum opacity.
///
/// Unlike brush, this kernel has no hardness/density — opacity is `1.0 - i/l`.
/// The C uses `sqf(x)` = `(x)*(x)` which expands to int multiply when `x` is
/// int; `sqrtf` then converts the int sum to f32. We use `i64` intermediate to
/// avoid debug-mode overflow panics while producing identical f32 results.
pub fn path_falloff(
    buffer: &mut [f32],
    bw: i32,
    p0: [i32; 2],
    p1: [i32; 2],
    posx: i32,
    posy: i32,
) {
    // segment length: int l = sqrtf(sqf(dx) + sqf(dy)) + 1
    // sqf(int) = (int)*(int) = int multiply; sqrtf converts to f32.
    let dx = (p1[0] - p0[0]) as i64;
    let dy = (p1[1] - p0[1]) as i64;
    let l = ((dx * dx + dy * dy) as f32).sqrt() as i32 + 1;

    let lx = (p1[0] - p0[0]) as f32;
    let ly = (p1[1] - p0[1]) as f32;
    let l_f = l as f32;
    let bw_usize = bw as usize;

    for i in 0..l {
        let i_f = i as f32;
        // x = (int)((float)i * lx / (float)l) + p0[0] - posx
        let x = (i_f * lx / l_f) as i32 + p0[0] - posx;
        let y = (i_f * ly / l_f) as i32 + p0[1] - posy;
        let op = 1.0f32 - i_f / l_f;

        // buffer[y*bw + x] = fmaxf(buffer[idx], op)
        if x >= 0 && y >= 0 {
            let idx = (y as usize) * bw_usize + (x as usize);
            if idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
        // adjacent pixel (left) — C: if(x > 0)
        if x > 0 {
            let idx = (y as usize) * bw_usize + (x - 1) as usize;
            if y >= 0 && idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
        // adjacent pixel (above) — C: if(y > 0)
        if y > 0 {
            let idx = (y - 1) as usize * bw_usize + (x as usize);
            if x >= 0 && idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
    }
}

/// ROI-path falloff for a single path segment (port of `_path_falloff_roi`,
/// path.c:3558).
///
/// Unlike [`path_falloff`], this variant has per-pixel bounds checking
/// `x >= 0 && x < bw && y >= 0 && y < bh` and writes to adjacent pixels in
/// the stepping direction (`dx`, `dy`). Uses `lx < 0 ? -1 : 1` (strict `<`).
/// No `posx`/`posy` offset — coordinates are already in ROI buffer space.
pub fn path_falloff_roi(
    buffer: &mut [f32],
    bw: i32,
    bh: i32,
    p0: [i32; 2],
    p1: [i32; 2],
) {
    // C uses direct int multiply (not sqf): (p1[0]-p0[0])*(p1[0]-p0[0])
    // Result is int; sqrtf converts to f32. i64 intermediate avoids overflow.
    let dx = (p1[0] - p0[0]) as i64;
    let dy = (p1[1] - p0[1]) as i64;
    let l = ((dx * dx + dy * dy) as f32).sqrt() as i32 + 1;

    let lx = (p1[0] - p0[0]) as f32;
    let ly = (p1[1] - p0[1]) as f32;
    let l_f = l as f32;

    // strict < (not <=): path uses lx < 0 ? -1 : 1
    let dx_step = if lx < 0.0 { -1 } else { 1 };
    let dy_step = if ly < 0.0 { -1 } else { 1 };

    let bw_usize = bw as usize;

    for i in 0..l {
        let i_f = i as f32;
        // x = (int)((float)i * lx / (float)l) + p0[0]
        let x = (i_f * lx / l_f) as i32 + p0[0];
        let y = (i_f * ly / l_f) as i32 + p0[1];
        // op = 1.0f - (float)i / (float)l
        let op = 1.0f32 - i_f / l_f;

        // C: float *buf = buffer + (size_t)y * bw + x;
        // Then bounds-checked writes to buf[0], buf[dx], buf[dpy].
        // We compute indices directly to avoid underflows.

        // buf[0] = buffer[y*bw + x]
        if x >= 0 && x < bw && y >= 0 && y < bh {
            let idx = (y as usize) * bw_usize + (x as usize);
            buffer[idx] = buffer[idx].max(op);
        }
        // buf[dx] = buffer[y*bw + x + dx]
        if x + dx_step >= 0 && x + dx_step < bw && y >= 0 && y < bh {
            let idx = (y as usize) * bw_usize + ((x + dx_step) as usize);
            buffer[idx] = buffer[idx].max(op);
        }
        // buf[dpy] = buffer[(y+dy)*bw + x]
        if x >= 0 && x < bw && y + dy_step >= 0 && y + dy_step < bh {
            let idx = ((y + dy_step) as usize) * bw_usize + (x as usize);
            buffer[idx] = buffer[idx].max(op);
        }
    }
}

/// Even-odd fill for the whole-pipe path mask (port of the fill loop at
/// path.c:3327). Toggles `state` when `v == 1.0f`, writing 1.0 inside the path.
pub fn fill_plain(buffer: &mut [f32], wb: i32, hb: i32) {
    let wb_usize = wb as usize;
    for yy in 0..hb {
        let mut state = false;
        for xx in 0..wb {
            let idx = (yy as usize) * wb_usize + (xx as usize);
            let v = buffer[idx];
            if v == 1.0f32 {
                state = !state;
            }
            if state {
                buffer[idx] = 1.0f32;
            }
        }
    }
}

/// Even-odd fill within an ROI (port of the fill loop at path.c:3835).
/// Toggles `state` when `v > 0.5f`, writing 1.0 inside the path. Uses `width`
/// (not `wb`) as the buffer stride, and is bounded to `[xxmin..=xxmax] × [yymin..=yymax]`.
pub fn fill_plain_roi(
    buffer: &mut [f32],
    width: i32,
    xxmin: i32,
    xxmax: i32,
    yymin: i32,
    yymax: i32,
) {
    let width_usize = width as usize;
    for yy in yymin..=yymax {
        let mut state = false;
        for xx in xxmin..=xxmax {
            let index = (yy as usize) * width_usize + (xx as usize);
            let v = buffer[index];
            if v > 0.5f32 {
                state = !state;
            }
            if state {
                buffer[index] = 1.0f32;
            }
        }
    }
}

// ── FFI exports ────────────────────────────────────────────────────────────────

/// # Safety
/// `buffer` must hold at least `bw*bh` floats. `p0`/`p1` are integer segment
/// endpoints (already truncated from float by the C caller). `posx`/`posy`
/// offset the buffer origin in image coordinates.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_path_falloff(
    buffer: *mut f32,
    bw: i32,
    bh: i32,
    p0x: i32,
    p0y: i32,
    p1x: i32,
    p1y: i32,
    posx: i32,
    posy: i32,
) {
    if buffer.is_null() {
        return;
    }
    if bw <= 0 || bh <= 0 {
        return;
    }
    let Some(total) = bw.checked_mul(bh) else { return };
    let buffer = std::slice::from_raw_parts_mut(buffer, total as usize);

    path_falloff(buffer, bw, [p0x, p0y], [p1x, p1y], posx, posy);
}

/// # Safety
/// `buffer` must hold at least `bw*bh` floats. `segments` must hold at least
/// `4*nsegments` ints (each segment is [p0x, p0y, p1x, p1y]). Replaces the
/// OMP loop at path.c:3918 that calls `_path_falloff_roi` per segment.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_path_falloff_roi(
    buffer: *mut f32,
    bw: i32,
    bh: i32,
    segments: *const i32,
    nsegments: i32,
) {
    if buffer.is_null() || segments.is_null() {
        return;
    }
    if bw <= 0 || bh <= 0 || nsegments <= 0 {
        return;
    }
    let Some(total) = bw.checked_mul(bh) else { return };
    let buffer = std::slice::from_raw_parts_mut(buffer, total as usize);
    let n = nsegments as usize;
    let segments = std::slice::from_raw_parts(segments, 4 * n);

    for s in 0..n {
        let base = s * 4;
        let p0 = [segments[base], segments[base + 1]];
        let p1 = [segments[base + 2], segments[base + 3]];
        path_falloff_roi(buffer, bw, bh, p0, p1);
    }
}

/// # Safety
/// `buffer` must hold at least `wb*hb` floats. Replaces the OMP loop at
/// path.c:3327 that does the whole-pipe even-odd fill.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_path_fill_plain(
    buffer: *mut f32,
    wb: i32,
    hb: i32,
) {
    if buffer.is_null() {
        return;
    }
    if wb <= 0 || hb <= 0 {
        return;
    }
    let Some(total) = wb.checked_mul(hb) else { return };
    let buffer = std::slice::from_raw_parts_mut(buffer, total as usize);

    fill_plain(buffer, wb, hb);
}

/// # Safety
/// `buffer` must hold at least `width * (yymax+1)` floats (stride `width`).
/// Replaces the OMP loop at path.c:3835 that does the ROI even-odd fill.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_path_fill_plain_roi(
    buffer: *mut f32,
    width: i32,
    xxmin: i32,
    xxmax: i32,
    yymin: i32,
    yymax: i32,
) {
    if buffer.is_null() {
        return;
    }
    if width <= 0 || xxmin > xxmax || yymin > yymax {
        return;
    }
    let buf_len = (yymax as usize + 1) * (width as usize);
    let buffer = std::slice::from_raw_parts_mut(buffer, buf_len);

    fill_plain_roi(buffer, width, xxmin, xxmax, yymin, yymax);
}

// ── Reference implementations for bit-exactness tests ─────────────────────────

/// Reference for `path_falloff` — mirrors `_path_falloff` in path.c:3132.
fn ref_path_falloff(
    buffer: &mut [f32],
    bw: i32,
    p0: [i32; 2],
    p1: [i32; 2],
    posx: i32,
    posy: i32,
) {
    let dx = (p1[0] - p0[0]) as i64;
    let dy = (p1[1] - p0[1]) as i64;
    let l = ((dx * dx + dy * dy) as f32).sqrt() as i32 + 1;

    let lx = (p1[0] - p0[0]) as f32;
    let ly = (p1[1] - p0[1]) as f32;
    let l_f = l as f32;
    let bw_usize = bw as usize;

    for i in 0..l {
        let i_f = i as f32;
        let x = (i_f * lx / l_f) as i32 + p0[0] - posx;
        let y = (i_f * ly / l_f) as i32 + p0[1] - posy;
        let op = 1.0f32 - i_f / l_f;

        if x >= 0 && y >= 0 {
            let idx = (y as usize) * bw_usize + (x as usize);
            if idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
        if x > 0 {
            let idx = (y as usize) * bw_usize + (x - 1) as usize;
            if y >= 0 && idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
        if y > 0 {
            let idx = (y - 1) as usize * bw_usize + (x as usize);
            if x >= 0 && idx < buffer.len() {
                buffer[idx] = buffer[idx].max(op);
            }
        }
    }
}

/// Reference for `path_falloff_roi` — mirrors `_path_falloff_roi` in path.c:3558.
fn ref_path_falloff_roi(
    buffer: &mut [f32],
    bw: i32,
    bh: i32,
    p0: [i32; 2],
    p1: [i32; 2],
) {
    let dx = (p1[0] - p0[0]) as i64;
    let dy = (p1[1] - p0[1]) as i64;
    let l = ((dx * dx + dy * dy) as f32).sqrt() as i32 + 1;

    let lx = (p1[0] - p0[0]) as f32;
    let ly = (p1[1] - p0[1]) as f32;
    let l_f = l as f32;

    let dx_step = if lx < 0.0 { -1 } else { 1 };
    let dy_step = if ly < 0.0 { -1 } else { 1 };

    let bw_usize = bw as usize;

    for i in 0..l {
        let i_f = i as f32;
        let x = (i_f * lx / l_f) as i32 + p0[0];
        let y = (i_f * ly / l_f) as i32 + p0[1];
        let op = 1.0f32 - i_f / l_f;

        if x >= 0 && x < bw && y >= 0 && y < bh {
            let idx = (y as usize) * bw_usize + (x as usize);
            buffer[idx] = buffer[idx].max(op);
        }
        if x + dx_step >= 0 && x + dx_step < bw && y >= 0 && y < bh {
            let idx = (y as usize) * bw_usize + ((x + dx_step) as usize);
            buffer[idx] = buffer[idx].max(op);
        }
        if x >= 0 && x < bw && y + dy_step >= 0 && y + dy_step < bh {
            let idx = ((y + dy_step) as usize) * bw_usize + (x as usize);
            buffer[idx] = buffer[idx].max(op);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── path_falloff (whole-pipe) tests ──────────────────────────────────────

    #[test]
    fn path_falloff_horizontal() {
        // Horizontal segment (3,5)→(8,5): dx=5, dy=0, l=sqrt(25)+1=6
        let mut buf = vec![0f32; 10 * 10];
        path_falloff(&mut buf, 10, [3, 5], [8, 5], 0, 0);

        // i=0: x=3, y=5, op=1.0 → buf[55]=1.0, buf[54]=1.0 (left adj)
        assert_eq!(buf[5 * 10 + 3], 1.0);
        assert_eq!(buf[5 * 10 + 2], 1.0); // adjacent left

        let mut ref_buf = vec![0f32; 10 * 10];
        ref_path_falloff(&mut ref_buf, 10, [3, 5], [8, 5], 0, 0);
        assert_eq!(buf, ref_buf, "horizontal mismatch vs reference");
    }

    #[test]
    fn path_falloff_diagonal() {
        let mut buf = vec![0f32; 20 * 20];
        path_falloff(&mut buf, 20, [2, 2], [10, 8], 0, 0);
        let mut ref_buf = vec![0f32; 20 * 20];
        ref_path_falloff(&mut ref_buf, 20, [2, 2], [10, 8], 0, 0);
        assert_eq!(buf, ref_buf, "diagonal mismatch");
    }

    #[test]
    fn path_falloff_with_offset() {
        // posx/posy offset the buffer origin
        let mut buf = vec![0f32; 10 * 10];
        path_falloff(&mut buf, 10, [12, 12], [17, 12], 10, 10);
        let mut ref_buf = vec![0f32; 10 * 10];
        ref_path_falloff(&mut ref_buf, 10, [12, 12], [17, 12], 10, 10);
        assert_eq!(buf, ref_buf, "offset mismatch");
    }

    #[test]
    fn path_falloff_single_pixel_segment() {
        // p0 == p1: dx=0, dy=0, l=1
        let mut buf = vec![0f32; 5 * 5];
        path_falloff(&mut buf, 5, [2, 2], [2, 2], 0, 0);
        // i=0: x=2, y=2, op=1.0 → buf[12]=1.0
        assert_eq!(buf[2 * 5 + 2], 1.0);
        // adjacent left (x>0): buf[11]
        assert_eq!(buf[2 * 5 + 1], 1.0);
        // adjacent above (y>0): buf[7]
        assert_eq!(buf[1 * 5 + 2], 1.0);
        // no other writes
        assert_eq!(buf[2 * 5 + 3], 0.0);
        assert_eq!(buf[3 * 5 + 2], 0.0);

        let mut ref_buf = vec![0f32; 5 * 5];
        ref_path_falloff(&mut ref_buf, 5, [2, 2], [2, 2], 0, 0);
        assert_eq!(buf, ref_buf);
    }

    #[test]
    fn path_falloff_overlapping_segments() {
        let mut buf = vec![0f32; 10 * 10];
        path_falloff(&mut buf, 10, [3, 5], [8, 5], 0, 0);
        path_falloff(&mut buf, 10, [3, 5], [8, 5], 0, 0);

        let mut ref_buf = vec![0f32; 10 * 10];
        ref_path_falloff(&mut ref_buf, 10, [3, 5], [8, 5], 0, 0);
        ref_path_falloff(&mut ref_buf, 10, [3, 5], [8, 5], 0, 0);
        assert_eq!(buf, ref_buf, "overlapping mismatch");
    }

    // ── path_falloff_roi tests ───────────────────────────────────────────────

    #[test]
    fn path_falloff_roi_inside_bounds() {
        let mut buf = vec![0f32; 20 * 20];
        path_falloff_roi(&mut buf, 20, 20, [2, 2], [15, 15]);
        let mut ref_buf = vec![0f32; 20 * 20];
        ref_path_falloff_roi(&mut ref_buf, 20, 20, [2, 2], [15, 15]);
        assert_eq!(buf, ref_buf, "ROI inside mismatch");
    }

    #[test]
    fn path_falloff_roi_all_directions() {
        for (p0, p1) in [
            ([2, 2], [15, 15]),  // SW→NE
            ([15, 15], [2, 2]),  // NE→SW
            ([2, 15], [15, 2]),  // NW→SE
            ([15, 2], [2, 15]),  // SE→NW
            ([5, 5], [5, 15]),   // vertical
            ([5, 15], [5, 5]),   // vertical reversed
            ([5, 5], [15, 5]),   // horizontal
            ([15, 5], [5, 5]),   // horizontal reversed
        ] {
            let mut buf = vec![0f32; 20 * 20];
            path_falloff_roi(&mut buf, 20, 20, p0, p1);
            let mut ref_buf = vec![0f32; 20 * 20];
            ref_path_falloff_roi(&mut ref_buf, 20, 20, p0, p1);
            assert_eq!(buf, ref_buf, "ROI direction mismatch for {p0:?}→{p1:?}");
        }
    }

    #[test]
    fn path_falloff_roi_clip_outside() {
        // Segment partially inside — adjacent writes should be clipped
        let mut buf = vec![0f32; 10 * 10];
        path_falloff_roi(&mut buf, 10, 10, [0, 0], [5, 5]);
        let mut ref_buf = vec![0f32; 10 * 10];
        ref_path_falloff_roi(&mut ref_buf, 10, 10, [0, 0], [5, 5]);
        assert_eq!(buf, ref_buf, "ROI clip mismatch");
    }

    #[test]
    fn path_falloff_roi_single_pixel() {
        let mut buf = vec![0f32; 5 * 5];
        path_falloff_roi(&mut buf, 5, 5, [2, 2], [2, 2]);
        // dx=0→lx=0.0, lx<0 is false → dx=1; dy=0→ly=0.0, ly<0 is false → dy=1
        // i=0: x=2, y=2, op=1.0 → buf[12]=1.0
        assert_eq!(buf[2 * 5 + 2], 1.0);
        // adj x: (3,2), in bounds → buf[13]=1.0
        assert_eq!(buf[2 * 5 + 3], 1.0);
        // adj y: (2,3), in bounds → buf[17]=1.0
        assert_eq!(buf[3 * 5 + 2], 1.0);

        let mut ref_buf = vec![0f32; 5 * 5];
        ref_path_falloff_roi(&mut ref_buf, 5, 5, [2, 2], [2, 2]);
        assert_eq!(buf, ref_buf);
    }

    // ── fill_plain tests ─────────────────────────────────────────────────────

    #[test]
    fn fill_plain_basic_even_odd() {
        // Set up a buffer with some 1.0f markers (even-odd trigger)
        let mut buf = vec![0f32; 5 * 5];
        buf[1] = 1.0f32; // column boundary
        buf[3] = 1.0f32; // column boundary
        fill_plain(&mut buf, 5, 5);

        // State toggles at x=1 (now 1.0), fills x=2
        // State toggles at x=3 (now 0.0), fills nothing more in row 0
        // Row 0: [0,1,1,1,0] — after toggle at 1: state=1, fills 2; at 3: state=0
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[1], 1.0);
        assert_eq!(buf[2], 1.0);
        assert_eq!(buf[3], 1.0);
        assert_eq!(buf[4], 0.0);
    }

    #[test]
    fn fill_plain_multiple_rows() {
        let mut buf = vec![0f32; 4 * 3];
        // Row 0: boundaries at x=1
        buf[1] = 1.0f32;
        // Row 1: boundaries at x=0 and x=2
        buf[4 + 0] = 1.0f32;
        buf[4 + 2] = 1.0f32;
        // Row 2: no boundaries
        fill_plain(&mut buf, 4, 3);

        // Row 0: toggle at 1 → state=1, fill 2,3; no toggle back → fill stays
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[1], 1.0);
        assert_eq!(buf[2], 1.0);
        assert_eq!(buf[3], 1.0);

        // Row 1: toggle at 0 → state=1, fill 1; toggle at 2 → state=0
        assert_eq!(buf[4], 1.0);
        assert_eq!(buf[5], 1.0);
        assert_eq!(buf[6], 1.0);
        assert_eq!(buf[7], 0.0);

        // Row 2: no boundaries → state stays 0
        for i in 0..4 {
            assert_eq!(buf[8 + i], 0.0);
        }
    }

    // ── fill_plain_roi tests ─────────────────────────────────────────────────

    #[test]
    fn fill_plain_roi_basic() {
        let mut buf = vec![0f32; 10 * 10];
        // Set boundary marker (v > 0.5f triggers toggle)
        buf[3 * 10 + 2] = 0.7f32; // > 0.5f triggers
        buf[3 * 10 + 7] = 0.6f32; // > 0.5f triggers
        fill_plain_roi(&mut buf, 10, 0, 9, 3, 3);

        // Row 3: toggle at 2 (state=1, overwrites 0.7→1.0), fill 3-6;
        // toggle at 7 (state=0, so 0.6 is NOT overwritten — stays 0.6)
        assert_eq!(buf[3 * 10 + 0], 0.0);
        assert_eq!(buf[3 * 10 + 2], 1.0); // was 0.7, overwritten by fill
        assert_eq!(buf[3 * 10 + 3], 1.0);
        assert_eq!(buf[3 * 10 + 6], 1.0);
        assert_eq!(buf[3 * 10 + 7], 0.6); // toggle makes state=0, not overwritten
        assert_eq!(buf[3 * 10 + 8], 0.0);
    }

    #[test]
    fn fill_plain_roi_uses_gt_half() {
        // v > 0.5f (not v == 1.0f like whole-pipe)
        let mut buf = vec![0f32; 5 * 5];
        buf[1] = 0.51f32; // just above 0.5f
        fill_plain_roi(&mut buf, 5, 0, 4, 0, 0);
        // toggle at x=1 (0.51 > 0.5) → state=1, overwrites 0.51 with 1.0, fills 2,3,4
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[1], 1.0); // was 0.51, overwritten by fill
        assert_eq!(buf[2], 1.0);
        assert_eq!(buf[4], 1.0);
    }

    // ── FFI round-trip tests ─────────────────────────────────────────────────

    #[test]
    fn ffi_falloff_round_trip() {
        unsafe {
            let mut buf = vec![0f32; 10 * 10];
            darkroom_masks_path_falloff(
                buf.as_mut_ptr(), 10, 10,
                3, 5, 8, 5,  // p0, p1
                0, 0,        // posx, posy
            );

            // Compare with direct call
            let mut direct = vec![0f32; 10 * 10];
            path_falloff(&mut direct, 10, [3, 5], [8, 5], 0, 0);
            assert_eq!(buf, direct, "FFI falloff mismatch vs direct");
        }
    }

    #[test]
    fn ffi_falloff_roi_round_trip() {
        unsafe {
            // segments: [p0x, p0y, p1x, p1y, p0x, p0y, p1x, p1y]
            let segments: Vec<i32> = vec![2, 2, 15, 15, 5, 5, 12, 8];
            let mut buf = vec![0f32; 20 * 20];
            darkroom_masks_path_falloff_roi(
                buf.as_mut_ptr(), 20, 20,
                segments.as_ptr(), 2, // 2 segments
            );

            let mut direct = vec![0f32; 20 * 20];
            path_falloff_roi(&mut direct, 20, 20, [2, 2], [15, 15]);
            path_falloff_roi(&mut direct, 20, 20, [5, 5], [12, 8]);
            assert_eq!(buf, direct, "FFI ROI falloff mismatch vs direct");
        }
    }

    #[test]
    fn ffi_fill_plain_round_trip() {
        unsafe {
            let mut buf = vec![0f32; 5 * 5];
            buf[1] = 1.0f32;
            buf[13] = 1.0f32;
            darkroom_masks_path_fill_plain(buf.as_mut_ptr(), 5, 5);

            let mut direct = vec![0f32; 5 * 5];
            direct[1] = 1.0f32;
            direct[13] = 1.0f32;
            fill_plain(&mut direct, 5, 5);
            assert_eq!(buf, direct, "FFI fill_plain mismatch");
        }
    }

    #[test]
    fn ffi_fill_plain_roi_round_trip() {
        unsafe {
            let mut buf = vec![0f32; 10 * 10];
            buf[3 * 10 + 2] = 0.7f32;
            buf[3 * 10 + 7] = 0.6f32;
            darkroom_masks_path_fill_plain_roi(buf.as_mut_ptr(), 10, 0, 9, 3, 3);

            let mut direct = vec![0f32; 10 * 10];
            direct[3 * 10 + 2] = 0.7f32;
            direct[3 * 10 + 7] = 0.6f32;
            fill_plain_roi(&mut direct, 10, 0, 9, 3, 3);
            assert_eq!(buf, direct, "FFI fill_plain_roi mismatch");
        }
    }

    #[test]
    fn ffi_null_guards() {
        unsafe {
            darkroom_masks_path_falloff(
                std::ptr::null_mut(), 10, 10, 3, 5, 8, 5, 0, 0,
            );
            darkroom_masks_path_falloff_roi(
                std::ptr::null_mut(), 10, 10, std::ptr::null(), 2,
            );
            darkroom_masks_path_fill_plain(std::ptr::null_mut(), 5, 5);
            darkroom_masks_path_fill_plain_roi(
                std::ptr::null_mut(), 10, 0, 9, 0, 9,
            );
        }
    }

    #[test]
    fn ffi_zero_dimension_guards() {
        unsafe {
            let mut buf = vec![0f32; 10];
            darkroom_masks_path_falloff(buf.as_mut_ptr(), 0, 0, 0, 0, 1, 1, 0, 0);
            assert!(buf.iter().all(|&v| v == 0.0));

            darkroom_masks_path_fill_plain(buf.as_mut_ptr(), 0, 5);
            assert!(buf.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn path_falloff_matches_reference_over_many_segments() {
        // Verify bit-exactness with varied segments
        let segments: Vec<([i32; 2], [i32; 2])> = vec![
            ([3, 5], [8, 5]),
            ([2, 2], [15, 15]),
            ([0, 0], [0, 0]),
            ([-2, 3], [7, 3]),
            ([5, 0], [5, 9]),
        ];
        for (p0, p1) in segments {
            let mut buf = vec![0f32; 30 * 30];
            path_falloff(&mut buf, 30, p0, p1, 0, 0);
            let mut ref_buf = vec![0f32; 30 * 30];
            ref_path_falloff(&mut ref_buf, 30, p0, p1, 0, 0);
            assert_eq!(buf, ref_buf, "mismatch for p0={p0:?} p1={p1:?}");
        }
    }

    #[test]
    fn path_falloff_roi_matches_reference_over_many_segments() {
        let segments: Vec<([i32; 2], [i32; 2])> = vec![
            ([3, 5], [8, 5]),
            ([2, 2], [15, 15]),
            ([0, 0], [0, 0]),
            ([0, 0], [5, 5]),
            ([5, 5], [0, 0]),
        ];
        for (p0, p1) in segments {
            let mut buf = vec![0f32; 30 * 30];
            path_falloff_roi(&mut buf, 30, 30, p0, p1);
            let mut ref_buf = vec![0f32; 30 * 30];
            ref_path_falloff_roi(&mut ref_buf, 30, 30, p0, p1);
            assert_eq!(buf, ref_buf, "ROI mismatch for p0={p0:?} p1={p1:?}");
        }
    }
}
