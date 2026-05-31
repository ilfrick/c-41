//! Shared geometry helpers for the distort_transform / distort_backtransform /
//! distort_mask functions in crop, flip, borders, enlargecanvas, rotatepixels.
//! All operate on a flat `[x0, y0, x1, y1, …]` coordinate buffer.

use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

// ── Placeholder IopProcess stubs ─────────────────────────────────────────────
// The actual pixel pipelines in these IOPs delegate to shared helpers (flip
// buffers, dt_iop_copy_image_with_border, dt_interpolation, …). Only the
// distort_* coordinate loops are ported here.

macro_rules! geom_iop_stub {
    ($name:ident, $str:expr) => {
        pub struct $name;
        impl IopProcess for $name {
            fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
                Err(crate::Error::Pipeline("not implemented".into()))
            }
            fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
                Err(crate::Error::Pipeline("not implemented".into()))
            }
            fn name(&self) -> &'static str { $str }
        }
    };
}

geom_iop_stub!(Crop,          "crop");
geom_iop_stub!(Flip,          "flip");
geom_iop_stub!(Borders,       "borders");
geom_iop_stub!(Enlargecanvas, "enlargecanvas");
geom_iop_stub!(Rotatepixels,  "rotatepixels");
geom_iop_stub!(Scalepixels,   "scalepixels");

// ── Coordinate-shift ─────────────────────────────────────────────────────────

/// Add `(dx, dy)` to every coordinate pair in `pts`.
///
/// Matches the `distort_transform` OMP loops in
/// `crop.c`, `borders.c`, `enlargecanvas.c` (all equivalent).
#[no_mangle]
pub unsafe extern "C" fn darkroom_geom_shift_coords(
    pts: *mut f32,
    points_count: usize,
    dx: f32,
    dy: f32,
) {
    if points_count == 0 { return; }
    let buf = std::slice::from_raw_parts_mut(pts, points_count * 2);
    let mut i = 0;
    while i < points_count * 2 {
        buf[i]     += dx;
        buf[i + 1] += dy;
        i += 2;
    }
}

/// Subtract `(dx, dy)` from every coordinate pair in `pts`.
/// Inverse of `darkroom_geom_shift_coords`.
///
/// Matches the `distort_backtransform` OMP loops in
/// `crop.c`, `borders.c`, `enlargecanvas.c`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_geom_unshift_coords(
    pts: *mut f32,
    points_count: usize,
    dx: f32,
    dy: f32,
) {
    darkroom_geom_shift_coords(pts, points_count, -dx, -dy);
}

// ── Orientation / flip ───────────────────────────────────────────────────────

// C flag constants (same values as dt_image_orientation_t):
const FLIP_Y:  u32 = 1 << 0; // 1
const FLIP_X:  u32 = 1 << 1; // 2
const SWAP_XY: u32 = 1 << 2; // 4

/// Apply flip/transpose orientation to every coordinate pair.
///
/// Matches the `distort_transform` OMP loop in `flip.c:224`.
/// Forward transform:
///   if FLIP_X:    x = img_width  - x
///   if FLIP_Y:    y = img_height - y
///   if SWAP_XY:   (x, y) = (y, x)   applied AFTER the flips
#[no_mangle]
pub unsafe extern "C" fn darkroom_geom_flip_coords(
    pts: *mut f32,
    points_count: usize,
    orientation: u32,
    img_width: f32,
    img_height: f32,
) {
    if points_count == 0 || orientation == 0 { return; }
    let buf = std::slice::from_raw_parts_mut(pts, points_count * 2);
    let flip_x  = (orientation & FLIP_X)  != 0;
    let flip_y  = (orientation & FLIP_Y)  != 0;
    let swap_xy = (orientation & SWAP_XY) != 0;

    let mut i = 0;
    while i < points_count * 2 {
        let mut x = buf[i];
        let mut y = buf[i + 1];
        if flip_x { x = img_width  - x; }
        if flip_y { y = img_height - y; }
        if swap_xy { std::mem::swap(&mut x, &mut y); }
        buf[i]     = x;
        buf[i + 1] = y;
        i += 2;
    }
}

/// Inverse of `darkroom_geom_flip_coords` (backtransform).
///
/// The C source applies SWAP_XY FIRST, then the individual flips.
/// Matches the `distort_backtransform` OMP loop in `flip.c:259`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_geom_unflip_coords(
    pts: *mut f32,
    points_count: usize,
    orientation: u32,
    img_width: f32,
    img_height: f32,
) {
    if points_count == 0 || orientation == 0 { return; }
    let buf = std::slice::from_raw_parts_mut(pts, points_count * 2);
    let flip_x  = (orientation & FLIP_X)  != 0;
    let flip_y  = (orientation & FLIP_Y)  != 0;
    let swap_xy = (orientation & SWAP_XY) != 0;

    let mut i = 0;
    while i < points_count * 2 {
        // Inverse order: undo SWAP_XY first, then flips
        let (mut x, mut y) = if swap_xy {
            (buf[i + 1], buf[i]) // swap back
        } else {
            (buf[i], buf[i + 1])
        };
        if flip_x { x = img_width  - x; }
        if flip_y { y = img_height - y; }
        buf[i]     = x;
        buf[i + 1] = y;
        i += 2;
    }
}

// ── Row-by-row blit ──────────────────────────────────────────────────────────

/// Copy a single-channel (float per pixel) input into an output buffer with
/// a border offset, one row at a time.
///
/// For each row j in 0..in_height:
///   memcpy(out[(j+border_y)*out_width + border_x ..], in[j*in_width ..], in_width floats)
///
/// Matches the `distort_mask` `DT_OMP_FOR()` in `borders.c:420` and
/// `enlargecanvas.c:324`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_geom_blit_rows(
    in_buf: *const f32,
    out_buf: *mut f32,
    in_width: usize,
    in_height: usize,
    out_width: usize,
    border_x: usize,
    border_y: usize,
) {
    if in_width == 0 || in_height == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf,  in_width * in_height);
    let out = std::slice::from_raw_parts_mut(out_buf, out_width * (in_height + border_y));

    for j in 0..in_height {
        let src = &inp[j * in_width..(j + 1) * in_width];
        let dst_start = (j + border_y) * out_width + border_x;
        out[dst_start..dst_start + in_width].copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_coords_adds_offset() {
        let mut pts = vec![1.0_f32, 2.0, 3.0, 4.0];
        unsafe { darkroom_geom_shift_coords(pts.as_mut_ptr(), 2, 10.0, 20.0); }
        assert_eq!(pts, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn unshift_is_inverse_of_shift() {
        let orig = vec![5.0_f32, 7.0];
        let mut pts = orig.clone();
        unsafe {
            darkroom_geom_shift_coords(pts.as_mut_ptr(), 1, 3.0, -1.0);
            darkroom_geom_unshift_coords(pts.as_mut_ptr(), 1, 3.0, -1.0);
        }
        assert!((pts[0] - orig[0]).abs() < 1e-6);
        assert!((pts[1] - orig[1]).abs() < 1e-6);
    }

    #[test]
    fn flip_x_mirrors_horizontally() {
        let mut pts = vec![3.0_f32, 5.0];
        unsafe { darkroom_geom_flip_coords(pts.as_mut_ptr(), 1, FLIP_X, 10.0, 20.0); }
        assert_eq!(pts, vec![7.0, 5.0]); // 10 - 3 = 7
    }

    #[test]
    fn flip_y_mirrors_vertically() {
        let mut pts = vec![3.0_f32, 5.0];
        unsafe { darkroom_geom_flip_coords(pts.as_mut_ptr(), 1, FLIP_Y, 10.0, 20.0); }
        assert_eq!(pts, vec![3.0, 15.0]); // 20 - 5 = 15
    }

    #[test]
    fn swap_xy_transposes() {
        let mut pts = vec![3.0_f32, 5.0];
        unsafe { darkroom_geom_flip_coords(pts.as_mut_ptr(), 1, SWAP_XY, 10.0, 20.0); }
        assert_eq!(pts, vec![5.0, 3.0]);
    }

    #[test]
    fn unflip_is_inverse_of_flip() {
        let orig = vec![3.0_f32, 5.0];
        let mut pts = orig.clone();
        let orientation = FLIP_X | FLIP_Y | SWAP_XY;
        unsafe {
            darkroom_geom_flip_coords(pts.as_mut_ptr(), 1, orientation, 10.0, 20.0);
            darkroom_geom_unflip_coords(pts.as_mut_ptr(), 1, orientation, 10.0, 20.0);
        }
        assert!((pts[0] - orig[0]).abs() < 1e-5, "x={} orig={}", pts[0], orig[0]);
        assert!((pts[1] - orig[1]).abs() < 1e-5, "y={} orig={}", pts[1], orig[1]);
    }

    #[test]
    fn blit_rows_copies_with_offset() {
        // 2×2 input → place at (1,1) in a 4×4 output
        let inp = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2×2
        let mut out = vec![0.0_f32; 4 * 4];
        unsafe {
            darkroom_geom_blit_rows(inp.as_ptr(), out.as_mut_ptr(), 2, 2, 4, 1, 1);
        }
        // row 0 of inp → out row 1, starting at col 1
        assert_eq!(out[1 * 4 + 1], 1.0); assert_eq!(out[1 * 4 + 2], 2.0);
        // row 1 of inp → out row 2, starting at col 1
        assert_eq!(out[2 * 4 + 1], 3.0); assert_eq!(out[2 * 4 + 2], 4.0);
        // rest untouched
        assert_eq!(out[0], 0.0);
    }
}
