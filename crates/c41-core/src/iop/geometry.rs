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

// `CropIop` (not `Crop`) to avoid a homonym with the preview-side crop pass
// `crate::geometry::Crop`; the IOP `name()` string "crop" is unchanged.
geom_iop_stub!(CropIop,       "crop");
geom_iop_stub!(Flip,          "flip");
geom_iop_stub!(Borders,       "borders");
geom_iop_stub!(Enlargecanvas, "enlargecanvas");
geom_iop_stub!(Rotatepixels,  "rotatepixels");
geom_iop_stub!(Scalepixels,   "scalepixels");
// IOPs whose process() delegates entirely to shared C utilities (DWT, bilateral,
// NLM, equaliser filter, clamp-and-scale) — no OMP loops remain to migrate in
// the IOP file itself. Registered as stubs to track coverage.
geom_iop_stub!(Equalizer,     "equalizer");
geom_iop_stub!(Finalscale,    "finalscale");
geom_iop_stub!(Nlmeans,       "nlmeans");
geom_iop_stub!(Bilat,         "bilat");
geom_iop_stub!(Spots,         "spots");
// IOPs with remaining OMP loops blocked on infrastructure not yet in Rust
// (color-space transforms, interpolation, bilateral grid, NLM, keystone, etc.)
geom_iop_stub!(Demosaic,            "demosaic");
geom_iop_stub!(Ashift,              "ashift");
geom_iop_stub!(Clipping,            "clipping");
geom_iop_stub!(Colorreconstruction, "colorreconstruction");
geom_iop_stub!(Colorharmonizer,     "colorharmonizer");
geom_iop_stub!(Colorbalancergb,     "colorbalancergb");
geom_iop_stub!(Liquify,             "liquify");
geom_iop_stub!(Denoiseprofile,      "denoiseprofile");
geom_iop_stub!(Retouch,             "retouch");
geom_iop_stub!(Sharpen,             "sharpen");
geom_iop_stub!(Rawoverexposed,      "rawoverexposed");

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

// ── 2×2 matrix × 2D coordinate ───────────────────────────────────────────────

/// Apply a 2×2 rotation matrix + translation to every coordinate pair.
///
/// For each point i:
///   pi = (x - rx*scale, y - ry*scale)
///   o  = M * pi
/// where M = [m[0], m[1]; m[2], m[3]] (row-major 2×2).
///
/// Matches `transform()` called from `distort_transform` in
/// src/iop/rotatepixels.c:138.
#[no_mangle]
pub unsafe extern "C" fn darkroom_geom_rotate_coords(
    pts: *mut f32,
    points_count: usize,
    m: *const f32,  // 4 floats: [m00, m01, m10, m11]
    rx: f32,
    ry: f32,
    scale: f32,
) {
    if points_count == 0 { return; }
    let buf = std::slice::from_raw_parts_mut(pts, points_count * 2);
    let mat = std::slice::from_raw_parts(m, 4);
    let mut i = 0;
    while i < points_count * 2 {
        let pix = buf[i]     - rx * scale;
        let piy = buf[i + 1] - ry * scale;
        buf[i]     = pix * mat[0] + piy * mat[1];
        buf[i + 1] = pix * mat[2] + piy * mat[3];
        i += 2;
    }
}

/// Inverse rotation: apply the transposed matrix then add translation.
///
/// For each point i:
///   rt = [m[0], -m[1], -m[2], m[3]]  (transpose of rotation)
///   o  = rt * x
///   o += (rx*scale, ry*scale)
///
/// Matches `backtransform()` called from `distort_backtransform` in
/// src/iop/rotatepixels.c:162.
#[no_mangle]
pub unsafe extern "C" fn darkroom_geom_unrotate_coords(
    pts: *mut f32,
    points_count: usize,
    m: *const f32,
    rx: f32,
    ry: f32,
    scale: f32,
) {
    if points_count == 0 { return; }
    let buf = std::slice::from_raw_parts_mut(pts, points_count * 2);
    let mat = std::slice::from_raw_parts(m, 4);
    // transposed rotation matrix: rt = [m[0], -m[1], -m[2], m[3]]
    let rt = [mat[0], -mat[1], -mat[2], mat[3]];
    let mut i = 0;
    while i < points_count * 2 {
        let x = buf[i];
        let y = buf[i + 1];
        buf[i]     = x * rt[0] + y * rt[1] + rx * scale;
        buf[i + 1] = x * rt[2] + y * rt[3] + ry * scale;
        i += 2;
    }
}

/// Full rotatepixels process() loop.
///
/// For each output pixel (i, j):
///   pi  = (roi_out_x + i, roi_out_y + j)
///   po  = M^T * pi + (rx, ry) * scale      (inverse rotation / backtransform)
///   po -= (roi_in_x, roi_in_y)              (convert to roi-relative coords)
///   if po in [0, in_w) x [0, in_h): sample in_buf with interpolator
///   else: zero the output pixel
///
/// `m`: 4-float 2x2 rotation matrix (d->m, row-major).
/// `in_width/in_height`: roi_in dimensions (NOT full image).
/// `interp_type`: 0=bilinear 1=bicubic 2=lanczos2 3=lanczos3.
/// Matches the DT_OMP_FOR in src/iop/rotatepixels.c:256.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rotatepixels_process(
    in_buf:      *const f32,
    out_buf:     *mut f32,
    out_width:   i32,
    out_height:  i32,
    roi_out_x:   f32,
    roi_out_y:   f32,
    in_width:    i32,
    in_height:   i32,
    roi_in_x:    f32,
    roi_in_y:    f32,
    scale:       f32,
    m:           *const f32,
    rx:          f32,
    ry:          f32,
    interp_type: u32,
) {
    if out_width <= 0 || out_height <= 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf, (in_width * in_height * 4) as usize);
    let outp = std::slice::from_raw_parts_mut(out_buf, (out_width * out_height * 4) as usize);
    let mref = std::slice::from_raw_parts(m, 4);

    // Transpose of 2x2 rotation matrix with sign convention from C backtransform():
    //   rt = {m[0], -m[1], -m[2], m[3]}
    let rt = [mref[0], -mref[1], -mref[2], mref[3]];

    let ls = in_width * 4; // linestride in floats
    let iw = in_width  as f32;
    let ih = in_height as f32;

    for j in 0..out_height as usize {
        let piy = roi_out_y + j as f32;
        for i in 0..out_width as usize {
            let pix = roi_out_x + i as f32;

            let pox = rt[0] * pix + rt[1] * piy + rx * scale - roi_in_x;
            let poy = rt[2] * pix + rt[3] * piy + ry * scale - roi_in_y;

            let base = (j * out_width as usize + i) * 4;
            if pox >= 0.0 && poy >= 0.0 && pox < iw && poy < ih {
                let px = crate::interp::compute_pixel4c(
                    inp, pox, poy, in_width, in_height, ls, interp_type,
                );
                outp[base]     = px[0];
                outp[base + 1] = px[1];
                outp[base + 2] = px[2];
                outp[base + 3] = px[3];
            } else {
                outp[base]     = 0.0;
                outp[base + 1] = 0.0;
                outp[base + 2] = 0.0;
                outp[base + 3] = 0.0;
            }
        }
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
    fn rotate_coords_applies_matrix_and_translation() {
        // Identity matrix, no translation: output = input
        let mut pts = vec![3.0_f32, 4.0];
        let m = [1.0_f32, 0.0, 0.0, 1.0]; // identity
        unsafe { darkroom_geom_rotate_coords(pts.as_mut_ptr(), 1, m.as_ptr(), 0.0, 0.0, 1.0); }
        assert!((pts[0] - 3.0).abs() < 1e-6);
        assert!((pts[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn unrotate_is_inverse_of_rotate() {
        // 90° rotation matrix: [[0,-1],[1,0]], translation (1,2)
        let m = [0.0_f32, -1.0, 1.0, 0.0];
        let orig = vec![3.0_f32, 4.0];
        let mut pts = orig.clone();
        unsafe {
            darkroom_geom_rotate_coords(pts.as_mut_ptr(), 1, m.as_ptr(), 1.0, 2.0, 1.0);
            darkroom_geom_unrotate_coords(pts.as_mut_ptr(), 1, m.as_ptr(), 1.0, 2.0, 1.0);
        }
        assert!((pts[0] - orig[0]).abs() < 1e-5, "x={}", pts[0]);
        assert!((pts[1] - orig[1]).abs() < 1e-5, "y={}", pts[1]);
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
