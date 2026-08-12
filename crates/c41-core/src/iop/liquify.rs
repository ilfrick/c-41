//! Liquify IOP helpers -- 6 DT_OMP_FOR loops, all using float complex pairs.
//! C's `float complex` is stored as two consecutive f32 (real, imag).

use crate::{params::IopParams, roi::RoiIn, Result, interp};
use super::{ClBuffer, IopProcess};

pub struct Liquify;
impl IopProcess for Liquify {
    fn name(&self) -> &'static str { "liquify" }
    fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("liquify: use C FFI path".into()))
    }
    fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("liquify: no OpenCL path".into()))
    }
}

// Complex helpers: treat float complex as [f32; 2] = [real, imag]
#[inline(always)]
fn re(p: &[f32; 2]) -> f32 { p[0] }
#[inline(always)]
fn im(p: &[f32; 2]) -> f32 { p[1] }

/// Apply a round displacement stamp to the global distortion map.
///
/// `center` points to the (x=0,y=0) position in the map.
/// For warp_type == 0 (LINEAR): writes -(strength_re, strength_im)*LUT[d] to 4 quadrants.
/// For warp_type == 1 (RADIAL): writes ±abs_strength*LUT[d]/r*(x, ±y) to quadrants.
/// Matches apply_round_stamp DT_OMP_FOR at liquify.c:956.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_apply_stamp(
    center:      *mut f32,       // float complex* (stride 2) at map center
    global_width: usize,
    iradius:     usize,
    lookup_table: *const f32,
    table_size:  usize,
    oversample:  usize,
    warp_type:   i32,            // 0=linear, 1=radial
    strength_re: f32,            // real part of -(strength) for linear
    strength_im: f32,            // imag part
    abs_strength: f32,           // for radial
) {
    let stride = 2usize;
    let lt = std::slice::from_raw_parts(lookup_table, table_size);

    for y in 0..=iradius {
        let y2 = (y * y) as f32;
        let mut x = 0usize;
        loop {
            if x > iradius { break; }
            let dist = ((x * x) as f32 + y2).sqrt();
            let idist = (dist * oversample as f32).round() as usize;
            if idist >= table_size { break; }

            // Quadrant pointers (2 floats each):
            // q1: center - y*w + x   (positive x, negative y)
            // q2: center - y*w - x
            // q3: center + y*w - x
            // q4: center + y*w + x
            let base = center as *mut f32;
            let off = |dy: isize, dx: isize| -> *mut f32 {
                base.offset((dy * global_width as isize + dx) * stride as isize)
            };
            let q1 = off(-(y as isize), x as isize);
            let q2 = off(-(y as isize), -(x as isize));
            let q3 = off(y as isize,    -(x as isize));
            let q4 = off(y as isize,    x as isize);

            let lv = lt[idist];
            if warp_type == 0 {
                let wr = -strength_re * lv;
                let wi = -strength_im * lv;
                let add = |p: *mut f32| { *p += wr; *p.add(1) += wi; };
                add(q1);
                if x != 0 { add(q2); }
                if x != 0 && y != 0 { add(q3); }
                if y != 0 { add(q4); }
            } else {
                let av = abs_strength * lv / iradius as f32;
                // q1 -= av*(x - y*i)
                *q1     -= av * x as f32;
                *q1.add(1) += av * y as f32;
                // q2 += av*(x + y*i)
                if x != 0 {
                    *q2     += av * x as f32;
                    *q2.add(1) -= av * y as f32;
                }
                // q3 += av*(x - y*i)
                if x != 0 && y != 0 {
                    *q3     += av * x as f32;
                    *q3.add(1) -= av * y as f32;
                }
                // q4 -= av*(x + y*i)
                if y != 0 {
                    *q4     -= av * x as f32;
                    *q4.add(1) += av * y as f32;
                }
            }
            x += 1;
        }
    }
}

/// Apply the global distortion map to the image (pixel warp).
/// `map` is a float complex array stored as interleaved [re, im] f32 pairs.
/// Matches _apply_global_distortion_map DT_OMP_FOR at liquify.c:1030.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_apply_map(
    in_buf:     *const f32,
    out_buf:    *mut f32,
    roi_in_x:   i32, roi_in_y: i32, roi_in_w: i32, roi_in_h: i32,
    roi_out_x:  i32, roi_out_y: i32, roi_out_w: i32, roi_out_h: i32,
    extent_x:   i32, extent_y: i32, extent_w: i32, extent_h: i32,
    map:        *const f32,   // float complex array (stride 2)
    ch:         i32,
    interp_type: u32,
) {
    let in_n   = (roi_in_w * roi_in_h * ch) as usize;
    let out_n  = (roi_out_w * roi_out_h * ch) as usize;
    let inp    = std::slice::from_raw_parts(in_buf, in_n);
    let outp   = std::slice::from_raw_parts_mut(out_buf, out_n);
    let ch_width = roi_in_w * ch;

    let min_y  = roi_out_y.max(extent_y) as usize;
    let max_y  = (roi_out_y + roi_out_h).min(extent_y + extent_h) as usize;
    let min_x  = roi_out_x.max(extent_x) as usize;
    let max_x  = (roi_out_x + roi_out_w).min(extent_x + extent_w) as usize;

    for y in min_y..max_y {
        let row_base = (y - extent_y as usize) * extent_w as usize + (min_x - extent_x as usize);
        let out_row  = (y - roi_out_y as usize) * roi_out_w as usize;

        for x in min_x..max_x {
            let map_off  = (row_base + (x - min_x)) * 2;
            let map_re   = *map.add(map_off);
            let map_im   = *map.add(map_off + 1);

            if map_re == 0.0 && map_im == 0.0 { continue; }

            let src_x = x as f32 + map_re - roi_in_x as f32;
            let src_y = y as f32 + map_im - roi_in_y as f32;

            if ch == 1 {
                let v = interp::compute_sample_1c(inp, src_x, src_y,
                    roi_in_w, roi_in_h, roi_in_w, interp_type);
                outp[out_row + x - roi_out_x as usize] = v.clamp(0.0, 1.0);
            } else {
                let px = interp::compute_pixel4c(inp, src_x, src_y,
                    roi_in_w, roi_in_h, ch_width, interp_type);
                let bi = (out_row + x - roi_out_x as usize) * ch as usize;
                for c in 0..ch as usize { outp[bi + c] = px[c]; }
            }
        }
    }
}

/// Invert the distortion map: imap[nx+ny*w] = -map[x+y*w].
/// Matches create_global_distortion_map DT_OMP_FOR at liquify.c:1129.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_invert_map(
    map:   *const f32,   // float complex
    imap:  *mut f32,
    width: i32,
    height: i32,
) {
    let w  = width as usize;
    let h  = height as usize;
    let m  = std::slice::from_raw_parts(map,  w * h * 2);
    let im = std::slice::from_raw_parts_mut(imap, w * h * 2);
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 2;
            let re = m[src]; let i = m[src + 1];
            let nx = x as i32 + re as i32;
            let ny = y as i32 + i  as i32;
            if nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1 {
                let dst = (ny as usize * w + nx as usize) * 2;
                im[dst] = -re; im[dst + 1] = -i;
            }
        }
    }
}

/// Fill zero gaps in inverted map by propagating nearest non-zero value per row.
/// Matches create_global_distortion_map DT_OMP_FOR at liquify.c:1153.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_fill_gaps(
    imap:  *mut f32,
    width: i32,
    height: i32,
) {
    let w  = width as usize;
    let h  = height as usize;
    let m  = std::slice::from_raw_parts_mut(imap, w * h * 2);
    let half = w / 2 + 1;
    for y in 0..h {
        let row = y * w;
        let mut last_l = [0.0f32; 2];
        let mut last_r = [0.0f32; 2];
        for x in 0..half {
            let cl = (row + x) * 2;
            let cr = (row + w - x) * 2;
            if x != 0 {
                if m[cl] == 0.0 && m[cl+1] == 0.0 {
                    m[cl] = last_l[0]; m[cl+1] = last_l[1];
                }
                if m[cr] == 0.0 && m[cr+1] == 0.0 {
                    m[cr] = last_r[0]; m[cr+1] = last_r[1];
                }
            }
            last_l = [m[cl], m[cl+1]];
            last_r = [m[cr], m[cr+1]];
        }
    }
}

/// Compute bounding box of scaled coordinate pairs.
/// Returns (xmin, xmax, ymin, ymax) via output pointers.
/// Matches distort_transform/backtransform DT_OMP_FOR at liquify.c:1258.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_bounding_box(
    points: *const f32,
    n:      usize,
    scale:  f32,
    xmin_out: *mut f32, xmax_out: *mut f32,
    ymin_out: *mut f32, ymax_out: *mut f32,
) {
    let pts = std::slice::from_raw_parts(points, n * 2);
    let (mut xmin, mut xmax) = (f32::MAX, f32::MIN);
    let (mut ymin, mut ymax) = (f32::MAX, f32::MIN);
    for i in (0..n * 2).step_by(2) {
        let x = pts[i]   * scale;
        let y = pts[i+1] * scale;
        xmin = xmin.min(x); xmax = xmax.max(x);
        ymin = ymin.min(y); ymax = ymax.max(y);
    }
    *xmin_out = xmin; *xmax_out = xmax;
    *ymin_out = ymin; *ymax_out = ymax;
}

/// Apply distortion map to coordinate pairs.
/// Matches distort_transform/backtransform DT_OMP_FOR at liquify.c:1297.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_apply_distortion(
    points:    *mut f32,
    n:         usize,
    scale:     f32,
    map:       *const f32,  // float complex
    extent_x:  i32, extent_y: i32, extent_w: i32,
    map_size:  i32,
) {
    let pts = std::slice::from_raw_parts_mut(points, n * 2);
    let ms  = map_size as usize;
    let x_last = extent_x + extent_w;
    for i in 0..n {
        let x = pts[i*2]   * scale;
        let y = pts[i*2+1] * scale;
        let map_off = ((x - 0.5) as i32 - extent_x)
            + (((y - 0.5) as i32 - extent_y) * extent_w);
        if x >= extent_x as f32 && x < x_last as f32
            && y >= extent_y as f32 && map_off >= 0 && (map_off as usize) < ms
        {
            let off = (map_off as usize) * 2;
            let re  = *map.add(off);
            let im  = *map.add(off + 1);
            pts[i*2]   += re / scale;
            pts[i*2+1] += im / scale;
        }
    }
}
