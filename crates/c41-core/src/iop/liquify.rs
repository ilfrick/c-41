//! Liquify IOP helpers -- 6 DT_OMP_FOR loops, all using float complex pairs.
//! C's `float complex` is stored as two consecutive f32 (real, imag).

use super::{ClBuffer, IopProcess};
use crate::{interp, params::IopParams, roi::RoiIn, Result};

#[inline]
fn f32_slice_len_is_valid(len: usize) -> bool {
    len <= isize::MAX as usize / std::mem::size_of::<f32>()
}

pub struct Liquify;
impl IopProcess for Liquify {
    fn name(&self) -> &'static str {
        "liquify"
    }
    fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("liquify: use C FFI path".into()))
    }
    fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("liquify: no OpenCL path".into()))
    }
}

// Complex helpers: treat float complex as [f32; 2] = [real, imag]
#[inline(always)]
fn re(p: &[f32; 2]) -> f32 {
    p[0]
}
#[inline(always)]
fn im(p: &[f32; 2]) -> f32 {
    p[1]
}

/// Apply a round displacement stamp to the global distortion map.
///
/// `center` points to the (x=0,y=0) position in the map.
/// For warp_type == 0 (LINEAR): writes -(strength_re, strength_im)*LUT[d] to 4 quadrants.
/// For warp_type == 1 (RADIAL): writes ±abs_strength*LUT[d]/r*(x, ±y) to quadrants.
/// Matches apply_round_stamp DT_OMP_FOR at liquify.c:956.
///
/// # Safety
/// `center` must point inside one allocated interleaved-complex map of row
/// width `global_width`, with the complete `iradius` square around it in bounds;
/// `lookup_table` must be valid for `table_size` floats and must not overlap that
/// mutable map. Dimensions must fit `isize` offset arithmetic. For radial warps,
/// `iradius` must be nonzero.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_apply_stamp(
    center: *mut f32, // float complex* (stride 2) at map center
    global_width: usize,
    iradius: usize,
    lookup_table: *const f32,
    table_size: usize,
    oversample: usize,
    warp_type: i32,    // 0=linear, 1=radial
    strength_re: f32,  // real part of -(strength) for linear
    strength_im: f32,  // imag part
    abs_strength: f32, // for radial
) {
    if global_width == 0
        || table_size == 0
        || !f32_slice_len_is_valid(table_size)
        || oversample == 0
        || (warp_type != 0 && iradius == 0)
        || iradius.checked_mul(iradius).is_none()
    {
        return;
    }
    let (Ok(radius), Ok(width)) = (isize::try_from(iradius), isize::try_from(global_width)) else {
        return;
    };
    if radius
        .checked_mul(width)
        .and_then(|v| v.checked_add(radius))
        .is_none_or(|v| v > (isize::MAX as usize / (2 * std::mem::size_of::<f32>())) as isize)
    {
        return;
    }
    let stride = 2usize;
    let lt = std::slice::from_raw_parts(lookup_table, table_size);

    for y in 0..=iradius {
        let y2 = (y * y) as f32;
        let mut x = 0usize;
        loop {
            if x > iradius {
                break;
            }
            let dist = ((x * x) as f32 + y2).sqrt();
            let idist = (dist * oversample as f32).round() as usize;
            if idist >= table_size {
                break;
            }

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
            let q3 = off(y as isize, -(x as isize));
            let q4 = off(y as isize, x as isize);

            let lv = lt[idist];
            if warp_type == 0 {
                let wr = -strength_re * lv;
                let wi = -strength_im * lv;
                let add = |p: *mut f32| {
                    *p += wr;
                    *p.add(1) += wi;
                };
                add(q1);
                if x != 0 {
                    add(q2);
                }
                if x != 0 && y != 0 {
                    add(q3);
                }
                if y != 0 {
                    add(q4);
                }
            } else {
                let av = abs_strength * lv / iradius as f32;
                // q1 -= av*(x - y*i)
                *q1 -= av * x as f32;
                *q1.add(1) += av * y as f32;
                // q2 += av*(x + y*i)
                if x != 0 {
                    *q2 += av * x as f32;
                    *q2.add(1) -= av * y as f32;
                }
                // q3 += av*(x - y*i)
                if x != 0 && y != 0 {
                    *q3 += av * x as f32;
                    *q3.add(1) -= av * y as f32;
                }
                // q4 -= av*(x + y*i)
                if y != 0 {
                    *q4 -= av * x as f32;
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
///
/// # Safety
/// `in_buf`, `out_buf`, and `map` must point to valid, mutually non-overlapping
/// buffers sized for the declared ROIs and extent; channel count must be 1 or 4.
/// ROI origins may be negative; dimensions must be positive and arithmetic-sized.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_apply_map(
    in_buf: *const f32,
    out_buf: *mut f32,
    roi_in_x: i32,
    roi_in_y: i32,
    roi_in_w: i32,
    roi_in_h: i32,
    roi_out_x: i32,
    roi_out_y: i32,
    roi_out_w: i32,
    roi_out_h: i32,
    extent_x: i32,
    extent_y: i32,
    extent_w: i32,
    extent_h: i32,
    map: *const f32, // float complex array (stride 2)
    ch: i32,
    interp_type: u32,
) {
    if roi_in_w <= 0
        || roi_in_h <= 0
        || roi_out_w <= 0
        || roi_out_h <= 0
        || extent_w <= 0
        || extent_h <= 0
        || (ch != 1 && ch != 4)
    {
        return;
    }
    let ch = ch as usize;
    let roi_in_w_usize = roi_in_w as usize;
    let roi_in_h_usize = roi_in_h as usize;
    let roi_out_w_usize = roi_out_w as usize;
    let roi_out_h_usize = roi_out_h as usize;
    let extent_w_usize = extent_w as usize;
    let extent_h_usize = extent_h as usize;
    let Some(in_n) = roi_in_w_usize
        .checked_mul(roi_in_h_usize)
        .and_then(|n| n.checked_mul(ch))
    else {
        return;
    };
    let Some(out_n) = roi_out_w_usize
        .checked_mul(roi_out_h_usize)
        .and_then(|n| n.checked_mul(ch))
    else {
        return;
    };
    let Some(map_n) = extent_w_usize
        .checked_mul(extent_h_usize)
        .and_then(|n| n.checked_mul(2))
    else {
        return;
    };
    if !f32_slice_len_is_valid(in_n)
        || !f32_slice_len_is_valid(out_n)
        || !f32_slice_len_is_valid(map_n)
    {
        return;
    }
    let Some(ch_width) = roi_in_w.checked_mul(ch as i32) else {
        return;
    };
    let inp = std::slice::from_raw_parts(in_buf, in_n);
    let outp = std::slice::from_raw_parts_mut(out_buf, out_n);
    let _map_len_guard = map_n; // validates map index arithmetic before pointer access

    // Signed intermediates preserve valid negative-origin ROIs and avoid i32
    // overflow in the endpoint additions.
    let roi_out_x = roi_out_x as i64;
    let roi_out_y = roi_out_y as i64;
    let extent_x = extent_x as i64;
    let extent_y = extent_y as i64;
    let roi_out_right = roi_out_x + roi_out_w as i64;
    let roi_out_bottom = roi_out_y + roi_out_h as i64;
    let extent_right = extent_x + extent_w as i64;
    let extent_bottom = extent_y + extent_h as i64;
    let min_y = roi_out_y.max(extent_y);
    let max_y = roi_out_bottom.min(extent_bottom);
    let min_x = roi_out_x.max(extent_x);
    let max_x = roi_out_right.min(extent_right);
    if min_x >= max_x || min_y >= max_y {
        return;
    }

    for y in min_y..max_y {
        let row_base = (y - extent_y) as usize * extent_w_usize + (min_x - extent_x) as usize;
        let out_row = (y - roi_out_y) as usize * roi_out_w_usize;
        for x in min_x..max_x {
            let map_off = (row_base + (x - min_x) as usize) * 2;
            let map_re = *map.add(map_off);
            let map_im = *map.add(map_off + 1);
            if map_re == 0.0 && map_im == 0.0 {
                continue;
            }

            let src_x = x as f32 + map_re - roi_in_x as f32;
            let src_y = y as f32 + map_im - roi_in_y as f32;
            let out_pixel = out_row + (x - roi_out_x) as usize;
            if ch == 1 {
                let v = interp::compute_sample_1c(
                    inp,
                    src_x,
                    src_y,
                    roi_in_w,
                    roi_in_h,
                    roi_in_w,
                    interp_type,
                );
                outp[out_pixel] = v.clamp(0.0, 1.0);
            } else {
                let px = interp::compute_pixel4c(
                    inp,
                    src_x,
                    src_y,
                    roi_in_w,
                    roi_in_h,
                    ch_width,
                    interp_type,
                );
                let out_i = out_pixel * ch;
                for c in 0..ch {
                    outp[out_i + c] = px[c];
                }
            }
        }
    }
}

/// Invert the distortion map: imap[nx+ny*w] = -map[x+y*w].
/// Matches create_global_distortion_map DT_OMP_FOR at liquify.c:1129.
///
/// # Safety
/// For positive dimensions, `map` and `imap` must point to non-overlapping
/// buffers of at least `width * height * 2` floats. Nonpositive dimensions are
/// a no-op and may use null pointers.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_invert_map(
    map: *const f32, // float complex
    imap: *mut f32,
    width: i32,
    height: i32,
) {
    if width <= 0 || height <= 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let Some(len) = w.checked_mul(h).and_then(|n| n.checked_mul(2)) else {
        return;
    };
    if !f32_slice_len_is_valid(len) {
        return;
    }
    let m = std::slice::from_raw_parts(map, len);
    let im = std::slice::from_raw_parts_mut(imap, len);
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 2;
            let re = m[src];
            let i = m[src + 1];
            let Some(nx) = (x as i64).checked_add(re as i64) else {
                continue;
            };
            let Some(ny) = (y as i64).checked_add(i as i64) else {
                continue;
            };
            if nx > 0 && nx < width as i64 - 1 && ny > 0 && ny < height as i64 - 1 {
                let dst = (ny as usize * w + nx as usize) * 2;
                im[dst] = -re;
                im[dst + 1] = -i;
            }
        }
    }
}

/// Fill zero gaps in inverted map by propagating nearest non-zero value per row.
/// Matches create_global_distortion_map DT_OMP_FOR at liquify.c:1153.
///
/// # Safety
/// For positive dimensions, `imap` must point to at least `width * height * 2`
/// writable floats. Nonpositive dimensions are a no-op and may use a null
/// pointer.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_fill_gaps(imap: *mut f32, width: i32, height: i32) {
    if width <= 0 || height <= 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let Some(len) = w.checked_mul(h).and_then(|n| n.checked_mul(2)) else {
        return;
    };
    if !f32_slice_len_is_valid(len) {
        return;
    }
    let m = std::slice::from_raw_parts_mut(imap, len);
    let half = w / 2 + 1;
    for y in 0..h {
        let row = y * w;
        let mut last_l = [0.0f32; 2];
        let mut last_r = [0.0f32; 2];
        for x in 0..half {
            let cl = (row + x) * 2;
            let cr = (row + w - x - 1) * 2;
            if x != 0 {
                if m[cl] == 0.0 && m[cl + 1] == 0.0 {
                    m[cl] = last_l[0];
                    m[cl + 1] = last_l[1];
                }
                if m[cr] == 0.0 && m[cr + 1] == 0.0 {
                    m[cr] = last_r[0];
                    m[cr + 1] = last_r[1];
                }
            }
            last_l = [m[cl], m[cl + 1]];
            last_r = [m[cr], m[cr + 1]];
        }
    }
}

/// Compute bounding box of scaled coordinate pairs.
/// Returns (xmin, xmax, ymin, ymax) via output pointers.
/// Matches distort_transform/backtransform DT_OMP_FOR at liquify.c:1258.
///
/// # Safety
/// `points` must reference `2 * n` readable floats and must not overlap the
/// mutually non-overlapping output pointers, each valid for one writable float.
/// `scale` must be
/// finite. For `n == 0` or non-finite scale, `points` may be null and outputs
/// receive empty-box sentinels (`f32::MAX` minima, `f32::MIN` maxima).
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_bounding_box(
    points: *const f32,
    n: usize,
    scale: f32,
    xmin_out: *mut f32,
    xmax_out: *mut f32,
    ymin_out: *mut f32,
    ymax_out: *mut f32,
) {
    let (mut xmin, mut xmax) = (f32::MAX, f32::MIN);
    let (mut ymin, mut ymax) = (f32::MAX, f32::MIN);
    if !scale.is_finite() {
        *xmin_out = xmin;
        *xmax_out = xmax;
        *ymin_out = ymin;
        *ymax_out = ymax;
        return;
    }
    if n == 0 {
        *xmin_out = xmin;
        *xmax_out = xmax;
        *ymin_out = ymin;
        *ymax_out = ymax;
        return;
    }
    let Some(len) = n.checked_mul(2) else {
        *xmin_out = xmin;
        *xmax_out = xmax;
        *ymin_out = ymin;
        *ymax_out = ymax;
        return;
    };
    if !f32_slice_len_is_valid(len) {
        *xmin_out = xmin;
        *xmax_out = xmax;
        *ymin_out = ymin;
        *ymax_out = ymax;
        return;
    }
    let pts = std::slice::from_raw_parts(points, len);
    for i in (0..len).step_by(2) {
        let x = pts[i] * scale;
        let y = pts[i + 1] * scale;
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    *xmin_out = xmin;
    *xmax_out = xmax;
    *ymin_out = ymin;
    *ymax_out = ymax;
}

/// Apply distortion map to coordinate pairs.
/// Matches distort_transform/backtransform DT_OMP_FOR at liquify.c:1297.
///
/// # Safety
/// `points` must reference `2 * n` readable/writable floats and `map` must
/// reference `2 * map_size` readable floats; the buffers must not overlap.
/// `scale` must be finite and nonzero. `extent_w` and `map_size` must be positive.
#[no_mangle]
pub unsafe extern "C" fn darkroom_liquify_apply_distortion(
    points: *mut f32,
    n: usize,
    scale: f32,
    map: *const f32, // float complex
    extent_x: i32,
    extent_y: i32,
    extent_w: i32,
    map_size: i32,
) {
    if n == 0 || map_size <= 0 || extent_w <= 0 || !scale.is_finite() || scale == 0.0 {
        return;
    }
    let Some(point_len) = n.checked_mul(2) else {
        return;
    };
    let Some(map_len) = (map_size as usize).checked_mul(2) else {
        return;
    };
    if !f32_slice_len_is_valid(point_len) || !f32_slice_len_is_valid(map_len) {
        return;
    }
    let pts = std::slice::from_raw_parts_mut(points, point_len);
    let ms = map_size as i64;
    let extent_x = extent_x as i64;
    let extent_y = extent_y as i64;
    let extent_w = extent_w as i64;
    let x_last = extent_x + extent_w;
    for i in 0..n {
        let x = pts[i * 2] * scale;
        let y = pts[i * 2 + 1] * scale;
        if !x.is_finite()
            || !y.is_finite()
            || x < extent_x as f32
            || x >= x_last as f32
            || y < extent_y as f32
        {
            continue;
        }
        let cell_x = (x - 0.5) as i64 - extent_x;
        let cell_y = (y - 0.5) as i64 - extent_y;
        let Some(map_off) = cell_y
            .checked_mul(extent_w)
            .and_then(|v| v.checked_add(cell_x))
        else {
            continue;
        };
        if map_off < 0 || map_off >= ms {
            continue;
        }
        let off = map_off as usize * 2;
        let re = *map.add(off);
        let im = *map.add(off + 1);
        pts[i * 2] += re / scale;
        pts[i * 2 + 1] += im / scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(map: &[f32], width: usize, x: usize, y: usize) -> [f32; 2] {
        let i = (y * width + x) * 2;
        [map[i], map[i + 1]]
    }

    #[test]
    fn ffi_guards_reject_unrepresentable_slice_lengths() {
        let max_f32_len = isize::MAX as usize / std::mem::size_of::<f32>();
        assert!(f32_slice_len_is_valid(max_f32_len));
        assert!(!f32_slice_len_is_valid(max_f32_len + 1));

        let lut = [1.0f32];
        unsafe {
            darkroom_liquify_apply_stamp(
                std::ptr::null_mut(),
                10_000_000_000,
                1_000_000_000,
                lut.as_ptr(),
                lut.len(),
                1,
                0,
                0.0,
                0.0,
                0.0,
            );
            darkroom_liquify_apply_stamp(
                std::ptr::null_mut(),
                1,
                1,
                std::ptr::null(),
                usize::MAX,
                1,
                0,
                0.0,
                0.0,
                0.0,
            );
            darkroom_liquify_invert_map(std::ptr::null(), std::ptr::null_mut(), i32::MAX, i32::MAX);
            darkroom_liquify_fill_gaps(std::ptr::null_mut(), i32::MAX, i32::MAX);
            darkroom_liquify_apply_map(
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                0,
                i32::MAX,
                i32::MAX,
                0,
                0,
                i32::MAX,
                i32::MAX,
                0,
                0,
                i32::MAX,
                i32::MAX,
                std::ptr::null(),
                4,
                0,
            );
            darkroom_liquify_apply_distortion(
                std::ptr::null_mut(),
                usize::MAX,
                1.0,
                std::ptr::null(),
                0,
                0,
                1,
                1,
            );
        }
    }

    #[test]
    fn apply_stamp_linear_writes_symmetric_quadrants() {
        let width = 3usize;
        let mut map = vec![0.0f32; width * width * 2];
        let center = unsafe { map.as_mut_ptr().add((width + 1) * 2) };
        let lut = [1.0f32, 0.5];
        unsafe {
            darkroom_liquify_apply_stamp(
                center,
                width,
                1,
                lut.as_ptr(),
                lut.len(),
                1,
                0,
                2.0,
                3.0,
                0.0,
            );
        }
        assert_eq!(pixel(&map, width, 1, 1), [-2.0, -3.0]);
        for (x, y) in [
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2),
            (0, 0),
            (2, 0),
            (0, 2),
            (2, 2),
        ] {
            assert_eq!(pixel(&map, width, x, y), [-1.0, -1.5]);
        }
    }

    #[test]
    fn apply_stamp_radial_uses_opposite_real_directions() {
        let width = 3usize;
        let mut map = vec![0.0f32; width * width * 2];
        let center = unsafe { map.as_mut_ptr().add((width + 1) * 2) };
        let lut = [1.0f32, 1.0];
        unsafe {
            darkroom_liquify_apply_stamp(
                center,
                width,
                1,
                lut.as_ptr(),
                lut.len(),
                1,
                1,
                0.0,
                0.0,
                2.0,
            );
        }
        assert_eq!(pixel(&map, width, 2, 1), [-2.0, 0.0]);
        assert_eq!(pixel(&map, width, 0, 1), [2.0, 0.0]);
        assert_eq!(pixel(&map, width, 1, 0), [0.0, 2.0]);
        assert_eq!(pixel(&map, width, 1, 2), [0.0, 2.0]);
    }

    #[test]
    fn apply_map_warps_only_nonzero_map_pixels() {
        let input: Vec<f32> = (0..9).map(|i| i as f32 / 10.0).collect();
        assert_ne!(
            input[4], 0.5,
            "the shifted sample must differ from the source center"
        );
        let mut output = vec![-1.0f32; 9];
        let mut map = vec![0.0f32; 9 * 2];
        map[(1 * 3 + 1) * 2] = 1.0;
        unsafe {
            darkroom_liquify_apply_map(
                input.as_ptr(),
                output.as_mut_ptr(),
                0,
                0,
                3,
                3,
                0,
                0,
                3,
                3,
                0,
                0,
                3,
                3,
                map.as_ptr(),
                1,
                0,
            );
        }
        assert_eq!(output[4], 0.5);
        for i in [0, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(
                output[i], -1.0,
                "zero displacement must leave pixel {i} untouched"
            );
        }
    }

    #[test]
    fn apply_map_handles_negative_roi_origins() {
        let input = [0.1f32, 0.8, 0.2, 0.9];
        let mut output = [-1.0f32; 4];
        let mut map = [0.0f32; 8];
        map[0] = 1.0;
        unsafe {
            darkroom_liquify_apply_map(
                input.as_ptr(),
                output.as_mut_ptr(),
                -1,
                -1,
                2,
                2,
                -1,
                -1,
                2,
                2,
                -1,
                -1,
                2,
                2,
                map.as_ptr(),
                1,
                0,
            );
        }
        assert_eq!(output[0], 0.8);
        assert_eq!(&output[1..], &[-1.0, -1.0, -1.0]);
    }

    #[test]
    fn invert_map_moves_and_negates_only_interior_displacements() {
        let width = 4i32;
        let height = 3i32;
        let mut map = vec![0.0f32; width as usize * height as usize * 2];
        let mut inverse = vec![0.0f32; map.len()];
        let source = (1 * width as usize + 1) * 2;
        map[source] = 1.0;
        map[source + 1] = -0.5;
        let target = (1 * width as usize + 2) * 2;
        map[target + 1] = 1.0; // points out of the interior, so it cannot overwrite source's inverse
        unsafe {
            darkroom_liquify_invert_map(map.as_ptr(), inverse.as_mut_ptr(), width, height);
        }
        assert_eq!(&inverse[target..target + 2], &[-1.0, 0.5]);
        assert_eq!(inverse.iter().filter(|v| **v != 0.0).count(), 2);
    }

    #[test]
    fn fill_gaps_propagates_edges_independently_for_odd_and_even_rows() {
        let width = 5usize;
        let height = 2usize;
        let mut map = vec![0.0f32; width * height * 2];
        map[0..2].copy_from_slice(&[1.0, 2.0]);
        map[8..10].copy_from_slice(&[9.0, 10.0]);
        map[10..12].copy_from_slice(&[30.0, 40.0]);
        map[18..20].copy_from_slice(&[70.0, 80.0]);
        unsafe {
            darkroom_liquify_fill_gaps(map.as_mut_ptr(), width as i32, height as i32);
        }
        assert_eq!(&map[0..2], &[1.0, 2.0]);
        assert_eq!(&map[2..4], &[1.0, 2.0]);
        assert_eq!(
            &map[4..6],
            &[1.0, 2.0],
            "odd-width center follows the left edge"
        );
        assert_eq!(&map[6..8], &[9.0, 10.0]);
        assert_eq!(&map[8..10], &[9.0, 10.0]);
        assert_eq!(&map[12..14], &[30.0, 40.0]);
        assert_eq!(&map[14..16], &[30.0, 40.0]);
        assert_eq!(&map[16..18], &[70.0, 80.0]);
        assert_eq!(&map[18..20], &[70.0, 80.0]);

        let mut even = [1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0];
        unsafe {
            darkroom_liquify_fill_gaps(even.as_mut_ptr(), 4, 1);
        }
        assert_eq!(&even[2..4], &[1.0, 2.0]);
        assert_eq!(&even[4..6], &[9.0, 10.0]);

        unsafe {
            darkroom_liquify_fill_gaps(std::ptr::null_mut(), 0, 1);
        }
        unsafe {
            darkroom_liquify_fill_gaps(std::ptr::null_mut(), 1, 0);
        }
        unsafe {
            darkroom_liquify_invert_map(std::ptr::null(), std::ptr::null_mut(), 0, 1);
        }
    }

    #[test]
    fn bounding_box_scales_all_coordinate_pairs() {
        let points = [1.0f32, 2.0, 3.0, -4.0, -2.0, 5.0];
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (0.0, 0.0, 0.0, 0.0);
        unsafe {
            darkroom_liquify_bounding_box(
                points.as_ptr(),
                3,
                2.0,
                &mut xmin,
                &mut xmax,
                &mut ymin,
                &mut ymax,
            );
        }
        assert_eq!((xmin, xmax, ymin, ymax), (-4.0, 6.0, -8.0, 10.0));

        unsafe {
            darkroom_liquify_bounding_box(
                std::ptr::null(),
                0,
                1.0,
                &mut xmin,
                &mut xmax,
                &mut ymin,
                &mut ymax,
            );
        }
        assert_eq!(
            (xmin, xmax, ymin, ymax),
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN)
        );
        unsafe {
            darkroom_liquify_bounding_box(
                std::ptr::null(),
                usize::MAX,
                1.0,
                &mut xmin,
                &mut xmax,
                &mut ymin,
                &mut ymax,
            );
        }
        assert_eq!(
            (xmin, xmax, ymin, ymax),
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN)
        );
        unsafe {
            darkroom_liquify_bounding_box(
                std::ptr::null(),
                1,
                f32::NAN,
                &mut xmin,
                &mut xmax,
                &mut ymin,
                &mut ymax,
            );
        }
        assert_eq!(
            (xmin, xmax, ymin, ymax),
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN)
        );
    }

    #[test]
    fn apply_distortion_handles_negative_extent_origins() {
        let mut points = [-0.5f32, -0.5];
        let mut map = vec![0.0f32; 3 * 3 * 2];
        map[0..2].copy_from_slice(&[0.5, -0.25]);
        unsafe {
            darkroom_liquify_apply_distortion(
                points.as_mut_ptr(),
                1,
                1.0,
                map.as_ptr(),
                -1,
                -1,
                3,
                9,
            );
        }
        assert_eq!(points, [0.0, -0.75]);
    }

    #[test]
    fn apply_distortion_samples_scaled_coordinates_and_respects_extent() {
        let mut points = [1.5f32, 1.5, 3.0, 1.0];
        let mut map = vec![0.0f32; 3 * 3 * 2];
        map[(1 * 3 + 1) * 2..][..2].copy_from_slice(&[2.0, -1.0]);
        unsafe {
            darkroom_liquify_apply_distortion(
                points.as_mut_ptr(),
                2,
                1.0,
                map.as_ptr(),
                0,
                0,
                3,
                map.len() as i32,
            );
        }
        assert_eq!(&points[0..2], &[3.5, 0.5]);
        assert_eq!(
            &points[2..4],
            &[3.0, 1.0],
            "outside-map point must not move"
        );
    }
}
