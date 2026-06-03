//! Retouch IOP helpers -- portable OMP loops from retouch.c.
//! The four RGB<->Lab conversion loops remain in C (blocked on ICC math).

use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Retouch;
impl IopProcess for Retouch {
    fn name(&self) -> &'static str { "retouch" }
    fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("retouch: use C FFI path".into()))
    }
    fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("retouch: no OpenCL path".into()))
    }
}

/// Copy y_to rows from `in` (with offset xoffs,yoffs) into `out`.
/// rowsize is in bytes. Matches rt_copy_in_to_out DT_OMP_FOR at retouch.c:3250.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_copy_rows(
    in_buf:    *const f32,
    out_buf:   *mut f32,
    y_to:      i32,
    xoffs:     i32,
    yoffs:     i32,
    in_width:  i32,
    out_width: i32,
    ch:        i32,
    rowsize:   usize,
) {
    let bytes_in  = ((yoffs + y_to) * in_width * ch) as usize * 4;
    let bytes_out = (y_to * out_width * ch) as usize * 4;
    let inp  = std::slice::from_raw_parts(in_buf  as *const u8, bytes_in);
    let outp = std::slice::from_raw_parts_mut(out_buf as *mut u8, bytes_out);
    for y in 0..y_to as usize {
        let si = ((y as i32 + yoffs) * in_width + xoffs) as usize * ch as usize * 4;
        let di = y * out_width as usize * ch as usize * 4;
        outp[di..di + rowsize].copy_from_slice(&inp[si..si + rowsize]);
    }
}

/// Nearest-neighbour mask scaling into `mask_tmp`.
/// Matches rt_build_scaled_mask DT_OMP_FOR at retouch.c:3300.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_build_mask(
    mask:       *const f32,
    mask_tmp:   *mut f32,
    roi_mask_x: i32, roi_mask_y: i32,
    roi_mask_w: i32, roi_mask_h: i32,
    roi_ms_x:   i32, roi_ms_y:   i32,
    roi_ms_w:   i32, roi_ms_h:   i32,
    x_to:       i32, y_to:       i32,
    scale:      f32,
) {
    let m  = std::slice::from_raw_parts(mask,     (roi_mask_w * roi_mask_h) as usize);
    let ms = std::slice::from_raw_parts_mut(mask_tmp, (roi_ms_w * roi_ms_h) as usize);
    for yy in roi_ms_y..y_to {
        let mi = (yy as f32 / scale) as i32 - roi_mask_y;
        if mi < 0 || mi >= roi_mask_h { continue; }
        let ms_row = (yy - roi_ms_y) * roi_ms_w;
        for xx in roi_ms_x..x_to {
            let mx = (xx as f32 / scale) as i32 - roi_mask_x;
            if mx < 0 || mx >= roi_mask_w { continue; }
            ms[(ms_row + xx - roi_ms_x) as usize] = m[(mi * roi_mask_w + mx) as usize];
        }
    }
}

/// Masked alpha blend: dest = dest*(1-f) + src*f  where f = mask*opacity.
/// dest_npixels is roi_dest->width * roi_dest->height.
/// Matches rt_copy_image_masked DT_OMP_FOR at retouch.c:3333.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_copy_masked(
    src:          *const f32,
    dest:         *mut f32,
    dest_roi_x:   i32, dest_roi_y: i32, dest_roi_w: i32,
    dest_npixels: usize,
    mask:         *const f32,
    mask_roi_x:   i32, mask_roi_y: i32,
    mask_w:       i32, mask_h:     i32,
    opacity:      f32,
) {
    let s = std::slice::from_raw_parts(src,  (mask_w * mask_h * 4) as usize);
    let d = std::slice::from_raw_parts_mut(dest, dest_npixels * 4);
    let m = std::slice::from_raw_parts(mask, (mask_w * mask_h) as usize);
    for yy in 0..mask_h as usize {
        let mi = yy * mask_w as usize;
        let si = mi * 4;
        let di = ((yy as i32 + mask_roi_y - dest_roi_y) * dest_roi_w
                  + (mask_roi_x - dest_roi_x)) as usize * 4;
        for xx in 0..mask_w as usize {
            let f  = m[mi + xx] * opacity;
            let f1 = 1.0 - f;
            for c in 0..4 {
                d[di + xx*4 + c] = d[di + xx*4 + c] * f1 + s[si + xx*4 + c] * f;
            }
        }
    }
}

/// Update dest alpha: d[3] = max(d[3], mask*opacity).
/// img_npixels = roi_img->width * roi_img->height.
/// Matches rt_copy_mask_to_alpha DT_OMP_FOR at retouch.c:3366.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_copy_mask_to_alpha(
    img:           *mut f32,
    roi_img_x:     i32, roi_img_y: i32, roi_img_w: i32,
    img_npixels:   usize,
    ch:            i32,
    mask:          *const f32,
    mask_roi_x:    i32, mask_roi_y: i32,
    mask_w:        i32, mask_h:     i32,
    opacity:       f32,
) {
    let d = std::slice::from_raw_parts_mut(img, img_npixels * ch as usize);
    let m = std::slice::from_raw_parts(mask, (mask_w * mask_h) as usize);
    for yy in 0..mask_h as usize {
        let mi = yy * mask_w as usize;
        let di = ((yy as i32 + mask_roi_y - roi_img_y) * roi_img_w
                  + (mask_roi_x - roi_img_x)) as usize * ch as usize;
        for xx in 0..mask_w as usize {
            let f = m[mi + xx] * opacity;
            let alpha = &mut d[di + xx * ch as usize + 3];
            if f > *alpha { *alpha = f; }
        }
    }
}

/// Fill masked region: dest = dest*(1-f) + fill_color*f where f = mask*opacity.
/// dest_npixels = roi_in->width * roi_in->height.
/// Matches _retouch_fill DT_OMP_FOR at retouch.c:3392.
#[no_mangle]
pub unsafe extern "C" fn darkroom_retouch_fill(
    dest:          *mut f32,
    roi_in_x:      i32, roi_in_y: i32, roi_in_w: i32,
    dest_npixels:  usize,
    mask:          *const f32,
    mask_roi_x:    i32, mask_roi_y: i32,
    mask_w:        i32, mask_h:     i32,
    opacity:       f32,
    fill_color:    *const f32,   // 4 floats
) {
    let d    = std::slice::from_raw_parts_mut(dest, dest_npixels * 4);
    let m    = std::slice::from_raw_parts(mask, (mask_w * mask_h) as usize);
    let fill = std::slice::from_raw_parts(fill_color, 4);
    for yy in 0..mask_h as usize {
        let mi = yy * mask_w as usize;
        let di = ((yy as i32 + mask_roi_y - roi_in_y) * roi_in_w
                  + (mask_roi_x - roi_in_x)) as usize * 4;
        for xx in 0..mask_w as usize {
            let f  = m[mi + xx] * opacity;
            let f1 = 1.0 - f;
            for c in 0..4 {
                d[di + xx*4 + c] = d[di + xx*4 + c] * f1 + fill[c] * f;
            }
        }
    }
}
