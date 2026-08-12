use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Rawprepare;

impl IopProcess for Rawprepare {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "rawprepare" }
}

/// Compute the Bayer-quadrant index for the given absolute (row, col) in the
/// uncropped sensor coordinate space.
///
/// Mirrors the C _BL() helper in src/iop/rawprepare.c:
///   id = ((row & 1) << 1) | (col & 1)   // values 0..3
///
/// The caller is expected to have already added `roi_out->y + d->top` to
/// `row` and `roi_out->x + d->left` to `col` so we work in absolute sensor
/// coordinates.
#[inline(always)]
fn bayer_quadrant(row: i32, col: i32) -> usize {
    let r = (row.rem_euclid(2)) as usize;
    let c = (col.rem_euclid(2)) as usize;
    (r << 1) | c
}

/// Linearise a uint16 raw Bayer/X-Trans frame: subtract per-quadrant black
/// level and divide by per-quadrant scale.
///
/// Matches the `TYPE_UINT16` branch of process() in src/iop/rawprepare.c.
/// `in_buf` is a single-plane uint16 buffer of size `in_width * in_height`;
/// `out_buf` is float of size `out_width * out_height`.
/// `csx, csy` are the input crop offsets; `x0, y0` are the absolute sensor
/// coordinates of the output's (0,0) pixel (i.e. roi_out.x + d->left, etc.).
/// `sub` and `div` are 4-float arrays indexed by `bayer_quadrant`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawprepare_mosaic_u16(
    in_buf: *const u16,
    out_buf: *mut f32,
    out_width: usize,
    out_height: usize,
    in_width: usize,
    csx: i32,
    csy: i32,
    x0: i32,
    y0: i32,
    sub: *const f32,
    div: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, in_width * (out_height + csy.max(0) as usize));
    let output = std::slice::from_raw_parts_mut(out_buf, out_width * out_height);
    let sub = std::slice::from_raw_parts(sub, 4);
    let div = std::slice::from_raw_parts(div, 4);

    for j in 0..out_height {
        for i in 0..out_width {
            let pin = ((j as i32 + csy) as usize) * in_width + (i as i32 + csx) as usize;
            let pout = j * out_width + i;
            let id = bayer_quadrant(j as i32 + y0, i as i32 + x0);
            output[pout] = (input[pin] as f32 - sub[id]) / div[id];
        }
    }
}

/// Float-input variant of `darkroom_rawprepare_mosaic_u16`. Same semantics,
/// just reads f32 from `in_buf` rather than u16.
/// Matches the `TYPE_FLOAT` branch of process() in src/iop/rawprepare.c.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawprepare_mosaic_f32(
    in_buf: *const f32,
    out_buf: *mut f32,
    out_width: usize,
    out_height: usize,
    in_width: usize,
    csx: i32,
    csy: i32,
    x0: i32,
    y0: i32,
    sub: *const f32,
    div: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, in_width * (out_height + csy.max(0) as usize));
    let output = std::slice::from_raw_parts_mut(out_buf, out_width * out_height);
    let sub = std::slice::from_raw_parts(sub, 4);
    let div = std::slice::from_raw_parts(div, 4);

    for j in 0..out_height {
        for i in 0..out_width {
            let pin = ((j as i32 + csy) as usize) * in_width + (i as i32 + csx) as usize;
            let pout = j * out_width + i;
            let id = bayer_quadrant(j as i32 + y0, i as i32 + x0);
            output[pout] = (input[pin] - sub[id]) / div[id];
        }
    }
}

/// RGBA pre-downsampled variant: per-channel black/scale rather than per-quadrant.
///
/// Matches the `else` (no-mosaic) branch of process() in src/iop/rawprepare.c.
/// `sub[c]` / `div[c]` are 4-channel arrays. `ch` is the channel count
/// (typically 4 for RGBA).
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawprepare_rgba(
    in_buf: *const f32,
    out_buf: *mut f32,
    out_width: usize,
    out_height: usize,
    in_width: usize,
    csx: i32,
    csy: i32,
    sub: *const f32,
    div: *const f32,
    ch: usize,
) {
    let input = std::slice::from_raw_parts(in_buf, in_width * (out_height + csy.max(0) as usize) * ch);
    let output = std::slice::from_raw_parts_mut(out_buf, out_width * out_height * ch);
    let sub = std::slice::from_raw_parts(sub, ch);
    let div = std::slice::from_raw_parts(div, ch);

    for j in 0..out_height {
        for i in 0..out_width {
            let in_pixel_base = ch * ((j as i32 + csy) as usize * in_width + (i as i32 + csx) as usize);
            let out_pixel_base = ch * (j * out_width + i);
            for c in 0..ch {
                output[out_pixel_base + c] =
                    (input[in_pixel_base + c] - sub[c]) / div[c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer_quadrant_returns_zero_one_two_three() {
        assert_eq!(bayer_quadrant(0, 0), 0);
        assert_eq!(bayer_quadrant(0, 1), 1);
        assert_eq!(bayer_quadrant(1, 0), 2);
        assert_eq!(bayer_quadrant(1, 1), 3);
    }

    #[test]
    fn bayer_quadrant_negative_rows_wrap() {
        // rem_euclid keeps the parity consistent for negative offsets
        assert_eq!(bayer_quadrant(-1, 0), 2);
        assert_eq!(bayer_quadrant(-2, 0), 0);
    }

    #[test]
    fn mosaic_u16_subtracts_per_quadrant_black_and_divides() {
        // 2x2 image. Quadrant 0 = (R), 1 = (G1), 2 = (G2), 3 = (B).
        let input: Vec<u16> = vec![100, 200, 300, 400];
        let mut out = vec![-1.0_f32; 4];
        let sub = [10.0_f32, 20.0, 30.0, 40.0];
        let div = [2.0_f32, 4.0, 5.0, 8.0];
        unsafe {
            darkroom_rawprepare_mosaic_u16(
                input.as_ptr(), out.as_mut_ptr(), 2, 2, 2, 0, 0, 0, 0,
                sub.as_ptr(), div.as_ptr(),
            );
        }
        // (100-10)/2 = 45, (200-20)/4 = 45, (300-30)/5 = 54, (400-40)/8 = 45
        assert!((out[0] - 45.0).abs() < 1e-5);
        assert!((out[1] - 45.0).abs() < 1e-5);
        assert!((out[2] - 54.0).abs() < 1e-5);
        assert!((out[3] - 45.0).abs() < 1e-5);
    }

    #[test]
    fn mosaic_u16_respects_csx_csy_input_crop() {
        // 4x4 input, 2x2 output starting at (1, 1) of input.
        let input: Vec<u16> = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        ];
        let mut out = vec![0.0_f32; 4];
        let sub = [0.0_f32; 4];
        let div = [1.0_f32; 4];
        unsafe {
            darkroom_rawprepare_mosaic_u16(
                input.as_ptr(), out.as_mut_ptr(), 2, 2, 4, 1, 1, 0, 0,
                sub.as_ptr(), div.as_ptr(),
            );
        }
        // out should be input[(1,1), (1,2), (2,1), (2,2)] = [6, 7, 10, 11]
        assert_eq!(out, vec![6.0, 7.0, 10.0, 11.0]);
    }

    #[test]
    fn mosaic_f32_matches_formula() {
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; 4];
        let sub = [0.5_f32; 4];
        let div = [2.0_f32; 4];
        unsafe {
            darkroom_rawprepare_mosaic_f32(
                input.as_ptr(), out.as_mut_ptr(), 2, 2, 2, 0, 0, 0, 0,
                sub.as_ptr(), div.as_ptr(),
            );
        }
        // (x-0.5)/2 for each input
        for k in 0..4 {
            assert!((out[k] - (input[k] - 0.5) / 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn rgba_applies_per_channel_black_and_scale() {
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0_f32; 4];
        let sub = [0.0_f32, 0.5, 1.0, 1.5];
        let div = [2.0_f32, 2.0, 2.0, 2.0];
        unsafe {
            darkroom_rawprepare_rgba(
                input.as_ptr(), out.as_mut_ptr(), 1, 1, 1, 0, 0,
                sub.as_ptr(), div.as_ptr(), 4,
            );
        }
        assert!((out[0] - 0.5).abs() < 1e-6); // (1-0)/2
        assert!((out[1] - 0.75).abs() < 1e-6); // (2-0.5)/2
        assert!((out[2] - 1.0).abs() < 1e-6);  // (3-1)/2
        assert!((out[3] - 1.25).abs() < 1e-6); // (4-1.5)/2
    }
}

/// Shift an interleaved (x,y) coordinate array by (-dx, -dy): forward distort.
///
/// points layout: [x0, y0, x1, y1, ...] — points_count pairs.
/// Matches DT_OMP_FOR_SIMD in src/iop/rawprepare.c:219.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawprepare_distort_transform(
    points: *mut f32,
    points_count: usize,
    dx: f32,
    dy: f32,
) {
    let buf = std::slice::from_raw_parts_mut(points, points_count * 2);
    for i in (0..points_count * 2).step_by(2) {
        buf[i]     -= dx;
        buf[i + 1] -= dy;
    }
}

/// Shift an interleaved (x,y) coordinate array by (+dx, +dy): back-transform.
///
/// Matches DT_OMP_FOR_SIMD in src/iop/rawprepare.c:244.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawprepare_distort_backtransform(
    points: *mut f32,
    points_count: usize,
    dx: f32,
    dy: f32,
) {
    let buf = std::slice::from_raw_parts_mut(points, points_count * 2);
    for i in (0..points_count * 2).step_by(2) {
        buf[i]     += dx;
        buf[i + 1] += dy;
    }
}

/// Apply per-pixel Bayer-channel gain maps (bilinear interpolation) to a
/// single-channel raw output buffer.
///
/// For each output pixel (i, j):
///   channel = (((j + roi_y + top) & 1) << 1) | ((i + roi_x + left) & 1)   [_BL]
///   map_x   = clamp(((roi_x+csx+i)*im_to_rel_x - map_origin_h)*rel_to_map_x, 0, map_w)
///   map_y   = clamp(((roi_y+csy+j)*im_to_rel_y - map_origin_v)*rel_to_map_y, 0, map_h)
///   gain    = bilinear(maps[channel], map_x, map_y)
///   out[j*width+i] *= gain
///
/// `maps` must point to an array of 4 `const float *` pointers, each pointing
/// to a `map_w * map_h` float array.
///
/// Matches the DT_OMP_FOR in src/iop/rawprepare.c:379.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawprepare_apply_gainmaps(
    out: *mut f32,
    out_width: i32,
    out_height: i32,
    roi_x: i32,
    roi_y: i32,
    csx: i32,
    csy: i32,
    top: i32,
    left: i32,
    im_to_rel_x: f32,
    im_to_rel_y: f32,
    rel_to_map_x: f32,
    rel_to_map_y: f32,
    map_origin_h: f32,
    map_origin_v: f32,
    map_w: u32,
    map_h: u32,
    maps: *const *const f32,
) {
    if out_width <= 0 || out_height <= 0 { return; }
    let w = out_width as usize;
    let h = out_height as usize;
    let mw = map_w as usize;
    let mh = map_h as usize;
    let output = std::slice::from_raw_parts_mut(out, w * h);
    let map_ptrs = std::slice::from_raw_parts(maps, 4);
    let map_data: [&[f32]; 4] = [
        std::slice::from_raw_parts(map_ptrs[0], mw * mh),
        std::slice::from_raw_parts(map_ptrs[1], mw * mh),
        std::slice::from_raw_parts(map_ptrs[2], mw * mh),
        std::slice::from_raw_parts(map_ptrs[3], mw * mh),
    ];

    for j in 0..h {
        let y_raw = ((roi_y + csy + j as i32) as f32 * im_to_rel_y - map_origin_v) * rel_to_map_y;
        let y_map = y_raw.clamp(0.0, map_h as f32);
        let y_i0 = (y_map as usize).min(mh - 1);
        let y_i1 = (y_i0 + 1).min(mh - 1);
        let y_frac = y_map - y_i0 as f32;

        for i in 0..w {
            let ch = ((((j as i32 + roi_y + top) & 1) << 1) | ((i as i32 + roi_x + left) & 1)) as usize;
            let x_raw = ((roi_x + csx + i as i32) as f32 * im_to_rel_x - map_origin_h) * rel_to_map_x;
            let x_map = x_raw.clamp(0.0, map_w as f32);
            let x_i0 = (x_map as usize).min(mw - 1);
            let x_i1 = (x_i0 + 1).min(mw - 1);
            let x_frac = x_map - x_i0 as f32;

            let row0 = &map_data[ch][y_i0 * mw..];
            let row1 = &map_data[ch][y_i1 * mw..];
            let gain_top    = (1.0 - x_frac) * row0[x_i0] + x_frac * row0[x_i1];
            let gain_bottom = (1.0 - x_frac) * row1[x_i0] + x_frac * row1[x_i1];
            output[j * w + i] *= (1.0 - y_frac) * gain_top + y_frac * gain_bottom;
        }
    }
}
