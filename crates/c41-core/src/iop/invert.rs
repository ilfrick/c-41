use crate::{params::IopParams, raw, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Invert;

impl IopProcess for Invert {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "invert" }
}

/// Invert a Bayer mosaic by subtracting each pixel from its per-channel film value.
///
///   out[j*width + i] = clamp(film_rgb[FC(j+roi_y, i+roi_x, filters)] - in[...], 0, 1)
///
/// Matches the Bayer DT_OMP_FOR in src/iop/invert.c:304.
#[no_mangle]
pub unsafe extern "C" fn darkroom_invert_bayer(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    filters: u32,
    roi_x: i32,
    roi_y: i32,
    film_rgb: *const f32, // 4 floats [R, G1, G2, B]
) {
    if width == 0 || height == 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf,   width * height);
    let out  = std::slice::from_raw_parts_mut(out_buf, width * height);
    let film = std::slice::from_raw_parts(film_rgb, 4);

    for j in 0..height {
        for i in 0..width {
            let c  = raw::fc_bayer((j as i32) + roi_y, (i as i32) + roi_x, filters);
            let p  = j * width + i;
            out[p] = (film[c] - inp[p]).clamp(0.0, 1.0);
        }
    }
}

/// Invert an X-Trans mosaic.
///
///   out[j*width + i] = clamp(film_rgb[FCxtrans(j+roi_y, i+roi_x)] - in[...], 0, 1)
///
/// Matches the X-Trans DT_OMP_FOR in src/iop/invert.c:253.
#[no_mangle]
pub unsafe extern "C" fn darkroom_invert_xtrans(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    xtrans: *const u8, // flat 36-byte 6x6
    roi_x: i32,
    roi_y: i32,
    film_rgb: *const f32, // 3 floats [R, G, B]
) {
    if width == 0 || height == 0 { return; }
    let inp  = std::slice::from_raw_parts(in_buf,   width * height);
    let out  = std::slice::from_raw_parts_mut(out_buf, width * height);
    let film = std::slice::from_raw_parts(film_rgb, 3);
    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 { for c in 0..6 { xt[r][c] = xt_bytes[r * 6 + c]; } }

    for j in 0..height {
        for i in 0..width {
            let c  = raw::fc_xtrans((j as i32) + roi_y, (i as i32) + roi_x, &xt);
            let p  = j * width + i;
            out[p] = (film[c.min(2)] - inp[p]).clamp(0.0, 1.0);
        }
    }
}

/// Non-mosaiced (4-channel RGBA) inversion: out[k][c] = color[c] - in[k][c].
///
/// Replaces the non-raw DT_OMP_FOR loop in src/iop/invert.c::process().
/// color points to 4 floats: { d->color[0], d->color[1], d->color[2], 1.0f }.
/// X-Trans and Bayer mosaic paths remain in C.
#[no_mangle]
pub unsafe extern "C" fn darkroom_invert_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    color: *const f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let col    = std::slice::from_raw_parts(color, 4);
    for k in 0..npixels {
        output[k * 4]     = col[0] - input[k * 4];
        output[k * 4 + 1] = col[1] - input[k * 4 + 1];
        output[k * 4 + 2] = col[2] - input[k * 4 + 2];
        output[k * 4 + 3] = col[3] - input[k * 4 + 3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_known_values() {
        let input = vec![0.2f32, 0.5, 0.8, 0.0,
                         1.0f32, 0.0, 0.3, 0.0];
        let color = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 8];
        unsafe { darkroom_invert_process(input.as_ptr(), out.as_mut_ptr(), 2, color.as_ptr()); }
        assert!((out[0] - 0.8).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.2).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6); // 1 - 0 = 1
        assert!((out[4] - 0.0).abs() < 1e-6); // 1 - 1 = 0
    }

    #[test]
    fn invert_identity_color_one() {
        let input = vec![0.5f32, 0.5, 0.5, 0.0];
        let color = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        unsafe { darkroom_invert_process(input.as_ptr(), out.as_mut_ptr(), 1, color.as_ptr()); }
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bayer_invert_subtracts_film_per_channel() {
        // 2×2 RGGB: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B
        let inp   = vec![0.3_f32, 0.5, 0.2, 0.8];
        let mut out = vec![0.0_f32; 4];
        let film  = [1.0_f32, 0.9, 0.8, 0.0];
        unsafe {
            darkroom_invert_bayer(inp.as_ptr(), out.as_mut_ptr(), 2, 2, 0x94949494u32, 0, 0, film.as_ptr());
        }
        assert!((out[0] - 0.7).abs() < 1e-5);
        assert!((out[1] - 0.4).abs() < 1e-5);
        assert!((out[2] - 0.7).abs() < 1e-5);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn bayer_invert_clamps_to_zero() {
        let inp  = vec![1.0_f32; 4];
        let mut out = vec![-1.0_f32; 4];
        let film = [0.5_f32; 4];
        unsafe { darkroom_invert_bayer(inp.as_ptr(), out.as_mut_ptr(), 2, 2, 0x94949494u32, 0, 0, film.as_ptr()); }
        for v in &out { assert_eq!(*v, 0.0); }
    }
}
