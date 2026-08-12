use crate::{math, params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Censorize;

impl IopProcess for Censorize {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "censorize" }
}

#[inline(always)]
fn clamp_i(v: isize, lo: isize, hi: isize) -> isize {
    v.max(lo).min(hi)
}

/// Pixelate (mosaic) an RGBA image: divide into blocks of size `2*pixel_radius`,
/// sample 5 points per block (top-left, top-right, centre, bottom-left, bottom-right),
/// average them, and fill every pixel of the block with that average.
///
/// Matches the `pixelate` loop in src/iop/censorize.c (process()).
/// `pixel_radius` is the half-block size; if it's 0 the function is a no-op and
/// the caller must not invoke us. width and height are the image dimensions.
#[no_mangle]
pub unsafe extern "C" fn darkroom_censorize_pixelate(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    pixel_radius: usize,
) {
    if pixel_radius == 0 || width == 0 || height == 0 {
        return;
    }
    let input = std::slice::from_raw_parts(in_buf, width * height * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, width * height * 4);

    let pixels_x = width  / (2 * pixel_radius);
    let pixels_y = height / (2 * pixel_radius);
    let w_i = width  as isize;
    let h_i = height as isize;
    let r_i = pixel_radius as isize;

    for j in 0..=pixels_y {
        for i in 0..=pixels_x {
            let tl_x = clamp_i((2 * r_i) * i as isize, 0, w_i - 1);
            let tl_y = clamp_i((2 * r_i) * j as isize, 0, h_i - 1);
            let cc_x = clamp_i(tl_x + r_i, 0, w_i - 1);
            let cc_y = clamp_i(tl_y + r_i, 0, h_i - 1);
            let br_x = clamp_i(cc_x + r_i, 0, w_i - 1);
            let br_y = clamp_i(cc_y + r_i, 0, h_i - 1);

            let box_pts: [(isize, isize); 5] = [
                (tl_x, tl_y), (br_x, tl_y), (cc_x, cc_y),
                (tl_x, br_y), (br_x, br_y),
            ];

            let mut rgb = [0.0_f32; 4];
            for &(x, y) in &box_pts {
                let idx = ((y as usize) * width + x as usize) * 4;
                for c in 0..4 {
                    rgb[c] += input[idx + c] / 5.0;
                }
            }

            // paint the block tl..br (half-open in both dims)
            for jj in (tl_y as usize)..(br_y as usize) {
                for ii in (tl_x as usize)..(br_x as usize) {
                    let pidx = (jj * width + ii) * 4;
                    for c in 0..4 {
                        output[pidx + c] = rgb[c];
                    }
                }
            }
        }
    }
}

/// Add per-pixel Gaussian noise to a pixelated/blurred RGBA buffer.
///
/// For each pixel (i, j):
///   seed xoshiro128+ from (j+1, (j+1)*(i+3), 1337, 666) → 4 warm-up
///   norm  = output[pix + 1]   (second channel = G in RGB)
///   eps   = gaussian_noise(norm, noise*norm, (i%2 || j%2)) / norm
///   out[c] = max(out[c] * eps, 0)  for c in 0..3
///
/// Matches `make_noise()` in src/iop/censorize.c:107.
#[no_mangle]
pub unsafe extern "C" fn darkroom_censorize_make_noise(
    output: *mut f32,
    noise: f32,
    width: usize,
    height: usize,
) {
    if width == 0 || height == 0 { return; }
    let out = std::slice::from_raw_parts_mut(output, width * height * 4);

    for i in 0..height {
        for j in 0..width {
            let mut state = [
                math::splitmix32((j + 1) as u64),
                math::splitmix32(((j + 1) * (i + 3)) as u64),
                math::splitmix32(1337),
                math::splitmix32(666),
            ];
            for _ in 0..4 { math::xoshiro128plus(&mut state); }

            let idx = (i * width + j) * 4;
            let norm = out[idx + 1];
            let flip = (i % 2 != 0) || (j % 2 != 0);
            let eps  = if norm.abs() > 1e-10 {
                math::gaussian_noise(norm, noise * norm, flip, &mut state) / norm
            } else {
                0.0
            };
            for c in 0..3 {
                out[idx + c] = (out[idx + c] * eps).max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid(width: usize, height: usize, rgba: [f32; 4]) -> Vec<f32> {
        let mut buf = vec![0.0_f32; width * height * 4];
        for px in 0..(width * height) {
            for c in 0..4 { buf[px * 4 + c] = rgba[c]; }
        }
        buf
    }

    #[test]
    fn solid_image_stays_solid() {
        let w = 16; let h = 16;
        let input = make_solid(w, h, [0.2, 0.4, 0.6, 1.0]);
        let mut out = vec![-1.0_f32; w * h * 4];
        unsafe { darkroom_censorize_pixelate(input.as_ptr(), out.as_mut_ptr(), w, h, 2); }
        // every painted pixel should be (0.2, 0.4, 0.6, 1.0); the rightmost / bottom
        // sliver may stay at -1.0 because the loops are half-open, so we check the
        // interior only.
        for jj in 0..(h - 4) {
            for ii in 0..(w - 4) {
                let pidx = (jj * w + ii) * 4;
                for (c, &expected) in [0.2_f32, 0.4, 0.6, 1.0].iter().enumerate() {
                    assert!((out[pidx + c] - expected).abs() < 1e-5,
                            "pix({ii},{jj}) c={c} got {} want {}", out[pidx + c], expected);
                }
            }
        }
    }

    #[test]
    fn averages_corner_samples() {
        // 4x4 image with a single corner pixel set, radius=2 → single block 0..4 x 0..4.
        // box samples: tl(0,0), tr(3,0), cc(2,2), bl(0,3), br(3,3).
        // Set the 5 sample points to known values; everything else 0.
        let w = 4; let h = 4;
        let mut input = vec![0.0_f32; w * h * 4];
        let samples = [(0, 0, 0.5_f32), (3, 0, 0.25), (2, 2, 1.0), (0, 3, 0.75), (3, 3, 0.0)];
        for (x, y, v) in samples.iter() {
            let idx = (y * w + x) * 4;
            input[idx] = *v; // just R channel
        }
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe { darkroom_censorize_pixelate(input.as_ptr(), out.as_mut_ptr(), w, h, 2); }
        // Expected R average: (0.5 + 0.25 + 1.0 + 0.75 + 0.0) / 5 = 0.5
        // The block fills [tl_y..br_y) x [tl_x..br_x) = [0..2) x [0..2) — only the
        // top-left 2x2 region (because r=2 → tl=(0,0), cc=(2,2), br=(2,2) clamped 3).
        // Specifically: tl=(0,0), cc=clamp(0+2,0,3)=(2,2), br=clamp(2+2,0,3)=(3,3).
        // So block fills [0..3) x [0..3) = 3x3.
        for jj in 0..3 {
            for ii in 0..3 {
                let r = out[(jj * w + ii) * 4];
                assert!((r - 0.5).abs() < 1e-5, "({ii},{jj}) R={r}");
            }
        }
    }

    #[test]
    fn alpha_channel_is_averaged_too() {
        let w = 4; let h = 4;
        let mut input = vec![0.0_f32; w * h * 4];
        // all 5 sample points get alpha=1.0; others 0
        for (x, y) in [(0,0), (3,0), (2,2), (0,3), (3,3)].iter() {
            input[(y * w + x) * 4 + 3] = 1.0;
        }
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe { darkroom_censorize_pixelate(input.as_ptr(), out.as_mut_ptr(), w, h, 2); }
        // alpha avg = 5 * 1.0 / 5 = 1.0
        let a = out[3];
        assert!((a - 1.0).abs() < 1e-5, "alpha={a}");
    }

    #[test]
    fn zero_radius_is_noop() {
        let w = 4; let h = 4;
        let input = make_solid(w, h, [0.5, 0.5, 0.5, 1.0]);
        let mut out = vec![-1.0_f32; w * h * 4];
        unsafe { darkroom_censorize_pixelate(input.as_ptr(), out.as_mut_ptr(), w, h, 0); }
        for &v in out.iter() { assert_eq!(v, -1.0); }
    }
}
