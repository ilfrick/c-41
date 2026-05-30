use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Useless;

impl IopProcess for Useless {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "useless" }
}

/// Apply the "useless" checkerboard-dimming effect.
///
/// For every pixel (i, j):
///   wi = (roi_in_x + i) * scale
///   wj = (roi_in_y + j) * scale
///   if ((wi / checker_scale) + (wj / checker_scale)) is odd:
///     out[c] = in[c] * (1 - factor)   for c in 0..ch
///     mask[j*width + i] = 1.0          (if mask is non-null)
///   else:
///     out[c] = in[c]                   (passthrough)
///
/// `in_buf`/`out_buf` are RGBA (or `ch`-channel) float buffers of size
/// `width * height * ch`. `mask_buf` may be null (skipped then).
///
/// Matches `process()` in src/iop/useless.c:393.
#[no_mangle]
pub unsafe extern "C" fn darkroom_useless_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    mask_buf: *mut f32,
    width: usize,
    height: usize,
    ch: usize,
    roi_in_x: i32,
    roi_in_y: i32,
    scale: f32,
    checker_scale: i32,
    factor: f32,
) {
    if width == 0 || height == 0 || ch == 0 { return; }
    let np = width * height;
    let input  = std::slice::from_raw_parts(in_buf, np * ch);
    let output = std::slice::from_raw_parts_mut(out_buf, np * ch);
    let mut mask   = if mask_buf.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(mask_buf, np))
    };
    let cs = checker_scale.max(1) as i32;
    let dim = 1.0 - factor;

    for j in 0..height {
        let wj = (roi_in_y + j as i32) as f32 * scale;
        let checker_j = (wj as i32) / cs;

        for i in 0..width {
            let wi = (roi_in_x + i as i32) as f32 * scale;
            let checker_i = (wi as i32) / cs;
            let pix = (j * width + i) * ch;

            if (checker_i + checker_j) & 1 == 1 {
                for c in 0..ch {
                    output[pix + c] = input[pix + c] * dim;
                }
                if let Some(ref mut m) = mask {
                    m[j * width + i] = 1.0;
                }
            } else {
                output[pix..pix + ch].copy_from_slice(&input[pix..pix + ch]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_checker_cell_dims_pixels() {
        // 2×2 image, checker_scale = 1, scale = 1.0, roi origin (0, 0).
        // cell (wi/1 + wj/1) & 1: (0,0)=0, (1,0)=1, (0,1)=1, (1,1)=0
        let inp = vec![1.0_f32; 2 * 2 * 4];
        let mut out = vec![0.0_f32; 2 * 2 * 4];
        unsafe {
            darkroom_useless_process(
                inp.as_ptr(), out.as_mut_ptr(), std::ptr::null_mut(),
                2, 2, 4, 0, 0, 1.0, 1, 0.5,
            );
        }
        // (0,0) even → passthrough = 1.0
        assert_eq!(out[0], 1.0);
        // (1,0) odd → 1.0 * (1 - 0.5) = 0.5
        assert_eq!(out[4], 0.5);
        // (0,1) odd → 0.5
        assert_eq!(out[8], 0.5);
        // (1,1) even → 1.0
        assert_eq!(out[12], 1.0);
    }

    #[test]
    fn even_checker_passes_through() {
        // scale = 0 → all wi/wj = 0 → all cells are 0 → even → passthrough
        let inp = vec![0.7_f32, 0.3, 0.1, 1.0];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_useless_process(
                inp.as_ptr(), out.as_mut_ptr(), std::ptr::null_mut(),
                1, 1, 4, 0, 0, 0.0, 1, 0.5,
            );
        }
        assert_eq!(&out[..], &inp[..]);
    }

    #[test]
    fn mask_written_for_odd_cells() {
        let inp = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 4];
        let mut mask = vec![-1.0_f32; 1];
        // scale = 1, checker = 1, pixel (0,0) → (0+0) & 1 = 0 → even, mask not written
        unsafe {
            darkroom_useless_process(
                inp.as_ptr(), out.as_mut_ptr(), mask.as_mut_ptr(),
                1, 1, 4, 0, 0, 1.0, 1, 0.3,
            );
        }
        assert_eq!(mask[0], -1.0); // sentinel unchanged for even cell

        // Shift roi_in_x by 1 → wi=1 → (1+0)&1 = 1 → odd, mask written
        unsafe {
            darkroom_useless_process(
                inp.as_ptr(), out.as_mut_ptr(), mask.as_mut_ptr(),
                1, 1, 4, 1, 0, 1.0, 1, 0.3,
            );
        }
        assert_eq!(mask[0], 1.0);
    }

    #[test]
    fn null_mask_is_safe() {
        let inp = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_useless_process(
                inp.as_ptr(), out.as_mut_ptr(), std::ptr::null_mut(),
                1, 1, 4, 1, 0, 1.0, 1, 0.5,
            );
        }
        // odd cell → 0.5; didn't crash
        assert_eq!(out[0], 0.5);
    }
}
