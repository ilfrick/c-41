//! Sharpen IOP — separable Gaussian blur + unsharp mask.
//!
//! Ports the single `DT_OMP_FOR` row loop in `src/iop/sharpen.c::process()`.
//! The Gaussian kernel `mat` is still built on the C side by
//! `init_gaussian_kernel()`; Rust consumes its first `2*rad + 1` taps.

use rayon::prelude::*;

/// Separable unsharp-mask sharpen over the luma (channel 0) plane.
///
/// For each interior row the luma channel is first blurred vertically into a
/// per-row scratch buffer, then blurred horizontally; the high-frequency
/// residual (input luma minus blurred luma) is thresholded and added back,
/// scaled by `amount`. Chroma channels 1 and 2 pass through unchanged and the
/// alpha channel (3) is left untouched, exactly as the C loop does. The top and
/// bottom `rad` rows, and the left/right `rad` border columns, are copied
/// verbatim from the input (all four channels).
///
/// # Safety
/// `in_buf`/`out_buf` must point to `width * height * 4` floats and `mat` to at
/// least `2 * rad + 1` floats. Caller guarantees `rad >= 1` and both
/// `width`/`height >= 2 * rad + 1` (the C caller's fast-path handles the rest).
#[no_mangle]
pub unsafe extern "C" fn darkroom_sharpen_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    mat: *const f32,
    width: usize,
    height: usize,
    rad: i32,
    threshold: f32,
    amount: f32,
) {
    let rad = rad as usize;
    let wd = 2 * rad + 1;
    let n = width * height;
    let input = std::slice::from_raw_parts(in_buf, n * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, n * 4);
    let mat = std::slice::from_raw_parts(mat, wd);

    output
        .par_chunks_exact_mut(width * 4)
        .enumerate()
        .for_each(|(j, row_out)| {
            // Top/bottom border rows: pass through unchanged (all 4 channels).
            if j < rad || j >= height - rad {
                let base = j * width * 4;
                row_out.copy_from_slice(&input[base..base + width * 4]);
                return;
            }

            // Vertically blur the luma channel of every column into scratch.
            let mut temp = vec![0f32; width];
            let start_row = j - rad;
            let end_row = j + rad;
            for (i, t) in temp.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for k in start_row..=end_row {
                    sum += mat[k - start_row] * input[4 * (k * width + i)];
                }
                *t = sum;
            }

            // Left border columns: unsharpened pass-through.
            for i in 0..rad {
                let src = 4 * (j * width + i);
                row_out[4 * i..4 * i + 4].copy_from_slice(&input[src..src + 4]);
            }

            // Horizontally blur the vertically-blurred luma, then unsharp-mask.
            for i in rad..width - rad {
                let mut sum = 0.0f32;
                for k in (i - rad)..=(i + rad) {
                    sum += mat[k - (i - rad)] * temp[k];
                }
                let index = 4 * (j * width + i);
                let diff = input[index] - sum;
                let absdiff = diff.abs();
                let detail = if absdiff > threshold {
                    (absdiff - threshold).max(0.0).copysign(diff)
                } else {
                    0.0
                };
                row_out[4 * i] = input[index] + detail * amount;
                row_out[4 * i + 1] = input[index + 1];
                row_out[4 * i + 2] = input[index + 2];
                // channel 3 (alpha) left untouched, matching the C loop.
            }

            // Right border columns: unsharpened pass-through.
            for i in (width - rad)..width {
                let src = 4 * (j * width + i);
                row_out[4 * i..4 * i + 4].copy_from_slice(&input[src..src + 4]);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat (constant) image must come out unchanged: the blur of a constant
    /// equals the constant, so the high-frequency residual is zero everywhere.
    #[test]
    fn flat_image_passes_through() {
        let (w, h) = (8usize, 8usize);
        let input = vec![0.5f32; w * h * 4];
        let mut output = vec![0f32; w * h * 4];
        // 3-tap normalized Gaussian (rad = 1).
        let mat = [0.25f32, 0.5, 0.25];
        unsafe {
            darkroom_sharpen_process(
                input.as_ptr(),
                output.as_mut_ptr(),
                mat.as_ptr(),
                w,
                h,
                1,
                0.0,
                1.0,
            );
        }
        for i in 0..w * h {
            assert!((output[4 * i] - 0.5).abs() < 1e-6, "luma at {i}");
        }
    }

    /// Chroma channels (1, 2) must pass through untouched in the interior.
    #[test]
    fn chroma_passthrough() {
        let (w, h) = (6usize, 6usize);
        let mut input = vec![0f32; w * h * 4];
        for i in 0..w * h {
            input[4 * i] = 0.3;
            input[4 * i + 1] = 0.6;
            input[4 * i + 2] = 0.9;
        }
        let mut output = vec![0f32; w * h * 4];
        let mat = [0.25f32, 0.5, 0.25];
        unsafe {
            darkroom_sharpen_process(
                input.as_ptr(),
                output.as_mut_ptr(),
                mat.as_ptr(),
                w,
                h,
                1,
                0.0,
                1.0,
            );
        }
        // interior pixel (2,2)
        let idx = 4 * (2 * w + 2);
        assert!((output[idx + 1] - 0.6).abs() < 1e-6);
        assert!((output[idx + 2] - 0.9).abs() < 1e-6);
    }

    /// A bright single-pixel spike should be amplified (positive detail added).
    #[test]
    fn spike_is_amplified() {
        let (w, h) = (7usize, 7usize);
        let mut input = vec![0.1f32; w * h * 4];
        let center = 3 * w + 3;
        input[4 * center] = 0.9; // bright luma spike
        let mut output = vec![0f32; w * h * 4];
        let mat = [0.25f32, 0.5, 0.25];
        unsafe {
            darkroom_sharpen_process(
                input.as_ptr(),
                output.as_mut_ptr(),
                mat.as_ptr(),
                w,
                h,
                1,
                0.0,
                1.0,
            );
        }
        // The spike's luma should be pushed above its input value.
        assert!(output[4 * center] > 0.9, "spike not amplified: {}", output[4 * center]);
    }
}
