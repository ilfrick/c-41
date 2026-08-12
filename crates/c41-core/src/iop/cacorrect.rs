use crate::{params::IopParams, raw, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Cacorrect;

impl IopProcess for Cacorrect {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "cacorrect" }
}

/// Copy the non-green Bayer channel from the raw input into a half-resolution buffer.
///
/// For each row, the starting column is `FC(row, 0, filters) & 1`, stepping by 2.
/// `oldraw[row * h_width + col/2] = in[row * full_width + col]`
///
/// Matches the DT_OMP_FOR loop at src/iop/cacorrect.c:327.
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrect_save_oldraw(
    in_buf: *const f32,
    oldraw_buf: *mut f32,
    full_width: usize,
    height: usize,
    h_width: usize,
    filters: u32,
) {
    if full_width == 0 || height == 0 { return; }
    let inp    = std::slice::from_raw_parts(in_buf, full_width * height);
    let oldraw = std::slice::from_raw_parts_mut(oldraw_buf, h_width * height);

    for row in 0..height {
        let first_col = (raw::fc_bayer(row as i32, 0, filters) & 1) as usize;
        let mut col = first_col;
        while col < full_width {
            oldraw[row * h_width + col / 2] = inp[row * full_width + col];
            col += 2;
        }
    }
}

/// Compute per-pixel R/B correction factors for the avoidshift pass.
///
/// For each non-green pixel (row, col) in the Bayer mosaic:
///   nongreen[(row/2)*h_width + col/2] = clamp(oldraw[oindex] / in[index], 0.5, 2.0)
///
/// `red_buf` receives R-channel factors; `blue_buf` receives B-channel factors.
/// The caller must pass the correct h_width (= full_width/2 rounded up).
///
/// Matches the DT_OMP_FOR loop at src/iop/cacorrect.c:1125.
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrect_compute_factors(
    in_buf: *const f32,
    oldraw_buf: *const f32,
    red_buf: *mut f32,
    blue_buf: *mut f32,
    full_width: usize,
    height: usize,
    h_width: usize,
    filters: u32,
) {
    if full_width == 0 || height == 0 { return; }
    let inp    = std::slice::from_raw_parts(in_buf,     full_width * height);
    let oldraw = std::slice::from_raw_parts(oldraw_buf, h_width    * height);
    let red    = std::slice::from_raw_parts_mut(red_buf,  h_width * (height / 2 + 1));
    let blue   = std::slice::from_raw_parts_mut(blue_buf, h_width * (height / 2 + 1));

    for row in 0..height {
        let first_col = (raw::fc_bayer(row as i32, 0, filters) & 1) as usize;
        let color     = raw::fc_bayer(row as i32, first_col as i32, filters);
        let nongreen: &mut [f32] = if color == 0 { red } else { blue };
        let mut col = first_col;
        while col < full_width {
            let index  = row * full_width + col;
            let oindex = row * h_width + col / 2;
            let raw_val = inp[index];
            let factor = if raw_val.abs() > 1e-9 {
                (oldraw[oindex] / raw_val).clamp(0.5, 2.0)
            } else {
                1.0
            };
            nongreen[(row / 2) * h_width + col / 2] = factor;
            col += 2;
        }
    }
}

/// Apply the blurred per-pixel correction factors to the output buffer.
///
/// For each non-green pixel (row, col) in [2..height-2) × [firstcol..width-2):
///   out[row*width + col] *= nongreen[row/2*h_width + col/2]
///
/// Matches the DT_OMP_FOR loop at src/iop/cacorrect.c:1172.
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrect_apply_factors(
    out_buf: *mut f32,
    red_buf: *const f32,
    blue_buf: *const f32,
    full_width: usize,
    height: usize,
    h_width: usize,
    filters: u32,
) {
    if full_width < 5 || height < 5 { return; }
    let out  = std::slice::from_raw_parts_mut(out_buf, full_width * height);
    let red  = std::slice::from_raw_parts(red_buf,  h_width * (height / 2 + 1));
    let blue = std::slice::from_raw_parts(blue_buf, h_width * (height / 2 + 1));

    for row in 2..(height - 2) {
        let first_col = (raw::fc_bayer(row as i32, 0, filters) & 1) as usize;
        let color     = raw::fc_bayer(row as i32, first_col as i32, filters);
        let nongreen: &[f32] = if color == 0 { red } else { blue };
        let mut col = first_col;
        while col < full_width - 2 {
            let correction = nongreen[(row / 2) * h_width + col / 2];
            out[row * full_width + col] *= correction;
            col += 2;
        }
    }
}

/// Write out the corrected raw buffer to the output roi, applying a scale factor.
///
/// For each output pixel (row, col):
///   input  coords: irow = row + roi_out_y, icol = col + roi_out_x
///   if (irow, icol) is within (roi_in_height × roi_in_width):
///     output[row * out_width + col] = out[irow * in_width + icol] * scaler
///
/// Matches the DT_OMP_FOR(collapse(2)) at src/iop/cacorrect.c:1190.
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrect_writeout(
    corrected: *const f32,  // full internal buffer (roi_in dimensions)
    output: *mut f32,       // roi_out buffer
    out_width: usize,
    out_height: usize,
    in_width: usize,
    in_height: usize,
    roi_out_x: i32,
    roi_out_y: i32,
    scaler: f32,
) {
    if out_width == 0 || out_height == 0 { return; }
    let corr = std::slice::from_raw_parts(corrected, in_width * in_height);
    let out  = std::slice::from_raw_parts_mut(output, out_width * out_height);

    for row in 0..out_height {
        let irow = (row as i32 + roi_out_y) as usize;
        for col in 0..out_width {
            let icol = (col as i32 + roi_out_x) as usize;
            if irow < in_height && icol < in_width {
                out[row * out_width + col] = corr[irow * in_width + icol] * scaler;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGGB: u32 = 0x94949494;

    #[test]
    fn save_oldraw_extracts_bayer_columns() {
        // RGGB: row 0, FC(0,0)=0 (R), FC(0,1)=1 (G). R is at col 0 (even).
        // first_col = FC(0,0,RGGB) & 1 = 0 → cols 0, 2, 4 ...
        let inp = vec![1.0_f32, 2.0, 3.0, 4.0]; // 4×1 row
        let mut oldraw = vec![0.0_f32; 2]; // h_width = 2
        unsafe {
            darkroom_cacorrect_save_oldraw(inp.as_ptr(), oldraw.as_mut_ptr(), 4, 1, 2, RGGB);
        }
        assert_eq!(oldraw[0], 1.0); // col 0
        assert_eq!(oldraw[1], 3.0); // col 2
    }

    #[test]
    fn writeout_applies_scaler_and_bounds_check() {
        // 3×3 internal, write a 2×2 roi offset by (1,1)
        let corrected = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 2.0,
            0.0, 3.0, 4.0,
        ];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_cacorrect_writeout(
                corrected.as_ptr(), out.as_mut_ptr(),
                2, 2, 3, 3, 1, 1, 2.0,
            );
        }
        // (0,0)→ corrected[1*3+1]*2 = 1*2=2; (0,1)→2*2=4; (1,0)→3*2=6; (1,1)→4*2=8
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
        assert!((out[2] - 6.0).abs() < 1e-6);
        assert!((out[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn apply_factors_multiplies_nongreen_pixels() {
        // 8×1 row (full_width=8, height=5 so border rows are skipped)
        // Use row 2, RGGB: first_col = 0, color = R → uses red_buf
        let mut out = vec![2.0_f32; 8 * 5]; // constant 2.0
        let h_width = 4;
        let red  = vec![3.0_f32; h_width * 3]; // correction=3.0
        let blue = vec![1.0_f32; h_width * 3];
        unsafe {
            darkroom_cacorrect_apply_factors(
                out.as_mut_ptr(), red.as_ptr(), blue.as_ptr(),
                8, 5, h_width, RGGB,
            );
        }
        // Row 2, col 0 is R → out[2*8+0] = 2*3 = 6
        assert!((out[2 * 8] - 6.0).abs() < 1e-6, "out={}", out[2 * 8]);
        // Row 2, col 1 is G (skipped) → unchanged = 2.0
        assert_eq!(out[2 * 8 + 1], 2.0);
    }
}
