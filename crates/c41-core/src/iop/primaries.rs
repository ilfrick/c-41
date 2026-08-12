use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct Primaries;

impl IopProcess for Primaries {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "primaries" }
}

/// Apply a transposed 4×4 color matrix (only first 3 rows/cols active).
///
/// Matches dt_apply_transposed_color_matrix: out[r] = Σ matrix[c][r] * in[c] for c in 0..3
#[no_mangle]
pub unsafe extern "C" fn darkroom_primaries_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    matrix: *const f32, // float[4][4] = 16 floats, row-major
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    // matrix[row][col] stored row-major as 16 floats
    // dt_apply_transposed: out[r] = matrix[0][r]*in[0] + matrix[1][r]*in[1] + matrix[2][r]*in[2]
    // In flat indexing: matrix[row*4 + col]
    let m = std::slice::from_raw_parts(matrix, 16);
    for px in 0..npixels {
        let i = &inp[px * 4..px * 4 + 4];
        let o = &mut out[px * 4..px * 4 + 4];
        o[0] = m[0] * i[0] + m[4] * i[1] + m[8]  * i[2];
        o[1] = m[1] * i[0] + m[5] * i[1] + m[9]  * i[2];
        o[2] = m[2] * i[0] + m[6] * i[1] + m[10] * i[2];
        o[3] = i[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matrix_passes_through() {
        #[rustfmt::skip]
        let matrix: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let inp = [0.2f32, 0.5, 0.8, 1.0];
        let mut out = [0f32; 4];
        unsafe { darkroom_primaries_process(inp.as_ptr(), out.as_mut_ptr(), 1, matrix.as_ptr()) };
        assert!((out[0] - 0.2).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.8).abs() < 1e-6);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn swap_channels() {
        // swap R and G: matrix[0] = (0,1,0,0), matrix[1] = (1,0,0,0)
        #[rustfmt::skip]
        let matrix: [f32; 16] = [
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let inp = [0.1f32, 0.9, 0.5, 0.0];
        let mut out = [0f32; 4];
        unsafe { darkroom_primaries_process(inp.as_ptr(), out.as_mut_ptr(), 1, matrix.as_ptr()) };
        assert!((out[0] - 0.9).abs() < 1e-6);
        assert!((out[1] - 0.1).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
    }
}
