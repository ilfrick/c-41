use crate::{params::IopParams, raw, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Temperature;

impl IopProcess for Temperature {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "temperature" }
}

/// Apply white-balance coefficients to a Bayer mosaic (multiply each sensel by its channel coefficient).
///
///   out[j*width + i] = in[...] * coeffs[FC(j, i, filters)]
///
/// Note: temperature uses `FC(j, i, filters)` WITHOUT roi offset — the sensor
/// pattern repeats, so only the parity matters.
/// Matches the Bayer DT_OMP_FOR in src/iop/temperature.c:590.
#[no_mangle]
pub unsafe extern "C" fn darkroom_temperature_bayer(
    in_buf:  *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    filters: u32,
    coeffs:  *const f32, // 4 floats [R, G1, G2, B]
) {
    if width == 0 || height == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf,   width * height);
    let out = std::slice::from_raw_parts_mut(out_buf, width * height);
    let c4  = std::slice::from_raw_parts(coeffs, 4);

    for j in 0..height {
        for i in 0..width {
            let c  = raw::fc_bayer(j as i32, i as i32, filters);
            let p  = j * width + i;
            out[p] = inp[p] * c4[c];
        }
    }
}

/// Apply white-balance coefficients to an X-Trans mosaic.
///
///   out[j*width + i] = in[...] * coeffs[FCNxtrans(j, i, xtrans)]
///
/// Matches the X-Trans DT_OMP_FOR in src/iop/temperature.c:552.
#[no_mangle]
pub unsafe extern "C" fn darkroom_temperature_xtrans(
    in_buf:  *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    xtrans: *const u8, // flat 36-byte 6x6
    coeffs:  *const f32, // 3 floats [R, G, B]
) {
    if width == 0 || height == 0 { return; }
    let inp = std::slice::from_raw_parts(in_buf,   width * height);
    let out = std::slice::from_raw_parts_mut(out_buf, width * height);
    let c3  = std::slice::from_raw_parts(coeffs, 3);
    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 { for c in 0..6 { xt[r][c] = xt_bytes[r * 6 + c]; } }

    for j in 0..height {
        for i in 0..width {
            let c  = raw::fc_xtrans(j as i32, i as i32, &xt);
            let p  = j * width + i;
            out[p] = inp[p] * c3[c.min(2)];
        }
    }
}

/// Non-mosaiced (RGB/RGBA) white-balance multiply.
///
/// Replaces the DT_OMP_FOR loop in the `else` branch of temperature.c::process().
/// coeffs[0..4] = d->coeffs — one scalar multiplier per RGBA channel.
#[no_mangle]
pub unsafe extern "C" fn darkroom_temperature_process_rgb(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    coeffs: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let c = std::slice::from_raw_parts(coeffs, 4);
    for k in 0..npixels {
        output[k * 4]     = input[k * 4]     * c[0];
        output[k * 4 + 1] = input[k * 4 + 1] * c[1];
        output[k * 4 + 2] = input[k * 4 + 2] * c[2];
        output[k * 4 + 3] = input[k * 4 + 3] * c[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_multiply_scales_all_channels() {
        let input = vec![0.5f32, 0.25, 0.125, 1.0,
                         1.0f32, 0.0,  0.5,   0.5];
        let mut out = vec![0.0f32; 8];
        let coeffs = [2.0f32, 4.0, 8.0, 1.0];
        unsafe {
            darkroom_temperature_process_rgb(
                input.as_ptr(), out.as_mut_ptr(), 2, coeffs.as_ptr()
            );
        }
        // pixel 0
        assert!((out[0] - 1.0).abs() < 1e-6);  // 0.5 * 2
        assert!((out[1] - 1.0).abs() < 1e-6);  // 0.25 * 4
        assert!((out[2] - 1.0).abs() < 1e-6);  // 0.125 * 8
        assert!((out[3] - 1.0).abs() < 1e-6);  // 1.0 * 1
        // pixel 1
        assert!((out[4] - 2.0).abs() < 1e-6);  // 1.0 * 2
        assert!((out[5] - 0.0).abs() < 1e-6);  // 0.0 * 4
        assert!((out[6] - 4.0).abs() < 1e-6);  // 0.5 * 8
        assert!((out[7] - 0.5).abs() < 1e-6);  // 0.5 * 1
    }

    #[test]
    fn unity_coefficients_are_passthrough() {
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0f32; 8];
        let coeffs = [1.0f32; 4];
        unsafe {
            darkroom_temperature_process_rgb(
                input.as_ptr(), out.as_mut_ptr(), 2, coeffs.as_ptr()
            );
        }
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn bayer_wb_scales_each_channel() {
        // RGGB: FC(0,0)=0=R, FC(0,1)=1=G, FC(1,0)=1=G, FC(1,1)=2=B
        // coeffs: [R_scale, G_scale, B_scale, unused]
        let inp = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 4];
        let coeffs = [2.0_f32, 1.5, 0.8, 0.0];
        unsafe { darkroom_temperature_bayer(inp.as_ptr(), out.as_mut_ptr(), 2, 2, 0x94949494u32, coeffs.as_ptr()); }
        assert!((out[0] - 2.0).abs() < 1e-5); // R: 1.0 * 2.0
        assert!((out[1] - 1.5).abs() < 1e-5); // G: 1.0 * 1.5
        assert!((out[2] - 1.5).abs() < 1e-5); // G: 1.0 * 1.5
        assert!((out[3] - 0.8).abs() < 1e-5); // B: 1.0 * 0.8
    }

    #[test]
    fn bayer_wb_zero_coeffs_produces_zeros() {
        let inp = vec![1.0_f32; 4];
        let mut out = vec![-1.0_f32; 4];
        let coeffs = [0.0_f32; 4];
        unsafe { darkroom_temperature_bayer(inp.as_ptr(), out.as_mut_ptr(), 2, 2, 0x94949494u32, coeffs.as_ptr()); }
        for v in &out { assert_eq!(*v, 0.0); }
    }
}
