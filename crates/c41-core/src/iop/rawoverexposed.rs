//! Raw-overexposure indicator (src/iop/rawoverexposed.c).
//!
//! The C process() walks output rows, back-transforms each row's pixel
//! coordinates through the pipe's distortions (a C pipeline callback that
//! stays in C), then marks output pixels whose originating raw photosite is
//! clipped. The DT_OMP_FOR row loops were replaced by serial C row loops
//! calling the two functions below around the back-transform.

use crate::raw;

/// dt_iop_rawoverexposed_colors (rawoverexposed.c:44): the per-CFA-colour
/// overlay colours (red, green, blue, black RGBA).
const CFA_COLORS: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
];

// dt_dev_rawoverexposed_mode_t (develop.h:69)
const MODE_MARK_CFA: i32 = 0;
const MODE_MARK_SOLID: i32 = 1;
const MODE_FALSECOLOR: i32 = 2;

/// Fill one row's pre-distortion pixel coordinates: for every output column
/// `i`, buf[2i] = (x + i) / scale and buf[2i+1] = (y + row) / scale.
/// Replaces the coordinate-fill inner loop shared by both DT_OMP_FOR row
/// loops in rawoverexposed.c (CPU process and the OpenCL coord prepass).
///
/// # Safety
/// `buf` holds `2 * width` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawoverexposed_fill_coords(
    buf: *mut f32, row: i32, width: usize, x: i32, y: i32, scale: f32,
) {
    let b = std::slice::from_raw_parts_mut(buf, 2 * width);
    for i in 0..width {
        b[2 * i] = (x + i as i32) as f32 / scale;
        b[2 * i + 1] = (y + row) as f32 / scale;
    }
}

/// Mark one output row from the back-transformed coordinates: look up each
/// pixel's originating raw photosite, and when its value reaches the
/// per-colour clipping threshold paint the output according to `mode`
/// (CFA colour, solid colour, or zeroing the clipped channel). Replaces the
/// marking inner loop of the CPU DT_OMP_FOR in rawoverexposed.c:158.
///
/// Like the C, the float→int coordinate conversion truncates toward zero;
/// Rust saturates ±inf/out-of-range to the i32 extremes (rejected by the
/// bounds test like in C) and maps NaN — never produced by the
/// back-transform on the finite inputs here, and UB in the original C —
/// to 0.
///
/// # Safety
/// `out_row` holds `ch * width` floats; `coords` holds `2 * width` floats;
/// `raw_buf` holds `raw_width * raw_height` u16; `thresholds` 4 uints;
/// `solid_color` 4 floats; `xtrans` 36 bytes (read only when filters == 9).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_rawoverexposed_mark_row(
    out_row: *mut f32, width: usize, ch: usize,
    coords: *const f32,
    raw_buf: *const u16, raw_width: usize, raw_height: usize,
    filters: u32, xtrans: *const u8,
    thresholds: *const u32,
    mode: i32, solid_color: *const f32,
) {
    let out = std::slice::from_raw_parts_mut(out_row, ch * width);
    let c2 = std::slice::from_raw_parts(coords, 2 * width);
    let raw_data = std::slice::from_raw_parts(raw_buf, raw_width * raw_height);
    let thr = std::slice::from_raw_parts(thresholds, 4);
    let color = std::slice::from_raw_parts(solid_color, 4);
    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = xt_bytes[r * 6 + c]; }
    }

    for i in 0..width {
        let pout = ch * i;

        // not sure which float -> int to use here (sic, rawoverexposed.c)
        let i_raw = c2[2 * i] as i32;
        let j_raw = c2[2 * i + 1] as i32;

        if i_raw < 0 || j_raw < 0 || i_raw >= raw_width as i32 || j_raw >= raw_height as i32 {
            continue;
        }

        let c = raw::fcol(j_raw, i_raw, filters, &xt);

        let pin = j_raw as usize * raw_width + i_raw as usize;
        let inval = raw_data[pin] as f32;

        // was the raw pixel clipped?
        if inval < thr[c] as f32 {
            continue;
        }

        match mode {
            MODE_MARK_CFA => out[pout..pout + 4].copy_from_slice(&CFA_COLORS[c]),
            MODE_MARK_SOLID => out[pout..pout + 4].copy_from_slice(&color[0..4]),
            MODE_FALSECOLOR => out[pout + c] = 0.0,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGGB: u32 = 0x94949494;
    const XT0: [u8; 36] = [0; 36];

    #[test]
    fn fill_coords_scales_and_offsets() {
        let mut buf = vec![0.0_f32; 8];
        unsafe { darkroom_rawoverexposed_fill_coords(buf.as_mut_ptr(), 3, 4, 10, 20, 0.5); }
        assert_eq!(buf[0], 20.0); // (10+0)/0.5
        assert_eq!(buf[1], 46.0); // (20+3)/0.5
        assert_eq!(buf[6], 26.0); // (10+3)/0.5
        assert_eq!(buf[7], 46.0);
    }

    fn mark(mode: i32, thr: [u32; 4]) -> Vec<f32> {
        // 4x4 RGGB raw; photosite (1,1)=B clipped at 1000
        let (rw, rh) = (4usize, 4usize);
        let mut raw_buf = vec![100_u16; rw * rh];
        raw_buf[rw + 1] = 1000;
        let width = 2usize;
        let ch = 4usize;
        // identity-ish coords: out col 0 → raw (1,1) [B, clipped], col 1 → raw (0,0) [R, not]
        let coords = [1.2_f32, 1.7, 0.0, 0.0];
        let mut out = vec![0.5_f32; ch * width];
        let solid = [0.9_f32, 0.8, 0.7, 1.0];
        unsafe {
            darkroom_rawoverexposed_mark_row(
                out.as_mut_ptr(), width, ch, coords.as_ptr(),
                raw_buf.as_ptr(), rw, rh, RGGB, XT0.as_ptr(),
                thr.as_ptr(), mode, solid.as_ptr(),
            );
        }
        out
    }

    #[test]
    fn mark_cfa_paints_cfa_color_of_clipped_site() {
        let out = mark(MODE_MARK_CFA, [500, 500, 500, 500]);
        assert_eq!(&out[0..4], &[0.0, 0.0, 1.0, 1.0]); // blue overlay
        assert_eq!(&out[4..8], &[0.5, 0.5, 0.5, 0.5]); // unclipped → untouched
    }

    #[test]
    fn mark_solid_paints_chosen_color() {
        let out = mark(MODE_MARK_SOLID, [500, 500, 500, 500]);
        assert_eq!(&out[0..4], &[0.9, 0.8, 0.7, 1.0]);
    }

    #[test]
    fn falsecolor_zeroes_clipped_channel_only() {
        let out = mark(MODE_FALSECOLOR, [500, 500, 500, 500]);
        assert_eq!(&out[0..4], &[0.5, 0.5, 0.0, 0.5]); // B channel zeroed
    }

    #[test]
    fn threshold_below_keeps_pixel() {
        let out = mark(MODE_MARK_CFA, [2000, 2000, 2000, 2000]);
        assert_eq!(&out[0..4], &[0.5, 0.5, 0.5, 0.5]); // 1000 < 2000 → untouched
    }

    #[test]
    fn out_of_bounds_coords_skipped() {
        let coords = [-0.5_f32, 1.0, 9.0, 1.0]; // col0: i_raw=0 ok? (-0.5 as i32 = 0)… and col1 OOB
        // C: (int)-0.5 == 0 → NOT skipped (i_raw 0 >= 0); replicate and verify
        let (rw, rh) = (4usize, 4usize);
        let raw_buf = vec![2000_u16; rw * rh]; // everything clipped
        let mut out = vec![0.5_f32; 8];
        let solid = [0.9_f32, 0.8, 0.7, 1.0];
        unsafe {
            darkroom_rawoverexposed_mark_row(
                out.as_mut_ptr(), 2, 4, coords.as_ptr(),
                raw_buf.as_ptr(), rw, rh, RGGB, XT0.as_ptr(),
                [500_u32; 4].as_ptr(), MODE_MARK_SOLID, solid.as_ptr(),
            );
        }
        assert_eq!(&out[0..4], &[0.9, 0.8, 0.7, 1.0]); // truncated to (0,1) → marked
        assert_eq!(&out[4..8], &[0.5, 0.5, 0.5, 0.5]); // i_raw=9 OOB → skipped
    }
}
