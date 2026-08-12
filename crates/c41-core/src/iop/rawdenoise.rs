use crate::{params::IopParams, raw, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Rawdenoise;

impl IopProcess for Rawdenoise {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "rawdenoise" }
}

/// vstransform: variance-stabilising forward transform. Mirrors the C inline.
#[inline(always)]
fn vstransform(v: f32) -> f32 { v.max(0.0).sqrt() }

/// Collect one Bayer channel (c in 0..4) into a half-size monochrome buffer
/// applying the variance-stabilising sqrt() transform.
///
/// Matches the first DT_OMP_FOR in `wavelet_denoise()` (rawdenoise.c:221).
///
/// `c` selects the Bayer cell: 0=R, 1=G1, 2=G2, 3=B (same encoding as the
/// C loop `c in 0..nc`). The output buffer must have size halfwidth * halfheight
/// where:
///   halfwidth  = width / 2 + (width  & (!(c >> 1)) & 1)
///   halfheight = height / 2 + (height & (!c) & 1)
///
/// These dimensions must be computed by the caller and passed here; the
/// function trusts `fimg_buf` is large enough.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawdenoise_bayer_collect(
    in_buf: *const f32,
    fimg_buf: *mut f32,
    width: usize,
    height: usize,
    halfwidth: usize,
    c: u32,
) {
    if width == 0 || height == 0 || halfwidth == 0 { return; }
    let c = c as usize;
    let offset = (c & 2) >> 1;
    let inp = std::slice::from_raw_parts(in_buf, width * height);
    // halfheight is implicit; the row loop strides by 2 so the number of
    // iterations matches halfheight exactly.
    // fimg size = halfwidth * halfheight — the caller already allocated it.
    let fimg_len = halfwidth * ((height / 2) + 1).max(height / 2 + (height & ((!c) & 1)));
    let fimg = std::slice::from_raw_parts_mut(fimg_buf, fimg_len);

    // senselwidth must equal halfwidth for the fimg index to stay in-bounds.
    // This is guaranteed by the C caller (halfwidth = (width - offset + 1) / 2),
    // but we assert here so a mis-wired call fails loudly rather than silently
    // reading/writing out-of-range memory.
    let senselwidth = (width - offset + 1) / 2;
    debug_assert_eq!(
        halfwidth, senselwidth,
        "halfwidth {halfwidth} != senselwidth {senselwidth}: caller must pass (width-offset+1)/2"
    );

    let start_row = c & 1;
    let mut half_row = 0usize;
    let mut row = start_row;
    while row < height {
        let row_start = row * width;
        let fimg_row_start = half_row * halfwidth;
        for col in 0..senselwidth {
            fimg[fimg_row_start + col] = vstransform(inp[row_start + offset + 2 * col]);
        }
        row += 2;
        half_row += 1;
    }
}

/// Scatter a denoised Bayer channel back to the output buffer, squaring
/// to invert the variance-stabilising transform.
///
/// Matches the second DT_OMP_FOR in `wavelet_denoise()` (rawdenoise.c:237).
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawdenoise_bayer_scatter(
    fimg_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    halfwidth: usize,
    c: u32,
) {
    if width == 0 || height == 0 || halfwidth == 0 { return; }
    let c = c as usize;
    let offset = (c & 2) >> 1;
    let fimg_len = halfwidth * ((height / 2) + 1).max(height / 2 + (height & ((!c) & 1)));
    let fimg = std::slice::from_raw_parts(fimg_buf, fimg_len);
    let out  = std::slice::from_raw_parts_mut(out_buf, width * height);

    let senselwidth = (width - offset + 1) / 2;
    debug_assert_eq!(
        halfwidth, senselwidth,
        "halfwidth {halfwidth} != senselwidth {senselwidth}: caller must pass (width-offset+1)/2"
    );

    let start_row = c & 1;
    let mut half_row = 0usize;
    let mut row = start_row;
    while row < height {
        let row_start = row * width;
        let fimg_row_start = half_row * halfwidth;
        for col in 0..senselwidth {
            let d = fimg[fimg_row_start + col];
            out[row_start + offset + 2 * col] = d * d;
        }
        row += 2;
        half_row += 1;
    }
}

/// Collect one X-Trans colour channel (c in 0..3: R=0, G=1, B=2) into a
/// full-size monochrome buffer with nearest-neighbour interpolation.
///
/// The C source wraps this in an OMP loop divided into `nthreads` chunks.
/// The *chunk-restoration* block (C:~416–446) runs only when
/// `pastend < height`, i.e., when a following chunk exists and its
/// first-row values may have been clobbered by this chunk's propagation.
/// In the single-threaded Rust port the entire image is one chunk so
/// `pastend == height` always holds and the restoration block is omitted.
///
/// Border safety: The C allocates `fimg = img + width` (one guard row
/// prepended) so red/blue pixel propagation from `row=1` into `fimgp[-width]`
/// writes to the throw-away guard row and never touches fimg's row 0.  The
/// Rust port has no guard row, so above-row writes from `row=1` are
/// suppressed via `if row > 1` guards to avoid clobbering the
/// caller-initialised 0.5 border row.
///
/// Matches the DT_OMP_FOR(num_threads(nthreads)) in
/// `wavelet_denoise_xtrans()` (rawdenoise.c:339).
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawdenoise_xtrans_collect(
    in_buf: *const f32,
    fimg_buf: *mut f32,
    width: usize,
    height: usize,
    xtrans: *const u8, // flat 36-byte 6x6
    c: u32,
) {
    if width == 0 || height == 0 { return; }
    let c = c as usize;
    let inp  = std::slice::from_raw_parts(in_buf,  width * height);
    let fimg = std::slice::from_raw_parts_mut(fimg_buf, width * height);

    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 { for col in 0..6 { xt[r][col] = xt_bytes[r * 6 + col]; } }

    // Top and bottom border rows are preset to 0.5 by the caller before
    // this function; only the interior [1..height-1) is written here.
    // (The C sets them in-line before starting the OMP section.)

    // Single chunk: start=0 (skipping row 0), pastend = height (no restoration needed).
    for row in 1..(height - 1) {
        let row_off = row * width;
        // left-boundary: handle col=0
        if c != 1 && raw::fc_xtrans(row as i32, 0, &xt) == c {
            let d = vstransform(inp[row_off]);
            fimg[row_off] = d;
            // Guard: the C writes to fimgp[-width] (a throw-away guard row
            // prepended to fimg). The Rust has no guard row, so above-row
            // writes at row=1 would clobber the caller-initialised border.
            if row > 1 {
                fimg[(row - 1) * width]     = d; // above
                fimg[(row - 1) * width + 1] = d; // above-right
            }
        }

        // main inner loop (col 1..width-2 for non-green, 0.. for green)
        let col_start = if c != 1 { 1 } else { 0 };
        for col in col_start..(width - 1) {
            if raw::fc_xtrans(row as i32, col as i32, &xt) == c {
                let d = vstransform(inp[row_off + col]);
                fimg[row_off + col] = d;
                if c == 1 {
                    // green: copy right and down
                    fimg[row_off + col + 1]         = d;
                    fimg[(row + 1) * width + col]   = d;
                } else {
                    // red/blue: copy to 8 neighbours.
                    // Above-row writes are suppressed at row=1 to avoid
                    // clobbering the caller-initialised border (no guard row).
                    if row > 1 {
                        let above = (row - 1) * width;
                        if col > 0 { fimg[above + col - 1] = d; }
                        fimg[above + col]     = d;
                        fimg[above + col + 1] = d;
                    }
                    // left and right
                    if col > 0 { fimg[row_off + col - 1] = d; }
                    fimg[row_off + col + 1] = d;
                    // row below (only when not the last chunk row — always safe
                    // in a single chunk since row < height - 1)
                    let below = (row + 1) * width;
                    if col > 0 { fimg[below + col - 1] = d; }
                    fimg[below + col]     = d;
                    fimg[below + col + 1] = d;
                }
            }
        }

        // Fill leftmost pixel if it wasn't set by the left-boundary block
        if raw::fc_xtrans(row as i32, 0, &xt) != c {
            let src_off = if row > 1 && raw::fc_xtrans((row - 1) as i32, 0, &xt) == c {
                row_off - width // above
            } else if raw::fc_xtrans(row as i32, 1, &xt) == c {
                row_off + 1 // right
            } else if row > 1 && raw::fc_xtrans((row - 1) as i32, 1, &xt) == c {
                row_off - width + 1 // above-right
            } else {
                row_off // fallback: current even if wrong colour
            };
            fimg[row_off] = vstransform(inp[src_off]);
        }

        // right-boundary handling
        let last = width - 1;
        if c != 1 && raw::fc_xtrans(row as i32, last as i32, &xt) == c {
            let d = vstransform(inp[row_off + last]);
            fimg[row_off + last - 1] = d;
            fimg[row_off + last]     = d;
            if row > 0 { fimg[(row - 1) * width + last - 1] = d; } // above-left (fimg[-1] in C)
        } else if raw::fc_xtrans(row as i32, last as i32, &xt) != c {
            let src = if raw::fc_xtrans(row as i32, (last - 1) as i32, &xt) == c {
                row_off + last - 1
            } else if row > 1 && raw::fc_xtrans((row - 1) as i32, last as i32, &xt) == c {
                (row - 1) * width + last
            } else if row > 1 && raw::fc_xtrans((row - 1) as i32, (last - 1) as i32, &xt) == c {
                (row - 1) * width + last - 1
            } else {
                row_off + last // fallback
            };
            fimg[row_off + last] = vstransform(inp[src]);
        }
    }
}

/// Scatter a denoised X-Trans channel back, squaring to invert vstransform.
///
/// Matches the DT_OMP_FOR in `wavelet_denoise_xtrans()` (rawdenoise.c:454).
#[no_mangle]
pub unsafe extern "C" fn darkroom_rawdenoise_xtrans_scatter(
    fimg_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    xtrans: *const u8,
    c: u32,
) {
    if width == 0 || height == 0 { return; }
    let c = c as usize;
    let fimg = std::slice::from_raw_parts(fimg_buf, width * height);
    let out  = std::slice::from_raw_parts_mut(out_buf, width * height);

    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 { for col in 0..6 { xt[r][col] = xt_bytes[r * 6 + col]; } }

    for row in 0..height {
        for col in 0..width {
            if raw::fc_xtrans(row as i32, col as i32, &xt) == c {
                let d = fimg[row * width + col];
                out[row * width + col] = d * d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical Fujifilm X-Trans CFA pattern for tests.
    const XTRANS: [[u8; 6]; 6] = [
        [1, 2, 1, 1, 0, 1],
        [0, 1, 0, 2, 1, 2],
        [1, 2, 1, 1, 0, 1],
        [1, 0, 1, 1, 2, 1],
        [2, 1, 2, 0, 1, 0],
        [1, 0, 1, 1, 2, 1],
    ];

    fn xt_flat() -> Vec<u8> { XTRANS.iter().flatten().copied().collect() }

    #[test]
    fn bayer_collect_applies_vstransform() {
        // 4x4 image; c=0 (R at row 0, col 0). offset=0, stride=2.
        // Row 0, halfwidth = 4/2 = 2.
        let inp = vec![4.0_f32, 9.0, 0.0, 16.0,   // row 0 — stride 2 → R=4, R=0
                       0.0,  0.0, 0.0,  0.0,
                       0.0,  0.0, 0.0,  0.0,
                       0.0,  0.0, 0.0,  0.0];
        let halfwidth = 2;
        let halfheight_buf = 4; // oversized — safe
        let mut fimg = vec![0.0_f32; halfwidth * halfheight_buf];
        unsafe {
            darkroom_rawdenoise_bayer_collect(
                inp.as_ptr(), fimg.as_mut_ptr(), 4, 4, halfwidth, 0,
            );
        }
        // fimg[0] = sqrt(4) = 2, fimg[1] = sqrt(0) = 0
        assert!((fimg[0] - 2.0).abs() < 1e-6);
        assert_eq!(fimg[1], 0.0);
    }

    #[test]
    fn bayer_scatter_squares_and_writes_back() {
        let mut fimg = vec![2.0_f32, 3.0]; // halfwidth=2
        let mut out  = vec![0.0_f32; 4 * 4];
        unsafe {
            darkroom_rawdenoise_bayer_scatter(
                fimg.as_ptr(), out.as_mut_ptr(), 4, 4, 2, 0,
            );
        }
        // c=0: offset=0, row 0 → out[0]=2^2=4, out[2]=3^2=9
        assert_eq!(out[0], 4.0);
        assert_eq!(out[2], 9.0);
        // Make fimg mutable to suppress the warning; not strictly needed here.
        fimg[0] = 0.0;
    }

    #[test]
    fn bayer_round_trip_identity() {
        // collect then scatter with no denoising should be identity
        let mut inp = vec![0.0_f32; 4 * 4];
        inp[0] = 0.25; inp[2] = 0.64; // c=0 row 0 pixels
        let halfwidth = 2;
        let mut fimg = vec![0.0_f32; halfwidth * 4];
        let mut out  = vec![0.0_f32; 4 * 4];
        unsafe {
            darkroom_rawdenoise_bayer_collect(inp.as_ptr(), fimg.as_mut_ptr(), 4, 4, halfwidth, 0);
            darkroom_rawdenoise_bayer_scatter(fimg.as_ptr(), out.as_mut_ptr(), 4, 4, halfwidth, 0);
        }
        assert!((out[0] - inp[0]).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[2] - inp[2]).abs() < 1e-5, "out[2]={}", out[2]);
    }

    #[test]
    fn xtrans_scatter_squares_matching_channel_only() {
        let w = 6; let h = 6;
        // fimg = uniform 2.0; scatter c=0 (R) → out[k] = 4.0 only where CFA=R
        let fimg = vec![2.0_f32; w * h];
        let mut out  = vec![0.0_f32; w * h];
        let xt_flat = xt_flat();
        unsafe {
            darkroom_rawdenoise_xtrans_scatter(
                fimg.as_ptr(), out.as_mut_ptr(), w, h, xt_flat.as_ptr(), 0,
            );
        }
        for row in 0..h {
            for col in 0..w {
                let c = raw::fc_xtrans(row as i32, col as i32, &XTRANS);
                let expected = if c == 0 { 4.0_f32 } else { 0.0_f32 };
                assert_eq!(out[row * w + col], expected,
                    "({row},{col}) CFA={c} out={}", out[row * w + col]);
            }
        }
    }

    #[test]
    fn xtrans_scatter_does_not_touch_wrong_channels() {
        let w = 6; let h = 6;
        let fimg = vec![3.0_f32; w * h];
        let mut out = vec![99.0_f32; w * h]; // sentinel
        let xt_flat = xt_flat();
        // scatter c=1 (G); only G pixels should be overwritten
        unsafe {
            darkroom_rawdenoise_xtrans_scatter(
                fimg.as_ptr(), out.as_mut_ptr(), w, h, xt_flat.as_ptr(), 1,
            );
        }
        for row in 0..h {
            for col in 0..w {
                let c = raw::fc_xtrans(row as i32, col as i32, &XTRANS);
                if c == 1 {
                    assert_eq!(out[row * w + col], 9.0);
                } else {
                    assert_eq!(out[row * w + col], 99.0);
                }
            }
        }
    }
}
