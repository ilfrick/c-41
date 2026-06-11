use crate::{params::IopParams, raw, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Highlights;

impl IopProcess for Highlights {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "highlights" }
}

/// Build the per-pixel highlight-clipping raster mask for an sRAW (already-RGB)
/// input. For every pixel:
///
///   ref_c = max(0.5, 0.95 * clips[c])
///   mval  = max over c of (in[i + c] - ref_c) / ref_c
///   tmp[k] = max(0.0, mval)
///
/// Matches the `filters == 0` branch of `_provide_raster_mask()` in
/// src/iop/highlights.c. `in_buf` is an RGBA float buffer, `tmp_buf` is a
/// single-plane float mask of size `width * height`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_mask_sraw(
    in_buf: *const f32,
    tmp_buf: *mut f32,
    width: usize,
    height: usize,
    clips: *const f32,
) {
    let n = width * height;
    let input = std::slice::from_raw_parts(in_buf, n * 4);
    let tmp = std::slice::from_raw_parts_mut(tmp_buf, n);
    let clips = std::slice::from_raw_parts(clips, 4);

    // Precompute the per-channel reference levels (max(0.5, 0.95*clip[c])).
    let mut refs = [0.0_f32; 3];
    for c in 0..3 {
        refs[c] = (0.95 * clips[c]).max(0.5);
    }

    for ox in 0..n {
        let ix = ox * 4;
        let mut mval = 0.0_f32;
        for c in 0..3 {
            let v = (input[ix + c] - refs[c]) / refs[c];
            if v > mval { mval = v; }
        }
        tmp[ox] = mval.max(0.0);
    }
}

/// Build the per-pixel highlight-clipping raster mask for a mosaic
/// (Bayer / X-Trans) input. For every pixel, look up its CFA colour via
/// `fcol(irow, icol, filters, xtrans)`, where `irow = row + roi_y` and
/// `icol = col + roi_x`, then apply the same formula as the sRAW path
/// using the per-colour reference.
///
/// Matches the `filters != 0` branch of `_provide_raster_mask()` in
/// src/iop/highlights.c. `in_buf` is a single-plane raw float buffer of
/// size `width * height`; the xtrans pattern is read only when `filters == 9`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_mask_mosaic(
    in_buf: *const f32,
    tmp_buf: *mut f32,
    width: usize,
    height: usize,
    filters: u32,
    xtrans: *const u8, // 6*6 = 36 bytes
    clips: *const f32,
    irow_offset: i32,
    icol_offset: i32,
) {
    let n = width * height;
    let input = std::slice::from_raw_parts(in_buf, n);
    let tmp = std::slice::from_raw_parts_mut(tmp_buf, n);
    let clips = std::slice::from_raw_parts(clips, 4);

    // Reconstruct the 6x6 xtrans table from the raw byte pointer.
    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = xt_bytes[r * 6 + c]; }
    }

    let mut refs = [0.0_f32; 4];
    for c in 0..4 {
        refs[c] = (0.95 * clips[c]).max(0.5);
    }

    for row in 0..height {
        for col in 0..width {
            let ox = row * width + col;
            let irow = row as i32 + irow_offset;
            let icol = col as i32 + icol_offset;
            let c = raw::fcol(irow, icol, filters, &xt);
            let r = refs[c];
            tmp[ox] = ((input[ox] - r) / r).max(0.0);
        }
    }
}

/// CLIP mode for the sRAW path: simple per-component clamp to `clip`.
///
///   out[k] = min(clip, in[k])  for every float in the buffer.
///
/// Matches the `ch == 4` branch of process_clip() in src/iop/highlights.c.
/// `nfloats` is the total number of floats (npixels * 4 for RGBA).
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_clip_sraw(
    in_buf: *const f32,
    out_buf: *mut f32,
    nfloats: usize,
    clip: f32,
) {
    if nfloats == 0 { return; }
    let input = std::slice::from_raw_parts(in_buf, nfloats);
    let output = std::slice::from_raw_parts_mut(out_buf, nfloats);
    for k in 0..nfloats {
        let v = input[k];
        // Match C `fminf(clip, v)` IEEE-754 Annex F NaN semantics: if exactly
        // one operand is NaN, return the non-NaN one; if both are NaN, return
        // NaN. Rust's `f32::min` gets this right when the receiver is NaN but
        // diverges for the case `clip.is_nan() && !v.is_nan()` (it returns
        // clip, fminf returns v). Explicit decomposition keeps us bit-for-bit
        // identical to the C path.
        output[k] = if v.is_nan() { clip }
                    else if clip.is_nan() { v }
                    else { v.min(clip) };
    }
}

/// Visualise clipping on a sRAW (RGBA) buffer.
///
/// For every pixel k, c in 0..3:
///   out[k+c] = (in[k+c] < clips[c]) ? 0.2 * in[k+c] : 1.0
///   out[k+3] = 0.0
///
/// Matches the `filters == 0` branch of process_visualize() in
/// src/iop/highlights.c.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_visualize_sraw(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    clips: *const f32, // 4 floats
) {
    if npixels == 0 { return; }
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let clips = std::slice::from_raw_parts(clips, 4);

    for k in 0..npixels {
        let j = k * 4;
        // The C source uses `for_each_channel(c)` (which iterates 0..3 on this
        // platform — `DT_PIXEL_SIMD_CHANNELS = 4`) and then overrides
        // `out[k+3] = 0.0f`. Iterating 0..3 explicitly here and writing 0.0 to
        // index 3 yields the same final values; if `for_each_channel` ever
        // grows a 5-channel variant, this loop must be revisited.
        for c in 0..3 {
            let v = input[j + c];
            output[j + c] = if v < clips[c] { 0.2 * v } else { 1.0 };
        }
        output[j + 3] = 0.0;
    }
}

/// Visualise clipping on a single-plane mosaic (Bayer / X-Trans) buffer.
///
/// For every output pixel (row, col):
///   irow = row + irow_offset   // = roi_out.y - roi_in.y
///   icol = col + icol_offset   // = roi_out.x - roi_in.x
///   if (irow, icol) is in [0, input_height) x [0, input_width):
///     c = fcol(irow, icol, filters, xtrans)
///     v = in[irow * input_width + icol]
///     out[k] = (v < clips[c]) ? 0.2 * v : 1.0
///   else:
///     out[k] = 0.0
///
/// `xtrans` is a flat 36-byte 6x6 pattern; read only when filters==9.
/// Matches the `filters != 0` branch of process_visualize() in
/// src/iop/highlights.c.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_visualize_mosaic(
    in_buf: *const f32,
    out_buf: *mut f32,
    out_width: usize,
    out_height: usize,
    in_width: usize,
    in_height: usize,
    filters: u32,
    xtrans: *const u8,
    clips: *const f32, // 4 floats
    irow_offset: i32,
    icol_offset: i32,
) {
    if out_width == 0 || out_height == 0 { return; }
    let in_total = in_width.saturating_mul(in_height);
    if in_total == 0 { return; }

    let input = std::slice::from_raw_parts(in_buf, in_total);
    let output = std::slice::from_raw_parts_mut(out_buf, out_width * out_height);
    let clips = std::slice::from_raw_parts(clips, 4);

    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = xt_bytes[r * 6 + c]; }
    }

    // Width/height arrive as `usize`; the bounds check has to compare against
    // signed i32 because the irow/icol can go negative under non-trivial
    // roi offsets. Assert the dimensions fit so the cast is lossless — a
    // silent wrap would make the bounds check trivially false and zero the
    // entire output without warning.
    let in_w_i = i32::try_from(in_width).expect("in_width exceeds i32::MAX");
    let in_h_i = i32::try_from(in_height).expect("in_height exceeds i32::MAX");

    for row in 0..out_height {
        for col in 0..out_width {
            let ox = row * out_width + col;
            let irow = row as i32 + irow_offset;
            let icol = col as i32 + icol_offset;
            if icol >= 0 && irow >= 0 && irow < in_h_i && icol < in_w_i {
                let ix = (irow as usize) * in_width + (icol as usize);
                let c = crate::raw::fcol(irow, icol, filters, &xt);
                let v = input[ix];
                output[ox] = if v < clips[c] { 0.2 * v } else { 1.0 };
            } else {
                output[ox] = 0.0;
            }
        }
    }
}

// ── LCH highlight reconstruction (src/iop/hlreconstruct/lch.c) ───────────────

// sqrt(3) and 2*sqrt(3); the C macros are long-double literals — rounded here
// to the nearest f32 (sub-ulp differences are invisible in reconstruction).
const SQRT3: f32 = 1.732_050_8;
const SQRT12: f32 = 3.464_101_6; // 2*SQRT3

/// LCH backtransform: rebuild RGB from luminance L and rescaled C/H.
/// `(C,H)` is scaled by `sqrt((Co²+Ho²)/(C²+H²))` when well-defined, pulling
/// the unclipped chroma/hue magnitude onto the full-range L.
#[inline(always)]
fn lch_backtransform(l: f32, mut c: f32, mut h: f32, co: f32, ho: f32, distinct: bool) -> [f32; 3] {
    if distinct {
        let ratio = ((co * co + ho * ho) / (c * c + h * h)).sqrt();
        c *= ratio;
        h *= ratio;
    }
    // R = L - H/6 + C/sqrt(12); G = L - H/6 - C/sqrt(12); B = L + H/3
    [l - h / 6.0 + c / SQRT12, l - h / 6.0 - c / SQRT12, l + h / 3.0]
}

/// LCH highlight reconstruction for Bayer sensors.
/// Replaces the DT_OMP_FOR(collapse(2)) loop in process_lch_bayer()
/// (src/iop/hlreconstruct/lch.c:41). `in`/`out` are single-channel planes of
/// `width * height` floats (the C indexes both with roi_out->width).
///
/// # Safety
/// `in_buf` and `out_buf` hold `width * height` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_lch_bayer(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    filters: u32,
    clip: f32,
) {
    if width == 0 || height == 0 { return; }
    let n = width * height;
    let input = std::slice::from_raw_parts(in_buf, n);
    let output = std::slice::from_raw_parts_mut(out_buf, n);

    for j in 0..height {
        for i in 0..width {
            let idx = j * width + i;
            if i == width - 1 || j == height - 1 {
                // fast path for border
                output[idx] = clip.min(input[idx]);
                continue;
            }

            // sample 1 bayer block (2x2), giving 2 green values
            let mut clipped = false;
            let (mut r, mut gmin, mut gmax, mut b) = (0.0_f32, f32::MAX, f32::MIN, 0.0_f32);
            for jj in 0..=1usize {
                for ii in 0..=1usize {
                    let val = input[idx + jj * width + ii];
                    clipped = clipped || val > clip;
                    match raw::fc_bayer((j + jj) as i32, (i + ii) as i32, filters) {
                        0 => r = val,
                        1 => { gmin = gmin.min(val); gmax = gmax.max(val); }
                        2 => b = val,
                        _ => {}
                    }
                }
            }

            if clipped {
                let ro = r.min(clip);
                let go = gmin.min(clip);
                let bo = b.min(clip);

                let l = (r + gmax + b) / 3.0;
                let c = SQRT3 * (r - gmax);
                let h = 2.0 * b - gmax - r;
                let co = SQRT3 * (ro - go);
                let ho = 2.0 * bo - go - ro;

                let rgb = lch_backtransform(l, c, h, co, ho, r != gmax && gmax != b);
                output[idx] = rgb[raw::fc_bayer(j as i32, i as i32, filters)];
            } else {
                output[idx] = input[idx];
            }
        }
    }
}

/// LCH highlight reconstruction for X-Trans sensors.
/// Replaces the DT_OMP_FOR loop in process_lch_xtrans()
/// (src/iop/hlreconstruct/lch.c:142). `out` is a `width_out * height_out`
/// plane; `in` rows are strided by `width_in` (roi_in->width >= width_out).
///
/// # Safety
/// `out_buf` holds `width_out * height_out` floats; `in_buf` holds at least
/// `height_out` rows of `width_in` floats (rows j-2..j+2 are sampled inside
/// the border guard, and j±1 by the clipping ring buffer).
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_lch_xtrans(
    in_buf: *const f32,
    out_buf: *mut f32,
    width_out: usize,
    height_out: usize,
    width_in: usize,
    xtrans: *const u8, // 6*6 = 36 bytes
    clip: f32,
) {
    if width_out == 0 || height_out == 0 { return; }
    let input = std::slice::from_raw_parts(in_buf, height_out * width_in);
    let output = std::slice::from_raw_parts_mut(out_buf, height_out * width_out);

    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = xt_bytes[r * 6 + c]; }
    }

    let riw = width_in as isize;
    // int-style bounds (C uses int arithmetic; usize would underflow when the
    // ROI is narrower than 3 pixels)
    let (w, h) = (width_out as isize, height_out as isize);
    for j in 0..height_out {
        let ji = j as isize;
        // bit vector used as ring buffer to remember clipping of current and
        // last two columns, checking current pixel and its vertical neighbours
        let mut cl: u32 = 0;
        for i in 0..width_out {
            let ii_ = i as isize;
            let base = (j * width_in + i) as isize;
            let at = |off: isize| -> f32 { input[(base + off) as usize] };

            // update clipping ring buffer
            cl = (cl << 1) & 6;
            if ji >= 2 && ji <= h - 3 {
                cl |= (at(-riw) > clip || at(0) > clip || at(riw) > clip) as u32;
            }

            let oidx = j * width_out + i;
            if ii_ < 2 || ii_ > w - 3 || ji < 2 || ji > h - 3 {
                // fast path for border
                output[oidx] = clip.min(at(0));
                continue;
            }

            // if the current pixel is clipped, always reconstruct
            let mut clipped = at(0) > clip;
            if !clipped && cl != 0 {
                // Slow case: reconstruct only if every 3x3 block touching the
                // pixel contains a clipped value (avoids zippering at edges of
                // clipped regions — the X-Trans pattern is prone to it).
                clipped = true;
                for offset_j in -2..=0isize {
                    for offset_i in -2..=0isize {
                        if clipped {
                            clipped = false;
                            for jj in offset_j..=offset_j + 2 {
                                for ii in offset_i..=offset_i + 2 {
                                    clipped = clipped || at(jj * riw + ii) > clip;
                                }
                            }
                        }
                    }
                }
            }

            if clipped {
                let mut mean = [0.0_f32; 3];
                let mut rgb_max = [f32::MIN; 3];
                let mut cnt = [0_i32; 3];
                for jj in -1..=1isize {
                    for ii in -1..=1isize {
                        let val = at(jj * riw + ii);
                        let c = raw::fc_xtrans(j as i32 + jj as i32, i as i32 + ii as i32, &xt);
                        mean[c] += val;
                        cnt[c] += 1;
                        rgb_max[c] = rgb_max[c].max(val);
                    }
                }

                let ro = (mean[0] / cnt[0] as f32).min(clip);
                let go = (mean[1] / cnt[1] as f32).min(clip);
                let bo = (mean[2] / cnt[2] as f32).min(clip);

                let (r, g, b) = (rgb_max[0], rgb_max[1], rgb_max[2]);
                let l = (r + g + b) / 3.0;
                let c = SQRT3 * (r - g);
                let h = 2.0 * b - g - r;
                let co = SQRT3 * (ro - go);
                let ho = 2.0 * bo - go - ro;

                let rgb = lch_backtransform(l, c, h, co, ho, r != g && g != b);
                output[oidx] = rgb[raw::fc_xtrans(j as i32, i as i32, &xt)];
            } else {
                output[oidx] = at(0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // RGGB Bayer mask (see raw.rs tests).
    const RGGB: u32 = 0x94949494;

    // Genuine Fuji X-Trans CFA (R=0,G=1,B=2). Unlike an arbitrary 6x6 table,
    // every 3x3 window contains all three colours — process_lch_xtrans relies
    // on that (cnt[c] > 0) exactly like the C original.
    const XTRANS: [[u8; 6]; 6] = [
        [1, 2, 1, 1, 0, 1], [0, 1, 0, 2, 1, 2],
        [1, 2, 1, 1, 0, 1], [1, 0, 1, 1, 2, 1],
        [2, 1, 2, 0, 1, 0], [1, 0, 1, 1, 2, 1],
    ];

    #[test]
    fn lch_bayer_unclipped_passes_through() {
        let (w, h) = (4usize, 4usize);
        let inp: Vec<f32> = (0..w * h).map(|k| 0.1 + 0.01 * k as f32).collect();
        let mut out = vec![-1.0_f32; w * h];
        unsafe {
            darkroom_highlights_lch_bayer(inp.as_ptr(), out.as_mut_ptr(), w, h, RGGB, 1.0);
        }
        assert_eq!(out, inp); // nothing clipped; borders min(clip, in) = in
    }

    #[test]
    fn lch_bayer_border_clamps_to_clip() {
        let (w, h) = (4usize, 4usize);
        let mut inp = vec![0.5_f32; w * h];
        inp[w * h - 1] = 5.0; // bottom-right border pixel above clip
        let mut out = vec![0.0_f32; w * h];
        unsafe {
            darkroom_highlights_lch_bayer(inp.as_ptr(), out.as_mut_ptr(), w, h, RGGB, 1.0);
        }
        assert_eq!(out[w * h - 1], 1.0);
    }

    #[test]
    fn lch_bayer_reconstructs_clipped_r_pixel() {
        // RGGB 2x2 block at (0,0): R=2.0 (clipped), G=G=0.5, B=0.5.
        // Gmax == B → no ratio rescale. L=1, H=-1.5, C=1.5*sqrt3;
        // out = L - H/6 + C/sqrt12 = 1 + 0.25 + 0.75 = 2.0.
        let (w, h) = (4usize, 4usize);
        let mut inp = vec![0.5_f32; w * h];
        inp[0] = 2.0;
        let mut out = vec![0.0_f32; w * h];
        unsafe {
            darkroom_highlights_lch_bayer(inp.as_ptr(), out.as_mut_ptr(), w, h, RGGB, 1.0);
        }
        assert!((out[0] - 2.0).abs() < 1e-5, "out[0]={}", out[0]);
        // neighbour (1,1) sees the clipped block too (its 2x2 spans rows 1-2) → untouched value
        assert!((out[w + 2] - 0.5).abs() < 1e-6); // far pixel unaffected
    }

    #[test]
    fn lch_xtrans_unclipped_passes_through() {
        let (w, h) = (8usize, 8usize);
        let inp: Vec<f32> = (0..w * h).map(|k| 0.1 + 0.005 * k as f32).collect();
        let mut out = vec![-1.0_f32; w * h];
        unsafe {
            darkroom_highlights_lch_xtrans(
                inp.as_ptr(), out.as_mut_ptr(), w, h, w, XTRANS.as_ptr() as *const u8, 1.0,
            );
        }
        assert_eq!(out, inp);
    }

    #[test]
    fn lch_xtrans_fully_clipped_field() {
        // every value 2.0 > clip=1: interior reconstructs to L=2 (R=G=B=2, C=H=0),
        // borders clamp to clip=1.
        let (w, h) = (8usize, 8usize);
        let inp = vec![2.0_f32; w * h];
        let mut out = vec![0.0_f32; w * h];
        unsafe {
            darkroom_highlights_lch_xtrans(
                inp.as_ptr(), out.as_mut_ptr(), w, h, w, XTRANS.as_ptr() as *const u8, 1.0,
            );
        }
        for j in 0..h {
            for i in 0..w {
                let v = out[j * w + i];
                if i < 2 || i > w - 3 || j < 2 || j > h - 3 {
                    assert_eq!(v, 1.0, "border ({j},{i})");
                } else {
                    assert!((v - 2.0).abs() < 1e-5, "interior ({j},{i})={v}");
                }
            }
        }
    }

    #[test]
    fn lch_xtrans_tiny_roi_takes_border_path() {
        // 2x2 ROI: every pixel is border; must not underflow the int-style guards.
        let inp = vec![5.0_f32; 4];
        let mut out = vec![0.0_f32; 4];
        unsafe {
            darkroom_highlights_lch_xtrans(
                inp.as_ptr(), out.as_mut_ptr(), 2, 2, 2, XTRANS.as_ptr() as *const u8, 1.0,
            );
        }
        assert_eq!(out, vec![1.0; 4]);
    }

    #[test]
    fn sraw_mask_zero_for_clean_pixels() {
        // Pixels well below threshold → mval = 0
        let inp = vec![0.1_f32, 0.2, 0.3, 1.0];
        let mut tmp = vec![-1.0_f32; 1];
        let clips = [1.0_f32, 1.0, 1.0, 1.0];
        unsafe {
            darkroom_highlights_mask_sraw(inp.as_ptr(), tmp.as_mut_ptr(), 1, 1, clips.as_ptr());
        }
        // refs = max(0.5, 0.95) = 0.95; (0.1-0.95)/0.95 ≈ -0.89 → max with 0 = 0
        assert_eq!(tmp[0], 0.0);
    }

    #[test]
    fn sraw_mask_positive_for_clipped_pixels() {
        // One channel exceeds the reference
        let inp = vec![2.0_f32, 0.2, 0.3, 1.0];
        let mut tmp = vec![0.0_f32; 1];
        let clips = [1.0_f32, 1.0, 1.0, 1.0];
        unsafe {
            darkroom_highlights_mask_sraw(inp.as_ptr(), tmp.as_mut_ptr(), 1, 1, clips.as_ptr());
        }
        // refs = 0.95; (2.0 - 0.95) / 0.95 ≈ 1.105
        assert!((tmp[0] - (2.0_f32 - 0.95) / 0.95).abs() < 1e-5);
    }

    #[test]
    fn sraw_mask_uses_max_channel() {
        // Multiple channels deviate; mask should pick the largest
        let inp = vec![0.6_f32, 0.7, 5.0, 1.0];
        let mut tmp = vec![0.0_f32; 1];
        let clips = [1.0_f32, 1.0, 1.0, 1.0];
        unsafe {
            darkroom_highlights_mask_sraw(inp.as_ptr(), tmp.as_mut_ptr(), 1, 1, clips.as_ptr());
        }
        let expected = (5.0_f32 - 0.95) / 0.95;
        assert!((tmp[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn sraw_mask_respects_lower_bound_on_reference() {
        // Very small clip → ref clamped at 0.5
        let inp = vec![0.55_f32, 0.0, 0.0, 1.0];
        let mut tmp = vec![0.0_f32; 1];
        let clips = [0.1_f32, 0.1, 0.1, 0.1]; // 0.95*0.1 = 0.095, below 0.5
        unsafe {
            darkroom_highlights_mask_sraw(inp.as_ptr(), tmp.as_mut_ptr(), 1, 1, clips.as_ptr());
        }
        // ref clamped to 0.5; (0.55 - 0.5) / 0.5 = 0.1
        assert!((tmp[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn clip_sraw_clamps_each_float_independently() {
        let inp = vec![0.5_f32, 1.5, 0.7, 0.9, 2.0, 0.0, 0.6, 1.2];
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { darkroom_highlights_clip_sraw(inp.as_ptr(), out.as_mut_ptr(), inp.len(), 1.0); }
        let expected = vec![0.5_f32, 1.0, 0.7, 0.9, 1.0, 0.0, 0.6, 1.0];
        assert_eq!(out, expected);
    }

    #[test]
    fn visualize_sraw_marks_unclipped_as_dim_clipped_as_white() {
        let inp = vec![0.5_f32, 0.7, 0.9, 0.42, 1.5, 0.2, 2.0, 1.0];
        let mut out = vec![-1.0_f32; inp.len()];
        let clips = [1.0_f32; 4];
        unsafe {
            darkroom_highlights_visualize_sraw(
                inp.as_ptr(), out.as_mut_ptr(), 2, clips.as_ptr(),
            );
        }
        // pixel 0: all RGB below clip → 0.2 * v; alpha forced to 0
        assert!((out[0] - 0.1).abs() < 1e-6);
        assert!((out[1] - 0.14).abs() < 1e-6);
        assert!((out[2] - 0.18).abs() < 1e-6);
        assert_eq!(out[3], 0.0);
        // pixel 1: R=1.5 → 1.0, G=0.2 → 0.04, B=2.0 → 1.0; alpha 0
        assert_eq!(out[4], 1.0);
        assert!((out[5] - 0.04).abs() < 1e-6);
        assert_eq!(out[6], 1.0);
        assert_eq!(out[7], 0.0);
    }

    #[test]
    fn visualize_mosaic_handles_out_of_bounds_pixels() {
        // 2x2 output, in is also 2x2; with irow_offset = -1 the first row of
        // output reaches into the negative-row region, which must yield 0.
        let inp = vec![2.0_f32, 0.5, 0.5, 2.0];
        let mut out = vec![-7.0_f32; 4];
        let clips = [1.0_f32; 4];
        let xt = [[0_u8; 6]; 6];
        unsafe {
            darkroom_highlights_visualize_mosaic(
                inp.as_ptr(), out.as_mut_ptr(),
                2, 2, 2, 2,
                0x94949494, xt.as_ptr() as *const u8,
                clips.as_ptr(),
                -1, 0, // shift output one row up
            );
        }
        // out row 0 (maps to in row -1) → both 0.0
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        // out row 1 (maps to in row 0): in[0]=2.0 ≥ clip → 1.0; in[1]=0.5 < clip → 0.1
        assert_eq!(out[2], 1.0);
        assert!((out[3] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn visualize_mosaic_handles_xtrans_pattern() {
        // Real Fujifilm X-Trans 6x6 (0=R, 1=G, 2=B). filters == 9 routes
        // through raw::fc_xtrans rather than fc_bayer.
        let xt_pattern: [[u8; 6]; 6] = [
            [1, 2, 1, 1, 0, 1],
            [0, 1, 0, 2, 1, 2],
            [1, 2, 1, 1, 0, 1],
            [1, 0, 1, 1, 2, 1],
            [2, 1, 2, 0, 1, 0],
            [1, 0, 1, 1, 2, 1],
        ];
        let xt_flat: Vec<u8> = xt_pattern.iter().flatten().copied().collect();
        // Single 1×1 input — pixel at (0,0) is colour 1 (G).
        let inp = vec![0.5_f32];
        let mut out = vec![-1.0_f32; 1];
        // Per-colour clips: R=1.0, G=0.4, B=1.0, alpha=1.0
        let clips = [1.0_f32, 0.4, 1.0, 1.0];
        unsafe {
            darkroom_highlights_visualize_mosaic(
                inp.as_ptr(), out.as_mut_ptr(),
                1, 1, 1, 1,
                9, // filters == 9 → X-Trans branch
                xt_flat.as_ptr(),
                clips.as_ptr(),
                0, 0,
            );
        }
        // 0.5 > clip_G=0.4 → 1.0
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn clip_sraw_matches_fminf_nan_semantics() {
        // C fminf(clip, NaN) returns clip (non-NaN). Rust's f32::min returns
        // clip too. Verify the wrapper does the same in both orderings.
        let inp = vec![f32::NAN, 0.5];
        let mut out = vec![-1.0_f32; 2];
        unsafe { darkroom_highlights_clip_sraw(inp.as_ptr(), out.as_mut_ptr(), 2, 0.8); }
        // out[0] = fminf(0.8, NaN) = 0.8
        assert_eq!(out[0], 0.8);
        // out[1] = fminf(0.8, 0.5) = 0.5
        assert_eq!(out[1], 0.5);

        // fminf(NaN, value) returns value; Rust f32::min disagrees here.
        let inp = vec![0.5_f32];
        let mut out = vec![-1.0_f32; 1];
        unsafe { darkroom_highlights_clip_sraw(inp.as_ptr(), out.as_mut_ptr(), 1, f32::NAN); }
        // C: fminf(NaN, 0.5) = 0.5. Our wrapper must return 0.5, not NaN.
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn mosaic_mask_uses_bayer_quadrant_clips() {
        // 2x2 RGGB Bayer image; each pixel gets its colour-specific reference
        let inp = vec![2.0_f32, 0.0, 0.0, 3.0]; // R, G, G, B
        let mut tmp = vec![0.0_f32; 4];
        let clips = [1.0_f32, 0.5, 2.0, 1.0]; // R=1, G=0.5, B=2
        let xt = [[0_u8; 6]; 6];
        let rggb: u32 = 0x94949494;
        unsafe {
            darkroom_highlights_mask_mosaic(
                inp.as_ptr(), tmp.as_mut_ptr(),
                2, 2,
                rggb, xt.as_ptr() as *const u8,
                clips.as_ptr(),
                0, 0,
            );
        }
        // refs: R = max(0.5, 0.95*1.0) = 0.95
        //       G = max(0.5, 0.95*0.5) = 0.5
        //       B = max(0.5, 0.95*2.0) = 1.9
        // (0,0)=R: (2.0-0.95)/0.95 ≈ 1.105
        // (0,1)=G: (0.0-0.5)/0.5 = -1 → 0
        // (1,0)=G: (0.0-0.5)/0.5 = -1 → 0
        // (1,1)=B: (3.0-1.9)/1.9 ≈ 0.579
        assert!((tmp[0] - (2.0_f32 - 0.95) / 0.95).abs() < 1e-5);
        assert_eq!(tmp[1], 0.0);
        assert_eq!(tmp[2], 0.0);
        assert!((tmp[3] - (3.0_f32 - 1.9) / 1.9).abs() < 1e-5);
    }
}
