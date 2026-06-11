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

// ── Opposed highlight reconstruction (src/iop/hlreconstruct/opposed.c) ────────

/// x³ — matches fcube() in src/common/math.h:295.
#[inline(always)]
fn fcube(a: f32) -> f32 { a * a * a }

/// Opposing-channel reference for an sRAW RGBA pixel: cube of the mean of the
/// two other channels' cube roots. Matches _calc_linear_refavg() (opposed.c:49).
#[inline(always)]
fn calc_linear_refavg(pix: &[f32], color: usize) -> f32 {
    let ins = [
        pix[0].max(0.0).cbrt(),
        pix[1].max(0.0).cbrt(),
        pix[2].max(0.0).cbrt(),
    ];
    let opp = [
        0.5 * (ins[1] + ins[2]),
        0.5 * (ins[0] + ins[2]),
        0.5 * (ins[0] + ins[1]),
    ];
    fcube(opp[color])
}

/// Raw-mosaic opposing-channel reference: per-colour means over the
/// (row-1..row+1, col-1..col+1)-ish window (note the C's asymmetric clamping
/// at the bottom/right edges), cube-rooted with `correction`, then the mean of
/// the two opposing channels; cubed when `linear`.
/// Matches _calc_refavg() in src/iop/hlreconstruct/segbased.c:186.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn calc_refavg(
    input: &[f32], xt: &[[u8; 6]; 6], filters: u32,
    row: usize, col: usize, width: usize, height: usize,
    correction: &[f32; 4], linear: bool,
) -> f32 {
    let color = raw::fcol(row as i32, col as i32, filters, xt);
    let mut mean = [0.0_f32; 3];
    let mut cnt = [0.0_f32; 3];

    let dymin = row.saturating_sub(1);
    let dxmin = col.saturating_sub(1);
    let dymax = (height - 1).min(row + 2); // exclusive — mirrors the C `dy < dymax`
    let dxmax = (width - 1).min(col + 2);

    for dy in dymin..dymax {
        for dx in dxmin..dxmax {
            let val = input[dy * width + dx].max(0.0);
            let c = raw::fcol(dy as i32, dx as i32, filters, xt);
            mean[c] += val;
            cnt[c] += 1.0;
        }
    }
    for c in 0..3 {
        mean[c] = if cnt[c] > 0.0 { ((correction[c] * mean[c]) / cnt[c]).cbrt() } else { 0.0 };
    }
    let croot_refavg = [
        0.5 * (mean[1] + mean[2]),
        0.5 * (mean[0] + mean[2]),
        0.5 * (mean[0] + mean[1]),
    ];
    if linear { fcube(croot_refavg[color]) } else { croot_refavg[color] }
}

/// Coarse-map index of a raw photosite: 3x3 superpixels. Matches
/// _raw_to_cmap() (opposed.c:57), with the row/col **clamped to the map** —
/// the C version can index one cmap row/col past the end when
/// width/height ≡ 1 (mod 3) (silent OOB read there; clamped here).
#[inline(always)]
fn raw_to_cmap(mwidth: usize, mheight: usize, row: usize, col: usize) -> usize {
    (row / 3).min(mheight - 1) * mwidth + (col / 3).min(mwidth - 1)
}

/// 7x7-ish dilation tap pattern around `center`. Matches _mask_dilated()
/// (opposed.c:62). Caller guarantees ±3 rows/cols around `center` are in
/// bounds (the loop bounds below do).
#[inline(always)]
fn mask_dilated(m: &[u8], center: usize, w1: usize) -> u8 {
    let at = |off: isize| -> u8 { m[(center as isize + off) as usize] };
    if at(0) != 0 { return 1; }
    let w1 = w1 as isize;
    if at(-w1 - 1) | at(-w1) | at(-w1 + 1) | at(-1) | at(1) | at(w1 - 1) | at(w1) | at(w1 + 1) != 0 {
        return 1;
    }
    let w2 = 2 * w1;
    let w3 = 3 * w1;
    let ring = at(-w3 - 2) | at(-w3 - 1) | at(-w3) | at(-w3 + 1) | at(-w3 + 2)
        | at(-w2 - 3) | at(-w2 - 2) | at(-w2 - 1) | at(-w2) | at(-w2 + 1) | at(-w2 + 2) | at(-w2 + 3)
        | at(-w1 - 3) | at(-w1 - 2) | at(-w1 + 2) | at(-w1 + 3)
        | at(-3) | at(-2) | at(2) | at(3)
        | at(w1 - 3) | at(w1 - 2) | at(w1 + 2) | at(w1 + 3)
        | at(w2 - 3) | at(w2 - 2) | at(w2 - 1) | at(w2) | at(w2 + 1) | at(w2 + 2) | at(w2 + 3)
        | at(w3 - 2) | at(w3 - 1) | at(w3) | at(w3 + 1) | at(w3 + 2);
    if ring != 0 { 1 } else { 0 }
}

/// Read the 6x6 X-Trans table from a raw byte pointer.
/// # Safety: `xtrans` points to 36 bytes.
#[inline(always)]
unsafe fn read_xtrans(xtrans: *const u8) -> [[u8; 6]; 6] {
    let b = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = b[r * 6 + c]; }
    }
    xt
}

/// sRAW clipped-superpixel mask build. Returns 1 if any superpixel clipped.
/// Replaces the DT_OMP_FOR(reduction(|:anyclipped)) loop in
/// _process_linear_opposed() (opposed.c:124). Note: like the C, the clip test
/// reads channel 0 (`input[idx]`) for all three colour comparisons.
///
/// # Safety
/// `in_buf` holds `width*height*4` floats; `mask_buf` holds `6*msize` bytes
/// (channels 0..2 written); `clips` holds 3 floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_opposed_mask_sraw(
    in_buf: *const f32, mask_buf: *mut u8,
    width: usize, height: usize, mwidth: usize, mheight: usize, msize: usize,
    clips: *const f32,
) -> i32 {
    if width < 2 || height < 2 { return 0; }
    let input = std::slice::from_raw_parts(in_buf, width * height * 4);
    let mask = std::slice::from_raw_parts_mut(mask_buf, 6 * msize);
    let clips = std::slice::from_raw_parts(clips, 3);
    let mut anyclipped = false;

    for row in 0..height - 1 {
        for col in 0..width - 1 {
            let idx = (row * width + col) * 4;
            let mdx = raw_to_cmap(mwidth, mheight, row, col);
            for c in 0..3 {
                if input[idx] >= clips[c] && mask[c * msize + mdx] == 0 {
                    mask[c * msize + mdx] |= 1;
                    anyclipped = true;
                }
            }
        }
    }
    anyclipped as i32
}

/// sRAW mask dilation (interior only, rows/cols 3..m-4). Replaces the
/// DT_OMP_FOR(collapse(2)) loop in _process_linear_opposed() (opposed.c:151).
///
/// # Safety
/// `mask_buf` holds `6*msize` bytes (reads channels 0..2, writes 3..5).
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_opposed_dilate_sraw(
    mask_buf: *mut u8, mwidth: usize, mheight: usize, msize: usize,
) {
    if mwidth < 7 || mheight < 7 { return; }
    let mask = std::slice::from_raw_parts_mut(mask_buf, 6 * msize);
    for row in 3..mheight - 3 {
        for col in 3..mwidth - 3 {
            let mx = row * mwidth + col;
            mask[3 * msize + mx] = mask_dilated(mask, mx, mwidth);
            mask[4 * msize + mx] = mask_dilated(mask, msize + mx, mwidth);
            mask[5 * msize + mx] = mask_dilated(mask, 2 * msize + mx, mwidth);
        }
    }
}

/// sRAW chrominance sums: accumulate (inval - linear refavg) for unclipped
/// pixels near clipped areas. Replaces the DT_OMP_FOR(reduction(+:sums,cnts))
/// loop in _process_linear_opposed() (opposed.c:163). `sums`/`cnts` are
/// caller-zeroed 4-float accumulators.
///
/// # Safety
/// Buffers per `darkroom_highlights_opposed_mask_sraw`; `sums`/`cnts` hold 4
/// floats each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_opposed_chroma_sraw(
    in_buf: *const f32, mask_buf: *const u8,
    width: usize, height: usize, mwidth: usize, mheight: usize, msize: usize,
    clips: *const f32, sums: *mut f32, cnts: *mut f32,
) {
    if width < 7 || height < 7 { return; }
    let input = std::slice::from_raw_parts(in_buf, width * height * 4);
    let mask = std::slice::from_raw_parts(mask_buf, 6 * msize);
    let clips = std::slice::from_raw_parts(clips, 3);
    let sums = std::slice::from_raw_parts_mut(sums, 4);
    let cnts = std::slice::from_raw_parts_mut(cnts, 4);

    for row in 3..height - 3 {
        for col in 3..width - 3 {
            let idx = (row * width + col) * 4;
            let mdx = raw_to_cmap(mwidth, mheight, row, col);
            for c in 0..3 {
                let inval = input[idx + c];
                if inval > 0.2 * clips[c] && inval < clips[c] && mask[(c + 3) * msize + mdx] != 0 {
                    sums[c] += inval - calc_linear_refavg(&input[idx..idx + 4], c);
                    cnts[c] += 1.0;
                }
            }
        }
    }
}

/// sRAW final output: clipped channels become max(in, refavg + chrominance).
/// Replaces the DT_OMP_FOR(collapse(2)) loop in _process_linear_opposed()
/// (opposed.c:201). Only channels 0..2 are written (alpha untouched, as in C).
///
/// # Safety
/// `in_buf`/`out_buf` hold `npixels*4` floats; `clips`/`chrominance` hold >= 3
/// floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_opposed_output_sraw(
    in_buf: *const f32, out_buf: *mut f32, npixels: usize,
    clips: *const f32, chrominance: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let clips = std::slice::from_raw_parts(clips, 3);
    let chrominance = std::slice::from_raw_parts(chrominance, 3);

    for k in 0..npixels {
        let idx = k * 4;
        for c in 0..3 {
            let refv = calc_linear_refavg(&input[idx..idx + 4], c);
            let inval = input[idx + c].max(0.0);
            output[idx + c] = if inval >= clips[c] { inval.max(refv + chrominance[c]) } else { inval };
        }
    }
}

/// Raw clipped-superpixel mask build over 3x3 photosite blocks. Returns 1 if
/// any superpixel clipped. Replaces the DT_OMP_FOR(reduction(|:anyclipped)
/// collapse(2)) loop in _process_opposed() (opposed.c:267).
///
/// # Safety
/// `in_buf` holds `>= (3*(mheight-1)) * width` floats (single-channel raw);
/// `mask_buf` holds `6*msize` bytes; `clips` 3 floats; `xtrans` 36 bytes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_opposed_mask_raw(
    in_buf: *const f32, mask_buf: *mut u8,
    width: usize, mwidth: usize, mheight: usize, msize: usize,
    filters: u32, xtrans: *const u8, clips: *const f32,
) -> i32 {
    if mwidth < 1 || mheight < 1 { return 0; }
    // the raw frame has >= 3*mheight rows (mheight = height/3)
    let input = std::slice::from_raw_parts(in_buf, 3 * mheight * width);
    let mask = std::slice::from_raw_parts_mut(mask_buf, 6 * msize);
    let clips = std::slice::from_raw_parts(clips, 3);
    let xt = read_xtrans(xtrans);
    let mut anyclipped = false;

    for mrow in 0..mheight.saturating_sub(1) {
        for mcol in 0..mwidth.saturating_sub(1) {
            let mut mbuff = [0_u8; 3];
            for y in 0..3 {
                for x in 0..3 {
                    let r = 3 * mrow + y;
                    let cidx = 3 * mcol + x;
                    let idx = r * width + cidx;
                    let color = raw::fcol(r as i32, cidx as i32, filters, &xt);
                    if input[idx] >= clips[color] { mbuff[color] += 1; }
                }
            }
            for c in 0..3 {
                mask[c * msize + mrow * mwidth + mcol] = (mbuff[c] != 0) as u8;
                anyclipped |= mbuff[c] != 0;
            }
        }
    }
    anyclipped as i32
}

/// Raw mask dilation over the whole map; border cells copy the source mask.
/// Replaces the DT_OMP_FOR(collapse(2)) loop in _process_opposed()
/// (opposed.c:301). The `safe` test uses signed arithmetic: for maps narrower
/// than 8 cells every cell takes the copy branch (the C's size_t arithmetic
/// would wrap and read out of bounds there).
///
/// # Safety
/// `mask_buf` holds `6*msize` bytes.
#[no_mangle]
pub unsafe extern "C" fn darkroom_highlights_opposed_dilate_raw(
    mask_buf: *mut u8, mwidth: usize, mheight: usize, msize: usize,
) {
    let mask = std::slice::from_raw_parts_mut(mask_buf, 6 * msize);
    let (w, h) = (mwidth as isize, mheight as isize);
    for row in 0..mheight {
        for col in 0..mwidth {
            let mx = row * mwidth + col;
            let safe = col >= 3 && row >= 3 && (col as isize) < w - 4 && (row as isize) < h - 4;
            if safe {
                mask[3 * msize + mx] = mask_dilated(mask, mx, mwidth);
                mask[4 * msize + mx] = mask_dilated(mask, msize + mx, mwidth);
                mask[5 * msize + mx] = mask_dilated(mask, 2 * msize + mx, mwidth);
            } else {
                mask[3 * msize + mx] = mask[mx];
                mask[4 * msize + mx] = mask[mx + msize];
                mask[5 * msize + mx] = mask[mx + 2 * msize];
            }
        }
    }
}

/// Raw chrominance sums via _calc_refavg. Replaces the
/// DT_OMP_FOR(reduction(+:sums,cnts) collapse(2)) loop in _process_opposed()
/// (opposed.c:316). `sums`/`cnts` are caller-zeroed 4-float accumulators.
///
/// # Safety
/// `in_buf` holds `width*height` floats; `mask_buf` `6*msize` bytes;
/// `clips` 3, `correction` 4, `sums`/`cnts` 4 floats; `xtrans` 36 bytes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_opposed_chroma_raw(
    in_buf: *const f32, mask_buf: *const u8,
    width: usize, height: usize, mwidth: usize, mheight: usize, msize: usize,
    filters: u32, xtrans: *const u8, clips: *const f32, correction: *const f32,
    sums: *mut f32, cnts: *mut f32,
) {
    let input = std::slice::from_raw_parts(in_buf, width * height);
    let mask = std::slice::from_raw_parts(mask_buf, 6 * msize);
    let clips = std::slice::from_raw_parts(clips, 3);
    let corr: &[f32; 4] = &*(correction as *const [f32; 4]);
    let sums = std::slice::from_raw_parts_mut(sums, 4);
    let cnts = std::slice::from_raw_parts_mut(cnts, 4);
    let xt = read_xtrans(xtrans);

    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            let color = raw::fcol(row as i32, col as i32, filters, &xt);
            let inval = input[idx];
            if inval < clips[color]
                && inval > 0.2 * clips[color]
                && mask[(color + 3) * msize + raw_to_cmap(mwidth, mheight, row, col)] != 0
            {
                sums[color] += inval - calc_refavg(input, &xt, filters, row, col, width, height, corr, true);
                cnts[color] += 1.0;
            }
        }
    }
}

/// Raw full-frame reconstruction into `tmpout` (the `keep` path). Replaces the
/// DT_OMP_FOR(collapse(2)) loop in _process_opposed() (opposed.c:361).
///
/// # Safety
/// `in_buf`/`tmpout` hold `width*height` floats; scalars as in chroma_raw.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_opposed_tmpout_raw(
    in_buf: *const f32, tmpout: *mut f32,
    width: usize, height: usize,
    filters: u32, xtrans: *const u8, clips: *const f32,
    chrominance: *const f32, correction: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, width * height);
    let out = std::slice::from_raw_parts_mut(tmpout, width * height);
    let clips = std::slice::from_raw_parts(clips, 3);
    let chrominance = std::slice::from_raw_parts(chrominance, 3);
    let corr: &[f32; 4] = &*(correction as *const [f32; 4]);
    let xt = read_xtrans(xtrans);

    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            let color = raw::fcol(row as i32, col as i32, filters, &xt);
            let inval = input[idx];
            out[idx] = if inval >= clips[color] {
                let refv = calc_refavg(input, &xt, filters, row, col, width, height, corr, true);
                inval.max(refv + chrominance[color])
            } else {
                inval
            };
        }
    }
}

/// Raw output crop/reconstruct into the roi_out plane. When `tmpout` is
/// non-null the reconstruction is taken from it; otherwise it is recomputed
/// on the fly. Out-of-input positions write 0. Replaces the
/// DT_OMP_FOR(collapse(2)) loop in _process_opposed() (opposed.c:380).
///
/// # Safety
/// `out_buf` holds `out_width*out_height` floats; `in_buf` (and `tmpout` when
/// non-null) hold `in_width*in_height` floats; scalars as in chroma_raw.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_opposed_output_raw(
    in_buf: *const f32, tmpout: *const f32, out_buf: *mut f32,
    out_width: usize, out_height: usize, out_x: i32, out_y: i32,
    in_width: usize, in_height: usize,
    filters: u32, xtrans: *const u8, clips: *const f32,
    chrominance: *const f32, correction: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, in_width * in_height);
    let tmp = if tmpout.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(tmpout, in_width * in_height))
    };
    let output = std::slice::from_raw_parts_mut(out_buf, out_width * out_height);
    let clips = std::slice::from_raw_parts(clips, 3);
    let chrominance = std::slice::from_raw_parts(chrominance, 3);
    let corr: &[f32; 4] = &*(correction as *const [f32; 4]);
    let xt = read_xtrans(xtrans);

    for row in 0..out_height {
        for col in 0..out_width {
            let odx = row * out_width + col;
            // C adds the (possibly negative) roi offsets in size_t and relies
            // on wrap-around to fail the bounds test; signed math here.
            let irow = row as i64 + out_y as i64;
            let icol = col as i64 + out_x as i64;
            let mut oval = 0.0_f32;
            if irow >= 0 && (irow as usize) < in_height && icol >= 0 && (icol as usize) < in_width {
                let (irow, icol) = (irow as usize, icol as usize);
                let ix = irow * in_width + icol;
                oval = match tmp {
                    Some(t) => t[ix],
                    None => {
                        let color = raw::fcol(irow as i32, icol as i32, filters, &xt);
                        let v = input[ix];
                        if v >= clips[color] {
                            let refv = calc_refavg(input, &xt, filters, irow, icol, in_width, in_height, corr, true);
                            v.max(refv + chrominance[color])
                        } else {
                            v
                        }
                    }
                };
            }
            output[odx] = oval;
        }
    }
}

// ── Segmentation morphology (src/iop/hlreconstruct/segmentation.c) ───────────

/// Progressive-radius dilation test (radius 1..8): OR of the ring taps,
/// returning early once any tap is set or the radius is exhausted.
/// Matches _test_dilate() (segmentation.c:104). The two asymmetric taps in
/// ring 8 (`-w5+6` at line 194, `-w7+6` at line 206 where symmetry suggests
/// `+7`/`+w7+6`) are replicated verbatim from the C.
fn test_dilate(img: &[u32], i: usize, w1: usize, radius: i32) -> u32 {
    let at = |off: isize| -> u32 { img[(i as isize + off) as usize] };
    let w1 = w1 as isize;

    let mut retval = at(-w1 - 1) | at(-w1) | at(-w1 + 1)
        | at(-1) | at(0) | at(1)
        | at(w1 - 1) | at(w1) | at(w1 + 1);
    if retval != 0 || radius < 2 { return retval; }

    let w2 = 2 * w1;
    retval = at(-w2 - 1) | at(-w2) | at(-w2 + 1)
        | at(-w1 - 2) | at(-w1 + 2)
        | at(-2) | at(2)
        | at(w1 - 2) | at(w1 + 2)
        | at(w2 - 1) | at(w2) | at(w2 + 1);
    if retval != 0 || radius < 3 { return retval; }

    let w3 = 3 * w1;
    retval = at(-w3 - 2) | at(-w3 - 1) | at(-w3) | at(-w3 + 1) | at(-w3 + 2)
        | at(-w2 - 3) | at(-w2 - 2) | at(-w2 + 2) | at(-w2 + 3)
        | at(-w1 - 3) | at(-w1 + 3)
        | at(-3) | at(3)
        | at(w1 - 3) | at(w1 + 3)
        | at(w2 - 3) | at(w2 - 2) | at(w2 + 2) | at(w2 + 3)
        | at(w3 - 2) | at(w3 - 1) | at(w3) | at(w3 + 1) | at(w3 + 2);
    if retval != 0 || radius < 4 { return retval; }

    let w4 = 4 * w1;
    retval = at(-w4 - 2) | at(-w4 - 1) | at(-w4) | at(-w4 + 1) | at(-w4 + 2)
        | at(-w3 - 3) | at(-w3 + 3)
        | at(-w2 - 4) | at(-w2 + 4)
        | at(-w1 - 4) | at(-w1 + 4)
        | at(-4) | at(4)
        | at(w1 - 4) | at(w1 + 4)
        | at(w2 - 4) | at(w2 + 4)
        | at(w3 - 3) | at(w3 + 3)
        | at(w4 - 2) | at(w4 - 1) | at(w4) | at(w4 + 1) | at(w4 + 2);
    if retval != 0 || radius < 5 { return retval; }

    let w5 = 5 * w1;
    retval = at(-w5 - 2) | at(-w5 - 1) | at(-w5) | at(-w5 + 1) | at(-w5 + 2)
        | at(-w4 - 4) | at(-w4 - 3) | at(-w4 + 3) | at(-w4 + 4)
        | at(-w3 - 4) | at(-w3 + 4)
        | at(-w2 - 5) | at(-w2 + 5)
        | at(-w1 - 5) | at(-w1 + 5)
        | at(-5) | at(5)
        | at(w1 - 5) | at(w1 + 5)
        | at(w2 - 5) | at(w2 + 5)
        | at(w3 - 4) | at(w3 + 4)
        | at(w4 - 4) | at(w4 - 3) | at(w4 + 3) | at(w4 + 4)
        | at(w5 - 2) | at(w5 - 1) | at(w5) | at(w5 + 1) | at(w5 + 2);
    if retval != 0 || radius < 6 { return retval; }

    let w6 = 6 * w1;
    retval = at(-w6 - 2) | at(-w6 - 1) | at(-w6) | at(-w6 + 1) | at(-w6 + 2)
        | at(-w5 - 4) | at(-w5 - 3) | at(-w5 + 3) | at(-w5 + 4)
        | at(-w4 - 5) | at(-w4 + 5)
        | at(-w3 - 5) | at(-w3 + 5)
        | at(-w2 - 6) | at(-w2 + 6)
        | at(-w1 - 6) | at(-w1 + 6)
        | at(-6) | at(6)
        | at(w1 - 6) | at(w1 + 6)
        | at(w2 - 6) | at(w2 + 6)
        | at(w3 - 5) | at(w3 + 5)
        | at(w4 - 5) | at(w4 + 5)
        | at(w5 - 4) | at(w5 - 3) | at(w5 + 3) | at(w5 + 4)
        | at(w6 - 2) | at(w6 - 1) | at(w6) | at(w6 + 1) | at(w6 + 2);
    if retval != 0 || radius < 7 { return retval; }

    let w7 = 7 * w1;
    retval = at(-w7 - 3) | at(-w7 - 2) | at(-w7 - 1) | at(-w7) | at(-w7 + 1) | at(-w7 + 2) | at(-w7 + 3)
        | at(-w6 - 4) | at(-w6 - 3) | at(-w6 + 3) | at(-w6 + 4)
        | at(-w5 - 6) | at(-w5 - 5) | at(-w5 + 5) | at(-w5 + 6)
        | at(-w4 - 6) | at(-w4 + 6)
        | at(-w3 - 7) | at(-w3 - 6) | at(-w3 + 6) | at(-w3 + 7)
        | at(-w2 - 7) | at(-w2 + 7)
        | at(-w1 - 7) | at(-w1 + 7)
        | at(-7) | at(7)
        | at(w1 - 7) | at(w1 + 7)
        | at(w2 - 7) | at(w2 + 7)
        | at(w3 - 7) | at(w3 - 6) | at(w3 + 6) | at(w3 + 7)
        | at(w4 - 6) | at(w4 + 6)
        | at(w5 - 6) | at(w5 - 5) | at(w5 + 5) | at(w5 + 6)
        | at(w6 - 4) | at(w6 - 3) | at(w6 + 3) | at(w6 + 4)
        | at(w7 - 3) | at(w7 - 2) | at(w7 - 1) | at(w7) | at(w7 + 1) | at(w7 + 2) | at(w7 + 3);
    if retval != 0 || radius < 8 { return retval; }

    let w8 = 8 * w1;
    at(-w8 - 4) | at(-w8 - 3) | at(-w8 - 2) | at(-w8 - 1) | at(-w8) | at(-w8 + 1) | at(-w8 + 2) | at(-w8 + 3) | at(-w8 + 4)
        | at(-w7 - 6) | at(-w7 - 5) | at(-w7 - 4) | at(-w7 + 4) | at(-w7 + 5) | at(-w7 + 6)
        | at(-w6 - 6) | at(-w6 - 5) | at(-w6 + 5) | at(-w6 + 6)
        | at(-w5 - 7) | at(-w5 + 6) // C quirk: -w5+6, not +7 (segmentation.c:194)
        | at(-w4 - 8) | at(-w4 - 7) | at(-w4 + 7) | at(-w4 + 8)
        | at(-w3 - 8) | at(-w3 - 7) | at(-w3 + 7) | at(-w3 + 8)
        | at(-w2 - 8) | at(-w2 + 8)
        | at(-w1 - 8) | at(-w1 + 8)
        | at(-8) | at(8)
        | at(w1 - 8) | at(w1 + 8)
        | at(w2 - 8) | at(w2 + 8)
        | at(w3 - 8) | at(w3 - 7) | at(w3 + 7) | at(w3 + 8)
        | at(w4 - 8) | at(w4 - 7) | at(w4 + 7) | at(w4 + 8)
        | at(w5 - 7) | at(w5 + 7)
        | at(w6 - 6) | at(w6 - 5) | at(w6 + 5) | at(w6 + 6)
        | at(w7 - 6) | at(w7 - 5) | at(w7 - 4) | at(w7 + 4) | at(w7 + 5) | at(-w7 + 6) // C quirk: -w7+6 (segmentation.c:206)
        | at(w8 - 4) | at(w8 - 3) | at(w8 - 2) | at(w8 - 1) | at(w8) | at(w8 + 1) | at(w8 + 2) | at(w8 + 3) | at(w8 + 4)
}

/// Progressive-radius erosion test (radius 1..5): AND of the ring taps,
/// returning early once any tap is clear or the radius is exhausted.
/// Matches _test_erode() (segmentation.c:229).
fn test_erode(img: &[u32], i: usize, w1: usize, radius: i32) -> u32 {
    let at = |off: isize| -> u32 { img[(i as isize + off) as usize] };
    let w1 = w1 as isize;

    let mut retval = at(-w1 - 1) & at(-w1) & at(-w1 + 1)
        & at(-1) & at(0) & at(1)
        & at(w1 - 1) & at(w1) & at(w1 + 1);
    if retval == 0 || radius < 2 { return retval; }

    let w2 = 2 * w1;
    retval = at(-w2 - 1) & at(-w2) & at(-w2 + 1)
        & at(-w1 - 2) & at(-w1 + 2)
        & at(-2) & at(2)
        & at(w1 - 2) & at(w1 + 2)
        & at(w2 - 1) & at(w2) & at(w2 + 1);
    if retval == 0 || radius < 3 { return retval; }

    let w3 = 3 * w1;
    retval = at(-w3 - 2) & at(-w3 - 1) & at(-w3) & at(-w3 + 1) & at(-w3 + 2)
        & at(-w2 - 3) & at(-w2 - 2) & at(-w2 + 2) & at(-w2 + 3)
        & at(-w1 - 3) & at(-w1 + 3)
        & at(-3) & at(3)
        & at(w1 - 3) & at(w1 + 3)
        & at(w2 - 3) & at(w2 - 2) & at(w2 + 2) & at(w2 + 3)
        & at(w3 - 2) & at(w3 - 1) & at(w3) & at(w3 + 1) & at(w3 + 2);
    if retval == 0 || radius < 4 { return retval; }

    let w4 = 4 * w1;
    retval = at(-w4 - 2) & at(-w4 - 1) & at(-w4) & at(-w4 + 1) & at(-w4 + 2)
        & at(-w3 - 3) & at(-w3 + 3)
        & at(-w2 - 4) & at(-w2 + 4)
        & at(-w1 - 4) & at(-w1 + 4)
        & at(-4) & at(4)
        & at(w1 - 4) & at(w1 + 4)
        & at(w2 - 4) & at(w2 + 4)
        & at(w3 - 3) & at(w3 + 3)
        & at(w4 - 2) & at(w4 - 1) & at(w4) & at(w4 + 1) & at(w4 + 2);
    if retval == 0 || radius < 5 { return retval; }

    let w5 = 5 * w1;
    at(-w5 - 2) & at(-w5 - 1) & at(-w5) & at(-w5 + 1) & at(-w5 + 2)
        & at(-w4 - 4) & at(-w4 - 3) & at(-w4 + 3) & at(-w4 + 4)
        & at(-w3 - 4) & at(-w3 + 4)
        & at(-w2 - 5) & at(-w2 + 5)
        & at(-w1 - 5) & at(-w1 + 5)
        & at(-5) & at(5)
        & at(w1 - 5) & at(w1 + 5)
        & at(w2 - 5) & at(w2 + 5)
        & at(w3 - 4) & at(w3 + 4)
        & at(w4 - 4) & at(w4 - 3) & at(w4 + 3) & at(w4 + 4)
        & at(w5 - 2) & at(w5 - 1) & at(w5) & at(w5 + 1) & at(w5 + 2)
}

/// Morphological dilation of the segmentation bitmap interior.
/// Replaces the DT_OMP_FOR(collapse(2)) loop in _dilating()
/// (segmentation.c:218): o[i] = test_dilate(...) ? 1 : 0 for the
/// border-inset interior.
///
/// # Safety
/// `img`/`out` hold `width*height` u32s; `border >= radius` per the C caller
/// contract (ring-r taps reach at most ±r rows/cols around the pixel).
#[no_mangle]
pub unsafe extern "C" fn darkroom_segmentation_dilate(
    img: *const u32, out: *mut u32,
    width: usize, height: usize, border: i32, radius: i32,
) {
    let input = std::slice::from_raw_parts(img, width * height);
    let output = std::slice::from_raw_parts_mut(out, width * height);
    let b = border.max(0) as usize;
    if 2 * b >= height || 2 * b >= width { return; } // empty interior, like the C int loop
    for row in b..height - b {
        for col in b..width - b {
            let i = row * width + col;
            output[i] = (test_dilate(input, i, width, radius) != 0) as u32;
        }
    }
}

/// Morphological erosion of the segmentation bitmap interior.
/// Replaces the DT_OMP_FOR(collapse(2)) loop in _eroding()
/// (segmentation.c:289).
///
/// # Safety
/// Same contract as `darkroom_segmentation_dilate` (taps reach ±(5*width+5)).
#[no_mangle]
pub unsafe extern "C" fn darkroom_segmentation_erode(
    img: *const u32, out: *mut u32,
    width: usize, height: usize, border: i32, radius: i32,
) {
    let input = std::slice::from_raw_parts(img, width * height);
    let output = std::slice::from_raw_parts_mut(out, width * height);
    let b = border.max(0) as usize;
    if 2 * b >= height || 2 * b >= width { return; }
    for row in b..height - b {
        for col in b..width - b {
            let i = row * width + col;
            output[i] = (test_erode(input, i, width, radius) != 0) as u32;
        }
    }
}

// ── Laplacian highlight reconstruction (src/iop/hlreconstruct/laplacian.c) ───

const RED: usize = 0;
const GREEN: usize = 1;
const BLUE: usize = 2;
const ALPHA: usize = 3;

// wavelets_scale_t bits
const FIRST_SCALE: u32 = 1 << 1;
const LAST_SCALE: u32 = 1 << 2;

/// B_SPLINE_TO_LAPLACIAN from src/common/bspline.h:39.
const B_SPLINE_TO_LAPLACIAN: f32 = 3.182_727_4;

#[inline(always)]
fn sqf(x: f32) -> f32 { x * x }

/// Bilinear CFA interpolation + per-channel clipping mask.
/// Replaces the DT_OMP_FOR loop in _interpolate_and_mask() (laplacian.c:53).
/// `interpolated`/`clipping_mask` are RGBA planes; channel 3 holds the
/// Euclidean norm / any-clipped flag respectively.
///
/// # Safety
/// `input` holds `width*height` floats; `interpolated`/`clipping_mask`
/// hold `width*height*4`; `clips`/`wb` hold 4 floats.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_interpolate_and_mask(
    input: *const f32, interpolated: *mut f32, clipping_mask: *mut f32,
    clips: *const f32, wb: *const f32,
    filters: u32, width: usize, height: usize,
) {
    let inp = std::slice::from_raw_parts(input, width * height);
    let interp = std::slice::from_raw_parts_mut(interpolated, width * height * 4);
    let cmask = std::slice::from_raw_parts_mut(clipping_mask, width * height * 4);
    let clips = std::slice::from_raw_parts(clips, 4);
    let wb = std::slice::from_raw_parts(wb, 4);

    for i in 0..height {
        for j in 0..width {
            let c = raw::fc_bayer(i as i32, j as i32, filters);
            let i_center = i * width;
            let center = inp[i_center + j];

            let (mut r, mut g, mut b) = (0.0_f32, 0.0_f32, 0.0_f32);
            let (r_clipped, g_clipped, b_clipped);

            if i == 0 || j == 0 || i == height - 1 || j == width - 1 {
                // image edges: no demosaic, R = G = B = center
                r = center; g = center; b = center;
                let cl = center > clips[c];
                r_clipped = cl; g_clipped = cl; b_clipped = cl;
            } else {
                let i_prev = (i - 1) * width;
                let i_next = (i + 1) * width;
                let (j_prev, j_next) = (j - 1, j + 1);

                let north = inp[i_prev + j];
                let south = inp[i_next + j];
                let west = inp[i_center + j_prev];
                let east = inp[i_center + j_next];
                let north_east = inp[i_prev + j_next];
                let north_west = inp[i_prev + j_prev];
                let south_east = inp[i_next + j_next];
                let south_west = inp[i_next + j_prev];

                if c == GREEN {
                    g = center;
                    g_clipped = center > clips[GREEN];
                } else {
                    // interpolate inside an X/Y cross
                    g = (north + south + east + west) / 4.0;
                    g_clipped = north > clips[GREEN] || south > clips[GREEN]
                        || east > clips[GREEN] || west > clips[GREEN];
                }

                let fc = |di: i32, dj: i32| raw::fc_bayer(i as i32 + di, j as i32 + dj, filters);
                if c == RED {
                    r = center;
                    r_clipped = center > clips[RED];
                } else if fc(-1, 0) == RED && fc(1, 0) == RED {
                    // red column → interpolate column-wise
                    r = (north + south) / 2.0;
                    r_clipped = north > clips[RED] || south > clips[RED];
                } else if fc(0, -1) == RED && fc(0, 1) == RED {
                    // red row → interpolate row-wise
                    r = (west + east) / 2.0;
                    r_clipped = west > clips[RED] || east > clips[RED];
                } else {
                    // blue row → interpolate inside a square
                    r = (north_west + north_east + south_east + south_west) / 4.0;
                    r_clipped = north_west > clips[RED] || north_east > clips[RED]
                        || south_west > clips[RED] || south_east > clips[RED];
                }

                if c == BLUE {
                    b = center;
                    b_clipped = center > clips[BLUE];
                } else if fc(-1, 0) == BLUE && fc(1, 0) == BLUE {
                    b = (north + south) / 2.0;
                    b_clipped = north > clips[BLUE] || south > clips[BLUE];
                } else if fc(0, -1) == BLUE && fc(0, 1) == BLUE {
                    b = (west + east) / 2.0;
                    b_clipped = west > clips[BLUE] || east > clips[BLUE];
                } else {
                    b = (north_west + north_east + south_east + south_west) / 4.0;
                    b_clipped = north_west > clips[BLUE] || north_east > clips[BLUE]
                        || south_west > clips[BLUE] || south_east > clips[BLUE];
                }
            }

            let rgb = [r, g, b, (sqf(r) + sqf(g) + sqf(b)).sqrt()];
            let any = r_clipped || g_clipped || b_clipped;
            let clipped = [
                r_clipped as u32 as f32, g_clipped as u32 as f32, b_clipped as u32 as f32,
                if any { 1.0 } else { 0.0 },
            ];
            let idx = (i * width + j) * 4;
            for k in 0..4 {
                interp[idx + k] = (rgb[k] / wb[k]).max(0.0);
                cmask[idx + k] = clipped[k];
            }
        }
    }
}

/// Remosaic the reconstructed RGBA plane back to the CFA and alpha-blend with
/// the original where unclipped. Replaces the DT_OMP_FOR loop in
/// _remosaic_and_replace() (laplacian.c:189).
///
/// # Safety
/// `input`/`output` hold `width*height` floats; `interpolated`/`clipping_mask`
/// hold `width*height*4`; `wb` holds 4 floats.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_remosaic_and_replace(
    input: *const f32, interpolated: *const f32, clipping_mask: *const f32,
    output: *mut f32, wb: *const f32,
    filters: u32, width: usize, height: usize,
) {
    let inp = std::slice::from_raw_parts(input, width * height);
    let interp = std::slice::from_raw_parts(interpolated, width * height * 4);
    let cmask = std::slice::from_raw_parts(clipping_mask, width * height * 4);
    let out = std::slice::from_raw_parts_mut(output, width * height);
    let wb = std::slice::from_raw_parts(wb, 4);

    for i in 0..height {
        for j in 0..width {
            let c = raw::fc_bayer(i as i32, j as i32, filters);
            let idx = i * width + j;
            let index = idx * 4;
            let opacity = cmask[index + ALPHA].clamp(0.0, 1.0);
            out[idx] = opacity * (interp[index + c] * wb[c]).max(0.0)
                + (1.0 - opacity) * inp[idx];
        }
    }
}

/// Gather the 3x3 ring of non-local (mult-strided) RGBA neighbours around
/// (i, j) into a contiguous array — the shared prologue of guide_laplacians
/// and heat_PDE_diffusion.
#[inline(always)]
fn gather_neighbours(
    hf: &[f32], i: usize, j: usize, width: usize, height: usize, mult: usize,
) -> [[f32; 4]; 9] {
    let i_n = [
        i.saturating_sub(mult) * width,
        i * width,
        (i + mult).min(height - 1) * width,
    ];
    let j_n = [j.saturating_sub(mult), j, (j + mult).min(width - 1)];
    let mut n = [[0.0_f32; 4]; 9];
    for (ki, &iv) in i_n.iter().enumerate() {
        for (kj, &jv) in j_n.iter().enumerate() {
            let base = 4 * (iv + jv);
            n[3 * ki + kj].copy_from_slice(&hf[base..base + 4]);
        }
    }
    n
}

/// Chromaticity laplacian guided by the most-detailed channel; one wavelet
/// scale of the RGB reconstruction. Replaces the DT_OMP_FOR loop in
/// guide_laplacians() (laplacian.c:218).
///
/// # Safety
/// `high_freq`/`low_freq`/`clipping_mask`/`output` hold `width*height*4`
/// floats; `mult >= 1`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_guide_laplacians(
    high_freq: *const f32, low_freq: *const f32, clipping_mask: *const f32,
    output: *mut f32, width: usize, height: usize,
    mult: i32, noise_level: f32, salt: i32, scale: u32, radius_sq: f32,
) {
    let hf = std::slice::from_raw_parts(high_freq, width * height * 4);
    let lf = std::slice::from_raw_parts(low_freq, width * height * 4);
    let cmask = std::slice::from_raw_parts(clipping_mask, width * height * 4);
    let out = std::slice::from_raw_parts_mut(output, width * height * 4);
    let mult = mult.max(1) as usize;

    const FLIP: [bool; 4] = [true, false, true, false];

    for row in 0..height {
        // interleave rows to minimise cache misses (matches the C scheduling;
        // result is row-order independent)
        let i = crate::math::dwt_interleave_rows(row, height, mult);
        for j in 0..width {
            let idx = i * width + j;
            let index = idx * 4;

            let alpha = cmask[index + ALPHA];
            let alpha_comp = 1.0 - cmask[index + ALPHA];

            let mut high_frequency = [hf[index], hf[index + 1], hf[index + 2], hf[index + 3]];

            if alpha > 0.0 {
                // reconstruct
                let neigh = gather_neighbours(hf, i, j, width, height, mult);

                let mut means = [0.0_f32; 4];
                for px in &neigh {
                    for c in 0..4 { means[c] += px[c] / 9.0; }
                }
                let mut variance = [0.0_f32; 4];
                for px in &neigh {
                    for c in 0..4 { variance[c] += sqf(px[c] - means[c]) / 9.0; }
                }

                // channel most likely to contain details = argmax variance
                let mut guide = ALPHA;
                let mut guiding_value = 0.0_f32;
                for c in 0..3 {
                    if variance[c] > guiding_value {
                        guiding_value = variance[c];
                        guide = c;
                    }
                }

                let mut covariance = [0.0_f32; 4];
                for px in &neigh {
                    for c in 0..4 {
                        covariance[c] += (px[c] - means[c]) * (px[guide] - means[guide]) / 9.0;
                    }
                }

                let scale_multiplier = 1.0 / radius_sq;
                let alpha_ch = [
                    cmask[index + RED], cmask[index + GREEN],
                    cmask[index + BLUE], cmask[index + ALPHA],
                ];

                // snapshot: the C for_each_channel loop is a 4-lane SIMD
                // update — all lanes read high_frequency[guide] from the same
                // pre-update register. A sequential loop without this snapshot
                // would use a partially-updated value when c > guide.
                let hf_guide = high_frequency[guide];
                for c in 0..4 {
                    let a = (covariance[c] / variance[guide]).max(0.0);
                    let b = means[c] - a * means[guide];
                    high_frequency[c] = alpha_ch[c] * scale_multiplier * (a * hf_guide + b)
                        + (1.0 - alpha_ch[c] * scale_multiplier) * high_frequency[c];
                }
            }

            if scale & FIRST_SCALE != 0 {
                for c in 0..4 { out[index + c] = high_frequency[c]; }
            } else {
                for c in 0..4 { out[index + c] += high_frequency[c]; }
            }

            if scale & LAST_SCALE != 0 {
                for c in 0..4 { out[index + c] = (out[index + c] + lf[index + c]).max(0.0); }
            }

            // last step of RGB reconstruct: add noise
            if scale & LAST_SCALE != 0 && salt != 0 && alpha > 0.0 {
                let mut state = [
                    crate::math::splitmix32((j + 1) as u64),
                    crate::math::splitmix32(((j + 1) * (i + 3)) as u64),
                    crate::math::splitmix32(1337),
                    crate::math::splitmix32(666),
                ];
                for _ in 0..4 { crate::math::xoshiro128plus(&mut state); }

                let mu = [out[index], out[index + 1], out[index + 2], out[index + 3]];
                let mut sigma = [0.0_f32; 4];
                for c in 0..4 { sigma[c] = out[index + c] * noise_level; }

                let noise = crate::math::dt_noise_generator_4ch(2 /*POISSONIAN*/, &mu, &sigma, &FLIP, &mut state);
                for c in 0..4 {
                    // noise only brightens the image, since it's clipped
                    let n = out[index + c] + (noise[c] - out[index + c]).abs();
                    out[index + c] = (alpha * n + alpha_comp * out[index + c]).max(0.0);
                }
            }

            if scale & LAST_SCALE != 0 {
                // break RGB into ratios + norm for the next reconstruction step
                let norm = (sqf(out[index + RED]) + sqf(out[index + GREEN]) + sqf(out[index + BLUE]))
                    .sqrt()
                    .max(1e-6);
                for c in 0..4 { out[index + c] /= norm; }
                out[index + ALPHA] = norm;
            }
        }
    }
}

/// Anisotropic heat-transfer (PDE) diffusion of the chromaticity ratios; one
/// wavelet scale. Replaces the DT_OMP_FOR loop in heat_PDE_diffusion()
/// (laplacian.c:402).
///
/// # Safety
/// Same buffer contract as `darkroom_highlights_guide_laplacians`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_highlights_heat_pde_diffusion(
    high_freq: *const f32, low_freq: *const f32, clipping_mask: *const f32,
    output: *mut f32, width: usize, height: usize,
    mult: i32, scale: u32, first_order_factor: f32,
) {
    let hf = std::slice::from_raw_parts(high_freq, width * height * 4);
    let lf = std::slice::from_raw_parts(low_freq, width * height * 4);
    let cmask = std::slice::from_raw_parts(clipping_mask, width * height * 4);
    let out = std::slice::from_raw_parts_mut(output, width * height * 4);
    let mult = mult.max(1) as usize;

    // laplacian in the direction parallel to the steepest gradient on the norm
    const KERNEL: [f32; 9] = [0.25, 0.5, 0.25, 0.5, -3.0, 0.5, 0.25, 0.5, 0.25];

    for row in 0..height {
        let i = crate::math::dwt_interleave_rows(row, height, mult);
        for j in 0..width {
            let idx = i * width + j;
            let index = idx * 4;

            let alpha = [
                cmask[index + RED], cmask[index + GREEN],
                cmask[index + BLUE], cmask[index + ALPHA],
            ];
            let mut high_frequency = [hf[index], hf[index + 1], hf[index + 2], hf[index + 3]];
            // don't diffuse the norm: store and restore channel 3
            let norm_backup = high_frequency[3];

            if alpha[ALPHA] > 0.0 {
                // reconstruct
                let neigh = gather_neighbours(hf, i, j, width, height, mult);

                let mut laplacian = [0.0_f32; 4];
                for (k, px) in neigh.iter().enumerate() {
                    for c in 0..4 { laplacian[c] += px[c] * KERNEL[k]; }
                }

                let multipliers = [
                    1.0 / B_SPLINE_TO_LAPLACIAN, 1.0 / B_SPLINE_TO_LAPLACIAN,
                    1.0 / B_SPLINE_TO_LAPLACIAN, 0.0,
                ];
                for c in 0..4 {
                    high_frequency[c] +=
                        alpha[c] * multipliers[c] * (laplacian[c] - first_order_factor * high_frequency[c]);
                }
                high_frequency[3] = norm_backup;
            }

            if scale & FIRST_SCALE != 0 {
                for c in 0..4 { out[index + c] = high_frequency[c]; }
            } else {
                for c in 0..4 { out[index + c] += high_frequency[c]; }
            }

            if scale & LAST_SCALE != 0 {
                // add the residual and clamp
                for c in 0..4 { out[index + c] = (out[index + c] + lf[index + c]).max(0.0); }

                // renormalize ratios
                if alpha[ALPHA] > 0.0 {
                    let norm =
                        (sqf(out[index + RED]) + sqf(out[index + GREEN]) + sqf(out[index + BLUE])).sqrt();
                    for c in 0..4 {
                        if c != ALPHA && norm > 1e-4 { out[index + c] /= norm; }
                    }
                }

                // reconstruct RGB from ratios and norm — norm stays in channel 3
                for c in 0..3 { out[index + c] *= out[index + ALPHA]; }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Segmentation-based reconstruction (src/iop/hlreconstruct/segbased.c).
// The dt_iop_segmentation_t struct and the flood-fill machinery stay in C;
// these functions replace the DT_OMP_FOR loops and receive the struct fields
// (data / val1 / val2 / nr / width / height / border) individually.
// ──────────────────────────────────────────────────────────────────────────

/// Segment-id bit mask, segmentation.c:31.
const SEG_ID_MASK: u32 = 0x40000;
/// Outer plane border of the segmentation working planes, segbased.c:89.
const HL_BORDER: usize = 8;

/// Matches _get_segment_id() (segmentation.c:95): 0 outside the
/// segmentized region or when the masked id is not a real segment (2..nr-1).
#[inline(always)]
fn get_segment_id(data: &[u32], width: usize, height: usize, border: usize, nr: u32, loc: usize) -> u32 {
    if loc >= width * (height - border) {
        return 0;
    }
    let id = data[loc] & (SEG_ID_MASK - 1);
    if id > 1 && id < nr { id } else { 0 }
}

/// Plane index of a raw photosite (3x3 superpixels, HL_BORDER inset).
/// Matches _raw_to_plane() (segbased.c:414).
#[inline(always)]
fn raw_to_plane(pwidth: usize, row: usize, col: usize) -> usize {
    (HL_BORDER + row / 3) * pwidth + col / 3 + HL_BORDER
}

/// Seed the gradient plane from the blurred luminance at distance 0..2.
/// Replaces the DT_OMP_FOR loop in _initial_gradients() (segbased.c:230).
///
/// # Safety
/// `luminance`, `distance` and `gradient` hold `pwidth * pheight` floats;
/// pwidth/pheight >= 2*(HL_BORDER+2).
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_initial_gradients(
    luminance: *const f32, distance: *const f32, gradient: *mut f32,
    pwidth: usize, pheight: usize,
) {
    let lum = std::slice::from_raw_parts(luminance, pwidth * pheight);
    let dist = std::slice::from_raw_parts(distance, pwidth * pheight);
    let grad = std::slice::from_raw_parts_mut(gradient, pwidth * pheight);

    for row in (HL_BORDER + 2)..(pheight - HL_BORDER - 2) {
        for col in (HL_BORDER + 2)..(pwidth - HL_BORDER - 2) {
            let v = row * pwidth + col;
            let mut g = 0.0_f32;
            if dist[v] > 0.0 && dist[v] < 2.0 {
                g = 4.0 * crate::math::scharr_gradient(lum, v, pwidth);
            }
            grad[v] = g;
        }
    }
}

/// Maximum distance-transform value inside segment `id` over the (caller
/// clamped) bounding box. Replaces the DT_OMP_FOR(reduction(max)) loop in
/// _segment_maxdistance() (segbased.c:254).
///
/// # Safety
/// `distance` and `seg_data` hold `seg_width * seg_height` elements;
/// 0 <= xmin <= xmax <= seg_width, same for y.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_maxdistance(
    distance: *const f32, seg_data: *const u32,
    seg_width: usize, seg_height: usize,
    xmin: i32, xmax: i32, ymin: i32, ymax: i32, id: u32,
) -> f32 {
    if xmax <= xmin || ymax <= ymin { return 0.0; }
    let dist = std::slice::from_raw_parts(distance, seg_width * seg_height);
    let data = std::slice::from_raw_parts(seg_data, seg_width * seg_height);

    let mut max_distance = 0.0_f32;
    for row in ymin as usize..ymax as usize {
        for col in xmin as usize..xmax as usize {
            let v = row * seg_width + col;
            if id == data[v] {
                max_distance = max_distance.max(dist[v]);
            }
        }
    }
    max_distance
}

/// One distance ring of the iterative gradient propagation: cells of segment
/// `id` with distance in [dist, dist+1.5) average the gradients of their 5x5
/// neighbours one ring closer ([dist-1.5, dist)). Replaces the DT_OMP_FOR
/// loop in _calc_distance_ring() (segbased.c:299).
///
/// # Safety
/// `gradient`, `distance`, `seg_data` hold `seg_width * seg_height` elements;
/// the bounds are inset by seg->border >= 2 on every side (the C callers
/// guarantee this), so the ±2 neighbourhood stays in bounds.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_distance_ring(
    gradient: *mut f32, distance: *const f32, seg_data: *const u32,
    seg_width: usize, seg_height: usize,
    xmin: i32, xmax: i32, ymin: i32, ymax: i32,
    attenuate: f32, dist: f32, id: u32,
) {
    if xmax <= xmin || ymax <= ymin { return; }
    let grad = std::slice::from_raw_parts_mut(gradient, seg_width * seg_height);
    let dst = std::slice::from_raw_parts(distance, seg_width * seg_height);
    let data = std::slice::from_raw_parts(seg_data, seg_width * seg_height);

    for row in ymin as usize..ymax as usize {
        for col in xmin as usize..xmax as usize {
            let v = row * seg_width + col;
            let dv = dst[v];
            if dv >= dist && dv < dist + 1.5 && id == data[v] {
                let mut grd = 0.0_f32;
                let mut cnt = 0.0_f32;
                for y in -2_isize..3 {
                    for x in -2_isize..3 {
                        let p = (v as isize + x + seg_width as isize * y) as usize;
                        let dd = dst[p];
                        if dd >= dist - 1.5 && dd < dist {
                            cnt += 1.0;
                            grd += grad[p];
                        }
                    }
                }
                if cnt > 0.0 {
                    grad[v] = (1.5_f32).min((grd / cnt) * (1.0 + 1.0 / dv.powf(attenuate)));
                }
            }
        }
    }
}

/// Copy the segment bounding box of `gradient` into the densely packed `tmp`
/// buffer for box-blurring. Replaces the first DT_OMP_FOR loop of the
/// maxdist > 4 branch in _segment_gradients() (segbased.c:354).
///
/// # Safety
/// `gradient` holds `seg_width * ymax` floats at least; `tmp` holds
/// `(ymax-ymin) * (xmax-xmin)` floats; 0 <= xmin < xmax, 0 <= ymin < ymax.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_box_in(
    gradient: *const f32, tmp: *mut f32,
    seg_width: usize, xmin: i32, xmax: i32, ymin: i32, ymax: i32,
) {
    if xmax <= xmin || ymax <= ymin { return; }
    let (xmin, xmax, ymin, ymax) = (xmin as usize, xmax as usize, ymin as usize, ymax as usize);
    let bw = xmax - xmin;
    let grad = std::slice::from_raw_parts(gradient, seg_width * ymax);
    let tmp = std::slice::from_raw_parts_mut(tmp, (ymax - ymin) * bw);

    for row in ymin..ymax {
        for col in xmin..xmax {
            tmp[(row - ymin) * bw + (col - xmin)] = grad[row * seg_width + col];
        }
    }
}

/// Copy the blurred box back into `gradient`, only at locations belonging to
/// segment `id`. Replaces the second DT_OMP_FOR loop of the maxdist > 4
/// branch in _segment_gradients() (segbased.c:363).
///
/// # Safety
/// Same buffers as darkroom_segbased_box_in, plus `seg_data` of
/// `seg_width * ymax` elements.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_box_out(
    gradient: *mut f32, tmp: *const f32, seg_data: *const u32,
    seg_width: usize, xmin: i32, xmax: i32, ymin: i32, ymax: i32, id: u32,
) {
    if xmax <= xmin || ymax <= ymin { return; }
    let (xmin, xmax, ymin, ymax) = (xmin as usize, xmax as usize, ymin as usize, ymax as usize);
    let bw = xmax - xmin;
    let grad = std::slice::from_raw_parts_mut(gradient, seg_width * ymax);
    let tmp = std::slice::from_raw_parts(tmp, (ymax - ymin) * bw);
    let data = std::slice::from_raw_parts(seg_data, seg_width * ymax);

    for row in ymin..ymax {
        for col in xmin..xmax {
            let v = row * seg_width + col;
            if id == data[v] {
                grad[v] = tmp[(row - ymin) * bw + (col - xmin)];
            }
        }
    }
}

/// Scale the gradients of segment `id` by the recovery strength. Replaces
/// the final DT_OMP_FOR loop in _segment_gradients() (segbased.c:374).
///
/// # Safety
/// `gradient` and `seg_data` hold `seg_width * ymax` elements at least.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_apply_strength(
    gradient: *mut f32, seg_data: *const u32,
    seg_width: usize, xmin: i32, xmax: i32, ymin: i32, ymax: i32,
    id: u32, strength: f32,
) {
    if xmax <= xmin || ymax <= ymin { return; }
    let grad = std::slice::from_raw_parts_mut(gradient, seg_width * ymax as usize);
    let data = std::slice::from_raw_parts(seg_data, seg_width * ymax as usize);

    for row in ymin as usize..ymax as usize {
        for col in xmin as usize..xmax as usize {
            let v = row * seg_width + col;
            if id == data[v] {
                grad[v] *= strength;
            }
        }
    }
}

/// Replicate the inner border of a single-channel mask outward: rows first
/// (left/right columns), then columns (top/bottom rows, with the source
/// column clamped to the interior). Replaces both DT_OMP_FOR loops in
/// _masks_extend_border() (segbased.c:425/435).
///
/// # Safety
/// `mask` holds `width * height` floats; border < min(width, height)/2.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_extend_border(
    mask: *mut f32, width: usize, height: usize, border: i32,
) {
    if border <= 0 { return; }
    let border = border as usize;
    let m = std::slice::from_raw_parts_mut(mask, width * height);

    for row in border..height - border {
        let idx = row * width;
        for i in 0..border {
            m[idx + i] = m[idx + border];
            m[idx + width - i - 1] = m[idx + width - border - 1];
        }
    }
    for col in 0..width {
        let src = (width - border - 1).min(col.max(border));
        let top = m[border * width + src];
        let bot = m[(height - border - 1) * width + src];
        for i in 0..border {
            m[col + i * width] = top;
            m[col + (height - i - 1) * width] = bot;
        }
    }
}

/// Populate the 3x3-superpixel colour planes, the cube-root opposed-channel
/// refavg planes and the per-plane clipping seeds from the mosaic. Returns
/// `anyclipped` (sum of clipped channel counts) and sets `*has_allclipped`
/// when any superpixel clips in all three planes. Replaces the
/// DT_OMP_FOR(reduction(|)+reduction(+)) loop in _process_segmentation()
/// (segbased.c:520).
///
/// # Safety
/// `tmpout` holds `width * height` floats; `planes`/`refavgs` point to 3 and
/// `seg_datas` to 4 plane pointers of `pwidth * pheight` elements each (all
/// distinct); `correction`/`cube_coeffs` hold 4 floats; `xtrans` 36 bytes;
/// `has_allclipped` is a valid int pointer.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_populate_planes(
    tmpout: *const f32, width: usize, height: usize,
    filters: u32, xtrans: *const u8,
    correction: *const f32, cube_coeffs: *const f32, xshifter: i32,
    planes: *const *mut f32, refavgs: *const *mut f32, seg_datas: *const *mut u32,
    pwidth: usize, pheight: usize,
    has_allclipped: *mut i32,
) -> i32 {
    let tmp = std::slice::from_raw_parts(tmpout, width * height);
    let xt = read_xtrans(xtrans);
    let corr = std::slice::from_raw_parts(correction, 4);
    let cube = std::slice::from_raw_parts(cube_coeffs, 4);
    let psize = pwidth * pheight;
    let planes = std::slice::from_raw_parts(planes, 3);
    let refavgs = std::slice::from_raw_parts(refavgs, 3);
    let seg_datas = std::slice::from_raw_parts(seg_datas, 4);
    let mut plane: Vec<&mut [f32]> =
        planes.iter().map(|&p| std::slice::from_raw_parts_mut(p, psize)).collect();
    let mut refavg: Vec<&mut [f32]> =
        refavgs.iter().map(|&p| std::slice::from_raw_parts_mut(p, psize)).collect();
    let mut segs: Vec<&mut [u32]> =
        seg_datas.iter().map(|&p| std::slice::from_raw_parts_mut(p, psize)).collect();
    let xshifter = xshifter as usize;

    let mut anyclipped = 0_i32;
    let mut allclipped_any = false;

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            // calc all color planes in a 3x3 area. For chroma noise stability in
            // bayer sensors we make sure to align the box with a green photosite
            // in centre so we always have a 5:2:2 ratio
            if col % 3 == xshifter && row % 3 == 1 {
                let mut mean = [0.0_f32; 3];
                let mut cnt = [0.0_f32; 3];
                for dy in row - 1..row + 2 {
                    for dx in col - 1..col + 2 {
                        let val = tmp[dy * width + dx];
                        let c = raw::fcol(dy as i32, dx as i32, filters, &xt);
                        mean[c] += val;
                        cnt[c] += 1.0;
                    }
                }
                for c in 0..3 {
                    mean[c] = if cnt[c] > 0.0 { (corr[c] * mean[c] / cnt[c]).cbrt() } else { 0.0 };
                }
                let cube_refavg = [
                    0.5 * (mean[1] + mean[2]),
                    0.5 * (mean[0] + mean[2]),
                    0.5 * (mean[0] + mean[1]),
                ];

                let o = raw_to_plane(pwidth, row, col);
                let mut allclipped = 0;
                for c in 0..3 {
                    plane[c][o] = mean[c];
                    refavg[c][o] = cube_refavg[c];
                    if mean[c] > cube[c] {
                        allclipped += 1;
                        segs[c][o] = 1;
                    }
                }
                segs[3][o] = if allclipped == 3 { 1 } else { 0 };
                allclipped_any |= allclipped == 3;
                anyclipped += allclipped;
            }
        }
    }
    *has_allclipped = allclipped_any as i32;
    anyclipped
}

/// Inpaint clipped photosites from their segment's candidate: the local
/// cube-root refavg shifted by (candidate - candidate_reference), cubed back
/// to linear. Writes both the raw `tmpout` and the colour plane. Replaces
/// the DT_OMP_FOR loop after _calc_plane_candidates() in
/// _process_segmentation() (segbased.c:594).
///
/// # Safety
/// `input`/`tmpout` hold `width * height` floats; `planes` points to 3 plane
/// pointers and `seg_datas`/`seg_val1s`/`seg_val2s` to 3 buffer pointers per
/// colour (data: pwidth*pheight, val1/val2: >= seg_nrs[c] floats);
/// `seg_nrs` holds 3 ints; `clips`/`correction` hold 4 floats; `xtrans` 36 bytes.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_candidates_apply(
    input: *const f32, tmpout: *mut f32, width: usize, height: usize,
    filters: u32, xtrans: *const u8,
    clips: *const f32, correction: *const f32,
    planes: *const *mut f32,
    seg_datas: *const *const u32, seg_val1s: *const *const f32, seg_val2s: *const *const f32,
    seg_nrs: *const i32,
    pwidth: usize, pheight: usize, seg_border: i32,
) {
    let inp = std::slice::from_raw_parts(input, width * height);
    let tmp = std::slice::from_raw_parts_mut(tmpout, width * height);
    let xt = read_xtrans(xtrans);
    let clips = std::slice::from_raw_parts(clips, 4);
    let corr_s = std::slice::from_raw_parts(correction, 4);
    let corr = [corr_s[0], corr_s[1], corr_s[2], corr_s[3]];
    let psize = pwidth * pheight;
    let planes = std::slice::from_raw_parts(planes, 3);
    let mut plane: Vec<&mut [f32]> =
        planes.iter().map(|&p| std::slice::from_raw_parts_mut(p, psize)).collect();
    let nrs = std::slice::from_raw_parts(seg_nrs, 3);
    let datas: Vec<&[u32]> = std::slice::from_raw_parts(seg_datas, 3)
        .iter().map(|&p| std::slice::from_raw_parts(p, psize)).collect();
    let val1s: Vec<&[f32]> = std::slice::from_raw_parts(seg_val1s, 3)
        .iter().enumerate().map(|(c, &p)| std::slice::from_raw_parts(p, nrs[c].max(0) as usize)).collect();
    let val2s: Vec<&[f32]> = std::slice::from_raw_parts(seg_val2s, 3)
        .iter().enumerate().map(|(c, &p)| std::slice::from_raw_parts(p, nrs[c].max(0) as usize)).collect();
    let border = seg_border.max(0) as usize;

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let idx = row * width + col;
            let inval = inp[idx].max(0.0);
            let color = raw::fcol(row as i32, col as i32, filters, &xt);
            if inval > clips[color] {
                let o = raw_to_plane(pwidth, row, col);
                let nr = nrs[color] as u32;
                let pid = get_segment_id(datas[color], pwidth, pheight, border, nr, o);
                if pid > 1 && pid < nr {
                    let candidate = val1s[color][pid as usize];
                    if candidate != 0.0 {
                        let cand_reference = val2s[color][pid as usize];
                        let refavg_here =
                            calc_refavg(inp, &xt, filters, row, col, width, height, &corr, false);
                        let oval = fcube(refavg_here + candidate - cand_reference);
                        let v = inval.max(oval);
                        plane[color][o] = v;
                        tmp[idx] = v;
                    }
                }
            }
        }
    }
}

/// Prepare the recovery working planes: temporary luminance (coeff-weighted
/// plane mean) and the distance-transform seed (DT_DISTANCE_TRANSFORM_MAX
/// inside all-clipped superpixels). Replaces the DT_OMP_FOR loop in the
/// do_recovery||do_masking block of _process_segmentation() (segbased.c:637).
///
/// # Safety
/// `plane0..2`, `tmp`, `distance` and `segall_data` hold `pwidth * pheight`
/// elements; `icoeffs` holds 3 floats; border < min(pwidth, pheight)/2.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_prepare_lumdist(
    plane0: *const f32, plane1: *const f32, plane2: *const f32,
    icoeffs: *const f32, tmp: *mut f32, distance: *mut f32,
    segall_data: *const u32, pwidth: usize, pheight: usize, border: i32,
) {
    const DISTANCE_TRANSFORM_MAX: f32 = 1e20;
    let psize = pwidth * pheight;
    let p0 = std::slice::from_raw_parts(plane0, psize);
    let p1 = std::slice::from_raw_parts(plane1, psize);
    let p2 = std::slice::from_raw_parts(plane2, psize);
    let ic = std::slice::from_raw_parts(icoeffs, 3);
    let tmp = std::slice::from_raw_parts_mut(tmp, psize);
    let dist = std::slice::from_raw_parts_mut(distance, psize);
    let data = std::slice::from_raw_parts(segall_data, psize);
    let border = border.max(0) as usize;

    for row in border..pheight - border {
        for col in border..pwidth - border {
            let i = row * pwidth + col;
            // prepare the temporary luminance for later blurring and also
            // prefill the distance plane
            tmp[i] = (p0[i] * ic[0] + p1[i] * ic[1] + p2[i] * ic[2]) / 3.0;
            dist[i] = if data[i] == 1 { DISTANCE_TRANSFORM_MAX } else { 0.0 };
        }
    }
}

/// Add the recovered gradient back to clipped photosites, sigmoid-attenuated
/// by the distance transform. Replaces the DT_OMP_FOR loop at the end of the
/// do_recovery block in _process_segmentation() (segbased.c:684).
///
/// # Safety
/// `input`/`tmpout` hold `width * height` floats; `distance`/`gradient`
/// hold `pwidth * pheight` floats (pheight implied by raw_to_plane bounds);
/// `clips` holds 4 floats; `xtrans` 36 bytes.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_apply_recovery(
    input: *const f32, tmpout: *mut f32, width: usize, height: usize,
    filters: u32, xtrans: *const u8, clips: *const f32,
    distance: *const f32, gradient: *const f32,
    pwidth: usize, pheight: usize,
    strength: f32, dshift: f32,
) {
    let inp = std::slice::from_raw_parts(input, width * height);
    let tmp = std::slice::from_raw_parts_mut(tmpout, width * height);
    let xt = read_xtrans(xtrans);
    let clips = std::slice::from_raw_parts(clips, 4);
    let psize = pwidth * pheight;
    let dist = std::slice::from_raw_parts(distance, psize);
    let grad = std::slice::from_raw_parts(gradient, psize);

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let idx = row * width + col;
            let color = raw::fcol(row as i32, col as i32, filters, &xt);
            let ival = inp[idx].max(0.0);
            if ival > clips[color] {
                let o = raw_to_plane(pwidth, row, col);
                let effect = strength / (1.0 + (-(dist[o] - dshift)).exp());
                tmp[idx] += (grad[o] * effect).max(0.0);
            }
        }
    }
}

/// Final output loop: crop/copy `tmpout` into the output ROI, or — in mask
/// visualizing modes — the dimmed luminance plus the requested overlay
/// (combined segments, candidates, or strength-weighted gradient). Replaces
/// the last DT_OMP_FOR loop in _process_segmentation() (segbased.c:703).
///
/// # Safety
/// `output` holds `out_width * out_height` floats; `tmpout` holds
/// `in_width * in_height` floats; `luminance`/`gradient` hold
/// `pwidth * pheight` floats; `seg_datas`/`seg_val1s` point to 3 buffer
/// pointers (data: pwidth*pheight, val1: >= seg_nrs[c] floats); `seg_nrs`
/// holds 3 ints; `segall_data` holds pwidth*pheight elements; `xtrans` 36 bytes.
#[no_mangle]
pub unsafe extern "C" fn darkroom_segbased_final_output(
    output: *mut f32, tmpout: *const f32,
    luminance: *const f32, gradient: *const f32,
    out_width: usize, out_height: usize, out_x: i32, out_y: i32,
    in_width: usize, in_height: usize,
    filters: u32, xtrans: *const u8,
    seg_datas: *const *const u32, seg_val1s: *const *const f32, seg_nrs: *const i32,
    segall_data: *const u32, segall_nr: i32,
    pwidth: usize, pheight: usize, seg_border: i32,
    do_masking: i32, vmode: i32, strength: f32,
) {
    // dt_highlights_mask_t (src/iop/highlights.c:96)
    const MASK_COMBINE: i32 = 1;
    const MASK_CANDIDATING: i32 = 2;
    const MASK_STRENGTH: i32 = 3;

    let out = std::slice::from_raw_parts_mut(output, out_width * out_height);
    let tmp = std::slice::from_raw_parts(tmpout, in_width * in_height);
    let psize = pwidth * pheight;
    let lum = std::slice::from_raw_parts(luminance, psize);
    let grad = std::slice::from_raw_parts(gradient, psize);
    let xt = read_xtrans(xtrans);
    let nrs = std::slice::from_raw_parts(seg_nrs, 3);
    let datas: Vec<&[u32]> = std::slice::from_raw_parts(seg_datas, 3)
        .iter().map(|&p| std::slice::from_raw_parts(p, psize)).collect();
    let val1s: Vec<&[f32]> = std::slice::from_raw_parts(seg_val1s, 3)
        .iter().enumerate().map(|(c, &p)| std::slice::from_raw_parts(p, nrs[c].max(0) as usize)).collect();
    let alldata = std::slice::from_raw_parts(segall_data, psize);
    let border = seg_border.max(0) as usize;
    let do_masking = do_masking != 0;

    for row in 0..out_height {
        for col in 0..out_width {
            let inrow = row as i32 + out_y;
            let incol = col as i32 + out_x;
            let odx = row * out_width + col;

            if inrow >= 0 && (inrow as usize) < in_height && incol >= 0 && (incol as usize) < in_width {
                let (inrow, incol) = (inrow as usize, incol as usize);
                let ppos = raw_to_plane(pwidth, inrow, incol);
                let idx = inrow * in_width + incol;

                out[odx] = if do_masking { (0.2_f32).min(0.2 * lum[ppos]) } else { tmp[idx] };
                if do_masking && inrow > 0 && incol > 0 && inrow < in_height - 1 && incol < in_width - 1 {
                    let color = raw::fcol(inrow as i32, incol as i32, filters, &xt);
                    let pid = get_segment_id(datas[color], pwidth, pheight, border, nrs[color] as u32, ppos);

                    if vmode == MASK_COMBINE && pid != 0 {
                        out[odx] += if datas[color][ppos] & SEG_ID_MASK != 0 { 1.0 } else { 0.6 };
                    } else if vmode == MASK_CANDIDATING {
                        // C: pid && !feqf(val1[pid], 0.0f, 1e-9)
                        if pid != 0 && val1s[color][pid as usize].abs() >= 1e-9 {
                            out[odx] += 1.0;
                        }
                    } else if vmode == MASK_STRENGTH {
                        let allid = get_segment_id(alldata, pwidth, pheight, border, segall_nr as u32, ppos);
                        let allseg = allid > 1 && allid < segall_nr as u32;
                        out[odx] += if allseg { strength * grad[ppos] } else { 0.0 };
                    }
                }
            } else {
                out[odx] = 0.0;
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
    fn calc_linear_refavg_grey_is_identity() {
        // equal channels → opposing means equal the channel itself
        let pix = [0.343_f32, 0.343, 0.343, 0.0];
        for c in 0..3 {
            assert!((calc_linear_refavg(&pix, c) - 0.343).abs() < 1e-5);
        }
    }

    #[test]
    fn calc_refavg_uniform_raw_is_identity() {
        // uniform mosaic: every per-colour mean is v → refavg = v (correction=1)
        let (w, h) = (8usize, 8usize);
        let v = 0.62_f32;
        let input = vec![v; w * h];
        let corr = [1.0_f32; 4];
        let xt = [[0u8; 6]; 6]; // unused for bayer
        let r = calc_refavg(&input, &xt, RGGB, 4, 4, w, h, &corr, true);
        assert!((r - v).abs() < 1e-5, "r={r}");
    }

    #[test]
    fn opposed_sraw_mask_and_output_roundtrip() {
        // 12x12 RGBA frame, one clipped R pixel in the middle.
        let (w, h) = (12usize, 12usize);
        let (mw, mh) = (w / 3, h / 3);
        let r4 = |x: usize| (x + 3) & !3; // dt_round_size(x, 4)
        let msize = r4(mw) * r4(mh);

        let clips = [1.0_f32, 1.0, 1.0];
        let mut input = vec![0.4_f32; w * h * 4];
        let cidx = (6 * w + 6) * 4;
        input[cidx] = 2.0; // clipped R at (6,6)

        let mut mask = vec![0_u8; 6 * msize];
        let any = unsafe {
            darkroom_highlights_opposed_mask_sraw(
                input.as_ptr(), mask.as_mut_ptr(), w, h, mw, mh, msize, clips.as_ptr(),
            )
        };
        assert_eq!(any, 1);
        // (6,6) maps to cmap (2,2); channel 0 — but note C tests input[idx]
        // (channel 0) against all three clips, so all 3 channels get flagged.
        assert_eq!(mask[2 * mw + 2], 1);

        // output: unclipped pixels pass through, the clipped one >= original
        let mut out = vec![0.0_f32; w * h * 4];
        let chroma = [0.0_f32; 3];
        unsafe {
            darkroom_highlights_opposed_output_sraw(
                input.as_ptr(), out.as_mut_ptr(), w * h, clips.as_ptr(), chroma.as_ptr(),
            );
        }
        assert!((out[4] - 0.4).abs() < 1e-6); // ordinary pixel untouched
        assert!(out[cidx] >= 2.0 - 1e-6); // clipped stays at least the input
    }

    #[test]
    fn opposed_dilate_raw_copies_borders_and_dilates_interior() {
        let (mw, mh) = (16usize, 16usize);
        let msize = mw * mh; // already multiples of 4
        let mut mask = vec![0_u8; 6 * msize];
        mask[8 * mw + 8] = 1; // single set cell, channel 0
        mask[0] = 1; // border cell, channel 0
        unsafe {
            darkroom_highlights_opposed_dilate_raw(mask.as_mut_ptr(), mw, mh, msize);
        }
        // border copies source
        assert_eq!(mask[3 * msize], 1);
        // dilation spreads to taps within the pattern (e.g. (8,5) is a -3 tap)
        assert_eq!(mask[3 * msize + 8 * mw + 5], 1);
        assert_eq!(mask[3 * msize + 5 * mw + 8], 1); // -3 rows tap
        assert_eq!(mask[3 * msize + 8 * mw + 8], 1); // itself
        // far cell untouched
        assert_eq!(mask[3 * msize + 12 * mw + 14], 0);
    }

    #[test]
    fn opposed_raw_output_crops_and_zeroes_outside() {
        // 8x8 input, 4x4 output at offset (6,6): rows/cols 6..9 — 8,9 out of range → 0
        let (iw, ih) = (8usize, 8usize);
        let input: Vec<f32> = (0..iw * ih).map(|k| k as f32 * 0.001).collect();
        let (ow, oh) = (4usize, 4usize);
        let mut out = vec![9.0_f32; ow * oh];
        let clips = [10.0_f32; 3]; // nothing clipped
        let chroma = [0.0_f32; 3];
        let corr = [1.0_f32; 4];
        let xt = [[0u8; 6]; 6];
        unsafe {
            darkroom_highlights_opposed_output_raw(
                input.as_ptr(), std::ptr::null(), out.as_mut_ptr(),
                ow, oh, 6, 6, iw, ih,
                RGGB, xt.as_ptr() as *const u8, clips.as_ptr(), chroma.as_ptr(), corr.as_ptr(),
            );
        }
        // (0,0) of out = input(6,6); (2,2) of out = input(8,8) → out of range → 0
        assert!((out[0] - input[6 * iw + 6]).abs() < 1e-6);
        assert_eq!(out[2 * ow + 2], 0.0);
        assert_eq!(out[3 * ow + 3], 0.0);
    }

    #[test]
    fn segmentation_dilate_spreads_single_pixel() {
        // single set pixel at the centre of a 32x32 map, border 9, radius 3:
        // every interior pixel within ring distance 3 turns on.
        let (w, h) = (32usize, 32usize);
        let mut img = vec![0_u32; w * h];
        img[16 * w + 16] = 1;
        let mut out = vec![7_u32; w * h];
        unsafe {
            darkroom_segmentation_dilate(img.as_ptr(), out.as_mut_ptr(), w, h, 9, 3);
        }
        assert_eq!(out[16 * w + 16], 1); // itself (ring 1 includes centre)
        assert_eq!(out[16 * w + 13], 1); // -3 col tap (ring 3)
        assert_eq!(out[13 * w + 16], 1); // -3 row tap
        assert_eq!(out[16 * w + 12], 0); // distance 4 — beyond radius 3
        assert_eq!(out[0], 7); // border untouched
    }

    #[test]
    fn segmentation_erode_requires_full_neighbourhood() {
        // all-ones map: erosion keeps interior 1. Poke one hole: pixels whose
        // ring includes the hole go 0.
        let (w, h) = (32usize, 32usize);
        let mut img = vec![1_u32; w * h];
        let mut out = vec![7_u32; w * h];
        unsafe {
            darkroom_segmentation_erode(img.as_ptr(), out.as_mut_ptr(), w, h, 9, 2);
        }
        assert_eq!(out[16 * w + 16], 1);

        img[16 * w + 16] = 0; // hole
        unsafe {
            darkroom_segmentation_erode(img.as_ptr(), out.as_mut_ptr(), w, h, 9, 2);
        }
        assert_eq!(out[16 * w + 16], 0); // centre sees its own hole
        assert_eq!(out[16 * w + 17], 0); // ring-1 neighbour sees the hole
        assert_eq!(out[16 * w + 19], 1); // distance 3 — outside radius-2 rings
    }

    #[test]
    fn segmentation_dilate_radius8_uses_outer_ring() {
        // pixel exactly 8 rows above: only reachable at radius 8.
        let (w, h) = (40usize, 40usize);
        let mut img = vec![0_u32; w * h];
        img[12 * w + 20] = 1;
        let mut out = vec![0_u32; w * h];
        unsafe {
            darkroom_segmentation_dilate(img.as_ptr(), out.as_mut_ptr(), w, h, 10, 8);
        }
        assert_eq!(out[20 * w + 20], 1); // -w8 tap fires
        let mut out7 = vec![0_u32; w * h];
        unsafe {
            darkroom_segmentation_dilate(img.as_ptr(), out7.as_mut_ptr(), w, h, 10, 7);
        }
        assert_eq!(out7[20 * w + 20], 0); // radius 7 can't reach 8 rows
    }

    #[test]
    fn interpolate_and_mask_grey_field_is_neutral() {
        // uniform unclipped mosaic, wb=1: every interpolated channel = v,
        // norm = v*sqrt(3), nothing clipped.
        let (w, h) = (8usize, 8usize);
        let v = 0.25_f32;
        let input = vec![v; w * h];
        let mut interp = vec![0.0_f32; w * h * 4];
        let mut cmask = vec![9.0_f32; w * h * 4];
        let clips = [1.0_f32; 4];
        let wb = [1.0_f32; 4];
        unsafe {
            darkroom_highlights_interpolate_and_mask(
                input.as_ptr(), interp.as_mut_ptr(), cmask.as_mut_ptr(),
                clips.as_ptr(), wb.as_ptr(), RGGB, w, h,
            );
        }
        let idx = (3 * w + 3) * 4; // interior pixel
        for c in 0..3 { assert!((interp[idx + c] - v).abs() < 1e-6, "c={c}"); }
        assert!((interp[idx + 3] - v * 3.0_f32.sqrt()).abs() < 1e-6);
        for c in 0..4 { assert_eq!(cmask[idx + c], 0.0); }
    }

    #[test]
    fn remosaic_blends_by_mask_opacity() {
        let (w, h) = (4usize, 4usize);
        let input = vec![0.2_f32; w * h];
        let mut interp = vec![0.8_f32; w * h * 4];
        let mut cmask = vec![0.0_f32; w * h * 4];
        // pixel (1,1): opacity 0.5
        cmask[(w + 1) * 4 + 3] = 0.5;
        interp[(w + 1) * 4 + 2] = 0.6; // its CFA colour at (1,1) RGGB = B(2)
        let wb = [1.0_f32; 4];
        let mut out = vec![0.0_f32; w * h];
        unsafe {
            darkroom_highlights_remosaic_and_replace(
                input.as_ptr(), interp.as_ptr(), cmask.as_ptr(), out.as_mut_ptr(),
                wb.as_ptr(), RGGB, w, h,
            );
        }
        assert!((out[0] - 0.2).abs() < 1e-6); // opacity 0 → original
        assert!((out[w + 1] - (0.5 * 0.6 + 0.5 * 0.2)).abs() < 1e-6);
    }

    #[test]
    fn guide_laplacians_unclipped_first_scale_copies_hf() {
        // alpha = 0 everywhere → no reconstruction; FIRST_SCALE → out = HF.
        let (w, h) = (6usize, 6usize);
        let hf: Vec<f32> = (0..w * h * 4).map(|k| k as f32 * 0.01).collect();
        let lf = vec![0.0_f32; w * h * 4];
        let cmask = vec![0.0_f32; w * h * 4];
        let mut out = vec![-1.0_f32; w * h * 4];
        unsafe {
            darkroom_highlights_guide_laplacians(
                hf.as_ptr(), lf.as_ptr(), cmask.as_ptr(), out.as_mut_ptr(),
                w, h, 1, 0.0, 0, FIRST_SCALE, 1.0,
            );
        }
        assert_eq!(out, hf);
    }

    #[test]
    fn heat_pde_last_scale_rebuilds_rgb_from_ratios() {
        // FIRST|LAST single-scale, alpha=0: out = max(HF+LF,0) then RGB *= norm.
        let (w, h) = (4usize, 4usize);
        let mut hf = vec![0.0_f32; w * h * 4];
        let mut lf = vec![0.0_f32; w * h * 4];
        for k in 0..w * h {
            // ratios (0.6, 0.8, 0.0), norm 2.0 split across HF+LF
            hf[k * 4] = 0.6;
            lf[k * 4 + 1] = 0.8;
            hf[k * 4 + 3] = 1.5;
            lf[k * 4 + 3] = 0.5;
        }
        let cmask = vec![0.0_f32; w * h * 4];
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe {
            darkroom_highlights_heat_pde_diffusion(
                hf.as_ptr(), lf.as_ptr(), cmask.as_ptr(), out.as_mut_ptr(),
                w, h, 1, FIRST_SCALE | LAST_SCALE, 0.0,
            );
        }
        // alpha=0 → no renormalize; RGB = ratio * norm, norm kept in ch 3
        let i = 4 * (w + 1);
        assert!((out[i] - 0.6 * 2.0).abs() < 1e-6);
        assert!((out[i + 1] - 0.8 * 2.0).abs() < 1e-6);
        assert!((out[i + 2] - 0.0).abs() < 1e-6);
        assert!((out[i + 3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dwt_interleave_rows_is_a_permutation() {
        for (h, s) in [(10usize, 4usize), (16, 2), (7, 3), (5, 8)] {
            let mut seen = vec![false; h];
            for r in 0..h {
                let i = crate::math::dwt_interleave_rows(r, h, s);
                assert!(i < h && !seen[i], "h={h} s={s} r={r} i={i}");
                seen[i] = true;
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

    // ── segbased.c ports ────────────────────────────────────────────────────

    #[test]
    fn segbased_initial_gradients_gates_on_distance() {
        let (pw, ph) = (24usize, 24usize);
        // luminance ramp by column → scharr magnitude 512/255 everywhere
        let lum: Vec<f32> = (0..pw * ph).map(|k| (k % pw) as f32).collect();
        let mut dist = vec![0.0_f32; pw * ph];
        let v = 12 * pw + 12;
        dist[v] = 1.0; // in (0, 2) → gradient written
        dist[v + 1] = 3.0; // >= 2 → stays 0
        let mut grad = vec![-1.0_f32; pw * ph];
        unsafe {
            darkroom_segbased_initial_gradients(lum.as_ptr(), dist.as_ptr(), grad.as_mut_ptr(), pw, ph);
        }
        assert!((grad[v] - 4.0 * 512.0 / 255.0).abs() < 1e-5, "grad[v]={}", grad[v]);
        assert_eq!(grad[v + 1], 0.0);
        assert_eq!(grad[v + pw], 0.0); // distance 0 → 0
        assert_eq!(grad[0], -1.0); // border region untouched
    }

    #[test]
    fn segbased_maxdistance_respects_id_and_bounds() {
        let (w, h) = (8usize, 8usize);
        let mut data = vec![0_u32; w * h];
        let mut dist = vec![0.0_f32; w * h];
        data[2 * w + 2] = 5;
        dist[2 * w + 2] = 7.0;
        data[3 * w + 3] = 6; // other segment — ignored
        dist[3 * w + 3] = 9.0;
        data[7 * w + 7] = 5; // outside the bbox — ignored
        dist[7 * w + 7] = 11.0;
        let m = unsafe {
            darkroom_segbased_maxdistance(dist.as_ptr(), data.as_ptr(), w, h, 1, 5, 1, 5, 5)
        };
        assert_eq!(m, 7.0);
    }

    #[test]
    fn segbased_distance_ring_averages_inner_ring() {
        let (w, h) = (12usize, 12usize);
        let id = 2_u32;
        let mut data = vec![0_u32; w * h];
        // background distance 5.0 keeps the rest of the 5x5 window out of the
        // inner ring [dist-1.5, dist) — note 0.0 would qualify for dist=1.5!
        let mut dist = vec![5.0_f32; w * h];
        let mut grad = vec![0.0_f32; w * h];
        let v = 6 * w + 6;
        data[v] = id;
        dist[v] = 2.0; // in [1.5, 3.0) for dist=1.5
        dist[v - 1] = 1.0; // inner ring: in [0.0, 1.5)
        grad[v - 1] = 0.5;
        unsafe {
            darkroom_segbased_distance_ring(grad.as_mut_ptr(), dist.as_ptr(), data.as_ptr(),
                                            w, h, 4, 9, 4, 9, 1.0, 1.5, id);
        }
        // grd/cnt = 0.5; attenuate=1 → 0.5 * (1 + 1/2.0) = 0.75
        assert!((grad[v] - 0.75).abs() < 1e-6, "grad[v]={}", grad[v]);
    }

    #[test]
    fn segbased_box_roundtrip_and_strength() {
        let (w, _h) = (10usize, 8usize);
        let (xmin, xmax, ymin, ymax) = (2_i32, 6, 1, 5);
        let id = 3_u32;
        let mut grad: Vec<f32> = (0..w * 8).map(|k| k as f32).collect();
        let mut data = vec![0_u32; w * 8];
        data[2 * w + 3] = id;
        let mut tmp = vec![0.0_f32; 16];
        unsafe {
            darkroom_segbased_box_in(grad.as_ptr(), tmp.as_mut_ptr(), w, xmin, xmax, ymin, ymax);
        }
        assert_eq!(tmp[0], (1 * w + 2) as f32); // (ymin, xmin)
        assert_eq!(tmp[15], (4 * w + 5) as f32); // (ymax-1, xmax-1)
        // perturb tmp, copy back: only the id-cell changes
        for t in tmp.iter_mut() { *t += 100.0; }
        let before = grad.clone();
        unsafe {
            darkroom_segbased_box_out(grad.as_mut_ptr(), tmp.as_ptr(), data.as_ptr(),
                                      w, xmin, xmax, ymin, ymax, id);
        }
        for (k, (&g, &b)) in grad.iter().zip(before.iter()).enumerate() {
            if k == 2 * w + 3 {
                assert_eq!(g, b + 100.0);
            } else {
                assert_eq!(g, b);
            }
        }
        unsafe {
            darkroom_segbased_apply_strength(grad.as_mut_ptr(), data.as_ptr(),
                                             w, xmin, xmax, ymin, ymax, id, 0.5);
        }
        assert_eq!(grad[2 * w + 3], (before[2 * w + 3] + 100.0) * 0.5);
        assert_eq!(grad[2 * w + 4], before[2 * w + 4]); // not in segment
    }

    #[test]
    fn segbased_masks_extend_border_replicates_edges() {
        let (w, h, b) = (8usize, 8usize, 2_i32);
        let mut m = vec![0.0_f32; w * h];
        // interior filled with a recognizable value per row
        for row in 2..6 {
            for col in 2..6 { m[row * w + col] = (10 * row + col) as f32; }
        }
        unsafe { darkroom_masks_extend_border(m.as_mut_ptr(), w, h, b); }
        // rows: left/right replicate from col 2 / col 5
        assert_eq!(m[3 * w], m[3 * w + 2]);
        assert_eq!(m[3 * w + 1], m[3 * w + 2]);
        assert_eq!(m[3 * w + 7], m[3 * w + 5]);
        // cols: top/bottom replicate row 2 / row 5 (col clamped to interior)
        assert_eq!(m[3], m[2 * w + 3]);
        assert_eq!(m[w + 3], m[2 * w + 3]);
        assert_eq!(m[7 * w + 3], m[5 * w + 3]);
        // corner: clamped to (2,2)
        assert_eq!(m[0], m[2 * w + 2]);
    }

    #[test]
    fn segbased_populate_planes_counts_clipping() {
        let (w, h) = (12usize, 12usize);
        let (pw, ph) = (24usize, 24usize);
        let psize = pw * ph;
        let tmpout = vec![0.5_f32; w * h];
        let corr = [1.0_f32; 4];
        let cube = [0.5_f32; 4]; // cbrt(0.5) ≈ 0.794 > 0.5 → all clipped
        let mut planes: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0; psize]).collect();
        let mut refavgs: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0; psize]).collect();
        let mut segs: Vec<Vec<u32>> = (0..4).map(|_| vec![0; psize]).collect();
        let pptr: Vec<*mut f32> = planes.iter_mut().map(|p| p.as_mut_ptr()).collect();
        let rptr: Vec<*mut f32> = refavgs.iter_mut().map(|p| p.as_mut_ptr()).collect();
        let sptr: Vec<*mut u32> = segs.iter_mut().map(|p| p.as_mut_ptr()).collect();
        let mut has_all = 0_i32;
        // RGGB: FC(0,0)=R → xshifter = 2
        let anyclipped = unsafe {
            darkroom_segbased_populate_planes(
                tmpout.as_ptr(), w, h, RGGB, XTRANS.as_ptr() as *const u8,
                corr.as_ptr(), cube.as_ptr(), 2,
                pptr.as_ptr(), rptr.as_ptr(), sptr.as_ptr(), pw, ph, &mut has_all,
            )
        };
        // superpixel centres: rows {1,4,7,10} x cols {2,5,8} = 12, all 3 channels clipped
        assert_eq!(anyclipped, 36);
        assert_eq!(has_all, 1);
        let o = raw_to_plane(pw, 1, 2);
        let expect = 0.5_f32.cbrt();
        for c in 0..3 {
            assert!((planes[c][o] - expect).abs() < 1e-6);
            assert!((refavgs[c][o] - expect).abs() < 1e-6); // opposing means equal for grey
            assert_eq!(segs[c][o], 1);
        }
        assert_eq!(segs[3][o], 1);
    }

    #[test]
    fn segbased_candidates_apply_inpaints_from_candidate() {
        let (w, h) = (12usize, 12usize);
        let (pw, ph, border) = (24usize, 24usize, 9_i32);
        let psize = pw * ph;
        let input = vec![1.0_f32; w * h];
        let mut tmpout = input.clone();
        let clips = [0.5_f32, 10.0, 10.0, 10.0]; // only red photosites clip
        let corr = [1.0_f32; 4];
        let mut planes: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0; psize]).collect();
        let pptr: Vec<*mut f32> = planes.iter_mut().map(|p| p.as_mut_ptr()).collect();
        let mut data = vec![vec![0_u32; psize], vec![0_u32; psize], vec![0_u32; psize]];
        let nr = [3_i32, 3, 3];
        // segment id 2 only at the plane cell of photosite (4,4) (red in RGGB)
        let o = raw_to_plane(pw, 4, 4);
        data[0][o] = 2;
        let val1 = [vec![0.0_f32, 0.0, 2.0], vec![0.0; 3], vec![0.0; 3]]; // candidate 2.0
        let val2 = [vec![0.0_f32; 3], vec![0.0; 3], vec![0.0; 3]]; // reference 0.0
        let dptr: Vec<*const u32> = data.iter().map(|d| d.as_ptr()).collect();
        let v1ptr: Vec<*const f32> = val1.iter().map(|v| v.as_ptr()).collect();
        let v2ptr: Vec<*const f32> = val2.iter().map(|v| v.as_ptr()).collect();
        unsafe {
            darkroom_segbased_candidates_apply(
                input.as_ptr(), tmpout.as_mut_ptr(), w, h, RGGB, XTRANS.as_ptr() as *const u8,
                clips.as_ptr(), corr.as_ptr(), pptr.as_ptr(),
                dptr.as_ptr(), v1ptr.as_ptr(), v2ptr.as_ptr(), nr.as_ptr(),
                pw, ph, border,
            );
        }
        // refavg of an all-1.0 window = 1.0 → oval = (1 + 2 - 0)^3 = 27
        let idx = 4 * w + 4;
        assert!((tmpout[idx] - 27.0).abs() < 1e-4, "tmpout={}", tmpout[idx]);
        assert!((planes[0][o] - 27.0).abs() < 1e-4);
        // other red photosites clip too but have no segment → untouched
        assert_eq!(tmpout[4 * w + 6], 1.0);
    }

    #[test]
    fn segbased_prepare_lumdist_seeds_distance() {
        let (pw, ph, b) = (12usize, 12usize, 2_i32);
        let psize = pw * ph;
        let p0 = vec![0.3_f32; psize];
        let p1 = vec![0.6_f32; psize];
        let p2 = vec![0.9_f32; psize];
        let ic = [1.0_f32, 2.0, 3.0];
        let mut tmp = vec![-1.0_f32; psize];
        let mut dist = vec![-1.0_f32; psize];
        let mut data = vec![0_u32; psize];
        let i = 5 * pw + 5;
        data[i] = 1;
        unsafe {
            darkroom_segbased_prepare_lumdist(p0.as_ptr(), p1.as_ptr(), p2.as_ptr(), ic.as_ptr(),
                                              tmp.as_mut_ptr(), dist.as_mut_ptr(), data.as_ptr(),
                                              pw, ph, b);
        }
        // (0.3*1 + 0.6*2 + 0.9*3)/3 = 1.4
        assert!((tmp[i] - 1.4).abs() < 1e-6);
        assert_eq!(dist[i], 1e20);
        assert_eq!(dist[i + 1], 0.0);
        assert_eq!(tmp[0], -1.0); // border untouched
    }

    #[test]
    fn segbased_apply_recovery_sigmoid_at_dshift() {
        let (w, h) = (12usize, 12usize);
        let (pw, ph) = (24usize, 24usize);
        let psize = pw * ph;
        let input = vec![1.0_f32; w * h];
        let mut tmpout = input.clone();
        let clips = [0.5_f32, 10.0, 10.0, 10.0];
        let mut dist = vec![0.0_f32; psize];
        let mut grad = vec![0.0_f32; psize];
        let o = raw_to_plane(pw, 4, 4);
        dist[o] = 2.0; // == dshift → sigmoid = 0.5
        grad[o] = 0.8;
        unsafe {
            darkroom_segbased_apply_recovery(input.as_ptr(), tmpout.as_mut_ptr(), w, h,
                                             RGGB, XTRANS.as_ptr() as *const u8, clips.as_ptr(),
                                             dist.as_ptr(), grad.as_ptr(), pw, ph, 1.0, 2.0);
        }
        let idx = 4 * w + 4;
        assert!((tmpout[idx] - (1.0 + 0.8 * 0.5)).abs() < 1e-6, "tmpout={}", tmpout[idx]);
        // unclipped green neighbour untouched
        assert_eq!(tmpout[4 * w + 5], 1.0);
    }

    #[test]
    fn segbased_final_output_copy_and_combine_mask() {
        let (iw, ih) = (12usize, 12usize);
        let (ow, oh) = (6usize, 6usize);
        let (pw, ph, border) = (24usize, 24usize, 9_i32);
        let psize = pw * ph;
        let tmpout: Vec<f32> = (0..iw * ih).map(|k| k as f32).collect();
        let lum = vec![10.0_f32; psize]; // 0.2*10 = 2 → min(0.2, 2) = 0.2
        let grad = vec![0.0_f32; psize];
        let mut data = vec![vec![0_u32; psize], vec![0_u32; psize], vec![0_u32; psize]];
        let alldata = vec![0_u32; psize];
        let nr = [3_i32, 3, 3];
        let val1 = [vec![0.0_f32; 3], vec![0.0; 3], vec![0.0; 3]];
        let o = raw_to_plane(pw, 3, 3);
        data[2][o] = 2 | SEG_ID_MASK; // (3,3) is blue in RGGB; flagged border bit
        let dptr: Vec<*const u32> = data.iter().map(|d| d.as_ptr()).collect();
        let v1ptr: Vec<*const f32> = val1.iter().map(|v| v.as_ptr()).collect();
        let mut out = vec![-5.0_f32; ow * oh];

        // 1) no masking: plain crop copy with offset (2,2)
        unsafe {
            darkroom_segbased_final_output(
                out.as_mut_ptr(), tmpout.as_ptr(), lum.as_ptr(), grad.as_ptr(),
                ow, oh, 2, 2, iw, ih, RGGB, XTRANS.as_ptr() as *const u8,
                dptr.as_ptr(), v1ptr.as_ptr(), nr.as_ptr(), alldata.as_ptr(), 2,
                pw, ph, border, 0, 0, 1.0,
            );
        }
        assert_eq!(out[0], (2 * iw + 2) as f32);
        assert_eq!(out[ow + 1], (3 * iw + 3) as f32);

        // 2) COMBINE masking: dim luminance + 1.0 for the flagged segment cell
        unsafe {
            darkroom_segbased_final_output(
                out.as_mut_ptr(), tmpout.as_ptr(), lum.as_ptr(), grad.as_ptr(),
                ow, oh, 2, 2, iw, ih, RGGB, XTRANS.as_ptr() as *const u8,
                dptr.as_ptr(), v1ptr.as_ptr(), nr.as_ptr(), alldata.as_ptr(), 2,
                pw, ph, border, 1, 1, 1.0,
            );
        }
        // (out 1,1) ← in (3,3): 0.2 + 1.0 (DT_SEG_ID_MASK set)
        assert!((out[ow + 1] - 1.2).abs() < 1e-6, "out={}", out[ow + 1]);
        // a cell with no segment: just the dimmed luminance
        assert!((out[0] - 0.2).abs() < 1e-6, "out={}", out[0]);
    }
}
