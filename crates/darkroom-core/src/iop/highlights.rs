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
