//! À-trous wavelet decompose / denoise — a faithful port of the CPU path of
//! `src/common/dwt.c` (the GIMP "Wavelet Decompose" algorithm by Marco Rossini).
//! Shared infrastructure for the IOPs that operate in the wavelet domain
//! (`atrous`, `retouch`, and the wavelet mode of `denoiseprofile`).
//!
//! The decomposition is the classic *à trous* ("with holes") scheme: at each
//! scale `lev` the image is smoothed with a dilated 3×3 B-spline-ish hat kernel
//! (taps at offset `2^lev`), the smoothed result is the **coarse** layer, and the
//! **detail** layer is `input − coarse`. The coarse layer feeds the next, wider
//! scale. Because `detail_lev = input_lev − coarse_lev` and `coarse_lev` becomes
//! `input_{lev+1}`, the sum of every detail layer plus the final residual
//! telescopes back to the original image exactly — so a decompose/recompose with
//! an identity layer callback is a (float-rounding) identity. Callers hook the
//! per-scale [`decompose`] callback to reweight/threshold each detail layer before
//! it is summed back in.
//!
//! Ported: the CPU `dwt_decompose` (RGBA, `ch == 4`) with its layer callback, the
//! scale-count helpers, and the 1-channel `dwt_denoise`. Not ported: all OpenCL
//! (`dwt_*_cl`), and the `dwt_wavelet_decompose` callback orchestration /
//! allocation strategy (still C).
//!
//! The 1-channel denoise passes are additionally exported over FFI
//! ([`darkroom_dwt_denoise_vert_1ch`] / [`darkroom_dwt_denoise_horiz_1ch`],
//! m4-179) and drive `src/common/dwt.c`'s denoise loops. The RGBA decompose
//! passes are likewise exported ([`darkroom_dwt_decompose_vert`] /
//! [`darkroom_dwt_decompose_horiz`], m4-184) and drive `dwt_decompose_vert` /
//! `dwt_decompose_horiz` in `src/common/dwt.c`.
//!
//! **Rust-vs-C hardening.** The C code reflects out-of-bounds edge taps with
//! unsigned/`int` index arithmetic that reads slightly out of bounds (benign UB)
//! on degenerate inputs where a wavelet scale is comparable to the image
//! dimension. `dwt_decompose` clamps `scales` to [`DwtParams::get_max_scale`], so
//! those cases never arise on its own path; `dwt_denoise` does **not** clamp its
//! `bands`. To keep Rust panic-free we clamp every reflected tap index into the
//! valid range. For all in-range inputs (every tap already inside the image — i.e.
//! every scale smaller than the dimension) the clamp is a no-op, so the result is
//! bit-identical to the C for the inputs that matter; only genuinely degenerate
//! inputs (where C would read past the buffer) diverge, and there into a defined,
//! finite value instead of UB.
//!
//! **Cache interleave omitted.** The C decompose reorders its vertical-pass rows
//! via `dwt_interleave_rows` purely for cache friendliness (its own comment says
//! so). Each output row depends only on *input* rows that are never mutated during
//! the pass, so the visiting order has no effect on the result; this port iterates
//! rows in natural order. The [`decompose_vert_reference`] used by the m4-184
//! tests visits rows in the C's interleave order, so kernel/reference agreement
//! locks that equivalence bit-exactly.

/// Parameters for [`decompose`], mirroring C's `dwt_params_t`. The image buffer is
/// passed separately (as a `&mut [f32]`) rather than stored as a raw pointer.
///
/// `ch` must be `4` (RGBA, matching the C `assert(p->ch == 4)`).
#[derive(Clone, Copy, Debug)]
pub struct DwtParams {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Channels per pixel — must be 4.
    pub ch: usize,
    /// Number of detail scales to decompose. Clamped down to `get_max_scale()`.
    pub scales: i32,
    /// `0` → return the recomposed image; `1..=scales` → return that detail scale;
    /// `scales + 1` → return the residual (coarsest) image.
    pub return_layer: i32,
    /// If `> 0`, detail scales from this scale on are merged together before the
    /// callback sees them.
    pub merge_from_scale: i32,
    /// Zoom factor of the buffer relative to the full image (`1.0` at full res).
    pub preview_scale: f32,
}

/// `1 << lev`, saturating instead of overflowing for absurd `lev` (the natural
/// path keeps `lev` small, well under the shift width).
#[inline]
fn pow2(lev: usize) -> usize {
    1usize.checked_shl(lev as u32).unwrap_or(usize::MAX)
}

impl DwtParams {
    /// Maximum number of scales the image size supports — port of
    /// `dwt_get_max_scale` → `_get_max_scale(width / preview_scale, …)`.
    pub fn get_max_scale(&self) -> i32 {
        let ps = if self.preview_scale <= 0.0 { 1.0 } else { self.preview_scale };
        let w = (self.width as f32 / ps) as i32;
        let h = (self.height as f32 / ps) as i32;
        max_scale_raw(w, h, ps)
    }

    /// First detail scale that is visible at the current zoom — port of
    /// `dt_dwt_first_scale_visible` → `_first_scale_visible`.
    pub fn first_scale_visible(&self) -> i32 {
        for lev in 0..self.scales {
            // C: `int sc = 1 << lev; sc *= preview_scale;` (truncated to int)
            let sc = ((pow2(lev as usize) as f32) * self.preview_scale) as i32;
            if sc > 0 {
                return lev + 1;
            }
        }
        0
    }
}

/// Port of `_get_max_scale`. `width`/`height` are already divided by the zoom.
fn max_scale_raw(width: i32, height: i32, preview_scale: f32) -> i32 {
    let mut maxscale: i32 = 0;

    // smallest edge must be >= 2^scales; count how many halvings stay positive.
    let mut size: u32 = width.min(height).max(0) as u32;
    size >>= 1;
    let mut size_tmp = size as f32 * preview_scale;
    while size_tmp > 0.0 {
        size >>= 1;
        size_tmp = size as f32 * preview_scale;
        maxscale += 1;
    }

    // avoid rounding issues (C loops with the original, undivided-again `size`).
    let size2 = width.min(height).max(0) as u32;
    while maxscale > 0 && ((1i64 << maxscale) as f32 * preview_scale) >= size2 as f32 {
        maxscale -= 1;
    }

    maxscale
}

/// `dst += src`, elementwise — port of `dt_iop_image_add_image`.
#[inline]
fn image_add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

/// Borrow two distinct elements of a length-2 buffer array mutably, returning
/// `(buf[i], buf[j])`. `i != j` always holds on the decompose ping-pong.
#[inline]
fn two_mut(buf: &mut [Vec<f32>; 2], i: usize, j: usize) -> (&mut Vec<f32>, &mut Vec<f32>) {
    debug_assert_ne!(i, j);
    let (a, b) = buf.split_at_mut(1);
    if i == 0 {
        (&mut a[0], &mut b[0])
    } else {
        (&mut b[0], &mut a[0])
    }
}

// "Vertical" pass of one decomposition scale (RGBA): out = 2·center + above +
// below, reflecting at the top/bottom edges. Port of `dwt_decompose_vert`
// (m4-184 safe kernel; the FFI export [`darkroom_dwt_decompose_vert`] forwards
// here).
//
// Rows run in natural order (see the module-level interleave note).
// `vscale` caps at `height - 1`, so for `height >= 1` both reflected rows stay
// in `[0, height-1]` with no further clamping; `lev` needs no clamp beyond the
// saturating `1 << lev` in [`pow2`]. `out` and `inp` must each hold
// `4 * width * height` floats and must not overlap.
pub fn decompose_vert(out: &mut [f32], inp: &[f32], height: usize, width: usize, lev: usize) {
    if width == 0 || height == 0 {
        return;
    }
    debug_assert_eq!(out.len(), 4 * width * height);
    debug_assert_eq!(inp.len(), 4 * width * height);
    // vscale capped at height-1, so both reflected rows stay in [0, height-1].
    let vscale = pow2(lev).min(height.saturating_sub(1));
    for row in 0..height {
        let rowstart = 4 * row * width;
        let above_row = row.abs_diff(vscale);
        let below_row = if row + vscale < height {
            row + vscale
        } else {
            2 * (height - 1) - (row + vscale)
        };
        let above = 4 * above_row * width;
        let below = 4 * below_row * width;
        for col in (0..4 * width).step_by(4) {
            for c in 0..4 {
                out[rowstart + col + c] =
                    2.0 * inp[rowstart + col + c] + inp[above + col + c] + inp[below + col + c];
            }
        }
    }
}

// Horizontal pass (RGBA): writes the normalised 'coarse' back into `out` and the
// 'details' (input − coarse) into `inp`. Port of `dwt_decompose_horiz` (m4-184
// safe kernel; the FFI export [`darkroom_dwt_decompose_horiz`] forwards here).
//
// `hscale` caps at `width`. Taps read from `out` (the vertical-pass result,
// never mutated during the row — only the scratch row is written, then copied
// back), so overlapping tap reads reuse identical values when
// `hscale > width/2` makes the edge ranges share taps (matching the C).
// Reflected taps outside `[0, width-1]` are clamped into range; for in-range
// inputs the clamp is a no-op. `out`/`inp` must each hold `4 * width * height`
// floats and must not overlap.
pub fn decompose_horiz(out: &mut [f32], inp: &mut [f32], height: usize, width: usize, lev: usize) {
    if width == 0 || height == 0 {
        return;
    }
    debug_assert_eq!(out.len(), 4 * width * height);
    debug_assert_eq!(inp.len(), 4 * width * height);
    let hscale = pow2(lev).min(width);
    let last = width.saturating_sub(1);
    let mut temprow = vec![0.0f32; 4 * width];
    for row in 0..height {
        let ri = 4 * row * width;
        // interior columns: reflected left, direct right
        for col in 0..width.saturating_sub(hscale) {
            let leftcol = (col as i32 - hscale as i32).unsigned_abs() as usize;
            let leftpos = 4 * leftcol.min(last);
            let rightpos = 4 * (col + hscale).min(last);
            let base = 4 * col;
            for c in 0..4 {
                let l = out[ri + leftpos + c];
                let r = out[ri + rightpos + c];
                let hat = (2.0 * out[ri + base + c] + l + r) / 16.0;
                temprow[base + c] = hat;
                inp[ri + base + c] -= hat;
            }
        }
        // right edge: reflect the right tap around the image boundary
        for col in width.saturating_sub(hscale)..width {
            let leftcol = (col as i32 - hscale as i32).unsigned_abs() as usize;
            let leftpos = 4 * leftcol.min(last);
            let rightcol = (2 * width as i32 - 2 - (col as i32 + hscale as i32)).max(0) as usize;
            let rightpos = 4 * rightcol.min(last);
            let base = 4 * col;
            for c in 0..4 {
                let l = out[ri + leftpos + c];
                let r = out[ri + rightpos + c];
                let hat = (2.0 * out[ri + base + c] + l + r) / 16.0;
                temprow[base + c] = hat;
                inp[ri + base + c] -= hat;
            }
        }
        // overwrite the vertical-pass intermediate with the final coarse layer.
        out[ri..ri + 4 * width].copy_from_slice(&temprow);
    }
}

/// The actual decomposing algorithm — port of `dwt_wavelet_decompose`. `p` is
/// assumed already clamped by [`decompose`]. The result is written into `image`.
fn wavelet_decompose<F>(image: &mut [f32], p: &DwtParams, layer_func: &mut F)
where
    F: FnMut(&mut [f32], &DwtParams, i32),
{
    assert_eq!(p.ch, 4, "dwt: only ch == 4 is supported");
    let n = 4 * p.width * p.height;

    // buffer[0] starts as the (copied) image; buffer[1] is scratch.
    let mut buffer: [Vec<f32>; 2] = [image.to_vec(), vec![0.0f32; n]];
    let mut layers = vec![0.0f32; n]; // reconstruction accumulator (cleared)
    let do_merge = p.merge_from_scale > 0;
    let mut merged_layers = if do_merge { vec![0.0f32; n] } else { Vec::new() };

    // scale 0: the original image
    layer_func(&mut buffer[0], p, 0);

    if p.scales <= 0 {
        image.copy_from_slice(&buffer[0]);
        return;
    }

    let mut hpass = 0usize;
    let mut bcontinue = true;
    let mut lev = 0i32;
    while lev < p.scales && bcontinue {
        let lpass = 1 - (lev as usize & 1);

        // split input[hpass] into coarse (→ buffer[lpass]) and details (→ buffer[hpass])
        {
            let (out, inp) = two_mut(&mut buffer, lpass, hpass);
            decompose_vert(out, inp, p.height, p.width, lev as usize);
            decompose_horiz(out, inp, p.height, p.width, lev as usize);
        }

        if p.merge_from_scale == 0 || p.merge_from_scale > lev + 1 {
            // not merging (yet): let the caller process this detail scale
            layer_func(&mut buffer[hpass], p, lev + 1);

            if p.return_layer == lev + 1 {
                image.copy_from_slice(&buffer[hpass]);
                bcontinue = false;
            } else if p.return_layer == 0 {
                image_add(&mut layers, &buffer[hpass]);
            }
        } else {
            // within the merge range: accumulate then process the merged scale
            image_add(&mut merged_layers, &buffer[hpass]);
            layer_func(&mut merged_layers, p, lev + 1);

            if p.return_layer == lev + 1 {
                image.copy_from_slice(&merged_layers);
                bcontinue = false;
            }
        }

        hpass = lpass;
        lev += 1;
    }

    if bcontinue {
        // all scales processed — `buffer[hpass]` now holds the residual image
        layer_func(&mut buffer[hpass], p, p.scales + 1);

        if p.return_layer == p.scales + 1 {
            image.copy_from_slice(&buffer[hpass]);
        } else if p.return_layer == 0 {
            if p.merge_from_scale > 0 {
                image_add(&mut layers, &merged_layers);
            }
            image_add(&mut layers, &buffer[hpass]);
            layer_func(&mut layers, p, p.scales + 2);
            image.copy_from_slice(&layers);
        }
    }
}

/// Decompose `image` (RGBA, `width*height*4` floats) into wavelet scales, invoking
/// `layer_func(layer, params, scale)` for the original image (`scale == 0`), each
/// detail scale (`1..=scales`), the residual (`scales + 1`), and — when returning
/// the recomposed image — the final reconstruction (`scales + 2`). The chosen
/// output (see [`DwtParams::return_layer`]) is written back into `image`.
///
/// `p` is adjusted in place exactly as C's `dwt_decompose` does (zoom guard,
/// `return_layer`/`scales` clamped to the image's maximum supported scale count).
pub fn decompose<F>(image: &mut [f32], p: &mut DwtParams, mut layer_func: F)
where
    F: FnMut(&mut [f32], &DwtParams, i32),
{
    if p.width == 0 || p.height == 0 {
        return;
    }

    // this is a zoom scale, not a wavelet scale
    if p.preview_scale <= 0.0 {
        p.preview_scale = 1.0;
    }

    // a single requested scale cannot exceed the residual
    if p.return_layer > p.scales + 1 {
        p.return_layer = p.scales + 1;
    }
    // out-of-contract guard (no C equivalent): a negative return_layer is
    // meaningless. C would leave the caller's aliased buffer holding leftover
    // ping-pong state; clamp to 0 (recomposed image) so `image` is well-defined.
    if p.return_layer < 0 {
        p.return_layer = 0;
    }

    let max_scale = p.get_max_scale();
    if p.scales > max_scale {
        if p.return_layer > p.scales {
            p.return_layer = max_scale + 1;
        } else if p.return_layer > max_scale {
            p.return_layer = max_scale;
        }
        p.scales = max_scale;
    }

    wavelet_decompose(image, p, &mut layer_func);
}

// ---------------------------------------------------------------------------
// 1-channel denoise (dwt_denoise)
// ---------------------------------------------------------------------------

// Vertical pass, single channel: out = 2·center + above + below with edge
// reflection. Port of `dwt_denoise_vert_1ch` (note vscale caps at `height`, not
// `height-1`, so reflected rows are clamped to stay in bounds — see module docs).
//
// Rows are visited in natural order: the C visits them via `dwt_interleave_rows`
// purely for cache friendliness (see `dwt.h`), and each output row depends only
// on *input* rows that are never mutated during the pass, so the visiting order
// has no effect on the result.
//
// `out` and `inp` must each hold `width * height` floats and must not overlap.
pub fn denoise_vert_1ch(out: &mut [f32], inp: &[f32], height: usize, width: usize, lev: usize) {
    if width == 0 || height == 0 {
        return;
    }
    debug_assert_eq!(out.len(), width * height);
    debug_assert_eq!(inp.len(), width * height);
    let vscale = pow2(lev).min(height);
    let last = height - 1;
    for row in 0..height {
        let rowstart = row * width;
        let above = ((row as i32 - vscale as i32).unsigned_abs() as usize).min(last);
        let below = if row + vscale < height {
            row + vscale
        } else {
            (2 * (height as i32 - 1) - (row as i32 + vscale as i32)).clamp(0, last as i32) as usize
        };
        let a = above * width;
        let b = below * width;
        for col in 0..width {
            out[rowstart + col] = 2.0 * inp[rowstart + col] + inp[a + col] + inp[b + col];
        }
    }
}

// Horizontal pass, single channel: computes the coarse layer from `coarse`
// (vertical-pass result), overwrites `details` (the running image) with it, and
// accumulates the soft-thresholded detail into `accum`. On the last band the
// accumulated detail is added back into `details`. Port of
// `dwt_denoise_horiz_1ch`.
//
// The soft threshold is `MAX(diff − thold, 0) + MIN(diff + thold, 0)` exactly as
// in C (a single expression, so NaN / negative-threshold / signed-zero inputs
// behave identically). `coarse` must not overlap `details` or `accum`; all
// three slices must hold `width * height` floats.
#[allow(clippy::too_many_arguments)]
pub fn denoise_horiz_1ch(
    coarse: &[f32],
    details: &mut [f32],
    accum: &mut [f32],
    height: usize,
    width: usize,
    lev: usize,
    thold: f32,
    last: bool,
) {
    if width == 0 || height == 0 {
        return;
    }
    debug_assert_eq!(coarse.len(), width * height);
    debug_assert_eq!(details.len(), width * height);
    debug_assert_eq!(accum.len(), width * height);
    let hscale = pow2(lev).min(width);
    let wlast = width - 1;
    for row in 0..height {
        let ri = row * width;
        // left edge
        for col in 0..hscale.min(width) {
            let lp = ri + (hscale - col).min(wlast);
            let rp = ri + (col + hscale).min(wlast);
            let hat = (2.0 * coarse[ri + col] + coarse[lp] + coarse[rp]) / 16.0;
            let diff = details[ri + col] - hat;
            details[ri + col] = hat;
            accum[ri + col] += (diff - thold).max(0.0) + (diff + thold).min(0.0);
        }
        // interior
        for col in hscale..width.saturating_sub(hscale) {
            let hat = (2.0 * coarse[ri + col] + coarse[ri + col - hscale] + coarse[ri + col + hscale])
                / 16.0;
            let diff = details[ri + col] - hat;
            details[ri + col] = hat;
            accum[ri + col] += (diff - thold).max(0.0) + (diff + thold).min(0.0);
        }
        // right edge
        for col in width.saturating_sub(hscale)..width {
            let lcol = (col as i32 - hscale as i32).max(0) as usize;
            let rcol = (2 * width as i32 - 2 - (col as i32 + hscale as i32)).clamp(0, wlast as i32)
                as usize;
            let hat = (2.0 * coarse[ri + col] + coarse[ri + lcol] + coarse[ri + rcol]) / 16.0;
            let diff = details[ri + col] - hat;
            details[ri + col] = hat;
            accum[ri + col] += (diff - thold).max(0.0) + (diff + thold).min(0.0);
        }
        if last {
            for col in 0..width {
                details[ri + col] += accum[ri + col];
            }
        }
    }
}

/// Denoise a single-channel image in place by decomposing it into `bands` wavelet
/// scales and recomposing from only the portion of each scale whose magnitude
/// exceeds the per-band `noise` threshold. Port of `dwt_denoise`.
///
/// `img` holds `width * height` floats; `noise` holds one threshold per band.
pub fn denoise(img: &mut [f32], width: usize, height: usize, bands: usize, noise: &[f32]) {
    if width == 0 || height == 0 || bands == 0 {
        return;
    }
    debug_assert!(noise.len() >= bands);
    let np = width * height;
    let mut accum = vec![0.0f32; np]; // the accumulator ('details' in C), zeroed
    let mut interm = vec![0.0f32; np];

    for (lev, &nz) in noise.iter().take(bands).enumerate() {
        let last = lev + 1 == bands;
        denoise_vert_1ch(&mut interm, img, height, width, lev);
        denoise_horiz_1ch(&interm, img, &mut accum, height, width, lev, nz, last);
    }
}

// ---------------------------------------------------------------------------
// Structurally divergent reference implementations (m4-179)
// ---------------------------------------------------------------------------
//
// These compute the same clamped semantics as the public kernels above but with
// deliberately different loop structure, so the differential tests below catch
// transcription slips rather than re-running the same code. The scalar value
// expressions keep the C association order (`(2·c + a) + b`, `/ 16`, the
// single-expression soft threshold) — that order is load-bearing for bit-exact
// agreement, so it is shared on purpose; everything around it differs.

// Reference for [`denoise_vert_1ch`]: a single flat pixel loop with the row/column
// recovered by division/remainder, and the reflected tap rows resolved by small
// helpers over `i64` (instead of the kernel's nested row-major loops with inline
// `usize`/`i32` index math).
fn denoise_vert_1ch_reference(
    out: &mut [f32],
    inp: &[f32],
    height: usize,
    width: usize,
    lev: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    debug_assert_eq!(out.len(), width * height);
    debug_assert_eq!(inp.len(), width * height);
    let vscale = pow2(lev).min(height) as i64;
    let h = height as i64;
    let w = width as i64;
    let reflect_top = |row: i64| (row - vscale).abs().min(h - 1);
    let reflect_bottom = |row: i64| {
        if row + vscale < h {
            row + vscale
        } else {
            (2 * (h - 1) - (row + vscale)).clamp(0, h - 1)
        }
    };
    for idx in 0..width * height {
        let idx64 = idx as i64;
        let row = idx64 / w;
        let col = idx64 % w;
        let a = reflect_top(row) * w + col;
        let b = reflect_bottom(row) * w + col;
        out[idx] = 2.0 * inp[idx] + inp[a as usize] + inp[b as usize];
    }
}

// Reference for [`denoise_horiz_1ch`]: the same three positional tap ranges driven
// through per-column helper closures, with the `last` fold as a whole-buffer
// pass afterwards (instead of the kernel's inline tap math and per-row fold).
//
// The three ranges are kept (not unified): when `hscale > width/2` the left and
// right ranges overlap, and the C filters those shared columns twice — the
// second time from the already-overwritten `details`. That double filtering is
// C behaviour, so the reference reproduces it.
#[allow(clippy::too_many_arguments)]
fn denoise_horiz_1ch_reference(
    coarse: &[f32],
    details: &mut [f32],
    accum: &mut [f32],
    height: usize,
    width: usize,
    lev: usize,
    thold: f32,
    last: bool,
) {
    if width == 0 || height == 0 {
        return;
    }
    debug_assert_eq!(coarse.len(), width * height);
    debug_assert_eq!(details.len(), width * height);
    debug_assert_eq!(accum.len(), width * height);
    let hscale = pow2(lev).min(width);
    let wlast = (width - 1) as i64;
    let clamp_col = |c: i64| c.clamp(0, wlast) as usize;
    for row in 0..height {
        let ri = row * width;
        let mut step = |col: usize, lcol: usize, rcol: usize| {
            let hat = (2.0 * coarse[ri + col] + coarse[ri + lcol] + coarse[ri + rcol]) / 16.0;
            let diff = details[ri + col] - hat;
            details[ri + col] = hat;
            accum[ri + col] += (diff - thold).max(0.0) + (diff + thold).min(0.0);
        };
        // left edge: left tap reflected as hscale − col
        for col in 0..hscale.min(width) {
            step(col, (hscale - col).min(width - 1), (col + hscale).min(width - 1));
        }
        // interior: direct taps
        for col in hscale..width.saturating_sub(hscale) {
            step(col, col - hscale, col + hscale);
        }
        // right edge: right tap reflected around the boundary
        for col in width.saturating_sub(hscale)..width {
            step(
                col,
                clamp_col(col as i64 - hscale as i64),
                clamp_col(2 * width as i64 - 2 - (col as i64 + hscale as i64)),
            );
        }
    }
    if last {
        for (d, a) in details.iter_mut().zip(accum.iter()) {
            *d += *a;
        }
    }
}

// ---------------------------------------------------------------------------
// FFI exports (m4-179) — drive `dwt_denoise_vert_1ch` / `dwt_denoise_horiz_1ch`
// in `src/common/dwt.c`.
// ---------------------------------------------------------------------------

/// Vertical denoise pass over FFI — port of the `dwt_denoise_vert_1ch` loop body.
///
/// `vscale = min(1<<lev, height)`; `out[row] = 2·in[row] + in[|row−vscale|] +
/// in[reflect(row+vscale)]`, reflected taps clamped into range.
///
/// # Safety
/// `out` and `inp` must be non-null, non-overlapping, and each hold at least
/// `width * height` floats (the `dwt_denoise` caller allocates exactly that).
/// Zero `width`/`height` is a no-op.
#[no_mangle]
pub unsafe extern "C" fn darkroom_dwt_denoise_vert_1ch(
    out: *mut f32,
    inp: *const f32,
    height: usize,
    width: usize,
    lev: usize,
) {
    if out.is_null() || inp.is_null() || width == 0 || height == 0 {
        return;
    }
    let Some(npix) = width.checked_mul(height) else {
        return;
    };
    let out = std::slice::from_raw_parts_mut(out, npix);
    let inp = std::slice::from_raw_parts(inp, npix);
    denoise_vert_1ch(out, inp, height, width, lev);
}

/// Horizontal denoise pass over FFI — port of the `dwt_denoise_horiz_1ch` loop body.
///
/// `coarse` is the vertical-pass result (read-only); `details` (the running
/// image) is overwritten with the coarse layer while the soft-thresholded detail
/// `MAX(diff−thold,0)+MIN(diff+thold,0)` accumulates into `accum`. Non-zero
/// `last` folds the accumulation back into `details`.
///
/// # Safety
/// All three pointers must be non-null; `coarse` must not overlap `details` or
/// `accum`; each buffer must hold at least `width * height` floats. Zero
/// `width`/`height` is a no-op.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_dwt_denoise_horiz_1ch(
    coarse: *const f32,
    details: *mut f32,
    accum: *mut f32,
    height: usize,
    width: usize,
    lev: usize,
    thold: f32,
    last: i32,
) {
    if coarse.is_null() || details.is_null() || accum.is_null() || width == 0 || height == 0 {
        return;
    }
    let Some(npix) = width.checked_mul(height) else {
        return;
    };
    let coarse = std::slice::from_raw_parts(coarse, npix);
    let details = std::slice::from_raw_parts_mut(details, npix);
    let accum = std::slice::from_raw_parts_mut(accum, npix);
    denoise_horiz_1ch(coarse, details, accum, height, width, lev, thold, last != 0);
}

// ---------------------------------------------------------------------------
// Structurally divergent reference implementations for the RGBA decompose
// passes (m4-184)
// ---------------------------------------------------------------------------
//
// These compute the same clamped semantics as [`decompose_vert`] /
// [`decompose_horiz`] but with deliberately different loop structure, so the
// differential tests below catch transcription slips rather than re-running the
// same code. The scalar value expressions keep the C association order
// (`2·center + above + below`, `(2·c + l + r) / 16` — that order is
// load-bearing for bit-exact agreement, so it is shared on purpose);
// everything around it differs.

// Reference for [`decompose_vert`]: rows visited in the C's
// `dwt_interleave_rows` order (via [`crate::math::dwt_interleave_rows`]) with
// the taps resolved through `i64` helpers and a single flat per-float loop per
// row (instead of the kernel's natural-order row-major loops with inline
// `usize` index math and a channel sub-loop — all four channels share the same
// row mapping, so the flat loop is equivalent).
//
// `vscale == 0` (1-px-tall images, or any height with `lev` saturating to a
// zero cap) bypasses the interleave helper: the stride-0 call would divide by
// zero (in the C too), while semantically every row maps to itself.
fn decompose_vert_reference(
    out: &mut [f32],
    inp: &[f32],
    height: usize,
    width: usize,
    lev: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    debug_assert_eq!(out.len(), 4 * width * height);
    debug_assert_eq!(inp.len(), 4 * width * height);
    let vscale = pow2(lev).min(height.saturating_sub(1));
    let h = height as i64;
    let stride = 4 * width;
    for rowid in 0..height {
        let row = if vscale == 0 {
            rowid
        } else {
            crate::math::dwt_interleave_rows(rowid, height, vscale)
        } as i64;
        let above = (row - (vscale as i64)).abs().min(h - 1) as usize * stride;
        let centre = row as usize * stride;
        let below = if row + (vscale as i64) < h {
            (row + (vscale as i64)) as usize * stride
        } else {
            (2 * (h - 1) - (row + (vscale as i64))).clamp(0, h - 1) as usize * stride
        };
        let dst = row as usize * stride;
        for k in 0..stride {
            out[dst + k] = 2.0 * inp[centre + k] + inp[above + k] + inp[below + k];
        }
    }
}

// Reference for [`decompose_horiz`]: the same two positional tap ranges driven
// through a per-column `step` closure with flat offset math, and the final
// coarse write-back as an element loop (instead of the kernel's inline tap
// math, channel sub-loops, and `copy_from_slice`).
//
// The two ranges are kept (not unified): when `hscale > width/2` their tap
// reads overlap, but each column is still visited once (the taps are re-read
// from the unmutated `out`, so any shared tap value is identical). The
// reference reproduces that shared-tap behavior.
#[allow(clippy::manual_memcpy)] // the element loop is intentional divergence
// from the kernel's copy_from_slice.
fn decompose_horiz_reference(
    out: &mut [f32],
    inp: &mut [f32],
    height: usize,
    width: usize,
    lev: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    debug_assert_eq!(out.len(), 4 * width * height);
    debug_assert_eq!(inp.len(), 4 * width * height);
    let hscale = pow2(lev).min(width);
    let hscale_i = hscale as i64;
    let wlast = width.saturating_sub(1) as i64;
    let clamp_col = |c: i64| c.clamp(0, wlast) as usize;
    let mut temprow = vec![0.0f32; 4 * width];
    for row in 0..height {
        let ri = 4 * row * width;
        let mut step = |col: usize, lcol: usize, rcol: usize| {
            for c in 0..4 {
                let hat =
                    (2.0 * out[ri + 4 * col + c] + out[ri + 4 * lcol + c] + out[ri + 4 * rcol + c])
                        / 16.0;
                temprow[4 * col + c] = hat;
                inp[ri + 4 * col + c] -= hat;
            }
        };
        // interior columns: reflected left, direct right
        for col in 0..width.saturating_sub(hscale) {
            step(
                col,
                ((col as i64 - hscale_i).unsigned_abs() as usize).min(width - 1),
                col + hscale,
            );
        }
        // right edge: reflect the right tap around the image boundary. Note the
        // left tap keeps the C's abs() (unlike the 1-ch denoise port, whose C
        // clamps with max(0)): |col−hscale| then clamped into range.
        for col in width.saturating_sub(hscale)..width {
            step(
                col,
                ((col as i64 - hscale_i).abs() as usize).min(width - 1),
                clamp_col(2 * width as i64 - 2 - (col as i64 + hscale_i)),
            );
        }
        for k in 0..4 * width {
            out[ri + k] = temprow[k];
        }
    }
}

// ---------------------------------------------------------------------------
// FFI exports (m4-184) — drive `dwt_decompose_vert` / `dwt_decompose_horiz`
// in `src/common/dwt.c`.
// ---------------------------------------------------------------------------

/// Vertical decompose pass over FFI — port of the `dwt_decompose_vert` loop body.
///
/// `vscale = min(1<<lev, height-1)` (saturating shift); each output row is
/// `2·center + above + below` with top/bottom reflection. Rows run in natural
/// order (the C row interleave is a pure cache optimisation over a read-only
/// input — see the module docs). `out` and `inp` must be non-null,
/// non-overlapping, and each hold at least `4 * width * height` floats. Zero
/// `width`/`height` is a no-op.
///
/// # Safety
/// See the buffer contract above; overflowing dimension products are also
/// guarded no-ops.
#[no_mangle]
pub unsafe extern "C" fn darkroom_dwt_decompose_vert(
    out: *mut f32,
    inp: *const f32,
    height: usize,
    width: usize,
    lev: usize,
) {
    if out.is_null() || inp.is_null() || width == 0 || height == 0 {
        return;
    }
    let Some(npix) = width.checked_mul(height).and_then(|n| n.checked_mul(4)) else {
        return;
    };
    let out = std::slice::from_raw_parts_mut(out, npix);
    let inp = std::slice::from_raw_parts(inp, npix);
    decompose_vert(out, inp, height, width, lev);
}

/// Horizontal decompose pass over FFI — port of the `dwt_decompose_horiz` loop body.
///
/// Reads the vertical-pass 'coarse' from `out`, writes the normalised
/// `(2·center + left + right) / 16` back into `out`, and subtracts it from
/// `details` (`inp`, the running image) to leave the detail layer. Edge taps
/// reflect around the image boundary (clamped into range); when
/// `hscale = min(1<<lev, width)` exceeds `width/2` the edge ranges share taps
/// (each column is still visited once), exactly like the C. Carries its own
/// scratch — the C `temp`/`padded_size` per-thread buffer is retained (but no
/// longer consumed) for signature stability.
///
/// # Safety
/// `out` and `details` must be non-null, non-overlapping, and each hold at
/// least `4 * width * height` floats. Zero `width`/`height` is a no-op;
/// overflowing dimension products are guarded no-ops.
#[no_mangle]
pub unsafe extern "C" fn darkroom_dwt_decompose_horiz(
    out: *mut f32,
    details: *mut f32,
    height: usize,
    width: usize,
    lev: usize,
) {
    if out.is_null() || details.is_null() || width == 0 || height == 0 {
        return;
    }
    let Some(npix) = width.checked_mul(height).and_then(|n| n.checked_mul(4)) else {
        return;
    };
    let out = std::slice::from_raw_parts_mut(out, npix);
    let details = std::slice::from_raw_parts_mut(details, npix);
    decompose_horiz(out, details, height, width, lev);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(width: usize, height: usize, scales: i32, return_layer: i32) -> DwtParams {
        DwtParams {
            width,
            height,
            ch: 4,
            scales,
            return_layer,
            merge_from_scale: 0,
            preview_scale: 1.0,
        }
    }

    /// Pseudo-random but deterministic RGBA fill.
    fn fill(width: usize, height: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 4 * width * height];
        let mut s: u32 = 0x1234_5678;
        for x in v.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *x = (s >> 8) as f32 / 16_777_216.0; // in [0, 1)
        }
        v
    }

    #[test]
    fn max_scale_square_256() {
        // hand-traced: 256 → 7
        assert_eq!(params(256, 256, 8, 0).get_max_scale(), 7);
    }

    #[test]
    fn perfect_reconstruction() {
        // decompose + recompose with an identity callback == original image.
        let (w, h) = (24, 20);
        let orig = fill(w, h);
        let mut img = orig.clone();
        let mut p = params(w, h, 4, 0);
        decompose(&mut img, &mut p, |_layer, _p, _s| {});
        for (a, b) in img.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-3, "reconstruction drift {a} vs {b}");
        }
    }

    #[test]
    fn reconstruction_with_merge() {
        // merging the coarse scales must still recompose the original.
        let (w, h) = (30, 18);
        let orig = fill(w, h);
        let mut img = orig.clone();
        let mut p = params(w, h, 4, 0);
        p.merge_from_scale = 2;
        decompose(&mut img, &mut p, |_l, _p, _s| {});
        for (a, b) in img.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-3, "merge reconstruction drift {a} vs {b}");
        }
    }

    #[test]
    fn constant_image_has_no_detail() {
        // a flat image: every detail scale is zero, the residual is the constant.
        let (w, h) = (16, 16);
        let flat = vec![0.375f32; 4 * w * h];

        // detail scale 1 → all zeros
        let mut d1 = flat.clone();
        let mut p1 = params(w, h, 3, 1);
        decompose(&mut d1, &mut p1, |_l, _p, _s| {});
        assert!(d1.iter().all(|v| v.abs() < 1e-6), "flat image had detail");

        // residual → still the constant
        let mut res = flat.clone();
        let mut pr = params(w, h, 3, p1.scales + 1);
        decompose(&mut res, &mut pr, |_l, _p, _s| {});
        assert!(res.iter().all(|v| (v - 0.375).abs() < 1e-5), "flat residual moved");
    }

    #[test]
    fn callback_sees_every_scale() {
        let (w, h) = (20, 20);
        let mut img = fill(w, h);
        let mut p = params(w, h, 3, 0);
        let mut scales_seen = Vec::new();
        decompose(&mut img, &mut p, |_l, _p, s| scales_seen.push(s));
        // 0 (orig), 1..=3 (details), 4 (residual), 5 (reconstruction)
        assert_eq!(scales_seen, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn denoise_zero_threshold_is_identity() {
        let (w, h) = (18, 14);
        let orig = fill(w, h);
        let mut img: Vec<f32> = orig.iter().take(w * h).copied().collect();
        let before = img.clone();
        denoise(&mut img, w, h, 4, &[0.0; 4]);
        for (a, b) in img.iter().zip(before.iter()) {
            assert!((a - b).abs() < 1e-4, "zero-threshold denoise changed image");
        }
    }

    #[test]
    fn denoise_huge_threshold_removes_all_detail() {
        // with a threshold larger than any detail, only the coarsest residual
        // survives, so the output is a heavily smoothed version — strictly
        // different from the (non-flat) input, and finite.
        let (w, h) = (18, 14);
        let src = fill(w, h);
        let mut img: Vec<f32> = src.iter().take(w * h).copied().collect();
        let before = img.clone();
        denoise(&mut img, w, h, 4, &[1.0e9; 4]);
        assert!(img.iter().all(|v| v.is_finite()));
        let changed = img.iter().zip(before.iter()).any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(changed, "huge-threshold denoise left detail in place");
    }

    #[test]
    fn degenerate_dims_no_panic() {
        // 1-px-wide / 1-px-tall images and over-large scale requests must not panic.
        for (w, h) in [(1, 8), (8, 1), (1, 1), (3, 3)] {
            let mut img = fill(w, h);
            let mut p = params(w, h, 6, 0);
            decompose(&mut img, &mut p, |_l, _p, _s| {});
            assert!(img.iter().all(|v| v.is_finite()));

            let mut d: Vec<f32> = fill(w, h).into_iter().take(w * h).collect();
            denoise(&mut d, w, h, 6, &[0.5; 6]);
            assert!(d.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn detail_scales_plus_residual_recompose() {
        // value-level telescoping: summing every individual detail scale plus the
        // residual must equal the original — catches a detail-scale off-by-one that
        // the flat-image test (all-zero details) would miss.
        let (w, h) = (24, 20);
        let orig = fill(w, h);
        let scales = 4; // == get_max_scale(24×20), so no clamping
        assert_eq!(params(w, h, scales, 0).get_max_scale(), scales);

        let mut sum = vec![0.0f32; 4 * w * h];
        for k in 1..=scales {
            let mut layer = orig.clone();
            let mut p = params(w, h, scales, k);
            decompose(&mut layer, &mut p, |_l, _p, _s| {});
            image_add(&mut sum, &layer);
        }
        let mut resid = orig.clone();
        let mut pr = params(w, h, scales, scales + 1);
        decompose(&mut resid, &mut pr, |_l, _p, _s| {});
        image_add(&mut sum, &resid);

        for (a, b) in sum.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-3, "sum(details)+residual != original: {a} vs {b}");
        }
    }

    #[test]
    fn preview_scale_clamps_scales() {
        // a downscaled preview must clamp an over-large `scales` to get_max_scale,
        // and return_layer=0 still reconstructs the original regardless.
        let (w, h) = (64, 64);
        let orig = fill(w, h);
        let mut img = orig.clone();
        let mut p = params(w, h, 20, 0);
        p.preview_scale = 0.5;
        let expected_max = {
            let mut q = params(w, h, 20, 0);
            q.preview_scale = 0.5;
            q.get_max_scale()
        };
        decompose(&mut img, &mut p, |_l, _p, _s| {});
        assert_eq!(p.scales, expected_max, "scales not clamped to max_scale under zoom");
        assert!(expected_max < 20 && expected_max > 0, "sanity: clamp actually engaged");
        for (a, b) in img.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-3, "zoomed reconstruction drift {a} vs {b}");
        }
    }

    #[test]
    fn merged_return_layer_matches_sum_of_details() {
        // returning a *merged* scale must equal the sum of the individual detail
        // scales in the merge range (locks the merge-branch accumulation path).
        let (w, h) = (40, 32);
        let orig = fill(w, h);
        let scales = 4; // == get_max_scale(40×32)
        assert_eq!(params(w, h, scales, 0).get_max_scale(), scales);
        let (merge_from, ret) = (2, 3);

        let mut merged = orig.clone();
        let mut pm = params(w, h, scales, ret);
        pm.merge_from_scale = merge_from;
        decompose(&mut merged, &mut pm, |_l, _p, _s| {});

        let mut sum = vec![0.0f32; 4 * w * h];
        for k in merge_from..=ret {
            let mut layer = orig.clone();
            let mut p = params(w, h, scales, k);
            decompose(&mut layer, &mut p, |_l, _p, _s| {});
            image_add(&mut sum, &layer);
        }

        for (a, b) in merged.iter().zip(sum.iter()) {
            assert!((a - b).abs() < 1e-3, "merged scale != sum of its details: {a} vs {b}");
        }
    }

    #[test]
    fn negative_return_layer_reconstructs() {
        // out-of-contract negative return_layer is clamped to 0 (recomposed image).
        let (w, h) = (16, 16);
        let orig = fill(w, h);
        let mut img = orig.clone();
        let mut p = params(w, h, 3, -5);
        decompose(&mut img, &mut p, |_l, _p, _s| {});
        assert_eq!(p.return_layer, 0);
        for (a, b) in img.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn first_scale_visible_basic() {
        // full res (preview_scale 1): 1<<0 * 1 = 1 > 0 at lev 0 → returns 1.
        assert_eq!(params(64, 64, 4, 0).first_scale_visible(), 1);
        // heavily zoomed out (preview_scale 0.1): 1<<0*0.1 = 0 (int), 1<<1*0.1 = 0,
        // ... 1<<4*0.1 = 1 → first visible detail scale is lev 4 → 5.
        let mut p = params(64, 64, 8, 0);
        p.preview_scale = 0.1;
        assert_eq!(p.first_scale_visible(), 5);
    }

    // ── m4-179: 1-channel denoise passes ─────────────────────────────────────

    /// Deterministic 1-channel fill in [0, 1).
    fn fill1(n: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        let mut s: u32 = 0x9e37_79b9;
        for _ in 0..n {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push((s >> 8) as f32 / 16_777_216.0);
        }
        v
    }

    /// Row-constant image: pixel (r, c) == r as f32.
    fn row_ramp(width: usize, height: usize) -> Vec<f32> {
        (0..height)
            .flat_map(|r| std::iter::repeat_n(r as f32, width))
            .collect()
    }

    /// Elementwise closeness assertion with per-index diagnostics.
    fn assert_close(got: &[f32], expected: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() <= tol, "{what}[{i}]: {g} vs {e}");
        }
    }

    #[test]
    fn denoise_vert_top_and_bottom_reflection() {
        // lev 0 → vscale 1 on a 4×5 row ramp. Top row reflects around row 0
        // (|0−1| = 1), bottom row around height−1 (2·4−(4+1) = 3).
        let (w, h) = (4, 5);
        let inp = row_ramp(w, h);
        let mut out = vec![0.0f32; w * h];
        denoise_vert_1ch(&mut out, &inp, h, w, 0);
        // row 0: 2·0 + 1 + 1 = 2; row 2 (interior): 2·2 + 1 + 3 = 8;
        // row 4: 2·4 + 3 + 3 = 14.
        assert!(out[0..w].iter().all(|&v| v == 2.0), "top reflection: {out:?}");
        assert!(
            out[2 * w..3 * w].iter().all(|&v| v == 8.0),
            "interior: {out:?}"
        );
        assert!(
            out[4 * w..5 * w].iter().all(|&v| v == 14.0),
            "bottom reflection: {out:?}"
        );
    }

    #[test]
    fn denoise_vert_wider_scale_reflection() {
        // lev 1 → vscale 2 on a 3×6 row ramp: row 0 reads |0−2| = 2,
        // row 5 reads 2·5−(5+2) = 3.
        let (w, h) = (3, 6);
        let inp = row_ramp(w, h);
        let mut out = vec![0.0f32; w * h];
        denoise_vert_1ch(&mut out, &inp, h, w, 1);
        assert!(out[0..w].iter().all(|&v| v == 4.0), "top: {out:?}");
        assert!(out[5 * w..6 * w].iter().all(|&v| v == 16.0), "bottom: {out:?}");
    }

    #[test]
    fn denoise_vert_overscale_clamps_instead_of_oob() {
        // height 3, lev 5 → vscale capped to 3; the top tap |0−3| = 3 is out of
        // range (C would read past the buffer) and clamps to row 2, while the
        // bottom tap 2·2−(0+3) = 1 stays in range.
        let (w, h) = (2, 3);
        let inp = row_ramp(w, h);
        let mut out = vec![0.0f32; w * h];
        denoise_vert_1ch(&mut out, &inp, h, w, 5);
        assert!(out[0..w].iter().all(|&v| v == 3.0), "clamped top: {out:?}");
    }

    #[test]
    fn denoise_horiz_edges_hat_and_zero_threshold() {
        // width 5, lev 0 → hscale 1. Coarse and details start identical, so with
        // thold 0 the accumulator receives the full detail (diff) at every
        // column, exercising the left-reflection, interior, and
        // right-reflection tap selections against hand-computed hats.
        let (w, h) = (5, 1);
        let coarse = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut details = coarse.clone();
        let mut accum = vec![0.0f32; w * h];
        denoise_horiz_1ch(&coarse, &mut details, &mut accum, h, w, 0, 0.0, false);
        let hats = [3.75f32, 5.0, 7.5, 10.0, 11.25];
        assert_close(&details, &hats, 1e-6, "hat");
        let expected_accum: Vec<f32> =
            coarse.iter().zip(hats.iter()).map(|(&c, &e)| c - e).collect();
        assert_close(&accum, &expected_accum, 1e-6, "accum");
    }

    #[test]
    fn denoise_horiz_overlap_filters_shared_column_twice() {
        // width 3, lev 1 → hscale 2 overlaps the left/right ranges. The shared
        // column must be filtered first from the input detail, then again from
        // the already-overwritten detail value.
        let (w, h) = (3, 1);
        let coarse = vec![0.0f32, 16.0, 32.0];
        let expected_details = vec![4.0f32, 3.0, 4.0];
        let expected_accum = vec![-3.0f32, -1.0, -1.0];
        for use_reference in [false, true] {
            let mut details = vec![1.0f32, 2.0, 3.0];
            let mut accum = vec![0.0f32; w * h];
            if use_reference {
                denoise_horiz_1ch_reference(&coarse, &mut details, &mut accum, h, w, 1, 0.0, false);
            } else {
                denoise_horiz_1ch(&coarse, &mut details, &mut accum, h, w, 1, 0.0, false);
            }
            assert_eq!(details, expected_details, "use_reference={use_reference}");
            assert_eq!(accum, expected_accum, "use_reference={use_reference}");
        }
    }

    #[test]
    fn denoise_horiz_threshold_gates_accum_not_coarse() {
        // A threshold larger than any detail zeroes the accumulation, but the
        // coarse layer is still written into `details`.
        let (w, h) = (5, 1);
        let coarse = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut details = coarse.clone();
        let mut accum = vec![0.0f32; w * h];
        denoise_horiz_1ch(&coarse, &mut details, &mut accum, h, w, 0, 1.0e9, false);
        assert!(accum.iter().all(|&v| v == 0.0));
        let hats = [3.75f32, 5.0, 7.5, 10.0, 11.25];
        assert_close(&details, &hats, 1e-6, "coarse");
    }

    #[test]
    fn denoise_horiz_soft_threshold_branches_and_deadzone() {
        // Flat-zero coarse → hat 0 everywhere, so diff == details. With
        // thold 1 the dead zone (|diff| ≤ 1) contributes nothing, positive
        // excess keeps diff−thold, negative excess keeps diff+thold — the C
        // MAX(diff−thold,0)+MIN(diff+thold,0) behaviour on both sides.
        let (w, h) = (5, 1);
        let coarse = vec![0.0f32; w];
        let mut details = vec![-50.0f32, -0.25, 0.0, 0.25, 50.0];
        let mut accum = vec![0.0f32; w];
        denoise_horiz_1ch(&coarse, &mut details, &mut accum, h, w, 0, 1.0, false);
        assert!(details.iter().all(|&v| v == 0.0), "hats must be zero");
        let expected = [-49.0f32, 0.0, 0.0, 0.0, 49.0];
        assert_close(&accum, &expected, 1e-6, "soft threshold");
    }

    #[test]
    fn denoise_horiz_threshold_edge_values_are_exact() {
        // Single-pixel input makes the horizontal hat exactly zero, isolating
        // the soft-threshold expression on the supplied detail value.
        let (w, h) = (1, 1);
        let coarse = vec![0.0f32];
        for (input, thold, expected) in [
            (f32::NAN, 0.25f32, 0.0f32),
            (2.0f32, -1.0f32, 3.0f32),
            (-2.0f32, -1.0f32, -3.0f32),
            (0.0f32, 1.0f32, 0.0f32),
            (-0.0f32, 1.0f32, 0.0f32),
        ] {
            let mut details = vec![input];
            let mut accum = vec![0.0f32];
            denoise_horiz_1ch(&coarse, &mut details, &mut accum, h, w, 0, thold, false);
            assert_eq!(details[0].to_bits(), 0.0f32.to_bits(), "input={input}");
            assert_eq!(
                accum[0].to_bits(),
                expected.to_bits(),
                "input={input} thold={thold}"
            );
        }
    }

    #[test]
    fn denoise_horiz_last_fold_restores_input_at_zero_threshold() {
        // last=true with thold 0: details = hat + (orig − hat) == orig.
        let (w, h) = (9, 3);
        let coarse = fill1(w * h);
        let orig = row_ramp(w, h);
        let mut details = orig.clone();
        let mut accum = vec![0.0f32; w * h];
        denoise_horiz_1ch(&coarse, &mut details, &mut accum, h, w, 1, 0.0, true);
        for (d, o) in details.iter().zip(orig.iter()) {
            assert!((d - o).abs() < 1e-5, "last fold did not restore input");
        }
    }

    #[test]
    fn denoise_horiz_last_false_leaves_fold_out() {
        // last=false must not add the accumulation back into details.
        let (w, h) = (9, 3);
        let coarse = fill1(w * h);
        let orig = row_ramp(w, h);
        let mut details = orig.clone();
        let mut accum = vec![0.0f32; w * h];
        denoise_horiz_1ch(&coarse, &mut details, &mut accum, h, w, 1, 0.0, false);
        let changed = details.iter().zip(orig.iter()).any(|(d, o)| (d - o).abs() > 1e-6);
        assert!(changed, "details unexpectedly unchanged without fold");
        // The accumulation is non-trivial, so the test above is not vacuous:
        // skipping the fold must leave details different from a folded run.
        assert!(accum.iter().any(|a| *a != 0.0), "accum unexpectedly all zero");
        let mut folded = details.clone();
        for (d, a) in folded.iter_mut().zip(accum.iter()) {
            *d += *a;
        }
        assert!(
            folded.iter().zip(orig.iter()).all(|(d, o)| (d - o).abs() < 1e-5),
            "folded details should restore the input at zero threshold"
        );
    }

    #[test]
    fn denoise_kernels_agree_with_references_bit_exact() {
        // Differential agreement over sizes (including short/degenerate ones
        // where edge regions overlap), scales, thresholds, and last flags.
        let dims = [(1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (5, 5), (9, 3), (16, 12)];
        let tholds = [0.0f32, 0.25, 1.0e9];
        for &(w, h) in &dims {
            for lev in 0..4 {
                // vertical pass
                let inp = fill1(w * h);
                let mut a = vec![0.0f32; w * h];
                let mut b = vec![0.0f32; w * h];
                denoise_vert_1ch(&mut a, &inp, h, w, lev);
                denoise_vert_1ch_reference(&mut b, &inp, h, w, lev);
                assert!(
                    a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "vert mismatch {w}x{h} lev {lev}"
                );
                assert!(a.iter().all(|v| v.is_finite()));
                // horizontal pass
                for &th in &tholds {
                    for &last in &[false, true] {
                        let coarse = fill1(w * h);
                        let orig = fill1(w * h);
                        let mut d1 = orig.clone();
                        let mut d2 = orig.clone();
                        let mut a1 = vec![0.0f32; w * h];
                        let mut a2 = vec![0.0f32; w * h];
                        denoise_horiz_1ch(&coarse, &mut d1, &mut a1, h, w, lev, th, last);
                        denoise_horiz_1ch_reference(&coarse, &mut d2, &mut a2, h, w, lev, th, last);
                        assert!(
                            d1.iter().zip(d2.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "horiz details mismatch {w}x{h} lev {lev} th {th} last {last}"
                        );
                        assert!(
                            a1.iter().zip(a2.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "horiz accum mismatch {w}x{h} lev {lev} th {th} last {last}"
                        );
                        assert!(d1.iter().all(|v| v.is_finite()));
                        assert!(a1.iter().all(|v| v.is_finite()));
                    }
                }
            }
        }
    }

    #[test]
    fn denoise_ffi_matches_safe_kernels() {
        // The FFI exports must pass lev/thold/last/width/height through
        // unchanged, including non-zero `last` values other than 1.
        let (w, h) = (11, 7);
        let inp = fill1(w * h);
        let mut via_safe = vec![0.0f32; w * h];
        let mut via_ffi = vec![0.0f32; w * h];
        denoise_vert_1ch(&mut via_safe, &inp, h, w, 2);
        unsafe {
            darkroom_dwt_denoise_vert_1ch(via_ffi.as_mut_ptr(), inp.as_ptr(), h, w, 2);
        }
        assert!(via_safe.iter().zip(via_ffi.iter()).all(|(x, y)| x.to_bits() == y.to_bits()));

        for &last in &[0, 1, 2] {
            let coarse = fill1(w * h);
            let orig = fill1(w * h);
            let mut d1 = orig.clone();
            let mut d2 = orig.clone();
            let mut a1 = vec![0.0f32; w * h];
            let mut a2 = vec![0.0f32; w * h];
            denoise_horiz_1ch(&coarse, &mut d1, &mut a1, h, w, 2, 0.125, last != 0);
            unsafe {
                darkroom_dwt_denoise_horiz_1ch(
                    coarse.as_ptr(),
                    d2.as_mut_ptr(),
                    a2.as_mut_ptr(),
                    h,
                    w,
                    2,
                    0.125,
                    last,
                );
            }
            assert!(
                d1.iter().zip(d2.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
                "ffi details mismatch last={last}"
            );
            assert!(
                a1.iter().zip(a2.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
                "ffi accum mismatch last={last}"
            );
        }
    }

    #[test]
    fn denoise_ffi_guards_do_not_crash() {
        // Null pointers and degenerate dimensions are no-ops, never UB.
        unsafe {
            darkroom_dwt_denoise_vert_1ch(std::ptr::null_mut(), std::ptr::null(), 4, 4, 0);
            darkroom_dwt_denoise_vert_1ch(std::ptr::null_mut(), std::ptr::null(), 0, 0, 0);
            darkroom_dwt_denoise_horiz_1ch(
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                4,
                4,
                0,
                0.0,
                0,
            );
            darkroom_dwt_denoise_horiz_1ch(
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                0,
                0,
                0.0,
                1,
            );

            // Every null-pointer permutation is rejected before any slice exists.
            let valid_in = vec![1.0f32; 16];
            let mut valid_out = vec![2.0f32; 16];
            darkroom_dwt_denoise_vert_1ch(
                std::ptr::null_mut(),
                valid_in.as_ptr(),
                4,
                4,
                0,
            );
            darkroom_dwt_denoise_vert_1ch(
                valid_out.as_mut_ptr(),
                std::ptr::null(),
                4,
                4,
                0,
            );
            let valid_coarse = vec![3.0f32; 16];
            let mut valid_details = vec![4.0f32; 16];
            let mut valid_accum = vec![5.0f32; 16];
            darkroom_dwt_denoise_horiz_1ch(
                std::ptr::null(),
                valid_details.as_mut_ptr(),
                valid_accum.as_mut_ptr(),
                4,
                4,
                0,
                0.0,
                0,
            );
            darkroom_dwt_denoise_horiz_1ch(
                valid_coarse.as_ptr(),
                std::ptr::null_mut(),
                valid_accum.as_mut_ptr(),
                4,
                4,
                0,
                0.0,
                0,
            );
            darkroom_dwt_denoise_horiz_1ch(
                valid_coarse.as_ptr(),
                valid_details.as_mut_ptr(),
                std::ptr::null_mut(),
                4,
                4,
                0,
                0.0,
                0,
            );

            // Partial-zero dimensions are also no-ops when the other pointers
            // are valid, and must leave those buffers untouched.
            darkroom_dwt_denoise_vert_1ch(valid_out.as_mut_ptr(), valid_in.as_ptr(), 0, 4, 0);
            darkroom_dwt_denoise_vert_1ch(valid_out.as_mut_ptr(), valid_in.as_ptr(), 4, 0, 0);
            darkroom_dwt_denoise_horiz_1ch(
                valid_coarse.as_ptr(),
                valid_details.as_mut_ptr(),
                valid_accum.as_mut_ptr(),
                0,
                4,
                0,
                0.0,
                0,
            );
            darkroom_dwt_denoise_horiz_1ch(
                valid_coarse.as_ptr(),
                valid_details.as_mut_ptr(),
                valid_accum.as_mut_ptr(),
                4,
                0,
                0,
                0.0,
                0,
            );
            assert!(valid_out.iter().all(|&v| v == 2.0));
            assert!(valid_details.iter().all(|&v| v == 4.0));
            assert!(valid_accum.iter().all(|&v| v == 5.0));

            // Overflowing dimension products are rejected before slices exist;
            // these pointers are never dereferenced.
            let dangling = std::ptr::NonNull::<f32>::dangling().as_ptr();
            darkroom_dwt_denoise_vert_1ch(
                dangling as *mut f32,
                dangling,
                usize::MAX,
                2,
                0,
            );
            darkroom_dwt_denoise_horiz_1ch(
                dangling,
                dangling as *mut f32,
                dangling as *mut f32,
                2,
                usize::MAX,
                0,
                0.0,
                0,
            );
        }
        // Degenerate dimensions on the safe kernels are no-ops too.
        let mut empty: Vec<f32> = Vec::new();
        denoise_vert_1ch(&mut empty, &[], 0, 0, 0);
        denoise_vert_1ch(&mut empty, &[], 0, 4, 0);
        denoise_vert_1ch(&mut empty, &[], 4, 0, 0);
        denoise_horiz_1ch(&[], &mut [], &mut [], 0, 0, 0, 0.0, true);
        denoise_horiz_1ch(&[], &mut [], &mut [], 0, 4, 0, 0.0, true);
        denoise_horiz_1ch(&[], &mut [], &mut [], 4, 0, 0, 0.0, true);
    }

    // ── m4-184: RGBA decompose passes ───────────────────────────────────────

    /// Deterministic RGBA fill in [0, 1).
    fn fill4(width: usize, height: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(4 * width * height);
        let mut s: u32 = 0x51ab_3f29;
        for _ in 0..4 * width * height {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push((s >> 8) as f32 / 16_777_216.0);
        }
        v
    }

    /// RGBA row ramp: pixel (r, c) holds r as f32 in all four channels.
    fn row_ramp4(width: usize, height: usize) -> Vec<f32> {
        (0..height)
            .flat_map(|r| std::iter::repeat_n([r as f32; 4], width).flatten())
            .collect()
    }

    /// Elementwise `to_bits` equality with per-index diagnostics.
    fn assert_bits_eq(got: &[f32], expected: &[f32], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(g.to_bits() == e.to_bits(), "{what}[{i}]: {g} vs {e}");
        }
    }

    #[test]
    fn decompose_vert_top_and_bottom_reflection() {
        // lev 0 → vscale 1 on a 4×5 row ramp (all channels equal). Top row
        // reflects around row 0 (|0−1| = 1), bottom row around height−1
        // (2·4−(4+1) = 3).
        let (w, h) = (4, 5);
        let inp = row_ramp4(w, h);
        let mut out = vec![0.0f32; 4 * w * h];
        decompose_vert(&mut out, &inp, h, w, 0);
        // row 0: 2·0 + 1 + 1 = 2; row 2 (interior): 2·2 + 1 + 3 = 8;
        // row 4: 2·4 + 3 + 3 = 14 — in every channel.
        for (r, expected) in [(0usize, 2.0f32), (2, 8.0), (4, 14.0)] {
            assert!(
                out[4 * r * w..4 * (r + 1) * w].iter().all(|&v| v == expected),
                "row {r}: {:?}",
                &out[4 * r * w..4 * (r + 1) * w]
            );
        }
    }

    #[test]
    fn decompose_vert_wider_scale_reflection() {
        // lev 1 → vscale 2 on a 3×6 row ramp: row 0 reads |0−2| = 2
        // (2·0 + 2 + 2 = 4), row 5 reads 2·5−(5+2) = 3 (2·5 + 3 + 3 = 16).
        let (w, h) = (3, 6);
        let inp = row_ramp4(w, h);
        let mut out = vec![0.0f32; 4 * w * h];
        decompose_vert(&mut out, &inp, h, w, 1);
        assert!(out[0..4 * w].iter().all(|&v| v == 4.0), "top: {out:?}");
        assert!(
            out[4 * 5 * w..].iter().all(|&v| v == 16.0),
            "bottom: {out:?}"
        );
    }

    #[test]
    fn decompose_vert_lev_clamping() {
        // Absurd `lev` must saturate (never panic/shift-overflow) and behave
        // exactly like the capped scale: height 5 caps vscale at 4, i.e. lev 2.
        let (w, h) = (4, 5);
        let inp = fill4(w, h);
        let mut a = vec![0.0f32; 4 * w * h];
        let mut b = vec![0.0f32; 4 * w * h];
        let mut c = vec![0.0f32; 4 * w * h];
        decompose_vert(&mut a, &inp, h, w, 2);
        decompose_vert(&mut b, &inp, h, w, 100);
        decompose_vert(&mut c, &inp, h, w, usize::MAX);
        assert_bits_eq(&b, &a, "lev 100 vs capped lev 2");
        assert_bits_eq(&c, &a, "lev MAX vs capped lev 2");
    }

    #[test]
    fn decompose_vert_height_one_is_scaled_copy() {
        // height 1 → vscale 0, so every tap is the row itself: out = 4·in.
        // (The C would divide by zero in `dwt_interleave_rows(rowid, 1, 0)`;
        // the kernel's natural order has no such edge.)
        let (w, h) = (3, 1);
        let inp = fill4(w, h);
        let mut out = vec![0.0f32; 4 * w * h];
        decompose_vert(&mut out, &inp, h, w, 0);
        for (o, i) in out.iter().zip(inp.iter()) {
            assert!((o - 4.0 * i).abs() <= 1e-6, "{o} vs 4·{i}");
        }
        // …and the reference agrees bit-exactly (it bypasses the interleave
        // helper for vscale 0 for the same reason).
        let mut refr = vec![0.0f32; 4 * w * h];
        decompose_vert_reference(&mut refr, &inp, h, w, 0);
        assert_bits_eq(&refr, &out, "height-1 reference");
    }

    #[test]
    fn decompose_horiz_edges_hat_and_details() {
        // width 5, lev 0 → hscale 1. Coarse and details start identical; with
        // hand-computed hats [3.75, 5, 7.5, 10, 11.25] the coarse row must
        // equal the hats and details must equal input − hat, in all channels.
        let (w, h) = (5, 1);
        let pix = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let coarse: Vec<f32> = pix.iter().flat_map(|&v| [v; 4]).collect();
        let mut out = coarse.clone();
        let mut details = coarse.clone();
        decompose_horiz(&mut out, &mut details, h, w, 0);
        let hats = [3.75f32, 5.0, 7.5, 10.0, 11.25];
        let expected_coarse: Vec<f32> = hats.iter().flat_map(|&v| [v; 4]).collect();
        let expected_details: Vec<f32> = pix
            .iter()
            .zip(hats.iter())
            .flat_map(|(&c, &e)| [c - e; 4])
            .collect();
        assert_close(&out, &expected_coarse, 1e-6, "coarse");
        assert_close(&details, &expected_details, 1e-6, "details");
    }

    #[test]
    fn decompose_horiz_overlap_shares_taps() {
        // width 3, lev 1 → hscale 2 exceeds width/2, so the edge ranges share
        // taps. Each column is still visited once (taps re-read from the
        // unmutated `out`): details = orig − hat there.
        // coarse [0,16,32], details start [1,2,3]:
        //   col 0: hat (0+32+32)/16 = 4 → temp 4, details 1−4 = −3
        //   col 1: hat (32+16+16)/16 = 4 → temp 4, details 2−4 = −2
        //   col 2: hat (64+0+0)/16 = 4 → temp 4, details 3−4 = −1
        let (w, h) = (3, 1);
        let coarse: Vec<f32> = [0.0f32, 16.0, 32.0].iter().flat_map(|&v| [v; 4]).collect();
        let expected_coarse = vec![4.0f32; 4 * w * h];
        let expected_details: Vec<f32> = [-3.0f32, -2.0, -1.0].iter().flat_map(|&v| [v; 4]).collect();
        for use_reference in [false, true] {
            let mut out = coarse.clone();
            let mut details: Vec<f32> =
                [1.0f32, 2.0, 3.0].iter().flat_map(|&v| [v; 4]).collect();
            if use_reference {
                decompose_horiz_reference(&mut out, &mut details, h, w, 1);
            } else {
                decompose_horiz(&mut out, &mut details, h, w, 1);
            }
            assert_eq!(out, expected_coarse, "use_reference={use_reference}");
            assert_eq!(details, expected_details, "use_reference={use_reference}");
        }
    }

    #[test]
    fn decompose_kernels_agree_with_references_bit_exact() {
        // Differential agreement over sizes (including short/degenerate ones
        // where edge regions overlap or scales exceed the dims) and scales.
        let dims = [
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 3),
            (3, 2),
            (3, 3),
            (5, 5),
            (9, 3),
            (16, 12),
        ];
        for &(w, h) in &dims {
            for lev in 0..6 {
                // vertical pass
                let inp = fill4(w, h);
                let mut a = vec![0.0f32; 4 * w * h];
                let mut b = vec![0.0f32; 4 * w * h];
                decompose_vert(&mut a, &inp, h, w, lev);
                decompose_vert_reference(&mut b, &inp, h, w, lev);
                assert_bits_eq(&a, &b, "vert {w}x{h} lev {lev}");
                assert!(a.iter().all(|v| v.is_finite()));
                // horizontal pass
                let coarse = fill4(w, h);
                let orig = fill4(w, h);
                let mut o1 = coarse.clone();
                let mut o2 = coarse.clone();
                let mut d1 = orig.clone();
                let mut d2 = orig.clone();
                decompose_horiz(&mut o1, &mut d1, h, w, lev);
                decompose_horiz_reference(&mut o2, &mut d2, h, w, lev);
                assert_bits_eq(&o1, &o2, "horiz coarse {w}x{h} lev {lev}");
                assert_bits_eq(&d1, &d2, "horiz details {w}x{h} lev {lev}");
                assert!(o1.iter().all(|v| v.is_finite()));
                assert!(d1.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn decompose_vert_interleave_order_equivalent() {
        // Non-trivial interleave strides (vscale > 1 with height not a multiple
        // of vscale): the natural-order kernel must equal the
        // interleave-ordered reference bit-exactly.
        let (w, h) = (6, 18);
        let inp = fill4(w, h);
        for lev in [2, 3] {
            let vscale = (1usize << lev).min(h - 1);
            assert!(vscale > 1 && h % vscale != 0, "sanity: stride is non-trivial");
            let mut a = vec![0.0f32; 4 * w * h];
            let mut b = vec![0.0f32; 4 * w * h];
            decompose_vert(&mut a, &inp, h, w, lev);
            decompose_vert_reference(&mut b, &inp, h, w, lev);
            assert_bits_eq(&a, &b, "interleave lev {lev}");
        }
    }

    #[test]
    fn decompose_kernels_degenerate_dims_no_panic() {
        // Zero dims are no-ops; 1-px-wide/tall and over-large scales stay
        // finite (clamped reflection, never OOB).
        let mut empty: Vec<f32> = Vec::new();
        decompose_vert(&mut empty, &[], 0, 0, 0);
        decompose_vert(&mut empty, &[], 0, 4, 0);
        decompose_vert(&mut empty, &[], 4, 0, 0);
        decompose_horiz(&mut [], &mut [], 0, 0, 0);
        decompose_horiz(&mut [], &mut [], 0, 4, 0);
        decompose_horiz(&mut [], &mut [], 4, 0, 0);
        for (w, h) in [(1, 1), (1, 8), (8, 1), (2, 2), (3, 3)] {
            let inp = fill4(w, h);
            let mut o = vec![0.0f32; 4 * w * h];
            decompose_vert(&mut o, &inp, h, w, 7);
            assert!(o.iter().all(|v| v.is_finite()), "vert {w}x{h}");
            let mut o2 = inp.clone();
            let mut d2 = inp.clone();
            decompose_horiz(&mut o2, &mut d2, h, w, 7);
            assert!(o2.iter().all(|v| v.is_finite()), "horiz coarse {w}x{h}");
            assert!(d2.iter().all(|v| v.is_finite()), "horiz details {w}x{h}");
        }
    }

    #[test]
    fn decompose_ffi_matches_safe_kernels() {
        // The FFI exports must pass height/width/lev through unchanged.
        let (w, h) = (11, 7);
        let inp = fill4(w, h);
        let mut via_safe = vec![0.0f32; 4 * w * h];
        let mut via_ffi = vec![0.0f32; 4 * w * h];
        decompose_vert(&mut via_safe, &inp, h, w, 2);
        unsafe {
            darkroom_dwt_decompose_vert(via_ffi.as_mut_ptr(), inp.as_ptr(), h, w, 2);
        }
        assert_bits_eq(&via_ffi, &via_safe, "vert ffi");

        let coarse = fill4(w, h);
        let orig = fill4(w, h);
        let mut o1 = coarse.clone();
        let mut o2 = coarse.clone();
        let mut d1 = orig.clone();
        let mut d2 = orig.clone();
        decompose_horiz(&mut o1, &mut d1, h, w, 2);
        unsafe {
            darkroom_dwt_decompose_horiz(o2.as_mut_ptr(), d2.as_mut_ptr(), h, w, 2);
        }
        assert_bits_eq(&o2, &o1, "horiz ffi coarse");
        assert_bits_eq(&d2, &d1, "horiz ffi details");
    }

    #[test]
    fn decompose_ffi_guards_do_not_crash() {
        // Null pointers and degenerate dimensions are no-ops, never UB.
        unsafe {
            darkroom_dwt_decompose_vert(std::ptr::null_mut(), std::ptr::null(), 4, 4, 0);
            darkroom_dwt_decompose_vert(std::ptr::null_mut(), std::ptr::null(), 0, 0, 0);
            darkroom_dwt_decompose_horiz(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                4,
                4,
                0,
            );
            darkroom_dwt_decompose_horiz(std::ptr::null_mut(), std::ptr::null_mut(), 0, 0, 0);

            // Every null-pointer permutation is rejected before any slice exists.
            let valid_in = vec![1.0f32; 64];
            let mut valid_out = vec![2.0f32; 64];
            darkroom_dwt_decompose_vert(std::ptr::null_mut(), valid_in.as_ptr(), 4, 4, 0);
            darkroom_dwt_decompose_vert(valid_out.as_mut_ptr(), std::ptr::null(), 4, 4, 0);
            let mut valid_details = vec![4.0f32; 64];
            darkroom_dwt_decompose_horiz(
                std::ptr::null_mut(),
                valid_details.as_mut_ptr(),
                4,
                4,
                0,
            );
            darkroom_dwt_decompose_horiz(valid_out.as_mut_ptr(), std::ptr::null_mut(), 4, 4, 0);

            // Partial-zero dimensions are also no-ops when the other pointers
            // are valid, and must leave those buffers untouched.
            darkroom_dwt_decompose_vert(valid_out.as_mut_ptr(), valid_in.as_ptr(), 0, 4, 0);
            darkroom_dwt_decompose_vert(valid_out.as_mut_ptr(), valid_in.as_ptr(), 4, 0, 0);
            darkroom_dwt_decompose_horiz(
                valid_out.as_mut_ptr(),
                valid_details.as_mut_ptr(),
                0,
                4,
                0,
            );
            darkroom_dwt_decompose_horiz(
                valid_out.as_mut_ptr(),
                valid_details.as_mut_ptr(),
                4,
                0,
                0,
            );
            assert!(valid_out.iter().all(|&v| v == 2.0));
            assert!(valid_details.iter().all(|&v| v == 4.0));

            // Overflowing dimension products are rejected before slices exist;
            // these pointers are never dereferenced.
            let dangling = std::ptr::NonNull::<f32>::dangling().as_ptr();
            darkroom_dwt_decompose_vert(
                dangling as *mut f32,
                dangling,
                usize::MAX,
                4,
                0,
            );
            darkroom_dwt_decompose_horiz(
                dangling as *mut f32,
                dangling as *mut f32,
                usize::MAX,
                4,
                0,
            );
        }
        // Degenerate dimensions on the safe kernels are no-ops too (covered
        // again here for the FFI-adjacent paths).
        let mut empty: Vec<f32> = Vec::new();
        decompose_vert(&mut empty, &[], 0, 0, 0);
        decompose_horiz(&mut [], &mut [], 0, 0, 0);
    }
}
