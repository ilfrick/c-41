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
//! (`dwt_*_cl`).
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
//! rows in natural order.

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
// below, reflecting at the top/bottom edges. Port of `dwt_decompose_vert`.
fn decompose_vert(out: &mut [f32], inp: &[f32], height: usize, width: usize, lev: usize) {
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
// 'details' (input − coarse) into `inp`. Port of `dwt_decompose_horiz`.
fn decompose_horiz(out: &mut [f32], inp: &mut [f32], height: usize, width: usize, lev: usize) {
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
fn denoise_vert_1ch(out: &mut [f32], inp: &[f32], height: usize, width: usize, lev: usize) {
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
#[allow(clippy::too_many_arguments)]
fn denoise_horiz_1ch(
    coarse: &[f32],
    details: &mut [f32],
    accum: &mut [f32],
    height: usize,
    width: usize,
    lev: usize,
    thold: f32,
    last: bool,
) {
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
}
