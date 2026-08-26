//! Local Laplacian filtering — a faithful port of `src/common/locallaplacian.c`
//! (Paris et al. pyramid local Laplacian filtering), the engine behind
//! darktable's Local Contrast module (`src/iop/bilat.c`, "local laplacian
//! filter" mode). Builds a Gaussian pyramid of the luminance channel, applies
//! a brightness-relative tone curve at several gamma segments, and reassembles
//! the image from interpolated Laplacian detail — boosting/reducing contrast
//! of details relative to their brightness level, halo-free.
//!
//! Bit-exactness discipline (repo standard): every arithmetic expression keeps
//! the C source's evaluation order, including the two-lane reduction trick in
//! [`gauss_reduce`] (whose vertical/horizontal groupings are *not* the naive
//! ones) and the `dt_fast_expf` bit-hack used by [`curve_scalar`] (see
//! [`crate::math::fast_expf`]). Tests pin the tricky stencils against
//! independently transcribed references.
//!
//! Ported: the self-contained buffer path (`local_laplacian_internal` with a
//! NULL boundary struct), plus `local_laplacian_memory_use` /
//! `local_laplacian_singlebuffer_size`.
//!
//! Not ported (deliberate divergence, revisit if a tiled/export path lands):
//! the preview/full boundary coupling (`local_laplacian_boundary_t`, modes
//! 1/2) exists so darktable can run the preview pixelpipe concurrently with a
//! clipped full-res ROI, padding the full-res borders from a downsampled
//! preview pyramid. c41's pipeline processes one buffer per stage and has no
//! dual-pipe ROI coupling to feed it; there is nothing to couple to yet.

/// Maximum number of levels for the Gaussian pyramid (`max_levels`).
pub const MAX_LEVELS: usize = 30;

/// Number of segments for the piecewise-linear brightness interpolation
/// (`num_gamma`).
const NUM_GAMMA: usize = 6;

/// User parameters of the filter, mirroring `bilat.c`'s mapping onto
/// `local_laplacian_internal`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalLaplacianParams {
    /// Brightness separation of shadows/mid-tones/highlights (Lab L units,
    /// scaled to [0,1] internally by the caller's `*0.01` padding step —
    /// this is the raw user sigma, typically ~1.2..3.0).
    pub sigma: f32,
    /// Lift/compress shadows (−1..1-ish; 0 = neutral).
    pub shadows: f32,
    /// Compress highlights (−1..1-ish; 0 = neutral).
    pub highlights: f32,
    /// Midtone local contrast/clarity (−1..1-ish; 0 = neutral).
    pub clarity: f32,
}

/// Downsample `size` to the given pyramid level: repeatedly `(size-1)/2+1`
/// (`dl()` in the C source).
#[inline]
fn dl(size: usize, level: usize) -> usize {
    let mut s = size;
    for _ in 0..level {
        s = (s - 1) / 2 + 1;
    }
    s
}

/// Upsample stencil (`ll_expand_gaussian`): needs a 1px boundary around
/// `(i, j)` — more precisely `1 <= i < wd-1` for even `wd` and `1 <= i < wd-2`
/// for odd `wd` (likewise `j`/`ht`). Four cases by the parity of `(i, j)`,
/// each a different binomial-filter footprint over the coarse buffer.
///
/// Precision note (bit-exactness): the C literals `4.`/`256.`, `24.0` and
/// `4.0` (no `f` suffix) are *double*, while `6.0f`/`.25f` are float. Cases
/// 1 and 2 therefore run their outer arithmetic in f64 over f32 leaf sums,
/// and case 0 scales its f32 total by the exact power-of-two double
/// `4./256.` — all replicated here, since rounding differs from pure-f32 math.
#[inline]
fn ll_expand_gaussian(coarse: &[f32], i: usize, j: usize, wd: usize, _ht: usize) -> f32 {
    let cw = (wd - 1) / 2 + 1;
    let ind = (j / 2) * cw + i / 2;
    match (i & 1) + 2 * (j & 1) {
        // both are even, 3x3 stencil — f32 interior, f64 final scale
        0 => {
            let s = 6.0f32 * (coarse[ind - cw] + coarse[ind - 1] + 6.0f32 * coarse[ind]
                + coarse[ind + 1]
                + coarse[ind + cw])
                + coarse[ind - cw - 1]
                + coarse[ind - cw + 1]
                + coarse[ind + cw - 1]
                + coarse[ind + cw + 1];
            ((4.0f64 / 256.0f64) * s as f64) as f32
        }
        // i is odd, 2x3 stencil — f32 leaf sums, f64 outer arithmetic
        1 => {
            let a = coarse[ind] + coarse[ind + 1];
            let b = coarse[ind - cw] + coarse[ind - cw + 1] + coarse[ind + cw]
                + coarse[ind + cw + 1];
            ((4.0f64 / 256.0f64)
                * (24.0f64 * a as f64 + 4.0f64 * b as f64)) as f32
        }
        // j is odd, 3x2 stencil — same precision mix as case 1
        2 => {
            let a = coarse[ind] + coarse[ind + cw];
            let b = coarse[ind - 1] + coarse[ind + 1] + coarse[ind + cw - 1]
                + coarse[ind + cw + 1];
            ((4.0f64 / 256.0f64)
                * (24.0f64 * a as f64 + 4.0f64 * b as f64)) as f32
        }
        // both are odd, 2x2 stencil — all-float in C too
        _ => 0.25f32 * (coarse[ind] + coarse[ind + 1] + coarse[ind + cw] + coarse[ind + cw + 1]),
    }
}

/// Copy one whole row onto another within a single buffer (C just memcpys
/// overlapping-free row pairs; Rust needs split_at_mut for that).
fn copy_row(buf: &mut [f32], w: usize, dst: usize, src: usize) {
    debug_assert_ne!(dst, src);
    if dst < src {
        let (head, tail) = buf.split_at_mut(src * w);
        let d = dst * w;
        head[d..d + w].copy_from_slice(&tail[..w]);
    } else {
        let (head, tail) = buf.split_at_mut(dst * w);
        let s = src * w;
        tail[..w].copy_from_slice(&head[s..s + w]);
    }
}

/// Fill in a one-pixel boundary by copying the adjacent interior sample
/// (`ll_fill_boundary1`).
fn ll_fill_boundary1(input: &mut [f32], wd: usize, ht: usize) {
    for j in 1..ht - 1 {
        input[j * wd] = input[j * wd + 1];
    }
    for j in 1..ht - 1 {
        input[j * wd + wd - 1] = input[j * wd + wd - 2];
    }
    copy_row(input, wd, 0, 1);
    copy_row(input, wd, ht - 1, ht - 2);
}

/// Fill in a two-pixel boundary by copying (`ll_fill_boundary2`); for odd
/// dimensions the outermost row/column duplicates its neighbour once, for even
/// dimensions twice.
fn ll_fill_boundary2(input: &mut [f32], wd: usize, ht: usize) {
    for j in 1..ht - 1 {
        input[j * wd] = input[j * wd + 1];
    }
    if wd & 1 != 0 {
        for j in 1..ht - 1 {
            input[j * wd + wd - 1] = input[j * wd + wd - 2];
        }
    } else {
        for j in 1..ht - 1 {
            // C chains `in[j*wd+wd-1] = in[j*wd+wd-2] = in[j*wd+wd-3]`,
            // i.e. right-to-left: both outer columns take the ORIGINAL wd-3.
            input[j * wd + wd - 2] = input[j * wd + wd - 3];
            input[j * wd + wd - 1] = input[j * wd + wd - 2];
        }
    }
    copy_row(input, wd, 0, 1);
    if ht & 1 == 0 {
        copy_row(input, wd, ht - 2, ht - 3);
    }
    copy_row(input, wd, ht - 1, ht - 2);
}

/// Fill `padding` rows top and bottom by replicating the adjacent interior
/// row (`pad_by_replication`).
fn pad_by_replication(buf: &mut [f32], w: usize, h: usize, padding: usize) {
    for j in 0..padding {
        copy_row(buf, w, j, padding);
        copy_row(buf, w, h - padding + j, h - padding - 1);
    }
}

/// Upsample: gaussian-expand the coarse level into the fine one, then repair
/// the boundary (`gauss_expand`).
fn gauss_expand(input: &[f32], fine: &mut [f32], wd: usize, ht: usize) {
    for j in 1..((ht - 1) & !1) {
        for i in 1..((wd - 1) & !1) {
            fine[j * wd + i] = ll_expand_gaussian(input, i, j, wd, ht);
        }
    }
    ll_fill_boundary2(fine, wd, ht);
}

/// Vertical 1-4-6-4-1 convolution of five rows at one horizontal position
/// (`_convolve_14641_vert`). The C source computes four horizontal positions
/// at once in SIMD lanes; here `pos` selects which of those four, producing
/// the identical scalar value with the identical addition order:
/// `r0 = v0+v4; r0 = r0+2*v2; r1 = v1+v2+v3; conv = r0 + 4*r1`.
#[inline]
fn convolve_14641_vert(input: &[f32], base: usize, wd: usize, pos: usize) -> f32 {
    let r0 = input[base + pos];
    let r1 = input[base + wd + pos];
    let r2 = input[base + 2 * wd + pos];
    let r3 = input[base + 3 * wd + pos];
    let r4 = input[base + 4 * wd + pos];
    let a = r0 + r4;
    let b = a + r2 + r2;
    let c = r1 + r2 + r3;
    b + 4.0f32 * c
}

/// Downsample by the 1-4-6-4-1 binomial separated by axis, storing only the
/// coarse resolution (`gauss_reduce`). The C source walks pairs of output
/// columns sharing four input columns; the two-per-iteration formulas below
/// reproduce its exact groupings, so results are bit-identical to the SIMD
/// lane version while staying scalar.
fn gauss_reduce(input: &[f32], coarse: &mut [f32], wd: usize, ht: usize) {
    let cw = (wd - 1) / 2 + 1;
    let ch = (ht - 1) / 2 + 1;
    const KERNEL: [f32; 4] = [1.0, 4.0, 6.0, 4.0];
    for j in 1..ch.saturating_sub(1) {
        // base = input + 2*(j-1)*wd, advanced by 4 input columns per iteration
        let mut base = 2 * (j - 1) * wd;
        let out = j * cw + 1;
        // prime the vertical axis
        let mut left = [
            convolve_14641_vert(input, base, wd, 0),
            convolve_14641_vert(input, base, wd, 1),
            convolve_14641_vert(input, base, wd, 2),
            convolve_14641_vert(input, base, wd, 3),
        ];
        let mut col = 0usize;
        // C: `for(size_t col=0; col<cw-3; col += 2)` (cw >= 3 by pyramid
        // construction, so no underflow)
        while col + 3 < cw {
            // convolve the next four-pixel-wide vertical slice
            base += 4;
            let right = [
                convolve_14641_vert(input, base, wd, 0),
                convolve_14641_vert(input, base, wd, 1),
                convolve_14641_vert(input, base, wd, 2),
                convolve_14641_vert(input, base, wd, 3),
            ];
            // horizontal pass, two output values from 1 4 6 4 1:
            // the first uses pixels 0-4, the second uses 2-6
            let mut conv = [0.0f32; 4];
            for c in 0..4 {
                conv[c] = left[c] * KERNEL[c];
            }
            coarse[out + col] =
                (conv[0] + conv[1] + conv[2] + conv[3] + right[0]) / 256.0f32;
            coarse[out + col + 1] = (left[2]
                + 4.0f32 * (left[3] + right[1])
                + 6.0f32 * right[0]
                + right[2])
                / 256.0f32;
            // shift to next pair of output columns (four input columns)
            left = right;
            col += 2;
        }
        // handle the left-over pixel if the output size is odd
        if cw % 2 == 1 {
            base += 4;
            // convolve the right-most column
            let right = input[base]
                + 4.0f32 * (input[base + wd] + input[base + 3 * wd])
                + 6.0f32 * input[base + 2 * wd]
                + input[base + 4 * wd];
            let mut conv = [0.0f32; 4];
            for c in 0..4 {
                conv[c] = left[c] * KERNEL[c];
            }
            coarse[out + cw - 3] =
                (conv[0] + conv[1] + conv[2] + conv[3] + right) / 256.0f32;
        }
    }
    ll_fill_boundary1(coarse, cw, ch);
}

/// Allocate the level-0 luminance buffer padded by `max_supp` on all sides,
/// Lab-L rescaled by ×0.01 into [0,1] (`ll_pad_input`, replication arm).
fn ll_pad_input(input: &[f32], wd: usize, ht: usize, max_supp: usize) -> Vec<f32> {
    let stride = 4usize;
    let wd2 = 2 * max_supp + wd;
    let ht2 = 2 * max_supp + ht;
    let mut out = vec![0.0f32; wd2 * ht2];
    for j in 0..ht {
        for i in 0..max_supp {
            out[(j + max_supp) * wd2 + i] = input[stride * wd * j] * 0.01f32;
        }
        for i in 0..wd {
            out[(j + max_supp) * wd2 + i + max_supp] = input[stride * (wd * j + i)] * 0.01f32;
        }
        for i in wd + max_supp..wd2 {
            out[(j + max_supp) * wd2 + i] = input[stride * (j * wd + wd - 1)] * 0.01f32;
        }
    }
    pad_by_replication(&mut out, wd2, ht2, max_supp);
    out
}

/// Laplacian detail at a fine pixel: fine gaussian minus its upsampled coarse
/// counterpart (`ll_laplacian`).
#[inline]
fn ll_laplacian(
    coarse: &[f32],
    fine: &[f32],
    i: usize,
    j: usize,
    wd: usize,
    ht: usize,
) -> f32 {
    let ci = i.clamp(1, ((wd - 1) & !1) - 1);
    let cj = j.clamp(1, ((ht - 1) & !1) - 1);
    let c = ll_expand_gaussian(coarse, ci, cj, wd, ht);
    fine[j * wd + i] - c
}

/// The remapping curve for one gamma segment (`curve_scalar`): quadratic-bezier
/// contrast boost around the segment centre `g`, plus a Gaussian-weighted
/// midtone clarity term. Evaluation order matches the C source exactly.
#[allow(clippy::too_many_arguments)]
fn curve_scalar(x: f32, g: f32, sigma: f32, shadows: f32, highlights: f32, clarity: f32) -> f32 {
    let c = x - g;
    let mut val;
    // blend in via quadratic bezier
    if c > 2.0 * sigma {
        val = g + sigma + shadows * (c - sigma);
    } else if c < -2.0 * sigma {
        val = g - sigma + highlights * (c + sigma);
    } else if c > 0.0 {
        // shadow contrast
        let t = (c / (2.0 * sigma)).clamp(0.0, 1.0);
        let t2 = t * t;
        let mt = 1.0 - t;
        val = g + sigma * 2.0 * mt * t + t2 * (sigma + sigma * shadows);
    } else {
        // highlight contrast
        let t = (-c / (2.0 * sigma)).clamp(0.0, 1.0);
        let t2 = t * t;
        let mt = 1.0 - t;
        val = g - sigma * 2.0 * mt * t + t2 * (-sigma - sigma * highlights);
    }
    // midtone local contrast
    val += clarity * c * crate::math::fast_expf(-c * c / (2.0f32 * sigma * sigma / 3.0f32));
    val
}

/// Apply the remapping curve to the padded level-0 buffer, repairing the
/// boundary afterwards (`apply_curve`).
#[allow(clippy::too_many_arguments)]
fn apply_curve(
    out: &mut [f32],
    input: &[f32],
    w: usize,
    h: usize,
    padding: usize,
    g: f32,
    sigma: f32,
    shadows: f32,
    highlights: f32,
    clarity: f32,
) {
    for j in padding..h - padding {
        for i in padding..w - padding {
            out[j * w + i] =
                curve_scalar(input[j * w + i], g, sigma, shadows, highlights, clarity);
        }
        // row-edge replication (the C resets its cursor and fills both sides)
        let row = j * w;
        for i in 0..padding {
            out[row + i] = out[row + padding];
        }
        for i in w - padding..w {
            out[row + i] = out[row + w - padding - 1];
        }
    }
    pad_by_replication(out, w, h, padding);
}

/// Pyramid depth for a buffer whose smaller side is `min_dim`: as many halving
/// steps as fit, capped at [`MAX_LEVELS`] (`31-__builtin_clz(MIN(wd,ht))`).
/// The C counts leading zeros of a 32-bit int, so cast to u32 first —
/// usize::leading_zeros would be a 64-bit count.
#[inline]
fn num_levels_for(min_dim: usize) -> usize {
    (MAX_LEVELS).min((31u32.saturating_sub((min_dim.min(u32::MAX as usize) as u32).leading_zeros())) as usize)
}

/// Local Laplacian filter over a stride-4 Lab buffer (`local_laplacian` /
/// `local_laplacian_internal` with no boundary struct).
///
/// Reads channel 0 (Lab L, in [0,100]) of `input` and writes channels 0..3 of
/// `out` — L remapped through the filter, a/b copied verbatim (channel 3 is
/// left untouched, exactly as in C). Aliasing is impossible in safe Rust and
/// the port reads `input` freely while writing `out`, so DISTINCT buffers are
/// required (darktable's bilat passes distinct in/out too).
///
/// Panics-free for tiny images: `wd <= 1 || ht <= 1` returns untouched (as in
/// C); smaller-side 2..3 would drive the C source past its pyramid arrays
/// (undefined behaviour), so the port instead copies input through unchanged —
/// call sites should gate on the smaller side being at least 4.
///
/// Memory: unlike C's graceful copy-through on allocation failure, Rust aborts
/// on OOM — peak need is [`local_laplacian_memory_use`] (~11 GB padded level-0
/// alone for a 20k×20k input), so callers wiring this into a pipeline should
/// pre-check it against a budget, as darktable's tiling code does.
pub fn local_laplacian(
    input: &[f32],
    out: &mut [f32],
    wd: usize,
    ht: usize,
    params: LocalLaplacianParams,
) {
    if wd <= 1 || ht <= 1 {
        return;
    }
    let LocalLaplacianParams { sigma, shadows, highlights, clarity } = params;

    // don't divide by 2 more often than we can
    let num_levels = num_levels_for(wd.min(ht));
    if num_levels < 2 {
        // C would index padded[-1] here (UB); degrade to pass-through like
        // its allocation-failure path: copy all 4*wd*ht elements through.
        out[..4 * wd * ht].copy_from_slice(&input[..4 * wd * ht]);
        return;
    }
    let last_level = num_levels - 1;
    let max_supp = 1usize << last_level;

    let w = 2 * max_supp + wd;
    let h = 2 * max_supp + ht;
    let mut padded: Vec<Vec<f32>> = Vec::with_capacity(num_levels);
    padded.push(ll_pad_input(input, wd, ht, max_supp));
    for l in 1..=last_level {
        padded.push(vec![0.0f32; dl(w, l) * dl(h, l)]);
    }

    let mut output: Vec<Vec<f32>> = (0..=last_level)
        .map(|l| vec![0.0f32; dl(w, l) * dl(h, l)])
        .collect();

    // create gauss pyramid of padded input, write coarse directly to output
    for l in 1..last_level {
        let (src_w, src_h) = (dl(w, l - 1), dl(h, l - 1));
        let (lower, upper) = padded.split_at_mut(l);
        gauss_reduce(&lower[l - 1], &mut upper[0], src_w, src_h);
    }
    {
        // different vectors, so the borrows don't collide
        let sw = dl(w, last_level - 1);
        let sh = dl(h, last_level - 1);
        gauss_reduce(&padded[last_level - 1], &mut output[last_level], sw, sh);
    }

    // evenly sample brightness [0,1]
    let gamma: [f32; NUM_GAMMA] =
        std::array::from_fn(|k| (k as f32 + 0.5) / NUM_GAMMA as f32);

    // intermediate laplacian pyramids, one per gamma segment
    let mut buf: Vec<Vec<Vec<f32>>> = Vec::with_capacity(NUM_GAMMA);
    for _k in 0..NUM_GAMMA {
        buf.push((0..=last_level).map(|l| vec![0.0f32; dl(w, l) * dl(h, l)]).collect());
    }

    for k in 0..NUM_GAMMA {
        // process images: remap level 0, then build that segment's pyramid
        apply_curve(
            &mut buf[k][0],
            &padded[0],
            w,
            h,
            max_supp,
            gamma[k],
            sigma,
            shadows,
            highlights,
            clarity,
        );
        for l in 1..=last_level {
            let (src_w, src_h) = (dl(w, l - 1), dl(h, l - 1));
            let (lower, upper) = buf[k].split_at_mut(l);
            gauss_reduce(&lower[l - 1], &mut upper[0], src_w, src_h);
        }
    }

    // assemble output pyramid coarse to fine
    for l in (0..last_level).rev() {
        let pw = dl(w, l);
        let ph = dl(h, l);
        {
            let (lower, upper) = output.split_at_mut(l + 1);
            gauss_expand(&upper[0], &mut lower[l], pw, ph);
        }
        // go through all coefficients in the upsampled gauss buffer
        for j in 0..ph {
            for i in 0..pw {
                let v = padded[l][j * pw + i];
                let mut hi = 1usize;
                while hi < NUM_GAMMA - 1 && gamma[hi] <= v {
                    hi += 1;
                }
                let lo = hi - 1;
                let a = ((v - gamma[lo]) / (gamma[hi] - gamma[lo])).clamp(0.0, 1.0);
                let l0 = ll_laplacian(&buf[lo][l + 1], &buf[lo][l], i, j, pw, ph);
                let l1 = ll_laplacian(&buf[hi][l + 1], &buf[hi][l], i, j, pw, ph);
                output[l][j * pw + i] += l0 * (1.0f32 - a) + l1 * a;
            }
        }
    }

    // write back: filtered L (scaled back up), original colour channels kept
    for j in 0..ht {
        for i in 0..wd {
            let o = 4 * (j * wd + i);
            out[o] = 100.0f32 * output[0][(j + max_supp) * w + max_supp + i];
            out[o + 1] = input[o + 1];
            out[o + 2] = input[o + 2];
        }
    }
}

/// Peak extra memory the filter needs for a `width`×`height` input, in bytes
/// (`local_laplacian_memory_use`) — the padded input plus `num_gamma` scratch
/// pyramids at every level.
pub fn local_laplacian_memory_use(width: usize, height: usize) -> usize {
    let num_levels = num_levels_for(width.min(height));
    let max_supp = 1usize << (num_levels.saturating_sub(1));
    let paddwd = width + 2 * max_supp;
    let paddht = height + 2 * max_supp;
    let mut memory_use = 0usize;
    for l in 0..num_levels {
        memory_use += 4 * (2 + NUM_GAMMA) * dl(paddwd, l) * dl(paddht, l);
    }
    memory_use
}

/// Size in bytes of the largest single buffer the filter allocates for a
/// `width`×`height` input (`local_laplacian_singlebuffer_size`).
pub fn local_laplacian_singlebuffer_size(width: usize, height: usize) -> usize {
    let num_levels = num_levels_for(width.min(height));
    let max_supp = 1usize << (num_levels.saturating_sub(1));
    let paddwd = width + 2 * max_supp;
    let paddht = height + 2 * max_supp;
    4 * dl(paddwd, 0) * dl(paddht, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic pseudo-random fill (LCG), so tests never depend on rand
    fn noise(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / 16777216.0
            })
            .collect()
    }

    // ── dl() ────────────────────────────────────────────────────────────────

    #[test]
    fn dl_matches_the_c_series() {
        // hand-evaluated (size-1)/2+1 chains (integer division truncates)
        let want38 = [38usize, 19, 10, 5, 3, 2, 1];
        for (l, &w) in want38.iter().enumerate() {
            assert_eq!(dl(38, l), w, "dl(38,{l})");
        }
        let want128 = [128usize, 64, 32, 16, 8, 4, 2, 1];
        for (l, &w) in want128.iter().enumerate() {
            assert_eq!(dl(128, l), w, "dl(128,{l})");
        }
        let want45 = [45usize, 23, 12, 6, 3, 2, 1];
        for (l, &w) in want45.iter().enumerate() {
            assert_eq!(dl(45, l), w, "dl(45,{l})");
        }
        assert_eq!(dl(1, 0), 1);
        assert_eq!(dl(2, 5), 1);
    }

    // ── expand stencil ──────────────────────────────────────────────────────

    #[test]
    fn expand_preserves_a_constant_field_in_all_four_cases() {
        // weights of every stencil sum to 1 (4/256*(6*10+4), 4/256*(48+16),
        // 4/256*(48+16), .25*4), so a constant field maps to itself; probe
        // one fine position per parity case. All-dyadic math => bit equality.
        let wd = 9usize;
        let ht = 9usize;
        let c = vec![0.25f32; wd * ht];
        for &(i, j) in &[(2usize, 2usize), (3, 2), (2, 3), (3, 3)] {
            let v = ll_expand_gaussian(&c, i, j, wd, ht);
            assert_eq!(v, 0.25, "case ({i},{j})");
        }
    }

    #[test]
    fn expand_impulse_case0_central_tap_only() {
        // impulse 256 at the coarse centre; the fine pixel sitting exactly on
        // it (even i, even j) sees only the central 6*6 tap through the 3x3
        // stencil: 4/256 * 6*6*256 = 144 exactly (dyadic).
        let cw = 5usize;
        let mut c = vec![0.0f32; cw * cw];
        c[2 * cw + 2] = 256.0;
        let v = ll_expand_gaussian(&c, 4, 4, 9, 9);
        assert_eq!(v, 144.0);
    }

    /// Asymmetric 5x5 grid around ind=(2,2): every tap a distinguishable
    /// value, so a transposed index in any stencil case cannot cancel.
    /// Layout: up=1 left=2 ctr=3 right=4 down=5 / ul=6 ur=7 ll=8 lr=9 dr=19.
    fn asymmetric_coarse() -> Vec<f32> {
        let cw = 5usize;
        let mut c = vec![0.0f32; cw * cw];
        c[2 * cw + 2] = 3.0; // centre
        c[1 * cw + 2] = 1.0; // up
        c[2 * cw + 1] = 2.0; // left
        c[2 * cw + 3] = 4.0; // right
        c[3 * cw + 2] = 5.0; // down
        c[1 * cw + 1] = 6.0; // upper-left
        c[1 * cw + 3] = 7.0; // upper-right
        c[3 * cw + 1] = 8.0; // lower-left
        c[3 * cw + 3] = 9.0; // lower-right (= ctr+cw+1, a real tap)
        c[3 * cw + 4] = 19.0; // decoy just past it — must never be read
        c
    }

    #[test]
    fn expand_case0_taps_are_hand_computable() {
        // s = 6*(u+l+6*ctr+r+d)+ul+ur+ll+lr = 6*(1+2+18+4+5)+6+7+8+9 = 210,
        // scaled by the exact dyadic 4/256: 210*4/256 = 105/32
        let c = asymmetric_coarse();
        assert_eq!(ll_expand_gaussian(&c, 4, 4, 9, 9), 105.0 / 32.0);
    }

    #[test]
    fn expand_case1_taps_are_hand_computable() {
        // i odd: a=(ctr+r)=7, b=(u+ur+d+lr)=1+7+5+9=22 → (24*7+4*22)/64 = 4
        let c = asymmetric_coarse();
        assert_eq!(ll_expand_gaussian(&c, 5, 4, 9, 9), 4.0);
    }

    #[test]
    fn expand_case2_taps_are_hand_computable() {
        // j odd: a=(ctr+d)=8, b=(l+r+ll+lr)=2+4+8+9=23 → (24*8+4*23)/64
        let c = asymmetric_coarse();
        assert_eq!(ll_expand_gaussian(&c, 4, 5, 9, 9), 284.0 / 64.0);
    }

    #[test]
    fn expand_case3_taps_are_hand_computable() {
        // both odd: .25*(ctr+r+d+lr) = .25*(3+4+5+9)
        let c = asymmetric_coarse();
        assert_eq!(ll_expand_gaussian(&c, 5, 5, 9, 9), 21.0 * 0.25);
    }

    // ── gauss_reduce vs independent reference ───────────────────────────────

    /// Vertical 14641 through `_convolve_14641_vert`'s exact op order.
    fn ref_vert(input: &[f32], wd: usize, row0: usize, col: usize) -> f32 {
        let v = |r: usize| input[(row0 + r) * wd + col];
        let a = (v(0) + v(4)) + v(2) + v(2);
        let b = (v(1) + v(2)) + v(3);
        a + 4.0f32 * b
    }

    /// Vertical through the scalar left-over formula, whose grouping differs
    /// from [`ref_vert`] in the C source too (and therefore rounds differently).
    fn ref_vert_scalar_tail(input: &[f32], wd: usize, row0: usize, col: usize) -> f32 {
        let g = |r: usize| input[(row0 + r) * wd + col];
        // grouping mirrors the C scalar tail formula exactly
        ((g(0) + 4.0f32 * (g(1) + g(3))) + 6.0f32 * g(2)) + g(4)
    }

    /// Independent gauss_reduce reference: every output derives its five input
    /// columns from its own coordinates (no shared left/right lane state), so
    /// an indexing slip in the port cannot cancel against the reference.
    fn reduce_reference(input: &[f32], wd: usize, ht: usize) -> Vec<f32> {
        let cw = (wd - 1) / 2 + 1;
        let ch = (ht - 1) / 2 + 1;
        let mut out = vec![0.0f32; cw * ch];
        for y in 1..ch - 1 {
            let row0 = 2 * (y - 1);
            for x in 1..cw - 1 {
                let c0 = 2 * x - 2;
                // the odd-width tail column uses the scalar vertical form
                let leftover = cw % 2 == 1 && x == cw - 2;
                let taps = [
                    ref_vert(input, wd, row0, c0),
                    ref_vert(input, wd, row0, c0 + 1),
                    ref_vert(input, wd, row0, c0 + 2),
                    ref_vert(input, wd, row0, c0 + 3),
                    if leftover {
                        ref_vert_scalar_tail(input, wd, row0, c0 + 4)
                    } else {
                        ref_vert(input, wd, row0, c0 + 4)
                    },
                ];
                if x % 2 == 1 {
                    // first output of a pair: per-lane products, then summed
                    let t = [taps[0] * 1.0, taps[1] * 4.0, taps[2] * 6.0, taps[3] * 4.0];
                    out[y * cw + x] = (t[0] + t[1] + t[2] + t[3] + taps[4]) / 256.0f32;
                } else {
                    // second output of a pair: the fused grouping
                    out[y * cw + x] =
                        (taps[0] + 4.0f32 * (taps[1] + taps[3]) + 6.0f32 * taps[2] + taps[4])
                            / 256.0f32;
                }
            }
        }
        // boundary replication, identical to ll_fill_boundary1
        for j in 1..ch - 1 {
            out[j * cw] = out[j * cw + 1];
        }
        for j in 1..ch - 1 {
            out[j * cw + cw - 1] = out[j * cw + cw - 2];
        }
        copy_row(&mut out, cw, 0, 1);
        copy_row(&mut out, cw, ch - 1, ch - 2);
        out
    }

    fn assert_reduce_matches_reference(wd: usize, ht: usize) {
        let input = noise(wd * ht, 0xC41);
        let cw = (wd - 1) / 2 + 1;
        let ch = (ht - 1) / 2 + 1;
        let mut coarse = vec![0.0f32; cw * ch];
        gauss_reduce(&input, &mut coarse, wd, ht);
        let want = reduce_reference(&input, wd, ht);
        assert_eq!(coarse.len(), want.len(), "{wd}x{ht}");
        for k in 0..coarse.len() {
            assert_eq!(
                coarse[k].to_bits(),
                want[k].to_bits(),
                "{wd}x{ht} idx {k} ({k}%{cw},{k}/{cw}): {} vs {}",
                coarse[k],
                want[k]
            );
        }
    }

    #[test]
    fn gauss_reduce_bitmatches_reference_even_dims() {
        assert_reduce_matches_reference(16, 12);
    }

    #[test]
    fn gauss_reduce_bitmatches_reference_odd_dims() {
        assert_reduce_matches_reference(17, 13);
    }

    #[test]
    fn gauss_reduce_bitmatches_reference_small_odd_width_leftover() {
        // cw=5: one pair iteration plus the scalar left-over path
        assert_reduce_matches_reference(9, 8);
        // minimal interior: cw=ch=3, single left-over sample, no pair pass
        assert_reduce_matches_reference(6, 6);
    }

    // ── curve_scalar ────────────────────────────────────────────────────────

    #[test]
    fn curve_scalar_neutral_is_identity_at_segment_centre() {
        let v = curve_scalar(0.5, 0.5, 2.0, 0.0, 0.0, 0.0);
        assert_eq!(v, 0.5);
    }

    #[test]
    fn curve_scalar_far_field_follows_the_linear_wings() {
        let g = 0.4f32;
        let sigma = 1.5f32;
        // far above: g+sigma+shadows*(c-sigma)
        let hi = curve_scalar(g + 6.0, g, sigma, 0.3, 0.0, 0.0);
        assert_eq!(hi, g + sigma + 0.3 * (6.0 - sigma));
        // far below: g-sigma+highlights*(c+sigma)
        let lo = curve_scalar(g - 6.0, g, sigma, 0.0, -0.2, 0.0);
        assert_eq!(lo, g - sigma - 0.2 * (-6.0 + sigma));
    }

    #[test]
    fn curve_scalar_is_continuous_at_the_wing_joints() {
        let g = 0.45f32;
        let sigma = 0.8f32;
        for &sh in &[-0.4f32, 0.0, 0.6] {
            let bezier = curve_scalar(g + 2.0 * sigma, g, sigma, sh, 0.0, 0.0);
            let wing = g + sigma + sh * (2.0 * sigma - sigma);
            assert!(
                (bezier - wing).abs() < 1e-5,
                "sh={sh}: bezier={bezier} wing={wing}"
            );
        }
    }

    #[test]
    fn clarity_term_peaks_at_the_segment_centre_and_decays() {
        let g = 0.5f32;
        let sigma = 1.0f32;
        let neutral = |x: f32| curve_scalar(x, g, sigma, 0.0, 0.0, 0.0);
        let clarity = |x: f32| curve_scalar(x, g, sigma, 0.0, 0.0, 1.0);
        let near = clarity(g + 0.5) - neutral(g + 0.5);
        let far = clarity(g + 3.5) - neutral(g + 3.5);
        assert!(near > 0.0, "midtone boost should lift above centre: {near}");
        assert!(far.abs() < near.abs(), "clarity must decay: near={near} far={far}");
    }

    // ── end-to-end ──────────────────────────────────────────────────────────

    fn lab_buffer(wd: usize, ht: usize, l: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut buf = vec![0.0f32; 4 * wd * ht];
        for j in 0..ht {
            for i in 0..wd {
                let o = 4 * (j * wd + i);
                buf[o] = l(i, j);
                buf[o + 1] = 12.5 + i as f32 * 0.1;
                buf[o + 2] = -8.0 + j as f32 * 0.05;
            }
        }
        buf
    }

    #[test]
    fn constant_input_maps_to_itself_even_with_active_params() {
        // A constant field carries (almost) zero Laplacian detail, so the
        // filter must reproduce it to float noise — regardless of the curve,
        // whose segment pyramids stay constant too. Not bit-exact BY DESIGN:
        // the C's odd-width tail computes its vertical with a different
        // addition grouping than the SIMD-lane path, so coarse levels differ
        // by ~1 ulp even on constants; the port replicates that faithfully.
        let (wd, ht) = (24usize, 18usize);
        let input = lab_buffer(wd, ht, |_, _| 50.0);
        for params in [
            LocalLaplacianParams { sigma: 2.0, shadows: 0.0, highlights: 0.0, clarity: 0.0 },
            LocalLaplacianParams { sigma: 1.2, shadows: 0.35, highlights: -0.25, clarity: 0.6 },
        ] {
            let mut out = input.clone();
            out[0] = 0.0; // poison: prove the filter actually wrote L
            local_laplacian(&input, &mut out, wd, ht, params);
            for k in 0..wd * ht {
                assert!(
                    (out[4 * k] - 50.0).abs() < 1e-3,
                    "px {k}: {} vs 50.0",
                    out[4 * k]
                );
            }
        }
    }

    #[test]
    fn colour_channels_pass_through_and_channel3_untouched() {
        let (wd, ht) = (20usize, 14usize);
        let input = lab_buffer(wd, ht, |i, j| 40.0 + (i * 7 + j * 3) as f32 % 20.0);
        let mut out = vec![0.0f32; input.len()];
        for k in 0..wd * ht {
            out[4 * k + 3] = 777.0; // sentinel in the untouched channel
        }
        local_laplacian(
            &input,
            &mut out,
            wd,
            ht,
            LocalLaplacianParams { sigma: 2.0, shadows: 0.2, highlights: 0.1, clarity: 0.3 },
        );
        for k in 0..wd * ht {
            assert_eq!(out[4 * k + 1].to_bits(), input[4 * k + 1].to_bits());
            assert_eq!(out[4 * k + 2].to_bits(), input[4 * k + 2].to_bits());
            assert_eq!(out[4 * k + 3], 777.0);
            assert!(out[4 * k].is_finite());
        }
    }

    #[test]
    fn gradient_input_stays_finite_and_ordered() {
        // a monotone ramp should come out finite everywhere; with mild
        // positive contrast the output range stays sane (no explosion)
        let (wd, ht) = (32usize, 24usize);
        let input = lab_buffer(wd, ht, |i, _| (i as f32 / wd as f32) * 100.0);
        let mut out = input.clone();
        local_laplacian(
            &input,
            &mut out,
            wd,
            ht,
            LocalLaplacianParams { sigma: 2.5, shadows: 0.3, highlights: -0.3, clarity: 0.5 },
        );
        for k in 0..wd * ht {
            let v = out[4 * k];
            assert!(v.is_finite(), "px {k} = {v}");
            assert!((-100.0..200.0).contains(&v), "px {k} = {v} out of range");
        }
    }

    #[test]
    fn tiny_images_do_not_panic() {
        for wd in 1..=6usize {
            for ht in 1..=6usize {
                let input = lab_buffer(wd, ht, |_, _| 42.0);
                let mut out = input.clone();
                let before = out.clone();
                local_laplacian(
                    &input,
                    &mut out,
                    wd,
                    ht,
                    LocalLaplacianParams { sigma: 2.0, shadows: 0.1, highlights: 0.1, clarity: 0.1 },
                );
                if wd <= 1 || ht <= 1 {
                    // documented C behaviour: untouched
                    assert_eq!(out, before, "{wd}x{ht} should be untouched");
                }
                // anything else must have terminated without panic; smaller-
                // side 2..3 degrades to the documented pass-through
            }
        }
    }

    // ── memory estimates ────────────────────────────────────────────────────

    #[test]
    fn memory_estimates_are_consistent() {
        for &(w, h) in &[(64usize, 48usize), (1024, 768), (3000, 2000)] {
            let mem = local_laplacian_memory_use(w, h);
            let sb = local_laplacian_singlebuffer_size(w, h);
            assert!(mem > 0 && sb > 0);
            // the total covers num_gamma+2 buffers per level, so it strictly
            // exceeds the largest single allocation
            assert!(mem > sb, "{w}x{h}: mem={mem} sb={sb}");
            // hand-compute level 0: padded dims, 4 bytes/float
            let num_levels = (30usize).min(
                (31u32 - (w.min(h) as u32).leading_zeros()) as usize
            );
            let max_supp = 1usize << (num_levels - 1);
            assert_eq!(sb, 4 * (w + 2 * max_supp) * (h + 2 * max_supp));
        }
    }
}
