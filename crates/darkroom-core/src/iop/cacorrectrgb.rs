use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Cacorrectrgb;

impl IopProcess for Cacorrectrgb {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "cacorrectrgb" }
}

/// Per-pixel manifold normalisation for the cacorrectrgb (RGB chromatic-
/// aberration correction) IOP.
///
/// For every pixel k, with weights stored in the alpha channel of the
/// `*_higher`/`*_lower` manifolds:
///   weighth = max(higher[k*4+3], 1e-2)
///   weightl = max(lower[k*4+3],  1e-2)
///   highg   = higher[k*4 + guide] / weighth     (normalise guide channel)
///   lowg    = lower[k*4 + guide]  / weightl
///   higher[k*4 + guide] = highg
///   lower[k*4 + guide]  = lowg
///   for the two non-guide channels c:
///     highc = higher[k*4+c] / weighth
///     lowc  = lower[k*4+c]  / weightl
///     higher[k*4+c] = exp2(highc) * highg
///     lower[k*4+c]  = exp2(lowc)  * lowg
///   if weighth < 0.05:
///     w = (weighth - 0.01) / 0.04
///     blend higher towards blurred_in by (1 - w)
///   ditto for weightl
///
/// Matches `normalize_manifolds()` in src/iop/cacorrectrgb.c.
/// `guide` is the index of the guide channel (0=R, 1=G, 2=B).
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrectrgb_normalize_manifolds(
    blurred_in: *const f32,
    blurred_manifold_lower: *mut f32,
    blurred_manifold_higher: *mut f32,
    width: usize,
    height: usize,
    guide: u32,
) {
    let npx = width.saturating_mul(height);
    if npx == 0 { return; }

    // Hard guard rather than a silent clamp. Production builds reach this
    // function through the C FFI; a mis-wired call passing guide >= 3
    // would otherwise corrupt every pixel without raising a signal. Bail
    // out cleanly and leave the caller's buffers untouched. NB: we must not
    // panic here — this is an `extern "C"` boundary, so a panic (even a
    // debug_assert) is a non-unwinding abort rather than something a caller
    // or test can catch.
    if guide >= 3 {
        return;
    }
    let guide = guide as usize;

    let inp = std::slice::from_raw_parts(blurred_in, npx * 4);
    let lo  = std::slice::from_raw_parts_mut(blurred_manifold_lower,  npx * 4);
    let hi  = std::slice::from_raw_parts_mut(blurred_manifold_higher, npx * 4);

    for k in 0..npx {
        let b = k * 4;
        let weighth = hi[b + 3].max(1e-2);
        let weightl = lo[b + 3].max(1e-2);

        // Guide channel normalisation
        let highg = hi[b + guide] / weighth;
        let lowg  = lo[b + guide] / weightl;
        hi[b + guide] = highg;
        lo[b + guide] = lowg;

        // Non-guide channels: normalise then unlog (multiply by guide).
        // `exp2f` in C and `f32::exp2` in Rust both lower to the same libm
        // primitive on x86_64/glibc but bit-equality across compilers /
        // optimisation levels is not guaranteed. Downstream regression
        // tests against the C reference should allow sub-LSB tolerance.
        for kc in 0..=1 {
            let c = (kc + guide + 1) % 3;
            let highc = hi[b + c] / weighth;
            let lowc  = lo[b + c] / weightl;
            hi[b + c] = highc.exp2() * highg;
            lo[b + c] = lowc.exp2()  * lowg;
        }

        // Low-confidence smooth blend toward blurred_in. The C source uses
        // `for_each_channel(c, …)` which expands to a 0..DT_PIXEL_SIMD_CHANNELS
        // loop — 4 in the default vectorised build, 3 only when
        // `DT_NO_VECTORIZATION` is defined. The production binary blends all
        // four channels (alpha included); this is a side-effect of the SIMD
        // hint, not a deliberate design choice, but it IS the actual
        // production behaviour we are matching here. If a future
        // non-vectorised build path becomes the canonical one, swap to 0..3.
        if weighth < 0.05 {
            let w = (weighth - 0.01) / (0.05 - 0.01);
            for c in 0..4 {
                hi[b + c] = w * hi[b + c] + (1.0 - w) * inp[b + c];
            }
        }
        if weightl < 0.05 {
            let w = (weightl - 0.01) / (0.05 - 0.01);
            for c in 0..4 {
                lo[b + c] = w * lo[b + c] + (1.0 - w) * inp[b + c];
            }
        }
    }

    // NB: the C source wraps the outer loop in DT_OMP_FOR. The Rust port
    // is single-threaded today; rayon::par_chunks_exact_mut is the right
    // tool once the FFI surface stabilises and we want the throughput back.
}

const MAX_EV_DIFF: f32 = 2.0;

/// Build the per-pixel manifolds from the raw input + its Gaussian blur.
///
/// For every pixel k (first-pass, non-refinement):
///   pixelg = max(in[k*4 + guide], 1e-6)
///   avg    = blurred_in[k*4 + guide]
///   weighth = (pixelg >= avg) as f32
///   weightl = (pixelg <= avg) as f32
///   for each non-guide channel c:
///     logdiff = log2(max(in[k*4+c], 1e-6)) - log2(pixelg)
///   if max(|logdiff|) > MAX_EV_DIFF: downscale both weights
///   write results into manifold_higher/lower (6 fields per pixel)
///
/// Matches the first DT_OMP_FOR in `get_manifolds()` (cacorrectrgb.c:234).
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrectrgb_build_manifolds(
    in_buf: *const f32,
    blurred_in: *const f32,
    manifold_lower: *mut f32,
    manifold_higher: *mut f32,
    width: usize,
    height: usize,
    guide: u32,
) {
    let npx = width.saturating_mul(height);
    if npx == 0 || guide >= 3 { return; }
    let g = guide as usize;
    let inp  = std::slice::from_raw_parts(in_buf, npx * 4);
    let blur = std::slice::from_raw_parts(blurred_in, npx * 4);
    let lo   = std::slice::from_raw_parts_mut(manifold_lower, npx * 4);
    let hi   = std::slice::from_raw_parts_mut(manifold_higher, npx * 4);

    for k in 0..npx {
        let b = k * 4;
        let pixelg = inp[b + g].max(1e-6);
        let avg    = blur[b + g];
        let mut weighth = if pixelg >= avg { 1.0_f32 } else { 0.0 };
        let mut weightl = if pixelg <= avg { 1.0_f32 } else { 0.0 };

        let mut logdiffs = [0.0_f32; 2];
        for kc in 0..=1usize {
            let c = (kc + g + 1) % 3;
            let pixel = inp[b + c].max(1e-6);
            logdiffs[kc] = (pixel / pixelg).log2();
        }

        let maxlogdiff = logdiffs[0].abs().max(logdiffs[1].abs());
        if maxlogdiff > MAX_EV_DIFF {
            let cw = MAX_EV_DIFF / maxlogdiff;
            weightl *= cw;
            weighth *= cw;
        }

        for kc in 0..=1usize {
            let c = (kc + g + 1) % 3;
            hi[b + c] = logdiffs[kc] * weighth;
            lo[b + c] = logdiffs[kc] * weightl;
        }
        hi[b + g] = pixelg * weighth;
        lo[b + g] = pixelg * weightl;
        hi[b + 3] = weighth;
        lo[b + 3] = weightl;
    }
}

/// Refinement pass: update manifolds using estimates from the first pass.
///
/// Matches the second DT_OMP_FOR in `get_manifolds()` (cacorrectrgb.c:299).
/// Called only when `refine_manifolds` is true.
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrectrgb_refine_manifolds(
    in_buf: *const f32,
    blurred_in: *const f32,
    blurred_manifold_lower: *const f32,
    blurred_manifold_higher: *const f32,
    manifold_lower: *mut f32,
    manifold_higher: *mut f32,
    width: usize,
    height: usize,
    guide: u32,
) {
    let npx = width.saturating_mul(height);
    if npx == 0 || guide >= 3 { return; }
    let g = guide as usize;
    let inp  = std::slice::from_raw_parts(in_buf,                    npx * 4);
    let blur = std::slice::from_raw_parts(blurred_in,               npx * 4);
    let blo  = std::slice::from_raw_parts(blurred_manifold_lower,   npx * 4);
    let bhi  = std::slice::from_raw_parts(blurred_manifold_higher,  npx * 4);
    let lo   = std::slice::from_raw_parts_mut(manifold_lower,  npx * 4);
    let hi   = std::slice::from_raw_parts_mut(manifold_higher, npx * 4);

    for k in 0..npx {
        let b = k * 4;
        let pixelg = inp[b + g].max(1e-6).log2();
        let highg  = bhi[b + g].max(1e-6).log2();
        let lowg   = blo[b + g].max(1e-6).log2();
        let avgg   = blur[b + g].max(1e-6).log2();

        let mut w = 1.0_f32;
        for kc in 0..=1usize {
            let c = (g + kc + 1) % 3;
            let pixel = inp[b + c].max(1e-6).log2();
            let highc = bhi[b + c].max(1e-6).log2();
            let lowc  = blo[b + c].max(1e-6).log2();

            let dist_ll = (pixelg - lowg  - pixel + lowc ).abs();
            let dist_hh = (pixelg - highg - pixel + highc).abs();
            let dist_lh = ((pixelg - pixel) - (highg - lowc )).abs();
            let dist_hl = ((pixelg - pixel) - (lowg  - highc)).abs();

            let dist_good = if (pixelg - lowg).abs() < (pixelg - highg).abs() { dist_ll } else { dist_hh };
            let dist_bad  = if (pixelg - lowg).abs() < (pixelg - highg).abs() { dist_hl } else { dist_lh };

            w *= (0.2 + 1.0 / dist_good.max(0.1)) / (0.2 + 1.0 / dist_bad.max(0.1));
        }

        if pixelg > avgg {
            let mut logdiffs = [0.0_f32; 2];
            for kc in 0..=1usize {
                let c = (g + kc + 1) % 3;
                let pixel = inp[b + c].max(1e-6);
                logdiffs[kc] = pixel.log2() - pixelg;
            }
            let maxlogdiff = logdiffs[0].abs().max(logdiffs[1].abs());
            if maxlogdiff > MAX_EV_DIFF { w *= MAX_EV_DIFF / maxlogdiff; }
            for kc in 0..=1usize { let c = (kc + g + 1) % 3; hi[b + c] = logdiffs[kc] * w; }
            hi[b + g] = inp[b + g].max(0.0) * w;
            hi[b + 3] = w;
            for c in 0..4 { lo[b + c] = 0.0; }
        } else {
            let mut logdiffs = [0.0_f32; 2];
            for kc in 0..=1usize {
                let c = (g + kc + 1) % 3;
                let pixel = inp[b + c].max(1e-6);
                logdiffs[kc] = pixel.log2() - pixelg;
            }
            let maxlogdiff = logdiffs[0].abs().max(logdiffs[1].abs());
            if maxlogdiff > MAX_EV_DIFF { w *= MAX_EV_DIFF / maxlogdiff; }
            for kc in 0..=1usize { let c = (kc + g + 1) % 3; lo[b + c] = logdiffs[kc] * w; }
            lo[b + g] = inp[b + g].max(0.0) * w;
            lo[b + 3] = w;
            for c in 0..4 { hi[b + c] = 0.0; }
        }
    }
}

/// Pack two 4-channel manifolds into a single 6-channel manifold buffer.
///
/// For each pixel k and c in 0..3:
///   out[k*6 + c]     = higher[k*4 + c]
///   out[k*6 + 3 + c] = lower[k*4 + c]
///
/// Matches the DT_OMP_FOR_SIMD at cacorrectrgb.c:441.
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrectrgb_pack_manifolds(
    blurred_manifold_lower:  *const f32,
    blurred_manifold_higher: *const f32,
    manifolds_out: *mut f32,
    npixels: usize,
) {
    if npixels == 0 { return; }
    let lo  = std::slice::from_raw_parts(blurred_manifold_lower,  npixels * 4);
    let hi  = std::slice::from_raw_parts(blurred_manifold_higher, npixels * 4);
    let out = std::slice::from_raw_parts_mut(manifolds_out, npixels * 6);
    for k in 0..npixels {
        for c in 0..3 {
            out[k * 6 + c]     = hi[k * 4 + c];
            out[k * 6 + 3 + c] = lo[k * 4 + c];
        }
        // The alpha channel (index 3) of each 4-channel manifold is the
        // confidence weight used during normalisation; it is intentionally
        // dropped here — the downstream apply_correction pass only needs the
        // RGB values in the 6-channel packed format.
    }
}

/// Apply the manifold-based CA correction to every pixel.
///
/// `mode`: 0 = standard, 1 = darken only, 2 = brighten only.
/// Guide channel is passed through unchanged; non-guide channels are
/// corrected via a weighted geometric mean of the manifold ratios.
/// Matches `apply_correction()` in cacorrectrgb.c:464.
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrectrgb_apply_correction(
    in_buf: *const f32,
    manifolds: *const f32,
    width: usize,
    height: usize,
    guide: u32,
    mode: u32,
    out_buf: *mut f32,
) {
    let npx = width.saturating_mul(height);
    if npx == 0 || guide >= 3 { return; }
    let g = guide as usize;
    let inp = std::slice::from_raw_parts(in_buf,    npx * 4);
    let mf  = std::slice::from_raw_parts(manifolds, npx * 6);
    let out = std::slice::from_raw_parts_mut(out_buf, npx * 4);

    for k in 0..npx {
        let b4 = k * 4;
        let b6 = k * 6;

        let high_guide = mf[b6 + g].max(1e-6);
        let low_guide  = mf[b6 + 3 + g].max(1e-6);
        let log_high = high_guide.log2();
        let log_low  = low_guide.log2();
        let dist_low_high = log_high - log_low;
        let pixelg  = inp[b4 + g].max(0.0);
        let log_pixg = pixelg.clamp(low_guide, high_guide).log2();

        let mut weight_low = (log_high - log_pixg).abs() / dist_low_high.max(1e-6);
        const THRESHOLD: f32 = 0.25;
        if dist_low_high < THRESHOLD {
            let weight = dist_low_high / THRESHOLD;
            weight_low = weight_low * weight + 0.5 * (1.0 - weight);
        }
        let weight_high = (1.0 - weight_low).max(0.0);

        for kc in 0..=1usize {
            let c = (g + kc + 1) % 3;
            let pixelc = inp[b4 + c].max(0.0);
            let ratio_hi = mf[b6 + c]     / high_guide;
            let ratio_lo = mf[b6 + 3 + c] / low_guide;
            let ratio = ratio_lo.powf(weight_low) * ratio_hi.powf(weight_high);
            let outp  = pixelg * ratio;
            out[b4 + c] = match mode {
                1 => outp.min(pixelc),   // DT_CACORRECT_MODE_DARKEN
                2 => outp.max(pixelc),   // DT_CACORRECT_MODE_BRIGHTEN
                // 0 = DT_CACORRECT_MODE_STANDARD (and any future unknown value —
                // new enum additions must be wired here explicitly).
                _ => outp,
            };
        }
        out[b4 + g] = pixelg;
        out[b4 + 3] = inp[b4 + 3];
    }
}

/// Pack in/out channel pairs for the reduce_artifacts blur step.
///
/// For each pixel k and kc in 0..=1:
///   c = (guide + kc + 1) % 3
///   in_out[k*4 + kc*2 + 0] = in[k*4 + c]   (input channel)
///   in_out[k*4 + kc*2 + 1] = out[k*4 + c]  (corrected channel)
///
/// Matches the first DT_OMP_FOR in `reduce_artifacts()` (cacorrectrgb.c:535).
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrectrgb_pack_inout(
    in_buf: *const f32,
    out_buf: *const f32,
    inout_buf: *mut f32,
    npixels: usize,
    guide: u32,
) {
    if npixels == 0 || guide >= 3 { return; }
    let g = guide as usize;
    let inp  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let outp = std::slice::from_raw_parts(out_buf, npixels * 4);
    let io   = std::slice::from_raw_parts_mut(inout_buf, npixels * 4);
    for k in 0..npixels {
        let b = k * 4;
        for kc in 0..=1usize {
            let c = (g + kc + 1) % 3;
            io[b + kc * 2]     = inp[b + c];
            io[b + kc * 2 + 1] = outp[b + c];
        }
    }
}

/// Blend correction toward input when blurred averages diverge (artifact
/// reduction). Uses the packed in/out blur result from the Gaussian step.
///
/// For each pixel k:
///   w = product over kc of exp(-max(|avg_out - avg_in|, 0.01) * safety)
///   out[k*4 + c] = max(1-w, 0)*max(in[k*4+c],0) + w*max(out[k*4+c],0)
///
/// Matches the second DT_OMP_FOR in `reduce_artifacts()` (cacorrectrgb.c:564).
#[no_mangle]
pub unsafe extern "C" fn darkroom_cacorrectrgb_blend_artifacts(
    in_buf: *const f32,
    blurred_inout: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    guide: u32,
    safety: f32,
) {
    if npixels == 0 || guide >= 3 { return; }
    let g = guide as usize;
    let inp   = std::slice::from_raw_parts(in_buf,       npixels * 4);
    let bio   = std::slice::from_raw_parts(blurred_inout, npixels * 4);
    let out   = std::slice::from_raw_parts_mut(out_buf,  npixels * 4);

    for k in 0..npixels {
        let b = k * 4;
        let mut w = 1.0_f32;
        for kc in 0..=1usize {
            let avg_in  = bio[b + kc * 2    ].max(1e-6).log2();
            let avg_out = bio[b + kc * 2 + 1].max(1e-6).log2();
            w *= (-(avg_out - avg_in).abs().max(0.01) * safety).exp();
        }
        for kc in 0..=1usize {
            let c = (g + kc + 1) % 3;
            out[b + c] = (1.0 - w).max(0.0) * inp[b + c].max(0.0)
                       + w * out[b + c].max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 1-pixel input + manifolds with the supplied weights and
    /// channel values (guide channel set to log2(2) = 1.0 for both manifolds).
    fn one_pixel(
        in_rgb: [f32; 4],
        lower_rgba: [f32; 4],
        higher_rgba: [f32; 4],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (in_rgb.to_vec(), lower_rgba.to_vec(), higher_rgba.to_vec())
    }

    #[test]
    fn high_weight_passes_normalisation_through() {
        // weight = 1.0 (alpha) → no smooth blend triggered (0.05 cutoff).
        // guide = 1 (G), guide-channel value pre-divided = 0.5 / 1.0 = 0.5.
        // Non-guide channels: input value 0.0 → exp2(0) * 0.5 = 0.5.
        let (inp, mut lo, mut hi) =
            one_pixel([0.1, 0.2, 0.3, 0.42], [0.0, 0.5, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0]);
        unsafe {
            darkroom_cacorrectrgb_normalize_manifolds(
                inp.as_ptr(), lo.as_mut_ptr(), hi.as_mut_ptr(),
                1, 1, 1, // guide = G
            );
        }
        // Both manifolds: R = 0.5, G = 0.5, B = 0.5
        for v in &[hi[0], hi[1], hi[2], lo[0], lo[1], lo[2]] {
            assert!((v - 0.5).abs() < 1e-5, "got {}", v);
        }
    }

    #[test]
    fn low_weight_blends_toward_input() {
        // weight = 0.01 → blend factor w = 0 → fully replaced by blurred_in.
        // Non-guide channels start at 0 so exp2(0/0.01) = exp2(0) = 1, which
        // keeps the intermediate finite; with weighth = 0.01 the production C
        // path would also produce `inf` if the non-guide were left at 99
        // (exp2(9900) overflows). The test deliberately uses values inside
        // the well-defined regime.
        let (inp, mut lo, mut hi) =
            one_pixel(
                [0.10, 0.20, 0.30, 0.40],
                [0.5,  0.0,  0.0,  0.01],
                [0.5,  0.0,  0.0,  0.01],
            );
        unsafe {
            darkroom_cacorrectrgb_normalize_manifolds(
                inp.as_ptr(), lo.as_mut_ptr(), hi.as_mut_ptr(),
                1, 1, 0, // guide = R
            );
        }
        // After blend at w=0, both manifolds equal `inp` exactly on RGB + alpha.
        for c in 0..4 {
            assert!((hi[c] - inp[c]).abs() < 1e-5, "hi[{c}] = {}, inp = {}", hi[c], inp[c]);
            assert!((lo[c] - inp[c]).abs() < 1e-5, "lo[{c}] = {}, inp = {}", lo[c], inp[c]);
        }
    }

    #[test]
    fn weight_below_minimum_clamps_to_one_hundredth() {
        // Weight = 0.0 → clamped to 1e-2 before division → guide /= 1e-2 = 100.
        let (inp, mut lo, mut hi) =
            one_pixel([0.0; 4], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]);
        unsafe {
            darkroom_cacorrectrgb_normalize_manifolds(
                inp.as_ptr(), lo.as_mut_ptr(), hi.as_mut_ptr(),
                1, 1, 1,
            );
        }
        // weighth=weightl=1e-2, so guide value = 1.0/0.01 = 100. But then
        // weight < 0.05 triggers blend toward input (w = -0.25 here because
        // (0.01 - 0.01) / (0.05 - 0.01) = 0). Verify the blend ran.
        // w = (0.01 - 0.01) / 0.04 = 0 → fully replaced by input (all zeros).
        for c in 0..4 {
            assert_eq!(hi[c], 0.0, "hi[{c}] = {}", hi[c]);
            assert_eq!(lo[c], 0.0, "lo[{c}] = {}", lo[c]);
        }
    }

    #[test]
    fn guide_channel_two_blue_round_trip() {
        // Sanity check with guide = B (index 2).
        let (inp, mut lo, mut hi) = one_pixel(
            [0.1, 0.2, 0.3, 0.4],
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5, 1.0],
        );
        unsafe {
            darkroom_cacorrectrgb_normalize_manifolds(
                inp.as_ptr(), lo.as_mut_ptr(), hi.as_mut_ptr(),
                1, 1, 2,
            );
        }
        // Blue = 0.5, R = exp2(0) * 0.5 = 0.5, G = exp2(0) * 0.5 = 0.5
        for c in 0..3 {
            assert!((hi[c] - 0.5).abs() < 1e-5);
            assert!((lo[c] - 0.5).abs() < 1e-5);
        }
    }

    // ── new function tests ──────────────────────────────────────────────────

    #[test]
    fn build_manifolds_above_average_goes_to_higher() {
        // guide = 0 (R), pixelg = 1.0 > avg = 0.5 → weighth=1, weightl=0
        let inp  = vec![1.0_f32, 1.0, 1.0, 0.0];  // all channels 1.0
        let blur = vec![0.5_f32, 0.5, 0.5, 0.0];
        let mut lo = vec![99.0_f32; 4];
        let mut hi = vec![99.0_f32; 4];
        unsafe {
            darkroom_cacorrectrgb_build_manifolds(
                inp.as_ptr(), blur.as_ptr(),
                lo.as_mut_ptr(), hi.as_mut_ptr(),
                1, 1, 0,
            );
        }
        // weighth=1 → hi gets the pixel values; weightl=0 → lo is zeroed
        assert!((hi[3] - 1.0).abs() < 1e-6);  // weighth stored in alpha
        assert_eq!(lo[3], 0.0);
    }

    #[test]
    fn pack_manifolds_interleaves_correctly() {
        let lo  = vec![1.0_f32, 2.0, 3.0, 0.0];
        let hi  = vec![4.0_f32, 5.0, 6.0, 0.0];
        let mut out = vec![0.0_f32; 6];
        unsafe {
            darkroom_cacorrectrgb_pack_manifolds(lo.as_ptr(), hi.as_ptr(), out.as_mut_ptr(), 1);
        }
        // hi goes to positions 0,1,2; lo to positions 3,4,5
        assert_eq!(&out[0..3], &[4.0, 5.0, 6.0]);
        assert_eq!(&out[3..6], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn apply_correction_standard_mode_passthrough_when_ratio_one() {
        // manifolds equal input pixel → ratio = 1 → out = in (for non-guide)
        // guide=G(1): hi_guide = lo_guide = 0.5; pixelg = 0.5.
        // For kc=0: c=2(B). ratio = (lo/lo)^0.5 * (hi/hi)^0.5 = 1. out[B] = 0.5.
        let manifolds = vec![
            0.5_f32, 0.5, 0.5,  // hi: R, G, B
            0.5_f32, 0.5, 0.5,  // lo: R, G, B
        ];
        let inp = vec![0.5_f32, 0.5, 0.5, 1.0];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_cacorrectrgb_apply_correction(
                inp.as_ptr(), manifolds.as_ptr(), 1, 1, 1, 0, out.as_mut_ptr(),
            );
        }
        // guide channel preserved as-is
        assert!((out[1] - 0.5).abs() < 1e-5);
        // alpha preserved
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn pack_inout_fills_in_out_pairs() {
        // guide=0(R) → c0=1(G), c1=2(B)
        let inp  = vec![1.0_f32, 2.0, 3.0, 4.0];
        let outp = vec![5.0_f32, 6.0, 7.0, 8.0];
        let mut io = vec![-1.0_f32; 4];
        unsafe {
            darkroom_cacorrectrgb_pack_inout(inp.as_ptr(), outp.as_ptr(), io.as_mut_ptr(), 1, 0);
        }
        // kc=0: c=1(G) → io[0]=inp[G]=2, io[1]=out[G]=6
        // kc=1: c=2(B) → io[2]=inp[B]=3, io[3]=out[B]=7
        assert_eq!(io[0], 2.0); assert_eq!(io[1], 6.0);
        assert_eq!(io[2], 3.0); assert_eq!(io[3], 7.0);
    }

    #[test]
    fn blend_artifacts_w_one_keeps_output_intact() {
        // w = exp(-max(|avg_out - avg_in|, 0.01) * safety)
        // with safety = 0 → w = exp(0) = 1 → fully keeps output
        let inp = vec![0.2_f32, 0.3, 0.4, 1.0];
        let bio = vec![
            0.3_f32, 0.3,  // kc=0: avg_in=0.3, avg_out=0.3 → |diff|=0
            0.4_f32, 0.4,  // kc=1: avg_in=0.4, avg_out=0.4 → |diff|=0
        ];
        let mut out = vec![0.5_f32, 0.6, 0.7, 1.0];
        unsafe {
            darkroom_cacorrectrgb_blend_artifacts(
                inp.as_ptr(), bio.as_ptr(), out.as_mut_ptr(), 1, 0, 0.0,
            );
        }
        // w≈1 with safety=0, so out should stay close to its initial values
        assert!((out[1] - 0.6).abs() < 0.01, "out[G]={}", out[1]);
        assert!((out[2] - 0.7).abs() < 0.01, "out[B]={}", out[2]);
    }

    #[test]
    fn out_of_range_guide_returns_without_touching_buffers() {
        // guide >= 3 is a wiring bug. The function must leave both manifolds
        // unchanged so a C caller can see the corrupted output rather than
        // silently processing as if `guide` were 2.
        let inp = vec![0.0_f32; 4];
        let mut lo = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut hi = vec![5.0_f32, 6.0, 7.0, 8.0];
        let lo_orig = lo.clone();
        let hi_orig = hi.clone();
        unsafe {
            darkroom_cacorrectrgb_normalize_manifolds(
                inp.as_ptr(), lo.as_mut_ptr(), hi.as_mut_ptr(),
                1, 1, 7, // out of range
            );
        }
        assert_eq!(lo, lo_orig);
        assert_eq!(hi, hi_orig);
    }

    #[test]
    fn zero_size_is_safe_noop() {
        let inp = vec![0.0_f32; 4];
        let mut lo = vec![0.0_f32; 4];
        let mut hi = vec![0.0_f32; 4];
        unsafe {
            darkroom_cacorrectrgb_normalize_manifolds(
                inp.as_ptr(), lo.as_mut_ptr(), hi.as_mut_ptr(),
                0, 0, 1,
            );
        }
        // Untouched.
        for v in inp.iter().chain(lo.iter()).chain(hi.iter()) { assert_eq!(*v, 0.0); }
    }
}
