//! DWT detail recovery for the RGB denoise task (u7c).
//!
//! Ports `dt_restore_apply_detail_recovery` (`src/common/ai/restore_rgb.c:730-784`)
//! plus its helpers `_compute_adaptive_noise` (`:104-134`) and
//! `_dwt_sigma_mul_default` (`:63-74`): given the original frame and the
//! model's denoised output, the per-pixel Rec.709 luma residual
//! (original minus denoised, `:748-760`) is DWT-filtered with adaptive
//! per-band noise thresholds and a fraction `alpha` of the filtered residual
//! is added back to every RGB lane (`:774-781`).
//!
//! Buffer-shape adaptation: the C holds BOTH frames as interleaved 4-channel
//! buffers and adds the scalar residual to lanes 0..2 (`:778-780`); the u7b
//! pipeline hands us the original as interleaved RGBA (`denoise_rgb` input
//! contract) but the denoised frame as PACKED RGB (`denoise_rgb` output, no
//! alpha lane). The luma reads therefore index RGBA for the original and
//! packed RGB for the denoised frame, and the result is packed RGB. The alpha
//! lane of the original is ignored (it never enters the luma, same as the C
//! which reads lanes 0..2 only).
//!
//! Deliberate deviation: the C reads per-band multipliers from darktablerc
//! (`plugins/lighttable/neural_restore/detail_recovery_bands`, `:109-121`)
//! with `_dwt_sigma_mul_default` as the fallback. This port hardcodes the
//! defaults (`DETAIL_SIGMA_MUL`) and skips the config override — c41 has no
//! darktablerc, and a second tuning surface for an experimental panel is not
//! worth the plumbing. Recorded for the PARITY_AUDIT 2.7 follow-up, not
//! silently dropped.
//!
//! The DWT itself is [`crate::dwt::denoise`] (ported in m4-79): single
//! channel, in place, `bands` scales with one threshold per band — the same
//! `dwt_denoise(lum_residual, width, height, DWT_DETAIL_BANDS, noise)` call
//! the C makes at `:764-765`.
//!
//! `alpha` comes from the panel strength slider via [`strength_to_alpha`]:
//! `recovery_alpha = 1 - strength/100` (`src/libs/neural_restore.c:885`),
//! so 100 means full denoise (no recovery) and 0 means source-like. An
//! `alpha` of exactly 0 short-circuits to a copy of the denoised frame —
//! bit-exact, and it skips the DWT entirely, mirroring the C's
//! `need_buffer = recovery_alpha > 0` gate (`neural_restore.c:886`).

/// Wavelet detail bands: `DWT_DETAIL_BANDS` in the C (restore_rgb.c:764).
pub const DETAIL_BANDS: usize = 5;

/// Per-band residual-sigma multipliers, band 0 (finest) first.
/// Exact copy of `_dwt_sigma_mul_default` (restore_rgb.c:68-74): fine-scale
/// features are hardest to tell from noise, so the finest band is suppressed
/// most and the coarsest band keeps almost everything. The C allows a
/// darktablerc override; this port deliberately does not (see module doc).
pub const DETAIL_SIGMA_MUL: [f32; 5] = [0.25, 0.15, 0.05, 0.02, 0.01];

/// Rec.709 luma weights (the Y row of sRGB-to-XYZ D65), matching the C's
/// inline `0.2126/0.7152/0.0722` coefficients (restore_rgb.c:750-758) and
/// `_luma_rec709` (`:99-102`).
pub const LUMA_REC709: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Map the panel Strength slider (0..100, default 100) to the recovery
/// fraction `alpha`: `recovery_alpha = 1 - strength/100`
/// (`src/libs/neural_restore.c:885`). 100 gives 0 (full model output, no
/// recovery); 0 gives 1 (the whole filtered residual back, source-like).
/// Out-of-range inputs clamp to the slider domain first — the C relies on
/// the widget bounds; a library fn cannot.
#[inline]
pub fn strength_to_alpha(strength: f32) -> f32 {
    1.0 - strength.clamp(0.0, 100.0) / 100.0
}

/// Per-band adaptive noise thresholds from the residual's standard deviation:
/// population sigma (`sqrt(sum2/n - mean^2)`, f64 accumulation exactly like
/// `restore_rgb.c:123-130`) times each band multiplier (`:132-133`).
/// An empty residual yields all-zero thresholds (the C would divide by zero).
pub fn adaptive_noise(residual: &[f32]) -> [f32; DETAIL_BANDS] {
    if residual.is_empty() {
        return [0.0; DETAIL_BANDS];
    }
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for &v in residual {
        let d = v as f64;
        sum += d;
        sum2 += d * d;
    }
    let n = residual.len() as f64;
    let mean = sum / n;
    let sigma = (sum2 / n - mean * mean).max(0.0).sqrt() as f32;
    let mut noise = [0.0f32; DETAIL_BANDS];
    for (b, n) in noise.iter_mut().enumerate() {
        *n = sigma * DETAIL_SIGMA_MUL[b];
    }
    noise
}

/// Blend denoised output with DWT-filtered original detail.
///
/// `original_rgba` is interleaved RGBA (`w*h*4` f32, linear working space —
/// the `denoise_rgb` input contract); `denoised_rgb` is packed RGB
/// (`w*h*3` f32 — the `denoise_rgb` output). Returns packed RGB
/// (`w*h*3` f32): `denoised + alpha * filtered_residual` per RGB lane, the
/// same scalar added to all three lanes like the C (`:776-780`).
///
/// Contract notes: `alpha == 0.0` returns the denoised frame bit-exact
/// (short-circuit, no DWT — the C `need_buffer` gate). An empty frame
/// returns an empty vec. Mismatched buffer lengths degrade to a copy of the
/// denoised frame rather than panicking — the panel drives this from cached
/// buffers that always agree, so a mismatch means a bug upstream, and a
/// wrong-but-finite image beats a crashed UI thread.
pub fn apply_detail_recovery(
    original_rgba: &[f32],
    denoised_rgb: &[f32],
    w: u32,
    h: u32,
    alpha: f32,
) -> Vec<f32> {
    let npix = w as usize * h as usize;
    if npix == 0 {
        return Vec::new();
    }
    if original_rgba.len() != npix * 4 || denoised_rgb.len() != npix * 3 {
        return denoised_rgb.to_vec();
    }
    if alpha == 0.0 {
        return denoised_rgb.to_vec();
    }
    // Luma residual, orig minus den (restore_rgb.c:748-760).
    let mut residual = vec![0.0f32; npix];
    let (rgba4, _) = original_rgba.as_chunks::<4>();
    let (rgb3, _) = denoised_rgb.as_chunks::<3>();
    for ((o, d), r) in rgba4.iter().zip(rgb3.iter()).zip(residual.iter_mut()) {
        let lum_orig =
            LUMA_REC709[0] * o[0] + LUMA_REC709[1] * o[1] + LUMA_REC709[2] * o[2];
        let lum_den =
            LUMA_REC709[0] * d[0] + LUMA_REC709[1] * d[1] + LUMA_REC709[2] * d[2];
        *r = lum_orig - lum_den;
    }
    // Adaptive thresholds + 5-band DWT filter in place (:762-765).
    let noise = adaptive_noise(&residual);
    crate::dwt::denoise(&mut residual, w as usize, h as usize, DETAIL_BANDS, &noise);
    // Re-inject alpha-scaled residual into every RGB lane (:774-781).
    let mut out = denoised_rgb.to_vec();
    let (quads, _) = out.as_chunks_mut::<3>();
    for (d, &r) in quads.iter_mut().zip(residual.iter()) {
        let add = alpha * r;
        d[0] += add;
        d[1] += add;
        d[2] += add;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn bands_and_multipliers_pin_the_c_defaults() {
        // DWT_DETAIL_BANDS is 5 and _dwt_sigma_mul_default reads
        // 0.25/0.15/0.05/0.02/0.01 (restore_rgb.c:63-74).
        assert_eq!(DETAIL_BANDS, 5);
        assert_eq!(DETAIL_SIGMA_MUL, [0.25, 0.15, 0.05, 0.02, 0.01]);
        assert_eq!(DETAIL_SIGMA_MUL.len(), DETAIL_BANDS);
        assert_eq!(LUMA_REC709, [0.2126, 0.7152, 0.0722]);
    }

    #[test]
    fn strength_maps_to_alpha_like_the_c_slider() {
        // recovery_alpha = 1 - strength/100 (neural_restore.c:885):
        // 100 = full denoise, 0 = source-like.
        assert_eq!(strength_to_alpha(100.0), 0.0);
        assert_eq!(strength_to_alpha(0.0), 1.0);
        assert!((strength_to_alpha(75.0) - 0.25).abs() < EPS);
        assert!((strength_to_alpha(50.0) - 0.5).abs() < EPS);
        // Out-of-range clamps to the widget domain.
        assert_eq!(strength_to_alpha(140.0), 0.0);
        assert_eq!(strength_to_alpha(-20.0), 1.0);
    }

    #[test]
    fn sigma_matches_hand_computed_population_variance() {
        // residual [1,2,3,4]: mean 2.5, E[x^2] 7.5, var 1.25, sigma ~= 1.118.
        let residual = [1.0f32, 2.0, 3.0, 4.0];
        let noise = adaptive_noise(&residual);
        let sigma = (1.25f64).sqrt() as f32;
        for b in 0..DETAIL_BANDS {
            assert!(
                (noise[b] - sigma * DETAIL_SIGMA_MUL[b]).abs() < 1e-6,
                "band {b}: {} vs {}",
                noise[b],
                sigma * DETAIL_SIGMA_MUL[b]
            );
        }
        // Flat residual has zero variance, hence zero thresholds.
        assert_eq!(adaptive_noise(&[0.3f32; 16]), [0.0; DETAIL_BANDS]);
        assert_eq!(adaptive_noise(&[]), [0.0; DETAIL_BANDS]);
    }

    #[test]
    fn residual_math_on_hand_values() {
        // 1x2 image. Pixel 0: orig (0.5,0.5,0.5), den (0.4,0.4,0.4).
        // Pixel 1: orig (0.2,0.6,0.1), den (0.3,0.3,0.3).
        // Luma weights sum to 1, so grey pixel 0 has residual exactly 0.1.
        let orig = vec![0.5, 0.5, 0.5, 1.0, 0.2, 0.6, 0.1, 1.0];
        let den = vec![0.4, 0.4, 0.4, 0.3, 0.3, 0.3];
        let w = 2u32;
        let h = 1u32;
        // Rebuild the expected residual by hand, then check the alpha=1
        // output equals denoised + filtered residual per lane. The DWT
        // filter is deterministic: pin it by running the same two steps
        // (adaptive_noise + dwt::denoise) the fn performs.
        let mut residual = vec![0.0f32; 2];
        let (orig4, _) = orig.as_chunks::<4>();
        let (den3, _) = den.as_chunks::<3>();
        for ((o, d), r) in orig4.iter().zip(den3.iter()).zip(residual.iter_mut()) {
            let lo = LUMA_REC709[0] * o[0] + LUMA_REC709[1] * o[1] + LUMA_REC709[2] * o[2];
            let ld = LUMA_REC709[0] * d[0] + LUMA_REC709[1] * d[1] + LUMA_REC709[2] * d[2];
            *r = lo - ld;
        }
        assert!((residual[0] - 0.1).abs() < 1e-6, "grey residual: {}", residual[0]);
        let hand1 = 0.2126 * 0.2 + 0.7152 * 0.6 + 0.0722 * 0.1 - 0.3;
        assert!((residual[1] - hand1).abs() < 1e-6, "colour residual: {}", residual[1]);
        let noise = adaptive_noise(&residual);
        crate::dwt::denoise(&mut residual, w as usize, h as usize, DETAIL_BANDS, &noise);
        let out = apply_detail_recovery(&orig, &den, w, h, 1.0);
        assert_eq!(out.len(), 6);
        for i in 0..2 {
            for c in 0..3 {
                let expect = den[i * 3 + c] + residual[i];
                assert!(
                    (out[i * 3 + c] - expect).abs() < 1e-6,
                    "px {i} lane {c}: {} vs {expect}",
                    out[i * 3 + c]
                );
            }
        }
        // The same scalar lands on all three lanes (the C adds one `d`).
        for i in 0..2 {
            let adds = [
                out[i * 3] - den[i * 3],
                out[i * 3 + 1] - den[i * 3 + 1],
                out[i * 3 + 2] - den[i * 3 + 2],
            ];
            assert!((adds[0] - adds[1]).abs() < 1e-6 && (adds[1] - adds[2]).abs() < 1e-6);
        }
    }

    #[test]
    fn alpha_zero_is_a_bit_exact_copy() {
        let orig = vec![0.7f32, 0.2, 0.9, 1.0, 0.1, 0.4, 0.6, 1.0];
        let den = vec![0.65f32, 0.25, 0.8, 0.15, 0.35, 0.55];
        let out = apply_detail_recovery(&orig, &den, 2, 1, 0.0);
        assert_eq!(out.len(), den.len());
        for (i, (o, d)) in out.iter().zip(den.iter()).enumerate() {
            assert!(o.to_bits() == d.to_bits(), "sample {i}: {o} vs {d}");
        }
    }

    #[test]
    fn alpha_effect_is_linear() {
        // out(alpha) - den == alpha * out(1) - den, lane by lane: the blend
        // is a single scalar multiply of the filtered residual.
        let (w, h) = (7u32, 5u32);
        let mut orig = vec![0.0f32; w as usize * h as usize * 4];
        let mut den = vec![0.0f32; w as usize * h as usize * 3];
        for (i, v) in orig.iter_mut().enumerate() {
            *v = 0.05 + ((i * 37) % 90) as f32 / 100.0;
        }
        for (i, v) in den.iter_mut().enumerate() {
            *v = 0.05 + ((i * 53) % 80) as f32 / 100.0;
        }
        let full = apply_detail_recovery(&orig, &den, w, h, 1.0);
        for &alpha in &[0.25f32, 0.5, 0.75] {
            let out = apply_detail_recovery(&orig, &den, w, h, alpha);
            for i in 0..out.len() {
                let expect = den[i] + alpha * (full[i] - den[i]);
                assert!((out[i] - expect).abs() < 1e-5, "alpha {alpha} sample {i}");
            }
        }
    }

    #[test]
    fn empty_and_mismatched_inputs_degrade_gracefully() {
        assert!(apply_detail_recovery(&[], &[], 0, 0, 0.5).is_empty());
        let den = vec![0.1f32, 0.2, 0.3];
        // Short original: copy of denoised, no panic.
        assert_eq!(apply_detail_recovery(&[0.5f32; 4], &den, 2, 1, 0.5), den);
        // Short denoised: echo back untouched.
        let short = vec![0.1f32, 0.2];
        assert_eq!(
            apply_detail_recovery(&[0.5f32; 8], &short, 2, 1, 0.5),
            short
        );
    }

    #[test]
    fn identical_frames_recover_nothing() {
        // Zero residual everywhere: sigma is 0, thresholds are 0, and the
        // re-injected amount is 0 — output equals denoised within float dust.
        let (w, h) = (9u32, 9u32);
        let mut orig = vec![0.0f32; w as usize * h as usize * 4];
        let mut den = vec![0.0f32; w as usize * h as usize * 3];
        for i in 0..w as usize * h as usize {
            let v = 0.1 + (i % 13) as f32 / 40.0;
            orig[i * 4] = v;
            orig[i * 4 + 1] = v;
            orig[i * 4 + 2] = v;
            orig[i * 4 + 3] = 1.0;
            den[i * 3] = v;
            den[i * 3 + 1] = v;
            den[i * 3 + 2] = v;
        }
        let out = apply_detail_recovery(&orig, &den, w, h, 1.0);
        for (i, (o, d)) in out.iter().zip(den.iter()).enumerate() {
            assert!((o - d).abs() < 1e-5, "sample {i}: {o} vs {d}");
        }
    }
}
