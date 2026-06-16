//! Live preview pipeline: applies an ordered chain of migrated `darkroom-core`
//! IOPs to the decoded 8-bit preview image so the darkroom view shows
//! *processed* output (not just the file). This is the UI↔core processing seam
//! — a stepping-stone toward a full Rust pixelpipe (RUST_MIGRATION_PLAN.md
//! Phase 3 milestone 2).
//!
//! Phase 3-ui-13 generalises the original single-IOP (exposure) seam into a
//! small pipeline of stages, each backed by a migrated core IOP that works in
//! RGB [0,1]:
//!   1. **exposure** — `out = (in - black) * scale`, `scale = 2^ev`
//!   2. **velvia**   — saturation-weighted chroma boost
//!
//! Both run on the colour channels (0..min(3,nch)); any alpha channel and the
//! rowstride padding are preserved byte-for-byte.

use darkroom_core::iop::{exposure, velvia};

/// Live, user-tunable parameters for the preview pipeline. Each enabled stage
/// runs the corresponding migrated `darkroom-core` IOP, in pixelpipe order
/// (exposure then velvia).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreviewParams {
    /// Exposure stage on/off.
    pub exposure_on: bool,
    /// Black point subtracted before scaling (in normalised [0,1]).
    pub black: f32,
    /// Exposure value; the multiplicative scale is `2^ev`.
    pub ev: f32,
    /// Velvia stage on/off.
    pub velvia_on: bool,
    /// Velvia strength on the C slider's 0..100 scale (divided by 100 for core).
    pub velvia_strength: f32,
    /// Velvia mid-tones bias, 0..1.
    pub velvia_bias: f32,
}

impl Default for PreviewParams {
    fn default() -> Self {
        Self {
            exposure_on: true,
            black: 0.0,
            ev: 0.0,
            velvia_on: false,
            velvia_strength: 25.0,
            velvia_bias: 1.0,
        }
    }
}

impl PreviewParams {
    /// True when the pipeline would leave the image unchanged, so the caller
    /// can skip re-processing/uploading. Exposure is a no-op at ev 0 & black 0;
    /// velvia is a no-op when off or at strength 0.
    pub fn is_identity(&self) -> bool {
        // Exact `== 0.0` is intentional: ev/black come straight from GTK slider
        // positions (cast of an f64 that is exactly 0.0 at the default), never
        // from arithmetic, so 0.0 is exactly representable here.
        let exp_identity = !self.exposure_on || (self.ev == 0.0 && self.black == 0.0);
        let vel_identity = !self.velvia_on || self.velvia_strength <= 0.0;
        exp_identity && vel_identity
    }
}

/// Run the preview pipeline over an 8-bit interleaved image buffer, preserving
/// layout (rowstride) and any alpha channel. Colour channels (0..min(3,nch))
/// are normalised to [0,1], chained through the enabled migrated IOPs and
/// written back; alpha (channel 3, if present) and inter-row padding are kept
/// byte-for-byte from the input.
///
/// Note: this operates on gamma-encoded 8-bit data (normalised to [0,1] from
/// the decoded pixbuf), *not* the linear scene-referred float of the real
/// darktable pixelpipe. IOPs with non-linear behaviour (velvia, tone curves)
/// will therefore not match their pixelpipe equivalents exactly — this is a
/// deliberate stepping-stone (see the module doc), not the final pipeline.
///
/// Allocation note: each call allocates a few full-image buffers (gather +
/// per-stage). Acceptable for the stepping-stone preview; reuse a working
/// buffer once the real pixelpipe orchestrator lands.
pub fn apply_pipeline(
    base: &[u8],
    width: usize,
    height: usize,
    rowstride: usize,
    nch: usize,
    params: &PreviewParams,
) -> Vec<u8> {
    // Degenerate input: nothing to process (also guards `colour - 1` below
    // against underflow when nch == 0).
    if nch == 0 || width == 0 || height == 0 {
        return base.to_vec();
    }
    let colour = nch.min(3);
    let n = width * height;

    // ── gather colour → packed RGB f32 [0,1] ───────────────────────────────
    // Sources with fewer than 3 colour channels replicate their last channel
    // (e.g. greyscale → R=G=B); pixbufs are normally 3 or 4 channels.
    let mut rgb = vec![0.0f32; n * 3];
    for y in 0..height {
        let row = y * rowstride;
        for x in 0..width {
            let p = row + x * nch;
            let o = (y * width + x) * 3;
            for c in 0..3 {
                let src = c.min(colour - 1);
                rgb[o + c] = base[p + src] as f32 / 255.0;
            }
        }
    }

    // ── exposure stage (operates element-wise on RGB) ──────────────────────
    if params.exposure_on && (params.ev != 0.0 || params.black != 0.0) {
        let scale = 2.0f32.powf(params.ev);
        let mut out = vec![0.0f32; rgb.len()];
        exposure::process_pixels(&rgb, &mut out, params.black, scale);
        rgb = out;
    }

    // ── velvia stage (wants RGBA 4-wide chunks; alpha is copied through) ────
    if params.velvia_on && params.velvia_strength > 0.0 {
        let mut rgba = vec![0.0f32; n * 4];
        for i in 0..n {
            rgba[i * 4] = rgb[i * 3];
            rgba[i * 4 + 1] = rgb[i * 3 + 1];
            rgba[i * 4 + 2] = rgb[i * 3 + 2];
            rgba[i * 4 + 3] = 1.0; // velvia only reads alpha to pass it through
        }
        let mut out = vec![0.0f32; rgba.len()];
        velvia::process_pixels(
            &rgba,
            &mut out,
            params.velvia_strength / 100.0,
            params.velvia_bias,
        );
        for i in 0..n {
            rgb[i * 3] = out[i * 4];
            rgb[i * 3 + 1] = out[i * 4 + 1];
            rgb[i * 3 + 2] = out[i * 4 + 2];
        }
    }

    // ── scatter back, preserving alpha + rowstride padding ─────────────────
    let mut outbuf = base.to_vec();
    for y in 0..height {
        let row = y * rowstride;
        for x in 0..width {
            let p = row + x * nch;
            let o = (y * width + x) * 3;
            for c in 0..colour {
                outbuf[p + c] = (rgb[o + c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    outbuf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exposure_only(black: f32, ev: f32) -> PreviewParams {
        PreviewParams {
            exposure_on: true,
            black,
            ev,
            velvia_on: false,
            ..PreviewParams::default()
        }
    }

    #[test]
    fn identity_params_leave_image_unchanged() {
        // default: exposure on but ev 0 & black 0, velvia off ⇒ no-op
        let p = PreviewParams::default();
        assert!(p.is_identity());
        let base = vec![10u8, 20, 30, 200, 100, 50];
        let out = apply_pipeline(&base, 2, 1, 6, 3, &p);
        assert_eq!(out, base);
    }

    #[test]
    fn ev_plus_one_doubles_and_clamps_keeps_alpha() {
        // RGBA: ev=+1 ⇒ scale 2; colour doubles (clamped at 255), alpha kept.
        // 50/255*2 = 0.392 → *255 ≈ 100; 200/255*2 clamps to 255.
        let base = vec![50u8, 200, 25, 111];
        let out = apply_pipeline(&base, 1, 1, 4, 4, &exposure_only(0.0, 1.0));
        assert_eq!(out[0], 100);
        assert_eq!(out[1], 255); // clamped
        assert_eq!(out[2], 50);
        assert_eq!(out[3], 111); // alpha unchanged
    }

    #[test]
    fn black_point_lifts_shadows() {
        // ev=0 (scale 1), black=0.1 ⇒ out = in - 0.1 (in [0,1]).
        // 128/255 = 0.502 → 0.402 → *255 ≈ 102.5 → 103
        let base = vec![128u8, 128, 128];
        let out = apply_pipeline(&base, 1, 1, 3, 3, &exposure_only(0.1, 0.0));
        assert_eq!(out[0], 103);
    }

    #[test]
    fn respects_rowstride_padding() {
        // 1x2 RGB with 2 padding bytes per row; padding must be preserved.
        let base = vec![10u8, 20, 30, 0xAA, 0xBB, 40, 50, 60, 0xCC, 0xDD];
        let out = apply_pipeline(&base, 1, 2, 5, 3, &PreviewParams::default());
        assert_eq!(out, base); // identity, padding intact
    }

    #[test]
    fn velvia_boosts_saturation_of_a_colourful_pixel() {
        // A saturated pixel should get *more* saturated (max channel up or min
        // channel down) once velvia is enabled; a neutral grey is untouched.
        let mut p = PreviewParams::default();
        p.velvia_on = true;
        p.velvia_strength = 50.0;

        // neutral grey: velvia leaves it (very nearly) unchanged
        let grey = vec![128u8, 128, 128];
        let g_out = apply_pipeline(&grey, 1, 1, 3, 3, &p);
        assert_eq!(g_out, grey);

        // saturated reddish pixel: spread between max and min must not shrink
        let base = vec![200u8, 60, 40];
        let out = apply_pipeline(&base, 1, 1, 3, 3, &p);
        let in_spread = 200i32 - 40;
        let out_spread = out[0] as i32 - out[2] as i32;
        assert!(
            out_spread >= in_spread,
            "velvia should not reduce saturation: {out_spread} < {in_spread}"
        );
    }

    #[test]
    fn degenerate_input_returns_copy_without_panicking() {
        // nch == 0 must not underflow `colour - 1`; zero dims short-circuit.
        let mut p = PreviewParams::default();
        p.ev = 1.0;
        assert_eq!(apply_pipeline(&[], 0, 0, 0, 0, &p), Vec::<u8>::new());
        let base = vec![10u8, 20, 30];
        assert_eq!(apply_pipeline(&base, 0, 1, 3, 3, &p), base); // width 0
        assert_eq!(apply_pipeline(&base, 1, 1, 3, 0, &p), base); // nch 0
    }

    #[test]
    fn velvia_off_is_identity_even_at_high_strength() {
        let mut p = PreviewParams::default();
        p.velvia_on = false;
        p.velvia_strength = 100.0;
        assert!(p.is_identity());
        let base = vec![200u8, 60, 40];
        assert_eq!(apply_pipeline(&base, 1, 1, 3, 3, &p), base);
    }
}
