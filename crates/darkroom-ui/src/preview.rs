//! Live preview pipeline: applies an ordered chain of migrated `darkroom-core`
//! IOPs to the decoded 8-bit preview image so the darkroom view shows
//! *processed* output (not just the file). This is the UI↔core processing seam
//! — a stepping-stone toward a full Rust pixelpipe (RUST_MIGRATION_PLAN.md
//! Phase 3 milestone 2).
//!
//! Phase 3-m2-2: the stage chaining now lives in `darkroom_core::pipeline`.
//! [`PreviewParams::to_pipeline`] maps the UI sliders (UI ranges) to a
//! `Pipeline` of physical-param `Stage`s (exposure → velvia → splittoning →
//! monochrome); [`apply_pipeline`] just marshals the 8-bit pixbuf to/from the
//! float RGBA the core pipeline runs on, preserving the source alpha channel and
//! rowstride padding byte-for-byte.
//!
//! The 8-bit sRGB pixbuf is **decoded to linear light** (sRGB EOTF) on the way
//! in and re-encoded on the way out, so the core stages run in linear — the
//! same domain as the real pixelpipe. The remaining gap is that the input is a
//! display-referred 8-bit image, not true scene-referred raw: a stepping-stone
//! until a raw-decode/demosaic front end feeds `core::pipeline` directly.

use darkroom_core::pipeline::{Pipeline, Stage};

/// Live, user-tunable parameters for the preview pipeline. Each enabled stage
/// runs the corresponding migrated `darkroom-core` IOP, in pixelpipe order
/// (exposure → velvia → splittoning → monochrome; see [`Self::to_pipeline`]).
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
    /// Split-toning stage on/off.
    pub split_on: bool,
    /// Shadow hue, 0..1 (normalised, = degrees/360).
    pub split_shadow_hue: f32,
    /// Shadow saturation, 0..1.
    pub split_shadow_sat: f32,
    /// Highlight hue, 0..1.
    pub split_highlight_hue: f32,
    /// Highlight saturation, 0..1.
    pub split_highlight_sat: f32,
    /// Centre luminance of the shadow→highlight gradient, 0..1.
    pub split_balance: f32,
    /// Compress range on the C slider's 0..100 scale (core gets `(c/110)/2`).
    pub split_compress: f32,
    /// Monochrome (channel-mixer B&W) stage on/off.
    pub mono_on: bool,
    /// Grayscale mix weights for R, G, B (channelmixer GRAY mode, row 0).
    pub mono_r: f32,
    pub mono_g: f32,
    pub mono_b: f32,
    /// Sigmoid tone-mapping (scene-linear → display) stage on/off. Default-on
    /// for raws (scene-linear input); off for already-display-referred JPEGs.
    pub sigmoid_on: bool,
    /// Sigmoid contrast (middle_grey_contrast), 0.1..10, darktable default 1.5.
    pub sigmoid_contrast: f32,
    /// Sigmoid skew (contrast_skewness), -1..1, default 0.
    pub sigmoid_skew: f32,
}

impl Default for PreviewParams {
    fn default() -> Self {
        // Mirrors the darktable defaults for each IOP.
        Self {
            exposure_on: true,
            black: 0.0,
            ev: 0.0,
            velvia_on: false,
            velvia_strength: 25.0,
            velvia_bias: 1.0,
            split_on: false,
            split_shadow_hue: 0.0,
            split_shadow_sat: 0.5,
            split_highlight_hue: 0.2,
            split_highlight_sat: 0.5,
            split_balance: 0.5,
            split_compress: 33.0,
            mono_on: false,
            // Rec.709 luminance weights — a neutral B&W starting point.
            mono_r: 0.21,
            mono_g: 0.72,
            mono_b: 0.07,
            // Off by default (JPEG path); the darkroom view turns it on for raws.
            sigmoid_on: false,
            sigmoid_contrast: 1.5,
            sigmoid_skew: 0.0,
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
        // Split-toning has no value at which it is a strict no-op while enabled
        // (even sat 0 desaturates toned zones toward luminance), so on==off.
        let split_identity = !self.split_on;
        let mono_identity = !self.mono_on; // grayscale conversion is never a no-op
        let sigmoid_identity = !self.sigmoid_on; // a tone curve is never a no-op
        exp_identity && vel_identity && split_identity && mono_identity && sigmoid_identity
    }

    /// A copy with every stage disabled — `apply_pipeline` with it returns the
    /// input unchanged. Used by the darkroom view's before/after toggle to show
    /// the unprocessed image (and its histogram) without disturbing the params.
    pub fn bypassed(&self) -> Self {
        Self {
            exposure_on: false,
            velvia_on: false,
            split_on: false,
            mono_on: false,
            sigmoid_on: false,
            ..*self
        }
    }

    /// Serialise to a compact, versioned little-endian blob for DB persistence:
    /// `[version, 5×bool(u8), 15×f32_le]`. Decoded by [`PreviewParams::decode`].
    /// This is darkroom-ui's own layout (NOT a C IOP `op_params`), stored under a
    /// synthetic operation name the C reader ignores.
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(ENCODED_LEN);
        v.push(ENCODE_VERSION);
        for b in [self.exposure_on, self.velvia_on, self.split_on, self.mono_on, self.sigmoid_on] {
            v.push(b as u8);
        }
        for f in [
            self.black, self.ev,
            self.velvia_strength, self.velvia_bias,
            self.split_shadow_hue, self.split_shadow_sat,
            self.split_highlight_hue, self.split_highlight_sat,
            self.split_balance, self.split_compress,
            self.mono_r, self.mono_g, self.mono_b,
            self.sigmoid_contrast, self.sigmoid_skew,
        ] {
            v.extend_from_slice(&f.to_le_bytes());
        }
        v
    }

    /// Inverse of [`PreviewParams::encode`]. Returns `None` for the wrong version
    /// byte or wrong length (e.g. a blob written by an older/other schema), so
    /// the caller falls back to defaults rather than loading garbage.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != ENCODED_LEN || bytes[0] != ENCODE_VERSION {
            return None;
        }
        let bools = &bytes[1..6];
        // length is checked above, so exactly 15 f32 chunks follow
        let f: Vec<f32> = bytes[6..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Self {
            exposure_on: bools[0] != 0,
            velvia_on: bools[1] != 0,
            split_on: bools[2] != 0,
            mono_on: bools[3] != 0,
            sigmoid_on: bools[4] != 0,
            black: f[0], ev: f[1],
            velvia_strength: f[2], velvia_bias: f[3],
            split_shadow_hue: f[4], split_shadow_sat: f[5],
            split_highlight_hue: f[6], split_highlight_sat: f[7],
            split_balance: f[8], split_compress: f[9],
            mono_r: f[10], mono_g: f[11], mono_b: f[12],
            sigmoid_contrast: f[13], sigmoid_skew: f[14],
        })
    }

    /// Map the UI params to a `darkroom_core::pipeline::Pipeline`, converting UI
    /// ranges to the physical params the core stages expect (EV→scale, velvia
    /// strength /100, split compress (c/110)/2) and including only the enabled
    /// stages that would actually change the image (so a bypassed/neutral set
    /// yields an empty, identity pipeline).
    pub fn to_pipeline(&self) -> Pipeline {
        let mut p = Pipeline::new();
        if self.exposure_on && (self.ev != 0.0 || self.black != 0.0) {
            p.push(Stage::Exposure { black: self.black, scale: 2.0f32.powf(self.ev) });
        }
        if self.velvia_on && self.velvia_strength > 0.0 {
            p.push(Stage::Velvia { strength: self.velvia_strength / 100.0, bias: self.velvia_bias });
        }
        if self.split_on {
            p.push(Stage::Splittoning {
                shadow_hue: self.split_shadow_hue,
                shadow_sat: self.split_shadow_sat,
                highlight_hue: self.split_highlight_hue,
                highlight_sat: self.split_highlight_sat,
                balance: self.split_balance,
                compress: (self.split_compress / 110.0) / 2.0,
            });
        }
        if self.mono_on {
            p.push(Stage::Monochrome { r: self.mono_r, g: self.mono_g, b: self.mono_b });
        }
        // Sigmoid is the scene-linear → display tone map, so it runs LAST. White
        // (100%) / black (0.0152%) targets are fixed at the darktable defaults
        // (both > 0 ⇒ no NaN); only contrast & skew are user-facing here.
        if self.sigmoid_on {
            let [white_target, black_target, paper_exp, film_fog, film_power, paper_power] =
                darkroom_core::iop::sigmoid::rgb_ratio_params(
                    self.sigmoid_contrast, self.sigmoid_skew, 100.0, 0.0152,
                );
            p.push(Stage::Sigmoid {
                white_target, black_target, paper_exp, film_fog, film_power, paper_power,
            });
        }
        p
    }
}

/// Bump when the [`PreviewParams::encode`] layout changes (old blobs then decode
/// to `None` → defaults, rather than mis-parsing). v2 added the sigmoid stage.
const ENCODE_VERSION: u8 = 2;
/// 1 version byte + 5 bool bytes + 15 little-endian f32.
const ENCODED_LEN: usize = 1 + 5 + 15 * 4;

/// Run the preview pipeline over an 8-bit interleaved image buffer, preserving
/// layout (rowstride) and any alpha channel. Colour channels (0..min(3,nch))
/// are normalised to [0,1] into a packed RGBA `f32` buffer (4th channel = 1.0,
/// scratch), run through [`PreviewParams::to_pipeline`] /
/// `darkroom_core::pipeline`, then written back; the source alpha (channel 3,
/// if present) and inter-row padding are kept byte-for-byte from the input.
///
/// The colour channels are sRGB-decoded to linear before the pipeline and
/// re-encoded after, so the stages run in linear light (see the module doc).
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
    // Defensive: a malformed pixbuf (rowstride/len smaller than the geometry
    // implies) would otherwise panic on indexing user-supplied image data.
    if base.len() < (height - 1) * rowstride + width * nch {
        return base.to_vec();
    }
    // No active stage ⇒ return the source untouched (byte-exact; also avoids a
    // pointless sRGB linearise/encode round-trip that could drift ±1 LSB).
    let pipeline = params.to_pipeline();
    if pipeline.stages.is_empty() {
        return base.to_vec();
    }

    let colour = nch.min(3);
    let n = width * height;

    // ── gather colour → packed RGBA f32, LINEAR light [0,1] ────────────────
    // The 8-bit pixbuf is gamma-encoded sRGB; decode to linear so the core
    // stages (exposure, velvia, …) run in the same scene-referred-ish domain as
    // the real pixelpipe. 4th channel = 1.0 scratch (exposure scales all four;
    // we discard it on scatter and keep the real source alpha). Sources with <3
    // colour channels (greyscale) replicate their last channel.
    let mut rgba = vec![0.0f32; n * 4];
    for y in 0..height {
        let row = y * rowstride;
        for x in 0..width {
            let p = row + x * nch;
            let o = (y * width + x) * 4;
            for c in 0..3 {
                let src = c.min(colour - 1);
                rgba[o + c] = darkroom_core::color::srgb_to_linear(base[p + src] as f32 / 255.0);
            }
            rgba[o + 3] = 1.0;
        }
    }

    let processed = pipeline.process(&rgba);

    // ── scatter colour back (linear → sRGB), preserving alpha + padding ────
    let mut outbuf = base.to_vec();
    for y in 0..height {
        let row = y * rowstride;
        for x in 0..width {
            let p = row + x * nch;
            let o = (y * width + x) * 4;
            for c in 0..colour {
                let enc = darkroom_core::color::linear_to_srgb(processed[o + c]);
                outbuf[p + c] = (enc.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    outbuf
}

/// Per-channel (R, G, B) 256-bin histogram of an 8-bit interleaved image.
pub type Histogram = [[u32; 256]; 3];

/// Compute the [`Histogram`] of an 8-bit interleaved buffer, honouring
/// rowstride and channel count; alpha (channel 3) is ignored. Sources with
/// fewer than 3 colour channels (greyscale) count their last channel into all
/// three. Intended to run over the *processed* preview so the histogram tracks
/// the live pipeline output.
pub fn compute_histogram(
    buf: &[u8],
    width: usize,
    height: usize,
    rowstride: usize,
    nch: usize,
) -> Histogram {
    let mut h = [[0u32; 256]; 3];
    if nch == 0 || width == 0 || height == 0 {
        return h;
    }
    // Defensive: short/malformed buffer ⇒ empty histogram rather than a panic.
    if buf.len() < (height - 1) * rowstride + width * nch {
        return h;
    }
    let colour = nch.min(3);
    for y in 0..height {
        let row = y * rowstride;
        for x in 0..width {
            let p = row + x * nch;
            for (c, hc) in h.iter_mut().enumerate() {
                let src = c.min(colour - 1);
                hc[buf[p + src] as usize] += 1;
            }
        }
    }
    h
}

/// Map a click at widget-space `(x, y)` to an image pixel `(col, row)` for a
/// `Picture` using `ContentFit::Contain` (image scaled to fit, preserving aspect,
/// centred with letterbox/pillarbox bars). Returns `None` when the click lands on
/// the bars (outside the image) or for degenerate sizes.
pub fn map_widget_to_image(
    widget_w: f64,
    widget_h: f64,
    img_w: usize,
    img_h: usize,
    x: f64,
    y: f64,
) -> Option<(usize, usize)> {
    if img_w == 0 || img_h == 0 || widget_w <= 0.0 || widget_h <= 0.0 {
        return None;
    }
    let (iwf, ihf) = (img_w as f64, img_h as f64);
    let scale = (widget_w / iwf).min(widget_h / ihf); // Contain: fit the tighter axis
    let (disp_w, disp_h) = (iwf * scale, ihf * scale);
    let (off_x, off_y) = ((widget_w - disp_w) / 2.0, (widget_h - disp_h) / 2.0);
    let ix = (x - off_x) / scale;
    let iy = (y - off_y) / scale;
    if ix < 0.0 || iy < 0.0 || ix >= iwf || iy >= ihf {
        return None;
    }
    Some((ix as usize, iy as usize))
}

/// Read the RGB triplet at image pixel `(x, y)` from an 8-bit interleaved buffer
/// (alpha ignored). Greyscale (<3 colour channels) replicates its last channel.
/// Returns `None` if the pixel is out of range or the buffer is too short.
pub fn sample_pixel(
    buf: &[u8],
    width: usize,
    height: usize,
    rowstride: usize,
    nch: usize,
    x: usize,
    y: usize,
) -> Option<(u8, u8, u8)> {
    if nch == 0 || x >= width || y >= height {
        return None;
    }
    let colour = nch.min(3);
    let p = y * rowstride + x * nch;
    if p + colour > buf.len() {
        return None;
    }
    Some((
        buf[p],
        buf[p + 1.min(colour - 1)],
        buf[p + 2.min(colour - 1)],
    ))
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
    fn ev_plus_one_brightens_clamps_keeps_alpha() {
        // RGBA: ev=+1 doubles in LINEAR light, then re-encodes to sRGB. Each
        // colour channel brightens; a high channel clamps; alpha is untouched.
        let base = vec![50u8, 200, 25, 111];
        let out = apply_pipeline(&base, 1, 1, 4, 4, &exposure_only(0.0, 1.0));
        assert!(out[0] > 50, "R should brighten, got {}", out[0]);
        assert_eq!(out[1], 255, "200 doubled clamps"); // 0.578·2 > 1 → 255
        assert!(out[2] > 25, "B should brighten, got {}", out[2]);
        assert_eq!(out[3], 111); // alpha unchanged
    }

    #[test]
    fn black_point_darkens() {
        // ev=0 (scale 1), black=0.1 subtracted in linear ⇒ darker than input.
        let base = vec![128u8, 128, 128];
        let out = apply_pipeline(&base, 1, 1, 3, 3, &exposure_only(0.1, 0.0));
        assert!(out[0] < 128 && out[0] > 0, "darkened, got {}", out[0]);
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

        // neutral grey: velvia leaves it (very nearly) unchanged — allow ±1 LSB
        // for the sRGB linearise/encode round-trip.
        let grey = vec![128u8, 128, 128];
        let g_out = apply_pipeline(&grey, 1, 1, 3, 3, &p);
        for c in 0..3 {
            assert!((g_out[c] as i32 - 128).abs() <= 1, "grey ch{c} = {}", g_out[c]);
        }

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

    #[test]
    fn bypassed_disables_every_stage_and_returns_input() {
        let mut p = PreviewParams::default();
        p.ev = 2.0; // would brighten strongly
        p.velvia_on = true;
        p.velvia_strength = 80.0;
        p.split_on = true;
        let b = p.bypassed();
        assert!(b.is_identity());
        // non-stage params (e.g. velvia_strength) are preserved for restore
        assert_eq!(b.velvia_strength, 80.0);
        assert_eq!(b.ev, 2.0);
        let base = vec![60u8, 120, 180];
        assert_eq!(apply_pipeline(&base, 1, 1, 3, 3, &b), base);
    }

    #[test]
    fn monochrome_produces_equal_rgb_from_weighted_mix() {
        // weights (0.2,0.7,0.1) on (1.0, 0.502, 0.0):
        // gray = 0.2 + 0.7*0.502 + 0 = 0.5514 → *255 ≈ 141, R=G=B.
        let mut p = PreviewParams::default();
        p.exposure_on = false;
        p.mono_on = true;
        p.mono_r = 0.2;
        p.mono_g = 0.7;
        p.mono_b = 0.1;
        assert!(!p.is_identity());
        let base = vec![255u8, 128, 0];
        let out = apply_pipeline(&base, 1, 1, 3, 3, &p);
        // grayscale ⇒ R=G=B (the mix runs in linear, so the exact 8-bit value
        // differs from a naive gamma-space weighting; the equality is the point)
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
        assert!(out[0] > 0, "non-black gray, got {}", out[0]);
    }

    #[test]
    fn monochrome_off_is_identity() {
        let mut p = PreviewParams::default();
        p.mono_on = false;
        assert!(p.is_identity());
        let base = vec![200u8, 60, 40];
        assert_eq!(apply_pipeline(&base, 1, 1, 3, 3, &p), base);
    }

    #[test]
    fn splittoning_off_is_identity() {
        let mut p = PreviewParams::default();
        p.split_on = false;
        p.split_shadow_sat = 1.0;
        assert!(p.is_identity());
        let base = vec![200u8, 60, 40];
        assert_eq!(apply_pipeline(&base, 1, 1, 3, 3, &p), base);
    }

    #[test]
    fn histogram_counts_per_channel_and_respects_rowstride() {
        // 2x1 RGB, 2 pad bytes at row end: pixels (10,20,30) and (10,40,30).
        let buf = vec![10u8, 20, 30, 10, 40, 30, 0xFF, 0xFF];
        let h = compute_histogram(&buf, 2, 1, 8, 3);
        assert_eq!(h[0][10], 2); // both reds = 10
        assert_eq!(h[1][20], 1);
        assert_eq!(h[1][40], 1);
        assert_eq!(h[2][30], 2); // both blues = 30
        // total counts per channel == pixel count, padding not counted
        for c in 0..3 {
            assert_eq!(h[c].iter().sum::<u32>(), 2);
        }
        assert_eq!(h[0][0xFF], 0); // the pad byte must not be counted
    }

    #[test]
    fn histogram_degenerate_input_is_empty() {
        assert_eq!(compute_histogram(&[], 0, 0, 0, 0), [[0u32; 256]; 3]);
    }

    #[test]
    fn is_identity_matches_empty_pipeline() {
        // The two "is this a no-op" sources of truth (is_identity gates whether
        // the caller re-uploads; to_pipeline gates what runs) must agree, or a
        // visible edit could be skipped. Check across representative configs.
        let mut cfgs = vec![PreviewParams::default(), PreviewParams::default().bypassed()];
        let mut on = PreviewParams::default();
        on.ev = 0.5;
        cfgs.push(on);
        let mut v = PreviewParams::default();
        v.velvia_on = true;
        v.velvia_strength = 10.0;
        cfgs.push(v);
        let mut s = PreviewParams::default();
        s.split_on = true;
        cfgs.push(s);
        let mut m = PreviewParams::default();
        m.mono_on = true;
        cfgs.push(m);
        let mut sig = PreviewParams::default();
        sig.sigmoid_on = true;
        cfgs.push(sig);
        // velvia enabled but strength 0 ⇒ still identity (matches to_pipeline gate)
        let mut v0 = PreviewParams::default();
        v0.velvia_on = true;
        v0.velvia_strength = 0.0;
        cfgs.push(v0);
        for c in cfgs {
            assert_eq!(
                c.is_identity(),
                c.to_pipeline().stages.is_empty(),
                "is_identity vs empty-pipeline disagree for {c:?}"
            );
        }
    }

    #[test]
    fn params_encode_decode_roundtrips() {
        let p = PreviewParams {
            exposure_on: true, black: 0.05, ev: -1.25,
            velvia_on: true, velvia_strength: 42.0, velvia_bias: 0.75,
            split_on: true, split_shadow_hue: 0.1, split_shadow_sat: 0.6,
            split_highlight_hue: 0.9, split_highlight_sat: 0.3,
            split_balance: 0.4, split_compress: 60.0,
            mono_on: true, mono_r: -0.2, mono_g: 1.5, mono_b: 0.33,
            sigmoid_on: true, sigmoid_contrast: 1.8, sigmoid_skew: -0.3,
        };
        let blob = p.encode();
        assert_eq!(blob.len(), 1 + 5 + 15 * 4);
        assert_eq!(PreviewParams::decode(&blob), Some(p));
        // default round-trips too
        let d = PreviewParams::default();
        assert_eq!(PreviewParams::decode(&d.encode()), Some(d));
    }

    #[test]
    fn decode_rejects_bad_version_and_length() {
        let mut blob = PreviewParams::default().encode();
        // wrong length
        assert_eq!(PreviewParams::decode(&blob[..blob.len() - 1]), None);
        assert_eq!(PreviewParams::decode(&[]), None);
        // wrong (future) version byte
        blob[0] = 99;
        assert_eq!(PreviewParams::decode(&blob), None);
        // an old v1 blob (version 1, 57 bytes) must be rejected → caller defaults,
        // never misread as v2 (lengths differ: 57 vs 66).
        let v1 = {
            let mut b = vec![0u8; 1 + 4 + 13 * 4];
            b[0] = 1;
            b
        };
        assert_eq!(PreviewParams::decode(&v1), None);
    }

    #[test]
    fn widget_to_image_maps_centre_and_letterbox() {
        // 200x100 widget, 100x100 image → Contain scale 1.0, pillarbox: image
        // occupies x∈[50,150), full height. Centre of widget → image (50,50).
        assert_eq!(map_widget_to_image(200.0, 100.0, 100, 100, 100.0, 50.0), Some((50, 50)));
        // left bar (x=10) is outside the image
        assert_eq!(map_widget_to_image(200.0, 100.0, 100, 100, 10.0, 50.0), None);
        // just inside the left image edge (x=50) → col 0
        assert_eq!(map_widget_to_image(200.0, 100.0, 100, 100, 50.0, 50.0), Some((0, 50)));
        // a 2x downscale: 100x100 widget, 200x200 image → scale 0.5, no bars
        assert_eq!(map_widget_to_image(100.0, 100.0, 200, 200, 10.0, 20.0), Some((20, 40)));
        // degenerate
        assert_eq!(map_widget_to_image(0.0, 100.0, 100, 100, 1.0, 1.0), None);
        assert_eq!(map_widget_to_image(100.0, 100.0, 0, 0, 1.0, 1.0), None);
    }

    #[test]
    fn sample_pixel_reads_rgb_and_bounds_check() {
        // 2x1 RGB with row padding
        let buf = vec![10u8, 20, 30, 40, 50, 60, 0xFF, 0xFF];
        assert_eq!(sample_pixel(&buf, 2, 1, 8, 3, 0, 0), Some((10, 20, 30)));
        assert_eq!(sample_pixel(&buf, 2, 1, 8, 3, 1, 0), Some((40, 50, 60)));
        assert_eq!(sample_pixel(&buf, 2, 1, 8, 3, 2, 0), None); // x out of range
        // greyscale replicates the single channel
        let g = vec![77u8, 88];
        assert_eq!(sample_pixel(&g, 2, 1, 2, 1, 0, 0), Some((77, 77, 77)));
    }

    #[test]
    fn short_buffer_no_ops_without_panicking() {
        // Geometry claims 2x2 RGB (needs 12 bytes) but only 3 are supplied.
        let buf = vec![10u8, 20, 30];
        assert_eq!(compute_histogram(&buf, 2, 2, 6, 3), [[0u32; 256]; 3]);
        let p = PreviewParams { ev: 1.0, ..PreviewParams::default() };
        assert_eq!(apply_pipeline(&buf, 2, 2, 6, 3, &p), buf);
    }

    #[test]
    fn splittoning_tones_a_dark_pixel_toward_shadow_hue() {
        // Dark, near-neutral pixel in the shadow zone: a red shadow hue (0.0)
        // at full saturation must push red above blue.
        let mut p = PreviewParams::default();
        p.exposure_on = false; // isolate the split-toning stage
        p.split_on = true;
        p.split_shadow_hue = 0.0; // red
        p.split_shadow_sat = 1.0;
        p.split_balance = 0.5;
        p.split_compress = 0.0; // widest toning range
        assert!(!p.is_identity());
        let base = vec![30u8, 30, 30];
        let out = apply_pipeline(&base, 1, 1, 3, 3, &p);
        assert!(out[0] > out[2], "red {} should exceed blue {}", out[0], out[2]);
    }
}
