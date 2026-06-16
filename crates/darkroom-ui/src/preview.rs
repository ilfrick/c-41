//! Live preview pipeline: applies an ordered chain of migrated `darkroom-core`
//! IOPs to the decoded 8-bit preview image so the darkroom view shows
//! *processed* output (not just the file). This is the UI↔core processing seam
//! — a stepping-stone toward a full Rust pixelpipe (RUST_MIGRATION_PLAN.md
//! Phase 3 milestone 2).
//!
//! Phase 3-ui-13+ generalises the original single-IOP (exposure) seam into a
//! small pipeline of stages, each backed by a migrated core IOP that works in
//! RGB [0,1], in pixelpipe order:
//!   1. **exposure**    — `out = (in - black) * scale`, `scale = 2^ev`
//!   2. **velvia**      — saturation-weighted chroma boost
//!   3. **splittoning** — hue toning of shadows / highlights
//!
//! All run on the colour channels (0..min(3,nch)); any alpha channel and the
//! rowstride padding are preserved byte-for-byte.

use darkroom_core::iop::{channelmixer, exposure, splittoning, velvia};

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
        exp_identity && vel_identity && split_identity && mono_identity
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
            ..*self
        }
    }

    /// Serialise to a compact, versioned little-endian blob for DB persistence:
    /// `[version, 4×bool(u8), 13×f32_le]`. Decoded by [`PreviewParams::decode`].
    /// This is darkroom-ui's own layout (NOT a C IOP `op_params`), stored under a
    /// synthetic operation name the C reader ignores.
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(ENCODED_LEN);
        v.push(ENCODE_VERSION);
        for b in [self.exposure_on, self.velvia_on, self.split_on, self.mono_on] {
            v.push(b as u8);
        }
        for f in [
            self.black, self.ev,
            self.velvia_strength, self.velvia_bias,
            self.split_shadow_hue, self.split_shadow_sat,
            self.split_highlight_hue, self.split_highlight_sat,
            self.split_balance, self.split_compress,
            self.mono_r, self.mono_g, self.mono_b,
        ] {
            v.extend_from_slice(&f.to_le_bytes());
        }
        v
    }

    /// Inverse of [`PreviewParams::encode`]. Returns `None` for the wrong version
    /// byte or wrong length (e.g. a blob written by a future/other schema), so
    /// the caller falls back to defaults rather than loading garbage.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != ENCODED_LEN || bytes[0] != ENCODE_VERSION {
            return None;
        }
        let bools = &bytes[1..5];
        // length is checked above, so exactly 13 f32 chunks follow
        let f: Vec<f32> = bytes[5..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Self {
            exposure_on: bools[0] != 0,
            velvia_on: bools[1] != 0,
            split_on: bools[2] != 0,
            mono_on: bools[3] != 0,
            black: f[0], ev: f[1],
            velvia_strength: f[2], velvia_bias: f[3],
            split_shadow_hue: f[4], split_shadow_sat: f[5],
            split_highlight_hue: f[6], split_highlight_sat: f[7],
            split_balance: f[8], split_compress: f[9],
            mono_r: f[10], mono_g: f[11], mono_b: f[12],
        })
    }
}

/// Bump when the [`PreviewParams::encode`] layout changes (old blobs then decode
/// to `None` → defaults, rather than mis-parsing).
const ENCODE_VERSION: u8 = 1;
/// 1 version byte + 4 bool bytes + 13 little-endian f32.
const ENCODED_LEN: usize = 1 + 4 + 13 * 4;

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
    // Defensive: a malformed pixbuf (rowstride/len smaller than the geometry
    // implies) would otherwise panic on indexing user-supplied image data.
    if base.len() < (height - 1) * rowstride + width * nch {
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

    // ── velvia stage (RGBA core loop; see run_rgba_stage) ──────────────────
    if params.velvia_on && params.velvia_strength > 0.0 {
        let (strength, bias) = (params.velvia_strength / 100.0, params.velvia_bias);
        run_rgba_stage(&mut rgb, n, |inp, out| {
            velvia::process_pixels(inp, out, strength, bias)
        });
    }

    // ── split-toning stage (RGBA core loop) ────────────────────────────────
    if params.split_on {
        // The C UI compress slider (0..100) is pre-scaled by commit_params.
        let compress = (params.split_compress / 110.0) / 2.0;
        let (sh, ss) = (params.split_shadow_hue, params.split_shadow_sat);
        let (hh, hs) = (params.split_highlight_hue, params.split_highlight_sat);
        let bal = params.split_balance;
        run_rgba_stage(&mut rgb, n, |inp, out| {
            splittoning::process_pixels(inp, out, sh, ss, hh, hs, bal, compress)
        });
    }

    // ── monochrome stage (channelmixer GRAY mode; RGBA core loop) ──────────
    if params.mono_on {
        // GRAY mode reads only row 0 of rgb_matrix (the R,G,B → gray weights);
        // the hsl_matrix is unused. operation_mode 1 = OPERATION_MODE_GRAY.
        let rgb_matrix = [params.mono_r, params.mono_g, params.mono_b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hsl_matrix = [0.0f32; 9];
        run_rgba_stage(&mut rgb, n, |inp, out| {
            channelmixer::process_pixels(inp, out, &hsl_matrix, &rgb_matrix, 1)
        });
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

/// Run a 4-channel (RGBA) core IOP over the packed 3-wide RGB buffer in place:
/// repack to RGBA with alpha = 1.0 (these stages only pass alpha through, so the
/// value is irrelevant and the result alpha is discarded), run `process`, and
/// copy the RGB channels back. Used for IOPs whose core loop is
/// `chunks_exact(4)` (velvia, splittoning).
fn run_rgba_stage(rgb: &mut [f32], n: usize, process: impl Fn(&[f32], &mut [f32])) {
    let mut rgba = vec![0.0f32; n * 4];
    for i in 0..n {
        rgba[i * 4] = rgb[i * 3];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 1.0;
    }
    let mut out = vec![0.0f32; rgba.len()];
    process(&rgba, &mut out);
    for i in 0..n {
        rgb[i * 3] = out[i * 4];
        rgb[i * 3 + 1] = out[i * 4 + 1];
        rgb[i * 3 + 2] = out[i * 4 + 2];
    }
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
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
        assert!((out[0] as i32 - 141).abs() <= 1, "gray ~141, got {}", out[0]);
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
    fn params_encode_decode_roundtrips() {
        let p = PreviewParams {
            exposure_on: true, black: 0.05, ev: -1.25,
            velvia_on: true, velvia_strength: 42.0, velvia_bias: 0.75,
            split_on: true, split_shadow_hue: 0.1, split_shadow_sat: 0.6,
            split_highlight_hue: 0.9, split_highlight_sat: 0.3,
            split_balance: 0.4, split_compress: 60.0,
            mono_on: true, mono_r: -0.2, mono_g: 1.5, mono_b: 0.33,
        };
        let blob = p.encode();
        assert_eq!(blob.len(), 1 + 4 + 13 * 4);
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
        // wrong version byte
        blob[0] = 2;
        assert_eq!(PreviewParams::decode(&blob), None);
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
