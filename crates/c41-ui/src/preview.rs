//! Live preview pipeline: applies an ordered chain of migrated `c41-core`
//! IOPs to the decoded 8-bit preview image so the darkroom view shows
//! *processed* output (not just the file). This is the UI↔core processing seam
//! — a stepping-stone toward a full Rust pixelpipe (RUST_MIGRATION_PLAN.md
//! Phase 3 milestone 2).
//!
//! Phase 3-m2-2: the stage chaining now lives in `c41_core::pipeline`.
//! [`PreviewParams::to_pipeline`] maps the UI sliders (UI ranges) to a
//! `Pipeline` of physical-param `Stage`s in darktable's canonical iop order
//! (invert → temperature → exposure → monochrome → sharpen → vibrance → color
//! correction → color contrast → color zones → sigmoid → levels → velvia →
//! colorize → splittoning);
//! [`apply_pipeline`] just marshals the 8-bit pixbuf to/from the
//! float RGBA the core pipeline runs on, preserving the source alpha channel and
//! rowstride padding byte-for-byte.
//!
//! The 8-bit sRGB pixbuf is **decoded to linear light** (sRGB EOTF) on the way
//! in and re-encoded on the way out, so the core stages run in linear — the
//! same domain as the real pixelpipe. The remaining gap is that the input is a
//! display-referred 8-bit image, not true scene-referred raw: a stepping-stone
//! until a raw-decode/demosaic front end feeds `core::pipeline` directly.

use c41_core::iop::colorzones;
use c41_core::pipeline::{ColorSpace, Pipeline, Stage};

/// Live, user-tunable parameters for the preview pipeline. Each enabled stage
/// runs the corresponding migrated `c41-core` IOP, in pixelpipe order
/// (invert → temperature → exposure → monochrome → sharpen → vibrance → color
/// correction → color contrast → color zones → sigmoid → levels → velvia →
/// colorize → splittoning; see
/// [`Self::to_pipeline`]).
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
    /// Sharpen (unsharp mask) stage on/off.
    pub sharpen_on: bool,
    /// Gaussian radius on the C 0..100 slider (0 = no-op, 100 = max radius).
    pub sharpen_radius: f32,
    /// Sharpening strength on the C 0..5 slider (0 = no-op, 1 = default, 5 = extreme).
    pub sharpen_amount: f32,
    /// Detail threshold in Lab-L units [0, 100] (darktable default 0.5). 0 = sharpen all.
    pub sharpen_threshold: f32,
    /// Vibrance (saturation-weighted chroma boost) stage on/off.
    pub vibrance_on: bool,
    /// Vibrance strength on the C 0..100 slider (already ×0.01 by encode/decode;
    /// `to_pipeline` passes it divided by 100 to the core).
    pub vibrance_amount: f32,
    /// Color contrast (green-magenta / blue-yellow) stage on/off.
    pub color_contrast_on: bool,
    /// Green-magenta contrast steepness [0, 5] (default 1.0 = no-op).
    pub color_contrast_a_steepness: f32,
    /// Blue-yellow contrast steepness [0, 5] (default 1.0 = no-op).
    pub color_contrast_b_steepness: f32,
    /// Invert (film-camera negative) stage on/off.
    pub invert_on: bool,
    /// Per-channel film-back colour: out = color - in (default 1.0 = negate).
    pub invert_r: f32,
    pub invert_g: f32,
    pub invert_b: f32,
    /// Temperature (white balance) stage on/off.
    pub temperature_on: bool,
    /// Per-channel multipliers (default 1.0 = no change). 0..=4 range.
    pub temperature_r: f32,
    pub temperature_g: f32,
    pub temperature_b: f32,
    /// Colorize (HSL colour replacement) stage on/off.
    pub colorize_on: bool,
    /// Colour hue, 0..1 (normalised angle on the colour wheel).
    pub colorize_hue: f32,
    /// Colour saturation, 0..1.
    pub colorize_sat: f32,
    /// Colour lightness, 0..100 (darktable slider scale).
    pub colorize_lightness: f32,
    /// Source lightness mix, 0..100 (darktable slider scale; core gets ×0.01).
    pub colorize_lightness_mix: f32,
    /// Color correction (Lab a/b scaling) stage on/off.
    pub color_correction_on: bool,
    /// Shadow a-channel offset (loa) — additive on Lab a for shadows.
    pub color_correction_loa: f32,
    /// Highlight a-channel offset (hia) — additive on Lab a for highlights.
    pub color_correction_hia: f32,
    /// Shadow b-channel offset (lob).
    pub color_correction_lob: f32,
    /// Highlight b-channel offset (hib).
    pub color_correction_hib: f32,
    /// Global chroma saturation (-3..3, default 1.0 = no change).
    pub color_correction_saturation: f32,
    /// Color zones (LCH equaliser) stage on/off.
    pub colorzones_on: bool,
    /// Strength applied to curve y values: `y' = y + (y - 0.5) * (strength/100)`.
    pub colorzones_strength: f32,
    /// Select-by channel: 0 = L, 1 = C, 2 = h.
    pub colorzones_channel: f32,
    /// Process mode: 0 = smooth (v3), 1 = strong (v1).
    pub colorzones_mode: f32,
    /// Number of nodes per channel [L, C, h]; each ≤ 8.
    pub colorzones_num_nodes: [f32; 3],
    /// Spline type per channel: 0 = CUBIC, 1 = CATMULL_ROM, 2 = MONOTONE.
    pub colorzones_curve_type: [f32; 3],
    /// Curve x-coordinates: 3 channels × 8 nodes per channel.
    pub colorzones_curve_x: [[f32; 8]; 3],
    /// Curve y-coordinates: 3 channels × 8 nodes per channel.
    pub colorzones_curve_y: [[f32; 8]; 3],
    /// Levels (black / grey / white point + gamma) stage on/off.
    pub levels_on: bool,
    /// Black point on the darktable 0..100 slider (core gets it /100).
    pub levels_black: f32,
    /// Grey (midtone) point, 0..100. Centred between black and white ⇒ gamma 1.
    pub levels_grey: f32,
    /// White point, 0..100.
    pub levels_white: f32,
    /// Vignetting stage on/off.
    pub vignette_on: bool,
    /// Fall-off start (inner radius), 0..200 % of the largest image dimension.
    pub vignette_scale: f32,
    /// Fall-off radius, 0..200. Outer radius = inner + this.
    pub vignette_falloff: f32,
    /// Brightness reduction strength, -1..1 (darktable default -0.5).
    pub vignette_brightness: f32,
    /// Saturation reduction strength, -1..1 (darktable default -0.5).
    pub vignette_saturation: f32,
    /// Vignette centre offset, -1..1 in each axis (0 = image centre).
    pub vignette_center_x: f32,
    pub vignette_center_y: f32,
    /// Shape exponent, 0..5 (1 = ellipse; higher = squarer).
    pub vignette_shape: f32,
    /// Lowlight (scotopic "night vision") stage on/off.
    pub lowlight_on: bool,
    /// Blue shift applied to the scotopic response, 0..100.
    pub lowlight_blueness: f32,
    /// Transition-curve band heights, 6 nodes at x = k/5, each 0..1.
    /// 0.5 everywhere = an even scotopic/photopic blend at every luminance.
    pub lowlight_transition: [f32; 6],
    /// Graduated ND stage on/off.
    pub gradnd_on: bool,
    /// Filter density in EV, -8..8. Negative brightens instead of darkening.
    pub gradnd_density: f32,
    /// Edge hardness, 0..100 (0 = soft gradient, 100 = hard line).
    pub gradnd_hardness: f32,
    /// Rotation of the gradient line in degrees, -180..180.
    pub gradnd_rotation: f32,
    /// Line offset across the frame, 0..100 (50 = centred).
    pub gradnd_offset: f32,
    /// Filter tint; saturation 0 = a neutral ND filter.
    pub gradnd_hue: f32,
    pub gradnd_saturation: f32,
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
            sharpen_on: false,
            sharpen_radius: 2.0,
            sharpen_amount: 0.5,
            sharpen_threshold: 0.5,
            // Darktable default: off. 0..100 slider (amount).
            vibrance_on: false,
            vibrance_amount: 0.0,
            // Darktable default: off. Steepness 1.0 = identity (no contrast change).
            color_contrast_on: false,
            color_contrast_a_steepness: 1.0,
            color_contrast_b_steepness: 1.0,
            // Darktable default: off, unity WB multipliers.
            temperature_on: false,
            temperature_r: 1.0,
            temperature_g: 1.0,
            temperature_b: 1.0,
            // Darktable default: off, film-back colour (1,1,1,1) = standard negate.
            invert_on: false,
            invert_r: 1.0,
            invert_g: 1.0,
            invert_b: 1.0,
            // Darktable default: off. Hue 0, sat 0, lightness 50, mix 50.
            colorize_on: false,
            colorize_hue: 0.0,
            colorize_sat: 0.0,
            colorize_lightness: 50.0,
            colorize_lightness_mix: 50.0,
            // Darktable default: off, saturation 1.0 (no change), offsets 0.
            color_correction_on: false,
            color_correction_loa: 0.0,
            color_correction_hia: 0.0,
            color_correction_lob: 0.0,
            color_correction_hib: 0.0,
            color_correction_saturation: 1.0,
            // ColorZones defaults from _reset_parameters (colorzones.c:735):
            // channel=h, mode=SMOOTH, 2 nodes per channel at x=0.25,0.75, y=0.5.
            colorzones_on: false,
            colorzones_strength: 0.0,
            colorzones_channel: 2.0, // h
            colorzones_mode: 0.0,    // SMOOTH
            colorzones_num_nodes: [2.0, 2.0, 2.0],
            colorzones_curve_type: [1.0, 1.0, 1.0], // CATMULL_ROM
            colorzones_curve_x: [
                [0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            colorzones_curve_y: [[0.5; 8]; 3],
            // Darktable defaults (levels.c): off, black 0 / grey 50 / white 100
            // ⇒ grey exactly centred ⇒ gamma 1 ⇒ identity tone curve.
            levels_on: false,
            levels_black: 0.0,
            levels_grey: 50.0,
            levels_white: 100.0,
            // darktable defaults (vignette.c params struct): off, fall-off
            // start 80, radius 50, brightness/saturation -0.5, centred,
            // shape 1, automatic ratio (so whratio is not user-facing here).
            vignette_on: false,
            vignette_scale: 80.0,
            vignette_falloff: 50.0,
            vignette_brightness: -0.5,
            vignette_saturation: -0.5,
            vignette_center_x: 0.0,
            vignette_center_y: 0.0,
            vignette_shape: 1.0,
            // darktable defaults (lowlight.c init): off, no blue shift, all six
            // transition bands at 0.5.
            lowlight_on: false,
            lowlight_blueness: 0.0,
            lowlight_transition: [0.5; 6],
            // darktable defaults (graduatednd.c params): off, 1 EV, soft edge,
            // horizontal, centred, neutral (saturation 0).
            gradnd_on: false,
            gradnd_density: 1.0,
            gradnd_hardness: 0.0,
            gradnd_rotation: 0.0,
            gradnd_offset: 50.0,
            gradnd_hue: 0.0,
            gradnd_saturation: 0.0,
        }
    }
}

impl PreviewParams {
    /// Whether a `Stage::Levels` will actually be emitted. Single source of
    /// truth for [`Self::is_identity`] and [`Self::to_pipeline`] — they must
    /// agree, or `is_identity` claims an edit that the pipeline never applies.
    ///
    /// Off when disabled, at the darktable defaults (0/50/100 — the identity
    /// curve), or with a degenerate range (`white <= black`), where the C
    /// `(white-black)/2` divisor would blow up.
    fn levels_stage_active(&self) -> bool {
        self.levels_on
            && self.levels_white - self.levels_black >= LEVELS_MIN_RANGE
            && !(self.levels_black == 0.0
                && self.levels_grey == 50.0
                && self.levels_white == 100.0)
    }

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
        let sharpen_identity = !self.sharpen_on || self.sharpen_amount <= 0.0 || self.sharpen_radius <= 0.0;
        let vibrance_identity = !self.vibrance_on || self.vibrance_amount <= 0.0;
        // Color contrast (a_steepness == 1.0 && b_steepness == 1.0) is an affine
        // no-op on Lab a/b; any deviation from 1.0 changes chroma.
        let cc_identity = !self.color_contrast_on
            || (self.color_contrast_a_steepness == 1.0 && self.color_contrast_b_steepness == 1.0);
        let temp_identity = !self.temperature_on
            || (self.temperature_r == 1.0 && self.temperature_g == 1.0 && self.temperature_b == 1.0);
        // Invert has no value at which it is a no-op while enabled (even
        // color == (1,1,1,1) negates the image), so on == off.
        let invert_identity = !self.invert_on;
        // Colorize with sat 0 produces a grey (zero chroma) — that's a real
        // change (replacing a/b), so on == off is the only no-op guard.
        let colorize_identity = !self.colorize_on;
        // Color correction: saturation 1.0 with all offsets 0 is identity
        // (a_base=0, b_base=0, scales=0 ⇒ out.a = in.a, out.b = in.b).
        let cc_corr_identity = !self.color_correction_on
            || (self.color_correction_saturation == 1.0
                && self.color_correction_loa == 0.0
                && self.color_correction_hia == 0.0
                && self.color_correction_lob == 0.0
                && self.color_correction_hib == 0.0);
        // ColorZones: the Lab→LCH→LUT→Lab round-trip is never a strict no-op
        // while enabled (even with all-neutral LUTs, float rounding differs),
        // so on == off is the only no-op guard.
        let cz_identity = !self.colorzones_on;
        // Shares one predicate with `to_pipeline` so the two can never disagree
        // about whether a Levels stage exists (an inverted range is non-default
        // but still emits nothing).
        let levels_identity = !self.levels_stage_active();
        // Vignette with both strengths at 0 leaves every pixel untouched (the
        // weight still varies, but it scales nothing); otherwise it always
        // changes the image, so on == off is the only other no-op.
        // Lowlight always blends toward the scotopic response while enabled —
        // even a flat 0.5 curve is a real 50% mix — so on == off is the only
        // no-op guard.
        let lowlight_identity = !self.lowlight_on;
        // Density 0 makes the filter exp2(0) = 1 everywhere — a true no-op.
        let gradnd_identity = !self.gradnd_on || self.gradnd_density == 0.0;
        let vignette_identity = !self.vignette_on
            || (self.vignette_brightness == 0.0 && self.vignette_saturation == 0.0);
        exp_identity && vel_identity && split_identity && mono_identity && sigmoid_identity
            && sharpen_identity && vibrance_identity && cc_identity && temp_identity
            && invert_identity && colorize_identity && cc_corr_identity && cz_identity
            && levels_identity && vignette_identity && lowlight_identity
            && gradnd_identity
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
            sharpen_on: false,
            vibrance_on: false,
            color_contrast_on: false,
            temperature_on: false,
            invert_on: false,
            colorize_on: false,
            color_correction_on: false,
            colorzones_on: false,
            levels_on: false,
            vignette_on: false,
            lowlight_on: false,
            gradnd_on: false,
            ..*self
        }
    }

    /// Serialise to a compact, versioned little-endian blob for DB persistence:
    /// `[version, 12×bool(u8), 36×f32_le]`. Decoded by [`PreviewParams::decode`].
    /// This is c41-ui's own layout (NOT a C IOP `op_params`), stored under a
    /// synthetic operation name the C reader ignores.
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(ENCODED_LEN);
        v.push(ENCODE_VERSION);
        for b in [self.exposure_on, self.velvia_on, self.split_on, self.mono_on, self.sigmoid_on, self.sharpen_on, self.vibrance_on, self.color_contrast_on, self.temperature_on, self.invert_on, self.colorize_on, self.color_correction_on, self.colorzones_on, self.levels_on, self.vignette_on, self.lowlight_on, self.gradnd_on] {
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
            self.sharpen_radius, self.sharpen_amount, self.sharpen_threshold,
            self.vibrance_amount,
            self.color_contrast_a_steepness, self.color_contrast_b_steepness,
            self.temperature_r, self.temperature_g, self.temperature_b,
            self.invert_r, self.invert_g, self.invert_b,
            self.colorize_hue, self.colorize_sat, self.colorize_lightness, self.colorize_lightness_mix,
            self.color_correction_loa, self.color_correction_hia, self.color_correction_lob, self.color_correction_hib, self.color_correction_saturation,
            self.colorzones_strength, self.colorzones_channel, self.colorzones_mode,
            self.colorzones_num_nodes[0], self.colorzones_num_nodes[1], self.colorzones_num_nodes[2],
            self.colorzones_curve_type[0], self.colorzones_curve_type[1], self.colorzones_curve_type[2],
            self.colorzones_curve_x[0][0], self.colorzones_curve_x[0][1], self.colorzones_curve_x[0][2], self.colorzones_curve_x[0][3], self.colorzones_curve_x[0][4], self.colorzones_curve_x[0][5], self.colorzones_curve_x[0][6], self.colorzones_curve_x[0][7],
            self.colorzones_curve_x[1][0], self.colorzones_curve_x[1][1], self.colorzones_curve_x[1][2], self.colorzones_curve_x[1][3], self.colorzones_curve_x[1][4], self.colorzones_curve_x[1][5], self.colorzones_curve_x[1][6], self.colorzones_curve_x[1][7],
            self.colorzones_curve_x[2][0], self.colorzones_curve_x[2][1], self.colorzones_curve_x[2][2], self.colorzones_curve_x[2][3], self.colorzones_curve_x[2][4], self.colorzones_curve_x[2][5], self.colorzones_curve_x[2][6], self.colorzones_curve_x[2][7],
            self.colorzones_curve_y[0][0], self.colorzones_curve_y[0][1], self.colorzones_curve_y[0][2], self.colorzones_curve_y[0][3], self.colorzones_curve_y[0][4], self.colorzones_curve_y[0][5], self.colorzones_curve_y[0][6], self.colorzones_curve_y[0][7],
            self.colorzones_curve_y[1][0], self.colorzones_curve_y[1][1], self.colorzones_curve_y[1][2], self.colorzones_curve_y[1][3], self.colorzones_curve_y[1][4], self.colorzones_curve_y[1][5], self.colorzones_curve_y[1][6], self.colorzones_curve_y[1][7],
            self.colorzones_curve_y[2][0], self.colorzones_curve_y[2][1], self.colorzones_curve_y[2][2], self.colorzones_curve_y[2][3], self.colorzones_curve_y[2][4], self.colorzones_curve_y[2][5], self.colorzones_curve_y[2][6], self.colorzones_curve_y[2][7],
            self.levels_black, self.levels_grey, self.levels_white,
            self.vignette_scale, self.vignette_falloff,
            self.vignette_brightness, self.vignette_saturation,
            self.vignette_center_x, self.vignette_center_y, self.vignette_shape,
            self.lowlight_blueness,
            self.lowlight_transition[0], self.lowlight_transition[1], self.lowlight_transition[2],
            self.lowlight_transition[3], self.lowlight_transition[4], self.lowlight_transition[5],
            self.gradnd_density, self.gradnd_hardness, self.gradnd_rotation,
            self.gradnd_offset, self.gradnd_hue, self.gradnd_saturation,
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
        let bools = &bytes[1..18];
        // length is checked above, so exactly 116 f32 chunks follow
        let f: Vec<f32> = bytes[18..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Self {
            exposure_on: bools[0] != 0,
            velvia_on: bools[1] != 0,
            split_on: bools[2] != 0,
            mono_on: bools[3] != 0,
            sigmoid_on: bools[4] != 0,
            sharpen_on: bools[5] != 0,
            vibrance_on: bools[6] != 0,
            color_contrast_on: bools[7] != 0,
            temperature_on: bools[8] != 0,
            invert_on: bools[9] != 0,
            colorize_on: bools[10] != 0,
            color_correction_on: bools[11] != 0,
            colorzones_on: bools[12] != 0,
            black: f[0], ev: f[1],
            velvia_strength: f[2], velvia_bias: f[3],
            split_shadow_hue: f[4], split_shadow_sat: f[5],
            split_highlight_hue: f[6], split_highlight_sat: f[7],
            split_balance: f[8], split_compress: f[9],
            mono_r: f[10], mono_g: f[11], mono_b: f[12],
            sigmoid_contrast: f[13], sigmoid_skew: f[14],
            sharpen_radius: f[15], sharpen_amount: f[16], sharpen_threshold: f[17],
            vibrance_amount: f[18],
            color_contrast_a_steepness: f[19], color_contrast_b_steepness: f[20],
            temperature_r: f[21], temperature_g: f[22], temperature_b: f[23],
            invert_r: f[24], invert_g: f[25], invert_b: f[26],
            colorize_hue: f[27], colorize_sat: f[28], colorize_lightness: f[29], colorize_lightness_mix: f[30],
            color_correction_loa: f[31], color_correction_hia: f[32], color_correction_lob: f[33], color_correction_hib: f[34], color_correction_saturation: f[35],
            colorzones_strength: f[36], colorzones_channel: f[37], colorzones_mode: f[38],
            colorzones_num_nodes: [f[39], f[40], f[41]],
            colorzones_curve_type: [f[42], f[43], f[44]],
            colorzones_curve_x: [
                [f[45], f[46], f[47], f[48], f[49], f[50], f[51], f[52]],
                [f[53], f[54], f[55], f[56], f[57], f[58], f[59], f[60]],
                [f[61], f[62], f[63], f[64], f[65], f[66], f[67], f[68]],
            ],
            colorzones_curve_y: [
                [f[69], f[70], f[71], f[72], f[73], f[74], f[75], f[76]],
                [f[77], f[78], f[79], f[80], f[81], f[82], f[83], f[84]],
                [f[85], f[86], f[87], f[88], f[89], f[90], f[91], f[92]],
            ],
            levels_on: bools[13] != 0,
            levels_black: f[93], levels_grey: f[94], levels_white: f[95],
            vignette_on: bools[14] != 0,
            vignette_scale: f[96], vignette_falloff: f[97],
            vignette_brightness: f[98], vignette_saturation: f[99],
            vignette_center_x: f[100], vignette_center_y: f[101], vignette_shape: f[102],
            lowlight_on: bools[15] != 0,
            lowlight_blueness: f[103],
            lowlight_transition: [f[104], f[105], f[106], f[107], f[108], f[109]],
            gradnd_on: bools[16] != 0,
            gradnd_density: f[110], gradnd_hardness: f[111], gradnd_rotation: f[112],
            gradnd_offset: f[113], gradnd_hue: f[114], gradnd_saturation: f[115],
        })
    }

    /// Map the UI params to a `c41_core::pipeline::Pipeline`, converting UI
    /// ranges to the physical params the core stages expect (EV→scale, velvia
    /// strength /100, split compress (c/110)/2) and including only the enabled
    /// stages that would actually change the image (so a bypassed/neutral set
    /// yields an empty, identity pipeline).
    ///
    /// Stage order follows darktable's canonical v3.0 iop order
    /// (`src/common/iop_order.c`: invert 2 → temperature 3 → exposure 21 →
    /// channelmixerrgb 39 → sharpen 53 → colorcorrection 55 → colorcontrast 56 →
    /// colorzones 60 → sigmoid 45.3 → levels 49 → velvia 57 → colorize 62 →
    /// splittoning 67). The scene-referred stages
    /// (exposure, the channel-mix to grey) run on unbounded linear data; sigmoid
    /// tone-maps to display range; the display-referred creative stages (levels,
    /// velvia, splittoning) run after it, where their [0,1] clamps are semantically
    /// correct — running them before the tone map would hard-clip scene-linear
    /// highlights (>1.0) that sigmoid is meant to roll off.
    ///
    /// Note the grey conversion here ports the *legacy* `channelmixer`, which in
    /// darktable ran display-referred (~pos 65); we place it at the scene-referred
    /// `channelmixerrgb` position 39 because linear luminance is the correct
    /// domain to tone-map as luminance. A visible consequence of the reorder: with
    /// `mono_on && split_on`, splittoning now tints the tone-mapped B&W image (the
    /// intended split-toned-monochrome result). In the old order mono ran last and
    /// silently discarded whatever velvia/splittoning had done.
    ///
    /// This only eliminates the highlight crush when sigmoid is *enabled*. With
    /// `sigmoid_on == false` and a scene-linear source (the raw path, values
    /// >1.0), velvia and splittoning still assume [0,1] display-referred input
    /// and clamp — sigmoid is off for already-display-referred JPEGs, where that
    /// assumption holds, and on for raws, where it is what saves the highlights.
    pub fn to_pipeline(&self, space: ColorSpace, scale: f32) -> Pipeline {
        let mut p = Pipeline::new();
        // Invert (film-camera negative, iop_order.c pos 2, before temperature 3) —
        // per-channel `out = color - in` on the decoded linear buffer. Unlike
        // temperature, color=[1,1,1] is NOT identity (it negates), so the only
        // guard is the stage enable flag itself.
        if self.invert_on {
            p.push(Stage::Invert {
                color: [self.invert_r, self.invert_g, self.invert_b, 1.0],
            });
        }
        // Temperature (white balance, iop_order.c pos ~20, before exposure 21) —
        // per-channel RGB multipliers on the decoded linear buffer. Runs first so
        // all downstream stages see corrected white balance.
        if self.temperature_on
            && (self.temperature_r != 1.0 || self.temperature_g != 1.0 || self.temperature_b != 1.0)
        {
            p.push(Stage::Temperature {
                coeffs: [self.temperature_r, self.temperature_g, self.temperature_b, 1.0],
            });
        }
        if self.exposure_on && (self.ev != 0.0 || self.black != 0.0) {
            p.push(Stage::Exposure { black: self.black, scale: 2.0f32.powf(self.ev) });
        }
        // Graduated ND (iop_order.c pos 25 — scene-referred, right after
        // exposure 21 and before the channel mix 28.5). Early placement is
        // correct: it is an optical filter, modelling glass in front of the
        // lens, so it belongs on linear scene data before any tone or colour
        // work. Density 0 is exp2(0) = 1 everywhere, a true no-op, so it is
        // skipped. The geometry depends on the buffer size and is derived in
        // Stage::apply, not here.
        if self.gradnd_on && self.gradnd_density != 0.0 {
            p.push(Stage::GraduatedNd {
                density: self.gradnd_density,
                hardness: self.gradnd_hardness,
                rotation: self.gradnd_rotation,
                offset: self.gradnd_offset,
                hue: self.gradnd_hue,
                saturation: self.gradnd_saturation,
            });
        }
        if self.mono_on {
            p.push(Stage::Monochrome { r: self.mono_r, g: self.mono_g, b: self.mono_b });
        }
        // Sharpen (scene-referred spatial stage, between channelmixer and sigmoid
        // per darktable iop_order.c). `space`/`scale` come from the render context:
        // the non-raw preview is sRGB at full res; the raw preview is Rec.2020,
        // scale 1.0 (WYSIWYG scale-tracking is a follow-up — see pipeline.rs docs).
        if self.sharpen_on && self.sharpen_amount > 0.0 && self.sharpen_radius > 0.0 {
            p.push(Stage::Sharpen {
                radius: self.sharpen_radius,
                amount: self.sharpen_amount,
                threshold: self.sharpen_threshold,
                space,
                scale,
            });
        }
        // Vibrance (scene-referred, iop_order.c pos 39.1) — runs after sharpen but
        // before the tone map so it boosts chroma in the wide-gamut linear domain.
        if self.vibrance_on && self.vibrance_amount > 0.0 {
            p.push(Stage::Vibrance {
                amount: self.vibrance_amount / 100.0,
                space,
            });
        }
        // Color correction (iop_order.c pos 55, after sharpen 53, before colorcontrast
        // 56) — luminance-dependent Lab a/b scaling + global saturation. The HSL-
        // style params (loa/hia/lob/hib/saturation) are converted to the core's
        // physical params (a_scale/a_base/b_scale/b_base/saturation) via
        // commit_params: a_scale = (hia - loa) / 100, a_base = loa, etc.
        if self.color_correction_on
            && (self.color_correction_saturation != 1.0
                || self.color_correction_loa != 0.0
                || self.color_correction_hia != 0.0
                || self.color_correction_lob != 0.0
                || self.color_correction_hib != 0.0)
        {
            p.push(Stage::ColorCorrection {
                a_scale: (self.color_correction_hia - self.color_correction_loa) / 100.0,
                a_base: self.color_correction_loa,
                b_scale: (self.color_correction_hib - self.color_correction_lob) / 100.0,
                b_base: self.color_correction_lob,
                saturation: self.color_correction_saturation,
                space,
            });
        }
        // Color contrast (iop_order.c pos 56-57, after sharpen 55) — operates in
        // Lab, altering chroma via a_steepness/b_steepness about the mid-slope.
        // 1.0/1.0 is identity; slider range is 0..=5.
        if self.color_contrast_on
            && (self.color_contrast_a_steepness != 1.0 || self.color_contrast_b_steepness != 1.0)
        {
            p.push(Stage::ColorContrast {
                a_steepness: self.color_contrast_a_steepness,
                b_steepness: self.color_contrast_b_steepness,
                space,
            });
        }
        // Color zones (iop_order.c pos 60, after colorcontrast 56, before sigmoid
        // 45.3) — LCH equaliser via 3×65536-entry LUTs built from spline curve
        // nodes. The LUTs are built here (at pipeline-build time) from the curve
        // data stored in PreviewParams; they are large (768KB) and not stored in
        // the params struct, so the params stay Copy-friendly.
        if self.colorzones_on {
            let periodic = self.colorzones_channel as i32 == 2; // h channel is periodic
            let channel = self.colorzones_channel as i32;
            let mode = self.colorzones_mode as i32;
            let lut_l = colorzones::build_lut(
                &self.colorzones_curve_x[0],
                &self.colorzones_curve_y[0],
                self.colorzones_num_nodes[0] as usize,
                self.colorzones_curve_type[0] as u32,
                periodic,
                self.colorzones_strength,
            );
            let lut_c = colorzones::build_lut(
                &self.colorzones_curve_x[1],
                &self.colorzones_curve_y[1],
                self.colorzones_num_nodes[1] as usize,
                self.colorzones_curve_type[1] as u32,
                periodic,
                self.colorzones_strength,
            );
            let lut_h = colorzones::build_lut(
                &self.colorzones_curve_x[2],
                &self.colorzones_curve_y[2],
                self.colorzones_num_nodes[2] as usize,
                self.colorzones_curve_type[2] as u32,
                periodic,
                self.colorzones_strength,
            );
            p.push(Stage::ColorZones { lut_l, lut_c, lut_h, channel, mode, space });
        }
        // Sigmoid is the scene-linear → display tone map. White (100%) / black
        // (0.0152%) targets are fixed at the darktable defaults (both > 0 ⇒ no
        // NaN); only contrast & skew are user-facing here.
        if self.sigmoid_on {
            let [white_target, black_target, paper_exp, film_fog, film_power, paper_power] =
                c41_core::iop::sigmoid::rgb_ratio_params(
                    self.sigmoid_contrast, self.sigmoid_skew, 100.0, 0.0152,
                );
            p.push(Stage::Sigmoid {
                white_target, black_target, paper_exp, film_fog, film_power, paper_power,
            });
        }
        // Levels (iop_order.c pos 49, in the display-referred cluster with
        // colisa 47 / tonecurve 48 / shadhi 50) — runs AFTER the tone map, not
        // before it. That ordering is load-bearing, not cosmetic: levels clips
        // everything at or below its black point to L=0 and treats Lab L as
        // 0..100, so on scene-linear input (L > 100 for highlights) it would
        // crush exactly the highlights sigmoid exists to roll off.
        //
        // `levels_stage_active` holds the skip rules (identity defaults,
        // degenerate range) and is shared with `is_identity`. The 65536-entry
        // LUT and derived inverse gamma are built here, as C `commit_params` does.
        if self.levels_stage_active() {
            // darktable's levels GUI structurally enforces black < grey < white
            // (levels.c `color_picker_apply` nudges the neighbours by
            // FLT_EPSILON, and so does the drag handler), which is what bounds
            // `tmp = (grey-mid)/delta` to [-1, 1] and hence inv_gamma to
            // [0.1, 10]. Three independent sliders give no such guarantee: e.g.
            // black 0 / grey 100 / white 1 is a legal set of positions and
            // yields 10^199 = +inf, which poisons the `pct > 1` branch of
            // process_pixels and lands NaN in the preview buffer. Re-impose the
            // invariant here, at the same layer the C does.
            let black = self.levels_black / 100.0;
            let white = self.levels_white / 100.0;
            let grey = (self.levels_grey / 100.0)
                .clamp(black + f32::EPSILON, white - f32::EPSILON);
            let stops = [black, grey, white];
            let (inv_gamma, lut) = c41_core::iop::levels::build_lut(stops);
            p.push(Stage::Levels {
                black: stops[0],
                range: stops[2] - stops[0],
                inv_gamma,
                // Unsized coercion + into_vec: reuses the allocation instead of
                // memcpying 256 KB on every slider tick.
                lut: (lut as Box<[f32]>).into_vec(),
                space,
            });
        }
        if self.velvia_on && self.velvia_strength > 0.0 {
            p.push(Stage::Velvia { strength: self.velvia_strength / 100.0, bias: self.velvia_bias });
        }
        // Colorize (iop_order.c pos 62, between velvia 57 and splittoning 67) —
        // replaces a/b channels with a fixed Lab colour, blends L from input.
        // The HSL params are converted to Lab here (in to_pipeline) via
        // hsl2rgb→sRGB→XYZ(D50)→Lab, matching darktable's commit_params. The stage
        // then round-trips RGB↔Lab per-pixel in Stage::apply.
        if self.colorize_on {
            let (r, g, b, _) = c41_core::color::hsl2rgb(
                self.colorize_hue,
                self.colorize_sat,
                self.colorize_lightness / 100.0,
            );
            let lab = c41_core::color::srgb_to_lab([r, g, b, 0.0]);
            p.push(Stage::Colorize {
                color_l: lab[0],
                color_a: lab[1],
                color_b: lab[2],
                mix: self.colorize_lightness_mix / 100.0,
                space,
            });
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
        // Lowlight (iop_order.c pos 63, between colorize 62 and monochrome 64)
        // — a display-referred creative module, so it sits after the tone map
        // with the others. The transition LUT is built here, as commit_params
        // does; the stage then does the RGB↔Lab round-trip per pixel.
        if self.lowlight_on {
            let x = [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0]; // bands at k/5, per init()
            let lut = c41_core::iop::lowlight::build_transition_lut(
                &x, &self.lowlight_transition,
            );
            p.push(Stage::Lowlight {
                blueness: self.lowlight_blueness,
                lut: (lut as Box<[f32]>).into_vec(),
                space,
            });
        }
        // Vignette (iop_order.c pos 68, last of the creative modules — after
        // splittoning 67). Both strengths at 0 scale nothing, so that is a
        // genuine no-op and is skipped. The falloff geometry is NOT computed
        // here: it depends on the buffer's dimensions, which only `Stage::apply`
        // knows, so caching it in the stage would go stale at another zoom.
        //
        // Dither is left off: darktable's default is DITHER_OFF, and the TEA
        // stream is seeded per row, so it belongs to the full-res render rather
        // than a downscaled preview.
        if self.vignette_on
            && (self.vignette_brightness != 0.0 || self.vignette_saturation != 0.0)
        {
            p.push(Stage::Vignette {
                scale: self.vignette_scale,
                falloff: self.vignette_falloff,
                brightness: self.vignette_brightness,
                saturation: self.vignette_saturation,
                center_x: self.vignette_center_x,
                center_y: self.vignette_center_y,
                shape: self.vignette_shape,
                // Automatic ratio: the vignette follows the image's own aspect,
                // which is darktable's behaviour for an un-set w/h ratio and
                // avoids exposing a second, subtler control for the same thing.
                autoratio: true,
                whratio: 1.0,
                dither_amt: 0.0,
                unbound: false,
            });
        }
        p
    }
}

/// Bump when the [`PreviewParams::encode`] layout changes (old blobs then decode
/// to `None` → defaults, rather than mis-parsing). v2 added the sigmoid stage.
/// v3 added the sharpen stage. v4 added vibrance. v5 added color contrast.
/// v6 added temperature (white balance). v7 adds invert (film-camera negative).
/// v8 adds colorize (HSL colour replacement). v9 adds color correction.
/// v10 adds color zones (LCH equaliser). v11 adds levels (black/grey/white).
/// Minimum black→white separation, on the 0..100 slider scale, for which a
/// Levels stage is emitted. A hairline range is not a meaningful edit (it maps
/// the whole tonal scale onto a sliver) and it drives `pct` — and hence the
/// `pct^gamma` branch of `process_pixels` — to absurd magnitudes. One slider
/// unit is far tighter than any useful edit while keeping the arithmetic sane;
/// the actual overflow guarantee comes from the output clamp in
/// `levels::process_pixels`, not from this.
const LEVELS_MIN_RANGE: f32 = 1.0;

const ENCODE_VERSION: u8 = 11;
/// 1 version byte + 17 bool bytes + 116 little-endian f32.
const ENCODED_LEN: usize = 1 + 17 + 116 * 4;

/// Run the preview pipeline over an 8-bit interleaved image buffer, preserving
/// layout (rowstride) and any alpha channel. Colour channels (0..min(3,nch))
/// are normalised to [0,1] into a packed RGBA `f32` buffer (4th channel = 1.0,
/// scratch), run through [`PreviewParams::to_pipeline`] /
/// `c41_core::pipeline`, then written back; the source alpha (channel 3,
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
    let pipeline = params.to_pipeline(ColorSpace::LinearSrgb, 1.0);
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
                rgba[o + c] = c41_core::color::srgb_to_linear(base[p + src] as f32 / 255.0);
            }
            rgba[o + 3] = 1.0;
        }
    }

    let processed = pipeline.process(&rgba, width, height);

    // ── scatter colour back (linear → sRGB), preserving alpha + padding ────
    let mut outbuf = base.to_vec();
    for y in 0..height {
        let row = y * rowstride;
        for x in 0..width {
            let p = row + x * nch;
            let o = (y * width + x) * 4;
            for c in 0..colour {
                let enc = c41_core::color::linear_to_srgb(processed[o + c]);
                outbuf[p + c] = (enc.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    outbuf
}

/// 16-bit sibling of [`apply_pipeline`] for the non-raw **export** path: packed
/// RGB `u16` sRGB in → packed RGB `u16` sRGB out, with the colour pipeline applied
/// in linear `f32`. Unlike the 8-bit preview path this preserves a 16-bit source's
/// precision and cuts requantisation banding on an edited gradient. Empty pipeline
/// ⇒ **byte-exact passthrough**, so an unedited 16-bit source round-trips
/// losslessly. Input is always tightly packed (rowstride = `width*3`, 3 channels;
/// the export compositor flattens alpha over white before calling this).
pub fn apply_pipeline_rgb16(
    base: &[u16],
    width: usize,
    height: usize,
    params: &PreviewParams,
) -> Vec<u16> {
    let n = width.saturating_mul(height);
    if width == 0 || height == 0 || base.len() < n * 3 {
        return base.to_vec();
    }
    let pipeline = params.to_pipeline(ColorSpace::LinearSrgb, 1.0);
    if pipeline.stages.is_empty() {
        return base.to_vec(); // no edit ⇒ lossless 16-bit passthrough
    }
    // sRGB (16-bit) → linear f32 RGBA; 4th channel = 1.0 scratch (exposure scales
    // all four), discarded on scatter.
    let mut rgba = vec![0.0f32; n * 4];
    for i in 0..n {
        for c in 0..3 {
            rgba[i * 4 + c] =
                c41_core::color::srgb_to_linear(base[i * 3 + c] as f32 / 65535.0);
        }
        rgba[i * 4 + 3] = 1.0;
    }
    let processed = pipeline.process(&rgba, width, height);
    let mut out = vec![0u16; n * 3];
    for i in 0..n {
        for c in 0..3 {
            let enc = c41_core::color::linear_to_srgb(processed[i * 4 + c]);
            out[i * 3 + c] = (enc.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    out
}

/// Run the preview pipeline on a packed **linear Rec.2020** RGBA `f32` buffer
/// (the raw path decodes to the Rec.2020 working space, values possibly >1.0),
/// convert to sRGB, and encode to tightly-packed 8-bit sRGB **RGB** for display.
/// Unlike [`apply_pipeline`], there is no 8-bit round-trip on the way in, so a
/// tone-map stage (sigmoid) sees the *unclipped* highlights and can roll them
/// off. The stages run in Rec.2020 (wide gamut → less premature clipping); the
/// `REC2020_TO_SRGB` map just before the OETF is neutral-preserving, and any
/// out-of-sRGB-gamut colour goes negative and hard-clips at the encode (the
/// display can't show it). `linear` must be `width*height*4` long.
pub fn render_linear_to_srgb8(
    linear: &[f32],
    width: usize,
    height: usize,
    params: &PreviewParams,
) -> Vec<u8> {
    let n = width.saturating_mul(height);
    if linear.len() < n * 4 {
        return vec![0u8; n * 3];
    }
    srgb_encode_rgb(linear, width, height, params)
        .iter()
        .map(|&e| (e.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

/// 16-bit sRGB variant of [`render_linear_to_srgb8`] for high-bit-depth export
/// (PNG/TIFF): same pipeline + Rec.2020→sRGB + OETF, quantised to 16 bits so
/// tonal gradients keep more headroom than an 8-bit encode. Tightly-packed
/// **RGB** `u16`, `width*height*3`.
pub fn render_linear_to_srgb16(
    linear: &[f32],
    width: usize,
    height: usize,
    params: &PreviewParams,
) -> Vec<u16> {
    let n = width.saturating_mul(height);
    if linear.len() < n * 4 {
        return vec![0u16; n * 3];
    }
    srgb_encode_rgb(linear, width, height, params)
        .iter()
        .map(|&e| (e.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
        .collect()
}

/// Shared render core: run the preview pipeline, map the Rec.2020 working space
/// to sRGB, and apply the sRGB OETF, yielding tightly-packed **RGB** sRGB floats
/// (`width*height*3`, pre-quantisation — values may fall outside `[0,1]` for
/// out-of-gamut colours, which the callers clamp). `linear` must be
/// `width*height*4` (callers guard the short case).
fn srgb_encode_rgb(linear: &[f32], width: usize, height: usize, params: &PreviewParams) -> Vec<f32> {
    let n = width.saturating_mul(height);
    let mut processed = params.to_pipeline(ColorSpace::Rec2020, 1.0).process(&linear[..n * 4], width, height);
    // Working space (Rec.2020) → sRGB before the display OETF (m4-35).
    c41_core::rawimage::apply_color_matrix(
        &mut processed,
        c41_core::rawimage::REC2020_TO_SRGB,
    );
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n {
        for c in 0..3 {
            out[i * 3 + c] = c41_core::color::linear_to_srgb(processed[i * 4 + c]);
        }
    }
    out
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

/// The on-screen rectangle an image of `img_w × img_h` occupies inside a
/// `widget_w × widget_h` widget under `ContentFit::Contain` (scaled to fit the
/// tighter axis, preserving aspect, centred with letterbox/pillarbox bars):
/// top-left offset, displayed size, and the image→widget `scale`. Shared by the
/// colour-picker hit-test and the snapshot wipe overlay so both letterbox with
/// *identical* geometry — features line up across the wipe divider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContainRect {
    pub off_x: f64,
    pub off_y: f64,
    pub disp_w: f64,
    pub disp_h: f64,
    pub scale: f64,
}

/// Compute the [`ContainRect`] for `img_w × img_h` inside `widget_w × widget_h`.
/// Returns `None` for degenerate (non-positive) sizes.
pub fn contain_rect(widget_w: f64, widget_h: f64, img_w: usize, img_h: usize) -> Option<ContainRect> {
    if img_w == 0 || img_h == 0 || widget_w <= 0.0 || widget_h <= 0.0 {
        return None;
    }
    let (iwf, ihf) = (img_w as f64, img_h as f64);
    let scale = (widget_w / iwf).min(widget_h / ihf); // Contain: fit the tighter axis
    let (disp_w, disp_h) = (iwf * scale, ihf * scale);
    let (off_x, off_y) = ((widget_w - disp_w) / 2.0, (widget_h - disp_h) / 2.0);
    Some(ContainRect { off_x, off_y, disp_w, disp_h, scale })
}

/// Clamp pointer `x` (widget space) to a fraction in `[0, 1]` across the
/// displayed image width of `rect` (left edge → 0, right edge → 1). Positions the
/// snapshot wipe divider; clamps so a drag past the letterbox bars pins to an end.
pub fn wipe_fraction(rect: &ContainRect, x: f64) -> f64 {
    if rect.disp_w <= 0.0 {
        return 0.0;
    }
    ((x - rect.off_x) / rect.disp_w).clamp(0.0, 1.0)
}

/// Pack an 8-bit interleaved `src` image into a cairo `Rgb24`-layout buffer of
/// `dst_stride * height` bytes: each pixel becomes 4 bytes in native-endian order
/// (little-endian: B, G, R, x — the order cairo's `Rgb24` expects). Greyscale
/// (<3 channels) replicates its last channel; a pixel whose source bytes run past
/// `src` is left black (defends a short/corrupt buffer); `nch == 0` yields an
/// all-zero buffer. Pure (no GTK), so the byte-swap/greyscale logic is unit-tested
/// headless — the thin cairo-surface wrapper lives in the darkroom view.
pub fn pack_rgb24(
    src: &[u8],
    width: usize,
    height: usize,
    rowstride: usize,
    nch: usize,
    dst_stride: usize,
) -> Vec<u8> {
    debug_assert!(dst_stride >= width * 4, "dst_stride must hold one Rgb24 pixel per column");
    let mut data = vec![0u8; dst_stride * height];
    let colour = nch.min(3);
    if colour == 0 {
        return data;
    }
    for y in 0..height {
        for x in 0..width {
            let sp = y * rowstride + x * nch;
            if sp + colour > src.len() {
                continue;
            }
            let r = src[sp];
            let g = src[sp + 1.min(colour - 1)];
            let b = src[sp + 2.min(colour - 1)];
            let dp = y * dst_stride + x * 4;
            data[dp] = b;
            data[dp + 1] = g;
            data[dp + 2] = r;
            data[dp + 3] = 0xff;
        }
    }
    data
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
    let r = contain_rect(widget_w, widget_h, img_w, img_h)?;
    let ix = (x - r.off_x) / r.scale;
    let iy = (y - r.off_y) / r.scale;
    if ix < 0.0 || iy < 0.0 || ix >= img_w as f64 || iy >= img_h as f64 {
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
        // sharpen enabled but amount=0 ⇒ still identity
        let mut sh0 = PreviewParams::default();
        sh0.sharpen_on = true;
        sh0.sharpen_amount = 0.0;
        cfgs.push(sh0);
        // sharpen enabled but radius=0 ⇒ still identity (Stage early-returns)
        let mut shr0 = PreviewParams::default();
        shr0.sharpen_on = true;
        shr0.sharpen_amount = 1.0;
        shr0.sharpen_radius = 0.0;
        cfgs.push(shr0);
        // sharpen enabled with positive amount+radius ⇒ non-identity
        let mut sh = PreviewParams::default();
        sh.sharpen_on = true;
        sh.sharpen_amount = 1.0;
        cfgs.push(sh);
        // levels enabled at the default stops ⇒ still identity (identity curve)
        let mut lv_def = PreviewParams::default();
        lv_def.levels_on = true;
        cfgs.push(lv_def);
        // levels with an inverted range ⇒ non-default but emits no stage, so
        // is_identity must agree. This is the case that falsified the invariant
        // before the two checks were factored into `levels_stage_active`.
        let mut lv_inv = PreviewParams::default();
        lv_inv.levels_on = true;
        lv_inv.levels_black = 60.0;
        lv_inv.levels_white = 40.0;
        cfgs.push(lv_inv);
        // levels with a real curve ⇒ non-identity
        let mut lv = PreviewParams::default();
        lv.levels_on = true;
        lv.levels_grey = 35.0;
        cfgs.push(lv);
        // colorize enabled ⇒ non-identity (on == off is the only no-op; sat 0 still
        // replaces a/b channels)
        let mut cz = PreviewParams::default();
        cz.colorize_on = true;
        cfgs.push(cz);
        // color correction with saturation=1.0 and all offsets 0 ⇒ identity
        // (matches to_pipeline gate: no stage pushed)
        let mut cc_id = PreviewParams::default();
        cc_id.color_correction_on = true;
        cc_id.color_correction_saturation = 1.0;
        cfgs.push(cc_id);
        // color correction with saturation != 1.0 ⇒ non-identity
        let mut cc = PreviewParams::default();
        cc.color_correction_on = true;
        cc.color_correction_saturation = 1.5;
        cfgs.push(cc);
        for c in cfgs {
            assert_eq!(
                c.is_identity(),
                c.to_pipeline(ColorSpace::LinearSrgb, 1.0).stages.is_empty(),
                "is_identity vs empty-pipeline disagree for {c:?}"
            );
        }
    }

    #[test]
    fn render_linear_identity_encodes_srgb() {
        // empty pipeline ⇒ Rec.2020→sRGB (neutral-preserving, so a no-op for
        // greys) then the linear→sRGB OETF, RGB8 (alpha dropped). Three grey
        // levels exercise black/mid/white: 0→0, 0.214≈sRGB 0.5→128, 1→255.
        let lin = vec![
            0.0f32, 0.0, 0.0, 1.0, // black
            0.214, 0.214, 0.214, 1.0, // mid grey (0.214 ≈ sRGB 0.5)
            1.0, 1.0, 1.0, 1.0, // white
        ];
        let out = render_linear_to_srgb8(&lin, 3, 1, &PreviewParams::default());
        assert_eq!(out.len(), 9);
        for c in 0..3 {
            assert_eq!(out[c], 0, "black");
            assert!((out[3 + c] as i32 - 128).abs() <= 1, "mid {}", out[3 + c]);
            assert_eq!(out[6 + c], 255, "white");
        }
    }

    #[test]
    fn to_pipeline_orders_stages_canonically() {
        // Pins darktable's v3.0 scene-referred iop order (iop_order.c): the
        // display-referred creative stages (velvia, splittoning) must run AFTER
        // the sigmoid tone map, the scene-referred ones (exposure, channel-mix)
        // before it. A regression here silently re-clips scene-linear
        // highlights (see velvia_after_sigmoid_* below).
        let mut p = PreviewParams::default();
        p.exposure_on = true;
        p.ev = 0.5;
        p.mono_on = true;
        p.sigmoid_on = true;
        p.velvia_on = true;
        p.velvia_strength = 50.0;
        p.split_on = true;
        p.sharpen_on = true;
        p.sharpen_amount = 1.0;
        p.color_correction_on = true;
        p.color_correction_saturation = 1.5;
        p.colorize_on = true;
        p.levels_on = true;
        p.levels_grey = 40.0; // off-default so the stage is actually emitted
        let names: Vec<&str> = p.to_pipeline(ColorSpace::LinearSrgb, 1.0).stages.iter().map(|s| s.name()).collect();
        assert_eq!(
            names,
            ["exposure", "channelmixer", "sharpen", "colorcorrection", "sigmoid", "levels", "velvia", "colorize", "splittoning"]
        );
        // Levels is display-referred (iop_order.c pos 49, after sigmoid 45.3):
        // it clips at its black point and treats L as 0..100, so running it
        // before the tone map would crush the scene-linear highlights sigmoid
        // is there to roll off.
        let sig = names.iter().position(|n| *n == "sigmoid").unwrap();
        let lev = names.iter().position(|n| *n == "levels").unwrap();
        assert!(lev > sig, "levels must run after sigmoid: {names:?}");
    }

    #[test]
    fn levels_slider_domain_is_nan_free() {
        // The three sliders move independently, so nothing stops a user from
        // setting black=0, grey=100, white=1 — legal positions that pass the
        // `white > black` gate but give tmp = 199 ⇒ 10^199 ⇒ +inf ⇒ NaN L in
        // the preview buffer. `to_pipeline` clamps the grey stop into
        // (black, white) to restore the invariant darktable's GUI enforces.
        // Sweep the whole grid, including inverted and coincident stops, and
        // demand a finite render every time.
        let vals = [0.0f32, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0];
        // Includes an L above the white point to hit the `pct > 1` powf branch.
        let px = vec![0.35f32, 0.5, 0.9, 1.0, 3.0, 3.0, 3.0, 1.0];
        for &black in &vals {
            for &grey in &vals {
                for &white in &vals {
                    let mut p = PreviewParams::default();
                    p.levels_on = true;
                    p.levels_black = black;
                    p.levels_grey = grey;
                    p.levels_white = white;
                    let out = p
                        .to_pipeline(ColorSpace::LinearSrgb, 1.0)
                        .process(&px, 2, 1);
                    assert!(
                        out.iter().all(|v| v.is_finite()),
                        "non-finite output for black={black} grey={grey} white={white}: {out:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn levels_black_and_white_points_move_tones_as_expected() {
        // End-to-end through Stage::Levels::apply (the stage-emission tests
        // above never exercise the pixel path).
        //
        // Note the grey stop is the *midtone* control and is held at the
        // midpoint here. Moving black or white alone also moves the midpoint
        // (mid = black + (white-black)/2), so a grey left at 50 becomes
        // off-centre and changes gamma too — that is why this test moves grey
        // with the endpoint rather than asserting a naive
        // "raising black always darkens".
        let mid = vec![0.35f32, 0.35, 0.35, 1.0];

        // Raise black, keeping grey centred (gamma stays 1): the tonal range
        // below the new black point is crushed, so the midtone darkens.
        let mut raise_black = PreviewParams::default();
        raise_black.levels_on = true;
        raise_black.levels_black = 30.0;
        raise_black.levels_grey = 65.0; // centred in [30, 100]
        let darker = raise_black
            .to_pipeline(ColorSpace::LinearSrgb, 1.0)
            .process(&mid, 1, 1);
        assert!(
            darker[0] < mid[0],
            "raising the black point (grey centred) should darken a midtone: {} !< {}",
            darker[0], mid[0]
        );

        // Lower white, keeping grey centred: the range compresses upward, so
        // the same midtone brightens.
        let mut lower_white = PreviewParams::default();
        lower_white.levels_on = true;
        lower_white.levels_white = 60.0;
        lower_white.levels_grey = 30.0; // centred in [0, 60]
        let brighter = lower_white
            .to_pipeline(ColorSpace::LinearSrgb, 1.0)
            .process(&mid, 1, 1);
        assert!(
            brighter[0] > mid[0],
            "lowering the white point (grey centred) should brighten a midtone: {} !> {}",
            brighter[0], mid[0]
        );

        // The grey stop alone drives gamma: below the midpoint brightens.
        let mut lift_grey = PreviewParams::default();
        lift_grey.levels_on = true;
        lift_grey.levels_grey = 35.0;
        let lifted = lift_grey
            .to_pipeline(ColorSpace::LinearSrgb, 1.0)
            .process(&mid, 1, 1);
        assert!(
            lifted[0] > mid[0],
            "grey below the midpoint should brighten: {} !> {}", lifted[0], mid[0]
        );

        assert!(darker.iter().chain(brighter.iter()).chain(lifted.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn levels_after_sigmoid_preserves_scene_linear_highlight() {
        // The behavioural counterpart to the name-position assertion in
        // to_pipeline_orders_stages_canonically. Levels treats Lab L as 0..100
        // and clips at its black point, so running it BEFORE the tone map
        // crushes a scene-linear highlight that the canonical order rolls off.
        let hot = [3.0f32, 3.0, 3.0, 1.0]; // L well above 100 before tone mapping
        let [white_target, black_target, paper_exp, film_fog, film_power, paper_power] =
            c41_core::iop::sigmoid::rgb_ratio_params(1.5, 0.0, 100.0, 0.0152);
        let sigmoid = Stage::Sigmoid {
            white_target, black_target, paper_exp, film_fog, film_power, paper_power,
        };
        let (inv_gamma, lut) = c41_core::iop::levels::build_lut([0.2, 0.5, 0.8]);
        let levels = Stage::Levels {
            black: 0.2,
            range: 0.6,
            inv_gamma,
            lut: (lut as Box<[f32]>).into_vec(),
            space: ColorSpace::LinearSrgb,
        };

        let canonical = Pipeline::with_stages(vec![sigmoid.clone(), levels.clone()])
            .process(&hot, 1, 1);
        let reversed = Pipeline::with_stages(vec![levels, sigmoid]).process(&hot, 1, 1);

        assert!(
            canonical[0] > reversed[0],
            "canonical (sigmoid→levels) must retain more highlight than levels-first: \
             {} !> {}", canonical[0], reversed[0]
        );
    }

    #[test]
    fn levels_default_stops_emit_no_stage() {
        // black 0 / grey 50 / white 100 is the identity curve, and a degenerate
        // range (white <= black) would divide by zero in the gamma derivation.
        // Neither should reach the pipeline.
        let mut p = PreviewParams::default();
        p.levels_on = true;
        assert!(p.is_identity(), "default stops should be identity");
        assert!(!p.to_pipeline(ColorSpace::LinearSrgb, 1.0).stages.iter().any(|s| s.name() == "levels"));

        p.levels_black = 60.0;
        p.levels_white = 40.0; // inverted ⇒ degenerate
        let stages = p.to_pipeline(ColorSpace::LinearSrgb, 1.0);
        assert!(!stages.stages.iter().any(|s| s.name() == "levels"), "degenerate range must be skipped");
    }

    #[test]
    fn velvia_after_sigmoid_preserves_scene_linear_highlight() {
        // Velvia is identity on greys (the chroma boost is 0 when R=G=B) but
        // hard-clamps its output to [0,1] — a faithful port of the
        // display-referred C module. Before the tone map that clamp crushed a
        // scene-linear neutral highlight (2.0 → 1.0), dimming what sigmoid
        // rendered. In canonical order (velvia AFTER sigmoid) enabling velvia
        // must not change a neutral highlight at all.
        let lin = vec![2.0f32, 2.0, 2.0, 1.0];
        let mut sig_only = PreviewParams::default();
        sig_only.sigmoid_on = true;
        let mut with_velvia = sig_only;
        with_velvia.velvia_on = true;
        with_velvia.velvia_strength = 80.0;
        assert_eq!(
            render_linear_to_srgb8(&lin, 1, 1, &sig_only),
            render_linear_to_srgb8(&lin, 1, 1, &with_velvia),
            "velvia on a tone-mapped neutral must be an exact no-op"
        );
    }

    #[test]
    fn canonical_order_preserves_chromatic_highlight_velvia_first_crushes_it() {
        // The grey test above cannot distinguish stage orders — velvia is
        // identity on R=G=B regardless of position. This one does, on a
        // *chromatic* scene-linear highlight whose red channel is >1.0.
        //
        // Canonical (m4-36) order sigmoid → velvia: the tone map rolls the 3.0
        // red into display range first, so velvia's [0,1] clamp never touches
        // the highlight. Pre-m4-36 order velvia → sigmoid: velvia clamps the
        // 3.0 red down to 1.0 *before* the tone map, permanently discarding
        // everything the sigmoid rolloff would have preserved. Building the two
        // pipelines explicitly (bypassing to_pipeline) pins that the canonical
        // order keeps the red highlight strictly brighter than the reversed one.
        let lin = [3.0f32, 1.5, 0.5, 1.0];
        let [white_target, black_target, paper_exp, film_fog, film_power, paper_power] =
            c41_core::iop::sigmoid::rgb_ratio_params(1.5, 0.0, 100.0, 0.0152);
        let sigmoid = Stage::Sigmoid {
            white_target, black_target, paper_exp, film_fog, film_power, paper_power,
        };
        let velvia = Stage::Velvia { strength: 0.8, bias: 1.0 };

        let canonical = Pipeline::with_stages(vec![sigmoid.clone(), velvia.clone()]).process(&lin, lin.len() / 4, 1);
        let reversed = Pipeline::with_stages(vec![velvia, sigmoid]).process(&lin, lin.len() / 4, 1);

        assert!(
            canonical[0] > reversed[0],
            "canonical (sigmoid→velvia) red {} must exceed velvia-first red {}: \
             velvia-first clipped the scene-linear highlight before the tone map",
            canonical[0], reversed[0]
        );
    }

    #[test]
    fn sigmoid_rolls_off_unclipped_highlight() {
        // The whole point of the float path: a scene-linear highlight >1 must NOT
        // hard-clip to 255 once sigmoid is on — it rolls off below white.
        let lin = vec![2.0f32, 2.0, 2.0, 1.0]; // linear, well above display white
        let plain = render_linear_to_srgb8(&lin, 1, 1, &PreviewParams::default());
        assert_eq!(plain[0], 255, "no tone-map ⇒ clips to white");
        let mut p = PreviewParams::default();
        p.sigmoid_on = true;
        let toned = render_linear_to_srgb8(&lin, 1, 1, &p);
        assert!(toned[0] < 255, "sigmoid should roll the highlight off: {}", toned[0]);
        assert!(toned[0] > 200, "but still bright: {}", toned[0]);
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
            sharpen_on: true, sharpen_radius: 4.0, sharpen_amount: 1.0, sharpen_threshold: 0.05,
            vibrance_on: true, vibrance_amount: 33.0,
            color_contrast_on: true, color_contrast_a_steepness: 2.0, color_contrast_b_steepness: 1.5,
            temperature_on: true, temperature_r: 1.2, temperature_g: 0.9, temperature_b: 0.8,
            invert_on: true, invert_r: 0.9, invert_g: 0.8, invert_b: 0.7,
            colorize_on: true, colorize_hue: 0.1, colorize_sat: 0.6, colorize_lightness: 75.0, colorize_lightness_mix: 50.0,
            color_correction_on: true, color_correction_loa: 5.0, color_correction_hia: 10.0,
            color_correction_lob: -3.0, color_correction_hib: 7.0, color_correction_saturation: 1.5,
            colorzones_on: true, colorzones_strength: 25.0, colorzones_channel: 1.0,
            colorzones_mode: 0.0,
            colorzones_num_nodes: [2.0, 2.0, 2.0],
            colorzones_curve_type: [1.0, 1.0, 1.0],
            colorzones_curve_x: [[0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 3],
            colorzones_curve_y: [[0.5; 8]; 3],
            levels_on: true, levels_black: 5.0, levels_grey: 45.0, levels_white: 95.0,
            vignette_on: true, vignette_scale: 70.0, vignette_falloff: 40.0,
            vignette_brightness: -0.6, vignette_saturation: 0.3,
            vignette_center_x: -0.2, vignette_center_y: 0.15, vignette_shape: 1.4,
            lowlight_on: true, lowlight_blueness: 30.0,
            lowlight_transition: [0.1, 0.3, 0.5, 0.6, 0.8, 0.9],
            gradnd_on: true, gradnd_density: -2.5, gradnd_hardness: 40.0,
            gradnd_rotation: -75.0, gradnd_offset: 35.0,
            gradnd_hue: 0.6, gradnd_saturation: 0.4,
        };
        let blob = p.encode();
        assert_eq!(blob.len(), ENCODED_LEN);
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
        // never misread as the current version (lengths differ).
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
    fn contain_rect_centres_and_letterboxes() {
        // 200x100 widget, 100x100 image → scale 1.0, pillarbox (off_x 50).
        let r = contain_rect(200.0, 100.0, 100, 100).unwrap();
        assert_eq!((r.scale, r.off_x, r.off_y), (1.0, 50.0, 0.0));
        assert_eq!((r.disp_w, r.disp_h), (100.0, 100.0));
        // 2x downscale, no bars.
        let r2 = contain_rect(100.0, 100.0, 200, 200).unwrap();
        assert_eq!((r2.scale, r2.off_x, r2.off_y), (0.5, 0.0, 0.0));
        // degenerate sizes → None.
        assert!(contain_rect(0.0, 100.0, 10, 10).is_none());
        assert!(contain_rect(100.0, 100.0, 0, 10).is_none());
    }

    #[test]
    fn wipe_fraction_clamps_across_displayed_width() {
        // image spans widget x∈[50,150].
        let r = contain_rect(200.0, 100.0, 100, 100).unwrap();
        assert_eq!(wipe_fraction(&r, 50.0), 0.0); // left edge
        assert_eq!(wipe_fraction(&r, 100.0), 0.5); // centre
        assert_eq!(wipe_fraction(&r, 150.0), 1.0); // right edge
        assert_eq!(wipe_fraction(&r, 0.0), 0.0); // past the left bar → clamped
        assert_eq!(wipe_fraction(&r, 999.0), 1.0); // past the right bar → clamped
    }

    #[test]
    fn pack_rgb24_swaps_channels_and_handles_greyscale() {
        // 2x1 RGB, dst stride 8: pixel0 (10,20,30) → B,G,R,x = 30,20,10,255.
        let src = vec![10u8, 20, 30, 40, 50, 60];
        let out = pack_rgb24(&src, 2, 1, 6, 3, 8);
        assert_eq!(&out[0..4], &[30, 20, 10, 0xff]);
        assert_eq!(&out[4..8], &[60, 50, 40, 0xff]);
        // 4-channel: alpha dropped, RGB still swapped.
        let rgba = vec![1u8, 2, 3, 99];
        assert_eq!(&pack_rgb24(&rgba, 1, 1, 4, 4, 4)[0..4], &[3, 2, 1, 0xff]);
        // greyscale replicates the single channel.
        assert_eq!(&pack_rgb24(&[77u8], 1, 1, 1, 1, 4)[0..4], &[77, 77, 77, 0xff]);
        // truncated source: present pixel packed, missing pixel left black (no panic).
        let short = vec![5u8, 6, 7]; // only pixel0 of a 2x1 RGB
        let ot = pack_rgb24(&short, 2, 1, 6, 3, 8);
        assert_eq!(&ot[0..4], &[7, 6, 5, 0xff]);
        assert_eq!(&ot[4..8], &[0, 0, 0, 0]);
        // nch 0 → all zeros.
        assert_eq!(pack_rgb24(&[], 1, 1, 0, 0, 4), vec![0u8; 4]);
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
