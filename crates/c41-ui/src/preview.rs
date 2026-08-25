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
use c41_core::iop::primaries;
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
    /// Contrast/brightness/saturation (colisa) stage on/off.
    pub colisa_on: bool,
    /// All three on darktable's -1..1 scale; 0 is neutral for each.
    pub colisa_contrast: f32,
    pub colisa_brightness: f32,
    pub colisa_saturation: f32,
    /// Basic adjustments (basicadj) stage on/off.
    ///
    /// Upstream's own iop_order comment calls this a "module mixing view/model/
    /// control at once, usage should be discouraged" — it overlaps exposure,
    /// filmic and colorbalancergb. Exposed because the processing is ported and
    /// darktable still ships it, not because it is the recommended path.
    pub basicadj_on: bool,
    pub basicadj_black_point: f32,
    pub basicadj_exposure: f32,
    pub basicadj_hlcompr: f32,
    pub basicadj_hlcomprthresh: f32,
    pub basicadj_contrast: f32,
    /// `dt_iop_rgb_norms_t` as a float so it rides the existing f32 payload.
    /// 0 = off (per-channel LUT contrast), 1 = luminance, 2 = max RGB, …
    pub basicadj_preserve_colors: f32,
    pub basicadj_middle_grey: f32,
    pub basicadj_brightness: f32,
    pub basicadj_saturation: f32,
    pub basicadj_vibrance: f32,
    /// Lowpass (local contrast enhancement) stage on/off.
    pub lowpass_on: bool,
    /// Gaussian blur radius (darktable 0.1..500, default 10.0).
    pub lowpass_radius: f32,
    /// Contrast curve strength (darktable -3..3, default 1.0 = identity).
    pub lowpass_contrast: f32,
    /// Brightness curve adjustment (darktable -3..3, default 0.0 = no shift).
    pub lowpass_brightness: f32,
    /// a/b channel saturation multiplier (darktable -3..3, default 1.0 = no change).
    pub lowpass_saturation: f32,
    /// Shadows/Highlights (shadhi.c) stage on/off. iop_order.c position 50.0.
    ///
    /// A Gaussian-blurred base layer of the Lab buffer is merged with the original:
    /// shadows lifts dark regions, highlights recovers blown highlights. Not
    /// pixel-local (the blur reads neighbours), so it stays on the serial path.
    /// The C default algorithm is bilateral; we hardcode Gaussian because
    /// `crate::gaussian` only implements that (the shadow/highlight math is identical).
    pub shadhi_on: bool,
    /// Shadows lift, -100..100 (C `shadows` slider, darktable default 50).
    pub shadhi_shadows: f32,
    /// Highlights recovery, -100..100 (C `highlights` slider, default -50).
    pub shadhi_highlights: f32,
    /// White point shift, -10..10 (C `whitepoint` slider, default 0).
    pub shadhi_whitepoint: f32,
    /// Blur radius, 0.1..500 (C `radius` slider, default 100). Sigma = max(0.1, radius) * scale.
    pub shadhi_radius: f32,
    /// Compression strength, 0..100 (C `compress` slider, default 50).
    pub shadhi_compress: f32,
    /// Shadows colour correction, 0..100 (C `shadows_ccorrect` slider, default 100).
    pub shadhi_shadows_ccorrect: f32,
    /// Highlights colour correction, 0..100 (C `highlights_ccorrect` slider, default 50).
    pub shadhi_highlights_ccorrect: f32,
    // ── Primaries (primaries.c, iop_order.c v50_order pos 28.5) ──────────
    // RGB chromaticity adjustment: rotates each working-space primary
    // around the white point and scales its distance from it. Hue is stored
    // in degrees (slider range -180..180); purity is a multiplier (1.0 =
    // unchanged). Achromatic tint purity 0 keeps the white point fixed.
    /// Primaries module enabled.
    pub primaries_on: bool,
    /// Achromatic tint hue, degrees (-180..180, default 0).
    pub primaries_achromatic_tint_hue: f32,
    /// Achromatic tint purity, 0..0.99 (default 0 = white point fixed).
    pub primaries_achromatic_tint_purity: f32,
    /// Red primary hue shift, degrees (-180..180, default 0).
    pub primaries_red_hue: f32,
    /// Red primary purity scale, 0.01..5.0 (default 1.0 = unchanged).
    pub primaries_red_purity: f32,
    /// Green primary hue shift, degrees (-180..180, default 0).
    pub primaries_green_hue: f32,
    /// Green primary purity scale, 0.01..5.0 (default 1.0 = unchanged).
    pub primaries_green_purity: f32,
    /// Blue primary hue shift, degrees (-180..180, default 0).
    pub primaries_blue_hue: f32,
    /// Blue primary purity scale, 0.01..5.0 (default 1.0 = unchanged).
    pub primaries_blue_purity: f32,
    /// Negadoctor (film negative scan inversion) stage on/off.
    ///
    /// Replaces the old invert module for colour negatives, taking the film's
    /// Dmin substrate colour, white-balance coefficients, Dmax and paper-grade
    /// gamma into account. iop_order.c position 28.5 — display-referred, right
    /// after graduatednd (28.0) and alongside channelmixerrgb/primaries.
    pub negadoctor_on: bool,
    /// Film stock: 0 = B&W, 1 = colour (darktable `DT_FILMSTOCK_NB` / `DT_FILMSTOCK_COLOR`).
    pub negadoctor_film_stock: f32,
    /// Dmin substrate colour multipliers (RGB). For B&W film the mono default
    /// collapses to (1.0, 1.0, 1.0).
    pub negadoctor_dmin_r: f32,
    pub negadoctor_dmin_g: f32,
    pub negadoctor_dmin_b: f32,
    /// White-balance RGB coefficients (illuminant); core gets `wb_high / D_max`.
    pub negadoctor_wb_high_r: f32,
    pub negadoctor_wb_high_g: f32,
    pub negadoctor_wb_high_b: f32,
    /// White-balance RGB offsets (base light); combined with `offset` and `wb_high`
    /// in `to_pipeline` to form the per-channel density offset.
    pub negadoctor_wb_low_r: f32,
    pub negadoctor_wb_low_g: f32,
    pub negadoctor_wb_low_b: f32,
    /// Max density of the film (core divides `wb_high` by `D_max` per channel).
    pub negadoctor_d_max: f32,
    /// Inversion offset (scan exposure bias), -1..1, default -0.05.
    pub negadoctor_offset: f32,
    /// Display black level (paper black, density correction), -0.5..0.5, default 0.0755.
    pub negadoctor_black: f32,
    /// Paper grade (gamma), 1.0..8.0, default 4.0.
    pub negadoctor_gamma: f32,
    /// Highlights roll-off (paper gloss), 0.0001..1.0, default 0.75.
    pub negadoctor_soft_clip: f32,
    /// Print exposure adjustment, 0.5..2.0, default 0.9245.
    pub negadoctor_exposure: f32,
    // ── Tone equalizer (toneequal.c, iop_order.c v50_order pos 24.0) ─────
    // Scene-referred tone mapping by exposure channel: nine gain sliders, one
    // per exposure band from −8 EV to 0 EV, interpolated by a Gaussian RBF fit.
    // Runs in the `details == DT_TONEEQ_NONE` configuration ("preserve details:
    // no"); the guided-filter modes are not ported. Gains are EV offsets
    // (slider range −2..+2, darktable $MIN/$MAX), all zero = identity.
    /// Tone equalizer module enabled.
    pub toneeq_on: bool,
    /// Gain at −8 EV ("blacks" in the C params struct: `noise`), −2..+2 EV.
    pub toneeq_noise: f32,
    /// Gain at −7 EV (`ultra_deep_blacks`, "deep shadows"), −2..+2 EV.
    pub toneeq_ultra_deep_blacks: f32,
    /// Gain at −6 EV (`deep_blacks`, "shadows"), −2..+2 EV.
    pub toneeq_deep_blacks: f32,
    /// Gain at −5 EV (`blacks`, "light shadows"), −2..+2 EV.
    pub toneeq_blacks: f32,
    /// Gain at −4 EV (`shadows`, "mid-tones"), −2..+2 EV.
    pub toneeq_shadows: f32,
    /// Gain at −3 EV (`midtones`, "dark highlights"), −2..+2 EV.
    pub toneeq_midtones: f32,
    /// Gain at −2 EV (`highlights`), −2..+2 EV.
    pub toneeq_highlights: f32,
    /// Gain at −1 EV (`whites`), −2..+2 EV.
    pub toneeq_whites: f32,
    /// Gain at 0 EV (`speculars`), −2..+2 EV.
    pub toneeq_speculars: f32,
    // ── Color balance RGB (colorbalancergb.c, iop_order.c v50_order pos 41.5) ─
    // Scene-referred grading in Filmlight's Yrg space with perceptual
    // saturation/brilliance in dt-UCS (default) or JzAzBz. Field names and
    // defaults mirror `dt_iop_colorbalancergb_params_t` (v5); ranges are
    // darktable's soft ranges where set, else $MIN/$MAX.
    /// Color balance RGB module enabled.
    pub cb_on: bool,
    /// Shadows luminance offset (`shadows_Y`), −1..1.
    pub cb_shadows_y: f32,
    /// Shadows chroma (`shadows_C`), soft 0..0.5.
    pub cb_shadows_c: f32,
    /// Shadows hue in degrees (`shadows_H`), 0..360.
    pub cb_shadows_h: f32,
    /// Mid-tones luminance exponent offset (`midtones_Y`), soft −0.25..0.25.
    pub cb_midtones_y: f32,
    /// Mid-tones chroma of the colour exponent (`midtones_C`), soft 0..0.1.
    pub cb_midtones_c: f32,
    /// Mid-tones hue in degrees (`midtones_H`), 0..360.
    pub cb_midtones_h: f32,
    /// Highlights luminance offset (`highlights_Y`), soft −0.5..0.5.
    pub cb_highlights_y: f32,
    /// Highlights chroma (`highlights_C`), soft 0..0.2.
    pub cb_highlights_c: f32,
    /// Highlights hue in degrees (`highlights_H`), 0..360.
    pub cb_highlights_h: f32,
    /// Global luminance offset (`global_Y`), soft −0.05..0.05.
    pub cb_global_y: f32,
    /// Global chroma offset (`global_C`), soft 0..0.01.
    pub cb_global_c: f32,
    /// Global hue offset in degrees (`global_H`), 0..360.
    pub cb_global_h: f32,
    /// Shadows zone fall-off weight (`shadows_weight`), 0..3, default 1.
    pub cb_shadows_weight: f32,
    /// White fulcrum as an EV exponent (`white_fulcrum`; commit_params takes
    /// exp2 of it), soft −2..+2 EV, default 0.
    pub cb_white_fulcrum: f32,
    /// Highlights zone fall-off weight (`highlights_weight`), 0..3, default 1.
    pub cb_highlights_weight: f32,
    /// Chroma boost, shadows (`chroma_shadows`), ±1.
    pub cb_chroma_shadows: f32,
    /// Chroma boost, highlights (`chroma_highlights`), ±1.
    pub cb_chroma_highlights: f32,
    /// Chroma boost, global (`chroma_global`), soft ±0.5.
    pub cb_chroma_global: f32,
    /// Chroma boost, mid-tones (`chroma_midtones`), ±1.
    pub cb_chroma_midtones: f32,
    /// Perceptual saturation, global (`saturation_global`), ±1.
    pub cb_saturation_global: f32,
    /// Perceptual saturation, highlights (`saturation_highlights`), ±1.
    pub cb_saturation_highlights: f32,
    /// Perceptual saturation, mid-tones (`saturation_midtones`), ±1.
    pub cb_saturation_midtones: f32,
    /// Perceptual saturation, shadows (`saturation_shadows`), ±1.
    pub cb_saturation_shadows: f32,
    /// Global hue shift in degrees (`hue_angle`), ±180.
    pub cb_hue_angle: f32,
    /// Brilliance (luminance-correlated saturation), global, ±1.
    pub cb_brilliance_global: f32,
    /// Brilliance, highlights (`brilliance_highlights`), ±1.
    pub cb_brilliance_highlights: f32,
    /// Brilliance, mid-tones (`brilliance_midtones`), ±1.
    pub cb_brilliance_midtones: f32,
    /// Brilliance, shadows (`brilliance_shadows`), ±1.
    pub cb_brilliance_shadows: f32,
    /// Middle-grey fulcrum of the luminance masks (`mask_grey_fulcrum`),
    /// 0..1, default 0.1845.
    pub cb_mask_grey_fulcrum: f32,
    /// Vibrance — chroma boost weighted toward low chroma (`vibrance`),
    /// soft ±0.5.
    pub cb_vibrance: f32,
    /// Contrast grey fulcrum (`grey_fulcrum`), soft 0.1..0.5, default 0.1845.
    pub cb_grey_fulcrum: f32,
    /// Fulcrumed contrast strength (`contrast`), soft ±0.5.
    pub cb_contrast: f32,
    /// Saturation formula selector: 0 = JzAzBz (2021), 1 = dt-UCS (2022, the
    /// darktable default). Encoded as a float to keep the append-only layout.
    pub cb_formula: f32,
    // ── Filmic RGB (filmicrgb.c, iop_order.c v50_order pos 46.0) ─────────
    // Scene-referred display transform: log-encodes the exposure range between
    // the source black/white points and tone-maps it through a five-node spline
    // into the display targets, with a Yrg gamut map. Like sigmoid (45.3) it is
    // a tone curve — never a no-op while enabled — but unlike sigmoid there is
    // no "auto-enable for raws" workflow logic here: darktable turns filmic on
    // via its scene-referred default preset plus reload_defaults' auto-exposure,
    // which we don't replicate, so the module ships off like every other stage.
    /// Filmic RGB module enabled.
    pub filmic_on: bool,
    /// Black relative exposure in EV (`black_point_source`), −16..−0.1.
    pub filmic_black_point_source: f32,
    /// White relative exposure in EV (`white_point_source`), 0.1..16.
    pub filmic_white_point_source: f32,
    /// Display hardness / output power (`output_power`), 1..10, default 4.
    pub filmic_output_power: f32,
    /// Linear-region width in % (`latitude`), 0.01..99.
    pub filmic_latitude: f32,
    /// Contrast (`contrast`), 0..5 — centre-segment slope via the v3 relation.
    pub filmic_contrast: f32,
    /// Shadows ↔ highlights balance in % (`balance`), −50..50.
    pub filmic_balance: f32,
    /// Extreme-luminance saturation in % (`saturation`), −200..200.
    pub filmic_saturation: f32,
    // ── Highlight reconstruction (highlights.c, raw front end) ─────────────
    // Runs on the mosaic BEFORE demosaic (darktable's temperature → highlights
    // → demosaic order), so unlike every other module it is not a pipeline
    // stage — it threads into the raw decode (`RawImage::to_linear_rgba_with`)
    // and changing it re-decodes the preview. Raw inputs only: the non-raw
    // path is an 8-bit pixbuf front end with no float pre-demosaic domain.
    /// Highlight reconstruction module enabled. Ships off: darktable
    /// auto-enables "inpaint opposed" via its scene-referred default preset,
    /// which we don't replicate; off reproduces the legacy hard-clip-at-white.
    pub hl_on: bool,
    /// Method: true = "inpaint opposed" (darktable's default), false =
    /// "clip highlights". The other C methods are unwired.
    pub hl_opposed: bool,
    /// Clipping threshold (`clip`), 0..2, darktable default 1.0.
    pub hl_clip: f32,
    // ── Denoise (profiled) (denoiseprofile.c, wavelets mode) ────────────────
    // A normal pipeline stage (iop_order.c v50 pos 9/10, right after demosaic
    // 8): non-local à-trous wavelet shrinkage on the whole frame.
    /// Denoise module enabled. Ships off: denoising is expensive and
    /// darktable's default preset only auto-enables it via profileled noise
    /// detection, which we don't replicate (generic Poissonian a=1e-4 only).
    pub dn_on: bool,
    /// Colour mode: true = "Y0U0V0" (darktable's default — luma/chroma split),
    /// false = "RGB". The other C modes are unwired.
    pub dn_mode_y0u0v0: bool,
    /// Noise strength (`strength`), C introspection 0.001..1000 with a soft
    /// slider max of 4.0 (denoiseprofile.c:3555-3556); we expose 0.001..4.0,
    /// default 1.0. Scales the VST forward/backtransform pair (via the colour
    /// matrices in Y0U0V0 mode and wb in RGB mode).
    pub dn_strength: f32,
    /// Adjustor for blocksize-independent (`shadows`) mixing inside the
    /// strength compensation, 0..1.8, default 1.0.
    pub dn_shadows: f32,
    /// Bias applied to the VST backtransform (`bias`), C hard range ±1000 but
    /// soft slider range −10..10 (denoiseprofile.c:3559-3560); we expose
    /// −10..10, default 0. Effective correction is bias·(strength·2.5|1).
    pub dn_bias: f32,
    // ── Bloom (bloom.c) ─────────────────────────────────────────────────────
    // Display-referred creative module (iop_order.c v50 pos 61, between
    // colorzones 60 and colorize 62): gathers Lab L above a threshold,
    // box-blurs it and screen-blends it back — a whole-frame stage.
    /// Bloom module enabled. Ships off, like darktable's default.
    pub bl_on: bool,
    /// Glow size (`size`), 0..100, darktable default 20. Sets the box radius:
    /// rad = int(256·(min(100,size+1)/100)), capped at 256.
    pub bl_size: f32,
    /// Light threshold (`threshold`), 0..100 (on Lab L), darktable default 90.
    /// Only L values above it contribute gathered light.
    pub bl_threshold: f32,
    /// Glow strength (`strength`), 0..100, darktable default 25. Scales the
    /// gathered light by exp2(min(100,strength+1)/100) before the blur.
    pub bl_strength: f32,
    // ── Tone curve (tonecurve.c) ────────────────────────────────────────────
    // Three-channel Lab LUT module (iop_order.c pos 48, between colisa 47 and
    // levels 49). First slice: the L channel editor only — a/b keep their C
    // defaults (identity 3-node curves). Anchors are sampled through the V1
    // `curve_tools` port (`curve_data_sample`) exactly like commit_params.
    /// Tone curve module enabled. Ships off, like darktable's default.
    pub tc_on: bool,
    /// Spline type for all channels (`CurveData.m_spline_type`):
    /// 0 = cubic spline, 1 = Catmull-Rom, 2 = monotone Hermite (C default).
    pub tc_type: f32,
    /// a/b re-derivation mode (`autoscale_ab`): 0 manual, 1 automatic (XYZ),
    /// 2 automatic XYZ, 3 automatic RGB (C default).
    pub tc_autoscale: f32,
    /// Unbounded a/b curves (`unbound_ab`, C default true) — lets the
    /// extrapolated a/b tails leave [0,255].
    pub tc_unbound: bool,
    /// Colour-preservation norm for linked a/b (`preserve_colors`,
    /// DT_RGB_NORM_AVERAGE = 3 is the C default).
    pub tc_preserve: f32,
    /// Number of L-curve anchors in use (2..=20); the tail of
    /// [`PreviewParams::tc_nodes_l`] beyond this count is ignored.
    pub tc_nnodes: f32,
    /// L-curve anchor positions in curve-box coordinates ([0,1]²),
    /// x-sorted, first fixed at x=0 and last at x=1.
    pub tc_nodes_l: [(f32, f32); 20],
    // ── RGB curve (m4-123, rgbcurve) ────────────────────────────────────────
    /// RGB curve module enabled. Ships off, like darktable's default.
    pub rc_on: bool,
    /// Spline type per channel (`curve[ch].m_spline_type`): 0 = cubic spline,
    /// 1 = Catmull-Rom, 2 = monotone Hermite (C default). The UI exposes ONE
    /// interpolator dropdown that writes all three, mirroring C's
    /// `interpolator_callback` — but they are stored per channel because C does.
    pub rc_type_r: f32,
    pub rc_type_g: f32,
    pub rc_type_b: f32,
    /// Channel linking (`autoscale`): 0 = AUTOMATIC_RGB (the R curve drives all
    /// channels), 1 = MANUAL_RGB (independent per-channel curves). C default 0.
    pub rc_autoscale: f32,
    /// Colour-preservation norm for linked mode (`preserve_colors`):
    /// 0 none, 1 luminance (C default), 2 max, 3 average, 4 sum, 5 norm,
    /// 6 power.
    pub rc_preserve: f32,
    /// Anchor counts in use per channel (2..=20); tails of the node arrays
    /// beyond these counts are ignored.
    pub rc_nnodes_r: f32,
    pub rc_nnodes_g: f32,
    pub rc_nnodes_b: f32,
    /// Per-channel anchor positions in curve-box coordinates ([0,1]²),
    /// x-sorted, first fixed at x=0 and last at x=1.
    pub rc_nodes_r: [(f32, f32); 20],
    pub rc_nodes_g: [(f32, f32); 20],
    pub rc_nodes_b: [(f32, f32); 20],
    // ── Base curve (m4-124, basecurve) ──────────────────────────────────────
    /// Base curve module enabled. Ships off: darktable auto-applies its
    /// "display-referred default" preset to raws, which C41 does not run
    /// (same documented deviation as filmicrgb).
    pub bc_on: bool,
    /// Spline type of channel 0 (`basecurve_type[0]`): 0 = cubic spline,
    /// 1 = Catmull-Rom, 2 = monotone Hermite (C default). C stores three
    /// channels but reads only channel 0 (`const int ch = 0`, commit_params).
    pub bc_type: f32,
    /// Colour-preservation norm (`preserve_colors`): 0 = none (legacy shared-
    /// table path), 1 = luminance (C $DEFAULT), 2 max, 3 average, 4 sum,
    /// 5 norm, 6 power.
    pub bc_preserve: f32,
    /// Anchor count in use (2..=20); tail of [`Self::bc_nodes`] beyond this
    /// count is ignored.
    pub bc_nnodes: f32,
    /// Exposure-fusion steps (`exposure_fusion`): 0 = plain LUT (process_lut),
    /// 1/2 = two/three exposures blended through a laplacian pyramid.
    pub bc_exposure_fusion: f32,
    /// Stops between fused exposures (`exposure_stops`, C default 1.0).
    pub bc_exposure_stops: f32,
    /// Fusion direction (`exposure_bias`, −1 highlights .. +1 shadows). The C
    /// *param* default is 1.0; the widget's double-click default is 0. We carry
    /// the param default.
    pub bc_exposure_bias: f32,
    /// Channel-0 anchor positions in curve-box coordinates ([0,1]²),
    /// x-sorted, first fixed at x=0 and last at x=1.
    pub bc_nodes: [(f32, f32); 20],
    /// Lens correction (lens.cc LENSFUN method) stage enabled. Ships off:
    /// darktable auto-fills gear from EXIF and only corrects once the user
    /// picks a lens; C41 has no EXIF lens tag, so the module starts neutral
    /// until a camera+lens are chosen (the choice persists per image in
    /// `main.darkroom_lens_choice`, not here — the blob carries no strings).
    pub lens_on: bool,
    /// Correct ↔ distort direction swap (`inverse`): off = correct an image
    /// shot with this lens (default), on = simulate the lens on an image.
    pub lens_inverse: bool,
    /// Corrections combo (`modify_flags`), darktable's dropdown *value*:
    /// 7 = all (default), 5 = distortion & TCA, 6 = distortion & vignetting,
    /// 3 = TCA & vignetting, 4 = only distortion, 1 = only TCA, 2 = only
    /// vignetting.
    pub lens_modify_flags: f32,
    /// Manual scale factor (`scale`, C introspection `$DEFAULT: 1.0`). The
    /// camera's crop factor is supplied at pipeline-build time from the
    /// resolved gear (darktable takes it from the camera, never a slider).
    pub lens_scale: f32,
    /// Focal length in mm at capture (`focal`). No C introspection default —
    /// darktable's reload_defaults fills it from EXIF, and C41 has no EXIF
    /// lens tags, so this is a chosen neutral value (documented deviation).
    pub lens_focal: f32,
    /// Aperture as f-number (`aperture`). Same story as [`Self::lens_focal`] —
    /// EXIF-driven in C, a neutral chosen value here.
    pub lens_aperture: f32,
    /// Focus distance in metres (`distance`). C's no-EXIF fallback is 1000.0
    /// (lens.cc:3455); C41 has no EXIF focus distance at all, so that fallback
    /// IS our default — it feeds lensfun's vignetting interpolation.
    pub lens_distance: f32,
    /// Target projection (`target_geom`, `lfLensType` value; C default 1 =
    /// rectilinear).
    pub lens_target_geom: f32,
}

/// The C-default 2-node identity curve `[(0,0), (1,1)]`, tail zeroed — shared
/// by the tone-curve and RGB-curve default anchors.
fn identity_nodes_20() -> [(f32, f32); 20] {
    let mut n = [(0.0f32, 0.0f32); 20];
    n[1] = (1.0, 1.0);
    n
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
            // darktable defaults (colisa.c): off, all three neutral at 0.
            colisa_on: false,
            colisa_contrast: 0.0,
            colisa_brightness: 0.0,
            colisa_saturation: 0.0,
            // darktable defaults (basicadj.c $DEFAULT): off, neutral, middle
            // grey 18.42, preserve_colors = LUMINANCE (1).
            basicadj_on: false,
            basicadj_black_point: 0.0,
            basicadj_exposure: 0.0,
            basicadj_hlcompr: 0.0,
            basicadj_hlcomprthresh: 0.0,
            basicadj_contrast: 0.0,
            basicadj_preserve_colors: 1.0,
            basicadj_middle_grey: 18.42,
            basicadj_brightness: 0.0,
            basicadj_saturation: 0.0,
            basicadj_vibrance: 0.0,
            // darktable defaults (lowpass.c params struct): off, radius 10,
            // contrast 1.0 (identity), brightness 0.0, saturation 1.0,
            // unbound 1 (not surfaced — see commit_params in lowpass.rs).
            lowpass_on: false,
            lowpass_radius: 10.0,
            lowpass_contrast: 1.0,
            lowpass_brightness: 0.0,
            lowpass_saturation: 1.0,
            // Shadhi defaults mirror dt_iop_shadhi_params_t (shadhi.c lines 70-80):
            // off, radius 100, shadows 50, whitepoint 0, highlights -50,
            // compress 50, shadows_ccorrect 100, highlights_ccorrect 50.
            // `flags` (UNBOUND_DEFAULT=127) and `low_approximation` (0.000001)
            // are hardcoded in the Stage apply arm, not surfaced in the UI.
            shadhi_on: false,
            shadhi_shadows: 50.0,
            shadhi_highlights: -50.0,
            shadhi_whitepoint: 0.0,
            shadhi_radius: 100.0,
            shadhi_compress: 50.0,
            shadhi_shadows_ccorrect: 100.0,
            shadhi_highlights_ccorrect: 50.0,
            // Primaries defaults: off, all hues at 0, all RGB purities at 1.0
            // (unchanged), achromatic tint purity at 0 (white point fixed).
            primaries_on: false,
            primaries_achromatic_tint_hue: 0.0,
            primaries_achromatic_tint_purity: 0.0,
            primaries_red_hue: 0.0,
            primaries_red_purity: 1.0,
            primaries_green_hue: 0.0,
            primaries_green_purity: 1.0,
            primaries_blue_hue: 0.0,
            primaries_blue_purity: 1.0,
            // Negadoctor defaults mirror the darktable struct $DEFAULTs and
            // default_params(): colour film stock, Dmin {1.0, 0.45, 0.25, 1.0},
            // unit WB coeffs, D_max 2.046, offset -0.05, black 0.0755, gamma
            // 4.0, soft_clip 0.75, exposure 0.9245. Off by default.
            negadoctor_on: false,
            negadoctor_film_stock: 1.0, // DT_FILMSTOCK_COLOR
            negadoctor_dmin_r: 1.0,
            negadoctor_dmin_g: 0.45,
            negadoctor_dmin_b: 0.25,
            negadoctor_wb_high_r: 1.0,
            negadoctor_wb_high_g: 1.0,
            negadoctor_wb_high_b: 1.0,
            negadoctor_wb_low_r: 1.0,
            negadoctor_wb_low_g: 1.0,
            negadoctor_wb_low_b: 1.0,
            negadoctor_d_max: 2.046,
            negadoctor_offset: -0.05,
            negadoctor_black: 0.0755,
            negadoctor_gamma: 4.0,
            negadoctor_soft_clip: 0.75,
            negadoctor_exposure: 0.9245,
            toneeq_on: false,
            toneeq_noise: 0.0,
            toneeq_ultra_deep_blacks: 0.0,
            toneeq_deep_blacks: 0.0,
            toneeq_blacks: 0.0,
            toneeq_shadows: 0.0,
            toneeq_midtones: 0.0,
            toneeq_highlights: 0.0,
            toneeq_whites: 0.0,
            toneeq_speculars: 0.0,
            // Color balance RGB defaults mirror the $DEFAULTs in
            // dt_iop_colorbalancergb_params_t: the neutral edit (offset 0 /
            // slope 1 / power 1 everywhere), weights at 1, both fulcrums at
            // middle grey 0.1845, dt-UCS saturation formula.
            cb_on: false,
            cb_shadows_y: 0.0,
            cb_shadows_c: 0.0,
            cb_shadows_h: 0.0,
            cb_midtones_y: 0.0,
            cb_midtones_c: 0.0,
            cb_midtones_h: 0.0,
            cb_highlights_y: 0.0,
            cb_highlights_c: 0.0,
            cb_highlights_h: 0.0,
            cb_global_y: 0.0,
            cb_global_c: 0.0,
            cb_global_h: 0.0,
            cb_shadows_weight: 1.0,
            cb_white_fulcrum: 0.0,
            cb_highlights_weight: 1.0,
            cb_chroma_shadows: 0.0,
            cb_chroma_highlights: 0.0,
            cb_chroma_global: 0.0,
            cb_chroma_midtones: 0.0,
            cb_saturation_global: 0.0,
            cb_saturation_highlights: 0.0,
            cb_saturation_midtones: 0.0,
            cb_saturation_shadows: 0.0,
            cb_hue_angle: 0.0,
            cb_brilliance_global: 0.0,
            cb_brilliance_highlights: 0.0,
            cb_brilliance_midtones: 0.0,
            cb_brilliance_shadows: 0.0,
            cb_mask_grey_fulcrum: 0.1845,
            cb_vibrance: 0.0,
            cb_grey_fulcrum: 0.1845,
            cb_contrast: 0.0,
            cb_formula: 1.0, // SaturationFormula::DtUcs
            // Filmic RGB defaults mirror the $DEFAULTs of
            // dt_iop_filmicrgb_params_t: black −8 EV, white +4 EV, power 4
            // ("hard"), latitude 0.01%, contrast 1, balance 0, saturation 0.
            // Off by default — see the field doc above for why this module has
            // no identity-at-defaults shortcut.
            filmic_on: false,
            filmic_black_point_source: -8.0,
            filmic_white_point_source: 4.0,
            filmic_output_power: 4.0,
            filmic_latitude: 0.01,
            filmic_contrast: 1.0,
            filmic_balance: 0.0,
            filmic_saturation: 0.0,
            hl_on: false,
            hl_opposed: true,
            hl_clip: 1.0,
            dn_on: false,
            dn_mode_y0u0v0: true, // darktable reload_defaults: Y0U0V0 mode
            dn_strength: 1.0,
            dn_shadows: 1.0,
            dn_bias: 0.0,
            bl_on: false,
            bl_size: 20.0,      // bloom.h $DEFAULT
            bl_threshold: 90.0, // bloom.h $DEFAULT
            bl_strength: 25.0,  // bloom.h $DEFAULT
            tc_on: false,
            tc_type: 2.0,        // MONOTONE_HERMITE ($DEFAULT annotation)
            tc_autoscale: 3.0,   // DT_S_SCALE_AUTOMATIC_RGB (C default)
            tc_unbound: true,    // unbound_ab = TRUE in C init()
            tc_preserve: 3.0,    // DT_RGB_NORM_AVERAGE (C default)
            // tonecurve.c $DEFAULT L curve: two nodes (0,0)→(1,1) — identity.
            tc_nnodes: 2.0,
            tc_nodes_l: {
                let mut n = [(0.0f32, 0.0f32); 20];
                n[1] = (1.0, 1.0);
                n
            },
            rc_on: false,
            rc_type_r: 2.0, // MONOTONE_HERMITE ($DEFAULT annotation)
            rc_type_g: 2.0,
            rc_type_b: 2.0,
            rc_autoscale: 0.0, // DT_S_SCALE_AUTOMATIC_RGB (C default)
            rc_preserve: 1.0,  // DT_RGB_NORM_LUMINANCE (C default)
            // rgbcurve.c $DEFAULT curves: two identity nodes per channel.
            rc_nnodes_r: 2.0,
            rc_nnodes_g: 2.0,
            rc_nnodes_b: 2.0,
            rc_nodes_r: identity_nodes_20(),
            rc_nodes_g: identity_nodes_20(),
            rc_nodes_b: identity_nodes_20(),
            // basecurve.c $DEFAULTs: 2-node identity channel-0 curve,
            // MONOTONE_HERMITE, LUMINANCE preservation, no fusion.
            bc_on: false,
            bc_type: 2.0,           // MONOTONE_HERMITE ($DEFAULT annotation)
            bc_preserve: 1.0,       // DT_RGB_NORM_LUMINANCE ($DEFAULT annotation)
            bc_nnodes: 2.0,
            bc_exposure_fusion: 0.0,
            bc_exposure_stops: 1.0, // $DEFAULT
            bc_exposure_bias: 1.0,  // $DEFAULT (the slider's double-click default is 0 — GUI quirk only)
            bc_nodes: identity_nodes_20(),
            // lens.cc defaults where introspection provides one (flags/scale/
            // target_geom); focal/aperture/distance are EXIF-driven in C with
            // no $DEFAULT — see the field docs. The module ships off (no gear
            // selected yet).
            lens_on: false,
            lens_inverse: false,
            lens_modify_flags: 7.0, // DT_IOP_LENS_MODFLAG_ALL
            lens_scale: 1.0,
            lens_focal: 50.0,
            lens_aperture: 3.5,
            lens_distance: 1000.0, // C's no-EXIF fallback (lens.cc:3455)
            lens_target_geom: 1.0, // LF_RECTILINEAR
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
        // All three at 0 rescale to contrast 1 / brightness 0 / saturation 1,
        // which is the identity curve pair and an unchanged a/b.
        let colisa_identity = !self.colisa_on
            || (self.colisa_contrast == 0.0
                && self.colisa_brightness == 0.0
                && self.colisa_saturation == 0.0);
        // Identity when off, or when every slider that can change a pixel is at
        // its neutral. middle_grey and preserve_colors are NOT in this list:
        // both only matter via contrast, which is checked.
        let basicadj_identity = !self.basicadj_on
            || (self.basicadj_black_point == 0.0
                && self.basicadj_exposure == 0.0
                && self.basicadj_hlcompr == 0.0
                && self.basicadj_contrast == 0.0
                && self.basicadj_brightness == 0.0
                && self.basicadj_saturation == 0.0
                && self.basicadj_vibrance == 0.0);
        let vignette_identity = !self.vignette_on
            || (self.vignette_brightness == 0.0 && self.vignette_saturation == 0.0);
        // Lowpass is identity when off. Contrast 1.0 + brightness 0.0 + saturation
        // 1.0 means the LUTs are identity curves and a/b is unscaled, so that is
        // a no-op too; radius doesn't matter then (the blur output is unchanged
        // by the identity LUTs).
        let lowpass_identity = !self.lowpass_on
            || (self.lowpass_contrast == 1.0
                && self.lowpass_brightness == 0.0
                && self.lowpass_saturation == 1.0);
        // Shadhi is identity when off, or when shadows and highlights are both 0
        // (no overlay to blend). whitepoint alone at 0 is already identity
        // (max(1-0/100, 0.01) = 1.0); compress, ccorrect and radius have no effect
        // without a non-zero shadow/highlight to drive them.
        let shadhi_identity = !self.shadhi_on
            || (self.shadhi_shadows == 0.0 && self.shadhi_highlights == 0.0 && self.shadhi_whitepoint == 0.0);
        // Primaries: the matrix is identity when all hues are 0 (no rotation)
        // The gate mirrors `to_pipeline`.
        let primaries_identity = !self.primaries_on || self.primaries_is_neutral();
        // Negadoctor is a film-negative inverter — there is no neutral value
        // while enabled (it always inverts). The gate mirrors `to_pipeline`:
        // only the enable flag matters.
        let negadoctor_identity = !self.negadoctor_on;
        // Tone equalizer: all-zero gains are exp2(0) = 1 at every channel (the
        // fitted curve tracks unity to ≤0.7% RBF residual — see
        // solve_weights_flat_unity_at_default_gains in c41-core), so that is
        // identity regardless of the on flag — but keep the flag in the gate to
        // mirror `to_pipeline`, which only pushes the stage when a gain moved.
        let toneeq_identity = !self.toneeq_on
            || (self.toneeq_noise == 0.0
                && self.toneeq_ultra_deep_blacks == 0.0
                && self.toneeq_deep_blacks == 0.0
                && self.toneeq_blacks == 0.0
                && self.toneeq_shadows == 0.0
                && self.toneeq_midtones == 0.0
                && self.toneeq_highlights == 0.0
                && self.toneeq_whites == 0.0
                && self.toneeq_speculars == 0.0);
        // Color balance RGB is identity when off, or when the mapped params are
        // exactly darktable's neutral edit (the gate mirrors `to_pipeline`,
        // which builds CbRgbParams and compares against `default()`). Note the
        // C would still run its near-no-op gamut map; we skip it like every
        // other identity module.
        let cb_identity = !self.cb_on || self.cb_is_neutral();
        // Filmic RGB is a display transform (a tone curve) — never a no-op while
        // enabled. The gate mirrors `to_pipeline`: only the enable flag matters.
        let filmic_identity = !self.filmic_on;
        // Highlight reconstruction lives in the raw decode, not the stage list:
        // off = the legacy hard-clip-at-white decoder, on = always re-decodes.
        let hl_identity = !self.hl_on;
        // Denoise (profiled): a real edge-preserving smoother — no value is a
        // strict no-op while enabled (even strength 0 leaves the coarse
        // residual path). The gate mirrors `to_pipeline`: only the enable flag.
        let dn_identity = !self.dn_on;
        // Bloom gates on the enable flag alone (like Colorize): while on, the
        // screen blend lifts any pixel whose gathered neighbours pass the
        // threshold, and darktable's own defaults (threshold 90) do real work.
        let bl_identity = !self.bl_on;
        // Tone curve gates on the enable flag alone: while on, even the
        // default identity anchors still run the full LUT + a/b re-derivation
        // round-trip (float noise, if nothing else) — same policy as Bloom.
        let tc_identity = !self.tc_on;
        // RGB curve: identical policy — commit_params is trivial and process()
        // always rebuilds + applies the LUTs while enabled.
        let rc_identity = !self.rc_on;
        // Base curve: same flag-only gate (Bloom/ToneCurve/RgbCurve policy).
        // Note darktable's identity-at-defaults argument doesn't even apply:
        // the shipped "display-referred default" preset is a real tone curve,
        // not the 2-node identity.
        let bc_identity = !self.bc_on;
        // Lens correction: flag-only gate (same policy). A warp is never a
        // no-op while enabled; with gear unresolved the stage is skipped but
        // reporting non-identity only costs one unchanged re-render — it can
        // never mask an applied edit.
        let lens_identity = !self.lens_on;
        exp_identity && vel_identity && split_identity && mono_identity && sigmoid_identity
            && sharpen_identity && vibrance_identity && cc_identity && temp_identity
            && invert_identity && colorize_identity && cc_corr_identity && cz_identity
            && levels_identity && vignette_identity && lowlight_identity
            && gradnd_identity && colisa_identity && basicadj_identity
            && lowpass_identity && shadhi_identity
            && primaries_identity && negadoctor_identity
            && toneeq_identity
            && cb_identity
            && filmic_identity
            && hl_identity
            && dn_identity
            && bl_identity
            && tc_identity
            && rc_identity
            && bc_identity
            && lens_identity
    }

    /// Highlight-reconstruction options for the raw front end; `None` while the
    /// module is off (the legacy hard-clip-at-white decode). The single mapping
    /// site shared by the darkroom preview decode and export, so they can never
    /// disagree about what the controls mean. Non-raw inputs ignore it (no
    /// float pre-demosaic domain there).
    pub fn hl_opts(&self) -> Option<c41_core::iop::highlights::HlOpts> {
        if !self.hl_on {
            return None;
        }
        Some(c41_core::iop::highlights::HlOpts {
            mode: if self.hl_opposed {
                c41_core::iop::highlights::HlMode::Opposed
            } else {
                c41_core::iop::highlights::HlMode::Clip
            },
            clip: self.hl_clip,
        })
    }

    /// The UI fields + resolved gear mapped onto the core's `LensParams` —
    /// the single construction site shared by `to_pipeline_with` and the lens
    /// module's autoscale button, so they can never disagree about what the
    /// controls mean. The camera supplies the crop factor (`p->crop =
    /// cam->CropFactor`, lens.c commit_params); the numeric sliders come from
    /// `self`.
    pub(crate) fn lens_params(
        &self,
        cam: &c41_core::iop::lens::ResolvedCamera,
        lens: &c41_core::iop::lens::ResolvedLens,
    ) -> c41_core::iop::lens::LensParams {
        c41_core::iop::lens::LensParams {
            camera_maker: cam.maker.clone(),
            camera_model: cam.model.clone(),
            lens: lens.model.clone(),
            modify_flags: self.lens_modify_flags as i32,
            inverse: self.lens_inverse,
            scale: self.lens_scale,
            crop: cam.crop_factor,
            focal: self.lens_focal,
            aperture: self.lens_aperture,
            distance: self.lens_distance,
            target_geom: self.lens_target_geom as i32,
        }
    }

    /// The UI fields mapped onto the core's `FilmicParams` — the single
    /// construction site shared by `is_identity` and `to_pipeline`, so they can
    /// never disagree about what the sliders mean.
    fn filmic_params(&self) -> c41_core::iop::filmicrgb::FilmicParams {
        c41_core::iop::filmicrgb::FilmicParams {
            black_point_source: self.filmic_black_point_source,
            white_point_source: self.filmic_white_point_source,
            grey_point_target: 18.45, // not surfaced in the preview UI
            black_point_target: 0.01517634,
            white_point_target: 100.0,
            output_power: self.filmic_output_power,
            latitude: self.filmic_latitude,
            contrast: self.filmic_contrast,
            balance: self.filmic_balance,
            saturation: self.filmic_saturation,
            custom_grey: false,
        }
    }

    /// The UI fields mapped onto the core's `CbRgbParams` — the single
    /// construction site shared by `is_identity`, `cb_is_neutral` and
    /// `to_pipeline`, so they can never disagree about what the sliders mean.
    fn cb_params(&self) -> c41_core::iop::colorbalancergb::CbRgbParams {
        use c41_core::iop::colorbalancergb::{CbRgbParams, SaturationFormula};
        CbRgbParams {
            shadows_y: self.cb_shadows_y,
            shadows_c: self.cb_shadows_c,
            shadows_h: self.cb_shadows_h,
            midtones_y: self.cb_midtones_y,
            midtones_c: self.cb_midtones_c,
            midtones_h: self.cb_midtones_h,
            highlights_y: self.cb_highlights_y,
            highlights_c: self.cb_highlights_c,
            highlights_h: self.cb_highlights_h,
            global_y: self.cb_global_y,
            global_c: self.cb_global_c,
            global_h: self.cb_global_h,
            shadows_weight: self.cb_shadows_weight,
            white_fulcrum: self.cb_white_fulcrum,
            highlights_weight: self.cb_highlights_weight,
            chroma_shadows: self.cb_chroma_shadows,
            chroma_highlights: self.cb_chroma_highlights,
            chroma_global: self.cb_chroma_global,
            chroma_midtones: self.cb_chroma_midtones,
            saturation_global: self.cb_saturation_global,
            saturation_highlights: self.cb_saturation_highlights,
            saturation_midtones: self.cb_saturation_midtones,
            saturation_shadows: self.cb_saturation_shadows,
            hue_angle: self.cb_hue_angle,
            brilliance_global: self.cb_brilliance_global,
            brilliance_highlights: self.cb_brilliance_highlights,
            brilliance_midtones: self.cb_brilliance_midtones,
            brilliance_shadows: self.cb_brilliance_shadows,
            mask_grey_fulcrum: self.cb_mask_grey_fulcrum,
            vibrance: self.cb_vibrance,
            grey_fulcrum: self.cb_grey_fulcrum,
            contrast: self.cb_contrast,
            saturation_formula: if self.cb_formula < 0.5 {
                SaturationFormula::Jzazbz
            } else {
                SaturationFormula::DtUcs
            },
        }
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
            colisa_on: false,
            basicadj_on: false,
            lowpass_on: false,
            shadhi_on: false,
            primaries_on: false,
            negadoctor_on: false,
            toneeq_on: false,
            cb_on: false,
            filmic_on: false,
            hl_on: false,
            dn_on: false,
            bl_on: false,
            tc_on: false,
            rc_on: false,
            bc_on: false,
            lens_on: false,
            ..*self
        }
    }

    /// Whether the color balance RGB params are darktable's neutral edit (the
    /// mapped `CbRgbParams` equals its default — offset 0 / slope 1 / power 1).
    /// Single source of truth for `is_identity`; `to_pipeline` shares the same
    /// comparison through [`Self::cb_params`].
    fn cb_is_neutral(&self) -> bool {
        use c41_core::iop::colorbalancergb::CbRgbParams;
        self.cb_params() == CbRgbParams::default()
    }

    /// Whether the primaries params are at their neutral defaults (all hues 0,
    /// all RGB purities 1.0, achromatic tint purity 0). This is the single source
    /// of truth for the three places that need to know (is_identity, to_pipeline,
    /// and the `default_params_are_neutral` test).
    pub fn primaries_is_neutral(&self) -> bool {
        self.primaries_achromatic_tint_hue == 0.0
            && self.primaries_achromatic_tint_purity == 0.0
            && self.primaries_red_hue == 0.0
            && self.primaries_red_purity == 1.0
            && self.primaries_green_hue == 0.0
            && self.primaries_green_purity == 1.0
            && self.primaries_blue_hue == 0.0
            && self.primaries_blue_purity == 1.0
    }

    /// Read up to 20 interleaved (x, y) anchors starting at float index `base`,
    /// falling back to `dflt` slot-by-slot so short (older-version) blobs keep
    /// their defaults instead of panicking.
    fn decode_nodes(f: &[f32], base: usize, dflt: [(f32, f32); 20]) -> [(f32, f32); 20] {
        let mut nodes = dflt;
        for (k, slot) in nodes.iter_mut().enumerate() {
            if let (Some(&x), Some(&y)) = (f.get(base + 2 * k), f.get(base + 1 + 2 * k)) {
                *slot = (x, y);
            }
        }
        nodes
    }

    /// Serialise to a compact, versioned little-endian blob for DB persistence:
    /// `[version, N×bool(u8), M×f32_le]` (see [`ENCODED_LEN`]). Decoded by
    /// [`PreviewParams::decode`].
    /// This is c41-ui's own layout (NOT a C IOP `op_params`), stored under a
    /// synthetic operation name the C reader ignores.
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(ENCODED_LEN);
        v.push(ENCODE_VERSION);
        for b in [self.exposure_on, self.velvia_on, self.split_on, self.mono_on, self.sigmoid_on, self.sharpen_on, self.vibrance_on, self.color_contrast_on, self.temperature_on, self.invert_on, self.colorize_on, self.color_correction_on, self.colorzones_on, self.levels_on, self.vignette_on, self.lowlight_on, self.gradnd_on, self.colisa_on, self.basicadj_on, self.lowpass_on, self.shadhi_on, self.primaries_on, self.negadoctor_on, self.toneeq_on, self.cb_on, self.filmic_on, self.hl_on, self.hl_opposed, self.dn_on, self.dn_mode_y0u0v0, self.bl_on, self.tc_on, self.tc_unbound, self.rc_on, self.bc_on, self.lens_on, self.lens_inverse] {
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
            self.colisa_contrast, self.colisa_brightness, self.colisa_saturation,
            self.basicadj_black_point, self.basicadj_exposure,
            self.basicadj_hlcompr, self.basicadj_hlcomprthresh,
            self.basicadj_contrast, self.basicadj_preserve_colors,
            self.basicadj_middle_grey, self.basicadj_brightness,
            self.basicadj_saturation, self.basicadj_vibrance,
            self.lowpass_radius, self.lowpass_contrast, self.lowpass_brightness, self.lowpass_saturation,
            self.shadhi_shadows, self.shadhi_highlights, self.shadhi_whitepoint,
            self.shadhi_radius, self.shadhi_compress,
            self.shadhi_shadows_ccorrect, self.shadhi_highlights_ccorrect,
            self.primaries_achromatic_tint_hue, self.primaries_achromatic_tint_purity,
            self.primaries_red_hue, self.primaries_red_purity,
            self.primaries_green_hue, self.primaries_green_purity,
            self.primaries_blue_hue, self.primaries_blue_purity,
            self.negadoctor_film_stock,
            self.negadoctor_dmin_r, self.negadoctor_dmin_g, self.negadoctor_dmin_b,
            self.negadoctor_wb_high_r, self.negadoctor_wb_high_g, self.negadoctor_wb_high_b,
            self.negadoctor_wb_low_r, self.negadoctor_wb_low_g, self.negadoctor_wb_low_b,
            self.negadoctor_d_max, self.negadoctor_offset,
            self.negadoctor_black, self.negadoctor_gamma,
            self.negadoctor_soft_clip, self.negadoctor_exposure,
            self.toneeq_noise, self.toneeq_ultra_deep_blacks,
            self.toneeq_deep_blacks, self.toneeq_blacks,
            self.toneeq_shadows, self.toneeq_midtones,
            self.toneeq_highlights, self.toneeq_whites,
            self.toneeq_speculars,
            // Color balance RGB — field order mirrors
            // dt_iop_colorbalancergb_params_t (v5), so the blob reads like the
            // C struct.
            self.cb_shadows_y, self.cb_shadows_c, self.cb_shadows_h,
            self.cb_midtones_y, self.cb_midtones_c, self.cb_midtones_h,
            self.cb_highlights_y, self.cb_highlights_c, self.cb_highlights_h,
            self.cb_global_y, self.cb_global_c, self.cb_global_h,
            self.cb_shadows_weight, self.cb_white_fulcrum, self.cb_highlights_weight,
            self.cb_chroma_shadows, self.cb_chroma_highlights,
            self.cb_chroma_global, self.cb_chroma_midtones,
            self.cb_saturation_global, self.cb_saturation_highlights,
            self.cb_saturation_midtones, self.cb_saturation_shadows,
            self.cb_hue_angle,
            self.cb_brilliance_global, self.cb_brilliance_highlights,
            self.cb_brilliance_midtones, self.cb_brilliance_shadows,
            self.cb_mask_grey_fulcrum,
            self.cb_vibrance, self.cb_grey_fulcrum, self.cb_contrast,
            self.cb_formula,
            // Filmic RGB — field order mirrors dt_iop_filmicrgb_params_t.
            self.filmic_black_point_source, self.filmic_white_point_source,
            self.filmic_output_power, self.filmic_latitude,
            self.filmic_contrast, self.filmic_balance, self.filmic_saturation,
            // Highlight reconstruction (m4-119).
            self.hl_clip,
            // Denoise (profiled) (m4-120).
            self.dn_strength,
            self.dn_shadows,
            self.dn_bias,
            // Bloom (m4-121).
            self.bl_size,
            self.bl_threshold,
            self.bl_strength,
            // Tone curve (m4-122): scalar controls, then the L anchors as
            // interleaved (x0, y0, x1, y1, …) pairs — 20 slots always written
            // so the layout stays fixed.
            self.tc_type,
            self.tc_autoscale,
            self.tc_preserve,
            self.tc_nnodes,
        ] {
            v.extend_from_slice(&f.to_le_bytes());
        }
        // Tone curve L anchors (m4-122): interleaved (x0, y0, x1, y1, …) —
        // all 20 slots always written so the layout stays fixed.
        for &(x, y) in &self.tc_nodes_l {
            v.extend_from_slice(&x.to_le_bytes());
            v.extend_from_slice(&y.to_le_bytes());
        }
        // RGB curve (m4-123): per-channel spline types + mode scalars (floats
        // 264–271), written as their own block AFTER the tone-curve anchors —
        // strictly append-only, so v23 blobs keep reading tc anchors at 224.
        for s in [
            self.rc_type_r,
            self.rc_type_g,
            self.rc_type_b,
            self.rc_autoscale,
            self.rc_preserve,
            self.rc_nnodes_r,
            self.rc_nnodes_g,
            self.rc_nnodes_b,
        ] {
            v.extend_from_slice(&s.to_le_bytes());
        }
        // RGB curve anchors (m4-123): R, then G, then B — each 20 interleaved
        // (x, y) pairs, all slots always written so the layout stays fixed.
        for nodes in [&self.rc_nodes_r, &self.rc_nodes_g, &self.rc_nodes_b] {
            for &(x, y) in nodes {
                v.extend_from_slice(&x.to_le_bytes());
                v.extend_from_slice(&y.to_le_bytes());
            }
        }
        // Base curve (v25): channel-0 scalars, then its 20 interleaved pairs —
        // appended after the RGB-curve block, keeping the layout append-only.
        for s in [
            self.bc_type,
            self.bc_preserve,
            self.bc_nnodes,
            self.bc_exposure_fusion,
            self.bc_exposure_stops,
            self.bc_exposure_bias,
        ] {
            v.extend_from_slice(&s.to_le_bytes());
        }
        for &(x, y) in &self.bc_nodes {
            v.extend_from_slice(&x.to_le_bytes());
            v.extend_from_slice(&y.to_le_bytes());
        }
        // Lens correction (v26): combo/scalars appended after the base-curve
        // block. The gear identity (camera/lens names) is NOT here — it lives
        // in `main.darkroom_lens_choice` (strings can't join the f32 blob).
        for s in [
            self.lens_modify_flags,
            self.lens_scale,
            self.lens_focal,
            self.lens_aperture,
            self.lens_distance,
            self.lens_target_geom,
        ] {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    /// Inverse of [`PreviewParams::encode`]. Returns `None` for a blob whose
    /// version byte or length doesn't match any known layout, so the caller falls
    /// back to defaults rather than loading garbage.
    ///
    /// **Backward compatible.** Because the layout is strictly append-only (new
    /// modules append bool + f32 fields at the end), an older blob decodes with
    /// the new fields defaulted — so bumping `ENCODE_VERSION` does NOT silently
    /// delete saved styles or history entries.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let n_bools = PARAMS_LAYOUTS
            .iter()
            .find(|(v, nb, nf)| bytes.first() == Some(v) && bytes.len() == 1 + nb + nf * 4)
            .map(|(_, nb, _)| *nb)?;
        let bools = &bytes[1..1 + n_bools];
        let f: Vec<f32> = bytes[1 + n_bools..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // v13-and-earlier blobs lack the shadhi fields (appended in v14).
        // Use .get() with defaults from PreviewParams::default() so they decode
        // cleanly instead of panicking on a short slice — and so the fallback
        // values can never drift from Default.
        let d = Self::default();
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
            colisa_on: bools[17] != 0,
            colisa_contrast: f[116], colisa_brightness: f[117], colisa_saturation: f[118],
            basicadj_on: bools[18] != 0,
            basicadj_black_point: f[119], basicadj_exposure: f[120],
            basicadj_hlcompr: f[121], basicadj_hlcomprthresh: f[122],
            basicadj_contrast: f[123], basicadj_preserve_colors: f[124],
            basicadj_middle_grey: f[125], basicadj_brightness: f[126],
            basicadj_saturation: f[127], basicadj_vibrance: f[128],
            lowpass_on: bools[19] != 0,
            lowpass_radius: f[129], lowpass_contrast: f[130],
            lowpass_brightness: f[131], lowpass_saturation: f[132],
            shadhi_on: bools.get(20).map_or(d.shadhi_on, |&b| b != 0),
            primaries_on: bools.get(21).map_or(d.primaries_on, |&b| b != 0),
            shadhi_shadows: f.get(133).copied().unwrap_or(d.shadhi_shadows),
            shadhi_highlights: f.get(134).copied().unwrap_or(d.shadhi_highlights),
            shadhi_whitepoint: f.get(135).copied().unwrap_or(d.shadhi_whitepoint),
            shadhi_radius: f.get(136).copied().unwrap_or(d.shadhi_radius),
            shadhi_compress: f.get(137).copied().unwrap_or(d.shadhi_compress),
            shadhi_shadows_ccorrect: f.get(138).copied().unwrap_or(d.shadhi_shadows_ccorrect),
            shadhi_highlights_ccorrect: f.get(139).copied().unwrap_or(d.shadhi_highlights_ccorrect),
            primaries_achromatic_tint_hue: f.get(140).copied().unwrap_or(d.primaries_achromatic_tint_hue),
            primaries_achromatic_tint_purity: f.get(141).copied().unwrap_or(d.primaries_achromatic_tint_purity),
            primaries_red_hue: f.get(142).copied().unwrap_or(d.primaries_red_hue),
            primaries_red_purity: f.get(143).copied().unwrap_or(d.primaries_red_purity),
            primaries_green_hue: f.get(144).copied().unwrap_or(d.primaries_green_hue),
            primaries_green_purity: f.get(145).copied().unwrap_or(d.primaries_green_purity),
            primaries_blue_hue: f.get(146).copied().unwrap_or(d.primaries_blue_hue),
            primaries_blue_purity: f.get(147).copied().unwrap_or(d.primaries_blue_purity),
            negadoctor_on: bools.get(22).map_or(d.negadoctor_on, |&b| b != 0),
            negadoctor_film_stock: f.get(148).copied().unwrap_or(d.negadoctor_film_stock),
            negadoctor_dmin_r: f.get(149).copied().unwrap_or(d.negadoctor_dmin_r),
            negadoctor_dmin_g: f.get(150).copied().unwrap_or(d.negadoctor_dmin_g),
            negadoctor_dmin_b: f.get(151).copied().unwrap_or(d.negadoctor_dmin_b),
            negadoctor_wb_high_r: f.get(152).copied().unwrap_or(d.negadoctor_wb_high_r),
            negadoctor_wb_high_g: f.get(153).copied().unwrap_or(d.negadoctor_wb_high_g),
            negadoctor_wb_high_b: f.get(154).copied().unwrap_or(d.negadoctor_wb_high_b),
            negadoctor_wb_low_r: f.get(155).copied().unwrap_or(d.negadoctor_wb_low_r),
            negadoctor_wb_low_g: f.get(156).copied().unwrap_or(d.negadoctor_wb_low_g),
            negadoctor_wb_low_b: f.get(157).copied().unwrap_or(d.negadoctor_wb_low_b),
            negadoctor_d_max: f.get(158).copied().unwrap_or(d.negadoctor_d_max),
            negadoctor_offset: f.get(159).copied().unwrap_or(d.negadoctor_offset),
            negadoctor_black: f.get(160).copied().unwrap_or(d.negadoctor_black),
            negadoctor_gamma: f.get(161).copied().unwrap_or(d.negadoctor_gamma),
            negadoctor_soft_clip: f.get(162).copied().unwrap_or(d.negadoctor_soft_clip),
            negadoctor_exposure: f.get(163).copied().unwrap_or(d.negadoctor_exposure),
            toneeq_on: bools.get(23).map_or(d.toneeq_on, |&b| b != 0),
            toneeq_noise: f.get(164).copied().unwrap_or(d.toneeq_noise),
            toneeq_ultra_deep_blacks: f.get(165).copied().unwrap_or(d.toneeq_ultra_deep_blacks),
            toneeq_deep_blacks: f.get(166).copied().unwrap_or(d.toneeq_deep_blacks),
            toneeq_blacks: f.get(167).copied().unwrap_or(d.toneeq_blacks),
            toneeq_shadows: f.get(168).copied().unwrap_or(d.toneeq_shadows),
            toneeq_midtones: f.get(169).copied().unwrap_or(d.toneeq_midtones),
            toneeq_highlights: f.get(170).copied().unwrap_or(d.toneeq_highlights),
            toneeq_whites: f.get(171).copied().unwrap_or(d.toneeq_whites),
            toneeq_speculars: f.get(172).copied().unwrap_or(d.toneeq_speculars),
            cb_on: bools.get(24).map_or(d.cb_on, |&b| b != 0),
            cb_shadows_y: f.get(173).copied().unwrap_or(d.cb_shadows_y),
            cb_shadows_c: f.get(174).copied().unwrap_or(d.cb_shadows_c),
            cb_shadows_h: f.get(175).copied().unwrap_or(d.cb_shadows_h),
            cb_midtones_y: f.get(176).copied().unwrap_or(d.cb_midtones_y),
            cb_midtones_c: f.get(177).copied().unwrap_or(d.cb_midtones_c),
            cb_midtones_h: f.get(178).copied().unwrap_or(d.cb_midtones_h),
            cb_highlights_y: f.get(179).copied().unwrap_or(d.cb_highlights_y),
            cb_highlights_c: f.get(180).copied().unwrap_or(d.cb_highlights_c),
            cb_highlights_h: f.get(181).copied().unwrap_or(d.cb_highlights_h),
            cb_global_y: f.get(182).copied().unwrap_or(d.cb_global_y),
            cb_global_c: f.get(183).copied().unwrap_or(d.cb_global_c),
            cb_global_h: f.get(184).copied().unwrap_or(d.cb_global_h),
            cb_shadows_weight: f.get(185).copied().unwrap_or(d.cb_shadows_weight),
            cb_white_fulcrum: f.get(186).copied().unwrap_or(d.cb_white_fulcrum),
            cb_highlights_weight: f.get(187).copied().unwrap_or(d.cb_highlights_weight),
            cb_chroma_shadows: f.get(188).copied().unwrap_or(d.cb_chroma_shadows),
            cb_chroma_highlights: f.get(189).copied().unwrap_or(d.cb_chroma_highlights),
            cb_chroma_global: f.get(190).copied().unwrap_or(d.cb_chroma_global),
            cb_chroma_midtones: f.get(191).copied().unwrap_or(d.cb_chroma_midtones),
            cb_saturation_global: f.get(192).copied().unwrap_or(d.cb_saturation_global),
            cb_saturation_highlights: f.get(193).copied().unwrap_or(d.cb_saturation_highlights),
            cb_saturation_midtones: f.get(194).copied().unwrap_or(d.cb_saturation_midtones),
            cb_saturation_shadows: f.get(195).copied().unwrap_or(d.cb_saturation_shadows),
            cb_hue_angle: f.get(196).copied().unwrap_or(d.cb_hue_angle),
            cb_brilliance_global: f.get(197).copied().unwrap_or(d.cb_brilliance_global),
            cb_brilliance_highlights: f.get(198).copied().unwrap_or(d.cb_brilliance_highlights),
            cb_brilliance_midtones: f.get(199).copied().unwrap_or(d.cb_brilliance_midtones),
            cb_brilliance_shadows: f.get(200).copied().unwrap_or(d.cb_brilliance_shadows),
            cb_mask_grey_fulcrum: f.get(201).copied().unwrap_or(d.cb_mask_grey_fulcrum),
            cb_vibrance: f.get(202).copied().unwrap_or(d.cb_vibrance),
            cb_grey_fulcrum: f.get(203).copied().unwrap_or(d.cb_grey_fulcrum),
            cb_contrast: f.get(204).copied().unwrap_or(d.cb_contrast),
            cb_formula: f.get(205).copied().unwrap_or(d.cb_formula),
            filmic_on: bools.get(25).map_or(d.filmic_on, |&b| b != 0),
            filmic_black_point_source: f.get(206).copied().unwrap_or(d.filmic_black_point_source),
            filmic_white_point_source: f.get(207).copied().unwrap_or(d.filmic_white_point_source),
            filmic_output_power: f.get(208).copied().unwrap_or(d.filmic_output_power),
            filmic_latitude: f.get(209).copied().unwrap_or(d.filmic_latitude),
            filmic_contrast: f.get(210).copied().unwrap_or(d.filmic_contrast),
            filmic_balance: f.get(211).copied().unwrap_or(d.filmic_balance),
            filmic_saturation: f.get(212).copied().unwrap_or(d.filmic_saturation),
            hl_on: bools.get(26).map_or(d.hl_on, |&b| b != 0),
            hl_opposed: bools.get(27).map_or(d.hl_opposed, |&b| b != 0),
            hl_clip: f.get(213).copied().unwrap_or(d.hl_clip),
            dn_on: bools.get(28).map_or(d.dn_on, |&b| b != 0),
            dn_mode_y0u0v0: bools.get(29).map_or(d.dn_mode_y0u0v0, |&b| b != 0),
            dn_strength: f.get(214).copied().unwrap_or(d.dn_strength),
            dn_shadows: f.get(215).copied().unwrap_or(d.dn_shadows),
            dn_bias: f.get(216).copied().unwrap_or(d.dn_bias),
            bl_on: bools.get(30).map_or(d.bl_on, |&b| b != 0),
            bl_size: f.get(217).copied().unwrap_or(d.bl_size),
            bl_threshold: f.get(218).copied().unwrap_or(d.bl_threshold),
            bl_strength: f.get(219).copied().unwrap_or(d.bl_strength),
            tc_on: bools.get(31).map_or(d.tc_on, |&b| b != 0),
            tc_unbound: bools.get(32).map_or(d.tc_unbound, |&b| b != 0),
            tc_type: f.get(220).copied().unwrap_or(d.tc_type),
            tc_autoscale: f.get(221).copied().unwrap_or(d.tc_autoscale),
            tc_preserve: f.get(222).copied().unwrap_or(d.tc_preserve),
            tc_nnodes: f.get(223).copied().unwrap_or(d.tc_nnodes),
            // L anchors start at float 224 as 40 interleaved (x, y) values.
            // Older blobs (v22) have none of these entries — every `.get`
            // misses and the slot keeps the C-default identity anchor.
            tc_nodes_l: Self::decode_nodes(&f, 224, d.tc_nodes_l),
            // RGB curve (m4-123): scalars at floats 264–271, then R/G/B node
            // arrays of 40 interleaved values each (272 / 312 / 352). v23-and-
            // earlier blobs end before any of this — every `.get` misses and
            // each slot keeps its C-default value.
            rc_on: bools.get(33).map_or(d.rc_on, |&b| b != 0),
            rc_type_r: f.get(264).copied().unwrap_or(d.rc_type_r),
            rc_type_g: f.get(265).copied().unwrap_or(d.rc_type_g),
            rc_type_b: f.get(266).copied().unwrap_or(d.rc_type_b),
            rc_autoscale: f.get(267).copied().unwrap_or(d.rc_autoscale),
            rc_preserve: f.get(268).copied().unwrap_or(d.rc_preserve),
            rc_nnodes_r: f.get(269).copied().unwrap_or(d.rc_nnodes_r),
            rc_nnodes_g: f.get(270).copied().unwrap_or(d.rc_nnodes_g),
            rc_nnodes_b: f.get(271).copied().unwrap_or(d.rc_nnodes_b),
            rc_nodes_r: Self::decode_nodes(&f, 272, d.rc_nodes_r),
            rc_nodes_g: Self::decode_nodes(&f, 312, d.rc_nodes_g),
            rc_nodes_b: Self::decode_nodes(&f, 352, d.rc_nodes_b),
            // Base curve (m4-124): scalars at floats 392–397, then the
            // channel-0 node array (40 interleaved values) at 398. v24-and-
            // earlier blobs end before any of this — every `.get` misses and
            // each slot keeps its C-default value.
            bc_on: bools.get(34).map_or(d.bc_on, |&b| b != 0),
            bc_type: f.get(392).copied().unwrap_or(d.bc_type),
            bc_preserve: f.get(393).copied().unwrap_or(d.bc_preserve),
            bc_nnodes: f.get(394).copied().unwrap_or(d.bc_nnodes),
            bc_exposure_fusion: f.get(395).copied().unwrap_or(d.bc_exposure_fusion),
            bc_exposure_stops: f.get(396).copied().unwrap_or(d.bc_exposure_stops),
            bc_exposure_bias: f.get(397).copied().unwrap_or(d.bc_exposure_bias),
            bc_nodes: Self::decode_nodes(&f, 398, d.bc_nodes),
            // Lens correction (m4-130): bools 35–36, floats 438–443. v25-and-
            // earlier blobs end before any of this — defaults hold.
            lens_on: bools.get(35).map_or(d.lens_on, |&b| b != 0),
            lens_inverse: bools.get(36).map_or(d.lens_inverse, |&b| b != 0),
            lens_modify_flags: f.get(438).copied().unwrap_or(d.lens_modify_flags),
            lens_scale: f.get(439).copied().unwrap_or(d.lens_scale),
            lens_focal: f.get(440).copied().unwrap_or(d.lens_focal),
            lens_aperture: f.get(441).copied().unwrap_or(d.lens_aperture),
            lens_distance: f.get(442).copied().unwrap_or(d.lens_distance),
            lens_target_geom: f.get(443).copied().unwrap_or(d.lens_target_geom),
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
        self.to_pipeline_with(space, scale, None)
    }

    /// [`Self::to_pipeline`] with resolved lens-correction gear. The darkroom
    /// view passes the per-image `(camera, lens)` pair it resolved against the
    /// lensfun database (cached on the preview context); export passes the same
    /// pair so full-res output matches the preview. `None` simply omits the
    /// stage — as does `lens_on` without gear (darktable shows "no data" and
    /// does the same).
    ///
    /// This is the variant for buffers where **geometry has not run yet** —
    /// today the non-raw funnels (`apply_pipeline_gear`,
    /// [`Self::apply_pipeline_rgb16_gear`]), whose 8-bit sources are never
    /// cropped, so a full-frame warp inside the pipeline is exactly
    /// darktable's pre-crop placement. The raw funnels must use
    /// [`Self::to_pipeline_lens_preapplied`] instead: their geometry pass has
    /// already cropped the frame by the time the pipeline runs (m4-131).
    pub fn to_pipeline_with(
        &self,
        space: ColorSpace,
        scale: f32,
        lens_gear: Option<&(
            c41_core::iop::lens::ResolvedCamera,
            c41_core::iop::lens::ResolvedLens,
        )>,
    ) -> Pipeline {
        self.to_pipeline_inner(space, scale, lens_gear, true)
    }

    /// [`Self::to_pipeline_with`] for buffers the lens warp has **already been
    /// applied to** upstream — the raw preview/export paths run
    /// [`apply_lens_prepass`] on the full-frame decoded buffer *before* the
    /// crop/straighten pass (darktable runs lens at iop_order 13, before crop),
    /// so emitting the stage here would re-warp the already-cropped frame
    /// around a shifted centre and vignette against the wrong frame edges.
    pub fn to_pipeline_lens_preapplied(
        &self,
        space: ColorSpace,
        scale: f32,
        lens_gear: Option<&(
            c41_core::iop::lens::ResolvedCamera,
            c41_core::iop::lens::ResolvedLens,
        )>,
    ) -> Pipeline {
        self.to_pipeline_inner(space, scale, lens_gear, false)
    }

    /// Shared builder; `lens_stage` selects whether the lens-correction stage
    /// is emitted (see the two public wrappers for which callers want which).
    fn to_pipeline_inner(
        &self,
        space: ColorSpace,
        scale: f32,
        lens_gear: Option<&(
            c41_core::iop::lens::ResolvedCamera,
            c41_core::iop::lens::ResolvedLens,
        )>,
        lens_stage: bool,
    ) -> Pipeline {
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
        // Denoise (profiled) (denoiseprofile.c wavelets mode, iop_order.c
        // v50_order pos 9/10 — immediately after demosaic 8, well before
        // exposure 21). Noise estimation and shrinkage must see scene-linear,
        // un-tone-mapped data, so it runs here like every other scene-referred
        // stage. On-by-itself emits the stage (on is the gate, mirroring
        // is_identity: even strength 0 leaves the coarse-residual path).
        // Deviations from the C tracked in c41-core::denoiseprofile (generic
        // Poissonian profile, wb=[1,1,1] since the buffer arrives post-WB).
        if self.dn_on {
            p.push(Stage::DenoiseProfile {
                strength: self.dn_strength,
                shadows: self.dn_shadows,
                bias: self.dn_bias,
                mode_y0u0v0: self.dn_mode_y0u0v0,
            });
        }
        // Lens correction (lens.c, iop_order.c v50_order pos 13 — after
        // denoiseprofile 9/10, before exposure 21): a coordinate warp +
        // per-pixel vignetting gain driven by the lensfun calibration for the
        // selected camera/lens pair. The gear identity is NOT part of
        // PreviewParams (strings can't join the f32 blob) — it lives in
        // `main.darkroom_lens_choice` and arrives here pre-resolved; the camera
        // crop factor rides along at build time (`p->crop = cam->CropFactor`,
        // lens.c commit_params).
        //
        // Only for pipelines whose input geometry has NOT been cropped yet —
        // see [`Self::to_pipeline_with`] vs [`Self::to_pipeline_lens_preapplied`].
        if lens_stage && self.lens_on {
            if let Some((cam, lens)) = lens_gear {
                p.push(Stage::LensCorrection {
                    lens: lens.clone(),
                    params: self.lens_params(cam, lens),
                });
            }
        }
        if self.exposure_on && (self.ev != 0.0 || self.black != 0.0) {
            p.push(Stage::Exposure { black: self.black, scale: 2.0f32.powf(self.ev) });
        }
        // Tone equalizer (toneequal.c, iop_order.c v50_order pos 24.0 — "last
        // module that need enlarged roi_in", after exposure 21 and BEFORE
        // graduatednd 25 / the channelmixerrgb·negadoctor·primaries group 28.5).
        // Scene-referred tone mapping by exposure channel: nine gains (EV
        // offsets, −8…0 EV) are least-squares-fitted to a Gaussian RBF and
        // applied per pixel as a correction looked up at that pixel's own
        // luminance. The stage carries the raw gains; the solve + LUT build
        // happen in `Stage::apply`, memoised.
        //
        // The gate mirrors `is_identity`: all-zero gains are exp2(0) = 1 at
        // every channel (fitted curve ≈1, ≤0.7% RBF residual) — a flat unity
        // correction — so they are skipped.
        if self.toneeq_on
            && (self.toneeq_noise != 0.0
                || self.toneeq_ultra_deep_blacks != 0.0
                || self.toneeq_deep_blacks != 0.0
                || self.toneeq_blacks != 0.0
                || self.toneeq_shadows != 0.0
                || self.toneeq_midtones != 0.0
                || self.toneeq_highlights != 0.0
                || self.toneeq_whites != 0.0
                || self.toneeq_speculars != 0.0)
        {
            p.push(Stage::ToneEqual {
                gains: [
                    self.toneeq_noise,
                    self.toneeq_ultra_deep_blacks,
                    self.toneeq_deep_blacks,
                    self.toneeq_blacks,
                    self.toneeq_shadows,
                    self.toneeq_midtones,
                    self.toneeq_highlights,
                    self.toneeq_whites,
                    self.toneeq_speculars,
                ],
            });
        }
        // Graduated ND (iop_order.c v50_order pos 25 — scene-referred, after
        // the tone equalizer 24 and before the channel mix 28.5). Early
        // placement is correct: it is an optical filter, modelling glass in
        // front of the lens, so it belongs on linear scene data before any tone
        // or colour work. Density 0 is exp2(0) = 1 everywhere, a true no-op, so
        // it is skipped. The geometry depends on the buffer size and is derived
        // in Stage::apply, not here.
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
        // Negadoctor (negadoctor.c, iop_order.c v50_order pos 28.5 — after
        // graduatednd 25, alongside channelmixerrgb/primaries at the same
        // position). Display-referred
        // film-negative inversion via Cineon-style log-density: undoes the scanner's
        // log-density exposure and simulates print-on-paper (gamma, black, soft-clip).
        // Runs on linear RGB before any colour-space adjustment.
        //
        // commit_params logic (src/iop/negadoctor.c:239-267) is replicated here:
        //   wb_high[c] = wb_high[c] / D_max        (premultiply to spare one div/pixel)
        //   offset[c]  = wb_high[c] * offset * wb_low[c]
        //   Dmin: monochrome collapse (film_stock == NB) → all channels = Dmin[0]
        //   black = -exposure * (1 + black)         (arithmetic trick for FMA)
        //   soft_clip_comp = 1 - soft_clip
        if self.negadoctor_on {
            let film_stock_nb = (self.negadoctor_film_stock as i32) == 0; // DT_FILMSTOCK_NB = 0
            // Defensive floor: darktable's slider enforces $MIN 0.1, but a
            // loaded style or programmatic value could be zero — guard the
            // division in wb_high above.
            let d_max = self.negadoctor_d_max.max(f32::MIN_POSITIVE);
            let negadoctor_wb_high_div = [
                self.negadoctor_wb_high_r / d_max,
                self.negadoctor_wb_high_g / d_max,
                self.negadoctor_wb_high_b / d_max,
                1.0,
            ];
            let negadoctor_offset = [
                self.negadoctor_wb_high_r * self.negadoctor_offset * self.negadoctor_wb_low_r,
                self.negadoctor_wb_high_g * self.negadoctor_offset * self.negadoctor_wb_low_g,
                self.negadoctor_wb_high_b * self.negadoctor_offset * self.negadoctor_wb_low_b,
                0.0,
            ];
            let dmin = if film_stock_nb {
                let mono = self.negadoctor_dmin_r;
                [mono, mono, mono, 1.0]
            } else {
                [self.negadoctor_dmin_r, self.negadoctor_dmin_g, self.negadoctor_dmin_b, 1.0]
            };
            p.push(Stage::Negadoctor {
                dmin,
                wb_high: negadoctor_wb_high_div,
                offset: negadoctor_offset,
                black: -self.negadoctor_exposure * (1.0 + self.negadoctor_black),
                gamma: self.negadoctor_gamma,
                soft_clip: self.negadoctor_soft_clip,
                soft_clip_comp: 1.0 - self.negadoctor_soft_clip,
                exposure: self.negadoctor_exposure,
            });
        }
        // Primaries (primaries.c, iop_order.c v50_order pos 28.5 — between
        // graduatednd 28.0 and lowpass 33.0; shares 28.5 with channelmixerrgb,
        // which the v50 table lists first). Scene-referred colour-space
        // adjustment: rotates and scales the working-space primaries. The 4×4
        // matrix is pre-computed here from the 8 UI params (hue in degrees →
        // radians, purity as multiplier). The gate mirrors `is_identity`: off,
        // or at the neutral defaults.
        if self.primaries_on && !self.primaries_is_neutral()
        {
            p.push(Stage::Primaries {
                matrix: primaries::compute_matrix(
                    space,
                    self.primaries_achromatic_tint_hue.to_radians(),
                    self.primaries_achromatic_tint_purity,
                    self.primaries_red_hue.to_radians(),
                    self.primaries_red_purity,
                    self.primaries_green_hue.to_radians(),
                    self.primaries_green_purity,
                    self.primaries_blue_hue.to_radians(),
                    self.primaries_blue_purity,
                ),
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
        // Basic adjustments (iop_order.c pos 40, between channelmixer 39 and
        // colorbalance 41) — scene-referred, so it lands well before the tone
        // map, unlike colisa 47 which is display-referred.
        //
        // The gate mirrors `is_identity`: middle_grey and preserve_colors cannot
        // move a pixel on their own, so they are not in it. Without that, a user
        // who only nudged middle_grey would add a stage that does nothing but
        // cost a full-buffer pass.
        if self.basicadj_on
            && (self.basicadj_black_point != 0.0
                || self.basicadj_exposure != 0.0
                || self.basicadj_hlcompr != 0.0
                || self.basicadj_contrast != 0.0
                || self.basicadj_brightness != 0.0
                || self.basicadj_saturation != 0.0
                || self.basicadj_vibrance != 0.0)
        {
            p.push(Stage::Basicadj {
                black_point: self.basicadj_black_point,
                exposure: self.basicadj_exposure,
                hlcompr: self.basicadj_hlcompr,
                hlcomprthresh: self.basicadj_hlcomprthresh,
                contrast: self.basicadj_contrast,
                preserve_colors: self.basicadj_preserve_colors as i32,
                middle_grey: self.basicadj_middle_grey,
                brightness: self.basicadj_brightness,
                saturation: self.basicadj_saturation,
                vibrance: self.basicadj_vibrance,
                space,
            });
        }
        // Color balance RGB (colorbalancergb.c, iop_order.c v50_order pos 41.5
        // — after basicadj 40.0, before rgblevels 43.0). Scene-referred grading
        // in Filmlight's Yrg space with perceptual saturation/brilliance in
        // dt-UCS (default) or JzAzBz. The whole commit_params derivation — zone
        // vectors, weights, fulcrums and the hue-indexed 512-entry gamut LUT
        // (the dt-UCS build alone marches the RGB gamut boundary 25 600 times)
        // — happens here, ONCE per render; the finished `CbRgbData` travels in
        // the stage so each per-band apply call is pure pixel math.
        //
        // The gate mirrors `is_identity`: darktable's default params are a
        // neutral edit, so an enabled module at its defaults emits no stage.
        // (The C would still run its near-no-op gamut map on every pixel; we
        // skip it like every other identity module.)
        if self.cb_on && !self.cb_is_neutral() {
            // The gamut-LUT matrix and the stage's pixel-loop converters must
            // be the same working space: raw previews grade Rec.2020, non-raw
            // linear sRGB. Mismatched primaries would silently shift hues.
            let lut_matrix_t = match space {
                ColorSpace::Rec2020 => &c41_core::color::REC2020_TO_XYZ_D65_T4,
                ColorSpace::LinearSrgb => &c41_core::color::SRGB_TO_XYZ_D65_T4,
            };
            p.push(Stage::ColorBalanceRgb {
                data: Box::new(c41_core::iop::colorbalancergb::CbRgbData::from_params(
                    &self.cb_params(),
                    lut_matrix_t,
                )),
                space,
            });
        }
        // RGB curve (rgbcurve.c, iop_order.c v50_order pos 42.0 — between
        // colorbalancergb 41.5 and rgblevels 43.0, i.e. BEFORE the whole
        // display-referred tone-mapping cluster; the C comment calls it "a
        // really versatile way to edit colour in scene-referred AND
        // display-referred workflow"). Per-channel LUTs built from the R/G/B
        // anchors through the V1 `curve_tools` sampler (the same machinery
        // darktable's `dt_draw_curve_calc_values` uses), so the drawn curve IS
        // the applied curve. Unlike tonecurve there is no autoscale
        // re-derivation: commit_params is trivial in C and the tables are built
        // in process(), exactly like `rgbcurve::build_luts` does. The stage
        // applies its LUTs directly on the working RGB lanes (C
        // default_colorspace is IOP_CS_RGB) — no Lab sandwich. (An earlier cut
        // of this module placed it after levels, citing pos 50.5 — that entry
        // is in legacy_order, not v50_order.)
        if self.rc_on {
            // Keep the anchors x-sorted with endpoints pinned per channel: the
            // editor maintains this invariant, but a decoded blob might not.
            let pinned = |count: f32, nodes_in: &[(f32, f32); 20]| -> Vec<(f32, f32)> {
                let nnodes = (count.round() as usize).clamp(2, 20);
                let mut nodes = *nodes_in;
                nodes[..nnodes]
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                nodes[0].0 = 0.0;
                nodes[nnodes - 1].0 = 1.0;
                nodes[..nnodes].to_vec()
            };
            let luts = c41_core::iop::rgbcurve::build_luts(
                &pinned(self.rc_nnodes_r, &self.rc_nodes_r),
                self.rc_type_r as i32 as u32,
                &pinned(self.rc_nnodes_g, &self.rc_nodes_g),
                self.rc_type_g as i32 as u32,
                &pinned(self.rc_nnodes_b, &self.rc_nodes_b),
                self.rc_type_b as i32 as u32,
            );
            p.push(Stage::RgbCurve {
                table_r: luts.table_r,
                table_g: luts.table_g,
                table_b: luts.table_b,
                coeffs: luts.unbounded_coeffs,
                autoscale: self.rc_autoscale as i32,
                preserve_colors: self.rc_preserve as i32,
            });
        }
        // Base curve (basecurve.c, iop_order.c v50_order pos 44.0 — between
        // rgblevels 43.0 and sigmoid 45.3). C reads only channel 0 of the curve
        // table (`const int ch = 0`), so a single LUT + unbounded tail coeffs
        // suffices. default_colorspace is IOP_CS_RGB with no Lab sandwich: like
        // rgbcurve we sample at 65536 steps in process() exactly as
        // basecurve.c:commit_params does. Exposure fusion > 0 switches to the
        // Laplacian-pyramid exposure-blend path (process_fusion), which is NOT
        // pixel-local — the stage carries `fusion` so pipeline.rs routes it to
        // the serial whole-buffer branch.
        if self.bc_on {
            let pinned = |count: f32, nodes_in: &[(f32, f32); 20]| -> Vec<(f32, f32)> {
                let nnodes = (count.round() as usize).clamp(2, 20);
                let mut nodes = *nodes_in;
                nodes[..nnodes]
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                nodes[0].0 = 0.0;
                nodes[nnodes - 1].0 = 1.0;
                nodes[..nnodes].to_vec()
            };
            let lut = c41_core::iop::basecurve::build_table(
                &pinned(self.bc_nnodes, &self.bc_nodes),
                self.bc_type as i32 as u32,
            );
            p.push(Stage::Basecurve {
                table: lut.table,
                coeffs: lut.unbounded_coeffs,
                preserve_colors: self.bc_preserve as i32,
                fusion: (self.bc_exposure_fusion.round() as i32).clamp(0, 2),
                stops: self.bc_exposure_stops,
                bias: self.bc_exposure_bias,
                space,
            });
        }
        // Shadows/Highlights (shadhi.c, iop_order.c v50_order pos 50.0 — between
        // basicadj 40.0 and colorcorrection 55.0). A Gaussian-blurred base layer is merged with the
        // original Lab pixels to lift shadows / recover highlights. NOT pixel-local
        // (the blur reads neighbours), so it stays on the serial whole-buffer path
        // like Sharpen.
        //
        // Identity when off, or when shadows + highlights + whitepoint are all 0
        // (the gate mirrors `is_identity`). `compress`, `ccorrect` and `radius` have
        // no effect without a non-zero shadow/highlight to drive them, so they are
        // not in the gate — a user who only nudged radius shouldn't get a stage that
        // does nothing but cost a full-buffer blur.
        if self.shadhi_on
            && (self.shadhi_shadows != 0.0
                || self.shadhi_highlights != 0.0
                || self.shadhi_whitepoint != 0.0)
        {
            p.push(Stage::Shadhi {
                shadows: self.shadhi_shadows,
                highlights: self.shadhi_highlights,
                whitepoint: self.shadhi_whitepoint,
                radius: self.shadhi_radius,
                compress: self.shadhi_compress,
                shadows_ccorrect: self.shadhi_shadows_ccorrect,
                highlights_ccorrect: self.shadhi_highlights_ccorrect,
                scale,
                space,
            });
        }
        // Lowpass (iop_order.c v50_order pos 33.0) — a Gaussian-blur-based local
        // contrast enhancement that runs in Lab: blur a copy, then apply the
        // contrast/brightness LUT pair + a/b saturation to the blurred pixels. It
        // is NOT pixel-local (the blur reads neighbours), so it stays on the serial
        // whole-buffer path like Sharpen.
        //
        // The gate mirrors `is_identity`: contrast 1.0 + brightness 0.0 +
        // saturation 1.0 yields identity LUTs, so radius doesn't matter then.
        // `unbound` is hardcoded true (the C default; darktable does not expose it
        // in the GUI).
        if self.lowpass_on
            && (self.lowpass_contrast != 1.0
                || self.lowpass_brightness != 0.0
                || self.lowpass_saturation != 1.0)
        {
            p.push(Stage::Lowpass {
                radius: self.lowpass_radius,
                contrast: self.lowpass_contrast,
                brightness: self.lowpass_brightness,
                saturation: self.lowpass_saturation,
                scale,
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
        // Filmic RGB (iop_order.c pos 46.0 — display transform cluster: after
        // sigmoid 45.3, before colisa 47). Another scene-linear → display tone
        // map; like every spline curve it is never a no-op while enabled, so
        // the enable flag alone gates the stage.
        if self.filmic_on {
            p.push(Stage::FilmicRgb {
                data: Box::new(c41_core::iop::filmicrgb::FilmicData::from_params(
                    &self.filmic_params(),
                )),
                space,
            });
        }
        // Colisa (iop_order.c pos 47 — display-referred, just before tonecurve
        // 48 and levels 49). Its own comment upstream is "edit contrast while
        // damaging colour", which is why it sits in that cluster rather than
        // with the scene-referred stages. All three sliders at 0 is the identity
        // curve pair, so the stage is skipped there.
        if self.colisa_on
            && (self.colisa_contrast != 0.0
                || self.colisa_brightness != 0.0
                || self.colisa_saturation != 0.0)
        {
            p.push(Stage::Colisa {
                contrast: self.colisa_contrast,
                brightness: self.colisa_brightness,
                saturation: self.colisa_saturation,
                space,
            });
        }
        // Tone curve (iop_order.c pos 48, between colisa 47 and levels 49) —
        // three-channel Lab LUT built from the L-curve anchors through the V1
        // `curve_tools` sampler (the same machinery darktable's
        // `dt_draw_curve_calc_values` uses), so the drawn curve IS the applied
        // curve. First slice exposes the L channel; a/b keep their C-default
        // identity curves (3 nodes at (0,0),(0.5,0.5),(1,1), MONOTONE_HERMITE)
        // and the C-default autoscale/unbound/preserve modes.
        if self.tc_on {
            let nnodes = (self.tc_nnodes.round() as usize).clamp(2, 20);
            let mut nodes_l = self.tc_nodes_l;
            // Keep the anchors x-sorted and the endpoints pinned: the editor
            // maintains this invariant, but a decoded blob might not.
            nodes_l[..nnodes].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            nodes_l[0].0 = 0.0;
            nodes_l[nnodes - 1].0 = 1.0;
            let tables = c41_core::iop::tonecurve::build_lut(
                &nodes_l[..nnodes],
                self.tc_type as i32 as u32,
                &TONECURVE_DEFAULT_AB_NODES,
                c41_core::curve_tools::MONOTONE_HERMITE,
                &TONECURVE_DEFAULT_AB_NODES,
                c41_core::curve_tools::MONOTONE_HERMITE,
                self.tc_autoscale as i32,
            );
            p.push(Stage::ToneCurve {
                table_l: tables.table_l,
                table_a: tables.table_a,
                table_b: tables.table_b,
                coeffs_l: tables.unbounded_coeffs_l,
                coeffs_ab: tables.unbounded_coeffs_ab,
                autoscale_ab: self.tc_autoscale as i32,
                unbound_ab: self.tc_unbound as i32,
                preserve_colors: self.tc_preserve as i32,
                space,
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
        // Bloom (iop_order.c pos 61, creative cluster: after colorzones 60,
        // before colorize 62) — display-referred glow: gather Lab L above the
        // threshold, box-blur it, screen-blend back. The enable flag alone
        // gates the stage (matching Colorize): with darktable's own defaults
        // the screen blend does visible work whenever anything passes the
        // threshold.
        if self.bl_on {
            p.push(Stage::Bloom {
                size: self.bl_size,
                threshold: self.bl_threshold,
                strength: self.bl_strength,
                space,
            });
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

/// tonecurve.c $DEFAULT a/b curves: 3 identity nodes, evaluated with the
/// MONOTONE_HERMITE spline. The first slice exposes only the L editor, so a/b
/// always build from these (matching C `commit_params`, which samples whatever
/// the curve widgets hold — identity by default).
const TONECURVE_DEFAULT_AB_NODES: [(f32, f32); 3] = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)];

/// Minimum black→white separation, on the 0..100 slider scale, for which a
/// Levels stage is emitted. A hairline range is not a meaningful edit (it maps
/// the whole tonal scale onto a sliver) and it drives `pct` — and hence the
/// `pct^gamma` branch of `process_pixels` — to absurd magnitudes. One slider
/// unit is far tighter than any useful edit while keeping the arithmetic sane;
/// the actual overflow guarantee comes from the output clamp in
/// `levels::process_pixels`, not from this.
const LEVELS_MIN_RANGE: f32 = 1.0;

/// Bump when the [`PreviewParams::encode`] layout changes (old blobs in
/// [`PARAMS_LAYOUTS`] decode with the new fields defaulted, rather than
/// mis-parsing). v2 added the sigmoid stage.
/// v3 added the sharpen stage. v4 added vibrance. v5 added color contrast.
/// v6 added temperature (white balance). v7 adds invert (film-camera negative).
/// v8 adds colorize (HSL colour replacement). v9 adds color correction.
/// v10 adds color zones (LCH equaliser). v11 adds levels (black/grey/white).
/// v12 adds basicadj (basic adjustments). v13 adds lowpass.
/// v14 adds shadhi (shadows/highlights). v15 adds primaries (RGB adjustment).
/// v16 adds negadoctor (film negative inversion).
/// v17 adds toneequalizer (exposure-channel tone mapping).
/// v18 adds colorbalancergb (colour balance RGB, 1 bool + 33 f32).
/// v19 adds filmicrgb (filmic RGB display transform, 1 bool + 7 f32).
/// v20 adds highlight reconstruction (2 bools + 1 f32; runs pre-demosaic in
/// the raw front end, not as a pipeline stage — see [`PreviewParams::hl_opts`]).
/// v21 adds denoise profiled (2 bools + 3 f32, normal pipeline stage).
/// v22 adds bloom (1 bool + 3 f32).
/// v23 adds tone curve (2 bools + 4 f32 scalars + 40 interleaved L-anchor
/// coordinates).
/// v24 adds RGB curve (1 bool + 8 f32 scalars + 3×40 interleaved R/G/B anchor
/// coordinates).
/// v25 adds base curve (1 bool + 6 f32 scalars + 40 interleaved node coords).
/// v26 adds lens correction (2 bools + 6 f32; the camera/lens identity lives
/// in `main.darkroom_lens_choice` — the blob carries no strings).
const ENCODE_VERSION: u8 = 26;
/// 1 version byte + 37 bool bytes + 444 little-endian f32.
const ENCODED_LEN: usize = 1 + 37 + 444 * 4;

/// `(version, n_bools, n_f32s)` for every `PreviewParams` layout ever written.
/// Append-only: a new module appends to both regions. Public so
/// `HistoryStack::decode` can size entries without hardcoding the current
/// default length (which would misparse older blobs).
pub(crate) const PARAMS_LAYOUTS: &[(u8, usize, usize)] = &[
    (12, 19, 129), // v12: basicadj was the last module
    (13, 20, 133), // v13: lowpass added
    (14, 21, 140), // v14: shadhi added
    (15, 22, 148), // v15: primaries added
    (16, 23, 164), // v16: negadoctor added
    (17, 24, 173), // v17: toneequalizer added
    (18, 25, 206), // v18: colorbalancergb added
    (19, 26, 213), // v19: filmicrgb added
    (20, 28, 214), // v20: highlight reconstruction added
    (21, 30, 217), // v21: denoise profiled added
    (22, 31, 220), // v22: bloom added
    (23, 33, 264), // v23: tone curve added
    (24, 34, 392), // v24: RGB curve added
    (25, 35, 438), // v25: base curve added
    (26, 37, 444), // v26: lens correction added
];

/// Encoded byte length of a `PreviewParams` blob at `version`, or `None` if the
/// version is not in [`PARAMS_LAYOUTS`] (and thus can't be decoded).
pub(crate) fn encoded_len_for_version(version: u8) -> Option<usize> {
    PARAMS_LAYOUTS
        .iter()
        .find(|(v, _, _)| *v == version)
        .map(|(_, nb, nf)| 1 + nb + nf * 4)
}

/// Resolved lens-correction gear: the `(camera, lens)` pair behind a
/// [`Stage`](c41_core::pipeline::Stage)::LensCorrection. The camera/lens
/// identity can't join the [`PreviewParams`] blob (strings), so callers that
/// have it resolved pass it alongside the params — the darkroom view caches an
/// `Arc` of this per image, export carries the same `Arc` — and every pipeline
/// build site takes it via the `_gear` variants below, so previews and exports
/// can never disagree about the correction being applied.
pub type LensGear = (
    c41_core::iop::lens::ResolvedCamera,
    c41_core::iop::lens::ResolvedLens,
);

/// Resolve a persisted [`crate::persist::LensChoice`] into pipeline-ready
/// gear. `None` when nothing is selected (`lens_model` empty — nothing chosen
/// yet) or the identity doesn't match the database exactly (then the module
/// shows "no data", like darktable).
pub fn resolve_gear(choice: &crate::persist::LensChoice) -> Option<std::sync::Arc<LensGear>> {
    if choice.lens_model.is_empty() {
        return None;
    }
    c41_core::iop::lens::resolve(
        &choice.camera_maker,
        &choice.camera_model,
        &choice.lens_maker,
        &choice.lens_model,
    )
    .map(std::sync::Arc::new)
}

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
    apply_pipeline_gear(base, width, height, rowstride, nch, params, None)
}

/// [`Self::apply_pipeline`] with resolved lens-correction gear — the variant
/// every caller that has gear cached must use, so an enabled lens module
/// actually applies. See [`LensGear`].
pub fn apply_pipeline_gear(
    base: &[u8],
    width: usize,
    height: usize,
    rowstride: usize,
    nch: usize,
    params: &PreviewParams,
    lens_gear: Option<&LensGear>,
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
    let pipeline = params.to_pipeline_with(ColorSpace::LinearSrgb, 1.0, lens_gear);
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
    apply_pipeline_rgb16_gear(base, width, height, params, None)
}

/// [`Self::apply_pipeline_rgb16`] with resolved lens-correction gear. See
/// [`LensGear`].
pub fn apply_pipeline_rgb16_gear(
    base: &[u16],
    width: usize,
    height: usize,
    params: &PreviewParams,
    lens_gear: Option<&LensGear>,
) -> Vec<u16> {
    let n = width.saturating_mul(height);
    if width == 0 || height == 0 || base.len() < n * 3 {
        return base.to_vec();
    }
    let pipeline = params.to_pipeline_with(ColorSpace::LinearSrgb, 1.0, lens_gear);
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
    render_linear_to_srgb8_gear(linear, width, height, params, None)
}

/// The lens-correction **pre-pass** (m4-131): warp + vignette `linear` on the
/// full, un-cropped frame — darktable runs lens at iop_order 13, *before*
/// crop/straighten, so the distortion centre and the vignetting falloff are
/// measured against the whole sensor frame. Every raw path calls this right
/// after decode and feeds the result through its geometry pass; the pipeline
/// builders for those paths then skip the stage
/// ([`PreviewParams::to_pipeline_lens_preapplied`]). The non-raw funnels keep
/// the in-pipeline stage instead — their sources are never cropped, so a
/// full-frame warp inside the pipeline is exactly this placement.
///
/// Gate mirrors the pipeline emission exactly: `lens_on` plus resolved gear.
/// Returns an owned buffer either way (the active arm is a full-frame warp).
pub fn apply_lens_prepass(
    linear: &[f32],
    width: usize,
    height: usize,
    params: &PreviewParams,
    lens_gear: Option<&LensGear>,
) -> Vec<f32> {
    if let Some((cam, lens)) = lens_gear.filter(|_| params.lens_on) {
        // `process` fully overwrites its output in every branch (warp_into
        // writes all four lanes of every pixel), so zero-init beats cloning
        // the input into the destination first.
        let mut out = vec![0.0f32; linear.len()];
        c41_core::iop::lens::process(
            linear,
            &mut out,
            width,
            height,
            lens,
            &params.lens_params(cam, lens),
        );
        out
    } else {
        linear.to_vec()
    }
}

/// [`Self::render_linear_to_srgb8`] with resolved lens-correction gear — the
/// variant the darkroom preview and the export path both use so a lens
/// correction applies identically in both. See [`LensGear`].
pub fn render_linear_to_srgb8_gear(
    linear: &[f32],
    width: usize,
    height: usize,
    params: &PreviewParams,
    lens_gear: Option<&LensGear>,
) -> Vec<u8> {
    let n = width.saturating_mul(height);
    if linear.len() < n * 4 {
        return vec![0u8; n * 3];
    }
    srgb_encode_rgb(linear, width, height, params, lens_gear)
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
    render_linear_to_srgb16_gear(linear, width, height, params, None)
}

/// [`Self::render_linear_to_srgb16`] with resolved lens-correction gear. See
/// [`LensGear`].
pub fn render_linear_to_srgb16_gear(
    linear: &[f32],
    width: usize,
    height: usize,
    params: &PreviewParams,
    lens_gear: Option<&LensGear>,
) -> Vec<u16> {
    let n = width.saturating_mul(height);
    if linear.len() < n * 4 {
        return vec![0u16; n * 3];
    }
    srgb_encode_rgb(linear, width, height, params, lens_gear)
        .iter()
        .map(|&e| (e.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
        .collect()
}

/// Shared render core: run the preview pipeline, map the Rec.2020 working space
/// to sRGB, and apply the sRGB OETF, yielding tightly-packed **RGB** sRGB floats
/// (`width*height*3`, pre-quantisation — values may fall outside `[0,1]` for
/// out-of-gamut colours, which the callers clamp). `linear` must be
/// `width*height*4` (callers guard the short case).
fn srgb_encode_rgb(
    linear: &[f32],
    width: usize,
    height: usize,
    params: &PreviewParams,
    lens_gear: Option<&LensGear>,
) -> Vec<f32> {
    let n = width.saturating_mul(height);
    // The raw funnels feed an already-lens-warped buffer (the pre-pass ran on
    // the full frame before geometry), so the pipeline must NOT emit the lens
    // stage again — see [`PreviewParams::to_pipeline_lens_preapplied`].
    let mut processed =
        params.to_pipeline_lens_preapplied(ColorSpace::Rec2020, 1.0, lens_gear).process(&linear[..n * 4], width, height);
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
    fn tonecurve_darkens_midtones_end_to_end() {
        // An L anchor pulling 0.5 down to 0.35 must darken an 8-bit mid-gray,
        // while white stays near the top — the drawn curve IS the applied
        // curve (both go through curve_data_sample).
        let mut p = PreviewParams::default();
        p.exposure_on = false; // isolate the curve
        p.tc_on = true;
        p.tc_type = 1.0; // Catmull-Rom
        p.tc_nnodes = 3.0;
        p.tc_nodes_l[1] = (0.5, 0.35);
        p.tc_nodes_l[2] = (1.0, 1.0);
        assert!(!p.is_identity());
        let base = vec![128u8, 128, 128, 250, 250, 250];
        let out = apply_pipeline(&base, 2, 1, 3, 3, &p);
        assert!(out[0] < 110, "midtone must darken, got {}", out[0]);
        assert!(
            out[3] > 200 && out[3] >= out[0],
            "near-white must stay near-white, got {}",
            out[3]
        );
    }

    #[test]
    fn rgbcurve_darkens_midtones_end_to_end() {
        // An R-channel anchor pulling 0.5 down to 0.35 must darken a mid-grey —
        // linked AUTOMATIC_RGB mode (the default) applies the R curve to every
        // channel, so the grey stays grey while getting darker. The drawn curve
        // IS the applied curve (both go through curve_data_sample).
        let mut p = PreviewParams::default();
        p.exposure_on = false; // isolate the curve
        p.rc_on = true;
        p.rc_type_r = 1.0; // Catmull-Rom
        p.rc_nnodes_r = 3.0;
        p.rc_nodes_r[1] = (0.5, 0.35);
        p.rc_nodes_r[2] = (1.0, 1.0);
        assert!(!p.is_identity());
        let base = vec![128u8, 128, 128];
        let out = apply_pipeline(&base, 1, 1, 3, 3, &p);
        assert!(
            out.iter().enumerate().all(|(k, &v)| v < base[k]),
            "every channel must darken, got {out:?} from {base:?}"
        );
        assert_eq!(out[0], out[1], "grey must stay neutral (equal ratios)");
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn basecurve_darkens_midtones_end_to_end() {
        // An anchor pulling 0.5 down to 0.35 must darken a mid-grey while it
        // stays neutral: the default LUMINANCE colour-preservation scales every
        // channel by table(x)/Y-luma(x), so equal-RGB in → equal-RGB out. The
        // drawn curve IS the applied curve (both sample the same LUT build).
        let mut p = PreviewParams::default();
        p.exposure_on = false; // isolate the curve
        p.bc_on = true;
        p.bc_type = 1.0; // Catmull-Rom
        p.bc_nnodes = 3.0;
        p.bc_nodes[1] = (0.5, 0.35);
        p.bc_nodes[2] = (1.0, 1.0);
        assert!(!p.is_identity());
        let base = vec![128u8, 128, 128];
        let out = apply_pipeline(&base, 1, 1, 3, 3, &p);
        assert!(
            out.iter().enumerate().all(|(k, &v)| v < base[k]),
            "every channel must darken, got {out:?} from {base:?}"
        );
        assert_eq!(out[0], out[1], "grey must stay neutral under LUMINANCE preservation");
        assert_eq!(out[1], out[2]);
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
        // negadoctor disabled ⇒ identity (matches to_pipeline gate: off ⇒ no stage)
        cfgs.push(PreviewParams::default());
        // negadoctor enabled ⇒ non-identity (on == the only gate)
        let mut nd = PreviewParams::default();
        nd.negadoctor_on = true;
        cfgs.push(nd);
        // denoise disabled ⇒ identity; enabled ⇒ non-identity (same gate shape)
        cfgs.push(PreviewParams::default());
        let mut dn = PreviewParams::default();
        dn.dn_on = true;
        cfgs.push(dn);
        // bloom disabled ⇒ identity; enabled ⇒ non-identity (same gate shape)
        cfgs.push(PreviewParams::default());
        let mut bl = PreviewParams::default();
        bl.bl_on = true;
        cfgs.push(bl);
        // tone curve disabled ⇒ identity; enabled ⇒ non-identity (same gate shape)
        cfgs.push(PreviewParams::default());
        let mut tc = PreviewParams::default();
        tc.tc_on = true;
        cfgs.push(tc);
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
        p.primaries_on = true;
        p.primaries_red_hue = 10.0; // off-default so the stage is emitted
        p.mono_on = true;
        p.sigmoid_on = true;
        p.velvia_on = true;
        p.velvia_strength = 50.0;
        p.split_on = true;
        p.sharpen_on = true;
        p.sharpen_amount = 1.0;
        p.basicadj_on = true;
        p.basicadj_exposure = 0.5; // off-default so the stage is emitted
        p.shadhi_on = true;
        p.shadhi_shadows = 25.0; // off-default so the stage is emitted
        p.lowpass_on = true;
        p.lowpass_contrast = 0.6; // off-default (identity is contrast 1.0)
        p.color_correction_on = true;
        p.color_correction_saturation = 1.5;
        p.colorize_on = true;
        p.levels_on = true;
        p.levels_grey = 40.0; // off-default so the stage is actually emitted
        // negadoctor: on by itself is enough to emit the stage (on is the gate).
        // Positioned at iop_order 28.5 — after graduatednd 25, before primaries
        // in table order.
        p.negadoctor_on = true;
        // tone equalizer: pos 24.0 in v50_order ("last module that need enlarged
        // roi_in") — after exposure 21, before graduatednd 25 and the 28.5 group
        // (negadoctor/primaries). A single non-zero gain is enough to emit it.
        p.toneeq_on = true;
        p.toneeq_shadows = 0.5;
        // color balance RGB: pos 41.5 in v50_order — after basicadj 40.0, before
        // rgblevels 43.0.
        p.cb_on = true;
        p.cb_contrast = 0.3;
        // filmic RGB: pos 46.0 in v50_order — after sigmoid 45.3, before colisa
        // 47.0. On by itself is enough to emit the stage (on is the gate).
        p.filmic_on = true;
        // graduatednd enabled too, so the toneequal-before-graduatednd order is
        // actually pinned by this test.
        p.gradnd_on = true;
        p.gradnd_density = -2.0;
        // denoise profiled: pos 9/10 in v50_order — immediately after demosaic
        // 8, before exposure. On by itself is enough to emit the stage.
        p.dn_on = true;
        // bloom: pos 61 in v50_order — creative cluster, after velvia 57,
        // before colorize 62. On by itself is enough to emit the stage.
        p.bl_on = true;
        // tone curve: pos 48 in v50_order — display-referred cluster, between
        // colisa 47 and levels 49. On by itself is enough to emit the stage.
        p.tc_on = true;
        // RGB curve: pos 42.0 in v50_order — scene-referred cluster, between
        // colorbalancergb 41.5 and rgblevels 43.0 (the 50.5 entry is in
        // legacy_order, not v50). On by itself is enough to emit the stage.
        p.rc_on = true;
        // Base curve: pos 44.0 in v50_order ("conversion from scene-referred to
        // display referred") — after rgblevels 43.0, before sigmoid 45.3. On by
        // itself is enough to emit the stage.
        p.bc_on = true;
        let names: Vec<&str> = p.to_pipeline(ColorSpace::LinearSrgb, 1.0).stages.iter().map(|s| s.name()).collect();
        // Pinned to v50_order, *except* Lowpass — v50 puts it at pos 33 (before
        // basicadj 40), but we run it after (after shadhi 50), matching the legacy
        // placement. This is a known deviation tracked for a follow-up commit.
        assert_eq!(
            names,
            ["denoiseprofile", "exposure", "toneequal", "graduatednd", "negadoctor", "primaries", "channelmixer", "sharpen", "basicadj", "colorbalancergb", "rgbcurve", "basecurve", "shadhi", "lowpass", "colorcorrection", "sigmoid", "filmicrgb", "tonecurve", "levels", "velvia", "bloom", "colorize", "splittoning"]
        );
        // Base curve is the scene→display conversion point (iop_order.c pos
        // 44.0): after rgbcurve/rgblevels, before sigmoid and filmicrgb.
        let bc_pos = names.iter().position(|n| *n == "basecurve").unwrap();
        assert!(bc_pos > names.iter().position(|n| *n == "rgbcurve").unwrap(),
            "basecurve must run after rgbcurve: {names:?}");
        assert!(bc_pos < names.iter().position(|n| *n == "sigmoid").unwrap(),
            "basecurve must run before sigmoid: {names:?}");
        // Tone curve sits in the display-referred cluster (iop_order.c pos 48):
        // after the scene-referred tone map, before levels.
        let tc_pos = names.iter().position(|n| *n == "tonecurve").unwrap();
        assert!(tc_pos > names.iter().position(|n| *n == "sigmoid").unwrap(),
            "tonecurve must run after sigmoid: {names:?}");
        assert!(tc_pos < names.iter().position(|n| *n == "levels").unwrap(),
            "tonecurve must run before levels: {names:?}");
        // Bloom sits in the display-referred creative cluster (iop_order.c pos
        // 61): after the tone map, and screen-blending on Lab L only makes sense
        // there.
        let bl_pos = names.iter().position(|n| *n == "bloom").unwrap();
        assert!(bl_pos > names.iter().position(|n| *n == "velvia").unwrap(),
            "bloom must run after velvia: {names:?}");
        assert!(bl_pos < names.iter().position(|n| *n == "colorize").unwrap(),
            "bloom must run before colorize: {names:?}");
        // Denoise is scene-referred and noise-thresholds against raw-domain
        // statistics: it must run BEFORE any tone mapping (exposure onwards).
        let dn_pos = names.iter().position(|n| *n == "denoiseprofile").unwrap();
        assert!(dn_pos < names.iter().position(|n| *n == "exposure").unwrap(),
            "denoiseprofile must run before exposure: {names:?}");
        // Color balance RGB is scene-referred (v50_order 41.5): it must run
        // before the sigmoid tone map (45.3), on unbounded linear data.
        let cb = names.iter().position(|n| *n == "colorbalancergb").unwrap();
        let sig_pos = names.iter().position(|n| *n == "sigmoid").unwrap();
        assert!(cb < sig_pos, "colorbalancergb must run before sigmoid: {names:?}");
        // Filmic RGB is a display transform at v50_order 46.0: after the
        // sigmoid tone map (45.3) and still in the display-referred cluster
        // (before colisa 47 / levels 49).
        let filmic = names.iter().position(|n| *n == "filmicrgb").unwrap();
        assert!(filmic > sig_pos, "filmicrgb must run after sigmoid: {names:?}");
        // Levels is display-referred (iop_order.c pos 49): it clips at its
        // black point and treats L as 0..100, so running it before a tone map
        // would crush the scene-linear highlights those maps exist to roll off.
        let lev = names.iter().position(|n| *n == "levels").unwrap();
        assert!(lev > sig_pos, "levels must run after sigmoid: {names:?}");
        assert!(lev > filmic, "levels must run after filmicrgb: {names:?}");
        // RGB curve is v50_order 42.0 — scene-referred: after color balance
        // RGB (41.5), before every tone-mapping module (sigmoid 45.3 onwards).
        let rc = names.iter().position(|n| *n == "rgbcurve").unwrap();
        assert!(
            rc > names.iter().position(|n| *n == "colorbalancergb").unwrap(),
            "rgbcurve must run after colorbalancergb: {names:?}"
        );
        assert!(rc < sig_pos, "rgbcurve must run before sigmoid: {names:?}");
    }

    #[test]
    fn to_pipeline_orders_lens_between_denoise_and_exposure() {
        // lens.c sits at v50_order pos 13 — after denoiseprofile 9/10, before
        // exposure 21. The stage carries a database-resolved lens, so this needs
        // the real lensfun data and skips where the package is absent (the
        // canonical-order test above can't pin it for the same reason).
        let Some(gear) = c41_core::iop::lens::resolve(
            "Canon",
            "Canon EOS 5D Mark II",
            "Canon",
            "Canon EF 50mm f/1.4 USM",
        ) else {
            eprintln!("skip: lensfun database unavailable");
            return;
        };
        let mut p = PreviewParams::default();
        p.lens_on = true;
        p.dn_on = true;
        p.exposure_on = true;
        p.ev = 0.5; // off-default so exposure is emitted
        let names: Vec<&str> = p
            .to_pipeline_with(ColorSpace::LinearSrgb, 1.0, Some(&gear))
            .stages
            .iter()
            .map(|s| s.name())
            .collect();
        assert_eq!(names, ["denoiseprofile", "lens", "exposure"], "{names:?}");
    }

    #[test]
    fn to_pipeline_lens_preapplied_omits_only_the_lens_stage() {
        // The raw funnels receive an already-warped buffer (the m4-131 pre-pass
        // ran on the full frame before geometry), so their builder must drop
        // exactly the lens stage and nothing else.
        let Some(gear) = c41_core::iop::lens::resolve(
            "Canon",
            "Canon EOS 5D Mark II",
            "Canon",
            "Canon EF 50mm f/1.4 USM",
        ) else {
            eprintln!("skip: lensfun database unavailable");
            return;
        };
        let mut p = PreviewParams::default();
        p.lens_on = true;
        p.dn_on = true;
        p.exposure_on = true;
        p.ev = 0.5; // off-default so exposure is emitted
        let with = p.to_pipeline_with(ColorSpace::LinearSrgb, 1.0, Some(&gear));
        assert!(with.stages.iter().any(|s| s.name() == "lens"));
        let without = p.to_pipeline_lens_preapplied(ColorSpace::LinearSrgb, 1.0, Some(&gear));
        assert!(!without.stages.iter().any(|s| s.name() == "lens"), "{:?}", without.stages.len());
        let kept: Vec<&str> = without.stages.iter().map(|s| s.name()).collect();
        let expected: Vec<&str> = with
            .stages
            .iter()
            .map(|s| s.name())
            .filter(|n| *n != "lens")
            .collect();
        assert_eq!(kept, expected, "{kept:?} vs {expected:?}");
    }

    #[test]
    fn lens_stage_omitted_without_gear() {
        // `lens_on` with no resolved gear emits nothing (darktable's module
        // shows "no data" and does the same) — but `is_identity` still reports
        // non-identity, which only costs one unchanged re-render.
        let mut p = PreviewParams::default();
        p.lens_on = true;
        assert!(!p.is_identity());
        assert!(p
            .to_pipeline_with(ColorSpace::LinearSrgb, 1.0, None)
            .stages
            .is_empty());
    }

    /// Ties PreviewParams::default() (the UI defaults) to the identity matrix in
    /// core. This is the cross-crate invariant that M1 (the duplicated "neutral"
    /// predicate) is designed to protect: if someone changes a default in preview.rs
    /// without updating primaries_is_neutral, this fails.
    #[test]
    fn default_primaries_params_are_a_true_no_op() {
        let d = PreviewParams::default();
        assert!(d.primaries_is_neutral(), "default params must be neutral");
        for space in [ColorSpace::Rec2020, ColorSpace::LinearSrgb] {
            let m = primaries::compute_matrix(
                space,
                d.primaries_achromatic_tint_hue.to_radians(),
                d.primaries_achromatic_tint_purity,
                d.primaries_red_hue.to_radians(),
                d.primaries_red_purity,
                d.primaries_green_hue.to_radians(),
                d.primaries_green_purity,
                d.primaries_blue_hue.to_radians(),
                d.primaries_blue_purity,
            );
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0f32 } else { 0.0 };
                    assert!(
                        (m[i * 4 + j] - expected).abs() < 1e-5,
                        "{space:?}: matrix[{i}][{j}] = {} ≠ {}",
                        m[i * 4 + j],
                        expected
                    );
                }
            }
        }
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
            colisa_on: true, colisa_contrast: 0.3,
            colisa_brightness: -0.2, colisa_saturation: 0.45,
            // Every value distinct, so a field packed or unpacked at the wrong
            // offset shows up as a mismatch rather than coincidentally matching
            // its neighbour.
            basicadj_on: true, basicadj_black_point: 0.02,
            basicadj_exposure: 1.25, basicadj_hlcompr: 70.0,
            basicadj_hlcomprthresh: 30.0, basicadj_contrast: 0.6,
            basicadj_preserve_colors: 2.0, basicadj_middle_grey: 22.5,
            basicadj_brightness: -0.4, basicadj_saturation: 0.15,
            basicadj_vibrance: -0.35,
            lowpass_on: true, lowpass_radius: 42.0,
            lowpass_contrast: 0.6, lowpass_brightness: -0.4,
            lowpass_saturation: 0.15,
            shadhi_on: true, shadhi_shadows: 25.0,
            shadhi_highlights: -30.0, shadhi_whitepoint: 2.0,
            shadhi_radius: 80.0, shadhi_compress: 60.0,
            shadhi_shadows_ccorrect: 75.0, shadhi_highlights_ccorrect: 40.0,
            primaries_on: true,
            primaries_achromatic_tint_hue: 10.0,
            primaries_achromatic_tint_purity: 0.3,
            primaries_red_hue: -15.0,
            primaries_red_purity: 1.5,
            primaries_green_hue: 5.0,
            primaries_green_purity: 0.8,
            primaries_blue_hue: 20.0,
            primaries_blue_purity: 1.2,
            negadoctor_on: true,
            negadoctor_film_stock: 0.0, // DT_FILMSTOCK_NB
            negadoctor_dmin_r: 0.2,
            negadoctor_dmin_g: 0.35,
            negadoctor_dmin_b: 0.5,
            negadoctor_wb_high_r: 1.4,
            negadoctor_wb_high_g: 1.1,
            negadoctor_wb_high_b: 0.9,
            negadoctor_wb_low_r: 0.8,
            negadoctor_wb_low_g: 0.95,
            negadoctor_wb_low_b: 1.05,
            negadoctor_d_max: 1.8,
            negadoctor_offset: 0.1,
            negadoctor_black: 0.02,
            negadoctor_gamma: 3.0,
            negadoctor_soft_clip: 0.4,
            negadoctor_exposure: 1.5,
            // Nine distinct gains so any wrong-offset packing shows as a mismatch
            // (they are adjacent f32s in the blob).
            toneeq_on: true,
            toneeq_noise: 0.11,
            toneeq_ultra_deep_blacks: -0.22,
            toneeq_deep_blacks: 0.33,
            toneeq_blacks: -0.44,
            toneeq_shadows: 0.55,
            toneeq_midtones: -0.66,
            toneeq_highlights: 0.77,
            toneeq_whites: -0.88,
            toneeq_speculars: 0.99,
            // Color balance RGB: every field set off-default and pairwise
            // distinct, so a wrong offset in the 33-float block shows as a
            // mismatch.
            cb_on: true,
            cb_shadows_y: 0.01,
            cb_shadows_c: -0.02,
            cb_shadows_h: 3.0,
            cb_midtones_y: 0.04,
            cb_midtones_c: -0.05,
            cb_midtones_h: 6.0,
            cb_highlights_y: 0.07,
            cb_highlights_c: -0.08,
            cb_highlights_h: 9.0,
            cb_global_y: 0.10,
            cb_global_c: -0.11,
            cb_global_h: 12.0,
            cb_shadows_weight: 1.13,
            cb_white_fulcrum: -1.14,
            cb_highlights_weight: 2.15,
            cb_chroma_shadows: 0.16,
            cb_chroma_highlights: -0.17,
            cb_chroma_global: 0.18,
            cb_chroma_midtones: -0.19,
            cb_saturation_global: 0.20,
            cb_saturation_highlights: -0.21,
            cb_saturation_midtones: 0.22,
            cb_saturation_shadows: -0.23,
            cb_hue_angle: 24.0,
            cb_brilliance_global: 0.25,
            cb_brilliance_highlights: -0.26,
            cb_brilliance_midtones: 0.27,
            cb_brilliance_shadows: -0.28,
            cb_mask_grey_fulcrum: 0.29,
            cb_vibrance: -0.30,
            cb_grey_fulcrum: 0.31,
            cb_contrast: -0.32,
            cb_formula: 0.0, // JzAzBz — the non-default formula
            // Filmic RGB: every value off-default and pairwise distinct so a
            // wrong offset in the trailing block shows as a mismatch.
            filmic_on: true,
            filmic_black_point_source: -6.5,
            filmic_white_point_source: 5.5,
            filmic_output_power: 3.5,
            filmic_latitude: 12.5,
            filmic_contrast: 1.7,
            filmic_balance: -25.0,
            filmic_saturation: 60.0,
            // Highlight reconstruction: both bools distinct from their defaults
            // (false/true) plus a non-default threshold, so a wrong offset in
            // the trailing block shows as a mismatch.
            hl_on: true,
            hl_opposed: false,
            hl_clip: 1.5,
            // Denoise (profiled): same pattern — bools flipped from their
            // (false/true) defaults, floats off-default and distinct.
            dn_on: true,
            dn_mode_y0u0v0: false,
            dn_strength: 0.8,
            dn_shadows: 1.3,
            dn_bias: -0.25,
            // Bloom: bool flipped from default, floats off-default and distinct.
            bl_on: true,
            bl_size: 33.0,
            bl_threshold: 72.5,
            bl_strength: 60.5,
            // Tone curve: both bools flipped from their (false/true) defaults,
            // scalars off-default and pairwise distinct, and a third node
            // moved so a wrong offset inside the 40-float anchor block shows
            // as a mismatch (slots 0/1 are the C-default identity endpoints —
            // slot 2 is the first one that differs from the default blob).
            tc_on: true,
            tc_unbound: false,
            tc_type: 1.0,      // Catmull-Rom
            tc_autoscale: 0.0, // manual
            tc_preserve: 2.0,  // MAX norm
            tc_nnodes: 3.0,
            tc_nodes_l: {
                let mut n = [(0.0f32, 0.0f32); 20];
                n[1] = (0.75, 0.75);
                n[2] = (1.0, 1.0);
                n
            },
            // RGB curve: bool flipped from default; every scalar off-default and
            // pairwise distinct so a wrong offset inside the appended block shows
            // as a mismatch; each channel's node array edited differently (R gets
            // a third node, G keeps 2 but moves its endpoint, B gets two extra)
            // so per-channel offsets can't silently swap.
            rc_on: true,
            rc_type_r: 1.0,
            rc_type_g: 0.0,
            rc_type_b: 2.0,
            rc_autoscale: 1.0, // MANUAL_RGB
            rc_preserve: 3.0,  // AVERAGE norm
            rc_nnodes_r: 3.0,
            rc_nnodes_g: 2.0,
            rc_nnodes_b: 4.0,
            rc_nodes_r: {
                let mut n = [(0.0f32, 0.0f32); 20];
                n[1] = (0.75, 0.4);
                n[2] = (1.0, 1.0);
                n
            },
            rc_nodes_g: {
                let mut n = [(0.0f32, 0.0f32); 20];
                n[1] = (0.85, 0.6);
                n
            },
            rc_nodes_b: {
                let mut n = [(0.0f32, 0.0f32); 20];
                n[1] = (0.25, 0.45);
                n[2] = (0.6, 0.55);
                n[3] = (1.0, 1.0);
                n
            },
            // Base curve: bool flipped from default; every scalar off-default
            // and pairwise distinct so a wrong offset inside the appended block
            // shows as a mismatch; node array edited at slots 1..4 (slots 0 and
            // beyond stay identity) so per-slot offsets can't silently swap.
            bc_on: true,
            bc_type: 0.0,     // CUBIC_SPLINE — non-default interpolator
            bc_preserve: 2.0, // DT_RGB_NORM_MAX
            bc_nnodes: 5.0,
            bc_exposure_fusion: 1.0, // two exposures
            bc_exposure_stops: 1.75,
            bc_exposure_bias: -0.35,
            bc_nodes: {
                let mut n = [(0.0f32, 0.0f32); 20];
                n[1] = (0.2, 0.15);
                n[2] = (0.45, 0.55);
                n[3] = (0.7, 0.8);
                n[4] = (1.0, 1.0);
                n
            },
            // Lens correction: both bools flipped from their defaults, every
            // float off-default and pairwise distinct so a wrong offset in the
            // trailing block shows as a mismatch. modify_flags/target_geom are
            // enum dropdowns — kept at exact small integers.
            lens_on: true,
            lens_inverse: true,
            lens_modify_flags: 5.0, // DIST_TCA
            lens_scale: 1.15,
            lens_focal: 85.0,
            lens_aperture: 5.6,
            lens_distance: 7.5,
            lens_target_geom: 3.0, // fisheye
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
    fn decode_v13_blob_defaults_shadhi_fields() {
        // A v13 blob (before shadhi was added — 20 bools / 133 f32s) must
        // decode successfully: the new shadhi fields fall back to their defaults,
        // so a saved style from the pre-shadhi era loads cleanly instead of
        // being silently discarded.
        let v13 = {
            let mut b = vec![0u8; 1 + 20 + 133 * 4];
            b[0] = 13; // version 13
            b
        };
        let decoded = PreviewParams::decode(&v13)
            .expect("v13 blob must decode (backward compat)");
        // Shadhi fields should be at their defaults, not garbage.
        let def = PreviewParams::default();
        assert_eq!(decoded.shadhi_on, def.shadhi_on);
        assert_eq!(decoded.shadhi_shadows, def.shadhi_shadows);
        assert_eq!(decoded.shadhi_highlights, def.shadhi_highlights);
        assert_eq!(decoded.shadhi_radius, def.shadhi_radius);
        assert_eq!(decoded.shadhi_compress, def.shadhi_compress);
        assert_eq!(decoded.shadhi_shadows_ccorrect, def.shadhi_shadows_ccorrect);
        assert_eq!(decoded.shadhi_highlights_ccorrect, def.shadhi_highlights_ccorrect);
    }

    #[test]
    fn decode_v14_blob_defaults_primaries_fields() {
        // A v14 blob (before primaries was added — 21 bools / 140 f32s) must
        // decode successfully: the new primaries fields fall back to their
        // defaults, so a saved style from the pre-primaries era loads cleanly
        // instead of being silently discarded.
        let v14 = {
            let mut b = vec![0u8; 1 + 21 + 140 * 4];
            b[0] = 14; // version 14
            b
        };
        let decoded = PreviewParams::decode(&v14)
            .expect("v14 blob must decode (backward compat)");
        // Primaries fields should be at their defaults, not garbage.
        let def = PreviewParams::default();
        assert_eq!(decoded.primaries_on, def.primaries_on);
        assert_eq!(decoded.primaries_achromatic_tint_hue, def.primaries_achromatic_tint_hue);
        assert_eq!(decoded.primaries_achromatic_tint_purity, def.primaries_achromatic_tint_purity);
        assert_eq!(decoded.primaries_red_hue, def.primaries_red_hue);
        assert_eq!(decoded.primaries_red_purity, def.primaries_red_purity);
        assert_eq!(decoded.primaries_green_hue, def.primaries_green_hue);
        assert_eq!(decoded.primaries_green_purity, def.primaries_green_purity);
        assert_eq!(decoded.primaries_blue_hue, def.primaries_blue_hue);
        assert_eq!(decoded.primaries_blue_purity, def.primaries_blue_purity);
    }

    #[test]
    fn decode_v20_blob_defaults_dn_fields() {
        // A v20 blob (before denoise was added — 28 bools / 214 f32s) must
        // decode successfully: the new denoise fields fall back to their
        // defaults, so a saved style from the pre-denoise era loads cleanly
        // instead of being silently discarded.
        let v20 = {
            let mut b = vec![0u8; 1 + 28 + 214 * 4];
            b[0] = 20; // version 20
            b
        };
        let decoded = PreviewParams::decode(&v20)
            .expect("v20 blob must decode (backward compat)");
        // Denoise fields should be at their defaults, not garbage.
        let def = PreviewParams::default();
        assert_eq!(decoded.dn_on, def.dn_on);
        assert_eq!(decoded.dn_mode_y0u0v0, def.dn_mode_y0u0v0);
        assert_eq!(decoded.dn_strength, def.dn_strength);
        assert_eq!(decoded.dn_shadows, def.dn_shadows);
        assert_eq!(decoded.dn_bias, def.dn_bias);
    }

    #[test]
    fn decode_v22_blob_defaults_tc_fields() {
        // A v22 blob (before tone curve was added — 31 bools / 220 f32s) must
        // decode successfully: the new tone-curve fields fall back to their
        // C defaults, so a saved style from the pre-tonecurve era loads cleanly.
        let v22 = {
            let mut b = vec![0u8; 1 + 31 + 220 * 4];
            b[0] = 22; // version 22
            b
        };
        let decoded = PreviewParams::decode(&v22)
            .expect("v22 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.tc_on, def.tc_on);
        assert_eq!(decoded.tc_type, def.tc_type);
        assert_eq!(decoded.tc_autoscale, def.tc_autoscale);
        assert_eq!(decoded.tc_unbound, def.tc_unbound);
        assert_eq!(decoded.tc_preserve, def.tc_preserve);
        assert_eq!(decoded.tc_nnodes, def.tc_nnodes);
        assert_eq!(decoded.tc_nodes_l, def.tc_nodes_l);
    }

    #[test]
    fn decode_v23_blob_defaults_rc_fields() {
        // A v23 blob (before RGB curve was added — 33 bools / 264 f32s) must
        // decode successfully: the new rgbcurve fields fall back to their
        // C defaults, so a saved style from the pre-rgbcurve era loads cleanly.
        let v23 = {
            let mut b = vec![0u8; 1 + 33 + 264 * 4];
            b[0] = 23; // version 23
            b
        };
        let decoded = PreviewParams::decode(&v23)
            .expect("v23 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.rc_on, def.rc_on);
        assert_eq!(decoded.rc_type_r, def.rc_type_r);
        assert_eq!(decoded.rc_type_g, def.rc_type_g);
        assert_eq!(decoded.rc_type_b, def.rc_type_b);
        assert_eq!(decoded.rc_autoscale, def.rc_autoscale);
        assert_eq!(decoded.rc_preserve, def.rc_preserve);
        assert_eq!(decoded.rc_nnodes_r, def.rc_nnodes_r);
        assert_eq!(decoded.rc_nnodes_g, def.rc_nnodes_g);
        assert_eq!(decoded.rc_nnodes_b, def.rc_nnodes_b);
        assert_eq!(decoded.rc_nodes_r, def.rc_nodes_r);
        assert_eq!(decoded.rc_nodes_g, def.rc_nodes_g);
        assert_eq!(decoded.rc_nodes_b, def.rc_nodes_b);
    }

    #[test]
    fn decode_v24_blob_defaults_bc_fields() {
        // A v24 blob (before base curve was added — 34 bools / 392 f32s) must
        // decode successfully: the new base-curve fields fall back to their
        // defaults, so a saved style from the pre-basecurve era loads cleanly.
        let v24 = {
            let mut b = vec![0u8; 1 + 34 + 392 * 4];
            b[0] = 24; // version 24
            b
        };
        let decoded = PreviewParams::decode(&v24)
            .expect("v24 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.bc_on, def.bc_on);
        assert_eq!(decoded.bc_type, def.bc_type);
        assert_eq!(decoded.bc_preserve, def.bc_preserve);
        assert_eq!(decoded.bc_nnodes, def.bc_nnodes);
        assert_eq!(decoded.bc_exposure_fusion, def.bc_exposure_fusion);
        assert_eq!(decoded.bc_exposure_stops, def.bc_exposure_stops);
        assert_eq!(decoded.bc_exposure_bias, def.bc_exposure_bias);
        assert_eq!(decoded.bc_nodes, def.bc_nodes);
        // and the defaulted module stays out of the pipeline
        assert!(
            decoded.to_pipeline(ColorSpace::LinearSrgb, 1.0)
                .stages.iter()
                .all(|s| s.name() != "basecurve"),
            "v24 blob must not emit a basecurve stage: defaults are off"
        );
    }

    #[test]
    fn decode_v21_blob_defaults_bl_fields() {
        // A v21 blob (before bloom was added — 30 bools / 217 f32s) must
        // decode successfully: the new bloom fields fall back to their
        // defaults, so a saved style from the pre-bloom era loads cleanly.
        let v21 = {
            let mut b = vec![0u8; 1 + 30 + 217 * 4];
            b[0] = 21; // version 21
            b
        };
        let decoded = PreviewParams::decode(&v21)
            .expect("v21 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.bl_on, def.bl_on);
        assert_eq!(decoded.bl_size, def.bl_size);
        assert_eq!(decoded.bl_threshold, def.bl_threshold);
        assert_eq!(decoded.bl_strength, def.bl_strength);
    }

    #[test]
    fn decode_v15_blob_defaults_negadoctor_fields() {
        // A v15 blob (before negadoctor was added — 22 bools / 148 f32s) must
        // decode successfully: the new negadoctor fields fall back to their
        // defaults, so a saved style from the pre-negadoctor era loads cleanly.
        let v15 = {
            let mut b = vec![0u8; 1 + 22 + 148 * 4];
            b[0] = 15; // version 15
            b
        };
        let decoded = PreviewParams::decode(&v15)
            .expect("v15 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.negadoctor_on, def.negadoctor_on);
        assert_eq!(decoded.negadoctor_film_stock, def.negadoctor_film_stock);
        assert_eq!(decoded.negadoctor_dmin_r, def.negadoctor_dmin_r);
        assert_eq!(decoded.negadoctor_dmin_g, def.negadoctor_dmin_g);
        assert_eq!(decoded.negadoctor_dmin_b, def.negadoctor_dmin_b);
        assert_eq!(decoded.negadoctor_wb_high_r, def.negadoctor_wb_high_r);
        assert_eq!(decoded.negadoctor_wb_high_g, def.negadoctor_wb_high_g);
        assert_eq!(decoded.negadoctor_wb_high_b, def.negadoctor_wb_high_b);
        assert_eq!(decoded.negadoctor_wb_low_r, def.negadoctor_wb_low_r);
        assert_eq!(decoded.negadoctor_wb_low_g, def.negadoctor_wb_low_g);
        assert_eq!(decoded.negadoctor_wb_low_b, def.negadoctor_wb_low_b);
        assert_eq!(decoded.negadoctor_d_max, def.negadoctor_d_max);
        assert_eq!(decoded.negadoctor_offset, def.negadoctor_offset);
        assert_eq!(decoded.negadoctor_black, def.negadoctor_black);
        assert_eq!(decoded.negadoctor_gamma, def.negadoctor_gamma);
        assert_eq!(decoded.negadoctor_soft_clip, def.negadoctor_soft_clip);
        assert_eq!(decoded.negadoctor_exposure, def.negadoctor_exposure);
    }

    #[test]
    fn decode_v16_blob_defaults_toneeq_fields() {
        // A v16 blob (before the tone equalizer was added — 23 bools / 164 f32s)
        // must decode successfully: the new toneeq fields fall back to their
        // defaults, so a saved style from the pre-toneequalizer era loads cleanly.
        let v16 = {
            let mut b = vec![0u8; 1 + 23 + 164 * 4];
            b[0] = 16; // version 16
            b
        };
        let decoded = PreviewParams::decode(&v16)
            .expect("v16 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.toneeq_on, def.toneeq_on);
        assert_eq!(decoded.toneeq_noise, def.toneeq_noise);
        assert_eq!(decoded.toneeq_ultra_deep_blacks, def.toneeq_ultra_deep_blacks);
        assert_eq!(decoded.toneeq_deep_blacks, def.toneeq_deep_blacks);
        assert_eq!(decoded.toneeq_blacks, def.toneeq_blacks);
        assert_eq!(decoded.toneeq_shadows, def.toneeq_shadows);
        assert_eq!(decoded.toneeq_midtones, def.toneeq_midtones);
        assert_eq!(decoded.toneeq_highlights, def.toneeq_highlights);
        assert_eq!(decoded.toneeq_whites, def.toneeq_whites);
        assert_eq!(decoded.toneeq_speculars, def.toneeq_speculars);
    }

    #[test]
    fn negadoctor_commit_params_matches_darktable() {
        // Pins the commit_params arithmetic in src/iop/negadoctor.c:239-267:
        //   wb_high[c] = p->wb_high[c] / p->D_max   (premultiply, spare one div/pixel)
        //   offset[c]  = p->wb_high[c] * p->offset * p->wb_low[c]  (uses ORIGINAL wb_high)
        //   Dmin B&W   = collapse to p->Dmin[0] for all channels
        //   black      = -p->exposure * (1.0f + p->black)          (FMA trick)
        //   soft_clip_comp = 1.0f - p->soft_clip
        // Uses non-default values so that any regression in the derivation is
        // caught — identity params (1/1/1, D_max=1, offset=0) trivially pass.
        {
            // ── Color film stock ──────────────────────────────────────────────
            let mut p = PreviewParams::default();
            p.negadoctor_on = true;
            p.negadoctor_film_stock = 1.0; // DT_FILMSTOCK_COLOR
            p.negadoctor_dmin_r = 1.13;
            p.negadoctor_dmin_g = 0.49;
            p.negadoctor_dmin_b = 0.27;
            p.negadoctor_wb_high_r = 1.2;
            p.negadoctor_wb_high_g = 0.8;
            p.negadoctor_wb_high_b = 1.1;
            p.negadoctor_wb_low_r = 1.1;
            p.negadoctor_wb_low_g = 0.9;
            p.negadoctor_wb_low_b = 1.3;
            p.negadoctor_d_max = 2.0;
            p.negadoctor_offset = -0.03;
            p.negadoctor_black = 0.0755;
            p.negadoctor_gamma = 4.0;
            p.negadoctor_soft_clip = 0.75;
            p.negadoctor_exposure = 0.9245;

            let pipe = p.to_pipeline(ColorSpace::LinearSrgb, 1.0);
            let stage = pipe
                .stages
                .iter()
                .find(|s| s.name() == "negadoctor")
                .expect("negadoctor stage should be emitted");
            let neg = match stage {
                Stage::Negadoctor { dmin, wb_high, offset, black, gamma, soft_clip, soft_clip_comp, exposure } => {
                    (dmin, wb_high, offset, black, gamma, soft_clip, soft_clip_comp, exposure)
                }
                _ => panic!("expected Stage::Negadoctor, got {stage:?}"),
            };

            // wb_high = wb_high_original / D_max  (per darktable:247)
            assert!((neg.1[0] - 1.2 / 2.0).abs() < 1e-6, "wb_high_r");
            assert!((neg.1[1] - 0.8 / 2.0).abs() < 1e-6, "wb_high_g");
            assert!((neg.1[2] - 1.1 / 2.0).abs() < 1e-6, "wb_high_b");

            // offset = ORIGINAL wb_high * offset * wb_low  (per darktable:249)
            assert!((neg.2[0] - 1.2 * (-0.03) * 1.1).abs() < 1e-6, "offset_r");
            assert!((neg.2[1] - 0.8 * (-0.03) * 0.9).abs() < 1e-6, "offset_g");
            assert!((neg.2[2] - 1.1 * (-0.03) * 1.3).abs() < 1e-6, "offset_b");

            // Dmin color: channels copied directly (channel 3 is inert — the
            // process loop only iterates 0..3, so the sentinel 1.0 is harmless).
            assert!((neg.0[0] - 1.13).abs() < 1e-6, "dmin_r");
            assert!((neg.0[1] - 0.49).abs() < 1e-6, "dmin_g");
            assert!((neg.0[2] - 0.27).abs() < 1e-6, "dmin_b");

            // black = -exposure * (1 + black)  (FMA trick, per darktable:258)
            let expected_black = -0.9245 * (1.0 + 0.0755);
            assert!((neg.3 - expected_black).abs() < 1e-6, "black: {} vs {}", neg.3, expected_black);

            // soft_clip / soft_clip_comp = 1 - soft_clip
            assert!((neg.4 - 4.0).abs() < 1e-6, "gamma");
            assert!((neg.5 - 0.75).abs() < 1e-6, "soft_clip");
            assert!((neg.6 - (1.0 - 0.75)).abs() < 1e-6, "soft_clip_comp");

            // exposure passthrough
            assert!((neg.7 - 0.9245).abs() < 1e-6, "exposure");
        }

        // ── B&W film stock: Dmin mono-collapse ─────────────────────────────────
        {
            let mut p = PreviewParams::default();
            p.negadoctor_on = true;
            p.negadoctor_film_stock = 0.0; // DT_FILMSTOCK_NB
            p.negadoctor_dmin_r = 1.0;
            p.negadoctor_dmin_g = 0.5;
            p.negadoctor_dmin_b = 0.3;
            p.negadoctor_d_max = 2.2;

            let pipe = p.to_pipeline(ColorSpace::LinearSrgb, 1.0);
            let stage = pipe
                .stages
                .iter()
                .find(|s| s.name() == "negadoctor")
                .expect("negadoctor stage should be emitted");
            let neg = match stage {
                Stage::Negadoctor { dmin, .. } => dmin,
                _ => panic!("expected Stage::Negadoctor"),
            };
            // All RGB channels collapse to Dmin[0] per darktable:254-255
            for c in 0..3 {
                assert!((neg[c] - 1.0).abs() < 1e-6, "dmin channel {c} should collapse to Dmin[0]=1.0, got {}", neg[c]);
            }
        }
    }

    #[test]
    fn toneequal_gains_reach_stage_raw_and_identity_gate_holds() {
        // Unlike negadoctor, toneequal's to_pipeline does NO param arithmetic:
        // the nine EV gains are carried raw into Stage::ToneEqual and all the
        // derivation (exp2 conversion, RBF solve, LUT build) happens in
        // `Stage::apply`, mirroring darktable where commit_params solves and
        // process builds the LUT. This pins (a) raw passthrough, in
        // get_channels_gains order (noise −8 EV … speculars 0 EV,
        // toneequal.c:1210), and (b) the identity gate — an enabled module with
        // all-zero gains emits NO stage.
        {
            let mut p = PreviewParams::default();
            p.toneeq_on = true;
            p.toneeq_noise = -0.8;
            p.toneeq_ultra_deep_blacks = -0.7;
            p.toneeq_deep_blacks = -0.6;
            p.toneeq_blacks = -0.5;
            p.toneeq_shadows = 1.0;
            p.toneeq_midtones = -0.3;
            p.toneeq_highlights = -0.2;
            p.toneeq_whites = -0.1;
            p.toneeq_speculars = 0.9;

            let pipe = p.to_pipeline(ColorSpace::LinearSrgb, 1.0);
            let stage = pipe
                .stages
                .iter()
                .find(|s| s.name() == "toneequal")
                .expect("toneequal stage should be emitted");
            match stage {
                Stage::ToneEqual { gains } => assert_eq!(
                    gains,
                    &[-0.8, -0.7, -0.6, -0.5, 1.0, -0.3, -0.2, -0.1, 0.9]
                ),
                _ => panic!("expected Stage::ToneEqual, got {stage:?}"),
            }
        }
        // All-zero gains ⇒ no stage even while enabled (flat unity correction).
        let mut p = PreviewParams::default();
        p.toneeq_on = true;
        assert!(
            p.to_pipeline(ColorSpace::LinearSrgb, 1.0)
                .stages
                .iter()
                .all(|s| s.name() != "toneequal"),
            "all-zero gains must not emit a toneequal stage"
        );
    }

    #[test]
    fn toneequal_midtone_boost_brightens_midgrey_end_to_end() {
        // End-to-end through to_pipeline + Pipeline::process: a +1 EV mid-tones
        // gain on a dark-grey patch must brighten it (the RBF fit trades peak
        // amplitude for smoothness, so ≈×1.7 rather than ×2 at −4 EV — pinned
        // numerically by c41-core's `single_channel_boost_lands_near_exp2_gain`).
        // Here we only demand direction + a sane magnitude band.
        let mut p = PreviewParams::default();
        p.toneeq_on = true;
        p.toneeq_shadows = 1.0; // −4 EV channel ("mid-tones" slider)
        // A uniform grey of 0.25 sits at norm-2 luminance √3·0.25 ≈ 0.43 ≈
        // −1.2 EV, between the −2 and −1 EV channels. Use a darker patch so it
        // lands near the boosted −4 EV centre: 0.25·2⁻³ ≈ 0.031 → expo ≈ −5 EV.
        let dark = [0.031_f32, 0.031, 0.031, 1.0];
        let input: Vec<f32> = dark.repeat(16);
        let out = p.to_pipeline(ColorSpace::LinearSrgb, 1.0).process(&input, 4, 4);
        for chunk in out.chunks_exact(4) {
            assert!(
                chunk[0] > dark[0] * 1.4 && chunk[0] < dark[0] * 2.1,
                "dark grey should be boosted ≈×1.75, got ×{}",
                chunk[0] / dark[0]
            );
        }
    }

    #[test]
    fn decode_v17_blob_defaults_cb_fields() {
        // A v17 blob (before color balance RGB was added — 24 bools / 173 f32s)
        // must decode successfully: the new cb_* fields fall back to their
        // defaults, so a saved style from the pre-colorbalancergb era loads
        // cleanly.
        let v17 = {
            let mut b = vec![0u8; 1 + 24 + 173 * 4];
            b[0] = 17; // version 17
            b
        };
        let decoded = PreviewParams::decode(&v17)
            .expect("v17 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.cb_on, def.cb_on);
        assert_eq!(decoded.cb_shadows_y, def.cb_shadows_y);
        assert_eq!(decoded.cb_shadows_h, def.cb_shadows_h);
        assert_eq!(decoded.cb_midtones_y, def.cb_midtones_y);
        assert_eq!(decoded.cb_highlights_c, def.cb_highlights_c);
        assert_eq!(decoded.cb_global_y, def.cb_global_y);
        assert_eq!(decoded.cb_global_h, def.cb_global_h);
        assert_eq!(decoded.cb_shadows_weight, def.cb_shadows_weight);
        assert_eq!(decoded.cb_white_fulcrum, def.cb_white_fulcrum);
        assert_eq!(decoded.cb_chroma_shadows, def.cb_chroma_shadows);
        assert_eq!(decoded.cb_saturation_global, def.cb_saturation_global);
        assert_eq!(decoded.cb_saturation_shadows, def.cb_saturation_shadows);
        assert_eq!(decoded.cb_hue_angle, def.cb_hue_angle);
        assert_eq!(decoded.cb_brilliance_global, def.cb_brilliance_global);
        assert_eq!(decoded.cb_brilliance_shadows, def.cb_brilliance_shadows);
        assert_eq!(decoded.cb_mask_grey_fulcrum, def.cb_mask_grey_fulcrum);
        assert_eq!(decoded.cb_vibrance, def.cb_vibrance);
        assert_eq!(decoded.cb_grey_fulcrum, def.cb_grey_fulcrum);
        assert_eq!(decoded.cb_contrast, def.cb_contrast);
        assert_eq!(decoded.cb_formula, def.cb_formula);
    }

    #[test]
    fn decode_v18_blob_defaults_filmic_fields() {
        // A v18 blob (before filmic RGB was added — 25 bools / 206 f32s) must
        // decode successfully: the new filmic_* fields fall back to their
        // defaults, so a saved style from the pre-filmicrgb era loads cleanly.
        let v18 = {
            let mut b = vec![0u8; 1 + 25 + 206 * 4];
            b[0] = 18; // version 18
            b
        };
        let decoded = PreviewParams::decode(&v18)
            .expect("v18 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.filmic_on, def.filmic_on);
        assert_eq!(decoded.filmic_black_point_source, def.filmic_black_point_source);
        assert_eq!(decoded.filmic_white_point_source, def.filmic_white_point_source);
        assert_eq!(decoded.filmic_output_power, def.filmic_output_power);
        assert_eq!(decoded.filmic_latitude, def.filmic_latitude);
        assert_eq!(decoded.filmic_contrast, def.filmic_contrast);
        assert_eq!(decoded.filmic_balance, def.filmic_balance);
        assert_eq!(decoded.filmic_saturation, def.filmic_saturation);
    }

    #[test]
    fn decode_v19_blob_defaults_hl_fields() {
        // A v19 blob (before highlight reconstruction was added — 26 bools /
        // 213 f32s) must decode successfully: the new hl_* fields fall back to
        // their defaults (off), so a saved style from the pre-hl era loads
        // cleanly and — importantly — stays OFF rather than suddenly running
        // reconstruction over raws that were decoded clamped before.
        let v19 = {
            let mut b = vec![0u8; 1 + 26 + 213 * 4];
            b[0] = 19; // version 19
            b
        };
        let decoded = PreviewParams::decode(&v19)
            .expect("v19 blob must decode (backward compat)");
        let def = PreviewParams::default();
        assert_eq!(decoded.hl_on, def.hl_on);
        assert_eq!(decoded.hl_opposed, def.hl_opposed);
        assert_eq!(decoded.hl_clip, def.hl_clip);
        assert!(decoded.hl_opts().is_none());
    }

    #[test]
    fn hl_opts_maps_mode_and_threshold() {
        // Off → None (no reconstruction, legacy clamped decode path).
        let mut p = PreviewParams::default();
        assert!(p.hl_opts().is_none());
        // On → Some with the UI's method mapping (opposed is the default).
        p.hl_on = true;
        use c41_core::iop::highlights::{HlMode, HlOpts};
        assert_eq!(
            p.hl_opts(),
            Some(HlOpts { mode: HlMode::Opposed, clip: 1.0 })
        );
        // The dropdown's second entry ("Clip highlights") maps to HlMode::Clip.
        p.hl_opposed = false;
        assert_eq!(
            p.hl_opts(),
            Some(HlOpts { mode: HlMode::Clip, clip: 1.0 })
        );
        p.hl_clip = 1.5;
        assert_eq!(
            p.hl_opts(),
            Some(HlOpts { mode: HlMode::Clip, clip: 1.5 })
        );
    }

    #[test]
    fn filmic_params_map_into_the_stage_and_gate_holds() {
        // The UI fields must land on the right FilmicParams slots (positional,
        // easy to scramble), and an enabled module must emit a Stage::FilmicRgb
        // carrying finished data in the pipeline's working space.
        let mut p = PreviewParams::default();
        p.filmic_on = true;
        p.filmic_black_point_source = -6.5;
        p.filmic_white_point_source = 3.5;
        p.filmic_output_power = 3.0;
        p.filmic_latitude = 10.0;
        p.filmic_contrast = 1.4;
        p.filmic_balance = 20.0;
        p.filmic_saturation = -50.0;
        let pipe = p.to_pipeline(ColorSpace::Rec2020, 1.0);
        let Some(Stage::FilmicRgb { data, space }) =
            pipe.stages.iter().find(|s| s.name() == "filmicrgb")
        else {
            panic!("filmic stage missing: {:?}", pipe.stages.iter().map(|s| s.name()).collect::<Vec<_>>());
        };
        assert_eq!(*space, ColorSpace::Rec2020);
        assert_eq!(data.grey_source, 0.1845);
        assert_eq!(data.black_source, -6.5);
        assert_eq!(data.dynamic_range, 10.0); // 3.5 − (−6.5)
        assert_eq!(data.output_power, 3.0);
        assert_eq!(data.saturation, -0.5); // −50% → weight
        // norm endpoints: grey · 2^(dr·{0,1} + black) — min sits at the black
        // exposure, max one dynamic range above it.
        assert_eq!(data.norm_min, 0.1845f32 * (-6.5f32).exp2());
        assert_eq!(data.norm_max, 0.1845f32 * (3.5f32).exp2());
        // The spline's grey node sits at |black|/dr on the log axis.
        let spline = c41_core::iop::filmicrgb::compute_spline(&p.filmic_params());
        assert_eq!(data.spline, spline);
    }

    #[test]
    fn cbrgb_params_map_into_the_stage_and_gate_holds() {
        // The UI fields must land on the right CbRgbParams slots (the mapping is
        // positional and easy to scramble), the commit-path derivation must run
        // at pipeline-build time (the stage carries finished data), and an
        // enabled module at its defaults must emit NO stage.
        {
            let mut p = PreviewParams::default();
            p.cb_on = true;
            p.cb_global_y = 0.02;
            p.cb_shadows_h = 30.0;
            p.cb_contrast = 0.25;
            p.cb_formula = 0.0; // JzAzBz

            let pipe = p.to_pipeline(ColorSpace::Rec2020, 1.0);
            let stage = pipe
                .stages
                .iter()
                .find(|s| s.name() == "colorbalancergb")
                .expect("colorbalancergb stage should be emitted");
            match stage {
                Stage::ColorBalanceRgb { data, .. } => {
                    use c41_core::iop::colorbalancergb::{
                        CbRgbData, CbRgbParams, SaturationFormula,
                    };
                    // Re-derive from an equal params set and compare: pins the
                    // whole commit_params pass-through field-for-field.
                    let mut expect = CbRgbParams::default();
                    expect.global_y = 0.02;
                    expect.shadows_h = 30.0;
                    expect.contrast = 0.25;
                    expect.saturation_formula = SaturationFormula::Jzazbz;
                    let expected = CbRgbData::from_params(
                        &expect,
                        &c41_core::color::REC2020_TO_XYZ_D65_T4,
                    );
                    assert_eq!(**data, expected);
                }
                _ => panic!("expected Stage::ColorBalanceRgb, got {stage:?}"),
            }
        }
        // The gamut LUT must follow the working space: identical params built
        // for Rec.2020 vs linear sRGB produce different stage data, because the
        // sRGB primaries move the gamut boundary the LUT encodes.
        {
            let stage_data = |sp: ColorSpace| {
                let mut q = PreviewParams::default();
                q.cb_on = true;
                q.cb_saturation_global = 0.4;
                match q
                    .to_pipeline(sp, 1.0)
                    .stages
                    .into_iter()
                    .find(|s| s.name() == "colorbalancergb")
                    .expect("stage should be emitted in both spaces")
                {
                    Stage::ColorBalanceRgb { data, .. } => data,
                    other => panic!("expected Stage::ColorBalanceRgb, got {other:?}"),
                }
            };
            assert_ne!(
                stage_data(ColorSpace::Rec2020),
                stage_data(ColorSpace::LinearSrgb),
                "gamut LUT must track the working space"
            );
        }
        // Enabled at defaults ⇒ no stage (darktable's neutral edit).
        let mut p = PreviewParams::default();
        p.cb_on = true;
        assert!(
            p.to_pipeline(ColorSpace::Rec2020, 1.0)
                .stages
                .iter()
                .all(|s| s.name() != "colorbalancergb"),
            "neutral colorbalancergb params must not emit a stage"
        );
        // Disabled with non-neutral sliders ⇒ still no stage.
        let mut p = PreviewParams::default();
        p.cb_contrast = 0.5;
        assert!(
            p.to_pipeline(ColorSpace::Rec2020, 1.0)
                .stages
                .iter()
                .all(|s| s.name() != "colorbalancergb"),
            "a disabled module must never emit a stage"
        );
        // Default UI params are neutral (guards a default/`cb_is_neutral`
        // divergence the way `default_primaries_params_are_a_true_no_op` does).
        assert!(PreviewParams::default().cb_is_neutral());
    }

    #[test]
    fn cbrgb_global_saturation_boost_end_to_end() {
        // End-to-end through to_pipeline + Pipeline::process: a global
        // saturation boost on a saturated patch must increase chroma while
        // keeping luminance roughly stable (dt-UCS brilliance axis untouched),
        // and the output must stay finite/non-negative (the gamut map's job).
        let mut p = PreviewParams::default();
        // Defaults leave only exposure enabled, inert at ev 0 / black 0 — so
        // the pipeline under test is exactly one ColorBalanceRgb stage.
        p.cb_on = true;
        p.cb_saturation_global = 0.5;
        // A mid orange in Rec.2020-linear terms.
        let input: Vec<f32> = [0.45_f32, 0.22, 0.05, 1.0].repeat(16);
        let out = p.to_pipeline(ColorSpace::Rec2020, 1.0).process(&input, 4, 4);
        for chunk in out.chunks_exact(4) {
            assert!(chunk.iter().all(|v| v.is_finite() && *v >= 0.0), "bad px {chunk:?}");
        }
        let lum = |c: &[f32]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        // Chroma proxy: max channel spread.
        let spread = |c: &[f32]| (c[0] - c[2]).abs();
        assert!(
            spread(&out[0..4]) > spread(&input[0..4]),
            "global saturation should widen channel spread: {} -> {}",
            spread(&input[0..4]),
            spread(&out[0..4])
        );
        let l_in = lum(&input[0..4]);
        let l_out = lum(&out[0..4]);
        assert!(
            (l_out - l_in).abs() < 0.15 * l_in,
            "luminance should stay roughly put under pure saturation: {l_in} -> {l_out}"
        );
    }

    #[test]
    fn layouts_covers_current_version() {
        // PARAMS_LAYOUTS (used by decode and encoded_len_for_version) must
        // track ENCODE_VERSION and ENCODED_LEN in lock-step. Bump one without
        // the other and you get blobs that encode() emits but decode() rejects
        // — a silent round-trip break. This test guards the three-way invariant.
        let &curr = PARAMS_LAYOUTS.last().unwrap();
        assert_eq!(curr.0, ENCODE_VERSION, "PARAMS_LAYOUTS must have a row for the current version");
        assert_eq!(1 + curr.1 + curr.2 * 4, ENCODED_LEN, "current PARAMS_LAYOUTS row must match ENCODED_LEN");
        assert!(PARAMS_LAYOUTS.windows(2).all(|w| w[0].0 < w[1].0), "PARAMS_LAYOUTS must be version-ascending");
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
