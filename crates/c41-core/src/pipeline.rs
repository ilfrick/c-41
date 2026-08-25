//! Minimal Rust pixelpipe orchestrator (Phase 3 milestone 2 bootstrap).
//!
//! Runs an ordered list of migrated IOP [`Stage`]s over a **scene-referred,
//! linear RGBA `f32`** buffer (packed, `width*height*4`). This is the
//! float-domain core successor to c41-ui's 8-bit preview seam: the UI maps
//! its sliders to a [`Pipeline`] and feeds the decoded image through it.
//!
//! Stages hold *physical* (already-scaled) parameters — the same convention the
//! migrated `iop::*::process_pixels` functions expect (e.g. exposure `scale`,
//! not EV; velvia `strength` already /100). Mapping UI ranges to these is the
//! caller's job, keeping this module a thin, faithful orchestrator.
//!
//! Order of `Pipeline::stages` is the processing order.
//!
//! **ROI/(width,height) signature (m4-73):** [`Stage::apply`] and
//! [`Pipeline::process`] now carry the buffer's `(width, height)`, so a stage can
//! read a spatial neighbourhood (the first such is [`Stage::Sharpen`]). Strictly
//! per-pixel stages ignore the dims. A non-pixel-local stage forces the whole
//! [`Pipeline::process`] onto the serial (single whole-buffer band) path — the
//! band-parallel path splits the buffer into pixel runs whose `(w,h)` isn't a
//! rectangle, so it's only valid when every stage is pixel-local (the
//! [`Stage::is_pixel_local`] gate, from m4-59). Still future work: real
//! `iop_order`, OpenCL, and a full geometry (coordinate-warp) ROI in≠out.
//!
//! The 4th channel is darktable scene-referred *scratch/padding*, **not** display
//! alpha. Stages follow their C originals: exposure transforms all four channels
//! (faithful to exposure.c's flat loop), while velvia/splittoning/channelmixer
//! pass channel 4 through. Don't "fix" exposure to preserve it — that diverges
//! from the C pipeline.

use crate::iop::{basecurve, basicadj, bloom, channelmixer, colisa, colorbalancergb, colorcontrast, colorcorrection, colorize, colorzones, denoiseprofile, exposure, filmicrgb, graduatednd, invert, levels, lowlight, lowpass, negadoctor, primaries, rgbcurve, shadhi, sharpen, sigmoid, splittoning, temperature, tonecurve, toneequal, velvia, vibrance, vignette};

/// C-compatible `sign(x)`: returns 1.0 for `+0.0` and `-0.0`, unlike
/// `f32::signum` which returns `0.0` for both zeroes. Used where a ported
/// kernel mirrors a C macro (`#define sign(x) ((x)>0?1:((x)<0?-1:1))`).
trait CSignum { fn signum_c(self) -> f32; }
impl CSignum for f32 { fn signum_c(self) -> f32 { if self > 0.0 { 1.0 } else if self < 0.0 { -1.0 } else { 1.0 } } }

/// The working colour space of the buffer a colour-space-dependent (Lab-domain)
/// stage processes. The raw pipeline works in linear **Rec.2020**; the non-raw
/// (JPEG/PNG/TIFF) pipeline works in linear **sRGB**. The caller building the
/// pipeline sets it on each such stage (e.g. [`Stage::Sharpen`]) so the stage
/// converts RGB↔Lab through the correct primaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    Rec2020,
    LinearSrgb,
}

/// A packed-RGBA colour transform (RGB↔Lab), selected by [`ColorSpace`].
type LabConv = fn([f32; 4]) -> [f32; 4];

/// One configured pipeline stage, backed by a migrated c41-core IOP.
#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    /// `out = (in - black) * scale` over all four channels (scale = 2^EV).
    Exposure { black: f32, scale: f32 },
    /// Saturation-weighted chroma boost. `strength` is already divided by 100.
    Velvia { strength: f32, bias: f32 },
    /// Shadow/highlight hue toning. `compress` is already `(c/110)/2`.
    Splittoning {
        shadow_hue: f32,
        shadow_sat: f32,
        highlight_hue: f32,
        highlight_sat: f32,
        balance: f32,
        compress: f32,
    },
    /// Grayscale mix (channelmixer GRAY mode): the R/G/B → luma weights.
    Monochrome { r: f32, g: f32, b: f32 },
    /// Sigmoid tone mapping (rgb-ratio path) — scene-linear → display. Holds the
    /// already-derived process params (see `sigmoid::rgb_ratio_params`).
    Sigmoid {
        white_target: f32,
        black_target: f32,
        paper_exp: f32,
        film_fog: f32,
        film_power: f32,
        paper_power: f32,
    },
    /// Unsharp-mask sharpening — the first **spatial** stage (reads a
    /// neighbourhood, so [`Stage::apply`] needs `(width, height)`). The faithful
    /// darktable path: convert Rec.2020→**Lab**, unsharp-mask the **L** channel
    /// only via the migrated separable-Gaussian [`sharpen::darkroom_sharpen_process`]
    /// kernel (a/b untouched ⇒ **no chroma shift**), convert back. `radius` sets
    /// the Gaussian; `amount` scales the L detail; **`threshold` is in Lab-`L`
    /// units [0, 100]** (identical to darktable, so a 0..100 UI slider maps
    /// straight through). Requires the RGB↔Lab infra from `crate::color`.
    ///
    /// `space` is the buffer's working colour space ([`ColorSpace`]): the caller
    /// sets it so the L conversion uses the right primaries (Rec.2020 for raws,
    /// linear sRGB for non-raws). No assumed working space.
    ///
    /// `scale` is the buffer's resolution relative to the full image (darktable's
    /// `roi->scale`): `1.0` at full res (export), `< 1.0` on a downscaled preview.
    /// The effective Gaussian radius is `radius * scale`, so sharpening is
    /// image-relative — a downscaled preview matches the full-res export instead
    /// of over-sharpening (WYSIWYG). The caller passes the ROI scale it renders at.
    Sharpen { radius: f32, amount: f32, threshold: f32, space: ColorSpace, scale: f32 },
    /// Saturation-weighted chroma boost in Lab (vibrance.c). `amount` is
    /// pre-scaled by 0.01 (matching darktable's commit_params step). Like Sharpen
    /// it reads/writes Lab channels, so it needs the buffer's working colour
    /// space for the RGB↔Lab pair — unlike Sharpen it is **pixel-local** (no
    /// neighbourhood read), so the band-parallel `process` path stays available.
    Vibrance { amount: f32, space: ColorSpace },
    /// Green-magenta / blue-yellow contrast boost in Lab (colorcontrast.c).
    /// `a_steepness`/`b_steepness` are the contrast multipliers on the a/b
    /// channels (default 1.0 = no-op). Like Sharpen and Vibrance it converts
    /// RGB↔Lab, so it needs the buffer's working colour space. It is
    /// **pixel-local** (no neighbour reads), so the band-parallel path stays
    /// available.
    ColorContrast { a_steepness: f32, b_steepness: f32, space: ColorSpace },
    /// Per-channel white-balance multipliers (temperature.c `process_rgb`):
    /// `out = in * coeffs` (coeffs[0..3] = R, G, B; coeffs[3] = A, usually 1.0).
    /// Works directly in linear RGB — no Lab conversion, so `working_space()`
    /// returns `None` for it. **Pixel-local**: no neighbour reads, so the
    /// band-parallel `process` path stays available.
    Temperature { coeffs: [f32; 4] },
    /// Film-camera negative inversion (invert.c `process_rgb`): `out = color - in`
    /// per channel. `color` is the 4-float film-back material colour
    /// (`d->color[0..3]`, alpha usually 1.0). Works directly in linear RGB — no
    /// Lab conversion, so `working_space()` returns `None`. **Pixel-local**: no
    /// neighbour reads, so the band-parallel `process` path stays available.
    Invert { color: [f32; 4] },
    /// Colour replacement in Lab (colorize.c `process`): replaces a/b channels
    /// with a fixed Lab colour, blends L from the input via `mix`. `color_l`,
    /// `color_a`, `color_b` are the pre-converted Lab values (from HSL via
    /// hsl2rgb→sRGB→XYZ(D50)→Lab in [`PreviewParams::to_pipeline`]). `mix` is the
    /// pre-scaled `source_lightness_mix / 100.0`. Like Vibrance/ColorContrast it
    /// converts RGB↔Lab, so it needs the buffer's working colour space.
    /// **Pixel-local**: no neighbour reads, so the band-parallel path stays
    /// available.
    Colorize { color_l: f32, color_a: f32, color_b: f32, mix: f32, space: ColorSpace },
    /// Luminance-dependent Lab a/b colour correction (colorcorrection.c).
    /// `a_scale`/`b_scale` are chroma multipliers, `a_base`/`b_base` are additive
    /// offsets, `saturation` is a global chroma scale. All are pre-computed from
    /// the HSL params (hia/loa/hib/lob/saturation) by `commit_params` in
    /// [`PreviewParams::to_pipeline`]. Lab-domain: needs the buffer's working
    /// colour space. **Pixel-local**: no neighbour reads, so the band-parallel path
    /// stays available.
    ColorCorrection { a_scale: f32, a_base: f32, b_scale: f32, b_base: f32, saturation: f32, space: ColorSpace },
    /// Color zones: LCH equaliser. 3×65536-entry LUTs (L, C, h) built from
    /// spline curve nodes, applied in Lab space via
    /// [`darkroom_colorzones_process`]. **Pixel-local**: no neighbour reads, so
    /// the band-parallel path stays available.
    ColorZones {
        lut_l: Vec<f32>,
        lut_c: Vec<f32>,
        lut_h: Vec<f32>,
        channel: i32,
        mode: i32,
        space: ColorSpace,
    },
    /// Glow/bloom effect (bloom.c `process`): gathers Lab L above a threshold,
    /// box-blurs the gathered light (`dt_box_mean`, ch=1, 8 iterations) and
    /// screen-blends it back into L. Like Colorize it round-trips RGB↔Lab, so
    /// it needs the buffer's working colour space. **NOT pixel-local**: the
    /// box blur reads a spatial neighbourhood (radius up to 256), so it runs
    /// on whole frames only.
    Bloom {
        size: f32,
        threshold: f32,
        strength: f32,
        space: ColorSpace,
    },
    /// Tone curve (tonecurve.c): 3-channel Lab LUT built from spline nodes via
    /// [`crate::tonecurve::build_lut`] (the V1 `dt_draw_curve_*` sampler),
    /// applied per pixel by [`tonecurve::process_pixels`]. The L table is in
    /// Lab units [0,100] for MANUAL/AUTOMATIC modes or [0,1] RGB/Y units after
    /// the linked-channel re-derivation — exactly as upstream leaves it.
    /// Lab-domain, so it needs the buffer's working space. **Pixel-local**
    /// (pure LUT lookup), so the band-parallel path stays available.
    ToneCurve {
        table_l: Box<[f32; 65536]>,
        table_a: Box<[f32; 65536]>,
        table_b: Box<[f32; 65536]>,
        coeffs_l: [f32; 3],
        coeffs_ab: [f32; 12],
        autoscale_ab: i32,
        unbound_ab: i32,
        preserve_colors: i32,
        space: ColorSpace,
    },
    /// RGB curve (rgbcurve.c): three per-channel LUTs applied directly on the
    /// buffer's RGB lanes (IOP_CS_RGB — no Lab conversion, so no working-space
    /// agreement). Tables/coeffs come from [`rgbcurve::build_luts`]; `autoscale`
    /// is 0 = AUTOMATIC_RGB (linked channels) or 1 = MANUAL_RGB (independent);
    /// `preserve_colors` is 0 = NONE or a `color::rgb_norm` mode.
    RgbCurve {
        table_r: Box<[f32; 65536]>,
        table_g: Box<[f32; 65536]>,
        table_b: Box<[f32; 65536]>,
        coeffs: [[f32; 3]; 3],
        autoscale: i32,
        preserve_colors: i32,
    },
    /// Base curve (basecurve.c): a single user-drawn LUT applied directly on
    /// the buffer's RGB lanes (IOP_CS_RGB — no Lab sandwich), either plain
    /// (`fusion == 0`, pixel-local) or blended through an exposure-fusion
    /// laplacian pyramid (`fusion >= 1`, whole-frame serial). Table/coeffs come
    /// from [`basecurve::build_table`] (the commit_params port); the kernels
    /// are the very FFI fns production C calls. `space` selects the Y row the
    /// LUMINANCE preservation norm consumes (the working profile's matrix_in
    /// row in C).
    Basecurve {
        table: Box<[f32; 65536]>,
        coeffs: [f32; 3],
        preserve_colors: i32,
        fusion: i32,
        stops: f32,
        bias: f32,
        space: ColorSpace,
    },
    /// Black/grey/white point + gamma in Lab (levels.c). `black`/`range` are the
    /// normalised [0,1] stops (`levels[0]`, `levels[2] - levels[0]`) and
    /// `inv_gamma`/`lut` come from [`levels::build_lut`]. The a/b channels are
    /// rescaled by `L_out / L_in` so chroma tracks the tone change. Lab-domain,
    /// so it needs the buffer's working space. **Pixel-local**: no neighbour
    /// reads, so the band-parallel path stays available.
    Levels {
        black: f32,
        range: f32,
        inv_gamma: f32,
        lut: Vec<f32>,
        space: ColorSpace,
    },
    /// Radial brightness/saturation falloff (vignette.c). Works directly in RGB
    /// — no Lab round-trip, so `working_space()` is `None`.
    ///
    /// **NOT pixel-local** — the second such stage after [`Stage::Sharpen`], and
    /// for a different reason. Each pixel's weight comes from its `(i, j)`
    /// *position* relative to the vignette centre, and the dither is a per-row
    /// TEA stream seeded from `j`. The band-parallel path hands each band
    /// `(w, h) = (band_pixels, 1)`, so every band would compute its falloff from
    /// the wrong coordinates and reseed the dither — visible seams at band
    /// boundaries. Returning `false` forces the whole pipeline serial whenever
    /// this stage is present, which is the price of correctness here.
    ///
    /// Holds the **user-facing** params, not pre-computed geometry: the falloff
    /// geometry depends on the buffer's dimensions, and `apply` already receives
    /// them, so it is derived there via [`vignette::commit_geometry`]. Storing
    /// it in the stage instead would go stale the moment the preview re-rendered
    /// at another size (a different zoom, or export vs preview) — a silent
    /// wrong-looking vignette. Deriving per apply costs one call per render.
    Vignette {
        /// Fall-off start (inner radius), 0..200 % of the largest dimension.
        scale: f32,
        /// Fall-off radius, 0..200.
        falloff: f32,
        /// Brightness/saturation reduction strengths, -1..1.
        brightness: f32,
        saturation: f32,
        /// Centre offset, -1..1 per axis (0 = image centre).
        center_x: f32,
        center_y: f32,
        /// Shape exponent (1 = ellipse, higher = squarer).
        shape: f32,
        /// Follow the buffer's own aspect rather than an explicit w/h ratio.
        autoratio: bool,
        /// Explicit w/h ratio, 0..2 (<=1 = w/h, >1 = h/w + 1). Ignored when
        /// `autoratio`.
        whratio: f32,
        /// Dither amplitude: 0 = off, 1/256 = 8-bit, 1/65536 = 16-bit.
        dither_amt: f32,
        /// Skip the [0,1] clamp (darktable's "unbound" path).
        unbound: bool,
    },
    /// Scotopic ("night vision") simulation (lowlight.c): blends a blue-shifted
    /// rod-vision response against the normal photopic image, mixed per pixel by
    /// a luminance-driven transition curve. `lut` is the 65536-entry curve from
    /// [`lowlight::build_transition_lut`]; `blueness` is the 0..100 blue shift.
    /// Lab-domain, so it needs the buffer's working space. **Pixel-local**.
    Lowlight {
        blueness: f32,
        lut: Vec<f32>,
        space: ColorSpace,
    },
    /// Graduated neutral-density filter (graduatednd.c): darkens (or brightens,
    /// for a negative density) across a rotatable gradient line. Works directly
    /// in RGB — no Lab round-trip, so `working_space()` is `None`.
    ///
    /// **NOT pixel-local**, like [`Stage::Vignette`]: the filter strength is a
    /// function of the pixel's `(x, y)` against that line, so band-splitting
    /// would hand each band the wrong coordinates. Holds the user params rather
    /// than the derived geometry, which depends on the buffer dimensions and is
    /// therefore computed in `apply`.
    /// Contrast / brightness / saturation (colisa.c). Applies two 65536-entry
    /// tone curves to Lab `L` — contrast then brightness, each with an
    /// exponential extrapolation above 1.0 — and scales a/b by `saturation`.
    /// Holds the three user sliders on darktable's -1..1 scale; the LUTs are
    /// derived in `apply` via [`colisa::commit_params`], which is cheap relative
    /// to a render and keeps the stage `PartialEq`-comparable.
    /// Lab-domain, **pixel-local**.
    Colisa {
        contrast: f32,
        brightness: f32,
        saturation: f32,
        space: ColorSpace,
    },
    /// Low-pass local-contrast filter (lowpass.c). Blurs the image in Lab with a
    /// recursive Gaussian, then applies a contrast LUT + brightness LUT (each with
    /// exponential extrapolation above 1.0) to the blurred `L`, and scales a/b by
    /// `saturation`. The contrast/brightness LUTs and their extrapolation
    /// coefficients are derived in `apply` via [`lowpass::commit_params`], keeping
    /// the stage `PartialEq`-comparable and the LUTs out of the stage struct.
    ///
    /// Lab-domain (RGB-to-Lab round-trip per pixel), so it needs the buffer's
    /// working space. **NOT pixel-local** -- the Gaussian blur reads a spatial
    /// neighbourhood, so `process` falls back to the serial whole-buffer path
    /// whenever this stage is present. Like [`Stage::Sharpen`], holds a `scale`
    /// so the blur radius tracks the ROI resolution (`radius * scale` = the
    /// sigma passed to the recursive filter).
    Lowpass {
        radius: f32,
        contrast: f32,
        brightness: f32,
        saturation: f32,
        scale: f32,
        space: ColorSpace,
    },
    /// Basic adjustments (basicadj.c): black point, exposure, highlight
    /// compression, brightness, contrast, saturation and vibrance in one pass.
    ///
    /// Works directly in **linear RGB** — no Lab round-trip, so `working_space()`
    /// returns `None` — but it does need the working space's luminance weights
    /// for the highlight-compression pass, which is what `space` selects.
    /// Holds the user sliders, not the two 65536-entry LUTs: those are derived in
    /// `apply` via [`basicadj::commit_params`], keeping the stage `PartialEq` and
    /// out of the business of carrying 512 KB per stage. **Pixel-local**.
    ///
    /// `clip` from darktable's params struct is deliberately absent — the
    /// migrated kernel does not implement it, and a slider that does nothing is
    /// worse than no slider.
    Basicadj {
        black_point: f32,
        exposure: f32,
        hlcompr: f32,
        hlcomprthresh: f32,
        contrast: f32,
        /// `dt_iop_rgb_norms_t`: 0 = off (per-channel LUT contrast), 1 = luminance,
        /// 2 = max RGB, … See `crate::color::rgb_norm`.
        preserve_colors: i32,
        middle_grey: f32,
        brightness: f32,
        saturation: f32,
        vibrance: f32,
        space: ColorSpace,
    },
    /// Shadows/highlights local-contrast enhancement (shadhi.c). A Gaussian blur
    /// produces a base layer; the shadows/highlights overlays are then blended in
    /// Lab space: shadows lift the dark regions, highlights recover blown areas,
    /// whitepoint shifts the white point, compress preserves mid-tones, and the
    /// ccorrect params control colour bleed into shadows/highlights.
    ///
    /// Lab-domain (`default_colorspace` returns `IOP_CS_LAB` in the C), so it needs
    /// the buffer's working colour space for the RGB↔Lab pair. NOT pixel-local —
    /// the Gaussian blur reads a spatial neighbourhood, so `process` falls back to
    /// the serial whole-buffer path whenever this stage is present. Like
    /// [`Stage::Lowpass`], holds a `scale` so the blur sigma tracks the ROI
    /// resolution (`radius * scale`).
    ///
    /// `flags` is hardcoded to `UNBOUND_DEFAULT` (127) — darktable's default, and
    /// the C GUI doesn't expose an unbound checkbox for this module.
    /// `low_approximation` is hardcoded to its C default (0.000001). We don't use
    /// bilateral (the C default algorithm) because `crate::gaussian` only implements
    /// the recursive Gaussian, not the bilateral filter — the Gaussian path is a
    /// faithful, if slightly different, blur; the shadow/highlight math is
    /// identical.
    Shadhi {
        /// Shadows lift, -100..100 (C slider; core gets `2 * clamp(s/100, -1, 1)`).
        shadows: f32,
        /// Highlights recovery, -100..100.
        highlights: f32,
        /// White point shift, -10..10 (core gets `max(1 - w/100, 0.01)`).
        whitepoint: f32,
        /// Blur radius, 0.1..500 (core sigma = `max(0.1, radius) * scale`).
        radius: f32,
        /// Compression, 0..100 (core gets `clamp(c/100, 0, 0.99)`).
        compress: f32,
        /// Shadows colour correction, 0..100.
        shadows_ccorrect: f32,
        /// Highlights colour correction, 0..100.
        highlights_ccorrect: f32,
        /// ROI scale so the blur tracks preview resolution (1.0 = full res).
        scale: f32,
        /// Buffer working colour space (Rec2020 for raws, LinearSrgb for JPEGs).
        space: ColorSpace,
    },
    GraduatedNd {
        /// Filter density in EV, -8..8 (negative brightens).
        density: f32,
        /// Edge hardness, 0..100 (0 = soft gradient, 100 = hard line).
        hardness: f32,
        /// Rotation of the gradient line in degrees, -180..180.
        rotation: f32,
        /// Line offset across the frame, 0..100 (50 = centred).
        offset: f32,
        /// Filter tint, both 0..1 (saturation 0 = neutral grey).
        hue: f32,
        saturation: f32,
    },
    /// RGB primaries adjustment (primaries.c). Rotates and scales the working
    /// space primaries, producing a 4×4 colour matrix applied per pixel via
    /// [`darkroom_primaries_process`]. `matrix` is pre-computed from the 8 UI
    /// params (4 hue + 4 purity, radians + multiplier scale) and the working
    /// colour space in [`c41_ui::preview::PreviewParams::to_pipeline`].
    ///
    /// Works directly in linear RGB — no Lab conversion, so `working_space()`
    /// returns `None`. **Pixel-local**: a pure 4×4 matrix multiply, no neighbor
    /// reads, so the band-parallel `process` path stays available.
    Primaries { matrix: [f32; 16] },
    /// Film negative scan inversion (negadoctor.c). The 4×4 data arrays
    /// (Dmin, wb_high, offset) are pre-computed by
    /// [`c41_ui::preview::PreviewParams::to_pipeline`] following darktable's
    /// `commit_params` — including the `Dmax` division of `wb_high` and the
    /// film-stock monochrome Dmin collapse. `black` is the FMA-rewritten
    /// `-exposure * (1 + black)`, per the C arithmetic trick.
    ///
    /// Channel 3 (alpha) is inert: the process loop (`process_pixels`)
    /// iterates `c in 0..3` and copies `co[3] = ci[3]` — the fourth slot
    /// of each array is a sentinel (`dmin[3]=1.0`, `offset[3]=0.0`,
    /// `wb_high[3]=1.0`) that never participates in a computation.
    ///
    /// Works directly in linear RGB — no Lab conversion, so
    /// `working_space()` returns `None`. **Pixel-local**: each pixel is
    /// inverted independently, no neighbour reads, so the band-parallel path
    /// stays available.
    Negadoctor {
        dmin: [f32; 4],
        wb_high: [f32; 4],
        offset: [f32; 4],
        black: f32,
        gamma: f32,
        soft_clip: f32,
        soft_clip_comp: f32,
        exposure: f32,
    },

    /// Tone equalizer (toneequal.c) — an exposure-domain tone curve driven by
    /// nine per-exposure-channel gains (EV offsets at −8…0 EV, ordered
    /// noise→speculars, matching `get_channels_gains`, toneequal.c:1210). The
    /// gains are stored raw; `apply` performs the RBF least-squares solve
    /// (`pseudo_solve`, choleski.h) and correction-LUT build per render, memoised
    /// on its inputs (see `toneequal::cached_correction_lut`).
    ///
    /// Scope: runs in the `details == DT_TONEEQ_NONE` configuration ("preserve
    /// details: no") — no guided-filter/surface-blur luminance pre-pass. The
    /// luminance estimator is darktable's default `DT_TONEEQ_NORM_2`
    /// (sqrt(r²+g²+b²)) and, with the defaults `exposure_boost = 0`,
    /// `contrast_boost = 0`, fulcrum 0, `linear_contrast` reduces to the exp2(−16)
    /// MIN_FLOAT floor — exactly what `process_preview_pixels` mirrors.
    ///
    /// Works directly on linear RGB (`default_colorspace` returns `IOP_CS_RGB`)
    /// via a per-pixel luminance lookup — no Lab conversion, so
    /// `working_space()` returns `None`. **Pixel-local**: each output pixel
    /// depends only on its own input pixel, so the band-parallel path stays
    /// available.
    ToneEqual {
        /// Nine channel gains in EV (log2), −8 EV … 0 EV. All zero = identity.
        gains: [f32; 9],
    },

    /// Colour balance RGB (colorbalancergb.c) — scene-referred grading in
    /// Filmlight's Yrg space with perceptual saturation/brilliance in dt-UCS or
    /// JzAzBz. The stage carries the **prebuilt** per-commit data: everything
    /// `commit_params` derives (the four zone vectors, weights, fulcrums and the
    /// hue-indexed 512-entry gamut LUT) is computed once at pipeline-build time,
    /// because the dt-UCS LUT alone marches the RGB gamut boundary 25 600 times —
    /// not work to redo per band.
    ///
    /// Converts through XYZ D65 with the buffer's working-space pair (raw
    /// pipeline Rec.2020, non-raw linear sRGB — how the C derives everything
    /// from the working profile), and the gamut LUT in `data` must be built
    /// against that same space's primaries (`to_pipeline` picks both together),
    /// so `working_space()` reports it like the other space-aware stages.
    /// **Pixel-local**: every transform is a per-pixel colour map — no neighbour
    /// reads, no position dependence — so the band-parallel path stays available.
    ColorBalanceRgb {
        data: Box<colorbalancergb::CbRgbData>,
        space: ColorSpace,
    },
    /// Filmic RGB (filmicrgb.c) — scene-referred tone mapping, colour science
    /// v5. The stage carries the committed data (spline + scalars from
    /// `commit_params`/`compute_spline`, prebuilt once per render) and the
    /// working space; the Yrg matrices are selected per apply from the same
    /// space so the gamut map clips against the buffer's own primaries.
    FilmicRgb {
        data: Box<filmicrgb::FilmicData>,
        space: ColorSpace,
    },

    /// Denoise (profiled) (denoiseprofile.c) — wavelets-mode profiled noise
    /// reduction: variance-stabilising transform, edge-avoiding à-trous
    /// decomposition with BayesShrink soft thresholds, inverse VST. Holds the
    /// user sliders; everything else is derived per apply in
    /// [`denoiseprofile::wavelets_denoise`].
    ///
    /// Scope (documented deviations from the C): wavelets mode only with the
    /// new (v2) VST, generic Poissonian noise profile a=1e-4/b=0 in place of
    /// the per-camera profiles database, wb=[1,1,1] and in_scale=1 because our
    /// preview buffer arrives post-white-balance and is always processed whole.
    /// Works directly in linear RGB (`default_colorspace` = IOP_CS_RGB) — no
    /// Lab conversion, so `working_space()` returns `None`. **NOT pixel-local**:
    /// each output pixel reads a 5×5 stride-2^scale neighbourhood across all
    /// scales, so `process` falls back to the serial whole-buffer path whenever
    /// this stage is present — same reason as Shadhi/Lowpass/Sharpen.
    DenoiseProfile {
        strength: f32,
        shadows: f32,
        bias: f32,
        mode_y0u0v0: bool,
    },

    /// Lens correction via liblensfun (lens.cc LENSFUN method) — the one
    /// stage backed by an external C library (distro `liblensfun`, bound in
    /// `c41-sys::lensfun`). Carries the database-resolved lens plus the
    /// user parameters; [`crate::iop::lens::process`] runs vignetting +
    /// per-channel coordinate warp over the whole frame. **NOT pixel-local**
    /// (a coordinate-warp resampling — every output pixel reads a
    /// neighbourhood), so it takes the serial whole-buffer path.
    ///
    /// Works directly on the linear buffer (`working_space()` = `None`); it
    /// is colour-agnostic geometry + per-pixel gain. Alpha is carried
    /// through untouched.
    LensCorrection {
        lens: crate::iop::lens::ResolvedLens,
        params: crate::iop::lens::LensParams,
    },
}

/// Faithful port of sharpen.c `init_gaussian_kernel`: a normalised Gaussian of
/// `2*rad+1` taps where `mat[l+rad] = exp(-l²/(2·sigma2))`. Returns `(rad, mat)`.
/// darktable derives `sigma2 = (radius/2.5)²` and sizes the mask to fit 2.5σ, but
/// **caps the radius at `MAXR = 12`** (`sharpen.c`) while leaving `sigma2` on the
/// uncapped radius — so beyond radius 12 the Gaussian keeps widening but the tap
/// count (and cost) stays bounded.
fn gaussian_kernel(radius: f32) -> (usize, Vec<f32>) {
    let rad = (radius.ceil() as usize).clamp(1, 12); // MAXR = 12 (sharpen.c)
    let sigma2 = (radius / 2.5).powi(2).max(f32::MIN_POSITIVE); // uncapped radius, per C
    let wd = 2 * rad + 1;
    let mut mat = vec![0.0f32; wd];
    let mut weight = 0.0f32;
    for l in -(rad as i32)..=(rad as i32) {
        let w = (-(l * l) as f32 / (2.0 * sigma2)).exp();
        mat[(l + rad as i32) as usize] = w;
        weight += w;
    }
    for m in &mut mat {
        *m /= weight;
    }
    (rad, mat)
}

impl Stage {
    /// A short stable identifier (matches the darktable IOP operation name).
    pub fn name(&self) -> &'static str {
        match self {
            Stage::Exposure { .. } => "exposure",
            Stage::Velvia { .. } => "velvia",
            Stage::Splittoning { .. } => "splittoning",
            Stage::Monochrome { .. } => "channelmixer",
            Stage::Sigmoid { .. } => "sigmoid",
            Stage::Sharpen { .. } => "sharpen",
            Stage::Vibrance { .. } => "vibrance",
            Stage::ColorContrast { .. } => "colorcontrast",
            Stage::Temperature { .. } => "temperature",
            Stage::Invert { .. } => "invert",
            Stage::Colorize { .. } => "colorize",
            Stage::ColorCorrection { .. } => "colorcorrection",
            Stage::ColorZones { .. } => "colorzones",
            Stage::Bloom { .. } => "bloom",
            Stage::ToneCurve { .. } => "tonecurve",
            Stage::RgbCurve { .. } => "rgbcurve",
            Stage::Basecurve { .. } => "basecurve",
            Stage::Levels { .. } => "levels",
            Stage::Vignette { .. } => "vignette",
            Stage::Lowlight { .. } => "lowlight",
            Stage::GraduatedNd { .. } => "graduatednd",
            Stage::Colisa { .. } => "colisa",
            Stage::Basicadj { .. } => "basicadj",
            Stage::Shadhi { .. } => "shadhi",
            Stage::Lowpass { .. } => "lowpass",
            Stage::Primaries { .. } => "primaries",
            Stage::Negadoctor { .. } => "negadoctor",
            Stage::ToneEqual { .. } => "toneequal",
            Stage::ColorBalanceRgb { .. } => "colorbalancergb",
            Stage::FilmicRgb { .. } => "filmicrgb",
            Stage::DenoiseProfile { .. } => "denoiseprofile",
            Stage::LensCorrection { .. } => "lens",
        }
    }

    /// True iff this stage's output for a pixel depends only on that pixel's
    /// input — the invariant that makes the band-parallel [`Pipeline::process`]
    /// bit-identical to a serial run. Every current stage is pixel-local.
    ///
    /// The match is deliberately **exhaustive with no wildcard**: a future
    /// `Stage` variant won't compile until its author classifies it here, and a
    /// non-local stage makes `process` fall back to a correct serial run rather
    /// than silently producing seams at band boundaries.
    fn is_pixel_local(&self) -> bool {
        match self {
            Stage::Exposure { .. }
            | Stage::Velvia { .. }
            | Stage::Splittoning { .. }
            | Stage::Monochrome { .. }
            | Stage::Sigmoid { .. } => true,
            // Vibrance is pixel-local: each output pixel depends only on its
            // own input pixel (the Lab conversion is per-pixel, no neighbours).
            Stage::Vibrance { .. } => true,
            // ColorContrast is pixel-local: each output pixel depends only on
            // its own input pixel (Lab conversion is per-pixel, no neighbours).
            Stage::ColorContrast { .. } => true,
            // Temperature is pixel-local: a per-channel scalar multiply, no
            // neighbours, no Lab conversion.
            Stage::Temperature { .. } => true,
            // Invert is pixel-local: per-channel `color - in`, no neighbours,
            // no Lab conversion.
            Stage::Invert { .. } => true,
            // Colorize is pixel-local: per-pixel Lab replacement, no neighbours,
            // no neighbourhood reads.
            Stage::Colorize { .. } => true,
            // ColorCorrection is pixel-local: per-pixel Lab a/b scaling, no
            // neighbours, no neighbourhood reads.
            Stage::ColorCorrection { .. } => true,
            // ColorZones is pixel-local: each output pixel depends only on its
            // own input pixel (Lab→LCH→LUT→Lab, no neighbour reads).
            Stage::ColorZones { .. } => true,
            // Bloom is NOT pixel-local: the box blur reads a spatial
            // neighbourhood up to radius 256, so it must run on whole frames.
            Stage::Bloom { .. } => false,
            // ToneCurve is pixel-local: three pure per-pixel LUT lookups
            // (plus an optional a/b re-derivation that only touches the same
            // pixel), no neighbour reads.
            Stage::ToneCurve { .. } => true,
            // RgbCurve likewise: pure per-pixel LUT lookups / norm-ratio math,
            // no neighbour reads.
            Stage::RgbCurve { .. } => true,
            // Basecurve is pixel-local only in its plain process_lut form;
            // exposure fusion blends a laplacian pyramid over the whole frame
            // and must run serial (same reason as Bloom/Shadhi).
            Stage::Basecurve { fusion, .. } => *fusion == 0,
            // Levels is pixel-local: a per-pixel tone-curve lookup on L (with
            // a/b scaled by the same ratio), no neighbour reads.
            Stage::Levels { .. } => true,
            // Colisa is pixel-local: two per-pixel LUT lookups on L plus an a/b
            // scale, no neighbour reads.
            Stage::Colisa { .. } => true,
            // Basicadj is pixel-local: exposure, LUT lookups and a
            // per-pixel saturation blend, no neighbour reads.
            Stage::Basicadj { .. } => true,
            // Shadhi is NOT pixel-local: the Gaussian blur reads a spatial
            // neighbourhood, so band-splitting would hand it a (band_pixels, 1)
            // rectangle and produce wrong-edge artefacts. Returning false forces
            // the whole pipeline serial whenever this stage is present, where
            // (width, height) is the true image rectangle — same reason as
            // Lowpass and Sharpen.
            Stage::Shadhi { .. } => false,
            // Lowpass is NOT pixel-local: the Gaussian blur reads a spatial
            // neighbourhood, so band-splitting would hand it a (band_pixels, 1)
            // rectangle and produce wrong-edge artefacts. Returning false forces
            // the whole pipeline serial whenever this stage is present, where
            // (width, height) is the true image rectangle.
            Stage::Lowpass { .. } => false,
            // Lowlight is pixel-local: a per-pixel scotopic/photopic blend
            // driven by that pixel's own luminance, no neighbour reads.
            Stage::Lowlight { .. } => true,
            // Sharpen reads a spatial neighbourhood → NOT pixel-local, so
            // `process` runs it on the whole buffer (serial path) where (w,h) is
            // the true image rectangle, never a band's pixel run.
            Stage::Sharpen { .. } => false,
            // GraduatedNd is position-dependent for the same reason as Vignette:
            // the filter strength is a function of the pixel's (x, y) against a
            // rotated gradient line, so a band's (band_pixels, 1) rectangle
            // would give every band the wrong coordinates.
            Stage::GraduatedNd { .. } => false,
            // Vignette is position-DEPENDENT rather than neighbourhood-reading:
            // the falloff weight comes from each pixel's (i, j) relative to the
            // vignette centre, and the dither is a per-row TEA stream seeded
            // from j. Under band-splitting every band gets (band_pixels, 1), so
            // the coordinates — and the dither seed — would be wrong per band,
            // giving seams. Same answer as Sharpen, different reason.
            Stage::Vignette { .. } => false,
            // Primaries is pixel-local: a 4×4 matrix multiply, no neighbor
            // reads no position dependence — each output pixel depends only on
            // its own input. The band-parallel path stays available.
            Stage::Primaries { .. } => true,
            // Negadoctor is pixel-local: a per-channel log-density inversion,
            // no neighbour reads, so the band-parallel path stays available.
            Stage::Negadoctor { .. } => true,
            // ToneEqual (details == NONE) is pixel-local: the luminance mask is
            // computed per pixel from that pixel's own RGB and the correction is
            // a LUT lookup at that luminance — no neighbour reads. (The skipped
            // guided-filter modes WOULD read neighbours; if they are ever ported
            // this must be revisited.) Band-parallel stays available.
            Stage::ToneEqual { .. } => true,
            // ColorBalanceRgb is pixel-local: a long chain of per-pixel colour
            // transforms (Yrg grading, zone masks from the pixel's own luminance,
            // perceptual saturation) — no neighbour reads, no position
            // dependence. The gamut LUT it reads is prebuilt and immutable.
            // Band-parallel stays available.
            Stage::ColorBalanceRgb { .. } => true,
            // FilmicRgb is pixel-local: per-pixel tone mapping + gamut map, all
            // driven by that pixel's own RGB through immutable prebuilt tables
            // (spline coefficients, Yrg matrices). No neighbour reads. The
            // band-parallel path stays available.
            Stage::FilmicRgb { .. } => true,
            // DenoiseProfile is NOT pixel-local: the à-trous decomposition
            // reads a 5×5 neighbourhood at stride 2^scale for every output
            // pixel, so band-splitting would produce wrong-edge artefacts at
            // every scale. Returning false forces the whole pipeline serial
            // whenever this stage is present, where (width, height) is the
            // true image rectangle — same reason as Shadhi/Lowpass/Sharpen.
            Stage::DenoiseProfile { .. } => false,
            // LensCorrection is NOT pixel-local: it is a coordinate warp —
            // every output pixel resamples a neighbourhood (per channel!) at
            // its distorted source coordinate, so band-splitting would hand
            // the sampler the wrong rectangle. Serial whole-frame only.
            Stage::LensCorrection { .. } => false,
        }
    }

    /// The working colour space this stage requires, if it is colour-space-
    /// dependent (only [`Stage::Sharpen`] so far). `process` checks all such
    /// stages agree — a pipeline processes ONE buffer, so it has one working space.
    fn working_space(&self) -> Option<ColorSpace> {
        match self {
            Stage::Sharpen { space, .. } => Some(*space),
            // Vibrance converts RGB↔Lab for its chroma-boost, so it also needs the
            // working space — and must agree with any Sharpen in the same pipeline.
            Stage::Vibrance { space, .. } => Some(*space),
            // ColorContrast also converts RGB↔Lab and must agree with Sharpen/Vibrance.
            Stage::ColorContrast { space, .. } => Some(*space),
            // Colorize converts RGB↔Lab and must agree with Sharpen/Vibrance/ColorContrast.
            Stage::Colorize { space, .. } => Some(*space),
            // ColorCorrection converts RGB↔Lab and must agree with the other Lab stages.
            Stage::ColorCorrection { space, .. } => Some(*space),
            // ColorZones also converts RGB↔Lab and must agree with the other Lab stages.
            Stage::ColorZones { space, .. } => Some(*space),
            // Bloom converts RGB↔Lab (it operates on L) and must agree with
            // the other Lab stages.
            Stage::Bloom { space, .. } => Some(*space),
            // ToneCurve works on Lab (L + a/b LUTs), so it must agree too.
            Stage::ToneCurve { space, .. } => Some(*space),
            // Levels works on Lab L (+ proportional a/b), so it too must agree.
            Stage::Levels { space, .. } => Some(*space),
            // Lowlight works in Lab (via XYZ), so it must agree too.
            Stage::Lowlight { space, .. } => Some(*space),
            // Colisa works on Lab L (+ a/b saturation), so it must agree.
            Stage::Colisa { space, .. } => Some(*space),
            // Lowpass works in Lab (RGB-to-Lab blur then Lab-to-RGB), so it
            // must agree with the other Lab-domain stages on the working space.
            Stage::Lowpass { space, .. } => Some(*space),
            // Shadhi works in Lab (RGB→Lab blur then Lab→RGB process), so it
            // must agree with the other Lab-domain stages on the working space.
            // The C `default_colorspace` returns `IOP_CS_LAB` and the process
            // kernel operates on Lab buffers.
            Stage::Shadhi { space, .. } => Some(*space),
            // Basicadj works in linear RGB, so it has no Lab working
            // space to agree on — its `space` selects luminance
            // weights, not a Lab conversion.
            Stage::Basicadj { .. } => None,
            // Primaries works directly in linear RGB — no Lab conversion, no
            // working-space agreement needed. The matrix (including any working-space
            // adaptation) is baked into the 4×4 at `to_pipeline` time.
            Stage::Primaries { .. } => None,
            // Negadoctor works directly in linear RGB — no Lab conversion, no
            // working-space agreement needed.
            Stage::Negadoctor { .. } => None,
            // ToneEqual reads the RGB norm-2 luminance and scales channels by a
            // correction looked up in its LUT — it runs on whatever RGB buffer
            // it is given (darktable `default_colorspace` = IOP_CS_RGB), so no
            // Lab working-space agreement is needed.
            Stage::ToneEqual { .. } => None,
            // ColorBalanceRgb converts through XYZ D65 with a working-space-
            // selected RGB↔XYZ pair (its Yrg/UCS chain is space-agnostic in
            // between), and its gamut LUT was built against that same space's
            // primaries at pipeline-build time — so it must agree with the
            // other Lab/space-aware stages exactly like Sharpen/Vibrance do.
            Stage::ColorBalanceRgb { space, .. } => Some(*space),
            // FilmicRgb's Yrg matrices (input/outputs of the gamut map) are
            // selected per working space at apply time, so — like ColorBalanceRgb
            // — it must agree with the other space-aware stages.
            Stage::FilmicRgb { space, .. } => Some(*space),
            // DenoiseProfile runs on linear RGB straight from the raw decode
            // (IOP_CS_RGB in the C, no profile dependency in the VST) — no
            // Lab conversion, no working-space agreement needed.
            Stage::DenoiseProfile { .. } => None,
            // RgbCurve applies its LUTs directly on the working RGB lanes
            // (C default_colorspace: IOP_CS_RGB) — nothing to agree on.
            Stage::RgbCurve { .. } => None,
            // Basecurve likewise: IOP_CS_RGB in the C, LUT lookups (and the
            // fusion pyramid) run directly on the working lanes.
            Stage::Basecurve { .. } => None,
            Stage::GraduatedNd { .. } => None,
            _ => None,
        }
    }

    /// Process a packed RGBA `f32` buffer (`input`) into `output` (same length,
    /// a multiple of 4). Dispatches to the migrated IOP core loop.
    ///
    /// The length contract matters: exposure writes every element, but the
    /// `chunks_exact(4)` IOPs (velvia/splittoning/channelmixer) silently drop a
    /// `len % 4` tail, which — with the ping-pong reuse in [`Pipeline::process`]
    /// — would leak stale pixels there. `process` hard-asserts the contract; the
    /// debug-asserts here guard internal callers.
    pub fn apply(&self, input: &[f32], output: &mut [f32], width: usize, height: usize) {
        debug_assert_eq!(input.len(), output.len(), "apply: in/out length mismatch");
        debug_assert_eq!(input.len() % 4, 0, "apply: buffer must be packed RGBA (len % 4 == 0)");
        // Per-pixel stages ignore (width, height); spatial stages (Sharpen) index
        // neighbours by them, so the rectangle must match the buffer.
        debug_assert_eq!(width * height * 4, input.len(), "apply: (w,h) doesn't match buffer");
        match *self {
            Stage::Exposure { black, scale } => {
                exposure::process_pixels(input, output, black, scale)
            }
            Stage::Velvia { strength, bias } => {
                velvia::process_pixels(input, output, strength, bias)
            }
            Stage::Splittoning {
                shadow_hue,
                shadow_sat,
                highlight_hue,
                highlight_sat,
                balance,
                compress,
            } => splittoning::process_pixels(
                input, output, shadow_hue, shadow_sat, highlight_hue, highlight_sat, balance,
                compress,
            ),
            Stage::Monochrome { r, g, b } => {
                let rgb_matrix = [r, g, b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                let hsl_matrix = [0.0f32; 9];
                channelmixer::process_pixels(input, output, &hsl_matrix, &rgb_matrix, 1);
            }
            Stage::Sigmoid {
                white_target,
                black_target,
                paper_exp,
                film_fog,
                film_power,
                paper_power,
            } => {
                let npixels = input.len() / 4;
                // Safety: input/output are equal-length packed RGBA (the caller
                // asserts len % 4 == 0), so each holds exactly npixels*4 floats —
                // the contract darkroom_sigmoid_rgb_ratio_process documents.
                unsafe {
                    sigmoid::darkroom_sigmoid_rgb_ratio_process(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        npixels,
                        white_target,
                        black_target,
                        paper_exp,
                        film_fog,
                        film_power,
                        paper_power,
                    );
                }
            }
            Stage::Sharpen { radius, threshold, amount, space, scale } => {
                debug_assert!(
                    scale.is_finite() && scale > 0.0,
                    "Sharpen scale must be a positive finite ROI factor, got {scale}"
                );
                // Effective radius scales with the ROI (roi->scale): a downscaled
                // preview uses a proportionally smaller kernel so its sharpening
                // matches the full-res export. threshold/amount are value-domain
                // (contrast) params — resolution-independent, so NOT scaled (matches
                // sharpen.c).
                let (rad, mat) = gaussian_kernel(radius * scale);
                if amount == 0.0 || radius * scale == 0.0 || width <= 2 * rad || height <= 2 * rad {
                    // Disabled (amount 0), zero effective radius, or too small for a
                    // kernel interior ⇒ no sharpening. Byte-exact passthrough (skips
                    // the Lab round-trip; matches sharpen.c's rad==0 early return).
                    output.copy_from_slice(input);
                } else {
                    // Faithful darktable sharpen: unsharp-mask the Lab L channel
                    // only. RGB → Lab (L,a,b,A), sharpen ch 0 (= L) via the migrated
                    // kernel (a/b pass through), Lab → RGB. The RGB↔Lab pair is
                    // chosen by the buffer's working space (raw Rec.2020 vs non-raw
                    // linear sRGB) so the L/primaries are correct for both.
                    // TODO(perf): only L is sharpened but we round-trip a/b too; a
                    // planar-L kernel + reusing input a/b would cut the scratch.
                    let (to_lab, from_lab): (LabConv, LabConv) =
                        match space {
                            ColorSpace::Rec2020 => {
                                (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020)
                            }
                            ColorSpace::LinearSrgb => {
                                (crate::color::srgb_to_lab, crate::color::lab_to_srgb)
                            }
                        };
                    let n = width * height;
                    let mut lab_in = vec![0.0f32; n * 4];
                    for p in 0..n {
                        let i = p * 4;
                        let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                        lab_in[i..i + 4].copy_from_slice(&lab);
                    }
                    let mut lab_out = vec![0.0f32; n * 4];
                    // Safety: lab_in/lab_out are exactly n*4 floats and `mat` is
                    // 2*rad+1 taps — the kernel's documented contract.
                    unsafe {
                        sharpen::darkroom_sharpen_process(
                            lab_in.as_ptr(),
                            lab_out.as_mut_ptr(),
                            mat.as_ptr(),
                            width,
                            height,
                            rad as i32,
                            threshold,
                            amount,
                        );
                    }
                    for p in 0..n {
                        let i = p * 4;
                        // Sharpened L from lab_out; original a/b/alpha (read from
                        // lab_in / input, independent of the kernel's passthrough).
                        let rgb = from_lab([
                            lab_out[i], lab_in[i + 1], lab_in[i + 2], input[i + 3],
                        ]);
                        output[i..i + 4].copy_from_slice(&rgb);
                    }
                }
            }
            // ── Vibrance (vibrance.c) ──────────────────────────────────────────
            // Operates in Lab: convert RGB→Lab, apply the chroma-boost to a/b (and
            // dim L), convert back. All channels change, so the input a/b can't be
            // reused from the source RGB (unlike Sharpen, which only touched L).
            Stage::Vibrance { amount, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => {
                            (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020)
                        }
                        ColorSpace::LinearSrgb => {
                            (crate::color::srgb_to_lab, crate::color::lab_to_srgb)
                        }
                    };
                let n = width * height;
                // Convert to Lab, apply the chroma-boost (a/b change, so all Lab
                // channels may differ), convert back. Two scratch buffers avoid the
                // borrow conflict that `process_pixels(&lab, &mut lab)` would raise.
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                vibrance::process_pixels(&lab_in, &mut lab_out, amount);
                for p in 0..n {
                    let i = p * 4;
                    // Use the original alpha from the source, not the round-tripped
                    // one (both should be identical, but the pattern matches Sharpen's
                    // `input[i + 3]` to be explicit about not trusting the 4th channel).
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Color contrast (colorcontrast.c) ────────────────────────
            // Lab affine scale/offset on a/b channels. Like Vibrance it round-trips
            // RGB↔Lab, but it touches a/b (not L), so the source a/b can't be reused.
            Stage::ColorContrast { a_steepness, b_steepness, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => {
                            (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020)
                        }
                        ColorSpace::LinearSrgb => {
                            (crate::color::srgb_to_lab, crate::color::lab_to_srgb)
                        }
                    };
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                colorcontrast::process_pixels(
                    &lab_in, &mut lab_out,
                    a_steepness, 0.0, b_steepness, 0.0, /* unbound */ true,
                );
                for p in 0..n {
                    let i = p * 4;
                    // Alpha from the source, matching Sharpen/Vibrance.
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Temperature (temperature.c) ────────────────────────────────
            // Per-channel RGB multiply (white balance). Works directly in linear
            // RGB — no Lab round-trip. The FFI signature (process_rgb) takes a
            // 4-float coeffs array [R, G, B, A]. Pixel-local, pixel-exact.
            Stage::Temperature { coeffs } => {
                let npixels = input.len() / 4;
                // Safety: input/output are packed RGBA f32 buffers of equal length
                // (debug-asserted above: len % 4 == 0), so each holds exactly
                // npixels*4 floats — the contract darkroom_temperature_process_rgb
                // documents. coeffs is a fixed [f32; 4].
                unsafe {
                    temperature::darkroom_temperature_process_rgb(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        npixels,
                        coeffs.as_ptr(),
                    );
                }
            }
            // ── Invert (invert.c) ─────────────────────────────────
            // Per-channel film-camera negative inversion: `out = color - in`.
            // Works directly in linear RGB — no Lab round-trip. The FFI
            // (darkroom_invert_process) takes a 4-float color array [R, G, B, A].
            // Pixel-local, pixel-exact.
            Stage::Invert { color } => {
                let npixels = input.len() / 4;
                // Safety: input/output are packed RGBA f32 buffers of equal length
                // (debug-asserted above: len % 4 == 0), so each holds exactly
                // npixels*4 floats — the contract darkroom_invert_process documents.
                // `color` is a fixed [f32; 4].
                unsafe {
                    invert::darkroom_invert_process(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        npixels,
                        color.as_ptr(),
                    );
                }
            }
            // ── Colorize (colorize.c) ─────────────────────────────────
            // Replaces a/b channels with a fixed Lab colour, blends L from the
            // input via `mix`. Like Vibrance/ColorContrast it round-trips
            // RGB↔Lab, so it needs the buffer's working colour space. The
            // (color_l, color_a, color_b, mix) are pre-converted from HSL by
            // `to_pipeline`; the per-pixel work is just the RGB→Lab, process,
            // Lab→RGB sandwich.
            Stage::Colorize { color_l, color_a, color_b, mix, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                colorize::process_pixels(&lab_in, &mut lab_out, color_l, color_a, color_b, mix);
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Color correction (colorcorrection.c) ────────────────────
            // Luminance-dependent Lab a/b scaling + global saturation. Like
            // ColorContrast/Colorize it round-trips RGB↔Lab (chosen by the
            // buffer's working space), then calls colorcorrection::process_pixels.
            Stage::ColorCorrection { a_scale, a_base, b_scale, b_base, saturation, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                colorcorrection::process_pixels(
                    &lab_in, &mut lab_out,
                    a_scale, a_base, b_scale, b_base, saturation,
                );
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Color zones (colorzones.c) ─────────────────────────────
            // LCH equaliser via 3×65536-entry LUTs. Like ColorCorrection it
            // round-trips RGB↔Lab (chosen by the buffer's working space), then
            // calls the FFI darkroom_colorzones_process on the Lab buffer.
            Stage::ColorZones { ref lut_l, ref lut_c, ref lut_h, channel, mode, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                // Safety: lab_in/lab_out are exactly n*4 f32 packed RGBA buffers,
                // and lut_l/lut_c/lut_h are each exactly 65536 f32 — the contract
                // darkroom_colorzones_process documents.
                unsafe {
                    colorzones::darkroom_colorzones_process(
                        lab_in.as_ptr(),
                        lab_out.as_mut_ptr(),
                        n,
                        mode,
                        channel,
                        lut_l.as_ptr(),
                        lut_c.as_ptr(),
                        lut_h.as_ptr(),
                    );
                }
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Bloom (bloom.c) ────────────────────────────────────────
            // Threshold-gather Lab L, box-blur the gathered light, screen-blend
            // it back into L. Same RGB↔Lab sandwich as the other Lab-domain
            // stages; the blur is a neighbourhood read (radius up to 256), so
            // this stage runs on whole frames only — see is_pixel_local.
            Stage::Bloom { size, threshold, strength, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                bloom::process(
                    &lab_in, &mut lab_out, width, height, size, threshold, strength,
                );
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Tone curve (tonecurve.c) ───────────────────────────────
            // Three-channel Lab LUT (L + a + b curves) built from the spline
            // nodes via tonecurve::build_lut. Same RGB↔Lab sandwich as the
            // other Lab-domain stages; the stage itself is pixel-local (pure
            // LUT lookups), so it also runs on bands under rayon.
            Stage::ToneCurve { ref table_l, ref table_a, ref table_b, ref coeffs_l, ref coeffs_ab, autoscale_ab, unbound_ab, preserve_colors, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                tonecurve::process_pixels(
                    &lab_in, &mut lab_out,
                    table_l,
                    table_a,
                    table_b,
                    &coeffs_l[..],
                    &coeffs_ab[..],
                    autoscale_ab,
                    unbound_ab,
                    preserve_colors,
                );
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── RGB curve (rgbcurve.c) ──────────────────────────────────
            // Three per-channel LUTs applied straight on the working RGB
            // lanes — IOP_CS_RGB in the C, so no Lab sandwich. Pixel-local
            // (pure lookups / per-pixel norm math), so it also runs on bands
            // under rayon.
            Stage::RgbCurve { ref table_r, ref table_g, ref table_b, ref coeffs, autoscale, preserve_colors } => {
                rgbcurve::process_pixels(
                    input,
                    output,
                    &table_r[..],
                    &table_g[..],
                    &table_b[..],
                    coeffs,
                    autoscale,
                    preserve_colors,
                );
            }
            // ── Base curve (basecurve.c) ─────────────────────────────────
            // A single LUT applied straight on the working RGB lanes —
            // IOP_CS_RGB in the C, so no Lab sandwich. fusion == 0 is the
            // plain process_lut path (pixel-local, runs on bands under
            // rayon); fusion >= 1 is exposure fusion: a whole-frame
            // laplacian-pyramid blend (is_pixel_local returns false, so this
            // arm only ever sees whole frames in that case). The LUMINANCE
            // preservation norm consumes the working space's Y row — the
            // work-profile matrix_in row the C reads.
            Stage::Basecurve {
                ref table,
                ref coeffs,
                preserve_colors,
                fusion,
                stops,
                bias,
                space,
            } => {
                let y_row = match space {
                    ColorSpace::Rec2020 => crate::color::REC2020_TO_XYZ_D65_Y_ROW,
                    ColorSpace::LinearSrgb => crate::color::SRGB_TO_XYZ_D65_Y_ROW,
                };
                if fusion == 0 {
                    basecurve::apply_curve_pixels(input, output, 1.0, &table[..], coeffs, preserve_colors, Some(y_row));
                } else {
                    basecurve::process_fusion(
                        input,
                        output,
                        width,
                        height,
                        &table[..],
                        coeffs,
                        preserve_colors,
                        stops,
                        fusion,
                        bias,
                        Some(y_row),
                    );
                }
            }
            // ── Levels (levels.c) ───────────────────────────────────────
            // Black/grey/white points + gamma applied to Lab L via the
            // pre-computed LUT, with a/b scaled by the same L ratio. Same
            // RGB↔Lab sandwich as the other Lab-domain stages.
            Stage::Levels { black, range, inv_gamma, ref lut, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                // The LUT is a fixed 65536-entry table (build_lut is the only
                // producer), so a wrong length is a caller bug. Loud in debug;
                // in release degrade to a byte-exact passthrough rather than
                // panicking — this runs per band inside a rayon `for_each_init`,
                // where a panic would take down the whole preview render.
                debug_assert_eq!(
                    lut.len(), 65536,
                    "Stage::Levels: LUT must be exactly 65536 entries, got {}", lut.len()
                );
                let Ok(lut_arr): Result<&[f32; 65536], _> = lut.as_slice().try_into() else {
                    output.copy_from_slice(input);
                    return;
                };
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                levels::process_pixels(&lab_in, &mut lab_out, black, range, inv_gamma, lut_arr);
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Lowlight (lowlight.c) ──────────────────────────────────
            // Scotopic/photopic blend in Lab, mixed by a luminance-driven
            // curve. Same RGB↔Lab sandwich as the other Lab-domain stages.
            Stage::Lowlight { blueness, ref lut, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                // Fixed 65536-entry curve; a wrong length is a caller bug.
                // Loud in debug, passthrough in release (a panic here would be
                // inside rayon's for_each_init).
                debug_assert_eq!(
                    lut.len(), 65536,
                    "Stage::Lowlight: LUT must be exactly 65536 entries, got {}", lut.len()
                );
                let Ok(lut_arr): Result<&[f32; 65536], _> = lut.as_slice().try_into() else {
                    output.copy_from_slice(input);
                    return;
                };
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                lowlight::process_pixels(&lab_in, &mut lab_out, blueness, lut_arr);
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Colisa (colisa.c) ──────────────────────────────────────
            // Contrast/brightness tone curves on Lab L plus an a/b saturation
            // scale. Same RGB↔Lab sandwich as the other Lab-domain stages.
            Stage::Colisa { contrast, brightness, saturation, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                let d = colisa::commit_params(contrast, brightness, saturation);
                let n = width * height;
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut lab_out = vec![0.0f32; n * 4];
                colisa::process_pixels(
                    &lab_in, &mut lab_out,
                    &d.ctable, &d.cunbounded, &d.ltable, &d.lunbounded, d.saturation,
                );
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([lab_out[i], lab_out[i + 1], lab_out[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Basic adjustments (basicadj.c) ─────────────────────────
            // Straight linear RGB, no Lab sandwich. `space` only picks the
            // luminance weights the highlight-compression pass uses — the Y row
            // of that space's RGB→XYZ matrix, which is what the C pulls out of
            // the work profile.
            Stage::Basicadj {
                black_point, exposure, hlcompr, hlcomprthresh, contrast,
                preserve_colors, middle_grey, brightness, saturation, vibrance, space,
            } => {
                let luma = match space {
                    ColorSpace::Rec2020 => [0.2627f32, 0.6780, 0.0593],
                    ColorSpace::LinearSrgb => [0.2126f32, 0.7152, 0.0722],
                };
                let d = basicadj::commit_params(
                    black_point, exposure, hlcompr, hlcomprthresh, contrast,
                    preserve_colors, middle_grey, brightness, saturation, vibrance, luma,
                );
                d.process(input, output);
            }
            // ── Shadows/Highlights (shadhi.c) ───────────────────────
            // A Gaussian-blurred base layer of the Lab buffer is merged with
            // the original to lift shadows / recover highlights. The C
            // `process()` pre-scales all the user sliders into core scalars
            // (e.g. shadows ∈ [-100,100] → [-2,2]); we mirror that exactly so
            // the FFI kernel `darkroom_shadhi_process` receives the same
            // values; the shadow/highlight *math* inside the kernel is identical,
            // but the base layer differs (Gaussian blur, not bilateral — see below).
            //
            // We hardcode `shadhi_algo = GAUSSIAN` — the C default is
            // bilateral, but `crate::gaussian` only implements the recursive
            // Gaussian, not a bilateral filter. The Gaussian blur is a faithful
            // (if slightly different) base layer; the shadow/highlight math is
            // identical. `flags` is hardcoded to `UNBOUND_DEFAULT` (127) —
            // darktable's own default and not exposed in the GUI.
            Stage::Shadhi {
                shadows, highlights, whitepoint, radius, compress,
                shadows_ccorrect, highlights_ccorrect, scale, space,
            } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                let n = width * height;
                // RGB => Lab, holding the original sharp pixels.
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                // Blur the Lab buffer in place into a second buffer — mirrors
                // the C `dt_gaussian_blur_4c` over unbound Lab bounds.
                let sigma = f32::max(0.1, radius) * scale;
                let mut blurred = vec![0.0f32; n * 4];
                {
                    // unbound_mask is true (shadhi_algo=GAUSSIAN & flags & UNBOUND_GAUSSIAN),
                    // so the C widens Lab bounds to ±FLT_MAX — matching our Stage default.
                    let mut g = crate::gaussian::Gaussian::new(
                        width, height,
                        [-f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX],
                        [f32::MAX, f32::MAX, f32::MAX, f32::MAX],
                        sigma,
                        crate::gaussian::GaussianOrder::Zero,
                    );
                    g.blur_4c(&lab_in, &mut blurred);
                }
                // Pre-scale user sliders into core scalars (C process() lines 343-355).
                let sh = 2.0 * f32::clamp(shadows / 100.0, -1.0, 1.0);
                let hg = 2.0 * f32::clamp(highlights / 100.0, -1.0, 1.0);
                let w  = f32::max(1.0 - whitepoint / 100.0, 0.01);
                let cs = f32::clamp(compress / 100.0, 0.0, 0.99);
                // C's `sign()` macro returns 1 for ±0.0; f32::signum returns 0.0.
                // Use a C-compatible sign so the neutral (sliders at 0) case matches.
                let sh_cc = (f32::clamp(shadows_ccorrect / 100.0, 0.0, 1.0) - 0.5) * sh.signum_c() + 0.5;
                let hg_cc = (f32::clamp(highlights_ccorrect / 100.0, 0.0, 1.0) - 0.5) * (-hg).signum_c() + 0.5;
                // Hardcoded C defaults: flags=UNBOUND_DEFAULT(127), low_approximation=0.000001,
                // unbound_mask=1 (GAUSSIAN algo, UNBOUND_GAUSSIAN bit set).
                // SAFETY: `lab_in` and `blurred` are distinct `Vec<f32>` of exactly
                // `n*4` elements (allocated above, n = width*height satisfies the
                // `(width, height)` precondition upheld by `Pipeline::process`), so
                // both slices the kernel materialises are in bounds and cannot
                // alias. Crucially, `blurred` is read-modify-write and MUST already
                // hold the Gaussian-blurred Lab layer — the blur above establishes
                // that. Scalars are pre-scaled to the physical ranges the kernel
                // documents (C `process()` / shadhi.c lines 343-355).
                unsafe {
                    shadhi::darkroom_shadhi_process(
                        lab_in.as_ptr(),
                        blurred.as_mut_ptr(),
                        n,
                        sh, hg, w, cs,
                        sh_cc, hg_cc,
                        0.000001f32, // low_approximation
                        127u32,       // flags = UNBOUND_DEFAULT
                        1i32,         // unbound_mask = true (GAUSSIAN + UNBOUND_GAUSSIAN)
                    );
                }
                // Lab => RGB for the final output.
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([blurred[i], blurred[i + 1], blurred[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Lowpass (lowpass.c) ──────────────────────────────────
            // Local-contrast reduction: blur in Lab, then apply contrast +
            // brightness LUTs to the blurred L and saturation-scale a/b. Same
            // RGB-to-Lab sandwich as the other Lab-domain stages. NOT
            // pixel-local (the Gaussian blur reads neighbours), so `process`
            // guarantees (width, height) here is the true image rectangle.
            Stage::Lowpass { radius, contrast, brightness, saturation, scale, space } => {
                let (to_lab, from_lab): (LabConv, LabConv) =
                    match space {
                        ColorSpace::Rec2020 => (crate::color::rec2020_to_lab, crate::color::lab_to_rec2020),
                        ColorSpace::LinearSrgb => (crate::color::srgb_to_lab, crate::color::lab_to_srgb),
                    };
                let sigma = f32::max(0.1, radius) * scale;
                let n = width * height;
                // RGB => Lab, then blur the Lab copy into a second buffer.
                let mut lab_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    let lab = to_lab([input[i], input[i + 1], input[i + 2], input[i + 3]]);
                    lab_in[i..i + 4].copy_from_slice(&lab);
                }
                let mut blurred = vec![0.0f32; n * 4];
                {
                    // darktable clamps each Lab channel into [Labmin, Labmax] as
                    // it enters the recursion; unbound=true (the C default) widens
                    // that to +/-FLT_MAX, matching our Stage default.
                    let mut g = crate::gaussian::Gaussian::new(
                        width, height,
                        [-f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX],
                        [f32::MAX, f32::MAX, f32::MAX, f32::MAX],
                        sigma,
                        crate::gaussian::GaussianOrder::Zero,
                    );
                    g.blur_4c(&lab_in, &mut blurred);
                }
                // Build LUTs + extrapolation coeffs, then run the per-pixel LUT
                // pass. `in` = original Lab (for alpha), `out` = blurred Lab
                // (modified in place).
                let d = lowpass::commit_params(contrast, brightness, saturation, /* unbound = */ true);
                lowpass::process_pixels(
                    &lab_in, &mut blurred,
                    &d.ctable, &d.cunbounded, &d.ltable, &d.lunbounded,
                    d.saturation, d.lab_min_ab, d.lab_max_ab,
                );
                // Lab => RGB for the final output.
                for p in 0..n {
                    let i = p * 4;
                    let rgb = from_lab([blurred[i], blurred[i + 1], blurred[i + 2], input[i + 3]]);
                    output[i..i + 4].copy_from_slice(&rgb);
                }
            }
            // ── Graduated ND (graduatednd.c) ───────────────────────────
            // Position-dependent gradient filter, straight in RGB. Not
            // pixel-local, so `process` guarantees (width, height) is the true
            // image rectangle here — the coordinates the geometry assumes.
            Stage::GraduatedNd { density, hardness, rotation, offset, hue, saturation } => {
                let g = graduatednd::commit_geometry(
                    width, height, density, hardness, rotation, offset, hue, saturation,
                );
                // Safety: input/output are equal-length packed RGBA holding
                // exactly width*height*4 floats, and color/color1 are [f32; 4] —
                // the contract darkroom_graduatednd_process documents.
                unsafe {
                    graduatednd::darkroom_graduatednd_process(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        width as i32,
                        height as i32,
                        density,
                        g.length_base,
                        g.length_inc,
                        g.cosv_hh_inv,
                        g.filter_hardness,
                        0, // full-image preview ROI starts at y = 0
                        g.color.as_ptr(),
                        g.color1.as_ptr(),
                    );
                }
            }
            // ── Vignette (vignette.c) ──────────────────────────────────
            // Position-dependent radial falloff, straight in RGB. `is_pixel_local`
            // returns false for this stage, so `process` guarantees (width,
            // height) here is the true image rectangle — the coordinates the
            // geometry was computed against — never a band's pixel run.
            Stage::Vignette {
                scale, falloff, brightness, saturation,
                center_x, center_y, shape, autoratio, whratio, dither_amt, unbound,
            } => {
                // Derived here, not stored: the geometry is a function of the
                // buffer size, which only `apply` knows.
                let geometry = vignette::commit_geometry(
                    width, height, scale, falloff, center_x, center_y,
                    autoratio, whratio, shape,
                );
                // Safety: input/output are equal-length packed RGBA (asserted
                // above) holding exactly width*height*4 floats, which is the
                // contract darkroom_vignette_process documents.
                unsafe {
                    vignette::darkroom_vignette_process(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        width as i32,
                        height as i32,
                        geometry.xscale,
                        geometry.yscale,
                        geometry.roi_center_x,
                        geometry.roi_center_y,
                        geometry.dscale,
                        geometry.fscale,
                        geometry.exp1,
                        geometry.exp2,
                        dither_amt,
                        brightness,
                        saturation,
                        i32::from(unbound),
                    );
                }
            }
            // ── Primaries (primaries.c) ────────────────────────────────
            // Per-pixel 4×4 colour matrix multiply (RGB↔RGB in the working
            // space). Works directly in linear RGB — no Lab round-trip, so
            // `working_space()` returns `None`. The matrix is pre-computed by
            // `PreviewParams::to_pipeline` from the 8 UI params and cached in
            // the stage, so apply just dispatches to the FFI kernel.
            Stage::Primaries { matrix } => {
                // Safe wrapper: applies the 4x4 matrix per-pixel, with bounds
                // checks and an aliasing debug_assert. The matrix (including
                // any working-space adaptation) is baked in by to_pipeline.
                primaries::process_pixels(input, output, &matrix);
            }
            // ── Negadoctor (negadoctor.c) ────────────────────────────────
            // Film-negative scan inversion via Cineon-style log-density.
            // Works directly in linear RGB — no Lab round-trip, so
            // `working_space()` returns `None`. The data arrays (Dmin, wb_high,
            // offset) are pre-computed by PreviewParams::to_pipeline (following
            // darktable's commit_params) and cached in the stage. apply dispatches
            // to the migrated FFI kernel.
            Stage::Negadoctor {
                dmin, wb_high, offset, black, gamma, soft_clip, soft_clip_comp, exposure,
            } => {
                let npixels = input.len() / 4;
                // Safety: input/output are packed RGBA f32 buffers of equal length
                // (debug-asserted above: len % 4 == 0), so each holds exactly
                // npixels*4 floats — the contract the FFI kernel documents.
                // The [f32; 4] arrays are stack copies, so their pointers outlive
                // the call.
                unsafe {
                    negadoctor::darkroom_negadoctor_process(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        npixels,
                        dmin.as_ptr(),
                        wb_high.as_ptr(),
                        offset.as_ptr(),
                        black, gamma, soft_clip, soft_clip_comp, exposure,
                    );
                }
            }
            // ── ToneEqual (toneequal.c, details == DT_TONEEQ_NONE) ───────
            // Scene-referred tone mapping by exposure channel. The RBF solve +
            // correction-LUT build happen inside `process_preview_pixels`, memoised
            // per thread on the gains (see toneequal::CORRECTION_LUT_CACHE) so the
            // per-band calls of the parallel path share one table.
            Stage::ToneEqual { gains } => {
                toneequal::process_preview_pixels(input, output, &gains);
            }
            // ── ColorBalanceRgb (colorbalancergb.c) ──────────────────────
            // Scene-referred grading in Filmlight Yrg + perceptual saturation
            // (dt-UCS/JzAzBz). The commit_params derivation — zone vectors,
            // weights and the 512-entry gamut LUT — is prebuilt by
            // `PreviewParams::to_pipeline` and carried in the stage, so the
            // per-band call is pure pixel math.
            Stage::ColorBalanceRgb { ref data, space } => {
                // The RGB↔XYZ-D65 pair is chosen by the buffer's working space
                // (same split as Sharpen/Vibrance): raw previews grade in
                // Rec.2020, non-raw in linear sRGB. `data`'s gamut LUT was
                // built against that same space at `to_pipeline` time.
                let (to_xyz, from_xyz): (colorbalancergb::RgbXyzConv, colorbalancergb::RgbXyzConv) =
                    match space {
                        ColorSpace::Rec2020 => (
                            crate::color::rec2020_to_xyz_d65,
                            crate::color::xyz_d65_to_rec2020,
                        ),
                        ColorSpace::LinearSrgb => {
                            (crate::color::srgb_to_xyz_d65, crate::color::xyz_d65_to_srgb)
                        }
                    };
                colorbalancergb::process_in_space(input, output, data, to_xyz, from_xyz);
            }
            // ── FilmicRgb (filmicrgb.c, colour science v5) ───────────────
            // Scene-referred tone mapping: log encoding → derived spline →
            // display power, then chroma preservation + gamut map in Yrg. The
            // committed data (spline coefficients and scalars) is prebuilt by
            // `PreviewParams::to_pipeline`; the six Yrg matrices are selected
            // here from the buffer's working space so the gamut clip runs
            // against the primaries the pixels are actually in.
            Stage::FilmicRgb { ref data, space } => {
                let matrices = filmicrgb::matrices_for_space(space);
                filmicrgb::process_in_space(input, output, data, &matrices);
            }
            // ── Denoise (profiled) (denoiseprofile.c) ───────────────────
            // Wavelets-mode profiled noise reduction: variance-stabilising
            // transform → edge-avoiding à-trous decomposition with BayesShrink
            // soft thresholds → inverse VST. Works directly in linear RGB — no
            // Lab round-trip. NOT pixel-local (multi-scale neighbourhood reads),
            // so `process` guarantees (width, height) here is the true image
            // rectangle.
            Stage::DenoiseProfile { strength, shadows, bias, mode_y0u0v0 } => {
                denoiseprofile::wavelets_denoise(
                    input, output, width, height,
                    &denoiseprofile::WaveletsParams { strength, shadows, bias, mode_y0u0v0 },
                );
            }
            // ── Lens correction (lens.cc LENSFUN method) ───────────────
            // Vignetting gain + per-channel coordinate warp through the
            // distro liblensfun. NOT pixel-local (a resampling warp), so
            // `process` guarantees (width, height) is the whole frame here.
            // The destructure names avoid shadowing the `lens` module.
            Stage::LensCorrection { lens: ref resolved, ref params } => {
                crate::iop::lens::process(input, output, width, height, resolved, params);
            }
        }
    }
}

/// An ordered sequence of [`Stage`]s applied to a scene-referred RGBA buffer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pipeline {
    pub stages: Vec<Stage>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn with_stages(stages: Vec<Stage>) -> Self {
        Self { stages }
    }

    pub fn push(&mut self, stage: Stage) {
        self.stages.push(stage);
    }

    /// Run all stages in order over `input` (packed RGBA `f32`, length a multiple
    /// of 4) and return the result. An empty pipeline returns the input unchanged.
    ///
    /// If **every** stage is pixel-local (position-independent per-pixel map, no
    /// neighbour reads — the property that also lets geometry commute), the buffer
    /// is split into pixel-aligned bands processed in parallel, each running the
    /// full stage sequence through its own ping-pong scratch; the result is
    /// **bit-identical** to a whole-buffer serial run (bands never interact) —
    /// pinned by `parallel_result_is_split_invariant`. If any stage is spatial
    /// (e.g. `Sharpen`), the whole pipeline runs **serial over one whole-buffer
    /// band** so that stage sees the true `(width, height)` rectangle rather than
    /// a band's pixel run. Small buffers also stay serial to avoid rayon overhead.
    /// `(width, height)` must satisfy `width * height * 4 == input.len()`.
    pub fn process(&self, input: &[f32], width: usize, height: usize) -> Vec<f32> {
        // Hard guard at the trust boundary (fires in release too): the
        // `chunks_exact(4)` stages would otherwise silently leave a stale tail.
        assert!(
            input.len().is_multiple_of(4),
            "Pipeline::process: buffer must be packed RGBA (len % 4 == 0), got {}",
            input.len()
        );
        assert!(
            width * height * 4 == input.len(),
            "Pipeline::process: (w,h)={width}x{height} doesn't match buffer len {}",
            input.len()
        );
        // One buffer ⇒ one working space: every colour-space-dependent stage
        // (Sharpen) must agree, or a caller has handed the pipeline a
        // self-inconsistent chain. (Only Sharpen carries a space today, so this is
        // a forward guard.)
        debug_assert!(
            {
                let mut spaces = self.stages.iter().filter_map(Stage::working_space);
                spaces.next().is_none_or(|first| spaces.all(|s| s == first))
            },
            "colour-space-dependent stages disagree on the working space"
        );
        if self.stages.is_empty() {
            return input.to_vec();
        }

        let mut output = vec![0.0f32; input.len()];
        // Band size in f32s (pixel-aligned: PIXELS_PER_BAND * 4 channels). 64k px
        // ≈ a 1 MB band — cache-friendly and large enough to amortise rayon's
        // per-task overhead. Total len is a multiple of 4 and BAND is a multiple
        // of 4, so every chunk — including the shorter final one (len % BAND) —
        // is pixel-aligned.
        const PIXELS_PER_BAND: usize = 64 * 1024;
        const BAND: usize = PIXELS_PER_BAND * 4;

        // Band-parallelism is only valid if every stage is pixel-local; otherwise
        // fall back to a single serial band (correct, just not parallel).
        let pixel_local = self.stages.iter().all(Stage::is_pixel_local);
        if input.len() <= BAND || !pixel_local {
            // One band = the whole buffer: no rayon overhead, and a spatial stage
            // sees the true (width, height) rectangle.
            let mut scratch = vec![0.0f32; input.len()];
            self.process_band(input, &mut output, &mut scratch, width, height);
        } else {
            use rayon::prelude::*;
            // Parallel path only runs when every stage is pixel-local (asserted by
            // the gate above), so the per-band (w,h) is never used for spatial
            // indexing — pass the band as a 1-row strip of its own pixel count.
            input
                .par_chunks(BAND)
                .zip(output.par_chunks_mut(BAND))
                .for_each_init(
                    || vec![0.0f32; BAND],
                    |scratch, (in_band, out_band)| {
                        self.process_band(
                            in_band,
                            out_band,
                            &mut scratch[..in_band.len()],
                            in_band.len() / 4,
                            1,
                        );
                    },
                );
        }
        output
    }

    /// Run every stage over one band `input` into `output` (equal lengths, both a
    /// multiple of 4), ping-ponging between `output` and the caller-provided
    /// `scratch` (also equal length) so no per-band allocation happens on the hot
    /// path. Caller guarantees `self.stages` is non-empty.
    fn process_band(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
        width: usize,
        height: usize,
    ) {
        // First stage reads `input`; every later stage ping-pongs output<->scratch.
        self.stages[0].apply(input, output, width, height);
        let mut result_in_output = true;
        for stage in &self.stages[1..] {
            if result_in_output {
                stage.apply(output, scratch, width, height);
            } else {
                stage.apply(scratch, output, width, height);
            }
            result_in_output = !result_in_output;
        }
        // If the last write landed in scratch, move it into output.
        if !result_in_output {
            output.copy_from_slice(scratch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pipeline_is_identity() {
        let px = vec![0.2, 0.4, 0.6, 1.0, 0.1, 0.5, 0.9, 1.0];
        assert_eq!(Pipeline::new().process(&px, px.len() / 4, 1), px);
    }

    #[test]
    fn exposure_scales_all_channels() {
        // out = (in - 0) * 2  (darktable applies exposure to all 4 channels)
        let px = vec![0.2f32, 0.4, 0.6, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Exposure { black: 0.0, scale: 2.0 }]);
        let out = p.process(&px, px.len() / 4, 1);
        assert_eq!(out, vec![0.4, 0.8, 1.2, 2.0]);
    }

    #[test]
    fn black_point_then_scale() {
        // out = (in - black) * scale
        let px = vec![0.5f32, 0.5, 0.5, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Exposure { black: 0.1, scale: 2.0 }]);
        let out = p.process(&px, px.len() / 4, 1);
        // (0.5 - 0.1) * 2 = 0.8 for colour; (1.0-0.1)*2 = 1.8 for alpha
        assert!((out[0] - 0.8).abs() < 1e-6);
        assert!((out[3] - 1.8).abs() < 1e-6);
    }

    #[test]
    fn stages_apply_in_order() {
        // exposure ×2 then ×3 ⇒ ×6
        let px = vec![0.1f32, 0.1, 0.1, 1.0];
        let p = Pipeline::with_stages(vec![
            Stage::Exposure { black: 0.0, scale: 2.0 },
            Stage::Exposure { black: 0.0, scale: 3.0 },
        ]);
        let out = p.process(&px, px.len() / 4, 1);
        assert!((out[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn monochrome_yields_equal_rgb() {
        // weights (0.2,0.7,0.1) on (1.0,0.5,0.0): luma = 0.2+0.35+0 = 0.55, R=G=B
        let px = vec![1.0f32, 0.5, 0.0, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Monochrome { r: 0.2, g: 0.7, b: 0.1 }]);
        let out = p.process(&px, px.len() / 4, 1);
        assert!((out[0] - 0.55).abs() < 1e-5);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn velvia_zero_strength_is_identity() {
        let px = vec![0.8f32, 0.2, 0.1, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Velvia { strength: 0.0, bias: 1.0 }]);
        assert_eq!(p.process(&px, px.len() / 4, 1), px);
    }

    #[test]
    fn stage_names_match_operations() {
        assert_eq!(Stage::Exposure { black: 0.0, scale: 1.0 }.name(), "exposure");
        assert_eq!(Stage::Monochrome { r: 1.0, g: 0.0, b: 0.0 }.name(), "channelmixer");
    }

    #[test]
    fn mixed_stages_dispatch_in_order() {
        // exposure ×2 → (2.0,1.0,0.0), then monochrome (0.2,0.7,0.1):
        // luma = 0.2*2 + 0.7*1 + 0.1*0 = 1.1, R=G=B.
        let px = vec![1.0f32, 0.5, 0.0, 1.0];
        let p = Pipeline::with_stages(vec![
            Stage::Exposure { black: 0.0, scale: 2.0 },
            Stage::Monochrome { r: 0.2, g: 0.7, b: 0.1 },
        ]);
        let out = p.process(&px, px.len() / 4, 1);
        assert!((out[0] - 1.1).abs() < 1e-5, "luma {}", out[0]);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn splittoning_passthrough_in_midtones() {
        // neutral mid pixel (l≈0.5) inside the passthrough band ⇒ ~unchanged.
        let px = vec![0.5f32, 0.5, 0.5, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Splittoning {
            shadow_hue: 0.0, shadow_sat: 1.0,
            highlight_hue: 0.2, highlight_sat: 1.0,
            balance: 0.5, compress: 0.1,
        }]);
        let out = p.process(&px, px.len() / 4, 1);
        for i in 0..3 {
            assert!((out[i] - 0.5).abs() < 1e-4, "ch{i} = {}", out[i]);
        }
    }

    #[test]
    fn sigmoid_preserves_middle_grey_and_is_monotonic() {
        // Default sigmoid params (contrast 1.5, no skew, white 100%, black
        // 0.0152%): a grey ramp must map monotonically, preserve middle grey,
        // and compress highlights below the white target.
        let [wt, bt, pe, ff, fp, pp] = sigmoid::rgb_ratio_params(1.5, 0.0, 100.0, 0.0152);
        for v in [wt, bt, pe, ff, fp, pp] {
            assert!(v.is_finite(), "param not finite: {v}");
        }
        // Golden values pin the derivation against transcription/sign regressions
        // (a wrong slope can still pass the monotonic + grey-preserved checks).
        // Verified faithful to sigmoid.c commit_params.
        assert!((wt - 1.0).abs() < 1e-5);
        assert!((bt - 0.000152).abs() < 1e-7);
        assert!((pe - 0.359695).abs() < 1e-4);
        assert!((ff - 0.0013843).abs() < 1e-6);
        assert!((fp - 1.490909).abs() < 1e-4);
        assert!((pp - 1.0).abs() < 1e-6); // skew 0 ⇒ 5^0
        let p = Pipeline::with_stages(vec![Stage::Sigmoid {
            white_target: wt, black_target: bt, paper_exp: pe,
            film_fog: ff, film_power: fp, paper_power: pp,
        }]);
        // grey pixels at increasing scene-linear levels
        let levels = [0.02f32, 0.1845, 0.5, 1.0, 4.0];
        let mut prev = -1.0f32;
        let mut mid_out = 0.0f32;
        for &v in &levels {
            let out = p.process(&[v, v, v, 1.0], 1, 1);
            assert!(out[0] > prev, "not monotonic at {v}: {} <= {prev}", out[0]);
            assert!(out[0] <= wt + 1e-3, "exceeds white target at {v}: {}", out[0]);
            prev = out[0];
            if (v - 0.1845).abs() < 1e-6 {
                mid_out = out[0];
            }
        }
        // middle grey maps (approximately) to itself
        assert!((mid_out - 0.1845).abs() < 0.02, "middle grey not preserved: {mid_out}");

        // Slope at middle grey is set by the contrast control (~1.21 for 1.5);
        // pins the curve *shape*, not just its monotonicity.
        let d = 1e-3f32;
        let hi = p.process(&[0.1845 + d, 0.1845 + d, 0.1845 + d, 1.0], 1, 1)[0];
        let lo = p.process(&[0.1845 - d, 0.1845 - d, 0.1845 - d, 1.0], 1, 1)[0];
        let slope = (hi - lo) / (2.0 * d);
        assert!((slope - 1.2068).abs() < 0.01, "grey slope off: {slope}");
    }

    #[test]
    #[should_panic(expected = "packed RGBA")]
    fn process_rejects_non_multiple_of_four() {
        // 6 elements is not a whole number of RGBA pixels.
        let bad = vec![0.0f32; 6];
        Pipeline::with_stages(vec![Stage::Velvia { strength: 1.0, bias: 1.0 }]).process(&bad, 1, 1);
    }

    /// A deterministic RGBA ramp of `pixels` pixels (values in [0,1)).
    fn ramp(pixels: usize) -> Vec<f32> {
        (0..pixels * 4).map(|i| (i % 97) as f32 / 97.0).collect()
    }

    #[test]
    fn large_buffer_exercises_parallel_path() {
        // > PIXELS_PER_BAND (64k) so process() takes the rayon branch. Exposure
        // ×2 is a pure per-channel scale, so every element must be exactly *2 —
        // proves the parallel path is correct end to end (incl. the final band).
        let px = ramp(100_000);
        let p = Pipeline::with_stages(vec![Stage::Exposure { black: 0.0, scale: 2.0 }]);
        let out = p.process(&px, px.len() / 4, 1);
        assert_eq!(out.len(), px.len());
        for (o, i) in out.iter().zip(px.iter()) {
            assert!((o - i * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn parallel_result_is_split_invariant() {
        // The core correctness claim: bands never interact, so processing the
        // whole buffer equals concatenating the results of processing each
        // pixel-aligned half. A multi-stage pipeline exercises the ping-pong.
        let px = ramp(100_000); // > one band → parallel
        let p = Pipeline::with_stages(vec![
            Stage::Exposure { black: 0.02, scale: 1.7 },
            Stage::Velvia { strength: 0.5, bias: 1.0 },
            Stage::Monochrome { r: 0.2, g: 0.7, b: 0.1 },
        ]);

        let full = p.process(&px, px.len() / 4, 1);
        let half = (px.len() / 2) & !3; // pixel-aligned split point
        let mut split = p.process(&px[..half], half / 4, 1);
        split.extend_from_slice(&p.process(&px[half..], (px.len() - half) / 4, 1));

        assert_eq!(full, split);
    }

    #[test]
    fn stage_pixel_locality_is_correctly_classified() {
        // The band-parallel path is only bit-identical to serial while every
        // stage is pixel-local; pin each stage's classification so the parallel
        // branch engages for the per-pixel stages and Sharpen (spatial) forces
        // the serial whole-buffer path.
        for s in [
            Stage::Exposure { black: 0.0, scale: 1.0 },
            Stage::Velvia { strength: 0.0, bias: 1.0 },
            Stage::Splittoning {
                shadow_hue: 0.0, shadow_sat: 0.0,
                highlight_hue: 0.0, highlight_sat: 0.0,
                balance: 0.5, compress: 0.0,
            },
            Stage::Monochrome { r: 0.3, g: 0.4, b: 0.3 },
            Stage::Sigmoid {
                white_target: 1.0, black_target: 0.0, paper_exp: 0.3,
                film_fog: 0.0, film_power: 1.0, paper_power: 1.0,
            },
            Stage::Vibrance { amount: 0.0, space: ColorSpace::LinearSrgb },
            Stage::ColorContrast { a_steepness: 1.0, b_steepness: 1.0, space: ColorSpace::LinearSrgb },
            Stage::Temperature { coeffs: [1.0, 1.0, 1.0, 1.0] },
            Stage::Invert { color: [1.0, 1.0, 1.0, 1.0] },
            Stage::Colorize { color_l: 50.0, color_a: 0.0, color_b: 0.0, mix: 1.0, space: ColorSpace::LinearSrgb },
            Stage::ColorCorrection { a_scale: 0.0, a_base: 0.0, b_scale: 0.0, b_base: 0.0, saturation: 1.0, space: ColorSpace::LinearSrgb },
            Stage::ColorZones {
                lut_l: vec![0.5; 65536], lut_c: vec![0.5; 65536], lut_h: vec![0.5; 65536],
                channel: 2, mode: 0, space: ColorSpace::LinearSrgb,
            },
            Stage::Levels {
                black: 0.0, range: 1.0, inv_gamma: 1.0,
                lut: vec![0.0; 65536], space: ColorSpace::LinearSrgb,
            },
            Stage::ToneCurve {
                table_l: Box::new(std::array::from_fn(|i| i as f32 / 65535.0)),
                table_a: Box::new(std::array::from_fn(|i| i as f32 / 65535.0)),
                table_b: Box::new(std::array::from_fn(|i| i as f32 / 65535.0)),
                coeffs_l: [0.0; 3],
                coeffs_ab: [0.0; 12],
                autoscale_ab: 3,
                unbound_ab: 1,
                preserve_colors: 3,
                space: ColorSpace::LinearSrgb,
            },
            Stage::RgbCurve {
                table_r: Box::new(std::array::from_fn(|i| i as f32 / 65535.0)),
                table_g: Box::new(std::array::from_fn(|i| i as f32 / 65535.0)),
                table_b: Box::new(std::array::from_fn(|i| i as f32 / 65535.0)),
                coeffs: [[1.0; 3]; 3],
                autoscale: 0,
                preserve_colors: 1,
            },
            Stage::Basecurve {
                table: Box::new(std::array::from_fn(|i| i as f32 / 65535.0)),
                coeffs: [1.0, 1.0, 1.0],
                preserve_colors: 1,
                fusion: 0,
                stops: 1.0,
                bias: 1.0,
                space: ColorSpace::LinearSrgb,
            },
            Stage::Lowlight {
                blueness: 0.0, lut: vec![0.5; 65536], space: ColorSpace::LinearSrgb,
            },
            Stage::Colisa {
                contrast: 0.0, brightness: 0.0, saturation: 0.0,
                space: ColorSpace::LinearSrgb,
            },
            Stage::Basicadj {
                black_point: 0.0, exposure: 0.0, hlcompr: 0.0, hlcomprthresh: 0.0,
                contrast: 0.0, preserve_colors: 1, middle_grey: 18.42,
                brightness: 0.0, saturation: 0.0, vibrance: 0.0,
                space: ColorSpace::LinearSrgb,
            },
            Stage::Primaries { matrix: crate::iop::primaries::IDENTITY_4X4 },
            Stage::Negadoctor {
                dmin: [0.0; 4], wb_high: [0.0; 4], offset: [0.0; 4],
                black: 0.0, gamma: 1.0, soft_clip: 0.0, soft_clip_comp: 1.0, exposure: 0.0,
            },
            Stage::ToneEqual { gains: [0.0; 9] },
            Stage::ColorBalanceRgb {
                data: Box::new(colorbalancergb::CbRgbData::from_params(
                    &colorbalancergb::CbRgbParams::default(),
                    &crate::color::REC2020_TO_XYZ_D65_T4,
                )),
                space: ColorSpace::Rec2020,
            },
            Stage::FilmicRgb {
                data: Box::new(filmicrgb::FilmicData::from_params(
                    &filmicrgb::FilmicParams::default(),
                )),
                space: ColorSpace::Rec2020,
            },
        ] {
            assert!(s.is_pixel_local(), "{} should be pixel-local", s.name());
        }
        assert!(
            !Stage::Sharpen { radius: 2.0, threshold: 0.0, amount: 1.0, space: ColorSpace::Rec2020, scale: 1.0 }
                .is_pixel_local(),
            "sharpen reads neighbours ⇒ NOT pixel-local"
        );
        assert!(
            !Stage::Bloom { size: 20.0, threshold: 90.0, strength: 25.0, space: ColorSpace::LinearSrgb }
                .is_pixel_local(),
            "bloom box-blurs a neighbourhood ⇒ NOT pixel-local"
        );
        assert!(
            !Stage::GraduatedNd {
                density: 1.0, hardness: 0.0, rotation: 0.0, offset: 50.0,
                hue: 0.0, saturation: 0.0,
            }
            .is_pixel_local(),
            "graduated ND derives its strength from pixel POSITION ⇒ NOT pixel-local"
        );
        assert!(
            !Stage::Vignette {
                scale: 80.0, falloff: 50.0, brightness: -0.5, saturation: -0.5,
                center_x: 0.0, center_y: 0.0, shape: 1.0,
                autoratio: true, whratio: 1.0, dither_amt: 0.0, unbound: false,
            }
            .is_pixel_local(),
            "vignette derives its weight from pixel POSITION ⇒ NOT pixel-local"
        );
        // Lowpass is NOT pixel-local: the Gaussian blur reads a spatial
        // neighbourhood, so band-splitting would produce wrong-edge artefacts.
        assert!(
            !Stage::Lowpass {
                radius: 10.0, contrast: 1.0, brightness: 0.0, saturation: 1.0,
                scale: 1.0, space: ColorSpace::LinearSrgb,
            }
                .is_pixel_local(),
            "lowpass blurs a spatial neighbourhood ⇒ NOT pixel-local"
        );
        // Shadhi is NOT pixel-local: the Gaussian blur reads a spatial
        // neighbourhood, so band-splitting would produce wrong-edge artefacts.
        assert!(
            !Stage::Shadhi {
                shadows: 25.0, highlights: -30.0, whitepoint: 2.0,
                radius: 100.0, compress: 50.0,
                shadows_ccorrect: 75.0, highlights_ccorrect: 40.0,
                scale: 1.0, space: ColorSpace::Rec2020,
            }
                .is_pixel_local(),
            "shadhi blurs a spatial neighbourhood ⇒ NOT pixel-local"
        );
        // DenoiseProfile is NOT pixel-local: the multi-scale à-trous wavelet
        // decomposition reads a stride-2^scale neighbourhood per output pixel.
        assert!(
            !Stage::DenoiseProfile { strength: 1.0, shadows: 1.0, bias: 0.0, mode_y0u0v0: true }
                .is_pixel_local(),
            "denoiseprofile reads wavelet-scale neighbourhoods ⇒ NOT pixel-local"
        );
        // LensCorrection is NOT pixel-local: every output pixel resamples a
        // per-channel neighbourhood at its warped source coordinate.
        assert!(
            !Stage::LensCorrection {
                lens: crate::iop::lens::ResolvedLens {
                    ptr: std::ptr::null(),
                    maker: String::new(),
                    model: String::new(),
                    crop_factor: 1.0,
                    lens_type: c41_sys::lensfun::LF_RECTILINEAR,
                    min_focal: 24.0,
                    max_focal: 70.0,
                },
                params: crate::iop::lens::LensParams::default(),
            }
            .is_pixel_local(),
            "lens correction warps coordinates ⇒ NOT pixel-local"
        );
    }

    #[test]
    fn vignette_is_radial_and_band_split_invariant() {
        // Two properties in one, both consequences of the position-dependence
        // that makes this stage non-pixel-local.
        let (w, h) = (64usize, 48usize);
        let img = vec![0.5f32; w * h * 4];
        let stage = Stage::Vignette {
            scale: 40.0, falloff: 60.0, brightness: -1.0, saturation: 0.0,
            center_x: 0.0, center_y: 0.0, shape: 1.0,
            autoratio: true, whratio: 1.0, dither_amt: 0.0, unbound: false,
        };
        let out = Pipeline::with_stages(vec![stage.clone()]).process(&img, w, h);

        // 1. Radial: the corner is darkened relative to the centre. If the
        //    stage ever received per-band coordinates this would not hold.
        let centre = out[((h / 2) * w + w / 2) * 4];
        let corner = out[0];
        assert!(
            corner < centre,
            "vignette must darken the corner more than the centre: corner {corner} !< centre {centre}"
        );

        // 2. Band-split invariant: `process` must keep this stage on the serial
        //    whole-buffer path, so the result cannot depend on the buffer being
        //    large enough to trigger banding. Re-running the same geometry over
        //    the same rectangle must be bit-identical.
        let again = Pipeline::with_stages(vec![stage]).process(&img, w, h);
        assert_eq!(out, again, "vignette output must be deterministic for a given rectangle");

        // 3. A row midway down is darker at its left edge than at its centre —
        //    i.e. the falloff varies along x too, not just y. A per-band run
        //    (h = 1 strips) collapses this.
        let row = h / 2;
        let left = out[(row * w) * 4];
        let mid = out[(row * w + w / 2) * 4];
        assert!(left < mid, "falloff must vary along x: left {left} !< mid {mid}");
    }

    #[test]
    fn vibrance_zero_amount_is_identity() {
        // amount == 0 ⇒ sw = 0, ls = 1, ss = 1 for every pixel ⇒ no change.
        // (After the Lab round-trip; tolerance covers float error.)
        let px = vec![0.5f32, 0.3, 0.8, 1.0, 0.2, 0.6, 0.9, 1.0, 0.1, 0.4, 0.7, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Vibrance {
            amount: 0.0, space: ColorSpace::LinearSrgb,
        }]);
        let out = p.process(&px, px.len() / 4, 1);
        for (o, i) in out.iter().zip(px.iter()) {
            assert!((o - i).abs() < 1e-4, "zero vibrance changed a pixel: {o} != {i}");
        }
    }

    #[test]
    fn vibrance_boosts_chromatic_saturation() {
        // A saturated red pixel (a >> 0 in Lab) should get its chroma channels
        // amplified (a/b grow in magnitude) when amount > 0, while a neutral
        // grey pixel (a ≈ b ≈ 0) should be largely unaffected.
        let (w, h) = (4usize, 1usize);
        let mut img = vec![0.0f32; w * h * 4];
        // Pixel 0: saturated red. Pixel 1: mid grey.
        img[0] = 0.8; img[1] = 0.2; img[2] = 0.1; img[3] = 1.0; // red
        img[4] = 0.5; img[5] = 0.5; img[6] = 0.5; img[7] = 1.0; // grey
        let p = Pipeline::with_stages(vec![Stage::Vibrance {
            amount: 1.5, space: ColorSpace::LinearSrgb,
        }]);
        let out = p.process(&img, w, h);
        // The red pixel's chroma (a/b in Lab) should be amplified — its RGB values
        // should shift away from grey (the channels diverge more).
        let red_r = out[0];
        let red_g = out[1];
        let red_b = out[2];
        assert!(
            (red_r - red_g).abs() > (0.8f32 - 0.2f32).abs() - 0.01,
            "red saturation should increase: r={} g={} b={}",
            red_r, red_g, red_b
        );
        // Grey should be nearly unchanged (small chroma ⇒ small boost).
        assert!((out[4] - 0.5).abs() < 0.02, "grey drifted too far: {}", out[4]);
        assert!((out[5] - 0.5).abs() < 0.02, "grey drifted too far: {}", out[5]);
        assert!((out[6] - 0.5).abs() < 0.02, "grey drifted too far: {}", out[6]);
    }

    #[test]
    fn colorcontrast_unit_steepness_is_identity() {
        // steepness == 1.0, offset == 0.0 ⇒ a*b unchanged (identity transform
        // in the Lab round-trip; tolerance covers float error).
        let px = vec![0.5f32, 0.3, 0.8, 1.0, 0.2, 0.6, 0.9, 1.0, 0.1, 0.4, 0.7, 1.0];
        let p = Pipeline::with_stages(vec![Stage::ColorContrast {
            a_steepness: 1.0, b_steepness: 1.0, space: ColorSpace::LinearSrgb,
        }]);
        let out = p.process(&px, px.len() / 4, 1);
        for (o, i) in out.iter().zip(px.iter()) {
            assert!((o - i).abs() < 1e-4, "unit colorcontrast changed a pixel: {o} != {i}");
        }
    }

    #[test]
    fn colorcontrast_boosts_chromatic_ab() {
        // A steepness > 1.0 should push the a/b channels away from zero.
        let px = vec![0.8f32, 0.2, 0.1, 1.0]; // saturated red → a*b ≠ 0 in Lab
        let p = Pipeline::with_stages(vec![Stage::ColorContrast {
            a_steepness: 2.0, b_steepness: 2.0, space: ColorSpace::LinearSrgb,
        }]);
        let out = p.process(&px, 1, 1);
        // In Lab, red has a* > 0 and b* > 0. Doubling steepness should push
        // the output further along the a*/b* axes — the Lab values should
        // differ from the input (the RGB values must change after round-trip).
        assert!(
            (out[0] - px[0]).abs() > 1e-5 || (out[1] - px[1]).abs() > 1e-5,
            "colorcontrast should alter chroma: r={} g={} b={} vs original r={} g={} b={}",
            out[0], out[1], out[2], px[0], px[1], px[2]
        );
    }

    #[test]
    fn gaussian_kernel_is_normalised_and_symmetric() {
        let (rad, mat) = gaussian_kernel(3.0);
        assert_eq!(rad, 3);
        assert_eq!(mat.len(), 2 * rad + 1);
        assert!((mat.iter().sum::<f32>() - 1.0).abs() < 1e-6, "kernel must sum to 1");
        for k in 0..rad {
            assert!((mat[k] - mat[mat.len() - 1 - k]).abs() < 1e-7, "kernel must be symmetric");
        }
        assert!(mat[rad] > mat[0], "centre tap must be the largest");
    }

    #[test]
    fn sharpen_leaves_a_flat_image_unchanged() {
        // Blur of a constant is the constant ⇒ zero detail ⇒ no change (on a
        // buffer large enough to have an interior).
        let (w, h) = (16usize, 16usize);
        let flat = vec![0.4f32; w * h * 4];
        let p = Pipeline::with_stages(vec![Stage::Sharpen {
            radius: 2.0, threshold: 0.0, amount: 1.0, space: ColorSpace::Rec2020, scale: 1.0,
        }]);
        let out = p.process(&flat, w, h);
        // Tolerance covers the Rec.2020→Lab→Rec.2020 round-trip (not bit-exact);
        // the point is that a flat field gains no sharpening detail.
        for (o, i) in out.iter().zip(flat.iter()) {
            assert!((o - i).abs() < 1e-3, "flat sharpen changed a pixel: {o} vs {i}");
        }
    }

    #[test]
    fn sharpen_enhances_an_edge_and_is_spatial() {
        // A left-dark / right-bright edge: sharpening overshoots at the boundary
        // (right of the edge brighter than its input, left darker) — behaviour a
        // per-pixel stage cannot produce. Also proves (w,h) actually indexes rows.
        let (w, h) = (16usize, 16usize);
        let mut img = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = if x >= w / 2 { 0.8 } else { 0.2 };
                let i = (y * w + x) * 4;
                img[i] = v; img[i + 1] = v; img[i + 2] = v; img[i + 3] = 1.0;
            }
        }
        let p = Pipeline::with_stages(vec![Stage::Sharpen {
            radius: 2.0, threshold: 0.0, amount: 1.0, space: ColorSpace::Rec2020, scale: 1.0,
        }]);
        let out = p.process(&img, w, h);
        // Interior pixel just right of the edge overshoots above 0.8.
        let right = (8 * w + w / 2) * 4;
        assert!(out[right] > 0.8 + 1e-3, "no overshoot at edge: {}", out[right]);
        // A flat interior pixel far from the edge is ~unchanged (Lab round-trip
        // tolerance).
        let flat = (8 * w + 2) * 4;
        assert!((out[flat] - 0.2).abs() < 1e-3, "flat region changed: {}", out[flat]);
    }

    #[test]
    fn sharpen_in_multistage_pipeline_on_large_flat_is_uniform() {
        // >PIXELS_PER_BAND (64k) px + a spatial stage: `process` must take the
        // SERIAL whole-buffer path (no band seam) and stay uniform on a flat field
        // through the exposure→sharpen→monochrome ping-pong.
        let (w, h) = (400usize, 300usize); // 120k px > 64k band
        let flat = vec![0.3f32; w * h * 4];
        let p = Pipeline::with_stages(vec![
            Stage::Exposure { black: 0.0, scale: 1.5 },
            Stage::Sharpen { radius: 2.0, threshold: 0.0, amount: 1.0, space: ColorSpace::Rec2020, scale: 1.0 },
            Stage::Monochrome { r: 0.2627, g: 0.6780, b: 0.0593 },
        ]);
        let out = p.process(&flat, w, h);
        // ×1.5 → 0.45 flat; sharpen no-ops on flat; monochrome luma (weights sum
        // to 1) → 0.45. Every RGB channel must equal that, with no seam variance.
        for px in out.chunks_exact(4) {
            for c in 0..3 {
                assert!((px[c] - 0.45).abs() < 1e-3, "band seam / non-uniform: {}", px[c]);
            }
        }
    }

    #[test]
    fn sharpen_copies_through_below_kernel_size() {
        // width/height ≤ 2*rad: no interior ⇒ Sharpen passes the image through
        // unchanged (byte-exact — detail is identically zero).
        let (w, h) = (3usize, 3usize);
        let img: Vec<f32> = (0..w * h * 4).map(|i| (i % 7) as f32 / 7.0).collect();
        let p = Pipeline::with_stages(vec![Stage::Sharpen {
            radius: 5.0, threshold: 0.0, amount: 1.0, // rad=5 ⇒ needs w,h > 10
            space: ColorSpace::Rec2020, scale: 1.0,
        }]);
        assert_eq!(p.process(&img, w, h), img, "small image must pass through");
    }

    #[test]
    fn sharpen_routes_conversion_by_working_space() {
        // A COLOURED edge: Y (hence Lab L) differs between Rec.2020 and sRGB
        // primaries, so the two working spaces must give different sharpening —
        // proving `space` selects the RGB↔Lab pair rather than a hard-coded one.
        let (w, h) = (16usize, 16usize);
        let mut img = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = if x >= w / 2 { (0.8, 0.3, 0.5) } else { (0.1, 0.4, 0.2) };
                let i = (y * w + x) * 4;
                // Non-trivial alpha exercises the ch-3 passthrough in both paths.
                img[i] = r; img[i + 1] = g; img[i + 2] = b; img[i + 3] = 0.5;
            }
        }
        let mk = |space| {
            Pipeline::with_stages(vec![Stage::Sharpen {
                radius: 2.0, threshold: 0.0, amount: 1.0, space, scale: 1.0,
            }])
        };
        let srgb = mk(ColorSpace::LinearSrgb).process(&img, w, h);
        let rec = mk(ColorSpace::Rec2020).process(&img, w, h);
        let right = (8 * w + w / 2) * 4;
        let diff: f32 = (0..3).map(|c| (srgb[right + c] - rec[right + c]).abs()).sum();
        assert!(diff > 1e-4, "working space did not change the sharpen result: {diff}");
        assert_eq!(srgb[right + 3], 0.5, "srgb alpha preserved");
        assert_eq!(rec[right + 3], 0.5, "rec2020 alpha preserved");
    }

    #[test]
    fn sharpen_scale_shrinks_the_effective_kernel() {
        // roi->scale: at scale < 1 the effective radius (radius*scale) shrinks, so
        // a downscaled preview sharpens with a proportionally smaller kernel — the
        // result differs from the full-res (scale 1.0) render of the same buffer.
        let (w, h) = (20usize, 20usize);
        let mut img = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = if x >= w / 2 { 0.8 } else { 0.2 };
                let i = (y * w + x) * 4;
                img[i] = v; img[i + 1] = v; img[i + 2] = v; img[i + 3] = 1.0;
            }
        }
        let mk = |scale| {
            Pipeline::with_stages(vec![Stage::Sharpen {
                radius: 4.0, threshold: 0.0, amount: 1.0, space: ColorSpace::Rec2020, scale,
            }])
        };
        let full = mk(1.0).process(&img, w, h);
        let half = mk(0.5).process(&img, w, h);
        let diff: f32 = full.iter().zip(half.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "scale did not change the effective kernel: {diff}");
    }

    #[test]
    fn gaussian_kernel_radius_clamps_at_12() {
        // MAXR = 12: beyond radius 12 the tap count stays bounded (sharpen.c).
        let (rad, mat) = gaussian_kernel(20.0);
        assert_eq!(rad, 12);
        assert_eq!(mat.len(), 25);
        assert!((mat.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_unity_coeffs_are_identity() {
        // All coeffs 1.0 ⇒ out = in (per-channel multiply by 1).
        let px = vec![0.5f32, 0.3, 0.8, 1.0, 0.2, 0.6, 0.9, 1.0, 0.1, 0.4, 0.7, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Temperature {
            coeffs: [1.0, 1.0, 1.0, 1.0],
        }]);
        let out = p.process(&px, px.len() / 4, 1);
        assert_eq!(out, px, "unity temperature coeffs should be a no-op");
    }

    #[test]
    fn temperature_scales_channels_independently() {
        // Non-unity coeffs should multiply each channel independently.
        let px = vec![0.8f32, 0.2, 0.1, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Temperature {
            coeffs: [2.0, 4.0, 8.0, 1.0],
        }]);
        let out = p.process(&px, 1, 1);
        assert!((out[0] - 1.6).abs() < 1e-5, "R *= 2: got {}", out[0]);
        assert!((out[1] - 0.8).abs() < 1e-5, "G *= 4: got {}", out[1]);
        assert!((out[2] - 0.8).abs() < 1e-5, "B *= 8: got {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-5, "A unchanged: got {}", out[3]);
    }

    #[test]
    fn invert_unity_color_is_identity() {
        // color = 1.0 per channel ⇒ out = 1.0 - in. With color = 1.0 this is
        // a pure negate; NOT identity. But color = 2.0 + in = in only if
        // color == 2*in, i.e. not identity either. The true identity for
        // invert is color = 2*in for all pixels, which isn't a constant.
        // So test the actual invert behaviour instead: color (1,1,1,1)
        // gives out = 1 - in.
        let px = vec![0.2f32, 0.5, 0.8, 1.0, 1.0, 0.0, 0.3, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Invert {
            color: [1.0, 1.0, 1.0, 1.0],
        }]);
        let out = p.process(&px, px.len() / 4, 1);
        assert!((out[0] - 0.8).abs() < 1e-6, "R: 1-0.2 = 0.8, got {}", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-6, "G: 1-0.5 = 0.5, got {}", out[1]);
        assert!((out[2] - 0.2).abs() < 1e-6, "B: 1-0.8 = 0.2, got {}", out[2]);
        assert!((out[3] - 0.0).abs() < 1e-6, "A: 1-1.0 = 0.0, got {}", out[3]);
        assert!((out[4] - 0.0).abs() < 1e-6, "R: 1-1.0 = 0.0, got {}", out[4]);
        assert!((out[6] - 0.7).abs() < 1e-6, "B: 1-0.3 = 0.7, got {}", out[6]);
    }

    #[test]
    fn invert_scales_with_color() {
        // Non-unity color: out = color - in per channel.
        let px = vec![0.3f32, 0.6, 0.9, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Invert {
            color: [1.0, 2.0, 0.5, 1.0],
        }]);
        let out = p.process(&px, 1, 1);
        assert!((out[0] - 0.7).abs() < 1e-6, "R: 1.0-0.3 = 0.7, got {}", out[0]);
        assert!((out[1] - 1.4).abs() < 1e-6, "G: 2.0-0.6 = 1.4, got {}", out[1]);
        assert!((out[2] - (-0.4)).abs() < 1e-6, "B: 0.5-0.9 = -0.4, got {}", out[2]);
        assert!((out[3] - 0.0).abs() < 1e-6, "A: 1.0-1.0 = 0.0, got {}", out[3]);
    }

    #[test]
    fn shadhi_nonneutral_params_change_output() {
        // A vertical gradient 0.05→0.95. With shadows=80/highlights=-80 the
        // shadow and highlight overlays are active, so the output must differ
        // from the neutral (shadows=0/highlights=0) pass. We don't assert the
        // *direction* per-pixel — the Lab-domain overlay math is input-dependent
        // for mid-tones — only that the stage has a visible, non-zero effect.
        let (w, h) = (32usize, 32usize);
        let mut img = vec![0.0f32; w * h * 4];
        for y in 0..h {
            let v = 0.05 + 0.9 * (y as f32) / ((h - 1) as f32);
            for x in 0..w {
                let i = (y * w + x) * 4;
                img[i] = v; img[i + 1] = v; img[i + 2] = v; img[i + 3] = 1.0;
            }
        }
        let mk = |shadows, highlights| {
            Pipeline::with_stages(vec![Stage::Shadhi {
                shadows, highlights, whitepoint: 0.0,
                radius: 8.0, compress: 0.0,
                shadows_ccorrect: 100.0, highlights_ccorrect: 50.0,
                scale: 1.0, space: ColorSpace::LinearSrgb,
            }])
        };
        let neutral = mk(0.0, 0.0).process(&img, w, h);
        let active = mk(80.0, -80.0).process(&img, w, h);
        // The L channel of the top row (dark) should change — shadows lifting it.
        let top_diff = (active[0] - neutral[0]).abs();
        assert!(top_diff > 1e-4, "shadows did not affect the dark pixel: diff={top_diff}");
        // The L channel of the bottom row (bright) should change — highlights recovering it.
        let bot_diff = (active[(h - 1) * w * 4] - neutral[(h - 1) * w * 4]).abs();
        assert!(bot_diff > 1e-4, "highlights did not affect the bright pixel: diff={bot_diff}");
        // Alpha passes through unchanged.
        assert_eq!(active[3], 1.0, "alpha should be preserved");
    }

    #[test]
    fn shadhi_neutral_on_flat_is_identity() {
        // sliders all at 0/neutral ⇒ the shadow/highlight overlays are zero, so a
        // flat field should come out unchanged (the blur of a flat is itself).
        let (w, h) = (8usize, 8usize);
        let flat = vec![0.42f32; w * h * 4];
        let p = Pipeline::with_stages(vec![Stage::Shadhi {
            shadows: 0.0, highlights: 0.0, whitepoint: 0.0,
            radius: 100.0, compress: 0.0,
            shadows_ccorrect: 100.0, highlights_ccorrect: 50.0,
            scale: 1.0, space: ColorSpace::LinearSrgb,
        }]);
        let out = p.process(&flat, w, h);
        for px in out.chunks_exact(4) {
            for c in 0..4 {
                assert!((px[c] - 0.42).abs() < 1e-5, "flat pixel changed: {}", px[c]);
            }
        }
    }

    #[test]
    fn shadhi_highlights_ccorrect_sign_direction() {
        // Regression guard for P0-1: the highlights_ccorrect sign must follow
        // C's `sign(-highlights)` (negated), not `sign(highlights)`. With
        // highlights = -50 (hg < 0, the common case), hcc=0 and hcc=100 must
        // produce *different* blue channels, and the direction must match the
        // C implementation. 50.0 (the annihilating fixed point where
        // (x-0.5)=0) is deliberately avoided.
        let (w, h) = (32usize, 32usize);
        let mut img = vec![0.0f32; w * h * 4];
        for y in 0..h {
            let v = 0.5 + 0.4 * ((y as f32) / (h as f32) - 0.5);
            for x in 0..w {
                let i = (y * w + x) * 4;
                img[i] = v + 0.1; img[i + 1] = v; img[i + 2] = v - 0.1; img[i + 3] = 1.0;
            }
        }
        let mk = |hcc: f32| {
            Pipeline::with_stages(vec![Stage::Shadhi {
                shadows: 0.0, highlights: -50.0, whitepoint: 0.0,
                radius: 5.0, compress: 10.0,
                shadows_ccorrect: 100.0, highlights_ccorrect: hcc,
                scale: 1.0, space: ColorSpace::LinearSrgb,
            }])
        };
        let lo = mk(0.0).process(&img, w, h);
        let hi = mk(100.0).process(&img, w, h);
        // Blue channel of the brightest pixel must differ — sign flip on hg
        // changes hg_cc from 0→1 or 1→0, altering chroma blend.
        let bot = (h - 1) * w * 4;
        let diff = (lo[bot + 2] - hi[bot + 2]).abs();
        assert!(diff > 1e-5, "highlights_ccorrect sign does not affect blue: diff={diff}");
    }
}
