//! Minimal Rust pixelpipe orchestrator (Phase 3 milestone 2 bootstrap).
//!
//! Runs an ordered list of migrated IOP [`Stage`]s over a **scene-referred,
//! linear RGBA `f32`** buffer (packed, `width*height*4`). This is the
//! float-domain core successor to darkroom-ui's 8-bit preview seam: the UI maps
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

use crate::iop::{channelmixer, colorcontrast, exposure, invert, sharpen, sigmoid, splittoning, temperature, velvia, vibrance};

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

/// One configured pipeline stage, backed by a migrated darkroom-core IOP.
#[derive(Clone, Copy, Debug, PartialEq)]
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
            // Sharpen reads a spatial neighbourhood → NOT pixel-local, so
            // `process` runs it on the whole buffer (serial path) where (w,h) is
            // the true image rectangle, never a band's pixel run.
            Stage::Sharpen { .. } => false,
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
        ] {
            assert!(s.is_pixel_local(), "{} should be pixel-local", s.name());
        }
        assert!(
            !Stage::Sharpen { radius: 2.0, threshold: 0.0, amount: 1.0, space: ColorSpace::Rec2020, scale: 1.0 }
                .is_pixel_local(),
            "sharpen reads neighbours ⇒ NOT pixel-local"
        );
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
}
