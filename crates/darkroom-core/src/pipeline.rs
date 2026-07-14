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

use crate::iop::{channelmixer, exposure, sharpen, sigmoid, splittoning, velvia};

/// Rec.2020 luminance weights (the pipeline works in linear Rec.2020). Used by
/// [`Stage::Sharpen`] to build the luma channel it sharpens.
const REC2020_LUMA: [f32; 3] = [0.2627, 0.6780, 0.0593];

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
    /// neighbourhood, so [`Stage::apply`] needs `(width, height)`). Sharpens a
    /// Rec.2020 **luma** channel via the migrated separable-Gaussian
    /// [`sharpen::darkroom_sharpen_process`] kernel, then adds the luma detail
    /// back to R/G/B. `radius` sets the Gaussian; `amount` scales the added detail.
    ///
    /// **`threshold` is in LINEAR-luma units (~[0, 1])**, NOT darktable's Lab-`L`
    /// [0, 100]: this pipeline sharpens scene-linear Rec.2020 luma, so a caller
    /// mapping a 0..100 darktable slider must divide by ~100 — a raw 100 here
    /// zeroes all detail (`apply` `debug_assert!`s `threshold <= 1.0` as a
    /// tripwire).
    ///
    /// This is a luminance unsharp mask, NOT the bit-exact darktable Lab-`L`
    /// sharpen (which needs the RGB↔Lab color-space infra — a separate
    /// migration): adding one scalar detail to all of R/G/B shifts chroma on
    /// saturated edges (desaturates on overshoot). A ratio-preserving variant is
    /// the upgrade once Lab lands.
    Sharpen { radius: f32, threshold: f32, amount: f32 },
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
            // Sharpen reads a spatial neighbourhood → NOT pixel-local, so
            // `process` runs it on the whole buffer (serial path) where (w,h) is
            // the true image rectangle, never a band's pixel run.
            Stage::Sharpen { .. } => false,
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
            Stage::Sharpen { radius, threshold, amount } => {
                // threshold is linear-luma (~[0,1]), not darktable's Lab-L [0,100]
                // — catch a mis-mapped 0..100 slider value loudly (see the doc).
                debug_assert!(threshold <= 1.0, "Sharpen threshold is linear-luma (~[0,1]); got {threshold}");
                let n = width * height;
                // Sharpen the Rec.2020 luma: pack (luma,0,0,0), run the migrated
                // separable-Gaussian unsharp kernel (it sharpens channel 0), then
                // add the resulting luma detail back to R/G/B (luminance unsharp
                // mask — shifts chroma on saturated edges; see the Stage doc).
                // Alpha (ch 3) passes through.
                // TODO(perf): luma_in/out are n*4 but only ch 0 is used — a
                // planar-luma (stride-1) kernel would cut ~75% of this scratch and
                // the wasted chroma/border work at export scale.
                let mut luma_in = vec![0.0f32; n * 4];
                for p in 0..n {
                    let i = p * 4;
                    luma_in[i] = REC2020_LUMA[0] * input[i]
                        + REC2020_LUMA[1] * input[i + 1]
                        + REC2020_LUMA[2] * input[i + 2];
                }
                let (rad, mat) = gaussian_kernel(radius);
                let mut luma_out = vec![0.0f32; n * 4];
                // Kernel needs width,height >= 2*rad+1; below that it can't form
                // an interior, so copy through (no sharpening) — matches the C
                // caller's small-image fast path.
                if width > 2 * rad && height > 2 * rad {
                    // Safety: luma_in/luma_out are exactly n*4 floats and `mat` is
                    // 2*rad+1 taps — the kernel's documented contract.
                    unsafe {
                        sharpen::darkroom_sharpen_process(
                            luma_in.as_ptr(),
                            luma_out.as_mut_ptr(),
                            mat.as_ptr(),
                            width,
                            height,
                            rad as i32,
                            threshold,
                            amount,
                        );
                    }
                } else {
                    luma_out.copy_from_slice(&luma_in);
                }
                for p in 0..n {
                    let i = p * 4;
                    let detail = luma_out[i] - luma_in[i];
                    output[i] = input[i] + detail;
                    output[i + 1] = input[i + 1] + detail;
                    output[i + 2] = input[i + 2] + detail;
                    output[i + 3] = input[i + 3];
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
        ] {
            assert!(s.is_pixel_local(), "{} should be pixel-local", s.name());
        }
        assert!(
            !Stage::Sharpen { radius: 2.0, threshold: 0.0, amount: 1.0 }.is_pixel_local(),
            "sharpen reads neighbours ⇒ NOT pixel-local"
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
            radius: 2.0, threshold: 0.0, amount: 1.0,
        }]);
        let out = p.process(&flat, w, h);
        for (o, i) in out.iter().zip(flat.iter()) {
            assert!((o - i).abs() < 1e-5, "flat sharpen changed a pixel");
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
            radius: 2.0, threshold: 0.0, amount: 1.0,
        }]);
        let out = p.process(&img, w, h);
        // Interior pixel just right of the edge overshoots above 0.8.
        let right = (8 * w + w / 2) * 4;
        assert!(out[right] > 0.8 + 1e-3, "no overshoot at edge: {}", out[right]);
        // A flat interior pixel far from the edge is unchanged.
        let flat = (8 * w + 2) * 4;
        assert!((out[flat] - 0.2).abs() < 1e-4, "flat region changed: {}", out[flat]);
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
            Stage::Sharpen { radius: 2.0, threshold: 0.0, amount: 1.0 },
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
        }]);
        assert_eq!(p.process(&img, w, h), img, "small image must pass through");
    }

    #[test]
    fn gaussian_kernel_radius_clamps_at_12() {
        // MAXR = 12: beyond radius 12 the tap count stays bounded (sharpen.c).
        let (rad, mat) = gaussian_kernel(20.0);
        assert_eq!(rad, 12);
        assert_eq!(mat.len(), 25);
        assert!((mat.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
