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
//! Order of `Pipeline::stages` is the processing order. Future work: real
//! `iop_order`, OpenCL, a raw-decode/demosaic front end as the input, and a
//! ROI/(width,height) signature once a geometry-aware IOP is added (the current
//! stages are all strictly per-pixel, so a size-agnostic `&[f32]` suffices).
//!
//! The 4th channel is darktable scene-referred *scratch/padding*, **not** display
//! alpha. Stages follow their C originals: exposure transforms all four channels
//! (faithful to exposure.c's flat loop), while velvia/splittoning/channelmixer
//! pass channel 4 through. Don't "fix" exposure to preserve it — that diverges
//! from the C pipeline.

use crate::iop::{channelmixer, exposure, sigmoid, splittoning, velvia};

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
    pub fn apply(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len(), "apply: in/out length mismatch");
        debug_assert_eq!(input.len() % 4, 0, "apply: buffer must be packed RGBA (len % 4 == 0)");
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
    /// Every [`Stage`] is a position-independent per-pixel map (no stage reads a
    /// neighbouring pixel — the same property that lets geometry commute with the
    /// pipeline), so the buffer is split into pixel-aligned bands processed in
    /// parallel; each band runs the full stage sequence through its own ping-pong
    /// scratch. The result is **bit-identical** to a whole-buffer serial run
    /// (bands never interact) — pinned by `parallel_result_is_split_invariant`.
    /// Small buffers stay serial to avoid rayon overhead.
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        // Hard guard at the trust boundary (fires in release too): the
        // `chunks_exact(4)` stages would otherwise silently leave a stale tail.
        assert!(
            input.len().is_multiple_of(4),
            "Pipeline::process: buffer must be packed RGBA (len % 4 == 0), got {}",
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
            // One band: no rayon overhead.
            let mut scratch = vec![0.0f32; input.len()];
            self.process_band(input, &mut output, &mut scratch);
        } else {
            use rayon::prelude::*;
            // for_each_init reuses one scratch buffer per worker thread (sized to
            // a full band, sliced to the current chunk length) — no per-band alloc.
            input
                .par_chunks(BAND)
                .zip(output.par_chunks_mut(BAND))
                .for_each_init(
                    || vec![0.0f32; BAND],
                    |scratch, (in_band, out_band)| {
                        self.process_band(in_band, out_band, &mut scratch[..in_band.len()]);
                    },
                );
        }
        output
    }

    /// Run every stage over one band `input` into `output` (equal lengths, both a
    /// multiple of 4), ping-ponging between `output` and the caller-provided
    /// `scratch` (also equal length) so no per-band allocation happens on the hot
    /// path. Caller guarantees `self.stages` is non-empty.
    fn process_band(&self, input: &[f32], output: &mut [f32], scratch: &mut [f32]) {
        // First stage reads `input`; every later stage ping-pongs output<->scratch.
        self.stages[0].apply(input, output);
        let mut result_in_output = true;
        for stage in &self.stages[1..] {
            if result_in_output {
                stage.apply(output, scratch);
            } else {
                stage.apply(scratch, output);
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
        assert_eq!(Pipeline::new().process(&px), px);
    }

    #[test]
    fn exposure_scales_all_channels() {
        // out = (in - 0) * 2  (darktable applies exposure to all 4 channels)
        let px = vec![0.2f32, 0.4, 0.6, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Exposure { black: 0.0, scale: 2.0 }]);
        let out = p.process(&px);
        assert_eq!(out, vec![0.4, 0.8, 1.2, 2.0]);
    }

    #[test]
    fn black_point_then_scale() {
        // out = (in - black) * scale
        let px = vec![0.5f32, 0.5, 0.5, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Exposure { black: 0.1, scale: 2.0 }]);
        let out = p.process(&px);
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
        let out = p.process(&px);
        assert!((out[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn monochrome_yields_equal_rgb() {
        // weights (0.2,0.7,0.1) on (1.0,0.5,0.0): luma = 0.2+0.35+0 = 0.55, R=G=B
        let px = vec![1.0f32, 0.5, 0.0, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Monochrome { r: 0.2, g: 0.7, b: 0.1 }]);
        let out = p.process(&px);
        assert!((out[0] - 0.55).abs() < 1e-5);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn velvia_zero_strength_is_identity() {
        let px = vec![0.8f32, 0.2, 0.1, 1.0];
        let p = Pipeline::with_stages(vec![Stage::Velvia { strength: 0.0, bias: 1.0 }]);
        assert_eq!(p.process(&px), px);
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
        let out = p.process(&px);
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
        let out = p.process(&px);
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
            let out = p.process(&[v, v, v, 1.0]);
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
        let hi = p.process(&[0.1845 + d, 0.1845 + d, 0.1845 + d, 1.0])[0];
        let lo = p.process(&[0.1845 - d, 0.1845 - d, 0.1845 - d, 1.0])[0];
        let slope = (hi - lo) / (2.0 * d);
        assert!((slope - 1.2068).abs() < 0.01, "grey slope off: {slope}");
    }

    #[test]
    #[should_panic(expected = "packed RGBA")]
    fn process_rejects_non_multiple_of_four() {
        // 6 elements is not a whole number of RGBA pixels.
        let bad = vec![0.0f32; 6];
        Pipeline::with_stages(vec![Stage::Velvia { strength: 1.0, bias: 1.0 }]).process(&bad);
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
        let out = p.process(&px);
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

        let full = p.process(&px);
        let half = (px.len() / 2) & !3; // pixel-aligned split point
        let mut split = p.process(&px[..half]);
        split.extend_from_slice(&p.process(&px[half..]));

        assert_eq!(full, split);
    }

    #[test]
    fn all_current_stages_are_pixel_local() {
        // The band-parallel path is only bit-identical to serial while every
        // stage is pixel-local; pin that all shipping stages qualify so the
        // parallel branch actually engages (and flag any future regression).
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
    }
}
