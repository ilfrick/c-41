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

use crate::iop::{channelmixer, exposure, splittoning, velvia};

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
}

impl Stage {
    /// A short stable identifier (matches the darktable IOP operation name).
    pub fn name(&self) -> &'static str {
        match self {
            Stage::Exposure { .. } => "exposure",
            Stage::Velvia { .. } => "velvia",
            Stage::Splittoning { .. } => "splittoning",
            Stage::Monochrome { .. } => "channelmixer",
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

    /// Run all stages in order over a copy of `input` (packed RGBA `f32`,
    /// length a multiple of 4) and return the result. An empty pipeline returns
    /// the input unchanged. Uses two buffers ping-ponged across stages.
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
        let mut front = input.to_vec();
        let mut back = vec![0.0f32; input.len()];
        for stage in &self.stages {
            stage.apply(&front, &mut back);
            std::mem::swap(&mut front, &mut back);
        }
        front
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
    #[should_panic(expected = "packed RGBA")]
    fn process_rejects_non_multiple_of_four() {
        // 6 elements is not a whole number of RGBA pixels.
        let bad = vec![0.0f32; 6];
        Pipeline::with_stages(vec![Stage::Velvia { strength: 1.0, bias: 1.0 }]).process(&bad);
    }
}
