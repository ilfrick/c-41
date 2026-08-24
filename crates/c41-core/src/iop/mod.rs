//! IOP (Image OPerator) modules — Phase 1 target.
//!
//! Each IOP from src/iop/*.c will become a struct in this module
//! implementing `IopProcess`. Migration order follows RUST_MIGRATION_PLAN.md.

pub mod agx;
pub mod ashift;
pub mod clipping;
pub mod denoiseprofile;
pub mod liquify;
pub mod retouch;
pub mod basicadj;
pub mod bloom;
pub mod globaltonemap;
pub mod dither;
pub mod eaw;
pub mod highpass;
pub mod invert;
pub mod monochrome;
pub mod channelmixer;
pub mod colorbalance;
pub mod colorbalancergb;
pub mod colorout;
pub mod filmic;
pub mod lowpass;
pub mod shadhi;
pub mod soften;
pub mod sharpen;
pub mod colisa;
pub mod colorcontrast;
pub mod colorcorrection;
pub mod colorize;
pub mod colorzones;
pub mod exposure;
pub mod grain;
pub mod graduatednd;
pub mod levels;
pub mod lut3d;
pub mod lowlight;
pub mod negadoctor;
pub mod primaries;
pub mod profile_gamma;
pub mod relight;
pub mod rgbcurve;
pub mod rgblevels;
pub mod sigmoid;
pub mod splittoning;
pub mod tonecurve;
pub mod velvia;
pub mod vibrance;
pub mod vignette;
pub mod overlay;
pub mod atrous;
pub mod colorin;
pub mod temperature;
pub mod watermark;
pub mod zonesystem;
pub mod channelmixerrgb;
pub mod basecurve;
pub mod hazeremoval;
pub mod censorize;
pub mod overexposed;
pub mod hotpixels;
pub mod clahe;
pub mod rawprepare;
pub mod highlights;
pub mod defringe;
pub mod colorchecker;
pub mod rasterfile;
pub mod diffuse;
pub mod colortransfer;
pub mod cacorrectrgb;
pub mod rawdenoise;
pub mod rawoverexposed;
pub mod demosaic;
pub mod markesteijn;
pub mod fdc_tables;
pub mod fdc;
pub mod colormapping;
pub mod colorequal;
pub mod toneequal;
pub mod useless;
pub mod gamma;
pub mod blurs;
pub mod filmicrgb;
pub mod cacorrect;
pub mod geometry;
pub mod colorharmonizer;

use crate::{params::IopParams, roi::RoiIn, Result};

/// Placeholder for an OpenCL device buffer.
/// Will be replaced by a proper `opencl3`-backed type in Phase 1.
pub struct ClBuffer {
    _priv: (),
}

/// Core trait every IOP module must implement.
pub trait IopProcess: Send + Sync {
    /// CPU path: process `input` pixels (row-major RGBA f32) into `output`.
    fn process(
        &self,
        input: &[f32],
        output: &mut [f32],
        params: &IopParams,
        roi: &RoiIn,
    ) -> Result<()>;

    /// GPU path — must produce identical results to `process()`.
    fn process_cl(&self, buf: &mut ClBuffer, params: &IopParams) -> Result<()>;

    /// Human-readable name used in UI and logging.
    fn name(&self) -> &'static str;

    fn has_opencl(&self) -> bool {
        false
    }
}
