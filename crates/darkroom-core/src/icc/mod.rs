//! Pure-Rust ICC colour-management engine — the replacement for darktable's
//! Little-CMS (`lcms2`) dependency in the input/output colour transforms.
//!
//! Goal: the same functionality as the LCMS path (parse arbitrary ICC v2/v4
//! profiles — matrix-shaper **and** cLUT/LUT profiles — and transform pixels
//! between them), at accuracy ≥ LCMS, entirely in Rust (no C link). Because the
//! shipped `darkroom-rs` product ships no LCMS, there is no bit-exact-to-LCMS
//! constraint — the engine only has to be *correct* (and we aim for more accurate
//! where LCMS takes shortcuts, e.g. we interpolate/evaluate in `f32` throughout
//! rather than LCMS's default 16-bit fixed-point path).
//!
//! Increments:
//! - **m4-89 (this module):** the ICC binary parser — profile header, tag table,
//!   and the `XYZ `/`curv`/`para` tag types — enough to reconstruct a
//!   matrix-shaper profile (RGB colorants → 3×3, per-channel TRC curves).
//! - m4-90: the cLUT tag types (`mft1`/`mft2` v2 LUTs, `mAB `/`mBA ` v4
//!   multi-process) + N-D LUT interpolation (tetrahedral for 3-in).
//! - m4-91: transform assembly (device→PCS→device, PCS Lab/XYZ, rendering
//!   intents, chromatic adaptation) and wiring colorin/colorout's LUT path.

mod clut;
mod parser;

pub use clut::Clut;
pub use parser::{Curve, IccError, Profile, Xyz};
