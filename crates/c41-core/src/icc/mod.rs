//! Pure-Rust ICC colour-management engine — the replacement for darktable's
//! Little-CMS (`lcms2`) dependency in the input/output colour transforms.
//!
//! Goal: the same functionality as the LCMS path (parse arbitrary ICC v2/v4
//! profiles — matrix-shaper **and** cLUT/LUT profiles — and transform pixels
//! between them), at accuracy ≥ LCMS, entirely in Rust (no C link). Because the
//! shipped `c41-rs` product ships no LCMS, there is no bit-exact-to-LCMS
//! constraint — the engine only has to be *correct* (and we aim for more accurate
//! where LCMS takes shortcuts, e.g. we interpolate/evaluate in `f32` throughout
//! rather than LCMS's default 16-bit fixed-point path).
//!
//! Increments:
//! - **m4-89:** the ICC binary parser — profile header, tag table, and the
//!   `XYZ `/`curv`/`para` tag types — enough to reconstruct a matrix-shaper
//!   profile (RGB colorants → 3×3, per-channel TRC curves).
//! - m4-90: the N-D LUT interpolation core ([`clut::Clut`]: LCMS-matched
//!   tetrahedral for 3-in RGB + general N-linear).
//! - m4-91/m4-92: the cLUT tag types (`mft1`/`mft2` v2 LUTs, `mAB `/`mBA ` v4
//!   multi-process) parsed into [`Pipeline`]s of curve/matrix/CLUT stages.
//! - m4-93a/b: the device→PCS direction ([`Profile::a2b_pipeline`], with the
//!   appended PCS-decode stage normalising LUT output to raw values).
//! - m4-127: the PCS→device direction ([`Profile::b2a_pipeline`] — ICC-encode
//!   prepend, `B2A{intent}` preference, matrix-shaper fallback via the new
//!   [`Curve::inverse`]) and full transform assembly ([`transform::Transform`]:
//!   device→PCS→device across two profiles, Lab↔XYZ bridging, rendering intents
//!   incl. absolute's white-ratio scaling).
//! - **m4-129 (this increment):** the C boundary and the band-processing path:
//!   allocation-free 3-channel evaluation ([`Pipeline::eval_into3`] /
//!   [`Transform::eval_into`]) plus the `darkroom_icc_transform_*` FFI exports
//!   ([`ffi`]) that colorin/colorout's LUT path calls in place of LCMS.
//!   Replacing those C call sites is the follow-up (needs the full-app Docker
//!   C build).

mod clut;
mod ffi;
mod lut;
mod parser;
mod transform;

pub use clut::Clut;
pub use lut::{parse_lut_tag, parse_lut_v2, parse_lut_v4, Pipeline, Stage};
pub use parser::{Curve, IccError, Profile, Xyz};
pub use transform::Transform;
