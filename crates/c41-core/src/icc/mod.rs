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
//! - **m4-245 (this increment):** [`embed`] — lifting the ICC profile out of a
//!   JPEG/PNG/TIFF container, the input-profile half of G2. Pure: nothing calls
//!   it yet, and no decode path changes behaviour. It exists because neither
//!   decoder we use (gdk-pixbuf for preview, the `image` crate for export)
//!   exposes an ICC API, so the container has to be read for the profile
//!   ourselves. Applying what it finds is the next increment.
//! - **G2a2 (this increment):** [`standard`] — the sRGB destination profile and
//!   the buffer loops, plus [`standard::input_profile`], the decision an
//!   already-decoded image gets. This is the first module in `c41-core::icc`
//!   the *product* calls: a file's embedded profile now reaches the screen and
//!   the exported file instead of being ignored. Raw stays out (see
//!   [`standard`]'s docs), and a profile that is missing or unusable is
//!   assumed to be sRGB unless the user says otherwise: [`standard::
//!   InputAssumption`] carries the per-file choice the darkroom preview prompts
//!   for on the first open of an untagged container, and export resolves the
//!   same remembered answer.
//! - **G2b (this increment):** the output side — [`standard::srgb_profile`]
//!   now also carries the `desc`/`cprt`/`chad` tags a conformant display
//!   profile needs, and the UI embeds those bytes in every exported
//!   JPEG/PNG/TIFF (and the neural-restore TIFF), so what lands on disk is
//!   tagged with the sRGB space its pixels were rendered in. Before this, an
//!   export was untagged: colour-managed viewers guessed (usually wrong for
//!   anything that had been transformed).

mod clut;
mod embed;
mod ffi;
mod lut;
mod parser;
mod standard;
mod transform;

pub use clut::Clut;
pub use embed::{extract as extract_embedded, extract_file as extract_embedded_file, EmbedError, Embedded};
pub use lut::{parse_lut_tag, parse_lut_v2, parse_lut_v4, Pipeline, Stage};
pub use parser::{Curve, IccError, Profile, Xyz};
pub use standard::{
    apply_rgb8, apply_rgb16, adobe_rgb1998_profile, input_profile, srgb_profile, InputAssumption,
    InputProfile, INPUT_INTENT,
};
pub use transform::Transform;
