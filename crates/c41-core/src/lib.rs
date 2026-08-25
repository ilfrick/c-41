//! Image processing pipeline — Phase 1 target.
//!
//! Defines `IopProcess`, the trait every IOP module must implement.
//! Phase 0: trait + types defined. Phase 1: one module per src/iop/*.c file.

pub mod bilateral;
pub mod color;
pub mod curve_tools;
pub mod colorreconstruct;
pub mod dwt;
pub mod error;
pub mod exif;
pub mod gaussian;
pub mod geometry;
pub mod icc;
pub mod interp;
pub mod iop;
pub mod math;
pub mod params;
pub mod pipeline;
pub mod raw;
pub mod rawimage;
pub mod roi;
pub mod splines;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub use c41_sys::{dt_imgid_t, dt_is_valid_imgid, NO_IMGID};
