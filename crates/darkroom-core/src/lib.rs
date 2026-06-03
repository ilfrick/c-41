//! Image processing pipeline — Phase 1 target.
//!
//! Defines `IopProcess`, the trait every IOP module must implement.
//! Phase 0: trait + types defined. Phase 1: one module per src/iop/*.c file.

pub mod color;
pub mod error;
pub mod interp;
pub mod iop;
pub mod math;
pub mod params;
pub mod raw;
pub mod roi;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub use darkroom_sys::{dt_imgid_t, dt_is_valid_imgid, NO_IMGID};
