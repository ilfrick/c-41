//! Database and collections layer — Phase 2 target.
//!
//! Will replace src/common/{collection,image,tags,history,metadata,film}.c.
//! Phase 2-db-1: tags module implemented; FFI trampolines in ffi.rs.

pub mod collection;
pub mod colorlabels;
pub mod film;
pub mod history;
pub mod image;
pub mod metadata;
pub mod schema;
pub mod tags;
pub mod ffi;

pub use c41_sys::dt_imgid_t;
