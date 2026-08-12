// Pre-generated FFI bindings for the Darkroom C core.
//
// To regenerate from the C headers (requires libclang):
//   LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu \
//   DARKROOM_GENERATE_BINDINGS=1 \
//   cargo build -p c41-sys
//
// Source: src/common/darktable.h lines 165-186

// ── Primary ID types ─────────────────────────────────────────────────────────
pub type dt_imgid_t   = i32;
pub type dt_filmid_t  = i32;
pub type dt_mask_id_t = i32;

pub const NO_IMGID:       dt_imgid_t   = 0;
pub const NO_FILMID:      dt_filmid_t  = 0;
pub const INVALID_MASKID: dt_mask_id_t = -1;
pub const NO_MASKID:      dt_mask_id_t = 0;

#[inline(always)]
pub fn dt_is_valid_imgid(n: dt_imgid_t) -> bool  { n > NO_IMGID }
#[inline(always)]
pub fn dt_is_valid_filmid(n: dt_filmid_t) -> bool { n > NO_FILMID }

// ── Image flags (src/common/image.h) ─────────────────────────────────────────
pub type dt_image_flags_t = u32;
pub const DT_IMAGE_MONOCHROME:      dt_image_flags_t = 0x0002;
pub const DT_IMAGE_HAS_MASKS:       dt_image_flags_t = 0x0004;
pub const DT_IMAGE_IS_VIRTUAL_COPY: dt_image_flags_t = 0x0200;
pub const DT_IMAGE_HAS_LOCALCOPY:   dt_image_flags_t = 0x0400;

// ── Opaque pointer to the global darktable_t state ───────────────────────────
// Only the pointer type is exposed here; the full struct lives in darktable.h.
// Rust code should never construct or move this value — use *mut darktable_t.
#[repr(C)]
pub struct darktable_t {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}
