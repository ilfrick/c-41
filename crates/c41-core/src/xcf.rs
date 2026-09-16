//! Kernels ported from `src/imageio/format/xcf.c` (`write_image`, m4-218):
//! the 8-bit and 16-bit raster-mask channel packs. Each kernel replaces
//! its whole loop body: every single-channel float mask lane is clamped
//! to `[0, 1]`, scaled to the integer full scale, and rounded into the
//! XCF channel plane.
//!
//! What the C loops do, per pixel `i` (`npixels = width * height`):
//! - u8 (`d->bpp == 8`): `ch[i] = (uint8_t)roundf(CLIP(mask[i]) * 255.0f)`
//! - u16 (`d->bpp == 16`): `ch[i] = (uint16_t)roundf(CLIP(mask[i]) * 65535.0f)`
//! where `CLIP(x)` is darktable's `src/common/math.h` macro
//! `(((x) >= 0) ? ((x) <= 1 ? (x) : 1) : 0`. The `d->bpp == 32` branch
//! keeps the float plane by reference and has no loop; it stays in C.
//!
//! Bit-exactness notes:
//! - The clamp runs BEFORE the scale, in f32, exactly as parenthesised
//!   in C (`CLIP(mask) * 255`). This is the opposite order from the
//!   AVIF/HEIF export kernels (m4-213/m4-214/m4-216), whose C loops scale
//!   first and clamp second; the spelling is kept byte-identical to C
//!   regardless, so no helper is shared across the two spellings.
//! - The clamp is spelled branch-by-branch, low bound first exactly like
//!   the `CLIP` macro (`>= 0` then `<= 1`), instead of `f32::clamp`,
//!   which panics on NaN. A NaN lane takes the macro's else branch to
//!   `0.0` in C (`NaN >= 0` is false), and the same `>=` comparison is
//!   false in Rust, so NaN deterministically yields 0 in both languages
//!   with NO divergence — unlike the AVIF/HEIF kernels, where NaN falls
//!   through the clamp to `round` and the languages then disagree on the
//!   final cast. The NaN pin below locks this.
//! - `f32::round` is round-half-away-from-zero, matching `roundf`, so a
//!   `127.5` product lands on 128 (round-half-even would give 128 too
//!   there; the `0.5`-of-unit pin below distinguishes them at 1 vs 0).
//! - The rounded value always lies in `[0, 255]` / `[0, 65535]`, so the
//!   final `as u8` / `as u16` cast is exact in both languages.
//! - Buffers are tightly packed single-channel planes (the C caller
//!   mallocs exactly `npixels` lanes); there is no row stride and the
//!   alpha lane concept does not apply (the mask has one lane per pixel).
//!
//! The Rust kernels are single-threaded sequential; the C loops were
//! `DT_OMP_FOR_SIMD` over the pixel index, but each output lane reads
//! only its own source lane, so thread scheduling cannot change the
//! result.

/// Full scale of the 8-bit XCF mask pack (`255.0f32` in C).
pub const XCF_U8_MAX: f32 = 255.0;
/// Full scale of the 16-bit XCF mask pack (`65535.0f32` in C).
pub const XCF_U16_MAX: f32 = 65535.0;

/// Clamp one float lane exactly as darktable's `CLIP(x)` macro spells it
/// (`src/common/math.h`): `(x >= 0) ? ((x <= 1) ? x : 1) : 0`. Shared by
/// both kernels and both divergent references below so the four spellings
/// cannot drift, while still differing structurally. NaN takes the else
/// branch to `0.0` in both languages (see the module docs).
fn clip01(v: f32) -> f32 {
    if v >= 0.0 {
        if v <= 1.0 { v } else { 1.0 }
    } else {
        0.0
    }
}

/// Quantize one mask lane exactly as the u8 C loop spells it:
/// `(uint8_t)roundf(CLIP(v) * 255.0f)`. Shared by the kernel and its
/// divergent reference so the two cannot drift on the spelling while
/// still differing structurally.
fn mask_lane_u8(v: f32) -> u8 {
    (clip01(v) * XCF_U8_MAX).round() as u8
}

/// Quantize one mask lane exactly as the u16 C loop spells it:
/// `(uint16_t)roundf(CLIP(v) * 65535.0f)`. Shared by the kernel and its
/// divergent reference so the two cannot drift on the spelling while
/// still differing structurally.
fn mask_lane_u16(v: f32) -> u16 {
    (clip01(v) * XCF_U16_MAX).round() as u16
}

/// Pack a single-channel float raster mask into an 8-bit XCF channel.
///
/// Port of the former element-wise loop in the `d->bpp == 8` branch of
/// `write_image()` (`src/imageio/format/xcf.c`): per pixel `i`,
/// `out[i] = (uint8_t)roundf(CLIP(mask[i]) * 255.0f)`. `npixels` is the
/// C loop bound (`(size_t)width * height`); short buffers iterate
/// clamped (no panic, no out-of-bounds access). A zero `npixels` is a
/// no-op.
pub fn xcf_mask_to_u8(mask: &[f32], out: &mut [u8], npixels: usize) {
    let n = npixels.min(mask.len()).min(out.len());
    for i in 0..n {
        out[i] = mask_lane_u8(mask[i]);
    }
}

/// Structurally divergent reference for `xcf_mask_to_u8`: folds the two
/// planes through zipped iterators (the kernel indexes both sides with
/// `i`), so the sweep test cross-checks indexing as well as values. The
/// lane arithmetic is the shared `mask_lane_u8` helper by construction
/// (see the module docs: a scale-then-clamp respelling must not be
/// used). Same well-formed-buffers precondition, enforced here by the
/// same clamping.
#[cfg(test)]
fn ref_xcf_mask_to_u8(mask: &[f32], out: &mut [u8], npixels: usize) {
    let n = npixels.min(mask.len()).min(out.len());
    for (dst, &v) in out.iter_mut().zip(mask.iter()).take(n) {
        *dst = mask_lane_u8(v);
    }
}

/// Pack a single-channel float raster mask into a 16-bit XCF channel.
///
/// Port of the former element-wise loop in the `d->bpp == 16` branch of
/// `write_image()` (`src/imageio/format/xcf.c`): per pixel `i`,
/// `out[i] = (uint16_t)roundf(CLIP(mask[i]) * 65535.0f)`. `npixels` is
/// the C loop bound (`(size_t)width * height`); short buffers iterate
/// clamped (no panic, no out-of-bounds access). A zero `npixels` is a
/// no-op.
pub fn xcf_mask_to_u16(mask: &[f32], out: &mut [u16], npixels: usize) {
    let n = npixels.min(mask.len()).min(out.len());
    for i in 0..n {
        out[i] = mask_lane_u16(mask[i]);
    }
}

/// Structurally divergent reference for `xcf_mask_to_u16`: folds the two
/// planes through zipped iterators (the kernel indexes both sides with
/// `i`), so the sweep test cross-checks indexing as well as values. The
/// lane arithmetic is the shared `mask_lane_u16` helper by construction
/// (see the module docs: a scale-then-clamp respelling must not be
/// used). Same well-formed-buffers precondition, enforced here by the
/// same clamping.
#[cfg(test)]
fn ref_xcf_mask_to_u16(mask: &[f32], out: &mut [u16], npixels: usize) {
    let n = npixels.min(mask.len()).min(out.len());
    for (dst, &v) in out.iter_mut().zip(mask.iter()).take(n) {
        *dst = mask_lane_u16(v);
    }
}

/// # Safety
/// `mask` must hold at least `npixels` floats (the C caller passes the
/// raster mask from `dt_dev_get_raster_mask`, one lane per pixel) and
/// `out` at least `npixels` bytes (the C caller passes the malloc'd
/// `uint8_t` channel plane). The two buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn darkroom_xcf_mask_to_u8(
    mask: *const f32,
    out: *mut u8,
    npixels: usize,
) {
    if mask.is_null() || out.is_null() || npixels == 0 {
        return;
    }
    // Refuse lengths that cannot back a slice so the constructions below
    // stay inside the language model; the safe kernel re-checks
    // defensively via clamped iteration.
    if npixels > isize::MAX as usize {
        return;
    }
    let src = std::slice::from_raw_parts(mask, npixels);
    let dst = std::slice::from_raw_parts_mut(out, npixels);
    xcf_mask_to_u8(src, dst, npixels);
}

/// # Safety
/// `mask` must hold at least `npixels` floats (the C caller passes the
/// raster mask from `dt_dev_get_raster_mask`, one lane per pixel) and
/// `out` at least `npixels` lanes (the C caller passes the malloc'd
/// `uint16_t` channel plane). The two buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn darkroom_xcf_mask_to_u16(
    mask: *const f32,
    out: *mut u16,
    npixels: usize,
) {
    if mask.is_null() || out.is_null() || npixels == 0 {
        return;
    }
    // Refuse lengths that cannot back a slice so the constructions below
    // stay inside the language model; the safe kernel re-checks
    // defensively via clamped iteration.
    if npixels > isize::MAX as usize {
        return;
    }
    let src = std::slice::from_raw_parts(mask, npixels);
    let dst = std::slice::from_raw_parts_mut(out, npixels);
    xcf_mask_to_u16(src, dst, npixels);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Rails of both packs: 0.0 and -0.0 clamp to 0, 1.0 and above clamp
    // to the full scale, negatives clamp to 0, and 0.5 pins the
    // clamp-then-scale spelling (0.5 * 255 = 127.5 -> 128,
    // round-half-away; 0.5 * 65535 = 32767.5 -> 32768).
    #[test]
    fn rails_pin() {
        let mask = [0.0f32, -0.0, 0.5, 1.0, 2.0, -1.0];
        let mut out8 = vec![0xAAu8; 6];
        xcf_mask_to_u8(&mask, &mut out8, 6);
        assert_eq!(out8, vec![0, 0, 128, 255, 255, 0]);
        let mut out16 = vec![0xAAAAu16; 6];
        xcf_mask_to_u16(&mask, &mut out16, 6);
        assert_eq!(out16, vec![0, 0, 32768, 65535, 65535, 0]);
    }

    // NaN and infinities: CLIP runs first, so NaN takes the macro's else
    // branch to 0 (NaN >= 0 is false in both languages — no divergence,
    // unlike the scale-first AVIF/HEIF kernels), +inf clamps to 1 and
    // -inf clamps to 0.
    #[test]
    fn nan_and_inf_pin() {
        let mask = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let mut out8 = vec![0xAAu8; 3];
        xcf_mask_to_u8(&mask, &mut out8, 3);
        assert_eq!(out8, vec![0, 255, 0]);
        let mut out16 = vec![0xAAAAu16; 3];
        xcf_mask_to_u16(&mask, &mut out16, 3);
        assert_eq!(out16, vec![0, 65535, 0]);
    }

    // Round-half-away (roundf) spelling: 0.5/255 scales to exactly 0.5
    // and rounds to 1 (round-half-even would give 0); 2.5/255 scales to
    // exactly 2.5 and rounds to 3 (half-even would give 2). The u16 pins
    // are the same fractions of 65535.
    #[test]
    fn rounding_is_half_away() {
        let unit8 = 1.0f32 / 255.0;
        let mask = [0.5 * unit8, 2.5 * unit8, 0.0, 0.0];
        let mut out = vec![0xAAu8; 1];
        xcf_mask_to_u8(&mask, &mut out, 1);
        assert_eq!(out[0], 1);
        xcf_mask_to_u8(&mask[1..2], &mut out, 1);
        assert_eq!(out[0], 3);
        let unit16 = 1.0f32 / 65535.0;
        let mask16 = [0.5 * unit16, 2.5 * unit16];
        let mut out16 = vec![0xAAAAu16; 1];
        xcf_mask_to_u16(&mask16, &mut out16, 1);
        assert_eq!(out16[0], 1);
        xcf_mask_to_u16(&mask16[1..2], &mut out16, 1);
        assert_eq!(out16[0], 3);
    }

    // Kernel and reference must agree exactly over several shapes with
    // inputs spanning below 0 and above 1 (so both rails and the
    // rounding are exercised), plus embedded NaN and infinities.
    #[test]
    fn matches_reference_over_sweep() {
        for n in [1usize, 2, 7, 20, 64] {
            let mut mask = vec![0.0f32; n];
            for (i, v) in mask.iter_mut().enumerate() {
                // deterministic sweep across roughly [-0.5, 1.5]
                let k =
                    ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(0x9E37) % 2001) as f32;
                *v = k / 1000.0 - 0.5;
            }
            if n >= 3 {
                mask[0] = f32::NAN;
                mask[1] = f32::INFINITY;
                mask[2] = f32::NEG_INFINITY;
            }
            let (mut d8, mut r8) = (vec![0xAAu8; n], vec![0x55u8; n]);
            xcf_mask_to_u8(&mask, &mut d8, n);
            ref_xcf_mask_to_u8(&mask, &mut r8, n);
            assert_eq!(d8, r8, "u8 shape {n}");
            let (mut d16, mut r16) = (vec![0xAAAAu16; n], vec![0x5555u16; n]);
            xcf_mask_to_u16(&mask, &mut d16, n);
            ref_xcf_mask_to_u16(&mask, &mut r16, n);
            assert_eq!(d16, r16, "u16 shape {n}");
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let mask = vec![1.0f32; 8];
        // zero pixels: outputs untouched
        let mut out8 = vec![0xAAu8; 2];
        xcf_mask_to_u8(&mask, &mut out8, 0);
        assert_eq!(out8, vec![0xAA; 2]);
        let mut out16 = vec![0xAAAAu16; 2];
        xcf_mask_to_u16(&mask, &mut out16, 0);
        assert_eq!(out16, vec![0xAAAA; 2]);
        // empty source with live dims: no panic, no writes possible
        let empty_src: Vec<f32> = vec![];
        let mut out8 = vec![0xAAu8; 2];
        xcf_mask_to_u8(&empty_src, &mut out8, 2);
        assert_eq!(out8, vec![0xAA; 2]);
        let mut out16 = vec![0xAAAAu16; 2];
        xcf_mask_to_u16(&empty_src, &mut out16, 2);
        assert_eq!(out16, vec![0xAAAA; 2]);
        // empty output with live dims: no panic, nothing written anywhere
        let mut empty8: Vec<u8> = vec![];
        xcf_mask_to_u8(&[1.0; 2], &mut empty8, 2);
        assert!(empty8.is_empty());
        let mut empty16: Vec<u16> = vec![];
        xcf_mask_to_u16(&[1.0; 2], &mut empty16, 2);
        assert!(empty16.is_empty());
        // truncated source: written prefix converts fully, the unwritten
        // tail stays untouched
        let mut trunc8 = vec![0xAAu8; 4];
        xcf_mask_to_u8(&[1.0], &mut trunc8, 4);
        assert_eq!(trunc8, vec![255, 0xAA, 0xAA, 0xAA]);
        let mut trunc16 = vec![0xAAAAu16; 4];
        xcf_mask_to_u16(&[1.0], &mut trunc16, 4);
        assert_eq!(trunc16, vec![65535, 0xAAAA, 0xAAAA, 0xAAAA]);
        // truncated output: no panic, partial planes never partially written
        let mut narrow8 = vec![0xAAu8; 1];
        xcf_mask_to_u8(&[1.0, 1.0], &mut narrow8, 2);
        assert_eq!(narrow8, vec![255]);
        let mut narrow16 = vec![0xAAAAu16; 1];
        xcf_mask_to_u16(&[1.0, 1.0], &mut narrow16, 2);
        assert_eq!(narrow16, vec![65535]);
    }

    #[test]
    fn ffi_round_trip() {
        let n = 37usize;
        let mut mask = vec![0.0f32; n];
        for (i, v) in mask.iter_mut().enumerate() {
            let k = ((i as u64).wrapping_mul(2_654_435_761) % 2001) as f32;
            *v = k / 1000.0 - 0.5;
        }
        let (mut ffi8, mut direct8) = (vec![0u8; n], vec![0u8; n]);
        unsafe {
            darkroom_xcf_mask_to_u8(mask.as_ptr(), ffi8.as_mut_ptr(), n);
        }
        xcf_mask_to_u8(&mask, &mut direct8, n);
        assert_eq!(ffi8, direct8);
        let (mut ffi16, mut direct16) = (vec![0u16; n], vec![0u16; n]);
        unsafe {
            darkroom_xcf_mask_to_u16(mask.as_ptr(), ffi16.as_mut_ptr(), n);
        }
        xcf_mask_to_u16(&mask, &mut direct16, n);
        assert_eq!(ffi16, direct16);
    }

    #[test]
    fn ffi_guards() {
        let mask = vec![1.0f32; 4];
        let (mut out8, mut out16) = (vec![0xAAu8; 4], vec![0xAAAAu16; 4]);
        unsafe {
            // null pointers
            darkroom_xcf_mask_to_u8(std::ptr::null(), out8.as_mut_ptr(), 1);
            darkroom_xcf_mask_to_u8(mask.as_ptr(), std::ptr::null_mut(), 1);
            darkroom_xcf_mask_to_u16(std::ptr::null(), out16.as_mut_ptr(), 1);
            darkroom_xcf_mask_to_u16(mask.as_ptr(), std::ptr::null_mut(), 1);
            // zero pixels
            darkroom_xcf_mask_to_u8(mask.as_ptr(), out8.as_mut_ptr(), 0);
            darkroom_xcf_mask_to_u16(mask.as_ptr(), out16.as_mut_ptr(), 0);
            // unsliceable dims
            darkroom_xcf_mask_to_u8(mask.as_ptr(), out8.as_mut_ptr(), usize::MAX);
            darkroom_xcf_mask_to_u16(mask.as_ptr(), out16.as_mut_ptr(), usize::MAX);
        }
        assert_eq!(out8, vec![0xAA; 4]); // untouched
        assert_eq!(out16, vec![0xAAAA; 4]); // untouched
    }
}
