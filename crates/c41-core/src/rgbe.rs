//! Kernel ported from `src/imageio/imageio_rgbe.c` (`dt_imageio_open_rgbe`,
//! the clamp-and-pack loop, m4-201). One kernel replaces the whole loop body:
//! each decoded Radiance RGB pixel is clamped to `[0, 10000]` and packed
//! into the 4-channel float mipmap buffer with alpha 0.0.
//!
//! What the C loop does, per pixel `i` over `npixels`:
//! - `pix[c] = fmaxf(0.0f, fminf(10000.0f, rgbe_buf[3*i + c]))` for `c` in
//!   `0..3`, into a zero-initialised `dt_aligned_pixel_t pix`.
//! - `copy_pixel_nontemporal(&mipbuf[4*i], pix)` — a 4-wide streaming store,
//!   value-identical to four plain assignments, so `mipbuf[4*i + 3]` keeps
//!   the zero-initialised slot: alpha is exactly `+0.0`.
//!
//! Bit-exactness notes:
//! - The repo-wide Release flags are `-O3 -ffast-math -fno-finite-math-only`
//!   (same note as the m4-175 PFM port): the trailing `-fno-finite-math-only`
//!   keeps IEEE NaN handling, so `fminf`/`fmaxf` have minNum/maxNum semantics
//!   (a NaN operand yields the other operand). Hence NaN maps to 10000.0:
//!   `fminf(10000, NaN)` is 10000, then `fmaxf(0, 10000)` is 10000. The Rust
//!   `clamp_channel` pins this explicitly with an `is_nan` arm rather than
//!   relying on any one toolchain's `min`/`max` NaN convention, so the
//!   mapping holds on every build.
//! - Signed zero: `-0.0` maps to `+0.0`, matching `fmaxf(+0.0, -0.0)` which
//!   returns `+0.0` per IEEE 754. Pinned by test via exact `to_bits()`.
//! - Infinities: `+inf` maps to 10000.0, `-inf` maps to 0.0 — the "repair
//!   nan/inf etc" the C comment advertises.
//! - Finite values inside `(0, 10000)` pass through untouched (no rounding:
//!   comparisons only, the value is copied, never recomputed).
//! - Buffers: the C caller allocates two distinct buffers (`rgbe_buf` at
//!   `3*npixels` via `dt_alloc_align_float`, `mipbuf` at `4*npixels` from the
//!   mipmap cache). The kernel must not be called with overlapping buffers:
//!   output index `4*i + c` runs ahead of source index `3*i + c` for `i > 0`,
//!   so an in-place call would corrupt the source before it is read.
//!
//! The Rust kernel is single-threaded sequential; the C loop was
//! `DT_OMP_FOR` over pixels, but the per-pixel work is independent (each
//! output quad reads only its own source triple), so thread scheduling
//! cannot change the result.

/// Clamp one decoded Radiance RGBE channel value into output range.
///
/// Port of `fmaxf(0.0f, fminf(10000.0f, v))` with the C minNum/maxNum NaN
/// semantics pinned explicitly: NaN yields 10000.0 (see the module docs),
/// `-0.0` yields `+0.0`, values outside `[0, 10000]` saturate, finite values
/// inside pass through bit-identically.
#[inline]
fn clamp_channel(v: f32) -> f32 {
    if v.is_nan() {
        10000.0
    } else if v <= 0.0 {
        0.0
    } else if v >= 10000.0 {
        10000.0
    } else {
        v
    }
}

/// Clamp decoded RGBE floats and pack RGB into the RGBA mipmap buffer.
///
/// Port of the former element-wise loop in `dt_imageio_open_rgbe`
/// (src/imageio/imageio_rgbe.c): `src` holds `3*npixels` floats (RGB),
/// `out` receives `4*npixels` floats (RGBA, alpha `+0.0`). See the module
/// docs for the clamp mapping, the alpha-zero fidelity point, and the
/// no-aliasing contract.
///
/// Degenerate `npixels == 0` is a no-op; short buffers are handled by
/// clamped iteration (no panic, no out-of-bounds access). For the
/// well-formed buffers the C caller allocates the clamps never engage and
/// the behaviour is exactly the C loop's.
pub fn rgbe_clamp_pack(src: &[f32], out: &mut [f32], npixels: usize) {
    let n = npixels.min(src.len() / 3).min(out.len() / 4);
    for i in 0..n {
        let s = 3 * i;
        let d = 4 * i;
        out[d] = clamp_channel(src[s]);
        out[d + 1] = clamp_channel(src[s + 1]);
        out[d + 2] = clamp_channel(src[s + 2]);
        out[d + 3] = 0.0;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `rgbe_clamp_pack`: walks the OUTPUT
/// buffer quad-by-quad via `chunks_exact_mut` (deriving each source triple
/// from the output position) with the clamp written as a nested
/// `!(v > 0.0)` / `v < 10000.0` decision tree instead of the kernel's
/// `is_nan`-first `if`/`else if` chain, where the kernel walks source
/// triples with explicit index writes. Same mapping for every input class
/// (NaN to 10000.0, signed zero to +0.0, infinities saturated, finite
/// values bit-identical). Same clamped-fit precondition: well-formed
/// buffers only.
#[cfg(test)]
fn ref_rgbe_clamp_pack(src: &[f32], out: &mut [f32], npixels: usize) {
    let Some(need_src) = npixels.checked_mul(3) else {
        return;
    };
    let Some(need_out) = npixels.checked_mul(4) else {
        return;
    };
    if src.len() < need_src || out.len() < need_out {
        return;
    }
    for (i, quad) in out.chunks_exact_mut(4).enumerate().take(npixels) {
        let s = 3 * i;
        for c in 0..3 {
            let v = src[s + c];
            quad[c] = if !(v > 0.0) {
                if v.is_nan() {
                    10000.0
                } else {
                    0.0
                }
            } else if v < 10000.0 {
                v
            } else {
                10000.0
            };
        }
        quad[3] = 0.0;
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `rgbe_buf` must hold at least `3 * npixels` floats (the C caller
/// allocates exactly that via `dt_alloc_align_float`) and `mipbuf` at least
/// `4 * npixels` floats (the mipmap-cache allocation for the 4-channel
/// float image). The two buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn darkroom_rgbe_clamp_pack(
    rgbe_buf: *const f32,
    mipbuf: *mut f32,
    npixels: usize,
) {
    if rgbe_buf.is_null() || mipbuf.is_null() || npixels == 0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap the lengths; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(src_len) = npixels.checked_mul(3) else {
        return;
    };
    let Some(out_len) = npixels.checked_mul(4) else {
        return;
    };
    let src = std::slice::from_raw_parts(rgbe_buf, src_len);
    let out = std::slice::from_raw_parts_mut(mipbuf, out_len);
    rgbe_clamp_pack(src, out, npixels);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // one pixel per row: (input triple, expected output quad) as bit
    // patterns, so signed zeros and NaN mappings are pinned exactly.
    #[test]
    fn clamp_pin() {
        let src = [
            -5.0f32, -0.0, 0.0, // -> +0.0, +0.0, +0.0
            2.5, 9999.99, 10000.0, // -> identity, identity, 10000.0
            1.0e9, f32::INFINITY, f32::NEG_INFINITY, // -> 10000, 10000, 0
            f32::from_bits(0x7FC0_0001), 3.25, -3.25, // NaN -> 10000
        ];
        let mut out = vec![7.0f32; 4 * 4];
        rgbe_clamp_pack(&src, &mut out, 4);
        let nan_bits = 10000.0f32.to_bits();
        assert_eq!(nan_bits, 0x461C_4000);
        let z = 0.0f32.to_bits();
        let expect = [
            z,
            z,
            z,
            z,
            2.5f32.to_bits(),
            9999.99f32.to_bits(),
            10000.0f32.to_bits(),
            z,
            10000.0f32.to_bits(),
            10000.0f32.to_bits(),
            z,
            z,
            nan_bits,
            3.25f32.to_bits(),
            z,
            z,
        ];
        for (o, e) in out.iter().zip(expect) {
            assert_eq!(o.to_bits(), e);
        }
        // and the negative-zero input really did normalise: +0.0, not -0.0
        assert_eq!(out[1].to_bits(), 0x0000_0000);
    }

    // deterministic sweep covering negatives, the saturation rails and
    // denormals, with infinities, signed zeros and NaNs injected: kernel
    // and reference must agree bit-exactly on every quad.
    #[test]
    fn matches_reference_over_sweep() {
        let n = 513usize;
        let mut src = Vec::with_capacity(3 * n);
        for i in 0..3 * n {
            // LCG-mapped pseudo-random value in [-10000.5, 10000.5): hits
            // below zero, inside the rails, and above 10000.
            let bits = (i as u64)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(0x9E37) as f64;
            src.push((bits % 2_000_100.0) as f32 / 100.0 - 10000.5);
        }
        src[0] = f32::INFINITY;
        src[1] = f32::NEG_INFINITY;
        src[2] = -0.0;
        src[3] = f32::from_bits(0x7FC1_2345);
        src[4] = f32::from_bits(0xFFC0_0000); // negative quiet NaN
        src[5] = 5.0e-45; // smallest positive denormal, passes through
        let mut direct = vec![0.0f32; 4 * n];
        let mut reference = vec![0.0f32; 4 * n];
        rgbe_clamp_pack(&src, &mut direct, n);
        ref_rgbe_clamp_pack(&src, &mut reference, n);
        assert_eq!(direct.len(), reference.len());
        for (k, (d, r)) in direct.iter().zip(reference.iter()).enumerate() {
            assert_eq!(d.to_bits(), r.to_bits(), "quad slot {k}");
        }
        // spot pins on the sweep output itself (out[4*i + c] maps to
        // src[3*i + c]): infinities saturated, signed zero normalised,
        // both NaN signs saturated, the denormal survived untouched, and
        // every alpha is +0.0
        assert_eq!(direct[0].to_bits(), 10000.0f32.to_bits()); // +inf
        assert_eq!(direct[1].to_bits(), 0.0f32.to_bits()); // -inf -> 0.0
        assert_eq!(direct[2].to_bits(), 0x0000_0000); // -0.0 -> +0.0
        assert_eq!(direct[4].to_bits(), 10000.0f32.to_bits()); // NaN
        assert_eq!(direct[5].to_bits(), 10000.0f32.to_bits()); // -NaN
        assert_eq!(direct[6].to_bits(), 5.0e-45f32.to_bits()); // denormal
        for (k, v) in direct.iter().enumerate() {
            if k % 4 == 3 {
                assert_eq!(v.to_bits(), 0x0000_0000, "alpha slot {k}");
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let src = vec![1.0f32; 12];
        // zero pixels: output untouched
        let mut out = vec![9.0f32; 16];
        rgbe_clamp_pack(&src, &mut out, 0);
        assert_eq!(out, vec![9.0f32; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_out = vec![0.0f32; 7]; // short for 2 quads
        rgbe_clamp_pack(&src, &mut short_out, 2);
        let short_src = vec![2.0f32; 5]; // short for 2 triples
        let mut out2 = vec![0.0f32; 8];
        rgbe_clamp_pack(&short_src, &mut out2, 2);
        let empty: Vec<f32> = vec![];
        let mut out3 = vec![0.0f32; 4];
        rgbe_clamp_pack(&empty, &mut out3, 1);
        assert_eq!(out3, vec![0.0f32; 4]);
    }

    #[test]
    fn ffi_round_trip() {
        let n = 65usize;
        let mut src = vec![0.0f32; 3 * n];
        for (i, v) in src.iter_mut().enumerate() {
            let bits = (i as u64).wrapping_mul(2_654_435_761) as f64;
            *v = (bits % 3_000_000.0) as f32 / 100.0 - 10000.0;
        }
        src[0] = f32::from_bits(0x7FC0_0000);
        let mut ffi_out = vec![0.0f32; 4 * n];
        let mut direct_out = vec![0.0f32; 4 * n];
        unsafe {
            darkroom_rgbe_clamp_pack(src.as_ptr(), ffi_out.as_mut_ptr(), n);
        }
        rgbe_clamp_pack(&src, &mut direct_out, n);
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn ffi_guards() {
        let src = vec![1.0f32; 12];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_rgbe_clamp_pack(std::ptr::null(), out.as_mut_ptr(), 4);
            darkroom_rgbe_clamp_pack(src.as_ptr(), std::ptr::null_mut(), 4);
            // zero pixels
            darkroom_rgbe_clamp_pack(src.as_ptr(), out.as_mut_ptr(), 0);
            // overflowing pixel count (3 * npixels wraps on 64-bit only
            // past usize::MAX / 3; 4 * npixels wraps sooner — both reject)
            darkroom_rgbe_clamp_pack(src.as_ptr(), out.as_mut_ptr(), usize::MAX);
            darkroom_rgbe_clamp_pack(src.as_ptr(), out.as_mut_ptr(), usize::MAX / 3 + 1);
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }
}
