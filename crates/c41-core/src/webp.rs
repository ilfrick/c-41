//! Kernel ported from `src/imageio/imageio_webp.c` (`dt_imageio_open_webp`,
//! the u8-to-float normalize loop, m4-207). The kernel replaces the whole
//! loop body: each decoded WebP byte is scaled by `1/255` into the
//! 4-channel float mipmap buffer, with the alpha lane zeroed.
//!
//! What the C loop does, per pixel `i` in `0..npixels`:
//! - `mipbuf[4*i + c] = int_RGBA_buffer[4*i + c] / 255.f` for `c` in `0..2`
//! - `mipbuf[4*i + 3] = 0.0` (the C loop copies a zero-initialised pixel,
//!   so only the three RGB lanes are ever stored; an opaque `255` alpha
//!   byte becomes exactly `0.0`, not `1.0`).
//!
//! This differs from both neighbours on purpose: QOI (m4-202) scales all
//! four lanes including alpha, while PNG (m4-203/m4-204) preserves the
//! alpha lane as allocated. The WebP kernel must zero it.
//!
//! Bit-exactness notes:
//! - `u8 as f32` is exact for all 256 inputs, and both sides divide in
//!   `f32` with the same rounding, so `byte / 255.f` is bit-identical in C
//!   and Rust. The kernel must stay a division: `byte * (1.0 / 255.0)`
//!   can round differently (the reciprocal is itself inexact), so the
//!   multiply form is NOT an acceptable substitute. Pinned by test via
//!   exact `to_bits()` on the rails (`0 -> +0.0`, `255 -> 1.0`).
//! - All inputs are finite and non-negative, so there are no NaN, infinity,
//!   or signed-zero cases to pin: `0 / 255` is `+0.0` on both sides.
//! - Buffers: the C caller allocates two distinct buffers
//!   (`int_RGBA_buffer` at `4*npixels` bytes from `WebPDecodeRGBAInto`,
//!   `mipbuf` at `4*npixels` floats from the mipmap cache). The kernel
//!   must not be called with overlapping buffers.
//!
//! The Rust kernel is single-threaded sequential; the C loop was
//! `DT_OMP_FOR` over pixels, but each output quad reads only its own
//! source quad, so thread scheduling cannot change the result.

/// Scale decoded WebP bytes into the float mipmap buffer.
///
/// Port of the former element-wise loop in `dt_imageio_open_webp`
/// (src/imageio/imageio_webp.c): `src` holds `4*npixels` bytes (RGBA from
/// `WebPDecodeRGBAInto`), `out` receives `4*npixels` floats (RGB lanes
/// divided by 255, alpha lane zeroed). See the module docs for the
/// division-vs-multiply fidelity point, the alpha-zero contract, and the
/// no-aliasing contract.
///
/// Degenerate `npixels == 0` is a no-op; short buffers are handled by
/// clamped iteration (no panic, no out-of-bounds access). For the
/// well-formed buffers the C caller allocates the clamps never engage and
/// the behaviour is exactly the C loop's.
pub fn webp_u8_to_float(src: &[u8], out: &mut [f32], npixels: usize) {
    let n = npixels.min(src.len() / 4).min(out.len() / 4);
    for i in 0..n {
        let s = 4 * i;
        let d = 4 * i;
        out[d] = src[s] as f32 / 255.0f32;
        out[d + 1] = src[s + 1] as f32 / 255.0f32;
        out[d + 2] = src[s + 2] as f32 / 255.0f32;
        out[d + 3] = 0.0f32;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `webp_u8_to_float`: zips source
/// quads with output quads via `chunks_exact` iterators and a lane loop
/// (the kernel walks both buffers with explicit stride arithmetic), so the
/// sweep test cross-checks indexing as well as values. The arithmetic is
/// the same `byte as f32 / 255.0` division by construction (see the module
/// docs: the reciprocal-multiply form is not bit-identical and must not be
/// used), and lane 3 is assigned the literal `0.0`. Same
/// well-formed-buffers precondition, enforced here by early return rather
/// than clamping.
#[cfg(test)]
fn ref_webp_u8_to_float(src: &[u8], out: &mut [f32], npixels: usize) {
    let Some(src_need) = npixels.checked_mul(4) else {
        return;
    };
    let Some(out_need) = npixels.checked_mul(4) else {
        return;
    };
    if src.len() < src_need || out.len() < out_need {
        return;
    }
    for (quad_in, quad_out) in src
        .chunks_exact(4)
        .zip(out.chunks_exact_mut(4))
        .take(npixels)
    {
        for lane in 0..3 {
            quad_out[lane] = quad_in[lane] as f32 / 255.0f32;
        }
        quad_out[3] = 0.0f32;
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `rgba_buf` must hold at least `4 * npixels` bytes (the C caller passes
/// the `WebPDecodeRGBAInto` output for a `width * height` image) and
/// `mipbuf` at least `4 * npixels` floats (the mipmap-cache allocation for
/// the 4-channel float image). The two buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn darkroom_webp_u8_to_float(
    rgba_buf: *const u8,
    mipbuf: *mut f32,
    npixels: usize,
) {
    if rgba_buf.is_null() || mipbuf.is_null() || npixels == 0 {
        return;
    }
    // validate the product BEFORE building the slices below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(len) = npixels.checked_mul(4) else {
        return;
    };
    let src = std::slice::from_raw_parts(rgba_buf, len);
    let out = std::slice::from_raw_parts_mut(mipbuf, len);
    webp_u8_to_float(src, out, npixels);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // rails pinned as exact bit patterns: 0 scales to +0.0, 255 to 1.0.
    // Both are forced by IEEE arithmetic (not by the implementation), so
    // they pin the kernel to the C operation rather than to itself. Alpha
    // is pinned to +0.0 even for an opaque 255 input byte: the C loop
    // never stores the source alpha, and this is the deliberate
    // difference from the QOI kernel (which maps 255 alpha to 1.0).
    #[test]
    fn rails_pin() {
        let src = [0u8, 1, 128, 255, 255, 0, 254, 0];
        let mut out = vec![7.0f32; 8];
        webp_u8_to_float(&src, &mut out, 2);
        assert_eq!(out[0].to_bits(), 0x0000_0000); // 0 -> +0.0
        assert_eq!(out[4].to_bits(), 0x3F80_0000); // 255 -> 1.0
        // interior values follow the same division the C loop performs
        assert_eq!(out[1].to_bits(), (1u8 as f32 / 255.0).to_bits());
        assert_eq!(out[2].to_bits(), (128u8 as f32 / 255.0).to_bits());
        assert_eq!(out[5].to_bits(), (0u8 as f32 / 255.0).to_bits());
        assert_eq!(out[6].to_bits(), (254u8 as f32 / 255.0).to_bits());
        // alpha lanes are zeroed, never copied from the source
        assert_eq!(out[3].to_bits(), 0x0000_0000);
        assert_eq!(out[7].to_bits(), 0x0000_0000);
    }

    // every byte value at several pixel counts, plus opaque alphas in the
    // stream: kernel and reference must agree bit-exactly on every lane,
    // and lane 3 must read +0.0 everywhere.
    #[test]
    fn matches_reference_over_sweep() {
        for n in [1usize, 3, 65, 513] {
            let mut src = Vec::with_capacity(4 * n);
            for i in 0..4 * n {
                // LCG over the full 0..=255 byte range
                let v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37) % 256;
                src.push(v as u8);
            }
            src[0] = 0;
            src[1] = 255;
            // force some opaque alpha bytes: they must still decode to 0.0
            for k in 0..n {
                src[4 * k + 3] = 255;
            }
            let mut direct = vec![-1.0f32; 4 * n];
            let mut reference = vec![-1.0f32; 4 * n];
            webp_u8_to_float(&src, &mut direct, n);
            ref_webp_u8_to_float(&src, &mut reference, n);
            assert_eq!(direct.len(), reference.len());
            for (k, (d, r)) in direct.iter().zip(reference.iter()).enumerate() {
                assert_eq!(d.to_bits(), r.to_bits(), "lane {k}");
                if k % 4 == 3 {
                    assert_eq!(*d, 0.0f32, "alpha lane {k} zeroed");
                }
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let src = vec![200u8; 16];
        // zero pixels: output untouched
        let mut out = vec![9.0f32; 16];
        webp_u8_to_float(&src, &mut out, 0);
        assert_eq!(out, vec![9.0f32; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path. The first
        // quad still converts (n clamps to 1); the tail stays untouched.
        let mut short_out = vec![0.0f32; 7]; // short for 2 quads
        webp_u8_to_float(&src, &mut short_out, 2);
        let lane = 200u8 as f32 / 255.0f32;
        assert_eq!(short_out[0].to_bits(), lane.to_bits());
        assert_eq!(short_out[1].to_bits(), lane.to_bits());
        assert_eq!(short_out[2].to_bits(), lane.to_bits());
        assert_eq!(short_out[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(&short_out[4..7], &[0.0f32; 3]);
        let short_src = vec![200u8; 7]; // short for 2 quads
        let mut out2 = vec![0.0f32; 8];
        webp_u8_to_float(&short_src, &mut out2, 2);
        assert_eq!(out2[0].to_bits(), lane.to_bits());
        assert_eq!(out2[1].to_bits(), lane.to_bits());
        assert_eq!(out2[2].to_bits(), lane.to_bits());
        assert_eq!(out2[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(&out2[4..8], &[0.0f32; 4]);
        let empty: Vec<u8> = vec![];
        let mut out3 = vec![0.0f32; 4];
        webp_u8_to_float(&empty, &mut out3, 1);
        assert_eq!(out3, vec![0.0f32; 4]);
    }

    #[test]
    fn ffi_round_trip() {
        let n = 65usize;
        let mut src = vec![0u8; 4 * n];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i as u64).wrapping_mul(2_654_435_761) % 256) as u8;
        }
        let mut ffi_out = vec![-2.0f32; 4 * n];
        let mut direct_out = vec![-2.0f32; 4 * n];
        unsafe {
            darkroom_webp_u8_to_float(src.as_ptr(), ffi_out.as_mut_ptr(), n);
        }
        webp_u8_to_float(&src, &mut direct_out, n);
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn ffi_guards() {
        let src = vec![200u8; 16];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_webp_u8_to_float(std::ptr::null(), out.as_mut_ptr(), 4);
            darkroom_webp_u8_to_float(src.as_ptr(), std::ptr::null_mut(), 4);
            // zero pixels
            darkroom_webp_u8_to_float(src.as_ptr(), out.as_mut_ptr(), 0);
            // overflowing pixel count (4 * npixels wraps — both reject)
            darkroom_webp_u8_to_float(src.as_ptr(), out.as_mut_ptr(), usize::MAX);
            darkroom_webp_u8_to_float(src.as_ptr(), out.as_mut_ptr(), usize::MAX / 4 + 1);
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }
}
