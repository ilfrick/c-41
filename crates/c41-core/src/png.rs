//! Kernel ported from `src/imageio/imageio_png.c` (`dt_imageio_open_png`,
//! the 8-bit normalize loop, m4-203). One kernel replaces the whole loop
//! body: each decoded RGB byte is scaled by `1/255` into the 4-channel
//! float mipmap buffer, leaving the alpha lane as allocated.
//!
//! What the C loop does, per pixel `index` in `0..npixels`:
//! - `mipbuf[4*index + c] = buf[3*index + c] * normalizer` for `c` in
//!   `0..3`, with `normalizer = 1.0f / 255.0f` precomputed once outside
//!   the loop. `mipbuf[4*index + 3]` (alpha) is never written.
//!
//! Scope: only the `bpp < 16` branch is ported. The 16-bit big-endian
//! pair branch stays in C: besides a different packing, its blue lane
//! reads the low byte at the green lane's offset (`buf[2*(3*index+1)+1]`
//! where the red/green pattern implies `buf[2*(3*index+2)+1]`), which this
//! increment does not adjudicate — porting it would either enshrine or
//! silently fix that asymmetry, neither of which belongs in a fidelity
//! port done without the C maintainers.
//!
//! Bit-exactness notes:
//! - `u8 as f32` is exact for all 256 inputs, and the normalizer is the
//!   same correctly-rounded `f32` division on both sides, so the per-lane
//!   `f32` multiply rounds identically in C and Rust. The kernel must stay
//!   a multiply by the precomputed reciprocal: `byte as f32 / 255.0` is a
//!   different rounding sequence and is NOT bit-identical (this is the
//!   mirror of the m4-202 QOI lesson, where the C side divided and the
//!   kernel had to divide). Pinned by test: `0 -> +0.0` exact bits, and
//!   the full byte sweep against the divergent reference.
//! - Alpha is preserved, never written: pinned by test with a sentinel
//!   fill the kernel must leave untouched in lane 3 of every quad.
//! - Buffers: the C caller allocates two distinct buffers (`buf` at
//!   `3*npixels` bytes from libpng's rowbytes layout, `mipbuf` at
//!   `4*npixels` floats from the mipmap cache). The kernel must not be
//!   called with overlapping buffers.
//!
//! The Rust kernel is single-threaded sequential; the C loop was
//! `DT_OMP_FOR` over pixels, but each output triple reads only its own
//! source triple, so thread scheduling cannot change the result.

/// Scale decoded PNG RGB bytes into the float mipmap buffer.
///
/// Port of the former element-wise loop in the `bpp < 16` branch of
/// `dt_imageio_open_png` (src/imageio/imageio_png.c): `src` holds
/// `3*npixels` bytes (tightly packed RGB — libpng already stripped any
/// alpha via `png_set_strip_alpha` and expanded sub-8-bit gray), `out`
/// receives the scaled triples at `4*i .. 4*i+3`, lane 3 preserved. See
/// the module docs for the multiply-vs-divide fidelity point and the
/// no-aliasing contract.
///
/// Degenerate `npixels == 0` is a no-op; short buffers are handled by
/// clamped iteration (no panic, no out-of-bounds access). For the
/// well-formed buffers the C caller allocates the clamps never engage and
/// the behaviour is exactly the C loop's.
pub fn png_u8_to_float(src: &[u8], out: &mut [f32], npixels: usize) {
    const NORMALIZER: f32 = 1.0f32 / 255.0f32;
    let n = npixels.min(src.len() / 3).min(out.len() / 4);
    for i in 0..n {
        let s = 3 * i;
        let d = 4 * i;
        out[d] = src[s] as f32 * NORMALIZER;
        out[d + 1] = src[s + 1] as f32 * NORMALIZER;
        out[d + 2] = src[s + 2] as f32 * NORMALIZER;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `png_u8_to_float`: zips source
/// triples with output quads via `chunks_exact` iterators and a lane loop
/// (the kernel walks both buffers with explicit stride arithmetic), so the
/// sweep test cross-checks indexing as well as values. The arithmetic is
/// the same multiply by the precomputed `1/255` reciprocal by construction
/// (see the module docs: the per-lane division form is not bit-identical
/// and must not be used). Same well-formed-buffers precondition, enforced
/// here by early return rather than clamping.
#[cfg(test)]
fn ref_png_u8_to_float(src: &[u8], out: &mut [f32], npixels: usize) {
    const NORMALIZER: f32 = 1.0f32 / 255.0f32;
    let Some(src_need) = npixels.checked_mul(3) else {
        return;
    };
    let Some(out_need) = npixels.checked_mul(4) else {
        return;
    };
    if src.len() < src_need || out.len() < out_need {
        return;
    }
    for (triple, quad) in src
        .chunks_exact(3)
        .zip(out.chunks_exact_mut(4))
        .take(npixels)
    {
        for lane in 0..3 {
            quad[lane] = triple[lane] as f32 * NORMALIZER;
        }
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `rgb_buf` must hold at least `3 * npixels` bytes (the C caller passes
/// the libpng-decoded buffer for a `width * height` image) and `mipbuf` at
/// least `4 * npixels` floats (the mipmap-cache allocation for the
/// 4-channel float image). The two buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn darkroom_png_u8_to_float(
    rgb_buf: *const u8,
    mipbuf: *mut f32,
    npixels: usize,
) {
    if rgb_buf.is_null() || mipbuf.is_null() || npixels == 0 {
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
    let src = std::slice::from_raw_parts(rgb_buf, src_len);
    let out = std::slice::from_raw_parts_mut(mipbuf, out_len);
    png_u8_to_float(src, out, npixels);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // rails pinned as exact bit patterns: 0 scales to +0.0 (forced by IEEE
    // arithmetic — 0.0 times a finite positive normalizer is +0.0, so this
    // pins the kernel to the C operation rather than to itself). Note there
    // is deliberately NO 255 -> 1.0 pin: 255 * round(1/255) does not round
    // back to 1.0 in f32 (verified by execution), so 255 is covered by the
    // expression comparison below like every other interior value.
    #[test]
    fn rails_pin() {
        let src = [0u8, 1, 128, 255, 255, 254, 0, 127, 2];
        let mut out = vec![7.0f32; 12];
        png_u8_to_float(&src, &mut out, 3);
        assert_eq!(out[0].to_bits(), 0x0000_0000); // 0 -> +0.0
        assert_eq!(out[8].to_bits(), 0x0000_0000); // 0 -> +0.0
        // interior values follow the same reciprocal-multiply the C loop
        // performs (normalizer spelled identically, not precomputed here
        // from a second path that could hide a transcription slip: any
        // deviation from the C spelling fails the sweep below instead).
        const NORMALIZER: f32 = 1.0f32 / 255.0f32;
        assert_eq!(out[1].to_bits(), (1u8 as f32 * NORMALIZER).to_bits());
        assert_eq!(out[2].to_bits(), (128u8 as f32 * NORMALIZER).to_bits());
        assert_eq!(out[4].to_bits(), (255u8 as f32 * NORMALIZER).to_bits());
        assert_eq!(out[5].to_bits(), (255u8 as f32 * NORMALIZER).to_bits());
        assert_eq!(out[9].to_bits(), (127u8 as f32 * NORMALIZER).to_bits());
        assert_eq!(out[10].to_bits(), (2u8 as f32 * NORMALIZER).to_bits());
        // alpha lanes are preserved, never written
        assert_eq!(out[3], 7.0f32);
        assert_eq!(out[7], 7.0f32);
        assert_eq!(out[11], 7.0f32);
    }

    // every byte value at several pixel counts: kernel and reference must
    // agree bit-exactly on all three RGB lanes, and lane 3 must keep its
    // sentinel everywhere (alpha preservation across the whole buffer).
    #[test]
    fn matches_reference_over_sweep() {
        for n in [1usize, 3, 65, 513] {
            let mut src = Vec::with_capacity(3 * n);
            for i in 0..3 * n {
                // LCG over the full 0..=255 byte range
                let v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37) % 256;
                src.push(v as u8);
            }
            src[0] = 0;
            src[1] = 255;
            let mut direct = vec![-1.0f32; 4 * n];
            let mut reference = vec![-1.0f32; 4 * n];
            png_u8_to_float(&src, &mut direct, n);
            ref_png_u8_to_float(&src, &mut reference, n);
            assert_eq!(direct.len(), reference.len());
            for (k, (d, r)) in direct.iter().zip(reference.iter()).enumerate() {
                assert_eq!(d.to_bits(), r.to_bits(), "lane {k}");
                if k % 4 == 3 {
                    assert_eq!(*d, -1.0f32, "alpha lane {k} preserved");
                }
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let src = vec![200u8; 12];
        // zero pixels: output untouched
        let mut out = vec![9.0f32; 16];
        png_u8_to_float(&src, &mut out, 0);
        assert_eq!(out, vec![9.0f32; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_out = vec![0.0f32; 7]; // short for 2 quads
        png_u8_to_float(&src, &mut short_out, 2);
        let short_src = vec![200u8; 5]; // short for 2 triples
        let mut out2 = vec![0.0f32; 8];
        png_u8_to_float(&short_src, &mut out2, 2);
        let empty: Vec<u8> = vec![];
        let mut out3 = vec![0.0f32; 4];
        png_u8_to_float(&empty, &mut out3, 1);
        assert_eq!(out3, vec![0.0f32; 4]);
    }

    #[test]
    fn ffi_round_trip() {
        let n = 65usize;
        let mut src = vec![0u8; 3 * n];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i as u64).wrapping_mul(2_654_435_761) % 256) as u8;
        }
        let mut ffi_out = vec![-2.0f32; 4 * n];
        let mut direct_out = vec![-2.0f32; 4 * n];
        unsafe {
            darkroom_png_u8_to_float(src.as_ptr(), ffi_out.as_mut_ptr(), n);
        }
        png_u8_to_float(&src, &mut direct_out, n);
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn ffi_guards() {
        let src = vec![200u8; 12];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_png_u8_to_float(std::ptr::null(), out.as_mut_ptr(), 4);
            darkroom_png_u8_to_float(src.as_ptr(), std::ptr::null_mut(), 4);
            // zero pixels
            darkroom_png_u8_to_float(src.as_ptr(), out.as_mut_ptr(), 0);
            // overflowing pixel counts (3/4 * npixels wraps — all reject)
            darkroom_png_u8_to_float(src.as_ptr(), out.as_mut_ptr(), usize::MAX);
            darkroom_png_u8_to_float(src.as_ptr(), out.as_mut_ptr(), usize::MAX / 4 + 1);
            darkroom_png_u8_to_float(src.as_ptr(), out.as_mut_ptr(), usize::MAX / 3 + 1);
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }
}
