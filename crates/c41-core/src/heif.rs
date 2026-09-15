//! Kernel ported from `src/imageio/imageio_heif.c` (`dt_imageio_open_heif`,
//! the u16-to-float normalize loop, m4-208). The kernel replaces the whole
//! loop body: each decoded 16-bit HEIF lane is scaled by `1/max_channel_f`
//! into the 4-channel float mipmap buffer, with the alpha lane zeroed.
//!
//! What the C loop does, per pixel `(y, x)`:
//! - `mipbuf[4*(y*width+x) + c] = (float)in_pixel[c] * (1.0f / max_channel_f)`
//!   for `c` in `0..2`, where `in_pixel` is the `uint16_t` triple at byte
//!   offset `y*rowbytes + 6*x` of the interleaved `RRGGBB_LE` plane
//!   (`heif_chroma_interleaved_RRGGBB_LE`), and
//! - `mipbuf[4*(y*width+x) + 3] = 0.0` (alpha zeroed, as in the WebP
//!   kernel of m4-207).
//!
//! Bit-exactness notes:
//! - `u16 as f32` is exact for all 65536 inputs. The C loop multiplies by
//!   the parenthesised reciprocal `(1.0f / max_channel_f)`, so the kernel
//!   computes `inv = 1.0f32 / max_channel` once and multiplies — the same
//!   operation order, bit for bit. A per-lane division `lane / max_channel`
//!   is NOT an acceptable substitute (the reciprocal is itself inexact, so
//!   multiply-by-reciprocal can round differently from division).
//! - The C loop reads through a `uint16_t *` cast of the byte plane, i.e. a
//!   native-endian 16-bit load. darktable only supports little-endian hosts
//!   (stated in the C file next to the decode call), so the kernel decodes
//!   each lane with `u16::from_le_bytes` explicitly rather than relying on
//!   the host endianness.
//! - Input rows may carry padding (`rowbytes > 6*width`); padding bytes are
//!   never read. Output rows are tightly packed (`4*width` floats each).
//! - Buffers: the C caller passes the libheif plane (`data`) and the
//!   mipmap-cache allocation. The kernel must not be called with
//!   overlapping buffers.
//!
//! The Rust kernel is single-threaded sequential; the C loop was
//! `DT_OMP_FOR_SIMD(collapse(2))` over rows and columns, but each output
//! quad reads only its own source triple, so thread scheduling cannot
//! change the result.

/// Scale decoded HEIF 16-bit lanes into the float mipmap buffer.
///
/// Port of the former element-wise loop in `dt_imageio_open_heif`
/// (src/imageio/imageio_heif.c): `src` holds the interleaved `RRGGBB_LE`
/// plane (`height` rows, `rowbytes` bytes each, 3 little-endian u16 lanes
/// per pixel), `out` receives `4*width*height` floats (RGB lanes times the
/// `1/max_channel` reciprocal, alpha lane zeroed). See the module docs for
/// the reciprocal-multiply fidelity point, the explicit little-endian
/// decode, and the no-aliasing contract.
///
/// Degenerate dims (`width == 0` or `height == 0`) or a non-positive
/// `max_channel` are no-ops; short buffers and a `rowbytes` narrower than
/// one pixel row are handled by clamped iteration (no panic, no
/// out-of-bounds access). For the well-formed buffers the C caller passes
/// the clamps never engage and the behaviour is exactly the C loop's.
pub fn heif_u16_to_float(
    src: &[u8],
    out: &mut [f32],
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if width == 0 || height == 0 || !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    let inv = 1.0f32 / max_channel;
    let Some(row_need) = width.checked_mul(6) else {
        return;
    };
    let Some(out_row) = width.checked_mul(4) else {
        return;
    };
    if rowbytes < row_need {
        return;
    }
    for y in 0..height {
        let Some(base) = y.checked_mul(rowbytes) else {
            break;
        };
        let Some(src_end) = base.checked_add(row_need) else {
            break;
        };
        if src.len() < src_end {
            break;
        }
        let Some(d) = y.checked_mul(out_row) else {
            break;
        };
        let Some(out_end) = d.checked_add(out_row) else {
            break;
        };
        if out.len() < out_end {
            break;
        }
        for x in 0..width {
            let s = base + 6 * x;
            let lane0 = u16::from_le_bytes([src[s], src[s + 1]]) as f32;
            let lane1 = u16::from_le_bytes([src[s + 2], src[s + 3]]) as f32;
            let lane2 = u16::from_le_bytes([src[s + 4], src[s + 5]]) as f32;
            let o = d + 4 * x;
            out[o] = lane0 * inv;
            out[o + 1] = lane1 * inv;
            out[o + 2] = lane2 * inv;
            out[o + 3] = 0.0f32;
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `heif_u16_to_float`: walks pixels
/// by flat index with `div`/`mod` (the kernel uses nested row/column loops
/// with explicit stride arithmetic) and reads each lane through a
/// two-byte chunk iterator (the kernel indexes byte pairs directly), so
/// the sweep test cross-checks indexing as well as values. The arithmetic
/// is the same `lane as f32 * (1.0 / max_channel)` reciprocal multiply by
/// construction (see the module docs: per-lane division is not
/// bit-identical and must not be used), and lane 3 is assigned the
/// literal `0.0`. Same well-formed-buffers precondition, enforced here by
/// early return rather than clamping.
#[cfg(test)]
fn ref_heif_u16_to_float(
    src: &[u8],
    out: &mut [f32],
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if width == 0 || height == 0 || !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    let Some(row_need) = width.checked_mul(6) else {
        return;
    };
    if rowbytes < row_need {
        return;
    }
    let Some(npixels) = width.checked_mul(height) else {
        return;
    };
    let Some(out_need) = npixels.checked_mul(4) else {
        return;
    };
    let Some(src_need) = rowbytes.checked_mul(height) else {
        return;
    };
    if src.len() < src_need || out.len() < out_need {
        return;
    }
    let inv = 1.0f32 / max_channel;
    for p in 0..npixels {
        let y = p / width;
        let x = p % width;
        let s = y * rowbytes + 6 * x;
        let mut lanes = [0.0f32; 3];
        for (lane, bytes) in lanes.iter_mut().zip(src[s..s + 6].chunks_exact(2)) {
            *lane = u16::from_le_bytes([bytes[0], bytes[1]]) as f32 * inv;
        }
        let o = 4 * p;
        out[o] = lanes[0];
        out[o + 1] = lanes[1];
        out[o + 2] = lanes[2];
        out[o + 3] = 0.0f32;
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `rgb_buf` must hold at least `(height - 1) * rowbytes + 6 * width`
/// bytes (the C caller passes the libheif interleaved `RRGGBB_LE` plane,
/// `rowbytes` its per-row byte stride) and `mipbuf` at least
/// `4 * width * height` floats (the mipmap-cache allocation for the
/// 4-channel float image). The two buffers must not overlap.
/// `max_channel` is the C `max_channel_f`, i.e.
/// `(float)((1 << decoded_values_bit_depth) - 1)`, strictly positive.
#[no_mangle]
pub unsafe extern "C" fn darkroom_heif_u16_to_float(
    rgb_buf: *const u8,
    mipbuf: *mut f32,
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if rgb_buf.is_null() || mipbuf.is_null() || width == 0 || height == 0 {
        return;
    }
    if !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(row_need) = width.checked_mul(6) else {
        return;
    };
    if rowbytes < row_need {
        return;
    }
    let src_len = match height
        .checked_sub(1)
        .and_then(|h| h.checked_mul(rowbytes))
        .and_then(|base| base.checked_add(row_need))
    {
        Some(n) => n,
        None => return,
    };
    let out_len = match width.checked_mul(height).and_then(|n| n.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    let src = std::slice::from_raw_parts(rgb_buf, src_len);
    let out = std::slice::from_raw_parts_mut(mipbuf, out_len);
    heif_u16_to_float(src, out, width, height, rowbytes, max_channel);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 10-bit rails: 0 scales to +0.0 exactly; the alpha lane is pinned to
    // +0.0; the full-scale lane pins the reciprocal-multiply spelling the
    // C loop uses (1023 * (1/1023), not 1023/1023).
    #[test]
    fn rails_pin() {
        let max = 1023.0f32;
        // one pixel: lanes (0, 1, 1023)
        let src = [0u8, 0, 1, 0, 255, 3];
        let mut out = vec![7.0f32; 4];
        heif_u16_to_float(&src, &mut out, 1, 1, 6, max);
        assert_eq!(out[0].to_bits(), 0x0000_0000); // 0 -> +0.0
        assert_eq!(out[1].to_bits(), (1.0f32 * (1.0f32 / max)).to_bits());
        assert_eq!(out[2].to_bits(), (1023.0f32 * (1.0f32 / max)).to_bits());
        assert_eq!(out[3].to_bits(), 0x0000_0000); // alpha zeroed
    }

    // kernel and reference must agree bit-exactly on every lane over
    // several shapes (including a padded stride and a 12-bit max), and
    // lane 3 must read +0.0 everywhere.
    #[test]
    fn matches_reference_over_sweep() {
        // (width, height, rowbytes, max_channel)
        let shapes = [
            (1usize, 1usize, 6usize, 1023.0f32),
            (3, 2, 18, 1023.0),
            (5, 4, 40, 1023.0), // 10 bytes of padding per row
            (7, 5, 42, 4095.0), // 12-bit, tight
            (16, 9, 100, 4095.0),
        ];
        for (w, h, rb, max) in shapes {
            let mut src = vec![0u8; rb * h];
            for (i, v) in src.iter_mut().enumerate() {
                // LCG over the full byte range
                *v = ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(0x9E37) % 256) as u8;
            }
            // force padding bytes to 0xFF: they must never leak into output
            if rb > 6 * w {
                for y in 0..h {
                    for b in (y * rb + 6 * w)..((y + 1) * rb) {
                        src[b] = 0xFF;
                    }
                }
            }
            let mut direct = vec![-1.0f32; 4 * w * h];
            let mut reference = vec![-1.0f32; 4 * w * h];
            heif_u16_to_float(&src, &mut direct, w, h, rb, max);
            ref_heif_u16_to_float(&src, &mut reference, w, h, rb, max);
            assert_eq!(direct.len(), reference.len());
            for (k, (d, r)) in direct.iter().zip(reference.iter()).enumerate() {
                assert_eq!(d.to_bits(), r.to_bits(), "shape ({w},{h},{rb}) lane {k}");
                if k % 4 == 3 {
                    assert_eq!(*d, 0.0f32, "alpha lane {k} zeroed");
                }
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let src = vec![0xCDu8; 12];
        // zero dims: output untouched
        let mut out = vec![9.0f32; 8];
        heif_u16_to_float(&src, &mut out, 0, 2, 6, 1023.0);
        heif_u16_to_float(&src, &mut out, 2, 0, 12, 1023.0);
        assert_eq!(out, vec![9.0f32; 8]);
        // non-positive max: output untouched
        heif_u16_to_float(&src, &mut out, 1, 2, 6, 0.0);
        heif_u16_to_float(&src, &mut out, 1, 2, 6, -3.0);
        heif_u16_to_float(&src, &mut out, 1, 2, 6, f32::NAN);
        assert_eq!(out, vec![9.0f32; 8]);
        // rowbytes narrower than one row: no-op
        heif_u16_to_float(&src, &mut out, 2, 1, 6, 1023.0);
        assert_eq!(out, vec![9.0f32; 8]);
        // truncated source: no panic, first row converts fully (12 of the
        // 13 bytes), unwritten rows stay untouched.
        // 2x2 tight needs 24 bytes; 13 bytes hold row 0 (12) plus one byte.
        let short_src = vec![0x02u8; 13];
        let mut short_out = vec![0.0f32; 16];
        heif_u16_to_float(&short_src, &mut short_out, 2, 2, 12, 1023.0);
        let lane = 0x0202u16 as f32 * (1.0f32 / 1023.0f32);
        for q in 0..2 {
            for c in 0..3 {
                assert_eq!(short_out[4 * q + c].to_bits(), lane.to_bits());
            }
            assert_eq!(short_out[4 * q + 3].to_bits(), 0.0f32.to_bits());
        }
        assert_eq!(&short_out[8..16], &[0.0f32; 8]);
        // empty buffers with live dims: no panic, no writes possible
        let empty: Vec<u8> = vec![];
        let mut out2 = vec![0.0f32; 4];
        heif_u16_to_float(&empty, &mut out2, 1, 1, 6, 1023.0);
        assert_eq!(out2, vec![0.0f32; 4]);
    }

    #[test]
    fn ffi_round_trip() {
        let (w, h, rb, max) = (9usize, 5usize, 60usize, 1023.0f32);
        let mut src = vec![0u8; rb * h];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i as u64).wrapping_mul(2_654_435_761) % 256) as u8;
        }
        let mut ffi_out = vec![-2.0f32; 4 * w * h];
        let mut direct_out = vec![-2.0f32; 4 * w * h];
        unsafe {
            darkroom_heif_u16_to_float(src.as_ptr(), ffi_out.as_mut_ptr(), w, h, rb, max);
        }
        heif_u16_to_float(&src, &mut direct_out, w, h, rb, max);
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn ffi_guards() {
        let src = vec![0xCDu8; 24];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_heif_u16_to_float(std::ptr::null(), out.as_mut_ptr(), 2, 2, 12, 1023.0);
            darkroom_heif_u16_to_float(src.as_ptr(), std::ptr::null_mut(), 2, 2, 12, 1023.0);
            // zero dims
            darkroom_heif_u16_to_float(src.as_ptr(), out.as_mut_ptr(), 0, 2, 12, 1023.0);
            darkroom_heif_u16_to_float(src.as_ptr(), out.as_mut_ptr(), 2, 0, 12, 1023.0);
            // non-positive / non-finite max
            darkroom_heif_u16_to_float(src.as_ptr(), out.as_mut_ptr(), 2, 2, 12, 0.0);
            darkroom_heif_u16_to_float(src.as_ptr(), out.as_mut_ptr(), 2, 2, 12, f32::NAN);
            darkroom_heif_u16_to_float(src.as_ptr(), out.as_mut_ptr(), 2, 2, 12, f32::INFINITY);
            // rowbytes narrower than one row
            darkroom_heif_u16_to_float(src.as_ptr(), out.as_mut_ptr(), 2, 2, 6, 1023.0);
            // overflowing dims
            darkroom_heif_u16_to_float(src.as_ptr(), out.as_mut_ptr(), usize::MAX, 2, 12, 1023.0);
            darkroom_heif_u16_to_float(src.as_ptr(), out.as_mut_ptr(), 2, usize::MAX, 12, 1023.0);
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }
}
