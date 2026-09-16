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

/// Quantize float RGB lanes into the 8-bit HEIF export plane.
///
/// Port of the `case 8` branch of `write_image`
/// (`src/imageio/format/heif.c`, m4-213): `src` holds `4*width*height`
/// tightly packed floats (RGBA; the alpha lane is never read), `out`
/// receives the interleaved `RGB` plane (`height` rows, `rowbytes` bytes
/// each, 3 u8 lanes per pixel). Per lane,
/// `out = (u8)roundf(CLAMP(src[c] * max_channel, 0, max_channel))` with
/// `max_channel = 255.0` at 8-bit depth.
///
/// Fidelity notes (each pinned by a test below):
/// - The multiply runs before the clamp, in f32, exactly as parenthesised
///   in C. Clamping first would differ for out-of-range inputs (e.g. a
///   `-0.1` lane clamps to 0 rather than rounding to -0).
/// - The clamp is spelled branch-by-branch, high bound first exactly like
///   glib's `CLAMP`, instead of `f32::clamp`, which panics on NaN. A NaN
///   lane falls through to `round` just as in C, where `roundf(NaN)` stays
///   NaN; the final `as u8` saturates NaN to 0 in Rust (defined), matching
///   the x86 convert result the C cast produces in practice.
/// - `f32::round` is round-half-away-from-zero, matching `roundf`, so a
///   `127.5` product lands on 128.
/// - Degenerate dims (`width == 0` or `height == 0`), a non-finite or
///   non-positive `max_channel`, a `rowbytes` narrower than one pixel row,
///   and short buffers are guarded no-ops / clamped iteration (no panic,
///   no out-of-bounds access). For the well-formed buffers the C caller
///   passes the clamps never engage and the behaviour is exactly the C
///   loop's.
///
/// The Rust kernel is single-threaded sequential; the C loop was
/// `DT_OMP_FOR(collapse(2))` over rows and columns, but each output triple
/// reads only its own source quad, so thread scheduling cannot change the
/// result.
pub fn heif_float_to_u8(
    src: &[f32],
    out: &mut [u8],
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if width == 0 || height == 0 || !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    let Some(src_row) = width.checked_mul(4) else {
        return;
    };
    let Some(dst_row) = width.checked_mul(3) else {
        return;
    };
    if rowbytes < dst_row {
        return;
    }
    for y in 0..height {
        let Some(s) = y.checked_mul(src_row) else {
            break;
        };
        let Some(s_end) = s.checked_add(src_row) else {
            break;
        };
        if src.len() < s_end {
            break;
        }
        let Some(d) = y.checked_mul(rowbytes) else {
            break;
        };
        let Some(d_end) = d.checked_add(dst_row) else {
            break;
        };
        if out.len() < d_end {
            break;
        }
        for x in 0..width {
            let si = s + 4 * x;
            let di = d + 3 * x;
            for c in 0..3 {
                let v = src[si + c] * max_channel;
                // glib CLAMP order (high bound first); NaN falls through.
                let cl = if v > max_channel {
                    max_channel
                } else if v < 0.0 {
                    0.0
                } else {
                    v
                };
                out[di + c] = cl.round() as u8;
            }
        }
    }
}

// ── Independent reference for the export kernel (bit-exactness tests) ────────

/// Structurally divergent reference for `heif_float_to_u8`: walks pixels
/// by flat index with `div`/`mod` (the kernel uses nested row/column loops
/// with explicit stride arithmetic) and converts each lane through a
/// scalar helper over zipped slice iterators (the kernel indexes both
/// sides directly), so the sweep test cross-checks indexing as well as
/// values. The arithmetic is the same multiply-then-branch-clamp-then-round
/// spelling by construction (see the kernel docs: clamp-first and
/// `f32::clamp` must not be used). Same well-formed-buffers
/// precondition, enforced here by early return rather than clamping.
#[cfg(test)]
fn ref_lane(v: f32, max_channel: f32) -> u8 {
    let scaled = v * max_channel;
    let cl = if scaled > max_channel {
        max_channel
    } else if scaled < 0.0 {
        0.0
    } else {
        scaled
    };
    cl.round() as u8
}

#[cfg(test)]
fn ref_heif_float_to_u8(
    src: &[f32],
    out: &mut [u8],
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if width == 0 || height == 0 || !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    let (Some(src_row), Some(dst_row)) = (width.checked_mul(4), width.checked_mul(3)) else {
        return;
    };
    if rowbytes < dst_row {
        return;
    }
    let Some(npixels) = width.checked_mul(height) else {
        return;
    };
    if src.len() < npixels.saturating_mul(4) || out.len() < height.saturating_mul(rowbytes) {
        return;
    }
    for p in 0..npixels {
        let y = p / width;
        let x = p % width;
        let lanes: Vec<u8> = src[y * src_row + 4 * x..y * src_row + 4 * x + 3]
            .iter()
            .map(|&v| ref_lane(v, max_channel))
            .collect();
        for (slot, lane) in out[y * rowbytes + 3 * x..y * rowbytes + 3 * x + 3]
            .iter_mut()
            .zip(lanes.iter())
        {
            *slot = *lane;
        }
    }
}

/// # Safety
/// `in_data` must hold at least `4 * width * height` floats (the C caller
/// passes the export pipeline buffer, tightly packed RGBA) and `out` at
/// least `(height - 1) * rowbytes + 3 * width` bytes (the libheif
/// interleaved `RGB` plane, `rowbytes` its per-row byte stride). The two
/// buffers must not overlap. `max_channel` is the C `max_channel_f`, i.e.
/// `(float)((1 << bit_depth) - 1)`, strictly positive (255.0 at 8-bit).
#[no_mangle]
pub unsafe extern "C" fn darkroom_heif_float_to_u8(
    in_data: *const f32,
    out: *mut u8,
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if in_data.is_null() || out.is_null() || width == 0 || height == 0 {
        return;
    }
    if !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(dst_row) = width.checked_mul(3) else {
        return;
    };
    if rowbytes < dst_row {
        return;
    }
    let src_len = match width.checked_mul(height).and_then(|n| n.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    let dst_len = match height
        .checked_sub(1)
        .and_then(|h| h.checked_mul(rowbytes))
        .and_then(|base| base.checked_add(dst_row))
    {
        Some(n) => n,
        None => return,
    };
    let src = std::slice::from_raw_parts(in_data, src_len);
    let out = std::slice::from_raw_parts_mut(out, dst_len);
    heif_float_to_u8(src, out, width, height, rowbytes, max_channel);
}

/// Quantize float RGB lanes into the 10/12-bit HEIF export plane.
///
/// Port of the `case 10` / `case 12` branch of `write_image`
/// (`src/imageio/format/heif.c`, m4-214): `src` holds `4*width*height`
/// tightly packed floats (RGBA; the alpha lane is never read), `out`
/// receives the interleaved `RRGGBB_LE` plane (`height` rows, `rowbytes`
/// bytes each, 3 little-endian u16 lanes per pixel). Per lane,
/// `out = (u16)roundf(CLAMP(src[c] * max_channel, 0, max_channel))` with
/// `max_channel` 1023.0 at 10-bit depth and 4095.0 at 12-bit depth.
///
/// Fidelity notes (each pinned by a test below; same spelling as the
/// 8-bit sibling `heif_float_to_u8` from m4-213):
/// - The multiply runs before the clamp, in f32, exactly as parenthesised
///   in C. The clamp is spelled branch-by-branch, high bound first exactly
///   like glib's `CLAMP`, instead of `f32::clamp`, which panics on NaN. A
///   NaN lane falls through to `round` just as in C, where `roundf(NaN)`
///   stays NaN; the final `as u16` saturates NaN to 0 in Rust (defined),
///   matching the x86 convert result the C cast produces in practice.
/// - `f32::round` is round-half-away-from-zero, matching `roundf`, so a
///   `511.5` product lands on 512.
/// - The C loop stores through a `uint16_t *` cast of the byte plane, i.e.
///   a native-endian 16-bit store. The plane is created with the explicit
///   `heif_chroma_interleaved_RRGGBB_LE` chroma (and darktable only
///   supports little-endian hosts, as stated in `imageio_heif.c`), so the
///   kernel encodes each lane with `u16::to_le_bytes` explicitly rather
///   than relying on the host endianness (mirrors the explicit
///   `from_le_bytes` loads of the m4-208 import kernel).
/// - Degenerate dims (`width == 0` or `height == 0`), a non-finite or
///   non-positive `max_channel`, a `rowbytes` narrower than one pixel row,
///   and short buffers are guarded no-ops / clamped iteration (no panic,
///   no out-of-bounds access). For the well-formed buffers the C caller
///   passes the clamps never engage and the behaviour is exactly the C
///   loop's.
///
/// The Rust kernel is single-threaded sequential; the C loop was
/// `DT_OMP_FOR(collapse(2))` over rows and columns, but each output triple
/// reads only its own source quad, so thread scheduling cannot change the
/// result.
pub fn heif_float_to_u16(
    src: &[f32],
    out: &mut [u8],
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if width == 0 || height == 0 || !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    let Some(src_row) = width.checked_mul(4) else {
        return;
    };
    let Some(dst_row) = width.checked_mul(6) else {
        return;
    };
    if rowbytes < dst_row {
        return;
    }
    for y in 0..height {
        let Some(s) = y.checked_mul(src_row) else {
            break;
        };
        let Some(s_end) = s.checked_add(src_row) else {
            break;
        };
        if src.len() < s_end {
            break;
        }
        let Some(d) = y.checked_mul(rowbytes) else {
            break;
        };
        let Some(d_end) = d.checked_add(dst_row) else {
            break;
        };
        if out.len() < d_end {
            break;
        }
        for x in 0..width {
            let si = s + 4 * x;
            let di = d + 6 * x;
            for c in 0..3 {
                let v = src[si + c] * max_channel;
                // glib CLAMP order (high bound first); NaN falls through.
                let cl = if v > max_channel {
                    max_channel
                } else if v < 0.0 {
                    0.0
                } else {
                    v
                };
                let bytes = (cl.round() as u16).to_le_bytes();
                out[di + 2 * c] = bytes[0];
                out[di + 2 * c + 1] = bytes[1];
            }
        }
    }
}

// ── Independent reference for the 10/12-bit export kernel ────────────────────

/// Structurally divergent reference for `heif_float_to_u16`: walks pixels
/// by flat index with `div`/`mod` (the kernel uses nested row/column loops
/// with explicit stride arithmetic) and converts each lane through a
/// scalar helper over zipped slice iterators (the kernel indexes both
/// sides directly), so the sweep test cross-checks indexing as well as
/// values. The arithmetic is the same multiply-then-branch-clamp-then-round
/// spelling by construction (see the kernel docs: clamp-first and
/// `f32::clamp` must not be used). Same well-formed-buffers
/// precondition, enforced here by early return rather than clamping —
/// so on a multi-row short buffer the kernel partial-writes row by row
/// while this reference writes nothing; only well-formed buffers (which
/// the C caller always passes) are compared by the sweep test.
#[cfg(test)]
fn ref_lane_u16(v: f32, max_channel: f32) -> u16 {
    let scaled = v * max_channel;
    let cl = if scaled > max_channel {
        max_channel
    } else if scaled < 0.0 {
        0.0
    } else {
        scaled
    };
    cl.round() as u16
}

#[cfg(test)]
fn ref_heif_float_to_u16(
    src: &[f32],
    out: &mut [u8],
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if width == 0 || height == 0 || !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    let (Some(src_row), Some(dst_row)) = (width.checked_mul(4), width.checked_mul(6)) else {
        return;
    };
    if rowbytes < dst_row {
        return;
    }
    let Some(npixels) = width.checked_mul(height) else {
        return;
    };
    if src.len() < npixels.saturating_mul(4) || out.len() < height.saturating_mul(rowbytes) {
        return;
    }
    for p in 0..npixels {
        let y = p / width;
        let x = p % width;
        let lanes: Vec<u16> = src[y * src_row + 4 * x..y * src_row + 4 * x + 3]
            .iter()
            .map(|&v| ref_lane_u16(v, max_channel))
            .collect();
        for (slot, lane) in out[y * rowbytes + 6 * x..y * rowbytes + 6 * x + 6]
            .chunks_exact_mut(2)
            .zip(lanes.iter())
        {
            slot.copy_from_slice(&lane.to_le_bytes());
        }
    }
}

/// # Safety
/// `in_data` must hold at least `4 * width * height` floats (the C caller
/// passes the export pipeline buffer, tightly packed RGBA) and `out` at
/// least `(height - 1) * rowbytes + 6 * width` bytes (the libheif
/// interleaved `RRGGBB_LE` plane, `rowbytes` its per-row byte stride). The
/// two buffers must not overlap. `max_channel` is the C `max_channel_f`,
/// i.e. `(float)((1 << bit_depth) - 1)`, strictly positive (1023.0 at
/// 10-bit depth, 4095.0 at 12-bit depth).
#[no_mangle]
pub unsafe extern "C" fn darkroom_heif_float_to_u16(
    in_data: *const f32,
    out: *mut u8,
    width: usize,
    height: usize,
    rowbytes: usize,
    max_channel: f32,
) {
    if in_data.is_null() || out.is_null() || width == 0 || height == 0 {
        return;
    }
    if !max_channel.is_finite() || max_channel <= 0.0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(dst_row) = width.checked_mul(6) else {
        return;
    };
    if rowbytes < dst_row {
        return;
    }
    let src_len = match width.checked_mul(height).and_then(|n| n.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    let dst_len = match height
        .checked_sub(1)
        .and_then(|h| h.checked_mul(rowbytes))
        .and_then(|base| base.checked_add(dst_row))
    {
        Some(n) => n,
        None => return,
    };
    let src = std::slice::from_raw_parts(in_data, src_len);
    let out = std::slice::from_raw_parts_mut(out, dst_len);
    heif_float_to_u16(src, out, width, height, rowbytes, max_channel);
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

    // Export rails at 8-bit depth (max 255): 0.0 -> 0, 1.0 -> 255, the
    // 0.5 lane pins the reciprocal-free multiply-then-round spelling
    // (0.5 * 255 = 127.5 -> 128), and out-of-range lanes clamp on both
    // sides. The alpha lane is never read: out holds 3 lanes per pixel.
    #[test]
    fn float_to_u8_rails_pin() {
        let max = 255.0f32;
        let src = [0.0f32, 0.5, 1.0, 0.9, 2.0, -0.1, 0.25, 0.0];
        let mut out = vec![0xCDu8; 6];
        heif_float_to_u8(&src, &mut out, 2, 1, 6, max);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], ((0.5f32 * max).round() as u8)); // 127.5 -> 128
        assert_eq!(out[1], 128);
        assert_eq!(out[2], 255);
        assert_eq!(out[3], 255); // 2.0 * 255 clamps to 255
        assert_eq!(out[4], 0); // negative clamps to 0
        assert_eq!(out[5], ((0.25f32 * max).round() as u8));
    }

    // Multiply-then-clamp order matters: a -0.1 lane scales to -25.5 and
    // clamps to 0 (clamp-first would round -25.5 to -26 and then clamp,
    // same result here, but a lane of exactly -0.5/255 would round to -1
    // under round-first and must read 0). NaN falls through the
    // branch-clamp like glib's CLAMP and saturates to 0 via `as u8`.
    #[test]
    fn float_to_u8_clamp_order_and_nan() {
        let max = 255.0f32;
        // one pixel: tiny-negative, NaN, and -1.0 lanes all read 0 (the
        // alpha slot is never read)
        let src = [-0.5f32 / 255.0, f32::NAN, -1.0, 9.9];
        let mut out = vec![0xCDu8; 3];
        heif_float_to_u8(&src, &mut out, 1, 1, 3, max);
        assert_eq!(out, vec![0, 0, 0]);
        // just-past-full-scale and far-past lanes clamp to 255
        let src2 = [1.0f32 + 0.5 / 255.0, 7.0, 1.0, 0.0];
        let mut out2 = vec![0xCDu8; 3];
        heif_float_to_u8(&src2, &mut out2, 1, 1, 3, max);
        assert_eq!(out2, vec![255, 255, 255]);
    }

    // Kernel and reference agree bit-exactly over several shapes
    // (including a padded stride), with inputs spanning below 0 and above
    // 1 so both clamp sides and the rounding are exercised. Padding bytes
    // are forced to 0xCD and must never leak into output.
    #[test]
    fn float_to_u8_matches_reference_over_sweep() {
        // (width, height, rowbytes)
        let shapes = [
            (1usize, 1usize, 3usize),
            (3, 2, 9),
            (5, 4, 20), // 5 bytes of padding per row
            (7, 5, 21),
            (16, 9, 55),
        ];
        for (w, h, rb) in shapes {
            let mut src = vec![0.0f32; 4 * w * h];
            for (i, v) in src.iter_mut().enumerate() {
                // deterministic sweep across [-0.5, 1.5]
                let k = ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(0x9E37) % 2001) as f32;
                *v = k / 1000.0 - 0.5;
            }
            let mut direct = vec![0xCDu8; rb * h];
            let mut reference = vec![0xCDu8; rb * h];
            heif_float_to_u8(&src, &mut direct, w, h, rb, 255.0);
            ref_heif_float_to_u8(&src, &mut reference, w, h, rb, 255.0);
            assert_eq!(direct, reference, "shape ({w},{h},{rb})");
            // padding bytes never written
            if rb > 3 * w {
                for y in 0..h {
                    assert_eq!(&direct[y * rb + 3 * w..(y + 1) * rb], &vec![0xCDu8; rb - 3 * w][..]);
                }
            }
        }
    }

    #[test]
    fn float_to_u8_degenerate_guards_no_op() {
        let src = vec![1.0f32; 8];
        let mut out = vec![0xCDu8; 6];
        // zero dims: output untouched
        heif_float_to_u8(&src, &mut out, 0, 2, 6, 255.0);
        heif_float_to_u8(&src, &mut out, 2, 0, 6, 255.0);
        assert_eq!(out, vec![0xCDu8; 6]);
        // non-positive / non-finite max: output untouched
        heif_float_to_u8(&src, &mut out, 2, 1, 6, 0.0);
        heif_float_to_u8(&src, &mut out, 2, 1, 6, -3.0);
        heif_float_to_u8(&src, &mut out, 2, 1, 6, f32::NAN);
        heif_float_to_u8(&src, &mut out, 2, 1, 6, f32::INFINITY);
        assert_eq!(out, vec![0xCDu8; 6]);
        // rowbytes narrower than one row: no-op
        heif_float_to_u8(&src, &mut out, 2, 1, 5, 255.0);
        assert_eq!(out, vec![0xCDu8; 6]);
        // truncated source: no panic, nothing written
        let short = vec![1.0f32; 3];
        heif_float_to_u8(&short, &mut out, 2, 1, 6, 255.0);
        assert_eq!(out, vec![0xCDu8; 6]);
    }

    #[test]
    fn float_to_u8_ffi_round_trip() {
        let (w, h, rb) = (9usize, 5usize, 30usize);
        let mut src = vec![0.0f32; 4 * w * h];
        for (i, v) in src.iter_mut().enumerate() {
            let k = ((i as u64).wrapping_mul(2_654_435_761) % 2001) as f32;
            *v = k / 1000.0 - 0.5;
        }
        let mut ffi_out = vec![0xCDu8; rb * h];
        let mut direct_out = vec![0xCDu8; rb * h];
        unsafe {
            darkroom_heif_float_to_u8(src.as_ptr(), ffi_out.as_mut_ptr(), w, h, rb, 255.0);
        }
        heif_float_to_u8(&src, &mut direct_out, w, h, rb, 255.0);
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn float_to_u8_ffi_guards() {
        let src = vec![1.0f32; 16];
        let mut out = vec![0xCDu8; 12];
        unsafe {
            // null pointers
            darkroom_heif_float_to_u8(std::ptr::null(), out.as_mut_ptr(), 2, 2, 6, 255.0);
            darkroom_heif_float_to_u8(src.as_ptr(), std::ptr::null_mut(), 2, 2, 6, 255.0);
            // zero dims
            darkroom_heif_float_to_u8(src.as_ptr(), out.as_mut_ptr(), 0, 2, 6, 255.0);
            darkroom_heif_float_to_u8(src.as_ptr(), out.as_mut_ptr(), 2, 0, 6, 255.0);
            // non-positive / non-finite max
            darkroom_heif_float_to_u8(src.as_ptr(), out.as_mut_ptr(), 2, 2, 6, 0.0);
            darkroom_heif_float_to_u8(src.as_ptr(), out.as_mut_ptr(), 2, 2, 6, f32::NAN);
            darkroom_heif_float_to_u8(src.as_ptr(), out.as_mut_ptr(), 2, 2, 6, f32::INFINITY);
            // rowbytes narrower than one row
            darkroom_heif_float_to_u8(src.as_ptr(), out.as_mut_ptr(), 2, 2, 5, 255.0);
            // overflowing dims
            darkroom_heif_float_to_u8(src.as_ptr(), out.as_mut_ptr(), usize::MAX, 2, 6, 255.0);
            darkroom_heif_float_to_u8(src.as_ptr(), out.as_mut_ptr(), 2, usize::MAX, 6, 255.0);
        }
        assert_eq!(out, vec![0xCDu8; 12]); // untouched
    }

    // Export rails at 10-bit depth (max 1023): 0.0 -> 0, 1.0 -> 1023, the
    // 0.5 lane pins the multiply-then-round spelling (0.5 * 1023 = 511.5
    // -> 512, round-half-away), and out-of-range lanes clamp on both
    // sides. The alpha lane is never read: out holds 3 lanes per pixel.
    // Full scale also pins the little-endian byte order (1023 = 0x03FF).
    #[test]
    fn float_to_u16_rails_pin() {
        let max = 1023.0f32;
        let src = [0.0f32, 0.5, 1.0, 0.9, 2.0, -0.1, 0.25, 0.0];
        let mut out = vec![0xCDu8; 12];
        heif_float_to_u16(&src, &mut out, 2, 1, 12, max);
        assert_eq!(&out[0..2], &u16::to_le_bytes(0));
        assert_eq!(&out[2..4], &u16::to_le_bytes(512)); // 511.5 -> 512
        assert_eq!(&out[4..6], &u16::to_le_bytes(1023));
        assert_eq!(&out[0..2], &[0x00, 0x00]);
        assert_eq!(&out[4..6], &[0xFF, 0x03]); // LE byte order pin
        assert_eq!(&out[6..8], &u16::to_le_bytes(1023)); // 2.0 * 1023 clamps
        assert_eq!(&out[8..10], &u16::to_le_bytes(0)); // negative clamps to 0
        assert_eq!(&out[10..12], &u16::to_le_bytes((0.25f32 * max).round() as u16));
        // 12-bit full scale: 4095 = 0x0FFF
        let mut out12 = vec![0xCDu8; 6];
        heif_float_to_u16(&[1.0f32, 1.0, 1.0, 0.0], &mut out12, 1, 1, 6, 4095.0);
        assert_eq!(&out12[0..2], &[0xFF, 0x0F]);
        assert_eq!(&out12[2..4], &[0xFF, 0x0F]);
        assert_eq!(&out12[4..6], &[0xFF, 0x0F]);
    }

    // Multiply-then-clamp order matters: a -0.5/1023 lane scales to -0.5
    // and clamps to 0 (round-first would give -1). NaN falls through the
    // branch-clamp like glib's CLAMP and saturates to 0 via `as u16`.
    #[test]
    fn float_to_u16_clamp_order_and_nan() {
        let max = 1023.0f32;
        // one pixel: tiny-negative, NaN, and -1.0 lanes all read 0 (the
        // alpha slot is never read)
        let src = [-0.5f32 / 1023.0, f32::NAN, -1.0, 9.9];
        let mut out = vec![0xCDu8; 6];
        heif_float_to_u16(&src, &mut out, 1, 1, 6, max);
        assert_eq!(out, vec![0, 0, 0, 0, 0, 0]);
        // just-past-full-scale and far-past lanes clamp to 1023
        let src2 = [1.0f32 + 0.5 / 1023.0, 7.0, 1.0, 0.0];
        let mut out2 = vec![0xCDu8; 6];
        heif_float_to_u16(&src2, &mut out2, 1, 1, 6, max);
        assert_eq!(&out2[0..2], &u16::to_le_bytes(1023));
        assert_eq!(&out2[2..4], &u16::to_le_bytes(1023));
        assert_eq!(&out2[4..6], &u16::to_le_bytes(1023));
    }

    // Kernel and reference agree bit-exactly over several shapes
    // (including padded strides and both 10- and 12-bit maxima), with
    // inputs spanning below 0 and above 1 so both clamp sides and the
    // rounding are exercised. Padding bytes are forced to 0xCD and must
    // never leak into output.
    #[test]
    fn float_to_u16_matches_reference_over_sweep() {
        // (width, height, rowbytes, max_channel)
        let shapes = [
            (1usize, 1usize, 6usize, 1023.0f32),
            (3, 2, 18, 1023.0),
            (5, 4, 40, 1023.0), // 10 bytes of padding per row
            (7, 5, 42, 4095.0), // 12-bit, tight
            (16, 9, 110, 4095.0),
        ];
        for (w, h, rb, max) in shapes {
            let mut src = vec![0.0f32; 4 * w * h];
            for (i, v) in src.iter_mut().enumerate() {
                // deterministic sweep across [-0.5, 1.5]
                let k = ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(0x9E37) % 2001) as f32;
                *v = k / 1000.0 - 0.5;
            }
            let mut direct = vec![0xCDu8; rb * h];
            let mut reference = vec![0xCDu8; rb * h];
            heif_float_to_u16(&src, &mut direct, w, h, rb, max);
            ref_heif_float_to_u16(&src, &mut reference, w, h, rb, max);
            assert_eq!(direct, reference, "shape ({w},{h},{rb},{max})");
            // padding bytes never written
            if rb > 6 * w {
                for y in 0..h {
                    assert_eq!(&direct[y * rb + 6 * w..(y + 1) * rb], &vec![0xCDu8; rb - 6 * w][..]);
                }
            }
        }
    }

    #[test]
    fn float_to_u16_degenerate_guards_no_op() {
        let src = vec![1.0f32; 8];
        let mut out = vec![0xCDu8; 12];
        // zero dims: output untouched
        heif_float_to_u16(&src, &mut out, 0, 2, 12, 1023.0);
        heif_float_to_u16(&src, &mut out, 2, 0, 12, 1023.0);
        assert_eq!(out, vec![0xCDu8; 12]);
        // non-positive / non-finite max: output untouched
        heif_float_to_u16(&src, &mut out, 2, 1, 12, 0.0);
        heif_float_to_u16(&src, &mut out, 2, 1, 12, -3.0);
        heif_float_to_u16(&src, &mut out, 2, 1, 12, f32::NAN);
        heif_float_to_u16(&src, &mut out, 2, 1, 12, f32::INFINITY);
        assert_eq!(out, vec![0xCDu8; 12]);
        // rowbytes narrower than one row: no-op
        heif_float_to_u16(&src, &mut out, 2, 1, 11, 1023.0);
        assert_eq!(out, vec![0xCDu8; 12]);
        // truncated source: no panic, nothing written
        let short = vec![1.0f32; 7];
        heif_float_to_u16(&short, &mut out, 2, 1, 12, 1023.0);
        assert_eq!(out, vec![0xCDu8; 12]);
    }

    #[test]
    fn float_to_u16_ffi_round_trip() {
        let (w, h, rb, max) = (9usize, 5usize, 60usize, 4095.0f32);
        let mut src = vec![0.0f32; 4 * w * h];
        for (i, v) in src.iter_mut().enumerate() {
            let k = ((i as u64).wrapping_mul(2_654_435_761) % 2001) as f32;
            *v = k / 1000.0 - 0.5;
        }
        let mut ffi_out = vec![0xCDu8; rb * h];
        let mut direct_out = vec![0xCDu8; rb * h];
        unsafe {
            darkroom_heif_float_to_u16(src.as_ptr(), ffi_out.as_mut_ptr(), w, h, rb, max);
        }
        heif_float_to_u16(&src, &mut direct_out, w, h, rb, max);
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn float_to_u16_ffi_guards() {
        let src = vec![1.0f32; 16];
        let mut out = vec![0xCDu8; 24];
        unsafe {
            // null pointers
            darkroom_heif_float_to_u16(std::ptr::null(), out.as_mut_ptr(), 2, 2, 12, 1023.0);
            darkroom_heif_float_to_u16(src.as_ptr(), std::ptr::null_mut(), 2, 2, 12, 1023.0);
            // zero dims
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), 0, 2, 12, 1023.0);
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), 2, 0, 12, 1023.0);
            // non-positive / non-finite max
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), 2, 2, 12, 0.0);
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), 2, 2, 12, -3.0);
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), 2, 2, 12, f32::NAN);
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), 2, 2, 12, f32::INFINITY);
            // rowbytes narrower than one row
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), 2, 2, 11, 1023.0);
            // overflowing dims
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), usize::MAX, 2, 12, 1023.0);
            darkroom_heif_float_to_u16(src.as_ptr(), out.as_mut_ptr(), 2, usize::MAX, 12, 1023.0);
        }
        assert_eq!(out, vec![0xCDu8; 24]); // untouched
    }
}
