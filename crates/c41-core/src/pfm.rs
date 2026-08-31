//! Kernel ported from `src/common/pfm.c` (`dt_read_pfm`, the two data-
//! unpacking loops at pfm.c:174-208, m4-175). One kernel replaces both the
//! `channels == 3` (colour) and the `channels == 1` (monochrome) branch,
//! with `channels` as a parameter branching internally — the same
//! two-C-track-loops-become-one-kernel shape as the m4-168 quantize port.
//!
//! What the loops do: `readbuf` holds `width*height*channels` floats as read
//! from the PFM file; `image` receives `width*height*planes` floats. Per
//! output pixel:
//! - Row flip: the source row is `target_row = made_by_photoshop ? row :
//!   height - 1 - row`. The de facto standard (set by the first PFM
//!   implementation) scanline order is bottom-to-top, so the flip is the
//!   default; Photoshop writes top-to-bottom and is detected by the C
//!   caller (whitespace in the header line) via `made_by_photoshop`.
//! - Byte swap: when `swap_byte_order` is set (little-endian host reading a
//!   big-endian-marked file or vice versa, as computed by the C caller from
//!   the scale factor sign), each source f32 is byte-swapped through the C
//!   union type-pun `value.as_int = GUINT32_SWAP_LE_BE(value.as_int)` — a
//!   full 32-bit byte swap, `f32::from_bits(v.to_bits().swap_bytes())` here.
//! - RGB branch (`channels == 3`): the three swapped channels land in a
//!   zero-initialised `dt_aligned_pixel_t pix` (4 floats), which is then
//!   copied to the output for `c in 0..planes` — so with `planes == 4` the
//!   fourth output channel is the 0.0 the pixel was initialised with.
//! - Mono branch (`channels == 1`): the single swapped value is broadcast
//!   to every output plane, `c in 0..planes`.
//! - Output indexing is `image[planes*(row*width + column) + c]` in both
//!   branches.
//!
//! Bit-exactness notes:
//! - The C file is compiled with the repo-wide Release flags
//!   `-O3 -ffast-math -fno-finite-math-only`, but for THIS port that is
//!   irrelevant: there are NO floating-point arithmetic operations at all.
//!   Every value moves by pure bit moves (a conditional `swap_bytes` on the
//!   bit pattern plus plain copies), so the port is exactly bit-preserving —
//!   including NaN payloads and signalling-NaN bits — for every input, with
//!   no ULP slack and no compiler-flags dependence. This is pinned by tests
//!   via exact `to_bits()` comparisons.
//! - `planes` contract: the kernel accepts `planes in 1..=4`. The C code
//!   would read out of bounds of its 4-float `pix` array for `planes > 4`
//!   in the RGB branch (the mono branch would be fine but shares the call);
//!   all real callers pass `planes ∈ {3, 4}` (chart/main.c:360 and
//!   rasterfile.c:256 pass 3, imageio_pfm.c:41 passes 4), so `planes > 4`
//!   is rejected as a guard rather than ported.
//! - `channels` contract: only 1 and 3 exist in the PFM format (and in the
//!   C if/else); anything else is rejected. The C if/else treats every
//!   non-3 value as mono, but the caller can only produce 1 or 3, so the
//!   kernel is stricter on purpose.
//! - Degenerate/short buffers: iteration is clamped so that no source or
//!   target index can go out of bounds (rows capped by what fits in each
//!   buffer at its stride, columns capped per row). For the well-formed
//!   buffers the C caller allocates (`width*height*channels` and
//!   `width*height*planes` exactly) the clamps never engage and the
//!   behaviour is exactly the C loop's.
//!
//! The Rust kernel is single-threaded sequential; the C loops were
//! `DT_OMP_FOR` over rows, but the per-pixel work is pure data movement, so
//! the merge order is irrelevant and the result is identical.

/// Conditional full 32-bit byte swap of an f32's bit pattern — the Rust
/// equivalent of the C union pun with `GUINT32_SWAP_LE_BE`. Pure bit move:
/// preserves every bit pattern, NaN payloads included.
#[inline]
fn maybe_swap(v: f32, swap_byte_order: bool) -> f32 {
    if swap_byte_order {
        f32::from_bits(v.to_bits().swap_bytes())
    } else {
        v
    }
}

/// Unpack PFM file data into the planar-strided output image.
///
/// Port of the former element-wise unpack loops at pfm.c:174-208 (both the
/// RGB and the mono branch; `channels` selects the path internally).
/// `readbuf` holds `width*height*channels` floats; `image` receives
/// `width*height*planes` floats at stride `planes` per pixel. See the
/// module docs for the row-flip semantics, the byte-swap bit preservation,
/// and the `channels ∈ {1, 3}`, `planes ≤ 4` contracts.
///
/// Degenerate dimensions (`width`/`height`/`planes == 0`), `planes > 4` or
/// a `channels` value other than 1 or 3 are a no-op; short buffers are
/// handled by clamped iteration (no panic, no out-of-bounds access).
#[allow(clippy::too_many_arguments)]
pub fn pfm_unpack(
    readbuf: &[f32],
    image: &mut [f32],
    width: usize,
    height: usize,
    planes: usize,
    channels: usize,
    swap_byte_order: bool,
    made_by_photoshop: bool,
) {
    if width == 0 || height == 0 || planes == 0 || planes > 4 {
        return;
    }
    if channels != 1 && channels != 3 {
        return;
    }
    // clamped iteration: cap the rows at what fits in each buffer at its
    // stride, then cap the columns per row (no-op for well-formed callers,
    // which allocate exactly width*height*channels / width*height*planes)
    let src_pixels = readbuf.len() / channels;
    let out_pixels = image.len() / planes;
    let rows = height.min(src_pixels / width).min(out_pixels / width);
    for row in 0..rows {
        // PFM de facto standard is bottom-to-top; Photoshop writes
        // top-to-bottom. `rows` (not `height`) only differs when the
        // clamps engaged on a short buffer, where behaviour is
        // unconstrained anyway.
        let target_row = if made_by_photoshop { row } else { rows - 1 - row };
        let src_base = target_row * width;
        let out_base = row * width;
        let cols = width
            .min(src_pixels - src_base)
            .min(out_pixels - out_base);
        for column in 0..cols {
            match channels {
                3 => {
                    let s = src_base + column;
                    // the C zero-initialised dt_aligned_pixel_t: with
                    // planes == 4 the fourth output channel stays 0.0
                    let pix = [
                        maybe_swap(readbuf[3 * s], swap_byte_order),
                        maybe_swap(readbuf[3 * s + 1], swap_byte_order),
                        maybe_swap(readbuf[3 * s + 2], swap_byte_order),
                        0.0f32,
                    ];
                    for c in 0..planes {
                        image[planes * (out_base + column) + c] = pix[c];
                    }
                }
                _ => {
                    let v = maybe_swap(readbuf[src_base + column], swap_byte_order);
                    for c in 0..planes {
                        image[planes * (out_base + column) + c] = v;
                    }
                }
            }
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `pfm_unpack`: walks the OUTPUT
/// buffer pixel-by-pixel via `chunks_exact_mut` (deriving each source index
/// from the output coordinates) and finishes each pixel with
/// `copy_from_slice`, where the kernel walks source rows with explicit
/// index writes. Same clamped-fit precondition: well-formed buffers only.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ref_pfm_unpack(
    readbuf: &[f32],
    image: &mut [f32],
    width: usize,
    height: usize,
    planes: usize,
    channels: usize,
    swap_byte_order: bool,
    made_by_photoshop: bool,
) {
    if width == 0 || height == 0 || planes == 0 || planes > 4 {
        return;
    }
    if channels != 1 && channels != 3 {
        return;
    }
    let Some(npix) = width.checked_mul(height) else {
        return;
    };
    if readbuf.len() < channels.saturating_mul(npix) || image.len() < planes.saturating_mul(npix) {
        return;
    }
    for (p, out_pix) in image.chunks_exact_mut(planes).enumerate().take(npix) {
        let row = p / width;
        let column = p % width;
        let target_row = if made_by_photoshop { row } else { height - 1 - row };
        let src_pixel = target_row * width + column;
        let mut pix = [0.0f32; 4];
        if channels == 3 {
            for (c, slot) in pix.iter_mut().enumerate().take(3) {
                let bits = readbuf[3 * src_pixel + c].to_bits();
                *slot = f32::from_bits(if swap_byte_order {
                    bits.swap_bytes()
                } else {
                    bits
                });
            }
        } else {
            let bits = readbuf[src_pixel].to_bits();
            let v = f32::from_bits(if swap_byte_order {
                bits.swap_bytes()
            } else {
                bits
            });
            for slot in pix.iter_mut().take(planes) {
                *slot = v;
            }
        }
        out_pix.copy_from_slice(&pix[..planes]);
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `readbuf` must hold at least `channels * width * height` floats and
/// `image` at least `planes * width * height` floats (the C caller
/// allocates exactly that). `swap_byte_order` and `made_by_photoshop` are
/// C ints used as booleans (nonzero = true, matching `gboolean` usage).
#[no_mangle]
pub unsafe extern "C" fn darkroom_pfm_unpack(
    readbuf: *const f32,
    image: *mut f32,
    width: usize,
    height: usize,
    planes: usize,
    channels: usize,
    swap_byte_order: i32,
    made_by_photoshop: i32,
) {
    if readbuf.is_null()
        || image.is_null()
        || width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
        // validate channels/planes BEFORE multiplying them into the slice
        // lengths below (a misuse caller could otherwise overflow the
        // product; the safe kernel re-checks defensively)
        || (channels != 1 && channels != 3)
        || planes == 0
        || planes > 4
    {
        return;
    }
    // with width, height <= i32::MAX and channels/planes validated above,
    // the exact slice lengths the C caller allocates fit usize without
    // overflow
    let src_len = channels * width * height;
    let out_len = planes * width * height;
    let readbuf = std::slice::from_raw_parts(readbuf, src_len);
    let image = std::slice::from_raw_parts_mut(image, out_len);
    pfm_unpack(
        readbuf,
        image,
        width,
        height,
        planes,
        channels,
        swap_byte_order != 0,
        made_by_photoshop != 0,
    );
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    // 2x2 RGB source: pixel (r, c) gets the three consecutive floats
    // 3*(2r+c)+1 .. +3, so every slot is distinguishable.
    fn rgb_src() -> Vec<f32> {
        (1..=12).map(|v| v as f32).collect()
    }

    #[test]
    fn rgb_row_flip_pin() {
        // photoshop=false: PFM default bottom-to-top, output rows are the
        // source rows reversed (planes == 3 → straight copy per pixel).
        let src = rgb_src();
        let mut out = vec![0.0f32; 2 * 2 * 3];
        pfm_unpack(&src, &mut out, 2, 2, 3, 3, false, false);
        assert_eq!(out, vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rgb_no_flip_pin() {
        // photoshop=true: rows keep file order; planes == 3 == channels →
        // output equals input exactly.
        let src = rgb_src();
        let mut out = vec![0.0f32; 2 * 2 * 3];
        pfm_unpack(&src, &mut out, 2, 2, 3, 3, false, true);
        assert_eq!(out, src);
    }

    #[test]
    fn rgb_byte_swap_pin() {
        // swap_byte_order swaps the raw 32-bit patterns: 1.0 = 0x3F800000 →
        // 0x0000803F, -2.5 = 0xC0200000 → 0x000020C0. Exact bit comparison.
        let src = [1.0f32, -2.5, 0.5];
        let mut out = vec![0.0f32; 3];
        pfm_unpack(&src, &mut out, 1, 1, 3, 3, true, true);
        let expect: Vec<u32> = src
            .iter()
            .map(|v| v.to_bits().swap_bytes())
            .collect();
        for (o, e) in out.iter().zip(expect) {
            assert_eq!(o.to_bits(), e);
        }
        // and the swapped bits are NOT the identity for these values
        assert_ne!(out[0].to_bits(), src[0].to_bits());
    }

    #[test]
    fn rgb_planes4_zero_channel_pin() {
        // planes == 4: the 4th channel comes from the zero-initialised pix
        // slot → exactly 0.0 (bit pattern 0), flipped rows on top.
        let src = rgb_src();
        let mut out = vec![1.0f32; 2 * 2 * 4]; // non-zero fill to catch no-writes
        pfm_unpack(&src, &mut out, 2, 2, 4, 3, false, false);
        assert_eq!(
            out,
            vec![
                7.0f32, 8.0, 9.0, 0.0, 10.0, 11.0, 12.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0,
                0.0
            ]
        );
    }

    #[test]
    fn mono_broadcast_pin() {
        // mono, planes == 3: single value per pixel broadcast to all three
        // planes, rows flipped (photoshop=false).
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 2 * 2 * 3];
        pfm_unpack(&src, &mut out, 2, 2, 3, 1, false, false);
        assert_eq!(out, vec![3.0f32, 3.0, 3.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn mono_planes4_broadcast_pin() {
        // mono, planes == 4: broadcast fills all four planes.
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 2 * 2 * 4];
        pfm_unpack(&src, &mut out, 2, 2, 4, 1, false, true);
        assert_eq!(
            out,
            vec![1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn mono_flip_and_swap_pin() {
        // mono with flip AND byte swap: bits exact.
        let src = [1.0f32, -2.5, 3.0, -4.0];
        let mut out = vec![0.0f32; 2 * 2 * 1];
        pfm_unpack(&src, &mut out, 2, 2, 1, 1, true, false);
        // flipped rows: [3.0, -4.0] then [1.0, -2.5], each swap_bytes'd
        let expect: Vec<u32> = [3.0f32, -4.0, 1.0, -2.5]
            .iter()
            .map(|v| v.to_bits().swap_bytes())
            .collect();
        for (o, e) in out.iter().zip(expect) {
            assert_eq!(o.to_bits(), e);
        }
    }

    #[test]
    fn nan_payload_preserved() {
        // the union pun and the Rust bit swap both preserve every bit,
        // NaN payloads included — pinned exactly.
        let nan = f32::from_bits(0x7FC1_2345);
        let src = [nan, 1.0, 2.0, 3.0, 4.0, 5.0];
        // no swap, RGB
        let mut out = vec![0.0f32; 6];
        pfm_unpack(&src, &mut out, 2, 1, 3, 3, false, true);
        assert_eq!(out[0].to_bits(), 0x7FC1_2345);
        assert!(out[0].is_nan());
        // swap, RGB: payload byte-swapped exactly. Note the swapped
        // pattern need not BE a NaN (0x4523C17F is a normal float) — the
        // contract is bit preservation, not NaN-ness, exactly like the C
        // union pun would behave.
        let mut out2 = vec![0.0f32; 6];
        pfm_unpack(&src, &mut out2, 2, 1, 3, 3, true, true);
        assert_eq!(out2[0].to_bits(), 0x7FC1_2345u32.swap_bytes());
        // a payload chosen so the swapped pattern is still a NaN:
        // 0x4523_C17F -> 0x7FC1_2345 (the swap moves the original LSB into
        // the sign+exponent byte, so the source must already carry 0x7F
        // there reversed: exponent all-ones, quiet bit set)
        let nan2 = f32::from_bits(0x4523_C17F);
        let src4 = [nan2, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out4 = vec![0.0f32; 6];
        pfm_unpack(&src4, &mut out4, 2, 1, 3, 3, true, true);
        assert_eq!(out4[0].to_bits(), 0x4523_C17Fu32.swap_bytes());
        assert!(out4[0].is_nan());
        // no swap, mono
        let mut out3 = vec![0.0f32; 2];
        pfm_unpack(&src, &mut out3, 2, 1, 1, 1, false, true);
        assert_eq!(out3[0].to_bits(), 0x7FC1_2345);
    }

    #[test]
    fn rgb_matches_reference_over_lcg() {
        for swap in [false, true] {
            let w = 13usize;
            let h = 7usize;
            let planes = 4usize;
            let mut src = vec![0.0f32; w * h * 3];
            lcg_fill(&mut src, 0x6F21, 2.0);
            let mut direct = vec![0.0f32; w * h * planes];
            let mut reference = vec![0.0f32; w * h * planes];
            pfm_unpack(&src, &mut direct, w, h, planes, 3, swap, false);
            ref_pfm_unpack(&src, &mut reference, w, h, planes, 3, swap, false);
            assert_eq!(direct, reference, "swap = {swap}");
        }
    }

    #[test]
    fn mono_matches_reference_over_lcg() {
        for swap in [false, true] {
            let w = 11usize;
            let h = 9usize;
            let planes = 3usize;
            let mut src = vec![0.0f32; w * h];
            lcg_fill(&mut src, 0x6F22, 2.0);
            let mut direct = vec![0.0f32; w * h * planes];
            let mut reference = vec![0.0f32; w * h * planes];
            pfm_unpack(&src, &mut direct, w, h, planes, 1, swap, false);
            ref_pfm_unpack(&src, &mut reference, w, h, planes, 1, swap, false);
            assert_eq!(direct, reference, "swap = {swap}");
        }
    }

    #[test]
    fn matches_reference_over_lcg_all_combos() {
        // both channels values x both flip modes x planes 1..=4, exact bits
        for &(channels, src_ch) in &[(3usize, 3usize), (1usize, 1usize)] {
            for &photoshop in &[false, true] {
                for planes in 1..=4usize {
                    let w = 5usize;
                    let h = 4usize;
                    let mut src = vec![0.0f32; w * h * src_ch];
                    lcg_fill(&mut src, 0x6F23 + channels as u32, 2.0);
                    let mut direct = vec![0.0f32; w * h * planes];
                    let mut reference = vec![0.0f32; w * h * planes];
                    pfm_unpack(&src, &mut direct, w, h, planes, channels, true, photoshop);
                    ref_pfm_unpack(&src, &mut reference, w, h, planes, channels, true, photoshop);
                    assert_eq!(direct, reference, "ch={channels} ps={photoshop} planes={planes}");
                    // sanity: with swap on and arbitrary LCG bits, output
                    // must differ from a no-swap run for most pixels
                    let mut noswap = vec![0.0f32; w * h * planes];
                    pfm_unpack(&src, &mut noswap, w, h, planes, channels, false, photoshop);
                    assert_ne!(direct, noswap);
                }
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let src = vec![1.0f32; 12];
        // zero dims
        for &(w, h, p, ch) in &[
            (0usize, 2usize, 3usize, 3usize),
            (2usize, 0usize, 3usize, 3usize),
            (2usize, 2usize, 0usize, 3usize),
            (2usize, 2usize, 5usize, 3usize), // planes > 4 (C pix OOB contract)
            (2usize, 2usize, 3usize, 2usize), // channels not in {1, 3}
            (2usize, 2usize, 3usize, 0usize),
        ] {
            let mut out = vec![9.0f32; 24];
            pfm_unpack(&src, &mut out, w, h, p, ch, false, false);
            assert_eq!(out, vec![9.0f32; 24], "w={w} h={h} p={p} ch={ch}");
        }
    }

    #[test]
    fn short_buffers_clamped_no_panic() {
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let src = vec![1.0f32; 5]; // short for 2x2x3
        let mut out = vec![0.0f32; 7]; // short for 2x2x3
        pfm_unpack(&src, &mut out, 2, 2, 3, 3, false, false);
        let src1 = vec![2.0f32; 3]; // short for 2x2x1
        let mut out1 = vec![0.0f32; 5];
        pfm_unpack(&src1, &mut out1, 2, 2, 3, 1, true, false);
    }

    #[test]
    fn ffi_round_trip() {
        for &(channels, planes) in &[(3usize, 4usize), (1usize, 3usize)] {
            let w = 9usize;
            let h = 6usize;
            let mut src = vec![0.0f32; w * h * channels];
            lcg_fill(&mut src, 0x6F24 + channels as u32, 2.0);
            let mut ffi_out = vec![0.0f32; w * h * planes];
            let mut direct_out = vec![0.0f32; w * h * planes];
            unsafe {
                darkroom_pfm_unpack(
                    src.as_ptr(),
                    ffi_out.as_mut_ptr(),
                    w,
                    h,
                    planes,
                    channels,
                    1, // nonzero int = true
                    0,
                );
            }
            pfm_unpack(&src, &mut direct_out, w, h, planes, channels, true, false);
            assert_eq!(ffi_out, direct_out, "ch={channels}");
        }
    }

    #[test]
    fn ffi_guards() {
        let src = vec![1.0f32; 12];
        let mut out = vec![7.0f32; 24];
        unsafe {
            // null pointers
            darkroom_pfm_unpack(std::ptr::null(), out.as_mut_ptr(), 2, 2, 3, 3, 0, 0);
            darkroom_pfm_unpack(src.as_ptr(), std::ptr::null_mut(), 2, 2, 3, 3, 0, 0);
            // zero dims
            darkroom_pfm_unpack(src.as_ptr(), out.as_mut_ptr(), 0, 2, 3, 3, 0, 0);
            darkroom_pfm_unpack(src.as_ptr(), out.as_mut_ptr(), 2, 0, 3, 3, 0, 0);
            // channels not in {1, 3}
            darkroom_pfm_unpack(src.as_ptr(), out.as_mut_ptr(), 2, 2, 3, 2, 0, 0);
            // planes > 4
            darkroom_pfm_unpack(src.as_ptr(), out.as_mut_ptr(), 2, 2, 5, 3, 0, 0);
            // i32::MAX caps
            darkroom_pfm_unpack(
                src.as_ptr(),
                out.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                2,
                3,
                3,
                0,
                0,
            );
            darkroom_pfm_unpack(
                src.as_ptr(),
                out.as_mut_ptr(),
                2,
                (i32::MAX as usize) + 1,
                3,
                3,
                0,
                0,
            );
        }
        assert_eq!(out, vec![7.0f32; 24]); // untouched
    }
}
