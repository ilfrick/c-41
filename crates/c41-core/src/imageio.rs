//! Kernel ported from `src/imageio/imageio.c`
//! (`dt_imageio_export_with_flags`, the 8-bit non-display-byteorder tail,
//! m4-205). The kernel replaces one whole loop body: the in-place
//! red/blue lane swap over the 8-bit RGBA export buffer.
//!
//! What the C loop does, per pixel `k` in `0..npixels` where `npixels` is
//! `processed_width * processed_height`:
//! `tmp = buf[4*k]; buf[4*k] = buf[4*k+2]; buf[4*k+2] = tmp`,
//! i.e. lanes 0 and 2 (R and B) exchange values while lanes 1 and 3 (G and
//! alpha) are never touched. The swap runs after the 8-bit pipeline has
//! produced its output in `pipe.backbuf`; when the host byte order is not
//! the display order the export formats expect the channels flipped,
//! hence this tail.
//!
//! Bit-exactness notes:
//! - The operation is a pure byte permutation, so bit-exactness holds by
//!   construction: no arithmetic, no rounding, no clamping. The kernel
//!   exchanges lanes 0 and 2 of each quad with explicit stride arithmetic
//!   (the same lane selection the C loop spells with a temporary byte);
//!   the reference reaches the same permutation through disjoint split
//!   borrows, so the sweep test cross-checks indexing as well as values.
//! - In place over disjoint quads: each quad is read and written within
//!   its own 4 bytes, so iteration order cannot matter.
//!
//! The Rust kernel is single-threaded sequential; the C loop was
//! `DT_OMP_FOR` over pixels, but each output quad reads only its own
//! input quad, so thread scheduling cannot change the result.
//!
//! m4-206 adds a second kernel in this module: `u8_to_float`, port of the
//! `!orientation` fast path of `dt_imageio_flip_buffers_ui8_to_float`
//! (same C file). The oriented stride path stays in C; see that kernel's
//! docs for the split.

/// Swap the R and B lanes of an 8-bit RGBA buffer in place.
///
/// Port of the former element-wise loop in the `bpp == 8`
/// `!display_byteorder` branch of `dt_imageio_export_with_flags`
/// (src/imageio/imageio.c): `buf` holds `4*npixels` bytes (tightly packed
/// RGBA from the export pipeline), lanes 0 and 2 of every quad are
/// exchanged, lanes 1 and 3 preserved. See the module docs for the
/// no-arithmetic fidelity point.
///
/// Degenerate `npixels == 0` is a no-op; short buffers are handled by
/// clamped iteration (no panic, no out-of-bounds access). For the
/// well-formed buffer the C caller passes the clamp never engages and the
/// behaviour is exactly the C loop's.
pub fn swap_rb_inplace(buf: &mut [u8], npixels: usize) {
    let n = npixels.min(buf.len() / 4);
    for k in 0..n {
        let b = 4 * k;
        buf.swap(b, b + 2);
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `swap_rb_inplace`: walks the
/// buffer with `chunks_exact_mut` quads and exchanges lanes 0 and 2
/// through disjoint `split_at_mut` borrows (the kernel walks with explicit
/// stride arithmetic and `slice::swap`), so the sweep test
/// cross-checks indexing as well as values. Same well-formed-buffers
/// precondition, enforced here by early return rather than clamping.
#[cfg(test)]
fn ref_swap_rb_inplace(buf: &mut [u8], npixels: usize) {
    let Some(need) = npixels.checked_mul(4) else {
        return;
    };
    if buf.len() < need {
        return;
    }
    for quad in buf.chunks_exact_mut(4).take(npixels) {
        let (head, tail) = quad.split_at_mut(2);
        std::mem::swap(&mut head[0], &mut tail[0]);
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `buf` must hold at least `4 * npixels` bytes (the C caller passes
/// `pipe.backbuf` for a `processed_width * processed_height` 8-bit RGBA
/// image). The buffer is swapped in place.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_swap_rb(buf: *mut u8, npixels: usize) {
    if buf.is_null() || npixels == 0 {
        return;
    }
    // validate the product BEFORE building the slice below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(len) = npixels.checked_mul(4) else {
        return;
    };
    let buf = std::slice::from_raw_parts_mut(buf, len);
    swap_rb_inplace(buf, npixels);
}

// ── 8-bit to float normalise (m4-206) ────────────────────────────────────────

/// Safe 8-bit to float normalisation kernel.
///
/// Port of the `!orientation` fast-path `DT_OMP_FOR` row loop of
/// `dt_imageio_flip_buffers_ui8_to_float` (src/imageio/imageio.c): per
/// pixel `(row, col)` and lane `k < ch`,
/// `out[4*(row*wd+col)+k] = (inp[row*stride+ch*col+k] as f32 - black) * scale`
/// with `scale = 1/(white-black)` computed once up front, exactly as the C
/// does (`const float scale = 1.0f / (white - black)` outside the loop, so
/// the single division cannot drift per element).
///
/// Fidelity notes:
/// - `u8 as f32` is exact, so the subtraction and multiply replay the C's
///   usual-arithmetic-conversion promotion bit for bit.
/// - Output lanes `ch..4` of each quad are never written, matching the C
///   loop (which only stores lanes below `ch`); callers must not expect
///   them zeroed. Input row padding (`stride > ch*wd`) is skipped.
/// - `ch` is 1..=4 by contract: the C loop would scribble past the RGBA
///   quad for larger values, so the FFI wrapper refuses those instead of
///   reproducing the overflow. Dims are non-zero (validated by the FFI
///   wrapper; `debug_assert`ed here).
/// - The C loop is `DT_OMP_FOR` over rows, but each output float reads
///   only its own input byte, so sequential iteration is identical.
pub fn u8_to_float(
    out: &mut [f32],
    inp: &[u8],
    black: f32,
    white: f32,
    ch: usize,
    wd: usize,
    ht: usize,
    stride: usize,
) {
    debug_assert!(wd > 0 && ht > 0 && stride > 0);
    debug_assert!((1..=4).contains(&ch));
    debug_assert!(out.len() >= 4 * wd * ht);
    debug_assert!(inp.len() >= (ht - 1) * stride + ch * wd);

    let scale = 1.0f32 / (white - black);
    for j in 0..ht {
        for i in 0..wd {
            for k in 0..ch {
                out[4 * (j * wd + i) + k] = (inp[j * stride + ch * i + k] as f32 - black) * scale;
            }
        }
    }
}

/// Structurally divergent reference for `u8_to_float`: same single-scale
/// setup, but a flat `while` pixel walk (`row = n / wd`, `col = n % wd`)
/// instead of the kernel's nested row/col/lane `for` loops, so the sweep
/// test cross-checks traversal as well as values. Must return
/// bit-identical output to [`u8_to_float`] (compared with `to_bits`, so
/// even NaN payloads from a `white == black` caller must agree).
#[cfg(test)]
fn ref_u8_to_float(
    out: &mut [f32],
    inp: &[u8],
    black: f32,
    white: f32,
    ch: usize,
    wd: usize,
    ht: usize,
    stride: usize,
) {
    let scale = 1.0f32 / (white - black);
    let mut n = 0usize;
    while n < wd * ht {
        let j = n / wd;
        let i = n % wd;
        let mut k = 0usize;
        while k < ch {
            out[4 * n + k] = (inp[j * stride + ch * i + k] as f32 - black) * scale;
            k += 1;
        }
        n += 1;
    }
}

/// 8-bit to float normalisation for the no-orientation import path.
///
/// Replaces the `!orientation` fast-path loop of
/// `dt_imageio_flip_buffers_ui8_to_float` (src/imageio/imageio.c); the C
/// wrapper keeps its signature and the oriented stride path, so the single
/// `imageio_jpeg.c` caller is unchanged. `black`/`white` are the C
/// parameters (the scale is derived inside, one IEEE division, exactly as
/// the C's hoisted `const float scale`).
///
/// `out` must hold at least `4*wd*ht` floats, `inp` at least
/// `(ht-1)*stride + ch*wd` bytes. Null pointers, non-positive dims, `ch`
/// outside 1..=4, and overflowing dim products are guarded no-ops that
/// never touch memory.
///
/// # Safety
/// The buffers must hold the documented lengths; the wrapper validates
/// the products with checked arithmetic (plus an `isize::MAX` cap before
/// building the slices) but takes the lengths themselves on trust, matching
/// the module's `darkroom_imageio_swap_rb` contract.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_u8_to_float(
    out: *mut f32,
    inp: *const u8,
    black: f32,
    white: f32,
    ch: i32,
    wd: i32,
    ht: i32,
    stride: i32,
) {
    if out.is_null() || inp.is_null() {
        return;
    }
    if wd <= 0 || ht <= 0 || stride <= 0 || !(1..=4).contains(&ch) {
        return;
    }
    let (wdu, htu, chu, strideu) = (wd as usize, ht as usize, ch as usize, stride as usize);
    let need_out = match wdu.checked_mul(htu).and_then(|p| p.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    // Last input byte read is (ht-1)*stride + ch*wd - 1, hence +1 for the
    // length; ht >= 1 here so ht - 1 cannot underflow.
    let need_in = match (htu - 1)
        .checked_mul(strideu)
        .and_then(|b| chu.checked_mul(wdu).and_then(|r| b.checked_add(r)))
    {
        Some(n) => n,
        None => return,
    };
    // Slices can never span more than isize::MAX bytes; bail before
    // building one (debug builds abort on the from_raw_parts precondition
    // otherwise). need_out counts f32 lanes, so its byte span is 4x.
    let out_bytes = match need_out.checked_mul(4) {
        Some(n) => n,
        None => return,
    };
    if out_bytes > isize::MAX as usize || need_in > isize::MAX as usize {
        return;
    }
    let out_slice = std::slice::from_raw_parts_mut(out, need_out);
    let inp_slice = std::slice::from_raw_parts(inp, need_in);
    u8_to_float(out_slice, inp_slice, black, white, chu, wdu, htu, strideu);
}

fn mono_rgbx_lengths(width: usize, height: usize) -> Option<(usize, usize)> {
    if width == 0 || height == 0 {
        return None;
    }
    let npixels = width.checked_mul(height)?;
    let span = npixels.checked_mul(4)?.checked_sub(1)?;
    if span > isize::MAX as usize {
        return None;
    }
    Some((npixels, span))
}

pub fn has_mono_rgbx(inp: &[u8], width: usize, height: usize) -> bool {
    let Some((npixels, span)) = mono_rgbx_lengths(width, height) else {
        return false;
    };
    if inp.len() < span {
        return false;
    }
    for k in 0..npixels {
        let b = 4 * k;
        if inp[b] != inp[b + 1] || inp[b] != inp[b + 2] {
            return false;
        }
    }
    true
}

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_has_mono_rgbx(
    inp: *const u8,
    width: i32,
    height: i32,
) -> std::ffi::c_int {
    if inp.is_null() || width <= 0 || height <= 0 {
        return 0;
    }
    let Some((npixels, _)) = mono_rgbx_lengths(width as usize, height as usize) else {
        return 0;
    };
    for k in 0..npixels {
        let pixel = inp.add(4 * k);
        let r = pixel.read();
        let g = pixel.add(1).read();
        let b = pixel.add(2).read();
        if r != g || r != b {
            return 0;
        }
    }
    1
}

fn unoriented_lengths(
    bpp: usize,
    wd: usize,
    ht: usize,
    stride: usize,
) -> Option<(usize, usize, usize)> {
    if bpp == 0 || wd == 0 || ht == 0 {
        return None;
    }
    let row_bytes = bpp.checked_mul(wd)?;
    let out_bytes = row_bytes.checked_mul(ht)?;
    let in_bytes = (ht - 1).checked_mul(stride)?.checked_add(row_bytes)?;
    if out_bytes > isize::MAX as usize || in_bytes > isize::MAX as usize {
        return None;
    }
    Some((row_bytes, out_bytes, in_bytes))
}

/// Copy `bpp * wd` bytes per row from strided input into packed output (m4-224).
/// Strides smaller than a row are supported; zero stride repeats the source row.
/// Zero dimensions or bpp, overflowing or over-isize spans, and short buffers
/// are no-ops. Only the last row's payload, not its padding, is needed.
pub fn flip_buffers_unoriented(
    out: &mut [u8],
    inp: &[u8],
    bpp: usize,
    wd: usize,
    ht: usize,
    stride: usize,
) {
    let Some((row_bytes, out_bytes, in_bytes)) = unoriented_lengths(bpp, wd, ht, stride) else {
        return;
    };
    if out.len() < out_bytes || inp.len() < in_bytes {
        return;
    }
    for (j, row) in out[..out_bytes].chunks_exact_mut(row_bytes).enumerate() {
        row.copy_from_slice(&inp[j * stride..j * stride + row_bytes]);
    }
}

/// Byte-row copy for `dt_imageio_flip_buffers`'s `!orientation` branch.
/// Null pointers, non-positive dimensions, zero bpp, negative stride with
/// multiple rows, overflowing or over-isize actual spans are no-ops. Zero
/// stride repeats the source row; any stride is ignored for a single row.
/// Output is packed; rows run sequentially with the same bytes as the C loop.
///
/// # Safety
/// For accepted arguments, `out` must provide `bpp * wd * ht` writable bytes
/// within one allocation, exclusively accessible during the call. `inp` must
/// span `(ht - 1) * stride + bpp * wd` bytes within one allocation; each row's
/// `bpp * wd` payload must be valid for reads, but need not be initialized.
/// The spans must not overlap. Raw copies preserve initialization state without
/// references; neither output nor input padding needs initialization. No final
/// padding is accessed; the input span is just `bpp * wd` for a single row.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_flip_buffers_unoriented(
    out: *mut std::ffi::c_char,
    inp: *const std::ffi::c_char,
    bpp: usize,
    wd: std::ffi::c_int,
    ht: std::ffi::c_int,
    stride: std::ffi::c_int,
) {
    if out.is_null() || inp.is_null() || wd <= 0 || ht <= 0 || (stride < 0 && ht > 1) {
        return;
    }
    let stride = if ht == 1 { 0 } else { stride as usize };
    let (wd, ht) = (wd as usize, ht as usize);
    let Some((row_bytes, _, _)) = unoriented_lengths(bpp, wd, ht, stride) else {
        return;
    };
    for j in 0..ht {
        std::ptr::copy_nonoverlapping(
            inp.cast::<u8>().add(j * stride),
            out.cast::<u8>().add(j * row_bytes),
            row_bytes,
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_has_mono_rgbx(inp: &[u8], width: usize, height: usize) -> bool {
        let mut colored = 0;
        for row in inp.chunks(4 * width).take(height) {
            for pixel in row.chunks(4) {
                colored += usize::from(pixel[..3].windows(2).any(|pair| pair[0] != pair[1]));
            }
        }
        colored == 0
    }

    fn assert_mono_rgbx(inp: &[u8], width: usize, height: usize, expected: bool) {
        assert_eq!(has_mono_rgbx(inp, width, height), expected);
        assert_eq!(
            unsafe { darkroom_imageio_has_mono_rgbx(inp.as_ptr(), width as i32, height as i32) },
            std::ffi::c_int::from(expected)
        );
    }

    #[test]
    fn mono_rgbx_matches_reference_over_sweep() {
        let mut state = 0x9E37_79B9u32;
        for (width, height) in [(1, 1), (1, 33), (35, 1), (3, 5), (31, 32), (32, 32), (37, 41)] {
            for mode in 0..3 {
                let mut inp = vec![0u8; 4 * width * height];
                for pixel in inp.chunks_exact_mut(4) {
                    for byte in pixel.iter_mut() {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        *byte = (state >> 24) as u8;
                    }
                    if mode != 0 {
                        pixel[1] = pixel[0];
                        pixel[2] = pixel[0];
                    }
                }
                if mode == 2 {
                    let k = state as usize % (width * height);
                    inp[4 * k + 1] ^= 1;
                }
                let original = inp.clone();
                let expected = ref_has_mono_rgbx(&inp, width, height);
                assert_eq!(expected, mode == 1);
                assert_mono_rgbx(&inp, width, height, expected);
                assert_eq!(inp, original);
            }
        }
    }

    #[test]
    fn mono_rgbx_every_pixel_and_channel_plus_minus_one() {
        for (width, height) in [(1, 1), (1, 5), (5, 1), (7, 5), (32, 33)] {
            let mut inp = vec![128u8; 4 * width * height];
            for k in 0..width * height {
                for channel in 0..3 {
                    for value in [127, 129] {
                        inp[4 * k + channel] = value;
                        assert_mono_rgbx(&inp, width, height, false);
                        inp[4 * k + channel] = 128;
                    }
                }
            }
            assert_mono_rgbx(&inp, width, height, true);
        }
    }

    #[test]
    fn mono_rgbx_varying_x_and_exact_span() {
        for gray in [0, 1, 127, 128, 254, 255] {
            for x in 0..=255 {
                assert_mono_rgbx(&[gray, gray, gray, x], 1, 1, true);
            }
            for (width, height) in [(1, 1), (1, 7), (9, 1), (3, 5), (32, 32)] {
                let mut inp = vec![gray; 4 * width * height - 1].into_boxed_slice();
                for (k, byte) in inp.iter_mut().skip(3).step_by(4).enumerate() {
                    *byte = k as u8;
                }
                assert_mono_rgbx(&inp, width, height, true);
                let last = inp.len() - 1;
                inp[last] ^= 1;
                assert_mono_rgbx(&inp, width, height, false);
            }
        }
        assert_mono_rgbx(&[8, 8, 8, 255, 1, 2, 3], 1, 1, true);
    }

    #[test]
    fn mono_rgbx_ffi_uninitialized_x() {
        use std::mem::MaybeUninit;

        for (width, height) in [(1, 1), (3, 5), (32, 33)] {
            for omit_last_x in [false, true] {
                let span = 4 * width * height - usize::from(omit_last_x);
                let mut inp = vec![MaybeUninit::<u8>::uninit(); span].into_boxed_slice();
                for k in 0..width * height {
                    for channel in 0..3 {
                        inp[4 * k + channel].write(k as u8);
                    }
                }
                assert_eq!(
                    unsafe {
                        darkroom_imageio_has_mono_rgbx(inp.as_ptr().cast(), width as i32, height as i32)
                    },
                    1
                );
                let last = 4 * (width * height - 1) + 2;
                inp[last].write(((width * height - 1) as u8) ^ 1);
                assert_eq!(
                    unsafe {
                        darkroom_imageio_has_mono_rgbx(inp.as_ptr().cast(), width as i32, height as i32)
                    },
                    0
                );
            }
        }
    }

    #[test]
    fn mono_rgbx_ffi_short_circuits_colored_pixel() {
        use std::mem::MaybeUninit;

        let mut inp = [MaybeUninit::<u8>::uninit(); 15];
        inp[0].write(8);
        inp[1].write(9);
        inp[2].write(8);
        assert_eq!(
            unsafe { darkroom_imageio_has_mono_rgbx(inp.as_ptr().cast(), 2, 2) },
            0
        );
    }

    #[test]
    fn mono_rgbx_safe_invalid_and_short_buffers() {
        for (width, height) in [
            (0, 1),
            (1, 0),
            (usize::MAX, 2),
            (2, usize::MAX),
            (usize::MAX / 4 + 1, 1),
            (isize::MAX as usize / 4 + 2, 1),
        ] {
            assert!(!has_mono_rgbx(&[7; 16], width, height));
            assert_eq!(mono_rgbx_lengths(width, height), None);
        }
        for len in 0..15 {
            assert!(!has_mono_rgbx(&[7; 16][..len], 2, 2));
        }
        assert!(has_mono_rgbx(&[7; 15], 2, 2));
    }

    #[test]
    fn mono_rgbx_span_boundary() {
        let limit = isize::MAX as usize;
        let npixels = limit / 4 + 1;
        assert_eq!(mono_rgbx_lengths(npixels, 1), Some((npixels, limit)));
        assert_eq!(mono_rgbx_lengths(1, npixels), Some((npixels, limit)));
        assert_eq!(mono_rgbx_lengths(npixels + 1, 1), None);
        assert_eq!(mono_rgbx_lengths(usize::MAX / 4 + 1, 1), None);
    }

    #[test]
    fn mono_rgbx_ffi_null_dimension_and_overflow_guards() {
        let inp = [7u8; 3];
        assert_eq!(
            unsafe { darkroom_imageio_has_mono_rgbx(std::ptr::null(), 1, 1) },
            0
        );
        for (width, height) in [
            (0, 1),
            (1, 0),
            (-1, 1),
            (1, -1),
            (i32::MIN, 1),
            (1, i32::MIN),
            (i32::MAX, i32::MAX),
        ] {
            assert_eq!(
                unsafe { darkroom_imageio_has_mono_rgbx(inp.as_ptr(), width, height) },
                0
            );
        }
        if usize::BITS == 32 {
            for (width, height) in [(i32::MAX, 1), (1, i32::MAX), (65536, 65536)] {
                assert_eq!(
                    unsafe { darkroom_imageio_has_mono_rgbx(inp.as_ptr(), width, height) },
                    0
                );
            }
        }
        assert_eq!(inp, [7; 3]);
    }

    fn ref_flip_buffers_unoriented(
        out: &mut [u8],
        inp: &[u8],
        bpp: usize,
        wd: usize,
        ht: usize,
        stride: usize,
    ) {
        let row_bytes = bpp * wd;
        for (i, byte) in out.iter_mut().take(row_bytes * ht).enumerate() {
            *byte = inp[(i / row_bytes) * stride + i % row_bytes];
        }
    }

    #[test]
    fn unoriented_padded_stride_exact_touched_length() {
        for bpp in [1, 3, 4, 16] {
            let (wd, ht) = (3, 4);
            let row_bytes = bpp * wd;
            let stride = row_bytes + 5;
            let in_bytes = (ht - 1) * stride + row_bytes;
            let out_bytes = row_bytes * ht;
            let mut inp = vec![0xAD; in_bytes + 2];
            let mut expected = Vec::new();
            for j in 0..ht {
                for k in 0..row_bytes {
                    let byte = ((j * row_bytes + k) % 256) as u8;
                    inp[1 + j * stride + k] = byte;
                    expected.push(byte);
                }
            }
            let original = inp.clone();
            let mut out = vec![0xE7; out_bytes + 2];
            unsafe {
                darkroom_imageio_flip_buffers_unoriented(
                    out.as_mut_ptr().add(1).cast(),
                    inp.as_ptr().add(1).cast(),
                    bpp,
                    wd as i32,
                    ht as i32,
                    stride as i32,
                );
            }
            assert_eq!(&out[1..1 + out_bytes], expected);
            assert_eq!(out[0], 0xE7);
            assert_eq!(out[out_bytes + 1], 0xE7);
            assert_eq!(inp, original);
            out.fill(0xE7);
            flip_buffers_unoriented(
                &mut out[1..1 + out_bytes],
                &inp[1..1 + in_bytes],
                bpp,
                wd,
                ht,
                stride,
            );
            assert_eq!(&out[1..1 + out_bytes], expected);
            assert_eq!(out[0], 0xE7);
            assert_eq!(out[out_bytes + 1], 0xE7);
        }
    }

    #[test]
    fn unoriented_matches_reference_over_sweep() {
        for (wd, ht) in [(1, 1), (3, 1), (1, 4), (5, 3), (17, 9)] {
            for bpp in [1, 2, 3, 4, 16] {
                for stride in [0, 1, bpp * wd, bpp * wd + 1, bpp * wd + 13] {
                    let inp: Vec<u8> = (0..(ht - 1) * stride + bpp * wd)
                        .map(|i| ((i as u64 * 2_654_435_761 + 0x9E37) % 256) as u8)
                        .collect();
                    let mut expected = vec![0; bpp * wd * ht];
                    ref_flip_buffers_unoriented(&mut expected, &inp, bpp, wd, ht, stride);
                    let mut direct = vec![0xCD; expected.len()];
                    let mut ffi = direct.clone();
                    flip_buffers_unoriented(&mut direct, &inp, bpp, wd, ht, stride);
                    unsafe {
                        darkroom_imageio_flip_buffers_unoriented(
                            ffi.as_mut_ptr().cast(),
                            inp.as_ptr().cast(),
                            bpp,
                            wd as i32,
                            ht as i32,
                            stride as i32,
                        );
                    }
                    assert_eq!(direct, expected, "safe: {bpp}/{wd}/{ht}/{stride}");
                    assert_eq!(ffi, expected, "ffi: {bpp}/{wd}/{ht}/{stride}");
                }
            }
        }
    }

    #[test]
    fn unoriented_single_row_ignores_signed_stride() {
        let inp = [1u8, 2, 3, 4, 5, 6];
        for stride in [i32::MIN, -1, 0, 1, i32::MAX] {
            let mut out = [0u8; 6];
            unsafe {
                darkroom_imageio_flip_buffers_unoriented(
                    out.as_mut_ptr().cast(),
                    inp.as_ptr().cast(),
                    2,
                    3,
                    1,
                    stride,
                );
            }
            assert_eq!(out, inp);
        }
        let mut out = [0u8; 6];
        flip_buffers_unoriented(&mut out, &inp, 2, 3, 1, usize::MAX);
        assert_eq!(out, inp);
    }

    #[test]
    fn unoriented_actual_span_boundary() {
        let limit = isize::MAX as usize;
        let stride = limit - 1;
        assert!(stride.checked_mul(2).unwrap() > limit);
        assert_eq!(unoriented_lengths(1, 1, 2, stride), Some((1, 2, limit)));
        assert_eq!(unoriented_lengths(1, 1, 2, limit), None);
        assert_eq!(unoriented_lengths(1, 1, 1, usize::MAX), Some((1, 1, 1)));
    }

    #[test]
    fn unoriented_ffi_uninitialized_storage() {
        use std::mem::MaybeUninit;

        for bpp in [1, 2, 3, 4, 16] {
            let (wd, ht) = (3, 4);
            let row_bytes = bpp * wd;
            let stride = row_bytes + 5;
            let in_bytes = (ht - 1) * stride + row_bytes;
            let mut inp = vec![MaybeUninit::<u8>::uninit(); in_bytes];
            let mut out = vec![MaybeUninit::<u8>::uninit(); row_bytes * ht];
            let mut expected = Vec::new();
            for j in 0..ht {
                for k in 0..row_bytes {
                    let byte = ((j * row_bytes + k) % 256) as u8;
                    inp[j * stride + k].write(byte);
                    expected.push(byte);
                }
            }
            unsafe {
                darkroom_imageio_flip_buffers_unoriented(
                    out.as_mut_ptr().cast(),
                    inp.as_ptr().cast(),
                    bpp,
                    wd as i32,
                    ht as i32,
                    stride as i32,
                );
            }
            for (byte, expected) in out.iter().zip(expected) {
                assert_eq!(unsafe { byte.assume_init() }, expected);
            }
        }
    }

    #[test]
    fn unoriented_safe_degenerate_and_short_buffers() {
        let inp = [9; 16];
        let mut out = [7; 16];
        for (bpp, wd, ht, stride) in [
            (0, 2, 2, 8),
            (4, 0, 2, 8),
            (4, 2, 0, 8),
            (4, 3, 2, 12),
            (4, 2, 2, 9),
            (usize::MAX, 2, 1, 1),
            (1, 1, 2, usize::MAX),
            (1, 1, 2, isize::MAX as usize),
            (isize::MAX as usize, 1, 2, 1),
        ] {
            flip_buffers_unoriented(&mut out, &inp, bpp, wd, ht, stride);
            assert_eq!(out, [7; 16]);
        }
        flip_buffers_unoriented(&mut out[..15], &inp, 4, 2, 2, 8);
        flip_buffers_unoriented(&mut out, &inp[..15], 4, 2, 2, 8);
        flip_buffers_unoriented(&mut [], &[], 1, 1, 1, 1);
        assert_eq!(out, [7; 16]);
        assert_eq!(inp, [9; 16]);
    }

    #[test]
    fn unoriented_ffi_null_and_degenerate_guards() {
        let inp = [9u8; 16];
        let mut out = [7u8; 16];
        unsafe {
            darkroom_imageio_flip_buffers_unoriented(
                std::ptr::null_mut(),
                inp.as_ptr().cast(),
                4,
                2,
                2,
                8,
            );
            darkroom_imageio_flip_buffers_unoriented(
                out.as_mut_ptr().cast(),
                std::ptr::null(),
                4,
                2,
                2,
                8,
            );
            for (bpp, wd, ht, stride) in [
                (0, 2, 2, 8),
                (4, 0, 2, 8),
                (4, -1, 2, 8),
                (4, 2, 0, 8),
                (4, 2, -1, 8),
                (4, 2, 2, -1),
            ] {
                darkroom_imageio_flip_buffers_unoriented(
                    out.as_mut_ptr().cast(),
                    inp.as_ptr().cast(),
                    bpp,
                    wd,
                    ht,
                    stride,
                );
                assert_eq!(out, [7; 16]);
            }
        }
        assert_eq!(out, [7; 16]);
        assert_eq!(inp, [9; 16]);
    }

    #[test]
    fn unoriented_overflow_guards() {
        let inp = [9u8; 1];
        let mut out = [7u8; 1];
        for (bpp, wd, ht, stride) in [
            (usize::MAX, 2, 1, 1),
            (usize::MAX / 2 + 1, 1, 2, 1),
            (isize::MAX as usize + 1, 1, 1, 1),
            (isize::MAX as usize, 1, 2, 1),
            (16, i32::MAX, i32::MAX, i32::MAX),
        ] {
            unsafe {
                darkroom_imageio_flip_buffers_unoriented(
                    out.as_mut_ptr().cast(),
                    inp.as_ptr().cast(),
                    bpp,
                    wd,
                    ht,
                    stride,
                );
            }
            assert_eq!(out, [7]);
        }
        assert_eq!(inp, [9]);
        assert_eq!(unoriented_lengths(1, 1, 2, usize::MAX), None);
        assert_eq!(unoriented_lengths(1, 1, 2, isize::MAX as usize), None);
        assert_eq!(unoriented_lengths(usize::MAX, 1, 2, 1), None);
        assert_eq!(
            unoriented_lengths(1, 1, 1, isize::MAX as usize),
            Some((1, 1, 1))
        );
        if usize::BITS == 32 {
            unsafe {
                darkroom_imageio_flip_buffers_unoriented(
                    out.as_mut_ptr().cast(),
                    inp.as_ptr().cast(),
                    1,
                    1,
                    2,
                    i32::MAX,
                );
            }
            assert_eq!(out, [7]);
        }
    }

    // known-answer pin: R and B exchange, G and alpha keep their values,
    // including a pixel whose R already equals B (swap must be a no-op
    // there, not a corruption).
    #[test]
    fn swap_exchanges_r_and_b() {
        let mut buf = vec![
            10u8, 20, 30, 40, // pixel 0: R=10 G=20 B=30 A=40
            5u8, 6, 5, 7, // pixel 1: R == B
        ];
        swap_rb_inplace(&mut buf, 2);
        assert_eq!(buf, vec![30u8, 20, 10, 40, 5u8, 6, 5, 7]);
    }

    // double application is the identity on every byte pattern: the swap
    // is its own inverse, so any deviation in lane selection fails here.
    #[test]
    fn double_swap_is_identity() {
        for n in [1usize, 3, 65] {
            let mut buf = Vec::with_capacity(4 * n);
            for i in 0..4 * n {
                // LCG over the full 0..=255 byte range
                let v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37)
                    % 256;
                buf.push(v as u8);
            }
            let original = buf.clone();
            swap_rb_inplace(&mut buf, n);
            // after one swap lanes 1 and 3 are already untouched
            for (k, quad) in buf.chunks_exact(4).enumerate() {
                assert_eq!(quad[1], original[4 * k + 1], "green lane {k}");
                assert_eq!(quad[3], original[4 * k + 3], "alpha lane {k}");
            }
            swap_rb_inplace(&mut buf, n);
            assert_eq!(buf, original);
        }
    }

    // kernel and reference agree byte-exactly over sweeps at several pixel
    // counts (well-formed buffers: both sides fully engage).
    #[test]
    fn matches_reference_over_sweep() {
        for n in [1usize, 3, 65, 513] {
            let mut src = Vec::with_capacity(4 * n);
            for i in 0..4 * n {
                // LCG over the full 0..=255 byte range
                let v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37)
                    % 256;
                src.push(v as u8);
            }
            let mut direct = src.clone();
            let mut reference = src.clone();
            swap_rb_inplace(&mut direct, n);
            ref_swap_rb_inplace(&mut reference, n);
            assert_eq!(direct, reference, "n = {n}");
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        // zero pixels: buffer untouched
        let mut buf = vec![11u8, 22, 33, 44];
        swap_rb_inplace(&mut buf, 0);
        assert_eq!(buf, vec![11u8, 22, 33, 44]);
        // truncated buffer: clamped iteration swaps only the complete
        // quad, never panics nor writes out of bounds; well-formed
        // callers never hit this path.
        let mut short = vec![1u8, 2, 3, 4, 5, 6, 7];
        swap_rb_inplace(&mut short, 2);
        assert_eq!(short, vec![3u8, 2, 1, 4, 5, 6, 7]);
        // empty buffer with nonzero count: no-op
        let mut empty: Vec<u8> = vec![];
        swap_rb_inplace(&mut empty, 1);
        assert!(empty.is_empty());
        // over-long buffer: the tail past 4*npixels is untouched
        let mut long = vec![9u8, 8, 7, 6, 100, 101, 102];
        swap_rb_inplace(&mut long, 1);
        assert_eq!(long, vec![7u8, 8, 9, 6, 100, 101, 102]);
    }

    #[test]
    fn ffi_round_trip() {
        let n = 65usize;
        let mut src = vec![0u8; 4 * n];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i as u64).wrapping_mul(2_654_435_761) % 256) as u8;
        }
        let mut ffi_buf = src.clone();
        let mut direct_buf = src.clone();
        unsafe {
            darkroom_imageio_swap_rb(ffi_buf.as_mut_ptr(), n);
        }
        swap_rb_inplace(&mut direct_buf, n);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_guards() {
        let mut buf = vec![7u8; 16];
        unsafe {
            // null pointer
            darkroom_imageio_swap_rb(std::ptr::null_mut(), 4);
            // zero pixels
            darkroom_imageio_swap_rb(buf.as_mut_ptr(), 0);
            // overflowing pixel counts (4 * npixels wraps — all reject)
            darkroom_imageio_swap_rb(buf.as_mut_ptr(), usize::MAX);
            darkroom_imageio_swap_rb(buf.as_mut_ptr(), usize::MAX / 4 + 1);
        }
        assert_eq!(buf, vec![7u8; 16]); // untouched
    }

    // helper: bit-exact float comparison (NaN payloads must agree too,
    // for the white == black degenerate sweep below).
    fn assert_bits_eq(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "float {i}");
        }
    }

    // known-answer pin: black level maps exactly to +0.0, and a mid value
    // lands where the single hoisted scale puts it (checked against the
    // reference, plus the exact-zero property the C shares).
    #[test]
    fn u8_to_float_known_answer() {
        // 2x1 RGBA, tight stride
        let inp = vec![0u8, 128, 255, 64, 16, 32, 48, 200];
        let mut out = vec![-1.0f32; 8];
        let mut reference = vec![-1.0f32; 8];
        u8_to_float(&mut out, &inp, 0.0, 255.0, 4, 2, 1, 8);
        ref_u8_to_float(&mut reference, &inp, 0.0, 255.0, 4, 2, 1, 8);
        assert_bits_eq(&out, &reference);
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[4].to_bits(), (16.0f32 * (1.0f32 / 255.0)).to_bits());
    }

    // kernel and reference agree bit-exactly over sweeps of dims, channel
    // counts, strides, and black/white points (including degenerate
    // white == black and an inverted range).
    #[test]
    fn u8_to_float_matches_reference_over_sweep() {
        for (wd, ht) in [(1usize, 1), (3, 1), (1, 4), (5, 3), (17, 9)] {
            for ch in [1usize, 2, 3, 4] {
                for pad in [0usize, 1, 3] {
                    let stride = ch * wd + pad;
                    for (black, white) in [
                        (0.0f32, 255.0f32),
                        (16.0, 235.0),
                        (0.0, 1.0),
                        (255.0, 0.0),
                        (5.0, 5.0),
                    ] {
                        let mut inp = vec![0u8; ht * stride];
                        for (i, v) in inp.iter_mut().enumerate() {
                            // LCG over the full byte range; stride padding
                            // lands in the stream too, like real rows.
                            *v = ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(0x9E37)
                                % 256) as u8;
                        }
                        let mut direct = vec![0.0f32; 4 * wd * ht];
                        let mut reference = vec![0.0f32; 4 * wd * ht];
                        u8_to_float(&mut direct, &inp, black, white, ch, wd, ht, stride);
                        ref_u8_to_float(&mut reference, &inp, black, white, ch, wd, ht, stride);
                        assert_bits_eq(&direct, &reference);
                    }
                }
            }
        }
    }

    // the C loop only stores lanes below ch: with ch == 3 the alpha lane
    // keeps whatever the output buffer held (here a sentinel).
    #[test]
    fn u8_to_float_leaves_unwritten_lanes() {
        let inp = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mut out = vec![-7.0f32; 16];
        u8_to_float(&mut out, &inp, 0.0, 255.0, 3, 1, 3, 4);
        for row in 0..3 {
            assert_eq!(out[4 * row + 3].to_bits(), (-7.0f32).to_bits(), "row {row}");
            // written lanes still match the reference
            let mut one = vec![-7.0f32; 4];
            let row_in = &inp[4 * row..4 * row + 4];
            ref_u8_to_float(&mut one, row_in, 0.0, 255.0, 3, 1, 1, 4);
            assert_bits_eq(&out[4 * row..4 * row + 3], &one[0..3]);
        }
    }

    // stride padding bytes are never read: a padded input converts
    // identically to the tight packing of the same pixels.
    #[test]
    fn u8_to_float_skips_stride_padding() {
        let wd = 3usize;
        let ht = 2usize;
        let ch = 4usize;
        let tight: Vec<u8> = (0..(ch * wd * ht) as u8).collect();
        let stride = ch * wd + 2;
        let mut padded = vec![0xABu8; ht * stride];
        for j in 0..ht {
            padded[j * stride..j * stride + ch * wd]
                .copy_from_slice(&tight[j * ch * wd..(j + 1) * ch * wd]);
        }
        let mut from_tight = vec![0.0f32; 4 * wd * ht];
        let mut from_padded = vec![0.0f32; 4 * wd * ht];
        u8_to_float(&mut from_tight, &tight, 16.0, 235.0, ch, wd, ht, ch * wd);
        u8_to_float(&mut from_padded, &padded, 16.0, 235.0, ch, wd, ht, stride);
        assert_bits_eq(&from_tight, &from_padded);
    }

    #[test]
    fn u8_to_float_ffi_round_trip() {
        let (wd, ht, ch) = (7usize, 5usize, 4usize);
        let stride = ch * wd + 1;
        let mut inp = vec![0u8; ht * stride];
        for (i, v) in inp.iter_mut().enumerate() {
            *v = ((i as u64).wrapping_mul(2_654_435_761) % 256) as u8;
        }
        let mut ffi_out = vec![0.0f32; 4 * wd * ht];
        let mut direct_out = vec![0.0f32; 4 * wd * ht];
        unsafe {
            darkroom_imageio_u8_to_float(
                ffi_out.as_mut_ptr(),
                inp.as_ptr(),
                16.0,
                235.0,
                ch as i32,
                wd as i32,
                ht as i32,
                stride as i32,
            );
        }
        u8_to_float(&mut direct_out, &inp, 16.0, 235.0, ch, wd, ht, stride);
        assert_bits_eq(&ffi_out, &direct_out);
    }

    #[test]
    fn u8_to_float_ffi_guards() {
        let inp = vec![9u8; 64];
        let mut out = vec![3.0f32; 64];
        unsafe {
            // null pointers
            darkroom_imageio_u8_to_float(std::ptr::null_mut(), inp.as_ptr(), 0.0, 255.0, 4, 2, 2, 8);
            darkroom_imageio_u8_to_float(out.as_mut_ptr(), std::ptr::null(), 0.0, 255.0, 4, 2, 2, 8);
            // degenerate dims
            darkroom_imageio_u8_to_float(out.as_mut_ptr(), inp.as_ptr(), 0.0, 255.0, 4, 0, 2, 8);
            darkroom_imageio_u8_to_float(out.as_mut_ptr(), inp.as_ptr(), 0.0, 255.0, 4, 2, 0, 8);
            darkroom_imageio_u8_to_float(out.as_mut_ptr(), inp.as_ptr(), 0.0, 255.0, 4, 2, 2, 0);
            darkroom_imageio_u8_to_float(out.as_mut_ptr(), inp.as_ptr(), 0.0, 255.0, 4, -2, 2, 8);
            // ch outside 1..=4 (0, and 5 which would scribble past the
            // RGBA quad in C)
            darkroom_imageio_u8_to_float(out.as_mut_ptr(), inp.as_ptr(), 0.0, 255.0, 0, 2, 2, 8);
            darkroom_imageio_u8_to_float(out.as_mut_ptr(), inp.as_ptr(), 0.0, 255.0, 5, 2, 2, 8);
            // overflowing dim products (4*wd*ht wraps — must reject
            // before building any slice, so the small buffers stay valid)
            darkroom_imageio_u8_to_float(
                out.as_mut_ptr(),
                inp.as_ptr(),
                0.0,
                255.0,
                4,
                i32::MAX,
                i32::MAX,
                i32::MAX,
            );
        }
        assert_eq!(out, vec![3.0f32; 64]); // untouched
        assert_eq!(inp, vec![9u8; 64]); // untouched
    }
}
