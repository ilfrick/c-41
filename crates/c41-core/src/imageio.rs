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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
