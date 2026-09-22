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
//!
//! m4-227 adds the oriented counterpart `u8_to_float_oriented`, closing the
//! last `DT_OMP_FOR` in `src/imageio/imageio.c`. It shares the m4-226
//! `oriented_layout` geometry in pixel units, scaled to float lanes.

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

// ── Oriented 8-bit to float normalise (m4-227) ───────────────────────────────

/// Safe 8-bit to float normalisation kernel for the oriented import path.
///
/// Port of the oriented `DT_OMP_FOR` row loop of
/// `dt_imageio_flip_buffers_ui8_to_float` (src/imageio/imageio.c): per
/// pixel `(i, j)` and lane `k < ch`,
/// `out[4 * dest(i, j) + k] = (inp[j * stride + ch * i + k] as f32 - black) * scale`
/// with `scale = 1.0 / (white - black)` derived once up front, exactly as
/// the C's hoisted `const float scale`. The destination pixel `dest(i, j)`
/// is the [`oriented_layout`] mapping of `flip_buffers_oriented` (m4-226)
/// evaluated in pixel units and scaled to lanes, so the orientation
/// semantics (flag values per src/common/image.h: FLIP_Y = 1, FLIP_X = 2,
/// SWAP_XY = 4) are shared, not re-derived.
///
/// Fidelity notes:
/// - `u8 as f32` is exact, so the subtraction and multiply replay the C
///   loop's promotion bit for bit; the single hoisted division cannot drift
///   per element on either side.
/// - Output lanes `ch..4` of each written quad are never stored, matching
///   the C loop (which only stores lanes below `ch`); callers must not
///   expect them zeroed. Input row padding (`stride > ch * wd`) is skipped.
/// - `ch` is 1..=4 by contract: the C loop would scribble past the RGBA
///   quad for larger values, so the kernel refuses those instead of
///   reproducing the overflow. Dims are non-zero and `fwd`/`fht` cover the
///   flipped extents (validated by the layout helper).
/// - `out` and `inp` must not overlap.
/// - Stride follows the m4-226 oriented convention: zero repeats the source
///   row (well-defined in C), a negative stride with more than one row
///   would read out of bounds in C so the FFI wrapper rejects it, and any
///   stride is accepted for a single row (the C loop never advances).
///   Short rows (`stride < ch * wd`) overlap exactly as in C.
/// - The C loop is `DT_OMP_FOR` over rows, but each output float reads only
///   its own input byte and every destination quad is written by exactly
///   one source pixel (the mapping is a permutation onto its span), so
///   sequential iteration is identical.
#[allow(clippy::too_many_arguments)]
pub fn u8_to_float_oriented(
    out: &mut [f32],
    inp: &[u8],
    black: f32,
    white: f32,
    ch: usize,
    wd: usize,
    ht: usize,
    fwd: usize,
    fht: usize,
    stride: usize,
    orientation: i32,
) {
    if !(1..=4).contains(&ch) {
        return;
    }
    let Some(layout) = oriented_layout(1, wd, ht, fwd, fht, stride, orientation) else {
        return;
    };
    let Some(out_lanes) = layout.out_bytes.checked_mul(4) else {
        return;
    };
    let Some(row_bytes) = ch.checked_mul(wd) else {
        return;
    };
    // Last input byte read is (ht - 1) * stride + ch * wd - 1, hence +1
    // for the length; ht >= 1 here so ht - 1 cannot underflow, and ht == 1
    // ignores the stride exactly as the C row loop does.
    let Some(need_in) = (ht - 1)
        .checked_mul(stride)
        .and_then(|base| base.checked_add(row_bytes))
    else {
        return;
    };
    if out.len() < out_lanes || inp.len() < need_in {
        return;
    }

    let scale = 1.0f32 / (white - black);
    for j in 0..ht {
        for i in 0..wd {
            let src = j * stride + ch * i;
            let dst = layout.destination(i, j, 1) * 4;
            for k in 0..ch {
                out[dst + k] = (inp[src + k] as f32 - black) * scale;
            }
        }
    }
}

/// Oriented 8-bit to float normalisation for the import path.
///
/// Replaces the oriented `DT_OMP_FOR` loop of
/// `dt_imageio_flip_buffers_ui8_to_float` (src/imageio/imageio.c); the C
/// wrapper keeps its signature and the `!orientation` fast path, so the
/// single `imageio_jpeg.c` caller is unchanged. `black`/`white` are the C
/// parameters (the scale is derived inside, one IEEE division, exactly as
/// the C's hoisted `const float scale`).
///
/// `out` must hold at least `4 * dest_span` floats where `dest_span` is the
/// `oriented_layout` pixel span (`wd * ht` when `fwd == wd` and
/// `fht == ht`, larger with displaced extents), `inp` at least
/// `(ht - 1) * stride + ch * wd` bytes. Null pointers, non-positive dims,
/// `ch` outside 1..=4, a negative stride with more than one row, flipped
/// extents smaller than the image, and overflowing dim products are guarded
/// no-ops that never touch memory. A zero stride repeats the source row
/// and any stride is accepted for a single row, matching the C loop.
///
/// # Safety
/// The buffers must hold the documented lengths and must not overlap; the
/// wrapper validates
/// the products with checked arithmetic (plus an `isize::MAX` cap before
/// building the slices) but takes the lengths themselves on trust, matching
/// the module's `darkroom_imageio_u8_to_float` contract.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_u8_to_float_oriented(
    out: *mut f32,
    inp: *const u8,
    black: f32,
    white: f32,
    ch: std::ffi::c_int,
    wd: std::ffi::c_int,
    ht: std::ffi::c_int,
    fwd: std::ffi::c_int,
    fht: std::ffi::c_int,
    stride: std::ffi::c_int,
    orientation: std::ffi::c_int,
) {
    if out.is_null() || inp.is_null() {
        return;
    }
    if wd <= 0 || ht <= 0 || fwd <= 0 || fht <= 0 || !(1..=4).contains(&ch) {
        return;
    }
    if stride < 0 && ht > 1 {
        return;
    }
    let strideu = if ht == 1 { 0 } else { stride as usize };
    let (wdu, htu, fwdu, fhtu, chu) = (
        wd as usize,
        ht as usize,
        fwd as usize,
        fht as usize,
        ch as usize,
    );
    let Some(layout) = oriented_layout(1, wdu, htu, fwdu, fhtu, strideu, orientation) else {
        return;
    };
    let Some(out_lanes) = layout.out_bytes.checked_mul(4) else {
        return;
    };
    let Some(row_bytes) = chu.checked_mul(wdu) else {
        return;
    };
    // Last input byte read is (ht - 1) * stride + ch * wd - 1, hence +1
    // for the length; ht >= 1 here so ht - 1 cannot underflow.
    let Some(need_in) = (htu - 1)
        .checked_mul(strideu)
        .and_then(|base| base.checked_add(row_bytes))
    else {
        return;
    };
    // Slices can never span more than isize::MAX bytes; bail before
    // building one. out_lanes counts f32 lanes, so its byte span is 4x.
    let Some(out_span) = out_lanes.checked_mul(4) else {
        return;
    };
    if out_span > isize::MAX as usize || need_in > isize::MAX as usize {
        return;
    }
    let out_slice = std::slice::from_raw_parts_mut(out, out_lanes);
    let inp_slice = std::slice::from_raw_parts(inp, need_in);
    u8_to_float_oriented(
        out_slice, inp_slice, black, white, chu, wdu, htu, fwdu, fhtu, strideu, orientation,
    );
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

#[derive(Debug, PartialEq)]
struct OrientedLayout {
    x_origin: Option<usize>,
    y_origin: Option<usize>,
    x_pitch: usize,
    y_pitch: usize,
    in_bytes: usize,
    out_bytes: usize,
}

fn oriented_layout(
    bpp: usize,
    wd: usize,
    ht: usize,
    fwd: usize,
    fht: usize,
    stride: usize,
    orientation: i32,
) -> Option<OrientedLayout> {
    if bpp == 0 || wd == 0 || ht == 0 || fwd == 0 || fht == 0 {
        return None;
    }
    let x_origin = if orientation & 2 != 0 {
        fwd.checked_sub(wd)?;
        Some(fwd - 1)
    } else {
        None
    };
    let y_origin = if orientation & 1 != 0 {
        fht.checked_sub(ht)?;
        Some(fht - 1)
    } else {
        None
    };
    let (x_pitch, y_pitch) = if orientation & 4 != 0 {
        (ht, 1)
    } else {
        (1, wd)
    };
    let out_bytes = x_origin
        .unwrap_or(wd - 1)
        .checked_mul(x_pitch)?
        .checked_add(y_origin.unwrap_or(ht - 1).checked_mul(y_pitch)?)?
        .checked_add(1)?
        .checked_mul(bpp)?;
    let in_bytes = (ht - 1)
        .checked_mul(stride)?
        .checked_add(wd.checked_mul(bpp)?)?;
    if out_bytes > isize::MAX as usize || in_bytes > isize::MAX as usize {
        return None;
    }
    Some(OrientedLayout {
        x_origin,
        y_origin,
        x_pitch,
        y_pitch,
        in_bytes,
        out_bytes,
    })
}

impl OrientedLayout {
    fn destination(&self, i: usize, j: usize, bpp: usize) -> usize {
        let x = self.x_origin.map_or(i, |origin| origin - i);
        let y = self.y_origin.map_or(j, |origin| origin - j);
        (x * self.x_pitch + y * self.y_pitch) * bpp
    }
}

#[allow(clippy::too_many_arguments)]
pub fn flip_buffers_oriented(
    out: &mut [u8],
    inp: &[u8],
    bpp: usize,
    wd: usize,
    ht: usize,
    fwd: usize,
    fht: usize,
    stride: usize,
    orientation: i32,
) {
    let Some(layout) = oriented_layout(bpp, wd, ht, fwd, fht, stride, orientation) else {
        return;
    };
    if out.len() < layout.out_bytes || inp.len() < layout.in_bytes {
        return;
    }
    for j in 0..ht {
        for i in 0..wd {
            let src = j * stride + i * bpp;
            let dst = layout.destination(i, j, bpp);
            out[dst..dst + bpp].copy_from_slice(&inp[src..src + bpp]);
        }
    }
}

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_flip_buffers_oriented(
    out: *mut std::ffi::c_char,
    inp: *const std::ffi::c_char,
    bpp: usize,
    wd: std::ffi::c_int,
    ht: std::ffi::c_int,
    fwd: std::ffi::c_int,
    fht: std::ffi::c_int,
    stride: std::ffi::c_int,
    orientation: std::ffi::c_int,
) -> std::ffi::c_int {
    if out.is_null()
        || inp.is_null()
        || wd <= 0
        || ht <= 0
        || fwd <= 0
        || fht <= 0
        || (stride < 0 && ht > 1)
    {
        return 0;
    }
    let stride = if ht == 1 { 0 } else { stride as usize };
    let (wd, ht) = (wd as usize, ht as usize);
    let Some(layout) =
        oriented_layout(bpp, wd, ht, fwd as usize, fht as usize, stride, orientation)
    else {
        return 0;
    };
    for j in 0..ht {
        for i in 0..wd {
            std::ptr::copy_nonoverlapping(
                inp.cast::<u8>().add(j * stride + i * bpp),
                out.cast::<u8>().add(layout.destination(i, j, bpp)),
                bpp,
            );
        }
    }
    1
}

// ── PNM PGM 8-bit gray-row normalize (m4-228) ────────────────────────────────

/// Safe 8-bit gray-row to float RGBA kernel.
///
/// Port of the inner per-pixel loop of the `max <= 255` branch of `_read_pgm`
/// (src/imageio/imageio_pnm.c): per column `x < width`,
/// `value = line[x] as f32 / max as f32`, then `out[4*x + c] = value` for
/// `c` in 0..2 and `out[4*x + 3] = 0.0` (the explicitly zeroed alpha lane).
/// The row `fread`, the line allocation, and the PBM/PPM/16-bit sibling
/// branches stay in C.
///
/// Fidelity notes:
/// - The kernel must stay a division: `byte as f32 / max as f32` replays
///   the C usual-arithmetic-conversion promotion bit for bit, while
///   `byte as f32 * (1.0 / max as f32)` can round differently (the
///   reciprocal is itself inexact). The denominator conversion is hoisted
///   once; this is bit-identical because `max` is loop-invariant, and the
///   conversion itself is exact over the contracted `1..=255` range.
/// - `max` is `1..=255` by contract (the C caller rejects `max == 0` and
///   `max > 255` before reaching this branch); a zero `max` is a guarded
///   no-op instead of reproducing a divide-by-zero.
/// - The serial C row loop has no cross-pixel dependencies, so sequential
///   iteration is identical.
///
/// Degenerate `width == 0` is a no-op; short buffers are handled by
/// clamped iteration (no panic, no out-of-bounds access). For the
/// well-formed row buffers the C caller passes the clamp never engages
/// and the behaviour is exactly the C loop's.
pub fn pnm_pgm_u8_row_to_float(out: &mut [f32], line: &[u8], width: usize, max: u32) {
    if max == 0 {
        return;
    }
    let n = width.min(line.len()).min(out.len() / 4);
    let denom = max as f32;
    for (x, &byte) in line.iter().enumerate().take(n) {
        let value = byte as f32 / denom;
        let b = 4 * x;
        out[b] = value;
        out[b + 1] = value;
        out[b + 2] = value;
        out[b + 3] = 0.0;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `pnm_pgm_u8_row_to_float`: walks the
/// OUTPUT row quad-by-quad via `as_chunks_mut` with the source byte
/// derived from the running quad index (the kernel walks source columns
/// with explicit stride writes), so the sweep test cross-checks indexing
/// as well as values. The arithmetic is the same `byte as f32 / max as f32`
/// division by construction (see the kernel docs: the reciprocal-multiply
/// form is not bit-identical and must not be used). Well-formed buffers
/// only: short inputs are an early return here rather than clamping.
#[cfg(test)]
fn ref_pnm_pgm_u8_row_to_float(line: &[u8], out: &mut [f32], width: usize, max: u32) {
    if max == 0 {
        return;
    }
    let Some(out_need) = width.checked_mul(4) else {
        return;
    };
    if line.len() < width || out.len() < out_need {
        return;
    }
    let denom = max as f32;
    let (quads, _) = out.as_chunks_mut::<4>();
    for (x, quad) in quads.iter_mut().take(width).enumerate() {
        let value = line[x] as f32 / denom;
        quad[0] = value;
        quad[1] = value;
        quad[2] = value;
        quad[3] = 0.0;
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `out` must hold at least `4 * width` floats (the C caller passes the
/// mipmap-row cursor for an `img->width` row) and `line` at least `width`
/// bytes (the row `fread` buffer). The two buffers must not overlap.
/// `max` is the PGM maxval of the `max <= 255` branch (`1..=255`).
#[no_mangle]
pub unsafe extern "C" fn darkroom_pnm_pgm_u8_row_to_float(
    out: *mut f32,
    line: *const u8,
    width: usize,
    max: std::ffi::c_uint,
) {
    if out.is_null() || line.is_null() || width == 0 || max == 0 {
        return;
    }
    // validate the product BEFORE building the slices below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(out_len) = width.checked_mul(4) else {
        return;
    };
    let out = std::slice::from_raw_parts_mut(out, out_len);
    let line = std::slice::from_raw_parts(line, width);
    pnm_pgm_u8_row_to_float(out, line, width, max);
}

// ── PNM PGM 16-bit gray-row normalize (m4-229) ───────────────────────────────

/// Safe 16-bit gray-row to float RGBA kernel.
///
/// Port of the inner per-pixel loop of the `max > 255` branch of `_read_pgm`
/// (src/imageio/imageio_pnm.c): per column `x < width`,
/// `intvalue = line[x].swap_bytes()` (PGM file order is big-endian with the
/// most significant byte first; the C loop swaps under
/// `G_BYTE_ORDER != G_BIG_ENDIAN`, and darktable supports only
/// little-endian hosts (src/imageio/imageio_heif.c:167), so the swap is
/// unconditional here — the same
/// explicit-decode stance as the m4-208/m4-209 little-endian kernels),
/// then `value = intvalue as f32 / max as f32`,
/// `out[4*x + c] = value` for `c` in 0..2 and `out[4*x + 3] = 0.0` (the
/// explicitly zeroed alpha lane). The row `fread`, the line allocation, and
/// the PBM/PPM sibling branches stay in C.
///
/// Fidelity notes:
/// - The kernel must stay a division: `decoded as f32 / max as f32` replays
///   the C usual-arithmetic-conversion promotion bit for bit, while
///   `decoded as f32 * (1.0 / max as f32)` can round differently (the
///   reciprocal is itself inexact). The denominator conversion is hoisted
///   once; this is bit-identical because `max` is loop-invariant and the
///   conversion itself is exact over the contracted `1..=65535` range
///   (every u16 and every u32 below 2^24 widens to f32 exactly), applied in
///   the same order the C performs it (numerator conversion, denominator
///   conversion, then one IEEE division per pixel).
/// - `max` is `256..=65535` by contract (the C caller rejects `max == 0`
///   and `max > 65535` before reaching `_read_pgm`, and `max <= 255` takes
///   the m4-228 u8 branch); a zero `max` is a guarded no-op instead of
///   reproducing a divide-by-zero.
/// - The serial C row loop has no cross-pixel dependencies, so sequential
///   iteration is identical.
///
/// Degenerate `width == 0` is a no-op; short buffers are handled by
/// clamped iteration (no panic, no out-of-bounds access). For the
/// well-formed row buffers the C caller passes the clamp never engages
/// and the behaviour is exactly the C loop's.
pub fn pnm_pgm_u16_row_to_float(out: &mut [f32], line: &[u16], width: usize, max: u32) {
    if max == 0 {
        return;
    }
    let n = width.min(line.len()).min(out.len() / 4);
    let denom = max as f32;
    for (x, &word) in line.iter().enumerate().take(n) {
        let value = word.swap_bytes() as f32 / denom;
        let b = 4 * x;
        out[b] = value;
        out[b + 1] = value;
        out[b + 2] = value;
        out[b + 3] = 0.0;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `pnm_pgm_u16_row_to_float`: walks the
/// OUTPUT row quad-by-quad via `as_chunks_mut` with the source word
/// derived from the running quad index (the kernel walks source columns
/// with explicit stride writes), so the sweep test cross-checks indexing
/// as well as values. The big-endian decode and the `word as f32 / max as
/// f32` division match by construction (see the kernel docs: the
/// reciprocal-multiply form is not bit-identical and must not be used).
/// Well-formed buffers only: short inputs are an early return here rather
/// than clamping.
#[cfg(test)]
fn ref_pnm_pgm_u16_row_to_float(line: &[u16], out: &mut [f32], width: usize, max: u32) {
    if max == 0 {
        return;
    }
    let Some(out_need) = width.checked_mul(4) else {
        return;
    };
    if line.len() < width || out.len() < out_need {
        return;
    }
    let denom = max as f32;
    let (quads, _) = out.as_chunks_mut::<4>();
    for (x, quad) in quads.iter_mut().take(width).enumerate() {
        let value = line[x].swap_bytes() as f32 / denom;
        quad[0] = value;
        quad[1] = value;
        quad[2] = value;
        quad[3] = 0.0;
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `out` must hold at least `4 * width` floats (the C caller passes the
/// mipmap-row cursor for an `img->width` row) and `line` at least `width`
/// u16 words (the row `fread` buffer, still in big-endian file order — the
/// kernel performs the byte swap). The two buffers must not overlap.
/// `max` is the PGM maxval of the `max > 255` branch (`256..=65535`).
#[no_mangle]
pub unsafe extern "C" fn darkroom_pnm_pgm_u16_row_to_float(
    out: *mut f32,
    line: *const u16,
    width: usize,
    max: std::ffi::c_uint,
) {
    if out.is_null() || line.is_null() || width == 0 || max == 0 {
        return;
    }
    // validate the product BEFORE building the slices below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(out_len) = width.checked_mul(4) else {
        return;
    };
    let out = std::slice::from_raw_parts_mut(out, out_len);
    let line = std::slice::from_raw_parts(line, width);
    pnm_pgm_u16_row_to_float(out, line, width, max);
}

// ── PNM PPM 8-bit RGB-triplet normalize (m4-230) ────────────────────────────

// Safe 8-bit RGB-triplet row to float RGBA kernel.
//
// Port of the inner per-pixel loop of the `max <= 255` branch of `_read_ppm`
// (src/imageio/imageio_pnm.c): per column `x < width` and lane `c < 3`,
// `out[4*x + c] = line[3*x + c] as f32 / max as f32`, then
// `out[4*x + 3] = 0.0` (the explicitly zeroed alpha lane). The row `fread`,
// the line allocation, and the PBM/PGM/16-bit sibling branches stay in C.
//
// Fidelity notes:
// - The kernel must stay a division: `byte as f32 / max as f32` replays
//   the C usual-arithmetic-conversion promotion bit for bit, while
//   `byte as f32 * (1.0 / max as f32)` can round differently (the
//   reciprocal is itself inexact). The denominator conversion is hoisted
//   once; this is bit-identical because `max` is loop-invariant, and the
//   conversion itself is exact over the contracted `1..=255` range.
// - Unlike the gray PGM sibling (m4-228) there is no fan-out: each RGB
//   output lane reads its own triplet byte, so lane interleave is pinned
//   by the tests below.
// - `max` is `1..=255` by contract (the C `_read_ppm` rejects `max == 0` and
//   `max > 65535` at entry, and `max > 255` takes the u16 triplet branch);
//   a zero `max` is a guarded no-op instead of reproducing a divide-by-zero.
// - The serial C row loop has no cross-pixel dependencies, so sequential
//   iteration is identical.
//
// Degenerate `width == 0` is a no-op; short buffers are handled by
// clamped iteration (no panic, no out-of-bounds access). For the
// well-formed row buffers the C caller passes the clamp never engages
// and the behaviour is exactly the C loop's.
pub fn pnm_ppm_u8_row_to_float(out: &mut [f32], line: &[u8], width: usize, max: u32) {
    if max == 0 {
        return;
    }
    let n = width.min(line.len() / 3).min(out.len() / 4);
    let denom = max as f32;
    for x in 0..n {
        let s = 3 * x;
        let b = 4 * x;
        out[b] = line[s] as f32 / denom;
        out[b + 1] = line[s + 1] as f32 / denom;
        out[b + 2] = line[s + 2] as f32 / denom;
        out[b + 3] = 0.0;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `pnm_ppm_u8_row_to_float`: walks the
/// source triplets and the output quads as zipped `as_chunks` chunk
/// iterators (the kernel walks columns with explicit stride writes), so the
/// sweep test cross-checks indexing as well as values. The arithmetic is the
/// same per-lane `byte as f32 / max as f32` division by construction (see
/// the kernel docs: the reciprocal-multiply form is not bit-identical and
/// must not be used). Well-formed buffers only: short inputs are an early
/// return here rather than clamping.
#[cfg(test)]
fn ref_pnm_ppm_u8_row_to_float(line: &[u8], out: &mut [f32], width: usize, max: u32) {
    if max == 0 {
        return;
    }
    let Some(out_need) = width.checked_mul(4) else {
        return;
    };
    let Some(line_need) = width.checked_mul(3) else {
        return;
    };
    if line.len() < line_need || out.len() < out_need {
        return;
    }
    let denom = max as f32;
    let (triplets, _) = line.as_chunks::<3>();
    let (quads, _) = out.as_chunks_mut::<4>();
    for (triplet, quad) in triplets.iter().take(width).zip(quads.iter_mut().take(width)) {
        quad[0] = triplet[0] as f32 / denom;
        quad[1] = triplet[1] as f32 / denom;
        quad[2] = triplet[2] as f32 / denom;
        quad[3] = 0.0;
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `out` must hold at least `4 * width` floats (the C caller passes the
/// mipmap-row cursor for an `img->width` row) and `line` at least
/// `3 * width` bytes (the row `fread` buffer of RGB triplets). The two
/// buffers must not overlap. `max` is the PPM maxval of the `max <= 255`
/// branch (`1..=255`).
#[no_mangle]
pub unsafe extern "C" fn darkroom_pnm_ppm_u8_row_to_float(
    out: *mut f32,
    line: *const u8,
    width: usize,
    max: std::ffi::c_uint,
) {
    if out.is_null() || line.is_null() || width == 0 || max == 0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(out_len) = width.checked_mul(4) else {
        return;
    };
    let Some(line_len) = width.checked_mul(3) else {
        return;
    };
    let out = std::slice::from_raw_parts_mut(out, out_len);
    let line = std::slice::from_raw_parts(line, line_len);
    pnm_ppm_u8_row_to_float(out, line, width, max);
}

// ── PNM PPM 16-bit RGB-triplet normalize (m4-231) ───────────────────────────

// Safe 16-bit RGB-triplet row to float RGBA kernel.
//
// Port of the inner per-pixel loop of the `max > 255` branch of `_read_ppm`
// (src/imageio/imageio_pnm.c): per column `x < width` and lane `c < 3`,
// `decoded = line[3*x + c].swap_bytes()` (PPM file order is big-endian with
// the most significant byte first; the C loop swaps under
// `G_BYTE_ORDER != G_BIG_ENDIAN`, and darktable supports only
// little-endian hosts (src/imageio/imageio_heif.c:167), so the swap is
// unconditional here — the same
// explicit-decode stance as the m4-208/m4-209 little-endian kernels and
// the m4-229 gray u16 sibling), then `out[4*x + c] = decoded as f32 / max
// as f32`, and `out[4*x + 3] = 0.0` (the explicitly zeroed alpha lane).
// The row `fread`, the line allocation, and the PBM/PGM/8-bit sibling
// branches stay in C.
//
// Fidelity notes:
// - The kernel must stay a division: `decoded as f32 / max as f32` replays
//   the C usual-arithmetic-conversion promotion bit for bit, while
//   `decoded as f32 * (1.0 / max as f32)` can round differently (the
//   reciprocal is itself inexact). The denominator conversion is hoisted
//   once; this is bit-identical because `max` is loop-invariant and the
//   conversion itself is exact over the contracted `256..=65535` range
//   (every u16 and every u32 below 2^24 widens to f32 exactly), applied in
//   the same order the C performs it (numerator conversion, denominator
//   conversion, then one IEEE division per lane).
// - Unlike the gray PGM sibling (m4-229) there is no fan-out: each RGB
//   output lane reads its own triplet word, so lane interleave is pinned
//   by the tests below.
// - `max` is `256..=65535` by contract (the C `_read_ppm` rejects `max == 0`
//   and `max > 65535` at entry, and `max <= 255` takes the m4-230 u8 triplet
//   branch); a zero `max` is a guarded no-op instead of reproducing a
//   divide-by-zero.
// - The serial C row loop has no cross-pixel dependencies, so sequential
//   iteration is identical.
//
// Degenerate `width == 0` is a no-op; short buffers are handled by
// clamped iteration (no panic, no out-of-bounds access). For the
// well-formed row buffers the C caller passes the clamp never engages
// and the behaviour is exactly the C loop's.
pub fn pnm_ppm_u16_row_to_float(out: &mut [f32], line: &[u16], width: usize, max: u32) {
    if max == 0 {
        return;
    }
    let n = width.min(line.len() / 3).min(out.len() / 4);
    let denom = max as f32;
    for x in 0..n {
        let s = 3 * x;
        let b = 4 * x;
        out[b] = line[s].swap_bytes() as f32 / denom;
        out[b + 1] = line[s + 1].swap_bytes() as f32 / denom;
        out[b + 2] = line[s + 2].swap_bytes() as f32 / denom;
        out[b + 3] = 0.0;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `pnm_ppm_u16_row_to_float`: walks the
/// source triplets and the output quads as zipped `as_chunks` chunk
/// iterators (the kernel walks columns with explicit stride writes), so the
/// sweep test cross-checks indexing as well as values. The big-endian decode
/// and the per-lane `word as f32 / max as f32` division match by construction
/// (see the kernel docs: the reciprocal-multiply form is not bit-identical
/// and must not be used). Well-formed buffers only: short inputs are an early
/// return here rather than clamping.
#[cfg(test)]
fn ref_pnm_ppm_u16_row_to_float(line: &[u16], out: &mut [f32], width: usize, max: u32) {
    if max == 0 {
        return;
    }
    let Some(out_need) = width.checked_mul(4) else {
        return;
    };
    let Some(line_need) = width.checked_mul(3) else {
        return;
    };
    if line.len() < line_need || out.len() < out_need {
        return;
    }
    let denom = max as f32;
    let (triplets, _) = line.as_chunks::<3>();
    let (quads, _) = out.as_chunks_mut::<4>();
    for (triplet, quad) in triplets.iter().take(width).zip(quads.iter_mut().take(width)) {
        quad[0] = triplet[0].swap_bytes() as f32 / denom;
        quad[1] = triplet[1].swap_bytes() as f32 / denom;
        quad[2] = triplet[2].swap_bytes() as f32 / denom;
        quad[3] = 0.0;
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `out` must hold at least `4 * width` floats (the C caller passes the
/// mipmap-row cursor for an `img->width` row) and `line` at least
/// `3 * width` u16 words (the row `fread` buffer of RGB triplets, still in
/// big-endian file order — the kernel performs the byte swap). The two
/// buffers must not overlap. `max` is the PPM maxval of the `max > 255`
/// branch (`256..=65535`).
#[no_mangle]
pub unsafe extern "C" fn darkroom_pnm_ppm_u16_row_to_float(
    out: *mut f32,
    line: *const u16,
    width: usize,
    max: std::ffi::c_uint,
) {
    if out.is_null() || line.is_null() || width == 0 || max == 0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(out_len) = width.checked_mul(4) else {
        return;
    };
    let Some(line_len) = width.checked_mul(3) else {
        return;
    };
    let out = std::slice::from_raw_parts_mut(out, out_len);
    let line = std::slice::from_raw_parts(line, line_len);
    pnm_ppm_u16_row_to_float(out, line, width, max);
}

// ── PNM PBM bit-unpack row to float RGBA (m4-239) ──────────────────────────

// Safe packed-bit row to float RGBA kernel.
//
// Port of the inner byte/bit nest of `_read_pbm`
// (src/imageio/imageio_pnm.c): per column `x < width`, the file bit is
// MSB-first within its pack byte (`(line[x / 8] >> (7 - x % 8)) & 1`,
// the per-pixel form of the C loop's `byte & 0x80` test plus `byte <<= 1`
// shift register), INVERTED (`line[x] ^ 0xff` in C: PBM 1 is black, so a
// set file bit decodes to 0.0 and a clear file bit to 1.0), then
// `out[4*x + c] = value` for `c` in 0..2 and `out[4*x + 3] = 0.0` (the
// explicitly zeroed alpha lane). Bits past `width` in the last pack byte
// are never read (the C `x * 8 + bit < width` tail guard). PBM has no
// maxval. The row `fread`, the line allocation, and the PGM/PPM sibling
// branches stay in C.
//
// Fidelity notes:
// - Pure bit shuffle plus exact 0.0/1.0 selection: no arithmetic, no
//   rounding, so bit-exactness holds by construction as long as the bit
//   order and the polarity match. `f32::from(1 - bit)` selects the same
//   rails the C `((byte & 0x80) >> 7) * 1.0` computes (`* 1.0` is exact).
// - The serial C row loop has no cross-pixel dependencies, so sequential
//   iteration is identical.
//
// Degenerate `width == 0` is a no-op; short buffers are handled by
// clamped iteration (no panic, no out-of-bounds access): the pixel count
// is capped by both the output quads and the pack bytes on hand. For the
// well-formed row buffers the C caller passes the clamp never engages
// and the behaviour is exactly the C loop's.
pub fn pnm_pbm_row_to_float(out: &mut [f32], line: &[u8], width: usize) {
    let have = line.len().saturating_mul(8).min(width);
    let n = have.min(out.len() / 4);
    for x in 0..n {
        let bit = (line[x / 8] >> (7 - x % 8)) & 1;
        let value = f32::from(1 - bit);
        let b = 4 * x;
        out[b] = value;
        out[b + 1] = value;
        out[b + 2] = value;
        out[b + 3] = 0.0;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `pnm_pbm_row_to_float`: walks the
/// pack bytes outer / bits inner with a mutating `byte <<= 1` shift
/// register exactly like the C nest (the kernel walks pixels with the
/// per-pixel shift expression), so the sweep test cross-checks traversal
/// as well as values. The `^ 0xff` inversion, the MSB-first order, the
/// `x < width` tail guard, and the zeroed alpha match by construction.
/// Well-formed buffers only: short inputs are an early return here rather
/// than clamping.
#[cfg(test)]
fn ref_pnm_pbm_row_to_float(line: &[u8], out: &mut [f32], width: usize) {
    let Some(out_need) = width.checked_mul(4) else {
        return;
    };
    let Some(line_need) = width.checked_add(7).map(|w| w / 8) else {
        return;
    };
    if line.len() < line_need || out.len() < out_need {
        return;
    }
    let mut x = 0usize;
    for &raw in line.iter().take(line_need) {
        let mut byte = raw ^ 0xff;
        for _ in 0..8 {
            if x >= width {
                break;
            }
            let value = f32::from((byte >> 7) & 1);
            let b = 4 * x;
            out[b] = value;
            out[b + 1] = value;
            out[b + 2] = value;
            out[b + 3] = 0.0;
            byte <<= 1;
            x += 1;
        }
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `out` must hold at least `4 * width` floats (the C caller passes the
/// mipmap-row cursor for an `img->width` row) and `line` at least
/// `(width + 7) / 8` bytes (the row `fread` buffer of packed MSB-first
/// bits). The two buffers must not overlap. PBM has no maxval.
#[no_mangle]
pub unsafe extern "C" fn darkroom_pnm_pbm_row_to_float(
    out: *mut f32,
    line: *const u8,
    width: usize,
) {
    if out.is_null() || line.is_null() || width == 0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(out_len) = width.checked_mul(4) else {
        return;
    };
    let Some(line_len) = width.checked_add(7).map(|w| w / 8) else {
        return;
    };
    let out = std::slice::from_raw_parts_mut(out, out_len);
    let line = std::slice::from_raw_parts(line, line_len);
    pnm_pbm_row_to_float(out, line, width);
}

// ── JPEG RGBA row strip to RGB24 (m4-232) ──────────────────────────────────

// Safe RGBA to RGB24 row-strip kernel.
//
// Port of the inner i/k nest of `dt_imageio_jpeg_compress`
// (src/imageio/imageio_jpeg.c:299-300): per column `i < width` and lane
// `k < 3`, `row[3*i + k] = buf[4*i + k]`. The alpha lane `buf[4*i + 3]`
// is dropped: never read, never written anywhere. The row allocation,
// the scanline loop, and the libjpeg calls stay in C, as does the
// duplicate i/k nest in `dt_imageio_jpeg_write_with_icc_profile`
// (imageio_jpeg.c:561-562, the m4-233 follow-up).
//
// Fidelity notes:
// - Pure byte shuffle: no arithmetic, no conversion, no rounding, so
//   bit-exactness holds by construction as long as lane selection matches.
// - The serial C row loop has no cross-pixel dependencies, so sequential
//   iteration is identical.
//
// Degenerate `width == 0` is a no-op; short buffers are handled by
// clamped iteration (no panic, no out-of-bounds access). For the
// well-formed row buffers the C caller passes the clamp never engages
// and the behaviour is exactly the C loop's.
pub fn jpeg_rgba_row_to_rgb24(row: &mut [u8], buf: &[u8], width: usize) {
    let n = width.min(row.len() / 3).min(buf.len() / 4);
    for i in 0..n {
        let d = 3 * i;
        let s = 4 * i;
        row[d] = buf[s];
        row[d + 1] = buf[s + 1];
        row[d + 2] = buf[s + 2];
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `jpeg_rgba_row_to_rgb24`: walks the
/// source quads and the destination triplets as zipped `as_chunks` chunk
/// iterators (the kernel walks columns with explicit stride writes), so the
/// sweep test cross-checks indexing as well as values. The alpha lane is
/// skipped by construction (the chunk destructure only names lanes 0..2).
/// Well-formed buffers only: short inputs are an early return here rather
/// than clamping.
#[cfg(test)]
fn ref_jpeg_rgba_row_to_rgb24(buf: &[u8], row: &mut [u8], width: usize) {
    let Some(row_need) = width.checked_mul(3) else {
        return;
    };
    let Some(buf_need) = width.checked_mul(4) else {
        return;
    };
    if buf.len() < buf_need || row.len() < row_need {
        return;
    }
    let (quads, _) = buf.as_chunks::<4>();
    let (triplets, _) = row.as_chunks_mut::<3>();
    for (quad, triplet) in quads.iter().take(width).zip(triplets.iter_mut().take(width)) {
        triplet[0] = quad[0];
        triplet[1] = quad[1];
        triplet[2] = quad[2];
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `row` must hold at least `3 * width` bytes (the C caller passes the
/// `3 * width` scanline strip) and `buf` at least `4 * width` bytes (the
/// RGBA row cursor for an `image_width` row). The two buffers must not
/// overlap.
#[no_mangle]
pub unsafe extern "C" fn darkroom_jpeg_rgba_row_to_rgb24(
    row: *mut u8,
    buf: *const u8,
    width: usize,
) {
    if row.is_null() || buf.is_null() || width == 0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(row_len) = width.checked_mul(3) else {
        return;
    };
    let Some(buf_len) = width.checked_mul(4) else {
        return;
    };
    let row = std::slice::from_raw_parts_mut(row, row_len);
    let buf = std::slice::from_raw_parts(buf, buf_len);
    jpeg_rgba_row_to_rgb24(row, buf, width);
}

// ── JPEG RGB24 row expand to RGBA (m4-237) ──────────────────────────────────

// Safe RGB24 to RGBA row-expand kernel.
//
// Port of the inner i/k nest of `decompress_plain`/`read_plain`
// (src/imageio/imageio_jpeg.c:184-186): per column `i < width` and lane
// `k < 3`, `tmp[4*i + k] = row[3*i + k]`. Lane 3 (`tmp[4*i + 3]`) is
// never written: the C loop stores only lanes 0..2, so whatever byte was
// already in the destination alpha slot survives. At the C call site that
// slot holds uninitialized allocator output: `tmp` is a cursor into `out`,
// which the `dt_imageio_jpeg_decompress` caller provides as a fresh
// `dt_alloc_align_uint8(4 * width * height)` allocation
// (src/imageio/imageio.c:680) that nothing zeroes before the row loop.
// The kernel therefore must not touch lane 3 either (no zeroing, no
// sentinel fill); callers must not expect it initialized. This is the
// exact inverse of the m4-232 `jpeg_rgba_row_to_rgb24` strip kernel.
//
// The row allocation, the scanline while-loop, the setjmp handling, and
// the libjpeg calls stay in C; the duplicate i/k nest in `read_plain`
// (imageio_jpeg.c:644) was wired to this same symbol in m4-238, so both
// plain paths now call it (call sites at imageio_jpeg.c:184 and :644).
//
// Fidelity notes:
// - Pure byte shuffle: no arithmetic, no conversion, no rounding, so
//   bit-exactness holds by construction as long as lane selection matches.
// - The serial C row loop has no cross-pixel dependencies, so sequential
//   iteration is identical.
//
// Degenerate `width == 0` is a no-op; short buffers are handled by
// clamped iteration (no panic, no out-of-bounds access). For the
// well-formed row buffers the C caller passes the clamp never engages
// and the behaviour is exactly the C loop's.
pub fn jpeg_rgb24_row_to_rgba(tmp: &mut [u8], row: &[u8], width: usize) {
    let n = width.min(tmp.len() / 4).min(row.len() / 3);
    for i in 0..n {
        let d = 4 * i;
        let s = 3 * i;
        tmp[d] = row[s];
        tmp[d + 1] = row[s + 1];
        tmp[d + 2] = row[s + 2];
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `jpeg_rgb24_row_to_rgba`: walks the
/// source triplets and the destination quads as zipped `as_chunks` chunk
/// iterators (the kernel walks columns with explicit stride writes), so the
/// sweep test cross-checks indexing as well as values. The alpha lane is
/// never stored by construction (only slots 0..2 are assigned, mirroring
/// the C loop leaving lane 3 untouched).
/// Well-formed buffers only: short inputs are an early return here rather
/// than clamping.
#[cfg(test)]
fn ref_jpeg_rgb24_row_to_rgba(row: &[u8], tmp: &mut [u8], width: usize) {
    let Some(row_need) = width.checked_mul(3) else {
        return;
    };
    let Some(tmp_need) = width.checked_mul(4) else {
        return;
    };
    if row.len() < row_need || tmp.len() < tmp_need {
        return;
    }
    let (triplets, _) = row.as_chunks::<3>();
    let (quads, _) = tmp.as_chunks_mut::<4>();
    for (triplet, quad) in triplets.iter().take(width).zip(quads.iter_mut().take(width)) {
        quad[0] = triplet[0];
        quad[1] = triplet[1];
        quad[2] = triplet[2];
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `tmp` must hold at least `4 * width` bytes (the C callers pass the
/// RGBA row cursor of the `decompress_plain`/`read_plain` output buffer;
/// its alpha slots are never written, see the kernel docs) and `row` at
/// least `3 * width` bytes (the libjpeg scanline strip). The two buffers
/// must not overlap.
#[no_mangle]
pub unsafe extern "C" fn darkroom_jpeg_rgb24_row_to_rgba(
    tmp: *mut u8,
    row: *const u8,
    width: usize,
) {
    if tmp.is_null() || row.is_null() || width == 0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(tmp_len) = width.checked_mul(4) else {
        return;
    };
    let Some(row_len) = width.checked_mul(3) else {
        return;
    };
    let tmp = std::slice::from_raw_parts_mut(tmp, tmp_len);
    let row = std::slice::from_raw_parts(row, row_len);
    jpeg_rgb24_row_to_rgba(tmp, row, width);
}

// ── 16-bit export float-to-u16 downconvert (m4-234) ─────────────────────────

// Safe in-place float-to-u16 export kernel.
//
// Port of the y/x/i nest of the `bpp == 16` branch of
// `dt_imageio_export_with_flags` (src/imageio/imageio.c): per pixel
// `k = width * y + x` and lane `i < 3`,
// `buf16[4*k+i] = roundf(CLAMP(buff[4*k+i] * 0xffff, 0, 0xffff))`, where
// `buff` (float) and `buf16` (u16) are two views of the same
// `pipe.backbuf` allocation (`npixels * 16` bytes: 4 floats per pixel).
// Lane 3 (alpha) is never read nor written, matching the C loop (which
// only stores lanes below 3); the `bpp == 8` `display_byteorder`
// R/B-swapped twin branch stays in C as the m4-236 follow-up.
//
// Fidelity notes:
// - `v * 65535.0` replays the C `buff[...] * 0xffff` bit for bit: the int
//   literal widens to f32 exactly (65535 is below 2^24) and the single
//   multiply rounds once, in the same order the C performs it.
// - The clamp mirrors the glib `CLAMP(v, 0, 0xffff)` expansion order
//   (`v > hi ? hi : v < lo ? lo : v`) with both bounds as f32 (exact for
//   0 and 65535); NaN therefore passes through, exactly as in C.
// - `f32::round` is round-half-away-from-zero, the same as C `roundf`.
// - The final conversion sees an integral value in `[0, 65535]`, except
//   for NaN input (which survives the clamp on both sides): C leaves a
//   NaN-to-integer conversion undefined, while Rust `as` saturates NaN
//   to 0, which is what the assignment produces on x86-64 in practice
//   (the convert instruction yields `0x80000000`, whose low 16 bits are
//   0). No other input can reach the conversion unclamped.
// - In-place aliasing: the u16 writes for pixel k land at byte offsets
//   `8*k..8*k+6`, strictly below the float reads of every later pixel
//   (`16*j..` for `j > k`); within the pixel, lanes run in ascending order
//   and each float lane is read into a local before its lane is written
//   (lane-2's bytes overlap lane-1's already-consumed read span, which is
//   safe precisely because lane 1 is consumed first), so the forward walk
//   can never clobber an unread input — the same order the C loop relies
//   on. The single `&mut [u8]` view keeps this aliasing inside safe code
//   (the m4-205 `swap_rb` precedent: one pointer in, one slice built).
//
// Degenerate `npixels == 0` is a no-op; short buffers are handled by
// clamped iteration (no panic, no out-of-bounds access). For the
// well-formed export buffer the C caller passes the clamp never engages
// and the behaviour is exactly the C loop's.
#[allow(clippy::manual_clamp)]
pub fn float_to_u16_inplace(buf: &mut [u8], npixels: usize) {
    let n = npixels.min(buf.len() / 16);
    for k in 0..n {
        let r = 16 * k;
        let w = 8 * k;
        for i in 0..3 {
            let o = r + 4 * i;
            let v = f32::from_ne_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            let scaled = v * 65535.0;
            let clamped = if scaled > 65535.0 {
                65535.0
            } else if scaled < 0.0 {
                0.0
            } else {
                scaled
            };
            let bytes = (clamped.round() as u16).to_ne_bytes();
            buf[w + 2 * i] = bytes[0];
            buf[w + 2 * i + 1] = bytes[1];
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `float_to_u16_inplace`: walks
/// separate source/destination slices as zipped `as_chunks` quad
/// iterators (the kernel walks one aliased byte buffer with explicit
/// stride reads and writes), so the sweep test cross-checks indexing as
/// well as values. The `v * 65535.0` multiply, the high-test-first clamp,
/// and the half-away `round` match by construction (see the kernel docs:
/// a reciprocal multiply or a banker's rounding would not be
/// bit-identical and must not be used). Well-formed buffers only: short
/// inputs are an early return here rather than clamping.
#[cfg(test)]
#[allow(clippy::manual_clamp)]
fn ref_float_to_u16(src: &[f32], dst: &mut [u16], npixels: usize) {
    let Some(src_need) = npixels.checked_mul(4) else {
        return;
    };
    let Some(dst_need) = npixels.checked_mul(4) else {
        return;
    };
    if src.len() < src_need || dst.len() < dst_need {
        return;
    }
    let (quads, _) = src.as_chunks::<4>();
    let (words, _) = dst.as_chunks_mut::<4>();
    for (quad, word) in quads.iter().take(npixels).zip(words.iter_mut().take(npixels)) {
        for (lane, slot) in quad[..3].iter().zip(word[..3].iter_mut()) {
            let scaled = *lane * 65535.0;
            let clamped = if scaled > 65535.0 {
                65535.0
            } else if scaled < 0.0 {
                0.0
            } else {
                scaled
            };
            *slot = clamped.round() as u16;
        }
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `buf` must hold at least `16 * npixels` bytes (the C caller passes
/// `pipe.backbuf` for a `processed_width * processed_height` float-RGBA
/// image, converted in place). Lane 3 of every pixel and all bytes past
/// `8 * npixels` keep their values. No alignment requirement: the kernel
/// only performs byte loads and stores.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_float_to_u16(buf: *mut u8, npixels: usize) {
    if buf.is_null() || npixels == 0 {
        return;
    }
    // validate the product BEFORE building the slice below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(len) = npixels.checked_mul(16) else {
        return;
    };
    if len > isize::MAX as usize {
        return;
    }
    let buf = std::slice::from_raw_parts_mut(buf, len);
    float_to_u16_inplace(buf, npixels);
}

// ── 8-bit export float-to-u8 downconvert, plain lane order (m4-235) ──────────

// Safe in-place float-to-u8 export kernel, plain lane order.
//
// Port of the per-pixel k loop of the `bpp == 8` `!display_byteorder`
// `hq_process` branch of `dt_imageio_export_with_flags`
// (src/imageio/imageio.c): per pixel `k` and lane `i < 3`,
// `outbuf[4*k+i] = roundf(CLAMP(inbuf[4*k+i] * 0xff, 0, 0xff))`, where
// `inbuf` (float) and `outbuf` (u8) are two views of the same
// `pipe.backbuf` allocation (`npixels * 16` bytes: 4 floats per pixel).
// Lane 3 (alpha) is never read nor written, matching the C loop (which
// only stores lanes 0..2); the `display_byteorder` R/B-swapped twin
// branch stays in C as the m4-236 follow-up.
//
// Fidelity notes:
// - `v * 255.0` replays the C `inbuf[...] * 0xff` bit for bit: the int
//   literal widens to f32 exactly (255 is below 2^24) and the single
//   multiply rounds once, in the same order the C performs it.
// - The clamp mirrors the glib `CLAMP(v, 0, 0xff)` expansion order
//   (`v > hi ? hi : v < lo ? lo : v`) with both bounds as f32 (exact for
//   0 and 255); NaN therefore passes through, exactly as in C.
// - `f32::round` is round-half-away-from-zero, the same as C `roundf`.
// - The final conversion sees an integral value in `[0, 255]`, except
//   for NaN input (which survives the clamp on both sides): C leaves a
//   NaN-to-integer conversion undefined, while Rust `as` saturates NaN
//   to 0, which is what the assignment produces on x86-64 in practice
//   (the convert instruction yields `0x80000000`, whose low 8 bits are
//   0). No other input can reach the conversion unclamped.
// - In-place aliasing: the u8 writes for pixel k land at byte offsets
//   `4*k..4*k+3`, strictly below the float reads of every later pixel
//   (`16*j..` for `j > k`); within the pixel, lanes run in ascending
//   order and each float lane is read into a local before its lane is
//   written, so the forward walk can never clobber an unread input —
//   the same order the C loop relies on. The single `&mut [u8]` view
//   keeps this aliasing inside safe code (the m4-234 `float_to_u16`
//   precedent: one pointer in, one slice built).
//
// Degenerate `npixels == 0` is a no-op; short buffers are handled by
// clamped iteration (no panic, no out-of-bounds access). For the
// well-formed export buffer the C caller passes the clamp never engages
// and the behaviour is exactly the C loop's.
#[allow(clippy::manual_clamp)]
pub fn float_to_u8_inplace(buf: &mut [u8], npixels: usize) {
    let n = npixels.min(buf.len() / 16);
    for k in 0..n {
        let r = 16 * k;
        let w = 4 * k;
        for i in 0..3 {
            let o = r + 4 * i;
            let v = f32::from_ne_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            let scaled = v * 255.0;
            let clamped = if scaled > 255.0 {
                255.0
            } else if scaled < 0.0 {
                0.0
            } else {
                scaled
            };
            buf[w + i] = clamped.round() as u8;
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `float_to_u8_inplace`: walks
/// separate source/destination slices as zipped `as_chunks` quad
/// iterators (the kernel walks one aliased byte buffer with explicit
/// stride reads and writes), so the sweep test cross-checks indexing as
/// well as values. The `v * 255.0` multiply, the high-test-first clamp,
/// and the half-away `round` match by construction (see the kernel docs:
/// a reciprocal multiply or a banker's rounding would not be
/// bit-identical and must not be used). Well-formed buffers only: short
/// inputs are an early return here rather than clamping.
#[cfg(test)]
#[allow(clippy::manual_clamp)]
fn ref_float_to_u8(src: &[f32], dst: &mut [u8], npixels: usize) {
    let Some(src_need) = npixels.checked_mul(4) else {
        return;
    };
    let Some(dst_need) = npixels.checked_mul(4) else {
        return;
    };
    if src.len() < src_need || dst.len() < dst_need {
        return;
    }
    let (quads, _) = src.as_chunks::<4>();
    let (bytes, _) = dst.as_chunks_mut::<4>();
    for (quad, word) in quads.iter().take(npixels).zip(bytes.iter_mut().take(npixels)) {
        for (lane, slot) in quad[..3].iter().zip(word[..3].iter_mut()) {
            let scaled = *lane * 255.0;
            let clamped = if scaled > 255.0 {
                255.0
            } else if scaled < 0.0 {
                0.0
            } else {
                scaled
            };
            *slot = clamped.round() as u8;
        }
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `buf` must hold at least `16 * npixels` bytes (the C caller passes
/// `pipe.backbuf` for a `processed_width * processed_height` float-RGBA
/// image, converted in place). Lane 3 of every pixel and all bytes past
/// `4 * npixels` keep their values. No alignment requirement: the kernel
/// only performs byte loads and stores.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_float_to_u8(buf: *mut u8, npixels: usize) {
    if buf.is_null() || npixels == 0 {
        return;
    }
    // validate the product BEFORE building the slice below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(len) = npixels.checked_mul(16) else {
        return;
    };
    if len > isize::MAX as usize {
        return;
    }
    let buf = std::slice::from_raw_parts_mut(buf, len);
    float_to_u8_inplace(buf, npixels);
}

// ── 8-bit export float-to-u8 downconvert, R/B-swapped lane order (m4-236) ───

// Safe in-place float-to-u8 export kernel, R/B-swapped lane order.
//
// Port of the per-pixel k loop of the `bpp == 8` `display_byteorder`
// `hq_process` branch of `dt_imageio_export_with_flags`
// (src/imageio/imageio.c): per pixel `k`,
// `r = roundf(CLAMP(inbuf[4*k+2] * 0xff, 0, 0xff))`,
// `g = roundf(CLAMP(inbuf[4*k+1] * 0xff, 0, 0xff))`,
// `b = roundf(CLAMP(inbuf[4*k+0] * 0xff, 0, 0xff))`,
// then `outbuf[4*k+0] = r`, `outbuf[4*k+1] = g`, `outbuf[4*k+2] = b`,
// where `inbuf` (float) and `outbuf` (u8) are two views of the same
// `pipe.backbuf` allocation (`npixels * 16` bytes: 4 floats per pixel).
// Read lane order is (2, 1, 0) into write slots (0, 1, 2): R and B cross
// while G stays. Lane 3 (alpha) is never read nor written, matching the C
// loop (which only stores lanes 0..2); this is the swapped twin of the
// m4-235 `float_to_u8_inplace` plain branch.
//
// Fidelity notes:
// - `v * 255.0` replays the C `inbuf[...] * 0xff` bit for bit: the int
//   literal widens to f32 exactly (255 is below 2^24) and the single
//   multiply rounds once, in the same order the C performs it.
// - The clamp mirrors the glib `CLAMP(v, 0, 0xff)` expansion order
//   (`v > hi ? hi : v < lo ? lo : v`) with both bounds as f32 (exact for
//   0 and 255); NaN therefore passes through, exactly as in C.
// - `f32::round` is round-half-away-from-zero, the same as C `roundf`.
// - The final conversion sees an integral value in `[0, 255]`, except
//   for NaN input (which survives the clamp on both sides): C leaves a
//   NaN-to-integer conversion undefined, while Rust `as` saturates NaN
//   to 0, which is what the assignment produces on x86-64 in practice
//   (the convert instruction yields `0x80000000`, whose low 8 bits are
//   0). No other input can reach the conversion unclamped.
// - In-place aliasing: the u8 writes for pixel k land at byte offsets
//   `4*k..4*k+3`, strictly below the float reads of every later pixel
//   (`16*j..` for `j > k`); within the pixel ALL THREE float lanes are
//   read into locals before ANY lane is written, because write slot 0
//   overlaps lane 0's bytes at `k == 0` while its value comes from lane
//   2 — a per-lane read-then-write in slot order would clobber the
//   unread lane-0 input. The C loop hoists the same way (r/g/b locals
//   first, stores after). The single `&mut [u8]` view keeps this
//   aliasing inside safe code (the m4-235 `float_to_u8_inplace`
//   precedent: one pointer in, one slice built).
//
// Degenerate `npixels == 0` is a no-op; short buffers are handled by
// clamped iteration (no panic, no out-of-bounds access). For the
// well-formed export buffer the C caller passes the clamp never engages
// and the behaviour is exactly the C loop's.
#[allow(clippy::manual_clamp)]
pub fn float_to_u8_swap_rb_inplace(buf: &mut [u8], npixels: usize) {
    let n = npixels.min(buf.len() / 16);
    for k in 0..n {
        let r = 16 * k;
        let w = 4 * k;
        let v0 = f32::from_ne_bytes([buf[r], buf[r + 1], buf[r + 2], buf[r + 3]]);
        let v1 = f32::from_ne_bytes([buf[r + 4], buf[r + 5], buf[r + 6], buf[r + 7]]);
        let v2 = f32::from_ne_bytes([buf[r + 8], buf[r + 9], buf[r + 10], buf[r + 11]]);
        for (i, v) in [v2, v1, v0].into_iter().enumerate() {
            let scaled = v * 255.0;
            let clamped = if scaled > 255.0 {
                255.0
            } else if scaled < 0.0 {
                0.0
            } else {
                scaled
            };
            buf[w + i] = clamped.round() as u8;
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `float_to_u8_swap_rb_inplace`:
/// walks separate source/destination slices as zipped `as_chunks` quad
/// iterators (the kernel walks one aliased byte buffer with explicit
/// stride reads and writes), so the sweep test cross-checks indexing as
/// well as values. The `v * 255.0` multiply, the high-test-first clamp,
/// the half-away `round`, and the (2, 1, 0) read-to-(0, 1, 2) slot swap
/// match by construction (see the kernel docs: a reciprocal multiply or
/// a banker's rounding would not be bit-identical and must not be used).
/// Well-formed buffers only: short inputs are an early return here
/// rather than clamping.
#[cfg(test)]
#[allow(clippy::manual_clamp)]
fn ref_float_to_u8_swap_rb(src: &[f32], dst: &mut [u8], npixels: usize) {
    let Some(src_need) = npixels.checked_mul(4) else {
        return;
    };
    let Some(dst_need) = npixels.checked_mul(4) else {
        return;
    };
    if src.len() < src_need || dst.len() < dst_need {
        return;
    }
    let (quads, _) = src.as_chunks::<4>();
    let (bytes, _) = dst.as_chunks_mut::<4>();
    for (quad, word) in quads.iter().take(npixels).zip(bytes.iter_mut().take(npixels)) {
        for (slot, lane) in [(0, 2), (1, 1), (2, 0)] {
            let scaled = quad[lane] * 255.0;
            let clamped = if scaled > 255.0 {
                255.0
            } else if scaled < 0.0 {
                0.0
            } else {
                scaled
            };
            word[slot] = clamped.round() as u8;
        }
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `buf` must hold at least `16 * npixels` bytes (the C caller passes
/// `pipe.backbuf` for a `processed_width * processed_height` float-RGBA
/// image, converted in place with R/B lanes swapped). Lane 3 of every
/// pixel and all bytes past `4 * npixels` keep their values. No
/// alignment requirement: the kernel only performs byte loads and
/// stores.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_float_to_u8_swap_rb(buf: *mut u8, npixels: usize) {
    if buf.is_null() || npixels == 0 {
        return;
    }
    // validate the product BEFORE building the slice below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(len) = npixels.checked_mul(16) else {
        return;
    };
    if len > isize::MAX as usize {
        return;
    }
    let buf = std::slice::from_raw_parts_mut(buf, len);
    float_to_u8_swap_rb_inplace(buf, npixels);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_oriented_mapping(
        bpp: usize,
        wd: usize,
        ht: usize,
        fwd: usize,
        fht: usize,
        stride: usize,
        orientation: i32,
    ) -> Vec<(usize, usize)> {
        let (dx, dy) = if orientation & 4 == 0 {
            (1i128, wd as i128)
        } else {
            (ht as i128, 1i128)
        };
        let (base_x, step_x) = if orientation & 2 == 0 {
            (0, dx)
        } else {
            ((fwd as i128 - 1) * dx, -dx)
        };
        let (base_y, step_y) = if orientation & 1 == 0 {
            (0, dy)
        } else {
            ((fht as i128 - 1) * dy, -dy)
        };
        (0..bpp * wd * ht)
            .map(|byte| {
                let pixel = byte / bpp;
                let dst_pixel =
                    base_x + base_y + step_x * (pixel % wd) as i128 + step_y * (pixel / wd) as i128;
                let dst = usize::try_from(dst_pixel).unwrap() * bpp + byte % bpp;
                let src = (pixel / wd) * stride + byte % (wd * bpp);
                (dst, src)
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_oriented_case(
        bpp: usize,
        wd: usize,
        ht: usize,
        fwd: usize,
        fht: usize,
        stride: usize,
        orientation: i32,
        bits: bool,
    ) {
        let mapping = ref_oriented_mapping(bpp, wd, ht, fwd, fht, stride, orientation);
        let in_bytes = mapping.iter().map(|&(_, src)| src + 1).max().unwrap();
        let out_bytes = mapping.iter().map(|&(dst, _)| dst + 1).max().unwrap();
        let pattern: Vec<u8> = [0x7fc1_2345u32, 0x8000_0000, 0, 0xffa5_4321]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect();
        let inp: Box<[u8]> = (0..in_bytes)
            .map(|i| {
                if bits {
                    pattern[i % pattern.len()]
                } else {
                    ((i as u64 * 2_654_435_761 + 0x9E37) % 256) as u8
                }
            })
            .collect();
        let original = inp.clone();
        let mut expected = vec![0xCD; out_bytes + 2];
        for &(dst, src) in &mapping {
            expected[1 + dst] = inp[src];
        }
        let mut safe = vec![0xCD; out_bytes + 2];
        let mut ffi = safe.clone();
        flip_buffers_oriented(
            &mut safe[1..1 + out_bytes],
            &inp,
            bpp,
            wd,
            ht,
            fwd,
            fht,
            stride,
            orientation,
        );
        assert_eq!(
            safe, expected,
            "safe: {bpp}/{wd}/{ht}/{fwd}/{fht}/{stride}/{orientation}"
        );
        assert_eq!(
            unsafe {
                darkroom_imageio_flip_buffers_oriented(
                    ffi.as_mut_ptr().add(1).cast(),
                    inp.as_ptr().cast(),
                    bpp,
                    wd as i32,
                    ht as i32,
                    fwd as i32,
                    fht as i32,
                    stride as i32,
                    orientation,
                )
            },
            1
        );
        assert_eq!(
            ffi, expected,
            "ffi: {bpp}/{wd}/{ht}/{fwd}/{fht}/{stride}/{orientation}"
        );
        let mut exact = vec![0xCD; out_bytes].into_boxed_slice();
        assert_eq!(
            unsafe {
                darkroom_imageio_flip_buffers_oriented(
                    exact.as_mut_ptr().cast(),
                    inp.as_ptr().cast(),
                    bpp,
                    wd as i32,
                    ht as i32,
                    fwd as i32,
                    fht as i32,
                    stride as i32,
                    orientation,
                )
            },
            1
        );
        assert_eq!(&*exact, &expected[1..1 + out_bytes]);
        assert_eq!(inp, original);
    }

    #[test]
    fn oriented_matches_reference_all_flags_shapes_bpp_strides() {
        for orientation in 0..8 {
            for (wd, ht) in [(1, 1), (7, 1), (1, 9), (3, 5), (5, 3), (17, 9)] {
                for bpp in [1, 2, 3, 4, 16] {
                    for stride in [0, 1, bpp * wd - 1, bpp * wd, bpp * wd + 7] {
                        assert_oriented_case(bpp, wd, ht, wd, ht, stride, orientation, false);
                    }
                }
            }
        }
    }

    #[test]
    fn oriented_displaced_and_unused_extents_exact_spans_and_gaps() {
        for orientation in 0..8 {
            for (wd, ht) in [(1, 1), (1, 7), (9, 1), (3, 5), (5, 3)] {
                for fwd in [wd, wd + 1, wd + 7] {
                    for fht in [ht, ht + 2, ht + 9] {
                        assert_oriented_case(3, wd, ht, fwd, fht, 3 * wd + 5, orientation, false);
                    }
                }
                let fwd = if orientation & 2 == 0 { 1 } else { wd };
                let fht = if orientation & 1 == 0 { 1 } else { ht };
                assert_oriented_case(2, wd, ht, fwd, fht, 0, orientation, false);
            }
        }
        assert_oriented_case(4, 3, 5, 3, 5, 12, 0x78, false);
    }

    #[test]
    fn oriented_preserves_nan_signed_zero_and_alpha_bytes() {
        for orientation in 0..8 {
            for bpp in [1, 2, 3, 4, 16] {
                for (wd, ht) in [(3, 5), (5, 3)] {
                    for stride in [0, bpp * wd - 1, bpp * wd, bpp * wd + 5] {
                        assert_oriented_case(bpp, wd, ht, wd, ht, stride, orientation, true);
                    }
                }
            }
        }
    }

    #[test]
    fn oriented_single_row_ignores_signed_stride() {
        let inp = [1u8, 2, 3, 4, 5, 6];
        for orientation in 0..8 {
            let mut expected = [0; 6];
            for (dst, src) in ref_oriented_mapping(2, 3, 1, 3, 1, 0, orientation) {
                expected[dst] = inp[src];
            }
            for stride in [i32::MIN, -1, 0, 1, i32::MAX] {
                let mut out = [0; 6];
                assert_eq!(
                    unsafe {
                        darkroom_imageio_flip_buffers_oriented(
                            out.as_mut_ptr().cast(),
                            inp.as_ptr().cast(),
                            2,
                            3,
                            1,
                            3,
                            1,
                            stride,
                            orientation,
                        )
                    },
                    1
                );
                assert_eq!(out, expected);
            }
            let mut out = [0; 6];
            flip_buffers_oriented(&mut out, &inp, 2, 3, 1, 3, 1, usize::MAX, orientation);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn oriented_ffi_maybe_uninit_payload_padding_and_output() {
        use std::mem::MaybeUninit;

        for orientation in 0..8 {
            for bpp in [1, 2, 3, 4, 16] {
                let (wd, ht, fwd, fht) = (3, 5, 6, 7);
                for stride in [0, 1, bpp * wd, bpp * wd + 5] {
                    let mapping = ref_oriented_mapping(bpp, wd, ht, fwd, fht, stride, orientation);
                    let in_bytes = mapping.iter().map(|&(_, src)| src + 1).max().unwrap();
                    let out_bytes = mapping.iter().map(|&(dst, _)| dst + 1).max().unwrap();
                    for partial_payload in [false, true] {
                        let mut inp =
                            vec![MaybeUninit::<u8>::uninit(); in_bytes].into_boxed_slice();
                        let mut out =
                            vec![MaybeUninit::<u8>::uninit(); out_bytes].into_boxed_slice();
                        for &(_, src) in &mapping {
                            if !partial_payload || src % 3 == 0 {
                                inp[src].write((src % 251) as u8);
                            }
                        }
                        assert_eq!(
                            unsafe {
                                darkroom_imageio_flip_buffers_oriented(
                                    out.as_mut_ptr().cast(),
                                    inp.as_ptr().cast(),
                                    bpp,
                                    wd as i32,
                                    ht as i32,
                                    fwd as i32,
                                    fht as i32,
                                    stride as i32,
                                    orientation,
                                )
                            },
                            1
                        );
                        for &(dst, src) in &mapping {
                            if !partial_payload || src % 3 == 0 {
                                assert_eq!(unsafe { out[dst].assume_init() }, (src % 251) as u8);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn oriented_safe_degenerate_overflow_and_short_buffers() {
        let limit = isize::MAX as usize;
        for (bpp, wd, ht, fwd, fht, stride, orientation) in [
            (0, 2, 2, 2, 2, 2, 7),
            (1, 0, 2, 2, 2, 2, 7),
            (1, 2, 0, 2, 2, 2, 7),
            (1, 2, 2, 0, 2, 2, 0),
            (1, 2, 2, 2, 0, 2, 0),
            (1, 2, 2, 1, 2, 2, 2),
            (1, 2, 2, 2, 1, 2, 1),
            (usize::MAX, 2, 1, 2, 1, 0, 2),
            (1, 1, 2, 1, 2, usize::MAX, 1),
            (1, 1, 2, 1, 2, limit, 1),
            (limit + 1, 1, 1, 1, 1, 0, 7),
            (limit, 1, 2, 1, 2, 0, 7),
            (1, 2, 2, usize::MAX, 2, 0, 6),
            (1, 2, 2, 2, usize::MAX, 0, 1),
            (1, 1, 1, usize::MAX, 2, 0, 3),
            (1, usize::MAX, 1, usize::MAX, 1, 0, 0),
            (1, 1, usize::MAX, 1, usize::MAX, 0, 0),
        ] {
            assert_eq!(
                oriented_layout(bpp, wd, ht, fwd, fht, stride, orientation),
                None
            );
            let mut out = [7; 16];
            flip_buffers_oriented(
                &mut out,
                &[9; 16],
                bpp,
                wd,
                ht,
                fwd,
                fht,
                stride,
                orientation,
            );
            assert_eq!(out, [7; 16]);
        }
        for orientation in 0..8 {
            let mapping = ref_oriented_mapping(3, 3, 5, 6, 7, 11, orientation);
            let out_bytes = mapping.iter().map(|&(dst, _)| dst + 1).max().unwrap();
            let in_bytes = 53;
            let inp = vec![9; in_bytes];
            for short in 0..out_bytes {
                let mut out = vec![7; out_bytes];
                flip_buffers_oriented(&mut out[..short], &inp, 3, 3, 5, 6, 7, 11, orientation);
                assert!(out.iter().all(|&byte| byte == 7));
            }
            for short in 0..in_bytes {
                let mut out = vec![7; out_bytes];
                flip_buffers_oriented(&mut out, &inp[..short], 3, 3, 5, 6, 7, 11, orientation);
                assert!(out.iter().all(|&byte| byte == 7));
            }
        }
    }

    #[test]
    fn oriented_layout_actual_span_boundaries() {
        let limit = isize::MAX as usize;
        let layout = oriented_layout(1, 1, 2, 1, 2, limit - 1, 7).unwrap();
        assert_eq!((layout.in_bytes, layout.out_bytes), (limit, 2));
        assert_eq!(oriented_layout(1, 1, 2, 1, 2, limit, 7), None);
        for orientation in 0..8 {
            let layout = oriented_layout(limit, 1, 1, 1, 1, usize::MAX, orientation).unwrap();
            assert_eq!((layout.in_bytes, layout.out_bytes), (limit, limit));
            assert_eq!(oriented_layout(limit + 1, 1, 1, 1, 1, 0, orientation), None);
        }
        for (fwd, fht, orientation) in [(limit, 1, 2), (1, limit, 1)] {
            let layout = oriented_layout(1, 1, 1, fwd, fht, 0, orientation).unwrap();
            assert_eq!((layout.in_bytes, layout.out_bytes), (1, limit));
            assert_eq!(
                oriented_layout(1, 1, 1, fwd + 1, fht + 1, 0, orientation),
                None
            );
        }
        let layout = oriented_layout(1, 2, 3, usize::MAX, usize::MAX, 2, 4).unwrap();
        assert_eq!((layout.in_bytes, layout.out_bytes), (6, 6));
    }

    #[test]
    fn oriented_ffi_null_dimension_negative_stride_and_overflow_guards() {
        let inp = [9u8; 16];
        let mut out = [7u8; 16];
        unsafe {
            assert_eq!(
                darkroom_imageio_flip_buffers_oriented(
                    std::ptr::null_mut(),
                    inp.as_ptr().cast(),
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    7,
                ),
                0
            );
            assert_eq!(
                darkroom_imageio_flip_buffers_oriented(
                    out.as_mut_ptr().cast(),
                    std::ptr::null(),
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    7,
                ),
                0
            );
        }
        for (bpp, wd, ht, fwd, fht, stride, orientation) in [
            (0, 2, 2, 2, 2, 2, 7),
            (1, 0, 2, 2, 2, 2, 7),
            (1, -1, 2, 2, 2, 2, 7),
            (1, 2, 0, 2, 2, 2, 7),
            (1, 2, -1, 2, 2, 2, 7),
            (1, 2, 2, 0, 2, 2, 0),
            (1, 2, 2, -1, 2, 2, 0),
            (1, 2, 2, 2, 0, 2, 0),
            (1, 2, 2, 2, -1, 2, 0),
            (1, 2, 2, 1, 2, 2, 2),
            (1, 2, 2, 2, 1, 2, 1),
            (1, 2, 2, 2, 2, -1, 7),
            (1, 2, 2, 2, 2, i32::MIN, 7),
            (usize::MAX, 2, 1, 2, 1, 0, 2),
            (isize::MAX as usize + 1, 1, 1, 1, 1, 0, 7),
            (isize::MAX as usize, 1, 2, 1, 2, 0, 7),
            (16, i32::MAX, i32::MAX, i32::MAX, i32::MAX, i32::MAX, 7),
            (16, 1, 2, i32::MAX, i32::MAX, 0, 1),
        ] {
            // On 64-bit this row is a legitimate Some layout (the span fits
            // isize) but the 16-byte stack fixture would be overrun since the
            // FFI takes no length args, so skip it per the caller-size
            // contract, cf. m4-224.
            if bpp == 16 && wd == 1 && usize::BITS > 32 {
                continue;
            }
            assert_eq!(
                unsafe {
                    darkroom_imageio_flip_buffers_oriented(
                        out.as_mut_ptr().cast(),
                        inp.as_ptr().cast(),
                        bpp,
                        wd,
                        ht,
                        fwd,
                        fht,
                        stride,
                        orientation,
                    )
                },
                0
            );
            assert_eq!(out, [7; 16]);
        }
        assert_eq!(inp, [9; 16]);
    }

    #[test]
    fn oriented_golden_vectors_from_c_formula() {
        // Hand-derived from the pre-m4-226 C si/sj loop in
        // src/imageio/imageio.c: si starts at bpp (1), sj at wd * bpp (2);
        // ORIENTATION_SWAP_XY swaps them to sj = 1, si = ht * bpp = 3;
        // ORIENTATION_FLIP_Y (flag 1) sets jj = fht - 1 = 2 and negates sj;
        // ORIENTATION_FLIP_X (flag 2) sets ii = fwd - 1 = 1 and negates si
        // (flag values per src/common/image.h: FLIP_Y = 1, FLIP_X = 2,
        // SWAP_XY = 4). Input rows are [0, 1], [2, 3], [4, 5].
        // Orientation 5 (FLIP_Y plus SWAP): jj = 2, sj = -1, ii = 0, si = 3,
        // so row j = 0 writes out[2] = in[0], out[5] = in[1]; j = 1 writes
        // out[1] = in[2], out[4] = in[3]; j = 2 writes out[0] = in[4],
        // out[3] = in[5], giving [4, 2, 0, 5, 3, 1].
        // Orientation 6 (FLIP_X plus SWAP): jj = 0, sj = 1, ii = 1, si = -3,
        // so row j = 0 writes out[3] = in[0], out[0] = in[1]; j = 1 writes
        // out[4] = in[2], out[1] = in[3]; j = 2 writes out[5] = in[4],
        // out[2] = in[5], giving [1, 3, 5, 0, 2, 4].
        // Orientation -1 has every low bit set, so it takes the same
        // branches as 7 (transverse): jj = 2, sj = -1, ii = 1, si = -3;
        // j = 0 writes out[5] = in[0], out[2] = in[1]; j = 1 writes
        // out[4] = in[2], out[1] = in[3]; j = 2 writes out[3] = in[4],
        // out[0] = in[5], giving [5, 3, 1, 4, 2, 0].
        fn run_safe(inp: &[u8; 6], orientation: i32) -> [u8; 6] {
            let mut out = [0xCD; 6];
            flip_buffers_oriented(&mut out, inp, 1, 2, 3, 2, 3, 2, orientation);
            out
        }
        fn run_ffi(inp: &[u8; 6], orientation: i32) -> [u8; 6] {
            let mut out = [0xCD; 6];
            assert_eq!(
                unsafe {
                    darkroom_imageio_flip_buffers_oriented(
                        out.as_mut_ptr().cast(),
                        inp.as_ptr().cast(),
                        1,
                        2,
                        3,
                        2,
                        3,
                        2,
                        orientation,
                    )
                },
                1
            );
            out
        }
        let inp = [0u8, 1, 2, 3, 4, 5];
        assert_eq!(run_safe(&inp, 5), [4, 2, 0, 5, 3, 1]);
        assert_eq!(run_ffi(&inp, 5), [4, 2, 0, 5, 3, 1]);
        assert_eq!(run_safe(&inp, 6), [1, 3, 5, 0, 2, 4]);
        assert_eq!(run_ffi(&inp, 6), [1, 3, 5, 0, 2, 4]);
        assert_eq!(run_safe(&inp, -1), [5, 3, 1, 4, 2, 0]);
        assert_eq!(run_safe(&inp, 7), [5, 3, 1, 4, 2, 0]);
        assert_eq!(run_ffi(&inp, -1), run_ffi(&inp, 7));
        assert_eq!(run_ffi(&inp, -1), [5, 3, 1, 4, 2, 0]);
    }

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

    // ── Oriented 8-bit to float normalise (m4-227) ───────────────────────────

    /// Structurally divergent reference for `u8_to_float_oriented`: a textual
    /// port of the pre-m4-227 C loop (si starts at 4 lanes, sj at wd * 4;
    /// SWAP_XY exchanges them to sj = 4, si = ht * 4; FLIP_Y sets jj =
    /// fht - 1 and negates sj; FLIP_X sets ii = fwd - 1 and negates si)
    /// instead of the kernel's pitch/origin mapping, so the sweep test
    /// cross-checks traversal as well as values. Must return bit-identical
    /// output to [`u8_to_float_oriented`] (compared with `to_bits`, so even
    /// NaN payloads from a `white == black` caller must agree).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn ref_u8_to_float_oriented(
        out: &mut [f32],
        inp: &[u8],
        black: f32,
        white: f32,
        ch: usize,
        wd: usize,
        ht: usize,
        fwd: usize,
        fht: usize,
        stride: usize,
        orientation: i32,
    ) {
        let scale = 1.0f32 / (white - black);
        let (mut ii, mut jj) = (0i128, 0i128);
        let (mut si, mut sj) = (4i128, wd as i128 * 4);
        if orientation & 4 != 0 {
            sj = 4;
            si = ht as i128 * 4;
        }
        if orientation & 1 != 0 {
            jj = fht as i128 - 1;
            sj = -sj;
        }
        if orientation & 2 != 0 {
            ii = fwd as i128 - 1;
            si = -si;
        }
        // C base: out + |sj| * jj + |si| * ii lanes, input rows at stride * j.
        for j in 0..ht {
            for i in 0..wd {
                let dst = (sj.abs() * jj + si.abs() * ii + sj * j as i128 + si * i as i128)
                    as usize;
                let src = j * stride + ch * i;
                for k in 0..ch {
                    out[dst + k] = (inp[src + k] as f32 - black) * scale;
                }
            }
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn assert_u8_oriented_case(
        ch: usize,
        wd: usize,
        ht: usize,
        fwd: usize,
        fht: usize,
        stride: usize,
        orientation: i32,
        black: f32,
        white: f32,
    ) {
        // Exact touched spans: sentinel-guarded buffers with the work window
        // offset by one element, so an over/under-run by a single lane or
        // byte fails. Both the reference and the kernel start from the same
        // sentinel, so agreement on the whole window also pins the untouched
        // lanes (ch..4) and displaced-extent gaps.
        const SENTINEL: f32 = -7.0;
        let layout = oriented_layout(1, wd, ht, fwd, fht, stride, orientation).unwrap();
        let out_lanes = layout.out_bytes * 4;
        let need_in = (ht - 1) * stride + ch * wd;
        let inp: Box<[u8]> = (0..need_in)
            .map(|i| ((i as u64 * 2_654_435_761 + 0x9E37) % 256) as u8)
            .collect();
        let original = inp.clone();
        let mut expected = vec![SENTINEL; out_lanes + 2];
        ref_u8_to_float_oriented(
            &mut expected[1..1 + out_lanes],
            &inp,
            black,
            white,
            ch,
            wd,
            ht,
            fwd,
            fht,
            stride,
            orientation,
        );
        let mut safe = vec![SENTINEL; out_lanes + 2];
        u8_to_float_oriented(
            &mut safe[1..1 + out_lanes],
            &inp,
            black,
            white,
            ch,
            wd,
            ht,
            fwd,
            fht,
            stride,
            orientation,
        );
        assert_bits_eq(&safe, &expected);
        let mut ffi = vec![SENTINEL; out_lanes + 2];
        unsafe {
            darkroom_imageio_u8_to_float_oriented(
                ffi.as_mut_ptr().add(1),
                inp.as_ptr(),
                black,
                white,
                ch as i32,
                wd as i32,
                ht as i32,
                fwd as i32,
                fht as i32,
                stride as i32,
                orientation,
            );
        }
        assert_bits_eq(&ffi, &expected);
        assert_eq!(inp, original);
    }

    #[test]
    fn u8_to_float_oriented_matches_reference_over_sweep() {
        for orientation in [0, 1, 2, 3, 4, 5, 6, 7, -1] {
            for (wd, ht) in [(1, 1), (7, 1), (1, 9), (3, 5), (5, 3), (17, 9)] {
                for ch in [1usize, 2, 3, 4] {
                    for stride in [0, ch * wd - 1, ch * wd, ch * wd + 7] {
                        for (black, white) in [
                            (0.0f32, 255.0f32),
                            (16.0, 235.0),
                            (255.0, 0.0),
                            (5.0, 5.0),
                        ] {
                            assert_u8_oriented_case(
                                ch,
                                wd,
                                ht,
                                wd,
                                ht,
                                stride,
                                orientation,
                                black,
                                white,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn u8_to_float_oriented_displaced_extents_and_gaps() {
        for orientation in [0, 1, 2, 3, 4, 5, 6, 7, -1] {
            for (wd, ht) in [(1, 1), (1, 7), (9, 1), (3, 5), (5, 3)] {
                for (fwd, fht) in [(wd, ht), (wd + 1, ht), (wd, ht + 2), (wd + 7, ht + 9)] {
                    assert_u8_oriented_case(3, wd, ht, fwd, fht, 3 * wd + 5, orientation, 0.0, 255.0);
                }
                let fwd = if orientation & 2 == 0 { 1 } else { wd };
                let fht = if orientation & 1 == 0 { 1 } else { ht };
                assert_u8_oriented_case(2, wd, ht, fwd, fht, 0, orientation, 16.0, 235.0);
            }
        }
    }

    #[test]
    fn u8_to_float_oriented_golden_vectors_from_c_formula() {
        // Hand-derived from the pre-m4-227 C si/sj loop in
        // src/imageio/imageio.c with wd = 2, ht = 3, ch = 1, fwd = 2,
        // fht = 3, stride = 2, black = 0, white = 255: si starts at 4
        // lanes, sj at wd * 4 = 8; SWAP_XY swaps them to sj = 4,
        // si = ht * 4 = 12. Input bytes are [0, 1, 2, 3, 4, 5].
        // Orientation 5 (FLIP_Y plus SWAP): jj = 2, sj = -4, ii = 0, so
        // row j = 0 writes quads 2, 5; j = 1 writes quads 1, 4; j = 2
        // writes quads 0, 3, giving lane 0 == [4, 2, 0, 5, 3, 1] / 255.
        // Orientation 6 (FLIP_X plus SWAP): jj = 0, sj = 4, ii = 1,
        // si = -12, so row j = 0 writes quads 3, 0; j = 1 writes quads 4,
        // 1; j = 2 writes quads 5, 2, giving [1, 3, 5, 0, 2, 4] / 255.
        // Orientation -1 has every low bit set, so it takes the same
        // branches as 7 (transverse): jj = 2, sj = -4, ii = 1, si = -12,
        // giving [5, 3, 1, 4, 2, 0] / 255.
        const SENTINEL: f32 = -7.0;
        fn run_safe(inp: &[u8; 6], orientation: i32) -> [f32; 26] {
            let mut out = [SENTINEL; 26];
            u8_to_float_oriented(&mut out[1..25], inp, 0.0, 255.0, 1, 2, 3, 2, 3, 2, orientation);
            out
        }
        fn run_ffi(inp: &[u8; 6], orientation: i32) -> [f32; 26] {
            let mut out = [SENTINEL; 26];
            unsafe {
                darkroom_imageio_u8_to_float_oriented(
                    out.as_mut_ptr().add(1),
                    inp.as_ptr(),
                    0.0,
                    255.0,
                    1,
                    2,
                    3,
                    2,
                    3,
                    2,
                    orientation,
                );
            }
            out
        }
        fn lane0(order: [u8; 6]) -> [f32; 6] {
            order.map(|b| b as f32 * (1.0f32 / 255.0))
        }
        let inp = [0u8, 1, 2, 3, 4, 5];
        for (orientation, order) in [
            (5, [4u8, 2, 0, 5, 3, 1]),
            (6, [1u8, 3, 5, 0, 2, 4]),
            (7, [5u8, 3, 1, 4, 2, 0]),
            (-1, [5u8, 3, 1, 4, 2, 0]),
        ] {
            let want = lane0(order);
            for out in [run_safe(&inp, orientation), run_ffi(&inp, orientation)] {
                // Lane 0 of each quad is one element per 4 lanes, not a
                // contiguous run: gather it before comparing.
                let got: Vec<f32> = (0..6).map(|q| out[1 + 4 * q]).collect();
                assert_bits_eq(&got, &want);
                // Only lane 0 of each quad is stored with ch == 1; lanes
                // 1..4 keep the sentinel, as do the guard elements.
                for q in 0..6 {
                    for lane in 1..4 {
                        assert_eq!(out[1 + 4 * q + lane].to_bits(), SENTINEL.to_bits());
                    }
                }
                assert_eq!(out[0].to_bits(), SENTINEL.to_bits());
                assert_eq!(out[25].to_bits(), SENTINEL.to_bits());
            }
        }
        assert_bits_eq(&run_safe(&inp, -1)[1..25], &run_safe(&inp, 7)[1..25]);
        assert_bits_eq(&run_ffi(&inp, -1)[1..25], &run_ffi(&inp, 7)[1..25]);
    }

    #[test]
    fn u8_to_float_oriented_skips_stride_padding() {
        let (wd, ht, ch) = (3usize, 2usize, 4usize);
        let tight: Vec<u8> = (0..(ch * wd * ht) as u8).collect();
        let stride = ch * wd + 2;
        let mut padded = vec![0xABu8; ht * stride];
        for j in 0..ht {
            padded[j * stride..j * stride + ch * wd]
                .copy_from_slice(&tight[j * ch * wd..(j + 1) * ch * wd]);
        }
        for orientation in [0, 1, 2, 3, 4, 5, 6, 7] {
            let layout = oriented_layout(1, wd, ht, wd, ht, stride, orientation).unwrap();
            let out_lanes = layout.out_bytes * 4;
            let mut from_tight = vec![0.0f32; out_lanes];
            let mut from_padded = vec![0.0f32; out_lanes];
            u8_to_float_oriented(
                &mut from_tight,
                &tight,
                16.0,
                235.0,
                ch,
                wd,
                ht,
                wd,
                ht,
                ch * wd,
                orientation,
            );
            u8_to_float_oriented(
                &mut from_padded,
                &padded,
                16.0,
                235.0,
                ch,
                wd,
                ht,
                wd,
                ht,
                stride,
                orientation,
            );
            assert_bits_eq(&from_tight, &from_padded);
        }
    }

    #[test]
    fn u8_to_float_oriented_single_row_ignores_signed_stride() {
        let inp = [0u8, 1, 2, 3, 4, 5, 6, 7];
        for orientation in [0, 1, 2, 3, 4, 5, 6, 7, -1] {
            let mut baseline = vec![-7.0f32; 16];
            u8_to_float_oriented(&mut baseline, &inp, 0.0, 255.0, 4, 2, 1, 2, 1, 0, orientation);
            for stride in [i32::MIN, -1, 0, 1, 7, i32::MAX] {
                let mut out = vec![-7.0f32; 16];
                unsafe {
                    darkroom_imageio_u8_to_float_oriented(
                        out.as_mut_ptr(),
                        inp.as_ptr(),
                        0.0,
                        255.0,
                        4,
                        2,
                        1,
                        2,
                        1,
                        stride,
                        orientation,
                    );
                }
                assert_bits_eq(&out, &baseline);
            }
        }
    }

    #[test]
    fn u8_to_float_oriented_ffi_maybe_uninit_storage() {
        use std::mem::MaybeUninit;

        for orientation in [0, 1, 2, 3, 4, 5, 6, 7] {
            for ch in [1usize, 2, 3, 4] {
                let (wd, ht) = (3usize, 5usize);
                let stride = ch * wd + 5;
                let layout = oriented_layout(1, wd, ht, wd, ht, stride, orientation).unwrap();
                let out_lanes = layout.out_bytes * 4;
                let need_in = (ht - 1) * stride + ch * wd;
                // Payload rows initialised, padding left uninitialised: the
                // kernel must never read padding (compared lane-exactly
                // against the safe kernel over fully initialised copies).
                let mut init_inp = vec![0u8; need_in];
                let mut partial =
                    vec![MaybeUninit::<u8>::uninit(); need_in].into_boxed_slice();
                for j in 0..ht {
                    for k in 0..ch * wd {
                        let byte = ((j * ch * wd + k) % 251) as u8;
                        init_inp[j * stride + k] = byte;
                        partial[j * stride + k].write(byte);
                    }
                }
                let mut init_out = vec![-7.0f32; out_lanes];
                u8_to_float_oriented(
                    &mut init_out,
                    &init_inp,
                    16.0,
                    235.0,
                    ch,
                    wd,
                    ht,
                    wd,
                    ht,
                    stride,
                    orientation,
                );
                let mut ffi_out =
                    vec![MaybeUninit::<f32>::uninit(); out_lanes].into_boxed_slice();
                unsafe {
                    darkroom_imageio_u8_to_float_oriented(
                        ffi_out.as_mut_ptr().cast(),
                        partial.as_ptr().cast(),
                        16.0,
                        235.0,
                        ch as i32,
                        wd as i32,
                        ht as i32,
                        wd as i32,
                        ht as i32,
                        stride as i32,
                        orientation,
                    );
                }
                // Every written lane agrees; lanes ch..4 were never stored.
                for j in 0..ht {
                    for i in 0..wd {
                        let dst = layout.destination(i, j, 1) * 4;
                        for k in 0..ch {
                            assert_eq!(
                                unsafe { ffi_out[dst + k].assume_init() }.to_bits(),
                                init_out[dst + k].to_bits()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn u8_to_float_oriented_safe_degenerate_and_short_buffers() {
        const SENTINEL: f32 = -7.0;
        let mut out = [SENTINEL; 16];
        let inp = [9u8; 16];
        for (ch, wd, ht, fwd, fht, stride, orientation) in [
            (0, 2, 2, 2, 2, 8, 0),
            (5, 2, 2, 2, 2, 8, 0),
            (4, 0, 2, 2, 2, 8, 0),
            (4, 2, 0, 2, 2, 8, 0),
            (4, 2, 2, 0, 2, 8, 0),
            (4, 2, 2, 2, 0, 8, 0),
            (4, 2, 2, 1, 2, 8, 2),
            (4, 2, 2, 2, 1, 8, 1),
            (4, 2, 2, 2, 2, usize::MAX, 0),
            (4, 1, 2, 1, 2, usize::MAX, 0),
            (usize::MAX, 2, 1, 2, 1, 0, 0),
            // Accepted layouts whose spans dwarf the fixtures: the safe
            // kernel re-checks real slice lengths, so these are genuine
            // no-ops here (the FFI cannot take this path, having no length
            // args to re-check against).
            (4, 1, 2, i32::MAX as usize, i32::MAX as usize, 0, 1),
            (
                4,
                i32::MAX as usize,
                i32::MAX as usize,
                i32::MAX as usize,
                i32::MAX as usize,
                i32::MAX as usize,
                7,
            ),
        ] {
            u8_to_float_oriented(&mut out, &inp, 0.0, 255.0, ch, wd, ht, fwd, fht, stride, orientation);
            assert_eq!(out, [SENTINEL; 16]);
        }
        // Short buffers on either side are no-ops.
        u8_to_float_oriented(&mut out[..15], &inp, 0.0, 255.0, 4, 2, 2, 2, 2, 8, 0);
        u8_to_float_oriented(&mut out, &inp[..15], 0.0, 255.0, 4, 2, 2, 2, 2, 8, 0);
        u8_to_float_oriented(&mut [], &[], 0.0, 255.0, 1, 1, 1, 1, 1, 1, 0);
        assert_eq!(out, [SENTINEL; 16]);
        assert_eq!(inp, [9u8; 16]);
    }

    #[test]
    fn u8_to_float_oriented_ffi_null_dimension_stride_and_overflow_guards() {
        const SENTINEL: f32 = 3.0;
        let inp = [9u8; 64];
        let mut out = [SENTINEL; 64];
        unsafe {
            darkroom_imageio_u8_to_float_oriented(
                std::ptr::null_mut(),
                inp.as_ptr(),
                0.0,
                255.0,
                4,
                2,
                2,
                2,
                2,
                8,
                0,
            );
            darkroom_imageio_u8_to_float_oriented(
                out.as_mut_ptr(),
                std::ptr::null(),
                0.0,
                255.0,
                4,
                2,
                2,
                2,
                2,
                8,
                0,
            );
            for (ch, wd, ht, fwd, fht, stride, orientation) in [
                (0, 2, 2, 2, 2, 8, 0),
                (5, 2, 2, 2, 2, 8, 0),
                (-1, 2, 2, 2, 2, 8, 0),
                (4, 0, 2, 2, 2, 8, 0),
                (4, -1, 2, 2, 2, 8, 0),
                (4, 2, 0, 2, 2, 8, 0),
                (4, 2, -1, 2, 2, 8, 0),
                (4, 2, 2, 0, 2, 8, 0),
                (4, 2, 2, -1, 2, 8, 0),
                (4, 2, 2, 2, 0, 8, 0),
                (4, 2, 2, 2, -1, 8, 0),
                (4, 2, 2, 1, 2, 8, 2),
                (4, 2, 2, 2, 1, 8, 1),
                (4, 2, 2, 2, 2, -1, 0),
                (4, 2, 2, 2, 2, i32::MIN, 7),
                (4, i32::MAX, i32::MAX, i32::MAX, i32::MAX, i32::MAX, 7),
                // (4, 1, 2, i32::MAX, i32::MAX, 0, 1) is a legitimate
                // accepted layout (the span fits isize) but the 64-lane
                // stack fixture would be overrun since the FFI takes no
                // length args, so it is skipped per the caller-size
                // contract, cf. m4-226; the safe kernel covers it below.
            ] {
                darkroom_imageio_u8_to_float_oriented(
                    out.as_mut_ptr(),
                    inp.as_ptr(),
                    0.0,
                    255.0,
                    ch,
                    wd,
                    ht,
                    fwd,
                    fht,
                    stride,
                    orientation,
                );
                assert_eq!(out, [SENTINEL; 64]);
            }
        }
        assert_eq!(out, [SENTINEL; 64]);
        assert_eq!(inp, [9u8; 64]);
    }

    // rails pinned as exact bit patterns for the PGM gray row (m4-228):
    // 0 scales to +0.0, maxval to exactly 1.0, and every alpha lane is
    // exactly +0.0. Both rails are forced by IEEE arithmetic (not by the
    // implementation), so they pin the kernel to the C operation.
    #[test]
    fn pnm_pgm_u8_row_rails_pin() {
        let line = [0u8, 1, 128, 254, 255];
        let mut out = vec![7.0f32; 20];
        pnm_pgm_u8_row_to_float(&mut out, &line, 5, 255);
        assert_eq!(out[0].to_bits(), 0x0000_0000); // 0 -> +0.0
        assert_eq!(out[16].to_bits(), 0x3F80_0000); // 255 -> 1.0
        // interior values follow the same division the C loop performs
        // (u8 widens to f32 exactly, so float literals pin the same bits)
        assert_eq!(out[4].to_bits(), (1.0f32 / 255.0).to_bits());
        assert_eq!(out[8].to_bits(), (128.0f32 / 255.0).to_bits());
        assert_eq!(out[12].to_bits(), (254.0f32 / 255.0).to_bits());
        // gray fan-out plus zeroed alpha on every quad
        for x in 0..5 {
            let b = 4 * x;
            assert_eq!(out[b].to_bits(), out[b + 1].to_bits(), "x={x}");
            assert_eq!(out[b].to_bits(), out[b + 2].to_bits(), "x={x}");
            assert_eq!(out[b + 3].to_bits(), 0x0000_0000, "x={x}");
        }
        // maxval 1: every nonzero byte is exactly 1.0
        let mut tiny = vec![7.0f32; 8];
        pnm_pgm_u8_row_to_float(&mut tiny, &[0u8, 1], 2, 1);
        assert_eq!(tiny[0].to_bits(), 0x0000_0000);
        assert_eq!(tiny[4].to_bits(), 0x3F80_0000);
        assert_eq!(tiny[7].to_bits(), 0x0000_0000);
    }

    // every byte value at several widths and maxvals, sentinel-padded:
    // kernel and reference must agree bit-exactly on every lane, and only
    // the exact 4*width span may be written.
    #[test]
    fn pnm_pgm_u8_row_matches_reference() {
        for width in [1usize, 3, 17, 65] {
            for max in [1u32, 2, 100, 255] {
                let mut line = Vec::with_capacity(width);
                for i in 0..width {
                    // LCG over the full 0..=255 byte range
                    let v = (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(0x9E37) % 256;
                    line.push(v as u8);
                }
                line[0] = 0;
                if width > 1 {
                    line[1] = max.min(255) as u8;
                }
                // head/tail sentinel lanes around the exact 4*width span
                let mut direct = vec![-1.0f32; 4 * width + 8];
                let mut reference = vec![-2.0f32; 4 * width + 8];
                pnm_pgm_u8_row_to_float(&mut direct[4..4 + 4 * width], &line, width, max);
                ref_pnm_pgm_u8_row_to_float(
                    &line,
                    &mut reference[4..4 + 4 * width],
                    width,
                    max,
                );
                assert_eq!(direct[..4], [-1.0; 4], "direct head");
                assert_eq!(direct[4 + 4 * width..], [-1.0; 4], "direct tail");
                assert_eq!(reference[..4], [-2.0; 4], "reference head");
                assert_eq!(reference[4 + 4 * width..], [-2.0; 4], "reference tail");
                for k in 0..4 * width {
                    assert_eq!(
                        direct[4 + k].to_bits(),
                        reference[4 + k].to_bits(),
                        "lane {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn pnm_pgm_u8_row_degenerate_guards_no_op() {
        let line = vec![200u8; 16];
        // zero width: output untouched
        let mut out = vec![9.0f32; 16];
        pnm_pgm_u8_row_to_float(&mut out, &line, 0, 255);
        assert_eq!(out, vec![9.0f32; 16]);
        // zero maxval: output untouched (the C branch never sees this)
        let mut out = vec![9.0f32; 16];
        pnm_pgm_u8_row_to_float(&mut out, &line, 4, 0);
        assert_eq!(out, vec![9.0f32; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_out = vec![0.0f32; 7]; // short for 2 quads
        pnm_pgm_u8_row_to_float(&mut short_out, &line, 2, 255);
        let short_line = [200u8; 1]; // short for 2 columns
        let mut out2 = vec![-3.0f32; 8];
        pnm_pgm_u8_row_to_float(&mut out2, &short_line, 2, 255);
        assert_eq!(out2[4..], vec![-3.0f32; 4]); // second quad untouched
        let empty: Vec<u8> = vec![];
        let mut out3 = vec![5.0f32; 4];
        pnm_pgm_u8_row_to_float(&mut out3, &empty, 1, 255);
        assert_eq!(out3, vec![5.0f32; 4]);
    }

    #[test]
    fn pnm_pgm_u8_row_ffi_uninit_row() {
        use std::mem::MaybeUninit;

        // uninitialized row storage is realistic: the C row cursor points
        // into the mipmap-cache allocation, whose lanes the kernel fully
        // overwrites before any read.
        for (width, max) in [(1usize, 1u32), (3, 100), (65, 255)] {
            let mut line = Vec::with_capacity(width);
            for i in 0..width {
                line.push(((i as u64 * 2_654_435_761 + 0x9E37) % 256) as u8);
            }
            let mut ffi_out = vec![MaybeUninit::<f32>::uninit(); 4 * width]
                .into_boxed_slice();
            unsafe {
                darkroom_pnm_pgm_u8_row_to_float(
                    ffi_out.as_mut_ptr().cast(),
                    line.as_ptr(),
                    width,
                    max,
                );
            }
            // every lane initialized by the call: safe to assume_init now
            let ffi_out =
                unsafe { Box::<[f32]>::from_raw(Box::into_raw(ffi_out) as *mut [f32]) };
            let mut direct = vec![0.0f32; 4 * width];
            pnm_pgm_u8_row_to_float(&mut direct, &line, width, max);
            assert_eq!(ffi_out.len(), direct.len());
            for (k, (f, d)) in ffi_out.iter().zip(direct.iter()).enumerate() {
                assert_eq!(f.to_bits(), d.to_bits(), "lane {k}");
            }
        }
    }

    #[test]
    fn pnm_pgm_u8_row_ffi_guards() {
        let line = [200u8; 16];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_pnm_pgm_u8_row_to_float(std::ptr::null_mut(), line.as_ptr(), 4, 255);
            darkroom_pnm_pgm_u8_row_to_float(out.as_mut_ptr(), std::ptr::null(), 4, 255);
            // zero width and zero maxval
            darkroom_pnm_pgm_u8_row_to_float(out.as_mut_ptr(), line.as_ptr(), 0, 255);
            darkroom_pnm_pgm_u8_row_to_float(out.as_mut_ptr(), line.as_ptr(), 4, 0);
            // overflowing lane count (4 * width wraps — rejected)
            darkroom_pnm_pgm_u8_row_to_float(out.as_mut_ptr(), line.as_ptr(), usize::MAX, 255);
            darkroom_pnm_pgm_u8_row_to_float(
                out.as_mut_ptr(),
                line.as_ptr(),
                usize::MAX / 4 + 1,
                255,
            );
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }

    // from big-endian file bytes into the as-loaded u16 words the C fread
    // leaves on a little-endian host (each pair byte-swapped in its slot
    // until the kernel swaps it back). Only the byte order is pinned here;
    // the value assertions live in the tests below.
    fn pgm_u16_words_from_file_bytes(file_bytes: &[u8]) -> Vec<u16> {
        file_bytes.as_chunks::<2>().0.iter().map(|pair| u16::from_le_bytes(*pair))
            .collect()
    }

    // rails pinned as exact bit patterns for the 16-bit PGM gray row
    // (m4-229): 0 scales to +0.0, maxval to exactly 1.0, and every alpha
    // lane is exactly +0.0. The asymmetric 0x1234 word pins the byte swap
    // itself (unswapped it would decode as 0x3412), the odd 0x0001 word
    // pins a swap that changes only the low byte, and 0xFFFF pins the
    // symmetric rail. Both rails are forced by IEEE arithmetic (not by the
    // implementation), so they pin the kernel to the C operation.
    #[test]
    fn pnm_pgm_u16_row_rails_pin() {
        let line = pgm_u16_words_from_file_bytes(&[
            0x00, 0x00, // 0x0000
            0x00, 0x01, // 0x0001, odd value
            0x12, 0x34, // 0x1234, asymmetric: pins swap direction
            0xFF, 0xFE, // 0xFFFE
            0xFF, 0xFF, // 0xFFFF
        ]);
        let mut out = vec![7.0f32; 20];
        pnm_pgm_u16_row_to_float(&mut out, &line, 5, 65535);
        assert_eq!(out[0].to_bits(), 0x0000_0000); // 0 -> +0.0
        assert_eq!(out[16].to_bits(), 0x3F80_0000); // 0xFFFF -> 1.0
        // interior values follow the same division the C loop performs
        // (u16 widens to f32 exactly, so float expressions pin the bits)
        assert_eq!(out[4].to_bits(), (1.0f32 / 65535.0).to_bits());
        assert_eq!(out[8].to_bits(), (0x1234u16 as f32 / 65535.0).to_bits());
        assert_eq!(out[12].to_bits(), (0xFFFEu16 as f32 / 65535.0).to_bits());
        // gray fan-out plus zeroed alpha on every quad
        for x in 0..5 {
            let b = 4 * x;
            assert_eq!(out[b].to_bits(), out[b + 1].to_bits(), "x={x}");
            assert_eq!(out[b].to_bits(), out[b + 2].to_bits(), "x={x}");
            assert_eq!(out[b + 3].to_bits(), 0x0000_0000, "x={x}");
        }
        // smallest maxval of this branch: 0x0000 -> +0.0, 0x0100 -> 1.0
        let tiny = pgm_u16_words_from_file_bytes(&[0x00, 0x00, 0x01, 0x00]);
        let mut tiny_out = vec![7.0f32; 8];
        pnm_pgm_u16_row_to_float(&mut tiny_out, &tiny, 2, 256);
        assert_eq!(tiny_out[0].to_bits(), 0x0000_0000);
        assert_eq!(tiny_out[4].to_bits(), 0x3F80_0000);
        assert_eq!(tiny_out[7].to_bits(), 0x0000_0000);
    }

    // LCG word sweep at several widths and maxvals, sentinel-padded:
    // kernel and reference must agree bit-exactly on every lane, and only
    // the exact 4*width span may be written.
    #[test]
    fn pnm_pgm_u16_row_matches_reference() {
        for width in [1usize, 3, 17, 65] {
            for max in [256u32, 1000, 32768, 65535] {
                // native words covering the full 0..=0xFFFF range, built
                // from big-endian file bytes as fread would leave them
                let mut file_bytes = Vec::with_capacity(2 * width);
                for i in 0..width {
                    // LCG over the full 0..=0xFFFF word range
                    let v = (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(0x9E37) % 65536;
                    file_bytes.extend_from_slice(&(v as u16).to_be_bytes());
                }
                // pin the rails at the sweep edges: first word 0x0000,
                // second word exactly maxval (in file order)
                file_bytes[0] = 0;
                file_bytes[1] = 0;
                if width > 1 {
                    let max_bytes = (max as u16).to_be_bytes();
                    file_bytes[2] = max_bytes[0];
                    file_bytes[3] = max_bytes[1];
                }
                let line = pgm_u16_words_from_file_bytes(&file_bytes);
                // head/tail sentinel lanes around the exact 4*width span
                let mut direct = vec![-1.0f32; 4 * width + 8];
                let mut reference = vec![-2.0f32; 4 * width + 8];
                pnm_pgm_u16_row_to_float(&mut direct[4..4 + 4 * width], &line, width, max);
                ref_pnm_pgm_u16_row_to_float(
                    &line,
                    &mut reference[4..4 + 4 * width],
                    width,
                    max,
                );
                assert_eq!(direct[..4], [-1.0; 4], "direct head");
                assert_eq!(direct[4 + 4 * width..], [-1.0; 4], "direct tail");
                assert_eq!(reference[..4], [-2.0; 4], "reference head");
                assert_eq!(reference[4 + 4 * width..], [-2.0; 4], "reference tail");
                for k in 0..4 * width {
                    assert_eq!(
                        direct[4 + k].to_bits(),
                        reference[4 + k].to_bits(),
                        "lane {k}"
                    );
                }
                // maxval rail decodes to exactly 1.0 through the swap
                assert_eq!(direct[4].to_bits(), 0x0000_0000);
                if width > 1 {
                    assert_eq!(direct[8].to_bits(), 0x3F80_0000);
                }
            }
        }
    }

    #[test]
    fn pnm_pgm_u16_row_degenerate_guards_no_op() {
        let line = vec![0x3412u16; 16];
        // zero width: output untouched
        let mut out = vec![9.0f32; 16];
        pnm_pgm_u16_row_to_float(&mut out, &line, 0, 65535);
        assert_eq!(out, vec![9.0f32; 16]);
        // zero maxval: output untouched (the C branch never sees this)
        let mut out = vec![9.0f32; 16];
        pnm_pgm_u16_row_to_float(&mut out, &line, 4, 0);
        assert_eq!(out, vec![9.0f32; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_out = vec![0.0f32; 7]; // short for 2 quads
        pnm_pgm_u16_row_to_float(&mut short_out, &line, 2, 65535);
        // clamped to a single quad: first quad decodes 0x3412 (swapped),
        // remaining lanes untouched
        let short_expected = (0x3412u16.swap_bytes() as f32 / 65535.0).to_bits();
        assert_eq!(short_out[0].to_bits(), short_expected);
        assert_eq!(short_out[1].to_bits(), short_expected);
        assert_eq!(short_out[2].to_bits(), short_expected);
        assert_eq!(short_out[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(short_out[4..], [0.0f32; 3]);
        let short_line = [0x3412u16; 1]; // short for 2 columns
        let mut out2 = vec![-3.0f32; 8];
        pnm_pgm_u16_row_to_float(&mut out2, &short_line, 2, 65535);
        assert_eq!(out2[4..], vec![-3.0f32; 4]); // second quad untouched
        let empty: Vec<u16> = vec![];
        let mut out3 = vec![5.0f32; 4];
        pnm_pgm_u16_row_to_float(&mut out3, &empty, 1, 65535);
        assert_eq!(out3, vec![5.0f32; 4]);
    }

    #[test]
    fn pnm_pgm_u16_row_ffi_uninit_row() {
        use std::mem::MaybeUninit;

        // uninitialized row storage is realistic: the C row cursor points
        // into the mipmap-cache allocation, whose lanes the kernel fully
        // overwrites before any read.
        for (width, max) in [(1usize, 256u32), (3, 1000), (65, 65535)] {
            let line: Vec<u16> = (0..width)
                .map(|i| {
                    let v = ((i as u64 * 2_654_435_761 + 0x9E37) % 65536) as u16;
                    u16::from_le_bytes(v.to_be_bytes())
                })
                .collect();
            let mut ffi_out = vec![MaybeUninit::<f32>::uninit(); 4 * width]
                .into_boxed_slice();
            unsafe {
                darkroom_pnm_pgm_u16_row_to_float(
                    ffi_out.as_mut_ptr().cast(),
                    line.as_ptr(),
                    width,
                    max,
                );
            }
            // every lane initialized by the call: safe to assume_init now
            let ffi_out =
                unsafe { Box::<[f32]>::from_raw(Box::into_raw(ffi_out) as *mut [f32]) };
            let mut direct = vec![0.0f32; 4 * width];
            pnm_pgm_u16_row_to_float(&mut direct, &line, width, max);
            assert_eq!(ffi_out.len(), direct.len());
            for (k, (f, d)) in ffi_out.iter().zip(direct.iter()).enumerate() {
                assert_eq!(f.to_bits(), d.to_bits(), "lane {k}");
            }
        }
    }

    #[test]
    fn pnm_pgm_u16_row_ffi_guards() {
        let line = [0x3412u16; 16];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_pnm_pgm_u16_row_to_float(std::ptr::null_mut(), line.as_ptr(), 4, 65535);
            darkroom_pnm_pgm_u16_row_to_float(out.as_mut_ptr(), std::ptr::null(), 4, 65535);
            // zero width and zero maxval
            darkroom_pnm_pgm_u16_row_to_float(out.as_mut_ptr(), line.as_ptr(), 0, 65535);
            darkroom_pnm_pgm_u16_row_to_float(out.as_mut_ptr(), line.as_ptr(), 4, 0);
            // overflowing lane count (4 * width wraps — rejected)
            darkroom_pnm_pgm_u16_row_to_float(out.as_mut_ptr(), line.as_ptr(), usize::MAX, 65535);
            darkroom_pnm_pgm_u16_row_to_float(
                out.as_mut_ptr(),
                line.as_ptr(),
                usize::MAX / 4 + 1,
                65535,
            );
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }

    // rails pinned as exact bit patterns for the 8-bit PPM triplet row
    // (m4-230): 0 scales to +0.0, maxval to exactly 1.0, and every alpha
    // lane is exactly +0.0. The asymmetric middle triplet pins the direct
    // per-lane write (a gray fan-out would collapse its lanes); both rails
    // are forced by IEEE arithmetic (not by the implementation), so they
    // pin the kernel to the C operation.
    #[test]
    fn pnm_ppm_u8_row_rails_pin() {
        let line = [
            0u8, 0, 0, // black triplet
            255, 255, 255, // white triplet
            10, 128, 250, // asymmetric: pins lane interleave, not fan-out
        ];
        let mut out = vec![7.0f32; 12];
        pnm_ppm_u8_row_to_float(&mut out, &line, 3, 255);
        assert_eq!(out[0].to_bits(), 0x0000_0000); // 0 -> +0.0
        assert_eq!(out[4].to_bits(), 0x3F80_0000); // 255 -> 1.0
        assert_eq!(out[5].to_bits(), 0x3F80_0000);
        assert_eq!(out[6].to_bits(), 0x3F80_0000);
        // interior lanes follow the same division the C loop performs
        // (u8 widens to f32 exactly, so float expressions pin the bits)
        assert_eq!(out[8].to_bits(), (10.0f32 / 255.0).to_bits());
        assert_eq!(out[9].to_bits(), (128.0f32 / 255.0).to_bits());
        assert_eq!(out[10].to_bits(), (250.0f32 / 255.0).to_bits());
        // lanes stay distinct: a fan-out bug would collapse them
        assert_ne!(out[8].to_bits(), out[9].to_bits());
        assert_ne!(out[9].to_bits(), out[10].to_bits());
        // zeroed alpha on every quad
        for x in 0..3 {
            assert_eq!(out[4 * x + 3].to_bits(), 0x0000_0000, "x={x}");
        }
        // maxval 1: every nonzero byte is exactly 1.0
        let mut tiny = vec![7.0f32; 8];
        pnm_ppm_u8_row_to_float(&mut tiny, &[0u8, 0, 0, 1, 1, 1], 2, 1);
        assert_eq!(tiny[0].to_bits(), 0x0000_0000);
        assert_eq!(tiny[4].to_bits(), 0x3F80_0000);
        assert_eq!(tiny[5].to_bits(), 0x3F80_0000);
        assert_eq!(tiny[6].to_bits(), 0x3F80_0000);
        assert_eq!(tiny[7].to_bits(), 0x0000_0000);
    }

    // every byte value at several widths and maxvals, sentinel-padded:
    // kernel and reference must agree bit-exactly on every lane, and only
    // the exact 4*width span may be written.
    #[test]
    fn pnm_ppm_u8_row_matches_reference() {
        for width in [1usize, 3, 17, 65] {
            for max in [1u32, 2, 100, 255] {
                let mut line = Vec::with_capacity(3 * width);
                for i in 0..3 * width {
                    // LCG over the full 0..=255 byte range, so the three
                    // lanes of each triplet generally differ (interleave
                    // coverage, not just fan-out-compatible grays)
                    let v = (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(0x9E37) % 256;
                    line.push(v as u8);
                }
                line[0] = 0;
                line[1] = 0;
                line[2] = 0;
                if width > 1 {
                    let m = max.min(255) as u8;
                    line[3] = m;
                    line[4] = m;
                    line[5] = m;
                }
                // head/tail sentinel lanes around the exact 4*width span
                let mut direct = vec![-1.0f32; 4 * width + 8];
                let mut reference = vec![-2.0f32; 4 * width + 8];
                pnm_ppm_u8_row_to_float(&mut direct[4..4 + 4 * width], &line, width, max);
                ref_pnm_ppm_u8_row_to_float(
                    &line,
                    &mut reference[4..4 + 4 * width],
                    width,
                    max,
                );
                assert_eq!(direct[..4], [-1.0; 4], "direct head");
                assert_eq!(direct[4 + 4 * width..], [-1.0; 4], "direct tail");
                assert_eq!(reference[..4], [-2.0; 4], "reference head");
                assert_eq!(reference[4 + 4 * width..], [-2.0; 4], "reference tail");
                for k in 0..4 * width {
                    assert_eq!(
                        direct[4 + k].to_bits(),
                        reference[4 + k].to_bits(),
                        "lane {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn pnm_ppm_u8_row_degenerate_guards_no_op() {
        let line = vec![200u8; 48];
        // zero width: output untouched
        let mut out = vec![9.0f32; 16];
        pnm_ppm_u8_row_to_float(&mut out, &line, 0, 255);
        assert_eq!(out, vec![9.0f32; 16]);
        // zero maxval: output untouched (the C branch never sees this)
        let mut out = vec![9.0f32; 16];
        pnm_ppm_u8_row_to_float(&mut out, &line, 4, 0);
        assert_eq!(out, vec![9.0f32; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_out = vec![0.0f32; 7]; // short for 2 quads
        pnm_ppm_u8_row_to_float(&mut short_out, &line, 2, 255);
        // clamped to a single quad: first triplet decodes per lane,
        // remaining lanes untouched
        let expected = (200.0f32 / 255.0).to_bits();
        assert_eq!(short_out[0].to_bits(), expected);
        assert_eq!(short_out[1].to_bits(), expected);
        assert_eq!(short_out[2].to_bits(), expected);
        assert_eq!(short_out[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(short_out[4..], [0.0f32; 3]);
        let short_line = [200u8; 4]; // one triplet plus a stray byte: short for 2 columns
        let mut out2 = vec![-3.0f32; 8];
        pnm_ppm_u8_row_to_float(&mut out2, &short_line, 2, 255);
        assert_eq!(out2[0].to_bits(), expected);
        assert_eq!(out2[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(out2[4..], vec![-3.0f32; 4]); // second quad untouched
        let empty: Vec<u8> = vec![];
        let mut out3 = vec![5.0f32; 4];
        pnm_ppm_u8_row_to_float(&mut out3, &empty, 1, 255);
        assert_eq!(out3, vec![5.0f32; 4]);
    }

    #[test]
    fn pnm_ppm_u8_row_ffi_uninit_row() {
        use std::mem::MaybeUninit;

        // uninitialized row storage is realistic: the C row cursor points
        // into the mipmap-cache allocation, whose lanes the kernel fully
        // overwrites before any read.
        for (width, max) in [(1usize, 1u32), (3, 100), (65, 255)] {
            let mut line = Vec::with_capacity(3 * width);
            for i in 0..3 * width {
                line.push(((i as u64 * 2_654_435_761 + 0x9E37) % 256) as u8);
            }
            let mut ffi_out = vec![MaybeUninit::<f32>::uninit(); 4 * width]
                .into_boxed_slice();
            unsafe {
                darkroom_pnm_ppm_u8_row_to_float(
                    ffi_out.as_mut_ptr().cast(),
                    line.as_ptr(),
                    width,
                    max,
                );
            }
            // every lane initialized by the call: safe to assume_init now
            let ffi_out =
                unsafe { Box::<[f32]>::from_raw(Box::into_raw(ffi_out) as *mut [f32]) };
            let mut direct = vec![0.0f32; 4 * width];
            pnm_ppm_u8_row_to_float(&mut direct, &line, width, max);
            assert_eq!(ffi_out.len(), direct.len());
            for (k, (f, d)) in ffi_out.iter().zip(direct.iter()).enumerate() {
                assert_eq!(f.to_bits(), d.to_bits(), "lane {k}");
            }
        }
    }

    #[test]
    fn pnm_ppm_u8_row_ffi_guards() {
        let line = [200u8; 48];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_pnm_ppm_u8_row_to_float(std::ptr::null_mut(), line.as_ptr(), 4, 255);
            darkroom_pnm_ppm_u8_row_to_float(out.as_mut_ptr(), std::ptr::null(), 4, 255);
            // zero width and zero maxval
            darkroom_pnm_ppm_u8_row_to_float(out.as_mut_ptr(), line.as_ptr(), 0, 255);
            darkroom_pnm_ppm_u8_row_to_float(out.as_mut_ptr(), line.as_ptr(), 4, 0);
            // overflowing lane counts (either product wraps — rejected
            // before building any slice)
            darkroom_pnm_ppm_u8_row_to_float(out.as_mut_ptr(), line.as_ptr(), usize::MAX, 255);
            darkroom_pnm_ppm_u8_row_to_float(
                out.as_mut_ptr(),
                line.as_ptr(),
                usize::MAX / 4 + 1,
                255,
            );
            darkroom_pnm_ppm_u8_row_to_float(
                out.as_mut_ptr(),
                line.as_ptr(),
                usize::MAX / 3 + 1,
                255,
            );
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }

    // rails pinned as exact bit patterns for the 16-bit PPM triplet row
    // (m4-231): 0 scales to +0.0, maxval to exactly 1.0, and every alpha
    // lane is exactly +0.0. The asymmetric 0x1234 word pins the byte swap
    // itself (unswapped it would decode as 0x3412), the odd 0x0001 word
    // pins a swap that changes only the low byte, and the distinct lanes
    // of the third triplet pin the direct per-lane write (a gray fan-out
    // would collapse them). Both rails are forced by IEEE arithmetic (not
    // by the implementation), so they pin the kernel to the C operation.
    #[test]
    fn pnm_ppm_u16_row_rails_pin() {
        let line = pgm_u16_words_from_file_bytes(&[
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // black triplet
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // white triplet
            0x12, 0x34, 0x00, 0x01, 0xFF, 0xFE, // asymmetric triplet
        ]);
        let mut out = vec![7.0f32; 12];
        pnm_ppm_u16_row_to_float(&mut out, &line, 3, 65535);
        assert_eq!(out[0].to_bits(), 0x0000_0000); // 0 -> +0.0
        assert_eq!(out[4].to_bits(), 0x3F80_0000); // 0xFFFF -> 1.0
        assert_eq!(out[5].to_bits(), 0x3F80_0000);
        assert_eq!(out[6].to_bits(), 0x3F80_0000);
        // interior lanes follow the same division the C loop performs
        // (u16 widens to f32 exactly, so float expressions pin the bits)
        assert_eq!(out[8].to_bits(), (0x1234u16 as f32 / 65535.0).to_bits());
        assert_eq!(out[9].to_bits(), (1.0f32 / 65535.0).to_bits());
        assert_eq!(out[10].to_bits(), (0xFFFEu16 as f32 / 65535.0).to_bits());
        // lanes stay distinct: a fan-out bug would collapse them
        assert_ne!(out[8].to_bits(), out[9].to_bits());
        assert_ne!(out[9].to_bits(), out[10].to_bits());
        assert_ne!(out[8].to_bits(), out[10].to_bits());
        // zeroed alpha on every quad
        for x in 0..3 {
            assert_eq!(out[4 * x + 3].to_bits(), 0x0000_0000, "x={x}");
        }
        // smallest maxval of this branch: 0x0000 -> +0.0, 0x0100 -> 1.0
        let tiny = pgm_u16_words_from_file_bytes(&[
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // black triplet
            0x01, 0x00, 0x01, 0x00, 0x01, 0x00, // white triplet at max 256
        ]);
        let mut tiny_out = vec![7.0f32; 8];
        pnm_ppm_u16_row_to_float(&mut tiny_out, &tiny, 2, 256);
        assert_eq!(tiny_out[0].to_bits(), 0x0000_0000);
        assert_eq!(tiny_out[4].to_bits(), 0x3F80_0000);
        assert_eq!(tiny_out[5].to_bits(), 0x3F80_0000);
        assert_eq!(tiny_out[6].to_bits(), 0x3F80_0000);
        assert_eq!(tiny_out[7].to_bits(), 0x0000_0000);
    }

    // LCG word sweep at several widths and maxvals, sentinel-padded:
    // kernel and reference must agree bit-exactly on every lane, and only
    // the exact 4*width span may be written. Words run through big-endian
    // file bytes (swap-direction coverage) and the three lanes of each
    // triplet generally differ (interleave coverage, not just
    // fan-out-compatible grays).
    #[test]
    fn pnm_ppm_u16_row_matches_reference() {
        for width in [1usize, 3, 17, 65] {
            for max in [256u32, 1000, 32768, 65535] {
                // native words covering the full 0..=0xFFFF range, built
                // from big-endian file bytes as fread would leave them
                let mut file_bytes = Vec::with_capacity(6 * width);
                for i in 0..3 * width {
                    // LCG over the full 0..=0xFFFF word range
                    let v = (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(0x9E37) % 65536;
                    file_bytes.extend_from_slice(&(v as u16).to_be_bytes());
                }
                // pin the rails at the sweep edges: first triplet 0x0000,
                // second triplet exactly maxval (in file order)
                for b in file_bytes.iter_mut().take(6) {
                    *b = 0;
                }
                if width > 1 {
                    let max_bytes = (max as u16).to_be_bytes();
                    for c in 0..3 {
                        file_bytes[6 + 2 * c] = max_bytes[0];
                        file_bytes[6 + 2 * c + 1] = max_bytes[1];
                    }
                }
                let line = pgm_u16_words_from_file_bytes(&file_bytes);
                // head/tail sentinel lanes around the exact 4*width span
                let mut direct = vec![-1.0f32; 4 * width + 8];
                let mut reference = vec![-2.0f32; 4 * width + 8];
                pnm_ppm_u16_row_to_float(&mut direct[4..4 + 4 * width], &line, width, max);
                ref_pnm_ppm_u16_row_to_float(
                    &line,
                    &mut reference[4..4 + 4 * width],
                    width,
                    max,
                );
                assert_eq!(direct[..4], [-1.0; 4], "direct head");
                assert_eq!(direct[4 + 4 * width..], [-1.0; 4], "direct tail");
                assert_eq!(reference[..4], [-2.0; 4], "reference head");
                assert_eq!(reference[4 + 4 * width..], [-2.0; 4], "reference tail");
                for k in 0..4 * width {
                    assert_eq!(
                        direct[4 + k].to_bits(),
                        reference[4 + k].to_bits(),
                        "lane {k}"
                    );
                }
                // rails decode through the swap: black -> +0.0, maxval -> 1.0
                assert_eq!(direct[4].to_bits(), 0x0000_0000);
                assert_eq!(direct[5].to_bits(), 0x0000_0000);
                assert_eq!(direct[6].to_bits(), 0x0000_0000);
                if width > 1 {
                    assert_eq!(direct[8].to_bits(), 0x3F80_0000);
                    assert_eq!(direct[9].to_bits(), 0x3F80_0000);
                    assert_eq!(direct[10].to_bits(), 0x3F80_0000);
                }
            }
        }
    }

    #[test]
    fn pnm_ppm_u16_row_degenerate_guards_no_op() {
        let line = vec![0x3412u16; 16];
        // zero width: output untouched
        let mut out = vec![9.0f32; 16];
        pnm_ppm_u16_row_to_float(&mut out, &line, 0, 65535);
        assert_eq!(out, vec![9.0f32; 16]);
        // zero maxval: output untouched (the C branch never sees this)
        let mut out = vec![9.0f32; 16];
        pnm_ppm_u16_row_to_float(&mut out, &line, 4, 0);
        assert_eq!(out, vec![9.0f32; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_out = vec![0.0f32; 7]; // short for 2 quads
        pnm_ppm_u16_row_to_float(&mut short_out, &line, 2, 65535);
        // clamped to a single triplet: first quad decodes 0x3412 (swapped)
        // per lane, remaining lanes untouched
        let short_expected = (0x3412u16.swap_bytes() as f32 / 65535.0).to_bits();
        assert_eq!(short_out[0].to_bits(), short_expected);
        assert_eq!(short_out[1].to_bits(), short_expected);
        assert_eq!(short_out[2].to_bits(), short_expected);
        assert_eq!(short_out[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(short_out[4..], [0.0f32; 3]);
        let short_line = [0x3412u16; 4]; // one triplet plus a stray word: short for 2 columns
        let mut out2 = vec![-3.0f32; 8];
        pnm_ppm_u16_row_to_float(&mut out2, &short_line, 2, 65535);
        assert_eq!(out2[0].to_bits(), short_expected);
        assert_eq!(out2[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(out2[4..], vec![-3.0f32; 4]); // second quad untouched
        let empty: Vec<u16> = vec![];
        let mut out3 = vec![5.0f32; 4];
        pnm_ppm_u16_row_to_float(&mut out3, &empty, 1, 65535);
        assert_eq!(out3, vec![5.0f32; 4]);
    }

    #[test]
    fn pnm_ppm_u16_row_ffi_uninit_row() {
        use std::mem::MaybeUninit;

        // uninitialized row storage is realistic: the C row cursor points
        // into the mipmap-cache allocation, whose lanes the kernel fully
        // overwrites before any read.
        for (width, max) in [(1usize, 256u32), (3, 1000), (65, 65535)] {
            let line: Vec<u16> = (0..3 * width)
                .map(|i| {
                    let v = ((i as u64 * 2_654_435_761 + 0x9E37) % 65536) as u16;
                    u16::from_le_bytes(v.to_be_bytes())
                })
                .collect();
            let mut ffi_out = vec![MaybeUninit::<f32>::uninit(); 4 * width]
                .into_boxed_slice();
            unsafe {
                darkroom_pnm_ppm_u16_row_to_float(
                    ffi_out.as_mut_ptr().cast(),
                    line.as_ptr(),
                    width,
                    max,
                );
            }
            // every lane initialized by the call: safe to assume_init now
            let ffi_out =
                unsafe { Box::<[f32]>::from_raw(Box::into_raw(ffi_out) as *mut [f32]) };
            let mut direct = vec![0.0f32; 4 * width];
            pnm_ppm_u16_row_to_float(&mut direct, &line, width, max);
            assert_eq!(ffi_out.len(), direct.len());
            for (k, (f, d)) in ffi_out.iter().zip(direct.iter()).enumerate() {
                assert_eq!(f.to_bits(), d.to_bits(), "lane {k}");
            }
        }
    }

    #[test]
    fn pnm_ppm_u16_row_ffi_guards() {
        let line = [0x3412u16; 48];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_pnm_ppm_u16_row_to_float(std::ptr::null_mut(), line.as_ptr(), 4, 65535);
            darkroom_pnm_ppm_u16_row_to_float(out.as_mut_ptr(), std::ptr::null(), 4, 65535);
            // zero width and zero maxval
            darkroom_pnm_ppm_u16_row_to_float(out.as_mut_ptr(), line.as_ptr(), 0, 65535);
            darkroom_pnm_ppm_u16_row_to_float(out.as_mut_ptr(), line.as_ptr(), 4, 0);
            // overflowing lane counts (either product wraps — rejected
            // before building any slice)
            darkroom_pnm_ppm_u16_row_to_float(out.as_mut_ptr(), line.as_ptr(), usize::MAX, 65535);
            darkroom_pnm_ppm_u16_row_to_float(
                out.as_mut_ptr(),
                line.as_ptr(),
                usize::MAX / 4 + 1,
                65535,
            );
            darkroom_pnm_ppm_u16_row_to_float(
                out.as_mut_ptr(),
                line.as_ptr(),
                usize::MAX / 3 + 1,
                65535,
            );
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }

    // rails pinned as exact bit patterns for the PBM bit-unpack row
    // (m4-239): a clear file bit decodes to exactly 1.0 (PBM 0 is white),
    // a set file bit to exactly +0.0 (PBM 1 is black — the INVERTED
    // polarity of the C `line[x] ^ 0xff`), MSB-first within each pack
    // byte, and every alpha lane exactly +0.0. The asymmetric 0x9C byte
    // pins the bit order itself (it is not a bit-palindrome, so LSB-first
    // would give an observably different vector),
    // and the width-5 case pins the tail guard (the 3 padding bits of the
    // last pack byte are never read). All rails are forced by the C
    // operation (exact 0.0/1.0 selection, no arithmetic), so they pin the
    // kernel directly.
    #[test]
    fn pnm_pbm_row_rails_pin() {
        // full bytes: 0x00 -> all white, 0xFF -> all black
        let mut out = vec![7.0f32; 8];
        pnm_pbm_row_to_float(&mut out, &[0x00], 8);
        assert_eq!(out[0].to_bits(), 0x3F80_0000);
        let mut black = vec![7.0f32; 8];
        pnm_pbm_row_to_float(&mut black, &[0xFF], 8);
        assert_eq!(black[0].to_bits(), 0x0000_0000);
        // asymmetric byte 0x9C = file bits 1,0,0,1,1,1,0,0 MSB-first, so
        // decoded values are 0,1,1,0,0,0,1,1. 0x9C is not a bit-palindrome:
        // an LSB-first misread would decode 1,1,0,0,0,1,1,0 instead, so the
        // first- and last-pixel pins below catch a reversed bit order.
        let mut asym = vec![7.0f32; 32];
        pnm_pbm_row_to_float(&mut asym, &[0x9C], 8);
        let expected = [
            0x0000_0000u32,
            0x3F80_0000,
            0x3F80_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x3F80_0000,
            0x3F80_0000,
        ];
        for x in 0..8 {
            assert_eq!(asym[4 * x].to_bits(), expected[x], "x={x}");
            assert_eq!(asym[4 * x + 1].to_bits(), expected[x], "x={x}");
            assert_eq!(asym[4 * x + 2].to_bits(), expected[x], "x={x}");
            assert_eq!(asym[4 * x + 3].to_bits(), 0x0000_0000, "x={x}");
        }
        // bit-order pins: an LSB-first misread of 0x9C would give 1.0 at
        // pixel 0 and 0.0 at pixel 7 — the reverse of the pins above.
        assert_eq!(asym[0].to_bits(), 0x0000_0000);
        assert_eq!(asym[28].to_bits(), 0x3F80_0000);
        // tail guard: width 5 reads only the top 5 bits of 0b10110_000
        // (file bits 1,0,1,1,0 -> values 0,1,0,0,1); the 3 padding bits
        // are ignored, so padding 000 and padding 111 decode identically
        for padding in [0x00u8, 0x07] {
            let mut tail = vec![7.0f32; 20];
            pnm_pbm_row_to_float(&mut tail, &[0xB0 | padding], 5);
            let values: Vec<u32> = (0..5).map(|x| tail[4 * x].to_bits()).collect();
            assert_eq!(
                values,
                vec![0x0000_0000, 0x3F80_0000, 0x0000_0000, 0x0000_0000, 0x3F80_0000],
                "padding={padding:#04x}"
            );
            for x in 0..5 {
                assert_eq!(tail[4 * x + 3].to_bits(), 0x0000_0000, "x={x}");
            }
        }
    }

    // LCG bit sweep at several widths (including non-multiples of 8, so
    // the tail guard is covered), sentinel-padded: kernel and reference
    // must agree bit-exactly on every lane, and only the exact 4*width
    // span may be written.
    #[test]
    fn pnm_pbm_row_matches_reference() {
        for width in [1usize, 3, 5, 7, 8, 9, 17, 65] {
            let packed = width.div_ceil(8);
            let mut line = Vec::with_capacity(packed);
            let mut state = 0x9E37_79B9u32;
            for _ in 0..packed {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                line.push((state >> 24) as u8);
            }
            // head/tail sentinel lanes around the exact 4*width span
            let mut direct = vec![-1.0f32; 4 * width + 8];
            let mut reference = vec![-2.0f32; 4 * width + 8];
            pnm_pbm_row_to_float(&mut direct[4..4 + 4 * width], &line, width);
            ref_pnm_pbm_row_to_float(&line, &mut reference[4..4 + 4 * width], width);
            assert_eq!(direct[..4], [-1.0; 4], "direct head");
            assert_eq!(direct[4 + 4 * width..], [-1.0; 4], "direct tail");
            assert_eq!(reference[..4], [-2.0; 4], "reference head");
            assert_eq!(reference[4 + 4 * width..], [-2.0; 4], "reference tail");
            for k in 0..4 * width {
                assert_eq!(
                    direct[4 + k].to_bits(),
                    reference[4 + k].to_bits(),
                    "lane {k}"
                );
            }
        }
    }

    #[test]
    fn pnm_pbm_row_degenerate_guards_no_op() {
        let line = vec![0xA5u8; 16];
        // zero width: output untouched
        let mut out = vec![9.0f32; 16];
        pnm_pbm_row_to_float(&mut out, &line, 0);
        assert_eq!(out, vec![9.0f32; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_out = vec![0.0f32; 7]; // short for 2 quads
        pnm_pbm_row_to_float(&mut short_out, &[0x00], 2);
        // clamped to a single quad: first pixel decodes clear-bit 1.0,
        // remaining lanes untouched
        assert_eq!(short_out[0].to_bits(), 0x3F80_0000);
        assert_eq!(short_out[1].to_bits(), 0x3F80_0000);
        assert_eq!(short_out[2].to_bits(), 0x3F80_0000);
        assert_eq!(short_out[3].to_bits(), 0x0000_0000);
        assert_eq!(short_out[4..], [0.0f32; 3]);
        let empty: Vec<u8> = vec![];
        let mut out2 = vec![-3.0f32; 8];
        pnm_pbm_row_to_float(&mut out2, &empty, 2);
        assert_eq!(out2, vec![-3.0f32; 8]); // no pack bytes: nothing written
        let mut out3 = vec![5.0f32; 4];
        pnm_pbm_row_to_float(&mut out3, &empty, 1);
        assert_eq!(out3, vec![5.0f32; 4]);
    }

    #[test]
    fn pnm_pbm_row_ffi_uninit_row() {
        use std::mem::MaybeUninit;

        // uninitialized row storage is realistic: the C row cursor points
        // into the mipmap-cache allocation, whose lanes the kernel fully
        // overwrites before any read.
        for width in [1usize, 5, 9, 65] {
            let packed = width.div_ceil(8);
            let mut line = Vec::with_capacity(packed);
            for i in 0..packed {
                line.push(((i as u64 * 2_654_435_761 + 0x9E37) % 256) as u8);
            }
            let mut ffi_out = vec![MaybeUninit::<f32>::uninit(); 4 * width]
                .into_boxed_slice();
            unsafe {
                darkroom_pnm_pbm_row_to_float(ffi_out.as_mut_ptr().cast(), line.as_ptr(), width);
            }
            // every lane initialized by the call: safe to assume_init now
            let ffi_out =
                unsafe { Box::<[f32]>::from_raw(Box::into_raw(ffi_out) as *mut [f32]) };
            let mut direct = vec![0.0f32; 4 * width];
            pnm_pbm_row_to_float(&mut direct, &line, width);
            assert_eq!(ffi_out.len(), direct.len());
            for (k, (f, d)) in ffi_out.iter().zip(direct.iter()).enumerate() {
                assert_eq!(f.to_bits(), d.to_bits(), "lane {k}");
            }
        }
    }

    #[test]
    fn pnm_pbm_row_ffi_guards() {
        let line = [0xA5u8; 16];
        let mut out = vec![7.0f32; 16];
        unsafe {
            // null pointers
            darkroom_pnm_pbm_row_to_float(std::ptr::null_mut(), line.as_ptr(), 4);
            darkroom_pnm_pbm_row_to_float(out.as_mut_ptr(), std::ptr::null(), 4);
            // zero width
            darkroom_pnm_pbm_row_to_float(out.as_mut_ptr(), line.as_ptr(), 0);
            // overflowing lane count (4 * width wraps — rejected before
            // building any slice) and overflowing pack span (width + 7
            // wraps — rejected likewise)
            darkroom_pnm_pbm_row_to_float(out.as_mut_ptr(), line.as_ptr(), usize::MAX);
            darkroom_pnm_pbm_row_to_float(
                out.as_mut_ptr(),
                line.as_ptr(),
                usize::MAX / 4 + 1,
            );
            darkroom_pnm_pbm_row_to_float(out.as_mut_ptr(), line.as_ptr(), usize::MAX - 3);
        }
        assert_eq!(out, vec![7.0f32; 16]); // untouched
    }

    // rails pinned as exact bytes for the JPEG RGBA row strip (m4-232):
    // each triplet lane copies its own quad lane, the alpha lane is
    // dropped, and lanes stay distinct (a lane-rotation bug would move
    // them). Pure byte shuffle, so `assert_eq` on bytes pins the kernel
    // to the C operation directly.
    #[test]
    fn jpeg_rgba_row_rails_pin() {
        let buf = [
            1u8, 2, 3, 77, // alpha 77 must not leak anywhere
            0, 128, 255, 0, // black-to-white spread across lanes
            10, 20, 30, 40, // asymmetric: pins per-lane copy, not rotation
        ];
        let mut row = vec![9u8; 9];
        jpeg_rgba_row_to_rgb24(&mut row, &buf, 3);
        assert_eq!(row, vec![1u8, 2, 3, 0, 128, 255, 10, 20, 30]);
        // lanes stay distinct: a rotation bug would collapse or shift them
        assert_ne!(row[0], row[1]);
        assert_ne!(row[1], row[2]);
        assert_ne!(row[6], row[7]);
        assert_ne!(row[7], row[8]);
        // alpha-drop pin: same RGB with different alphas gives the same row
        let buf_alt = [
            1u8, 2, 3, 0, // alpha flipped 77 -> 0
            0, 128, 255, 200, // alpha flipped 0 -> 200
            10, 20, 30, 255, // alpha flipped 40 -> 255
        ];
        let mut row_alt = vec![9u8; 9];
        jpeg_rgba_row_to_rgb24(&mut row_alt, &buf_alt, 3);
        assert_eq!(row_alt, row);
        // single column: only the first three lanes land
        let mut single = vec![9u8; 3];
        jpeg_rgba_row_to_rgb24(&mut single, &[250u8, 251, 252, 253], 1);
        assert_eq!(single, vec![250u8, 251, 252]);
    }

    // LCG byte sweep at several widths, sentinel-padded: kernel and
    // reference must agree exactly on every lane, and only the exact
    // 3*width span may be written. The LCG keeps the three lanes of each
    // quad generally distinct (interleave coverage, not just grays).
    #[test]
    fn jpeg_rgba_row_matches_reference() {
        for width in [1usize, 3, 17, 65] {
            let mut buf = Vec::with_capacity(4 * width);
            for i in 0..4 * width {
                let v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37) % 256;
                buf.push(v as u8);
            }
            buf[0] = 0;
            buf[1] = 0;
            buf[2] = 0;
            // head/tail sentinel lanes around the exact 3*width span
            let mut direct = vec![0xA5u8; 3 * width + 8];
            let mut reference = vec![0x5Au8; 3 * width + 8];
            jpeg_rgba_row_to_rgb24(&mut direct[4..4 + 3 * width], &buf, width);
            ref_jpeg_rgba_row_to_rgb24(
                &buf,
                &mut reference[4..4 + 3 * width],
                width,
            );
            assert_eq!(direct[..4], [0xA5u8; 4], "direct head");
            assert_eq!(direct[4 + 3 * width..], [0xA5u8; 4], "direct tail");
            assert_eq!(reference[..4], [0x5Au8; 4], "reference head");
            assert_eq!(reference[4 + 3 * width..], [0x5Au8; 4], "reference tail");
            assert_eq!(direct[4..4 + 3 * width], reference[4..4 + 3 * width]);
        }
    }

    #[test]
    fn jpeg_rgba_row_degenerate_guards_no_op() {
        let buf = vec![200u8; 64];
        // zero width: output untouched
        let mut row = vec![9u8; 12];
        jpeg_rgba_row_to_rgb24(&mut row, &buf, 0);
        assert_eq!(row, vec![9u8; 12]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_row = vec![0u8; 5]; // short for 2 triplets
        jpeg_rgba_row_to_rgb24(&mut short_row, &buf, 2);
        // clamped to a single column: first triplet copied per lane,
        // remaining lanes untouched
        assert_eq!(short_row[..3], [200u8; 3]);
        assert_eq!(short_row[3..], [0u8; 2]);
        let short_buf = [200u8; 5]; // one quad plus a stray byte: short for 2 columns
        let mut row2 = vec![7u8; 6];
        jpeg_rgba_row_to_rgb24(&mut row2, &short_buf, 2);
        assert_eq!(row2[..3], [200u8; 3]);
        assert_eq!(row2[3..], vec![7u8; 3]); // second triplet untouched
        let empty: Vec<u8> = vec![];
        let mut row3 = vec![5u8; 3];
        jpeg_rgba_row_to_rgb24(&mut row3, &empty, 1);
        assert_eq!(row3, vec![5u8; 3]);
        let mut row4 = vec![5u8; 3];
        jpeg_rgba_row_to_rgb24(&mut row4, &buf, 1);
        // empty row with a live source: nothing to write, no panic
        let mut empty_row: Vec<u8> = vec![];
        jpeg_rgba_row_to_rgb24(&mut empty_row, &buf, 1);
        assert!(empty_row.is_empty());
        assert_eq!(row4, vec![200u8; 3]);
    }

    #[test]
    fn jpeg_rgba_row_ffi_uninit_row() {
        use std::mem::MaybeUninit;

        // uninitialized row storage is realistic: the C strip is a fresh
        // `dt_alloc_align_uint8(3 * width)` allocation the kernel fully
        // overwrites before any read.
        for width in [1usize, 3, 17, 65] {
            let mut buf = Vec::with_capacity(4 * width);
            for i in 0..4 * width {
                buf.push(((i as u64 * 2_654_435_761 + 0x9E37) % 256) as u8);
            }
            let mut ffi_row = vec![MaybeUninit::<u8>::uninit(); 3 * width]
                .into_boxed_slice();
            unsafe {
                darkroom_jpeg_rgba_row_to_rgb24(
                    ffi_row.as_mut_ptr().cast(),
                    buf.as_ptr(),
                    width,
                );
            }
            // every lane initialized by the call: safe to assume_init now
            let ffi_row =
                unsafe { Box::<[u8]>::from_raw(Box::into_raw(ffi_row) as *mut [u8]) };
            let mut direct = vec![0u8; 3 * width];
            jpeg_rgba_row_to_rgb24(&mut direct, &buf, width);
            assert_eq!(ffi_row.len(), direct.len());
            assert_eq!(&*ffi_row, &direct[..]);
        }
    }

    #[test]
    fn jpeg_rgba_row_ffi_guards() {
        let buf = [200u8; 64];
        let mut row = vec![7u8; 12];
        unsafe {
            // null pointers
            darkroom_jpeg_rgba_row_to_rgb24(std::ptr::null_mut(), buf.as_ptr(), 4);
            darkroom_jpeg_rgba_row_to_rgb24(row.as_mut_ptr(), std::ptr::null(), 4);
            // zero width
            darkroom_jpeg_rgba_row_to_rgb24(row.as_mut_ptr(), buf.as_ptr(), 0);
            // overflowing lane counts (either product wraps — rejected
            // before building any slice)
            darkroom_jpeg_rgba_row_to_rgb24(row.as_mut_ptr(), buf.as_ptr(), usize::MAX);
            darkroom_jpeg_rgba_row_to_rgb24(
                row.as_mut_ptr(),
                buf.as_ptr(),
                usize::MAX / 3 + 1,
            );
            darkroom_jpeg_rgba_row_to_rgb24(
                row.as_mut_ptr(),
                buf.as_ptr(),
                usize::MAX / 4 + 1,
            );
        }
        assert_eq!(row, vec![7u8; 12]); // untouched
    }

    // rails pinned as exact bytes for the JPEG RGB24 row expand (m4-237):
    // each quad lane copies its own triplet lane, the alpha lane keeps
    // whatever byte was already there (the C loop never stores lane 3),
    // and lanes stay distinct (a lane-rotation bug would move them).
    // Pure byte shuffle, so `assert_eq` on bytes pins the kernel to the
    // C operation directly.
    #[test]
    fn jpeg_rgb24_row_rails_pin() {
        let row = [
            1u8, 2, 3, // column 0
            0, 128, 255, // black-to-white spread across lanes
            10, 20, 30, // asymmetric: pins per-lane copy, not rotation
        ];
        // alpha slots pre-filled with sentinels: the kernel must not write them
        let mut tmp = vec![
            9u8, 9, 9, 0xA5, // column 0 keeps 0xA5
            9u8, 9, 9, 0x5A, // column 1 keeps 0x5A
            9u8, 9, 9, 0x00, // column 2 keeps 0x00
        ];
        jpeg_rgb24_row_to_rgba(&mut tmp, &row, 3);
        assert_eq!(
            tmp,
            vec![1u8, 2, 3, 0xA5, 0, 128, 255, 0x5A, 10, 20, 30, 0x00]
        );
        // lanes stay distinct: a rotation bug would collapse or shift them
        assert_ne!(tmp[0], tmp[1]);
        assert_ne!(tmp[1], tmp[2]);
        assert_ne!(tmp[4], tmp[5]);
        assert_ne!(tmp[5], tmp[6]);
        // same RGB with a different source has no alpha to leak: output
        // lanes 0..2 follow the source exactly
        let mut single = [9u8, 9, 9, 0xC3];
        jpeg_rgb24_row_to_rgba(&mut single, &[250u8, 251, 252], 1);
        assert_eq!(single, [250u8, 251, 252, 0xC3]);
    }

    // LCG byte sweep at several widths, sentinel-padded: kernel and
    // reference must agree exactly on lanes 0..2, alpha slots must keep
    // their sentinel values on both paths, and only the exact 4*width
    // span may be written. The LCG keeps the three lanes of each
    // triplet generally distinct (interleave coverage, not just grays).
    #[test]
    fn jpeg_rgb24_row_matches_reference() {
        for width in [1usize, 3, 17, 65] {
            let mut row = Vec::with_capacity(3 * width);
            for i in 0..3 * width {
                let v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37) % 256;
                row.push(v as u8);
            }
            row[0] = 0;
            row[1] = 0;
            row[2] = 0;
            // alpha slots pre-filled with a distinct sentinel pattern
            let mut direct = vec![0xA5u8; 4 * width + 8];
            let mut reference = vec![0x5Au8; 4 * width + 8];
            for k in 0..width {
                direct[4 + 4 * k + 3] = 0x3C;
                reference[4 + 4 * k + 3] = 0xC3;
            }
            jpeg_rgb24_row_to_rgba(&mut direct[4..4 + 4 * width], &row, width);
            ref_jpeg_rgb24_row_to_rgba(
                &row,
                &mut reference[4..4 + 4 * width],
                width,
            );
            assert_eq!(direct[..4], [0xA5u8; 4], "direct head");
            assert_eq!(direct[4 + 4 * width..], [0xA5u8; 4], "direct tail");
            assert_eq!(reference[..4], [0x5Au8; 4], "reference head");
            assert_eq!(reference[4 + 4 * width..], [0x5Au8; 4], "reference tail");
            for k in 0..width {
                assert_eq!(&direct[4 + 4 * k..4 + 4 * k + 3], &reference[4 + 4 * k..4 + 4 * k + 3]);
                assert_eq!(direct[4 + 4 * k + 3], 0x3C, "direct alpha untouched {k}");
                assert_eq!(reference[4 + 4 * k + 3], 0xC3, "reference alpha untouched {k}");
            }
        }
    }

    #[test]
    fn jpeg_rgb24_row_degenerate_guards_no_op() {
        let row = vec![200u8; 64];
        // zero width: output untouched
        let mut tmp = vec![9u8; 16];
        jpeg_rgb24_row_to_rgba(&mut tmp, &row, 0);
        assert_eq!(tmp, vec![9u8; 16]);
        // truncated buffers: clamped iteration must neither panic nor write
        // out of bounds; well-formed callers never hit this path.
        let mut short_tmp = vec![0xAAu8; 7]; // short for 2 quads
        jpeg_rgb24_row_to_rgba(&mut short_tmp, &row, 2);
        // clamped to a single column: first triplet copied per lane,
        // its alpha slot and remaining bytes untouched
        assert_eq!(short_tmp[..3], [200u8; 3]);
        assert_eq!(short_tmp[3..], [0xAAu8; 4]);
        let short_row = [200u8; 5]; // one triplet plus stray bytes: short for 2 columns
        let mut tmp2 = vec![7u8; 8];
        jpeg_rgb24_row_to_rgba(&mut tmp2, &short_row, 2);
        assert_eq!(tmp2[..3], [200u8; 3]);
        assert_eq!(tmp2[3..], vec![7u8; 5]); // second quad untouched
        let empty: Vec<u8> = vec![];
        let mut tmp3 = vec![5u8; 4];
        jpeg_rgb24_row_to_rgba(&mut tmp3, &empty, 1);
        assert_eq!(tmp3, vec![5u8; 4]);
        let mut tmp4 = vec![5u8; 4];
        jpeg_rgb24_row_to_rgba(&mut tmp4, &row, 1);
        // empty tmp with a live source: nothing to write, no panic
        let mut empty_tmp: Vec<u8> = vec![];
        jpeg_rgb24_row_to_rgba(&mut empty_tmp, &row, 1);
        assert!(empty_tmp.is_empty());
        assert_eq!(tmp4[..3], vec![200u8; 3]);
        assert_eq!(tmp4[3], 5u8); // alpha slot untouched
    }

    #[test]
    fn jpeg_rgb24_row_ffi_uninit_row() {
        use std::mem::MaybeUninit;

        // uninitialized tmp storage is realistic: the C destination is a
        // cursor into the fresh `dt_alloc_align_uint8(4 * width * height)`
        // output buffer, whose alpha slots the kernel never writes. Lanes
        // 0..2 are fully overwritten before any read; lane 3 keeps a
        // pre-written sentinel, so only lanes 0..2 go through assume_init.
        for width in [1usize, 3, 17, 65] {
            let mut row = Vec::with_capacity(3 * width);
            for i in 0..3 * width {
                row.push(((i as u64 * 2_654_435_761 + 0x9E37) % 256) as u8);
            }
            let mut ffi_tmp = vec![MaybeUninit::<u8>::uninit(); 4 * width]
                .into_boxed_slice();
            for k in 0..width {
                ffi_tmp[4 * k + 3].write(0x3C);
            }
            unsafe {
                darkroom_jpeg_rgb24_row_to_rgba(
                    ffi_tmp.as_mut_ptr().cast(),
                    row.as_ptr(),
                    width,
                );
            }
            for k in 0..width {
                for lane in 0..3 {
                    assert_eq!(unsafe { ffi_tmp[4 * k + lane].assume_init() }, row[3 * k + lane]);
                }
                assert_eq!(unsafe { ffi_tmp[4 * k + 3].assume_init() }, 0x3C);
            }
            let mut direct = vec![0x3Cu8; 4 * width];
            for k in 0..width {
                direct[4 * k + 3] = 0x3C;
            }
            jpeg_rgb24_row_to_rgba(&mut direct, &row, width);
            for k in 0..width {
                for lane in 0..3 {
                    assert_eq!(unsafe { ffi_tmp[4 * k + lane].assume_init() }, direct[4 * k + lane]);
                }
            }
        }
    }

    #[test]
    fn jpeg_rgb24_row_ffi_guards() {
        let row = [200u8; 64];
        let mut tmp = vec![7u8; 16];
        unsafe {
            // null pointers
            darkroom_jpeg_rgb24_row_to_rgba(std::ptr::null_mut(), row.as_ptr(), 4);
            darkroom_jpeg_rgb24_row_to_rgba(tmp.as_mut_ptr(), std::ptr::null(), 4);
            // zero width
            darkroom_jpeg_rgb24_row_to_rgba(tmp.as_mut_ptr(), row.as_ptr(), 0);
            // overflowing lane counts (either product wraps — rejected
            // before building any slice)
            darkroom_jpeg_rgb24_row_to_rgba(tmp.as_mut_ptr(), row.as_ptr(), usize::MAX);
            darkroom_jpeg_rgb24_row_to_rgba(
                tmp.as_mut_ptr(),
                row.as_ptr(),
                usize::MAX / 4 + 1,
            );
            darkroom_jpeg_rgb24_row_to_rgba(
                tmp.as_mut_ptr(),
                row.as_ptr(),
                usize::MAX / 3 + 1,
            );
        }
        assert_eq!(tmp, vec![7u8; 16]); // untouched
    }

    // test helpers: pack float quads to the aliased byte view the kernel
    // works on, and read back u16 lanes the way the C buf16 view would.
    fn pack_float_pixels(pixels: &[f32]) -> Vec<u8> {
        pixels.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn read_u16_lane(buf: &[u8], lane: usize) -> u16 {
        u16::from_ne_bytes([buf[2 * lane], buf[2 * lane + 1]])
    }

    // rails pinned as exact ints for the 16-bit export downconvert
    // (m4-234): 0.0 maps to 0, 1.0 maps to 65535, out-of-range inputs
    // clamp to the rails, and lane 3 keeps its bytes (the C loop never
    // touches it).
    #[test]
    fn float_to_u16_rails_pin() {
        let alpha = 0.5f32;
        let pixels = [
            0.0f32, 0.0, 0.0, alpha, // black pixel
            1.0, 1.0, 1.0, alpha, // white pixel
            -1.0, 2.0, -0.0, alpha, // clamp rails plus negative zero
        ];
        let mut buf = pack_float_pixels(&pixels);
        let original = buf.clone();
        float_to_u16_inplace(&mut buf, 3);
        assert_eq!((read_u16_lane(&buf, 0), read_u16_lane(&buf, 1), read_u16_lane(&buf, 2)), (0, 0, 0));
        assert_eq!((read_u16_lane(&buf, 4), read_u16_lane(&buf, 5), read_u16_lane(&buf, 6)), (65535, 65535, 65535));
        assert_eq!(
            (read_u16_lane(&buf, 8), read_u16_lane(&buf, 9), read_u16_lane(&buf, 10)),
            (0, 65535, 0)
        );
        // lane 3 of every pixel is never written: its bytes keep their
        // original values (the C loop only stores lanes below 3)
        for k in 0..3 {
            assert_eq!(
                &buf[8 * k + 6..8 * k + 8],
                &original[8 * k + 6..8 * k + 8],
                "alpha lane {k}"
            );
        }
        // everything past the 8*npixels u16 span is untouched input
        assert_eq!(&buf[24..], &pack_float_pixels(&pixels)[24..]);
    }

    // rounding pins for the 16-bit export downconvert (m4-234): the exact
    // halves below are verified bit-exact through the f32 divide/multiply
    // (IEEE-754, hence deterministic), and the odd-half cases pin
    // round-half-away-from-zero against banker's rounding, matching C
    // roundf.
    #[test]
    fn float_to_u16_rounding_half_away() {
        // (input float, expected u16): scaled products are exactly 32767.5,
        // 2.5, 1.5, 0.5, and 1.0 respectively.
        for (value, expected) in [
            (0.5f32, 32768u16),
            (2.5f32 / 65535.0, 3),
            (1.5f32 / 65535.0, 2),
            (0.5f32 / 65535.0, 1),
            (1.0f32 / 65535.0, 1),
        ] {
            let mut buf = pack_float_pixels(&[value, value, value, 0.25]);
            float_to_u16_inplace(&mut buf, 1);
            assert_eq!(
                (read_u16_lane(&buf, 0), read_u16_lane(&buf, 1), read_u16_lane(&buf, 2)),
                (expected, expected, expected),
                "value {value}"
            );
            // cross-check the divergent reference on the same value
            let mut dst = [0x77u16; 4];
            ref_float_to_u16(&[value, value, value, 0.25], &mut dst, 1);
            assert_eq!((dst[0], dst[1], dst[2]), (expected, expected, expected));
        }
        // negative half-away: -0.5 scales to exactly -32767.5, which rounds
        // away from zero to -32768 and clamps to the zero rail.
        let mut buf = pack_float_pixels(&[-0.5f32, -0.5, -0.5, 0.25]);
        float_to_u16_inplace(&mut buf, 1);
        assert_eq!(
            (read_u16_lane(&buf, 0), read_u16_lane(&buf, 1), read_u16_lane(&buf, 2)),
            (0, 0, 0)
        );
    }

    // LCG sweep over raw float bit patterns (covers negatives, denormals,
    // infinities, and NaNs) at several image sizes: kernel and reference
    // must agree exactly on lanes 0..2 of every pixel, lane 3 must keep
    // its bytes, and all bytes past the 8*npixels u16 span must be
    // untouched input.
    #[test]
    fn float_to_u16_matches_reference_lcg_sweep() {
        for npixels in [1usize, 2, 3, 17, 65] {
            let mut state = 0x9E37_79B9u32;
            let pixels: Vec<f32> = (0..4 * npixels)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    f32::from_bits(state)
                })
                .collect();
            let mut buf = pack_float_pixels(&pixels);
            let original = buf.clone();
            let mut expected = vec![0x77u16; 4 * npixels];
            ref_float_to_u16(&pixels, &mut expected, npixels);
            float_to_u16_inplace(&mut buf, npixels);
            for k in 0..npixels {
                for i in 0..3 {
                    assert_eq!(
                        read_u16_lane(&buf, 4 * k + i),
                        expected[4 * k + i],
                        "pixel {k} lane {i}"
                    );
                }
                assert_eq!(
                    &buf[8 * k + 6..8 * k + 8],
                    &original[8 * k + 6..8 * k + 8],
                    "alpha lane {k}"
                );
            }
            assert_eq!(&buf[8 * npixels..], &original[8 * npixels..], "tail");
        }
    }

    #[test]
    fn float_to_u16_ffi_uninit_window() {
        use std::mem::MaybeUninit;

        // uninitialized export storage is realistic: pipe.backbuf is a
        // fresh pipeline allocation whose float lanes the kernel reads
        // only after the pixel pipeline wrote them, and whose u16 lanes
        // it fully overwrites for lanes 0..2.
        for npixels in [1usize, 3, 17] {
            let pixels: Vec<f32> = (0..4 * npixels)
                .map(|i| ((i as u64 * 2_654_435_761 + 0x9E37) % 65536) as f32 / 65535.0 - 0.25)
                .collect();
            let bytes = pack_float_pixels(&pixels);
            // head/tail sentinel bytes around the exact 16*npixels span,
            // window offset by 4 so a single-byte over/under-run fails.
            let mut storage = vec![MaybeUninit::<u8>::uninit(); 16 * npixels + 8];
            storage[..4].iter_mut().for_each(|b| {
                b.write(0xA5);
            });
            storage[4 + 16 * npixels..].iter_mut().for_each(|b| {
                b.write(0x5A);
            });
            for (slot, byte) in storage[4..4 + 16 * npixels].iter_mut().zip(bytes.iter()) {
                slot.write(*byte);
            }
            unsafe {
                darkroom_imageio_float_to_u16(storage.as_mut_ptr().add(4).cast(), npixels);
            }
            let storage_init: Vec<u8> =
                unsafe { storage.iter().map(|b| b.assume_init()).collect() };
            assert_eq!(&storage_init[..4], &[0xA5u8; 4], "head");
            assert_eq!(&storage_init[4 + 16 * npixels..], &[0x5Au8; 4], "tail");
            let mut direct = bytes.clone();
            float_to_u16_inplace(&mut direct, npixels);
            assert_eq!(&storage_init[4..4 + 16 * npixels], &direct[..]);
        }
    }

    #[test]
    fn float_to_u16_degenerate_guards_no_op() {
        // zero pixels: buffer untouched
        let mut buf = pack_float_pixels(&[0.5f32, 0.5, 0.5, 0.5]);
        float_to_u16_inplace(&mut buf, 0);
        assert_eq!(buf, pack_float_pixels(&[0.5f32, 0.5, 0.5, 0.5]));
        // truncated buffer: clamped iteration converts only the complete
        // pixel, never panics nor writes out of bounds; well-formed
        // callers never hit this path.
        let full = pack_float_pixels(&[1.0f32, 0.0, 0.5, 0.25, 1.0, 0.0, 0.5, 0.25]);
        let mut short = full[..20].to_vec(); // one pixel plus 4 stray bytes
        float_to_u16_inplace(&mut short, 2);
        assert_eq!(
            (read_u16_lane(&short, 0), read_u16_lane(&short, 1), read_u16_lane(&short, 2)),
            (65535, 0, 32768)
        );
        assert_eq!(&short[8..], &full[8..20]);
        // empty buffer with nonzero count: no-op
        let mut empty: Vec<u8> = vec![];
        float_to_u16_inplace(&mut empty, 1);
        assert!(empty.is_empty());
        // FFI guards: null, zero count, and overflowing 16*npixels
        // products are no-ops that never touch memory.
        let mut guarded = vec![7u8; 32];
        unsafe {
            darkroom_imageio_float_to_u16(std::ptr::null_mut(), 2);
            darkroom_imageio_float_to_u16(guarded.as_mut_ptr(), 0);
            darkroom_imageio_float_to_u16(guarded.as_mut_ptr(), usize::MAX);
            darkroom_imageio_float_to_u16(guarded.as_mut_ptr(), usize::MAX / 16 + 1);
        }
        assert_eq!(guarded, vec![7u8; 32]); // untouched
    }

    // rails pinned as exact ints for the 8-bit export downconvert
    // (m4-235): 0.0 maps to 0, 1.0 maps to 255, out-of-range inputs
    // clamp to the rails, and lane 3 keeps its byte (the C loop never
    // touches it).
    #[test]
    fn float_to_u8_rails_pin() {
        let alpha = 0.5f32;
        let pixels = [
            0.0f32, 0.0, 0.0, alpha, // black pixel
            1.0, 1.0, 1.0, alpha, // white pixel
            -1.0, 2.0, -0.0, alpha, // clamp rails plus negative zero
        ];
        let mut buf = pack_float_pixels(&pixels);
        let original = buf.clone();
        float_to_u8_inplace(&mut buf, 3);
        assert_eq!((buf[0], buf[1], buf[2]), (0, 0, 0));
        assert_eq!((buf[4], buf[5], buf[6]), (255, 255, 255));
        assert_eq!((buf[8], buf[9], buf[10]), (0, 255, 0));
        // lane 3 of every pixel is never written: its byte keeps its
        // original value (the C loop only stores lanes 0..2)
        for k in 0..3 {
            assert_eq!(&buf[4 * k + 3..4 * k + 4], &original[4 * k + 3..4 * k + 4], "alpha lane {k}");
        }
        // everything past the 4*npixels u8 span is untouched input
        assert_eq!(&buf[12..], &pack_float_pixels(&pixels)[12..]);
    }

    // rounding pins for the 8-bit export downconvert (m4-235): the exact
    // halves below are verified bit-exact through the f32 divide/multiply
    // (IEEE-754, hence deterministic), and the odd-half cases pin
    // round-half-away-from-zero against banker's rounding, matching C
    // roundf.
    #[test]
    fn float_to_u8_rounding_half_away() {
        // (input float, expected u8): scaled products are exactly 127.5,
        // 2.5, 1.5, 0.5, and 1.0 respectively.
        for (value, expected) in [
            (0.5f32, 128u8),
            (2.5f32 / 255.0, 3),
            (1.5f32 / 255.0, 2),
            (0.5f32 / 255.0, 1),
            (1.0f32 / 255.0, 1),
        ] {
            let mut buf = pack_float_pixels(&[value, value, value, 0.25]);
            float_to_u8_inplace(&mut buf, 1);
            assert_eq!((buf[0], buf[1], buf[2]), (expected, expected, expected), "value {value}");
            // cross-check the divergent reference on the same value
            let mut dst = [0x77u8; 4];
            ref_float_to_u8(&[value, value, value, 0.25], &mut dst, 1);
            assert_eq!((dst[0], dst[1], dst[2]), (expected, expected, expected));
        }
        // negative half-away: -0.5 scales to exactly -127.5, which rounds
        // away from zero to -128 and clamps to the zero rail.
        let mut buf = pack_float_pixels(&[-0.5f32, -0.5, -0.5, 0.25]);
        float_to_u8_inplace(&mut buf, 1);
        assert_eq!((buf[0], buf[1], buf[2]), (0, 0, 0));
    }

    // LCG sweep over raw float bit patterns (covers negatives, denormals,
    // infinities, and NaNs) at several image sizes: kernel and reference
    // must agree exactly on lanes 0..2 of every pixel, lane 3 must keep
    // its byte, and all bytes past the 4*npixels u8 span must be
    // untouched input.
    #[test]
    fn float_to_u8_matches_reference_lcg_sweep() {
        for npixels in [1usize, 2, 3, 17, 65] {
            let mut state = 0x9E37_79B9u32;
            let pixels: Vec<f32> = (0..4 * npixels)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    f32::from_bits(state)
                })
                .collect();
            let mut buf = pack_float_pixels(&pixels);
            let original = buf.clone();
            let mut expected = vec![0x77u8; 4 * npixels];
            ref_float_to_u8(&pixels, &mut expected, npixels);
            float_to_u8_inplace(&mut buf, npixels);
            for k in 0..npixels {
                for i in 0..3 {
                    assert_eq!(buf[4 * k + i], expected[4 * k + i], "pixel {k} lane {i}");
                }
                assert_eq!(
                    &buf[4 * k + 3..4 * k + 4],
                    &original[4 * k + 3..4 * k + 4],
                    "alpha lane {k}"
                );
            }
            assert_eq!(&buf[4 * npixels..], &original[4 * npixels..], "tail");
        }
    }

    #[test]
    fn float_to_u8_ffi_uninit_window() {
        use std::mem::MaybeUninit;

        // uninitialized export storage is realistic: pipe.backbuf is a
        // fresh pipeline allocation whose float lanes the kernel reads
        // only after the pixel pipeline wrote them, and whose u8 lanes
        // it fully overwrites for lanes 0..2.
        for npixels in [1usize, 3, 17] {
            let pixels: Vec<f32> = (0..4 * npixels)
                .map(|i| ((i as u64 * 2_654_435_761 + 0x9E37) % 256) as f32 / 255.0 - 0.25)
                .collect();
            let bytes = pack_float_pixels(&pixels);
            // head/tail sentinel bytes around the exact 16*npixels span,
            // window offset by 4 so a single-byte over/under-run fails.
            let mut storage = vec![MaybeUninit::<u8>::uninit(); 16 * npixels + 8];
            storage[..4].iter_mut().for_each(|b| {
                b.write(0xA5);
            });
            storage[4 + 16 * npixels..].iter_mut().for_each(|b| {
                b.write(0x5A);
            });
            for (slot, byte) in storage[4..4 + 16 * npixels].iter_mut().zip(bytes.iter()) {
                slot.write(*byte);
            }
            unsafe {
                darkroom_imageio_float_to_u8(storage.as_mut_ptr().add(4).cast(), npixels);
            }
            let storage_init: Vec<u8> =
                unsafe { storage.iter().map(|b| b.assume_init()).collect() };
            assert_eq!(&storage_init[..4], &[0xA5u8; 4], "head");
            assert_eq!(&storage_init[4 + 16 * npixels..], &[0x5Au8; 4], "tail");
            let mut direct = bytes.clone();
            float_to_u8_inplace(&mut direct, npixels);
            assert_eq!(&storage_init[4..4 + 16 * npixels], &direct[..]);
        }
    }

    #[test]
    fn float_to_u8_degenerate_guards_no_op() {
        // zero pixels: buffer untouched
        let mut buf = pack_float_pixels(&[0.5f32, 0.5, 0.5, 0.5]);
        float_to_u8_inplace(&mut buf, 0);
        assert_eq!(buf, pack_float_pixels(&[0.5f32, 0.5, 0.5, 0.5]));
        // truncated buffer: clamped iteration converts only the complete
        // pixel, never panics nor writes out of bounds; well-formed
        // callers never hit this path.
        let full = pack_float_pixels(&[1.0f32, 0.0, 0.5, 0.25, 1.0, 0.0, 0.5, 0.25]);
        let mut short = full[..20].to_vec(); // one pixel plus 4 stray bytes
        float_to_u8_inplace(&mut short, 2);
        assert_eq!((short[0], short[1], short[2]), (255, 0, 128));
        assert_eq!(&short[4..], &full[4..20]);
        // empty buffer with nonzero count: no-op
        let mut empty: Vec<u8> = vec![];
        float_to_u8_inplace(&mut empty, 1);
        assert!(empty.is_empty());
        // FFI guards: null, zero count, and overflowing 16*npixels
        // products are no-ops that never touch memory.
        let mut guarded = vec![7u8; 32];
        unsafe {
            darkroom_imageio_float_to_u8(std::ptr::null_mut(), 2);
            darkroom_imageio_float_to_u8(guarded.as_mut_ptr(), 0);
            darkroom_imageio_float_to_u8(guarded.as_mut_ptr(), usize::MAX);
            darkroom_imageio_float_to_u8(guarded.as_mut_ptr(), usize::MAX / 16 + 1);
        }
        assert_eq!(guarded, vec![7u8; 32]); // untouched
    }

    // rails pinned as exact ints for the R/B-swapped 8-bit export
    // downconvert (m4-236): read lane order is (2, 1, 0) into write slots
    // (0, 1, 2), so asymmetric R != B inputs prove the lanes cross — an
    // unswapped implementation would produce the mirror image.
    #[test]
    fn float_to_u8_swap_rb_rails_pin() {
        let alpha = 0.5f32;
        let pixels = [
            0.0f32, 0.5, 1.0, alpha, // R=0 G=0.5 B=1 crosses to (255, 128, 0)
            1.0, 0.25, 0.0, alpha, // R=1 G=0.25 B=0 crosses to (0, 64, 255)
            -1.0, 2.0, -0.0, alpha, // clamp rails plus negative zero
        ];
        let mut buf = pack_float_pixels(&pixels);
        let original = buf.clone();
        float_to_u8_swap_rb_inplace(&mut buf, 3);
        assert_eq!((buf[0], buf[1], buf[2]), (255, 128, 0));
        assert_ne!((buf[0], buf[1], buf[2]), (0, 128, 255), "lanes must cross");
        assert_eq!((buf[4], buf[5], buf[6]), (0, 64, 255));
        assert_ne!((buf[4], buf[5], buf[6]), (255, 64, 0), "lanes must cross");
        assert_eq!((buf[8], buf[9], buf[10]), (0, 255, 0));
        // lane 3 of every pixel is never written: its byte keeps its
        // original value (the C loop only stores lanes 0..2)
        for k in 0..3 {
            assert_eq!(&buf[4 * k + 3..4 * k + 4], &original[4 * k + 3..4 * k + 4], "alpha lane {k}");
        }
        // everything past the 4*npixels u8 span is untouched input
        assert_eq!(&buf[12..], &pack_float_pixels(&pixels)[12..]);
    }

    // rounding pins for the swapped 8-bit export downconvert (m4-236): the
    // exact halves below are verified bit-exact through the f32
    // divide/multiply (IEEE-754, hence deterministic), and the odd-half
    // cases pin round-half-away-from-zero against banker's rounding,
    // matching C roundf. Symmetric values (swap-invisible) isolate the
    // formula from the lane mapping pinned above.
    #[test]
    fn float_to_u8_swap_rb_rounding_half_away() {
        // (input float, expected u8): scaled products are exactly 127.5,
        // 2.5, 1.5, 0.5, and 1.0 respectively.
        for (value, expected) in [
            (0.5f32, 128u8),
            (2.5f32 / 255.0, 3),
            (1.5f32 / 255.0, 2),
            (0.5f32 / 255.0, 1),
            (1.0f32 / 255.0, 1),
        ] {
            let mut buf = pack_float_pixels(&[value, value, value, 0.25]);
            float_to_u8_swap_rb_inplace(&mut buf, 1);
            assert_eq!((buf[0], buf[1], buf[2]), (expected, expected, expected), "value {value}");
            // cross-check the divergent reference on the same value
            let mut dst = [0x77u8; 4];
            ref_float_to_u8_swap_rb(&[value, value, value, 0.25], &mut dst, 1);
            assert_eq!((dst[0], dst[1], dst[2]), (expected, expected, expected));
        }
        // negative half-away: -0.5 scales to exactly -127.5, which rounds
        // away from zero to -128 and clamps to the zero rail.
        let mut buf = pack_float_pixels(&[-0.5f32, -0.5, -0.5, 0.25]);
        float_to_u8_swap_rb_inplace(&mut buf, 1);
        assert_eq!((buf[0], buf[1], buf[2]), (0, 0, 0));
    }

    // LCG sweep over raw float bit patterns (covers negatives, denormals,
    // infinities, and NaNs) at several image sizes: kernel and reference
    // must agree exactly on lanes 0..2 of every pixel, lane 3 must keep
    // its byte, and all bytes past the 4*npixels u8 span must be
    // untouched input.
    #[test]
    fn float_to_u8_swap_rb_matches_reference_lcg_sweep() {
        for npixels in [1usize, 2, 3, 17, 65] {
            let mut state = 0x9E37_79B9u32;
            let pixels: Vec<f32> = (0..4 * npixels)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    f32::from_bits(state)
                })
                .collect();
            let mut buf = pack_float_pixels(&pixels);
            let original = buf.clone();
            let mut expected = vec![0x77u8; 4 * npixels];
            ref_float_to_u8_swap_rb(&pixels, &mut expected, npixels);
            float_to_u8_swap_rb_inplace(&mut buf, npixels);
            for k in 0..npixels {
                for i in 0..3 {
                    assert_eq!(buf[4 * k + i], expected[4 * k + i], "pixel {k} lane {i}");
                }
                assert_eq!(
                    &buf[4 * k + 3..4 * k + 4],
                    &original[4 * k + 3..4 * k + 4],
                    "alpha lane {k}"
                );
            }
            assert_eq!(&buf[4 * npixels..], &original[4 * npixels..], "tail");
        }
    }

    #[test]
    fn float_to_u8_swap_rb_ffi_uninit_window() {
        use std::mem::MaybeUninit;

        // uninitialized export storage is realistic: pipe.backbuf is a
        // fresh pipeline allocation whose float lanes the kernel reads
        // only after the pixel pipeline wrote them, and whose u8 lanes
        // it fully overwrites for lanes 0..2.
        for npixels in [1usize, 3, 17] {
            let pixels: Vec<f32> = (0..4 * npixels)
                .map(|i| ((i as u64 * 2_654_435_761 + 0x9E37) % 256) as f32 / 255.0 - 0.25)
                .collect();
            let bytes = pack_float_pixels(&pixels);
            // head/tail sentinel bytes around the exact 16*npixels span,
            // window offset by 4 so a single-byte over/under-run fails.
            let mut storage = vec![MaybeUninit::<u8>::uninit(); 16 * npixels + 8];
            storage[..4].iter_mut().for_each(|b| {
                b.write(0xA5);
            });
            storage[4 + 16 * npixels..].iter_mut().for_each(|b| {
                b.write(0x5A);
            });
            for (slot, byte) in storage[4..4 + 16 * npixels].iter_mut().zip(bytes.iter()) {
                slot.write(*byte);
            }
            unsafe {
                darkroom_imageio_float_to_u8_swap_rb(storage.as_mut_ptr().add(4).cast(), npixels);
            }
            let storage_init: Vec<u8> =
                unsafe { storage.iter().map(|b| b.assume_init()).collect() };
            assert_eq!(&storage_init[..4], &[0xA5u8; 4], "head");
            assert_eq!(&storage_init[4 + 16 * npixels..], &[0x5Au8; 4], "tail");
            let mut direct = bytes.clone();
            float_to_u8_swap_rb_inplace(&mut direct, npixels);
            assert_eq!(&storage_init[4..4 + 16 * npixels], &direct[..]);
        }
    }

    #[test]
    fn float_to_u8_swap_rb_degenerate_guards_no_op() {
        // zero pixels: buffer untouched
        let mut buf = pack_float_pixels(&[0.5f32, 0.5, 0.5, 0.5]);
        float_to_u8_swap_rb_inplace(&mut buf, 0);
        assert_eq!(buf, pack_float_pixels(&[0.5f32, 0.5, 0.5, 0.5]));
        // truncated buffer: clamped iteration converts only the complete
        // pixel, never panics nor writes out of bounds; well-formed
        // callers never hit this path. Lanes (1.0, 0.0, 0.5) cross to
        // slots (128, 0, 255), pinning the swap on the short path too.
        let full = pack_float_pixels(&[1.0f32, 0.0, 0.5, 0.25, 1.0, 0.0, 0.5, 0.25]);
        let mut short = full[..20].to_vec(); // one pixel plus 4 stray bytes
        float_to_u8_swap_rb_inplace(&mut short, 2);
        assert_eq!((short[0], short[1], short[2]), (128, 0, 255));
        assert_eq!(&short[4..], &full[4..20]);
        // empty buffer with nonzero count: no-op
        let mut empty: Vec<u8> = vec![];
        float_to_u8_swap_rb_inplace(&mut empty, 1);
        assert!(empty.is_empty());
        // FFI guards: null, zero count, and overflowing 16*npixels
        // products are no-ops that never touch memory.
        let mut guarded = vec![7u8; 32];
        unsafe {
            darkroom_imageio_float_to_u8_swap_rb(std::ptr::null_mut(), 2);
            darkroom_imageio_float_to_u8_swap_rb(guarded.as_mut_ptr(), 0);
            darkroom_imageio_float_to_u8_swap_rb(guarded.as_mut_ptr(), usize::MAX);
            darkroom_imageio_float_to_u8_swap_rb(guarded.as_mut_ptr(), usize::MAX / 16 + 1);
        }
        assert_eq!(guarded, vec![7u8; 32]); // untouched
    }
}
