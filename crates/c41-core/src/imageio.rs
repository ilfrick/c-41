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
}
