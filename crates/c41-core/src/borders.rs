//! Border compose helper: `dt_iop_copy_image_with_border`.
//!
//! Ports the flat row loop of `dt_iop_copy_image_with_border`
//! (`src/develop/borders_helper.c:51`, loop ~51-112), which composes the
//! output image from the input row plus a border of `bcolor` and an optional
//! frameline of `flcolor`. The C orchestration (signature, both callers in
//! `src/iop/borders.c:572` and `src/iop/enlargecanvas.c:383`) is unchanged;
//! only the loop body moved here behind the stable C boundary.
//!
//! Row classes (all spans are 4-float RGBA pixels):
//! - outside the frameline (`row < border_top || row >= border_bot`):
//!   the whole row is `bcolor`;
//! - frameline band (`row < fl_top || row >= fl_bot`): left `bcolor` run,
//!   middle `flcolor` run (`border_left..border_right`), right `bcolor` run;
//! - inner border band (`row < image_top || row >= image_bot`): five runs,
//!   `bcolor` / `flcolor` / `bcolor` / `flcolor` / `bcolor` split at
//!   `border_left`, `fl_left`, `fl_right`, `border_right`;
//! - image band: left `bcolor` run, optional inner frameline plus inner
//!   `bcolor` run when `image_left > border_left`, the copied image row
//!   sourced at `(row - image_top) * stride`, right `bcolor` run, and the
//!   optional outer frameline plus outer `bcolor` run when
//!   `width > fl_right`.
//!
//! Fidelity notes:
//! - `set_pixels` / `copy_pixels` are inlined into the kernel below; each
//!   `copy_pixel_nontemporal` 4-float store becomes a plain 4-float store.
//!   The trailing `dt_omploop_sfence` is dropped on purpose: the fence only
//!   ordered SSE streaming stores for other cores, and normal stores need no
//!   fence (the old OpenMP barrier implied the same visibility; the serial
//!   Rust kernel has no cross-thread readers at all).
//! - A non-positive span (`npixels <= 0` in C) writes nothing; the helpers
//!   below return early on `count <= 0` for the same effect.
//! - Out-of-range indices are clamped to the row instead of panicking in the
//!   safe kernel; the C callers always pass sane geometry, so clamping only
//!   fires on corrupt input.

// The C struct starts with two `dt_aligned_pixel_t` (16-byte aligned
// `float[4]`) fields, so the whole struct is 16-byte aligned and padded to a
// multiple of 16. `repr(C, align(16))` reproduces that exactly: 32 bytes of
// color plus 30 ints (120 bytes) plus 8 bytes of tail padding = 160 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct BorderPositions {
    pub bcolor: [f32; 4],
    pub flcolor: [f32; 4],
    pub border_top: i32,
    pub fl_top: i32,
    pub image_top: i32,
    pub border_left: i32,
    pub fl_left: i32,
    pub image_left: i32,
    pub image_right: i32,
    pub fl_right: i32,
    pub border_right: i32,
    pub width: i32,
    pub image_bot: i32,
    pub fl_bot: i32,
    pub border_bot: i32,
    pub height: i32,
    pub stride: i32,
    pub border_in_x: i32,
    pub border_in_y: i32,
    pub border_size_t: i32,
    pub border_size_b: i32,
    pub border_size_l: i32,
    pub border_size_r: i32,
    pub frame_size: i32,
    pub frame_tl_in_x: i32,
    pub frame_tl_out_x: i32,
    pub frame_tl_in_y: i32,
    pub frame_tl_out_y: i32,
    pub frame_br_in_x: i32,
    pub frame_br_out_x: i32,
    pub frame_br_in_y: i32,
    pub frame_br_out_y: i32,
}

/// Fill `count` RGBA pixels of `out_row` with `color`, starting at pixel
/// `start`. Matches `set_pixels`: a non-positive count writes nothing, and
/// the range is clamped to the row so corrupt bounds can never panic.
fn fill_span(out_row: &mut [f32], width: usize, start: i64, count: i64, color: &[f32; 4]) {
    if count <= 0 {
        return;
    }
    let max_pix = (out_row.len() / 4).min(width) as i64;
    let s = start.clamp(0, max_pix) as usize;
    let e = (start + count).clamp(0, max_pix) as usize;
    if e <= s {
        return;
    }
    for p in s..e {
        let o = p * 4;
        out_row[o..o + 4].copy_from_slice(color);
    }
}

/// Copy `count` RGBA pixels into `out_row` at pixel `dest_start` from `inp`
/// at float offset `src_base`. Matches `copy_pixels`: a non-positive count
/// copies nothing; out-of-range pixels on either side are skipped so short
/// buffers can never panic.
fn copy_span(
    out_row: &mut [f32],
    inp: &[f32],
    src_base: i64,
    dest_start: i64,
    count: i64,
    width: usize,
) {
    if count <= 0 {
        return;
    }
    let max_dst = (out_row.len() / 4).min(width) as i64;
    let in_len = inp.len() as i64;
    for i in 0..count {
        let d = dest_start + i;
        if d < 0 || d >= max_dst {
            continue;
        }
        let s = src_base + i * 4;
        if s < 0 || s + 4 > in_len {
            continue;
        }
        let di = d as usize * 4;
        let si = s as usize;
        out_row[di..di + 4].copy_from_slice(&inp[si..si + 4]);
    }
}

/// Safe kernel behind [`darkroom_borders_copy_with_border`].
///
/// Composes `width * height` RGBA pixels of `out` from `inp` (input rows are
/// `stride` RGBA pixels wide) following the four row classes described in
/// the module docs. Degenerate dims (`width <= 0`, `height <= 0`,
/// `stride <= 0`) are a no-op; rows or spans that do not fit the slices are
/// skipped instead of panicking.
pub fn copy_image_with_border_kernel(out: &mut [f32], inp: &[f32], b: &BorderPositions) {
    let width = b.width as i64;
    let height = b.height as i64;
    let stride = b.stride as i64;
    if width <= 0 || height <= 0 || stride <= 0 {
        return;
    }
    let w = width as usize;
    let image_width = b.image_right as i64 - b.image_left as i64;
    let border_top = b.border_top as i64;
    let border_bot = b.border_bot as i64;
    let fl_top = b.fl_top as i64;
    let fl_bot = b.fl_bot as i64;
    let image_top = b.image_top as i64;
    let image_bot = b.image_bot as i64;
    let border_left = b.border_left as i64;
    let border_right = b.border_right as i64;
    let fl_left = b.fl_left as i64;
    let fl_right = b.fl_right as i64;
    let image_left = b.image_left as i64;
    let image_right = b.image_right as i64;
    for row in 0..height {
        let base = row as usize * w * 4;
        if base.saturating_add(w * 4) > out.len() {
            continue;
        }
        let out_row = &mut out[base..base + w * 4];
        if row < border_top || row >= border_bot {
            // Top/bottom border outside the frameline: entirely border color.
            fill_span(out_row, w, 0, width, &b.bcolor);
        } else if row < fl_top || row >= fl_bot {
            // Top/bottom frameline: border / frame / border.
            fill_span(out_row, w, 0, border_left, &b.bcolor);
            fill_span(out_row, w, border_left, border_right - border_left, &b.flcolor);
            fill_span(out_row, w, border_right, width - border_right, &b.bcolor);
        } else if row < image_top || row >= image_bot {
            // Top/bottom border inside the frameline: five runs.
            fill_span(out_row, w, 0, border_left, &b.bcolor);
            fill_span(out_row, w, border_left, fl_left - border_left, &b.flcolor);
            fill_span(out_row, w, fl_left, fl_right - fl_left, &b.bcolor);
            fill_span(out_row, w, fl_right, border_right - fl_right, &b.flcolor);
            fill_span(out_row, w, border_right, width - border_right, &b.bcolor);
        } else {
            // Image band: left border (with optional frame line), copied
            // image row, right border (with optional frame line).
            fill_span(out_row, w, 0, border_left, &b.bcolor);
            if b.image_left > b.border_left {
                fill_span(out_row, w, border_left, fl_left - border_left, &b.flcolor);
                fill_span(out_row, w, fl_left, image_left - fl_left, &b.bcolor);
            }
            copy_span(
                out_row,
                inp,
                (row - image_top) * stride * 4,
                image_left,
                image_width,
                w,
            );
            fill_span(out_row, w, image_right, fl_right - image_right, &b.bcolor);
            if b.width > b.fl_right {
                fill_span(out_row, w, fl_right, border_right - fl_right, &b.flcolor);
                fill_span(out_row, w, border_right, width - border_right, &b.bcolor);
            }
        }
    }
}

/// Structurally divergent reference for [`copy_image_with_border_kernel`].
///
/// Same guards and row-band predicates, but each output pixel is classified
/// independently by a per-pixel predicate chain (no span helpers, no span
/// arithmetic) and image floats are copied lane by lane instead of pixel by
/// pixel. Column iteration runs outermost per row band state cached in
/// locals — a different visiting shape that must still agree bit-exactly.
#[allow(dead_code)]
fn ref_copy_image_with_border(out: &mut [f32], inp: &[f32], b: &BorderPositions) {
    let width = b.width as i64;
    let height = b.height as i64;
    let stride = b.stride as i64;
    if width <= 0 || height <= 0 || stride <= 0 {
        return;
    }
    let w = width as usize;
    let in_len = inp.len() as i64;
    for row in 0..height {
        // Same short-buffer rule as the kernel: rows that do not fit
        // entirely are left untouched instead of partially written.
        let base = row as usize * w * 4;
        if base.saturating_add(w * 4) > out.len() {
            continue;
        }
        // Row band: 0 = outer border, 1 = frameline band, 2 = inner border, 3 = image.
        let band: u8 = if row < b.border_top as i64 || row >= b.border_bot as i64 {
            0
        } else if row < b.fl_top as i64 || row >= b.fl_bot as i64 {
            1
        } else if row < b.image_top as i64 || row >= b.image_bot as i64 {
            2
        } else {
            3
        };
        for col in 0..width {
            let di = (row as usize * w + col as usize) * 4;
            if di + 4 > out.len() {
                continue;
            }
            // Per-pixel color selector; returns None for copied image lanes.
            let mut fill: Option<[f32; 4]> = None;
            let mut src: Option<i64> = None;
            match band {
                0 => fill = Some(b.bcolor),
                1 => {
                    fill = Some(if col < b.border_left as i64 {
                        b.bcolor
                    } else if col < b.border_right as i64 {
                        b.flcolor
                    } else {
                        b.bcolor
                    });
                }
                2 => {
                    fill = Some(if col < b.border_left as i64 {
                        b.bcolor
                    } else if col < b.fl_left as i64 {
                        b.flcolor
                    } else if col < b.fl_right as i64 {
                        b.bcolor
                    } else if col < b.border_right as i64 {
                        b.flcolor
                    } else {
                        b.bcolor
                    });
                }
                _ => {
                    if col < b.border_left as i64 {
                        fill = Some(b.bcolor);
                    } else if b.image_left > b.border_left
                        && col >= b.border_left as i64
                        && col < b.fl_left as i64
                    {
                        fill = Some(b.flcolor);
                    } else if b.image_left > b.border_left
                        && col >= b.fl_left as i64
                        && col < b.image_left as i64
                    {
                        fill = Some(b.bcolor);
                    } else if col >= b.image_left as i64 && col < b.image_right as i64 {
                        let lane = col - b.image_left as i64;
                        let s = (row - b.image_top as i64) * stride * 4 + lane * 4;
                        src = Some(s);
                    } else if col < b.fl_right as i64 {
                        fill = Some(b.bcolor);
                    } else if b.width > b.fl_right && col < b.border_right as i64 {
                        fill = Some(b.flcolor);
                    } else {
                        fill = Some(b.bcolor);
                    }
                }
            }
            if let Some(c) = fill {
                out[di..di + 4].copy_from_slice(&c);
            } else if let Some(s) = src {
                // Pixel-granular like copy_span: a source pixel that does
                // not fit entirely leaves the destination pixel untouched.
                if s >= 0 && s + 4 <= in_len {
                    let si = s as usize;
                    out[di..di + 4].copy_from_slice(&inp[si..si + 4]);
                }
            }
        }
    }
}

/// Compose the output image with border and optional frameline.
///
/// Port of `dt_iop_copy_image_with_border` (`src/develop/borders_helper.c:51`):
/// `out` holds `width * height` RGBA floats, `inp` holds the input rows of
/// `stride` RGBA pixels each, and `binfo` carries the border / frameline / image
/// bounds plus `bcolor` / `flcolor` (see [`BorderPositions`]). The C loop ran
/// under OpenMP with SSE streaming stores plus a trailing fence; this runs
/// the same per-row composition serially with plain 4-float stores, which
/// need no fence.
///
/// NULL pointers, degenerate dims (`width`, `height` or `stride` <= 0), and
/// overflowing `width * height * 4` or input-row products are no-ops. The
/// safe kernel additionally clamps short Rust slices instead of panicking,
/// but the FFI entry derives slice lengths from the struct fields, so the
/// caller must guarantee the sizes below.
///
/// # Safety
/// `out` must hold at least `width * height * 4` floats and `inp` at least
/// `(image_bot - image_top) * stride * 4` floats (or be NULL, in which case
/// this is a no-op), with `width`, `height`, `stride` read from `binfo`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_borders_copy_with_border(
    out: *mut f32,
    inp: *const f32,
    binfo: *const BorderPositions,
) {
    if out.is_null() || inp.is_null() || binfo.is_null() {
        return;
    }
    let b = &*binfo;
    let width = b.width as i64;
    let height = b.height as i64;
    let stride = b.stride as i64;
    if width <= 0 || height <= 0 || stride <= 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let s = stride as usize;
    let Some(out_n) = w.checked_mul(h).and_then(|v| v.checked_mul(4)) else {
        return;
    };
    if out_n == 0 {
        return;
    }
    let rows = (b.image_bot as i64 - b.image_top as i64).max(0) as usize;
    let Some(in_n) = rows.checked_mul(s).and_then(|v| v.checked_mul(4)) else {
        return;
    };
    let out_slice = std::slice::from_raw_parts_mut(out, out_n);
    let in_slice = std::slice::from_raw_parts(inp, in_n);
    copy_image_with_border_kernel(out_slice, in_slice, b);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn lcg_fill(buf: &mut [f32], mut state: u32) {
        for x in buf.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *x = (state as f32) / (u32::MAX as f32);
        }
    }

    // Geometry without a frame: every boundary collapses onto the image
    // rect, so the frameline spans are empty and the frame guards are false.
    fn no_frame_binfo() -> BorderPositions {
        BorderPositions {
            bcolor: [0.25, 0.5, 0.75, 1.0],
            flcolor: [1.0, 0.0, 0.0, 1.0],
            border_top: 1,
            fl_top: 1,
            image_top: 1,
            border_left: 2,
            fl_left: 2,
            image_left: 2,
            image_right: 5,
            fl_right: 7,
            border_right: 7,
            width: 7,
            image_bot: 3,
            fl_bot: 4,
            border_bot: 4,
            height: 4,
            stride: 3,
            border_in_x: 2,
            border_in_y: 1,
            border_size_t: 1,
            border_size_b: 1,
            border_size_l: 2,
            border_size_r: 2,
            frame_size: 0,
            frame_tl_in_x: 0,
            frame_tl_out_x: 0,
            frame_tl_in_y: 0,
            frame_tl_out_y: 0,
            frame_br_in_x: 0,
            frame_br_out_x: 0,
            frame_br_in_y: 0,
            frame_br_out_y: 0,
        }
    }

    // Geometry with a one-pixel frameline around a 2x2 image in a 6x6 out.
    fn frame_binfo() -> BorderPositions {
        BorderPositions {
            bcolor: [0.1, 0.2, 0.3, 1.0],
            flcolor: [0.9, 0.8, 0.7, 1.0],
            border_top: 1,
            fl_top: 2,
            image_top: 3,
            border_left: 1,
            fl_left: 2,
            image_left: 3,
            image_right: 5,
            fl_right: 6,
            border_right: 7,
            width: 8,
            image_bot: 5,
            fl_bot: 6,
            border_bot: 7,
            height: 8,
            stride: 2,
            border_in_x: 3,
            border_in_y: 3,
            border_size_t: 3,
            border_size_b: 3,
            border_size_l: 3,
            border_size_r: 3,
            frame_size: 1,
            frame_tl_in_x: 2,
            frame_tl_out_x: 1,
            frame_tl_in_y: 2,
            frame_tl_out_y: 1,
            frame_br_in_x: 6,
            frame_br_out_x: 7,
            frame_br_in_y: 6,
            frame_br_out_y: 7,
        }
    }

    #[test]
    fn struct_layout_matches_c() {
        // Two 16-byte colors + 30 ints, padded to a multiple of 16.
        assert_eq!(std::mem::size_of::<BorderPositions>(), 160);
        assert_eq!(std::mem::align_of::<BorderPositions>(), 16);
    }

    #[test]
    fn outer_rows_are_full_border_fill() {
        // No-frame geometry: rows 0 and 3 lie outside the frameline, so every
        // pixel must be bcolor; rows 1-2 are image rows.
        let b = no_frame_binfo();
        let inp = vec![9.0f32; 2 * 3 * 4];
        let mut out = vec![-1.0f32; 7 * 4 * 4];
        let mut reference = vec![-2.0f32; 7 * 4 * 4];
        copy_image_with_border_kernel(&mut out, &inp, &b);
        ref_copy_image_with_border(&mut reference, &inp, &b);
        assert_eq!(to_bits(&out), to_bits(&reference));
        for row in [0usize, 3] {
            for col in 0..7usize {
                let o = (row * 7 + col) * 4;
                assert_eq!(to_bits(&out[o..o + 4]), to_bits(&b.bcolor));
            }
        }
    }

    #[test]
    fn frameline_band_has_border_frame_border_spans() {
        // Framed geometry, row 1 is in [border_top, fl_top): pixels 0..1
        // bcolor, 1..7 flcolor, 7..8 bcolor.
        let b = frame_binfo();
        let inp = vec![0.0f32; 2 * 2 * 4];
        let mut out = vec![-1.0f32; 8 * 8 * 4];
        copy_image_with_border_kernel(&mut out, &inp, &b);
        let row = 1usize;
        for col in 0..8usize {
            let o = (row * 8 + col) * 4;
            let want = if col < 1 {
                b.bcolor
            } else if col < 7 {
                b.flcolor
            } else {
                b.bcolor
            };
            assert_eq!(to_bits(&out[o..o + 4]), to_bits(&want), "col {col}");
        }
    }

    #[test]
    fn inner_border_band_has_five_spans() {
        // Framed geometry, row 2 is in [fl_top, image_top): bcolor for
        // 0..1, flcolor for 1..2, bcolor for 2..6, flcolor for 6..7,
        // bcolor for 7..8.
        let b = frame_binfo();
        let inp = vec![0.0f32; 2 * 2 * 4];
        let mut out = vec![-1.0f32; 8 * 8 * 4];
        copy_image_with_border_kernel(&mut out, &inp, &b);
        let row = 2usize;
        let spans: [(std::ops::Range<usize>, [f32; 4]); 5] = [
            (0..1, b.bcolor),
            (1..2, b.flcolor),
            (2..6, b.bcolor),
            (6..7, b.flcolor),
            (7..8, b.bcolor),
        ];
        for (cols, want) in spans {
            for col in cols {
                let o = (row * 8 + col) * 4;
                assert_eq!(to_bits(&out[o..o + 4]), to_bits(&want), "col {col}");
            }
        }
    }

    #[test]
    fn image_band_copies_row_with_stride_and_frames() {
        // Framed geometry: image rows 3..5 copy input rows 0..1 (stride 2)
        // at columns 3..5; column 0 is outer border, 1 inner frame,
        // 2 inner border, 5 inner border, 6 outer frame, 7 outer border.
        let b = frame_binfo();
        let mut inp = vec![0.0f32; 2 * 2 * 4];
        for r in 0..2usize {
            for c in 0..2usize {
                let o = (r * 2 + c) * 4;
                inp[o] = (r * 10 + c) as f32;
                inp[o + 1] = (r * 10 + c) as f32 + 0.5;
                inp[o + 2] = (r * 10 + c) as f32 + 0.25;
                inp[o + 3] = 1.0;
            }
        }
        let mut out = vec![-1.0f32; 8 * 8 * 4];
        copy_image_with_border_kernel(&mut out, &inp, &b);
        for (row, src_row) in [(3usize, 0usize), (4, 1)] {
            // Left outer border, inner frame, inner border.
            assert_eq!(to_bits(&out[(row * 8) * 4..(row * 8) * 4 + 4]), to_bits(&b.bcolor));
            assert_eq!(
                to_bits(&out[(row * 8 + 1) * 4..(row * 8 + 1) * 4 + 4]),
                to_bits(&b.flcolor)
            );
            assert_eq!(
                to_bits(&out[(row * 8 + 2) * 4..(row * 8 + 2) * 4 + 4]),
                to_bits(&b.bcolor)
            );
            // Copied image pixels.
            for c in 0..2usize {
                let o = (row * 8 + 3 + c) * 4;
                let s = (src_row * 2 + c) * 4;
                assert_eq!(to_bits(&out[o..o + 4]), to_bits(&inp[s..s + 4]));
            }
            // Inner border, outer frame, outer border.
            assert_eq!(
                to_bits(&out[(row * 8 + 5) * 4..(row * 8 + 5) * 4 + 4]),
                to_bits(&b.bcolor)
            );
            assert_eq!(
                to_bits(&out[(row * 8 + 6) * 4..(row * 8 + 6) * 4 + 4]),
                to_bits(&b.flcolor)
            );
            assert_eq!(
                to_bits(&out[(row * 8 + 7) * 4..(row * 8 + 7) * 4 + 4]),
                to_bits(&b.bcolor)
            );
        }
    }

    #[test]
    fn image_band_without_frame_skips_frame_spans() {
        // No-frame geometry: image rows copy with no frame runs; the right
        // border run starts straight at image_right.
        let b = no_frame_binfo();
        let mut inp = vec![0.0f32; 2 * 3 * 4];
        lcg_fill(&mut inp, 0xB0BD);
        let mut out = vec![-1.0f32; 7 * 4 * 4];
        copy_image_with_border_kernel(&mut out, &inp, &b);
        for (row, src_row) in [(1usize, 0usize), (2, 1)] {
            for col in 0..2usize {
                let o = (row * 7 + col) * 4;
                assert_eq!(to_bits(&out[o..o + 4]), to_bits(&b.bcolor), "col {col}");
            }
            for c in 0..3usize {
                let o = (row * 7 + 2 + c) * 4;
                let s = (src_row * 3 + c) * 4;
                assert_eq!(to_bits(&out[o..o + 4]), to_bits(&inp[s..s + 4]));
            }
            for col in 5..7usize {
                let o = (row * 7 + col) * 4;
                assert_eq!(to_bits(&out[o..o + 4]), to_bits(&b.bcolor), "col {col}");
            }
        }
    }

    #[test]
    fn kernel_matches_reference_with_and_without_frame() {
        for (mut b, seed) in [(no_frame_binfo(), 0x1234u32), (frame_binfo(), 0x5678u32)] {
            b.bcolor = [0.11, 0.22, 0.33, 0.44];
            b.flcolor = [0.99, 0.88, 0.77, 0.66];
            let rows = (b.image_bot - b.image_top).max(0) as usize;
            let mut inp = vec![0.0f32; rows * b.stride as usize * 4];
            lcg_fill(&mut inp, seed);
            let n = b.width as usize * b.height as usize * 4;
            let mut direct = vec![0.0f32; n];
            let mut reference = vec![0.0f32; n];
            lcg_fill(&mut direct, seed ^ 0x9E37);
            reference.copy_from_slice(&direct);
            copy_image_with_border_kernel(&mut direct, &inp, &b);
            ref_copy_image_with_border(&mut reference, &inp, &b);
            assert_eq!(to_bits(&direct), to_bits(&reference));
        }
    }

    #[test]
    fn golden_vector_framed() {
        // 4x3 output, 2x1 image at (1,1), one-pixel frame ring at the
        // output edge is absent here: border ring is bcolor (B), frame
        // ring is flcolor (F), image pixels are I.
        let b = BorderPositions {
            bcolor: [0.0, 0.0, 0.0, 1.0],
            flcolor: [1.0, 1.0, 1.0, 1.0],
            border_top: 0,
            fl_top: 0,
            image_top: 1,
            border_left: 0,
            fl_left: 0,
            image_left: 1,
            image_right: 3,
            fl_right: 4,
            border_right: 4,
            width: 4,
            image_bot: 2,
            fl_bot: 3,
            border_bot: 3,
            height: 3,
            stride: 2,
            border_in_x: 1,
            border_in_y: 1,
            border_size_t: 1,
            border_size_b: 1,
            border_size_l: 1,
            border_size_r: 1,
            frame_size: 0,
            frame_tl_in_x: 0,
            frame_tl_out_x: 0,
            frame_tl_in_y: 0,
            frame_tl_out_y: 0,
            frame_br_in_x: 0,
            frame_br_out_x: 0,
            frame_br_in_y: 0,
            frame_br_out_y: 0,
        };
        let inp = vec![0.5f32, 0.25, 0.125, 1.0, 0.75, 0.5, 0.25, 1.0];
        let mut out = vec![-1.0f32; 4 * 3 * 4];
        copy_image_with_border_kernel(&mut out, &inp, &b);
        let bpx = [0.0f32, 0.0, 0.0, 1.0];
        let mut expected = Vec::new();
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&inp[0..4]);
        expected.extend_from_slice(&inp[4..8]);
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&bpx);
        expected.extend_from_slice(&bpx);
        assert_eq!(to_bits(&out), to_bits(&expected));
    }

    #[test]
    fn degenerate_dims_are_noop() {
        let mut b = frame_binfo();
        let inp = vec![1.0f32; 16];
        let mut out = vec![2.0f32; 8 * 8 * 4];
        let before = out.clone();
        for (w, h, s) in [(0, 8, 2), (8, 0, 2), (8, 8, 0), (-3, 8, 2), (8, -1, 2)] {
            b.width = w;
            b.height = h;
            b.stride = s;
            copy_image_with_border_kernel(&mut out, &inp, &b);
            assert_eq!(out, before);
        }
    }

    #[test]
    fn short_buffers_do_not_panic() {
        // Truncated out/in: kernel clamps instead of panicking.
        let b = frame_binfo();
        let mut tiny_out = vec![0.0f32; 8];
        let tiny_in = vec![0.0f32; 4];
        copy_image_with_border_kernel(&mut tiny_out, &tiny_in, &b);
        let mut empty_out: Vec<f32> = vec![];
        let empty_in: Vec<f32> = vec![];
        copy_image_with_border_kernel(&mut empty_out, &empty_in, &b);
        // Reference agrees on the same truncated buffers.
        let mut ref_out = vec![0.0f32; 8];
        ref_copy_image_with_border(&mut ref_out, &tiny_in, &b);
        assert_eq!(to_bits(&tiny_out), to_bits(&ref_out));
    }

    #[test]
    fn ffi_round_trip_matches_kernel() {
        let b = frame_binfo();
        let rows = (b.image_bot - b.image_top) as usize;
        let mut inp = vec![0.0f32; rows * b.stride as usize * 4];
        lcg_fill(&mut inp, 0xF00D);
        let n = b.width as usize * b.height as usize * 4;
        let mut via_ffi = vec![0.0f32; n];
        let mut direct = vec![0.0f32; n];
        unsafe {
            darkroom_borders_copy_with_border(via_ffi.as_mut_ptr(), inp.as_ptr(), &b);
        }
        copy_image_with_border_kernel(&mut direct, &inp, &b);
        assert_eq!(to_bits(&via_ffi), to_bits(&direct));
    }

    #[test]
    fn ffi_null_and_dims_guards() {
        let b = frame_binfo();
        let inp = vec![1.0f32; 16];
        let mut out = vec![2.0f32; 8 * 8 * 4];
        unsafe {
            darkroom_borders_copy_with_border(std::ptr::null_mut(), inp.as_ptr(), &b);
            darkroom_borders_copy_with_border(out.as_mut_ptr(), std::ptr::null(), &b);
            darkroom_borders_copy_with_border(
                out.as_mut_ptr(),
                inp.as_ptr(),
                std::ptr::null(),
            );
        }
        assert_eq!(out, vec![2.0; 8 * 8 * 4]);
        let mut bad = b;
        bad.width = 0;
        unsafe {
            darkroom_borders_copy_with_border(out.as_mut_ptr(), inp.as_ptr(), &bad);
        }
        assert_eq!(out, vec![2.0; 8 * 8 * 4]);
        bad = b;
        bad.stride = -1;
        unsafe {
            darkroom_borders_copy_with_border(out.as_mut_ptr(), inp.as_ptr(), &bad);
        }
        assert_eq!(out, vec![2.0; 8 * 8 * 4]);
    }
}
