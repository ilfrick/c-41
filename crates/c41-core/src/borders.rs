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
#[derive(Clone, Copy, Default)]
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

// ─────────────────────────────────────────────────────────────────────────────
// Pure-Rust `Framing` module (darktable `borders`, iop_order.c v50 pos 76).
//
// This is a *whole-frame, scale-1* port of the module's constant-border path
// — the product's only case (the UI drives no aspect-ratio or basis sliders,
// so `aspect` is locked to the C's `DT_IOP_BORDERS_ASPECT_CONSTANT_VALUE`
// (`-1.0`) and `basis` to `DT_IOP_BORDERS_BASIS_AUTO`, the C default). The
// boundary geometry (`modify_roi_out`, whole-frame `dt_iop_setup_binfo`) is
// ported exactly; the row-loop compose reuses [`copy_image_with_border_kernel`].
//
// Framing runs in the **geometry family**: like crop it is a base-image
// compose that changes the buffer dimensions *before* the per-pixel pipeline
// runs. See `c41-ui`'s `apply_geometry_to_base` / export funnels for where
// each render path invokes [`Borders::apply`] / [`Borders::apply_srgb8`].

/// The c41 wiring subset of `dt_iop_borders_params_t` (v4). Fields mirror the
/// C struct's names, ranges and `$DEFAULT`s for the constant-border path:
/// `color` (border colour, 1.0 = white), `size` (border width as a fraction
/// of the chosen basis, 0..0.5, 0.1), `pos_h`/`pos_v` (picture position ratio
/// into the final frame, 0..1, 0.5 — i.e. centred), `frame_size` (frame-line
/// width relative to the border width, 0..1, 0.0 = no frame), `frame_offset`
/// (frame position relative to the picture edge, 0..1, 0.5) and
/// `frame_color` (frame-line colour, 0.0 = black).
///
/// Omitted by design: `aspect`/`aspect_text`/`aspect_orient` (locked to the
/// constant border), `basis` (locked to AUTO → LONGER for constant borders),
/// `max_border_size` (TRUE is the only surviving behaviour) and the text
/// sentinels. The UI stores exactly these fields in `PreviewParams` and
/// round-trips them through `c41_core::borders::BordersParams`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BordersParams {
    /// Border colour (RGB, 0..1, darktable default 1.0 = white).
    pub color: [f32; 3],
    /// Border width relative to the chosen basis (0..0.5, default 0.1).
    pub size: f32,
    /// Picture horizontal position ratio into the final frame (0..1, 0.5).
    pub pos_h: f32,
    /// Picture vertical position ratio into the final frame (0..1, 0.5).
    pub pos_v: f32,
    /// Frame-line width relative to the border width (0..1, 0.0 = no frame).
    pub frame_size: f32,
    /// Frame-line position relative to the picture edge (0..1, 0.5).
    pub frame_offset: f32,
    /// Frame-line colour (RGB, 0..1, default 0.0 = black).
    pub frame_color: [f32; 3],
}

impl Default for BordersParams {
    fn default() -> Self {
        // Mirrors dt_iop_borders_params_t's `$DEFAULT`s for the fields we
        // wire (`color 1.0`, `size 0.1`, `pos_* 0.5`, `frame_size 0.0`,
        // `frame_offset 0.5`, `frame_color 0.0`; `default_v3` in borders.c
        // also seeds these exact values).
        Self {
            color: [1.0, 1.0, 1.0],
            size: 0.1,
            pos_h: 0.5,
            pos_v: 0.5,
            frame_size: 0.0,
            frame_offset: 0.5,
            frame_color: [0.0, 0.0, 0.0],
        }
    }
}

impl BordersParams {
    /// True when every field is a finite number. A corrupt blob with a NaN/Inf
    /// parameter is treated as a no-op by the compose (like
    /// [`crate::geometry::apply_rotate`]) rather than being allowed to poison
    /// the geometry arithmetic.
    fn is_finite(&self) -> bool {
        self.size.is_finite()
            && self.pos_h.is_finite()
            && self.pos_v.is_finite()
            && self.frame_size.is_finite()
            && self.frame_offset.is_finite()
            && self.color.iter().all(|c| c.is_finite())
            && self.frame_color.iter().all(|c| c.is_finite())
    }
}

/// The `Framing`/`borders` compose. See the module docs for the port scope.
///
/// `apply` is the linear-RGBA (4-float) form used by the raw preview and raw
/// export funnels; `apply_srgb8` is the 8-bit form used by the non-raw paths.
pub struct Borders;

impl Borders {
    /// Port of `modify_roi_out`'s **constant-border** path
    /// (`src/iop/borders.c`, `is_constant_border` arm). For a constant border
    /// the C's `DTO_IOP_BORDERS_BASIS_AUTO` resolves via `LONGER` to whichever
    /// of width/height is the larger side; that side is scaled by
    /// `roundf(basis_in / (1 - size))` and the other by
    /// `other_in + basis_out - basis_in`, then both are `CLAMP`ed to
    /// `[1, 3 * max(w, h)]`. `basis` is not a parameter here — it is locked to
    /// AUTO exactly as described.
    fn modify_roi_out(w: usize, h: usize, size: f32) -> (usize, usize) {
        // Mirrors the C's `const float size = fabsf(d->size);` — the UI slider
        // is 0..0.5, but a persisted blob is untrusted input and the C path
        // takes the magnitude before any ratio.
        let size = size.abs();
        // `size >= 1.0` makes `basis/(1-size)` non-finite (and the subsequent
        // `oh = hi + ow - wi` i64 arithmetic overflow in a debug build). The C
        // is UB here; treat anything at/above the cap as a no-op — "geometry
        // never fails a render". Not reachable from the UI (slider max 0.5).
        if !(size < 1.0) {
            return (w, h);
        }
        let (wi, hi) = (w as i64, h as i64);
        let max_dim = wi.max(hi);
        let (mut ow, mut oh) = (wi, hi);
        // basis AUTO → LONGER → WIDTH when `width > height`, else HEIGHT.
        if wi > hi {
            let basis_out = ((w as f32) / (1.0 - size)).round();
            ow = basis_out as i64;
            oh = hi + ow - wi;
        } else {
            let basis_out = ((h as f32) / (1.0 - size)).round();
            oh = basis_out as i64;
            ow = wi + oh - hi;
        }
        ow = ow.clamp(1, 3 * max_dim);
        oh = oh.clamp(1, 3 * max_dim);
        (ow as usize, oh as usize)
    }

    /// Port of the **whole-frame, scale-1, `roi_out.x/y` = 0** geometry of
    /// `dt_iop_setup_binfo` (`src/develop/borders_helper.c`). Generates a
    /// [`BorderPositions`] for an `in_w × in_h` input placed in an
    /// `out_w × out_h` frame at `pos_v`/`pos_h`, with the border/frame colours
    /// and the frame-line branch (`frame_size`/`frame_offset`, capped by the
    /// `border_min_size` logic). Integer rounding, `CLAMP`/`MAX` and the
    /// bit-exact ordering match the C; `pos_v`/`pos_h` are genuine parameters.
    fn setup_binfo(
        in_w: i32,
        in_h: i32,
        out_w: i32,
        out_h: i32,
        pos_v: f32,
        pos_h: f32,
        bcolor: &[f32; 3],
        fcolor: &[f32; 3],
        f_size: f32,
        f_offset: f32,
    ) -> BorderPositions {
        let has_left = pos_h > 0.0;
        let has_right = pos_h < 1.0;
        let has_top = pos_v > 0.0;
        let has_bottom = pos_v < 1.0;

        // `ceilf((buf_out.width - buf_in.width) * roi_in->scale)` with scale 1:
        // ceil of an integer difference is itself, so this is exact. `i32`
        // matches the C's `int` fields; the intermediate products explicitly
        // widen to f32 exactly where the C's int×float promotions happen.
        let border_tot_width = out_w - in_w;
        let border_tot_height = out_h - in_h;

        let border_size_t = if has_top { (border_tot_height as f32 * pos_v) as i32 } else { 0 };
        let border_size_b = if has_bottom { border_tot_height - border_size_t } else { 0 };
        let border_size_l = if has_left { (border_tot_width as f32 * pos_h) as i32 } else { 0 };
        let border_size_r = if has_right { border_tot_width - border_size_l } else { 0 };

        let (image_right, border_in_x) = if has_right {
            let bx = border_size_l.clamp(0, out_w);
            (bx + in_w, bx)
        } else {
            (out_w, border_tot_width.clamp(0, out_w))
        };
        let (image_bottom, border_in_y) = if has_bottom {
            let by = border_size_t.clamp(0, out_h);
            (by + in_h, by)
        } else {
            (out_h, border_tot_height.clamp(0, out_h))
        };

        let mut b = BorderPositions::default();
        for c in 0..3 {
            b.bcolor[c] = bcolor[c];
            b.flcolor[c] = fcolor[c];
        }
        b.bcolor[3] = 1.0;
        b.flcolor[3] = 1.0;

        b.border_top = border_in_y;
        b.fl_top = border_in_y;
        b.image_top = border_in_y;
        b.border_left = border_in_x;
        b.fl_left = border_in_x;
        b.image_left = border_in_x;
        b.image_right = image_right;
        b.fl_right = out_w;
        b.border_right = out_w;
        b.width = out_w;
        b.image_bot = image_bottom;
        b.fl_bot = out_h;
        b.border_bot = out_h;
        b.height = out_h;
        b.stride = in_w;
        b.border_in_x = border_in_x;
        b.border_in_y = border_in_y;
        // The per-side border sizes, exactly as the C writes them into
        // `binfo` ahead of the frame branch (`dt_iop_setup_binfo` assigns
        // `binfo->border_size_{t,b,l,r}` before the sub-rectangle geometry).
        b.border_size_t = border_size_t;
        b.border_size_b = border_size_b;
        b.border_size_l = border_size_l;
        b.border_size_r = border_size_r;

        // Frame-line branch.
        let border_min_size = border_size_t
            .min(border_size_b)
            .min(border_size_l.min(border_size_r));
        b.frame_size = (border_min_size as f32 * f_size) as i32;

        if b.frame_size > 0 {
            let image_lx = border_size_l; // - roi_out.x (0)
            let image_ty = border_size_t; // - roi_out.y (0)
            let frame_space = border_min_size - b.frame_size;
            let frame_offset = (frame_space as f32 * f_offset) as i32;

            b.frame_tl_in_x = (border_in_x - frame_offset).max(0);
            b.frame_tl_out_x = (b.frame_tl_in_x - b.frame_size).max(0);
            b.frame_tl_in_y = (border_in_y - frame_offset).max(0);
            b.frame_tl_out_y = (b.frame_tl_in_y - b.frame_size).max(0);
            b.border_top = b.frame_tl_out_y;
            b.fl_top = b.frame_tl_in_y;
            b.border_left = b.frame_tl_out_x.clamp(0, out_w);
            b.fl_left = b.frame_tl_in_x.clamp(0, out_w);
            let frame_in_width = ((in_w as f32) + frame_offset as f32 * 2.0).floor() as i32;
            let frame_in_height = ((in_h as f32) + frame_offset as f32 * 2.0).floor() as i32;
            let frame_out_width = frame_in_width + b.frame_size * 2;
            let frame_out_height = frame_in_height + b.frame_size * 2;

            b.frame_br_in_x =
                (image_lx - frame_offset + frame_in_width - 1).clamp(0, out_w - 1);
            b.frame_br_in_y =
                (image_ty - frame_offset + frame_in_height - 1).clamp(0, out_h - 1);

            // If 100% frame_offset we ensure the frame line sticks to the
            // outer border (mirrors the C's two conditional assignments).
            b.frame_br_out_x = if f_offset == 1.0
                && (border_size_l.min(border_size_r) - border_min_size < 2)
            {
                out_w
            } else {
                (image_lx - frame_offset - b.frame_size + frame_out_width - 1).clamp(0, out_w - 1)
            };
            b.frame_br_out_y = if f_offset == 1.0
                && (border_size_t.min(border_size_b) - border_min_size < 2)
            {
                out_h
            } else {
                (image_ty - frame_offset - b.frame_size + frame_out_height - 1).clamp(0, out_h - 1)
            };

            // Need end+1 for these coordinates.
            b.fl_right = b.frame_br_in_x;
            b.border_right = b.frame_br_out_x;
            b.fl_bot = b.frame_br_in_y;
            b.border_bot = b.frame_br_out_y;
        }
        b
    }

    /// Compose a packed linear RGBA `f32` buffer (`pixels`, `w × h`) with the
    /// `Framing` params: `modify_roi_out` → `setup_binfo` →
    /// [`copy_image_with_border_kernel`], returning `(ow, oh, pixels)` on the
    /// enlarged canvas. The pipeline never needs the re-routed `modify_roi_in`
    /// (whole-frame scale-1 input), so it is not ported.
    ///
    /// Identity (border `size == 0` — the C's constant-border no-op, which
    /// also zeroes `frame_size` since a zero border cannot host a frame) or a
    /// degenerate/malformed buffer returns the source dimensions and a
    /// verbatim copy, as does any non-finite parameter — geometry never fails
    /// a render.
    pub fn apply(
        pixels: &[f32],
        w: usize,
        h: usize,
        params: BordersParams,
    ) -> (usize, usize, Vec<f32>) {
        if w == 0
            || h == 0
            || pixels.len() < w.saturating_mul(h).saturating_mul(4)
            || params.size == 0.0
            || !params.is_finite()
        {
            return (w, h, pixels.to_vec());
        }
        let (ow, oh) = Self::modify_roi_out(w, h, params.size);
        let binfo = Self::setup_binfo(
            w as i32,
            h as i32,
            ow as i32,
            oh as i32,
            params.pos_v,
            params.pos_h,
            &params.color,
            &params.frame_color,
            params.frame_size,
            params.frame_offset,
        );
        let n = ow.saturating_mul(oh).saturating_mul(4);
        let mut out = vec![0.0f32; n];
        copy_image_with_border_kernel(&mut out, pixels, &binfo);
        (ow, oh, out)
    }

    /// Compose an **8-bit** interleaved image (`bytes`, `w × h`, `nch`
    /// channels, `rowstride` bytes per row) with the `Framing` params, for the
    /// non-raw preview / non-raw export paths that never see a linear float
    /// buffer. The output is tightly packed (`ow * nch` bytes per row); filled
    /// border pixels get the colour bytes and, for 4-channel buffers, alpha
    /// 255 (the C compose's `bcolor`/`flcolor` alpha = 1.0); image pixels are
    /// copied verbatim (alpha preserved). Returns `None` when the borders are
    /// off/empty or the buffer is malformed — the caller then keeps the
    /// original as-is.
    pub fn apply_srgb8(
        bytes: &[u8],
        w: usize,
        h: usize,
        nch: usize,
        rowstride: usize,
        params: BordersParams,
    ) -> Option<(usize, usize, Vec<u8>)> {
        if nch == 0 || w == 0 || h == 0 || params.size == 0.0 || !params.is_finite() {
            return None;
        }
        let packed_row = w.checked_mul(nch)?;
        let required = rowstride.checked_mul(h)?;
        if rowstride < packed_row || bytes.len() < required {
            return None;
        }
        let (ow, oh) = Self::modify_roi_out(w, h, params.size);
        let binfo = Self::setup_binfo(
            w as i32,
            h as i32,
            ow as i32,
            oh as i32,
            params.pos_v,
            params.pos_h,
            &params.color,
            &params.frame_color,
            params.frame_size,
            params.frame_offset,
        );
        let out_row = ow.checked_mul(nch)?;
        let total = oh.checked_mul(out_row)?;
        let mut out = vec![0u8; total];
        for row in 0..oh {
            let r = row as i64;
            let src_row = r - binfo.image_top as i64;
            for col in 0..ow {
                let c = col as i64;
                let o = row * out_row + col * nch;
                match classify_pixel(&binfo, r, c, &params.color, &params.frame_color) {
                    PixelClass::Fill(fill) => {
                        let n = nch.min(3);
                        let b = to_byte(&fill);
                        out[o..o + n].copy_from_slice(&b[..n]);
                        if nch == 4 {
                            out[o + 3] = 255;
                        }
                    }
                    PixelClass::Image => {
                        let sc = (c - binfo.image_left as i64) as usize;
                        let sr = src_row.max(0) as usize;
                        let src = sr * rowstride + sc * nch;
                        if src + nch <= bytes.len() {
                            out[o..o + nch].copy_from_slice(&bytes[src..src + nch]);
                        }
                    }
                }
            }
        }
        Some((ow, oh, out))
    }

    /// Compose an **16-bit** interleaved image (`samples: &[u16]`, `w × h`,
    /// `nch` channels, `rowstride` samples per row) with the `Framing` params —
    /// the lossless twin of [`Self::apply_srgb8`] for the PNG/TIFF export leg
    /// (a 16-bit source round-trips without an 8-bit detour, exactly like the
    /// orientation slice it sits next to in `render_nonraw_export`). Filled
    /// border pixels get `round(color · 65535)` and, for 4-channel buffers,
    /// alpha 65535; image pixels are copied verbatim. Returns `None` when the
    /// borders are off/empty or the buffer is malformed.
    pub fn apply_srgb16(
        samples: &[u16],
        w: usize,
        h: usize,
        nch: usize,
        rowstride: usize,
        params: BordersParams,
    ) -> Option<(usize, usize, Vec<u16>)> {
        if nch == 0 || w == 0 || h == 0 || params.size == 0.0 || !params.is_finite() {
            return None;
        }
        let packed_row = w.checked_mul(nch)?;
        let required = rowstride.checked_mul(h)?;
        if rowstride < packed_row || samples.len() < required {
            return None;
        }
        let (ow, oh) = Self::modify_roi_out(w, h, params.size);
        let binfo = Self::setup_binfo(
            w as i32,
            h as i32,
            ow as i32,
            oh as i32,
            params.pos_v,
            params.pos_h,
            &params.color,
            &params.frame_color,
            params.frame_size,
            params.frame_offset,
        );
        let out_row = ow.checked_mul(nch)?;
        let total = oh.checked_mul(out_row)?;
        let mut out = vec![0u16; total];
        for row in 0..oh {
            let r = row as i64;
            let src_row = r - binfo.image_top as i64;
            for col in 0..ow {
                let c = col as i64;
                let o = row * out_row + col * nch;
                match classify_pixel(&binfo, r, c, &params.color, &params.frame_color) {
                    PixelClass::Fill(fill) => {
                        let n = nch.min(3);
                        let v = to_u16(&fill);
                        out[o..o + n].copy_from_slice(&v[..n]);
                        if nch == 4 {
                            out[o + 3] = 65535;
                        }
                    }
                    PixelClass::Image => {
                        let sc = (c - binfo.image_left as i64) as usize;
                        let sr = src_row.max(0) as usize;
                        let src = sr * rowstride + sc * nch;
                        if src + nch <= samples.len() {
                            out[o..o + nch].copy_from_slice(&samples[src..src + nch]);
                        }
                    }
                }
            }
        }
        Some((ow, oh, out))
    }
}

/// Why a border-space output pixel is drawn the way it is.
#[derive(Clone, Copy)]
enum PixelClass {
    Fill([f32; 3]),
    Image,
}

/// Classify one output pixel against [`BorderPositions`] using the same
/// four-band, five-run layout as [`copy_image_with_border_kernel`]; `Fill`
/// carries the original float colour so both the 8-bit and 16-bit composes
/// can scale it once at fill time, `Image` means copy the source pixel. The
/// band precedence and run boundaries reproduce the kernel exactly, so all
/// the integer composes agree pixel-for-pixel for the same geometry.
fn classify_pixel(
    b: &BorderPositions,
    row: i64,
    col: i64,
    bcolour: &[f32; 3],
    flcolour: &[f32; 3],
) -> PixelClass {
    // The kernel derives its i64 bands by casting the i32 struct fields; do
    // the same here so the composes share one comparison lattice.
    let (border_top, fl_top, image_top) =
        (b.border_top as i64, b.fl_top as i64, b.image_top as i64);
    let (image_bot, fl_bot, border_bot) =
        (b.image_bot as i64, b.fl_bot as i64, b.border_bot as i64);
    let (border_left, fl_left, image_left) =
        (b.border_left as i64, b.fl_left as i64, b.image_left as i64);
    let (image_right, fl_right, border_right, width) =
        (b.image_right as i64, b.fl_right as i64, b.border_right as i64, b.width as i64);
    let bg = |c: &[f32; 3]| PixelClass::Fill(*c);
    if row < border_top || row >= border_bot {
        bg(bcolour)
    } else if row < fl_top || row >= fl_bot {
        bg(if col < border_left || col >= border_right { bcolour } else { flcolour })
    } else if row < image_top || row >= image_bot {
        bg(if col < border_left {
            bcolour
        } else if col < fl_left {
            flcolour
        } else if col < fl_right {
            bcolour
        } else if col < border_right {
            flcolour
        } else {
            bcolour
        })
    } else if col < border_left {
        bg(bcolour)
    } else if image_left > border_left && col >= border_left && col < fl_left {
        bg(flcolour)
    } else if image_left > border_left && col >= fl_left && col < image_left {
        bg(bcolour)
    } else if col >= image_left && col < image_right {
        PixelClass::Image
    } else if col < fl_right {
        bg(bcolour)
    } else if width > fl_right && col < border_right {
        bg(flcolour)
    } else {
        bg(bcolour)
    }
}

/// Scale a `[f32; 3]` colour in `[0, 1]` to `u8` bytes (clamped, with
/// half-away-from-zero rounding).
fn to_byte(color: &[f32; 3]) -> [u8; 3] {
    [
        (color[0] * 255.0).round().clamp(0.0, 255.0) as u8,
        (color[1] * 255.0).round().clamp(0.0, 255.0) as u8,
        (color[2] * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Scale a `[f32; 3]` colour in `[0, 1]` to `u16` samples (clamped, with
/// half-away-from-zero rounding), for the 16-bit compose.
fn to_u16(color: &[f32; 3]) -> [u16; 3] {
    [
        (color[0] * 65535.0).round().clamp(0.0, 65535.0) as u16,
        (color[1] * 65535.0).round().clamp(0.0, 65535.0) as u16,
        (color[2] * 65535.0).round().clamp(0.0, 65535.0) as u16,
    ]
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

    // ── Framing (Borders module) geometry ────────────────────────────────────

    /// Hand-checks `modify_roi_out`'s constant-border math against the C
    /// formulas. `basis` is AUTO → LONGER → width when `width > height`, else
    /// height; `basis_out = roundf(basis_in / (1 - size))`, the other side gets
    /// `other_in + basis_out - basis_in`; both CLAMP to [1, 3·max].
    #[test]
    fn borders_modify_roi_out_constant_path_golden() {
        // Square 100×100, size 0.1: base = height (100 not > 100),
        // roundf(100/0.9) = 111 both ways → 111×111.
        assert_eq!(Borders::modify_roi_out(100, 100, 0.1), (111, 111));
        // Landscape 100×50, size 0.1: base = width, roundf(100/0.9) = 111;
        // height = 50 + 111 − 100 = 61 → 111×61.
        assert_eq!(Borders::modify_roi_out(100, 50, 0.1), (111, 61));
        // Portrait 50×100: base = height, roundf(100/0.9) = 111;
        // width = 50 + 111 − 100 = 61 → 61×111.
        assert_eq!(Borders::modify_roi_out(50, 100, 0.1), (61, 111));
        // size 0 (identity) → exactly the input dims.
        assert_eq!(Borders::modify_roi_out(80, 60, 0.0), (80, 60));
        // Large size (0.5 cap) on 200×100: base = width, roundf(200/0.5)=400,
        // height = 100 + 400 − 200 = 300 → 400×300.
        assert_eq!(Borders::modify_roi_out(200, 100, 0.5), (400, 300));
        // Clamp: size 0.9 on 4×4 → base = roundf(4/0.1)=40, other=4+40−4=40,
        // both well under 3·max=12 after CLAMP → 12×12.
        assert_eq!(Borders::modify_roi_out(4, 4, 0.9), (12, 12));
    }

    /// Whole-frame centered `setup_binfo` on a square 100×100 → 111×111,
    /// pos 0.5: borders split t/l = trunc(11·0.5)=5, b/r = 6; image rect at
    /// (5,5) sized 100×100. No frame (frame_size 0).
    #[test]
    fn borders_setup_binfo_centered_golden() {
        let b = Borders::setup_binfo(
            100, 100, 111, 111, 0.5, 0.5, &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], 0.0, 0.5,
        );
        assert_eq!(b.width, 111);
        assert_eq!(b.height, 111);
        assert_eq!(b.stride, 100);
        assert_eq!(b.border_size_t, 5);
        assert_eq!(b.border_size_b, 6);
        assert_eq!(b.border_size_l, 5);
        assert_eq!(b.border_size_r, 6);
        // Image placed at (5,5), 100×100.
        assert_eq!((b.border_in_x, b.border_in_y), (5, 5));
        assert_eq!((b.image_left, b.image_top), (5, 5));
        assert_eq!((b.image_right, b.image_bot), (105, 105));
        // No frame: frame_size stays 0 and the frame sub-rectangle fields
        // remain the struct defaults (the C only writes them inside its
        // `frame_size > 0` branch; the compose reads the `border_/fl_/image_`
        // bands, whose values above are the frame-agnostic ones).
        assert_eq!(b.frame_size, 0);
        assert_eq!(b.frame_tl_out_y, 0);
        assert_eq!(b.frame_tl_out_x, 0);
        // Colours with opaque alpha.
        assert_eq!(b.bcolor, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.flcolor, [0.0, 0.0, 0.0, 1.0]);
    }

    /// The frame-line branch: on a 100×100 → 125×125 canvas with pos 0.5,
    /// size 0.2, border 12/12/13/13, border_min = 12, frame_size = 12·0.5 = 6
    /// → the top/left frame-line inner/outer bands at 9 and 3.
    #[test]
    fn borders_setup_binfo_frame_branch_golden() {
        // roundf(100/0.8) = 125 → 125×125.
        assert_eq!(Borders::modify_roi_out(100, 100, 0.2), (125, 125));
        let b = Borders::setup_binfo(
            100, 100, 125, 125, 0.5, 0.5, &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], 0.5, 0.5,
        );
        // border_size_t = trunc(25·0.5) = 12; sized 12/13.
        assert_eq!((b.border_size_t, b.border_size_b), (12, 13));
        assert_eq!((b.border_size_l, b.border_size_r), (12, 13));
        // border_min_size = 12 → frame_size = 6.
        assert_eq!(b.frame_size, 6);
        // frame_space = 12 − 6 = 6, frame_offset = trunc(6·0.5) = 3.
        // frame_tl_in_x = max(border_in_x=12 − 3, 0) = 9;
        // frame_tl_out_x = max(9 − 6, 0) = 3.
        assert_eq!(b.border_in_x, 12);
        assert_eq!(b.frame_tl_in_x, 9);
        assert_eq!(b.frame_tl_out_x, 3);
        // frame_in_width = floor(100 + 3·2) = 106;
        // frame_out_width = 106 + 12 = 118.
        // frame_br_in_x = clamp(12 − 3 + 106 − 1, 0, 124) = 114.
        assert_eq!(b.frame_br_in_x, 114);
    }

    /// pos extremes push the image to the frame edges: pos (0,0) leaves the
    /// image at the top-left with all border on the right/bottom.
    #[test]
    fn borders_setup_binfo_pos_extremes() {
        let b = Borders::setup_binfo(
            100, 100, 111, 111, 0.0, 0.0, &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], 0.0, 0.5,
        );
        assert_eq!(b.border_size_t, 0);
        assert_eq!(b.border_size_b, 11);
        assert_eq!(b.border_size_l, 0);
        assert_eq!(b.border_size_r, 11);
        // Image at the top-left corner.
        assert_eq!((b.border_in_x, b.border_in_y), (0, 0));
        assert_eq!((b.image_left, b.image_top), (0, 0));
        assert_eq!((b.image_right, b.image_bot), (100, 100));

        // pos (1,1): image at bottom-right, all border on top/left.
        let c = Borders::setup_binfo(
            100, 100, 111, 111, 1.0, 1.0, &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], 0.0, 0.5,
        );
        assert_eq!(c.border_size_t, 11);
        assert_eq!(c.border_size_b, 0);
        assert_eq!(c.border_size_l, 11);
        assert_eq!(c.border_size_r, 0);
        assert_eq!((c.border_in_x, c.border_in_y), (11, 11));
        assert_eq!((c.image_left, c.image_top), (11, 11));
    }

    /// `Borders::apply` on a uniform f32 RGBA field: the enlarged canvas has
    /// border-colour corners, the copied image region holds the source pixels
    /// at the exact source offsets (via a per-pixel reference), and the dims
    /// equal `modify_roi_out`.
    #[test]
    fn borders_apply_composes_field_with_border_corners() {
        let (w, h) = (12usize, 8usize);
        let mut px = vec![0.0f32; w * h * 4];
        for i in 0..h {
            for j in 0..w {
                let o = (i * w + j) * 4;
                px[o] = j as f32;
                px[o + 1] = i as f32;
                px[o + 2] = 0.25;
                px[o + 3] = 1.0;
            }
        }
        // size 0.2 on 12×8: basis = width (12 > 8), roundf(12/0.8) = 15 →
        // 15×11 (height = 8 + 15 − 12 = 11); border totals 3×3, pos 0.5 gives
        // l/t = 1, r = 2, b = 2.
        let params = BordersParams { color: [0.2, 0.3, 0.4], size: 0.2, ..Default::default() };
        let (ow, oh, out) = Borders::apply(&px, w, h, params);
        assert_eq!((ow, oh), Borders::modify_roi_out(w, h, 0.2));
        assert_eq!((ow, oh), (15, 11));
        // Build the reference with the same geometry.
        let binfo = Borders::setup_binfo(
            w as i32,
            h as i32,
            ow as i32,
            oh as i32,
            params.pos_v,
            params.pos_h,
            &params.color,
            &params.frame_color,
            params.frame_size,
            params.frame_offset,
        );
        let mut reference = vec![0.0f32; ow * oh * 4];
        ref_copy_image_with_border(&mut reference, &px, &binfo);
        assert_eq!(to_bits(&out), to_bits(&reference), "compose must match the per-pixel reference");
        // Top-left corner is border colour, opaque.
        let co = 0usize;
        assert_eq!(to_bits(&out[co..co + 4]), to_bits(&[0.2, 0.3, 0.4, 1.0]));
        // Centre pixel is a copied source pixel (image band, well inside).
        // With the 1-px border (image at (1,1)): output (cx,cy) ← source
        // (cx−1, cy−1).
        let (cx, cy) = (ow / 2, oh / 2);
        let cio = (cy * ow + cx) * 4;
        let si = ((cy - 1) * w + (cx - 1)) * 4;
        assert_eq!(
            to_bits(&out[cio..cio + 4]),
            to_bits(&px[si..si + 4]),
            "bordered pixel must equal the source pixel at its offset"
        );
        // A known image pixel: with 1-px border, output (5,4) ← source (4,3).
        let di = (4 * ow + 5) * 4;
        let si = (3 * w + 4) * 4;
        assert_eq!(to_bits(&out[di..di + 4]), to_bits(&px[si..si + 4]),
                   "bordered pixel must equal the source pixel at its offset");
    }

    /// Identity (size 0), malformed and non-finite inputs return a verbatim
    /// copy (or `None` for the 8-bit form) and never panic.
    #[test]
    fn borders_apply_identity_and_degenerate_inputs() {
        let px = vec![1.0f32; 8 * 8 * 4];
        // size 0 → verbatim, same dims.
        let identity = BordersParams { size: 0.0, ..Default::default() };
        let (w, h, out) = Borders::apply(&px, 8, 8, identity);
        assert_eq!((w, h), (8, 8));
        assert_eq!(out, px);
        // Non-finite size → verbatim (geometry no-op, like rotate).
        let nan = BordersParams { size: f32::NAN, ..Default::default() };
        let (w2, h2, out2) = Borders::apply(&px, 8, 8, nan);
        assert_eq!((w2, h2), (8, 8));
        assert_eq!(out2, px);
        // Short buffer claims 8x8 but only holds 3 pixels → verbatim copy.
        let short = vec![0.0f32; 12];
        let (w3, h3, out3) = Borders::apply(&short, 8, 8, Default::default());
        assert_eq!((w3, h3), (8, 8));
        assert_eq!(out3, short);
    }

    #[test]
    fn borders_apply_srgb8_identity_and_compose() {
        // Identity: size 0 → None (caller keeps the source; no copy needed).
        let identity = BordersParams { size: 0.0, ..Default::default() };
        let src = vec![0u8; 3 * 3 * 3];
        let o = Borders::apply_srgb8(&src, 3, 3, 3, 9, identity);
        assert!(o.is_none());
        // Malformed (short buffer) → None.
        assert!(Borders::apply_srgb8(&vec![0u8; 4], 3, 3, 3, 9, Default::default()).is_none());
        // Compose a uniform 3-colour 2x2 → 3x3 8-bit buffer; border pixels get
        // the colour bytes, image pixels are copied verbatim.
        let params = BordersParams { color: [1.0, 0.0, 0.0], size: 0.5, ..Default::default() };
        let (ow, oh) = Borders::modify_roi_out(2, 2, 0.5);
        // roundf(2/0.5) = 4 → 4×4.
        assert_eq!((ow, oh), (4, 4));
        let mut img = vec![0u8; 2 * 2 * 3];
        // Left half red-ish, right half green-ish, so copied region is
        // distinguishable.
        img[0..3].copy_from_slice(&[100, 200, 50]); // (0,0)
        img[3..6].copy_from_slice(&[110, 210, 60]); // (1,0)
        img[6..9].copy_from_slice(&[120, 220, 70]); // (0,1)
        img[9..12].copy_from_slice(&[130, 230, 80]); // (1,1)
        let (zw, zh, out) = Borders::apply_srgb8(&img, 2, 2, 3, 6, params).expect("compose");
        assert_eq!((zw, zh), (4, 4));
        assert_eq!(out.len(), 4 * 4 * 3);
        // pos 0.5 on 2x2 → 4x4: border_size each = trunc(2*0.5)=1. Image at
        // (1,1), 2x2 in a 4x4 out (border t/l 1, b/r 1).
        // Top-left corner = red border bytes.
        assert_eq!(&out[0..3], &[255, 0, 0]);
        // A copied image pixel at (2,2) maps to source (1,1) = [130,230,80].
        let di = (2 * 4 + 2) * 3;
        assert_eq!(&out[di..di + 3], &[130, 230, 80]);
    }

    #[test]
    fn borders_apply_srgb16_is_the_lossless_twin() {
        // Identity and malformed inputs → None, same contract as the 8-bit form.
        assert!(Borders::apply_srgb16(&vec![0u16; 4], 3, 3, 3, 9, Default::default()).is_none());
        let params = BordersParams {
            color: [1.0, 0.0, 0.0],
            size: 0.5,
            ..Default::default()
        };
        let mut img = vec![0u16; 2 * 2 * 3];
        for (i, v) in [100u16, 30000, 5000, 200, 30001, 6000, 300, 30002, 7000, 400, 30003, 8000]
            .into_iter()
            .enumerate()
        {
            img[i] = v;
        }
        let (ow, oh, out) = Borders::apply_srgb16(&img, 2, 2, 3, 6, params).expect("compose");
        // Same geometry as the 8-bit test: 2x2 → 4x4, image at (1,1).
        assert_eq!((ow, oh), (4, 4));
        // Corner = full-scale red (65535, 0, 0), not a quantized 8-bit value.
        assert_eq!(&out[0..3], &[65535, 0, 0]);
        // Copied image pixel (2,2) → source (1,1) verbatim at 16-bit.
        let di = (2 * 4 + 2) * 3;
        assert_eq!(&out[di..di + 3], &[400, 30003, 8000]);
        // Interior copy is bit-exact (16→16 needs no requantisation).
        assert_eq!(out, {
            let mut expect = vec![0u16; 4 * 4 * 3];
            for row in 1..3 {
                for col in 1..3 {
                    let s = ((row - 1) * 2 + (col - 1)) * 3;
                    let d = (row * 4 + col) * 3;
                    expect[d..d + 3].copy_from_slice(&img[s..s + 3]);
                }
            }
            for row in 0..4 {
                for col in 0..4 {
                    if row == 0 || row == 3 || col == 0 || col == 3 {
                        let d = (row * 4 + col) * 3;
                        expect[d..d + 3].copy_from_slice(&[65535, 0, 0]);
                    }
                }
            }
            expect
        });
    }

    /// `Borders::apply` on an RGBA buffer must agree bit-exactly with the
    /// manual `copy_image_with_border_kernel` reference for the same geometry,
    /// across a couple of param sets (with and without a frame).
    #[test]
    fn borders_apply_matches_the_kernel_reference() {
        let (w, h) = (10usize, 6usize);
        let mut px = vec![0.0f32; w * h * 4];
        lcg_fill(&mut px, 0x5A17);
        for params in [
            BordersParams { color: [0.2, 0.3, 0.4], size: 0.25, pos_h: 0.4, pos_v: 0.6, ..Default::default() },
            BordersParams {
                color: [0.9, 0.1, 0.5],
                size: 0.2,
                frame_size: 0.5,
                frame_offset: 0.5,
                frame_color: [0.0, 1.0, 0.25],
                ..Default::default()
            },
        ] {
            let (ow, oh, direct) = Borders::apply(&px, w, h, params);
            let binfo = Borders::setup_binfo(
                w as i32,
                h as i32,
                ow as i32,
                oh as i32,
                params.pos_v,
                params.pos_h,
                &params.color,
                &params.frame_color,
                params.frame_size,
                params.frame_offset,
            );
            let mut reference = vec![0.0f32; ow * oh * 4];
            ref_copy_image_with_border(&mut reference, &px, &binfo);
            assert_eq!(to_bits(&direct), to_bits(&reference));
        }
    }

    /// The composed canvas keeps the border colour on the border runs, the
    /// source pixels intact inside the image band, and — when the frame-line
    /// branch fires — the frame colour in its ring, on a hand-computed
    /// geometry (10×10, size 0.2 → 13×13; border t/b/l/r = 1/2/1/2,
    /// border_min = 1, frame_size = 1, frame_offset = 0 — frame ring at the
    /// very top/left edge, row 0 lights the frame colour everywhere).
    #[test]
    fn borders_apply_frame_line_golden() {
        let (w, h) = (10usize, 10usize);
        let mut px = vec![0.0f32; w * h * 4];
        for i in 0..h {
            for j in 0..w {
                let o = (i * w + j) * 4;
                px[o] = 0.5;
                px[o + 1] = 0.5;
                px[o + 2] = 0.5;
                px[o + 3] = 1.0;
            }
        }
        let params = BordersParams {
            color: [0.0, 0.0, 0.0], // black border
            size: 0.2,
            frame_size: 1.0,       // frame line occupies the whole border
            frame_color: [1.0, 1.0, 1.0], // white frame line
            ..Default::default()
        };
        let (ow, oh, out) = Borders::apply(&px, w, h, params);
        assert_eq!((ow, oh), (13, 13));
        // Row 0 is the frameline band (frame_tl_out = 0): fully white.
        assert_eq!(to_bits(&out[0..4]), to_bits(&[1.0, 1.0, 1.0, 1.0]));
        // (1,1) is an image pixel copy (grey).
        let p = (1 * ow + 1) * 4;
        assert_eq!(to_bits(&out[p..p + 4]), to_bits(&[0.5, 0.5, 0.5, 1.0]));
        // Bottom-right corner (12,12): outer border band → black.
        let p = (12 * ow + 12) * 4;
        assert_eq!(to_bits(&out[p..p + 4]), to_bits(&[0.0, 0.0, 0.0, 1.0]));
        // And the last pure-border row (12, 0): black border again.
        let p = (12 * ow + 0) * 4;
        assert_eq!(to_bits(&out[p..p + 4]), to_bits(&[0.0, 0.0, 0.0, 1.0]));
    }
}
