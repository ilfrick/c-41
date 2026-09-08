//! Kernels ported from `src/common/heal.c` (`_heal_sub`, m4-177;
//! `_heal_add`, heal.c:59-94, m4-178).
//!
//! What the functions do: `_heal_sub` subtracts the reference pattern
//! (`bottom_buffer`) from the sample image (`top_buffer`) pixel by pixel and
//! stores the difference split by red/black checkerboard colour into two
//! contiguous buffers, ready for the Gauss-Seidel Laplace solver
//! (`_heal_laplace_loop`). `_heal_add` is the inverse: it re-interleaves the
//! solved red/black buffers and adds the reference image (`second_buffer`)
//! back pixel by pixel, storing the healed image in `result_buffer`. It reads
//! split rows at the same padding-aware `(row + 1) * res_stride` offsets,
//! walks each row pairwise (one red and one black pixel at a time, `buf1`
//! holding the left-most pixel's colour), takes the left-over pixel on odd
//! widths from `buf1`, and writes all four channels including alpha —
//! matching the default vectorised C build (see the channel contract below).
//! Both split buffers carry one padding row above and one below the image
//! rows (boundary conditions for the solver), cleared to zero here.
//!
//! Buffer layout: with `res_stride = 4 * ((width + 1) / 2)` floats per row,
//! each of `red_buffer`/`black_buffer` holds `(height + 2) * res_stride`
//! floats. Image row `r` lives at `buf[(r + 1) * res_stride ..]`, leaving
//! rows 0 and `height + 1` as zero padding. Pixel `(r, c)` has checker colour
//! red when `(r + c)` is odd, black when even; within its row it occupies
//! slot `c / 2` (4 floats) of the same-colour buffer, i.e. pixels `(r, 2k)`
//! and `(r, 2k + 1)` form a pair with one pixel in each buffer. Which buffer
//! is "first" alternates per row: even rows store the even pixel in black
//! and the odd pixel in red, odd rows the reverse — so the left-most pixel
//! always lands in `buf1` (black on even rows, red on odd rows). For odd
//! widths the left-over right-most pixel goes into `buf1`; the
//! opposite-colour tail slot is zeroed by `heal_sub`, while `heal_add` never
//! reads it.
//!
//! Channel contract: the C loop body is `for_each_channel`, which iterates
//! over `DT_PIXEL_SIMD_CHANNELS` — 4 in the default (vectorised) build
//! (`DT_NO_VECTORIZATION` is commented out in `src/common/darktable.h`), so
//! all four channels including alpha are subtracted. The kernel below
//! processes exactly channels 0..4 to match the production build. Under a
//! non-default `DT_NO_VECTORIZATION` build, the C loop would write only three
//! channels and leave alpha untouched; the Rust kernel would still write it.
//!
//! Exactness notes:
//! - The only arithmetic is one f32 addition/subtraction per channel per
//!   kernel — a single IEEE-754 operation with no reassociation. That makes
//!   the Rust kernels equivalent by construction to the former single-threaded
//!   C loops; the `to_bits` comparisons pin each kernel against its
//!   independent Rust reference, not against a retained C golden dump.
//! - The C loops were `DT_OMP_FOR` over rows, but rows write disjoint regions
//!   (distinct `row_start` offsets in each buffer); `heal_sub`'s padding
//!   clears also run after its loop (`heal_add` has no clears), so each
//!   sequential port is identical for any thread count.
//!
//! Degenerate/short buffers: zero `width`/`height`, a dimension arithmetic
//! overflow, or a buffer shorter than the C caller allocates
//! (`4*width*height` floats for the image buffers,
//! `res_stride*(height + 2)` for the split buffers) is a no-op — no panic,
//! no out-of-bounds access. Well-formed callers (like `dt_heal`, which
//! allocates exactly those sizes) never hit this path.

/// Split-subtract the reference pattern from the sample image by checker colour.
///
/// Port of the former element-wise loop plus padding clears in
/// `_heal_sub` (heal.c:50-92). `top_buffer`/`bottom_buffer` hold
/// `4*width*height` floats; `red_buffer`/`black_buffer` each hold
/// `4*((width+1)/2)*(height+2)` floats. See the module docs for the layout,
/// the 4-channel contract, and the degenerate-input behaviour.
pub fn heal_sub(
    top_buffer: &[f32],
    bottom_buffer: &[f32],
    red_buffer: &mut [f32],
    black_buffer: &mut [f32],
    width: usize,
    height: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    let Some(sum) = width.checked_add(1) else {
        return;
    };
    let half = sum / 2;
    let (Some(res_stride), Some(src_len)) = (
        half.checked_mul(4),
        4usize
            .checked_mul(width)
            .and_then(|p| p.checked_mul(height)),
    ) else {
        return;
    };
    let Some(out_len) = height
        .checked_add(2)
        .and_then(|h| res_stride.checked_mul(h))
    else {
        return;
    };
    if top_buffer.len() < src_len
        || bottom_buffer.len() < src_len
        || red_buffer.len() < out_len
        || black_buffer.len() < out_len
    {
        return;
    }
    for row in 0..height {
        let parity = row & 1;
        let row_start = (row + 1) * res_stride;
        // buf1 holds the left-most pixel's colour (black on even rows, red
        // on odd rows), buf2 the opposite colour — exactly the C ternary.
        let (buf1, buf2): (&mut [f32], &mut [f32]) = if parity == 0 {
            (
                &mut black_buffer[row_start..],
                &mut red_buffer[row_start..],
            )
        } else {
            (
                &mut red_buffer[row_start..],
                &mut black_buffer[row_start..],
            )
        };
        // handle the pixels of the row pairwise, one red and one black at a time
        for col in 0..width / 2 {
            let idx = 4 * (row * width + 2 * col);
            for c in 0..4 {
                buf1[4 * col + c] = top_buffer[idx + c] - bottom_buffer[idx + c];
                buf2[4 * col + c] = top_buffer[idx + 4 + c] - bottom_buffer[idx + 4 + c];
            }
        }
        if (width & 1) == 1 {
            // Handle the left-over pixel on odd widths. Its colour is always
            // the same as the left-most pixel, so it goes into buf1; the
            // buf2 tail slot is zeroed.
            let res_idx = (width - 1) / 2;
            let idx = 4 * (row * width + (width - 1));
            for c in 0..4 {
                buf1[4 * res_idx + c] = top_buffer[idx + c] - bottom_buffer[idx + c];
                buf2[4 * res_idx + c] = 0.0;
            }
        }
    }
    // clear the top and bottom rows, used for padding
    red_buffer[..res_stride].fill(0.0);
    black_buffer[..res_stride].fill(0.0);
    red_buffer[(height + 1) * res_stride..out_len].fill(0.0);
    black_buffer[(height + 1) * res_stride..out_len].fill(0.0);
}

/// Re-interleave the solved red/black buffers and add the reference image.
///
/// Port of the former element-wise loop in `_heal_add` (heal.c:59-94).
/// `red_buffer`/`black_buffer` each hold `4*((width+1)/2)*(height+2)` floats
/// with image row `r` at `buf[(r + 1) * res_stride ..]` (padding rows are
/// never read); `second_buffer`/`result_buffer` each hold `4*width*height`
/// floats. Pixel `(r, c)` is read from slot `c / 2` of the buffer matching
/// the `(r + c)` checker parity (odd = red), i.e. each row is walked pairwise
/// with `buf1` holding the left-most pixel's colour (black on even rows, red
/// on odd rows), and the left-over pixel on odd widths comes from `buf1`
/// (the opposite-colour tail slot is never read). All four channels including
/// alpha are added, matching the default vectorised build. Degenerate/short
/// inputs are a no-op exactly as documented for [`heal_sub`].
pub fn heal_add(
    red_buffer: &[f32],
    black_buffer: &[f32],
    second_buffer: &[f32],
    result_buffer: &mut [f32],
    width: usize,
    height: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    let Some(sum) = width.checked_add(1) else {
        return;
    };
    let half = sum / 2;
    let (Some(res_stride), Some(src_len)) = (
        half.checked_mul(4),
        4usize
            .checked_mul(width)
            .and_then(|p| p.checked_mul(height)),
    ) else {
        return;
    };
    let Some(split_len) = height
        .checked_add(2)
        .and_then(|h| res_stride.checked_mul(h))
    else {
        return;
    };
    if red_buffer.len() < split_len
        || black_buffer.len() < split_len
        || second_buffer.len() < src_len
        || result_buffer.len() < src_len
    {
        return;
    }
    for row in 0..height {
        let parity = row & 1;
        let row_start = (row + 1) * res_stride;
        // buf1 holds the left-most pixel's colour (black on even rows, red
        // on odd rows), buf2 the opposite colour — exactly the C ternary.
        let (buf1, buf2): (&[f32], &[f32]) = if parity == 0 {
            (&black_buffer[row_start..], &red_buffer[row_start..])
        } else {
            (&red_buffer[row_start..], &black_buffer[row_start..])
        };
        // handle the pixels of the row pairwise, one red and one black at a time
        for col in 0..width / 2 {
            let idx = 4 * (row * width + 2 * col);
            for c in 0..4 {
                result_buffer[idx + c] = buf1[4 * col + c] + second_buffer[idx + c];
                result_buffer[idx + 4 + c] = buf2[4 * col + c] + second_buffer[idx + 4 + c];
            }
        }
        if (width & 1) == 1 {
            // Handle the left-over pixel on odd widths. Its colour is always
            // the same as the left-most pixel, so it comes from buf1; the
            // buf2 tail slot is never read.
            let res_idx = (width - 1) / 2;
            let idx = 4 * (row * width + (width - 1));
            for c in 0..4 {
                result_buffer[idx + c] = buf1[4 * res_idx + c] + second_buffer[idx + c];
            }
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `heal_sub`: walks every pixel
/// individually deriving its checker colour from `(row + col)` parity (where
/// the kernel walks pairs through alternating `buf1`/`buf2` borrows), zeroes
/// the odd-width opposite-colour tail slot in a separate step, and clears
/// the padding rows with explicit per-element loops (where the kernel uses
/// slice `fill`). Same no-op preconditions: well-formed buffers only.
#[allow(dead_code)]
fn ref_heal_sub(
    top_buffer: &[f32],
    bottom_buffer: &[f32],
    red_buffer: &mut [f32],
    black_buffer: &mut [f32],
    width: usize,
    height: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    let Some(sum) = width.checked_add(1) else {
        return;
    };
    let half = sum / 2;
    let (Some(res_stride), Some(src_len)) = (
        half.checked_mul(4),
        4usize
            .checked_mul(width)
            .and_then(|product| product.checked_mul(height)),
    ) else {
        return;
    };
    let Some(out_len) = height
        .checked_add(2)
        .and_then(|padded| res_stride.checked_mul(padded))
    else {
        return;
    };
    if top_buffer.len() < src_len
        || bottom_buffer.len() < src_len
        || red_buffer.len() < out_len
        || black_buffer.len() < out_len
    {
        return;
    }
    for row in 0..height {
        for col in 0..width {
            let src = 4 * (row * width + col);
            let dst = (row + 1) * res_stride + 4 * (col / 2);
            let is_red = ((row + col) & 1) == 1;
            for c in 0..4 {
                let d = top_buffer[src + c] - bottom_buffer[src + c];
                if is_red {
                    red_buffer[dst + c] = d;
                } else {
                    black_buffer[dst + c] = d;
                }
            }
        }
        if (width & 1) == 1 {
            // zero the opposite-colour tail slot beside the left-over pixel
            // (the pixel itself was stored above under its own colour).
            let dst = (row + 1) * res_stride + 4 * ((width - 1) / 2);
            let tail_is_red = ((row + (width - 1)) & 1) == 1;
            for c in 0..4 {
                if tail_is_red {
                    black_buffer[dst + c] = 0.0;
                } else {
                    red_buffer[dst + c] = 0.0;
                }
            }
        }
    }
    for i in 0..res_stride {
        red_buffer[i] = 0.0;
        black_buffer[i] = 0.0;
    }
    for i in (height + 1) * res_stride..out_len {
        red_buffer[i] = 0.0;
        black_buffer[i] = 0.0;
    }
}

/// Structurally divergent reference for `heal_add`: walks every pixel
/// individually deriving its checker colour from `(row + col)` parity (where
/// the kernel walks pairs through alternating `buf1`/`buf2` borrows). Same
/// no-op preconditions: well-formed buffers only.
#[allow(dead_code)]
fn ref_heal_add(
    red_buffer: &[f32],
    black_buffer: &[f32],
    second_buffer: &[f32],
    result_buffer: &mut [f32],
    width: usize,
    height: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    let Some(sum) = width.checked_add(1) else {
        return;
    };
    let half = sum / 2;
    let (Some(res_stride), Some(src_len)) = (
        half.checked_mul(4),
        4usize
            .checked_mul(width)
            .and_then(|product| product.checked_mul(height)),
    ) else {
        return;
    };
    let Some(split_len) = height
        .checked_add(2)
        .and_then(|padded| res_stride.checked_mul(padded))
    else {
        return;
    };
    if red_buffer.len() < split_len
        || black_buffer.len() < split_len
        || second_buffer.len() < src_len
        || result_buffer.len() < src_len
    {
        return;
    }
    for row in 0..height {
        for col in 0..width {
            let src = 4 * (row * width + col);
            let dst = (row + 1) * res_stride + 4 * (col / 2);
            let is_red = ((row + col) & 1) == 1;
            for c in 0..4 {
                let s = if is_red {
                    red_buffer[dst + c]
                } else {
                    black_buffer[dst + c]
                };
                result_buffer[src + c] = s + second_buffer[src + c];
            }
        }
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `top_buffer`/`bottom_buffer` must hold at least `4*width*height` floats
/// and `red_buffer`/`black_buffer` at least `4*((width+1)/2)*(height+2)`
/// floats (the `dt_heal` caller allocates exactly that).
#[no_mangle]
pub unsafe extern "C" fn darkroom_heal_sub(
    top_buffer: *const f32,
    bottom_buffer: *const f32,
    red_buffer: *mut f32,
    black_buffer: *mut f32,
    width: usize,
    height: usize,
) {
    if top_buffer.is_null()
        || bottom_buffer.is_null()
        || red_buffer.is_null()
        || black_buffer.is_null()
        || width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
    {
        return;
    }
    // with width, height <= i32::MAX the products below still need checking
    // (4*width*height can overflow usize in theory); validate BEFORE
    // constructing the slices (the safe kernel re-checks defensively)
    let half = (width + 1) / 2;
    let Some(res_stride) = half.checked_mul(4) else {
        return;
    };
    let Some(src_len) = 4usize
        .checked_mul(width)
        .and_then(|p| p.checked_mul(height))
    else {
        return;
    };
    let Some(out_len) = height
        .checked_add(2)
        .and_then(|h| res_stride.checked_mul(h))
    else {
        return;
    };
    let top_buffer = std::slice::from_raw_parts(top_buffer, src_len);
    let bottom_buffer = std::slice::from_raw_parts(bottom_buffer, src_len);
    let red_buffer = std::slice::from_raw_parts_mut(red_buffer, out_len);
    let black_buffer = std::slice::from_raw_parts_mut(black_buffer, out_len);
    heal_sub(
        top_buffer,
        bottom_buffer,
        red_buffer,
        black_buffer,
        width,
        height,
    );
}

/// # Safety
/// `red_buffer`/`black_buffer` must hold at least
/// `4*((width+1)/2)*(height+2)` floats and `second_buffer`/`result_buffer` at
/// least `4*width*height` floats (the `dt_heal` caller allocates exactly
/// that).
#[no_mangle]
pub unsafe extern "C" fn darkroom_heal_add(
    red_buffer: *const f32,
    black_buffer: *const f32,
    second_buffer: *const f32,
    result_buffer: *mut f32,
    width: usize,
    height: usize,
) {
    if red_buffer.is_null()
        || black_buffer.is_null()
        || second_buffer.is_null()
        || result_buffer.is_null()
        || width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
    {
        return;
    }
    // with width, height <= i32::MAX the products below still need checking
    // (4*width*height can overflow usize in theory); validate BEFORE
    // constructing the slices (the safe kernel re-checks defensively)
    let half = (width + 1) / 2;
    let Some(res_stride) = half.checked_mul(4) else {
        return;
    };
    let Some(src_len) = 4usize
        .checked_mul(width)
        .and_then(|p| p.checked_mul(height))
    else {
        return;
    };
    let Some(split_len) = height
        .checked_add(2)
        .and_then(|h| res_stride.checked_mul(h))
    else {
        return;
    };
    let red_buffer = std::slice::from_raw_parts(red_buffer, split_len);
    let black_buffer = std::slice::from_raw_parts(black_buffer, split_len);
    let second_buffer = std::slice::from_raw_parts(second_buffer, src_len);
    let result_buffer = std::slice::from_raw_parts_mut(result_buffer, src_len);
    heal_add(
        red_buffer,
        black_buffer,
        second_buffer,
        result_buffer,
        width,
        height,
    );
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    /// Sentinel far outside any test value range (LCG fills stay in
    /// `[0, scale)` with small scales, diffs within `(-scale, scale)`) to
    /// prove every output slot is written.
    const SENTINEL: f32 = 1.0e10;

    fn buffers(width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let src_len = 4 * width * height;
        let out_len = 4 * ((width + 1) / 2) * (height + 2);
        (vec![0.0; src_len], vec![0.0; src_len], vec![0.0; out_len], vec![0.0; out_len])
    }

    #[test]
    fn even_width_pin() {
        // width=2, height=2, bottom=0 so diffs equal the top values 1..=16.
        // Row 0 (even): even pixel -> black, odd pixel -> red.
        // Row 1 (odd):  even pixel -> red,   odd pixel -> black.
        let (mut top, mut bottom, mut red, mut black) = buffers(2, 2);
        top.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (i + 1) as f32);
        bottom.fill(0.0);
        heal_sub(&top, &bottom, &mut red, &mut black, 2, 2);
        assert_eq!(
            red,
            vec![0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            black,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 13.0, 14.0, 15.0, 16.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn odd_width_tail_pin() {
        // width=3, height=1: pair (pixels 0,1) split black/red on the even
        // row; left-over pixel 2 (even => black) into buf1, red tail zeroed.
        let (mut top, mut bottom, mut red, mut black) = buffers(3, 1);
        top.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (i + 1) as f32);
        bottom.fill(0.0);
        heal_sub(&top, &bottom, &mut red, &mut black, 3, 1);
        assert_eq!(
            red,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // padding row 0
                5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, // row 0: pixel 1, zero tail
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // padding row 2
            ]
        );
        assert_eq!(
            black,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // padding row 0
                1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, // row 0: pixels 0, 2
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // padding row 2
            ]
        );
    }

    #[test]
    fn odd_width_tail_odd_row_pin() {
        // width=3, height=2: row 1 is odd, so the left-over pixel 2 lands in
        // red (buf1) and the black tail slot is zeroed — the mirror image of
        // the even-row case above.
        let (mut top, mut bottom, mut red, mut black) = buffers(3, 2);
        top.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (i + 1) as f32);
        bottom.fill(1.0);
        heal_sub(&top, &bottom, &mut red, &mut black, 3, 2);
        // row 1 source pixels: 13..=24 minus 1 => 12..=23.
        // pair: pixel (1,0)=12..15 -> red (buf1), pixel (1,1)=16..19 -> black.
        // tail: pixel (1,2)=20..23 -> red, black tail zeroed.
        let stride = 8;
        let r1 = (1 + 1) * stride;
        assert_eq!(&red[r1..r1 + 4], &[12.0, 13.0, 14.0, 15.0]);
        assert_eq!(&red[r1 + 4..r1 + 8], &[20.0, 21.0, 22.0, 23.0]);
        assert_eq!(&black[r1..r1 + 4], &[16.0, 17.0, 18.0, 19.0]);
        assert_eq!(&black[r1 + 4..r1 + 8], &[0.0; 4]);
    }

    #[test]
    fn checker_parity_full_coverage() {
        // every pixel of a 5x4 image must land in exactly the buffer dictated
        // by (row + col) parity, and no sentinel may survive anywhere: data
        // rows are fully written (pairs + zeroed odd tail) and padding rows
        // are cleared.
        let (w, h) = (5usize, 4usize);
        let stride = 4 * ((w + 1) / 2);
        let (mut top, mut bottom, mut red, mut black) = buffers(w, h);
        lcg_fill(&mut top, 0xBEA1, 10.0);
        lcg_fill(&mut bottom, 0xBEA2, 10.0);
        red.fill(SENTINEL);
        black.fill(SENTINEL);
        heal_sub(&top, &bottom, &mut red, &mut black, w, h);
        assert!(!red.iter().any(|&v| v == SENTINEL));
        assert!(!black.iter().any(|&v| v == SENTINEL));
        for row in 0..h {
            for col in 0..w {
                let src = 4 * (row * w + col);
                let dst = (row + 1) * stride + 4 * (col / 2);
                for c in 0..4 {
                    let expect = top[src + c] - bottom[src + c];
                    if ((row + col) & 1) == 1 {
                        assert_eq!(red[dst + c], expect, "r={row} c={col} ch={c}");
                    } else {
                        assert_eq!(black[dst + c], expect, "r={row} c={col} ch={c}");
                    }
                }
            }
        }
        // padding rows are zero in both buffers
        assert_eq!(&red[..stride], &vec![0.0; stride][..]);
        assert_eq!(&red[(h + 1) * stride..], &vec![0.0; stride][..]);
        assert_eq!(&black[..stride], &vec![0.0; stride][..]);
        assert_eq!(&black[(h + 1) * stride..], &vec![0.0; stride][..]);
    }

    #[test]
    fn padding_rows_zeroed_over_sentinel() {
        // prefill with sentinel: padding rows must come back exactly zero
        // while data rows hold real diffs.
        let (w, h) = (4usize, 3usize);
        let stride = 4 * ((w + 1) / 2);
        let (mut top, mut bottom, mut red, mut black) = buffers(w, h);
        lcg_fill(&mut top, 0x6AD1, 5.0);
        lcg_fill(&mut bottom, 0x6AD2, 5.0);
        red.fill(SENTINEL);
        black.fill(SENTINEL);
        heal_sub(&top, &bottom, &mut red, &mut black, w, h);
        for buf in [&red[..], &black[..]] {
            assert!(buf[..stride].iter().all(|&v| v == 0.0));
            assert!(buf[(h + 1) * stride..].iter().all(|&v| v == 0.0));
            assert!(buf[stride..(h + 1) * stride].iter().all(|&v| v != SENTINEL));
        }
    }

    #[test]
    fn matches_reference_over_lcg() {
        // even/odd widths, small and larger heights, exact and to_bits
        // equality against the structurally divergent reference.
        for &(w, h) in &[
            (1usize, 1usize),
            (2, 1),
            (1, 3),
            (2, 2),
            (3, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (7, 5),
            (8, 8),
            (13, 7),
            (16, 9),
        ] {
            let (mut top, mut bottom, _, _) = buffers(w, h);
            lcg_fill(&mut top, 0x600D + w as u32, 10.0);
            lcg_fill(&mut bottom, 0x600E + h as u32, 10.0);
            let out_len = 4 * ((w + 1) / 2) * (h + 2);
            let mut direct_r = vec![SENTINEL; out_len];
            let mut direct_b = vec![SENTINEL; out_len];
            let mut ref_r = vec![SENTINEL; out_len];
            let mut ref_b = vec![SENTINEL; out_len];
            heal_sub(&top, &bottom, &mut direct_r, &mut direct_b, w, h);
            ref_heal_sub(&top, &bottom, &mut ref_r, &mut ref_b, w, h);
            assert_eq!(direct_r, ref_r, "red w={w} h={h}");
            assert_eq!(direct_b, ref_b, "black w={w} h={h}");
            for (d, r) in direct_r.iter().zip(ref_r.iter()) {
                assert_eq!(d.to_bits(), r.to_bits(), "red bits w={w} h={h}");
            }
            for (d, r) in direct_b.iter().zip(ref_b.iter()) {
                assert_eq!(d.to_bits(), r.to_bits(), "black bits w={w} h={h}");
            }
        }
    }

    #[test]
    fn nan_propagates_identically() {
        // subtraction is a single IEEE op in both kernel and reference:
        // NaN payloads must agree bit-for-bit.
        let (w, h) = (3usize, 2usize);
        let (mut top, mut bottom, _, _) = buffers(w, h);
        lcg_fill(&mut top, 0x6A01, 4.0);
        lcg_fill(&mut bottom, 0x6A02, 4.0);
        top[0] = f32::from_bits(0x7FC1_2345);
        bottom[5] = f32::from_bits(0xFFC0_00AB);
        let out_len = 4 * ((w + 1) / 2) * (h + 2);
        let mut direct_r = vec![0.0; out_len];
        let mut direct_b = vec![0.0; out_len];
        let mut ref_r = vec![0.0; out_len];
        let mut ref_b = vec![0.0; out_len];
        heal_sub(&top, &bottom, &mut direct_r, &mut direct_b, w, h);
        ref_heal_sub(&top, &bottom, &mut ref_r, &mut ref_b, w, h);
        // NOTE: no assert_eq! here — the buffers intentionally contain NaN
        // (NaN != NaN); bit patterns are compared below instead.
        for (d, r) in direct_r.iter().zip(ref_r.iter()) {
            assert_eq!(d.to_bits(), r.to_bits());
        }
        for (d, r) in direct_b.iter().zip(ref_b.iter()) {
            assert_eq!(d.to_bits(), r.to_bits());
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        // zero dims leave buffers untouched
        for &(w, h) in &[(0usize, 4usize), (4usize, 0usize), (0usize, 0usize)] {
            let (top, bottom, mut red, mut black) = buffers(4, 4);
            red.fill(9.0);
            black.fill(9.0);
            heal_sub(&top, &bottom, &mut red, &mut black, w, h);
            assert!(red.iter().all(|&v| v == 9.0), "w={w} h={h}");
            assert!(black.iter().all(|&v| v == 9.0), "w={w} h={h}");
        }
        // short buffers: clamped precondition fails -> no-op, no panic
        let (mut top, mut bottom, mut red, mut black) = buffers(4, 4);
        lcg_fill(&mut top, 0x5E07, 2.0);
        lcg_fill(&mut bottom, 0x5E08, 2.0);
        red.fill(9.0);
        black.fill(9.0);
        heal_sub(&top[..10], &bottom, &mut red, &mut black, 4, 4);
        heal_sub(&top, &bottom[..10], &mut red, &mut black, 4, 4);
        heal_sub(&top, &bottom, &mut red[..10], &mut black, 4, 4);
        heal_sub(&top, &bottom, &mut red, &mut black[..10], 4, 4);
        assert!(red.iter().all(|&v| v == 9.0));
        assert!(black.iter().all(|&v| v == 9.0));
        // boundary dimension arithmetic must no-op without panicking, even though
        // the undersized buffers would otherwise be invalid.
        heal_sub(&[], &[], &mut [], &mut [], usize::MAX, 1);
        ref_heal_sub(&[], &[], &mut [], &mut [], usize::MAX, 1);
    }

    #[test]
    fn ffi_round_trip() {
        for &(w, h) in &[(4usize, 2usize), (5usize, 3usize)] {
            let (mut top, mut bottom, _, _) = buffers(w, h);
            lcg_fill(&mut top, 0xFF11 + w as u32, 8.0);
            lcg_fill(&mut bottom, 0xFF12 + h as u32, 8.0);
            let out_len = 4 * ((w + 1) / 2) * (h + 2);
            let mut ffi_r = vec![0.0; out_len];
            let mut ffi_b = vec![0.0; out_len];
            let mut direct_r = vec![0.0; out_len];
            let mut direct_b = vec![0.0; out_len];
            unsafe {
                darkroom_heal_sub(
                    top.as_ptr(),
                    bottom.as_ptr(),
                    ffi_r.as_mut_ptr(),
                    ffi_b.as_mut_ptr(),
                    w,
                    h,
                );
            }
            heal_sub(&top, &bottom, &mut direct_r, &mut direct_b, w, h);
            assert_eq!(ffi_r, direct_r, "red w={w} h={h}");
            assert_eq!(ffi_b, direct_b, "black w={w} h={h}");
        }
    }

    #[test]
    fn ffi_guards() {
        let (top, bottom, mut red, mut black) = buffers(4, 4);
        red.fill(7.0);
        black.fill(7.0);
        unsafe {
            // null pointers
            darkroom_heal_sub(
                std::ptr::null(),
                bottom.as_ptr(),
                red.as_mut_ptr(),
                black.as_mut_ptr(),
                4,
                4,
            );
            darkroom_heal_sub(
                top.as_ptr(),
                std::ptr::null(),
                red.as_mut_ptr(),
                black.as_mut_ptr(),
                4,
                4,
            );
            darkroom_heal_sub(
                top.as_ptr(),
                bottom.as_ptr(),
                std::ptr::null_mut(),
                black.as_mut_ptr(),
                4,
                4,
            );
            darkroom_heal_sub(
                top.as_ptr(),
                bottom.as_ptr(),
                red.as_mut_ptr(),
                std::ptr::null_mut(),
                4,
                4,
            );
            // zero dims
            darkroom_heal_sub(
                top.as_ptr(),
                bottom.as_ptr(),
                red.as_mut_ptr(),
                black.as_mut_ptr(),
                0,
                4,
            );
            darkroom_heal_sub(
                top.as_ptr(),
                bottom.as_ptr(),
                red.as_mut_ptr(),
                black.as_mut_ptr(),
                4,
                0,
            );
            // i32::MAX caps
            darkroom_heal_sub(
                top.as_ptr(),
                bottom.as_ptr(),
                red.as_mut_ptr(),
                black.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                4,
            );
            darkroom_heal_sub(
                top.as_ptr(),
                bottom.as_ptr(),
                red.as_mut_ptr(),
                black.as_mut_ptr(),
                4,
                (i32::MAX as usize) + 1,
            );
        }
        assert!(red.iter().all(|&v| v == 7.0)); // untouched
        assert!(black.iter().all(|&v| v == 7.0)); // untouched
    }

    // ── heal_add tests ───────────────────────────────────────────────────

    /// Split/image buffers for `heal_add`: `(red, black, second, result)`.
    /// Split buffers carry the solver padding rows; image buffers are packed.
    fn add_buffers(width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let split_len = 4 * ((width + 1) / 2) * (height + 2);
        let img_len = 4 * width * height;
        (vec![0.0; split_len], vec![0.0; split_len], vec![0.0; img_len], vec![0.0; img_len])
    }

    #[test]
    fn add_even_width_pin() {
        // width=2, height=2, second=0 so the result is the pure re-interleave.
        // Row 0 (even): pixel 0 from black, pixel 1 from red.
        // Row 1 (odd):  pixel 0 from red,   pixel 1 from black.
        let (mut red, mut black, mut second, mut result) = add_buffers(2, 2);
        let stride = 4;
        red[stride..stride + 4].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        black[stride..stride + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        red[2 * stride..2 * stride + 4].copy_from_slice(&[9.0, 10.0, 11.0, 12.0]);
        black[2 * stride..2 * stride + 4].copy_from_slice(&[13.0, 14.0, 15.0, 16.0]);
        second.fill(0.0);
        heal_add(&red, &black, &second, &mut result, 2, 2);
        assert_eq!(
            result,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ]
        );
    }

    #[test]
    fn add_second_offsets_result() {
        // same layout as above, but second holds a constant bias: every
        // output pixel is split + bias on all four channels.
        let (mut red, mut black, mut second, mut result) = add_buffers(2, 1);
        let stride = 4;
        red[stride..stride + 4].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        black[stride..stride + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        second.fill(100.0);
        heal_add(&red, &black, &second, &mut result, 2, 1);
        assert_eq!(
            result,
            vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
        );
    }

    #[test]
    fn add_odd_width_tail_pin() {
        // width=3, height=1: pair (pixels 0,1) split black/red on the even
        // row; left-over pixel 2 (even => black) comes from buf1. The red
        // tail slot holds sentinel and must never be read.
        let (mut red, mut black, mut second, mut result) = add_buffers(3, 1);
        let stride = 8;
        black[stride..stride + 8].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);
        red[stride..stride + 8].copy_from_slice(&[5.0, 6.0, 7.0, 8.0, SENTINEL, SENTINEL, SENTINEL, SENTINEL]);
        second.fill(0.0);
        heal_add(&red, &black, &second, &mut result, 3, 1);
        assert_eq!(
            result,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn add_odd_width_tail_odd_row_pin() {
        // width=3, height=2: row 1 is odd, so the pair swaps (pixel (1,0)
        // from red, pixel (1,1) from black) and the left-over pixel (1,2)
        // comes from red (buf1). The black tail slot holds sentinel and must
        // never be read.
        let (mut red, mut black, mut second, mut result) = add_buffers(3, 2);
        let stride = 8;
        let r1 = 2 * stride;
        red[r1..r1 + 8].copy_from_slice(&[12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0]);
        black[r1..r1 + 8].copy_from_slice(&[16.0, 17.0, 18.0, 19.0, SENTINEL, SENTINEL, SENTINEL, SENTINEL]);
        second.fill(1.0);
        heal_add(&red, &black, &second, &mut result, 3, 2);
        // row 0 untouched here (split rows still zero, second=1): all ones.
        assert_eq!(&result[..12], &[1.0; 12]);
        assert_eq!(
            &result[12..],
            &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
        );
    }

    #[test]
    fn add_checker_parity_full_coverage() {
        // every output pixel of a 5x4 image must equal the slot dictated by
        // (row + col) parity plus second, and no sentinel may survive.
        let (w, h) = (5usize, 4usize);
        let stride = 4 * ((w + 1) / 2);
        let (mut red, mut black, mut second, mut result) = add_buffers(w, h);
        lcg_fill(&mut red, 0xAD01, 10.0);
        lcg_fill(&mut black, 0xAD02, 10.0);
        lcg_fill(&mut second, 0xAD03, 10.0);
        result.fill(SENTINEL);
        heal_add(&red, &black, &second, &mut result, w, h);
        assert!(!result.iter().any(|&v| v == SENTINEL));
        for row in 0..h {
            for col in 0..w {
                let src = 4 * (row * w + col);
                let dst = (row + 1) * stride + 4 * (col / 2);
                for c in 0..4 {
                    let s = if ((row + col) & 1) == 1 {
                        red[dst + c]
                    } else {
                        black[dst + c]
                    };
                    assert_eq!(result[src + c], s + second[src + c], "r={row} c={col} ch={c}");
                }
            }
        }
    }

    #[test]
    fn add_padding_rows_never_read() {
        // sentinel in the split padding rows (0 and height+1) must not leak
        // into the result: padding-filled and zero-padded runs agree exactly.
        // The odd-width opposite-colour tail slots likewise hold sentinel.
        let (w, h) = (5usize, 3usize);
        let stride = 4 * ((w + 1) / 2);
        let split_len = stride * (h + 2);
        let (red0, black0, mut second, _) = add_buffers(w, h);
        let mut red = red0;
        let mut black = black0;
        lcg_fill(&mut red[stride..(h + 1) * stride], 0x9AD1, 6.0);
        lcg_fill(&mut black[stride..(h + 1) * stride], 0x9AD2, 6.0);
        // zero the odd-width opposite-colour tail slots in the clean copies
        let mut red_clean = red.clone();
        let mut black_clean = black.clone();
        for row in 0..h {
            let tail = (row + 1) * stride + 4 * ((w - 1) / 2);
            if (row & 1) == 0 {
                red_clean[tail..tail + 4].fill(0.0);
            } else {
                black_clean[tail..tail + 4].fill(0.0);
            }
        }
        lcg_fill(&mut second, 0x9AD3, 6.0);
        // poison padding rows plus the never-read tail slots
        red[..stride].fill(SENTINEL);
        black[..stride].fill(SENTINEL);
        red[(h + 1) * stride..split_len].fill(SENTINEL);
        black[(h + 1) * stride..split_len].fill(SENTINEL);
        for row in 0..h {
            let tail = (row + 1) * stride + 4 * ((w - 1) / 2);
            if (row & 1) == 0 {
                red[tail..tail + 4].fill(SENTINEL);
            } else {
                black[tail..tail + 4].fill(SENTINEL);
            }
        }
        let mut poisoned = vec![0.0; 4 * w * h];
        let mut clean = vec![0.0; 4 * w * h];
        heal_add(&red, &black, &second, &mut poisoned, w, h);
        heal_add(&red_clean, &black_clean, &second, &mut clean, w, h);
        assert_eq!(poisoned, clean);
        for (d, r) in poisoned.iter().zip(clean.iter()) {
            assert_eq!(d.to_bits(), r.to_bits());
        }
    }

    #[test]
    fn add_matches_reference_over_lcg() {
        // even/odd widths, small and larger heights, exact and to_bits
        // equality against the structurally divergent reference.
        for &(w, h) in &[
            (1usize, 1usize),
            (2, 1),
            (1, 3),
            (2, 2),
            (3, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (7, 5),
            (8, 8),
            (13, 7),
            (16, 9),
        ] {
            let (mut red, mut black, mut second, _) = add_buffers(w, h);
            lcg_fill(&mut red, 0x60AD + w as u32, 10.0);
            lcg_fill(&mut black, 0x60BE + h as u32, 10.0);
            lcg_fill(&mut second, 0x60C0 + (w ^ h) as u32, 10.0);
            let img_len = 4 * w * h;
            let mut direct = vec![SENTINEL; img_len];
            let mut refr = vec![SENTINEL; img_len];
            heal_add(&red, &black, &second, &mut direct, w, h);
            ref_heal_add(&red, &black, &second, &mut refr, w, h);
            assert_eq!(direct, refr, "w={w} h={h}");
            for (d, r) in direct.iter().zip(refr.iter()) {
                assert_eq!(d.to_bits(), r.to_bits(), "bits w={w} h={h}");
            }
        }
    }

    #[test]
    fn add_nan_propagates_identically() {
        // addition is a single IEEE op in both kernel and reference:
        // NaN payloads must agree bit-for-bit.
        let (w, h) = (3usize, 2usize);
        let (mut red, mut black, mut second, _) = add_buffers(w, h);
        lcg_fill(&mut red, 0x6A01, 4.0);
        lcg_fill(&mut black, 0x6A02, 4.0);
        lcg_fill(&mut second, 0x6A03, 4.0);
        red[8] = f32::from_bits(0x7FC1_2345);
        second[5] = f32::from_bits(0xFFC0_00AB);
        let img_len = 4 * w * h;
        let mut direct = vec![0.0; img_len];
        let mut refr = vec![0.0; img_len];
        heal_add(&red, &black, &second, &mut direct, w, h);
        ref_heal_add(&red, &black, &second, &mut refr, w, h);
        // NOTE: no assert_eq! here — the buffers intentionally contain NaN
        // (NaN != NaN); bit patterns are compared below instead.
        for (d, r) in direct.iter().zip(refr.iter()) {
            assert_eq!(d.to_bits(), r.to_bits());
        }
    }

    #[test]
    fn add_degenerate_guards_no_op() {
        // zero dims leave the result untouched
        for &(w, h) in &[(0usize, 4usize), (4usize, 0usize), (0usize, 0usize)] {
            let (red, black, second, mut result) = add_buffers(4, 4);
            result.fill(9.0);
            heal_add(&red, &black, &second, &mut result, w, h);
            assert!(result.iter().all(|&v| v == 9.0), "w={w} h={h}");
            // reference agrees on the no-op
            let mut refr = vec![9.0; 4 * 4 * 4];
            ref_heal_add(&red, &black, &second, &mut refr, w, h);
            assert!(refr.iter().all(|&v| v == 9.0), "ref w={w} h={h}");
        }
        // short buffers: length preconditions fail -> no-op, no panic
        let (mut red, mut black, mut second, mut result) = add_buffers(4, 4);
        lcg_fill(&mut red, 0x5EA1, 2.0);
        lcg_fill(&mut black, 0x5EA2, 2.0);
        lcg_fill(&mut second, 0x5EA3, 2.0);
        result.fill(9.0);
        heal_add(&red[..10], &black, &second, &mut result, 4, 4);
        heal_add(&red, &black[..10], &second, &mut result, 4, 4);
        heal_add(&red, &black, &second[..10], &mut result, 4, 4);
        heal_add(&red, &black, &second, &mut result[..10], 4, 4);
        assert!(result.iter().all(|&v| v == 9.0));
        // boundary dimension arithmetic must no-op without panicking, even though
        // the undersized buffers would otherwise be invalid.
        heal_add(&[], &[], &[], &mut [], usize::MAX, 1);
        ref_heal_add(&[], &[], &[], &mut [], usize::MAX, 1);
    }

    #[test]
    fn add_ffi_round_trip() {
        for &(w, h) in &[(4usize, 2usize), (5usize, 3usize)] {
            let (mut red, mut black, mut second, _) = add_buffers(w, h);
            lcg_fill(&mut red, 0xFFA1 + w as u32, 8.0);
            lcg_fill(&mut black, 0xFFA2 + h as u32, 8.0);
            lcg_fill(&mut second, 0xFFA3 + (w ^ h) as u32, 8.0);
            let img_len = 4 * w * h;
            let mut ffi_out = vec![0.0; img_len];
            let mut direct = vec![0.0; img_len];
            unsafe {
                darkroom_heal_add(
                    red.as_ptr(),
                    black.as_ptr(),
                    second.as_ptr(),
                    ffi_out.as_mut_ptr(),
                    w,
                    h,
                );
            }
            heal_add(&red, &black, &second, &mut direct, w, h);
            assert_eq!(ffi_out, direct, "w={w} h={h}");
        }
    }

    #[test]
    fn add_ffi_guards() {
        let (red, black, second, mut result) = add_buffers(4, 4);
        result.fill(7.0);
        unsafe {
            // null pointers
            darkroom_heal_add(
                std::ptr::null(),
                black.as_ptr(),
                second.as_ptr(),
                result.as_mut_ptr(),
                4,
                4,
            );
            darkroom_heal_add(
                red.as_ptr(),
                std::ptr::null(),
                second.as_ptr(),
                result.as_mut_ptr(),
                4,
                4,
            );
            darkroom_heal_add(
                red.as_ptr(),
                black.as_ptr(),
                std::ptr::null(),
                result.as_mut_ptr(),
                4,
                4,
            );
            darkroom_heal_add(
                red.as_ptr(),
                black.as_ptr(),
                second.as_ptr(),
                std::ptr::null_mut(),
                4,
                4,
            );
            // zero dims
            darkroom_heal_add(
                red.as_ptr(),
                black.as_ptr(),
                second.as_ptr(),
                result.as_mut_ptr(),
                0,
                4,
            );
            darkroom_heal_add(
                red.as_ptr(),
                black.as_ptr(),
                second.as_ptr(),
                result.as_mut_ptr(),
                4,
                0,
            );
            // i32::MAX caps
            darkroom_heal_add(
                red.as_ptr(),
                black.as_ptr(),
                second.as_ptr(),
                result.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                4,
            );
            darkroom_heal_add(
                red.as_ptr(),
                black.as_ptr(),
                second.as_ptr(),
                result.as_mut_ptr(),
                4,
                (i32::MAX as usize) + 1,
            );
        }
        assert!(result.iter().all(|&v| v == 7.0)); // untouched
    }
}
