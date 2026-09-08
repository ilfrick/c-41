//! Kernel ported from `src/common/heal.c` (`_heal_sub`, heal.c:50-92, m4-177).
//!
//! What the function does: subtract the reference pattern (`bottom_buffer`)
//! from the sample image (`top_buffer`) pixel by pixel and store the
//! difference split by red/black checkerboard colour into two contiguous
//! buffers, ready for the Gauss-Seidel Laplace solver (`_heal_laplace_loop`).
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
//! widths the left-over right-most pixel goes into `buf1` and the
//! opposite-colour tail slot is zeroed.
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
//! - The only arithmetic is one f32 subtraction per channel — a single
//!   IEEE-754 operation with no reassociation. That makes the Rust kernel
//!   equivalent by construction to the former single-threaded C loop; the
//!   `to_bits` comparisons pin the kernel against the independent Rust
//!   reference, not against a retained C golden dump.
//! - The C loop was `DT_OMP_FOR` over rows, but rows write disjoint regions
//!   (distinct `row_start` offsets in each buffer) and the padding clears run
//!   after the loop, so the sequential port is identical for any thread count.
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
}
