//! Float-mask threshold scan from `src/common/ras2vect.c::ras2forms` (m4-223).
//! Bitmap words match potrace's `unsigned long`, with pixels stored MSB-first.
//! Strict `<` preserves C's equality, signed-zero and unordered NaN decisions.
//! Padding bits and stride words are preserved. Rows run sequentially rather
//! than under OpenMP; each row is independent, so results are unchanged.

use std::os::raw::{c_int, c_ulong};

pub const RAS2VECT_WORD_BITS: usize = c_ulong::BITS as usize;

fn bitmap_lengths(dy: usize, width: usize, height: usize) -> Option<(usize, usize)> {
    if width == 0 || height == 0 || dy < width.div_ceil(RAS2VECT_WORD_BITS) {
        return None;
    }
    let nwords = dy.checked_mul(height)?;
    let npixels = width.checked_mul(height)?;
    if nwords > isize::MAX as usize / std::mem::size_of::<c_ulong>()
        || npixels > isize::MAX as usize / std::mem::size_of::<f32>()
    {
        return None;
    }
    Some((nwords, npixels))
}

/// Set or clear each pixel's bitmap bit using `mask[y * width + x] < threshold`.
/// `dy` is the bitmap row stride in native unsigned-long words. Invalid or
/// overflowing dimensions, insufficient stride and short buffers are no-ops.
/// Bits beyond `width` and padding words in each row are left untouched.
pub fn ras2vect_threshold_bitmap(
    map: &mut [c_ulong],
    dy: usize,
    width: usize,
    height: usize,
    mask: &[f32],
    threshold: f32,
) {
    let Some((nwords, npixels)) = bitmap_lengths(dy, width, height) else {
        return;
    };
    if map.len() < nwords || mask.len() < npixels {
        return;
    }
    for (row, mask_row) in map[..nwords]
        .chunks_exact_mut(dy)
        .zip(mask[..npixels].chunks_exact(width))
    {
        for (x, &value) in mask_row.iter().enumerate() {
            let bit = (1 as c_ulong) << (RAS2VECT_WORD_BITS - 1 - x % RAS2VECT_WORD_BITS);
            let word = &mut row[x / RAS2VECT_WORD_BITS];
            if value < threshold {
                *word |= bit;
            } else {
                *word &= !bit;
            }
        }
    }
}

/// Threshold the bitmap used by `ras2forms`, preserving row padding.
/// Null pointers, non-positive dimensions, insufficient stride and lengths
/// exceeding Rust's slice byte limit are guarded no-ops.
///
/// # Safety
/// For accepted dimensions, `map` must point to `dy * height` initialized,
/// aligned, writable `c_ulong` words and `mask` to `width * height` initialized,
/// aligned, readable floats. Each range must lie within one allocation; the
/// ranges must not overlap and `map` must be exclusively accessible.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ras2vect_threshold_bitmap(
    map: *mut c_ulong,
    dy: c_int,
    width: c_int,
    height: c_int,
    mask: *const f32,
    threshold: f32,
) {
    if map.is_null() || mask.is_null() || dy <= 0 || width <= 0 || height <= 0 {
        return;
    }
    let (dy, width, height) = (dy as usize, width as usize, height as usize);
    let Some((nwords, npixels)) = bitmap_lengths(dy, width, height) else {
        return;
    };
    let map = std::slice::from_raw_parts_mut(map, nwords);
    let mask = std::slice::from_raw_parts(mask, npixels);
    ras2vect_threshold_bitmap(map, dy, width, height, mask, threshold);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        map: &mut [c_ulong],
        dy: usize,
        width: usize,
        height: usize,
        mask: &[f32],
        threshold: f32,
    ) {
        let hibit = (1 as c_ulong) << (8 * std::mem::size_of::<c_ulong>() - 1);
        for y in 0..height {
            for x in 0..width {
                let index = x + y * width;
                let word = y * dy + x / RAS2VECT_WORD_BITS;
                let bit = hibit >> (x & (RAS2VECT_WORD_BITS - 1));
                if mask[index] < threshold {
                    map[word] |= bit;
                } else {
                    map[word] &= !bit;
                }
            }
        }
    }

    #[test]
    fn msb_first_bit_order() {
        let mask = [f32::from_bits(1.0f32.to_bits() - 1), 1.0, -0.0, 0.0];
        let mut map = [0];
        ras2vect_threshold_bitmap(&mut map, 1, 4, 1, &mask, 1.0);
        assert_eq!(map[0], (0b1011 as c_ulong) << (RAS2VECT_WORD_BITS - 4));
    }

    #[test]
    fn word_boundary_and_clear_path() {
        let wb = RAS2VECT_WORD_BITS;
        let mut mask = vec![1.0; wb + 1];
        mask[wb] = 0.0;
        let mut map = [c_ulong::MAX; 2];
        ras2vect_threshold_bitmap(&mut map, 2, wb + 1, 1, &mask, 1.0);
        assert_eq!(map, [0, c_ulong::MAX]);
    }

    #[test]
    fn nan_and_inf_pins() {
        let mask = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0];
        let mut map = [c_ulong::MAX];
        ras2vect_threshold_bitmap(&mut map, 1, 4, 1, &mask, 1.0);
        let padding = c_ulong::MAX >> 4;
        assert_eq!(map[0], (1 << (RAS2VECT_WORD_BITS - 3)) | padding);
        ras2vect_threshold_bitmap(&mut map, 1, 4, 1, &mask, f32::NAN);
        assert_eq!(map[0], padding);
        ras2vect_threshold_bitmap(&mut map, 1, 4, 1, &[-0.0, 0.0, -1.0, 1.0], 0.0);
        assert_eq!(map[0], (1 << (RAS2VECT_WORD_BITS - 3)) | padding);
    }

    #[test]
    fn matches_reference_over_sweep() {
        let wb = RAS2VECT_WORD_BITS;
        for width in [1, 2, 7, wb - 1, wb, wb + 1, 3 * wb + 5] {
            for height in [1, 2, 5] {
                for padding in [0, 2] {
                    let dy = width.div_ceil(wb) + padding;
                    let mask: Vec<_> = (0..width * height)
                        .map(|i| match i % 13 {
                            0 => f32::NAN,
                            1 => f32::INFINITY,
                            2 => f32::NEG_INFINITY,
                            3 => -0.0,
                            _ => ((i * 37 % 101) as f32 - 50.0) / 16.0,
                        })
                        .collect();
                    for threshold in [0.0, -2.5, 1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let initial: Vec<_> = (0..dy * height + 1)
                            .map(|i| !(i as c_ulong).wrapping_mul(0x9E37_79B9))
                            .collect();
                        let mut actual = initial.clone();
                        let mut expected = initial.clone();
                        reference(&mut expected, dy, width, height, &mask, threshold);
                        ras2vect_threshold_bitmap(&mut actual, dy, width, height, &mask, threshold);
                        assert_eq!(
                            actual, expected,
                            "{width}x{height}, stride {dy}, {threshold}"
                        );
                        assert_eq!(actual[dy * height], initial[dy * height]);
                        for y in 0..height {
                            let used = width.div_ceil(wb);
                            assert_eq!(
                                actual[y * dy + used..(y + 1) * dy],
                                initial[y * dy + used..(y + 1) * dy]
                            );
                            if width % wb != 0 {
                                let bits = c_ulong::MAX >> (width % wb);
                                let last = y * dy + used - 1;
                                assert_eq!(actual[last] & bits, initial[last] & bits);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let mut map = [0xAA; 2];
        for (dy, w, h) in [
            (0, 4, 1),
            (1, 0, 1),
            (1, 4, 0),
            (1, RAS2VECT_WORD_BITS + 1, 1),
            (usize::MAX, 1, 2),
            (1, usize::MAX, usize::MAX),
            (isize::MAX as usize, 1, 1),
            (3, 1, 1),
            (1, 4, 2),
        ] {
            ras2vect_threshold_bitmap(&mut map, dy, w, h, &[0.0; 4], 1.0);
            assert_eq!(map, [0xAA; 2]);
        }
        ras2vect_threshold_bitmap(&mut map, 1, 4, 1, &[0.0], 1.0);
        assert_eq!(map, [0xAA; 2]);
        ras2vect_threshold_bitmap(&mut [], 1, 1, 1, &[0.0], 1.0);
    }

    #[test]
    fn ffi_round_trip() {
        let width = RAS2VECT_WORD_BITS + 3;
        let height = 3;
        let dy = width.div_ceil(RAS2VECT_WORD_BITS) + 1;
        let mask: Vec<_> = (0..width * height).map(|i| (i % 7) as f32 / 4.0).collect();
        let mut actual = vec![c_ulong::MAX; dy * height];
        let mut expected = actual.clone();
        reference(&mut expected, dy, width, height, &mask, 1.0);
        unsafe {
            darkroom_ras2vect_threshold_bitmap(
                actual.as_mut_ptr(),
                dy as c_int,
                width as c_int,
                height as c_int,
                mask.as_ptr(),
                1.0,
            );
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn ffi_guards() {
        let mut map = [0xAA; 2];
        let mask = [0.0; 4];
        unsafe {
            darkroom_ras2vect_threshold_bitmap(std::ptr::null_mut(), 1, 4, 1, mask.as_ptr(), 1.0);
            darkroom_ras2vect_threshold_bitmap(map.as_mut_ptr(), 1, 4, 1, std::ptr::null(), 1.0);
            for (dy, w, h) in [
                (0, 4, 1),
                (1, 0, 1),
                (1, 4, 0),
                (-1, 4, 1),
                (1, -1, 1),
                (1, 4, -1),
                (1, RAS2VECT_WORD_BITS as c_int + 1, 1),
                (c_int::MAX, c_int::MAX, c_int::MAX),
                (
                    c_int::MAX / RAS2VECT_WORD_BITS as c_int + 1,
                    c_int::MAX,
                    c_int::MAX,
                ),
            ] {
                darkroom_ras2vect_threshold_bitmap(map.as_mut_ptr(), dy, w, h, mask.as_ptr(), 1.0);
            }
        }
        assert_eq!(map, [0xAA; 2]);
    }
}
