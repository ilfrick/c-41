//! Shared CFA / mosaic-pattern primitives for Bayer and X-Trans sensors.
//!
//! These mirror the inline helpers in src/develop/imageop_math.h and are
//! the keystone for migrating any IOP that walks a raw mosaic — highlights
//! mask, rawoverexposed, hotpixels' X-Trans branch, demosaic, etc.

/// Bayer pattern colour index (0=R, 1=G1/G2, 2=B) for the given row/col.
///
/// Mirrors `FC(row, col, filters)` from imageop_math.h:
///   FC = (filters >> (((row<<1 & 14) + (col & 1)) << 1)) & 3
#[inline(always)]
pub fn fc_bayer(row: i32, col: i32, filters: u32) -> usize {
    let r = (row as u32) & 0xffff_ffff;
    let c = (col as u32) & 0xffff_ffff;
    let shift = (((r << 1) & 14) + (c & 1)) << 1;
    ((filters >> shift) & 3) as usize
}

/// X-Trans CFA colour index (0..5, mapping to R/G/B by sensor) for the
/// given row/col through a 6x6 pattern table.
///
/// Mirrors `FCNxtrans(row, col, xtrans)` from imageop_math.h. The +600
/// offset shields negative row/col values (some IOPs use them) from the
/// modulo wrap.
#[inline(always)]
pub fn fc_xtrans(row: i32, col: i32, xtrans: &[[u8; 6]; 6]) -> usize {
    let irow = ((row + 600) as usize) % 6;
    let icol = ((col + 600) as usize) % 6;
    xtrans[irow][icol] as usize
}

/// Unified CFA colour lookup: dispatches to `fc_bayer` for normal filter masks
/// or to `fc_xtrans` when `filters == 9` (the X-Trans sentinel value).
///
/// Mirrors `fcol(row, col, filters, xtrans)` from imageop_math.h.
#[inline(always)]
pub fn fcol(row: i32, col: i32, filters: u32, xtrans: &[[u8; 6]; 6]) -> usize {
    if filters == 9 {
        fc_xtrans(row, col, xtrans)
    } else {
        fc_bayer(row, col, filters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Common Bayer filters mask used by tests — RGGB layout for filters = 0x94949494
    // (each 2-bit group encodes one colour, replicated for 4 row pairs).
    const RGGB: u32 = 0x94949494;

    #[test]
    fn fc_bayer_returns_rggb_pattern() {
        // RGGB: row 0 col 0 = R, row 0 col 1 = G, row 1 col 0 = G, row 1 col 1 = B
        assert_eq!(fc_bayer(0, 0, RGGB), 0);
        assert_eq!(fc_bayer(0, 1, RGGB), 1);
        assert_eq!(fc_bayer(1, 0, RGGB), 1);
        assert_eq!(fc_bayer(1, 1, RGGB), 2);
    }

    #[test]
    fn fc_bayer_repeats_with_period_two() {
        // The Bayer pattern repeats every 2 rows and 2 cols.
        assert_eq!(fc_bayer(2, 2, RGGB), fc_bayer(0, 0, RGGB));
        assert_eq!(fc_bayer(3, 5, RGGB), fc_bayer(1, 1, RGGB));
    }

    #[test]
    fn fc_xtrans_negative_row_safe() {
        let pattern: [[u8; 6]; 6] = [
            [1, 0, 1, 1, 2, 1], [1, 1, 2, 1, 1, 0],
            [1, 2, 1, 1, 1, 1], [2, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 2, 1], [1, 1, 2, 1, 1, 0],
        ];
        // Negative row should not panic, returns the same as (row + 6) mod 6
        assert_eq!(fc_xtrans(-1, 0, &pattern), pattern[5][0] as usize);
        assert_eq!(fc_xtrans(-7, 0, &pattern), pattern[5][0] as usize);
    }

    #[test]
    fn fcol_dispatches_on_filters_nine() {
        let pattern: [[u8; 6]; 6] = [[3; 6]; 6]; // marker value
        assert_eq!(fcol(0, 0, 9, &pattern), 3);     // X-Trans branch
        assert_eq!(fcol(0, 0, RGGB, &pattern), 0);  // Bayer branch
    }
}
