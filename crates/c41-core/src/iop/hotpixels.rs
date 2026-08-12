use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Hotpixels;

impl IopProcess for Hotpixels {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "hotpixels" }
}

/// Bayer-sensor hot-pixel correction.
///
/// For every interior pixel (row, col) in 2..h-2 x 2..w-2 of the single-plane
/// raw buffer:
///   * If `input[k] > threshold`, examine the four same-color Bayer neighbours
///     at offsets (-2,0), (+2,0), (0,-2), (0,+2).
///   * Let `mid = input[k] * multiplier`. Count how many neighbours satisfy
///     `mid > neighbour`, and track the maximum of those neighbours.
///   * If `count >= min_neighbours`, replace `output[k]` with that maximum
///     value and increment the returned `fixed` counter.
///
/// `mark_fixed` is the UI debug overlay: when enabled, the corrected pixel
/// stamps its original value into the same row at column offsets ±2, ±4 …
/// ±10 (same Bayer colour positions) so the user can see where corrections
/// happened. Matches the markfixed branch in src/iop/hotpixels.c exactly.
///
/// Returns the count of pixels that were replaced.
#[no_mangle]
pub unsafe extern "C" fn darkroom_hotpixels_bayer(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    threshold: f32,
    multiplier: f32,
    min_neighbours: i32,
    mark_fixed: i32,
) -> i32 {
    let mark_fixed = mark_fixed != 0;
    if width < 5 || height < 5 {
        return 0;
    }
    let n = width * height;
    let input = std::slice::from_raw_parts(in_buf, n);
    let output = std::slice::from_raw_parts_mut(out_buf, n);
    let width_x2 = width * 2;
    let mut fixed: i32 = 0;

    for row in 2..(height - 2) {
        let row_start = row * width;
        for col in 2..(width - 2) {
            let k = row_start + col;
            let v = input[k];
            if v <= threshold { continue; }

            let mid = v * multiplier;
            let mut count = 0;
            let mut maxin = 0.0_f32;

            // four same-colour Bayer neighbours
            let neighbours = [
                input[k - 2],
                input[k - width_x2],
                input[k + 2],
                input[k + width_x2],
            ];
            for &other in &neighbours {
                if mid > other {
                    count += 1;
                    if other > maxin { maxin = other; }
                }
            }

            if count >= min_neighbours {
                output[k] = maxin;
                fixed += 1;

                if mark_fixed {
                    // Stamp original value at column offsets -10..-2 (step 2)
                    let mut i: isize = -2;
                    while i >= -10 && (col as isize + i) >= 0 {
                        output[(k as isize + i) as usize] = v;
                        i -= 2;
                    }
                    // and +2..+10 (step 2)
                    let mut i: isize = 2;
                    while i <= 10 && (col as isize + i) < width as isize {
                        output[(k as isize + i) as usize] = v;
                        i += 2;
                    }
                }
            }
        }
    }

    fixed
}

/// Multi-plane monochrome hot-pixel correction.
///
/// Same algorithm as the Bayer variant but neighbours are at offset ±1 (not
/// ±2) — there's no Bayer-colour aliasing for a monochrome sensor — and the
/// pixel itself spans `planes` channels. When a pixel is identified as hot,
/// every channel of that pixel gets the same replacement maximum, and the
/// mark-fixed overlay stamps the original value into the same row at
/// column offsets ±1..±10 (step 1).
///
/// Note the subtle stride: the C source iterates by `planes` per pixel,
/// reading neighbour offsets as `-planes`, `+planes`, `-planes*width`,
/// `+planes*width` — i.e. the neighbour is the next pixel of the same
/// channel. The mark-fixed branch uses an offset `4*i + c` (hard-coded 4
/// despite `planes` possibly differing) — we mirror that quirk exactly.
///
/// Matches _process_monochrome() in src/iop/hotpixels.c.
#[no_mangle]
pub unsafe extern "C" fn darkroom_hotpixels_monochrome(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    planes: usize,
    threshold: f32,
    multiplier: f32,
    min_neighbours: i32,
    mark_fixed: i32,
) -> i32 {
    if width < 3 || height < 3 || planes == 0 {
        return 0;
    }
    let mark_fixed = mark_fixed != 0;
    let n = width * height * planes;
    let input = std::slice::from_raw_parts(in_buf, n);
    let output = std::slice::from_raw_parts_mut(out_buf, n);
    let mut fixed: i32 = 0;

    for row in 1..(height - 1) {
        for col in 1..(width - 1) {
            // Index of the current pixel's first channel.
            let k = (row * width + col) * planes;
            let v = input[k];
            if v <= threshold { continue; }

            let mid = v * multiplier;
            let mut count = 0;
            let mut maxin = 0.0_f32;

            // four 4-neighbour offsets, same channel
            let offsets = [-(planes as isize), -((planes * width) as isize),
                           planes as isize, (planes * width) as isize];
            for &off in &offsets {
                let other = input[(k as isize + off) as usize];
                if mid > other {
                    count += 1;
                    if other > maxin { maxin = other; }
                }
            }

            if count >= min_neighbours {
                for c in 0..planes {
                    output[k + c] = maxin;
                }
                fixed += 1;

                if mark_fixed {
                    // Same-row stamps: i ∈ [-10..-1] then [1..10], hard-coded
                    // stride 4 (matches the C source verbatim — see notes above).
                    let mut i: isize = -1;
                    while i >= -10 && (col as isize + i) >= 0 {
                        let base = (k as isize + 4 * i) as usize;
                        if base + planes <= n {
                            for c in 0..planes {
                                output[base + c] = v;
                            }
                        }
                        i -= 1;
                    }
                    let mut i: isize = 1;
                    while i <= 10 && (col as isize + i) < width as isize {
                        let base = (k as isize + 4 * i) as usize;
                        if base + planes <= n {
                            for c in 0..planes {
                                output[base + c] = v;
                            }
                        }
                        i += 1;
                    }
                }
            }
        }
    }

    fixed
}

/// Pre-compute the per-cell same-colour neighbour offset table for an X-Trans
/// 6x6 CFA pattern.
///
/// Mirrors the loop at the top of `_process_xtrans()` in src/iop/hotpixels.c.
/// For each (j, i) in [0..6)x[0..6), walks a 20-entry search ring and records
/// the (dx, dy) of the first 4 neighbours whose CFA colour matches the centre.
fn xtrans_neighbour_offsets(xtrans: &[[u8; 6]; 6]) -> [[[(i32, i32); 4]; 6]; 6] {
    const SEARCH: [(i32, i32); 20] = [
        (-1,  0), (1,  0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2,  0), (2,  0), (0, -2), (0, 2),
        (-2, -1), (-2, 1), (2, -1), (2, 1),
        (-1, -2), (1, -2), (-1, 2), (1, 2),
    ];
    let mut out = [[[(0_i32, 0_i32); 4]; 6]; 6];
    for j in 0..6 {
        for i in 0..6 {
            let c = crate::raw::fc_xtrans(j as i32, i as i32, xtrans);
            let mut found = 0;
            for s in 0..20 {
                if found == 4 { break; }
                let (dx, dy) = SEARCH[s];
                let cn = crate::raw::fc_xtrans(j as i32 + dy, i as i32 + dx, xtrans);
                if cn == c {
                    out[j][i][found] = (dx, dy);
                    found += 1;
                }
            }
        }
    }
    out
}

/// X-Trans-sensor hot-pixel correction.
///
/// Mirrors `_process_xtrans()` in src/iop/hotpixels.c. For every interior
/// pixel (row, col) in 2..h-2 x 2..w-2:
///   * If `input[k] > threshold`, examine the 4 pre-computed same-colour
///     neighbours of (row%6, col%6).
///   * Let mid = `input[k] * multiplier`. Count how many neighbours
///     satisfy mid > neighbour and track their max.
///   * If `count >= min_neighbours`, replace `output[k]` with that max.
///
/// When `mark_fixed` is set, stamps the original value at column offsets
/// +/-2..+/-10 where the X-Trans CFA colour at (row, col+i) matches the
/// centre — same as the C source, NOT every position.
///
/// Returns the count of pixels replaced.
#[no_mangle]
pub unsafe extern "C" fn darkroom_hotpixels_xtrans(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    xtrans: *const u8, // flat 36-byte 6x6
    threshold: f32,
    multiplier: f32,
    min_neighbours: i32,
    mark_fixed: i32,
) -> i32 {
    if width < 5 || height < 5 { return 0; }
    let mark_fixed = mark_fixed != 0;
    let n = width * height;
    let input = std::slice::from_raw_parts(in_buf, n);
    let output = std::slice::from_raw_parts_mut(out_buf, n);

    // Lift the 36-byte buffer into a 6x6 array for cheap lookups.
    let xt_bytes = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = xt_bytes[r * 6 + c]; }
    }

    let offsets = xtrans_neighbour_offsets(&xt);
    let w_i = width as isize;
    let mut fixed: i32 = 0;

    for row in 2..(height - 2) {
        for col in 2..(width - 2) {
            let k = row * width + col;
            let v = input[k];
            if v <= threshold { continue; }
            let mid = v * multiplier;

            let mut count = 0;
            let mut maxin = 0.0_f32;
            let cell = &offsets[row % 6][col % 6];
            for &(dx, dy) in cell {
                let off = dx as isize + dy as isize * w_i;
                let other = input[(k as isize + off) as usize];
                if mid > other {
                    count += 1;
                    if other > maxin { maxin = other; }
                }
            }

            if count >= min_neighbours {
                output[k] = maxin;
                fixed += 1;

                if mark_fixed {
                    let c = crate::raw::fc_xtrans(row as i32, col as i32, &xt);
                    // Same-row stamps at offsets -10..-2 step -1 (C source uses -1 step)
                    let mut i: isize = -2;
                    while i >= -10 && (col as isize + i) >= 0 {
                        let cc = crate::raw::fc_xtrans(row as i32, (col as i32) + i as i32, &xt);
                        if cc == c {
                            output[(k as isize + i) as usize] = v;
                        }
                        i -= 1;
                    }
                    let mut i: isize = 2;
                    while i <= 10 && (col as isize + i) < width as isize {
                        let cc = crate::raw::fc_xtrans(row as i32, (col as i32) + i as i32, &xt);
                        if cc == c {
                            output[(k as isize + i) as usize] = v;
                        }
                        i += 1;
                    }
                }
            }
        }
    }

    fixed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: usize, height: usize, val: f32) -> Vec<f32> {
        vec![val; width * height]
    }

    #[test]
    fn no_pixels_above_threshold_no_change() {
        let w = 8; let h = 8;
        let input = flat(w, h, 0.1);
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_bayer(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, 0.5, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 0);
        assert_eq!(output, input);
    }

    #[test]
    fn isolated_hot_pixel_gets_replaced() {
        let w = 8; let h = 8;
        let mut input = flat(w, h, 0.1);
        // Hot pixel at (4, 4) — well above threshold and much brighter than its
        // four same-colour Bayer neighbours (all 0.1).
        let k = 4 * w + 4;
        input[k] = 10.0;
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_bayer(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, 1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 1, "expected exactly one fix");
        // mid = 10*0.5 = 5; all neighbours = 0.1 → max = 0.1
        assert!((output[k] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn fewer_than_min_neighbours_keeps_pixel() {
        let w = 8; let h = 8;
        let mut input = flat(w, h, 0.1);
        let k = 4 * w + 4;
        input[k] = 10.0;
        // Also raise three neighbours above mid so only ONE neighbour satisfies
        // mid > other. With min_neighbours = 4, no fix.
        input[k - 2] = 9.0;
        input[k - 16] = 9.0; // width*2
        input[k + 2] = 9.0;
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_bayer(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, 1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 0);
        assert_eq!(output[k], 10.0);
    }

    #[test]
    fn permissive_min_three_still_fixes_clean_hot_pixel() {
        // With a clean isolated hot pixel (all 4 same-colour neighbours below
        // mid), min_neighbours = 3 must also fix it — the permissive flag
        // only changes whether 3-of-4 is good enough; a 4-of-4 case is always
        // good enough regardless of the flag.
        let w = 8; let h = 8;
        let mut input = flat(w, h, 0.1);
        let k = 4 * w + 4;
        input[k] = 10.0;
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_bayer(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, 1.0, 0.5, 3, 0,
            )
        };
        assert_eq!(n, 1);
    }

    #[test]
    fn border_pixels_are_skipped() {
        let w = 8; let h = 8;
        let mut input = flat(w, h, 0.1);
        // Hot pixel in the border (row=0) — must not be touched (out of range).
        input[0] = 100.0;
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_bayer(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, 1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 0);
        assert_eq!(output[0], 100.0);
    }

    #[test]
    fn mark_fixed_stamps_original_value_in_row() {
        // Need col > 10 and width - col > 10 so both ±10 markers fit, matching
        // the boundary checks in the original C code (`i >= -col`, `i < width-col`).
        let w = 24; let h = 16;
        let row = 8;
        let col = 12;
        let mut input = flat(w, h, 0.1);
        let k = row * w + col;
        input[k] = 10.0;
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_bayer(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, 1.0, 0.5, 4, 1, // mark_fixed = true
            )
        };
        assert_eq!(n, 1);
        // Centre replaced with neighbours' max
        assert!((output[k] - 0.1).abs() < 1e-6);
        // Same-row marker stamps at col offsets ±2, ±4, ±6, ±8, ±10 (step 2)
        for off in [-10_isize, -8, -6, -4, -2, 2, 4, 6, 8, 10].iter() {
            let idx = (k as isize + off) as usize;
            assert!((output[idx] - 10.0).abs() < 1e-6,
                    "marker missing at offset {off}: got {}", output[idx]);
        }
    }

    #[test]
    fn monochrome_isolated_hot_pixel_gets_replaced() {
        // 4-channel monochrome buffer, 8x8.
        let w = 8; let h = 8; let p = 4;
        let mut input = vec![0.1_f32; w * h * p];
        let k = (4 * w + 4) * p;
        for c in 0..p { input[k + c] = 10.0; }
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_monochrome(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, p, 1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 1);
        // Replacement = max of 4 same-channel neighbours = 0.1
        for c in 0..p {
            assert!((output[k + c] - 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn monochrome_neighbour_offset_uses_planes_stride() {
        // With planes=2, neighbours are at ±2 and ±2*width — confirm we look at
        // the right pixel, not the next channel of the same pixel.
        let w = 8; let h = 8; let p = 2;
        let mut input = vec![0.1_f32; w * h * p];
        let k = (4 * w + 4) * p;
        input[k] = 10.0;     // hot pixel channel 0
        input[k + 1] = 10.0; // hot pixel channel 1
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_monochrome(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, p, 1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 1);
    }

    #[test]
    fn monochrome_border_pixels_are_skipped() {
        let w = 8; let h = 8; let p = 1;
        let mut input = vec![0.1_f32; w * h * p];
        input[0] = 100.0; // top-left border
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_monochrome(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, p, 1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 0);
        assert_eq!(output[0], 100.0);
    }

    /// A canonical Fujifilm X-Trans pattern. 0=R, 1=G, 2=B.
    const XTRANS_FUJI: [[u8; 6]; 6] = [
        [1, 2, 1, 1, 0, 1],
        [0, 1, 0, 2, 1, 2],
        [1, 2, 1, 1, 0, 1],
        [1, 0, 1, 1, 2, 1],
        [2, 1, 2, 0, 1, 0],
        [1, 0, 1, 1, 2, 1],
    ];

    #[test]
    fn xtrans_offsets_always_finds_four_neighbours() {
        let offs = xtrans_neighbour_offsets(&XTRANS_FUJI);
        for j in 0..6 {
            for i in 0..6 {
                // Every cell must have 4 non-zero (dx, dy) entries (we recorded
                // only matches; zero-zero would be the centre itself, which is
                // never in the search ring).
                let cell = &offs[j][i];
                for n in 0..4 {
                    assert!(cell[n] != (0, 0), "cell ({j},{i}) entry {n} is (0,0)");
                }
            }
        }
    }

    #[test]
    fn xtrans_offsets_neighbours_have_same_colour() {
        let offs = xtrans_neighbour_offsets(&XTRANS_FUJI);
        for j in 0..6 {
            for i in 0..6 {
                let centre = crate::raw::fc_xtrans(j as i32, i as i32, &XTRANS_FUJI);
                for n in 0..4 {
                    let (dx, dy) = offs[j][i][n];
                    let nb = crate::raw::fc_xtrans(j as i32 + dy, i as i32 + dx, &XTRANS_FUJI);
                    assert_eq!(nb, centre, "({j},{i}) neighbour {n} colour differs");
                }
            }
        }
    }

    #[test]
    fn xtrans_isolated_hot_pixel_gets_replaced() {
        let w = 16; let h = 16;
        let mut input = vec![0.1_f32; w * h];
        // Pick a pixel well inside the loop range
        let row = 8;
        let col = 8;
        input[row * w + col] = 10.0;
        let mut output = input.clone();
        let xt_flat: Vec<u8> = XTRANS_FUJI.iter().flatten().copied().collect();
        let n = unsafe {
            darkroom_hotpixels_xtrans(
                input.as_ptr(), output.as_mut_ptr(),
                w, h,
                xt_flat.as_ptr(),
                1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 1, "expected exactly one fix");
        // Replacement should be 0.1 (the max of the same-colour neighbours).
        assert!((output[row * w + col] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn xtrans_below_threshold_no_op() {
        let w = 16; let h = 16;
        let input = vec![0.1_f32; w * h];
        let mut output = input.clone();
        let xt_flat: Vec<u8> = XTRANS_FUJI.iter().flatten().copied().collect();
        let n = unsafe {
            darkroom_hotpixels_xtrans(
                input.as_ptr(), output.as_mut_ptr(),
                w, h,
                xt_flat.as_ptr(),
                1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 0);
        assert_eq!(output, input);
    }

    #[test]
    fn returns_count_of_fixes() {
        let w = 16; let h = 16;
        let mut input = flat(w, h, 0.1);
        // Two hot pixels spaced widely so their marker regions don't overlap.
        input[3 * w + 3] = 10.0;
        input[12 * w + 12] = 10.0;
        let mut output = input.clone();
        let n = unsafe {
            darkroom_hotpixels_bayer(
                input.as_ptr(), output.as_mut_ptr(),
                w, h, 1.0, 0.5, 4, 0,
            )
        };
        assert_eq!(n, 2);
    }
}
