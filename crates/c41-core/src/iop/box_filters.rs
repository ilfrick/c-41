//! Port of the single-channel path of darktable's `common/box_filters.cc` —
//! `dt_box_mean(buf, h, w, ch=1, radius, iterations)` — the separable
//! sliding-window box average used by the bloom module (and available to any
//! future caller needing an edge-aware-of-bounds blur).
//!
//! Semantics preserved from `_blur_horizontal`/`_blur_vertical`
//! (box_filters.cc:185/308):
//!
//! * The window at each output position is the *intersection* of
//!   `[pos-radius, pos+radius]` with the row/column bounds — border pixels
//!   average over fewer taps rather than replicating or clamping (the
//!   five-phase structure: left-half accumulate, grow, a stall phase for
//!   `radius > dim/2`, bulk subtract+add, right-end decrement).
//! * Passes run horizontal-then-vertical per iteration, `iterations` times.
//! * Summation is plain f32 (`compensated = false`, i.e. no Kahan), matching
//!   bloom's `ch == 1` dispatch — bloom.c calls `dt_box_mean(..., 1, ...)` and
//!   only the `ch | BOXFILTER_KAHAN_SUM` variants compensate.
//!
//! Deviations from the C (numerics-neutral, documented per the porting
//! convention): the C vectorises rows/columns MAX_VECT lanes at a time and
//! uses a power-of-two-masked circular scratch to bound its working set; both
//! are pure layout optimisations — every row/column is independent and the
//! arithmetic per element is identical — so we process one line at a time with
//! a full-length scratch instead. The C also subtracts before adding in the
//! bulk phase; that order is kept here so accumulation rounding matches.

/// In-place separable box mean over a window of size `2*radius + 1`, applied
/// `iterations` times, on a packed `height × width` plane of f32 values.
///
/// This is `dt_box_mean(buf, height, width, 1, radius, iterations)` with
/// `BOXFILTER_KAHAN_SUM` unset. `radius == 0` is a no-op pass-through (every
/// window degenerates to the pixel itself); `iterations == 0` leaves the
/// buffer untouched.
pub fn box_mean_1ch(buf: &mut [f32], height: usize, width: usize, radius: usize, iterations: u32) {
    if buf.is_empty() || width == 0 || height == 0 || radius == 0 || iterations == 0 {
        return;
    }
    debug_assert_eq!(buf.len(), height * width, "plane must be tightly packed");

    // One scratch line shared by every row/column of a pass; sized for the
    // larger dimension so both helpers can use it without reallocating.
    let mut scratch = vec![0.0f32; height.max(width)];
    for _ in 0..iterations {
        for row in 0..height {
            blur_horizontal(&mut buf[row * width..][..width], &mut scratch[..width], radius);
        }
        for col in 0..width {
            blur_vertical(buf, height, width, col, &mut scratch[..height], radius);
        }
    }
}

/// One horizontal pass over a single row (box_filters.cc:185). `scratch`
/// receives copies of the input values as they enter the running sum, so later
/// subtractions remove pre-blur values even though `row` is overwritten in
/// place — same role as the C's `_load_add(scratch …)`.
fn blur_horizontal(row: &mut [f32], scratch: &mut [f32], radius: usize) {
    let width = row.len();
    debug_assert!(scratch.len() >= width);
    let mut sum = 0.0f32;
    let mut hits = 0usize;

    // add up the left half of the window
    for x in 0..radius.min(width) {
        hits += 1;
        scratch[x] = row[x];
        sum += row[x];
    }
    // blur up to the point where values start leaving the moving average
    let mut x = 0usize;
    while x <= radius && x + radius < width {
        hits += 1;
        scratch[x + radius] = row[x + radius];
        sum += row[x + radius];
        row[x] = sum / hits as f32;
        x += 1;
    }
    // radius > width/2: neither add nor remove possible — just store
    while x <= radius && x < width {
        row[x] = sum / hits as f32;
        x += 1;
    }
    // bulk of the scan line: subtract the outgoing value, add the incoming one
    while x + radius < width {
        sum -= scratch[x - radius - 1];
        scratch[x + radius] = row[x + radius];
        sum += row[x + radius];
        row[x] = sum / hits as f32;
        x += 1;
    }
    // right end: no more values enter the sum
    while x < width {
        hits -= 1;
        sum -= scratch[x - radius - 1];
        row[x] = sum / hits as f32;
        x += 1;
    }
}

/// One vertical pass over column `col` of a packed `height × width` plane,
/// mirroring `_blur_vertical` (box_filters.cc:308). With a full-height scratch
/// there is no aliasing between the value being stored and the history being
/// kept, so the C's power-of-two mask reduces to direct indexing.
fn blur_vertical(buf: &mut [f32], height: usize, width: usize, col: usize, scratch: &mut [f32], radius: usize) {
    debug_assert!(scratch.len() >= height);
    let mut sum = 0.0f32;
    let mut hits = 0usize;

    for y in 0..radius.min(height) {
        hits += 1;
        scratch[y] = buf[col + y * width];
        sum += buf[col + y * width];
    }
    let mut y = 0usize;
    while y <= radius && y + radius < height {
        hits += 1;
        scratch[y + radius] = buf[col + (y + radius) * width];
        sum += buf[col + (y + radius) * width];
        buf[col + y * width] = sum / hits as f32;
        y += 1;
    }
    while y <= radius && y < height {
        buf[col + y * width] = sum / hits as f32;
        y += 1;
    }
    while y + radius < height {
        sum -= scratch[y - radius - 1];
        scratch[y + radius] = buf[col + (y + radius) * width];
        sum += buf[col + (y + radius) * width];
        buf[col + y * width] = sum / hits as f32;
        y += 1;
    }
    while y < height {
        hits -= 1;
        sum -= scratch[y - radius - 1];
        buf[col + y * width] = sum / hits as f32;
        y += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct O(n·r) reference implementing the same window∩bounds semantics
    /// independently of the sliding-window machinery under test.
    fn naive_box_mean_1ch(buf: &[f32], height: usize, width: usize, radius: usize, iterations: u32) -> Vec<f32> {
        let mut cur = buf.to_vec();
        let mut next = vec![0.0f32; height * width];
        for _ in 0..iterations {
            for y in 0..height {
                for x in 0..width {
                    let lo = x.saturating_sub(radius);
                    let hi = (x + radius).min(width - 1);
                    let mut s = 0.0f32;
                    for xx in lo..=hi {
                        s += cur[y * width + xx];
                    }
                    next[y * width + x] = s / (hi - lo + 1) as f32;
                }
            }
            std::mem::swap(&mut cur, &mut next);
            for y in 0..height {
                for x in 0..width {
                    let lo = y.saturating_sub(radius);
                    let hi = (y + radius).min(height - 1);
                    let mut s = 0.0f32;
                    for yy in lo..=hi {
                        s += cur[yy * width + x];
                    }
                    next[y * width + x] = s / (hi - lo + 1) as f32;
                }
            }
            std::mem::swap(&mut cur, &mut next);
        }
        cur
    }

    #[test]
    fn hand_computed_row_edges_shrink_the_window() {
        // width=4, radius=1: windows are [0..1],[0..2],[1..3],[2..3] — the
        // borders average over fewer taps, exactly.
        let mut row = [1.0f32, 2.0, 3.0, 4.0];
        blur_horizontal(&mut row, &mut [0.0f32; 4], 1);
        assert_eq!(row, [1.5, 2.0, 3.0, 3.5]);
    }

    #[test]
    fn constant_field_stays_constant() {
        let (h, w) = (23, 31);
        // 40 ≥ w exercises the radius>width stall path
        for radius in [1usize, 7, 40] {
            let mut buf = vec![37.0f32; h * w];
            box_mean_1ch(&mut buf, h, w, radius, 8);
            for v in buf {
                assert!((v - 37.0).abs() < 1e-3, "radius {radius}: {v}");
            }
        }
    }

    #[test]
    fn matches_naive_reference_across_shapes_and_radii() {
        // xorshift LCG keeps this deterministic without a rng dependency.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) % 1000) as f32 / 10.0 // Lab-L-ish magnitudes 0..100
        };
        for &(h, w) in &[(16usize, 24usize), (5, 5), (1, 30), (30, 1), (9, 61)] {
            let buf: Vec<f32> = (0..h * w).map(|_| next()).collect();
            for radius in [0usize, 1, 3, 8, 30 /* > w for some shapes */] {
                for iters in [1u32, 3] {
                    let mut got = buf.clone();
                    box_mean_1ch(&mut got, h, w, radius, iters);
                    let want = naive_box_mean_1ch(&buf, h, w, radius, iters);
                    for (g, wv) in got.iter().zip(&want) {
                        assert!(
                            (g - wv).abs() < 5e-3,
                            "h{h} w{w} r{radius} i{iters}: {g} vs {wv}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn impulse_spreads_over_the_window_only() {
        // A lone impulse must bleed exactly into its 2r+1 neighbourhood after
        // one iteration and nowhere else; total mass is conserved by a mean.
        let (h, w) = (11usize, 11usize);
        let mut buf = vec![0.0f32; h * w];
        buf[5 * 11 + 5] = 99.0;
        box_mean_1ch(&mut buf, h, w, 2, 1);
        for y in 0..h {
            for x in 0..w {
                let inside = (5i32 - y as i32).abs() <= 2 && (5i32 - x as i32).abs() <= 2;
                if !inside {
                    assert_eq!(buf[y * 11 + x], 0.0, "bleed outside window at {x},{y}");
                } else {
                    assert!(buf[y * 11 + x] > 0.0, "missing glow at {x},{y}");
                }
            }
        }
        let total: f32 = buf.iter().sum();
        // Mass conservation holds *here* only because the impulse is central
        // with fully interior support — border windows drop taps under the
        // window∩bounds semantics, so it is not a global invariant of the
        // filter (e.g. row [0,0,0,8], r=1 sums to 6.67).
        assert!(
            (total - 99.0).abs() < 1e-2,
            "interior mean conserves mass: {total}"
        );
    }

    #[test]
    fn zero_radius_and_zero_iterations_are_noops() {
        let mut buf = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = buf.clone();
        box_mean_1ch(&mut buf, 2, 2, 0, 8);
        box_mean_1ch(&mut buf, 2, 2, 1, 0);
        assert_eq!(buf, orig);
    }
}
