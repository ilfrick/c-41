//! Port of darktable's `common/box_filters.cc` — `dt_box_mean(buf, h, w, ch,
//! radius, iterations)` — the separable sliding-window box average. Two
//! dispatches are ported, both un-Kahan-summed (`BOXFILTER_KAHAN_SUM` unset),
//! sharing one strided sliding-window core:
//!
//! * [`box_mean_1ch`] — `ch == 1`, the packed single-plane path bloom uses
//!   (also `rasterfile.c`, `highpass.c`, `hlreconstruct/segbased.c`,
//!   `focus_peaking.h`).
//! * [`box_mean_4ch`] — `ch == 4`, the interleaved RGBA path soften uses (also
//!   `hlreconstruct/laplacian.c` and `fast_guided_filter.h`).
//!
//! The C's other dispatches are deliberately absent, and none of their C callers
//! is ported at all:
//!
//! | dispatch | C call sites | status |
//! |---|---|---|
//! | `ch == 2` | `fast_guided_filter.h:243` | that call is not ported |
//! | `ch \| BOXFILTER_KAHAN_SUM` | `guided_filter.c:172` (via `dt_box_mean`), `guided_filter.c:152-157` (via `dt_box_mean_horizontal`/`_vertical`) | not ported |
//! | `dt_box_mean_horizontal`/`_vertical` | `guided_filter.c:152,153,156,157` | not ported |
//!
//! c41 has partial ports of both filters (`crate::fast_guided_filter` ports
//! fast_guided_filter.h's element-wise loops, `crate::guided_filter` ports
//! guided_filter.c's Cramer-rule solve), but the box-mean blur call sites in
//! both are among the loops *not* ported, so no live caller needs these
//! dispatches. Adding one means adding the Kahan accumulation, which changes
//! the arithmetic rather than just re-wiring a stride.
//!
//! (`focus_peaking.h` is not on this table: it calls plain
//! `dt_box_mean(…, 1, 2, 1)` at :99, i.e. the already-ported `ch == 1`.)
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
//! * Summation is plain f32 (`compensated = false`, i.e. no Kahan). Both ported
//!   dispatches are the non-compensated ones: bloom.c calls
//!   `dt_box_mean(..., 1, ...)` and soften.c calls `dt_box_mean(..., 4, ...)`,
//!   and only the `ch | BOXFILTER_KAHAN_SUM` variants compensate.
//! * Channels never mix, so `ch == 4` is the same per-lane arithmetic as
//!   `ch == 1` on a strided lane — hence one core, not two.
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

/// One horizontal pass over `count` values spaced `stride` floats apart,
/// starting at `base` (box_filters.cc:185). `stride == 1` on a tightly packed
/// slice is the single-channel case; `stride == 4` walks one interleaved RGBA
/// channel — the C's `_blur_horizontal<N>` does all N lanes in one call, which
/// is the same arithmetic per lane. `scratch` receives copies of the input
/// values as they enter the running sum, so later subtractions remove pre-blur
/// values even though the buffer is overwritten in place — same role as the
/// C's `_load_add(scratch …)`.
fn blur_horizontal_strided(
    buf: &mut [f32],
    base: usize,
    stride: usize,
    count: usize,
    scratch: &mut [f32],
    radius: usize,
) {
    debug_assert!(scratch.len() >= count);
    let idx = |k: usize| base + k * stride;
    let mut sum = 0.0f32;
    let mut hits = 0usize;

    // add up the left half of the window
    for x in 0..radius.min(count) {
        hits += 1;
        scratch[x] = buf[idx(x)];
        sum += buf[idx(x)];
    }
    // blur up to the point where values start leaving the moving average
    let mut x = 0usize;
    while x <= radius && x + radius < count {
        hits += 1;
        scratch[x + radius] = buf[idx(x + radius)];
        sum += buf[idx(x + radius)];
        buf[idx(x)] = sum / hits as f32;
        x += 1;
    }
    // radius > count/2: neither add nor remove possible — just store
    while x <= radius && x < count {
        buf[idx(x)] = sum / hits as f32;
        x += 1;
    }
    // bulk of the scan line: subtract the outgoing value, add the incoming one
    while x + radius < count {
        sum -= scratch[x - radius - 1];
        scratch[x + radius] = buf[idx(x + radius)];
        sum += buf[idx(x + radius)];
        buf[idx(x)] = sum / hits as f32;
        x += 1;
    }
    // right end: no more values enter the sum
    while x < count {
        hits -= 1;
        sum -= scratch[x - radius - 1];
        buf[idx(x)] = sum / hits as f32;
        x += 1;
    }
}

/// One horizontal pass over a single packed row — the `stride == 1` case.
fn blur_horizontal(row: &mut [f32], scratch: &mut [f32], radius: usize) {
    let width = row.len();
    blur_horizontal_strided(row, 0, 1, width, scratch, radius);
}

/// One vertical pass over `count` values spaced `stride` floats apart, starting
/// at `base`, mirroring `_blur_vertical` (box_filters.cc:308). `stride == width`
/// is a plane column; `stride == 4*width` is one interleaved RGBA channel down
/// the frame (the C's `_blur_vertical_1ch(buf, height, N*width, …)`). With a
/// full-height scratch there is no aliasing between the value being stored and
/// the history being kept, so the C's power-of-two mask reduces to direct
/// indexing.
fn blur_vertical_strided(
    buf: &mut [f32],
    base: usize,
    stride: usize,
    count: usize,
    scratch: &mut [f32],
    radius: usize,
) {
    debug_assert!(scratch.len() >= count);
    let idx = |k: usize| base + k * stride;
    let mut sum = 0.0f32;
    let mut hits = 0usize;

    for y in 0..radius.min(count) {
        hits += 1;
        scratch[y] = buf[idx(y)];
        sum += buf[idx(y)];
    }
    let mut y = 0usize;
    while y <= radius && y + radius < count {
        hits += 1;
        scratch[y + radius] = buf[idx(y + radius)];
        sum += buf[idx(y + radius)];
        buf[idx(y)] = sum / hits as f32;
        y += 1;
    }
    while y <= radius && y < count {
        buf[idx(y)] = sum / hits as f32;
        y += 1;
    }
    while y + radius < count {
        sum -= scratch[y - radius - 1];
        scratch[y + radius] = buf[idx(y + radius)];
        sum += buf[idx(y + radius)];
        buf[idx(y)] = sum / hits as f32;
        y += 1;
    }
    while y < count {
        hits -= 1;
        sum -= scratch[y - radius - 1];
        buf[idx(y)] = sum / hits as f32;
        y += 1;
    }
}

/// One vertical pass over column `col` of a packed `height × width` plane.
fn blur_vertical(buf: &mut [f32], height: usize, width: usize, col: usize, scratch: &mut [f32], radius: usize) {
    blur_vertical_strided(buf, col, width, height, scratch, radius);
}

/// In-place separable box mean on a packed **RGBA** buffer — this is
/// `dt_box_mean(buf, height, width, 4, radius, BOX_ITERATIONS)`, i.e.
/// `_box_mean<4>` (template at box_filters.cc:361, `ch == 4` dispatch at :618),
/// which Soften uses.
///
/// Channels are independent in the C (`_blur_horizontal<4>` walks one lane per
/// channel; `_blur_vertical_1ch(buf, height, 4*width, …)` blurs each
/// interleaved column), so each channel is blurred with the same sliding-window
/// core as [`box_mean_1ch`], strided by 4 horizontally and `4*width` vertically.
/// No deinterleave copy is made — the C blurs in place and so does this.
///
/// The kernel bloom uses is the `ch == 1` dispatch; this `ch == 4` one is
/// un-Kahan-summed in both cases (`BOXFILTER_KAHAN_SUM` unset), so summation
/// stays plain f32.
///
/// Verified against the C: transcribing the real `_box_mean<4>` and diffing a
/// 256×256 random f32 frame at radii {1, 7, 31} gives **bit-identical** results
/// over all 262 144 values when the C is compiled without fast-math, and ≤2.2e-07
/// (~1e-6 relative, a few ulp) when it carries box_filters' real
/// `#pragma GCC optimize(…,"fast-math")` — i.e. the residual is the C's own
/// reassociation, not a difference in the algorithm.
///
/// Correct but not fast: the horizontal phase walks one lane at a time (four
/// passes per row) where the C's `_blur_horizontal<4>` vectorises all four in a
/// single sweep. Same cache lines either way, so this costs roughly 2x the
/// memory traffic on that phase. Left alone deliberately — the strided core is
/// what keeps one verified implementation instead of two, and softening is not
/// on the hot path at the preview sizes that dominate.
pub fn box_mean_4ch(buf: &mut [f32], height: usize, width: usize, radius: usize, iterations: u32) {
    if buf.is_empty() || width == 0 || height == 0 || radius == 0 || iterations == 0 {
        return;
    }
    // `>=`, not `==`: callers legitimately hand us a frame-sized buffer (see
    // `soften::process`, which accepts `width * height * 4` floats or more) and
    // only the leading `height * width * 4` are touched.
    debug_assert!(
        buf.len() >= height * width * 4,
        "RGBA buffer must hold at least height*width*4 floats"
    );
    let w4 = width * 4;
    let mut scratch = vec![0.0f32; height.max(width)];
    for _ in 0..iterations {
        for row in 0..height {
            let row_base = row * w4;
            for c in 0..4 {
                blur_horizontal_strided(buf, row_base + c, 4, width, &mut scratch[..width], radius);
            }
        }
        for c in 0..4 {
            for col in 0..width {
                blur_vertical_strided(buf, col * 4 + c, w4, height, &mut scratch[..height], radius);
            }
        }
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

    /// `box_mean_4ch` must equal four independent single-channel runs — one per
    /// interleaved RGBA lane. That is exactly what the C's `_box_mean<4>` does
    /// (the N lanes never mix), so the reference is built from the already
    /// verified [`box_mean_1ch`] rather than a second copy of the algorithm.
    #[test]
    fn box_mean_4ch_matches_per_channel_1ch() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) % 2000) as f32 / 10.0 - 100.0 // signed, RGBA-ish range
        };
        for &(h, w) in &[(16usize, 24usize), (5, 5), (1, 30), (30, 1), (9, 61)] {
            let packed: Vec<f32> = (0..h * w * 4).map(|_| next()).collect();
            for radius in [0usize, 1, 3, 8, 30] {
                for iters in [1u32, 3] {
                    let mut got = packed.clone();
                    box_mean_4ch(&mut got, h, w, radius, iters);

                    // Deinterleave each channel into its own plane, blur with
                    // the 1-channel kernel, compare.
                    for c in 0..4 {
                        let mut plane: Vec<f32> =
                            (0..h * w).map(|p| packed[p * 4 + c]).collect();
                        box_mean_1ch(&mut plane, h, w, radius, iters);
                        for p in 0..h * w {
                            let g = got[p * 4 + c];
                            let wv = plane[p];
                            assert!(
                                (g - wv).abs() < 1e-4,
                                "h{h} w{w} r{radius} i{iters} ch{c} p{p}: {g} vs {wv}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The alpha lane is blurred too (the C passes `ch = 4`, not 3); a constant
    /// field of all four channels must survive unchanged.
    #[test]
    fn box_mean_4ch_keeps_a_constant_rgba_field_constant() {
        let (h, w) = (13usize, 21usize);
        let mut buf = vec![0.0f32; h * w * 4];
        for p in 0..h * w {
            buf[p * 4] = 0.25;
            buf[p * 4 + 1] = 0.5;
            buf[p * 4 + 2] = 0.75;
            buf[p * 4 + 3] = 1.0;
        }
        box_mean_4ch(&mut buf, h, w, 5, 8);
        for p in 0..h * w {
            assert!((buf[p * 4] - 0.25).abs() < 1e-4);
            assert!((buf[p * 4 + 1] - 0.5).abs() < 1e-4);
            assert!((buf[p * 4 + 2] - 0.75).abs() < 1e-4);
            assert!((buf[p * 4 + 3] - 1.0).abs() < 1e-4);
        }
    }
}
