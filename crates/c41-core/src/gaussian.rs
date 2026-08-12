//! Recursive (Young–van Vliet) Gaussian blur — a faithful port of the CPU path of
//! `src/common/gaussian.c` (`_compute_gauss_params` + `dt_gaussian_blur_4c`).
//! Shared infrastructure for the many IOPs that need a fast separable Gaussian
//! (bloom, highpass, lowpass, shadhi, hazeremoval, …).
//!
//! The recursive filter runs a forward (causal) + backward (anti-causal) IIR pass
//! along each axis — O(pixels) regardless of sigma, unlike a direct convolution.
//! Vertical pass first (into a temp buffer), then horizontal (into the output);
//! each axis sums its forward and backward passes. Values are clamped per channel
//! to `[min, max]` as they enter the recursion (darktable's `CLAMPF`).
//!
//! Ported: the RGBA (`blur_4c`) path. Not ported: the generic N-channel
//! `dt_gaussian_blur`, the direct `_fast_9x9` small-sigma path, and OpenCL.

/// darktable's `dt_gaussian_order_t` — which derivative of the Gaussian to apply.
/// `Zero` is the plain blur (what almost every caller uses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianOrder {
    Zero,
    One,
    Two,
}

struct Params {
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    b1: f32,
    b2: f32,
    coefp: f32,
    coefn: f32,
}

/// Van Vliet recursive-Gaussian coefficients from sigma (port of
/// `_compute_gauss_params`).
fn compute_params(sigma: f32, order: GaussianOrder) -> Params {
    let alpha = 1.695 / sigma;
    let ema = (-alpha).exp();
    let ema2 = (-2.0 * alpha).exp();
    let b1 = -2.0 * ema;
    let b2 = ema2;
    let (a0, a1, a2, a3) = match order {
        GaussianOrder::Zero => {
            let k = (1.0 - ema) * (1.0 - ema) / (1.0 + (2.0 * alpha * ema) - ema2);
            (k, k * (alpha - 1.0) * ema, k * (alpha + 1.0) * ema, -k * ema2)
        }
        GaussianOrder::One => {
            let a0 = (1.0 - ema) * (1.0 - ema);
            (a0, 0.0, -a0, 0.0)
        }
        GaussianOrder::Two => {
            let k = -(ema2 - 1.0) / (2.0 * alpha * ema);
            let mut kn = -2.0 * (-1.0 + (3.0 * ema) - (3.0 * ema * ema) + (ema * ema * ema));
            kn /= (3.0 * ema) + 1.0 + (3.0 * ema * ema) + (ema * ema * ema);
            (kn, -kn * (1.0 + (k * alpha)) * ema, kn * (1.0 - (k * alpha)) * ema, -kn * ema2)
        }
    };
    let denom = 1.0 + b1 + b2;
    Params { a0, a1, a2, a3, b1, b2, coefp: (a0 + a1) / denom, coefn: (a2 + a3) / denom }
}

/// Mirror darktable's `CLAMPF(a, mn, mx) = a >= mn ? (a <= mx ? a : mx) : mn`
/// (`src/common/math.h`). The `>=`/`<=` ordering is load-bearing: a NaN fails
/// `v >= lo` and maps to `lo`, exactly as the C macro does. The naive
/// `if v < lo … else if v > hi …` form would instead *propagate* a NaN, which in
/// the IIR recursion poisons the whole column/row (Rust is not `-ffast-math`).
#[inline]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v >= lo {
        if v <= hi { v } else { hi }
    } else {
        lo
    }
}

/// A recursive-Gaussian blur configured for one image size + sigma + clamp range.
pub struct Gaussian {
    width: usize,
    height: usize,
    sigma: f32,
    order: GaussianOrder,
    min: [f32; 4],
    max: [f32; 4],
    buf: Vec<f32>, // temp (width*height*4) — the vertical-pass result
}

impl Gaussian {
    /// Allocate for a `width × height` RGBA image (matches `dt_gaussian_init` with
    /// `channels == 4`). `min`/`max` clamp each channel entering the recursion.
    pub fn new(
        width: usize,
        height: usize,
        min: [f32; 4],
        max: [f32; 4],
        sigma: f32,
        order: GaussianOrder,
    ) -> Self {
        Self { width, height, sigma, order, min, max, buf: vec![0.0f32; width * height * 4] }
    }

    /// Blur a packed RGBA `f32` buffer (`in`) into `out` (both `width*height*4`) —
    /// the RGBA recursive Gaussian (port of `dt_gaussian_blur_4c`). `in` and `out`
    /// may be the same buffer's contents conceptually, but are separate slices here.
    pub fn blur_4c(&mut self, input: &[f32], out: &mut [f32]) {
        if self.width == 0 || self.height == 0 {
            return; // C reads benign OOB on an empty image; Rust would panic.
        }
        let n = self.width * self.height * 4;
        debug_assert_eq!(input.len(), n, "input must be width*height*4");
        debug_assert_eq!(out.len(), n, "out must be width*height*4");
        let p = compute_params(self.sigma, self.order);
        let (w, h) = (self.width, self.height);
        let (mn, mx) = (self.min, self.max);
        let temp = &mut self.buf;

        // ── vertical blur, column by column ──────────────────────────────────
        for i in 0..w {
            // forward filter (top → bottom)
            let mut xp = [0.0f32; 4];
            let mut yb = [0.0f32; 4];
            let mut yp = [0.0f32; 4];
            for k in 0..4 {
                xp[k] = clampf(input[4 * i + k], mn[k], mx[k]);
                yb[k] = xp[k] * p.coefp;
                yp[k] = yb[k];
            }
            for j in 0..h {
                let offset = 4 * (j * w + i);
                for k in 0..4 {
                    let xc = clampf(input[offset + k], mn[k], mx[k]);
                    let yc = p.a0 * xc + p.a1 * xp[k] - p.b1 * yp[k] - p.b2 * yb[k];
                    xp[k] = xc;
                    yb[k] = yp[k];
                    yp[k] = yc;
                    temp[offset + k] = yc;
                }
            }
            // backward filter (bottom → top), accumulated into temp
            let mut xn = [0.0f32; 4];
            let mut xa = [0.0f32; 4];
            let mut yn = [0.0f32; 4];
            let mut ya = [0.0f32; 4];
            for k in 0..4 {
                xn[k] = clampf(input[4 * ((h - 1) * w + i) + k], mn[k], mx[k]);
                xa[k] = xn[k];
                yn[k] = xn[k] * p.coefn;
                ya[k] = yn[k];
            }
            for j in (0..h).rev() {
                let offset = 4 * (j * w + i);
                for k in 0..4 {
                    let xc = clampf(input[offset + k], mn[k], mx[k]);
                    let yc = p.a2 * xn[k] + p.a3 * xa[k] - p.b1 * yn[k] - p.b2 * ya[k];
                    xa[k] = xn[k];
                    xn[k] = xc;
                    ya[k] = yn[k];
                    yn[k] = yc;
                    temp[offset + k] += yc;
                }
            }
        }

        // ── horizontal blur, line by line (temp → out) ───────────────────────
        for j in 0..h {
            // forward filter (left → right)
            let mut xp = [0.0f32; 4];
            let mut yb = [0.0f32; 4];
            let mut yp = [0.0f32; 4];
            for k in 0..4 {
                xp[k] = clampf(temp[4 * (j * w) + k], mn[k], mx[k]);
                yb[k] = xp[k] * p.coefp;
                yp[k] = yb[k];
            }
            for i in 0..w {
                let offset = 4 * (j * w + i);
                for k in 0..4 {
                    let xc = clampf(temp[offset + k], mn[k], mx[k]);
                    let yc = p.a0 * xc + p.a1 * xp[k] - p.b1 * yp[k] - p.b2 * yb[k];
                    out[offset + k] = yc;
                    xp[k] = xc;
                    yb[k] = yp[k];
                    yp[k] = yc;
                }
            }
            // backward filter (right → left), accumulated into out
            let mut xn = [0.0f32; 4];
            let mut xa = [0.0f32; 4];
            let mut yn = [0.0f32; 4];
            let mut ya = [0.0f32; 4];
            for k in 0..4 {
                xn[k] = clampf(temp[4 * ((j + 1) * w - 1) + k], mn[k], mx[k]);
                xa[k] = xn[k];
                yn[k] = xn[k] * p.coefn;
                ya[k] = yn[k];
            }
            for i in (0..w).rev() {
                let offset = 4 * (j * w + i);
                for k in 0..4 {
                    let xc = clampf(temp[offset + k], mn[k], mx[k]);
                    let yc = p.a2 * xn[k] + p.a3 * xa[k] - p.b1 * yn[k] - p.b2 * ya[k];
                    xa[k] = xn[k];
                    xn[k] = xc;
                    ya[k] = yn[k];
                    yn[k] = yc;
                    out[offset + k] += yc;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDE: [f32; 4] = [1e9, 1e9, 1e9, 1e9];
    const LO: [f32; 4] = [-1e9, -1e9, -1e9, -1e9];

    fn rgba(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h * 4];
        for j in 0..h {
            for i in 0..w {
                let val = f(i, j);
                let o = 4 * (j * w + i);
                for k in 0..4 { v[o + k] = val; }
            }
        }
        v
    }

    fn blur(inp: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
        let mut g = Gaussian::new(w, h, LO, WIDE, sigma, GaussianOrder::Zero);
        let mut out = vec![0.0f32; w * h * 4];
        g.blur_4c(inp, &mut out);
        out
    }

    #[test]
    fn flat_field_is_preserved() {
        // A constant image must stay constant (a normalised blur of a constant is
        // the constant — checks the recursion's DC gain is 1).
        let (w, h) = (24usize, 20usize);
        let inp = rgba(w, h, |_, _| 0.7);
        let out = blur(&inp, w, h, 3.0);
        for j in 2..h - 2 {
            for i in 2..w - 2 {
                let v = out[4 * (j * w + i)];
                assert!((v - 0.7).abs() < 2e-3, "flat blur drifted at ({i},{j}): {v}");
            }
        }
    }

    #[test]
    fn impulse_response_matches_an_analytic_gaussian() {
        // THE faithfulness test: a correct recursive Gaussian's impulse response
        // must approximate exp(-r²/2σ²)/(2πσ²). This catches a transposed
        // recursion variable / wrong coefficient that a "still blurry" behavioural
        // test would miss.
        let (w, h) = (41usize, 41usize);
        let (cx, cy) = (20usize, 20usize);
        let sigma = 4.0f32;
        let inp = rgba(w, h, |i, j| if i == cx && j == cy { 1.0 } else { 0.0 });
        let out = blur(&inp, w, h, sigma);

        let norm = 1.0 / (2.0 * std::f32::consts::PI * sigma * sigma);
        let mut max_err = 0.0f32;
        let mut sum = 0.0f32;
        for j in 0..h {
            for i in 0..w {
                let got = out[4 * (j * w + i)];
                sum += got;
                let dx = i as f32 - cx as f32;
                let dy = j as f32 - cy as f32;
                let want = norm * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                max_err = max_err.max((got - want).abs());
            }
        }
        // The van Vliet approximation is close to a true Gaussian; peak ≈ 0.0099.
        assert!(max_err < 1.5e-3, "impulse response off a true Gaussian: max_err={max_err}");
        // Energy is (approximately) preserved — a normalised blur of a unit impulse
        // integrates to ~1.
        assert!((sum - 1.0).abs() < 0.02, "impulse energy not preserved: {sum}");
    }

    #[test]
    fn impulse_response_is_symmetric() {
        // Separable + symmetric filter ⇒ the impulse response is symmetric in x and
        // y (a swapped forward/backward init would break this).
        let (w, h) = (31usize, 31usize);
        let (c, sigma) = (15usize, 3.0f32);
        let inp = rgba(w, h, |i, j| if i == c && j == c { 1.0 } else { 0.0 });
        let out = blur(&inp, w, h, sigma);
        let at = |i: usize, j: usize| out[4 * (j * w + i)];
        for d in 1..=6usize {
            assert!((at(c + d, c) - at(c - d, c)).abs() < 1e-5, "x-asymmetry at d={d}");
            assert!((at(c, c + d) - at(c, c - d)).abs() < 1e-5, "y-asymmetry at d={d}");
            assert!((at(c + d, c) - at(c, c + d)).abs() < 1e-5, "x/y-asymmetry at d={d}");
        }
        assert!(at(c, c) > at(c + 1, c), "not peaked at the impulse");
    }

    #[test]
    fn clamp_bounds_are_honoured() {
        // min/max clamp values as they enter the recursion: a spike well above max
        // is clamped, so the blurred field can't exceed max.
        let (w, h) = (20usize, 20usize);
        let inp = rgba(w, h, |i, j| if i == 10 && j == 10 { 5.0 } else { 0.1 });
        let mut g = Gaussian::new(w, h, [0.0; 4], [1.0; 4], 2.0, GaussianOrder::Zero);
        let mut out = vec![0.0f32; w * h * 4];
        g.blur_4c(&inp, &mut out);
        for v in out.iter() {
            assert!(*v <= 1.0 + 1e-4, "exceeded max clamp: {v}");
        }
    }

    #[test]
    fn params_are_finite_for_a_range_of_sigmas() {
        // Exercise all three orders — the One/Two arms (kn rational, k*alpha
        // cross-terms) would otherwise never execute since callers use Zero.
        for order in [GaussianOrder::Zero, GaussianOrder::One, GaussianOrder::Two] {
            for &s in &[0.8f32, 1.5, 3.0, 10.0, 50.0] {
                let p = compute_params(s, order);
                for v in [p.a0, p.a1, p.a2, p.a3, p.b1, p.b2, p.coefp, p.coefn] {
                    assert!(v.is_finite(), "non-finite param at sigma {s} ({order:?})");
                }
            }
        }
    }

    #[test]
    fn all_orders_produce_finite_output() {
        let (w, h) = (16usize, 16usize);
        let inp = rgba(w, h, |i, j| if i == 8 && j == 8 { 1.0 } else { 0.0 });
        for order in [GaussianOrder::Zero, GaussianOrder::One, GaussianOrder::Two] {
            let mut g = Gaussian::new(w, h, LO, WIDE, 3.0, order);
            let mut out = vec![0.0f32; w * h * 4];
            g.blur_4c(&inp, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "non-finite output ({order:?})");
        }
    }

    #[test]
    fn nan_is_clamped_to_min_like_c() {
        // C's CLAMPF scrubs a NaN input to `min` on read, keeping it local; a
        // NaN-propagating clamp would smear NaN across the whole image via the IIR.
        let (w, h) = (8usize, 8usize);
        let inp = rgba(w, h, |i, j| if i == 4 && j == 4 { f32::NAN } else { 0.5 });
        let mut g = Gaussian::new(w, h, [0.0; 4], [1.0; 4], 2.0, GaussianOrder::Zero);
        let mut out = vec![0.0f32; w * h * 4];
        g.blur_4c(&inp, &mut out);
        assert!(out.iter().all(|v| v.is_finite()), "NaN leaked through the blur");
    }

    #[test]
    fn channels_do_not_bleed() {
        // All other tests use channel-uniform data; a [k]-indexing bug mixing
        // channels would pass them. Distinct per-channel constants must survive.
        let (w, h) = (16usize, 16usize);
        let mut inp = vec![0.0f32; w * h * 4];
        for p in inp.chunks_mut(4) {
            p.copy_from_slice(&[0.2, 0.4, 0.6, 0.8]);
        }
        let mut g = Gaussian::new(w, h, LO, WIDE, 2.0, GaussianOrder::Zero);
        let mut out = vec![0.0f32; w * h * 4];
        g.blur_4c(&inp, &mut out);
        let c = 4 * (8 * w + 8);
        for (k, want) in [0.2f32, 0.4, 0.6, 0.8].iter().enumerate() {
            assert!((out[c + k] - want).abs() < 5e-3, "channel {k} bled: {}", out[c + k]);
        }
    }

    #[test]
    fn degenerate_dims_do_not_panic() {
        // darktable's fast-math C survives benign OOB reads on size-1 axes; Rust
        // bounds-checks and would panic. Confirm 1×1 / 1×N / N×1 / 2×2 are safe
        // (the offsets stay in range because each pass reads only its own axis).
        for (w, h) in [(1usize, 1usize), (1, 17), (17, 1), (2, 2)] {
            let inp = vec![0.5f32; w * h * 4];
            let mut g = Gaussian::new(w, h, LO, WIDE, 3.0, GaussianOrder::Zero);
            let mut out = vec![0.0f32; w * h * 4];
            g.blur_4c(&inp, &mut out);
        }
    }
}
