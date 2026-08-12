//! 3-D bilateral grid (Chen/Paris/Durand), a faithful port of
//! `src/common/bilateral.c`. Shared edge-aware-filter infrastructure used by the
//! `lowpass`, `shadhi`, `retouch`, `monochrome`, `globaltonemap`, `colormapping`,
//! `ashift`, and `bilat` IOPs. (NOTE: `colorreconstruction` does NOT use this —
//! it carries its own bespoke 4-field {L,a,b,weight} grid with a different index
//! convention; that's a separate port.)
//!
//! Pipeline: [`Bilateral::splat`] scatters each pixel's L (channel 0 of a packed
//! RGBA buffer) into a coarse `size_x × size_y × size_z` grid with trilinear
//! weights; [`Bilateral::blur`] runs a separable Gaussian over x and y plus a
//! derivative filter over z; [`Bilateral::slice`] reads the blurred grid back
//! with trilinear interpolation and mixes it into L by `detail` (−1 = bilateral
//! smooth, +1 = local-contrast boost).
//!
//! **Serial vs the C:** the C splat shards rows across threads into per-thread
//! grid slices, then merges by addition. That is a write-contention workaround —
//! summing every pixel's trilinear contributions into ONE grid (as here) writes
//! the same additive payloads to the same cells, just without the slice
//! bookkeeping. Equivalent up to floating-point add order (the C's own result is
//! thread-count-dependent, since float addition isn't associative — the serial
//! sum is arguably the more canonical reference).
//!
//! Grid stride convention matches the C: `ox = size_z`, `oy = size_x*size_z`,
//! `oz = 1` (z fastest, then x, then y).

/// The clamps the C uses to bound insane grid memory (the grid stays a faithful
/// approximation; tiling/scale handle the rest).
const MAX_RES_S: i32 = 3000;
const MAX_RES_R: i32 = 50;
/// The L range the grid spans (Lab L is [0,100]).
const L_RANGE: f32 = 100.0;

/// A bilateral grid sized for one image + its sigmas.
pub struct Bilateral {
    size_x: usize,
    size_y: usize,
    size_z: usize,
    sigma_s: f32,
    sigma_r: f32,
    sigma_s_inv: f32,
    sigma_r_inv: f32,
    width: usize,
    height: usize,
    buf: Vec<f32>,
}

// Matches C CLAMPS = MIN(MAX(v,lo),hi). A NaN `v` passes through (then `as usize`
// saturates to 0) — an intentional divergence from C's `(int)NaN` UB, harmless
// under the fast-math "no NaN" assumption the grid runs on.
#[inline]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

impl Bilateral {
    /// Allocate a grid for a `width × height` image (matches `dt_bilateral_init`
    /// and `dt_bilateral_grid_size`). `sigma_s` is the spatial blur (pixel
    /// coords); `sigma_r` the range blur (L values).
    pub fn new(width: usize, height: usize, sigma_s: f32, sigma_r: f32) -> Self {
        // sigma_s < 0.5 would make the grid larger than the image with mostly
        // unused points — the C floors it at 0.5.
        let sigma_s = sigma_s.max(0.5);
        let clampi = |v: f32, lo: i32, hi: i32| (v.round() as i32).clamp(lo, hi) as f32;
        let _x = clampi(width as f32 / sigma_s, 4, MAX_RES_S);
        let _y = clampi(height as f32 / sigma_s, 4, MAX_RES_S);
        let _z = clampi(L_RANGE / sigma_r, 4, MAX_RES_R);
        // Effective sigma_s once the dims were (possibly) clamped — one value for
        // both spatial axes, per the C.
        let sigma_s = (height as f32 / _y).max(width as f32 / _x);
        let sigma_r = L_RANGE / _z;
        let sigma_s_inv = 1.0 / sigma_s;
        let sigma_r_inv = 1.0 / sigma_r;
        let size_x = (width as f32 * sigma_s_inv).ceil() as usize + 1;
        let size_y = (height as f32 * sigma_s_inv).ceil() as usize + 1;
        let size_z = (L_RANGE * sigma_r_inv).ceil() as usize + 1;
        Self {
            size_x, size_y, size_z,
            sigma_s, sigma_r, sigma_s_inv, sigma_r_inv,
            width, height,
            buf: vec![0.0f32; size_x * size_y * size_z],
        }
    }

    /// Grid → 8-neighbour trilinear index + fractions (matches `image_to_grid`).
    #[inline]
    fn image_to_grid(&self, i: usize, j: usize, l: f32) -> (usize, f32, f32, f32) {
        let x = clampf(i as f32 * self.sigma_s_inv, 0.0, (self.size_x - 1) as f32);
        let y = clampf(j as f32 * self.sigma_s_inv, 0.0, (self.size_y - 1) as f32);
        let z = clampf(l * self.sigma_r_inv, 0.0, (self.size_z - 1) as f32);
        let xi = (x as usize).min(self.size_x - 2);
        let yi = (y as usize).min(self.size_y - 2);
        let zi = (z as usize).min(self.size_z - 2);
        let gi = ((xi + yi * self.size_x) * self.size_z) + zi;
        (gi, x - xi as f32, y - yi as f32, z - zi as f32)
    }

    /// Scatter each pixel's L into the grid (matches `dt_bilateral_splat`, serial).
    /// `input` is packed RGBA `f32` (`width*height*4`); L is channel 0.
    pub fn splat(&mut self, input: &[f32]) {
        let ox = self.size_z;
        let oy = self.size_x * self.size_z;
        let oz = 1usize;
        let offsets = [0, ox, oy, ox + oy, oz, oz + ox, oz + oy, oz + ox + oy];
        // payload weight: 100 / sigma_s² (C: sigma_s = b->sigma_s²; 100/sigma_s).
        let payload = 100.0 / (self.sigma_s * self.sigma_s);
        for j in 0..self.height {
            let y = clampf(j as f32 * self.sigma_s_inv, 0.0, (self.size_y - 1) as f32);
            let yi = (y as usize).min(self.size_y - 2);
            let yf = y - yi as f32;
            let base = yi * oy;
            for i in 0..self.width {
                let l = input[4 * (j * self.width + i)];
                // relative (x,z) grid cell for this pixel.
                let x = clampf(i as f32 * self.sigma_s_inv, 0.0, (self.size_x - 1) as f32);
                let z = clampf(l * self.sigma_r_inv, 0.0, (self.size_z - 1) as f32);
                let xi = (x as usize).min(self.size_x - 2);
                let zi = (z as usize).min(self.size_z - 2);
                let xf = x - xi as f32;
                let zf = z - zi as f32;
                let grid_index = base + xi * self.size_z + zi;
                let contrib = [
                    (1.0 - xf) * (1.0 - yf) * payload,
                    xf * (1.0 - yf) * payload,
                    (1.0 - xf) * yf * payload,
                    xf * yf * payload,
                ];
                for k in 0..4 {
                    self.buf[grid_index + offsets[k]] += contrib[k] * (1.0 - zf);
                    self.buf[grid_index + offsets[k + 4]] += contrib[k] * zf;
                }
            }
        }
    }

    /// Separable Gaussian over x, y + a derivative filter over z (matches
    /// `dt_bilateral_blur`).
    pub fn blur(&mut self) {
        let ox = self.size_z;
        let oy = self.size_x * self.size_z;
        let oz = 1usize;
        // gaussian along x, then y
        blur_line(&mut self.buf, oz, oy, ox, self.size_z, self.size_y, self.size_x);
        blur_line(&mut self.buf, oz, ox, oy, self.size_z, self.size_x, self.size_y);
        // −2nd-derivative of the gaussian along z: x·exp(−x²)
        blur_line_z(&mut self.buf, ox, oy, oz, self.size_x, self.size_y, self.size_z);
    }

    /// Trilinear read-back of the blurred grid, mixed into L by `detail` (matches
    /// `dt_bilateral_slice`). `detail`: 0 = unchanged, −1 = bilateral smooth,
    /// +1 = contrast boost. `output` is packed RGBA (colour/alpha copied from
    /// `input`, only L updated); `input`/`output` are `width*height*4`.
    pub fn slice(&self, input: &[f32], output: &mut [f32], detail: f32) {
        let norm = -detail * self.sigma_r * 0.04;
        for j in 0..self.height {
            for i in 0..self.width {
                let index = 4 * (j * self.width + i);
                let l = input[index];
                let (gi, xf, yf, zf) = self.image_to_grid(i, j, l);
                output[index..index + 4].copy_from_slice(&input[index..index + 4]);
                output[index] = (l + norm * self.interp(gi, xf, yf, zf)).max(0.0);
            }
        }
    }

    /// Like [`Bilateral::slice`] but **accumulates** into L and does NOT copy the
    /// colour/alpha channels: `out[L] = max(0, out[L] + norm·interp)` (matches
    /// `dt_bilateral_slice_to_output`; used by `ashift` / `globaltonemap`).
    /// `output` must already hold the buffer being accumulated into.
    pub fn slice_to_output(&self, input: &[f32], output: &mut [f32], detail: f32) {
        let norm = -detail * self.sigma_r * 0.04;
        for j in 0..self.height {
            for i in 0..self.width {
                let index = 4 * (j * self.width + i);
                let l = input[index];
                let (gi, xf, yf, zf) = self.image_to_grid(i, j, l);
                output[index] = (output[index] + norm * self.interp(gi, xf, yf, zf)).max(0.0);
            }
        }
    }

    /// 8-tap trilinear read of the blurred grid at cell `gi` with fractions
    /// `(xf,yf,zf)`. Shared by [`Bilateral::slice`] and
    /// [`Bilateral::slice_to_output`] so they can't drift.
    #[inline]
    fn interp(&self, gi: usize, xf: f32, yf: f32, zf: f32) -> f32 {
        let ox = self.size_z;
        let oy = self.size_x * self.size_z;
        let oz = 1usize;
        self.buf[gi] * (1.0 - xf) * (1.0 - yf) * (1.0 - zf)
            + self.buf[gi + ox] * xf * (1.0 - yf) * (1.0 - zf)
            + self.buf[gi + oy] * (1.0 - xf) * yf * (1.0 - zf)
            + self.buf[gi + ox + oy] * xf * yf * (1.0 - zf)
            + self.buf[gi + oz] * (1.0 - xf) * (1.0 - yf) * zf
            + self.buf[gi + ox + oz] * xf * (1.0 - yf) * zf
            + self.buf[gi + oy + oz] * (1.0 - xf) * yf * zf
            + self.buf[gi + ox + oy + oz] * xf * yf * zf
    }
}

/// Separable `[1 4 6 4 1]/16` Gaussian along the `offset3` axis (`size3` elements),
/// for each of `size1 × size2` lines. In-place, running-buffer boundary handling —
/// a faithful port of the C `blur_line`.
fn blur_line(
    buf: &mut [f32],
    offset1: usize,
    offset2: usize,
    offset3: usize,
    size1: usize,
    size2: usize,
    size3: usize,
) {
    // <4 grid points: the boundary block reads `buf[index + 2*offset3]`, which is
    // out of the line. C survives that as a benign OOB heap read (fast-math UB);
    // Rust would panic. The filter is degenerate on so few points anyway, so make
    // it a defined no-op (only reachable for extreme aspect ratios where a spatial
    // axis collapses to 2 — e.g. a 4px-wide crop).
    if size3 < 4 {
        return;
    }
    let (w0, w1, w2) = (6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0);
    for k in 0..size1 {
        for j in 0..size2 {
            // Line start (the C runs a single accumulator with a signed
            // `+= offset2 - offset3*size3` fixup; recomputing per line is
            // equivalent and avoids a usize underflow when offset2 < offset3*size3).
            let mut index = k * offset1 + j * offset2;
            let mut tmp1 = buf[index];
            buf[index] = buf[index] * w0 + w1 * buf[index + offset3] + w2 * buf[index + 2 * offset3];
            index += offset3;
            let mut tmp2 = buf[index];
            buf[index] =
                buf[index] * w0 + w1 * (buf[index + offset3] + tmp1) + w2 * buf[index + 2 * offset3];
            index += offset3;
            for _i in 2..size3 - 2 {
                let tmp3 = buf[index];
                buf[index] = buf[index] * w0
                    + w1 * (buf[index + offset3] + tmp2)
                    + w2 * (buf[index + 2 * offset3] + tmp1);
                index += offset3;
                tmp1 = tmp2;
                tmp2 = tmp3;
            }
            let tmp3 = buf[index];
            buf[index] = buf[index] * w0 + w1 * (buf[index + offset3] + tmp2) + w2 * tmp1;
            index += offset3;
            buf[index] = buf[index] * w0 + w1 * tmp3 + w2 * tmp2;
        }
    }
}

/// `−2nd`-derivative-of-gaussian filter along `offset3` (the z axis), for each of
/// `size1 × size2` lines. Faithful port of the C `blur_line_z`.
fn blur_line_z(
    buf: &mut [f32],
    offset1: usize,
    offset2: usize,
    offset3: usize,
    size1: usize,
    size2: usize,
    size3: usize,
) {
    if size3 < 4 {
        return; // see blur_line: guards the OOB read on a collapsed axis
    }
    let (w1, w2) = (4.0 / 16.0, 2.0 / 16.0);
    for k in 0..size1 {
        for j in 0..size2 {
            let mut index = k * offset1 + j * offset2; // per-line start (see blur_line)
            let mut tmp1 = buf[index];
            buf[index] = w1 * buf[index + offset3] + w2 * buf[index + 2 * offset3];
            index += offset3;
            let mut tmp2 = buf[index];
            buf[index] = w1 * (buf[index + offset3] - tmp1) + w2 * buf[index + 2 * offset3];
            index += offset3;
            for _i in 2..size3 - 2 {
                let tmp3 = buf[index];
                buf[index] =
                    w1 * (buf[index + offset3] - tmp2) + w2 * (buf[index + 2 * offset3] - tmp1);
                index += offset3;
                tmp1 = tmp2;
                tmp2 = tmp3;
            }
            let tmp3 = buf[index];
            buf[index] = w1 * (buf[index + offset3] - tmp2) - w2 * tmp1;
            index += offset3;
            buf[index] = -w1 * tmp3 - w2 * tmp2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a packed-RGBA buffer from an L-per-pixel closure (colour = L, α = 1).
    fn img(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h * 4];
        for j in 0..h {
            for i in 0..w {
                let l = f(i, j);
                let p = 4 * (j * w + i);
                v[p] = l; v[p + 1] = l; v[p + 2] = l; v[p + 3] = 1.0;
            }
        }
        v
    }

    fn filter(input: &[f32], w: usize, h: usize, ss: f32, sr: f32, detail: f32) -> Vec<f32> {
        let mut b = Bilateral::new(w, h, ss, sr);
        b.splat(input);
        b.blur();
        let mut out = vec![0.0f32; w * h * 4];
        b.slice(input, &mut out, detail);
        out
    }

    #[test]
    fn detail_zero_is_identity_on_l() {
        // detail 0 ⇒ norm 0 ⇒ L unchanged (and colour/alpha copied through).
        let (w, h) = (32usize, 24usize);
        let inp = img(w, h, |i, j| 10.0 + (i + j) as f32 % 40.0);
        let out = filter(&inp, w, h, 8.0, 8.0, 0.0);
        for (o, i) in out.iter().zip(inp.iter()) {
            assert!((o - i).abs() < 1e-4, "detail 0 changed a value: {o} vs {i}");
        }
    }

    #[test]
    fn flat_field_is_unchanged() {
        // A constant image bilateral-smooths to itself (nothing to average).
        let (w, h) = (40usize, 30usize);
        let inp = img(w, h, |_, _| 42.0);
        let out = filter(&inp, w, h, 6.0, 10.0, -1.0);
        for p in 0..w * h {
            assert!((out[4 * p] - 42.0).abs() < 1e-2, "flat L moved: {}", out[4 * p]);
        }
    }

    #[test]
    fn smoothing_reduces_noise_variance_but_keeps_an_edge() {
        // Left half dark, right half bright, plus per-pixel noise. Bilateral
        // smoothing (detail -1) must (a) cut the within-region variance yet
        // (b) preserve the step (a plain blur would bleed it).
        let (w, h) = (64usize, 48usize);
        let base = |i: usize| if i >= w / 2 { 80.0 } else { 20.0 };
        // deterministic pseudo-noise
        let noise = |i: usize, j: usize| (((i * 131 + j * 977) % 21) as f32 - 10.0) * 0.4;
        let inp = img(w, h, |i, j| base(i) + noise(i, j));
        let out = filter(&inp, w, h, 8.0, 12.0, -1.0);

        // (a) variance within the right (bright) region drops.
        let region_var = |buf: &[f32]| -> f32 {
            let mut vals = Vec::new();
            for j in 8..h - 8 {
                for i in (w / 2 + 6)..(w - 6) {
                    vals.push(buf[4 * (j * w + i)]);
                }
            }
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32
        };
        assert!(
            region_var(&out) < region_var(&inp) * 0.6,
            "smoothing didn't cut variance: {} -> {}",
            region_var(&inp), region_var(&out)
        );
        // (b) the step is preserved: mean(right) − mean(left) stays large (a
        // spatial blur would collapse it toward 0).
        let mean_of = |buf: &[f32], lo: usize, hi: usize| -> f32 {
            let mut s = 0.0; let mut n = 0;
            for j in 8..h - 8 {
                for i in lo..hi {
                    s += buf[4 * (j * w + i)]; n += 1;
                }
            }
            s / n as f32
        };
        let step = mean_of(&out, w / 2 + 8, w - 4) - mean_of(&out, 4, w / 2 - 8);
        assert!(step > 45.0, "edge not preserved (step {step}, input ~60)");
    }

    #[test]
    fn grid_dims_match_the_c_formula() {
        // Pin the grid sizing (dt_bilateral_grid_size) so a refactor can't drift it.
        let b = Bilateral::new(1000, 800, 10.0, 20.0);
        // _x=round(1000/10)=100, _y=round(800/10)=80, _z=round(100/20)=5
        // sigma_s=max(800/80,1000/100)=10, sigma_r=100/5=20
        // size_x=ceil(1000/10)+1=101, size_y=ceil(800/10)+1=81, size_z=ceil(100/20)+1=6
        assert_eq!((b.size_x, b.size_y, b.size_z), (101, 81, 6));
        assert!((b.sigma_s - 10.0).abs() < 1e-4 && (b.sigma_r - 20.0).abs() < 1e-4);
    }

    #[test]
    fn tiny_sigma_is_floored_so_the_grid_stays_bounded() {
        // sigma_s < 0.5 is floored; grid dims stay finite/sane.
        let b = Bilateral::new(50, 50, 0.1, 1.0);
        assert!(b.sigma_s >= 0.5 - 1e-6);
        assert!(b.size_x >= 5 && b.size_z >= 5);
    }

    #[test]
    fn narrow_image_collapses_an_axis_without_panicking() {
        // A 4px-wide image at ss=10 collapses size_x to 2, so blur_line along x has
        // size3=2 < 4. Must be a defined no-op, not a Rust OOB panic (C reads OOB
        // heap and survives; Rust would crash on a real 4px-wide crop export).
        let (w, h) = (4usize, 3000usize);
        let b = Bilateral::new(w, h, 10.0, 20.0);
        assert_eq!(b.size_x, 2, "this aspect should collapse the x axis");
        let inp = img(w, h, |i, j| 10.0 + ((i + j) % 40) as f32);
        let out = filter(&inp, w, h, 10.0, 20.0, -1.0); // must not panic
        assert_eq!(out.len(), w * h * 4);
    }

    #[test]
    fn slice_to_output_accumulates_and_leaves_colour_untouched() {
        // slice_to_output adds norm·interp into out[L] only — colour/alpha stay as
        // the caller left them (unlike slice, which copies them from input).
        let (w, h) = (32usize, 24usize);
        let inp = img(w, h, |i, _| if i >= w / 2 { 70.0 } else { 30.0 });
        let mut b = Bilateral::new(w, h, 8.0, 12.0);
        b.splat(&inp);
        b.blur();
        // Sentinel colour so we can prove it's NOT overwritten.
        let mut out = vec![-1.0f32; w * h * 4];
        b.slice_to_output(&inp, &mut out, -1.0);
        let p = 4 * (12 * w + w / 2 + 5);
        assert!(out[p] >= 0.0, "L clamped to >= 0");
        assert_eq!(out[p + 1], -1.0, "colour must be untouched by slice_to_output");
        assert_eq!(out[p + 3], -1.0, "alpha must be untouched");
        // detail 0 ⇒ norm 0 ⇒ a pure no-op add: out[L] keeps the caller's value.
        let mut out0 = vec![5.0f32; w * h * 4];
        b.slice_to_output(&inp, &mut out0, 0.0);
        assert!((out0[p] - 5.0).abs() < 1e-4, "detail 0 changed out[L]: {}", out0[p]);
    }
}
