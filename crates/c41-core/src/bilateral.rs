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
    /// Delegates to [`slice_kernel`] so the method and the FFI export below run
    /// one implementation.
    pub fn slice(&self, input: &[f32], output: &mut [f32], detail: f32) {
        slice_kernel(&self.buf, self.size_x, self.size_y, self.size_z,
                     self.sigma_s_inv, self.sigma_r_inv, self.sigma_r,
                     self.width, self.height, input, output, detail);
    }

    /// Like [`Bilateral::slice`] but **accumulates** into L and does NOT copy the
    /// colour/alpha channels: `out[L] = max(0, out[L] + norm·interp)` (matches
    /// `dt_bilateral_slice_to_output`; used by `ashift` / `globaltonemap`).
    /// `output` must already hold the buffer being accumulated into.
    pub fn slice_to_output(&self, input: &[f32], output: &mut [f32], detail: f32) {
        slice_to_output_kernel(&self.buf, self.size_x, self.size_y, self.size_z,
                               self.sigma_s_inv, self.sigma_r_inv, self.sigma_r,
                               self.width, self.height, input, output, detail);
    }
}

/// Grid → 8-neighbour trilinear index + fractions (matches C `image_to_grid`).
/// Shared by the [`slice_kernel`] / [`slice_to_output_kernel`] slice kernels
/// below so they run one implementation. Stride convention
/// matches the C: `ox = size_z`, `oy = size_x*size_z`, `oz = 1`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn grid_lookup(size_x: usize, size_y: usize, size_z: usize,
               sigma_s_inv: f32, sigma_r_inv: f32,
               i: usize, j: usize, l: f32) -> (usize, f32, f32, f32) {
    let x = clampf(i as f32 * sigma_s_inv, 0.0, (size_x - 1) as f32);
    let y = clampf(j as f32 * sigma_s_inv, 0.0, (size_y - 1) as f32);
    let z = clampf(l * sigma_r_inv, 0.0, (size_z - 1) as f32);
    let xi = (x as usize).min(size_x - 2);
    let yi = (y as usize).min(size_y - 2);
    let zi = (z as usize).min(size_z - 2);
    let gi = ((xi + yi * size_x) * size_z) + zi;
    (gi, x - xi as f32, y - yi as f32, z - zi as f32)
}

/// 8-tap trilinear read of the blurred grid at cell `gi` with fractions
/// `(xf,yf,zf)`, in the **exact C tap order**
/// (`gi, +ox, +oy, +ox+oy, +oz, +ox+oz, +oy+oz, +ox+oy+oz`) with the C's
/// left-associative per-tap factor chains (`buf * (1-xf) * (1-yf) * (1-zf)`).
/// Shared by the [`slice_kernel`] / [`slice_to_output_kernel`] slice kernels
/// so they can't drift.
#[inline]
fn interp_grid(grid: &[f32], size_x: usize, size_z: usize,
               gi: usize, xf: f32, yf: f32, zf: f32) -> f32 {
    let ox = size_z;
    let oy = size_x * size_z;
    let oz = 1usize;
    grid[gi] * (1.0 - xf) * (1.0 - yf) * (1.0 - zf)
        + grid[gi + ox] * xf * (1.0 - yf) * (1.0 - zf)
        + grid[gi + oy] * (1.0 - xf) * yf * (1.0 - zf)
        + grid[gi + ox + oy] * xf * yf * (1.0 - zf)
        + grid[gi + oz] * (1.0 - xf) * (1.0 - yf) * zf
        + grid[gi + ox + oz] * xf * (1.0 - yf) * zf
        + grid[gi + oy + oz] * (1.0 - xf) * yf * zf
        + grid[gi + ox + oy + oz] * xf * yf * zf
}

/// Safe `dt_bilateral_slice_to_output` kernel (m4-180): accumulates the blurred
/// grid's trilinear read-back into L —
/// `out[L] = max(0, out[L] + norm·interp)` with `norm = −detail·sigma_r·0.04` —
/// leaving the colour/alpha channels untouched.
///
/// `grid` holds `size_x·size_y·size_z` floats (z fastest, then x, then y);
/// `input`/`output` each hold `width·height·4` packed-RGBA floats (L is channel
/// 0). `input` and `output` may be the same buffer (each pixel is fully read
/// before its own L is written, as in the `ashift` caller) but must not
/// partially overlap.
///
/// Invalid inputs (a grid axis < 2 — trilinear taps need ≥ 2 cells —, zero
/// image dims, or slice lengths disagreeing with the dims) are a silent no-op
/// rather than a panic, so the FFI export below can share this body.
#[allow(clippy::too_many_arguments)]
pub fn slice_to_output_kernel(grid: &[f32],
                              size_x: usize, size_y: usize, size_z: usize,
                              sigma_s_inv: f32, sigma_r_inv: f32, sigma_r: f32,
                              width: usize, height: usize,
                              input: &[f32], output: &mut [f32], detail: f32) {
    let cells = match size_x.checked_mul(size_y).and_then(|v| v.checked_mul(size_z)) {
        Some(n) if n != 0 => n,
        _ => return,
    };
    if size_x < 2 || size_y < 2 || size_z < 2 || width == 0 || height == 0 {
        return;
    }
    let npix = match width.checked_mul(height).and_then(|v| v.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    if grid.len() != cells || input.len() < npix || output.len() < npix {
        return;
    }
    // detail: 0 is leave as is, −1 is bilateral filtered, +1 is contrast boost
    let norm = -detail * sigma_r * 0.04;
    for j in 0..height {
        for i in 0..width {
            let index = 4 * (j * width + i);
            let l = input[index];
            // trilinear lookup:
            let (gi, xf, yf, zf) =
                grid_lookup(size_x, size_y, size_z, sigma_s_inv, sigma_r_inv, i, j, l);
            let interp = interp_grid(grid, size_x, size_z, gi, xf, yf, zf);
            output[index] = (output[index] + norm * interp).max(0.0);
        }
    }
}

/// Structurally divergent reference for [`slice_to_output_kernel`]: flat pixel
/// loop, hand-rolled clamp/index math, and an offsets-table + weight-array
/// accumulation instead of nested loops and one long tap expression. Each tap
/// keeps the kernel's exact left-associative factor chain and taps accumulate
/// in the same order, so agreement checks spelling-level equivalence; the
/// hand-computed tap tests remain the backstop against identical drift.
#[allow(clippy::too_many_arguments)]
fn slice_to_output_reference(grid: &[f32],
                             size_x: usize, size_y: usize, size_z: usize,
                             sigma_s_inv: f32, sigma_r_inv: f32, sigma_r: f32,
                             width: usize, height: usize,
                             input: &[f32], output: &mut [f32], detail: f32) {
    let cells = match size_x.checked_mul(size_y).and_then(|v| v.checked_mul(size_z)) {
        Some(n) if n != 0 => n,
        _ => return,
    };
    if size_x < 2 || size_y < 2 || size_z < 2 || width == 0 || height == 0 {
        return;
    }
    let npix = match width.checked_mul(height).and_then(|v| v.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    if grid.len() != cells || input.len() < npix || output.len() < npix {
        return;
    }
    let norm = -detail * sigma_r * 0.04;
    let ox = size_z;
    let oy = size_x * size_z;
    let oz = 1usize;
    let npixels = width * height;
    for p in 0..npixels {
        let i = p % width;
        let j = p / width;
        let index = 4 * p;
        let l = input[index];
        // grid cell + fractions, spelled out without the shared helpers:
        let mut x = i as f32 * sigma_s_inv;
        if x < 0.0 { x = 0.0; } else if x > (size_x - 1) as f32 { x = (size_x - 1) as f32; }
        let mut y = j as f32 * sigma_s_inv;
        if y < 0.0 { y = 0.0; } else if y > (size_y - 1) as f32 { y = (size_y - 1) as f32; }
        let mut z = l * sigma_r_inv;
        if z < 0.0 { z = 0.0; } else if z > (size_z - 1) as f32 { z = (size_z - 1) as f32; }
        let mut xi = x as usize;
        if xi > size_x - 2 { xi = size_x - 2; }
        let mut yi = y as usize;
        if yi > size_y - 2 { yi = size_y - 2; }
        let mut zi = z as usize;
        if zi > size_z - 2 { zi = size_z - 2; }
        let gi = ((xi + yi * size_x) * size_z) + zi;
        let xf = x - xi as f32;
        let yf = y - yi as f32;
        let zf = z - zi as f32;
        // same 8 taps in the same order, via an offset table + weight array:
        let taps = [gi, gi + ox, gi + oy, gi + ox + oy,
                    gi + oz, gi + ox + oz, gi + oy + oz, gi + ox + oy + oz];
        let ax = 1.0 - xf;
        let ay = 1.0 - yf;
        let az = 1.0 - zf;
        let weights = [
            grid[taps[0]] * ax * ay * az,
            grid[taps[1]] * xf * ay * az,
            grid[taps[2]] * ax * yf * az,
            grid[taps[3]] * xf * yf * az,
            grid[taps[4]] * ax * ay * zf,
            grid[taps[5]] * xf * ay * zf,
            grid[taps[6]] * ax * yf * zf,
            grid[taps[7]] * xf * yf * zf,
        ];
        let mut interp = weights[0];
        for w in weights.iter().skip(1) {
            interp += *w;
        }
        output[index] = (output[index] + norm * interp).max(0.0);
    }
}

/// Safe `dt_bilateral_slice` kernel (m4-191): copies each packed-RGBA pixel
/// from `input` to `output`, then recomputes L as
/// `out[L] = max(0, L + norm·interp)` with `norm = −detail·sigma_r·0.04` and
/// `interp` the 8-tap trilinear read of the blurred grid.
///
/// `grid` holds `size_x·size_y·size_z` floats (z fastest, then x, then y);
/// `input`/`output` each hold `width·height·4` packed-RGBA floats (L is channel
/// 0). `input` and `output` may be the same buffer — each pixel's L is read
/// before its own write and no pixel reads another pixel's data, as in the
/// in-place `retouch` / `monochrome` / `colormapping` callers — but must not
/// partially overlap.
///
/// Invalid inputs (a grid axis < 2 — trilinear taps need ≥ 2 cells —, zero
/// image dims, or slice lengths disagreeing with the dims) are a silent no-op
/// rather than a panic, so the FFI export below can share this body.
#[allow(clippy::too_many_arguments)]
pub fn slice_kernel(grid: &[f32],
                    size_x: usize, size_y: usize, size_z: usize,
                    sigma_s_inv: f32, sigma_r_inv: f32, sigma_r: f32,
                    width: usize, height: usize,
                    input: &[f32], output: &mut [f32], detail: f32) {
    let cells = match size_x.checked_mul(size_y).and_then(|v| v.checked_mul(size_z)) {
        Some(n) if n != 0 => n,
        _ => return,
    };
    if size_x < 2 || size_y < 2 || size_z < 2 || width == 0 || height == 0 {
        return;
    }
    let npix = match width.checked_mul(height).and_then(|v| v.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    if grid.len() != cells || input.len() < npix || output.len() < npix {
        return;
    }
    // detail: 0 is leave as is, −1 is bilateral filtered, +1 is contrast boost
    let norm = -detail * sigma_r * 0.04;
    for j in 0..height {
        for i in 0..width {
            let index = 4 * (j * width + i);
            let l = input[index];
            // trilinear lookup:
            let (gi, xf, yf, zf) =
                grid_lookup(size_x, size_y, size_z, sigma_s_inv, sigma_r_inv, i, j, l);
            let interp = interp_grid(grid, size_x, size_z, gi, xf, yf, zf);
            // copy colour and alpha, then update L (matches C `copy_pixel` + write)
            output[index..index + 4].copy_from_slice(&input[index..index + 4]);
            output[index] = (l + norm * interp).max(0.0);
        }
    }
}

/// Structurally divergent reference for [`slice_kernel`]: flat pixel loop,
/// hand-rolled clamp/index math, an offsets-table + weight-array accumulation
/// instead of nested loops and one long tap expression, and a per-channel copy
/// staged through a temporary pixel instead of `copy_from_slice`. Each tap
/// keeps the kernel's exact left-associative factor chain and taps accumulate
/// in the same order, so agreement checks spelling-level equivalence; the
/// hand-computed tap tests remain the backstop against identical drift.
#[allow(clippy::too_many_arguments)]
fn slice_reference(grid: &[f32],
                   size_x: usize, size_y: usize, size_z: usize,
                   sigma_s_inv: f32, sigma_r_inv: f32, sigma_r: f32,
                   width: usize, height: usize,
                   input: &[f32], output: &mut [f32], detail: f32) {
    let cells = match size_x.checked_mul(size_y).and_then(|v| v.checked_mul(size_z)) {
        Some(n) if n != 0 => n,
        _ => return,
    };
    if size_x < 2 || size_y < 2 || size_z < 2 || width == 0 || height == 0 {
        return;
    }
    let npix = match width.checked_mul(height).and_then(|v| v.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    if grid.len() != cells || input.len() < npix || output.len() < npix {
        return;
    }
    let norm = -detail * sigma_r * 0.04;
    let ox = size_z;
    let oy = size_x * size_z;
    let oz = 1usize;
    let npixels = width * height;
    for p in 0..npixels {
        let i = p % width;
        let j = p / width;
        let index = 4 * p;
        let l = input[index];
        // grid cell + fractions, spelled out without the shared helpers:
        let mut x = i as f32 * sigma_s_inv;
        if x < 0.0 { x = 0.0; } else if x > (size_x - 1) as f32 { x = (size_x - 1) as f32; }
        let mut y = j as f32 * sigma_s_inv;
        if y < 0.0 { y = 0.0; } else if y > (size_y - 1) as f32 { y = (size_y - 1) as f32; }
        let mut z = l * sigma_r_inv;
        if z < 0.0 { z = 0.0; } else if z > (size_z - 1) as f32 { z = (size_z - 1) as f32; }
        let mut xi = x as usize;
        if xi > size_x - 2 { xi = size_x - 2; }
        let mut yi = y as usize;
        if yi > size_y - 2 { yi = size_y - 2; }
        let mut zi = z as usize;
        if zi > size_z - 2 { zi = size_z - 2; }
        let gi = ((xi + yi * size_x) * size_z) + zi;
        let xf = x - xi as f32;
        let yf = y - yi as f32;
        let zf = z - zi as f32;
        // same 8 taps in the same order, via an offset table + weight array:
        let taps = [gi, gi + ox, gi + oy, gi + ox + oy,
                    gi + oz, gi + ox + oz, gi + oy + oz, gi + ox + oy + oz];
        let ax = 1.0 - xf;
        let ay = 1.0 - yf;
        let az = 1.0 - zf;
        let weights = [
            grid[taps[0]] * ax * ay * az,
            grid[taps[1]] * xf * ay * az,
            grid[taps[2]] * ax * yf * az,
            grid[taps[3]] * xf * yf * az,
            grid[taps[4]] * ax * ay * zf,
            grid[taps[5]] * xf * ay * zf,
            grid[taps[6]] * ax * yf * zf,
            grid[taps[7]] * xf * yf * zf,
        ];
        let mut interp = weights[0];
        for w in weights.iter().skip(1) {
            interp += *w;
        }
        // copy colour and alpha channel by channel, then update L:
        let mut px = [0.0f32; 4];
        for c in 0..4 {
            px[c] = input[index + c];
        }
        px[0] = (l + norm * interp).max(0.0);
        for c in 0..4 {
            output[index + c] = px[c];
        }
    }
}

// ── FFI boundary (m4-180): `dt_bilateral_slice_to_output` in
// `src/common/bilateral.c` forwards its grid fields + pixel buffers here
// instead of running the OpenMP loop in C. Runs the SAME serial scalar code
// as [`slice_to_output_kernel`] (which [`Bilateral::slice_to_output`]
// delegates to), so method-level tests pin both callers at once; dedicated
// FFI parity tests re-drive this export against the kernel and the reference.

/// # Safety
/// `grid` must hold `size_x·size_y·size_z` floats (z fastest, then x, then y —
/// `dt_bilateral_t.buf`); `input`/`output` each hold `width·height·4`
/// packed-RGBA floats. `input` and `output` may be the same buffer (the
/// `ashift` caller passes `out, out`: each pixel is fully read before its own
/// L is written) but must not partially overlap. Null pointers, degenerate
/// dims (a grid axis < 2, non-positive image dims), and overflowing dim
/// products are guarded no-ops; the stated buffer lengths remain a caller
/// contract.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_bilateral_slice_to_output(
    grid: *const f32,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    sigma_s_inv: f32,
    sigma_r_inv: f32,
    sigma_r: f32,
    width: i32,
    height: i32,
    input: *const f32,
    output: *mut f32,
    detail: f32,
) {
    if grid.is_null() || input.is_null() || output.is_null() {
        return;
    }
    // Trilinear taps need ≥ 2 cells per axis; real grids are always ≥ 5 per
    // axis (`dt_bilateral_init` sizes are ceil()+1 over dims clamped to ≥ 4) —
    // this only guards corrupt callers instead of indexing out of bounds.
    if size_x < 2 || size_y < 2 || size_z < 2 {
        return;
    }
    // Same i32-cast wrap guard as the other grid exports: the lookup casts
    // pixel coords with `as usize`, and real grids are ≤ 3001 cells per axis.
    if size_x > i32::MAX as usize || size_y > i32::MAX as usize || size_z > i32::MAX as usize {
        return;
    }
    if width <= 0 || height <= 0 {
        return;
    }
    let (w, h) = (width as usize, height as usize);
    let cells = match size_x.checked_mul(size_y).and_then(|v| v.checked_mul(size_z)) {
        Some(n) => n,
        None => return,
    };
    let npix = match w.checked_mul(h).and_then(|v| v.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    let grid = std::slice::from_raw_parts(grid, cells);
    let input = std::slice::from_raw_parts(input, npix);
    let output = std::slice::from_raw_parts_mut(output, npix);
    slice_to_output_kernel(grid, size_x, size_y, size_z,
                           sigma_s_inv, sigma_r_inv, sigma_r,
                           w, h, input, output, detail);
}

// ── FFI boundary (m4-191): `dt_bilateral_slice` in
// `src/common/bilateral.c` forwards its grid fields + pixel buffers here
// instead of running the OpenMP loop in C. Runs the SAME serial scalar code
// as [`slice_kernel`] (which [`Bilateral::slice`] delegates to), so
// method-level tests pin both callers at once; dedicated FFI parity tests
// re-drive this export against the kernel and the reference.

/// # Safety
/// `grid` must hold `size_x·size_y·size_z` floats (z fastest, then x, then y —
/// `dt_bilateral_t.buf`); `input`/`output` each hold `width·height·4`
/// packed-RGBA floats. `input` and `output` may be the same buffer (the
/// `retouch`, `monochrome` and `colormapping` callers pass `in` as `out`:
/// each pixel's L is read before its own write, and no pixel reads another
/// pixel's data) but must not partially overlap. Null pointers, degenerate
/// dims (a grid axis < 2, non-positive image dims), and overflowing dim
/// products are guarded no-ops; the stated buffer lengths remain a caller
/// contract.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_bilateral_slice(
    grid: *const f32,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    sigma_s_inv: f32,
    sigma_r_inv: f32,
    sigma_r: f32,
    width: i32,
    height: i32,
    input: *const f32,
    output: *mut f32,
    detail: f32,
) {
    if grid.is_null() || input.is_null() || output.is_null() {
        return;
    }
    // Trilinear taps need ≥ 2 cells per axis; real grids are always ≥ 5 per
    // axis (`dt_bilateral_init` sizes are ceil()+1 over dims clamped to ≥ 4) —
    // this only guards corrupt callers instead of indexing out of bounds.
    if size_x < 2 || size_y < 2 || size_z < 2 {
        return;
    }
    // Same i32-cast wrap guard as the other grid exports: the lookup casts
    // pixel coords with `as usize`, and real grids are ≤ 3001 cells per axis.
    if size_x > i32::MAX as usize || size_y > i32::MAX as usize || size_z > i32::MAX as usize {
        return;
    }
    if width <= 0 || height <= 0 {
        return;
    }
    let (w, h) = (width as usize, height as usize);
    let cells = match size_x.checked_mul(size_y).and_then(|v| v.checked_mul(size_z)) {
        Some(n) => n,
        None => return,
    };
    let npix = match w.checked_mul(h).and_then(|v| v.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    let grid = std::slice::from_raw_parts(grid, cells);
    let input = std::slice::from_raw_parts(input, npix);
    let output = std::slice::from_raw_parts_mut(output, npix);
    slice_kernel(grid, size_x, size_y, size_z,
                 sigma_s_inv, sigma_r_inv, sigma_r,
                 w, h, input, output, detail);
}

// ── FFI boundary (m4-198/m4-199): `dt_bilateral_blur` in
// `src/common/bilateral.c` forwards all three passes here instead of
// running the OpenMP `blur_line` / `blur_line_z` in C. Each runs the SAME
// serial scalar code as [`blur_line`] / [`blur_line_z`] (which
// [`Bilateral::blur`] delegates to), so method-level tests pin every caller
// at once; dedicated FFI parity tests re-drive these exports against the
// method. Lines are independent — each (k, j) line touches disjoint
// cells — so the serial loops are bit-exact with the C's `DT_OMP_FOR` split
// over k: no reduction or cross-line read whose order serial could change.

/// # Safety
/// `buf` must hold at least
/// `offset1·(size1−1) + offset2·(size2−1) + offset3·(size3−1) + 1` floats —
/// the highest index the line loop touches. Cells are mutated strictly in
/// place, one independent line at a time.
///
/// The bound is sound for every input *because* [`blur_line`] early-returns
/// when `size3 < 4`: with fewer than four cells on the innermost axis the
/// [1 4 6 4 1] stencil has no valid centre, so no wider indexing occurs.
/// Null pointers, zero sizes, and overflowing offset bounds are guarded
/// no-ops; the stated buffer length remains a caller contract.
#[no_mangle]
pub unsafe extern "C" fn darkroom_bilateral_blur_line(
    buf: *mut f32,
    offset1: usize,
    offset2: usize,
    offset3: usize,
    size1: usize,
    size2: usize,
    size3: usize,
) {
    if buf.is_null() || size1 == 0 || size2 == 0 || size3 == 0 {
        return;
    }
    let max_index = match offset1
        .checked_mul(size1 - 1)
        .and_then(|a| a.checked_add(offset2.checked_mul(size2 - 1)?))
        .and_then(|b| b.checked_add(offset3.checked_mul(size3 - 1)?))
    {
        Some(m) => m,
        None => return,
    };
    let len = match max_index.checked_add(1) {
        Some(n) => n,
        None => return,
    };
    let buf = std::slice::from_raw_parts_mut(buf, len);
    blur_line(buf, offset1, offset2, offset3, size1, size2, size3);
}

/// # Safety
/// `buf` must hold at least
/// `offset1·(size1−1) + offset2·(size2−1) + offset3·(size3−1) + 1` floats —
/// the highest index the line loop touches. Cells are mutated strictly in
/// place, one independent line at a time.
///
/// The bound is sound for every input *because* [`blur_line_z`] early-returns
/// when `size3 < 4`: with fewer than four cells on the innermost axis the
/// derivative stencil has no valid centre, so no wider indexing occurs.
/// Null pointers, zero sizes, and overflowing offset bounds are guarded
/// no-ops; the stated buffer length remains a caller contract.
#[no_mangle]
pub unsafe extern "C" fn darkroom_bilateral_blur_line_z(
    buf: *mut f32,
    offset1: usize,
    offset2: usize,
    offset3: usize,
    size1: usize,
    size2: usize,
    size3: usize,
) {
    if buf.is_null() || size1 == 0 || size2 == 0 || size3 == 0 {
        return;
    }
    let max_index = match offset1
        .checked_mul(size1 - 1)
        .and_then(|a| a.checked_add(offset2.checked_mul(size2 - 1)?))
        .and_then(|b| b.checked_add(offset3.checked_mul(size3 - 1)?))
    {
        Some(m) => m,
        None => return,
    };
    let len = match max_index.checked_add(1) {
        Some(n) => n,
        None => return,
    };
    let buf = std::slice::from_raw_parts_mut(buf, len);
    blur_line_z(buf, offset1, offset2, offset3, size1, size2, size3);
}

/// Separable `[1 4 6 4 1]/16` Gaussian along the `offset3` axis (`size3` elements),
/// for each of size1×size2 lines. In-place, running-buffer boundary handling —
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

/// Structurally divergent reference for [`blur_line_z`]: two-phase instead of
/// running-buffer — each line is snapshotted into scratch, then every output
/// is recomputed from pristine snapshot cells with direct index arithmetic
/// (no tmp chain, no in-place forward reads). Equivalent because the kernel's
/// forward reads (`buf[index + offset3]`, `buf[index + 2*offset3]`) always hit
/// cells ahead of the write frontier, and the tmp chain replays pristine
/// predecessors (`tmp1` = snap[i-2], `tmp2` = snap[i-1]); each expression
/// below keeps the kernel's exact left-associative spelling so agreement
/// checks formula-level equivalence to the bit.
fn blur_line_z_reference(
    buf: &mut [f32],
    offset1: usize,
    offset2: usize,
    offset3: usize,
    size1: usize,
    size2: usize,
    size3: usize,
) {
    if size3 < 4 {
        return;
    }
    let (w1, w2) = (4.0 / 16.0, 2.0 / 16.0);
    let mut line = vec![0.0f32; size3];
    for k in 0..size1 {
        for j in 0..size2 {
            let start = k * offset1 + j * offset2;
            for i in 0..size3 {
                line[i] = buf[start + i * offset3];
            }
            buf[start] = w1 * line[1] + w2 * line[2];
            buf[start + offset3] = w1 * (line[2] - line[0]) + w2 * line[3];
            for i in 2..size3 - 2 {
                buf[start + i * offset3] =
                    w1 * (line[i + 1] - line[i - 1]) + w2 * (line[i + 2] - line[i - 2]);
            }
            let n = size3;
            buf[start + (n - 2) * offset3] = w1 * (line[n - 1] - line[n - 3]) - w2 * line[n - 4];
            buf[start + (n - 1) * offset3] = -w1 * line[n - 2] - w2 * line[n - 3];
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

    // ── m4-180: `dt_bilateral_slice_to_output` kernel tests ──

    /// Drive the safe kernel directly on a hand-built grid (bypasses
    /// `Bilateral::new` so a tiny synthetic grid is possible).
    fn run_kernel(grid: &[f32], sx: usize, sy: usize, sz: usize,
                  ss_inv: f32, sr_inv: f32, sr: f32,
                  w: usize, h: usize, input: &[f32], out_init: f32,
                  detail: f32) -> Vec<f32> {
        let mut out = vec![out_init; w * h * 4];
        slice_to_output_kernel(grid, sx, sy, sz, ss_inv, sr_inv, sr,
                               w, h, input, &mut out, detail);
        out
    }

    #[test]
    fn trilinear_weights_match_hand_computed_taps() {
        // 2×2×2 grid, unit inverse sigmas, one pixel at L=0.5:
        // x=0,y=0,z=0.5 → base cell 0, xf=yf=0, zf=0.5, so only the two z taps
        // contribute: interp = buf[0]*0.5 + buf[1]*0.5.
        let grid = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mut input = [0.0f32; 4];
        input[0] = 0.5;
        // detail −1, sigma_r 25 → norm = 1*25*0.04 = 1: out = max(0, 0+15) = 15.
        let out = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, 0.0, -1.0);
        assert!((out[0] - 15.0).abs() < 1e-5, "z taps wrong: {}", out[0]);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[3], 0.0);

        // Two-pixel-wide image: pixel (1,0) has x=1 → xi clamps to 0 with
        // xf=1, so the +ox taps take over:
        // interp = buf[2]*0.5 + buf[3]*0.5 = 30*0.5+40*0.5 = 35.
        let mut input2 = [0.0f32; 8];
        input2[0] = 0.5;
        input2[4] = 0.5;
        let out2 = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 2, 1, &input2, 0.0, -1.0);
        assert!((out2[0] - 15.0).abs() < 1e-5, "pixel 0 moved: {}", out2[0]);
        assert!((out2[4] - 35.0).abs() < 1e-5, "x taps wrong: {}", out2[4]);
    }

    #[test]
    fn detail_sign_and_scale_behave() {
        // Same synthetic setup as above: interp = 15 at the single pixel.
        let grid = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mut input = [0.0f32; 4];
        input[0] = 0.5;
        // detail −1 adds, detail +1 subtracts: with a large base both stay
        // clear of the clamp, so the deltas are exact opposites in sign.
        let base = 50.0f32;
        let neg = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, base, -1.0);
        let pos = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, base, 1.0);
        assert!(neg[0] > base, "detail −1 should brighten: {}", neg[0]);
        assert!(pos[0] < base, "detail +1 should darken: {}", pos[0]);
        assert!((neg[0] - base + (pos[0] - base)).abs() < 1e-4,
                "sign not symmetric: {neg:?} vs {pos:?}");
        // Doubling the magnitude doubles the stored result bit-exactly when the
        // base is 0 (out = norm·interp; scaling by 2 is exact in binary FP).
        let one = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, 0.0, -1.0);
        let two = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, 0.0, -2.0);
        assert_eq!(two[0].to_bits(), (2.0 * one[0]).to_bits(),
                   "detail scale not exact: {} vs {}", two[0], one[0]);
        // detail 0 is a pure no-op add whatever the grid holds.
        let zero = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, 7.25, 0.0);
        assert_eq!(zero[0].to_bits(), 7.25f32.to_bits());
    }

    #[test]
    fn negative_accumulation_clamps_to_zero() {
        // detail +1 with a zero base drives L negative → MAX(0, …) clamps.
        let grid = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mut input = [0.0f32; 4];
        input[0] = 0.5;
        let out = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, 0.0, 1.0);
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits(), "expected hard clamp, got {}", out[0]);
        // A base smaller than the subtracted term clamps too (0.5·|norm·15| < 15).
        let out2 = run_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, 5.0, 1.0);
        assert_eq!(out2[0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn degenerate_and_short_inputs_are_noops() {
        let grid = [1.0f32; 8];
        let input = [10.0f32; 16];
        // grid axis < 2 (trilinear needs a neighbour cell):
        let mut out = vec![3.0f32; 16];
        slice_to_output_kernel(&grid[..4], 1, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, &mut out, -1.0);
        assert!(out.iter().all(|&v| v == 3.0), "size_x<2 must no-op");
        // zero image dims:
        let mut out = vec![3.0f32; 4];
        slice_to_output_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 0, 1, &[], &mut out, -1.0);
        assert!(out.iter().all(|&v| v == 3.0), "width 0 must no-op");
        // short grid / input / output slices:
        let mut out = vec![3.0f32; 16];
        slice_to_output_kernel(&grid[..7], 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, &mut out, -1.0);
        assert!(out.iter().all(|&v| v == 3.0), "short grid must no-op");
        slice_to_output_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input[..3], &mut out, -1.0);
        assert!(out.iter().all(|&v| v == 3.0), "short input must no-op");
        let mut short_out = vec![3.0f32; 3];
        slice_to_output_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, &mut short_out, -1.0);
        assert!(short_out.iter().all(|&v| v == 3.0), "short output must no-op");
    }

    #[test]
    fn ffi_guards_leave_output_untouched() {
        // Null pointers and degenerate dims must no-op, never crash.
        let grid = [1.0f32; 8];
        let input = [10.0f32; 16];
        let mut out = [3.0f32; 16];
        unsafe {
            darkroom_bilateral_slice_to_output(
                std::ptr::null(), 2, 2, 2, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice_to_output(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 1, 1,
                std::ptr::null(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice_to_output(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), std::ptr::null_mut(), -1.0);
            // degenerate grid axis / image dims:
            darkroom_bilateral_slice_to_output(
                grid.as_ptr(), 1, 2, 2, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice_to_output(
                grid.as_ptr(), 2, 1, 2, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice_to_output(
                grid.as_ptr(), 2, 2, 1, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice_to_output(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 0, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice_to_output(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 1, 0,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice_to_output(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 1, -2,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
        }
        assert!(out.iter().all(|&v| v == 3.0), "guarded FFI calls must no-op: {out:?}");
    }

    #[test]
    fn ffi_matches_kernel_and_method_bit_exact() {
        // A realistic blurred grid through all three entry points must agree
        // to the bit — including the in-place (input == output) aliasing the
        // `ashift` caller relies on.
        let (w, h) = (24usize, 18usize);
        let inp = img(w, h, |i, j| ((i * 7 + j * 13) % 100) as f32);
        let mut b = Bilateral::new(w, h, 6.0, 10.0);
        b.splat(&inp);
        b.blur();
        for detail in [-1.0f32, -0.25, 0.0, 0.5, 2.0] {
            let mut via_method = vec![2.5f32; w * h * 4];
            b.slice_to_output(&inp, &mut via_method, detail);
            let mut via_kernel = vec![2.5f32; w * h * 4];
            slice_to_output_kernel(&b.buf, b.size_x, b.size_y, b.size_z,
                                   b.sigma_s_inv, b.sigma_r_inv, b.sigma_r,
                                   w, h, &inp, &mut via_kernel, detail);
            let mut via_ffi = vec![2.5f32; w * h * 4];
            unsafe {
                darkroom_bilateral_slice_to_output(
                    b.buf.as_ptr(), b.size_x, b.size_y, b.size_z,
                    b.sigma_s_inv, b.sigma_r_inv, b.sigma_r,
                    w as i32, h as i32,
                    inp.as_ptr(), via_ffi.as_mut_ptr(), detail);
            }
            for p in 0..w * h * 4 {
                assert_eq!(via_kernel[p].to_bits(), via_method[p].to_bits(),
                           "kernel vs method differ at {p} (detail {detail})");
                assert_eq!(via_ffi[p].to_bits(), via_method[p].to_bits(),
                           "FFI vs method differ at {p} (detail {detail})");
            }
            // in-place aliasing: out starts as a copy of in, FFI reads L from
            // the same buffer it accumulates into.
            let mut aliased = inp.clone();
            unsafe {
                darkroom_bilateral_slice_to_output(
                    b.buf.as_ptr(), b.size_x, b.size_y, b.size_z,
                    b.sigma_s_inv, b.sigma_r_inv, b.sigma_r,
                    w as i32, h as i32,
                    aliased.as_ptr(), aliased.as_mut_ptr(), detail);
            }
            let mut separate = inp.clone();
            unsafe {
                darkroom_bilateral_slice_to_output(
                    b.buf.as_ptr(), b.size_x, b.size_y, b.size_z,
                    b.sigma_s_inv, b.sigma_r_inv, b.sigma_r,
                    w as i32, h as i32,
                    inp.as_ptr(), separate.as_mut_ptr(), detail);
            }
            for p in 0..w * h {
                assert_eq!(aliased[4 * p].to_bits(), separate[4 * p].to_bits(),
                           "aliasing changed L at pixel {p} (detail {detail})");
                for c in 1..4 {
                    assert_eq!(aliased[4 * p + c].to_bits(), inp[4 * p + c].to_bits(),
                               "aliasing touched channel {c} at pixel {p}");
                }
            }
        }
    }

    #[test]
    fn slice_to_output_matches_reference_bit_exact() {
        // Kernel vs the structurally divergent reference over varied grid
        // shapes, sigma scales, L ranges (incl. negative and > 100, which hit
        // the clamp rails), and detail signs.
        let cases = [
            (2usize, 2usize, 2usize, 1.0f32, 1.0f32, 25.0f32),
            (3, 4, 5, 0.5, 0.2, 8.0),
            (5, 3, 6, 2.0, 0.05, 40.0),
        ];
        for (k, &(sx, sy, sz, ss_inv, sr_inv, sr)) in cases.iter().enumerate() {
            let (w, h) = (8usize, 6usize);
            let grid: Vec<f32> = (0..sx * sy * sz)
                .map(|n| ((n * 37 + k * 11) % 97) as f32 * 0.7 - 5.0)
                .collect();
            let input = img(w, h, |i, j| {
                // span the clamp rails: negative, in-range, and over-100 L
                [-12.0, 0.0, 37.5, 100.0, 140.0][(i + 3 * j + k) % 5]
            });
            for detail in [-2.0f32, -1.0, 0.0, 0.75, 10.0] {
                let mut a = vec![1.5f32; w * h * 4];
                let mut r = vec![1.5f32; w * h * 4];
                slice_to_output_kernel(&grid, sx, sy, sz, ss_inv, sr_inv, sr,
                                       w, h, &input, &mut a, detail);
                slice_to_output_reference(&grid, sx, sy, sz, ss_inv, sr_inv, sr,
                                          w, h, &input, &mut r, detail);
                for p in 0..w * h * 4 {
                    assert_eq!(a[p].to_bits(), r[p].to_bits(),
                               "kernel vs reference differ at {p} case {k} detail {detail}");
                }
            }
        }
    }

    // ── m4-191: `dt_bilateral_slice` kernel tests ──

    /// Drive the safe slice kernel directly on a hand-built grid (bypasses
    /// `Bilateral::new` so a tiny synthetic grid is possible).
    fn run_slice(grid: &[f32], sx: usize, sy: usize, sz: usize,
                 ss_inv: f32, sr_inv: f32, sr: f32,
                 w: usize, h: usize, input: &[f32],
                 detail: f32) -> Vec<f32> {
        let mut out = vec![f32::NAN; w * h * 4];
        slice_kernel(grid, sx, sy, sz, ss_inv, sr_inv, sr,
                     w, h, input, &mut out, detail);
        out
    }

    #[test]
    fn slice_trilinear_weights_match_hand_computed_taps() {
        // 2×2×2 grid, unit inverse sigmas, one pixel at L=0.5 with distinct
        // colour channels: x=0,y=0,z=0.5 → base cell 0, xf=yf=0, zf=0.5, so
        // only the two z taps contribute: interp = buf[0]*0.5 + buf[1]*0.5
        // = 10*0.5+20*0.5 = 15.
        let grid = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let input = [0.5f32, 7.0, 8.0, 9.0];
        // detail −1, sigma_r 25 → norm = 1*25*0.04 = 1:
        // out[L] = max(0, 0.5+15) = 15.5; colour/alpha copied through.
        let out = run_slice(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, -1.0);
        assert_eq!(out[0].to_bits(), 15.5f32.to_bits(), "z taps wrong: {}", out[0]);
        assert_eq!(out[1].to_bits(), 7.0f32.to_bits());
        assert_eq!(out[2].to_bits(), 8.0f32.to_bits());
        assert_eq!(out[3].to_bits(), 9.0f32.to_bits());

        // Two-pixel-wide image: pixel (1,0) has x=1 → xi clamps to 0 with
        // xf=1, so the +ox taps take over:
        // interp = buf[2]*0.5 + buf[3]*0.5 = 30*0.5+40*0.5 = 35.
        let input2 = [0.5f32, 7.0, 8.0, 9.0, 0.5, 1.0, 2.0, 3.0];
        let out2 = run_slice(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 2, 1, &input2, -1.0);
        assert_eq!(out2[0].to_bits(), 15.5f32.to_bits(), "pixel 0 moved: {}", out2[0]);
        assert_eq!(out2[4].to_bits(), 35.5f32.to_bits(), "x taps wrong: {}", out2[4]);
        assert_eq!(out2[5].to_bits(), 1.0f32.to_bits());
        assert_eq!(out2[7].to_bits(), 3.0f32.to_bits());
    }

    #[test]
    fn slice_detail_zero_copies_input_bit_exact() {
        // detail 0 ⇒ norm 0 ⇒ L unchanged AND colour copied: the whole pixel
        // is bit-identical to the input.
        let grid = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let input = [0.5f32, 7.25, -3.0, 9.5, 99.0, 1.0, 2.0, 0.0];
        let out = run_slice(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 2, 1, &input, 0.0);
        for p in 0..8 {
            assert_eq!(out[p].to_bits(), input[p].to_bits(),
                       "detail 0 changed lane {p}: {} vs {}", out[p], input[p]);
        }
    }

    #[test]
    fn slice_detail_sign_and_scale_behave() {
        // Same synthetic setup as above: interp = 15 at the single pixel.
        let grid = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let input = [50.0f32, 7.0, 8.0, 9.0];
        // detail −1 adds, detail +1 subtracts: with L=50 both stay clear of
        // the clamp, so the deltas are exact opposites in sign.
        let neg = run_slice(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, -1.0);
        let pos = run_slice(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, 1.0);
        assert!(neg[0] > 50.0, "detail −1 should brighten: {}", neg[0]);
        assert!(pos[0] < 50.0, "detail +1 should darken: {}", pos[0]);
        assert!((neg[0] - 50.0 + (pos[0] - 50.0)).abs() < 1e-4,
                "sign not symmetric: {neg:?} vs {pos:?}");
        // Doubling the magnitude doubles the delta bit-exactly
        // (out − L = norm·interp; scaling by 2 is exact in binary FP).
        let one = run_slice(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, -1.0);
        let two = run_slice(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, -2.0);
        assert_eq!((two[0] - 50.0).to_bits(), (2.0 * (one[0] - 50.0)).to_bits(),
                   "detail scale not exact: {} vs {}", two[0], one[0]);
    }

    #[test]
    fn slice_negative_result_clamps_to_zero() {
        // detail +1 with L=0.5 drives L negative → MAX(0, …) clamps.
        let grid = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let input = [0.5f32, 7.0, 8.0, 9.0];
        let out = run_slice(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, 1.0);
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits(), "expected hard clamp, got {}", out[0]);
        // Colour/alpha still copy through under the clamp.
        assert_eq!(out[1].to_bits(), 7.0f32.to_bits());
        assert_eq!(out[3].to_bits(), 9.0f32.to_bits());
    }

    #[test]
    fn slice_nan_poison_is_deterministic() {
        // A NaN L poisons the trilinear weights to NaN, so L + norm·interp is
        // NaN and f32::max (like C fmaxf) returns the other argument: 0.0.
        // Kernel and reference must agree to the bit, and colour must still
        // copy through untouched.
        let grid = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let input = [f32::NAN, 7.0, 8.0, 9.0];
        let mut a = vec![0.0f32; 4];
        let mut r = vec![0.0f32; 4];
        slice_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, &mut a, -1.0);
        slice_reference(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, &mut r, -1.0);
        for p in 0..4 {
            assert_eq!(a[p].to_bits(), r[p].to_bits(),
                       "kernel vs reference differ at {p} under NaN L");
        }
        assert_eq!(a[0].to_bits(), 0.0f32.to_bits(), "NaN L must clamp to 0, got {}", a[0]);
        assert_eq!(a[1].to_bits(), 7.0f32.to_bits());
        // A NaN grid cell poisons the same way (finite L, NaN tap).
        let mut nan_grid = grid;
        nan_grid[1] = f32::NAN;
        let finite = [0.5f32, 7.0, 8.0, 9.0];
        let mut a2 = vec![0.0f32; 4];
        let mut r2 = vec![0.0f32; 4];
        slice_kernel(&nan_grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &finite, &mut a2, -1.0);
        slice_reference(&nan_grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &finite, &mut r2, -1.0);
        for p in 0..4 {
            assert_eq!(a2[p].to_bits(), r2[p].to_bits(),
                       "kernel vs reference differ at {p} under NaN grid");
        }
        assert_eq!(a2[0].to_bits(), 0.0f32.to_bits(), "NaN grid must clamp to 0, got {}", a2[0]);
    }

    #[test]
    fn slice_degenerate_and_short_inputs_are_noops() {
        let grid = [1.0f32; 8];
        let input = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                     18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0];
        // grid axis < 2 (trilinear needs a neighbour cell):
        let mut out = vec![3.0f32; 16];
        slice_kernel(&grid[..4], 1, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, &mut out, -1.0);
        assert!(out.iter().all(|&v| v == 3.0), "size_x<2 must no-op");
        // zero image dims:
        let mut out = vec![3.0f32; 4];
        slice_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 0, 1, &[], &mut out, -1.0);
        assert!(out.iter().all(|&v| v == 3.0), "width 0 must no-op");
        // short grid / input / output slices:
        let mut out = vec![3.0f32; 16];
        slice_kernel(&grid[..7], 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, &mut out, -1.0);
        assert!(out.iter().all(|&v| v == 3.0), "short grid must no-op");
        slice_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input[..3], &mut out, -1.0);
        assert!(out.iter().all(|&v| v == 3.0), "short input must no-op");
        let mut short_out = vec![3.0f32; 3];
        slice_kernel(&grid, 2, 2, 2, 1.0, 1.0, 25.0, 1, 1, &input, &mut short_out, -1.0);
        assert!(short_out.iter().all(|&v| v == 3.0), "short output must no-op");
    }

    #[test]
    fn slice_ffi_guards_leave_output_untouched() {
        // Null pointers and degenerate dims must no-op, never crash.
        let grid = [1.0f32; 8];
        let input = [10.0f32; 16];
        let mut out = [3.0f32; 16];
        unsafe {
            darkroom_bilateral_slice(
                std::ptr::null(), 2, 2, 2, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 1, 1,
                std::ptr::null(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), std::ptr::null_mut(), -1.0);
            // degenerate grid axis / image dims:
            darkroom_bilateral_slice(
                grid.as_ptr(), 1, 2, 2, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice(
                grid.as_ptr(), 2, 1, 2, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice(
                grid.as_ptr(), 2, 2, 1, 1.0, 1.0, 25.0, 1, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 0, 1,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 1, 0,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
            darkroom_bilateral_slice(
                grid.as_ptr(), 2, 2, 2, 1.0, 1.0, 25.0, 1, -2,
                input.as_ptr(), out.as_mut_ptr(), -1.0);
        }
        assert!(out.iter().all(|&v| v == 3.0), "guarded FFI calls must no-op: {out:?}");
    }

    #[test]
    fn slice_ffi_matches_kernel_and_method_bit_exact() {
        // A realistic blurred grid through all three entry points must agree
        // to the bit — including the in-place (input == output) aliasing the
        // `retouch` / `monochrome` / `colormapping` callers rely on.
        let (w, h) = (24usize, 18usize);
        let inp = img(w, h, |i, j| ((i * 7 + j * 13) % 100) as f32);
        let mut b = Bilateral::new(w, h, 6.0, 10.0);
        b.splat(&inp);
        b.blur();
        for detail in [-1.0f32, -0.25, 0.0, 0.5, 2.0] {
            let mut via_method = vec![f32::NAN; w * h * 4];
            b.slice(&inp, &mut via_method, detail);
            let mut via_kernel = vec![f32::NAN; w * h * 4];
            slice_kernel(&b.buf, b.size_x, b.size_y, b.size_z,
                         b.sigma_s_inv, b.sigma_r_inv, b.sigma_r,
                         w, h, &inp, &mut via_kernel, detail);
            let mut via_ffi = vec![f32::NAN; w * h * 4];
            unsafe {
                darkroom_bilateral_slice(
                    b.buf.as_ptr(), b.size_x, b.size_y, b.size_z,
                    b.sigma_s_inv, b.sigma_r_inv, b.sigma_r,
                    w as i32, h as i32,
                    inp.as_ptr(), via_ffi.as_mut_ptr(), detail);
            }
            for p in 0..w * h * 4 {
                assert_eq!(via_kernel[p].to_bits(), via_method[p].to_bits(),
                           "kernel vs method differ at {p} (detail {detail})");
                assert_eq!(via_ffi[p].to_bits(), via_method[p].to_bits(),
                           "FFI vs method differ at {p} (detail {detail})");
            }
            // in-place aliasing: out starts as a copy of in, FFI reads L from
            // the same buffer it writes back into.
            let mut aliased = inp.clone();
            unsafe {
                darkroom_bilateral_slice(
                    b.buf.as_ptr(), b.size_x, b.size_y, b.size_z,
                    b.sigma_s_inv, b.sigma_r_inv, b.sigma_r,
                    w as i32, h as i32,
                    aliased.as_ptr(), aliased.as_mut_ptr(), detail);
            }
            let mut separate = vec![f32::NAN; w * h * 4];
            unsafe {
                darkroom_bilateral_slice(
                    b.buf.as_ptr(), b.size_x, b.size_y, b.size_z,
                    b.sigma_s_inv, b.sigma_r_inv, b.sigma_r,
                    w as i32, h as i32,
                    inp.as_ptr(), separate.as_mut_ptr(), detail);
            }
            for p in 0..w * h {
                assert_eq!(aliased[4 * p].to_bits(), separate[4 * p].to_bits(),
                           "aliasing changed L at pixel {p} (detail {detail})");
                for c in 1..4 {
                    assert_eq!(aliased[4 * p + c].to_bits(), inp[4 * p + c].to_bits(),
                               "aliasing touched channel {c} at pixel {p}");
                }
            }
        }
    }

    #[test]
    fn slice_matches_reference_bit_exact() {
        // Kernel vs the structurally divergent reference over varied grid
        // shapes, sigma scales, L ranges (incl. negative and > 100, which hit
        // the clamp rails), and detail signs.
        let cases = [
            (2usize, 2usize, 2usize, 1.0f32, 1.0f32, 25.0f32),
            (3, 4, 5, 0.5, 0.2, 8.0),
            (5, 3, 6, 2.0, 0.05, 40.0),
        ];
        for (k, &(sx, sy, sz, ss_inv, sr_inv, sr)) in cases.iter().enumerate() {
            let (w, h) = (8usize, 6usize);
            let grid: Vec<f32> = (0..sx * sy * sz)
                .map(|n| ((n * 37 + k * 11) % 97) as f32 * 0.7 - 5.0)
                .collect();
            let input = img(w, h, |i, j| {
                // span the clamp rails: negative, in-range, and over-100 L
                [-12.0, 0.0, 37.5, 100.0, 140.0][(i + 3 * j + k) % 5]
            });
            for detail in [-2.0f32, -1.0, 0.0, 0.75, 10.0] {
                let mut a = vec![1.5f32; w * h * 4];
                let mut r = vec![1.5f32; w * h * 4];
                slice_kernel(&grid, sx, sy, sz, ss_inv, sr_inv, sr,
                             w, h, &input, &mut a, detail);
                slice_reference(&grid, sx, sy, sz, ss_inv, sr_inv, sr,
                                w, h, &input, &mut r, detail);
                for p in 0..w * h * 4 {
                    assert_eq!(a[p].to_bits(), r[p].to_bits(),
                               "kernel vs reference differ at {p} case {k} detail {detail}");
                }
            }
        }
    }

    // ── m4-198: `dt_bilateral_blur` Gaussian-pass FFI parity ──

    #[test]
    fn blur_line_ffi_guards_leave_buffer_untouched() {
        // Null pointers, zero sizes, and overflowing offset bounds must
        // no-op, never crash or touch the buffer.
        let mut buf = [3.0f32; 64];
        unsafe {
            darkroom_bilateral_blur_line(std::ptr::null_mut(), 1, 1, 1, 2, 2, 4);
            darkroom_bilateral_blur_line(buf.as_mut_ptr(), 1, 1, 1, 0, 2, 4);
            darkroom_bilateral_blur_line(buf.as_mut_ptr(), 1, 1, 1, 2, 0, 4);
            darkroom_bilateral_blur_line(buf.as_mut_ptr(), 1, 1, 1, 2, 2, 0);
            // overflowing highest-touched-index bound:
            darkroom_bilateral_blur_line(buf.as_mut_ptr(), usize::MAX, 1, 1, 2, 2, 4);
            // degenerate line (size3 < 4): the kernel early-returns, so the
            // buffer is unchanged (the C reads OOB heap here and survives;
            // Rust makes it a defined no-op instead).
            darkroom_bilateral_blur_line(buf.as_mut_ptr(), 1, 8, 2, 2, 2, 2);
        }
        assert!(buf.iter().all(|&v| v == 3.0), "guarded FFI calls must no-op: {buf:?}");
    }

    #[test]
    fn blur_line_serial_ffi_matches_method_blur_bit_exact() {
        // All three serial FFI passes must reproduce Bilateral::blur to the
        // bit: the method and the FFI exports run one implementation, and
        // lines are independent so the serial loops are bit-exact with the
        // C's former OMP splits over k.
        let (w, h) = (24usize, 18usize);
        let input = img(w, h, |i, j| ((i * 7 + j * 13) % 100) as f32);
        let mut m = Bilateral::new(w, h, 6.0, 10.0);
        m.splat(&input);
        let mut raw = m.buf.clone();
        m.blur();
        let want = m.buf.clone();

        let (sx, sy, sz) = (m.size_x, m.size_y, m.size_z);
        let ox = sz;
        let oy = sx * sz;
        let oz = 1usize;
        unsafe {
            darkroom_bilateral_blur_line(raw.as_mut_ptr(), oz, oy, ox, sz, sy, sx);
            darkroom_bilateral_blur_line(raw.as_mut_ptr(), oz, ox, oy, sz, sx, sy);
            darkroom_bilateral_blur_line_z(raw.as_mut_ptr(), ox, oy, oz, sx, sy, sz);
        }
        assert_eq!(raw.len(), want.len());
        for (k, (a, b)) in raw.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "cell {k} diverged after blur");
        }
    }

    // ── m4-199: `dt_bilateral_blur` derivative-pass FFI parity ──

    #[test]
    fn blur_line_z_ffi_guards_leave_buffer_untouched() {
        // Null pointers, zero sizes, and overflowing offset bounds must
        // no-op, never crash or touch the buffer.
        let mut buf = [3.0f32; 64];
        unsafe {
            darkroom_bilateral_blur_line_z(std::ptr::null_mut(), 1, 1, 1, 2, 2, 4);
            darkroom_bilateral_blur_line_z(buf.as_mut_ptr(), 1, 1, 1, 0, 2, 4);
            darkroom_bilateral_blur_line_z(buf.as_mut_ptr(), 1, 1, 1, 2, 0, 4);
            darkroom_bilateral_blur_line_z(buf.as_mut_ptr(), 1, 1, 1, 2, 2, 0);
            // overflowing highest-touched-index bound:
            darkroom_bilateral_blur_line_z(buf.as_mut_ptr(), usize::MAX, 1, 1, 2, 2, 4);
            // degenerate line (size3 < 4): the kernel early-returns, so the
            // buffer is unchanged (the C reads OOB heap here and survives;
            // Rust makes it a defined no-op instead).
            darkroom_bilateral_blur_line_z(buf.as_mut_ptr(), 1, 8, 2, 2, 2, 2);
        }
        assert!(buf.iter().all(|&v| v == 3.0), "guarded FFI calls must no-op: {buf:?}");
    }

    #[test]
    fn blur_line_z_matches_reference_bit_exact() {
        // Kernel vs the two-phase snapshot reference over varied line counts,
        // lengths (head-only size3 == 4 through headed/bodied/tailed), and
        // strides (unit, gapped, and grid-like ox/oy/oz layouts).
        let layouts = [
            (1usize, 1usize, 1usize),
            (1, 2, 3),
            (4, 3, 2),
        ];
        for (case, &(o1, o2, o3)) in layouts.iter().enumerate() {
            for size3 in [4usize, 5, 6, 9, 16] {
                let (size1, size2) = (3usize, 2usize);
                let len = o1 * (size1 - 1) + o2 * (size2 - 1) + o3 * (size3 - 1) + 1;
                let seed = |n: usize| ((n * 37 + case * 11 + size3 * 5) % 97) as f32 * 0.7 - 5.0;
                let base: Vec<f32> = (0..len).map(seed).collect();
                let mut a = base.clone();
                let mut r = base.clone();
                blur_line_z(&mut a, o1, o2, o3, size1, size2, size3);
                blur_line_z_reference(&mut r, o1, o2, o3, size1, size2, size3);
                for (k, (x, y)) in a.iter().zip(&r).enumerate() {
                    assert_eq!(x.to_bits(), y.to_bits(),
                               "kernel vs reference differ at {k} case {case} size3 {size3}");
                }
                // the pass must actually change something (guards the test
                // against a vacuous both-no-op agreement):
                assert!(a.iter().zip(&base).any(|(x, y)| x.to_bits() != y.to_bits()),
                        "pass was a no-op in case {case} size3 {size3}");
            }
        }
    }

    #[test]
    fn blur_line_z_ffi_matches_kernel_bit_exact() {
        // Re-driving the export over a grid-like layout must reproduce the
        // kernel to the bit (they share the body; this pins the FFI plumbing).
        let (sx, sy, sz) = (5usize, 4usize, 7usize);
        let (ox, oy, oz) = (sz, sx * sz, 1usize);
        let len = ox * (sx - 1) + oy * (sy - 1) + oz * (sz - 1) + 1;
        let base: Vec<f32> = (0..len).map(|n| ((n * 13) % 53) as f32 * 0.5 - 3.0).collect();
        let mut via_kernel = base.clone();
        let mut via_ffi = base.clone();
        blur_line_z(&mut via_kernel, ox, oy, oz, sx, sy, sz);
        unsafe {
            darkroom_bilateral_blur_line_z(via_ffi.as_mut_ptr(), ox, oy, oz, sx, sy, sz);
        }
        for (k, (x, y)) in via_kernel.iter().zip(&via_ffi).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "FFI vs kernel differ at cell {k}");
        }
    }
}
