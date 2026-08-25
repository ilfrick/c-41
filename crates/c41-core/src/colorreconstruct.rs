//! `colorreconstruction`'s bespoke 4-field bilateral grid — a faithful port of the
//! private grid in `src/iop/colorreconstruction.c` (`dt_iop_colorreconstruct_*`).
//! This is **NOT** the shared 3-D bilateral grid ([`crate::bilateral`]): that one
//! carries a single L payload with a z-fastest index; this one carries a full
//! `{L, a, b, weight}` cell per grid point with an **x-fastest** index
//! (`xi + size_x·(yi + size_y·zi)`), and reconstructs the a/b (chroma) of clipped
//! highlights by pulling in surrounding colour. Hence a separate port.
//!
//! Pipeline (matches the C):
//! - [`ColorReconstruct::splat`] scatters every **sub-threshold** pixel into the
//!   grid by **nearest-integer** cell (not trilinear), accumulating
//!   `L·w, a·w, b·w, w` where the per-pixel weight `w` depends on the
//!   [`Precedence`] mode (none = 1, chroma = `√(a²+b²)`, hue = a Gaussian around a
//!   target hue). Pixels with `L > threshold` (the clipped highlights themselves)
//!   are deliberately skipped so they don't pollute the colour they'll borrow.
//! - [`ColorReconstruct::blur`] runs the separable `[1 4 6 4 1]/16` Gaussian over
//!   x, then y, then z (all three axes — unlike [`crate::bilateral`], there is no
//!   derivative-of-Gaussian z pass here).
//! - [`ColorReconstruct::slice`] reads the blurred grid back with trilinear
//!   interpolation and, for pixels bright enough to be `blend`ed (near/above
//!   threshold), replaces their a/b with the neighbourhood colour ratio
//!   `a_grid/L_grid · L_pixel` (L and alpha are passed through untouched).
//!
//! The grid dimensions are always `clamp(round(dim/σ), 4, MAX)+1 ≥ 5`, so — unlike
//! the shared bilateral grid — an axis can never collapse below the 4-point
//! boundary the blur needs; the `size3 < 4` guard in [`blur_line`] is kept only as
//! cheap defence.
//!
//! Not ported: the pixelpipe grid freeze/thaw caching (the FULL-vs-preview grid
//! "stealing" in the C `process`) — that is pipeline plumbing, not grid algorithm —
//! and all OpenCL. `hue_conversion` (GUI HSL-hue → LCH hue) is also left to the
//! caller: pass the already-converted hue into [`Precedence::Hue`].
//!
//! **m4-138**: the C IOP's three former OpenMP loops now call the
//! `darkroom_colorreconstruct_{splat,blur_line,slice}` exports below instead of
//! running in C (m4-86 convention). The exports share the exact loop bodies with
//! the methods via [`splat_cells`]/[`slice_cells`]/[`blur_line`] over a
//! `#[repr(C)]`-aliased cell buffer, so method-level tests pin both callers at
//! once; dedicated FFI parity tests re-drive each export against its method.

use crate::roi::RoiIn;

/// Grid resolution clamps from the C (`DT_COLORRECONSTRUCT_BILATERAL_MAX_RES_*`).
const MAX_RES_S: i32 = 500;
const MAX_RES_R: i32 = 100;
/// The L range the grid's z axis spans (Lab L is [0, 100]).
const L_RANGE: f32 = 100.0;

/// One grid cell: accumulated `{L, a, b, weight}` (matches
/// `dt_iop_colorreconstruct_Lab_t`). `repr(C)` pins the field order so the
/// FFI exports below can alias the C-side cell buffer directly — pinned by
/// [`tests::cell_layout_matches_the_c_struct_for_ffi_aliasing`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
struct Cell {
    l: f32,
    a: f32,
    b: f32,
    weight: f32,
}

impl core::ops::Add for Cell {
    type Output = Cell;
    #[inline]
    fn add(self, o: Cell) -> Cell {
        Cell { l: self.l + o.l, a: self.a + o.a, b: self.b + o.b, weight: self.weight + o.weight }
    }
}

impl core::ops::Mul<f32> for Cell {
    type Output = Cell;
    #[inline]
    fn mul(self, s: f32) -> Cell {
        Cell { l: self.l * s, a: self.a * s, b: self.b * s, weight: self.weight * s }
    }
}

/// Per-pixel splat weighting mode (matches `dt_iop_colorreconstruct_precedence_t`).
#[derive(Clone, Copy, Debug)]
pub enum Precedence {
    /// Every sub-threshold pixel contributes with weight 1.
    None,
    /// Weight by chroma magnitude `√(a² + b²)` — favours saturated neighbours.
    Chroma,
    /// Weight by a Gaussian around a target `hue` (LCH, radians) with variance
    /// `sigma_sq` (the C uses `π²/8`). Favours neighbours of a chosen hue.
    Hue { hue: f32, sigma_sq: f32 },
}

/// C `CLAMPS(A, L, H)` — exact, including branch order so a NaN `A` returns `L`
/// (the low bound), matching the C macro rather than propagating the NaN.
#[inline]
fn clamps(v: f32, lo: f32, hi: f32) -> f32 {
    if v > lo {
        if v < hi {
            v
        } else {
            hi
        }
    } else {
        lo
    }
}

/// The scalar part of the grid (the `b->…` fields of
/// `dt_iop_colorreconstruct_bilateral_t`), shared by the [`ColorReconstruct`]
/// methods and the FFI exports so both run ONE implementation of each loop body.
#[derive(Clone, Copy, Debug)]
struct GridHeader {
    size_x: usize,
    size_y: usize,
    size_z: usize,
    /// init-roi origin (`b->x`, `b->y`) — used by slice's grid_rescale.
    x: i32,
    y: i32,
    /// `iscale / roi.scale` at init (`b->scale`).
    scale: f32,
    sigma_s: f32,
    sigma_r: f32,
    /// init-roi dimensions, for splat.
    width: usize,
    height: usize,
}

/// Grid coordinates `(x, y, z)` for image position `(i, j)` and luma `l`
/// (matches `image_to_grid`; `i`/`j` are floats so slice can pass its rescaled,
/// fractional positions).
#[inline]
fn image_to_grid_h(h: &GridHeader, i: f32, j: f32, l: f32) -> (f32, f32, f32) {
    (
        clamps(i / h.sigma_s, 0.0, (h.size_x - 1) as f32),
        clamps(j / h.sigma_s, 0.0, (h.size_y - 1) as f32),
        clamps(l / h.sigma_r, 0.0, (h.size_z - 1) as f32),
    )
}

/// Image position `(i, j)` in the slice ROI → grid-space `(px, py)` accounting
/// for a possibly different ROI/scale than the grid was built at (matches
/// `grid_rescale`).
#[inline]
fn grid_rescale_h(h: &GridHeader, i: usize, j: usize, roi: RoiIn, rescale: f32) -> (f32, f32) {
    (
        (roi.x + i as i32) as f32 * rescale - h.x as f32,
        (roi.y + j as i32) as f32 * rescale - h.y as f32,
    )
}

/// The `colorreconstruction` 4-field bilateral grid.
pub struct ColorReconstruct {
    size_x: usize,
    size_y: usize,
    size_z: usize,
    /// init-roi origin (`b->x`, `b->y`) — used by [`Self::slice`]'s `grid_rescale`.
    x: i32,
    y: i32,
    /// `iscale / roi.scale` at init (`b->scale`).
    scale: f32,
    sigma_s: f32,
    sigma_r: f32,
    /// init-roi dimensions, for [`Self::splat`].
    width: usize,
    height: usize,
    buf: Vec<Cell>,
}

impl ColorReconstruct {
    /// Allocate the grid for the input ROI (matches
    /// `dt_iop_colorreconstruct_bilateral_init`). `iscale` is `piece->iscale`;
    /// `sigma_s`/`sigma_r` are the spatial/range sigmas the caller already derived
    /// (`spatial/scale` and `range` in the C `process`).
    pub fn new(roi: RoiIn, iscale: f32, sigma_s: f32, sigma_r: f32) -> Self {
        let _x = (roi.width as f32 / sigma_s).round() as i32;
        let _y = (roi.height as f32 / sigma_s).round() as i32;
        let _z = (L_RANGE / sigma_r).round() as i32;
        let size_x = (_x.clamp(4, MAX_RES_S) + 1) as usize;
        let size_y = (_y.clamp(4, MAX_RES_S) + 1) as usize;
        let size_z = (_z.clamp(4, MAX_RES_R) + 1) as usize;
        // effective sigmas after the (possible) dimension clamp
        let sigma_s = (roi.height as f32 / (size_y as f32 - 1.0))
            .max(roi.width as f32 / (size_x as f32 - 1.0));
        let sigma_r = L_RANGE / (size_z as f32 - 1.0);
        Self {
            size_x,
            size_y,
            size_z,
            x: roi.x,
            y: roi.y,
            scale: iscale / roi.scale,
            sigma_s,
            sigma_r,
            width: roi.width.max(0) as usize,
            height: roi.height.max(0) as usize,
            buf: vec![Cell::default(); size_x * size_y * size_z],
        }
    }

    fn header(&self) -> GridHeader {
        GridHeader {
            size_x: self.size_x,
            size_y: self.size_y,
            size_z: self.size_z,
            x: self.x,
            y: self.y,
            scale: self.scale,
            sigma_s: self.sigma_s,
            sigma_r: self.sigma_r,
            width: self.width,
            height: self.height,
        }
    }

    /// Scatter every sub-threshold pixel of the packed-Lab input (`width*height*4`
    /// floats) into the grid by nearest-integer cell (matches
    /// `dt_iop_colorreconstruct_bilateral_splat`, serial — the C shards rows across
    /// threads with atomic adds into one grid; a serial accumulation writes the
    /// same additive payloads to the same cells, equal up to float add order).
    pub fn splat(&mut self, input: &[f32], threshold: f32, precedence: Precedence) {
        let h = self.header();
        splat_cells(&mut self.buf, &h, input, threshold, precedence);
    }

    /// Separable `[1 4 6 4 1]/16` Gaussian over x, then y, then z (matches
    /// `dt_iop_colorreconstruct_bilateral_blur`).
    pub fn blur(&mut self) {
        let (sx, sy, sz) = (self.size_x, self.size_y, self.size_z);
        // along x (offset3 = 1), lines over z (outer) and y (inner)
        blur_line(&mut self.buf, sx * sy, sx, 1, sz, sy, sx);
        // along y (offset3 = size_x), lines over z and x
        blur_line(&mut self.buf, sx * sy, 1, sx, sz, sx, sy);
        // along z (offset3 = size_x*size_y), lines over x and y
        blur_line(&mut self.buf, 1, sx, sx * sy, sx, sy, sz);
    }

    /// Trilinear read-back: reconstruct the a/b of near/above-threshold pixels from
    /// the blurred grid (matches `dt_iop_colorreconstruct_bilateral_slice`). `input`
    /// and `output` are packed Lab (`roi.width*roi.height*4`). L and alpha are
    /// copied straight through; only a/b are rewritten, and only where `blend > 0`.
    ///
    /// `roi`/`iscale` are the *slice-time* ROI and `piece->iscale`; when they match
    /// the ROI the grid was built at, `rescale == 1` and [`grid_rescale_h`] is identity.
    pub fn slice(&self, input: &[f32], output: &mut [f32], threshold: f32, roi: RoiIn, iscale: f32) {
        slice_cells(&self.buf, &self.header(), input, output, threshold, roi, iscale);
    }
}

/// Scatter pass — the loop body shared by [`ColorReconstruct::splat`] and the
/// `darkroom_colorreconstruct_splat` FFI export.
fn splat_cells(
    cells: &mut [Cell],
    h: &GridHeader,
    input: &[f32],
    threshold: f32,
    precedence: Precedence,
) {
    for j in 0..h.height {
        for i in 0..h.width {
            let index = 4 * (j * h.width + i);
            let lin = input[index];
            let ain = input[index + 1];
            let bin = input[index + 2];
            // deliberately ignore pixels above threshold (the clipped ones)
            if lin > threshold {
                continue;
            }

            let weight = match precedence {
                Precedence::None => 1.0,
                Precedence::Chroma => (ain * ain + bin * bin).sqrt(),
                Precedence::Hue { hue, sigma_sq } => {
                    let mut m = bin.atan2(ain) - hue;
                    // readjust m into [-pi, +pi]
                    m = if m > core::f32::consts::PI {
                        m - core::f32::consts::TAU
                    } else if m < -core::f32::consts::PI {
                        m + core::f32::consts::TAU
                    } else {
                        m
                    };
                    (-m * m / sigma_sq).exp()
                }
            };

            let (x, y, z) = image_to_grid_h(h, i as f32, j as f32, lin);
            // closest-integer splatting
            let xi = (x.round() as i32).clamp(0, h.size_x as i32 - 1) as usize;
            let yi = (y.round() as i32).clamp(0, h.size_y as i32 - 1) as usize;
            let zi = (z.round() as i32).clamp(0, h.size_z as i32 - 1) as usize;
            let gi = xi + h.size_x * (yi + h.size_y * zi);

            let cell = &mut cells[gi];
            cell.l += lin * weight;
            cell.a += ain * weight;
            cell.b += bin * weight;
            cell.weight += weight;
        }
    }
}

/// Read-back pass — the loop body shared by [`ColorReconstruct::slice`] and the
/// `darkroom_colorreconstruct_slice` FFI export.
fn slice_cells(
    cells: &[Cell],
    h: &GridHeader,
    input: &[f32],
    output: &mut [f32],
    threshold: f32,
    roi: RoiIn,
    iscale: f32,
) {
    let rescale = iscale / (roi.scale * h.scale);
    let ox = 1usize;
    let oy = h.size_x;
    let oz = h.size_y * h.size_x;
    let rw = roi.width.max(0) as usize;
    let rh = roi.height.max(0) as usize;

    for j in 0..rh {
        for i in 0..rw {
            let index = 4 * (j * rw + i);
            let lin = input[index];
            let ain = input[index + 1];
            let bin = input[index + 2];
            // pass L, a, b, alpha through first (a/b may be overwritten below)
            output[index] = lin;
            output[index + 1] = ain;
            output[index + 2] = bin;
            output[index + 3] = input[index + 3];

            let blend = clamps(20.0 / threshold * lin - 19.0, 0.0, 1.0);
            if blend == 0.0 {
                continue;
            }

            let (px, py) = grid_rescale_h(h, i, j, roi, rescale);
            let (x, y, z) = image_to_grid_h(h, px, py, lin);
            // trilinear lookup base cell + fractions
            let xi = (x as i32).clamp(0, h.size_x as i32 - 2) as usize;
            let yi = (y as i32).clamp(0, h.size_y as i32 - 2) as usize;
            let zi = (z as i32).clamp(0, h.size_z as i32 - 2) as usize;
            let xf = x - xi as f32;
            let yf = y - yi as f32;
            let zf = z - zi as f32;
            let gi = xi + h.size_x * (yi + h.size_y * zi);

            let out = interp_cells(cells, gi, ox, oy, oz, xf, yf, zf);
            let lout = out.l.max(0.01);
            if out.weight > 0.0 {
                output[index + 1] = ain * (1.0 - blend) + out.a * lin / lout * blend;
                output[index + 2] = bin * (1.0 - blend) + out.b * lin / lout * blend;
            }
            // out.weight <= 0: a/b keep the passed-through input (already set)
        }
    }
}

/// 8-tap trilinear read of the blurred grid at base cell `gi` with fractions
/// `(xf, yf, zf)`, returning the interpolated `{L, a, b, weight}` cell.
///
/// The 8 taps are summed in the **exact C order** (`gi, +ox, +oy, +ox+oy, +oz,
/// +ox+oz, +oy+oz, +ox+oy+oz`). Within each tap we factor the weight product
/// once (`Cell * (wx·wy·wz)`) whereas the C multiplies field-first and
/// left-associatively (`buf[gi].L * (1-xf) * (1-yf) * (1-zf)`). IEEE mul is
/// commutative but not associative, so this can differ by ~1 ULP per tap.
/// This is intentional and immaterial: **bit-exact parity with the shipped C
/// is not achievable for this grid anyway** — its splat uses non-deterministic
/// atomic float-add ordering across threads, so the grid contents already vary
/// run-to-run by far more than a slicing ULP. Keeping the `Cell * scalar` form
/// buys readability at no meaningful accuracy cost.
#[inline]
#[allow(clippy::too_many_arguments)]
fn interp_cells(
    cells: &[Cell],
    gi: usize,
    ox: usize,
    oy: usize,
    oz: usize,
    xf: f32,
    yf: f32,
    zf: f32,
) -> Cell {
    cells[gi] * ((1.0 - xf) * (1.0 - yf) * (1.0 - zf))
        + cells[gi + ox] * (xf * (1.0 - yf) * (1.0 - zf))
        + cells[gi + oy] * ((1.0 - xf) * yf * (1.0 - zf))
        + cells[gi + ox + oy] * (xf * yf * (1.0 - zf))
        + cells[gi + oz] * ((1.0 - xf) * (1.0 - yf) * zf)
        + cells[gi + ox + oz] * (xf * (1.0 - yf) * zf)
        + cells[gi + oy + oz] * ((1.0 - xf) * yf * zf)
        + cells[gi + ox + oy + oz] * (xf * yf * zf)
}

// ── FFI boundary (m4-138): src/iop/colorreconstruction.c calls these instead of
// running its former OpenMP loops; each export is the SAME serial scalar code
// the methods above use (splat_cells / blur_line / slice_cells), over a cell
// buffer aliased in place via `#[repr(C)]` Cell ≡ `dt_iop_colorreconstruct_Lab_t`.

fn map_precedence(precedence: i32, hue: f32, sigma_sq: f32) -> Precedence {
    match precedence {
        // C enum: NONE=0, CHROMA=1, HUE=2; the switch's `default:` arm weights
        // every unknown value like NONE
        1 => Precedence::Chroma,
        2 => Precedence::Hue { hue, sigma_sq },
        _ => Precedence::None,
    }
}

/// # Safety
/// `grid_buf` must hold `4·size_x·size_y·size_z` floats laid out as consecutive
/// `{L, a, b, weight}` cells (`dt_iop_colorreconstruct_Lab_t`, layout pinned by
/// test); `input` must hold `4·width·height` floats of packed Lab. Aliasing rules
/// as in C: `input` may not overlap `grid_buf`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorreconstruct_splat(
    grid_buf: *mut f32,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    scale: f32,
    sigma_s: f32,
    sigma_r: f32,
    input: *const f32,
    threshold: f32,
    precedence: i32,
    hue: f32,
    hue_sigma_sq: f32,
) {
    if grid_buf.is_null()
        || input.is_null()
        || width == 0
        || height == 0
        // real grids are always ≥5 per axis (`init` clamps to 4+1); refuse
        // corrupt callers instead of panicking on an inverted clamp range
        || size_x == 0
        || size_y == 0
        || size_z == 0
        // the loop bodies cast cell counts to i32 for clamping; refuse dims a
        // real grid can never have rather than wrap (real grids are ≤501/axis)
        || size_x > i32::MAX as usize
        || size_y > i32::MAX as usize
        || size_z > i32::MAX as usize
        || width > i32::MAX as usize
        || height > i32::MAX as usize
    {
        return;
    }
    let n = match size_x.checked_mul(size_y).and_then(|v| v.checked_mul(size_z)) {
        Some(n) => n,
        None => return,
    };
    let cells = std::slice::from_raw_parts_mut(grid_buf.cast::<Cell>(), n);
    let input = std::slice::from_raw_parts(input, width * height * 4);
    let h = GridHeader {
        size_x,
        size_y,
        size_z,
        x,
        y,
        scale,
        sigma_s,
        sigma_r,
        width,
        height,
    };
    splat_cells(cells, &h, input, threshold, map_precedence(precedence, hue, hue_sigma_sq));
}

/// # Safety
/// `buf` must hold at least
/// `offset1·(size1−1) + offset2·(size2−1) + offset3·(size3−1) + 1` cells — the
/// highest index the line loop touches. Cells are mutated strictly in place.
///
/// The bound is sound for every input *because* [`blur_line`] early-returns
/// when `size3 < 4`: with fewer than four cells on the innermost axis the
/// [1 4 6 4 1] stencil has no valid centre, so no wider indexing occurs.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorreconstruct_blur_line(
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
    let cells = std::slice::from_raw_parts_mut(buf.cast::<Cell>(), max_index + 1);
    blur_line(cells, offset1, offset2, offset3, size1, size2, size3);
}

/// # Safety
/// `grid_buf` holds `4·size_x·size_y·size_z` cells (const); `input`/`output`
/// each hold `4·roi_width·roi_height` floats; `output` may alias `input` (each
/// pixel is fully read before any write, and pass-through writes only touch the
/// current pixel).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorreconstruct_slice(
    grid_buf: *const f32,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    x: i32,
    y: i32,
    scale: f32,
    sigma_s: f32,
    sigma_r: f32,
    input: *const f32,
    output: *mut f32,
    threshold: f32,
    roi_x: i32,
    roi_y: i32,
    roi_width: i32,
    roi_height: i32,
    roi_scale: f32,
    iscale: f32,
) {
    if grid_buf.is_null()
        || input.is_null()
        || output.is_null()
        // trilinear taps need ≥2 cells per axis; real grids are always ≥5
        // (`init` clamps to 4+1) — this only guards corrupt callers
        || size_x < 2
        || size_y < 2
        || size_z < 2
        // same i32-cast wrap guard as splat (roi dims are i32 already)
        || size_x > i32::MAX as usize
        || size_y > i32::MAX as usize
        || size_z > i32::MAX as usize
    {
        return;
    }
    let n = match size_x.checked_mul(size_y).and_then(|v| v.checked_mul(size_z)) {
        Some(n) => n,
        None => return,
    };
    if roi_width <= 0 || roi_height <= 0 {
        return;
    }
    let cells = std::slice::from_raw_parts(grid_buf.cast::<Cell>(), n);
    let input = std::slice::from_raw_parts(input, roi_width as usize * roi_height as usize * 4);
    let output =
        std::slice::from_raw_parts_mut(output, roi_width as usize * roi_height as usize * 4);
    let h = GridHeader {
        size_x,
        size_y,
        size_z,
        x,
        y,
        scale,
        sigma_s,
        sigma_r,
        // splat-only fields, unused on this path
        width: 0,
        height: 0,
    };
    slice_cells(
        cells,
        &h,
        input,
        output,
        threshold,
        RoiIn { x: roi_x, y: roi_y, width: roi_width, height: roi_height, scale: roi_scale },
        iscale,
    );
}

/// Separable `[1 4 6 4 1]/16` Gaussian along the `offset3` axis (`size3` cells),
/// for each of `size1 × size2` lines — a faithful port of the C `blur_line`
/// (4-field cell variant). In-place, with running-buffer boundary handling.
fn blur_line(
    buf: &mut [Cell],
    offset1: usize,
    offset2: usize,
    offset3: usize,
    size1: usize,
    size2: usize,
    size3: usize,
) {
    // <4 cells: the boundary block reads `buf[index + 2*offset3]`, off the line. The
    // grid is always >= 5 per axis so this never fires on the real path; kept as
    // cheap defence (C would do a benign OOB read there; Rust would panic).
    if size3 < 4 {
        return;
    }
    let (w0, w1, w2) = (6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0);
    for k in 0..size1 {
        for j in 0..size2 {
            // Per-line start. The C runs one accumulator with a signed
            // `+= offset2 - offset3*size3` fixup at each line end; recomputing the
            // start is equivalent and avoids a usize underflow when
            // `offset2 < offset3*size3`.
            let mut index = k * offset1 + j * offset2;

            let mut tmp1 = buf[index];
            buf[index] = buf[index] * w0 + buf[index + offset3] * w1 + buf[index + 2 * offset3] * w2;
            index += offset3;

            let mut tmp2 = buf[index];
            buf[index] = buf[index] * w0
                + (buf[index + offset3] + tmp1) * w1
                + buf[index + 2 * offset3] * w2;
            index += offset3;

            for _i in 2..size3 - 2 {
                let tmp3 = buf[index];
                buf[index] = buf[index] * w0
                    + (buf[index + offset3] + tmp2) * w1
                    + (buf[index + 2 * offset3] + tmp1) * w2;
                index += offset3;
                tmp1 = tmp2;
                tmp2 = tmp3;
            }

            let tmp3 = buf[index];
            buf[index] = buf[index] * w0 + (buf[index + offset3] + tmp2) * w1 + tmp1 * w2;
            index += offset3;
            buf[index] = buf[index] * w0 + tmp3 * w1 + tmp2 * w2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUE_SIGMA_SQ: f32 = core::f32::consts::PI * core::f32::consts::PI / 8.0;

    fn roi(w: i32, h: i32) -> RoiIn {
        RoiIn { x: 0, y: 0, width: w, height: h, scale: 1.0 }
    }

    /// Packed-Lab buffer from per-pixel `(L, a, b)` closure (alpha = 1).
    fn img(w: usize, h: usize, f: impl Fn(usize, usize) -> (f32, f32, f32)) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h * 4];
        for j in 0..h {
            for i in 0..w {
                let (l, a, b) = f(i, j);
                let p = 4 * (j * w + i);
                v[p] = l;
                v[p + 1] = a;
                v[p + 2] = b;
                v[p + 3] = 1.0;
            }
        }
        v
    }

    fn run(input: &[f32], w: i32, h: i32, thr: f32, ss: f32, sr: f32, prec: Precedence) -> Vec<f32> {
        let r = roi(w, h);
        let mut g = ColorReconstruct::new(r, 1.0, ss, sr);
        g.splat(input, thr, prec);
        g.blur();
        let mut out = vec![0.0f32; (w * h * 4) as usize];
        g.slice(input, &mut out, thr, r, 1.0);
        out
    }

    #[test]
    fn grid_dims_match_the_c_formula() {
        // roi 400×300, sigma_s=10, sigma_r=20:
        //   _x=round(400/10)=40 -> clamp(40,4,500)+1 = 41
        //   _y=round(300/10)=30 -> 31
        //   _z=round(100/20)=5  -> clamp(5,4,100)+1 = 6
        //   sigma_s=max(300/30, 400/40)=10, sigma_r=100/5=20
        let g = ColorReconstruct::new(roi(400, 300), 1.0, 10.0, 20.0);
        assert_eq!((g.size_x, g.size_y, g.size_z), (41, 31, 6));
        assert!((g.sigma_s - 10.0).abs() < 1e-4 && (g.sigma_r - 20.0).abs() < 1e-4);
    }

    #[test]
    fn grid_dims_never_collapse_below_five() {
        // even a 1×1 ROI clamps every axis to the minimum 4+1 = 5.
        let g = ColorReconstruct::new(roi(1, 1), 1.0, 100.0, 100.0);
        assert!(g.size_x >= 5 && g.size_y >= 5 && g.size_z >= 5);
    }

    #[test]
    fn below_threshold_pixels_keep_their_chroma() {
        // Every pixel dark (L=10) and coloured; threshold high enough that blend=0
        // for all (blend>0 needs L > 0.95*threshold = 0.95*50 = 47.5). a/b unchanged.
        let (w, h) = (24usize, 20usize);
        let inp = img(w, h, |_, _| (10.0, 7.0, -3.0));
        let out = run(&inp, w as i32, h as i32, 50.0, 8.0, 20.0, Precedence::None);
        for p in 0..w * h {
            assert!((out[4 * p + 1] - 7.0).abs() < 1e-4, "a changed below threshold");
            assert!((out[4 * p + 2] + 3.0).abs() < 1e-4, "b changed below threshold");
        }
    }

    #[test]
    fn l_and_alpha_pass_through() {
        // L (chan 0) and alpha (chan 3) are never modified by slice, even where
        // a/b get reconstructed.
        let (w, h) = (28usize, 22usize);
        let inp = img(w, h, |i, _| {
            if i >= w / 2 { (95.0, 0.0, 0.0) } else { (30.0, 12.0, 6.0) }
        });
        let out = run(&inp, w as i32, h as i32, 60.0, 8.0, 20.0, Precedence::None);
        for p in 0..w * h {
            assert_eq!(out[4 * p], inp[4 * p], "L must pass through");
            assert_eq!(out[4 * p + 3], inp[4 * p + 3], "alpha must pass through");
        }
    }

    #[test]
    fn clipped_highlight_borrows_neighbour_colour() {
        // The purpose of the module: a bright desaturated patch (L=95, a=b=0, above
        // threshold so it's excluded from the grid) surrounded by a strongly
        // coloured field (L=30, a=+20) should have its a pulled toward the
        // neighbour colour after reconstruction.
        let (w, h) = (48usize, 40usize);
        let in_patch = |i: usize, j: usize| i >= 20 && i < 28 && j >= 16 && j < 24;
        let inp = img(w, h, |i, j| {
            if in_patch(i, j) { (95.0, 0.0, 0.0) } else { (30.0, 20.0, 0.0) }
        });
        let out = run(&inp, w as i32, h as i32, 60.0, 8.0, 20.0, Precedence::None);
        // centre of the clipped patch
        let p = 4 * (20 * w + 24);
        assert!(
            out[p + 1] > 5.0,
            "clipped highlight did not borrow neighbour chroma (a = {})",
            out[p + 1]
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn precedence_variants_are_finite() {
        let (w, h) = (32usize, 24usize);
        let inp = img(w, h, |i, j| (20.0 + ((i + j) % 30) as f32, (i % 11) as f32 - 5.0, (j % 7) as f32 - 3.0));
        for prec in [
            Precedence::None,
            Precedence::Chroma,
            Precedence::Hue { hue: 0.5, sigma_sq: HUE_SIGMA_SQ },
        ] {
            let out = run(&inp, w as i32, h as i32, 60.0, 8.0, 20.0, prec);
            assert!(out.iter().all(|v| v.is_finite()), "non-finite output for {prec:?}");
        }
    }

    #[test]
    fn empty_grid_leaves_ab_untouched() {
        // If every pixel is above threshold, the grid is empty (all weights 0), so
        // slice must leave a/b as the input (the `weight > 0` guard).
        let (w, h) = (16usize, 16usize);
        let inp = img(w, h, |_, _| (95.0, 4.0, -6.0));
        let out = run(&inp, w as i32, h as i32, 50.0, 8.0, 20.0, Precedence::None);
        for p in 0..w * h {
            assert!((out[4 * p + 1] - 4.0).abs() < 1e-4, "a moved on empty grid");
            assert!((out[4 * p + 2] + 6.0).abs() < 1e-4, "b moved on empty grid");
        }
    }

    #[test]
    fn clamps_sends_nan_to_low_bound() {
        assert_eq!(clamps(f32::NAN, 0.0, 5.0), 0.0);
        assert_eq!(clamps(7.0, 0.0, 5.0), 5.0);
        assert_eq!(clamps(-1.0, 0.0, 5.0), 0.0);
        assert_eq!(clamps(3.0, 0.0, 5.0), 3.0);
    }

    // ── m4-138 FFI parity ──

    /// The method grid's cells as the flat float layout the C buffer uses.
    fn cells_as_f32(g: &ColorReconstruct) -> Vec<f32> {
        let n = g.size_x * g.size_y * g.size_z * 4;
        unsafe { std::slice::from_raw_parts(g.buf.as_ptr() as *const f32, n) }.to_vec()
    }

    #[test]
    fn cell_layout_matches_the_c_struct_for_ffi_aliasing() {
        // repr(C) pins size and field order: {L, a, b, weight} as 4 f32s
        assert_eq!(std::mem::size_of::<Cell>(), 16);
        let c = Cell { l: 1.0, a: 2.0, b: 3.0, weight: 4.0 };
        let raw = unsafe { std::slice::from_raw_parts(&c as *const Cell as *const f32, 4) };
        assert_eq!(raw, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ffi_splat_matches_the_method_for_every_precedence() {
        let (w, h) = (23usize, 17usize);
        let input = img(w, h, |i, j| {
            (((i * 7 + j * 3) % 100) as f32,
             ((i as i64 * 5 - j as i64 * 11).rem_euclid(40)) as f32 - 20.0,
             ((j as i64 * 13 - i as i64 * 2).rem_euclid(40)) as f32 - 20.0)
        });
        let cases = [
            Precedence::None,
            Precedence::Chroma,
            // hue chosen to exercise the [-π, π] wrap-around branch
            Precedence::Hue { hue: -2.9, sigma_sq: HUE_SIGMA_SQ },
            Precedence::Hue { hue: 0.7, sigma_sq: HUE_SIGMA_SQ },
        ];
        for prec in cases {
            let mut m = ColorReconstruct::new(roi(w as i32, h as i32), 1.0, 8.0, 20.0);
            m.splat(&input, 90.0, prec);
            let want = cells_as_f32(&m);

            let n = m.size_x * m.size_y * m.size_z;
            let mut raw = vec![0f32; n * 4];
            unsafe {
                darkroom_colorreconstruct_splat(
                    raw.as_mut_ptr(), m.size_x, m.size_y, m.size_z,
                    m.width, m.height, m.x, m.y, m.scale, m.sigma_s, m.sigma_r,
                    input.as_ptr(), 90.0,
                    match prec {
                        Precedence::None => 0,
                        Precedence::Chroma => 1,
                        Precedence::Hue { .. } => 2,
                    },
                    if let Precedence::Hue { hue, .. } = prec { hue } else { 0.0 },
                    HUE_SIGMA_SQ,
                );
            }
            assert_eq!(raw.len(), want.len(), "{prec:?}");
            for (k, (a, b)) in raw.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "cell float {k} diverged ({prec:?})");
            }
        }
    }

    #[test]
    fn ffi_blur_line_matches_the_method_blur_bit_exact() {
        let (w, h) = (19usize, 14usize);
        let input = img(w, h, |i, j| (((i * 11 + j) % 95) as f32, (i % 17) as f32 - 8.0,
                                     (j % 15) as f32 - 7.0));
        // method path
        let mut m = ColorReconstruct::new(roi(w as i32, h as i32), 1.0, 6.0, 25.0);
        m.splat(&input, 90.0, Precedence::Chroma);
        m.blur();
        let want = cells_as_f32(&m);

        // FFI path over a byte-copy of the pre-blur grid
        let mut pre = ColorReconstruct::new(roi(w as i32, h as i32), 1.0, 6.0, 25.0);
        pre.splat(&input, 90.0, Precedence::Chroma);
        let mut raw = cells_as_f32(&pre);
        let (sx, sy, sz) = (pre.size_x, pre.size_y, pre.size_z);
        unsafe {
            darkroom_colorreconstruct_blur_line(raw.as_mut_ptr(), sx * sy, sx, 1, sz, sy, sx);
            darkroom_colorreconstruct_blur_line(raw.as_mut_ptr(), sx * sy, 1, sx, sz, sx, sy);
            darkroom_colorreconstruct_blur_line(raw.as_mut_ptr(), 1, sx, sx * sy, sx, sy, sz);
        }
        assert_eq!(raw.len(), want.len());
        for (k, (a, b)) in raw.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "cell float {k} diverged after blur");
        }
    }

    #[test]
    fn ffi_slice_matches_the_method_bit_exact_incl_rescaled_roi() {
        let (w, h) = (21usize, 16usize);
        let input = img(w, h, |i, j| (((i * 9 + j * 5) % 100) as f32, ((i * 3) % 30) as f32 - 15.0,
                                      ((j * 7) % 30) as f32 - 15.0));
        // NONZERO init origin so grid_rescale_h's `- h.x`/`- h.y` subtrahends
        // are actually exercised (roi() builds x:0,y:0)
        let mut g = ColorReconstruct::new(
            RoiIn { x: 5, y: 7, width: w as i32, height: h as i32, scale: 1.0 },
            1.0,
            7.0,
            18.0,
        );
        g.splat(&input, 88.0, Precedence::Chroma);
        g.blur();

        // slice ROI differs from the init ROI → rescale ≠ 1 and an offset origin
        let sroi = RoiIn { x: 3, y: 2, width: w as i32 - 5, height: h as i32 - 4, scale: 0.8 };
        let mut out_m = vec![0f32; sroi.width as usize * sroi.height as usize * 4];
        g.slice(&input, &mut out_m, 88.0, sroi, 1.0);

        let mut out_f = vec![0f32; out_m.len()];
        // slice indexes its input LINEARLY over the roi dims (rw×rh packed Lab),
        // exactly like the C hands it a roi-sized buffer
        let sinput = input[..out_m.len()].to_vec();
        unsafe {
            darkroom_colorreconstruct_slice(
                g.buf.as_ptr() as *const f32, g.size_x, g.size_y, g.size_z,
                g.x, g.y, g.scale, g.sigma_s, g.sigma_r,
                sinput.as_ptr(), out_f.as_mut_ptr(),
                88.0, sroi.x, sroi.y, sroi.width, sroi.height, sroi.scale, 1.0,
            );
        }
        assert_eq!(out_f.len(), out_m.len());
        for (k, (a, b)) in out_f.iter().zip(&out_m).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "output float {k} diverged");
        }

        // aliased in==out leg: the safety doc permits it (per-pixel
        // read-before-write); pin it so a future tiling/vectorising refactor of
        // slice_cells cannot silently break the contract
        let mut alias = sinput.clone();
        unsafe {
            darkroom_colorreconstruct_slice(
                g.buf.as_ptr() as *const f32, g.size_x, g.size_y, g.size_z,
                g.x, g.y, g.scale, g.sigma_s, g.sigma_r,
                alias.as_ptr(), alias.as_mut_ptr(),
                88.0, sroi.x, sroi.y, sroi.width, sroi.height, sroi.scale, 1.0,
            );
        }
        assert_eq!(alias.len(), out_m.len());
        for (k, (a, b)) in alias.iter().zip(&out_m).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "aliased output float {k} diverged");
        }
    }

    #[test]
    fn ffi_splat_unknown_precedence_behaves_like_none() {
        // the C switch's default arm weights every out-of-enum value like NONE
        let (w, h) = (11usize, 9usize);
        let input = img(w, h, |i, j| {
            (((i * 3 + j) % 90) as f32, (i % 13) as f32 - 6.0, (j % 11) as f32 - 5.0)
        });
        let g = ColorReconstruct::new(roi(w as i32, h as i32), 1.0, 8.0, 20.0);
        let mut unknown = vec![0f32; g.buf.len() * 4];
        let mut none = vec![0f32; g.buf.len() * 4];
        unsafe {
            for (buf, prec) in [(&mut unknown, 7i32), (&mut none, 0i32)] {
                darkroom_colorreconstruct_splat(
                    buf.as_mut_ptr(), g.size_x, g.size_y, g.size_z, w, h,
                    g.x, g.y, g.scale, g.sigma_s, g.sigma_r,
                    input.as_ptr(), 90.0, prec, 1.5, HUE_SIGMA_SQ,
                );
            }
        }
        assert_eq!(unknown, none);
    }

    #[test]
    fn ffi_exports_reject_null_and_degenerate_args_without_panicking() {
        unsafe {
            darkroom_colorreconstruct_splat(
                std::ptr::null_mut(), 1, 1, 1, 1, 1, 0, 0, 1.0, 1.0, 1.0,
                std::ptr::null(), 100.0, 0, 0.0, HUE_SIGMA_SQ,
            );
            darkroom_colorreconstruct_blur_line(std::ptr::null_mut(), 1, 1, 1, 1, 1, 1);
            darkroom_colorreconstruct_slice(
                std::ptr::null(), 1, 1, 1, 0, 0, 1.0, 1.0, 1.0,
                std::ptr::null(), std::ptr::null_mut(), 100.0,
                0, 0, 0, 0, 1.0, 1.0,
            );
            // zero/overflowing dims must not construct slices either
            let mut buf = [0f32; 16];
            let inp = [50f32; 4];
            darkroom_colorreconstruct_splat(
                buf.as_mut_ptr(), 0, 0, 0, 1, 1, 0, 0, 1.0, 1.0, 1.0,
                inp.as_ptr(), 100.0, 0, 0.0, HUE_SIGMA_SQ,
            );
            darkroom_colorreconstruct_slice(
                buf.as_ptr(), 1, 1, 1, 0, 0, 1.0, 1.0, 1.0,
                inp.as_ptr(), buf.as_mut_ptr(), 100.0,
                0, 0, 0, -3, 1.0, 1.0,
            );
        }
    }
}
