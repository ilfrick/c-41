//! Drawn-mask shape rendering, ported from `src/develop/masks/`.
//!
//! Each shape module ports the OMP loops of its `<shape>_get_mask` (whole-pipe
//! form mask) and `<shape>_get_mask_roi` (grid-sampled + interpolated mask)
//! implementations. The loops that interleave with pixelpipe callbacks (`dt_dev_distort_backtransform_plus`
//! and friends) stay in C — only the pure point/mask arithmetic moves here,
//! exposed through `darkroom_masks_*` FFI exports exactly like every other
//! replaced C loop.
//!
//! Bit-exactness notes carried per kernel: integer sub-expressions are
//! evaluated in `i32` BEFORE the float conversion (e.g. `(grid*i + px)`),
//! `CLIP(x)` is clamp-to-[0,1], and `dt_masks_roundup(n, m)` rounds up to a
//! multiple of `m`.

pub mod brush;
pub mod circle;
pub mod ellipse;
pub mod gradient;

/// `DT_2PI_F` (`src/common/math.h:60`). Written as `TAU`: the C literal's
/// extra digits vanish in the f32 rounding, so the bits are identical.
pub(crate) const DT_2PI_F: f32 = std::f32::consts::TAU;

/// `M_PI_F` (`src/common/math.h:49`). The `f` suffix rounds the double literal
/// to the same f32 bits as Rust's `PI`. Reserved for future mask shapes that need
/// angle arithmetic (gradient rotation, brush orientation, etc.).
#[allow(dead_code)]
pub(crate) const M_PI_F: f32 = std::f32::consts::PI;

/// `DT_MASKS_GRADIENT_STATE_LINEAR` (`src/develop/masks.h:93`).
pub(crate) const GRADIENT_STATE_LINEAR: i32 = 1;

/// `dt_masks_roundup` (`src/develop/masks.h:1098`): round `num` up to a
/// multiple of `mult`.
#[inline]
pub(crate) fn masks_roundup(num: i32, mult: i32) -> i32 {
    let rem = num % mult;
    if rem == 0 { num } else { num + mult - rem }
}

/// Quadratic feather falloff shared by the circle kernels: 1.0 inside the
/// hard radius, squared smoothstep down to 0.0 at the outer (radius+border)
/// ring — `sqf(CLIP((total2 - l2) / border2))`.
#[inline]
pub(crate) fn circle_feather(l2: f32, total2: f32, border2: f32) -> f32 {
    let ratio = (total2 - l2) / border2;
    let f = ratio.clamp(0.0, 1.0);
    f * f
}

/// Fill `points` (2 floats per pixel) with the pipe-area coordinate grid,
/// `points[(i*w+j)*2] = pos_x + j`, `points[(i*w+j)*2+1] = pos_y + i`.
/// Shared by `_circle_get_mask` and `_ellipse_get_mask` (identical C loops).
pub(crate) fn fill_coord_grid(points: &mut [f32], w: usize, h: usize, pos_x: f32, pos_y: f32) {
    assert_eq!(points.len(), 2 * w * h, "grid buffer must hold w*h pairs");
    let slots = points.as_chunks_mut::<2>().0;
    for i in 0..h {
        let y = i as f32 + pos_y;
        for (j, slot) in slots[i * w..(i + 1) * w].iter_mut().enumerate() {
            slot[0] = pos_x + j as f32;
            slot[1] = y;
        }
    }
}

/// Populate `points` (bbw*bbh pairs) with grid-sampled module coordinates.
/// The C computes `(grid*i + px)` in INTEGER arithmetic before the float
/// conversion — replicated here. Shared by `_circle_get_mask_roi` and
/// `_ellipse_get_mask_roi` (identical C loops, differing only OMP conditions).
#[inline]
pub(crate) fn fill_grid_points(
    points: &mut [f32],
    bbw: usize,
    bbh: usize,
    bbxm: i32,
    bbym: i32,
    px: i32,
    py: i32,
    iscale: f32,
    grid: i32,
) {
    assert_eq!(points.len(), 2 * bbw * bbh);
    for j in 0..bbh as i32 {
        let gy = (grid * (bbym + j) + py) as f32 * iscale;
        for i in 0..bbw as i32 {
            let index = (j * bbw as i32 + i) as usize;
            points[2 * index] = (grid * (bbxm + i) + px) as f32 * iscale;
            points[2 * index + 1] = gy;
        }
    }
}

/// Bilinear-splat the bbw*bbh sampled values (even lanes of `points`) over
/// rows [start_j, end_j) × cols [start_i, end_i) of the `w`-wide `buffer`.
/// The sequential float multiplications mirror the C expression order
/// (`v*(g-ii)*(g-jj)` evaluates left-to-right, converting each int operand
/// at its own multiply), and the denominator is the INT product `grid*grid`
/// converted once at the division. Shared by all shapes (identical C loops).
#[inline]
pub(crate) fn interpolate_into_buffer(
    buffer: &mut [f32],
    w: usize,
    points: &[f32],
    bbw: usize,
    start_i: i32,
    end_i: i32,
    start_j: i32,
    end_j: i32,
    grid: i32,
) {
    let g = grid;
    let denom = (g * g) as f32;
    for j in start_j..end_j {
        let jj = j.rem_euclid(g);
        let mj = j.div_euclid(g) - start_j.div_euclid(g);
        for i in start_i..end_i {
            let ii = i.rem_euclid(g);
            let mi = i.div_euclid(g) - start_i.div_euclid(g);
            let mindex = (mj * bbw as i32 + mi) as usize;
            buffer[j as usize * w + i as usize] =
                (points[mindex * 2] * (g - ii) as f32 * (g - jj) as f32
                    + points[(mindex + 1) * 2] * ii as f32 * (g - jj) as f32
                    + points[(mindex + bbw) as usize * 2] * (g - ii) as f32 * jj as f32
                    + points[(mindex + bbw + 1) * 2] * ii as f32 * jj as f32)
                    / denom;
        }
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    /// Deterministic LCG fill for test buffers (no rand dependency), matching
    /// the house pattern (`locallaplacian.rs` tests).
    pub fn lcg_fill(buffer: &mut [f32], seed: u32, scale: f32) {
        let mut s = seed;
        for v in buffer.iter_mut() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = ((s >> 16) % 1024) as f32 / 1024.0 * scale;
        }
    }
}
