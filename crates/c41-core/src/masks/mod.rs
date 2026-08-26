//! Drawn-mask shape rendering, ported from `src/develop/masks/`.
//!
//! Each shape module ports the OMP loops of its `<shape>_get_mask` (full-form
//! mask over the whole pipe area) and `<shape>_get_mask_roi` (grid-sampled +
//! interpolated mask for an arbitrary ROI) implementations. The loops that
//! interleave with pixelpipe callbacks (`dt_dev_distort_backtransform_plus`
//! and friends) stay in C — only the pure point/mask arithmetic moves here,
//! exposed through `darkroom_masks_*` FFI exports exactly like every other
//! replaced C loop.
//!
//! Bit-exactness notes carried per kernel: integer sub-expressions are
//! evaluated in `i32` BEFORE the float conversion (e.g. `(grid*i + px)`),
//! `CLIP(x)` is clamp-to-[0,1], and `dt_masks_roundup(n, m)` rounds up to a
//! multiple of `m`.

pub mod circle;

/// `DT_2PI_F` (`src/common/math.h:60`). Written as `TAU`: the C literal's
/// extra digits vanish in the f32 rounding, so the bits are identical.
pub(crate) const DT_2PI_F: f32 = std::f32::consts::TAU;

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
fn circle_feather(l2: f32, total2: f32, border2: f32) -> f32 {
    let ratio = (total2 - l2) / border2;
    let f = ratio.clamp(0.0, 1.0);
    f * f
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
