//! Fast math approximations shared by multiple IOPs.
//!
//! Mirrors the inline helpers in src/common/math.h. These are bit-twiddled
//! polynomial approximations to logf / expf with documented error bounds
//! and matching constants, so the Rust pipeline produces byte-identical
//! pixel output to the C path even when the host CPU lacks SVML.

pub const M_LN2: f32 = std::f32::consts::LN_2;

/// IEEE-754 polynomial approximation of log2(x), matching `fastlog2()` in
/// src/common/math.h byte-for-byte.
///
/// Valid for positive x; behaviour for x <= 0 is undefined (the C version
/// reads the float bits unconditionally and returns garbage rather than NaN).
#[inline(always)]
pub fn fastlog2(x: f32) -> f32 {
    let vx_i = x.to_bits();
    let mx_i = (vx_i & 0x007F_FFFF) | 0x3f00_0000;
    let mx_f = f32::from_bits(mx_i);
    let y = vx_i as f32 * 1.1920928955078125e-7_f32;

    y - 124.22551499
        - 1.498030302 * mx_f
        - 1.72587999 / (0.3520887068 + mx_f)
}

/// Natural log via `fastlog2`, matching `fastlog()` in src/common/math.h.
#[inline(always)]
pub fn fastlog(x: f32) -> f32 {
    M_LN2 * fastlog2(x)
}

// ── PRNG: xoshiro128+ / splitmix32 / noise generators ────────────────────────
// Mirrors src/develop/noise_generator.h exactly so all three noise types
// (uniform, Gaussian, Poissonian) produce bit-identical output to the C path.

/// `splitmix32(seed)` — fast seed hashing for xoshiro128+ state init.
/// Matches splitmix32() in noise_generator.h:31.
#[inline(always)]
pub fn splitmix32(seed: u64) -> u32 {
    let result = (seed ^ (seed >> 33)).wrapping_mul(0x62a9d9ed799705f5u64);
    let result = (result ^ (result >> 28)).wrapping_mul(0xcb24d0a5c88c35b3u64);
    (result >> 32) as u32
}

/// Rotate-left 32.
#[inline(always)]
fn rol32(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

/// `xoshiro128+(state)` — one step of the xoshiro128+ PRNG.
/// Advances `state[4]` in-place and returns a float in [0, 1).
/// Matches xoshiro128plus() in noise_generator.h:49.
#[inline(always)]
pub fn xoshiro128plus(state: &mut [u32; 4]) -> f32 {
    let result = state[0].wrapping_add(state[3]);
    let t      = state[1] << 9;

    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3]  = rol32(state[3], 11);

    (result >> 8) as f32 * (1.0_f32 / (1u32 << 24) as f32)
}

/// Uniform noise: `mu + 2 * (xoshiro128+ - 0.5) * sigma`.
#[inline(always)]
pub fn uniform_noise(mu: f32, sigma: f32, state: &mut [u32; 4]) -> f32 {
    mu + 2.0 * (xoshiro128plus(state) - 0.5) * sigma
}

/// Gaussian noise (Box-Muller transform).
/// `flip` must alternate each call to use both cos/sin branches.
/// Matches gaussian_noise() in noise_generator.h:77.
#[inline(always)]
pub fn gaussian_noise(mu: f32, sigma: f32, flip: bool, state: &mut [u32; 4]) -> f32 {
    use std::f32::consts::TAU;
    let u1 = xoshiro128plus(state).max(f32::MIN_POSITIVE);
    let u2 = xoshiro128plus(state);
    let n  = if flip {
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    } else {
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).sin()
    };
    n * sigma + mu
}

/// Poissonian noise (Gaussian with Anscombe transform).
/// Matches poisson_noise() in noise_generator.h:95.
#[inline(always)]
pub fn poisson_noise(mu: f32, sigma: f32, flip: bool, state: &mut [u32; 4]) -> f32 {
    use std::f32::consts::TAU;
    let u1 = xoshiro128plus(state).max(f32::MIN_POSITIVE);
    let u2 = xoshiro128plus(state);
    let n  = if flip {
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    } else {
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).sin()
    };
    let r = n * sigma + 2.0 * (mu + 3.0 / 8.0).max(0.0).sqrt();
    (r * r - sigma * sigma) / 4.0 - 3.0 / 8.0
}

/// Dispatch to the correct noise distribution (scalar path).
/// `distribution`: 0 = uniform, 1 = gaussian, 2 = poissonian.
/// Matches dt_noise_generator() in noise_generator.h:111.
#[inline(always)]
pub fn dt_noise_generator(
    distribution: u32,
    mu: f32,
    param: f32,
    flip: bool,
    state: &mut [u32; 4],
) -> f32 {
    match distribution {
        1 => gaussian_noise(mu, param, flip, state),
        2 => poisson_noise(mu, param, flip, state),
        _ => uniform_noise(mu, param, state),
    }
}

/// 4-channel uniform noise.
/// Matches uniform_noise_simd() in noise_generator.h:139.
/// Only fills channels 0..3 (channel 3 stays as mu[3] with 0-sigma).
#[inline(always)]
pub fn uniform_noise_4ch(mu: &[f32; 4], sigma: &[f32; 4], state: &mut [u32; 4]) -> [f32; 4] {
    let mut out = *mu;
    for c in 0..3 {
        let n = xoshiro128plus(state);
        out[c] = mu[c] + 2.0 * (n - 0.5) * sigma[c];
    }
    out
}

/// 4-channel Gaussian noise (Box-Muller, per-channel flip).
/// `flip[c]` selects cos vs sin branch.
/// Matches gaussian_noise_simd() in noise_generator.h:157.
#[inline(always)]
pub fn gaussian_noise_4ch(
    mu: &[f32; 4],
    sigma: &[f32; 4],
    flip: &[bool; 4],
    state: &mut [u32; 4],
) -> [f32; 4] {
    use std::f32::consts::TAU;
    let mut u1 = [0.0_f32; 4];
    let mut u2 = [0.0_f32; 4];
    for c in 0..3 { u1[c] = xoshiro128plus(state).max(f32::MIN_POSITIVE); }
    for c in 0..3 { u2[c] = xoshiro128plus(state); }
    let mut out = [0.0_f32; 4];
    for c in 0..4 {
        let n = if flip[c] {
            (-2.0 * u1[c].ln()).sqrt() * (TAU * u2[c]).cos()
        } else {
            (-2.0 * u1[c].ln()).sqrt() * (TAU * u2[c]).sin()
        };
        out[c] = n * sigma[c] + mu[c];
    }
    out
}

/// 4-channel Poissonian noise (Gaussian + Anscombe transform).
/// Matches poisson_noise_simd() in noise_generator.h:196.
#[inline(always)]
pub fn poisson_noise_4ch(
    mu: &[f32; 4],
    sigma: &[f32; 4],
    flip: &[bool; 4],
    state: &mut [u32; 4],
) -> [f32; 4] {
    use std::f32::consts::TAU;
    let mut u1 = [0.0_f32; 4];
    let mut u2 = [0.0_f32; 4];
    for c in 0..3 {
        u1[c] = xoshiro128plus(state).max(f32::MIN_POSITIVE);
        u2[c] = xoshiro128plus(state);
    }
    let mut out = [0.0_f32; 4];
    for c in 0..4 {
        let n = if flip[c] {
            (-2.0 * u1[c].ln()).sqrt() * (TAU * u2[c]).cos()
        } else {
            (-2.0 * u1[c].ln()).sqrt() * (TAU * u2[c]).sin()
        };
        let r = n * sigma[c] + 2.0 * (mu[c] + 3.0 / 8.0).max(0.0).sqrt();
        out[c] = (r * r - sigma[c] * sigma[c]) / 4.0 - 3.0 / 8.0;
    }
    out
}

/// 4-channel noise generator dispatcher.
/// `distribution`: 0=uniform, 1=gaussian, 2=poissonian.
/// `flip`: [true, false, true, false] as used in filmicrgb.
/// Matches dt_noise_generator_simd() in noise_generator.h:237.
#[inline(always)]
pub fn dt_noise_generator_4ch(
    distribution: u32,
    mu: &[f32; 4],
    param: &[f32; 4],
    flip: &[bool; 4],
    state: &mut [u32; 4],
) -> [f32; 4] {
    match distribution {
        1 => gaussian_noise_4ch(mu, param, flip, state),
        2 => poisson_noise_4ch(mu, param, flip, state),
        _ => uniform_noise_4ch(mu, param, state),
    }
}

/// Cache-friendly row interleaving for à-trous wavelet passes: rows are
/// visited stride-apart so adjacent threads touch adjacent memory.
/// Matches `dwt_interleave_rows()` in src/common/dwt.h:87.
#[inline(always)]
pub fn dwt_interleave_rows(rowid: usize, height: usize, stride: usize) -> usize {
    if height <= stride {
        return rowid;
    }
    let per_pass = height.div_ceil(stride);
    let long_passes = height % stride;
    if long_passes == 0 || rowid < long_passes * per_pass {
        return (rowid / per_pass) + stride * (rowid % per_pass);
    }
    let rowid2 = rowid - long_passes * per_pass;
    long_passes + (rowid2 / (per_pass - 1)) + stride * (rowid2 % (per_pass - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastlog2_approximates_log2() {
        for &x in &[0.25_f32, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0] {
            let approx = fastlog2(x);
            let exact = x.log2();
            assert!((approx - exact).abs() < 0.05, "x={x} approx={approx} exact={exact}");
        }
    }

    #[test]
    fn fastlog_one_is_near_zero() {
        let r = fastlog(1.0);
        assert!(r.abs() < 0.05);
    }

    #[test]
    fn fastlog_uses_natural_base() {
        let r = fastlog(std::f32::consts::E);
        assert!((r - 1.0).abs() < 0.05);
    }

    // ── PRNG tests ──────────────────────────────────────────────────────────

    fn seed_state() -> [u32; 4] {
        [splitmix32(1), splitmix32(2 * 3), splitmix32(1337), splitmix32(666)]
    }

    #[test]
    fn xoshiro_output_in_unit_range() {
        let mut state = seed_state();
        for _ in 0..1000 {
            let v = xoshiro128plus(&mut state);
            assert!(v >= 0.0 && v < 1.0, "out of [0,1): {v}");
        }
    }

    #[test]
    fn xoshiro_advances_state() {
        let mut s = seed_state();
        let v1 = xoshiro128plus(&mut s);
        let v2 = xoshiro128plus(&mut s);
        assert_ne!(v1.to_bits(), v2.to_bits(), "state should advance");
    }

    #[test]
    fn gaussian_noise_mean_is_approximately_mu() {
        let mut state = seed_state();
        let n = 10_000;
        let sum: f32 = (0..n).map(|i| gaussian_noise(5.0, 1.0, i % 2 == 0, &mut state)).sum();
        let mean = sum / n as f32;
        assert!((mean - 5.0).abs() < 0.1, "mean={mean}");
    }

    #[test]
    fn uniform_noise_stays_within_bounds() {
        let mut state = seed_state();
        for _ in 0..1000 {
            let v = uniform_noise(0.5, 0.5, &mut state);
            // mu ± 2*sigma covers [−0.5, 1.5] in the worst case; check rough bounds
            assert!(v > -1.0 && v < 2.0, "uniform out of expected range: {v}");
        }
    }

    #[test]
    fn splitmix32_different_seeds_produce_different_outputs() {
        let a = splitmix32(1);
        let b = splitmix32(2);
        assert_ne!(a, b);
    }
}
