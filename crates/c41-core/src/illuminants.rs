//! Illuminant colour-temperature helpers ported from
//! `src/common/illuminants.h`.
//!
//! `CCT_reverse_lookup` (formerly a `DT_OMP_FOR` loop with a custom
//! `pairmin` reduction, illuminants.h:554) brute-force scans a 65536-point
//! planckian-locus LUT to find the correlated colour temperature whose
//! chromaticity is closest to `(x, y)`. The loop body calls the pure
//! polynomial helpers `CCT_to_xy_daylight` / `CCT_to_xy_blackbody`, which
//! are ported here as private copies (the C originals stay in the header —
//! they have other callers there).
//!
//! The C reduction `pair_min(r, n)` returns `n` only when
//! `n.radius < r.radius` (strict less-than), so ties keep the earlier
//! (lower-temperature) sample; the sequential Rust port replicates that
//! first-wins semantics exactly. For non-NaN radii the minimum *radius* is
//! independent of any OpenMP thread partitioning, but an exact-float tie
//! between two threads' partial minima is combined in an
//! implementation-defined order, so the parallel C could return the
//! higher-temperature tie member where the sequential port always returns
//! the lowest — a measure-zero case for real chromaticities, bounded by one
//! LUT step (~1 K).
//!
//! Bit-exactness notes:
//! - `dt_fast_hypotf` is `sqrtf(x*x + y*y)` under the project's `-ffast-math`
//!   build (src/common/math.h:393) — ported as `(dx*dx + dy*dy).sqrt()`.
//! - The polynomial chains and `powf(step, 4)` are dense multiply-add
//!   patterns; GCC may FMA-contract them in the historical C binary where
//!   Rust does not contract, giving order-ULP differences (the same accepted
//!   class as `imagebuf.rs::linear_blend`). Test tolerances account for it.
//! - The `x_temp`/`y_temp` branch structure (including which ranges are
//!   inclusive/exclusive) is transcribed exactly — e.g. daylight is used
//!   from `T >= 4000`, and the blackbody y-polynomial switches at 2222 K.

/// Port of `CCT_to_xy_daylight` (illuminants.h:138): the closest daylight
/// illuminant chromaticity for a colour temperature in 4000–25000 K.
fn cct_to_xy_daylight(t: f32) -> (f32, f32) {
    let mut x_temp = 0.0f32;
    if (4000.0f32..=7000.0f32).contains(&t) {
        x_temp = ((-4.6070e9f32 / t + 2.9678e6f32) / t + 0.09911e3f32) / t + 0.244063f32;
    } else if t > 7000.0f32 && t <= 25000.0f32 {
        x_temp = ((-2.0064e9f32 / t + 1.9018e6f32) / t + 0.24748e3f32) / t + 0.237040f32;
    }
    let y_temp = (-3.0f32 * x_temp + 2.87f32) * x_temp - 0.275f32;
    (x_temp, y_temp)
}

/// Port of `CCT_to_xy_blackbody` (illuminants.h:157): the closest blackbody
/// illuminant chromaticity for a colour temperature in 1667–25000 K.
fn cct_to_xy_blackbody(t: f32) -> (f32, f32) {
    let mut x_temp = 0.0f32;
    if (1667.0f32..=4000.0f32).contains(&t) {
        x_temp = ((-0.2661239e9f32 / t - 0.2343589e6f32) / t + 0.8776956e3f32) / t + 0.179910f32;
    } else if t > 4000.0f32 && t <= 25000.0f32 {
        x_temp = ((-3.0258469e9f32 / t + 2.1070379e6f32) / t + 0.2226347e3f32) / t + 0.240390f32;
    }
    let mut y_temp = 0.0f32;
    if (1667.0f32..=2222.0f32).contains(&t) {
        y_temp = ((-1.1063814f32 * x_temp - 1.34811020f32) * x_temp + 2.18555832f32) * x_temp
            - 0.20219683f32;
    } else if t > 2222.0f32 && t <= 4000.0f32 {
        y_temp = ((-0.9549476f32 * x_temp - 1.37418593f32) * x_temp + 2.09137015f32) * x_temp
            - 0.16748867f32;
    } else if t > 4000.0f32 && t <= 25000.0f32 {
        y_temp = ((3.0817580f32 * x_temp - 5.87338670f32) * x_temp + 3.75112997f32) * x_temp
            - 0.37001483f32;
    }
    (x_temp, y_temp)
}

/// `dt_fast_hypotf` as compiled under `-ffast-math` (math.h:393).
fn fast_hypot(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

const T_MIN: f32 = 1667.0f32;
const T_MAX: f32 = 25000.0f32;
const T_RANGE: f32 = T_MAX - T_MIN;
const LUT_SAMPLES: usize = 1 << 16;

/// Port of `CCT_reverse_lookup` (illuminants.h:540): brute-force
/// reverse-lookup of the closest correlated colour temperature over the
/// planckian locus for an arbitrary `(x, y)` chromaticity. Returns the
/// temperature whose locus point is nearest (Euclidean distance in xy).
pub fn cct_reverse_lookup(x: f32, y: f32) -> f32 {
    let mut min_radius = f32::MAX;
    let mut min_temperature = 0.0f32;
    for i in 0..LUT_SAMPLES {
        // we need more values for the low temperatures, so we scale the step with a power
        let step = (i as f32 / (LUT_SAMPLES - 1) as f32).powf(4.0f32);
        // Current temperature in the lookup range
        let t = T_MIN + step * T_RANGE;
        // Current x, y chromaticity
        let (x_bb, y_bb) = if t >= 4000.0f32 {
            cct_to_xy_daylight(t)
        } else {
            cct_to_xy_blackbody(t)
        };
        // Compute distance between current planckian chromaticity and input
        let radius = fast_hypot(x_bb - x, y_bb - y);
        // If we found a smaller radius, save it (strict <, first wins ties)
        if radius < min_radius {
            min_radius = radius;
            min_temperature = t;
        }
    }
    min_temperature
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// Replaces the former brute-force reverse-lookup loop in `CCT_reverse_lookup`
/// (illuminants.h).
#[no_mangle]
pub unsafe extern "C" fn darkroom_illuminants_cct_reverse_lookup(x: f32, y: f32) -> f32 {
    cct_reverse_lookup(x, y)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nearly(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // Reference values generated by compiling the actual C helpers from
    // src/common/illuminants.h with the project's Release flags
    // (-O3 -ffast-math -fno-finite-math-only) and printing %.9g — tolerance
    // absorbs the FMA-contraction order-ULP differences documented above.
    #[test]
    fn blackbody_polynomial_matches_c_reference() {
        let pins: [(f32, f32, f32); 6] = [
            (1667.0, 0.564638317, 0.402887106),
            (1800.0, 0.54955405, 0.40811646),
            (2000.0, 0.526902616, 0.413264871),
            (2222.0, 0.503187537, 0.415250897),
            (3000.0, 0.4365789, 0.404174477),
            (4000.0, 0.380528301, 0.376733512),
        ];
        for (t, x_ref, y_ref) in pins {
            let (x, y) = cct_to_xy_blackbody(t);
            assert!(nearly(x, x_ref, 1e-6), "bb x at {t}: {x} vs {x_ref}");
            assert!(nearly(y, y_ref, 1e-6), "bb y at {t}: {y} vs {y_ref}");
        }
    }

    #[test]
    fn daylight_polynomial_matches_c_reference() {
        let pins: [(f32, f32, f32); 5] = [
            (4000.0, 0.38234365, 0.383766204),
            (5000.0, 0.345741004, 0.358666092),
            (7000.0, 0.305357426, 0.321646303),
            (12000.0, 0.26970917, 0.280836135),
            (25000.0, 0.249853671, 0.254799396),
        ];
        for (t, x_ref, y_ref) in pins {
            let (x, y) = cct_to_xy_daylight(t);
            assert!(nearly(x, x_ref, 1e-6), "dl x at {t}: {x} vs {x_ref}");
            assert!(nearly(y, y_ref, 1e-6), "dl y at {t}: {y} vs {y_ref}");
        }
    }

    #[test]
    fn out_of_range_temperatures_return_zero() {
        // Blackbody leaves BOTH outputs at 0 outside its valid ranges
        assert_eq!(cct_to_xy_blackbody(1000.0), (0.0, 0.0));
        assert_eq!(cct_to_xy_blackbody(30000.0), (0.0, 0.0));
        // Daylight's x_temp is range-gated but y_temp is computed
        // unconditionally from x_temp, so out-of-range yields (0, -0.275)
        assert_eq!(cct_to_xy_daylight(3000.0), (0.0, -0.275));
        assert_eq!(cct_to_xy_daylight(30000.0), (0.0, -0.275));
    }

    #[test]
    fn reverse_lookup_recovers_locus_temperatures() {
        // Self-consistency: reverse-looking-up a locus point must return a
        // temperature near the one that generated it (LUT quantization is
        // sub-kelvin at these temperatures). Reference values from the C
        // loop compiled with the project flags.
        let cases: [(f32, f32); 4] = [
            (2000.0, 1999.98669),   // blackbody region
            (3500.0, 3499.92627),   // blackbody region
            (5000.0, 4999.90723),   // daylight region
            (15000.0, 15000.2695),  // daylight region
        ];
        for (t, t_ref) in cases {
            let (x, y) = if t >= 4000.0 {
                cct_to_xy_daylight(t)
            } else {
                cct_to_xy_blackbody(t)
            };
            let found = cct_reverse_lookup(x, y);
            assert!(
                nearly(found, t_ref, 1.0),
                "reverse lookup at {t}K: {found} vs C reference {t_ref}"
            );
            assert!(nearly(found, t, 2.0), "reverse lookup at {t}K: {found}");
        }
    }

    #[test]
    fn reverse_lookup_off_locus_points() {
        // Off-locus chromaticities: pinned against the C loop's output
        assert!(nearly(cct_reverse_lookup(0.35, 0.25), 8534.87793, 2.0));
        assert!(nearly(cct_reverse_lookup(0.30, 0.30), 8190.95898, 2.0));
    }

    #[test]
    fn reverse_lookup_at_locus_point() {
        // The lookup at an exact locus point reproduces its generating
        // temperature (within LUT quantization). Displacements move the
        // answer along/off the locus — bounds measured empirically: the
        // locus is steep near 6500 K, so Δx = 0.002 moves ~61 K, and the
        // gross Δx = 0.05 displacement moves ~1349 K.
        let (x0, y0) = cct_to_xy_daylight(6500.0);
        let t_at = cct_reverse_lookup(x0, y0);
        assert!(nearly(t_at, 6500.0, 2.0));
        let t_near = cct_reverse_lookup(x0 + 0.002, y0);
        assert!(
            (t_near - 6500.0).abs() < 100.0,
            "near displacement moved to {t_near}"
        );
        let t_far = cct_reverse_lookup(x0 + 0.05, y0);
        assert!(
            (t_far - 6500.0).abs() > 500.0,
            "far displacement stayed at {t_far}"
        );
    }

    #[test]
    fn ffi_round_trip() {
        let direct = cct_reverse_lookup(0.3457, 0.3586);
        let ffi = unsafe { darkroom_illuminants_cct_reverse_lookup(0.3457, 0.3586) };
        assert_eq!(direct.to_bits(), ffi.to_bits());
    }
}
