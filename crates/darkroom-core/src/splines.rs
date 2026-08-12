//! Cubic-spline machinery shared by the curve-driven IOPs.
//!
//! A Rust port of the spline classes in `src/common/splines.cpp` (V2) whose
//! tangent rules the older `src/common/curve_tools.c` (V1) shares: V1's
//! `catmull_rom_set` computes exactly the non-periodic tangents of
//! `Catmull_Rom_spline::init`, and both evaluate the same cubic Hermite basis.
//! The two differ only in their *samplers* — rounding and clamping — which is
//! why those live with each IOP rather than here.
//!
//! Extracted from `iop::colorzones` when `iop::lowlight` needed the same code
//! (m4-110); keeping one copy means a fix to the interpolation reaches every
//! curve module.

/// Spline type codes, matching `src/common/curve_tools.h`.
pub const CUBIC_SPLINE: u32 = 0;
pub const CATMULL_ROM: u32 = 1;
pub const MONOTONE_HERMITE: u32 = 2;

/// One knot of a cubic Hermite spline: an (x, y) pair with a stored tangent dy.
#[derive(Clone, Debug)]
pub(crate) struct Knot {
    pub x: f32,
    pub y: f32,
    pub dy: f32,
}

/// Evaluate a cubic Hermite spline at `x` using the pre-computed knots.
/// Ports `spline_base::operator()` — non-periodic clamps x and uses linear
/// extrapolation at the boundaries; periodic wraps x via fmod.
pub(crate) fn eval_knots(knots: &[Knot], x: f32, x_lim: (f32, f32), y_lim: (f32, f32), periodic: bool) -> f32 {
    if knots.len() == 1 {
        return knots[0].y;
    }
    let n = knots.len();
    let (n0, n1, h);
    let x = if periodic {
        let period = x_lim.1 - x_lim.0;
        let mut xw = x % period;
        if xw < knots[0].x {
            xw += period;
        }
        let n0_raw = knots.partition_point(|k| k.x <= xw);
        let n0_idx = if n0_raw > 0 { n0_raw - 1 } else { n - 1 };
        let n1_idx = if n0_idx + 1 < n { n0_idx + 1 } else { 0 };
        n0 = n0_idx;
        n1 = n1_idx;
        if n1 > n0 {
            h = knots[n1].x - knots[n0].x;
        } else {
            h = knots[n1].x - (knots[n0].x - period);
        }
        xw
    } else {
        let xc = x.clamp(x_lim.0, x_lim.1);
        if xc >= knots[0].x {
            let n0_raw = knots.partition_point(|k| k.x <= xc);
            n0 = if n0_raw > 0 {
                (n0_raw - 1).min(n - 2)
            } else {
                0
            };
        } else {
            n0 = 0;
        }
        n1 = n0 + 1;
        h = knots[n1].x - knots[n0].x;
        xc
    };

    let y = if !periodic && (x <= knots[0].x || x >= knots[n - 1].x) {
        // Linear extrapolation at the boundaries
        let p = if x <= knots[0].x { &knots[0] } else { &knots[n - 1] };
        p.y + (x - p.x) * p.dy
    } else {
        let dx = (x - knots[n0].x) / h;
        let dx2 = dx * dx;
        let dx3 = dx2 * dx;
        let h00 = 2.0 * dx3 - 3.0 * dx2 + 1.0;
        let h10 = dx3 - 2.0 * dx2 + dx;
        let h01 = -2.0 * dx3 + 3.0 * dx2;
        let h11 = dx3 - dx2;
        h00 * knots[n0].y + h10 * h * knots[n0].dy + h01 * knots[n1].y + h11 * h * knots[n1].dy
    };
    y.clamp(y_lim.0, y_lim.1)
}

/// Catmull-Rom spline tangents: central differences, with periodic wrapping
/// for the endpoints.
pub(crate) fn compute_catmull_rom_tangents(knots: &mut [Knot], periodic: bool) {
    let n = knots.len();
    if n == 1 {
        knots[0].dy = 0.0;
        return;
    }
    if periodic {
        let period = knots[n - 1].x - knots[0].x;
        knots[0].dy = (knots[1].y - knots[n - 1].y)
            / (knots[1].x - knots[n - 1].x + period);
        for i in 1..n - 1 {
            knots[i].dy = (knots[i + 1].y - knots[i - 1].y)
                / (knots[i + 1].x - knots[i - 1].x);
        }
        knots[n - 1].dy = (knots[0].y - knots[n - 2].y)
            / (knots[0].x - knots[n - 2].x + period);
    } else {
        knots[0].dy = (knots[1].y - knots[0].y) / (knots[1].x - knots[0].x);
        for i in 1..n - 1 {
            knots[i].dy = (knots[i + 1].y - knots[i - 1].y)
                / (knots[i + 1].x - knots[i - 1].x);
        }
        knots[n - 1].dy = (knots[n - 1].y - knots[n - 2].y)
            / (knots[n - 1].x - knots[n - 2].x);
    }
}

/// Fritsch-Carlson monotone Hermite tangents (non-periodic).
pub(crate) fn compute_monotone_hermite_tangents(knots: &mut [Knot]) {
    let n = knots.len();
    if n == 1 {
        knots[0].dy = 0.0;
        return;
    }
    let mut delta = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        delta.push((knots[i + 1].y - knots[i].y) / (knots[i + 1].x - knots[i].x));
    }
    knots[0].dy = delta[0];
    for i in 1..n - 1 {
        if delta[i - 1] * delta[i] <= 0.0 {
            knots[i].dy = 0.0;
        } else {
            knots[i].dy = (delta[i - 1] + delta[i]) / 2.0;
        }
    }
    if n >= 2 {
        knots[n - 1].dy = delta[n - 2];
    }
    for i in 0..n - 1 {
        if delta[i].abs() < f32::EPSILON {
            knots[i].dy = 0.0;
            knots[i + 1].dy = 0.0;
        } else {
            let alpha = knots[i].dy / delta[i];
            let beta = knots[i + 1].dy / delta[i];
            let tau = alpha * alpha + beta * beta;
            if tau > 9.0 {
                knots[i].dy = 3.0 * alpha * delta[i] / tau.sqrt();
                knots[i + 1].dy = 3.0 * beta * delta[i] / tau.sqrt();
            }
        }
    }
}

/// SIAM-variant monotone Hermite: `G(S1, S2, h1, h2)` function.
/// Ports `monotone_hermite_spline_variant::G`.
pub(crate) fn monotone_g(s1: f32, s2: f32, h1: f32, h2: f32) -> f32 {
    if s1 * s2 > 0.0 {
        let alpha = (h1 + 2.0 * h2) / (3.0 * (h1 + h2));
        s1 * s2 / (alpha * s2 + (1.0 - alpha) * s1)
    } else {
        0.0
    }
}

/// SIAM-variant monotone Hermite tangents (periodic).
/// Ports `monotone_hermite_spline_variant::init` (periodic branch).
pub(crate) fn compute_monotone_hermite_variant_tangents_periodic(knots: &mut [Knot], period: f32) {
    let n = knots.len();
    if n == 1 {
        knots[0].dy = 0.0;
        return;
    }
    let mut h = Vec::with_capacity(n);
    let mut delta = Vec::with_capacity(n);
    for i in 0..n - 1 {
        h.push(knots[i + 1].x - knots[i].x);
        delta.push((knots[i + 1].y - knots[i].y) / (knots[i + 1].x - knots[i].x));
    }
    h.push(knots[0].x - knots[n - 1].x + period);
    delta.push((knots[0].y - knots[n - 1].y) / (knots[0].x - knots[n - 1].x + period));

    knots[0].dy = monotone_g(delta[n - 1], delta[0], h[n - 1], h[0]);
    for i in 1..n {
        knots[i].dy = monotone_g(delta[i - 1], delta[i], h[i - 1], h[i]);
    }
}

/// Natural cubic spline tangents (periodic or non-periodic).
/// Ports `smooth_cubic_spline::init` — set up and solve the tridiagonal
/// (banded) linear system for non-periodic, full N×N for periodic.
pub(crate) fn compute_smooth_cubic_tangents(knots: &mut [Knot], periodic: bool, period: f32) {
    let n = knots.len();
    if n == 1 {
        knots[0].dy = 0.0;
        return;
    }

    let mut delta_x = Vec::with_capacity(if periodic { n } else { n - 1 });
    let mut delta_y = Vec::with_capacity(if periodic { n } else { n - 1 });
    for i in 0..n - 1 {
        delta_x.push(knots[i + 1].x - knots[i].x);
        delta_y.push(knots[i + 1].y - knots[i].y);
    }
    if periodic {
        delta_x.push(knots[0].x - knots[n - 1].x + period);
        delta_y.push(knots[0].y - knots[n - 1].y);
    }

    let is_banded = !periodic;
    let mat_size = if is_banded { 3 * n } else { n * n };
    let mut a = vec![0.0f32; mat_size];
    let mut b = vec![0.0f32; n];

    // Helper: read matrix element
    let a_get = |a: &[f32], i: usize, j: usize| -> f32 {
        if is_banded {
            if i == j { return a[i + n]; }
            if i + 1 == j { return a[i]; }
            if i == j + 1 { return a[i + 2 * n]; }
            return 0.0;
        }
        a[i + n * j]
    };
    // For writes, we inline directly to avoid borrow conflicts.

    // Set up interior rows, matrix, LU factor, and solve.
    // Split into two code paths so the compiler can see is_banded is constant.
    if is_banded {
        // ── Banded (non-periodic) ──────────────────────────────────────
        for i in 1..n - 1 {
            a[i + 2 * n] = delta_x[i - 1] / 6.0;                    // (i, i-1)
            a[i + n] = (delta_x[i - 1] + delta_x[i]) / 3.0;         // (i, i)
            a[i] = delta_x[i] / 6.0;                                 // (i, i+1)
            b[i] = delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1];
        }
        a[0 + n] = 1.0;                // (0, 0)
        a[(n - 1) + n] = 1.0;          // (n-1, n-1)
        b[0] = 0.0;
        b[n - 1] = 0.0;

        // LU factorisation
        for i in 0..n - 1 {
            let t1 = a_get(&a, i, i);
            if t1 == 0.0 { return; }
            let v = a_get(&a, i + 1, i) / t1;
            a[(i + 1) + 2 * n] = v;                                  // (i+1, i)
            let v = a_get(&a, i + 1, i + 1) - a_get(&a, i + 1, i) * a_get(&a, i, i + 1);
            a[(i + 1) + n] = v;                                       // (i+1, i+1)
        }
        // LU solve
        for i in 0..n {
            if i > 0 {
                b[i] -= a_get(&a, i, i - 1) * b[i - 1];
            }
        }
        for i in (0..n).rev() {
            if i + 1 < n {
                b[i] -= a_get(&a, i, i + 1) * b[i + 1];
            }
            b[i] /= a_get(&a, i, i);
        }
    } else {
        // ── Full matrix (periodic) ────────────────────────────────────
        for i in 1..n - 1 {
            a[i + n * (i - 1)] = delta_x[i - 1] / 6.0;
            a[i + n * i] = (delta_x[i - 1] + delta_x[i]) / 3.0;
            a[i + n * (i + 1)] = delta_x[i] / 6.0;
            b[i] = delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1];
        }
        // Column-major: element (row, col) lives at `row + n * col`. Column 0
        // is written as a bare row index (clippy rejects the literal `n * 0`).
        a[0] = (delta_x[n - 1] + delta_x[0]) / 3.0;                       // (0, 0)
        a[(n - 1) + n * (n - 1)] = (delta_x[n - 2] + delta_x[n - 1]) / 3.0; // (n-1, n-1)
        b[0] = delta_y[0] / delta_x[0] - delta_y[n - 1] / delta_x[n - 1];
        b[n - 1] = delta_y[n - 1] / delta_x[n - 1] - delta_y[n - 2] / delta_x[n - 2];
        if n > 2 {
            a[n] = delta_x[0] / 6.0;                                      // (0, 1)
            a[(n - 1) + n * (n - 2)] = delta_x[n - 2] / 6.0;              // (n-1, n-2)
            a[n * (n - 1)] = delta_x[n - 1] / 6.0;                        // (0, n-1)
            a[n - 1] = delta_x[n - 1] / 6.0;                              // (n-1, 0)
        } else {
            a[n] = (delta_x[0] + delta_x[1]) / 6.0;                       // (0, 1)
            a[1] = (delta_x[0] + delta_x[1]) / 6.0;                       // (1, 0)
        }

        // LU factorisation
        for i in 0..n - 1 {
            let t1 = a_get(&a, i, i);
            if t1 == 0.0 { return; }
            for k in i + 1..n {
                let v = a_get(&a, k, i) / t1;
                a[k + n * i] = v;
                for j in i + 1..n {
                    let v = a_get(&a, k, j) - a_get(&a, k, i) * a_get(&a, i, j);
                    a[k + n * j] = v;
                }
            }
        }
        // LU solve
        for i in 0..n {
            for k in 0..i {
                b[i] -= a_get(&a, i, k) * b[k];
            }
        }
        for i in (0..n).rev() {
            for k in i + 1..n {
                b[i] -= a_get(&a, i, k) * b[k];
            }
            b[i] /= a_get(&a, i, i);
        }
    }

    // Compute tangents from second derivatives
    let mut c_i = 0.0f32;
    for i in 0..n - 1 {
        c_i = delta_y[i] / delta_x[i] - delta_x[i] / 6.0 * (b[i + 1] - b[i]);
        knots[i].dy = -delta_x[i] * b[i] / 2.0 + c_i;
    }
    if periodic {
        knots[n - 1].dy = delta_x[n - 2] * b[n - 1] / 2.0 + c_i;
    } else {
        knots[n - 1].dy = c_i;
    }
}
