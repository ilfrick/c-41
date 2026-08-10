use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct ColorZones;

impl IopProcess for ColorZones {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colorzones" }
}

const DT_IOP_COLORZONES_LUT_RES: usize = 0x10000;

// ---------------------------------------------------------------------------
// Spline helpers for build_lut (ported from src/common/splines.cpp)
// ---------------------------------------------------------------------------

pub const CUBIC_SPLINE: u32 = 0;
pub const CATMULL_ROM: u32 = 1;
pub const MONOTONE_HERMITE: u32 = 2;

/// One knot of a cubic Hermite spline: an (x, y) pair with a stored tangent dy.
#[derive(Clone, Debug)]
struct Knot {
    x: f32,
    y: f32,
    dy: f32,
}

/// Evaluate a cubic Hermite spline at `x` using the pre-computed knots.
/// Ports `spline_base::operator()` — non-periodic clamps x and uses linear
/// extrapolation at the boundaries; periodic wraps x via fmod.
fn eval_knots(knots: &[Knot], x: f32, x_lim: (f32, f32), y_lim: (f32, f32), periodic: bool) -> f32 {
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
fn compute_catmull_rom_tangents(knots: &mut [Knot], periodic: bool) {
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
fn compute_monotone_hermite_tangents(knots: &mut [Knot]) {
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
fn monotone_g(s1: f32, s2: f32, h1: f32, h2: f32) -> f32 {
    if s1 * s2 > 0.0 {
        let alpha = (h1 + 2.0 * h2) / (3.0 * (h1 + h2));
        s1 * s2 / (alpha * s2 + (1.0 - alpha) * s1)
    } else {
        0.0
    }
}

/// SIAM-variant monotone Hermite tangents (periodic).
/// Ports `monotone_hermite_spline_variant::init` (periodic branch).
fn compute_monotone_hermite_variant_tangents_periodic(knots: &mut [Knot], period: f32) {
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
fn compute_smooth_cubic_tangents(knots: &mut [Knot], periodic: bool, period: f32) {
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

/// Build a 65536-entry LUT from spline curve nodes, porting
/// `CurveDataSampleV2` / `CurveDataSampleV2Periodic` from `src/common/splines.cpp`.
///
/// `strength` is applied to node y values before spline construction:
/// `y' = y + (y - 0.5) * (strength / 100.0)`.
///
/// Returns a `Vec<f32>` of length `DT_IOP_COLORZONES_LUT_RES` (65536), with each
/// entry quantised as `round(clamp(s(x), 0, 1) * 65535) / 65536` (u16-aware).
pub fn build_lut(
    nodes_x: &[f32],
    nodes_y: &[f32],
    num_nodes: usize,
    curve_type: u32,
    periodic: bool,
    strength: f32,
) -> Vec<f32> {
    let n = num_nodes.min(nodes_x.len()).min(nodes_y.len());
    if n == 0 {
        // C code's CurveDataSampleV2 with zero anchors creates a straight line from
        // (m_min_x, m_min_y) = (0, 0) to (m_max_x, m_max_y) = (1, 1).
        let mut lut = vec![0.0f32; DT_IOP_COLORZONES_LUT_RES];
        for i in 0..DT_IOP_COLORZONES_LUT_RES {
            lut[i] = i as f32 / (DT_IOP_COLORZONES_LUT_RES as f32 - 1.0);
        }
        return lut;
    }

    // Apply strength to y values
    let mut knots: Vec<Knot> = (0..n)
        .map(|k| {
            let y = nodes_y[k] + (nodes_y[k] - 0.5) * (strength / 100.0);
            Knot { x: nodes_x[k], y, dy: 0.0 }
        })
        .collect();

    if periodic {
        // Wrap knot x into [0, 1) and sort
        let period = 1.0f32;
        for k in &mut knots {
            k.x = k.x % period;
            if k.x < 0.0 {
                k.x += period;
            }
        }
        knots.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    } else {
        knots.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    }

    // Compute tangents
    match curve_type {
        CATMULL_ROM => compute_catmull_rom_tangents(&mut knots, periodic),
        MONOTONE_HERMITE => {
            if periodic {
                compute_monotone_hermite_variant_tangents_periodic(&mut knots, 1.0);
            } else {
                compute_monotone_hermite_tangents(&mut knots);
            }
        }
        _ => compute_smooth_cubic_tangents(&mut knots, periodic, 1.0),
    }

    let x_lim = if periodic {
        (0.0, 1.0)
    } else {
        (knots[0].x, knots[n - 1].x)
    };
    let y_lim = (0.0, 1.0);

    let output_res = DT_IOP_COLORZONES_LUT_RES as f32;
    let sampling_res = DT_IOP_COLORZONES_LUT_RES;
    let res = 1.0f32 / (sampling_res as f32 - 1.0);

    let mut lut = vec![0.0f32; DT_IOP_COLORZONES_LUT_RES];

    if periodic {
        for i in 0..sampling_res {
            let s = eval_knots(&knots, i as f32 * res, x_lim, y_lim, true);
            let val = (s.clamp(0.0, 1.0) * (output_res - 1.0)).round() as i32;
            lut[i] = val.clamp(0, 65535) as u16 as f32 / output_res;
        }
    } else {
        let first_point_x = (knots[0].x * (sampling_res as f32 - 1.0)) as i32;
        let first_point_y = (knots[0].y * (output_res - 1.0)) as i32;
        let last_point_x = (knots[n - 1].x * (sampling_res as f32 - 1.0)) as i32;
        let last_point_y = (knots[n - 1].y * (output_res - 1.0)) as i32;
        let max_y = (output_res - 1.0) as i32;
        let min_y = 0i32;

        for i in 0..sampling_res {
            let i_i32 = i as i32;
            let val = if i_i32 < first_point_x {
                first_point_y
            } else if i_i32 > last_point_x {
                last_point_y
            } else {
                let s = eval_knots(&knots, i as f32 * res, x_lim, y_lim, false);
                let mut v = (s.clamp(0.0, 1.0) * (output_res - 1.0)).round() as i32;
                if v > max_y { v = max_y; }
                if v < min_y { v = min_y; }
                v
            };
            lut[i] = val.clamp(0, 65535) as u16 as f32 / output_res;
        }
    }

    lut
}
const DT_2PI: f32 = std::f32::consts::TAU;

/// Linear interpolation in a 65536-entry LUT, index in [0,1].
#[inline(always)]
fn lut_lookup(lut: &[f32], i: f32) -> f32 {
    let bin0 = ((DT_IOP_COLORZONES_LUT_RES as f32 * i) as usize).clamp(0, 0xffff);
    let bin1 = (bin0 + 1).min(0xffff);
    let f = DT_IOP_COLORZONES_LUT_RES as f32 * i - bin0 as f32;
    lut[bin1] * f + lut[bin0] * (1.0 - f)
}

/// Lab → LCH: L unchanged, C = hypot(a,b), h = atan2(b,a)/(2π) in [0,1).
#[inline(always)]
fn lab_to_lch(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let var_h = b.atan2(a);
    let h = if var_h > 0.0 {
        var_h / DT_2PI
    } else {
        1.0 - var_h.abs() / DT_2PI
    };
    (l, a.hypot(b), h)
}

/// LCH → Lab: L unchanged, a = C*cos(h*2π), b = C*sin(h*2π).
#[inline(always)]
fn lch_to_lab(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    let (sin_h, cos_h) = (DT_2PI * h).sin_cos();
    (l, cos_h * c, sin_h * c)
}

/// Color-zones IOP — luminance/chroma/hue equalizer in LCH space.
///
/// mode: 0 = v3 smooth (DT_IOP_COLORZONES_MODE_SMOOTH), non-zero = v1 legacy
/// channel: 0 = L, 1 = C, 2 = h (which dimension drives selection)
/// lut_l/a/b: each 65536 floats (d->lut[0..2]).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorzones_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    mode: i32,    // 0 = smooth/v3 (DT_IOP_COLORZONES_MODE_SMOOTH), non-zero = flat/v1
    channel: i32, // 0=L, 1=C, 2=h
    lut_l: *const f32,
    lut_a: *const f32,
    lut_b: *const f32,
) {
    const NORMALIZE_C: f32 = 1.0 / (128.0 * std::f32::consts::SQRT_2);

    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let ll = std::slice::from_raw_parts(lut_l, DT_IOP_COLORZONES_LUT_RES);
    let la = std::slice::from_raw_parts(lut_a, DT_IOP_COLORZONES_LUT_RES);
    let lb = std::slice::from_raw_parts(lut_b, DT_IOP_COLORZONES_LUT_RES);

    for px in 0..npixels {
        let base = px * 4;
        let i = &inp[base..base + 4];
        let o = &mut out[base..base + 4];
        let (in_l, in_a, in_b) = (i[0], i[1], i[2]);

        if mode != 0 {
            // v1: legacy flat mode (DT_IOP_COLORZONES_MODE_FLAT)
            let (l, c, h) = lab_to_lch(in_l, in_a, in_b);
            let select = (match channel {
                0 => l * 0.01,
                1 => c * NORMALIZE_C,
                _ => h,
            }).clamp(0.0, 1.0);

            let out_l = l * 2.0f32.powf(4.0 * (lut_lookup(ll, select) - 0.5));
            let out_c = c * 2.0 * lut_lookup(la, select);
            let out_h = h + lut_lookup(lb, select) - 0.5;
            let (rl, ra, rb) = lch_to_lab(out_l, out_c, out_h);
            o[0] = rl;
            o[1] = ra;
            o[2] = rb;
        } else {
            // v3: smooth mode (DT_IOP_COLORZONES_MODE_SMOOTH = 0) — edit in a/b space directly
            let a = in_a;
            let b = in_b;
            let h = (b.atan2(a) + DT_2PI).rem_euclid(DT_2PI) / DT_2PI;
            let c = (b * b + a * a).sqrt();
            let (select, blend) = match channel {
                0 => (((in_l / 100.0).min(1.0)), 0.0f32),
                1 => ((c / 128.0).min(1.0), 0.0f32),
                _ => (h, (1.0 - c / 128.0) * (1.0 - c / 128.0)),
            };
            let lm = (blend * 0.5 + (1.0 - blend) * lut_lookup(ll, select)) - 0.5;
            let hm = (blend * 0.5 + (1.0 - blend) * lut_lookup(lb, select)) - 0.5;
            let cm = 2.0 * lut_lookup(la, select);
            let out_l = in_l * 2.0f32.powf(4.0 * lm);
            o[0] = out_l;
            let new_h = h + hm;
            let (sin_h, cos_h) = (DT_2PI * new_h).sin_cos();
            o[1] = cos_h * cm * c;
            o[2] = sin_h * cm * c;
        }
        o[3] = i[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half_lut() -> Vec<f32> {
        vec![0.5f32; DT_IOP_COLORZONES_LUT_RES]
    }

    #[test]
    fn v1_flat_lut_zeroes_lch_offsets() {
        // lut_a = 0.5 → C *= 2*0.5 = 1 (no change)
        // lut_l = 0.5 → L *= 2^(4*(0.5-0.5)) = 1 (no change)
        // lut_b = 0.5 → h += 0.5 - 0.5 = 0 (no change)
        let half = half_lut();
        let inp = [50.0f32, 30.0, 10.0, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_colorzones_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1, 2, // v1 (non-zero = legacy), h channel
                half.as_ptr(), half.as_ptr(), half.as_ptr(),
            )
        };
        assert!((out[0] - 50.0).abs() < 0.5);
        assert!((out[1] - 30.0).abs() < 0.5);
        assert!((out[2] - 10.0).abs() < 0.5);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn v3_flat_lut_neutral() {
        // lut_l = 0.5 → Lm = 0 → L *= 2^0 = 1
        // lut_a = 0.5 → Cm = 1 → chroma scaled by 1
        // lut_b = 0.5 → hm = 0 → hue unchanged
        let half = half_lut();
        let inp = [60.0f32, 20.0, 10.0, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_colorzones_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                0, 2, // v3/smooth (mode=0), h channel
                half.as_ptr(), half.as_ptr(), half.as_ptr(),
            )
        };
        assert!((out[0] - 60.0).abs() < 0.5);
        // a/b may shift slightly due to float precision but should be close
        let c_in  = (inp[1]*inp[1] + inp[2]*inp[2]).sqrt();
        let c_out = (out[1]*out[1] + out[2]*out[2]).sqrt();
        assert!((c_out - c_in).abs() < 0.5);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn alpha_passes_through() {
        let half = half_lut();
        let inp = [50.0f32, 0.0, 0.0, 0.42];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_colorzones_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                1, 0, // v1 (non-zero = legacy), L channel
                half.as_ptr(), half.as_ptr(), half.as_ptr(),
            )
        };
        assert_eq!(out[3], 0.42);
    }
}

#[inline(always)]
fn lab_2_lch(lab: &[f32]) -> [f32; 3] {
    let h_raw = lab[2].atan2(lab[1]);
    let h = if h_raw > 0.0 {
        h_raw / std::f32::consts::TAU
    } else {
        1.0 - h_raw.abs() / std::f32::consts::TAU
    };
    [lab[0], lab[1].hypot(lab[2]), h]
}

/// Mask-display pass for colorzones process_display().
///
/// For each pixel: Lab → LCh, select L/C/h, look up the display-channel LUT,
/// write mask alpha into out[3]. Input and output are RGBA (4 floats/pixel).
/// The caller copies ivoid → ovoid first; this function only writes alpha.
///
/// `channel`: 0=L, 1=C, 2=h (DT_IOP_COLORZONES_{L,C,h} enum values).
/// `lut`: pointer to `d->lut[display_channel]` — exactly 65536 floats.
///
/// Matches the DT_OMP_FOR in src/iop/colorzones.c:444.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorzones_display(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    channel: i32,
    lut: *const f32,
) {
    if npixels == 0 { return; }
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lut_s = std::slice::from_raw_parts(lut, DT_IOP_COLORZONES_LUT_RES);
    const NORM_C: f32 = 1.0 / (128.0 * std::f32::consts::SQRT_2);

    for k in 0..npixels {
        let px = &input[k * 4..k * 4 + 4];
        let lch = lab_2_lch(px);
        let select = match channel {
            0 => lch[0] * 0.01,
            1 => lch[1] * NORM_C,
            _ => lch[2],
        };
        let select = select.clamp(0.0, 1.0);
        let alpha = (lut_lookup(lut_s, select) - 0.5).abs() * 4.0;
        output[k * 4 + 3] = alpha.clamp(0.0, 1.0);
    }
}
