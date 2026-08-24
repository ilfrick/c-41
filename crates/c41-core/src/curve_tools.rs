//! Port of darktable's V1 curve sampler — `src/common/curve_tools.c` (the
//! UFraw nikon-curve lineage) plus the composition performed by the
//! `dt_draw_curve_*` inline wrappers in `src/gui/draw.h`. This is the machinery
//! behind tonecurve's LUT construction (`dt_draw_curve_new` /
//! `dt_draw_curve_add_point` / `dt_draw_curve_calc_values`).
//!
//! [`crate::splines`] implements the *V2* sampler (`splines.cpp`) used by
//! rgbcurve/colorzones/etc.; its numerics differ from V1 deliberately (natural
//! cubic spline with second-derivative-0 boundary conditions here vs V2's
//! different end conditions; Fritsch-Carlson monotone tangents with the
//! EPSILON zeroing quirk here vs V2's variant). The two live side by side on
//! purpose — each module must reproduce its own C code path.
//!
//! Deviations from the C, all numerics-neutral:
//! * allocation failures are impossible in Rust, so the NULL-return error
//!   paths become early exits that fall back to a straight line between the
//!   box corners (the C leaves the sample buffer holding whatever it held
//!   before; unreachable for well-formed node sets);
//! * the C vectorises nothing here, so there is nothing else to flatten.

/// Spline types (curve_tools.h), matching `m_spline_type` dispatch.
pub const CUBIC_SPLINE: u32 = 0;
pub const CATMULL_ROM: u32 = 1;
pub const MONOTONE_HERMITE: u32 = 2;

/// MAX_RESOLUTION (curve_tools.h): sampling/output resolution ceiling.
pub const MAX_RESOLUTION: usize = 65536;
/// MAX_ANCHORS (curve_tools.h).
pub const MAX_ANCHORS: usize = 20;

/// `EPSILON` from curve_tools.c:29 — 2·FLT_MIN.
const EPSILON: f32 = 2.0 * f32::MIN_POSITIVE;

// ── d3_np_fs — Thomas-algorithm tridiagonal solver (curve_tools.c:95) ────────

/// Solves the tridiagonal system built by [`spline_cubic_set_internal`] in the
/// same collapsed layout the C keeps it in. Returns `None` where the C's
/// `d3_np_fs` returns NULL (zero diagonal, bad n).
fn d3_np_fs(n: usize, a: &mut [f32], b: &[f32]) -> Option<Vec<f32>> {
    if n == 0 || n > MAX_ANCHORS {
        return None;
    }
    for i in 0..n {
        if a[1 + i * 3] == 0.0 {
            return None;
        }
    }
    let mut x = b.to_vec();
    // forward elimination
    for i in 1..n {
        let xmult = a[2 + (i - 1) * 3] / a[1 + (i - 1) * 3];
        a[1 + i * 3] -= xmult * a[0 + i * 3];
        x[i] -= xmult * x[i - 1];
    }
    // back substitution
    x[n - 1] /= a[1 + (n - 1) * 3];
    for i in (0..(n - 1)).rev() {
        x[i] = (x[i] - a[0 + (i + 1) * 3] * x[i + 1]) / a[1 + i * 3];
    }
    Some(x)
}

// ── Natural cubic spline (spline_cubic_set_internal, curve_tools.c:245) ──────

/// Second derivatives of a cubic spline through `(t[], y[])` with Burkardt's
/// boundary-condition flags; callers use the natural-spline wrapper
/// `(ibcbeg, ybcbeg, ibcend, ybcend) = (2, 0, 2, 0)` — zero second derivative
/// at both ends. Mirrors the C including the calloc-zeroed coefficient array,
/// so entries upstream never writes behave exactly as upstream.
#[allow(clippy::too_many_arguments)]
fn spline_cubic_set_internal(
    n: usize,
    t: &[f32],
    y: &[f32],
    ibcbeg: u32,
    ybcbeg: f32,
    ibcend: u32,
    ybcend: f32,
) -> Option<Vec<f32>> {
    if n <= 1 {
        return None;
    }
    for i in 0..n - 1 {
        if t[i + 1] <= t[i] {
            return None;
        }
    }
    // calloc(3*n): entries upstream never writes stay 0 exactly as in C.
    let mut a = vec![0.0f32; 3 * n];
    let mut b = vec![0.0f32; n];

    match ibcbeg {
        0 => {
            b[0] = 0.0;
            a[1] = 1.0;
            a[3] = -1.0;
        }
        1 => {
            b[0] = (y[1] - y[0]) / (t[1] - t[0]) - ybcbeg;
            a[1] = (t[1] - t[0]) / 3.0;
            a[3] = (t[1] - t[0]) / 6.0;
        }
        2 => {
            b[0] = ybcbeg;
            a[1] = 1.0;
            a[3] = 0.0;
        }
        _ => return None,
    }
    for i in 1..n - 1 {
        b[i] = (y[i + 1] - y[i]) / (t[i + 1] - t[i])
            - (y[i] - y[i - 1]) / (t[i] - t[i - 1]);
        a[2 + (i - 1) * 3] = (t[i] - t[i - 1]) / 6.0;
        a[1 + i * 3] = (t[i + 1] - t[i - 1]) / 3.0;
        a[0 + (i + 1) * 3] = (t[i + 1] - t[i]) / 6.0;
    }
    match ibcend {
        0 => {
            b[n - 1] = 0.0;
            a[2 + (n - 2) * 3] = -1.0;
            a[1 + (n - 1) * 3] = 1.0;
        }
        1 => {
            b[n - 1] = ybcend - (y[n - 1] - y[n - 2]) / (t[n - 1] - t[n - 2]);
            a[2 + (n - 2) * 3] = (t[n - 1] - t[n - 2]) / 6.0;
            // curve_tools.c:324: the diagonal carries Δt/3, not 1.
            a[1 + (n - 1) * 3] = (t[n - 1] - t[n - 2]) / 3.0;
        }
        2 => {
            b[n - 1] = ybcend;
            a[2 + (n - 2) * 3] = 0.0;
            a[1 + (n - 1) * 3] = 1.0;
        }
        _ => return None,
    }

    if n == 2 && ibcbeg == 0 && ibcend == 0 {
        return Some(vec![0.0, 0.0]);
    }
    d3_np_fs(n, &mut a, &b)
}

/// `spline_cubic_set` (curve_tools.c:376) — the natural-spline wrapper used by
/// the V1 dispatch table.
fn spline_cubic_set(n: usize, t: &[f32], y: &[f32]) -> Option<Vec<f32>> {
    spline_cubic_set_internal(n, t, y, 2, 0.0, 2, 0.0)
}

/// Evaluate the piecewise cubic at `tval`, extrapolating outside
/// `[t[0], t[n-1]]` (spline_cubic_val, curve_tools.c:616).
fn spline_cubic_val(n: usize, t: &[f32], tval: f32, y: &[f32], ypp: &[f32]) -> f32 {
    let mut ival = n - 2;
    for i in 0..n - 1 {
        if tval < t[i + 1] {
            ival = i;
            break;
        }
    }
    let dt = tval - t[ival];
    let h = t[ival + 1] - t[ival];
    y[ival]
        + dt * ((y[ival + 1] - y[ival]) / h
            - (ypp[ival + 1] / 6.0 + ypp[ival] / 3.0) * h
            + dt * (0.5 * ypp[ival] + dt * ((ypp[ival + 1] - ypp[ival]) / (6.0 * h))))
}

// ── Catmull-Rom tangents + Hermite-basis evaluation ───────────────────────────

/// Central-difference tangents (catmull_rom_set, curve_tools.c:467).
fn catmull_rom_set(n: usize, x: &[f32], y: &[f32]) -> Option<Vec<f32>> {
    if n <= 1 {
        return None;
    }
    for i in 0..n - 1 {
        if x[i + 1] <= x[i] {
            return None;
        }
    }
    let mut m = vec![0.0f32; n];
    m[0] = (y[1] - y[0]) / (x[1] - x[0]);
    for i in 1..n - 1 {
        m[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }
    m[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    Some(m)
}

/// Piecewise Hermite-basis evaluation (catmull_rom_val, curve_tools.c:524).
/// Also serves MONOTONE_HERMITE (see `interpolate_val`'s dispatch table).
fn catmull_rom_val(n: usize, x: &[f32], xval: f32, y: &[f32], tangents: &[f32]) -> f32 {
    let mut ival = n - 2;
    for i in 0..n - 2 {
        if xval < x[i + 1] {
            ival = i;
            break;
        }
    }
    let m0 = tangents[ival];
    let m1 = tangents[ival + 1];
    let h = x[ival + 1] - x[ival];
    let dx = (xval - x[ival]) / h;
    let dx2 = dx * dx;
    let dx3 = dx * dx2;
    let h00 = 2.0 * dx3 - 3.0 * dx2 + 1.0;
    let h10 = dx3 - 2.0 * dx2 + dx;
    let h01 = -2.0 * dx3 + 3.0 * dx2;
    let h11 = dx3 - dx2;
    h00 * y[ival] + h10 * h * m0 + h01 * y[ival + 1] + h11 * h * m1
}

// ── Monotone-hermite tangents (monotone_hermite_set, curve_tools.c:393) ──────

/// Fritsch-Carlson tangents exactly as curve_tools.c computes them, including
/// the replicated final delta and the tangents array being one longer than
/// evaluation needs (the clamp loop may write index `n`; evaluation never
/// reads past `n−1`).
fn monotone_hermite_set(n: usize, x: &[f32], y: &[f32]) -> Option<Vec<f32>> {
    if n <= 1 {
        return None;
    }
    for i in 0..n - 1 {
        if x[i + 1] <= x[i] {
            return None;
        }
    }
    let mut delta = vec![0.0f32; n];
    let mut m = vec![0.0f32; n + 1];
    for i in 0..n - 1 {
        delta[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }
    delta[n - 1] = delta[n - 2];

    m[0] = delta[0];
    m[n - 1] = delta[n - 1];
    for i in 1..n - 1 {
        m[i] = (delta[i - 1] + delta[i]) * 0.5;
    }
    for i in 0..n {
        if delta[i].abs() < EPSILON {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let alpha = m[i] / delta[i];
            let beta = m[i + 1] / delta[i];
            let tau = alpha * alpha + beta * beta;
            if tau > 9.0 {
                m[i] = 3.0 * alpha * delta[i] / tau.sqrt();
                m[i + 1] = 3.0 * beta * delta[i] / tau.sqrt();
            }
        }
    }
    Some(m)
}

// ── Dispatch tables (curve_tools.c:41-44) ─────────────────────────────────────

/// `interpolate_set`: per-type tangent/second-derivative computation.
/// Note the C's table: `{ spline_cubic_set, catmull_rom_set,
/// monotone_hermite_set }` — MONOTONE_HERMITE gets its own tangents but
/// evaluates through `catmull_rom_val`.
fn interpolate_set(n: usize, x: &[f32], y: &[f32], type_: u32) -> Option<Vec<f32>> {
    match type_ {
        CUBIC_SPLINE => spline_cubic_set(n, x, y),
        CATMULL_ROM | MONOTONE_HERMITE => {
            if type_ == CATMULL_ROM { catmull_rom_set(n, x, y) } else { monotone_hermite_set(n, x, y) }
        }
        _ => None,
    }
}

/// `interpolate_val` dispatch. MONOTONE_HERMITE shares `catmull_rom_val`
/// (curve_tools.c:42: `spline_val[] = { spline_cubic_val, catmull_rom_val,
/// catmull_rom_val }`).
fn interpolate_val(n: usize, x: &[f32], xval: f32, y: &[f32], tangents: &[f32], type_: u32) -> f32 {
    match type_ {
        CUBIC_SPLINE => spline_cubic_val(n, x, xval, y, tangents),
        _ => catmull_rom_val(n, x, xval, y, tangents),
    }
}

// ── CurveDataSample + dt_draw composition ─────────────────────────────────────

/// Port of `CurveDataSample` (curve_tools.c:664) followed by the float
/// conversion in `dt_draw_curve_smaple_values` (draw.h:389): fills `out[k] =
/// min_y + (max_y − min_y) · samples[k] · (1/0x10000)` where samples are the
/// u16 quantised curve values.
///
/// `anchors` are box-relative coordinates (tonecurve's box is [0,1], so they
/// pass through unchanged); `min_y`/`max_y` are the output range both
/// dt_draw_curve_new and dt_draw_curve_calc_values receive (tonecurve:
/// 0..1). Sampling/output resolution follow draw.h's 0x10000 setup.
///
/// Error paths (NULL tangent sets, anchors > MAX_ANCHORS) fall back to the
/// straight box diagonal rather than the C's uninitialised buffer.
pub fn curve_data_sample(
    anchors: &[(f32, f32)],
    spline_type: u32,
    min_y: f32,
    max_y: f32,
    out: &mut [f32],
) {
    let res_samples = out.len().min(MAX_RESOLUTION);
    if res_samples < 2 || anchors.len() > MAX_ANCHORS {
        return;
    }

    // Box transform (box is [0,1] for every darktable caller): x·w+min etc.
    let (min_x, max_x) = (0.0f32, 1.0f32);
    let box_w = max_x - min_x;
    let box_h = max_y - min_y;

    // Build arrays; zero anchors → straight line over the box corners.
    let (x, y): (Vec<f32>, Vec<f32>) = if anchors.is_empty() {
        (
            vec![min_x, max_x],
            vec![min_y, max_y],
        )
    } else {
        (
            anchors.iter().map(|&(ax, _)| ax * box_w + min_x).collect(),
            anchors.iter().map(|&(_, ay)| ay * box_h + min_y).collect(),
        )
    };
    let n = x.len();

    let sampling_res = res_samples;
    let output_res = MAX_RESOLUTION; // draw.h sets m_outputRes = 0x10000 always

    let res = 1.0f32 / (sampling_res - 1) as f32;
    let first_point_x = (x[0] * (sampling_res - 1) as f32) as i32;
    let first_point_y = (y[0] * (output_res - 1) as f32) as i32;
    let last_point_x = (x[n - 1] * (sampling_res - 1) as f32) as i32;
    let last_point_y = (y[n - 1] * (output_res - 1) as f32) as i32;
    let max_y_i = (max_y * (output_res - 1) as f32) as i32;
    let pre_clamp_min = (min_y * (output_res - 1) as f32) as i32;

    // interpolate_set returning NULL → straight line fallback (deviation, see
    // module doc).
    // interpolate_set returning NULL (degenerate node set: n ≤ 1 after the
    // straight-line expansion can't happen, but non-increasing anchors can)
    // → straight box-diagonal fallback instead of the C's stale buffer.
    let tangents = match interpolate_set(n, &x, &y, spline_type) {
        Some(t) => t,
        None => {
            for (k, v) in out[..res_samples].iter_mut().enumerate() {
                *v = min_y + (max_y - min_y) * (k as f32 / (res_samples - 1) as f32);
            }
            return;
        }
    };

    for (k, slot) in out[..res_samples].iter_mut().enumerate() {
        let k = k as i32;
        let sample: i32 = if k < first_point_x {
            first_point_y
        } else if k > last_point_x {
            last_point_y
        } else {
            // int truncation of val·(outputRes−1)+0.5, clamped to [minY,maxY]
            let val = (interpolate_val(n, &x, k as f32 * res, &y, &tangents, spline_type)
                * (output_res - 1) as f32
                + 0.5) as i32;
            val.clamp(pre_clamp_min, max_y_i)
        };
        *slot = min_y + (max_y - min_y) * sample as f32 * (1.0 / MAX_RESOLUTION as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-solved tridiagonal system:
    /// ```text
    /// 2·x0 + 1·x1        = 4
    /// 1·x0 + 3·x1 + 1·x2 = 5
    ///         1·x1 + 2·x2 = 6
    /// ```
    /// has solution (2, 0, 3). The D3 slots follow how
    /// [`spline_cubic_set_internal`] fills them: `a[3i]` carries row i−1's
    /// super coefficient, `a[3i+1]` row i's diagonal, `a[3i+2]` row i+1's
    /// sub coefficient.
    #[test]
    fn d3_np_fs_solves_known_system() {
        let mut a = [
            0.0f32, 2.0, 1.0, // row 0: diag=2, sub₁=1 (super₋₁ slot unused)
            1.0f32, 3.0, 1.0, // row 1: super₀=1, diag=3, sub₂=1
            1.0f32, 2.0, 0.0, // row 2: super₁=1, diag=2 (sub₃ slot unused)
        ];
        let b = [4.0f32, 5.0, 6.0];
        let x = d3_np_fs(3, &mut a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-5, "{}", x[0]);
        assert!((x[1] - 0.0).abs() < 1e-5, "{}", x[1]);
        assert!((x[2] - 3.0).abs() < 1e-5, "{}", x[2]);
    }

    #[test]
    fn d3_np_fs_rejects_zero_diagonal() {
        let mut a = [0.0f32; 9];
        a[1] = 0.0; // zero diagonal on row 0
        assert!(d3_np_fs(3, &mut a, &[1.0; 3]).is_none());
    }

    /// Natural cubic spline through (0,0),(1,1),(2,0): the hand-solved second
    /// derivatives are ypp = [0, −3, 0] (natural ends force ypp₀=ypp₂=0 and
    /// the interior equation gives ⅔·ypp₁ = −2). Evaluation at the knots must
    /// interpolate exactly; at x=1.5 the Burkardt form gives A+Bt+Ct²+Dt³ =
    /// 1 + 0 − 1.5·0.25 + 0.5·0.125 = 0.6875.
    #[test]
    fn natural_cubic_spline_hand_solved() {
        let t = [0.0f32, 1.0, 2.0];
        let y = [0.0f32, 1.0, 0.0];
        let ypp = spline_cubic_set(3, &t, &y).unwrap();
        assert!((ypp[0]).abs() < 1e-6);
        assert!((ypp[1] - (-3.0)).abs() < 1e-5, "ypp1={}", ypp[1]);
        assert!((ypp[2]).abs() < 1e-6);

        for (tv, want) in [(0.0f32, 0.0), (1.0, 1.0), (2.0, 0.0)] {
            let v = spline_cubic_val(3, &t, tv, &y, &ypp);
            assert!((v - want).abs() < 1e-5, "val({tv})={v}");
        }
        let mid = spline_cubic_val(3, &t, 1.5, &y, &ypp);
        assert!((mid - 0.6875).abs() < 1e-5, "val(1.5)={mid}");
    }

    /// n == 2 with natural boundaries is special-cased in the C to ypp = 0
    /// (the spline degenerates to the straight segment).
    #[test]
    fn cubic_two_points_is_linear() {
        let t = [0.0f32, 1.0];
        let y = [0.25f32, 0.75];
        let ypp = spline_cubic_set(2, &t, &y).unwrap();
        assert_eq!(ypp, vec![0.0, 0.0]);
        let v = spline_cubic_val(2, &t, 0.5, &y, &ypp);
        assert!((v - 0.5).abs() < 1e-6);
    }

    /// Catmull-Rom tangents are central differences in the interior.
    #[test]
    fn catmull_rom_central_difference_tangents() {
        let x = [0.0f32, 1.0, 2.0, 3.0];
        let y = [0.0f32, 2.0, 4.0, 9.0];
        let m = catmull_rom_set(4, &x, &y).unwrap();
        assert!((m[0] - 2.0).abs() < 1e-6); // forward diff
        assert!((m[1] - 2.0).abs() < 1e-6); // (4−0)/(2−0)
        assert!((m[2] - 3.5).abs() < 1e-6); // (9−2)/(3−1)
        assert!((m[3] - 5.0).abs() < 1e-6); // backward diff
    }

    #[test]
    fn catmull_rom_interpolates_knots() {
        let x = [0.0f32, 0.5, 1.0];
        let y = [0.1f32, 0.7, 0.2];
        let m = catmull_rom_set(3, &x, &y).unwrap();
        for k in 0..3 {
            let v = catmull_rom_val(3, &x, x[k], &y, &m);
            assert!((v - y[k]).abs() < 1e-5, "knot {k}: {v}");
        }
    }

    /// Monotone data through MONOTONE_HERMITE yields a monotone LUT — the
    /// whole point of the Fritsch-Carlson clamp. The contrast case is asserted
    /// at evaluation level (CurveDataSample clamps to the box, so a raw LUT can
    /// never show the overshoot).
    #[test]
    fn monotone_hermite_preserves_monotonicity() {
        // Steep-then-flat data that overshoots under plain Hermite evaluation.
        let anchors = [(0.0f32, 0.0), (0.2, 1.0), (0.25, 1.0), (1.0, 1.0)];
        let (x, y): (Vec<f32>, Vec<f32>) =
            (anchors.iter().map(|a| a.0).collect(), anchors.iter().map(|a| a.1).collect());
        let cr = catmull_rom_set(4, &x, &y).unwrap();
        let mh = monotone_hermite_set(4, &x, &y).unwrap();
        let mut cr_overshot = false;
        for s in 0..=1000u32 {
            let xv = s as f32 / 1000.0;
            let v_cr = catmull_rom_val(4, &x, xv, &y, &cr);
            let v_mh = catmull_rom_val(4, &x, xv, &y, &mh);
            if v_cr > 1.0 + 1e-3 {
                cr_overshot = true;
            }
            assert!(v_mh <= 1.0 + 1e-6, "monotone hermite overshot: {v_mh} @ {xv}");
            assert!(v_mh >= -1e-6);
        }
        assert!(cr_overshot, "catmull-rom expected to overshoot this data");

        // And through the full sampler the MH LUT is non-decreasing.
        let mut out = vec![0.0f32; MAX_RESOLUTION];
        curve_data_sample(&anchors, MONOTONE_HERMITE, 0.0, 1.0, &mut out);
        for w in out.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "{} → {}", w[0], w[1]);
        }
    }

    /// Identity anchors through every sampler produce the ramp within u16
    /// quantisation (±1 LSB of 1/65536 after the ÷0x10000 conversion).
    #[test]
    fn identity_anchors_produce_ramp() {
        for ty in [CUBIC_SPLINE, CATMULL_ROM, MONOTONE_HERMITE] {
            let mut out = vec![0.0f32; MAX_RESOLUTION];
            curve_data_sample(&[(0.0, 0.0), (1.0, 1.0)], ty, 0.0, 1.0, &mut out);
            for (k, &v) in out.iter().enumerate() {
                let want = k as f32 / MAX_RESOLUTION as f32;
                assert!(
                    (v - want).abs() < 2.0 / MAX_RESOLUTION as f32,
                    "{ty} @ {k}: {v} vs {want}"
                );
            }
        }
    }

    /// Zero anchors expand to the straight box line (CurveDataSample's n==2
    /// path with the box corners).
    #[test]
    fn empty_anchors_expand_to_diagonal() {
        let mut out = vec![0.0f32; 256];
        curve_data_sample(&[], CUBIC_SPLINE, 0.0, 1.0, &mut out);
        for (k, &v) in out.iter().enumerate() {
            assert!((v - k as f32 / 255.0).abs() < 1e-3, "@{k}: {v}");
        }
    }

    /// Anchors not starting at x=0 clamp the leading samples flat at the
    /// first anchor's height (firstPointX/firstPointY logic).
    #[test]
    fn leading_gap_clamps_to_first_anchor_height() {
        let mut out = vec![0.0f32; MAX_RESOLUTION];
        curve_data_sample(&[(0.25, 0.1), (1.0, 0.9)], CUBIC_SPLINE, 0.0, 1.0, &mut out);
        let fp_x = (0.25f32 * (MAX_RESOLUTION - 1) as f32) as i32;
        // Well before firstPointX everything equals firstPointY ≈ 0.1·65535/65536.
        assert!((out[10] - 0.1).abs() < 1e-3, "{}", out[10]);
        assert!((out[(fp_x - 5) as usize] - 0.1).abs() < 1e-3);
        // At the far end we reach the last anchor's height.
        assert!((out[MAX_RESOLUTION - 1] - 0.9).abs() < 1e-3, "{}", out[MAX_RESOLUTION - 1]);
    }

    /// Non-monotone anchors make the tangent sets NULL in the C — our fallback
    /// is the box diagonal rather than upstream's stale buffer.
    #[test]
    fn non_increasing_anchors_fall_back_to_diagonal() {
        let mut out = vec![0.0f32; 64];
        curve_data_sample(&[(0.5, 0.0), (0.2, 1.0)], MONOTONE_HERMITE, 0.0, 1.0, &mut out);
        for (k, &v) in out.iter().enumerate() {
            assert!((v - k as f32 / 63.0).abs() < 1e-4, "@{k}: {v}");
        }
    }

    // NOTE: there is deliberately no test for min_y/max_y ranges other than
    // 0..1 — with such ranges the C composition double-applies the offset
    // (anchors are scaled into [min,max], then smaple_values scales the u16
    // samples back up by (max−min)) and wraps its uint16 sample buffer. Every
    // darktable call site passes 0..1 to both dt_draw_curve_new and
    // dt_draw_curve_calc_values, so tonecurve does too; other values here
    // reproduce upstream's arithmetic including that breakage.
}
