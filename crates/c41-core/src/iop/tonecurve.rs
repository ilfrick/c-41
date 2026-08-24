//! Tone-curve IOP — 3-channel Lab LUT with four autoscale modes.
//!
//! Replaces the OMP loop in src/iop/tonecurve.c::process().
//!
//! autoscale_ab modes:
//!   0 = DT_S_SCALE_MANUAL   — independent L/a/b LUT lookup with optional unbounded extrapolation
//!   1 = DT_S_SCALE_AUTOMATIC — scale a/b proportionally to L curve output
//!   2 = DT_S_SCALE_AUTOMATIC_XYZ — apply L curve to XYZ channels
//!   3 = DT_S_SCALE_AUTOMATIC_RGB — apply L curve to ProPhoto RGB (with preserve_colors)
//!
//! table_l/a/b  : 65536 floats each (d->table[ch_L/a/b])
//! unbounded_coeffs_l  : 3 floats (d->unbounded_coeffs_L)
//! unbounded_coeffs_ab : 12 floats (d->unbounded_coeffs_ab; 4 groups of 3 for a-right, a-left, b-right, b-left)

use crate::{
    color::{
        eval_exp, lab_to_prophotorgb, lab_to_xyz, prophotorgb_to_lab, rgb_norm, xyz_to_lab,
    },
    curve_tools,
    iop::{colisa::estimate_exp, ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

pub struct ToneCurve;

// ── LUT construction (tonecurve.c commit_params :625) ────────────────────────

/// autoscale_ab modes (dt_iop_tonecurve_autoscale_t).
pub const DT_S_SCALE_MANUAL: i32 = 0;
pub const DT_S_SCALE_AUTOMATIC: i32 = 1;
pub const DT_S_SCALE_AUTOMATIC_XYZ: i32 = 2;
pub const DT_S_SCALE_AUTOMATIC_RGB: i32 = 3;

/// Everything [`process_pixels`] needs, produced by [`build_lut`].
pub struct ToneCurveTables {
    pub table_l: Box<[f32; 65536]>,
    pub table_a: Box<[f32; 65536]>,
    pub table_b: Box<[f32; 65536]>,
    pub unbounded_coeffs_l: [f32; 3],
    pub unbounded_coeffs_ab: [f32; 12],
}

/// Port of tonecurve.c `commit_params`: build the three channel tables from
/// the node sets via the V1 sampler (`curve_tools::curve_data_sample`, the
/// dt_draw_curve_* machinery), rescale them to Lab units (L×100, a/b
/// ×256−128), re-derive the L table through the XYZ/RGB round-trip for the
/// linked-channels modes, and fit the exponential tail extrapolations.
///
/// Node anchors are box-relative [0,1] coordinates exactly as stored in the C
/// params struct. `autoscale_ab` is one of the DT_S_SCALE_* constants.
#[allow(clippy::too_many_arguments)]
pub fn build_lut(
    nodes_l: &[(f32, f32)],
    type_l: u32,
    nodes_a: &[(f32, f32)],
    type_a: u32,
    nodes_b: &[(f32, f32)],
    type_b: u32,
    autoscale_ab: i32,
) -> ToneCurveTables {
    let mut raw = vec![0.0f32; 0x1_0000];
    let mut t_l = Box::new([0.0f32; 65536]);
    let mut t_a = Box::new([0.0f32; 65536]);
    let mut t_b = Box::new([0.0f32; 65536]);

    // Per-channel: sample the curve at 0x10000 resolution into [0,1], then
    // rescale to the channel's Lab units (:659-666).
    curve_tools::curve_data_sample(nodes_l, type_l, 0.0, 1.0, &mut raw);
    for k in 0..0x1_0000 {
        t_l[k] = raw[k] * 100.0;
    }
    curve_tools::curve_data_sample(nodes_a, type_a, 0.0, 1.0, &mut raw);
    for k in 0..0x1_0000 {
        t_a[k] = raw[k] * 256.0 - 128.0;
    }
    curve_tools::curve_data_sample(nodes_b, type_b, 0.0, 1.0, &mut raw);
    for k in 0..0x1_0000 {
        t_b[k] = raw[k] * 256.0 - 128.0;
    }

    if autoscale_ab == DT_S_SCALE_AUTOMATIC_XYZ {
        // Derive the Y→Y mapping (:669-681): gray XYZ in, curved XYZ out.
        // Reads and writes share t_l in-place, iteration order preserved.
        for k in 0..0x1_0000 {
            let g = k as f32 / 0x1_0000 as f32;
            let lab = xyz_to_lab([g, g, g, 1.0]);
            let idx = ((lab[0] / 100.0 * 0x1_0000 as f32) as i64).clamp(0, 0xffff) as usize;
            let mut lab2 = lab;
            lab2[0] = t_l[idx];
            let xyz = lab_to_xyz(lab2);
            t_l[k] = xyz[1];
        }
    } else if autoscale_ab == DT_S_SCALE_AUTOMATIC_RGB {
        // Derive the G→G mapping (:682-694) through ProPhoto.
        for k in 0..0x1_0000 {
            let g = k as f32 / 0x1_0000 as f32;
            let lab = prophotorgb_to_lab([g, g, g, 1.0]);
            let idx = ((lab[0] / 100.0 * 0x1_0000 as f32) as i64).clamp(0, 0xffff) as usize;
            let mut lab2 = lab;
            lab2[0] = t_l[idx];
            let rgb = lab_to_prophotorgb(lab2);
            t_l[k] = rgb[1];
        }
    }

    // Exponential tails (dt_iop_estimate_exp over {0.7,0.8,0.9,1.0}·xm
    // samples of each table; ab mirrors x for the left side).
    let lut_index = |v: f32| -> usize { ((v * 0x1_0000_u32 as f32) as i64).clamp(0, 0xffff) as usize };

    let coeffs_l = {
        let xm = nodes_l[nodes_l.len() - 1].0;
        let xs: Vec<f32> = [0.7f32, 0.8, 0.9, 1.0].iter().map(|m| m * xm).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| t_l[lut_index(x)]).collect();
        estimate_exp(&xs, &ys)
    };

    // ab tails: right side of each channel, then the mirrored left side —
    // commit_params order a-right, a-left, b-right, b-left (:709-745).
    let mut coeffs_ab = [0.0f32; 12];
    for (slot, (table, edge_x, mirror)) in [
        (&t_a, nodes_a[nodes_a.len() - 1].0, false),
        (&t_a, nodes_a[0].0, true),
        (&t_b, nodes_b[nodes_b.len() - 1].0, false),
        (&t_b, nodes_b[0].0, true),
    ]
    .iter()
    .enumerate()
    {
        let xm = if *mirror { 1.0 - *edge_x } else { *edge_x };
        let xs: Vec<f32> = [0.7f32, 0.8, 0.9, 1.0].iter().map(|m| m * xm).collect();
        let ys: Vec<f32> = xs
            .iter()
            .map(|&x| table[lut_index(if *mirror { 1.0 - x } else { x })])
            .collect();
        coeffs_ab[slot * 3..slot * 3 + 3].copy_from_slice(&estimate_exp(&xs, &ys));
    }

    ToneCurveTables {
        table_l: t_l,
        table_a: t_a,
        table_b: t_b,
        unbounded_coeffs_l: coeffs_l,
        unbounded_coeffs_ab: coeffs_ab,
    }
}

impl IopProcess for ToneCurve {
    fn name(&self) -> &'static str {
        "tonecurve"
    }
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(Error::Pipeline("tonecurve: use the C FFI entry point (LUTs cannot be cast from raw params)".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("tonecurve: OpenCL path not yet ported".into()))
    }
}

// ── LUT eval helpers ──────────────────────────────────────────────────────────

#[inline(always)]
fn lut_clamp(x: f32) -> usize {
    ((x * 0x1_0000_u32 as f32) as i64).clamp(0, 0xffff) as usize
}

#[inline(always)]
fn table_lookup(table: &[f32; 65536], x: f32) -> f32 {
    table[lut_clamp(x)]
}

#[inline(always)]
fn eval_l(table: &[f32; 65536], coeffs: &[f32], xm: f32, x: f32) -> f32 {
    if x < xm {
        table_lookup(table, x)
    } else {
        eval_exp(coeffs, x)
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    table_l: &[f32; 65536],
    table_a: &[f32; 65536],
    table_b: &[f32; 65536],
    coeffs_l: &[f32],   // 3 floats
    coeffs_ab: &[f32],  // 12 floats (4 × 3)
    autoscale_ab: i32,
    unbound_ab: i32,
    preserve_colors: i32,
) {
    let xm_l = 1.0 / coeffs_l[0];
    let xm_ar = 1.0 / coeffs_ab[0];
    let xm_al = 1.0 - 1.0 / coeffs_ab[3];
    let xm_br = 1.0 / coeffs_ab[6];
    let xm_bl = 1.0 - 1.0 / coeffs_ab[9];
    let low_approx = table_l[((0.01 * 0x1_0000_u32 as f32) as usize).min(0xffff)];

    for (ci, co) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let l_in = ci[0] / 100.0;
        let l_out = eval_l(table_l, coeffs_l, xm_l, l_in);
        co[0] = l_out;

        match autoscale_ab {
            0 => {
                // DT_S_SCALE_MANUAL
                let a_in = (ci[1] + 128.0) / 256.0;
                let b_in = (ci[2] + 128.0) / 256.0;
                if unbound_ab == 0 {
                    co[1] = table_lookup(table_a, a_in);
                    co[2] = table_lookup(table_b, b_in);
                } else {
                    co[1] = if a_in > xm_ar {
                        eval_exp(&coeffs_ab[0..3], a_in)
                    } else if a_in < xm_al {
                        eval_exp(&coeffs_ab[3..6], 1.0 - a_in)
                    } else {
                        table_lookup(table_a, a_in)
                    };
                    co[2] = if b_in > xm_br {
                        eval_exp(&coeffs_ab[6..9], b_in)
                    } else if b_in < xm_bl {
                        eval_exp(&coeffs_ab[9..12], 1.0 - b_in)
                    } else {
                        table_lookup(table_b, b_in)
                    };
                }
            }
            1 => {
                // DT_S_SCALE_AUTOMATIC — scale a/b by L ratio
                if l_in > 0.01 {
                    co[1] = ci[1] * l_out / ci[0];
                    co[2] = ci[2] * l_out / ci[0];
                } else {
                    co[1] = ci[1] * low_approx;
                    co[2] = ci[2] * low_approx;
                }
            }
            2 => {
                // DT_S_SCALE_AUTOMATIC_XYZ — apply L curve per XYZ channel
                let mut xyz = lab_to_xyz([ci[0], ci[1], ci[2], ci[3]]);
                for c in 0..3 {
                    xyz[c] = eval_l(table_l, coeffs_l, xm_l, xyz[c]);
                }
                let lab_out = xyz_to_lab(xyz);
                co[0] = lab_out[0];
                co[1] = lab_out[1];
                co[2] = lab_out[2];
            }
            _ => {
                // DT_S_SCALE_AUTOMATIC_RGB — apply L curve to ProPhoto RGB channels
                let mut rgb = lab_to_prophotorgb([ci[0], ci[1], ci[2], ci[3]]);
                if preserve_colors == 0 {
                    // DT_RGB_NORM_NONE — per-channel
                    for c in 0..3 {
                        rgb[c] = eval_l(table_l, coeffs_l, xm_l, rgb[c]);
                    }
                } else {
                    let lum = rgb_norm(rgb[0], rgb[1], rgb[2], preserve_colors);
                    if lum > 0.0 {
                        let curve_lum = eval_l(table_l, coeffs_l, xm_l, lum);
                        let ratio = curve_lum / lum;
                        rgb[0] *= ratio;
                        rgb[1] *= ratio;
                        rgb[2] *= ratio;
                    }
                }
                let lab_out = prophotorgb_to_lab(rgb);
                co[0] = lab_out[0];
                co[1] = lab_out[1];
                co[2] = lab_out[2];
            }
        }
        co[3] = ci[3];
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// # Safety
/// `table_l/a/b` must each point to 65536 floats.
/// `unbounded_coeffs_l` must point to 3 floats.
/// `unbounded_coeffs_ab` must point to 12 floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_tonecurve_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    table_l: *const f32,
    table_a: *const f32,
    table_b: *const f32,
    unbounded_coeffs_l: *const f32,
    unbounded_coeffs_ab: *const f32,
    autoscale_ab: i32,
    unbound_ab: i32,
    preserve_colors: i32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tl: &[f32; 65536] = &*(table_l as *const [f32; 65536]);
    let ta: &[f32; 65536] = &*(table_a as *const [f32; 65536]);
    let tb: &[f32; 65536] = &*(table_b as *const [f32; 65536]);
    let cl = std::slice::from_raw_parts(unbounded_coeffs_l, 3);
    let cab = std::slice::from_raw_parts(unbounded_coeffs_ab, 12);
    process_pixels(input, output, tl, ta, tb, cl, cab, autoscale_ab, unbound_ab, preserve_colors);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve_tools::{CATMULL_ROM, MONOTONE_HERMITE};

    /// C default node sets (init(), tonecurve.c:828-837).
    fn default_nodes_l() -> Vec<(f32, f32)> {
        vec![(0.0, 0.0), (1.0, 1.0)]
    }
    fn default_nodes_ab() -> Vec<(f32, f32)> {
        vec![(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    }

    fn identity_lut() -> Box<[f32; 65536]> {
        // L identity: maps L/100 → L  (table value = 100 * i/65536 ≈ L)
        // But the lookup is done on L/100, so table[i] = 100*i/65536 → output ≈ L
        // Actually table stores output in [0,100] for L; simpler: identity in [0,1]
        let mut t = Box::new([0.0f32; 65536]);
        for (i, v) in t.iter_mut().enumerate() {
            *v = i as f32 / 65535.0;
        }
        t
    }

    fn flat_lut(val: f32) -> Box<[f32; 65536]> {
        Box::new([val; 65536])
    }

    fn identity_coeffs_l() -> [f32; 3] {
        // eval_exp used when x >= xm_l = 1/coeffs_l[0]; set coeffs_l[0]=1e-9 so xm_l is huge
        [1e-9, 1.0, 1.0]
    }

    fn identity_coeffs_ab() -> [f32; 12] {
        // same trick — xm_ar/br very large, xm_al/bl very small
        let c = [1e-9f32, 1.0, 1.0];
        let mut r = [0.0f32; 12];
        for i in 0..4 {
            r[i * 3..i * 3 + 3].copy_from_slice(&c);
        }
        r
    }

    #[test]
    fn manual_mode_identity_lut() {
        let lut_id = identity_lut();
        let lut_flat = flat_lut(0.5); // a/b curves → constant 0.5 (just testing L path)
        let input = vec![50.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(
            &input, &mut output,
            &lut_id, &lut_flat, &lut_flat,
            &identity_coeffs_l(), &identity_coeffs_ab(),
            0, 0, 0,
        );
        // L_in = 0.5 → lut[32767] = 32767/65535 ≈ 0.5; L_out stays ≈ 0.5
        assert!((output[0] - 0.5).abs() < 0.01, "L out: {}", output[0]);
    }

    #[test]
    fn automatic_mode_scales_ab() {
        // If L goes from 50 to 60, a should scale by 60/50=1.2
        let mut lut_l = Box::new([0.0f32; 65536]);
        for (i, v) in lut_l.iter_mut().enumerate() {
            *v = 1.2 * i as f32 / 65535.0; // L_out = 1.2 * L_in (in 0..1 units)
        }
        let lut_flat = flat_lut(0.0);
        let input = vec![50.0f32, 10.0, -5.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(
            &input, &mut output,
            &lut_l, &lut_flat, &lut_flat,
            &identity_coeffs_l(), &identity_coeffs_ab(),
            1, 0, 0,
        );
        // co[1] = ci[1] * l_out / ci[0] = 10.0 * ~0.6 / 50.0 ≈ 0.12
        let l_out = output[0];
        let expected_a = input[1] * l_out / input[0]; // divide by Lab L (50.0), not l_in
        assert!((output[1] - expected_a).abs() < 0.01, "a scaled: {}", output[1]);
    }

    #[test]
    fn alpha_passes_through() {
        let lut_id = identity_lut();
        let lut_flat = flat_lut(0.5);
        let input = vec![50.0f32, 0.0, 0.0, 0.77];
        let mut output = vec![0.0f32; 4];
        process_pixels(
            &input, &mut output,
            &lut_id, &lut_flat, &lut_flat,
            &identity_coeffs_l(), &identity_coeffs_ab(),
            0, 0, 0,
        );
        assert!((output[3] - 0.77).abs() < 1e-7, "alpha: {}", output[3]);
    }

    #[test]
    fn xyz_mode_output_finite() {
        let lut_id = identity_lut();
        let lut_flat = flat_lut(0.5);
        let input = vec![60.0f32, 10.0, -8.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(
            &input, &mut output,
            &lut_id, &lut_flat, &lut_flat,
            &identity_coeffs_l(), &identity_coeffs_ab(),
            2, 0, 0,
        );
        for (i, &v) in output[..3].iter().enumerate() {
            assert!(v.is_finite(), "ch{i}: {v}");
        }
    }

    // ── build_lut (commit_params port) ────────────────────────────────────

    /// Identity curve + RGB-linked mode: the G→G re-derivation leaves t_l as
    /// the identity ramp in [0,1] units (within u16 quantisation).
    #[test]
    fn build_lut_identity_is_ramp_under_rgb_mode() {
        let tabs = build_lut(
            &default_nodes_l(),
            MONOTONE_HERMITE,
            &default_nodes_ab(),
            MONOTONE_HERMITE,
            &default_nodes_ab(),
            MONOTONE_HERMITE,
            DT_S_SCALE_AUTOMATIC_RGB,
        );
        for k in [0usize, 1, 4096, 32768, 65280, 65535] {
            let want = k as f32 / 65536.0;
            // Tolerance covers u16 quantisation plus ProPhoto round-trip
            // float noise (~1e-4 worst case near the top of the ramp).
            assert!(
                (tabs.table_l[k] - want).abs() < 2e-4,
                "@{k}: {} vs {want}",
                tabs.table_l[k]
            );
        }
        // Tail coefficients finite with positive inverse-x (eval usable).
        assert!(tabs.unbounded_coeffs_l[0] > 0.0);
        assert!(tabs.unbounded_coeffs_l.iter().all(|c| c.is_finite()));
    }

    /// A mid-tones-dropping L curve must darken gray input end to end:
    /// process_pixels on a neutral Lab pixel with the built tables moves L
    /// down and keeps a/b at ~0 (preserve colors AVERAGE).
    #[test]
    fn build_lut_contrast_curve_darkens_neutral_pixel() {
        let nodes = vec![(0.0f32, 0.0), (0.5, 0.35), (1.0, 1.0)];
        let tabs = build_lut(
            &nodes,
            CATMULL_ROM,
            &default_nodes_ab(),
            MONOTONE_HERMITE,
            &default_nodes_ab(),
            MONOTONE_HERMITE,
            DT_S_SCALE_AUTOMATIC_RGB,
        );
        let input = vec![50.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(
            &input, &mut output,
            &tabs.table_l, &tabs.table_a, &tabs.table_b,
            &tabs.unbounded_coeffs_l, &tabs.unbounded_coeffs_ab,
            DT_S_SCALE_AUTOMATIC_RGB, 1, /* DT_RGB_NORM_AVERAGE */ 3,
        );
        assert!(
            output[0] < 45.0,
            "contrast curve must darken L=50: got {}",
            output[0]
        );
        assert!(output[0] > 30.0, "not that dark: {}", output[0]);
        // Neutral stays near-neutral under AVERAGE preservation.
        assert!(output[1].abs() < 2.0 && output[2].abs() < 2.0,
            "a/b: {}, {}", output[1], output[2]);
    }

    /// MANUAL mode skips the RGB re-derivation, so tables keep their Lab-unit
    /// scaling (L×100 / ab ×256−128): an identity a-curve maps mid-gray a
    /// (raw 0.5 → −128+128 = 0) back to ≈0.
    #[test]
    fn build_lut_manual_mode_keeps_lab_units() {
        let tabs = build_lut(
            &default_nodes_l(),
            MONOTONE_HERMITE,
            &[(0.0, 0.0), (1.0, 1.0)],
            MONOTONE_HERMITE,
            &[(0.0, 0.0), (1.0, 1.0)],
            MONOTONE_HERMITE,
            DT_S_SCALE_MANUAL,
        );
        // raw table_a[32768] ≈ 0.5·65535/65536 → ·256−128 ≈ −0.002.
        assert!(tabs.table_a[32768].abs() < 0.05, "{}", tabs.table_a[32768]);
        assert!((tabs.table_l[32768] - 50.0).abs() < 0.05, "{}", tabs.table_l[32768]);
    }

    /// XYZ-linked re-derivation maps Y→Y: identity curve stays the identity
    /// ramp (XYZ luminance of equal channels is Y itself after round-trip).
    #[test]
    fn build_lut_xyz_identity_stays_identity() {
        let tabs = build_lut(
            &default_nodes_l(),
            MONOTONE_HERMITE,
            &default_nodes_ab(),
            MONOTONE_HERMITE,
            &default_nodes_ab(),
            MONOTONE_HERMITE,
            DT_S_SCALE_AUTOMATIC_XYZ,
        );
        for k in [0usize, 8192, 32768, 60000, 65535] {
            let want = k as f32 / 65536.0;
            assert!(
                (tabs.table_l[k] - want).abs() < 2e-3,
                "@{k}: {} vs {want}",
                tabs.table_l[k]
            );
        }
    }

    /// Full round trip: built-identity tables through process_pixels leave a
    /// neutral pixel untouched (the module's neutral-state contract).
    #[test]
    fn built_identity_tables_are_pixel_neutral() {
        let tabs = build_lut(
            &default_nodes_l(),
            MONOTONE_HERMITE,
            &default_nodes_ab(),
            MONOTONE_HERMITE,
            &default_nodes_ab(),
            MONOTONE_HERMITE,
            DT_S_SCALE_AUTOMATIC_RGB,
        );
        let input = vec![50.0f32, 4.0, -3.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(
            &input, &mut output,
            &tabs.table_l, &tabs.table_a, &tabs.table_b,
            &tabs.unbounded_coeffs_l, &tabs.unbounded_coeffs_ab,
            DT_S_SCALE_AUTOMATIC_RGB, 1, 3,
        );
        assert!((output[0] - 50.0).abs() < 0.5, "L: {}", output[0]);
        assert!((output[1] - 4.0).abs() < 0.5, "a: {}", output[1]);
        assert!((output[2] - (-3.0)).abs() < 0.5, "b: {}", output[2]);
    }
}
