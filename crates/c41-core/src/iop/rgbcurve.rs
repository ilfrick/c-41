use crate::{params::IopParams, roi::RoiIn, Result};
use super::{colisa::estimate_exp, IopProcess};
use crate::{color::rgb_norm, curve_tools};

pub struct RgbCurve;

impl IopProcess for RgbCurve {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "rgbcurve" }
}

/// `coeff[1] * (x * coeff[0]).powf(coeff[2])` — matches dt_iop_eval_exp.
#[inline(always)]
fn eval_exp(coeffs: &[f32], x: f32) -> f32 {
    coeffs[1] * (x * coeffs[0]).powf(coeffs[2])
}

#[inline(always)]
fn lut_or_exp(tbl: &[f32], coeffs: &[f32], xm: f32, v: f32) -> f32 {
    if v < xm {
        tbl[((v * 0x1_0000_u32 as f32) as usize).clamp(0, 0xffff)]
    } else {
        eval_exp(coeffs, v)
    }
}

/// Prebuilt per-channel LUTs + tail-extrapolation coefficients: the output of
/// `_generate_curve_lut` (rgbcurve.c:1677), produced by [`build_luts`] once per
/// render and consumed by [`process_pixels`] / the FFI kernel.
pub struct RgbCurveLuts {
    pub table_r: Box<[f32; 65536]>,
    pub table_g: Box<[f32; 65536]>,
    pub table_b: Box<[f32; 65536]>,
    /// Right-hand exponential tails per channel (`unbounded_coeffs[ch][3]`);
    /// `xm = 1 / coeffs[ch][0]` is derived at apply time, exactly like C's
    /// process() does.
    pub unbounded_coeffs: [[f32; 3]; 3],
}

/// Port of `_generate_curve_lut`'s sampling half (rgbcurve.c:1725-1749): each
/// channel's anchors are sampled at 0x10000 resolution through the V1 sampler
/// (`dt_draw_curve_calc_values`), then an exponential tail is fitted over
/// x∈{0.7,0.8,0.9,1.0}·xm with lookups mirrored from the table. Unlike
/// tonecurve there is no autoscale re-derivation here — commit_params is
/// trivial and the tables are built in process().
///
/// Each node slice must be non-empty with its last anchor at x ≤ 1 (the
/// pipeline pins endpoints before calling, as the editor guarantees).
pub fn build_luts(
    nodes_r: &[(f32, f32)],
    type_r: u32,
    nodes_g: &[(f32, f32)],
    type_g: u32,
    nodes_b: &[(f32, f32)],
    type_b: u32,
) -> RgbCurveLuts {
    let channels = [(nodes_r, type_r), (nodes_g, type_g), (nodes_b, type_b)];
    let mut raw = vec![0.0f32; 0x1_0000];
    let mut tables = [
        Box::new([0.0f32; 65536]),
        Box::new([0.0f32; 65536]),
        Box::new([0.0f32; 65536]),
    ];
    let mut coeffs = [[0.0f32; 3]; 3];
    // Same int-truncation + clamp lookup the pixel loop uses
    // ((int)(x·0x10000) CLAMPed to [0, 0xffff]).
    let lut_index =
        |v: f32| -> usize { ((v * 0x1_0000_u32 as f32) as i64).clamp(0, 0xffff) as usize };
    for (ch, &(nodes, ty)) in channels.iter().enumerate() {
        curve_tools::curve_data_sample(nodes, ty, 0.0, 1.0, &mut raw);
        tables[ch].copy_from_slice(&raw);
        // Extrapolation for the unbounded right tail (:1733-1748).
        let xm = nodes[nodes.len() - 1].0;
        let xs: Vec<f32> = [0.7f32, 0.8, 0.9, 1.0].iter().map(|m| m * xm).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| tables[ch][lut_index(x)]).collect();
        coeffs[ch] = estimate_exp(&xs, &ys);
    }
    let [table_r, table_g, table_b] = tables;
    RgbCurveLuts {
        table_r,
        table_g,
        table_b,
        unbounded_coeffs: coeffs,
    }
}

/// Safe whole-buffer application used by the preview/export pipeline — the
/// same semantics as [`darkroom_rgbcurve_process`], which delegates here so
/// the two paths cannot drift. `autoscale`: 0 = AUTOMATIC_RGB (linked
/// channels), anything else = MANUAL_RGB (independent per-channel curves);
/// `preserve_colors`: 0 = NONE, otherwise a [`crate::color::rgb_norm`] mode.
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    table_r: &[f32],
    table_g: &[f32],
    table_b: &[f32],
    unbounded_coeffs: &[[f32; 3]; 3],
    autoscale: i32,
    preserve_colors: i32,
) {
    // C process() derives xm from the coefficients at apply time
    // (rgbcurve.c:1754-1756).
    let [xm_r, xm_g, xm_b] = [
        1.0 / unbounded_coeffs[0][0],
        1.0 / unbounded_coeffs[1][0],
        1.0 / unbounded_coeffs[2][0],
    ];
    for (o, i) in output.chunks_exact_mut(4).zip(input.chunks_exact(4)) {
        if autoscale == 1 {
            // MANUAL_RGB: independent per-channel curves.
            o[0] = lut_or_exp(table_r, &unbounded_coeffs[0], xm_r, i[0]);
            o[1] = lut_or_exp(table_g, &unbounded_coeffs[1], xm_g, i[1]);
            o[2] = lut_or_exp(table_b, &unbounded_coeffs[2], xm_b, i[2]);
        } else if preserve_colors == 0 {
            // AUTOMATIC_RGB, no colour preservation: the R curve drives all
            // three channels identically.
            o[0] = lut_or_exp(table_r, &unbounded_coeffs[0], xm_r, i[0]);
            o[1] = lut_or_exp(table_r, &unbounded_coeffs[0], xm_r, i[1]);
            o[2] = lut_or_exp(table_r, &unbounded_coeffs[0], xm_r, i[2]);
        } else {
            // AUTOMATIC_RGB + preservation: curve the norm luminance, then
            // ratio-scale RGB so chroma is untouched.
            let lum = rgb_norm(i[0], i[1], i[2], preserve_colors);
            if lum > 0.0 {
                let ratio = lut_or_exp(table_r, &unbounded_coeffs[0], xm_r, lum) / lum;
                o[0] = ratio * i[0];
                o[1] = ratio * i[1];
                o[2] = ratio * i[2];
            } else {
                o[0] = i[0];
                o[1] = i[1];
                o[2] = i[2];
            }
        }
        o[3] = i[3];
    }
}

/// RGB-curve IOP: per-channel or linked-channel LUT tone mapping.
///
/// autoscale: 0 = AUTOMATIC_RGB (linked), 1 = MANUAL_RGB (independent channels)
/// preserve_colors: 0 = NONE, non-zero = luminance-norm mode (uses rgb_norm from color.rs)
///
/// Each table is 65536 floats; each unbounded_coeffs group is 3 floats.
/// unbounded_coeffs layout: [R*3 | G*3 | B*3] = 9 floats total.
/// xm_r/g/b = 1.0 / unbounded_coeffs[channel][0] (pre-computed by C caller).
#[no_mangle]
pub unsafe extern "C" fn darkroom_rgbcurve_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    table_r: *const f32,        // 65536 floats
    table_g: *const f32,        // 65536 floats
    table_b: *const f32,        // 65536 floats
    unbounded_r: *const f32,    // 3 floats
    unbounded_g: *const f32,    // 3 floats
    unbounded_b: *const f32,    // 3 floats
    _xm_r: f32,
    _xm_g: f32,
    _xm_b: f32,
    autoscale: i32,     // 0 = AUTOMATIC_RGB, 1 = MANUAL_RGB
    preserve_colors: i32, // 0 = NONE
) {
    let inp = std::slice::from_raw_parts(in_buf, npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let tr = std::slice::from_raw_parts(table_r, 0x10000);
    let tg = std::slice::from_raw_parts(table_g, 0x10000);
    let tb = std::slice::from_raw_parts(table_b, 0x10000);
    let ur = std::slice::from_raw_parts(unbounded_r, 3);
    let ug = std::slice::from_raw_parts(unbounded_g, 3);
    let ub = std::slice::from_raw_parts(unbounded_b, 3);

    // The xm arguments mirror C process()'s derived values; process_pixels
    // re-derives them from the coefficients (identical arithmetic), so the
    // raw pointers are folded into per-CHANNEL coeff rows here (row ch must be
    // channel ch's full [c0,c1,c2] triple — not a transpose).
    let mut coeffs = [[0.0f32; 3]; 3];
    for (dst, src) in coeffs.iter_mut().zip([ur, ug, ub]) {
        dst.copy_from_slice(src);
    }
    process_pixels(inp, out, tr, tg, tb, &coeffs, autoscale, preserve_colors);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve_tools::{CATMULL_ROM, CUBIC_SPLINE, MONOTONE_HERMITE};

    fn identity_lut() -> Vec<f32> {
        (0..0x10000usize).map(|i| i as f32 / 0xffff as f32).collect()
    }
    fn linear_coeffs() -> [f32; 3] { [1.0, 1.0, 1.0] }

    fn identity_nodes() -> Vec<(f32, f32)> {
        vec![(0.0, 0.0), (1.0, 1.0)]
    }

    #[test]
    fn build_luts_identity_is_ramp_with_linear_tail() {
        let luts = build_luts(
            &identity_nodes(), MONOTONE_HERMITE,
            &identity_nodes(), MONOTONE_HERMITE,
            &identity_nodes(), MONOTONE_HERMITE,
        );
        // Every table is the identity ramp.
        for t in [&luts.table_r[..], &luts.table_g[..], &luts.table_b[..]] {
            assert!(
                t.iter()
                    .enumerate()
                    .all(|(k, &v)| (v - k as f32 / 0xffff as f32).abs() < 1e-4),
                "identity anchors must sample to a ramp"
            );
        }
        // The tail fitted over a straight segment recovers the linear
        // exponential y = x (coeffs ≈ [1, 1, 1]).
        for c in &luts.unbounded_coeffs {
            assert!((c[0] - 1.0).abs() < 1e-3 && (c[1] - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn linked_preserve_mode_scales_all_channels_by_one_ratio() {
        // Midtone-pulling R curve, applied linked with luminance preservation.
        let pull = vec![(0.0f32, 0.0f32), (0.5, 0.3), (1.0, 1.0)];
        let luts = build_luts(
            &pull, CATMULL_ROM,
            &identity_nodes(), MONOTONE_HERMITE,
            &identity_nodes(), MONOTONE_HERMITE,
        );
        let inp = [0.5f32, 0.4, 0.3, 1.0];
        let mut out = [0f32; 4];
        process_pixels(&inp, &mut out, &luts.table_r[..], &luts.table_g[..], &luts.table_b[..],
            &luts.unbounded_coeffs, 0, 1);
        // Chroma preservation: every channel is scaled by the SAME factor.
        let r = |o: f32, i: f32| o / i;
        assert!((r(out[0], inp[0]) - r(out[1], inp[1])).abs() < 1e-4);
        assert!((r(out[1], inp[1]) - r(out[2], inp[2])).abs() < 1e-4);
        // …and the factor darkens (the curve sits below the diagonal near the
        // midtone pivot).
        assert!(out[0] < inp[0] && out[1] < inp[1] && out[2] < inp[2]);
    }

    #[test]
    fn manual_mode_routes_each_channel_through_its_own_table() {
        let mut up = identity_lut();
        let mut down = identity_lut();
        for v in &mut up { *v = (*v * 2.0).min(1.0); }
        for v in &mut down { *v *= 0.5; }
        let coeffs = [linear_coeffs(), linear_coeffs(), linear_coeffs()];
        let inp = [0.3f32, 0.4, 0.5, 1.0];
        let mut out = [0f32; 4];
        process_pixels(&inp, &mut out, &up, &down, &identity_lut(), &coeffs, 1, 0);
        assert!((out[0] - 0.6).abs() < 1e-3, "R doubles");
        assert!((out[1] - 0.2).abs() < 1e-3, "G halves");
        assert!((out[2] - 0.5).abs() < 1e-3, "B unchanged");
    }

    #[test]
    fn ffi_kernel_matches_safe_path() {
        // The FFI entry must stay byte-equivalent to process_pixels (it now
        // delegates, but pin it so a future rewrite cannot drift).
        let pull = vec![(0.0f32, 0.0f32), (0.5, 0.35), (1.0, 1.0)];
        let luts = build_luts(
            &pull, CUBIC_SPLINE,
            &identity_nodes(), MONOTONE_HERMITE,
            &identity_nodes(), MONOTONE_HERMITE,
        );
        let npx = 7;
        let mut inp = Vec::with_capacity(npx * 4);
        for k in 0..npx {
            let v = k as f32 / (npx - 1) as f32;
            inp.extend_from_slice(&[v * 0.9, v * 0.5, v, 0.8]);
        }
        let mut safe = inp.clone();
        process_pixels(&inp, &mut safe, &luts.table_r[..], &luts.table_g[..], &luts.table_b[..],
            &luts.unbounded_coeffs, 0, 2);
        let mut ffi_out = inp.clone();
        unsafe {
            darkroom_rgbcurve_process(
                inp.as_ptr(), ffi_out.as_mut_ptr(), npx as usize,
                luts.table_r.as_ptr(), luts.table_g.as_ptr(), luts.table_b.as_ptr(),
                luts.unbounded_coeffs[0].as_ptr(), luts.unbounded_coeffs[1].as_ptr(),
                luts.unbounded_coeffs[2].as_ptr(),
                1.0, 1.0, 1.0,
                0, 2,
            )
        };
        assert_eq!(safe, ffi_out, "FFI and safe paths must agree exactly");
    }

    #[test]
    fn manual_mode_identity() {
        let tbl = identity_lut();
        let c = linear_coeffs();
        let inp = [0.3f32, 0.5, 0.8, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgbcurve_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tbl.as_ptr(), tbl.as_ptr(), tbl.as_ptr(),
                c.as_ptr(), c.as_ptr(), c.as_ptr(),
                1.0, 1.0, 1.0,
                1, 0,
            )
        };
        assert!((out[0] - 0.3).abs() < 1e-4);
        assert!((out[1] - 0.5).abs() < 1e-4);
        assert!((out[2] - 0.8).abs() < 1e-4);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn automatic_none_applies_r_curve_to_all() {
        let mut tbl = identity_lut();
        // Double all values
        for v in &mut tbl { *v = (*v * 2.0).min(1.0); }
        let c = linear_coeffs();
        let inp = [0.2f32, 0.4, 0.6, 0.5];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgbcurve_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tbl.as_ptr(), tbl.as_ptr(), tbl.as_ptr(),
                c.as_ptr(), c.as_ptr(), c.as_ptr(),
                1.0, 1.0, 1.0,
                0, 0,
            )
        };
        // All channels should be doubled (clamped to 1.0 where necessary)
        assert!((out[0] - (0.2f32 * 2.0).min(1.0)).abs() < 1e-3);
        assert!((out[1] - (0.4f32 * 2.0).min(1.0)).abs() < 1e-3);
        assert!((out[2] - (0.6f32 * 2.0).min(1.0)).abs() < 1e-3);
    }

    #[test]
    fn alpha_always_passes_through() {
        let tbl = identity_lut();
        let c = linear_coeffs();
        let inp = [0.5f32, 0.5, 0.5, 0.75];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_rgbcurve_process(
                inp.as_ptr(), out.as_mut_ptr(), 1,
                tbl.as_ptr(), tbl.as_ptr(), tbl.as_ptr(),
                c.as_ptr(), c.as_ptr(), c.as_ptr(),
                1.0, 1.0, 1.0,
                1, 0,
            )
        };
        assert_eq!(out[3], 0.75);
    }
}
