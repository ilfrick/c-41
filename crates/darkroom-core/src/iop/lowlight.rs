//! Lowlight IOP — scotopic (rod-vision) simulation in Lab colorspace.
//!
//! Replaces the OMP loop in src/iop/lowlight.c::process().
//!
//! Per-pixel algorithm (Lab input, 4 channels):
//!   XYZ = Lab_to_XYZ(in)
//!   // scotopic luminance (Purkinje effect)
//!   if XYZ[0] > 0.01:
//!     V = XYZ[1] * (1.33*(1 + (XYZ[1]+XYZ[2])/XYZ[0]) - 1.68)
//!   else:
//!     V = XYZ[1] * (1.33*(1 + (XYZ[1]+XYZ[2])/0.01) - 1.68)
//!   V = CLIP(0.5 * V)
//!   w = lut_lookup(lut, in[0] / 100)   // blending weight from curve
//!   XYZ_scotopic = V * XYZ_sw           // XYZ_sw = Lab_to_XYZ([100, 0, −blueness, 0])
//!   XYZ_out = w*XYZ + (1−w)*XYZ_scotopic
//!   out = XYZ_to_Lab(XYZ_out)
//!
//! lut must point to exactly 65536 floats (DT_IOP_LOWLIGHT_LUT_RES = 0x10000).

use crate::{
    color::{lab_to_xyz, xyz_to_lab},
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

pub struct Lowlight;

impl IopProcess for Lowlight {
    fn name(&self) -> &'static str {
        "lowlight"
    }
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(Error::Pipeline("lowlight: use the C FFI entry point (LUT cannot be cast from raw params)".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("lowlight: OpenCL path not yet ported".into()))
    }
}

// ── LUT interpolation (matches C lookup()) ────────────────────────────────────

#[inline(always)]
fn lut_lookup(lut: &[f32; 65536], i: f32) -> f32 {
    const RES: f32 = 65536.0;
    let fi = RES * i;
    let bin0 = (fi as usize).min(65535);
    let bin1 = (fi as usize + 1).min(65535);
    let f = fi - bin0 as f32;
    lut[bin1] * f + lut[bin0] * (1.0 - f)
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// `lut` must have exactly 65536 entries.
#[inline]
/// Build the 65536-entry transition LUT from the 6 band nodes, porting
/// lowlight.c `commit_params` + the **V1** sampler (`CurveDataSample` in
/// `src/common/curve_tools.c`, reached via `dt_draw_curve_calc_values`).
///
/// Two things differ from the V2 sampler used by colorzones, and both matter:
/// V1 rounds with a truncating `+ 0.5` rather than `round()`, and it does not
/// clamp the interpolated value to `[min_y, max_y]` before storing. The spline
/// itself is the same — V1's `catmull_rom_set` computes exactly the non-periodic
/// tangents of `Catmull_Rom_spline::init` — so the shared `crate::splines`
/// machinery is reused.
///
/// `commit_params` wraps the 6 user nodes in two padding anchors (one before,
/// one after) so the curve extends past the [0,1] domain; those are reproduced
/// here, including darktable's asymmetric choice of which y each takes.
pub fn build_transition_lut(transition_x: &[f32; 6], transition_y: &[f32; 6]) -> Box<[f32; 65536]> {
    use crate::splines::{compute_catmull_rom_tangents, eval_knots, Knot};
    const BANDS: usize = 6;
    const RES: usize = 0x10000;

    // Padding anchors, per commit_params: the leading one takes x[BANDS-2] - 1
    // with y[0]; the trailing one x[1] + 1 with y[BANDS-1].
    let mut knots: Vec<Knot> = Vec::with_capacity(BANDS + 2);
    knots.push(Knot { x: transition_x[BANDS - 2] - 1.0, y: transition_y[0], dy: 0.0 });
    for k in 0..BANDS {
        knots.push(Knot { x: transition_x[k], y: transition_y[k], dy: 0.0 });
    }
    knots.push(Knot { x: transition_x[1] + 1.0, y: transition_y[BANDS - 1], dy: 0.0 });

    knots.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    compute_catmull_rom_tangents(&mut knots, false);

    let n = knots.len();
    let x_lim = (knots[0].x, knots[n - 1].x);
    // V1 does not clamp inside the loop, so the evaluator must not either;
    // an effectively infinite y window reproduces that.
    let y_lim = (f32::NEG_INFINITY, f32::INFINITY);

    let out_res = RES as f32;
    let res = 1.0f32 / (RES as f32 - 1.0);
    let first_x = (knots[0].x * (RES as f32 - 1.0)) as i32;
    let first_y = (knots[0].y * (out_res - 1.0)) as i32;
    let last_x = (knots[n - 1].x * (RES as f32 - 1.0)) as i32;
    let last_y = (knots[n - 1].y * (out_res - 1.0)) as i32;
    let (min_y, max_y) = (0i32, (out_res - 1.0) as i32);

    let mut lut: Box<[f32; 65536]> = vec![0.0f32; RES]
        .into_boxed_slice()
        .try_into()
        .expect("65536-element vec converts to a fixed-size array");
    for (i, slot) in lut.iter_mut().enumerate() {
        let ii = i as i32;
        let val = if ii < first_x {
            first_y
        } else if ii > last_x {
            last_y
        } else {
            let s = eval_knots(&knots, i as f32 * res, x_lim, y_lim, false);
            // V1's truncating `+ 0.5`, not round().
            (s * (out_res - 1.0) + 0.5) as i32
        }
        .clamp(min_y, max_y);
        // dt_draw_curve_smaple_values: min + (max-min) * sample / 0x10000,
        // with min=0, max=1 for this module.
        *slot = val as f32 / out_res;
    }
    lut
}

pub fn process_pixels(input: &[f32], output: &mut [f32], blueness: f32, lut: &[f32; 65536]) {
    const COEFF: f32 = 0.5;
    const THRESHOLD: f32 = 0.01;

    // scotopic white: Lab=[100, 0, -blueness] → XYZ
    let xyz_sw = lab_to_xyz([100.0, 0.0, -blueness, 0.0]);

    for (ci, co) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let lab_in = [ci[0], ci[1], ci[2], ci[3]];
        let xyz = lab_to_xyz(lab_in);

        let denom = if xyz[0] > THRESHOLD { xyz[0] } else { THRESHOLD };
        let v = COEFF * xyz[1] * (1.33 * (1.0 + (xyz[1] + xyz[2]) / denom) - 1.68);
        let v = v.clamp(0.0, 1.0);

        let w = lut_lookup(lut, ci[0] / 100.0);

        let xyz_out = [
            w * xyz[0] + (1.0 - w) * v * xyz_sw[0],
            w * xyz[1] + (1.0 - w) * v * xyz_sw[1],
            w * xyz[2] + (1.0 - w) * v * xyz_sw[2],
            ci[3],
        ];

        let lab_out = xyz_to_lab(xyz_out);
        co.copy_from_slice(&lab_out);
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// `lut` must point to an array of exactly 65536 floats.
///
/// # Safety
/// All pointer arguments must be valid for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn darkroom_lowlight_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    blueness: f32,
    lut: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lut_arr: &[f32; 65536] = &*(lut as *const [f32; 65536]);
    process_pixels(input, output, blueness, lut_arr);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_lut_at_defaults_is_flat_half() {
        // darktable's init(): 6 bands at x = k/5, every y = 0.5. A Catmull-Rom
        // through co-linear points is that same constant, so the whole LUT
        // should read 0.5 — the "no transition" state where the module blends
        // scotopic and photopic evenly at every luminance.
        let x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let y = [0.5f32; 6];
        let lut = build_transition_lut(&x, &y);
        for i in [0usize, 1, 1000, 32768, 65535] {
            assert!(
                (lut[i] - 0.5).abs() < 1e-3,
                "lut[{i}] = {} should be ~0.5 at the defaults", lut[i]
            );
        }
    }

    #[test]
    fn transition_lut_follows_the_band_nodes() {
        // A ramp from 0 to 1 across the bands must come out monotonically
        // increasing, near 0 at the bottom and near 1 at the top. This is what
        // catches a padding-anchor or tangent mistake: a wrong leading anchor
        // shows up as a non-monotonic dip near index 0.
        let x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let lut = build_transition_lut(&x, &y);
        assert!(lut[0] < 0.05, "bottom should be near 0: {}", lut[0]);
        assert!(lut[65535] > 0.95, "top should be near 1: {}", lut[65535]);
        let mut prev = lut[0];
        for i in (0..65536).step_by(256) {
            assert!(
                lut[i] >= prev - 1e-3,
                "LUT must be monotonic for a monotonic ramp; dipped at {i}: {} < {prev}", lut[i]
            );
            prev = lut[i];
        }
        assert!(lut.iter().all(|v| v.is_finite()), "LUT holds a non-finite entry");
    }

    fn flat_lut(v: f32) -> Box<[f32; 65536]> {
        Box::new([v; 65536])
    }

    #[test]
    fn full_weight_passthrough() {
        // lut=1 everywhere → w=1 → out = XYZ_to_Lab(Lab_to_XYZ(in)) ≈ in (round-trip)
        let lut = flat_lut(1.0);
        let input = vec![50.0f32, 10.0, -5.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0, &lut);
        assert!((output[0] - 50.0).abs() < 0.01, "L round-trip: {}", output[0]);
        assert!((output[1] - 10.0).abs() < 0.01, "a round-trip: {}", output[1]);
    }

    #[test]
    fn zero_weight_scotopic_blend() {
        // lut=0 everywhere → w=0 → output is purely scotopic (a/b collapse toward blue)
        let lut = flat_lut(0.0);
        let input = vec![50.0f32, 20.0, 30.0, 1.0];
        let mut output = vec![0.0f32; 4];
        // blueness=50 → scotopic Lab has -50 in b channel
        process_pixels(&input, &mut output, 50.0, &lut);
        // output a/b should move toward scotopic a/b (0, -50-ish scaled by V)
        assert!(output[2] < 0.0, "b should be negative under scotopic with blueness: {}", output[2]);
    }

    #[test]
    fn output_is_finite() {
        let lut = flat_lut(0.5);
        let input = vec![60.0f32, 5.0, -10.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 20.0, &lut);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "ch{i}: {v}");
        }
    }
}
