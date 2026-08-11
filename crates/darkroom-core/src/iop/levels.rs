//! Levels IOP — black/white-point + gamma correction via pre-computed LUT.
//!
//! Replaces the OMP loop in src/iop/levels.c::process().
//!
//! Per-pixel algorithm (Lab input, 4 channels):
//!   L_in  = in.L / 100
//!   if L_in ≤ level_black: L_out = 0
//!   else:
//!     pct   = (L_in - level_black) / level_range
//!     L_out = pct < 1 ? lut[(pct * 65536) as usize] : 100 * pct^inv_gamma
//!   denom = max(in.L, 0.01)
//!   out.L = L_out
//!   out.a = in.a * L_out / denom   (contrast-preserving chroma scale)
//!   out.b = in.b * L_out / denom
//!   out.α = in.α

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct Levels;

impl IopProcess for Levels {
    fn name(&self) -> &'static str {
        "levels"
    }

    fn process(
        &self,
        _input: &[f32],
        _output: &mut [f32],
        _params: &IopParams,
        _roi: &RoiIn,
    ) -> Result<()> {
        // The 65536-entry LUT lives inside dt_iop_levels_data_t and cannot be
        // trivially cast from IopParams bytes. Use the C FFI entry point instead.
        Err(Error::Pipeline(
            "levels: use the C FFI entry point (LUT cannot be cast from raw params)".into(),
        ))
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("levels: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// Upper bound on the L this module will emit from its overrange (`pct > 1`)
/// branch. Lab→XYZ raises `(L+16)/116` to the third power, which overflows f32
/// above L ≈ 8.1e14; 1e6 is ~4 orders of magnitude above the most extreme real
/// highlight (L=100 is diffuse white) and ~8 orders below that overflow, so it
/// bounds the arithmetic without touching any value a real edit produces.
const L_MAX: f32 = 1.0e6;

/// `lut` must have exactly 65536 entries.
#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    level_black: f32,
    level_range: f32,
    inv_gamma: f32,
    lut: &[f32; 65536],
) {
    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let l_in = chunk_in[0] / 100.0;
        let l_out = if l_in <= level_black {
            0.0_f32
        } else {
            let pct = (l_in - level_black) / level_range;
            if pct < 1.0 {
                let idx = (pct * 65536.0) as usize;
                lut[idx.min(65535)]
            } else {
                // Overrange branch. levels.c leaves this unbounded because it
                // runs display-referred, where L <= 100 keeps pct small. Our
                // pipeline can hand it scene-linear input (L > 100), and a
                // narrow range then makes pct large enough that pct^inv_gamma
                // overflows f32 — Lab→XYZ cubes L, so anything past ~8e14
                // becomes inf and lands as NaN in the pixel buffer. Clamp to
                // L_MAX, far above any real highlight but safely finite
                // through the cube.
                (100.0 * pct.powf(inv_gamma)).min(L_MAX)
            }
        };

        let denom = chunk_in[0].max(0.01);
        chunk_out[0] = l_out;
        chunk_out[1] = chunk_in[1] * l_out / denom;
        chunk_out[2] = chunk_in[2] * l_out / denom;
        chunk_out[3] = chunk_in[3];
    }
}

// ── LUT construction ──────────────────────────────────────────────────────────

/// Port of `compute_lut` (src/iop/levels.c): derive the inverse gamma from the
/// three level stops and fill the 65536-entry LUT with `100 * pct^inv_gamma`.
///
/// `levels` are the black / grey / white stops in the **normalised [0,1]**
/// domain (the C `d->levels[]`, i.e. the 0..100 UI sliders divided by 100).
/// Returns `(inv_gamma, lut)`; feed both to [`process_pixels`] along with
/// `level_black = levels[0]` and `level_range = levels[2] - levels[0]`.
///
/// The grey stop sits at `mid = black + delta` where `delta = (white-black)/2`,
/// so a centred grey gives `tmp = 0` ⇒ `inv_gamma = 1` ⇒ an identity ramp.
///
/// **Contract: `levels[0] < levels[1] < levels[2]`.** darktable guarantees this
/// in the GUI layer (levels.c `color_picker_apply` nudges neighbouring stops by
/// `FLT_EPSILON`), and it is what bounds `tmp` to `[-1, 1]` and `inv_gamma` to
/// `[0.1, 10]`. Callers must re-impose it — `PreviewParams::to_pipeline` clamps
/// the grey stop for exactly this reason. `tmp` is additionally clamped here so
/// an out-of-contract caller degrades to darktable's extreme gamma instead of
/// overflowing to `+inf` (which would put NaN into the pixel buffer via the
/// `pct > 1` branch of [`process_pixels`]).
pub fn build_lut(levels: [f32; 3]) -> (f32, Box<[f32; 65536]>) {
    debug_assert!(
        levels[0] < levels[1] && levels[1] < levels[2],
        "levels stops must satisfy black < grey < white, got {levels:?}"
    );
    // A zero/negative range is a caller bug (to_pipeline rejects it); the floor
    // only keeps the arithmetic defined so we return an extreme-but-finite
    // gamma rather than NaN.
    let delta = ((levels[2] - levels[0]) / 2.0).max(f32::MIN_POSITIVE);
    let mid = levels[0] + delta;
    // Clamped to the range darktable's ordered stops can produce.
    let tmp = ((levels[1] - mid) / delta).clamp(-1.0, 1.0);
    let inv_gamma = 10.0f32.powf(tmp);

    // Heap-allocated directly: `Box::new([0.0; 65536])` materialises 256 KB on
    // the stack first (LLVM elides it under -O but not in debug), which would
    // be a hazard if this ever ran on a rayon worker's 2 MB stack.
    let mut lut: Box<[f32; 65536]> = vec![0.0f32; 65536]
        .into_boxed_slice()
        .try_into()
        .expect("65536-element vec converts to a fixed-size array");
    for (i, v) in lut.iter_mut().enumerate() {
        let percentage = i as f32 / 65536.0;
        *v = 100.0 * percentage.powf(inv_gamma);
    }
    (inv_gamma, lut)
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/levels.c in place of the OMP loop.
///
/// `lut` points to `dt_iop_levels_data_t.lut` (65536 floats).
/// `level_range` is `d->levels[2] - d->levels[0]`, pre-computed in the C wrapper.
///
/// # Safety
/// All pointer arguments must be valid for the duration of this call.
/// `lut` must point to an array of at least 65536 floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_levels_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    level_black: f32,
    level_range: f32,
    inv_gamma: f32,
    lut: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lut_arr: &[f32; 65536] = &*(lut as *const [f32; 65536]);
    process_pixels(input, output, level_black, level_range, inv_gamma, lut_arr);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut() -> Box<[f32; 65536]> {
        // identity: percentage pct → 100 * pct^1.0 = 100 * pct
        // Since lut[i] = 100 * i / 65536
        let mut t = Box::new([0.0f32; 65536]);
        for (i, v) in t.iter_mut().enumerate() {
            *v = 100.0 * i as f32 / 65536.0;
        }
        t
    }

    #[test]
    fn below_black_point_zeroes_l() {
        let lut = identity_lut();
        // level_black=0.5 → any L_in/100 ≤ 0.5 clips to 0
        let input = vec![40.0f32, 10.0, -5.0, 1.0]; // L_in = 0.4 < level_black=0.5
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.5, 0.5, 1.0, &lut);
        assert!(output[0].abs() < 1e-6, "L should be 0 below black point: {}", output[0]);
    }

    #[test]
    fn identity_params_approximate_passthrough() {
        let lut = identity_lut();
        // level_black=0, level_range=1, inv_gamma=1 → L_out ≈ L_in (LUT quantization ~0.15)
        let input = vec![60.0f32, 20.0, -10.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0, 1.0, 1.0, &lut);
        assert!((output[0] - 60.0).abs() < 0.2, "L round-trip: {}", output[0]);
    }

    #[test]
    fn ab_scale_proportional_to_l() {
        let lut = identity_lut();
        // L=50 in → L_out ≈ 50; denom=50; out.a = 10 * 50/50 = 10
        let input = vec![50.0f32, 10.0, -5.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0, 1.0, 1.0, &lut);
        // out.a/in.a ≈ out.L/in.L
        let expected_a = input[1] * output[0] / input[0];
        assert!((output[1] - expected_a).abs() < 0.1, "a scaling: {}", output[1]);
    }

    #[test]
    fn alpha_passes_through() {
        let lut = identity_lut();
        let input = vec![50.0f32, 10.0, -5.0, 0.75];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0, 1.0, 1.0, &lut);
        assert!((output[3] - 0.75).abs() < 1e-7);
    }

    #[test]
    fn build_lut_default_stops_are_the_identity_ramp() {
        // darktable defaults (black 0, grey 50, white 100 on the 0..100 sliders
        // ⇒ 0.0/0.5/1.0 normalised): grey sits exactly at mid, so tmp = 0 and
        // inv_gamma = 10^0 = 1 — the LUT is the plain 100*pct ramp the other
        // tests hand-roll.
        let (inv_gamma, lut) = build_lut([0.0, 0.5, 1.0]);
        assert!((inv_gamma - 1.0).abs() < 1e-6, "inv_gamma: {inv_gamma}");
        let reference = identity_lut();
        for i in [0usize, 1, 12345, 65535] {
            assert!(
                (lut[i] - reference[i]).abs() < 1e-3,
                "lut[{i}] = {} != {}", lut[i], reference[i]
            );
        }
    }

    #[test]
    fn build_lut_gamma_follows_the_grey_stop() {
        // Grey below mid ⇒ tmp < 0 ⇒ inv_gamma < 1 (brightens the midtones);
        // grey above mid ⇒ inv_gamma > 1 (darkens them). Pins the direction so
        // a sign slip in (grey - mid)/delta can't pass.
        let (dark_grey, _) = build_lut([0.0, 0.25, 1.0]);
        let (light_grey, _) = build_lut([0.0, 0.75, 1.0]);
        assert!(dark_grey < 1.0, "grey below mid should give inv_gamma < 1: {dark_grey}");
        assert!(light_grey > 1.0, "grey above mid should give inv_gamma > 1: {light_grey}");
        // 10^((0.25-0.5)/0.5) = 10^-0.5, and the light case is its reciprocal.
        assert!((dark_grey - 10.0f32.powf(-0.5)).abs() < 1e-5);
        assert!((dark_grey * light_grey - 1.0).abs() < 1e-4);
    }

    #[test]
    fn build_lut_stays_bounded_across_the_in_contract_domain() {
        // Regression: the earlier version of this test used one degenerate
        // triple ([0.5,0.5,0.5]) that happened to give tmp = -1, so it passed
        // while `black=0, grey=1.0, white=0.01` produced tmp = 199 ⇒ 10^199 ⇒
        // +inf ⇒ NaN L in the pixel buffer. Sweep every ordered triple and
        // demand darktable's bounded gamma and a finite LUT throughout.
        // (Out-of-contract triples are the caller's job to prevent — see the
        // debug_assert on build_lut and `levels_slider_domain_is_nan_free` in
        // darkroom-ui, which covers the unordered case end to end.)
        let stops = [0.0f32, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0];
        let mut checked = 0;
        for &b in &stops {
            for &g in &stops {
                for &w in &stops {
                    if !(b < g && g < w) {
                        continue;
                    }
                    checked += 1;
                    let (inv_gamma, lut) = build_lut([b, g, w]);
                    assert!(
                        (0.1..=10.0).contains(&inv_gamma),
                        "inv_gamma outside darktable's [0.1,10] for \
                         black={b} grey={g} white={w}: {inv_gamma}"
                    );
                    assert!(
                        lut.iter().all(|v| v.is_finite()),
                        "LUT holds a non-finite entry for black={b} grey={g} white={w}"
                    );
                }
            }
        }
        assert!(checked > 20, "sweep degenerated to {checked} cases");
    }

    #[test]
    fn build_lut_clamp_bounds_gamma_at_the_domain_edges() {
        // The extremes an ordered triple can reach: grey pinned just above
        // black ⇒ tmp → -1 ⇒ gamma → 0.1; just below white ⇒ tmp → +1 ⇒
        // gamma → 10. These are the values the clamp exists to cap at, so a
        // regression that removed it would show up here as an overshoot.
        let (lo, _) = build_lut([0.0, f32::EPSILON, 1.0]);
        let (hi, _) = build_lut([0.0, 1.0 - f32::EPSILON, 1.0]);
        assert!((lo - 0.1).abs() < 1e-3, "low-end gamma: {lo}");
        assert!((hi - 10.0).abs() < 1e-2, "high-end gamma: {hi}");
    }

    #[test]
    fn over_range_uses_powf() {
        let lut = identity_lut();
        // pct > 1: L_out = 100 * pct^inv_gamma
        // level_black=0, level_range=0.5 → pct = (1.0 - 0) / 0.5 = 2.0
        let input = vec![100.0f32, 0.0, 0.0, 1.0]; // L_in = 1.0
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, 0.0, 0.5, 2.0, &lut);
        let expected = 100.0 * 2.0_f32.powf(2.0); // = 400 (unclamped)
        assert!((output[0] - expected).abs() < 0.1, "powf path: {}", output[0]);
    }
}
