//! Colisa IOP — contrast, brightness, saturation via pre-computed LUTs.
//!
//! Replaces the OMP loop in src/iop/colisa.c::process().
//!
//! Per-pixel algorithm (Lab input, 4 channels):
//!   L' = ctable[L/100 * 65536]  if L < 100, else eval_exp(cunbounded_coeffs, L/100)
//!   L''= ltable[L'/100 * 65536] if L'< 100, else eval_exp(lunbounded_coeffs, L'/100)
//!   a' = a * saturation
//!   b' = b * saturation
//!
//! The LUT tables and unbounded coefficients are owned by the C data struct and
//! passed in as raw pointers — Rust borrows them for the duration of the call.

use crate::{
    iop::{ClBuffer, IopProcess},
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};

/// Matches `dt_iop_eval_exp()` from src/develop/imageop_math.h:
///   `coeff[1] * pow(x * coeff[0], coeff[2])`
#[inline(always)]
fn eval_exp(coeff: &[f32; 3], x: f32) -> f32 {
    coeff[1] * (x * coeff[0]).powf(coeff[2])
}

// ── IopProcess impl ───────────────────────────────────────────────────────────

pub struct Colisa;

impl IopProcess for Colisa {
    fn name(&self) -> &'static str {
        "colisa"
    }

    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        // Colisa params contain 65K-entry LUT tables that are not trivially
        // cast via IopParams::cast. Call through the C FFI path instead.
        Err(Error::Pipeline(
            "colisa: use the C FFI entry point (LUT tables cannot be cast from raw params)".into(),
        ))
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("colisa: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// Apply contrast (via `ctable`) + brightness (via `ltable`) to L and
/// saturation scaling to a/b channels.
///
/// Both LUT slices must have exactly 65536 entries.
/// Both `unbounded_coeffs` slices must have exactly 3 entries.
/// Everything `process_pixels` needs, derived from the three user sliders.
pub struct ColisaData {
    pub ctable: Box<[f32; 65536]>,
    pub cunbounded: [f32; 3],
    pub ltable: Box<[f32; 65536]>,
    pub lunbounded: [f32; 3],
    pub saturation: f32,
}

/// Port of `dt_iop_estimate_exp` (src/develop/imageop_math.h:98).
///
/// Fits `y = y0 * (x/x0)^g` with `(x0, y0)` pinned to the LAST sample, then
/// averages `g = log(y/y0) / log(x/x0)` over the remaining points. The pairs
/// must be ordered by ascending x. Samples where either ratio is non-positive
/// are skipped (the log would be undefined); if that leaves nothing, `g`
/// defaults to 1, i.e. a straight line — the C does the same rather than
/// producing a NaN coefficient.
///
/// The result extrapolates the tone curves beyond 1.0, which is why it matters
/// here: without it, scene-linear highlights above the LUT's domain would clamp.
pub fn estimate_exp(x: &[f32], y: &[f32]) -> [f32; 3] {
    let n = x.len().min(y.len());
    if n == 0 {
        return [1.0, 1.0, 1.0];
    }
    let (x0, y0) = (x[n - 1], y[n - 1]);
    let mut g = 0.0f32;
    let mut cnt = 0u32;
    for k in 0..n - 1 {
        let yy = y[k] / y0;
        let xx = x[k] / x0;
        if yy > 0.0 && xx > 0.0 {
            g += (y[k] / y0).ln() / (x[k] / x0).ln();
            cnt += 1;
        }
    }
    g = if cnt > 0 { g / cnt as f32 } else { 1.0 };
    // x0 == 0 would make coeff[0] infinite; the caller's sample set ends at 1.0,
    // but guard so a degenerate input cannot poison the pixel loop.
    let inv_x0 = if x0 != 0.0 { 1.0 / x0 } else { 0.0 };
    [inv_x0, y0, g]
}

/// Port of colisa.c `commit_params`: rescale the sliders, build both 65536-entry
/// LUTs and fit their unbounded-extrapolation coefficients.
///
/// The three sliders arrive on darktable's -1..1 scale and are rescaled exactly
/// as the C does: contrast and saturation to 0..2 (0 meaning "no contrast, grey
/// plane" / "no saturation, b&w") and brightness to -2..2. The brightness LUT
/// takes a gamma derived asymmetrically — `1/(1+b)` when lifting, `1-b` when
/// darkening — which keeps the two directions visually symmetric.
pub fn commit_params(contrast: f32, brightness: f32, saturation: f32) -> ColisaData {
    let contrast = contrast + 1.0;
    let brightness = brightness * 2.0;
    let saturation = saturation + 1.0;

    let mut ctable: Box<[f32; 65536]> = vec![0.0f32; 65536]
        .into_boxed_slice()
        .try_into()
        .expect("65536-element vec converts to a fixed-size array");
    let mut ltable: Box<[f32; 65536]> = vec![0.0f32; 65536]
        .into_boxed_slice()
        .try_into()
        .expect("65536-element vec converts to a fixed-size array");

    // Safety: both pointers address exactly 0x10000 floats, the documented
    // contract of the two builders.
    unsafe {
        darkroom_colisa_build_contrast_lut(ctable.as_mut_ptr(), contrast);
        let gamma = if brightness >= 0.0 { 1.0 / (1.0 + brightness) } else { 1.0 - brightness };
        darkroom_colisa_build_brightness_lut(ltable.as_mut_ptr(), gamma);
    }

    // Sample the top of each curve and fit the extrapolation, per the C.
    let xs = [0.7f32, 0.8, 0.9, 1.0];
    let sample = |t: &[f32; 65536]| -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (i, x) in xs.iter().enumerate() {
            let idx = ((x * 65536.0) as i32).clamp(0, 0xffff) as usize;
            out[i] = t[idx];
        }
        out
    };
    let cunbounded = estimate_exp(&xs, &sample(&ctable));
    let lunbounded = estimate_exp(&xs, &sample(&ltable));

    ColisaData { ctable, cunbounded, ltable, lunbounded, saturation }
}

#[inline]
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    ctable: &[f32; 65536],
    cunbounded: &[f32; 3],
    ltable: &[f32; 65536],
    lunbounded: &[f32; 3],
    saturation: f32,
) {
    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let l_in = chunk_in[0];

        // contrast LUT
        let l_contrast = if l_in < 100.0 {
            let idx = ((l_in / 100.0 * 65536.0) as usize).min(65535);
            ctable[idx]
        } else {
            eval_exp(cunbounded, l_in / 100.0)
        };

        // brightness LUT
        chunk_out[0] = if l_contrast < 100.0 {
            let idx = ((l_contrast / 100.0 * 65536.0) as usize).min(65535);
            ltable[idx]
        } else {
            eval_exp(lunbounded, l_contrast / 100.0)
        };

        chunk_out[1] = chunk_in[1] * saturation;
        chunk_out[2] = chunk_in[2] * saturation;
        chunk_out[3] = chunk_in[3];
    }
}

// ── C FFI entry point ─────────────────────────────────────────────────────────

/// Called from src/iop/colisa.c in place of the OMP loop.
///
/// `ctable` and `ltable` point to `dt_iop_colisa_data_t.ctable/ltable`
/// (each 65536 floats). `cunbounded_coeffs`/`lunbounded_coeffs` each have 3 floats.
///
/// # Safety
/// All pointer arguments must be valid for the duration of this call.
/// `ctable`/`ltable` must point to arrays of at least 65536 floats.
/// `cunbounded_coeffs`/`lunbounded_coeffs` must point to arrays of at least 3 floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colisa_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    ctable: *const f32,
    cunbounded_coeffs: *const f32,
    ltable: *const f32,
    lunbounded_coeffs: *const f32,
    saturation: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    // Safety: caller guarantees these are valid 65536/3-entry arrays.
    let ct: &[f32; 65536] = &*(ctable as *const [f32; 65536]);
    let cu: &[f32; 3] = &*(cunbounded_coeffs as *const [f32; 3]);
    let lt: &[f32; 65536] = &*(ltable as *const [f32; 65536]);
    let lu: &[f32; 3] = &*(lunbounded_coeffs as *const [f32; 3]);
    process_pixels(input, output, ct, cu, lt, lu, saturation);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_exp_recovers_a_known_power_law() {
        // y = 3 * (x/1.0)^2 sampled on the same grid commit_params uses.
        let xs = [0.7f32, 0.8, 0.9, 1.0];
        let ys: Vec<f32> = xs.iter().map(|x| 3.0 * x.powf(2.0)).collect();
        let c = estimate_exp(&xs, &ys);
        assert!((c[0] - 1.0).abs() < 1e-5, "1/x0: {}", c[0]);
        assert!((c[1] - 3.0).abs() < 1e-5, "y0: {}", c[1]);
        assert!((c[2] - 2.0).abs() < 1e-3, "exponent: {}", c[2]);
    }

    #[test]
    fn estimate_exp_defaults_to_linear_when_no_sample_is_usable() {
        // Non-positive ratios make the logs undefined; the C skips those and
        // falls back to g = 1 rather than emitting NaN. A NaN here would
        // propagate into every extrapolated highlight.
        let xs = [0.7f32, 1.0];
        let ys = [-1.0f32, 2.0];
        let c = estimate_exp(&xs, &ys);
        assert_eq!(c[2], 1.0, "should fall back to linear");
        assert!(c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn commit_params_rescales_and_stays_finite() {
        // Neutral sliders: contrast/saturation 0 -> 1.0, brightness 0 -> 0.0.
        let d = commit_params(0.0, 0.0, 0.0);
        assert_eq!(d.saturation, 1.0, "saturation rescale");
        assert!(d.ctable.iter().all(|v| v.is_finite()), "contrast LUT non-finite");
        assert!(d.ltable.iter().all(|v| v.is_finite()), "brightness LUT non-finite");
        assert!(d.cunbounded.iter().all(|v| v.is_finite()));
        assert!(d.lunbounded.iter().all(|v| v.is_finite()));

        // The extremes of every slider must also stay finite — these are the
        // reachable ends of the UI, not hypothetical inputs.
        for c in [-1.0f32, 1.0] {
            for b in [-1.0f32, 1.0] {
                for sat in [-1.0f32, 1.0] {
                    let d = commit_params(c, b, sat);
                    assert!(
                        d.ctable.iter().all(|v| v.is_finite())
                            && d.ltable.iter().all(|v| v.is_finite()),
                        "non-finite LUT at contrast={c} brightness={b} saturation={sat}"
                    );
                }
            }
        }
    }

    fn identity_lut() -> Box<[f32; 65536]> {
        // identity LUT: index i → 100 * i/65536
        let mut t = Box::new([0.0f32; 65536]);
        for (i, v) in t.iter_mut().enumerate() {
            *v = 100.0 * i as f32 / 65536.0;
        }
        t
    }

    #[test]
    fn eval_exp_basic() {
        // coeff = [1, 1, 1] → y = 1 * pow(x * 1, 1) = x
        let coeff = [1.0f32, 1.0, 1.0];
        assert!((eval_exp(&coeff, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn identity_luts_pass_through_l() {
        let lut = identity_lut();
        let coeff = [1.0f32, 1.0, 1.0]; // eval_exp(c, x) = x
        let input = vec![50.0f32, 10.0, -5.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, &lut, &coeff, &lut, &coeff, 1.0);
        // With identity LUTs and saturation=1, L should round-trip approximately.
        assert!((output[0] - 50.0).abs() < 0.1, "L round-trip failed: {}", output[0]);
        assert!((output[1] - 10.0).abs() < 1e-5);
        assert!((output[2] - (-5.0)).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn saturation_zero_zeroes_ab() {
        let lut = identity_lut();
        let coeff = [1.0f32, 1.0, 1.0];
        let input = vec![60.0f32, 30.0, -20.0, 1.0];
        let mut output = vec![0.0f32; 4];
        process_pixels(&input, &mut output, &lut, &coeff, &lut, &coeff, 0.0);
        assert!(output[1].abs() < 1e-7);
        assert!(output[2].abs() < 1e-7);
    }
}

/// Build the contrast LUT (65536 entries) for colisa commit_params.
/// ≤ 1.0: linear; > 1.0: sigmoid (boost=20). Matches colisa.c:180.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colisa_build_contrast_lut(
    ctable: *mut f32,
    contrast: f32,
) {
    let lut = std::slice::from_raw_parts_mut(ctable, 0x10000);
    const N: f32 = 0x10000 as f32;
    if contrast <= 1.0 {
        for k in 0..0x10000usize {
            lut[k] = contrast * (100.0 * k as f32 / N - 50.0) + 50.0;
        }
    } else {
        let boost = 20.0_f32;
        let cm1sq = boost * (contrast - 1.0).powi(2);
        let cscale = (1.0 + cm1sq).sqrt();
        for k in 0..0x10000usize {
            let kx2m1 = 2.0 * k as f32 / N - 1.0;
            lut[k] = 50.0 * (cscale * kx2m1 / (1.0 + cm1sq * kx2m1 * kx2m1).sqrt() + 1.0);
        }
    }
}

/// Build the brightness LUT (65536 entries) for colisa commit_params.
/// ltable[k] = 100 * (k/0x10000)^gamma. Matches colisa.c:209.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colisa_build_brightness_lut(
    ltable: *mut f32,
    gamma: f32,
) {
    let lut = std::slice::from_raw_parts_mut(ltable, 0x10000);
    const N: f32 = 0x10000 as f32;
    for k in 0..0x10000usize {
        lut[k] = 100.0 * (k as f32 / N).powf(gamma);
    }
}
