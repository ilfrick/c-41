use crate::{
    iop::colisa::estimate_exp,
    params::IopParams,
    roi::RoiIn,
    Error, Result,
};
use super::{ClBuffer, IopProcess};

pub struct Lowpass;

impl IopProcess for Lowpass {
    fn name(&self) -> &'static str {
        "lowpass"
    }

    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        // Lowpass params contain 65K-entry LUT tables that are not trivially
        // cast via IopParams::cast. Call through the C FFI path instead.
        Err(Error::Pipeline(
            "lowpass: use the C FFI entry point (LUT tables cannot be cast from raw params)".into(),
        ))
    }

    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(Error::OpenCl("lowpass: OpenCL path not yet ported".into()))
    }
}

// ── Core pixel loop ───────────────────────────────────────────────────────────

/// Hold the derived tables that [`process_pixels`] needs, owned by the Rust side
/// after [`commit_params`] builds them from the three user sliders.
///
/// Mirrors darktable's `dt_iop_lowpass_data_t`: the contrast and brightness
/// 65536-entry LUTs, their unbounded-extrapolation coefficients, the saturation
/// multiplier and the Lab a/b clamp range.
pub struct LowpassData {
    pub ctable: Box<[f32; 65536]>,
    pub cunbounded: [f32; 3],
    pub ltable: Box<[f32; 65536]>,
    pub lunbounded: [f32; 3],
    pub saturation: f32,
    /// `lab_min_ab` / `lab_max_ab` clamp range for the a/b channels after the
    /// saturation scale. `unbound == true` widens them to ±FLT_MAX.
    pub lab_min_ab: f32,
    pub lab_max_ab: f32,
}

/// Port of lowpass.c `commit_params` (src/iop/lowpass.c:442).
///
/// The lowpass sliders arrive on darktable's own scale and are used **directly**
/// — unlike colisa, which rescales contrast/saturation/brightness from the
/// -1..1 universal slider scale. Here `contrast` feeds the contrast LUT builder
/// straight off the slider (-3..3, default 1.0 = identity), `brightness`
/// selects the LUT gamma via the same asymmetric formula colisa uses, and
/// `saturation` is the raw a/b multiplier (default 1.0 = unchanged).
///
/// `unbound` widens the a/b clamp from ±128 to ±FLT_MAX — darktable's default
/// (the checkbox in the GUI is unchecked only for the scene-referred safety
/// path, which we don't surface here).
pub fn commit_params(contrast: f32, brightness: f32, saturation: f32, unbound: bool) -> LowpassData {
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
        darkroom_lowpass_build_contrast_lut(ctable.as_mut_ptr(), contrast);
        let gamma = if brightness >= 0.0 {
            1.0 / (1.0 + brightness)
        } else {
            1.0 - brightness
        };
        darkroom_lowpass_build_brightness_lut(ltable.as_mut_ptr(), gamma);
    }

    // Sample the top of each curve and fit the extrapolation, per the C.
    let xs = [0.7f32, 0.8, 0.9, 1.0];
    let sample = |t: &Box<[f32; 65536]>| -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (i, x) in xs.iter().enumerate() {
            let idx = ((x * 65536.0) as i32).clamp(0, 0xffff) as usize;
            out[i] = t[idx];
        }
        out
    };
    let cunbounded = estimate_exp(&xs, &sample(&ctable));
    let lunbounded = estimate_exp(&xs, &sample(&ltable));

    let (lab_min_ab, lab_max_ab) = if unbound {
        (-f32::MAX, f32::MAX)
    } else {
        (-128.0, 128.0)
    };

    LowpassData { ctable, cunbounded, ltable, lunbounded, saturation, lab_min_ab, lab_max_ab }
}

#[inline]
/// Apply the contrast/brightness LUT pair + a/b saturation to a blurred Lab
/// buffer, matching the per-pixel loop in `darkroom_lowpass_process`.
///
/// `input` is the original (pre-blur) Lab RGBA — alpha is taken from it.
/// `output` holds the blurred Lab on entry and receives the final result in place.
pub fn process_pixels(
    input: &[f32],
    output: &mut [f32],
    ctable: &[f32; 65536],
    cunbounded: &[f32; 3],
    ltable: &[f32; 65536],
    lunbounded: &[f32; 3],
    saturation: f32,
    lab_min_ab: f32,
    lab_max_ab: f32,
) {
    for (chunk_in, chunk_out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let l_in = chunk_out[0];

        // 1. Contrast LUT on L
        let l_contrast = if l_in < 100.0 {
            let idx = ((l_in / 100.0 * 65536.0) as usize).min(65535);
            ctable[idx]
        } else {
            eval_exp(cunbounded, l_in / 100.0)
        };

        // 2. Brightness LUT on L
        chunk_out[0] = if l_contrast < 100.0 {
            let idx = ((l_contrast / 100.0 * 65536.0) as usize).min(65535);
            ltable[idx]
        } else {
            eval_exp(lunbounded, l_contrast / 100.0)
        };

        // 3. Saturation on a/b, clamped
        chunk_out[1] = (chunk_out[1] * saturation).clamp(lab_min_ab, lab_max_ab);
        chunk_out[2] = (chunk_out[2] * saturation).clamp(lab_min_ab, lab_max_ab);
        // 4. Alpha from the original (pre-blur) pixel
        chunk_out[3] = chunk_in[3];
    }
}

/// Matches dt_iop_eval_exp(): coeff[1] * (x * coeff[0])^coeff[2]
#[inline(always)]
fn eval_exp(coeff: &[f32], x: f32) -> f32 {
    coeff[1] * (x * coeff[0]).powf(coeff[2])
}

/// Low-pass IOP pixel loop (contrast + brightness LUT, saturation scale on a/b).
///
/// Replaces the DT_OMP_FOR loop in src/iop/lowpass.c::process() (after the blur).
/// out_buf already contains the blurred Lab image when this is called.
///
/// ctable/ltable:      float[0x10000] contrast/brightness LUT (L in 0..100 → new L in 0..100)
/// cunbounded/lunbounded: float[3] extrapolation coeffs for L >= 100
/// saturation:         d->saturation (a/b multiplier)
/// lab_min_ab/lab_max_ab: clamping range for a/b channels
///   unbound=0: ±128; unbound=1: ±FLT_MAX (pass f32::MAX/-f32::MAX)
/// Alpha is copied from in_buf (original, pre-blur pixel).
#[no_mangle]
pub unsafe extern "C" fn darkroom_lowpass_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    ctable: *const f32,
    cunbounded: *const f32,
    ltable: *const f32,
    lunbounded: *const f32,
    saturation: f32,
    lab_min_ab: f32,
    lab_max_ab: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let ct = std::slice::from_raw_parts(ctable,    0x10000);
    let cu = std::slice::from_raw_parts(cunbounded, 3);
    let lt = std::slice::from_raw_parts(ltable,    0x10000);
    let lu = std::slice::from_raw_parts(lunbounded, 3);

    for k in (0..npixels * 4).step_by(4) {
        // 1. Contrast LUT on L
        let mut l = output[k];
        l = if l < 100.0 {
            ct[((l / 100.0 * 0x10000_u32 as f32) as i32).clamp(0, 0xffff) as usize]
        } else {
            eval_exp(cu, l / 100.0)
        };

        // 2. Brightness LUT on L
        l = if l < 100.0 {
            lt[((l / 100.0 * 0x10000_u32 as f32) as i32).clamp(0, 0xffff) as usize]
        } else {
            eval_exp(lu, l / 100.0)
        };

        output[k]     = l;
        output[k + 1] = (output[k + 1] * saturation).clamp(lab_min_ab, lab_max_ab);
        output[k + 2] = (output[k + 2] * saturation).clamp(lab_min_ab, lab_max_ab);
        output[k + 3] = input[k + 3]; // alpha from original (pre-blur) pixel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(blurred: &[f32], original_alpha: f32, sat: f32, min_ab: f32, max_ab: f32) -> Vec<f32> {
        let n = blurred.len() / 4;
        let mut out = blurred.to_vec();
        // Identity LUT: ctable[k] = 100*k/0x10000, ltable[k] = 100*k/0x10000
        let ctable: Vec<f32> = (0..0x10000u32).map(|k| 100.0 * k as f32 / 65536.0).collect();
        let ltable = ctable.clone();
        let cu = vec![1.0f32, 1.0, 1.0];
        let lu = vec![1.0f32, 1.0, 1.0];
        let input_with_alpha: Vec<f32> = blurred.chunks(4).enumerate()
            .flat_map(|(_, p)| vec![p[0], p[1], p[2], original_alpha])
            .collect();
        let mut out2 = blurred.to_vec();
        unsafe {
            darkroom_lowpass_process(
                input_with_alpha.as_ptr(), out2.as_mut_ptr(), n,
                ctable.as_ptr(), cu.as_ptr(),
                ltable.as_ptr(), lu.as_ptr(),
                sat, min_ab, max_ab,
            );
        }
        out2
    }

    #[test]
    fn identity_lut_passthrough_l() {
        // identity LUTs: ctable[k]=100*k/65536, ltable same
        // L=50 → index 50/100*65536=32768 → value 100*32768/65536=50.0
        let blurred = vec![50.0, 10.0, -5.0, 0.75];
        let out = run(&blurred, 0.75, 1.0, -128.0, 128.0);
        assert!((out[0] - 50.0).abs() < 0.05, "L: {}", out[0]);
    }

    #[test]
    fn saturation_zero_zeroes_ab() {
        let blurred = vec![50.0, 20.0, -10.0, 1.0];
        let out = run(&blurred, 1.0, 0.0, -128.0, 128.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn alpha_comes_from_input_not_blur() {
        let blurred = vec![50.0, 0.0, 0.0, 0.99]; // blurred alpha=0.99
        let out = run(&blurred, 0.42, 1.0, -128.0, 128.0); // original alpha=0.42
        assert_eq!(out[3], 0.42);
    }

    #[test]
    fn ab_clamped() {
        let blurred = vec![50.0, 200.0, -200.0, 0.0]; // a/b out of normal range
        let out = run(&blurred, 0.0, 1.0, -128.0, 128.0);
        assert!(out[1] <= 128.0);
        assert!(out[2] >= -128.0);
    }
}

/// Build the contrast LUT (65536 entries) for lowpass commit_params.
///
/// Two variants depending on contrast:
///   ≤ 1.0: ctable[k] = contrast * (100*k/0x10000 - 50) + 50  (linear)
///   > 1.0: sigmoid variant (same formula as colisa)
///
/// Matches the DT_OMP_FOR loops at src/iop/lowpass.c:477.
#[no_mangle]
pub unsafe extern "C" fn darkroom_lowpass_build_contrast_lut(
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
        let boost = 5.0_f32;
        let cm1sq = boost * (contrast.abs() - 1.0).powi(2);
        let cscale = (1.0 + cm1sq).sqrt() * contrast.signum();
        for k in 0..0x10000usize {
            let kx2m1 = 2.0 * k as f32 / N - 1.0;
            lut[k] = 50.0 * (cscale * kx2m1 / (1.0 + cm1sq * kx2m1 * kx2m1).sqrt() + 1.0);
        }
    }
}

/// Build the brightness LUT (65536 entries) for lowpass commit_params.
/// ltable[k] = 100 * (k/0x10000)^gamma
/// Matches src/iop/lowpass.c:498.
#[no_mangle]
pub unsafe extern "C" fn darkroom_lowpass_build_brightness_lut(
    ltable: *mut f32,
    gamma: f32,
) {
    let lut = std::slice::from_raw_parts_mut(ltable, 0x10000);
    const N: f32 = 0x10000 as f32;
    for k in 0..0x10000usize {
        lut[k] = 100.0 * (k as f32 / N).powf(gamma);
    }
}
