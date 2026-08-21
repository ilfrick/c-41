use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};
use crate::color::rgb_norm;

pub struct Basicadj;

impl IopProcess for Basicadj {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "basicadj" }
}

fn hlcurve(level: f32, hlcomp: f32, hlrange: f32) -> f32 {
    if hlcomp > 0.0 {
        let mut val = level + (hlrange - 1.0);
        if val == 0.0 { val = 0.000001; }
        let mut y = (val / hlrange) * hlcomp;
        if y <= -1.0 { y = -0.999999; }
        let r = hlrange / (val * hlcomp);
        y.ln_1p() * r
    } else {
        1.0
    }
}

fn lut_gamma(x: f32, gamma: f32, lut: &[f32]) -> f32 {
    if x > 1.0 {
        x.powf(gamma)
    } else {
        lut[((x * 65536.0) as i32).clamp(0, 65535) as usize]
    }
}

fn lut_contrast(x: f32, contrast: f32, mg: f32, inv_mg: f32, lut: &[f32]) -> f32 {
    if x > 1.0 {
        (x * inv_mg).powf(contrast) * mg
    } else {
        lut[((x * 65536.0) as i32).clamp(0, 65535) as usize]
    }
}

#[no_mangle]
/// # Safety
///
/// `in_buf` and `out_buf` must each point to at least `npixels * 4` valid,
/// readable/writable `f32` values (packed RGBA). `lut_gamma_ptr` and
/// `lut_contrast_ptr` must each point to exactly 65536 valid `f32` values.
/// All pointers must be properly aligned and non-null. The caller is
/// responsible for ensuring no aliasing between `in_buf` and `out_buf`.
pub unsafe extern "C" fn darkroom_basicadj_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    // exposure
    black_point: f32,
    scale: f32,
    // highlight compression
    process_hlcompr: i32,
    hlcomp: f32,
    hlrange: f32,
    lum_r: f32,
    lum_g: f32,
    lum_b: f32,
    // gamma LUT
    process_gamma: i32,
    gamma: f32,
    lut_gamma_ptr: *const f32,
    // contrast
    plain_contrast: i32,
    preserve_colors: i32,
    contrast: f32,
    middle_grey: f32,
    inv_middle_grey: f32,
    lut_contrast_ptr: *const f32,
    // saturation / vibrance
    process_saturation_vibrance: i32,
    saturation: f32,
    vibrance: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    let lg = std::slice::from_raw_parts(lut_gamma_ptr,   65536);
    let lc = std::slice::from_raw_parts(lut_contrast_ptr, 65536);

    for k in (0..npixels * 4).step_by(4) {
        // 1. Exposure
        output[k]     = (input[k]     - black_point) * scale;
        output[k + 1] = (input[k + 1] - black_point) * scale;
        output[k + 2] = (input[k + 2] - black_point) * scale;

        // 2. Highlight compression
        if process_hlcompr != 0 {
            let lum = output[k] * lum_r + output[k + 1] * lum_g + output[k + 2] * lum_b;
            if lum > 0.0 {
                let ratio = hlcurve(lum, hlcomp, hlrange);
                output[k]     *= ratio;
                output[k + 1] *= ratio;
                output[k + 2] *= ratio;
            }
        }

        // 3. Gamma (per channel, values > 0 only)
        if process_gamma != 0 {
            for c in 0..3 {
                if output[k + c] > 0.0 {
                    output[k + c] = lut_gamma(output[k + c], gamma, lg);
                }
            }
        }

        // 4. Plain contrast (per channel, mutually exclusive with preserve_colors)
        if plain_contrast != 0 {
            for c in 0..3 {
                if output[k + c] > 0.0 {
                    output[k + c] = lut_contrast(output[k + c], contrast, middle_grey, inv_middle_grey, lc);
                }
            }
        }

        // 5. Contrast with preserve colors (luminance-based ratio)
        if preserve_colors != 0 {
            // Mode 1 = DT_RGB_NORM_LUMINANCE: the C dt_rgb_norm() uses the working
            // profile's RGB->XYZ Y-row (via dt_ioppr_get_rgb_matrix_luminance), which
            // is exactly `luma` — the same coefficients already used for hlcompr above.
            // rgb_norm() hardcodes ProPhoto for mode 1 (see its doc comment), which
            // diverges from the work profile; use luma directly to stay faithful.
            let lum = if preserve_colors == 1 {
                output[k] * lum_r + output[k + 1] * lum_g + output[k + 2] * lum_b
            } else {
                rgb_norm(output[k], output[k + 1], output[k + 2], preserve_colors)
            };
            if lum > 0.0 {
                let contrast_lum = (lum * inv_middle_grey).powf(contrast) * middle_grey;
                let ratio = contrast_lum / lum;
                output[k]     *= ratio;
                output[k + 1] *= ratio;
                output[k + 2] *= ratio;
            }
        }

        // 6. Saturation / vibrance
        if process_saturation_vibrance != 0 {
            let avg = (output[k] + output[k + 1] + output[k + 2]) / 3.0;
            let d0 = avg - output[k];
            let d1 = avg - output[k + 1];
            let d2 = avg - output[k + 2];
            let delta = (d0 * d0 + d1 * d1 + d2 * d2).sqrt();
            let p = vibrance * (1.0 - delta.powf(vibrance.abs()));
            let factor = saturation + p;
            output[k]     = avg + factor * (output[k]     - avg);
            output[k + 1] = avg + factor * (output[k + 1] - avg);
            output[k + 2] = avg + factor * (output[k + 2] - avg);
        }

        // 7. Alpha passthrough
        output[k + 3] = input[k + 3];
    }
}

/// Everything `darkroom_basicadj_process` needs, derived once per commit.
///
/// Holds the two 65536-entry LUTs, so it is deliberately NOT stored in a
/// `Stage` — that would carry 512 KB per stage and make the enum non-`PartialEq`.
/// The stage keeps the user-facing sliders and builds this on demand, the same
/// split colisa uses.
pub struct BasicadjData {
    pub black_point: f32,
    pub scale: f32,
    pub process_hlcompr: i32,
    pub hlcomp: f32,
    pub hlrange: f32,
    pub luma: [f32; 3],
    pub process_gamma: i32,
    pub gamma: f32,
    pub lut_gamma: std::rc::Rc<[f32; 65536]>,
    pub plain_contrast: i32,
    pub preserve_colors: i32,
    pub contrast: f32,
    pub middle_grey: f32,
    pub inv_middle_grey: f32,
    pub lut_contrast: std::rc::Rc<[f32; 65536]>,
    pub process_saturation_vibrance: i32,
    pub saturation: f32,
    pub vibrance: f32,
}

/// Identity of a LUT pair: every input the two tables depend on, as raw bits so
/// the comparison is exact (and so a NaN key never compares equal to itself,
/// which would be worse than a miss).
type LutKey = (i32, u32, i32, u32, u32, u32);

/// Cached LUT pair keyed by the inputs that produce it, so repeated
/// `commit_params` calls within a render reuse the same tables. See
/// [`LUT_CACHE`] for the memoisation rationale.
type Cached = std::cell::RefCell<Option<(LutKey, std::rc::Rc<[f32; 65536]>, std::rc::Rc<[f32; 65536]>)>>;

thread_local! {
    /// Memo of the last LUT pair built on this thread.
    ///
    /// `Pipeline::process` splits the image into ~64k-pixel bands and calls
    /// `Stage::apply` — hence `commit_params` — **once per band**. On a 20 MP
    /// export that is 306 rebuilds of two 65536-entry tables, i.e. ~40 M `powf`
    /// calls and ~156 MB of allocation churn for tables that are byte-identical
    /// every time, because the params do not vary across bands.
    /// Measured on a 20 MP buffer: 465 ms with the rebuild, 90 ms for a bare
    /// exposure stage over the same data.
    ///
    /// Keyed on the inputs, so this is pure memoisation: same key ⇒ same tables.
    /// Thread-local rather than shared, so it needs no lock and the `Rc` never
    /// crosses a thread; rayon reuses its workers, so the hit rate is
    /// (bands - threads) / bands.
    #[allow(clippy::type_complexity)]
    static LUT_CACHE: Cached = const { std::cell::RefCell::new(None) };
}

fn cached_luts(
    process_gamma: i32,
    gamma: f32,
    plain_contrast: i32,
    contrast: f32,
    middle_grey: f32,
    inv_middle_grey: f32,
) -> (std::rc::Rc<[f32; 65536]>, std::rc::Rc<[f32; 65536]>) {
    let key: LutKey = (
        process_gamma,
        gamma.to_bits(),
        plain_contrast,
        contrast.to_bits(),
        middle_grey.to_bits(),
        inv_middle_grey.to_bits(),
    );
    LUT_CACHE.with(|c| {
        if let Some((k, g, ct)) = c.borrow().as_ref() {
            if *k == key {
                return (g.clone(), ct.clone());
            }
        }
        let mut lut_gamma = Box::new([0.0f32; 65536]);
        let mut lut_contrast = Box::new([0.0f32; 65536]);
        // Only the [0,1] domain is tabulated; the kernel falls back to powf
        // above 1.0. Skipping the fill entirely when neither pass is active
        // matters: an active pass reading an all-zero table would crush every
        // value below 1.0 to black.
        if process_gamma != 0 || plain_contrast != 0 {
            for i in 0..0x10000usize {
                let percentage = i as f32 / 0x10000u32 as f32;
                if process_gamma != 0 {
                    lut_gamma[i] = get_gamma(percentage, gamma);
                }
                if plain_contrast != 0 {
                    lut_contrast[i] = get_contrast(percentage, contrast, middle_grey, inv_middle_grey);
                }
            }
        }
        let g: std::rc::Rc<[f32; 65536]> = std::rc::Rc::from(lut_gamma);
        let ct: std::rc::Rc<[f32; 65536]> = std::rc::Rc::from(lut_contrast);
        *c.borrow_mut() = Some((key, g.clone(), ct.clone()));
        (g, ct)
    })
}

fn get_gamma(x: f32, gamma: f32) -> f32 {
    x.powf(gamma)
}

fn get_contrast(x: f32, contrast: f32, middle_grey: f32, inv_middle_grey: f32) -> f32 {
    (x * inv_middle_grey).powf(contrast) * middle_grey
}

/// Port of basicadj.c `commit_params` plus the per-`process` derivations at
/// `src/iop/basicadj.c:1401-1422` — the C splits them across two places, but
/// they are one computation and the pipeline needs all of it up front.
///
/// `luma` is the Y row of the working space's RGB→XYZ matrix, which is what the
/// C reads out of the work profile (`dt_ioppr_get_rgb_matrix_luminance`). It is
/// passed in rather than derived here so this module does not depend on
/// `pipeline::ColorSpace` — same layering as the rest of `iop`.
///
/// Note `clip` from the C params struct is absent: the migrated
/// `darkroom_basicadj_process` does not implement it, so exposing a slider for
/// it would be a control that does nothing.
#[allow(clippy::too_many_arguments)]
pub fn commit_params(
    black_point: f32,
    exposure: f32,
    hlcompr: f32,
    hlcomprthresh: f32,
    contrast: f32,
    preserve_colors: i32,
    middle_grey: f32,
    brightness: f32,
    saturation: f32,
    vibrance: f32,
    luma: [f32; 3],
) -> BasicadjData {
    // exposure2white(x) = exp2f(-x), then scale = 1 / (white - black_point).
    let white = (-exposure).exp2();
    let scale = 1.0 / (white - black_point);

    let saturation_c = saturation + 1.0;
    let vibrance_c = vibrance / 1.4;
    let contrast_c = contrast + 1.0;
    let middle_grey_c = if middle_grey > 0.0 { middle_grey / 100.0 } else { 0.1842 };
    let inv_middle_grey = 1.0 / middle_grey_c;
    let brightness_c = brightness * 2.0;
    let gamma = if brightness_c >= 0.0 { 1.0 / (1.0 + brightness_c) } else { 1.0 - brightness_c };

    let hlcomp = hlcompr / 100.0;
    let shoulder = ((hlcomprthresh / 100.0) / 8.0) + 0.1;
    let hlrange = 1.0 - shoulder;

    // `preserve_colors` and `plain_contrast` are mutually exclusive, and BOTH are
    // gated on contrast being non-zero — a preserve-colors mode with contrast 0
    // must not run, or every pixel gets a ratio of exactly 1 computed the slow way.
    let plain_contrast = i32::from(preserve_colors == 0 && contrast != 0.0);
    let preserve_colors = if contrast != 0.0 { preserve_colors } else { 0 };
    let process_gamma = i32::from(brightness != 0.0);
    let process_saturation_vibrance = i32::from(saturation != 0.0 || vibrance != 0.0);
    let process_hlcompr = i32::from(hlcompr > 0.0);

    let (lut_gamma, lut_contrast) = cached_luts(
        process_gamma, gamma, plain_contrast, contrast_c, middle_grey_c, inv_middle_grey,
    );

    BasicadjData {
        black_point,
        scale,
        process_hlcompr,
        hlcomp,
        hlrange,
        luma,
        process_gamma,
        gamma,
        lut_gamma,
        plain_contrast,
        preserve_colors,
        contrast: contrast_c,
        middle_grey: middle_grey_c,
        inv_middle_grey,
        lut_contrast,
        process_saturation_vibrance,
        saturation: saturation_c,
        vibrance: vibrance_c,
    }
}

impl BasicadjData {
    /// Safe wrapper over the migrated kernel. `input`/`output` are packed RGBA.
    pub fn process(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        let npixels = input.len() / 4;
        // Safety: both slices hold npixels*4 floats and both LUTs are exactly
        // 65536 entries — the documented contract of the kernel.
        unsafe {
            darkroom_basicadj_process(
                input.as_ptr(),
                output.as_mut_ptr(),
                npixels,
                self.black_point,
                self.scale,
                self.process_hlcompr,
                self.hlcomp,
                self.hlrange,
                self.luma[0],
                self.luma[1],
                self.luma[2],
                self.process_gamma,
                self.gamma,
                self.lut_gamma.as_ptr(),
                self.plain_contrast,
                self.preserve_colors,
                self.contrast,
                self.middle_grey,
                self.inv_middle_grey,
                self.lut_contrast.as_ptr(),
                self.process_saturation_vibrance,
                self.saturation,
                self.vibrance,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(
        input: &[f32],
        black: f32, scale: f32,
        hlcompr: i32, hlcomp: f32, hlrange: f32,
        pg: i32, gamma: f32,
        pc: i32, pv: i32, contrast: f32, mg: f32,
        psv: i32, sat: f32, vib: f32,
    ) -> Vec<f32> {
        let n = input.len() / 4;
        let mut out = vec![0f32; input.len()];
        let lut_g = vec![0f32; 65536];
        let lut_c = vec![0f32; 65536];
        unsafe {
            darkroom_basicadj_process(
                input.as_ptr(), out.as_mut_ptr(), n,
                black, scale,
                hlcompr, hlcomp, hlrange, 0.2126, 0.7152, 0.0722,
                pg, gamma, lut_g.as_ptr(),
                pc, pv, contrast, mg, 1.0 / mg, lut_c.as_ptr(),
                psv, sat, vib,
            );
        }
        out
    }

    #[test]
    fn exposure_identity() {
        let input = vec![0.5, 0.4, 0.3, 1.0];
        let out = call(&input, 0.0, 1.0, 0, 0.0, 1.0, 0, 1.0, 0, 0, 1.0, 0.1842, 0, 1.0, 0.0);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.4).abs() < 1e-6);
        assert!((out[2] - 0.3).abs() < 1e-6);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn exposure_black_and_scale() {
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let out = call(&input, 0.1, 2.0, 0, 0.0, 1.0, 0, 1.0, 0, 0, 1.0, 0.1842, 0, 1.0, 0.0);
        assert!((out[0] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn alpha_passes_through() {
        let input = vec![0.5, 0.5, 0.5, 0.75];
        let out = call(&input, 0.0, 1.0, 0, 0.0, 1.0, 0, 1.0, 0, 0, 1.0, 0.1842, 0, 1.0, 0.0);
        assert_eq!(out[3], 0.75);
    }

    #[test]
    fn hlcurve_zero_hlcomp_returns_one() {
        assert_eq!(hlcurve(0.8, 0.0, 0.8), 1.0);
    }

    #[test]
    fn saturation_zero_is_grey() {
        // saturation=0 → all channels collapse to average (p=0 when vibrance=0)
        let input = vec![0.8, 0.4, 0.2, 1.0];
        let out = call(&input, 0.0, 1.0, 0, 0.0, 1.0, 0, 1.0, 0, 0, 1.0, 0.1842, 1, 0.0, 0.0);
        let avg = (0.8 + 0.4 + 0.2) / 3.0;
        assert!((out[0] - avg).abs() < 1e-5);
        assert!((out[1] - avg).abs() < 1e-5);
        assert!((out[2] - avg).abs() < 1e-5);
    }

    // ── commit_params (derivations ported from basicadj.c:1401-1422) ────────

    const SRGB_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

    fn commit_default() -> BasicadjData {
        // darktable's $DEFAULTs: everything neutral, middle_grey 18.42.
        commit_params(0.0, 0.0, 0.0, 0.0, 0.0, 1, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA)
    }

    #[test]
    fn defaults_are_a_no_op_and_skip_every_optional_pass() {
        let d = commit_default();
        // white = exp2(-0) = 1, black = 0 -> scale = 1: exposure stage is identity.
        assert!((d.scale - 1.0).abs() < 1e-6, "scale {}", d.scale);
        assert_eq!(d.process_gamma, 0);
        assert_eq!(d.plain_contrast, 0);
        assert_eq!(d.process_hlcompr, 0);
        assert_eq!(d.process_saturation_vibrance, 0);
        // contrast == 0 must also disable preserve_colors, or every pixel pays
        // for a ratio that is exactly 1.
        assert_eq!(d.preserve_colors, 0);

        let input = vec![0.25f32, 0.5, 0.75, 1.0];
        let mut out = vec![0f32; 4];
        d.process(&input, &mut out);
        for i in 0..3 {
            assert!((out[i] - input[i]).abs() < 1e-6, "channel {i}: {:?}", out);
        }
    }

    #[test]
    fn exposure_scales_by_powers_of_two() {
        // white = exp2(-exposure); scale = 1/(white - black). +1 EV doubles.
        let d = commit_params(0.0, 1.0, 0.0, 0.0, 0.0, 0, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert!((d.scale - 2.0).abs() < 1e-5, "scale {}", d.scale);
        let input = vec![0.25f32, 0.25, 0.25, 1.0];
        let mut out = vec![0f32; 4];
        d.process(&input, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-5, "{:?}", out);
    }

    #[test]
    fn black_point_lifts_the_floor_before_scaling() {
        // out = (in - black) * scale, and scale itself depends on black.
        let d = commit_params(0.1, 0.0, 0.0, 0.0, 0.0, 0, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert!((d.scale - (1.0 / 0.9)).abs() < 1e-5, "scale {}", d.scale);
        let input = vec![0.1f32, 0.1, 0.1, 1.0];
        let mut out = vec![0f32; 4];
        d.process(&input, &mut out);
        assert!(out[0].abs() < 1e-6, "black point should map to 0, got {:?}", out);
    }

    #[test]
    fn brightness_sign_picks_the_gamma_branch() {
        // gamma = brightness>=0 ? 1/(1+2b) : 1-2b  (brightness is doubled first).
        let up = commit_params(0.0, 0.0, 0.0, 0.0, 0.0, 0, 18.42, 0.5, 0.0, 0.0, SRGB_LUMA);
        assert!((up.gamma - (1.0 / 2.0)).abs() < 1e-6, "gamma {}", up.gamma);
        let down = commit_params(0.0, 0.0, 0.0, 0.0, 0.0, 0, 18.42, -0.5, 0.0, 0.0, SRGB_LUMA);
        assert!((down.gamma - 2.0).abs() < 1e-6, "gamma {}", down.gamma);
        assert_eq!(up.process_gamma, 1);
        // Gamma < 1 brightens: a mid grey must come out higher than it went in.
        let input = vec![0.25f32, 0.25, 0.25, 1.0];
        let mut out = vec![0f32; 4];
        up.process(&input, &mut out);
        assert!(out[0] > 0.25, "brightness +0.5 should lift 0.25, got {}", out[0]);
    }

    #[test]
    fn hlcompr_threshold_maps_to_the_shoulder() {
        // shoulder = ((thresh/100)/8) + 0.1 ; hlrange = 1 - shoulder.
        let d = commit_params(0.0, 0.0, 50.0, 0.0, 0.0, 0, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert_eq!(d.process_hlcompr, 1);
        assert!((d.hlcomp - 0.5).abs() < 1e-6, "hlcomp {}", d.hlcomp);
        assert!((d.hlrange - 0.9).abs() < 1e-6, "hlrange {}", d.hlrange);
        let d2 = commit_params(0.0, 0.0, 50.0, 80.0, 0.0, 0, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert!((d2.hlrange - 0.8).abs() < 1e-6, "hlrange {}", d2.hlrange);
    }

    #[test]
    fn contrast_selects_exactly_one_of_the_two_paths() {
        // preserve_colors = 0 -> the per-channel LUT path.
        let plain = commit_params(0.0, 0.0, 0.0, 0.0, 0.5, 0, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert_eq!((plain.plain_contrast, plain.preserve_colors), (1, 0));
        // preserve_colors set -> the luminance-ratio path, and NOT the LUT one.
        let keep = commit_params(0.0, 0.0, 0.0, 0.0, 0.5, 1, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert_eq!((keep.plain_contrast, keep.preserve_colors), (0, 1));
    }

    #[test]
    fn middle_grey_falls_back_when_non_positive() {
        let d = commit_params(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert!((d.middle_grey - 0.1842).abs() < 1e-6, "mg {}", d.middle_grey);
        assert!((d.inv_middle_grey - 1.0 / 0.1842).abs() < 1e-3);
    }

    #[test]
    fn contrast_lut_is_only_filled_when_the_plain_path_runs() {
        // The builder skips both tables entirely when neither pass is active —
        // worth pinning, because a stale all-zero LUT read by an active pass
        // would silently crush every value below 1.0 to black.
        let off = commit_default();
        assert_eq!(off.lut_contrast[30000], 0.0);
        let on = commit_params(0.0, 0.0, 0.0, 0.0, 0.5, 0, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert!(on.lut_contrast[30000] > 0.0, "contrast LUT not built");
        // get_contrast is monotonic in x, so the table must rise.
        assert!(on.lut_contrast[40000] > on.lut_contrast[30000]);
    }

    #[test]
    fn saturation_minus_one_is_neutral_grey() {
        let d = commit_params(0.0, 0.0, 0.0, 0.0, 0.0, 0, 18.42, 0.0, -1.0, 0.0, SRGB_LUMA);
        assert_eq!(d.process_saturation_vibrance, 1);
        let input = vec![0.8f32, 0.2, 0.4, 1.0];
        let mut out = vec![0f32; 4];
        d.process(&input, &mut out);
        assert!((out[0] - out[1]).abs() < 1e-5 && (out[1] - out[2]).abs() < 1e-5,
                "saturation -1 should collapse to grey, got {:?}", out);
    }

    #[test]
    fn preserve_colors_luminance_uses_working_space_coefficients() {
        // preserve_colors = 1 (LUMINANCE): luminance must be computed from the
        // working-space Y row (luma), NOT ProPhoto.  We detect this by using a
        // working space whose Y row is far from ProPhoto and checking the ratio.
        // SRGB_LUMA = [0.2126, 0.7152, 0.0722]; ProPhoto = [0.288, 0.712, 0.0001].
        // For a pixel of [0.5, 0.0, 0.5], SRGB luma = 0.5 * 0.2126 + 0 + 0.5 * 0.0722 = 0.1424
        // ProPhoto luma = 0.5 * 0.288 + 0 + 0.5 * 0.0001 = 0.14405
        // The difference is small here because green is 0; use a pixel with blue
        // to amplify: [0.0, 0.0, 0.5, 1.0].
        // SRGB: 0.5 * 0.0722 = 0.0361
        // ProPhoto: 0.5 * 0.0001 = 0.00005  (effectively zero)
        // With contrast > 0, the ratio differs dramatically.
        let d = commit_params(0.0, 0.0, 0.0, 0.0, 0.5, 1, 18.42, 0.0, 0.0, 0.0, SRGB_LUMA);
        assert_eq!((d.plain_contrast, d.preserve_colors), (0, 1));
        let input = vec![0.0f32, 0.0, 0.5, 1.0];
        let mut out = vec![0f32; 4];
        d.process(&input, &mut out);
        // With SRGB luma (0.0361), contrast_lum = (0.0361 * 1/0.1842)^1.5 * 0.1842
        // ratio > 0, so the blue channel is scaled non-trivially.
        // If ProPhoto were used, lum ≈ 0.00005 → ratio ≈ 0 → blue nearly zero.
        assert!(out[2] > 0.1, "blue channel should retain significant value with SRGB luma, got {}", out[2]);
    }
}
