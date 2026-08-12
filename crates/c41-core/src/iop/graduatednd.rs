use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct GraduatedNd;

impl IopProcess for GraduatedNd {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "graduatednd" }
}

#[inline(always)]
fn compute_density(dens: f32, length: f32) -> f32 {
    let clamped = (0.5 + length).clamp(0.0, 1.0);
    (dens * clamped).exp2()
}

/// The pre-computed geometry + colours `darkroom_graduatednd_process` takes,
/// derived from the user params and the buffer dimensions. Mirrors the block at
/// the top of graduatednd.c `process()` plus its `commit_params` colour step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradNdGeometry {
    pub length_base: f32,
    pub length_inc: f32,
    pub cosv_hh_inv: f32,
    pub filter_hardness: f32,
    /// Filter colour and its complement (`color1 = 1 - color`).
    pub color: [f32; 4],
    pub color1: [f32; 4],
}

/// Port of the geometry derivation in graduatednd.c `process()` (lines ~759-790)
/// together with the colour step from `commit_params`.
///
/// `width`/`height` are the buffer's dimensions. The preview renders the whole
/// image at scale 1.0 with the ROI at the origin, so `ix`/`iy` are 0 — the
/// caller passes `iy` separately to [`darkroom_graduatednd_process`].
///
/// `density` is in EV (-8..8), `hardness` and `offset` are the 0..100 sliders,
/// `rotation` is degrees (-180..180), and `hue`/`saturation` are 0..1.
///
/// The colour is `hsl2rgb(hue, saturation, 0.5)` with alpha 0; for a **negative**
/// density the C inverts it (`1 - c`) before taking the complement, which is what
/// makes a negative density brighten rather than darken.
pub fn commit_geometry(
    width: usize,
    height: usize,
    density: f32,
    hardness: f32,
    rotation: f32,
    offset: f32,
    hue: f32,
    saturation: f32,
) -> GradNdGeometry {
    let iw = width.max(1) as f32;
    let ih = height.max(1) as f32;
    let hw = iw / 2.0;
    let hh = ih / 2.0;
    let hw_inv = 1.0 / hw;
    let hh_inv = 1.0 / hh;

    // C: deg2radf(-data->rotation) — note the negation.
    let v = (-rotation).to_radians();
    let sinv = v.sin();
    let cosv = v.cos();
    let cosv_hh_inv = cosv * hh_inv;
    let filter_radie = hh.hypot(hw) / hh;
    let offset = offset / 100.0 * 2.0;

    let filter_hardness =
        (1.0 / filter_radie) / (1.0 - (0.5 + (hardness / 100.0) * 0.9 / 2.0)) * 0.5;

    // ix is 0 for the full-image preview ROI.
    let length_base = sinv * -1.0 + cosv - 1.0 + offset;
    let length_inc = sinv * hw_inv * filter_hardness;

    let (r, g, b, _) = crate::color::hsl2rgb(hue, saturation, 0.5);
    let mut color = [r, g, b, 0.0];
    if density < 0.0 {
        for c in &mut color {
            *c = 1.0 - *c;
        }
    }
    let color1 = [
        1.0 - color[0],
        1.0 - color[1],
        1.0 - color[2],
        1.0 - color[3],
    ];

    GradNdGeometry { length_base, length_inc, cosv_hh_inv, filter_hardness, color, color1 }
}

/// Graduated neutral-density filter IOP.
///
/// Pre-computed geometry scalars (from [`commit_geometry`], or the C
/// `process()` when called over FFI):
///   length_base  = sinv*(-1+ix*hw_inv) + cosv - 1 + offset
///   length_inc   = sinv * hw_inv * filter_hardness
///   cosv_hh_inv  = cosv * hh_inv
///   filter_hardness — see C source
///   iy           = roi_in->y
///
/// color / color1 each point to 4 floats (dt_aligned_pixel_t).
///
/// # Safety
/// All pointers must be valid for the duration of the call: `in_buf`/`out_buf`
/// for `width*height*4` floats each, `color`/`color1` for 4 floats each.
#[no_mangle]
pub unsafe extern "C" fn darkroom_graduatednd_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: i32,
    height: i32,
    density: f32,
    length_base: f32,
    length_inc: f32,
    cosv_hh_inv: f32,
    filter_hardness: f32,
    iy: i32,
    color: *const f32,  // 4 floats
    color1: *const f32, // 4 floats
) {
    let w = width as usize;
    let h = height as usize;
    let inp = std::slice::from_raw_parts(in_buf, w * h * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, w * h * 4);
    let c  = std::slice::from_raw_parts(color,  4);
    let c1 = std::slice::from_raw_parts(color1, 4);

    if density > 0.0 {
        for y in 0..h {
            let row_length = (length_base - (iy + y as i32) as f32 * cosv_hh_inv) * filter_hardness;
            for x in 0..w {
                let length = row_length + x as f32 * length_inc;
                let curr_density = compute_density(density, length);
                let base = (y * w + x) * 4;
                for l in 0..4 {
                    out[base + l] = inp[base + l] / (c[l] + c1[l] * curr_density);
                }
            }
        }
    } else {
        for y in 0..h {
            let row_length = (length_base - (iy + y as i32) as f32 * cosv_hh_inv) * filter_hardness;
            for x in 0..w {
                let length = row_length + x as f32 * length_inc;
                let curr_density = compute_density(-density, -length);
                let base = (y * w + x) * 4;
                for l in 0..4 {
                    out[base + l] = inp[base + l] * (c[l] + c1[l] * curr_density);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_density_is_passthrough() {
        // density=0 falls into the else branch; compute_density(0, ...) = exp2(0) = 1
        // color=[1,1,1,1], color1=[0,0,0,0] → multiply by 1 → passthrough
        let inp = [0.5f32, 0.4, 0.3, 1.0,  0.1, 0.2, 0.3, 1.0];
        let mut out = [0f32; 8];
        let color  = [1.0f32, 1.0, 1.0, 1.0];
        let color1 = [0.0f32; 4];
        unsafe {
            darkroom_graduatednd_process(
                inp.as_ptr(), out.as_mut_ptr(),
                2, 1, 0.0,
                0.0, 0.0, 0.0, 1.0, 0,
                color.as_ptr(), color1.as_ptr(),
            )
        };
        for i in 0..8 { assert!((out[i] - inp[i]).abs() < 1e-5, "idx {i}"); }
    }

    #[test]
    fn positive_density_divides() {
        // density=1, length=0 → compute_density(1, 0) = exp2(1*0.5) = sqrt(2)
        // color=[0,0,0,0], color1=[1,1,1,1] → out = in / exp2(0.5)
        let v = 0.8f32;
        let inp = [v, v, v, v];
        let mut out = [0f32; 4];
        let color  = [0.0f32; 4];
        let color1 = [1.0f32, 1.0, 1.0, 1.0];
        unsafe {
            darkroom_graduatednd_process(
                inp.as_ptr(), out.as_mut_ptr(),
                1, 1, 1.0,
                0.0, 0.0, 0.0, 1.0, 0,
                color.as_ptr(), color1.as_ptr(),
            )
        };
        let expected = v / 2.0f32.powf(0.5);
        assert!((out[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn negative_density_multiplies() {
        // density=-1, length=0: curr_density = compute_density(1, 0) = exp2(0.5)
        // color=[0,0,0,0], color1=[1,1,1,1] → out = in * exp2(0.5)
        let v = 0.4f32;
        let inp = [v, v, v, v];
        let mut out = [0f32; 4];
        let color  = [0.0f32; 4];
        let color1 = [1.0f32, 1.0, 1.0, 1.0];
        unsafe {
            darkroom_graduatednd_process(
                inp.as_ptr(), out.as_mut_ptr(),
                1, 1, -1.0,
                0.0, 0.0, 0.0, 1.0, 0,
                color.as_ptr(), color1.as_ptr(),
            )
        };
        let expected = v * 2.0f32.powf(0.5);
        assert!((out[0] - expected).abs() < 1e-5);
    }
}
