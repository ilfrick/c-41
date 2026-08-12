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

/// Graduated neutral-density filter IOP.
///
/// Pre-computed geometry scalars (computed by C process() from roi/piece):
///   length_base  = sinv*(-1+ix*hw_inv) + cosv - 1 + offset
///   length_inc   = sinv * hw_inv * filter_hardness
///   cosv_hh_inv  = cosv * hh_inv
///   filter_hardness — see C source
///   iy           = roi_in->y
///
/// color / color1 each point to 4 floats (dt_aligned_pixel_t).
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
