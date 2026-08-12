use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct Vignette;

impl IopProcess for Vignette {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "vignette" }
}

// TEA (Tiny Encryption Algorithm) — matches src/common/tea.h exactly
const TEA_ROUNDS: usize = 8;
const TEA_KEY: [u32; 4] = [0xa341316c, 0xc8013ea4, 0xad90777d, 0x7e95761e];
const TEA_DELTA: u32 = 0x9e3779b9;

fn encrypt_tea(state: &mut [u32; 2]) {
    let mut v0 = state[0];
    let mut v1 = state[1];
    let mut sum: u32 = 0;
    for _ in 0..TEA_ROUNDS {
        sum = sum.wrapping_add(TEA_DELTA);
        v0 = v0.wrapping_add(((v1 << 4).wrapping_add(TEA_KEY[0])) ^ (v1.wrapping_add(sum)) ^ ((v1 >> 5).wrapping_add(TEA_KEY[1])));
        v1 = v1.wrapping_add(((v0 << 4).wrapping_add(TEA_KEY[2])) ^ (v0.wrapping_add(sum)) ^ ((v0 >> 5).wrapping_add(TEA_KEY[3])));
    }
    state[0] = v0;
    state[1] = v1;
}

fn tpdf(urandom: u32) -> f32 {
    let frandom = urandom as f32 / 0xFFFF_FFFFu32 as f32;
    if frandom < 0.5 {
        (2.0 * frandom).sqrt() - 1.0
    } else {
        1.0 - (2.0 * (1.0 - frandom)).sqrt()
    }
}

/// The pre-computed geometry `darkroom_vignette_process` takes, derived from the
/// user-facing params plus the buffer dimensions. Mirrors the block at the top
/// of vignette.c `process()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VignetteGeometry {
    pub xscale: f32,
    pub yscale: f32,
    pub roi_center_x: f32,
    pub roi_center_y: f32,
    pub dscale: f32,
    pub fscale: f32,
    pub exp1: f32,
    pub exp2: f32,
}

/// Port of the geometry derivation in vignette.c `process()` (the block above
/// the pixel loop). Kept here, beside the loop it feeds, so the two can't drift.
///
/// `width`/`height` are the buffer's dimensions; the preview runs at `scale`
/// 1.0 with the ROI origin at the image origin, so `roi_center` reduces to the
/// vignette centre in pixels.
///
/// `scale_pct` (fall-off start) and `falloff_pct` (fall-off radius) are the C
/// 0..200 sliders; `center_x`/`center_y` are the -1..1 offsets; `whratio` is
/// 0..2 (<=1 = w/h, >1 = h/w + 1); `shape` is 0..5. `autoratio` follows the
/// buffer's own aspect instead of `whratio`.
pub fn commit_geometry(
    width: usize,
    height: usize,
    scale_pct: f32,
    falloff_pct: f32,
    center_x: f32,
    center_y: f32,
    autoratio: bool,
    whratio: f32,
    shape: f32,
) -> VignetteGeometry {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;

    // vignette_center = buffer centre + the normalised offset, in pixels.
    let vignette_center_x = w / 2.0 + center_x * w / 2.0;
    let vignette_center_y = h / 2.0 + center_y * h / 2.0;

    let (xscale, yscale) = if autoratio {
        (2.0 / w, 2.0 / h)
    } else {
        // Proportional to the longest side, then skewed by the ratio. The
        // 0..1 / 1..2 split is darktable's encoding of w/h vs h/w.
        let basis = 2.0 / w.max(h);
        if whratio <= 1.0 {
            let yscale = basis;
            // whratio 0 would divide by zero; the C relies on its slider
            // minimum, so floor it here.
            (yscale / whratio.max(f32::MIN_POSITIVE), yscale)
        } else {
            let xscale = basis;
            (xscale, xscale / (2.0 - whratio).max(f32::MIN_POSITIVE))
        }
    };

    let dscale = scale_pct / 100.0;
    // A minimum falloff smooths aliasing on small buffers (C comment).
    let min_falloff = 100.0 / w.min(h);
    let fscale = falloff_pct.max(min_falloff) / 100.0;
    let shape = shape.max(0.001); // C: MAX(data->shape, 0.001f)

    VignetteGeometry {
        xscale,
        yscale,
        roi_center_x: vignette_center_x * xscale,
        roi_center_y: vignette_center_y * yscale,
        dscale,
        fscale,
        exp1: 2.0 / shape,
        exp2: shape / 2.0,
    }
}

/// Vignette IOP — radial brightness/saturation falloff with optional dithering.
///
/// Geometry scalars pre-computed by C caller (from data + roi + buf_in):
///   xscale, yscale: pixel → normalized coordinates
///   roi_center_x/y: vignette center in output pixel space, pre-scaled
///   dscale = data->scale / 100.0
///   fscale = max(data->falloff_scale, min_falloff) / 100.0
///   exp1 = 2.0 / shape,  exp2 = shape / 2.0
///   dither_amt: 0.0=off, 1/256=8bit, 1/65536=16bit
///   brightness, saturation: from data struct
///   unbound: 0=clip to [0,1], non-zero=no clip
#[no_mangle]
pub unsafe extern "C" fn darkroom_vignette_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: i32,
    height: i32,
    xscale: f32,
    yscale: f32,
    roi_center_x: f32,  // roi_center_scaled.x
    roi_center_y: f32,  // roi_center_scaled.y
    dscale: f32,
    fscale: f32,
    exp1: f32,
    exp2: f32,
    dither_amt: f32,
    brightness: f32,
    saturation: f32,
    unbound: i32,
) {
    let w = width as usize;
    let h = height as usize;
    let inp = std::slice::from_raw_parts(in_buf, w * h * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, w * h * 4);

    for j in 0..h {
        // Seed TEA state per row, matching C: tea_state[0] = j * roi_out->height
        let mut tea_state = [j as u32 * height as u32, 0u32];

        for i in 0..w {
            let base = (j * w + i) * 4;
            let inp_px = &inp[base..base + 4];

            // Distance from vignette center in normalized coords
            let pvx = (i as f32 * xscale - roi_center_x).abs();
            let pvy = (j as f32 * yscale - roi_center_y).abs();
            let cplen = (pvx.powf(exp1) + pvy.powf(exp1)).powf(exp2);

            let mut weight = 0.0f32;
            let mut dith = 0.0f32;

            if cplen >= dscale {
                weight = ((cplen - dscale) / fscale).clamp(0.0, 1.0);
                if weight > 0.0 && weight < 1.0 && dither_amt != 0.0 {
                    weight = 0.5 - (std::f32::consts::PI * weight).cos() / 2.0;
                    encrypt_tea(&mut tea_state);
                    dith = dither_amt * tpdf(tea_state[0]);
                }
            }

            let mut col = [inp_px[0], inp_px[1], inp_px[2], inp_px[3]];

            if weight > 0.0 {
                if brightness < 0.0 {
                    let falloff = 1.0 + weight * brightness;
                    col[0] = col[0] * falloff + dith;
                    col[1] = col[1] * falloff + dith;
                    col[2] = col[2] * falloff + dith;
                } else {
                    let falloff = weight * brightness;
                    col[0] = col[0] + falloff + dith;
                    col[1] = col[1] + falloff + dith;
                    col[2] = col[2] + falloff + dith;
                }

                if unbound == 0 {
                    col[0] = col[0].clamp(0.0, 1.0);
                    col[1] = col[1].clamp(0.0, 1.0);
                    col[2] = col[2].clamp(0.0, 1.0);
                }

                let mv = (col[0] + col[1] + col[2]) / 3.0;
                let wss = weight * saturation;
                col[0] = col[0] - (mv - col[0]) * wss;
                col[1] = col[1] - (mv - col[1]) * wss;
                col[2] = col[2] - (mv - col[2]) * wss;

                if unbound == 0 {
                    col[0] = col[0].clamp(0.0, 1.0);
                    col[1] = col[1].clamp(0.0, 1.0);
                    col[2] = col[2].clamp(0.0, 1.0);
                }
            }

            let o = &mut out[base..base + 4];
            o[0] = col[0];
            o[1] = col[1];
            o[2] = col[2];
            o[3] = col[3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tea_encrypt_deterministic() {
        let mut s = [42u32, 0u32];
        encrypt_tea(&mut s);
        let first = s[0];
        let mut s2 = [42u32, 0u32];
        encrypt_tea(&mut s2);
        assert_eq!(s2[0], first);
    }

    #[test]
    fn tpdf_symmetric_at_half() {
        // tpdf(0.5 * 0xFFFFFFFF) ≈ 0
        let mid = 0x7FFF_FFFFu32;
        let v = tpdf(mid);
        assert!(v.abs() < 0.01);
    }

    #[test]
    fn zero_weight_is_passthrough() {
        // Center pixel with dscale=1.0 → cplen<dscale → weight=0 → passthrough
        let inp = [0.5f32, 0.3, 0.7, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_vignette_process(
                inp.as_ptr(), out.as_mut_ptr(),
                1, 1,
                1.0, 1.0,   // xscale, yscale
                0.5, 0.5,   // center at (0.5, 0.5) → pixel (0,0) is at distance (0.5,0.5)
                2.0,        // dscale — cplen will be less than this for small images
                0.1, 2.0, 1.0,
                0.0, -0.5, 0.5, 0,
            )
        };
        // pixel is inside the vignette inner circle
        assert!((out[0] - inp[0]).abs() < 0.01);
    }

    #[test]
    fn full_weight_darkens() {
        // Single pixel at distance far from center: xscale=1, pixel 0 → pvx=0.5
        // cplen = 0.5^2 + 0.5^2 = 0.5, dscale=0, fscale=1 → weight=0.5 (after cosine)
        // brightness = -1 → darkens
        let inp = [0.8f32, 0.8, 0.8, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_vignette_process(
                inp.as_ptr(), out.as_mut_ptr(),
                1, 1,
                1.0, 1.0,
                0.5, 0.5,
                0.0, 1.0, 2.0, 1.0, // dscale=0 → always in falloff zone
                0.0, -1.0, 0.0, 0,
            )
        };
        // Output should be darker than input
        assert!(out[0] < inp[0]);
    }
}
