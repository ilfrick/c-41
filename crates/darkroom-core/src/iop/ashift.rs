//! Ashift IOP helpers -- perspective correction preprocessing.
//!
//! Ports all portable static helper functions from ashift.c that were
//! previously parallelised with DT_OMP_FOR.  The main process() and
//! distort_mask() loops are in interp.rs (darkroom_ashift_process /
//! darkroom_ashift_distort_mask) since they depend on the interp engine.

use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Ashift;
impl IopProcess for Ashift {
    fn name(&self) -> &'static str { "ashift" }
    fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("ashift: use C FFI path".into()))
    }
    fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("ashift: no OpenCL path".into()))
    }
}

// ── Homographic coordinate transforms ────────────────────────────────────

#[inline(always)]
fn mat3mulv(m: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
        m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
        m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
    ]
}

/// Apply homography to a flat (x,y) interleaved point array; subtract (cx, cy).
/// pts[i]   = (homograph * [x, y, 1])[0] / [2] - cx
/// pts[i+1] = (homograph * [x, y, 1])[1] / [2] - cy
/// Matches ashift.c:1026.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_transform_coords(
    pts:       *mut f32,
    n:         usize,
    homograph: *const f32,
    cx:        f32,
    cy:        f32,
) {
    let buf = std::slice::from_raw_parts_mut(pts, n * 2);
    let h: &[f32; 9] = &*(homograph as *const [f32; 9]);
    let mut i = 0;
    while i < n * 2 {
        let po = mat3mulv(h, [buf[i], buf[i+1], 1.0]);
        let w = po[2];
        buf[i]   = po[0] / w - cx;
        buf[i+1] = po[1] / w - cy;
        i += 2;
    }
}

/// Inverse homographic coordinate transform; add (cx, cy) before projection.
/// pts[i]   = (ihomograph * [x+cx, y+cy, 1])[0] / [2]
/// pts[i+1] = (ihomograph * [x+cx, y+cy, 1])[1] / [2]
/// Matches ashift.c:1064.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_backtransform_coords(
    pts:        *mut f32,
    n:          usize,
    ihomograph: *const f32,
    cx:         f32,
    cy:         f32,
) {
    let buf = std::slice::from_raw_parts_mut(pts, n * 2);
    let h: &[f32; 9] = &*(ihomograph as *const [f32; 9]);
    let mut i = 0;
    while i < n * 2 {
        let po = mat3mulv(h, [buf[i] + cx, buf[i+1] + cy, 1.0]);
        let w = po[2];
        buf[i]   = po[0] / w;
        buf[i+1] = po[1] / w;
        i += 2;
    }
}

// ── Line-detection preprocessing helpers ─────────────────────────────────

/// Convert RGBA float pixels to a double grayscale buffer.
/// out[k] = (0.3*R + 0.59*G + 0.11*B) * 256.
/// Matches ashift.c:1271.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_rgb_to_gray(
    in_buf:   *const f32,
    out_buf:  *mut f64,
    npixels:  usize,
) {
    let inp = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels);
    for k in 0..npixels {
        let r = inp[k*4]     as f64;
        let g = inp[k*4 + 1] as f64;
        let b = inp[k*4 + 2] as f64;
        out[k] = (0.3 * r + 0.59 * g + 0.11 * b) * 256.0;
    }
}

/// 3×3 Sobel edge enhancement in one direction (0=horizontal, 1=vertical).
/// Fills the interior (excluding 1-pixel border); border is filled by
/// darkroom_ashift_sobel_border.  Operates on double arrays.
/// Matches ashift.c:1302.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_sobel_1d(
    in_buf:    *const f64,
    out_buf:   *mut f64,
    width:     i32,
    height:    i32,
    direction: i32, // 0 = horizontal (Gx), 1 = vertical (Gy)
) {
    // Sobel kernels, row-major
    const HKERNEL: [f64; 9] = [1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
    const VKERNEL: [f64; 9] = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0];

    let w = width as usize;
    let h = height as usize;
    let inp  = std::slice::from_raw_parts(in_buf, w * h);
    let outp = std::slice::from_raw_parts_mut(out_buf, w * h);
    let kernel = if direction == 0 { &HKERNEL } else { &VKERNEL };

    for j in 1..h-1 {
        for i in 1..w-1 {
            let mut sum = 0.0f64;
            for jj in 0..3usize {
                let k = jj * 3;
                let l = j + jj - 1;
                for ii in 0..3usize {
                    sum += inp[l * w + (i + ii - 1)] * kernel[k + ii];
                }
            }
            outp[j * w + i] = sum;
        }
    }
}

/// Mirror-fill the 1-pixel Sobel border using nearest interior values.
/// Matches ashift.c:1324.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_sobel_border(
    buf:    *mut f64,
    width:  i32,
    height: i32,
) {
    let w = width  as usize;
    let h = height as usize;
    let b = std::slice::from_raw_parts_mut(buf, w * h);
    for j in 0..h {
        let mut i = 0;
        while i < w {
            let val = if j < 1 {
                b[(2 - j*2) * w + i]   // = b[(1-j + 1) * w + i] — but j==0 → b[1*w+i]
            } else if j >= h - 1 {
                b[(j - 1) * w + i]
            } else if i < 1 {
                b[j * w + 1]
            } else if i >= w - 1 {
                b[j * w + i - 1]
            } else {
                b[j * w + i]           // interior: no change, but we'll skip to end
            };
            b[j * w + i] = val;
            // Skip interior columns
            if i == 0 && j >= 1 && j < h - 1 {
                i = w - 1; // jump to last column
            } else {
                i += 1;
            }
        }
    }
}

/// Compute gradient magnitude: out[k] = sqrt(Gx[k]^2 + Gy[k]^2).
/// Matches ashift.c:1367.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_gradient_magnitude(
    gx:      *const f64,
    gy:      *const f64,
    out_buf: *mut f64,
    n:       usize,
) {
    let gx  = std::slice::from_raw_parts(gx,  n);
    let gy  = std::slice::from_raw_parts(gy,  n);
    let out = std::slice::from_raw_parts_mut(out_buf, n);
    for k in 0..n {
        out[k] = (gx[k]*gx[k] + gy[k]*gy[k]).sqrt();
    }
}

/// Apply gamma correction: out[c] = in[c]^0.45 for each RGB channel.
/// Alpha channel (index 3) is passed through unchanged.
/// Matches ashift.c:1441 with LSD_GAMMA = 0.45.
#[no_mangle]
pub unsafe extern "C" fn darkroom_ashift_gamma_correct(
    in_buf:  *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    const LSD_GAMMA: f32 = 0.45;
    let inp = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    for k in 0..npixels {
        out[k*4]     = inp[k*4    ].powf(LSD_GAMMA);
        out[k*4 + 1] = inp[k*4 + 1].powf(LSD_GAMMA);
        out[k*4 + 2] = inp[k*4 + 2].powf(LSD_GAMMA);
        out[k*4 + 3] = inp[k*4 + 3]; // alpha unchanged
    }
}
