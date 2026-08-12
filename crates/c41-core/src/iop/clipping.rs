//! Clipping IOP -- 4 DT_OMP_FOR loops: 2 coordinate transforms + 2 image warps.

use crate::{params::IopParams, roi::RoiIn, Result, interp};
use super::{ClBuffer, IopProcess};

pub struct Clipping;
impl IopProcess for Clipping {
    fn name(&self) -> &'static str { "clipping" }
    fn process(&self, _: &[f32], _: &mut [f32], _: &IopParams, _: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("clipping: use C FFI path".into()))
    }
    fn process_cl(&self, _: &mut ClBuffer, _: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("clipping: no OpenCL path".into()))
    }
}

// ── Internal transform primitives ────────────────────────────────────────

/// Projective (homographic) forward: maps input point to keystone-corrected coords.
#[inline(always)]
fn keystone_fwd(i: [f32; 2], k: &[f32; 4], a:f32,b:f32,d:f32,e:f32,g:f32,h:f32, kxa:f32,kya:f32) -> [f32; 2] {
    let xx = i[0] - kxa;
    let yy = i[1] - kya;
    let div = g*xx + h*yy + 1.0;
    [(a*xx + b*yy)/div + k[0], (d*xx + e*yy)/div + k[1]]
}

/// Projective inverse: maps keystone-corrected coords back to input space.
#[inline(always)]
fn keystone_inv(i: [f32; 2], k: &[f32; 4], a:f32,b:f32,d:f32,e:f32,g:f32,h:f32, kxa:f32,kya:f32) -> [f32; 2] {
    let xx = i[0] - k[0];
    let yy = i[1] - k[1];
    let div = (d*xx - a*yy)*h + (b*yy - e*xx)*g + a*e - b*d;
    [(e*xx - b*yy)/div + kxa, -(d*xx - a*yy)/div + kya]
}

/// Forward affine + tangential corrections: o = m*x, then taper.
#[inline(always)]
fn affine_fwd(x: [f32; 2], m: &[f32; 4], k_h: f32, k_v: f32) -> [f32; 2] {
    let mut o = [m[0]*x[0]+m[1]*x[1], m[2]*x[0]+m[3]*x[1]];
    o[1] *= 1.0 + o[0] * k_h;
    o[0] *= 1.0 + o[1] * k_v;
    o
}

/// Inverse affine + tangential: undo taper then m*x.
#[inline(always)]
fn affine_inv(x: [f32; 2], m: &[f32; 4], k_h: f32, k_v: f32) -> [f32; 2] {
    let mut p = x;
    p[1] /= 1.0 + p[0] * k_h;
    p[0] /= 1.0 + p[1] * k_v;
    [m[0]*p[0]+m[1]*p[1], m[2]*p[0]+m[3]*p[1]]
}

// ── Coordinate-pair batch transforms ─────────────────────────────────────

/// Forward coordinate transform for each (x,y) pair.
/// Matches distort_transform DT_OMP_FOR at clipping.c:503.
#[no_mangle]
pub unsafe extern "C" fn darkroom_clipping_distort_transform(
    points: *mut f32, n: usize,
    k_apply: i32, k_space: *const f32,
    ma:f32,mb:f32,md:f32,me:f32,mg:f32,mh:f32, kxa:f32,kya:f32,
    tx:f32, ty:f32,
    inv_m: *const f32,  // 4 floats
    k_h:f32, k_v:f32, flip:i32,
    enlarge_x:f32, enlarge_y:f32, cix:f32, ciy:f32,
    factor:f32,
) {
    let pts = std::slice::from_raw_parts_mut(points, n*2);
    let k: &[f32; 4] = &*(k_space as *const [f32; 4]);
    let im: &[f32; 4] = &*(inv_m   as *const [f32; 4]);
    let txf = tx/factor; let tyf = ty/factor;
    for i in (0..n*2).step_by(2) {
        let mut pi = [pts[i], pts[i+1]];
        if k_apply != 0 { pi = keystone_fwd(pi, k, ma,mb,md,me,mg,mh, kxa,kya); }
        pi[0] -= txf; pi[1] -= tyf;
        let mut po = affine_fwd(pi, im, k_h, k_v);
        if flip != 0 { po[1] += txf; po[0] += tyf; }
        else         { po[0] += txf; po[1] += tyf; }
        pts[i]   = po[0] - (cix - enlarge_x)/factor;
        pts[i+1] = po[1] - (ciy - enlarge_y)/factor;
    }
}

/// Inverse coordinate transform for each (x,y) pair.
/// Matches distort_backtransform DT_OMP_FOR_SIMD at clipping.c:571.
#[no_mangle]
pub unsafe extern "C" fn darkroom_clipping_distort_backtransform(
    points: *mut f32, n: usize,
    k_apply: i32, k_space: *const f32,
    ma:f32,mb:f32,md:f32,me:f32,mg:f32,mh:f32, kxa:f32,kya:f32,
    tx:f32, ty:f32,
    m: *const f32,
    k_h:f32, k_v:f32, flip:i32,
    enlarge_x:f32, enlarge_y:f32, cix:f32, ciy:f32,
    factor:f32,
) {
    let pts = std::slice::from_raw_parts_mut(points, n*2);
    let k: &[f32; 4] = &*(k_space as *const [f32; 4]);
    let mref: &[f32; 4] = &*(m as *const [f32; 4]);
    let txf = tx/factor; let tyf = ty/factor;
    let exi = enlarge_x; let eyi = enlarge_y;
    for i in (0..n*2).step_by(2) {
        let mut pi = [-(exi-cix)/factor + pts[i], -(eyi-ciy)/factor + pts[i+1]];
        if flip != 0 { pi[1] -= txf; pi[0] -= tyf; }
        else         { pi[0] -= txf; pi[1] -= tyf; }
        let mut po = affine_inv(pi, mref, k_h, k_v);
        po[0] += txf; po[1] += tyf;
        if k_apply != 0 { po = keystone_inv(po, k, ma,mb,md,me,mg,mh, kxa,kya); }
        pts[i] = po[0]; pts[i+1] = po[1];
    }
}

// ── Per-pixel image warps ─────────────────────────────────────────────────

/// Common backtransform pipeline for pixel (i,j) in output space to input coords.
#[inline(always)]
fn pixel_backtransform(
    i: i32, j: i32,
    roi_out_x:f32, roi_out_y:f32, roi_out_scale:f32,
    roi_in_x:f32,  roi_in_y:f32,  roi_in_scale:f32,
    tx:f32, ty:f32,
    k_apply:i32, k_space: &[f32;4],
    ma:f32,mb:f32,md:f32,me:f32,mg:f32,mh:f32, kxa:f32,kya:f32,
    m: &[f32;4], k_h:f32, k_v:f32, flip:i32,
    enlarge_x:f32, enlarge_y:f32, cix:f32, ciy:f32,
) -> [f32; 2] {
    let mut pi = [roi_out_x - roi_out_scale*enlarge_x + roi_out_scale*cix + i as f32 + 0.5,
                  roi_out_y - roi_out_scale*enlarge_y + roi_out_scale*ciy + j as f32 + 0.5];
    if flip != 0 { pi[1] -= tx*roi_out_scale; pi[0] -= ty*roi_out_scale; }
    else         { pi[0] -= tx*roi_out_scale; pi[1] -= ty*roi_out_scale; }
    pi[0] /= roi_out_scale; pi[1] /= roi_out_scale;
    let mut po = affine_inv(pi, m, k_h, k_v);
    po[0] *= roi_in_scale; po[1] *= roi_in_scale;
    po[0] += tx*roi_in_scale; po[1] += ty*roi_in_scale;
    if k_apply != 0 { po = keystone_inv(po, k_space, ma,mb,md,me,mg,mh, kxa,kya); }
    po[0] -= roi_in_x + 0.5; po[1] -= roi_in_y + 0.5;
    po
}

/// 1-channel mask warp (crop, rotate, keystone + interpolation).
/// Matches distort_mask DT_OMP_FOR at clipping.c:640.
#[no_mangle]
pub unsafe extern "C" fn darkroom_clipping_distort_mask(
    in_buf: *const f32, out_buf: *mut f32,
    roi_out_x:f32, roi_out_y:f32, roi_out_scale:f32,
    roi_out_w:i32, roi_out_h:i32,
    roi_in_x:f32,  roi_in_y:f32,  roi_in_scale:f32,
    roi_in_w:i32,  roi_in_h:i32,
    k_apply:i32, k_space: *const f32,
    ma:f32,mb:f32,md:f32,me:f32,mg:f32,mh:f32, kxa:f32,kya:f32,
    tx:f32, ty:f32, m: *const f32, k_h:f32, k_v:f32, flip:i32,
    enlarge_x:f32, enlarge_y:f32, cix:f32, ciy:f32,
    interp_type: u32,
) {
    let inp = std::slice::from_raw_parts(in_buf, (roi_in_w*roi_in_h) as usize);
    let out = std::slice::from_raw_parts_mut(out_buf, (roi_out_w*roi_out_h) as usize);
    let k: &[f32;4]  = &*(k_space as *const [f32;4]);
    let mref: &[f32;4] = &*(m as *const [f32;4]);
    for j in 0..roi_out_h {
        for i in 0..roi_out_w {
            let po = pixel_backtransform(i, j, roi_out_x, roi_out_y, roi_out_scale,
                roi_in_x, roi_in_y, roi_in_scale, tx, ty,
                k_apply, k, ma,mb,md,me,mg,mh, kxa,kya,
                mref, k_h, k_v, flip, enlarge_x, enlarge_y, cix, ciy);
            let v = interp::compute_sample_1c(inp, po[0], po[1], roi_in_w, roi_in_h, roi_in_w, interp_type);
            out[(j*roi_out_w + i) as usize] = v.clamp(0.0, 1.0);
        }
    }
}

/// 4-channel RGBA image warp (crop, rotate, keystone + interpolation).
/// Matches process DT_OMP_FOR at clipping.c:1028.
#[no_mangle]
pub unsafe extern "C" fn darkroom_clipping_process(
    in_buf: *const f32, out_buf: *mut f32,
    roi_out_x:f32, roi_out_y:f32, roi_out_scale:f32,
    roi_out_w:i32, roi_out_h:i32,
    roi_in_x:f32,  roi_in_y:f32,  roi_in_scale:f32,
    roi_in_w:i32,  roi_in_h:i32,
    k_apply:i32, k_space: *const f32,
    ma:f32,mb:f32,md:f32,me:f32,mg:f32,mh:f32, kxa:f32,kya:f32,
    tx:f32, ty:f32, m: *const f32, k_h:f32, k_v:f32, flip:i32,
    enlarge_x:f32, enlarge_y:f32, cix:f32, ciy:f32,
    ch: i32, interp_type: u32,
) {
    let ch_w = roi_in_w * ch;
    let inp  = std::slice::from_raw_parts(in_buf, (roi_in_w*roi_in_h*ch) as usize);
    let out  = std::slice::from_raw_parts_mut(out_buf, (roi_out_w*roi_out_h*ch) as usize);
    let k:    &[f32;4] = &*(k_space as *const [f32;4]);
    let mref: &[f32;4] = &*(m as *const [f32;4]);
    for j in 0..roi_out_h {
        for i in 0..roi_out_w {
            let po = pixel_backtransform(i, j, roi_out_x, roi_out_y, roi_out_scale,
                roi_in_x, roi_in_y, roi_in_scale, tx, ty,
                k_apply, k, ma,mb,md,me,mg,mh, kxa,kya,
                mref, k_h, k_v, flip, enlarge_x, enlarge_y, cix, ciy);
            let px = interp::compute_pixel4c(inp, po[0], po[1], roi_in_w, roi_in_h, ch_w, interp_type);
            let base = (j*roi_out_w + i) as usize * ch as usize;
            for c in 0..ch as usize { out[base+c] = px[c]; }
        }
    }
}
