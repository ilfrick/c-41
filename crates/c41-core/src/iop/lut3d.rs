use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Lut3d;

impl IopProcess for Lut3d {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "lut3d" }
}

// ── shared helpers ────────────────────────────────────────────────────────────

/// Quantize one float coordinate to a grid index, clamped to [0, level-2].
#[inline(always)]
fn grid_index(v: f32, flevel_1: f32, max: usize) -> (usize, f32) {
    let scaled = v.clamp(0.0, 1.0) * flevel_1;
    let i = (scaled as usize).min(max);
    (i, scaled - i as f32)
}

// ── trilinear ─────────────────────────────────────────────────────────────────

/// Trilinear 3D-LUT interpolation.
///
/// Replaces DT_OMP_FOR in _correct_pixel_trilinear() in lut3d.c.
/// clut: 3 × level³ floats (R,G,B per grid point, no alpha).
/// Output alpha is always 0.
#[no_mangle]
pub unsafe extern "C" fn darkroom_lut3d_trilinear(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    clut: *const f32,
    level: u16,
) {
    let lev = level as usize;
    let lev2 = lev * lev;
    let cl = std::slice::from_raw_parts(clut, 3 * lev * lev * lev);
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    let flevel_1 = (lev - 1) as f32;
    let max_i = lev - 2;
    let s1 = 3 * lev;    // level1_stride
    let s2 = 3 * lev2;   // level2_stride
    let s12 = s1 + s2;   // level12_stride

    for k in 0..npixels {
        let b = k * 4;
        let (ri, rd) = grid_index(input[b],     flevel_1, max_i);
        let (gi, gd) = grid_index(input[b + 1], flevel_1, max_i);
        let (bi, bd) = grid_index(input[b + 2], flevel_1, max_i);

        let i = (ri + lev * gi + lev2 * bi) * 3;
        let omr = 1.0 - rd;
        let omg = 1.0 - gd;

        let mut tmp1 = [0.0f32; 3];
        let mut tmp2 = [0.0f32; 3];
        let mut tmp3 = [0.0f32; 3];

        for c in 0..3 {
            tmp1[c] = cl[i+c] * omr + cl[i+3+c] * rd;
            tmp2[c] = cl[i+s1+c] * omr + cl[i+s1+3+c] * rd;
            tmp3[c] = tmp1[c] * omg + tmp2[c] * gd;
            tmp1[c] = cl[i+s2+c] * omr + cl[i+s2+3+c] * rd;
            tmp2[c] = cl[i+s12+c] * omr + cl[i+s12+3+c] * rd;
            tmp1[c] = tmp1[c] * omg + tmp2[c] * gd;
            output[b + c] = tmp3[c] * (1.0 - bd) + tmp1[c] * bd;
        }
        output[b + 3] = 0.0;
    }
}

// ── tetrahedral ───────────────────────────────────────────────────────────────

/// Tetrahedral 3D-LUT interpolation (Sakamoto method).
///
/// Replaces DT_OMP_FOR in _correct_pixel_tetrahedral() in lut3d.c.
/// Output alpha is always 0.
#[no_mangle]
pub unsafe extern "C" fn darkroom_lut3d_tetrahedral(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    clut: *const f32,
    level: u16,
) {
    let lev = level as usize;
    let lev2 = lev * lev;
    let cl = std::slice::from_raw_parts(clut, 3 * lev * lev * lev);
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    let flevel_1 = (lev - 1) as f32;
    let max_i = lev - 2;
    let s1 = 3 * lev;
    let s2 = 3 * lev2;
    let s12 = s1 + s2;

    for k in 0..npixels {
        let b = k * 4;
        let (ri, rd) = grid_index(input[b],     flevel_1, max_i);
        let (gi, gd) = grid_index(input[b + 1], flevel_1, max_i);
        let (bi, bd) = grid_index(input[b + 2], flevel_1, max_i);

        let color = ri + gi * lev + bi * lev2;
        let i000 = color * 3;
        let i100 = i000 + 3;
        let i010 = i000 + s1;
        let i110 = i010 + 3;
        let i001 = i000 + s2;
        let i101 = i001 + 3;
        let i011 = i000 + s12;
        let i111 = i011 + 3;

        for c in 0..3 {
            output[b + c] = if rd > gd {
                if gd > bd {
                    (1.0-rd)*cl[i000+c] + (rd-gd)*cl[i100+c]
                        + (gd-bd)*cl[i110+c] + bd*cl[i111+c]
                } else if rd > bd {
                    (1.0-rd)*cl[i000+c] + (rd-bd)*cl[i100+c]
                        + (bd-gd)*cl[i101+c] + gd*cl[i111+c]
                } else {
                    (1.0-bd)*cl[i000+c] + (bd-rd)*cl[i001+c]
                        + (rd-gd)*cl[i101+c] + gd*cl[i111+c]
                }
            } else {
                if bd > gd {
                    (1.0-bd)*cl[i000+c] + (bd-gd)*cl[i001+c]
                        + (gd-rd)*cl[i011+c] + rd*cl[i111+c]
                } else if bd > rd {
                    (1.0-gd)*cl[i000+c] + (gd-bd)*cl[i010+c]
                        + (bd-rd)*cl[i011+c] + rd*cl[i111+c]
                } else {
                    (1.0-gd)*cl[i000+c] + (gd-rd)*cl[i010+c]
                        + (rd-bd)*cl[i110+c] + bd*cl[i111+c]
                }
            };
        }
        output[b + 3] = 0.0;
    }
}

// ── pyramid ───────────────────────────────────────────────────────────────────

/// Pyramid 3D-LUT interpolation.
///
/// Replaces DT_OMP_FOR in _correct_pixel_pyramid() in lut3d.c.
/// Output alpha is always 0.
#[no_mangle]
pub unsafe extern "C" fn darkroom_lut3d_pyramid(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    clut: *const f32,
    level: u16,
) {
    let lev = level as usize;
    let lev2 = lev * lev;
    let cl = std::slice::from_raw_parts(clut, 3 * lev * lev * lev);
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    let flevel_1 = (lev - 1) as f32;
    let max_i = lev - 2;
    let s1 = 3 * lev;
    let s2 = 3 * lev2;
    let s12 = s1 + s2;

    for k in 0..npixels {
        let b = k * 4;
        let (ri, rd) = grid_index(input[b],     flevel_1, max_i);
        let (gi, gd) = grid_index(input[b + 1], flevel_1, max_i);
        let (bi, bd) = grid_index(input[b + 2], flevel_1, max_i);

        let color = ri + gi * lev + bi * lev2;
        let i000 = color * 3;
        let i100 = i000 + 3;
        let i010 = i000 + s1;
        let i110 = i010 + 3;
        let i001 = i000 + s2;
        let i101 = i001 + 3;
        let i011 = i000 + s12;
        let i111 = i011 + 3;

        for c in 0..3 {
            output[b + c] = if gd > rd && bd > rd {
                cl[i000+c] + (cl[i111+c]-cl[i011+c])*rd
                    + (cl[i010+c]-cl[i000+c])*gd + (cl[i001+c]-cl[i000+c])*bd
                    + (cl[i011+c]-cl[i001+c]-cl[i010+c]+cl[i000+c])*gd*bd
            } else if rd > gd && bd > gd {
                cl[i000+c] + (cl[i100+c]-cl[i000+c])*rd
                    + (cl[i111+c]-cl[i101+c])*gd + (cl[i001+c]-cl[i000+c])*bd
                    + (cl[i101+c]-cl[i001+c]-cl[i100+c]+cl[i000+c])*rd*bd
            } else {
                cl[i000+c] + (cl[i100+c]-cl[i000+c])*rd
                    + (cl[i010+c]-cl[i000+c])*gd + (cl[i111+c]-cl[i110+c])*bd
                    + (cl[i110+c]-cl[i100+c]-cl[i010+c]+cl[i000+c])*rd*gd
            };
        }
        output[b + 3] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a 2×2×2 identity-like LUT: clut[r,g,b] = (r,g,b) normalized
    fn identity_clut_2() -> Vec<f32> {
        // level=2: grid points at 0 and 1
        // clut[b*4 + g*2 + r] for each (r,g,b) ∈ {0,1}^3, 3 floats per entry
        let mut lut = vec![0.0f32; 3 * 8];
        for b in 0..2usize {
            for g in 0..2usize {
                for r in 0..2usize {
                    let idx = (r + 2*g + 4*b) * 3;
                    lut[idx]   = r as f32;
                    lut[idx+1] = g as f32;
                    lut[idx+2] = b as f32;
                }
            }
        }
        lut
    }

    #[test]
    fn trilinear_identity_passthrough() {
        let lut = identity_clut_2();
        let input = vec![0.5f32, 0.25, 0.75, 1.0];
        let mut out = vec![0.0f32; 4];
        unsafe {
            darkroom_lut3d_trilinear(input.as_ptr(), out.as_mut_ptr(), 1, lut.as_ptr(), 2);
        }
        // identity 2×2×2 LUT maps [r,g,b] → [r,g,b]
        assert!((out[0] - 0.5).abs()  < 1e-5, "R={}", out[0]);
        assert!((out[1] - 0.25).abs() < 1e-5, "G={}", out[1]);
        assert!((out[2] - 0.75).abs() < 1e-5, "B={}", out[2]);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn tetrahedral_identity_passthrough() {
        let lut = identity_clut_2();
        let input = vec![0.3f32, 0.6, 0.9, 1.0];
        let mut out = vec![0.0f32; 4];
        unsafe {
            darkroom_lut3d_tetrahedral(input.as_ptr(), out.as_mut_ptr(), 1, lut.as_ptr(), 2);
        }
        assert!((out[0] - 0.3).abs() < 1e-5, "R={}", out[0]);
        assert!((out[1] - 0.6).abs() < 1e-5, "G={}", out[1]);
        assert!((out[2] - 0.9).abs() < 1e-5, "B={}", out[2]);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn pyramid_identity_passthrough() {
        let lut = identity_clut_2();
        let input = vec![0.2f32, 0.8, 0.4, 1.0];
        let mut out = vec![0.0f32; 4];
        unsafe {
            darkroom_lut3d_pyramid(input.as_ptr(), out.as_mut_ptr(), 1, lut.as_ptr(), 2);
        }
        assert!((out[0] - 0.2).abs() < 1e-5, "R={}", out[0]);
        assert!((out[1] - 0.8).abs() < 1e-5, "G={}", out[1]);
        assert!((out[2] - 0.4).abs() < 1e-5, "B={}", out[2]);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn trilinear_corner_black() {
        let lut = identity_clut_2();
        let input = vec![0.0f32, 0.0, 0.0, 1.0];
        let mut out = vec![1.0f32; 4];
        unsafe {
            darkroom_lut3d_trilinear(input.as_ptr(), out.as_mut_ptr(), 1, lut.as_ptr(), 2);
        }
        assert!((out[0]).abs() < 1e-5);
        assert!((out[1]).abs() < 1e-5);
        assert!((out[2]).abs() < 1e-5);
    }

    #[test]
    fn tetrahedral_corner_white() {
        let lut = identity_clut_2();
        let input = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        unsafe {
            darkroom_lut3d_tetrahedral(input.as_ptr(), out.as_mut_ptr(), 1, lut.as_ptr(), 2);
        }
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        assert!((out[2] - 1.0).abs() < 1e-5);
    }
}
