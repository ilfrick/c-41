use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct Grain;

impl IopProcess for Grain {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "grain" }
}

// ----------------------------------------------------------------------------
// Stefan Gustavson's 3D simplex noise — ported from grain.c
// ----------------------------------------------------------------------------

#[rustfmt::skip]
static GRAD3: [[f64; 3]; 12] = [
    [ 1.0,  1.0,  0.0], [-1.0,  1.0,  0.0],
    [ 1.0, -1.0,  0.0], [-1.0, -1.0,  0.0],
    [ 1.0,  0.0,  1.0], [-1.0,  0.0,  1.0],
    [ 1.0,  0.0, -1.0], [-1.0,  0.0, -1.0],
    [ 0.0,  1.0,  1.0], [ 0.0, -1.0,  1.0],
    [ 0.0,  1.0, -1.0], [ 0.0, -1.0, -1.0],
];

#[rustfmt::skip]
static PERMUTATION: [usize; 256] = [
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
];

fn build_perm() -> ([usize; 512], [usize; 512]) {
    let mut perm = [0usize; 512];
    let mut perm_mod = [0usize; 512];
    for i in 0..512 {
        perm[i] = PERMUTATION[i & 255];
        perm_mod[i] = perm[i] % 12;
    }
    (perm, perm_mod)
}

#[inline(always)]
fn fastfloor(x: f64) -> i32 {
    if x > 0.0 { x as i32 } else { x as i32 - 1 }
}

#[inline(always)]
fn dot3(g: &[f64; 3], x: f64, y: f64, z: f64) -> f64 {
    g[0] * x + g[1] * y + g[2] * z
}

fn simplex_noise(xin: f64, yin: f64, zin: f64, perm: &[usize; 512], perm_mod: &[usize; 512]) -> f64 {
    let f3 = 1.0 / 3.0;
    let g3 = 1.0 / 6.0;
    let s = (xin + yin + zin) * f3;
    let i = fastfloor(xin + s);
    let j = fastfloor(yin + s);
    let k = fastfloor(zin + s);
    let t = (i + j + k) as f64 * g3;
    let x0 = xin - (i as f64 - t);
    let y0 = yin - (j as f64 - t);
    let z0 = zin - (k as f64 - t);

    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0      { (1,0,0, 1,1,0) }
        else if x0 >= z0 { (1,0,0, 1,0,1) }
        else             { (0,0,1, 1,0,1) }
    } else {
        if y0 < z0       { (0,0,1, 0,1,1) }
        else if x0 < z0  { (0,1,0, 0,1,1) }
        else             { (0,1,0, 1,1,0) }
    };

    let x1 = x0 - i1 as f64 + g3;
    let y1 = y0 - j1 as f64 + g3;
    let z1 = z0 - k1 as f64 + g3;
    let x2 = x0 - i2 as f64 + 2.0 * g3;
    let y2 = y0 - j2 as f64 + 2.0 * g3;
    let z2 = z0 - k2 as f64 + 2.0 * g3;
    let x3 = x0 - 1.0 + 3.0 * g3;
    let y3 = y0 - 1.0 + 3.0 * g3;
    let z3 = z0 - 1.0 + 3.0 * g3;

    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let kk = (k & 255) as usize;
    let gi0 = perm_mod[ii     + perm[jj     + perm[kk    ]]];
    let gi1 = perm_mod[ii+i1  + perm[jj+j1  + perm[kk+k1 ]]];
    let gi2 = perm_mod[ii+i2  + perm[jj+j2  + perm[kk+k2 ]]];
    let gi3 = perm_mod[ii+1   + perm[jj+1   + perm[kk+1  ]]];

    let corner = |gi: usize, x: f64, y: f64, z: f64| -> f64 {
        let t = 0.6 - x*x - y*y - z*z;
        if t < 0.0 { 0.0 } else { let t2 = t*t; t2*t2 * dot3(&GRAD3[gi], x, y, z) }
    };

    32.0 * (corner(gi0, x0, y0, z0)
          + corner(gi1, x1, y1, z1)
          + corner(gi2, x2, y2, z2)
          + corner(gi3, x3, y3, z3))
}

fn simplex_2d_noise(x: f64, y: f64, z: f64, perm: &[usize; 512], perm_mod: &[usize; 512]) -> f64 {
    const F: [f64; 3] = [0.4910, 0.9441, 1.7280];
    const A: [f64; 3] = [0.2340, 0.7850, 1.2150];
    let mut total = 0.0;
    for octave in 0..3 {
        total += simplex_noise(x * F[octave] / z, y * F[octave] / z, octave as f64, perm, perm_mod) * A[octave];
    }
    total
}

// ----------------------------------------------------------------------------
// Grain LUT (128×128 photographic paper response model)
// ----------------------------------------------------------------------------

const GRAIN_LUT_SIZE: usize = 128;
#[allow(dead_code)]
const GRAIN_LUT_DELTA_MAX: f32 = 2.0;
#[allow(dead_code)]
const GRAIN_LUT_DELTA_MIN: f32 = 0.0001;
#[allow(dead_code)]
const GRAIN_LUT_PAPER_GAMMA: f32 = 1.0;

#[allow(dead_code)]
fn paper_resp(exposure: f32, mb: f32, gp: f32) -> f32 {
    let delta = GRAIN_LUT_DELTA_MAX * ((mb / 100.0) * GRAIN_LUT_DELTA_MIN.ln()).exp();
    (1.0 + 2.0 * delta) / (1.0 + ((4.0 * gp * (0.5 - exposure)) / (1.0 + 2.0 * delta)).exp()) - delta
}

#[allow(dead_code)]
fn paper_resp_inverse(density: f32, mb: f32, gp: f32) -> f32 {
    let delta = GRAIN_LUT_DELTA_MAX * ((mb / 100.0) * GRAIN_LUT_DELTA_MIN.ln()).exp();
    -((1.0 + 2.0 * delta) / (density + delta) - 1.0).ln() * (1.0 + 2.0 * delta) / (4.0 * gp) + 0.5
}

#[allow(dead_code)]
fn build_grain_lut(mb: f32) -> [f32; GRAIN_LUT_SIZE * GRAIN_LUT_SIZE] {
    let mut lut = [0f32; GRAIN_LUT_SIZE * GRAIN_LUT_SIZE];
    for i in 0..GRAIN_LUT_SIZE {
        for j in 0..GRAIN_LUT_SIZE {
            let gu = i as f32 / (GRAIN_LUT_SIZE - 1) as f32 - 0.5;
            let l  = j as f32 / (GRAIN_LUT_SIZE - 1) as f32;
            lut[j * GRAIN_LUT_SIZE + i] = 100.0
                * (paper_resp(gu + paper_resp_inverse(l, mb, GRAIN_LUT_PAPER_GAMMA), mb, GRAIN_LUT_PAPER_GAMMA) - l);
        }
    }
    lut
}

fn lut_lookup_2d(grain_lut: &[f32], x: f32, y: f32) -> f32 {
    let sz = GRAIN_LUT_SIZE as f32;
    let _x = ((x + 0.5) * (sz - 1.0)).clamp(0.0, sz - 1.0);
    let _y = (y * (sz - 1.0)).clamp(0.0, sz - 1.0);
    let x0 = if _x < sz - 2.0 { _x as usize } else { GRAIN_LUT_SIZE - 2 };
    let y0 = if _y < sz - 2.0 { _y as usize } else { GRAIN_LUT_SIZE - 2 };
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let xd = _x - x0 as f32;
    let yd = _y - y0 as f32;
    let l00 = grain_lut[y0 * GRAIN_LUT_SIZE + x0];
    let l01 = grain_lut[y0 * GRAIN_LUT_SIZE + x1];
    let l10 = grain_lut[y1 * GRAIN_LUT_SIZE + x0];
    let l11 = grain_lut[y1 * GRAIN_LUT_SIZE + x1];
    let xy0 = (1.0 - yd) * l00 + l10 * yd;
    let xy1 = (1.0 - yd) * l01 + l11 * yd;
    xy0 * (1.0 - xd) + xy1 * xd
}

const GRAIN_LIGHTNESS_STRENGTH_SCALE: f32 = 0.15;

/// Grain IOP — simulate silver grain using simplex noise on the L channel.
///
/// Implements both the fast (non-filter) and downsampled (filter) paths from grain.c.
///
/// Caller pre-computes from C:
///   strength = data->strength / 100.0
///   zoom     = (1.0 + 8*data->scale/100) / 800.0
///   wd       = fminf(piece->buf_in.width, piece->buf_in.height)
///   scale    = roi_out->scale
///   hash     = _hash_string(filename) % max(roi->width*0.3, 1)
///   filter   = !fastmode && fabsf(roi_out->scale - 1.0f) > 0.01f
///   filtermul = piece->iscale / (roi_out->scale * wd)   [only used when filter != 0]
///   grain_lut = data->grain_lut (128×128 floats from commit_params)
#[no_mangle]
pub unsafe extern "C" fn darkroom_grain_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    roi_x: i32,
    roi_y: i32,
    width: i32,
    height: i32,
    strength: f32,
    zoom: f64,
    wd: f64,
    scale: f64,
    hash: i32,
    filter: i32,       // 0 = fast path; non-zero = rank-1 lattice downsampling
    filtermul: f64,
    grain_lut: *const f32, // 128×128 floats from data->grain_lut
) {
    const FIB1: f64 = 34.0;
    const FIB2: f64 = 21.0;
    const FIB1DIV2: f64 = FIB1 / FIB2;
    const FIB2INV: f64 = 1.0 / FIB2;

    let (perm, perm_mod) = build_perm();
    let w = width as usize;
    let h = height as usize;
    let inp = std::slice::from_raw_parts(in_buf, w * h * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, w * h * 4);
    let lut = std::slice::from_raw_parts(grain_lut, GRAIN_LUT_SIZE * GRAIN_LUT_SIZE);

    for j in 0..h {
        let wy = (roi_y + j as i32) as f64 / scale;
        let y = wy / wd;
        for i in 0..w {
            let wx = (roi_x + i as i32) as f64 / scale;
            let x = wx / wd;

            let noise = if filter != 0 {
                let mut n = 0.0f64;
                for l in 0..FIB2 as usize {
                    let px = l as f64 / FIB2;
                    let mut py = l as f64 * FIB1DIV2;
                    py -= py as i64 as f64; // fmod 1
                    let dx = px * filtermul;
                    let dy = py * filtermul;
                    n += FIB2INV * simplex_2d_noise(x + dx + hash as f64, y + dy, zoom, &perm, &perm_mod);
                }
                n as f32
            } else {
                simplex_2d_noise(x + hash as f64, y, zoom, &perm, &perm_mod) as f32
            };

            let base = (j * w + i) * 4;
            out[base + 0] = inp[base + 0]
                + lut_lookup_2d(lut, noise * strength * GRAIN_LIGHTNESS_STRENGTH_SCALE, inp[base + 0] / 100.0);
            out[base + 1] = inp[base + 1];
            out[base + 2] = inp[base + 2];
            out[base + 3] = inp[base + 3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_noise_bounded() {
        let (perm, perm_mod) = build_perm();
        for &(x, y, z) in &[(0.1, 0.2, 1.0), (1.0, 2.0, 3.0), (-0.5, 0.7, 2.0)] {
            let n = simplex_noise(x, y, z, &perm, &perm_mod);
            assert!(n.abs() <= 1.0 + 1e-6, "noise={n} out of [-1,1]");
        }
    }

    #[test]
    fn grain_lut_midpoint() {
        // At the center x-index and mid-lightness, grain contribution should be near 0.
        let lut = build_grain_lut(50.0);
        let mid = GRAIN_LUT_SIZE / 2;
        let v = lut[mid * GRAIN_LUT_SIZE + mid];
        assert!(v.abs() < 5.0, "mid LUT value={v} too far from 0");
    }

    #[test]
    fn grain_channels_1_2_pass_through() {
        let lut = build_grain_lut(0.0);
        let inp = [50.0f32, 10.0, -5.0, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_grain_process(
                inp.as_ptr(), out.as_mut_ptr(),
                0, 0, 1, 1,
                0.5, 0.01, 1000.0, 1.0, 0,
                0, 0.0,
                lut.as_ptr(),
            )
        };
        assert_eq!(out[1], inp[1]);
        assert_eq!(out[2], inp[2]);
        assert_eq!(out[3], inp[3]);
    }

    #[test]
    fn zero_strength_is_passthrough() {
        let lut = build_grain_lut(0.0);
        let inp = [60.0f32, 5.0, -3.0, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_grain_process(
                inp.as_ptr(), out.as_mut_ptr(),
                0, 0, 1, 1,
                0.0, 0.01, 1000.0, 1.0, 0, // strength=0
                0, 0.0,
                lut.as_ptr(),
            )
        };
        // strength=0 → noise*strength=0 → lut_lookup(0, L/100)
        // lut at x=0 (center) is ~0, so out[0] ≈ in[0]
        assert!((out[0] - inp[0]).abs() < 2.0);
    }
}
