use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Diffuse;

impl IopProcess for Diffuse {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "diffuse" }
}

// ── heat_PDE_diffusion: anisotropic heat-transfer diffusion kernel ──

/// Fast exp approximation (bit-manipulation), a faithful port of `dt_vector_exp`
/// (math.h) — NOT libm `expf`. Designed for `x ∈ [-100, 0]`; diverges badly for
/// `x > 0`, but the caller only feeds it `-magnitude·anisotropy ≤ 0`.
#[inline(always)]
fn vector_exp(x: [f32; 4]) -> [f32; 4] {
    const I1: i32 = 0x3f80_0000;
    const I2: i32 = 0x402d_f854;
    let mut r = [0.0f32; 4];
    for c in 0..4 {
        // k0 = i1 + (int)(x·(i2-i1)); reinterpret the int bits as f32, floor at 0.
        let k0 = I1.wrapping_add((x[c] * (I2 - I1) as f32) as i32);
        r[c] = f32::from_bits(if k0 > 0 { k0 as u32 } else { 0 });
    }
    r
}

/// Centred finite-difference gradient in a 3×3 stencil (`find_gradients`):
/// `xy[0]` (vertical) = `(p[7]-p[1])/2`, `xy[1]` (horizontal) = `(p[5]-p[3])/2`.
#[inline(always)]
fn find_gradients(pixels: &[[f32; 4]; 9]) -> [[f32; 4]; 2] {
    let mut xy = [[0.0f32; 4]; 2];
    for c in 0..4 {
        xy[0][c] = (pixels[7][c] - pixels[1][c]) / 2.0;
        xy[1][c] = (pixels[5][c] - pixels[3][c]) / 2.0;
    }
    xy
}

/// Rotation-invariant isotropic Laplacian kernel (`isotrope_laplacian`).
#[inline(always)]
fn isotrope_laplacian() -> [[f32; 4]; 9] {
    const W: [f32; 9] = [0.25, 0.5, 0.25, 0.5, -3.0, 0.5, 0.25, 0.5, 0.25];
    let mut k = [[0.0f32; 4]; 9];
    for (kk, &w) in k.iter_mut().zip(W.iter()) {
        *kk = [w; 4];
    }
    k
}

/// 3×3 anisotropic Laplacian kernel from the 2×2 rotation matrix `a`
/// (`build_matrix`).
#[inline(always)]
fn build_matrix(a: &[[[f32; 4]; 2]; 2]) -> [[f32; 4]; 9] {
    let mut kernel = [[0.0f32; 4]; 9];
    for c in 0..4 {
        let b11 = a[0][1][c] / 2.0;
        let b13 = -b11;
        let b22 = -2.0 * (a[0][0][c] + a[1][1][c]);
        kernel[0][c] = b11;
        kernel[1][c] = a[1][1][c];
        kernel[2][c] = b13;
        kernel[3][c] = a[0][0][c];
        kernel[4][c] = b22;
        kernel[5][c] = a[0][0][c];
        kernel[6][c] = b13;
        kernel[7][c] = a[1][1][c];
        kernel[8][c] = b11;
    }
    kernel
}

/// Build the anisotropic convolution kernel for one diffusion term, dispatching
/// on the isotropy mode (`compute_kernel`; 0 = isotrope, 1 = isophote, 2 = gradient).
fn compute_kernel(
    c2: [f32; 4],
    cos_sin: [f32; 4],
    cos2: [f32; 4],
    sin2: [f32; 4],
    isotropy_type: i32,
) -> [[f32; 4]; 9] {
    match isotropy_type {
        // ISOPHOTE: dampen the gradient direction
        1 => {
            let mut a = [[[0.0f32; 4]; 2]; 2];
            for c in 0..4 {
                a[0][0][c] = cos2[c] + c2[c] * sin2[c];
                a[1][1][c] = c2[c] * cos2[c] + sin2[c];
                let off = (c2[c] - 1.0) * cos_sin[c];
                a[0][1][c] = off;
                a[1][0][c] = off;
            }
            build_matrix(&a)
        }
        // GRADIENT: dampen the isophote direction
        2 => {
            let mut a = [[[0.0f32; 4]; 2]; 2];
            for c in 0..4 {
                a[0][0][c] = c2[c] * cos2[c] + sin2[c];
                a[1][1][c] = cos2[c] + c2[c] * sin2[c];
                let off = (1.0 - c2[c]) * cos_sin[c];
                a[0][1][c] = off;
                a[1][0][c] = off;
            }
            build_matrix(&a)
        }
        // ISOTROPE (0) and default
        _ => isotrope_laplacian(),
    }
}

/// Anisotropic heat-transfer diffusion over an à-trous wavelet HF/LF layer pair —
/// a faithful port of `heat_PDE_diffusion()` (diffuse.c, the single remaining
/// DT_OMP_FOR). Writes `output = clamp(HF·strength + Σ_k derivativesₖ·ABCDₖ /
/// variance + LF, ≥0)` per pixel; where the mask is 0 it copies `HF+LF`.
///
/// `high_freq`/`low_freq`/`output`: RGBA, `width*height*4` floats (distinct
/// buffers). `mask`: `width*height` bytes (used iff `has_mask != 0`).
/// `anisotropy`/`abcd`: 4 floats; `isotropy_type`: 4 ints. The C interleaves the
/// row visiting order for cache locality only — each output row reads only the
/// unchanging HF/LF, so this port iterates rows naturally (result-identical).
///
/// # Safety
/// All pointers valid for the stated lengths.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_diffuse_heat_pde(
    high_freq: *const f32,
    low_freq: *const f32,
    mask: *const u8,
    has_mask: i32,
    output: *mut f32,
    width: usize,
    height: usize,
    anisotropy: *const f32,
    isotropy_type: *const i32,
    regularization: f32,
    variance_threshold: f32,
    current_radius_square: f32,
    mult: i32,
    abcd: *const f32,
    strength: f32,
) {
    if width == 0 || height == 0 {
        return;
    }
    let n = width * height;
    let hf = std::slice::from_raw_parts(high_freq, n * 4);
    let lf = std::slice::from_raw_parts(low_freq, n * 4);
    let out = std::slice::from_raw_parts_mut(output, n * 4);
    let aniso = std::slice::from_raw_parts(anisotropy, 4);
    let iso = std::slice::from_raw_parts(isotropy_type, 4);
    let abcd = std::slice::from_raw_parts(abcd, 4);
    let has_mask = has_mask != 0;
    let mask = if has_mask { Some(std::slice::from_raw_parts(mask, n)) } else { None };

    let regularization_factor = regularization * current_radius_square / 9.0;
    let hh = height as i32 - 1;
    let ww = width as i32 - 1;

    for i in 0..height {
        // 'above'/'below' rows, clamped to the image (H = 1, so offset = mult)
        let above = ((i as i32 - mult).max(0) as usize) * width;
        let below = ((i as i32 + mult).min(hh) as usize) * width;
        let i_neigh = [above, i * width, below];

        for j in 0..width {
            let idx = i * width + j;
            let index = idx * 4;
            let opacity = mask.map_or(1u8, |m| m[idx]);

            if opacity != 0 {
                let jl = (j as i32 - mult).max(0) as usize;
                let jr = (j as i32 + mult).min(ww) as usize;
                let j_neigh = [jl, j, jr];

                // gather the 3×3 non-local HF/LF neighbourhood contiguously
                let mut nhf = [[0.0f32; 4]; 9];
                let mut nlf = [[0.0f32; 4]; 9];
                for ii in 0..3 {
                    for jj in 0..3 {
                        let nb = 4 * (i_neigh[ii] + j_neigh[jj]);
                        for c in 0..4 {
                            nhf[3 * ii + jj][c] = hf[nb + c];
                            nlf[3 * ii + jj][c] = lf[nb + c];
                        }
                    }
                }

                let mut gradient = find_gradients(&nlf);
                let mut laplacian = find_gradients(&nhf);

                let mut c2 = [[0.0f32; 4]; 4];
                let (mut cgs, mut sgs, mut csg) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                for c in 0..4 {
                    let mag = (gradient[0][c] * gradient[0][c] + gradient[1][c] * gradient[1][c]).sqrt();
                    c2[0][c] = -mag * aniso[0];
                    c2[2][c] = -mag * aniso[2];
                    gradient[0][c] = if mag != 0.0 { gradient[0][c] / mag } else { 1.0 };
                    gradient[1][c] = if mag != 0.0 { gradient[1][c] / mag } else { 0.0 };
                    cgs[c] = gradient[0][c] * gradient[0][c];
                    sgs[c] = gradient[1][c] * gradient[1][c];
                    csg[c] = gradient[0][c] * gradient[1][c];
                }
                let (mut cls, mut sls, mut csl) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
                for c in 0..4 {
                    let mag = (laplacian[0][c] * laplacian[0][c] + laplacian[1][c] * laplacian[1][c]).sqrt();
                    c2[1][c] = -mag * aniso[1];
                    c2[3][c] = -mag * aniso[3];
                    laplacian[0][c] = if mag != 0.0 { laplacian[0][c] / mag } else { 1.0 };
                    laplacian[1][c] = if mag != 0.0 { laplacian[1][c] / mag } else { 0.0 };
                    cls[c] = laplacian[0][c] * laplacian[0][c];
                    sls[c] = laplacian[1][c] * laplacian[1][c];
                    csl[c] = laplacian[0][c] * laplacian[1][c];
                }

                for k in 0..4 {
                    c2[k] = vector_exp(c2[k]);
                }

                let kern_first = compute_kernel(c2[0], csg, cgs, sgs, iso[0]);
                let kern_second = compute_kernel(c2[1], csl, cls, sls, iso[1]);
                let kern_third = compute_kernel(c2[2], csg, cgs, sgs, iso[2]);
                let kern_fourth = compute_kernel(c2[3], csl, cls, sls, iso[3]);

                let mut deriv = [[0.0f32; 4]; 4];
                let mut variance = [0.0f32; 4];
                for k in 0..9 {
                    for c in 0..4 {
                        deriv[0][c] += kern_first[k][c] * nlf[k][c];
                        deriv[1][c] += kern_second[k][c] * nlf[k][c];
                        deriv[2][c] += kern_third[k][c] * nhf[k][c];
                        deriv[3][c] += kern_fourth[k][c] * nhf[k][c];
                        variance[c] += nhf[k][c] * nhf[k][c];
                    }
                }
                for c in 0..4 {
                    variance[c] = variance_threshold + variance[c] * regularization_factor;
                }

                let mut acc = [0.0f32; 4];
                for k in 0..4 {
                    for c in 0..4 {
                        acc[c] += deriv[k][c] * abcd[k];
                    }
                }
                for c in 0..4 {
                    let a = hf[index + c] * strength + acc[c] / variance[c];
                    out[index + c] = (a + lf[index + c]).max(0.0);
                }
            } else {
                for c in 0..4 {
                    out[index + c] = hf[index + c] + lf[index + c];
                }
            }
        }
    }
}

/// Compute the diffuse-reconstruction mask: 1 for every pixel where at least
/// one of R, G, B exceeds the threshold, 0 otherwise.
///
/// Matches `build_mask()` in src/iop/diffuse.c. `in_buf` is an RGBA float
/// buffer of length `npixels * 4`; `mask` is a single-byte mask buffer of
/// length `npixels`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_diffuse_build_mask(
    in_buf: *const f32,
    mask: *mut u8,
    npixels: usize,
    threshold: f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let mask  = std::slice::from_raw_parts_mut(mask, npixels);

    for k in 0..npixels {
        let i = k * 4;
        let hit = input[i] > threshold
               || input[i + 1] > threshold
               || input[i + 2] > threshold;
        mask[k] = if hit { 1 } else { 0 };
    }
}

/// Initialise the inpaint buffer with per-pixel Gaussian noise inside masked
/// areas, and copy original pixel values outside them.
///
/// For each pixel k (stride 4):
///   if mask[k/4]:
///     seed state from (i, j) where i = k/width, j = k - i
///     warm-up state ×4
///     for c in 0..4: inpainted[k+c] = abs(gaussian_noise(orig[k+c], orig[k+c], i%2||j%2))
///   else:
///     inpainted[k..k+4] = original[k..k+4]
///
/// Matches `inpaint_mask()` in src/iop/diffuse.c:1302.
#[no_mangle]
pub unsafe extern "C" fn darkroom_diffuse_inpaint_mask(
    inpainted_buf: *mut f32,
    original_buf: *const f32,
    mask_buf: *const u8,
    width: usize,
    height: usize,
) {
    let npx = width * height;
    if npx == 0 { return; }
    let inpainted = std::slice::from_raw_parts_mut(inpainted_buf, npx * 4);
    let original  = std::slice::from_raw_parts(original_buf, npx * 4);
    let mask      = std::slice::from_raw_parts(mask_buf, npx);

    for px in 0..npx {
        let k  = px * 4;         // byte-like offset in the C sense (k strides by 4)
        let kc = k as u32;       // C uses uint32_t arithmetic
        // C: i = k / width;  j = k - i  (where k is 4*pixel_idx)
        let i  = kc / width as u32;
        let j  = kc - i;

        if mask[px] != 0 {
            // Per-pixel deterministic state
            let mut state = [
                crate::math::splitmix32((j + 1) as u64),
                crate::math::splitmix32(((j + 1) as u64) * ((i + 3) as u64)),
                crate::math::splitmix32(1337),
                crate::math::splitmix32(666),
            ];
            // 4 warm-up rounds (matches C)
            for _ in 0..4 { crate::math::xoshiro128plus(&mut state); }

            let flip = (i % 2 != 0) || (j % 2 != 0);
            for c in 0..4 {
                let mu    = original[k + c];
                let sigma = original[k + c];
                let noise = crate::math::gaussian_noise(mu, sigma, flip, &mut state);
                inpainted[k + c] = noise.abs();
            }
        } else {
            inpainted[k..k + 4].copy_from_slice(&original[k..k + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_below_threshold_produces_zero_mask() {
        let input = vec![0.1_f32, 0.2, 0.3, 1.0];
        let mut mask = vec![0xff_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn any_channel_above_threshold_sets_mask() {
        // R alone exceeds the threshold
        let input = vec![0.6_f32, 0.0, 0.0, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 1);

        // G alone
        let input = vec![0.0_f32, 0.6, 0.0, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 1);

        // B alone
        let input = vec![0.0_f32, 0.0, 0.6, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 1);
    }

    #[test]
    fn equal_to_threshold_is_not_a_hit() {
        // Predicate is strict `>` so equality fails
        let input = vec![0.5_f32, 0.5, 0.5, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn alpha_channel_does_not_influence_mask() {
        // High alpha but RGB below threshold → mask 0
        let input = vec![0.1_f32, 0.1, 0.1, 1.0];
        let mut mask = vec![0_u8; 1];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 1, 0.5); }
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn inpaint_mask_passes_through_unmasked_pixels() {
        let orig = vec![0.3_f32, 0.5, 0.7, 1.0];
        let mut inp = vec![-1.0_f32; 4];
        let mask = vec![0_u8; 1]; // pixel not masked
        unsafe {
            darkroom_diffuse_inpaint_mask(inp.as_mut_ptr(), orig.as_ptr(), mask.as_ptr(), 1, 1);
        }
        assert_eq!(inp, orig);
    }

    #[test]
    fn inpaint_mask_fills_masked_pixel_with_nonneg_noise() {
        // When masked, the pixel is filled with |gaussian_noise(v, v)| → always ≥ 0
        let orig = vec![0.5_f32, 0.3, 0.8, 0.9];
        let mut inp = vec![-1.0_f32; 4];
        let mask = vec![1_u8]; // masked
        unsafe {
            darkroom_diffuse_inpaint_mask(inp.as_mut_ptr(), orig.as_ptr(), mask.as_ptr(), 1, 1);
        }
        // All channels non-negative (abs applied)
        for c in 0..4 { assert!(inp[c] >= 0.0, "c={c} val={}", inp[c]); }
        // Result differs from original (noise was added)
        // (with high probability — pathological exact-zero noise would be a 1-in-2^24 event)
    }

    #[test]
    fn inpaint_mask_is_deterministic() {
        // Same seed = same output every time
        let orig = vec![0.4_f32, 0.6, 0.2, 1.0];
        let mask = vec![1_u8];
        let mut out1 = vec![0.0_f32; 4];
        let mut out2 = vec![0.0_f32; 4];
        unsafe {
            darkroom_diffuse_inpaint_mask(out1.as_mut_ptr(), orig.as_ptr(), mask.as_ptr(), 1, 1);
            darkroom_diffuse_inpaint_mask(out2.as_mut_ptr(), orig.as_ptr(), mask.as_ptr(), 1, 1);
        }
        assert_eq!(out1, out2);
    }

    #[test]
    fn multi_pixel_mixed_result() {
        let input = vec![
            0.1, 0.1, 0.1, 1.0,  // below
            0.9, 0.1, 0.1, 1.0,  // R above
            0.1, 0.9, 0.1, 1.0,  // G above
            0.1, 0.1, 0.1, 1.0,  // below
        ];
        let mut mask = vec![0_u8; 4];
        unsafe { darkroom_diffuse_build_mask(input.as_ptr(), mask.as_mut_ptr(), 4, 0.5); }
        assert_eq!(mask, vec![0, 1, 1, 0]);
    }

    // ── heat_PDE_diffusion (m4-87) ──

    #[allow(clippy::too_many_arguments)]
    fn run_pde(
        hf: &[f32], lf: &[f32], mask: Option<&[u8]>, w: usize, h: usize, aniso: [f32; 4],
        iso: [i32; 4], reg: f32, var_thr: f32, radius_sq: f32, mult: i32, abcd: [f32; 4], strength: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h * 4];
        let (mp, hm) = match mask {
            Some(m) => (m.as_ptr(), 1),
            None => (std::ptr::null(), 0),
        };
        unsafe {
            darkroom_diffuse_heat_pde(
                hf.as_ptr(), lf.as_ptr(), mp, hm, out.as_mut_ptr(), w, h, aniso.as_ptr(),
                iso.as_ptr(), reg, var_thr, radius_sq, mult, abcd.as_ptr(), strength,
            );
        }
        out
    }

    #[test]
    fn vector_exp_matches_c_reference() {
        // exp(0) reinterprets 0x3f800000 → 1.0; large-negative floors to 0.
        assert_eq!(vector_exp([0.0; 4]), [1.0; 4]);
        assert_eq!(vector_exp([-1e9; 4]), [0.0; 4]);
        // monotone increasing toward 0 from below, all finite & positive
        let a = vector_exp([-2.0; 4])[0];
        let b = vector_exp([-1.0; 4])[0];
        assert!(0.0 < a && a < b && b < 1.0, "a={a} b={b}");
    }

    #[test]
    fn flat_field_is_preserved() {
        // every diffusion kernel is zero-sum, so a flat LF with zero HF diffuses to
        // itself: out = HF·strength(0·s) + 0 + LF = LF. Holds for any isotropy.
        let (w, h) = (5, 4);
        let lf = vec![0.37f32; w * h * 4];
        let hf = vec![0.0f32; w * h * 4];
        for iso in [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [1, 0, 2, 1]] {
            let out = run_pde(&hf, &lf, None, w, h, [0.5, 0.4, 0.3, 0.2], iso, 1.0, 0.01, 4.0, 1, [0.5, -0.3, 0.2, 0.1], 1.0);
            for (o, l) in out.iter().zip(lf.iter()) {
                assert!((o - l).abs() < 1e-4, "flat field moved: {o} vs {l} (iso={iso:?})");
            }
        }
    }

    #[test]
    fn masked_out_pixels_copy_hf_plus_lf() {
        let (w, h) = (3, 3);
        let hf: Vec<f32> = (0..w * h * 4).map(|i| i as f32 * 0.01).collect();
        let lf: Vec<f32> = (0..w * h * 4).map(|i| i as f32 * 0.02).collect();
        let mask = vec![0u8; w * h]; // all masked out → copy HF+LF
        let out = run_pde(&hf, &lf, Some(&mask), w, h, [0.5; 4], [1; 4], 1.0, 0.01, 4.0, 1, [0.5; 4], 1.0);
        for i in 0..w * h * 4 {
            assert!((out[i] - (hf[i] + lf[i])).abs() < 1e-6, "masked copy wrong at {i}");
        }
    }

    #[test]
    fn nontrivial_input_is_finite_and_nonneg() {
        let (w, h) = (8, 6);
        let mut s: u32 = 0x9e37_79b9;
        let mut rnd = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / 16_777_216.0
        };
        let hf: Vec<f32> = (0..w * h * 4).map(|_| rnd() - 0.5).collect();
        let lf: Vec<f32> = (0..w * h * 4).map(|_| rnd()).collect();
        let out = run_pde(&hf, &lf, None, w, h, [0.6, 0.5, 0.4, 0.3], [1, 2, 0, 1], 2.0, 0.02, 9.0, 2, [0.4, -0.2, 0.3, -0.1], 1.5);
        assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0), "bad output");
    }
}
