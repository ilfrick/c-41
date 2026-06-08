use crate::{math, params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Colorchecker;

impl IopProcess for Colorchecker {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colorchecker" }
}

/// Thin-plate-spline radial basis function used by colorchecker.
///
///   r^2 = sum_c (x[c] - y[c])^2     (c in 0..3)
///   phi(r) = r^2 * fastlog(max(1e-8, r^2))
///
/// Matches `kernel()` in src/iop/colorchecker.c byte-for-byte.
#[inline(always)]
fn kernel(x: &[f32; 4], y: &[f32; 4]) -> f32 {
    let dx = x[0] - y[0];
    let dy = x[1] - y[1];
    let dz = x[2] - y[2];
    let r2 = dx * dx + dy * dy + dz * dz;
    r2 * math::fastlog(r2.max(1e-8))
}

/// Apply a colour-checker correction: per pixel, evaluate a thin-plate-spline
/// surface defined by N source patches plus an affine polynomial fall-off.
///
/// Algorithm (verbatim from src/iop/colorchecker.c process()):
///   sums[c] = polynomial_<c>[0]*Lab[0] + polynomial_<c>[1]*Lab[1] + polynomial_<c>[2]*Lab[2]
///   res[c]  = patches[N][c] + sums[c]                         // affine offset
///   for each patch p in 0..N:
///     res[c] += patches[p][c] * kernel(Lab, sources[p])
///   out[k] = (res[0], res[1], res[2], 0.0)
///
/// `polynomial_L`, `polynomial_a`, `polynomial_b` are 3-float arrays packing
/// the per-channel polynomial coefficients (stride 4 in memory but only the
/// first three components are read; matches the dt_aligned_pixel_t layout).
/// `patches` is `(num_patches + 1) * 4` floats (one extra "intercept"
/// row at the end). `sources` is `num_patches * 4` floats (each source
/// uses the L,a,b components only).
///
/// `npixels` is `width * height`; in/out are RGBA float buffers.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorchecker_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    num_patches: usize,
    sources: *const f32,
    patches: *const f32,
    polynomial_l: *const f32,
    polynomial_a: *const f32,
    polynomial_b: *const f32,
) {
    let input = std::slice::from_raw_parts(in_buf, npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);

    // `sources` may legitimately be NULL when there are no patches (the loop
    // below never reads it in that case). `slice::from_raw_parts` is UB on a
    // null pointer even for length 0, so hand back an empty slice instead.
    let sources_slice = if num_patches == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(sources, num_patches * 4)
    };
    let patches_slice = std::slice::from_raw_parts(patches, (num_patches + 1) * 4);
    let pl = std::slice::from_raw_parts(polynomial_l, 3);
    let pa = std::slice::from_raw_parts(polynomial_a, 3);
    let pb = std::slice::from_raw_parts(polynomial_b, 3);

    // Intercept patch (the (N+1)-th row of `patches`) is read once.
    let intercept = [
        patches_slice[num_patches * 4],
        patches_slice[num_patches * 4 + 1],
        patches_slice[num_patches * 4 + 2],
    ];

    for k in 0..npixels {
        let j = k * 4;
        let lab = [input[j], input[j + 1], input[j + 2], 0.0_f32];

        // Affine polynomial fall-off (sum over the L,a,b input channels).
        let sum_l = pl[0] * lab[0] + pl[1] * lab[1] + pl[2] * lab[2];
        let sum_a = pa[0] * lab[0] + pa[1] * lab[1] + pa[2] * lab[2];
        let sum_b = pb[0] * lab[0] + pb[1] * lab[1] + pb[2] * lab[2];

        let mut res = [
            intercept[0] + sum_l,
            intercept[1] + sum_a,
            intercept[2] + sum_b,
        ];

        // Thin-plate-spline RBF sum.
        for p in 0..num_patches {
            let src = [
                sources_slice[p * 4],
                sources_slice[p * 4 + 1],
                sources_slice[p * 4 + 2],
                0.0,
            ];
            let phi = kernel(&lab, &src);
            let pe = p * 4;
            res[0] += patches_slice[pe] * phi;
            res[1] += patches_slice[pe + 1] * phi;
            res[2] += patches_slice[pe + 2] * phi;
        }

        output[j]     = res[0];
        output[j + 1] = res[1];
        output[j + 2] = res[2];
        output[j + 3] = 0.0; // matches the C dt_aligned_pixel_t initialisation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros4() -> [f32; 4] { [0.0; 4] }

    #[test]
    fn kernel_zero_when_x_equals_y() {
        // r2 = 0 → fastlog(max(1e-8, 0)) is finite, multiplied by r2=0 → 0
        let x = [0.5, 0.5, 0.5, 0.0];
        let y = x;
        assert_eq!(kernel(&x, &y), 0.0);
    }

    #[test]
    fn kernel_positive_for_diverging_inputs() {
        // r2 = 1.0; fastlog(1.0) ≈ 0 → kernel ≈ 0
        // r2 = 4.0; fastlog(4.0) ≈ ln 4 > 0 → kernel > 0
        let x = [0.0, 0.0, 0.0, 0.0];
        let y = [2.0, 0.0, 0.0, 0.0];
        let k = kernel(&x, &y);
        assert!(k > 0.0, "k={k}");
    }

    #[test]
    fn process_with_zero_patches_runs_polynomial_only() {
        // 1 pixel, 0 patches → only the intercept and the polynomial sum apply.
        let inp = vec![10.0_f32, 20.0, 30.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        // patches has 0+1 = 1 row (the intercept)
        let intercept = [1.0_f32, 2.0, 3.0, 0.0];
        // polynomial_L = identity on L (coeff[0]=1) → sum_L = L
        let pl = [1.0_f32, 0.0, 0.0];
        let pa = [0.0_f32, 1.0, 0.0];
        let pb = [0.0_f32, 0.0, 1.0];
        unsafe {
            darkroom_colorchecker_process(
                inp.as_ptr(), out.as_mut_ptr(), 1, 0,
                std::ptr::null(), intercept.as_ptr(),
                pl.as_ptr(), pa.as_ptr(), pb.as_ptr(),
            );
        }
        // out should be intercept + identity-mapped input
        assert!((out[0] - (1.0 + 10.0)).abs() < 1e-5);
        assert!((out[1] - (2.0 + 20.0)).abs() < 1e-5);
        assert!((out[2] - (3.0 + 30.0)).abs() < 1e-5);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn process_alpha_channel_is_zeroed() {
        let inp = vec![0.0_f32, 0.0, 0.0, 0.42];
        let mut out = vec![1.0_f32; 4];
        let intercept = zeros4();
        let pl = [0.0_f32; 3];
        let pa = [0.0_f32; 3];
        let pb = [0.0_f32; 3];
        unsafe {
            darkroom_colorchecker_process(
                inp.as_ptr(), out.as_mut_ptr(), 1, 0,
                std::ptr::null(), intercept.as_ptr(),
                pl.as_ptr(), pa.as_ptr(), pb.as_ptr(),
            );
        }
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn process_uses_polynomial_per_channel_independently() {
        // pl identity, pa zero, pb zero → out_L = L, out_a = 0, out_b = 0
        let inp = vec![5.0_f32, 7.0, 9.0, 0.0];
        let mut out = vec![0.0_f32; 4];
        let intercept = [0.0_f32; 4];
        let pl = [1.0_f32, 0.0, 0.0];
        let pa = [0.0_f32; 3];
        let pb = [0.0_f32; 3];
        unsafe {
            darkroom_colorchecker_process(
                inp.as_ptr(), out.as_mut_ptr(), 1, 0,
                std::ptr::null(), intercept.as_ptr(),
                pl.as_ptr(), pa.as_ptr(), pb.as_ptr(),
            );
        }
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
    }
}
