use crate::{params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Colortransfer;

impl IopProcess for Colortransfer {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "colortransfer" }
}

/// Max number of clusters (`#define MAXN 5`).
const MAXN: usize = 5;

/// Fuzzy cluster membership weights for a pixel's `(a, b)` (`get_clusters`):
/// squared distance to each cluster mean, min-max normalised, then sum-normalised.
#[inline]
fn get_clusters(a: f32, b: f32, n: usize, mean: &[f32], weight: &mut [f32; MAXN]) {
    let (mut max_d, mut min_d) = (0.0f32, f32::MAX);
    for k in 0..n {
        let da = a - mean[k * 2];
        let db = b - mean[k * 2 + 1];
        let dist = da * da + db * db;
        weight[k] = dist;
        if dist < min_d {
            min_d = dist;
        }
        if dist > max_d {
            max_d = dist;
        }
    }
    if max_d - min_d > 0.0 {
        for w in weight.iter_mut().take(n) {
            *w = (*w - min_d) / (max_d - min_d);
        }
    }
    let sum: f32 = weight[..n].iter().sum();
    if sum > 0.0 {
        for w in weight.iter_mut().take(n) {
            *w /= sum;
        }
    }
}

/// Apply the a/b cluster-transfer pass of the colortransfer IOP — a faithful port
/// of the second DT_OMP_FOR loop in the APPLY branch (colortransfer.c:348, the
/// fuzzy-weighting `#else` path). The L channel is left untouched (already set by
/// [`darkroom_colortransfer_apply_l_histogram`]); this writes a/b (and copies
/// alpha). Deterministic given the cluster data (the k-means step stays in C).
///
/// `mean`/`var`: this image's `n` input clusters (`n·2` floats, a/b per cluster,
/// `var` is the std-dev). `data_mean`/`data_var`: the acquired target clusters.
/// `mapio`: `n` ints mapping each input cluster to a target cluster. `n ≤ MAXN`.
/// Note: no guard against `var[c] == 0` (matches the C, which can emit NaN there).
///
/// # Safety
/// All pointers valid for the stated lengths; `in_buf`/`out_buf` cover
/// `width*height*ch` floats.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn darkroom_colortransfer_apply_ab(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    ch: usize,
    n: i32,
    mean: *const f32,
    var: *const f32,
    data_mean: *const f32,
    data_var: *const f32,
    mapio: *const i32,
) {
    let n = (n as usize).min(MAXN);
    let npix = width * height;
    let input = std::slice::from_raw_parts(in_buf, npix * ch);
    let output = std::slice::from_raw_parts_mut(out_buf, npix * ch);
    let mean = std::slice::from_raw_parts(mean, n * 2);
    let var = std::slice::from_raw_parts(var, n * 2);
    let data_mean = std::slice::from_raw_parts(data_mean, n * 2);
    let data_var = std::slice::from_raw_parts(data_var, n * 2);
    let mapio = std::slice::from_raw_parts(mapio, n);

    for p in 0..npix {
        let j = ch * p;
        let a = input[j + 1];
        let b = input[j + 2];

        let mut weight = [0.0f32; MAXN];
        get_clusters(a, b, n, mean, &mut weight);

        let (mut out_a, mut out_b) = (0.0f32, 0.0f32);
        for c in 0..n {
            let m = mapio[c] as usize;
            out_a += weight[c]
                * ((a - mean[c * 2]) * data_var[m * 2] / var[c * 2] + data_mean[m * 2]);
            out_b += weight[c]
                * ((b - mean[c * 2 + 1]) * data_var[m * 2 + 1] / var[c * 2 + 1] + data_mean[m * 2 + 1]);
        }
        output[j + 1] = out_a;
        output[j + 2] = out_b;
        output[j + 3] = input[j + 3];
    }
}

/// Apply the L-histogram-matching pass of the colortransfer IOP.
///
/// This function migrates ONLY the first DT_OMP_FOR loop of the APPLY
/// branch in src/iop/colortransfer.c (line 327). It does not touch the
/// a/b channels — that is the second OMP loop (line 352) which depends on
/// k-means clustering and remains in C for now. Production callers must
/// follow this call with the C-side ab-cluster pass; failure to do so
/// leaves ab unchanged in the output. The unit tests below verify the
/// function in isolation only.
///
/// Per pixel:
///   src_bin     = clamp((histn as f32) * in_L / 100, 0, histn - 1)
///   target_bin  = cdf_lut[src_bin]      (already normalised to [0, histn-1])
///   out_L       = clamp(inverse_cdf[target_bin], 0, 100)
///
/// `cdf_lut` is the normalised cumulative-distribution lookup produced by
/// `capture_histogram()` in C (line 139 normalises values to [0, HISTN-1]
/// via `hist[k] = CLAMP(hist[k] * HISTN / hist[HISTN-1], 0, HISTN-1)`).
/// `inverse_cdf` is the inverse-CDF lookup produced by `invert_histogram()`
/// — values in [0, 100). We clamp the final output to [0, 100] defensively.
///
/// Both LUTs are exactly `histn` entries long.
#[no_mangle]
pub unsafe extern "C" fn darkroom_colortransfer_apply_l_histogram(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    ch: usize,
    cdf_lut: *const i32,
    inverse_cdf: *const f32,
    histn: usize,
) {
    if ch == 0 || histn == 0 { return; }
    let npx = width * height;
    if npx == 0 { return; } // Guards against from_raw_parts(NULL, 0) UB.

    debug_assert!(
        width.checked_mul(height).and_then(|n| n.checked_mul(ch)).is_some(),
        "width * height * ch overflows usize"
    );

    let input = std::slice::from_raw_parts(in_buf, npx * ch);
    let output = std::slice::from_raw_parts_mut(out_buf, npx * ch);
    let cdf = std::slice::from_raw_parts(cdf_lut, histn);
    let inv = std::slice::from_raw_parts(inverse_cdf, histn);
    let last = (histn - 1) as i32;

    for k in 0..npx {
        let j = k * ch;
        let l = input[j];
        // First clamp: float-domain saturation of HISTN * L / 100 into [0, last].
        let bin_f = ((histn as f32) * l / 100.0).clamp(0.0, last as f32);
        // Second clamp is intentional, not paranoia: float→int cast can yield a
        // negative i32 for `-0.0` or denormal inputs that survive the float
        // clamp on some targets. Keep it as the load-bearing safety net.
        let src_bin = (bin_f as i32).clamp(0, last) as usize;

        // The CDF LUT is already normalised by capture_histogram (line 139 of
        // colortransfer.c) so this clamp is also defensive — it protects against
        // a non-normalised LUT being passed in by future callers.
        debug_assert!(
            cdf[src_bin] >= 0 && cdf[src_bin] < histn as i32,
            "cdf_lut[{}] = {} is out of [0, {}); caller must run capture_histogram first",
            src_bin, cdf[src_bin], histn
        );
        let target_bin = cdf[src_bin].clamp(0, last) as usize;
        output[j] = inv[target_bin].clamp(0.0, 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HISTN: usize = 1 << 11;

    /// Build the trivial identity pair: cdf[i] = i and inv[i] = 100*i/(HISTN-1).
    fn identity_histograms() -> (Vec<i32>, Vec<f32>) {
        let cdf: Vec<i32> = (0..HISTN as i32).collect();
        let inv: Vec<f32> = (0..HISTN).map(|i| 100.0 * i as f32 / (HISTN - 1) as f32).collect();
        (cdf, inv)
    }

    #[test]
    fn identity_histograms_preserve_luminance() {
        let (cdf, inv) = identity_histograms();
        let inp = vec![25.0_f32, 0.0, 0.0, 1.0];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_colortransfer_apply_l_histogram(
                inp.as_ptr(), out.as_mut_ptr(), 1, 1, 4,
                cdf.as_ptr(), inv.as_ptr(), HISTN,
            );
        }
        assert!((out[0] - 25.0).abs() < 0.06, "got {}", out[0]);
    }

    #[test]
    fn negative_luminance_clamps_to_first_bin() {
        let (cdf, inv) = identity_histograms();
        let inp = vec![-10.0_f32, 0.0, 0.0, 0.0];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_colortransfer_apply_l_histogram(
                inp.as_ptr(), out.as_mut_ptr(), 1, 1, 4,
                cdf.as_ptr(), inv.as_ptr(), HISTN,
            );
        }
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn above_one_hundred_clamps_to_last_bin() {
        let (cdf, inv) = identity_histograms();
        let inp = vec![999.0_f32, 0.0, 0.0, 0.0];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_colortransfer_apply_l_histogram(
                inp.as_ptr(), out.as_mut_ptr(), 1, 1, 4,
                cdf.as_ptr(), inv.as_ptr(), HISTN,
            );
        }
        assert!((out[0] - 100.0).abs() < 1e-3, "got {}", out[0]);
    }

    #[test]
    fn inverse_cdf_output_is_clamped_to_unit_range() {
        // If the inverse CDF LUT has out-of-range values (e.g. a corrupted
        // data->hist) the function must still clamp.
        let cdf: Vec<i32> = vec![0; HISTN];
        let mut inv = vec![0.0_f32; HISTN];
        inv[0] = 200.0;
        let inp = vec![50.0_f32, 0.0, 0.0, 0.0];
        let mut out = vec![-1.0_f32; 4];
        unsafe {
            darkroom_colortransfer_apply_l_histogram(
                inp.as_ptr(), out.as_mut_ptr(), 1, 1, 4,
                cdf.as_ptr(), inv.as_ptr(), HISTN,
            );
        }
        assert_eq!(out[0], 100.0);
    }

    #[test]
    fn function_in_isolation_does_not_touch_ab_or_alpha() {
        // ⚠ This test verifies the L-histogram-matching pass alone. The full
        // colortransfer APPLY branch in C also runs a k-means-driven ab pass
        // (src/iop/colortransfer.c line 352) which rewrites out[..1..3]; that
        // pass is still in C for now, so the IOP as a whole DOES modify ab,
        // alpha. This test only proves that THIS function leaves them alone
        // — confirming the row stride math and the j+0-only writes.
        let (cdf, inv) = identity_histograms();
        let inp = vec![50.0_f32, -100.0, 100.0, 0.42];
        let mut out = vec![-7.0_f32, -7.0, -7.0, -7.0];
        unsafe {
            darkroom_colortransfer_apply_l_histogram(
                inp.as_ptr(), out.as_mut_ptr(), 1, 1, 4,
                cdf.as_ptr(), inv.as_ptr(), HISTN,
            );
        }
        assert!((out[0] - 50.0).abs() < 0.06);
        assert_eq!(out[1], -7.0);
        assert_eq!(out[2], -7.0);
        assert_eq!(out[3], -7.0);
    }

    #[test]
    fn zero_width_height_is_a_safe_noop() {
        let (cdf, inv) = identity_histograms();
        unsafe {
            darkroom_colortransfer_apply_l_histogram(
                std::ptr::null(), std::ptr::null_mut(), 0, 0, 4,
                cdf.as_ptr(), inv.as_ptr(), HISTN,
            );
        }
    }

    #[test]
    fn multi_pixel_row_stride_correct() {
        // 3-pixel row with ch=4 — ensure we touch out[0], out[4], out[8].
        let (cdf, inv) = identity_histograms();
        let inp = vec![
            10.0, -1.0, -1.0, -1.0,
            20.0, -1.0, -1.0, -1.0,
            30.0, -1.0, -1.0, -1.0,
        ];
        let mut out = vec![-7.0_f32; 12];
        unsafe {
            darkroom_colortransfer_apply_l_histogram(
                inp.as_ptr(), out.as_mut_ptr(), 3, 1, 4,
                cdf.as_ptr(), inv.as_ptr(), HISTN,
            );
        }
        assert!((out[0] - 10.0).abs() < 0.06);
        assert!((out[4] - 20.0).abs() < 0.06);
        assert!((out[8] - 30.0).abs() < 0.06);
        // Untouched neighbour slots still hold the sentinel
        assert_eq!(out[1], -7.0);
        assert_eq!(out[5], -7.0);
        assert_eq!(out[11], -7.0);
    }

    // ── a/b cluster transfer (m4-88) ──

    #[allow(clippy::too_many_arguments)]
    fn run_ab(
        input: &[f32], w: usize, h: usize, ch: usize, n: i32, mean: &[f32], var: &[f32],
        dmean: &[f32], dvar: &[f32], mapio: &[i32], out_init: f32,
    ) -> Vec<f32> {
        let mut out = vec![out_init; w * h * ch];
        unsafe {
            darkroom_colortransfer_apply_ab(
                input.as_ptr(), out.as_mut_ptr(), w, h, ch, n, mean.as_ptr(), var.as_ptr(),
                dmean.as_ptr(), dvar.as_ptr(), mapio.as_ptr(),
            );
        }
        out
    }

    #[test]
    fn get_clusters_weights_sum_to_one() {
        let mean = [5.0, 5.0, 10.0, -3.0];
        let mut w = [0.0f32; MAXN];
        get_clusters(7.0, 1.0, 2, &mean, &mut w);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-6, "weights={w:?}");
        // pixel exactly at cluster 0's mean → min-dist cluster gets weight 0
        let mut w2 = [0.0f32; MAXN];
        get_clusters(5.0, 5.0, 2, &mean, &mut w2);
        assert!(w2[0].abs() < 1e-6 && (w2[1] - 1.0).abs() < 1e-6, "weights2={w2:?}");
    }

    #[test]
    fn apply_ab_is_identity_when_target_equals_source() {
        // data_mean=mean, data_var=var, mapio=identity ⇒ each term reduces to `a`,
        // and Σweight=1 ⇒ out a/b == input a/b.
        let (w, h, ch) = (2, 1, 4);
        let mean = [5.0, 5.0, 10.0, -3.0];
        let var = [2.0, 2.0, 3.0, 3.0];
        let mapio = [0i32, 1];
        // two pixels, neither at a cluster mean
        let input = [50.0, 7.0, 1.0, 1.0, 40.0, 8.0, -2.0, 0.5];
        let out = run_ab(&input, w, h, ch, 2, &mean, &var, &mean, &var, &mapio, -99.0);
        assert!((out[1] - 7.0).abs() < 1e-3 && (out[2] - 1.0).abs() < 1e-3, "px0 {:?}", &out[0..4]);
        assert!((out[5] - 8.0).abs() < 1e-3 && (out[6] + 2.0).abs() < 1e-3, "px1 {:?}", &out[4..8]);
    }

    #[test]
    fn apply_ab_leaves_l_untouched_and_copies_alpha() {
        let (w, h, ch) = (1, 1, 4);
        let mean = [5.0, 5.0];
        let var = [2.0, 2.0];
        let input = [42.0, 7.0, 1.0, 0.75];
        let out = run_ab(&input, w, h, ch, 1, &mean, &var, &mean, &var, &[0i32], -99.0);
        assert_eq!(out[0], -99.0, "L must be left as-is (set by the histogram pass)");
        assert_eq!(out[3], 0.75, "alpha copied from input");
        assert!(out[1].is_finite() && out[2].is_finite());
    }
}
