use crate::{color, params::IopParams, roi::RoiIn, Result};
use super::{ClBuffer, IopProcess};

pub struct Clahe;

impl IopProcess for Clahe {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "clahe" }
}

const BINS: usize = 256;

/// Convert a pixel index into a histogram bin: round-positive of luminance*BINS,
/// clamped to the valid range. Matches the C ROUND_POSISTIVE macro
/// (which is `(int)(x + 0.5f)`) followed by an implicit clamp through the
/// histogram array bounds.
#[inline(always)]
fn bin(l: f32) -> usize {
    let v = (l * BINS as f32 + 0.5) as i32;
    v.clamp(0, BINS as i32) as usize
}

/// Apply the contrast-limited adaptive histogram equalisation in src/iop/clahe.c
/// to an RGBA float image.
///
/// Pipeline (matches the C process() function exactly):
///   1. Per pixel: luminance[k] = (max(R,G,B) + min(R,G,B)) / 2 (clipped to [0,1])
///   2. Per row: maintain a sliding (2*rad+1)x(2*rad+1) histogram of `luminance`,
///      clip it at `slope*n/BINS` and redistribute the excess uniformly,
///      build the CDF, and look up the new luminance for the centre pixel.
///   3. For each pixel: RGB → HSL, swap L with the equalised value, HSL → RGB.
///
/// The output `out_buf` must be a separate buffer from `in_buf` (in-place use
/// is supported by the C code only because the inner Apply pass writes to
/// `out` and then the loop reads it back; we replicate that contract).
#[no_mangle]
pub unsafe extern "C" fn darkroom_clahe_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    width: usize,
    height: usize,
    rad: i32,
    slope: f32,
) {
    if width == 0 || height == 0 { return; }
    let npx = width * height;
    let input = std::slice::from_raw_parts(in_buf, npx * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npx * 4);

    // ── PASS 1: per-pixel luminance map ──────────────────────────────────────
    let mut luminance = vec![0.0_f32; npx];
    for j in 0..height {
        for i in 0..width {
            let base = (j * width + i) * 4;
            let r = input[base].clamp(0.0, 1.0);
            let g = input[base + 1].clamp(0.0, 1.0);
            let b = input[base + 2].clamp(0.0, 1.0);
            let pmax = r.max(g).max(b);
            let pmin = r.min(g).min(b);
            luminance[j * width + i] = (pmax + pmin) * 0.5;
        }
    }

    let rad = rad.max(0) as usize;
    let mut hist = [0_i32; BINS + 1];
    let mut clippedhist = [0_i32; BINS + 1];
    let mut dest = vec![0.0_f32; width];

    // ── PASS 2: per-row sliding-window CLAHE ─────────────────────────────────
    for j in 0..height {
        let y_min = j.saturating_sub(rad);
        let y_max = (j + rad + 1).min(height);
        let h = y_max - y_min;

        // initial fill: columns [0 - rad .. 0 + rad], clamped to image
        let x_min0 = 0_usize; // 0 - rad clamped to 0
        let x_max0 = rad.min(width.saturating_sub(1));

        hist.fill(0);
        for yi in y_min..y_max {
            for xi in x_min0..x_max0 {
                let b = bin(luminance[yi * width + xi]);
                hist[b] += 1;
            }
        }

        for i in 0..width {
            let v = bin(luminance[j * width + i]);

            let x_min = i.saturating_sub(rad);
            let x_max_raw = i + rad + 1;
            let w = x_max_raw.min(width) - x_min;
            let n = h * w;

            let limit = (slope * n as f32 / BINS as f32 + 0.5) as i32;

            // remove the column that just left the window
            if x_min > 0 {
                let x_min1 = x_min - 1;
                for yi in y_min..y_max {
                    let b = bin(luminance[yi * width + x_min1]);
                    hist[b] -= 1;
                }
            }
            // add the column that just entered the window
            if x_max_raw <= width {
                let x_max1 = x_max_raw - 1;
                for yi in y_min..y_max {
                    let b = bin(luminance[yi * width + x_max1]);
                    hist[b] += 1;
                }
            }

            // clip + redistribute excess uniformly (iterate to convergence)
            clippedhist.copy_from_slice(&hist);
            let mut ce: i32 = 0;
            let mut ceb: i32;
            loop {
                ceb = ce;
                ce = 0;
                for b in 0..=BINS {
                    let d = clippedhist[b] - limit;
                    if d > 0 {
                        ce += d;
                        clippedhist[b] = limit;
                    }
                }

                let d = (ce as f32 / (BINS + 1) as f32) as i32;
                let m = ce % (BINS + 1) as i32;
                for b in 0..=BINS { clippedhist[b] += d; }

                if m != 0 {
                    let s = (BINS as f32 / m as f32) as i32;
                    if s > 0 {
                        let mut b = 0_i32;
                        while b <= BINS as i32 {
                            clippedhist[b as usize] += 1;
                            b += s;
                        }
                    }
                }
                if ce == ceb { break; }
            }

            // build CDF: cdf = sum from h_min to v; cdfMax = total above h_min;
            // cdfMin = clippedhist[h_min]
            let mut h_min = BINS;
            for b in 0..h_min {
                if clippedhist[b] != 0 { h_min = b; break; }
            }

            let mut cdf: i32 = 0;
            for b in h_min..=v {
                cdf += clippedhist[b];
            }
            let mut cdf_max = cdf;
            for b in (v + 1)..=BINS {
                cdf_max += clippedhist[b];
            }
            let cdf_min = clippedhist[h_min];

            let denom = (cdf_max - cdf_min) as f32;
            dest[i] = if denom > 0.0 {
                (cdf - cdf_min) as f32 / denom
            } else {
                0.0
            };
        }

        // ── Apply row: swap L channel with equalised value ────────────────
        for r in 0..width {
            let base = (j * width + r) * 4;
            let pr = input[base];
            let pg = input[base + 1];
            let pb = input[base + 2];
            let pa = input[base + 3];

            let (h, s, _l_old) = color::rgb2hsl(pr, pg, pb);
            let (or, og, ob, _) = color::hsl2rgb(h, s, dest[r]);
            output[base]     = or;
            output[base + 1] = og;
            output[base + 2] = ob;
            output[base + 3] = pa;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_clamps_extremes() {
        assert_eq!(bin(-10.0), 0);
        assert_eq!(bin(1000.0), BINS);
        assert_eq!(bin(0.0), 0);
        assert!(bin(0.5) >= BINS / 2 - 1 && bin(0.5) <= BINS / 2 + 1);
    }

    #[test]
    fn solid_grey_image_stays_grey() {
        // Uniform grey should map to itself (CLAHE on constant histogram is a no-op
        // for the centre value, and HSL round-trip is lossless for greys with S=0).
        let w = 16; let h = 16;
        let mut input = vec![0.0_f32; w * h * 4];
        for k in 0..(w * h) {
            input[k * 4]     = 0.5;
            input[k * 4 + 1] = 0.5;
            input[k * 4 + 2] = 0.5;
            input[k * 4 + 3] = 1.0;
        }
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe { darkroom_clahe_process(input.as_ptr(), out.as_mut_ptr(), w, h, 4, 2.0); }

        // CLAHE on uniform luminance produces an output of 0 (the bin spread is
        // 0 / 0); we accept any well-defined grey value. The key invariant is
        // that the image is still neutral (R == G == B) everywhere.
        for k in 0..(w * h) {
            let r = out[k * 4];
            let g = out[k * 4 + 1];
            let b = out[k * 4 + 2];
            assert!((r - g).abs() < 1e-4, "non-grey at {k}: r={r} g={g}");
            assert!((g - b).abs() < 1e-4, "non-grey at {k}: g={g} b={b}");
            assert_eq!(out[k * 4 + 3], 1.0, "alpha changed at {k}");
        }
    }

    #[test]
    fn alpha_channel_is_preserved() {
        let w = 8; let h = 8;
        let mut input = vec![0.5_f32; w * h * 4];
        for k in 0..(w * h) {
            input[k * 4 + 3] = 0.42; // distinct alpha
        }
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe { darkroom_clahe_process(input.as_ptr(), out.as_mut_ptr(), w, h, 2, 2.0); }
        for k in 0..(w * h) {
            assert!((out[k * 4 + 3] - 0.42).abs() < 1e-6, "alpha at {k} = {}", out[k * 4 + 3]);
        }
    }

    #[test]
    fn zero_radius_does_not_panic() {
        let w = 4; let h = 4;
        let input = vec![0.3_f32; w * h * 4];
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe { darkroom_clahe_process(input.as_ptr(), out.as_mut_ptr(), w, h, 0, 1.0); }
        // With rad=0 the window has zero area in pass 1 (xMax0 = 0); the
        // function must still run to completion without out-of-bounds.
    }
}
