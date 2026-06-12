//! Demosaic shared basics (src/iop/demosaicing/basics.c): green-channel
//! pre-median filter, RGB colour smoothing, and the two Bayer green
//! equilibration variants. The demosaic algorithms themselves (PPG, RCD,
//! VNG, X-Trans, capture sharpening) remain in C for now.

use crate::raw;

const GREEN: usize = 1; // imageop_math.h:160

/// Green-channel pre-median filter. Matches pre_median_b()
/// (basics.c:27) including its quirks: every pass reads the *original*
/// input (passes are idempotent), out-of-threshold neighbours join the
/// median shifted by +64, and a single in-threshold sample (cnt == 1)
/// selects med[4] - 64. With cnt == 0 (possible when threshold <= 0),
/// (cnt-1)/2 truncates to 0 exactly like the C int division.
///
/// # Safety
/// `out`/`in_buf` hold `width * height` floats; width/height >= 7.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_pre_median(
    out: *mut f32, in_buf: *const f32,
    width: usize, height: usize,
    filters: u32, num_passes: i32, threshold: f32,
) {
    let inb = std::slice::from_raw_parts(in_buf, width * height);
    let o = std::slice::from_raw_parts_mut(out, width * height);
    o.copy_from_slice(inb);

    // degenerate ROI: the C loops (int bounds) simply didn't run
    if width < 7 || height < 7 {
        return;
    }

    // now green:
    const LIM: [i32; 5] = [0, 1, 2, 1, 0];
    for _pass in 0..num_passes {
        for row in 3..height - 3 {
            let mut med = [0.0_f32; 9];
            let mut col = 3_usize;
            let f = raw::fc_bayer(row as i32, col as i32, filters);
            if f != 1 && f != 3 {
                col += 1;
            }
            while col < width - 3 {
                let pix = row * width + col;
                let mut cnt = 0_i32;
                let mut k = 0_usize;
                for i in 0..5_i32 {
                    let mut j = -LIM[i as usize];
                    while j <= LIM[i as usize] {
                        let v = inb[(pix as isize + (i - 2) as isize * width as isize + j as isize)
                            as usize];
                        if (v - inb[pix]).abs() < threshold {
                            med[k] = v;
                            cnt += 1;
                        } else {
                            med[k] = 64.0 + v;
                        }
                        k += 1;
                        j += 2;
                    }
                }
                for i in 0..8 {
                    for ii in i + 1..9 {
                        if med[i] > med[ii] {
                            med.swap(i, ii);
                        }
                    }
                }
                o[pix] = if cnt == 1 { med[4] - 64.0 } else { med[((cnt - 1) / 2) as usize] };
                col += 2;
            }
        }
    }
}

/// One SWAPmed step of the optimal 9-element median network (basics.c:88).
#[inline(always)]
fn swap_med(med: &mut [f32; 9], i: usize, j: usize) {
    if med[i] > med[j] {
        med.swap(i, j);
    }
}

/// RGB colour smoothing by median filtering of the R-G / B-G differences.
/// Matches color_smoothing() (basics.c:91): per pass and per channel
/// c in {0, 2}, channel 3 is first prefilled with channel c for every
/// pixel, then each interior pixel's channel c becomes
/// max(0, median9(ch3 - ch1 over the 3x3 hood) + ch1). The exact partial
/// sorting network is replicated (it leaves the median in med[4] without
/// fully sorting). In-place safe: writes touch channel c only while the
/// neighbourhood reads channels 3 and 1.
///
/// # Safety
/// `out` holds `width * height * 4` floats; width/height >= 3.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_color_smoothing(
    out: *mut f32, width: usize, height: usize, num_passes: i32,
) {
    let o = std::slice::from_raw_parts_mut(out, width * height * 4);
    let w4 = 4 * width;

    for _pass in 0..num_passes {
        for c in [0_usize, 2] {
            for px in o.chunks_exact_mut(4) {
                px[3] = px[c];
            }
            // degenerate ROI: the C interior loop (int bounds) didn't run
            if width < 3 || height < 3 {
                continue;
            }
            for j in 1..height - 1 {
                for i in 1..width - 1 {
                    let p = 4 * (j * width + i);
                    let d = |off: isize| -> f32 {
                        let b = (p as isize + off) as usize;
                        o[b + 3] - o[b + 1]
                    };
                    let w4 = w4 as isize;
                    let mut med = [
                        d(-w4 - 4), d(-w4), d(-w4 + 4),
                        d(-4),      d(0),   d(4),
                        d(w4 - 4),  d(w4),  d(w4 + 4),
                    ];
                    /* optimal 9-element median search */
                    swap_med(&mut med, 1, 2);
                    swap_med(&mut med, 4, 5);
                    swap_med(&mut med, 7, 8);
                    swap_med(&mut med, 0, 1);
                    swap_med(&mut med, 3, 4);
                    swap_med(&mut med, 6, 7);
                    swap_med(&mut med, 1, 2);
                    swap_med(&mut med, 4, 5);
                    swap_med(&mut med, 7, 8);
                    swap_med(&mut med, 0, 3);
                    swap_med(&mut med, 5, 8);
                    swap_med(&mut med, 4, 7);
                    swap_med(&mut med, 3, 6);
                    swap_med(&mut med, 1, 4);
                    swap_med(&mut med, 2, 5);
                    swap_med(&mut med, 4, 7);
                    swap_med(&mut med, 4, 2);
                    swap_med(&mut med, 6, 4);
                    swap_med(&mut med, 4, 2);
                    o[p + c] = (med[4] + o[p + 1]).max(0.0);
                }
            }
        }
    }
}

/// Local-average green equilibration. Matches green_equilibration_lavg()
/// (basics.c:149): for every second green site (offset probed via FC), when
/// the diagonal (G1) and straight (G2) neighbour means are positive, their
/// ratio bounded, the pixel unsaturated and both neighbourhoods flat, the
/// green value is scaled by m1/m2.
///
/// # Safety
/// `out`/`in_buf` hold `width * height` floats; width/height >= 5.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_green_eq_lavg(
    out: *mut f32, in_buf: *const f32,
    width: usize, height: usize, filters: u32, thr: f32,
) {
    let inb = std::slice::from_raw_parts(in_buf, width * height);
    let o = std::slice::from_raw_parts_mut(out, width * height);
    let maximum = 1.0_f32;

    let (mut oj, mut oi) = (2_i32, 2_i32);
    if raw::fc_bayer(oj, oi, filters) != GREEN {
        oj += 1;
    }
    if raw::fc_bayer(oj, oi, filters) != GREEN {
        oi += 1;
    }
    if raw::fc_bayer(oj, oi, filters) != GREEN {
        oj -= 1;
    }

    o.copy_from_slice(inb);

    // degenerate ROI guard. NB: the C compared size_t j against the int
    // expression height-2, which for height < 2 wrapped to a huge unsigned
    // value and read out of bounds — clamped here (like raw_to_cmap was).
    if width < 5 || height < 5 {
        return;
    }

    let mut j = oj as usize;
    while j < height - 2 {
        let mut i = oi as usize;
        while i < width - 2 {
            let o1_1 = inb[(j - 1) * width + i - 1];
            let o1_2 = inb[(j - 1) * width + i + 1];
            let o1_3 = inb[(j + 1) * width + i - 1];
            let o1_4 = inb[(j + 1) * width + i + 1];
            let o2_1 = inb[(j - 2) * width + i];
            let o2_2 = inb[(j + 2) * width + i];
            let o2_3 = inb[j * width + i - 2];
            let o2_4 = inb[j * width + i + 2];

            let m1 = (o1_1 + o1_2 + o1_3 + o1_4) / 4.0;
            let m2 = (o2_1 + o2_2 + o2_3 + o2_4) / 4.0;

            // prevent divide by zero, guard against hot pixels from m2 too
            // small, and require m1 positive
            if m2 > 0.0 && m1 > 0.0 && m1 / m2 < maximum * 2.0 {
                let c1 = ((o1_1 - o1_2).abs() + (o1_1 - o1_3).abs() + (o1_1 - o1_4).abs()
                    + (o1_2 - o1_3).abs() + (o1_3 - o1_4).abs() + (o1_2 - o1_4).abs())
                    / 6.0;
                let c2 = ((o2_1 - o2_2).abs() + (o2_1 - o2_3).abs() + (o2_1 - o2_4).abs()
                    + (o2_2 - o2_3).abs() + (o2_3 - o2_4).abs() + (o2_2 - o2_4).abs())
                    / 6.0;
                if inb[j * width + i] < maximum * 0.95 && c1 < maximum * thr && c2 < maximum * thr
                {
                    o[j * width + i] = (inb[j * width + i] * m1 / m2).max(0.0);
                }
            }
            i += 2;
        }
        j += 2;
    }
}

/// Full-average green equilibration. Matches green_equilibration_favg()
/// (basics.c:200): double-precision sums of the two green sublattices, then
/// the G1 sites are scaled by sum2/sum1 (no-op when either sum is zero —
/// the copy already happened, like the C's early return after the memcpy).
///
/// # Safety
/// `out`/`in_buf` hold `width * height` floats; width/height >= 2.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_green_eq_favg(
    out: *mut f32, in_buf: *const f32,
    width: usize, height: usize, filters: u32,
) {
    let inb = std::slice::from_raw_parts(in_buf, width * height);
    let o = std::slice::from_raw_parts_mut(out, width * height);

    let oj = 0_usize;
    let mut oi = 0_usize;
    if raw::fc_bayer(0, 0, filters) & 1 != 1 {
        oi += 1;
    }
    let g2_offset: isize = if oi != 0 { -1 } else { 1 };

    o.copy_from_slice(inb);

    // degenerate ROI: with g2_offset == 1 and width <= 2 the C int bound
    // width-1-g2_offset is <= 0 and its loops didn't run
    let i_end_s = width as isize - 1 - g2_offset;
    if height < 2 || i_end_s <= oi as isize {
        return;
    }
    let i_end = i_end_s as usize;

    let mut sum1 = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut j = oj;
    while j < height - 1 {
        let mut i = oi;
        while i < i_end {
            sum1 += inb[j * width + i] as f64;
            sum2 += inb[(j + 1) * width + (i as isize + g2_offset) as usize] as f64;
            i += 2;
        }
        j += 2;
    }

    if !(sum1 > 0.0 && sum2 > 0.0) {
        return;
    }
    let gr_ratio = sum2 / sum1;

    let mut j = oj;
    while j < height - 1 {
        let mut i = oi;
        while i < i_end {
            o[j * width + i] = ((inb[j * width + i] as f64 * gr_ratio) as f32).max(0.0);
            i += 2;
        }
        j += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGGB: u32 = 0x94949494;

    #[test]
    fn pre_median_uniform_field_unchanged() {
        let (w, h) = (10usize, 10usize);
        let inp = vec![0.4_f32; w * h];
        let mut out = vec![0.0_f32; w * h];
        unsafe {
            darkroom_demosaic_pre_median(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, 1, 0.1);
        }
        assert_eq!(out, inp); // median of identical values = the value
    }

    #[test]
    fn pre_median_suppresses_green_outlier() {
        let (w, h) = (12usize, 12usize);
        let mut inp = vec![0.4_f32; w * h];
        // (5,4) is green in RGGB (row odd, col even → G); make it an outlier
        assert_eq!(raw::fc_bayer(5, 4, RGGB), 1);
        inp[5 * w + 4] = 5.0;
        let mut out = vec![0.0_f32; w * h];
        unsafe {
            darkroom_demosaic_pre_median(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, 1, 0.1);
        }
        // the outlier's window: centre matches itself only (cnt==1) →
        // med[4]-64; its 8 same-colour neighbours are 0.4, shifted +64 →
        // sorted: [0.4+64 ×8, 5.0] → med[4] = 64.4 → 0.4 (±1.5e-6: the
        // +64 round-trip loses low bits in f32, exactly as in the C)
        assert!((out[5 * w + 4] - 0.4).abs() < 1e-5, "out={}", out[5 * w + 4]);
    }

    #[test]
    fn color_smoothing_flattens_chroma_speck() {
        let (w, h) = (8usize, 8usize);
        // uniform grey RGBA, one red speck in the middle
        let mut buf: Vec<f32> = (0..w * h).flat_map(|_| [0.5_f32, 0.5, 0.5, 0.0]).collect();
        buf[4 * (3 * w + 3)] = 1.5;
        unsafe {
            darkroom_demosaic_color_smoothing(buf.as_mut_ptr(), w, h, 1);
        }
        // median of R-G over the hood is 0 → R = 0 + G = 0.5
        assert!((buf[4 * (3 * w + 3)] - 0.5).abs() < 1e-6);
        // an untouched far pixel keeps its values
        assert_eq!(buf[4 * (6 * w + 6)], 0.5);
    }

    #[test]
    fn green_eq_lavg_scales_imbalanced_green() {
        let (w, h) = (12usize, 12usize);
        // G1 lattice (odd row, even col) = 0.4; everything else 0.5
        let mut inp = vec![0.5_f32; w * h];
        for j in 0..h {
            for i in 0..w {
                if raw::fc_bayer(j as i32, i as i32, RGGB) == 1 && j % 2 == 1 {
                    inp[j * w + i] = 0.4;
                }
            }
        }
        let mut out = vec![0.0_f32; w * h];
        unsafe {
            darkroom_demosaic_green_eq_lavg(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, 10.0);
        }
        // site (3,2): diagonal neighbours are G2 (=0.5), straight are G1
        // (=0.4): m1/m2 = 1.25 → 0.4*1.25 = 0.5
        assert!((out[3 * w + 2] - 0.5).abs() < 1e-6, "out={}", out[3 * w + 2]);
        // non-green site untouched
        assert_eq!(out[3 * w + 3], inp[3 * w + 3]);
    }

    #[test]
    fn green_eq_favg_applies_global_ratio() {
        let (w, h) = (8usize, 8usize);
        // RGGB: FC(0,0)&1 == 0 → oi=1, g2_offset=-1: G1 = (even row, odd
        // col), G2 = (odd row, even col). G1=0.4, G2=0.6 → ratio 1.5.
        let mut inp = vec![0.1_f32; w * h];
        for j in 0..h {
            for i in 0..w {
                if raw::fc_bayer(j as i32, i as i32, RGGB) == 1 {
                    inp[j * w + i] = if j % 2 == 0 { 0.4 } else { 0.6 };
                }
            }
        }
        let mut out = vec![0.0_f32; w * h];
        unsafe {
            darkroom_demosaic_green_eq_favg(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB);
        }
        // G1 sites scaled by 1.5 → 0.6 (within the swept range)
        assert!((out[2 * w + 3] - 0.6).abs() < 1e-6, "out={}", out[2 * w + 3]);
        // G2 sites and non-green untouched
        assert_eq!(out[1 * w + 2], 0.6);
        assert_eq!(out[2 * w + 2], 0.1);
    }

    #[test]
    fn degenerate_roi_is_copy_only_no_panic() {
        // 4x4 is below every interior-loop minimum: all four entry points
        // must degrade to a plain copy (or stay untouched for the in-place
        // smoothing) without panicking across the FFI boundary.
        let (w, h) = (4usize, 4usize);
        let inp: Vec<f32> = (0..w * h).map(|k| 0.1 * k as f32).collect();
        let mut out = vec![-1.0_f32; w * h];
        unsafe {
            darkroom_demosaic_pre_median(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, 2, 0.1);
            assert_eq!(out, inp);
            darkroom_demosaic_green_eq_lavg(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, 0.1);
            assert_eq!(out, inp);
            // favg: 4x4 is NOT degenerate for it (its sweep fits); a single
            // row is — the C int bound height-1 made the loops no-ops
            let row: Vec<f32> = inp[0..w].to_vec();
            let mut rout = vec![-1.0_f32; w];
            darkroom_demosaic_green_eq_favg(rout.as_mut_ptr(), row.as_ptr(), w, 1, RGGB);
            assert_eq!(rout, row);
            // 2x2 four-channel buffer for the smoothing: interior empty
            let mut rgba = vec![0.25_f32; 2 * 2 * 4];
            darkroom_demosaic_color_smoothing(rgba.as_mut_ptr(), 2, 2, 3);
            for px in rgba.chunks_exact(4) {
                assert_eq!(&px[0..3], &[0.25, 0.25, 0.25]); // prefill touches only ch3
            }
        }
    }
}
