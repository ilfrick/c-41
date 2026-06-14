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

/// Monochrome "demosaic": replicate the single raw channel into RGB and
/// zero the alpha. Matches passthrough_monochrome() (passthrough.c:20).
///
/// # Safety
/// `in_buf` holds `width * height` floats; `out` holds `width * height * 4`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_passthrough_monochrome(
    out: *mut f32, in_buf: *const f32, width: usize, height: usize,
) {
    let inb = std::slice::from_raw_parts(in_buf, width * height);
    let o = std::slice::from_raw_parts_mut(out, width * height * 4);
    for (px, &v) in o.chunks_exact_mut(4).zip(inb.iter()) {
        px[0] = v;
        px[1] = v;
        px[2] = v;
        px[3] = 0.0;
    }
}

/// Debug "demosaic": place each photosite's value in its CFA colour channel,
/// zeroing the others. Matches passthrough_color() (passthrough.c:39).
///
/// # Safety
/// `in_buf` holds `width * height` floats; `out` holds `width * height * 4`;
/// `xtrans` must point to 36 valid bytes (copied unconditionally; only
/// consulted by fcol when filters == 9).
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_passthrough_color(
    out: *mut f32, in_buf: *const f32, width: usize, height: usize,
    filters: u32, xtrans: *const u8,
) {
    let inb = std::slice::from_raw_parts(in_buf, width * height);
    let o = std::slice::from_raw_parts_mut(out, width * height * 4);
    let xb = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = xb[r * 6 + c]; }
    }

    for row in 0..height {
        for col in 0..width {
            let val = inb[row * width + col];
            let offset = 4 * (row * width + col);
            let ch = raw::fcol(row as i32, col as i32, filters, &xt);
            o[offset] = 0.0;
            o[offset + 1] = 0.0;
            o[offset + 2] = 0.0;
            o[offset + 3] = 0.0;
            o[offset + ch] = val;
        }
    }
}

/// Dual-demosaic mask visualization: copy the blend mask into the alpha
/// channel of the RGBA image. Replaces the first DT_OMP_FOR_SIMD loop in
/// dual_demosaic() (dual.c:62).
///
/// # Safety
/// `high_data` holds `msize * 4` floats; `mask` holds `msize`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_dual_mask_to_alpha(
    high_data: *mut f32, mask: *const f32, msize: usize,
) {
    let hd = std::slice::from_raw_parts_mut(high_data, msize * 4);
    let m = std::slice::from_raw_parts(mask, msize);
    for (px, &mv) in hd.chunks_exact_mut(4).zip(m.iter()) {
        px[3] = mv;
    }
}

/// Dual-demosaic blend: per pixel, lerp the high-frequency demosaic towards
/// the VNG one by the detail mask — interpolatef(m, high, vng) =
/// m*(high-vng)+vng (math.h:141) — and zero the alpha. Replaces the second
/// DT_OMP_FOR_SIMD loop in dual_demosaic() (dual.c:74).
///
/// # Safety
/// `high_data`/`vng_image` hold `msize * 4` floats; `mask` holds `msize`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_dual_blend(
    high_data: *mut f32, vng_image: *const f32, mask: *const f32, msize: usize,
) {
    let hd = std::slice::from_raw_parts_mut(high_data, msize * 4);
    let vng = std::slice::from_raw_parts(vng_image, msize * 4);
    let m = std::slice::from_raw_parts(mask, msize);
    for idx in 0..msize {
        let o = idx * 4;
        for c in 0..3 {
            hd[o + c] = m[idx] * (hd[o + c] - vng[o + c]) + vng[o + c];
        }
        hd[o + 3] = 0.0;
    }
}

/// 3x3 box-average fallback demosaic: every output pixel's channel c is the
/// mean of the (clamped-to-image) 3x3 neighbourhood's photosites of colour
/// c, with negative raw values clipped to 0; channels with no contributor
/// (always alpha) divide by 1 and stay 0. Matches demosaic_box3()
/// (rcd.c:86).
///
/// # Safety
/// `in_buf` holds `width * height` floats; `out` holds `width * height * 4`;
/// `xtrans` must point to 36 valid bytes (consulted when filters == 9).
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_box3(
    out: *mut f32, in_buf: *const f32, width: usize, height: usize,
    filters: u32, xtrans: *const u8,
) {
    let inb = std::slice::from_raw_parts(in_buf, width * height);
    let o = std::slice::from_raw_parts_mut(out, width * height * 4);
    let xb = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0_u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = xb[r * 6 + c]; }
    }

    for row in 0..height as i32 {
        for col in 0..width as i32 {
            let mut sum = [0.0_f32; 4];
            let mut cnt = [0.0_f32; 4];
            for y in row - 1..row + 2 {
                for x in col - 1..col + 2 {
                    if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
                        let color = raw::fcol(y, x, filters, &xt);
                        sum[color] += inb[y as usize * width + x as usize].max(0.0);
                        cnt[color] += 1.0;
                    }
                }
            }
            let op = (row as usize * width + col as usize) * 4;
            for c in 0..4 {
                o[op + c] = sum[c] / cnt[c].max(1.0);
            }
        }
    }
}

/// PPG green-channel interpolation sweep (first DT_OMP_FOR of
/// demosaic_ppg, ppg.c:69): for rows/cols 3..-3 (optionally only a
/// `margin`+3 ring), green at R/B sites is the direction-selected
/// second-order guess clamped to the neighbour min/max. `input` is the
/// (possibly pre-median-filtered) raw plane; like the C, the cursor
/// switches to the *original* raw plane `in_orig` after the ring skip
/// (a C quirk only reachable when median filtering and a finite margin
/// combine — never with today's call sites). The C left the other two
/// colour channels uninitialized here; they are zeroed instead (every
/// pixel this loop touches has them recomputed by the red/blue sweep).
///
/// # Safety
/// `input`/`in_orig` hold `width * height` floats; `out` holds
/// `width * height * 4`; margin >= 0 (call sites pass >= 10).
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_ppg_green(
    out: *mut f32, input: *const f32, in_orig: *const f32,
    width: usize, height: usize, filters: u32, margin: i32,
) {
    if width < 7 || height < 7 {
        return; // C int bounds made the loop a no-op
    }
    let o = std::slice::from_raw_parts_mut(out, width * height * 4);
    let med = std::slice::from_raw_parts(input, width * height);
    let orig = std::slice::from_raw_parts(in_orig, width * height);
    let (w, h) = (width as i32, height as i32);
    // border == margin+3 >= 3 for any sane caller. Only margin == 0 makes the
    // ring-skip land on column w-3 (the first column the `i < w-3` bound
    // excludes), which would drop that one interpolation; no call site passes 0.
    let border = margin.saturating_add(3);

    for j in 3..h - 3 {
        let mut i = 3_i32;
        let mut bufp = (4 * w * j + 4 * 3) as usize;
        let mut inp = (w * j + 3) as usize;
        let mut src = med;
        // one-shot: for w < 2*border the C's ring skip jumped backwards and
        // looped forever; real call sites always have w >= 2*border
        let mut skipped = false;
        // C bound is `i < width - 3`: the rightmost 3 columns are left to the
        // border-interpolate pass, not touched by this green sweep.
        while i < w - 3 {
            if !skipped && i == border && j >= border && j < h - border {
                skipped = true;
                i = w - border;
                bufp = (4 * w * j + 4 * i) as usize;
                inp = (w * j + i) as usize;
                src = orig;
            }
            if i == w {
                break;
            }

            let c = raw::fc_bayer(j, i, filters);
            let mut color = [0.0_f32; 4];
            let pc = src[inp];
            if c == 0 || c == 2 {
                color[c] = pc;
                // get stuff (hopefully from cache)
                let pym = src[inp - width];
                let pym2 = src[inp - 2 * width];
                let pym3 = src[inp - 3 * width];
                let pym_ = src[inp + width];
                let pym2_ = src[inp + 2 * width];
                let pym3_ = src[inp + 3 * width];
                let pxm = src[inp - 1];
                let pxm2 = src[inp - 2];
                let pxm3 = src[inp - 3];
                let pxm_ = src[inp + 1];
                let pxm2_ = src[inp + 2];
                let pxm3_ = src[inp + 3];

                let guessx = (pxm + pc + pxm_) * 2.0 - pxm2_ - pxm2;
                let diffx = ((pxm2 - pc).abs() + (pxm2_ - pc).abs() + (pxm - pxm_).abs()) * 3.0
                    + ((pxm3_ - pxm_).abs() + (pxm3 - pxm).abs()) * 2.0;
                let guessy = (pym + pc + pym_) * 2.0 - pym2_ - pym2;
                let diffy = ((pym2 - pc).abs() + (pym2_ - pc).abs() + (pym - pym_).abs()) * 3.0
                    + ((pym3_ - pym_).abs() + (pym3 - pym).abs()) * 2.0;
                if diffx > diffy {
                    // use guessy
                    let m = pym.min(pym_);
                    let mx = pym.max(pym_);
                    color[1] = (guessy * 0.25).min(mx).max(m);
                } else {
                    let m = pxm.min(pxm_);
                    let mx = pxm.max(pxm_);
                    color[1] = (guessx * 0.25).min(mx).max(m);
                }
            } else {
                color[1] = pc;
            }
            color[3] = 0.0;

            for k in 0..4 {
                o[bufp + k] = color[k].max(0.0);
            }
            bufp += 4;
            inp += 1;
            i += 1;
        }
    }
}

/// PPG red/blue interpolation sweep (second DT_OMP_FOR of demosaic_ppg,
/// ppg.c:138), in-place on the RGBA buffer: green sites get R and B from
/// the 4-neighbourhood, R sites get B (and vice versa) from the better
/// diagonal pair. The outermost row/column only re-clamp to >= 0. The C
/// OMP version raced on neighbouring rows' fresh R/B values (schedule-
/// dependent); this serial port realizes the deterministic row-major
/// schedule. An `i >= width` guard replaces the C's latent OOB when
/// margin == 0 (no call site passes 0).
///
/// # Safety
/// `out` holds `width * height * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_ppg_redblue(
    out: *mut f32, width: usize, height: usize, filters: u32, margin: i32,
) {
    let o = std::slice::from_raw_parts_mut(out, width * height * 4);
    let (w, h) = (width as i32, height as i32);
    let w4 = 4 * width;

    for j in 0..h {
        let mut i = 0_i32;
        let mut bufp = (4 * w * j) as usize;
        let mut skipped = false; // one-shot, see ppg_green
        while i < w {
            if !skipped && i == margin && j >= margin && j < h - margin {
                skipped = true;
                i = w - margin;
                bufp = (4 * (w * j + i)) as usize;
            }
            if i >= w {
                break;
            }
            let mut color = [o[bufp], o[bufp + 1], o[bufp + 2], o[bufp + 3]];

            if j > 0 && i > 0 && i < w - 1 && j < h - 1 {
                let c = raw::fc_bayer(j, i, filters);
                if c & 1 != 0 {
                    // green pixel: red and blue from the 4-neighbourhood
                    let nt = bufp - w4;
                    let nb = bufp + w4;
                    let nl = bufp - 4;
                    let nr = bufp + 4;
                    if raw::fc_bayer(j, i + 1, filters) == 0 {
                        // red neighbour in the same row
                        color[2] = (o[nt + 2] + o[nb + 2] + 2.0 * color[1] - o[nt + 1] - o[nb + 1]) * 0.5;
                        color[0] = (o[nl] + o[nr] + 2.0 * color[1] - o[nl + 1] - o[nr + 1]) * 0.5;
                    } else {
                        // blue neighbour
                        color[0] = (o[nt] + o[nb] + 2.0 * color[1] - o[nt + 1] - o[nb + 1]) * 0.5;
                        color[2] = (o[nl + 2] + o[nr + 2] + 2.0 * color[1] - o[nl + 1] - o[nr + 1]) * 0.5;
                    }
                } else {
                    // diagonal star neighbourhood
                    let ntl = bufp - 4 - w4;
                    let ntr = bufp + 4 - w4;
                    let nbl = bufp - 4 + w4;
                    let nbr = bufp + 4 + w4;
                    // src channel: blue for red pixels, red for blue pixels
                    let s = if c == 0 { 2 } else { 0 };
                    let diff1 =
                        (o[ntl + s] - o[nbr + s]).abs() + (o[ntl + 1] - color[1]).abs() + (o[nbr + 1] - color[1]).abs();
                    let guess1 = o[ntl + s] + o[nbr + s] + 2.0 * color[1] - o[ntl + 1] - o[nbr + 1];
                    let diff2 =
                        (o[ntr + s] - o[nbl + s]).abs() + (o[ntr + 1] - color[1]).abs() + (o[nbl + 1] - color[1]).abs();
                    let guess2 = o[ntr + s] + o[nbl + s] + 2.0 * color[1] - o[ntr + 1] - o[nbl + 1];
                    color[s] = if diff1 > diff2 {
                        guess2 * 0.5
                    } else if diff1 < diff2 {
                        guess1 * 0.5
                    } else {
                        (guess1 + guess2) * 0.25
                    };
                }
            }
            for k in 0..4 {
                o[bufp + k] = color[k].max(0.0);
            }
            bufp += 4;
            i += 1;
        }
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
    fn passthrough_monochrome_replicates_channel() {
        let inp = [0.1_f32, 0.2, 0.3, 0.4];
        let mut out = vec![9.0_f32; 16];
        unsafe {
            darkroom_demosaic_passthrough_monochrome(out.as_mut_ptr(), inp.as_ptr(), 2, 2);
        }
        assert_eq!(&out[0..4], &[0.1, 0.1, 0.1, 0.0]);
        assert_eq!(&out[12..16], &[0.4, 0.4, 0.4, 0.0]);
    }

    #[test]
    fn passthrough_color_places_value_in_cfa_channel() {
        let inp = [0.1_f32, 0.2, 0.3, 0.4]; // 2x2 RGGB: R G / G B
        let mut out = vec![9.0_f32; 16];
        let xt = [0_u8; 36];
        unsafe {
            darkroom_demosaic_passthrough_color(out.as_mut_ptr(), inp.as_ptr(), 2, 2,
                                                RGGB, xt.as_ptr());
        }
        assert_eq!(&out[0..4], &[0.1, 0.0, 0.0, 0.0]); // R site
        assert_eq!(&out[4..8], &[0.0, 0.2, 0.0, 0.0]); // G site
        assert_eq!(&out[8..12], &[0.0, 0.3, 0.0, 0.0]); // G site
        assert_eq!(&out[12..16], &[0.0, 0.0, 0.4, 0.0]); // B site
    }

    #[test]
    fn dual_mask_to_alpha_and_blend() {
        let mut hd = vec![1.0_f32, 1.0, 1.0, 9.0, /*px2*/ 0.0, 0.0, 0.0, 9.0];
        let mask = [0.25_f32, 1.0];
        unsafe {
            darkroom_demosaic_dual_mask_to_alpha(hd.as_mut_ptr(), mask.as_ptr(), 2);
        }
        assert_eq!(hd[3], 0.25);
        assert_eq!(hd[7], 1.0);

        let vng = vec![0.0_f32, 0.0, 0.0, 5.0, /*px2*/ 2.0, 2.0, 2.0, 5.0];
        unsafe {
            darkroom_demosaic_dual_blend(hd.as_mut_ptr(), vng.as_ptr(), mask.as_ptr(), 2);
        }
        // px1: 0.25*(1-0)+0 = 0.25; alpha zeroed
        assert_eq!(&hd[0..4], &[0.25, 0.25, 0.25, 0.0]);
        // px2: 1.0*(0-2)+2 = 0.0
        assert_eq!(&hd[4..8], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn box3_uniform_field_recovers_all_channels() {
        // uniform 0.5 raw: every channel's box mean is 0.5 wherever a
        // contributor exists; alpha (no contributors) stays 0
        let (w, h) = (6usize, 6usize);
        let inp = vec![0.5_f32; w * h];
        let mut out = vec![9.0_f32; w * h * 4];
        let xt = [0_u8; 36];
        unsafe {
            darkroom_demosaic_box3(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, xt.as_ptr());
        }
        for px in out.chunks_exact(4) {
            assert_eq!(&px[0..3], &[0.5, 0.5, 0.5]);
            assert_eq!(px[3], 0.0);
        }
    }

    #[test]
    fn box3_corner_clamps_window() {
        // 2x2 RGGB image, corner (0,0): window covers all four photosites
        // exactly once → R=in[0], G=(in[1]+in[2])/2, B=in[3]
        let inp = [0.1_f32, 0.2, 0.4, 0.8];
        let mut out = vec![9.0_f32; 16];
        let xt = [0_u8; 36];
        unsafe {
            darkroom_demosaic_box3(out.as_mut_ptr(), inp.as_ptr(), 2, 2, RGGB, xt.as_ptr());
        }
        assert!((out[0] - 0.1).abs() < 1e-7);
        assert!((out[1] - 0.3).abs() < 1e-7);
        assert!((out[2] - 0.8).abs() < 1e-7);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn ppg_uniform_field_demosaics_flat() {
        // uniform raw: green guess = (g+pc+g)*2 - g - g clamped to [g,g] = g;
        // red/blue interpolation likewise returns the channel value → every
        // interior pixel ends up (v, v, v, 0)
        let (w, h) = (12usize, 12usize);
        let v = 0.5_f32;
        let inp = vec![v; w * h];
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe {
            // sentinel margin like the PPG call site: ring skip disabled
            darkroom_demosaic_ppg_green(out.as_mut_ptr(), inp.as_ptr(), inp.as_ptr(),
                                        w, h, RGGB, 100000);
            darkroom_demosaic_ppg_redblue(out.as_mut_ptr(), w, h, RGGB, 100000);
        }
        // interior of the green sweep ∩ interior of the r/b sweep
        for j in 3..h - 3 {
            for i in 3..w - 3 {
                let p = 4 * (j * w + i);
                for c in 0..3 {
                    assert!((out[p + c] - v).abs() < 1e-6, "({j},{i}) c{c} = {}", out[p + c]);
                }
                assert_eq!(out[p + 3], 0.0);
            }
        }
    }

    #[test]
    fn ppg_green_clamps_to_neighbour_range() {
        // spike a red site; its green estimate must stay within the
        // min/max of the chosen direction's direct green neighbours
        let (w, h) = (12usize, 12usize);
        let mut inp = vec![0.5_f32; w * h];
        inp[4 * w + 4] = 8.0; // red site in RGGB
        let mut out = vec![0.0_f32; w * h * 4];
        unsafe {
            darkroom_demosaic_ppg_green(out.as_mut_ptr(), inp.as_ptr(), inp.as_ptr(),
                                        w, h, RGGB, 100000);
        }
        let p = 4 * (4 * w + 4);
        assert_eq!(out[p], 8.0); // red carried through
        assert!((out[p + 1] - 0.5).abs() < 1e-6, "green={}", out[p + 1]); // clamped
    }

    #[test]
    fn ppg_green_leaves_right_border_untouched() {
        // C bound is `i < width - 3`: the rightmost 3 columns belong to the
        // border-interpolate pass and must NOT be written by the green sweep.
        // Use a non-uniform field so directional interp != any sentinel.
        let (w, h) = (16usize, 16usize);
        let inp: Vec<f32> = (0..w * h).map(|k| (k % 7) as f32 * 0.13).collect();
        const SENTINEL: f32 = -999.0;
        let mut out = vec![SENTINEL; w * h * 4];
        unsafe {
            darkroom_demosaic_ppg_green(out.as_mut_ptr(), inp.as_ptr(), inp.as_ptr(),
                                        w, h, RGGB, 100000);
        }
        for j in 3..h - 3 {
            for i in w - 3..w {
                let p = 4 * (j * w + i);
                assert_eq!(out[p + 1], SENTINEL,
                           "right-border green ({j},{i}) was overwritten: {}", out[p + 1]);
            }
            // and the last interior column it *should* write is w-4
            assert_ne!(out[4 * (j * w + (w - 4)) + 1], SENTINEL,
                       "interior green ({j},{}) not written", w - 4);
        }
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
