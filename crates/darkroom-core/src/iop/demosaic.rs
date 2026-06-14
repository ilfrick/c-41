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

/// VNG border colour interpolation: the first DT_OMP_FOR of
/// _vng_lininterpolate (vng.c:32). For the outer 1-pixel frame (and, for
/// interior rows, only columns 0 and width-1 — the C `col == 1` jump) each
/// output channel c is the mean of the clamped 3x3 neighbourhood's
/// photosites of colour c, falling back to the clamped centre value when c
/// is the site's own colour or has no contributor. `colors` is 4 for Bayer
/// (FC can yield 0..3), 3 for X-Trans; the extra Bayer channel always takes
/// the centre fallback (its count stays 0).
///
/// # Safety
/// `in_buf` holds `width*height` floats; `out` holds `width*height*4`;
/// `xtrans` points to 36 valid bytes (consulted when filters == 9).
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_vng_border(
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
    let (w, h) = (width as i32, height as i32);
    let colors = if filters == 9 { 3 } else { 4 };

    for row in 0..h {
        let mut col = 0_i32;
        while col < w {
            // interior rows: skip the middle, do only the left/right edge
            if col == 1 && row >= 1 && row < h - 1 {
                col = w - 1;
            }
            let mut sum = [0.0_f32; 4];
            let mut count = [0_u32; 4];
            let mut y = row - 1;
            while y != row + 2 {
                let mut x = col - 1;
                while x != col + 2 {
                    if y >= 0 && x >= 0 && y < h && x < w {
                        let f = raw::fcol(y, x, filters, &xt);
                        sum[f] += inb[(y * w + x) as usize].max(0.0);
                        count[f] += 1;
                    }
                    x += 1;
                }
                y += 1;
            }
            let f = raw::fcol(row, col, filters, &xt);
            let centre = inb[(row * w + col) as usize].max(0.0);
            let base = 4 * (row * w + col) as usize;
            for c in 0..colors {
                o[base + c] = if c != f && count[c] != 0 {
                    sum[c] / count[c] as f32
                } else {
                    centre
                };
            }
            col += 1;
        }
    }
}

/// VNG threshold-gradient interior interpolation: the second DT_OMP_FOR of
/// _vng_lininterpolate (vng.c:105), driven by the C-built `lookup` table
/// (flat int[16][16][32]; per [row%size][col%size]: [0]=np, then np
/// (offset, weight, colour) triples, then (colours-1) (colour, weight_sum)
/// pairs, then the centre colour). Each interior pixel sums weighted
/// neighbours per colour, divides by the colour's total weight, copies the
/// centre's own raw value into its channel, then clips all four channels to
/// >= 0 (dt_vector_clipneg). `border` gates the optional ring skip — VNG
/// passes 1000000 (disabled) for Bayer, pad_tile for X-Trans.
///
/// # Safety
/// `in_buf` holds `width*height` floats; `out` holds `width*height*4`;
/// `lookup` points to 16*16*32 valid i32.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_vng_lookup(
    out: *mut f32, in_buf: *const f32, width: usize, height: usize,
    filters: u32, border: i32, lookup: *const i32,
) {
    if width < 3 || height < 3 {
        return; // C: `row < height - 1` / `col < width - 1` empty
    }
    let inb = std::slice::from_raw_parts(in_buf, width * height);
    let o = std::slice::from_raw_parts_mut(out, width * height * 4);
    let lut = std::slice::from_raw_parts(lookup, 16 * 16 * 32);
    let (w, h) = (width as i32, height as i32);
    let colors = if filters == 9 { 3 } else { 4 };
    let size = if filters == 9 { 6 } else { 16 };

    for row in 1..h - 1 {
        let mut col = 1_i32;
        let mut skipped = false; // one-shot, see ppg_green
        while col < w - 1 {
            if !skipped && col == border && row >= border && row < h - border {
                skipped = true;
                col = w - border;
            }
            if col == w {
                break; // C guard for border == 0 (no real call site)
            }
            let lbase = (((row % size) * 16 + (col % size)) as usize) * 32;
            let np = lut[lbase] as usize;
            let inp = (row * w + col) as isize;
            let mut sum = [0.0_f32; 4];
            let mut k = lbase + 1;
            for _ in 0..np {
                let offset = lut[k] as isize;
                let weight = lut[k + 1] as f32;
                let color = lut[k + 2] as usize;
                sum[color] += inb[(inp + offset) as usize].max(0.0) * weight;
                k += 3;
            }
            let base = 4 * (row * w + col) as usize;
            // (colors-1) interpolated channels: buf[c] = sum[c] / weight_sum[c].
            // The Bayer 4th channel (FC only yields 0..2) has weight_sum 0, so
            // 0/0 = NaN here — squashed to 0 by the final clamp, exactly as the
            // C does via dt_vector_clipneg (max picks the non-NaN operand).
            for _ in 0..colors - 1 {
                let c = lut[k] as usize;
                let wsum = lut[k + 1] as f32;
                o[base + c] = sum[c] / wsum;
                k += 2;
            }
            // centre's own colour gets the raw value, then clip all 4 to >= 0
            let f = lut[k] as usize;
            o[base + f] = inb[inp as usize];
            for c in 0..4 {
                o[base + c] = o[base + c].max(0.0);
            }
            col += 1;
        }
    }
}

/// VNG output finishing pass (the `finish:` DT_OMP_FOR of vng_interpolate,
/// vng.c:265): when `mix_greens` (Bayer with separated G1/G2), average the
/// two green channels into channel 1 and zero channel 3; then clip all four
/// channels of every pixel to >= 0 (dt_vector_clipneg). `npixels` is
/// width*height (the count of RGBA pixels).
///
/// # Safety
/// `out` holds `npixels * 4` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_vng_finish(
    out: *mut f32, npixels: usize, mix_greens: i32,
) {
    let o = std::slice::from_raw_parts_mut(out, npixels * 4);
    for px in o.chunks_exact_mut(4) {
        if mix_greens != 0 {
            px[1] = 0.5 * (px[1] + px[3]);
            px[3] = 0.0;
        }
        for c in 0..4 {
            px[c] = px[c].max(0.0);
        }
    }
}

/// VNG gradient interpolation, one image row — the dcraw threshold-based
/// variable-number-of-gradients kernel (the `DT_OMP_FOR(private(ip))` col
/// loop of vng_interpolate, vng.c:201). For each interior column (2..w-2)
/// it walks the C-built `code_row` stream (selected by col%pcol): a list of
/// (offset1, offset2, weight, gradient-bits…, -1) terms terminated by
/// INT_MAX, then 8 (neighbour-offset, chood-colour) pairs. It accumulates 8
/// directional gradients, thresholds at gmin + gmax*0.5, averages the
/// qualifying neighbours, and writes the refined RGBA pixel into `brow2`
/// (the caller's ring-buffer row; the C defers the copy to `out` by two
/// rows). Reads `out` read-only ⇒ columns are independent, so this serial
/// sweep equals the original OMP schedule. All stream offsets are signed
/// (they reach ±2 rows/cols), so index arithmetic is done in `isize`.
///
/// # Safety
/// `out` holds width*height*4 floats; `brow2` holds width*4; `xtrans`
/// points to 36 bytes; `code_row[0..pcol]` are valid streams built by the C
/// precalc (INT_MAX-terminated term list + 8*2 chood ints).
#[no_mangle]
pub unsafe extern "C" fn darkroom_demosaic_vng_gradient_row(
    out: *const f32, brow2: *mut f32,
    width: usize, height: usize, row: i32,
    filters4: u32, xtrans: *const u8,
    colors: i32, code_row: *const *const i32, pcol: i32,
) {
    let o = std::slice::from_raw_parts(out, width * height * 4);
    let b2 = std::slice::from_raw_parts_mut(brow2, width * 4);
    let xb = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 { xt[r][c] = xb[r * 6 + c]; }
    }
    let w = width as i32;
    let pcolu = pcol as usize;

    for col in 2..w - 2 {
        let cs = *code_row.add((col as usize) % pcolu); // this pixel's stream
        let pib = (4 * (row as usize * width + col as usize)) as isize;
        let mut gval = [0.0_f32; 8];

        // --- gradient accumulation: walk terms until INT_MAX ---
        let mut p: isize = 0;
        loop {
            let g0 = *cs.offset(p);
            if g0 == i32::MAX {
                break;
            }
            let off2 = *cs.offset(p + 1);
            let weight = *cs.offset(p + 2) as f32;
            let diff = (o[(pib + g0 as isize) as usize] - o[(pib + off2 as isize) as usize]).abs()
                * weight;
            gval[*cs.offset(p + 3) as usize] += diff;
            p += 5;
            let gm1 = *cs.offset(p - 1);
            if gm1 == -1 {
                continue;
            }
            gval[gm1 as usize] += diff;
            loop {
                let gg = *cs.offset(p);
                p += 1;
                if gg == -1 {
                    break;
                }
                gval[gg as usize] += diff;
            }
        }
        p += 1; // skip INT_MAX → chood section

        // --- choose a threshold ---
        let mut gmin = gval[0];
        let mut gmax = gval[0];
        for g in 1..8 {
            if gmin > gval[g] { gmin = gval[g]; }
            if gmax < gval[g] { gmax = gval[g]; }
        }
        let bo = col as usize * 4;
        if gmax == 0.0 {
            b2[bo..bo + 4].copy_from_slice(&o[pib as usize..pib as usize + 4]);
            continue;
        }
        let thold = gmin + gmax * 0.5;

        // --- average the qualifying neighbours (chood section) ---
        let mut sum = [0.0_f32; 4];
        let color = raw::fcol(row, col, filters4, &xt);
        let mut num = 0_i32;
        for g in 0..8 {
            let c0 = *cs.offset(p) as isize; // neighbour pixel offset
            let c1 = *cs.offset(p + 1); // chood-colour value (0 = none)
            p += 2;
            if gval[g] <= thold {
                for c in 0..colors as usize {
                    if c == color && c1 != 0 {
                        sum[c] += (o[(pib + c as isize) as usize]
                            + o[(pib + c1 as isize) as usize])
                            * 0.5;
                    } else {
                        sum[c] += o[(pib + c0 + c as isize) as usize];
                    }
                }
                num += 1;
            }
        }

        // --- write the refined pixel ---
        for c in 0..colors as usize {
            let mut tot = o[(pib + color as isize) as usize];
            if c != color {
                tot += (sum[c] - sum[color]) / num as f32;
            }
            b2[bo + c] = tot;
        }
    }
}

const CAPTURE_KERNEL_ALIGN: usize = 32; // capture.c:42
const CAPTURE_YMIN: f32 = 0.001; // capture.c:45
// gd->gauss_coeffs is (UCHAR_MAX+1) * CAPTURE_KERNEL_ALIGN floats (capture.c:1096)
const CAPTURE_KERNELS_LEN: usize = 256 * CAPTURE_KERNEL_ALIGN;

/// Capture-sharpen separable-radius convolution value at pixel `i` (the
/// shared inner of _blur_mul/_blur_div, capture.c:535). `kern` is the
/// per-pixel kernel slice; the interior fast path reproduces the C's
/// hand-unrolled symmetric sums verbatim (operand order preserved); the
/// border path falls back to the general `kern[5*|ir|+|ic|]` gather.
#[inline(always)]
fn capture_blur_val(
    inb: &[f32], i: usize, kern: &[f32], small: bool, bd: i32,
    col: i32, row: i32, w1: i32, height: i32,
) -> f32 {
    let (w1i, w2, w3, w4) = (w1 as isize, 2 * w1 as isize, 3 * w1 as isize, 4 * w1 as isize);
    if col >= bd && row >= bd && col < w1 - bd && row < height - bd {
        let di = i as isize;
        let g = |off: isize| inb[(di + off) as usize];
        if small {
            kern[5 + 2] * (g(-w2 - 1) + g(-w2 + 1) + g(-w1i - 2) + g(-w1i + 2) + g(w1i - 2) + g(w1i + 2) + g(w2 - 1) + g(w2 + 1))
                + kern[2] * (g(-w2) + g(-2) + g(2) + g(w2))
                + kern[5 + 1] * (g(-w1i - 1) + g(-w1i + 1) + g(w1i - 1) + g(w1i + 1))
                + kern[1] * (g(-w1i) + g(-1) + g(1) + g(w1i))
                + kern[0] * g(0)
        } else {
            kern[10 + 4] * (g(-w4 - 2) + g(-w4 + 2) + g(-w2 - 4) + g(-w2 + 4) + g(w2 - 4) + g(w2 + 4) + g(w4 - 2) + g(w4 + 2))
                + kern[5 + 4] * (g(-w4 - 1) + g(-w4 + 1) + g(-w1i - 4) + g(-w1i + 4) + g(w1i - 4) + g(w1i + 4) + g(w4 - 1) + g(w4 + 1))
                + kern[4] * (g(-w4) + g(-4) + g(4) + g(w4))
                + kern[15 + 3] * (g(-w3 - 3) + g(-w3 + 3) + g(w3 - 3) + g(w3 + 3))
                + kern[10 + 3] * (g(-w3 - 2) + g(-w3 + 2) + g(-w2 - 3) + g(-w2 + 3) + g(w2 - 3) + g(w2 + 3) + g(w3 - 2) + g(w3 + 2))
                + kern[5 + 3] * (g(-w3 - 1) + g(-w3 + 1) + g(-w1i - 3) + g(-w1i + 3) + g(w1i - 3) + g(w1i + 3) + g(w3 - 1) + g(w3 + 1))
                + kern[3] * (g(-w3) + g(-3) + g(3) + g(w3))
                + kern[10 + 2] * (g(-w2 - 2) + g(-w2 + 2) + g(w2 - 2) + g(w2 + 2))
                + kern[5 + 2] * (g(-w2 - 1) + g(-w2 + 1) + g(-w1i - 2) + g(-w1i + 2) + g(w1i - 2) + g(w1i + 2) + g(w2 - 1) + g(w2 + 1))
                + kern[2] * (g(-w2) + g(-2) + g(2) + g(w2))
                + kern[5 + 1] * (g(-w1i - 1) + g(-w1i + 1) + g(w1i - 1) + g(w1i + 1))
                + kern[1] * (g(-w1i) + g(-1) + g(1) + g(w1i))
                + kern[0] * g(0)
        }
    } else {
        let mut val = 0.0_f32;
        for ir in -bd..=bd {
            let irow = row + ir;
            if irow >= 0 && irow < height {
                for ic in -bd..=bd {
                    let icol = col + ic;
                    if icol >= 0 && icol < w1 {
                        val += kern[(5 * ir.abs() + ic.abs()) as usize]
                            * inb[(irow as isize * w1 as isize + icol as isize) as usize];
                    }
                }
            }
        }
        val
    }
}

/// Capture-sharpen multiply-blur (_blur_mul, capture.c:523): where blend>0,
/// multiply `out[i]` by the radius-selected Gaussian convolution of `in`.
/// `idx_small` is _sigma_to_index(CAPTURE_SMALL) (passed from C); pixels with
/// table[i] < idx_small use the 2-pixel kernel, others the 4-pixel one.
///
/// # Safety
/// `in_buf`/`out`/`blend` hold `w1*height` floats; `table` `w1*height` bytes;
/// `kernels` is the 256*32-float gauss_coeffs buffer.
#[no_mangle]
pub unsafe extern "C" fn darkroom_capture_blur_mul(
    in_buf: *const f32, out: *mut f32, blend: *const f32,
    kernels: *const f32, table: *const u8,
    w1: i32, height: i32, idx_small: u8,
) {
    let n = w1 as usize * height as usize;
    let inb = std::slice::from_raw_parts(in_buf, n);
    let o = std::slice::from_raw_parts_mut(out, n);
    let bl = std::slice::from_raw_parts(blend, n);
    let tab = std::slice::from_raw_parts(table, n);
    let kerns = std::slice::from_raw_parts(kernels, CAPTURE_KERNELS_LEN);

    for row in 0..height {
        for col in 0..w1 {
            let i = (row as usize) * (w1 as usize) + col as usize;
            if bl[i] > 0.0 {
                let kb = CAPTURE_KERNEL_ALIGN * tab[i] as usize;
                let kern = &kerns[kb..kb + CAPTURE_KERNEL_ALIGN];
                let small = tab[i] < idx_small;
                let bd = if small { 2 } else { 4 };
                let val = capture_blur_val(inb, i, kern, small, bd, col, row, w1, height);
                o[i] *= val;
            }
        }
    }
}

/// Capture-sharpen divide-blur (_blur_div, capture.c:604): where blend>0,
/// set `out[i] = luminance[i] / max(val, CAPTURE_YMIN)` with `val` the same
/// radius-selected convolution as _blur_mul.
///
/// # Safety
/// As _blur_mul, plus `luminance` holds `w1*height` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_capture_blur_div(
    in_buf: *const f32, out: *mut f32, luminance: *const f32, blend: *const f32,
    kernels: *const f32, table: *const u8,
    w1: i32, height: i32, idx_small: u8,
) {
    let n = w1 as usize * height as usize;
    let inb = std::slice::from_raw_parts(in_buf, n);
    let o = std::slice::from_raw_parts_mut(out, n);
    let lum = std::slice::from_raw_parts(luminance, n);
    let bl = std::slice::from_raw_parts(blend, n);
    let tab = std::slice::from_raw_parts(table, n);
    let kerns = std::slice::from_raw_parts(kernels, CAPTURE_KERNELS_LEN);

    for row in 0..height {
        for col in 0..w1 {
            let i = (row as usize) * (w1 as usize) + col as usize;
            if bl[i] > 0.0 {
                let kb = CAPTURE_KERNEL_ALIGN * tab[i] as usize;
                let kern = &kerns[kb..kb + CAPTURE_KERNEL_ALIGN];
                let small = tab[i] < idx_small;
                let bd = if small { 2 } else { 4 };
                let val = capture_blur_val(inb, i, kern, small, bd, col, row, w1, height);
                o[i] = lum[i] / val.max(CAPTURE_YMIN);
            }
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

    // Faithful re-creation of the C VNG lookup-table builder (vng.c:77-103),
    // so the Rust consumer is validated against a C-equivalent table.
    fn build_vng_lookup(width: i32, filters: u32, xt: &[[u8; 6]; 6]) -> Vec<i32> {
        let colors = if filters == 9 { 3 } else { 4 };
        let size = if filters == 9 { 6 } else { 16 };
        let mut lut = vec![0_i32; 16 * 16 * 32];
        for row in 0..size {
            for col in 0..size {
                let lb = ((row * 16 + col) as usize) * 32;
                let f = raw::fcol(row, col, filters, xt);
                let mut wsum = [0_i32; 4];
                let mut k = lb + 1;
                for y in -1..=1 {
                    for x in -1..=1 {
                        let weight = 1_i32 << (((y == 0) as i32) + ((x == 0) as i32));
                        let color = raw::fcol(row + y, col + x, filters, xt);
                        if color == f {
                            continue;
                        }
                        lut[k] = width * y + x;
                        lut[k + 1] = weight;
                        lut[k + 2] = color as i32;
                        wsum[color] += weight;
                        k += 3;
                    }
                }
                lut[lb] = ((k - lb) / 3) as i32;
                for c in 0..colors {
                    if c != f {
                        lut[k] = c as i32;
                        lut[k + 1] = wsum[c];
                        k += 2;
                    }
                }
                lut[k] = f as i32;
            }
        }
        lut
    }

    #[test]
    fn vng_border_uniform_field_is_flat() {
        // uniform raw: every neighbour average and the centre fallback equal v,
        // so each written channel (incl. the Bayer 4th, count==0 → centre) is v
        let (w, h) = (10usize, 10usize);
        let v = 0.375_f32;
        let inp = vec![v; w * h];
        let mut out = vec![-1.0_f32; w * h * 4];
        let xt = [[0u8; 6]; 6];
        unsafe {
            darkroom_demosaic_vng_border(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, xt.as_ptr() as *const u8);
        }
        // border frame + the single-pixel left/right edges of interior rows
        for (j, i) in (0..h).flat_map(|j| (0..w).map(move |i| (j, i)))
            .filter(|&(j, i)| j == 0 || j == h - 1 || i == 0 || i == w - 1)
        {
            let p = 4 * (j * w + i);
            for c in 0..4 {
                assert!((out[p + c] - v).abs() < 1e-6, "border ({j},{i}) c{c} = {}", out[p + c]);
            }
        }
    }

    #[test]
    fn vng_lookup_uniform_field_is_flat() {
        // sum[c] = v * weight_sum[c] ⇒ buf[c] = v; centre channel = v too
        let (w, h) = (12usize, 12usize);
        let v = 0.6_f32;
        let inp = vec![v; w * h];
        let mut out = vec![0.0_f32; w * h * 4];
        let xt = [[0u8; 6]; 6];
        let lut = build_vng_lookup(w as i32, RGGB, &xt);
        unsafe {
            // border = 1000000 like the Bayer VNG call site: ring skip disabled
            darkroom_demosaic_vng_lookup(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, 1000000, lut.as_ptr());
        }
        for j in 1..h - 1 {
            for i in 1..w - 1 {
                let p = 4 * (j * w + i);
                for c in 0..3 {
                    assert!((out[p + c] - v).abs() < 1e-6, "({j},{i}) c{c} = {}", out[p + c]);
                }
            }
        }
    }

    #[test]
    fn vng_lookup_negative_centre_clipped() {
        // a single negative raw site: its own colour channel is the raw value
        // pre-clip, so it must come out clamped to 0
        let (w, h) = (12usize, 12usize);
        let mut inp = vec![0.5_f32; w * h];
        inp[5 * w + 5] = -3.0; // RGGB: (5,5) is blue
        let mut out = vec![0.0_f32; w * h * 4];
        let xt = [[0u8; 6]; 6];
        let lut = build_vng_lookup(w as i32, RGGB, &xt);
        unsafe {
            darkroom_demosaic_vng_lookup(out.as_mut_ptr(), inp.as_ptr(), w, h, RGGB, 1000000, lut.as_ptr());
        }
        let f = raw::fc_bayer(5, 5, RGGB); // own-colour channel
        let p = 4 * (5 * w + 5);
        assert_eq!(out[p + f], 0.0, "centre own-colour not clipped: {}", out[p + f]);
        for c in 0..4 {
            assert!(out[p + c] >= 0.0, "channel {c} negative: {}", out[p + c]);
        }
    }

    #[test]
    fn vng_finish_mixes_greens_and_clips() {
        // pixel 0: greens 0.2 & 0.6 → 0.4 mix, ch3 zeroed; negative R clipped.
        // pixel 1: same data but mix_greens off → greens untouched, only clip.
        let mut a = vec![-0.5_f32, 0.2, 0.3, 0.6, -0.5, 0.2, 0.3, 0.6];
        unsafe { darkroom_demosaic_vng_finish(a.as_mut_ptr(), 2, 1); }
        assert_eq!(a[0], 0.0); // clipped
        assert!((a[1] - 0.4).abs() < 1e-6, "green mix = {}", a[1]);
        assert_eq!(a[3], 0.0); // G2 zeroed
        assert!((a[5] - 0.4).abs() < 1e-6); // second pixel mixed too

        let mut b = vec![-0.5_f32, 0.2, 0.3, 0.6];
        unsafe { darkroom_demosaic_vng_finish(b.as_mut_ptr(), 1, 0); }
        assert_eq!(b[0], 0.0); // clipped
        assert_eq!(b[1], 0.2); // green untouched
        assert_eq!(b[3], 0.6); // ch3 untouched (no mix)
    }

    // One synthetic VNG code stream exercising the multi-gradient inner walk:
    // a single term whose grad-bitmask covers all 8 gradients, then INT_MAX,
    // then 8 chood pairs (g0..6 → self offset 0; g7 → offset +4). cs[0]=off1,
    // cs[1]=off2 (the two pixels differenced into `diff`).
    fn synthetic_vng_code() -> Vec<i32> {
        let mut cs = vec![8, -8, 1, 0, 1, 2, 3, 4, 5, 6, 7, -1, i32::MAX];
        for _ in 0..7 { cs.push(0); cs.push(0); } // g0..6: (self, none)
        cs.push(4); cs.push(0); // g7: neighbour at +4, no chood colour
        cs
    }

    #[test]
    fn vng_gradient_uniform_field_copies_pixel() {
        // uniform ⇒ every diff 0 ⇒ all gval 0 ⇒ gmax==0 ⇒ copy pix through
        let (w, h) = (8usize, 8usize);
        let v = 0.42_f32;
        let out = vec![v; w * h * 4];
        let mut brow2 = vec![-1.0_f32; w * 4];
        let xt = [[0u8; 6]; 6];
        let cs = synthetic_vng_code();
        let code_row = [cs.as_ptr()];
        unsafe {
            darkroom_demosaic_vng_gradient_row(out.as_ptr(), brow2.as_mut_ptr(), w, h, 4,
                RGGB, xt.as_ptr() as *const u8, 4, code_row.as_ptr(), 1);
        }
        for col in 2..w - 2 {
            for c in 0..4 {
                assert!((brow2[col * 4 + c] - v).abs() < 1e-6, "col {col} c{c} = {}", brow2[col * 4 + c]);
            }
        }
    }

    #[test]
    fn vng_gradient_averages_qualifying_neighbours() {
        // Hand-traced: term gives all 8 gradients gval=1.0 (diff=|1-0|*1), so
        // gmin=gmax=1, thold=1.5, all qualify (num=8). chood: g0..6 add the
        // pixel itself, g7 adds the +4 neighbour. color=0 at (4,4) for RGGB.
        let (w, h) = (8usize, 8usize);
        let mut out = vec![0.0_f32; w * h * 4];
        let pib = 4 * (4 * w + 4); // 144
        out[pib + 8] = 1.0; // off1=8 → diff numerator
        out[pib - 8] = 0.0; // off2=-8
        out[pib..pib + 4].copy_from_slice(&[0.2, 0.5, 0.1, 0.9]); // pix
        out[pib + 4..pib + 8].copy_from_slice(&[0.6, 0.0, 0.8, 0.3]); // +4 neighbour
        let mut brow2 = vec![0.0_f32; w * 4];
        let xt = [[0u8; 6]; 6];
        let cs = synthetic_vng_code();
        let code_row = [cs.as_ptr()];
        unsafe {
            darkroom_demosaic_vng_gradient_row(out.as_ptr(), brow2.as_mut_ptr(), w, h, 4,
                RGGB, xt.as_ptr() as *const u8, 4, code_row.as_ptr(), 1);
        }
        // sum[c] = 7*pix[c] + neighbour[c]; out[c]=pix[0]+(sum[c]-sum[0])/8 (c!=0)
        let expect = [0.2_f32, 0.3875, 0.1375, 0.775];
        let bo = 4 * 4;
        for c in 0..4 {
            assert!((brow2[bo + c] - expect[c]).abs() < 1e-6, "c{c} = {} want {}", brow2[bo + c], expect[c]);
        }
    }

    #[test]
    fn capture_blur_mul_div_identity_kernel() {
        // kernel with only the centre weight (kern[0]=1) ⇒ val == in[i] on
        // both the interior fast path and the border gather path.
        let (w1, h) = (8i32, 8i32);
        let n = (w1 * h) as usize;
        let inb: Vec<f32> = (0..n).map(|k| 0.1 * k as f32 + 0.5).collect();
        let mut kernels = vec![0.0_f32; 256 * 32];
        kernels[0] = 1.0;
        let mut table = vec![0u8; n];
        table[5] = 0; // stays < idx_small
        let mut blend = vec![1.0_f32; n];
        blend[10] = 0.0; // gated off → untouched

        // _blur_mul: out *= val(=in)
        let mut out_mul = vec![2.0_f32; n];
        unsafe {
            darkroom_capture_blur_mul(inb.as_ptr(), out_mul.as_mut_ptr(), blend.as_ptr(),
                kernels.as_ptr(), table.as_ptr(), w1, h, 1);
        }
        for i in 0..n {
            let want = if blend[i] > 0.0 { 2.0 * inb[i] } else { 2.0 };
            assert!((out_mul[i] - want).abs() < 1e-5, "mul[{i}] = {} want {}", out_mul[i], want);
        }

        // _blur_div: out = lum / max(val(=in), YMIN)
        let lum: Vec<f32> = (0..n).map(|k| 0.3 * k as f32).collect();
        let mut out_div = vec![-1.0_f32; n];
        unsafe {
            darkroom_capture_blur_div(inb.as_ptr(), out_div.as_mut_ptr(), lum.as_ptr(), blend.as_ptr(),
                kernels.as_ptr(), table.as_ptr(), w1, h, 1);
        }
        for i in 0..n {
            if blend[i] > 0.0 {
                let want = lum[i] / inb[i].max(0.001);
                assert!((out_div[i] - want).abs() < 1e-5, "div[{i}] = {} want {}", out_div[i], want);
            } else {
                assert_eq!(out_div[i], -1.0); // untouched
            }
        }
    }

    #[test]
    fn capture_blur_large_radius_branch() {
        // Force the bd=4 large-radius path (table[i]=200 >= idx_small=1) and
        // verify the identity kernel still yields val==in[i] through both the
        // interior fast path and the wider border gather (needs >=10px to have
        // interior pixels with bd=4).
        let (w1, h) = (12i32, 12i32);
        let n = (w1 * h) as usize;
        let inb: Vec<f32> = (0..n).map(|k| 0.05 * k as f32 + 0.3).collect();
        let mut kernels = vec![0.0_f32; 256 * 32];
        kernels[200 * 32] = 1.0; // centre weight of kernel index 200
        let table = vec![200u8; n];
        let blend = vec![1.0_f32; n];
        let mut out = vec![3.0_f32; n];
        unsafe {
            darkroom_capture_blur_mul(inb.as_ptr(), out.as_mut_ptr(), blend.as_ptr(),
                kernels.as_ptr(), table.as_ptr(), w1, h, 1);
        }
        for i in 0..n {
            assert!((out[i] - 3.0 * inb[i]).abs() < 1e-5, "large[{i}] = {} want {}", out[i], 3.0 * inb[i]);
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
