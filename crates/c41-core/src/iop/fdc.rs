//! Frequency-Domain Chroma (FDC) X-Trans demosaic — whole-function port of
//! `xtrans_fdc_interpolate` (src/iop/demosaicing/xtrans.c:62).
//!
//! Shares the Markesteijn front-end (tile copy, gmin/gmax, green/red-blue/2x2
//! interpolation, YPbPr derivatives, homogeneity) with [`super::markesteijn`],
//! but: it is always single-pass (ndir = 4), the tile copy keeps raw (un-clamped)
//! input and also fills the `i_src` luma plane, the solitary-green and 2x2 passes
//! differ slightly, and the chroma comes from a frequency-domain stage using the
//! precomputed complex tables in [`super::fdc_tables`] (`harr` convolution,
//! `modarr` modulators, `Minv` demodulation) blended with the Markesteijn luma.
//! Same flat-index modelling as markesteijn.rs (`rfx[k][ch]` = `rgb[rfx+k*3+ch]`).

use crate::iop::fdc_tables::{Cf, HARR, MINV, MODARR};
use crate::raw;

const TS: usize = 122; // DT_FDC_TS

#[inline(always)]
fn clamps(a: f32, l: f32, h: f32) -> f32 {
    if a > l {
        if a < h { a } else { h }
    } else {
        l
    }
}

#[inline(always)]
fn translate(n: i32, size: i32) -> i32 {
    if n >= size { 2 * size - n - 2 } else { n.abs() }
}

#[inline(always)]
fn sqrf(x: f32) -> f32 {
    x * x
}

#[inline(always)]
fn pix_sort(a: &mut f32, b: &mut f32) {
    if *a > *b {
        std::mem::swap(a, b);
    }
}

/// FDC X-Trans interpolation. `hybrid0`/`hybrid1` are the C `hybrid_fdc[2]`
/// (computed C-side from iso vs the fdc_xover_iso config): {1,0} = hybrid,
/// {0,1} = pure FDC. Writes the RGBA `out`.
///
/// # Safety
/// `out` holds `width*height*4` floats; `in_buf` holds `width*height`; `xtrans`
/// points to 36 valid bytes.
#[no_mangle]
pub unsafe extern "C" fn darkroom_xtrans_fdc(
    out: *mut f32,
    in_buf: *const f32,
    width: i32,
    height: i32,
    xtrans: *const u8,
    hybrid0: f32,
    hybrid1: f32,
) {
    let w = width as usize;
    let inb = std::slice::from_raw_parts(in_buf, w * height as usize);
    let o = std::slice::from_raw_parts_mut(out, w * height as usize * 4);
    let xb = std::slice::from_raw_parts(xtrans, 36);
    let mut xt = [[0u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            xt[r][c] = xb[r * 6 + c];
        }
    }
    let fc = |row: i32, col: i32| -> i32 { raw::fc_xtrans(row, col, &xt) as i32 };

    let ts = TS as i32;
    let tts = TS * TS;
    const ORTH: [i32; 12] = [1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1];
    const PATT: [[i32; 16]; 2] = [
        [0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0],
        [0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1],
    ];
    let dir = [1i32, ts, ts + 1, ts - 1];
    const DIRECTIONALITY: [f32; 8] = [1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5];
    let ndir: usize = 4;
    let pad_tile = 13i32;

    // hex map + solitary-green location
    let mut allhex = [[[0i32; 8]; 3]; 3];
    let mut sgrow = 0i32;
    let mut sgcol = 0i32;
    for row in 0..3i32 {
        for col in 0..3i32 {
            let mut ng = 0i32;
            let mut d = 0i32;
            while d < 10 {
                let g = (fc(row, col) == 1) as i32;
                if fc(row + ORTH[d as usize], col + ORTH[(d + 2) as usize]) == 1 {
                    ng = 0;
                } else {
                    ng += 1;
                }
                if ng == 4 {
                    sgrow = row;
                    sgcol = col;
                }
                if ng == g + 1 {
                    for c in 0..8usize {
                        let v = ORTH[d as usize] * PATT[g as usize][c * 2]
                            + ORTH[(d + 1) as usize] * PATT[g as usize][c * 2 + 1];
                        let h = ORTH[(d + 2) as usize] * PATT[g as usize][c * 2]
                            + ORTH[(d + 3) as usize] * PATT[g as usize][c * 2 + 1];
                        allhex[row as usize][col as usize][c ^ ((g * 2 & d) as usize)] = h + v * ts;
                    }
                }
                d += 2;
            }
        }
    }
    let hexmap = |row: i32, col: i32| -> [i32; 8] {
        allhex[((row + 600) % 3) as usize][((col + 600) % 3) as usize]
    };
    let cmod = |a: i32, b: i32| -> i32 { a % b };

    // rowoffset/coloffset for the modarr modulator lookup
    let mut rowoffset = 0i32;
    let mut coloffset = 0i32;
    'outer: for row in 0..6i32 {
        if cmod(row - sgrow, 3) == 0 {
            for col in 0..6i32 {
                if cmod(col - sgcol, 3) == 0 && fc(row, col + 1) == 0 {
                    rowoffset = 37 - row - pad_tile;
                    coloffset = 37 - col - pad_tile;
                    break 'outer;
                }
            }
            break;
        }
    }

    let mut rgb = vec![0.0f32; ndir * tts * 3];
    let mut yuv = vec![0.0f32; 3 * tts];
    let mut drv = vec![0.0f32; ndir * tts];
    let mut gmin = vec![0.0f32; tts];
    let mut gmax = vec![0.0f32; tts];
    let mut homo = vec![0u8; ndir * tts];
    let mut homosum = vec![0u8; ndir * tts];
    let mut i_src = vec![0.0f32; tts];
    let mut fdc_chroma = vec![0.0f32; 2 * tts];

    let tile_step = ts - pad_tile * 2;
    let tdir = (tts * 3) as isize;
    let pidx = |d: isize, rr: i32, cc: i32| -> isize {
        d * tdir + ((rr as isize) * TS as isize + cc as isize) * 3
    };

    let mut top = -pad_tile;
    while top < height - pad_tile {
        let mut left = -pad_tile;
        while left < width - pad_tile {
            let mrow = (top + ts).min(height + pad_tile);
            let mcol = (left + ts).min(width + pad_tile);

            // --- Copy tile to rgb[0] (RAW, no clamp) + fill i_src luma ---
            for row in top..mrow {
                for col in left..mcol {
                    let pb = (((row - top) as usize) * TS + (col - left) as usize) * 3;
                    let ii = (row - top) as usize * TS + (col - left) as usize;
                    if col >= 0 && row >= 0 && col < width && row < height {
                        let f = fc(row, col);
                        let v = inb[(width * row + col) as usize];
                        for c in 0..3i32 {
                            rgb[pb + c as usize] = if c == f { v } else { 0.0 };
                        }
                        i_src[ii] = v;
                    } else {
                        let c = fc(row, col);
                        for cc in 0..3i32 {
                            if cc != c {
                                rgb[pb + cc as usize] = 0.0;
                            } else {
                                let cy = translate(row, height);
                                let cx = translate(col, width);
                                if c == fc(cy, cx) {
                                    let v = inb[(width * cy + cx) as usize];
                                    rgb[pb + c as usize] = v;
                                    i_src[ii] = v;
                                } else {
                                    let mut sum = 0.0f32;
                                    let mut count = 0u8;
                                    for y in row - 1..=row + 1 {
                                        for x in col - 1..=col + 1 {
                                            let yy = translate(y, height);
                                            let xx = translate(x, width);
                                            if fc(yy, xx) == c {
                                                sum += inb[(width * yy + xx) as usize];
                                                count += 1;
                                            }
                                        }
                                    }
                                    rgb[pb + c as usize] = sum / count as f32;
                                    i_src[ii] = rgb[pb + c as usize];
                                }
                            }
                        }
                    }
                }
            }

            for c in 1..=3usize {
                rgb.copy_within(0..tts * 3, c * tts * 3);
            }

            // --- gmin/gmax (identical to markesteijn) ---
            let mut row = top + 3;
            while row < mrow - 3 {
                let mut min = f32::MAX;
                let mut max = 0.0f32;
                let mut col = left + 3;
                while col < mcol - 3 {
                    if fc(row, col) == 1 {
                        min = f32::MAX;
                        max = 0.0;
                        col += 1;
                        continue;
                    }
                    if max == 0.0 {
                        let pb = pidx(0, row - top, col - left);
                        let hex = hexmap(row, col);
                        for c in 0..6usize {
                            let val = rgb[(pb + hex[c] as isize * 3 + 1) as usize];
                            if min > val {
                                min = val;
                            }
                            if max < val {
                                max = val;
                            }
                        }
                    }
                    let gi = (row - top) as usize * TS + (col - left) as usize;
                    gmin[gi] = min;
                    gmax[gi] = max;
                    match cmod(row - sgrow, 3) {
                        1 => {
                            if row < mrow - 4 {
                                row += 1;
                                col -= 1;
                            }
                        }
                        2 => {
                            min = f32::MAX;
                            max = 0.0;
                            col += 2;
                            if col < mcol - 4 && row > top + 3 {
                                row -= 1;
                            }
                        }
                        _ => {}
                    }
                    col += 1;
                }
                row += 1;
            }

            // --- green interpolation in 4 directions (identical) ---
            for row in top + 3..mrow - 3 {
                for col in left + 3..mcol - 3 {
                    let f = fc(row, col);
                    if f == 1 {
                        continue;
                    }
                    let pb = pidx(0, row - top, col - left);
                    let hex = hexmap(row, col);
                    let pg = |k: isize| rgb[(pb + k * 3 + 1) as usize];
                    let pf = |k: isize| rgb[(pb + k * 3 + f as isize) as usize];
                    let mut color = [0.0f32; 8];
                    color[0] = 0.6796875 * (pg(hex[1] as isize) + pg(hex[0] as isize))
                        - 0.1796875 * (pg(2 * hex[1] as isize) + pg(2 * hex[0] as isize));
                    color[1] = 0.87109375 * pg(hex[3] as isize) + pg(hex[2] as isize) * 0.13
                        + 0.359375 * (pf(0) - pf(-hex[2] as isize));
                    for c in 0..2usize {
                        color[2 + c] = 0.640625 * pg(hex[4 + c] as isize)
                            + 0.359375 * pg(-2 * hex[4 + c] as isize)
                            + 0.12890625
                                * (2.0 * pf(0) - pf(3 * hex[4 + c] as isize)
                                    - pf(-3 * hex[4 + c] as isize));
                    }
                    let gi = (row - top) as usize * TS + (col - left) as usize;
                    let (lo, hi) = (gmin[gi], gmax[gi]);
                    let inv = (cmod(row - sgrow, 3) == 0) as usize;
                    for c in 0..4usize {
                        rgb[(pidx((c ^ inv) as isize, row - top, col - left) + 1) as usize] =
                            clamps(color[c], lo, hi);
                    }
                }
            }

            // --- solitary-green R/B (FDC variant: color[h][d], copy-on-diff) ---
            let pad_rb_g = 6;
            let mut row = (top - sgrow + pad_rb_g + 2) / 3 * 3 + sgrow;
            while row < mrow - pad_rb_g {
                let mut col = (left - sgcol + pad_rb_g + 2) / 3 * 3 + sgcol;
                while col < mcol - pad_rb_g {
                    let mut rfx = pidx(0, row - top, col - left);
                    let mut h = fc(row, col + 1);
                    let mut diff = [0.0f32; 6];
                    let mut color = [[0.0f32; 8]; 3];
                    let mut i = 1i32;
                    let mut d = 0i32;
                    while d < 6 {
                        let mut c = 0i32;
                        while c < 2 {
                            let k = (i << c) as isize;
                            let rg = |kk: isize| rgb[(rfx + kk * 3 + 1) as usize];
                            let rh = |kk: isize| rgb[(rfx + kk * 3 + h as isize) as usize];
                            let g = 2.0 * rg(0) - rg(k) - rg(-k);
                            color[h as usize][d as usize] = g + rh(k) + rh(-k);
                            if d > 1 {
                                diff[d as usize] +=
                                    sqrf(rg(k) - rg(-k) - rh(k) + rh(-k)) + sqrf(g);
                            }
                            c += 1;
                            h ^= 2;
                        }
                        if d > 1 && (d & 1) != 0 && diff[(d - 1) as usize] < diff[d as usize] {
                            for c in 0..2usize {
                                color[c * 2][d as usize] = color[c * 2][(d - 1) as usize];
                            }
                        }
                        if d < 2 || (d & 1) != 0 {
                            for c in 0..2usize {
                                rgb[(rfx + (c * 2) as isize) as usize] =
                                    color[c * 2][d as usize] / 2.0;
                            }
                            rfx += tdir;
                        }
                        d += 1;
                        i ^= ts ^ 1;
                        h ^= 2;
                    }
                    col += 3;
                }
                row += 3;
            }

            // --- R for blue & vice versa (identical to markesteijn) ---
            let pad_rb_br = 6;
            for row in top + pad_rb_br..mrow - pad_rb_br {
                for col in left + pad_rb_br..mcol - pad_rb_br {
                    let f = 2 - fc(row, col);
                    if f == 1 {
                        continue;
                    }
                    let c = if cmod(row - sgrow, 3) != 0 { ts } else { 1 };
                    let h = 3 * (c ^ ts ^ 1);
                    let mut rfx = pidx(0, row - top, col - left);
                    for d in 0..4i32 {
                        let rg = |k: isize| rgb[(rfx + k * 3 + 1) as usize];
                        let i = if d > 1
                            || ((d ^ c) & 1) != 0
                            || ((rg(0) - rg(c as isize)).abs() + (rg(0) - rg(-c as isize)).abs())
                                < 2.0
                                    * ((rg(0) - rg(h as isize)).abs()
                                        + (rg(0) - rg(-h as isize)).abs())
                        {
                            c
                        } else {
                            h
                        };
                        let ii = i as isize;
                        let rf = |k: isize| rgb[(rfx + k * 3 + f as isize) as usize];
                        rgb[(rfx + f as isize) as usize] =
                            (rf(ii) + rf(-ii) + 2.0 * rg(0) - rg(ii) - rg(-ii)) / 2.0;
                        rfx += tdir;
                    }
                }
            }

            // --- 2x2 green fill (FDC variant: store redblue + diagonal fill) ---
            let pad_g22 = 8;
            for row in top + pad_g22..mrow - pad_g22 {
                if cmod(row - sgrow, 3) == 0 {
                    continue;
                }
                for col in left + pad_g22..mcol - pad_g22 {
                    if cmod(col - sgcol, 3) == 0 {
                        continue;
                    }
                    let hex = hexmap(row, col);
                    let mut redblue = [[0.0f32; 3]; 3];
                    let mut rfx = pidx(0, row - top, col - left);
                    let mut d = 0usize;
                    while d < ndir {
                        let hd = hex[d] as isize;
                        let hd1 = hex[d + 1] as isize;
                        let rg = |k: isize| rgb[(rfx + k * 3 + 1) as usize];
                        if hex[d] + hex[d + 1] != 0 {
                            let g = 3.0 * rg(0) - 2.0 * rg(hd) - rg(hd1);
                            let mut c = 0isize;
                            while c < 4 {
                                let val = (g
                                    + 2.0 * rgb[(rfx + hd * 3 + c) as usize]
                                    + rgb[(rfx + hd1 * 3 + c) as usize])
                                    / 3.0;
                                rgb[(rfx + c) as usize] = val;
                                redblue[d][c as usize] = val;
                                c += 2;
                            }
                        } else {
                            let g = 2.0 * rg(0) - rg(hd) - rg(hd1);
                            let mut c = 0isize;
                            while c < 4 {
                                let val = (g
                                    + rgb[(rfx + hd * 3 + c) as usize]
                                    + rgb[(rfx + hd1 * 3 + c) as usize])
                                    / 2.0;
                                rgb[(rfx + c) as usize] = val;
                                redblue[d][c as usize] = val;
                                c += 2;
                            }
                        }
                        d += 2;
                        rfx += tdir;
                    }
                    // diagonal directions: average of redblue[0] and redblue[2]
                    let mut d = 0usize;
                    while d < ndir {
                        let mut c = 0isize;
                        while c < 4 {
                            rgb[(rfx + c) as usize] =
                                (redblue[0][c as usize] + redblue[2][c as usize]) * 0.5;
                            c += 2;
                        }
                        d += 2;
                        rfx += tdir;
                    }
                }
            }

            // work in tile-local coords now
            let mrowl = mrow - top;
            let mcoll = mcol - left;

            // --- YPbPr + derivatives (identical to markesteijn) ---
            for d in 0..ndir {
                let pad_yuv = 8;
                for row in pad_yuv..mrowl - pad_yuv {
                    for col in pad_yuv..mcoll - pad_yuv {
                        let rx = pidx(d as isize, row, col);
                        let r0 = rgb[rx as usize];
                        let r1 = rgb[(rx + 1) as usize];
                        let r2 = rgb[(rx + 2) as usize];
                        let y = 0.2627 * r0 + 0.6780 * r1 + 0.0593 * r2;
                        let yi = (row as usize) * TS + col as usize;
                        yuv[yi] = y;
                        yuv[tts + yi] = (r2 - y) * 0.56433;
                        yuv[2 * tts + yi] = (r0 - y) * 0.67815;
                    }
                }
                let f = dir[d & 3] as isize;
                let pad_drv = 9;
                for row in pad_drv..mrowl - pad_drv {
                    for col in pad_drv..mcoll - pad_drv {
                        let yi = (row as isize) * TS as isize + col as isize;
                        let yf =
                            |ch: isize, off: isize| yuv[(ch * tts as isize + yi + off) as usize];
                        drv[d * tts + (row as usize) * TS + col as usize] =
                            sqrf(2.0 * yf(0, 0) - yf(0, f) - yf(0, -f))
                                + sqrf(2.0 * yf(1, 0) - yf(1, f) - yf(1, -f))
                                + sqrf(2.0 * yf(2, 0) - yf(2, f) - yf(2, -f));
                    }
                }
            }

            // --- homogeneity maps (identical) ---
            for v in homo.iter_mut() {
                *v = 0;
            }
            let pad_homo = 10;
            for row in pad_homo..mrowl - pad_homo {
                for col in pad_homo..mcoll - pad_homo {
                    let mut tr = f32::MAX;
                    for d in 0..ndir {
                        let dv = drv[d * tts + row as usize * TS + col as usize];
                        if tr > dv {
                            tr = dv;
                        }
                    }
                    tr *= 8.0;
                    for d in 0..ndir {
                        let mut cnt = 0u8;
                        for vv in -1..=1i32 {
                            for hh in -1..=1i32 {
                                if drv[d * tts + ((row + vv) as usize) * TS + (col + hh) as usize]
                                    <= tr
                                {
                                    cnt += 1;
                                }
                            }
                        }
                        homo[d * tts + row as usize * TS + col as usize] += cnt;
                    }
                }
            }

            // --- 5x5 homogeneity sums (identical) ---
            for d in 0..ndir {
                for row in pad_tile..mrowl - pad_tile {
                    let mut col = pad_tile - 5;
                    let mut v5sum = [0u8; 5];
                    homosum[d * tts + row as usize * TS + col as usize] = 0;
                    col += 1;
                    while col < mcoll - pad_tile {
                        let mut colsum = 0u8;
                        for vv in -2..=2i32 {
                            colsum = colsum.wrapping_add(
                                homo[d * tts + ((row + vv) as usize) * TS + (col + 2) as usize],
                            );
                        }
                        let prev = homosum[d * tts + row as usize * TS + (col - 1) as usize];
                        let cm = (col % 5) as usize;
                        homosum[d * tts + row as usize * TS + col as usize] =
                            prev.wrapping_sub(v5sum[cm]).wrapping_add(colsum);
                        v5sum[cm] = colsum;
                        col += 1;
                    }
                }
            }

            // --- FDC chroma stage ---
            let pad_fdc = 6;
            for row in pad_fdc..mrowl - pad_fdc {
                for col in pad_fdc..mcoll - pad_fdc {
                    let mut hm = [0u8; 8];
                    let mut maxval = 0u8;
                    for d in 0..ndir {
                        hm[d] = homosum[d * tts + row as usize * TS + col as usize];
                        if maxval < hm[d] {
                            maxval = hm[d];
                        }
                    }
                    maxval -= maxval >> 3;
                    let mut dircount = 0.0f32;
                    let mut dirsum = 0.0f32;
                    for d in 0..ndir {
                        if hm[d] >= maxval {
                            dircount += 1.0;
                            dirsum += DIRECTIONALITY[d];
                        }
                    }
                    let wgt = dirsum / dircount;
                    // 13x13 complex convolutions of i_src against harr[0..3]
                    let conv = |filt: &[[Cf; 13]; 13]| -> Cf {
                        let mut acc = Cf::new(0.0, 0.0);
                        let mut fr = 0usize;
                        let mut myrow = row - 6;
                        while fr < 13 {
                            let mut fco = 0usize;
                            let mut mycol = col - 6;
                            while fco < 13 {
                                let iv = i_src[(TS as i32 * myrow + mycol) as usize];
                                acc = acc + filt[12 - fr][12 - fco] * iv;
                                fco += 1;
                                mycol += 1;
                            }
                            fr += 1;
                            myrow += 1;
                        }
                        acc
                    };
                    let mut c2m = conv(&HARR[0]);
                    let c5m = conv(&HARR[1]);
                    let c7m = conv(&HARR[2]);
                    let c10m = conv(&HARR[3]);
                    let mr = ((row + rowoffset) % 6) as usize;
                    let mc = ((col + coloffset) % 6) as usize;
                    let modu = MODARR[mr][mc];
                    let mut qmat = [Cf::new(0.0, 0.0); 8];
                    qmat[4] = wgt * c10m * modu[0] - (1.0 - wgt) * c2m * modu[1];
                    qmat[6] = qmat[4].conj();
                    qmat[1] = c5m * modu[6];
                    qmat[2] = (-0.5f32 * qmat[1]).conj();
                    qmat[5] = qmat[2].conj();
                    qmat[3] = c7m * modu[7];
                    qmat[7] = qmat[1].conj();
                    // get L
                    c2m = qmat[4] * (modu[0].conj() - modu[1].conj());
                    let c3m = qmat[6] * (modu[2] - modu[3]);
                    let c6m = qmat[2] * (modu[4].conj() + modu[5].conj());
                    let c12m = qmat[5] * (modu[4] + modu[5]);
                    let c18m = qmat[7] * modu[6];
                    let i0 = i_src[(row * TS as i32 + col) as usize];
                    qmat[0] = Cf::new(i0, 0.0) - c2m - c3m - c5m - c6m - 2.0f32 * c7m - c12m - c18m;
                    // demodulate via Minv (float += complex takes the real part)
                    let mut rgbpix = [0.0f32; 3];
                    for color in 0..3usize {
                        for c in 0..8usize {
                            rgbpix[color] += (MINV[color][c] * qmat[c]).re;
                        }
                    }
                    let y = 0.2627 * rgbpix[0] + 0.6780 * rgbpix[1] + 0.0593 * rgbpix[2];
                    fdc_chroma[(row as usize) * TS + col as usize] = (rgbpix[2] - y) * 0.56433;
                    fdc_chroma[tts + (row as usize) * TS + col as usize] = (rgbpix[0] - y) * 0.67815;
                }
            }

            // --- final output: homogeneous-average luma + FDC chroma ---
            for row in pad_tile..mrowl - pad_tile {
                for col in pad_tile..mcoll - pad_tile {
                    let mut hm = [0u8; 8];
                    let mut maxval = 0u8;
                    for d in 0..ndir {
                        hm[d] = homosum[d * tts + row as usize * TS + col as usize];
                        if maxval < hm[d] {
                            maxval = hm[d];
                        }
                    }
                    maxval -= maxval >> 3;
                    // ndir-4 == 0 ⇒ the hm[d]<hm[d+4] zeroing loop does not run
                    let mut avg = [0.0f32; 4];
                    for d in 0..ndir {
                        if hm[d] >= maxval {
                            let rx = pidx(d as isize, row, col);
                            avg[0] += rgb[rx as usize];
                            avg[1] += rgb[(rx + 1) as usize];
                            avg[2] += rgb[(rx + 2) as usize];
                            avg[3] += 1.0;
                        }
                    }
                    let rp = [avg[0] / avg[3], avg[1] / avg[3], avg[2] / avg[3]];
                    let y = 0.2627 * rp[0] + 0.6780 * rp[1] + 0.0593 * rp[2];
                    let um = (rp[2] - y) * 0.56433;
                    let vm = (rp[0] - y) * 0.67815;
                    // 5-pixel median filter of fdc_chroma per channel
                    let mut uvf = [0.0f32; 2];
                    for chrm in 0..2usize {
                        let base = chrm * tts;
                        let r = row as usize;
                        let c = col as usize;
                        let mut t = [
                            fdc_chroma[base + (r - 1) * TS + c],
                            fdc_chroma[base + r * TS + (c - 1)],
                            fdc_chroma[base + r * TS + c],
                            fdc_chroma[base + r * TS + (c + 1)],
                            fdc_chroma[base + (r + 1) * TS + c],
                        ];
                        let (mut a, mut b, mut cc, mut d, mut e) =
                            (t[0], t[1], t[2], t[3], t[4]);
                        pix_sort(&mut a, &mut b);
                        pix_sort(&mut d, &mut e);
                        pix_sort(&mut a, &mut d);
                        pix_sort(&mut b, &mut e);
                        pix_sort(&mut b, &mut cc);
                        pix_sort(&mut cc, &mut d);
                        pix_sort(&mut b, &mut cc);
                        let _ = (&mut t, a, e);
                        uvf[chrm] = cc;
                    }
                    let mut uv = [0.0f32; 2];
                    uv[0] = (if (uvf[0].abs() < um.abs()) && (uvf[1].abs() < 1.02 * vm.abs()) {
                        uvf[0]
                    } else {
                        um
                    }) * hybrid0
                        + uvf[0] * hybrid1;
                    uv[1] = (if (uvf[1].abs() < vm.abs()) && (uvf[0].abs() < 1.02 * vm.abs()) {
                        uvf[1]
                    } else {
                        vm
                    }) * hybrid0
                        + uvf[1] * hybrid1;
                    let mut rgbpix = [0.0f32; 3];
                    rgbpix[0] = y + 1.474600014746 * uv[1];
                    rgbpix[1] = y - 0.15498578286403 * uv[0] - 0.571353132557189 * uv[1];
                    rgbpix[2] = y + 1.77201282937288 * uv[0];
                    let oi = 4 * ((width * (row + top) + col + left) as usize);
                    for c in 0..3usize {
                        o[oi + c] = rgbpix[c].max(0.0);
                    }
                }
            }

            left += tile_step;
        }
        top += tile_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XTRANS: [[u8; 6]; 6] = [
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
        [0, 2, 1, 2, 0, 1],
    ];

    #[test]
    fn fdc_runs_over_a_full_tile_without_oob() {
        // Index-safety smoke test over the full FDC pipeline (incl. the 13x13
        // harr convolution + Minv demodulation) on a genuine X-Trans CFA.
        let (w, h) = (80usize, 80usize);
        let inb = vec![0.5_f32; w * h];
        let xt: Vec<u8> = XTRANS.iter().flatten().copied().collect();
        // hybrid path
        let mut out = vec![-999.0_f32; w * h * 4];
        unsafe {
            darkroom_xtrans_fdc(out.as_mut_ptr(), inb.as_ptr(), w as i32, h as i32,
                xt.as_ptr() as *const u8, 1.0, 0.0);
        }
        assert_ne!(out[4 * (40 * w + 40)], -999.0, "hybrid: interior not written");
        // pure FDC path
        let mut out2 = vec![-999.0_f32; w * h * 4];
        unsafe {
            darkroom_xtrans_fdc(out2.as_mut_ptr(), inb.as_ptr(), w as i32, h as i32,
                xt.as_ptr() as *const u8, 0.0, 1.0);
        }
        assert_ne!(out2[4 * (40 * w + 40)], -999.0, "pure: interior not written");
    }
}
