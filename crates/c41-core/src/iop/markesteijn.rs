//! Frank Markesteijn's X-Trans demosaic — whole-function port of
//! `xtrans_markesteijn_interpolate` (src/iop/demosaicing/xtrans.c:45).
//!
//! The C uses a per-thread OMP scratch buffer with several views aliased onto
//! the same memory (homo/homosum bytes over the yuv/gmin float region) purely to
//! save space; those phases are temporally disjoint, so this serial port uses
//! separate buffers and obtains identical results. Pointer walks over the
//! `rgb[ndir][TS][TS][3]` tile become flat f32-index arithmetic: a C
//! `float(*rfx)[3]` is a flat index `rfx` into `rgb` and `rfx[k][ch]` is
//! `rgb[rfx + k*3 + ch]`; one direction-tile (`rfx += TS*TS`) is `rfx += TS*TS*3`.
//! The `rgb += 4` of pass 1 is tracked by the `rb` base offset. The trailing
//! `_vng_lininterpolate` call stays in C.

use crate::raw;

const TS: usize = 122; // DT_MARKESTEIJN_TS

#[inline(always)]
fn clamps(a: f32, l: f32, h: f32) -> f32 {
    // CLAMPS(A,L,H) = (A>L) ? (A<H?A:H) : L
    if a > l {
        if a < h { a } else { h }
    } else {
        l
    }
}

#[inline(always)]
fn translate(n: i32, size: i32) -> i32 {
    // TRANSLATE(n,size) = (n>=size) ? (2*size-n-2) : abs(n)
    if n >= size { 2 * size - n - 2 } else { n.abs() }
}

#[inline(always)]
fn sqrf(x: f32) -> f32 {
    x * x
}

/// Markesteijn X-Trans interpolation (passes = 1 or 3). Writes the RGBA `out`.
///
/// # Safety
/// `out` holds `width*height*4` floats; `in_buf` holds `width*height`; `xtrans`
/// points to 36 valid bytes. width/height are the real ROI dims.
#[no_mangle]
pub unsafe extern "C" fn darkroom_xtrans_markesteijn(
    out: *mut f32,
    in_buf: *const f32,
    width: i32,
    height: i32,
    xtrans: *const u8,
    passes: i32,
    _filters: u32,
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

    // Map a green hexagon around each non-green pixel and vice versa.
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
                        let idx = c ^ ((g * 2 & d) as usize);
                        allhex[row as usize][col as usize][idx] = h + v * ts;
                    }
                }
                d += 2;
            }
        }
    }
    let hexmap = |row: i32, col: i32| -> [i32; 8] {
        allhex[((row + 600) % 3) as usize][((col + 600) % 3) as usize]
    };
    // C truncated modulo (sign follows dividend)
    let cmod = |a: i32, b: i32| -> i32 { a % b };

    let ndir: usize = 4usize << ((passes > 1) as usize); // 4 or 8
    let pad_tile = if passes == 1 { 12i32 } else { 17 };

    // scratch buffers (separate, see module note)
    let mut rgb = vec![0.0f32; ndir * tts * 3];
    let mut yuv = vec![0.0f32; 3 * tts];
    let mut drv = vec![0.0f32; ndir * tts];
    let mut gmin = vec![0.0f32; tts];
    let mut gmax = vec![0.0f32; tts];
    let mut homo = vec![0u8; ndir * tts];
    let mut homosum = vec![0u8; ndir * tts];

    let tile_step = ts - pad_tile * 2;
    let tdir = (tts * 3) as isize; // one direction-tile in f32 units
    // flat f32 base index of rgb[d][rr][cc] (tile-local rr,cc), plus base `rb`
    let pidx = |rb: isize, d: isize, rr: i32, cc: i32| -> isize {
        rb + d * tdir + ((rr as isize) * TS as isize + cc as isize) * 3
    };

    let mut top = -pad_tile;
    while top < height - pad_tile {
        let mut left = -pad_tile;
        while left < width - pad_tile {
            let mrow = (top + ts).min(height + pad_tile);
            let mcol = (left + ts).min(width + pad_tile);

            // --- Copy current tile from `in` to rgb[0], mirroring borders ---
            for row in top..mrow {
                for col in left..mcol {
                    let pb = (((row - top) as usize) * TS + (col - left) as usize) * 3;
                    if col >= 0 && row >= 0 && col < width && row < height {
                        let f = fc(row, col);
                        let v = inb[(width * row + col) as usize].max(0.0);
                        for c in 0..3i32 {
                            rgb[pb + c as usize] = if c == f { v } else { 0.0 };
                        }
                    } else {
                        let c = fc(row, col);
                        for cc in 0..3i32 {
                            if cc != c {
                                rgb[pb + cc as usize] = 0.0;
                            } else {
                                let cy = translate(row, height);
                                let cx = translate(col, width);
                                if c == fc(cy, cx) {
                                    rgb[pb + c as usize] = inb[(width * cy + cx) as usize].max(0.0);
                                } else {
                                    let mut sum = 0.0f32;
                                    let mut count = 0u8;
                                    for y in row - 1..=row + 1 {
                                        for x in col - 1..=col + 1 {
                                            let yy = translate(y, height);
                                            let xx = translate(x, width);
                                            if fc(yy, xx) == c {
                                                sum += inb[(width * yy + xx) as usize].max(0.0);
                                                count += 1;
                                            }
                                        }
                                    }
                                    rgb[pb + c as usize] = sum / count as f32;
                                }
                            }
                        }
                    }
                }
            }

            // duplicate rgb[0] to rgb[1..=3]
            for c in 1..=3usize {
                rgb.copy_within(0..tts * 3, c * tts * 3);
            }

            // --- green1/green3 = min/max of surrounding greens (3px border) ---
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
                        let pb = pidx(0, 0, row - top, col - left);
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

            // --- Interpolate green in 4 directions (3px border) ---
            for row in top + 3..mrow - 3 {
                for col in left + 3..mcol - 3 {
                    let f = fc(row, col);
                    if f == 1 {
                        continue;
                    }
                    let pb = pidx(0, 0, row - top, col - left);
                    let hex = hexmap(row, col);
                    let pg = |k: isize| rgb[(pb + k * 3 + 1) as usize]; // green
                    let pf = |k: isize| rgb[(pb + k * 3 + f as isize) as usize]; // own colour
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
                    let lo = gmin[gi];
                    let hi = gmax[gi];
                    let inv = (cmod(row - sgrow, 3) == 0) as usize; // !((row-sgrow)%3)
                    for c in 0..4usize {
                        let dd = (c ^ inv) as isize;
                        rgb[(pidx(0, dd, row - top, col - left) + 1) as usize] =
                            clamps(color[c], lo, hi);
                    }
                }
            }

            // --- multipass loop ---
            let mut rb: isize = 0;
            for pass in 0..passes {
                if pass == 1 {
                    // copy rgb[0..4] to rgb[4..8], then work on the second set
                    rgb.copy_within(0..4 * tts * 3, 4 * tts * 3);
                    rb += 4 * tdir;
                }

                // recalculate green from interpolated closer pixels (pass>0)
                if pass != 0 {
                    for row in top + 6..mrow - 6 {
                        for col in left + 6..mcol - 6 {
                            let f = fc(row, col);
                            if f == 1 {
                                continue;
                            }
                            let hex = hexmap(row, col);
                            let inv = (cmod(row - sgrow, 3) == 0) as i32;
                            let gi = (row - top) as usize * TS + (col - left) as usize;
                            for d in 3..6usize {
                                let dd = ((d as i32 - 2) ^ inv) as isize;
                                let rfx = pidx(rb, dd, row - top, col - left);
                                let rg = |k: isize| rgb[(rfx + k * 3 + 1) as usize];
                                let rf = |k: isize| rgb[(rfx + k * 3 + f as isize) as usize];
                                let h = hex[d] as isize;
                                let val = rg(-2 * h) + 2.0 * rg(h) - rf(-2 * h) - 2.0 * rf(h)
                                    + 3.0 * rf(0);
                                rgb[(rfx + 1) as usize] = clamps(val / 3.0, gmin[gi], gmax[gi]);
                            }
                        }
                    }
                }

                // interpolate red/blue for solitary green pixels
                let pad_rb_g = if passes == 1 { 6 } else { 5 };
                let mut row = (top - sgrow + pad_rb_g + 2) / 3 * 3 + sgrow;
                while row < mrow - pad_rb_g {
                    let mut col = (left - sgcol + pad_rb_g + 2) / 3 * 3 + sgcol;
                    while col < mcol - pad_rb_g {
                        let mut rfx = pidx(rb, 0, row - top, col - left);
                        let mut h = fc(row, col + 1);
                        let mut diff = [0.0f32; 6];
                        let mut color = [[0.0f32; 6]; 2];
                        let mut i = 1i32;
                        let mut d = 0i32;
                        while d < 6 {
                            let mut c = 0i32;
                            while c < 2 {
                                let k = (i << c) as isize;
                                let rg = |kk: isize| rgb[(rfx + kk * 3 + 1) as usize];
                                let rh = |kk: isize| rgb[(rfx + kk * 3 + h as isize) as usize];
                                let g = 2.0 * rg(0) - rg(k) - rg(-k);
                                color[(h != 0) as usize][d as usize] = g + rh(k) + rh(-k);
                                if d > 1 {
                                    diff[d as usize] +=
                                        sqrf(rg(k) - rg(-k) - rh(k) + rh(-k)) + sqrf(g);
                                }
                                c += 1;
                                h ^= 2;
                            }
                            if d < 2 || (d & 1) != 0 {
                                let d_out =
                                    d - (((d > 1) && (diff[(d - 1) as usize] < diff[d as usize]))
                                        as i32);
                                rgb[rfx as usize] = color[0][d_out as usize] / 2.0;
                                rgb[(rfx + 2) as usize] = color[1][d_out as usize] / 2.0;
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

                // interpolate red for blue pixels and vice versa
                let pad_rb_br = if passes == 1 { 6 } else { 5 };
                for row in top + pad_rb_br..mrow - pad_rb_br {
                    for col in left + pad_rb_br..mcol - pad_rb_br {
                        let f = 2 - fc(row, col);
                        if f == 1 {
                            continue;
                        }
                        let c = if cmod(row - sgrow, 3) != 0 { ts } else { 1 };
                        let h = 3 * (c ^ ts ^ 1);
                        let mut rfx = pidx(rb, 0, row - top, col - left);
                        for d in 0..4i32 {
                            let rg = |k: isize| rgb[(rfx + k * 3 + 1) as usize];
                            let i = if d > 1
                                || ((d ^ c) & 1) != 0
                                || ((rg(0) - rg(c as isize)).abs()
                                    + (rg(0) - rg(-c as isize)).abs())
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
                            let val = (rf(ii) + rf(-ii) + 2.0 * rg(0) - rg(ii) - rg(-ii)) / 2.0;
                            rgb[(rfx + f as isize) as usize] = val;
                            rfx += tdir;
                        }
                    }
                }

                // fill in red and blue for 2x2 blocks of green
                let pad_g22 = if passes == 1 { 8 } else { 4 };
                for row in top + pad_g22..mrow - pad_g22 {
                    if cmod(row - sgrow, 3) == 0 {
                        continue;
                    }
                    for col in left + pad_g22..mcol - pad_g22 {
                        if cmod(col - sgcol, 3) == 0 {
                            continue;
                        }
                        let hex = hexmap(row, col);
                        let mut rfx = pidx(rb, 0, row - top, col - left);
                        let mut d = 0usize;
                        while d < ndir {
                            let hd = hex[d] as isize;
                            let hd1 = hex[d + 1] as isize;
                            let rg = |k: isize| rgb[(rfx + k * 3 + 1) as usize];
                            if hex[d] + hex[d + 1] != 0 {
                                let g = 3.0 * rg(0) - 2.0 * rg(hd) - rg(hd1);
                                let mut c = 0isize;
                                while c < 4 {
                                    rgb[(rfx + c) as usize] = (g
                                        + 2.0 * rgb[(rfx + hd * 3 + c) as usize]
                                        + rgb[(rfx + hd1 * 3 + c) as usize])
                                        / 3.0;
                                    c += 2;
                                }
                            } else {
                                let g = 2.0 * rg(0) - rg(hd) - rg(hd1);
                                let mut c = 0isize;
                                while c < 4 {
                                    rgb[(rfx + c) as usize] = (g
                                        + rgb[(rfx + hd * 3 + c) as usize]
                                        + rgb[(rfx + hd1 * 3 + c) as usize])
                                        / 2.0;
                                    c += 2;
                                }
                            }
                            d += 2;
                            rfx += tdir;
                        }
                    }
                }
            } // end multipass

            // back to the first rgb set; work in tile-local coords now
            let mrowl = mrow - top;
            let mcoll = mcol - left;

            // --- Convert to YPbPr and differentiate in all directions ---
            for d in 0..ndir {
                let pad_yuv = if passes == 1 { 8 } else { 13 };
                for row in pad_yuv..mrowl - pad_yuv {
                    for col in pad_yuv..mcoll - pad_yuv {
                        let rx = pidx(0, d as isize, row, col);
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
                let pad_drv = if passes == 1 { 9 } else { 14 };
                for row in pad_drv..mrowl - pad_drv {
                    for col in pad_drv..mcoll - pad_drv {
                        let yi = (row as usize as isize) * TS as isize + col as isize;
                        // yfx[ch][0][off] = yuv[ch*tts + yi + off]
                        let yf = |ch: isize, off: isize| yuv[(ch * tts as isize + yi + off) as usize];
                        let v = sqrf(2.0 * yf(0, 0) - yf(0, f) - yf(0, -f))
                            + sqrf(2.0 * yf(1, 0) - yf(1, f) - yf(1, -f))
                            + sqrf(2.0 * yf(2, 0) - yf(2, f) - yf(2, -f));
                        drv[d * tts + (row as usize) * TS + col as usize] = v;
                    }
                }
            }

            // --- Build homogeneity maps from the derivatives ---
            for v in homo.iter_mut() {
                *v = 0;
            }
            let pad_homo = if passes == 1 { 10 } else { 15 };
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
                                let dv = drv[d * tts
                                    + ((row + vv) as usize) * TS
                                    + (col + hh) as usize];
                                if dv <= tr {
                                    cnt += 1;
                                }
                            }
                        }
                        homo[d * tts + row as usize * TS + col as usize] += cnt;
                    }
                }
            }

            // --- 5x5 sum of homogeneity maps per pixel & direction ---
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
                        let nv = prev.wrapping_sub(v5sum[cm]).wrapping_add(colsum);
                        homosum[d * tts + row as usize * TS + col as usize] = nv;
                        v5sum[cm] = colsum;
                        col += 1;
                    }
                }
            }

            // --- Average the most homogeneous pixels for the final result ---
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
                    for d in 0..ndir - 4 {
                        if hm[d] < hm[d + 4] {
                            hm[d] = 0;
                        } else if hm[d] > hm[d + 4] {
                            hm[d + 4] = 0;
                        }
                    }
                    let mut avg = [0.0f32; 4];
                    for d in 0..ndir {
                        if hm[d] >= maxval {
                            let rx = pidx(0, d as isize, row, col);
                            avg[0] += rgb[rx as usize];
                            avg[1] += rgb[(rx + 1) as usize];
                            avg[2] += rgb[(rx + 2) as usize];
                            avg[3] += 1.0;
                        }
                    }
                    let oi = 4 * ((width * (row + top) + col + left) as usize);
                    for c in 0..3usize {
                        o[oi + c] = (avg[c] / avg[3]).max(0.0);
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

    // Standard Fuji X-Trans 6x6 CFA (0=R,1=G,2=B).
    const XTRANS: [[u8; 6]; 6] = [
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
        [0, 2, 1, 2, 0, 1],
    ];

    #[test]
    fn markesteijn_runs_over_a_full_tile_without_oob() {
        // Exercises the whole tile pipeline (all passes/phases) on a genuine
        // X-Trans CFA. The point is index safety: a stray rfx/hex offset would
        // panic on the bounds-checked Vec. Output correctness isn't asserted
        // (no reference), only that the interior gets written.
        let (w, h) = (80usize, 80usize);
        let inb = vec![0.5_f32; w * h];
        let mut out = vec![-999.0_f32; w * h * 4];
        let xt: Vec<u8> = XTRANS.iter().flatten().copied().collect();
        unsafe {
            darkroom_xtrans_markesteijn(out.as_mut_ptr(), inb.as_ptr(), w as i32, h as i32,
                xt.as_ptr() as *const u8, 1, 9);
        }
        // an interior pixel must have been written by the output pass
        assert_ne!(out[4 * (40 * w + 40)], -999.0, "interior pixel not written");

        // also run the 3-pass variant (ndir=8) for index coverage
        let mut out3 = vec![-999.0_f32; w * h * 4];
        unsafe {
            darkroom_xtrans_markesteijn(out3.as_mut_ptr(), inb.as_ptr(), w as i32, h as i32,
                xt.as_ptr() as *const u8, 3, 9);
        }
        assert_ne!(out3[4 * (40 * w + 40)], -999.0, "3-pass interior not written");
    }
}
