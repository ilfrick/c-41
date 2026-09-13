//! 8-bit RGBA thumbnail downscale with orientation — port of the loop in
//! `dt_iop_flip_and_zoom_8` (`src/develop/imageop_math.c:30`, loop ~69).
//!
//! Each output row walks the input with a float `stepi += scale` column cursor
//! and writes the 4-tap box average `/ 4` (clamped to `0..=255`) of the RGB
//! lanes; alpha is never written. Orientation is applied through the `si` /
//! `sj` pixel strides (flip-X, flip-Y, swap-XY), so all 8 EXIF orientations
//! share one code path. The kernel never upscales (`scale >= 1.0`).
//!
//! Bit-exactness notes (all integer-typed taps, float ops in `f32`):
//! - `scale = max(1, max(iwd / ow, iht / oh))` uses `f32` division and
//!   `f32::max`, matching `fmaxf` (NaN propagates the same way on both sides).
//! - `wd = min(ow, iwd / scale)` converts through `f32` exactly like the C
//!   `MIN` macro (int operand promoted to float, float result truncated to
//!   `u32` on assignment).
//! - `half_pixel = (0.5 * scale) as i32` truncates toward zero, matching the C
//!   float-to-`int32_t` conversion (never rounds up).
//! - `(scale * j) as i32` and `stepi as i32` truncate toward zero per output
//!   pixel, matching the C casts; `stepi` accumulates with `f32` `+= scale`
//!   in the same order, so the truncation sequence is identical.
//! - The tap sum (`u8` values widened to `i32`, at most `4 * 255`) cannot
//!   overflow, and `/ 4` on non-negative `i32` matches C integer division;
//!   the `CLAMP(..., 0, 255)` is a no-op on both sides and is kept for shape.
//! - The in-bounds guard compares flat byte offsets (`in3 + offm >= 0` and
//!   `in3 + offM < total`) exactly as the C pointer comparison does. All tap
//!   bases are 4-byte aligned and `total` is a multiple of 4, so a passing
//!   guard implies every tap byte is strictly inside the buffer (no
//!   over-read); a failing guard leaves the output pixel untouched, matching
//!   the C skip.
//!
//! The reference implementation ([`ref_flip_and_zoom_8`]) regroups the base
//! index, walks with `while` loops, checks each tap with its own pixel-range
//! test, and gathers taps into an array before averaging — same bytes out,
//! different structure.

/// Flip-Y orientation bit (`ORIENTATION_FLIP_Y`).
const ORIENTATION_FLIP_Y: u32 = 1 << 0;
/// Flip-X orientation bit (`ORIENTATION_FLIP_X`).
const ORIENTATION_FLIP_X: u32 = 1 << 1;
/// Transpose orientation bit (`ORIENTATION_SWAP_XY`).
const ORIENTATION_SWAP_XY: u32 = 1 << 2;

/// Bytes per pixel (fixed RGBA8 layout).
const BPP: i64 = 4;

/// Shared header math: oriented input dims, downscale-only factor, output dims.
///
/// Mirrors the C orchestration exactly: `iwd`/`iht` swap under
/// `ORIENTATION_SWAP_XY`, `scale` never drops below `1.0`, and `wd`/`ht` round
/// down through `f32` before truncating to `u32`.
fn header(iw: usize, ih: usize, ow: usize, oh: usize, orientation: u32) -> (f32, u32, u32) {
    let swap = orientation & ORIENTATION_SWAP_XY != 0;
    let iwd = if swap { ih } else { iw } as f32;
    let iht = if swap { iw } else { ih } as f32;
    let scale = 1.0f32.max((iwd / ow as f32).max(iht / oh as f32));
    let wd = (ow as f32).min(iwd / scale) as u32;
    let ht = (oh as f32).min(iht / scale) as u32;
    (scale, wd, ht)
}

/// Shared stride setup: start corner (`ii`, `jj`) and pixel steps (`si`, `sj`).
///
/// Mirrors the C init: `si = 1`, `sj = iw`, negated under the flip bits, then
/// exchanged under `ORIENTATION_SWAP_XY`.
fn strides(iw: i64, ih: i64, orientation: u32) -> (i64, i64, i64, i64) {
    let (mut ii, mut jj) = (0i64, 0i64);
    let (mut si, mut sj) = (1i64, iw);
    if orientation & ORIENTATION_FLIP_Y != 0 {
        jj = ih - jj - 1;
        sj = -sj;
    }
    if orientation & ORIENTATION_FLIP_X != 0 {
        ii = iw - ii - 1;
        si = -si;
    }
    if orientation & ORIENTATION_SWAP_XY != 0 {
        std::mem::swap(&mut si, &mut sj);
    }
    (ii, jj, si, sj)
}

/// Safe 8-bit RGBA thumbnail kernel.
///
/// Ports the `DT_OMP_FOR` row loop of `dt_iop_flip_and_zoom_8`: per output row
/// the input base advances by `(scale * j) as i32` pixels along `sj`, and per
/// output column a `f32` `stepi` cursor accumulates `+= scale` and advances
/// `(stepi as i32)` pixels along `si`. Each in-bounds pixel writes the 4-tap
/// box average of RGB; alpha and out-of-guard pixels are left untouched.
///
/// `inp` holds at least `4 * iw * ih` bytes, `out` at least `4 * ow * oh`
/// bytes; dims are non-zero (validated by the FFI wrapper; `debug_assert`ed
/// here). Returns `(wd, ht)`, the actual thumbnail dims the C writes through
/// its `width`/`height` out-params.
pub fn flip_and_zoom_8(
    out: &mut [u8],
    inp: &[u8],
    iw: usize,
    ih: usize,
    ow: usize,
    oh: usize,
    orientation: u32,
) -> (u32, u32) {
    debug_assert!(iw > 0 && ih > 0 && ow > 0 && oh > 0);
    debug_assert!(inp.len() >= 4 * iw * ih);
    debug_assert!(out.len() >= 4 * ow * oh);

    let (scale, wd, ht) = header(iw, ih, ow, oh, orientation);
    let (ii, jj, si, sj) = strides(iw as i64, ih as i64, orientation);
    // C: (int32_t)(.5f * scale) — truncation, never rounds up.
    let half_pixel = (0.5f32 * scale) as i32 as i64;
    // C names: offm / offM. i64 math: realistic offsets are tiny, and the
    // FFI length checks guarantee the slices cover the full frames.
    let off_min = half_pixel * BPP * (0.min(si).min(sj.min(si + sj)));
    let off_max = half_pixel * BPP * (0.max(si).max(sj.max(si + sj)));
    let total = BPP * iw as i64 * ih as i64;

    for j in 0..ht {
        let row_base = iw as i64 * jj + ii + sj * ((scale * j as f32) as i32 as i64);
        let mut stepi = 0.0f32;
        for i in 0..wd {
            let in3 = BPP * (row_base + (stepi as i32 as i64) * si);
            // Preserved verbatim from the C: the branch predictor hint stays a
            // real branch; a miss leaves the output pixel untouched.
            if in3 + off_min >= 0 && in3 + off_max < total {
                let o = (4u64 * (wd as u64 * j as u64 + i as u64)) as usize;
                let t_sj = BPP * half_pixel * sj;
                let t_si = BPP * half_pixel * si;
                let t_both = BPP * half_pixel * (si + sj);
                for k in 0..3 {
                    let k64 = k as i64;
                    let sum = inp[(in3 + t_sj + k64) as usize] as i32
                        + inp[(in3 + t_both + k64) as usize] as i32
                        + inp[(in3 + t_si + k64) as usize] as i32
                        + inp[(in3 + k64) as usize] as i32;
                    out[o + k] = (sum / 4).clamp(0, 255) as u8;
                }
            }
            stepi += scale;
        }
    }
    (wd, ht)
}

/// Structurally divergent reference for bit-exactness tests.
///
/// Same header math and stride setup (those define the mapping), but the walk
/// uses `while` loops with a regrouped base expression
/// (`iw*jj + ii + sj*row_step + si*col_step`), validates each tap with its own
/// pixel-range test instead of the single byte-range guard, and gathers the
/// four taps into an array before averaging. Must return byte-identical output
/// to [`flip_and_zoom_8`] for every orientation.
#[cfg(test)]
fn ref_flip_and_zoom_8(
    out: &mut [u8],
    inp: &[u8],
    iw: usize,
    ih: usize,
    ow: usize,
    oh: usize,
    orientation: u32,
) -> (u32, u32) {
    let (scale, wd, ht) = header(iw, ih, ow, oh, orientation);
    let (ii, jj, si, sj) = strides(iw as i64, ih as i64, orientation);
    let half_pixel = (0.5f32 * scale) as i32 as i64;
    let npix = iw as i64 * ih as i64;
    let base0 = iw as i64 * jj + ii;

    let mut j: u32 = 0;
    while j < ht {
        let row_step = (scale * j as f32) as i32 as i64;
        let mut stepi = 0.0f32;
        let mut i: u32 = 0;
        while i < wd {
            let col_step = stepi as i32 as i64;
            let base = base0 + sj * row_step + si * col_step;
            // Per-tap pixel-range checks: 4-byte-aligned taps make this
            // exactly equivalent to the kernel single-range byte guard.
            let taps = [base, base + half_pixel * si, base + half_pixel * sj,
                base + half_pixel * (si + sj)];
            let mut ok = true;
            let mut t = 0;
            while t < 4 {
                if taps[t] < 0 || taps[t] >= npix {
                    ok = false;
                }
                t += 1;
            }
            if ok {
                let o = (4u64 * (wd as u64 * j as u64 + i as u64)) as usize;
                let mut k = 0;
                while k < 3 {
                    let mut gathered = [0u8; 4];
                    let mut g = 0;
                    while g < 4 {
                        gathered[g] = inp[(4 * taps[g] + k as i64) as usize];
                        g += 1;
                    }
                    let mut sum = 0i32;
                    let mut g = 0;
                    while g < 4 {
                        sum += gathered[g] as i32;
                        g += 1;
                    }
                    out[o + k as usize] = (sum / 4).clamp(0, 255) as u8;
                    k += 1;
                }
            }
            stepi += scale;
            i += 1;
        }
        j += 1;
    }
    (wd, ht)
}

// ── FFI export ────────────────────────────────────────────────────────────────

/// 8-bit RGBA thumbnail downscale with orientation.
///
/// Ports the loop body of `dt_iop_flip_and_zoom_8`
/// (`src/develop/imageop_math.c:30`); the C wrapper keeps its signature and
/// forwards, so the three `mipmap_cache.c` callers are unchanged.
/// `orientation` carries the `ORIENTATION_FLIP_X` / `ORIENTATION_FLIP_Y` /
/// `ORIENTATION_SWAP_XY` bits; `width` / `height` receive the actual `(wd, ht)`
/// thumbnail dims. Only RGB lanes are written — alpha and guard-skipped pixels
/// keep their incoming bytes. Null pointers, degenerate dims (`<= 0`),
/// overflowing dim products, and short buffers (`in_len < 4*iw*ih` or
/// `out_len < 4*ow*oh`) are no-ops that never touch memory.
///
/// # Safety
/// `inp` must hold at least `in_len` bytes, `out` at least `out_len` bytes,
/// `width` / `height` must be valid for one `u32` write each (unless this is a
/// guarded no-op, in which case nothing is touched).
#[no_mangle]
pub unsafe extern "C" fn darkroom_flip_and_zoom_8(
    inp: *const u8,
    iw: i32,
    ih: i32,
    out: *mut u8,
    ow: i32,
    oh: i32,
    orientation: u32,
    in_len: usize,
    out_len: usize,
    width: *mut u32,
    height: *mut u32,
) {
    if inp.is_null() || out.is_null() || width.is_null() || height.is_null() {
        return;
    }
    if iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 {
        return;
    }
    let (iwu, ihu, owu, ohu) = (iw as usize, ih as usize, ow as usize, oh as usize);
    let need_in = match iwu.checked_mul(ihu).and_then(|p| p.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    let need_out = match owu.checked_mul(ohu).and_then(|p| p.checked_mul(4)) {
        Some(n) => n,
        None => return,
    };
    // Slices can never span more than isize::MAX bytes; bail before building
    // one (debug builds abort on the from_raw_parts precondition otherwise).
    if need_in > isize::MAX as usize || need_out > isize::MAX as usize {
        return;
    }
    if in_len < need_in || out_len < need_out {
        return;
    }
    let inp_slice = std::slice::from_raw_parts(inp, need_in);
    let out_slice = std::slice::from_raw_parts_mut(out, need_out);
    let (wd, ht) = flip_and_zoom_8(out_slice, inp_slice, iwu, ihu, owu, ohu, orientation);
    *width = wd;
    *height = ht;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_fill_u8(buf: &mut [u8], mut s: u32) {
        for v in buf.iter_mut() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = (s >> 16) as u8;
        }
    }

    fn run_kernel(
        iw: usize,
        ih: usize,
        ow: usize,
        oh: usize,
        orientation: u32,
        seed: u32,
    ) -> (Vec<u8>, u32, u32) {
        let mut inp = vec![0u8; 4 * iw * ih];
        lcg_fill_u8(&mut inp, seed);
        let mut out = vec![0xA5u8; 4 * ow * oh];
        let (wd, ht) = flip_and_zoom_8(&mut out, &inp, iw, ih, ow, oh, orientation);
        (out, wd, ht)
    }

    fn run_ref(
        iw: usize,
        ih: usize,
        ow: usize,
        oh: usize,
        orientation: u32,
        seed: u32,
    ) -> (Vec<u8>, u32, u32) {
        let mut inp = vec![0u8; 4 * iw * ih];
        lcg_fill_u8(&mut inp, seed);
        let mut out = vec![0xA5u8; 4 * ow * oh];
        let (wd, ht) = ref_flip_and_zoom_8(&mut out, &inp, iw, ih, ow, oh, orientation);
        (out, wd, ht)
    }

    #[test]
    fn golden_4x4_downscale_to_2x2() {
        // Pixel p holds (p, 100+p, 200+p, 0xA5 sentinel alpha).
        // scale = 2, half_pixel = 1: 2x2 box taps at {0,1,4,5} offsets.
        let (iw, ih, ow, oh) = (4usize, 4usize, 2usize, 2usize);
        let mut inp = vec![0u8; 4 * iw * ih];
        for p in 0..16 {
            inp[4 * p] = p as u8;
            inp[4 * p + 1] = 100 + p as u8;
            inp[4 * p + 2] = 200 + p as u8;
            inp[4 * p + 3] = 0xA5;
        }
        let mut out = vec![0xA5u8; 4 * ow * oh];
        let (wd, ht) = flip_and_zoom_8(&mut out, &inp, iw, ih, ow, oh, 0);
        assert_eq!((wd, ht), (2, 2));
        let expected: Vec<u8> = vec![
            2, 102, 202, 0xA5, 4, 104, 204, 0xA5, //
            10, 110, 210, 0xA5, 12, 112, 212, 0xA5,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn all_eight_orientations_match_reference() {
        // Non-square dims so SWAP_XY actually transposes the mapping.
        for orientation in 0..8u32 {
            let (direct, wd, ht) = run_kernel(6, 4, 3, 2, orientation, 0x1234);
            let (reference, rwd, rht) = run_ref(6, 4, 3, 2, orientation, 0x1234);
            assert_eq!((wd, ht), (rwd, rht), "orientation={orientation}");
            assert_eq!(direct, reference, "orientation={orientation}");
        }
    }

    #[test]
    fn orientations_match_reference_across_shapes() {
        // Square, wide, tall, and prime-ish dims with several seeds.
        let shapes = [(8, 8, 4, 4), (7, 3, 5, 2), (3, 9, 2, 6), (5, 5, 5, 5), (9, 7, 4, 3)];
        for (n, (iw, ih, ow, oh)) in shapes.iter().enumerate() {
            for orientation in 0..8u32 {
                let seed = 0xBEEF + n as u32 * 0x101;
                let (direct, wd, ht) = run_kernel(*iw, *ih, *ow, *oh, orientation, seed);
                let (reference, rwd, rht) = run_ref(*iw, *ih, *ow, *oh, orientation, seed);
                assert_eq!((wd, ht), (rwd, rht), "shape={n} orientation={orientation}");
                assert_eq!(direct, reference, "shape={n} orientation={orientation}");
            }
        }
    }

    #[test]
    fn never_upscales_small_input_to_large_output() {
        // ow/oh exceed the input: scale pins at 1, wd/ht cover the input only,
        // half_pixel truncates to 0 so each output pixel copies its source.
        let (iw, ih, ow, oh) = (2usize, 2usize, 8usize, 8usize);
        let mut inp = vec![0u8; 4 * iw * ih];
        lcg_fill_u8(&mut inp, 0x77);
        let mut out = vec![0xA5u8; 4 * ow * oh];
        let (wd, ht) = flip_and_zoom_8(&mut out, &inp, iw, ih, ow, oh, 0);
        assert_eq!((wd, ht), (2, 2));
        // RGB lanes copy the source pixels; alpha lanes keep the 0xA5 fill
        // (kernel never writes k=3, input alpha is LCG noise).
        for p in 0..4 {
            assert_eq!(&out[4 * p..4 * p + 3], &inp[4 * p..4 * p + 3]);
            assert_eq!(out[4 * p + 3], 0xA5);
        }
        assert!(out[16..].iter().all(|&b| b == 0xA5));
    }

    #[test]
    fn alpha_lanes_preserved_everywhere() {
        // Sentinel alpha in both input and output: output alpha must equal the
        // incoming output bytes for every orientation (kernel never writes k=3).
        for orientation in 0..8u32 {
            let (out, _, _) = run_kernel(6, 4, 3, 2, orientation, 0x55);
            for px in out.chunks_exact(4) {
                assert_eq!(px[3], 0xA5, "orientation={orientation}");
            }
        }
    }

    #[test]
    fn ffi_round_trip_matches_kernel() {
        for orientation in 0..8u32 {
            let (iw, ih, ow, oh) = (6usize, 4usize, 3usize, 2usize);
            let mut inp = vec![0u8; 4 * iw * ih];
            lcg_fill_u8(&mut inp, 0xC0DE);
            let mut ffi_out = vec![0xA5u8; 4 * ow * oh];
            let mut direct_out = vec![0xA5u8; 4 * ow * oh];
            let (wd, ht) =
                flip_and_zoom_8(&mut direct_out, &inp, iw, ih, ow, oh, orientation);
            let mut fwd: u32 = 0xDEAD;
            let mut fht: u32 = 0xDEAD;
            unsafe {
                darkroom_flip_and_zoom_8(
                    inp.as_ptr(),
                    iw as i32,
                    ih as i32,
                    ffi_out.as_mut_ptr(),
                    ow as i32,
                    oh as i32,
                    orientation,
                    inp.len(),
                    ffi_out.len(),
                    &mut fwd,
                    &mut fht,
                );
            }
            assert_eq!((fwd, fht), (wd, ht), "orientation={orientation}");
            assert_eq!(ffi_out, direct_out, "orientation={orientation}");
        }
    }

    #[test]
    fn ffi_null_guards_leave_memory_untouched() {
        let inp = vec![1u8; 64];
        let mut out = vec![2u8; 16];
        let mut w: u32 = 7;
        let mut h: u32 = 9;
        unsafe {
            darkroom_flip_and_zoom_8(
                std::ptr::null(),
                4, 4, out.as_mut_ptr(), 2, 2, 0, 64, 16, &mut w, &mut h,
            );
            darkroom_flip_and_zoom_8(
                inp.as_ptr(),
                4, 4, std::ptr::null_mut(), 2, 2, 0, 64, 16, &mut w, &mut h,
            );
            darkroom_flip_and_zoom_8(
                inp.as_ptr(),
                4, 4, out.as_mut_ptr(), 2, 2, 0, 64, 16, std::ptr::null_mut(), &mut h,
            );
            darkroom_flip_and_zoom_8(
                inp.as_ptr(),
                4, 4, out.as_mut_ptr(), 2, 2, 0, 64, 16, &mut w, std::ptr::null_mut(),
            );
        }
        assert_eq!(out, vec![2u8; 16]);
        assert_eq!((w, h), (7, 9));
    }

    #[test]
    fn ffi_degenerate_dims_are_noops() {
        let inp = vec![1u8; 64];
        let bad_dims = [(0, 4, 2, 2), (4, 0, 2, 2), (4, 4, 0, 2), (4, 4, 2, 0), (-3, 4, 2, 2),
            (4, -4, 2, 2), (4, 4, -2, 2)];
        for (iw, ih, ow, oh) in bad_dims {
            let mut out = vec![2u8; 16];
            let mut w: u32 = 7;
            let mut h: u32 = 9;
            unsafe {
                darkroom_flip_and_zoom_8(
                    inp.as_ptr(), iw, ih, out.as_mut_ptr(), ow, oh, 0, 64, 16, &mut w, &mut h,
                );
            }
            assert_eq!(out, vec![2u8; 16], "dims=({iw},{ih},{ow},{oh})");
            assert_eq!((w, h), (7, 9), "dims=({iw},{ih},{ow},{oh})");
        }
    }

    #[test]
    fn ffi_short_and_overflowing_lengths_are_noops() {
        let inp = vec![1u8; 64]; // 4x4 needs 64 bytes in, 2x2 needs 16 out
        // Short input length.
        let mut out = vec![2u8; 16];
        let mut w: u32 = 7;
        let mut h: u32 = 9;
        unsafe {
            darkroom_flip_and_zoom_8(
                inp.as_ptr(), 4, 4, out.as_mut_ptr(), 2, 2, 0, 63, 16, &mut w, &mut h,
            );
        }
        assert_eq!(out, vec![2u8; 16]);
        assert_eq!((w, h), (7, 9));
        // Short output length.
        let mut out = vec![2u8; 16];
        unsafe {
            darkroom_flip_and_zoom_8(
                inp.as_ptr(), 4, 4, out.as_mut_ptr(), 2, 2, 0, 64, 15, &mut w, &mut h,
            );
        }
        assert_eq!(out, vec![2u8; 16]);
        assert_eq!((w, h), (7, 9));
        // Extreme dims: 4*iw*ih exceeds isize::MAX, so no slice is ever
        // built and no memory is touched (honest small buffers throughout).
        let mut out = vec![2u8; 16];
        unsafe {
            darkroom_flip_and_zoom_8(
                inp.as_ptr(), i32::MAX, i32::MAX, out.as_mut_ptr(), 2, 2, 0, 64, 16, &mut w,
                &mut h,
            );
            darkroom_flip_and_zoom_8(
                inp.as_ptr(), 4, 4, out.as_mut_ptr(), i32::MAX, i32::MAX, 0, 64, 16, &mut w,
                &mut h,
            );
        }
        assert_eq!(out, vec![2u8; 16]);
        assert_eq!((w, h), (7, 9));
    }
}
