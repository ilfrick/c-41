//! Flat CYGM/RGB matrix kernels ported from `src/common/colorspaces.c`
//! (m4-174). Three loops are ported here:
//! - `dt_colorspaces_cygm_to_rgb` (colorspaces.c:2462, the former
//!   element-wise loop): in-place 4x3 CYGM->RGB matrix per packed
//!   4-float pixel. LIVE and bulk-called (demosaic.c passes
//!   `width * t_rows` pixels for 4-bayer VNG4 output; invert.c passes 1).
//! - `dt_colorspaces_rgb_to_cygm` (colorspaces.c:2479, the former
//!   element-wise loop): in-place 4x3 RGB->CYGM matrix over pixels at
//!   stride 3. LIVE but only ever called with num = 1 (invert.c:106 and
//!   :218, on a `dt_aligned_pixel_t`).
//! - `dt_colorspaces_cygm_apply_coeffs_to_rgb` (colorspaces.c:2446, the
//!   former element-wise loop): 3x3 RGB->WB'd-RGB matrix, separate
//!   in/out buffers. DEAD exported code — declared in colorspaces.h but
//!   with zero callers anywhere in src/ (verified by grep); ported only
//!   to move the loop body out of C. The scalar matrix setup before the
//!   loop (CAM_to_RGB_WB scaling and the RGB_to_RGB_WB double matmul)
//!   stays in C; the kernel receives the precomputed 3x3 doubles.
//!
//! Bit-exactness notes:
//! - The C file is compiled with the repo-wide Release flags
//!   `-O3 -ffast-math -fno-finite-math-only` and GCC's default
//!   `-ffp-contract=fast` for C99+. The ported loops are dense
//!   multiply-accumulate chains, so the C binary may contract `a*b + c`
//!   sites into FMAs where Rust (no contraction in the release profile)
//!   keeps separate mul/add ops. The C-vs-Rust difference is the
//!   order-ULP class accepted repo-wide (cf. `guided_filter.rs`,
//!   `eigf.rs`); `-ffast-math` reassociation has no clean ULP bound.
//! - Numeric contract replicated exactly: every accumulation step is
//!   `o = (float)((double)o + (double)m * (double)v)` — the matrix
//!   product is computed in f64, added to the promoted f32 accumulator
//!   in f64, and rounded back to f32 at EVERY step (the C accumulators
//!   are `float` locals / float memory, the matrices are `double`). The
//!   kernels never accumulate in f64 across steps.
//! - Matrix layouts (row-major flat doubles, exactly as the C 2-D
//!   parameters decay):
//!   `cygm_to_rgb`: CAM_to_RGB is 3x4 = 12 doubles, `[c][k]` at `c*4+k`.
//!   `rgb_to_cygm`: RGB_to_CAM is 4x3 = 12 doubles, `[c][k]` at `c*3+k`.
//!   `cygm_apply_coeffs_to_rgb`: RGB_to_RGB_WB is 3x3 = 9 doubles,
//!   `[a][b]` at `a*3+b`.
//!
//! The rgb_to_cygm stride quirk (C carries a literal
//! `//FIXME: is this correct or should it be i*4 ?`): the loop reads
//! pixels at stride 3 (`in = &out[i*3]`) but writes FOUR floats
//! (`in[c] = o[c]` for c in 0..4), so pixel i's 4th write lands at
//! `out[i*3+3]` — pixel i+1's first READ. The loop is therefore a
//! sequential cross-pixel chain: pixel i+1's o[0] folds in pixel i's
//! o[3]. Facts (verified against the callers):
//! - the only real callers pass num = 1, so the chain never executes in
//!   practice (at num = 1 the 4th write just overwrites the alpha slot
//!   of the single `dt_aligned_pixel_t`);
//! - the parallel C was an OpenMP data race for num > 1
//!   (nondeterministic); the sequential Rust port is well-defined for
//!   any num and matches C's serial fallback deterministically.
//! The buffer therefore needs `3 * num + 1` floats to hold every
//! element the C loop can touch; the kernel caps the pixel count at
//! what fits (`(len - 1) / 3`), like the bounds-clamped iteration in
//! the other ported modules.
//!
//! `cygm_apply_coeffs_to_rgb` reads only input channels 0..3
//! (b runs 0..3 in the C sum, so channels 0..2) and leaves the output
//! alpha slot untouched. out and in are separate buffers in the C
//! signature; since the function is dead, the no-alias contract is ours
//! to set and the kernel requires `out` and `input` not to overlap.

/// In-place CYGM -> RGB matrix over `num` packed 4-float pixels.
///
/// Port of the former element-wise loop at colorspaces.c:2462. Per
/// pixel: reads all four floats (the 4th channel IS an input), computes
/// `o[c] = Σ_{k=0..3} matrix[c*4+k] * v[k]` for c in 0..3 with the
/// f64-product/f32-round-per-step contract (see module docs), writes
/// o back to floats 0..3 and leaves float 3 untouched on write.
pub fn cygm_to_rgb(buf: &mut [f32], num: usize, matrix: &[f64]) {
    let n = num.min(buf.len() / 4);
    for i in 0..n {
        let p = 4 * i;
        let v = [buf[p], buf[p + 1], buf[p + 2], buf[p + 3]];
        let mut o0 = 0.0f32;
        let mut o1 = 0.0f32;
        let mut o2 = 0.0f32;
        for k in 0..4 {
            let vk = v[k] as f64;
            o0 = ((o0 as f64) + matrix[k] * vk) as f32;
            o1 = ((o1 as f64) + matrix[4 + k] * vk) as f32;
            o2 = ((o2 as f64) + matrix[8 + k] * vk) as f32;
        }
        buf[p] = o0;
        buf[p + 1] = o1;
        buf[p + 2] = o2;
    }
}

/// In-place RGB -> CYGM matrix over `num` pixels at stride 3.
///
/// Port of the former element-wise loop at colorspaces.c:2479 (the one
/// carrying the C `//FIXME: is this correct or should it be i*4 ?`).
/// Per pixel i: reads `buf[i*3 .. i*3+3]`, computes
/// `o[c] = Σ_{k=0..2} matrix[c*3+k] * v[k]` for c in 0..4 (same
/// f64-product/f32-round-per-step contract), then writes all FOUR o
/// values at `buf[i*3 .. i*3+4]` — the 4th write overwrites pixel i+1's
/// first read slot, reproducing the C sequential chain exactly (see
/// module docs; real callers only ever pass num = 1). `buf` must hold
/// at least `3 * num + 1` floats for all num pixels to be processed.
pub fn rgb_to_cygm(buf: &mut [f32], num: usize, matrix: &[f64]) {
    let n = num.min(buf.len().saturating_sub(1) / 3);
    for i in 0..n {
        let p = 3 * i;
        let v = [buf[p], buf[p + 1], buf[p + 2]];
        let mut o0 = 0.0f32;
        let mut o1 = 0.0f32;
        let mut o2 = 0.0f32;
        let mut o3 = 0.0f32;
        for k in 0..3 {
            let vk = v[k] as f64;
            o0 = ((o0 as f64) + matrix[k] * vk) as f32;
            o1 = ((o1 as f64) + matrix[3 + k] * vk) as f32;
            o2 = ((o2 as f64) + matrix[6 + k] * vk) as f32;
            o3 = ((o3 as f64) + matrix[9 + k] * vk) as f32;
        }
        buf[p] = o0;
        buf[p + 1] = o1;
        buf[p + 2] = o2;
        buf[p + 3] = o3;
    }
}

/// RGB -> white-balanced-RGB matrix over `num` packed 4-float pixels.
///
/// Port of the former element-wise loop at colorspaces.c:2446 inside
/// `dt_colorspaces_cygm_apply_coeffs_to_rgb` — DEAD exported code with
/// zero callers in src/ (declared in colorspaces.h only); the scalar
/// CAM_to_RGB_WB / RGB_to_RGB_WB matrix setup stays in C and only the
/// precomputed 3x3 double matrix arrives here. Per pixel: reads input
/// channels 0..2, computes `out[a] = Σ_{b=0..2} matrix[a*3+b] * v[b]`
/// for a in 0..3 with the f64-product/f32-round-per-step contract, and
/// leaves the output alpha slot untouched. `out` and `input` are
/// distinct buffers and must not alias (dead function, no callers —
/// the contract is ours to set).
pub fn cygm_apply_coeffs_to_rgb(out: &mut [f32], input: &[f32], num: usize, matrix: &[f64]) {
    let n = num.min(out.len() / 4).min(input.len() / 4);
    for i in 0..n {
        let p = 4 * i;
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        for b in 0..3 {
            let vb = input[p + b] as f64;
            a0 = ((a0 as f64) + matrix[b] * vb) as f32;
            a1 = ((a1 as f64) + matrix[3 + b] * vb) as f32;
            a2 = ((a2 as f64) + matrix[6 + b] * vb) as f32;
        }
        out[p] = a0;
        out[p + 1] = a1;
        out[p + 2] = a2;
        // alpha (float 3) is untouched — passthrough
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `buf` must hold at least `num * 4` floats (packed CYGM pixels on
/// entry, RGB pixels on return); `matrix` must hold 12 doubles
/// (CAM_to_RGB, row-major 3x4).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorspaces_cygm_to_rgb(
    buf: *mut f32,
    num: usize,
    matrix: *const f64,
) {
    if buf.is_null() || matrix.is_null() || num == 0 || num > i32::MAX as usize {
        return;
    }
    let buf = std::slice::from_raw_parts_mut(buf, num * 4);
    let matrix = std::slice::from_raw_parts(matrix, 12);
    cygm_to_rgb(buf, num, matrix);
}

/// # Safety
/// `buf` must hold at least `3 * num + 1` floats — the stride-3 /
/// write-4 quirk means pixel `num - 1` writes one element past the
/// packed 3*num region (the only live caller passes num = 1 on a
/// 4-float `dt_aligned_pixel_t`). `matrix` must hold 12 doubles
/// (RGB_to_CAM, row-major 4x3).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorspaces_rgb_to_cygm(
    buf: *mut f32,
    num: usize,
    matrix: *const f64,
) {
    if buf.is_null() || matrix.is_null() || num == 0 || num > i32::MAX as usize {
        return;
    }
    let buf = std::slice::from_raw_parts_mut(buf, num * 3 + 1);
    let matrix = std::slice::from_raw_parts(matrix, 12);
    rgb_to_cygm(buf, num, matrix);
}

/// # Safety
/// `out` and `input` must each hold at least `num * 4` floats and must
/// not alias; `matrix` must hold 9 doubles (RGB_to_RGB_WB, row-major
/// 3x3). Currently no C callers exist (dead exported function).
#[no_mangle]
pub unsafe extern "C" fn darkroom_colorspaces_cygm_apply_coeffs(
    out: *mut f32,
    input: *const f32,
    num: usize,
    matrix: *const f64,
) {
    if out.is_null() || input.is_null() || matrix.is_null() || num == 0 || num > i32::MAX as usize {
        return;
    }
    let out = std::slice::from_raw_parts_mut(out, num * 4);
    let input = std::slice::from_raw_parts(input, num * 4);
    let matrix = std::slice::from_raw_parts(matrix, 9);
    cygm_apply_coeffs_to_rgb(out, input, num, matrix);
}

// ── Independent reference implementations for bit-exactness tests ─────────────
//
// Structurally divergent from the kernels (chunk/iterator walks instead of
// index loops, per-pixel arrays instead of unrolled scalars) but keeping the
// exact FP evaluation order: f64 product, f64 add of the promoted f32
// accumulator, rounded back to f32 every step.

#[allow(dead_code)]
fn ref_step(acc: f32, m: f64, v: f32) -> f32 {
    (acc as f64 + m * v as f64) as f32
}

#[allow(dead_code)]
fn ref_cygm_to_rgb(buf: &mut [f32], num: usize, matrix: &[f64]) {
    for px in buf.chunks_exact_mut(4).take(num) {
        let v = [px[0], px[1], px[2], px[3]];
        let mut o = [0.0f32; 3];
        for (c, oc) in o.iter_mut().enumerate() {
            for (k, vk) in v.iter().enumerate() {
                *oc = ref_step(*oc, matrix[4 * c + k], *vk);
            }
        }
        px[0] = o[0];
        px[1] = o[1];
        px[2] = o[2];
    }
}

#[allow(dead_code)]
fn ref_rgb_to_cygm(buf: &mut [f32], num: usize, matrix: &[f64]) {
    let n = num.min(buf.len().saturating_sub(1) / 3);
    let mut i = 0;
    while i < n {
        let p = 3 * i;
        let v = [buf[p], buf[p + 1], buf[p + 2]];
        let o: Vec<f32> = (0..4)
            .map(|c| {
                v.iter()
                    .enumerate()
                    .fold(0.0f32, |acc, (k, vk)| ref_step(acc, matrix[3 * c + k], *vk))
            })
            .collect();
        for (c, oc) in o.iter().enumerate() {
            buf[p + c] = *oc;
        }
        i += 1;
    }
}

#[allow(dead_code)]
fn ref_cygm_apply_coeffs_to_rgb(out: &mut [f32], input: &[f32], num: usize, matrix: &[f64]) {
    let n = num.min(out.len() / 4).min(input.len() / 4);
    let pairs = out.chunks_exact_mut(4).zip(input.chunks_exact(4)).take(n);
    for (opx, ipx) in pairs {
        let mut o = [0.0f32; 3];
        for (a, oa) in o.iter_mut().enumerate() {
            for (b, vb) in ipx[..3].iter().enumerate() {
                *oa = ref_step(*oa, matrix[3 * a + b], *vb);
            }
        }
        opx[0] = o[0];
        opx[1] = o[1];
        opx[2] = o[2];
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    // ── cygm_to_rgb ─────────────────────────────────────────────────────────

    #[test]
    fn cygm_to_rgb_pin() {
        // Power-of-two matrix and pixels -> every product and sum is exact,
        // so the pin also validates the f64/f32 promotion contract.
        // Row 2 = [0, 0, 1, 1] folds the 4th channel into blue: it must be
        // READ as input and left untouched on write.
        let matrix: [f64; 12] = [
            0.5, 0.25, 0.125, 0.0625, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 1.0, // row 2
        ];
        let mut buf = vec![1.0f32, 2.0, 4.0, 8.0, -1.5, 3.0, 6.0, 0.75];
        cygm_to_rgb(&mut buf, 2, &matrix);
        // pixel 0: o0 = 0.5+0.5+0.5+0.5 = 2.0; o1 = 2.0; o2 = 4.0+8.0 = 12.0
        // pixel 1: o0 = -0.75+0.75+0.75+0.046875 = 0.796875; o1 = 3.0;
        //          o2 = 6.0+0.75 = 6.75
        assert_eq!(
            buf,
            vec![2.0f32, 2.0, 12.0, 8.0, 0.796875, 3.0, 6.75, 0.75]
        );
        // 4th floats untouched on write (8.0 and 0.75 above).
    }

    #[test]
    fn cygm_to_rgb_per_step_rounding_contract() {
        // The accumulation rounds to f32 after EVERY step:
        // o = (float)((double)o + m * (double)v), not a single f64 sum.
        //
        // Case 1 (m = 1/3 four times): per-step rounding and one-shot f64
        // rounding happen to agree here, so this leg alone is NOT
        // discriminating — it pins the literal expression shape.
        let m = 1.0f64 / 3.0;
        let mut e = 0.0f32;
        e = ((e as f64) + m * 1.0f64) as f32;
        e = ((e as f64) + m * 1.0f64) as f32;
        e = ((e as f64) + m * 1.0f64) as f32;
        e = ((e as f64) + m * 1.0f64) as f32;
        let matrix = [m; 12];
        let mut buf = vec![1.0f32; 4];
        cygm_to_rgb(&mut buf, 1, &matrix);
        assert_eq!(buf[0], e);
        //
        // Case 2 (DISCRIMINATING): accumulator parked at 2^24 (f32 spacing
        // 2.0 there), then three +1.0 products. Per-step: 2^24+1 is an
        // exact halfway tie in f32 → ties-to-even keeps 2^24 every time →
        // result 16777216.0. One-shot f64 sum: 2^24+3 = 16777219 → also an
        // exact tie → ties-to-even rounds UP to 16777220.0. The two
        // strategies demonstrably differ; the kernel must match per-step.
        let matrix2: [f64; 12] = [
            16777216.0, 1.0, 1.0, 1.0, // row 0
            0.0, 0.0, 0.0, 0.0, // row 1
            0.0, 0.0, 0.0, 0.0, // row 2
        ];
        let mut buf2 = vec![1.0f32; 4];
        cygm_to_rgb(&mut buf2, 1, &matrix2);
        assert_eq!(buf2[0], 16777216.0f32);
        assert_ne!(buf2[0], 16777220.0f32); // proves discrimination
    }

    #[test]
    fn cygm_to_rgb_matches_reference_over_lcg() {
        let mut buf = vec![0.0f32; 333 * 4];
        lcg_fill(&mut buf, 0xC41A, 2.0);
        let mut m32 = vec![0.0f32; 12];
        lcg_fill(&mut m32, 0xC41B, 1.0);
        let matrix: Vec<f64> = m32.iter().map(|&x| x as f64).collect();

        let mut direct = buf.clone();
        let mut reference = buf.clone();
        cygm_to_rgb(&mut direct, 333, &matrix);
        ref_cygm_to_rgb(&mut reference, 333, &matrix);
        assert_eq!(direct, reference);
    }

    #[test]
    fn ffi_cygm_to_rgb_round_trip() {
        let mut buf = vec![0.0f32; 257 * 4];
        lcg_fill(&mut buf, 0xC41C, 2.0);
        let mut m32 = vec![0.0f32; 12];
        lcg_fill(&mut m32, 0xC41D, 1.0);
        let matrix: Vec<f64> = m32.iter().map(|&x| x as f64).collect();

        let mut ffi_buf = buf.clone();
        let mut direct_buf = buf.clone();
        unsafe {
            darkroom_colorspaces_cygm_to_rgb(ffi_buf.as_mut_ptr(), 257, matrix.as_ptr());
        }
        cygm_to_rgb(&mut direct_buf, 257, &matrix);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_cygm_to_rgb_guards() {
        let matrix = [0.5f64; 12];
        let mut buf = vec![1.0f32; 8];
        unsafe {
            darkroom_colorspaces_cygm_to_rgb(std::ptr::null_mut(), 2, matrix.as_ptr());
            darkroom_colorspaces_cygm_to_rgb(buf.as_mut_ptr(), 2, std::ptr::null());
            darkroom_colorspaces_cygm_to_rgb(buf.as_mut_ptr(), 0, matrix.as_ptr());
            darkroom_colorspaces_cygm_to_rgb(
                buf.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                matrix.as_ptr(),
            );
        }
        assert_eq!(buf, vec![1.0f32; 8]); // untouched
    }

    // ── rgb_to_cygm ─────────────────────────────────────────────────────────

    #[test]
    fn rgb_to_cygm_num1_pin() {
        // Mirrors the only real callers (invert.c:106/:218): num = 1 on a
        // dt_aligned_pixel_t. The 4th output overwrites the alpha slot —
        // the stride-3/read-3/write-4 quirk of the C FIXME.
        let matrix: [f64; 12] = [
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
            0.25, 0.5, 0.25, // row 3
        ];
        let mut buf = vec![1.0f32, 2.0, 4.0, 999.0];
        rgb_to_cygm(&mut buf, 1, &matrix);
        assert_eq!(buf, vec![1.0f32, 2.0, 4.0, 2.25]);
    }

    #[test]
    fn rgb_to_cygm_num3_chain_pin() {
        // Hand-computed demonstration of the sequential cross-pixel chain
        // for num > 1 (the C carries `//FIXME: is this correct or should
        // it be i*4 ?`; the parallel C was an OpenMP data race for num>1,
        // this port matches C's serial fallback deterministically).
        // Pixel i writes buf[i*3 .. i*3+4]; pixel 1's first input is
        // pixel 0's 4th output, pixel 2's first input is pixel 1's 4th
        // output.
        let matrix: [f64; 12] = [
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
            1.0, 1.0, 1.0, // row 3
        ];
        // 3 pixels at stride 3 + the one overflow float.
        let mut buf = vec![1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 7.0];
        rgb_to_cygm(&mut buf, 3, &matrix);
        // pixel 0: in = (1,1,1) -> o = (1,1,1,3); write covers buf[0..4]
        // pixel 1: in = (3,2,2) -> o = (3,2,2,7); write covers buf[3..7]
        //          (first input 3 IS pixel 0's 4th output)
        // pixel 2: in = (7,3,3) -> o = (7,3,3,13); write covers buf[6..10]
        //          (first input 7 IS pixel 1's 4th output)
        assert_eq!(
            buf,
            vec![1.0f32, 1.0, 1.0, 3.0, 2.0, 2.0, 7.0, 3.0, 3.0, 13.0]
        );
    }

    #[test]
    fn rgb_to_cygm_matches_reference_over_lcg() {
        let num = 111usize;
        let mut buf = vec![0.0f32; num * 3 + 1];
        lcg_fill(&mut buf, 0xC41E, 2.0);
        let mut m32 = vec![0.0f32; 12];
        lcg_fill(&mut m32, 0xC41F, 1.0);
        let matrix: Vec<f64> = m32.iter().map(|&x| x as f64).collect();

        let mut direct = buf.clone();
        let mut reference = buf.clone();
        rgb_to_cygm(&mut direct, num, &matrix);
        ref_rgb_to_cygm(&mut reference, num, &matrix);
        assert_eq!(direct, reference);
    }

    #[test]
    fn ffi_rgb_to_cygm_round_trip() {
        let num = 65usize;
        let mut buf = vec![0.0f32; num * 3 + 1];
        lcg_fill(&mut buf, 0xC420, 2.0);
        let mut m32 = vec![0.0f32; 12];
        lcg_fill(&mut m32, 0xC421, 1.0);
        let matrix: Vec<f64> = m32.iter().map(|&x| x as f64).collect();

        let mut ffi_buf = buf.clone();
        let mut direct_buf = buf.clone();
        unsafe {
            darkroom_colorspaces_rgb_to_cygm(ffi_buf.as_mut_ptr(), num, matrix.as_ptr());
        }
        rgb_to_cygm(&mut direct_buf, num, &matrix);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_rgb_to_cygm_guards() {
        let matrix = [0.5f64; 12];
        let mut buf = vec![1.0f32; 10];
        unsafe {
            darkroom_colorspaces_rgb_to_cygm(std::ptr::null_mut(), 2, matrix.as_ptr());
            darkroom_colorspaces_rgb_to_cygm(buf.as_mut_ptr(), 2, std::ptr::null());
            darkroom_colorspaces_rgb_to_cygm(buf.as_mut_ptr(), 0, matrix.as_ptr());
            darkroom_colorspaces_rgb_to_cygm(
                buf.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                matrix.as_ptr(),
            );
        }
        assert_eq!(buf, vec![1.0f32; 10]); // untouched
    }

    // ── cygm_apply_coeffs_to_rgb (dead function, no callers) ──────────────────

    #[test]
    fn apply_coeffs_pin() {
        // DEAD exported code — currently uncalled anywhere in src/; pinned
        // anyway since the port replaces the former loop body.
        // Identity matrix: RGB passes through, alpha slot untouched.
        let mut out = vec![-7.0f32; 4];
        let input = vec![0.5f32, 0.25, 0.125, 77.0];
        let identity: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        cygm_apply_coeffs_to_rgb(&mut out, &input, 1, &identity);
        assert_eq!(out, vec![0.5f32, 0.25, 0.125, -7.0]);

        // Non-identity diagonal: o = (1*v0, 2*v1, 0.5*v2); the input alpha
        // (77.0) is NOT read (b runs 0..2 only) and out alpha stays -7.
        let mut out2 = vec![-7.0f32; 4];
        let diag: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.5];
        cygm_apply_coeffs_to_rgb(&mut out2, &input, 1, &diag);
        assert_eq!(out2, vec![0.5f32, 0.5, 0.0625, -7.0]);
    }

    #[test]
    fn apply_coeffs_alpha_passthrough_pin() {
        // A dense (non-diagonal) matrix proving no output channel depends
        // on the input alpha: garbage in input[3] must not change anything.
        let matrix: [f64; 9] = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 1.0, 2.0, 4.0];
        let base = vec![1.0f32, 2.0, 3.0];
        let mut out_a = vec![0.0f32; 8];
        let mut out_b = vec![0.0f32; 8];
        let in_a: Vec<f32> = base.iter().copied().chain([42.0]).collect();
        let in_b: Vec<f32> = base.iter().copied().chain([-42.0]).collect();
        cygm_apply_coeffs_to_rgb(&mut out_a, &in_a, 2, &matrix);
        cygm_apply_coeffs_to_rgb(&mut out_b, &in_b, 2, &matrix);
        assert_eq!(out_a, out_b);
        // pixel 0: o0 = 0.5+0.5+0.375 = 1.375; o1 = 0.0625+0.0625+0.046875
        //          = 0.171875; o2 = 1+4+12 = 17
        assert_eq!(out_a[0], 1.375);
        assert_eq!(out_a[1], 0.171875);
        assert_eq!(out_a[2], 17.0);
    }

    #[test]
    fn apply_coeffs_matches_reference_over_lcg() {
        let mut input = vec![0.0f32; 190 * 4];
        lcg_fill(&mut input, 0xC422, 2.0);
        let mut out = vec![0.0f32; 190 * 4];
        lcg_fill(&mut out, 0xC423, 2.0);
        let mut m32 = vec![0.0f32; 9];
        lcg_fill(&mut m32, 0xC424, 1.0);
        let matrix: Vec<f64> = m32.iter().map(|&x| x as f64).collect();

        let mut direct_out = out.clone();
        let mut reference_out = out.clone();
        cygm_apply_coeffs_to_rgb(&mut direct_out, &input, 190, &matrix);
        ref_cygm_apply_coeffs_to_rgb(&mut reference_out, &input, 190, &matrix);
        assert_eq!(direct_out, reference_out);
    }

    #[test]
    fn ffi_apply_coeffs_round_trip() {
        let mut input = vec![0.0f32; 96 * 4];
        lcg_fill(&mut input, 0xC425, 2.0);
        let mut out = vec![0.0f32; 96 * 4];
        lcg_fill(&mut out, 0xC426, 2.0);
        let mut m32 = vec![0.0f32; 9];
        lcg_fill(&mut m32, 0xC427, 1.0);
        let matrix: Vec<f64> = m32.iter().map(|&x| x as f64).collect();

        let mut ffi_out = out.clone();
        let mut direct_out = out.clone();
        unsafe {
            darkroom_colorspaces_cygm_apply_coeffs(
                ffi_out.as_mut_ptr(),
                input.as_ptr(),
                96,
                matrix.as_ptr(),
            );
        }
        cygm_apply_coeffs_to_rgb(&mut direct_out, &input, 96, &matrix);
        assert_eq!(ffi_out, direct_out);
    }

    #[test]
    fn ffi_apply_coeffs_guards() {
        let matrix = [0.5f64; 9];
        let input = vec![1.0f32; 8];
        let mut out = vec![1.0f32; 8];
        unsafe {
            darkroom_colorspaces_cygm_apply_coeffs(
                std::ptr::null_mut(),
                input.as_ptr(),
                2,
                matrix.as_ptr(),
            );
            darkroom_colorspaces_cygm_apply_coeffs(
                out.as_mut_ptr(),
                std::ptr::null(),
                2,
                matrix.as_ptr(),
            );
            darkroom_colorspaces_cygm_apply_coeffs(
                out.as_mut_ptr(),
                input.as_ptr(),
                2,
                std::ptr::null(),
            );
            darkroom_colorspaces_cygm_apply_coeffs(out.as_mut_ptr(), input.as_ptr(), 0, matrix.as_ptr());
            darkroom_colorspaces_cygm_apply_coeffs(
                out.as_mut_ptr(),
                input.as_ptr(),
                (i32::MAX as usize) + 1,
                matrix.as_ptr(),
            );
        }
        assert_eq!(out, vec![1.0f32; 8]); // untouched
    }
}
