//! Kernel ported from `src/imageio/imageio.c`
//! (`dt_imageio_export_with_flags`, the 8-bit non-display-byteorder tail,
//! m4-205). The kernel replaces one whole loop body: the in-place
//! red/blue lane swap over the 8-bit RGBA export buffer.
//!
//! What the C loop does, per pixel `k` in `0..npixels` where `npixels` is
//! `processed_width * processed_height`:
//! `tmp = buf[4*k]; buf[4*k] = buf[4*k+2]; buf[4*k+2] = tmp`,
//! i.e. lanes 0 and 2 (R and B) exchange values while lanes 1 and 3 (G and
//! alpha) are never touched. The swap runs after the 8-bit pipeline has
//! produced its output in `pipe.backbuf`; when the host byte order is not
//! the display order the export formats expect the channels flipped,
//! hence this tail.
//!
//! Bit-exactness notes:
//! - The operation is a pure byte permutation, so bit-exactness holds by
//!   construction: no arithmetic, no rounding, no clamping. The kernel
//!   exchanges lanes 0 and 2 of each quad with explicit stride arithmetic
//!   (the same lane selection the C loop spells with a temporary byte);
//!   the reference reaches the same permutation through disjoint split
//!   borrows, so the sweep test cross-checks indexing as well as values.
//! - In place over disjoint quads: each quad is read and written within
//!   its own 4 bytes, so iteration order cannot matter.
//!
//! The Rust kernel is single-threaded sequential; the C loop was
//! `DT_OMP_FOR` over pixels, but each output quad reads only its own
//! input quad, so thread scheduling cannot change the result.

/// Swap the R and B lanes of an 8-bit RGBA buffer in place.
///
/// Port of the former element-wise loop in the `bpp == 8`
/// `!display_byteorder` branch of `dt_imageio_export_with_flags`
/// (src/imageio/imageio.c): `buf` holds `4*npixels` bytes (tightly packed
/// RGBA from the export pipeline), lanes 0 and 2 of every quad are
/// exchanged, lanes 1 and 3 preserved. See the module docs for the
/// no-arithmetic fidelity point.
///
/// Degenerate `npixels == 0` is a no-op; short buffers are handled by
/// clamped iteration (no panic, no out-of-bounds access). For the
/// well-formed buffer the C caller passes the clamp never engages and the
/// behaviour is exactly the C loop's.
pub fn swap_rb_inplace(buf: &mut [u8], npixels: usize) {
    let n = npixels.min(buf.len() / 4);
    for k in 0..n {
        let b = 4 * k;
        buf.swap(b, b + 2);
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `swap_rb_inplace`: walks the
/// buffer with `chunks_exact_mut` quads and exchanges lanes 0 and 2
/// through disjoint `split_at_mut` borrows (the kernel walks with explicit
/// stride arithmetic and `slice::swap`), so the sweep test
/// cross-checks indexing as well as values. Same well-formed-buffers
/// precondition, enforced here by early return rather than clamping.
#[cfg(test)]
fn ref_swap_rb_inplace(buf: &mut [u8], npixels: usize) {
    let Some(need) = npixels.checked_mul(4) else {
        return;
    };
    if buf.len() < need {
        return;
    }
    for quad in buf.chunks_exact_mut(4).take(npixels) {
        let (head, tail) = quad.split_at_mut(2);
        std::mem::swap(&mut head[0], &mut tail[0]);
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// # Safety
/// `buf` must hold at least `4 * npixels` bytes (the C caller passes
/// `pipe.backbuf` for a `processed_width * processed_height` 8-bit RGBA
/// image). The buffer is swapped in place.
#[no_mangle]
pub unsafe extern "C" fn darkroom_imageio_swap_rb(buf: *mut u8, npixels: usize) {
    if buf.is_null() || npixels == 0 {
        return;
    }
    // validate the product BEFORE building the slice below (a misuse
    // caller could otherwise wrap the length; the safe kernel re-checks
    // defensively via clamped iteration)
    let Some(len) = npixels.checked_mul(4) else {
        return;
    };
    let buf = std::slice::from_raw_parts_mut(buf, len);
    swap_rb_inplace(buf, npixels);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // known-answer pin: R and B exchange, G and alpha keep their values,
    // including a pixel whose R already equals B (swap must be a no-op
    // there, not a corruption).
    #[test]
    fn swap_exchanges_r_and_b() {
        let mut buf = vec![
            10u8, 20, 30, 40, // pixel 0: R=10 G=20 B=30 A=40
            5u8, 6, 5, 7, // pixel 1: R == B
        ];
        swap_rb_inplace(&mut buf, 2);
        assert_eq!(buf, vec![30u8, 20, 10, 40, 5u8, 6, 5, 7]);
    }

    // double application is the identity on every byte pattern: the swap
    // is its own inverse, so any deviation in lane selection fails here.
    #[test]
    fn double_swap_is_identity() {
        for n in [1usize, 3, 65] {
            let mut buf = Vec::with_capacity(4 * n);
            for i in 0..4 * n {
                // LCG over the full 0..=255 byte range
                let v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37)
                    % 256;
                buf.push(v as u8);
            }
            let original = buf.clone();
            swap_rb_inplace(&mut buf, n);
            // after one swap lanes 1 and 3 are already untouched
            for (k, quad) in buf.chunks_exact(4).enumerate() {
                assert_eq!(quad[1], original[4 * k + 1], "green lane {k}");
                assert_eq!(quad[3], original[4 * k + 3], "alpha lane {k}");
            }
            swap_rb_inplace(&mut buf, n);
            assert_eq!(buf, original);
        }
    }

    // kernel and reference agree byte-exactly over sweeps at several pixel
    // counts (well-formed buffers: both sides fully engage).
    #[test]
    fn matches_reference_over_sweep() {
        for n in [1usize, 3, 65, 513] {
            let mut src = Vec::with_capacity(4 * n);
            for i in 0..4 * n {
                // LCG over the full 0..=255 byte range
                let v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(0x9E37)
                    % 256;
                src.push(v as u8);
            }
            let mut direct = src.clone();
            let mut reference = src.clone();
            swap_rb_inplace(&mut direct, n);
            ref_swap_rb_inplace(&mut reference, n);
            assert_eq!(direct, reference, "n = {n}");
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        // zero pixels: buffer untouched
        let mut buf = vec![11u8, 22, 33, 44];
        swap_rb_inplace(&mut buf, 0);
        assert_eq!(buf, vec![11u8, 22, 33, 44]);
        // truncated buffer: clamped iteration swaps only the complete
        // quad, never panics nor writes out of bounds; well-formed
        // callers never hit this path.
        let mut short = vec![1u8, 2, 3, 4, 5, 6, 7];
        swap_rb_inplace(&mut short, 2);
        assert_eq!(short, vec![3u8, 2, 1, 4, 5, 6, 7]);
        // empty buffer with nonzero count: no-op
        let mut empty: Vec<u8> = vec![];
        swap_rb_inplace(&mut empty, 1);
        assert!(empty.is_empty());
        // over-long buffer: the tail past 4*npixels is untouched
        let mut long = vec![9u8, 8, 7, 6, 100, 101, 102];
        swap_rb_inplace(&mut long, 1);
        assert_eq!(long, vec![7u8, 8, 9, 6, 100, 101, 102]);
    }

    #[test]
    fn ffi_round_trip() {
        let n = 65usize;
        let mut src = vec![0u8; 4 * n];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i as u64).wrapping_mul(2_654_435_761) % 256) as u8;
        }
        let mut ffi_buf = src.clone();
        let mut direct_buf = src.clone();
        unsafe {
            darkroom_imageio_swap_rb(ffi_buf.as_mut_ptr(), n);
        }
        swap_rb_inplace(&mut direct_buf, n);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_guards() {
        let mut buf = vec![7u8; 16];
        unsafe {
            // null pointer
            darkroom_imageio_swap_rb(std::ptr::null_mut(), 4);
            // zero pixels
            darkroom_imageio_swap_rb(buf.as_mut_ptr(), 0);
            // overflowing pixel counts (4 * npixels wraps — all reject)
            darkroom_imageio_swap_rb(buf.as_mut_ptr(), usize::MAX);
            darkroom_imageio_swap_rb(buf.as_mut_ptr(), usize::MAX / 4 + 1);
        }
        assert_eq!(buf, vec![7u8; 16]); // untouched
    }
}
