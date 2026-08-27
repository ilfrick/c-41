//! Object drawn-mask rendering — port of the OMP loops in
//! `src/develop/masks/object.c`.
//!
//! Two kernels are ported:
//! - [`object_mask_iou`] — intersection-over-union reduction of two float
//!   masks, replacing the `DT_OMP_FOR(reduction(+:inter,uni))` at object.c:528
//!   inside `_mask_iou`.
//! - [`object_zero_peaks`] — zero out pixels within a circular exclusion zone
//!   around a peak point in a distance-transform buffer, replacing the
//!   `DT_OMP_FOR(collapse(2))` nested loop at object.c:573 inside
//!   `_find_peak_point`.
//!
//! The surrounding logic (model loading, distance transform, argmax) stays in
//! C; only the pure pixel arithmetic moves here.

/// Intersection-over-union of two float masks (port of `_mask_iou`,
/// object.c:522).
///
/// For each pixel `i` in `[0, n)`: `A = a[i] > threshold`, `B = b[i] > threshold`.
/// `inter` counts pixels where both are true; `uni` counts pixels where either
/// is true. Returns `inter/uni` as f32, or `0.0` if `uni == 0`.
///
/// The integer promotion (`a[i] > threshold` → 0/1) and the `(float)inter /
/// (float)uni` division order are matched exactly: `usize` counts are cast to
/// `f32` individually before division, matching C's size_t→float cast at the
/// same point.
pub fn object_mask_iou(a: &[f32], b: &[f32], threshold: f32) -> f32 {
    let n = a.len().min(b.len());
    let mut inter: usize = 0;
    let mut uni: usize = 0;
    for i in 0..n {
        // C: const int A = a[i] > threshold;  (relational → int 0/1)
        let a_above = a[i] > threshold;
        let b_above = b[i] > threshold;
        if a_above && b_above {
            inter += 1;
        }
        if a_above || b_above {
            uni += 1;
        }
    }
    if uni > 0 {
        // (float)inter / (float)uni  — matches C cast-then-divide
        inter as f32 / uni as f32
    } else {
        0.0f32
    }
}

/// Zero out pixels within a circular exclusion zone around `(px, py)` in the
/// distance-transform buffer (port of the `collapse(2)` loop at object.c:573
/// inside `_find_peak_point`).
///
/// The bounding box `[x0, x1] × [y0, y1]` is already clamped to
/// `[0, w-1] × [0, h-1]` by the C caller; Rust adds a slice-length guard
/// for safety. The condition `dx*dx + dy*dy < min_sep_sq` uses pure f32
/// arithmetic, matching C's `(float)x - px` and float multiplication.
pub fn object_zero_peaks(
    dist: &mut [f32],
    w: i32,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    px: f32,
    py: f32,
    min_sep_sq: f32,
) {
    let w_usize = w as usize;
    for y in y0..=y1 {
        let y_usize = y as usize;
        for x in x0..=x1 {
            let dx = x as f32 - px;
            let dy = y as f32 - py;
            if dx * dx + dy * dy < min_sep_sq {
                let idx = y_usize * w_usize + x as usize;
                if idx < dist.len() {
                    dist[idx] = 0.0f32;
                }
            }
        }
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `a` and `b` must each point to at least `n` readable floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_object_mask_iou(
    a: *const f32,
    b: *const f32,
    n: usize,
    threshold: f32,
) -> f32 {
    if a.is_null() || b.is_null() || n == 0 {
        return 0.0f32;
    }
    let a = std::slice::from_raw_parts(a, n);
    let b = std::slice::from_raw_parts(b, n);
    object_mask_iou(a, b, threshold)
}

/// # Safety
/// `dist` must hold at least `w*bh` floats. The bounding box
/// `[x0,x1]×[y0,y1]` is clamped by the C caller before this is called.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_object_zero_peaks(
    dist: *mut f32,
    w: i32,
    bh: i32,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    px: f32,
    py: f32,
    min_sep_sq: f32,
) {
    if dist.is_null() || w <= 0 || bh <= 0 {
        return;
    }
    let Some(total) = w.checked_mul(bh) else { return };
    let dist = std::slice::from_raw_parts_mut(dist, total as usize);
    object_zero_peaks(dist, w, x0, x1, y0, y1, px, py, min_sep_sq);
}

// ── Reference implementations for bit-exactness tests ───────────────────────

/// Reference for `object_mask_iou` — mirrors `_mask_iou` in object.c:522.
fn ref_object_mask_iou(a: &[f32], b: &[f32], threshold: f32) -> f32 {
    let n = a.len().min(b.len());
    let mut inter: usize = 0;
    let mut uni: usize = 0;
    for i in 0..n {
        let a_above = a[i] > threshold;
        let b_above = b[i] > threshold;
        if a_above && b_above {
            inter += 1;
        }
        if a_above || b_above {
            uni += 1;
        }
    }
    if uni > 0 {
        inter as f32 / uni as f32
    } else {
        0.0f32
    }
}

/// Reference for `object_zero_peaks` — mirrors the `collapse(2)` loop in
/// object.c:573.
fn ref_object_zero_peaks(
    dist: &mut [f32],
    w: i32,
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    px: f32,
    py: f32,
    min_sep_sq: f32,
) {
    let w_usize = w as usize;
    for y in y0..=y1 {
        let y_usize = y as usize;
        for x in x0..=x1 {
            let dx = x as f32 - px;
            let dy = y as f32 - py;
            if dx * dx + dy * dy < min_sep_sq {
                let idx = y_usize * w_usize + x as usize;
                if idx < dist.len() {
                    dist[idx] = 0.0f32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_util::lcg_fill;

    // ── object_mask_iou ─────────────────────────────────────────────────

    #[test]
    fn mask_iou_identical_masks() {
        // Identical masks → IoU = 1.0
        let a = vec![0.9f32, 0.8, 0.7, 0.6];
        let b = a.clone();
        let iou = object_mask_iou(&a, &b, 0.5);
        assert_eq!(iou, 1.0);
    }

    #[test]
    fn mask_iou_disjoint_masks() {
        // a above threshold, b below → no intersection, union = 4 → IoU = 0.0
        let a = vec![0.9f32, 0.8, 0.7, 0.6];
        let b = vec![0.1f32, 0.2, 0.3, 0.4];
        let iou = object_mask_iou(&a, &b, 0.5);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn mask_iou_partial_overlap() {
        // a = [1, 0, 1, 0], b = [1, 1, 0, 0], threshold = 0.5
        // A = [true, false, true, false], B = [true, true, false, false]
        // inter = 1 (only index 0), uni = 3 (indices 0,1,2)
        // IoU = 1/3
        let a = vec![1.0f32, 0.0, 1.0, 0.0];
        let b = vec![1.0f32, 1.0, 0.0, 0.0];
        let iou = object_mask_iou(&a, &b, 0.5);
        assert!((iou - 1.0f32 / 3.0f32).abs() < 1e-6);
    }

    #[test]
    fn mask_iou_empty_masks() {
        // All values below threshold → uni = 0 → return 0.0
        let a = vec![0.0f32, 0.1, 0.2];
        let b = vec![0.0f32, 0.1, 0.2];
        let iou = object_mask_iou(&a, &b, 0.5);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn mask_iou_threshold_boundary() {
        // Value exactly at threshold: > is strict, so 0.5 is NOT above 0.5
        let a = vec![0.5f32, 0.6, 0.5, 0.4];
        let b = vec![0.6f32, 0.5, 0.4, 0.5];
        // threshold = 0.5
        // A = [false, true, false, false], B = [true, false, false, false]
        // inter = 0, uni = 2 → IoU = 0.0
        let iou = object_mask_iou(&a, &b, 0.5);
        assert_eq!(iou, 0.0);

        // threshold = 0.4 → A = [true, true, true, false], B = [true, true, false, true]
        // inter = 2 (idx 0,1), uni = 4 (idx 0,1,2,3) → IoU = 0.5
        let iou2 = object_mask_iou(&a, &b, 0.4);
        assert!((iou2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mask_iou_nan_treated_as_below() {
        // NaN > threshold is always false in both C and Rust
        let a = vec![f32::NAN, 1.0f32, 0.0];
        let b = vec![1.0f32, f32::NAN, 0.0];
        let iou = object_mask_iou(&a, &b, 0.5);
        // A = [false, true, false], B = [true, false, false]
        // inter = 0, uni = 2 → IoU = 0.0
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn mask_iou_matches_reference_over_lcg() {
        let mut a = vec![0f32; 1024];
        let mut b = vec![0f32; 1024];
        lcg_fill(&mut a, 0xAAAA0001, 1.0);
        lcg_fill(&mut b, 0xBBBB0002, 1.0);
        for k in 0..10 {
            let thr = 0.1 + 0.1 * k as f32;
            assert_eq!(
                object_mask_iou(&a, &b, thr),
                ref_object_mask_iou(&a, &b, thr),
                "IoU mismatch at threshold {thr}"
            );
        }
    }

    #[test]
    fn ffi_mask_iou_round_trip() {
        unsafe {
            let a = vec![0.9f32, 0.1, 0.8, 0.2, 0.7, 0.3];
            let b = vec![0.8f32, 0.3, 0.7, 0.9, 0.1, 0.2];
            let direct = object_mask_iou(&a, &b, 0.5);

            let ffi_val = darkroom_masks_object_mask_iou(
                a.as_ptr(), b.as_ptr(), a.len(), 0.5f32);
            assert_eq!(direct, ffi_val, "FFI IoU mismatch");
        }
    }

    #[test]
    fn ffi_mask_iou_null_guard() {
        unsafe {
            assert_eq!(darkroom_masks_object_mask_iou(
                std::ptr::null(), std::ptr::null(), 100, 0.5f32), 0.0f32);
            // n == 0 → return 0.0 immediately
            let a = vec![0.5f32; 0];
            let b = vec![0.5f32; 0];
            assert_eq!(darkroom_masks_object_mask_iou(
                a.as_ptr(), b.as_ptr(), 0, 0.5f32), 0.0f32);
        }
    }

    // ── object_zero_peaks ─────────────────────────────────────────────────

    #[test]
    fn zero_peaks_basic_exclusion() {
        // 10x10 buffer, zero around (5,5) with min_sep=3.0 → min_sep_sq=9.0
        let mut buf = vec![1.0f32; 10 * 10];
        let w = 10;
        let x0 = 2; let x1 = 8; let y0 = 2; let y1 = 8;
        let px = 5.0f32; let py = 5.0f32; let min_sep_sq = 9.0f32;

        object_zero_peaks(&mut buf, w, x0, x1, y0, y1, px, py, min_sep_sq);

        // Pixel at (5,5): dx=0, dy=0, 0 < 9 → zeroed
        assert_eq!(buf[5 * 10 + 5], 0.0);
        // Pixel at (3,3): dx=2, dy=2, 8 < 9 → zeroed
        assert_eq!(buf[3 * 10 + 3], 0.0);
        // Pixel at (2,5): dx=3, dy=0, 9 >= 9 → NOT zeroed (strict <)
        assert_eq!(buf[5 * 10 + 2], 1.0);
        // Pixel at (5,2): dx=0, dy=3, 9 >= 9 → NOT zeroed
        assert_eq!(buf[2 * 10 + 5], 1.0);
        // Pixel at (8,8): dx=3, dy=3, 18 >= 9 → NOT zeroed
        assert_eq!(buf[8 * 10 + 8], 1.0);

        let mut ref_buf = vec![1.0f32; 10 * 10];
        ref_object_zero_peaks(&mut ref_buf, w, x0, x1, y0, y1, px, py, min_sep_sq);
        assert_eq!(buf, ref_buf, "zero_peaks mismatch vs reference");
    }

    #[test]
    fn zero_peaks_corner_clamp() {
        // Exclude point near top-left corner: bounding box already clamped
        // by C caller to [0, w-1] × [0, h-1]
        let mut buf = vec![1.0f32; 10 * 10];
        let w = 10;
        // px=1.0, py=1.0, min_sep=3.0 → clamped bbox [0,4]×[0,4]
        let x0 = 0; let x1 = 4; let y0 = 0; let y1 = 4;
        let px = 1.0f32; let py = 1.0f32; let min_sep_sq = 9.0f32;

        object_zero_peaks(&mut buf, w, x0, x1, y0, y1, px, py, min_sep_sq);

        // Pixel at (1,1): dx=0, dy=0 → zeroed
        assert_eq!(buf[1 * 10 + 1], 0.0);
        // Pixel at (0,1): dx=-1, dy=0, 1 < 9 → zeroed
        assert_eq!(buf[1 * 10 + 0], 0.0);
        // Pixel at (3,3): dx=2, dy=2, 8 < 9 → zeroed
        assert_eq!(buf[3 * 10 + 3], 0.0);
        // Pixel at (4,4): dx=3, dy=3, 18 >= 9 → NOT zeroed
        assert_eq!(buf[4 * 10 + 4], 1.0);

        let mut ref_buf = vec![1.0f32; 10 * 10];
        ref_object_zero_peaks(&mut ref_buf, w, x0, x1, y0, y1, px, py, min_sep_sq);
        assert_eq!(buf, ref_buf, "corner clamp mismatch vs reference");
    }

    #[test]
    fn zero_peaks_empty_range() {
        // x0 > x1 → empty loop, nothing happens
        let mut buf = vec![1.0f32; 10 * 10];
        let w = 10;
        object_zero_peaks(&mut buf, w, 5, 3, 0, 9, 5.0, 5.0, 25.0);
        assert!(buf.iter().all(|&v| v == 1.0), "empty range should not modify buffer");

        // y0 > y1 → empty loop
        let mut buf2 = vec![1.0f32; 10 * 10];
        object_zero_peaks(&mut buf2, w, 0, 9, 5, 3, 5.0, 5.0, 25.0);
        assert!(buf2.iter().all(|&v| v == 1.0), "empty y range should not modify buffer");
    }

    #[test]
    fn zero_peaks_min_sep_zero() {
        // min_sep_sq = 0 → no pixels satisfy dx*dx+dy*dy < 0
        let mut buf = vec![1.0f32; 5 * 5];
        object_zero_peaks(&mut buf, 5, 0, 4, 0, 4, 2.0, 2.0, 0.0);
        assert!(buf.iter().all(|&v| v == 1.0), "min_sep_sq=0 should not zero anything");
    }

    #[test]
    fn zero_peaks_full_circle() {
        // Large radius zeros most of the buffer in the bbox
        let mut buf = vec![1.0f32; 10 * 10];
        let w = 10;
        let x0 = 0; let x1 = 9; let y0 = 0; let y1 = 9;
        let px = 5.0f32; let py = 5.0f32; let min_sep_sq = 100.0f32;

        object_zero_peaks(&mut buf, w, x0, x1, y0, y1, px, py, min_sep_sq);

        // All pixels within distance < 10 from (5,5) should be zeroed
        assert_eq!(buf[5 * 10 + 5], 0.0);  // center
        assert_eq!(buf[5 * 10 + 0], 0.0);  // corner of bbox, dx=5, dy=0, 25 < 100
        assert_eq!(buf[9 * 10 + 9], 0.0);  // far corner, dx=4, dy=4, 32 < 100

        let mut ref_buf = vec![1.0f32; 10 * 10];
        ref_object_zero_peaks(&mut ref_buf, w, x0, x1, y0, y1, px, py, min_sep_sq);
        assert_eq!(buf, ref_buf, "full circle mismatch vs reference");
    }

    #[test]
    fn zero_peaks_matches_reference_over_lcg() {
        let mut buf = vec![0f32; 20 * 20];
        lcg_fill(&mut buf, 0xCAFE, 1.0);
        let mut ref_buf = buf.clone();

        let w = 20;
        let params: [(f32, f32, f32); 5] = [
            (5.0, 5.0, 9.0),
            (15.0, 10.0, 16.0),
            (1.0, 1.0, 4.0),
            (18.0, 18.0, 25.0),
            (10.0, 10.0, 100.0),
        ];
        for (px, py, min_sep_sq) in params {
            let x0 = (px - min_sep_sq.sqrt()).max(0.0).floor() as i32;
            let x1 = ((px + min_sep_sq.sqrt()).min((w - 1) as f32)).floor() as i32;
            let y0 = (py - min_sep_sq.sqrt()).max(0.0).floor() as i32;
            let y1 = ((py + min_sep_sq.sqrt()).min((w - 1) as f32)).floor() as i32;

            object_zero_peaks(&mut buf, w, x0, x1, y0, y1, px, py, min_sep_sq);
            ref_object_zero_peaks(&mut ref_buf, w, x0, x1, y0, y1, px, py, min_sep_sq);
        }
        assert_eq!(buf, ref_buf, "multi-exclude LCG mismatch");
    }

    #[test]
    fn ffi_zero_peaks_round_trip() {
        unsafe {
            let mut ffi_buf = vec![1.0f32; 10 * 10];
            let w = 10;
            let bh = 10;
            let x0 = 2; let x1 = 8; let y0 = 2; let y1 = 8;
            let px = 5.0f32; let py = 5.0f32; let min_sep_sq = 9.0f32;

            darkroom_masks_object_zero_peaks(
                ffi_buf.as_mut_ptr(), w, bh,
                x0, x1, y0, y1, px, py, min_sep_sq);

            // Direct call
            let mut direct_buf = vec![1.0f32; 10 * 10];
            object_zero_peaks(&mut direct_buf, w, x0, x1, y0, y1, px, py, min_sep_sq);

            assert_eq!(ffi_buf, direct_buf, "FFI zero_peaks mismatch");
        }
    }

    #[test]
    fn ffi_zero_peaks_null_guard() {
        unsafe {
            darkroom_masks_object_zero_peaks(
                std::ptr::null_mut(), 10, 10,
                0, 9, 0, 9, 5.0, 5.0, 25.0);
            // Should not crash, no-op
        }
    }

    #[test]
    fn ffi_zero_peaks_invalid_dims() {
        unsafe {
            let mut buf = vec![1.0f32; 10 * 10];
            // w <= 0 → no-op
            darkroom_masks_object_zero_peaks(
                buf.as_mut_ptr(), 0, 10,
                0, 9, 0, 9, 5.0, 5.0, 25.0);
            assert!(buf.iter().all(|&v| v == 1.0));

            // bh <= 0 → no-op
            darkroom_masks_object_zero_peaks(
                buf.as_mut_ptr(), 10, 0,
                0, 9, 0, 9, 5.0, 5.0, 25.0);
            assert!(buf.iter().all(|&v| v == 1.0));
        }
    }
}
