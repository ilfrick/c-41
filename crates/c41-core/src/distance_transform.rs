//! Distance-transform helpers ported from `src/common/distance_transform.c`.
//!
//! One loop is ported here (m4-172): the `DT_DISTANCE_TRANSFORM_MASK`
//! threshold loop (formerly at distance_transform.c:105) that binarises a
//! mask into `0.0f` / `DT_DISTANCE_TRANSFORM_MAX` seeds before the
//! two-pass Felzenszwalb–Huttenlocher transform (which stays in C — it is a
//! parallel region with per-thread work arrays, not a flat loop).
//!
//! `DT_DISTANCE_TRANSFORM_MAX` is the C literal `(1e20)` — a double
//! constant converted to float on assignment, i.e. `fl(1e20)`, which the
//! Rust `1e20_f32` literal reproduces exactly.

/// `DT_DISTANCE_TRANSFORM_MAX` from `common/distance_transform.h` (1e20).
const DISTANCE_TRANSFORM_MAX: f32 = 1e20_f32;

/// Binarise `src` into distance-transform seeds.
///
/// Port of the `DT_OMP_FOR` loop in `dt_image_distance_transform`'s
/// `DT_DISTANCE_TRANSFORM_MASK` branch (formerly distance_transform.c:105):
/// `out[k] = (src[k] < clip) ? 0.0f : DT_DISTANCE_TRANSFORM_MAX`.
/// Note the strict `<` — a pixel exactly at `clip` seeds MAX ("on").
pub fn mask_threshold(src: &[f32], out: &mut [f32], n_elements: usize, clip: f32) {
    let m = n_elements.min(src.len()).min(out.len());
    for k in 0..m {
        out[k] = if src[k] < clip {
            0.0f32
        } else {
            DISTANCE_TRANSFORM_MAX
        };
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `src` and `out` must each hold at least `n_elements` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_distance_transform_mask(
    src: *const f32,
    out: *mut f32,
    n_elements: usize,
    clip: f32,
) {
    if src.is_null() || out.is_null() || n_elements == 0 || n_elements > i32::MAX as usize {
        return;
    }
    let src_slice = std::slice::from_raw_parts(src, n_elements);
    let out_slice = std::slice::from_raw_parts_mut(out, n_elements);
    mask_threshold(src_slice, out_slice, n_elements, clip);
}

// ── Independent reference implementation ────────────────────────────────────

#[allow(dead_code)]
fn ref_mask_threshold(src: &[f32], out: &mut [f32], n_elements: usize, clip: f32) {
    // Restructured: match instead of ternary, iterator-based bound
    let m = n_elements.min(src.len()).min(out.len());
    for (k, v) in src[..m].iter().enumerate() {
        out[k] = match *v < clip {
            true => 0.0f32,
            false => DISTANCE_TRANSFORM_MAX,
        };
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    #[test]
    fn mask_threshold_basic() {
        let src = vec![0.1f32, 0.5, 0.9, 1.0];
        let mut out = vec![f32::NAN; 4];
        mask_threshold(&src, &mut out, 4, 0.5);
        assert_eq!(out[0], 0.0); // 0.1 < 0.5 → off
        assert_eq!(out[1], DISTANCE_TRANSFORM_MAX); // 0.5 not < 0.5 → on
        assert_eq!(out[2], DISTANCE_TRANSFORM_MAX);
        assert_eq!(out[3], DISTANCE_TRANSFORM_MAX);
    }

    #[test]
    fn mask_threshold_boundary_is_strict_less() {
        // A pixel exactly at clip seeds MAX ("on"); anything strictly below
        // clips to 0 ("off")
        let mut out = vec![f32::NAN; 1];
        let src = vec![0.5f32];
        mask_threshold(&src, &mut out, 1, 0.5);
        assert_eq!(out[0], DISTANCE_TRANSFORM_MAX); // exact clip: not < clip → on
        let src = vec![0.5f32 - f32::EPSILON];
        mask_threshold(&src, &mut out, 1, 0.5);
        assert_eq!(out[0], 0.0); // one step down: strictly < clip → off
        let src = vec![0.4999999f32];
        mask_threshold(&src, &mut out, 1, 0.5);
        assert_eq!(out[0], 0.0); // clearly below → off
    }

    #[test]
    fn mask_threshold_matches_reference_over_lcg() {
        let mut src = vec![0.0f32; 256];
        lcg_fill(&mut src, 0xD157, 1.0);
        let mut direct = vec![0.0f32; 256];
        let mut reference = vec![0.0f32; 256];
        mask_threshold(&src, &mut direct, 256, 0.5);
        ref_mask_threshold(&src, &mut reference, 256, 0.5);
        assert_eq!(direct, reference);
    }

    #[test]
    fn ffi_mask_threshold_round_trip() {
        let mut src = vec![0.0f32; 128];
        lcg_fill(&mut src, 0xD158, 1.0);
        let mut ffi_buf = vec![f32::NAN; 128];
        let mut direct_buf = vec![f32::NAN; 128];
        unsafe {
            darkroom_distance_transform_mask(src.as_ptr(), ffi_buf.as_mut_ptr(), 128, 0.5);
        }
        mask_threshold(&src, &mut direct_buf, 128, 0.5);
        assert_eq!(ffi_buf, direct_buf);
    }

    #[test]
    fn ffi_mask_threshold_guards() {
        unsafe {
            darkroom_distance_transform_mask(std::ptr::null(), std::ptr::null_mut(), 10, 0.5);
        }
        let src = vec![1.0f32; 4];
        let mut out = vec![1.0f32; 4];
        unsafe {
            darkroom_distance_transform_mask(src.as_ptr(), out.as_mut_ptr(), 0, 0.5);
            darkroom_distance_transform_mask(
                src.as_ptr(),
                out.as_mut_ptr(),
                (i32::MAX as usize) + 1,
                0.5,
            );
        }
        assert_eq!(out, vec![1.0f32; 4]); // untouched
    }
}
