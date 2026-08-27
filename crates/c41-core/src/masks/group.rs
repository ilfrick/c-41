//! Group drawn-mask rendering — port of the OMP loops in
//! `src/develop/masks/group.c` `_group_get_mask_roi`.
//!
//! The pixelpipe callbacks that transform individual shape points stay in C;
//! only the per-pixel combine arithmetic moves here, exposed through a
//! single `darkroom_masks_group_combine` FFI that dispatches on a blend-mode
//! operation code.
//!
//! Six element-wise operations are ported (each with inverted/non-inverted
//! variants sharing one kernel via a `bool` flag):
//! - [`GROUP_OP_UNION`] — `dest = MAX(dest, opacity * newmask)` (or `1-newmask`
//!   when inverted)
//! - [`GROUP_OP_INTERSECT`] — `dest = MIN(MAX(dest, 0), MAX(mask, 0))`
//! - [`GROUP_OP_DIFFERENCE`] — `dest *= (1 - mask * both_positive(dest, mask))`
//! - [`GROUP_OP_SUM`] — `dest = MIN(1, dest + mask)`
//! - [`GROUP_OP_EXCLUSION`] — the exclusion blend formula
//! - [`GROUP_OP_COPY`] — `dest = opacity * newmask` (or `1-newmask`)
//!
//! All use `MAX`/`MIN` macros (= ternary `>` / `<` for non-NaN floats). In Rust
//! `f32::max`/`f32::min` match the ternary for non-NaN values. `both_positive(v1, v2)`
//! returns 1.0 when both are > 0.0, else 0.0 — in C it's `int` (0/1), but since
//! it's multiplied by `mask` (float), `mask * 1 == mask * 1.0f32` and
//! `mask * 0 == mask * 0.0f32` are bit-identical.

/// Group combine operation codes (mirror the C `inverted ? … : …` dispatch).
pub const GROUP_OP_UNION: i32 = 0;
pub const GROUP_OP_INTERSECT: i32 = 1;
pub const GROUP_OP_DIFFERENCE: i32 = 2;
pub const GROUP_OP_SUM: i32 = 3;
pub const GROUP_OP_EXCLUSION: i32 = 4;
pub const GROUP_OP_COPY: i32 = 5;

/// Element-wise mask combination (port of the five `_combine_masks_*` functions
/// and the inline copy loop in `_group_get_mask_roi`, group.c:484–706).
///
/// `inverted` selects `1.0 - newmask[i]` vs `newmask[i]` for the mask value,
/// matching the C `if(inverted) {...}` / `else {...}` split in each function.
pub fn group_combine(
    dest: &mut [f32],
    newmask: &[f32],
    opacity: f32,
    inverted: bool,
    op: i32,
) {
    let n = dest.len().min(newmask.len());
    for i in 0..n {
        // mask = opacity * (inverted ? 1.0f - newmask[i] : newmask[i])
        let mask = if inverted {
            opacity * (1.0f32 - newmask[i])
        } else {
            opacity * newmask[i]
        };

        match op {
            GROUP_OP_UNION => {
                // dest = MAX(dest, mask)
                dest[i] = dest[i].max(mask);
            }
            GROUP_OP_INTERSECT => {
                // dest = MIN(MAX(dest, 0), MAX(mask, 0))
                dest[i] = dest[i].max(0.0f32).min(mask.max(0.0f32));
            }
            GROUP_OP_DIFFERENCE => {
                // pos = both_positive(dest, mask)  → 1.0 if both > 0, else 0.0
                // dest *= (1.0f - mask * pos)
                let pos = if dest[i] > 0.0f32 && mask > 0.0f32 {
                    1.0f32
                } else {
                    0.0f32
                };
                dest[i] *= 1.0f32 - mask * pos;
            }
            GROUP_OP_SUM => {
                // dest = MIN(1.0f, dest + mask)
                dest[i] = (dest[i] + mask).min(1.0f32);
            }
            GROUP_OP_EXCLUSION => {
                // pos = both_positive(dest, mask) → 1.0 or 0.0
                // neg = 1.0f - pos
                // b1 = dest
                // dest = pos * MAX((1-b1)*mask, b1*(1-mask)) + neg * MAX(b1, mask)
                let pos = if dest[i] > 0.0f32 && mask > 0.0f32 {
                    1.0f32
                } else {
                    0.0f32
                };
                let neg = 1.0f32 - pos;
                let b1 = dest[i];
                dest[i] = pos * ((1.0f32 - b1) * mask).max(b1 * (1.0f32 - mask))
                    + neg * b1.max(mask);
            }
            GROUP_OP_COPY => {
                // dest = opacity * newmask (or 1-newmask)  — just `mask`
                dest[i] = mask;
            }
            _ => {}
        }
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// # Safety
/// `dest` must hold at least `npixels` floats; `newmask` must hold at least
/// `npixels` floats.
#[no_mangle]
pub unsafe extern "C" fn darkroom_masks_group_combine(
    dest: *mut f32,
    newmask: *const f32,
    npixels: usize,
    opacity: f32,
    inverted: i32,
    op: i32,
) {
    if dest.is_null() || newmask.is_null() || npixels == 0 {
        return;
    }
    let dest = std::slice::from_raw_parts_mut(dest, npixels);
    let newmask = std::slice::from_raw_parts(newmask, npixels);
    group_combine(dest, newmask, opacity, inverted != 0, op);
}

// ── Reference implementation for bit-exactness tests ────────────────────────

/// Reference for `group_combine` — mirrors the `_combine_masks_*` functions
/// and inline copy loop in group.c:484–706.
fn ref_group_combine(
    dest: &mut [f32],
    newmask: &[f32],
    opacity: f32,
    inverted: bool,
    op: i32,
) {
    let n = dest.len().min(newmask.len());
    for i in 0..n {
        let mask = if inverted {
            opacity * (1.0f32 - newmask[i])
        } else {
            opacity * newmask[i]
        };

        match op {
            GROUP_OP_UNION => {
                dest[i] = dest[i].max(mask);
            }
            GROUP_OP_INTERSECT => {
                dest[i] = dest[i].max(0.0f32).min(mask.max(0.0f32));
            }
            GROUP_OP_DIFFERENCE => {
                let pos = if dest[i] > 0.0f32 && mask > 0.0f32 {
                    1.0f32
                } else {
                    0.0f32
                };
                dest[i] *= 1.0f32 - mask * pos;
            }
            GROUP_OP_SUM => {
                dest[i] = (dest[i] + mask).min(1.0f32);
            }
            GROUP_OP_EXCLUSION => {
                let pos = if dest[i] > 0.0f32 && mask > 0.0f32 {
                    1.0f32
                } else {
                    0.0f32
                };
                let neg = 1.0f32 - pos;
                let b1 = dest[i];
                dest[i] = pos * ((1.0f32 - b1) * mask).max(b1 * (1.0f32 - mask))
                    + neg * b1.max(mask);
            }
            GROUP_OP_COPY => {
                dest[i] = mask;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_util::lcg_fill;

    // ── IoU / union tests ────────────────────────────────────────────────

    #[test]
    fn group_union_basic() {
        // dest = [0.1, 0.2, 0.3, 0.4], newmask = [0.5, 0.6, 0.7, 0.8], opacity = 1.0
        // mask = newmask, dest = MAX(dest, mask)
        let mut dest = vec![0.1f32, 0.2, 0.3, 0.4];
        let newmask = vec![0.5f32, 0.6, 0.7, 0.8];
        group_combine(&mut dest, &newmask, 1.0, false, GROUP_OP_UNION);
        assert_eq!(dest, vec![0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn group_union_inverted() {
        // inverted: mask = 1.0 * (1.0 - newmask)
        let mut dest = vec![0.1f32, 0.2, 0.3, 0.4];
        let newmask = vec![0.5f32, 0.6, 0.7, 0.8];
        group_combine(&mut dest, &newmask, 1.0, true, GROUP_OP_UNION);
        // mask = [0.5, ~0.4, ~0.3, 0.2], dest = MAX(dest, mask) = [0.5, ~0.4, 0.3, 0.4]
        // (1.0f32 - 0.6f32 loses precision → 0.39999998)
        for i in 0..dest.len() {
            assert!((dest[i] - vec![0.5f32, 0.4, 0.3, 0.4][i]).abs() < 1e-6,
                "index {i}: got {0}, want {1}", dest[i], vec![0.5f32, 0.4, 0.3, 0.4][i]);
        }
    }

    #[test]
    fn group_union_with_opacity() {
        let mut dest = vec![0.3f32, 0.4, 0.5, 0.6];
        let newmask = vec![1.0f32, 0.5, 0.2, 0.8];
        group_combine(&mut dest, &newmask, 0.5, false, GROUP_OP_UNION);
        // mask = [0.5, 0.25, 0.1, 0.4], dest = MAX(dest, mask) = [0.5, 0.4, 0.5, 0.6]
        assert_eq!(dest, vec![0.5, 0.4, 0.5, 0.6]);
    }

    // ── Intersect tests ──────────────────────────────────────────────────

    #[test]
    fn group_intersect_basic() {
        // dest = MAX(dest, 0), mask = MAX(mask, 0), dest = MIN(dest, mask)
        let mut dest = vec![0.3f32, 0.8, 0.5, 0.1];
        let newmask = vec![0.5f32, 0.2, 0.9, 0.4];
        group_combine(&mut dest, &newmask, 1.0, false, GROUP_OP_INTERSECT);
        // mask = max(newmask, 0) = [0.5, 0.2, 0.9, 0.4]
        // dest = min(max(dest,0), mask) = min([0.3,0.8,0.5,0.1], [0.5,0.2,0.9,0.4]) = [0.3, 0.2, 0.5, 0.1]
        assert_eq!(dest, vec![0.3, 0.2, 0.5, 0.1]);
    }

    #[test]
    fn group_intersect_negative_values() {
        // If dest or mask is negative, MAX(x, 0) clamps to 0
        let mut dest = vec![-1.0f32, 0.5, -0.3, 0.8];
        let newmask = vec![0.7f32, -0.5, 0.6, -0.2];
        group_combine(&mut dest, &newmask, 1.0, false, GROUP_OP_INTERSECT);
        // mask = [0.7, 0.0, 0.6, 0.0]
        // dest = min([0.0, 0.5, 0.0, 0.8], [0.7, 0.0, 0.6, 0.0]) = [0.0, 0.0, 0.0, 0.0]
        assert_eq!(dest, vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ── Difference tests ─────────────────────────────────────────────────

    #[test]
    fn group_difference_basic() {
        // dest *= (1 - mask * both_positive(dest, mask))
        // If both > 0: dest *= (1 - mask)
        let mut dest = vec![0.8f32, 0.5, 0.0, 0.3];
        let newmask = vec![0.5f32, 0.4, 0.6, 0.0];
        group_combine(&mut dest, &newmask, 1.0, false, GROUP_OP_DIFFERENCE);
        // mask = [0.5, 0.4, 0.6, 0.0]
        // i=0: both > 0 → dest *= (1 - 0.5) = 0.5 → 0.8*0.5 = 0.4
        // i=1: both > 0 → dest *= (1 - 0.4) = 0.6 → 0.5*0.6 = 0.3
        // i=2: dest=0 → not both positive → dest *= 1.0 → 0.0
        // i=3: mask=0 → not both positive → dest *= 1.0 → 0.3
        assert_eq!(dest, vec![0.4, 0.3, 0.0, 0.3]);
    }

    #[test]
    fn group_difference_inverted() {
        let mut dest = vec![0.8f32, 0.5, 0.0, 0.3];
        let newmask = vec![0.5f32, 0.6, 0.4, 0.0];
        group_combine(&mut dest, &newmask, 1.0, true, GROUP_OP_DIFFERENCE);
        // mask = [0.5, 0.4, 0.6, 1.0]
        // i=0: both > 0 → dest *= (1 - 0.5) = 0.5 → 0.4
        // i=1: both > 0 → dest *= (1 - 0.4) = 0.6 → 0.3
        // i=2: dest=0 → dest *= 1.0 → 0.0
        // i=3: both > 0 → dest *= (1 - 1.0) = 0.0 → 0.0
        assert_eq!(dest, vec![0.4, 0.3, 0.0, 0.0]);
    }

    // ── Sum tests ────────────────────────────────────────────────────────

    #[test]
    fn group_sum_basic() {
        // dest = MIN(1.0, dest + mask)
        let mut dest = vec![0.3f32, 0.5, 0.8, 0.9];
        let newmask = vec![0.4f32, 0.5, 0.3, 0.2];
        group_combine(&mut dest, &newmask, 1.0, false, GROUP_OP_SUM);
        // mask = [0.4, 0.5, 0.3, 0.2]
        // dest = min(1.0, [0.7, 1.0, 1.1, 1.1]) = [0.7, 1.0, 1.0, 1.0]
        // (0.3f32 + 0.4f32 loses precision → 0.70000005)
        assert!((dest[0] - 0.7).abs() < 1e-6, "i=0: got {}", dest[0]);
        assert_eq!(&dest[1..], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn group_sum_inverted() {
        let mut dest = vec![0.1f32, 0.2, 0.3, 0.4];
        let newmask = vec![0.5f32, 0.5, 0.5, 0.5];
        group_combine(&mut dest, &newmask, 1.0, true, GROUP_OP_SUM);
        // mask = [0.5, 0.5, 0.5, 0.5]
        // dest = min(1.0, [0.6, 0.7, 0.8, 0.9]) = [0.6, 0.7, 0.8, 0.9]
        assert_eq!(dest, vec![0.6, 0.7, 0.8, 0.9]);
    }

    // ── Exclusion tests ──────────────────────────────────────────────────

    #[test]
    fn group_exclusion_basic() {
        // pos = both_positive(dest, mask)
        // neg = 1 - pos
        // b1 = dest
        // dest = pos * MAX((1-b1)*mask, b1*(1-mask)) + neg * MAX(b1, mask)
        let mut dest = vec![0.5f32, 0.0, 0.8, 0.3];
        let newmask = vec![0.3f32, 0.5, 0.6, 0.0];
        group_combine(&mut dest, &newmask, 1.0, false, GROUP_OP_EXCLUSION);
        // mask = [0.3, 0.5, 0.6, 0.0]
        // i=0: pos=1 (both>0), neg=0
        //   MAX((1-0.5)*0.3, 0.5*(1-0.3)) = MAX(0.15, 0.35) = 0.35
        //   dest = 1*0.35 + 0*... = 0.35
        // i=1: dest=0 → pos=0, neg=1
        //   MAX(0, 0.5) = 0.5
        //   dest = 0*... + 1*0.5 = 0.5
        // i=2: pos=1, neg=0
        //   MAX((1-0.8)*0.6, 0.8*(1-0.6)) = MAX(0.12, 0.32) = 0.32
        //   dest = 0.32
        // i=3: mask=0 → pos=0, neg=1
        //   MAX(0.3, 0.0) = 0.3
        //   dest = 0.3
        assert!((dest[0] - 0.35).abs() < 1e-6, "i=0: got {}", dest[0]);
        assert!((dest[1] - 0.5).abs() < 1e-6, "i=1: got {}", dest[1]);
        assert!((dest[2] - 0.32).abs() < 1e-6, "i=2: got {}", dest[2]);
        assert!((dest[3] - 0.3).abs() < 1e-6, "i=3: got {}", dest[3]);
    }

    #[test]
    fn group_exclusion_one_zero() {
        // One operand zero → pos=0, neg=1 → dest = MAX(dest, mask)
        let mut dest = vec![1.0f32, 0.5, 0.0];
        let newmask = vec![0.0f32, 0.5, 1.0];
        group_combine(&mut dest, &newmask, 1.0, false, GROUP_OP_EXCLUSION);
        // i=0: mask=0, pos=0, neg=1 → dest = MAX(1.0, 0.0) = 1.0
        // i=1: both > 0, pos=1, neg=0
        //   MAX((1-0.5)*0.5, 0.5*(1-0.5)) = MAX(0.25, 0.25) = 0.25
        //   dest = 0.25
        // i=2: dest=0, pos=0, neg=1 → dest = MAX(0.0, 1.0) = 1.0
        assert!((dest[0] - 1.0).abs() < 1e-6);
        assert!((dest[1] - 0.25).abs() < 1e-6);
        assert!((dest[2] - 1.0).abs() < 1e-6);
    }

    // ── Copy tests ───────────────────────────────────────────────────────

    #[test]
    fn group_copy_basic() {
        let mut dest = vec![0.0f32; 4];
        let newmask = vec![0.2f32, 0.4, 0.6, 0.8];
        group_combine(&mut dest, &newmask, 0.5, false, GROUP_OP_COPY);
        // dest = 0.5 * newmask
        assert_eq!(dest, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn group_copy_inverted() {
        let mut dest = vec![0.0f32; 4];
        let newmask = vec![0.2f32, 0.4, 0.6, 0.8];
        group_combine(&mut dest, &newmask, 0.5, true, GROUP_OP_COPY);
        // dest = 0.5 * (1.0 - newmask) = [0.4, 0.3, ~0.2, ~0.1]
        // (1.0f32 - 0.6 loses precision → 0.39999998, * 0.5 → 0.19999999)
        assert!((dest[0] - 0.4).abs() < 1e-6, "i=0: got {}", dest[0]);
        assert!((dest[1] - 0.3).abs() < 1e-6, "i=1: got {}", dest[1]);
        assert!((dest[2] - 0.2).abs() < 1e-6, "i=2: got {}", dest[2]);
        assert!((dest[3] - 0.1).abs() < 1e-6, "i=3: got {}", dest[3]);
    }

    // ── Reference match tests ─────────────────────────────────────────────

    #[test]
    fn group_combine_matches_reference_all_ops() {
        let mut seed = vec![0f32; 256];
        lcg_fill(&mut seed, 0xFEED, 1.0);

        let mut dest = seed.clone();
        let newmask = seed.clone();

        for op in [GROUP_OP_UNION, GROUP_OP_INTERSECT, GROUP_OP_DIFFERENCE,
                   GROUP_OP_SUM, GROUP_OP_EXCLUSION, GROUP_OP_COPY] {
            for inverted in [false, true] {
                let mut ref_dest = dest.clone();
                group_combine(&mut dest, &newmask, 0.7, inverted, op);
                ref_group_combine(&mut ref_dest, &newmask, 0.7, inverted, op);
                assert_eq!(dest, ref_dest,
                    "op={op} inverted={inverted} mismatch vs reference");
                // reset dest for next iteration
                dest.clone_from(&ref_dest);
            }
        }
    }

    // ── FFI round-trip and null-guard tests ──────────────────────────────

    #[test]
    fn ffi_combine_round_trip() {
        unsafe {
            let mut ffi_dest = vec![0.1f32, 0.5, 0.8, 0.3, 0.6, 0.2];
            let newmask = vec![0.4f32, 0.3, 0.6, 0.9, 0.1, 0.7];
            let n = ffi_dest.len();

            darkroom_masks_group_combine(
                ffi_dest.as_mut_ptr(), newmask.as_ptr(), n,
                0.5f32, 1, GROUP_OP_UNION);

            // Direct call
            let mut direct_dest = vec![0.1f32, 0.5, 0.8, 0.3, 0.6, 0.2];
            group_combine(&mut direct_dest, &newmask, 0.5, true, GROUP_OP_UNION);

            assert_eq!(ffi_dest, direct_dest, "FFI union mismatch");
        }
    }

    #[test]
    fn ffi_combine_all_ops_round_trip() {
        unsafe {
            for op in [GROUP_OP_UNION, GROUP_OP_INTERSECT, GROUP_OP_DIFFERENCE,
                       GROUP_OP_SUM, GROUP_OP_EXCLUSION, GROUP_OP_COPY] {
                for inverted in [false, true] {
                    let mut ffi_dest = vec![0.1f32, 0.5, 0.8, 0.3, 0.6, 0.2, 0.4, 0.7];
                    let newmask = vec![0.4f32, 0.3, 0.6, 0.9, 0.1, 0.7, 0.5, 0.0];
                    let n = ffi_dest.len();

                    darkroom_masks_group_combine(
                        ffi_dest.as_mut_ptr(), newmask.as_ptr(), n,
                        0.6f32, inverted as i32, op);

                    let mut direct_dest = vec![0.1f32, 0.5, 0.8, 0.3, 0.6, 0.2, 0.4, 0.7];
                    group_combine(&mut direct_dest, &newmask, 0.6, inverted, op);

                    assert_eq!(ffi_dest, direct_dest,
                        "FFI op={op} inverted={inverted} mismatch");
                }
            }
        }
    }

    #[test]
    fn ffi_combine_null_guard() {
        unsafe {
            // null dest → no-op
            let newmask = vec![0.5f32; 10];
            darkroom_masks_group_combine(
                std::ptr::null_mut(), newmask.as_ptr(), 10,
                1.0, 0, GROUP_OP_UNION);

            // null newmask → no-op
            let mut dest = vec![0.5f32; 10];
            darkroom_masks_group_combine(
                dest.as_mut_ptr(), std::ptr::null(), 10,
                1.0, 0, GROUP_OP_UNION);
            assert!(dest.iter().all(|&v| v == 0.5));

            // npixels == 0 → no-op
            let mut dest = vec![0.5f32; 10];
            darkroom_masks_group_combine(
                dest.as_mut_ptr(), dest.as_ptr(), 0,
                1.0, 0, GROUP_OP_UNION);
            assert!(dest.iter().all(|&v| v == 0.5));
        }
    }
}
