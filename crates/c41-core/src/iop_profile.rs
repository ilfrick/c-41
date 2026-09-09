//! Input-profile tone curves ported from `src/common/iop_profile.c`
//! (`_apply_tonecurves`, iop_profile.c:364-420, m4-187).
//!
//! What the loops do: each RGB pixel of a 4-lane (RGBA) image is mapped
//! through its per-channel tone curve — `extrapolate_lut` while the input
//! value is below 1.0, `eval_exp` (the unbounded exponential fit) at and
//! above 1.0. The C function holds two `DT_OMP_FOR(collapse(2))` loops: an
//! all-channels fast path taken when every LUT is active, and a
//! per-channel-sentinel path taken when only some LUTs are active. A LUT
//! whose first entry is negative (`lut[c][0] < 0.0`) marks a linear channel
//! and is skipped; when all three are linear neither branch runs.
//!
//! The single-threaded Rust port is identical for any OpenMP thread count:
//! pixels are independent (each output lane is written exactly once from its
//! own input lane), so the merge order is irrelevant.
//!
//! Semantic edge cases (all pinned by tests):
//! - Inactive (linear) channels are left **untouched** in `image_out` —
//!   they are NOT copied from `image_in`. Callers pre-fill or reuse the
//!   buffer (notably `_transform_lab_to_rgb_matrix` runs fully in place).
//!   This differs from `crate::color::apply_trc`, which passes inactive
//!   channels through from the input pixel.
//! - All three channels linear is a complete no-op: `image_out` is not
//!   written at all, including alpha.
//! - The alpha lane (`k + 3`) is never read or written in any branch; the C
//!   comment calls this out explicitly ("some code needs image_out[3]
//!   preserved").
//! - `v < 1.0` selects the LUT path (negative inputs clamp to `lut[0]` inside
//!   `extrapolate_lut`); `v >= 1.0` — including exactly 1.0 — selects
//!   `eval_exp`.
//! - The LUT/eval_exp arithmetic itself lives in the shared
//!   `crate::color::{extrapolate_lut, eval_exp}` helpers and is reused here,
//!   not duplicated.
//!
//! Degenerate/short buffers: zero `width`/`height`, `lutsize < 2` (below
//! which both the C loop and the shared helper index out of bounds — real
//! profiles carry `lutsize` floats per LUT, default `0x10000`), a dimension
//! arithmetic overflow, a LUT shorter than `lutsize`, or a coefficient slice
//! shorter than 3 floats is a no-op — no panic, no out-of-bounds access.
//! Well-formed callers (the two `_transform_*_matrix` wrappers, which forward
//! the profile's own `lutsize`-long LUTs over exactly `4*width*height`
//! floats) never hit this path.
//!
//! Aliasing: the split kernel requires non-overlapping buffers (like the C
//! `restrict` qualifiers). The in-place call
//! (`_transform_lab_to_rgb_matrix` invokes the C function with
//! `image_out, image_out`) is served by [`apply_tonecurves_inplace`]; the
//! FFI export dispatches on pointer equality.

use crate::color::{eval_exp, extrapolate_lut};

/// Map one channel value through its tone curve.
///
/// Returns `None` for inactive (linear, `lut[0] < 0.0`) channels so the
/// caller leaves that output lane untouched. Active channels take the LUT
/// below 1.0 and the unbounded exponential fit at and above 1.0.
#[inline(always)]
fn map_channel(v: f32, lut: &[f32], coeff: &[f32], lutsize: usize) -> Option<f32> {
    if lut[0] < 0.0 {
        return None;
    }
    Some(if v < 1.0 {
        extrapolate_lut(lut, v, lutsize)
    } else {
        eval_exp(coeff, v)
    })
}

/// Validate dimensions and buffer lengths for the tone-curve kernels.
///
/// Returns the pixel count (`width * height`) on success. Fails on zero
/// dimensions, `lutsize < 2`, dimension arithmetic overflow, image buffers
/// shorter than `4*width*height` floats, LUTs shorter than `lutsize`, or
/// coefficient slices shorter than 3 floats.
fn checked_len(
    image_in_len: usize,
    image_out_len: usize,
    luts: [&[f32]; 3],
    coeffs: [&[f32]; 3],
    width: usize,
    height: usize,
    lutsize: usize,
) -> Option<usize> {
    if width == 0 || height == 0 || lutsize < 2 {
        return None;
    }
    let npixels = width.checked_mul(height)?;
    let pix_len = npixels.checked_mul(4)?;
    if image_in_len < pix_len || image_out_len < pix_len {
        return None;
    }
    for lut in luts {
        if lut.len() < lutsize {
            return None;
        }
    }
    for coeff in coeffs {
        if coeff.len() < 3 {
            return None;
        }
    }
    Some(npixels)
}

/// Apply the input-profile tone curves (`_apply_tonecurves`, split buffers).
///
/// Port of the two `DT_OMP_FOR(collapse(2))` loops at iop_profile.c:387-398
/// (all LUTs active) and iop_profile.c:404-418 (per-channel sentinel), with
/// the branch structure preserved: the fast path writes all three channels
/// per pixel, the sentinel path writes only active channels, and the
/// all-linear case writes nothing. Channels 0..2 are mapped per
/// [`map_channel`]; channel 3 (alpha) is never touched. `lutr/lutg/lutb`
/// each hold `lutsize` floats; each coefficient slice holds 3 floats.
///
/// `image_in` and `image_out` must not overlap (matching the C `restrict`
/// qualifiers); use [`apply_tonecurves_inplace`] for the in-place call.
/// Degenerate/short inputs are a no-op as documented above.
#[allow(clippy::too_many_arguments)]
pub fn apply_tonecurves(
    image_in: &[f32],
    image_out: &mut [f32],
    width: usize,
    height: usize,
    lutr: &[f32],
    lutg: &[f32],
    lutb: &[f32],
    unbounded_coeffsr: &[f32],
    unbounded_coeffsg: &[f32],
    unbounded_coeffsb: &[f32],
    lutsize: usize,
) {
    let luts = [lutr, lutg, lutb];
    let coeffs = [unbounded_coeffsr, unbounded_coeffsg, unbounded_coeffsb];
    let Some(npixels) = checked_len(
        image_in.len(),
        image_out.len(),
        luts,
        coeffs,
        width,
        height,
        lutsize,
    ) else {
        return;
    };
    let active = [
        luts[0][0] >= 0.0,
        luts[1][0] >= 0.0,
        luts[2][0] >= 0.0,
    ];
    if active[0] && active[1] && active[2] {
        for px in 0..npixels {
            let b = px * 4;
            for c in 0..3 {
                let v = image_in[b + c];
                image_out[b + c] = if v < 1.0 {
                    extrapolate_lut(luts[c], v, lutsize)
                } else {
                    eval_exp(coeffs[c], v)
                };
            }
        }
    } else if active[0] || active[1] || active[2] {
        for px in 0..npixels {
            let b = px * 4;
            for c in 0..3 {
                if active[c] {
                    let v = image_in[b + c];
                    image_out[b + c] = if v < 1.0 {
                        extrapolate_lut(luts[c], v, lutsize)
                    } else {
                        eval_exp(coeffs[c], v)
                    };
                }
            }
        }
    }
    // all channels linear: neither C branch runs — output fully untouched.
}

/// Apply the input-profile tone curves in place.
///
/// Same semantics as [`apply_tonecurves`] on a single buffer (each lane is
/// read before it is written, so per-lane in-place mapping is exact). Serves
/// the `_transform_lab_to_rgb_matrix` call site, which invokes the C
/// function with `image_out, image_out`. Degenerate/short inputs are a
/// no-op as documented above.
#[allow(clippy::too_many_arguments)]
pub fn apply_tonecurves_inplace(
    image: &mut [f32],
    width: usize,
    height: usize,
    lutr: &[f32],
    lutg: &[f32],
    lutb: &[f32],
    unbounded_coeffsr: &[f32],
    unbounded_coeffsg: &[f32],
    unbounded_coeffsb: &[f32],
    lutsize: usize,
) {
    let luts = [lutr, lutg, lutb];
    let coeffs = [unbounded_coeffsr, unbounded_coeffsg, unbounded_coeffsb];
    let Some(npixels) = checked_len(image.len(), image.len(), luts, coeffs, width, height, lutsize)
    else {
        return;
    };
    let active = [
        luts[0][0] >= 0.0,
        luts[1][0] >= 0.0,
        luts[2][0] >= 0.0,
    ];
    if !(active[0] || active[1] || active[2]) {
        return;
    }
    for px in 0..npixels {
        let b = px * 4;
        for c in 0..3 {
            if active[c] {
                let v = image[b + c];
                image[b + c] = if v < 1.0 {
                    extrapolate_lut(luts[c], v, lutsize)
                } else {
                    eval_exp(coeffs[c], v)
                };
            }
        }
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for [`apply_tonecurves`]: channel-outer
/// looping (for each channel, sweep every pixel) through the shared
/// [`map_channel`] helper — writing a lane only when the helper returns
/// `Some` — where the kernel loops pixel-outer with an inlined branch.
/// Same no-op preconditions: well-formed buffers only.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ref_apply_tonecurves(
    image_in: &[f32],
    image_out: &mut [f32],
    width: usize,
    height: usize,
    lutr: &[f32],
    lutg: &[f32],
    lutb: &[f32],
    unbounded_coeffsr: &[f32],
    unbounded_coeffsg: &[f32],
    unbounded_coeffsb: &[f32],
    lutsize: usize,
) {
    let luts = [lutr, lutg, lutb];
    let coeffs = [unbounded_coeffsr, unbounded_coeffsg, unbounded_coeffsb];
    let Some(npixels) = checked_len(
        image_in.len(),
        image_out.len(),
        luts,
        coeffs,
        width,
        height,
        lutsize,
    ) else {
        return;
    };
    for (c, (&lut, &coeff)) in luts.iter().zip(coeffs.iter()).enumerate() {
        if lut[0] < 0.0 {
            continue;
        }
        for px in 0..npixels {
            let idx = px * 4 + c;
            if let Some(mapped) = map_channel(image_in[idx], lut, coeff, lutsize) {
                image_out[idx] = mapped;
            }
        }
    }
}

// ── FFI export ───────────────────────────────────────────────────────────────

/// Apply the input-profile tone curves (`_apply_tonecurves`).
///
/// `image_in`/`image_out` each hold `4*width*height` floats (RGBA; only
/// channels 0..2 are written, alpha preserved). `lutr/lutg/lutb` each hold
/// `lutsize` floats; each `unbounded_coeffs*` holds 3 floats. A LUT whose
/// first entry is negative marks a linear channel (left untouched). May be
/// called in place (`image_in == image_out`, as
/// `_transform_lab_to_rgb_matrix` does); otherwise the buffers must not
/// overlap (matching the C `restrict` qualifiers).
///
/// # Safety
/// All pointers must be valid for the stated lengths; the C caller forwards
/// the profile's own `lutsize`-long LUTs over exactly `4*width*height`
/// floats.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn darkroom_apply_tonecurves(
    image_in: *const f32,
    image_out: *mut f32,
    width: usize,
    height: usize,
    lutr: *const f32,
    lutg: *const f32,
    lutb: *const f32,
    unbounded_coeffsr: *const f32,
    unbounded_coeffsg: *const f32,
    unbounded_coeffsb: *const f32,
    lutsize: usize,
) {
    if image_in.is_null()
        || image_out.is_null()
        || lutr.is_null()
        || lutg.is_null()
        || lutb.is_null()
        || unbounded_coeffsr.is_null()
        || unbounded_coeffsg.is_null()
        || unbounded_coeffsb.is_null()
        || width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
        // lutsize mirrors the C `int` (negative values arrive as huge
        // size_t); below 2 both the C loop and `extrapolate_lut` index out
        // of bounds, so reject before constructing the slices below (the
        // safe kernels re-check defensively)
        || lutsize < 2
        || lutsize > i32::MAX as usize
    {
        return;
    }
    // with width, height <= i32::MAX the pixel product still needs checking
    // (4*width*height can overflow usize in theory); validate BEFORE
    // constructing the slices (the safe kernels re-check defensively)
    let Some(npixels) = width.checked_mul(height) else {
        return;
    };
    let Some(pix_len) = npixels.checked_mul(4) else {
        return;
    };
    let lutr = std::slice::from_raw_parts(lutr, lutsize);
    let lutg = std::slice::from_raw_parts(lutg, lutsize);
    let lutb = std::slice::from_raw_parts(lutb, lutsize);
    let unbounded_coeffsr = std::slice::from_raw_parts(unbounded_coeffsr, 3);
    let unbounded_coeffsg = std::slice::from_raw_parts(unbounded_coeffsg, 3);
    let unbounded_coeffsb = std::slice::from_raw_parts(unbounded_coeffsb, 3);
    if std::ptr::eq(image_in, image_out) {
        // in-place call (`_transform_lab_to_rgb_matrix` passes image_out
        // twice): a single mutable slice avoids shared+mutable aliasing.
        let image = std::slice::from_raw_parts_mut(image_out, pix_len);
        apply_tonecurves_inplace(
            image,
            width,
            height,
            lutr,
            lutg,
            lutb,
            unbounded_coeffsr,
            unbounded_coeffsg,
            unbounded_coeffsb,
            lutsize,
        );
    } else {
        let image_in = std::slice::from_raw_parts(image_in, pix_len);
        let image_out = std::slice::from_raw_parts_mut(image_out, pix_len);
        apply_tonecurves(
            image_in,
            image_out,
            width,
            height,
            lutr,
            lutg,
            lutb,
            unbounded_coeffsr,
            unbounded_coeffsg,
            unbounded_coeffsb,
            lutsize,
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::test_util::lcg_fill;

    /// Two-entry identity LUT: `extrapolate_lut([0,1], v) == v` exactly for
    /// `v in [0,1)` (`0*(1-v) + 1*v`), `lut[0]` for clamped negatives.
    fn identity_lut() -> Vec<f32> {
        vec![0.0, 1.0]
    }

    /// Identity exponential fit: `eval_exp([1,1,1], v) == v` exactly.
    fn identity_coeff() -> [f32; 3] {
        [1.0, 1.0, 1.0]
    }

    /// Linear-channel marker LUT: first entry negative, matching the C
    /// "omit luts marked as linear (negative as marker)" convention.
    fn linear_lut() -> Vec<f32> {
        vec![-1.0, 0.5]
    }

    #[test]
    fn all_active_identity_is_exact_and_preserves_alpha() {
        // 2x2 image, values in [0,1): identity LUT + identity exp fit maps
        // every RGB lane to itself bit-exactly; alpha prefilled with sentinel
        // must survive untouched.
        let (w, h) = (2usize, 2usize);
        let lr = identity_lut();
        let lg = identity_lut();
        let lb = identity_lut();
        let (cr, cg, cb) = (identity_coeff(), identity_coeff(), identity_coeff());
        let image_in: Vec<f32> = vec![
            0.0, 0.25, 0.5, 7.0, //
            0.75, 0.1, 0.9, 8.0, //
            0.33, 0.66, 0.99, 9.0, //
            0.125, 0.375, 0.625, 10.0,
        ];
        let mut image_out = vec![-99.0f32; 4 * w * h];
        apply_tonecurves(
            &image_in, &mut image_out, w, h, &lr, &lg, &lb, &cr, &cg, &cb, 2,
        );
        for px in 0..w * h {
            for c in 0..3 {
                assert_eq!(image_out[px * 4 + c], image_in[px * 4 + c]);
                assert_eq!(
                    image_out[px * 4 + c].to_bits(),
                    image_in[px * 4 + c].to_bits(),
                    "px={px} c={c}"
                );
            }
            // alpha preserved (was sentinel, never the input alpha)
            assert_eq!(image_out[px * 4 + 3].to_bits(), (-99.0f32).to_bits());
        }
    }

    #[test]
    fn mixed_sentinel_writes_only_active_channels() {
        // green marked linear: R and B map through identity, G and alpha
        // lanes keep their prefill sentinel (NOT copied from input).
        let (w, h) = (2usize, 1usize);
        let lr = identity_lut();
        let lg = linear_lut();
        let lb = identity_lut();
        let (cr, cg, cb) = (identity_coeff(), identity_coeff(), identity_coeff());
        let image_in: Vec<f32> = vec![0.2, 0.4, 0.6, 1.0, 0.8, 0.3, 0.1, 2.0];
        let mut image_out = vec![-77.0f32; 4 * w * h];
        apply_tonecurves(
            &image_in, &mut image_out, w, h, &lr, &lg, &lb, &cr, &cg, &cb, 2,
        );
        assert_eq!(image_out, vec![0.2, -77.0, 0.6, -77.0, 0.8, -77.0, 0.1, -77.0]);
        // single active channel (only blue): R/G untouched as well.
        let lr2 = linear_lut();
        let lg2 = linear_lut();
        let mut image_out2 = vec![-77.0f32; 4 * w * h];
        apply_tonecurves(
            &image_in, &mut image_out2, w, h, &lr2, &lg2, &lb, &cr, &cg, &cb, 2,
        );
        assert_eq!(
            image_out2,
            vec![-77.0, -77.0, 0.6, -77.0, -77.0, -77.0, 0.1, -77.0]
        );
    }

    #[test]
    fn all_linear_is_full_no_op() {
        // every LUT marked linear: neither C branch runs, output (including
        // alpha) keeps every prefill bit.
        let (w, h) = (3usize, 2usize);
        let l = linear_lut();
        let (cr, cg, cb) = (identity_coeff(), identity_coeff(), identity_coeff());
        let mut image_in = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut image_in, 0x1A9, 2.0);
        let mut image_out = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut image_out, 0x2B4, 3.0);
        let before = image_out.clone();
        apply_tonecurves(&image_in, &mut image_out, w, h, &l, &l, &l, &cr, &cg, &cb, 2);
        assert_eq!(image_out, before);
        // reference agrees on the no-op
        let mut ref_out = before.clone();
        ref_apply_tonecurves(&image_in, &mut ref_out, w, h, &l, &l, &l, &cr, &cg, &cb, 2);
        assert_eq!(ref_out, before);
    }

    #[test]
    fn above_one_takes_eval_exp_including_exactly_one() {
        // coeffs [1,2,0]: eval_exp = 2*(v)^0 = exactly 2.0 for every v
        // (powf(_, 0.0) == 1.0 is C99-mandated, unlike powf(x, 1.0)). v >= 1
        // — including exactly 1.0, which the C `< 1.0f` test sends to the
        // exp path — yields 2.0; v < 1 stays on the identity LUT.
        let (w, h) = (1usize, 4usize);
        let lr = identity_lut();
        let lg = identity_lut();
        let lb = identity_lut();
        let c2 = [1.0f32, 2.0, 0.0];
        let image_in: Vec<f32> = vec![
            1.0, 1.5, 0.5, 3.0, //
            2.0, 1.0, 1.25, 4.0, //
            0.0, 3.0, 0.999, 5.0, //
            10.0, 0.001, 1.0, 6.0,
        ];
        let mut image_out = vec![-1.0f32; 4 * w * h];
        apply_tonecurves(&image_in, &mut image_out, w, h, &lr, &lg, &lb, &c2, &c2, &c2, 2);
        assert_eq!(
            image_out,
            vec![
                2.0, 2.0, 0.5, -1.0, //
                2.0, 2.0, 2.0, -1.0, //
                0.0, 2.0, 0.999, -1.0, //
                2.0, 0.001, 2.0, -1.0,
            ]
        );
    }

    #[test]
    fn negative_inputs_clamp_to_lut_first_entry() {
        // extrapolate_lut clamps the position to [0, lutsize-1]: negatives
        // yield exactly lut[0]. Uses a non-identity LUT so the clamp value
        // (0.25) is distinguishable from the input.
        let lut = vec![0.25f32, 0.75];
        let c = identity_coeff();
        let image_in: Vec<f32> = vec![-0.5, -3.0, -0.0, 9.0, -100.0, 0.5, -1.0, 8.0];
        let mut image_out = vec![-1.0f32; 8];
        apply_tonecurves(&image_in, &mut image_out, 2, 1, &lut, &lut, &lut, &c, &c, &c, 2);
        // -0.0 clamps to lut[0] as well; 0.5 on [0.25,0.75] lerps to 0.5.
        assert_eq!(image_out, vec![0.25, 0.25, 0.25, -1.0, 0.25, 0.5, 0.25, -1.0]);
    }

    #[test]
    fn matches_reference_over_lcg() {
        // all-active, each-single-active, and mixed-sentinel LUT sets over
        // pseudo-random inputs spanning negatives, [0,1), 1.0 and above —
        // exact and to_bits equality against the divergent reference.
        let luts_all: Vec<Vec<f32>> = vec![
            vec![0.0, 0.2, 0.5, 0.9, 1.0],
            vec![0.1, 0.3, 0.4, 0.8, 1.0],
            vec![0.0, 0.0, 0.6, 0.7, 1.0],
        ];
        let coeff_sets: Vec<[f32; 3]> = vec![[1.0, 1.2, 0.9], [0.8, 1.0, 1.1], [1.1, 0.9, 1.0]];
        // sentinel variants: bit patterns for which channels are linear
        let sentinel_masks: Vec<[bool; 3]> = vec![
            [true, true, true],
            [true, false, true],
            [false, true, false],
            [true, false, false],
            [false, false, true],
        ];
        for mask in &sentinel_masks {
            let lr;
            let lg;
            let lb;
            // build per-mask LUT views without reallocating the active ones
            let lin = linear_lut();
            lr = if mask[0] { &luts_all[0] } else { &lin };
            lg = if mask[1] { &luts_all[1] } else { &lin };
            lb = if mask[2] { &luts_all[2] } else { &lin };
            for &(w, h) in &[(1usize, 1usize), (2, 1), (3, 2), (5, 4), (8, 7)] {
                let mut image_in = vec![0.0f32; 4 * w * h];
                // LCG in [0,2) shifted to [-0.5,1.5): covers clamp, LUT
                // and exp regions, including values near 1.0.
                lcg_fill(&mut image_in, 0x70C0 + w as u32 * 17 + h as u32, 2.0);
                for v in image_in.iter_mut().step_by(4) {
                    *v -= 0.5;
                }
                // force exact-boundary values into the stream
                if !image_in.is_empty() {
                    image_in[0] = 1.0;
                    if image_in.len() > 4 {
                        image_in[4] = -0.0;
                    }
                    if image_in.len() > 8 {
                        image_in[8] = 0.999_999_9;
                    }
                }
                let mut direct = vec![-33.0f32; 4 * w * h];
                let mut refr = vec![-33.0f32; 4 * w * h];
                apply_tonecurves(
                    &image_in, &mut direct, w, h, lr, lg, lb, &coeff_sets[0],
                    &coeff_sets[1], &coeff_sets[2], 5,
                );
                ref_apply_tonecurves(
                    &image_in, &mut refr, w, h, lr, lg, lb, &coeff_sets[0],
                    &coeff_sets[1], &coeff_sets[2], 5,
                );
                assert_eq!(direct, refr, "mask={mask:?} w={w} h={h}");
                for (d, r) in direct.iter().zip(refr.iter()) {
                    assert_eq!(d.to_bits(), r.to_bits(), "mask={mask:?} w={w} h={h}");
                }
                // alpha lanes never written in either implementation
                for px in 0..w * h {
                    assert_eq!(direct[px * 4 + 3].to_bits(), (-33.0f32).to_bits());
                }
            }
        }
    }

    #[test]
    fn inplace_matches_split() {
        // the in-place kernel must equal the split kernel with a copied
        // input, for both all-active and mixed-sentinel LUT sets.
        let luts_all: Vec<Vec<f32>> = vec![
            vec![0.0, 0.2, 0.5, 0.9, 1.0],
            vec![0.1, 0.3, 0.4, 0.8, 1.0],
            vec![0.0, 0.0, 0.6, 0.7, 1.0],
        ];
        let lin = linear_lut();
        let coeff_sets: Vec<[f32; 3]> = vec![[1.0, 1.2, 0.9], [0.8, 1.0, 1.1], [1.1, 0.9, 1.0]];
        let cases: Vec<(&[f32], &[f32], &[f32])> = vec![
            (&luts_all[0], &luts_all[1], &luts_all[2]),
            (&luts_all[0], &lin, &luts_all[2]),
            (&lin, &lin, &lin),
        ];
        for (lr, lg, lb) in cases {
            for &(w, h) in &[(1usize, 1usize), (4, 3), (7, 5)] {
                let mut base = vec![0.0f32; 4 * w * h];
                lcg_fill(&mut base, 0x10C4, 2.0);
                let mut split_out = vec![-5.0f32; 4 * w * h];
                // split kernel needs a defined output for untouched lanes:
                // seed with the input (what the in-place caller would hold).
                split_out.copy_from_slice(&base);
                apply_tonecurves(
                    &base, &mut split_out, w, h, lr, lg, lb, &coeff_sets[0],
                    &coeff_sets[1], &coeff_sets[2], 5,
                );
                let mut inplace = base.clone();
                apply_tonecurves_inplace(
                    &mut inplace, w, h, lr, lg, lb, &coeff_sets[0], &coeff_sets[1],
                    &coeff_sets[2], 5,
                );
                assert_eq!(inplace, split_out, "w={w} h={h}");
                for (a, b) in inplace.iter().zip(split_out.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "w={w} h={h}");
                }
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let lr = identity_lut();
        let c = identity_coeff();
        // zero dims leave buffers untouched
        for &(w, h) in &[(0usize, 4usize), (4usize, 0usize), (0usize, 0usize)] {
            let inn: Vec<f32> = vec![0.5; 64];
            let mut out = vec![9.0f32; 64];
            apply_tonecurves(&inn, &mut out, w, h, &lr, &lr, &lr, &c, &c, &c, 2);
            assert!(out.iter().all(|&v| v == 9.0), "w={w} h={h}");
            let mut io = vec![9.0f32; 64];
            apply_tonecurves_inplace(&mut io, w, h, &lr, &lr, &lr, &c, &c, &c, 2);
            assert!(io.iter().all(|&v| v == 9.0), "inplace w={w} h={h}");
        }
        // lutsize < 2 (both the C loop and extrapolate_lut index out of
        // bounds there) is a no-op, no panic
        for &lutsize in &[0usize, 1usize] {
            let inn: Vec<f32> = vec![0.5; 16];
            let mut out = vec![9.0f32; 16];
            apply_tonecurves(&inn, &mut out, 2, 2, &lr, &lr, &lr, &c, &c, &c, lutsize);
            assert_eq!(out, vec![9.0f32; 16], "lutsize={lutsize}");
        }
        // short buffers: length preconditions fail -> no-op, no panic
        let (w, h) = (4usize, 4usize);
        let mut inn = vec![0.0f32; 4 * w * h];
        lcg_fill(&mut inn, 0x5E07, 2.0);
        let mut out = vec![9.0f32; 4 * w * h];
        apply_tonecurves(&inn[..10], &mut out, w, h, &lr, &lr, &lr, &c, &c, &c, 2);
        apply_tonecurves(&inn, &mut out[..10], w, h, &lr, &lr, &lr, &c, &c, &c, 2);
        apply_tonecurves(&inn, &mut out, w, h, &lr[..1], &lr, &lr, &c, &c, &c, 2);
        apply_tonecurves(&inn, &mut out, w, h, &lr, &lr, &lr, &c[..2], &c, &c, 2);
        assert!(out.iter().all(|&v| v == 9.0));
        // boundary dimension arithmetic must no-op without panicking
        apply_tonecurves(&[], &mut [], usize::MAX, 1, &[], &[], &[], &[], &[], &[], 2);
        ref_apply_tonecurves(&[], &mut [], usize::MAX, 1, &[], &[], &[], &[], &[], &[], 2);
        apply_tonecurves_inplace(&mut [], usize::MAX, 1, &[], &[], &[], &[], &[], &[], 2);
    }

    #[test]
    fn ffi_round_trip() {
        // split and in-place FFI calls agree with the safe kernels.
        let luts_all: Vec<Vec<f32>> = vec![
            vec![0.0, 0.2, 0.5, 0.9, 1.0],
            vec![0.1, 0.3, 0.4, 0.8, 1.0],
            vec![0.0, 0.0, 0.6, 0.7, 1.0],
        ];
        let coeff_sets: Vec<[f32; 3]> = vec![[1.0, 1.2, 0.9], [0.8, 1.0, 1.1], [1.1, 0.9, 1.0]];
        for &(w, h) in &[(4usize, 2usize), (5usize, 3usize)] {
            let mut image_in = vec![0.0f32; 4 * w * h];
            lcg_fill(&mut image_in, 0xFF11 + w as u32, 2.0);
            let mut ffi_out = vec![-3.0f32; 4 * w * h];
            let mut direct = vec![-3.0f32; 4 * w * h];
            unsafe {
                darkroom_apply_tonecurves(
                    image_in.as_ptr(),
                    ffi_out.as_mut_ptr(),
                    w,
                    h,
                    luts_all[0].as_ptr(),
                    luts_all[1].as_ptr(),
                    luts_all[2].as_ptr(),
                    coeff_sets[0].as_ptr(),
                    coeff_sets[1].as_ptr(),
                    coeff_sets[2].as_ptr(),
                    5,
                );
            }
            apply_tonecurves(
                &image_in, &mut direct, w, h, &luts_all[0], &luts_all[1], &luts_all[2],
                &coeff_sets[0], &coeff_sets[1], &coeff_sets[2], 5,
            );
            assert_eq!(ffi_out, direct, "w={w} h={h}");
            // in-place FFI (same pointer twice, like
            // _transform_lab_to_rgb_matrix) matches the inplace kernel.
            let mut ffi_io = image_in.clone();
            let mut direct_io = image_in.clone();
            unsafe {
                darkroom_apply_tonecurves(
                    ffi_io.as_ptr(),
                    ffi_io.as_mut_ptr(),
                    w,
                    h,
                    luts_all[0].as_ptr(),
                    luts_all[1].as_ptr(),
                    luts_all[2].as_ptr(),
                    coeff_sets[0].as_ptr(),
                    coeff_sets[1].as_ptr(),
                    coeff_sets[2].as_ptr(),
                    5,
                );
            }
            apply_tonecurves_inplace(
                &mut direct_io, w, h, &luts_all[0], &luts_all[1], &luts_all[2],
                &coeff_sets[0], &coeff_sets[1], &coeff_sets[2], 5,
            );
            assert_eq!(ffi_io, direct_io, "inplace w={w} h={h}");
        }
    }

    #[test]
    fn ffi_guards() {
        let lr = identity_lut();
        let c = identity_coeff();
        let inn: Vec<f32> = vec![0.5; 64];
        let mut out = vec![7.0f32; 64];
        unsafe {
            // null pointers (each of the eight, one at a time)
            darkroom_apply_tonecurves(
                std::ptr::null(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), std::ptr::null_mut(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, std::ptr::null(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), std::ptr::null(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                std::ptr::null(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), std::ptr::null(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), std::ptr::null(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), std::ptr::null(), 2,
            );
            // zero dims and degenerate lutsize
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 0, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 0, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 1,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 0,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, 4, lr.as_ptr(), lr.as_ptr(),
                lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), (i32::MAX as usize) + 1,
            );
            // i32::MAX caps (C ints arrive non-negative; negatives would
            // wrap to huge size_t and must not reach slice construction)
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), (i32::MAX as usize) + 1, 4, lr.as_ptr(),
                lr.as_ptr(), lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
            darkroom_apply_tonecurves(
                inn.as_ptr(), out.as_mut_ptr(), 4, (i32::MAX as usize) + 1, lr.as_ptr(),
                lr.as_ptr(), lr.as_ptr(), c.as_ptr(), c.as_ptr(), c.as_ptr(), 2,
            );
        }
        assert!(out.iter().all(|&v| v == 7.0)); // untouched
    }
}
