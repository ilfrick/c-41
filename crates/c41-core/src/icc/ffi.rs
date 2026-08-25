//! C FFI boundary for the pure-Rust ICC transform ([`Transform`]) — the
//! entry points colorin/colorout's LUT path will call instead of LCMS
//! (`cmsCreateTransform`/`cmsDoTransform`), and the reason the engine exists.
//!
//! The C side owns profile *bytes* (it already reads them from disk); here we
//! parse, assemble, and evaluate. One handle = one (src, dst, intent) triple,
//! built once in `commit_params`, applied per band in `process`, freed in
//! `cleanup` — mirroring the cmsHTRANSFORM lifetime it replaces.

use super::{Profile, Transform};

/// Assemble a device→PCS→device transform between the two profiles given as
/// raw ICC bytes, under rendering `intent` (0 perceptual, 1 rel-colourimetric,
/// 2 saturation, 3 abs-colourimetric; anything above 3 is refused with NULL,
/// mirroring `cmsCreateTransform` failing on an unknown intent). Returns an
/// owned handle, or NULL when either profile fails to parse or the assembly is
/// impossible (e.g. no B2A tags and singular matrix-shaper colorants, a
/// non-3-channel GRAY/CMYK/N-colour profile, or a device colour space that is
/// neither RGB nor XYZ nor equal to the PCS — none of which this RGB engine
/// evaluates) — the caller must fall back exactly as it would have for a failed
/// `cmsCreateTransform`.
///
/// # Safety
/// `src`/`dst` must point at `src_len`/`dst_len` readable bytes. The returned
/// pointer must be released with [`darkroom_icc_transform_free`] exactly once.
#[no_mangle]
pub unsafe extern "C" fn darkroom_icc_transform_new(
    src: *const u8,
    src_len: usize,
    dst: *const u8,
    dst_len: usize,
    intent: u32,
) -> *mut Transform {
    if src.is_null() || dst.is_null() || intent > 3 {
        return std::ptr::null_mut();
    }
    let src = std::slice::from_raw_parts(src, src_len);
    let dst = std::slice::from_raw_parts(dst, dst_len);
    // Fail loud-and-null rather than panicking across the boundary: a corrupt
    // profile is a runtime condition here, not a bug.
    match Profile::parse(src).and_then(|s| Profile::parse(dst).and_then(|d| Transform::new(&s, &d, intent))) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Release a handle from [`darkroom_icc_transform_new`]. NULL is allowed (the
/// cleanup paths in colorin/colorout free conditionally-created transforms).
///
/// # Safety
/// `t` must be NULL or a handle from `darkroom_icc_transform_new` that has not
/// been freed yet; after this call it is dangling.
#[no_mangle]
pub unsafe extern "C" fn darkroom_icc_transform_free(t: *mut Transform) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// Transform `npixels` stride-4 RGBA floats from `in_buf` into `out_buf`,
/// passing the alpha lane through untouched and leaving colour lanes to the
/// assembled transform. **In-place operation (`in_buf == out_buf`) is
/// supported** — each pixel's RGB triplet is copied out before its output is
/// written, so there is no cross-pixel dependency (this mirrors the
/// `cmsDoTransform(xform, out, out, width)` callsites in colorin).
///
/// The colour lanes carry **raw** values in the profiles' PCS domain — raw Lab
/// (`L∈[0,100]`, `a,b∈[-128,127]`) or raw D50-referenced XYZ — exactly what a
/// `TYPE_LabA_FLT` / `TYPE_XYZA_FLT` LCMS float transform consumes and emits,
/// so slice-2's C wiring maps formats 1:1 with no rescaling.
///
/// The handle may be called concurrently from multiple band threads: evaluation
/// takes `&self` and touches no interior state.
///
/// # Safety
/// `t` must be a live handle; `in_buf`/`out_buf` must be valid for
/// `4·npixels` readable/writable floats (same buffer allowed).
#[no_mangle]
pub unsafe extern "C" fn darkroom_icc_transform_apply_rgba(
    t: *const Transform,
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
) {
    let t = match t.as_ref() {
        Some(t) => t,
        None => return, // a dead handle leaves the buffer untouched, not garbage
    };
    for j in 0..npixels {
        let p = in_buf.add(4 * j);
        let rgb = [*p, *p.add(1), *p.add(2)];
        let mut o = [0.0f32; 3];
        t.eval_into(&rgb, &mut o);
        let q = out_buf.add(4 * j);
        *q = o[0];
        *q.add(1) = o[1];
        *q.add(2) = o[2];
        *q.add(3) = *p.add(3); // alpha passthrough
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icc::lut::Stage;
    use crate::icc::transform::tests::test_helpers::*;

    /// sRGB-like D50 matrix-shaper source (full-matrix colorants so absolute
    /// intent doesn't cancel) against the same profile as destination: the
    /// identity-ish round trip every FFI test below builds on.
    fn srgb_pair_bytes() -> Vec<u8> {
        srgb_like_profile(b"XYZ ", [0.9642, 1.0, 0.8249])
    }

    #[test]
    fn ffi_handle_roundtrips_and_applies_bit_exact_to_rust_eval() {
        let bytes = srgb_pair_bytes();
        // SAFETY: byte slices outlive the calls; handles freed below.
        let t = unsafe {
            darkroom_icc_transform_new(bytes.as_ptr(), bytes.len(), bytes.as_ptr(), bytes.len(), 1)
        };
        assert!(!t.is_null(), "valid pair must assemble");

        // One grey pixel through the C ABI…
        let mut px = [0.25f32, 0.5, 0.75, 1.0];
        unsafe { darkroom_icc_transform_apply_rgba(t, px.as_ptr(), px.as_mut_ptr(), 1) };
        // …must equal the Rust-side evaluation of the same vector.
        let parsed = Profile::parse(&bytes).unwrap();
        let tr = Transform::new(&parsed, &parsed, 1).unwrap();
        let want = tr.eval(&[0.25, 0.5, 0.75]);
        for i in 0..3 {
            assert_eq!(px[i], want[i], "lane {i}");
        }
        assert_eq!(px[3], 1.0, "alpha passes through");
        unsafe { darkroom_icc_transform_free(t) };
    }

    #[test]
    fn ffi_apply_supports_in_place_rows_with_alpha_passthrough() {
        let bytes = srgb_pair_bytes();
        let t = unsafe {
            darkroom_icc_transform_new(bytes.as_ptr(), bytes.len(), bytes.as_ptr(), bytes.len(), 0)
        };
        assert!(!t.is_null());

        // A row of distinct pixels + alphas, transformed IN PLACE — the exact
        // shape colorin's `cmsDoTransform(xform, out, out, width)` callsites use.
        let mut row: Vec<f32> = (0..7usize)
            .flat_map(|j| [j as f32 * 0.1, 0.2, 0.9 - j as f32 * 0.05, j as f32 * 0.125])
            .collect();
        let snapshot = row.clone();
        unsafe { darkroom_icc_transform_apply_rgba(t, row.as_ptr(), row.as_mut_ptr(), 7) };

        let parsed = Profile::parse(&bytes).unwrap();
        let tr = Transform::new(&parsed, &parsed, 0).unwrap();
        for j in 0..7 {
            let want = tr.eval(&snapshot[4 * j..4 * j + 3]);
            assert_eq!(&row[4 * j..4 * j + 3], &want[..], "pixel {j} (in place)");
            assert_eq!(row[4 * j + 3], snapshot[4 * j + 3], "alpha {j}");
        }
        unsafe { darkroom_icc_transform_free(t) };
    }

    #[test]
    fn ffi_null_and_garbage_inputs_are_refused_not_panics() {
        // Null byte pointers → null handle.
        assert!(unsafe {
            darkroom_icc_transform_new(std::ptr::null(), 10, std::ptr::null(), 10, 0).is_null()
        });
        // Garbage bytes → parse error → null handle (never a panic across FFI).
        let junk = [0u8; 64];
        let ok_bytes = srgb_pair_bytes();
        assert!(unsafe {
            darkroom_icc_transform_new(junk.as_ptr(), junk.len(), ok_bytes.as_ptr(), ok_bytes.len(), 0)
                .is_null()
        });
        assert!(unsafe {
            darkroom_icc_transform_new(ok_bytes.as_ptr(), ok_bytes.len(), junk.as_ptr(), junk.len(), 0)
                .is_null()
        });
        // Unknown rendering intent → null handle, like cmsCreateTransform failing.
        assert!(unsafe {
            darkroom_icc_transform_new(ok_bytes.as_ptr(), ok_bytes.len(), ok_bytes.as_ptr(), ok_bytes.len(), 4)
                .is_null()
        });
        // Freeing NULL is a no-op by contract.
        unsafe { darkroom_icc_transform_free(std::ptr::null_mut()) };
        // Applying through a null handle must leave the buffer untouched.
        let mut px = [1.0f32, 2.0, 3.0, 4.0];
        unsafe { darkroom_icc_transform_apply_rgba(std::ptr::null(), px.as_ptr(), px.as_mut_ptr(), 1) };
        assert_eq!(px, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn non_three_channel_profile_yields_a_null_handle() {
        // A GRAY profile parses but violates the engine's 3-channel contract;
        // the FFI constructor must surface that as NULL so the C caller falls
        // back exactly as it did for a failed cmsCreateTransform — instead of
        // later panicking on the first apply_rgba pixel.
        let gray_bytes =
            build_profile(b"scnr", b"GRAY", b"XYZ ", &[(b"A2B0", mft1_gray_lut())]);
        Profile::parse(&gray_bytes).expect("the GRAY fixture itself must parse");
        let t = unsafe {
            darkroom_icc_transform_new(
                gray_bytes.as_ptr(),
                gray_bytes.len(),
                gray_bytes.as_ptr(),
                gray_bytes.len(),
                1,
            )
        };
        assert!(t.is_null(), "non-3-channel pipelines must be refused at assembly");
    }

    #[test]
    fn eval_into3_matches_pipeline_eval_bit_exact() {
        // The allocation-free plumbing must not change results on a pipeline
        // carrying every stage kind (identity mft1 → Curves + Matrix + Clut).
        let lut_bytes = mft1_identity_lut();
        let pipe = crate::icc::parse_lut_tag(&lut_bytes).expect("helper lut parses");
        assert!(matches!(pipe.stages.first(), Some(Stage::Curves(_))), "{pipe:?}");
        let x = [0.13f32, 0.57, 0.91];
        let mut o = [0.0f32; 3];
        pipe.eval_into3(&x, &mut o);
        assert_eq!(o.to_vec(), pipe.eval(&x), "eval_into3 ≡ eval");
    }
}
