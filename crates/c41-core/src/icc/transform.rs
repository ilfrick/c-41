//! Transform assembly (m4-127): chain an input profile's device→PCS direction
//! ([`Profile::a2b_pipeline`]) to an output profile's PCS→device direction
//! ([`Profile::b2a_pipeline`]), bridging the two profiles' PCS spaces and applying
//! the rendering intent's extra semantics.
//!
//! Why not one flat [`Pipeline`]: the Lab↔XYZ bridge is *non-linear* (the CIE
//! cube-root), and `Stage` is deliberately pure data (`Clone + PartialEq`, cheap
//! to test). A [`Transform`] therefore keeps head/bridge/tail apart and applies
//! the bridge between them in [`Transform::eval`].

use super::lut::{Pipeline, Stage};
use super::parser::{IccError, Profile};

/// ICC's reference white for the profile connection space: D50, the value the
/// specification fixes (not the brighter D65 our working space uses — that
/// adaptation belongs to colorin/colorout at the FFI layer).
const D50: [f32; 3] = [0.9642, 1.0, 0.8249];

// CIE Lab constants (ICC's exact rational forms)
const LAB_EPS: f32 = 216.0 / 24389.0;
const LAB_KAPPA: f32 = 24389.0 / 27.0;

#[inline]
fn lab_f(t: f32) -> f32 {
    if t > LAB_EPS {
        t.cbrt()
    } else {
        (LAB_KAPPA * t + 16.0) / 116.0
    }
}

/// Raw XYZ (D50-referenced) → raw Lab (`L∈[0,100]`, `a,b` around 0).
pub(crate) fn lab_from_xyz(xyz: &[f32]) -> [f32; 3] {
    let fx = lab_f(xyz[0] / D50[0]);
    let fy = lab_f(xyz[1] / D50[1]);
    let fz = lab_f(xyz[2] / D50[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Raw Lab → raw XYZ (D50-referenced). Inverse of [`lab_from_xyz`].
pub(crate) fn xyz_from_lab(lab: &[f32]) -> [f32; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    // invert lab_f per axis: f³ when above ε, else the linear leg
    let inv = |f: f32| -> f32 {
        let c = f * f * f;
        if c > LAB_EPS {
            c
        } else {
            (116.0 * f - 16.0) / LAB_KAPPA
        }
    };
    [inv(fx) * D50[0], inv(fy) * D50[1], inv(fz) * D50[2]]
}

/// An assembled device→PCS→device colour transform between two profiles.
///
/// Built once per (source, destination, intent) triple, evaluated per pixel
/// vector. All arithmetic is `f32`; construction errors are the profiles' own
/// ([`IccError`]).
pub struct Transform {
    head: Pipeline,
    /// Convert the head's raw Lab output to raw XYZ before anything downstream.
    /// Set when the source speaks Lab and either the destination speaks XYZ or
    /// absolute-intent scaling is active (which operates in the XYZ domain).
    to_xyz: bool,
    /// Absolute-intent media-white ratio (`wtpt_dst/wtpt_src`, componentwise),
    /// applied in the XYZ domain.
    abs_scale: Option<[f32; 3]>,
    /// Convert the (possibly scaled) raw XYZ signal to raw Lab for the tail.
    /// Set when the destination speaks Lab and the arriving signal is XYZ —
    /// because the source spoke XYZ, or [`Self::to_xyz`] converted it.
    from_xyz: bool,
    tail: Pipeline,
}

impl Transform {
    /// Assemble the transform from `src` (device → PCS) to `dst` (PCS → device)
    /// under rendering `intent` (0 perceptual, 1 rel-colourimetric, 2 saturation,
    /// 3 abs-colourimetric).
    ///
    /// Intent handling follows the tag-preference scheme of
    /// [`Profile::a2b_pipeline`]/[`Profile::b2a_pipeline`] (perceptual/saturation
    /// use their dedicated tables when present, else rel-colourimetric ones), plus
    /// **absolute** intent's white-point ratio scaling applied in the XYZ domain:
    /// `XYZ·(wtpt_dst/wtpt_src)` componentwise (the simple ICC ratio form; LCMS's
    /// full absolute path additionally round-trips chromatic adaptation through
    /// each profile's `chad` tag, which only matters for mismatched illuminants —
    /// documented deviation). If either profile lacks a usable `wtpt`, absolute
    /// degrades silently to the relative result rather than failing.
    pub fn new(src: &Profile, dst: &Profile, intent: u32) -> Result<Transform, IccError> {
        // The pipelines below interpret device values as RGB (or as XYZ/Lab for
        // colour-space-conversion profiles whose device space *is* the PCS).
        // Any other colour-space signature — say a device-`Lab ` profile with
        // `XYZ ` PCS and full matrix-shaper tags — would slide into the XYZ
        // shaper path and evaluate to plausible garbage instead of failing.
        // Refuse such profiles outright so the C caller falls back to LCMS.
        for p in [src, dst] {
            let conversion = p.data_space == p.pcs;
            let rgb_or_xyz = p.data_space == *b"RGB " || p.data_space == *b"XYZ ";
            if !conversion && !rgb_or_xyz {
                return Err(IccError::WrongTagType);
            }
        }
        let head = src.a2b_pipeline(intent)?;
        let tail = dst.b2a_pipeline(intent)?;

        // The band path ([`Pipeline::eval_into3`]) evaluates on stack `[f32; 3]`s,
        // so every pipeline must be 3-channel throughout. ICC profiles exist in
        // GRAY/N-channel/CMYK flavours whose LUT tags parse cleanly but would
        // mis-evaluate here — refuse them at assembly, converting what would be
        // an out-of-bounds panic (or silently dropped channels) into the FFI
        // boundary's documented NULL-on-failure contract.
        for stage in head.stages.iter().chain(tail.stages.iter()) {
            let three_channel = match stage {
                Stage::Curves(cs) => cs.len() == 3,
                Stage::Clut(c) => c.input_channels() == 3 && c.output_channels == 3,
                Stage::Matrix(_) => true,
            };
            if !three_channel {
                return Err(IccError::WrongTagType);
            }
        }

        // Absolute intent: scale by the media-white ratio in XYZ PCS. Both
        // whites must be usable (all-positive) — profiles are untrusted bytes,
        // and a zero/negative component would zero/flip a channel.
        let abs_scale = if intent == 3 {
            match (src.read_xyz(b"wtpt"), dst.read_xyz(b"wtpt")) {
                (Ok(s), Ok(d))
                    if [s.x, s.y, s.z].iter().all(|c| *c > 0.0)
                        && [d.x, d.y, d.z].iter().all(|c| *c > 0.0) =>
                {
                    Some([d.x / s.x, d.y / s.y, d.z / s.z])
                }
                _ => None,
            }
        } else {
            None
        };

        // The bridge steps, derived from the two PCS spaces and the scaling:
        // scaling must happen in XYZ, so a Lab-speaking source converts first.
        let to_xyz = src.pcs_is_lab() && (abs_scale.is_some() || !dst.pcs_is_lab());
        let arriving_xyz = !src.pcs_is_lab() || to_xyz;
        let from_xyz = dst.pcs_is_lab() && arriving_xyz;

        Ok(Transform { head, to_xyz, abs_scale, from_xyz, tail })
    }

    /// Evaluate on one pixel vector (`src` device channels wide → `dst`'s width —
    /// both 3 for every space C41 transforms today).
    pub fn eval(&self, input: &[f32]) -> Vec<f32> {
        let mut out = [0.0f32; 3];
        let src: [f32; 3] =
            input.try_into().expect("Transform evaluates 3-channel vectors only");
        self.eval_into(&src, &mut out);
        out.to_vec()
    }

    /// Evaluate on one 3-channel vector writing into `out`, without allocating.
    /// Bit-exact to [`Self::eval`] — same order, same arithmetic, only the
    /// plumbing differs. This is what the FFI apply path calls per pixel (hence
    /// crate-scoped: `eval` is the public entry point); the allocation-free
    /// property comes from [`Pipeline::eval_into3`]'s stack intermediates plus
    /// these stack arrays.
    pub(crate) fn eval_into(&self, input: &[f32; 3], out: &mut [f32; 3]) {
        let mut v = [0.0f32; 3];
        self.head.eval_into3(input, &mut v);
        if self.to_xyz {
            v = xyz_from_lab(&v);
        }
        if let Some(s) = &self.abs_scale {
            for i in 0..3 {
                v[i] *= s[i];
            }
        }
        if self.from_xyz {
            v = lab_from_xyz(&v);
        }
        self.tail.eval_into3(&v, out);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::icc::lut::{parse_lut_tag, Stage};
    use crate::icc::Curve;

    /// Shared synthetic-profile builders for sibling icc test modules (ffi.rs).
    /// One definition here; re-exported below so the FFI boundary tests parse
    /// the exact same bytes the engine tests do. Not every importer consumes
    /// every builder — that is what makes it a shared kit.
    pub(crate) mod test_helpers {
        #![allow(unused_imports)]
        pub(crate) use super::{build_profile, curv_gamma_tag, mft1_gray_lut, mft1_identity_lut,
                               matrix_profile, srgb_like_profile, xyz_tag};
    }

    /// Build a minimal but valid ICC profile (same shape as parser.rs's test
    /// helper): header + tag table + padded tag payloads.
    pub(crate) fn build_profile(
        class: &[u8; 4],
        data_space: &[u8; 4],
        pcs: &[u8; 4],
        tags: &[(&[u8; 4], Vec<u8>)],
    ) -> Vec<u8> {
        let header_len = 128;
        let table_len = 4 + tags.len() * 12;
        let mut data_off = header_len + table_len;
        let mut offsets = Vec::new();
        for (_, d) in tags {
            offsets.push(data_off);
            data_off += (d.len() + 3) & !3;
        }
        let total = data_off;
        let mut b = vec![0u8; total];
        b[0..4].copy_from_slice(&(total as u32).to_be_bytes());
        b[8] = 4; // version major → v4 encode/decode paths
        b[12..16].copy_from_slice(class);
        b[16..20].copy_from_slice(data_space);
        b[20..24].copy_from_slice(pcs);
        b[36..40].copy_from_slice(b"acsp");
        b[64..68].copy_from_slice(&1u32.to_be_bytes()); // rel colorimetric
        b[128..132].copy_from_slice(&(tags.len() as u32).to_be_bytes());
        for (i, ((sig, d), &off)) in tags.iter().zip(offsets.iter()).enumerate() {
            let base = 132 + i * 12;
            b[base..base + 4].copy_from_slice(*sig);
            b[base + 4..base + 8].copy_from_slice(&(off as u32).to_be_bytes());
            b[base + 8..base + 12].copy_from_slice(&(d.len() as u32).to_be_bytes());
            b[off..off + d.len()].copy_from_slice(d);
        }
        b
    }

    pub(crate) fn xyz_tag(x: f32, y: f32, z: f32) -> Vec<u8> {
        let mut d = vec![0u8; 20];
        d[0..4].copy_from_slice(b"XYZ ");
        for (i, v) in [x, y, z].iter().enumerate() {
            let fixed = (*v * 65536.0).round() as i32 as u32;
            d[8 + i * 4..12 + i * 4].copy_from_slice(&fixed.to_be_bytes());
        }
        d
    }

    pub(crate) fn curv_gamma_tag(g: f32) -> Vec<u8> {
        let mut d = vec![0u8; 14];
        d[0..4].copy_from_slice(b"curv");
        d[8..12].copy_from_slice(&1u32.to_be_bytes());
        let word = ((g * 256.0).round()) as u16;
        d[12..14].copy_from_slice(&word.to_be_bytes());
        d
    }

    /// A synthetic RGB matrix-shaper display profile whose primaries are the
    /// axis-aligned components of its white point (so device white ⇔ wtpt
    /// exactly, and both matrices involved stay diagonal/invertible), TRC =
    /// pure gamma.
    pub(crate) fn matrix_profile(pcs: &[u8; 4], wtpt: [f32; 3]) -> Vec<u8> {
        build_profile(
            b"mntr",
            b"RGB ",
            pcs,
            &[
                (b"wtpt", xyz_tag(wtpt[0], wtpt[1], wtpt[2])),
                (b"rXYZ", xyz_tag(wtpt[0], 0.0, 0.0)),
                (b"gXYZ", xyz_tag(0.0, wtpt[1], 0.0)),
                (b"bXYZ", xyz_tag(0.0, 0.0, wtpt[2])),
                (b"rTRC", curv_gamma_tag(2.2)),
                (b"gTRC", curv_gamma_tag(2.2)),
                (b"bTRC", curv_gamma_tag(2.2)),
            ],
        )
    }

    #[test]
    fn lab_xyz_roundtrip_is_exact_to_tolerance() {
        for xyz in [
            [0.9642, 1.0, 0.8249], // D50 white → L=100, a=b=0
            [0.2, 0.3, 0.4],
            [0.05, 0.05, 0.05],
            [0.001, 0.002, 0.003], // linear-leg territory
        ] {
            let lab = lab_from_xyz(&xyz);
            let back = xyz_from_lab(&lab);
            for i in 0..3 {
                assert!((back[i] - xyz[i]).abs() < 1e-4, "xyz {xyz:?} → {back:?}");
            }
        }
        // white is exactly L=100, a=b=0 (within f32 tolerance)
        let lab = lab_from_xyz(&[0.9642, 1.0, 0.8249]);
        assert!((lab[0] - 100.0).abs() < 1e-3, "{lab:?}");
        assert!(lab[1].abs() < 1e-3 && lab[2].abs() < 1e-3, "{lab:?}");
    }

    #[test]
    fn gamma_and_table_curves_invert_roundtrip() {
        let g = Curve::Gamma(2.2);
        let gi = g.inverse();
        for x in [0.0f32, 0.1, 0.5, 0.9, 1.0] {
            let y = gi.eval(g.eval(x));
            assert!((y - x).abs() < 1e-5, "gamma x={x} → {y}");
        }
        // degenerate gamma has no inverse; must fall back to identity
        assert_eq!(Curve::Gamma(0.0).inverse(), Curve::Identity);

        // a sampled sRGB-like table inverts within table resolution
        let n = 256;
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f32 / (n - 1) as f32;
            t.push(if x <= 0.04045 {
                (x / 12.92 * 65535.0).round() as u16
            } else {
                (((x + 0.055) / 1.055).powf(2.4) * 65535.0).round() as u16
            });
        }
        let tbl = Curve::Table(t);
        let ti = tbl.inverse();
        for x in [0.02f32, 0.1, 0.5, 0.9] {
            let y = ti.eval(tbl.eval(x));
            assert!((y - x).abs() < 0.005, "table x={x} → {y}");
        }
    }

    #[test]
    fn parametric_curve_inverts_by_sampling() {
        // para type 3 (sRGB's own form): Y = X^g for X ≥ d else cX
        let para = Curve::Parametric {
            func: 3,
            params: vec![2.4, 1.0 / 1.055, 0.055, 1.0 / 12.92, 0.04045],
        };
        let pi = para.inverse();
        for x in [0.01f32, 0.1, 0.5, 0.95] {
            let y = pi.eval(para.eval(x));
            assert!((y - x).abs() < 0.005, "para x={x} → {y}");
        }
    }

    /// Same shape as [`matrix_profile`] but with proper (non-diagonal,
    /// invertible) sRGB-D50 colorants whose columns sum exactly to D50 — so a
    /// white-ratio XYZ scaling cannot cancel against the inverse colorants.
    pub(crate) fn srgb_like_profile(pcs: &[u8; 4], wtpt: [f32; 3]) -> Vec<u8> {
        let c = [
            [0.4360f32, 0.3851, 0.1431], // rXYZ (x, y, z)
            [0.2225, 0.7169, 0.0606],    // gXYZ
            [0.0139, 0.0971, 0.7141],    // bXYZ
        ];
        build_profile(
            b"mntr",
            b"RGB ",
            pcs,
            &[
                (b"wtpt", xyz_tag(wtpt[0], wtpt[1], wtpt[2])),
                (b"rXYZ", xyz_tag(c[0][0], c[1][0], c[2][0])),
                (b"gXYZ", xyz_tag(c[0][1], c[1][1], c[2][1])),
                (b"bXYZ", xyz_tag(c[0][2], c[1][2], c[2][2])),
                (b"rTRC", curv_gamma_tag(2.2)),
                (b"gTRC", curv_gamma_tag(2.2)),
                (b"bTRC", curv_gamma_tag(2.2)),
            ],
        )
    }

    #[test]
    fn matrix_profile_transforms_to_itself() {
        // same profile on both sides, rel intent: device → XYZ → same device;
        // exercised on both a diagonal and a full-matrix colorant set
        for bytes in [
            matrix_profile(b"XYZ ", [0.9642, 1.0, 0.8249]),
            srgb_like_profile(b"XYZ ", [0.9642, 1.0, 0.8249]),
        ] {
            let p = Profile::parse(&bytes).unwrap();
            let t = Transform::new(&p, &p, 1).unwrap();
            for rgb in [[0.25f32, 0.5, 0.75], [0.1, 0.2, 0.3], [1.0, 0.0, 0.0]] {
                let out = t.eval(&rgb);
                for i in 0..3 {
                    assert!((out[i] - rgb[i]).abs() < 1e-3, "{rgb:?} → {out:?}");
                }
            }
        }
    }

    /// Minimal lut8 (`mft1`) payload: identity e-matrix (omitted by the parser),
    /// identity input/output tables, 2³ identity CLUT — encoded input in →
    /// encoded-shaped output unchanged. `zero_outputs` flattens the output
    /// tables instead, giving a distinguishable "wrong table was picked" probe.
    fn mft1_lut(zero_outputs: bool) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"mft1");
        d.extend_from_slice(&[0u8; 4]); // reserved
        d.push(3); // input channels
        d.push(3); // output channels
        d.push(2); // grid points per dim
        d.push(0); // pad — the e-matrix is aligned at byte 12
        for row in 0..3 {
            for col in 0..3 {
                let v: u32 = if row == col { 1 << 16 } else { 0 }; // identity, s15f16
                d.extend_from_slice(&v.to_be_bytes());
            }
        }
        // 256-entry identity input tables ×3 (lut8 stores u8)
        for _ in 0..3 {
            for i in 0..256u16 {
                d.push(i as u8);
            }
        }
        // CLUT: 8 nodes × 3 outputs, node (i,j,k) → (i,j,k)/255 (identity)
        for i in 0..2u8 {
            for j in 0..2u8 {
                for k in 0..2u8 {
                    d.extend_from_slice(&[i * 255, j * 255, k * 255]);
                }
            }
        }
        // 256-entry output tables ×3 (identity, or flat zero)
        for _ in 0..3 {
            for i in 0..256u16 {
                d.push(if zero_outputs { 0 } else { i as u8 });
            }
        }
        d
    }

    pub(crate) fn mft1_identity_lut() -> Vec<u8> {
        mft1_lut(false)
    }

    /// A 1-in/1-out lut8 payload — the A2B0 shape a GRAY ICC profile carries.
    /// The *parser* accepts it (channel counts are capped at LCMS's 16, not
    /// pinned to 3), which is exactly why assembly must reject such pipelines.
    pub(crate) fn mft1_gray_lut() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"mft1");
        d.extend_from_slice(&[0u8; 4]); // reserved
        d.push(1); // input channels
        d.push(1); // output channels
        d.push(2); // grid points per dim
        d.push(0); // pad
        // lut8 always carries the 36-byte e-matrix at offsets 12..48 (identity
        // here) — the parser's table cursor starts at 48 regardless of channels.
        d.extend_from_slice(&[0u8; 36]);
        // identity input table ×1 (lut8 stores u8)
        for i in 0..256u16 {
            d.push(i as u8);
        }
        // CLUT: 2 nodes × 1 output, identity
        d.extend_from_slice(&[0u8, 255]);
        // identity output table ×1
        for i in 0..256u16 {
            d.push(i as u8);
        }
        d
    }

    /// Flip a built profile's version-major header byte — exercises the v2
    /// encode/decode branches without duplicating every builder.
    fn with_version(mut bytes: Vec<u8>, major: u8) -> Vec<u8> {
        bytes[8] = major;
        bytes
    }

    #[test]
    fn mft1_helper_parses_with_expected_stage_layout() {
        let p = parse_lut_tag(&mft1_identity_lut()).unwrap();
        assert_eq!(p.stages.len(), 3, "curves → clut → curves: {:?}", p.stages);
        assert!(matches!(p.stages[1], Stage::Clut(_)));
    }

    #[test]
    fn xyz_pcs_source_bridges_to_lab_pcs_destination() {
        // Destination speaks Lab PCS and carries an identity mft1 B2A0 table:
        // its tail prepends the v4 Lab encode (L/100, a,b → a,b/255+128/255), so
        // the transform maps device RGB → raw XYZ → raw Lab → encoded Lab.
        let dst_bytes = build_profile(
            b"mntr",
            b"RGB ",
            b"Lab ",
            &[(b"B2A0", mft1_identity_lut())],
        );
        let src = Profile::parse(&matrix_profile(b"XYZ ", [0.9642, 1.0, 0.8249])).unwrap();
        let dst = Profile::parse(&dst_bytes).unwrap();
        let t = Transform::new(&src, &dst, 1).unwrap();

        // device 0.5 grey → linear 0.5^2.2 on each channel → XYZ ∝ D50 →
        // achromatic Lab → encoded (L/100, 128/255, 128/255)
        let out = t.eval(&[0.5, 0.5, 0.5]);
        let lin = 0.5f32.powf(2.2);
        let lab = lab_from_xyz(&[lin * 0.9642, lin, lin * 0.8249]);
        assert!(
            (out[0] - lab[0] / 100.0).abs() < 5e-3,
            "out={out:?} expected L_enc≈{}",
            lab[0] / 100.0
        );
        let neutral = 128.0f32 / 255.0;
        assert!(
            (out[1] - neutral).abs() < 1e-3 && (out[2] - neutral).abs() < 1e-3,
            "achromatic → a,b encoded ≈ {neutral}: {out:?}"
        );
    }

    #[test]
    fn lab_pcs_source_bridges_to_xyz_pcs_destination() {
        // Source speaks Lab PCS with an identity mft1 A2B0 (its output runs
        // through the v4 Lab decode); destination is our XYZ matrix profile whose
        // B2A fallback consumes raw XYZ directly. Encoded-Lab grey in →
        // neutral linear RGB out.
        let src_bytes = build_profile(
            b"mntr",
            b"RGB ",
            b"Lab ",
            &[(b"A2B0", mft1_identity_lut())],
        );
        let src = Profile::parse(&src_bytes).unwrap();
        let dst = Profile::parse(&matrix_profile(b"XYZ ", [0.9642, 1.0, 0.8249])).unwrap();
        let t = Transform::new(&src, &dst, 1).unwrap();

        // encoded Lab (0.5, 128/255, 128/255) → raw Lab (50, 0, 0) → XYZ →
        // diagonal-inverse → equal linear RGB → ^(1/2.2)
        let out = t.eval(&[0.5, 128.0 / 255.0, 128.0 / 255.0]);
        let fy = (50.0f32 + 16.0) / 116.0;
        let expected = fy.powi(3).powf(1.0 / 2.2);
        assert!(
            (out[0] - out[1]).abs() < 1e-4 && (out[1] - out[2]).abs() < 1e-4,
            "neutral Lab → equal RGB: {out:?}"
        );
        for o in &out {
            assert!((o - expected).abs() < 5e-3, "{out:?} vs {expected}");
        }
    }

    #[test]
    fn singular_colorants_fail_at_assembly() {
        // Rank-deficient colorants (three identical columns) have no inverse:
        // the B2A assembly must report the malformed profile instead of
        // silently rendering garbage through a placeholder identity.
        let bytes = build_profile(
            b"mntr",
            b"RGB ",
            b"XYZ ",
            &[
                (b"rXYZ", xyz_tag(0.5, 0.5, 0.5)),
                (b"gXYZ", xyz_tag(0.5, 0.5, 0.5)),
                (b"bXYZ", xyz_tag(0.5, 0.5, 0.5)),
                (b"rTRC", curv_gamma_tag(2.2)),
                (b"gTRC", curv_gamma_tag(2.2)),
                (b"bTRC", curv_gamma_tag(2.2)),
            ],
        );
        let p = Profile::parse(&bytes).unwrap();
        assert!(matches!(
            Transform::new(&p, &p, 1),
            Err(crate::icc::IccError::WrongTagType)
        ));
    }

    /// The bare Lab colour-space profile darktable links colorin/colorout
    /// transforms through (LCMS's `cmsCreateLab4Profile` equivalent): device
    /// space == PCS, no transform tags.
    pub(crate) fn lab_identity_profile() -> Vec<u8> {
        build_profile(b"mntr", b"Lab ", b"Lab ", &[])
    }

    #[test]
    fn colourspace_conversion_profiles_transform_identity() {
        // Device space == PCS ⇒ both directions are empty pipelines, and a full
        // Transform between two such profiles passes raw values through.
        let p = Profile::parse(&lab_identity_profile()).unwrap();
        assert!(p.a2b_pipeline(1).unwrap().stages.is_empty());
        assert!(p.b2a_pipeline(1).unwrap().stages.is_empty());
        let t = Transform::new(&p, &p, 1).unwrap();
        for lab in [[50.0f32, 4.0, -6.0], [100.0, 0.0, 0.0], [0.0, -128.0, 127.0]] {
            assert_eq!(t.eval(&lab), lab, "Lab passthrough {lab:?}");
        }

        let xyz = Profile::parse(&build_profile(b"mntr", b"XYZ ", b"XYZ ", &[])).unwrap();
        assert!(xyz.b2a_pipeline(1).unwrap().stages.is_empty());
        let t = Transform::new(&xyz, &xyz, 1).unwrap();
        assert_eq!(
            t.eval(&[0.31, 0.51, 0.17]),
            vec![0.31, 0.51, 0.17],
            "XYZ passthrough"
        );
    }

    #[test]
    fn lab_conversion_profile_links_to_device_profiles() {
        // colorout's shape: raw Lab in through the bare Lab source profile,
        // sRGB-like device out. Neutral Lab must land on neutral device RGB.
        let src = Profile::parse(&lab_identity_profile()).unwrap();
        let dst = Profile::parse(&srgb_like_profile(b"XYZ ", [0.9642, 1.0, 0.8249])).unwrap();
        let t = Transform::new(&src, &dst, 1).unwrap();

        let white = t.eval(&[100.0, 0.0, 0.0]);
        for o in &white {
            assert!((*o - 1.0).abs() < 5e-3, "D50 white → device white: {white:?}");
        }
        let grey = t.eval(&[50.0, 0.0, 0.0]);
        assert!(
            (grey[0] - grey[1]).abs() < 1e-4 && (grey[1] - grey[2]).abs() < 1e-4,
            "neutral stays neutral: {grey:?}"
        );
    }

    #[test]
    fn non_three_channel_pipelines_are_rejected_at_assembly() {
        // A GRAY profile's A2B0 parses cleanly (1-in/1-out is within the
        // parser's channel caps) but violates the 3-channel contract that the
        // band path's stack arrays rest on: assembly must refuse it rather than
        // let the first pixel index out of bounds across the FFI boundary.
        let bytes = build_profile(b"scnr", b"GRAY", b"XYZ ", &[(b"A2B0", mft1_gray_lut())]);
        let p = Profile::parse(&bytes).expect("the GRAY fixture itself must parse");
        assert!(matches!(
            Transform::new(&p, &p, 1),
            Err(IccError::WrongTagType)
        ));

        // The same LUT as a *destination* (B2A side) is equally refused.
        let dst_bytes = build_profile(b"scnr", b"GRAY", b"Lab ", &[(b"B2A0", mft1_gray_lut())]);
        let src = Profile::parse(&matrix_profile(b"XYZ ", [0.9642, 1.0, 0.8249])).unwrap();
        let d = Profile::parse(&dst_bytes).unwrap();
        assert!(matches!(
            Transform::new(&src, &d, 1),
            Err(IccError::WrongTagType)
        ));
    }

    #[test]
    fn odd_device_space_signatures_are_rejected_even_with_full_shaper_tags() {
        // A device-`Lab ` profile whose PCS is `XYZ ` is neither RGB nor XYZ
        // nor a colour-space conversion (data_space ≠ pcs). Its matrix-shaper
        // tags are all individually valid — the pipelines would build and
        // evaluate to plausible garbage by reading them as if the device were
        // XYZ. Assembly must refuse it so the C caller falls back to LCMS
        // instead of grading through nonsense.
        let bytes = build_profile(
            b"mntr",
            b"Lab ",
            b"XYZ ",
            &[
                (b"wtpt", xyz_tag(0.9642, 1.0, 0.8249)),
                (b"rXYZ", xyz_tag(0.9642, 0.0, 0.0)),
                (b"gXYZ", xyz_tag(0.0, 1.0, 0.0)),
                (b"bXYZ", xyz_tag(0.0, 0.0, 0.8249)),
                (b"rTRC", curv_gamma_tag(2.2)),
                (b"gTRC", curv_gamma_tag(2.2)),
                (b"bTRC", curv_gamma_tag(2.2)),
            ],
        );
        let p = Profile::parse(&bytes).expect("the fixture itself must parse");
        assert!(matches!(
            Transform::new(&p, &p, 1),
            Err(IccError::WrongTagType)
        ));
    }

    #[test]
    fn v2_lab_encode_path_uses_the_legacy_scale() {
        // Same wiring as the v4 bridge test but with the header version flipped
        // to 2: the encode stage must select the legacy scale (L=100 lives at
        // 0xFF00, not 0xFFFF), so a neutral a/b encodes to 32768/65535 rather
        // than v4's 128/255. Pins that version_major actually reaches the stage.
        let dst_bytes = with_version(
            build_profile(
                b"mntr",
                b"RGB ",
                b"Lab ",
                &[(b"B2A0", mft1_identity_lut())],
            ),
            2,
        );
        let src = Profile::parse(&matrix_profile(b"XYZ ", [0.9642, 1.0, 0.8249])).unwrap();
        let dst = Profile::parse(&dst_bytes).unwrap();
        assert_eq!(dst.version_major, 2);
        let t = Transform::new(&src, &dst, 1).unwrap();

        let out = t.eval(&[0.5, 0.5, 0.5]);
        // NB: the synthetic TRC tag carries u16 gamma, quantising 2.2 to 563/256
        // — the tight 1e-4 tolerance below sees the difference, so expect it.
        let g = 563.0f32 / 256.0;
        let lin = 0.5f32.powf(g);
        let lab = lab_from_xyz(&[lin * 0.9642, lin, lin * 0.8249]);
        let v2 = 65535.0f32 / 65280.0;
        let expect_l = lab[0] / (100.0 * v2);
        let expect_ab = 128.0 / (255.0 * v2); // == 32768/65535 exactly
        assert!(
            (out[0] - expect_l).abs() < 1e-4,
            "out={out:?} expected v2 L_enc≈{expect_l}"
        );
        assert!((out[1] - expect_ab).abs() < 1e-4 && (out[2] - expect_ab).abs() < 1e-4);
        // and provably NOT the v4 encoding (which would read ≈0.50196 here)
        assert!(
            (out[1] - 128.0f32 / 255.0).abs() > 1e-3,
            "v2 profile must not use the v4 scale: {out:?}"
        );
    }

    #[test]
    fn b2a_tag_preference_follows_intent() {
        // B2A1 = identity tables, B2A0 = flat-zero output tables: intent 1 must
        // prefer B2A1 (non-neutral output), intent 0 must land on B2A0 (zeros).
        let dst_bytes = build_profile(
            b"mntr",
            b"RGB ",
            b"Lab ",
            &[
                (b"B2A0", mft1_lut(true)),
                (b"B2A1", mft1_identity_lut()),
            ],
        );
        let src = Profile::parse(&matrix_profile(b"XYZ ", [0.9642, 1.0, 0.8249])).unwrap();
        let dst = Profile::parse(&dst_bytes).unwrap();

        let via_b2a1 = Transform::new(&src, &dst, 1).unwrap().eval(&[0.5, 0.5, 0.5]);
        assert!(
            (via_b2a1[0] - 0.54).abs() < 5e-2 && via_b2a1[0] > 0.4,
            "intent 1 should read through B2A1: {via_b2a1:?}"
        );

        let via_b2a0 = Transform::new(&src, &dst, 0).unwrap().eval(&[0.5, 0.5, 0.5]);
        assert!(
            via_b2a0.iter().all(|v| v.abs() < 1e-4),
            "intent 0 should fall on the zeroed B2A0: {via_b2a0:?}"
        );
    }

    #[test]
    fn absolute_intent_scales_only_between_different_whites() {
        let d50 = [0.9642f32, 1.0, 0.8249];
        let odd = [0.8f32, 0.9, 0.7];
        // NOTE: the diagonal `matrix_profile` pair would make this test vacuous —
        // with axis-aligned primaries the white-ratio scale cancels exactly
        // against the inverse colorants (abs ≡ rel by construction). The
        // full-matrix colorants below make the scaling observable.
        let p_d50 = Profile::parse(&srgb_like_profile(b"XYZ ", d50)).unwrap();
        let p_odd = Profile::parse(&srgb_like_profile(b"XYZ ", odd)).unwrap();

        // identical whites ⇒ absolute ≡ relative
        let rel = Transform::new(&p_d50, &p_d50, 1).unwrap();
        let abs = Transform::new(&p_d50, &p_d50, 3).unwrap();
        let a = rel.eval(&[0.3, 0.6, 0.9]);
        let b = abs.eval(&[0.3, 0.6, 0.9]);
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-5, "{a:?} vs {b:?}");
        }

        // different whites ⇒ absolute diverges (scaled by dst/src ratio)
        let abs2 = Transform::new(&p_d50, &p_odd, 3).unwrap();
        let c = abs2.eval(&[0.3, 0.6, 0.9]);
        assert!(
            (c[0] - a[0]).abs() > 1e-3 || (c[1] - a[1]).abs() > 1e-3,
            "absolute must move the result: {c:?} vs {a:?}"
        );

        // missing wtpt degrades to relative instead of erroring
        let no_wtpt = build_profile(
            b"mntr",
            b"RGB ",
            b"XYZ ",
            &[
                (b"rXYZ", xyz_tag(d50[0], 0.0, 0.0)),
                (b"gXYZ", xyz_tag(0.0, d50[1], 0.0)),
                (b"bXYZ", xyz_tag(0.0, 0.0, d50[2])),
                (b"rTRC", curv_gamma_tag(2.2)),
                (b"gTRC", curv_gamma_tag(2.2)),
                (b"bTRC", curv_gamma_tag(2.2)),
            ],
        );
        let pnw = Profile::parse(&no_wtpt).unwrap();
        let degraded = Transform::new(&pnw, &pnw, 3).unwrap();
        let d = degraded.eval(&[0.3, 0.6, 0.9]);
        assert!(d.iter().all(|v| v.is_finite()));
    }
}
