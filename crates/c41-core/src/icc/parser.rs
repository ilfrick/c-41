//! ICC profile binary parser (ICC.1 v2/v4). All integers are big-endian.
//!
//! Parses the 128-byte header, the tag table, and the tag types needed for
//! matrix-shaper profiles: `XYZ ` (colorants / white point) and the tone-curve
//! types `curv` / `para`. cLUT tag types land in the next increment.

use super::lut::{parse_lut_tag, Pipeline, Stage};
use std::collections::HashMap;

/// ICC parse failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IccError {
    /// The buffer is shorter than the structure being read.
    Truncated,
    /// Not an ICC profile (`acsp` signature at offset 36 missing).
    BadSignature,
    /// A tag's data has an unexpected type signature for the requested read.
    WrongTagType,
}

impl std::fmt::Display for IccError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IccError::Truncated => write!(f, "ICC profile truncated"),
            IccError::BadSignature => write!(f, "not an ICC profile ('acsp' missing)"),
            IccError::WrongTagType => write!(f, "ICC tag has an unexpected type"),
        }
    }
}
impl std::error::Error for IccError {}

#[inline]
fn be_u16(b: &[u8], o: usize) -> Result<u16, IccError> {
    b.get(o..o + 2).map(|s| u16::from_be_bytes([s[0], s[1]])).ok_or(IccError::Truncated)
}
#[inline]
fn be_u32(b: &[u8], o: usize) -> Result<u32, IccError> {
    b.get(o..o + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]])).ok_or(IccError::Truncated)
}
/// s15Fixed16Number → f32 (signed 16.16 fixed point).
#[inline]
fn s15f16(b: &[u8], o: usize) -> Result<f32, IccError> {
    be_u32(b, o).map(|v| (v as i32) as f32 / 65536.0)
}
#[inline]
fn sig4(b: &[u8], o: usize) -> Result<[u8; 4], IccError> {
    b.get(o..o + 4).map(|s| [s[0], s[1], s[2], s[3]]).ok_or(IccError::Truncated)
}

/// A CIE XYZ triple (from an `XYZ ` tag).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyz {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A tone reproduction curve, forward (device value → linear), evaluated over the
/// normalised domain `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub enum Curve {
    /// `curv` with count 0 — the identity `y = x`.
    Identity,
    /// `curv` with count 1 — a pure gamma `y = x^g`.
    Gamma(f32),
    /// `curv` with count > 1 — a sampled LUT (u16 entries, 0..65535 ↦ 0..1),
    /// linearly interpolated.
    Table(Vec<u16>),
    /// `para` — a parametric curve (function type 0–4 + its parameters).
    Parametric { func: u16, params: Vec<f32> },
}

impl Curve {
    /// Evaluate the curve at `x` (clamped to `[0, 1]`).
    pub fn eval(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        match self {
            Curve::Identity => x,
            Curve::Gamma(g) => x.powf(*g),
            Curve::Table(t) => {
                if t.is_empty() {
                    return x;
                }
                if t.len() == 1 {
                    return t[0] as f32 / 65535.0;
                }
                let pos = x * (t.len() - 1) as f32;
                let i = pos.floor() as usize;
                if i >= t.len() - 1 {
                    return t[t.len() - 1] as f32 / 65535.0;
                }
                let f = pos - i as f32;
                let a = t[i] as f32;
                let b = t[i + 1] as f32;
                (a + (b - a) * f) / 65535.0
            }
            Curve::Parametric { func, params } => eval_parametric(*func, params, x),
        }
    }
}

impl Curve {
    /// The inverse relation: a curve whose `eval` undoes `Self::eval` over `[0,1]`
    /// (`inverse().eval(eval(x)) ≈ x`). Needed to run an output profile's TRC
    /// backwards in [`Profile::b2a_pipeline`]'s matrix-shaper fallback (raw PCS →
    /// device RGB).
    ///
    /// - `Gamma(g)` inverts analytically to `Gamma(1/g)`; a degenerate `g == 0`
    ///   (`x^0 ≡ 1`, no inverse) falls back to identity.
    /// - `Identity` inverts to itself.
    /// - `Table` and `Parametric` invert by **sampling + bisection** (construction
    ///   time only): entry `j` is the smallest `x` with `fwd(x) ≥ j/(N−1)` — the
    ///   same convention LCMS's tabular inversion lands on, and how LCMS turns
    ///   parametric curves into invertible tables. Bottom plateaus (the `para`
    ///   dead zone below `-b/a`/`d`) therefore map every target they cover onto 0.
    pub fn inverse(&self) -> Curve {
        // 4096 entries: worst-case roundtrip error ≈ half a quarter-bit LSB at
        // 8-bit depth — cheap insurance now that this path feeds production
        // transforms (construction cost is one pass of bisections).
        const N: usize = 4096;
        match self {
            Curve::Identity => Curve::Identity,
            Curve::Gamma(g) if *g != 0.0 => Curve::Gamma(1.0 / g),
            Curve::Gamma(_) => Curve::Identity,
            _ => {
                let (lo, hi) = (self.eval(0.0), self.eval(1.0));
                let mut inv = vec![0u16; N];
                for (j, slot) in inv.iter_mut().enumerate().skip(1) {
                    let t = j as f32 / (N - 1) as f32;
                    if lo >= t {
                        // a plateau already covering `t` at x=0 maps there
                        *slot = 0;
                    } else if hi < t {
                        // above the curve's maximum → clamp to the domain end
                        *slot = 65535;
                    } else {
                        // monotone ⇒ a unique crossing in [0,1]; bisect it
                        let (mut a, mut b) = (0.0f32, 1.0f32);
                        for _ in 0..24 {
                            let mid = 0.5 * (a + b);
                            if self.eval(mid) < t {
                                a = mid;
                            } else {
                                b = mid;
                            }
                        }
                        *slot = (0.5 * (a + b) * 65535.0).round() as u16;
                    }
                }
                Curve::Table(inv)
            }
        }
    }
}

/// Parametric curve evaluation (ICC parametricCurveType, funcs 0–4).
fn eval_parametric(func: u16, p: &[f32], x: f32) -> f32 {
    // helper for safe param access (missing → 0, defensive)
    let g = |i: usize| p.get(i).copied().unwrap_or(0.0);
    // clamp the power base at 0 (matches LCMS/darktable): a non-integer power of a
    // negative base is NaN, which would silently corrupt colour downstream.
    match func {
        0 => x.max(0.0).powf(g(0)), // Y = X^g
        1 => {
            // g, a, b : Y = (aX+b)^g for X >= -b/a, else 0
            let (gg, a, b) = (g(0), g(1), g(2));
            if a != 0.0 && x >= -b / a {
                (a * x + b).max(0.0).powf(gg)
            } else {
                0.0
            }
        }
        2 => {
            // g, a, b, c : Y = (aX+b)^g + c for X >= -b/a, else c
            let (gg, a, b, c) = (g(0), g(1), g(2), g(3));
            if a != 0.0 && x >= -b / a {
                (a * x + b).max(0.0).powf(gg) + c
            } else {
                c
            }
        }
        3 => {
            // g, a, b, c, d : Y = (aX+b)^g for X>=d, else cX
            let (gg, a, b, c, d) = (g(0), g(1), g(2), g(3), g(4));
            if x >= d {
                (a * x + b).max(0.0).powf(gg)
            } else {
                c * x
            }
        }
        4 => {
            // g, a, b, c, d, e, f : Y = (aX+b)^g + e for X>=d, else cX + f
            let (gg, a, b, c, d, e, ff) = (g(0), g(1), g(2), g(3), g(4), g(5), g(6));
            if x >= d {
                (a * x + b).max(0.0).powf(gg) + e
            } else {
                c * x + ff
            }
        }
        _ => x, // unknown → identity (defensive)
    }
}

/// A parsed ICC profile: the header fields we use + a tag directory into the
/// original bytes.
#[derive(Debug, Clone)]
pub struct Profile {
    /// Profile/device class signature (e.g. `mntr`, `scnr`, `prtr`).
    pub device_class: [u8; 4],
    /// Data colour space (`RGB `, `GRAY`, `CMYK`, `XYZ `, `Lab `).
    pub data_space: [u8; 4],
    /// Profile connection space (`XYZ ` or `Lab `).
    pub pcs: [u8; 4],
    /// ICC major version (2 or 4).
    pub version_major: u8,
    /// Header rendering intent (0 perceptual, 1 rel-colorimetric, 2 saturation,
    /// 3 abs-colorimetric).
    pub rendering_intent: u32,
    tags: HashMap<[u8; 4], (usize, usize)>, // sig -> (offset, size)
    data: Vec<u8>,
}

impl Profile {
    /// Parse an ICC profile from its raw bytes.
    pub fn parse(bytes: &[u8]) -> Result<Profile, IccError> {
        if bytes.len() < 132 {
            return Err(IccError::Truncated);
        }
        // 'acsp' profile-file signature at offset 36
        if &bytes[36..40] != b"acsp" {
            return Err(IccError::BadSignature);
        }
        let version_major = bytes[8]; // version: BCD, high byte = major
        let device_class = sig4(bytes, 12)?;
        let data_space = sig4(bytes, 16)?;
        let pcs = sig4(bytes, 20)?;
        // intent lives in the low 16 bits; high 16 are reserved (must be 0).
        let rendering_intent = be_u32(bytes, 64)? & 0xFFFF;

        // tag table at offset 128
        let count = be_u32(bytes, 128)? as usize;
        // Validate the declared tag count against the bytes actually available
        // BEFORE reserving, so a hostile 32-bit count can't force a huge alloc
        // (these are untrusted profile bytes, embedded in user-opened images).
        if count > (bytes.len() - 132) / 12 {
            return Err(IccError::Truncated);
        }
        let mut tags = HashMap::with_capacity(count);
        for i in 0..count {
            let base = 132 + i * 12;
            let sig = sig4(bytes, base)?;
            let off = be_u32(bytes, base + 4)? as usize;
            let size = be_u32(bytes, base + 8)? as usize;
            // validate the tag's data range now so tag() can be infallible-ish
            if off.checked_add(size).map_or(true, |end| end > bytes.len()) {
                return Err(IccError::Truncated);
            }
            tags.insert(sig, (off, size));
        }

        Ok(Profile {
            device_class,
            data_space,
            pcs,
            version_major,
            rendering_intent,
            tags,
            data: bytes.to_vec(),
        })
    }

    /// Raw bytes of a tag (its type signature is the first 4 bytes), if present.
    pub fn tag(&self, sig: &[u8; 4]) -> Option<&[u8]> {
        let &(off, size) = self.tags.get(sig)?;
        self.data.get(off..off + size)
    }

    /// Read an `XYZType` tag (used for `rXYZ`/`gXYZ`/`bXYZ`/`wtpt`). Returns the
    /// first XYZ triple.
    pub fn read_xyz(&self, sig: &[u8; 4]) -> Result<Xyz, IccError> {
        let d = self.tag(sig).ok_or(IccError::Truncated)?;
        if sig4(d, 0)? != *b"XYZ " {
            return Err(IccError::WrongTagType);
        }
        // 8-byte header (type + reserved), then s15Fixed16 x,y,z
        Ok(Xyz {
            x: s15f16(d, 8)?,
            y: s15f16(d, 12)?,
            z: s15f16(d, 16)?,
        })
    }

    /// Read a tone-curve tag (`curv` or `para`).
    pub fn read_curve(&self, sig: &[u8; 4]) -> Result<Curve, IccError> {
        let d = self.tag(sig).ok_or(IccError::Truncated)?;
        Ok(parse_curve(d)?.0)
    }

    /// The RGB → XYZ (PCS) matrix from the `rXYZ`/`gXYZ`/`bXYZ` colorant tags,
    /// as `m[row=XYZ][col=RGB]` (so `XYZ = m · RGB`). `None` if the profile is not
    /// a matrix profile (no colorant tags).
    pub fn rgb_to_xyz_matrix(&self) -> Option<[[f32; 3]; 3]> {
        let r = self.read_xyz(b"rXYZ").ok()?;
        let g = self.read_xyz(b"gXYZ").ok()?;
        let b = self.read_xyz(b"bXYZ").ok()?;
        Some([
            [r.x, g.x, b.x],
            [r.y, g.y, b.y],
            [r.z, g.z, b.z],
        ])
    }

    /// The three RGB tone curves (`rTRC`/`gTRC`/`bTRC`), if all present.
    pub fn rgb_trc(&self) -> Option<[Curve; 3]> {
        Some([
            self.read_curve(b"rTRC").ok()?,
            self.read_curve(b"gTRC").ok()?,
            self.read_curve(b"bTRC").ok()?,
        ])
    }

    /// True iff the profile connection space is Lab (`Lab `); otherwise XYZ.
    pub fn pcs_is_lab(&self) -> bool {
        &self.pcs == b"Lab "
    }

    /// The device→PCS transform for a rendering `intent` (0 perceptual, 1
    /// rel-colorimetric, 2 saturation, 3 abs-colorimetric). Prefers the
    /// `A2B{intent}` LUT tag, falls back to `A2B0`, then to the matrix-shaper
    /// (RGB TRC curves → colorant matrix → XYZ). The output colour space is
    /// [`Self::pcs`].
    ///
    /// Both branches return the PCS in **raw (un-encoded) values, uniformly**: raw
    /// XYZ D50 (`Y ≈ 1` for white) when [`Self::pcs`] is `XYZ `, or Lab
    /// (`L∈[0,100]`, `a,b∈[-128,127]`) when it is `Lab `. The A2B (LUT) branch
    /// emits ICC-encoded PCS which is decoded to raw here (appended
    /// [`pcs_decode_stage`]); the matrix-shaper branch is already raw XYZ D50 (the
    /// colorant tags are D50-adapted, so no CAT). The caller uses [`Self::pcs_is_lab`]
    /// only to know the *space*, not the encoding.
    pub fn a2b_pipeline(&self, intent: u32) -> Result<Pipeline, IccError> {
        for sig in a2b_tag_sigs(intent) {
            if let Some(tag) = self.tag(&sig) {
                let mut p = parse_lut_tag(tag)?;
                // decode ICC-encoded PCS → raw values (uniform with the shaper branch)
                p.stages.push(pcs_decode_stage(self.pcs_is_lab(), self.version_major));
                return Ok(p);
            }
        }
        // matrix-shaper fallback (RGB matrix profiles → raw XYZ D50). The shaper
        // path is XYZ-only; a Lab-PCS profile with only colorants is malformed.
        if self.pcs_is_lab() {
            return Err(IccError::WrongTagType);
        }
        let m = self.rgb_to_xyz_matrix().ok_or(IccError::WrongTagType)?;
        let trc = self.rgb_trc().ok_or(IccError::WrongTagType)?;
        Ok(Pipeline {
            stages: vec![
                Stage::Curves(trc.to_vec()),
                Stage::Matrix([
                    [m[0][0], m[0][1], m[0][2], 0.0],
                    [m[1][0], m[1][1], m[1][2], 0.0],
                    [m[2][0], m[2][1], m[2][2], 0.0],
                ]),
            ],
        })
    }
}

/// The affine PCS-decode stage that maps ICC-encoded LUT output (`[0,1]` per
/// channel) to raw PCS values, so the A2B branch matches the raw matrix-shaper
/// branch. XYZ: `×(65535/32768)` (the 1.0↔0x8000 convention). Lab: `L=n·100`,
/// `a,b=n·255−128`, with the legacy v2 scale `65535/65280` on L/a/b.
fn pcs_decode_stage(is_lab: bool, version_major: u8) -> Stage {
    if is_lab {
        // v2 Lab encodes 100 at 0xFF00 (=65280), not 0xFFFF; v4 at 0xFFFF. The
        // legacy scale `255·(65535/65280)` is *exactly* `65535/256` (255·256=65280),
        // which is why it reduces to LCMS's `E/256 − 128`; don't "simplify" it away.
        let v2 = 65535.0 / 65280.0;
        let (s_l, s_ab) = if version_major >= 4 {
            (100.0, 255.0)
        } else {
            (100.0 * v2, 255.0 * v2)
        };
        Stage::Matrix([
            [s_l, 0.0, 0.0, 0.0],
            [0.0, s_ab, 0.0, -128.0],
            [0.0, 0.0, s_ab, -128.0],
        ])
    } else {
        let k = 65535.0 / 32768.0; // ≈ 1.99997
        Stage::Matrix([[k, 0.0, 0.0, 0.0], [0.0, k, 0.0, 0.0], [0.0, 0.0, k, 0.0]])
    }
}

/// The `A2B{intent}` tag signatures to try, in preference order (abs-colorimetric
/// intent 3 uses the rel-colorimetric `A2B1` table; all fall back to `A2B0`).
fn a2b_tag_sigs(intent: u32) -> Vec<[u8; 4]> {
    let specific = match intent {
        1 | 3 => *b"A2B1",
        2 => *b"A2B2",
        _ => *b"A2B0",
    };
    if specific == *b"A2B0" {
        vec![*b"A2B0"]
    } else {
        vec![specific, *b"A2B0"]
    }
}

impl Profile {
    /// The PCS→device transform for a rendering `intent` — the output-profile
    /// direction ([`Self::a2b_pipeline`] is the input direction). Prefers the
    /// `B2A{intent}` LUT tag (falling back to `B2A0`, abs intent 3 → `B2A1`),
    /// then to the matrix-shaper fallback: invert the colorant matrix and run
    /// each TRC backwards ([`Curve::inverse`]).
    ///
    /// The returned pipeline expects **raw** PCS on input (uniform with
    /// [`Self::a2b_pipeline`]'s output) and prepends an ICC-*encode* stage so the
    /// tag's own tables — which consume encoded `[0,1]` values — see what they
    /// expect.
    pub fn b2a_pipeline(&self, intent: u32) -> Result<Pipeline, IccError> {
        for sig in b2a_tag_sigs(intent) {
            if let Some(tag) = self.tag(&sig) {
                let mut p = parse_lut_tag(tag)?;
                p.stages.insert(0, pcs_encode_stage(self.pcs_is_lab(), self.version_major));
                return Ok(p);
            }
        }
        // matrix-shaper fallback. As with A2B, the shaper path is XYZ-only.
        if self.pcs_is_lab() {
            return Err(IccError::WrongTagType);
        }
        let m = self.rgb_to_xyz_matrix().ok_or(IccError::WrongTagType)?;
        let trc = self.rgb_trc().ok_or(IccError::WrongTagType)?;
        // Singular colorants (malformed profile) fail loudly at assembly rather
        // than silently rendering garbage — this runs once per transform, so the
        // error costs nothing.
        let minv = invert3(&m).ok_or(IccError::WrongTagType)?;
        // NOTE: no encode stage here — the colorant matrix consumes the same RAW
        // XYZ D50 the A2B shaper branch produces; encoding is only for LUT tables.
        Ok(Pipeline {
            stages: vec![
                Stage::Matrix([
                    [minv[0][0], minv[0][1], minv[0][2], 0.0],
                    [minv[1][0], minv[1][1], minv[1][2], 0.0],
                    [minv[2][0], minv[2][1], minv[2][2], 0.0],
                ]),
                Stage::Curves(trc.iter().map(|c| c.inverse()).collect()),
            ],
        })
    }
}

/// The inverse of the A2B branch's appended decode stage: raw PCS → the ICC-encoded
/// `[0,1]` values B2A tables are defined over. Exact algebraic inverse of each
/// decode line (`raw = n·s + o` ⇔ `n = raw/s − o/s`).
fn pcs_encode_stage(is_lab: bool, version_major: u8) -> Stage {
    if is_lab {
        if version_major >= 4 {
            Stage::Matrix([
                [1.0 / 100.0, 0.0, 0.0, 0.0],
                [0.0, 1.0 / 255.0, 0.0, 128.0 / 255.0],
                [0.0, 0.0, 1.0 / 255.0, 128.0 / 255.0],
            ])
        } else {
            // legacy v2 Lab: decode used s = 255·(65535/65280) etc.
            let v2 = 65535.0 / 65280.0;
            Stage::Matrix([
                [1.0 / (100.0 * v2), 0.0, 0.0, 0.0],
                [0.0, 1.0 / (255.0 * v2), 0.0, 128.0 / (255.0 * v2)],
                [0.0, 0.0, 1.0 / (255.0 * v2), 128.0 / (255.0 * v2)],
            ])
        }
    } else {
        let k = 32768.0 / 65535.0; // reciprocal of the decode scale
        Stage::Matrix([[k, 0.0, 0.0, 0.0], [0.0, k, 0.0, 0.0], [0.0, 0.0, k, 0.0]])
    }
}

/// The `B2A{intent}` tag signatures to try, in preference order (mirrors
/// [`a2b_tag_sigs`]).
fn b2a_tag_sigs(intent: u32) -> Vec<[u8; 4]> {
    let specific = match intent {
        1 | 3 => *b"B2A1",
        2 => *b"B2A2",
        _ => *b"B2A0",
    };
    if specific == *b"B2A0" {
        vec![*b"B2A0"]
    } else {
        vec![specific, *b"B2A0"]
    }
}

/// Invert a 3×3 matrix via the adjugate (`None` when singular or non-finite).
/// Used for the matrix-shaper output fallback — a singular colorant set means a
/// malformed profile, which the caller reports rather than rendering garbage.
fn invert3(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}
/// the curve and the number of bytes it occupies (unpadded). Shared by tag reads
/// and the v4 `mAB `/`mBA ` inline-curve parsing.
pub(super) fn parse_curve(d: &[u8]) -> Result<(Curve, usize), IccError> {
    let ty = sig4(d, 0)?;
    match &ty {
        b"curv" => {
            let n = be_u32(d, 8)? as usize;
            // curveType length = 12-byte header + `n` u16 entries (unpadded).
            let consumed = 12 + n * 2; // usize is 64-bit; n ≤ u32::MAX so no overflow
            let curve = match n {
                0 => Curve::Identity,
                1 => Curve::Gamma(be_u16(d, 12)? as f32 / 256.0),
                _ => {
                    // validate-before-reserve (untrusted length).
                    if (d.len().saturating_sub(12)) / 2 < n {
                        return Err(IccError::Truncated);
                    }
                    let mut t = Vec::with_capacity(n);
                    for i in 0..n {
                        t.push(be_u16(d, 12 + i * 2)?);
                    }
                    Curve::Table(t)
                }
            };
            Ok((curve, consumed))
        }
        b"para" => {
            let func = be_u16(d, 8)?;
            // ICC defines only funcs 0–4; reject unknown types (an unknown func in
            // a v4 curve set would otherwise misalign the following curves).
            let nparams = match func {
                0 => 1,
                1 => 3,
                2 => 4,
                3 => 5,
                4 => 7,
                _ => return Err(IccError::WrongTagType),
            };
            let mut params = Vec::with_capacity(nparams);
            for i in 0..nparams {
                params.push(s15f16(d, 12 + i * 4)?);
            }
            // parametricCurveType length = 12-byte header + params (unpadded).
            Ok((Curve::Parametric { func, params }, 12 + nparams * 4))
        }
        _ => Err(IccError::WrongTagType),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal but valid ICC profile: 128-byte header + a tag table with
    /// the given `(sig, data)` tags laid out after it.
    fn build_profile(class: &[u8; 4], data_space: &[u8; 4], pcs: &[u8; 4], tags: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let header_len = 128;
        let table_len = 4 + tags.len() * 12;
        let mut data_off = header_len + table_len;
        // pad each tag to a 4-byte boundary
        let mut offsets = Vec::new();
        for (_, d) in tags {
            offsets.push(data_off);
            data_off += (d.len() + 3) & !3;
        }
        let total = data_off;
        let mut b = vec![0u8; total];
        // header
        b[0..4].copy_from_slice(&(total as u32).to_be_bytes());
        b[8] = 4; // version major
        b[12..16].copy_from_slice(class);
        b[16..20].copy_from_slice(data_space);
        b[20..24].copy_from_slice(pcs);
        b[36..40].copy_from_slice(b"acsp");
        b[64..68].copy_from_slice(&1u32.to_be_bytes()); // rel colorimetric
        // tag table
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

    fn xyz_tag(x: f32, y: f32, z: f32) -> Vec<u8> {
        let mut d = vec![0u8; 20];
        d[0..4].copy_from_slice(b"XYZ ");
        for (i, v) in [x, y, z].iter().enumerate() {
            let fixed = (*v * 65536.0).round() as i32 as u32;
            d[8 + i * 4..12 + i * 4].copy_from_slice(&fixed.to_be_bytes());
        }
        d
    }

    fn curv_gamma_tag(g: f32) -> Vec<u8> {
        let mut d = vec![0u8; 14];
        d[0..4].copy_from_slice(b"curv");
        d[8..12].copy_from_slice(&1u32.to_be_bytes()); // count = 1
        d[12..14].copy_from_slice(&(((g * 256.0).round()) as u16).to_be_bytes());
        d
    }

    #[test]
    fn parses_header_and_rejects_non_icc() {
        let p = build_profile(b"mntr", b"RGB ", b"XYZ ", &[]);
        let prof = Profile::parse(&p).unwrap();
        assert_eq!(&prof.device_class, b"mntr");
        assert_eq!(&prof.data_space, b"RGB ");
        assert_eq!(&prof.pcs, b"XYZ ");
        assert_eq!(prof.version_major, 4);
        assert_eq!(prof.rendering_intent, 1);

        let mut bad = p.clone();
        bad[36] = b'x';
        assert_eq!(Profile::parse(&bad).unwrap_err(), IccError::BadSignature);
        assert_eq!(Profile::parse(&p[..100]).unwrap_err(), IccError::Truncated);
    }

    #[test]
    fn reads_colorant_matrix() {
        // sRGB-ish D65 colorants (rounded)
        let tags = vec![
            (b"rXYZ", xyz_tag(0.4361, 0.2225, 0.0139)),
            (b"gXYZ", xyz_tag(0.3851, 0.7169, 0.0971)),
            (b"bXYZ", xyz_tag(0.1431, 0.0606, 0.7141)),
        ];
        let tref: Vec<(&[u8; 4], Vec<u8>)> = tags.iter().map(|(s, d)| (*s, d.clone())).collect();
        let prof = Profile::parse(&build_profile(b"mntr", b"RGB ", b"XYZ ", &tref)).unwrap();
        let m = prof.rgb_to_xyz_matrix().unwrap();
        // m[XYZ][RGB]: column 1 (green) Y ≈ 0.7169
        assert!((m[1][1] - 0.7169).abs() < 1e-3, "{m:?}");
        assert!((m[0][0] - 0.4361).abs() < 1e-3);
        assert!((m[2][2] - 0.7141).abs() < 1e-3);
        // white (RGB 1,1,1) → summed colorants ≈ D50/D65 white Y ~1.0
        let wy = m[1][0] + m[1][1] + m[1][2];
        assert!((wy - 1.0).abs() < 5e-3, "white Y = {wy}");
    }

    /// Minimal identity `mft2` (3→3, grid `g`, ramp tables + identity CLUT).
    fn build_identity_mft2(g: usize) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"mft2");
        d.extend_from_slice(&[0, 0, 0, 0]);
        d.push(3);
        d.push(3);
        d.push(g as u8);
        d.push(0);
        for i in 0..3 {
            for j in 0..3 {
                let v: i32 = if i == j { 65536 } else { 0 };
                d.extend_from_slice(&v.to_be_bytes());
            }
        }
        d.extend_from_slice(&2u16.to_be_bytes()); // n_in
        d.extend_from_slice(&2u16.to_be_bytes()); // n_out
        for _ in 0..3 {
            d.extend_from_slice(&0u16.to_be_bytes());
            d.extend_from_slice(&65535u16.to_be_bytes());
        }
        for i in 0..g {
            for j in 0..g {
                for k in 0..g {
                    d.extend_from_slice(&(((i * 65535) / (g - 1)) as u16).to_be_bytes());
                    d.extend_from_slice(&(((j * 65535) / (g - 1)) as u16).to_be_bytes());
                    d.extend_from_slice(&(((k * 65535) / (g - 1)) as u16).to_be_bytes());
                }
            }
        }
        for _ in 0..3 {
            d.extend_from_slice(&0u16.to_be_bytes());
            d.extend_from_slice(&65535u16.to_be_bytes());
        }
        d
    }

    #[test]
    fn a2b_lut_branch_decodes_pcs_to_raw_xyz() {
        // Profile with A2B0 = identity mft2 (XYZ PCS): a2b_pipeline parses the LUT
        // and appends the XYZ decode, so a normalised 0.5 → raw ~1.0 (×65535/32768).
        let tags: Vec<(&[u8; 4], Vec<u8>)> = vec![(b"A2B0", build_identity_mft2(2))];
        let prof = Profile::parse(&build_profile(b"prtr", b"RGB ", b"XYZ ", &tags)).unwrap();
        let p = prof.a2b_pipeline(0).unwrap();
        let k = 65535.0f32 / 32768.0;
        let out = p.eval(&[0.5, 0.5, 0.5]);
        for c in 0..3 {
            assert!((out[c] - 0.5 * k).abs() < 5e-3, "decoded {out:?}, want ~{}", 0.5 * k);
        }
        let w = p.eval(&[1.0, 1.0, 1.0]);
        assert!((w[1] - k).abs() < 5e-3, "white {w:?} want ~{k}");
    }

    #[test]
    fn a2b_pipeline_matrix_shaper_produces_xyz() {
        // A matrix-shaper RGB→XYZ profile: gamma-2.2 TRCs + sRGB-ish colorants.
        // a2b_pipeline should linearise then apply the colorant matrix, so white
        // (1,1,1) → the summed colorants (≈ D50/D65 white, Y≈1).
        let tags = vec![
            (b"rXYZ", xyz_tag(0.4361, 0.2225, 0.0139)),
            (b"gXYZ", xyz_tag(0.3851, 0.7169, 0.0971)),
            (b"bXYZ", xyz_tag(0.1431, 0.0606, 0.7141)),
            (b"rTRC", curv_gamma_tag(2.2)),
            (b"gTRC", curv_gamma_tag(2.2)),
            (b"bTRC", curv_gamma_tag(2.2)),
        ];
        let tref: Vec<(&[u8; 4], Vec<u8>)> = tags.iter().map(|(s, d)| (*s, d.clone())).collect();
        let prof = Profile::parse(&build_profile(b"mntr", b"RGB ", b"XYZ ", &tref)).unwrap();
        assert!(!prof.pcs_is_lab());
        let p = prof.a2b_pipeline(0).unwrap();
        assert_eq!(p.stages.len(), 2); // TRC curves + colorant matrix

        // white → Y ≈ 1 (TRC(1)=1, then summed colorant Y)
        let w = p.eval(&[1.0, 1.0, 1.0]);
        assert!((w[1] - (0.2225 + 0.7169 + 0.0606)).abs() < 2e-3, "white XYZ = {w:?}");
        // a mid-grey linearises through the gamma before the matrix
        let g = p.eval(&[0.5, 0.5, 0.5]);
        let lin = 0.5f32.powf(2.1992); // u8Fixed8-quantised 2.2
        assert!((g[1] / w[1] - lin).abs() < 2e-3, "grey Y ratio {} vs {lin}", g[1] / w[1]);
    }

    #[test]
    fn evaluates_gamma_curve() {
        let tags: Vec<(&[u8; 4], Vec<u8>)> = vec![(b"rTRC", curv_gamma_tag(2.2))];
        let prof = Profile::parse(&build_profile(b"mntr", b"RGB ", b"XYZ ", &tags)).unwrap();
        let c = prof.read_curve(b"rTRC").unwrap();
        // u8Fixed8 quantises 2.2 → 563/256 = 2.1992; compare eval to the parsed g.
        let Curve::Gamma(g) = c else { panic!("expected gamma, got {c:?}") };
        assert!((g - 2.2).abs() < 1e-2, "g={g}");
        assert!((c.eval(0.5) - 0.5f32.powf(g)).abs() < 1e-5);
        assert_eq!(c.eval(0.0), 0.0);
        assert!((c.eval(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn identity_and_table_curves() {
        assert_eq!(Curve::Identity.eval(0.37), 0.37);
        // a 3-entry ramp table 0, 32768, 65535 → ~identity at endpoints & midpoint
        let t = Curve::Table(vec![0, 32768, 65535]);
        assert!((t.eval(0.0) - 0.0).abs() < 1e-4);
        assert!((t.eval(1.0) - 1.0).abs() < 1e-4);
        assert!((t.eval(0.5) - 0.5).abs() < 1e-2);
    }

    #[test]
    fn rejects_oversized_counts_without_ooming() {
        // absurd tag count must be rejected before any huge allocation.
        let mut p = build_profile(b"mntr", b"RGB ", b"XYZ ", &[]);
        p[128..132].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        assert_eq!(Profile::parse(&p).unwrap_err(), IccError::Truncated);

        // a curv tag declaring more entries than it can hold is likewise rejected.
        let mut bad_curv = vec![0u8; 16];
        bad_curv[0..4].copy_from_slice(b"curv");
        bad_curv[8..12].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes()); // count
        let tags: Vec<(&[u8; 4], Vec<u8>)> = vec![(b"rTRC", bad_curv)];
        let prof = Profile::parse(&build_profile(b"mntr", b"RGB ", b"XYZ ", &tags)).unwrap();
        assert_eq!(prof.read_curve(b"rTRC").unwrap_err(), IccError::Truncated);
    }

    #[test]
    fn parametric_negative_base_is_not_nan() {
        // func 1 with a negative base in the power segment must clamp, not NaN.
        let y = eval_parametric(1, &[2.4, -1.0, 0.2], 0.9); // a·x+b = -0.7 < 0
        assert!(y.is_finite(), "got {y}");
    }

    #[test]
    fn parametric_curve_type0_is_gamma() {
        let x = 0.6f32;
        assert!((eval_parametric(0, &[2.4], x) - x.powf(2.4)).abs() < 1e-6);
        // type 3 sRGB-like: linear below d, power above
        // g=2.4,a=1/1.055,b=0.055/1.055,c=1/12.92,d=0.04045
        let p = [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045];
        assert!((eval_parametric(3, &p, 0.02) - 0.02 / 12.92).abs() < 1e-6); // linear seg
        assert!(eval_parametric(3, &p, 0.5) > 0.0 && eval_parametric(3, &p, 0.5) < 0.5); // power seg
    }
}
