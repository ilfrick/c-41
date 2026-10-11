//! The sRGB profile that anchors the whole pipeline, and the loops that carry
//! a decoded image from its embedded device profile into the pipeline's
//! working space (G2a2, input profiling). G2b adds the same profile's **output**
//! role: every exported file is tagged with it, because a decoded-and-rendered
//! export is sRGB by construction.
//!
//! `c41-core::icc` has had a complete engine since m4-89…m4-129 and an
//! extractor since m4-245, but until now **nothing in the product called it**:
//! a JPEG tagged Adobe RGB was shown and written as though it were sRGB. This
//! module is the seam that closes that — and it is deliberately the only seam,
//! because the two decode paths (gdk-pixbuf for the preview, the `image` crate
//! for export) disagree about everything else and share nothing but this.
//!
//! # Why the destination is sRGB and not a preference
//!
//! Non-raw decode ends in an 8-bit buffer that [`crate::preview::
//! apply_pipeline_gear`] linearises with [`crate::color::srgb_to_linear`] and
//! re-encodes with [`crate::color::linear_to_srgb`]. The destination profile
//! therefore has to be IEC 61966-2-1 *exactly*: a gamma-2.2 stand-in would be a
//! second, different opinion about how to linearise, and the two would disagree
//! in the shadows on every frame. [`srgb_profile`] is built to the same curve
//! the pipeline already uses, and `srgb_trc_matches_the_pipelines_lineariser`
//! pins the two together.
//!
//! # What is deliberately not here
//!
//! **Raw.** [`crate::rawimage`] already runs its own input profile — the camera
//! → Rec.2020 matrix that darktable's `colorin` synthesises *as* an ICC profile
//! for raw (`src/iop/colorin.c:1379-1385`, feeding the same engine through
//! `_colorin_profile_bytes`). A sensor mosaic has no embedded ICC to lift out,
//! and applying one on top of that matrix would colour-manage twice. See
//! [`super::embed`]'s module docs, which scope raw out by design.

use std::sync::OnceLock;

use super::embed::Embedded;
use super::parser::Profile;
use super::transform::Transform;
use super::IccError;

/// Rendering intent used for input profiling: **perceptual** (`0`).
///
/// Matches darktable's `$DEFAULT: DT_INTENT_PERCEPTUAL` on colorin's intent
/// parameter (`src/iop/colorin.c:76`) and ICC's own numbering of it. For a
/// matrix-shaper source this is numerically identical to relative
/// colorimetric — [`Profile::a2b_pipeline`] falls back to the
/// rel-colorimetric table when a profile ships no perceptual one — so only a
/// cLUT profile's dedicated perceptual table can tell the two apart.
pub const INPUT_INTENT: u32 = 0;

/// The PCS white every colorant below is adapted to, exactly as the reference
/// sRGB profile writes it (colord's `sRGB.icc`): `63190 / 65536`, `1.0`,
/// `54061 / 65536`.
const D50: [f32; 3] = [0.964203, 1.0, 0.824905];

/// sRGB primaries as D50-adapted XYZ colorants, read off the same reference
/// profile rather than recomputed: the three columns sum to [`D50`] to within
/// one `s15Fixed16` quantum (checked by `white_is_the_sum_of_the_colorants`), so
/// device white reaches the PCS as white without a fudge factor.
const RXYZ: [f32; 3] = [0.435852, 0.222382, 0.013916];
const GXYZ: [f32; 3] = [0.385330, 0.717041, 0.097137];
const BXYZ: [f32; 3] = [0.143021, 0.060593, 0.713837];

/// The `chad` (chromatic adaptation) matrix the reference profile stores — the
/// standard D65→D50 adaptation, kept as the raw `s15Fixed16` integers so the
/// embedded bytes match colord's `sRGB.icc` exactly (G2b). The engine never
/// reads this tag; it is data a conformant third-party reader may honour.
const CHAD: [i32; 9] = [68682, 1507, -3286, 1947, 64903, -1118, -605, 984, 49300];

/// Adobe RGB (1998) primaries as D50-adapted XYZ colorants — column `i` is the
/// XYZ of device primary `i`, and the columns sum to [`D50`] (the trap the
/// fixtures documented it to avoid: spec D65 constants must *not* be used
/// here). This is the profile [`InputAssumption::AdobeRgb1998`] encodes for
/// untagged containers, so assuming Adobe RGB treats a file exactly like the
/// same pixels tagged with a real profile.
pub(crate) const ADOBE_RGB1998: [[f32; 3]; 3] = [
    [0.609634, 0.311035, 0.019470],
    [0.205399, 0.625763, 0.060898],
    [0.149170, 0.063187, 0.744522],
];

/// Adobe RGB (1998)'s display gamma as the `u8Fixed8`-style power `563/256` —
/// the same exponent the real profile's `para` type-0 curves carry.
pub(crate) const ADOBE_RGB1998_GAMMA: f32 = 563.0 / 256.0;

/// One `s15Fixed16` quantum in normalized units. Tests compare against it; the
/// production engine has no use for the constant itself.
#[cfg(test)]
const QUANTUM: f32 = 1.0 / 65536.0;

/// Tolerance for the "already sRGB" gate, in normalized units: **one 16-bit
/// LSB**, the finest precision any export path writes. A profile that agrees
/// with sRGB to within this cannot be made more accurate by running it, and
/// running it can only add rounding noise to a file that was already right —
/// which is why the gate exists at all: without it every ordinary sRGB JPEG
/// would take a matrix round trip on the way to the screen and come back a
/// fraction of a step off, for nothing.
const IDENTITY_TOL: f32 = 1.0 / 65535.0;

/// Probe colours for [`is_srgb`]'s gate: neutrals across the range, the three
/// primaries, the three secondaries, and one off-axis colour. A profile whose
/// transform maps all of these onto themselves is sRGB where it counts.
///
/// This is a gate, not a proof — a cLUT profile could in principle agree on
/// twelve points and disagree elsewhere. What it has to get right is the
/// direction: a genuine sRGB profile passes (so real photographs keep their
/// exact bytes), and anything that would visibly move a pixel — a gamma-2.2
/// "sRGB" stand-in, Adobe RGB, a wide-gamut profile — fails and gets transformed.
const PROBES: [[f32; 3]; 13] = [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.25, 0.25, 0.25],
    [0.5, 0.5, 0.5],
    [0.75, 0.75, 0.75],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.8, 0.3, 0.1],
    [0.1, 0.7, 0.4],
];

// ── profile construction ──────────────────────────────────────────────────
//
// The tag builders below mirror the test kit that `transform.rs` keeps under
// `#[cfg(test)]`. They are a separate copy rather than a shared one on purpose:
// that module's helpers are compiled out of every shipped binary, so making the
// product depend on them would mean either dragging test code into the release
// or repeatedly editing a file that has already been corrupted twice by exactly
// that move. `build_profile`/`xyz_tag` are `pub(crate)` so in-crate tests can
// build deliberately *non*-sRGB profiles to exercise the transform.

/// Assemble an ICC profile: 128-byte header, tag table, then each tag's
/// payload padded to the 4-byte boundary the spec requires. The assembler
/// writes exactly the tags it is given and nothing else.
///
/// The callers deliberately pass minimal tag sets where a profile stays
/// engine-internal — the Adobe assumption, the transform fixtures — because
/// [`Profile::parse`] reads only the tags it is asked for, and those bytes
/// never face a third-party reader. The one exception is [`srgb_profile`],
/// which adds the informational tags the ICC spec requires on a display
/// profile, because its bytes are shipped inside every exported file (G2b)
/// to arbitrary readers on other machines.
pub(crate) fn build_profile(
    class: &[u8; 4],
    data_space: &[u8; 4],
    pcs: &[u8; 4],
    tags: &[(&[u8; 4], Vec<u8>)],
) -> Vec<u8> {
    let header_len = 128;
    let table_len = 4 + tags.len() * 12;
    let mut data_off = header_len + table_len;
    let mut offsets = Vec::with_capacity(tags.len());
    for (_, d) in tags {
        offsets.push(data_off);
        data_off += (d.len() + 3) & !3;
    }
    let mut b = vec![0u8; data_off];
    b[0..4].copy_from_slice(&(data_off as u32).to_be_bytes());
    // v4.3.0 (BCD) — the published ICC.1:2010 revision. `Profile::parse` checks
    // only the major byte, so this is a statement of intent rather than a gate.
    b[8..12].copy_from_slice(&[0x04, 0x30, 0x00, 0x00]);
    b[12..16].copy_from_slice(class);
    b[16..20].copy_from_slice(data_space);
    b[20..24].copy_from_slice(pcs);
    // Creation date/time — mandatory (ICC.1:2010 §7.2.4). A *fixed* stamp keeps
    // `srgb_profile()` deterministic: the profile is a compile-time constant, so
    // two builds — and two calls in one process — must produce identical bytes.
    // Leaving it zero would encode an impossible year-0 date.
    for (i, v) in [2026u16, 1, 1, 0, 0, 0].iter().enumerate() {
        b[24 + i * 2..26 + i * 2].copy_from_slice(&v.to_be_bytes());
    }
    b[36..40].copy_from_slice(b"acsp");
    // PCS illuminant: ICC.1:2010 §7.2.16 mandates D50 in this field on every
    // profile *regardless of the actual media white*, and the reference sRGB
    // profile writes exactly these bytes. Strict third-party readers (libpng,
    // ImageMagick) warn when it is zero, and every exported file now carries
    // this header, so it has to be right.
    for (i, v) in D50.iter().enumerate() {
        let fixed = (v * 65536.0).round() as i32 as u32;
        b[68 + i * 4..72 + i * 4].copy_from_slice(&fixed.to_be_bytes());
    }
    // Default rendering intent = perceptual, matching [`INPUT_INTENT`] and the
    // reference profile. The caller passes its own intent to
    // [`Transform::new`], so this is a hint, not a control.
    b[64..68].copy_from_slice(&INPUT_INTENT.to_be_bytes());
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

/// `XYZType`: 8-byte header, then three `s15Fixed16` numbers.
pub(crate) fn xyz_tag(x: f32, y: f32, z: f32) -> Vec<u8> {
    let mut d = vec![0u8; 20];
    d[0..4].copy_from_slice(b"XYZ ");
    for (i, v) in [x, y, z].iter().enumerate() {
        let fixed = (v * 65536.0).round() as i32 as u32;
        d[8 + i * 4..12 + i * 4].copy_from_slice(&fixed.to_be_bytes());
    }
    d
}

/// `curveType` holding a single pure-gamma exponent — the TRC the Adobe RGB
/// (1998) built profile ships (its real `para` type-0 curves carry the same
/// `u8Fixed8` exponent), and the one test fixtures use to build a deliberately
/// non-sRGB profile. The sRGB profile itself needs the exact `para` rejection
/// curve instead, which is what `srgb_trc_tag` encodes.
pub(crate) fn curv_gamma_tag(g: f32) -> Vec<u8> {
    let mut d = vec![0u8; 14];
    d[0..4].copy_from_slice(b"curv");
    d[8..12].copy_from_slice(&1u32.to_be_bytes());
    let word = (g * 256.0).round() as u16;
    d[12..14].copy_from_slice(&word.to_be_bytes());
    d
}

/// The IEC 61966-2-1 EOTF as an ICC `parametricCurveType` **function type 3**
/// — `Y = (aX+b)^g` above `d`, else `Y = cX` — carrying
/// `2.4, 1/1.055, 0.055/1.055, 1/12.92, 0.04045`.
///
/// Not a `curv` gamma: sRGB's toe is not a power law, and a gamma-2.2 stand-in
/// would be a second opinion about linearisation disagreeing with the
/// pipeline's own [`crate::color::srgb_to_linear`].
///
/// The function type lives at **offset 8**, not 6 — verified against real
/// profiles rather than assumed: colord's `sRGB.icc` (and every `para` tag in
/// `/usr/share/color/icc`) stores `3` at `data[8..10]` with `data[6..8]`
/// reserved at zero, which is the layout [`super::parser::parse_curve`] reads.
/// The payload is 32 bytes, the same length the reference profile uses.
fn srgb_trc_tag() -> Vec<u8> {
    const PARAMS: [f32; 5] = [
        2.4,
        1.0 / 1.055,
        0.055 / 1.055,
        1.0 / 12.92,
        0.04045,
    ];
    let mut d = vec![0u8; 12 + 4 * PARAMS.len()];
    d[0..4].copy_from_slice(b"para");
    d[8..10].copy_from_slice(&3u16.to_be_bytes());
    for (i, v) in PARAMS.iter().enumerate() {
        let fixed = (v * 65536.0).round() as i32 as u32;
        d[12 + i * 4..16 + i * 4].copy_from_slice(&fixed.to_be_bytes());
    }
    d
}

/// `multiLocalizedUnicodeType` carrying one `en-US` record of `text` (ASCII at
/// both call sites, so UTF-16BE encoding is exact). The v4 way to write the
/// `desc` and `cprt` tags the ICC spec makes mandatory on a display profile;
/// layout mirrors the reference profile's: 16-byte header (`mluc`, reserved,
/// record count, record length), a 12-byte record (`en`/`US`, string length in
/// bytes, offset from the start of the tag), then the text at offset 28.
fn mluc_tag(text: &str) -> Vec<u8> {
    let mut chars = Vec::with_capacity(text.len() * 2);
    for c in text.encode_utf16() {
        chars.extend_from_slice(&c.to_be_bytes());
    }
    let mut d = vec![0u8; 28 + chars.len()];
    d[0..4].copy_from_slice(b"mluc");
    d[8..12].copy_from_slice(&1u32.to_be_bytes());
    d[12..16].copy_from_slice(&12u32.to_be_bytes());
    d[16..18].copy_from_slice(b"en");
    d[18..20].copy_from_slice(b"US");
    d[20..24].copy_from_slice(&(chars.len() as u32).to_be_bytes());
    d[24..28].copy_from_slice(&28u32.to_be_bytes());
    d[28..].copy_from_slice(&chars);
    d
}

/// `s15Fixed16ArrayType` carrying [`CHAD`]: type, reserved, then nine
/// `s15Fixed16` values in PCS order.
fn chad_tag() -> Vec<u8> {
    let mut d = vec![0u8; 8 + 4 * 9];
    d[0..4].copy_from_slice(b"sf32");
    for (i, v) in CHAD.iter().enumerate() {
        d[8 + i * 4..12 + i * 4].copy_from_slice(&v.to_be_bytes());
    }
    d
}

/// The sRGB profile the input path transforms *into*: `mntr`/`RGB `/`XYZ `,
/// the D50 white, D50-adapted sRGB colorants, and the exact sRGB EOTF on all
/// three channels.
///
/// Built rather than shipped as a vendored third-party file for two reasons:
/// there is no license to clear, and the bytes are inspectable — the whole
/// destination half of every colour transform the product performs is this one
/// function. The three `*TRC` tags are emitted as three copies rather than the
/// shared single offset a vendor profile would use; that costs 64 bytes and
/// keeps [`build_profile`] a straight tag list.
///
/// G2b: the same bytes are embedded in every exported file. The pipeline
/// renders every export into this working space, so an exported file's pixels
/// really *are* sRGB and the tag must say so; and because a third-party reader
/// — lcms, Windows, macOS, a browser — then sees the profile, it also carries
/// the three informational tags the ICC spec requires on a `mntr` display
/// profile: `desc` and `cprt` as [`mluc_tag`] records and the `chad`
/// adaptation matrix as [`chad_tag`], pinned byte-for-byte by
/// `srgb_profile_is_a_conformant_display_profile`. The colour-defining tags
/// are untouched, so this did not move a single pixel (pinned by
/// `added_informational_tags_leave_the_readers_alone`).
pub fn srgb_profile() -> Vec<u8> {
    build_profile(
        b"mntr",
        b"RGB ",
        b"XYZ ",
        &[
            (b"wtpt", xyz_tag(D50[0], D50[1], D50[2])),
            (b"rXYZ", xyz_tag(RXYZ[0], RXYZ[1], RXYZ[2])),
            (b"gXYZ", xyz_tag(GXYZ[0], GXYZ[1], GXYZ[2])),
            (b"bXYZ", xyz_tag(BXYZ[0], BXYZ[1], BXYZ[2])),
            (b"rTRC", srgb_trc_tag()),
            (b"gTRC", srgb_trc_tag()),
            (b"bTRC", srgb_trc_tag()),
            (b"desc", mluc_tag("sRGB (IEC 61966-2-1)")),
            (
                b"cprt",
                mluc_tag("Copyright (c) the c-41 project contributors. sRGB reference data per the public IEC 61966-2-1 specification."),
            ),
            (b"chad", chad_tag()),
        ],
    )
}

/// [`srgb_profile`], parsed once. Parsing is cheap but this runs per decoded
/// image, and a `static` keeps the destination identical for every transform
/// built in this process — a property worth having when two images are
/// compared side by side.
fn srgb() -> Result<&'static Profile, IccError> {
    static CACHE: OnceLock<Result<Profile, ()>> = OnceLock::new();
    match CACHE.get_or_init(|| Profile::parse(&srgb_profile()).map_err(|_| ())) {
        Ok(p) => Ok(p),
        Err(()) => Err(IccError::BadSignature),
    }
}

/// Adobe RGB (1998) as an in-memory interpretable profile — the one non-sRGB
/// assumption untagged containers can currently be given ([`InputAssumption::
/// AdobeRgb1998`]). Built in-process rather than shipped as a file for the
/// same reason [`srgb_profile`] is: the colourants come from the same
/// D50-consistent reference, nothing depends on a resource file, and the tests
/// can pin the generated bytes against a real profile's.
pub fn adobe_rgb1998_profile() -> Vec<u8> {
    build_profile(
        b"mntr",
        b"RGB ",
        b"XYZ ",
        &[
            (b"wtpt", xyz_tag(D50[0], D50[1], D50[2])),
            (
                b"rXYZ",
                xyz_tag(ADOBE_RGB1998[0][0], ADOBE_RGB1998[1][0], ADOBE_RGB1998[2][0]),
            ),
            (
                b"gXYZ",
                xyz_tag(ADOBE_RGB1998[0][1], ADOBE_RGB1998[1][1], ADOBE_RGB1998[2][1]),
            ),
            (
                b"bXYZ",
                xyz_tag(ADOBE_RGB1998[0][2], ADOBE_RGB1998[1][2], ADOBE_RGB1998[2][2]),
            ),
            (b"rTRC", curv_gamma_tag(ADOBE_RGB1998_GAMMA)),
            (b"gTRC", curv_gamma_tag(ADOBE_RGB1998_GAMMA)),
            (b"bTRC", curv_gamma_tag(ADOBE_RGB1998_GAMMA)),
        ],
    )
}

/// [`adobe_rgb1998_profile`], parsed once — the same two reasons [`srgb`]
/// caches its destination: cost per decoded image, and one canonical profile
/// so every transform built in this process agrees.
fn adobe_rgb1998() -> Result<&'static Profile, IccError> {
    static CACHE: OnceLock<Result<Profile, ()>> = OnceLock::new();
    match CACHE.get_or_init(|| Profile::parse(&adobe_rgb1998_profile()).map_err(|_| ())) {
        Ok(p) => Ok(p),
        Err(()) => Err(IccError::BadSignature),
    }
}

// ── the public seam ───────────────────────────────────────────────────────

/// What input profiling has to say about one decoded image.
///
/// Five states because there really are five, and the distinctions are what
/// the missing-profile prompt will have to act on: [`Self::Missing`] is a
/// container that says nothing (prompt-worthy), [`Self::Unknown`] is one this
/// module cannot read at all (nothing to prompt about), and [`Self::Unusable`]
/// is the uncomfortable case — a profile *is* present but cannot be applied,
/// so the user's pixels may genuinely not be sRGB and we are guessing.
pub enum InputProfile {
    /// The container carries no ICC profile. Interpreted as sRGB — the usual
    /// and usually correct assumption for an untagged JPEG.
    Missing,
    /// Carries a profile, and it *is* sRGB to within [`IDENTITY_TOL`]. Nothing
    /// to do; running the transform would only add rounding noise.
    Srgb,
    /// Carries one that will not parse, or describes a device space this
    /// engine cannot move (gray, CMYK): left alone, assumed sRGB.
    Unusable,
    /// Carries a profile worth applying. Convert the pixels with it.
    Transform(Transform),
    /// Not a container this module can scan (GIF, WebP, a raw file, or a file
    /// too broken to walk). No conclusion about presence, so no prompt and no
    /// transform.
    Unknown,
}

/// Diagnose a file's embedded profile for input profiling.
///
/// Cheap in every branch: [`Self::Srgb`] and [`Self::Unusable`] are the only
/// ones that build a transform, and [`Self::Srgb`] immediately drops it again.
pub fn input_profile(embedded: &Embedded) -> InputProfile {
    let Embedded::Present(bytes) = embedded else {
        return match embedded {
            Embedded::Absent => InputProfile::Missing,
            _ => InputProfile::Unknown,
        };
    };
    let Ok(src) = Profile::parse(bytes) else {
        // Present but unparsable — corrupt, or a container that stored
        // something that is not an ICC profile.
        return InputProfile::Unusable;
    };
    let Ok(dst) = srgb() else {
        // Unreachable: `srgb_profile` is built in-process from constants and is
        // pinned by `srgb_profile_parses`. Kept as a fallible path rather than
        // an unwrap so a future edit to the builder cannot panic a decode.
        return InputProfile::Unusable;
    };
    match Transform::new(&src, dst, INPUT_INTENT) {
        Ok(t) if is_srgb(&t) => InputProfile::Srgb,
        Ok(t) => InputProfile::Transform(t),
        Err(_) => InputProfile::Unusable,
    }
}

/// What an **untagged** container is taken to be, used when its embedded
/// profile is missing ([`InputProfile::Missing`]). A container that carries no
/// profile is otherwise — and by default — treated as sRGB, which is almost
/// always right for an untagged JPEG; the alternatives exist because untagged
/// wide-gamut images are real (scans, exports that dropped their profile) and
/// are indistinguishable from sRGB by content alone. The per-image prompt
/// (darkroom preview) asks this of the user once and remembers the answer.
///
/// The distinction is stored and then resolved against the engine: assuming
/// sRGB means *no transform* (the pixel bytes are the working space), while a
/// wider assumption moves the pixels, exactly as though the file had been
/// tagged with that profile. Raw files never get here — they are excluded at
/// the routing layer, and their input profile is the camera matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputAssumption {
    /// Assume untagged pixels already *are* sRGB: byte-exact, no transform.
    Srgb,
    /// Assume untagged pixels are Adobe RGB (1998): shape them into sRGB.
    AdobeRgb1998,
}

impl Default for InputAssumption {
    fn default() -> Self {
        InputAssumption::Srgb
    }
}

impl InputAssumption {
    /// Stable integer code for [`crate::persist`] (the DB stores this, so the
    /// numbering must not be renumbered once released).
    pub fn as_u8(self) -> u8 {
        match self {
            InputAssumption::Srgb => 0,
            InputAssumption::AdobeRgb1998 => 1,
        }
    }

    /// Inverse of [`Self::as_u8`]; unknown codes fall back to the default.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => InputAssumption::AdobeRgb1998,
            _ => InputAssumption::Srgb,
        }
    }

    /// The transform this assumption implies for a *missing*-profile container:
    /// `None` for [`Self::Srgb`] (the bytes are already the working space; a
    /// transform would only add rounding noise), the shape move for a wider
    /// assumption. Callers apply it only when classification was
    /// [`InputProfile::Missing`]; an embedded profile is never overridden.
    ///
    /// Infallible in current practice — both profiles are built in-process and
    /// pinned by tests — so the `Result` plumbing is swallowed here and a
    /// degenerate parse means "no move", consistent with the fallback for
    /// [`InputProfile::Unusable`].
    pub fn transform_for_missing(self) -> Option<Transform> {
        match self {
            InputAssumption::Srgb => None,
            InputAssumption::AdobeRgb1998 => match (srgb(), adobe_rgb1998()) {
                (Ok(dst), Ok(src)) => Transform::new(&src, dst, INPUT_INTENT).ok(),
                _ => None,
            },
        }
    }
}

/// True when applying `t` to every probe colour would leave it alone to within
/// [`IDENTITY_TOL`] — i.e. when the source profile already *is* sRGB.
fn is_srgb(t: &Transform) -> bool {
    let mut out = [0.0f32; 3];
    PROBES.iter().all(|p| {
        t.eval_into(p, &mut out);
        (0..3).all(|i| (out[i] - p[i]).abs() <= IDENTITY_TOL)
    })
}

/// Convert a packed RGB or RGBA buffer from its device profile into sRGB,
/// **in place**, honouring `rowstride` (gdk-pixbuf pads rows to 4 bytes, and
/// the padding is left exactly as it was) and leaving any fourth channel —
/// alpha — untouched.
///
/// `width` bounds how many pixels a row actually holds; `rowstride` may be
/// larger. That distinction is the whole reason for the parameter: RGB24 rows
/// are padded to a multiple of 4 bytes, so for a 2-pixel-wide image the stride
/// is 8 while only 6 bytes are data. Walking `rowstride / nch` pixels instead
/// would fold the padding into the next colour — a 2-pixel-wide RGB row would
/// read its own 2 padding bytes as the next pixel's red — and every row of
/// every non-multiple-of-4-wide image would be silently corrupted.
///
/// Both lengths are counted in *elements of `px`*: bytes for this function,
/// samples for [`apply_rgb16`]. A contiguous buffer's stride is `width * nch`.
///
/// `nch` below 3 is refused: there is no colour transform for a single channel
/// here, and guessing would mangle a grayscale buffer.
///
/// Returns `true` if a transform ran.
pub fn apply_rgb8(
    t: &Transform,
    px: &mut [u8],
    width: usize,
    rowstride: usize,
    nch: usize,
) -> bool {
    if nch < 3 || rowstride == 0 || width == 0 || px.is_empty() {
        return false;
    }
    let mut done = false;
    for row in px.chunks_mut(rowstride) {
        // `width * nch` is the data; anything past it in this stride is
        // padding, and `min(row.len)` also covers a short final row.
        let len = width.saturating_mul(nch).min(row.len());
        for c in row[..len].chunks_exact_mut(nch) {
            let input = [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0];
            let mut out = [0.0f32; 3];
            t.eval_into(&input, &mut out);
            for (i, v) in out.iter().enumerate() {
                c[i] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            done = true;
        }
    }
    done
}

/// [`apply_rgb8`] for the 16-bit PNG/TIFF export path. Same `width`,
/// `rowstride` and alpha rules; samples are normalized by 65535, which is what
/// an ICC profile means by a device value regardless of the container's bit
/// depth.
///
/// Returns `true` if a transform ran.
pub fn apply_rgb16(
    t: &Transform,
    px: &mut [u16],
    width: usize,
    rowstride: usize,
    nch: usize,
) -> bool {
    if nch < 3 || rowstride == 0 || width == 0 || px.is_empty() {
        return false;
    }
    let mut done = false;
    for row in px.chunks_mut(rowstride) {
        let len = width.saturating_mul(nch).min(row.len());
        for c in row[..len].chunks_exact_mut(nch) {
            let input = [c[0] as f32 / 65535.0, c[1] as f32 / 65535.0, c[2] as f32 / 65535.0];
            let mut out = [0.0f32; 3];
            t.eval_into(&input, &mut out);
            for (i, v) in out.iter().enumerate() {
                c[i] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
            done = true;
        }
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{linear_to_srgb, srgb_to_linear};

    // ── fixtures ─────────────────────────────────────────────────────────
    //
    // Two real profiles, rebuilt in memory so a test's expectation can be
    // derived from the same constants the fixture was built from.
    //

    /// The production assumption profile — same bytes, one source of truth:
    /// [`InputAssumption::AdobeRgb1998`] is built from these, so a test of the
    /// assumption (and of a *tagged* AdobeRGB file) can share the constants.
    use super::ADOBE_RGB1998 as ADOBE;
    use super::ADOBE_RGB1998_GAMMA as ADOBE_GAMMA;
    // The colorants are the **D50-adapted** XYZ triples read off the installed
    // profiles in `/usr/share/color/icc/colord`, *not* the D65 ones the Adobe
    // RGB spec quotes: ICC's profile connection space is D50, so a real file
    // stores the adapted matrix, and it is that matrix whose three columns sum
    // to `wtpt`. Using the spec's D65 numbers here is the trap — the fixture
    // then has no relationship to its own declared white, device white reaches
    // the PCS as something brighter than white, the destination saturates, and
    // a round trip that should close perfectly fails by 0.13. (Measured, not
    // guessed: that is exactly how `device_srgb_device_round_trips` failed on
    // its first run.)

    /// ProPhoto / ROMM — primaries far outside sRGB and a 1.8 power curve, so
    /// nothing about it can be mistaken for the destination.
    const PROPHOTO: [[f32; 3]; 3] = [
        [0.797668, 0.288040, 0.000000],
        [0.135193, 0.711884, 0.000000],
        [0.031342, 0.000092, 0.824905],
    ];
    const PROPHOTO_GAMMA: f32 = 1.8;

    /// Build one of the fixtures above as a matrix-shaper profile.
    ///
    /// `wtpt` is [`D50`] in both cases and the colorant columns sum to it to
    /// within 16e-6 — the property real profiles all have (checked across
    /// every matrix-shaper in `/usr/share/color/icc`), and the one the earlier
    /// fixture got wrong.
    fn fixture(colorants: &[[f32; 3]; 3], gamma: f32) -> Vec<u8> {
        build_profile(b"mntr", b"RGB ", b"XYZ ", &[
            (b"wtpt", xyz_tag(D50[0], D50[1], D50[2])),
            (b"rXYZ", xyz_tag(colorants[0][0], colorants[0][1], colorants[0][2])),
            (b"gXYZ", xyz_tag(colorants[1][0], colorants[1][1], colorants[1][2])),
            (b"bXYZ", xyz_tag(colorants[2][0], colorants[2][1], colorants[2][2])),
            (b"rTRC", curv_gamma_tag(gamma)),
            (b"gTRC", curv_gamma_tag(gamma)),
            (b"bTRC", curv_gamma_tag(gamma)),
        ])
    }

    /// The fixture as a transform, for tests whose subject is what it *does*
    /// rather than whether the gate lets it through.
    fn fixture_t(colorants: &[[f32; 3]; 3], gamma: f32) -> Transform {
        match input_profile(&Embedded::Present(fixture(colorants, gamma))) {
            InputProfile::Transform(t) => t,
            other => panic!(
                "fixture must produce a transform, got {}",
                match other {
                    InputProfile::Missing => "missing",
                    InputProfile::Srgb => "srgb",
                    InputProfile::Unusable => "unusable",
                    InputProfile::Unknown => "unknown",
                    InputProfile::Transform(_) => "transform",
                }
            ),
        }
    }

    // ── the destination profile ───────────────────────────────────────────

    #[test]
    fn srgb_profile_parses() {
        let p = Profile::parse(&srgb_profile()).expect("our own sRGB profile must parse");
        assert_eq!(&p.data_space, b"RGB ");
        assert_eq!(&p.pcs, b"XYZ ");
        // Every tag the matrix-shaper path needs, plus proof the builder is
        // deterministic (it is called twice on purpose).
        assert_eq!(srgb_profile(), srgb_profile(), "builder is not deterministic");
        for sig in [b"wtpt", b"rXYZ", b"gXYZ", b"bXYZ"] {
            assert!(p.read_xyz(sig).is_ok(), "missing or malformed {sig:?}");
        }
        let [r, g, b] = p.rgb_trc().expect("missing a TRC tag");
        assert_eq!(r, g, "the three channels must carry the same curve");
        assert_eq!(g, b, "the three channels must carry the same curve");
    }

    /// G2b: the exported bytes are what a third-party reader on another machine
    /// sees, so the profile has to be *conformant*, not just parseable by this
    /// engine. Pins the informational tags the ICC spec requires on a `mntr`
    /// display profile (`desc`/`cprt`/`chad`) and their exact on-disk layout.
    #[test]
    fn srgb_profile_is_a_conformant_display_profile() {
        let bytes = srgb_profile();
        let p = Profile::parse(&bytes).expect("our own sRGB profile must parse");

        // The declared size in the header must match the buffer — the assembler
        // writes it, so a padding bug would leave a short file on the wire.
        let declared = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        assert_eq!(declared, bytes.len(), "declared profile size must match the buffer");
        assert_eq!(p.device_class, *b"mntr");
        assert_eq!(&p.data_space, b"RGB ");
        assert_eq!(&p.pcs, b"XYZ ");

        // Creation date/time is mandatory (§7.2.4); pin the fixed stamp so the
        // builder stays deterministic and never ships a year-0 date.
        assert_eq!(
            &bytes[24..36],
            &[0x07, 0xea, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            "creation date/time must be a real, fixed date"
        );

        // The PCS illuminant (spec §7.2.16) is mandatory and must be D50 on
        // every profile; the reference sRGB profile writes exactly these bytes.
        // Strict readers (libpng, ImageMagick) warn on an all-zero illuminant,
        // and this header ships inside every exported file, so pin it.
        assert_eq!(
            &bytes[68..80],
            &[0x00, 0x00, 0xf6, 0xd6, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xd3, 0x2d],
            "PCS illuminant must be D50"
        );

        // Every tag's data is 4-byte aligned and wholly inside the file.
        let n = u32::from_be_bytes(bytes[128..132].try_into().unwrap()) as usize;
        for i in 0..n {
            let base = 132 + i * 12;
            let off = u32::from_be_bytes(bytes[base + 4..base + 8].try_into().unwrap()) as usize;
            let len = u32::from_be_bytes(bytes[base + 8..base + 12].try_into().unwrap()) as usize;
            assert_eq!(off % 4, 0, "tag {i} data not 4-byte aligned");
            assert!(off + len <= bytes.len(), "tag {i} data out of bounds");
        }

        // desc: a single en-US mluc record whose text names the profile. A
        // broken record count, length or offset would make a reader fall back
        // to a generic description — which is exactly what the tag exists to avoid.
        let desc = p.tag(b"desc").expect("a conformant mntr profile must carry desc");
        assert_eq!(&desc[0..4], b"mluc");
        assert_eq!(u32::from_be_bytes(desc[8..12].try_into().unwrap()), 1, "one record");
        assert_eq!(u32::from_be_bytes(desc[12..16].try_into().unwrap()), 12);
        assert_eq!(&desc[16..18], b"en");
        assert_eq!(&desc[18..20], b"US");
        let doff = u32::from_be_bytes(desc[24..28].try_into().unwrap()) as usize;
        assert_eq!(doff, 28, "first record's text starts right after the record");
        let dlen = u32::from_be_bytes(desc[20..24].try_into().unwrap()) as usize;
        let words: Vec<u16> = (0..dlen)
            .step_by(2)
            .map(|i| u16::from_be_bytes([desc[doff + i], desc[doff + i + 1]]))
            .collect();
        let desc_text = String::from_utf16(&words).unwrap();
        assert!(desc_text.contains("sRGB"), "description must say what the profile is: {desc_text:?}");

        // cprt is an mluc with non-empty text.
        let cprt = p.tag(b"cprt").expect("a conformant mntr profile must carry cprt");
        assert_eq!(&cprt[0..4], b"mluc");
        let clen = u32::from_be_bytes(cprt[20..24].try_into().unwrap()) as usize;
        assert!(clen > 0, "copyright text must not be empty");

        // chad: sf32 with the nine reference values, byte for byte.
        let chad = p.tag(b"chad").expect("a conformant mntr profile must carry chad");
        assert_eq!(&chad[0..4], b"sf32");
        let vals: Vec<i32> = (0..chad.len() - 8)
            .step_by(4)
            .map(|i| i32::from_be_bytes(chad[8 + i..12 + i].try_into().unwrap()))
            .collect();
        assert_eq!(vals, CHAD.to_vec());
    }

    /// The G2b additions must be inert for the engine: the matrix-shaper tags
    /// still parse to the same profile, and the sRGB gate still calls it sRGB —
    /// the property that keeps a re-export of an export from double-transforming.
    #[test]
    fn added_informational_tags_leave_the_readers_alone() {
        let p = Profile::parse(&srgb_profile()).unwrap();
        for sig in [b"wtpt", b"rXYZ", b"gXYZ", b"bXYZ"] {
            assert!(p.read_xyz(sig).is_ok(), "colour tag {sig:?} broke after G2b");
        }
        let [r, g, b] = p.rgb_trc().expect("TRC tags must survive the additions");
        assert_eq!(r, g);
        assert_eq!(g, b);
        assert!(
            matches!(
                input_profile(&Embedded::Present(srgb_profile())),
                InputProfile::Srgb
            ),
            "our own export profile must re-open as sRGB"
        );
    }

    /// The whole reason the destination is built rather than approximated: the
    /// profile's TRC and the pipeline's own lineariser must be *the same
    /// function*. If they ever drift, `apply_pipeline_gear` linearises with one
    /// curve and the input transform linearised with another, and every frame
    /// picks up a systematic shadow lift.
    #[test]
    fn srgb_trc_matches_the_pipelines_lineariser() {
        let p = Profile::parse(&srgb_profile()).unwrap();
        let [r, g, b] = p.rgb_trc().expect("TRC tags");
        for i in 0..=1000u32 {
            let x = i as f32 / 1000.0;
            let want = srgb_to_linear(x);
            for (name, c) in [("r", &r), ("g", &g), ("b", &b)] {
                let got = c.eval(x);
                assert!(
                    (got - want).abs() < 2e-5,
                    "{name}TRC({x}) = {got}, srgb_to_linear({x}) = {want}"
                );
            }
        }
    }

    #[test]
    fn white_is_the_sum_of_the_colorants() {
        let p = Profile::parse(&srgb_profile()).unwrap();
        let wt = p.read_xyz(b"wtpt").unwrap();
        let cols = p.rgb_to_xyz_matrix().unwrap();
        for (i, name) in ["X", "Y", "Z"].iter().enumerate() {
            let sum: f32 = cols[i].iter().sum();
            assert!(
                (sum - [wt.x, wt.y, wt.z][i]).abs() < 2.0 * QUANTUM,
                "{name}: colorants sum to {sum}, wtpt says {}",
                [wt.x, wt.y, wt.z][i]
            );
        }
    }

    /// Device white must arrive at the PCS as white, and black as black —
    /// otherwise a neutral would gain a colour cast on the way in.
    #[test]
    fn srgb_maps_white_and_black_to_themselves() {
        let dst = srgb().unwrap();
        let t = Transform::new(dst, dst, INPUT_INTENT).unwrap();
        let mut out = [0.0f32; 3];
        t.eval_into(&[1.0, 1.0, 1.0], &mut out);
        for v in out {
            assert!((v - 1.0).abs() < 1e-4, "white came out {out:?}");
        }
        t.eval_into(&[0.0, 0.0, 0.0], &mut out);
        for v in out {
            assert!(v.abs() < 1e-4, "black came out {out:?}");
        }
    }

    // ── classification ────────────────────────────────────────────────────

    /// An sRGB profile — including our own — must *not* produce a transform.
    /// Without this every untagged-or-sRGB photograph would take a matrix round
    /// trip and come back a fraction of a step off.
    #[test]
    fn an_srgb_profile_is_left_alone() {
        let emb = Embedded::Present(srgb_profile());
        assert!(matches!(input_profile(&emb), InputProfile::Srgb));
    }

    #[test]
    fn absent_and_unscannable_are_distinguished() {
        assert!(matches!(input_profile(&Embedded::Absent), InputProfile::Missing));
        assert!(matches!(
            input_profile(&Embedded::UnsupportedFormat),
            InputProfile::Unknown
        ));
    }

    #[test]
    fn a_broken_profile_is_unusable_not_a_panic() {
        let emb = Embedded::Present(b"not an ICC profile, just bytes".to_vec());
        assert!(matches!(input_profile(&emb), InputProfile::Unusable));
        // A truncated-but-plausible header: right magic, no tag table to read.
        let mut short = srgb_profile();
        short.truncate(133);
        assert!(matches!(
            input_profile(&Embedded::Present(short)),
            InputProfile::Unusable
        ));
    }

    /// A gray profile is a real, well-formed ICC file that this engine cannot
    /// move (it transforms three channels). Refusing it and falling back to the
    /// sRGB assumption is the honest outcome — a silent mis-transform would be
    /// worse than leaving the bytes alone.
    #[test]
    fn a_gray_profile_is_refused() {
        let bytes = build_profile(b"mntr", b"GRAY", b"XYZ ", &[
            (b"wtpt", xyz_tag(D50[0], D50[1], D50[2])),
            (b"kTRC", curv_gamma_tag(2.2)),
        ]);
        assert!(Profile::parse(&bytes).is_ok(), "fixture must be a valid profile");
        assert!(matches!(
            input_profile(&Embedded::Present(bytes)),
            InputProfile::Unusable
        ));
    }

    // ── the missing-profile assumption ────────────────────────────────────

    /// The per-image prompt's default: untagged pixels are sRGB — no move.
    #[test]
    fn assume_srgb_is_the_default_and_means_no_move() {
        assert_eq!(InputAssumption::default(), InputAssumption::Srgb);
        assert!(
            InputAssumption::Srgb.transform_for_missing().is_none(),
            "assuming sRGB must mean byte-exact, not a matrix round trip"
        );
    }

    /// Stable codes for the DB (persist.rs stores these; renumbering would
    /// re-label every saved per-file choice).
    #[test]
    fn assumption_codes_round_trip_and_are_stable() {
        assert_eq!(InputAssumption::Srgb.as_u8(), 0);
        assert_eq!(InputAssumption::AdobeRgb1998.as_u8(), 1);
        for v in [0, 1] {
            assert_eq!(InputAssumption::from_u8(v).as_u8(), v);
        }
        assert_eq!(InputAssumption::from_u8(9), InputAssumption::Srgb);
    }

    /// Assumed Adobe RGB is the SAME move a *tagged* AdobeRGB file gets: the
    /// untagged pixels the prompt labelled go through exactly the pipeline
    /// `input_profile` runs on a real profile. That identity is the whole
    /// contract of the assumption.
    #[test]
    fn assumed_adobe_rgb_matches_the_tagged_pipeline() {
        let Some(assumed) = InputAssumption::AdobeRgb1998.transform_for_missing() else {
            panic!("Adobe RGB assumption must imply a transform");
        };
        let InputProfile::Transform(tagged) =
            input_profile(&Embedded::Present(adobe_rgb1998_profile()))
        else {
            panic!("fixture must classify as a transform");
        };
        let probe = [0.5f32, 0.0, 0.8];
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        assumed.eval_into(&probe, &mut a);
        tagged.eval_into(&probe, &mut b);
        assert!(
            a.iter().zip(&probe).any(|(x, p)| (x - p).abs() > 0.05),
            "assumed Adobe RGB must move colours measurably: {a:?}"
        );
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() <= 2e-4,
                "assumed ({a:?}) and tagged ({b:?}) Adobe RGB diverge at {probe:?}"
            );
        }
    }

    /// The gate must not be so loose it swallows a genuinely different profile.
    /// Gamma 2.2 is the classic "close enough to sRGB" trap: it agrees at the
    /// ends and differs by up to ~4/1000 in the middle — far more than a 16-bit
    /// LSB — so it must be transformed, not skipped.
    #[test]
    fn a_gamma22_profile_is_transformed_not_skipped() {
        let bytes = build_profile(b"mntr", b"RGB ", b"XYZ ", &[
            (b"wtpt", xyz_tag(D50[0], D50[1], D50[2])),
            (b"rXYZ", xyz_tag(RXYZ[0], RXYZ[1], RXYZ[2])),
            (b"gXYZ", xyz_tag(GXYZ[0], GXYZ[1], GXYZ[2])),
            (b"bXYZ", xyz_tag(BXYZ[0], BXYZ[1], BXYZ[2])),
            (b"rTRC", curv_gamma_tag(2.2)),
            (b"gTRC", curv_gamma_tag(2.2)),
            (b"bTRC", curv_gamma_tag(2.2)),
        ]);
        let InputProfile::Transform(_) = input_profile(&Embedded::Present(bytes)) else {
            panic!("gamma 2.2 is not sRGB and must be transformed");
        };
    }

    /// A wide-gamut profile must survive the gate and actually move pixels —
    /// this is the whole product behaviour of G2a2.
    ///
    /// Three claims, in order of how much they pin:
    ///
    /// 1. the gate does **not** swallow it (it is not sRGB);
    /// 2. the **primaries** are doing work, not just the curve — a neutral
    ///    cannot show this, because a neutral only ever travels through the
    ///    sum of the colorants, which is D50 for every sane profile;
    /// 3. white still arrives as white, which is what proves the fixture is
    ///    physically consistent rather than merely non-sRGB.
    #[test]
    fn a_wide_gamut_profile_actually_moves_pixels() {
        let InputProfile::Transform(t) =
            input_profile(&Embedded::Present(fixture(&PROPHOTO, PROPHOTO_GAMMA)))
        else {
            panic!("ProPhoto is not sRGB and must be transformed");
        };
        let mut out = [0.0f32; 3];

        // A probe chosen to land *inside* sRGB. A saturated ProPhoto colour
        // clips at both ends and would prove only that the clamp works; this
        // one stays interior, so every difference below comes from the matrix.
        let dev = [0.6f32, 0.85, 0.75];
        t.eval_into(&dev, &mut out);

        // What sRGB's own primaries with this same 1.8 curve would give: with
        // identity colorants the matrix cancels and the answer is just the
        // curve. The two must disagree, or the colorants are being read and
        // then ignored.
        let gamma_only = [
            linear_to_srgb(dev[0].powf(PROPHOTO_GAMMA)),
            linear_to_srgb(dev[1].powf(PROPHOTO_GAMMA)),
            linear_to_srgb(dev[2].powf(PROPHOTO_GAMMA)),
        ];
        let colorant_effect = (0..3)
            .map(|i| (out[i] - gamma_only[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            colorant_effect > 0.2,
            "colorants had no visible effect: got {out:?}, gamma alone would give {gamma_only:?}"
        );

        let moved = (0..3).map(|i| (out[i] - dev[i]).abs()).fold(0.0f32, f32::max);
        assert!(
            moved > 0.2,
            "the image would not visibly change: {dev:?} came back {out:?}"
        );

        // Device white must still reach the PCS as white — the fixture's
        // colorants sum to `wtpt`, so no neutral gains a cast.
        t.eval_into(&[1.0, 1.0, 1.0], &mut out);
        for v in out {
            assert!((v - 1.0).abs() < 1e-3, "white became {out:?}");
        }
    }

    /// Round trip: taking a colour into a wider space and back must land where
    /// it started. Both legs get used, and between them both halves of
    /// [`super::parser::Curve::inverse`] — the sRGB → wide leg inverts Adobe's
    /// pure-power curve, which short-circuits to `Gamma(1/g)`, while the wide
    /// → sRGB leg inverts our parametric EOTF through the 4096-entry sampling
    /// table.
    ///
    /// The **direction** is the whole trick. Starting from sRGB, the narrower
    /// of the two spaces, every probe stays inside Adobe RGB, nothing clips,
    /// and the question has a clean answer: 0.0002, measured. Run the same
    /// round trip the other way and Adobe's saturated primaries leave the
    /// first pass pinned at 1.0 with their information gone — 0.565 of error
    /// on these same thirteen probes. That is clipping being *correct* rather
    /// than the transform being wrong, which is why this test only ever asks
    /// the question it can answer, and why `out_of_gamut_clamps` exists to
    /// pin the clipping half separately.
    #[test]
    fn srgb_round_trips_through_a_wider_profile() {
        let wide = Profile::parse(&fixture(&ADOBE, ADOBE_GAMMA)).unwrap();
        let dst = srgb().unwrap();
        let out_to_wide = Transform::new(dst, &wide, INPUT_INTENT).unwrap();
        let back_to_srgb = Transform::new(&wide, dst, INPUT_INTENT).unwrap();

        let mut mid = [0.0f32; 3];
        let mut back = [0.0f32; 3];
        for p in PROBES {
            out_to_wide.eval_into(&p, &mut mid);
            back_to_srgb.eval_into(&mid, &mut back);
            for i in 0..3 {
                assert!(
                    (back[i] - p[i]).abs() < 5e-3,
                    "round trip of {p:?} gave {back:?}"
                );
            }
        }
    }

    /// Cross-check the ICC assembly path against direct matrix maths.
    ///
    /// `Transform` reaches sRGB by parsing tags into pipelines; the expectation
    /// below is computed straight from the profile's own numbers using
    /// [`crate::color`]'s primitives. Two independent routes to the same answer —
    /// if they disagree, one of them is wrong and neither test alone would say so.
    #[test]
    fn transform_agrees_with_direct_matrix_maths() {
        // Adobe RGB's real numbers, taken from the same constants `fixture`
        // builds from — that is the point: the expectation below is derived
        // from what the profile actually says rather than from a second copy
        // of it that could drift out of step.
        let t = fixture_t(&ADOBE, ADOBE_GAMMA);

        // sRGB's colorants, inverted — the D50-adapted matrix the destination
        // profile declares. `ADOBE[i]` is column i (R, G, B).
        let m = [
            [ADOBE[0][0], ADOBE[1][0], ADOBE[2][0]],
            [ADOBE[0][1], ADOBE[1][1], ADOBE[2][1]],
            [ADOBE[0][2], ADOBE[1][2], ADOBE[2][2]],
        ];
        let srgb_m = [
            [RXYZ[0], GXYZ[0], BXYZ[0]],
            [RXYZ[1], GXYZ[1], BXYZ[1]],
            [RXYZ[2], GXYZ[2], BXYZ[2]],
        ];
        let srgb_inv = invert3(srgb_m).expect("sRGB matrix must be invertible");

        for p in [[0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [0.5, 0.0, 0.8], [0.0, 0.0, 0.0]] {
            let mut got = [0.0f32; 3];
            t.eval_into(&p, &mut got);

            // The independent route: device -> linear under the profile's own
            // gamma -> XYZ through its colorants -> sRGB linear through the
            // inverse sRGB colorants -> sRGB device.
            let lin = [
                p[0].max(0.0).powf(ADOBE_GAMMA),
                p[1].max(0.0).powf(ADOBE_GAMMA),
                p[2].max(0.0).powf(ADOBE_GAMMA),
            ];
            let mut xyz = [0.0f32; 3];
            for i in 0..3 {
                xyz[i] = m[i][0] * lin[0] + m[i][1] * lin[1] + m[i][2] * lin[2];
            }
            let mut lin_srgb = [0.0f32; 3];
            for i in 0..3 {
                lin_srgb[i] =
                    srgb_inv[i][0] * xyz[0] + srgb_inv[i][1] * xyz[1] + srgb_inv[i][2] * xyz[2];
            }
            for i in 0..3 {
                // Clamp both sides: an out-of-sRGB input is allowed to leave
                // the transform above 1.0, and `linear_to_srgb` does not clamp
                // for us (see its own doc). Comparing the saturated answers is
                // the question we are actually asking.
                let got = got[i].clamp(0.0, 1.0);
                let want = linear_to_srgb(lin_srgb[i].clamp(0.0, 1.0));
                assert!(
                    (got - want).abs() < 2.0e-3,
                    "device {p:?} channel {i}: transform gave {got}, direct maths gave {want}"
                );
            }
        }
    }

    fn invert3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
        let [[a, b, c], [d, e, f], [g, h, i]] = m;
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if det.abs() < 1e-12 {
            return None;
        }
        Some([
            [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
        ])
    }

    // ── the buffer loops ──────────────────────────────────────────────────

    /// A transform that is reliably *not* the identity, so the buffer loops can
    /// be watched actually moving pixels with no colour science in the way.
    ///
    /// It is identity-shaped colorants (`XYZ = device`) with a linear TRC, which
    /// is emphatically not sRGB twice over: the colorants are not sRGB's, and
    /// the curve is not its EOTF. Routing it through the sRGB destination gives
    /// a real mapping that changes every non-black sample — all these tests need
    /// from it is that a loop which silently skipped its pixels fails loudly.
    fn moving_t() -> Transform {
        let bytes = build_profile(b"mntr", b"RGB ", b"XYZ ", &[
            (b"wtpt", xyz_tag(D50[0], D50[1], D50[2])),
            (b"rXYZ", xyz_tag(1.0, 0.0, 0.0)),
            (b"gXYZ", xyz_tag(0.0, 1.0, 0.0)),
            (b"bXYZ", xyz_tag(0.0, 0.0, 1.0)),
            (b"rTRC", curv_gamma_tag(1.0)),
            (b"gTRC", curv_gamma_tag(1.0)),
            (b"bTRC", curv_gamma_tag(1.0)),
        ]);
        let InputProfile::Transform(t) = input_profile(&Embedded::Present(bytes)) else {
            panic!("identity colorants and a linear curve are not sRGB");
        };
        t
    }

    /// Row padding must survive untouched, and the layout here is the one that
    /// makes it hard: RGB24 rows round up to a multiple of 4 bytes, so a
    /// **3**-pixel-wide row is 9 bytes of pixel plus 3 bytes of padding — a
    /// whole extra `nch` of padding, which a loop walking the stride in
    /// `nch`-sized steps would happily consume as a fourth pixel.
    ///
    /// The earlier version of this fixture put its padding at byte 3 (a
    /// per-pixel alignment no loader produces) and so passed whichever loop
    /// ran it. Padding is at the *end* of the row; that is where it is now.
    #[test]
    fn rowstride_padding_is_left_alone() {
        let mut px = vec![
            0, 10, 20, // pixel 0
            30, 40, 50, // pixel 1
            60, 70, 80, // pixel 2
            99, 77, 55, // padding: 9 data bytes rounded up to a 12-byte stride
        ];
        let before = px.clone();
        assert!(apply_rgb8(&moving_t(), &mut px, 3, 12, 3));
        assert_eq!(&px[9..12], &before[9..12], "row padding must not be touched");
        for p in 0..3 {
            let i = p * 3;
            assert_ne!(
                px[i..i + 3],
                before[i..i + 3],
                "pixel {p} was left alone instead of transformed"
            );
        }
    }

    #[test]
    fn alpha_is_never_touched() {
        let mut px = vec![10u8, 20, 30, 128, 40, 50, 60, 255];
        assert!(apply_rgb8(&moving_t(), &mut px, 2, 8, 4));
        assert_eq!(px[3], 128, "alpha rewritten");
        assert_eq!(px[7], 255, "alpha rewritten");
        assert_ne!(px[0..3], [10, 20, 30], "colour not transformed");
    }

    #[test]
    fn degenerate_buffers_are_refused_not_panicked() {
        assert!(!apply_rgb8(&moving_t(), &mut [], 0, 0, 3), "empty buffer");
        // A zero stride would make `chunks_mut(0)` panic outright, so it is
        // refused before the loop is ever entered.
        assert!(!apply_rgb8(&moving_t(), &mut [0u8; 8], 1, 0, 3), "zero rowstride");
        assert!(!apply_rgb8(&moving_t(), &mut [0u8; 8], 0, 8, 3), "zero width");
        let mut gray = vec![1u8, 2, 3, 4];
        assert!(!apply_rgb8(&moving_t(), &mut gray, 1, 4, 1), "single channel");
        assert_eq!(gray, vec![1, 2, 3, 4], "grayscale buffer mutated anyway");
    }

    #[test]
    fn sixteen_bit_matches_eight_bit() {
        // Same colour through both loops: 16-bit must land on the same value
        // (they are the same transform, only the container differs).
        let mut a = vec![0x80u8, 0x40, 0xC0, 0x20, 0x10, 0xF0];
        let mut b = vec![
            0x8080u16, 0x4040, 0xC0C0, 0x2020, 0x1010, 0xF0F0,
        ];
        assert!(apply_rgb8(&moving_t(), &mut a, 2, 6, 3));
        assert!(apply_rgb16(&moving_t(), &mut b, 2, 6, 3));
        for i in 0..6 {
            // Quantize the 16-bit answer back down to 8 bits. Both loops saw
            // the identical f32 sample (0x80/255 == 0x8080/65535 exactly), so
            // they must encode the same colour to within one 8-bit step.
            // Comparing the raw integers instead would be meaningless: an
            // 8-bit 128 expanded by 257 and round(x * 65535) may legitimately
            // differ by 128 while standing for the very same sample.
            let down = ((b[i] as f32 / 65535.0) * 255.0).round() as u8;
            assert!(
                (down as i32 - a[i] as i32).abs() <= 1,
                "channel {i}: 8-bit path gave {a:?}, 16-bit gave {b:?}"
            );
        }
    }

    /// Out-of-gamut results saturate; they never wrap.
    ///
    /// The colorants are 1.5× sRGB's, so a bright neutral genuinely wants more
    /// than white: `200/255 × 1.5 = 1.176` in linear sRGB. That overshoot is
    /// dealt with twice — by [`super::parser::Curve::eval`], whose contract is
    /// an output clamped to `[0,1]` and which is what every destination TRC
    /// runs through, and then defensively by the buffer loop — and what this
    /// pins is the observable result: 255 where white was exceeded, 0 for
    /// black, and the f32 stage landing exactly on the ceiling rather than
    /// somewhere past it.
    ///
    /// `wtpt` is declared as 1.5 × D50 so the fixture stays self-consistent
    /// (its colorants sum to it, like every real profile's do). Rendering
    /// intent 0 does not read `wtpt`, so the declaration changes nothing about
    /// the transform — the *colorants* are what make white want more than
    /// white.
    #[test]
    fn out_of_gamut_clamps() {
        let bright_white = [D50[0] * 1.5, D50[1] * 1.5, D50[2] * 1.5];
        let bytes = build_profile(b"mntr", b"RGB ", b"XYZ ", &[
            (b"wtpt", xyz_tag(bright_white[0], bright_white[1], bright_white[2])),
            (b"rXYZ", xyz_tag(RXYZ[0] * 1.5, RXYZ[1] * 1.5, RXYZ[2] * 1.5)),
            (b"gXYZ", xyz_tag(GXYZ[0] * 1.5, GXYZ[1] * 1.5, GXYZ[2] * 1.5)),
            (b"bXYZ", xyz_tag(BXYZ[0] * 1.5, BXYZ[1] * 1.5, BXYZ[2] * 1.5)),
            (b"rTRC", curv_gamma_tag(1.0)),
            (b"gTRC", curv_gamma_tag(1.0)),
            (b"bTRC", curv_gamma_tag(1.0)),
        ]);
        let InputProfile::Transform(t) = input_profile(&Embedded::Present(bytes)) else {
            panic!("colorants 50% brighter than sRGB are not sRGB");
        };

        let mut out = [0.0f32; 3];
        t.eval_into(&[200.0 / 255.0; 3], &mut out);
        assert!(
            out.iter().all(|v| (v - 1.0).abs() < 1e-6),
            "the overshoot was not saturated: {out:?}"
        );

        let mut px = vec![200u8, 200, 200, 0, 0, 0];
        assert!(apply_rgb8(&t, &mut px, 2, 6, 3));
        assert_eq!(&px[0..3], &[255, 255, 255], "overshoot did not saturate");
        assert_eq!(&px[3..6], &[0, 0, 0], "the ceiling must not lift black");
    }
}
