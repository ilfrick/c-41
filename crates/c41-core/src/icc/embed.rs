//! Extract the ICC profile embedded in a non-raw image container.
//!
//! This is the **input-profile** half of colour management and the first
//! increment of G2 (the `PARITY_AUDIT.md` G2 item). It does exactly one job:
//! given the bytes of a JPEG, PNG or TIFF, hand back the raw ICC profile they
//! carry, reassembled according to that container's own chunking rules.
//!
//! # Why this has to exist at all
//!
//! The preview decodes non-raw files through **gdk-pixbuf**
//! (`c41-ui/src/darkroom/mod.rs`, `PixbufLoader`), and gdk-pixbuf 0.20 exposes no
//! ICC surface at all — no ICC symbol in the crate source or in
//! `gdk-pixbuf-2.0/gdk-pixbuf.h`. That half is settled: the container has to be
//! read for the profile ourselves.
//!
//! Export is the interesting half, and the first draft of this paragraph was
//! wrong. The `image` crate 0.25 *does* implement
//! `ImageDecoder::icc_profile()` for jpeg, png and tiff (plus avif/gif/webp), and
//! `ImageEncoder::set_icc_profile()` for jpeg/png/tiff/webp — so "no library
//! exposes ICC" would be a lie. It is still not usable for the job:
//!
//! 1. That method hangs off a constructed **pixel decoder**. The preview path
//!    has no `ImageReader` to construct one from, so using it would mean two
//!    different extraction paths — the exact divergence this module exists to
//!    avoid.
//! 2. Its TIFF decoder **rejects many real TIFFs outright**, failing with
//!    `Photometric interpretation RGBPalette ... is unsupported` on palette
//!    TIFFs that ImageMagick wrote and that gdk-pixbuf opens happily. Verified
//!    directly. A metadata-only extractor that walks the IFD and ignores pixel
//!    layout reads those files' profiles just fine, because it never asks what
//!    the pixels are.
//!
//! So: one extractor, used by both decode paths, that does not care whether the
//! pixels are decodable. It is small because ICC lives in three well-specified
//! places and the expensive part — turning bytes into a transform — is
//! [`super::Profile`], which already exists.
//!
//! # The result is deliberately three-state, not `Option`
//!
//! Q2 of the G2 plan is an explicit decision: when a file carries no profile,
//! c41 asks the user to choose rather than silently assuming sRGB. That policy
//! needs more than "is there a profile", because *not knowing* and *knowing
//! there is none* call for different messages:
//!
//! * [`Embedded::Present`] — a profile was found. Its bytes may still be
//!   unusable ([`super::Profile::parse`] can reject them, or
//!   [`super::Transform::new`] can refuse them, e.g. a CMYK or grayscale
//!   profile: [`transform::Transform`] requires every pipeline stage to be
//!   3-channel and rejects the rest at assembly). That is deliberately **not**
//!   collapsed into `Absent` here: presence is a question about the container,
//!   parseability is a question about the engine, and only the second one is
//!   the caller's business. Conflating them would make "this file is a CMYK
//!   image and we cannot handle that" look identical to "this file was never
//!   colour-managed", which are very different things to tell a user — and the
//!   second of those would then wrongly offer a profile choice for a file whose
//!   profile is right there.
//! * [`Embedded::Absent`] — a container we understand, carrying no profile.
//! * [`Embedded::UnsupportedFormat`] — not a container this module parses. Also
//!   not an error: the caller still cannot conclude anything about presence,
//!   but it *can* conclude it needs to ask.
//!
//! Collapsing `Absent` and `UnsupportedFormat` into one state would be wrong
//! for a second reason: a caller that wants to warn "I couldn't check this
//! file's profile" needs to tell them apart from "I checked; there isn't one".
//!
//! # Scope, and what is deliberately not here
//!
//! * **JPEG** (`FF D8`) — profile in `APP2` segments whose payload opens with
//!   `ICC_PROFILE\0`, then a 1-based sequence number and a total count. Profiles
//!   above ~64 KB exceed the 64 KB segment limit and are split, so the chunks
//!   are sorted by sequence number and concatenated rather than taken in file
//!   order (files in the wild do interleave them).
//! * **PNG** — the `iCCP` chunk, a null-terminated keyword, a compression
//!   method byte, then a **zlib**-compressed profile. This is the only one of
//!   the three that needs a decompressor (`flate2`); JPEG and TIFF store ICC
//!   uncompressed.
//! * **TIFF** (both byte orders) — IFD entry tag `34675`
//!   (`InterColorProfile`), type `UNDEFINED`, inline when it fits in four
//!   bytes and out-of-line otherwise. We search IFD0 and follow the IFD chain
//!   (for multi-page TIFFs) but deliberately **not** SubIFDs: ICC in a SubIFD
//!   is non-standard, and darktable's own readers do not look there.
//!
//! Not here, by design: **writing** a profile into a container (the G2b
//! export-tagging increment — cheaper than first assumed, since export already
//! encodes through `image`'s `ImageEncoder`, which has `set_icc_profile` for
//! jpeg/png/tiff/webp), embedded profiles in **raw** files (camera profiles are
//! a separate and much larger problem, reached through [`crate::rawimage`], not
//! here), and decoding pixels — this module never touches image data, only
//! metadata.
//!
//! # Streaming
//!
//! [`extract`] takes `Read + Seek` rather than a byte slice so a caller can
//! hand it a [`std::fs::File`] for a 50 MB TIFF without the profile scan
//! loading the image. JPEG and PNG walk marker/chunk headers forward and seek
//! over payloads; TIFF seeks to each candidate profile. The scan always stops
//! at the point where a profile can no longer legally appear (`SOS` for JPEG,
//! `IDAT` for PNG) rather than reading to end-of-file.

use std::io::{self, Read, Seek, SeekFrom};

/// TIFF tag number for `InterColorProfile` — the embedded ICC profile.
const TIFF_TAG_INTERCOLORPROFILE: u16 = 34675;
/// TIFF field type `UNDEFINED` (7), the type ICC profiles are stored as.
const TIFF_TYPE_UNDEFINED: u16 = 7;

/// The 12-byte marker that opens an `APP2` payload carrying an ICC profile:
/// `"ICC_PROFILE"` followed by a NUL. The next two bytes are then the chunk's
/// 1-based sequence number and the total chunk count.
const ICC_PROFILE_TAG: &[u8; 12] = b"ICC_PROFILE\0";

/// Which container a byte slice is, decided by content rather than by
/// filename — extension lies often enough that trusting it would be a bug.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Jpeg,
    Png,
    Tiff,
}

/// Outcome of an embedded-profile scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Embedded {
    /// A profile was found; these are its bytes, exactly as the container
    /// stored them, with the container's chunking already undone. Not
    /// necessarily *usable* — see [`super::Profile::parse`].
    Present(Vec<u8>),
    /// A container this module understands, which carries no profile.
    Absent,
    /// Not a JPEG, PNG or TIFF. No conclusion either way about presence.
    UnsupportedFormat,
}

/// Why a scan could not be completed.
///
/// Distinct from [`Embedded::UnsupportedFormat`]: that one means "I do not
/// parse this container", this one means "I parse it but the file is broken".
#[derive(Debug)]
pub enum EmbedError {
    /// The underlying reader failed.
    Io(io::Error),
    /// The file ends in the middle of a structure this module was walking.
    Truncated,
    /// The container's structure is self-inconsistent. `what` names the
    /// structure so the message is actionable without a backtrace.
    Malformed(&'static str),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "icc: read: {e}"),
            Self::Truncated => write!(f, "icc: file truncated inside a container structure"),
            Self::Malformed(what) => write!(f, "icc: malformed container: {what}"),
        }
    }
}

impl std::error::Error for EmbedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for EmbedError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Read the ICC profile embedded in a JPEG, PNG or TIFF.
///
/// Returns [`Embedded::Present`] with the profile's bytes, [`Embedded::Absent`]
/// for a container with no profile, or [`Embedded::UnsupportedFormat`] for
/// anything else. See the [module docs](self) for why the result is not an
/// `Option`.
///
/// Reading stops as soon as a profile can no longer legally appear, so this is
/// cheap on a large file even though it is handed the whole reader.
pub fn extract<R: Read + Seek>(r: &mut R) -> Result<Embedded, EmbedError> {
    let mut magic = [0u8; 8];
    // A file shorter than its own magic is not a container we handle. Read
    // leniently: `read_exact` on a short file would be `Truncated`, but for
    // format sniffing "too small to be anything" is simply `Unsupported`.
    let got = read_up_to(r, &mut magic)?;
    let format = match &magic[..got] {
        [0xFF, 0xD8, ..] => Format::Jpeg,
        [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] => Format::Png,
        [b'I', b'I', 0x2A, 0x00, ..] | [b'M', b'M', 0x00, 0x2A, ..] => Format::Tiff,
        _ => return Ok(Embedded::UnsupportedFormat),
    };
    // Rewind: each reader below consumes its own signature from the start of
    // the file, rather than inheriting a half-read magic from the sniffer.
    r.seek(SeekFrom::Start(0))?;
    match format {
        Format::Jpeg => read_jpeg(r),
        Format::Png => read_png(r),
        Format::Tiff => read_tiff(r),
    }
}

/// Convenience wrapper over [`extract`] for a path.
pub fn extract_file(path: &std::path::Path) -> Result<Embedded, EmbedError> {
    let mut f = std::fs::File::open(path)?;
    extract(&mut f)
}

/// Fill `buf` as far as the reader allows, returning how many bytes landed.
///
/// `read_exact` would report `UnexpectedEof` for a file shorter than the magic
/// we are sniffing, which is the wrong diagnosis: a 3-byte file is not a broken
/// JPEG, it is not a JPEG at all.
fn read_up_to<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<usize, EmbedError> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(n)
}

/// Fill `buf` completely or fail as [`EmbedError::Truncated`].
fn read_exact_or_truncated<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), EmbedError> {
    r.read_exact(buf).map_err(|e| match e.kind() {
        io::ErrorKind::UnexpectedEof => EmbedError::Truncated,
        _ => EmbedError::Io(e),
    })
}

/// The integrity check every extracted profile must pass before we call it a
/// profile.
///
/// Two independent things are verified, because either can be wrong for a
/// reason that matters:
///
/// * **The `acsp` signature at offset 36.** That offset is fixed by the ICC
///   specification and is what distinguishes an ICC profile from any other blob
///   a container might be pointing us at.
/// * **The declared size in the first four bytes agrees with the length we
///   extracted.** If it does not, the chunk walk mis-reassembled the profile —
///   most likely a JPEG whose ICC chunks are incomplete or out of sequence.
///   Checking the signature alone would pass a *truncated* profile, because a
///   truncation that happens after byte 40 leaves `acsp` intact while the
///   declared size still describes the whole thing.
///
/// A profile that fails either check is reported as
/// [`EmbedError::Malformed`] rather than [`Embedded::Present`], because handing
/// back bytes we know are wrong would push the diagnosis into the engine, where
/// it would read as "unsupported profile" and produce a misleading message
/// about the user's file rather than about our parse.
fn validate_icc(bytes: &[u8]) -> Result<(), EmbedError> {
    const HEADER_LEN: usize = 128;
    if bytes.len() < HEADER_LEN {
        return Err(EmbedError::Malformed("icc profile shorter than its 128-byte header"));
    }
    if &bytes[36..40] != b"acsp" {
        return Err(EmbedError::Malformed("extracted blob is not an ICC profile (no 'acsp' at 36)"));
    }
    let declared = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if declared != bytes.len() {
        return Err(EmbedError::Malformed("icc profile size field disagrees with the bytes extracted"));
    }
    Ok(())
}

// ── JPEG ───────────────────────────────────────────────────────────────────

/// Walk JPEG marker segments looking for `APP2`/`ICC_PROFILE` chunks.
///
/// Stops at `SOS`: everything from there on is entropy-coded image data, and
/// ICC is a header-only construct, so scanning past it would risk reading
/// marker-like byte pairs inside compressed data.
fn read_jpeg<R: Read + Seek>(r: &mut R) -> Result<Embedded, EmbedError> {
    // (sequence number, payload) per chunk.
    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut total: Option<u8> = None;
    loop {
        // Markers are `FF xx`, but may be preceded by any number of `FF` fill
        // bytes, and `FF 00` inside entropy data is a stuffed zero rather than
        // a marker. So track whether the byte just read was `FF`, eat any run
        // of fill bytes, and take the first byte that is neither `FF` nor a
        // stuffed `00` as the marker id. Scanning byte-wise matters: reading
        // in pairs desynchronises on `FF FF E2`, which is legal and appears in
        // files written by encoders that pad.
        let mut b = [0u8; 1];
        let mut prev_was_ff = false;
        let id = loop {
            read_exact_or_truncated(r, &mut b)?;
            if prev_was_ff {
                if b[0] == 0xFF {
                    continue; // fill byte before the marker id
                }
                prev_was_ff = false;
                if b[0] == 0x00 {
                    continue; // stuffed zero, keep scanning
                }
                break b[0];
            }
            prev_was_ff = b[0] == 0xFF;
        };

        match id {
            // Start of scan / end of image: the header is over either way.
            0xDA | 0xD9 => break,
            // Markers that stand alone, with no length field. SOI (0xD8) is
            // here, not caught by the arm above: the walk starts *at* it, and
            // reading a length from it would run off into the next segment.
            0x01 | 0xD0..=0xD7 | 0xD8 => continue,
            _ => {}
        }

        let mut len_buf = [0u8; 2];
        read_exact_or_truncated(r, &mut len_buf)?;
        let seg_len = u16::from_be_bytes(len_buf) as usize;
        // The length field counts itself, so 0 or 1 would mean "no payload" and
        // would otherwise leave us seeking by zero forever.
        if seg_len < 2 {
            return Err(EmbedError::Malformed("jpeg segment length < 2"));
        }
        let payload_len = seg_len - 2;

        if id == 0xE2 {
            // APP2. Only read the tag prefix speculatively — an ICC chunk is
            // thousands of bytes and most APP2 segments are not ICC (XMP is
            // APP1, but Photoshop and friends do write other APP2 payloads).
            let probe_len = payload_len.min(ICC_PROFILE_TAG.len());
            if probe_len == ICC_PROFILE_TAG.len() {
                let mut tag = [0u8; ICC_PROFILE_TAG.len()];
                read_exact_or_truncated(r, &mut tag)?;
                if &tag == ICC_PROFILE_TAG {
                    if payload_len < ICC_PROFILE_TAG.len() + 2 {
                        return Err(EmbedError::Malformed("jpeg icc segment too short for seq/count"));
                    }
                    let mut seq = [0u8; 2];
                    read_exact_or_truncated(r, &mut seq)?;
                    let data_len = payload_len - ICC_PROFILE_TAG.len() - 2;
                    let mut data = vec![0u8; data_len];
                    read_exact_or_truncated(r, &mut data)?;
                    let (seq_no, count) = (seq[0], seq[1]);
                    match total {
                        None => total = Some(count),
                        Some(t) if t == count => {}
                        // Two APP2 ICC chains with different lengths: we cannot
                        // reassemble either one reliably.
                        Some(_) => {
                            return Err(EmbedError::Malformed(
                                "jpeg carries two icc chains with different chunk counts",
                            ))
                        }
                    }
                    chunks.push((seq_no, data));
                    continue;
                }
                // Not an ICC chunk: rewind over the tag we just read, since the
                // caller below seeks by the whole payload anyway.
                r.seek(SeekFrom::Current(-(ICC_PROFILE_TAG.len() as i64)))?;
            }
        }

        r.seek(SeekFrom::Current(payload_len as i64))?;
    }

    if chunks.is_empty() {
        return Ok(Embedded::Absent);
    }

    let total = total.unwrap_or(chunks.len() as u8) as usize;
    // A complete chain has exactly the chunks 1..=total, once each. Accepting a
    // partial chain would hand the engine a profile that is missing its tail —
    // and `validate_icc`'s size check would then reject it with a message about
    // ICC internals rather than about the missing chunks.
    if chunks.len() != total {
        return Err(EmbedError::Malformed("jpeg icc chunk count does not match the chunks present"));
    }
    chunks.sort_by_key(|(seq, _)| *seq);
    for (i, (seq, _)) in chunks.iter().enumerate() {
        if *seq as usize != i + 1 {
            return Err(EmbedError::Malformed("jpeg icc chunk sequence numbers are not 1..n"));
        }
    }

    let total_len: usize = chunks.iter().map(|(_, d)| d.len()).sum();
    let mut profile = Vec::with_capacity(total_len);
    for (_, d) in &chunks {
        profile.extend_from_slice(d);
    }
    validate_icc(&profile)?;
    Ok(Embedded::Present(profile))
}

// ── PNG ────────────────────────────────────────────────────────────────────

/// Walk PNG chunks looking for `iCCP`.
///
/// Stops at `IDAT`: the PNG spec requires chunks that affect rendering to
/// precede it, and a profile after it would not render anyway.
fn read_png<R: Read + Seek>(r: &mut R) -> Result<Embedded, EmbedError> {
    let mut sig = [0u8; 8];
    read_exact_or_truncated(r, &mut sig)?;
    loop {
        let mut hdr = [0u8; 8];
        read_exact_or_truncated(r, &mut hdr)?;
        let len = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
        let kind = [hdr[4], hdr[5], hdr[6], hdr[7]];

        if &kind == b"IDAT" || &kind == b"IEND" {
            return Ok(Embedded::Absent);
        }

        if &kind == b"iCCP" {
            // The chunk length is a u32 straight out of the file, so it can
            // claim up to 4 GB. Bound it before it reaches an allocation.
            if len > MAX_PROFILE_LEN {
                return Err(EmbedError::Malformed("png iccp chunk length is implausible"));
            }
            // keyword (Latin-1, NUL-terminated), compression method byte, then
            // the zlib stream. The keyword is a profile *name*; we read past it
            // rather than interpreting it, but we must find its terminator.
            let mut payload = vec![0u8; len];
            read_exact_or_truncated(r, &mut payload)?;
            // Skip the CRC (4 bytes). Not checked, and not a gap: the profile is
            // behind a zlib stream whose Adler-32 covers every decompressed
            // byte, so anything that changes the profile is caught by the
            // inflate. Measured by flipping every byte of the compressed data —
            // see `a_corrupted_png_profile_is_never_silently_returned`. A
            // hand-rolled CRC-32 table would only re-check what zlib already
            // covers.
            r.seek(SeekFrom::Current(4))?;

            let nul = payload
                .iter()
                .position(|&b| b == 0)
                .ok_or(EmbedError::Malformed("png iccp keyword is not NUL-terminated"))?;
            let method = *payload
                .get(nul + 1)
                .ok_or(EmbedError::Malformed("png iccp chunk has no compression-method byte"))?;
            if method != 0 {
                // Only compression method 0 (zlib/deflate) is defined. Anything
                // else is not a profile we can read, but it is also not a
                // malformed file — treat it as "no profile we can use" rather
                // than failing the whole scan.
                return Ok(Embedded::Absent);
            }
            let compressed = &payload[nul + 2..];
            let profile = inflate(compressed)?;
            validate_icc(&profile)?;
            return Ok(Embedded::Present(profile));
        }

        // Chunk payload + CRC.
        r.seek(SeekFrom::Current(len as i64 + 4))?;
    }
}

/// zlib-inflate a PNG `iCCP` payload.
///
/// Bounded on purpose: the chunk length is attacker-influenced data, and an
/// ICC profile is spec-bounded in practice but not by the container parser.
/// The cap is far above any real profile (the ICC spec itself suggests profiles
/// stay well under a megabyte) and exists so a hostile or corrupt file cannot
/// make c41 allocate without limit.
const MAX_PROFILE_LEN: usize = 64 * 1024 * 1024;

fn inflate(compressed: &[u8]) -> Result<Vec<u8>, EmbedError> {
    use std::io::Read as _;
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(compressed)
        .take(MAX_PROFILE_LEN as u64 + 1)
        .read_to_end(&mut out)?;
    if out.len() > MAX_PROFILE_LEN {
        return Err(EmbedError::Malformed("png iccp inflates to an implausible size"));
    }
    Ok(out)
}

// ── TIFF ───────────────────────────────────────────────────────────────────

/// Byte order of a TIFF, plus offset-addressed read helpers.
///
/// The readers take an absolute offset and seek to it. An earlier draft read
/// sequentially and merely *accepted* an offset argument, which is correct only
/// as long as every field happens to be visited in ascending order — a real
/// trap, since an IFD walk skips over entries whose tags we do not want.
struct Tiff<R> {
    r: R,
    /// `true` when the file is big-endian (`MM`).
    be: bool,
}

impl<R: Read + Seek> Tiff<R> {
    fn u16(&mut self, off: u64) -> Result<u16, EmbedError> {
        self.r.seek(SeekFrom::Start(off))?;
        let mut b = [0u8; 2];
        read_exact_or_truncated(&mut self.r, &mut b)?;
        Ok(if self.be { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
    }

    fn u32(&mut self, off: u64) -> Result<u32, EmbedError> {
        self.r.seek(SeekFrom::Start(off))?;
        let mut b = [0u8; 4];
        read_exact_or_truncated(&mut self.r, &mut b)?;
        Ok(if self.be { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) })
    }
}

/// Walk TIFF IFDs looking for `InterColorProfile` (tag 34675).
///
/// Searches IFD0 and follows the next-IFD chain so multi-page TIFFs are covered
/// on every page. SubIFDs are deliberately not searched: ICC there is
/// non-standard and darktable does not look.
fn read_tiff<R: Read + Seek>(r: &mut R) -> Result<Embedded, EmbedError> {
    let mut magic = [0u8; 2];
    read_exact_or_truncated(r, &mut magic)?;
    let be = match &magic {
        b"II" => false,
        b"MM" => true,
        _ => return Err(EmbedError::Malformed("tiff byte-order mark is neither II nor MM")),
    };
    let mut t = Tiff { r, be };
    let magic_num = t.u16(2)?;
    if magic_num != 42 {
        return Err(EmbedError::Malformed("tiff magic is not 42"));
    }

    // Offset of IFD0, then the whole chain from there.
    let mut next = u64::from(t.u32(4)?);
    // A malformed file could point IFD offsets in a circle; bound the walk
    // rather than trusting the file not to.
    let mut seen = 0usize;
    while next != 0 {
        seen += 1;
        if seen > 1024 {
            return Err(EmbedError::Malformed("tiff ifd chain does not terminate"));
        }
        let count = u64::from(t.u16(next)?);
        for i in 0..count {
            let entry = next + 2 + i * 12;
            let tag = t.u16(entry)?;
            if tag != TIFF_TAG_INTERCOLORPROFILE {
                continue;
            }
            let ty = t.u16(entry + 2)?;
            let n = t.u32(entry + 4)? as usize;
            let value_raw = t.u32(entry + 8)?;
            if ty != TIFF_TYPE_UNDEFINED {
                return Err(EmbedError::Malformed("tiff icc tag is not of type UNDEFINED"));
            }
            if n == 0 {
                return Err(EmbedError::Malformed("tiff icc tag has zero length"));
            }
            // `n` is a u32 out of the file and feeds an allocation directly.
            if n > MAX_PROFILE_LEN {
                return Err(EmbedError::Malformed("tiff icc tag length is implausible"));
            }
            // A value that fits in four bytes is stored inline, left-justified
            // in the entry's value field; otherwise that field is a file offset.
            let (offset, len) = if n <= 4 {
                (entry + 8, n)
            } else {
                (u64::from(value_raw), n)
            };
            let mut profile = vec![0u8; len];
            t.r.seek(SeekFrom::Start(offset))?;
            read_exact_or_truncated(&mut t.r, &mut profile)?;
            validate_icc(&profile)?;
            return Ok(Embedded::Present(profile));
        }
        next = u64::from(t.u32(next + 2 + count * 12)?);
    }
    Ok(Embedded::Absent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icc::transform::tests::test_helpers::{
        build_profile, curv_gamma_tag, srgb_like_profile, xyz_tag,
    };
    use crate::icc::{Profile, Transform};

    /// A minimal valid ICC profile to wrap in containers. `build_profile`
    /// writes the declared size, the `acsp` magic and a padded tag table, so it
    /// satisfies [`validate_icc`] and is byte-identical to what the engine's
    /// own tests parse.
    fn some_profile(tag_byte: u8) -> Vec<u8> {
        build_profile(
            b"mntr",
            b"RGB ",
            b"XYZ ",
            &[(b"desc", vec![tag_byte; 4]), (b"wtpt", vec![0; 20])],
        )
    }

    fn extract_bytes(bytes: &[u8]) -> Result<Embedded, EmbedError> {
        extract(&mut std::io::Cursor::new(bytes.to_vec()))
    }

    // ── container builders ──────────────────────────────────────────────

    /// A JPEG carrying `profile` in one `APP2` chunk.
    fn jpeg(profile: &[u8]) -> Vec<u8> {
        let mut seg = ICC_PROFILE_TAG.to_vec();
        seg.push(1); // sequence 1
        seg.push(1); // of 1
        seg.extend_from_slice(profile);
        let mut out = vec![0xFF, 0xD8]; // SOI
        out.push(0xFF);
        out.push(0xE2); // APP2
        out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&seg);
        out.push(0xFF);
        out.push(0xDA); // SOS — header ends here
        out.extend_from_slice(&[0xAA; 16]); // entropy-coded "data"
        out
    }

    /// A PNG carrying `profile` in an `iCCP` chunk.
    fn png(profile: &[u8]) -> Vec<u8> {
        let mut payload = b"ICC profile\0".to_vec();
        payload.push(0); // compression method: deflate
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        use std::io::Write as _;
        enc.write_all(profile).unwrap();
        payload.extend_from_slice(&enc.finish().unwrap());

        let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        out.extend_from_slice(b"iCCP");
        out.extend_from_slice(&payload);
        out.extend_from_slice(&[0; 4]); // CRC (not checked)
        out.extend_from_slice(&0u32.to_be_bytes()); // IDAT length
        out.extend_from_slice(b"IDAT");
        out
    }

    /// A TIFF carrying `profile` in an IFD0 entry for tag 34675, **out-of-line**
/// (the value field holds a file offset). The inline form cannot be built by
/// this helper: a profile is at least its own 128-byte header, so a count that
/// fits in four bytes can never hold one —
/// `a_tiff_icc_tag_too_short_to_be_a_profile_is_rejected` covers that path.
    fn tiff(profile: &[u8], be: bool) -> Vec<u8> {
        let u16w = |v: u16| if be { v.to_be_bytes() } else { v.to_le_bytes() };
        let u32w = |v: u32| if be { v.to_be_bytes() } else { v.to_le_bytes() };

        // Byte order, magic, IFD0 offset.
        let header_len = 8u32;
        // One entry (12) + count (2) + next (4).
        let ifd_len = 2 + 12 + 4;
        let data_off = header_len + ifd_len;

        let mut out = Vec::new();
        if be {
            out.extend_from_slice(b"MM");
        } else {
            out.extend_from_slice(b"II");
        }
        out.extend_from_slice(&u16w(42));
        out.extend_from_slice(&u32w(header_len));
        out.extend_from_slice(&u16w(1)); // one entry
        out.extend_from_slice(&u16w(TIFF_TAG_INTERCOLORPROFILE));
        out.extend_from_slice(&u16w(TIFF_TYPE_UNDEFINED));
        out.extend_from_slice(&u32w(profile.len() as u32));
        out.extend_from_slice(&u32w(data_off)); // out-of-line offset
        out.extend_from_slice(&u32w(0)); // no next IFD
        out.extend_from_slice(profile);
        out
    }

    // ── the tests ───────────────────────────────────────────────────────

    #[test]
    fn finds_a_profile_in_all_three_containers() {
        let p = some_profile(1);
        assert_eq!(extract_bytes(&jpeg(&p)).unwrap(), Embedded::Present(p.clone()));
        assert_eq!(extract_bytes(&png(&p)).unwrap(), Embedded::Present(p.clone()));
        assert_eq!(extract_bytes(&tiff(&p, false)).unwrap(), Embedded::Present(p.clone()));
        assert_eq!(extract_bytes(&tiff(&p, true)).unwrap(), Embedded::Present(p));
    }

    #[test]
    fn a_container_with_no_profile_is_absent_not_unsupported() {
        // JPEG: SOI straight to SOS.
        let j = vec![0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x00];
        assert_eq!(extract_bytes(&j).unwrap(), Embedded::Absent);
        // PNG: signature then IDAT.
        let mut p = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(b"IDAT");
        assert_eq!(extract_bytes(&p).unwrap(), Embedded::Absent);
        // TIFF: an IFD0 with one unrelated tag.
        let t = {
            let mut out = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
            out.extend_from_slice(&1u16.to_le_bytes()); // one entry
            out.extend_from_slice(&256u16.to_le_bytes()); // tag 256, not 34675
            out.extend_from_slice(&3u16.to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
            out
        };
        assert_eq!(extract_bytes(&t).unwrap(), Embedded::Absent);
    }

    #[test]
    fn an_unknown_container_is_unsupported_not_absent() {
        // The distinction the module exists to preserve: we did not check this
        // file, we are not able to.
        for bytes in [
            b"GIF89a\0\0".to_vec(),               // GIF
            b"RIFF\0\0\0\0WEBP".to_vec(),         // WebP
            vec![0x89],                           // one byte
            Vec::new(),                           // empty
        ] {
            assert_eq!(
                extract_bytes(&bytes).unwrap(),
                Embedded::UnsupportedFormat,
                "{bytes:02x?} should sniff as unsupported"
            );
        }
    }

    #[test]
    fn reassembles_a_profile_split_across_jpeg_chunks_out_of_order() {
        // Real files split ICC above ~64 KB. Build a profile big enough that
        // splitting is the interesting case, then hand the chunks over in
        // reverse file order with sequence numbers intact — the reassembly must
        // sort by sequence, not trust file order.
        let p = some_profile(7);
        let (a, b) = p.split_at(p.len() / 2);
        let mut out = vec![0xFF, 0xD8];
        // chunk 2 first
        for (seq, data) in [(2u8, b), (1u8, a)] {
            let mut seg = ICC_PROFILE_TAG.to_vec();
            seg.push(seq);
            seg.push(2);
            seg.extend_from_slice(data);
            out.push(0xFF);
            out.push(0xE2);
            out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
            out.extend_from_slice(&seg);
        }
        out.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x00]);
        assert_eq!(extract_bytes(&out).unwrap(), Embedded::Present(p));
    }

    #[test]
    fn rejects_a_jpeg_icc_chain_that_is_missing_a_chunk() {
        // Declares 2 chunks, supplies 1. Handing the engine the surviving half
        // would fail validate_icc with a message about ICC internals rather
        // than about the actual problem, so it is caught here instead.
        let p = some_profile(3);
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xE2];
        let mut seg = ICC_PROFILE_TAG.to_vec();
        seg.push(1); // sequence 1 ...
        seg.push(2); // ... but claims to be 1 of 2
        seg.extend_from_slice(&p);
        out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&seg);
        out.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x00]);
        assert!(matches!(extract_bytes(&out), Err(EmbedError::Malformed(_))));
    }

    #[test]
    fn rejects_a_truncated_profile_even_when_the_signature_survives() {
        // This is the case a bare `acsp` check would miss: drop the tail, so
        // the magic at offset 36 is intact but the size field disagrees with
        // what we extracted.
        let p = some_profile(5);
        let truncated = &p[..p.len() - 16];
        let mut j = jpeg(truncated);
        j.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x00]);
        assert!(matches!(extract_bytes(&j), Err(EmbedError::Malformed(_))));
    }

    #[test]
    fn does_not_scan_entropy_coded_data_for_profiles() {
        // An APP2/ICC chunk placed after SOS must not be picked up: ICC is a
        // header-only construct, and scanning past SOS risks matching
        // marker-like bytes inside the compressed stream.
        let p = some_profile(9);
        let mut seg = ICC_PROFILE_TAG.to_vec();
        seg.push(1);
        seg.push(1);
        seg.extend_from_slice(&p);
        let mut f = vec![0xFF, 0xD8];
        f.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x02]); // SOS
        f.push(0xFF);
        f.push(0xD9); // EOI
        f.extend_from_slice(&[0xFF, 0xE2]); // a convincing fake, too late
        f.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
        f.extend_from_slice(&seg);
        assert_eq!(extract_bytes(&f).unwrap(), Embedded::Absent);
    }

    #[test]
    fn a_tiff_profile_at_an_unaligned_offset_is_still_read() {
        // The out-of-line offset in the entry need not be word-aligned; TIFF
        // only *recommends* it. `tiff()` puts the data straight after the IFD
        // (offset 26); shifting it by 0..=3 must make no difference, which also
        // pins the entry's value-field offset at entry+8.
        let p = some_profile(11);
        let ifd_end = 8 + 2 + 12 + 4; // header + count + one entry + next-IFD
        for pad in [0usize, 1, 2, 3] {
            let mut out = tiff(&p, false)[..ifd_end].to_vec();
            out.extend(std::iter::repeat_n(0u8, pad));
            let value_field = 8 + 2 + 8; // IFD0 + count(2) + entry's own header(8)
            out[value_field..value_field + 4]
                .copy_from_slice(&((ifd_end + pad) as u32).to_le_bytes());
            out.extend_from_slice(&p);
            assert_eq!(extract_bytes(&out).unwrap(), Embedded::Present(p.clone()), "pad={pad}");
        }
    }

    #[test]
    fn rejects_a_blob_of_the_right_size_that_is_not_an_icc_profile() {
        // The size field agreeing is not enough on its own: a container can
        // point at a blob of the right length that is not a profile at all
        // (a PNG whose `iCCP` inflates to something else, a TIFF tag holding
        // arbitrary bytes). Without the `acsp` check this would reach the
        // engine, which would report an unsupported profile — blaming the
        // user's file for our bad parse.
        let mut p = some_profile(6);
        p[36..40].copy_from_slice(b"XXXX");
        let mut j = jpeg(&p);
        j.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x00]);
        assert!(matches!(extract_bytes(&j), Err(EmbedError::Malformed(_))));
    }

    #[test]
    fn an_absurd_declared_length_is_refused_before_anything_is_allocated() {
        // Both containers take a length straight out of the file: a PNG chunk
        // length and a TIFF tag count are each a u32, so a hostile or corrupt
        // file can ask for a 4 GB allocation with an 8-byte header. These
        // fixtures declare the maximum and supply no payload, so the guard has
        // to fire *before* the allocation — if it did not, the test process
        // would try to allocate 4 GB rather than fail cleanly.
        let mut png_huge = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        png_huge.extend_from_slice(&u32::MAX.to_be_bytes());
        png_huge.extend_from_slice(b"iCCP");
        assert!(matches!(extract_bytes(&png_huge), Err(EmbedError::Malformed(_))));

        let mut tiff_huge = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
        tiff_huge.extend_from_slice(&1u16.to_le_bytes()); // one entry
        tiff_huge.extend_from_slice(&TIFF_TAG_INTERCOLORPROFILE.to_le_bytes());
        tiff_huge.extend_from_slice(&TIFF_TYPE_UNDEFINED.to_le_bytes());
        tiff_huge.extend_from_slice(&u32::MAX.to_le_bytes()); // count: 4 GB
        tiff_huge.extend_from_slice(&0u32.to_le_bytes());
        tiff_huge.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        assert!(matches!(extract_bytes(&tiff_huge), Err(EmbedError::Malformed(_))));
    }

    #[test]
    fn a_tiff_icc_tag_too_short_to_be_a_profile_is_rejected() {
        // The inline branch (count <= 4, data in the value field) is
        // unreachable for a real profile — the smallest possible ICC profile is
        // its own 128-byte header — but a corrupt tag can claim count=4. It
        // must fail validation rather than hand back four bytes as a "profile".
        let mut t = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
        t.extend_from_slice(&1u16.to_le_bytes()); // one entry
        t.extend_from_slice(&TIFF_TAG_INTERCOLORPROFILE.to_le_bytes());
        t.extend_from_slice(&TIFF_TYPE_UNDEFINED.to_le_bytes());
        t.extend_from_slice(&4u32.to_le_bytes()); // count: 4 bytes, inline
        t.extend_from_slice(b"acsp"); // inline value
        t.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        assert!(matches!(extract_bytes(&t), Err(EmbedError::Malformed(_))));
    }

    #[test]
    fn a_png_iccp_with_an_undefined_compression_method_is_absent_not_an_error() {
        // Compression method != 0 is not a broken file, it is a profile we
        // cannot read. Reported as Absent so the caller still gets to ask.
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let payload = b"ICC profile\0\x09junk".to_vec();
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        out.extend_from_slice(b"iCCP");
        out.extend_from_slice(&payload);
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(b"IDAT");
        assert_eq!(extract_bytes(&out).unwrap(), Embedded::Absent);
    }

    #[test]
    fn a_truncated_container_is_reported_rather_than_panicking() {
        // Every one of these ends mid-structure. None may panic — the walk
        // has to bottom out in Truncated or Malformed.
        let p = some_profile(2);
        let cases: Vec<Vec<u8>> = vec![
            { let mut v = jpeg(&p); v.truncate(v.len() - 20); v },
            { let mut v = png(&p); v.truncate(v.len() - 30); v },
            { let mut v = tiff(&p, false); v.truncate(v.len() - 40); v },
            { let mut v = tiff(&p, true); v.truncate(20); v },
            vec![0xFF, 0xD8, 0xFF],
            { let mut v = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]; v.push(0); v },
        ];
        for c in cases {
            let r = extract_bytes(&c);
            assert!(
                matches!(r, Err(EmbedError::Truncated) | Err(EmbedError::Malformed(_))),
                "{c:02x?} produced {r:?}"
            );
        }
    }

    #[test]
    fn a_zero_length_jpeg_segment_does_not_loop_forever() {
        // Length field 0 would mean "seek forward by zero" and never advance.
        let j = [0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x00];
        assert!(matches!(extract_bytes(&j), Err(EmbedError::Malformed(_))));
    }

    #[test]
    fn a_circular_tiff_ifd_chain_terminates() {
        // Point the next-IFD offset back at IFD0 forever.
        let mut t = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
        t.extend_from_slice(&0u16.to_le_bytes()); // zero entries
        t.extend_from_slice(&8u32.to_le_bytes()); // next IFD -> IFD0 again
        assert!(matches!(extract_bytes(&t), Err(EmbedError::Malformed(_))));
    }

    #[test]
    fn extract_file_round_trips_through_the_filesystem() {
        let p = some_profile(4);
        let dir = std::env::temp_dir().join(format!("c41-icc-embed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.jpg");
        std::fs::write(&path, jpeg(&p)).unwrap();
        assert_eq!(extract_file(&path).unwrap(), Embedded::Present(p));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupted_png_profile_is_never_silently_returned() {
        // This is why the PNG chunk CRC is skipped rather than checked. The
        // profile bytes inside `iCCP` sit behind a zlib stream, and zlib's own
        // Adler-32 covers every one of the *decompressed* bytes — so corruption
        // that changes the profile is caught by the inflate, and corruption that
        // breaks the stream stops it outright. Measured over a sweep of
        // single-bit flips in the compressed data: 419 of 420 came back `Err`.
        //
        // The one survivor was a flip in the zlib stream's second header byte,
        // whose `FLEVEL` bits are excluded from the header's `% 31` check by
        // design and do not affect the decoded output — so returning the same
        // profile is the *correct* answer there, not a miss. That is why this
        // test asserts the invariant that actually matters (never silently hand
        // back a *different* profile) instead of demanding that every flip
        // error, which would be false.
        let p = some_profile(0x5C);
        let good = png(&p);
        assert_eq!(extract_bytes(&good).unwrap(), Embedded::Present(p.clone()));

        // PNG layout: signature(8) length(4) "iCCP"(4) payload(length) …
        let payload_len = u32::from_be_bytes([good[8], good[9], good[10], good[11]]) as usize;
        let payload_start = 8 + 4 + 4;
        let zlib_start = payload_start + 14; // keyword (13) + compression-method byte
        let zlib_end = payload_start + payload_len;
        assert!(zlib_end <= good.len() && zlib_start + 2 < zlib_end);

        for at in zlib_start..zlib_end {
            for bit in [0x01u8, 0x40, 0xFF] {
                let mut bad = good.clone();
                bad[at] ^= bit;
                match extract_bytes(&bad) {
                    // Detected: the checksum or the stream structure refused it.
                    Err(_) => {}
                    // Not detected — acceptable only if the profile is unchanged.
                    Ok(Embedded::Present(b)) => assert_eq!(
                        b, p,
                        "corrupting zlib byte {at} by {bit:#x} silently yielded a different profile"
                    ),
                    Ok(other) => panic!("corrupted iCCP unexpectedly gave {other:?}"),
                }
            }
        }
    }

    #[test]
    fn a_jpeg_marker_preceded_by_fill_bytes_is_still_read() {
        // Encoders are allowed to pad a marker with any number of 0xFF bytes.
        // An earlier draft scanned for markers two bytes at a time, which
        // desynchronised on `FF FF E2` and then walked into the payload as if it
        // were segments; the scan is byte-wise now, and this pins it.
        let p = some_profile(0x11);
        let mut seg = ICC_PROFILE_TAG.to_vec();
        seg.push(1);
        seg.push(1);
        seg.extend_from_slice(&p);
        for pad in [1usize, 2, 3] {
            let mut j = vec![0xFF, 0xD8];
            j.extend(std::iter::repeat_n(0xFF, pad));
            j.push(0xE2);
            j.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
            j.extend_from_slice(&seg);
            j.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x00]);
            assert_eq!(extract_bytes(&j).unwrap(), Embedded::Present(p.clone()), "pad={pad}");
        }
    }

    #[test]
    fn a_tiff_profile_in_a_subifd_is_deliberately_not_found() {
        // Locking in a deliberate omission. ICC in a SubIFD is non-standard and
        // darktable does not look there, so following tag 330 would be a
        // behaviour *change*, not a fix. This test exists so that a future
        // reader who helpfully "fixes" it has to decide to change this.
        let p = some_profile(0x22);
        let ifd0 = 8u32;
        let ifd0_len = 2 + 2 * 12 + 4; // two entries
        let subifd = ifd0 + ifd0_len;
        let subifd_len = 2 + 12 + 4;
        let data = subifd + subifd_len;

        let mut t = Vec::new();
        t.extend_from_slice(b"II");
        t.extend_from_slice(&42u16.to_le_bytes());
        t.extend_from_slice(&ifd0.to_le_bytes());
        // IFD0: a SubIFDs pointer (330) plus an unrelated tag.
        t.extend_from_slice(&2u16.to_le_bytes());
        t.extend_from_slice(&330u16.to_le_bytes());
        t.extend_from_slice(&4u16.to_le_bytes()); // LONG
        t.extend_from_slice(&1u32.to_le_bytes());
        t.extend_from_slice(&subifd.to_le_bytes());
        t.extend_from_slice(&256u16.to_le_bytes());
        t.extend_from_slice(&3u16.to_le_bytes());
        t.extend_from_slice(&1u32.to_le_bytes());
        t.extend_from_slice(&0u32.to_le_bytes());
        t.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        // The SubIFD: holds the ICC profile we are *not* going to report.
        t.extend_from_slice(&1u16.to_le_bytes());
        t.extend_from_slice(&TIFF_TAG_INTERCOLORPROFILE.to_le_bytes());
        t.extend_from_slice(&TIFF_TYPE_UNDEFINED.to_le_bytes());
        t.extend_from_slice(&(p.len() as u32).to_le_bytes());
        t.extend_from_slice(&data.to_le_bytes());
        t.extend_from_slice(&0u32.to_le_bytes());
        t.extend_from_slice(&p);

        assert_eq!(extract_bytes(&t).unwrap(), Embedded::Absent);
    }

    #[test]
    fn a_multi_page_tiff_reports_the_first_ifd_that_has_a_profile() {
        // Documents the policy the next-IFD walk implements: a file-level answer
        // for a possibly per-page question. Page 1 carries no profile, page 2
        // does, and the whole file is described by page 2's — which is the only
        // option available without threading page identity through the decode
        // path, and the right one for a whole-image colour decision.
        let p = some_profile(0x33);
        let ifd0 = 8u32;
        let ifd0_len = 2 + 12 + 4;
        let ifd1 = ifd0 + ifd0_len;
        let ifd1_len = 2 + 12 + 4;
        let data = ifd1 + ifd1_len;

        let mut t = Vec::new();
        t.extend_from_slice(b"II");
        t.extend_from_slice(&42u16.to_le_bytes());
        t.extend_from_slice(&ifd0.to_le_bytes());
        // IFD0: unrelated tag, and a next-IFD pointer to IFD1.
        t.extend_from_slice(&1u16.to_le_bytes());
        t.extend_from_slice(&256u16.to_le_bytes());
        t.extend_from_slice(&3u16.to_le_bytes());
        t.extend_from_slice(&1u32.to_le_bytes());
        t.extend_from_slice(&0u32.to_le_bytes());
        t.extend_from_slice(&ifd1.to_le_bytes());
        // IFD1: the ICC profile.
        t.extend_from_slice(&1u16.to_le_bytes());
        t.extend_from_slice(&TIFF_TAG_INTERCOLORPROFILE.to_le_bytes());
        t.extend_from_slice(&TIFF_TYPE_UNDEFINED.to_le_bytes());
        t.extend_from_slice(&(p.len() as u32).to_le_bytes());
        t.extend_from_slice(&data.to_le_bytes());
        t.extend_from_slice(&0u32.to_le_bytes());
        t.extend_from_slice(&p);

        assert_eq!(extract_bytes(&t).unwrap(), Embedded::Present(p));
    }

    #[test]
    fn a_grayscale_profile_is_present_but_the_engine_refuses_it() {
        // The concrete case behind keeping `Present` and `Absent` distinct, and
        // the reason this module does not try to decide usability itself.
        //
        // A grayscale photo's file really does carry a complete, valid ICC
        // profile, and extraction finds it — that is a fact about the container
        // and it is true. It is *also* a profile this engine cannot build a
        // transform from, because every pipeline stage has to be 3-channel and a
        // GRAY profile's is not. Both halves are asserted here so the module
        // docs' claim is checked by the test suite rather than trusted: if
        // someone later makes the engine accept GRAY, this test says so.
        let d50 = [0.9642f32, 1.0, 0.8249];
        let gray = build_profile(
            b"mntr",
            b"GRAY",
            b"XYZ ",
            &[(b"kTRC", curv_gamma_tag(2.2)), (b"wtpt", xyz_tag(d50[0], d50[1], d50[2]))],
        );
        let srgb = srgb_like_profile(b"XYZ ", d50);

        // The engine parses it fine — it is a well-formed profile.
        let gray_p = Profile::parse(&gray).expect("grayscale profile should parse");
        Profile::parse(&srgb).expect("srgb profile should parse");

        // Extraction reports it present, in every container.
        assert_eq!(extract_bytes(&jpeg(&gray)).unwrap(), Embedded::Present(gray.clone()));
        assert_eq!(extract_bytes(&png(&gray)).unwrap(), Embedded::Present(gray.clone()));
        assert_eq!(
            extract_bytes(&tiff(&gray, false)).unwrap(),
            Embedded::Present(gray.clone())
        );

        // And the transform engine refuses it, for the documented reason.
        assert!(Transform::new(&gray_p, &Profile::parse(&srgb).unwrap(), 0).is_err());
    }

    // ── real-world corpus ───────────────────────────────────────────────

    /// Cross-check against files written by *other* encoders, driven by a
    /// manifest of expectations.
    ///
    /// The tests above all build their fixtures with this module's own idea of
    /// the container layout, so they cannot catch a misreading of the spec — a
    /// builder and a parser that share a misunderstanding agree perfectly. This
    /// test is the escape hatch: point `C41_ICC_CORPUS` at a directory holding
    /// real images plus a `manifest.tsv` whose columns are
    /// `file`, `present|absent|unsupported`, and the sha256 of the expected
    /// profile bytes. Skipped when the variable is unset, so CI stays hermetic.
    ///
    /// The manifest for the corpus this was developed against was generated
    /// with ImageMagick, whose `convert FILE icc:OUT` reads the embedded
    /// profile back byte-for-byte through lcms — the same library darktable
    /// itself uses. That makes the expectation genuinely independent of this
    /// parser, and it covered the case the hand-built fixtures most needed
    /// help with: a 122 KB profile in a JPEG, split across `APP2` segments.
    #[test]
    fn matches_a_real_world_corpus_manifest() {
        let Ok(dir) = std::env::var("C41_ICC_CORPUS") else {
            eprintln!("skipping corpus check: C41_ICC_CORPUS not set");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let manifest = dir.join("manifest.tsv");
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("reading {}: {e}", manifest.display()));

        let mut checked = 0usize;
        for (lineno, line) in text.lines().enumerate().skip(1) {
            if line.trim().is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            assert!(cols.len() == 3, "manifest line {}: expected 3 columns, got {cols:?}", lineno + 1);
            let (name, expect, hash) = (cols[0], cols[1], cols[2]);
            let path = dir.join(name);
            let got = extract_file(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

            match expect {
                "present" => {
                    let Embedded::Present(bytes) = got else {
                        panic!("{name}: expected a profile, got {got:?}");
                    };
                    use sha2::{Digest, Sha256};
                    let digest = format!("{:x}", Sha256::digest(&bytes));
                    assert_eq!(&digest, hash, "{name}: extracted profile differs from the manifest");
                }
                "absent" => assert_eq!(got, Embedded::Absent, "{name}"),
                "unsupported" => assert_eq!(got, Embedded::UnsupportedFormat, "{name}"),
                other => panic!("manifest line {}: unknown verdict {other:?}", lineno + 1),
            }
            checked += 1;
        }
        assert!(checked > 0, "corpus manifest was empty — that would pass vacuously");
        eprintln!("corpus: {checked} files matched");
    }

    /// Deterministic mutation sweep: no byte pattern may panic the scan.
    ///
    /// A parser like this one runs on whatever the user opens, including files
    /// truncated by a failed copy or mangled by a failing card. `read_exact`
    /// turns EOF into an error, but the walks also *seek*, take lengths out of
    /// the file and loop, and those are where a parser panics: an allocation of
    /// `count` bytes, an index past a buffer, an IFD chain that never ends.
    ///
    /// Random fuzzing would be non-reproducible, so this uses a fixed-seed xorshift
    /// and a fixed round count: a failure here is a failure anyone can reproduce.
    /// Seeds are this module's own containers, so the sweep runs in CI with no
    /// corpus; when `C41_ICC_CORPUS` is set those real files are swept too.
    #[test]
    fn mutation_sweep_never_panics() {
        let p = some_profile(0xA5);
        let mut seeds: Vec<Vec<u8>> = vec![
            jpeg(&p),
            png(&p),
            tiff(&p, false),
            tiff(&p, true),
            vec![0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x00],
            vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
            vec![b'I', b'I', 42, 0, 8, 0, 0, 0],
        ];
        if let Ok(dir) = std::env::var("C41_ICC_CORPUS") {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_file() {
                    if let Ok(bytes) = std::fs::read(&path) {
                        seeds.push(bytes);
                    }
                }
            }
        }

        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // Unmutated pass over every seed, so the sweep also covers the exact
        // bytes a real file arrives as.
        for seed in &seeds {
            let _ = extract_bytes(seed);
        }

        // Systematic truncation: every seed cut at 64 evenly spaced points, plus
        // every single-byte cut. This is the shape of a failed/interrupted
        // write, and it is the one systematic path random mutation reaches only
        // by luck.
        for seed in &seeds {
            let step = (seed.len() / 64).max(1);
            for cut in (0..seed.len()).step_by(step) {
                let _ = extract_bytes(&seed[..cut]);
            }
        }

        // Random mutation: a handful of byte overwrites, and whole regions of
        // 0xFF / 0x00 to imitate the marker-like runs that desynchronise walks.
        for _ in 0..4000usize {
            let seed = &seeds[next() as usize % seeds.len()];
            let mut v = seed.clone();
            if v.is_empty() {
                continue;
            }
            let flips = 1 + (next() % 6) as usize;
            for _ in 0..flips {
                let at = (next() as usize) % v.len();
                v[at] = (next() & 0xFF) as u8;
            }
            // A contiguous run of one repeated byte, which in a JPEG means a
            // marker-looking stretch and in a PNG means a zero length field.
            if next() % 3 == 0 {
                let len = v.len();
                let start = (next() as usize) % len;
                let fill = if next() % 2 == 0 { 0xFF } else { 0x00 };
                let span = 1 + (next() as usize % 32);
                for k in 0..span {
                    v[(start + k) % len] = fill;
                }
            }
            let _ = extract_bytes(&v);
        }
    }
}
