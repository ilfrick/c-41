//! Import-time EXIF probe (m4-135, parity 2.6 slice 3) — the four numeric
//! properties darktable's collection rules filter on: exposure (s), aperture
//! (f-number), ISO, focal length (mm).
//!
//! Scope: read the four standard EXIF tags out of whatever container the file
//! carries. Most raws are TIFF-family (ORF/NEF/CR2/ARW/RAF/DNG…), where these
//! live in IFD0 / ExifIFD; kamadak-exif walks that structure itself. Files with
//! unreadable or absent tags yield `None` fields — a missing value must never
//! invent one, because the values land in the catalogue and later feed numeric
//! rules (`exposure < 1`) where an invented 0 would silently match.
//!
//! Deviation from darktable recorded in PARITY_AUDIT: dt reads via exiv2 over
//! its whole format matrix at import; we cover what kamadak-exif can parse and
//! leave NULL elsewhere. Numeric rule semantics treat NULL as not-matching
//! (see `rule_stack`), so unprobed images stay out of numeric-rule results.

use exif::{In, Tag, Value};
use std::path::Path;

/// The import-relevant subset. All fields optional independently: a file can
/// carry exposure but no focal length (scans, some compacts).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExifMeta {
    /// Shutter duration in seconds (ExposureTime).
    pub exposure: Option<f64>,
    /// F-number (FNumber).
    pub aperture: Option<f64>,
    /// ISO sensitivity (ISOSpeedRatings / PhotographicSensitivity).
    pub iso: Option<f64>,
    /// True focal length in mm (FocalLength — not the 35mm-equivalent).
    pub focal_length: Option<f64>,
}

impl ExifMeta {
    pub const NONE: ExifMeta = ExifMeta {
        exposure: None,
        aperture: None,
        iso: None,
        focal_length: None,
    };
}

fn rational_f64(field: Option<&exif::Field>) -> Option<f64> {
    match field?.value {
        Value::Rational(ref r) if !r.is_empty() => Some(r[0].to_f64()),
        // Some writers put SRational there; accept it too rather than drop the
        // tag on the floor (a negative would be nonsense for our four, so the
        // sign bit just never shows up in practice).
        Value::SRational(ref r) if !r.is_empty() => Some(r[0].to_f64()),
        _ => None,
    }
}

fn short_u16(field: Option<&exif::Field>) -> Option<f64> {
    let v = &field?.value;
    match v {
        Value::Short(ref s) if !s.is_empty() => Some(f64::from(s[0])),
        Value::Long(ref l) if !l.is_empty() => Some(f64::from(l[0])),
        // ISO is Short almost everywhere; Byte/Long appear in exotic writers.
        Value::Byte(ref b) if !b.is_empty() => Some(f64::from(b[0])),
        Value::Rational(ref r) if !r.is_empty() => Some(r[0].to_f64()),
        _ => None,
    }
}

/// Read [`ExifMeta`] from `path`. `None` when the container can't be parsed at
/// all (not a TIFF family file, truncated, I/O error); per-field `None` when
/// the tag is simply absent.
pub fn probe(path: &Path) -> Option<ExifMeta> {
    let file = std::fs::File::open(path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut std::io::BufReader::new(file))
        .ok()?;
    Some(ExifMeta {
        exposure: rational_f64(exif.get_field(Tag::ExposureTime, In::PRIMARY)),
        aperture: rational_f64(exif.get_field(Tag::FNumber, In::PRIMARY)),
        iso: short_u16(exif.get_field(Tag::ISOSpeed, In::PRIMARY)),
        focal_length: rational_f64(exif.get_field(Tag::FocalLength, In::PRIMARY)),
    })
}

/// Build an all-`None` meta — the importer's fallback for unreadable files.
pub fn probe_or_none(path: &Path) -> ExifMeta {
    probe(path).unwrap_or(ExifMeta::NONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_meta_is_all_none() {
        assert_eq!(ExifMeta::NONE, probe_or_none(Path::new("/nonexistent/file.ORF")));
    }
}
