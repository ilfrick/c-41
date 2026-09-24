//! Import-time EXIF probe (m4-135, parity 2.6 slice 3) — the four numeric
//! properties darktable's collection rules filter on: exposure (s), aperture
//! (f-number), ISO, focal length (mm) — plus the geotagging triple (u2,
//! parity 2.7 leg): latitude, longitude, altitude, stored in darktable's own
//! `main.images` columns of the same names.
//!
//! Scope: read the standard EXIF tags out of whatever container the file
//! carries. Most raws are TIFF-family (ORF/NEF/CR2/ARW/RAF/DNG…), where these
//! live in IFD0 / ExifIFD / GPS IFD; kamadak-exif walks that structure itself.
//! Files with unreadable or absent tags yield `None` fields — a missing value
//! must never invent one, because the values land in the catalogue and later
//! feed numeric rules (`exposure < 1`) where an invented 0 would silently
//! match.
//!
//! Deviation from darktable recorded in PARITY_AUDIT: dt reads via exiv2 over
//! its whole format matrix at import; we cover what kamadak-exif can parse and
//! leave NULL elsewhere. Numeric rule semantics treat NULL as not-matching
//! (see `rule_stack`), so unprobed images stay out of numeric-rule results.

use exif::{In, Tag, Value};
use std::path::Path;

/// The import-relevant subset. All fields optional independently: a file can
/// carry exposure but no focal length (scans, some compacts), or GPS without
/// any exposure data (phone JPEGs stripped of the rest).
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
    /// Decimal degrees, +N / -S (GPSLatitude + GPSLatitudeRef).
    pub latitude: Option<f64>,
    /// Decimal degrees, +E / -W (GPSLongitude + GPSLongitudeRef).
    pub longitude: Option<f64>,
    /// Metres above (positive) or below (negative) sea level
    /// (GPSAltitude + GPSAltitudeRef).
    pub altitude: Option<f64>,
}

impl ExifMeta {
    pub const NONE: ExifMeta = ExifMeta {
        exposure: None,
        aperture: None,
        iso: None,
        focal_length: None,
        latitude: None,
        longitude: None,
        altitude: None,
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
    let value = |tag| exif.get_field(tag, In::PRIMARY).map(|f| &f.value);
    Some(ExifMeta {
        exposure: rational_f64(exif.get_field(Tag::ExposureTime, In::PRIMARY)),
        aperture: rational_f64(exif.get_field(Tag::FNumber, In::PRIMARY)),
        iso: short_u16(exif.get_field(Tag::ISOSpeed, In::PRIMARY)),
        focal_length: rational_f64(exif.get_field(Tag::FocalLength, In::PRIMARY)),
        latitude: gps_coord(
            value(Tag::GPSLatitude),
            value(Tag::GPSLatitudeRef),
            b'N',
            b'S',
        ),
        longitude: gps_coord(
            value(Tag::GPSLongitude),
            value(Tag::GPSLongitudeRef),
            b'E',
            b'W',
        ),
        altitude: gps_altitude(value(Tag::GPSAltitude), value(Tag::GPSAltitudeRef)),
    })
}

/// Decimal degrees from a GPS DMS triple plus its hemisphere reference.
///
/// `dms` is the 3-rational `[deg, min, sec]` value (GPSLatitude/GPSLongitude),
/// `gps_ref` the ASCII hemisphere letter (GPSLatitudeRef/GPSLongitudeRef).
/// `None` unless BOTH are present and well-formed: the direction is required
/// to interpret the sign, so a missing ref yields `None` rather than an
/// assumed hemisphere. A degenerate triple (wrong length, non-finite
/// component such as a zero denominator) is likewise `None`.
fn gps_coord(dms: Option<&Value>, gps_ref: Option<&Value>, pos: u8, neg: u8) -> Option<f64> {
    let rats = match dms {
        Some(Value::Rational(r)) => r,
        _ => return None,
    };
    if rats.len() != 3 {
        return None;
    }
    let mut parts = [0.0f64; 3];
    for (i, r) in rats.iter().enumerate() {
        let v = r.to_f64();
        if !v.is_finite() || v < 0.0 {
            return None;
        }
        parts[i] = v;
    }
    let dir = match gps_ref {
        Some(Value::Ascii(v)) => v.first().and_then(|s| s.iter().find(|b| **b != 0).copied()),
        _ => return None,
    }?;
    let sign = if dir == pos {
        1.0
    } else if dir == neg {
        -1.0
    } else {
        return None;
    };
    let decimal = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0;
    if !decimal.is_finite() {
        return None;
    }
    Some(sign * decimal)
}

/// Metres above/below sea level from GPSAltitude plus its reference byte.
///
/// `None` when the altitude value itself is missing or degenerate. A missing
/// or unrecognised reference defaults to positive (above sea level), matching
/// most writers, which omit GPSAltitudeRef for above-sea-level shots; a ref
/// of 1 — BYTE per the EXIF spec, SHORT accepted leniently — reads as below
/// sea level.
fn gps_altitude(value: Option<&Value>, gps_ref: Option<&Value>) -> Option<f64> {
    let v = match value {
        Some(Value::Rational(r)) => r.first()?.to_f64(),
        Some(Value::SRational(r)) => r.first()?.to_f64(),
        _ => return None,
    };
    if !v.is_finite() {
        return None;
    }
    let below = matches!(gps_ref, Some(Value::Byte(b)) if b.first() == Some(&1))
        || matches!(gps_ref, Some(Value::Short(s)) if s.first() == Some(&1));
    Some(if below { -v.abs() } else { v.abs() })
}

/// Build an all-`None` meta — the importer's fallback for unreadable files.
pub fn probe_or_none(path: &Path) -> ExifMeta {
    probe(path).unwrap_or(ExifMeta::NONE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use exif::Rational;

    fn rat(num: u32, denom: u32) -> Rational {
        Rational { num, denom }
    }

    fn dms(deg: (u32, u32), min: (u32, u32), sec: (u32, u32)) -> Value {
        Value::Rational(vec![rat(deg.0, deg.1), rat(min.0, min.1), rat(sec.0, sec.1)])
    }

    fn ascii(s: &[u8]) -> Value {
        Value::Ascii(vec![s.to_vec()])
    }

    #[test]
    fn none_meta_is_all_none() {
        assert_eq!(ExifMeta::NONE, probe_or_none(Path::new("/nonexistent/file.ORF")));
    }

    #[test]
    fn gps_north_and_east_read_positive() {
        // 48 deg 51 min 29.16 sec N ~ 48.8581 (Paris).
        let lat = gps_coord(
            Some(&dms((48, 1), (51, 1), (2916, 100))),
            Some(&ascii(b"N")),
            b'N',
            b'S',
        )
        .unwrap();
        assert!((lat - 48.8581).abs() < 1e-6, "{lat}");
        let lon = gps_coord(
            Some(&dms((2, 1), (21, 1), (900, 100))),
            Some(&ascii(b"E")),
            b'E',
            b'W',
        )
        .unwrap();
        assert!((lon - 2.3525).abs() < 1e-6, "{lon}");
    }

    #[test]
    fn gps_south_and_west_read_negative() {
        let lat = gps_coord(
            Some(&dms((33, 1), (51, 1), (0, 1))),
            Some(&ascii(b"S")),
            b'N',
            b'S',
        )
        .unwrap();
        assert!((lat + 33.85).abs() < 1e-9, "{lat}");
        let lon = gps_coord(
            Some(&dms((151, 1), (12, 1), (0, 1))),
            Some(&ascii(b"W")),
            b'E',
            b'W',
        )
        .unwrap();
        assert!((lon + 151.2).abs() < 1e-9, "{lon}");
    }

    #[test]
    fn gps_ref_with_nul_terminator_still_resolves() {
        // Some writers NUL-terminate the ASCII ref; the letter must still win.
        let lat = gps_coord(
            Some(&dms((10, 1), (0, 1), (0, 1))),
            Some(&ascii(b"S\0")),
            b'N',
            b'S',
        )
        .unwrap();
        assert_eq!(lat, -10.0);
    }

    #[test]
    fn gps_missing_or_malformed_tags_yield_none() {
        let good_dms = dms((10, 1), (20, 1), (30, 1));
        let good_ref = ascii(b"N");
        // Missing side of the pair invents nothing.
        assert_eq!(gps_coord(None, Some(&good_ref), b'N', b'S'), None);
        assert_eq!(gps_coord(Some(&good_dms), None, b'N', b'S'), None);
        // Wrong hemisphere letter, wrong value type, short triple.
        assert_eq!(
            gps_coord(Some(&good_dms), Some(&ascii(b"X")), b'N', b'S'),
            None
        );
        assert_eq!(
            gps_coord(Some(&Value::Short(vec![10, 20, 30])), Some(&good_ref), b'N', b'S'),
            None
        );
        assert_eq!(
            gps_coord(
                Some(&Value::Rational(vec![rat(10, 1), rat(20, 1)])),
                Some(&good_ref),
                b'N',
                b'S'
            ),
            None
        );
        // A zero denominator is not a number, so the whole fix is dropped.
        assert_eq!(
            gps_coord(Some(&dms((10, 0), (20, 1), (30, 1))), Some(&good_ref), b'N', b'S'),
            None
        );
    }

    #[test]
    fn altitude_sign_follows_its_reference_byte() {
        let v = Value::Rational(vec![rat(1234, 10)]);
        // Missing ref defaults to above sea level, matching most writers.
        assert_eq!(gps_altitude(Some(&v), None), Some(123.4));
        assert_eq!(gps_altitude(Some(&v), Some(&Value::Byte(vec![0]))), Some(123.4));
        // Ref value 1 is below sea level (the Dead Sea shore, roughly) —
        // BYTE per the spec, SHORT accepted leniently.
        assert_eq!(gps_altitude(Some(&v), Some(&Value::Byte(vec![1]))), Some(-123.4));
        assert_eq!(gps_altitude(Some(&v), Some(&Value::Short(vec![1]))), Some(-123.4));
        // An unrecognised ref shape is not a direction; stay positive.
        assert_eq!(
            gps_altitude(Some(&v), Some(&Value::Short(vec![0]))),
            Some(123.4)
        );
    }

    #[test]
    fn altitude_without_a_value_is_none() {
        assert_eq!(gps_altitude(None, Some(&Value::Byte(vec![0]))), None);
        assert_eq!(gps_altitude(None, None), None);
        // Zero-denominator rational is degenerate, ref or no ref.
        let bad = Value::Rational(vec![rat(5, 0)]);
        assert_eq!(gps_altitude(Some(&bad), Some(&Value::Byte(vec![1]))), None);
    }
}
