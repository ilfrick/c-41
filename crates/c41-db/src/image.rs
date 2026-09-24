//! Image record DAO — Rust impl of the SQL-only parts of src/common/image.c.
//!
//! Phase 2-db-6: core image lookups and mutations. GLib string allocation,
//! image cache invalidation, XMP sidecar writing, and signal emission stay
//! in C for now.

use rusqlite::{Connection, OptionalExtension, params};
use c41_sys::dt_imgid_t;

/// Partial mirror of dt_image_t — fields are added as needed per phase.
#[derive(Debug, Clone)]
pub struct ImageRow {
    pub id:         dt_imgid_t,
    pub film_id:    i32,
    pub filename:   String,
    pub width:      i32,
    pub height:     i32,
    pub flags:      i32,
    pub datetime_taken: String,
}

/// Check if an image ID exists in the database.
/// Mirrors `dt_image_exists()` in image.c:459.
pub fn image_exists(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<bool> {
    let count: i32 = conn.query_row(
        "SELECT COUNT(*) FROM main.images WHERE id = ?1",
        params![imgid],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// Get the filename (not the full path) for an image ID.
/// Mirrors `dt_image_get_filename()` in image.c:481.
pub fn image_get_filename(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT filename FROM main.images WHERE id = ?1",
        params![imgid],
        |row| row.get(0),
    )
    .optional()
}

/// Get the full filesystem path: `folder/filename`.
pub fn image_get_full_path(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT f.folder || '/' || i.filename \
         FROM main.images i JOIN main.film_rolls f ON f.id = i.film_id \
         WHERE i.id = ?1",
        params![imgid],
        |row| row.get(0),
    )
    .optional()
}

/// Load a minimal image row by ID.
pub fn image_get(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<Option<ImageRow>> {
    conn.query_row(
        "SELECT id, film_id, filename, width, height, flags, datetime_taken \
         FROM main.images WHERE id = ?1",
        params![imgid],
        |row| Ok(ImageRow {
            id:             row.get(0)?,
            film_id:        row.get(1)?,
            filename:       row.get(2)?,
            width:          row.get(3)?,
            height:         row.get(4)?,
            flags:          row.get(5)?,
            datetime_taken: row.get(6).unwrap_or_default(),
        }),
    )
    .optional()
}

/// List all image IDs in a film roll.
pub fn image_list_by_film(conn: &Connection, film_id: i32) -> rusqlite::Result<Vec<dt_imgid_t>> {
    let mut stmt = conn.prepare(
        "SELECT id FROM main.images WHERE film_id = ?1 ORDER BY id",
    )?;
    let ids = stmt
        .query_map(params![film_id], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ids)
}

/// Update the `flags` bitmask of an image.
pub fn image_set_flags(conn: &Connection, imgid: dt_imgid_t, flags: i32) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE main.images SET flags = ?1 WHERE id = ?2",
        params![flags, imgid],
    )?;
    Ok(())
}

/// Get the current `flags` value for an image.
pub fn image_get_flags(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<Option<i32>> {
    conn.query_row(
        "SELECT flags FROM main.images WHERE id = ?1",
        params![imgid],
        |row| row.get(0),
    )
    .optional()
}

/// Delete an image record (does NOT delete the file on disk).
pub fn image_remove(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM main.images WHERE id = ?1", params![imgid])?;
    Ok(())
}

/// Count the total number of images.
pub fn image_count_all(conn: &Connection) -> rusqlite::Result<i32> {
    conn.query_row("SELECT COUNT(*) FROM main.images", [], |row| row.get(0))
}

/// Look up image id by full path (folder/filename split).
pub fn image_get_id_by_path(
    conn: &Connection,
    full_path: &str,
) -> rusqlite::Result<Option<dt_imgid_t>> {
    use std::path::Path;
    let p        = Path::new(full_path);
    let filename = match p.file_name().and_then(|n| n.to_str()) {
        Some(f) => f,
        None    => return Ok(None),
    };
    let folder = match p.parent().and_then(|d| d.to_str()) {
        Some(d) => d,
        None    => return Ok(None),
    };
    conn.query_row(
        "SELECT i.id FROM main.images i \
         JOIN main.film_rolls f ON f.id = i.film_id \
         WHERE f.folder = ?1 AND i.filename = ?2",
        params![folder, filename],
        |row| row.get(0),
    )
    .optional()
}

/// The EXIF numeric subset stored alongside an image row (m4-135) plus the
/// geotagging triple (u2). Owned here rather than borrowed from `c41-core` so
/// this crate keeps its dependency graph minimal; the importer maps its probe
/// struct onto this one-for-one. `None` fields insert NULL — never 0, because
/// 0 would match numeric rules (`exposure < 1`) that unknown values must not
/// satisfy. Column names mirror darktable's `main.images` exactly
/// (`longitude`, `latitude`, `altitude` — see `schema::ensure_base_schema`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ImageExif {
    pub exposure: Option<f64>,
    pub aperture: Option<f64>,
    pub iso: Option<f64>,
    pub focal_length: Option<f64>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub altitude: Option<f64>,
}

/// Insert a new image record. Returns the new image id.
/// Skips insertion if an image with the same film_id and filename already exists.
pub fn image_insert(
    conn: &Connection,
    film_id: i32,
    filename: &str,
    width: i32,
    height: i32,
    exif: ImageExif,
) -> rusqlite::Result<dt_imgid_t> {
    // Return existing id if already present
    let existing: Option<dt_imgid_t> = conn
        .query_row(
            "SELECT id FROM main.images WHERE film_id = ?1 AND filename = ?2",
            params![film_id, filename],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(id) = existing {
        return Ok(id);
    }
    conn.execute(
        "INSERT INTO main.images (film_id, filename, width, height, flags, \
         exposure, aperture, iso, focal_length, latitude, longitude, altitude) \
         VALUES (?1, ?2, ?3, ?4, 0, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            film_id,
            filename,
            width,
            height,
            exif.exposure,
            exif.aperture,
            exif.iso,
            exif.focal_length,
            exif.latitude,
            exif.longitude,
            exif.altitude,
        ],
    )?;
    Ok(conn.last_insert_rowid() as dt_imgid_t)
}

/// Read an image's geotagging triple as `(latitude, longitude, altitude)`.
/// `None` per field when the column is NULL (never probed, or cleared) —
/// mirrors darktable's nullable REAL columns. Errors only on a genuinely
/// broken query (e.g. a catalog predating the geo migration); callers showing
/// read-only state should treat that as absent, not as zero.
pub fn image_get_geo(
    conn: &Connection,
    imgid: dt_imgid_t,
) -> rusqlite::Result<(Option<f64>, Option<f64>, Option<f64>)> {
    conn.query_row(
        "SELECT latitude, longitude, altitude FROM main.images WHERE id = ?1",
        params![imgid],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )
}

/// Write an image's geotagging triple. `None` fields store NULL (unknown /
/// cleared) — never 0, which on the equator or at sea level would be a real
/// claimed position. Single-image only: the caller resolves exactly one imgid.
pub fn image_set_geo(
    conn: &Connection,
    imgid: dt_imgid_t,
    latitude: Option<f64>,
    longitude: Option<f64>,
    altitude: Option<f64>,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE main.images SET latitude = ?1, longitude = ?2, altitude = ?3 \
         WHERE id = ?4",
        params![latitude, longitude, altitude, imgid],
    )?;
    Ok(())
}

/// One geotagged image for the map list (u3): the catalogue id, the full
/// `folder/filename` path (same join as [`image_get_full_path`]), and the fix.
/// Latitude and longitude are always populated here — the query only returns
/// rows where both are non-NULL; altitude stays optional because an unknown
/// altitude is still a plottable fix.
#[derive(Debug, Clone)]
pub struct GeoImage {
    pub id:        dt_imgid_t,
    pub path:      String,
    pub latitude:  f64,
    pub longitude: f64,
    pub altitude:  Option<f64>,
}

/// List every image carrying a full lat+lon fix, ordered by filename (ties
/// broken by folder, then id) — the map list's stable display order. Rows
/// with only one axis set are excluded: a half-known fix is not a position.
/// Out-of-range or non-finite axes (only writable by bypassing the panel's
/// range-validated parse) are excluded too: they are not positions either.
pub fn image_list_geotagged(conn: &Connection) -> rusqlite::Result<Vec<GeoImage>> {
    let mut stmt = conn.prepare(
        "SELECT i.id, f.folder || '/' || i.filename, i.latitude, i.longitude, i.altitude \
         FROM main.images i JOIN main.film_rolls f ON f.id = i.film_id \
         WHERE i.latitude IS NOT NULL AND i.longitude IS NOT NULL \
            AND abs(i.latitude) <= 90.0 AND abs(i.longitude) <= 180.0 \
         ORDER BY i.filename, f.folder, i.id",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok(GeoImage {
                id:        row.get(0)?,
                path:      row.get(1)?,
                latitude:  row.get(2)?,
                longitude: row.get(3)?,
                altitude:  row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Return the film_id for a given image.
pub fn image_get_film_id(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<Option<i32>> {
    conn.query_row(
        "SELECT film_id FROM main.images WHERE id = ?1",
        params![imgid],
        |row| row.get(0),
    )
    .optional()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use rusqlite::OpenFlags;

    fn open_test_db() -> Connection {
        static COUNTER: AtomicU32 = AtomicU32::new(1);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_URI;
        let conn = Connection::open_with_flags(
            &format!("file:img_main{n}?mode=memory&cache=shared"), flags,
        ).unwrap();
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS main.film_rolls (
                id INTEGER PRIMARY KEY, access_timestamp INTEGER, folder VARCHAR
            );
            CREATE TABLE IF NOT EXISTS main.images (
                id INTEGER PRIMARY KEY, film_id INTEGER, filename VARCHAR,
                width INTEGER DEFAULT 0, height INTEGER DEFAULT 0,
                flags INTEGER DEFAULT 0, datetime_taken VARCHAR DEFAULT ''
            );
        ").unwrap();
        // Seed a film roll and an image
        conn.execute_batch("
            INSERT INTO main.film_rolls (id, folder) VALUES (1, '/photos/test');
            INSERT INTO main.images (id, film_id, filename, width, height)
                VALUES (100, 1, 'IMG_0001.dng', 4000, 3000);
        ").unwrap();
        conn
    }

    #[test]
    fn exists_true_for_seeded_image() {
        let db = open_test_db();
        assert!(image_exists(&db, 100).unwrap());
    }

    #[test]
    fn exists_false_for_missing_image() {
        let db = open_test_db();
        assert!(!image_exists(&db, 999).unwrap());
    }

    #[test]
    fn get_filename_roundtrips() {
        let db = open_test_db();
        assert_eq!(
            image_get_filename(&db, 100).unwrap().as_deref(),
            Some("IMG_0001.dng")
        );
    }

    #[test]
    fn get_full_path_joins_folder_and_filename() {
        let db = open_test_db();
        assert_eq!(
            image_get_full_path(&db, 100).unwrap().as_deref(),
            Some("/photos/test/IMG_0001.dng")
        );
    }

    #[test]
    fn get_image_row_returns_correct_fields() {
        let db = open_test_db();
        let img = image_get(&db, 100).unwrap().unwrap();
        assert_eq!(img.id, 100);
        assert_eq!(img.film_id, 1);
        assert_eq!(img.filename, "IMG_0001.dng");
        assert_eq!(img.width, 4000);
    }

    #[test]
    fn get_returns_none_for_missing() {
        let db = open_test_db();
        assert!(image_get(&db, 999).unwrap().is_none());
    }

    #[test]
    fn list_by_film_returns_images() {
        let db = open_test_db();
        let ids = image_list_by_film(&db, 1).unwrap();
        assert!(ids.contains(&100));
    }

    #[test]
    fn set_and_get_flags() {
        let db = open_test_db();
        image_set_flags(&db, 100, 0x42).unwrap();
        assert_eq!(image_get_flags(&db, 100).unwrap(), Some(0x42));
    }

    #[test]
    fn remove_deletes_image() {
        let db = open_test_db();
        image_remove(&db, 100).unwrap();
        assert!(!image_exists(&db, 100).unwrap());
    }

    #[test]
    fn count_all_reflects_inserted_rows() {
        let db = open_test_db();
        assert_eq!(image_count_all(&db).unwrap(), 1);
    }

    #[test]
    fn get_film_id_returns_correct_parent() {
        let db = open_test_db();
        assert_eq!(image_get_film_id(&db, 100).unwrap(), Some(1));
    }

    #[test]
    fn geo_triple_roundtrips_through_insert_and_update() {
        // Same shape ensure_exif_columns produces: nullable REAL columns
        // added idempotently onto an existing images table — the four
        // pre-existing numeric columns plus the three new geo ones, since
        // image_insert names them all.
        let db = open_test_db();
        db.execute_batch(
            "ALTER TABLE main.images ADD COLUMN exposure REAL;
             ALTER TABLE main.images ADD COLUMN aperture REAL;
             ALTER TABLE main.images ADD COLUMN iso REAL;
             ALTER TABLE main.images ADD COLUMN focal_length REAL;
             ALTER TABLE main.images ADD COLUMN longitude REAL;
             ALTER TABLE main.images ADD COLUMN latitude REAL;
             ALTER TABLE main.images ADD COLUMN altitude REAL;",
        )
        .unwrap();
        let id = image_insert(
            &db,
            1,
            "geo.dng",
            100,
            100,
            ImageExif {
                latitude: Some(48.8581),
                longitude: Some(2.3525),
                altitude: Some(35.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            image_get_geo(&db, id).unwrap(),
            (Some(48.8581), Some(2.3525), Some(35.0))
        );
        // Clearing writes NULL, never 0 — 0 lat/lon is a real claimed position.
        image_set_geo(&db, id, None, None, None).unwrap();
        assert_eq!(image_get_geo(&db, id).unwrap(), (None, None, None));
        // A below-sea-level fix keeps its sign through the REAL column.
        image_set_geo(&db, id, Some(-33.85), Some(151.2), Some(-5.5)).unwrap();
        assert_eq!(
            image_get_geo(&db, id).unwrap(),
            (Some(-33.85), Some(151.2), Some(-5.5))
        );
    }

    #[test]
    fn insert_without_geo_columns_fails_loudly() {
        // Documents the coupling: image_insert names the geo columns, so a
        // catalog that never ran the migration fails the insert loudly (an
        // Err the importer logs) rather than silently dropping the fix.
        let db = open_test_db();
        let r = image_insert(&db, 1, "nogeo.dng", 10, 10, ImageExif::default());
        assert!(r.is_err());
    }

    fn open_geo_test_db() -> Connection {
        // Same shape ensure_exif_columns produces: the four pre-existing
        // numeric columns plus the three geo ones, since image_insert and
        // image_list_geotagged both name them.
        let db = open_test_db();
        db.execute_batch(
            "ALTER TABLE main.images ADD COLUMN exposure REAL;
             ALTER TABLE main.images ADD COLUMN aperture REAL;
             ALTER TABLE main.images ADD COLUMN iso REAL;
             ALTER TABLE main.images ADD COLUMN focal_length REAL;
             ALTER TABLE main.images ADD COLUMN longitude REAL;
             ALTER TABLE main.images ADD COLUMN latitude REAL;
             ALTER TABLE main.images ADD COLUMN altitude REAL;
             INSERT INTO main.film_rolls (id, folder) VALUES (2, '/photos/other');",
        )
        .unwrap();
        db
    }

    fn insert_geo(
        db: &Connection,
        film_id: i32,
        name: &str,
        lat: Option<f64>,
        lon: Option<f64>,
        alt: Option<f64>,
    ) -> dt_imgid_t {
        image_insert(
            db,
            film_id,
            name,
            100,
            100,
            ImageExif { latitude: lat, longitude: lon, altitude: alt, ..Default::default() },
        )
        .unwrap()
    }

    #[test]
    fn geotagged_list_needs_both_axes_with_paths_in_filename_order() {
        let db = open_geo_test_db();
        insert_geo(&db, 1, "b_athens.dng", Some(37.98), Some(23.73), None);
        insert_geo(&db, 1, "a_sydney.dng", Some(-33.85), Some(151.2), Some(-5.5));
        insert_geo(&db, 2, "0_first.dng", Some(51.5), Some(-0.12), None);
        // No fix at all, and both half-known shapes: none of these is a position.
        insert_geo(&db, 1, "c_plain.dng", None, None, None);
        insert_geo(&db, 1, "d_no_lon.dng", Some(10.0), None, None);
        insert_geo(&db, 1, "e_no_lat.dng", None, Some(20.0), None);
        // Out-of-range and non-finite axes (only writable past the panel's
        // validated parse) are not positions either.
        insert_geo(&db, 1, "f_far_lat.dng", Some(91.0), Some(0.0), None);
        insert_geo(&db, 1, "g_far_lon.dng", Some(0.0), Some(181.0), None);
        insert_geo(&db, 1, "h_nan.dng", Some(f64::NAN), Some(0.0), None);
        insert_geo(&db, 1, "i_inf.dng", Some(0.0), Some(f64::INFINITY), None);
        let listed = image_list_geotagged(&db).unwrap();
        assert_eq!(listed.len(), 3, "only in-range both-axes rows list, got {listed:?}");
        // Filename order across folders, with the folder join resolved per row.
        assert_eq!(listed[0].path, "/photos/other/0_first.dng");
        assert_eq!(listed[1].path, "/photos/test/a_sydney.dng");
        assert_eq!(listed[2].path, "/photos/test/b_athens.dng");
        // Negative coords and a below-sea-level altitude survive the REAL
        // round trip; unknown altitude stays None rather than becoming 0.
        assert_eq!((listed[1].latitude, listed[1].longitude), (-33.85, 151.2));
        assert_eq!(listed[1].altitude, Some(-5.5));
        assert_eq!(listed[2].altitude, None);
        assert_eq!((listed[0].latitude, listed[0].longitude), (51.5, -0.12));
    }

    #[test]
    fn geotagged_list_is_empty_when_nothing_carries_a_fix() {
        let db = open_geo_test_db();
        insert_geo(&db, 1, "plain.dng", None, None, None);
        insert_geo(&db, 1, "half.dng", Some(10.0), None, None);
        assert!(image_list_geotagged(&db).unwrap().is_empty());
    }
}
