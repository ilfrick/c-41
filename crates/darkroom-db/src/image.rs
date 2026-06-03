//! Image record DAO — Rust impl of the SQL-only parts of src/common/image.c.
//!
//! Phase 2-db-6: core image lookups and mutations. GLib string allocation,
//! image cache invalidation, XMP sidecar writing, and signal emission stay
//! in C for now.

use rusqlite::{Connection, OptionalExtension, params};
use darkroom_sys::dt_imgid_t;

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

/// Insert a new image record. Returns the new image id.
/// Skips insertion if an image with the same film_id and filename already exists.
pub fn image_insert(
    conn: &Connection,
    film_id: i32,
    filename: &str,
    width: i32,
    height: i32,
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
        "INSERT INTO main.images (film_id, filename, width, height, flags) \
         VALUES (?1, ?2, ?3, ?4, 0)",
        params![film_id, filename, width, height],
    )?;
    Ok(conn.last_insert_rowid() as dt_imgid_t)
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
}
