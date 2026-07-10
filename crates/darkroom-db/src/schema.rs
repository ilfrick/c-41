//! Idempotent bootstrap of the base catalog schema.
//!
//! Historically the C darktable app created `main.film_rolls` / `main.images`
//! (and the rest of its schema) in `dt_init()` on first launch, so the Rust DAO
//! layer only ever `INSERT`ed into pre-existing tables. Now that the Rust UI
//! (`darkroom-rs`) is the container's front-end and can boot against an empty
//! `library.db`, nothing creates those tables first — a fresh import would write
//! into a table that doesn't exist and silently register zero images.
//!
//! This module creates the `main`-scoped tables the standalone UI reads/writes.
//! The `data` and `memory` schemas darktable uses are separate ATTACH-ed
//! databases the standalone UI does not attach, so tables living there (tags,
//! collection scratch) are out of scope here and unaffected by this bootstrap.

use rusqlite::Connection;

/// Create the base catalog tables if they don't already exist.
///
/// Idempotent: safe to call on every startup, and a no-op (via
/// `IF NOT EXISTS`) on a `library.db` the C app already populated, so it never
/// clobbers or narrows an existing darktable schema. Column shapes mirror what
/// the DAO layer in [`crate::film`], [`crate::image`], and [`crate::colorlabels`]
/// reads and writes.
pub fn ensure_base_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS main.film_rolls (
            id INTEGER PRIMARY KEY, access_timestamp INTEGER, folder VARCHAR
         );
         CREATE TABLE IF NOT EXISTS main.images (
            id INTEGER PRIMARY KEY, film_id INTEGER, filename VARCHAR,
            width INTEGER DEFAULT 0, height INTEGER DEFAULT 0,
            flags INTEGER DEFAULT 0, datetime_taken VARCHAR DEFAULT ''
         );
         CREATE TABLE IF NOT EXISTS main.color_labels (imgid INTEGER, color INTEGER);
         CREATE UNIQUE INDEX IF NOT EXISTS main.color_labels_idx
            ON color_labels (imgid, color);",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{film, image};

    #[test]
    fn bootstrap_enables_import_on_a_fresh_db() {
        // A brand-new in-memory DB has no catalog tables — exactly the fresh
        // `library.db` case in the container.
        let conn = Connection::open_in_memory().unwrap();

        // Without the schema, a film-roll insert cannot succeed.
        assert!(film::film_new(&conn, "/photos/a").is_err());

        ensure_base_schema(&conn).unwrap();

        // Now the core import path works end to end.
        let film_id = film::film_new(&conn, "/photos/a").unwrap().unwrap();
        let img = image::image_insert(&conn, film_id, "IMG_0001.dng", 4000, 3000).unwrap();
        assert!(img > 0);
        assert_eq!(image::image_count_all(&conn).unwrap(), 1);
    }

    #[test]
    fn is_idempotent_and_preserves_existing_rows() {
        let conn = Connection::open_in_memory().unwrap();
        ensure_base_schema(&conn).unwrap();
        let film_id = film::film_new(&conn, "/photos/a").unwrap().unwrap();
        image::image_insert(&conn, film_id, "IMG_0001.dng", 100, 100).unwrap();

        // Calling again must not error nor drop the existing row.
        ensure_base_schema(&conn).unwrap();
        ensure_base_schema(&conn).unwrap();
        assert_eq!(image::image_count_all(&conn).unwrap(), 1);
    }
}
