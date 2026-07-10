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
use std::path::Path;

/// Create the `main`-scoped catalog tables if they don't already exist.
///
/// Idempotent: safe to call on every startup, and a no-op (via
/// `IF NOT EXISTS`) on a `library.db` the C app already populated, so it never
/// clobbers or narrows an existing darktable schema. Column shapes mirror what
/// the DAO layer in [`crate::film`], [`crate::image`], [`crate::colorlabels`],
/// and [`crate::tags`] reads and writes. Only touches `main`, so it works on a
/// bare connection with no `data`/`memory` attached.
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
            ON color_labels (imgid, color);
         CREATE TABLE IF NOT EXISTS main.tagged_images (
            imgid INTEGER, tagid INTEGER, position INTEGER,
            PRIMARY KEY (imgid, tagid)
         );",
    )
}

/// Ensure the full schema across `main` and the attached `data`/`memory`
/// schemas — the base tables plus the tag catalog (`data.tags`,
/// `memory.darktable_tags`; `main.tagged_images` comes from the base). The
/// `data` and `memory` schemas MUST already be attached (see [`open_catalog`]),
/// or the `data.tags` create errors with "unknown database data".
pub fn ensure_full_schema(conn: &Connection) -> rusqlite::Result<()> {
    ensure_base_schema(conn)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS data.tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR, synonyms VARCHAR, flags INTEGER
         );
         CREATE UNIQUE INDEX IF NOT EXISTS data.tags_name_idx ON tags (name);
         CREATE TABLE IF NOT EXISTS memory.darktable_tags (tagid INTEGER PRIMARY KEY);",
    )
}

/// Open a full catalog connection mirroring darktable's three-schema layout:
/// the main `library.db` at `db_path`, its sibling `data.db` (same configdir)
/// attached as `data`, and a fresh per-connection in-memory database attached as
/// `memory`. All schemas the standalone UI touches are ensured via
/// [`ensure_full_schema`], so tags work on a fresh `/config` where no C app ever
/// created `data.db`.
///
/// `memory` is intentionally per-connection and empty on open: it holds
/// darktable's ephemeral session scratch (e.g. `memory.darktable_tags` marks
/// which tags are internal), which is rebuilt each session, so a fresh empty one
/// per connection is correct — every tag then reads as a user tag.
pub fn open_catalog(db_path: &str) -> rusqlite::Result<Connection> {
    let conn = Connection::open(db_path)?;
    // Parity with the UI's open_rating_conn/open_colorlabels_conn: the metadata
    // writers (rating / colour labels) briefly hold library.db's write lock from
    // an off-thread `serialized_write`. A tag read/write runs on the main thread
    // and would otherwise get an immediate SQLITE_BUSY and silently drop the tag;
    // wait instead. Best-effort — a failure here doesn't stop the catalog opening.
    let _ = conn.busy_timeout(std::time::Duration::from_secs(3));
    // darktable stores data.db next to library.db in the configdir.
    let data_path = Path::new(db_path)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("data.db");
    // Bind the paths as parameters so a path with a quote can't break the SQL.
    conn.execute("ATTACH DATABASE ?1 AS data", [&*data_path.to_string_lossy()])?;
    conn.execute("ATTACH DATABASE ?1 AS memory", [":memory:"])?;
    ensure_full_schema(&conn)?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{film, image, tags};

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

    #[test]
    fn full_schema_enables_tag_roundtrip() {
        // Reproduce the three-schema layout on in-memory dbs and confirm the tag
        // DAOs (which reference data.tags / main.tagged_images / memory.*) work
        // once ensure_full_schema has created the tables.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("ATTACH DATABASE ?1 AS data", [":memory:"]).unwrap();
        conn.execute("ATTACH DATABASE ?1 AS memory", [":memory:"]).unwrap();
        ensure_full_schema(&conn).unwrap();

        let tid = tags::tag_new(&conn, "beach").unwrap().unwrap();
        assert!(tags::tag_attach(&conn, tid, 42).unwrap());
        let listed = tags::tag_list_with_counts(&conn).unwrap();
        assert_eq!(listed, vec![(tid, "beach".to_string(), 1)]);
    }

    #[test]
    fn open_catalog_attaches_data_and_memory() {
        // open_catalog derives data.db as a sibling file, so it needs a real
        // directory. Use a unique temp dir and clean it up.
        let dir = std::env::temp_dir().join(format!(
            "darkroom_schema_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("library.db");

        let conn = open_catalog(db_path.to_str().unwrap()).unwrap();

        // Both auxiliary schemas are attached under the expected names.
        let schemas: Vec<String> = conn
            .prepare("PRAGMA database_list")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(schemas.contains(&"data".to_string()), "{schemas:?}");
        assert!(schemas.contains(&"memory".to_string()), "{schemas:?}");

        // And the full flow works against real files: import + tag.
        let film_id = film::film_new(&conn, "/photos/x").unwrap().unwrap();
        image::image_insert(&conn, film_id, "a.dng", 10, 10).unwrap();
        let tid = tags::tag_new(&conn, "sunset").unwrap().unwrap();
        assert!(tags::tag_attach(&conn, tid, 1).unwrap());
        assert_eq!(tags::tag_list_with_counts(&conn).unwrap().len(), 1);

        // data.db was materialised beside library.db.
        assert!(dir.join("data.db").exists());

        drop(conn);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
