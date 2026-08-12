//! Idempotent bootstrap of the base catalog schema.
//!
//! Historically the C darktable app created `main.film_rolls` / `main.images`
//! (and the rest of its schema) in `dt_init()` on first launch, so the Rust DAO
//! layer only ever `INSERT`ed into pre-existing tables. Now that the Rust UI
//! (`c41-rs`) is the container's front-end and can boot against an empty
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

/// Create the **durable** tag catalog in the on-disk `data` schema
/// (`data.tags` + its unique-name index). `data` MUST be attached. This lives on
/// disk, so it only needs creating ONCE per catalog, not per connection — see
/// the [`open_catalog`] (full) vs [`open_catalog_session`] (session-only) split.
fn ensure_data_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS data.tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR, synonyms VARCHAR, flags INTEGER
         );
         CREATE UNIQUE INDEX IF NOT EXISTS data.tags_name_idx ON tags (name);",
    )
}

/// Ensure the **durable** (on-disk) schema: the `main` base tables plus the
/// `data.tags` catalog. Persists across connections, so a single bootstrap per
/// catalog suffices (the per-connection `memory` scratch is separate — see
/// [`ensure_session_schema`]). `data` MUST already be attached, or the
/// `data.tags` create errors with "unknown database data".
pub fn ensure_persistent_schema(conn: &Connection) -> rusqlite::Result<()> {
    ensure_base_schema(conn)?;
    ensure_data_schema(conn)
}

/// Ensure the **ephemeral, per-connection** `memory.darktable_tags` scratch
/// table. `memory` is a fresh in-memory database on every catalog open, so
/// unlike the durable schema this MUST run on each connection. `memory` MUST be
/// attached.
pub fn ensure_session_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memory.darktable_tags (tagid INTEGER PRIMARY KEY);",
    )
}

/// Ensure the full schema across `main` and the attached `data`/`memory`
/// schemas — the durable [`ensure_persistent_schema`] plus the per-connection
/// [`ensure_session_schema`]. The `data` and `memory` schemas MUST already be
/// attached (see [`open_catalog`]).
pub fn ensure_full_schema(conn: &Connection) -> rusqlite::Result<()> {
    ensure_persistent_schema(conn)?;
    ensure_session_schema(conn)
}

/// Open `library.db`, attach its sibling `data.db` (same configdir) as `data`
/// and a fresh per-connection in-memory database as `memory`, and set the busy
/// timeout — the shared connection setup behind [`open_catalog`] and
/// [`open_catalog_session`]. No schema is ensured here; the callers layer that.
fn attach_catalog(db_path: &str) -> rusqlite::Result<Connection> {
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
    Ok(conn)
}

/// Open a full catalog connection mirroring darktable's three-schema layout:
/// the main `library.db` at `db_path`, its sibling `data.db` (same configdir)
/// attached as `data`, and a fresh per-connection in-memory database attached as
/// `memory`. Ensures the FULL schema via [`ensure_full_schema`], so tags work on
/// a fresh `/config` where no C app ever created `data.db`. Use this for writes
/// and for the once-per-launch bootstrap; the read-hot path has the leaner
/// [`open_catalog_session`].
///
/// `memory` is intentionally per-connection and empty on open: it holds
/// darktable's ephemeral session scratch (e.g. `memory.darktable_tags` marks
/// which tags are internal), which is rebuilt each session, so a fresh empty one
/// per connection is correct — every tag then reads as a user tag.
pub fn open_catalog(db_path: &str) -> rusqlite::Result<Connection> {
    let conn = attach_catalog(db_path)?;
    ensure_full_schema(&conn)?;
    Ok(conn)
}

/// Like [`open_catalog`], but ensures ONLY the per-connection `memory` scratch
/// ([`ensure_session_schema`]) — it skips the durable `main`/`data` DDL probes,
/// which are assumed already bootstrapped by a prior [`open_catalog`] (the app
/// does this once at startup). Use this for read-hot paths — e.g. reloading an
/// image's tags on every lighttable selection — where re-probing the durable
/// schema on every open is wasted work. A caller that could run before any
/// [`open_catalog`] bootstrap must use the full [`open_catalog`] instead; a
/// missing durable table here would surface as an empty read, not a repair.
pub fn open_catalog_session(db_path: &str) -> rusqlite::Result<Connection> {
    let conn = attach_catalog(db_path)?;
    ensure_session_schema(&conn)?;
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

    #[test]
    fn session_open_reads_durable_tags_without_re_ensuring() {
        // The read-hot path (open_catalog_session) must skip the durable DDL yet
        // still see tags a prior full open_catalog persisted to data.db.
        let dir = std::env::temp_dir().join(format!(
            "darkroom_schema_session_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("library.db");
        let db = db_path.to_str().unwrap();

        // Bootstrap once (full) and persist a tag to the on-disk data.tags.
        let tid = {
            let boot = open_catalog(db).unwrap();
            let tid = tags::tag_new(&boot, "beach").unwrap().unwrap();
            assert!(tags::tag_attach(&boot, tid, 7).unwrap());
            tid
        };

        // A session open — which only ensures the memory scratch — still reads
        // the durable tag back (proves the durable schema is not re-created and
        // is correctly reused across connections).
        let sess = open_catalog_session(db).unwrap();
        assert_eq!(
            tags::tag_list_with_counts(&sess).unwrap(),
            vec![(tid, "beach".to_string(), 1)]
        );
        // The per-connection memory scratch exists on the session connection.
        let n: i64 = sess
            .query_row("SELECT count(*) FROM memory.darktable_tags", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);

        drop(sess);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
