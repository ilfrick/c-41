//! Persistence of the darkroom view's live [`PreviewParams`] to the database,
//! so edits survive closing/reopening an image (RUST_MIGRATION_PLAN.md Phase 3
//! milestone 1 — module/edit state to the db).
//!
//! Stored in a dedicated, darkroom-ui-owned table `main.darkroom_preview`
//! (`imgid` PRIMARY KEY, `params` BLOB) — one row per image, written with a real
//! `ON CONFLICT` upsert. A private table (rather than `main.history`) keeps our
//! own blob layout out of darktable's IOP history: `main.history.num` is a
//! rewritable stack index the C app compresses/truncates, so squatting there
//! risks `num` collisions; the PK here makes the single-row invariant
//! structural and the write atomic.
//!
//! The table is **best-effort**: it's a darkroom-ui-private table that darktable
//! doesn't know about, so an aggressive C-side schema rebuild (copy-known-tables
//! into a fresh db) could drop it — in which case load falls back to defaults,
//! by design.

use rusqlite::Connection;
use crate::preview::PreviewParams;

/// DDL for the preview-params table. `CREATE … IF NOT EXISTS` on every save so
/// it works against a fresh darktable `library.db` that predates this feature.
const PREVIEW_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS main.darkroom_preview \
     (imgid INTEGER PRIMARY KEY, params BLOB NOT NULL)";

/// Resolve a file path to its `images.id` via folder + filename (mirrors the
/// lighttable's lookup). `None` if the image isn't in the catalogue.
fn imgid_for_path(conn: &Connection, full_path: &str) -> Option<i32> {
    let p = std::path::Path::new(full_path);
    let filename = p.file_name()?.to_str()?;
    let folder = p.parent()?.to_str()?;
    conn.query_row(
        "SELECT i.id FROM main.images i \
         JOIN main.film_rolls f ON f.id = i.film_id \
         WHERE f.folder = ?1 AND i.filename = ?2 ORDER BY i.id LIMIT 1",
        rusqlite::params![folder, filename],
        |row| row.get::<_, i32>(0),
    )
    .ok()
}

/// Load saved preview params for the image at `full_path`, falling back to
/// [`PreviewParams::default`] on any miss (no db, image not catalogued, no saved
/// row, or an undecodable/old blob).
pub fn load_params(db_path: &str, full_path: &str) -> PreviewParams {
    load_saved(db_path, full_path).unwrap_or_default()
}

/// Load the saved preview params for the image, distinguishing **"no saved
/// edit"** (`None`) from a decoded row (`Some`) — the darkroom view uses that to
/// decide whether to apply raw-only defaults (e.g. sigmoid on). `None` on: no
/// db, uncatalogued image, no row, or an undecodable/old-schema blob.
pub fn load_saved(db_path: &str, full_path: &str) -> Option<PreviewParams> {
    if db_path.is_empty() {
        return None;
    }
    let conn = Connection::open(db_path).ok()?;
    let imgid = imgid_for_path(&conn, full_path)?;
    load_saved_conn(&conn, imgid)
}

/// Persist preview params for the image at `full_path`. Best-effort: silently
/// no-ops if there is no db or the image isn't catalogued.
pub fn save_params(db_path: &str, full_path: &str, params: &PreviewParams) {
    if db_path.is_empty() {
        return;
    }
    let Ok(conn) = Connection::open(db_path) else {
        return;
    };
    if let Some(imgid) = imgid_for_path(&conn, full_path) {
        let _ = save_params_conn(&conn, imgid, params);
    }
}

/// Testable core of [`load_params`]. Any error (no row, or the table not yet
/// created on a pre-feature db) yields defaults.
fn load_saved_conn(conn: &Connection, imgid: i32) -> Option<PreviewParams> {
    let blob: rusqlite::Result<Vec<u8>> = conn.query_row(
        "SELECT params FROM main.darkroom_preview WHERE imgid = ?1",
        rusqlite::params![imgid],
        |row| row.get(0),
    );
    match blob {
        Ok(b) => PreviewParams::decode(&b), // None for an old/garbage-version blob
        Err(_) => None,                     // no row / no table yet
    }
}

/// Testable core of [`save_params`]: upsert the single preview row for the image
/// (PK on `imgid` guarantees one row; `ON CONFLICT` makes it a real update).
fn save_params_conn(conn: &Connection, imgid: i32, params: &PreviewParams) -> rusqlite::Result<()> {
    conn.execute(PREVIEW_TABLE_DDL, [])?;
    conn.execute(
        "INSERT INTO main.darkroom_preview (imgid, params) VALUES (?1, ?2) \
         ON CONFLICT(imgid) DO UPDATE SET params = excluded.params",
        rusqlite::params![imgid, params.encode()],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_db() -> Connection {
        // Note: no darkroom_preview table here — save_params_conn must create it
        // (mirrors a fresh darktable library.db that predates this feature).
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE main.film_rolls (id INTEGER, folder VARCHAR);
             CREATE TABLE main.images (id INTEGER, film_id INTEGER, filename VARCHAR);
             INSERT INTO main.film_rolls (id, folder) VALUES (10, '/photos');
             INSERT INTO main.images (id, film_id, filename) VALUES (42, 10, 'img.jpg');",
        )
        .unwrap();
        conn
    }

    fn sample_params() -> PreviewParams {
        let mut p = PreviewParams::default();
        p.ev = -0.75;
        p.velvia_on = true;
        p.velvia_strength = 33.0;
        p.mono_on = true;
        p.mono_r = 0.4;
        p
    }

    #[test]
    fn imgid_resolves_from_path() {
        let db = open_db();
        assert_eq!(imgid_for_path(&db, "/photos/img.jpg"), Some(42));
        assert_eq!(imgid_for_path(&db, "/photos/missing.jpg"), None);
        assert_eq!(imgid_for_path(&db, "/elsewhere/img.jpg"), None);
    }

    #[test]
    fn save_then_load_roundtrips() {
        let db = open_db();
        let p = sample_params();
        save_params_conn(&db, 42, &p).unwrap();
        assert_eq!(load_saved_conn(&db, 42), Some(p));
    }

    #[test]
    fn load_saved_is_none_when_table_absent_or_no_row() {
        let db = open_db();
        // table doesn't exist yet → None (no panic), distinct from a saved row
        assert_eq!(load_saved_conn(&db, 42), None);
        // create the table via a save for a different image, then a missing row
        save_params_conn(&db, 99, &sample_params()).unwrap();
        assert_eq!(load_saved_conn(&db, 42), None);
    }

    #[test]
    fn save_upserts_single_row() {
        let db = open_db();
        save_params_conn(&db, 42, &sample_params()).unwrap();
        let mut p2 = PreviewParams::default();
        p2.ev = 1.5;
        save_params_conn(&db, 42, &p2).unwrap();
        // PK on imgid ⇒ exactly one row, holding the latest params
        let n: i32 = db
            .query_row(
                "SELECT COUNT(*) FROM main.darkroom_preview WHERE imgid = 42",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(load_saved_conn(&db, 42), Some(p2));
    }
}
