//! Film roll management — Rust impl of src/common/film.c.
//!
//! Phase 2-db-4: DAO layer for main.film_rolls.
//! GUI signals, view-manager reset, collection query stay in C.
//!
//! Schema (current, post-migration):
//!   main.film_rolls (id INTEGER PK, access_timestamp INTEGER, folder VARCHAR)

use rusqlite::{Connection, OptionalExtension, params};
use darkroom_sys::dt_imgid_t;

pub type FilmId = i32;

/// Get the film-roll ID for a given folder path.
/// Returns `None` if no matching roll exists.
pub fn film_get_id(conn: &Connection, folder: &str) -> rusqlite::Result<Option<FilmId>> {
    // Strip trailing slash (unless it's root "/")
    let folder = folder.trim_end_matches(|c| c == '/')
        .max("/");   // fallback: keep at least "/"

    conn.query_row(
        "SELECT id FROM main.film_rolls WHERE folder = ?1",
        params![folder],
        |row| row.get(0),
    )
    .optional()
}

/// Upsert a film roll for `directory`. Returns the existing or newly created ID.
pub fn film_new(conn: &Connection, directory: &str) -> rusqlite::Result<Option<FilmId>> {
    // Strip trailing slash
    let folder: &str = directory.trim_end_matches('/');
    let folder = if folder.is_empty() { "/" } else { folder };

    // Check for existing
    if let Some(id) = film_get_id(conn, folder)? {
        return Ok(Some(id));
    }

    conn.execute(
        "INSERT INTO main.film_rolls (id, access_timestamp, folder) \
         VALUES (NULL, strftime('%s', 'now'), ?1)",
        params![folder],
    )?;

    film_get_id(conn, folder)
}

/// Touch the `access_timestamp` of a film roll (record it was opened).
pub fn film_touch(conn: &Connection, film_id: FilmId) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE main.film_rolls SET access_timestamp = strftime('%s', 'now') WHERE id = ?1",
        params![film_id],
    )?;
    Ok(())
}

/// Get the folder path for a film roll ID.
pub fn film_get_folder(conn: &Connection, film_id: FilmId) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT folder FROM main.film_rolls WHERE id = ?1",
        params![film_id],
        |row| row.get(0),
    )
    .optional()
}

/// Return true if the film roll contains no images.
pub fn film_is_empty(conn: &Connection, film_id: FilmId) -> rusqlite::Result<bool> {
    let count: i32 = conn.query_row(
        "SELECT COUNT(*) FROM main.images WHERE film_id = ?1",
        params![film_id],
        |row| row.get(0),
    )?;
    Ok(count == 0)
}

/// Delete a film roll (and all its images via cascade FK).
pub fn film_remove(conn: &Connection, film_id: FilmId) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM main.images WHERE film_id = ?1", params![film_id])?;
    conn.execute("DELETE FROM main.film_rolls WHERE id = ?1", params![film_id])?;
    Ok(())
}

/// Remove all empty film rolls.
pub fn film_remove_empty(conn: &Connection) -> rusqlite::Result<usize> {
    let n = conn.execute(
        "DELETE FROM main.film_rolls WHERE id NOT IN \
         (SELECT DISTINCT film_id FROM main.images)",
        [],
    )?;
    Ok(n)
}

/// List all film rolls: returns (id, folder, access_timestamp).
pub fn film_list(conn: &Connection) -> rusqlite::Result<Vec<(FilmId, String, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT id, folder, access_timestamp FROM main.film_rolls ORDER BY access_timestamp DESC",
    )?;
    let rows = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Count images in a film roll.
pub fn film_image_count(conn: &Connection, film_id: FilmId) -> rusqlite::Result<i32> {
    conn.query_row(
        "SELECT COUNT(*) FROM main.images WHERE film_id = ?1",
        params![film_id],
        |row| row.get(0),
    )
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
            &format!("file:film_main{n}?mode=memory&cache=shared"),
            flags,
        ).unwrap();
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS main.film_rolls (
                id INTEGER PRIMARY KEY,
                access_timestamp INTEGER,
                folder VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS main.images (
                id INTEGER PRIMARY KEY,
                film_id INTEGER
            );
        ").unwrap();
        conn
    }

    #[test]
    fn new_film_creates_and_returns_id() {
        let db = open_test_db();
        let id = film_new(&db, "/home/user/Photos/2024").unwrap().unwrap();
        assert!(id > 0);
    }

    #[test]
    fn new_film_is_idempotent() {
        let db = open_test_db();
        let id1 = film_new(&db, "/home/user/Photos/2024").unwrap().unwrap();
        let id2 = film_new(&db, "/home/user/Photos/2024").unwrap().unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn trailing_slash_stripped() {
        let db = open_test_db();
        let id1 = film_new(&db, "/photos/trip").unwrap().unwrap();
        let id2 = film_get_id(&db, "/photos/trip/").unwrap();
        assert_eq!(Some(id1), id2);
    }

    #[test]
    fn get_id_returns_none_when_absent() {
        let db = open_test_db();
        assert_eq!(film_get_id(&db, "/nonexistent").unwrap(), None);
    }

    #[test]
    fn get_folder_roundtrips() {
        let db = open_test_db();
        let id = film_new(&db, "/photos/beach").unwrap().unwrap();
        assert_eq!(film_get_folder(&db, id).unwrap().as_deref(), Some("/photos/beach"));
    }

    #[test]
    fn film_is_empty_true_when_no_images() {
        let db = open_test_db();
        let id = film_new(&db, "/empty").unwrap().unwrap();
        assert!(film_is_empty(&db, id).unwrap());
    }

    #[test]
    fn film_is_empty_false_after_image_added() {
        let db = open_test_db();
        let id = film_new(&db, "/nonempty").unwrap().unwrap();
        db.execute("INSERT INTO main.images (id, film_id) VALUES (1, ?1)", params![id]).unwrap();
        assert!(!film_is_empty(&db, id).unwrap());
    }

    #[test]
    fn remove_empty_deletes_only_childless_rolls() {
        let db = open_test_db();
        let e = film_new(&db, "/empty").unwrap().unwrap();
        let f = film_new(&db, "/full").unwrap().unwrap();
        db.execute("INSERT INTO main.images (id, film_id) VALUES (1, ?1)", params![f]).unwrap();
        let n = film_remove_empty(&db).unwrap();
        assert_eq!(n, 1);
        assert_eq!(film_get_id(&db, "/empty").unwrap(), None);
        assert!(film_get_id(&db, "/full").unwrap().is_some());
        let _ = e;
    }

    #[test]
    fn film_list_returns_all_rolls() {
        let db = open_test_db();
        film_new(&db, "/a").unwrap();
        film_new(&db, "/b").unwrap();
        let list = film_list(&db).unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn image_count_correct() {
        let db = open_test_db();
        let id = film_new(&db, "/counted").unwrap().unwrap();
        db.execute("INSERT INTO main.images (id, film_id) VALUES (1, ?1)", params![id]).unwrap();
        db.execute("INSERT INTO main.images (id, film_id) VALUES (2, ?1)", params![id]).unwrap();
        assert_eq!(film_image_count(&db, id).unwrap(), 2);
    }
}
