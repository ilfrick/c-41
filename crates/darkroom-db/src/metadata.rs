//! Metadata management — Rust implementation of src/common/metadata.c.
//!
//! Phase 2-db-3: core key/value operations. Undo system, GLib list
//! return types, and signal emission stay in C for now.
//!
//! Schema:
//!   data.meta_data  (key INTEGER PK, tagname VARCHAR, name VARCHAR,
//!                    internal INTEGER, visible INTEGER, private INTEGER,
//!                    display_order INTEGER)
//!   main.meta_data  (id INTEGER, key INTEGER, value VARCHAR,
//!                    UNIQUE(id, key, value), FK id→images.id)

use rusqlite::{Connection, OptionalExtension, params};
use darkroom_sys::dt_imgid_t;

/// Look up the integer key ID for a metadata tag name (e.g. "Xmp.dc.title").
/// Returns `None` if the key is not registered in `data.meta_data`.
pub fn metadata_get_keyid(conn: &Connection, tagname: &str) -> rusqlite::Result<Option<i32>> {
    conn.query_row(
        "SELECT key FROM data.meta_data WHERE tagname = ?1",
        params![tagname],
        |row| row.get(0),
    )
    .optional()
}

/// Get the first (or only) value for a known metadata key on an image.
/// Returns `None` if no value is set.
pub fn metadata_get_value(
    conn: &Connection,
    imgid: dt_imgid_t,
    key_id: i32,
) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT value FROM main.meta_data WHERE id = ?1 AND key = ?2 LIMIT 1",
        params![imgid, key_id],
        |row| row.get(0),
    )
    .optional()
}

/// Get all (key_id, value) pairs for an image.
pub fn metadata_get_all(
    conn: &Connection,
    imgid: dt_imgid_t,
) -> rusqlite::Result<Vec<(i32, String)>> {
    let mut stmt = conn
        .prepare("SELECT key, value FROM main.meta_data WHERE id = ?1 ORDER BY key")?;
    let rows = stmt
        .query_map(params![imgid], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Set (upsert) a metadata value. Inserts if absent, replaces if present.
pub fn metadata_set_value(
    conn: &Connection,
    imgid: dt_imgid_t,
    key_id: i32,
    value: &str,
) -> rusqlite::Result<()> {
    // Delete existing then insert — matches the C _metadata_execute pattern
    // which does a DELETE followed by INSERT for the ADD action.
    conn.execute(
        "DELETE FROM main.meta_data WHERE id = ?1 AND key = ?2",
        params![imgid, key_id],
    )?;
    conn.execute(
        "INSERT INTO main.meta_data (id, key, value) VALUES (?1, ?2, ?3)",
        params![imgid, key_id, value],
    )?;
    Ok(())
}

/// Delete a specific metadata key from an image.
pub fn metadata_delete_key(
    conn: &Connection,
    imgid: dt_imgid_t,
    key_id: i32,
) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM main.meta_data WHERE id = ?1 AND key = ?2",
        params![imgid, key_id],
    )?;
    Ok(())
}

/// Delete all metadata for an image.
pub fn metadata_clear_image(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM main.meta_data WHERE id = ?1",
        params![imgid],
    )?;
    Ok(())
}

/// Check if a value for a particular key already exists for the image.
pub fn metadata_already_set(
    conn: &Connection,
    imgid: dt_imgid_t,
    key_id: i32,
) -> rusqlite::Result<bool> {
    let count: i32 = conn.query_row(
        "SELECT COUNT(*) FROM main.meta_data WHERE id = ?1 AND key = ?2",
        params![imgid, key_id],
        |row| row.get(0),
    )?;
    Ok(count > 0)
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
            &format!("file:md_main{n}?mode=memory&cache=shared"),
            flags,
        ).expect("open test db");
        conn.execute_batch(&format!(
            "ATTACH DATABASE 'file:md_data{n}?mode=memory&cache=shared' AS data;"
        )).expect("attach data");
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS data.meta_data (
                key INTEGER PRIMARY KEY,
                tagname VARCHAR,
                name VARCHAR,
                internal INTEGER,
                visible INTEGER,
                private INTEGER,
                display_order INTEGER
            );
            CREATE TABLE IF NOT EXISTS main.meta_data (
                id INTEGER, key INTEGER, value VARCHAR
            );
        ").expect("create schema");
        // Seed a couple of key definitions
        conn.execute_batch("
            INSERT OR IGNORE INTO data.meta_data (key, tagname, name) VALUES (0, 'Xmp.dc.title',       'Title');
            INSERT OR IGNORE INTO data.meta_data (key, tagname, name) VALUES (1, 'Xmp.dc.description', 'Description');
            INSERT OR IGNORE INTO data.meta_data (key, tagname, name) VALUES (2, 'Xmp.dc.creator',     'Creator');
        ").expect("seed keys");
        conn
    }

    #[test]
    fn keyid_lookup_finds_registered_key() {
        let db = open_test_db();
        let kid = metadata_get_keyid(&db, "Xmp.dc.title").unwrap();
        assert_eq!(kid, Some(0));
    }

    #[test]
    fn keyid_lookup_returns_none_for_unknown() {
        let db = open_test_db();
        assert_eq!(metadata_get_keyid(&db, "Xmp.nope.nope").unwrap(), None);
    }

    #[test]
    fn set_and_get_value() {
        let db = open_test_db();
        metadata_set_value(&db, 42, 0, "Sunrise in Tuscany").unwrap();
        let v = metadata_get_value(&db, 42, 0).unwrap();
        assert_eq!(v.as_deref(), Some("Sunrise in Tuscany"));
    }

    #[test]
    fn set_is_idempotent_replace() {
        let db = open_test_db();
        metadata_set_value(&db, 1, 0, "Old title").unwrap();
        metadata_set_value(&db, 1, 0, "New title").unwrap();
        let v = metadata_get_value(&db, 1, 0).unwrap();
        assert_eq!(v.as_deref(), Some("New title"));
    }

    #[test]
    fn get_value_returns_none_when_absent() {
        let db = open_test_db();
        assert_eq!(metadata_get_value(&db, 99, 0).unwrap(), None);
    }

    #[test]
    fn get_all_returns_all_keys_for_image() {
        let db = open_test_db();
        metadata_set_value(&db, 5, 0, "Title A").unwrap();
        metadata_set_value(&db, 5, 1, "Desc B").unwrap();
        let pairs = metadata_get_all(&db, 5).unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().any(|(k, v)| *k == 0 && v == "Title A"));
        assert!(pairs.iter().any(|(k, v)| *k == 1 && v == "Desc B"));
    }

    #[test]
    fn delete_key_removes_single_entry() {
        let db = open_test_db();
        metadata_set_value(&db, 7, 0, "Title").unwrap();
        metadata_set_value(&db, 7, 1, "Desc").unwrap();
        metadata_delete_key(&db, 7, 0).unwrap();
        assert_eq!(metadata_get_value(&db, 7, 0).unwrap(), None);
        assert_eq!(metadata_get_value(&db, 7, 1).unwrap().as_deref(), Some("Desc"));
    }

    #[test]
    fn clear_removes_all_metadata_for_image() {
        let db = open_test_db();
        metadata_set_value(&db, 3, 0, "T").unwrap();
        metadata_set_value(&db, 3, 2, "C").unwrap();
        metadata_clear_image(&db, 3).unwrap();
        assert_eq!(metadata_get_all(&db, 3).unwrap().len(), 0);
    }

    #[test]
    fn already_set_reflects_presence() {
        let db = open_test_db();
        assert!(!metadata_already_set(&db, 10, 0).unwrap());
        metadata_set_value(&db, 10, 0, "x").unwrap();
        assert!(metadata_already_set(&db, 10, 0).unwrap());
    }
}
