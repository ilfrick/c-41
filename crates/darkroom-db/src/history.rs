//! Edit history DAO — Rust impl of the SQL-only parts of src/common/history.c.
//!
//! Phase 2-db-7: history record CRUD. IOP pipeline application,
//! mask history, undo recording, and XMP writing stay in C.
//!
//! Schema:
//!   main.history (imgid, num, module, operation VARCHAR, op_params BLOB,
//!                 enabled, blendop_params BLOB, blendop_version,
//!                 multi_priority, multi_name VARCHAR)

use rusqlite::{Connection, params};
use darkroom_sys::dt_imgid_t;

/// One history entry (lightweight — blobs omitted for now).
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub imgid:     dt_imgid_t,
    pub num:       i32,
    pub operation: String,
    pub enabled:   bool,
    pub multi_priority: i32,
    pub multi_name: String,
}

/// Count history entries for an image.
pub fn history_count(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<i32> {
    conn.query_row(
        "SELECT COUNT(*) FROM main.history WHERE imgid = ?1",
        params![imgid],
        |row| row.get(0),
    )
}

/// List lightweight history entries for an image, ordered by num.
pub fn history_list(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<Vec<HistoryEntry>> {
    let mut stmt = conn.prepare(
        "SELECT imgid, num, operation, enabled, multi_priority, multi_name \
         FROM main.history WHERE imgid = ?1 ORDER BY num",
    )?;
    let rows = stmt
        .query_map(params![imgid], |row| {
            Ok(HistoryEntry {
                imgid:          row.get(0)?,
                num:            row.get(1)?,
                operation:      row.get(2)?,
                enabled:        row.get::<_, i32>(3)? != 0,
                multi_priority: row.get(4)?,
                multi_name:     row.get(5).unwrap_or_default(),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Delete all history for an image.
/// Mirrors the core of `dt_history_delete_on_image()` in history.c.
pub fn history_delete_on_image(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM main.history WHERE imgid = ?1", params![imgid])?;
    Ok(())
}

/// Delete history entries above a given step number (truncate).
/// Mirrors `dt_history_truncate_on_image()` in history.c:1348.
pub fn history_truncate_on_image(
    conn: &Connection,
    imgid: dt_imgid_t,
    history_end: i32,
) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM main.history WHERE imgid = ?1 AND num > ?2",
        params![imgid, history_end],
    )?;
    Ok(())
}

/// Return the maximum `num` (step index) in an image's history.
pub fn history_get_max_num(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<Option<i32>> {
    conn.query_row(
        "SELECT MAX(num) FROM main.history WHERE imgid = ?1",
        params![imgid],
        |row| row.get(0),
    )
    .map_err(|e| e)
    .map(|v: Option<i32>| v)
}

/// Check if a given operation name exists in the history.
/// Mirrors `dt_history_check_module_exists()` in history.c:1475.
pub fn history_module_exists(
    conn: &Connection,
    imgid: dt_imgid_t,
    operation: &str,
) -> rusqlite::Result<bool> {
    let count: i32 = conn.query_row(
        "SELECT COUNT(*) FROM main.history WHERE imgid = ?1 AND operation = ?2 AND enabled = 1",
        params![imgid, operation],
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
            &format!("file:hist_main{n}?mode=memory&cache=shared"), flags,
        ).unwrap();
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS main.history (
                imgid INTEGER, num INTEGER, module INTEGER,
                operation VARCHAR, op_params BLOB, enabled INTEGER,
                blendop_params BLOB, blendop_version INTEGER,
                multi_priority INTEGER, multi_name VARCHAR
            );
        ").unwrap();
        // Seed two history entries for image 1
        conn.execute_batch("
            INSERT INTO main.history (imgid, num, operation, enabled, multi_priority, multi_name)
                VALUES (1, 0, 'exposure',   1, 0, ''),
                       (1, 1, 'colorin',    1, 0, ''),
                       (1, 2, 'colorout',   0, 0, '');
        ").unwrap();
        conn
    }

    #[test]
    fn count_returns_all_entries() {
        let db = open_test_db();
        assert_eq!(history_count(&db, 1).unwrap(), 3);
    }

    #[test]
    fn count_zero_for_unknown_image() {
        let db = open_test_db();
        assert_eq!(history_count(&db, 99).unwrap(), 0);
    }

    #[test]
    fn list_returns_entries_in_order() {
        let db = open_test_db();
        let entries = history_list(&db, 1).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].operation, "exposure");
        assert_eq!(entries[0].num, 0);
        assert_eq!(entries[2].operation, "colorout");
    }

    #[test]
    fn delete_on_image_removes_all() {
        let db = open_test_db();
        history_delete_on_image(&db, 1).unwrap();
        assert_eq!(history_count(&db, 1).unwrap(), 0);
    }

    #[test]
    fn truncate_removes_entries_above_step() {
        let db = open_test_db();
        history_truncate_on_image(&db, 1, 1).unwrap(); // keep 0 and 1
        assert_eq!(history_count(&db, 1).unwrap(), 2);
    }

    #[test]
    fn get_max_num_returns_highest_step() {
        let db = open_test_db();
        assert_eq!(history_get_max_num(&db, 1).unwrap(), Some(2));
    }

    #[test]
    fn get_max_num_returns_none_for_empty() {
        let db = open_test_db();
        assert_eq!(history_get_max_num(&db, 99).unwrap(), None);
    }

    #[test]
    fn module_exists_true_for_enabled_op() {
        let db = open_test_db();
        assert!(history_module_exists(&db, 1, "exposure").unwrap());
    }

    #[test]
    fn module_exists_false_for_disabled_op() {
        let db = open_test_db();
        // colorout is disabled (enabled=0)
        assert!(!history_module_exists(&db, 1, "colorout").unwrap());
    }

    #[test]
    fn module_exists_false_for_absent_op() {
        let db = open_test_db();
        assert!(!history_module_exists(&db, 1, "sharpen").unwrap());
    }
}
