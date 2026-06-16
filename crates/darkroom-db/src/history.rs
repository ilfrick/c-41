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

/// Fields for a new `main.history` row. `op_params` is an opaque BLOB — the
/// serialised module parameters; its layout (and C-compatibility) is the
/// caller's concern. A struct (not positional args) keeps the two `i32` fields
/// (`module_version`, `multi_priority`) from being swapped at call sites.
pub struct NewHistoryEntry<'a> {
    pub imgid: dt_imgid_t,
    pub num: i32,
    pub operation: &'a str,
    /// The IOP version (the `module` column). darktable's reader compares this
    /// to the module's current version and rejects / legacy-converts the row on
    /// mismatch — a NULL/0 here would invalidate the row on load, so a faithful
    /// writer must set it (`develop.c` binds `module->version()` here).
    pub module_version: i32,
    pub enabled: bool,
    pub op_params: &'a [u8],
    pub multi_priority: i32,
    pub multi_name: &'a str,
}

/// Append a history entry. Mirrors the `INSERT INTO main.history` in
/// `dt_dev_write_history_item()` (history.c).
///
/// Pure append: the caller owns `num` and MUST keep `(imgid, num)` unique
/// (e.g. `history_get_max_num + 1`). The table has no UNIQUE constraint, so a
/// colliding `num` inserts a duplicate row that the C reader mis-loads as two
/// modules at one step.
pub fn history_add_entry(conn: &Connection, e: &NewHistoryEntry<'_>) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO main.history \
            (imgid, num, module, operation, op_params, enabled, multi_priority, multi_name) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            e.imgid, e.num, e.module_version, e.operation, e.op_params,
            e.enabled as i32, e.multi_priority, e.multi_name
        ],
    )?;
    Ok(())
}

/// Return the `op_params` BLOB of the highest-`num` (most recent) entry for the
/// given operation on an image, or `None` if the operation isn't in the history.
/// Used to restore a module's last-saved parameters.
pub fn history_get_op_params(
    conn: &Connection,
    imgid: dt_imgid_t,
    operation: &str,
) -> rusqlite::Result<Option<Vec<u8>>> {
    conn.query_row(
        "SELECT op_params FROM main.history \
         WHERE imgid = ?1 AND operation = ?2 ORDER BY num DESC LIMIT 1",
        params![imgid, operation],
        |row| row.get::<_, Option<Vec<u8>>>(0),
    )
    .or_else(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other),
    })
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
        // `Option` is load-bearing: `MAX(num)` over an empty history returns one
        // row holding NULL, so reading as `i32` would error/panic.
        |row| row.get::<_, Option<i32>>(0),
    )
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
        // Mirror the production main.history shape (database.c:3516), incl.
        // module + multi_name_hand_edited, so tests exercise the real columns.
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS main.history (
                imgid INTEGER, num INTEGER, module INTEGER,
                operation VARCHAR(256), op_params BLOB, enabled INTEGER,
                blendop_params BLOB, blendop_version INTEGER,
                multi_priority INTEGER, multi_name VARCHAR(256),
                multi_name_hand_edited INTEGER
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

    /// Build a velvia-ish entry for image 1 with module_version 1.
    fn entry<'a>(num: i32, op: &'a str, op_params: &'a [u8]) -> NewHistoryEntry<'a> {
        NewHistoryEntry {
            imgid: 1,
            num,
            operation: op,
            module_version: 1,
            enabled: true,
            op_params,
            multi_priority: 0,
            multi_name: "",
        }
    }

    #[test]
    fn add_entry_appends_and_roundtrips_op_params() {
        let db = open_test_db();
        let blob = vec![1u8, 2, 3, 4, 250];
        history_add_entry(&db, &entry(3, "velvia", &blob)).unwrap();
        assert_eq!(history_count(&db, 1).unwrap(), 4);
        let got = history_get_op_params(&db, 1, "velvia").unwrap();
        assert_eq!(got.as_deref(), Some(blob.as_slice()));
        // the appended entry is listed in num order at the end
        let entries = history_list(&db, 1).unwrap();
        assert_eq!(entries.last().unwrap().operation, "velvia");
        assert!(entries.last().unwrap().enabled);
    }

    #[test]
    fn add_entry_writes_module_version_column() {
        let db = open_test_db();
        let mut e = entry(3, "velvia", &[1, 2]);
        e.module_version = 7;
        history_add_entry(&db, &e).unwrap();
        let m: i32 = db
            .query_row(
                "SELECT module FROM main.history WHERE imgid=1 AND operation='velvia'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(m, 7); // not NULL/0 — would otherwise invalidate the row on C load
    }

    #[test]
    fn get_op_params_returns_latest_num() {
        let db = open_test_db();
        history_add_entry(&db, &entry(5, "velvia", &[10])).unwrap();
        history_add_entry(&db, &entry(7, "velvia", &[20])).unwrap(); // newer num
        history_add_entry(&db, &entry(6, "velvia", &[15])).unwrap(); // older num
        assert_eq!(
            history_get_op_params(&db, 1, "velvia").unwrap().as_deref(),
            Some([20u8].as_slice())
        );
    }

    #[test]
    fn empty_blob_roundtrips_as_some_empty() {
        let db = open_test_db();
        history_add_entry(&db, &entry(3, "velvia", &[])).unwrap();
        assert_eq!(history_get_op_params(&db, 1, "velvia").unwrap(), Some(vec![]));
    }

    #[test]
    fn null_op_params_reads_as_none() {
        // A NULL op_params blob is reported as None — same as no row. That's the
        // intended conflation for "no params to restore".
        let db = open_test_db();
        db.execute(
            "INSERT INTO main.history (imgid, num, operation, op_params, enabled) \
             VALUES (1, 9, 'velvia', NULL, 1)",
            [],
        )
        .unwrap();
        assert_eq!(history_get_op_params(&db, 1, "velvia").unwrap(), None);
    }

    #[test]
    fn duplicate_imgid_num_both_inserted() {
        // No UNIQUE on (imgid, num): a colliding num yields two rows (documents
        // that the UI layer must own num uniqueness).
        let db = open_test_db();
        history_add_entry(&db, &entry(3, "velvia", &[1])).unwrap();
        history_add_entry(&db, &entry(3, "sharpen", &[2])).unwrap();
        assert_eq!(history_count(&db, 1).unwrap(), 5);
    }

    #[test]
    fn get_op_params_none_for_absent_op() {
        let db = open_test_db();
        assert_eq!(history_get_op_params(&db, 1, "sharpen").unwrap(), None);
    }
}
