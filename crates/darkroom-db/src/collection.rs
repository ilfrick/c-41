//! Collection queries — Rust impl of the simple SQL-only parts of
//! src/common/collection.c.
//!
//! The full collection object (dt_collection_t) with its query-builder,
//! filter flags, sort orders, and GUI signals stays in C for now.
//! This module covers the standalone SQL operations that need no C state.

use rusqlite::{Connection, params};

/// Count images currently in the selection set.
/// Mirrors `dt_collection_get_selected_count()` in collection.c:907.
pub fn collection_get_selected_count(conn: &Connection) -> rusqlite::Result<u32> {
    conn.query_row("SELECT COUNT(*) FROM main.selected_images", [], |row| row.get(0))
}

/// Count images in the active collection (memory.collected_images).
/// Mirrors `dt_collection_get_collected_count()` in collection.c:920.
pub fn collection_get_collected_count(conn: &Connection) -> rusqlite::Result<u32> {
    conn.query_row("SELECT COUNT(*) FROM memory.collected_images", [], |row| row.get(0))
}

/// Add an image to the selection.
pub fn collection_select_image(conn: &Connection, imgid: i32) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO main.selected_images (imgid) VALUES (?1)",
        params![imgid],
    )?;
    Ok(())
}

/// Remove an image from the selection.
pub fn collection_deselect_image(conn: &Connection, imgid: i32) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM main.selected_images WHERE imgid = ?1", params![imgid])?;
    Ok(())
}

/// Clear the entire selection.
pub fn collection_clear_selection(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM main.selected_images", [])?;
    Ok(())
}

/// Return the list of currently selected image IDs.
pub fn collection_get_selected(conn: &Connection) -> rusqlite::Result<Vec<i32>> {
    let mut stmt = conn.prepare("SELECT imgid FROM main.selected_images ORDER BY imgid")?;
    let ids = stmt.query_map([], |row| row.get(0))?.collect::<Result<Vec<_>, _>>()?;
    Ok(ids)
}

/// Return true if the image is currently selected.
pub fn collection_image_is_selected(conn: &Connection, imgid: i32) -> rusqlite::Result<bool> {
    let count: i32 = conn.query_row(
        "SELECT COUNT(*) FROM main.selected_images WHERE imgid = ?1",
        params![imgid],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

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
            &format!("file:col_main{n}?mode=memory&cache=shared"), flags,
        ).unwrap();
        conn.execute_batch(&format!(
            "ATTACH DATABASE 'file:col_mem{n}?mode=memory&cache=shared' AS memory;"
        )).unwrap();
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS main.selected_images (imgid INTEGER PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS memory.collected_images (imgid INTEGER PRIMARY KEY);
        ").unwrap();
        conn
    }

    #[test]
    fn selected_count_zero_initially() {
        let db = open_test_db();
        assert_eq!(collection_get_selected_count(&db).unwrap(), 0);
    }

    #[test]
    fn select_increments_count() {
        let db = open_test_db();
        collection_select_image(&db, 1).unwrap();
        collection_select_image(&db, 2).unwrap();
        assert_eq!(collection_get_selected_count(&db).unwrap(), 2);
    }

    #[test]
    fn select_is_idempotent() {
        let db = open_test_db();
        collection_select_image(&db, 5).unwrap();
        collection_select_image(&db, 5).unwrap();
        assert_eq!(collection_get_selected_count(&db).unwrap(), 1);
    }

    #[test]
    fn deselect_removes_image() {
        let db = open_test_db();
        collection_select_image(&db, 10).unwrap();
        collection_deselect_image(&db, 10).unwrap();
        assert_eq!(collection_get_selected_count(&db).unwrap(), 0);
    }

    #[test]
    fn clear_selection_empties_all() {
        let db = open_test_db();
        collection_select_image(&db, 1).unwrap();
        collection_select_image(&db, 2).unwrap();
        collection_clear_selection(&db).unwrap();
        assert_eq!(collection_get_selected_count(&db).unwrap(), 0);
    }

    #[test]
    fn get_selected_returns_ids_in_order() {
        let db = open_test_db();
        collection_select_image(&db, 3).unwrap();
        collection_select_image(&db, 1).unwrap();
        collection_select_image(&db, 2).unwrap();
        assert_eq!(collection_get_selected(&db).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn image_is_selected_reflects_state() {
        let db = open_test_db();
        assert!(!collection_image_is_selected(&db, 7).unwrap());
        collection_select_image(&db, 7).unwrap();
        assert!(collection_image_is_selected(&db, 7).unwrap());
    }

    #[test]
    fn collected_count_reflects_memory_table() {
        let db = open_test_db();
        db.execute("INSERT INTO memory.collected_images (imgid) VALUES (1)", []).unwrap();
        db.execute("INSERT INTO memory.collected_images (imgid) VALUES (2)", []).unwrap();
        assert_eq!(collection_get_collected_count(&db).unwrap(), 2);
    }
}
