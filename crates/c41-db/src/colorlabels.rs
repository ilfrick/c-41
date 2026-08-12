//! Colour-label DAO — per-image colour flags (red/yellow/green/blue/purple).
//!
//! darktable stores colour labels in `main.color_labels(imgid, color)` with a
//! `UNIQUE(imgid, color)` index, one row per assigned label. An image can carry
//! any subset of the five colours (unlike the single star rating in
//! `images.flags`), so the natural in-memory shape is a 5-bit mask: bit `c` set
//! means colour `c` is assigned. This mirrors the C `dt_colorlabels_get_labels`,
//! which folds the rows into `colors |= 1 << color`.
//!
//! Colour indices (darktable convention): 0 red, 1 yellow, 2 green, 3 blue,
//! 4 purple. There are exactly [`COLOR_COUNT`] colours.

use rusqlite::{params, Connection};
use c41_sys::dt_imgid_t;

/// Number of distinct colour labels (red/yellow/green/blue/purple).
pub const COLOR_COUNT: u8 = 5;

/// Read an image's assigned colour labels as a 5-bit mask (bit `c` = colour `c`).
/// Mirrors C `dt_colorlabels_get_labels` (`src/common/colorlabels.c`), but rows
/// with an out-of-range `color` are skipped **by design** (the C reader would
/// `1 << color` into a wider int; here the domain is `0..COLOR_COUNT`, so a stray
/// value is dropped rather than smeared into a high/invalid bit). Returns `0` when
/// the image has no labels.
pub fn color_labels_get(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<u8> {
    let mut stmt = conn.prepare(
        "SELECT color FROM main.color_labels WHERE imgid = ?1",
    )?;
    let mut mask: u8 = 0;
    let rows = stmt.query_map(params![imgid], |row| row.get::<_, i64>(0))?;
    for color in rows {
        let c = color?;
        if (0..i64::from(COLOR_COUNT)).contains(&c) {
            mask |= 1 << c;
        }
    }
    Ok(mask)
}

/// Assign colour `color` to an image (C `dt_colorlabels_set_label`). Idempotent:
/// the `UNIQUE(imgid, color)` index makes a repeat a no-op (`INSERT OR IGNORE`).
/// An out-of-range `color` (`>= COLOR_COUNT`) is rejected as a no-op so the table
/// can't hold a "ghost" row that [`color_labels_get`] would silently drop on read.
/// Returns `true` if a row was newly inserted, `false` otherwise.
pub fn color_label_set(conn: &Connection, imgid: dt_imgid_t, color: u8) -> rusqlite::Result<bool> {
    if color >= COLOR_COUNT {
        return Ok(false);
    }
    let n = conn.execute(
        "INSERT OR IGNORE INTO main.color_labels (imgid, color) VALUES (?1, ?2)",
        params![imgid, color],
    )?;
    Ok(n > 0)
}

/// Remove colour `color` from an image (C `dt_colorlabels_remove_label`). An
/// out-of-range `color` is a no-op. Returns `true` if a row was deleted, `false`
/// if the label wasn't present.
pub fn color_label_remove(conn: &Connection, imgid: dt_imgid_t, color: u8) -> rusqlite::Result<bool> {
    if color >= COLOR_COUNT {
        return Ok(false);
    }
    let n = conn.execute(
        "DELETE FROM main.color_labels WHERE imgid = ?1 AND color = ?2",
        params![imgid, color],
    )?;
    Ok(n > 0)
}

/// Toggle colour `color` on an image: remove it if present, else assign it
/// (C `DT_CA_TOGGLE`). Returns the resulting state (`true` = label is now set).
/// One read of current state then the opposite write, so the callers (UI click,
/// future keyboard shortcut) share identical semantics. Not wrapped in a
/// transaction: the lighttable is single-writer, so the read-then-write gap is a
/// non-issue (revisit if a concurrent writer is ever introduced). An out-of-range
/// `color` is rejected as a no-op (also guards `1 << color` against overflow).
pub fn color_label_toggle(conn: &Connection, imgid: dt_imgid_t, color: u8) -> rusqlite::Result<bool> {
    if color >= COLOR_COUNT {
        return Ok(false);
    }
    let present = color_labels_get(conn, imgid)? & (1 << color) != 0;
    if present {
        color_label_remove(conn, imgid, color)?;
        Ok(false)
    } else {
        color_label_set(conn, imgid, color)?;
        Ok(true)
    }
}

/// Remove every colour label from an image (C `dt_colorlabels_remove_all_labels`).
/// Returns the number of rows deleted.
pub fn color_labels_remove_all(conn: &Connection, imgid: dt_imgid_t) -> rusqlite::Result<usize> {
    let n = conn.execute(
        "DELETE FROM main.color_labels WHERE imgid = ?1",
        params![imgid],
    )?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::OpenFlags;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn open_test_db() -> Connection {
        static COUNTER: AtomicU32 = AtomicU32::new(1);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_URI;
        let conn = Connection::open_with_flags(
            &format!("file:cl_main{n}?mode=memory&cache=shared"), flags,
        ).unwrap();
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS main.color_labels (imgid INTEGER, color INTEGER);
            CREATE UNIQUE INDEX IF NOT EXISTS main.color_labels_idx
                ON color_labels (imgid, color);
        ").unwrap();
        conn
    }

    #[test]
    fn get_is_zero_when_unlabelled() {
        let db = open_test_db();
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0);
    }

    #[test]
    fn set_then_get_folds_into_bitmask() {
        let db = open_test_db();
        assert!(color_label_set(&db, 100, 0).unwrap()); // red
        assert!(color_label_set(&db, 100, 2).unwrap()); // green
        // bits 0 and 2 set → 0b101 = 5
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0b0000_0101);
    }

    #[test]
    fn set_is_idempotent() {
        let db = open_test_db();
        assert!(color_label_set(&db, 100, 3).unwrap());  // newly inserted
        assert!(!color_label_set(&db, 100, 3).unwrap()); // already present
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0b0000_1000);
    }

    #[test]
    fn remove_reports_whether_present() {
        let db = open_test_db();
        color_label_set(&db, 100, 1).unwrap();
        assert!(color_label_remove(&db, 100, 1).unwrap());  // was present
        assert!(!color_label_remove(&db, 100, 1).unwrap()); // already gone
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0);
    }

    #[test]
    fn toggle_flips_and_returns_new_state() {
        let db = open_test_db();
        assert!(color_label_toggle(&db, 100, 4).unwrap());  // now set → true
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0b0001_0000);
        assert!(!color_label_toggle(&db, 100, 4).unwrap()); // now cleared → false
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0);
    }

    #[test]
    fn labels_are_per_image_independent() {
        let db = open_test_db();
        color_label_set(&db, 100, 0).unwrap();
        color_label_set(&db, 200, 2).unwrap();
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0b0000_0001);
        assert_eq!(color_labels_get(&db, 200).unwrap(), 0b0000_0100);
    }

    #[test]
    fn out_of_range_color_is_rejected_on_write_and_dropped_on_read() {
        let db = open_test_db();
        // Write guard: a colour >= COLOR_COUNT writes nothing (no ghost row).
        assert!(!color_label_set(&db, 100, COLOR_COUNT).unwrap());
        assert!(!color_label_set(&db, 100, 9).unwrap());
        assert!(!color_label_toggle(&db, 100, 9).unwrap());
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0);
        // Read guard: a stray out-of-range row inserted directly is dropped by
        // `_get` (never smeared into a high bit), and a valid sibling still reads.
        db.execute("INSERT INTO main.color_labels (imgid, color) VALUES (100, 9)", []).unwrap();
        color_label_set(&db, 100, 1).unwrap();
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0b0000_0010);
    }

    #[test]
    fn remove_all_clears_only_that_image() {
        let db = open_test_db();
        color_label_set(&db, 100, 0).unwrap();
        color_label_set(&db, 100, 1).unwrap();
        color_label_set(&db, 200, 3).unwrap();
        assert_eq!(color_labels_remove_all(&db, 100).unwrap(), 2);
        assert_eq!(color_labels_get(&db, 100).unwrap(), 0);
        assert_eq!(color_labels_get(&db, 200).unwrap(), 0b0000_1000); // untouched
    }
}
