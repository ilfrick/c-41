//! Tag management — Rust implementation of src/common/tags.c.
//!
//! Phase 2-db-1: core CRUD operations. The C side still owns the undo
//! system, GLib signal emission, and darktable.db lifetime; those are
//! threaded through here via a raw sqlite3 pointer for now and will be
//! replaced once the undo/signal infrastructure is in Rust.
//!
//! Schema (in data.db):
//!   data.tags         (id INTEGER PK, name VARCHAR UNIQUE, synonyms VARCHAR, flags INTEGER)
//!   main.tagged_images (imgid INTEGER, tagid INTEGER, position INTEGER, PK(imgid,tagid))
//!   memory.darktable_tags (tagid INTEGER PK)   -- internal dt: tags

use rusqlite::{Connection, OptionalExtension, params};

pub type TagId = u32;

/// Open an isolated in-memory database with the full schema for testing.
/// Each call gets a unique URI so parallel tests don't share state.
#[cfg(test)]
pub(crate) fn open_test_db() -> Connection {
    use rusqlite::OpenFlags;
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);

    let main_uri  = format!("file:testmain{n}?mode=memory&cache=shared");
    let data_uri  = format!("file:testdata{n}?mode=memory&cache=shared");
    let mem_uri   = format!("file:testmem{n}?mode=memory&cache=shared");

    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_CREATE
        | OpenFlags::SQLITE_OPEN_URI;
    let conn = Connection::open_with_flags(&main_uri, flags).expect("open test db");

    conn.execute_batch(&format!("
        ATTACH DATABASE '{data_uri}' AS data;
        ATTACH DATABASE '{mem_uri}'  AS memory;
    ")).expect("attach schemas");
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS data.tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name     VARCHAR,
            synonyms VARCHAR,
            flags    INTEGER
        );
    ").expect("create data.tags");
    // SQLite: schema prefix goes on INDEX name, NOT the table name.
    conn.execute_batch("
        CREATE UNIQUE INDEX IF NOT EXISTS data.tags_name_idx ON tags (name);
    ").expect("create tags index");
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS main.tagged_images (
            imgid INTEGER, tagid INTEGER, position INTEGER,
            PRIMARY KEY (imgid, tagid)
        );
        CREATE TABLE IF NOT EXISTS memory.darktable_tags (tagid INTEGER PRIMARY KEY);
    ").expect("create remaining tables");
    conn
}

// ── Core tag operations ───────────────────────────────────────────────────────

/// Upsert a tag by name. Returns the tag ID (existing or newly created).
/// An empty name returns `None`.
pub fn tag_new(conn: &Connection, name: &str) -> rusqlite::Result<Option<TagId>> {
    if name.is_empty() {
        return Ok(None);
    }

    // Check if already exists
    let existing: Option<TagId> = conn
        .query_row(
            "SELECT id FROM data.tags WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )
        .optional()?;

    if let Some(id) = existing {
        return Ok(Some(id));
    }

    // Insert new
    conn.execute(
        "INSERT INTO data.tags (id, name) VALUES (NULL, ?1)",
        params![name],
    )?;
    let id: TagId = conn
        .query_row(
            "SELECT id FROM data.tags WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )?;

    // If it's an internal darktable tag, record it in the in-memory table
    if name.starts_with("darktable|") {
        conn.execute(
            "INSERT OR IGNORE INTO memory.darktable_tags (tagid) VALUES (?1)",
            params![id],
        )?;
    }

    Ok(Some(id))
}

/// Check if a tag with `name` exists. Returns its ID if so.
pub fn tag_exists(conn: &Connection, name: &str) -> rusqlite::Result<Option<TagId>> {
    conn.query_row(
        "SELECT id FROM data.tags WHERE name = ?1",
        params![name],
        |row| row.get(0),
    )
    .optional()
}

/// Get the name of a tag by ID. Returns `None` if the tag does not exist.
pub fn tag_get_name(conn: &Connection, tagid: TagId) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT name FROM data.tags WHERE id = ?1",
        params![tagid],
        |row| row.get(0),
    )
    .optional()
}

/// Rename a single tag by id (flat, no hierarchy awareness). The UI renames via
/// [`tag_rename_subtree`] instead, so a parent rename carries its descendants and
/// can't orphan them; this primitive is kept for callers that genuinely want to
/// touch exactly one row.
pub fn tag_rename(conn: &Connection, tagid: TagId, new_name: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE data.tags SET name = ?1 WHERE id = ?2",
        params![new_name, tagid],
    )?;
    Ok(())
}

/// Escape the SQL `LIKE` metacharacters (`%`, `_`) and the escape char (`\`) in
/// `s`, for use as a literal segment in a `LIKE … ESCAPE '\'` pattern. Backslash
/// is escaped first so the escapes added for `%`/`_` aren't re-escaped.
fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

/// Rename a tag **and all its hierarchical descendants** by rewriting the `old`
/// path prefix to `new`. darktable stores hierarchy as `parent|child` in
/// `data.tags.name`, so renaming `places|Italy` → `places|Italia` must also move
/// `places|Italy|Rome` → `places|Italia|Rome`. The leading `old` segment of each
/// matched name is replaced with `new`, preserving the descendant suffix.
///
/// `substr` uses SQLite's `length(?1)` (character count), NOT Rust's byte length,
/// so multi-byte tag names (`Café|…`) rewrite at the correct offset. The whole
/// `UPDATE` is one atomic statement: if any rewritten name collides with the
/// `UNIQUE` `data.tags.name` index (the destination path already exists) the
/// statement rolls back entirely and the error surfaces — no partial rename and
/// no silent merge (merging into an existing tag is a later concern). Returns the
/// number of tag rows rewritten.
///
/// The caller must pass a non-empty, `|`-delimited `old` path (the UI builds it
/// from real tag segments, never empty/trailing-`|`) — an empty `old` would make
/// the LIKE pattern `|%` and match unrelated names. The UI also forbids a `|` in
/// the renamed segment, so the rewrite cannot map a row onto another row it is
/// itself moving (a self-collision); see `respliced_tag_path`.
pub fn tag_rename_subtree(conn: &Connection, old: &str, new: &str) -> rusqlite::Result<usize> {
    let descendants = format!("{}|%", escape_like(old));
    let n = conn.execute(
        "UPDATE data.tags \
         SET name = ?2 || substr(name, length(?1) + 1) \
         WHERE name = ?1 OR name LIKE ?3 ESCAPE '\\'",
        params![old, new, descendants],
    )?;
    Ok(n)
}

/// Attach a tag to an image. Returns `true` if the row was newly inserted.
pub fn tag_attach(conn: &Connection, tagid: TagId, imgid: i32) -> rusqlite::Result<bool> {
    let rows = conn.execute(
        "INSERT OR IGNORE INTO main.tagged_images (imgid, tagid, position) VALUES (?1, ?2, 0)",
        params![imgid, tagid],
    )?;
    Ok(rows > 0)
}

/// Detach a tag from an image. Returns `true` if a row was removed.
pub fn tag_detach(conn: &Connection, tagid: TagId, imgid: i32) -> rusqlite::Result<bool> {
    let rows = conn.execute(
        "DELETE FROM main.tagged_images WHERE tagid = ?1 AND imgid = ?2",
        params![tagid, imgid],
    )?;
    Ok(rows > 0)
}

/// Count how many user tags are attached to an image.
/// Internal `darktable|*` tags are excluded by default when `include_dt_tags` is false.
pub fn tag_count_attached(
    conn: &Connection,
    imgid: i32,
    include_dt_tags: bool,
) -> rusqlite::Result<u32> {
    let count: u32 = if include_dt_tags {
        conn.query_row(
            "SELECT COUNT(*) FROM main.tagged_images WHERE imgid = ?1",
            params![imgid],
            |row| row.get(0),
        )?
    } else {
        conn.query_row(
            "SELECT COUNT(*) FROM main.tagged_images ti
             JOIN data.tags t ON t.id = ti.tagid
             WHERE ti.imgid = ?1 AND t.name NOT LIKE 'darktable|%'",
            params![imgid],
            |row| row.get(0),
        )?
    };
    Ok(count)
}

/// List tag IDs attached to an image.
pub fn tag_get_attached(conn: &Connection, imgid: i32) -> rusqlite::Result<Vec<TagId>> {
    let mut stmt = conn.prepare(
        "SELECT tagid FROM main.tagged_images WHERE imgid = ?1 ORDER BY tagid",
    )?;
    let ids = stmt
        .query_map(params![imgid], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ids)
}

/// List every user tag with the number of images it is attached to, ordered by
/// name. Internal darktable tags (`darktable|…`, e.g. the colour-label and
/// rejected tags) are excluded so this drives a user-facing tag browser. The
/// count comes from a `LEFT JOIN` so tags attached to nothing still appear (count
/// 0). Returns `(id, name, count)` triples.
pub fn tag_list_with_counts(conn: &Connection) -> rusqlite::Result<Vec<(TagId, String, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT t.id, t.name, COUNT(ti.imgid) \
         FROM data.tags t \
         LEFT JOIN main.tagged_images ti ON ti.tagid = t.id \
         WHERE t.name NOT LIKE 'darktable|%' \
         GROUP BY t.id, t.name \
         ORDER BY t.name",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, TagId>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Delete a tag and all its image associations.
pub fn tag_delete(conn: &Connection, tagid: TagId) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM main.tagged_images WHERE tagid = ?1",
        params![tagid],
    )?;
    conn.execute(
        "DELETE FROM memory.darktable_tags WHERE tagid = ?1",
        params![tagid],
    )?;
    conn.execute("DELETE FROM data.tags WHERE id = ?1", params![tagid])?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tag_creates_and_returns_id() {
        let db = open_test_db();
        let id = tag_new(&db, "landscape").unwrap().unwrap();
        assert!(id > 0);
    }

    #[test]
    fn new_tag_is_idempotent() {
        let db = open_test_db();
        let id1 = tag_new(&db, "portrait").unwrap().unwrap();
        let id2 = tag_new(&db, "portrait").unwrap().unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn empty_name_returns_none() {
        let db = open_test_db();
        assert_eq!(tag_new(&db, "").unwrap(), None);
    }

    #[test]
    fn exists_returns_id_when_present() {
        let db = open_test_db();
        let id = tag_new(&db, "trip").unwrap().unwrap();
        assert_eq!(tag_exists(&db, "trip").unwrap(), Some(id));
    }

    #[test]
    fn exists_returns_none_when_absent() {
        let db = open_test_db();
        assert_eq!(tag_exists(&db, "nope").unwrap(), None);
    }

    #[test]
    fn get_name_roundtrips() {
        let db = open_test_db();
        let id = tag_new(&db, "city").unwrap().unwrap();
        assert_eq!(tag_get_name(&db, id).unwrap().as_deref(), Some("city"));
    }

    #[test]
    fn rename_changes_name() {
        let db = open_test_db();
        let id = tag_new(&db, "old").unwrap().unwrap();
        tag_rename(&db, id, "new").unwrap();
        assert_eq!(tag_get_name(&db, id).unwrap().as_deref(), Some("new"));
    }

    #[test]
    fn rename_subtree_rewrites_node_and_descendants() {
        let db = open_test_db();
        let parent = tag_new(&db, "places|Italy").unwrap().unwrap();
        let child  = tag_new(&db, "places|Italy|Rome").unwrap().unwrap();
        let n = tag_rename_subtree(&db, "places|Italy", "places|Italia").unwrap();
        assert_eq!(n, 2);
        assert_eq!(tag_get_name(&db, parent).unwrap().as_deref(), Some("places|Italia"));
        assert_eq!(tag_get_name(&db, child).unwrap().as_deref(),  Some("places|Italia|Rome"));
    }

    #[test]
    fn rename_subtree_leaves_siblings_and_prefixes_untouched() {
        let db = open_test_db();
        // `places|Italian` shares the textual prefix "places|Italy" only up to a
        // point — it must NOT match (the LIKE anchors on `places|Italy|`), and the
        // sibling `places|France` is independent.
        let italy   = tag_new(&db, "places|Italy").unwrap().unwrap();
        let italian = tag_new(&db, "places|Italian").unwrap().unwrap();
        let france  = tag_new(&db, "places|France").unwrap().unwrap();
        let n = tag_rename_subtree(&db, "places|Italy", "places|Italia").unwrap();
        assert_eq!(n, 1);
        assert_eq!(tag_get_name(&db, italy).unwrap().as_deref(),   Some("places|Italia"));
        assert_eq!(tag_get_name(&db, italian).unwrap().as_deref(), Some("places|Italian"));
        assert_eq!(tag_get_name(&db, france).unwrap().as_deref(),  Some("places|France"));
    }

    #[test]
    fn rename_subtree_handles_multibyte_prefix() {
        let db = open_test_db();
        // `length()` is SQLite's character count, so the substr offset is correct
        // even when the prefix has multi-byte UTF-8 (would be wrong with byte len).
        let parent = tag_new(&db, "Café").unwrap().unwrap();
        let child  = tag_new(&db, "Café|latte").unwrap().unwrap();
        let n = tag_rename_subtree(&db, "Café", "Coffee").unwrap();
        assert_eq!(n, 2);
        assert_eq!(tag_get_name(&db, parent).unwrap().as_deref(), Some("Coffee"));
        assert_eq!(tag_get_name(&db, child).unwrap().as_deref(),  Some("Coffee|latte"));
    }

    #[test]
    fn rename_subtree_rolls_back_on_unique_collision() {
        let db = open_test_db();
        // Renaming `a|x` → `a|y` collides with the existing `a|y`; the UNIQUE index
        // must abort the whole UPDATE, leaving every name unchanged.
        let ax = tag_new(&db, "a|x").unwrap().unwrap();
        let ay = tag_new(&db, "a|y").unwrap().unwrap();
        assert!(tag_rename_subtree(&db, "a|x", "a|y").is_err());
        assert_eq!(tag_get_name(&db, ax).unwrap().as_deref(), Some("a|x"));
        assert_eq!(tag_get_name(&db, ay).unwrap().as_deref(), Some("a|y"));
    }

    #[test]
    fn rename_subtree_deepening_collision_aborts_atomically() {
        let db = open_test_db();
        // The UI forbids a `|` in the segment so this can't be reached from the
        // popover, but the DAO must still behave: deepening `a|b` → `a|b|b` maps
        // `a|b`→`a|b|b` and the existing `a|b|b`→`a|b|b|b`. SQLite checks the
        // UNIQUE index per row in unspecified order, so if `a|b` is written before
        // the existing `a|b|b` is moved aside it transiently duplicates and the
        // statement ABORTs. Pin that it's all-or-nothing — never a partial rename.
        let ab  = tag_new(&db, "a|b").unwrap().unwrap();
        let abb = tag_new(&db, "a|b|b").unwrap().unwrap();
        let res = tag_rename_subtree(&db, "a|b", "a|b|b");
        match res {
            Ok(n) => {
                assert_eq!(n, 2);
                assert_eq!(tag_get_name(&db, ab).unwrap().as_deref(),  Some("a|b|b"));
                assert_eq!(tag_get_name(&db, abb).unwrap().as_deref(), Some("a|b|b|b"));
            }
            Err(_) => {
                // Aborted: both rows must be exactly as they started.
                assert_eq!(tag_get_name(&db, ab).unwrap().as_deref(),  Some("a|b"));
                assert_eq!(tag_get_name(&db, abb).unwrap().as_deref(), Some("a|b|b"));
            }
        }
    }

    #[test]
    fn escape_like_escapes_metachars_backslash_first() {
        // Backslash MUST be escaped before `%`/`_` so their added escapes aren't
        // themselves re-escaped (a reorder would silently double-escape).
        assert_eq!(escape_like("a%_\\b"), "a\\%\\_\\\\b");
        assert_eq!(escape_like("plain"), "plain");
    }

    #[test]
    fn attach_and_detach_image() {
        let db = open_test_db();
        let tid = tag_new(&db, "beach").unwrap().unwrap();
        assert!(tag_attach(&db, tid, 1001).unwrap());
        assert_eq!(tag_count_attached(&db, 1001, true).unwrap(), 1);
        assert!(tag_detach(&db, tid, 1001).unwrap());
        assert_eq!(tag_count_attached(&db, 1001, true).unwrap(), 0);
    }

    #[test]
    fn attach_is_idempotent() {
        let db = open_test_db();
        let tid = tag_new(&db, "sky").unwrap().unwrap();
        tag_attach(&db, tid, 42).unwrap();
        let second = tag_attach(&db, tid, 42).unwrap();
        assert!(!second); // no new row
        assert_eq!(tag_count_attached(&db, 42, true).unwrap(), 1);
    }

    #[test]
    fn get_attached_returns_all_tag_ids() {
        let db = open_test_db();
        let t1 = tag_new(&db, "a").unwrap().unwrap();
        let t2 = tag_new(&db, "b").unwrap().unwrap();
        tag_attach(&db, t1, 99).unwrap();
        tag_attach(&db, t2, 99).unwrap();
        let ids = tag_get_attached(&db, 99).unwrap();
        assert!(ids.contains(&t1) && ids.contains(&t2));
    }

    #[test]
    fn dt_tag_registered_in_memory_table() {
        let db = open_test_db();
        let id = tag_new(&db, "darktable|color|red").unwrap().unwrap();
        let count: u32 = db
            .query_row(
                "SELECT COUNT(*) FROM memory.darktable_tags WHERE tagid = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn count_excludes_dt_tags_by_default() {
        let db = open_test_db();
        let user = tag_new(&db, "sunset").unwrap().unwrap();
        let dt   = tag_new(&db, "darktable|style").unwrap().unwrap();
        tag_attach(&db, user, 7).unwrap();
        tag_attach(&db, dt,   7).unwrap();
        assert_eq!(tag_count_attached(&db, 7, false).unwrap(), 1); // user only
        assert_eq!(tag_count_attached(&db, 7, true).unwrap(),  2); // all
    }

    #[test]
    fn list_with_counts_orders_by_name_counts_and_excludes_dt_tags() {
        let db = open_test_db();
        let beach = tag_new(&db, "beach").unwrap().unwrap();
        let _city = tag_new(&db, "city").unwrap().unwrap();   // attached to nothing
        let arch  = tag_new(&db, "arch").unwrap().unwrap();
        let _dt   = tag_new(&db, "darktable|color|red").unwrap().unwrap();
        tag_attach(&db, beach, 1).unwrap();
        tag_attach(&db, beach, 2).unwrap();
        tag_attach(&db, arch,  1).unwrap();

        let list = tag_list_with_counts(&db).unwrap();
        // Internal darktable| tags excluded; user tags ordered by name.
        let names: Vec<&str> = list.iter().map(|(_, n, _)| n.as_str()).collect();
        assert_eq!(names, ["arch", "beach", "city"]);
        // Counts: arch=1, beach=2, city=0 (LEFT JOIN keeps the unattached tag).
        let by_name = |want: &str| list.iter().find(|(_, n, _)| n == want).unwrap().2;
        assert_eq!(by_name("arch"), 1);
        assert_eq!(by_name("beach"), 2);
        assert_eq!(by_name("city"), 0);
    }

    #[test]
    fn list_with_counts_excludes_only_pipe_namespaced_dt_tags() {
        // Guard the `NOT LIKE 'darktable|%'` filter: only the pipe-namespaced
        // internal tag is hidden; user tags that merely share the prefix stay.
        let db = open_test_db();
        tag_new(&db, "darkroom").unwrap();          // unrelated user tag
        tag_new(&db, "darktable").unwrap();         // bare word, not namespaced
        tag_new(&db, "darktable|style|foo").unwrap(); // internal → excluded
        let names: Vec<String> =
            tag_list_with_counts(&db).unwrap().into_iter().map(|(_, n, _)| n).collect();
        assert!(names.contains(&"darkroom".to_string()));
        assert!(names.contains(&"darktable".to_string()));
        assert!(!names.iter().any(|n| n.starts_with("darktable|")));
    }

    #[test]
    fn delete_removes_tag_and_associations() {
        let db = open_test_db();
        let id = tag_new(&db, "temp").unwrap().unwrap();
        tag_attach(&db, id, 5).unwrap();
        tag_delete(&db, id).unwrap();
        assert_eq!(tag_exists(&db, "temp").unwrap(), None);
        assert_eq!(tag_count_attached(&db, 5, true).unwrap(), 0);
    }
}
