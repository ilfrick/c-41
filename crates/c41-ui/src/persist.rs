//! Persistence of the darkroom view's live [`PreviewParams`] to the database,
//! so edits survive closing/reopening an image (RUST_MIGRATION_PLAN.md Phase 3
//! milestone 1 — module/edit state to the db).
//!
//! Stored in a dedicated, c41-ui-owned table `main.darkroom_preview`
//! (`imgid` PRIMARY KEY, `params` BLOB) — one row per image, written with a real
//! `ON CONFLICT` upsert. A private table (rather than `main.history`) keeps our
//! own blob layout out of darktable's IOP history: `main.history.num` is a
//! rewritable stack index the C app compresses/truncates, so squatting there
//! risks `num` collisions; the PK here makes the single-row invariant
//! structural and the write atomic.
//!
//! The table is **best-effort**: it's a c41-ui-private table that darktable
//! doesn't know about, so an aggressive C-side schema rebuild (copy-known-tables
//! into a fresh db) could drop it — in which case load falls back to defaults,
//! by design.

use rusqlite::Connection;
use c41_core::geometry::Geometry;
use c41_core::rawimage::DemosaicMethod;
use crate::history::HistoryStack;
use crate::preview::PreviewParams;

/// DDL for the preview-params table. `CREATE … IF NOT EXISTS` on every save so
/// it works against a fresh darktable `library.db` that predates this feature.
const PREVIEW_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS main.darkroom_preview \
     (imgid INTEGER PRIMARY KEY, params BLOB NOT NULL)";

/// DDL for the edit-history table (one blob per image, same private-table
/// rationale as `darkroom_preview`). A separate table so persisting history
/// never disturbs the params row read by the backward-compatible `load_saved`
/// path (old dbs have a params row but no history row → history falls back to a
/// fresh single-entry stack seeded from those params).
const HISTORY_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS main.darkroom_history \
     (imgid INTEGER PRIMARY KEY, stack BLOB NOT NULL)";

/// DDL for the per-image Bayer demosaic-method table (same private-table
/// rationale as `darkroom_preview`). A separate table — NOT a field of the
/// params blob — because the demosaic choice is *decode-time* state: changing
/// it re-decodes the raw, whereas params changes only re-run the pipeline; and
/// keeping it out of [`PreviewParams`] avoids rippling into its Copy /
/// history-snapshot / before-after-bypass machinery.
const DEMOSAIC_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS main.darkroom_demosaic \
     (imgid INTEGER PRIMARY KEY, method INTEGER NOT NULL)";

/// DDL for the per-image geometry (straighten + crop) table (same private-table
/// rationale as the others). A separate table from the params blob: geometry is
/// applied to the decoded buffer *before* the colour pipeline (it changes the
/// preview dimensions), not a `PreviewParams` pipeline field.
const GEOMETRY_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS main.darkroom_geometry \
     (imgid INTEGER PRIMARY KEY, geom BLOB NOT NULL)";

/// DDL for a small global (not per-image) UI-preference key/value store — same
/// private-table, best-effort rationale as the others. Used to persist lighttable
/// chrome state across sessions (m4-98d: the rating-filter comparator + floor).
/// `TEXT` keys/values keep it schema-light: each pref owns a stable key and codes
/// its own compact value string (e.g. the rating filter's `ge:3` / `rej` token).
const UI_PREFS_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS main.darkroom_ui_prefs \
     (key TEXT PRIMARY KEY, value TEXT NOT NULL)";

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

/// Load the saved edit-history stack for the image at `full_path`, or `None`
/// (no db, uncatalogued image, no row, table absent, or an undecodable/old-schema
/// blob). On `None` the darkroom view seeds a fresh single-entry stack.
pub fn load_history(db_path: &str, full_path: &str) -> Option<HistoryStack> {
    if db_path.is_empty() {
        return None;
    }
    let conn = Connection::open(db_path).ok()?;
    let imgid = imgid_for_path(&conn, full_path)?;
    load_history_conn(&conn, imgid)
}

/// Persist the edit-history stack for the image at `full_path`. Best-effort:
/// silently no-ops with no db or an uncatalogued image.
pub fn save_history(db_path: &str, full_path: &str, history: &HistoryStack) {
    if db_path.is_empty() {
        return;
    }
    let Ok(conn) = Connection::open(db_path) else {
        return;
    };
    if let Some(imgid) = imgid_for_path(&conn, full_path) {
        let _ = save_history_conn(&conn, imgid, history);
    }
}

/// Testable core of [`load_history`]: any error (no row, table not yet created)
/// or an undecodable blob yields `None`.
fn load_history_conn(conn: &Connection, imgid: i32) -> Option<HistoryStack> {
    let blob: rusqlite::Result<Vec<u8>> = conn.query_row(
        "SELECT stack FROM main.darkroom_history WHERE imgid = ?1",
        rusqlite::params![imgid],
        |row| row.get(0),
    );
    match blob {
        Ok(b) => HistoryStack::decode(&b),
        Err(_) => None,
    }
}

/// Testable core of [`save_history`]: upsert the single history row (PK on
/// `imgid` ⇒ one row; `ON CONFLICT` makes it an update).
fn save_history_conn(conn: &Connection, imgid: i32, history: &HistoryStack) -> rusqlite::Result<()> {
    conn.execute(HISTORY_TABLE_DDL, [])?;
    conn.execute(
        "INSERT INTO main.darkroom_history (imgid, stack) VALUES (?1, ?2) \
         ON CONFLICT(imgid) DO UPDATE SET stack = excluded.stack",
        rusqlite::params![imgid, history.encode()],
    )?;
    Ok(())
}

/// Load the saved Bayer [`DemosaicMethod`] for the image at `full_path`, falling
/// back to the default (RCD) on any miss (no db, uncatalogued image, no row, or
/// table absent). Unknown stored codes also decode to the default — see
/// [`DemosaicMethod::from_u8`].
pub fn load_demosaic(db_path: &str, full_path: &str) -> DemosaicMethod {
    if db_path.is_empty() {
        return DemosaicMethod::default();
    }
    let Ok(conn) = Connection::open(db_path) else {
        return DemosaicMethod::default();
    };
    match imgid_for_path(&conn, full_path) {
        Some(imgid) => load_demosaic_conn(&conn, imgid),
        None => DemosaicMethod::default(),
    }
}

/// Persist the Bayer [`DemosaicMethod`] for the image at `full_path`.
/// Best-effort: silently no-ops with no db or an uncatalogued image.
pub fn save_demosaic(db_path: &str, full_path: &str, method: DemosaicMethod) {
    if db_path.is_empty() {
        return;
    }
    let Ok(conn) = Connection::open(db_path) else {
        return;
    };
    if let Some(imgid) = imgid_for_path(&conn, full_path) {
        let _ = save_demosaic_conn(&conn, imgid, method);
    }
}

/// Testable core of [`load_demosaic`]: no row / no table / an unknown code all
/// yield the default method (never an error path).
fn load_demosaic_conn(conn: &Connection, imgid: i32) -> DemosaicMethod {
    let code: rusqlite::Result<i64> = conn.query_row(
        "SELECT method FROM main.darkroom_demosaic WHERE imgid = ?1",
        rusqlite::params![imgid],
        |row| row.get(0),
    );
    match code {
        Ok(v) => DemosaicMethod::from_u8(v as u8),
        Err(_) => DemosaicMethod::default(),
    }
}

/// Testable core of [`save_demosaic`]: upsert the single method row (PK on
/// `imgid` ⇒ one row; `ON CONFLICT` makes it an update).
fn save_demosaic_conn(conn: &Connection, imgid: i32, method: DemosaicMethod) -> rusqlite::Result<()> {
    conn.execute(DEMOSAIC_TABLE_DDL, [])?;
    conn.execute(
        "INSERT INTO main.darkroom_demosaic (imgid, method) VALUES (?1, ?2) \
         ON CONFLICT(imgid) DO UPDATE SET method = excluded.method",
        rusqlite::params![imgid, method.as_u8() as i64],
    )?;
    Ok(())
}

/// Load the saved [`Geometry`] for the image at `full_path`, or the default
/// (identity — no rotation, whole image) on any miss (no db, uncatalogued image,
/// no row, table absent, or an undecodable/old-version blob).
pub fn load_geometry(db_path: &str, full_path: &str) -> Geometry {
    if db_path.is_empty() {
        return Geometry::default();
    }
    let Ok(conn) = Connection::open(db_path) else {
        return Geometry::default();
    };
    match imgid_for_path(&conn, full_path) {
        Some(imgid) => load_geometry_conn(&conn, imgid).unwrap_or_default(),
        None => Geometry::default(),
    }
}

/// Load all persisted edit state for `full_path` over a **single** connection
/// with a **single** imgid resolution — for batch export, where calling
/// `load_saved` + `load_geometry` + `load_demosaic` separately would open three
/// connections and resolve the path three times per image. Returns the raw
/// pieces (saved colour params if any, geometry, demosaic method); the caller
/// decides what counts as "edited". Falls back to defaults on any db/lookup miss.
pub(crate) fn load_edit_state(
    db_path: &str,
    full_path: &str,
) -> (Option<PreviewParams>, Geometry, DemosaicMethod) {
    let fallback = || (None, Geometry::default(), DemosaicMethod::default());
    if db_path.is_empty() {
        return fallback();
    }
    let Ok(conn) = Connection::open(db_path) else {
        return fallback();
    };
    let Some(imgid) = imgid_for_path(&conn, full_path) else {
        return fallback();
    };
    (
        load_saved_conn(&conn, imgid),
        load_geometry_conn(&conn, imgid).unwrap_or_default(),
        load_demosaic_conn(&conn, imgid),
    )
}

/// Persist the [`Geometry`] for the image at `full_path`. Best-effort: silently
/// no-ops with no db or an uncatalogued image.
pub fn save_geometry(db_path: &str, full_path: &str, geom: &Geometry) {
    if db_path.is_empty() {
        return;
    }
    let Ok(conn) = Connection::open(db_path) else {
        return;
    };
    if let Some(imgid) = imgid_for_path(&conn, full_path) {
        let _ = save_geometry_conn(&conn, imgid, geom);
    }
}

/// Testable core of [`load_geometry`]: no row / no table / an undecodable blob
/// all yield `None` (→ the default at the call site).
fn load_geometry_conn(conn: &Connection, imgid: i32) -> Option<Geometry> {
    let blob: rusqlite::Result<Vec<u8>> = conn.query_row(
        "SELECT geom FROM main.darkroom_geometry WHERE imgid = ?1",
        rusqlite::params![imgid],
        |row| row.get(0),
    );
    match blob {
        Ok(b) => Geometry::decode(&b),
        Err(_) => None,
    }
}

/// Testable core of [`save_geometry`]: upsert the single geometry row (PK on
/// `imgid` ⇒ one row; `ON CONFLICT` makes it an update).
fn save_geometry_conn(conn: &Connection, imgid: i32, geom: &Geometry) -> rusqlite::Result<()> {
    conn.execute(GEOMETRY_TABLE_DDL, [])?;
    conn.execute(
        "INSERT INTO main.darkroom_geometry (imgid, geom) VALUES (?1, ?2) \
         ON CONFLICT(imgid) DO UPDATE SET geom = excluded.geom",
        rusqlite::params![imgid, geom.encode()],
    )?;
    Ok(())
}

/// Read a global UI preference by `key`, or `None` if there's no db, no table
/// (old/rebuilt catalog), or no row. Best-effort: any error ⇒ `None` ⇒ the call
/// site's default.
pub fn load_ui_pref(db_path: &str, key: &str) -> Option<String> {
    if db_path.is_empty() {
        return None;
    }
    let conn = Connection::open(db_path).ok()?;
    load_ui_pref_conn(&conn, key)
}

/// Write a global UI preference (`key` ⇒ `value`), creating the table on demand.
/// Best-effort: a failed open/write is swallowed (a lost UI pref is cosmetic).
pub fn save_ui_pref(db_path: &str, key: &str, value: &str) {
    if db_path.is_empty() {
        return;
    }
    if let Ok(conn) = Connection::open(db_path) {
        let _ = save_ui_pref_conn(&conn, key, value);
    }
}

/// Testable core of [`load_ui_pref`]: `None` on a missing table or absent key.
fn load_ui_pref_conn(conn: &Connection, key: &str) -> Option<String> {
    conn.query_row(
        "SELECT value FROM main.darkroom_ui_prefs WHERE key = ?1",
        rusqlite::params![key],
        |row| row.get::<_, String>(0),
    )
    .ok()
}

/// Testable core of [`save_ui_pref`]: upsert one key/value row (PK on `key`).
fn save_ui_pref_conn(conn: &Connection, key: &str, value: &str) -> rusqlite::Result<()> {
    conn.execute(UI_PREFS_TABLE_DDL, [])?;
    conn.execute(
        "INSERT INTO main.darkroom_ui_prefs (key, value) VALUES (?1, ?2) \
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![key, value],
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
    fn ui_pref_missing_table_reads_none_then_save_creates_and_upserts() {
        let db = open_db();
        // A fresh catalog has no prefs table → best-effort read is None, not an error.
        assert_eq!(load_ui_pref_conn(&db, "rating_filter"), None);
        // First save creates the table on demand; read-back round-trips.
        save_ui_pref_conn(&db, "rating_filter", "ge:3").unwrap();
        assert_eq!(load_ui_pref_conn(&db, "rating_filter").as_deref(), Some("ge:3"));
        // Second save on the same key updates in place (PK on key ⇒ single row).
        save_ui_pref_conn(&db, "rating_filter", "rej").unwrap();
        assert_eq!(load_ui_pref_conn(&db, "rating_filter").as_deref(), Some("rej"));
        let n: i64 = db
            .query_row("SELECT COUNT(*) FROM main.darkroom_ui_prefs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1, "upsert must not accumulate rows");
        // An unknown key still reads None even once the table exists.
        assert_eq!(load_ui_pref_conn(&db, "no_such_key"), None);
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

    #[test]
    fn demosaic_save_then_load_roundtrips() {
        let db = open_db();
        save_demosaic_conn(&db, 42, DemosaicMethod::Vng).unwrap();
        assert_eq!(load_demosaic_conn(&db, 42), DemosaicMethod::Vng);
        // upsert: PK ⇒ one row holding the latest method
        save_demosaic_conn(&db, 42, DemosaicMethod::Ppg).unwrap();
        let n: i32 = db
            .query_row(
                "SELECT COUNT(*) FROM main.darkroom_demosaic WHERE imgid = 42",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(load_demosaic_conn(&db, 42), DemosaicMethod::Ppg);
    }

    #[test]
    fn demosaic_load_defaults_when_table_absent_or_no_row() {
        let db = open_db();
        // table doesn't exist yet → default (no panic), not an error path
        assert_eq!(load_demosaic_conn(&db, 42), DemosaicMethod::default());
        // create the table via a save for a different image, then a missing row
        save_demosaic_conn(&db, 99, DemosaicMethod::Vng).unwrap();
        assert_eq!(load_demosaic_conn(&db, 42), DemosaicMethod::default());
    }

    #[test]
    fn demosaic_unknown_stored_code_decodes_to_default() {
        let db = open_db();
        db.execute(DEMOSAIC_TABLE_DDL, []).unwrap();
        db.execute(
            "INSERT INTO main.darkroom_demosaic (imgid, method) VALUES (42, 77)",
            [],
        )
        .unwrap();
        assert_eq!(load_demosaic_conn(&db, 42), DemosaicMethod::default());
    }

    fn sample_geometry() -> Geometry {
        Geometry {
            crop: c41_core::geometry::Crop { left: 0.1, top: 0.15, right: 0.9, bottom: 0.85 },
            angle: 0.03,
        }
    }

    #[test]
    fn geometry_save_then_load_roundtrips() {
        let db = open_db();
        let g = sample_geometry();
        save_geometry_conn(&db, 42, &g).unwrap();
        assert_eq!(load_geometry_conn(&db, 42), Some(g));
        // upsert: PK ⇒ one row holding the latest geometry
        let g2 = Geometry::default();
        save_geometry_conn(&db, 42, &g2).unwrap();
        let n: i32 = db
            .query_row(
                "SELECT COUNT(*) FROM main.darkroom_geometry WHERE imgid = 42",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(load_geometry_conn(&db, 42), Some(g2));
    }

    #[test]
    fn geometry_load_is_none_when_table_absent_or_no_row() {
        let db = open_db();
        assert_eq!(load_geometry_conn(&db, 42), None); // no table yet, no panic
        save_geometry_conn(&db, 99, &sample_geometry()).unwrap();
        assert_eq!(load_geometry_conn(&db, 42), None); // table exists, no row
    }

    #[test]
    fn geometry_undecodable_blob_is_none() {
        let db = open_db();
        db.execute(GEOMETRY_TABLE_DDL, []).unwrap();
        db.execute(
            "INSERT INTO main.darkroom_geometry (imgid, geom) VALUES (42, ?1)",
            rusqlite::params![vec![9u8, 9, 9]], // wrong length/version
        )
        .unwrap();
        assert_eq!(load_geometry_conn(&db, 42), None);
    }

    fn sample_history() -> HistoryStack {
        let mut h = HistoryStack::new("Original", PreviewParams::default());
        h.record("Exposure", sample_params());
        let mut p2 = sample_params();
        p2.ev = 0.9;
        h.record("Exposure", p2);
        h.undo(); // leave the cursor mid-stack
        h
    }

    #[test]
    fn history_save_then_load_roundtrips() {
        let db = open_db();
        let h = sample_history();
        save_history_conn(&db, 42, &h).unwrap();
        let got = load_history_conn(&db, 42).expect("load");
        assert_eq!(got.entries(), h.entries());
        assert_eq!(got.cursor(), h.cursor());
    }

    #[test]
    fn history_load_is_none_when_table_absent_or_no_row() {
        let db = open_db();
        assert!(load_history_conn(&db, 42).is_none()); // no table yet
        save_history_conn(&db, 99, &sample_history()).unwrap(); // creates table
        assert!(load_history_conn(&db, 42).is_none()); // still no row for 42
    }

    #[test]
    fn history_save_upserts_single_row() {
        let db = open_db();
        save_history_conn(&db, 42, &sample_history()).unwrap();
        let h2 = HistoryStack::new("Original", PreviewParams::default());
        save_history_conn(&db, 42, &h2).unwrap();
        let n: i32 = db
            .query_row(
                "SELECT COUNT(*) FROM main.darkroom_history WHERE imgid = 42",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(load_history_conn(&db, 42).map(|h| h.len()), Some(1));
    }
}

// ── Styles (parity 2.4) ──────────────────────────────────────────────────────

/// DDL for the styles table.
///
/// darktable stores a style as rows in `style_items`, one per IOP, each holding
/// that module's `op_params` blob. Our edits are not per-module blobs — they are
/// a single versioned [`PreviewParams`], which the whole UI, history stack and
/// export path already round-trip. So a style here is that same blob under a
/// name, which means saving and applying a style reuse the encode/decode we
/// already trust rather than a parallel serialisation.
///
/// The cost of that choice, stated plainly: a style is **all-or-nothing**. It
/// carries every module's settings, so applying one replaces the target's whole
/// edit rather than merging selected modules, which is what darktable's
/// per-item styles allow. Partial styles want a module mask alongside the blob;
/// the schema leaves room for it (`modules` is reserved, NULL today) so adding
/// them later does not need a migration.
const STYLES_TABLE_DDL: &str = "CREATE TABLE IF NOT EXISTS main.c41_styles \
     (name TEXT PRIMARY KEY, description TEXT NOT NULL DEFAULT '', \
      params BLOB NOT NULL, modules TEXT)";

/// One saved style.
#[derive(Clone, Debug, PartialEq)]
pub struct Style {
    pub name: String,
    pub description: String,
    pub params: PreviewParams,
}

/// Save (or overwrite) a style. Returns false if the name is blank or the write
/// fails — the caller surfaces that rather than silently doing nothing.
pub fn save_style(db_path: &str, name: &str, description: &str, params: &PreviewParams) -> bool {
    let name = name.trim();
    if db_path.is_empty() || name.is_empty() {
        return false;
    }
    let Ok(conn) = Connection::open(db_path) else { return false };
    if conn.execute(STYLES_TABLE_DDL, []).is_err() {
        return false;
    }
    conn.execute(
        "INSERT INTO main.c41_styles (name, description, params) VALUES (?1, ?2, ?3) \
         ON CONFLICT(name) DO UPDATE SET description = excluded.description, \
                                         params = excluded.params",
        rusqlite::params![name, description, params.encode()],
    )
    .is_ok()
}

/// All styles, name-ordered. Empty on any failure — a missing table just means
/// none have been saved yet.
pub fn load_styles(db_path: &str) -> Vec<Style> {
    if db_path.is_empty() {
        return Vec::new();
    }
    let Ok(conn) = Connection::open(db_path) else { return Vec::new() };
    let Ok(mut stmt) = conn.prepare(
        "SELECT name, description, params FROM main.c41_styles ORDER BY name COLLATE NOCASE",
    ) else {
        return Vec::new();
    };
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Vec<u8>>(2)?,
        ))
    });
    let Ok(rows) = rows else { return Vec::new() };
    rows.flatten()
        .filter_map(|(name, description, blob)| {
            // A style written by an older ENCODE_VERSION decodes to None. Skip
            // it rather than substituting defaults, which would silently apply
            // a *different* edit than the one the user saved.
            PreviewParams::decode(&blob).map(|params| Style { name, description, params })
        })
        .collect()
}

/// Delete a style by name. Returns whether a row was removed.
pub fn delete_style(db_path: &str, name: &str) -> bool {
    if db_path.is_empty() {
        return false;
    }
    let Ok(conn) = Connection::open(db_path) else { return false };
    conn.execute(
        "DELETE FROM main.c41_styles WHERE name = ?1",
        rusqlite::params![name],
    )
    .map(|n| n > 0)
    .unwrap_or(false)
}

/// Apply a style's params to every image in `full_paths`, returning how many
/// were **actually written**.
///
/// The count is writes, not attempts: params are keyed by `imgid`, so a path
/// that is not in the catalogue (`images` ⋈ `film_rolls`) has nowhere to store
/// an edit and is skipped. Reporting attempts would let the UI claim "applied
/// to 5 images" when none were catalogued.
///
/// Note this **replaces** the target's params wholesale — see the note on
/// [`STYLES_TABLE_DDL`].
pub fn apply_style_to(db_path: &str, full_paths: &[String], style: &Style) -> usize {
    if db_path.is_empty() {
        return 0;
    }
    let Ok(conn) = Connection::open(db_path) else { return 0 };
    let mut written = 0usize;
    for path in full_paths.iter().filter(|p| !p.is_empty()) {
        if let Some(imgid) = imgid_for_path(&conn, path) {
            if save_params_conn(&conn, imgid, &style.params).is_ok() {
                written += 1;
            }
        }
    }
    written
}

#[cfg(test)]
mod style_tests {
    use super::*;

    /// A real temp file, not `:memory:` — the style API takes a path and opens
    /// its own connection per call, which is exactly the behaviour worth
    /// testing (an in-memory DB would be a fresh empty database each time).
    ///
    /// Deletes any previous file at the path so a crashed run cannot leak state
    /// into the next; the guard removes it on drop.
    struct TmpDb(String);
    impl TmpDb {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            // Thread id keeps parallel test threads from sharing a file.
            p.push(format!("c41-styles-{tag}-{:?}.db", std::thread::current().id()));
            let path = p.to_string_lossy().into_owned();
            let _ = std::fs::remove_file(&path);
            TmpDb(path)
        }
    }
    impl Drop for TmpDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn tmp_db(tag: &str) -> (TmpDb, String) {
        let g = TmpDb::new(tag);
        let path = g.0.clone();
        (g, path)
    }

    fn params_with(ev: f32) -> PreviewParams {
        PreviewParams { ev, ..PreviewParams::default() }
    }

    #[test]
    fn save_load_round_trips_a_style() {
        let (_d, db) = tmp_db("roundtrip");
        assert!(save_style(&db, "Punchy", "high contrast", &params_with(1.5)));
        let styles = load_styles(&db);
        assert_eq!(styles.len(), 1);
        assert_eq!(styles[0].name, "Punchy");
        assert_eq!(styles[0].description, "high contrast");
        assert_eq!(styles[0].params.ev, 1.5, "params must survive verbatim");
    }

    #[test]
    fn save_overwrites_by_name_rather_than_duplicating() {
        let (_d, db) = tmp_db("upsert");
        assert!(save_style(&db, "S", "first", &params_with(1.0)));
        assert!(save_style(&db, "S", "second", &params_with(2.0)));
        let styles = load_styles(&db);
        assert_eq!(styles.len(), 1, "same name must upsert, not duplicate");
        assert_eq!(styles[0].description, "second");
        assert_eq!(styles[0].params.ev, 2.0);
    }

    #[test]
    fn blank_names_are_rejected() {
        let (_d, db) = tmp_db("blank");
        assert!(!save_style(&db, "", "x", &params_with(1.0)));
        assert!(!save_style(&db, "   ", "x", &params_with(1.0)));
        assert!(load_styles(&db).is_empty());
        // Names are trimmed, so " S " and "S" are the same style rather than two
        // rows that look identical in the list.
        assert!(save_style(&db, " S ", "", &params_with(1.0)));
        assert_eq!(load_styles(&db)[0].name, "S");
    }

    #[test]
    fn a_style_from_an_incompatible_version_is_skipped_not_defaulted() {
        // Substituting defaults for an undecodable blob would silently apply a
        // DIFFERENT edit than the one saved — worse than the style vanishing.
        let (_d, db) = tmp_db("badver");
        let conn = Connection::open(&db).unwrap();
        conn.execute(STYLES_TABLE_DDL, []).unwrap();
        conn.execute(
            "INSERT INTO main.c41_styles (name, description, params) VALUES ('old', '', ?1)",
            rusqlite::params![vec![0u8, 1, 2, 3]],
        )
        .unwrap();
        assert!(load_styles(&db).is_empty(), "undecodable style must be skipped");
    }

    #[test]
    fn delete_removes_only_the_named_style() {
        let (_d, db) = tmp_db("delete");
        save_style(&db, "a", "", &params_with(1.0));
        save_style(&db, "b", "", &params_with(2.0));
        assert!(delete_style(&db, "a"));
        assert!(!delete_style(&db, "a"), "second delete removes nothing");
        let names: Vec<_> = load_styles(&db).into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["b"]);
    }

    #[test]
    fn styles_list_is_name_ordered_case_insensitively() {
        let (_d, db) = tmp_db("order");
        for n in ["zebra", "Apple", "mango"] {
            save_style(&db, n, "", &params_with(0.5));
        }
        let names: Vec<_> = load_styles(&db).into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["Apple", "mango", "zebra"]);
    }

    /// Params are keyed by imgid, so a target must exist in the catalogue for
    /// an edit to have anywhere to live. Build the two tables the lookup joins.
    fn catalogue(db: &str, folder: &str, files: &[&str]) {
        let conn = Connection::open(db).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS main.film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE IF NOT EXISTS main.images \
                 (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO main.film_rolls (id, folder) VALUES (1, ?1)",
            rusqlite::params![folder],
        )
        .unwrap();
        for (i, f) in files.iter().enumerate() {
            conn.execute(
                "INSERT INTO main.images (id, film_id, filename) VALUES (?1, 1, ?2)",
                rusqlite::params![i as i32 + 1, f],
            )
            .unwrap();
        }
    }

    #[test]
    fn apply_writes_the_style_to_every_target() {
        let (_d, db) = tmp_db("apply");
        catalogue(&db, "/photos", &["a.dng", "b.dng"]);
        save_style(&db, "S", "", &params_with(2.5));
        let style = load_styles(&db).remove(0);
        let targets = vec!["/photos/a.dng".to_string(), "/photos/b.dng".to_string()];
        assert_eq!(apply_style_to(&db, &targets, &style), 2);
        for t in &targets {
            assert_eq!(load_params(&db, t).ev, 2.5, "{t} did not receive the style");
        }
    }

    #[test]
    fn apply_counts_writes_not_attempts() {
        // An uncatalogued path has no imgid, so its edit has nowhere to go. The
        // count must reflect that rather than claiming a write that never
        // happened — the UI reports this number back to the user.
        let (_d, db) = tmp_db("applycount");
        catalogue(&db, "/photos", &["a.dng"]);
        save_style(&db, "S", "", &params_with(1.5));
        let style = load_styles(&db).remove(0);
        let targets = vec![
            "/photos/a.dng".to_string(),
            "/photos/not-imported.dng".to_string(),
        ];
        assert_eq!(apply_style_to(&db, &targets, &style), 1, "only the catalogued image counts");
        assert_eq!(load_params(&db, "/photos/a.dng").ev, 1.5);
    }

    #[test]
    fn apply_skips_empty_paths() {
        let (_d, db) = tmp_db("applyempty");
        save_style(&db, "S", "", &params_with(1.0));
        let style = load_styles(&db).remove(0);
        assert_eq!(apply_style_to(&db, &["".to_string()], &style), 0);
    }
}
