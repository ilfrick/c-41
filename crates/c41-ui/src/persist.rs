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

/// Open a catalogue connection with the crate-standard 3s `busy_timeout`
/// (m4-147). The contention-exposed siblings already did this — the rating and
/// colour-label connections (`lighttable::open_rating_conn`/
/// `open_colorlabels_conn`), the timeline histogram reader, and `c41-db`'s
/// `attach_catalog`; `panels::query_exif` is the deliberate 250ms exception,
/// since it runs per arrow-keypress. The writers that can hold library.db's
/// lock while a UI thread wants in are OFF-THREAD: the rating/colour-label
/// `serialized_write` workers (`gio::spawn_blocking`) and the import worker's
/// bulk transaction. This module's writes — metadata editor included — run on
/// the main thread and previously opened BARE, i.e. with no busy handler:
/// SQLite reports BUSY on such a connection's first statement under a held
/// lock (the open itself takes none), so an instant BUSY here silently skipped
/// a write or read defaults — worst case, a failed `load_saved` handed back
/// default params that a later autosave wrote over the stored edit.
///
/// NOT to be confused with `c41_db::schema::open_catalog`, which ATTACHes
/// data.db/memory and ensures the full schema for app bootstrap; this is the
/// plain per-call open every production path in this module uses. `pub(crate)`
/// since m4-148 so formerly-bare opens outside this module
/// (`panels::load_film_rolls`, the lighttable name-filter loaders) share it;
/// four direct opens deliberately remain elsewhere (timeline + rating/colour +
/// query_exif), each installing its own timeout inline.
pub(crate) fn open_catalog(db_path: &str) -> rusqlite::Result<Connection> {
    let conn = Connection::open(db_path)?;
    let _ = conn.busy_timeout(std::time::Duration::from_secs(3));
    Ok(conn)
}

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
    let conn = open_catalog(db_path).ok()?;
    let imgid = imgid_for_path(&conn, full_path)?;
    load_saved_conn(&conn, imgid)
}

/// Persist preview params for the image at `full_path`. Best-effort: silently
/// no-ops if there is no db or the image isn't catalogued.
pub fn save_params(db_path: &str, full_path: &str, params: &PreviewParams) {
    if db_path.is_empty() {
        return;
    }
    let Ok(conn) = open_catalog(db_path) else {
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
    let conn = open_catalog(db_path).ok()?;
    let imgid = imgid_for_path(&conn, full_path)?;
    load_history_conn(&conn, imgid)
}

/// Persist the edit-history stack for the image at `full_path`. Best-effort:
/// silently no-ops with no db or an uncatalogued image.
pub fn save_history(db_path: &str, full_path: &str, history: &HistoryStack) {
    if db_path.is_empty() {
        return;
    }
    let Ok(conn) = open_catalog(db_path) else {
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

/// Remove every saved edit for the image at `full_path`: both the preview
/// params row and the edit-history stack row. This is the lighttable
/// "discard history" operation (parity row 2.2): afterwards the image renders
/// and reopens with default params and a fresh one-entry history stack.
///
/// Best-effort like every writer here: silently no-ops with no db or an
/// uncatalogued image.
pub fn discard_history(db_path: &str, full_path: &str) {
    if db_path.is_empty() {
        return;
    }
    let Ok(conn) = open_catalog(db_path) else { return };
    if let Some(imgid) = imgid_for_path(&conn, full_path) {
        let _ = discard_history_conn(&conn, imgid);
    }
}

/// Testable core of [`discard_history`]: clear both rows in one transaction so
/// a failure cannot leave params pointing at a discarded stack (or vice versa).
fn discard_history_conn(conn: &Connection, imgid: i32) -> rusqlite::Result<()> {
    let tx = conn.unchecked_transaction()?;
    tx.execute(PREVIEW_TABLE_DDL, [])?;
    tx.execute(HISTORY_TABLE_DDL, [])?;
    tx.execute("DELETE FROM main.darkroom_preview WHERE imgid = ?1", [imgid])?;
    tx.execute("DELETE FROM main.darkroom_history WHERE imgid = ?1", [imgid])?;
    tx.commit()
}

/// Load the saved Bayer [`DemosaicMethod`] for the image at `full_path`, falling
/// back to the default (RCD) on any miss (no db, uncatalogued image, no row, or
/// table absent). Unknown stored codes also decode to the default — see
/// [`DemosaicMethod::from_u8`].
pub fn load_demosaic(db_path: &str, full_path: &str) -> DemosaicMethod {
    if db_path.is_empty() {
        return DemosaicMethod::default();
    }
    let Ok(conn) = open_catalog(db_path) else {
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
    let Ok(conn) = open_catalog(db_path) else {
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
    let Ok(conn) = open_catalog(db_path) else {
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
    let Ok(conn) = open_catalog(db_path) else {
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
    let Ok(conn) = open_catalog(db_path) else {
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

/// DDL for the per-image lens-correction gear choice (same private-table
/// rationale as the others). Separate from the params blob for the same reason
/// as demosaic — it's *identity* state feeding a resolve step, not a pipeline
/// scalar — and because the blob format carries no strings. The lens is stored
/// as its structured `(maker, model)` identity pair exactly as the lensfun
/// database spells it (a display label can't be split back reliably, and the
/// database's fuzzy search can't re-find even half of its own entries from
/// their exact names). All four string columns empty = "no gear chosen"
/// (module can't resolve → no stage).
const LENS_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS main.darkroom_lens_choice \
     (imgid INTEGER PRIMARY KEY, camera_maker TEXT NOT NULL DEFAULT '', \
      camera_model TEXT NOT NULL DEFAULT '', lens_maker TEXT NOT NULL DEFAULT '', \
      lens_model TEXT NOT NULL DEFAULT '')";

/// The per-image lens-correction gear selection.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LensChoice {
    pub camera_maker: String,
    pub camera_model: String,
    /// Lens maker exactly as `c41_core::iop::lens::list_lenses` spells it.
    pub lens_maker: String,
    /// Lens model exactly as `c41_core::iop::lens::list_lenses` spells it.
    pub lens_model: String,
}

impl LensChoice {
    /// True when nothing is selected yet — no correction is possible.
    pub fn is_empty(&self) -> bool {
        self.camera_model.is_empty() && self.lens_model.is_empty()
    }
}

/// Load the saved lens-correction gear choice for the image at `full_path`,
/// or the default (nothing selected) on any miss.
pub fn load_lens(db_path: &str, full_path: &str) -> LensChoice {
    if db_path.is_empty() {
        return LensChoice::default();
    }
    let Ok(conn) = open_catalog(db_path) else {
        return LensChoice::default();
    };
    match imgid_for_path(&conn, full_path) {
        Some(imgid) => load_lens_conn(&conn, imgid).unwrap_or_default(),
        None => LensChoice::default(),
    }
}

/// Persist the lens-correction gear choice for the image at `full_path`.
/// Best-effort like the rest of this module.
pub fn save_lens(db_path: &str, full_path: &str, choice: &LensChoice) {
    if db_path.is_empty() {
        return;
    }
    let Ok(conn) = open_catalog(db_path) else {
        return;
    };
    if let Some(imgid) = imgid_for_path(&conn, full_path) {
        let _ = save_lens_conn(&conn, imgid, choice);
    }
}

/// Testable core of [`load_lens`]: no row / no table ⇒ `None` (→ default).
fn load_lens_conn(conn: &Connection, imgid: i32) -> Option<LensChoice> {
    conn.query_row(
        "SELECT camera_maker, camera_model, lens_maker, lens_model \
         FROM main.darkroom_lens_choice WHERE imgid = ?1",
        rusqlite::params![imgid],
        |row| {
            Ok(LensChoice {
                camera_maker: row.get(0)?,
                camera_model: row.get(1)?,
                lens_maker: row.get(2)?,
                lens_model: row.get(3)?,
            })
        },
    )
    .ok()
}

/// Testable core of [`save_lens`]: upsert the single choice row.
fn save_lens_conn(conn: &Connection, imgid: i32, choice: &LensChoice) -> rusqlite::Result<()> {
    conn.execute(LENS_TABLE_DDL, [])?;
    conn.execute(
        "INSERT INTO main.darkroom_lens_choice \
           (imgid, camera_maker, camera_model, lens_maker, lens_model) \
         VALUES (?1, ?2, ?3, ?4, ?5) \
         ON CONFLICT(imgid) DO UPDATE SET camera_maker = excluded.camera_maker, \
         camera_model = excluded.camera_model, lens_maker = excluded.lens_maker, \
         lens_model = excluded.lens_model",
        rusqlite::params![
            imgid,
            choice.camera_maker,
            choice.camera_model,
            choice.lens_maker,
            choice.lens_model
        ],
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
    let conn = open_catalog(db_path).ok()?;
    load_ui_pref_conn(&conn, key)
}

/// Write a global UI preference (`key` ⇒ `value`), creating the table on demand.
/// Best-effort: a failed open/write is swallowed (a lost UI pref is cosmetic).
pub fn save_ui_pref(db_path: &str, key: &str, value: &str) {
    if db_path.is_empty() {
        return;
    }
    if let Ok(conn) = open_catalog(db_path) {
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
/// The blob carries EVERY module's settings; `modules` narrows what applying it
/// touches (m4-149). NULL — the shape every pre-149 row has, and what a
/// whole-edit save writes — means all-or-nothing: applying replaces the target's
/// whole edit. A non-NULL value is a comma-separated list of module-group names
/// (`crate::stylemodules::MODULE_GROUPS`; those names contain no commas), and
/// applying merges only those groups over the target's saved edit, which is
/// what darktable's per-item styles do. An empty string is a valid zero-module
/// style: applying it changes nothing.
const STYLES_TABLE_DDL: &str = "CREATE TABLE IF NOT EXISTS main.c41_styles \
     (name TEXT PRIMARY KEY, description TEXT NOT NULL DEFAULT '', \
      params BLOB NOT NULL, modules TEXT)";

/// One saved style.
///
/// `modules` mirrors the column of the same name: `None` = a whole-edit style
/// (apply replaces everything), `Some(list)` = carry only those module groups.
#[derive(Clone, Debug, PartialEq)]
pub struct Style {
    pub name: String,
    pub description: String,
    pub params: PreviewParams,
    pub modules: Option<Vec<String>>,
}

/// Save (or overwrite) a style. Returns false if the name is blank or the write
/// fails — the caller surfaces that rather than silently doing nothing.
///
/// `modules`: `None` stores a NULL column (whole-edit style); `Some(groups)`
/// stores the group names comma-joined. Names are the caller's choice but only
/// [`crate::stylemodules::MODULE_GROUPS`] entries have meaning at apply time.
pub fn save_style(
    db_path: &str,
    name: &str,
    description: &str,
    params: &PreviewParams,
    modules: Option<&[&str]>,
) -> bool {
    let name = name.trim();
    if db_path.is_empty() || name.is_empty() {
        return false;
    }
    let Ok(conn) = open_catalog(db_path) else { return false };
    if conn.execute(STYLES_TABLE_DDL, []).is_err() {
        return false;
    }
    conn.execute(
        "INSERT INTO main.c41_styles (name, description, params, modules) VALUES (?1, ?2, ?3, ?4) \
         ON CONFLICT(name) DO UPDATE SET description = excluded.description, \
                                         params = excluded.params, \
                                         modules = excluded.modules",
        rusqlite::params![
            name,
            description,
            params.encode(),
            modules.map(|ms| ms.join(",")),
        ],
    )
    .is_ok()
}

/// All styles, name-ordered. Empty on any failure — a missing table just means
/// none have been saved yet.
pub fn load_styles(db_path: &str) -> Vec<Style> {
    if db_path.is_empty() {
        return Vec::new();
    }
    let Ok(conn) = open_catalog(db_path) else { return Vec::new() };
    let Ok(mut stmt) = conn.prepare(
        "SELECT name, description, params, modules FROM main.c41_styles ORDER BY name COLLATE NOCASE",
    ) else {
        return Vec::new();
    };
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Vec<u8>>(2)?,
            row.get::<_, Option<String>>(3)?,
        ))
    });
    let Ok(rows) = rows else { return Vec::new() };
    rows.flatten()
        .filter_map(|(name, description, blob, modules)| {
            // A style written by an older ENCODE_VERSION decodes to None. Skip
            // it rather than substituting defaults, which would silently apply
            // a *different* edit than the one the user saved.
            PreviewParams::decode(&blob).map(|params| Style {
                name,
                description,
                params,
                modules: modules.map(|m| {
                    m.split(',')
                        .filter(|t| !t.is_empty())
                        .map(String::from)
                        .collect()
                }),
            })
        })
        .collect()
}

/// Delete a style by name. Returns whether a row was removed.
pub fn delete_style(db_path: &str, name: &str) -> bool {
    if db_path.is_empty() {
        return false;
    }
    let Ok(conn) = open_catalog(db_path) else { return false };
    conn.execute(
        "DELETE FROM main.c41_styles WHERE name = ?1",
        rusqlite::params![name],
    )
    .map(|n| n > 0)
    .unwrap_or(false)
}

// ── Named collection presets (parity 2.6 close, m4-136) ─────────────────────
//
// darktable's collection module can store the current filter set under a name
// and recall it later. Here the "filter set" is exactly the five persisted
// quick-filter/rule tokens (rating, colour, aspect, year range, rule stack), so
// a preset stores their combined payload string — see
// `lighttable::collection_filter_payload`, which is the only writer of that
// format and its only interpreter. This module treats payloads as opaque text
// on purpose: decoding lives next to the codecs it composes. Rejection of a
// corrupt payload therefore happens at APPLY time in
// `lighttable::parse_collection_payload` — all-or-nothing on structure, with
// per-field leniency below that (each component token's own decoder falls back
// to no-filter); this module returns every stored row unfiltered.
//
// Same lazily-created table as styles: the DDL runs inside the save path and a
// missing table reads back as "no presets", so demo mode (empty db path) needs
// no special casing.

const COLLECTION_PRESETS_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS main.c41_collection_presets \
     (name TEXT PRIMARY KEY, payload TEXT NOT NULL)";

/// Save (or overwrite) a collection preset. `payload` must come from
/// `lighttable::collection_filter_payload()`; blank names/payloads are refused
/// so an empty row can never shadow a real one.
pub fn save_collection_preset(db_path: &str, name: &str, payload: &str) -> bool {
    let name = name.trim();
    if db_path.is_empty() || name.is_empty() || payload.is_empty() {
        return false;
    }
    let Ok(conn) = open_catalog(db_path) else { return false };
    if conn.execute(COLLECTION_PRESETS_TABLE_DDL, []).is_err() {
        return false;
    }
    conn.execute(
        "INSERT INTO main.c41_collection_presets (name, payload) VALUES (?1, ?2) \
         ON CONFLICT(name) DO UPDATE SET payload = excluded.payload",
        rusqlite::params![name, payload],
    )
    .is_ok()
}

/// All saved collection presets, name-ordered. Empty on any failure — a missing
/// table just means none have been saved yet.
pub fn load_collection_presets(db_path: &str) -> Vec<(String, String)> {
    if db_path.is_empty() {
        return Vec::new();
    }
    let Ok(conn) = open_catalog(db_path) else { return Vec::new() };
    let Ok(mut stmt) = conn.prepare(
        "SELECT name, payload FROM main.c41_collection_presets ORDER BY name COLLATE NOCASE",
    ) else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return Vec::new();
    };
    rows.flatten().collect()
}

/// Delete a collection preset by name. Returns whether a row was removed.
pub fn delete_collection_preset(db_path: &str, name: &str) -> bool {
    if db_path.is_empty() {
        return false;
    }
    let Ok(conn) = open_catalog(db_path) else { return false };
    conn.execute(
        "DELETE FROM main.c41_collection_presets WHERE name = ?1",
        rusqlite::params![name],
    )
    .map(|n| n > 0)
    .unwrap_or(false)
}

/// Apply a style to every catalogued image in `full_paths`, returning how many
/// targets were actually written.
///
/// The count is writes, not attempts: params are keyed by `imgid`, so a path
/// that is not in the catalogue (`images` ⋈ `film_rolls`) has nowhere to store
/// an edit and is skipped. Reporting attempts would let the UI claim "applied
/// to 5 images" when none were catalogued.
///
/// What lands on each target depends on [`Style::modules`]. A whole-edit style
/// (`None`) replaces the target's saved edit with the style's params, exactly
/// as styles always did. A partial style (m4-149) instead MERGES: the target's
/// saved edit is kept and only the listed module groups are copied over from
/// the style — darktable's per-item style behaviour. A target with no saved
/// edit yet merges onto defaults; a group name nothing knows about is simply
/// skipped.
///
/// Caveat shared by both arms: an image whose darkroom page is open elsewhere
/// holds its params in memory and will write them back on autosave, clobbering
/// what this wrote (pre-existing for whole styles; a partial merge sharpens the
/// surprise because the user believes their other modules survived). There is
/// no cross-page invalidation yet.
pub fn apply_style_to(db_path: &str, full_paths: &[String], style: &Style) -> usize {
    if db_path.is_empty() {
        return 0;
    }
    // open_catalog's busy_timeout matters double here (m4-146 review MINOR-1):
    // this loop issues up to N writes, and each one failing instantly under a
    // held file lock would silently shrink the reported count.
    let Ok(conn) = open_catalog(db_path) else { return 0 };
    let mut written = 0usize;
    for path in full_paths.iter().filter(|p| !p.is_empty()) {
        if let Some(imgid) = imgid_for_path(&conn, path) {
            let outcome = match &style.modules {
                None => save_params_conn(&conn, imgid, &style.params),
                Some(groups) => {
                    let base = load_saved_conn(&conn, imgid).unwrap_or_default();
                    let names: Vec<&str> = groups.iter().map(String::as_str).collect();
                    let merged = crate::stylemodules::merge_modules(&base, &style.params, &names);
                    save_params_conn(&conn, imgid, &merged)
                }
            };
            if outcome.is_ok() {
                written += 1;
            }
        }
    }
    written
}

// ── Image metadata editor (parity 2.3) ──────────────────────────────────────

// Storage lives in `main.meta_data`, which is DARKTABLE'S table, not one of the
// c41-ui-private ones above. Metadata is not our format: it is Dublin Core / XMP
// text the C app reads, writes and exports to sidecars, so keeping it anywhere
// else would produce metadata the rest of the application cannot see.
//
// Consequences of that, both learned the hard way:
//
//   * The table is created by `c41_db::schema::ensure_base_schema` with
//     darktable's exact shape (FK to images + three indexes). This module does
//     NOT issue its own `CREATE TABLE`. An earlier version did, and the ad-hoc
//     DDL it used was NARROWER than upstream's — it would have produced a
//     catalogue the C app could open but whose constraints silently differed.
//   * The reads and writes go through `c41_db::metadata`, which already existed
//     (Phase 2-db-3) and is the FFI-facing implementation of metadata.c. What
//     lives here is only the path→imgid resolution and the UI's field set.
//
// The unique index is on `(id, key, value)` — NOT `(id, key)`. So it does not
// enforce one value per key, and `ON CONFLICT(id, key)` has no index to target:
// an upsert is unavailable, and the write is delete-then-insert exactly as
// upstream does it (`src/common/metadata.c:310`, `:323`).

/// The user-editable metadata fields, in darktable's own display order.
///
/// The discriminants are darktable's key ids, seeded into `data.meta_data` in
/// `src/common/database.c:3253` as the index of each row in its `metadata_fields`
/// array. **They are persisted values, not ours to renumber** — a change here
/// silently re-labels every existing row (a creator would read back as a title).
///
/// Upstream defines nine keys; these are the five its metadata editor shows by
/// default. The remaining four are either internal (`image id`,
/// `preserved filename`) or off by default (`notes`, `version name`), and are
/// left alone rather than half-exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaField {
    Title = 2,
    Description = 3,
    Creator = 0,
    Publisher = 1,
    Rights = 4,
}

impl MetaField {
    /// Display order, matching upstream's `display_order` for these five.
    pub const ALL: [MetaField; 5] = [
        MetaField::Title,
        MetaField::Description,
        MetaField::Creator,
        MetaField::Publisher,
        MetaField::Rights,
    ];

    /// darktable's `meta_data.key`.
    pub fn key(self) -> i32 {
        self as i32
    }

    /// The label upstream uses for the field.
    pub fn label(self) -> &'static str {
        match self {
            MetaField::Title => "Title",
            MetaField::Description => "Description",
            MetaField::Creator => "Creator",
            MetaField::Publisher => "Publisher",
            MetaField::Rights => "Rights",
        }
    }
}

/// Read the five editable fields for an image, in [`MetaField::ALL`] order.
///
/// Always returns all five; a field with no row reads as an empty string, so the
/// caller can drive a fixed set of entries without distinguishing "absent" from
/// "blank". Empty on any failure — an uncatalogued image or a database that
/// predates the table simply has no metadata.
pub fn load_metadata(db_path: &str, full_path: &str) -> Vec<(MetaField, String)> {
    if db_path.is_empty() {
        return Vec::new();
    }
    let Ok(conn) = open_catalog(db_path) else { return Vec::new() };
    load_metadata_conn(&conn, full_path)
}

fn load_metadata_conn(conn: &Connection, full_path: &str) -> Vec<(MetaField, String)> {
    let Some(imgid) = imgid_for_path(conn, full_path) else { return Vec::new() };
    // One query for all keys, then project onto our field set — `metadata_get_all`
    // is the same read the FFI path uses, so both see identical values.
    let all = c41_db::metadata::metadata_get_all(conn, imgid).unwrap_or_default();
    MetaField::ALL
        .iter()
        .map(|&f| {
            let v = all
                .iter()
                .find(|(k, _)| *k == f.key())
                .map(|(_, v)| v.clone())
                .unwrap_or_default();
            (f, v)
        })
        .collect()
}

/// Like [`load_metadata`], but distinguishes failure from genuine blank:
/// `None` means the catalogue could not be consulted (database missing or
/// unreadable, image not catalogued); `Some` carries all five fields even
/// when every one is empty. Callers about to *erase* sidecar content from
/// these values must use this one — an all-empty read that is really a failed
/// lookup must not be mistaken for "the user blanked everything".
pub(crate) fn try_load_metadata(
    db_path: &str,
    full_path: &str,
) -> Option<Vec<(MetaField, String)>> {
    if db_path.is_empty() {
        return None;
    }
    let conn = open_catalog(db_path).ok()?;
    imgid_for_path(&conn, full_path)?;
    Some(load_metadata_conn(&conn, full_path))
}

/// Write metadata for an image. Returns whether the write landed.
///
/// Only the fields named in `fields` are touched, so a caller editing one entry
/// cannot blank the other four. An **empty value deletes** the row rather than
/// storing `''` — that is upstream's convention, and it keeps "no title" as one
/// state instead of two that compare unequal in a filter.
///
/// The XMP sidecar is synchronised by the CALLER after a successful write
/// (`crate::xmp::sync_sidecar`, m4-142) — mirroring upstream, where
/// `dt_metadata_set_list` is followed by `dt_image_synch_xmps`
/// (`src/libs/metadata.c:393`). This function stays storage-only so both
/// stores keep a clear order: catalogue first (authoritative), sidecar second
/// (a failed sidecar write is reported, never rolled back).
pub fn save_metadata(db_path: &str, full_path: &str, fields: &[(MetaField, String)]) -> bool {
    if db_path.is_empty() || fields.is_empty() {
        return false;
    }
    let path = full_path.to_string();
    save_metadata_many(db_path, std::slice::from_ref(&path), fields).len() == 1
}

/// Write `fields` to EVERY image in `full_paths`, returning the paths that were
/// actually written. An image with no catalogue row has no imgid to key
/// metadata against and is skipped — it is simply absent from the return, so
/// callers compare its length against `full_paths.len()` to surface the skip
/// (the panel reports `Saved … for K of N images`, m4-145 review BLOCKER-1).
///
/// m4-145: darktable's metadata editor writes each edited field to the WHOLE
/// lighttable selection (`dt_metadata_set_list` over the selected ids), so the
/// panel hands its snapshotted target list here. Each image keeps its own
/// transaction ([`save_metadata_conn`]) so one unresolvable or failing image
/// cannot roll back the others; sharing ONE connection across the loop simply
/// removes N−1 opens.
pub fn save_metadata_many(
    db_path: &str,
    full_paths: &[String],
    fields: &[(MetaField, String)],
) -> Vec<String> {
    if db_path.is_empty() || fields.is_empty() {
        return Vec::new();
    }
    let Ok(mut conn) = open_catalog(db_path) else { return Vec::new() };
    let mut written = Vec::new();
    for path in full_paths {
        if save_metadata_conn(&mut conn, path, fields).is_ok() {
            written.push(path.clone());
        }
    }
    written
}

fn save_metadata_conn(
    conn: &mut Connection,
    full_path: &str,
    fields: &[(MetaField, String)],
) -> rusqlite::Result<()> {
    // One transaction around the whole write: a delete that commits without its
    // insert would silently erase metadata the user was editing, not merely fail
    // to save it. The imgid lookup is inside it too, so the row cannot vanish
    // between resolving it and writing against it.
    let tx = conn.transaction()?;
    let Some(imgid) = imgid_for_path(&tx, full_path) else {
        // No catalogue row means no id to key metadata against.
        return Err(rusqlite::Error::QueryReturnedNoRows);
    };
    for (field, value) in fields {
        // Upstream trims spaces only (`_cleanup_metadata_value`,
        // src/common/metadata.c:400); matching that keeps values byte-identical
        // to darktable's for the same input.
        let v = value.trim_matches(' ');
        if v.is_empty() {
            // Blank deletes rather than storing '' — upstream's convention, and
            // it keeps "no title" as one state instead of two that compare
            // unequal in a filter.
            c41_db::metadata::metadata_delete_key(&tx, imgid, field.key())?;
        } else {
            c41_db::metadata::metadata_set_value(&tx, imgid, field.key(), v)?;
        }
    }
    tx.commit()
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    /// Same real-temp-file rationale as `style_tests::TmpDb`: the public API
    /// takes a path and opens its own connection, which an in-memory database
    /// would not exercise.
    /// Builds the catalogue from `ensure_base_schema` — the SAME DDL production
    /// uses — so these tests run against darktable's real constraints
    /// (`UNIQUE(id, key, value)`, the FK to `images`). An earlier version of this
    /// fixture created its own narrower `meta_data`, which meant no test ever
    /// exercised the constraints the app actually writes under.
    fn catalogued_db(tag: &str) -> (String, String) {
        let mut p = std::env::temp_dir();
        p.push(format!("c41-meta-{tag}-{:?}.db", std::thread::current().id()));
        let path = p.to_string_lossy().into_owned();
        let _ = std::fs::remove_file(&path);
        let conn = Connection::open(&path).unwrap();
        c41_db::schema::ensure_base_schema(&conn).unwrap();
        conn.execute("INSERT INTO main.film_rolls (id, folder) VALUES (1, '/photos')", [])
            .unwrap();
        conn.execute(
            "INSERT INTO main.images (id, film_id, filename) VALUES (7, 1, 'a.raw')",
            [],
        )
        .unwrap();
        (path, "/photos/a.raw".to_string())
    }

    #[test]
    fn key_ids_match_darktable() {
        // Pinned against src/common/database.c:3253 — these are persisted, and
        // renumbering them would re-label every existing row.
        assert_eq!(MetaField::Creator.key(), 0);
        assert_eq!(MetaField::Publisher.key(), 1);
        assert_eq!(MetaField::Title.key(), 2);
        assert_eq!(MetaField::Description.key(), 3);
        assert_eq!(MetaField::Rights.key(), 4);
    }

    #[test]
    fn all_is_in_darktable_display_order() {
        // display_order[] = {2,3,0,1,4,…} in src/common/database.c:3267 maps
        // field→position, so sorted it reads title, description, creator,
        // publisher, rights. This governs BOTH the on-screen row order and the
        // index alignment the panel relies on when it zips ALL against the
        // entries, so it is worth pinning separately from the key ids.
        use MetaField::*;
        assert_eq!(MetaField::ALL, [Title, Description, Creator, Publisher, Rights]);
    }

    #[test]
    fn schema_matches_darktables_unique_index() {
        // The index is on (id, key, VALUE) — not (id, key). That is why there is
        // no upsert and why the writers delete before inserting. If this ever
        // becomes (id, key), the delete-then-insert dance can be replaced by a
        // real ON CONFLICT upsert.
        let (db, _) = catalogued_db("schema");
        let conn = Connection::open(&db).unwrap();
        let sql: String = conn
            .query_row(
                "SELECT sql FROM main.sqlite_master WHERE type='index' AND name='metadata_index'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(sql.contains("(id, key, value)"), "unexpected index shape: {sql}");
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn round_trips_and_reads_absent_fields_as_blank() {
        let (db, img) = catalogued_db("roundtrip");
        assert!(save_metadata(&db, &img, &[(MetaField::Title, "Sunset".into())]));
        let got = load_metadata(&db, &img);
        assert_eq!(got.len(), 5, "all five fields are always returned");
        assert_eq!(got[0], (MetaField::Title, "Sunset".to_string()));
        // The untouched four read as empty, not as missing entries.
        assert!(got[1..].iter().all(|(_, v)| v.is_empty()), "{got:?}");
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn writing_one_field_leaves_the_others_alone() {
        let (db, img) = catalogued_db("partial");
        assert!(save_metadata(&db, &img, &[(MetaField::Title, "T".into()),
                                           (MetaField::Creator, "C".into())]));
        assert!(save_metadata(&db, &img, &[(MetaField::Title, "T2".into())]));
        let got = load_metadata(&db, &img);
        let creator = got.iter().find(|(f, _)| *f == MetaField::Creator).unwrap();
        assert_eq!(creator.1, "C", "editing the title blanked the creator");
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn rewrite_replaces_rather_than_duplicating() {
        // The table has no uniqueness constraint, so a missing DELETE would
        // accumulate rows and the reader would keep serving the FIRST one —
        // edits would appear to do nothing.
        let (db, img) = catalogued_db("dupe");
        save_metadata(&db, &img, &[(MetaField::Title, "first".into())]);
        save_metadata(&db, &img, &[(MetaField::Title, "second".into())]);
        let conn = Connection::open(&db).unwrap();
        let n: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM main.meta_data WHERE id = 7 AND key = 2",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "rewrite duplicated the row instead of replacing it");
        assert_eq!(load_metadata(&db, &img)[0].1, "second");
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn blank_value_deletes_the_row() {
        let (db, img) = catalogued_db("blank");
        save_metadata(&db, &img, &[(MetaField::Rights, "CC-BY".into())]);
        save_metadata(&db, &img, &[(MetaField::Rights, "   ".into())]);
        let conn = Connection::open(&db).unwrap();
        let n: i32 = conn
            .query_row("SELECT COUNT(*) FROM main.meta_data WHERE id = 7 AND key = 4", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(n, 0, "a blank value should delete, not store an empty string");
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn uncatalogued_image_is_reported_not_silently_dropped() {
        let (db, _) = catalogued_db("uncatalogued");
        assert!(!save_metadata(&db, "/photos/missing.raw", &[(MetaField::Title, "x".into())]));
        assert!(load_metadata(&db, "/photos/missing.raw").is_empty());
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn save_many_fans_out_and_reports_skips_by_absence() {
        // m4-145: the panel's whole-selection commit. Every catalogued target
        // gets the write; an uncatalogued path is skipped and REPORTED by its
        // absence from the return value — it must not abort the rest, and it
        // must not be silently counted as done.
        let (db, img) = catalogued_db("many");
        let conn = Connection::open(&db).unwrap();
        conn.execute("INSERT INTO main.images (id, film_id, filename) VALUES (8, 1, 'b.raw')", [])
            .unwrap();
        conn.execute("INSERT INTO main.images (id, film_id, filename) VALUES (9, 1, 'c.raw')", [])
            .unwrap();
        drop(conn);
        let targets = vec![
            img.clone(),
            "/photos/b.raw".to_string(),
            "/photos/gone.raw".to_string(),
            "/photos/c.raw".to_string(),
        ];
        let written =
            save_metadata_many(&db, &targets, &[(MetaField::Title, "batch".into())]);
        assert_eq!(
            written,
            vec![img.clone(), "/photos/b.raw".into(), "/photos/c.raw".into()],
            "the uncatalogued path is reported by absence, in input order"
        );
        for path in [img, "/photos/b.raw".to_string(), "/photos/c.raw".to_string()] {
            let meta = load_metadata(&db, &path);
            let title = meta
                .iter()
                .find(|(f, _)| *f == MetaField::Title)
                .map(|(_, v)| v.as_str())
                .unwrap_or("");
            assert_eq!(title, "batch", "{path} missed the fan-out");
        }
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn save_metadata_is_save_many_on_a_single_path() {
        // Delegation pin: since m4-145 the single-image API is
        // save_metadata_many over a one-element slice. If they ever drift,
        // single edits and selection fan-outs would disagree about what a
        // successful write even is.
        let (db, img) = catalogued_db("delegate");
        assert!(save_metadata(&db, &img, &[(MetaField::Creator, "Ansel".into())]));
        assert_eq!(
            save_metadata_many(&db, &[img.clone()], &[(MetaField::Creator, "Ansel".into())]),
            vec![img.clone()]
        );
        let creator = load_metadata(&db, &img)
            .iter()
            .find(|(f, _)| *f == MetaField::Creator)
            .map(|(_, v)| v.clone())
            .unwrap();
        assert_eq!(creator, "Ansel");
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn duplicate_rows_yield_a_stored_value_and_are_healed_by_a_write() {
        // The unique index is on (id, key, value), so two rows sharing (id, key)
        // with DIFFERENT values are legal — legacy databases do contain them, which
        // is why upstream ships a dedupe migration (src/common/database.c:1778).
        // Which of the two a read returns is NOT specified: the projection takes
        // the first row `metadata_get_all` yields, and its ORDER BY is on `key`,
        // so ties fall to whatever plan SQLite picks. Pinning "older" here would
        // be pinning the query planner, not our behaviour.
        //
        // What IS guaranteed, and what actually matters: a duplicate never reads
        // back blank, and the next write collapses it to one row.
        let (db, img) = catalogued_db("legacydupe");
        let conn = Connection::open(&db).unwrap();
        conn.execute("INSERT INTO main.meta_data VALUES (7, 2, 'older')", []).unwrap();
        conn.execute("INSERT INTO main.meta_data VALUES (7, 2, 'newer')", []).unwrap();
        drop(conn);

        let got = load_metadata(&db, &img)[0].1.clone();
        assert!(got == "older" || got == "newer", "duplicate read back as {got:?}");

        assert!(save_metadata(&db, &img, &[(MetaField::Title, "single".into())]));
        let conn = Connection::open(&db).unwrap();
        let n: i32 = conn
            .query_row("SELECT COUNT(*) FROM main.meta_data WHERE id = 7 AND key = 2", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(n, 1, "a write should collapse legacy duplicates to one row");
        assert_eq!(load_metadata(&db, &img)[0].1, "single");
        let _ = std::fs::remove_file(&db);
    }
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
        assert!(save_style(&db, "Punchy", "high contrast", &params_with(1.5), None));
        let styles = load_styles(&db);
        assert_eq!(styles.len(), 1);
        assert_eq!(styles[0].name, "Punchy");
        assert_eq!(styles[0].description, "high contrast");
        assert_eq!(styles[0].params.ev, 1.5, "params must survive verbatim");
    }

    #[test]
    fn open_catalog_installs_the_three_second_busy_timeout() {
        // The one load-bearing line of m4-147: without this pin, deleting the
        // busy_timeout call would leave every other test green while silently
        // reverting the whole increment (review MINOR-3). PRAGMA busy_timeout
        // reads back exactly what sqlite3_busy_timeout installed.
        let (_d, db) = tmp_db("busytimeout");
        let conn = open_catalog(&db).unwrap();
        let ms: i64 = conn
            .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ms, 3000);
    }

    #[test]
    fn save_overwrites_by_name_rather_than_duplicating() {
        let (_d, db) = tmp_db("upsert");
        assert!(save_style(&db, "S", "first", &params_with(1.0), None));
        assert!(save_style(&db, "S", "second", &params_with(2.0), None));
        let styles = load_styles(&db);
        assert_eq!(styles.len(), 1, "same name must upsert, not duplicate");
        assert_eq!(styles[0].description, "second");
        assert_eq!(styles[0].params.ev, 2.0);
    }

    #[test]
    fn blank_names_are_rejected() {
        let (_d, db) = tmp_db("blank");
        assert!(!save_style(&db, "", "x", &params_with(1.0), None));
        assert!(!save_style(&db, "   ", "x", &params_with(1.0), None));
        assert!(load_styles(&db).is_empty());
        // Names are trimmed, so " S " and "S" are the same style rather than two
        // rows that look identical in the list.
        assert!(save_style(&db, " S ", "", &params_with(1.0), None));
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
        save_style(&db, "a", "", &params_with(1.0), None);
        save_style(&db, "b", "", &params_with(2.0), None);
        assert!(delete_style(&db, "a"));
        assert!(!delete_style(&db, "a"), "second delete removes nothing");
        let names: Vec<_> = load_styles(&db).into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["b"]);
    }

    #[test]
    fn styles_list_is_name_ordered_case_insensitively() {
        let (_d, db) = tmp_db("order");
        for n in ["zebra", "Apple", "mango"] {
            save_style(&db, n, "", &params_with(0.5), None);
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
        save_style(&db, "S", "", &params_with(2.5), None);
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
        save_style(&db, "S", "", &params_with(1.5), None);
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
        save_style(&db, "S", "", &params_with(1.0), None);
        let style = load_styles(&db).remove(0);
        assert_eq!(apply_style_to(&db, &["".to_string()], &style), 0);
    }

    #[test]
    fn partial_style_roundtrips_its_module_list() {
        let (_d, db) = tmp_db("partialroundtrip");
        assert!(save_style(
            &db,
            "Velvia only",
            "",
            &params_with(9.0),
            Some(&["Velvia", "Levels"])
        ));
        let style = load_styles(&db).remove(0);
        assert_eq!(
            style.modules,
            Some(vec!["Velvia".to_string(), "Levels".to_string()])
        );

        // Re-saving the same name with different scope must UPDATE the column
        // (the upsert's ON CONFLICT arm carries it), not leave the old list.
        assert!(save_style(&db, "Velvia only", "", &params_with(9.0), None));
        assert_eq!(load_styles(&db).remove(0).modules, None);
    }

    #[test]
    fn whole_style_stores_a_null_modules_column() {
        // Legacy compatibility is the POINT of None: pre-149 rows have NULL and
        // must be indistinguishable from a fresh whole-edit save.
        let (_d, db) = tmp_db("nullcol");
        assert!(save_style(&db, "Whole", "", &params_with(1.0), None));
        let conn = open_catalog(&db).unwrap();
        let n: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM main.c41_styles WHERE modules IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);

        // An empty selection is a valid zero-module style — stored as '' so it
        // round-trips as Some(vec![]) rather than collapsing into NULL/whole.
        drop(conn);
        assert!(save_style(&db, "Nothing", "", &params_with(1.0), Some(&[])));
        let conn = open_catalog(&db).unwrap();
        let raw: String = conn
            .query_row(
                "SELECT modules FROM main.c41_styles WHERE name = 'Nothing'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(raw, "");
        assert_eq!(
            load_styles(&db)
                .into_iter()
                .find(|s| s.name == "Nothing")
                .unwrap()
                .modules,
            Some(Vec::new())
        );
    }

    #[test]
    fn applying_a_partial_style_merges_over_the_saved_edit() {
        // The m4-149 behaviour itself: the target keeps its own exposure and
        // gains only the listed group from the style.
        let (_d, db) = tmp_db("merge");
        catalogue(&db, "/photos", &["a.dng"]);
        let img = "/photos/a.dng";
        save_params(&db, img, &params_with(1.0));

        let mut overlay = params_with(9.0);
        overlay.velvia_strength = 42.0;
        assert!(save_style(&db, "Punch", "", &overlay, Some(&["Velvia"])));
        let style = load_styles(&db).remove(0);

        assert_eq!(apply_style_to(&db, &[img.to_string()], &style), 1);
        let merged = load_params(&db, img);
        assert_eq!(merged.ev, 1.0, "unlisted module must keep the target's value");
        assert_eq!(merged.velvia_strength, 42.0, "listed module must come from the style");
    }

    #[test]
    fn applying_a_partial_style_onto_an_unedited_image_starts_from_defaults() {
        // No saved edit yet → base is defaults, so unlisted fields land at
        // default values (NOT at the style's), matching darktable applying a
        // partial style onto an untouched image.
        let (_d, db) = tmp_db("mergenew");
        catalogue(&db, "/photos", &["a.dng"]);
        let img = "/photos/a.dng";
        assert!(load_saved(&db, img).is_none());

        let mut overlay = params_with(-2.0);
        overlay.velvia_strength = 42.0;
        assert!(save_style(&db, "Punch", "", &overlay, Some(&["Velvia"])));
        let style = load_styles(&db).remove(0);

        assert_eq!(apply_style_to(&db, &[img.to_string()], &style), 1);
        let merged = load_params(&db, img);
        assert_eq!(merged.ev, 0.0, "default, not the style's -2.0");
        assert_eq!(merged.velvia_strength, 42.0);
    }

    #[test]
    fn discard_history_clears_both_rows() {
        let (_d, db) = tmp_db("discard");
        catalogue(&db, "/photos", &["a.raw"]);
        let img = "/photos/a.raw";
        save_params(&db, img, &PreviewParams { ev: 1.5, ..Default::default() });
        let mut h = HistoryStack::new("Original", PreviewParams::default());
        h.record("Exposure", PreviewParams { ev: 0.9, ..Default::default() });
        save_history(&db, img, &h);
        assert!(load_saved(&db, img).is_some());
        assert!(load_history(&db, img).is_some());

        discard_history(&db, img);
        assert!(load_saved(&db, img).is_none(), "params row must go");
        assert!(load_history(&db, img).is_none(), "history row must go");
    }

    #[test]
    fn discard_is_a_noop_for_uncatalogued_images() {
        // No imgid for the path → there is nothing to delete; the call must not
        // panic and must leave the catalogue untouched.
        let (_d, db) = tmp_db("discardmiss");
        discard_history(&db, "/photos/never-seen.raw"); // must not panic
        assert!(load_saved(&db, "/photos/never-seen.raw").is_none());
        assert!(load_history(&db, "/photos/never-seen.raw").is_none());
    }

    #[test]
    fn collection_presets_round_trip_upsert_and_delete() {
        let (_d, db) = tmp_db("colpresets");
        // Blank/whitespace names and payloads are refused, and nothing exists yet.
        assert!(!save_collection_preset(&db, "", "v1 off off off off"));
        assert!(!save_collection_preset(&db, "  ", "v1 off off off off"));
        assert!(!save_collection_preset(&db, "x", ""));
        assert!(load_collection_presets(&db).is_empty());
        // Upsert by name: a collision replaces rather than duplicating.
        assert!(save_collection_preset(&db, " S ", "p1"));
        assert!(save_collection_preset(&db, "S", "p2"));
        assert_eq!(load_collection_presets(&db), vec![("S".into(), "p2".into())]);
        // Name-ordered case-insensitively, like styles.
        save_collection_preset(&db, "zebra", "z");
        save_collection_preset(&db, "Apple", "a");
        let names: Vec<_> = load_collection_presets(&db).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["Apple", "S", "zebra"]);
        // Delete removes exactly the named row; a second delete reports false.
        assert!(delete_collection_preset(&db, "S"));
        assert!(!delete_collection_preset(&db, "S"), "second delete removes nothing");
        assert_eq!(load_collection_presets(&db).len(), 2);
    }
}
