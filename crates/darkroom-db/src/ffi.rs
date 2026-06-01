//! C FFI trampolines for darkroom-db.
//!
//! Each function accepts the raw `sqlite3*` handle from `darktable.db` and
//! wraps it in a temporary `rusqlite::Connection` (without taking ownership
//! of the handle — `from_handle_shared` keeps the C side in control of
//! the connection lifetime).  The Rust logic lives in the sibling modules
//! (`tags`, `image`, …); these trampolines are thin adapters.

use rusqlite::Connection;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Wrap a raw sqlite3 handle into a non-owning rusqlite Connection.
///
/// # Safety
/// The caller must ensure `db_handle` is a valid, open sqlite3* that
/// outlives the returned Connection.
#[inline(always)]
unsafe fn conn_from_handle(db_handle: *mut std::ffi::c_void) -> Connection {
    // rusqlite's `from_handle` takes a *mut sqlite3 which is an opaque pointer;
    // we transmute from *mut c_void since C callers pass it as void*.
    let raw: *mut rusqlite::ffi::sqlite3 = db_handle.cast();
    // SAFETY: caller guarantees db_handle is a valid open sqlite3*.
    unsafe { Connection::from_handle(raw).expect("from_handle") }
}

// ── tags ──────────────────────────────────────────────────────────────────────

/// `dt_tag_new` Rust bridge.
///
/// Creates or upserts a tag. Writes the ID into `*out_tagid` if non-null.
/// Returns 1 on success, 0 on failure or empty name.
///
/// # Safety
/// `db_handle` must be a valid `sqlite3*`; `name` must be a valid NUL-terminated
/// C string; `out_tagid` may be null.
#[no_mangle]
pub unsafe extern "C" fn darkroom_tag_new(
    db_handle: *mut std::ffi::c_void,
    name:      *const std::os::raw::c_char,
    out_tagid: *mut u32,
) -> std::os::raw::c_int {
    let name_str = if name.is_null() {
        return 0;
    } else {
        match std::ffi::CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };
    let conn = conn_from_handle(db_handle);
    match crate::tags::tag_new(&conn, name_str) {
        Ok(Some(id)) => {
            if !out_tagid.is_null() { *out_tagid = id; }
            1
        }
        _ => 0,
    }
}

/// `dt_tag_exists` Rust bridge.
///
/// Returns 1 if the tag exists, 0 otherwise. Writes the ID into `*out_tagid`.
///
/// # Safety
/// Same as `darkroom_tag_new`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_tag_exists(
    db_handle: *mut std::ffi::c_void,
    name:      *const std::os::raw::c_char,
    out_tagid: *mut u32,
) -> std::os::raw::c_int {
    let name_str = if name.is_null() { return 0; }
    else { match std::ffi::CStr::from_ptr(name).to_str() { Ok(s) => s, Err(_) => return 0 } };
    let conn = conn_from_handle(db_handle);
    match crate::tags::tag_exists(&conn, name_str) {
        Ok(Some(id)) => { if !out_tagid.is_null() { *out_tagid = id; } 1 }
        Ok(None)     => { if !out_tagid.is_null() { *out_tagid = 0; }  0 }
        Err(_)       => 0,
    }
}

/// `dt_tag_attach` Rust bridge.
///
/// Associates tag `tagid` with image `imgid`. Returns 1 if the row was
/// newly inserted, 0 if it already existed or on error.
///
/// # Safety
/// `db_handle` must be a valid `sqlite3*`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_tag_attach(
    db_handle: *mut std::ffi::c_void,
    tagid:     u32,
    imgid:     i32,
) -> std::os::raw::c_int {
    let conn = conn_from_handle(db_handle);
    match crate::tags::tag_attach(&conn, tagid, imgid) {
        Ok(inserted) => if inserted { 1 } else { 0 },
        Err(_)       => 0,
    }
}

/// `dt_tag_detach` Rust bridge.
///
/// Removes the association between tag `tagid` and image `imgid`.
/// Returns 1 if a row was removed, 0 otherwise.
///
/// # Safety
/// `db_handle` must be a valid `sqlite3*`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_tag_detach(
    db_handle: *mut std::ffi::c_void,
    tagid:     u32,
    imgid:     i32,
) -> std::os::raw::c_int {
    let conn = conn_from_handle(db_handle);
    match crate::tags::tag_detach(&conn, tagid, imgid) {
        Ok(removed) => if removed { 1 } else { 0 },
        Err(_)      => 0,
    }
}

/// `dt_tag_count_attached` Rust bridge.
///
/// Returns the count of tags attached to `imgid`.
/// If `include_dt_tags` is 0, `darktable|*` tags are excluded.
///
/// # Safety
/// `db_handle` must be a valid `sqlite3*`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_tag_count_attached(
    db_handle:      *mut std::ffi::c_void,
    imgid:          i32,
    include_dt_tags: std::os::raw::c_int,
) -> u32 {
    let conn = conn_from_handle(db_handle);
    crate::tags::tag_count_attached(&conn, imgid, include_dt_tags != 0).unwrap_or(0)
}

/// `dt_tag_delete` Rust bridge.
///
/// Deletes the tag and all its image associations.
/// Returns 1 on success, 0 on error.
///
/// # Safety
/// `db_handle` must be a valid `sqlite3*`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_tag_delete(
    db_handle: *mut std::ffi::c_void,
    tagid:     u32,
) -> std::os::raw::c_int {
    let conn = conn_from_handle(db_handle);
    match crate::tags::tag_delete(&conn, tagid) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}
