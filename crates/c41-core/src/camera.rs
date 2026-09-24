//! Tethered camera capture via libgphoto2 (u6; parity audit 3.4, tethering leg).
//!
//! Scope is detect plus capture-from-the-first-camera only: no burst,
//! timelapse, live view, config widgets, or filesystem browsing. The open
//! sequence mirrors darktable's `_camera_initialize`
//! (`src/common/camera_control.c`): new, abilities lookup by model plus
//! set, port lookup by path plus set, then init — in exactly that order.
//! Capture mirrors the `_JOB_TYPE_EXECUTE_CAPTURE` job: capture to a
//! `CameraFilePath`, then file_get plus file_save.
//!
//! `detect_cameras` never panics and returns an empty vec when no camera is
//! attached (autodetect with nothing connected completes immediately).
//! `capture_to` saves to the exact path given, creating the parent dir;
//! `capture_into` picks the destination inside a directory from the camera's
//! own filename (`tether-YYYYMMDD-HHMMSS` stem plus the camera's extension,
//! sanitized, de-duplicated with a `-N` suffix) and returns the final path.
//!
//! The pure helpers (`camera_extension`,
//! `capture_dest_path`, `format_detect_status`, `tether_capture_dir`) are
//! GTK-free and unit-tested headless; the hardware paths are exercised only
//! by the graceful-absence test, which asserts `detect_cameras` returns
//! without asserting anything about its contents.

use c41_sys::gphoto;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::time::SystemTime;

// ── Error ────────────────────────────────────────────────────────────────────

/// A libgphoto2 failure tagged with the pipeline step that produced it, so a
/// status line can say where a capture died, not just the library's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraError {
    /// Pipeline step: one of `context`, `detect`, `open`, `abilities`,
    /// `port`, `init`, `capture`, `download`, `save`, `filename`, `makedir`.
    pub step: &'static str,
    /// `gp_result_as_string` text, or a plain message for non-library errors.
    pub message: String,
}

impl CameraError {
    pub fn new(step: &'static str, message: impl Into<String>) -> Self {
        Self {
            step,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CameraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.step, self.message)
    }
}

impl std::error::Error for CameraError {}

/// Wrap a libgphoto2 return code: `GP_OK` passes, anything else becomes a
/// `CameraError` at `step` with the library's own message (null-safe).
fn check_gp(code: std::ffi::c_int, step: &'static str) -> Result<(), CameraError> {
    if code == gphoto::GP_OK {
        return Ok(());
    }
    let message = unsafe {
        let ptr = gphoto::gp_result_as_string(code);
        if ptr.is_null() {
            format!("gphoto2 error {code}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(CameraError::new(step, message))
}

/// Read a NUL-terminated byte array field (the `CameraFilePath` members) as
/// an owned string; stops at the first NUL, lossy on non-UTF8.
fn field_to_string(field: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = field
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ── Detection ────────────────────────────────────────────────────────────────

/// List attached cameras as `(model, port)` pairs via `gp_camera_autodetect`.
/// Empty when none is attached — never panics, never blocks on hardware I/O
/// beyond what autodetect itself does (immediate with no camera).
pub fn detect_cameras() -> Vec<(String, String)> {
    unsafe {
        let ctx = gphoto::gp_context_new();
        if ctx.is_null() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut list: *mut gphoto::CameraList = null_mut();
        if gphoto::gp_list_new(&mut list) == gphoto::GP_OK && !list.is_null() {
            if gphoto::gp_camera_autodetect(list, ctx) == gphoto::GP_OK {
                let n = gphoto::gp_list_count(list).max(0);
                for i in 0..n {
                    let mut name: *const std::ffi::c_char = null_mut();
                    let mut value: *const std::ffi::c_char = null_mut();
                    if gphoto::gp_list_get_name(list, i, &mut name) != gphoto::GP_OK
                        || gphoto::gp_list_get_value(list, i, &mut value) != gphoto::GP_OK
                    {
                        continue;
                    }
                    if name.is_null() || value.is_null() {
                        continue;
                    }
                    out.push((
                        CStr::from_ptr(name).to_string_lossy().into_owned(),
                        CStr::from_ptr(value).to_string_lossy().into_owned(),
                    ));
                }
            }
            gphoto::gp_list_free(list);
        }
        gphoto::gp_context_unref(ctx);
        out
    }
}

// ── Open sequence (mirrors `_camera_initialize`) ─────────────────────────────

/// Open the camera with the given model/port strings. Steps run in darktable's
/// order: `gp_camera_new`, abilities lookup by model plus set, port lookup by
/// path plus set, then `gp_camera_init`. The caller owns the returned camera
/// and must pass it to `close_camera` on every path, success or error.
unsafe fn open_first_camera(
    ctx: *mut gphoto::GPContext,
    model: &str,
    port: &str,
) -> Result<*mut gphoto::Camera, CameraError> {
    let model_c = CString::new(model).map_err(|_| CameraError::new("open", "camera model is not a C string"))?;
    let port_c = CString::new(port).map_err(|_| CameraError::new("open", "camera port is not a C string"))?;

    let mut cam: *mut gphoto::Camera = null_mut();
    check_gp(gphoto::gp_camera_new(&mut cam), "open")?;
    if cam.is_null() {
        return Err(CameraError::new("open", "gp_camera_new returned NULL"));
    }
    // From here every error path must free `cam`; delegate the fallible tail
    // to an inner fn so there is exactly one cleanup site.
    let tail = open_camera_tail(ctx, cam, &model_c, &port_c);
    if tail.is_err() {
        close_camera(cam, ctx);
    }
    tail.map(|()| cam)
}

/// Abilities plus port setup plus init for an already-created camera.
unsafe fn open_camera_tail(
    ctx: *mut gphoto::GPContext,
    cam: *mut gphoto::Camera,
    model_c: &CString,
    port_c: &CString,
) -> Result<(), CameraError> {
    // Abilities: new, load, lookup the autodetected model, get, set.
    let mut alist: *mut gphoto::CameraAbilitiesList = null_mut();
    check_gp(gphoto::gp_abilities_list_new(&mut alist), "abilities")?;
    if alist.is_null() {
        return Err(CameraError::new("abilities", "gp_abilities_list_new returned NULL"));
    }
    let abilities_result = open_abilities_inner(alist, ctx, cam, model_c);
    gphoto::gp_abilities_list_free(alist);
    abilities_result?;

    // Ports: new, load, lookup the autodetected path, get, set.
    let mut plist: *mut gphoto::GPPortInfoList = null_mut();
    check_gp(gphoto::gp_port_info_list_new(&mut plist), "port")?;
    if plist.is_null() {
        return Err(CameraError::new("port", "gp_port_info_list_new returned NULL"));
    }
    let port_result = open_port_inner(plist, cam, port_c);
    gphoto::gp_port_info_list_free(plist);
    port_result?;

    check_gp(gphoto::gp_camera_init(cam, ctx), "init")
}

unsafe fn open_abilities_inner(
    alist: *mut gphoto::CameraAbilitiesList,
    ctx: *mut gphoto::GPContext,
    cam: *mut gphoto::Camera,
    model_c: &CString,
) -> Result<(), CameraError> {
    check_gp(gphoto::gp_abilities_list_load(alist, ctx), "abilities")?;
    let index = gphoto::gp_abilities_list_lookup_model(alist, model_c.as_ptr());
    if index < 0 {
        return check_gp(index, "abilities");
    }
    // Zeroed first: the getter fills every field on success, and a zeroed
    // struct can never carry a stale model into `gp_camera_set_abilities`.
    let mut abilities: gphoto::CameraAbilities = std::mem::zeroed();
    check_gp(
        gphoto::gp_abilities_list_get_abilities(alist, index, &mut abilities),
        "abilities",
    )?;
    check_gp(gphoto::gp_camera_set_abilities(cam, abilities), "abilities")
}

unsafe fn open_port_inner(
    plist: *mut gphoto::GPPortInfoList,
    cam: *mut gphoto::Camera,
    port_c: &CString,
) -> Result<(), CameraError> {
    check_gp(gphoto::gp_port_info_list_load(plist), "port")?;
    let index = gphoto::gp_port_info_list_lookup_path(plist, port_c.as_ptr());
    if index < 0 {
        return check_gp(index, "port");
    }
    let mut info: gphoto::GPPortInfo = null_mut();
    check_gp(gphoto::gp_port_info_list_get_info(plist, index, &mut info), "port")?;
    if info.is_null() {
        return Err(CameraError::new("port", "gp_port_info_list_get_info returned NULL"));
    }
    check_gp(gphoto::gp_camera_set_port_info(cam, info), "port")
}

/// Mirror of darktable's destroy order: exit first, then release. The exit
/// result is intentionally ignored — on a failed init there may be nothing
/// to exit, and cleanup must not fail.
unsafe fn close_camera(cam: *mut gphoto::Camera, ctx: *mut gphoto::GPContext) {
    if cam.is_null() {
        return;
    }
    let _ = gphoto::gp_camera_exit(cam, ctx);
    // `gp_camera_unref` is what `camera_control.c` uses to release; the
    // camera was created with refcount 1 so this frees it.
    let _ = gphoto::gp_camera_unref(cam);
}

// ── Capture + download (mirrors the `_JOB_TYPE_EXECUTE_CAPTURE` job) ─────────

/// Capture one image on an open camera and download it to `dest` via
/// `gp_camera_file_get` plus `gp_file_save`. Returns the camera-side
/// filename (the `fp.name` the camera reported) alongside.
unsafe fn capture_and_save(
    cam: *mut gphoto::Camera,
    ctx: *mut gphoto::GPContext,
    dest: &Path,
) -> Result<String, CameraError> {
    let mut fp: gphoto::CameraFilePath = std::mem::zeroed();
    check_gp(
        gphoto::gp_camera_capture(cam, gphoto::GP_CAPTURE_IMAGE, &mut fp, ctx),
        "capture",
    )?;
    let camera_name = field_to_string(&fp.name);
    let camera_folder = field_to_string(&fp.folder);

    let mut file: *mut gphoto::CameraFile = null_mut();
    check_gp(gphoto::gp_file_new(&mut file), "download")?;
    if file.is_null() {
        return Err(CameraError::new("download", "gp_file_new returned NULL"));
    }
    // `CameraFilePath` members are fixed arrays, not NUL-terminated pointers:
    // re-materialize them as CStrings for the folder-first, name-second
    // argument order `camera_control.c:323` uses.
    let folder_c = CString::new(camera_folder)
        .map_err(|_| CameraError::new("download", "camera folder is not a C string"))?;
    let name_c = CString::new(camera_name.clone())
        .map_err(|_| CameraError::new("download", "camera filename is not a C string"))?;
    let got = check_gp(
        gphoto::gp_camera_file_get(
            cam,
            folder_c.as_ptr(),
            name_c.as_ptr(),
            gphoto::GP_FILE_TYPE_NORMAL,
            file,
            ctx,
        ),
        "download",
    );
    if got.is_err() {
        gphoto::gp_file_free(file);
        return got.map(|()| String::new());
    }
    let dest_c = CString::new(dest.to_string_lossy().as_bytes())
        .map_err(|_| CameraError::new("save", "destination path is not a C string"))?;
    let saved = check_gp(gphoto::gp_file_save(file, dest_c.as_ptr()), "save");
    gphoto::gp_file_free(file);
    saved.map(|()| camera_name)
}

/// Run `f` with a fresh context; the context is released on every path.
unsafe fn with_context<T>(f: impl FnOnce(*mut gphoto::GPContext) -> Result<T, CameraError>) -> Result<T, CameraError> {
    let ctx = gphoto::gp_context_new();
    if ctx.is_null() {
        return Err(CameraError::new("context", "gp_context_new returned NULL"));
    }
    let result = f(ctx);
    gphoto::gp_context_unref(ctx);
    result
}

/// Resolve the first autodetected camera's model plus port without opening
/// anything. Errors at step `detect` when no camera is attached.
unsafe fn first_camera_identity(ctx: *mut gphoto::GPContext) -> Result<(String, String), CameraError> {
    let mut list: *mut gphoto::CameraList = null_mut();
    check_gp(gphoto::gp_list_new(&mut list), "detect")?;
    if list.is_null() {
        return Err(CameraError::new("detect", "gp_list_new returned NULL"));
    }
    let result = first_camera_identity_inner(list, ctx);
    gphoto::gp_list_free(list);
    result
}

unsafe fn first_camera_identity_inner(
    list: *mut gphoto::CameraList,
    ctx: *mut gphoto::GPContext,
) -> Result<(String, String), CameraError> {
    check_gp(gphoto::gp_camera_autodetect(list, ctx), "detect")?;
    if gphoto::gp_list_count(list) < 1 {
        return Err(CameraError::new("detect", "no camera detected"));
    }
    let mut name: *const std::ffi::c_char = null_mut();
    let mut value: *const std::ffi::c_char = null_mut();
    check_gp(gphoto::gp_list_get_name(list, 0, &mut name), "detect")?;
    check_gp(gphoto::gp_list_get_value(list, 0, &mut value), "detect")?;
    if name.is_null() || value.is_null() {
        return Err(CameraError::new("detect", "autodetect entry is NULL"));
    }
    Ok((
        CStr::from_ptr(name).to_string_lossy().into_owned(),
        CStr::from_ptr(value).to_string_lossy().into_owned(),
    ))
}

// ── Public entry points ──────────────────────────────────────────────────────

/// Capture one image from the first detected camera and save it to exactly
/// `path`. The parent directory is created when missing. The camera-side
/// filename is ignored — use `capture_into` for camera-derived naming.
pub fn capture_to(path: &Path) -> Result<(), CameraError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CameraError::new("makedir", format!("cannot create {}: {e}", parent.display())))?;
        }
    }
    unsafe {
        with_context(|ctx| {
            let (model, port) = first_camera_identity(ctx)?;
            let cam = open_first_camera(ctx, &model, &port)?;
            let saved = capture_and_save(cam, ctx, path);
            close_camera(cam, ctx);
            saved.map(|_| ())
        })
    }
}

/// Capture one image from the first detected camera into `dir`, naming the
/// file `tether-YYYYMMDD-HHMMSS.<camera-ext>` (see `capture_dest_path`) and
/// returning the final path. The directory is created when missing.
pub fn capture_into(dir: &Path) -> Result<PathBuf, CameraError> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CameraError::new("makedir", format!("cannot create {}: {e}", dir.display())))?;
    unsafe {
        with_context(|ctx| {
            let (model, port) = first_camera_identity(ctx)?;
            let cam = open_first_camera(ctx, &model, &port)?;
            let result = capture_into_inner(cam, ctx, dir);
            close_camera(cam, ctx);
            result
        })
    }
}

unsafe fn capture_into_inner(
    cam: *mut gphoto::Camera,
    ctx: *mut gphoto::GPContext,
    dir: &Path,
) -> Result<PathBuf, CameraError> {
    // Stage through a unique sidecar name first: the final name derives from
    // the camera's own filename, which is only known after the capture, and
    // the pid/nanos suffix keeps concurrent captures from sharing a stage.
    let staging = dir.join(format!(
        ".tether-capture-{}-{}.part",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let camera_name = capture_and_save(cam, ctx, &staging)?;
    let dest = capture_dest_path(dir, &camera_name, SystemTime::now(), &|p| p.exists());
    if let Err(e) = std::fs::rename(&staging, &dest) {
        let _ = std::fs::remove_file(&staging);
        return Err(CameraError::new(
            "save",
            format!("cannot move capture into place: {e}"),
        ));
    }
    Ok(dest)
}

// ── Pure naming + status helpers (GTK-free, headless-tested) ─────────────────

/// The extension to store a capture under: the camera filename's suffix when
/// it is 1..=5 ASCII-alphanumeric chars, else `jpg`. Lowercased so the
/// importer's extension gate (which lowercases before comparing) agrees.
pub fn camera_extension(camera_name: &str) -> String {
    let base = camera_name.rsplit(['/', '\\']).next().unwrap_or(camera_name);
    match base.rsplit_once('.') {
        Some((_, ext))
            if !ext.is_empty()
                && ext.len() <= 5
                && ext.chars().all(|c| c.is_ascii_alphanumeric()) =>
        {
            ext.to_ascii_lowercase()
        }
        _ => "jpg".to_string(),
    }
}

/// Convert unix seconds to `(year, month, day, hour, min, sec)` in UTC
/// (days-from-civil inverse; pre-epoch clamps to the epoch).
pub fn unix_to_ymd_hms(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let secs = secs.max(0) as u64;
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = (if m <= 2 { y + 1 } else { y }) as i32;
    (
        year,
        m,
        d,
        (rem / 3_600) as u32,
        ((rem % 3_600) / 60) as u32,
        (rem % 60) as u32,
    )
}

/// Build the destination for a capture inside `dir`: the
/// `tether-YYYYMMDD-HHMMSS` stem plus the camera's extension, with a `-N`
/// suffix while `exists` reports a collision (same-second repeat captures).
pub fn capture_dest_path(
    dir: &Path,
    camera_name: &str,
    now: SystemTime,
    exists: &dyn Fn(&Path) -> bool,
) -> PathBuf {
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = unix_to_ymd_hms(secs);
    let stem = format!("tether-{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}");
    let ext = camera_extension(camera_name);
    let mut candidate = dir.join(format!("{stem}.{ext}"));
    let mut n = 1u32;
    while exists(&candidate) {
        n += 1;
        candidate = dir.join(format!("{stem}-{n}.{ext}"));
    }
    candidate
}

/// Directory captures land in when no watch folder is set: an `incoming`
/// dir under the catalogue dir (created on use by the capture calls).
pub fn tether_capture_dir(catalogue_dir: &Path) -> PathBuf {
    catalogue_dir.join("incoming")
}

/// One-line status for the detect outcome: the camera list, or the honest
/// none-line. The UI shows exactly this string.
pub fn format_detect_status(cams: &[(String, String)]) -> String {
    if cams.is_empty() {
        "No camera detected".to_string()
    } else if cams.len() == 1 {
        format!("1 camera: {} ({})", cams[0].0, cams[0].1)
    } else {
        format!("{} cameras (using first: {}): ", cams.len(), cams[0].0)
            + &cams
                .iter()
                .map(|(m, p)| format!("{m} ({p})"))
                .collect::<Vec<_>>()
                .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn detect_returns_without_hanging_or_panicking() {
        // Graceful absence: with no hardware this completes immediately with
        // an empty vec. Only the return itself is asserted — a CI machine
        // with a virtual camera may legitimately report entries.
        let _ = detect_cameras();
    }

    #[test]
    fn camera_extension_uses_the_camera_suffix() {
        assert_eq!(camera_extension("IMG_0001.CR3"), "cr3");
        assert_eq!(camera_extension("DSCN0123.JPG"), "jpg");
        assert_eq!(camera_extension("/store/DCIM/100CANON/IMG_1.JPEG"), "jpeg");
    }

    #[test]
    fn camera_extension_falls_back_to_jpg() {
        assert_eq!(camera_extension("noext"), "jpg");
        assert_eq!(camera_extension("trailing."), "jpg");
        assert_eq!(camera_extension("toolong.abcdef"), "jpg");
        assert_eq!(camera_extension("weird.j-p"), "jpg");
    }

    #[test]
    fn unix_epoch_is_the_zero_stamp() {
        assert_eq!(
            unix_to_ymd_hms(0),
            (1970, 1, 1, 0, 0, 0),
            "the tether stem pins to UTC civil time"
        );
    }

    #[test]
    fn unix_to_ymd_hms_matches_civil_time() {
        // 2026-09-24 12:00:00 UTC, cross-checked with `date -u`.
        assert_eq!(
            unix_to_ymd_hms(1_790_251_200),
            (2026, 9, 24, 12, 0, 0)
        );
        // Pre-epoch clamps rather than wrapping.
        assert_eq!(unix_to_ymd_hms(-1), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn capture_dest_path_names_stem_plus_camera_ext() {
        let dir = Path::new("/photos/incoming");
        let now = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_790_251_200);
        let got = capture_dest_path(dir, "IMG_0001.CR3", now, &|_| false);
        assert_eq!(got, dir.join("tether-20260924-120000.cr3"));
    }

    #[test]
    fn capture_dest_path_dedupes_with_dash_n() {
        let dir = Path::new("/photos/incoming");
        let now = std::time::UNIX_EPOCH;
        let mut taken = HashSet::new();
        taken.insert(dir.join("tether-19700101-000000.jpg"));
        taken.insert(dir.join("tether-19700101-000000-2.jpg"));
        let got = capture_dest_path(dir, "IMG.JPG", now, &|p| taken.contains(p));
        assert_eq!(got, dir.join("tether-19700101-000000-3.jpg"));
    }

    #[test]
    fn tether_capture_dir_is_incoming_under_the_catalogue() {
        assert_eq!(
            tether_capture_dir(Path::new("/config/darkroom")),
            Path::new("/config/darkroom/incoming")
        );
    }

    #[test]
    fn detect_status_names_cameras_or_says_none() {
        assert_eq!(format_detect_status(&[]), "No camera detected");
        assert_eq!(
            format_detect_status(&[("Canon EOS R5".to_string(), "usb:001,002".to_string())]),
            "1 camera: Canon EOS R5 (usb:001,002)"
        );
        let two = vec![
            ("A".to_string(), "usb:1".to_string()),
            ("B".to_string(), "usb:2".to_string()),
        ];
        let line = format_detect_status(&two);
        assert!(line.starts_with("2 cameras (using first: A): "), "{line}");
        assert!(line.contains("B (usb:2)"), "{line}");
    }
}
