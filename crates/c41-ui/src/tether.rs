//! Tethering page: live camera capture plus watch-folder auto-import
//! (u6; parity audit 3.4, tethering leg).
//!
//! Live capture is real: `c41_core::camera` drives libgphoto2 (detect plus
//! capture-from-the-first-camera; no burst/timelapse/liveview). The page's
//! camera section holds a Detect button (lists attached models, or none),
//! a Capture button (always enabled — with no camera it reports a status
//! error, never a crash), and a status line for each outcome. Captures land
//! in the watch folder when one is set, else in a session `incoming` dir
//! under the catalogue dir (see [`capture_dest_dir`]), then register through
//! the single-file import ([`crate::dialogs::import_single_file_sync`]) and
//! reload the grid through the import sites' exact `on_done`.
//!
//! The file follows the established pure-model-then-widget discipline (see
//! [`crate::map`]): the diffing ([`new_files`]), the extension gate
//! ([`is_watch_candidate`]), the directory listing ([`list_watch_candidates`])
//! and the status text ([`format_watch_path`], [`format_import_status`]) are
//! GTK-free and unit-tested headless; the page widget over them is just the
//! GTK control surface.
//!
//! Import reuse: scans call [`crate::dialogs::import_folder_sync`], the exact
//! function the import dialog runs on its worker thread. Row-level dedupe
//! ([`c41_db::image::image_insert`] returning the existing id on a
//! film-plus-filename hit, and [`c41_db::film::film_new`] upserting the roll
//! by folder) means re-importing a folder never duplicates rows — so the
//! page's own `known` set exists to avoid redundant probe I/O and status
//! noise, not to prevent dupes. A scan that finds nothing new skips the
//! import call entirely, so idle watching mints no empty film rolls.
//!
//! Watching is a periodic scan every [`WATCH_INTERVAL_SECS`] seconds via
//! `glib::timeout_add_local`. The callback holds only a state `Rc` and widget
//! `WeakRef`s. Stopping (Stop button) or hiding (page pop, window minimize)
//! clears `active` and the next tick returns `Break`, removing the timer; a
//! later `map` with watching still on re-arms it, so minimize/restore resumes
//! while a popped (destroyed) page can never map again. No callback ever
//! touches a dead page: every widget goes through a `WeakRef` upgrade.

use adw::prelude::*;
use gtk4::{gio, glib};
use std::collections::HashSet;
use std::rc::Rc;

/// Navigation-page tag for the tether shell. Deliberately slash-free, so the
/// lighttable's `popped` cell re-sync (which only handles tags containing
/// `/`) and the view-switcher mirror (which lights Darkroom only for slash
/// tags) both ignore it — the same contract as the print and map pages.
pub const TETHER_PAGE_TAG: &str = "tether";

/// The camera status line before the first Detect press. Pinned by test so a
/// rewording stays deliberate; after Detect the line shows
/// [`c41_core::camera::format_detect_status`] instead.
pub const TETHER_NO_CAMERA_TEXT: &str = "No camera detected";

/// Seconds between watch-folder scans. Slow enough that the walk plus the
/// occasional import never contends with interactive use; fast enough that a
/// file dropped into the folder shows up promptly.
pub const WATCH_INTERVAL_SECS: u64 = 5;

// ── Pure watch model (GTK-free, headless-tested) ────────────────────────────

/// True when `path` names a file the importer would register: its extension
/// (case-insensitive) is in the import dialog's own list. Reuses
/// [`crate::dialogs::RAW_EXTENSIONS`] rather than a second list so the watch
/// can never disagree with a manual import about what counts as an image.
pub fn is_watch_candidate(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| crate::dialogs::RAW_EXTENSIONS.contains(&e.as_str()))
}

/// List the importable files directly inside `dir` as full path strings.
/// Depth and filtering mirror [`crate::dialogs::import_folder_sync`] (one
/// level, same extension list) so the diff below agrees with what an import
/// of this folder would actually register. Missing/unreadable dirs yield an
/// empty vec — the status line reports the import outcome, never a crash.
pub fn list_watch_candidates(dir: &str) -> Vec<String> {
    walkdir::WalkDir::new(dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_string_lossy().to_string())
        .filter(|p| is_watch_candidate(std::path::Path::new(p)))
        .collect()
}

/// Which of `current` (a fresh listing) are not in `known` (the last listing
/// already accounted for), sorted so scans and tests see a stable order. An
/// empty `known` treats everything as new — that is the Start-watching case.
pub fn new_files(known: &HashSet<String>, current: &[String]) -> Vec<String> {
    let mut out: Vec<String> = current
        .iter()
        .filter(|p| !known.contains(*p))
        .cloned()
        .collect();
    out.sort();
    out
}

/// Where a capture is saved: the watch folder when one is set, else a
/// session `incoming` dir under the catalogue dir (the parent of `db_path`).
/// With no catalogue open (empty `db_path`, demo mode) the system temp dir
/// backs the capture and the import step is skipped by the caller.
pub fn capture_dest_dir(watch_path: Option<&str>, db_path: &str) -> std::path::PathBuf {
    match watch_path {
        Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => {
            let catalogue = (!db_path.is_empty())
                .then(|| std::path::Path::new(db_path).parent())
                .flatten()
                .filter(|p| !p.as_os_str().is_empty());
            match catalogue {
                Some(dir) => c41_core::camera::tether_capture_dir(dir),
                None => std::env::temp_dir().join("c41-tether"),
            }
        }
    }
}

/// The watch-folder row text: the chosen path, or "none" before one is picked.
pub fn format_watch_path(path: Option<&str>) -> String {
    match path {
        Some(p) if !p.is_empty() => format!("Watch folder: {p}"),
        _ => "Watch folder: none".to_string(),
    }
}

/// The scan-result status line: the import count after a scan that imported,
/// "No new images" after a scan with nothing to do. Import failures use a
/// separate error line at the call site so a zero can never mean "broken".
pub fn format_import_status(new_count: usize) -> String {
    if new_count == 0 {
        "No new images".to_string()
    } else {
        format!("{new_count} new image(s) imported")
    }
}

// ── Watch state ─────────────────────────────────────────────────────────────

/// Mutable watch state shared by the buttons and the timer tick. `active` is
/// cleared on page hide (`unmap`) and restored on re-show (`map`) while
/// watching is still on; the tick checks it first and stops itself, so no
/// callback runs against a dead or hidden page. `timer_live` keeps repeated
/// Start presses from stacking timers.
struct WatchState {
    watch_path: Option<String>,
    watching: bool,
    busy: bool,
    active: bool,
    timer_live: bool,
    known: HashSet<String>,
}

impl WatchState {
    fn new() -> Self {
        Self {
            watch_path: None,
            watching: false,
            busy: false,
            active: true,
            timer_live: false,
            known: HashSet::new(),
        }
    }
}

// ── Scan driver (GTK-adjacent: spawns, never blocks) ────────────────────────

/// One scan: diff the folder against `known`, and only when something is new
/// run the shared folder import off-thread (the dialog's own shape:
/// `spawn_future_local` plus `gio::spawn_blocking`, since both the walk and
/// the probe do file I/O). The listing is recorded into `known` BEFORE the
/// import lands so the next tick cannot re-queue the same files while this
/// import is in flight; `busy` serialises overlapping imports. A quiet scan
/// writes "No new images" without touching the DB at all. The reported count
/// is the watcher's own fresh set (`fresh.len()`), not the importer's return:
/// that counts upsert/dedupe hits for the whole folder, which would overcount
/// every scan after the first. `on_done` (the lighttable reload) runs on a
/// successful import so the new rows become visible.
fn run_watch_scan(
    state: Rc<std::cell::RefCell<WatchState>>,
    db_path: String,
    status_w: glib::WeakRef<gtk4::Label>,
    on_done: Rc<dyn Fn()>,
) {
    let watch_dir = {
        let mut st = state.borrow_mut();
        if !st.active || !st.watching || st.busy {
            return;
        }
        let Some(dir) = st.watch_path.clone() else {
            return;
        };
        st.busy = true;
        dir
    };
    glib::spawn_future_local(async move {
        // Walk off-thread: directory I/O must not jank the tick.
        let current = gio::spawn_blocking({
            let watch_dir = watch_dir.clone();
            move || list_watch_candidates(&watch_dir)
        })
        .await
        .unwrap_or_default();
        let fresh = new_files(&state.borrow().known, &current);
        // A vanished/unreadable folder lists empty — say so instead of the
        // quiet-scan line, and retry next tick (busy is reset below).
        if !std::path::Path::new(&watch_dir).is_dir() {
            state.borrow_mut().busy = false;
            if let Some(lbl) = status_w.upgrade() {
                lbl.set_text("Watch folder unavailable");
            }
            return;
        }
        if fresh.is_empty() {
            state.borrow_mut().busy = false;
            if let Some(lbl) = status_w.upgrade() {
                lbl.set_text(&format_import_status(0));
            }
            return;
        }
        state.borrow_mut().known.extend(current);
        let fresh_count = fresh.len();
        let imported = gio::spawn_blocking(move || {
            crate::dialogs::import_folder_sync(&watch_dir, &db_path)
        })
        .await
        .ok()
        .flatten();
        state.borrow_mut().busy = false;
        match imported {
            Some(_) => {
                if let Some(lbl) = status_w.upgrade() {
                    lbl.set_text(&format_import_status(fresh_count));
                }
                on_done();
            }
            None => {
                if let Some(lbl) = status_w.upgrade() {
                    lbl.set_text("Watch error: import failed — see log");
                }
            }
        }
    });
}

/// Arm the periodic scan. Called at most once per live timer (guarded by
/// `timer_live`); the tick stops itself the first time it sees the page
/// inactive or unwatched.
fn arm_watch_timer(
    state: Rc<std::cell::RefCell<WatchState>>,
    db_path: String,
    status_w: glib::WeakRef<gtk4::Label>,
    on_done: Rc<dyn Fn()>,
) {
    let _ = glib::timeout_add_local(
        std::time::Duration::from_secs(WATCH_INTERVAL_SECS),
        move || {
            let still_running = {
                let st = state.borrow();
                st.active && st.watching
            };
            if !still_running {
                state.borrow_mut().timer_live = false;
                return glib::ControlFlow::Break;
            }
            run_watch_scan(state.clone(), db_path.clone(), status_w.clone(), on_done.clone());
            glib::ControlFlow::Continue
        },
    );
}

// ── Page widget ─────────────────────────────────────────────────────────────

/// Build the tethering shell page: the honest no-camera status plus the
/// watch-folder section (picker row, Start/Stop toggle, scan status line).
/// `on_done` runs after every successful auto-import (the lighttable reload
/// the import dialog's callers pass) so new rows become visible; it also runs
/// when the page is already gone — reloading the grid the user is looking at.
pub fn tether_page(db_path: String, on_done: Rc<dyn Fn()>) -> adw::NavigationPage {
    let state = Rc::new(std::cell::RefCell::new(WatchState::new()));

    let content = gtk4::Box::new(gtk4::Orientation::Vertical, 12);
    content.set_margin_start(12);
    content.set_margin_end(12);
    content.set_margin_top(12);
    content.set_margin_bottom(12);

    // Camera status: live detect/capture state (u6). Starts at the honest
    // none-line; Detect refreshes it from `format_detect_status`, Capture
    // reports each outcome here.
    let camera = gtk4::Label::new(Some(TETHER_NO_CAMERA_TEXT));
    camera.set_wrap(true);
    camera.set_halign(gtk4::Align::Start);
    camera.add_css_class("dim-label");
    content.append(&camera);

    // Detect + Capture row. Capture stays enabled with no camera — the
    // handler reports a status error instead of crashing.
    let cam_row = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
    let detect_btn = gtk4::Button::with_label("Detect camera");
    let capture_btn = gtk4::Button::with_label("Capture");
    cam_row.append(&detect_btn);
    cam_row.append(&capture_btn);
    content.append(&cam_row);

    {
        let status_w = camera.downgrade();
        let btn_w = detect_btn.downgrade();
        detect_btn.connect_clicked(move |_| {
            if let Some(b) = btn_w.upgrade() {
                b.set_sensitive(false);
            }
            if let Some(lbl) = status_w.upgrade() {
                lbl.set_text("Detecting…");
            }
            let status_w = status_w.clone();
            let btn_w = btn_w.clone();
            glib::spawn_future_local(async move {
                let cams = gio::spawn_blocking(c41_core::camera::detect_cameras)
                    .await
                    .unwrap_or_default();
                if let Some(lbl) = status_w.upgrade() {
                    lbl.set_text(&c41_core::camera::format_detect_status(&cams));
                }
                if let Some(b) = btn_w.upgrade() {
                    b.set_sensitive(true);
                }
            });
        });
    }

    {
        let state = state.clone();
        let db = db_path.clone();
        let done = on_done.clone();
        let status_w = camera.downgrade();
        let busy = Rc::new(std::cell::Cell::new(false));
        capture_btn.connect_clicked(move |btn| {
            if busy.get() {
                return;
            }
            busy.set(true);
            btn.set_sensitive(false);
            if let Some(lbl) = status_w.upgrade() {
                lbl.set_text("Capturing…");
            }
            let watch = state.borrow().watch_path.clone();
            let dest_dir = capture_dest_dir(watch.as_deref(), &db);
            let db_inner = db.clone();
            let done_inner = done.clone();
            let status_w = status_w.clone();
            let busy_inner = busy.clone();
            let btn_w = btn.downgrade();
            // Cloned up front: the async block below is `move`, and `state`
            // belongs to the `Fn` button closure that outlives it.
            let state_inner = state.clone();
            glib::spawn_future_local(async move {
                // USB I/O off-thread; the join error (panicked worker) is a
                // status line, never a crash.
                let outcome = gio::spawn_blocking(move || {
                    c41_core::camera::capture_into(&dest_dir)
                })
                .await;
                let release = |text: &str| {
                    if let Some(lbl) = status_w.upgrade() {
                        lbl.set_text(text);
                    }
                    busy_inner.set(false);
                    if let Some(b) = btn_w.upgrade() {
                        b.set_sensitive(true);
                    }
                };
                let saved = match outcome {
                    Ok(Ok(path)) => Some(path),
                    Ok(Err(e)) => {
                        release(&format!("Capture failed: {e}"));
                        None
                    }
                    Err(_) => {
                        release("Capture failed: capture task did not complete");
                        None
                    }
                };
                let Some(path) = saved else { return };
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let dir = path
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                if db_inner.is_empty() {
                    release(&format!("Captured {name} (no catalogue open — not imported)"));
                    return;
                }
                // Single-file import, off-thread like the watch scans: only
                // this file registers, never a re-walk of the whole folder.
                // Cloned for the worker: `name` is still needed below for the
                // status line and the watch-set insert.
                let name_inner = name.clone();
                let imported = gio::spawn_blocking(move || {
                    crate::dialogs::import_single_file_sync(&dir, &name_inner, &db_inner)
                })
                .await
                .unwrap_or(false);
                if imported {
                    // Teach the watch set about the capture: without this the
                    // next tick would re-report the just-captured file as new
                    // (the DB upsert dedupes the row, but the status line
                    // would lie). Same spelling the walk yields — dest_dir
                    // joined with the saved name — so the set agrees.
                    // Cloned: this async block is `move` and `state` belongs
                    // to the `Fn` button closure that outlives it.
                    state_inner.borrow_mut().known.insert(path.to_string_lossy().to_string());
                    release(&format!("Captured {name} — imported"));
                    done_inner();
                } else {
                    release(&format!("Captured {name} but the import failed — see log"));
                }
            });
        });
    }

    content.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

    let section = gtk4::Label::new(Some("Watch folder"));
    section.set_halign(gtk4::Align::Start);
    content.append(&section);

    let hint = gtk4::Label::new(Some(
        "New images appearing in the watched folder are imported automatically.",
    ));
    hint.set_wrap(true);
    hint.set_halign(gtk4::Align::Start);
    hint.add_css_class("dim-label");
    content.append(&hint);

    // Folder picker row: current path (or none) plus the chooser button.
    let picker_row = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
    let path_lbl = gtk4::Label::new(Some(&format_watch_path(None)));
    path_lbl.set_halign(gtk4::Align::Start);
    path_lbl.set_hexpand(true);
    path_lbl.set_ellipsize(gtk4::pango::EllipsizeMode::Middle);
    picker_row.append(&path_lbl);
    let choose_btn = gtk4::Button::with_label("Choose folder…");
    picker_row.append(&choose_btn);
    content.append(&picker_row);

    // Start/Stop toggle plus the last-scan status line.
    let watch_btn = gtk4::Button::with_label("Start watching");
    content.append(&watch_btn);
    let status_lbl = gtk4::Label::new(Some("Not watching"));
    status_lbl.set_halign(gtk4::Align::Start);
    status_lbl.add_css_class("dim-label");
    content.append(&status_lbl);

    {
        let state = state.clone();
        let path_w = path_lbl.downgrade();
        let status_w = status_lbl.downgrade();
        choose_btn.connect_clicked(move |_| {
            let dialog = gtk4::FileDialog::builder()
                .title("Choose watch folder")
                .build();
            let state = state.clone();
            // Clone the weak refs per activation: the inner callback is
            // `move` and this outer closure is `Fn` (every click reuses it).
            let path_w = path_w.clone();
            let status_w = status_w.clone();
            dialog.select_folder(
                None::<&gtk4::Window>,
                gtk4::gio::Cancellable::NONE,
                move |result| {
                    let folder = match result {
                        Ok(f) => f,
                        Err(_) => return,
                    };
                    let Some(p) = folder.path() else { return };
                    let dir = p.to_string_lossy().to_string();
                    let mut st = state.borrow_mut();
                    st.watch_path = Some(dir.clone());
                    // A new folder rescans fully: seeds nothing, so the next
                    // tick treats everything present as new. The import
                    // upserts the roll and dedupes per filename, and the
                    // known set stops re-imports, so re-picking a folder
                    // never duplicates rows.
                    st.known.clear();
                    if let Some(lbl) = path_w.upgrade() {
                        lbl.set_text(&format_watch_path(Some(&dir)));
                    }
                    if st.watching {
                        if let Some(lbl) = status_w.upgrade() {
                            lbl.set_text("Watching — scanning…");
                        }
                    }
                },
            );
        });
    }

    // Weak ref for the map/unmap pair below: `status_lbl` itself moves into
    // the Start/Stop handler next, so take this before that move.
    let status_w_page = status_lbl.downgrade();

    {
        let state = state.clone();
        let status_w = status_lbl.downgrade();
        let db = db_path.clone();
        let done = on_done.clone();
        watch_btn.connect_clicked(move |btn| {
            let mut st = state.borrow_mut();
            if st.watching {
                st.watching = false;
                btn.set_label("Start watching");
                status_lbl.set_text("Not watching");
                return;
            }
            if st.watch_path.is_none() {
                status_lbl.set_text("Choose a watch folder first");
                return;
            }
            st.watching = true;
            // First scan imports everything already present — the folder's
            // existing contents are new to the watcher by definition.
            st.known.clear();
            btn.set_label("Stop watching");
            status_lbl.set_text("Watching — scanning…");
            if !st.timer_live {
                st.timer_live = true;
                drop(st);
                arm_watch_timer(state.clone(), db.clone(), status_w.clone(), done.clone());
                // Run the first scan now rather than after one idle interval.
                run_watch_scan(state.clone(), db.clone(), status_w.clone(), done.clone());
            }
        });
    }

    let page = adw::NavigationPage::builder()
        .title("Tether")
        .tag(TETHER_PAGE_TAG)
        .child(&content)
        .build();
    // Hide/show drives the watch timer, not just popping: `unmap` fires on
    // minimize/hide as well as pop, so it only parks the timer (the tick
    // self-removes); `map` re-arms it while watching is still on. A popped
    // page is destroyed and never maps again, so reaching `map` with
    // watching on always means a transient hide (minimize, or another page
    // pushed on top) — never a resurrection. Every widget touch goes through
    // a WeakRef, so nothing here can call into a dead page.
    {
        let state = state.clone();
        let db = db_path.clone();
        let done = on_done.clone();
        // `status_w_page`, taken before `status_lbl` moved into the
        // Start/Stop handler above.
        let status_w = status_w_page.clone();
        // Each closure takes its own Rc clone: both are `Fn` (map/unmap can
        // fire repeatedly) and Rc is not Copy.
        let unmap_state = state.clone();
        let map_state = state.clone();
        page.connect_unmap(move |_| {
            unmap_state.borrow_mut().active = false;
        });
        page.connect_map(move |_| {
            let should_arm = {
                let mut st = map_state.borrow_mut();
                if st.watching && !st.timer_live {
                    st.active = true;
                    st.timer_live = true;
                    true
                } else {
                    false
                }
            };
            if should_arm {
                arm_watch_timer(state.clone(), db.clone(), status_w.clone(), done.clone());
            }
        });
    }
    page
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tether_tag_is_slash_free() {
        // Same contract as the print and map tags: the popped re-sync and the
        // switcher mirror only handle slash tags (image paths), so an
        // auxiliary tag containing `/` would be mistaken for a darkroom page.
        assert!(!TETHER_PAGE_TAG.contains('/'), "tag {TETHER_PAGE_TAG:?}");
    }

    #[test]
    fn no_camera_line_is_exact() {
        // The honest starting state: pin the wording so it can only change
        // deliberately, never by an idle edit. Detect refreshes the line
        // from `c41_core::camera::format_detect_status`.
        assert_eq!(TETHER_NO_CAMERA_TEXT, "No camera detected");
    }

    #[test]
    fn watch_path_names_the_folder_or_none() {
        assert_eq!(format_watch_path(None), "Watch folder: none");
        assert_eq!(format_watch_path(Some("")), "Watch folder: none");
        assert_eq!(
            format_watch_path(Some("/photos/incoming")),
            "Watch folder: /photos/incoming"
        );
    }

    #[test]
    fn new_files_returns_only_unknown_sorted() {
        let known: HashSet<String> =
            ["/w/b.dng", "/w/c.dng"].into_iter().map(str::to_string).collect();
        let current = ["/w/c.dng", "/w/a.dng", "/w/b.dng"].map(str::to_string);
        assert_eq!(new_files(&known, &current), vec!["/w/a.dng".to_string()]);
    }

    #[test]
    fn new_files_empty_known_treats_all_as_new() {
        // The Start-watching case: everything present is new to the watcher.
        let known = HashSet::new();
        let current = ["/w/b.dng", "/w/a.dng"].map(str::to_string);
        assert_eq!(
            new_files(&known, &current),
            vec!["/w/a.dng".to_string(), "/w/b.dng".to_string()]
        );
    }

    #[test]
    fn new_files_all_known_is_empty() {
        // The idle-tick case: nothing to do, and the caller skips the import
        // entirely so no empty film roll is minted.
        let known: HashSet<String> =
            ["/w/a.dng"].into_iter().map(str::to_string).collect();
        let current = ["/w/a.dng".to_string()];
        assert!(new_files(&known, &current).is_empty());
    }

    #[test]
    fn import_status_names_the_count_or_the_quiet() {
        assert_eq!(format_import_status(0), "No new images");
        assert_eq!(format_import_status(1), "1 new image(s) imported");
        assert_eq!(format_import_status(4), "4 new image(s) imported");
    }

    #[test]
    fn watch_candidates_match_the_importer_list() {
        // Every extension the import dialog accepts passes the watch gate in
        // both cases; non-image names never do. The first loop iterates the
        // importer's own list so the two cannot drift apart silently.
        for ext in crate::dialogs::RAW_EXTENSIONS {
            for name in [format!("img.{ext}"), format!("IMG.{ext}", ext = ext.to_ascii_uppercase())] {
                assert!(
                    is_watch_candidate(std::path::Path::new(&name)),
                    "{name} should be a watch candidate"
                );
            }
        }
        for name in ["notes.txt", "edit.xmp", "noext", "", "img.jpeeg"] {
            assert!(
                !is_watch_candidate(std::path::Path::new(name)),
                "{name} must not be a watch candidate"
            );
        }
    }

    #[test]
    fn watch_listing_lists_only_top_level_images() {
        // Temp dir: two top-level images (one upper-case), a text file, and a
        // nested image. The nested one must NOT appear — the importer walks
        // one level, and the watch diff must agree with what an import of
        // this folder would register.
        let base = std::env::temp_dir().join(format!(
            "c41_tether_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let sub = base.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(base.join("a.dng"), b"").unwrap();
        std::fs::write(base.join("B.NEF"), b"").unwrap();
        std::fs::write(base.join("notes.txt"), b"").unwrap();
        std::fs::write(sub.join("nested.dng"), b"").unwrap();

        let mut got = list_watch_candidates(base.to_str().unwrap());
        got.sort();
        let mut want = vec![
            base.join("B.NEF").to_string_lossy().to_string(),
            base.join("a.dng").to_string_lossy().to_string(),
        ];
        want.sort();
        assert_eq!(got, want);

        // A missing dir is an empty listing, not an error.
        assert!(list_watch_candidates(base.join("absent").to_str().unwrap()).is_empty());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn capture_dest_prefers_the_watch_folder() {
        // Watch folder set: captures land there whatever the catalogue is.
        assert_eq!(
            capture_dest_dir(Some("/photos/watch"), "/config/darkroom/library.db"),
            std::path::PathBuf::from("/photos/watch")
        );
        assert_eq!(
            capture_dest_dir(Some("/photos/watch"), ""),
            std::path::PathBuf::from("/photos/watch")
        );
    }

    #[test]
    fn capture_dest_falls_back_to_catalogue_incoming() {
        // No watch folder: a session `incoming` dir under the catalogue dir.
        assert_eq!(
            capture_dest_dir(None, "/config/darkroom/library.db"),
            std::path::PathBuf::from("/config/darkroom/incoming")
        );
        assert_eq!(
            capture_dest_dir(Some(""), "/config/darkroom/library.db"),
            std::path::PathBuf::from("/config/darkroom/incoming")
        );
    }

    #[test]
    fn capture_dest_without_catalogue_is_temp() {
        // Demo mode (no catalogue): temp-backed so the capture still lands
        // somewhere; the caller skips the import there.
        assert_eq!(
            capture_dest_dir(None, ""),
            std::env::temp_dir().join("c41-tether")
        );
    }

    #[test]
    fn watch_interval_is_a_sane_poll() {
        // Documents the "periodic scan" contract: slow enough not to contend
        // with interactive use. Pinned exact so a change stays deliberate.
        assert_eq!(WATCH_INTERVAL_SECS, 5);
    }
}
