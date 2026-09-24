//! Tethering shell, watch-folder slice only (u4; parity audit 3.4, tethering leg).
//!
//! Scope is deliberately narrow: there is NO live camera capture here.
//! libgphoto2 is absent from the shipping image and there is no Rust binding
//! for it, so live capture stays BLOCKED (like slippy-map tiles and neural
//! restore). The page says exactly that in its status area rather than
//! showing a fake "camera connected" state. What the page DOES do is watch a
//! user-chosen folder and auto-import new images through the SAME folder
//! import the import dialog uses.
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

/// The honest no-camera line shown in the page's status area. Pinned by test
/// so a rewording stays deliberate: the shell must never imply a camera path
/// exists in this build.
pub const TETHER_NO_CAMERA_TEXT: &str =
    "No camera detected — live capture needs libgphoto2, which this build does not ship";

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

    // Camera status: the honest line — no capture path in this build.
    let camera = gtk4::Label::new(Some(TETHER_NO_CAMERA_TEXT));
    camera.set_wrap(true);
    camera.set_halign(gtk4::Align::Start);
    camera.add_css_class("dim-label");
    content.append(&camera);

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
        // The shell's honesty is the feature: pin the wording so it can only
        // change deliberately, never by an idle edit.
        assert_eq!(
            TETHER_NO_CAMERA_TEXT,
            "No camera detected — live capture needs libgphoto2, which this build does not ship"
        );
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
    fn watch_interval_is_a_sane_poll() {
        // Documents the "periodic scan" contract: slow enough not to contend
        // with interactive use. Pinned exact so a change stays deliberate.
        assert_eq!(WATCH_INTERVAL_SECS, 5);
    }
}
