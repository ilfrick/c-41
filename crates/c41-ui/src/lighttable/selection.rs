//! Multi-image selection (m4-144): darktable's lighttable click model — more
//! than one image selectable at once, actions fan out over all of them. C-41's
//! grid runs on a SingleSelection whose cursor drives preview, darkroom entry
//! and the metadata panel; swapping it for a multi-selection model would churn
//! every consumer (culling window swap, preview stepping, panel binding). So —
//! like upstream's active-image-vs-selection split — the cursor stays what it
//! was, and the broader set lives here as PATH-keyed session state.
//!
//! Why paths, not indices: indices shift when the culling window slides or the
//! collection reloads; paths are the identity everything else already stamps on
//! widgets (thumb widget_names, repaint-by-path helpers), they survive model
//! swaps, and placeholders never carry one (no `/`).
//!
//! Interaction map (darktable parity):
//!   plain click -> select only that image (replaces the set)
//!   ctrl-click  -> toggle that image, keeping the rest
//!   shift-click -> range from the anchor (last clicked image), inclusive,
//!                  either direction
//!   ctrl+A      -> select the whole collection; Escape clears
//!
//! Main-thread only, like every piece of session state here. The bottom-bar
//! count label, the grid and the zoomable canvas each register themselves once
//! via `set_count_label`/`set_grid`/`set_canvas`; after mutating, callers run
//! [`notify_changed`], which repaints realized cells' frames, redraws the
//! canvas and refreshes the count.

use gtk4::glib;
use gtk4::prelude::*;
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    /// The selected paths. Deliberately NOT pruned on every mutation: entries
    /// for images no longer in the collection are inert (no cell shows them)
    /// and are swept by `prune_to` in lighttable::fill_grid — the one place
    /// every collection reload funnels through.
    static SELECTED: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    /// Where a shift-range starts: the last plainly/ctrl-clicked image. Cleared
    /// when it stops resolving against the model (`range_from_anchor`).
    static ANCHOR: RefCell<Option<String>> = RefCell::new(None);
    /// Bottom-bar count label, registered once at build time by lib.rs.
    static COUNT_LABEL: RefCell<Option<glib::WeakRef<gtk4::Label>>> =
        const { RefCell::new(None) };
    /// The grid, registered once at build time: lets one-parameter-free
    /// [`notify_changed`] repaint realized cells' frames from ANY click site
    /// (cell gestures, canvas clicks, keyboard shortcuts).
    static GRID_W: RefCell<Option<glib::WeakRef<gtk4::GridView>>> =
        const { RefCell::new(None) };
    /// The zoomable canvas, registered once at build time. Its paint reads the
    /// shared set lazily, so set-only mutations must queue it a redraw — see
    /// [`notify_changed`].
    static CANVAS_W: RefCell<Option<glib::WeakRef<gtk4::DrawingArea>>> =
        const { RefCell::new(None) };
}

// ── Registration ────────────────────────────────────────────────────────────

/// Register the bottom-bar count label (called once from lib.rs).
pub(crate) fn set_count_label(label: &gtk4::Label) {
    COUNT_LABEL.with(|c| {
        *c.borrow_mut() = Some(label.downgrade());
    });
}

/// Register the grid (called once from lighttable_page, right after build).
pub(crate) fn set_grid(grid: &gtk4::GridView) {
    GRID_W.with(|g| {
        *g.borrow_mut() = Some(grid.downgrade());
    });
}

/// Register the zoomable canvas (called once from lighttable_page, right after
/// the canvas is built): [`notify_changed`] queue_draws it so frames painted
/// from the shared set can't go stale across set-only mutations.
pub(crate) fn set_canvas(area: &gtk4::DrawingArea) {
    CANVAS_W.with(|c| {
        *c.borrow_mut() = Some(area.downgrade());
    });
}

/// The CURRENT view's rows as raw paths (placeholders included): what the
/// grid's selection model holds right now — under culling that is the window,
/// matching what the user can see and click. Empty when the grid is gone.
pub(crate) fn current_view_paths() -> Vec<String> {
    GRID_W.with(|g| g.borrow().as_ref().and_then(glib::WeakRef::upgrade))
        .and_then(|grid| {
            let sel = grid.model()?.downcast::<gtk4::SingleSelection>().ok()?;
            sel.model().map(|m| super::model_paths(&m))
        })
        .unwrap_or_default()
}

/// One entry point after ANY selection change: repaint the realized cells'
/// frames and refresh the bottom-bar count. Parameter-free so every mutation
/// site (cell gesture, canvas click, key handler, reload sweep) can end with
/// exactly this call.
pub(crate) fn notify_changed() {
    if let Some(grid) = GRID_W.with(|g| g.borrow().as_ref().and_then(glib::WeakRef::upgrade)) {
        super::refresh_selection_frames(&grid);
    }
    // The zoomable canvas draws its frames lazily from the shared set; a
    // mutation that never touches the grid (Esc, ctrl+A, a click in the other
    // surface) must still queue it a redraw or stale frames linger until the
    // next scroll/zoom/click (senior review MINOR-6).
    if let Some(area) = CANVAS_W.with(|c| c.borrow().as_ref().and_then(glib::WeakRef::upgrade)) {
        area.queue_draw();
    }
    sync_count_label();
}

// ── Queries ─────────────────────────────────────────────────────────────────

/// Is `path` currently selected?
pub(crate) fn contains(path: &str) -> bool {
    SELECTED.with(|s| s.borrow().contains(path))
}

/// How many images are selected right now.
pub(crate) fn selected_count() -> usize {
    SELECTED.with(|s| s.borrow().len())
}

/// Snapshot of the selected paths, sorted for deterministic action order (the
/// actions slice iterates this; DB writes are serialised downstream anyway).
pub(crate) fn paths_snapshot() -> Vec<String> {
    let mut v: Vec<String> = SELECTED.with(|s| s.borrow().iter().cloned().collect());
    v.sort();
    v
}

// ── Mutations ───────────────────────────────────────────────────────────────

/// Plain click: select ONLY `path`, anchored there.
pub(crate) fn select_exclusive(path: &str) {
    SELECTED.with(|s| {
        s.borrow_mut().clear();
        s.borrow_mut().insert(path.to_string());
    });
    ANCHOR.with(|a| *a.borrow_mut() = Some(path.to_string()));
}

/// Ctrl-click: toggle `path` in/out, anchored there (darktable re-anchors on
/// every click so the next shift-range starts where you last clicked).
pub(crate) fn toggle(path: &str) {
    SELECTED.with(|s| {
        if !s.borrow_mut().remove(path) {
            s.borrow_mut().insert(path.to_string());
        }
    });
    ANCHOR.with(|a| *a.borrow_mut() = Some(path.to_string()));
}

/// Ctrl+A over `model_paths` (raw strings, placeholders included): keep only
/// real image paths, select all of them.
pub(crate) fn select_all(model_paths: &[String]) {
    SELECTED.with(|s| {
        let mut sel = s.borrow_mut();
        sel.clear();
        sel.extend(real_paths(model_paths));
    });
    ANCHOR.take();
}

/// Escape / collection replaced: drop everything, including the anchor.
pub(crate) fn clear() {
    SELECTED.with(|s| s.borrow_mut().clear());
    ANCHOR.take();
}

/// Shift-click helper: replace the selection with the inclusive span between
/// ANCHOR and `target`, both resolved against `model_paths` (the CURRENT view's
/// rows, placeholders included — filtered out here). No anchor, or an anchor
/// that no longer resolves, degrades to selecting only the target. Returns the
/// new contents so callers can pin behaviour in tests.
pub(crate) fn range_from_anchor(model_paths: &[String], target: &str) -> Vec<String> {
    let real = real_paths(model_paths);
    let tgt_idx = real.iter().position(|p| p == target);
    let Some(tgt_idx) = tgt_idx else {
        return Vec::new(); // target is not a row here (stale event); change nothing
    };
    let anchor_idx = ANCHOR.with(|a| {
        a.borrow()
            .as_ref()
            .and_then(|ap| real.iter().position(|p| p == ap))
    });
    let picked: Vec<String> = match anchor_idx {
        Some(a) => {
            let span = span_indices(a, tgt_idx);
            real[span].to_vec()
        }
        None => vec![target.to_string()],
    };
    SELECTED.with(|s| {
        let mut sel = s.borrow_mut();
        sel.clear();
        sel.extend(picked.iter().cloned());
    });
    picked
}

/// Reload sweep: intersect with `keep` (the new collection's real paths).
/// Stale entries would be inert anyway; pruning keeps the count label honest.
pub(crate) fn prune_to(keep: &[String]) {
    let keep: HashSet<&String> = keep.iter().collect();
    SELECTED.with(|s| {
        s.borrow_mut().retain(|p| keep.contains(p));
    });
}

// ── Count label ─────────────────────────────────────────────────────────────

/// Refresh the bottom-bar count label from the live set. Hidden when nothing
/// is selected — the bar must look exactly as before when the feature is idle.
pub(crate) fn sync_count_label() {
    let n = selected_count();
    COUNT_LABEL.with(|c| {
        if let Some(label) = c.borrow().as_ref().and_then(glib::WeakRef::upgrade) {
            match n {
                0 => {
                    label.set_visible(false);
                }
                1 => {
                    label.set_text("1 image selected");
                    label.set_visible(true);
                }
                k => {
                    label.set_text(&format!("{k} images selected"));
                    label.set_visible(true);
                }
            }
        }
    });
}

// ── Pure helpers (unit-tested below) ────────────────────────────────────────

/// Keep only entries that contain `/` — placeholders never select.
fn real_paths(paths: &[String]) -> Vec<String> {
    paths.iter().filter(|p| p.contains('/')).cloned().collect()
}

/// Inclusive index span between `a` and `b`, either direction. Pure.
fn span_indices(a: usize, b: usize) -> std::ops::RangeInclusive<usize> {
    a.min(b)..=a.max(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Under --test-threads=1 every test shares one thread, so leave the state
    /// as found on every exit path; unique paths also keep the tests correct
    /// under parallel threads (each thread gets its own thread-locals).
    #[test]
    fn exclusive_replaces_and_anchors() {
        clear();
        select_exclusive("/d/a.nef");
        assert!(contains("/d/a.nef"));
        assert_eq!(selected_count(), 1);
        select_exclusive("/d/b.nef");
        assert!(!contains("/d/a.nef"), "plain click replaces");
        assert!(contains("/d/b.nef"));
        clear();
    }

    #[test]
    fn select_all_clears_the_anchor_so_shift_degrades_to_target() {
        // Ctrl+A deliberately drops the anchor (pinned per review NIT): a
        // shift-click afterwards selects only the clicked target instead of
        // silently spanning from some pre-select-all click point.
        clear();
        let model = vec!["/sa/0.nef".to_string(), "/sa/1.nef".to_string()];
        select_exclusive("/sa/0.nef");
        select_all(&model);
        assert_eq!(selected_count(), 2);
        assert_eq!(range_from_anchor(&model, "/sa/1.nef"), vec!["/sa/1.nef"]);
        clear();
    }

    #[test]
    fn toggle_keeps_the_rest() {
        clear();
        select_exclusive("/t/a.nef");
        toggle("/t/b.nef");
        assert!(contains("/t/a.nef") && contains("/t/b.nef"));
        toggle("/t/a.nef");
        assert!(!contains("/t/a.nef") && contains("/t/b.nef"));
        clear();
    }

    #[test]
    fn select_all_ignores_placeholders() {
        clear();
        let model = vec![
            "/s/one.nef".to_string(),
            "sentinel".to_string(), // placeholder rows carry no '/'
            "/s/two.nef".to_string(),
        ];
        select_all(&model);
        assert_eq!(selected_count(), 2);
        assert!(!contains("sentinel"));
        clear();
    }

    #[test]
    fn shift_range_spans_both_directions_from_the_anchor() {
        clear();
        let model: Vec<String> = (0..6)
            .map(|i| format!("/r/{i}.nef"))
            .collect();
        select_exclusive("/r/1.nef"); // plain click anchors at index 1
        let got = range_from_anchor(&model, "/r/4.nef"); // shift-click forward
        assert_eq!(got, vec!["/r/1.nef", "/r/2.nef", "/r/3.nef", "/r/4.nef"]);
        // Re-anchor backwards, then shift back up past the old anchor.
        select_exclusive("/r/4.nef");
        let got = range_from_anchor(&model, "/r/0.nef"); // shift-click backward
        assert_eq!(
            got,
            vec!["/r/0.nef", "/r/1.nef", "/r/2.nef", "/r/3.nef", "/r/4.nef"]
        );
        clear();
    }

    #[test]
    fn stale_anchor_degrades_to_target_only() {
        clear();
        let model: Vec<String> = (0..4).map(|i| format!("/g/{i}.nef")).collect();
        // No anchor at all yet:
        let got = range_from_anchor(&model, "/g/2.nef");
        assert_eq!(got, vec!["/g/2.nef"]);
        // Anchor that is NOT in this model (e.g. scrolled into another window):
        select_exclusive("/elsewhere/x.nef");
        let got = range_from_anchor(&model, "/g/0.nef");
        assert_eq!(got, vec!["/g/0.nef"]);
        clear();
    }

    #[test]
    fn unknown_target_changes_nothing() {
        clear();
        select_exclusive("/u/a.nef");
        let model = vec!["/u/a.nef".to_string()];
        let got = range_from_anchor(&model, "/not/in/model");
        assert!(got.is_empty());
        assert!(contains("/u/a.nef"), "selection untouched");
        clear();
    }

    #[test]
    fn range_filters_placeholders_inside_the_span() {
        clear();
        let model = vec![
            "/p/0.nef".to_string(),
            "sentinel".to_string(), // lands INSIDE the span below
            "/p/2.nef".to_string(),
        ];
        select_exclusive("/p/0.nef");
        let got = range_from_anchor(&model, "/p/2.nef");
        assert_eq!(got, vec!["/p/0.nef", "/p/2.nef"], "placeholder skipped");
        clear();
    }

    #[test]
    fn prune_intersects_with_the_new_collection() {
        clear();
        select_exclusive("/z/keep.nef");
        toggle("/z/gone.nef");
        prune_to(&["/z/keep.nef".to_string()]);
        assert!(contains("/z/keep.nef"));
        assert!(!contains("/z/gone.nef"));
        clear();
    }

    #[test]
    fn span_indices_is_inclusive_either_direction() {
        assert_eq!(span_indices(2, 5), 2..=5);
        assert_eq!(span_indices(5, 2), 2..=5);
        assert_eq!(span_indices(3, 3), 3..=3);
    }

    #[test]
    fn snapshot_is_sorted_for_deterministic_fanout() {
        clear();
        toggle("/q/c.nef");
        toggle("/q/a.nef");
        assert_eq!(paths_snapshot(), vec!["/q/a.nef", "/q/c.nef"]);
        clear();
    }
}
