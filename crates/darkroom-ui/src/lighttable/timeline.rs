//! The lighttable **date timeline** (Phase 3 m4-99) — our port of darktable's
//! bottom date-histogram strip: one bar per year sized by how many images were
//! taken in it, with year labels, and click-to-filter-by-year.
//!
//! ## Date storage
//!
//! darktable 4.x stores `images.datetime_taken` as **microseconds since
//! 0001-01-01** (a GLib `GDateTime`/`GTimeSpan` origin), not a Unix timestamp.
//! [`DT_EPOCH_OFFSET_SECS`] converts to Unix seconds; it is the exact proleptic
//! Gregorian day count from 0001-01-01 to 1970-01-01 (719162 × 86400), so the
//! observation that a year-0 origin reads a year early is a consequence, not a
//! coincidence. Cross-checked against the real catalog: `P7280008.ORF` in the
//! film roll `…/2018_07_28` decodes to 2018-07-28 22:07:53.
//!
//! Decoding uses SQLite's `'unixepoch'` — i.e. **UTC** — and that is correct, for
//! a non-obvious reason worth not "fixing": darktable builds the stored value
//! with `g_date_time_new_utc()` from the EXIF *wall clock*, so the stored µs are
//! wall-clock-as-UTC and decoding as UTC round-trips to the original EXIF time.
//! `'localtime'` would shift by the *viewer's* offset, making an image's year
//! depend on `TZ` and pushing e.g. a 00:30 New Year photo into the previous year.
//!
//! `datetime_taken` is `0`/NULL for images with no EXIF date; those are excluded
//! from the histogram *and* from a year filter (they have no year to belong to),
//! matching how [`crate::lighttable::SortOrder::DateTaken`] sorts them last.
//!
//! ## One expression, two uses
//!
//! [`year_sql_expr`] is the single source of truth for "which year is this row
//! in" — the histogram groups by it and the filter compares against it, so the
//! bar you click can't disagree with the rows you get. It's built from a
//! `const`-derived offset and a caller-supplied column name (never user text),
//! so it's injection-safe.

/// Seconds from darktable's `datetime_taken` origin (0001-01-01, the GLib
/// `GDateTime` epoch) to the Unix epoch (1970-01-01) — 719162 days.
pub(crate) const DT_EPOCH_OFFSET_SECS: i64 = 62_135_596_800;

/// SQL integer expression yielding the 4-digit year of a `datetime_taken` column
/// (µs since 0001-01-01), via SQLite's own date functions. `col` is a literal
/// column reference from our own code (e.g. `"i.datetime_taken"`), never user
/// input.
pub(crate) fn year_sql_expr(col: &str) -> String {
    format!(
        "CAST(strftime('%Y', ({col} / 1000000 - {off}), 'unixepoch') AS INTEGER)",
        off = DT_EPOCH_OFFSET_SECS,
    )
}

/// A trailing ` AND (…)` fragment restricting rows to the inclusive year range,
/// or `""` when no range is selected. Undated rows (`datetime_taken <= 0`) are
/// excluded — they belong to no year, and SQLite would otherwise decode 0 as year
/// 1 and sweep them into an early bucket. Years are `i32`s formatted as integers,
/// so nothing user-typed reaches the SQL.
pub(crate) fn year_range_and(range: Option<(i32, i32)>) -> String {
    match range {
        None => String::new(),
        Some((lo, hi)) => {
            // Tolerate an inverted range (a right-to-left drag) rather than
            // silently matching nothing.
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            format!(
                " AND i.datetime_taken > 0 AND {expr} BETWEEN {lo} AND {hi}",
                expr = year_sql_expr("i.datetime_taken"),
            )
        }
    }
}

/// SQL for the per-year image counts backing the histogram. Deliberately **not**
/// filtered by the active collection or the other quick-filters: the timeline is
/// a stable map of the whole catalog (as darktable's is), so filtering by a year
/// can't shrink the very bars you're navigating with.
pub(crate) fn histogram_sql() -> String {
    format!(
        "SELECT {expr} AS yr, COUNT(*) FROM main.images \
         WHERE datetime_taken > 0 GROUP BY yr ORDER BY yr",
        expr = year_sql_expr("datetime_taken"),
    )
}

/// Which bar index a click at `x` lands on, for `n` equal-width bars across
/// `width` pixels. `None` for a click outside the strip, an empty histogram, or a
/// non-positive width (a not-yet-allocated widget). Pure, so the hit-testing is
/// unit-testable under the display-free discipline.
pub(crate) fn bar_at_x(x: f64, width: f64, n: usize) -> Option<usize> {
    if n == 0 || width <= 0.0 || x < 0.0 || x >= width {
        return None;
    }
    // `min(n-1)` guards the exact-right-edge float case.
    Some(((x / width * n as f64) as usize).min(n - 1))
}

/// Inclusive bar-index span covered by a drag from `x0` to `x1` (m4-99b). Both
/// endpoints are **clamped** into the strip rather than rejected — a drag that
/// runs off the edge should select out to that edge, which is what the gesture
/// naturally produces. Returns `(lo, hi)` with `lo <= hi`, so a right-to-left
/// drag selects the same span as a left-to-right one. `None` only for an empty
/// histogram or an unallocated widget. Pure.
pub(crate) fn bar_span(x0: f64, x1: f64, width: f64, n: usize) -> Option<(usize, usize)> {
    if n == 0 || width <= 0.0 {
        return None;
    }
    // Clamp in INDEX space, not coordinate space. Nudging the coordinate below
    // `width` needs an epsilon that is wrong at both ends: too small to matter for
    // huge widths (the subtraction rounds away and a past-the-edge drag silently
    // does nothing) and larger than the width itself for tiny ones (`f64::clamp`
    // then panics on `min > max`). An index clamp is exact at every scale.
    let idx = |x: f64| -> usize {
        // NaN and anything at/left of the origin fold to bar 0 (an unordered
        // comparison must not leak into the index maths); `as usize` saturates, so
        // +inf lands on the last bar rather than being UB.
        if x.is_nan() || x <= 0.0 {
            return 0;
        }
        ((x / width * n as f64) as usize).min(n - 1)
    };
    let (a, b) = (idx(x0), idx(x1));
    Some((a.min(b), a.max(b)))
}

/// Whether any year in the inclusive bar-index span actually has images. A span of
/// only empty years would filter to a blank grid whose only way back is re-finding
/// invisible bars — the same dead end the single-click guard prevents. Pure.
///
/// An out-of-range span degrades to "no images" (a silent no-op) rather than
/// clamping, so the `debug_assert` makes a caller that breaks the precondition fail
/// loudly in tests instead of quietly doing nothing in release.
pub(crate) fn span_has_images(hist: &[(i32, u32)], lo: usize, hi: usize) -> bool {
    debug_assert!(
        lo <= hi && hi < hist.len(),
        "span {lo}..={hi} out of range for {} bars",
        hist.len()
    );
    hist.get(lo..=hi).is_some_and(|s| s.iter().any(|&(_, c)| c > 0))
}

/// What a completed press-drag-release on the strip means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DragIntent {
    /// Apply this inclusive year range.
    Apply(i32, i32),
    /// Clear the year filter (re-selecting what's already applied).
    Clear,
    /// Do nothing (empty histogram, or a selection of only gap years).
    Ignore,
}

/// Pure decision for a completed gesture: press at `sx`, released `off_x` away,
/// over a strip `width` wide showing `hist`, with `current` already applied.
///
/// Extracting this keeps the branchy part — slop classification, the gap-year dead
/// end guard, toggle-to-clear, and span→year mapping — unit-testable under the
/// display-free discipline. The leaves it composes were already tested; the
/// *combination* is where the double-click regression hid.
pub(crate) fn drag_intent(
    sx: f64,
    off_x: f64,
    width: f64,
    hist: &[(i32, u32)],
    current: Option<(i32, i32)>,
) -> DragIntent {
    let Some((lo, hi)) = (if off_x.abs() < DRAG_CLICK_SLOP_PX {
        // A release within the slop is a *click*: one bar, keyed off the press.
        bar_at_x(sx, width, hist.len()).map(|i| (i, i))
    } else {
        bar_span(sx, sx + off_x, width, hist.len())
    }) else {
        return DragIntent::Ignore;
    };
    // Selecting only gap years would filter to a blank grid whose only way back is
    // re-finding invisible bars. Applies to a single click and a whole span alike.
    if !span_has_images(hist, lo, hi) {
        return DragIntent::Ignore;
    }
    let range = (hist[lo].0, hist[hi].0);
    // Re-selecting exactly what's applied clears it — for a dragged span as well as
    // a clicked year, so the two interactions stay symmetric.
    if current == Some(range) {
        DragIntent::Clear
    } else {
        DragIntent::Apply(range.0, range.1)
    }
}

/// Whether this click is the repeat half of a multi-click on the same bar.
/// `GestureDrag` carries no `n_press`, so the second press of a double-click
/// arrives as an independent click — and with toggle-to-clear that would undo the
/// first press instantly, making the strip look inert. (`GestureClick` gave this
/// for free; unifying the gestures lost it, so it's reconstructed by time.) Pure.
pub(crate) fn is_repeat_click(
    prev: Option<(usize, std::time::Instant)>,
    bar: usize,
    now: std::time::Instant,
    window: std::time::Duration,
) -> bool {
    prev.is_some_and(|(pi, t)| pi == bar && now.duration_since(t) < window)
}

/// Height in pixels of a bar for `count`, given the tallest bar's `max` count and
/// the drawable `height`. Zero-count years (and a degenerate all-zero histogram)
/// yield 0 rather than dividing by zero. Pure.
pub(crate) fn bar_height(count: u32, max: u32, height: f64) -> f64 {
    if max == 0 || count == 0 || height <= 0.0 {
        return 0.0;
    }
    height * (count as f64 / max as f64)
}

/// Widest span of years the strip will draw a continuous axis for. A single
/// corrupt `datetime_taken` can decode to year 1 or 9999; without this the gap
/// fill would allocate thousands of bars and squash the real data into a sliver.
/// Beyond it we keep the sparse rows — a compressed axis beats an unreadable one.
const MAX_TIMELINE_SPAN_YEARS: usize = 200;

/// Insert zero-count entries for years missing between the first and last, so the
/// axis is **continuous in time**. Without this a gap year (the real catalog has
/// no 2017) is simply absent and the strip silently misrepresents a 2-year jump as
/// adjacent — darktable draws a continuous axis. Input must be ascending by year;
/// the histogram query's `ORDER BY yr` guarantees that. Pure.
pub(crate) fn fill_year_gaps(rows: Vec<(i32, u32)>) -> Vec<(i32, u32)> {
    let (Some(&(first, _)), Some(&(last, _))) = (rows.first(), rows.last()) else {
        return rows;
    };
    let span = (last - first).unsigned_abs() as usize + 1;
    // `last < first` (descending input) must bail BEFORE the fill: `unsigned_abs`
    // hides the sign, so such input would pass the span guards and then iterate an
    // empty `first..=last`, silently returning nothing. The ascending precondition
    // is a doc comment, not something the type system enforces, so check it —
    // unknown input shape passes through untouched rather than losing data.
    if last < first || span <= rows.len() || span > MAX_TIMELINE_SPAN_YEARS {
        return rows; // descending, already dense, or too wide to draw continuously
    }
    let mut out = Vec::with_capacity(span);
    let mut it = rows.into_iter().peekable();
    for year in first..=last {
        match it.peek() {
            Some(&(y, c)) if y == year => {
                out.push((year, c));
                it.next();
            }
            _ => out.push((year, 0)),
        }
    }
    out
}

/// Read the per-year histogram from the catalog, with empty years filled in so the
/// axis is continuous. Best-effort: an empty/absent db or a schema without
/// `datetime_taken` yields an empty histogram (the strip then draws nothing),
/// never an error the caller must handle.
pub(crate) fn load_histogram(db_path: &str) -> Vec<(i32, u32)> {
    if db_path.is_empty() {
        return Vec::new();
    }
    let Ok(conn) = rusqlite::Connection::open(db_path) else {
        return Vec::new();
    };
    // The rating/colour-label writers hold library.db's write lock from an
    // off-thread write, so a bare read can hit SQLITE_BUSY. This histogram is read
    // ONCE at construction with no retry, so losing that race would leave a blank
    // strip for the whole session — wait rather than give up instantly.
    let _ = conn.busy_timeout(std::time::Duration::from_secs(3));
    let Ok(mut stmt) = conn.prepare(&histogram_sql()) else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map([], |r| Ok((r.get::<_, i32>(0)?, r.get::<_, u32>(1)?))) else {
        return Vec::new();
    };
    fill_year_gaps(rows.flatten().collect())
}

/// Build the timeline strip: a `DrawingArea` of year bars, click to filter to a
/// year, click the selected year again to clear. Returns the widget for the
/// caller to place (the lighttable's bottom, under the toolbar).
///
/// The histogram is read **once** at construction — it maps the whole catalog and
/// is deliberately independent of the active filters (see [`histogram_sql`]), so
/// it only needs rebuilding after an import. The selection highlight tracks
/// [`super::current_year_range`] via the shared filter-observer bus, so clearing
/// the filter elsewhere un-highlights the strip too.
pub fn timeline_strip(db_path: &str) -> gtk4::DrawingArea {
    use gtk4::prelude::*;

    let hist = std::rc::Rc::new(load_histogram(db_path));

    let area = gtk4::DrawingArea::builder()
        .height_request(TIMELINE_HEIGHT)
        .hexpand(true)
        .build();
    area.add_css_class("darkroom-timeline");
    area.set_tooltip_text(Some(
        "Filter by date — click a year, drag across several, click again to clear",
    ));

    // In-progress drag span (bar indices), shared by the gesture and the draw func
    // so the strip previews the selection under the cursor before it's committed.
    // `None` whenever no drag is active, in which case the paint falls back to the
    // committed filter — one flag, so preview and committed state can't both show.
    let preview: std::rc::Rc<std::cell::Cell<Option<(usize, usize)>>> =
        std::rc::Rc::new(std::cell::Cell::new(None));

    // Press x, captured at drag-begin. `GestureDrag::start_point()` is only valid
    // while the gesture is active: by the time `drag-end` runs it has reset and
    // returns None (confirmed at runtime — the handler was bailing out and no span
    // was ever applied). So latch it ourselves rather than re-reading it later.
    let drag_start_x: std::rc::Rc<std::cell::Cell<Option<f64>>> =
        std::rc::Rc::new(std::cell::Cell::new(None));

    // (bar, when) of the last committed click, for the double-click guard.
    #[allow(clippy::type_complexity)]
    let last_click: std::rc::Rc<std::cell::Cell<Option<(usize, std::time::Instant)>>> =
        std::rc::Rc::new(std::cell::Cell::new(None));

    // ── Paint ──────────────────────────────────────────────────────────────
    area.set_draw_func({
        let hist = hist.clone();
        let preview = preview.clone();
        move |area, cr, w, h| {
            let (w, h) = (w as f64, h as f64);
            if hist.is_empty() {
                return;
            }
            // Colours come from the widget's own style context, so the strip
            // follows the (dark) theme instead of hard-coding greys.
            let fg = area.color();
            let max = hist.iter().map(|&(_, c)| c).max().unwrap_or(0);
            let n = hist.len();
            let bar_w = w / n as f64;
            let label_h = TIMELINE_LABEL_H;
            let plot_h = (h - label_h).max(0.0);
            // An in-progress drag preempts the committed filter for highlighting,
            // so the strip shows what you're about to select, not what's applied.
            let dragging = preview.get();
            let selected = super::current_year_range();

            for (i, &(year, count)) in hist.iter().enumerate() {
                let x = i as f64 * bar_w;
                let bh = bar_height(count, max, plot_h);
                // The highlighted year(s) are drawn solid and the rest dimmed, so
                // the strip reads as "what's filtered" at a glance.
                let (any, in_sel) = match dragging {
                    Some((lo, hi)) => (true, i >= lo && i <= hi),
                    None => (
                        selected.is_some(),
                        selected.is_some_and(|(lo, hi)| {
                            year >= lo.min(hi) && year <= lo.max(hi)
                        }),
                    ),
                };
                let alpha = if !any {
                    0.55
                } else if in_sel {
                    0.95
                } else {
                    0.20
                };
                cr.set_source_rgba(fg.red() as f64, fg.green() as f64, fg.blue() as f64, alpha);
                // Inset by 1px so adjacent bars read as separate columns.
                cr.rectangle(x + 1.0, plot_h - bh, (bar_w - 2.0).max(1.0), bh);
                let _ = cr.fill();

                // Year label, drawn only when the bars are wide enough to fit one
                // (otherwise they'd overlap into mush).
                if bar_w >= TIMELINE_MIN_LABEL_W {
                    cr.set_source_rgba(fg.red() as f64, fg.green() as f64, fg.blue() as f64, 0.75);
                    cr.set_font_size(TIMELINE_FONT_SIZE);
                    let text = year.to_string();
                    let tw = cr.text_extents(&text).map(|e| e.width()).unwrap_or(0.0);
                    cr.move_to(x + (bar_w - tw) / 2.0, h - 2.0);
                    let _ = cr.show_text(&text);
                }
            }
        }
    });

    // ── Click a bar, or drag across several (m4-99b) ───────────────────────
    // ONE `GestureDrag` serves both: a release within `DRAG_CLICK_SLOP_PX` of the
    // press is treated as a click. Two competing gestures (Click + Drag) on the
    // same widget would have to fight over claiming the sequence; a short drag is
    // exactly what a click is, so one handler is both simpler and more predictable.
    let drag = gtk4::GestureDrag::new();
    drag.connect_drag_begin({
        let drag_start_x = drag_start_x.clone();
        let preview = preview.clone();
        move |_, sx, _| {
            drag_start_x.set(Some(sx));
            preview.set(None); // belt-and-braces: a new gesture starts clean
        }
    });
    drag.connect_drag_update({
        let hist = hist.clone();
        let preview = preview.clone();
        let drag_start_x = drag_start_x.clone();
        move |g, off_x, _| {
            let Some(area) = g.widget() else { return };
            let Some(sx) = drag_start_x.get() else { return };
            // Live preview only once past the click slop, so a plain click never
            // flashes a selection before committing.
            let span = if off_x.abs() >= DRAG_CLICK_SLOP_PX {
                bar_span(sx, sx + off_x, area.width() as f64, hist.len())
            } else {
                None
            };
            if preview.get() != span {
                preview.set(span);
                area.queue_draw();
            }
        }
    });
    drag.connect_drag_end({
        let hist = hist.clone();
        let preview = preview.clone();
        let drag_start_x = drag_start_x.clone();
        let last_click = last_click.clone();
        move |g, off_x, _| {
            let Some(area) = g.widget() else { return };
            // Clear the preview FIRST, before any early return: it must not survive
            // a path that decides to do nothing, or the strip would keep showing a
            // span that was never applied (and hide the filter that is).
            let had_preview = preview.replace(None).is_some();
            let repaint = || {
                if had_preview {
                    area.queue_draw();
                }
            };
            let Some(sx) = drag_start_x.replace(None) else {
                repaint();
                return;
            };

            match drag_intent(sx, off_x, area.width() as f64, &hist, super::current_year_range()) {
                DragIntent::Ignore => repaint(),
                intent => {
                    // Swallow the repeat half of a double-click on the same bar —
                    // otherwise its second press toggles the first straight back off
                    // and the strip looks inert. Spans reset the stamp, so a click
                    // right after a drag is never eaten.
                    let bar = bar_at_x(sx, area.width() as f64, hist.len());
                    let now = std::time::Instant::now();
                    let is_click = off_x.abs() < DRAG_CLICK_SLOP_PX;
                    if is_click {
                        if let Some(b) = bar {
                            if is_repeat_click(last_click.get(), b, now, double_click_window()) {
                                last_click.set(Some((b, now))); // re-stamp: a triple
                                repaint(); //                      click is one action
                                return;
                            }
                            last_click.set(Some((b, now)));
                        }
                    } else {
                        last_click.set(None);
                    }
                    super::set_year_range(match intent {
                        DragIntent::Apply(lo, hi) => Some((lo, hi)),
                        _ => None,
                    });
                }
            }
        }
    });
    // A sequence can die WITHOUT `drag-end`: touch-cancel, a broken grab, the widget
    // being unmapped, or another controller claiming the sequence (ours then goes
    // DENIED). Without this the last preview stays latched forever. Every path that
    // arms the preview needs a path that disarms it — there are three exits.
    drag.connect_cancel({
        let preview = preview.clone();
        move |g, _| {
            // Clear ONLY the preview, never `drag_start_x`. GTK emits `cancel`
            // BEFORE `drag-end` on an ordinary button release here (confirmed by
            // probing the container logs), so wiping the latch would make every
            // completed drag bail out of `drag-end` — which is exactly the bug this
            // handler's first version shipped. The latch needs no cleanup anyway:
            // `drag-begin` always precedes the next read and overwrites it.
            if preview.replace(None).is_some() {
                if let Some(a) = g.widget() {
                    a.queue_draw();
                }
            }
        }
    });
    area.add_controller(drag);

    // Repaint whenever ANY filter control changes the year range (including a
    // clear made elsewhere), so the highlight can't drift from the live filter.
    // Held WEAKLY: the observer list is never pruned, so a strong ref would keep
    // every strip alive forever — harmless while this is built once, but the
    // planned rebuild-after-import would then redraw a pile of dead widgets.
    {
        let weak = area.downgrade();
        super::add_filter_observer(move || {
            if let Some(a) = weak.upgrade() {
                a.queue_draw();
            }
        });
    }

    // Nothing to show (no db — the demo/default launch path — or no dated images):
    // hide the strip rather than leave a blank 56px band with a tooltip promising
    // bars to click. ToolbarView honours child visibility, so the bar collapses.
    area.set_visible(!hist.is_empty());

    area
}

/// Overall height of the timeline strip, and how much of it the year labels take.
const TIMELINE_HEIGHT: i32 = 56;
const TIMELINE_LABEL_H: f64 = 14.0;
/// Below this bar width the year labels would overlap, so they're dropped.
const TIMELINE_MIN_LABEL_W: f64 = 28.0;
const TIMELINE_FONT_SIZE: f64 = 10.0;
/// The desktop's double-click interval, for the repeat-click guard — read from
/// GTK so it follows the user's setting rather than a hard-coded guess.
fn double_click_window() -> std::time::Duration {
    let ms = gtk4::Settings::default().map_or(400, |s| s.gtk_double_click_time().max(0));
    std::time::Duration::from_millis(ms as u64)
}

/// Horizontal movement (px) below which a press-and-release counts as a *click*
/// rather than a span drag. One gesture serves both, so this is the dividing line;
/// it also absorbs the hand tremor in a normal click.
const DRAG_CLICK_SLOP_PX: f64 = 4.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn year_expr_converts_dt_microseconds_to_a_year() {
        // Locks the origin constant AND the shape of the expression, since the
        // histogram and the filter both depend on them agreeing.
        let e = year_sql_expr("i.datetime_taken");
        assert_eq!(
            e,
            "CAST(strftime('%Y', (i.datetime_taken / 1000000 - 62135596800), 'unixepoch') AS INTEGER)"
        );
    }

    #[test]
    fn year_expr_decodes_real_catalog_values() {
        // End-to-end through SQLite itself: the value below is P7280008.ORF from
        // the real catalog, whose film roll is `2018_07_28` — so the decode must
        // land on 2018. This is the check that pins DT_EPOCH_OFFSET_SECS to the
        // year-1 origin (the year-0 variant yields 2017).
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, datetime_taken INTEGER);
             -- Real catalog values: P7280008.ORF (film roll 2018_07_28) and the
             -- catalog's earliest image. Both decode to the second.
             INSERT INTO images VALUES (1, 63668412473000000); -- 2018-07-28 22:07:53
             INSERT INTO images VALUES (2, 63556269799000000); -- 2015-01-07 23:23:19
             INSERT INTO images VALUES (3, 0);                 -- undated
             INSERT INTO images VALUES (4, NULL);              -- undated (NULL)",
        )
        .unwrap();
        let full = |id: i64| -> Option<String> {
            conn.query_row(
                "SELECT datetime(datetime_taken / 1000000 - 62135596800, 'unixepoch') \
                 FROM images WHERE id = ?1",
                [id],
                |r| r.get::<_, Option<String>>(0),
            )
            .unwrap()
        };
        // Pin the constant to the second, not just the year — a wrong origin that
        // happened to land in the right year would otherwise slip through.
        assert_eq!(full(1).as_deref(), Some("2018-07-28 22:07:53"));
        assert_eq!(full(2).as_deref(), Some("2015-01-07 23:23:19"));

        let year_of = |id: i64| -> Option<i32> {
            conn.query_row(
                &format!("SELECT {} FROM images WHERE id = ?1", year_sql_expr("datetime_taken")),
                [id],
                |r| r.get::<_, Option<i32>>(0),
            )
            .unwrap()
        };
        assert_eq!(year_of(1), Some(2018));
        assert_eq!(year_of(2), Some(2015));
        // An undated row decodes to year 1 — which is exactly why the filter and
        // the histogram both exclude `datetime_taken <= 0` rather than trusting it.
        assert_eq!(year_of(3), Some(1));
        // NULL takes a different SQLite path (NULL > 0 is NULL, not false) but must
        // land in the same place: excluded, never a year-1 bucket.
        assert_eq!(year_of(4), None);
    }

    #[test]
    fn year_bucketing_is_utc_and_exact_at_the_boundary() {
        // Locks UTC decoding: darktable stores the EXIF wall clock as UTC, so
        // these must bucket by the wall clock itself. Under 'localtime' the first
        // value would slide into 2019 (or the last into 2020) depending on the
        // machine's TZ, making a photo's year depend on who's looking at it.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE images (id INTEGER PRIMARY KEY, datetime_taken INTEGER);")
            .unwrap();
        // 2019-12-31 23:59:59 and 2020-01-01 00:00:00 as µs since 0001-01-01.
        let to_dt = |unix: i64| (unix + DT_EPOCH_OFFSET_SECS) * 1_000_000;
        let last_2019 = to_dt(1_577_836_799); // 2019-12-31T23:59:59Z
        let first_2020 = to_dt(1_577_836_800); // 2020-01-01T00:00:00Z
        conn.execute("INSERT INTO images VALUES (1, ?1)", [last_2019]).unwrap();
        conn.execute("INSERT INTO images VALUES (2, ?1)", [first_2020]).unwrap();
        let year_of = |id: i64| -> i32 {
            conn.query_row(
                &format!("SELECT {} FROM images WHERE id = ?1", year_sql_expr("datetime_taken")),
                [id],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(year_of(1), 2019, "one second before midnight stays in 2019");
        assert_eq!(year_of(2), 2020, "midnight exactly starts 2020");
    }

    #[test]
    fn year_range_filter_selects_only_that_span_and_never_undated() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, filename TEXT, datetime_taken INTEGER);
             INSERT INTO images VALUES (1, 'a.raw', 63556269799000000); -- 2015
             INSERT INTO images VALUES (2, 'b.raw', 63667433261000000); -- 2018
             INSERT INTO images VALUES (3, 'c.raw', 63908591112000000); -- 2026
             INSERT INTO images VALUES (4, 'undated.raw', 0);",
        )
        .unwrap();
        let run = |range: Option<(i32, i32)>| -> Vec<String> {
            let sql = format!(
                "SELECT i.filename FROM images i WHERE 1=1{} ORDER BY i.filename",
                year_range_and(range)
            );
            conn.prepare(&sql)
                .unwrap()
                .query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .flatten()
                .collect()
        };
        // No range → everything, undated included (it's the no-filter state).
        assert_eq!(run(None), ["a.raw", "b.raw", "c.raw", "undated.raw"]);
        // A single year, and a span — undated is excluded from both.
        assert_eq!(run(Some((2018, 2018))), ["b.raw"]);
        assert_eq!(run(Some((2015, 2018))), ["a.raw", "b.raw"]);
        // An inverted range (right-to-left drag) is normalised, not empty.
        assert_eq!(run(Some((2018, 2015))), ["a.raw", "b.raw"]);
        // A span matching nothing is legitimately empty.
        assert_eq!(run(Some((2020, 2021))), Vec::<String>::new());
    }

    #[test]
    fn histogram_sql_groups_by_year_excluding_undated() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE main.images (id INTEGER PRIMARY KEY, datetime_taken INTEGER);
             INSERT INTO main.images VALUES (1, 63556269799000000); -- 2015
             INSERT INTO main.images VALUES (2, 63667433261000000); -- 2018
             INSERT INTO main.images VALUES (3, 63667433262000000); -- 2018
             INSERT INTO main.images VALUES (4, 0);                 -- undated",
        )
        .unwrap();
        let rows: Vec<(i32, u32)> = conn
            .prepare(&histogram_sql())
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .flatten()
            .collect();
        // Ascending years, undated dropped entirely (no year-1 bucket).
        assert_eq!(rows, vec![(2015, 1), (2018, 2)]);
    }

    #[test]
    fn gap_years_are_filled_so_the_axis_is_continuous() {
        // The real catalog has no 2017; without the fill the strip would draw 2016
        // and 2018 adjacent, silently misrepresenting the time axis.
        assert_eq!(
            fill_year_gaps(vec![(2015, 1), (2016, 19), (2018, 1637)]),
            vec![(2015, 1), (2016, 19), (2017, 0), (2018, 1637)]
        );
        // Already-dense input is returned untouched.
        let dense = vec![(2020, 5), (2021, 6)];
        assert_eq!(fill_year_gaps(dense.clone()), dense);
        // Degenerate inputs don't panic.
        assert_eq!(fill_year_gaps(vec![]), vec![]);
        assert_eq!(fill_year_gaps(vec![(2020, 3)]), vec![(2020, 3)]);
        // A corrupt date spanning millennia falls back to the sparse rows rather
        // than allocating thousands of bars that squash the real data flat.
        let wild = vec![(1, 1), (2020, 500)];
        assert_eq!(fill_year_gaps(wild.clone()), wild);
        // Descending input must pass through, NOT come back empty: `unsigned_abs`
        // hides the sign, so without the `last < first` guard the span checks pass
        // and `first..=last` iterates nothing — silent total data loss, showing up
        // as a blank strip rather than a panic.
        let desc = vec![(2020, 1), (2015, 1)];
        assert_eq!(fill_year_gaps(desc.clone()), desc);
    }

    #[test]
    fn bar_at_x_maps_clicks_and_rejects_out_of_range() {
        // 4 bars across 100px ⇒ 25px each.
        assert_eq!(bar_at_x(0.0, 100.0, 4), Some(0));
        assert_eq!(bar_at_x(24.9, 100.0, 4), Some(0));
        assert_eq!(bar_at_x(25.0, 100.0, 4), Some(1));
        assert_eq!(bar_at_x(99.9, 100.0, 4), Some(3));
        // The right edge is exclusive — a click there is a miss, not bar n.
        // (The `.min(n - 1)` inside stays as belt-and-braces: float rounding just
        // below `width` could otherwise still compute an index of exactly `n`.)
        assert_eq!(bar_at_x(100.0, 100.0, 4), None, "past the edge is a miss");
        assert_eq!(bar_at_x(-1.0, 100.0, 4), None);
        // Degenerate inputs never panic or divide by zero.
        assert_eq!(bar_at_x(10.0, 100.0, 0), None, "empty histogram");
        assert_eq!(bar_at_x(10.0, 0.0, 4), None, "unallocated widget");
    }

    #[test]
    fn bar_span_orders_endpoints_and_clamps_to_the_strip() {
        // 4 bars across 100px ⇒ 25px each.
        assert_eq!(bar_span(10.0, 60.0, 100.0, 4), Some((0, 2)));
        // A right-to-left drag selects the same span as left-to-right.
        assert_eq!(bar_span(60.0, 10.0, 100.0, 4), Some((0, 2)));
        // A drag within one bar is that single bar.
        assert_eq!(bar_span(5.0, 20.0, 100.0, 4), Some((0, 0)));
        // Running off either edge selects out TO that edge rather than failing —
        // the gesture legitimately reports coordinates outside the widget.
        assert_eq!(bar_span(-50.0, 60.0, 100.0, 4), Some((0, 2)));
        assert_eq!(bar_span(60.0, 500.0, 100.0, 4), Some((2, 3)));
        assert_eq!(bar_span(-10.0, 999.0, 100.0, 4), Some((0, 3)), "whole strip");
        // Degenerate inputs never panic — the index-space clamp holds at every
        // scale, unlike a coordinate nudge (a width below the nudge made
        // `f64::clamp` panic on min > max; a huge one made it a no-op).
        assert_eq!(bar_span(10.0, 20.0, 100.0, 0), None, "empty histogram");
        assert_eq!(bar_span(10.0, 20.0, 0.0, 4), None, "unallocated widget");
        assert_eq!(bar_span(10.0, 20.0, 1e-12, 4), Some((3, 3)), "sub-pixel width");
        assert_eq!(bar_span(1e12, 2e12, 1e9, 4), Some((3, 3)), "far past a huge width");
        // NaN folds to bar 0 rather than propagating into the index maths.
        assert_eq!(bar_span(f64::NAN, 60.0, 100.0, 4), Some((0, 2)));
        assert_eq!(bar_span(f64::INFINITY, 60.0, 100.0, 4), Some((2, 3)));
    }

    #[test]
    fn drag_intent_classifies_click_span_toggle_and_dead_ends() {
        // 4 bars across 100px ⇒ 25px each; 2016 is a gap year.
        let hist = [(2015, 5), (2016, 0), (2017, 3), (2018, 9)];
        let w = 100.0;
        // A release within the slop is a click on the PRESS bar.
        assert_eq!(drag_intent(10.0, 0.0, w, &hist, None), DragIntent::Apply(2015, 2015));
        assert_eq!(drag_intent(10.0, 3.9, w, &hist, None), DragIntent::Apply(2015, 2015));
        // Past the slop it's a span.
        assert_eq!(drag_intent(10.0, 60.0, w, &hist, None), DragIntent::Apply(2015, 2017));
        // Right-to-left drags select the same span as left-to-right.
        assert_eq!(drag_intent(70.0, -60.0, w, &hist, None), DragIntent::Apply(2015, 2017));
        // Re-selecting exactly what's applied clears — for a click AND a span, so
        // the two interactions stay symmetric.
        assert_eq!(drag_intent(10.0, 0.0, w, &hist, Some((2015, 2015))), DragIntent::Clear);
        assert_eq!(drag_intent(10.0, 60.0, w, &hist, Some((2015, 2017))), DragIntent::Clear);
        // A *different* range replaces rather than clears.
        assert_eq!(drag_intent(10.0, 0.0, w, &hist, Some((2018, 2018))), DragIntent::Apply(2015, 2015));
        // Clicking a gap year is a dead end (blank grid, invisible way back).
        assert_eq!(drag_intent(30.0, 0.0, w, &hist, None), DragIntent::Ignore);
        // ...and so is a span of only gap years.
        assert_eq!(drag_intent(26.0, 20.0, w, &hist, None), DragIntent::Ignore);
        // A span *containing* a gap year is fine — it has data at the ends.
        assert_eq!(drag_intent(10.0, 40.0, w, &hist, None), DragIntent::Apply(2015, 2017));
        // Degenerate: no bars ⇒ nothing to do, never a panic.
        assert_eq!(drag_intent(10.0, 5.0, w, &[], None), DragIntent::Ignore);
    }

    #[test]
    fn repeat_click_guard_matches_same_bar_within_the_window() {
        use std::time::{Duration, Instant};
        let t0 = Instant::now();
        let win = Duration::from_millis(400);
        // The second press of a double-click on the same bar is swallowed —
        // otherwise it toggles the first straight back off and the strip looks inert.
        assert!(is_repeat_click(Some((3, t0)), 3, t0 + Duration::from_millis(120), win));
        // A different bar is a new click, however fast.
        assert!(!is_repeat_click(Some((3, t0)), 4, t0 + Duration::from_millis(20), win));
        // The same bar after the window is a deliberate re-click (toggle-to-clear).
        assert!(!is_repeat_click(Some((3, t0)), 3, t0 + Duration::from_millis(700), win));
        // No previous click ⇒ never a repeat.
        assert!(!is_repeat_click(None, 3, t0, win));
    }

    #[test]
    fn span_has_images_guards_the_all_empty_dead_end() {
        let hist = [(2015, 1), (2016, 0), (2017, 0), (2018, 5)];
        assert!(span_has_images(&hist, 0, 0), "a single non-empty year");
        assert!(span_has_images(&hist, 0, 3), "span containing data");
        assert!(span_has_images(&hist, 2, 3), "data at the far end still counts");
        // A span of only gap years would filter to a blank grid.
        assert!(!span_has_images(&hist, 1, 2));
        assert!(!span_has_images(&hist, 1, 1));
        // Out-of-range spans are a caller bug, not a supported input: the
        // `debug_assert` fires in test/debug builds so a future caller that breaks
        // the precondition is caught, and release degrades to "no images" (a silent
        // no-op) rather than panicking on a user's machine. `bar_span` clamps
        // through `bar_at_x`, so in practice the precondition always holds.
    }

    #[test]
    fn bar_height_scales_to_the_tallest_and_survives_degenerate_input() {
        assert_eq!(bar_height(50, 100, 40.0), 20.0);
        assert_eq!(bar_height(100, 100, 40.0), 40.0);
        assert_eq!(bar_height(0, 100, 40.0), 0.0, "empty year draws nothing");
        assert_eq!(bar_height(5, 0, 40.0), 0.0, "all-zero histogram: no divide by zero");
        assert_eq!(bar_height(5, 10, 0.0), 0.0, "unallocated height");
    }
}
