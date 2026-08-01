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
    area.set_tooltip_text(Some("Filter by year — click a bar, click again to clear"));

    // ── Paint ──────────────────────────────────────────────────────────────
    area.set_draw_func({
        let hist = hist.clone();
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
            let selected = super::current_year_range();

            for (i, &(year, count)) in hist.iter().enumerate() {
                let x = i as f64 * bar_w;
                let bh = bar_height(count, max, plot_h);
                // A selected year is drawn solid, the rest dimmed — the strip is
                // then readable as "what's filtered" at a glance.
                let in_sel = selected.is_some_and(|(lo, hi)| year >= lo.min(hi) && year <= lo.max(hi));
                let alpha = if selected.is_none() {
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

    // ── Click to filter ────────────────────────────────────────────────────
    let click = gtk4::GestureClick::new();
    click.connect_released({
        let hist = hist.clone();
        move |g, n_press, x, _| {
            // `released` fires for the 1st AND 2nd press of a double-click; with a
            // re-click-to-clear toggle that would select then instantly clear,
            // making the control look inert. Act on the first press only.
            if n_press > 1 {
                return;
            }
            let Some(area) = g.widget() else { return };
            let Some(i) = bar_at_x(x, area.width() as f64, hist.len()) else {
                return;
            };
            let (year, count) = hist[i];
            // A gap year (zero count, drawn as no bar) would filter to an empty
            // grid whose only way back is re-finding that same invisible bar.
            if count == 0 {
                return;
            }
            // Toggle: clicking the year already filtered to clears the filter,
            // mirroring the rating stars' re-click-to-clear behaviour.
            let next = if super::current_year_range() == Some((year, year)) {
                None
            } else {
                Some((year, year))
            };
            super::set_year_range(next);
        }
    });
    area.add_controller(click);

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
    fn bar_height_scales_to_the_tallest_and_survives_degenerate_input() {
        assert_eq!(bar_height(50, 100, 40.0), 20.0);
        assert_eq!(bar_height(100, 100, 40.0), 40.0);
        assert_eq!(bar_height(0, 100, 40.0), 0.0, "empty year draws nothing");
        assert_eq!(bar_height(5, 0, 40.0), 0.0, "all-zero histogram: no divide by zero");
        assert_eq!(bar_height(5, 10, 0.0), 0.0, "unallocated height");
    }
}
