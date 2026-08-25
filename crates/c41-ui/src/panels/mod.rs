//! Side-panel widgets -- collections list and metadata inspector.
//!
//! Phase 3-ui-6:
//!   LeftPanel  -- film rolls with image counts; clicking filters the grid
//!   MetadataPanel -- right panel that updates when an image is selected

use adw::prelude::*;
use glib::clone;
use std::cell::RefCell;
use std::rc::Rc;
use crate::lighttable::{
    self, LighttableModel, lighttable_load_by_folder,
    lighttable_load_by_tag_prefix, color_dot_markup,
    rule_stack::{self, Combinator, PropertyKind, Rule, RuleCmp, RuleProperty},
    COLOR_COUNT,
};
use c41_db;

// ── Left panel (collections) ──────────────────────────────────────────────

/// The tag-section state a tag rename/delete popover needs, split out of
/// `LeftPanel` (m4-27) so the per-row secondary-click gesture reconstructs
/// exactly these fields on demand rather than a whole `LeftPanel` (which would
/// otherwise drag along the folder filter and colour section the menu ignores).
/// All fields are GObject ref-counts (plus a `String`), so it is `Clone` and cheap
/// to hand to the deferred rename/delete closures. Owns every tag-mutation method
/// (refresh / append-row / rename / delete); `LeftPanel` delegates to it.
#[derive(Clone)]
struct TagPanel {
    /// Stable tag list box; only its rows are rebuilt on refresh so the
    /// folder↔tag selection-coordination handlers (bound once) stay valid.
    tag_box:     gtk4::ListBox,
    /// Section chrome whose visibility tracks whether any user tags exist.
    tags_header: gtk4::Label,
    tags_sep:    gtk4::Separator,
    db_path:     String,
    /// Optional notify fired after a library-wide tag mutation here (rename /
    /// delete), so the metadata panel can re-render the current image's chips.
    /// Mirror of `MetadataPanel::on_tags_changed`. Set via `set_on_tags_changed`.
    on_tags_changed: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>>,
}

/// The collections (left) panel: film rolls plus a live Tags section.
///
/// Clicking a film roll reloads the lighttable to show only that folder; the
/// first row ("All images") clears the filter. The Tags section can be rebuilt
/// in place via [`LeftPanel::refresh_tags`] after a tag is attached elsewhere
/// (e.g. from the metadata panel), so newly-created tags and changed counts
/// appear without restarting the app. The tag list + all tag-mutation logic live
/// in the [`TagPanel`] field; `LeftPanel` owns the folder filter and the colour /
/// collection-filter sections' widgets, whose state lives in `lighttable`'s
/// canonical quick-filters (m4-126/m4-128) rather than here.
///
/// All fields are GObject ref-counts (plus the `TagPanel`, itself ref-counts), so
/// `LeftPanel` is Clone and can be handed to the metadata panel's change callback
/// cheaply.
#[derive(Clone)]
pub struct LeftPanel {
    pub widget:  gtk4::Box,
    /// Film-roll (folder) list box, incl. the "All images" row. Held so
    /// [`LeftPanel::clear_filter_highlights`] can drop a stale highlight.
    list_box:    gtk4::ListBox,
    /// Tag section (list + all tag-mutation methods), split out so its rename/
    /// delete popover doesn't reconstruct the whole panel — see [`TagPanel`].
    tags:        TagPanel,
}

impl LeftPanel {
    /// `active_tag` is the shared "currently-filtering tag prefix" (None = no tag
    /// filter). It holds a tag's full `parent|child` path; the folder/tag click
    /// handlers keep it current so a later tag mutation can re-run the same
    /// hierarchical-prefix grid filter.
    pub fn new(
        db_path: &str,
        lt_model: &LighttableModel,
        active_tag: &std::rc::Rc<std::cell::RefCell<Option<String>>>,
    ) -> Self {
        let panel = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(0)
            .width_request(210)
            .build();

        // Both the Collections and Tags sections scroll together inside one
        // content box so neither steals the other's vertical space.
        let content = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(0)
            .build();

        // ── Import (parity 2.6) ───────────────────────────────────────────
        // darktable's left panel opens with an import module; ours had import
        // only as a header button, so the panel didn't match and the action was
        // less discoverable. This surfaces the SAME `win.import` action rather
        // than duplicating the dialog — one code path, one behaviour.
        //
        // `set_action_name` resolves through the widget hierarchy at activation
        // time, so it binds correctly once the panel is inside the window; no
        // window reference is needed here.
        let import_header = section_header("Import");
        let import_sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);
        let import_btn = gtk4::Button::builder()
            .label("Add images…")
            .tooltip_text("Import images into the library (Ctrl+I)")
            .margin_start(10).margin_end(10).margin_top(4).margin_bottom(6)
            .build();
        import_btn.set_action_name(Some("win.import"));
        content.append(&collapsible_section(
            &import_header,
            &[import_sep.clone().upcast::<gtk4::Widget>(), import_btn.clone().upcast()]
                .iter()
                .collect::<Vec<_>>(),
            true,
            db_path,
            IMPORT_SECTION_PREF_KEY,
        ));

        // ── Collections (film rolls) ──────────────────────────────────────
        let collections_header = section_header("Collections");
        let collections_sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);

        let list_box = gtk4::ListBox::builder()
            .selection_mode(gtk4::SelectionMode::Single)
            .build();
        list_box.add_css_class("navigation-sidebar");

        // "All images" row
        append_roll_row(&list_box, "All images", -1, None);

        // Film roll rows from DB
        let rolls = load_film_rolls(db_path);
        for (folder, count) in &rolls {
            append_roll_row(&list_box, folder, *count, Some(folder.as_str()));
        }

        // The Tags list box is built up-front (even if empty) so the folder
        // handler can clear its selection — the two SelectionMode::Single boxes
        // are mutually exclusive, so a folder/tag filter never leaves a stale
        // highlight in the other list implying an AND that isn't running.
        // Clicking "All images" (which clears the tag highlight too) is the way
        // out of a tag filter. The box is stable across refreshes; only its rows
        // are rebuilt, so the handlers bound below never go stale.
        let tag_box = gtk4::ListBox::builder()
            .selection_mode(gtk4::SelectionMode::Single)
            .build();
        tag_box.add_css_class("navigation-sidebar");

        // Colour-label quick-filter box (m4-126 reconcile): a multi-select set of
        // independent `CheckButton`s plus an Any/All mode toggle (m4-26), now ONE
        // MIRROR of the canonical filter state that lives in
        // `lighttable::set_colour_filter` alongside the rating/year filters — so it
        // composes ON TOP of whatever collection is active (folder / tag / search /
        // all) exactly like darktable's bar-mounted filters, and the top/bottom
        // bars' circles (lib.rs) drive the same state through the observer bus.
        // Collection switches therefore leave this filter alone (as they do the
        // stars); there is no mutual-exclusion clearing any more, because the AND
        // it implies really is running.
        let color_box = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .build();
        for idx in 0..COLOR_COUNT {
            append_color_check(&color_box, idx);
        }
        let mode_toggle = gtk4::ToggleButton::builder()
            .label("Match any")
            .tooltip_text("Match images with ANY selected colour; toggle for ALL")
            .halign(gtk4::Align::Start)
            .margin_start(12)
            .margin_end(8)
            .margin_top(4)
            .margin_bottom(4)
            .build();

        // Activate: reload lighttable with folder filter, dropping any tag filter.
        // The colour quick-filter is NOT dropped: like the star filter, it composes
        // on top of whatever collection this click selects (m4-126).
        let db = db_path.to_string();
        let at_folder = active_tag.clone();
        list_box.connect_row_activated(
            clone!(@weak lt_model, @weak tag_box => move |_, row| {
            tag_box.unselect_all();
            *at_folder.borrow_mut() = None;   // a folder/all view is not a tag filter
            let folder_filter: Option<String> = row
                .widget_name()
                .as_str()
                .ne("all")
                .then(|| row.widget_name().to_string());
            lighttable_load_by_folder(
                &lt_model,
                &db,
                folder_filter.as_deref(),
            );
        }));
        // darktable-style collapsible sections (parity 3.2): each panel section
        // folds away from its title row, which is what keeps a panel with this
        // many sections navigable without endless scrolling.
        content.append(&collapsible_section(
            &collections_header,
            &[collections_sep.clone().upcast(), list_box.clone().upcast()]
                .iter()
                .collect::<Vec<_>>(),
            true,
            db_path,
            COLLECTIONS_SECTION_PREF_KEY,
        ));

        // ── Colours (colour-label quick filter) ───────────────────────────
        // The five colour labels as independent checks plus an Any/All combine
        // toggle (m4-26), driving the CANONICAL compose-on-top filter state
        // (`lighttable::set_colour_filter`, m4-126) rather than a collection of
        // their own. Always present (the colour domain is fixed, not data-driven),
        // so no refresh/visibility toggle is needed. The `color_box` + checks were
        // built above.
        let colours_header = section_header("Colours");
        let colours_sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);

        // Seed the mirrors from the restored canonical state BEFORE connecting any
        // handler — same ordering contract as the bottom bar's comparator seeding:
        // a programmatic write with no handler connected fires nothing.
        seed_colour_controls(&color_box, &mode_toggle);

        // Each check pushes the box's full mask into the canonical state. An empty
        // mask means "no colour restriction" there, so unchecking the last colour
        // cleanly drops just this filter off the current view. Programmatic writes
        // (the sync observer below) are skipped via the shared guard.
        {
            let color_box = color_box.clone();
            let mut child = color_box.first_child();
            while let Some(w) = child {
                if let Some(check) = w.downcast_ref::<gtk4::CheckButton>() {
                    let color_box = color_box.clone();
                    check.connect_toggled(move |_| {
                        if lighttable::filter_sync_in_progress() { return; }
                        lighttable::set_colour_filter(
                            color_mask_from_box(&color_box),
                            lighttable::current_colour_all(),
                        );
                    });
                }
                child = w.next_sibling();
            }
        }

        // Any/All toggle flips the combine mode in the canonical state. Flipping
        // with no colours selected changes nothing on screen (empty fragment), but
        // writing it anyway keeps the state exactly what the controls show.
        mode_toggle.connect_toggled(move |btn| {
            if lighttable::filter_sync_in_progress() { return; }
            lighttable::set_colour_filter(lighttable::current_colour_mask(), btn.is_active());
        });

        // Observer half: repaint these mirrors whenever ANY control changes the
        // filter (this panel's own checks, either bar's circles). The bus invokes
        // this inside its sync pass, so these programmatic writes are covered by
        // the guard and the handlers above skip them — no guard consultation in
        // here (the pass itself guarantees it), only in the handlers.
        {
            let color_box = color_box.clone();
            let mode_toggle = mode_toggle.clone();
            lighttable::add_filter_observer(move || {
                sync_colour_controls_display(&color_box, &mode_toggle);
            });
        }
        content.append(&collapsible_section(
            &colours_header,
            &[
                colours_sep.clone().upcast::<gtk4::Widget>(),
                mode_toggle.clone().upcast(),
                color_box.clone().upcast(),
            ]
            .iter()
            .collect::<Vec<_>>(),
            true,
            db_path,
            COLOURS_SECTION_PREF_KEY,
        ));

        // ── Collection filters (m4-128) ───────────────────────────────────
        // First slice of darktable's "collection filters" expander
        // (src/libs/filtering.c): quick rules that compose ON TOP of whatever
        // collection is active. Its three stock aspect presets (square /
        // landscape / portrait) arrive as one dropdown driving the canonical
        // `lighttable::set_aspect_filter` state through the same observer bus as
        // the colour checks above.
        let filters_header = section_header("Collection filters");
        let filters_sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);
        let labels: Vec<&str> = lighttable::AspectFilter::ALL.iter().map(|f| f.label()).collect();
        let aspect_drop = gtk4::DropDown::from_strings(&labels);
        aspect_drop.set_tooltip_text(Some(
            "Filter by aspect ratio — darktable's square / landscape / portrait presets",
        ));
        aspect_drop.set_margin_start(12);
        aspect_drop.set_margin_end(12);
        aspect_drop.set_margin_top(4);
        aspect_drop.set_margin_bottom(6);

        // Seed from the restored canonical state BEFORE connecting the handler —
        // same ordering contract as the colour checks: a programmatic write with
        // no handler connected fires nothing.
        aspect_drop.set_selected(lighttable::current_aspect_filter().to_index());

        aspect_drop.connect_selected_notify(|dd| {
            if lighttable::filter_sync_in_progress() { return; }
            lighttable::set_aspect_filter(lighttable::AspectFilter::from_index(dd.selected()));
        });

        // Observer half: re-select the row matching the canonical state whenever
        // ANY control changes a filter. Runs inside the bus's sync pass, so this
        // write is covered by the guard and the handler above skips it.
        {
            let aspect_drop = aspect_drop.clone();
            lighttable::add_filter_observer(move || {
                aspect_drop.set_selected(lighttable::current_aspect_filter().to_index());
            });
        }

        // ── Rule stack (m4-134) ────────────────────────────────────────────
        // The arbitrary half of darktable's filtering expander: N rules joined
        // by AND / OR / AND NOT, composing on top of the collection like the
        // dropdown above. One writer shape throughout: widget change → collect →
        // `set_rule_stack`; the observer rebuilds rows ONLY when canonical state
        // diverges from what they show, because a rebuild on every keystroke
        // would destroy the Entry being typed in.
        //
        // Nested fns, not closures, for the shared plumbing — they take their
        // state as parameters, which keeps the add/delete/rebuild paths honest
        // about what each of them touches.
        struct RuleRow {
            comb: gtk4::DropDown,
            prop: gtk4::DropDown,
            /// Comparator dropdown for textual properties (`contains`/`excludes`).
            cmp_text: gtk4::DropDown,
            /// Comparator dropdown for numeric properties (`<`…`>`). Exactly one
            /// of the two is visible at a time, following the selected property's
            /// kind — the same visibility-switch pattern the combinator uses.
            cmp_num: gtk4::DropDown,
            entry: gtk4::Entry,
        }

        type Rows = Rc<RefCell<Vec<RuleRow>>>;

        fn string_dropdown(labels: &[&str], selected: u32) -> gtk4::DropDown {
            let dd = gtk4::DropDown::from_strings(labels);
            dd.set_selected(selected);
            dd
        }

        /// One kind's comparator dropdown: its set's labels, with `cmp`
        /// preselected (position 0 when `cmp` belongs to the other kind —
        /// canonical state never carries that combination, but a defensive
        /// fallback beats a panic).
        fn cmp_dropdown(set: &[RuleCmp], cmp: RuleCmp) -> gtk4::DropDown {
            let labels: Vec<&str> = set.iter().map(|c| c.label()).collect();
            let dd = gtk4::DropDown::from_strings(&labels);
            dd.set_selected(cmp.position_in(set).unwrap_or(0));
            dd
        }

        /// A fresh row with `rule`'s state preselected (defaults for the add
        /// path), WIRED before it returns. Handlers connect exactly once, here —
        /// rows are reused across add/delete rebuild passes (only the observer's
        /// wholesale replacement mints widgets), so wiring inside a layout pass
        /// would stack duplicate handlers on every surviving widget and multiply
        /// reload cycles per edit. Preselection happens BEFORE connecting so
        /// construction writes fire nothing.
        fn new_rule_row(rule: Option<&Rule>, rows: &Rows) -> RuleRow {
            let (prop_i, rule_cmp, comb_i, value) = match rule {
                Some(r) => (
                    r.property.to_index(),
                    r.cmp,
                    r.comb.to_index(),
                    r.value.clone(),
                ),
                None => (0, RuleCmp::Contains, 0, String::new()),
            };
            // The property decides which comparator family this row shows; each
            // dropdown preselects `rule_cmp` within its own set (position 0 if
            // it belongs to the other family).
            let property = RuleProperty::from_index(prop_i);
            let numeric = property.kind() == PropertyKind::Numeric;
            let placeholder = if numeric { "e.g. 1/60, 2.8, 1600…" } else { "substring…" };
            let comb = string_dropdown(&Combinator::ALL.map(|c| c.label()), comb_i);
            let prop = string_dropdown(&RuleProperty::ALL.map(|p| p.label()), prop_i);
            let cmp_text = cmp_dropdown(&RuleCmp::TEXT_SET, rule_cmp);
            let cmp_num = cmp_dropdown(&RuleCmp::NUMERIC_SET, rule_cmp);
            cmp_text.set_visible(!numeric);
            cmp_num.set_visible(numeric);
            let entry = {
                let e = gtk4::Entry::new();
                e.set_text(&value);
                e.set_placeholder_text(Some(placeholder));
                e
            };
            // Every widget funnels through apply_from; its sync-guard makes the
            // programmatic writes below inert during observer passes. The
            // property handler additionally flips the comparator visibility so
            // a kind change swaps families in place.
            let rows_comb = rows.clone();
            comb.connect_selected_notify(move |_| {
                apply_from(&rows_comb);
            });
            {
                let rows_prop = rows.clone();
                let cmp_text = cmp_text.clone();
                let cmp_num = cmp_num.clone();
                let entry = entry.clone();
                prop.connect_selected_notify(move |p| {
                    let numeric =
                        RuleProperty::from_index(p.selected()).kind() == PropertyKind::Numeric;
                    cmp_text.set_visible(!numeric);
                    cmp_num.set_visible(numeric);
                    entry.set_placeholder_text(Some(if numeric {
                        "e.g. 1/60, 2.8, 1600…"
                    } else {
                        "substring…"
                    }));
                    apply_from(&rows_prop);
                });
            }
            let rows_cmptext = rows.clone();
            cmp_text.connect_selected_notify(move |_| {
                apply_from(&rows_cmptext);
            });
            let rows_cmpnum = rows.clone();
            cmp_num.connect_selected_notify(move |_| {
                apply_from(&rows_cmpnum);
            });
            let rows_entry = rows.clone();
            entry.connect_changed(move |_| {
                apply_from(&rows_entry);
            });
            RuleRow { comb, prop, cmp_text, cmp_num, entry }
        }

        /// Read every row's widgets into canonical `Rule`s. Blank values are
        /// legal state — the SQL composer skips them — so "what the controls
        /// show" round-trips without surprises. The comparator comes from the
        /// row's VISIBLE dropdown, mapped back through its kind's set (the
        /// hidden sibling is ignored, so its stale position never leaks).
        fn collect(rows: &[RuleRow]) -> Vec<Rule> {
            rows.iter()
                .map(|r| {
                    let property = RuleProperty::from_index(r.prop.selected());
                    let cmp = match property.kind() {
                        PropertyKind::Text => {
                            let i = r.cmp_text.selected() as usize;
                            RuleCmp::TEXT_SET.get(i).copied().unwrap_or(RuleCmp::TEXT_SET[0])
                        }
                        PropertyKind::Numeric => {
                            let i = r.cmp_num.selected() as usize;
                            RuleCmp::NUMERIC_SET.get(i).copied().unwrap_or(RuleCmp::NUMERIC_SET[0])
                        }
                    };
                    Rule {
                        property,
                        cmp,
                        comb: Combinator::from_index(r.comb.selected()),
                        value: r.entry.text().to_string(),
                    }
                })
                .collect()
        }

        /// Apply what the widgets currently show, unless we're inside an
        /// observer sync (then this write was programmatic and must not recurse).
        fn apply_from(rows: &Rows) {
            if lighttable::filter_sync_in_progress() {
                return;
            }
            lighttable::set_rule_stack(collect(&rows.borrow()));
        }

        /// Lay out every row (wiring lives in `new_rule_row`; this pass only
        /// positions widgets and mints each row's fresh delete button). Runs on
        /// the add path, the delete path and the observer-rebuild path; never
        /// mid-keystroke.
        fn rebuild_rows(rules_box: &gtk4::Box, rows: &Rows, add_btn: &gtk4::Button) {
            while let Some(child) = rules_box.first_child() {
                rules_box.remove(&child);
            }
            let borrowed = rows.borrow();
            for (i, row) in borrowed.iter().enumerate() {
                let b = gtk4::Box::new(gtk4::Orientation::Horizontal, 4);
                // Row 0 has nothing to combine with yet; its dropdown hides and
                // reappears automatically as rows come and go.
                row.comb.set_visible(i > 0);
                if i > 0 {
                    row.comb.set_margin_start(12);
                }
                // Both comparator dropdowns ride every layout; exactly one is
                // visible (set in new_rule_row and flipped by the property
                // handler) — visibility survives reparenting, so the layout
                // pass never needs to re-derive it.
                for w in [&row.comb, &row.prop, &row.cmp_text, &row.cmp_num] {
                    w.set_size_request(84, -1);
                    b.append(w);
                }
                row.entry.set_hexpand(true);
                // Placeholder NOT touched here: it's kind-dependent (numeric
                // examples vs "substring…") and its owners are `new_rule_row`
                // and the property handler — the only two paths that can change
                // a row's kind. Re-asserting a text placeholder on every layout
                // pass would clobber a numeric row's (senior-review MAJOR-2,
                // m4-135). Like visibility, the property survives reparenting.
                b.append(&row.entry);
                let del = gtk4::Button::from_icon_name("window-close-symbolic");
                del.add_css_class("flat");
                // Delete: drop our row, then lay out + apply the remainder.
                // Capturing `i` is sound because every mutation that shifts
                // indices (add / delete / observer replace) ends in a full
                // rebuild_rows — this button (minted fresh each layout pass)
                // never outlives its own.
                {
                    let rows_del = rows.clone();
                    let rules_box_del = rules_box.clone();
                    let add_btn_del = add_btn.clone();
                    del.connect_clicked(move |_| {
                        debug_assert!(
                            i < rows_del.borrow().len(),
                            "stale rule-row index at delete time",
                        );
                        if i < rows_del.borrow().len() {
                            rows_del.borrow_mut().remove(i);
                            rebuild_rows(&rules_box_del, &rows_del, &add_btn_del);
                            apply_from(&rows_del);
                        }
                    });
                }
                b.append(&del);
                rules_box.append(&b);
            }
            add_btn.set_sensitive(borrowed.len() < rule_stack::MAX_RULES);
        }

        let rules_box = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        let rows: Rows = Rc::new(RefCell::new(Vec::new()));

        let add_btn = gtk4::Button::with_label("+ Add rule");
        add_btn.add_css_class("flat");
        add_btn.set_tooltip_text(Some(
            "Add a rule — rules combine with AND / OR / AND NOT and compose on top of the active collection",
        ));
        add_btn.set_halign(gtk4::Align::Start);
        add_btn.set_margin_start(12);
        add_btn.set_margin_end(12);

        {
            let rows_c = rows.clone();
            let rules_box_c = rules_box.clone();
            let add_btn_c = add_btn.clone();
            add_btn.connect_clicked(move |_| {
                if rows_c.borrow().len() >= rule_stack::MAX_RULES {
                    return;
                }
                let fresh = new_rule_row(None, &rows_c);
                rows_c.borrow_mut().push(fresh);
                rebuild_rows(&rules_box_c, &rows_c, &add_btn_c);
                // An added blank rule is inert SQL-wise (the composer skips
                // it), but applying anyway is what keeps state == widgets: if
                // we skipped it, the next unrelated filter change would see a
                // divergence and rebuild — silently deleting the row the user
                // just added.
                apply_from(&rows_c);
            });
        }

        // Seed whatever the startup token restored (applied in lib.rs BEFORE
        // this panel was built — same restore-before-build contract as the other
        // filters), so a persisted stack shows up without needing an observer
        // pass that would never otherwise fire.
        {
            let restored = lighttable::current_rule_stack();
            if !restored.is_empty() {
                let seeded: Vec<RuleRow> =
                    restored.iter().map(|r| new_rule_row(Some(r), &rows)).collect();
                *rows.borrow_mut() = seeded;
                rebuild_rows(&rules_box, &rows, &add_btn);
            }
        }

        // Observer half: only act on real divergence between the controls and
        // canonical state (e.g. some future second editor of the stack). Equal
        // state — including after our own apply — must not trigger a rebuild,
        // or typing would lose the Entry under the cursor.
        {
            let rows_obs = rows.clone();
            let rules_box_obs = rules_box.clone();
            let add_btn_obs = add_btn.clone();
            lighttable::add_filter_observer(move || {
                if collect(&rows_obs.borrow()) != lighttable::current_rule_stack() {
                    let canonical = lighttable::current_rule_stack();
                    let replaced: Vec<RuleRow> =
                        canonical.iter().map(|r| new_rule_row(Some(r), &rows_obs)).collect();
                    *rows_obs.borrow_mut() = replaced;
                    rebuild_rows(&rules_box_obs, &rows_obs, &add_btn_obs);
                }
            });
        }

        // ── Presets (m4-136): save / recall / delete the WHOLE filter set ──
        // darktable's collection module stores named presets; here one captures
        // all five filter tokens in a single payload string and applying it
        // goes through the real setters, so every control, the grid and the
        // per-key persistence observers fan out exactly as if each control had
        // been clicked by hand.
        //
        // Rows are rebuilt wholesale on every change and their Apply/Delete
        // buttons are minted fresh per pass — same soundness argument as
        // `rebuild_rows`: fresh buttons cannot accumulate handlers.
        fn refresh_preset_rows(list: &gtk4::ListBox, db_path: &str) {
            while let Some(child) = list.first_child() {
                list.remove(&child);
            }
            let presets = crate::persist::load_collection_presets(db_path);
            if presets.is_empty() {
                let row = gtk4::ListBoxRow::new();
                let lbl = gtk4::Label::builder()
                    .label("(no presets saved)")
                    .halign(gtk4::Align::Start)
                    .margin_start(8).margin_end(8).margin_top(2).margin_bottom(2)
                    .build();
                lbl.add_css_class("dim-label");
                row.set_child(Some(&lbl));
                row.set_selectable(false);
                row.set_widget_name("");
                list.append(&row);
                return;
            }
            for (name, payload) in presets {
                let row = gtk4::ListBoxRow::new();
                row.set_selectable(false);
                let b = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
                b.set_margin_start(8);
                b.set_margin_end(8);
                b.set_margin_top(2);
                b.set_margin_bottom(2);
                let lbl = gtk4::Label::builder()
                    .label(&name)
                    .halign(gtk4::Align::Start)
                    .hexpand(true)
                    .ellipsize(gtk4::pango::EllipsizeMode::End)
                    .build();
                b.append(&lbl);
                let apply = gtk4::Button::from_icon_name("object-select-symbolic");
                apply.add_css_class("flat");
                // A structurally corrupt payload (hand-edited/truncated db row)
                // would otherwise fail at apply time with zero feedback — the
                // same silent-failure class the styles section routes through a
                // toast. Here: judge BEFORE wiring and leave a visibly inert
                // button; per-field content garbage still applies (each
                // component decoder falls back to no-filter).
                if lighttable::parse_collection_payload(&payload).is_none() {
                    apply.set_sensitive(false);
                    apply.set_tooltip_text(Some("Preset data unreadable — delete and re-save"));
                } else {
                    apply.set_tooltip_text(Some("Apply this preset to the current filters"));
                }
                {
                    let payload = payload.clone();
                    apply.connect_clicked(move |_| {
                        lighttable::apply_collection_payload(&payload);
                    });
                }
                b.append(&apply);
                let del = gtk4::Button::from_icon_name("window-close-symbolic");
                del.add_css_class("flat");
                del.set_tooltip_text(Some("Delete this preset"));
                {
                    let db_del = db_path.to_string();
                    let name = name.clone();
                    let list_del = list.clone();
                    del.connect_clicked(move |_| {
                        if crate::persist::delete_collection_preset(&db_del, &name) {
                            refresh_preset_rows(&list_del, &db_del);
                        }
                    });
                }
                b.append(&del);
                row.set_child(Some(&b));
                list.append(&row);
            }
        }

        let presets_sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);
        let preset_entry = gtk4::Entry::builder()
            .placeholder_text("Save filters as…")
            .hexpand(true)
            .build();
        let preset_save_btn = gtk4::Button::with_label("Save");
        preset_save_btn.add_css_class("flat");
        // Insensitive while blank: "no name" is the ONLY input failure, and a
        // dead button documents it without needing a toast channel up here.
        preset_save_btn.set_sensitive(false);
        {
            let btn = preset_save_btn.clone();
            preset_entry.connect_changed(move |e| {
                btn.set_sensitive(!e.text().trim().is_empty());
            });
        }

        let presets_list = gtk4::ListBox::new();
        presets_list.add_css_class("boxed");
        presets_list.set_selection_mode(gtk4::SelectionMode::None);
        refresh_preset_rows(&presets_list, db_path);

        let preset_save_row = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
        preset_save_row.set_margin_start(12);
        preset_save_row.set_margin_end(12);
        preset_save_row.append(&preset_entry);
        preset_save_row.append(&preset_save_btn);

        {
            let entry_c = preset_entry.clone();
            let list_c = presets_list.clone();
            let db_c = db_path.to_string();
            preset_save_btn.connect_clicked(move |_| {
                let name = entry_c.text().trim().to_string();
                if name.is_empty() {
                    return;
                }
                // Upserts on name collision, matching styles' documented
                // behaviour. A `false` here means no catalogue is open (demo
                // mode's empty prefs path) — the button was reachable but there
                // is nowhere to store; that state also shows "(no presets
                // saved)" forever, which says the same thing. Known gap until
                // this panel gains a toast channel: a locked or read-only
                // catalogue file fails just as silently (senior-review m4,
                // m4-136).
                if crate::persist::save_collection_preset(
                    &db_c,
                    &name,
                    &lighttable::collection_filter_payload(),
                ) {
                    entry_c.set_text("");
                    refresh_preset_rows(&list_c, &db_c);
                }
            });
        }
        {
            let btn_act = preset_save_btn.clone();
            preset_entry.connect_activate(move |_| {
                btn_act.emit_clicked();
            });
        }

        content.append(&collapsible_section(
            &filters_header,
            &[
                filters_sep.clone().upcast::<gtk4::Widget>(),
                aspect_drop.clone().upcast(),
                rules_box.clone().upcast(),
                add_btn.clone().upcast(),
                presets_sep.clone().upcast(),
                preset_save_row.clone().upcast(),
                presets_list.clone().upcast(),
            ]
            .iter()
            .collect::<Vec<_>>(),
            true,
            db_path,
            FILTERS_SECTION_PREF_KEY,
        ));

        // ── Tags ──────────────────────────────────────────────────────────
        // The header/separator/box are always present; their visibility tracks
        // whether the library has any user tags (toggled in `refresh_tags`).
        let tags_header = section_header("Tags");
        let tags_sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);


        let db_tags = db_path.to_string();
        let at_tag = active_tag.clone();
        tag_box.connect_row_activated(
            clone!(@weak lt_model, @weak list_box => move |_, row| {
            list_box.unselect_all();
            // The colour quick-filter is NOT dropped here either: like the star
            // filter it composes on top of the collection (m4-126).
            // The full `parent|child` path is encoded in the row's widget name
            // (see append_tag_tree_row) for both real and virtual nodes. Clicking
            // either filters to that tag plus its whole hierarchical subtree.
            let prefix = row.widget_name().to_string();
            if !prefix.is_empty() {
                *at_tag.borrow_mut() = Some(prefix.clone());   // remember the filter
                lighttable_load_by_tag_prefix(&lt_model, &db_tags, &prefix);
            }
        }));
        content.append(&collapsible_section(
            &tags_header,
            &[tags_sep.clone().upcast::<gtk4::Widget>(), tag_box.clone().upcast()]
                .iter()
                .collect::<Vec<_>>(),
            true,
            db_path,
            TAGS_SECTION_PREF_KEY,
        ));

        let scroll = gtk4::ScrolledWindow::builder()
            .hscrollbar_policy(gtk4::PolicyType::Never)
            .vscrollbar_policy(gtk4::PolicyType::Automatic)
            .child(&content)
            .vexpand(true)
            .build();
        panel.append(&scroll);

        let tags = TagPanel {
            tag_box,
            tags_header,
            tags_sep,
            db_path: db_path.to_string(),
            on_tags_changed: std::rc::Rc::new(std::cell::RefCell::new(None)),
        };
        let lp = Self {
            widget: panel,
            list_box,
            tags,
        };
        lp.tags.refresh_tags();
        lp
    }

    /// Drop the selection highlight from the folder and tag list boxes — the two
    /// COLLECTION selectors. Called from lib.rs when something *outside* the left
    /// panel takes over the grid — a name search or an import/reset — so a
    /// highlighted row doesn't outlive the collection it stood for.
    ///
    /// The colour quick-filter is deliberately NOT touched: since m4-126 it is a
    /// compose-on-top filter (like the bottom bar's stars), not a collection, so
    /// it stays in force across these reloads and its mirrors keep telling the
    /// truth.
    ///
    /// Invariant: call this exactly on the paths that *supersede* the active
    /// collection (i.e. null `active_tag`). Do NOT call it from a path that reloads
    /// the grid while *preserving* the collection (e.g. `reapply_tag_filter`), or
    /// the highlight would be wrongly cleared from a filter that's still in force.
    pub fn clear_filter_highlights(&self) {
        self.list_box.unselect_all();
        self.tags.tag_box.unselect_all();
    }

    /// Register a callback fired after a tag is renamed or deleted here, so the
    /// metadata panel can re-render the current image's chips. Delegates to the
    /// [`TagPanel`]; see [`TagPanel::set_on_tags_changed`].
    pub fn set_on_tags_changed<F: Fn() + 'static>(&self, f: F) {
        self.tags.set_on_tags_changed(f);
    }

    /// Rebuild the Tags section in place from the current library state. Delegates
    /// to the [`TagPanel`]; see [`TagPanel::refresh_tags`].
    pub fn refresh_tags(&self) {
        self.tags.refresh_tags();
    }
}

impl TagPanel {
    /// Register a callback fired after a tag is renamed or deleted here, so the
    /// metadata panel can re-render the current image's chips. Mirror of
    /// `MetadataPanel::set_on_tags_changed`; replaces any previous callback. The
    /// callback must not re-enter a left-panel tag mutation, or it would loop.
    fn set_on_tags_changed<F: Fn() + 'static>(&self, f: F) {
        *self.on_tags_changed.borrow_mut() = Some(std::rc::Rc::new(f));
    }

    /// Fire the tags-changed notify, if set (clone out of the cell first so it
    /// isn't borrowed while the callback runs).
    fn fire_tags_changed(&self) {
        let cb = self.on_tags_changed.borrow().clone();
        if let Some(cb) = cb { cb(); }
    }

    /// Rebuild the Tags section in place from the current library state.
    ///
    /// Clears and repopulates only the tag rows (the box itself is stable, so
    /// the activation handler bound in `new` keeps working), then shows or hides
    /// the section depending on whether any user tags exist. Safe to call after
    /// a tag is attached from the metadata panel to surface new tags / counts.
    fn refresh_tags(&self) {
        while let Some(child) = self.tag_box.first_child() {
            self.tag_box.remove(&child);
        }
        let tags = load_tags_with_counts(&self.db_path);
        let rows = flatten_tag_tree(&tags);
        for r in &rows {
            self.append_tag_tree_row(r);
        }
        let has_tags = !rows.is_empty();
        self.tags_header.set_visible(has_tags);
        self.tags_sep.set_visible(has_tags);
        self.tag_box.set_visible(has_tags);
    }

    /// Append one hierarchical tag row, indented by its depth in the `|`-tree.
    ///
    /// Every row stashes its full `parent|child` path in the widget name so the
    /// activation handler can filter to that path plus its subtree (see the
    /// `tag_box` row-activated handler in `new`). A **real** tag (`row.id` is
    /// `Some`) additionally shows its count and gets a secondary-click
    /// rename/delete popover. A **virtual** parent (a path prefix with no tag of
    /// its own) is rendered dim and stays count-less and menu-less, but is still
    /// clickable — activating it filters the grid to its whole subtree.
    fn append_tag_tree_row(&self, row_data: &TagTreeRow) {
        let row = gtk4::ListBoxRow::new();

        // Indent by depth; the base margin matches the old flat rows (12px).
        let indent = 12 + (row_data.depth as i32) * 16;
        let hbox = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(8)
            .margin_start(indent).margin_end(8)
            .margin_top(6).margin_bottom(6)
            .build();

        let name_lbl = gtk4::Label::builder()
            .label(&row_data.label)
            .halign(gtk4::Align::Start)
            .hexpand(true)
            .ellipsize(gtk4::pango::EllipsizeMode::Middle)
            .build();
        hbox.append(&name_lbl);

        // The full `parent|child` path drives prefix filtering on click (real and
        // virtual rows alike); stash it in the widget name for the activation
        // handler. flatten_tag_tree never emits an empty `full_name`, so the
        // handler's non-empty guard always admits a genuine tag row.
        row.set_widget_name(&row_data.full_name);

        let Some(id) = row_data.id else {
            // Virtual parent: dim and count-less (a path prefix with no tag of its
            // own, so nothing to rename/delete), but still clickable — activating
            // it filters the grid to the whole subtree under this prefix.
            name_lbl.add_css_class("dim-label");
            row.set_tooltip_text(Some(&row_data.full_name));
            row.set_child(Some(&hbox));
            self.tag_box.append(&row);
            return;
        };

        let count_lbl = gtk4::Label::builder()
            .label(&row_data.count.to_string())
            .halign(gtk4::Align::End)
            .build();
        count_lbl.add_css_class("dim-label");
        count_lbl.add_css_class("numeric");
        hbox.append(&count_lbl);

        row.set_tooltip_text(Some(&row_data.full_name));
        row.set_child(Some(&hbox));

        // Secondary-click → rename/delete popover. The gesture lives as long as
        // the row (so for the app lifetime while the row is in `tag_box`); to
        // avoid a strong-ref cycle (tag_box→row→gesture→TagPanel→tag_box) it
        // captures only weak widget refs (+ the leaf `db`/`notify`) and rebuilds a
        // transient `TagPanel` on demand. Removed rows then free cleanly on refresh.
        // Since `TagPanel` holds ONLY the tag fields `show_tag_menu` needs (m4-27),
        // the reconstruction supplies nothing the menu ignores. The popover operates
        // on the FULL tag path (rename/delete are per-tag, hierarchy-unaware here).
        let gesture = gtk4::GestureClick::new();
        gesture.set_button(gtk4::gdk::BUTTON_SECONDARY);
        let tag_box_w = self.tag_box.downgrade();
        let header_w  = self.tags_header.downgrade();
        let sep_w     = self.tags_sep.downgrade();
        let db        = self.db_path.clone();
        // The notify Rc never references back at the panel widgets, so capturing it
        // strongly introduces no cycle.
        let notify     = self.on_tags_changed.clone();
        let name_owned = row_data.full_name.clone();
        let count      = row_data.count;
        let row_w     = row.downgrade();
        gesture.connect_pressed(move |g, _, x, y| {
            g.set_state(gtk4::EventSequenceState::Claimed);
            if let (Some(tag_box), Some(tags_header), Some(tags_sep), Some(row)) = (
                tag_box_w.upgrade(), header_w.upgrade(), sep_w.upgrade(), row_w.upgrade(),
            ) {
                let tp = TagPanel {
                    tag_box, tags_header, tags_sep,
                    db_path: db.clone(), on_tags_changed: notify.clone(),
                };
                tp.show_tag_menu(&row, id, &name_owned, count, x, y);
            }
        });
        row.add_controller(gesture);

        self.tag_box.append(&row);
    }

    /// Pop up the rename/delete menu for a tag row at the click point.
    fn show_tag_menu(&self, row: &gtk4::ListBoxRow, id: u32, name: &str, count: i64, x: f64, y: f64) {
        let popover = gtk4::Popover::builder().build();
        popover.set_parent(row);
        popover.set_pointing_to(Some(&gtk4::gdk::Rectangle::new(x as i32, y as i32, 1, 1)));

        let vbox = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(6)
            .margin_start(6).margin_end(6).margin_top(6).margin_bottom(6)
            .build();

        // Edit only this node's own segment (the last `|` component); the parent
        // prefix is fixed and the whole subtree moves with it (see do_rename).
        let segment = name.rsplit_once('|').map(|(_, s)| s).unwrap_or(name);
        let entry = gtk4::Entry::builder().text(segment).build();
        vbox.append(&entry);

        let hint = gtk4::Label::builder()
            .label("Renames this tag and any sub-tags")
            .halign(gtk4::Align::Start)
            .build();
        hint.add_css_class("dim-label");
        hint.add_css_class("caption");
        vbox.append(&hint);

        // Inline failure message (m4-32): a rename collision keeps the popover
        // open and shows why here, rather than dismissing to a modal. Hidden
        // until a rename actually fails.
        let error_label = gtk4::Label::builder()
            .halign(gtk4::Align::Start)
            .wrap(true)
            .max_width_chars(28)
            .visible(false)
            .build();
        error_label.add_css_class("error");
        error_label.add_css_class("caption");
        vbox.append(&error_label);
        // a11y: relate the error label to the entry so a screen reader announces
        // the reason when focus moves to the entry on a failed rename. Per ARIA,
        // an error message is only surfaced while the field is marked invalid, so
        // the Invalid state is toggled with the label's visibility below.
        entry.update_relation(&[gtk4::accessible::Relation::ErrorMessage(&[
            error_label.upcast_ref::<gtk4::Accessible>(),
        ])]);

        let rename_btn = gtk4::Button::with_label("Rename");
        rename_btn.add_css_class("suggested-action");
        vbox.append(&rename_btn);

        vbox.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

        let plural = if count == 1 { "" } else { "s" };
        let delete_btn = gtk4::Button::with_label(&format!("Delete (from {count} image{plural})"));
        delete_btn.add_css_class("destructive-action");
        vbox.append(&delete_btn);

        popover.set_child(Some(&vbox));

        // Rename on button click or Enter in the entry (skipped if blank or
        // unchanged — no needless write/refresh).
        //
        // Weak captures for `pop`/`entry`/`err_lbl`: all three live in THIS
        // popover's subtree, and this closure is stored in the button/entry signal
        // handlers (also in the subtree), so capturing them strongly would form a
        // self-cycle (e.g. popover→button→handler→closure→pop→popover) that keeps
        // the whole popover alive after `unparent` — leaking one popover per
        // right-click. `lp` stays strong: `TagPanel`'s `tag_box` lives OUTSIDE the
        // popover subtree, so it can't close a cycle back to the popover, which is
        // then freed on unparent. Mirrors `append_tag_tree_row`'s row-gesture
        // discipline. Upgrades fail only if the popover is already gone → no-op.
        let do_rename = {
            let lp = self.clone();
            let pop_w = popover.downgrade();
            let entry_w = entry.downgrade();
            let err_lbl_w = error_label.downgrade();
            let old_full = name.to_string();
            move || {
                let (Some(pop), Some(entry), Some(err_lbl)) =
                    (pop_w.upgrade(), entry_w.upgrade(), err_lbl_w.upgrade())
                else { return };
                // respliced_tag_path re-attaches the fixed parent prefix and
                // returns None for a blank/unchanged/`|` segment (a no-op edit —
                // treat as cancel and just dismiss).
                let new_segment = entry.text().to_string();
                let Some(new_full) = respliced_tag_path(&old_full, &new_segment) else {
                    pop.popdown();
                    return;
                };
                match lp.write_tag_rename(&old_full, &new_full) {
                    // Success: dismiss BEFORE refresh_tags removes the popover's
                    // parent row, so no orphaned subtree exists mid-call.
                    Ok(()) => {
                        pop.popdown();
                        lp.refresh_tags();
                        lp.fire_tags_changed();
                    }
                    // Failure (UNIQUE clash → atomic rollback, or db open error):
                    // nothing changed, so keep the popover open, show why, and
                    // refocus the entry so the user can correct the name in place.
                    Err(e) => {
                        eprintln!("darkroom: tag rename failed: {e}");
                        err_lbl.set_text(&rename_failure_message(&e, &new_full));
                        err_lbl.set_visible(true);
                        entry.update_state(&[gtk4::accessible::State::Invalid(
                            gtk4::AccessibleInvalidState::True,
                        )]);
                        entry.grab_focus();
                        entry.select_region(0, -1);
                    }
                }
            }
        };
        rename_btn.connect_clicked({
            let f = do_rename.clone();
            move |_| f()
        });
        entry.connect_activate({
            let f = do_rename.clone();
            move |_| f()
        });
        // Clear a stale failure message once the user edits the name — the label
        // should reflect only the last submit, not linger during correction. The
        // closure param IS this entry, so nothing extra is captured (only a weak
        // ref to the label, which lives in the same popover subtree).
        entry.connect_changed({
            let err_lbl = error_label.downgrade();
            move |e| {
                if let Some(l) = err_lbl.upgrade() {
                    l.set_visible(false);
                }
                // Clear the a11y invalid marker in step with the hidden label.
                e.update_state(&[gtk4::accessible::State::Invalid(
                    gtk4::AccessibleInvalidState::False,
                )]);
            }
        });

        // Delete (confirmed) on button click. Same weak-`pop` discipline as the
        // rename handler (this closure lives on `delete_btn`, a popover child, so a
        // strong `pop` would self-cycle and leak the popover); `lp` stays strong.
        {
            let lp = self.clone();
            let pop_w = popover.downgrade();
            let name_owned = name.to_string();
            delete_btn.connect_clicked(move |_| {
                if let Some(pop) = pop_w.upgrade() {
                    pop.popdown();
                }
                lp.confirm_delete_tag(id, &name_owned, count);
            });
        }

        // The popover is parented to the row; unparent it when dismissed so it
        // doesn't outlive (and leak against) the row it points at.
        popover.connect_closed(|p| p.unparent());
        popover.popup();
    }

    /// Write half of a subtree rename: open the library db and run the atomic
    /// `tag_rename_subtree` UPDATE (rewriting the whole `old_full`→`new_full`
    /// subtree so descendants move with it), returning the DB outcome and
    /// touching NO UI. `Ok(())` = the rewrite committed; `Err` = nothing changed
    /// — either a UNIQUE-name clash (the destination path or a descendant's
    /// already exists; the UPDATE is atomic so it rolls back leaving every tag
    /// unchanged) or the db couldn't open. The caller (the rename popover)
    /// orchestrates the UI: on `Ok` it dismisses BEFORE `refresh_tags` (so the
    /// popover's parent row isn't removed mid-call → no orphaned subtree), on
    /// `Err` it keeps the popover open and shows [`rename_failure_message`] inline
    /// (m4-32). An empty `db_path` (demo mode, no library) is a no-op `Ok`.
    fn write_tag_rename(&self, old_full: &str, new_full: &str) -> rusqlite::Result<()> {
        if self.db_path.is_empty() { return Ok(()); }
        // Full open_catalog (not the session opener): a rare user-initiated write
        // self-heals the durable schema if the startup bootstrap warned-and-continued.
        let conn = c41_db::schema::open_catalog(&self.db_path)?;
        c41_db::tags::tag_rename_subtree(&conn, old_full, new_full)?;
        Ok(())
    }

    /// Confirm, then delete a tag and all its image associations.
    fn confirm_delete_tag(&self, id: u32, name: &str, count: i64) {
        let plural = if count == 1 { "" } else { "s" };
        let dialog = adw::AlertDialog::new(
            Some("Delete tag?"),
            Some(&format!(
                "\u{201c}{name}\u{201d} will be removed from {count} image{plural}. \
                 This cannot be undone."
            )),
        );
        dialog.add_responses(&[("cancel", "Cancel"), ("delete", "Delete")]);
        dialog.set_response_appearance("delete", adw::ResponseAppearance::Destructive);
        dialog.set_default_response(Some("cancel"));
        dialog.set_close_response("cancel");
        let lp = self.clone();
        dialog.connect_response(None, move |_, resp| {
            if resp == "delete" { lp.delete_tag(id); }
        });
        // Any widget in the tree resolves the root window; `tag_box` is always in
        // it (TagPanel holds no top-level `widget` ref — m4-27).
        dialog.present(Some(&self.tag_box));
    }

    /// Delete a tag and its associations (best-effort; logs faults), then refresh.
    ///
    /// The active filter is a tag **path** (not an id), so a delete can't leave a
    /// dangling id-reuse hazard the way the old id-based filter could. We simply
    /// refresh the tree and fire the change notify: the wired `reapply` re-runs
    /// the current hierarchical-prefix filter, so deleting a parent keeps its
    /// surviving descendants on screen, while deleting the exact filtered-on leaf
    /// collapses to the empty-result placeholder (the user clicks a folder/All to
    /// leave it).
    fn delete_tag(&self, id: u32) {
        if self.db_path.is_empty() { return; }
        // Full open_catalog: a rare write self-heals the durable schema (vs the
        // session opener the read-hot paths use); see write_tag_rename.
        match c41_db::schema::open_catalog(&self.db_path) {
            Ok(conn) => {
                if let Err(e) = c41_db::tags::tag_delete(&conn, id) {
                    eprintln!("darkroom: tag delete failed: {e}");
                }
            }
            Err(e) => eprintln!("darkroom: cannot open library db to delete tag: {e}"),
        }
        self.refresh_tags();
        self.fire_tags_changed();
    }
}

/// Compute the new full tag path when a node's own segment is renamed in place.
/// `full_name` is the node's `parent|child` path; `new_segment` is the user's
/// replacement for the LAST segment (trimmed here). Returns the rewritten full
/// path, or `None` when the edit is a no-op — blank input or the segment is
/// unchanged — so the caller skips the DB write. A `new_segment` containing the
/// `|` hierarchy separator is rejected (also `None`): this popover renames a node
/// in place, so re-parenting/deepening the tree via a typed `|` is out of scope —
/// and forbidding it also rules out a rewrite that could self-collide against a
/// row it is itself moving. The parent prefix is preserved verbatim; a top-level
/// tag (no `|`) just becomes `new_segment`. Kept as a free function so the
/// (display-bound) rename popover has a unit-testable core.
/// `c41_ui_prefs` keys for the left panel's section fold state (parity 3.2).
/// One per section; the value uses the same `shown`/`hidden` encoding as the
/// side-panel collapse keys in `lib.rs`.
const IMPORT_SECTION_PREF_KEY: &str = "left_section_import";
const COLLECTIONS_SECTION_PREF_KEY: &str = "left_section_collections";
const COLOURS_SECTION_PREF_KEY: &str = "left_section_colours";
/// The m4-128 "Collection filters" section's fold state.
const FILTERS_SECTION_PREF_KEY: &str = "left_section_filters";
const TAGS_SECTION_PREF_KEY: &str = "left_section_tags";

fn respliced_tag_path(full_name: &str, new_segment: &str) -> Option<String> {
    let new_segment = new_segment.trim();
    if new_segment.is_empty() || new_segment.contains('|') {
        return None;
    }
    let (parent, segment) = match full_name.rsplit_once('|') {
        Some((p, s)) => (Some(p), s),
        None => (None, full_name),
    };
    if new_segment == segment {
        return None;
    }
    Some(match parent {
        Some(p) => format!("{p}|{new_segment}"),
        None => new_segment.to_string(),
    })
}

/// Turn a `tag_rename_subtree` failure into a user-facing message. A SQLite
/// `ConstraintViolation` means the UNIQUE `data.tags.name` index rejected the
/// rewrite — some rewritten path already exists — so the atomic UPDATE rolled
/// back and nothing changed; that common case gets its own wording. NOTE the
/// clash may be a *descendant's* new path, not `new_full` itself (renaming
/// `places`→`europe` can fail because `europe|Italy` exists though no `europe`
/// does), so the message says "would clash" rather than asserting `new_full`
/// itself already exists. Any other DB error gets a generic message. Pure so the
/// rename popover's (display-bound) inline error label — set in `show_tag_menu`
/// on a failed [`write_tag_rename`](TagPanel::write_tag_rename) — has a
/// unit-testable core.
fn rename_failure_message(err: &rusqlite::Error, new_full: &str) -> String {
    match err {
        rusqlite::Error::SqliteFailure(e, _)
            if e.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            format!(
                "Renaming to \u{201c}{new_full}\u{201d} would clash with an \
                 existing tag. Nothing was renamed."
            )
        }
        _ => format!("The tag could not be renamed to \u{201c}{new_full}\u{201d}."),
    }
}

/// One row of the hierarchical tag display, in pre-order render order.
/// `depth` is the nesting level (0 = top), `label` the path segment shown, and
/// `full_name` the cumulative `a|b|c` path (used by the rename/delete popover and
/// later by prefix filtering). `id`/`count` are set only for a **real** tag; a
/// **virtual** parent (a path prefix with no tag of its own) has `id: None` and
/// `count: 0` — descendant counts are deliberately NOT summed, since an image
/// carrying two sibling tags would be double-counted.
#[derive(Debug, Clone, PartialEq)]
struct TagTreeRow {
    depth:     usize,
    label:     String,
    full_name: String,
    id:        Option<u32>,
    count:     i64,
}

/// Build the pre-order hierarchical display rows from the flat `(id, name, count)`
/// tag list. Names use `|` as the hierarchy separator (darktable convention);
/// children are alphabetised, and intermediate path segments with no tag of their
/// own become virtual parent rows so a `places|Italy` tag still shows a `places`
/// group even when `places` itself is untagged.
fn flatten_tag_tree(tags: &[(u32, String, i64)]) -> Vec<TagTreeRow> {
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Node {
        children: BTreeMap<String, Node>,
        tag:      Option<(u32, i64)>,   // (id, count) when a real tag ends here
    }

    let mut root = Node::default();
    for (id, name, count) in tags {
        // Skip empty segments so a malformed name ("a|", "a||b", "|a") collapses
        // to its meaningful ancestry rather than rendering a blank-label row —
        // which would otherwise carry a working count + rename/delete popover on
        // a tag the user can't see. A name that is all separators yields nothing.
        let mut segs = name.split('|').filter(|s| !s.is_empty()).peekable();
        if segs.peek().is_none() {
            continue;
        }
        let mut cur = &mut root;
        while let Some(seg) = segs.next() {
            cur = cur.children.entry(seg.to_string()).or_default();
            if segs.peek().is_none() {
                // Safe last-write: `data.tags.name` is UNIQUE, so no two distinct
                // ids ever resolve to the same full path (no real-tag clobber).
                cur.tag = Some((*id, *count));
            }
        }
    }

    fn dfs(node: &Node, depth: usize, prefix: &str, out: &mut Vec<TagTreeRow>) {
        for (seg, child) in &node.children {
            let full_name = if prefix.is_empty() {
                seg.clone()
            } else {
                format!("{prefix}|{seg}")
            };
            let (id, count) = match child.tag {
                Some((id, count)) => (Some(id), count),
                None => (None, 0),
            };
            out.push(TagTreeRow { depth, label: seg.clone(), full_name: full_name.clone(), id, count });
            dfs(child, depth + 1, &full_name, out);
        }
    }

    let mut out = Vec::new();
    dfs(&root, 0, "", &mut out);
    out
}

/// A section heading label styled like the panel headers.
fn section_header(text: &str) -> gtk4::Label {
    let header = gtk4::Label::builder()
        .label(text)
        .halign(gtk4::Align::Start)
        .margin_top(12).margin_bottom(6)
        .margin_start(12).margin_end(12)
        .build();
    header.add_css_class("heading");
    header
}

/// Wrap a section's widgets in a darktable-style collapsible (parity 3.2).
///
/// darktable's panels are stacks of expanders: each section shows a disclosure
/// triangle and a lowercase title, and clicking the title folds the section
/// away. Ours were flat labels with the content always visible, which is why a
/// panel with several sections needs far more scrolling than darktable's.
///
/// Takes the already-built header label and content widgets so callers keep
/// their existing references (the tag section, for one, toggles its header's
/// visibility) — this only changes where those widgets are *parented*.
///
/// `pref_key` persists the fold state across sessions in `c41_ui_prefs`, using
/// the same `shown`/`hidden` encoding as the side-panel keys; `default_expanded`
/// applies on first run or if the stored token is unrecognised. Pass an empty
/// `db_path` to skip persistence (tests, or a panel built before the DB opens).
fn collapsible_section(
    header: &gtk4::Label,
    content: &[&gtk4::Widget],
    default_expanded: bool,
    db_path: &str,
    pref_key: &str,
) -> gtk4::Box {
    // Restore first: a stored token wins over the caller's default.
    let expanded = if db_path.is_empty() {
        default_expanded
    } else {
        crate::persist::load_ui_pref(db_path, pref_key)
            .and_then(|t| crate::parse_collapsed_token(&t))
            .map(|collapsed| !collapsed)
            .unwrap_or(default_expanded)
    };
    let outer = gtk4::Box::new(gtk4::Orientation::Vertical, 0);

    // The clickable title row: triangle + the caller's label.
    let title_row = gtk4::Box::new(gtk4::Orientation::Horizontal, 4);
    title_row.add_css_class("c41-section-title");
    let arrow = gtk4::Image::from_icon_name("pan-down-symbolic");
    arrow.set_valign(gtk4::Align::Center);
    title_row.append(&arrow);
    title_row.append(header);

    let body = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    for w in content {
        body.append(*w);
    }
    body.set_visible(expanded);
    if !expanded {
        arrow.set_icon_name(Some("pan-end-symbolic"));
    }

    let click = gtk4::GestureClick::new();
    let body_c = body.clone();
    let arrow_c = arrow.clone();
    let db = db_path.to_string();
    let key = pref_key.to_string();
    click.connect_released(move |_, _, _, _| {
        let now = !body_c.is_visible();
        body_c.set_visible(now);
        arrow_c.set_icon_name(Some(if now { "pan-down-symbolic" } else { "pan-end-symbolic" }));
        if !db.is_empty() {
            crate::persist::save_ui_pref(&db, &key, crate::collapsed_token(!now));
        }
    });
    title_row.add_controller(click);

    outer.append(&title_row);
    outer.append(&body);
    outer
}

fn load_tags_with_counts(db_path: &str) -> Vec<(u32, String, i64)> {
    if db_path.is_empty() {
        return Vec::new();
    }
    // Log faults so a corrupt/locked catalog reads differently from "no tags"
    // (the Tags section is simply hidden either way, but the cause is recoverable
    // from the logs — the established read-path discipline). Session-only open:
    // this read-hot path skips the durable-schema DDL, bootstrapped at startup.
    match c41_db::schema::open_catalog_session(db_path) {
        Ok(conn) => match c41_db::tags::tag_list_with_counts(&conn) {
            Ok(tags) => tags,
            Err(e) => {
                eprintln!("darkroom: tag list query failed: {e}");
                Vec::new()
            }
        },
        Err(e) => {
            eprintln!("darkroom: cannot open library db for tags: {e}");
            Vec::new()
        }
    }
}

fn append_roll_row(list_box: &gtk4::ListBox, label: &str, count: i64, folder: Option<&str>) {
    let row = gtk4::ListBoxRow::new();
    // Encode the folder in the widget name; "all" means no filter
    row.set_widget_name(folder.unwrap_or("all"));

    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .margin_start(12).margin_end(8)
        .margin_top(6).margin_bottom(6)
        .build();

    let name_str = if folder.is_none() {
        label.to_string()
    } else {
        label.rsplit('/').next().unwrap_or(label).to_string()
    };
    let name_lbl = gtk4::Label::builder()
        .label(&name_str)
        .halign(gtk4::Align::Start)
        .hexpand(true)
        .ellipsize(gtk4::pango::EllipsizeMode::Middle)
        .build();

    hbox.append(&name_lbl);

    if count >= 0 {
        let count_lbl = gtk4::Label::builder()
            .label(&count.to_string())
            .halign(gtk4::Align::End)
            .build();
        count_lbl.add_css_class("dim-label");
        count_lbl.add_css_class("numeric");
        hbox.append(&count_lbl);
    }

    if folder.is_some() {
        row.set_tooltip_text(Some(label));
    }
    row.set_child(Some(&hbox));
    list_box.append(&row);
}

/// Human-readable names for the colour-label filter rows, indexed by colour
/// (0 red … 4 purple) to match `c41_db::colorlabels` and the grid dots.
const COLOR_NAMES: [&str; COLOR_COUNT as usize] =
    ["Red", "Yellow", "Green", "Blue", "Purple"];

/// Name for colour index `idx`, or `None` if out of range. Pure (no GTK) so the
/// index↔name mapping has a unit-testable seam under the display-free discipline.
fn color_filter_name(idx: u8) -> Option<&'static str> {
    COLOR_NAMES.get(idx as usize).copied()
}

/// Append one colour-label filter check: a `CheckButton` whose child is a coloured
/// dot plus its name. The colour index is stashed in the check's widget name so
/// [`color_mask_from_box`] can read the mask back off the active checks. Independent
/// (not a Single-selection row) so several colours can be combined (m4-26).
fn append_color_check(color_box: &gtk4::Box, idx: u8) {
    let check = gtk4::CheckButton::builder()
        .margin_start(12).margin_end(8)
        .margin_top(2).margin_bottom(2)
        .build();
    check.set_widget_name(&idx.to_string());

    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .build();

    let dot = gtk4::Label::new(None);
    dot.set_markup(&color_dot_markup(idx, true));
    hbox.append(&dot);

    let name_lbl = gtk4::Label::builder()
        .label(color_filter_name(idx).unwrap_or(""))
        .halign(gtk4::Align::Start)
        .hexpand(true)
        .build();
    hbox.append(&name_lbl);

    check.set_child(Some(&hbox));
    color_box.append(&check);
}

/// Read the colour-label mask off `color_box`'s active checks: bit `c` set iff the
/// check for colour `c` is ticked. The index is parsed from each check's widget
/// name (stamped by [`append_color_check`]); an unparseable name is skipped (it
/// can't occur from `append_color_check`, but we never want a stray child to panic
/// the mask read).
fn color_mask_from_box(color_box: &gtk4::Box) -> u8 {
    let mut mask = 0u8;
    let mut child = color_box.first_child();
    while let Some(w) = child {
        if let Some(check) = w.downcast_ref::<gtk4::CheckButton>() {
            if check.is_active() {
                if let Ok(idx) = check.widget_name().parse::<u8>() {
                    mask |= 1 << idx;
                }
            }
        }
        child = w.next_sibling();
    }
    mask
}

/// Repaint the colour section's mirrors (checks + Any/All toggle) from the
/// canonical quick-filter state. The single display-write shared by the startup
/// seed ([`seed_colour_controls`]) and the filter observer, so both stay in exact
/// step with `lighttable::current_colour_mask/_all`. Runs under the observer bus's
/// sync guard whenever the bus invokes it; called bare at startup only because no
/// handler is connected yet at that point.
///
/// Colour indices come from each check's widget name (stamped by
/// [`append_color_check`]), the same source [`color_mask_from_box`] reads — one
/// indexing strategy over the children, so a stray non-check child could never
/// silently misalign the two directions.
fn sync_colour_controls_display(color_box: &gtk4::Box, mode_toggle: &gtk4::ToggleButton) {
    let mask = lighttable::current_colour_mask();
    let all = lighttable::current_colour_all();
    let mut child = color_box.first_child();
    while let Some(w) = child {
        if let Some(check) = w.downcast_ref::<gtk4::CheckButton>() {
            if let Ok(idx) = check.widget_name().parse::<u8>() {
                check.set_active(mask & (1 << idx) != 0);
            }
        }
        child = w.next_sibling();
    }
    mode_toggle.set_label(if all { "Match all" } else { "Match any" });
    // Fires `toggled` only on change; the handlers skip it regardless when the bus
    // is mid-pass.
    mode_toggle.set_active(all);
}

/// Seed the colour mirrors from the restored canonical state BEFORE any handler is
/// connected (the bottom-bar comparator's seeding contract: a programmatic write
/// with nothing listening fires nothing).
fn seed_colour_controls(color_box: &gtk4::Box, mode_toggle: &gtk4::ToggleButton) {
    sync_colour_controls_display(color_box, mode_toggle);
}

fn load_film_rolls(db_path: &str) -> Vec<(String, i64)> {
    let conn = if db_path.is_empty() {
        match rusqlite::Connection::open_in_memory() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        }
    } else {
        match rusqlite::Connection::open(db_path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        }
    };

    let mut rolls = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT f.folder, COUNT(i.id) \
         FROM main.film_rolls f \
         LEFT JOIN main.images i ON i.film_id = f.id \
         GROUP BY f.id, f.folder \
         ORDER BY f.folder",
    ) {
        let _ = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))
            .map(|rows| rolls.extend(rows.flatten()));
    }
    rolls
}

// ── Right panel (metadata + tags) ────────────────────────────────────────

/// Getter handed to [`MetadataPanel::wire_styles`]: the selected image's path
/// and its current edit, read at click time. Named because the same type is
/// spelled again at the call site in `lib.rs`.
pub type StyleParamsGetter =
    std::rc::Rc<dyn Fn() -> Option<(String, crate::preview::PreviewParams)>>;

/// Per-entry editing target for the metadata editor: `(paths, db_path,
/// original text)` captured when the entry gained focus. m4-145: `paths` is
/// the lighttable multi-selection (darktable's semantics — an edit lands on
/// every selected image), or the active image alone — see
/// `metadata_target_paths` for exactly which. A snapshot may briefly outlive a
/// collection prune (filter/folder change) until the reload-triggered
/// `update()` re-baselines; that matches upstream, which likewise acts on ids
/// captured at its last gui_update. See `MetadataPanel::new`.
type MetaTarget = std::rc::Rc<std::cell::RefCell<(Vec<String>, String, String)>>;

/// The images an edit made RIGHT NOW would land on: the whole multi-selection
/// WHEN it contains the image on screen, else just that image alone.
///
/// Why the containment rule (m4-145 review MAJOR-2): the grid cursor can move
/// without the selection following — native GridView keynav drives the
/// SingleSelection directly, and preview stepping sets it programmatically —
/// so a bare snapshot-else-cursor can bind edits to a set that EXCLUDES the
/// image whose text is on screen. Binding to the displayed image then is the
/// only rule under which the panel never shows one image while writing to
/// others (upstream gets this for free: its `last_act_on` drives both display
/// and write). Rating/colour keys deliberately keep the plain
/// snapshot-else-cursor read — they act on grid focus, not on text under it.
/// Shared by focus-enter and `update()`'s re-baseline so every snapshot site
/// agrees on what a target list looks like.
fn metadata_target_paths(cursor_path: &str) -> Vec<String> {
    let mut targets = crate::lighttable::selection::paths_snapshot();
    if cursor_path.is_empty() || !targets.iter().any(|p| p == cursor_path) {
        targets.clear();
        if !cursor_path.is_empty() {
            targets.push(cursor_path.to_string());
        }
    }
    targets
}

/// Text for the scope hint under the metadata entries: `None` hides it (the
/// single-image case must look exactly as before), otherwise the honest count.
/// Pure so the wording is pinned display-free.
fn meta_scope_hint_text(n: usize) -> Option<String> {
    match n {
        0 | 1 => None,
        k => Some(format!("Edits apply to all {k} selected images")),
    }
}

/// One honest line describing a commit's outcome — `None` when everything
/// landed, which must look exactly as before (no toast on plain success):
///   nothing written              -> "Could not save {what}"
///   some targets were skipped    -> "Saved {what} for K of N images"
///   sidecar writes failed        -> "Could not update X XMP sidecar(s)"
/// both non-silent cases combine with "; ". Pure so every wording is pinned
/// display-free. Exists because m4-145's first draft CLAIMED in comments that
/// uncatalogued skips were reported when the code silently dropped them
/// (review BLOCKER-1) — this makes the report real and testable instead.
fn metadata_commit_report(
    what: &str,
    written: usize,
    total: usize,
    xmp_failed: usize,
) -> Option<String> {
    if written == 0 {
        return Some(format!("Could not save {what}"));
    }
    let mut msg = String::new();
    if written < total {
        msg = format!("Saved {what} for {written} of {total} images");
    }
    if xmp_failed > 0 {
        let tail = format!("could not update {xmp_failed} XMP sidecar(s)");
        if msg.is_empty() {
            // Standalone sidecar line starts the sentence: capitalise it.
            let mut cs = tail.chars();
            if let Some(c) = cs.next() {
                msg.extend(c.to_uppercase());
                msg.push_str(cs.as_str());
            }
        } else {
            msg.push_str("; ");
            msg.push_str(&tail);
        }
    }
    if msg.is_empty() { None } else { Some(msg) }
}

/// A copied edit: the saved params blob plus the image's undo-stack, and the
/// source path it came from. Plain data so clipboard semantics are testable
/// without GTK.
#[derive(Clone)]
struct HistoryClipboard {
    /// Full path of the copy's source (names the origin in the paste toast).
    source: String,
    params: crate::preview::PreviewParams,
    stack: Option<crate::history::HistoryStack>,
}

impl HistoryClipboard {
    /// Copy the named image's saved edit. `None` when it has none.
    fn from_image(db_path: &str, full_path: &str) -> Option<Self> {
        let params = crate::persist::load_saved(db_path, full_path)?;
        Some(Self {
            source: full_path.to_string(),
            params,
            stack: crate::persist::load_history(db_path, full_path),
        })
    }

    /// Write the copied edit onto `full_path`, replacing both its rows —
    /// darktable's paste replaces the target stack rather than appending. When
    /// the copy carried no stack row (source predates the history feature or
    /// its blob failed to decode), seed a fresh one-entry stack from the pasted
    /// params instead of leaving the target's old stack describing edits that
    /// no longer exist — the same picture the loader paints for a params row
    /// without a stack row.
    fn apply_to(&self, db_path: &str, full_path: &str) {
        crate::persist::save_params(db_path, full_path, &self.params);
        match &self.stack {
            Some(h) => crate::persist::save_history(db_path, full_path, h),
            None => crate::persist::save_history(
                db_path,
                full_path,
                &crate::history::HistoryStack::new("Original", self.params),
            ),
        }
    }

    /// The source's file name for toasts ("Pasted edit from DSC_1234.NEF").
    fn source_basename(&self) -> &str {
        std::path::Path::new(&self.source)
            .file_name().and_then(|n| n.to_str()).unwrap_or("another image")
    }
}

/// Shared clipboard handle ([`HistoryClipboard`] behind an Rc cell).
type HistoryClipboardHandle = std::rc::Rc<std::cell::RefCell<Option<HistoryClipboard>>>;

/// Repaint the History section's one-line readout for `(db, path)`.
fn refresh_history_readout(lbl: &gtk4::Label, db: &str, path: &str) {
    lbl.set_label(&history_readout_text(db, path));
}

/// The readout text itself, split out so it is testable without GTK:
/// "(no image selected)" / "no saved edits" / "N-step edit stack".
fn history_readout_text(db: &str, path: &str) -> String {
    if path.is_empty() {
        return "(no image selected)".into();
    }
    // A saved params row IS an edit (even an all-neutral one): it changes how
    // the image renders vs raw defaults, e.g. raw-only sigmoid seeding.
    match crate::persist::load_saved(db, path) {
        None => "no saved edits".into(),
        Some(_) => {
            let steps = crate::persist::load_history(db, path)
                .map(|h| h.len())
                .unwrap_or(1);
            format!("{steps}-step edit stack")
        }
    }
}

/// Metadata inspector widget with an `update` method.
///
/// All GTK fields are GObject ref-counts so `MetadataPanel` is Clone.
#[derive(Clone)]
pub struct MetadataPanel {
    pub widget:   gtk4::Box,
    /// Styles section (parity 2.4). Handlers are wired in `wire_styles`, which
    /// the caller invokes once it can supply the current params — the panel
    /// itself has no view of the darkroom's live edit state.
    styles_list:      gtk4::ListBox,
    style_save_btn:   gtk4::Button,
    style_apply_btn:  gtk4::Button,
    style_delete_btn: gtk4::Button,
    /// Guards [`wire_styles`] against a second call. `connect_clicked` appends,
    /// so wiring twice would stack two save dialogs and apply a style twice —
    /// and `MetadataPanel` is `Clone`, so a clone reaching a second call site
    /// is a plausible accident rather than a theoretical one.
    styles_wired: std::rc::Rc<std::cell::Cell<bool>>,
    /// History-stack section (parity row 2.2): copy/paste/discard on the
    /// selected image, mirroring darktable's lighttable "actions on selection"
    /// history group adapted to single selection.
    history_copy_btn:    gtk4::Button,
    history_paste_btn:   gtk4::Button,
    history_discard_btn: gtk4::Button,
    /// One-line state readout ("no saved edits" / "N-step edit stack").
    history_lbl:         gtk4::Label,
    /// The copied edit (params + undo-stack blob) awaiting Paste. Process-local,
    /// like darktable's own history clipboard — it dies with the app.
    history_clipboard: HistoryClipboardHandle,
    filename_lbl: gtk4::Label,
    folder_lbl:   gtk4::Label,
    dims_lbl:     gtk4::Label,
    size_lbl:     gtk4::Label,
    // EXIF rows (m4-100), mirroring darktable's "image information" module.
    camera_lbl:   gtk4::Label,
    lens_lbl:     gtk4::Label,
    exposure_lbl: gtk4::Label,
    aperture_lbl: gtk4::Label,
    iso_lbl:      gtk4::Label,
    focal_lbl:    gtk4::Label,
    taken_lbl:    gtk4::Label,
    tags_flow:    gtk4::FlowBox,
    tag_entry:    gtk4::Entry,
    /// Shared (path, db_path) for the add-tag handler
    ctx: std::rc::Rc<std::cell::RefCell<(String, String)>>,
    /// Optional notify fired after a tag is attached, so other panels (the
    /// left-panel Tags section) can refresh. Set via [`set_on_tags_changed`].
    on_tags_changed: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>>,
    /// Metadata-editor entries, in [`crate::persist::MetaField::ALL`] order.
    meta_entries: Vec<gtk4::Entry>,
    /// Editing baselines, index-aligned with `meta_entries`. Held so `update()`
    /// can re-baseline an entry it repaints, and so a focused entry is never left
    /// showing one image's text while `ctx` points at another. Each target now
    /// carries the whole multi-selection (m4-145), not just one path.
    meta_targets: Vec<MetaTarget>,
    /// Dim hint under the entries, shown only when more than one image is
    /// selected: commits fan out, and typing without knowing that is how
    /// "I only meant to rename this one" disasters happen.
    meta_scope_lbl: gtk4::Label,
    /// Optional user-visible notifier, used to report a metadata save that could
    /// not land. Same shape as `on_tags_changed`; set via [`set_on_notify`].
    on_notify: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn(String)>>>>,
}

impl MetadataPanel {
    /// Rebuild the styles list from the database.
    ///
    /// Rows carry the style name as the widget name so the handlers can read
    /// the selection back without a parallel model — the list is short and
    /// rebuilt wholesale on every change, so a model would be more machinery
    /// than the feature needs.
    fn refresh_styles_list(list: &gtk4::ListBox, db_path: &str) {
        // Re-select the same style afterwards. A wholesale rebuild drops the
        // selection, and "pick a style, then pick a target, then Apply" is the
        // entire workflow — losing it mid-way makes Apply a silent no-op.
        let keep = list.selected_row().map(|r| r.widget_name().to_string());
        while let Some(child) = list.first_child() {
            list.remove(&child);
        }
        let styles = crate::persist::load_styles(db_path);
        if styles.is_empty() {
            let row = gtk4::ListBoxRow::new();
            let lbl = gtk4::Label::builder()
                .label("(no styles saved)")
                .halign(gtk4::Align::Start)
                .margin_start(8).margin_end(8).margin_top(4).margin_bottom(4)
                .build();
            lbl.add_css_class("dim-label");
            row.set_child(Some(&lbl));
            row.set_selectable(false);
            // Unnamed widgets report their TYPE name ("GtkListBoxRow"), not "".
            // The handlers guard on an empty name, so without this the
            // placeholder would read as a style called "GtkListBoxRow" the day
            // it stops being unselectable.
            row.set_widget_name("");
            list.append(&row);
            return;
        }
        for st in styles {
            let row = gtk4::ListBoxRow::new();
            row.set_widget_name(&st.name);
            let b = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
            b.set_margin_start(8);
            b.set_margin_end(8);
            b.set_margin_top(3);
            b.set_margin_bottom(3);
            let name = gtk4::Label::builder()
                .label(&st.name)
                .halign(gtk4::Align::Start)
                .build();
            b.append(&name);
            if !st.description.is_empty() {
                let desc = gtk4::Label::builder()
                    .label(&st.description)
                    .halign(gtk4::Align::Start)
                    .build();
                desc.add_css_class("dim-label");
                desc.add_css_class("caption");
                b.append(&desc);
            }
            let is_kept = keep.as_deref() == Some(st.name.as_str());
            row.set_child(Some(&b));
            list.append(&row);
            if is_kept {
                list.select_row(Some(&row));
            }
        }
    }

    /// Wire the Styles buttons. **Call exactly once** per panel — see
    /// `styles_wired`.
    ///
    /// `current_params` is a getter rather than a value because the panel is
    /// built once and the selected image's edit changes constantly; asking at
    /// click time is what makes "Save current" mean the edit on screen. It is
    /// also the single source of truth for *which* image is the target: reading
    /// the path from `self.ctx` instead would let Apply write onto a stale path
    /// once the selection empties, since `update()` (ctx's only writer) is not
    /// called when nothing is selected.
    ///
    /// `notify` surfaces outcomes to the user — every failure path here is
    /// otherwise invisible, which is the same silent-failure class as a CSS
    /// rule GTK drops on the floor.
    ///
    /// `db_path` is passed explicitly rather than read from `self.ctx.1`: that
    /// field is a process-wide constant threaded through a per-image tuple, and
    /// it is empty until the first `update()` — that is, whenever the catalogue
    /// is empty, in which case every handler would return at its guard.
    pub fn wire_styles(
        &self,
        db_path: &str,
        current_params: StyleParamsGetter,
        notify: std::rc::Rc<dyn Fn(String)>,
    ) {
        if self.styles_wired.replace(true) {
            debug_assert!(false, "wire_styles called twice — handlers would double-fire");
            return;
        }

        let db_path = db_path.to_string();
        let list = self.styles_list.clone();

        Self::refresh_styles_list(&list, &db_path);

        {
            let db_path = db_path.clone();
            let list = list.clone();
            let get = current_params.clone();
            let notify = notify.clone();
            self.style_save_btn.connect_clicked(move |btn| {
                let db = db_path.clone();
                if db.is_empty() {
                    notify("No catalogue open".into());
                    return;
                }
                let Some((_path, params)) = get() else {
                    notify("Select an image first".into());
                    return;
                };
                // adw::AlertDialog with an extra child, matching
                // dialogs::show_export_dialog. gtk4::Dialog is deprecated in
                // GTK4 and its content_area() did not lay the entries out at
                // all — the dialog rendered as an empty frame.
                let fields = gtk4::Box::builder()
                    .orientation(gtk4::Orientation::Vertical)
                    .spacing(6)
                    .build();
                let name_entry = gtk4::Entry::builder()
                    .placeholder_text("Style name")
                    .activates_default(true)
                    .build();
                let desc_entry = gtk4::Entry::builder()
                    .placeholder_text("Description (optional)")
                    .build();
                fields.append(&name_entry);
                fields.append(&desc_entry);

                let dialog = adw::AlertDialog::builder()
                    .heading("Save style")
                    .body("Save this image's current edit for reuse.")
                    .build();
                dialog.set_extra_child(Some(&fields));
                dialog.add_response("cancel", "Cancel");
                dialog.add_response("save", "Save");
                dialog.set_response_appearance("save", adw::ResponseAppearance::Suggested);
                dialog.set_default_response(Some("save"));

                let list_c = list.clone();
                let db_c = db.clone();
                let notify_c = notify.clone();
                // Present the overwrite prompt on the BUTTON, not on the
                // dialog: the dialog is dismissing as this fires.
                let parent = btn.clone().upcast::<gtk4::Widget>();
                dialog.connect_response(Some("save"), move |_, _| {
                    let name = name_entry.text().to_string();
                    let desc = desc_entry.text().to_string();
                    // save_style upserts, so a name collision silently replaces
                    // a saved edit with no way back — styles have no history
                    // stack behind them. Confirm first.
                    let clash = crate::persist::load_styles(&db_c)
                        .iter()
                        .any(|s| s.name == name.trim());
                    if clash {
                        let confirm = adw::AlertDialog::builder()
                            .heading("Replace style?")
                            .body(format!("A style named \"{}\" already exists.", name.trim()))
                            .build();
                        confirm.add_response("cancel", "Cancel");
                        confirm.add_response("replace", "Replace");
                        confirm.set_response_appearance(
                            "replace",
                            adw::ResponseAppearance::Destructive,
                        );
                        let (db_x, list_x, notify_x) =
                            (db_c.clone(), list_c.clone(), notify_c.clone());
                        let (name_x, desc_x, params_x) =
                            (name.clone(), desc.clone(), params.clone());
                        confirm.connect_response(Some("replace"), move |_, _| {
                            Self::save_style_reporting(
                                &db_x, &name_x, &desc_x, &params_x, &list_x, &notify_x,
                            );
                        });
                        confirm.present(Some(&parent));
                        return;
                    }
                    Self::save_style_reporting(
                        &db_c, &name, &desc, &params, &list_c, &notify_c,
                    );
                });
                dialog.present(Some(btn.upcast_ref::<gtk4::Widget>()));
            });
        }

        // Apply is reachable from the button and from double-clicking a row
        // (darktable applies on activation), so the body lives in one closure
        // rather than being written twice.
        let do_apply: std::rc::Rc<dyn Fn()> = {
            let db_path = db_path.clone();
            let list = list.clone();
            let get = current_params.clone();
            let notify = notify.clone();
            std::rc::Rc::new(move || {
                let db = db_path.clone();
                let Some(row) = list.selected_row() else {
                    notify("Select a style first".into());
                    return;
                };
                let name = row.widget_name().to_string();
                if name.is_empty() {
                    notify("Select a style first".into());
                    return;
                }
                // The path comes from the getter, not ctx — one reading of
                // "which image", so Apply cannot target a stale one.
                let Some((path, _)) = get() else {
                    notify("Select an image first".into());
                    return;
                };
                if path.is_empty() || db.is_empty() {
                    notify("Select an image first".into());
                    return;
                }
                let Some(style) =
                    crate::persist::load_styles(&db).into_iter().find(|s| s.name == name)
                else {
                    notify(format!("Style \"{name}\" is no longer available"));
                    return;
                };
                // apply_style_to counts WRITES: an uncatalogued path has no
                // imgid to store an edit against, so 0 here means "nothing
                // happened", not "applied to zero of one".
                if crate::persist::apply_style_to(&db, &[path], &style) > 0 {
                    notify(format!("Applied style \"{name}\""));
                } else {
                    notify("Could not apply the style to this image".into());
                }
            })
        };

        {
            let do_apply = do_apply.clone();
            self.style_apply_btn.connect_clicked(move |_| do_apply());
        }
        {
            let do_apply = do_apply.clone();
            list.connect_row_activated(move |_, _| do_apply());
        }

        {
            let db_path = db_path.clone();
            let list = list.clone();
            let notify = notify.clone();
            self.style_delete_btn.connect_clicked(move |btn| {
                let db = db_path.clone();
                let Some(row) = list.selected_row() else {
                    notify("Select a style first".into());
                    return;
                };
                let name = row.widget_name().to_string();
                if name.is_empty() {
                    notify("Select a style first".into());
                    return;
                }
                // Deleting a style is unrecoverable — confirm, as darktable does.
                let confirm = adw::AlertDialog::builder()
                    .heading("Delete style?")
                    .body(format!("\"{name}\" will be removed permanently."))
                    .build();
                confirm.add_response("cancel", "Cancel");
                confirm.add_response("delete", "Delete");
                confirm.set_response_appearance("delete", adw::ResponseAppearance::Destructive);
                let (db_c, list_c, notify_c) = (db.clone(), list.clone(), notify.clone());
                confirm.connect_response(Some("delete"), move |_, _| {
                    if crate::persist::delete_style(&db_c, &name) {
                        Self::refresh_styles_list(&list_c, &db_c);
                        notify_c(format!("Deleted style \"{name}\""));
                    } else {
                        notify_c("Could not delete the style".into());
                    }
                });
                confirm.present(Some(btn.upcast_ref::<gtk4::Widget>()));
            });
        }
    }

    /// Save, refresh the list, and tell the user what happened either way.
    ///
    /// `save_style` documents that the caller surfaces its `false` — a blank
    /// name or a locked database otherwise closes the dialog and changes
    /// nothing, with no explanation.
    fn save_style_reporting(
        db: &str,
        name: &str,
        desc: &str,
        params: &crate::preview::PreviewParams,
        list: &gtk4::ListBox,
        notify: &std::rc::Rc<dyn Fn(String)>,
    ) {
        if crate::persist::save_style(db, name, desc, params) {
            Self::refresh_styles_list(list, db);
            notify(format!("Saved style \"{}\"", name.trim()));
        } else if name.trim().is_empty() {
            notify("Style name cannot be empty".into());
        } else {
            notify("Could not save the style".into());
        }
    }

    pub fn new() -> Self {
        let panel = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(0)
            .width_request(210)
            .build();

        // ── Header ────────────────────────────────────────────────────────
        let header = gtk4::Label::builder()
            .label("Metadata")
            .halign(gtk4::Align::Start)
            .margin_top(12).margin_bottom(6)
            .margin_start(12).margin_end(12)
            .build();
        header.add_css_class("heading");
        panel.append(&header);
        panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

        // ── Info grid ─────────────────────────────────────────────────────
        let mk_key = |text: &str| {
            let l = gtk4::Label::builder().label(text).halign(gtk4::Align::End).build();
            l.add_css_class("dim-label");
            l
        };
        let mk_val = || gtk4::Label::builder()
            .halign(gtk4::Align::Start).hexpand(true)
            .max_width_chars(20).ellipsize(gtk4::pango::EllipsizeMode::Middle)
            .build();

        let filename_lbl = mk_val();
        let folder_lbl   = mk_val();
        let dims_lbl     = mk_val();
        let size_lbl     = mk_val();
        let camera_lbl   = mk_val();
        let lens_lbl     = mk_val();
        let exposure_lbl = mk_val();
        let aperture_lbl = mk_val();
        let iso_lbl      = mk_val();
        let focal_lbl    = mk_val();
        let taken_lbl    = mk_val();

        let grid = gtk4::Grid::builder()
            .row_spacing(4).column_spacing(8)
            .margin_start(12).margin_end(12).margin_top(10)
            .build();
        // Ordered as darktable's "image information": file identity first, then the
        // capture facts (camera → lens → exposure triangle → date).
        for (i, (key, val)) in [
            ("File",     &filename_lbl), ("Folder",   &folder_lbl),
            ("Size",     &dims_lbl),     ("Disk",     &size_lbl),
            ("Camera",   &camera_lbl),   ("Lens",     &lens_lbl),
            ("Exposure", &exposure_lbl), ("Aperture", &aperture_lbl),
            ("ISO",      &iso_lbl),      ("Focal",    &focal_lbl),
            ("Taken",    &taken_lbl),
        ].iter().enumerate() {
            grid.attach(&mk_key(key), 0, i as i32, 1, 1);
            grid.attach(*val, 1, i as i32, 1, 1);
        }
        panel.append(&grid);

        // ── Metadata editor (parity 2.3) ──────────────────────────────────
        // darktable's "metadata editor" module: the writable Dublin Core fields,
        // as opposed to the read-only EXIF above. Written straight into
        // darktable's own `main.meta_data` — see persist::MetaField.
        panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
        panel.append(&section_header("Metadata editor"));

        let meta_grid = gtk4::Grid::builder()
            .row_spacing(4).column_spacing(8)
            .margin_start(12).margin_end(12).margin_top(2).margin_bottom(6)
            .build();
        let mut meta_entries: Vec<gtk4::Entry> = Vec::new();
        for (i, field) in crate::persist::MetaField::ALL.iter().enumerate() {
            let e = gtk4::Entry::builder()
                .hexpand(true)
                .width_chars(8)
                .placeholder_text("—")
                .build();
            meta_grid.attach(&mk_key(field.label()), 0, i as i32, 1, 1);
            meta_grid.attach(&e, 1, i as i32, 1, 1);
            meta_entries.push(e);
        }
        panel.append(&meta_grid);

        // Scope hint (m4-145): with a multi-selection active, commits land on
        // every selected image — darktable's behaviour, but silent fan-out is
        // exactly the kind of power that must be VISIBLE. Hidden at ≤1 so the
        // single-image panel looks unchanged.
        let meta_scope_lbl = gtk4::Label::builder()
            .halign(gtk4::Align::Start)
            .wrap(true)
            .margin_start(12).margin_end(12).margin_bottom(4)
            .build();
        meta_scope_lbl.add_css_class("dim-label");
        meta_scope_lbl.set_visible(false);
        panel.append(&meta_scope_lbl);

        // ── Tags section ──────────────────────────────────────────────────
        let tags_header = gtk4::Label::builder()
            .label("Tags")
            .halign(gtk4::Align::Start)
            .margin_top(12).margin_bottom(4)
            .margin_start(12).margin_end(12)
            .build();
        tags_header.add_css_class("heading");
        panel.append(&tags_header);

        let tags_flow = gtk4::FlowBox::builder()
            .selection_mode(gtk4::SelectionMode::None)
            .homogeneous(false)
            .max_children_per_line(10)
            .margin_start(10).margin_end(10).margin_bottom(6)
            .build();
        panel.append(&tags_flow);

        // Add-tag entry
        let tag_entry = gtk4::Entry::builder()
            .placeholder_text("Add tag…")
            .margin_start(10).margin_end(10).margin_bottom(8)
            .build();
        panel.append(&tag_entry);

        // ── Placeholder ───────────────────────────────────────────────────
        let placeholder = gtk4::Label::builder()
            .label("Select an image\nto view metadata")
            .halign(gtk4::Align::Center).valign(gtk4::Align::Center)
            .vexpand(true).justify(gtk4::Justification::Center)
            .build();
        placeholder.add_css_class("dim-label");
        panel.append(&placeholder);

        let ctx = std::rc::Rc::new(std::cell::RefCell::new((String::new(), String::new())));
        let on_tags_changed: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));

        // Wire add-tag on Enter key
        {
            let ctx2      = ctx.clone();
            let flow_ref  = tags_flow.clone();
            let notify    = on_tags_changed.clone();
            tag_entry.connect_activate(move |entry| {
                let tag_name = entry.text().trim().to_string();
                if tag_name.is_empty() { return; }
                // Clone the (path, db) out so the cell isn't borrowed while
                // rebuild_tags_flow re-borrows it.
                let (path, db) = ctx2.borrow().clone();
                if !db.is_empty() {
                    add_tag_to_image(&path, &db, &tag_name);
                    rebuild_tags_flow(&flow_ref, &ctx2, &notify);
                    // Notify other panels (clone the callback out before invoking
                    // so the cell isn't borrowed while it runs).
                    let cb = notify.borrow().clone();
                    if let Some(cb) = cb { cb(); }
                }
                entry.set_text("");
            });
        }

        // ── Styles (parity 2.4) ───────────────────────────────────────────
        // darktable's styles module: save the current edit under a name, then
        // apply it to other images. Ours stores the whole PreviewParams blob
        // (see persist::STYLES_TABLE_DDL for why, and what that costs).
        let styles_header = section_header("Styles");
        panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
        panel.append(&styles_header);

        let styles_list = gtk4::ListBox::builder()
            .selection_mode(gtk4::SelectionMode::Single)
            .build();
        styles_list.add_css_class("navigation-sidebar");
        // Capped and scrolled: nothing bounds how many styles a user saves, and
        // the panel is a plain Box with no scroller of its own — an unbounded
        // list would push Export and the tag chips out of the panel.
        let styles_scroll = gtk4::ScrolledWindow::builder()
            .propagate_natural_height(true)
            .max_content_height(180)
            .hscrollbar_policy(gtk4::PolicyType::Never)
            .child(&styles_list)
            .build();
        panel.append(&styles_scroll);

        let styles_btns = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(4)
            .margin_start(10).margin_end(10).margin_top(4).margin_bottom(6)
            .build();
        let style_save_btn = gtk4::Button::builder()
            .label("Save current…")
            .tooltip_text("Save this image's edit as a reusable style")
            .hexpand(true)
            .build();
        let style_apply_btn = gtk4::Button::builder()
            .label("Apply")
            .tooltip_text("Apply the selected style to this image")
            .build();
        let style_delete_btn = gtk4::Button::builder()
            .icon_name("user-trash-symbolic")
            .tooltip_text("Delete the selected style")
            .build();
        styles_btns.append(&style_save_btn);
        styles_btns.append(&style_apply_btn);
        styles_btns.append(&style_delete_btn);
        panel.append(&styles_btns);

        // Toast channel shared with the metadata editor (declared here because
        // the history handlers below fire it too).
        let on_notify: std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn(String)>>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));

        // ── History stack (parity row 2.2) ────────────────────────────────
        // darktable's lighttable "actions on selection" history group — copy,
        // paste, discard — adapted to C-41's single-selection grid: the
        // actions act on the currently-selected image. Copy reads both saved
        // rows (params + undo-stack); paste REPLACES the target's rows, as
        // darktable's paste replaces the stack; discard clears both in one
        // transaction and confirms first (deliberate deviation: darktable acts
        // immediately, but its action targets a whole multi-image selection).
        panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
        panel.append(&section_header("History"));

        let history_lbl = gtk4::Label::builder()
            .label("(no image selected)")
            .halign(gtk4::Align::Start)
            .margin_start(12).margin_end(12).margin_top(2)
            .build();
        history_lbl.add_css_class("dim-label");
        panel.append(&history_lbl);

        let history_btns = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(4)
            .margin_start(10).margin_end(10).margin_top(4).margin_bottom(6)
            .build();
        let history_copy_btn = gtk4::Button::builder()
            .label("Copy")
            .tooltip_text("Copy this image's edit stack (darktable: copy history)")
            .hexpand(true)
            .build();
        let history_paste_btn = gtk4::Button::builder()
            .label("Paste")
            .tooltip_text("Paste the copied edit onto this image, replacing its own (darktable: paste history)")
            .hexpand(true)
            .sensitive(false) // clipboard starts empty
            .build();
        let history_discard_btn = gtk4::Button::builder()
            .label("Discard")
            .tooltip_text("Remove every saved edit from this image (darktable: discard history)")
            .hexpand(true)
            .build();
        history_btns.append(&history_copy_btn);
        history_btns.append(&history_paste_btn);
        history_btns.append(&history_discard_btn);
        panel.append(&history_btns);

        // The clipboard lives on the panel so `update()` can gate Paste's
        // sensitivity on it across selection changes.
        let history_clipboard: HistoryClipboardHandle =
            std::rc::Rc::new(std::cell::RefCell::new(None));
        {
            let clip = history_clipboard.clone();
            let paste_btn = history_paste_btn.clone();
            let notify = on_notify.clone();
            let ctx = ctx.clone();
            history_copy_btn.connect_clicked(move |_| {
                let (path, db) = ctx.borrow().clone();
                if path.is_empty() { return; }
                if db.is_empty() {
                    // Underlying reads would silently no-op; say so honestly.
                    if let Some(n) = notify.borrow().as_ref() {
                        n("No catalogue open".into());
                    }
                    return;
                }
                match HistoryClipboard::from_image(&db, &path) {
                    Some(c) => {
                        *clip.borrow_mut() = Some(c);
                        paste_btn.set_sensitive(true);
                    }
                    None => {
                        if let Some(n) = notify.borrow().as_ref() {
                            n("Nothing to copy — the selected image has no saved edits".into());
                        }
                    }
                }
            });
        }
        {
            let clip = history_clipboard.clone();
            let notify = on_notify.clone();
            let lbl = history_lbl.clone();
            let copy_btn = history_copy_btn.clone();
            let discard_btn = history_discard_btn.clone();
            let ctx = ctx.clone();
            history_paste_btn.connect_clicked(move |_| {
                let (path, db) = ctx.borrow().clone();
                if path.is_empty() { return; }
                if db.is_empty() {
                    // Underlying writes would silently no-op; say so honestly.
                    if let Some(n) = notify.borrow().as_ref() {
                        n("No catalogue open".into());
                    }
                    return;
                }
                // Apply and lift the source name to an owned String while the
                // clipboard borrow is alive; the guard drops at statement end.
                let pasted_from = clip.borrow().as_ref().map(|c| {
                    c.apply_to(&db, &path);
                    c.source_basename().to_string()
                });
                let Some(from) = pasted_from else { return };
                if let Some(n) = notify.borrow().as_ref() {
                    n(format!("Pasted edit from {from}"));
                }
                refresh_history_readout(&lbl, &db, &path);
                // The target just gained edits — mirror update()'s gating now
                // rather than waiting for the next selection change.
                let has_edits = crate::persist::load_saved(&db, &path).is_some();
                copy_btn.set_sensitive(has_edits);
                discard_btn.set_sensitive(has_edits);
            });
        }
        {
            // Discard is destructive with no undo behind it, so confirm first.
            // Deliberate deviation from darktable, which acts immediately: its
            // actions-on-selection target a whole multi-image selection and are
            // part of a keyboard-driven workflow. Present on the BUTTON, not on
            // the dialog itself — the same pattern wire_styles uses because the
            // dialog is dismissing as the response fires.
            let notify = on_notify.clone();
            let lbl = history_lbl.clone();
            let copy_btn = history_copy_btn.clone();
            let discard_btn = history_discard_btn.clone();
            let ctx = ctx.clone();
            history_discard_btn.connect_clicked(move |btn| {
                let (path, db) = ctx.borrow().clone();
                if path.is_empty() { return; }
                if db.is_empty() {
                    // Underlying writes would silently no-op; say so honestly.
                    if let Some(n) = notify.borrow().as_ref() {
                        n("No catalogue open".into());
                    }
                    return;
                }
                let dialog = adw::AlertDialog::builder()
                    .heading("Discard history?")
                    .body("Remove every saved edit from this image? This cannot be undone.")
                    .build();
                dialog.add_response("cancel", "Cancel");
                dialog.add_response("discard", "Discard");
                dialog.set_response_appearance(
                    "discard", adw::ResponseAppearance::Destructive);
                dialog.set_default_response(Some("cancel"));
                let (db_c, path_c, lbl_c, notify_c) =
                    (db.clone(), path.clone(), lbl.clone(), notify.clone());
                let (copy_c, discard_self) =
                    (copy_btn.clone(), discard_btn.clone());
                dialog.connect_response(Some("discard"), move |_, _| {
                    crate::persist::discard_history(&db_c, &path_c);
                    refresh_history_readout(&lbl_c, &db_c, &path_c);
                    // Nothing remains to copy or re-discard until new edits
                    // land; mirror update()'s gating immediately.
                    copy_c.set_sensitive(false);
                    discard_self.set_sensitive(false);
                    if let Some(n) = notify_c.borrow().as_ref() {
                        n("Discarded all edits".into());
                    }
                });
                let parent = btn.clone().upcast::<gtk4::Widget>();
                dialog.present(Some(&parent));
            });
        }

        // ── Export (parity 2.6) ───────────────────────────────────────────
        // darktable's right panel ends with an export module. Ours had export
        // only as a header button; this surfaces the same `win.export-selected`
        // action so there is one implementation, and puts it where a darktable
        // user looks for it. Pushed to the bottom so it does not displace the
        // metadata the panel exists to show.
        let export_header = section_header("Export");
        panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
        panel.append(&export_header);
        let export_btn = gtk4::Button::builder()
            .label("Export selected…")
            .tooltip_text("Export the selected images (Ctrl+E)")
            .margin_start(10).margin_end(10).margin_top(2).margin_bottom(8)
            .build();
        export_btn.set_action_name(Some("win.export-selected"));
        panel.append(&export_btn);

        // Commit a metadata field on Enter or on losing focus.
        //
        // Two guards, both taken from upstream, because either alone is not enough:
        //
        //  1. The target images are snapshotted when the entry GAINS focus, not read
        //     at save time — clicking a different thumbnail moves focus AND changes
        //     the selection, and `update()` rewrites `ctx`, so reading ctx on
        //     focus-out could save image A's text onto image B.
        //  2. The ORIGINAL text is snapshotted alongside them, and an unchanged entry
        //     never writes at all (`src/libs/metadata.c:360`).
        //
        // (2) is what makes this correct rather than merely lucky: GTK4's ordering
        // of focus-leave against selection-changed is not contractual, and (1)
        // alone would depend on it.
        //
        // m4-145: the snapshot is darktable's whole-selection semantics — every
        // image in the lighttable multi-selection gets the write
        // (`dt_metadata_set_list` over the selected ids), falling back to the
        // active image alone when nothing is multi-selected. The dim hint label
        // states the count at focus-enter, the moment targets actually bind.
        let meta_targets: Vec<MetaTarget> = Vec::new();
        let mut meta_targets = meta_targets;
        for (i, entry) in meta_entries.iter().enumerate() {
            let field = crate::persist::MetaField::ALL[i];
            // (target paths, db_path, original text) as they were when editing began.
            let target: MetaTarget =
                std::rc::Rc::new(std::cell::RefCell::new((Vec::new(), String::new(), String::new())));
            meta_targets.push(target.clone());

            let focus = gtk4::EventControllerFocus::new();
            {
                let ctx = ctx.clone();
                let target = target.clone();
                let entry = entry.clone();
                let scope = meta_scope_lbl.clone();
                focus.connect_enter(move |_| {
                    let (p, d) = ctx.borrow().clone();
                    let targets = metadata_target_paths(&p);
                    match meta_scope_hint_text(targets.len()) {
                        Some(text) => {
                            scope.set_text(&text);
                            scope.set_visible(true);
                        }
                        None => scope.set_visible(false),
                    }
                    *target.borrow_mut() = (targets, d, entry.text().to_string());
                });
            }

            let commit = {
                let target = target.clone();
                let notify = on_notify.clone();
                std::rc::Rc::new(move |e: &gtk4::Entry| {
                    let (paths, db, orig) = target.borrow().clone();
                    if paths.is_empty() || db.is_empty() {
                        return;
                    }
                    let value = e.text().to_string();
                    // The dirty check is what makes this safe, not the event
                    // ordering. Upstream does the same (`src/libs/metadata.c:360`
                    // compares against a stashed `text_orig` before adding a field
                    // to the write list). An entry the user never touched can then
                    // never write, so even a stale focus-leave — one that fires
                    // after the selection moved on — is a no-op instead of copying
                    // one image's text onto another. With fan-out this check
                    // matters more, not less: it is the only thing standing
                    // between an idle tab-through and N rewritten titles.
                    if value == orig {
                        return;
                    }
                    // Only this field is passed, so the other four are untouched.
                    // Images that are no longer in the catalogue drop out of
                    // `written`; `metadata_commit_report` counts them against
                    // `paths` ("Saved … for K of N images") instead of letting
                    // the skip pass silently (m4-145 review BLOCKER-1).
                    let written = crate::persist::save_metadata_many(
                        &db,
                        &paths,
                        &[(field, value.clone())],
                    );
                    if !written.is_empty() {
                        // Re-baseline, so the focus-leave that follows an Enter
                        // does not write the same value a second time.
                        target.borrow_mut().2 = value;
                    }
                    // Mirror the edit into each written image's `.xmp`
                    // sidecar (upstream's `dt_image_synch_xmps`,
                    // src/libs/metadata.c:393, which also loops the whole
                    // selection). The sync reads all five fields back from
                    // the catalogue, so every sidecar lands as one
                    // consistent set even though this commit wrote one
                    // field. Catalogue first, sidecars second: a failed
                    // sidecar write must not roll back or block the
                    // authoritative store, only be reported — aggregated
                    // into one toast, since N identical popups help nobody.
                    let failed = written
                        .iter()
                        .filter(|p| !crate::xmp::sync_sidecar(&db, p))
                        .count();
                    if let Some(msg) = metadata_commit_report(
                        &field.label().to_lowercase(),
                        written.len(),
                        paths.len(),
                        failed,
                    ) {
                        if let Some(n) = notify.borrow().as_ref() {
                            n(msg);
                        }
                    }
                })
            };

            {
                let commit = commit.clone();
                let entry = entry.clone();
                focus.connect_leave(move |_| commit(&entry));
            }
            entry.add_controller(focus);
            {
                let commit = commit.clone();
                entry.connect_activate(move |e| commit(e));
            }
            // Escape reverts, matching upstream's cancel button
            // (`src/libs/metadata.c:438`). A bare GtkEntry ignores Escape, which
            // would leave the edit in place to be committed by the next
            // focus-leave — the opposite of what the key means.
            {
                let target = target.clone();
                let entry_k = entry.clone();
                let keys = gtk4::EventControllerKey::new();
                keys.connect_key_pressed(move |_, key, _, _| {
                    if key == gtk4::gdk::Key::Escape {
                        // Restore the baseline. The focus-leave that follows then
                        // sees text == orig and commits nothing.
                        let orig = target.borrow().2.clone();
                        entry_k.set_text(&orig);
                        return gtk4::glib::Propagation::Stop;
                    }
                    gtk4::glib::Propagation::Proceed
                });
                entry.add_controller(keys);
            }
        }

        Self { widget: panel, styles_list, style_save_btn, style_apply_btn,
               style_delete_btn,
               history_copy_btn, history_paste_btn, history_discard_btn,
               history_lbl, history_clipboard,
               meta_entries, meta_targets, on_notify, meta_scope_lbl,
               styles_wired: std::rc::Rc::new(std::cell::Cell::new(false)),
               filename_lbl, folder_lbl, dims_lbl, size_lbl,
               camera_lbl, lens_lbl, exposure_lbl, aperture_lbl, iso_lbl,
               focal_lbl, taken_lbl,
               tags_flow, tag_entry, ctx, on_tags_changed }
    }

    /// Register a callback fired whenever a tag is attached from this panel.
    ///
    /// Used to keep the left-panel Tags list in sync with newly-created tags
    /// and changed image counts. Replaces any previously-set callback. This is
    /// the canonical "tags mutated" hook: future tag *detach* / *rename* paths
    /// should route through it too, so the left-panel count refresh stays
    /// single-sourced. The callback must not re-enter a metadata-panel tag
    /// mutation, or it would loop.
    pub fn set_on_tags_changed<F: Fn() + 'static>(&self, f: F) {
        *self.on_tags_changed.borrow_mut() = Some(std::rc::Rc::new(f));
    }

    /// Set the user-visible notifier (a toast, in practice). Used to report a
    /// metadata write that could not land — an uncatalogued image has no `imgid`
    /// to key metadata against, and typing into a field that silently discards
    /// the text is worse than not offering the field.
    pub fn set_on_notify<F: Fn(String) + 'static>(&self, f: F) {
        *self.on_notify.borrow_mut() = Some(std::rc::Rc::new(f));
    }

    /// Persist any metadata entry whose text differs from its editing baseline,
    /// each against the image it was edited on.
    ///
    /// Called before `update()` repaints, and on window close — GTK4 does not
    /// promise a focus-leave during teardown, so without the close hook the last
    /// edit could be dropped, which is the "persist-only-on-close" trap the
    /// darkroom view already learned once.
    ///
    /// Idempotent: it re-baselines each entry it writes, so calling it twice does
    /// not write twice.
    pub fn flush_metadata_edits(&self) {
        // Collect dirty entries grouped by target set first: several fields are
        // often dirty at once (window close after a burst of edits), and each
        // group becomes ONE fan-out write + ONE sidecar pass per image instead
        // of five of each. Since m4-145 a target is a LIST of paths, and two
        // entries can carry different lists if the selection moved between
        // their focus sessions — so the group key is (db, sorted-paths-as-Vec),
        // not just (db, path). A Vec rather than a join keeps distinct path
        // lists distinct even if a filename contained the join separator.
        // Entries with identical lists merge; different lists deliberately do
        // not.
        let mut groups: std::collections::HashMap<
            (String, Vec<String>),
            (Vec<(crate::persist::MetaField, String)>, Vec<(usize, String)>),
        > = std::collections::HashMap::new();
        for (i, entry) in self.meta_entries.iter().enumerate() {
            let Some(target) = self.meta_targets.get(i) else { continue };
            let (paths, db, orig) = target.borrow().clone();
            if paths.is_empty() || db.is_empty() {
                continue;
            }
            let value = entry.text().to_string();
            if value == orig {
                continue;
            }
            let field = crate::persist::MetaField::ALL[i];
            let mut key_paths = paths.clone();
            key_paths.sort();
            let (fields, baselines) = groups
                .entry((db, key_paths))
                .or_insert_with(|| (Vec::new(), Vec::new()));
            fields.push((field, value.clone()));
            baselines.push((i, value));
        }
        // Deterministic write order regardless of HashMap iteration order.
        let mut groups: Vec<_> = groups.into_iter().collect();
        groups.sort_by(|a, b| a.0.cmp(&b.0));
        for ((db, _), (fields, baselines)) in groups {
            // Every entry in a group shares one target list by construction of
            // the key; recover it from any member rather than storing it twice.
            let Some(targets) = self.meta_targets.get(baselines[0].0) else { continue };
            let paths = targets.borrow().0.clone();
            let labels: Vec<&str> = fields.iter().map(|(f, _)| f.label()).collect();
            let written = crate::persist::save_metadata_many(&db, &paths, &fields);
            if !written.is_empty() {
                for (i, value) in baselines {
                    if let Some(t) = self.meta_targets.get(i) {
                        t.borrow_mut().2 = value;
                    }
                }
            }
            // Same honest outcome line as the interactive commit path above —
            // skips and sidecar failures are counted here too, never dropped
            // silently (m4-145 review BLOCKER-1). One pass per written image,
            // failures aggregated into the single line.
            let what = labels.join(", ").to_lowercase();
            let failed =
                written.iter().filter(|p| !crate::xmp::sync_sidecar(&db, p)).count();
            if let Some(msg) =
                metadata_commit_report(&what, written.len(), paths.len(), failed)
            {
                if let Some(n) = self.on_notify.borrow().as_ref() {
                    n(msg);
                }
            }
        }
    }

    /// Re-render the current image's tag chips from the DB without changing the
    /// selected image. Used as the left-panel's "tags mutated" callback so a
    /// rename/delete there updates chips shown here immediately.
    pub fn refresh_tags_display(&self) {
        // Nothing selected yet → leave the placeholder; skip a pointless rebuild.
        if self.ctx.borrow().0.is_empty() { return; }
        rebuild_tags_flow(&self.tags_flow, &self.ctx, &self.on_tags_changed);
    }

    /// Refresh the panel for the image at `full_path`.
    pub fn update(&self, full_path: &str, db_path: &str) {
        use std::path::Path;
        let p        = Path::new(full_path);
        let filename = p.file_name().and_then(|n| n.to_str()).unwrap_or(full_path);
        let folder   = p.parent().and_then(|d| d.to_str()).unwrap_or("");

        self.filename_lbl.set_label(filename);
        self.folder_lbl.set_label(folder.rsplit('/').next().unwrap_or(folder));

        // Metadata editor. Flush first, then repaint — the same order as
        // upstream's `gui_update` (`src/libs/metadata.c:239`, writing at `:269`),
        // and for the same reason: it makes correctness independent of whether
        // GTK happens to deliver focus-leave before or after selection-changed.
        //
        // An earlier version skipped repainting a FOCUSED entry, to avoid
        // clobbering what the user was typing. That left a worse state: if the
        // selection changed while the entry kept focus, the entry went on showing
        // image A's text under image B, with nothing to ever re-sync it — and the
        // next focus-leave wrote A's text onto B. Flushing against the entry's own
        // snapshot and then repainting unconditionally cannot lose an edit,
        // because the flush has already persisted it.
        self.flush_metadata_edits();

        let meta = crate::persist::load_metadata(db_path, full_path);
        for (i, entry) in self.meta_entries.iter().enumerate() {
            // Match on the field rather than trusting index alignment between
            // `load_metadata`'s result and `meta_entries`.
            let field = crate::persist::MetaField::ALL[i];
            let text = meta
                .iter()
                .find(|(f, _)| *f == field)
                .map(|(_, v)| v.as_str())
                .unwrap_or("");
            entry.set_text(text);
            // Re-baseline to the current targets, so a subsequent focus-leave
            // compares against THIS image's value and writes to THIS selection.
            // m4-145: the target list is the whole multi-selection (or the
            // cursor image alone), exactly what focus-enter would snapshot — a
            // defensive fallback for a leave that never saw an enter.
            if let Some(t) = self.meta_targets.get(i) {
                let targets = metadata_target_paths(full_path);
                let mut t = t.borrow_mut();
                *t = (targets, db_path.to_string(), text.to_string());
            }
        }
        // The scope hint tracks the live count too, not just the focus-enter
        // moment: a ctrl-click toggles the set AND moves the cursor (GridView's
        // native handling proceeds after our gesture), which lands here via
        // selection-changed → update() — the one hook guaranteed to fire
        // without an entry refocusing. Refreshing from both sites is
        // idempotent, so the label never outlives the set it describes
        // (m4-145 review MINOR-3: the first draft claimed ctrl-click did NOT
        // move the cursor — it does).
        match meta_scope_hint_text(metadata_target_paths(full_path).len()) {
            Some(text) => {
                self.meta_scope_lbl.set_text(&text);
                self.meta_scope_lbl.set_visible(true);
            }
            None => self.meta_scope_lbl.set_visible(false),
        }

        // NOTE: the styles list is deliberately NOT refreshed here. It is
        // library-wide, not per-image, and `c41_styles` is only ever mutated by
        // the save/delete handlers — both of which refresh already. Rebuilding
        // on every selection change dropped the list selection, which broke the
        // feature's whole workflow: pick a style, pick the target image, Apply.
        // Step two silently deselected step one. It also cost a Connection::open
        // per arrow-key press.

        // One query for every catalog-sourced field (m4-100) — dimensions used to
        // need their own connection; they now ride along with the EXIF row.
        let exif = query_exif(full_path, db_path).unwrap_or_default();
        let camera = format_camera(exif.maker.as_deref(), exif.model.as_deref());
        let lens = format_opt(exif.lens.as_deref());
        let taken = format_opt(exif.datetime.as_deref());
        self.dims_lbl.set_label(&format_dims(exif.width, exif.height));
        self.camera_lbl.set_label(&camera);
        self.lens_lbl.set_label(&lens);
        self.exposure_lbl.set_label(&format_exposure(exif.exposure));
        self.aperture_lbl.set_label(&format_aperture(exif.aperture));
        self.iso_lbl.set_label(&format_iso(exif.iso));
        self.focal_lbl.set_label(&format_focal(exif.focal_length));
        self.taken_lbl.set_label(&taken);
        // Every label ellipsizes at 20 chars, so each value that can exceed that
        // carries the full text in a tooltip — set from the formatted string, not
        // read back off the widget. A placeholder gets no tooltip: hovering to be
        // told "—" is noise.
        let tip = |l: &gtk4::Label, v: &str| {
            l.set_tooltip_text(if v == NO_VALUE { None } else { Some(v) });
        };
        tip(&self.camera_lbl, &camera);
        tip(&self.lens_lbl, &lens);
        tip(&self.taken_lbl, &taken);
        tip(&self.filename_lbl, filename);
        tip(&self.folder_lbl, folder);

        let disk = std::fs::metadata(full_path)
            .map(|m| format_bytes(m.len()))
            .unwrap_or_else(|_| NO_VALUE.into());
        self.size_lbl.set_label(&disk);

        // Store context for the tag-entry / detach handlers
        *self.ctx.borrow_mut() = (full_path.to_string(), db_path.to_string());

        // History section: refresh the readout and re-derive button sensitivity
        // from THIS image's saved state (Copy/Discard need an edited image;
        // Paste needs a non-empty clipboard).
        let has_edits = crate::persist::load_saved(db_path, full_path).is_some();
        self.history_copy_btn.set_sensitive(has_edits);
        self.history_discard_btn.set_sensitive(has_edits);
        self.history_paste_btn.set_sensitive(
            self.history_clipboard.borrow().is_some());
        refresh_history_readout(&self.history_lbl, db_path, full_path);

        // Rebuild tag chips (display only — no notify on a mere selection change)
        rebuild_tags_flow(&self.tags_flow, &self.ctx, &self.on_tags_changed);
        self.tag_entry.set_text("");
    }
}

impl Default for MetadataPanel {
    fn default() -> Self { Self::new() }
}

// ── Tag helpers ───────────────────────────────────────────────────────────

/// Rebuild the per-image tag chips. Each chip carries an inline ✕ button that
/// detaches the tag and refreshes; `ctx` (live path/db) and `notify` (the
/// "tags mutated" hook) are threaded through so the detach handler can re-read
/// the current image and fan the change out to the left-panel Tags list.
///
/// Rebuilding alone never fires `notify` — only an actual detach does — so a
/// mere selection change repaints chips without spuriously refreshing siblings.
fn rebuild_tags_flow(
    flow: &gtk4::FlowBox,
    ctx: &std::rc::Rc<std::cell::RefCell<(String, String)>>,
    notify: &std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>>,
) {
    // Clear existing chips
    while let Some(child) = flow.first_child() {
        flow.remove(&child);
    }

    let (full_path, db_path) = ctx.borrow().clone();
    let tags = load_tags(&full_path, &db_path);
    if tags.is_empty() {
        let lbl = gtk4::Label::builder().label("(none)").build();
        lbl.add_css_class("dim-label");
        flow.insert(&lbl, -1);
        return;
    }

    for (tag_id, name) in tags {
        let chip = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(2)
            .margin_start(4).margin_end(4)
            .margin_top(2).margin_bottom(2)
            .build();
        chip.add_css_class("tag");

        let lbl = gtk4::Label::builder().label(&name).build();
        chip.append(&lbl);

        let close = gtk4::Button::builder()
            .icon_name("window-close-symbolic")
            .tooltip_text("Remove tag")
            .build();
        close.add_css_class("flat");
        close.add_css_class("circular");

        // Detach reads the *current* image from ctx at click time (the chip may
        // outlive a selection change in flight), removes the tag, rebuilds the
        // chips, then fires the shared notify so the left-panel counts update.
        let flow_w   = flow.downgrade();
        let ctx_c    = ctx.clone();
        let notify_c = notify.clone();
        close.connect_clicked(move |_| {
            let (p, d) = ctx_c.borrow().clone();
            if d.is_empty() { return; }
            detach_tag_from_image(&p, &d, tag_id);
            if let Some(flow) = flow_w.upgrade() {
                rebuild_tags_flow(&flow, &ctx_c, &notify_c);
            }
            let cb = notify_c.borrow().clone();
            if let Some(cb) = cb { cb(); }
        });
        chip.append(&close);

        flow.insert(&chip, -1);
    }
}

/// Tags attached to an image, as `(tag_id, name)` so chips can offer detach.
/// Best-effort: structural faults log (parity with `detach_tag_from_image`); an
/// uncatalogued image is a silent empty (nothing attached), as is a single tag
/// whose name can't be resolved (best-effort display, not worth spamming).
/// NOTE: this runs on every lighttable selection change, so the `Err` logs fire
/// per-repaint under a genuinely broken db — kept loud on purpose (the healthy
/// `Ok(None)` path is silent); if this ever ships to users, latch "db unhealthy"
/// once per session rather than making this log quieter.
fn load_tags(full_path: &str, db_path: &str) -> Vec<(u32, String)> {
    if db_path.is_empty() { return Vec::new(); }
    // Session-only open: reloading an image's tags fires on every lighttable
    // selection, so skip the durable-schema DDL (bootstrapped once at startup).
    let conn = match c41_db::schema::open_catalog_session(db_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("darkroom: cannot open library db to load tags: {e}"); return Vec::new(); }
    };
    let imgid = match c41_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        Ok(None) => return Vec::new(),   // image not in catalog — no tags
        Err(e) => { eprintln!("darkroom: image lookup failed on load tags: {e}"); return Vec::new(); }
    };
    let attached = match c41_db::tags::tag_get_attached(&conn, imgid) {
        Ok(ids) => ids,
        Err(e) => { eprintln!("darkroom: cannot read attached tags: {e}"); return Vec::new(); }
    };
    attached
        .into_iter()
        .filter_map(|id| {
            c41_db::tags::tag_get_name(&conn, id).ok().flatten().map(|n| (id, n))
        })
        .collect()
}

/// Detach a tag from the image at `full_path` (best-effort; logs faults).
fn detach_tag_from_image(full_path: &str, db_path: &str, tag_id: u32) {
    // Full open_catalog: a rare write self-heals the durable schema (vs the
    // session opener the read-hot paths use); see write_tag_rename.
    let conn = match c41_db::schema::open_catalog(db_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("darkroom: cannot open library db to detach tag: {e}"); return; }
    };
    let imgid = match c41_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        Ok(None) => return,            // image not in catalog — nothing to detach
        Err(e) => { eprintln!("darkroom: image lookup failed on tag detach: {e}"); return; }
    };
    if let Err(e) = c41_db::tags::tag_detach(&conn, tag_id, imgid) {
        eprintln!("darkroom: tag detach failed: {e}");
    }
}

/// Create the tag if needed and attach it to the image at `full_path`
/// (best-effort; logs faults — parity with `detach_tag_from_image`). An
/// uncatalogued image is a silent no-op (nothing to attach to).
fn add_tag_to_image(full_path: &str, db_path: &str, tag_name: &str) {
    // Full open_catalog: a rare write self-heals the durable schema (vs the
    // session opener the read-hot paths use); see write_tag_rename.
    let conn = match c41_db::schema::open_catalog(db_path) {
        Ok(c) => c,
        Err(e) => { eprintln!("darkroom: cannot open library db to add tag: {e}"); return; }
    };
    let imgid = match c41_db::image::image_get_id_by_path(&conn, full_path) {
        Ok(Some(id)) => id,
        Ok(None) => return,            // image not in catalog — nothing to attach
        Err(e) => { eprintln!("darkroom: image lookup failed on tag add: {e}"); return; }
    };
    // Create tag if it doesn't exist, then attach it.
    match c41_db::tags::tag_new(&conn, tag_name) {
        Ok(Some(tag_id)) => {
            if let Err(e) = c41_db::tags::tag_attach(&conn, tag_id, imgid) {
                eprintln!("darkroom: tag attach failed: {e}");
            }
        }
        Ok(None) => eprintln!("darkroom: could not create or find tag \u{201c}{tag_name}\u{201d}"),
        Err(e) => eprintln!("darkroom: tag create failed: {e}"),
    }
}

/// The EXIF facts darktable's "image information" module shows, as read from the
/// catalog (m4-100). Every field is optional because darktable leaves them NULL
/// for images it couldn't read EXIF from.
///
/// Reading each column with `.ok().flatten()` also contains a per-column type
/// error to that one field, so an oddly-typed value blanks one row rather than the
/// whole panel. It does **not** cover a missing *table* — that fails at prepare
/// time and is handled separately in [`query_exif_conn`].
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct ExifInfo {
    pub maker: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    /// Shutter time in seconds.
    pub exposure: Option<f64>,
    /// f-number.
    pub aperture: Option<f64>,
    pub iso: Option<f64>,
    /// Focal length in mm.
    pub focal_length: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    /// `YYYY-MM-DD HH:MM:SS`, already decoded by SQLite.
    pub datetime: Option<String>,
}

/// Read the EXIF row for an image in ONE query (rather than a connection per
/// field). Best-effort: a missing db, an uncatalogued image, or a catalog without
/// the maker/model/lens tables all yield `None`, and the panel then shows dashes.
fn query_exif(full_path: &str, db_path: &str) -> Option<ExifInfo> {
    if db_path.is_empty() {
        return None;
    }
    let conn = rusqlite::Connection::open(db_path).ok()?;
    // Brief wait, NOT the 3s used elsewhere: this runs on the GTK main thread on
    // every selection change, and library.db is in rollback-journal mode (no WAL),
    // so a reader really does block on an in-flight rating/colour write. A dash for
    // one frame beats freezing the window while arrow-keying through images.
    let _ = conn.busy_timeout(std::time::Duration::from_millis(250));
    let p = std::path::Path::new(full_path);
    let filename = p.file_name()?.to_str()?;
    let folder = p.parent()?.to_str()?;
    query_exif_conn(&conn, folder, filename)
}

/// Testable core of [`query_exif`] (same split as `persist.rs`'s `_conn` helpers),
/// so the SQL can be exercised against an in-memory catalog with no temp files.
fn query_exif_conn(conn: &rusqlite::Connection, folder: &str, filename: &str) -> Option<ExifInfo> {
    // A catalog predating the maker/model/lens lookup tables would fail this
    // statement at PREPARE time — losing every field, including the dimensions
    // that worked before this query absorbed them. So probe once and drop those
    // three columns (and their joins) when the tables aren't there: camera/lens
    // degrade to "unknown" while the rest of the row still populates. Per-column
    // NULL handling can't cover this, because the failure isn't per-column.
    let have_lookups = conn
        .query_row(
            "SELECT COUNT(*) FROM main.sqlite_master \
             WHERE type = 'table' AND name IN ('makers', 'models', 'lens')",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0)
        == 3;
    let (cols, joins) = if have_lookups {
        (
            "mk.name, md.name, l.name",
            "LEFT JOIN main.makers mk ON mk.id = i.maker_id \
             LEFT JOIN main.models md ON md.id = i.model_id \
             LEFT JOIN main.lens   l  ON l.id  = i.lens_id ",
        )
    } else {
        ("NULL, NULL, NULL", "")
    };
    // Date decoding reuses the timeline's epoch expression so the panel can't
    // disagree with the timeline about when a photo was taken.
    let sql = format!(
        "SELECT {cols}, i.exposure, i.aperture, i.iso, \
                i.focal_length, i.width, i.height, \
                CASE WHEN i.datetime_taken > 0 \
                     THEN datetime({secs}, 'unixepoch') ELSE NULL END \
         FROM main.images i \
         JOIN main.film_rolls f ON f.id = i.film_id \
         {joins}WHERE f.folder = ?1 AND i.filename = ?2",
        secs = crate::lighttable::timeline::unix_secs_sql_expr("i.datetime_taken"),
    );
    conn.query_row(&sql, rusqlite::params![folder, filename], |r| {
        Ok(ExifInfo {
            maker: r.get(0).ok().flatten(),
            model: r.get(1).ok().flatten(),
            lens: r.get(2).ok().flatten(),
            exposure: r.get(3).ok().flatten(),
            aperture: r.get(4).ok().flatten(),
            iso: r.get(5).ok().flatten(),
            focal_length: r.get(6).ok().flatten(),
            width: r.get(7).ok().flatten(),
            height: r.get(8).ok().flatten(),
            datetime: r.get(9).ok().flatten(),
        })
    })
    .ok()
}

/// Placeholder for a metadata value the catalog doesn't have (an em dash).
const NO_VALUE: &str = "\u{2014}";

/// Camera name from maker + model, as darktable presents it. The model often
/// already repeats the maker (`Canon` / `Canon EOS 5D`), so the prefix is dropped
/// to avoid "Canon Canon EOS 5D". Pure.
pub(crate) fn format_camera(maker: Option<&str>, model: Option<&str>) -> String {
    let maker = maker.unwrap_or("").trim();
    let model = model.unwrap_or("").trim();
    if model.is_empty() {
        return if maker.is_empty() { NO_VALUE.into() } else { maker.into() };
    }
    if maker.is_empty() {
        return model.into();
    }
    // Compare against the maker's FIRST WORD, case-insensitively. Makers are often
    // stored with a corporate suffix the model omits ("NIKON CORPORATION" vs
    // "NIKON D850"), so matching the whole maker string would miss the duplication
    // it's meant to catch. An empty remainder counts too, for maker == model
    // ("DJI"/"DJI" would otherwise render "DJI DJI").
    let m_low = model.to_lowercase();
    let mk_low = maker.to_lowercase();
    let first_word = mk_low.split_whitespace().next().unwrap_or("");
    if !first_word.is_empty() {
        if let Some(rest) = m_low.strip_prefix(first_word) {
            // Require a non-alphanumeric boundary so "Canonball" isn't treated as
            // "Canon" + "ball".
            if rest.is_empty() || rest.starts_with(|c: char| !c.is_alphanumeric()) {
                return model.into(); // the model already carries the maker
            }
        }
    }
    format!("{maker} {model}")
}

/// Shutter time the way photographers read it: `1/60 s` below a second (darktable
/// shows the reciprocal), `2.5 s` above. Pure.
pub(crate) fn format_exposure(secs: Option<f64>) -> String {
    match secs {
        Some(s) if s > 0.0 && s.is_finite() => {
            if s >= 1.0 {
                // Trim a trailing `.0` so 2 s doesn't read "2.0 s".
                let t = format!("{s:.1}");
                format!("{} s", t.strip_suffix(".0").unwrap_or(&t))
            } else {
                // Guard the reciprocal explicitly: `is_finite` above allows
                // denormals, whose 1/s overflows and saturates `as i64` to
                // i64::MAX — a nonsense "1/9223372036854775807 s".
                let denom = (1.0 / s).round();
                if denom.is_finite() && denom <= i64::MAX as f64 {
                    format!("1/{} s", denom as i64)
                } else {
                    NO_VALUE.into()
                }
            }
        }
        _ => NO_VALUE.into(),
    }
}

/// f-number as `f/2.8`, dropping a trailing `.0` (`f/8`, not `f/8.0`). Pure.
pub(crate) fn format_aperture(f: Option<f64>) -> String {
    match f {
        Some(v) if v > 0.0 && v.is_finite() => {
            let t = format!("{v:.1}");
            format!("f/{}", t.strip_suffix(".0").unwrap_or(&t))
        }
        _ => NO_VALUE.into(),
    }
}

/// ISO as a whole number (the catalog stores it as REAL). Pure.
pub(crate) fn format_iso(iso: Option<f64>) -> String {
    match iso {
        Some(v) if v > 0.0 && v.is_finite() => format!("{}", v.round() as i64),
        _ => NO_VALUE.into(),
    }
}

/// Focal length as `45 mm`, dropping a trailing `.0`. Pure.
pub(crate) fn format_focal(mm: Option<f64>) -> String {
    match mm {
        Some(v) if v > 0.0 && v.is_finite() => {
            let t = format!("{v:.1}");
            format!("{} mm", t.strip_suffix(".0").unwrap_or(&t))
        }
        _ => NO_VALUE.into(),
    }
}

/// Pixel dimensions as `4640 × 3472`. Pure.
pub(crate) fn format_dims(w: Option<i64>, h: Option<i64>) -> String {
    match (w, h) {
        (Some(w), Some(h)) if w > 0 && h > 0 => format!("{w} \u{00d7} {h}"),
        _ => NO_VALUE.into(),
    }
}

/// A value or the em-dash placeholder, for the plain string fields. Pure.
pub(crate) fn format_opt(v: Option<&str>) -> String {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => NO_VALUE.into(),
    }
}

// (`query_dims` was retired in m4-100: dimensions now ride along with the rest of
// the EXIF row in `query_exif`'s single query, instead of opening their own
// connection for two columns.)

fn format_bytes(n: u64) -> String {
    if n < 1024 { format!("{n} B") }
    else if n < 1024 * 1024 { format!("{:.1} KB", n as f64 / 1024.0) }
    else { format!("{:.1} MB", n as f64 / (1024.0 * 1024.0)) }
}

#[cfg(test)]
mod exif_format_tests {
    use super::*;

    #[test]
    fn camera_drops_a_maker_the_model_already_repeats() {
        // darktable's models often already carry the maker; naively concatenating
        // gives "Canon Canon EOS 5D".
        assert_eq!(format_camera(Some("Canon"), Some("Canon EOS 5D")), "Canon EOS 5D");
        assert_eq!(format_camera(Some("CANON"), Some("Canon EOS 5D")), "Canon EOS 5D");
        // The real catalog's shape: the model does NOT repeat the maker.
        assert_eq!(
            format_camera(Some("OLYMPUS CORPORATION"), Some("E-M10 Mark III")),
            "OLYMPUS CORPORATION E-M10 Mark III"
        );
        // Only a word-boundary prefix counts, so a model isn't mangled mid-word.
        assert_eq!(format_camera(Some("Canon"), Some("Canonball")), "Canon Canonball");
        // maker == model (DJI stores both the same) must collapse, not double up.
        assert_eq!(format_camera(Some("DJI"), Some("DJI")), "DJI");
        // The corporate-suffix shape: the model repeats only the maker's first
        // word, so matching the whole maker string would miss the duplication.
        assert_eq!(
            format_camera(Some("NIKON CORPORATION"), Some("NIKON D850")),
            "NIKON D850"
        );
        assert_eq!(
            format_camera(Some("SONY"), Some("SONY ILCE-7M3")),
            "SONY ILCE-7M3"
        );
        // Either side missing degrades to the other, never to a stray space.
        assert_eq!(format_camera(None, Some("E-M10")), "E-M10");
        assert_eq!(format_camera(Some("Nikon"), None), "Nikon");
        assert_eq!(format_camera(None, None), NO_VALUE);
        assert_eq!(format_camera(Some("  "), Some("")), NO_VALUE, "blank is not a value");
    }

    #[test]
    fn exposure_reads_as_a_shutter_speed() {
        // Sub-second times read as a reciprocal, the way a photographer says them.
        // 0.016666… is the real catalog's value for 1/60.
        assert_eq!(format_exposure(Some(0.0166666675359011)), "1/60 s");
        assert_eq!(format_exposure(Some(0.005)), "1/200 s");
        // A second or longer reads as a decimal, with a bare integer when exact.
        assert_eq!(format_exposure(Some(2.5)), "2.5 s");
        assert_eq!(format_exposure(Some(2.0)), "2 s");
        assert_eq!(format_exposure(Some(1.0)), "1 s");
        // Absent/degenerate values show the placeholder, never "1/inf" or NaN.
        assert_eq!(format_exposure(None), NO_VALUE);
        assert_eq!(format_exposure(Some(0.0)), NO_VALUE);
        assert_eq!(format_exposure(Some(-1.0)), NO_VALUE);
        assert_eq!(format_exposure(Some(f64::NAN)), NO_VALUE);
        assert_eq!(format_exposure(Some(f64::INFINITY)), NO_VALUE);
        // A denormal is finite, so the reciprocal needs its own guard or it
        // saturates `as i64` into "1/9223372036854775807 s".
        assert_eq!(format_exposure(Some(f64::MIN_POSITIVE / 2.0)), NO_VALUE);
    }

    #[test]
    fn aperture_iso_and_focal_drop_trailing_zeros() {
        // Real catalog values carry float noise (2.79999995…) — round, don't truncate.
        assert_eq!(format_aperture(Some(2.79999995231628)), "f/2.8");
        assert_eq!(format_aperture(Some(9.0)), "f/9");
        assert_eq!(format_aperture(None), NO_VALUE);
        assert_eq!(format_aperture(Some(0.0)), NO_VALUE);

        assert_eq!(format_iso(Some(640.0)), "640");
        assert_eq!(format_iso(Some(199.6)), "200");
        assert_eq!(format_iso(None), NO_VALUE);
        assert_eq!(format_iso(Some(f64::NAN)), NO_VALUE);

        assert_eq!(format_focal(Some(45.0)), "45 mm");
        assert_eq!(format_focal(Some(10.5)), "10.5 mm");
        assert_eq!(format_focal(None), NO_VALUE);
    }

    #[test]
    fn dims_and_opt_render_or_placeholder() {
        assert_eq!(format_dims(Some(4640), Some(3472)), "4640 × 3472");
        // A half-known or zero size is not a size.
        assert_eq!(format_dims(Some(4640), None), NO_VALUE);
        assert_eq!(format_dims(Some(0), Some(3472)), NO_VALUE);
        assert_eq!(format_dims(None, None), NO_VALUE);

        assert_eq!(format_opt(Some("Olympus M.Zuiko 45mm")), "Olympus M.Zuiko 45mm");
        assert_eq!(format_opt(Some("  padded  ")), "padded");
        assert_eq!(format_opt(Some("   ")), NO_VALUE, "whitespace is not a value");
        assert_eq!(format_opt(None), NO_VALUE);
    }

    #[test]
    fn query_exif_reads_a_seeded_catalog_end_to_end() {
        // Exercises the real SQL — the joins, the LEFT JOINs for a lens-less image,
        // and the shared date decode — against the schema shape darktable writes.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE makers (id INTEGER PRIMARY KEY, name TEXT);
             CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT);
             CREATE TABLE lens   (id INTEGER PRIMARY KEY, name TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT,
                                  maker_id INTEGER, model_id INTEGER, lens_id INTEGER,
                                  exposure REAL, aperture REAL, iso REAL,
                                  focal_length REAL, width INTEGER, height INTEGER,
                                  datetime_taken INTEGER);
             INSERT INTO film_rolls VALUES (1, '/photos');
             INSERT INTO makers VALUES (1, 'OLYMPUS CORPORATION');
             INSERT INTO models VALUES (1, 'E-M10 Mark III');
             INSERT INTO lens   VALUES (1, 'Olympus M.Zuiko Digital 45mm F1.8');
             -- Real values lifted from the catalog, incl. the 2018-07-28 timestamp.
             INSERT INTO images VALUES (1, 1, 'P1010153.ORF', 1, 1, 1,
                 0.0166666675359011, 2.79999995231628, 640.0, 45.0, 4640, 3472,
                 63668412473000000);
             -- An image with no EXIF at all: every LEFT JOIN misses, date is 0.
             INSERT INTO images VALUES (2, 1, 'bare.ORF', NULL, NULL, NULL,
                 NULL, NULL, NULL, NULL, NULL, NULL, 0);",
        )
        .unwrap();

        let e = query_exif_conn(&conn, "/photos", "P1010153.ORF").expect("row");
        assert_eq!(e.maker.as_deref(), Some("OLYMPUS CORPORATION"));
        assert_eq!(e.model.as_deref(), Some("E-M10 Mark III"));
        assert_eq!(e.lens.as_deref(), Some("Olympus M.Zuiko Digital 45mm F1.8"));
        assert_eq!(format_exposure(e.exposure), "1/60 s");
        assert_eq!(format_aperture(e.aperture), "f/2.8");
        assert_eq!(format_iso(e.iso), "640");
        assert_eq!(format_focal(e.focal_length), "45 mm");
        assert_eq!(format_dims(e.width, e.height), "4640 × 3472");
        // Decoded through the timeline's shared epoch expression.
        assert_eq!(e.datetime.as_deref(), Some("2018-07-28 22:07:53"));

        // A row with no EXIF yields Some(all-None) — the panel then shows dashes
        // rather than blanks, and never mistakes "no EXIF" for "no such image".
        let bare = query_exif_conn(&conn, "/photos", "bare.ORF").expect("row exists");
        assert_eq!(bare, ExifInfo::default());
        assert_eq!(format_camera(bare.maker.as_deref(), bare.model.as_deref()), NO_VALUE);

        // An uncatalogued image is None; an empty db path short-circuits before
        // any connection is opened.
        assert_eq!(query_exif_conn(&conn, "/photos", "missing.ORF"), None);
        assert_eq!(query_exif("/photos/P1010153.ORF", ""), None);
    }

    #[test]
    fn null_datetime_taken_reads_as_absent_not_year_one() {
        // The real catalog has NULL (not just 0) datetimes. `NULL > 0` is NULL, so
        // the CASE takes its ELSE branch — but that path was untested.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE makers (id INTEGER PRIMARY KEY, name TEXT);
             CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT);
             CREATE TABLE lens   (id INTEGER PRIMARY KEY, name TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT,
                                  maker_id INTEGER, model_id INTEGER, lens_id INTEGER,
                                  exposure REAL, aperture REAL, iso REAL,
                                  focal_length REAL, width INTEGER, height INTEGER,
                                  datetime_taken INTEGER);
             INSERT INTO film_rolls VALUES (1, '/photos');
             INSERT INTO images VALUES (1, 1, 'n.ORF', NULL, NULL, NULL,
                 NULL, NULL, NULL, NULL, 100, 50, NULL);",
        )
        .unwrap();
        let e = query_exif_conn(&conn, "/photos", "n.ORF").expect("row");
        assert_eq!(e.datetime, None);
        assert_eq!(format_opt(e.datetime.as_deref()), NO_VALUE);
        // The rest of the row still populates.
        assert_eq!(format_dims(e.width, e.height), "100 × 50");
    }

    #[test]
    fn a_catalog_without_the_lookup_tables_still_yields_the_other_fields() {
        // Pre-lookup-table catalogs would fail the joined statement at PREPARE
        // time, blanking EVERY field — including the dimensions that worked before
        // this query absorbed them. Camera/lens must degrade alone.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT,
                                  maker_id INTEGER, model_id INTEGER, lens_id INTEGER,
                                  exposure REAL, aperture REAL, iso REAL,
                                  focal_length REAL, width INTEGER, height INTEGER,
                                  datetime_taken INTEGER);
             INSERT INTO film_rolls VALUES (1, '/photos');
             INSERT INTO images VALUES (1, 1, 'old.ORF', 1, 1, 1,
                 0.0166666675359011, 2.79999995231628, 640.0, 45.0, 4640, 3472,
                 63668412473000000);",
        )
        .unwrap();
        let e = query_exif_conn(&conn, "/photos", "old.ORF").expect("row despite no lookups");
        assert_eq!(e.maker, None);
        assert_eq!(e.model, None);
        assert_eq!(e.lens, None);
        assert_eq!(format_camera(e.maker.as_deref(), e.model.as_deref()), NO_VALUE);
        // Everything not sourced from a lookup table survives.
        assert_eq!(format_exposure(e.exposure), "1/60 s");
        assert_eq!(format_aperture(e.aperture), "f/2.8");
        assert_eq!(format_iso(e.iso), "640");
        assert_eq!(format_dims(e.width, e.height), "4640 × 3472");
        assert_eq!(e.datetime.as_deref(), Some("2018-07-28 22:07:53"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_scope_hint_hides_for_zero_and_one_and_counts_above() {
        // The single-image case must look exactly as it did before m4-145, so
        // 0 and 1 hide the hint; anything more states the fan-out honestly.
        assert_eq!(meta_scope_hint_text(0), None);
        assert_eq!(meta_scope_hint_text(1), None);
        assert_eq!(meta_scope_hint_text(2).as_deref(), Some("Edits apply to all 2 selected images"));
        assert_eq!(
            meta_scope_hint_text(17).as_deref(),
            Some("Edits apply to all 17 selected images")
        );
    }

    #[test]
    fn metadata_target_paths_fall_back_to_the_cursor_only_without_a_set() {
        // The selection thread-local is empty in a fresh test process (and each
        // test thread gets its own), so this pins the fallback half of the
        // contract: no multi-selection means exactly the cursor image. The
        // set-populated half is exercised by lighttable::selection's own tests
        // plus the Docker pass; this one only needs to stay display-free.
        assert_eq!(metadata_target_paths("/d/only.nef"), vec!["/d/only.nef"]);
        assert!(metadata_target_paths("").is_empty(), "no image showing → nothing to target");
    }

    #[test]
    fn metadata_target_paths_bind_to_the_displayed_image_over_a_foreign_set() {
        // m4-145 review MAJOR-2: the grid cursor can leave the selection set
        // (native keynav, preview stepping) without the set following. When it
        // does, the DISPLAYED image must win — binding edits to a set that
        // excludes the text on screen would write where the user isn't looking.
        crate::lighttable::selection::clear();
        crate::lighttable::selection::toggle("/d/a.nef");
        crate::lighttable::selection::toggle("/d/b.nef");
        // Cursor inside the set: the whole set binds (the fan-out case).
        assert_eq!(
            metadata_target_paths("/d/b.nef"),
            vec!["/d/a.nef", "/d/b.nef"]
        );
        // Cursor outside the set: display wins, nothing smears onto {a, b}.
        assert_eq!(metadata_target_paths("/d/c.nef"), vec!["/d/c.nef"]);
        crate::lighttable::selection::clear();
    }

    #[test]
    fn commit_report_is_silent_only_when_everything_landed() {
        // Plain success must look exactly as before m4-145: no toast at all.
        assert_eq!(metadata_commit_report("title", 3, 3, 0), None);
    }

    #[test]
    fn commit_report_counts_uncatalogued_skips_instead_of_dropping_them() {
        // Review BLOCKER-1's scenario: a 100-image batch where 99 rows lost
        // their catalogue entries must SAY so, not pretend the batch landed.
        assert_eq!(
            metadata_commit_report("title", 1, 100, 0),
            Some("Saved title for 1 of 100 images".to_string())
        );
    }

    #[test]
    fn commit_report_keeps_the_could_not_save_line_when_nothing_lands() {
        assert_eq!(
            metadata_commit_report("title", 0, 2, 0),
            Some("Could not save title".to_string())
        );
    }

    #[test]
    fn commit_report_joins_sidecar_failures_and_combines_parts() {
        // Standalone sidecar failure reads exactly as the pre-review toast did:
        assert_eq!(
            metadata_commit_report("creator", 3, 3, 2),
            Some("Could not update 2 XMP sidecar(s)".to_string())
        );
        // Skip and sidecar failure combine into ONE line, not two popups:
        assert_eq!(
            metadata_commit_report("title", 2, 5, 1),
            Some(
                "Saved title for 2 of 5 images; could not update 1 XMP sidecar(s)"
                    .to_string()
            )
        );
    }

    #[test]
    fn section_pref_keys_are_distinct_and_namespaced() {
        // One key per section; a collision would make two sections share a fold
        // state, and an un-namespaced key could clash with an unrelated pref.
        let keys = [
            COLLECTIONS_SECTION_PREF_KEY,
            COLOURS_SECTION_PREF_KEY,
            TAGS_SECTION_PREF_KEY,
        ];
        let uniq: std::collections::BTreeSet<_> = keys.iter().collect();
        assert_eq!(uniq.len(), keys.len(), "duplicate section pref key");
        for k in keys {
            assert!(k.starts_with("left_section_"), "un-namespaced key: {k}");
        }
    }

    #[test]
    fn section_fold_state_round_trips_through_the_token_encoding() {
        // collapsible_section stores !expanded and restores !collapsed; pin the
        // double negation so a future edit can't invert the saved state (which
        // would silently reopen every section the user closed).
        for expanded in [true, false] {
            let tok = crate::collapsed_token(!expanded);
            let restored = crate::parse_collapsed_token(tok).map(|c| !c);
            assert_eq!(restored, Some(expanded), "round trip failed for {expanded}");
        }
    }

    fn t(id: u32, name: &str, count: i64) -> (u32, String, i64) {
        (id, name.to_string(), count)
    }

    #[test]
    fn rename_failure_message_flags_the_clash_only_for_constraint_violations() {
        // A UNIQUE clash (SQLITE_CONSTRAINT == 19) → the clash message, echoing
        // the destination path but only claiming a clash (not that new_full
        // itself exists — the collision may be a descendant's rewritten path).
        let dup = rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(19), None);
        let msg = rename_failure_message(&dup, "places|Italy");
        assert!(msg.contains("clash"), "{msg}");
        assert!(msg.contains("places|Italy"), "{msg}");

        // Any other error → the generic message, never claiming a clash.
        let other = rusqlite::Error::QueryReturnedNoRows;
        let msg = rename_failure_message(&other, "places|Italy");
        assert!(!msg.contains("clash"), "{msg}");
        assert!(msg.contains("could not be renamed"), "{msg}");
        assert!(msg.contains("places|Italy"), "{msg}");
    }

    #[test]
    fn resplice_rewrites_last_segment_keeping_parent() {
        assert_eq!(
            respliced_tag_path("places|Italy", "Italia").as_deref(),
            Some("places|Italia"),
        );
        assert_eq!(
            respliced_tag_path("a|b|c", "z").as_deref(),
            Some("a|b|z"),
        );
    }

    #[test]
    fn resplice_top_level_tag_has_no_parent() {
        assert_eq!(respliced_tag_path("landscape", "scenery").as_deref(), Some("scenery"));
    }

    #[test]
    fn resplice_is_noop_when_blank_or_unchanged() {
        assert_eq!(respliced_tag_path("places|Italy", "Italy"), None);
        assert_eq!(respliced_tag_path("places|Italy", "   "), None);
        assert_eq!(respliced_tag_path("places|Italy", ""), None);
        // Trimmed input equal to the current segment is still a no-op.
        assert_eq!(respliced_tag_path("places|Italy", "  Italy  "), None);
    }

    #[test]
    fn resplice_trims_input() {
        assert_eq!(
            respliced_tag_path("places|Italy", "  Italia  ").as_deref(),
            Some("places|Italia"),
        );
    }

    #[test]
    fn resplice_rejects_pipe_in_segment() {
        // A typed `|` would re-parent/deepen the tree (out of scope for an
        // in-place segment rename) and could let the rewrite self-collide.
        assert_eq!(respliced_tag_path("places|Italy", "Italy|north"), None);
        assert_eq!(respliced_tag_path("places|Italy", "a|b"), None);
        assert_eq!(respliced_tag_path("landscape", "a|b"), None);
    }

    #[test]
    fn color_filter_name_covers_every_label_in_range() {
        // Every colour in the DAO's domain has a name; the array length tracks
        // COLOR_COUNT so the two can't silently drift apart.
        assert_eq!(COLOR_NAMES.len(), COLOR_COUNT as usize);
        assert_eq!(color_filter_name(0), Some("Red"));
        assert_eq!(color_filter_name(4), Some("Purple"));
        for idx in 0..COLOR_COUNT {
            assert!(color_filter_name(idx).is_some(), "idx {idx} unnamed");
        }
    }

    #[test]
    fn color_filter_name_is_none_out_of_range() {
        assert_eq!(color_filter_name(COLOR_COUNT), None);
        assert_eq!(color_filter_name(99), None);
    }

    #[test]
    fn flatten_flat_tags_are_depth_zero_reals() {
        let rows = flatten_tag_tree(&[t(1, "landscape", 4), t(2, "portrait", 2)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], TagTreeRow { depth: 0, label: "landscape".into(), full_name: "landscape".into(), id: Some(1), count: 4 });
        assert_eq!(rows[1], TagTreeRow { depth: 0, label: "portrait".into(), full_name: "portrait".into(), id: Some(2), count: 2 });
    }

    #[test]
    fn flatten_synthesises_virtual_parent() {
        // "places" is not itself a tag — it must appear as a virtual group.
        let rows = flatten_tag_tree(&[t(7, "places|Italy", 3)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], TagTreeRow { depth: 0, label: "places".into(), full_name: "places".into(), id: None, count: 0 });
        assert_eq!(rows[1], TagTreeRow { depth: 1, label: "Italy".into(), full_name: "places|Italy".into(), id: Some(7), count: 3 });
    }

    #[test]
    fn flatten_real_parent_keeps_its_id_and_count() {
        // "places" is BOTH a tag and a parent.
        let rows = flatten_tag_tree(&[t(1, "places", 5), t(2, "places|Italy", 3)]);
        assert_eq!(rows[0], TagTreeRow { depth: 0, label: "places".into(), full_name: "places".into(), id: Some(1), count: 5 });
        assert_eq!(rows[1], TagTreeRow { depth: 1, label: "Italy".into(), full_name: "places|Italy".into(), id: Some(2), count: 3 });
    }

    #[test]
    fn flatten_orders_children_alphabetically_per_level() {
        // Input order is irrelevant; siblings sort, subtrees stay grouped.
        let rows = flatten_tag_tree(&[t(3, "b", 1), t(1, "a|z", 1), t(2, "a|m", 1)]);
        let shape: Vec<(usize, &str)> = rows.iter().map(|r| (r.depth, r.label.as_str())).collect();
        assert_eq!(shape, vec![(0, "a"), (1, "m"), (1, "z"), (0, "b")]);
    }

    #[test]
    fn flatten_handles_three_levels() {
        let rows = flatten_tag_tree(&[t(1, "a|b|c", 9)]);
        let shape: Vec<(usize, &str, Option<u32>)> =
            rows.iter().map(|r| (r.depth, r.label.as_str(), r.id)).collect();
        assert_eq!(shape, vec![(0, "a", None), (1, "b", None), (2, "c", Some(1))]);
    }

    #[test]
    fn flatten_collapses_trailing_separator() {
        // "a|" must render as a normal `a` tag, not `a > (blank)`.
        let rows = flatten_tag_tree(&[t(1, "a|", 2)]);
        assert_eq!(rows, vec![TagTreeRow { depth: 0, label: "a".into(), full_name: "a".into(), id: Some(1), count: 2 }]);
    }

    #[test]
    fn flatten_collapses_double_and_leading_separator() {
        let rows = flatten_tag_tree(&[t(1, "a||b", 1), t(2, "|c", 1)]);
        let shape: Vec<(usize, &str, Option<u32>)> =
            rows.iter().map(|r| (r.depth, r.label.as_str(), r.id)).collect();
        assert_eq!(shape, vec![(0, "a", None), (1, "b", Some(1)), (0, "c", Some(2))]);
    }

    #[test]
    fn flatten_drops_all_separator_name() {
        // A name with no representable segment contributes nothing.
        assert!(flatten_tag_tree(&[t(1, "|", 1), t(2, "", 1)]).is_empty());
    }

    /// Temp catalogue with two images (`a.raw` id 7 and `b.raw` id 8 in
    /// `/photos`) — the copy/paste pair for the history tests. Built from
    /// `ensure_base_schema`, the same DDL production writes under.
    fn history_catalogue(tag: &str) -> (String, String, String) {
        let mut p = std::env::temp_dir();
        p.push(format!("c41-hist-{tag}-{:?}.db", std::thread::current().id()));
        let path = p.to_string_lossy().into_owned();
        let _ = std::fs::remove_file(&path);
        let conn = rusqlite::Connection::open(&path).unwrap();
        c41_db::schema::ensure_base_schema(&conn).unwrap();
        conn.execute("INSERT INTO main.film_rolls (id, folder) VALUES (1, '/photos')", [])
            .unwrap();
        conn.execute("INSERT INTO main.images (id, film_id, filename) VALUES (7, 1, 'a.raw')", [])
            .unwrap();
        conn.execute("INSERT INTO main.images (id, film_id, filename) VALUES (8, 1, 'b.raw')", [])
            .unwrap();
        (path, "/photos/a.raw".to_string(), "/photos/b.raw".to_string())
    }

    #[test]
    fn history_readout_tracks_selection_and_edits() {
        let (db, a, _b) = history_catalogue("readout");
        assert_eq!(history_readout_text(&db, ""), "(no image selected)");
        assert_eq!(history_readout_text(&db, &a), "no saved edits");

        // A params row without a stack row still counts as one step: it changes
        // how the image renders vs raw defaults even with nothing to undo.
        crate::persist::save_params(
            &db,
            &a,
            &crate::preview::PreviewParams { ev: 0.4, ..Default::default() },
        );
        assert_eq!(history_readout_text(&db, &a), "1-step edit stack");

        let mut h = crate::history::HistoryStack::new(
            "Original",
            crate::preview::PreviewParams::default(),
        );
        h.record("Exposure", crate::preview::PreviewParams { ev: 0.9, ..Default::default() });
        crate::persist::save_history(&db, &a, &h);
        assert_eq!(history_readout_text(&db, &a), "2-step edit stack");
    }

    #[test]
    fn clipboard_copies_edits_between_images_replacing_the_target() {
        let (db, src, dst) = history_catalogue("paste");
        // Source: params + 2-step stack. Destination: an older edit that paste
        // must REPLACE (darktable's copy/paste replaces the target stack).
        let edited = crate::preview::PreviewParams { ev: 2.0, ..Default::default() };
        crate::persist::save_params(&db, &src, &edited);
        let mut h = crate::history::HistoryStack::new(
            "Original",
            crate::preview::PreviewParams::default(),
        );
        h.record("Exposure", edited.clone());
        crate::persist::save_history(&db, &src, &h);
        crate::persist::save_params(&db, &dst, &crate::preview::PreviewParams { ev: -1.0, ..Default::default() });

        let clip = HistoryClipboard::from_image(&db, &src).expect("source has edits");
        assert_eq!(clip.params.ev, 2.0);
        assert_eq!(clip.stack.as_ref().map(|s| s.len()), Some(2));
        assert_eq!(clip.source_basename(), "a.raw", "toast names the source file");

        clip.apply_to(&db, &dst);
        let pasted = crate::persist::load_saved(&db, &dst).expect("params pasted");
        assert_eq!(pasted.ev, 2.0, "destination edit was replaced, not kept");
        assert_eq!(
            crate::persist::load_history(&db, &dst).map(|s| s.len()),
            Some(2),
            "stack pasted wholesale"
        );

        // Copying an unedited image yields nothing — the UI keeps Paste disabled.
        assert!(HistoryClipboard::from_image(&db, "/photos/never-edited.raw").is_none());
    }

    #[test]
    fn clipboard_paste_without_stack_row_replaces_the_targets_stale_stack() {
        // A copy of an image whose params row has NO stack row (source predates
        // the history feature, was discarded, or its blob failed to decode) must
        // not leave the target's old stack describing edits that no longer
        // exist — paste replaces the whole pair.
        let (db, src, dst) = history_catalogue("pastenostack");
        crate::persist::save_params(
            &db,
            &src,
            &crate::preview::PreviewParams { ev: 1.0, ..Default::default() },
        );
        let stale = crate::preview::PreviewParams { ev: -2.0, ..Default::default() };
        let mut h = crate::history::HistoryStack::new("Original", stale);
        h.record("Exposure", stale);
        crate::persist::save_history(&db, &dst, &h);
        crate::persist::save_params(&db, &dst, &stale);

        HistoryClipboard::from_image(&db, &src)
            .expect("params to copy")
            .apply_to(&db, &dst);

        assert_eq!(crate::persist::load_saved(&db, &dst).unwrap().ev, 1.0);
        assert_eq!(
            crate::persist::load_history(&db, &dst).map(|s| s.len()),
            Some(1),
            "stale stack replaced by a fresh seed stack"
        );
    }
}
