//! GTK4 + libadwaita UI shell for Darkroom.
//!
//! Phase 3-ui-12: live exposure preview in the darkroom view via darkroom-core.

use adw::prelude::*;
use adw::Application;
use anyhow::Result;
use gtk4::ApplicationWindow;
use glib::clone;

pub mod catalog;
pub mod crop_overlay;
pub mod darkroom;
pub mod dialogs;
pub mod export;
pub mod export_panel;
pub mod history;
pub mod lighttable;
pub mod panels;
pub mod persist;
pub mod preview;
pub mod raw_preview;
pub mod snapshots;

pub const APP_ID:        &str = "org.darkroom.Darkroom";
pub const DEFAULT_WIDTH:  i32 = 1280;
pub const DEFAULT_HEIGHT: i32 = 800;

/// Lighttable thumb-size control (bottom toolbar, m4-98a): the value is the grid's
/// *upper* column bound — fewer columns ⇒ larger thumbnails. It caps columns rather
/// than fixing them (`min_columns` stays low) so a narrow framebuffer still fits
/// the row instead of clipping. `DEFAULT` mirrors darktable's ~mid density.
const THUMB_COLS_MIN:     u32 = 2;
const THUMB_COLS_MAX:     u32 = 12;
const THUMB_COLS_DEFAULT: u32 = 6;

/// `darkroom_ui_prefs` key under which the lighttable rating filter (comparator +
/// floor) is persisted across sessions (m4-98d). The value is the compact token
/// from [`lighttable::rating_filter_token`] (`off` / `ge:N` / `eq:N` / `le:N` /
/// `rej`).
const RATING_FILTER_PREF_KEY: &str = "rating_filter";

/// `darkroom_ui_prefs` key under which the thumbnail overlay mode is persisted
/// across sessions (m4-98e). The value is the token from
/// [`lighttable::overlay_mode_token`] (`none` / `normal` / `extended`).
const OVERLAY_MODE_PREF_KEY: &str = "overlay_mode";

pub fn run() -> Result<glib::ExitCode> {
    let app = Application::builder()
        .application_id(APP_ID)
        .build();
    // Register app-level keyboard accelerators
    app.set_accels_for_action("win.import", &["<Control>i"]);
    app.set_accels_for_action("win.export-selected", &["<Control>e"]);
    app.connect_activate(build_main_window);

    // SIGTERM/SIGINT graceful shutdown. A GtkApplication installs no handler,
    // so the default disposition terminates the process instantly — the
    // container autostart trap that forwards SIGTERM and waits up to 15s would
    // be waiting on an already-dead process, dropping any pending autosave. Call
    // app.quit() so the main loop unwinds through the normal teardown path.
    // 15 = SIGTERM, 2 = SIGINT (avoids pulling in libc for two constants).
    for signum in [15, 2] {
        let app = app.downgrade();
        glib::unix_signal_add_local(signum, move || {
            if let Some(app) = app.upgrade() {
                app.quit();
            }
            glib::ControlFlow::Break
        });
    }

    Ok(app.run())
}

/// Build the darktable-style `Lighttable | Darkroom | Other` linked view switcher.
/// Returns the container plus the Lighttable and Darkroom toggles so the caller
/// can set the active view and wire navigation. "Other" (map/print/tethering) is
/// always disabled — those views aren't ported. Shared by the lighttable header
/// (Lighttable active; Darkroom pushes the selected image) and the darkroom
/// header (Darkroom active; Lighttable pops back to the grid), so the control
/// looks and behaves identically in both views.
///
/// Hand-rolled linked ToggleButton group rather than adw::ViewSwitcher: the
/// latter binds to an adw::ViewStack, but our views are a NavigationView
/// (push/pop), so this is the right fit — don't "upgrade" it to a ViewSwitcher
/// without also changing the navigation model.
pub(crate) struct ViewSwitcher {
    pub container: gtk4::Box,
    pub lighttable: gtk4::ToggleButton,
    pub darkroom: gtk4::ToggleButton,
}

pub(crate) fn build_view_switcher() -> ViewSwitcher {
    let container = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
    container.add_css_class("linked");

    let lighttable = gtk4::ToggleButton::with_label("Lighttable");
    let darkroom = gtk4::ToggleButton::with_label("Darkroom");
    darkroom.set_group(Some(&lighttable));
    // "Other" (map / print / tethering) is a disabled placeholder — those views
    // aren't ported. Insensitive so it can't break the toggle-group's
    // single-active invariant; the greyed appearance signals "unavailable". (No
    // tooltip: an insensitive widget never emits query-tooltip, so one here would
    // never show — the greying is the affordance.)
    let other = gtk4::ToggleButton::with_label("Other");
    other.set_group(Some(&lighttable));
    other.set_sensitive(false);

    container.append(&lighttable);
    container.append(&darkroom);
    container.append(&other);
    ViewSwitcher { container, lighttable, darkroom }
}

/// Wrap a view-switcher container in the standard header title: the switcher on
/// top with a dim caption below it (the app name in the lighttable, the open
/// filename in the darkroom). Shared so both view headers are the same height and
/// layout — the switcher is identical and always sits over a one-line caption.
pub(crate) fn view_switcher_title(container: &gtk4::Box, subtitle: &str) -> gtk4::Box {
    container.set_halign(gtk4::Align::Center);
    let subtitle_label = gtk4::Label::new(Some(subtitle));
    subtitle_label.add_css_class("caption");
    subtitle_label.add_css_class("dim-label");
    subtitle_label.set_ellipsize(gtk4::pango::EllipsizeMode::End);
    subtitle_label.set_max_width_chars(30);
    let title_box = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    title_box.set_valign(gtk4::Align::Center);
    title_box.set_halign(gtk4::Align::Center);
    title_box.append(container);
    title_box.append(&subtitle_label);
    title_box
}

fn build_main_window(app: &Application) {
    // darktable ships a dark grey theme; match that first impression by forcing
    // libadwaita's dark colour scheme (the default follows the desktop setting,
    // which is light in the KasmVNC container). A custom CSS provider matching
    // darktable's exact greys is a later refinement (see RUST_MIGRATION_PLAN.md).
    adw::StyleManager::default().set_color_scheme(adw::ColorScheme::ForceDark);

    let window = ApplicationWindow::builder()
        .application(app)
        .title("Darkroom")
        .default_width(DEFAULT_WIDTH)
        .default_height(DEFAULT_HEIGHT)
        .build();

    let db_path = std::env::var("DARKROOM_LIBRARY_DB").unwrap_or_default();

    // Bootstrap the DURABLE catalog schema once on a real (non-demo) library.db
    // so a fresh /config — where no C app ever ran to create the tables — has a
    // working catalog before the lighttable query, an import, or a session-only
    // tag read (open_catalog_session) touches it. open_catalog attaches data.db +
    // a throwaway in-memory `memory` db and ensures main.* + data.tags; only the
    // durable part needs to persist, so the connection is immediately dropped.
    if !db_path.is_empty() {
        match darkroom_db::schema::open_catalog(&db_path) {
            Ok(_) => {}
            Err(e) => tracing::warn!("failed to bootstrap catalog schema: {e}"),
        }
    }

    // Restore the persisted lighttable rating filter (m4-98d) BEFORE the first
    // load below, so the initial grid already reflects the saved comparator +
    // floor. Seeds thread-local state only (no reload — no loader is registered
    // yet); the bottom bar reads it back when it builds its stars + dropdown.
    if let Some(tok) = persist::load_ui_pref(&db_path, RATING_FILTER_PREF_KEY) {
        lighttable::apply_rating_filter_token(&tok);
    }
    // Likewise restore the thumbnail overlay mode (m4-98e) before the grid binds
    // its first cells, so they're laid out right the first time (no visible flip).
    if let Some(tok) = persist::load_ui_pref(&db_path, OVERLAY_MODE_PREF_KEY) {
        lighttable::apply_overlay_mode_token(&tok);
    }

    // ── Toast overlay (wraps everything for in-app notifications) ──────────
    let toast_overlay = adw::ToastOverlay::new();
    let make_toast = {
        let to = toast_overlay.clone();
        move |msg: String| {
            to.add_toast(adw::Toast::new(&msg));
        }
    };

    // ── Lighttable ─────────────────────────────────────────────────────────
    let (scroll, lt_model, lt_selection) =
        lighttable::lighttable_page(db_path.clone());
    lighttable::lighttable_load_from_db(&lt_model, &db_path);

    // ── Panels ─────────────────────────────────────────────────────────────
    // Shared "currently-filtering tag prefix" (None = no tag filter): the full
    // `parent|child` path of the clicked tag-tree node. Set by the left-panel
    // folder/tag clicks and the search/import paths below; read by the
    // tag-mutation callbacks to decide whether to re-run the hierarchical filter.
    let active_tag: std::rc::Rc<std::cell::RefCell<Option<String>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));

    let left  = panels::LeftPanel::new(&db_path, &lt_model, &active_tag);
    let right = panels::MetadataPanel::new();

    // Re-run the grid filter after a tag mutation, but ONLY when a tag filter is
    // active — so e.g. detaching the filtered-on tag drops the image from the
    // grid, while ordinary tagging under an All/Folder/Search view leaves the
    // grid (and selection) untouched. Across the reload we preserve the selected
    // image so an unrelated attach/rename doesn't yank the user back to index 0;
    // if the image left the grid (detached the filtered-on tag) the default
    // selection stands.
    let reapply_tag_filter = {
        let at  = active_tag.clone();
        let mdl = lt_model.clone();
        let sel = lt_selection.clone();
        let db  = db_path.clone();
        move || {
            // Clone the prefix out before the load so the RefCell borrow is
            // released (the loader doesn't touch active_tag, but keep it tight).
            let cur = at.borrow().clone();
            if let Some(prefix) = cur {
                let prev = lighttable::selected_path(&sel);
                lighttable::lighttable_load_by_tag_prefix(&mdl, &db, &prefix);
                lighttable::reselect_path(&sel, prev.as_deref());
            }
        }
    };

    // Bidirectional tag-change refresh:
    //  • attaching/detaching a tag in the metadata panel refreshes the
    //    left-panel Tags list (new tags / changed counts);
    //  • renaming/deleting a tag in the left panel re-renders the metadata
    //    panel's chips for the current image.
    // Both then re-run the active tag filter. Neither callback re-enters the
    // other's mutation path, so there is no loop.
    {
        let lp = left.clone();
        let reapply = reapply_tag_filter.clone();
        right.set_on_tags_changed(move || { lp.refresh_tags(); reapply(); });
    }
    {
        let meta = right.clone();
        let reapply = reapply_tag_filter.clone();
        left.set_on_tags_changed(move || { meta.refresh_tags_display(); reapply(); });
    }

    // Selection → metadata
    {
        let meta = right.clone();
        let db   = db_path.clone();
        let mdl  = lt_model.clone();
        lt_selection.connect_selection_changed(move |sel, _, _| {
            if let Some(path) = mdl.item(sel.selected())
                .and_downcast::<gtk4::StringObject>()
                .map(|o| o.string().to_string())
            {
                if path.contains('/') {
                    meta.update(&path, &db);
                }
            }
        });
    }

    // ── Lighttable page layout ─────────────────────────────────────────────
    scroll.set_hexpand(true);

    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    hbox.append(&left.widget);
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    hbox.append(&scroll);
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    hbox.append(&right.widget);

    // ── Header bar ─────────────────────────────────────────────────────────
    let lt_header = adw::HeaderBar::new();
    lt_header.set_title_widget(Some(&adw::WindowTitle::new("Darkroom", "Lighttable")));

    // Search bar
    {
        let search = gtk4::SearchEntry::builder()
            .placeholder_text("Search images…")
            .width_request(200)
            .build();
        let db = db_path.clone();
        let at = active_tag.clone();
        let lp = left.clone();
        search.connect_search_changed(clone!(@weak lt_model => move |s| {
            let query = s.text().to_string();
            lp.clear_filter_highlights();   // a name search supersedes any filter highlight
            *at.borrow_mut() = None;   // a name search supersedes any tag filter
            lighttable::lighttable_filter_by_name(&lt_model, &db, &query);
        }));
        lt_header.pack_start(&search);
    }

    // Import button
    {
        let btn = gtk4::Button::builder()
            .icon_name("list-add-symbolic")
            .tooltip_text("Import folder")
            .build();
        let db         = db_path.clone();
        let toast_fn   = make_toast.clone();
        let at         = active_tag.clone();
        let lp         = left.clone();
        btn.connect_clicked(clone!(@weak window, @weak lt_model => move |_| {
            let db_inner    = db.clone();
            let toast_inner = toast_fn.clone();
            let lp_inner    = lp.clone();
            dialogs::show_import_dialog(
                window.upcast_ref::<gtk4::Window>(),
                db.clone(),
                clone!(@weak lt_model, @strong db_inner, @strong at => move || {
                    lp_inner.clear_filter_highlights();   // post-import view shows all images
                    *at.borrow_mut() = None;   // post-import view shows all images
                    lighttable::lighttable_load_from_db(&lt_model, &db_inner);
                }),
                toast_inner,
            );
        }));
        lt_header.pack_start(&btn);
    }

    // Export selected button
    {
        let btn = gtk4::Button::builder()
            .icon_name("document-send-symbolic")
            .tooltip_text("Export selected")
            .build();
        let toast_fn = make_toast.clone();
        let export_db = db_path.clone();
        btn.connect_clicked(clone!(@weak lt_model, @weak lt_selection, @weak window => move |_| {
            let pos = lt_selection.selected();
            let paths: Vec<String> = if let Some(path) = lt_model.item(pos)
                .and_downcast::<gtk4::StringObject>()
                .map(|o| o.string().to_string())
                .filter(|p| p.contains('/'))
            {
                vec![path]
            } else { vec![] };
            dialogs::show_export_dialog(
                window.upcast_ref::<gtk4::Window>(),
                paths,
                None, // no fixed edit — each image's edit is loaded from the catalog
                Some(export_db.clone()),
                toast_fn.clone(),
            );
        }));
        lt_header.pack_end(&btn);
    }

    // Image-count indicator (m4-97a) — darktable shows the current view's image
    // count in the top bar. Bound to the model's items-changed signal so it
    // tracks every load / filter / import / tag-filter without touching each
    // load call site. Capped views (GRID_CAP) read as e.g. "2000 images".
    {
        // The grid model also carries non-image sentinel rows (a "(No results)"
        // / empty-state placeholder, or a truncation notice) — the codebase's
        // convention is that real image rows contain '/', sentinels don't. Count
        // only real images so an empty collection reads "0 images", not "1".
        fn image_count(m: &gtk4::StringList) -> u32 {
            (0..m.n_items())
                .filter(|&i| m.string(i).is_some_and(|s| s.contains('/')))
                .count() as u32
        }
        fn fmt(n: u32) -> String {
            format!("{n} image{}", if n == 1 { "" } else { "s" })
        }
        let count_label = gtk4::Label::new(Some(&fmt(image_count(&lt_model))));
        count_label.add_css_class("dim-label");
        lt_model.connect_items_changed(clone!(@weak count_label => move |m, _, _, _| {
            count_label.set_label(&fmt(image_count(m)));
        }));
        lt_header.pack_end(&count_label);
    }

    // Quick-filter preset dropdown (m4-97c) — darktable's top-bar
    // `filter [all images ▾]`. Rows come from FilterPreset::ALL (labels live beside
    // the variants) plus a trailing "custom" row that is only *shown*, never
    // chosen: it's what the dropdown displays when the bottom bar has set a filter
    // no preset names (e.g. `≤ 3`). Presets don't implement filtering themselves —
    // each is a named (comparator, stars) pair applied to the same state the bottom
    // bar drives, and an observer keeps the two in sync in both directions.
    {
        const CUSTOM_ROW: &str = "custom";
        let mut labels: Vec<String> =
            lighttable::FilterPreset::ALL.iter().map(|p| p.label()).collect();
        labels.push(CUSTOM_ROW.to_string());
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

        let filter_dd = gtk4::DropDown::from_strings(&label_refs);
        filter_dd.set_tooltip_text(Some("Quick filter the collection"));

        // Reflect the live filter state into the row selection. Registered as a
        // filter observer so a change made in the BOTTOM bar moves this dropdown
        // too, landing on "custom" when the state isn't a named preset.
        let sync_from_state = {
            let filter_dd = filter_dd.clone();
            move || {
                let idx = lighttable::current_filter_preset()
                    .map_or(lighttable::FilterPreset::CUSTOM_INDEX, |p| p.to_index());
                // (Belt-and-braces: GTK already no-ops an unchanged position, and
                // the sync guard covers the emission either way.)
                if filter_dd.selected() != idx {
                    filter_dd.set_selected(idx);
                }
            }
        };
        // Seeding BEFORE the handler is connected is load-bearing: it mirrors the
        // filter restored at startup without the write being taken for a user edit
        // (same ordering the bottom bar's comparator relies on).
        sync_from_state();
        lighttable::add_filter_observer(sync_from_state);

        filter_dd.connect_selected_notify(move |dd| {
            // Ignore the programmatic write an observer just made (it's a mirror of
            // the state, not a user edit) — otherwise this would recurse.
            if lighttable::filter_sync_in_progress() {
                return;
            }
            // "custom" is display-only. `>=` (not `==`) also catches GTK's
            // INVALID_LIST_POSITION (u32::MAX), which `from_index` would otherwise
            // map to `AllImages` and *apply*. Snap the control back to the real
            // state via the observer bus rather than calling our own sync closure —
            // that would re-enter this handler unguarded.
            if dd.selected() >= lighttable::FilterPreset::CUSTOM_INDEX {
                lighttable::sync_filter_controls();
                return;
            }
            lighttable::set_filter_preset(lighttable::FilterPreset::from_index(dd.selected()));
        });
        lt_header.pack_start(&filter_dd);
    }

    // Sort-by dropdown (m4-97b) + direction toggle (m4-97e) — darktable's top-bar
    // "sort by" with the ascending/descending arrow beside it. Changing either
    // re-renders the *current* view under the new order; the loaders self-register
    // a reload closure, so this doesn't need to know which view is showing. The
    // two controls share a `.linked` box so the arrow stays glued to the dropdown.
    {
        let sort_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
        sort_box.add_css_class("linked");

        let sort_dd = gtk4::DropDown::from_strings(&["Filename", "Date taken", "Rating"]);
        sort_dd.set_tooltip_text(Some("Sort images by"));
        sort_dd.connect_selected_notify(|dd| {
            let order = match dd.selected() {
                1 => lighttable::SortOrder::DateTaken,
                2 => lighttable::SortOrder::Rating,
                _ => lighttable::SortOrder::Filename,
            };
            lighttable::set_sort_order(order);
        });
        sort_box.append(&sort_dd);

        // Direction toggle: unpressed = ascending (natural), pressed = descending
        // (reversed). Swap the icon so the arrow always reflects the active state.
        let dir_btn = gtk4::ToggleButton::builder()
            .icon_name("view-sort-ascending-symbolic")
            .tooltip_text("Reverse sort order")
            .build();
        dir_btn.connect_toggled(|b| {
            let reverse = b.is_active();
            b.set_icon_name(if reverse {
                "view-sort-descending-symbolic"
            } else {
                "view-sort-ascending-symbolic"
            });
            lighttable::set_sort_reverse(reverse);
        });
        sort_box.append(&dir_btn);

        lt_header.pack_end(&sort_box);
    }

    let lt_toolbar = adw::ToolbarView::new();
    lt_toolbar.add_top_bar(&lt_header);
    lt_toolbar.set_content(Some(&hbox));

    // ── Bottom toolbar (m4-98a/b/d) ────────────────────────────────────────
    // darktable's lighttable bottom bar. Right (m4-98a): the thumb-size stepper
    // (its "images per row" ± control) driving the grid's max-column bound live —
    // fewer columns ⇒ bigger thumbnails. Left (m4-98b/d): a star-rating filter —
    // a comparator dropdown (≥ / = / ≤ / rejected) plus five star buttons — that
    // composes with whatever collection is active (folder / tag / colour / search)
    // and persists across sessions. Later: colour filter + view-mode switcher.
    {
        let bottom = gtk4::CenterBox::new();
        bottom.add_css_class("toolbar");

        // Rating filter (m4-98b/d). The comparator dropdown picks how the star
        // count is applied; clicking star N sets the count (re-clicking the current
        // floor drops it to 0). Both the lit stars and the dropdown selection read
        // `current_min_rating()`/`current_rating_compare()` — the single source of
        // truth — so the display can't drift from the DB query, and every change
        // persists the compact filter token so it survives a restart.
        {
            let filter_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 4);
            filter_box.set_margin_start(6);

            let star_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
            star_box.add_css_class("linked");
            star_box.set_tooltip_text(Some("Filter by star rating"));

            let stars: std::rc::Rc<Vec<gtk4::Button>> = std::rc::Rc::new(
                (1..=5u8)
                    .map(|_| {
                        gtk4::Button::builder().icon_name("non-starred-symbolic").build()
                    })
                    .collect(),
            );
            for b in stars.iter() {
                star_box.append(b);
            }

            // Persist the current filter (comparator + floor) as its compact token.
            let persist_filter: std::rc::Rc<dyn Fn()> = {
                let db = db_path.clone();
                std::rc::Rc::new(move || {
                    persist::save_ui_pref(
                        &db,
                        RATING_FILTER_PREF_KEY,
                        &lighttable::rating_filter_token(),
                    );
                })
            };

            // Repaint the stars so 1..=floor are lit and the rest hollow; in the
            // `Rejected` mode the star count is irrelevant, so grey the whole row
            // out (and light none) to signal the dropdown alone drives the filter.
            let refresh_stars: std::rc::Rc<dyn Fn()> = {
                let stars = stars.clone();
                let star_box = star_box.clone();
                std::rc::Rc::new(move || {
                    let rejected =
                        lighttable::current_rating_compare() == lighttable::RatingCompare::Rejected;
                    star_box.set_sensitive(!rejected);
                    let floor = lighttable::current_min_rating();
                    for (i, b) in stars.iter().enumerate() {
                        let lit = !rejected && (i as u8) < floor; // star i+1 lit iff i+1 <= floor
                        b.set_icon_name(if lit { "starred-symbolic" } else { "non-starred-symbolic" });
                    }
                })
            };

            // Comparator dropdown, seeded from the restored filter BEFORE the
            // handler is connected so the seeding doesn't fire a spurious reload.
            let compare = gtk4::DropDown::from_strings(&["≥", "=", "≤", "⚑"]);
            compare.set_valign(gtk4::Align::Center);
            compare.set_tooltip_text(Some("Rating comparator: ≥ / = / ≤ / rejected only"));
            compare.set_selected(lighttable::current_rating_compare().to_index());
            compare.connect_selected_notify(move |d| {
                // Skip the programmatic write an observer just made (see below).
                if lighttable::filter_sync_in_progress() {
                    return;
                }
                lighttable::set_rating_compare(lighttable::RatingCompare::from_index(d.selected()));
            });

            // One observer owns re-syncing this bar's display and persisting the
            // filter, for changes from ANY control (these stars, the comparator, or
            // the top bar's preset dropdown). The click handlers below therefore
            // only set state — `set_*` runs this via `filter_changed()`.
            {
                let compare = compare.clone();
                let refresh_stars = refresh_stars.clone();
                let persist_filter = persist_filter.clone();
                lighttable::add_filter_observer(move || {
                    let idx = lighttable::current_rating_compare().to_index();
                    if compare.selected() != idx {
                        compare.set_selected(idx);
                    }
                    refresh_stars();
                    persist_filter();
                });
            }

            refresh_stars(); // sync stars to the restored filter

            for (i, b) in stars.iter().enumerate() {
                let n = (i + 1) as u8;
                b.connect_clicked(move |_| {
                    // Consult the sync guard like every other filter control. A
                    // plain Button emits `clicked` only on real input today, so this
                    // is inert — but it stops being inert the moment these become
                    // toggles an observer sets (darktable's stars are toggles), and
                    // "safe by accident" is not a property worth relying on.
                    if lighttable::filter_sync_in_progress() {
                        return;
                    }
                    // Toggle: re-clicking the current floor drops the count to 0.
                    let new = if lighttable::current_min_rating() == n { 0 } else { n };
                    lighttable::set_min_rating(new);
                });
            }

            filter_box.append(&compare);
            filter_box.append(&star_box);
            bottom.set_start_widget(Some(&filter_box));
        }

        if let Some(grid) = scroll.child().and_downcast::<gtk4::GridView>() {
            // The grid's own `max-columns` property is the single source of truth
            // for the current thumb size — no separate counter to keep in sync.
            grid.set_max_columns(THUMB_COLS_DEFAULT);

            // Thumbnail overlay mode (m4-98e), centre of the bar: how much metadata
            // each cell shows. Seeded from the restored pref BEFORE the handler is
            // connected, so seeding can't fire a spurious re-apply.
            {
                // Rows come from OverlayMode::ALL so the control can never drift
                // from the variants (labels live next to the enum). They're terse
                // on purpose: this CenterBox already carries the rating filter at
                // the start and the thumb stepper at the end, and its minimum width
                // is the sum of all three — the ~915px lighttable overflow gotcha.
                let labels: Vec<&str> =
                    lighttable::OverlayMode::ALL.iter().map(|m| m.label()).collect();
                let overlays = gtk4::DropDown::from_strings(&labels);
                overlays.set_valign(gtk4::Align::Center);
                overlays.set_tooltip_text(Some("Thumbnail overlays: none / stars + labels / full"));
                overlays.set_selected(lighttable::current_overlay_mode().to_index());
                overlays.connect_selected_notify({
                    let grid = grid.clone();
                    let db = db_path.clone();
                    move |d| {
                        let mode = lighttable::OverlayMode::from_index(d.selected());
                        lighttable::set_overlay_mode(&grid, mode);
                        // Persist the mode just applied (not a re-read of the global).
                        persist::save_ui_pref(
                            &db,
                            OVERLAY_MODE_PREF_KEY,
                            lighttable::overlay_mode_token_for(mode),
                        );
                    }
                });
                bottom.set_center_widget(Some(&overlays));
            }

            let zoom_out = gtk4::Button::builder()
                .icon_name("zoom-out-symbolic")
                .tooltip_text("Larger thumbnails (fewer per row)")
                .build();
            let count = gtk4::Label::new(Some(&THUMB_COLS_DEFAULT.to_string()));
            count.set_width_chars(2);
            let zoom_in = gtk4::Button::builder()
                .icon_name("zoom-in-symbolic")
                .tooltip_text("Smaller thumbnails (more per row)")
                .build();

            // Sync the label + button sensitivity to the grid's current bound, and
            // grey out a button once its end of the range is reached so the control
            // can't run past [THUMB_COLS_MIN, THUMB_COLS_MAX].
            let refresh = {
                let grid = grid.clone();
                let count = count.clone();
                let zoom_out = zoom_out.clone();
                let zoom_in = zoom_in.clone();
                std::rc::Rc::new(move || {
                    let n = grid.max_columns();
                    count.set_label(&n.to_string());
                    zoom_out.set_sensitive(n > THUMB_COLS_MIN);
                    zoom_in.set_sensitive(n < THUMB_COLS_MAX);
                })
            };
            refresh(); // sync initial button sensitivity to the default

            zoom_out.connect_clicked({
                let grid = grid.clone();
                let refresh = refresh.clone();
                move |_| {
                    grid.set_max_columns(grid.max_columns().saturating_sub(1).max(THUMB_COLS_MIN));
                    refresh();
                }
            });
            zoom_in.connect_clicked({
                let grid = grid.clone();
                let refresh = refresh.clone();
                move |_| {
                    grid.set_max_columns((grid.max_columns() + 1).min(THUMB_COLS_MAX));
                    refresh();
                }
            });

            let zoom_box = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
            zoom_box.set_margin_start(6);
            zoom_box.set_margin_end(6);
            zoom_box.append(&zoom_out);
            zoom_box.append(&count);
            zoom_box.append(&zoom_in);
            bottom.set_end_widget(Some(&zoom_box));
        }

        lt_toolbar.add_bottom_bar(&bottom);
    }

    let lt_page = adw::NavigationPage::builder()
        .title("Lighttable")
        .tag("lighttable")
        .child(&lt_toolbar)
        .build();

    // ── Navigation view ────────────────────────────────────────────────────
    let nav = adw::NavigationView::new();
    nav.push(&lt_page);

    // Double-click → darkroom page; F1–F5 → toggle colour label on selection.
    {
        if let Some(grid) = scroll.child().and_downcast::<gtk4::GridView>() {
            grid.connect_activate(clone!(@weak nav, @weak lt_model, @strong db_path => move |_, pos| {
                if let Some(path) = lt_model.item(pos)
                    .and_downcast::<gtk4::StringObject>()
                    .map(|o| o.string().to_string())
                    .filter(|p| p.contains('/'))
                {
                    let page = darkroom::darkroom_page(&path, &db_path);
                    // Tag the page with its image path so the pop handler below can
                    // recover which cell to re-sync (m4-25), regardless of how the
                    // page was dismissed (back button / Escape / swipe gesture).
                    page.set_tag(Some(&path));
                    nav.push(&page);
                }
            }));

            // m4-25/m4-28: when a darkroom page is popped, its colour labels and/or
            // star rating may have been changed in that view; re-query the DB and
            // repaint the returning grid cell's dot + star rows so the lighttable
            // doesn't show stale metadata until it rebinds. Every page pushed past
            // the lighttable root IS a darkroom page, and its tag carries the image
            // path; we guard on the `/` just like the loaders.
            nav.connect_popped(clone!(@weak grid, @strong db_path => move |_, page| {
                if let Some(path) = page.tag().map(|s| s.to_string()).filter(|p| p.contains('/')) {
                    lighttable::refresh_color_dots_for_path(&grid, &db_path, &path);
                    lighttable::refresh_stars_for_path(&grid, &db_path, &path);
                }
            }));

            // Metadata keyboard shortcuts on the selected grid image (darktable):
            // plain F1..F5 toggle colour label 0..4; plain 0..5 set the star rating.
            // Both repaint just that cell's row in place. The controller lives on the
            // grid, so it only fires when the lighttable has focus — not the darkroom
            // page, and not while the user is typing in the search entry. Modifier
            // combos (Ctrl/Alt + key) are left to propagate so they can't be mistaken
            // for a metadata shortcut.
            let key = gtk4::EventControllerKey::new();
            let sel = lt_selection.clone();
            let db  = db_path.clone();
            key.connect_key_pressed(clone!(@weak grid => @default-return glib::Propagation::Proceed, move |_, keyval, _, state| {
                if state.intersects(gtk4::gdk::ModifierType::CONTROL_MASK | gtk4::gdk::ModifierType::ALT_MASK) {
                    return glib::Propagation::Proceed;
                }
                if let Some(color) = lighttable::fkey_to_color(keyval) {
                    lighttable::toggle_selected_color(&grid, &sel, &db, color);
                    return glib::Propagation::Stop;
                }
                if let Some(rating) = lighttable::digit_to_rating(keyval) {
                    lighttable::set_selected_rating(&grid, &sel, &db, rating);
                    return glib::Propagation::Stop;
                }
                glib::Propagation::Proceed
            }));
            grid.add_controller(key);
        }
    }

    // ── View switcher (m4-97d) ─────────────────────────────────────────────
    // darktable-style Lighttable | Darkroom | Other toggle, as the lighttable
    // header's title widget. "Darkroom" opens the currently-selected image in
    // the editor (same push as a double-click); "Other" (map/print/tethering)
    // is disabled — those views aren't ported. The switcher shows only in the
    // lighttable header for now; mirroring it into the darkroom page header is a
    // follow-up (see RUST_MIGRATION_PLAN.md).
    {
        let sw = build_view_switcher();
        let lt_btn = sw.lighttable.clone();
        let dr_btn = sw.darkroom.clone();
        lt_btn.set_active(true); // the lighttable is the root view
        // Switcher alone as the title (no caption): the app is named "Darkroom",
        // but so is darktable's editing *view* — a "Darkroom" branding caption
        // right under the "Darkroom" switcher button reads as a confusing
        // duplicate. The darkroom view keeps a caption because there it's the
        // filename (real per-image context), not a redundant app name.
        sw.container.set_halign(gtk4::Align::Center);
        lt_header.set_title_widget(Some(&sw.container));

        // The buttons only *request* navigation on a real user click. GTK4
        // emits `clicked` for user activation only (programmatic `set_active`
        // emits `toggled`), so the nav-driven state sync below can't echo back
        // into another push. "Darkroom" opens the selected image; with no valid
        // selection (empty view / sentinel row) it snaps back to Lighttable.
        dr_btn.connect_clicked(clone!(
            @weak nav, @weak lt_selection, @weak lt_btn, @strong db_path => move |b| {
            if !b.is_active() { return; }
            if let Some(path) = lighttable::selected_path(&lt_selection) {
                let page = darkroom::darkroom_page(&path, &db_path);
                page.set_tag(Some(&path));
                nav.push(&page);
            } else {
                lt_btn.set_active(true);
            }
        }));

        // `nav` is the single source of truth for the current view; mirror it in
        // the switcher regardless of entry point (switcher / double-click /
        // keyboard push), so the toggle never lies about which view is active.
        // These fire only for pushes AFTER this point — the initial lighttable
        // root was pushed earlier, so startup keeps Lighttable active.
        nav.connect_pushed(clone!(@weak dr_btn => move |_| dr_btn.set_active(true)));
        nav.connect_popped(clone!(@weak lt_btn => move |_, _| lt_btn.set_active(true)));
    }

    // ── Window actions for keyboard shortcuts ──────────────────────────────
    {
        // win.import — Ctrl+I
        let db         = db_path.clone();
        let toast_fn   = make_toast.clone();
        let at         = active_tag.clone();
        let lp         = left.clone();
        let import_act = gtk4::gio::SimpleAction::new("import", None);
        import_act.connect_activate(clone!(@weak window, @weak lt_model => move |_, _| {
            let db_inner    = db.clone();
            let toast_inner = toast_fn.clone();
            let lp_inner    = lp.clone();
            dialogs::show_import_dialog(
                window.upcast_ref::<gtk4::Window>(),
                db.clone(),
                clone!(@weak lt_model, @strong db_inner, @strong at => move || {
                    lp_inner.clear_filter_highlights();   // post-import view shows all images
                    *at.borrow_mut() = None;   // post-import view shows all images
                    lighttable::lighttable_load_from_db(&lt_model, &db_inner);
                }),
                toast_inner,
            );
        }));
        window.add_action(&import_act);

        // win.export-selected — Ctrl+E
        let toast_fn2   = make_toast.clone();
        let export_act  = gtk4::gio::SimpleAction::new("export-selected", None);
        let export_db2  = db_path.clone();
        export_act.connect_activate(clone!(@weak lt_model, @weak lt_selection, @weak window => move |_, _| {
            let pos = lt_selection.selected();
            let paths: Vec<String> = lt_model.item(pos)
                .and_downcast::<gtk4::StringObject>()
                .map(|o| o.string().to_string())
                .filter(|p| p.contains('/'))
                .into_iter().collect();
            dialogs::show_export_dialog(
                window.upcast_ref::<gtk4::Window>(),
                paths,
                None, // no fixed edit — each image's edit is loaded from the catalog
                Some(export_db2.clone()),
                toast_fn2.clone(),
            );
        }));
        window.add_action(&export_act);
    }

    // ── Wire toast overlay + present ───────────────────────────────────────
    toast_overlay.set_child(Some(&nav));
    window.set_child(Some(&toast_overlay));
    // Maximize so the window always fills the display. In the KasmVNC container
    // the framebuffer is resized dynamically to the connecting browser's
    // viewport; without maximizing, the window keeps its fixed default size in
    // the top-left corner surrounded by black, which reads as "no main screen".
    // openbox (and any EWMH WM) honours the maximized hint and re-fits the
    // window when the framebuffer is later resized. The default_width/height
    // above remain the fallback for WMs that don't honour maximize.
    window.maximize();
    window.present();
}
