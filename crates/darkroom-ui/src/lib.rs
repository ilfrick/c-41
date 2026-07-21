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

    let lt_toolbar = adw::ToolbarView::new();
    lt_toolbar.add_top_bar(&lt_header);
    lt_toolbar.set_content(Some(&hbox));

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
