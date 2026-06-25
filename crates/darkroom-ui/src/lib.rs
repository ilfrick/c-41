//! GTK4 + libadwaita UI shell for Darkroom.
//!
//! Phase 3-ui-12: live exposure preview in the darkroom view via darkroom-core.

use adw::prelude::*;
use adw::Application;
use anyhow::Result;
use gtk4::ApplicationWindow;
use glib::clone;

pub mod catalog;
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
    Ok(app.run())
}

fn build_main_window(app: &Application) {
    let window = ApplicationWindow::builder()
        .application(app)
        .title("Darkroom")
        .default_width(DEFAULT_WIDTH)
        .default_height(DEFAULT_HEIGHT)
        .build();

    let db_path = std::env::var("DARKROOM_LIBRARY_DB").unwrap_or_default();

    // ── Toast overlay (wraps everything for in-app notifications) ──────────
    let toast_overlay = adw::ToastOverlay::new();
    let make_toast = {
        let to = toast_overlay.clone();
        move |msg: String| {
            to.add_toast(adw::Toast::new(&msg));
        }
    };

    // ── Lighttable ─────────────────────────────────────────────────────────
    let (lt_grid, lt_model, lt_selection) =
        lighttable::lighttable_page(db_path.clone());
    lighttable::lighttable_load_from_db(&lt_model, &db_path);

    // ── Panels ─────────────────────────────────────────────────────────────
    // Shared "currently-filtering tag id" (None = no tag filter). Set by the
    // left-panel folder/tag clicks and the search/import paths below; read by
    // the tag-mutation callbacks to decide whether to re-run the grid filter.
    let active_tag: std::rc::Rc<std::cell::Cell<Option<u32>>> =
        std::rc::Rc::new(std::cell::Cell::new(None));

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
            if let Some(id) = at.get() {
                let prev = lighttable::selected_path(&sel);
                lighttable::lighttable_load_by_tag(&mdl, &db, id);
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
    let scroll = lt_grid.child().unwrap();
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
        search.connect_search_changed(clone!(@weak lt_model => move |s| {
            let query = s.text().to_string();
            at.set(None);   // a name search supersedes any tag filter
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
        btn.connect_clicked(clone!(@weak window, @weak lt_model => move |_| {
            let db_inner    = db.clone();
            let toast_inner = toast_fn.clone();
            dialogs::show_import_dialog(
                window.upcast_ref::<gtk4::Window>(),
                db.clone(),
                clone!(@weak lt_model, @strong db_inner, @strong at => move || {
                    at.set(None);   // post-import view shows all images
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

    // Double-click → darkroom page
    {
        let scroll_ref = scroll.downcast_ref::<gtk4::ScrolledWindow>().unwrap();
        if let Some(grid) = scroll_ref.child().and_downcast::<gtk4::GridView>() {
            grid.connect_activate(clone!(@weak nav, @weak lt_model, @strong db_path => move |_, pos| {
                if let Some(path) = lt_model.item(pos)
                    .and_downcast::<gtk4::StringObject>()
                    .map(|o| o.string().to_string())
                    .filter(|p| p.contains('/'))
                {
                    nav.push(&darkroom::darkroom_page(&path, &db_path));
                }
            }));
        }
    }

    // ── Window actions for keyboard shortcuts ──────────────────────────────
    {
        // win.import — Ctrl+I
        let db         = db_path.clone();
        let toast_fn   = make_toast.clone();
        let at         = active_tag.clone();
        let import_act = gtk4::gio::SimpleAction::new("import", None);
        import_act.connect_activate(clone!(@weak window, @weak lt_model => move |_, _| {
            let db_inner    = db.clone();
            let toast_inner = toast_fn.clone();
            dialogs::show_import_dialog(
                window.upcast_ref::<gtk4::Window>(),
                db.clone(),
                clone!(@weak lt_model, @strong db_inner, @strong at => move || {
                    at.set(None);   // post-import view shows all images
                    lighttable::lighttable_load_from_db(&lt_model, &db_inner);
                }),
                toast_inner,
            );
        }));
        window.add_action(&import_act);

        // win.export-selected — Ctrl+E
        let toast_fn2   = make_toast.clone();
        let export_act  = gtk4::gio::SimpleAction::new("export-selected", None);
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
                toast_fn2.clone(),
            );
        }));
        window.add_action(&export_act);
    }

    // ── Wire toast overlay + present ───────────────────────────────────────
    toast_overlay.set_child(Some(&nav));
    window.set_child(Some(&toast_overlay));
    window.present();
}
