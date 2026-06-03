//! GTK4 + libadwaita UI shell for Darkroom.
//!
//! Phase 3-ui-6: collection filtering (left panel) + live metadata
//! inspector (right panel) connected to SingleSelection changes.

use adw::prelude::*;
use adw::Application;
use anyhow::Result;
use gtk4::ApplicationWindow;
use glib::clone;

pub mod darkroom;
pub mod dialogs;
pub mod lighttable;
pub mod panels;

pub const APP_ID:        &str = "org.darkroom.Darkroom";
pub const DEFAULT_WIDTH:  i32 = 1280;
pub const DEFAULT_HEIGHT: i32 = 800;

pub fn run() -> Result<glib::ExitCode> {
    let app = Application::builder()
        .application_id(APP_ID)
        .build();
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

    // ── Lighttable: grid + model + selection ───────────────────────────────
    let (lt_grid, lt_model, lt_selection) = lighttable::lighttable_page();
    lighttable::lighttable_load_from_db(&lt_model, &db_path);

    // ── Panels ─────────────────────────────────────────────────────────────
    let left  = panels::left_panel(&db_path, &lt_model);
    let right = panels::MetadataPanel::new();

    // Update metadata when selection changes
    {
        let meta_panel = right.clone();
        let db = db_path.clone();
        let model = lt_model.clone();
        lt_selection.connect_selection_changed(move |sel, _, _| {
            let pos = sel.selected();
            if let Some(path) = model.item(pos)
                .and_downcast::<gtk4::StringObject>()
                .map(|o| o.string().to_string())
            {
                if path.contains('/') {
                    meta_panel.update(&path, &db);
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
    hbox.append(&left);
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    hbox.append(&scroll);
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    hbox.append(&right.widget);

    let lt_header = adw::HeaderBar::new();
    lt_header.set_title_widget(Some(&adw::WindowTitle::new("Darkroom", "Lighttable")));

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

    // Double-click → push darkroom page
    {
        let scroll_ref = scroll.downcast_ref::<gtk4::ScrolledWindow>().unwrap();
        if let Some(grid) = scroll_ref.child().and_downcast::<gtk4::GridView>() {
            grid.connect_activate(clone!(@weak nav, @weak lt_model => move |_, pos| {
                if let Some(path) = lt_model.item(pos)
                    .and_downcast::<gtk4::StringObject>()
                    .map(|o| o.string().to_string())
                {
                    if path.contains('/') {
                        nav.push(&darkroom::darkroom_page(&path));
                    }
                }
            }));
        }
    }

    window.set_child(Some(&nav));
    window.present();
}
