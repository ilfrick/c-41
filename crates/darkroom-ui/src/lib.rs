//! GTK4 + libadwaita UI shell for Darkroom.
//!
//! Phase 3-ui-4: adw::NavigationView with lighttable root page; double-
//! clicking an image pushes a darkroom editing page onto the stack.

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

/// Boot the GTK4 application. Blocks until the main window is closed.
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

    // ── Lighttable page ────────────────────────────────────────────────────
    let (lt_grid, lt_model) = lighttable::lighttable_page();
    lighttable::lighttable_load_from_db(&lt_model, &db_path);

    // Three-column layout inside the lighttable NavigationPage
    let scroll = lt_grid.child().unwrap();
    scroll.set_hexpand(true);

    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    hbox.append(&panels::left_panel(&db_path));
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    hbox.append(&scroll);
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    hbox.append(&panels::right_panel());

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

    // ── Navigation view (stack) ────────────────────────────────────────────
    let nav = adw::NavigationView::new();
    nav.push(&lt_page);

    // ── Wire up double-click → push darkroom page ──────────────────────────
    // The GridView emits "activate" when an item is double-clicked or Enter
    // is pressed. We need a reference to the GridView to connect signals.
    // Re-extract it: scroll → viewport → GridView.
    {
        let scroll_ref = scroll.downcast_ref::<gtk4::ScrolledWindow>().unwrap();
        if let Some(grid) = scroll_ref.child().and_downcast::<gtk4::GridView>() {
            grid.connect_activate(clone!(@weak nav, @weak lt_model => move |_, pos| {
                if let Some(path) = lt_model.item(pos)
                    .and_downcast::<gtk4::StringObject>()
                    .map(|o| o.string().to_string())
                {
                    if path.contains('/') {
                        let page = darkroom::darkroom_page(&path);
                        nav.push(&page);
                    }
                }
            }));
        }
    }

    window.set_child(Some(&nav));
    window.present();
}
