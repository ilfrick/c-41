//! GTK4 + libadwaita UI shell for Darkroom.
//!
//! Phase 3-ui-2: three-column layout with a real GtkGridView lighttable
//! populated from darkroom-db (demo in-memory data until a real DB path
//! is passed through).

use adw::prelude::*;
use adw::Application;
use anyhow::Result;
use gtk4::ApplicationWindow;

pub mod dialogs;
pub mod lighttable;
pub mod panels;

pub const APP_ID:         &str = "org.darkroom.Darkroom";
pub const DEFAULT_WIDTH:   i32 = 1280;
pub const DEFAULT_HEIGHT:  i32 = 800;

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

    // ── Build the lighttable and load demo / real DB data ──────────────────
    let (lt_page, lt_model) = lighttable::lighttable_page();

    // Populate from the library DB (empty string → in-memory demo data)
    let db_path = std::env::var("DARKROOM_LIBRARY_DB").unwrap_or_default();
    lighttable::lighttable_load_from_db(&lt_model, &db_path);

    // Extract the ScrolledWindow from the NavigationPage so it fills the HBox
    let scroll = lt_page.child().unwrap();
    scroll.set_hexpand(true);

    // ── Three-column layout ────────────────────────────────────────────────
    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    hbox.append(&panels::left_panel());
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    hbox.append(&scroll);
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    hbox.append(&panels::right_panel());

    // ── Header bar ─────────────────────────────────────────────────────────
    let header = adw::HeaderBar::new();
    let title  = adw::WindowTitle::new("Darkroom", "Lighttable");
    header.set_title_widget(Some(&title));

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&hbox));

    window.set_child(Some(&toolbar_view));
    window.present();
}
