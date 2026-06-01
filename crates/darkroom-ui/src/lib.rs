//! GTK4 + libadwaita UI shell for Darkroom.
//!
//! Phase 3-ui-1: boots the application and presents a three-column layout
//! (left panel | lighttable GridView | right panel). The lighttable uses
//! a GtkGridView with a StringList model as a placeholder until the DB
//! thumbnail pipeline is connected.

use adw::prelude::*;
use adw::Application;
use anyhow::Result;
use gtk4::ApplicationWindow;

pub mod dialogs;
pub mod lighttable;
pub mod panels;

pub const APP_ID: &str = "org.darkroom.Darkroom";
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

    // ── Three-column layout: left panel | lighttable | right panel ──────────
    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();

    // Left panel (collections / film rolls)
    let left = panels::left_panel();
    hbox.append(&left);

    // Separator
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));

    // Lighttable — NavigationPage wraps the GridView
    let lt_page = lighttable::lighttable_page();
    // Unwrap the inner child (ScrolledWindow) from the NavigationPage
    // and put it directly in the HBox so it expands to fill.
    let scroll = lt_page.child().unwrap();
    scroll.set_hexpand(true);
    hbox.append(&scroll);

    // Separator
    hbox.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));

    // Right panel (metadata / history)
    let right = panels::right_panel();
    hbox.append(&right);

    // ── Header bar with view switcher ────────────────────────────────────────
    let header = adw::HeaderBar::new();
    let title = adw::WindowTitle::new("Darkroom", "Lighttable");
    header.set_title_widget(Some(&title));

    // ── Toolbar view: header + content ───────────────────────────────────────
    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&hbox));

    window.set_child(Some(&toolbar_view));
    window.present();
}
