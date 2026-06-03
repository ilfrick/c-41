//! Darkroom editing view — single-image editing with IOP module stack.
//!
//! Phase 3-ui-4: skeleton view showing the image at full viewport scale
//! with a right panel listing the active IOP modules (stubs). Navigation
//! back to the lighttable is via the NavigationView pop action.

use adw::prelude::*;
use glib::clone;
use crate::dialogs;

/// Build a NavigationPage for editing a single image at `file_path`.
///
/// The page title is set to the filename. The caller pushes this page onto
/// an `adw::NavigationView` and pops it to return to the lighttable.
pub fn darkroom_page(file_path: &str) -> adw::NavigationPage {
    let filename = std::path::Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(file_path)
        .to_string();

    // ── Main image area ────────────────────────────────────────────────────
    let picture = gtk4::Picture::builder()
        .hexpand(true)
        .vexpand(true)
        .content_fit(gtk4::ContentFit::Contain)
        .build();

    // Load the image asynchronously so the page appears immediately
    let path_for_load = file_path.to_string();
    glib::spawn_future_local(clone!(@weak picture => async move {
        let bytes = gio::spawn_blocking(move || std::fs::read(&path_for_load).ok())
            .await
            .ok()
            .flatten();
        if let Some(data) = bytes {
            let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
            let _ = loader.write(&data);
            let _ = loader.close();
            if let Some(pb) = loader.pixbuf() {
                picture.set_paintable(Some(&gtk4::gdk::Texture::for_pixbuf(&pb)));
            }
        }
    }));

    // ── IOP module list (right panel stub) ─────────────────────────────────
    let modules_panel = build_modules_panel();

    // ── Split view: image | modules ────────────────────────────────────────
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    content.append(&picture);
    content.append(&gtk4::Separator::new(gtk4::Orientation::Vertical));
    content.append(&modules_panel);

    // ── Header bar with Export button ─────────────────────────────────────
    let header = adw::HeaderBar::new();
    let title_widget = adw::WindowTitle::new(&filename, "Darkroom");
    header.set_title_widget(Some(&title_widget));

    let export_btn = gtk4::Button::builder()
        .label("Export")
        .tooltip_text("Export this image")
        .build();
    export_btn.add_css_class("suggested-action");
    let path_for_export = file_path.to_string();
    export_btn.connect_clicked(move |btn| {
        if let Some(root) = btn.root().and_downcast::<gtk4::Window>() {
            dialogs::show_export_dialog(
                root.upcast_ref::<gtk4::Window>(),
                vec![path_for_export.clone()],
                |msg| eprintln!("{msg}"), // darkroom view has no toast overlay yet
            );
        }
    });
    header.pack_end(&export_btn);

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&content));

    adw::NavigationPage::builder()
        .title(&filename)
        .child(&toolbar_view)
        .build()
}

/// Stub module list panel (to be replaced with real IOP module stack).
fn build_modules_panel() -> gtk4::Box {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .width_request(240)
        .build();

    let header = gtk4::Label::builder()
        .label("Modules")
        .halign(gtk4::Align::Start)
        .margin_top(12).margin_bottom(6)
        .margin_start(12).margin_end(12)
        .build();
    header.add_css_class("heading");
    panel.append(&header);
    panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));

    // Placeholder list of common darktable IOPs
    let modules = [
        ("Exposure",         true),
        ("Color calibration", true),
        ("Filmic RGB",       true),
        ("Color balance RGB", true),
        ("Noise reduction",  false),
        ("Lens correction",  true),
        ("Crop",             false),
    ];

    let list_box = gtk4::ListBox::builder()
        .selection_mode(gtk4::SelectionMode::None)
        .build();
    list_box.add_css_class("boxed-list");
    list_box.set_margin_start(12);
    list_box.set_margin_end(12);
    list_box.set_margin_top(8);

    for (name, enabled) in modules {
        let row = adw::ActionRow::builder()
            .title(name)
            .build();
        let toggle = gtk4::Switch::builder()
            .active(enabled)
            .valign(gtk4::Align::Center)
            .build();
        row.add_suffix(&toggle);
        list_box.append(&row);
    }

    panel.append(&list_box);
    panel
}
