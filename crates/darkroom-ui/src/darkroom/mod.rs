//! Darkroom editing view — single-image editing with IOP module stack.
//!
//! Phase 3-ui-12: darkroom view with a LIVE exposure (EV) slider that runs the
//! migrated darkroom-core exposure IOP on the preview; image at full viewport scale
//! with a right panel listing the IOP modules grouped by darktable's module
//! groups (from `crate::catalog`). Navigation back to the lighttable is via
//! the NavigationView pop action.

use adw::prelude::*;
use glib::clone;
use std::cell::RefCell;
use std::rc::Rc;
use crate::dialogs;

/// Decoded preview image kept for live re-processing.
#[derive(Clone)]
struct BaseImage {
    bytes: Vec<u8>,
    width: i32,
    height: i32,
    rowstride: usize,
    nch: usize,
}

/// Paint `picture` with the base preview processed at exposure `ev`, running
/// the migrated `darkroom-core` exposure IOP and uploading a gdk::MemoryTexture.
fn render_preview(picture: &gtk4::Picture, base: &Rc<RefCell<Option<BaseImage>>>, ev: f32) {
    if let Some(b) = base.borrow().as_ref() {
        let processed = crate::preview::apply_exposure(
            &b.bytes, b.width as usize, b.height as usize, b.rowstride, b.nch, ev,
        );
        let fmt = if b.nch == 4 {
            gtk4::gdk::MemoryFormat::R8g8b8a8
        } else {
            gtk4::gdk::MemoryFormat::R8g8b8
        };
        let gbytes = glib::Bytes::from_owned(processed);
        let tex = gtk4::gdk::MemoryTexture::new(b.width, b.height, fmt, &gbytes, b.rowstride);
        picture.set_paintable(Some(&tex));
    }
}

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

    // Shared decoded preview, kept for live re-processing by the EV slider.
    let base: Rc<RefCell<Option<BaseImage>>> = Rc::new(RefCell::new(None));

    // Load + decode the image asynchronously so the page appears immediately.
    let path_for_load = file_path.to_string();
    glib::spawn_future_local(clone!(@weak picture, @strong base => async move {
        let data = gio::spawn_blocking(move || std::fs::read(&path_for_load).ok())
            .await
            .ok()
            .flatten();
        if let Some(data) = data {
            let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
            let _ = loader.write(&data);
            let _ = loader.close();
            if let Some(pb) = loader.pixbuf() {
                *base.borrow_mut() = Some(BaseImage {
                    bytes: pb.read_pixel_bytes().to_vec(),
                    width: pb.width(),
                    height: pb.height(),
                    rowstride: pb.rowstride() as usize,
                    nch: pb.n_channels() as usize,
                });
                render_preview(&picture, &base, 0.0);
            }
        }
    }));

    // ── Exposure control (live, via darkroom-core) ─────────────────────────
    let ev_scale = gtk4::Scale::with_range(gtk4::Orientation::Horizontal, -3.0, 3.0, 0.01);
    ev_scale.set_value(0.0);
    ev_scale.set_hexpand(true);
    ev_scale.set_draw_value(true);
    ev_scale.set_value_pos(gtk4::PositionType::Right);
    ev_scale.connect_value_changed(clone!(@weak picture, @strong base => move |s| {
        render_preview(&picture, &base, s.value() as f32);
    }));
    let ev_bar = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .margin_start(8).margin_end(8).margin_top(4).margin_bottom(4)
        .build();
    ev_bar.append(&gtk4::Label::new(Some("Exposure (EV)")));
    ev_bar.append(&ev_scale);

    let image_area = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .hexpand(true)
        .build();
    image_area.append(&picture);
    image_area.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    image_area.append(&ev_bar);

    // ── IOP module list (right panel) ──────────────────────────────────────
    let modules_panel = build_modules_panel();

    // ── Split view: image | modules ────────────────────────────────────────
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .build();
    content.append(&image_area);
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

/// Module-stack panel: the darktable module groups (base/tone/color/correct/
/// effect) from [`crate::catalog`], each an enable-toggle row. The toggles are
/// not yet wired to the history stack (a later milestone).
fn build_modules_panel() -> gtk4::Widget {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .build();

    let header = gtk4::Label::builder()
        .label("Modules")
        .halign(gtk4::Align::Start)
        .build();
    header.add_css_class("title-4");
    panel.append(&header);

    for group in crate::catalog::module_catalog() {
        let pg = adw::PreferencesGroup::builder().title(group.name).build();
        for mi in group.modules {
            let row = adw::ActionRow::builder().title(mi.label).build();
            let toggle = gtk4::Switch::builder()
                .active(mi.default_on)
                .valign(gtk4::Align::Center)
                .build();
            row.add_suffix(&toggle);
            row.set_activatable_widget(Some(&toggle));
            pg.add(&row);
        }
        panel.append(&pg);
    }

    // Scrollable so the (long) module list never blows out the window height.
    gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vexpand(true)
        .width_request(280)
        .child(&panel)
        .build()
        .upcast()
}
