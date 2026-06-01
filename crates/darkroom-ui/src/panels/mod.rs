//! Side-panel stubs — collections, history, tagging, export.
//! Phase 3-ui-1: placeholders; real implementations follow.

use adw::prelude::*;

pub fn left_panel() -> gtk4::Box {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(6)
        .margin_top(12).margin_bottom(12)
        .margin_start(12).margin_end(12)
        .width_request(200)
        .build();
    let label = gtk4::Label::new(Some("Collections\n(coming soon)"));
    label.set_halign(gtk4::Align::Center);
    panel.append(&label);
    panel
}

pub fn right_panel() -> gtk4::Box {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(6)
        .margin_top(12).margin_bottom(12)
        .margin_start(12).margin_end(12)
        .width_request(200)
        .build();
    let label = gtk4::Label::new(Some("Metadata\n(coming soon)"));
    label.set_halign(gtk4::Align::Center);
    panel.append(&label);
    panel
}
