//! Lighttable view — thumbnail grid for browsing the image collection.
//!
//! Phase 3-ui-1: minimal `GtkGridView` placeholder that compiles and
//! presents inside the main window. Real thumbnail loading and DB
//! connection follow in subsequent patches.

use adw::prelude::*;
use gtk4::{GridView, ListItem, ScrolledWindow, SignalListItemFactory, SingleSelection};

pub const THUMB_SIZE: i32 = 180;

/// Build the lighttable page.
/// Returns an `adw::NavigationPage` with a scrollable `GtkGridView`.
pub fn lighttable_page() -> adw::NavigationPage {
    let model = gtk4::StringList::new(&[]);
    let selection = SingleSelection::new(Some(model));

    let factory = SignalListItemFactory::new();

    // Set-up: create a label card for each slot
    factory.connect_setup(|_, list_item| {
        let item = list_item.downcast_ref::<ListItem>().unwrap();
        let label = gtk4::Label::builder()
            .width_request(THUMB_SIZE)
            .height_request(THUMB_SIZE)
            .build();
        label.add_css_class("card");
        item.set_child(Some(&label));
    });

    // Bind: display the image filename as placeholder text
    factory.connect_bind(|_, list_item| {
        let item = list_item.downcast_ref::<ListItem>().unwrap();
        let label = item.child().and_downcast::<gtk4::Label>().unwrap();
        let string_obj = item.item().and_downcast::<gtk4::StringObject>().unwrap();
        label.set_label(&string_obj.string());
    });

    let grid = GridView::builder()
        .model(&selection)
        .factory(&factory)
        .max_columns(16)
        .min_columns(2)
        .build();

    let scroll = ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .child(&grid)
        .vexpand(true)
        .build();

    adw::NavigationPage::builder()
        .title("Lighttable")
        .child(&scroll)
        .build()
}
