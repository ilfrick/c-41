//! The persistent export panel (Phase 3 m4-7b): a reusable libadwaita widget that
//! collects the export configuration — format, JPEG quality, resize box, and the
//! output-path template — on top of the pure, headless-tested [`crate::export`]
//! model. The widget is host-agnostic (currently mounted in the export dialog; it
//! can be docked as a side panel later) and Clone (all fields are GObject
//! ref-counts). All the testable logic (argv build, resize math, path templating)
//! lives in [`crate::export`]; this module is just the GTK control surface.

use adw::prelude::*;
use crate::export::{ExportFormat, ExportSettings, Resize, DEFAULT_OUTPUT_TEMPLATE};

/// Export configuration controls, grouped for embedding in a dialog or side dock.
#[derive(Clone)]
pub struct ExportPanel {
    /// The root widget to embed.
    pub widget:    gtk4::Box,
    format_row:    adw::ComboRow,
    quality_row:   adw::SpinRow,
    resize_row:    adw::SwitchRow,
    width_row:     adw::SpinRow,
    height_row:    adw::SpinRow,
    upscale_row:   adw::SwitchRow,
    template_row:  adw::EntryRow,
}

impl ExportPanel {
    pub fn new() -> Self {
        let widget = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Vertical)
            .spacing(12)
            .build();

        // ── Format + quality ──────────────────────────────────────────────
        let fmt_group = adw::PreferencesGroup::builder().title("Format").build();

        let format_row = adw::ComboRow::builder().title("File format").build();
        let format_labels: Vec<&str> =
            ExportFormat::ALL.iter().map(|f| f.label()).collect();
        format_row.set_model(Some(&gtk4::StringList::new(&format_labels)));
        fmt_group.add(&format_row);

        let quality_row = adw::SpinRow::builder().title("JPEG quality").build();
        quality_row.set_adjustment(Some(&gtk4::Adjustment::new(95.0, 1.0, 100.0, 1.0, 10.0, 0.0)));
        fmt_group.add(&quality_row);
        widget.append(&fmt_group);

        // Quality only applies to formats that honour it; grey it out otherwise.
        {
            let qr = quality_row.downgrade();
            let sync = move |row: &adw::ComboRow| {
                if let Some(qr) = qr.upgrade() {
                    qr.set_sensitive(ExportFormat::from_index(row.selected()).uses_quality());
                }
            };
            sync(&format_row); // initial state
            format_row.connect_selected_notify(sync);
        }

        // ── Resize ────────────────────────────────────────────────────────
        let resize_group = adw::PreferencesGroup::builder().title("Resize").build();

        let resize_row = adw::SwitchRow::builder()
            .title("Limit size")
            .subtitle("Fit within a bounding box, preserving aspect")
            .build();
        resize_group.add(&resize_row);

        let width_row = adw::SpinRow::builder().title("Max width (px)").build();
        width_row.set_adjustment(Some(&gtk4::Adjustment::new(2048.0, 0.0, 65535.0, 1.0, 100.0, 0.0)));
        resize_group.add(&width_row);

        let height_row = adw::SpinRow::builder().title("Max height (px)").build();
        height_row.set_adjustment(Some(&gtk4::Adjustment::new(2048.0, 0.0, 65535.0, 1.0, 100.0, 0.0)));
        resize_group.add(&height_row);

        let upscale_row = adw::SwitchRow::builder()
            .title("Allow upscaling")
            .subtitle("Enlarge images smaller than the box")
            .build();
        resize_group.add(&upscale_row);
        widget.append(&resize_group);

        // Resize sub-controls are only meaningful when the limit is on.
        {
            let wr = width_row.downgrade();
            let hr = height_row.downgrade();
            let ur = upscale_row.downgrade();
            let sync = move |row: &adw::SwitchRow| {
                let on = row.is_active();
                if let Some(w) = wr.upgrade() { w.set_sensitive(on); }
                if let Some(h) = hr.upgrade() { h.set_sensitive(on); }
                if let Some(u) = ur.upgrade() { u.set_sensitive(on); }
            };
            sync(&resize_row); // initial state (off → greyed)
            resize_row.connect_active_notify(sync);
        }

        // ── Output path template ──────────────────────────────────────────
        let out_group = adw::PreferencesGroup::builder()
            .title("Output")
            .description("Destination path; the extension is added from the format. \
                          Supports $(FILE_FOLDER), $(FILE_NAME), $(SEQUENCE).")
            .build();
        let template_row = adw::EntryRow::builder().title("Path template").build();
        template_row.set_text(DEFAULT_OUTPUT_TEMPLATE);
        out_group.add(&template_row);
        widget.append(&out_group);

        Self {
            widget, format_row, quality_row,
            resize_row, width_row, height_row, upscale_row,
            template_row,
        }
    }

    /// Snapshot the current control values into an [`ExportSettings`].
    pub fn settings(&self) -> ExportSettings {
        let resize = if self.resize_row.is_active() {
            Some(Resize {
                max_w: self.width_row.value() as u32,
                max_h: self.height_row.value() as u32,
                allow_upscale: self.upscale_row.is_active(),
            })
        } else {
            None
        };
        ExportSettings {
            format: ExportFormat::from_index(self.format_row.selected()),
            quality: self.quality_row.value() as u32,
            resize,
        }
    }

    /// The output-path template, falling back to the default when left blank.
    pub fn template(&self) -> String {
        let t = self.template_row.text().to_string();
        if t.trim().is_empty() { DEFAULT_OUTPUT_TEMPLATE.to_string() } else { t }
    }
}

impl Default for ExportPanel {
    fn default() -> Self { Self::new() }
}
