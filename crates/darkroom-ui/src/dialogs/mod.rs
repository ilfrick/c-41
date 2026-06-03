//! Dialogs for Darkroom -- export, preferences, about.
//!
//! Phase 3-ui-5: export dialog that invokes darkroom-cli to export
//! selected images to a chosen output directory.

use adw::prelude::*;
use anyhow::Result;

/// Show the export dialog for a list of image paths.
///
/// Presents format and quality choices, then calls `darkroom-cli` for each
/// selected image. Output goes to an `exports/` sub-folder next to the source.
pub fn show_export_dialog(parent: &gtk4::Window, paths: Vec<String>) {
    if paths.is_empty() {
        return;
    }

    let dialog = adw::AlertDialog::builder()
        .heading("Export Images")
        .body(&format!("Export {} image(s)", paths.len()))
        .build();

    // Controls packed into the dialog body
    let content_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(8)
        .margin_top(8)
        .build();

    let format_row = adw::ComboRow::builder().title("Format").build();
    format_row.set_model(Some(&gtk4::StringList::new(&["JPEG (sRGB)", "TIFF 16-bit", "PNG"])));
    content_box.append(&format_row);

    let quality_row = adw::SpinRow::builder().title("JPEG quality").build();
    quality_row.set_adjustment(Some(&gtk4::Adjustment::new(95.0, 1.0, 100.0, 1.0, 10.0, 0.0)));
    content_box.append(&quality_row);

    dialog.set_extra_child(Some(&content_box));
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("export", "Export");
    dialog.set_response_appearance("export", adw::ResponseAppearance::Suggested);

    dialog.connect_response(Some("export"), move |_, _| {
        let fmt = match format_row.selected() {
            0 => "jpeg",
            1 => "tiff",
            _ => "png",
        };
        let quality  = quality_row.value() as u32;
        let out_paths = paths.clone();
        let fmt_str   = fmt.to_string();

        glib::spawn_future_local(async move {
            if let Err(e) = export_images_async(out_paths, fmt_str, quality).await {
                eprintln!("Export error: {e}");
            }
        });
    });

    dialog.present(Some(parent.upcast_ref::<gtk4::Widget>()));
}

/// Run `darkroom-cli` for each image asynchronously on a thread pool.
async fn export_images_async(paths: Vec<String>, format: String, quality: u32) -> Result<()> {
    gio::spawn_blocking(move || {
        for path in &paths {
            let out_dir = std::path::Path::new(path)
                .parent()
                .map(|p| p.join("exports"))
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));

            let _ = std::fs::create_dir_all(&out_dir);

            let status = std::process::Command::new("darkroom-cli")
                .args([
                    path.as_str(),
                    out_dir.to_str().unwrap_or("/tmp"),
                    "--style", "none",
                    "--out-ext", &format,
                    "--quality", &quality.to_string(),
                ])
                .status();

            match status {
                Ok(s) if !s.success() =>
                    eprintln!("darkroom-cli exit {s} for {path}"),
                Err(e) =>
                    eprintln!("darkroom-cli not found for {path}: {e}"),
                _ => {}
            }
        }
    }).await.map_err(|e| anyhow::anyhow!("thread panicked: {e:?}"))?;
    Ok(())
}
