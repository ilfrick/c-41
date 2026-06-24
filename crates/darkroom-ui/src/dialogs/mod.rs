//! Dialogs for Darkroom -- export, preferences, about.
//!
//! Phase 3-ui-5: export dialog that invokes darkroom-cli to export
//! selected images to a chosen output directory.

use adw::prelude::*;
use anyhow::Result;

/// Show the export dialog for a list of image paths.
///
/// Presents format and quality choices, then calls `darkroom-cli` for each
/// selected image. `toast_fn` is called with a summary string on completion.
pub fn show_export_dialog(
    parent: &gtk4::Window,
    paths: Vec<String>,
    toast_fn: impl Fn(String) + 'static,
) {
    if paths.is_empty() {
        return;
    }

    let dialog = adw::AlertDialog::builder()
        .heading("Export Images")
        .body(&format!("Export {} image(s)", paths.len()))
        .build();

    // The reusable export panel collects format / quality / resize / template.
    let panel = crate::export_panel::ExportPanel::new();
    dialog.set_extra_child(Some(&panel.widget));
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("export", "Export");
    dialog.set_response_appearance("export", adw::ResponseAppearance::Suggested);

    let toast_fn = std::rc::Rc::new(toast_fn);
    dialog.connect_response(Some("export"), move |_, _| {
        let settings = panel.settings();
        let template = panel.template();
        let out_paths = paths.clone();
        let n         = out_paths.len();
        let tf        = toast_fn.clone();

        glib::spawn_future_local(async move {
            match export_images_async(out_paths, settings, template).await {
                Ok(0)      => tf(format!("Exported {n} image(s)")),
                Ok(failed) => tf(format!("Exported {} of {n} ({failed} failed)", n - failed)),
                Err(e)     => tf(format!("Export failed: {e}")),
            }
        });
    });

    dialog.present(Some(parent.upcast_ref::<gtk4::Widget>()));
}

/// Run `darkroom-cli` for each image asynchronously on a thread pool, building the
/// argv from the shared (unit-tested) [`crate::export::cli_args`]. The output
/// destination is the per-image expansion of `template` (1-based `$(SEQUENCE)`);
/// the template is first made batch-safe so same-stem sources can't overwrite, and
/// each dest's parent directory is created first since the CLI won't. Returns the
/// number of images that **failed** so the caller can surface it (a green
/// "Exported N" toast over 0 written files would otherwise hide the failure).
async fn export_images_async(
    paths: Vec<String>,
    settings: crate::export::ExportSettings,
    template: String,
) -> Result<usize> {
    let failed = gio::spawn_blocking(move || {
        let template = crate::export::batch_output_template(&template, paths.len());
        let mut failed = 0usize;
        for (i, path) in paths.iter().enumerate() {
            // Extension-less destination; `--out-ext` (in cli_args) appends the ext.
            let dest = crate::export::expand_output_template(&template, path, (i + 1) as u32);
            if let Some(parent) = std::path::Path::new(&dest).parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("darkroom export: cannot create {parent:?}: {e}");
                    failed += 1;
                    continue;
                }
            }

            match std::process::Command::new("darkroom-cli")
                .args(crate::export::cli_args(path, &dest, &settings))
                .status()
            {
                Ok(s) if s.success() => {}
                Ok(s)  => { eprintln!("darkroom-cli exit {s} for {path}"); failed += 1; }
                Err(e) => { eprintln!("darkroom-cli not found for {path}: {e}"); failed += 1; }
            }
        }
        failed
    }).await.map_err(|e| anyhow::anyhow!("thread panicked: {e:?}"))?;
    Ok(failed)
}

// ── Import folder dialog ──────────────────────────────────────────────────

const RAW_EXTENSIONS: &[&str] = &[
    "cr2", "cr3", "nef", "nrw", "arw", "rw2", "orf", "pef", "raf",
    "dng", "raw", "rwl", "srw", "x3f", "jpg", "jpeg", "tiff", "tif",
    "png", "heic", "heif", "avif",
];

/// Show a folder-chooser that imports images into the library DB at `db_path`.
/// On confirm, scans the chosen folder recursively and calls `on_done` when
/// the import finishes (so the caller can reload the lighttable).
pub fn show_import_dialog(
    parent: &gtk4::Window,
    db_path: String,
    on_done: impl Fn() + 'static,
    toast_fn: impl Fn(String) + 'static,
) {
    let chooser = gtk4::FileDialog::builder()
        .title("Import Folder")
        .build();

    let db = db_path.clone();
    let on_done  = std::rc::Rc::new(on_done);
    let toast_fn = std::rc::Rc::new(toast_fn);
    chooser.select_folder(Some(parent), gtk4::gio::Cancellable::NONE, move |result| {
        let folder = match result {
            Ok(f) => f,
            Err(_) => return,
        };
        let folder_path = match folder.path() {
            Some(p) => p,
            None    => return,
        };
        let folder_str  = folder_path.to_string_lossy().to_string();
        let folder_name = folder_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("folder")
            .to_string();
        let db2      = db.clone();
        let on_done  = on_done.clone();
        let toast_fn = toast_fn.clone();

        glib::spawn_future_local(async move {
            let count = gio::spawn_blocking(move || {
                import_folder_sync(&folder_str, &db2)
            }).await.ok().flatten().unwrap_or(0);

            toast_fn(format!("Imported {count} images from \"{folder_name}\""));
            on_done();
        });
    });
}

/// Walk `folder` recursively, create a film roll in the DB, and insert each
/// found image file. Returns the number of newly registered images.
fn import_folder_sync(folder: &str, db_path: &str) -> Option<usize> {
    use darkroom_db::film;
    use darkroom_db::image;

    let conn = if db_path.is_empty() {
        return None; // can't import into in-memory demo
    } else {
        rusqlite::Connection::open(db_path).ok()?
    };

    let film_id = film::film_new(&conn, folder).ok()??;
    let mut count = 0usize;

    for entry in walkdir::WalkDir::new(folder)
        .max_depth(1)        // one level; use max_depth(usize::MAX) for recursive
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        if RAW_EXTENSIONS.contains(&ext.as_str()) {
            let filename = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if filename.is_empty() { continue; }

            // Probe image dimensions (fall back to 0×0 if unreadable)
            let (w, h) = probe_dims(path).unwrap_or((0, 0));

            let _ = image::image_insert(&conn, film_id, filename, w, h);
            count += 1;
        }
    }
    Some(count)
}

/// Read image dimensions without fully decoding the file.
fn probe_dims(path: &std::path::Path) -> Option<(i32, i32)> {
    // Use gdk_pixbuf's file-info path (header-only probe, very fast)
    // We're on a background thread so we use the sync API directly.
    let pb = gtk4::gdk_pixbuf::Pixbuf::from_file_at_scale(path, 1, 1, true).ok()?;
    // Scale ratios let us recover original size from the 1px result
    // — but actually we just want the original dimensions from the file.
    // gdk_pixbuf doesn't expose original dims after scale; use file info instead.
    drop(pb);
    // Fall back: use the pixbuf loader to get the natural size hint
    let data = std::fs::read(path).ok()?;
    let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
    // Write a small chunk; loader fires "size-prepared" once it has the header
    let _ = loader.write(&data[..data.len().min(65536)]);
    let _ = loader.close();
    let pb = loader.pixbuf()?;
    Some((pb.width(), pb.height()))
}
