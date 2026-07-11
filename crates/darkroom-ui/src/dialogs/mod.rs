//! Dialogs for Darkroom -- export, preferences, about.
//!
//! Phase 3-ui-5: export dialog that invokes darkroom-cli to export
//! selected images to a chosen output directory.

use adw::prelude::*;
use anyhow::Result;

/// Build the per-image [`ExportEdit`] for a lighttable batch export by loading
/// the image's persisted darkroom-ui state. Returns `Some` **only** for a raw
/// the user has actually edited — persisted colour params, a non-identity
/// crop/straighten, or a non-default demosaic method. An unedited raw (and every
/// non-raw) returns `None` so it falls back to `darktable-cli`, whose full
/// default module stack develops a better baseline than our subset pipeline.
///
/// The seeded params match the darkroom preview exactly (via
/// [`crate::darkroom::initial_params`]): saved params if present, else the
/// raw default (sigmoid on) — so exporting a raw the user only cropped still
/// tone-maps the way the preview does.
///
/// Note the "export == preview" invariant holds only for **edited** raws: an
/// unedited raw returns `None` and is developed by `darktable-cli`'s fuller
/// default stack (highlight-recon, base curve, …) which our subset pipeline
/// doesn't yet match — a better baseline than a WYSIWYG-but-weaker render, until
/// the pipeline reaches parity.
fn load_export_edit(db_path: &str, path: &str) -> Option<crate::export::ExportEdit> {
    if db_path.is_empty() || !crate::raw_preview::is_raw_path(path) {
        return None;
    }
    // One connection, one path→imgid resolution for all three pieces.
    let (saved, geometry, method) = crate::persist::load_edit_state(db_path, path);

    let edited = saved.is_some()
        || !geometry.is_identity()
        || method != darkroom_core::rawimage::DemosaicMethod::default();
    if !edited {
        return None;
    }

    let params = crate::darkroom::initial_params(saved, true);
    Some(crate::export::ExportEdit { method, geometry, params })
}

/// Show the export dialog for a list of image paths.
///
/// Presents format and quality choices, then renders each image — a raw with a
/// darkroom-ui edit through our pipeline, everything else via `darkroom-cli`.
/// `toast_fn` is called with a summary string on completion.
pub fn show_export_dialog(
    parent: &gtk4::Window,
    paths: Vec<String>,
    edit: Option<crate::export::ExportEdit>,
    db_path: Option<String>,
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
        let db        = db_path.clone();

        glib::spawn_future_local(async move {
            match export_images_async(out_paths, settings, template, edit, db).await {
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
    edit: Option<crate::export::ExportEdit>,
    db_path: Option<String>,
) -> Result<usize> {
    let failed = gio::spawn_blocking(move || {
        let template = crate::export::batch_output_template(&template, paths.len());
        let mut failed = 0usize;
        for (i, path) in paths.iter().enumerate() {
            // Extension-less destination; the CLI's `--out-ext` (or the Rust
            // encoder) appends the extension.
            let dest = crate::export::expand_output_template(&template, path, (i + 1) as u32);
            if let Some(parent) = std::path::Path::new(&dest).parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("darkroom export: cannot create {parent:?}: {e}");
                    failed += 1;
                    continue;
                }
            }

            // Rust-native render for a raw WITH a darkroom-ui edit (so the export
            // matches the preview + applies geometry/params); everything else
            // (JPEGs, or an unedited raw) via darktable-cli, which develops with
            // darktable's own default history. A fixed `edit` (single-image
            // darkroom export) applies to its one path; otherwise each image's
            // edit is loaded from the catalog (lighttable multi-export).
            let img_edit = edit.or_else(|| {
                db_path.as_deref().and_then(|db| load_export_edit(db, path))
            });
            if let Some(edit) = img_edit {
                if crate::raw_preview::is_raw_path(path) {
                    if let Err(e) = render_raw_export(path, &dest, &settings, edit) {
                        eprintln!("darkroom export: Rust render failed for {path}: {e}");
                        failed += 1;
                    }
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

/// Export one raw through the Rust pipeline (matching the darkroom preview):
/// decode → render (demosaic + geometry + colour params) → optional resize →
/// encode via the pure-Rust `image` crate to `<dest>.<ext>`. **PNG/TIFF are
/// 16-bit** (`render_export_rgb16`), JPEG is 8-bit (inherently). Runs on the
/// export thread pool; all types here are plain Rust (no GTK), so no main-thread
/// constraint. `w`/`h` come from `render_export_*` and always match the buffer.
fn render_raw_export(
    path: &str,
    dest: &str,
    settings: &crate::export::ExportSettings,
    edit: crate::export::ExportEdit,
) -> Result<()> {
    use crate::export::ExportFormat;
    use image::{imageops::FilterType, ImageBuffer, ImageEncoder, Rgb};

    let img = darkroom_core::rawimage::load(path).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    let dest_ext = format!("{dest}.{}", settings.format.out_ext());

    // Target size for the optional resize box (None ⇒ keep source size).
    // `fit_within` already guarantees each dimension ≥ 1. The resize below runs
    // in sRGB-encoded (gamma) space — technically inexact, but matches gdk-pixbuf
    // and virtually every other tool; acceptable for export.
    let target = |w: usize, h: usize| -> Option<(u32, u32)> {
        settings.resize.as_ref().map(|r| crate::export::fit_within(w as u32, h as u32, r))
    };

    match settings.format {
        ExportFormat::Jpeg => {
            let (w, h, rgb) =
                crate::export::render_export_rgb8(&img, edit.method, edit.geometry, &edit.params);
            let mut buf: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(w as u32, h as u32, rgb)
                .ok_or_else(|| anyhow::anyhow!("empty render"))?;
            if let Some((tw, th)) = target(w, h) {
                buf = image::imageops::resize(&buf, tw, th, FilterType::Triangle);
            }
            let mut f = std::io::BufWriter::new(std::fs::File::create(&dest_ext)?);
            let q = settings.quality.clamp(1, 100) as u8;
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, q)
                .write_image(buf.as_raw(), buf.width(), buf.height(), image::ExtendedColorType::Rgb8)
                .map_err(|e| anyhow::anyhow!("encode jpeg: {e}"))?;
        }
        ExportFormat::Png | ExportFormat::Tiff => {
            let (w, h, rgb) =
                crate::export::render_export_rgb16(&img, edit.method, edit.geometry, &edit.params);
            let mut buf: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_raw(w as u32, h as u32, rgb)
                .ok_or_else(|| anyhow::anyhow!("empty render"))?;
            if let Some((tw, th)) = target(w, h) {
                buf = image::imageops::resize(&buf, tw, th, FilterType::Triangle);
            }
            // `save` infers the encoder from the extension (.png/.tif → 16-bit).
            buf.save(&dest_ext).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        }
    }
    Ok(())
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

    if db_path.is_empty() {
        return None; // can't import into in-memory demo
    }
    // Route through open_catalog rather than a bare Connection::open: it
    // bootstraps the catalog schema on a fresh /config (a brand-new container
    // library.db has no tables — the C app used to create them on first launch)
    // AND sets a 3s busy_timeout, so the writes below wait out the metadata
    // writers' brief off-thread library.db write lock instead of taking an
    // immediate SQLITE_BUSY. That matters here because image_insert's error is
    // ignored (best-effort), so an un-waited lock would silently drop images.
    let conn = darkroom_db::schema::open_catalog(db_path).ok()?;

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

            // Count only rows that actually landed, and log the rest — a
            // swallowed insert error (not the lock wait) is what would otherwise
            // let images silently vanish AND make the "imported N" toast over-
            // report files that were walked but never written.
            match image::image_insert(&conn, film_id, filename, w, h) {
                Ok(_) => count += 1,
                Err(e) => eprintln!("darkroom import: image_insert failed for {filename:?}: {e}"),
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    // A unique temp library.db seeded so a raw path resolves to an image id.
    fn seeded_db() -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir().join(format!(
            "darkroom_export_edit_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("library.db");
        let dbs = db.to_str().unwrap().to_string();
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE main.film_rolls (id INTEGER, folder VARCHAR);
             CREATE TABLE main.images (id INTEGER, film_id INTEGER, filename VARCHAR);
             INSERT INTO main.film_rolls (id, folder) VALUES (1, '/p');
             INSERT INTO main.images (id, film_id, filename) VALUES (7, 1, 'a.dng');
             INSERT INTO main.images (id, film_id, filename) VALUES (8, 1, 'b.jpg');",
        )
        .unwrap();
        (dir, dbs)
    }

    #[test]
    fn export_edit_only_for_edited_raws() {
        let (dir, db) = seeded_db();

        // Unedited raw → None (falls back to darktable-cli).
        assert!(load_export_edit(&db, "/p/a.dng").is_none());
        // Empty db path → None.
        assert!(load_export_edit("", "/p/a.dng").is_none());

        // Persist an edit on the raw, then it must render via our pipeline.
        let mut params = crate::preview::PreviewParams::default();
        params.ev = -0.5;
        crate::persist::save_params(&db, "/p/a.dng", &params);

        let edit = load_export_edit(&db, "/p/a.dng").expect("edited raw exports via Rust");
        assert_eq!(edit.params, params);
        assert_eq!(edit.method, darkroom_core::rawimage::DemosaicMethod::default());

        // A non-raw with the same persisted params still uses darktable-cli.
        crate::persist::save_params(&db, "/p/b.jpg", &params);
        assert!(load_export_edit(&db, "/p/b.jpg").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn import_folder_bootstraps_fresh_config_and_counts_only_raws() {
        // The container first-run case: a config dir with no library.db / data.db
        // yet. import_folder_sync must bootstrap the catalog via open_catalog and
        // register exactly the raw-extension files it walked (count reflects rows
        // that landed, not files seen).
        let base = std::env::temp_dir().join(format!(
            "darkroom_import_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let cfg = base.join("config");
        let photos = base.join("photos");
        std::fs::create_dir_all(&cfg).unwrap();
        std::fs::create_dir_all(&photos).unwrap();
        // Empty files: probe_dims fails gracefully → 0×0, the insert still lands.
        for name in ["a.dng", "b.cr2", "c.NEF", "notes.txt"] {
            std::fs::write(photos.join(name), b"").unwrap();
        }
        let db = cfg.join("library.db");
        let dbs = db.to_str().unwrap();

        // a.dng, b.cr2, c.NEF are raws (case-insensitive); notes.txt is not.
        assert_eq!(import_folder_sync(photos.to_str().unwrap(), dbs), Some(3));
        // open_catalog materialised the sibling data.db during the bootstrap.
        assert!(cfg.join("data.db").exists(), "data.db not created by bootstrap");
        // Empty db path stays a no-op (demo mode, no library).
        assert_eq!(import_folder_sync(photos.to_str().unwrap(), ""), None);

        let _ = std::fs::remove_dir_all(&base);
    }
}
