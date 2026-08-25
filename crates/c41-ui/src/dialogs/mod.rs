//! Dialogs for Darkroom -- export, preferences, about.
//!
//! Phase 3-ui-5: export dialog that invokes darkroom-cli to export
//! selected images to a chosen output directory.

use adw::prelude::*;
use anyhow::Result;

/// Build the per-image [`ExportEdit`] for a lighttable batch export by loading
/// the image's persisted c41-ui state. Returns `Some` **only** for a raw
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
/// Returns `None` for an unedited raw (or a non-raw / empty db). The export loop
/// treats a `None` raw as [`default_raw_export_edit`] — the preview's seed for a
/// freshly-opened raw — so **every** raw now exports through the Rust pipeline
/// (WYSIWYG with the darkroom view); `Some` just carries the persisted edit for a
/// raw the user actually touched. (Milestone 5: this removed the darktable-cli
/// fallback for unedited raws; only non-raw formats still use the cli.)
fn load_export_edit(db_path: &str, path: &str) -> Option<crate::export::ExportEdit> {
    if db_path.is_empty() || !crate::raw_preview::is_raw_path(path) {
        return None;
    }
    // One connection, one path→imgid resolution for all three pieces.
    let (saved, geometry, method) = crate::persist::load_edit_state(db_path, path);

    let edited = saved.is_some()
        || !geometry.is_identity()
        || method != c41_core::rawimage::DemosaicMethod::default();
    if !edited {
        return None;
    }

    let params = crate::darkroom::initial_params(saved, true);
    // Lens-correction gear: resolve the persisted camera/lens choice while the
    // module is enabled, so a corrected preview exports corrected. An
    // unselected/unresolvable pair yields `None`, omitting the stage exactly
    // like the preview does.
    let lens = if params.lens_on {
        crate::preview::resolve_gear(&crate::persist::load_lens(db_path, path))
    } else {
        None
    };
    Some(crate::export::ExportEdit { method, geometry, params, lens })
}

/// The export edit for an **unedited** raw: exactly the seed the darkroom preview
/// shows for a freshly-opened raw — [`crate::darkroom::initial_params`] with no
/// saved edit (sigmoid on), the default demosaic method, and identity geometry.
/// Used by the export loop so an unedited raw develops through the Rust pipeline
/// (matching the preview) instead of darktable-cli. Keeping this in lockstep with
/// the preview's own seeding is the "export == preview" invariant for raws.
fn default_raw_export_edit() -> crate::export::ExportEdit {
    crate::export::ExportEdit {
        method: c41_core::rawimage::DemosaicMethod::default(),
        geometry: c41_core::geometry::Geometry::default(),
        params: crate::darkroom::initial_params(None, true),
        lens: None, // a fresh raw has no lens choice to resolve
    }
}

/// Show the export dialog for a list of image paths.
///
/// Presents format and quality choices, then renders each image — every raw
/// (edited or not) through our Rust pipeline, non-raw formats via `darkroom-cli`.
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
        // Shadow with an owned clone so the (non-Copy) fixed edit can move into
        // the async block while the response closure itself stays `Fn`.
        let edit = edit.clone();

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
        // A fixed `edit` is the single-image darkroom export (exactly one path);
        // the lighttable multi-export passes `None` + a db_path and resolves each
        // image's edit per-path. Baking one image's edit — especially its absolute
        // crop rectangle — onto a whole batch would silently mis-crop, so lock the
        // invariant here (call sites guarantee it; this catches a future regression).
        debug_assert!(
            edit.is_none() || paths.len() == 1,
            "a fixed ExportEdit must apply to a single-image export"
        );
        // Runtime guard too (debug_assert compiles out in release): the violation
        // is silent mis-crop of user files (one image's absolute crop baked onto
        // all), not a crash — so fail the whole batch loudly rather than corrupt.
        if edit.is_some() && paths.len() != 1 {
            eprintln!(
                "darkroom export: refusing to apply one fixed edit to {} images",
                paths.len()
            );
            return paths.len(); // count all as failed; export nothing
        }
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

            // Raws ALWAYS develop through the Rust pipeline so the export matches
            // the darkroom preview (WYSIWYG): an edited raw bakes its persisted
            // edit; an unedited raw uses the same seed the preview shows for a
            // freshly-opened file (`default_raw_export_edit` — initial_params,
            // default demosaic, identity geometry) rather than darktable-cli's
            // different default look. A fixed `edit` (single-image darkroom export)
            // applies to its one path; otherwise each raw's edit is loaded from the
            // catalog (lighttable multi-export).
            if crate::raw_preview::is_raw_path(path) {
                let img_edit = edit
                    .clone()
                    .or_else(|| db_path.as_deref().and_then(|db| load_export_edit(db, path)))
                    .unwrap_or_else(default_raw_export_edit);
                if let Err(e) = render_raw_export(path, &dest, &settings, img_edit) {
                    eprintln!("darkroom export: Rust render failed for {path}: {e}");
                    failed += 1;
                }
                continue;
            }

            // Non-raw formats the pure-Rust `image` crate can decode (JPEG/PNG/
            // TIFF) also develop through the Rust pipeline — the SAME colour params
            // as the preview (geometry/demosaic are raw-only), though not
            // byte-identical to it (different decoder + 8-bit; see
            // render_nonraw_export). Bakes a single-image darkroom edit if present,
            // else the per-path persisted params seeded like the preview (sigmoid
            // off). Only formats with no Rust decoder (heic/heif/avif) still use cli.
            if is_rust_image_path(path) {
                let (params, lens) = match &edit {
                    Some(e) => (e.params, e.lens.clone()),
                    None => {
                        let saved =
                            db_path.as_deref().and_then(|db| crate::persist::load_saved(db, path));
                        let params = crate::darkroom::initial_params(saved, false);
                        // Same gear resolution as the raw path: a persisted
                        // camera/lens choice with the module enabled exports
                        // corrected, matching the preview.
                        let lens = if params.lens_on {
                            db_path
                                .as_deref()
                                .and_then(|db| {
                                    crate::preview::resolve_gear(&crate::persist::load_lens(db, path))
                                })
                        } else {
                            None
                        };
                        (params, lens)
                    }
                };
                if let Err(e) = render_nonraw_export(path, &dest, &settings, &params, lens.as_deref()) {
                    eprintln!("darkroom export: Rust render failed for {path}: {e}");
                    failed += 1;
                }
                continue;
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
    use image::{imageops::FilterType, ImageBuffer, Rgb};

    let img = c41_core::rawimage::load(path).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    let dest_ext = format!("{dest}.{}", settings.format.out_ext());

    // Target size for the optional resize box (None ⇒ keep source size).
    // `fit_within` already guarantees each dimension ≥ 1. The resize below runs
    // in sRGB-encoded (gamma) space — technically inexact, but matches gdk-pixbuf
    // and virtually every other tool; acceptable for export.
    let target = |w: usize, h: usize| -> Option<(u32, u32)> {
        settings.resize.as_ref().map(|r| crate::export::fit_within(w as u32, h as u32, r))
    };

    // Encode to a temp file, then atomically rename onto `dest_ext` (see
    // `atomic_write`): a mid-encode failure must never leave a truncated file
    // that looks like a valid export, and a failed re-export must not clobber a
    // prior good file (File::create / the encoder truncate up front). The decode
    // above already fails before any file is touched, so a bad raw leaves nothing.
    atomic_write(&dest_ext, |out| {
        let lens_gear = edit.lens.as_deref();
        match settings.format {
            ExportFormat::Jpeg => {
                let (w, h, rgb) = crate::export::render_export_rgb8_gear(
                    &img,
                    edit.method,
                    edit.geometry,
                    &edit.params,
                    lens_gear,
                );
                let mut buf: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(w as u32, h as u32, rgb)
                    .ok_or_else(|| anyhow::anyhow!("empty render"))?;
                if let Some((tw, th)) = target(w, h) {
                    buf = image::imageops::resize(&buf, tw, th, FilterType::Triangle);
                }
                write_jpeg_rgb8(&buf, settings.quality, out)?;
            }
            ExportFormat::Png | ExportFormat::Tiff => {
                let (w, h, rgb) = crate::export::render_export_rgb16_gear(
                    &img,
                    edit.method,
                    edit.geometry,
                    &edit.params,
                    lens_gear,
                );
                let mut buf: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_raw(w as u32, h as u32, rgb)
                    .ok_or_else(|| anyhow::anyhow!("empty render"))?;
                if let Some((tw, th)) = target(w, h) {
                    buf = image::imageops::resize(&buf, tw, th, FilterType::Triangle);
                }
                // The temp path ends in `.part`, so `save`'s extension inference
                // (which the dest-ext path relied on) can't pick the encoder —
                // name the 16-bit format explicitly.
                let fmt = match settings.format {
                    ExportFormat::Png => image::ImageFormat::Png,
                    _ => image::ImageFormat::Tiff,
                };
                buf.save_with_format(out, fmt).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            }
        }
        Ok(())
    })
}

/// Write via a `<dest>.part` temp file and `rename` onto `dest` only on success,
/// so a failed `write` never leaves a truncated file at `dest` nor clobbers a
/// prior good one (image encoders truncate the destination up front). `rename`
/// within a directory is atomic on POSIX. On a write error the temp file is
/// removed so no `.part` turd survives.
fn atomic_write(dest: &str, write: impl FnOnce(&str) -> Result<()>) -> Result<()> {
    // Unique temp name (pid + process-local counter) in dest's own directory, so
    // two concurrent exports to the same dest can't grab each other's half-written
    // file, and the rename stays same-filesystem (atomic on POSIX).
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = format!(
        "{dest}.{}.{}.part",
        std::process::id(),
        NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    if let Err(e) = write(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // fsync the finished temp BEFORE the rename can promote it. On delalloc
    // filesystems (ext4/xfs/btrfs — every Linux target here) a plain write()
    // returns Ok even when the volume is full; ENOSPC is deferred to writeback.
    // Without this, a disk-full export would rename a truncated file over the
    // prior good one. (fsync on a read-only handle still flushes the inode's
    // dirty pages.) The parent-dir fsync for rename crash-durability is out of
    // scope — the invariant defended here is "never promote a truncated file".
    if let Err(e) = std::fs::File::open(&tmp).and_then(|f| f.sync_all()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow::anyhow!("fsync {tmp}: {e}"));
    }
    std::fs::rename(&tmp, dest).map_err(|e| {
        // A complete temp but a failed rename (dest dir vanished, perms) — unlink
        // it too so no orphaned `.part` survives.
        let _ = std::fs::remove_file(&tmp);
        anyhow::anyhow!("finalize {dest}: {e}")
    })
}

/// Encode an 8-bit RGB buffer as JPEG to `out`, flushing explicitly so a
/// `BufWriter` drop can't swallow the final-chunk write error (durability of the
/// bytes is `atomic_write`'s fsync). Shared by the raw and non-raw export paths.
fn write_jpeg_rgb8(
    buf: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    quality: u32,
    out: &str,
) -> Result<()> {
    use image::ImageEncoder;
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    let q = quality.clamp(1, 100) as u8;
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, q)
        .write_image(buf.as_raw(), buf.width(), buf.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| anyhow::anyhow!("encode jpeg: {e}"))?;
    f.into_inner().map_err(|e| anyhow::anyhow!("flush jpeg: {e}"))?;
    Ok(())
}

/// Non-raw formats the pure-Rust `image` crate can decode/encode (its enabled
/// features: png/tiff/jpeg). These export through the Rust pipeline rather than
/// darktable-cli; heic/heif/avif have no Rust decoder and still use the cli.
const RUST_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "tif", "tiff"];

/// True if `path` is a non-raw image the `image` crate can decode (see
/// [`RUST_IMAGE_EXTENSIONS`]). Disjoint from [`crate::raw_preview::is_raw_path`].
fn is_rust_image_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| RUST_IMAGE_EXTENSIONS.contains(&e.as_str()))
}

/// Flatten an 8-bit RGBA `image` buffer over a white matte into packed RGB. A
/// plain `to_rgb8()` would drop alpha and leave arbitrary under-pixels (often
/// black) in transparent regions; white is the conventional export matte. Opaque
/// pixels (a=255) pass through byte-for-byte.
fn composite_rgba8_over_white(rgba: &image::RgbaImage) -> Vec<u8> {
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for (i, px) in rgba.pixels().enumerate() {
        let a = px[3] as u32;
        for c in 0..3 {
            rgb[i * 3 + c] = ((px[c] as u32 * a + 255 * (255 - a)) / 255) as u8;
        }
    }
    rgb
}

/// 16-bit sibling of [`composite_rgba8_over_white`] (channel max 65535). Opaque
/// pixels (a=65535) pass through byte-for-byte, so a 16-bit source stays lossless.
fn composite_rgba16_over_white(rgba: &image::ImageBuffer<image::Rgba<u16>, Vec<u16>>) -> Vec<u16> {
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let mut rgb = vec![0u16; w * h * 3];
    for (i, px) in rgba.pixels().enumerate() {
        let a = px[3] as u64;
        for c in 0..3 {
            rgb[i * 3 + c] = ((px[c] as u64 * a + 65535 * (65535 - a)) / 65535) as u16;
        }
    }
    rgb
}

/// Export a **non-raw** image (JPEG/PNG/TIFF) through the Rust pipeline, applying
/// the same colour pipeline as the darkroom preview's non-raw path (geometry/
/// demosaic are raw-only, so ignored) → optional resize → encode. **JPEG is 8-bit**
/// (its container is); **PNG/TIFF are 16-bit** (via [`crate::preview::
/// apply_pipeline_rgb16`]) so a 16-bit source round-trips losslessly and an edited
/// gradient doesn't band. Alpha is composited over white before the pipeline.
///
/// NOT byte-identical WYSIWYG with the on-screen preview: the preview decodes via
/// GdkPixbuf, this via the `image` crate (JPEG base pixels ±1-2 LSB; PNG/TIFF
/// agree), and the preview is 8-bit whereas PNG/TIFF export is 16-bit.
///
/// Atomic + fsync-durable write via [`atomic_write`], like the raw path.
fn render_nonraw_export(
    path: &str,
    dest: &str,
    settings: &crate::export::ExportSettings,
    params: &crate::preview::PreviewParams,
    lens_gear: Option<&crate::preview::LensGear>,
) -> Result<()> {
    use crate::export::ExportFormat;
    use image::{imageops::FilterType, ImageBuffer, Rgb};

    let decoded = image::ImageReader::open(path)
        .map_err(|e| anyhow::anyhow!("open {path}: {e}"))?
        .with_guessed_format()
        .map_err(|e| anyhow::anyhow!("probe {path}: {e}"))?
        .decode()
        .map_err(|e| anyhow::anyhow!("decode {path}: {e}"))?;
    let (w, h) = (decoded.width() as usize, decoded.height() as usize);
    let dest_ext = format!("{dest}.{}", settings.format.out_ext());
    let target = settings
        .resize
        .as_ref()
        .map(|r| crate::export::fit_within(w as u32, h as u32, r));

    match settings.format {
        // JPEG is an 8-bit container — decode + process at 8-bit.
        ExportFormat::Jpeg => {
            let rgb = composite_rgba8_over_white(&decoded.to_rgba8());
            let processed =
                crate::preview::apply_pipeline_gear(&rgb, w, h, w * 3, 3, params, lens_gear);
            atomic_write(&dest_ext, move |out| {
                let mut buf: ImageBuffer<Rgb<u8>, _> =
                    ImageBuffer::from_raw(w as u32, h as u32, processed)
                        .ok_or_else(|| anyhow::anyhow!("empty render"))?;
                if let Some((tw, th)) = target {
                    buf = image::imageops::resize(&buf, tw, th, FilterType::Triangle);
                }
                write_jpeg_rgb8(&buf, settings.quality, out)
            })
        }
        // PNG/TIFF at 16-bit: preserve a 16-bit source's precision and cut
        // requantisation banding on an edited gradient (an unedited 16-bit source
        // is a lossless passthrough via apply_pipeline_rgb16).
        ExportFormat::Png | ExportFormat::Tiff => {
            let rgb = composite_rgba16_over_white(&decoded.to_rgba16());
            let processed =
                crate::preview::apply_pipeline_rgb16_gear(&rgb, w, h, params, lens_gear);
            let fmt = match settings.format {
                ExportFormat::Png => image::ImageFormat::Png,
                _ => image::ImageFormat::Tiff,
            };
            atomic_write(&dest_ext, move |out| {
                let mut buf: ImageBuffer<Rgb<u16>, _> =
                    ImageBuffer::from_raw(w as u32, h as u32, processed)
                        .ok_or_else(|| anyhow::anyhow!("empty render"))?;
                if let Some((tw, th)) = target {
                    buf = image::imageops::resize(&buf, tw, th, FilterType::Triangle);
                }
                buf.save_with_format(out, fmt).map_err(|e| anyhow::anyhow!("encode: {e}"))
            })
        }
    }
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
    use c41_db::film;
    use c41_db::image;

    if db_path.is_empty() {
        return None; // can't import into in-memory demo
    }

    // Phase 1 — walk + probe every raw OFF any DB lock. probe_dims decodes each
    // file's header (slow I/O), so it must run before we hold a write lock: doing
    // it inside the insert transaction would pin library.db's write lock across
    // all that I/O and block the rating/colour-label writers for the whole import.
    let mut pending: Vec<(String, i32, i32, c41_core::exif::ExifMeta)> = Vec::new();
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
        if !RAW_EXTENSIONS.contains(&ext.as_str()) { continue; }
        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };
        // Probe image dimensions (fall back to 0×0 if unreadable).
        let (w, h) = probe_dims(path).unwrap_or((0, 0));
        // Probe the EXIF numeric subset for the rule-stack's properties
        // (m4-135): unreadable/absent tags stay NULL — see ExifMeta's doc.
        let meta = c41_core::exif::probe_or_none(path);
        pending.push((filename, w, h, meta));
    }

    // Route through open_catalog rather than a bare Connection::open: it
    // bootstraps the catalog schema on a fresh /config (a brand-new container
    // library.db has no tables — the C app used to create them on first launch)
    // AND sets a 3s busy_timeout, so the insert burst waits out the metadata
    // writers' brief off-thread library.db write lock instead of an immediate
    // SQLITE_BUSY.
    let conn = c41_db::schema::open_catalog(db_path).ok()?;

    // Phase 2 — one transaction around film-roll creation + all inserts. N
    // per-image autocommits would be N fsyncs + N lock acquisitions; a single
    // transaction collapses that to one commit, and since probing already happened
    // the write lock is held only for the fast insert burst (no I/O interleaved).
    // Crash-safety improves too — the roll and its images commit atomically, so a
    // mid-import crash leaves no half-populated film roll. `unchecked_transaction`
    // (vs `transaction`) because the DAOs borrow `&Connection`, not `&mut`.
    let tx = conn.unchecked_transaction().ok()?;
    let film_id = film::film_new(&conn, folder).ok()??;
    let mut count = 0usize;
    for (filename, w, h, meta) in &pending {
        let exif = image::ImageExif {
            exposure: meta.exposure,
            aperture: meta.aperture,
            iso: meta.iso,
            focal_length: meta.focal_length,
        };
        match image::image_insert(&conn, film_id, filename, *w, *h, exif) {
            Ok(_) => count += 1,
            Err(e) => {
                eprintln!("darkroom import: image_insert failed for {filename:?}: {e}");
                // image_insert dedupes via SELECT (a name clash is an Ok), so the
                // only reachable Err here is a genuine engine error (SQLITE_FULL/
                // IOERR/NOMEM/BUSY) — and those auto-roll-back the WHOLE tx and
                // drop us into autocommit. Continuing would then autocommit later
                // rows against a film_id that no longer exists AND desync `count`
                // from reality (the m4-64 count-lie). Bail on a vanished tx.
                if conn.is_autocommit() {
                    eprintln!("darkroom import: transaction poisoned mid-insert; aborting");
                    return None; // tx already rolled back; nothing to commit
                }
            }
        }
    }
    if let Err(e) = tx.commit() {
        // A failed COMMIT rolls back → 0 rows persist, so "Imported 0" is honest;
        // log the cause so a failed import isn't a silent bare zero.
        eprintln!("darkroom import: commit failed, nothing persisted: {e}");
        return None;
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
        assert_eq!(edit.method, c41_core::rawimage::DemosaicMethod::default());

        // A non-raw with the same persisted params still uses darktable-cli.
        crate::persist::save_params(&db, "/p/b.jpg", &params);
        assert!(load_export_edit(&db, "/p/b.jpg").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unedited_raw_export_default_matches_preview_seed() {
        // Milestone 5: an unedited raw exports through the Rust pipeline using the
        // SAME seed the darkroom preview shows for a freshly-opened raw (no cli
        // fallback). Pin that "export == preview default" invariant so a change to
        // the preview's seeding can't silently desync the export look.
        let e = default_raw_export_edit();
        assert_eq!(e.params, crate::darkroom::initial_params(None, true));
        assert!(e.params.sigmoid_on, "raw default tone-maps (sigmoid on)");
        assert_eq!(e.method, c41_core::rawimage::DemosaicMethod::default());
        assert!(e.geometry.is_identity(), "no crop/straighten on an unedited raw");
    }

    #[test]
    fn atomic_write_no_clobber_on_failure_and_replaces_on_success() {
        let dir = std::env::temp_dir().join(format!(
            "darkroom_atomic_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Temp names are unique (pid+nonce), so assert on "any `.part` in dir".
        let no_part_left = |dir: &std::path::Path| -> bool {
            std::fs::read_dir(dir).unwrap().all(|e| {
                !e.unwrap().file_name().to_string_lossy().ends_with(".part")
            })
        };
        let dest = dir.join("out.jpg");
        let dests = dest.to_str().unwrap().to_string();

        // A prior good export exists.
        std::fs::write(&dest, b"GOOD").unwrap();

        // A failing encode (even one that wrote a partial temp) must NOT clobber
        // the prior file and must leave no `.part` behind.
        let r = atomic_write(&dests, |tmp| {
            std::fs::write(tmp, b"partial").unwrap();
            Err(anyhow::anyhow!("simulated encode failure"))
        });
        assert!(r.is_err());
        assert_eq!(std::fs::read(&dest).unwrap(), b"GOOD", "prior file was clobbered");
        assert!(no_part_left(&dir), "temp .part left behind after write failure");

        // A successful encode atomically replaces the file, no `.part` left.
        atomic_write(&dests, |tmp| Ok(std::fs::write(tmp, b"NEW")?)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"NEW");
        assert!(no_part_left(&dir));

        // A rename failure (dest is a directory) must also leave no `.part`.
        let as_dir = dir.join("adir");
        std::fs::create_dir(&as_dir).unwrap();
        let r = atomic_write(as_dir.to_str().unwrap(), |tmp| Ok(std::fs::write(tmp, b"x")?));
        assert!(r.is_err(), "rename onto a directory should fail");
        assert!(no_part_left(&dir), "temp .part left behind after rename failure");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn is_rust_image_path_matches_only_rust_decodable_nonraws() {
        for p in ["a.jpg", "b.JPEG", "c.png", "d.tif", "e.TIFF"] {
            assert!(is_rust_image_path(p), "{p} should be Rust-decodable");
        }
        // Raws go through the raw branch, not here; heic/avif have no Rust decoder.
        for p in ["x.cr2", "y.dng", "z.heic", "w.avif", "noext"] {
            assert!(!is_rust_image_path(p), "{p} must not take the Rust non-raw path");
        }
    }

    #[test]
    fn render_nonraw_export_roundtrips_a_png_passthrough() {
        use image::{ImageBuffer, Rgb};
        let dir = std::env::temp_dir().join(format!(
            "darkroom_nonraw_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("in.png");
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(4, 3, |x, y| {
            Rgb([(x * 60) as u8, (y * 80) as u8, ((x + y) * 30) as u8])
        });
        img.save(&src).unwrap();

        let dest = dir.join("out"); // extension-less; the encoder appends `.png`
        let settings = crate::export::ExportSettings {
            format: crate::export::ExportFormat::Png,
            quality: 90,
            resize: None,
        };
        // Default (identity) params ⇒ empty pipeline ⇒ byte-exact passthrough, so
        // a PNG (lossless) export must reproduce the source pixels exactly.
        render_nonraw_export(
            src.to_str().unwrap(),
            dest.to_str().unwrap(),
            &settings,
            &crate::preview::PreviewParams::default(),
            None,
        )
        .unwrap();

        let out_png = format!("{}.png", dest.to_str().unwrap());
        let out = image::ImageReader::open(&out_png).unwrap().decode().unwrap().to_rgb8();
        assert_eq!((out.width(), out.height()), (4, 3));
        assert_eq!(out.as_raw(), img.as_raw(), "passthrough export must be pixel-exact");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_nonraw_export_applies_a_nonidentity_edit() {
        use image::{ImageBuffer, Rgb};
        let dir = std::env::temp_dir().join(format!(
            "darkroom_nonraw_edit_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // A flat mid-grey opaque PNG (lossless, so the pixel change is purely the
        // pipeline, not re-encode).
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(4, 4, Rgb([128, 128, 128]));
        let src = dir.join("in.png");
        img.save(&src).unwrap();

        // +1 EV must brighten every channel (exposure doubles scene-linear light).
        let mut params = crate::preview::PreviewParams::default();
        params.ev = 1.0;
        let dest = dir.join("out");
        let settings = crate::export::ExportSettings {
            format: crate::export::ExportFormat::Png,
            quality: 90,
            resize: None,
        };
        render_nonraw_export(
            src.to_str().unwrap(),
            dest.to_str().unwrap(),
            &settings,
            &params,
            None,
        )
        .unwrap();

        let out = image::ImageReader::open(format!("{}.png", dest.to_str().unwrap()))
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
        assert!(
            out.get_pixel(0, 0)[0] > 130,
            "exposure edit did not brighten the 128 input: got {}",
            out.get_pixel(0, 0)[0]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_nonraw_export_16bit_is_lossless_png_and_tiff() {
        use image::{ImageBuffer, Rgb};
        let dir = std::env::temp_dir().join(format!(
            "darkroom_nonraw16_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // 16-bit values whose low byte is non-zero — the old 8-bit path would
        // have dropped it. from_fn keeps them distinct per pixel.
        let img: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_fn(3, 2, |x, y| {
            Rgb([0x1234 + x as u16, 0xABCD, 0x00FF + y as u16])
        });
        let src = dir.join("in16.png");
        img.save(&src).unwrap();

        // Default params ⇒ empty pipeline ⇒ the 16-bit source must round-trip
        // EXACTLY through BOTH 16-bit encoders (PNG and TIFF are distinct branches
        // of save_with_format — a silent 8-bit downconvert in either would fail).
        for format in [
            crate::export::ExportFormat::Png,
            crate::export::ExportFormat::Tiff,
        ] {
            let ext = format.out_ext();
            let dest = dir.join(format!("out_{ext}"));
            let settings = crate::export::ExportSettings {
                format,
                quality: 90,
                resize: None,
            };
            render_nonraw_export(
                src.to_str().unwrap(),
                dest.to_str().unwrap(),
                &settings,
                &crate::preview::PreviewParams::default(),
                None,
            )
            .unwrap();

            let out = image::ImageReader::open(format!("{}.{ext}", dest.to_str().unwrap()))
                .unwrap()
                .decode()
                .unwrap()
                .to_rgb16();
            assert_eq!(out.as_raw(), img.as_raw(), "16-bit {ext} passthrough must be lossless");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn composite_rgba16_over_white_blends_alpha() {
        use image::{ImageBuffer, Rgba};
        // opaque red | fully transparent | half-alpha black-over-white.
        let rgba: ImageBuffer<Rgba<u16>, Vec<u16>> = ImageBuffer::from_fn(3, 1, |x, _| match x {
            0 => Rgba([0x8000, 0, 0, 0xFFFF]),      // opaque ⇒ unchanged
            1 => Rgba([0x1111, 0x2222, 0x3333, 0]), // transparent ⇒ white
            _ => Rgba([0, 0, 0, 0x8000]),           // half alpha, black over white
        });
        let out = composite_rgba16_over_white(&rgba);
        assert_eq!(&out[0..3], &[0x8000, 0, 0], "opaque must pass through");
        assert_eq!(&out[3..6], &[0xFFFF, 0xFFFF, 0xFFFF], "transparent must be white");
        // a=0x8000, src=0 ⇒ (0 + 65535*(65535-32768))/65535 = 32767 = 0x7FFF.
        assert_eq!(&out[6..9], &[0x7FFF, 0x7FFF, 0x7FFF], "half-alpha black over white");
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
        // The returned count reflects rows the transaction actually committed —
        // re-open and confirm the 3 images (and their film roll) persisted.
        let check = c41_db::schema::open_catalog(dbs).unwrap();
        assert_eq!(c41_db::image::image_count_all(&check).unwrap(), 3);
        drop(check);
        // Empty db path stays a no-op (demo mode, no library).
        assert_eq!(import_folder_sync(photos.to_str().unwrap(), ""), None);

        let _ = std::fs::remove_dir_all(&base);
    }
}
