//! Neural restore panel (u7b + u7c): the "Neural restore" sidebar section
//! that runs the RGB denoise task on the selected image.
//!
//! darktable's neural-restore lib (src/libs/neural_restore.c) drives the AI
//! denoise/raw-denoise/upscale tasks from the lighttable. This panel covers
//! **Denoise** plus its **Strength slider and before/after split preview**
//! (u7c). Raw denoise and upscale are shown but greyed ("coming next" — they
//! are separate tasks with their own CFA-style preprocessing and tile
//! ladders, still future work).
//!
//! The Run path mirrors the export flow: decode the source to linear RGBA,
//! call [`c41_core::ai::infer::denoise_rgb`] (the tiled ORT run), encode the
//! packed result through the export TIFF writer
//! ([`crate::dialogs::write_rgb16_tiff_atomic`] — the same encoder the export
//! panel uses, not a second one), register the new file with the catalogue
//! ([`crate::dialogs::import_single_file_sync`], the tether capture's exact
//! single-file import) and reload the grid through the caller's `on_done` —
//! the same reload the tether page's auto-imports use.
//!
//! u7c detail recovery: after a run, the panel caches the original RGBA and
//! the denoised RGB beside the output path. Moving Strength re-blends via
//! [`c41_core::ai::detail::apply_detail_recovery`] (`recovery_alpha =
//! 1 - strength/100`, neural_restore.c:885) with NO re-inference, repaints
//! the split preview, and OVERWRITES the same TIFF in place — no re-import,
//! the grid picks the new bytes up through the existing `on_done` reload
//! (after evicting the path from the thumbnail pixel cache, which keys
//! `(path, bucket)` with no mtime check and would otherwise repaint stale
//! bytes — see `thumbs::evict_path`).
//! Re-blends run on a blocking worker behind a 50 ms debounce (the C
//! re-blends the cached preview on the UI thread debounced to 50 ms —
//! `RAW_PREVIEW_STRENGTH_DEBOUNCE_MS`, neural_restore.c:2295; ours needs the
//! worker because a full-frame DWT costs milliseconds, not microseconds).
//! The split preview paints the original left of a draggable divider and the
//! current blend right of it with a white divider line, mirroring the C
//! split draw (neural_restore.c:3432-3466).
//!
//! The Run path mirrors the export flow: decode the source to linear RGBA,
//! call [`c41_core::ai::infer::denoise_rgb`] (the tiled ORT run), encode the
//! packed result through the export TIFF writer
//! ([`crate::dialogs::write_rgb16_tiff_atomic`] — the same encoder the export
//! panel uses, not a second one), register the new file with the catalogue
//! ([`crate::dialogs::import_single_file_sync`], the tether capture's exact
//! single-file import) and reload the grid through the caller's `on_done` —
//! the same reload the tether page's auto-imports use.
//!
//! The model half reuses the u7a infrastructure: a store scan
//! ([`c41_core::ai::infer::downloaded_denoise_models`]) feeds the picker's
//! downloaded rows; the known releases (NIND 55 MB / NAFNet 108 MB,
//! [`c41_core::ai::infer::known_denoise_models`]) fill the download-on-demand
//! rows. Picking one downloads it through the registry-verified downloader
//! (sha256-gated, [`c41_core::ai::infer::ensure_denoise_model`]). Downloads
//! need the GitHub releases listing, so a registry failure reports
//! "registry unavailable" as a status line — never a crash.
//!
//! All heavy work (decode, ORT inference, TIFF encode, catalogue write) runs
//! on a `gio::spawn_blocking` worker; GTK is only ever touched from the main
//! thread through `WeakRef` upgrades, and every widget touch is guarded so a
//! popped page can never be called into (the tether page's no-crash
//! discipline). A busy guard serialises downloads against runs, and a
//! 200 ms `timeout_add_local` tick renders live progress from a shared
//! `Arc<Mutex<...>>` cell that the worker's `FnMut` progress callbacks write
//! (cross-thread, so the cell is a mutex rather than an `Rc<RefCell>`).
//!
//! The pure model (row merging, labels, progress text, output naming) is
//! GTK-free and unit-tested headless; the GTK wiring over it is a thin
//! control surface, matching the crate's display-free discipline.

use adw::prelude::*;
use c41_core::ai::detail::strength_to_alpha;
use c41_core::ai::infer::{known_denoise_models, KnownModel};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

/// How often the progress tick repaints the status line while a run or a
/// download is in flight.
pub const TICK_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);
/// Strength-slider re-blend debounce: the last value within the window wins,
/// so a fast drag issues one worker pass, not one per pixel of travel.
/// Matches the C's 50 ms strength-reblend debounce intent
/// (`RAW_PREVIEW_STRENGTH_DEBOUNCE_MS`, neural_restore.c:2295).
pub const STRENGTH_DEBOUNCE: std::time::Duration = std::time::Duration::from_millis(50);
/// Longest edge of the split-preview thumbnails, in px. Full frames stay in
/// the float blend cache; the preview only ever holds these small pixbufs.
pub const PREVIEW_MAX_DIM: u32 = 480;
/// The picker row shown before any model exists.
const EMPTY_ROWS_TEXT: &str = "(no denoise models)";
/// Suffix the result TIFF beside the source: `<stem>_denoise.tif`
/// (neural_restore.c `_task_suffix` for NEURAL_TASK_DENOISE).
const DENOISE_SUFFIX: &str = "_denoise";

// ── Pure model (GTK-free, headless-tested) ─────────────────────────────────

/// One model-picker row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NeuralRow {
    /// A downloaded archive in the store — runnable right away.
    Model { name: String },
    /// A known release not yet downloaded; picking it starts a download.
    Download { name: String, size_mi_b: u32 },
    /// Placeholder shown when there are no models at all (never selectable).
    Empty,
}

/// Merge the downloaded store listing and the known release list into the
/// picker's row set: downloaded models first (already filename-sorted by the
/// store scan), then a download-on-demand entry for every known model not yet
/// present. An empty result degrades to a single non-selectable placeholder.
pub fn model_rows(downloaded: &[String], known: &[KnownModel]) -> Vec<NeuralRow> {
    let mut rows: Vec<NeuralRow> = downloaded
        .iter()
        .map(|n| NeuralRow::Model { name: n.to_string() })
        .collect();
    for k in known {
        if !downloaded.iter().any(|d| d == &k.name) {
            rows.push(NeuralRow::Download { name: k.name.clone(), size_mi_b: k.size_mi_b });
        }
    }
    if rows.is_empty() {
        rows.push(NeuralRow::Empty);
    }
    rows
}

/// The combo-row text for a row.
pub fn row_label(row: &NeuralRow) -> String {
    match row {
        NeuralRow::Model { name } => name.to_string(),
        NeuralRow::Download { name, size_mi_b } => {
            format!("Download {name} ({size_mi_b} MB)")
        }
        NeuralRow::Empty => EMPTY_ROWS_TEXT.to_string(),
    }
}

/// Whether a row is a downloaded, runnable model.
pub fn row_is_runnable(row: &NeuralRow) -> bool {
    matches!(row, NeuralRow::Model { .. })
}

/// Live per-tile run status line, e.g. `Denoising 3/12…`.
pub fn format_run_progress(done: u32, total: u32) -> String {
    format!("Denoising {done}/{total}…")
}

/// Live download status line with byte counts in MiB, e.g.
/// `Downloading — 12.3 of 55.0 MB` (no total until the first size lands).
pub fn format_download_progress(done: u64, total: Option<u64>) -> String {
    let mb = done as f64 / 1_048_576.0;
    match total {
        Some(t) if t > 0 => {
            let tot = t as f64 / 1_048_576.0;
            format!("Downloading — {:.1} of {:.1} MB", mb, tot)
        }
        _ => format!("Downloading — {:.1} MB", mb),
    }
}

/// The result path stem for a source: `<dir>/<stem>_denoise` beside the
/// source, mirroring neural_restore.c:1421-1447 (`$(FILE_FOLDER)` +
/// basename + `_denoise`). A bare filename has no parent, so the stem roots
/// at "." (CWD-relative) like the export template expansion does. When the
/// stem already ends with the suffix (re-processing a denoised file) it is
/// NOT appended again — the C `has_suffix` skip (neural_restore.c:1440).
pub fn denoise_output_stem(src_path: &str) -> String {
    let p = std::path::Path::new(src_path);
    let dir = match p.parent().and_then(|d| d.to_str()) {
        None | Some("") => ".".to_string(),
        Some(d) => d.trim_end_matches('/').to_string(),
    };
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("img");
    if stem.ends_with(DENOISE_SUFFIX) {
        format!("{dir}/{stem}")
    } else {
        format!("{dir}/{stem}{}", DENOISE_SUFFIX)
    }
}

/// The first free `<stem>.tif`, `<stem>_1.tif`, … (the C collision loop,
/// neural_restore.c:1470-1495). Past 9999 collisions this returns `None`
/// and the run fails loudly — the C skips the file ("too many output
/// files") rather than overwriting, and so do we.
pub fn unique_output_path(dest_no_ext: &str) -> Option<String> {
    let plain = format!("{dest_no_ext}.tif");
    if !std::path::Path::new(&plain).exists() {
        return Some(plain);
    }
    for n in 1..10000u32 {
        let p = format!("{dest_no_ext}_{n}.tif");
        if !std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    None
}

// ── Pure split-preview + re-blend model (GTK-free, headless-tested) ─────────

/// Clamp a split-divider position to the image, 0..1. NaN (a drag event with
/// no valid coordinate) falls back to the C's initial 0.5
/// (neural_restore.c:4129).
pub fn clamp_split_pos(v: f64) -> f64 {
    if v.is_nan() {
        0.5
    } else {
        v.clamp(0.0, 1.0)
    }
}

/// Contain-fit geometry for the split preview: `(scale, ox, oy, div_x)` for
/// an `img_w`×`img_h` pixbuf inside an `alloc_w`×`alloc_h` allocation at
/// split `split` — the same quantities the C split draw derives
/// (neural_restore.c:3947-3952 for the drag side, :3432 for the paint side).
/// Deviation: the C floors the scale at 1.0 (`fmax(1.0, …)`) because it
/// previews a small fixed-size patch; here the pixbufs are thumbnails of
/// arbitrary frames, so a plain contain-fit (possibly < 1) is correct.
/// Degenerate inputs yield all zeros rather than NaN or infinities.
pub fn split_geometry(
    alloc_w: f64,
    alloc_h: f64,
    img_w: f64,
    img_h: f64,
    split: f64,
) -> (f64, f64, f64, f64) {
    if alloc_w <= 0.0 || alloc_h <= 0.0 || img_w <= 0.0 || img_h <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let scale = (alloc_w / img_w).min(alloc_h / img_h);
    if !scale.is_finite() || scale <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let ox = (alloc_w - img_w * scale) / 2.0;
    let oy = (alloc_h - img_h * scale) / 2.0;
    let div_x = ox + clamp_split_pos(split) * img_w * scale;
    (scale, ox, oy, div_x)
}

/// Divider position for a pointer x in allocation space: `CLAMP((ex - ox) /
/// img_w, 0, 1)` (neural_restore.c:3952). A degenerate image width keeps the
/// current position instead of dividing by zero.
pub fn split_pos_from_x(x: f64, ox: f64, img_w: f64, current: f64) -> f64 {
    if img_w <= 0.0 {
        clamp_split_pos(current)
    } else {
        clamp_split_pos((x - ox) / img_w)
    }
}

/// Linear working-space value to 8-bit sRGB for preview bytes: the same
/// OETF the export TIFF path applies, quantized to 255 levels.
pub fn linear_to_srgb8(v: f32) -> u8 {
    (c41_core::color::linear_to_srgb(v).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Preview thumbnail dims: longest edge capped at `max_dim`, aspect kept.
/// Images already under the cap pass through untouched.
pub fn thumbnail_dims(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    let longest = w.max(h);
    if longest <= max_dim || longest == 0 || max_dim == 0 {
        return (w, h);
    }
    let scale = max_dim as f64 / longest as f64;
    let tw = ((w as f64 * scale).round() as u32).max(1);
    let th = ((h as f64 * scale).round() as u32).max(1);
    (tw, th)
}

/// Nearest-neighbour downscale of packed linear RGB to sRGB bytes for the
/// split preview's "after" side. Pure and headless-tested; the pixbuf itself
/// (`Pixbuf::from_bytes`, main thread only — pixbufs are `!Send`) is built
/// by the caller from the returned bytes.
pub fn shrink_rgb_to_srgb8(src: &[f32], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    let mut out = vec![0u8; tw as usize * th as usize * 3];
    if w == 0 || h == 0 || tw == 0 || th == 0 {
        return out;
    }
    let expect = w as usize * h as usize * 3;
    if src.len() != expect {
        return out;
    }
    for y in 0..th {
        let sy = (y as u64 * h as u64 / th as u64) as u32;
        for x in 0..tw {
            let sx = (x as u64 * w as u64 / tw as u64) as u32;
            let si = (sy as usize * w as usize + sx as usize) * 3;
            let di = (y as usize * tw as usize + x as usize) * 3;
            out[di] = linear_to_srgb8(src[si]);
            out[di + 1] = linear_to_srgb8(src[si + 1]);
            out[di + 2] = linear_to_srgb8(src[si + 2]);
        }
    }
    out
}

/// Same as [`shrink_rgb_to_srgb8`] for the preview's "before" side, whose
/// cache is interleaved linear RGBA: flatten over white first, exactly like
/// the non-raw decode path below composites (`lin * a + (1 - a)`).
pub fn shrink_rgba_to_srgb8(src: &[f32], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    let mut out = vec![0u8; tw as usize * th as usize * 3];
    if w == 0 || h == 0 || tw == 0 || th == 0 {
        return out;
    }
    let expect = w as usize * h as usize * 4;
    if src.len() != expect {
        return out;
    }
    for y in 0..th {
        let sy = (y as u64 * h as u64 / th as u64) as u32;
        for x in 0..tw {
            let sx = (x as u64 * w as u64 / tw as u64) as u32;
            let si = (sy as usize * w as usize + sx as usize) * 4;
            let di = (y as usize * tw as usize + x as usize) * 3;
            let a = src[si + 3];
            for c in 0..3 {
                out[di + c] = linear_to_srgb8(src[si + c] * a + (1.0 - a));
            }
        }
    }
    out
}

/// Quantise packed linear RGB to u16 through the sRGB OETF with
/// round-half-up — the exact encode the export TIFF path applies
/// (`render_linear_to_srgb16_gear` then `(enc * 65535 + 0.5) as u16`,
/// preview.rs). Shared by the initial run worker and the strength
/// re-blend worker so both TIFFs encode identically.
pub fn quantize_srgb16(rgb: &[f32]) -> Vec<u16> {
    let mut out = vec![0u16; rgb.len()];
    for (i, v) in rgb.iter().enumerate() {
        let enc = c41_core::color::linear_to_srgb(*v);
        out[i] = (enc.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
    out
}

/// Status line after a strength re-blend lands, e.g. `Strength 80% — saved`.
pub fn format_strength_done(strength: f32) -> String {
    format!("Strength {:.0}% — saved", strength.clamp(0.0, 100.0))
}

// ── Worker (decode → denoise → write → import) ─────────────────────────────

/// What a completed run produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunReport {
    pub out_path: String,
    pub imported: bool,
}

/// Decode `path` to interleaved linear RGBA f32 at `w`×`h` — the
/// [`c41_core::ai::infer::denoise_rgb`] input contract. Raws develop through
/// the c41-core decoder (`to_linear_rgba_with`, defaults: no colour edits are
/// applied — a deviation from the C, which exports through the user's full
/// develop pipeline). Non-raw formats decode via the image crate and are
/// linearized (inverse sRGB OETF) since their containers store
/// gamma-encoded values — the same "working profile, treated as sRGB"
/// approximation restore_rgb.c makes without a profile.
fn decode_linear_rgba(path: &str) -> Result<(u32, u32, Vec<f32>), String> {
    if crate::raw_preview::is_raw_path(path) {
        let img = c41_core::rawimage::load(std::path::Path::new(path))
            .map_err(|e| format!("decode: {e}"))?;
        let (w, h, rgba) = img.to_linear_rgba();
        Ok((w as u32, h as u32, rgba))
    } else {
        let decoded = image::ImageReader::open(path)
            .map_err(|e| format!("open: {e}"))?
            .with_guessed_format()
            .map_err(|e| format!("probe: {e}"))?
            .decode()
            .map_err(|e| format!("decode: {e}"))?;
        let rgba = decoded.to_rgba16();
        let (w, h) = (rgba.width(), rgba.height());
        let mut out = vec![0f32; w as usize * h as usize * 4];
        for (i, px) in rgba.pixels().enumerate() {
            let a = px[3] as f32 / 65535.0;
            for c in 0..3 {
                let lin =
                    c41_core::color::srgb_to_linear(px[c] as f32 / 65535.0);
                out[i * 4 + c] = lin * a + (1.0 - a); // composite over white
            }
            out[i * 4 + 3] = 1.0;
        }
        Ok((w, h, out))
    }
}

/// The full-frame float buffers a completed run leaves behind so the
/// Strength slider can re-blend without re-inference (the C caches
/// src/denoised dims for the same reason, neural_restore.c:326). Buffers
/// are `Arc` so each debounced worker pass clones pointers, not pixels.
#[derive(Clone, Debug)]
struct BlendCache {
    original: Arc<Vec<f32>>,
    denoised: Arc<Vec<f32>>,
    w: u32,
    h: u32,
    out_path: String,
}

/// The whole denoise run for one image: decode → prepare the model →
/// [`c41_core::ai::infer::denoise_rgb`] (tiled ORT inference, progress
/// forwarded) → quantise the packed linear result to u16 → write the result
/// TIFF beside the source through the export writer → register it with the
/// catalogue. Runs entirely on a blocking worker; GTK is never touched here.
/// Returns the report plus the float buffers the Strength re-blend needs.
fn run_denoise_worker(
    path: &str,
    db: &str,
    model_name: &str,
    progress: impl FnMut(u32, u32),
) -> Result<(RunReport, BlendCache), String> {
    let (w, h, rgba) = decode_linear_rgba(path)?;
    let prepared = c41_core::ai::infer::prepare_denoise_model(model_name)
        .map_err(|e| format!("model prepare: {e}"))?;
    let out_rgb = c41_core::ai::infer::denoise_rgb(
        &rgba,
        w,
        h,
        &prepared.onnx_path,
        prepared.tile_size,
        progress,
    )
    .map_err(|e| format!("inference: {e}"))?;
    let rgb16 = quantize_srgb16(&out_rgb);
    let dest = match unique_output_path(&denoise_output_stem(path)) {
        Some(d) => d,
        None => return Err("too many output files beside the source".to_string()),
    };
    crate::dialogs::write_rgb16_tiff_atomic(&dest, w, h, rgb16)
        .map_err(|e| format!("write tiff: {e}"))?;
    let dir = std::path::Path::new(&dest)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let filename = std::path::Path::new(&dest)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut imported = false;
    if !db.is_empty() && !dir.is_empty() && !filename.is_empty() {
        imported = crate::dialogs::import_single_file_sync(&dir, &filename, db);
    }
    let cache = BlendCache {
        original: Arc::new(rgba),
        denoised: Arc::new(out_rgb),
        w,
        h,
        out_path: dest.clone(),
    };
    Ok((RunReport { out_path: dest, imported }, cache))
}

/// One Strength re-blend pass: `denoised + alpha * filtered_residual` via
/// [`c41_core::ai::detail::apply_detail_recovery`] (NO re-inference),
/// quantised and written OVER the same TIFF path in place — no re-import,
/// the grid reload picks the new bytes up. Returns the preview's "after"
/// sRGB bytes at thumbnail size plus their dims; the caller builds the
/// pixbuf on the main thread (pixbufs are `!Send`). Entirely worker-side.
fn run_reblend_worker(cache: &BlendCache, strength: f32) -> Result<(Vec<u8>, u32, u32), String> {
    let alpha = strength_to_alpha(strength);
    let blended = c41_core::ai::detail::apply_detail_recovery(
        &cache.original,
        &cache.denoised,
        cache.w,
        cache.h,
        alpha,
    );
    let rgb16 = quantize_srgb16(&blended);
    crate::dialogs::write_rgb16_tiff_atomic(&cache.out_path, cache.w, cache.h, rgb16)
        .map_err(|e| format!("rewrite tiff: {e}"))?;
    let (tw, th) = thumbnail_dims(cache.w, cache.h, PREVIEW_MAX_DIM);
    let thumb = shrink_rgb_to_srgb8(&blended, cache.w, cache.h, tw, th);
    Ok((thumb, tw, th))
}

// ── Shared panel state ─────────────────────────────────────────────────────

/// What the panel is currently doing, rendered by the progress tick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunPhase {
    Idle,
    Downloading { done: u64, total: Option<u64> },
    Denoising { done: u32, total: u32 },
}

/// Mutable panel state, shared between the GTK handlers (main thread) and
/// the blocking workers (via progress callbacks) behind an `Arc<Mutex<..>>`.
/// Pixbufs never live here — they are `!Send`, so the split preview owns
/// them on the main thread in [`SplitPreview`].
struct NeuralState {
    rows: Vec<NeuralRow>,
    selected: u32,
    busy: bool,
    run_phase: RunPhase,
    /// Full-frame buffers from the last completed run; `None` until then.
    blend: Option<BlendCache>,
    /// Last applied Strength value (slider domain 0..100).
    applied_strength: f32,
    /// A re-blend worker is in flight; slider ticks park in
    /// `pending_strength` instead of stacking workers.
    reblend_busy: bool,
    pending_strength: Option<f32>,
    /// Bumped on every re-blend kick AND every new run: a completion whose
    /// seq mismatches is stale (a newer run replaced the cache under it)
    /// and must not repaint the preview.
    reblend_seq: u64,
}

impl NeuralState {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            selected: 0,
            busy: false,
            run_phase: RunPhase::Idle,
            blend: None,
            applied_strength: 100.0,
            reblend_busy: false,
            pending_strength: None,
            reblend_seq: 0,
        }
    }
}

/// Main-thread-only split-preview paint state: the before/after pixbufs at
/// thumbnail size plus the draggable divider position 0..1 (the C's
/// `split_pos`, neural_restore.c:289). Owned by an `Rc<RefCell<..>>`
/// captured by the draw func and the drag gesture — never shared with
/// workers, since pixbufs are `!Send`.
struct SplitPreview {
    before: Option<gtk4::gdk_pixbuf::Pixbuf>,
    after: Option<gtk4::gdk_pixbuf::Pixbuf>,
    split: f64,
}

impl SplitPreview {
    fn new() -> Self {
        Self { before: None, after: None, split: 0.5 }
    }
}

/// Build a main-thread pixbuf from sRGB preview bytes. `None` on empty
/// input — the caller keeps the previous pixbuf instead of blanking.
fn pixbuf_from_srgb8(bytes: Vec<u8>, w: u32, h: u32) -> Option<gtk4::gdk_pixbuf::Pixbuf> {
    if bytes.is_empty() || w == 0 || h == 0 {
        return None;
    }
    let buf = glib::Bytes::from_owned(bytes);
    Some(gtk4::gdk_pixbuf::Pixbuf::from_bytes(
        &buf,
        gtk4::gdk_pixbuf::Colorspace::Rgb,
        false,
        8,
        w as i32,
        h as i32,
        w as i32 * 3,
    ))
}

// ── Widget helpers (main thread only) ──────────────────────────────────────

/// Rebuild the picker's row list from the store + known releases, keeping the
/// current selection when it still exists. Called on construction and after a
/// download lands.
fn refresh_model_picker(combo: &adw::ComboRow, state: &mut NeuralState) {
    let downloaded = c41_core::ai::infer::downloaded_denoise_models();
    let known = known_denoise_models();
    state.rows = model_rows(&downloaded, &known);
    let labels: Vec<String> = state.rows.iter().map(row_label).collect();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    combo.set_model(Some(&gtk4::StringList::new(&label_refs)));
    if state.selected >= state.rows.len() as u32 {
        state.selected = 0;
    }
    combo.set_selected(state.selected);
}

/// Recompute the Run button's sensitivity from the current selection, switch
/// and busy state.
fn refresh_run_sensitive(
    run_btn: &gtk4::Button,
    denoise_sw: &adw::SwitchRow,
    state: &NeuralState,
) {
    let runnable = !state.busy
        && denoise_sw.is_active()
        && state
            .rows
            .get(state.selected as usize)
            .is_some_and(row_is_runnable);
    run_btn.set_sensitive(runnable);
}

/// Arm the live-progress tick (self-stopping): repaints the status line from
/// the shared state cell until the phase returns to Idle, or the page dies
/// (WeakRef upgrade fails). Mirrors the tether watch timer's shape.
fn arm_progress_tick(
    state: Arc<Mutex<NeuralState>>,
    status: glib::WeakRef<gtk4::Label>,
) {
    let _ = glib::timeout_add_local(TICK_INTERVAL, move || {
        let phase = match state.lock() {
            Ok(st) => st.run_phase.clone(),
            Err(_) => return glib::ControlFlow::Break,
        };
        let Some(lbl) = status.upgrade() else {
            return glib::ControlFlow::Break;
        };
        match phase {
            RunPhase::Idle => {
                // Done — the continuation writes the final line; stop here.
                glib::ControlFlow::Break
            }
            RunPhase::Downloading { done, total } => {
                lbl.set_text(&format_download_progress(done, total));
                glib::ControlFlow::Continue
            }
            RunPhase::Denoising { done, total } => {
                lbl.set_text(&format_run_progress(done, total));
                glib::ControlFlow::Continue
            }
        }
    });
}

/// One-shot status write from a main-thread continuation.
fn set_status_lbl(status: &glib::WeakRef<gtk4::Label>, text: &str) {
    if let Some(lbl) = status.upgrade() {
        lbl.set_text(text);
    }
}

/// Kick one debounced Strength re-blend pass (u7c).
///
/// Serialises on `reblend_busy`: a tick arriving mid-pass parks its strength
/// in `pending_strength` instead of stacking workers, and the completion
/// drains it toward the latest value. `reblend_seq` invalidates completions
/// overtaken by a newer run (the cache they blended no longer owns the
/// preview). The worker re-blends with NO re-inference and overwrites the
/// same TIFF in place; this continuation installs the "after" pixbuf,
/// repaints the split, writes the status line and reloads the grid through
/// `on_done` so the new bytes show — no re-import, same file.
fn kick_reblend(
    state: Arc<Mutex<NeuralState>>,
    preview: Rc<RefCell<SplitPreview>>,
    area_w: glib::WeakRef<gtk4::DrawingArea>,
    status_w: glib::WeakRef<gtk4::Label>,
    on_done: Rc<dyn Fn()>,
    strength: f32,
) {
    let (cache, seq) = {
        let Ok(mut st) = state.lock() else { return };
        if st.busy || st.reblend_busy {
            st.pending_strength = Some(strength);
            return;
        }
        let Some(cache) = st.blend.clone() else { return };
        st.reblend_busy = true;
        st.applied_strength = strength;
        st.reblend_seq += 1;
        (cache, st.reblend_seq)
    };
    set_status_lbl(&status_w, &format!("Blending {:.0}%…", strength.clamp(0.0, 100.0)));

    // Per-activation shadows for the single-shot future below (the outer
    // `Fn` closure owns the same-named bindings — the download/run pattern).
    // `out_path` travels by value: the worker borrows `cache`, and the
    // continuation needs the path for cache eviction after it moves.
    let out_path = cache.out_path.clone();
    let state_a = state.clone();
    let preview_a = preview.clone();
    let area_a = area_w.clone();
    let status_a = status_w.clone();
    let done_a = on_done.clone();
    let state_b = state.clone();
    let preview_b = preview.clone();
    let area_b = area_w.clone();
    let status_b = status_w.clone();
    let done_b = on_done.clone();
    glib::spawn_future_local(async move {
        let outcome =
            gio::spawn_blocking(move || run_reblend_worker(&cache, strength)).await;
        // Drained follow-up pass, kicked after the lock is released.
        let mut follow: Option<f32> = None;
        // Whether the grid reload + repaint apply (a stale pass only
        // releases the busy flag).
        let mut fresh_ok = false;
        match outcome {
            Ok(Ok((thumb, tw, th))) => {
                let Ok(mut st) = state_a.lock() else { return };
                st.reblend_busy = false;
                if st.reblend_seq == seq {
                    fresh_ok = true;
                    if let Some(pb) = pixbuf_from_srgb8(thumb, tw, th) {
                        preview_a.borrow_mut().after = Some(pb);
                    }
                    set_status_lbl(&status_a, &format_strength_done(strength));
                    if let Some(next) = st.pending_strength.take() {
                        if (next - strength).abs() > f32::EPSILON {
                            follow = Some(next);
                        }
                    }
                } else if let Some(next) = st.pending_strength.take() {
                    // Overtaken by a newer run/kick: the parked value still
                    // wants applying against the current cache. Kicking is
                    // safe — it re-reads the cache and serialises on busy.
                    follow = Some(next);
                }
            }
            Ok(Err(e)) => {
                let Ok(mut st) = state_a.lock() else { return };
                st.reblend_busy = false;
                if st.reblend_seq == seq {
                    set_status_lbl(&status_a, &format!("Detail update failed: {e}"));
                    follow = st.pending_strength.take();
                } else {
                    follow = st.pending_strength.take();
                }
            }
            Err(_) => {
                let Ok(mut st) = state_a.lock() else { return };
                st.reblend_busy = false;
                if st.reblend_seq == seq {
                    set_status_lbl(&status_a, "Detail update did not complete");
                    follow = st.pending_strength.take();
                } else {
                    follow = st.pending_strength.take();
                }
            }
        }
        if fresh_ok {
            if let Some(area) = area_a.upgrade() {
                area.queue_draw();
            }
            // The TIFF was overwritten in place: evict it from the thumbnail
            // pixel cache BEFORE the grid reload, or the grid repaints the
            // pre-blend bytes (the cache keys `(path, bucket)` with no mtime
            // check). Main thread here by construction (spawn_future_local
            // continuation), which the thread-local cache requires.
            crate::lighttable::thumbs::evict_path(&out_path);
            done_a();
        }
        if let Some(next) = follow {
            kick_reblend(state_b, preview_b, area_b, status_b, done_b, next);
        }
    });
}

// ── Panel construction ──────────────────────────────────────────────────────

/// Build the "Neural restore" sidebar section.
///
/// * `db_path` — the open catalogue; empty means demo mode (denoise still
///   writes the TIFF, but skips the catalogue import).
/// * `get_selected` — the current selected image path, read at Run time.
/// * `on_done` — grid reload after a successful import (the tether reload).
/// * `notify` — toast channel for outcomes.
pub fn neural_restore_box(
    db_path: String,
    get_selected: Rc<dyn Fn() -> Option<String>>,
    on_done: Rc<dyn Fn()>,
    notify: Rc<dyn Fn(String)>,
) -> gtk4::Box {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .build();

    // Section chrome matching the metadata panel's own sections
    // (panels/mod.rs `section_header` + separator).
    let header = gtk4::Label::builder()
        .label("Neural restore")
        .halign(gtk4::Align::Start)
        .margin_top(12)
        .margin_bottom(6)
        .margin_start(12)
        .margin_end(12)
        .build();
    header.add_css_class("heading");
    panel.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
    panel.append(&header);

    let hint = gtk4::Label::builder()
        .label("AI denoise for the selected image (experimental)")
        .halign(gtk4::Align::Start)
        .margin_start(12)
        .margin_end(12)
        .margin_bottom(6)
        .build();
    hint.add_css_class("dim-label");
    panel.append(&hint);

    // Task rows: Denoise is live; raw denoise / upscale are explicit
    // placeholders so the section says what is and is not wired. The task
    // toggle is an `adw::SwitchRow` because that is the one widget whose
    // `active`-property notify connect the tree already proves
    // (export_panel.rs `resize_row.connect_active_notify`); `gtk4::Switch`
    // has `is_active`/`set_active` but no notify hook in gtk4 0.9.7.
    let denoise_sw = adw::SwitchRow::builder()
        .title("Denoise")
        .active(true)
        .build();
    panel.append(&denoise_sw);

    for text in ["Raw denoise — coming next", "Upscale — coming next"] {
        let dead = gtk4::Label::builder()
            .label(text)
            .halign(gtk4::Align::Start)
            .margin_start(12)
            .margin_end(12)
            .build();
        dead.add_css_class("dim-label");
        dead.set_sensitive(false);
        panel.append(&dead);
    }

    // Model picker: downloaded models + download-on-demand entries.
    let model_row = adw::ComboRow::builder().title("Model").build();
    let model_margins = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    model_margins.set_margin_start(8);
    model_margins.set_margin_end(8);
    model_margins.append(&model_row);
    panel.append(&model_margins);

    // Status line + Run button row.
    let status_lbl = gtk4::Label::builder()
        .label("No model run yet")
        .halign(gtk4::Align::Start)
        .margin_start(12)
        .margin_end(12)
        .margin_top(2)
        .margin_bottom(2)
        .build();
    status_lbl.add_css_class("dim-label");
    panel.append(&status_lbl);

    let run_btn = gtk4::Button::builder()
        .label("Run denoise")
        .halign(gtk4::Align::Start)
        .margin_start(12)
        .margin_end(12)
        .margin_bottom(10)
        .build();
    panel.append(&run_btn);

    // Strength row (u7c): 0..100, default 100 = full denoise
    // (`recovery_alpha = 1 - strength/100`, neural_restore.c:885).
    // Insensitive until a run caches its buffers.
    let strength_lbl = gtk4::Label::builder()
        .label("Strength")
        .halign(gtk4::Align::Start)
        .margin_start(12)
        .build();
    strength_lbl.add_css_class("dim-label");
    panel.append(&strength_lbl);
    let strength_scale = gtk4::Scale::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .adjustment(&gtk4::Adjustment::new(100.0, 0.0, 100.0, 1.0, 10.0, 0.0))
        .draw_value(true)
        .hexpand(true)
        .margin_start(12)
        .margin_end(12)
        .margin_bottom(6)
        .sensitive(false)
        .build();
    panel.append(&strength_scale);

    // Before/after split preview (u7c): a DrawingArea painting the original
    // left of a draggable divider and the current blend right of it, with a
    // white divider line — the C split draw (neural_restore.c:3432-3466).
    // Hidden until the first run caches its pixbufs.
    let split_hint = gtk4::Label::builder()
        .label("Run denoise to preview detail recovery")
        .halign(gtk4::Align::Start)
        .margin_start(12)
        .margin_end(12)
        .margin_bottom(6)
        .build();
    split_hint.add_css_class("dim-label");
    panel.append(&split_hint);
    let split_area = gtk4::DrawingArea::builder()
        .hexpand(true)
        .height_request(220)
        .margin_start(12)
        .margin_end(12)
        .margin_bottom(10)
        .visible(false)
        .build();
    panel.append(&split_area);

    let state = Arc::new(Mutex::new(NeuralState::new()));
    let preview: Rc<RefCell<SplitPreview>> = Rc::new(RefCell::new(SplitPreview::new()));

    // Paint: dark surround, before-pixbuf clipped left of the divider,
    // after-pixbuf clipped right, white divider line. Mirrors the C's
    // before/after/divider sequence (:3434-3466).
    {
        let preview = preview.clone();
        split_area.set_draw_func(move |_, cr, w, h| {
            cr.set_source_rgb(0.19, 0.19, 0.19);
            let _ = cr.paint();
            let pv = preview.borrow();
            let (Some(before), Some(after)) = (pv.before.as_ref(), pv.after.as_ref()) else {
                return;
            };
            let (bw, bh) = (f64::from(before.width()), f64::from(before.height()));
            let (scale, ox, oy, div_x) =
                split_geometry(f64::from(w), f64::from(h), bw, bh, pv.split);
            if scale <= 0.0 {
                return;
            }
            let img_w = bw * scale;
            let img_h = bh * scale;
            // Before, left of the divider.
            let _ = cr.save();
            cr.rectangle(ox, oy, div_x - ox, img_h);
            cr.clip();
            cr.translate(ox, oy);
            cr.scale(scale, scale);
            cr.set_source_pixbuf(before, 0.0, 0.0);
            let _ = cr.paint();
            let _ = cr.restore();
            // After, right of the divider.
            let _ = cr.save();
            cr.rectangle(div_x, oy, ox + img_w - div_x, img_h);
            cr.clip();
            cr.translate(ox, oy);
            cr.scale(scale, scale);
            cr.set_source_pixbuf(after, 0.0, 0.0);
            let _ = cr.paint();
            let _ = cr.restore();
            // Divider line.
            cr.set_source_rgb(1.0, 1.0, 1.0);
            cr.set_line_width(1.5);
            cr.move_to(div_x, oy);
            cr.line_to(div_x, oy + img_h);
            let _ = cr.stroke();
        });
    }

    // Divider drag: press/drag anywhere moves the divider to the pointer,
    // clamped to the image (`CLAMP((ex - ox) / img_w, 0, 1)`,
    // neural_restore.c:3952). A repaint is all it takes — both pixbufs are
    // already cached, so no worker is involved. Shape mirrors the darkroom
    // snapshot-wipe divider (darkroom/mod.rs `set_wipe_from_x`).
    {
        let preview = preview.clone();
        let area_w = split_area.downgrade();
        let set_split_from_x = Rc::new(move |x: f64| {
            let Some(area) = area_w.upgrade() else { return };
            let (bw, bh, cur) = {
                let pv = preview.borrow();
                match pv.before.as_ref() {
                    Some(p) => (f64::from(p.width()), f64::from(p.height()), pv.split),
                    None => return,
                }
            };
            let (scale, ox, _, _) = split_geometry(
                f64::from(area.width()),
                f64::from(area.height()),
                bw,
                bh,
                cur,
            );
            if scale <= 0.0 {
                return;
            }
            preview.borrow_mut().split = split_pos_from_x(x, ox, bw * scale, cur);
            area.queue_draw();
        });
        let drag = gtk4::GestureDrag::new();
        let begin_set = set_split_from_x.clone();
        drag.connect_drag_begin(move |_, x, _| begin_set(x));
        drag.connect_drag_update(move |g, off_x, _| {
            if let Some((start_x, _)) = g.start_point() {
                set_split_from_x(start_x + off_x);
            }
        });
        split_area.add_controller(drag);
    }

    // Initial picker fill + Run gating (before any download).
    {
        let mut st = state.lock().unwrap();
        refresh_model_picker(&model_row, &mut st);
    }
    refresh_run_sensitive(&run_btn, &denoise_sw, &state.lock().unwrap());

    // Denoise switch gates the Run button.
    {
        let run_btn_w = run_btn.downgrade();
        let denoise_sw_w = denoise_sw.downgrade();
        let state = state.clone();
        denoise_sw.connect_active_notify(move |_| {
            let st = match state.lock() {
                Ok(v) => v,
                Err(_) => return,
            };
            let Some(run_btn) = run_btn_w.upgrade() else { return };
            let Some(denoise_sw) = denoise_sw_w.upgrade() else { return };
            refresh_run_sensitive(&run_btn, &denoise_sw, &st);
        });
    }

    // Model picker: selecting a download-on-demand row starts the download.
    {
        let state = state.clone();
        let status_w = status_lbl.downgrade();
        let run_btn_w = run_btn.downgrade();
        let denoise_sw_w = denoise_sw.downgrade();
        let combo_w = model_row.downgrade();
        let notify = notify.clone();
        model_row.connect_selected_notify(move |row| {
            let idx = row.selected();
            let (name, size_mi_b, already_busy) = {
                let Ok(st) = state.lock() else { return };
                match st.rows.get(idx as usize) {
                    Some(NeuralRow::Download { name, size_mi_b }) => {
                        (name.clone(), *size_mi_b, st.busy)
                    }
                    _ => return,
                }
            };
            if already_busy {
                return;
            }
            {
                let Ok(mut st) = state.lock() else { return };
                st.selected = idx;
                st.busy = true;
                st.run_phase = RunPhase::Idle;
            }
            // (Re-)arm the progress tick here, not just once at construction:
            // a tick started while Idle breaks immediately, so each operation
            // needs a live one of its own.
            arm_progress_tick(state.clone(), status_w.clone());
            if let (Some(run_btn), Some(denoise_sw)) =
                (run_btn_w.upgrade(), denoise_sw_w.upgrade())
            {
                let Ok(st) = state.lock() else { return };
                refresh_run_sensitive(&run_btn, &denoise_sw, &st);
            }
            // Freeze the picker for the download: a mid-download re-pick
            // would be overwritten by the completion handler below.
            // Re-enabled in `release`.
            if let Some(combo) = combo_w.upgrade() {
                combo.set_sensitive(false);
            }
            set_status_lbl(&status_w, "Contacting model registry…");

            // Per-activation shadows for everything the async subtree below
            // touches: the same-named outer bindings belong to this `Fn`
            // closure (every click reuses it), so moving them into a
            // single-shot future would be a move-out-of-`Fn` error. The
            // shadows are owned by this activation and move freely.
            let state = state.clone();
            let status_w = status_w.clone();
            let combo_w = combo_w.clone();
            let run_btn_w = run_btn_w.clone();
            let denoise_sw_w = denoise_sw_w.clone();
            let notify = notify.clone();
            // The display name travels independently: `known` is moved into
            // the worker below, but the continuation still needs a name to
            // report the outcome.
            let known_name = name.clone();
            let known = KnownModel { name, size_mi_b };
            glib::spawn_future_local(async move {
                // Worker-side clones: the blocking closure is `move` and the
                // continuation below reuses these bindings afterwards.
                let state_b = state.clone();
                let known_name_b = known_name.clone();
                let outcome = gio::spawn_blocking(move || {
                    let assets =
                        c41_core::ai::registry::fetch_release_assets(
                            c41_core::ai::registry::DEFAULT_MODEL_REPO,
                            c41_core::ai::registry::DEFAULT_RELEASE_TAG,
                        );
                    match assets {
                        Ok(assets) => {
                            let done = c41_core::ai::infer::ensure_denoise_model(
                                &known,
                                &assets,
                                |done, total| {
                                    if let Ok(mut st) = state_b.lock() {
                                        st.run_phase = RunPhase::Downloading { done, total };
                                    }
                                },
                            );
                            done.map(|p| p.display().to_string())
                        }
                        Err(e) => Err(c41_core::ai::infer::InferError::Registry(
                            format!("registry unavailable for {known_name_b:?}: {e}")
                        )),
                    }
                })
                .await;
                let release = |text: &str| {
                    {
                        let Ok(mut st) = state.lock() else { return };
                        st.busy = false;
                        st.run_phase = RunPhase::Idle;
                    }
                    set_status_lbl(&status_w, text);
                    if let Some(combo) = combo_w.upgrade() {
                        let selected_now = combo.selected();
                        {
                            let Ok(mut st) = state.lock() else { return };
                            refresh_model_picker(&combo, &mut st);
                        }
                        // Keep the just-downloaded model selected so Run is
                        // the immediate next step.
                        let next_idx = {
                            let Ok(mut st) = state.lock() else { return };
                            let idx = st
                                .rows
                                .iter()
                                .position(|r| {
                                    matches!(r, NeuralRow::Model { name: n } if n == &known_name)
                                })
                                .unwrap_or(selected_now as usize);
                            st.selected = idx as u32;
                            combo.set_selected(idx as u32);
                            idx
                        };
                        let _ = next_idx;
                        if let (Some(run_btn), Some(denoise_sw)) =
                            (run_btn_w.upgrade(), denoise_sw_w.upgrade())
                        {
                            let Ok(st) = state.lock() else { return };
                            refresh_run_sensitive(&run_btn, &denoise_sw, &st);
                        }
                        combo.set_sensitive(true);
                    }
                };
                match outcome {
                    Ok(Ok(_store_path)) => {
                        release(&format!("{known_name} ready"));
                        notify(format!("Downloaded {known_name}"));
                    }
                    Ok(Err(e)) => release(&e.to_string()),
                    Err(_) => release("Download did not complete"),
                }
            });
        });
    }

    // Progress ticks are armed per operation (download/run), not here: a tick
    // started while Idle breaks immediately, so construction-time arming
    // would be dead on arrival.

    // Run: decode → denoise → write TIFF → import → reload the grid.
    {
        let state = state.clone();
        let db = db_path.clone();
        let get = get_selected.clone();
        let done = on_done.clone();
        let notify = notify.clone();
        let status_w = status_lbl.downgrade();
        let run_btn_w = run_btn.downgrade();
        let denoise_sw_w = denoise_sw.downgrade();
        let combo_w = model_row.downgrade();
        let strength_w = strength_scale.downgrade();
        let area_w = split_area.downgrade();
        let hint_w = split_hint.downgrade();
        let preview_run = preview.clone();
        run_btn.connect_clicked(move |btn| {
            // Busy guard: one run/download at a time (double-click protection).
            // Poisoned mutex (a worker panicked holding it) degrades to a
            // no-op click rather than a GTK-thread panic, here and everywhere
            // below in this handler.
            let Ok(busy) = state.lock().map(|st| st.busy) else {
                return;
            };
            if busy {
                return;
            }
            let Some(path) = get() else {
                notify("Select an image first".into());
                return;
            };
            let model_name = {
                let Ok(st) = state.lock() else {
                    notify("Neural restore is unavailable right now".into());
                    return;
                };
                match st.rows.get(st.selected as usize) {
                    Some(NeuralRow::Model { name }) => name.clone(),
                    Some(NeuralRow::Download { .. }) => {
                        notify("Download the model first".into());
                        return;
                    }
                    _ => {
                        notify("No denoise model downloaded".into());
                        return;
                    }
                }
            };
            let db_empty = db.is_empty();
            {
                let Ok(mut st) = state.lock() else {
                    return;
                };
                st.busy = true;
                st.run_phase = RunPhase::Idle;
                // A newer run invalidates any re-blend still in flight: its
                // completion must not repaint the preview it no longer owns.
                st.reblend_seq += 1;
            }
            // Arm the progress tick for this run (see the download path:
            // construction-time arming would already have broken).
            arm_progress_tick(state.clone(), status_w.clone());
            if let Some(denoise_sw) = denoise_sw_w.upgrade() {
                let Ok(st) = state.lock() else {
                    return;
                };
                refresh_run_sensitive(btn, &denoise_sw, &st);
            }
            if let Some(combo) = combo_w.upgrade() {
                combo.set_sensitive(false);
            }
            // The Strength slider only blends the cached run; freeze it while
            // a new run (and its cache) is being built. Re-enabled in
            // `release` when a cache exists.
            if let Some(strength) = strength_w.upgrade() {
                strength.set_sensitive(false);
            }
            set_status_lbl(&status_w, "Preparing…");

            // Per-activation shadows for everything the async subtree below
            // touches (worker + continuation): the same-named outer bindings
            // belong to this `Fn` closure (every click reuses it).
            let state = state.clone();
            let db = db.clone();
            let status_w = status_w.clone();
            let run_btn_w = run_btn_w.clone();
            let denoise_sw_w = denoise_sw_w.clone();
            let combo_w = combo_w.clone();
            let strength_w = strength_w.clone();
            let area_w = area_w.clone();
            let hint_w = hint_w.clone();
            let preview_run = preview_run.clone();
            let strength_scale_set = strength_w.clone();
            let notify = notify.clone();
            let done = done.clone();
            glib::spawn_future_local(async move {
                // Worker-side clone, same reason as the download handler:
                // the continuation below reuses `state`.
                let state_b = state.clone();
                let outcome = gio::spawn_blocking(move || {
                    run_denoise_worker(&path, &db, &model_name, |d, total| {
                        if let Ok(mut st) = state_b.lock() {
                            st.run_phase = RunPhase::Denoising { done: d, total };
                        }
                    })
                })
                .await;
                let release = |text: &str| {
                    {
                        let Ok(mut st) = state.lock() else { return };
                        st.busy = false;
                        st.run_phase = RunPhase::Idle;
                    }
                    set_status_lbl(&status_w, text);
                    if let Some(run_btn) = run_btn_w.upgrade() {
                        if let Some(denoise_sw) = denoise_sw_w.upgrade() {
                            let Ok(st) = state.lock() else { return };
                            refresh_run_sensitive(&run_btn, &denoise_sw, &st);
                        }
                    }
                    if let Some(combo) = combo_w.upgrade() {
                        combo.set_sensitive(true);
                    }
                    // The Strength slider lives off the blend cache: it
                    // stays usable across failures when an older run's
                    // cache (and TIFF) is still valid.
                    if let Some(strength) = strength_w.upgrade() {
                        let Ok(st) = state.lock() else { return };
                        strength.set_sensitive(st.blend.is_some());
                    }
                };
                match outcome {
                    Ok(Ok((report, cache))) => {
                        // Install the u7c blend cache + split pixbufs: the
                        // "after" side at strength 100 is the denoised frame
                        // itself (alpha 0 → bit-exact, no DWT needed).
                        let (tw, th) = thumbnail_dims(cache.w, cache.h, PREVIEW_MAX_DIM);
                        let before_bytes = shrink_rgba_to_srgb8(
                            &cache.original,
                            cache.w,
                            cache.h,
                            tw,
                            th,
                        );
                        let after_bytes = shrink_rgb_to_srgb8(
                            &cache.denoised,
                            cache.w,
                            cache.h,
                            tw,
                            th,
                        );
                        {
                            let Ok(mut st) = state.lock() else { return };
                            st.blend = Some(cache);
                            st.applied_strength = 100.0;
                            st.pending_strength = None;
                        }
                        {
                            let mut pv = preview_run.borrow_mut();
                            if let Some(pb) = pixbuf_from_srgb8(before_bytes, tw, th) {
                                pv.before = Some(pb);
                            }
                            if let Some(pb) = pixbuf_from_srgb8(after_bytes, tw, th) {
                                pv.after = Some(pb);
                            }
                            pv.split = 0.5;
                        }
                        if let Some(area) = area_w.upgrade() {
                            area.set_visible(true);
                            area.queue_draw();
                        }
                        if let Some(hint) = hint_w.upgrade() {
                            hint.set_visible(false);
                        }
                        // Reset to full denoise WITHOUT re-blending: the
                        // value handler early-returns when the value equals
                        // the applied strength (set above first).
                        if let Some(scale) = strength_scale_set.upgrade() {
                            scale.set_value(100.0);
                            scale.set_sensitive(true);
                        }
                        let msg = if report.imported {
                            "Denoised — written and imported"
                        } else if db_empty {
                            "Denoised — written (no catalogue open)"
                        } else {
                            "Denoised — written, not imported"
                        };
                        release(msg);
                        notify(format!("Neural denoise: {}", report.out_path));
                        if report.imported {
                            done();
                        }
                    }
                    Ok(Err(e)) => {
                        release(&format!("Denoise failed: {e}"));
                        // A value parked while the failed run was in flight
                        // still wants applying against the surviving
                        // (previous) cache — otherwise slider and applied
                        // strength disagree silently until the next drag.
                        // kick_reblend re-checks busy/cache itself.
                        let next = state
                            .lock()
                            .ok()
                            .and_then(|mut st| st.pending_strength.take());
                        if let Some(v) = next {
                            kick_reblend(
                                state.clone(),
                                preview_run.clone(),
                                area_w.clone(),
                                status_w.clone(),
                                done.clone(),
                                v,
                            );
                        }
                    }
                    Err(_) => {
                        release("Denoise did not complete");
                        let next = state
                            .lock()
                            .ok()
                            .and_then(|mut st| st.pending_strength.take());
                        if let Some(v) = next {
                            kick_reblend(
                                state.clone(),
                                preview_run.clone(),
                                area_w.clone(),
                                status_w.clone(),
                                done.clone(),
                                v,
                            );
                        }
                    }
                }
            });
        });
    }

    // Strength slider (u7c): debounced re-blend → TIFF overwrite → repaint.
    // The handler only parks the value; the STRENGTH_DEBOUNCE timer fires
    // the single worker pass, so a drag issues one pass per pause, not one
    // per pixel of travel. No-op when the value equals the applied strength
    // (covers the programmatic reset after each run).
    {
        let state = state.clone();
        let preview = preview.clone();
        let area_w = split_area.downgrade();
        let status_w = status_lbl.downgrade();
        let done = on_done.clone();
        let debounce: Rc<RefCell<Option<glib::SourceId>>> = Rc::new(RefCell::new(None));
        strength_scale.connect_value_changed(move |scale| {
            let v = scale.value() as f32;
            {
                let Ok(mut st) = state.lock() else { return };
                if st.blend.is_none() {
                    return;
                }
                if (v - st.applied_strength).abs() < f32::EPSILON
                    && st.pending_strength.is_none()
                {
                    return;
                }
                st.pending_strength = Some(v);
            }
            if let Some(id) = debounce.borrow_mut().take() {
                id.remove();
            }
            // Per-activation shadows for the single-shot timer below: the
            // outer `Fn` closure owns the same-named bindings.
            let state = state.clone();
            let preview = preview.clone();
            let area_w = area_w.clone();
            let status_w = status_w.clone();
            let done = done.clone();
            let d_deb = debounce.clone();
            let id = glib::timeout_add_local_once(STRENGTH_DEBOUNCE, move || {
                *d_deb.borrow_mut() = None;
                let next = {
                    let Ok(mut st) = state.lock() else { return };
                    st.pending_strength.take().unwrap_or(v)
                };
                kick_reblend(state, preview, area_w, status_w, done, next);
            });
            *debounce.borrow_mut() = Some(id);
        });
    }

    panel
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniqueness for test temp dirs (c41-core house pattern).
    static TEST_NONCE: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

    fn fixture_known() -> Vec<KnownModel> {
        known_denoise_models()
    }

    #[test]
    fn model_rows_merge_downloaded_first_then_known() {
        let known = fixture_known();
        let downloaded = vec!["denoise-nind.dtmodel".to_string()];
        let rows = model_rows(&downloaded, &known);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], NeuralRow::Model { name: "denoise-nind.dtmodel".to_string() });
        assert_eq!(
            rows[1],
            NeuralRow::Download { name: "denoise-nafnet.dtmodel".to_string(), size_mi_b: 108 }
        );
        assert!(row_is_runnable(&rows[0]));
        assert!(!row_is_runnable(&rows[1]));
    }

    #[test]
    fn model_rows_all_downloaded_leaves_only_models() {
        let known = fixture_known();
        let downloaded: Vec<String> = known.iter().map(|k| k.name.clone()).collect();
        let rows = model_rows(&downloaded, &known);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(row_is_runnable));
    }

    #[test]
    fn model_rows_empty_store_degrades_to_placeholder() {
        let rows = model_rows(&[], &[]);
        assert_eq!(rows, vec![NeuralRow::Empty]);
        assert!(!row_is_runnable(&rows[0]));
    }

    #[test]
    fn row_labels_name_the_action() {
        assert_eq!(row_label(&NeuralRow::Empty), EMPTY_ROWS_TEXT.to_string());
        assert_eq!(
            row_label(&NeuralRow::Download { name: "m".to_string(), size_mi_b: 55 }),
            "Download m (55 MB)"
        );
        assert_eq!(
            row_label(&NeuralRow::Model { name: "m.dtmodel".to_string() }),
            "m.dtmodel"
        );
    }

    #[test]
    fn progress_text_formats_tiles_and_megabytes() {
        assert_eq!(format_run_progress(3, 12), "Denoising 3/12…");
        // 12.5 MiB = 13107200 bytes.
        assert_eq!(format_download_progress(12_582_912, Some(55 * 1024 * 1024)),
            "Downloading — 12.0 of 55.0 MB");
        let no_total = format_download_progress(12_582_912, None);
        assert!(no_total.starts_with("Downloading — 12.0 MB"), "{no_total}");
    }

    #[test]
    fn output_stem_beside_source_with_denoise_suffix() {
        assert_eq!(
            denoise_output_stem("/photos/a/IMG_1.CR2"),
            "/photos/a/IMG_1_denoise"
        );
        assert_eq!(denoise_output_stem("nopath.raw"), "./nopath_denoise");
        // Re-processing does NOT double the suffix (the C has_suffix skip).
        assert_eq!(
            denoise_output_stem("/p/x_denoise.tif"),
            "/p/x_denoise"
        );
    }

    #[test]
    fn unique_output_path_skips_existing_files() {
        let dir = std::env::temp_dir().join(format!(
            "c41_neural_uniq_{}_{}",
            std::process::id(),
            TEST_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("img_denoise").to_string_lossy().to_string();
        assert_eq!(unique_output_path(&base), Some(format!("{base}.tif")));
        std::fs::write(format!("{base}.tif"), b"x").unwrap();
        assert_eq!(unique_output_path(&base), Some(format!("{base}_1.tif")));
        std::fs::write(format!("{base}_1.tif"), b"x").unwrap();
        assert_eq!(unique_output_path(&base), Some(format!("{base}_2.tif")));
        // The source stem itself makes a good suffix roundtrip: the `.tif`
        // extension is stripped by file_stem, and the existing suffix is
        // not doubled.
        let from_run = denoise_output_stem(&format!("{base}.tif"));
        assert!(from_run.ends_with("img_denoise"), "{from_run}");
        assert!(!from_run.ends_with("img_denoise_denoise"), "{from_run}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tick_interval_is_sane_and_constant() {
        // Documents the live-progress cadence for the panel contract.
        assert_eq!(TICK_INTERVAL.as_millis(), 200);
    }

    #[test]
    fn nonraw_linearization_matches_srgb_decode() {
        // A gamma-encoded ~0.5 value decodes to the classic linear 0.214.
        let v16 = (0.5 * 65535.0) as u16;
        let lin = c41_core::color::srgb_to_linear(v16 as f32 / 65535.0);
        assert!((lin - 0.21404).abs() < 1e-3);
    }

    // ── u7c: strength mapping, split geometry, preview bytes ───────────────

    #[test]
    fn strength_to_alpha_matches_the_c_recovery_slider() {
        // recovery_alpha = 1 - strength/100 (neural_restore.c:885): 100 is
        // full denoise, 0 is source-like. The panel drives
        // `apply_detail_recovery` through this, never re-inferring.
        assert_eq!(strength_to_alpha(100.0), 0.0);
        assert_eq!(strength_to_alpha(0.0), 1.0);
        assert!((strength_to_alpha(80.0) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn strength_debounce_matches_the_c_intent() {
        // The C debounces strength re-blends to 50 ms
        // (RAW_PREVIEW_STRENGTH_DEBOUNCE_MS, neural_restore.c:2295).
        assert_eq!(STRENGTH_DEBOUNCE.as_millis(), 50);
    }

    #[test]
    fn strength_done_line_names_the_value() {
        assert_eq!(format_strength_done(80.0), "Strength 80% — saved");
        assert_eq!(format_strength_done(100.0), "Strength 100% — saved");
    }

    #[test]
    fn split_clamps_to_the_image() {
        assert_eq!(clamp_split_pos(0.5), 0.5);
        assert_eq!(clamp_split_pos(-0.2), 0.0);
        assert_eq!(clamp_split_pos(1.4), 1.0);
        // NaN falls back to the C's initial split_pos 0.5 (:4129).
        assert_eq!(clamp_split_pos(f64::NAN), 0.5);
    }

    #[test]
    fn split_geometry_centers_and_clamps() {
        // 400x200 pixbuf in a 400x200 allocation: scale 1, no offsets,
        // divider at 200 for split 0.5.
        let (s, ox, oy, dx) = split_geometry(400.0, 200.0, 400.0, 200.0, 0.5);
        assert_eq!((s, ox, oy, dx), (1.0, 0.0, 0.0, 200.0));
        // Split 0/1 pin the divider to the image edges.
        assert_eq!(split_geometry(400.0, 200.0, 400.0, 200.0, 0.0).3, 0.0);
        assert_eq!(split_geometry(400.0, 200.0, 400.0, 200.0, 1.0).3, 400.0);
        // Letterboxed: 400x200 pixbuf in a 400x400 allocation scales by 1
        // and centers vertically; the divider still tracks the image.
        let (s, ox, oy, dx) = split_geometry(400.0, 400.0, 400.0, 200.0, 0.5);
        assert_eq!((s, ox, oy, dx), (1.0, 0.0, 100.0, 200.0));
        // Downscale contain-fit: 800x400 pixbuf in 400x200 → scale 0.5.
        let (s, ox, oy, dx) = split_geometry(400.0, 200.0, 800.0, 400.0, 0.25);
        assert_eq!((s, ox, oy, dx), (0.5, 0.0, 0.0, 100.0));
        // Degenerate inputs yield zeros, never NaN or infinities.
        for g in [
            split_geometry(0.0, 200.0, 400.0, 200.0, 0.5),
            split_geometry(400.0, 0.0, 400.0, 200.0, 0.5),
            split_geometry(400.0, 200.0, 0.0, 200.0, 0.5),
            split_geometry(400.0, 200.0, 400.0, 0.0, 0.5),
        ] {
            assert_eq!(g, (0.0, 0.0, 0.0, 0.0));
            assert!(g.0.is_finite() && g.3.is_finite());
        }
    }

    #[test]
    fn split_drag_maps_pointer_to_divider() {
        // Mirror of neural_restore.c:3952: CLAMP((ex - ox) / img_w, 0, 1).
        assert_eq!(split_pos_from_x(200.0, 0.0, 400.0, 0.5), 0.5);
        assert_eq!(split_pos_from_x(-50.0, 0.0, 400.0, 0.5), 0.0);
        assert_eq!(split_pos_from_x(500.0, 0.0, 400.0, 0.5), 1.0);
        // Letterbox offset shifts the mapping.
        assert_eq!(split_pos_from_x(100.0, 100.0, 400.0, 0.5), 0.0);
        assert_eq!(split_pos_from_x(300.0, 100.0, 400.0, 0.5), 0.5);
        // Degenerate width keeps the current position (no divide by zero).
        assert_eq!(split_pos_from_x(200.0, 0.0, 0.0, 0.3), 0.3);
    }

    #[test]
    fn thumbnail_dims_cap_the_longest_edge() {
        assert_eq!(thumbnail_dims(6000, 4000, 480), (480, 320));
        assert_eq!(thumbnail_dims(4000, 6000, 480), (320, 480));
        // Small images pass through untouched (no upscaling).
        assert_eq!(thumbnail_dims(320, 200, 480), (320, 200));
        assert_eq!(thumbnail_dims(480, 480, 480), (480, 480));
    }

    #[test]
    fn srgb8_encode_pins_endpoints_and_mid() {
        assert_eq!(linear_to_srgb8(0.0), 0);
        assert_eq!(linear_to_srgb8(1.0), 255);
        assert_eq!(linear_to_srgb8(-0.5), 0, "negatives clamp");
        assert_eq!(linear_to_srgb8(2.0), 255, "overshoot clamps");
        // linear 0.5 encodes to sRGB ~0.7354 → 188.
        assert_eq!(linear_to_srgb8(0.5), 188);
    }

    #[test]
    fn shrink_nearest_picks_the_right_texels() {
        // 4x2 RGB frame down to 2x1: nearest picks source columns 0 and 2
        // of row 0 (dy = 0*2/1 = 0, dx = x*4/2).
        let mut src = vec![0.0f32; 4 * 2 * 3];
        for i in 0..8 {
            src[i * 3] = i as f32 / 8.0;
            src[i * 3 + 1] = 0.0;
            src[i * 3 + 2] = 1.0;
        }
        let out = shrink_rgb_to_srgb8(&src, 4, 2, 2, 1);
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], linear_to_srgb8(0.0), "texel (0,0)");
        assert_eq!(out[3], linear_to_srgb8(2.0 / 8.0), "texel (2,0)");
        assert_eq!(out[2], 255, "blue lane encodes");
        // Length mismatch degrades to black, never panics.
        assert_eq!(shrink_rgb_to_srgb8(&[0.5f32; 5], 4, 2, 2, 1), vec![0u8; 6]);
    }

    #[test]
    fn shrink_rgba_composites_over_white() {
        // Transparent black over white reads as white; opaque red stays red.
        let src = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 1.0,
        ];
        let out = shrink_rgba_to_srgb8(&src, 2, 1, 2, 1);
        assert_eq!(&out[0..3], &[255, 255, 255]);
        assert_eq!(&out[3..6], &[255, 0, 0]);
    }

    #[test]
    fn quantize_pins_endpoints_and_rounding() {
        let q = quantize_srgb16(&[0.0, 1.0, 0.5, -0.2, 1.5]);
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 65535);
        let mid = (c41_core::color::linear_to_srgb(0.5).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        assert_eq!(q[2], mid);
        assert_eq!(q[3], 0, "negatives clamp");
        assert_eq!(q[4], 65535, "overshoot clamps");
    }

    #[test]
    fn blend_cache_shares_buffers_by_pointer() {
        // The re-blend contract: kicks clone the cache, not the pixels.
        // `Arc::ptr_eq` pins that installing one cache and kicking twice
        // never duplicates the full-frame buffers.
        let cache = BlendCache {
            original: std::sync::Arc::new(vec![0.1f32; 16]),
            denoised: std::sync::Arc::new(vec![0.2f32; 12]),
            w: 2,
            h: 2,
            out_path: "/tmp/x_denoise.tif".to_string(),
        };
        let again = cache.clone();
        assert!(std::sync::Arc::ptr_eq(&cache.original, &again.original));
        assert!(std::sync::Arc::ptr_eq(&cache.denoised, &again.denoised));
    }
}
