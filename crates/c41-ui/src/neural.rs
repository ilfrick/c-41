//! Neural restore panel (u7b): the "Neural restore" sidebar section that runs
//! the RGB denoise task on the selected image.
//!
//! darktable's neural-restore lib (src/libs/neural_restore.c) drives the AI
//! denoise/raw-denoise/upscale tasks from the lighttable. This panel is the
//! u7b slice of that: **Denoise** only. Raw denoise and upscale are shown but
//! greyed ("coming next" — they are separate tasks with their own CFA-style
//! preprocessing and tile ladders, recorded as u7c+ work), and the detail
//! slider / before-after split of the C lib are not part of this increment.
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
use c41_core::ai::infer::{known_denoise_models, KnownModel};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

/// How often the progress tick repaints the status line while a run or a
/// download is in flight.
pub const TICK_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);
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

/// The whole denoise run for one image: decode → prepare the model →
/// [`c41_core::ai::infer::denoise_rgb`] (tiled ORT inference, progress
/// forwarded) → quantise the packed linear result to u16 → write the result
/// TIFF beside the source through the export writer → register it with the
/// catalogue. Runs entirely on a blocking worker; GTK is never touched here.
fn run_denoise_worker(
    path: &str,
    db: &str,
    model_name: &str,
    progress: impl FnMut(u32, u32),
) -> Result<RunReport, String> {
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
    let mut rgb16 = vec![0u16; out_rgb.len()];
    for (i, v) in out_rgb.iter().enumerate() {
        // sRGB-OETF encode + round-half-up quantize, exactly like the export
        // TIFF path (`render_linear_to_srgb16_gear` → `srgb_encode_rgb` then
        // `(enc * 65535 + 0.5) as u16`, preview.rs): TIFF consumers read the
        // bytes as sRGB, so scene-linear values would display dark.
        let enc = c41_core::color::linear_to_srgb(*v);
        rgb16[i] = (enc.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
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
    Ok(RunReport { out_path: dest, imported })
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
struct NeuralState {
    rows: Vec<NeuralRow>,
    selected: u32,
    busy: bool,
    run_phase: RunPhase,
}

impl NeuralState {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            selected: 0,
            busy: false,
            run_phase: RunPhase::Idle,
        }
    }
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

    let state = Arc::new(Mutex::new(NeuralState::new()));

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
                };
                match outcome {
                    Ok(Ok(report)) => {
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
                    Ok(Err(e)) => release(&format!("Denoise failed: {e}")),
                    Err(_) => release("Denoise did not complete"),
                }
            });
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
}
