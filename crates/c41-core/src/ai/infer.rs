//! ONNX inference for the RGB denoise task (u7b).
//!
//! This is the inference half of the u7a infrastructure: where `registry`,
//! `download` and `package` find, verify and unpack a model, this module
//! builds an ONNX Runtime session on the extracted payload and runs the
//! scale-1 (denoise) tiler. The essential semantics of
//! `src/common/ai/restore_rgb.c` are reproduced exactly for the
//! no-working-profile path, minus what u7b deliberately defers:
//!
//! * working-profile color matrices (`ctx->has_profile` → the wp_to_srgb /
//!   srgb_to_wp 3x3 multiply): the model sees the values as sRGB, the same
//!   approximation restore_rgb.c makes when no profile is set
//!   (`restore_rgb.c:283-287`).
//! * shadow boost, gamut pass-through and detail recovery: model
//!   attributes u7b does not implement (`restore_rgb.c:255-269`,
//!   `:357-427`, `:730+`).
//! * upscale (`scale > 1`): the tiler here is scale-1 only; raw denoise is
//!   a separate task with its own CFA preprocessing.
//!
//! What IS mirrored (and tested headless) is the load-bearing shape:
//! sRGB gamma-encode + clamp on the way in (`restore_rgb.c:283-287`),
//! one NCHW `{1,3,T,T}` planar float tensor per tile (`:296-303`), the
//! mirror-padded tile grid with overlap O=64 / step=T-2O (`restore.c:36`,
//! `restore_rgb.c:549-572`), the valid-region strip offset by O from each
//! tile edge (`:681-702`), and inverse gamma on the way out (`:453-466`).
//!
//! ## Session lifecycle (every ort call, for gate verification)
//!
//! * `ort::session::Session::builder()` — zero-arg. This is the ONLY ort
//!   entry the library crate needs; the environment is created lazily by
//!   `ort::environment::Environment::current()` (read in the vendored
//!   ort 2.0.0-rc.13 sources, `environment.rs:141-164`) with the default
//!   provider set — CPU only, since u7a links `ort` with default features
//!   and no CUDA/TensorRT/etc provider is compiled in. ort's own module
//!   doc forbids a library crate from calling `ort::init()` /
//!   `ort::init_from()` ("Authors of libraries using ort should never have
//!   the library configure the environment itself"), so this crate never
//!   does — the application owns the environment.
//! * `SessionBuilder::commit_from_file(path)` — parses the ONNX graph and
//!   creates the session (`builder/impl_commit.rs:45-73`).
//! * `Session::inputs().len()` — read-only input-count probe, the same
//!   purpose as the C `dt_ai_get_input_count` (`restore_rgb.c:289`), so
//!   multi-input models still get their constant noise map
//!   (`restore_rgb.c:305-325`).
//! * `Tensor::from_array((shape, data))` — owned float tensor from a
//!   `(shape, Vec<f32>)` pair for the 25/255 noise map
//!   (`value/impl_tensor/create.rs:139-143`).
//! * `TensorRef::from_array_view((shape, &data))` — a borrowed view over
//!   the tile's planar buffer; the run below is synchronous so the tile
//!   stays alive for the whole call (`value/impl_tensor/create.rs:214-221`).
//! * `Session::run(ort::inputs![...])` — the inference call
//!   (`session/mod.rs:236-248`). Takes `&mut self`: ONNX Runtime's `Run` is
//!   not thread-safe, which is why u7b builds one session per call and
//!   never shares it across threads.
//! * `SessionOutputs` index (`outputs[0]`) + `try_extract_tensor::<f32>()`
//!   — reads the first output as a live `(&Shape, &[f32])` view
//!   (`session/output.rs:215-223`, `value/impl_tensor/extract.rs:155-158`).
//!
//! A session is built per `denoise_rgb` call and dropped when the call
//! returns. Caching was considered and deferred: a cached session would
//! need a mutex (calls serialize on the UI's busy guard today, but a
//! shared `&mut` object invites a future foot-gun), and the per-call cost
//! is one 55-128 MB graph parse per Run button press — acceptable for a
//! single-image operation and crash-free by construction (no global state
//! to corrupt). A memoized `(path, mtime) → Session` cache is a clean
//! follow-up.
//!
//! ## Model handling
//!
//! `ensure_denoise_model` downloads or verifies one known `.dtmodel`
//! archive through the u7a downloader (sha256-gated, atomic store);
//! `prepare_denoise_model` unpacks it on demand and resolves the payload +
//! manifest tile size exactly like `restore.c::_resolve_tile_size`
//! (`package::variant_tile_size("model")` falls back to the top-level
//! `input_sizes`, mirroring the C NULL-stem default payload `model.onnx`).
//!
//! Nothing in this module reaches github.com: tests exercise the pure tile
//! math with a fake tile-runner, and the model helpers against synthetic
//! `.dtmodel` fixtures — there are no model downloads and no network in
//! the test suite.

use std::path::{Path, PathBuf};

use super::download;
use super::package;
use super::registry;

/// Overlap in px for scale-1 (denoise) tiling: `restore.c` `OVERLAP_DENOISE`.
pub const O_DENOISE: u32 = 64;
/// The payload filename inside a single-model package: the C `_load` passes
/// a NULL stem for the denoise task, which `dt_ai_onnx_load_ext` resolves
/// to `model.onnx` (`backend_onnx.c:1813`).
pub const MODEL_ONNX_FILE: &str = "model.onnx";
/// Known RGB-denoise release asset names, for the download-on-demand list.
pub const DENOISE_NIND_ASSET: &str = "denoise-nind.dtmodel";
pub const DENOISE_NAFNET_ASSET: &str = "denoise-nafnet.dtmodel";

/// Inference / model-preparation failures. Every foreign error shape is
/// collapsed to a message string so callers never match on library types.
#[derive(Debug, PartialEq, Eq)]
pub enum InferError {
    /// Filesystem failure out of our control (store unreadable, ...).
    Io(String),
    /// The u7a downloader refused (no digest, checksum mismatch, network).
    Download(String),
    /// The model package is unreadable / missing the declared payload.
    Package(String),
    /// The release listing does not carry what we asked for.
    Registry(String),
    /// An ONNX Runtime call failed (message copied verbatim).
    Ort(String),
    /// The requested model archive is not in the store.
    MissingModel(String),
    /// A caller-provided argument is wrong (bad tensor shape, ...).
    InvalidArgument(String),
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "neural restore IO failed: {s}"),
            Self::Download(s) => write!(f, "{s}"),
            Self::Package(s) => write!(f, "{s}"),
            Self::Registry(s) => write!(f, "model registry: {s}"),
            Self::Ort(s) => write!(f, "ONNX inference failed: {s}"),
            Self::MissingModel(name) => write!(f, "model {name:?} is not downloaded"),
            Self::InvalidArgument(s) => write!(f, "invalid denoise request: {s}"),
        }
    }
}

impl std::error::Error for InferError {}

// ── Gamma (restore_rgb.c:79-93 semantics) ───────────────────────────────────

/// The model-input encode: linear working-space value → gamma-encoded sRGB,
/// clamped to [0,1] FIRST exactly like the C (`restore_rgb.c:283-287` ships
/// `_linear_to_srgb(fminf(v, 1.0f))`, and `_linear_to_srgb` returns 0 for
/// `v <= 0`). The clamp makes the reuse of [`crate::color::linear_to_srgb`]
/// (color.rs:332, the standard IEC 61966-2-1 curve) equal to the C on every
/// input: negatives hit 0 and overshoot clamps to 1 before the same curve.
#[inline]
pub fn model_linear_to_srgb(v: f32) -> f32 {
    crate::color::linear_to_srgb(v.clamp(0.0, 1.0))
}

/// The model-output decode: gamma-encoded sRGB → linear working-space
/// value, clamped at 0 exactly like the C (`_srgb_to_linear` returns 0 for
/// `v <= 0`, restore_rgb.c:87-93). Pairs with [`model_linear_to_srgb`];
/// for inputs already in [0,1] the pair is the identity (see the roundtrip
/// test).
#[inline]
pub fn model_srgb_to_linear(v: f32) -> f32 {
    crate::color::srgb_to_linear(v.max(0.0))
}

// ── Tile math (restore_rgb.c:549-702, restore_common.h:201-210) ────────────

/// Periodic mirror-pad index reflection, ported from `restore_common.h:201`
/// `_mirror`: any integer index maps into [0, n). `n <= 1` mirrors to 0.
pub fn mirror_coord(mut i: i64, n: i64) -> i64 {
    if n <= 1 {
        return 0;
    }
    if i < 0 {
        i = -i;
    }
    let period = 2 * (n - 1);
    let r = i % period;
    if r >= n {
        period - r
    } else {
        r
    }
}

/// One tile of the grid. `in_x`/`in_y` are the INPUT-space origin of the
/// T×T sample window (negative at the top/left edge — the gather mirrors);
/// `out_x`/`out_y` plus `valid_w`/`valid_h` are the OUTPUT-space rectangle
/// this tile owns, i.e. the O-offset strip clamped to the image bounds —
/// the same arithmetic as `restore_rgb.c:588-612` and `:681-702`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSpec {
    pub in_x: i64,
    pub in_y: i64,
    pub out_x: i64,
    pub out_y: i64,
    pub valid_w: u32,
    pub valid_h: u32,
    /// True when any part of the T×T window falls outside the image, so
    /// the gather must mirror-pad. When false the C takes a direct row copy
    /// path; the mirrored gather below is identical for in-bounds indices.
    pub needs_mirror: bool,
}

/// The full tiling of one image: grid dims, step, and the row-major tile
/// list (`ty` outer, `tx` inner — the C loop order, restore_rgb.c:588-708).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileGrid {
    pub cols: u32,
    pub rows: u32,
    pub step: u32,
    pub tiles: Vec<TileSpec>,
}

/// Partition `w`×`h` into T×T tiles with overlap `o` and step `T-2o` — the
/// exact grid of `restore_rgb.c:549-572`. Errors on an empty image or a tile
/// size that cannot overlap (`T <= 2o` would make `step <= 0`).
pub fn tile_grid(w: u32, h: u32, t: u32, o: u32) -> Result<TileGrid, InferError> {
    if w == 0 || h == 0 {
        return Err(InferError::InvalidArgument("cannot tile an empty image".to_string()));
    }
    if t <= 2 * o {
        return Err(InferError::InvalidArgument(format!(
            "tile size {t} must exceed 2*overlap {}",
            2 * o
        )));
    }
    let step = t - 2 * o;
    let cols = w.div_ceil(step);
    let rows = h.div_ceil(step);
    let mut tiles = Vec::with_capacity(cols as usize * rows as usize);
    let (wt, ht, tt, ot) = (w as i64, h as i64, t as i64, o as i64);
    for ty in 0..rows {
        let y = ty as i64 * step as i64;
        let valid_h = if y + step as i64 > ht { (ht - y) as u32 } else { step };
        for tx in 0..cols {
            let x = tx as i64 * step as i64;
            let valid_w = if x + step as i64 > wt { (wt - x) as u32 } else { step };
            let needs_mirror = x - ot < 0
                || y - ot < 0
                || x - ot + tt > wt
                || y - ot + tt > ht;
            tiles.push(TileSpec {
                in_x: x - ot,
                in_y: y - ot,
                out_x: x,
                out_y: y,
                valid_w,
                valid_h,
                needs_mirror,
            });
        }
    }
    Ok(TileGrid { cols, rows, step, tiles })
}

/// Gather one T×T tile from the interleaved RGBA `input` (w*h*4 f32) into a
/// planar NCHW `{1,3,T,T}` f32 buffer, applying the model-input encode
/// (clamp + gamma) per sample — the gather (restore_rgb.c:617-656) fused
/// with the pre-gamma of `run_patch` (`:283-287`). The mirror gather is
/// unconditional: for in-bounds coordinates `mirror_coord` is the identity,
/// so this equals the C fast path exactly.
pub fn sample_tile_planar(
    input: &[f32],
    w: u32,
    h: u32,
    spec: &TileSpec,
    t: usize,
) -> Vec<f32> {
    let plane = t * t;
    let mut planar = vec![0f32; 3 * plane];
    let (wt, ht, tt) = (w as i64, h as i64, t as i64);
    for dy in 0..tt {
        let sy = mirror_coord(spec.in_y + dy, ht);
        let row = sy * wt;
        for dx in 0..tt {
            let sx = mirror_coord(spec.in_x + dx, wt);
            let si = ((row + sx) as usize) * 4;
            let do_ = dy as usize * t + dx as usize;
            planar[do_] = model_linear_to_srgb(input[si]);
            planar[do_ + plane] = model_linear_to_srgb(input[si + 1]);
            planar[do_ + 2 * plane] = model_linear_to_srgb(input[si + 2]);
        }
    }
    planar
}

/// Inverse of the strip seam: copy this tile's valid O-offset strip out of
/// its planar `{1,3,T,T}` model output into the packed RGB output buffer
/// (w*h*3 f32), applying the model-output decode per sample. Ported from
/// `restore_rgb.c:681-702` (scale 1, so `O_out == O`).
fn write_strip(
    out: &mut [f32],
    w: u32,
    spec: &TileSpec,
    tile_out: &[f32],
    t: usize,
    o: u32,
) {
    let wt = w as i64;
    let plane = t * t;
    let o_us = o as usize;
    for dy in 0..spec.valid_h {
        let y = spec.out_y + dy as i64;
        for dx in 0..spec.valid_w {
            let si = (o_us + dy as usize) * t + (o_us + dx as usize);
            let di = ((y * wt + spec.out_x + dx as i64) as usize) * 3;
            out[di] = model_srgb_to_linear(tile_out[si]);
            out[di + 1] = model_srgb_to_linear(tile_out[si + plane]);
            out[di + 2] = model_srgb_to_linear(tile_out[si + 2 * plane]);
        }
    }
}

// ── The tiled driver (fake-session seam) ────────────────────────────────────

/// The shared tiled-denoiser driver: for each tile, gather the planar
/// input, hand it to `run_tile`, and write the valid strip of the returned
/// planar output back into the packed RGB result. `progress(tile, total)`
/// fires after each completed tile (1-based, monotonically increasing — the
/// same per-tile cadence as the C `dt_control_job_set_progress`, and the
/// contract the UI's progress tick relies on).
///
/// Injecting `run_tile` is the fake-session seam: the production closure
/// wraps an [`RtSession`]; tests pass a pure function over the planar tile
/// and never touch ONNX Runtime or the network.
pub fn run_denoise_tiled(
    input: &[f32],
    w: u32,
    h: u32,
    t: u32,
    o: u32,
    mut run_tile: impl FnMut(&[f32], usize) -> Result<Vec<f32>, InferError>,
    mut progress: impl FnMut(u32, u32),
) -> Result<Vec<f32>, InferError> {
    if input.len() != w as usize * h as usize * 4 {
        return Err(InferError::InvalidArgument(format!(
            "input is {} f32, expected w*h*4 = {}",
            input.len(),
            w as usize * h as usize * 4
        )));
    }
    let grid = tile_grid(w, h, t, o)?;
    let t_us = t as usize;
    let out_len = w as usize * h as usize * 3;
    let mut out = vec![0f32; out_len];
    let total = grid.tiles.len() as u32;
    for (i, spec) in grid.tiles.iter().enumerate() {
        let planar = sample_tile_planar(input, w, h, spec, t_us);
        let tile_out = run_tile(&planar, t_us)?;
        if tile_out.len() != 3 * t_us * t_us {
            return Err(InferError::InvalidArgument(format!(
                "session returned {} f32 for a {t}x{t} tile, need {}",
                tile_out.len(),
                3 * t_us * t_us
            )));
        }
        write_strip(&mut out, w, spec, &tile_out, t_us, o);
        progress(i as u32 + 1, total);
    }
    Ok(out)
}

// ── Session lifecycle (see module doc for the per-call rationale) ───────────

/// An owned ONNX Runtime session plus the input-count snapshot used to
/// append the optional noise map. Built once per denoise call, used on one
/// worker thread, then dropped — see the module doc.
#[derive(Debug)]
pub struct RtSession {
    session: ort::session::Session,
    input_count: usize,
}

/// Build a session over one `.onnx` payload. See the module doc for the
/// exact ort calls and why no `ort::init()` appears here.
pub fn rt_session_from_onnx(onnx_path: &Path) -> Result<RtSession, InferError> {
    if !onnx_path.is_file() {
        return Err(InferError::Io(format!(
            "model payload not found: {}",
            onnx_path.display()
        )));
    }
    // ort call 1: Session::builder() — creates the environment lazily via
    // Environment::current() (CPU default; nothing else compiled in).
    let mut builder = match ort::session::Session::builder() {
        Ok(b) => b,
        Err(e) => return Err(InferError::Ort(e.message().to_string())),
    };
    // ort call 2: SessionBuilder::commit_from_file(path) — parse + load.
    let session = match builder.commit_from_file(onnx_path) {
        Ok(s) => s,
        Err(e) => return Err(InferError::Ort(e.message().to_string())),
    };
    // ort call 3: Session::inputs().len() — input-count probe (restore.c:289).
    let input_count = session.inputs().len();
    Ok(RtSession::new(session, input_count))
}

impl RtSession {
    fn new(session: ort::session::Session, input_count: usize) -> Self {
        Self { session, input_count }
    }

    /// Run one planar `{1,3,T,T}` f32 tile through the graph and return the
    /// model's planar output. Multi-input models (input_count >= 2) receive
    /// the constant 25/255 noise map second, matching restore_rgb.c:305-325.
    pub fn run_tile(&mut self, planar: &[f32], t: usize) -> Result<Vec<f32>, InferError> {
        if planar.len() != 3 * t * t {
            return Err(InferError::InvalidArgument("tile planar buffer is not 3*T*T f32".to_string()));
        }
        let shape: [i64; 4] = [1i64, 3i64, t as i64, t as i64];
        if self.input_count >= 2 {
            // ort call 4: Tensor::from_array for the constant noise map.
            let noise_shape: [i64; 4] = [1i64, 1i64, t as i64, t as i64];
            let noise = vec![25.0f32 / 255.0; t * t];
            let noise_t = match ort::value::Tensor::from_array((noise_shape, noise)) {
                Ok(v) => v,
                Err(e) => return Err(InferError::Ort(e.message().to_string())),
            };
            self.run_inputs_borrowed(planar, t, shape, Some(noise_t))
        } else {
            self.run_inputs_borrowed(planar, t, shape, None)
        }
    }

    fn run_inputs_borrowed(
        &mut self,
        planar: &[f32],
        t: usize,
        shape: [i64; 4],
        noise: Option<ort::value::Tensor<f32>>,
    ) -> Result<Vec<f32>, InferError> {
        // ort call 5: TensorRef::from_array_view over the tile buffer.
        let input = match ort::value::TensorRef::from_array_view((shape, planar)) {
            Ok(v) => v,
            Err(e) => return Err(InferError::Ort(e.message().to_string())),
        };
        let outputs = match noise {
            // ort call 6: Session::run with both inputs (positional).
            Some(noise_t) => match self.session.run(ort::inputs![input, noise_t]) {
                Ok(o) => o,
                Err(e) => return Err(InferError::Ort(e.message().to_string())),
            },
            // ort call 6: Session::run with the single image input.
            None => match self.session.run(ort::inputs![input]) {
                Ok(o) => o,
                Err(e) => return Err(InferError::Ort(e.message().to_string())),
            },
        };
        if outputs.len() == 0 {
            return Err(InferError::Ort("model produced no outputs".to_string()));
        }
        // ort call 7: SessionOutputs[0] + try_extract_tensor::<f32>().
        let out0 = &outputs[0];
        let (shape_out, data) = match out0.try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => return Err(InferError::Ort(e.message().to_string())),
        };
        let dims: &[i64] = shape_out;
        if dims.len() < 4 || dims[2] != t as i64 || dims[3] != t as i64 {
            return Err(InferError::Ort(format!(
                "model returned spatial output {dims:?}; static-shape models must return {t}x{t}"
            )));
        }
        if data.len() != 3 * t * t {
            let got = data.len();
            return Err(InferError::Ort(format!(
                "model output has {got} f32, expected 3*{t}*{t}"
            )));
        }
        Ok(data.to_vec())
    }
}

/// Full denoise of one RGBA frame (interleaved float4 RGBA at `w`×`h`,
/// linear working-space values — the same layout `RawImage::
/// to_linear_rgba_with` produces) into a packed RGB frame (`w*h*3` f32,
/// linear working space), written to disk by the caller. `tile_size` is the
/// manifest-declared static input dim; `progress(tile, total)` fires after
/// each completed tile. See the module doc for session lifecycle.
pub fn denoise_rgb(
    input: &[f32],
    w: u32,
    h: u32,
    onnx_path: &Path,
    tile_size: u32,
    progress: impl FnMut(u32, u32),
) -> Result<Vec<f32>, InferError> {
    if input.len() != w as usize * h as usize * 4 {
        return Err(InferError::InvalidArgument(format!(
            "input is {} f32, expected w*h*4 = {}",
            input.len(),
            w as usize * h as usize * 4
        )));
    }
    let mut session = rt_session_from_onnx(onnx_path)?;
    run_denoise_tiled(input, w, h, tile_size, O_DENOISE, |planar, t| {
        session.run_tile(planar, t)
    }, progress)
}

// ── Model handling (download + unpack + manifest resolution) ────────────────

/// One known RGB-denoise release, for the download-on-demand picker rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnownModel {
    pub name: String,
    /// Approximate size in MiB, taken from the release asset `size` field.
    pub size_mi_b: u32,
}

/// The known download-on-demand entries: denoise-nind and denoise-nafnet.
/// Sizes are the release asset `size` fields, rounded (55,082,110 B and
/// 108,413,592 B on release-5.6.0) — display only, never used for logic.
pub fn known_denoise_models() -> Vec<KnownModel> {
    vec![
        KnownModel { name: DENOISE_NIND_ASSET.to_string(), size_mi_b: 55 },
        KnownModel { name: DENOISE_NAFNET_ASSET.to_string(), size_mi_b: 108 },
    ]
}

/// `denoise-*.dtmodel` archives present in the store, filename-sorted. A
/// missing/unreadable store yields an empty vec — the picker then shows only
/// the download-on-demand rows.
pub fn downloaded_denoise_models() -> Vec<String> {
    let Some(dir) = download::models_dir() else {
        return Vec::new();
    };
    let Some(read) = std::fs::read_dir(&dir).ok() else {
        return Vec::new();
    };
    let mut out: Vec<String> = read
        .flatten()
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .filter_map(|e| e.path().file_name().map(|n| n.to_string_lossy().to_string()))
        .filter(|n| n.starts_with("denoise-") && n.ends_with(".dtmodel"))
        .collect();
    out.sort();
    out
}

/// Download-or-verify one [`KnownModel`] into the store, forwarding
/// `progress(downloaded, total)` to the u7a downloader untouched. The asset
/// must be present WITH a sha256 digest in the release `assets` listing —
/// the same no-unverified-bytes policy as `download::ensure_asset` and
/// ai_models.c. Returns the verified archive's store path.
pub fn ensure_denoise_model(
    model: &KnownModel,
    assets: &[registry::ModelAsset],
    progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, InferError> {
    let Some(asset) = registry::find_asset(assets, &model.name) else {
        return Err(InferError::Registry(format!(
            "release has no {} asset",
            model.name
        )));
    };
    let Some(sha) = asset.sha256_hex() else {
        return Err(InferError::Registry(format!(
            "release asset {} carries no sha256 digest",
            model.name
        )));
    };
    let Some(dest) = download::model_store_path(&model.name) else {
        return Err(InferError::Io("model store is unavailable".to_string()));
    };
    let dest_ref = dest.clone();
    match download::ensure_asset(&asset.download_url, &dest_ref, sha, asset.size, progress) {
        Ok(_) => Ok(dest),
        Err(e) => Err(InferError::Download(e.to_string())),
    }
}

/// The runnable form of a downloaded model.
#[derive(Clone, Debug)]
pub struct PreparedModel {
    pub onnx_path: PathBuf,
    pub tile_size: u32,
}

/// Turn a downloaded archive into something the session can load: unpack it
/// (only when the `model.onnx` payload is missing) and resolve the
/// manifest-declared tile size from the TOP-LEVEL `input_sizes` — the
/// NULL-stem lookup the C denoise loader performs (a nested `model.*` key
/// is never consulted for this task, even when present). A package
/// declaring neither is refused — the C hard-errors on the same ("static
/// ONNX requires a fixed tile size", restore.c:301-320). Unpacking is
/// once-per-payload: a re-downloaded archive whose contents changed keeps
/// the previous payload until the unpack dir is removed (documented
/// limitation; digests only change when upstream ships a new release).
pub fn prepare_denoise_model(asset_name: &str) -> Result<PreparedModel, InferError> {
    let Some(archive) = download::model_store_path(asset_name) else {
        return Err(InferError::Io("model store is unavailable".to_string()));
    };
    if !archive.is_file() {
        return Err(InferError::MissingModel(asset_name.to_string()));
    }
    let stem = asset_name.strip_suffix(".dtmodel").unwrap_or(asset_name);
    if stem.is_empty()
        || stem.contains('/')
        || stem.contains('\\')
        || stem == ".."
        || stem == "."
    {
        return Err(InferError::InvalidArgument(format!(
            "unsafe model name {asset_name:?}"
        )));
    }
    let Some(base) = download::models_dir() else {
        return Err(InferError::Io("model store is unavailable".to_string()));
    };
    let unpack_dir = base.join(stem);
    let payload = unpack_dir.join(MODEL_ONNX_FILE);
    if !payload.is_file() {
        let _ = std::fs::remove_dir_all(&unpack_dir);
        let arc = archive.clone();
        let dir = unpack_dir.clone();
        match package::unpack_dtmodel(&arc, &dir) {
            Ok(_) => {}
            Err(e) => return Err(InferError::Package(e.to_string())),
        }
    }
    if !payload.is_file() {
        return Err(InferError::Package(format!(
            "archive {asset_name} has no {} payload",
            MODEL_ONNX_FILE
        )));
    }
    let manifest = match package::load_manifest_from_dir(&unpack_dir) {
        Ok(m) => m,
        Err(e) => return Err(InferError::Package(e.to_string())),
    };
    // NULL-stem semantics like the C denoise loader: top-level
    // `input_sizes` only, never a nested `model.*` key.
    let Some(tile_size) = package::top_level_tile_size(&manifest)
        .and_then(|t| if t > 0 { Some(t as u32) } else { None })
    else {
        return Err(InferError::Package(format!(
            "model {asset_name} declares no input_sizes"
        )));
    };
    Ok(PreparedModel { onnx_path: payload, tile_size })
}

#[cfg(test)]
mod tests {
    use super::super::download;
    use super::super::package;
    use super::super::registry;
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Serializes the tests that read or write the process-global
    /// `C41_MODELS_DIR` (cargo runs tests in parallel threads).
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    /// Uniqueness for test temp dirs (download.rs house pattern).
    static TEST_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    const EPS: f32 = 1e-3;

    // ── Gamma ─────────────────────────────────────────────────────────────

    #[test]
    fn gamma_matches_restore_rgb_curve() {
        // Values hand-derived from restore_rgb.c:79-93:
        //   v <= 0.0031308 -> 12.92v
        //   else 1.055 * v^(1/2.4) - 0.055
        //   srgb_in = _linear_to_srgb(fminf(v, 1.0))
        // 0.5 -> 1.055 * 0.5^(5/12) - 0.055 ~= 0.73536
        assert!((model_linear_to_srgb(0.5) - 0.73536).abs() < 1e-4);
        assert!((model_linear_to_srgb(0.001) - 12.92 * 0.001).abs() < 1e-9);
        assert_eq!(model_linear_to_srgb(-0.5), 0.0, "negatives clamp to 0");
        assert_eq!(model_linear_to_srgb(0.0), 0.0);
        assert!((model_linear_to_srgb(1.5) - 1.0).abs() < 1e-4, "overshoot clamps to 1");
        assert!((model_linear_to_srgb(1.0) - 1.0).abs() < 1e-4);
        // inverse: 0.5 -> ((0.055+0.5)/1.055)^2.4 ~= 0.21404
        assert!((model_srgb_to_linear(0.5) - 0.21404).abs() < 1e-4);
        assert_eq!(model_srgb_to_linear(-0.2), 0.0);
        assert_eq!(model_srgb_to_linear(0.0), 0.0);
        assert!((model_srgb_to_linear(1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn gamma_roundtrip_is_identity_in_range() {
        // For inputs already in [0,1] the encode/decode pair is the
        // identity — this is the passthrough an identity model relies on.
        for v in [0.0, 0.0001, 0.0031308, 0.04, 0.18, 0.5, 0.9, 1.0] {
            let back = model_srgb_to_linear(model_linear_to_srgb(v));
            assert!((back - v).abs() < EPS, "roundtrip of {v} gave {back}");
        }
    }

    // ── Mirror ─────────────────────────────────────────────────────────────

    #[test]
    fn mirror_matches_the_c_periodic_reflection() {
        // Hand-run restore_common.h:201-210 `_mirror` for each vector.
        for (i, n, e) in [
            (0, 10, 0), (1, 10, 1), (9, 10, 9), (10, 10, 8), (11, 10, 7),
            (18, 10, 0), (19, 10, 1), (-1, 10, 1), (-9, 10, 9), (-10, 10, 8),
            (-11, 10, 7), (0, 2, 0), (1, 2, 1), (2, 2, 0), (3, 2, 1),
            (-1, 2, 1), (7, 1, 0), (3, 1, 0),
        ] {
            assert_eq!(mirror_coord(i, n), e as i64, "mirror({i}, {n})");
        }
    }

    // ── Tile grid ──────────────────────────────────────────────────────────

    #[test]
    fn tile_grid_matches_restore_c_counts_and_offsets() {
        // w=100 h=60 with T=512, O=64: step 384, one tile per axis whose
        // window starts at -64 (mirrored) and whose valid region is the
        // whole image.
        let g = tile_grid(100, 60, 512, 64).unwrap();
        assert_eq!((g.cols, g.rows, g.step), (1, 1, 384));
        assert_eq!(g.tiles.len(), 1);
        let t0 = g.tiles[0];
        assert_eq!((t0.in_x, t0.in_y), (-64, -64));
        assert_eq!((t0.out_x, t0.out_y), (0, 0));
        assert_eq!((t0.valid_w, t0.valid_h), (100, 60));
        assert!(t0.needs_mirror);

        // 5000x3000 with T=512: 14x8 grid, 384 step; the last column's strip
        // clips to the 8 px that remain past 4992.
        let g2 = tile_grid(5000, 3000, 512, 64).unwrap();
        assert_eq!((g2.cols, g2.rows, g2.step), (14, 8, 384));
        assert_eq!(g2.tiles.len(), 14 * 8);
        let last = g2.tiles[13];
        assert_eq!((last.out_x, last.out_y), (4992, 0));
        assert_eq!((last.valid_w, last.valid_h), (8, 384));
        assert!(last.needs_mirror, "right-edge tile mirrors");
        let first = g2.tiles[0];
        assert_eq!((first.valid_w, first.valid_h), (384, 384));
        assert!(first.needs_mirror, "top-left tile mirrors both axes");
        // An interior tile (col 5, row 3) needs no mirror and owns a full
        // step strip.
        let mid = g2.tiles[3 * 14 + 5];
        assert_eq!((mid.valid_w, mid.valid_h), (384, 384));
        assert!(!mid.needs_mirror);
    }

    #[test]
    fn tile_grid_rejects_degenerate_inputs() {
        assert!(tile_grid(0, 10, 512, 64).is_err(), "empty width");
        assert!(tile_grid(10, 0, 512, 64).is_err(), "empty height");
        assert!(tile_grid(10, 10, 64, 64).is_err(), "T == 2O");
        assert!(tile_grid(10, 10, 128, 64).is_err(), "T < 2O");
        assert!(tile_grid(10, 10, 0, 0).is_err(), "zero tile");
        assert!(tile_grid(10, 10, 129, 64).is_ok(), "smallest usable tile");
    }

    #[test]
    fn tile_strips_pave_the_image_exactly() {
        // For several sizes the valid strips are a partition of the image:
        // every output pixel lies in exactly one strip, in-bounds.
        for (w, h, t, o) in [(512, 512, 512, 64), (100, 60, 512, 64), (257, 193, 96, 32), (1, 1, 512, 64), (384, 384, 512, 64)] {
            let g = tile_grid(w, h, t, o).unwrap();
            let mut covered = vec![0u32; w as usize * h as usize];
            for spec in &g.tiles {
                for dy in 0..spec.valid_h {
                    for dx in 0..spec.valid_w {
                        let x = spec.out_x + dx as i64;
                        let y = spec.out_y + dy as i64;
                        assert!(
                            x >= 0 && x < w as i64 && y >= 0 && y < h as i64,
                            "strip escapes image"
                        );
                        covered[(y as usize) * w as usize + x as usize] += 1;
                    }
                }
            }
            for c in covered {
                assert_eq!(c, 1, "pixel covered {c} times (not exactly once)");
            }
        }
    }

    // ── Fake-session driver ────────────────────────────────────────────────

    /// An identity model: returns its planar input untouched, so the tiled
    /// output must equal the clamped input after the gamma roundtrip.
    fn identity_tile(_planar: &[f32], _t: usize) -> Result<Vec<f32>, InferError> {
        Ok(_planar.to_vec())
    }

    #[test]
    fn identity_model_roundtrips_through_tiles() {
        let (w, h) = (97u32, 83u32);
        // Deterministic gradient in the safe [0.05, 0.95] linear zone.
        let mut input = vec![0f32; w as usize * h as usize * 4];
        for (i, v) in input.iter_mut().enumerate() {
            let px = i / 4;
            let x = (px % w as usize) as f32;
            let y = (px / w as usize) as f32;
            let c = (i % 4) as f32;
            *v = 0.05 + (x * 0.001 + y * 0.0007 + c * 0.11) % 0.9;
        }
        let mut seen: Vec<(u32, u32)> = Vec::new();
        let out = run_denoise_tiled(&input, w, h, 64, 16, identity_tile, |d, t| seen.push((d, t)))
            .unwrap();
        assert_eq!(out.len(), w as usize * h as usize * 3);
        let clamp01 = |v: f32| v.clamp(0.0, 1.0);
        for i in 0..out.len() {
            let expected = clamp01(input[(i / 3) * 4 + i % 3]); // rgb channels only
            let got = out[i];
            assert!((got - expected).abs() < EPS, "sample {i}: {got} vs {expected}");
        }
    }

    #[test]
    fn constant_model_covers_every_output_pixel_once() {
        let (w, h, t, o) = (257u32, 193u32, 96u32, 32u32);
        let input = vec![0.25f32; w as usize * h as usize * 4];
        // A model that always answers 0.5: whatever the tiles, every output
        // pixel must land on decode(0.5). A gap or a repeated region would
        // leave an untouched 0.0 (from the init) or a wrong decode.
        let out = run_denoise_tiled(
            &input, w, h, t, o,
            |_planar, _t| Ok(vec![0.5f32; 3usize * (t as usize) * (t as usize)]),
            |_, _| {},
        )
        .unwrap();
        let expect = model_srgb_to_linear(0.5);
        for (i, v) in out.iter().enumerate() {
            assert!((*v - expect).abs() < 1e-4, "pixel {i} = {v}, want {expect}");
        }
    }

    #[test]
    fn progress_is_monotonic_and_reaches_total() {
        let (w, h, t, o) = (400u32, 300u32, 128u32, 48u32);
        let input = vec![0.4f32; w as usize * h as usize * 4];
        let mut seen: Vec<(u32, u32)> = Vec::new();
        let _ = run_denoise_tiled(&input, w, h, t, o, identity_tile, |d, total| {
            seen.push((d, total));
        })
        .unwrap();
        assert!(!seen.is_empty());
        for wnd in seen.windows(2) {
            assert!(wnd[1].0 > wnd[0].0, "strictly increasing: {seen:?}");
            assert_eq!(wnd[0].1, wnd[1].1, "total constant: {seen:?}");
        }
        let total = seen.last().unwrap().1;
        assert_eq!(seen.last().unwrap().0, total, "final tile == total");
        // 400x300 with step 128-96=32: ceil(400/32)=13, ceil(300/32)=10.
        assert_eq!(total, 130);
    }

    #[test]
    fn driver_rejects_bad_lengths_without_panicking() {
        let (w, h) = (10u32, 10u32);
        let short = vec![0f32; 3];
        let r = run_denoise_tiled(&short, w, h, 512, 64, identity_tile, |_, _| {}).unwrap_err();
        assert!(matches!(r, InferError::InvalidArgument(_)));
        // A tile-runner returning the wrong length is an error, not a panic.
        let r = run_denoise_tiled(
            &vec![0f32; 400],
            w, h, 512, 64,
            |_p, _t| Ok(vec![1f32; 1]),
            |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(r, InferError::InvalidArgument(_)));
    }

    // ── Model handling (fixtures, no network) ──────────────────────────────

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn fresh(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "c41_ai_infer_{}_{}_{}",
                std::process::id(),
                name,
                TEST_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// A minimal single-model `.dtmodel`: config.json declaring a top-level
    /// tile size plus a fake `model.onnx` payload.
    fn synthetic_dtmodel(tile_size: u32) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            w.start_file(package::MANIFEST_FILENAME, opts).unwrap();
            w.write_all(format!(
                r#"{{"name":"fixture","attributes":{{"input_sizes":[{tile_size}]}}}}"#
            ).as_bytes())
            .unwrap();
            w.start_file(MODEL_ONNX_FILE, opts).unwrap();
            w.write_all(b"fake-onnx-payload").unwrap();
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    fn with_models_dir<T>(name: &str, f: impl FnOnce(&Path) -> T) -> T {
        let _env = ENV_LOCK.lock().unwrap();
        let dir = TempDir::fresh(name);
        let prev = std::env::var_os("C41_MODELS_DIR");
        std::env::set_var("C41_MODELS_DIR", &dir.path);
        let out = f(&dir.path);
        match prev {
            Some(v) => std::env::set_var("C41_MODELS_DIR", v),
            None => std::env::remove_var("C41_MODELS_DIR"),
        }
        out
    }

    #[test]
    fn known_models_pin_names_and_sizes() {
        // Sizes mirror the release asset `size` fields (rounded MiB);
        // display-only. If upstream re-exports change asset sizes, update
        // the pins alongside the doc comment on `known_denoise_models`.
        let known = known_denoise_models();
        assert_eq!(known.len(), 2);
        assert_eq!(known[0].name, DENOISE_NIND_ASSET);
        assert_eq!(known[0].size_mi_b, 55);
        assert_eq!(known[1].name, DENOISE_NAFNET_ASSET);
        assert_eq!(known[1].size_mi_b, 108);
    }

    #[test]
    fn downloaded_scan_lists_only_denoise_archives() {
        with_models_dir("scan", |dir| {
            std::fs::write(dir.join("denoise-nind.dtmodel"), b"a").unwrap();
            std::fs::write(dir.join("denoise-nafnet.dtmodel"), b"b").unwrap();
            std::fs::write(dir.join("rawdenoise-nind.dtmodel"), b"c").unwrap();
            std::fs::write(dir.join("versions.json"), b"d").unwrap();
            std::fs::create_dir_all(dir.join("denoise-subdir.dtmodel")).unwrap();
            // A DIRECTORY named like a model must not be listed as a model.
            assert_eq!(
                downloaded_denoise_models(),
                vec!["denoise-nafnet.dtmodel".to_string(), "denoise-nind.dtmodel".to_string()]
            );
        });
    }

    #[test]
    fn downloaded_scan_absent_store_is_empty() {
        with_models_dir("empty", |_dir| assert!(downloaded_denoise_models().is_empty()));
    }

    #[test]
    fn prepare_unpacks_and_resolves_manifest_tile_size() {
        with_models_dir("prepare", |dir| {
            let archive = dir.join(DENOISE_NIND_ASSET);
            std::fs::write(&archive, synthetic_dtmodel(256)).unwrap();
            let p = prepare_denoise_model(DENOISE_NIND_ASSET).unwrap();
            assert_eq!(p.tile_size, 256);
            assert_eq!(p.onnx_path, dir.join("denoise-nind").join(MODEL_ONNX_FILE));
            assert!(p.onnx_path.is_file());
            assert_eq!(std::fs::read(&p.onnx_path).unwrap(), b"fake-onnx-payload");
        });
    }

    #[test]
    fn prepare_refuses_missing_and_undeclared() {
        with_models_dir("prepare-err", |dir| {
            assert!(matches!(
                prepare_denoise_model("nope.dtmodel"),
                Err(InferError::MissingModel(_))
            ));
            let no_tile = dir.join("no-tile.dtmodel");
            let mut buf = std::io::Cursor::new(Vec::new());
            {
                let mut w = zip::ZipWriter::new(&mut buf);
                let opts = zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored);
                w.start_file(package::MANIFEST_FILENAME, opts).unwrap();
                w.write_all(br#"{"name":"x"}"#).unwrap();
                w.finish().unwrap();
            }
            std::fs::write(&no_tile, buf.into_inner()).unwrap();
            assert!(matches!(
                prepare_denoise_model("no-tile.dtmodel"),
                Err(InferError::Package(_))
            ));
        });
    }

    // Localhost HTTP stub (download.rs pattern) serving one fixed body, so
    // the ensure_denoise_model success path is exercised without a network.
    struct StubServer {
        addr: std::net::SocketAddr,
        hits: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl StubServer {
        fn start(body: Vec<u8>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let hits = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let (hits_c, stop_c) = (Arc::clone(&hits), Arc::clone(&stop));
            let handle = std::thread::spawn(move || {
                while !stop_c.load(std::sync::atomic::Ordering::Relaxed) {
                    let (mut s, _) = match listener.accept() {
                        Ok(v) => v,
                        Err(_) => {
                            std::thread::sleep(Duration::from_millis(5));
                            continue;
                        }
                    };
                    hits_c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    s.set_nonblocking(false).ok();
                    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
                    let mut req = vec![0u8; 4096];
                    let _ = s.read(&mut req);
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = s.write_all(head.as_bytes());
                    let _ = s.write_all(&body);
                }
            });
            Self { addr, hits, stop, handle: Some(handle) }
        }

        fn url(&self, path: &str) -> String {
            format!("http://{}{}", self.addr, path)
        }
    }

    impl Drop for StubServer {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    impl StubServer {
        fn hits(&self) -> usize {
            self.hits.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[test]
    fn ensure_denoise_downloads_through_the_verified_path() {
        let body = b"verified model bytes".to_vec();
        let body_len = body.len();
        let sha = download::sha256_hex_of_bytes(&body);
        let server = StubServer::start(body);
        let assets = vec![registry::ModelAsset {
            name: DENOISE_NIND_ASSET.to_string(),
            download_url: server.url("/denoise-nind.dtmodel"),
            size: Some(body_len as u64),
            digest: Some(format!("sha256:{sha}")),
        }];
        with_models_dir("ensure", |_dir| {
            let model = KnownModel { name: DENOISE_NIND_ASSET.to_string(), size_mi_b: 55 };
            let mut shots: Vec<(u64, Option<u64>)> = Vec::new();
            let dest = ensure_denoise_model(
                &model,
                &assets,
                |d, t| shots.push((d, t)),
            )
            .unwrap();
            assert!(dest.is_file());
            assert_eq!(std::fs::read(&dest).unwrap(), b"verified model bytes".to_vec());
            assert!(!shots.is_empty(), "progress forwarded");
            assert_eq!(server.hits(), 1, "exactly one download request");
        });
    }

    #[test]
    fn ensure_denoise_refuses_unknown_or_digestless_assets() {
        let no_asset = ensure_denoise_model(
            &known_denoise_models()[0],
            &[],
            |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(no_asset, InferError::Registry(_)));
        let digestless = vec![registry::ModelAsset {
            name: DENOISE_NIND_ASSET.to_string(),
            download_url: "http://127.0.0.1:1/x".to_string(),
            size: None,
            digest: None,
        }];
        let err = ensure_denoise_model(
            &known_denoise_models()[0],
            &digestless,
            |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(err, InferError::Registry(_)), "{err}");
    }
}