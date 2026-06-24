# Darkroom -- Rust Migration Plan

Incremental rewrite of the Darkroom codebase (C/GTK3) into Rust + GTK4 via the
`gtk4-rs` bindings. The chosen strategy is **incremental FFI-boundary migration**:
each subsystem is replaced one at a time behind a stable C FFI layer, keeping the
application runnable throughout.

---

## Current status -- 2026-06-04

### Phase 0 -- Infrastructure complete

### Phase 1 -- Image pipeline at hard boundary

| Metric | Value |
|---|---|
| IOP Rust modules registered | **93 / 93** |
| Unit tests passing | **474** |
| IOP `.rs` files | 95 (one per C IOP) |
| Shared modules | `color`, `math`, `raw`, `geometry`, `markesteijn`, `fdc` |
| Last patch | `Phase 2z+96` (xtrans.c — FDC Markesteijn variant; **demosaicing/ fully migrated, 0 OMP loops**) |
| CI status | `Rust` workflow green; `Fork CI` green |

**All 93 `src/iop/*.c` files have a corresponding Rust module.**
The migration has reached the hard boundary: every remaining `DT_OMP_FOR` loop
depends on shared infrastructure (color-space transforms, interpolation,
bilateral grid, NLM, perspective matrices) not yet in Rust. Those IOPs have
stub `IopProcess` impls registered; their loops will be ported once the
blocking infrastructure lands.

#### Fully migrated IOPs (all active OMP loops -> Rust, 0 remain in C)

`agx`, `atrous`, `basecurve`, `basicadj`, `bloom`, `cacorrect`, `cacorrectrgb`,
`censorize`, `channelmixer`, `clahe`, `clipping`,
`colorbalance`, `colorchecker`, `colorcontrast`, `colorcorrection`,
`colorize`, `colormapping`, `colorzones`, `defringe`, `denoiseprofile`,
`dither`, `exposure`,
`filmic`, `filmicrgb`, `globaltonemap`, `graduatednd`,
`grain`, `hazeremoval`, `highlights` (incl. all 6 hlreconstruct/ backends),
`highpass`, `hotpixels` (all 3 variants),
`invert`, `levels`, `liquify`, `lowlight`, `lowpass`, `lut3d`, `monochrome`,
`negadoctor`, `overexposed` (all 4 modes), `overlay`, `primaries`,
`profile_gamma`, `rasterfile`, `rawdenoise`, `rawoverexposed`, `rawprepare`,
`relight`, `rgbcurve`, `rgblevels`, `shadhi`, `sharpen`, `sigmoid`, `soften`,
`splittoning`, `temperature`, `toneequal` (main process loop is
`#else`-guarded dead code since `DT_TONEEQ_USE_LUT=TRUE`),
`useless`, `velvia`, `vibrance`, `vignette`, `watermark`, `zonesystem`.

Geometric distort loops fully migrated in `geometry.rs`:
`borders`, `crop`, `enlargecanvas`, `flip`, `rotatepixels` (distort only).

Commit-params LUT builders migrated:
`colisa`, `lowpass`, `profile_gamma` (contrast/brightness LUT fills).

The entire **`demosaicing/` cluster is fully migrated** (0 OMP loops): basics,
passthrough, dual, rcd (box3), ppg, vng (all of `_vng_lininterpolate`, the
dcraw gradient kernel, finishing pass), capture-sharpen (blur, blend,
sharpen-output, gauss-idx, radius calcs, auto-radius), and both X-Trans
Markesteijn variants (`markesteijn.rs`, `fdc.rs` + extracted `fdc_tables.rs`).

#### Partially migrated (some loops remain, blocked on infrastructure)

Loop counts verified 2026-06-12 (`grep -rcE 'DT_OMP_FOR(_SIMD)?\(' src/iop --include=*.c`):

| IOP | C loops remaining | Blocking dependency |
|-----|------------------|---------------------|
| `colorbalancergb` | 4 | Filmlight Yrg / `work_profile` |
| `colorreconstruction` | 3 | 3D bilateral grid |
| `colorin` | 3 | ICC matrix + LCMS |
| `channelmixerrgb` | 2 | B-spline local avg reduction (illuminant detection) |
| `colortransfer` | 2 | k-means with atomic accumulators |
| `retouch` | 2 | `dt_linearRGB_to_XYZ` / `dt_XYZ_to_Lab` ICC paths |
| `colorequal` | 1 | GUI background renderer (intentionally deferred) |
| `colorout` | 1 | LCMS `cmsDoTransform` |
| `diffuse` | 1 | anisotropic PDE solver (very complex) |
| `toneequal` | 1 | GUI LUT |

(`ashift`, `clipping`, `denoiseprofile`, `gamma`, `liquify`, `rawoverexposed`
previously listed here/as stubs are at 0 loops — fully migrated.)

#### What blocks the remaining loops

| Infrastructure | Unblocked IOPs |
|---|---|
| `dt_interpolation_*` | demosaicing cluster |
| 3D bilateral grid | colorreconstruction |
| Filmlight Yrg / `work_profile` callbacks | colorbalancergb, colorin |
| Per-pixel ICC / LCMS | colorin, colorout, retouch |
| GUI-only loops | colorequal, toneequal GUI LUT |

Pattern note (Phase 2z+80, rawoverexposed): loops interleaved with C
pipeline callbacks (`dt_dev_distort_backtransform_plus`) are split into
serial C row loops calling Rust for the work before/after the callback —
no Rust→C callback plumbing needed.

#### Shared darkroom-core modules

| Module | Purpose |
|--------|---------|
| `color` | RGB<->HSL, Lab<->XYZ, dt UCS 2.2 (JCH/HSB), CAT16 matrices, gamut helpers |
| `math` | `fastlog2`, `fastlog`, PRNG (`splitmix32`, `xoshiro128+`, all noise distributions) |
| `raw` | `fc_bayer`, `fc_xtrans`, `fcol` -- CFA Bayer/X-Trans primitives |
| `geometry` | Coord-shift, flip/swap, rotate (2x2 matrix), row-blit |

### Phase 2 -- Database complete

`darkroom-db` crate: full CRUD for tags, metadata, film, collection, image, history.
C FFI trampolines for tags. 61 DB tests passing.

### Phase 3 -- GTK4 UI (in progress)

`crates/darkroom-ui` (gtk4 0.9 + libadwaita 0.7). `darkroom_ui::run()` boots an
`adw::Application`. **Done (ui-1..ui-14):**
- **Lighttable** (functional): DB-backed `GridView` of thumbnails, collections
  left panel, metadata right panel, name search/filter, star ratings, import &
  export dialogs, `adw::ToastOverlay` notifications, Ctrl+I / Ctrl+E shortcuts.
- **Navigation**: `adw::NavigationView` lighttable ⇄ darkroom (double-click).
- **Darkroom view**: grouped IOP **module panel** sourced from a real module
  catalog, Export, and a **live multi-IOP preview pipeline** (`preview.rs`,
  ui-12/13/15): `PreviewParams` drives stages each chaining a *migrated
  `darkroom-core` IOP* over the decoded 8-bit preview, re-uploading a
  `gdk::MemoryTexture`. Stages so far (pixelpipe order): `exposure` (black+EV) →
  `velvia` (strength) → `splittoning` (shadow/highlight hue+sat, balance,
  compress) → `monochrome` (channelmixer GRAY B&W mix, ui-18). RGBA-loop IOPs
  share a `run_rgba_stage` helper. A stand-in for a
  real Rust pixelpipe — it processes the 8-bit pixbuf, not raw pipeline output,
  but exercises the genuine UI↔core seam.
- **Per-module param widgets (ui-14/15)**: the preview params live in their
  **module rows** in the panel, not a separate bar. Live modules (Exposure,
  Velvia, Split-toning) render as `adw::ExpanderRow`s whose built-in enable
  switch gates the pipeline stage (`*_on`) and whose child sliders drive the
  params; this converges the module-stack UI with the preview pipeline. A
  `catalog_has_live_modules` test guards the label→module dispatch.
- **Live histogram + state refactor (ui-16)**: shared preview state bundled in a
  `PreviewCtx` (weak widget refs + `Rc` data, no cycles); generic
  `module_expander`/`add_param_slider` collapse the per-module builders. A
  `gtk4::DrawingArea` under the image draws a live RGB histogram
  (`preview::compute_histogram`) of the *processed* output, refreshed each
  render. Begins **milestone 3** (darkroom interactions).

**This replaces (eventually):** `src/views/` (7: lighttable, darkroom, map,
print, slideshow, tethering), `src/libs/` (33 panels), `src/gui/` (16).

**Roadmap (next milestones):**
1. *Module stack realism* — module catalog → per-module param widgets, enable
   toggles wired to history/db, module groups & favourites (mirrors src/libs/
   modulegroups + the IOP GUIs). **In progress (ui-14/db-wiring):** Exposure,
   Velvia, Split-toning & Monochrome are live `ExpanderRow` modules; their params
   **persist to the db** and restore on reopen — `preview::PreviewParams::
   encode/decode` (versioned blob) stored in a dedicated `main.darkroom_preview`
   table (`crate::persist`), saved via a debounced autosave + flush on close.
   `darkroom-db` gained the history write path (`history_add_entry`,
   `history_get_op_params`) for future real-IOP-history wiring. Remaining: more
   live modules, module groups & favourites.
2. *Live preview* — the load-bearing piece: a Rust pixelpipe driver that runs
   the migrated `darkroom-core` IOPs and paints processed output into the
   darkroom view. **Bootstrapped (ui-12/13/14):** `preview.rs::apply_pipeline`
   chains migrated IOPs over the 8-bit pixbuf, live via the per-module widgets.
   **Core orchestrator landed (m2-1):** `darkroom_core::pipeline` — an ordered
   `Pipeline` of `Stage`s (exposure/velvia/splittoning/monochrome) over a
   scene-referred **float RGBA** buffer, ping-pong buffered, length-contract
   asserted. **darkroom-ui adopted it (m2-2)** via `PreviewParams::to_pipeline`,
   and the preview now runs in **linear light (m2-3)** — sRGB-decode on input /
   re-encode on output (`color::srgb_to_linear`/`linear_to_srgb`), so stages run
   in the real pipeline's domain. **Raw front end started (m2-4a):**
   `darkroom_core::rawimage` decodes camera raws via the pure-Rust `rawloader`
   (pinned `=0.37.1`) into a black/white-normalised linear CFA mosaic
   (`normalize_cfa`); Bayer only (X-Trans 6x6 rejected until its demosaic is
   wired). **m2-4b:** `demosaic_box` (reuses the migrated box3 kernel via a
   tested `filters_from_cfa` bridge) + `apply_white_balance` +
   `RawImage::to_linear_rgba()` give a full decode→demosaic→WB→linear-RGBA path
   ready for `pipeline`. **m2-4c:** the darkroom view now opens **camera raws**
   — `crate::raw_preview` (darkroom-ui) decodes off-thread, downscales in linear,
   sRGB-encodes to an 8-bit `BaseImage` that flows through the same preview
   pipeline as a JPEG; verified end-to-end on a real 16MP Olympus ORF
   (decode→demosaic→WB→downscale→display preview). **m2-4d:** EXIF orientation
   applied (`apply_orientation`) — visual confirmation caught portrait raws
   rendering upside-down; fixed (flips-then-transpose) and re-verified upright on
   a real EXIF-8 portrait ORF. **m2-5a:** `Stage::Sigmoid` tone mapping wired
   into the core pipeline — `sigmoid::rgb_ratio_params` ports sigmoid.c
   commit_params (contrast/skew/white/black → the 6 rgb-ratio process args),
   golden-pinned; the stage reuses the migrated `darkroom_sigmoid_rgb_ratio_process`.
   **m2-5b:** wired into the UI — `PreviewParams` gains sigmoid on/contrast/skew
   (codec bumped v1→v2), `to_pipeline` runs sigmoid LAST (display transform) with
   fixed safe white/black targets, and the darkroom view **defaults sigmoid ON
   for raws** (scene-linear) / off for JPEGs (display-referred). Visually verified
   on a real high-DR portrait raw (flat → proper contrast/depth). Known limit: the
   8-bit `BaseImage` clips scene-linear highlights >1 before sigmoid (fixed by the
   float `BaseImage`). **m2-5c:** Sigmoid module row (enable + contrast/skew
   sliders) wired into the panel; `persist::load_saved` now returns
   `Option<PreviewParams>` so the raw-default-sigmoid only fires with no saved
   edit (a user can persist sigmoid-off on a raw) — seeding extracted to a pure,
   unit-tested `initial_params`. **m2-6:** **float `BaseImage`** — `BaseImage` is
   now an enum (`Srgb8` JPEG path unchanged | `Linear` raw path, packed
   scene-linear RGBA f32, values >1). The raw preview runs the pipeline on the
   unclipped linear buffer (`render_linear_to_srgb8`), so sigmoid rolls off
   highlights >1 instead of them clamping to white at an 8-bit boundary —
   visually verified (sunset sky: blown-white → smooth gradient; preview range
   now [0, 2.08]). **m2-7a:** colour-picker samples a cached render (refcounted
   `glib::Bytes`, no per-click re-render). **m2-7b:** **PPG demosaic** — the raw
   front end now uses the migrated PPG green + red/blue kernels (with a ported
   3-pixel border interpolate) instead of box3; sharper, fewer colour artefacts,
   visually verified, box3 kept as the small-image fallback. **m2-7c:** **X-Trans
   (Fuji) support** — `RawImage` gains `xtrans: Option<[[u8;6];6]>`; a pure,
   closure-driven `classify_cfa` distinguishes 2×2 Bayer from 6×6 X-Trans
   (fail-closed on any other period / out-of-range colour, 5 file-free tests);
   `normalize_xtrans` + `demosaic_xtrans` route through the migrated single-pass
   **Markesteijn** kernel (self-tiling, so one full-frame call), `to_linear_rgba`
   switches on the sensor. Unit-verified on synthetic X-Trans mosaics (neutral +
   gradient reconstruction). **Deferred:** real-`.raf` visual verification — no
   Fuji sample is available; the synthetic tests build the mosaic from the same
   CFA table they demosaic with, so they are tautologically phase-aligned and
   **cannot catch a real-sensor CFA crop/phase offset** (the most likely
   real-image bug — shows as colour-swap/zippering). That is the headline item on
   the X-Trans verification ticket. Remaining: (a) the deferred X-Trans visual
   check; (b) higher-quality Bayer demosaic (PPG is the current baseline; RCD/VNG
   migrated); (c) ROI/(w,h) signature + geometry-aware IOPs; (d) OpenCL.
3. *Darkroom interactions* — zoom/pan, histogram (**done, ui-16**),
   before/after toggle (**done, ui-17**), reset-all (**done, ui-19**), colour
   picker (**done, ui-20**: click-to-sample the processed pixel; pure
   coordinate-mapping + sampling helpers in `preview.rs`).
4. *Remaining views/panels* — port src/libs panels (history stack, snapshots,
   tagging, export) and the other views. **m4-1: history-stack panel** —
   `history.rs` holds a pure `HistoryStack` of `PreviewParams` snapshots
   (record/undo/redo/jump, redo-tail truncation, dedup of identical states, a
   100-entry cap that pins the "Original" seed) plus `describe_change` for entry
   labels (first differing module group); 11 file-free tests incl. an exhaustive
   `PreviewParams` destructure that fails to compile if a field is added without
   extending `describe_change`. The darkroom view records one (700ms-debounced)
   entry per settled edit via a `HistoryRecorder` (mirrors `AutoSave`) and shows
   a clickable **History** panel above the modules; clicking an entry restores
   its params and repopulates the sliders (reusing the Reset path), the jump
   landing on its own entry so the render dedups. **m4-2: Undo/Redo header
   buttons** — step the history cursor via `undo()`/`redo()` (no-op at the ends,
   so no sensitivity tracking); a shared `apply_history_params` helper (used by
   jump + undo + redo) restores params, repopulates sliders, re-selects the
   cursor row, **and exits the before/after peek** (clears `bypass` + syncs the
   toggle, so a restore can't leave the viewport painting a bypassed image that
   disagrees with the params — architect S3). Reset icon changed to
   `edit-clear-all-symbolic` to disambiguate from Undo. **m4-3: keyboard
   shortcuts** — a page-root `ShortcutController` (`Local` scope on the
   `ToolbarView`, so it covers header + content and dies on page pop) binds
   Ctrl+Z → Undo and Ctrl+Shift+Z / Ctrl+Y → Redo by re-emitting the buttons'
   `clicked` (single source of truth; `undo`/`redo` bounds-check so repeats past
   the ends are safe no-ops). **m4-4: snapshots panel** — pure `snapshots.rs`
   (`SnapshotStore<P>`: monotonic auto-labels, cap-8 oldest-eviction, remove/
   clear; 6 tests) instantiated over the cached render. A "Take" button freezes
   `last_render` (refcounted `glib::Bytes`, no copy); a Snapshots list shows the
   captures (per-row remove); clicking one reveals it in a **second `Picture`**
   beside the live image (side-by-side compare, `cached_render_texture` shared
   with `render_preview`), "Stop comparing" hides it. The second Picture is a
   separate widget, so the `render_preview` sole-writer invariant (picker
   correctness) is untouched. Compare is approximate (independent `Contain`
   letterboxing), not a scale-locked wipe — that's a future increment.
   **m4-5: persist the edit-history stack** — `HistoryStack` gains a versioned LE
   `encode`/`decode` (fully bounds-checked: rejects bad version / truncation /
   trailing bytes / out-of-range cursor / 0-or-over-cap count; embeds the fixed
   `PreviewParams` blob per entry); `persist.rs` stores it in a new private
   `main.darkroom_history` table (separate from `darkroom_preview`, so the
   backward-compat params path is undisturbed); `AutoSave` now persists params +
   stack (its debounce covers new entries — the 700ms recorder fires first — and
   cursor moves via re-render), and flush force-records an in-flight edit only
   when it differs from the cursor entry (never truncates the redo tail on a
   clean close). Reopen restores the stack at its saved cursor (resume where you
   left off); old dbs (params row, no history) fall back to a fresh single-entry
   stack. Snapshots stay **session-only** by design (pixel-heavy, transient — as
   in darktable). **m4-6: scale-locked snapshot wipe** — replaced the approximate
   side-by-side compare (two independently-letterboxed `Picture`s) with a
   darktable-style **wipe**: a transparent `DrawingArea` layered over the live
   `Picture` via a `gtk4::Overlay` paints the selected snapshot (a cached cairo
   `Rgb24` surface) into the *same* `ContentFit::Contain` rect the live image
   occupies, clipped left of a draggable divider; the right side stays transparent
   so the live image shows through (`draw_wipe`). The alignment invariant is the
   Overlay's *equal child allocation* + a **shared** `preview::contain_rect`
   (extracted from `map_widget_to_image`, so picker and wipe letterbox
   identically) — a feature sits at the same panel pixel across the divider. A
   `GestureDrag` moves the divider (`wipe_fraction` clamps to the letterbox); the
   overlay is `can_target(false)` when idle so picker clicks fall through to the
   live image, `true` only while comparing. Pixel packing is the pure, headless-
   tested `preview::pack_rgb24` (R,G,B→cairo native-endian B,G,R,x, greyscale
   replication, short-buffer defence); the cairo surface is built once per
   selection. `WipeCompare::{show,clear}` own the overlay; the
   `render_preview` sole-writer invariant on the live `Picture`/`last_render` is
   untouched (the wipe owns a separate widget). The **history + snapshots cluster
   is complete**. **m4-7a: pure export model + CLI-contract fix** — extracted the
   export format/quality/resize model and the `darktable-cli` argv construction
   into a headless-tested `export.rs`, fixing two silent no-ops in the old export
   dialog: (1) JPEG quality was passed as a bare `--quality`, which the CLI does
   not parse (the value was swallowed as the positional output path) — now routed
   through the trailing `--core --conf`; (2) the quality conf was keyed on the file
   *extension* (`jpg`/`tif`) but the imageio core keys it on the *module* name
   (`jpeg`/`tiff`) — added `ExportFormat::module_name()` and key on it. Also emits a
   truthful `plugins/imageio/format/tiff/bpp=16` for the "TIFF 16-bit" option and
   collapses to a single trailing `--core` block. `cli_args`/`fit_within` are pure
   and unit-tested (conf-key pinning, `--core`-once, resize aspect math, ALL/
   from_index drift guard). **m4-7b: export panel + output templating** — a
   reusable `ExportPanel` libadwaita widget over that model (format combo, JPEG-
   quality spin greyed unless the format honours it, a resize box — limit switch +
   max width/height + allow-upscale with sub-controls greyed when off — and an
   output-path template), hosted in the export dialog for now (dockable later).
   Output paths come from a pure, headless-tested `expand_output_template` that
   owns `$(FILE_FOLDER)`/`$(FILE_NAME)`/`$(SEQUENCE)` and passes other `$(…)`
   tokens through to the CLI's `dt_variables`; it returns an *extension-less* path
   (`--out-ext` appends the real one, per the verified main.c:724-731 contract).
   Review-hardened for data safety: `batch_output_template` appends `_$(SEQUENCE)`
   when exporting >1 image with a template lacking it, so same-stem sources (a
   RAW+JPEG pair → `exports/IMG`) can't silently overwrite — the CLI's rename
   hinges on the disk `onsave_action`, which `darktable-cli` never sets; an empty
   `$(FILE_FOLDER)` falls back to `.` (no rooting at `/`); and the export counts
   per-image failures, surfacing "Exported X of N (Y failed)" instead of a green
   toast over zero written files. The remaining m4 panel is **tagging**.
5. *Cargo-native build* — once UI + pipeline run from Rust, retire CMake.

The UI work is largely independent of the per-IOP loop ports and can proceed in
parallel; milestone 2 is where the two streams converge.

---

## Architecture overview

```
+-------------------------------------------+
|              GTK4 UI shell (Rust)         |  Phase 3  in progress
|  lighttable . darkroom . panels . dialogs |
+-------------------------------------------+
|           Core services (Rust)            |  Phase 2  complete
|  collection . tags . history . metadata   |
+-------------------------------------------+
|          Image pipeline (Rust)            |  Phase 1  at boundary
|  pixelpipe . IOPs . demosaic . OpenCL    |
+-------------------------------------------+
|    C FFI shim (darkroom-sys)              |  Phase 0  complete
+-------------------------------------------+
```

---

## Goals

- Memory safety (eliminate buffer-overflow / use-after-free class of bugs)
- Modern UI toolkit: GTK4 + libadwaita via `gtk4-rs` 0.9+
- Cargo-native build: `cargo test`, `cargo bench`, `cargo clippy` in CI
- Keep existing Lua scripting API (via `mlua`)
- Keep OpenCL GPU pipeline (`opencl3` crate)
- End state: `cargo build --release` produces the full binary; CMake deleted
