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
   unit-tested `initial_params`. Remaining: (a) float `BaseImage` (unclipped
   highlights into sigmoid + no 8-bit round-trip); (b) higher-quality
   demosaic (PPG/RCD/VNG are migrated; box3 is the current baseline); (b) a
   float `BaseImage` so the preview skips the 8-bit round-trip; (c) ROI/(w,h)
   signature + geometry-aware IOPs; (d) OpenCL.
3. *Darkroom interactions* — zoom/pan, histogram (**done, ui-16**),
   before/after toggle (**done, ui-17**), reset-all (**done, ui-19**), colour
   picker (**done, ui-20**: click-to-sample the processed pixel; pure
   coordinate-mapping + sampling helpers in `preview.rs`).
4. *Remaining views/panels* — port src/libs panels (history stack, snapshots,
   tagging, export) and the other views.
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
