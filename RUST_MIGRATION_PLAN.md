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
   migrated); (c) ~~ROI/(w,h) signature + geometry-aware IOPs~~ **STARTED — m4-73**
   (the `(width,height)` signature + first spatial stage, Sharpen); (d) OpenCL.
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
   toast over zero written files. **m4-8: tagging panel.** **m4-8a** — filter the
   lighttable by tag: new `darkroom-db::tags::tag_list_with_counts` (all user tags
   with attached-image counts via `LEFT JOIN`, excluding pipe-namespaced
   `darktable|…` internal tags, ordered by name; unit-tested) + `lighttable_load_
   by_tag` (3-table JOIN filter). The left collections panel grows a conditional
   "Tags" section (rendered only when the library has user tags) sharing one scroll
   with Collections; tag rows encode the tag id in the GTK widget-name and clicking
   one filters the grid. **m4-8b** — the Collections and Tags lists (two
   `SelectionMode::Single` boxes) now clear each other's selection on activate (weak
   refs, no widget cycle) so a filter never leaves a stale highlight implying an
   AND; "All images" is the clear-tag-filter path; tag read faults are logged (a
   corrupt/locked catalog now reads differently from "no tags"). **m4-8c** — the
   Tags list now live-refreshes: the left panel became a `LeftPanel` struct with
   a stable `tag_box` and a `refresh_tags()` that rebuilds only its rows (so the
   folder↔tag handlers stay valid) and toggles the section's visibility; the
   metadata panel exposes an `on_tags_changed` notify (the canonical "tags
   mutated" hook) wired in lib.rs to fire `refresh_tags()` after each attach, so
   new tags / counts appear without restarting. Known non-blocker: a selected
   tag's highlight drops on refresh (the grid filter state lives outside the box,
   and attach doesn't re-run the filter) — to fold in when attach re-filters the
   grid. **m4-8 done; m4-1..m4-8 complete the original Milestone 4 panels.**
   **Tag-management depth (follow-on):** **m4-9** — the metadata-panel tag chips
   gained an inline ✕ detach button (`tag_detach` DAO), routed through the same
   `on_tags_changed` hook so left-panel counts refresh too; `load_tags` now
   returns `(id, name)`. **m4-10** — right-clicking a left-panel tag row opens a
   context popover with inline Rename (`tag_rename`; UNIQUE clash logs + no-op)
   and a confirmed Delete (`adw::AlertDialog` → `tag_delete`); the per-row gesture
   captures weak refs + db path (rehydrates a transient `LeftPanel` per click) to
   avoid a tag_box↔row↔gesture↔LeftPanel strong-ref cycle. Both refresh the list.
   **m4-11** — bidirectional tag-change refresh: `LeftPanel` gained a mirror
   `on_tags_changed` notify fired by `rename_tag`/`delete_tag`, wired in lib.rs to
   `MetadataPanel::refresh_tags_display()` so a left-panel rename/delete updates
   the current image's chips live (and attach/detach still refreshes the left
   list). Loop-free by construction: the notify fires ONLY from user mutation
   handlers, never from a rebuild, so each mutation yields exactly one cross-panel
   refresh. **m4-12** — re-run the active tag filter on tag mutation (closes the
   m4-8c non-blocker): a shared `active_tag: Rc<Cell<Option<u32>>>` tracks the
   currently-filtering tag id (set by the folder→None / tag→Some click handlers
   and the search/import→None paths); both `on_tags_changed` callbacks re-run
   `lighttable_load_by_tag` *only* when a tag filter is active, so detaching the
   filtered-on tag drops the image while ordinary tagging under All/Folder/Search
   leaves the grid (and selection) untouched. Deleting the filtered-on tag clears
   the now-dangling filter (guards against SQLite id-reuse) via the unit-tested
   `next_active_tag` helper and reverts the grid to "All images". **m4-13** —
   observability parity: `add_tag_to_image`/`load_tags` now log structural DB
   faults (open / image-lookup / attached-read / tag-create) like
   `detach_tag_from_image` instead of silently swallowing them, splitting the old
   catch-all `_`/`else`/`unwrap_or_default` into explicit `Ok(None)` (silent, an
   uncatalogued image) vs `Err` (logged) arms — no behaviour change. **m4-14** —
   preserve grid selection across the m4-12 reapply: the active-tag-filter reload
   now captures the selected image path (`lighttable::selected_path`) before
   reloading and restores it after (`reselect_path`, backed by the unit-tested
   pure `index_of_path`), so an unrelated attach/rename under a tag filter no
   longer yanks the user back to index 0; if the image left the grid (detached
   the filtered-on tag) the default index-0 selection stands. Scoped to the
   reapply path ONLY — folder/tag/search/import are deliberate view changes where
   resetting to the first image is expected. **m4-15** — hierarchical tag display
   (slice 1 of `a|b` tags): the left-panel Tags list now renders the flat tag
   list as an indented tree via the pure unit-tested `flatten_tag_tree` (splits
   names on `|` darktable-style, alphabetises children, synthesises virtual parent
   rows for path prefixes with no tag of their own, skips empty/malformed
   segments). Real tags keep their exact-tag click filter + count + rename/delete
   popover UNCHANGED (so `active_tag`/m4-12 is untouched); virtual parents render
   dim, non-selectable and inert (no count/menu). **m4-16** — hierarchical tag
   prefix filtering (slice 2): clicking any tag-tree node now filters the grid to
   that tag **plus all descendants**. The new loader
   `lighttable_load_by_tag_prefix` JOINs `data.tags` and matches `t.name = ?1 OR
   t.name LIKE ?2 ESCAPE '\'` (`?2 = escape_like(prefix)||'|%'`, `SELECT DISTINCT`),
   replacing the id-based `lighttable_load_by_tag`; the unit-tested `escape_like`
   neutralises `%`/`_`/`\` so a tag name can't widen the match. `active_tag`
   widened `Rc<Cell<Option<u32>>>` → `Rc<RefCell<Option<String>>>` (the full path);
   virtual parents became activatable (every row encodes its `full_name` in the
   widget-name, guarded by `!is_empty()` instead of `parse::<u32>()`). The m4-12
   dangling-delete special-case retired (a path filter has no SQLite id-reuse
   hazard) — `next_active_tag` + the `active_tag`/`lt_model` `LeftPanel` fields
   removed; `delete_tag` just refreshes + fires the notify, and the wired reapply
   re-runs the prefix (surviving descendants stay; deleting the exact filtered-on
   leaf collapses to the empty placeholder). **m4-17** — surface grid-result
   truncation: all four lighttable loaders silently cut at `LIMIT 2000`, which
   m4-16's subtree-gathering prefix made far likelier to hit. Now a
   `const GRID_CAP = 2000`; every loader queries `LIMIT GRID_CAP + 1` so a full
   page is distinguishable from an over-full one, routed through the pure
   unit-tested `cap_rows` (truncate + trailing notice "(showing first 2000 —
   refine your filter)") and the shared `fill_grid` tail (capped rows + notice +
   empty placeholder). The notice carries no `/`, so the grid's
   selection/activation/export guards skip it like the empty-state rows. **m4-18**
   — segment-only hierarchical rename. The rename popover used to edit the full
   `parent|child` path and call an id-based single-row rename, which orphaned a
   parent's descendants (rename `places` and `places|Italy` dangled). Now the
   popover pre-fills only the node's own last segment (with a "Renames this tag
   and any sub-tags" caption); the pure `respliced_tag_path` re-attaches the fixed
   parent prefix (rejecting blank, unchanged, or a `|`-containing segment — the
   last would re-parent/deepen the tree and could let the rewrite self-collide),
   and `darkroom_db::tags::tag_rename_subtree` rewrites the node **plus every
   descendant** in one atomic `UPDATE` (`SET name = ?new || substr(name,
   length(?old)+1)` over `name = ?old OR name LIKE ?old||'|%'`). `length()` is
   SQLite's char count so multi-byte prefixes rewrite at the right offset; the
   `|`-anchored LIKE excludes look-alikes (`places|Italian`); a UNIQUE-name clash
   ABORTs the whole statement (no partial rename, no merge) and is logged
   best-effort. NB: this is a destructive op with **no undo** (the C side still
   owns undo) — flagged for when undo lands in Rust. Known follow-ups: surface a
   collision to the user (this panel has no toast access; metadata panel does);
   tag merge-on-collision; a `with_image_id` helper if a fifth tag op appears.
   **m4-19** — colour-label DAO layer (`darkroom-db/src/colorlabels.rs`), the
   tested core for a new lighttable feature. darktable's 5 per-image colour labels
   (red/yellow/green/blue/purple, 0–4) live in `main.color_labels(imgid, color)`
   with a `UNIQUE(imgid, color)` index, so an image carries any **subset** (unlike
   the single star rating in `images.flags`). DAOs mirror C `colorlabels.c`:
   `color_labels_get` folds rows into a 5-bit mask (`mask |= 1 << c`, dropping
   out-of-range `color`s by design); `color_label_set` (`INSERT OR IGNORE`,
   idempotent), `_remove`, and `_toggle` (read-then-opposite-write) all reject a
   `color >= COLOR_COUNT` as a no-op — closing the write/read asymmetry (no "ghost"
   row `_get` would silently drop) and guarding `1 << color` against debug
   overflow; `_remove_all` clears one image. 8 unit tests (in-memory fixture). The
   UI consumer (a colour-dot row per lighttable cell, click-to-toggle, resolving
   imgid via the existing `image_get_id_by_path`) is the next slice (m4-20).
   **m4-20** — colour labels in the lighttable. Each grid cell gains a 5-dot row
   below the star row; dots render via a pure unit-tested `color_dot_markup(idx,
   lit)` (Pango `<span foreground>●</span>` — own hue when set, dim grey when not,
   out-of-range→grey, no markup-injection surface since only fixed hex constants
   are interpolated). Clicking a dot toggles that label off-thread
   (`gio::spawn_blocking` → `color_label_toggle`) and repaints from the returned
   mask. **GTK cell-recycling correctness** (architect-caught, fixed for the star
   row too): (1) the wire_* helpers run inside `connect_bind`, which fires on every
   recycle, so they now `clear_click_gestures` (strip prior `GestureClick`s via
   `observe_controllers`) before adding one — else a cell accumulated one
   stale-path gesture per bind and one click fanned out into N writes (far worse
   for the *relative* colour toggle than the *absolute* star set); (2) every
   async-painted widget is stamped with its bound path via `set_widget_name`
   (unconditionally, before the placeholder early-return) and each async read bails
   on resolve if the name no longer matches — so a slow read can't smear image A
   onto a cell recycled to image B (or to a placeholder). Display-bound wiring
   untested by discipline; the pure markup core has 3 unit tests. ui 98 tests.
   **m4-21** — colour-label filtering in the left panel. A new "Colours" section
   adds a third `SelectionMode::Single` ListBox (`color_box`) with five fixed rows
   (one per label, each a `color_dot_markup(idx, true)` swatch + name from a pure
   unit-tested `color_filter_name`); clicking a row runs the new
   `lighttable_load_by_color` loader (JOIN `main.color_labels` `WHERE cl.color=?1`,
   the GRID_CAP+1/`fill_grid` contract, no `DISTINCT` thanks to the
   `UNIQUE(imgid,color)` index). The three filter boxes (folders / tags / colours)
   are mutually exclusive: activating any one `unselect_all`s the other two and
   clears the shared `active_tag`, so no stale highlight implies an AND that isn't
   running. The colour filter is deliberately **not** wired into the
   `reapply_tag_filter` machinery (it clears `active_tag` to `None`) — toggling a
   label on a grid cell while colour-filtered doesn't auto-refresh the grid, the
   same non-reactivity a folder filter already has, avoiding a colour-changed
   callback up to lib.rs. `color_dot_markup`/`COLOR_COUNT` were promoted to
   `pub(crate)` so panel and grid share one source of truth for the hues. Known
   carry-forward (TODO in code): a name-search/import in lib.rs clears the active
   filter but can't reach the boxes to drop their highlight — pre-existing for
   `tag_box`, now also `color_box`; clean fix is a `LeftPanel::clear_filter_
   highlights()`. ui 100 tests.
   **m4-22** — `LeftPanel::clear_filter_highlights()` (DONE). Closes the m4-21
   carry-forward: a method that `unselect_all`s the folder / tag / colour boxes,
   wired into the three lib.rs grid-takeover paths (name-search, import button,
   import action) which clear `active_tag` but couldn't reach the boxes. Stored the
   `list_box` + `color_box` handles as `LeftPanel` fields (only `tag_box` was). The
   **supersede invariant** (architect): call it exactly on paths that null
   `active_tag`; NOT from a preserve-filter reload like `reapply_tag_filter`.
   Collateral: the `append_tag_tree_row` secondary-click hack reconstructs a
   transient `LeftPanel` from weak upgrades to call `&self show_tag_menu`, so it now
   captures two more weak refs the menu ignores — deferred clean fix is making
   `show_tag_menu` a free fn. Architect SHIP. ui 100 tests, clippy unchanged.

   **m4-23** — colour-label keyboard shortcuts (darktable F1–F5). An
   `EventControllerKey` on the `GridView` maps plain F1..F5 → colour 0..4 and
   toggles that label on the *selected* image, repainting just the touched cell's
   dot row in place (no full reload — scroll position and other cells' in-flight
   async loads are untouched). Pieces: pure `fkey_to_color` seam (unit-tested,
   display-free); `toggle_selected_color` (selected_path → off-thread DB toggle →
   in-place repaint); `repaint_color_dots_for_path` → `find_color_box_for_path`, a
   DFS of the grid's realized cells. The keyboard path holds no `colors_box` ref
   (unlike `wire_color_clicks`), so it locates the row by the bind-time
   `widget_name` stamp. Architect nit (SHIP-WITH-NITS): that stamp is shared by a
   cell's thumb/stars/colours and grid-path uniqueness is only an *assumption*
   (`index_of_path`), so the finder now requires BOTH thumb AND colour row to carry
   `path`; cross-cell uniqueness still rests on the loaders' distinct rows, worst
   case a self-healing transient repaint, DB write always correct. Controller on
   the grid scopes it to lighttable focus (not the darkroom page / search entry);
   Ctrl/Alt + F-key propagate. ui 102 tests, clippy unchanged. See
   `reference_gtk_signallistitemfactory_recycling`.

   **m4-24** — colour labels in the darkroom (single-image) view (DONE). A 5-dot
   colour row in the `adw::HeaderBar`, `pack_end`ed left of Export, mirroring the
   lighttable's grid dots. Reuses the lighttable colour toolkit as one source of
   truth: `build_color_dots_box()` (newly extracted, also adopted by the grid
   factory) builds the row; `query_color_labels` seeds the lit mask synchronously
   at open (consistent with the sibling `load_history`/`load_saved` sync reads);
   `wire_color_clicks` wires click-to-toggle. Key reuse seam: `wire_color_clicks`'
   repaint is guarded by the lighttable's cell-recycle check `widget_name() ==
   path`, so the static header box is stamped `widget_name = file_path` for the
   guard to pass (an unstamped box silently skips the repaint). Architect
   (SHIP-WITH-NITS) P2 fixes applied: colour-label DB connections now take a 3s
   `busy_timeout` (`open_colorlabels_conn`) so a toggle doesn't silently drop on
   `SQLITE_BUSY` against the view's autosave writer; the dot row gets an
   accessible Group role + "Colour labels" name (both views inherit it). Known
   gap (deferred): no cross-view live sync — a toggle in the darkroom leaves an
   already-realized lighttable cell stale until it rebinds. ui 102 tests, clippy
   unchanged.

   **m4-25** — cross-view colour-label sync (DONE). Closes the m4-24 known gap: a
   label toggled in the darkroom single-image view now shows in the lighttable cell
   on return, instead of staying stale until GTK rebinds. On `NavigationView` pop,
   re-query the DB for the popped image's mask and repaint that cell in place. New
   `refresh_color_dots_for_path(grid, db, path)` — the same off-thread-read →
   in-place-repaint shape as `toggle_selected_color`, minus the write (queries
   `query_color_labels`, reuses `repaint_color_dots_for_path`/`find_color_box_for_path`).
   The darkroom page is tagged with its image path (`set_tag`) at push so the
   `connect_popped` handler recovers which cell to refresh regardless of dismissal
   route (back / Escape / swipe / programmatic); the handler guards on `/` like the
   loaders. Self-healing failure mode: if the cell isn't realized when the read
   resolves, the repaint is a no-op and the next bind paints from the DB. Architect
   (SHIP-WITH-NITS): both nits cleared without code change — the toggle-then-fast-
   dismiss read race can't blank the dots because `open_colorlabels_conn`'s 3s
   `busy_timeout` (m4-24) makes the read wait out the in-flight write; dots-only
   scope is correct since the darkroom view has no star/rating control. ui 102
   tests, clippy unchanged. **Colour-label arc m4-19/20/21/23/24/25 now complete.**

   **m4-26** — multi-colour filter (Any/All) (DONE). Turns the left-panel colour
   filter from single-colour (m4-21) into a multi-select mask with an Any (OR) /
   All (AND) combine mode. Two slices. **m4-26a** (lighttable): pure unit-tested
   `colors_from_mask` + `build_color_mask_query(mask, match_all)` (None for empty
   mask; OR = `SELECT DISTINCT … WHERE cl.color IN (…)`, AND = `… GROUP BY i.id
   HAVING COUNT(DISTINCT cl.color) = N`; colour ints derived from the mask so the
   inlined `IN`-list is injection-safe) feeding `lighttable_load_by_color_mask`
   (empty mask → `lighttable_load_from_db`/show-all). Replaces the single-colour
   `lighttable_load_by_color` (removed — the one-bit mask subsumes it). 6 new
   tests. **m4-26b** (panels): `color_box` is now a `gtk4::Box` of 5 independent
   `CheckButton`s (index in `widget_name`) + a "Match any"/"Match all" `ToggleButton`.
   A shared `reload_colors: Rc<dyn Fn()>` reads the mask off the checks
   (`color_mask_from_box`), clears folder+tag highlights + `active_tag`, and loads
   by mask+mode. `color_suppress: Rc<Cell<bool>>` (new field) gates the checks'
   `connect_toggled` during programmatic resets; folder/tag activation + the lib.rs
   `clear_filter_highlights` takeover path now call `clear_color_checks` (save/
   restore suppress, nesting-proof). Mutual exclusion stays symmetric; mode is
   sticky across folder/tag switches (a preference, not a filter). Architect
   (SHIP-WITH-NITS): SQL AND/OR correct (GROUP BY i.id deterministic — PK +
   functional dep; UNIQUE(imgid,color) makes the DISTINCT count exact); suppress
   sound under GTK's synchronous single-thread model; no ref cycle (reload_colors
   holds weak widget refs + leaf Rcs); applied the actionable nit (drop orphaned
   `lighttable_load_by_color`) + the nesting-proof suppress. The transient-LeftPanel
   reconstruction in the tag-row gesture now threads `color_suppress` too — the
   deferred clean fix (make `show_tag_menu` a free fn) is now a real backlog item,
   not just a code comment. ui 108 tests, clippy unchanged. **Colour-label arc
   m4-19/20/21/23/24/25/26 now complete.**

   **m4-27** — `TagPanel` extraction / tag-menu cleanup (DONE). Pure refactor (no
   behaviour change) that retires the "transient-`LeftPanel` reconstruction" smell
   the m4-22/26 reviews flagged: the per-tag-row secondary-click gesture used to
   rebuild a whole `LeftPanel` (supplying folder/colour-filter fields the rename/
   delete menu ignores) to call `&self show_tag_menu`. Extracted a private
   `TagPanel` struct holding EXACTLY the tag fields (`tag_box`, `tags_header`,
   `tags_sep`, `db_path`, `on_tags_changed`) and moved the whole tag-mutation
   cluster onto `impl TagPanel` (`set_on_tags_changed`, `fire_tags_changed`,
   `refresh_tags`, `append_tag_tree_row`, `show_tag_menu`, `rename_tag_subtree`,
   `confirm_delete_tag`, `delete_tag`). `LeftPanel` now holds `tags: TagPanel` and
   delegates its unchanged public API (`refresh_tags`/`set_on_tags_changed`;
   `clear_filter_highlights` reaches `self.tags.tag_box`). The gesture now rebuilds
   a `TagPanel` from weak refs — nothing ignored; the `Clone`-and-reconstruct shape
   is kept (a single strong `TagPanel` capture would re-form the cycle). `confirm_
   delete_tag` presents its dialog off `self.tag_box` (any in-tree widget resolves
   the root window) since `TagPanel` holds no top-level widget. Architect
   (SHIP-WITH-NITS): behaviour-preservation clean pass on all runtime questions;
   fixed the one blocking nit (the inserted `TagPanel` doc had orphaned `LeftPanel`'s
   rustdoc — restored so the `pub` struct keeps its docs). ui 108 tests, clippy +
   `cargo doc` clean.

   **m4-28** — star rating in the darkroom single-image view (DONE). Star analogue
   of m4-24 (header readout + click) + m4-25 (pop-sync). A 5-star row in the darkroom
   `HeaderBar`, `pack_end`ed left of the m4-24 colour dots (so L→R: stars, colours,
   Export), reusing the lighttable star toolkit as one source of truth: extracted
   `build_stars_box()` (factory setup now calls it; also adds an `AccessibleRole::
   Group` + "Star rating" a11y label, inherited by every grid cell too) and bumped
   `set_stars`/`wire_star_clicks`/`query_rating` to `pub(crate)`. Rating read
   synchronously at open. **Unlike the colour dots, the star box needs NO
   `widget_name` stamp**: `wire_star_clicks` repaints synchronously then persists
   off-thread with no async read-back, so it has no recycle guard to satisfy.
   Cross-view sync: generalised the m4-23 cell finder into `find_cell_row_for_path
   (root, path, child_index)` (stars = idx 2, colours = idx 3; the dual thumb+row
   `widget_name==path` guard holds for both), added `refresh_stars_for_path` (star
   sibling of `refresh_color_dots_for_path`), and the m4-25 `connect_popped` handler
   now refreshes BOTH. Architect (SHIP-WITH-NITS): applied the one data-safety fix —
   new `open_rating_conn` gives `query_rating`/`save_rating` a 3s `busy_timeout`
   (mirrors `open_colorlabels_conn`), since m4-28 adds a rating WRITE from the
   darkroom header concurrent with the autosave writer → same `SQLITE_BUSY`
   silent-drop race m4-24 fixed for colours; returns `Result` so `save_rating`
   `?`-propagates. Toggle-to-zero intentionally NOT added (parity with the existing
   lighttable star control; if wanted, do once in shared `wire_star_clicks` later).
   The toggle-during-autosave race is display/timing-bound → not headless-verifiable
   (busy_timeout narrows the window; the write result is still `let _`-discarded, as
   on the colour path). ui 108 tests, clippy unchanged. **The darkroom single-image
   view now shows + edits both rating and colour labels, syncing both to the grid.**

   **m4-29** — star-rating keyboard shortcuts (0–5) in the lighttable (DONE).
   Direct analogue of the m4-23 F1–F5 colour keys, on the same grid
   `EventControllerKey`. Pieces (lighttable): pure unit-tested `digit_to_rating`
   (top-row `Key::_0.._5` AND keypad `Key::KP_0..KP_5` → rating 0–5, else None;
   keypad arm assumes NumLock on — documented); `set_selected_rating` (star sibling
   of `toggle_selected_color`, but an ABSOLUTE set — digit k → rating k, matching
   the star click handler + darktable — so it repaints from the KNOWN rating, no
   read-back → no blank-on-busy risk); extracted `repaint_stars_for_path` (find+set,
   now shared by `set_selected_rating` + `refresh_stars_for_path`). lib.rs: the key
   handler tries `fkey_to_color` first (Stop on match) then `digit_to_rating` (Stop),
   else Proceed; Ctrl/Alt still propagate; digit `0` clears the rating (a superset of
   the click handler, which only reaches 1–5). Architect **SHIP** (clean, faithful
   mirror); applied the one doc nit (NumLock note). ui 110 tests (2 new), clippy
   unchanged.

   **m4-30** — harden off-thread metadata writes (the whole class, one increment
   for symmetry) (DONE). All four write sites (star click, rating key, colour click,
   colour key) previously `let _`-discarded / `unwrap_or(0)`'d the off-thread
   result, so a `spawn_blocking` `JoinError` (task panic) AND a post-`busy_timeout`
   rusqlite `Err` were both silent; rapid same-image inputs also raced in the glib
   pool with no serialization (a colour toggle is a read-modify-write → lost
   update). Fix: `path_write_lock(path) -> Arc<Mutex<()>>` (process-wide registry,
   `OnceLock<Mutex<HashMap<..>>>`) + one `serialized_write(path, closure)` choke
   point all four route through — it takes the per-path lock *inside* the
   `spawn_blocking` worker (never on the main loop, so no UI stall) and logs the
   `JoinError`; `save_rating` `Err` logged at the two rating sites (`toggle_color_label`
   already logs its own). Guarantee (architect-clarified): per-write atomicity +
   repaint-tracks-committed-value (completion order == commit order, so UI never
   diverges from DB); NOT last-input-wins (`Mutex` unfair, `spawn_blocking` not
   FIFO) — harmless for commuting colour bit-flips, acceptable for ratings. Colour
   sites now skip the repaint on a worker panic (`None` → mask unknown, don't clear
   labels still in the DB; next rebind paints from DB). Architect **SHIP** after
   one must-fix doc reword (drop the implied input-ordering claim); deferred idea
   filed inline: a single DB-writer thread would bound pool use + coalesce. ui 111
   tests (1 new: `path_write_lock` same-Arc/distinct-Arc), clippy unchanged.

   **m4-31** — surface a tag rename collision to the user (closes the long-standing
   m4-18 S2 backlog) (DONE). `tag_rename_subtree`'s UNIQUE clash used to only
   `eprintln!` then unconditionally `refresh_tags`/`fire_tags_changed` — a silent
   no-op rename plus a spurious active-filter reapply. Now branch on the DAO
   result: `Ok` → refresh+notify as before; `Err` → log + `show_rename_error`
   (a dismiss-only `adw::AlertDialog` presented off `tag_box`, mirroring
   `confirm_delete_tag`) and NO refresh (the atomic UPDATE rolled back → nothing
   changed; single-autocommit-statement so `Err` ⇒ zero rows changed, no
   false-skip). Message from pure `rename_failure_message(err, new_full)`:
   `ConstraintViolation` → "would clash with an existing tag" (deliberately NOT
   "`new_full` already exists" — the clash may be a *descendant's* rewritten path,
   architect Q6 honesty fix), else generic. Architect **SHIP**, no blockers; folded
   in the reworded message + a darkroom-db regression test pinning the load-bearing
   invariant (parent rename onto an existing name aborts, leaving parent+descendant
   intact). Primary `code == ConstraintViolation` is sufficient (only the name
   UNIQUE index can trip this UPDATE). db 85 tests (+1), ui 112 tests (+1),
   clippy unchanged.

   **m4-32** — rename-failure inline UX (supersedes m4-31's modal) (DONE).
   Implements the exact deferred design from the m4-31 review: run the rename
   FIRST, branch on the result. Split the DB write into a UI-free
   `write_tag_rename -> rusqlite::Result<()>` (open db + atomic DAO UPDATE, touches
   no widgets); the popover's `do_rename` closure orchestrates: `Ok` → `popdown()`
   BEFORE `refresh_tags` (the ordering trap — dismiss before the refresh removes the
   popover's parent row, else orphaned subtree) + notify; `Err` → keep the popover
   open, set+show an inline `error_label` (`.error`/`.caption`, hidden until
   failure) via the pure `rename_failure_message`, `grab_focus`+`select_region(0,-1)`
   so the user corrects in place. Removed the m4-31 `show_rename_error` AlertDialog +
   the old `rename_tag_subtree`; repointed the `rename_failure_message` doc intra-link
   (m4-27 lesson: `cargo doc` after doc-link changes — verified no NEW broken-link
   warning). Added clear-on-`changed` (label hides once the user edits — standard
   inline-validation convention; closure captures only a weak label ref). Architect
   (Opus) **SHIP, no blockers** (faithful; failure-path lifetime clean —
   `write_tag_rename` drops its own conn, no borrow across the UI mutation). No new
   test (pure `rename_failure_message` already covered; orchestration is a trivial
   display-bound 2-arm match — architect agreed no seam worth extracting). ui 112
   tests, clippy unchanged. **Deferred follow-ups the architect flagged (neither
   blocking, can land together — both touch the popover handlers):** (a) **pre-existing
   cyclic-capture popover leak** — the rename/delete button + entry-activate handlers
   capture `pop`/`entry` STRONGLY, so the popover subtree leaks on every right-click
   (self-cycle prevents disposal); fix by weak-capturing `pop`/`entry`/`err_lbl` and
   `upgrade()` inside, matching `append_tag_tree_row`'s existing row-gesture
   discipline (`lp` can stay strong — its `tag_box` lives outside the popover). NOT a
   regression from m4-32. (b) **a11y**: associate the error label with the entry
   (`accessible::Relation::ErrorMessage` + `State::Invalid`) so screen readers
   announce the reason on the focus move — deferred to get the version-sensitive
   gtk4-rs signature right rather than guess.

   **m4-33** — popover handler weak-capture leak fix + a11y error relation
   (closes both m4-32 deferred follow-ups in one increment over the same handlers)
   (DONE). **(a) Leak:** the tag rename/delete popover is `set_parent`ed into a
   `tag_box` row, and its `do_rename`/delete closures (stored on the popover's own
   buttons + entry) captured `pop`/`entry`/`err_lbl` STRONGLY → a self-cycle
   (popover→button→handler→closure→pop→popover) kept the whole popover alive after
   `connect_closed`'s `unparent` — one leaked popover per right-click. Fix:
   `downgrade()` those three to weak in every subtree-resident handler,
   `upgrade()`-or-`return` inside; `lp` (TagPanel) stays strong — its `tag_box`
   lives OUTSIDE the popover subtree, and `unparent` severs the row→popover edge so
   the `lp` arm points strictly outward, never cycling back. Matches
   `append_tag_tree_row`'s existing `downgrade()` discipline (explicit weak/upgrade,
   NOT `clone!` — CLAUDE.md flags that deprecation as out-of-scope). **(b) a11y:**
   `entry.update_relation(Relation::ErrorMessage(&[error_label]))` set once
   (permanent association) + `State::Invalid(True)` on failure / `(False)` on
   `changed`, the ARIA contract (errormessage only surfaced while invalid) so a
   screen reader announces the reason on the error-path `grab_focus`. **gtk4-rs
   0.9.7 gotcha (compiler-verified): `Relation::ErrorMessage` takes `&[&Accessible]`
   (a many-targets SLICE), not the single ref first assumed — upcast the label into
   a 1-element slice.** Architect (Opus) **SHIP, correct & complete, no must-fix**;
   traced every subtree handler (cycle fully broken), confirmed the failure-path
   `grab_focus`/`select_region`/`set_text` don't emit `changed` so `Invalid=True`
   isn't self-clobbered, and actively recommended AGAINST gating `Invalid=False`
   (idempotent no-op; a captured bool for zero benefit). No new test (pure
   lifetime/a11y plumbing, display-bound; `rename_failure_message` + DB-rollback
   tests remain the coverage boundary). ui 112 tests, clippy unchanged.

   **m4-34** — camera-RGB → sRGB colour matrix in the preview pipeline (pipeline
   depth, user-chosen direction) (DONE). The Rust preview path (`rawimage::
   to_linear_rgba`, separate from the C FFI port) ignored the camera colour matrix,
   treating camera-native RGB as linear sRGB → wrong colours. Now derive camera→sRGB
   from `rawloader`'s `xyz_to_cam` and apply it after WB. Pure `darkroom-core`, no
   UI/persistence churn. New `srgb_from_cam_matrix(xyz_to_cam:[[f32;3];4]) ->
   [[f32;3];3]` follows dcraw's `cam_xyz_coeff`: `cam_rgb = xyz_to_cam·XYZ_RGB`
   (sRGB→XYZ, dcraw constants) → row-normalise (neutral-preserving) → `mat3_inverse`
   → camera→sRGB; `IDENTITY3` fallback on all-zero (unknown camera) or singular.
   Uses top 3 rows (RGB/X-Trans are 3-colour; 4th row ignored). `apply_color_matrix`
   per-pixel after `apply_white_balance` in `to_linear_rgba` (order = darktable
   WB→input-profile; exact no-op for IDENTITY3, so the synthetic/demo path is
   unchanged). Negatives left unclamped for the scene-linear tone map (darktable
   input-profile behaviour). Architect (Opus) verified the construction line-by-line
   vs dcraw (square-case inverse == pseudoinverse) — **SHIP after one should-fix**:
   the neutral test can't catch a wrong matrix (grey is preserved by construction
   for ANY invertible row-normalised matrix), so added a **golden regression** —
   Canon 5D Mk III `xyz_to_cam` (dcraw `adobe_coeff`) → expected camera→sRGB from an
   independent pure-Python dcraw impl; a transposed multiply / wrong constant /
   inversion bug all diverge from it (the non-symmetric matrix locks multiply
   order). darkroom-core 509→514 tests (+5), clippy clean.
   **KNOWN SIMPLIFICATION (architect-flagged, deferred):** the working space is
   sRGB primaries, so saturation ops (velvia, sigmoid) push colours out-of-gamut /
   negative sooner than darktable's Rec.2020 pipe would — fine for a preview; widen
   to a larger working gamut later.

   **m4-35** — linear Rec.2020 working colour space for the raw preview pipeline
   (pipeline depth; resolves the m4-34 KNOWN SIMPLIFICATION) (DONE). The raw path
   now decodes camera-native RGB into **linear Rec.2020** instead of linear sRGB:
   `srgb_from_cam_matrix` → `rec2020_from_cam_matrix` (same dcraw `cam_xyz_coeff`
   construction, target primaries = Rec.2020), `XYZ_RGB` → `REC2020_XYZ`
   (Rec.2020→XYZ D65, Lindbloom construction from CIE chromaticities), struct field
   `cam_to_srgb` → `cam_to_working`. The wide gamut gives the saturation/tone
   stages (velvia, sigmoid) headroom before clipping. Display seam: new
   `pub REC2020_TO_SRGB` (`inv(sRGB→XYZ)·(Rec.2020→XYZ)`, rows sum to 1 so
   neutrals map exactly) applied in `render_linear_to_srgb8` after
   `Pipeline::process`, just before the sRGB OETF; out-of-sRGB-gamut colours go
   negative and hard-clip at the encode. The `Srgb8`/`apply_pipeline` (JPEG) path
   never enters Rec.2020 — no seam, correctly; export shells out to the C
   `darktable-cli`, untouched. Architect (Opus) traced every buffer consumer:
   **no double- or missed-conversion path**; active stages (exposure, velvia,
   splittoning, monochrome, sigmoid) verified space-agnostic. Both matrices
   verified against an independent exact-arithmetic CIE-chromaticity derivation
   (~4e-5 agreement; residual = D65 rounding convention). Must-fix from review:
   non-grey golden for `REC2020_TO_SRGB` (Rec.2020 red → sRGB
   [1.660, -0.125, -0.018]) since the grey test passes for ANY row-normalised
   matrix; plus a Canon 5D Mk III camera→Rec.2020 golden re-derived offline.
   darkroom-core 514→516 tests (+2), all green in Docker.
   **Deferred (architect-flagged, out of scope):** velvia hard-clamps [0,1]
   pre-sigmoid (pre-existing, loses >1.0 scene-linear data); default monochrome
   weights are Rec.709 luma now applied to Rec.2020 primaries (off by default,
   user-adjustable); `basicadj.rs`/`gamma.rs` carry sRGB/Rec.601 luma constants
   but are not wired into the `Stage` enum — revisit if they ever join the
   pipeline.

   **m4-36** — canonical scene-referred iop stage order in the preview pipeline
   (pipeline depth; supersedes the m4-35-deferred "velvia clamps pre-sigmoid"
   item) (DONE, commit `84abf4a038`). `PreviewParams::to_pipeline` reordered
   from `exposure → velvia → splittoning → monochrome → sigmoid` to darktable's
   canonical v3.0 order `exposure → channelmixer(grey) → sigmoid → velvia →
   splittoning` (`src/common/iop_order.c`). velvia/splittoning are
   display-referred and hard-clamp output to [0,1] (faithful C ports); running
   them BEFORE the sigmoid tone map crushed scene-linear highlights (>1.0), and
   monochrome-last silently discarded velvia/splittoning output (split-toning a
   B&W image was a no-op). Now velvia runs post-sigmoid on display-referred data
   where the [0,1] clamp is correct — so the m4-35 velvia-clamp deferral is
   resolved without lifting the clamp. Docstring cites `channelmixerrgb 39` (the
   scene-referred module) as the ordering reference and notes the legacy
   `channelmixer` port is placed there for the photometric reason (linear
   luminance is the correct domain to tone-map as luminance), plus the
   sigmoid-off raw-path clipping caveat. Tests: `to_pipeline_orders_stages_
   canonically` pins the `.name()` order; `canonical_order_preserves_chromatic_
   highlight_velvia_first_crushes_it` builds both stage orders via
   `Pipeline::with_stages` and asserts the canonical order keeps a chromatic
   scene-linear highlight brighter than velvia-first (a grey pixel can't
   distinguish orders — velvia is identity on greys). Two Opus reviews (2nd read
   the C-port sources): reorder correct, **no blockers**; applied both should-fix
   items (the `channelmixerrgb`-vs-legacy-`channelmixer` citation fix, chromatic
   test) + the mono+splittoning-tint doc note. darkroom-ui 114→115 tests, Docker
   check/clippy/test green, both remotes synced.

   **m4-37** — port RCD (Ratio Corrected Demosaicing) to Rust (pipeline depth)
   (DONE, commit `e40ff6df3c`). New `demosaic_rcd` in
   `crates/darkroom-core/src/rawimage.rs`, a faithful port of darktable's
   `rcd_demosaic` (`src/iop/demosaicing/rcd.c`) — the default high-quality Bayer
   demosaicer, far fewer maze/zipper artefacts than PPG. **Correction to the
   m4-36 candidate note above:** RCD/VNG were NOT actually migrated — only VNG
   *helper* loops existed (no orchestrator) and RCD not at all; this ports RCD
   in full. Keeps darktable's exact 112px tiling (`RCD_TILEVALID=92`): the
   demosaic runs at full sensor resolution *before* the preview downscale, so a
   whole-image scratch would be hundreds of MB while per-tile buffers stay
   ~350 KB. A full-frame `demosaic_ppg` is the base; RCD overwrites each tile's
   valid interior (outermost tiles use `RCD_MARGIN=9`, joins `RCD_BORDER=10`).
   Tiles serial for now. **Intentional divergences from C:** step-1's rolling
   3-row VH window → two full-tile `vsq`/`hsq` buffers (clearer, same result);
   `lpf`/`pq_dir` separate (not aliased); every buffer zeroed per tile (stricter
   than C's partial-tile-only memset). Opus architect review verified all index
   arithmetic / bounds / parity / the step-1 rewrite: **no correctness
   blockers**; applied all 3 should-fix test-coverage additions (interior-tile
   join 250×250, single-tile 50×50, BGGR parity) + nits (lpf half-stride comment,
   step-3 `debug_assert`, hoisted `fc_bayer`). darkroom-core 516→523 tests.

   **m4-38** — use RCD as the Bayer demosaic in the preview pipeline (DONE,
   commit `daf8db6b1d`). One-line default swap: `RawImage::to_linear_rgba`'s
   Bayer branch `demosaic_ppg` → `demosaic_rcd` (X-Trans still Markesteijn). RCD
   falls back to the PPG base for sub-tile frames, so the small-fixture test is
   unchanged. Wiring-only follow-up to the m4-37 review; Docker green
   (core 523, ui 115, clippy clean). RCD runs once per file open, off-thread, at
   full res — a one-time load cost, not per-slider.

   **m4-39** — assemble a native-Rust VNG (Variable Number of Gradients) Bayer
   demosaic orchestrator (DONE, commit `b8fe557afe`). The VNG per-pass *kernels*
   were already migrated as C-ABI fns in `iop::demosaic` (`vng_border`/`_lookup`/
   `_gradient_row`/`_finish`) but had NO orchestrator and the two lookup-table
   builders stayed in C. New `demosaic_vng` in `rawimage.rs` ports
   `vng_interpolate` (`src/iop/demosaicing/vng.c`) natively: `build_vng_lookup`
   (the `lookup[16][16][32]` linear-interp table) + `build_vng_code` (the
   `code[prow][pcol]` gradient streams from the dcraw `terms[]`/`chood[]` tables,
   extracted programmatically as `VNG_TERMS`/`VNG_CHOOD` — grad bytes ≥0x80 are
   negative i8 but only bits 0..7 tested, reproducing C's signed-char→int
   promotion), plus the 3-row ring buffer with C's **2-row-deferred write-back**
   (so the gradient kernel always reads the un-refined base — no read-after-write
   hazard; `[Vec;3]::rotate_left(1)` == C's `brow[(g-1)&3]` rotation). Bayer only;
   RGGB greens split into G1/G2 (`filters4`) then re-merged by the finish pass;
   sub-interior frames fall back to linear-only VNG. **NOT wired as a default** —
   RCD stays the Bayer default (m4-38); this makes VNG available for a future
   demosaic-method selector. Opus review: faithful to C, no P0/P1 (verified
   builders, negative-coord `fcol`, ring rotation, 6×6 bounds, pointer
   lifetimes); applied the 2 P2 clarity comments. darkroom-core 523→527 tests.

   **m4-40..m4-43 — per-image Bayer demosaic-method selector (COMPLETE):** a
   user choice of RCD / VNG / PPG for the raw preview, persisted per image.
   - **m4-40** (`c120823050`): `DemosaicMethod` enum (Rcd default / Vng / Ppg,
     stable discriminants) + `RawImage::to_linear_rgba_with(method)` dispatching
     the Bayer branch; X-Trans ignores it (checked before method). No-arg
     `to_linear_rgba` delegates to Rcd → all callers byte-identical. +2 tests
     (dispatch distinctness on a 40×40 gradient; X-Trans method-invariance).
   - **m4-41** (`b4e09f87cc`): `DemosaicMethod::as_u8`/`from_u8` (stable 1-byte
     codec, unknown→default) + `raw_preview::decode_raw_preview_with(method)`
     (the no-arg wrapper delegates; examples unchanged). +1 codec test.
   - **m4-42** (`8ddafabff6`): persist to a dedicated `main.darkroom_demosaic`
     (imgid PK, method INTEGER) — a SEPARATE table, not a `PreviewParams` field,
     because the method is *decode-time* state (a change re-decodes the raw, not
     re-runs the pipeline) and to keep it out of PreviewParams' Copy /
     history-snapshot / before-after-bypass machinery. +3 tests.
   - **m4-43** (`d6945340a6`): the UI — a `gtk4::DropDown` in the darkroom right
     panel (raw only), seeded from the persisted method; changing it re-decodes
     + persists. Old inline load refactored into `spawn_decode`; a `decode_gen`
     generation guard makes only the newest decode paint (stale-paint guard for
     rapid switching). Opus review 9/10, no blockers; applied N1/N2/N3 (X-Trans
     tooltip caveat, failed-decode comment, stale raw_preview.rs doc fix).
     darkroom-ui 115→119 tests, darkroom-core 527→530.

   **m4-44/m4-45 — geometry primitives (crop + rotate), core done:** a new
   `darkroom_core::geometry` module, kept SEPARATE from the per-pixel
   `pipeline` (every current stage is position-independent, so geometry commutes
   with them and the ping-pong Pipeline stays size-agnostic; caveat noted in
   both modules — revisit once a spatially-varying IOP lands).
   - **m4-44** (`588bfa8a54`): `Crop` (resolution-independent fractional rect;
     `normalized()` clamps + NaN→identity-edge + inverted/sub-MIN_EXTENT axis →
     full; `is_identity()`, `pixel_rect()`) + `apply_crop` (row-slice, no-ops on
     identity/degenerate). Also renamed the colliding `iop::geometry::Crop` stub →
     `CropIop` (Rust-only; `name()` "crop" unchanged). Opus 8/10. 9 tests.
   - **m4-45** (`62134c6665`): `apply_rotate(pixels,w,h,angle)` + `MIN_ANGLE` —
     rotate about centre, EXPAND canvas to the bbox, transparent-black corners,
     bilinear via the shared `interp::compute_pixel4c` (NOT the C rotatepixels
     kernel — architect-endorsed). Positive = CCW. `ceil_dim` sub-pixel epsilon
     fixes a float-`ceil` bbox-inflation bug at near-axis angles (safe ≲22 000px).
     Composes with `apply_crop` for straighten-and-crop. Opus: correct, no
     blocker. 7 tests. darkroom-core 530→545 tests.

   **m4-46/m4-47 — geometry backend + straighten UI (done):**
   - **m4-46** (`d1411ef414`): `geometry::Geometry { crop, angle }` — one value
     for the per-image straighten+crop with `apply(pixels,w,h)` (rotate∘crop, crop
     in the rotated frame) + `is_identity()` + a versioned 21-byte `encode`/`decode`
     codec. Persistence mirrors m4-42: a separate `main.darkroom_geometry(imgid PK,
     geom BLOB)` table + `persist::{load,save}_geometry` (best-effort → default).
     Self-reviewed (mirrors reviewed patterns). core +4 / ui +3 tests.
   - **m4-47** (`a5036e5897`): straighten (rotate) slider wired into the darkroom
     view. `PreviewCtx` gains `pristine` (decoded raw BEFORE geometry) + `geometry`
     (Cell), seeded from `load_geometry` before the first decode. `spawn_decode`
     restructured (raw → store pristine + `apply_geometry_to_base`; JPEG → clear
     pristine, 8-bit base; `decode_gen` guard kept in both). A raw-only "Straighten"
     slider (−45..45°) re-applies `Geometry` to the pristine buffer (cheap resample,
     not a re-decode) + persists, debounced 160ms (cancel-and-rearm) for drag
     responsiveness. **fricktrade-architect review was quota-blocked (weekly limit)
     → self-reviewed; a follow-up architect pass is queued for the reset.**
     darkroom-ui 122→123.

   **m4-48 — interactive crop overlay (DONE):** completes the straighten-and-crop UX.
   - **m4-48a** (`bc646f5746`): pure `crop_overlay` interaction math (fraction
     space): `widget_to_fraction`, `hit_test` (corner>edge>inside), `resize_to`
     (clamp + MIN 2% + no-invert), `translate` (bounded). 4 headless tests.
   - **m4-48b** (`bb071aac0d`): the GTK overlay. Raw-only "Crop" header toggle →
     `PreviewCtx.crop_editing`; `apply_geometry_to_base` shows rotated-uncropped
     while editing (apply_rotate) else the cropped result (geom.apply). A second
     `DrawingArea` over the Picture (mirrors `WipeCompare`, click-through idle),
     `draw_crop` (dim + thirds + rect + 8 handles via `contain_rect`), a
     `GestureDrag` grabbing a handle and `resize_to`/`translate`-ing the drag-start
     crop (no drift) into `ctx.geometry`, `save_geometry` on drag-end. Opus review:
     no blockers; applied the should-fix (entering crop dismisses any active wipe
     compare — mutual-exclusion) + 2 nice-to-haves. **m4-47 deferred review also
     ran: clean, no blocker/should-fix.** darkroom-ui 123→127 tests.
   **m4-49/m4-50 — Rust-native export (DONE): geometry + colour params now reach
   the exported file** (user chose the Rust-render route over post-processing
   darktable-cli). Export previously ALWAYS shelled to `darktable-cli`, which
   develops the raw with darktable's own history and ignores every darkroom-ui
   edit; now the single-image darkroom export renders through OUR pipeline so it
   matches the preview.
   - **m4-49** (`d93b409429`): `export::render_export_rgb8(img, method, geometry,
     params) -> (w,h,rgb8)` — the preview pipeline at full res (demosaic + WB +
     Rec.2020 → geometry → colour pipeline + Rec.2020→sRGB + OETF). 2 tests.
   - **m4-50** (`74cc45e1e6`): `export::ExportEdit { method, geometry, params }`
     threaded through `show_export_dialog`/`export_images_async` as
     `Option<ExportEdit>`. Per image: `Some(edit) && is_raw_path` →
     `render_raw_export` (Rust render → `Pixbuf::from_bytes` → optional
     `fit_within` scale → `savev` png/jpeg/tiff), else `darktable-cli`. The
     darkroom Export button bakes the current edit at click time; both lighttable
     callers pass `None` (multi-export unchanged). gdk-pixbuf encode runs in the
     export `spawn_blocking` pool (Pixbuf/Bytes thread-local; captures Copy/Send).
     Opus review: no blockers; applied the should-fix ("TIFF 16-bit" label →
     "TIFF", since the Rust path is 8-bit gdk-pixbuf while darktable-cli is 16-bit).

   **m4-51..m4-55 — export/geometry/perf polish (all DONE):**
   - **m4-51** (`f00c98d26f`): "Reset crop & straighten" button (geometry-only
     reset; the header Reset deliberately leaves geometry). Closes the m4-47 gap.
   - **m4-52** (`2b4334b0d9`): **16-bit PNG/TIFF export** via the `image` crate.
     `render_linear_to_srgb8` refactored to a shared `srgb_encode_rgb` core +
     `render_linear_to_srgb16`; `render_raw_export` rewritten from gdk-pixbuf to
     `image` (JPEG 8-bit w/ quality, PNG/TIFF 16-bit, resize via imageops) — also
     removes the off-thread GObject concern (pure Rust). Opus: no blockers.
   - **m4-53** (`01778c194c`): **export toast** in the darkroom view (was
     `eprintln!`) — content wrapped in `adw::ToastOverlay`, export `toast_fn`
     routed to it.
   - **m4-54** (`cc3538a9ee`): **rayon-parallelise `demosaic_rcd`** (per-worker
     scratch via `for_each_init`, disjoint valid-region writes via a
     `Send`+`Sync` raw-ptr wrapper — the C `DT_OMP` analog). Speeds up full-res
     export. Opus: data-race-freedom PROVEN (`last_v(tv)==first_v(tv+1)`, tile
     never reads `out`). VNG stays serial (ring buffer serialises write-back).
   - **m4-55** (`9432ec9e5a`): **aspect-ratio-locked crop** (Free/1:1/3:2/2:3/
     4:3/16:9). `apply_aspect` (edge-derive on drag) + `fit_aspect` (fit-inside on
     selector change → immediate reshape). Opus: no blockers; math verified.

   **m4-56..m4-59 — containerise the Rust UI + make it self-sufficient (all DONE):**
   - **m4-56** (`38c1d10b7d`): the full-app Docker image now BUILDS the Rust/GTK4
     UI (`cargo build --release -p darkroom` → `darkroom-rs`) and the KasmVNC
     autostart LAUNCHES it instead of the C darktable app (C `darkroom-cli`
     retained for the export shell-out). Added GTK4/libadwaita build+runtime deps
     (+ `adwaita-icon-theme`, `gsettings-desktop-schemas`), `GSK_RENDERER=cairo`
     (KasmVNC's software X server black-windows GSK's default GL renderer), and
     `DARKROOM_LIBRARY_DB`. Two review BLOCKERs fixed: (1) production `darkroom-rs`
     never created the catalog schema (all `CREATE TABLE` were `#[cfg(test)]`
     fixtures; the C app's `dt_init` used to) → new `darkroom_db::schema::
     ensure_base_schema` (main-scoped) called at startup + before import, else a
     fresh `/config` imports 0 images; (2) the black-window renderer fix. Plus a
     SIGTERM/SIGINT handler in `run()` so the autostart's graceful-shutdown wait
     isn't racing an OS-killed process. Validated by a full image build + in-image
     checks (binary present, 0 missing libs).
   - **m4-57** (`d59ea29667`): tags now work in the standalone UI. darktable keeps
     tag NAMES in a separate `data.db` (attached as schema `data`) + an in-memory
     `memory` schema; the UI opened bare `Connection::open(library.db)` with NO
     attach, so every `data.tags`/`memory.darktable_tags` ref silently failed. New
     `schema::open_catalog(db_path)` opens library.db, attaches the sibling
     `data.db` + a per-connection `:memory:`, ensures the full schema, and sets a
     3s `busy_timeout` (parity with the rating/colour conns). Routed the 7
     tag-touching connection sites through it. Non-destructive on a real darktable
     catalog (ATTACH + CREATE IF NOT EXISTS only). Opus 8/10, shipped after the
     busy_timeout should-fix.
   - **m4-58** (`c6066aaf3a`): `tag_new`/`tag_delete` made atomic across the
     data/main/memory schemas via `conn.unchecked_transaction()` (atomic across
     on-disk DBs via SQLite's super-journal in the default rollback-journal mode).
     Fail-safe if ever nested under an outer txn (nested BEGIN rejected before any
     write). Opus: SHIP.
   - **m4-59** (`e2e45dcc17`): **rayon-parallelise `pipeline::process`.** Every
     `Stage` is a position-independent per-pixel map, so the RGBA-f32 buffer is
     split into pixel-aligned 64k-px bands run through the full stage sequence in
     parallel; **bit-identical to serial** regardless of thread count (export
     bit-matches preview). `process_band` shared ping-pong worker; `for_each_init`
     reuses one scratch per worker (peak memory LOWER than the old 2-buffer
     serial). Fail-safe guard: `Stage::is_pixel_local()` (exhaustive no-wildcard
     match) gates the parallel branch → a future spatial stage won't compile until
     classified, and non-local falls back to correct serial. Opus: SHIP, no
     blockers. darkroom-core 549→552 tests.

   **m4-60** (`1a82daaaab`) — lighttable batch export honours per-image edits.
   Previously only the single-image darkroom export rendered through our pipeline;
   lighttable multi-export always used `darktable-cli`, ignoring edits. New
   `dialogs::load_export_edit(db, path)` returns `Some(ExportEdit)` only for a raw
   the user actually edited (persisted params / non-identity crop-straighten /
   non-default demosaic method); unedited raws + non-raws stay on `darktable-cli`
   (fuller default stack until our subset pipeline reaches parity). Seeding matches
   the preview via the now-`pub(crate)` `darkroom::initial_params`. `export_images_
   async`/`show_export_dialog` gained a `db_path` param; the loop does
   `edit.or_else(|| load_export_edit(...))`. `persist::load_edit_state` loads all
   three pieces over ONE connection + ONE imgid resolution (architect S1). A
   Rust-render failure is counted+logged, NOT silently fallen back to cli. Opus:
   SHIP, no blockers. **Known (expected, not a bug):** single-image darkroom
   export bakes LIVE (incl. unsaved) edits; lighttable export reads PERSISTED state
   only — so the same raw can export differently from the two entry points.
   darkroom-ui 135→136 tests.

   **m4-61** (`70d5013ae4`) — hide the demosaic-method selector for X-Trans.
   The RCD/VNG/PPG dropdown only affects Bayer sensors; X-Trans (.raf) always
   uses fixed Markesteijn, so the control was a no-op there (previously masked
   by only a tooltip). `RawPreview` gains an `is_xtrans` flag (from
   `img.xtrans.is_some()`); the darkroom page wraps header+dropdown+separator in
   one `demosaic_box` held by a `glib::WeakRef` on `PreviewCtx`, and
   `spawn_decode` hides the whole section once a decode reveals X-Trans. Hides
   on first decode (flash stays on the rare X-Trans case, not common Bayer);
   WeakRef empty for non-raw paths so the decode arm no-ops; visibility re-set
   on every decode incl. Bayer method-change re-decodes is intentional. Placed
   after the generation guard so a stale X-Trans decode can't hide the selector
   under a newer Bayer one. Opus architect: SHIP, no blockers/majors. This
   closes the "Selector polish (deferred from m4-43 N3)" candidate below.
   darkroom-ui 136 tests pass.

   **m4-62** (`d6ac01b6ea`) — rayon-parallelise the VNG demosaic gradient phase,
   the last serial hot loop in `demosaic_vng`. C's ring buffer defers each row's
   write-back by two rows so a gradient never reads a refined row; freezing the
   post-lookup `out` into a read-only `src = out.clone()` feeds every row the
   same inputs the serial sweep saw, so `out.par_chunks_mut(row_len)` (disjoint
   rows, kernel reads `src` / writes only interior cols) is bit-identical.
   Written rows 2..=height-3 match the serial loop + its two tail copies exactly.
   **Deliberate trade-off:** the snapshot reverses C's ring-buffer memory saving
   — a transient n*4-float clone (~1 GB on a 60 MP frame) buys the parallelism;
   the clone is load-bearing (a future memory-pressure tiling would need per-band
   ring buffers to drop it). Opus architect: SHIP, bit-identical watertight.
   darkroom-core 552 tests pass. This closes the "parallelise `demosaic_vng`"
   candidate below; `pipeline::process` (m4-59) and RCD (m4-54) were already
   parallel, so the per-pixel demosaic/pipeline path is now fully multi-threaded.

   **m4-63** (`3b1268978e`) — split the catalog schema bootstrap from the
   read-hot tag opener (the deferred "S2" item). `open_catalog` ran the full
   CREATE-IF-NOT-EXISTS DDL on every open, incl. `load_tags` (fires on every
   lighttable selection). Durable tables (`main.*`, `data.tags`) persist on disk
   and need creating once per catalog; only `memory.darktable_tags` is
   per-connection. Extracted `attach_catalog` (open + busy_timeout + ATTACH, no
   DDL); split `ensure_full_schema` into `ensure_persistent_schema` +
   `ensure_session_schema` (identical statements, same order); added
   `open_catalog_session` (session scratch only). Startup bootstraps the durable
   schema once via `open_catalog` (was a bare `ensure_base_schema` that never
   made `data.tags`), synchronously before any panel/read. The two read-hot
   paths use the session opener; the four rare writes stay full (self-heal the
   durable schema if the bootstrap warned) — commented as intentional. Marginal
   by design (the two ATTACHes dominate the probes); the value is the correct
   once-vs-per-open split. Opus architect: SHIP, composition provably identical,
   ordering holds on every current path. darkroom-db 92 tests pass (new
   `session_open_reads_durable_tags_without_re_ensuring`). Architect's forward
   note: if a third read path or a non-`build_main_window` entry point appears,
   migrate to a one-opener + process-global per-path "already-ensured" guard
   (preserves read self-heal, drops the ordering coupling). (Residual bare
   `Connection::open` + `ensure_base_schema` in the import path folded into
   `open_catalog` in m4-64.)

   **m4-64** (`4eebc4ab01`) — route folder import through `open_catalog` + stop
   dropping insert errors (the residual bare-open from m4-63). `import_folder_
   sync` (off-thread) used a bare `Connection::open` + `ensure_base_schema`;
   switching to `open_catalog` consolidates on one opener AND gives it a 3s
   `busy_timeout` so its writes wait out the rating/colour-label writers' brief
   library.db lock instead of an immediate `SQLITE_BUSY`. The real fix (architect
   M1): `image_insert`'s `Result` was ignored while `count` incremented
   unconditionally — silently dropping failed rows and over-reporting the
   "imported N" toast. Now count reflects only rows that landed; failures are
   logged. New test seeds a fresh config (no library.db/data.db) with mixed
   raw/non-raw files → asserts count == raw files, data.db materialised, empty
   path is a no-op. Opus architect: APPROVE (Option C). darkroom-ui 136→137
   tests.

   **m4-65** (`e7b148c6cb`) — make folder import transactional (the m4-64 N1
   follow-up), in a two-phase, poison-guarded shape that resolves the tradeoffs
   the architect flagged. Phase 1 walks + `probe_dims`-decodes every header with
   NO db lock held; Phase 2 opens one transaction and does only the fast insert
   burst — so `BEGIN DEFERRED` holds the write lock just for the burst→commit, not
   across the slow probe I/O (a naive loop-wrap would have blocked the
   rating/colour writers for the whole import). Collapses N fsyncs→1 and makes
   roll+images atomic (no half-populated roll on crash). Count-integrity guard:
   `image_insert` dedupes via SELECT (a name clash is `Ok`), so the only reachable
   insert `Err` is an engine error (SQLITE_FULL/IOERR/NOMEM/BUSY) that
   auto-rolls-back the whole tx → the loop checks `conn.is_autocommit()` and
   aborts (`None`) if the tx vanished, instead of autocommitting later rows and
   re-opening the m4-64 count-lie. `commit()` failure is logged then `None`
   (truthful "Imported 0"). Best-effort per-image survives for any statement-level
   error (none reachable today). Test proves the tx committed via
   `image_count_all()==3` after re-open. Opus architect: MAJOR poison-guard +
   MINOR commit-log applied, two-phase design endorsed. darkroom-ui 137 tests.

   **m4-67** (`33506a48f5`) — export unedited raws through the Rust pipeline
   (milestone 5; drops the darktable-cli fallback for ALL raws). Previously only
   edited raws rendered Rust-native; an unedited raw exported via cli's fuller
   default look — different from the darkroom view, which already renders unedited
   raws through this exact subset pipeline (a WYSIWYG violation). Now every raw
   develops through `render_raw_export`; an unedited raw uses
   `default_raw_export_edit()` (= the preview seed: `initial_params(None,true)`
   → sigmoid on, default Rcd demosaic, identity geometry). Only non-raw formats
   still use cli. **Accepted trade-off (explicit user design call):** the subset
   is lighter than darktable's default stack (no highlight-recon / base curve),
   so an unedited raw's export can look flatter; no cli fallback if `rawloader`
   can't decode an exotic raw (counts as failed). Test
   `unedited_raw_export_default_matches_preview_seed` pins export==preview.
   **Review debt:** the fricktrade-architect review was blocked by an account
   session limit; user directed proceeding — obtain the deferred review later.
   darkroom-ui 137→138 tests. *(Review debt cleared — see m4-68.)*

   **m4-68** (`34d48a60c1`) — the deferred m4-67 review, run in full (two rounds),
   plus its fixes. `render_raw_export` created the destination then encoded into
   it → a mid-encode failure left a truncated file that looked valid and a failed
   re-export clobbered a prior good one (MAJOR). New `atomic_write` helper:
   encode to a unique `<dest>.<pid>.<n>.part` temp, `fsync` it, then `rename` onto
   dest only on success (both failure paths unlink the temp). Two subtleties the
   review caught: (a) `BufWriter::Drop` swallows flush errors → explicit
   `into_inner` on the JPEG arm; (b) on delalloc FSes (ext4/xfs/btrfs) `write()`
   returns Ok when the disk is full → the pre-rename `fsync` is what actually
   makes "never promote a truncated file" true. Also a release runtime guard
   (not just `debug_assert`) refusing a batch that carries a fixed single-image
   edit (would bake one crop onto all). **Known data-safety gap (deferred):** the
   non-raw `darktable-cli` export branch is still non-atomic (clobbers on
   mid-write failure) — not worth hardening code milestone 5 deletes once the
   Rust pipeline covers non-raw formats. darkroom-ui 138→139 tests.

   **m4-69** (`4ad67c5bc1`) — Rust-native non-raw (JPEG/PNG/TIFF) export
   (milestone 5). `render_nonraw_export` decodes via the `image` crate → composites
   alpha over white → runs the preview's non-raw `apply_pipeline` (colour params
   only; geometry/demosaic are raw-only) → resize → encode via the atomic+fsync
   `atomic_write`. Export loop now: raw → Rust; jpg/jpeg/png/tif/tiff
   (`is_rust_image_path`) → Rust; heic/heif/avif → cli. Fixes the "non-raw edit
   dropped" gap (a JPEG edited in the darkroom view now bakes its edit on export).
   DRY: shared `write_jpeg_rgb8` (with the into_inner flush fix) across the raw +
   non-raw JPEG arms. **NOT byte-WYSIWYG (documented):** preview decodes via
   GdkPixbuf, export via `image` (JPEG ±1-2 LSB); 8-bit out truncates 16-bit
   sources pre-pipeline (banding on edited gradients — 16-bit follow-up). Tests:
   extension predicate, real-PNG passthrough (pixel-exact), +1 EV brightens. Opus
   architect: APPROVE with minors, all applied before commit (alpha composite,
   doc-comments, non-passthrough test). darkroom-ui 139→142 tests.

   **m4-70** (`f213fb5834`) — 16-bit non-raw PNG/TIFF export (m4-69 follow-up).
   New `preview::apply_pipeline_rgb16` (16-bit sibling of `apply_pipeline`, packed
   RGB u16, pipeline in f32; empty ⇒ byte-exact passthrough) leaves the 8-bit
   preview hot path untouched. `render_nonraw_export` branches: JPEG → 8-bit,
   PNG/TIFF → 16-bit (`to_rgba16` → composite → `apply_pipeline_rgb16` →
   `save_with_format`). So an unedited 16-bit source round-trips losslessly and an
   edit quantises to 16 bits (bands less); an 8-bit JPEG source → 16-bit PNG is
   strictly ≥ the old output. Composite extracted to `composite_rgba8/16_over_white`.
   Tests: lossless round-trip over BOTH PNG+TIFF (non-zero low bytes), a 16-bit
   composite unit test (opaque/transparent/half-alpha). Opus architect: no
   blockers, SHIP after the two tests (added). **Backlog nit:** resize runs in
   gamma space (both paths, pre-existing) — resampling error now exceeds the
   quantisation m4-70 removed; do linear-light resize before tightening further.
   darkroom-ui 142→144 tests.

   **m4-73** (`f0b843e18f`) — **ROI/(width,height) pipeline signature + first
   spatial stage (Sharpen)**; the biggest architectural step for the pixel
   pipeline (previously strictly per-pixel / size-agnostic). `Stage::apply` /
   `Pipeline::process` / `process_band` now carry `(w,h)`; per-pixel stages ignore
   them, `process` hard-asserts `w*h*4==len`. Parallel safety reuses the m4-59
   `is_pixel_local` gate: the band-parallel path only runs when every stage is
   pixel-local (passing each band as `(band_px,1)`); a single spatial stage forces
   the SERIAL whole-buffer path where it sees the true rectangle. **Sharpen**
   (`is_pixel_local=false`): `gaussian_kernel` faithfully ports sharpen.c
   `init_gaussian_kernel` (sigma2 on uncapped radius, rad=ceil clamped to MAXR=12),
   sharpens a Rec.2020 luma via the migrated `darkroom_sharpen_process`, adds the
   luma detail back to R/G/B (luminance unsharp mask). **Documented caveats:** not
   the bit-exact darktable Lab-`L` sharpen (needs RGB↔Lab color-space infra — a
   separate migration); `threshold` is LINEAR-luma (~[0,1]) not Lab-L [0,100] (a
   100× footgun for a future UI, guarded by a debug_assert tripwire); equal RGB
   detail shifts chroma on saturated edges. No live caller yet (core stage only,
   like Sigmoid m2-5a before UI m2-5b). Opus architect: no blockers; MAJOR 1 (rad
   clamp) + MAJOR 2 (threshold domain) + stale-doc + test gaps all fixed before
   commit. darkroom-core pipeline 15→21 tests. **Follow-ups:** wire a Sharpen UI
   slider (mapping threshold /100); planar-luma kernel (drop 4× scratch);
   ratio-preserving RGB detail once Lab lands; per-stage band strategy (parallel
   the pixel-local prefix, spatial stages whole-buffer).

   **Candidate next increments after m4-60** (recorded so they survive a context
   clear — the colour-label arc m4-19/20/21/23/24/25/26 is complete; the tag
   rename/delete popover is leak-free + a11y-complete; the Rust UI runs in the
   container with working tags, a parallelised pipeline, and edit-honouring batch
   export):
   - ~~**Perf:** parallelise `demosaic_vng`.~~ **DONE — m4-62** (frozen-snapshot
     read of the border+lookup `out` makes the gradient rows independent).
     `pipeline::process` (m4-59) and RCD (m4-54) were already parallel.
   - ~~**Selector polish (deferred from m4-43 N3):** hide the demosaic DropDown
     for X-Trans files.~~ **DONE — m4-61.**
   - **Container follow-ups (deferred, reviewed non-blocking):** ~~S2 —
     `open_catalog` runs `ensure_full_schema` on every open.~~ **DONE — m4-63**
     (persistent-once-at-startup + session opener for the read-hot paths). Still
     open: a live end-to-end KasmVNC session check of `darkroom-rs` (validated at
     build + unit level so far, not a live GUI run — needs a display).
   - Smaller: `with_image_id(full_path, db, |conn, imgid| …)` helper to dedupe the
     tag-op open→lookup→fault-log ceremony, once a 5th tag op appears (not before).
5. *Cargo-native build* — once UI + pipeline run from Rust, retire CMake.

   **m4-66 (groundwork):** the standalone Rust app is already fully
   cargo-native-buildable — `cargo build --release -p darkroom --bin darkroom-rs`
   links the GTK4 binary against system GTK4/libadwaita with **no CMake and no C
   darktable** (verified in the rust-dev image, ~2.5 min from clean). CI now
   guards this: the `Rust` workflow gained a "cargo-native app binary build" step
   (`cargo check` only type-checks; it does not *link* the bin — the full
   container image only links it after the heavy CMake C build, so a linker
   regression in the standalone app would otherwise escape until the Docker
   build). `darkroom-sys` is a C-linkless bindings crate (committed `bindings.rs`,
   just `dt_imgid_t = i32` etc.), so nothing in the Rust workspace needs the C
   toolchain.

   **Remaining couplings before CMake can actually be retired** (both in the full
   `docker/Dockerfile`):
   - *Runtime:* `darkroom-rs` shells out to `darkroom-cli` (the C `darktable-cli`)
     only for **heic/heif/avif** (and unknown extensions) now. **m4-67:** all raws
     develop Rust-native (unedited → `default_raw_export_edit()`, the preview seed).
     **m4-69:** the non-raw formats the `image` crate decodes (jpg/jpeg/png/tif/
     tiff) also go Rust-native (`render_nonraw_export`, the preview's `apply_pipeline`
     over an `image`-crate decode, alpha composited over white). Remaining cli
     users: heic/heif/avif (no Rust decoder) + exotic raws `rawloader` can't decode
     (no cli fallback on raw decode failure — counts as failed). **m4-70:** non-raw
     PNG/TIFF export is now **16-bit** (JPEG stays 8-bit), so a 16-bit source
     round-trips losslessly and edited gradients band less. Remaining parity notes:
     preview uses GdkPixbuf vs export's `image` crate (JPEG ±1-2 LSB); resize runs
     in gamma space (pre-existing, both paths). **Blocker to full cli removal:** a
     Rust HEIF/AVIF decoder (or dropping those formats) + broad raw decode coverage.
   - *Build:* CMake drives a `cargo build` of `darkroom-core` as a static lib
     linked into the C app, and builds the C `darktable`/`darktable-cli` the
     runtime still needs. Once the export fallback is gone, the C build (and thus
     CMake) can be dropped and the image becomes a pure-cargo build of
     `darkroom-rs` + a GTK4 runtime.
   Ordered retirement path: reach export parity → drop the `darktable-cli`
   fallback in `dialogs::export_images_async` → strip the CMake C build from the
   Dockerfile (cargo-only) → retire `CMakeLists.txt` + `build.sh`.

   **m4-71** (`1d74f84407`) — **CMake-free container image** `docker/Dockerfile.
   cargo-native`: builds/ships only `darkroom-rs` (no CMake, no C darktable, no
   cli). Kept separate from the untouched production `docker/Dockerfile` to prove
   the path A/B before flipping production. Shallow clone without submodules
   (C-only); GTK4/adwaita runtime only. Added `librsvg2-common` (the SVG pixbuf
   loader the `*-symbolic.svg` UI icons need — architect M1) + `GTK_A11Y=none`.
   Validated headlessly: build ✓, `ldd` fully resolved, SVG loader registered,
   container boots and darkroom-rs stays up 80s with **0 autostart restarts**.
   heic/heif/avif export degrades gracefully (no cli). GUI rendering still needs a
   display to eyeball. **Next m5 steps:** verify the GUI live (display), decide
   heic/avif (drop vs keep cli optional), then flip production to cargo-only +
   retire CMakeLists/build.sh.
   - **Discovered bug (pre-existing, follow-up):** darkroom-rs logs one
     `Gtk-CRITICAL gtk_box_append: gtk_widget_get_parent(child) == NULL` at
     startup — a child appended while already parented; non-fatal, affects the
     full image too. Surface with `xvfb-run darkroom-rs` + grep Gtk-CRITICAL.

   **m4-72** (`7661c6b5fe`) — **production is now CMake-free.** `git mv` promoted
   the validated cargo-only image to `docker/Dockerfile` (the published `:latest`)
   and preserved the full C+Rust recipe as opt-in `docker/Dockerfile.full-c`.
   `docker.yml` drops the CMake build-args from `:latest` and gained an on-demand
   (`workflow_dispatch`) job publishing the full image as `:full-c` (so the
   heic/C-GUI capability stays in the registry + guards full-c from bit-rot).
   `docker-compose.yml` fixed: default `darkroom` is cargo-only; the GPU profiles
   (OpenCL is a C-core feature) build full-c. Build pinned to the exact CACHEBUST
   commit (M1). Architect: approve, 3 before-merge fixes applied. So **milestone 5
   is substantially DONE** — the shipped image and CI no longer use CMake; the C
   `src/` + CMakeLists/build.sh remain only as the ongoing IOP-migration
   reference (full C build available via `Dockerfile.full-c`). Remaining: live GUI
   eyeball (needs a display), the heic/heif/avif decision (drop vs keep full-c),
   and the deferred `--locked` (needs Cargo.lock committed — currently gitignored).

   **Before the Dockerfile-strip step, confirm each thing the C `install`
   currently provides to the runtime is Rust-side, not C-install-side** (checklist
   so the strip isn't a surprise break):
   - *DB schema* — must come from the Rust catalog bootstrap (`ensure_*_schema` /
     `open_catalog`, m4-56/m4-63/m4-64), NOT a darktable-created schema. Believed
     satisfied; verify no residual reliance.
   - *GTK assets* — icons/CSS the UI renders under KasmVNC (`GSK_RENDERER=cairo`).
     Confirm none resolve from darktable's installed icon theme under
     `/opt/darkroom/share`, or the chrome breaks when the C install goes (the
     runtime already installs `adwaita-icon-theme` — check that's sufficient).
   - *Decode/preview path is already C-free* — `darkroom-core` decodes RAW via the
     pure-Rust `rawloader = 0.37.1` and takes camera colour matrices from
     rawloader's `xyz_to_cam`, not darktable's `share/` data (architect-confirmed),
     so the runtime pixel path doesn't couple to the C install.
   - *Future data dep* — `denoiseprofile` is a stub today (no coupling), but if it
     is ever wired into the Rust subset it needs `noiseprofiles.json`; note it now
     so it isn't a surprise data dependency later.

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
