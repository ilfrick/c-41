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
| ~~`colorbalancergb`~~ **DONE — m4-81..85** (`iop/colorbalancergb.rs`) | 0 (process + gamut-LUT builders ported; the 2 GUI graph loops @1511/1555 are GUI-only) | ~~Filmlight Yrg / `work_profile`~~ |
| `colorreconstruction` | 3 | 3D bilateral grid |
| `colorin` | 3 | ICC matrix + LCMS |
| ~~`channelmixerrgb`~~ **DEAD CODE** — both loops (`:753`/`:822`, `_auto_detect_WB`) are inside `#ifdef AI_ACTIVATED` which is only a commented-out `// #define` (never defined) → **never compiled**, not in the binary. Grep-count only; nothing to port. | 2 (dead) | — |
| `colortransfer` | 1 | ~~fuzzy-cluster transfer `:348`~~ **PORTED — m4-88** (`771bd5b6ad`, `darkroom_colortransfer_apply_ab`; architect APPROVE bit-exact; full-c `-Werror` clean). Remaining `:243` **k-means** is **non-deterministic** (`dt_points_get` xorshift128+ PRNG over global/per-thread state + atomic accumulation) — no bit-exact port possible; a serial Rust port would cluster differently than any C run. Best-effort only. |
| ~~`retouch`~~ **GUI-only** (`:3082`/`:3135` = wavelet-scale-preview auto-levels; see ICC row) | 2 (GUI) | — |
| `colorequal` | 1 | GUI background renderer (intentionally deferred) |
| `colorout` | 1 | LCMS `cmsDoTransform` |
| ~~`diffuse`~~ **PORTED — m4-87** (`072fec679b`, `darkroom_diffuse_heat_pde`; `heat_PDE_diffusion` anisotropic-diffusion kernel; architect **APPROVE** bit-exact after quota-reset re-run — vector_exp bit-hack, isophote-vs-gradient not swapped, deriv↔HF/LF 4-way pairing all verified; full-c `-Werror` validating) | 0 | ~~anisotropic PDE~~ |
| `toneequal` | 1 | GUI LUT |

(`ashift`, `clipping`, `denoiseprofile`, `gamma`, `liquify`, `rawoverexposed`
previously listed here/as stubs are at 0 loops — fully migrated.)

#### What blocks the remaining loops

| Infrastructure | Unblocked IOPs |
|---|---|
| `dt_interpolation_*` | demosaicing cluster |
| ~~3D bilateral grid~~ **PORTED — m4-77** (`bilateral.rs`) | lowpass, shadhi, retouch, monochrome, globaltonemap, colormapping, ashift, bilat |
| ~~Recursive Gaussian (`dt_gaussian`)~~ **PORTED — m4-78** (`gaussian.rs`, RGBA `blur_4c`) | bloom, highpass, lowpass, shadhi, hazeremoval |
| ~~À-trous wavelet (`dwt.c`)~~ **PORTED — m4-79** (`dwt.rs`, `decompose` + `denoise`) | atrous, retouch, denoiseprofile (wavelet mode) |
| Filmlight Yrg / `work_profile` callbacks — **Yrg/UCS/JzAzBz conversions DONE** (color.rs, thru m4-81); colorbalancergb loop port in progress (m4-82/83) | colorbalancergb, colorin |
| Per-pixel ICC / LCMS — **DECISION (m4-86): pure-Rust matrix path, NO lcms2** (keeps the CMake-free / C-linkless build goal). Scope of the 20 remaining OMP loops re-surveyed: **colorout matrix path already Rust** (`darkroom_colorout_*`); **portable & DONE**: ~~colorin `:777` cmatrix-bm~~ **PORTED — m4-86** (`b0f20649e4`, `darkroom_colorin_cmatrix_bm`: tone-curve LUT + `_apply_blue_mapping` + matrix→Lab, incl. the nmatrix/lmatrix gamut-clip variant; architect APPROVE, transpose-equiv proven, full-c `-Werror` clean). So the meaningful portable-matrix ICC path is complete. **retouch ×2 (`:3082`/`:3135`) are GUI-ONLY** (re-checked m4-86 f/u): `rt_process_stats` runs only under `g && dt_pipe_is_full` w/ `preview_auto_levels==1`; `rt_adjust_levels` only when `dwt_p->return_layer>0` (the `display_wavelet_scale` toggle) — the wavelet-scale-preview auto-levels, not core processing. Reclassify as GUI-only (like colorequal/toneequal). **stays unsupported** (needs LCMS for non-matrix/LUT profiles) = colorin `:1054`/`:1097` "general lcms2 fallback", colorout's generic path. These are FFI-boundary ports (replace the C loop with a `#[no_mangle] extern "C"` call like `darkroom_colorin_cmatrix_fastpath_simple`), so they need C edits + the **full-app Docker build** (not cargo-only). **RESOLUTION (user chose option b): build a pure-Rust ICC engine** (`darkroom-core::icc`) to remove the lcms2 dependency entirely — same functionality (matrix **and** cLUT profiles), accuracy ≥ LCMS (f32 throughout vs LCMS's 16-bit path), pure Rust. NB: the shipped `darkroom-rs` product is ALREADY lcms-free (0 refs in `crates/`); the ~50 `cms*` calls live only in the transitional C darktable (colorin/colorout), of which only 7 are `cmsDoTransform` (the loops) — the rest are profile parse/build. So the engine is judged on *spec-correctness*, not bit-exact-to-LCMS. **ICC-engine roadmap:** m4-89 = parser (header/tag-table/`XYZ`/`curv`/`para`, matrix-shaper) ✅ (`c8fd82c4e5`, APPROVE-WITH-FIXES incl. a real DoS guard); m4-90 = cLUT **N-D interpolation core** (`icc/clut.rs`: LCMS-matched tetrahedral for 3-in RGB + general N-linear) ✅ (`7037102b86`, **APPROVE** — tetra vertex table verified arm-by-arm vs LCMS); m4-91 = parse the LUT **tags** (`mft1`/`mft2` v2, `mAB `/`mBA ` v4) into `Clut`+`Curve`+matrix (carry validate-before-reserve; add `Clut::validate()`; use stack arrays in N-linear when it goes hot); m4-92 = transform assembly (device→PCS→device, PCS Lab/XYZ, intents, CAT) + FFI-wire colorin/colorout LUT path. Large multi-increment build (~colorbalancergb scale); engine ~half done (matrix parse + interp core). | colorin, colorout, retouch |
| ~~bespoke {L,a,b,weight} grid (own, not common/bilateral)~~ **PORTED — m4-80** (`colorreconstruct.rs`) | colorreconstruction |
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
  - **m4-95 (fix):** the thumbnail grid never actually rendered in the packaged
    GUI — `lighttable_page()` returned the grid's `ScrolledWindow` wrapped in a
    throwaway `adw::NavigationPage`, and the caller appended the still-parented
    ScrolledWindow to the layout `hbox`, so `gtk_box_append` (asserts parent ==
    NULL) silently no-oped and dropped the grid. Users saw `[Collections |
    Metadata]` with no photos. Latent since ui-4; missed by display-free tests.
    Fixed by returning the `ScrolledWindow` directly (bug class deleted).
- **Lighttable ↔ darktable parity gaps (planned, user-requested 2026-07-21).**
  Current lighttable is a functional *subset*; the "feels like darktable" gaps,
  in priority order:
  - **m4-96 — dark theme.** darktable ships a dark grey theme; we use adwaita's
    default light. First pass: `adw::StyleManager::set_color_scheme(ForceDark)`.
    Follow-up: a custom CSS provider matching darktable's exact greys.
  - **m4-97 — view switcher + filter/sort bar.** Top-bar `lighttable | darkroom
    | other` switcher; filter by rating/colour/status + sort-by + live image
    count (darktable's top toolbar). We currently expose only a search box.
    - **m4-97a (done):** live image-count indicator in the lighttable top bar,
      bound to the grid model's `items-changed` (no per-call-site plumbing).
      Counts only real-image rows (contain `/`), excluding the model's sentinel
      rows (empty-state / truncation notice) so an empty view reads "0 images".
    - **m4-97b (done):** "sort by" dropdown (Filename / Date taken / Rating).
      A thread-local `SortOrder` + a self-registering reload closure: every
      loader records how to re-run itself, so changing the sort re-applies the
      *current* view (folder / search / tag / colour) with zero changes to the
      ~9 trigger sites. `order_clause()` is static (injection-safe); undated
      images sort last, rejected sorts below 0 (per review). Integration test
      runs all 3 orders against a seeded DB. **Follow-up (responsive header):**
      at narrow widths (~540px browser) the header overflows and clips the
      right-side controls (sort dropdown, image count, export) — pre-existing,
      now more visible. Needs `adw::Breakpoint`-based collapse/rearrange.
    - **m4-97c (done):** top-bar quick-filter dropdown — darktable's
      `filter [all images ▾]`. `FilterPreset {AllImages, UnstarredOnly,
      AtLeastStars(1..=5), RejectedOnly}` where each preset is **not** a parallel
      filter implementation but a named `(RatingCompare, stars)` pair applied via
      `set_filter_preset()` to the *same* state the m4-98b/d bottom bar drives —
      so the two controls can't disagree about what the grid shows. Rows (and
      labels) come from `FilterPreset::ALL`, plus a display-only trailing `custom`
      row for states no preset names (e.g. `≤ 3`), so the dropdown reflects rather
      than lies. **Two controls, one state** is made safe by a new observer bus:
      `add_filter_observer()` + `filter_changed()` (called by every `set_*`) push
      each change back out to all filter controls, with a `filter_sync_in_progress()`
      re-entrancy guard so an observer's programmatic `set_selected` isn't mistaken
      for a user edit and can't recurse. Observers clone out of the `RefCell`
      before running — the discipline `reload_current_view` documents. This also
      pays off the m4-98d review's flagged debt (a filter set outside the bottom
      bar used to leave its stars lying). The bottom bar's persistence + display
      refresh now live in one observer, so *any* control's change is persisted.
      Verified live: `ge:3` ⇒ top reads "★ 3 and higher", bottom shows ≥ with 3 lit
      stars, count 2000→95; `le:2` ⇒ top honestly reads "custom", bottom shows ≤ 2.
      4 new tests (preset↔state round-trip over ALL, index/label guards, clamping,
      and non-preset states correctly yielding `None`).
    - **m4-97e (done):** sort direction toggle (ascending/descending) beside the
      "sort by" dropdown, in a shared `.linked` box. `SortOrder::terms()` returns
      `(expr, ascending, reversible)` triples and `order_clause(reverse)` flips
      only the *reversible* terms — the DateTaken undated-last guard is marked
      non-reversible so undated images stay at the bottom in both directions. New
      thread-local `SORT_REVERSE` + `set_sort_reverse()`; the `set_sort_order`
      reload logic was extracted into a shared `reload_current_view()`. The
      `ToggleButton` swaps its up/down arrow icon to reflect state. Tests lock the
      exact reversed clauses and the reversed row order (incl. undated-still-last
      and rejected-now-top) via the seeded rusqlite fixture. Architect: "ship it".
    - **caption fix (done):** the lighttable switcher is single-row again (no
      "Darkroom" app-name caption): the app is named "Darkroom" but so is the
      editing *view*, so the caption duplicated the switcher button. The darkroom
      header keeps its caption (there it's the filename — real per-image context).
    - **m4-97d (done):** darktable-style `Lighttable | Darkroom | Other` view
      switcher as the lighttable header title widget. "Darkroom" opens the
      selected image in the editor (same push as double-click); "Other"
      (map/print/tethering) is disabled — unported. `NavigationView::popped`
      resets the toggle to Lighttable on return. Verified end-to-end in the
      container. The switcher builder is shared (`build_view_switcher()`) so both
      headers stay identical.
    - **m4-97d follow-up (done):** the switcher now also appears in the darkroom
      view's header — "Darkroom" active, filename shown below it (kept, not
      dropped), "Lighttable" pops back to the grid via the NavigationView's
      built-in `navigation.pop` action. Verified both directions in the container.
      Remaining: add the actual "Other" views (map/print/tethering).
    - **Follow-up (perf) — DONE:** `lighttable` `fill_grid` now replaces the model
      in one `StringList::splice(0, n_items, &rows)` instead of an O(N²)
      `remove(0)` clear loop + per-row `append` (~2N `items-changed` emissions).
      The 4 loaders no longer clear first (fill_grid owns clear+fill atomically —
      no transient empty-model window). One `items-changed` per load → one
      count-label update. All 144 UI tests pass; verified live (2000→89 on a
      folder click). Selection/scroll reset on reload is unchanged from before.
    - **Follow-up (CI reproducibility):** `rust-toolchain.toml` pins
      `channel = "stable"` (floating) — a new stable can fail `clippy` with zero
      commits from us. Pin to an explicit `1.XX.0`, bumped deliberately.
  - **m4-98 — bottom toolbar.** Quick rating/colour strip, view-mode switcher
    (file-manager / zoomable / culling), thumbnail-size control, overlay toggles.
    - **m4-98a (done):** the bottom bar itself (`adw::ToolbarView::add_bottom_bar`
      of a `.toolbar` `CenterBox`) + the thumb-size stepper (`[−] N [+]`) at the
      right, mirroring darktable's "images per row" control. It drives the grid's
      *max*-column bound (`THUMB_COLS_MIN=2 .. MAX=12`, `DEFAULT=6`) — capping,
      not fixing, columns so a narrow framebuffer still fits the row instead of
      clipping. `min_columns` stays 2. Buttons grey out at the range ends. Bottom
      bar + stepper verified rendering in-container.
    - **m4-98b (done):** star-rating filter at the left of the bottom bar (five
      star buttons; click star N ⇒ show images rated ≥ N, re-click the floor to
      clear). Unlike the mutually-exclusive left-panel filters it **composes** with
      whatever collection is active (folder / tag / colour / search), mirroring the
      sort infra: a thread-local `MIN_RATING` + pure `rating_and(min)` → a
      ` AND (i.flags & 8) = 0 AND (i.flags & 7) BETWEEN min AND 5` fragment spliced
      into every loader's WHERE (drops rejected images, then keeps only real N..5
      stars — excludes unrated 0 too). The no-filter/load-all branch uses
      `WHERE 1=1{rating}` for uniform splicing; the tag-prefix OR is now
      parenthesised so the rating AND applies to both disjuncts (latent-bug fix).
      Stars track `current_min_rating()` (single source of truth). **Bundled
      correctness fix:** aligned the whole rating path to darktable's real
      `images.flags` layout — the 0..5 star value in bits 0–2 (`flags & 7`) and
      "rejected" in the *separate* bit 3 (`= 8`). The prior Rust code stored/read
      stars in bits 1–3 (`(flags>>1)&7`), so it was **misreading darktable's own
      ratings** (a native 3-star `flags & 7 == 3` came back as 1 star; a native
      6/7 read as rejected). `query_rating`/`save_rating`/`SortOrder::Rating` all
      moved onto the correct convention (with `flags_star_rating`/
      `flags_with_star_rating` helpers), and the sort sinks any legacy `>5` value
      below 0-star so it can't out-rank a real 5-star. Verified against Nicola's
      real 11k-image catalog: all ratings are darktable-native (bit-0 population of
      11,274 is impossible under the old scheme), so no data migration is needed.
      6 new pure/seeded tests (fragment shape+clamp, bit round-trip preserving
      reject/high bits, colour-query composition, N..5 end-to-end).
    - **m4-98d (done):** rating-filter depth — a comparator dropdown (`≥` / `=` /
      `≤` / rejected-only, `RatingCompare`) to the left of the stars, and the
      filter now **persists across sessions**. `MIN_RATING` (star count) pairs with
      a new `RATING_COMPARE` thread-local; `rating_predicate(stars, cmp)` replaces
      `rating_and` and emits `= N` / `BETWEEN 0 AND N` / `BETWEEN N AND 5` /
      `(flags & 8) = 8` (rejected ignores the count). `AtLeast` + 0 stars stays the
      canonical no-filter state. Persistence rides a new global key/value table
      `main.darkroom_ui_prefs` (persist.rs, same best-effort private-table pattern)
      keyed `rating_filter`; the value is a compact token (`off`/`ge:N`/`eq:N`/
      `le:N`/`rej`) encoded by `rating_filter_token()` and restored at startup via
      `apply_rating_filter_token()` **before** the first load (no reload — no loader
      registered yet). `build_color_mask_query` now takes a prebuilt rating `&str`
      (decoupled from the enum). In `Rejected` mode the star row greys out. 4 new
      tests (predicate shape across all comparators, token round-trip+fallback,
      pref-table upsert/missing-table, comparator end-to-end over seeded flags).
    - **m4-98e (done):** thumbnail **overlay modes** — the "overlay toggles" half
      of this milestone. A centre dropdown (`No overlays` / `Stars + labels` /
      `Full info`) maps to `OverlayMode {Hidden, Normal, Extended}`, controlling
      which of a cell's three metadata rows show: `overlay_visibility(mode) ->
      (filename, stars, colours)` — pure, so the mapping is display-free testable.
      `Extended` is the default, so the out-of-box look is unchanged. Applied in
      two places, which together cover GTK's cell recycling: `connect_bind` (every
      newly-bound/recycled cell) and `set_overlay_mode(grid, mode)`, which walks
      the realized cells via `for_each_cell_vbox` (recognising a cell the way
      `find_cell_row_for_path` does — first child is the thumbnail `Picture`).
      **Placeholder rows carve-out:** "(No images…)" / the truncation notice speak
      through the label, so they always render `Extended` — hiding it would leave
      an unexplained empty grid. Persists via the m4-98d `darkroom_ui_prefs` table
      under `overlay_mode` (token `none`/`normal`/`extended`, shared prefix consts
      so encoder/decoder can't drift), restored **before** the first bind so cells
      are laid out right the first time. 2 new tests (row mapping per mode; index
      ↔ variant bijection + encode∘decode round-trip + corrupt-token fallback).
    - **m4-98c (next) — view-mode switcher. DESIGNED, not yet built.** darktable's
      file-manager / zoomable / culling layouts. This is a multi-increment arc, so
      the design is recorded here before any code:
      - **HARD CONSTRAINT — never swap the `ScrolledWindow`'s child.** Two call
        sites reach *through* the scroller for the grid
        (`lib.rs:601` thumb-size stepper, `lib.rs:714` overlay dropdown, both
        `scroll.child().and_downcast::<GridView>()`). Replacing the child makes
        those downcasts silently return `None` and both controls go inert with no
        error — the same "silent no-op" failure class that has bitten this repo
        repeatedly. **Modes therefore reconfigure the SAME `GridView`** (its model,
        column bounds and scroll policy); they never restructure the widget tree.
        If a mode ever genuinely needs a different widget, fix the coupling first
        by returning the `GridView` from `lighttable_page` instead of re-deriving
        it.
      - **Culling** = the same `GridView` with its model swapped for a
        `gtk4::SliceListModel(model, offset, n)`, `min_columns == max_columns == n`
        and scrolling disabled; arrow keys move `offset` by `n`. This reuses the
        entire cell factory (thumbnail, filename, stars, colour dots, overlay
        modes) for free, and keeps the rating/colour gestures working — which a
        hand-rolled comparison widget would not.
      - **Zoomable** is the awkward one: `GridView` cannot do an infinite zoom
        plane. Defer it; evaluate approximating it as file-manager with a wider
        column range driven by ctrl+scroll, and prefer shipping **full preview**
        (single image, darktable's `w`/`f`) first — more useful per unit of work.
      - Persist the mode in `darkroom_ui_prefs` under `view_mode`, reusing the
        m4-98e token pattern (`ALL`-derived rows + encoder/decoder inversion).
      - **Layout:** the bottom `CenterBox` has only three slots and they are all
        taken (rating filter / overlays / thumb stepper), so the centre slot
        becomes a `Box` holding `[view-mode switcher | overlays]`. Use icon-only
        `ToggleButton`s in a `.linked` box — the bar's minimum width feeds the
        known ~915px overflow, and terse labels were already needed once (m4-98e).
      - **Increments:** (a) `ViewMode` enum + persisted state + the switcher
        control, with only file-manager live and the rest disabled — plumbing
        first, zero behaviour change; (b) culling via `SliceListModel` + arrow-key
        navigation; (c) full preview; (d) zoomable, only if (c) proves it's worth
        it.
      - **m4-98c(a) — done.** `ViewMode` (`FileManager`/`Zoomable`/`Culling`) +
        `view_mode` pref + the switcher, only file-manager live.
        - **The coupling was fixed rather than documented.** The design said to
          return the `GridView` from `lighttable_page` "if a mode ever genuinely
          needs a different widget"; the switcher made it a *third* consumer of
          `scroll.child().and_downcast::<GridView>()`, so it was done now, on
          plumbing, instead of mid-culling. `lighttable_page` returns a
          `LighttablePage { scroll, grid, model, selection }` and nothing
          re-derives the grid — "the bottom-bar controls always find the grid" is
          now true by construction, not by comment. That also removed a bare
          `if let Some(grid) = …` with no `else`, which would have silently
          dropped the rating filter, switcher, overlay dropdown *and* stepper.
        - **`is_available()` gates both ends.** Unported modes are insensitive in
          the switcher *and* refused by `parse_view_mode_token`, so a `view_mode`
          pref written by a later build can't open the lighttable onto a layout
          that draws nothing. Implementing a mode is one edit (flip the arm),
          which lights up the button and the restore path together. The refusal is
          **non-destructive**: a stale `culling` pref is downgraded on read but
          never rewritten, so it will resurrect when culling lands — deliberate
          (it honours what the user asked for), and worth remembering.
        - `store_view_mode` is the single writer of the state, so "the current
          mode is always one this build can render" is enforced in one place; it's
          widget-free, so the gate the persist path depends on is testable with no
          display. `set_view_mode` = that write + `reconfigure_grid_for` (empty
          for file-manager, exhaustive so a new mode can't forget its layout).
        - **The restored mode is applied to the grid explicitly**, once, outside
          the toggle handlers — seeding a button only lights it. Inert today;
          without it, restoring culling would show a lit culling button over a
          file-manager grid with nothing to say so. Handlers are connected in
          phases (build+group → seed → connect → apply) because joining a group
          clears the joiner's active flag, and a `resync()` closure (behind a
          re-entrancy `Cell`, since `set_active` re-emits `toggled`) rolls the
          buttons back if a switch is ever refused — refusing to *persist* while
          leaving the button lit would display a mode never entered.
        - **Tooltips live on the box, not the buttons:** GTK4 never emits
          `query-tooltip` for an insensitive widget (the same finding as the
          header's disabled "Other" view), so per-button text on exactly the modes
          that need explaining would be unreadable. `view_mode_switcher_tooltip()`
          builds one string from `ALL` — pure, and tested to mention every mode.
        - Cost recorded honestly: increment (a) is behaviour-neutral but **not**
          layout-neutral — three icon toggles add ~110px to the bottom bar's
          minimum width, against the known ~915px overflow. No slot arrangement is
          cheaper (`CenterBox`'s minimum is start+centre+end regardless); the real
          fix is an `adw::Breakpoint` hiding the switcher on narrow windows, and
          this control is its first good client.
        - 4 new tests, chosen for what would actually regress: a refused switch
          must not mutate state; `current_view_mode()` is renderable after *any*
          token (GTK's `set_active` ignores sensitivity, so a mode whose button is
          insensitive would pin the group on a button nobody can click off);
          available modes round-trip while unavailable ones decode to file-manager;
          tokens are distinct and every mode is named in the tooltip.
        - Verified live in the container: switcher renders (icons checked against
          the container's Adwaita theme — a missing name draws the broken-image
          glyph, it doesn't fail), file-manager active with the other two greyed;
          **the thumb stepper still steps (6→8) and grid keys/activation still
          work after the plumbing change** — that is the regression the constraint
          guards; and a hand-planted `view_mode=culling` pref restarts into
          file-manager with the grid rendering normally. The overlay dropdown
          renders but its popover can't be driven by synthetic input — GTK
          popovers don't reach the KasmVNC framebuffer capture (same class as the
          black-dialog gotcha); it shares the exact grid binding the stepper
          exercised.
      - **m4-98c(b) — done.** Culling: the same `GridView` with a
        `gtk4::SliceListModel` window over the same base model, paged with ← / →.
        The whole cell factory (thumbnail, filename, stars, colour dots, overlay
        modes) and its gestures keep working, which a hand-rolled comparison
        widget would not.
        - **The window is swapped *inside* the existing `SingleSelection`**, not by
          installing a second selection model on the grid. `selected_path` /
          `reselect_path` and everything built on them then follow the window for
          free; a second selection object would have left every one of them reading
          a stale one, silently and only in culling.
        - **The blocking bug this exposed:** three call sites resolved the
          selection as `lt_model.item(selection.selected())` — the *full* model
          indexed by a **window-relative** index. Two of them were the export
          button and Ctrl+E, so culling into page 5 and exporting would have
          written **a different image from page 1** to disk, with nothing to
          signal it. All three now go through `selected_path`, and no
          selection-reading closure captures `lt_model` any more, which makes the
          bug class unrepresentable rather than fixed. `connect_activate` resolves
          through `gv.model()` for the same reason.
        - **The window is capped by what the viewport can show** in one row
          (`cull_capacity`), because a window wider than the viewport wraps — and
          two rows is not "one screenful side by side", it is the grid again.
          Pinning `min_columns` to force one row was tried and rejected: with the
          scroller's horizontal policy `Never` it converts wrapping into
          *clipping*. Capacity is `None` while the grid is unallocated (the mode is
          restored before the first layout), and the scroller's horizontal
          page-size notify re-fits on allocation and on every resize.
        - **The thumb stepper doubles as the "how many images" control**, and shows
          the count **actually on screen** rather than `max_columns` — a narrow
          viewport holds fewer than asked, and a label counting past what changed
          would be exactly the inert-control shape this repo keeps hitting. It
          deliberately does *not* write the capped value back, so a temporarily
          narrow window doesn't permanently overwrite the chosen thumb size.
        - Selection is carried across every model swap (enter, leave), the entry
          offset is derived from the selected image's page (darktable's behaviour),
          `cull_resync` mutates the installed slice instead of rebuilding it (a
          rebuild resets `SingleSelection` to index 0), `reselect_path` resolves
          against the *base* and moves the window to the image's page, and
          `set_view_mode` now rolls the mode back if `reconfigure_grid_for` fails
          rather than lighting a button over an unchanged layout.
        - Offset safety: paging **stops on the last whole page** (an offset past
          the end renders as an empty grid with no error), and an `items-changed`
          watch on the base re-clamps after any reload that shrinks the collection.
          The watch is tracked and disconnected unconditionally, so it can neither
          stack nor outlive the mode.
        - 6 new pure tests (window size, paging, clamping, capacity, entry offset,
          key mapping) — 194 total. The review's point stands and is recorded here:
          the helpers are arithmetic that was never in doubt, and what actually
          broke lived in GTK wiring that can't be tested without a display. That is
          an argument for deleting untestable surface (which the `selected_path`
          fix does), not for more helpers.
        - Verified live: culling shows one row of 5 at full width and re-fits to 4
          when the metadata panel widens; → pages by exactly one window; the
          metadata panel follows the window (index 4 after paging, not index 0 —
          the export bug's signature); leaving culling restores the scrolling grid
          **with the selection intact**; `view_mode` round-trips through the pref.
        - Known gaps, for (c)/(d): cells stay `THUMB_SIZE`, so this pages a fixed
          set rather than filling the viewport the way darktable does (the tooltip
          says so rather than promising otherwise); and selecting an image can
          re-fit the window when the metadata panel's width changes, which shifts
          the row under the cursor.
      - **m4-98c(c) — done.** Full preview (darktable's `f`): the selected image
        fills the centre view, ← / → step, `f`/Escape dismiss. New
        `lighttable/full_preview.rs`. **Not** a fourth `ViewMode` — it is
        orthogonal to the layout (works from the file manager and from culling),
        and persisting it would mean sessions that open onto a single image.
        - **The runtime-only bug this increment existed to find:** the preview was
          first built as a `gtk4::Stack` page beside the grid. A `Stack` **unmaps**
          the hidden page, GTK drops focus from an unmapped widget, and the
          lighttable's key controller lives on the `GridView` — so opening the
          preview made it a **keyboard trap**: `f`, Escape and the arrows all
          landed on whatever else took focus (the tag entry), with no way out but
          restarting. Every unit test passed throughout. Fixed by making it an
          `Overlay` child *over* the grid, which keeps the grid mapped, focused and
          driving its own controller — no focus juggling and no second controller.
          The layer needs `.background`: `ContentFit::Contain` letterboxes, and a
          transparent letterbox shows the grid ghosting through.
        - **The preview follows the SELECTION**, not just its own keys — the same
          observer shape the metadata panel uses. Otherwise a tag/folder/colour
          click, the timeline, or any reload that drops the previewed image leaves
          a full-screen image beside another image's metadata. `preview_target()`
          is the pure rule (`None` selection ⇒ close, not "hold the old image",
          which is indistinguishable from a hang).
        - Keys that would move the collection *under* the preview are **swallowed**
          rather than forwarded (`PreviewAction::Ignore` for ↑/↓/Home/End; Page
          Up/Down page the preview, not the culling window). At a culling window's
          edge, ← / → **page the window** and land on its near edge — the window is
          only 2..8 images, so stopping there would look like a freeze.
        - `FullPreview` holds only its own child widgets, never the containing
          `Overlay`: the key handler captures it and the grid owns that handler, so
          storing an ancestor would close a reference cycle keeping the whole
          centre subtree (grid, model, every cached texture) alive for the process.
        - Decode size comes from the widget allocation × scale factor (bounded
          512..4096), not a constant — full preview exists to judge focus, and a
          fixed 2048 has the user inspecting resampling artefacts on a 4K display.
          `PixbufLoader::connect_size_prepared` scales *during* decode; `write` and
          `close` are both unconditional, since a loader finalized without `close`
          emits a `g_warning` on every keypress that lands on a rejected file.
        - **Coverage limit, stated rather than hidden:** gdk-pixbuf decoding, so
          the preview shows exactly what the grid's thumbnails show. Raws it can't
          read (`.ORF`) get a centred "No preview available for …" instead of a
          blank page. Follow-up: lift the darkroom view's `BaseImage`/`render()`
          out of `darkroom/mod.rs` into something both views call, decoded
          off-thread (`gdk::Texture` is `Send`, `Pixbuf` is not), then add
          darktable's 100 % zoom/pan.
        - 4 pure tests (key gating, the swallow set, the target rule, step bounds
          incl. `INVALID_LIST_POSITION` — clamping that sentinel would silently
          select the second-to-last image). 198 total.
        - Verified live, by pressing the keys: `f` opens on the clicked image with
          the metadata panel in step; → moves to the next image and the panel
          follows; Escape closes and the grid returns with the selection on the
          previewed image; an `.ORF` shows the centred message with no pixbuf
          warnings in the container log.
      - **Live checks** (xdotool synthetic input is unreliable here — see the
        m4-99b lessons): switch each mode and confirm the thumb stepper AND
        overlay dropdown still work afterwards (that is the regression the hard
        constraint above guards); culling with fewer images than `n`; culling at
        the collection's last page; a mode restored from the pref at startup —
        and specifically that the restored mode **reconfigures the grid**, not
        just lights its button.
    - Also queued for this bar: colour quick-filter (compose-on-top would overlap
      the existing left-panel colour selector — reconcile first) and a
      "clear all filters" reset.
  - **m4-99 — date timeline (done).** darktable's bottom date-histogram strip:
    one bar per year sized by image count, year labels, click a bar to filter to
    that year, click it again to clear. New `lighttable/timeline.rs`.
    - **Date decoding:** `images.datetime_taken` is **µs since 0001-01-01** (a
      GLib `GDateTime`/`GTimeSpan` origin), *not* a Unix timestamp —
      `DT_EPOCH_OFFSET_SECS = 62_135_596_800`. Pinned empirically against the real
      catalog: `P7280008.ORF` in film roll `…/2018_07_28` decodes to 2018-07-28
      (the year-0 origin variant is a year off), and a test drives that value
      through SQLite itself.
    - **One expression, two uses:** `year_sql_expr()` (SQLite `strftime` over the
      converted seconds) is the single source of truth — the histogram groups by
      it and the filter compares against it, so the bar you click can't disagree
      with the rows you get. Undated rows (`datetime_taken <= 0`) are excluded
      from both; SQLite would otherwise decode 0 as year 1 and bucket them.
    - **Continuous axis:** `fill_year_gaps()` inserts zero-count years, so a gap
      (the real catalog has no 2017) doesn't render as if 2016 and 2018 were
      adjacent. Capped at `MAX_TIMELINE_SPAN_YEARS` so one corrupt date can't
      allocate thousands of bars and squash the real data flat.
    - **Filter plumbing:** new `YEAR_RANGE` thread-local + `set_year_range()`
      routed through the m4-97c observer bus (so the strip's highlight tracks
      clears made elsewhere). The loaders' spliced fragment is now
      `current_filters_sql()` = rating + year, so adding a compose-on-top filter
      never means touching the four loaders again.
    - Verified live: strip renders 2015–2026 with 2017 empty; clicking 2026 gives
      exactly 72 images (matching the histogram) and highlights that bar while
      dimming the rest; clicking again restores 2000. 8 new tests.
    - **m4-99b (done):** drag across bars to select a **multi-year span**, with a
      live preview of the span under the cursor before it commits. A *single*
      `GestureDrag` serves click and drag — a release within `DRAG_CLICK_SLOP_PX`
      of the press is a click — rather than two gestures fighting to claim the
      sequence. `bar_span()` clamps both endpoints into the strip (a drag off the
      edge selects out to it) and orders them, so a right-to-left drag selects the
      same span; `span_has_images()` extends the click's dead-end guard to a whole
      span of gap years. **Runtime-only bug found by probing the container logs:**
      `GestureDrag::start_point()` is valid during `drag-update` but returns `None`
      by `drag-end` (the gesture has reset), so the end handler bailed and no span
      was ever applied — the click path masked it. Fixed by latching the press x in
      `drag-begin`. No unit test could have caught this (pure helpers were already
      green); only reading the logs did. Verified live: dragging 2025→2026 yields
      exactly 180 images (108 + 72) and highlights both bars.
      Review fixes (Opus): the branchy commit logic is now a pure, tested
      `drag_intent()` (slop classification, gap-year dead-end guard, toggle,
      span→years) — the leaves were tested but the *combination* was where a
      regression hid; `bar_span` clamps in **index** space (the old
      `width - f64::EPSILON.max(1e-9)` had a dead `EPSILON` term, panicked for
      sub-pixel widths and no-op'd for huge ones); re-dragging an applied span now
      clears it, so span and click are symmetric; and a `cancel` handler stops a
      killed sequence latching the preview forever (it would show a span that was
      never applied *and hide* the filter that is).
      **Two runtime-only lessons, both from log probes:**
      (1) unifying `GestureClick` into `GestureDrag` silently dropped the
      `n_press > 1` double-click guard a previous review had added — the second
      press arrived as an independent click and toggled the first straight back
      off, so the strip looked inert. `GestureDrag` has no `n_press`, so it's
      reconstructed by time via a pure `is_repeat_click()` against GTK's own
      `gtk-double-click-time`.
      (2) GTK emits **`cancel` BEFORE `drag-end`** on an ordinary button release
      here, so the first `cancel` handler — which also cleared the start latch —
      made every completed drag bail out of `drag-end`. Cancel must clear only the
      preview; the latch is self-healing because `drag-begin` always precedes the
      next read. Verified live after the fixes: drag ⇒ 180, re-drag ⇒ cleared
      (2000), double-click ⇒ stays on (72).
    - Follow-ups: month/day zoom levels; rebuild the histogram after an import.
  - **m4-100 — image information (done).** The right panel showed only File /
    Folder / Size / Disk; it now carries darktable's "image information" EXIF:
    **Camera, Lens, Exposure, Aperture, ISO, Focal, Taken**, read from the
    catalog's `makers`/`models`/`lens` lookup tables and the `exposure`/`aperture`/
    `iso`/`focal_length`/`width`/`height`/`datetime_taken` columns.
    - **One query, one connection:** `query_exif` replaces the old `query_dims`
      (which opened its own connection for two columns); dimensions now ride along
      with the rest. Split `query_exif` / `query_exif_conn` per `persist.rs`'s
      `_conn` idiom, so the SQL is testable against an in-memory catalog with no
      temp files or new dev-dependency.
    - **Shared date decode:** the epoch shift moved into
      `timeline::unix_secs_sql_expr()`, now the single place it's written — the
      panel and the timeline cannot disagree about when a photo was taken.
    - Pure, tested formatters: `format_exposure` (reciprocal below 1 s — `1/60 s`
      — decimal above), `format_aperture`/`format_focal`/`format_iso` (round real
      values like `2.79999995` → `f/2.8`, drop a trailing `.0`), `format_camera`
      (drops a maker the model already repeats, on a word boundary, so
      `Canon` + `Canon EOS 5D` isn't `Canon Canon EOS 5D` while `Canonball`
      survives), and an em-dash placeholder everywhere a value is absent — so a
      missing field reads as "unknown", never as a blank the user must interpret.
      Long values (lens names) get a tooltip, since the labels ellipsize at 20 ch.
    - Review fixes (Opus), two of them blocking:
      (a) a catalog predating the `makers`/`models`/`lens` tables failed the
      statement at **prepare** time, blanking *every* field — including the
      dimensions that worked before this query absorbed them (a regression vs the
      deleted `query_dims`). Per-column NULL handling can't cover that, so the
      tables are probed once via `sqlite_master` and the three columns/joins are
      dropped when absent: camera/lens degrade alone. Regression-tested.
      (b) `format_camera` doubled up on the two duplications that actually occur —
      `maker == model` ("DJI"/"DJI" ⇒ "DJI DJI") and the corporate-suffix shape
      ("NIKON CORPORATION"/"NIKON D850"), because it matched the *whole* maker
      string. Now matches the maker's first word and accepts an empty remainder,
      while "Canonball" still survives.
      Also: the read's `busy_timeout` dropped 3s → 250ms because this runs on the
      GTK main thread on every selection change and `library.db` is in
      rollback-journal mode (no WAL), so a reader really blocks on an in-flight
      rating write — a dash for one frame beats a frozen window; tooltips are set
      from the formatted string (not read back off the widget) and suppressed for
      placeholders; `format_exposure` guards the reciprocal against denormals
      saturating `as i64`; `ExifInfo` is `pub(crate)`.
    - Verified live against the real catalog: `Camera OnePlus One A0001`,
      `Exposure 1/60 s`, `Aperture f/2`, `ISO 211`, `Focal 3.8 mm`,
      `Taken 2016-04-16 17:37:19` — the date independently corroborated by the
      file's own name (`IMG_20160416_173714`) — and an Olympus ORF reading
      `4640 × 3472`, `Olympus M.Zuiko … 45mm F1.8`, `f/2.8`, `ISO 640`, `45 mm`,
      with `Taken —` correctly shown for its NULL `datetime_taken`. 8 new tests.
  - Later: left-panel modules (import as a module, hierarchical collections,
    image-information, Lua scripts), right-panel modules (history stack, styles,
    metadata editor, geotagging, export as a panel), and the date timeline.
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
   commit. darkroom-core pipeline 15→21 tests.

   **m4-74** (`7a98fb2085`) — Rec.2020↔XYZ↔Lab color infra + **faithful Lab-`L`
   Sharpen** (first slice of the color-space-infrastructure scope). `color.rs`
   gains `rec2020_to_xyz_d65`/`xyz_d65_to_rec2020` (standard BT.2020 D65, the
   missing primitive) + `rec2020_to_lab`/`lab_to_rec2020` (compose the existing
   CAT16 + XYZ↔Lab, carrying alpha — `apply_transposed_color_matrix` sums ch 0..2
   so CAT16 zeroes ch 3). Sharpen upgraded from the m4-73 luminance-USM to the
   bit-faithful darktable path: Rec.2020→Lab, unsharp-mask **L only** (a/b
   untouched ⇒ no chroma shift), threshold now Lab-`L` [0,100] (0..100 UI slider
   maps straight through); `amount==0`/small images short-circuit byte-exact. Two
   dev bugs caught+fixed (inverse matrix stored un-transposed; alpha dropped
   through CAT16). Opus architect independently re-derived the matrices from
   primaries — correct + genuine inverses to ~5e-8; SHIP, no blockers.
   darkroom-core 549→563 tests. **HARD PREREQUISITES before Sharpen gets a live UI
   caller** (both latent now — no caller): ~~**(M1)** make the pipeline working
   space an explicit contract~~ **DONE — m4-75**; ~~**(M2)** scale the Gaussian
   sigma by the pipeline `roi->scale`~~ **DONE — m4-76** (`Stage::Sharpen.scale`,
   effective radius `radius*scale`, faithful to sharpen.c). Other follow-ups:
   planar-L kernel (perf); per-stage band strategy.

   **m4-76** (`c643c8c64b`) — **M2**: `Stage::Sharpen` gains a `scale: f32` (the
   ROI resolution factor, 1.0 export / <1 downscaled preview); apply feeds
   `gaussian_kernel(radius * scale)` so tap-count + sigma2 both scale with the
   ROI (faithful to sharpen.c; darkroom's single-buffer model collapses `iscale`
   to 1). threshold/amount stay unscaled (value-domain). `radius*scale==0`
   early-out (sharpen.c rad==0 parity) + a `scale>0` debug_assert. Opus architect:
   SHIP, faithful. pipeline 22→23 tests. **The Sharpen STAGE is now
   feature-complete (radius/threshold/amount/space/scale)** — both M1 & M2 done,
   so Sharpen is fully ready for a live caller. Remaining to expose it (a separate
   increment): the delicate `PreviewParams` codec bump (v2→v3) + `to_pipeline`
   (needs space+scale args) + `history` exhaustive-match/describe_change + export
   wiring, then a UI slider (display-gated). This touches persistence, so it's
   best done as its own focused increment.

   **m4-75** (`ff8d50e93a`) — pipeline **working-space contract** (resolves M1).
   `color.rs` gains `srgb_to_lab`/`lab_to_srgb` (linear-sRGB twins of the Rec.2020
   pair; the sRGB matrix is Bradford-D50 by design → no CAT16, alpha survives).
   `pipeline.rs` gains `pub enum ColorSpace { Rec2020, LinearSrgb }`; `Stage::
   Sharpen` gains a `space` field and picks the RGB↔Lab pair per space, so the raw
   (Rec.2020) and non-raw (linear-sRGB) paths both sharpen L with correct
   primaries. Chose enum-on-the-stage over re-churning the m4-73 process()
   signature; a `working_space()` helper + a debug_assert in process() enforces
   "one buffer, one working space" as stages grow. New test proves `space` routes
   the conversion (coloured edge: Rec2020 ≠ LinearSrgb) + alpha (0.5) survives both
   paths. Opus architect: SHIP, no blockers. darkroom-core 563→565 tests.
   **Sharpen is now unblocked for a live UI caller once M2 (roi->scale) lands.**

   **m4-77** (`d65d7bd60f`) — **3-D bilateral grid ported** (`darkroom-core/
   bilateral.rs`, faithful port of `src/common/bilateral.c`): shared edge-aware-
   filter infra for lowpass/shadhi/retouch/monochrome/globaltonemap/colormapping/
   ashift/bilat. `Bilateral::new/splat/blur/slice` (+ `slice_to_output`); the
   architect re-derived both load-bearing claims against the C (serial splat ≡
   the C's per-thread-slice-then-merge up to float add order; per-line start
   `k*offset1+j*offset2` ≡ the C's running accumulator, and avoids a usize
   underflow that panics in Rust). **Rust-vs-C hardening:** a spatial axis can
   collapse to size 2 on a narrow crop (4px → grid 2×301×6) where the C blur does
   a benign OOB read but Rust panics — guarded to no-op below 4 grid points.
   No consumer wired yet (algorithm+tests first, like interp.rs). Opus architect:
   APPROVE after 3 fixes (premise/consumer correction, `slice_to_output`, the
   panic guard — all applied). darkroom-core 566→573 tests. Follow-up: a
   golden-vector test vs a C `buf` dump. **This is the first of the ~5 shared-infra
   pieces gating the remaining ~230 C loops** (see the "what blocks" table above).

   **m4-78** (`01d0568432`) — **recursive Gaussian ported** (`darkroom-core/
   gaussian.rs`, faithful CPU-path port of `src/common/gaussian.c`): the second
   shared-infra piece, unblocking bloom/highpass/lowpass/shadhi/hazeremoval.
   `compute_params` ← `_compute_gauss_params` (Young–van Vliet IIR coefficients,
   all three orders); `Gaussian::blur_4c` ← `dt_gaussian_blur_4c` (RGBA two-pass
   separable IIR — vertical into a temp buffer, then horizontal into output; each
   axis sums a forward causal + backward anti-causal pass, per-channel clamp on
   read). Not ported: generic N-channel `dt_gaussian_blur`, the `_fast_9x9`
   small-sigma direct-conv path, all OpenCL. Opus architect re-derived every
   coefficient + recurrence against the C (exact, no transposed variables) and
   found **one real Rust-vs-C divergence**: the naive `clampf` propagates a NaN
   that the IIR then smears across the whole column/row, whereas C's asymmetric
   `CLAMPF` scrubs NaN to `min` (local) — fixed to mirror the C branch order.
   Also guards empty images. 10 module tests incl. an **analytic impulse-response
   test** (blurred unit impulse ≈ `exp(-r²/2σ²)/(2πσ²)` within 1.5e-3 — the guard
   a transposed recursion variable can't pass), NaN-to-min, all-orders-finite,
   channel independence, degenerate-dims no-panic. 582 darkroom-core tests; clippy
   clean. **Second of the ~5 shared-infra pieces; ~3 remain** (Filmlight-Yrg,
   per-pixel ICC/LCMS, colorreconstruction bespoke grid; dwt.c also outstanding).

   **m4-79** (`396c8ab491`) — **à-trous wavelet decompose/denoise ported**
   (`darkroom-core/dwt.rs`, faithful CPU-path port of `src/common/dwt.c` — the
   GIMP "Wavelet Decompose" algorithm): the third shared-infra piece, unblocking
   **atrous, retouch, denoiseprofile** (wavelet mode). Ported: `decompose` ←
   `dwt_decompose` (RGBA `ch==4`, dilated 3×3 hat kernel per scale, per-scale
   layer **callback** `FnMut(&mut [f32], &DwtParams, i32)` for the original /
   each detail / residual / reconstruction) with the full `dwt_wavelet_decompose`
   orchestration (buffer ping-pong, `merge_from_scale`, `return_layer`, layers /
   merged_layers accumulation); `get_max_scale`/`first_scale_visible`; and the
   1-channel `denoise` ← `dwt_denoise` (soft-thresholded wavelet denoise). Not
   ported: all OpenCL (`dwt_*_cl`). **Design notes:** (1) C's aliased
   `buffer[0]=p->image` pointer swap is replaced by an owned-Vec ping-pong that
   `copy_from_slice`s the chosen result back at each `dwt_get_image_layer` site —
   the architect walked all five write-back paths + the scale-0 in-place
   semantics and confirmed result-identical. (2) The `dwt_interleave_rows`
   cache reorder is **omitted** (each vertical-pass output row reads only *input*
   rows never mutated during the pass, so visiting order can't change the result
   — verified airtight). (3) **Rust-vs-C hardening:** every reflected edge tap is
   clamped into range; `dwt_decompose` clamps `scales` to `get_max_scale()` so on
   its path the clamp is a proven **no-op** (bit-identical to C), diverging only
   where C reads OOB (benign UB) on degenerate inputs. **F1 for consumer
   wire-up:** `denoise` does *not* clamp `bands` — clamp to `get_max_scale()` at
   the denoiseprofile/retouch/atrous call site so production never enters the
   OOB-in-C regime. Opus architect: **APPROVE** (10/10 faithfulness, re-derived
   reflection-index parity + telescoping reconstruction + ping-pong selection vs
   the C himself); applied F2 (negative-`return_layer` guard) + F3 (3 value-parity
   tests: detail-scales+residual telescoping, zoom `max_scale` clamp, merged-scale
   ≡ sum-of-details). 13 module tests, **595 darkroom-core tests**, clippy clean.
   **Third of the ~5 shared-infra pieces; ~2 remain** (Filmlight-Yrg /
   `work_profile`; per-pixel ICC/LCMS; colorreconstruction's own {L,a,b,weight}
   grid is a separate bespoke port).

   **m4-80** (`a1793b76ce`) — **colorreconstruction's bespoke 4-field bilateral
   grid ported** (`darkroom-core/colorreconstruct.rs`, faithful port of the
   private grid in `src/iop/colorreconstruction.c`): the fourth shared-infra
   piece, unblocking **colorreconstruction**. Distinct from the shared 3-D
   bilateral grid (m4-77): a full `{L,a,b,weight}` `Cell` per grid point, an
   **x-fastest** index (`xi + size_x·(yi + size_y·zi)`), **nearest-integer**
   splat (not trilinear), and a plain `[1 4 6 4 1]/16` Gaussian on all three
   axes (no derivative-z pass). Ported: `new` ← `..._init` (grid sizing
   `clamp(round(dim/σ),4,MAX)+1` → always ≥5, so no collapsed-axis OOB;
   effective-σ recompute), `splat` ← `..._splat` (skips `L>threshold`;
   `Precedence` none/chroma/hue weighting; serial ≡ the C's per-thread atomic
   adds up to float add order), `blur`/`blur_line` ← `..._blur` (running-buffer
   separable Gaussian on `Cell` via `Add`+`Mul<f32>`; per-line start recompute
   avoids the usize underflow), `slice` ← `..._slice` (trilinear read-back,
   `blend = CLAMPS(20/thr·L−19,0,1)`, `aout·Lin/lout` chroma reconstruction; **L
   and alpha pass through untouched**). Not ported: pixelpipe grid freeze/thaw
   caching (FULL-vs-preview grid stealing — plumbing), `hue_conversion` (caller
   supplies the LCH hue via `Precedence::Hue`), all OpenCL. `CLAMPS` matches the
   C branch order exactly (**NaN → low bound**), applying the m4-78 gaussian
   `CLAMPF`-NaN lesson proactively. Opus architect: **APPROVE-WITH-FIXES** —
   re-derived grid sizing, splat index convention, `blur_line` carry, the
   trilinear max-index (`= len−1`, exactly tight, no OOB), and the CLAMPS NaN
   choice; one **LOW** finding (`interp` factors the trilinear weight product,
   ~1 ULP/tap vs the C's field-first grouping — immaterial since the splat's
   atomic non-determinism dwarfs it), resolved by documenting per the reviewer's
   preferred option. 8 module tests (incl. the behavioral
   clipped-highlight-borrows-neighbour-colour); **603 darkroom-core tests**;
   clippy clean. **Fourth of the ~5 shared-infra pieces; ~2 remain**
   (Filmlight-Yrg / `work_profile` → colorbalancergb/colorin; per-pixel ICC/LCMS
   → colorin/colorout/retouch — the biggest and needs a colour-management
   approach). Also unported but NOT gating any "what blocks" row:
   `guided_filter.c`/`eigf.h`/`fast_guided_filter.h`, `nlmeans_core.c`.

   **m4-81** (`0b319b907e`) — **colorbalancergb colour helpers** (`color.rs`):
   groundwork for porting `colorbalancergb` (darktable's most complex IOP — no
   Rust process module yet, only a geometry stub). The Filmlight Yrg/Ych/LMS/UCS
   conversions were already in `color.rs` (from filmic v4); this adds the 8
   fixed-matrix / scalar helpers the process loop additionally needs: `lms_2006_↔
   xyz` (`LMS_to_XYZ`/`XYZ_to_LMS`, CIE-2006-LMS↔XYZ D65), `lms_↔grading_rgb`
   (reusing the existing `FILMLIGHT_*_T` constants — literally the same
   matrices), `soft_clip`, `dt_ucs_jch_↔hcb`, and `xyz_to_jzazbz` (the JzAzBz
   forward; the reverse `jzazbz_to_xyz_d65` already existed). Opus architect:
   **APPROVE** — every constant/matrix-row/transposition/formula checked
   bit-for-bit vs the C headers, 10/10 parity, no fixes. 5 round-trip tests incl.
   the strong anchor (`xyz_to_jzazbz` ∘ the independently-written
   `jzazbz_to_xyz_d65` recovers input within 2e-3). 608 darkroom-core tests.

   **m4-82** (`89d42e2ef7`) — **colorbalancergb per-pixel helpers** (`color.rs`):
   the 4 remaining small conversions both `commit_params` and the process loop
   need — `make_ych` (`make_Ych`), `ych_to_grading_rgb` (`Ych_to_gradingRGB`,
   Ych→Yrg→LMS→grading-RGB), `opacity_masks` (sigmoidal shadow/midtone/highlight
   masks + complement), `lookup_gamut` (cyclic hue→LUT linear interp) + the
   `LUT_ELEM = 512` const. 4 tests (fulcrum-symmetry 0.5/0.5/0.5, complement,
   linear-LUT index/interp exactness, hue=0→index 256, constant-LUT wrap). 612
   darkroom-core tests; clippy clean. **Review debt CLEARED** — the
   fricktrade-architect review was first quota-blocked (account session limit) so
   the commit went in on a bit-exact self-review; the formal Opus review was
   re-run after the 08:50 reset and returned **APPROVE** (all 5 ports bit-exact:
   opacity_masks signs/grouping, lookup_gamut wrap incl. `ceil==512→0` & negative
   two's-complement, `floor`/`ceil` double-vs-f32 immaterial for |x|<512,
   slot/compose orders). Optional deferred hardening: type `lookup_gamut`'s LUT as
   `&[f32; LUT_ELEM]` once the m4-84 process-loop caller lands.

   **m4-83** (`f2de894e44`) — **colorbalancergb gamut-boundary LUT builders**
   (new module `crates/darkroom-core/src/iop/colorbalancergb.rs`): the two
   `hue → max-saturation/colourfulness` LUT builders the process loop's
   `lookup_gamut` reads. `build_gamut_lut_jzazbz` (the `STEPS³`=92³ RGB-cube →
   JzAzBz → max-sat-per-hue + 5-tap box AA, ← `commit_params` JzAzBz branch) and
   `build_gamut_lut_ucs` (marches the RGB gamut boundary in xyY, accumulating
   dt-UCS colourfulness² per hue bin, ← `dt_UCS_22_build_gamut_LUT`), plus
   `hue_index`/`delta_h`. **Matrix contract:** both take the **transposed**
   RGB→XYZ-D65 matrix (C uses non-transposed `dot_product`; these use
   `apply_transposed_color_matrix`) — **the m4-84 caller MUST pass the transpose
   of the C `input_matrix` = `(XYZ_D50→D65_CAT16 · matrix_in)ᵀ`**; getting it
   wrong yields a silent wrong LUT (test via primary-hue-angle sanity). Opus
   architect: **APPROVE** — bit-faithful to the *serial* C, derived line-by-line
   (both AA boundary sets, all 3 UCS edge-intersection formulas, `t==clamp` NaN
   parity). Caveat (documented in code): the C marches the UCS builder with an
   OMP `reduction(+:)` whose FP add order is thread-count-dependent, so any golden
   dump must use `OMP_NUM_THREADS=1`. 5 property tests; 617 darkroom-core tests.

   **m4-84** (`c200e4ae02`) — **colorbalancergb `commit_params`**
   (`CbRgbData::from_params`): derives the process-ready data from `CbRgbParams`
   (v5 params + darktable `$DEFAULT`s; GUI checker fields omitted) — the 4
   colour-balance vectors (`make_ych`+`ych_to_grading_rgb` + offset/slope/reciprocal
   formulas around the achromatic ref), weights/fulcrums (`white=2^p`,
   `mask_grey=p^0.41012`, `midtones_weight=sqf·sqf/(sqf+sqf)`), `contrast=1+p`,
   `midtones_Y=1/(1+p)`, and gamut-LUT selection by `saturation_formula`. Added pub
   `color::REC2020_TO_XYZ_D65_T4` (the transposed RGB→XYZ-D65 matrix for the m4-83
   builders; substituting the fixed Rec.2020 matrix for the C's
   `CAT16·matrix_in` is the m4-75 fixed-working-space consequence). Opus architect:
   **APPROVE** — every derivation bit-exact incl. the `chroma/saturation/brilliance`
   data-slot order `[shadows,midtones,highlights]` (a param-vs-slot ordering trap)
   and global's `·global_Y` vs shadows/highlights' `+*_Y`. 9 module tests (neutral
   → no-op balance `global=[0;4]`, zones=`[1;4]`; formula-selects-builder;
   LUT-depends-on-matrix). 621 darkroom-core tests.
   **m4-85** (`2f75180f0b`) — **colorbalancergb main process loop** (`process()` +
   `saturation_jzazbz` + `saturation_dtucs`), port of `:662–943`. Completes the
   **colorbalancergb ALGORITHM** (m4-81 conversions → m4-82 helpers → m4-83 gamut
   LUTs → m4-84 commit_params → m4-85 process loop, all Opus-**APPROVE**). Chain:
   clip → CIE-2006 LMS → Yrg/Ych (opacity masks, 2×2 hue rotation, chroma+vibrance,
   `gamut_check_yrg`) → grading-RGB colour balance (offset / 2 masked slopes /
   sign-preserving midtones power) → Y gamma+fulcrumed contrast → XYZ-D65 → the two
   saturation branches (JzAzBz eigenvector-rotation + Iz/`AI_trans` gamut-clip; dt-UCS
   HCB-rotation + M²-LUT gamut-map) → pipeline Rec.2020. **Pipeline-vs-C:** the C's
   premultiplied pipeline↔LMS matrices become direct compositions of the fixed
   Rec.2020↔XYZ-D65 conversions (m4-75 contract); **alpha preserved from input**
   (C's matrices zero it — documented, no colour impact); GUI checker omitted. Opus
   architect: **APPROVE** — both saturation branches derived line-by-line (rotation
   signs, `SO[1]` clamp order + original-`SO[0]` use, `AI_trans` cols, HCB rotation,
   `max_chroma` powers). 14 module tests (neutral-preserves-grey both formulas
   within 3e-2 — colorbalancergb always gamut-maps so not bit-identity;
   finite/non-neg; alpha; param effects; multi-pixel). 626 darkroom-core tests.

   - **m4-86 (optional, remaining):** wire `colorbalancergb::process` as a live
     `pipeline::Stage` (enum variant carrying `CbRgbData`, `is_pixel_local()=true`,
     applied in `Pipeline::process`) so preview/export actually run it — needs
     params plumbing from the edit/history layer (the algorithm is ported at the
     darkroom-core level like the other IOP modules until then). Deferred m4-82
     hardening to apply here: type `lookup_gamut`'s LUT as `&[f32; LUT_ELEM]`.

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
