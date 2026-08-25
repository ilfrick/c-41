# C-41 — progress log

Append-only, newest last. One entry per unit of work that reaches a remote,
timestamped in UTC. Entries record what changed, how it was verified, and
anything that went wrong — failures and corrections are kept, not tidied away,
because the mistakes are the part worth rereading.

Format: `## YYYY-MM-DD HH:MM UTC — <id>: <title>` followed by what/verify/notes.

---

## 2026-08-10 20:46 UTC — m4-107: Color zones (LCH equaliser) live preview stage

**Commit** `e2c5a8f5` (GitHub + Gitea)

**What.** Wired the colorzones IOP into the Rust preview pipeline.
- `c41-core/iop/colorzones.rs`: ported `build_lut` from
  `CurveDataSampleV2`/`CurveDataSampleV2Periodic` (`src/common/splines.cpp`) —
  all three spline types (Catmull-Rom, natural cubic via banded/full LU without
  pivoting, monotone Hermite: Fritsch-Carlson non-periodic, SIAM `G()` variant
  periodic) and both boundary modes. Quantisation reproduces the C two-step
  path: `round(clamp(s(x),0,1) * 65535)` as u16, then `/0x10000`.
- `pipeline.rs`: `Stage::ColorZones`. Three 65536-entry LUTs are 768 KB, so
  `Stage` lost `Copy` (kept `Clone`) and `apply` matches by reference.
- `preview.rs`: PreviewParams +1 bool +57 f32; ENCODE_VERSION 9→10, len 157→386.
- `history.rs`: "Color zones" group. `darkroom/mod.rs`: module row.

**Verified.** Docker check/clippy/test — but see the failure below.

**Notes.** Senior review found no blockers; its two actionable notes (ColorZones
missing from the pixel-locality test, zero-anchor case diverging from C) were
fixed before commit.

---

## 2026-08-10 21:01 UTC — m4-107 fixup: clippy `erasing_op`

**Commit** `e3609b5d` (GitHub + Gitea) — **CI had failed on `e2c5a8f5`**

**What.** Three deny-by-default `erasing_op` errors in the periodic branch of
`compute_smooth_cubic_tangents`: column 0 of the matrix written as the literal
`n * 0`. Indexing was arithmetically correct (column-major `row + n*col`, so
column 0 *is* the row index) but the redundant form trips the lint. Rewritten to
bare row indices with the (row, col) position in trailing comments.

**Verified.** All four CI steps in Docker. CI green on the fixup.

**Notes — why it escaped.** The local clippy check grepped the `--> file:line`
locator lines for "error". Clippy prints the diagnostic on the line *before* the
locator, so the grep could never match and reported success on a failing run.
Debug-only tests were also run, while CI runs `--release` plus a binary link.
This is the direct cause of `scripts/ci-local.sh` (2026-08-11).

---

## 2026-08-11 13:35 UTC — m4-108: Levels (black/grey/white + gamma) live preview stage

**Commit** `ebb4c7a6` (GitHub + Gitea) — CI green first attempt

**What.**
- `c41-core/iop/levels.rs`: ported `build_lut` from C `compute_lut`
  (`inv_gamma = 10^((grey-mid)/delta)`, `lut[i] = 100*(i/65536)^inv_gamma` — the
  65536 divisor matches C's `(float)0x10000ul`). Documented and debug-asserted
  the `black < grey < white` contract darktable enforces in its GUI; clamped
  `tmp` to `[-1,1]`.
- Bounded the overrange (`pct > 1`) branch at `L_MAX = 1e6`.
- `pipeline.rs`: `Stage::Levels` — Lab-domain, pixel-local, in the locality test.
  Debug-asserts LUT length; release degrades to passthrough rather than panicking
  inside rayon's `for_each_init`.
- `preview.rs`: v11, len 386→399. `to_pipeline` clamps the grey stop. One
  `levels_stage_active` predicate shared by `is_identity` and `to_pipeline`.
- Placed **after** sigmoid (iop_order.c pos 49), not before.

**Verified.** All four CI steps in Docker; 972 tests (+9). CI green.

**Notes — two bugs and two self-corrections.**
1. *(review found)* Three independent sliders reach `black=0, grey=100, white=1`
   — legal positions passing the `white > black` guard — giving `10^199` = `+inf`
   → NaN across the preview. darktable's GUI structurally prevents this; three
   free sliders don't. The old "degenerate range" test passed only because its
   single triple happened to give `tmp = -1`.
2. *(found while fixing 1)* Even with gamma bounded, a narrow range drives
   `pct` large and `100*pct^gamma` is unbounded in the C. Lab→XYZ cubes L, so
   past ~8e14 it overflowed. darktable never hits it because levels runs
   display-referred (`L ≤ 100`); our pipeline can feed it scene-linear data.
3. *(my error)* Placed Levels before sigmoid citing "iop_order.c pos 30" — a
   number never verified. Real position is 49, *after* sigmoid. Not cosmetic:
   pre-tone-map it crushes exactly the highlights sigmoid rolls off.
4. *(my error)* Asserted "raising the black point darkens a midtone". It doesn't
   when grey stays put — moving black moves the midpoint, so a stationary grey
   becomes off-centre and *brightens*. The code was right; the assumption wasn't.

---

## 2026-08-11 14:50 UTC — Workflow: progress log, refreshed parity audit, local CI gate

**What.** Three process changes, no pipeline code.
- **`PROGRESS.md`** (this file): timestamped, append-only, pushed to both remotes.
- **`PARITY_AUDIT.md`** refreshed. It was dated 2026-08-08 and still listed as
  open several items that landed on 08-08/08-09 — severity 1.1 (resizable
  panels), 1.2 (collapsible panels), 1.4 (startup metadata). A stale priority
  document is actively misleading: it caused me to propose reworking finished
  items. Verified each claim against the code before rewriting.
- **`scripts/ci-local.sh`** runs the four CI steps in the CI image and keys off
  exit codes, never grepped text. **`scripts/pre-push`** gates every push on it;
  install with `git config core.hooksPath scripts`, bypass with `--no-verify`.

**Verified.** `scripts/ci-local.sh` run end to end; hook exercised on the push
that carries this entry.

**Notes.** The gate exists because of the m4-107 failure above — origin has dual
push URLs so a bad push lands on both remotes at once, and CI logs are 403 to
this account, which makes local detection the only cheap signal.

---

## 2026-08-11 21:15 UTC — parity-2.1a: make module-panel progress visible

**Commit** pending (GitHub + Gitea)

**What.** No new processing — this makes already-shipped work *visible*.
User report: "the UI shows no progress towards feature parity, it has been the
same for several weeks". Investigated; the report was correct and the cause was
presentation, not the build.

The panel rendered all 44 catalogue entries identically. 14 were live; the other
30 were `inert_module_row` placeholders — same row, same **interactive** switch,
same styling. Flipping one did nothing, which reads as "the app is broken"
rather than "not wired yet". Worse, the catalogue is authored in darktable's
presentation order, which front-loads unported modules: Base opened with three
dead rows, Effect with nine. A top-down scan hit placeholders first and Exposure
sat at position 6. Each shipped increment added one working module to a crowd of
44 lookalikes — invisible from outside.

- `LIVE_MODULE_LABELS` promoted from `#[cfg(test)]` to the real source of truth;
  new `is_live_module()` drives ordering and the count.
- Live modules sort to the top of each group (stable within each half).
- Inert rows: dimmed, subtitle "not yet wired", **no switch** (an icon reads as
  status; a disabled switch still invites a click), `set_sensitive(false)`.
- Header shows "N of M active".
- Crop and Rotate & perspective are *implemented* but driven by their own
  controls (Crop button, Straighten slider), so they get an "elsewhere" row
  pointing at those rather than a false "not yet wired".
- New test parses the dispatch arms from source and pins them against
  `LIVE_MODULE_LABELS`, so a module can't render live while being counted inert.

Result: **16 of 44 active**, working modules first in every group.

**Verified.** `scripts/ci-local.sh` (all four steps); 208 UI tests. Binary
hot-deployed to the running container and the panel model verified against the
rendering logic.

**Notes — process failure, not just a UI bug.** I had never once launched the
app. Every increment was verified by tests and CI and reported as shipped, while
the result was invisible on screen. Tests and CI measure whether code is
correct; they say nothing about whether a user can see it. The user's measure
was the right one. Visual verification is now part of the loop.

---

## 2026-08-12 07:40 UTC — m4-109: Vignetting live preview stage

**Commit** pending (GitHub + Gitea)

**What.** Wired the vignette IOP into the preview pipeline — the **17th** live
module, and the first *position-dependent* one.
- `c41-core/iop/vignette.rs`: new `commit_geometry()` porting the geometry
  block at the top of vignette.c `process()` — xscale/yscale (auto-ratio or the
  0..2 w/h encoding), the pixel-space centre, `dscale`, the size-dependent
  `min_falloff` floor, and `exp1`/`exp2` from `shape`.
- `pipeline.rs`: `Stage::Vignette`. Works in RGB, so `working_space()` is `None`.
- `preview.rs`: 7 params + on/off; ENCODE_VERSION 11→12, len 399→428. Pushed
  last (iop_order.c pos 68, after splittoning 67). Skipped when both strengths
  are 0 — the only true no-op.
- `history.rs`: "Vignetting" group. `darkroom/mod.rs`: module row with 7 sliders.

**The design point: `is_pixel_local()` returns `false`.** Each pixel's falloff
weight comes from its `(i, j)` relative to the vignette centre, and the dither is
a per-row TEA stream seeded from `j`. The band-parallel path hands each band
`(w, h) = (band_pixels, 1)`, so every band would compute its falloff from the
wrong coordinates and reseed the dither — visible seams. This is only the second
non-pixel-local stage (after Sharpen) and the first for *position* rather than
neighbourhood reads, so having it enabled forces the whole pipeline serial.
The exhaustive no-wildcard match in `is_pixel_local` is what forced the decision.

**Geometry is derived in `apply`, not stored in the stage.** It depends on the
buffer dimensions, which only `apply` knows; caching it would silently go stale
at a different zoom or on export.

**Verified.** `scripts/ci-local.sh` (all four steps); 974 tests (+2). Rendered
the falloff as ASCII and eyeballed it — bright elliptical centre, smooth
symmetric radial falloff, ellipse following the 100×40 aspect (auto-ratio
working). Binary deployed to the running container.

**Notes.** New test `vignette_is_radial_and_band_split_invariant` pins the three
properties that follow from position-dependence: corner darker than centre,
deterministic for a given rectangle, and falloff varying along x (which a
per-band `h = 1` run would collapse).

---

## 2026-08-12 09:05 UTC — m4-110: Lowlight vision live preview stage

**Commit** pending (GitHub + Gitea)

**What.** Wired the lowlight IOP (scotopic / "night vision") into the preview
pipeline — the **18th** live module.
- **`c41-core/src/splines.rs` (new)**: the cubic-spline machinery is lifted
  out of `iop::colorzones` into a shared module. lowlight needed the same
  Catmull-Rom code, and a second copy would mean a fix to the interpolation
  reaching only one caller.
- `iop/lowlight.rs`: `build_transition_lut()`, porting `commit_params` plus the
  **V1** sampler (`CurveDataSample`, via `dt_draw_curve_calc_values`). V1 differs
  from the V2 sampler colorzones uses in two ways that matter: it rounds with a
  truncating `+ 0.5` rather than `round()`, and it does not clamp to
  `[min_y, max_y]` inside the loop. The *spline* is identical — V1's
  `catmull_rom_set` computes exactly the non-periodic tangents of
  `Catmull_Rom_spline::init` — which is why the shared module works for both.
  The two padding anchors `commit_params` wraps around the 6 user nodes are
  reproduced, including darktable's asymmetric choice of which y each takes.
- `pipeline.rs`: `Stage::Lowlight` — Lab-domain, pixel-local.
- `preview.rs`: blueness + 6 transition bands; ENCODE_VERSION 12→13, len
  428→457. Pushed at iop_order.c pos 63 (between colorize 62 and monochrome 64).
- `catalog.rs`: added "Lowlight vision" — **without this the module renders
  inert**, since the panel dispatches on the catalogue label.
- `history.rs` group; `darkroom/mod.rs` module row (blue shift + 6 zone sliders).

**Verified.** `scripts/ci-local.sh` (all four steps); 978 tests (+4). Visually
checked the actual output: neutral greys shift blue (R down, B up — the Purkinje
shift), black stays black, and a saturated red desaturates hard toward luminance
(0.30,0.02,0.02 → 0.159,0.023,0.033), which is right because rods carry no
colour.

**Notes.** Two new LUT tests pin the curve: flat 0.5 at the defaults, and
monotonic for a monotonic ramp — the latter is what catches a padding-anchor or
tangent mistake, which shows up as a dip near index 0.

---

## 2026-08-12 10:20 UTC — Rename: darkroom → C-41

**Commit** pending (GitHub + Gitea, both repos renamed)

**What.** The project is now **C-41** (after the colour-negative development
process). Renamed: the 5 crates (`darkroom-core` → `c41-core`, …, umbrella
`darkroom` → `c41`, binary `darkroom-rs` → `c41-rs`), the Docker dev image
(`c41-rust-dev`), all documentation (README, CLAUDE, CHANGES, PARITY_AUDIT,
PROGRESS), and **both remotes** — `github.com/ilfrick/c-41` and the Gitea
mirror. Local `origin` dual push URLs re-pointed; GitHub auto-redirects the old
URL, so existing clones keep working.

**Deliberately NOT renamed** — each would break something real, so each was
checked rather than assumed:
- **311 `darkroom_*` FFI symbols**, called from 105 C files. Both sides would
  have to change atomically for zero functional gain.
- **`darkroom_ui_prefs`**, a live SQLite table (`persist.rs:63`). Renaming it
  orphans every saved panel width, filter and view mode.
- **`/config/darkroom/`, `DARKROOM_*` env vars.** Verified against the running
  container: `/config/darkroom/library.db` is 7.7 MB and holds the real 2000-image
  catalogue. Renaming strands it.
- **"darkroom view"** — darktable's own term for the edit view, not our name.

The reasoning is recorded in CLAUDE.md so the next session doesn't "finish the
job" and break one of them.

**Verified.** `scripts/ci-local.sh` (all four steps) after the rename; 311 FFI
exports and the prefs table confirmed intact by grep.

---

## 2026-08-12 12:05 UTC — App name fix + stop tracking runtime state

**Commits** `e11615be`, `532326198d` (GitHub + Gitea)

**What.** Two follow-ups from user-reported problems.

1. **The app still called itself "Darkroom"** (`e11615be`). The rename pass
   protected every `"Darkroom"` literal because darktable uses "darkroom" for
   its editing *view*, and renaming that would make the UI lie about the app it
   mirrors. Two of those literals were the app's own branding, not the view
   term: the `ApplicationWindow` title and the lighttable header title, plus
   `APP_ID` (`org.darkroom.Darkroom` → `org.c41.C41`). The view-switcher button
   still reads "Darkroom", correctly.

2. **26 MB of container runtime state was tracked** (`532326198d`). m4-109's
   `git add -A` swept in the bind-mounted `/config`: the user's live catalogue
   (`library.db`, `data.db`), thumbnail cache, KasmVNC session files and TLS
   certs. `library.db` is rewritten on every run, so every unrelated commit
   carried a binary diff of the user's photo database. `git rm -r --cached` plus
   a `/config/` gitignore entry; files stay on disk, container verified healthy.

**Verified.** `scripts/ci-local.sh` (all four steps); screenshotted the running
app — title bar reads C-41, switcher button unchanged.

**Notes — the other half of the report was "I still cannot see new features",
and the features were fine.** The KasmVNC desktop was **1084×348** (KasmVNC
sizes the virtual display to the client's browser window) while the app window
is 663px tall, so roughly half the UI — including the entire module list —
rendered below the visible area. Resizing the display to 1920×1080 showed
everything working: "Modules — 18 of 45 active", live modules sorted first,
"not yet wired" placeholders dimmed, and the Crop/Rotate rows pointing at their
own controls.

**Notes — my verification was wrong, twice over.** I "checked" the module panel
by byte-grepping the release binary for `not yet wired`, got zero matches, and
went hunting for a stale-build problem that did not exist: the optimiser had
split the literal into two 8-byte `movabs` immediates, so it never appears
contiguously. Grepping a stripped release binary proves nothing. A screenshot
settled it in seconds and should have been the first move — the same lesson as
2026-08-11, arrived at from a different direction.

---

## 2026-08-12 13:40 UTC — m4-111: Graduated ND live preview stage

**Commit** pending (GitHub + Gitea)

**What.** Wired the graduatednd IOP into the preview pipeline — the **19th**
live module, and the second position-dependent one.
- `c41-core/iop/graduatednd.rs`: `commit_geometry()`, porting the block at the
  top of graduatednd.c `process()` (`length_base`, `length_inc`, `cosv_hh_inv`,
  `filter_hardness` from the filter radius) plus the colour step from
  `commit_params` — `hsl2rgb(hue, sat, 0.5)`, inverted when density < 0, with
  `color1 = 1 - color`. Note the C negates the rotation before converting to
  radians.
- `pipeline.rs`: `Stage::GraduatedNd`, RGB-domain (`working_space()` = None) and
  **not pixel-local** — the filter strength is a function of the pixel's
  `(x, y)` against a rotated line, so band-splitting would give each band the
  wrong coordinates. Same reason as Vignette; second stage in that class.
- `preview.rs`: 6 params + on/off; ENCODE_VERSION 13→14, len 457→482. Pushed at
  iop_order.c **pos 25** — scene-referred, right after exposure 21 and before
  the channel mix 28.5. Early placement is correct: it models glass in front of
  the lens, so it belongs on linear scene data before any tone or colour work.
  Density 0 is `exp2(0) = 1` everywhere, a true no-op, so it is skipped.
- `history.rs` group; `darkroom/mod.rs` module row (6 sliders). The catalogue
  already had "Graduated density", so no catalog.rs change was needed.

**Verified.** `scripts/ci-local.sh` (all four steps); 976 tests. Rendered the
filter as ASCII at three settings: rotation 0 darkens top-to-bottom, rotation 90
turns the gradient left-to-right, and hardness 90 compresses the ramp into a
sharp edge — all three correct.

**Notes.** darktable also offers an on-canvas line handle for rotation/offset;
the sliders carry the same parameters until that overlay exists. Also answered a
user question: the header's **"Other"** button is empty because it fronts
darktable's map / print / slideshow / tethering views, none of which are ported
(parity 3.4). Of the four, slideshow is the only cheap one — map needs
libosmgpsmap and tethering needs gphoto2.

---

## 2026-08-12 16:30 UTC — parity-3.3: darktable-matched theme

**Commit** pending (GitHub + Gitea)

**What.** The app shipped stock libadwaita dark — blue accents, rounded corners,
GNOME idioms — against darktable's flat, square, entirely grey chrome. New
`c41-ui/src/theme.rs` installs a `CssProvider` carrying darktable's own palette,
lifted verbatim from `data/themes/darktable.css` in this repo:
`bg #262626` (grey_15), panels `#303030` (grey_20), text `#b9b9b9` (grey_75),
selection `#525252` (grey_35). Blue accent overridden to grey, `border-radius: 0`
throughout, flat headers/buttons/sliders/scrollbars.

**The canvas greys are functional, not decorative.** darktable puts the darkroom
canvas at a true middle grey (`grey_50 #777777`) and the lighttable at
`grey_40 #5e5e5e` because the surround changes how you judge the tone and colour
of the image sitting on it — the upstream CSS says so in as many words. Both are
now applied via `c41-darkroom-canvas` / `c41-lighttable-canvas` classes.

**Verified.** `scripts/ci-local.sh` (all four steps); 979 tests (+3). Deployed
and screenshotted, then sampled pixel values against the darktable reference
screenshot: panels within a few levels, grid surround exactly `(94,94,94)`.

**Notes.** First attempt at the canvas grey silently lost — the `GridView` sits
in a `ScrolledWindow` whose generic `.view` rule beat the class on specificity,
so the grid stayed `#303030`. Caught by sampling the pixel rather than trusting
that the CSS "looked right"; fixed with explicit child/viewport selectors.

Three tests guard the theme: the palette is pinned against the upstream values,
the CSS is checked for un-interpolated `{gNN}` placeholders (which GTK drops
silently, taking the whole rule with them), and **every colour literal is
asserted to be an even grey** — the point of the exercise is an achromatic UI
that cannot bias colour judgement, so a stray tint should fail the build.

**Not attempted** (deliberately, and recorded in the module docs): darktable's
"bauhaus" sliders are custom-drawn widgets, not styled GTK ranges, so matching
them needs new widgets rather than CSS; and the panel *layout* (which modules
live where) is parity 2.2-2.6, a separate track.

---

## 2026-08-13 09:15 UTC — parity-3.3b: bauhaus-style sliders

**Commit** pending (GitHub + Gitea)

**What.** New `c41-ui/src/bauhaus.rs`: a custom `DrawingArea` slider matching
darktable's control shape — a flat baseline bar with the filled portion showing
the value, an equilateral triangle indicator, and **the label and value drawn
inside the same rectangle** (left and right) rather than a separate label widget
beside a GTK `Scale`. Geometry follows `src/bauhaus/bauhaus.c`
`_draw_indicator_shape` (sin = 0.866r, cos = 0.5r).

`labeled_slider` now builds one of these, so all ~60 module sliders change at
once. The widget mirrors the slice of `gtk4::Scale`'s API the call sites use
(`value`, `set_value`, `connect_value_changed`), so only three sites needed
touching — and those only because the callback now receives the value directly
instead of the widget.

**Why a widget and not CSS:** GTK's `Scale` always renders trough + handle as
separate nodes and always reserves the handle's width. No stylesheet turns that
into "text and value inside a filled bar".

**Verified.** `scripts/ci-local.sh` (all four steps); 983 tests (+4). Deployed,
expanded the Exposure module and looked at it: EV and Black render as darktable
does, indicator at the right position for each (EV mid-range of -3..3, Black at
the left of 0..0.2).

**Notes.** First deploy had descenders touching the baseline bar; row height
26→30 and the text baseline moved up by 1.5×PAD. Four tests cover the pure value
logic (no display needed): step snapping, clamping, the range→unit mapping, and
two edge cases worth pinning — snapping must not *overshoot* the bounds (with
max 1.0 and step 0.3, `round(1.0/0.3)*0.3 = 1.2`), and a degenerate `min == max`
range must not divide by zero and put NaN in the draw path.

**Not attempted:** bauhaus's popup editor, gradient stops, soft/hard bounds and
the right-hand quad button — a much larger surface, none needed by the current
module rows.

---

## 2026-08-13 10:40 UTC — parity-3.2: collapsible panel sections

**Commit** pending (GitHub + Gitea)

**What.** The left panel's sections (Collections, Colours, Tags) were flat
labels with their content permanently expanded. darktable's panels are stacks of
expanders — a disclosure triangle and a title you click to fold the section
away — which is what lets it show many sections in one panel.

New `collapsible_section()` in `panels/mod.rs` wraps an existing header label
plus its content widgets in a title row with a `pan-down`/`pan-end` triangle and
a click gesture. It takes the *already-built* header so callers keep their
references — the Tags section toggles its header's visibility from three other
places, and this only changes where those widgets are parented.

**Verified.** `scripts/ci-local.sh` (all four steps). Deployed, screenshotted,
then clicked the Collections title: the triangle flipped, the 25-row film-roll
list folded away, and **Colours became visible without scrolling** — which is
the concrete payoff, since the colour-label filter was previously below the fold
on a 1040px-tall window.

**Notes.** Collapse state is per-session. Persisting it would want a prefs key
per section, alongside the existing panel width/collapsed keys — a follow-up.
The darkroom module panel already used `adw::ExpanderRow`, so it needed nothing.

---

## 2026-08-14 08:20 UTC — parity-3.2b: persist section fold state

**Commit** pending (GitHub + Gitea)

**What.** `collapsible_section()` now takes a `db_path` and pref key and stores
the fold state in `c41_ui_prefs`, reusing the `shown`/`hidden` token encoding the
side-panel collapse keys already use. Three keys: `left_section_collections`,
`left_section_colours`, `left_section_tags`. An empty `db_path` skips persistence
(for panels built before the DB opens).

**Verified.** `scripts/ci-local.sh` (all four steps); 985 tests (+2). Collapsed
Collections, restarted the app, and confirmed it came back collapsed with Colours
still expanded.

**Notes.** The stored value is `!expanded` and restore is `!collapsed`; a test
pins that double negation, because inverting it would silently reopen every
section the user had closed — the kind of bug that looks like "it just doesn't
remember" rather than an error.

---

## 2026-08-14 09:30 UTC — parity-2.6a: Import and Export panel sections

**Commit** pending (GitHub + Gitea)

**What.** darktable's left panel opens with an import module and its right panel
ends with an export module. Ours had both only as header buttons, so the panels
didn't match and the actions were less discoverable.

- **Import** section at the top of the left panel (collapsible, persisted under
  `left_section_import`), with an "Add images…" button.
- **Export** section at the foot of the right panel, with "Export selected…".

Both call `set_action_name` on the **existing** `win.import` /
`win.export-selected` actions rather than duplicating the dialog code — one
implementation, one behaviour, and the Ctrl+I / Ctrl+E accelerators keep working
unchanged. GTK resolves action names through the widget hierarchy at activation
time, so no window reference is needed in the panel constructors.

**Verified.** `scripts/ci-local.sh` (all four steps). Deployed and screenshotted
both panels: Import sits above Collections on the left, Export below Tags on the
right, and Collections was still remembered as collapsed from the previous run.

**Notes.** Export is not collapsible — `MetadataPanel::new()` takes no `db_path`,
so there is nowhere to persist a fold state, and a single button does not earn
one. Still missing from 2.6: darktable's **collection filters** expander, which
is a real feature (compound filter rules) rather than a re-presentation of
something we already have.

---

## 2026-08-14 11:15 UTC — m4-112: Contrast/brightness/saturation (colisa)

**Commit** pending (GitHub + Gitea)

**What.** Wired the colisa IOP into the preview pipeline — the **20th** live
module, and the first to need `dt_iop_estimate_exp`.
- `c41-core/iop/colisa.rs`: ported `estimate_exp` (src/develop/imageop_math.h:98)
  — fits `y = y0*(x/x0)^g` with the last sample pinned, averaging `g` over the
  rest and skipping samples where either log would be undefined. Plus
  `commit_params()`, porting colisa.c's: rescale contrast/saturation from -1..1
  to 0..2 and brightness to -2..2, build both 65536-entry LUTs (the two builders
  were already ported), and fit the unbounded-extrapolation coefficients from
  four samples at x = 0.7..1.0.
- `pipeline.rs`: `Stage::Colisa`, Lab-domain and pixel-local. Holds the three
  sliders rather than the derived LUTs, so the stage stays `PartialEq` and the
  384 KB of tables are not carried around per stage.
- `preview.rs`: v14→v15, len 482→495. Pushed at iop_order.c **pos 47** —
  display-referred, just before tonecurve 48 and levels 49. Upstream's own
  comment on it is "edit contrast while damaging colour", which is why it lives
  in that cluster.
- `catalog.rs` entry under Tone; history group; module row (3 sliders).

**Verified.** `scripts/ci-local.sh` (all four steps); 988 tests (+3). Checked
the actual numbers rather than trusting green: contrast +0.5 pushes darks down
(0.10→0.04) and lights up (0.66→0.90) — an S-curve about the midpoint;
brightness +0.5 lifts everything, most in the shadows; saturation -1 leaves the
grey ramp untouched and collapses a saturated red to exact neutral
(R=G=B=0.216), which is the correct b&w endpoint.

**Notes.** `estimate_exp` matters more than it looks: it extrapolates both tone
curves above 1.0, so without it scene-linear highlights past the LUT's domain
would clamp. A test recovers a known power law (y = 3x²) to confirm the fit, and
another pins the fallback — non-positive ratios make the logs undefined, and the
C returns g = 1 rather than NaN, which would otherwise propagate into every
extrapolated highlight.

---

## 2026-08-18 15:38 UTC — Styles panel UI (parity 2.4) + GTK4 CSS corrections

**Commit** pending (GitHub + Gitea)

**What.** The UI half of styles. The data layer shipped in `7a6a9744c1` and —
a process miss worth recording — landed with **no PROGRESS entry and no
PARITY_AUDIT update**; this entry covers both halves and 2.4 is marked here.

- `panels/mod.rs`: a **Styles** section in the right panel — list, "Save
  current…", "Apply", delete. `wire_styles()` takes a *getter* for the selected
  image's path + params, because the panel is built once and the selection
  changes constantly.
- `lib.rs`: wires it to `lt_selection` and the existing `make_toast`.

**Resumed into a tree that did not compile.** `theme.rs` had a comment reading
`"reported min height -4"` *inside* the one `format!` literal that generates the
whole stylesheet — the double quote closed the string:
`error: expected ',', found min`. All four CI steps were failing. The sheet is
~40% comments, so this is a standing trap; the fix uses backticks and says why.

**Two blockers from senior review, both real:**

- **Apply could never fire after choosing a target image.** `update()` rebuilt
  the styles list on every `selection-changed`, and a wholesale rebuild drops
  the `ListBox` selection. So "pick style → pick image → Apply" silently
  no-opped; only style-last worked, the opposite of what the button says. The
  rebuild was pointless anyway — `c41_styles` is mutated only by the save and
  delete handlers, and both already refresh. Removed it, and made
  `refresh_styles_list` selection-preserving as belt-and-braces.
- **The success path destroyed the user's collection.** `on_applied` called
  `lighttable_load_from_db`, which is *load the whole library, no folder*: it
  discarded any active folder/tag/search collection, overwrote `RELOAD_CURRENT`
  so the next sort re-loaded the wrong view, and reset the selection to image 0.
  It could not even achieve its stated purpose — thumbnails decode from the
  file's own bytes via `PixbufLoader` and never consult `PreviewParams`. Replaced
  with a toast.

Also from review: every failure path now reports through `notify` (`save_style`
documents "the caller surfaces that"; the caller didn't); Apply reads the target
path from the getter rather than `ctx`, removing a second source of truth that
could write onto a stale path once the selection emptied; `wire_styles` is
guarded against a double call (`connect_clicked` appends, and the panel is
`Clone`); confirm-on-overwrite and confirm-on-delete, since styles have no
history stack behind them; the list sits in a height-capped `ScrolledWindow`;
double-click applies.

**The `reported min height -4` warnings — the previous session blamed the wrong
rule.** The diff claimed deleting the dead `GtkScale` rules and the
`switch > slider` constraints was what silenced them. It wasn't: the process
running that binary still emitted 13 of them. The actual source was
`scrollbar slider { min-width: 8px; min-height: 8px }` — GTK subtracts the
node's border and padding from a declared `min-*`, and 8px minus Adwaita's 12px
is exactly the −4. Colour-only there too: **13 warnings → 0**, still 0 after
clicking around. The comments now state what was measured; the earlier ones
asserted a fix that had not happened.

`:root` and `:has()` were separately confirmed dead on this runtime (Ubuntu
Noble → GTK 4.14 / libadwaita 1.5; `:root` and custom properties arrived in
4.16), so the accent override never applied and libadwaita's blue survived.
Replaced with `@define-color`.

**Verified.** `scripts/ci-local.sh` — exit code 0, all four steps, 999 tests.
Deployed into the running `c-41` container and driven with xdotool: the Styles
section renders, and clicking **Apply with nothing selected raises the toast
"Select a style first"** — the notify wiring works end to end where before that
click was a silent no-op.

**Not verified, deliberately stated.** The save/overwrite/delete **dialogs were
not confirmed visually**: `adw::AlertDialog` constructs (new layout warnings
appear on click) but does not appear in the KasmVNC framebuffer — the known
black-dialog issue in this container, not something this change introduced. The
selection-preserving rebuild also has **no unit test**: the crate's suite is
display-free (no `gtk4::init()` anywhere), so a `ListBox` cannot be built in a
test. The primary fix — deleting the rebuild — is what actually restores the
workflow.

**Notes.** The denylist test that pins `:root`/`:has()` now strips `/* … */`
first: it was scanning its own comments, which forced every warning to be
phrased around naming its subject. It is a proxy, not a proof — the real check
is `CssProvider::load_from_string` with `connect_parsing_error`, which needs GTK
initialised. Its expiry is recorded in the test: past GTK 4.16 these selectors
become valid and libadwaita ≥1.6 prefers them.

---

## 2026-08-19 12:33 UTC — Metadata editor (parity 2.3)

**Commit** pending (GitHub + Gitea)

**What.** darktable's *metadata editor*: the five writable Dublin Core fields
(title, description, creator, publisher, rights) as entries in the right panel,
where before this the panel showed EXIF and nothing was editable.

Unlike every other table `persist.rs` touches, storage here is **darktable's own
`main.meta_data`**, not a c41-ui-private table. Metadata is not our format — it
is XMP/Dublin Core text the C app reads, writes and exports to sidecars, so
keeping it anywhere else would produce metadata the rest of the application
cannot see.

**Two blockers from senior review, the first of which was my own bad research.**

- **I documented the schema wrongly, and built to the wrong shape.** The comment
  claimed `main.meta_data` has no uniqueness constraint. It has
  `UNIQUE(id, key, value)` plus an FK to `images`. The error came from reading
  `database.c:264` — a legacy migration block — plus the `MIN(rowid)` dedupe at
  `:1778`, and stopping there; the live schema is `database.c:3547`, and
  `LAST_FULL_DATABASE_VERSION_LIBRARY` is 55, so *every* current catalogue has it.
  Confirmed afterwards against the running container's own `library.db`:
  `CREATE UNIQUE INDEX metadata_index ON meta_data (id, key, value)`.
  The conclusion (no upsert; delete-then-insert) survives, but for a different
  reason than stated: the index is on `(id, key, value)`, so `ON CONFLICT(id,key)`
  has no index to target.
  This was not just a comment. The ad-hoc `CREATE TABLE IF NOT EXISTS` in
  `persist.rs` was the production path on c41-created catalogues and built a
  **narrower** table than darktable's — no unique index, no FK, no secondary
  indexes — i.e. a catalogue the C app opens but whose constraints differ. The
  table is now created by `c41_db::schema::ensure_base_schema` with upstream's
  exact shape, and `persist.rs` issues no DDL at all.
- **No dirty check, so a stale focus-leave could write one image's text onto
  another.** Entries snapshot the target image on focus-*enter*, which was the
  right instinct but leaned on GTK4 delivering focus-leave before
  selection-changed — not contractual. Upstream doesn't rely on it either: it
  stashes `text_orig` and only writes a field that actually changed
  (`src/libs/metadata.c:360`), and flushes in `gui_update` *before* repainting
  (`:239`/`:269`). Both adopted. The old `has_focus()` guard was worse than
  useless: if the selection changed while an entry kept focus, the entry showed
  image A's text under image B permanently, and the next focus-leave wrote A onto
  B. Repaint is now unconditional, because the flush already persisted the edit.

**Also: this duplicated code that already existed.** `crates/c41-db/src/metadata.rs`
has been there since Phase 2-db-3 with `metadata_get_all` / `metadata_set_value` /
`metadata_delete_key`, `c41-ui` already depends on the crate, and its module doc
states the schema *correctly* — including the unique index the new comment denied.
Reads and writes now delegate to it, so the UI and the FFI path cannot diverge;
`persist.rs` keeps only path→imgid resolution and the UI's field set.

Smaller review items: Escape reverts (upstream cancels; a bare `GtkEntry` ignores
it and would commit on the next focus-leave); a `close-request` flush, since GTK4
does not promise focus-leave during teardown; trimming matches upstream's
spaces-only `_cleanup_metadata_value` rather than Rust's `.trim()`; the imgid
lookup moved inside the transaction.

**Verified.** `scripts/ci-local.sh` — exit 0, all four steps, 1008 tests. Then
driven in the running container with xdotool, re-run after the rework:
- type a title + Tab → `(11371, 2, 'Rework check')` in `main.meta_data`;
- select another image → fields blank; select back → value reloads;
- **Escape** after editing → value unchanged in the db;
- **type, then click a different thumbnail while the entry still has focus** →
  the write lands on 11371, *not* on the newly selected image. That is the exact
  cross-image write B2 described, shown not to happen.
- clearing a field deletes the row rather than storing `''`.
Test data was removed afterwards; `main.meta_data` is empty again.

**Notes.** Tests now build their fixture from `ensure_base_schema`, so they run
against the real constraints — the previous fixture created its own narrower
table, which meant no test ever exercised what production writes under. Added
pins for `MetaField::ALL` display order and the unique-index shape. The
duplicate-row test no longer asserts rowid ordering (the delegated read orders by
`key`, so ties are the planner's choice); it pins what is actually guaranteed —
a duplicate never reads back blank, and the next write collapses it to one row.

**Known gap, unchanged: the XMP sidecar is not written.** Upstream follows
`dt_metadata_set_list` with `dt_image_synch_xmps` (`src/libs/metadata.c:383`,
`:393`). We write the catalogue only, so a value set here does not travel with
the file until darktable rewrites the sidecar. Nothing is lost or corrupted —
`dt_exif_xmp_read` reads sidecars back *into* this table — it is unexported, not
wrong. An XMP writer is its own increment.

---

## 2026-08-21 09:45 UTC — parity-2.1: Basic adjustments (basicadj) live preview module

Ports darktable's `basicadj.c` IOP — black point, exposure, highlight compression,
brightness, contrast, saturation and vibrance in a single pass over linear RGB.
This is the 16th live darkroom module (catalog.rs Tone group already listed it;
this wires the full pipeline).

**What changed (6 files, 628 insertions):**

- **`c41-core/src/iop/basicadj.rs`** (new, 385 lines): `darkroom_basicadj_process`
  FFI kernel (unsafe extern "C", `# Safety` doc on pointer/array contract) + safe
  `BasicadjData::process` wrapper + `commit_params` porting basicadj.c:1401-1422
  (exposure2white, gamma-from-brightness, hlcompr shoulder, contrast/middle_grey
  fallback, mutual exclusion of plain_contrast vs preserve_colors). Thread-local
  `cached_luts` memo (keyed on raw `to_bits()`) avoids rebuilding two 65536-entry
  LUTs per band in the rayon-parallel path. 10 unit tests (added one for the
  LUMINANCE-mode luminance fix below).
- **`c41-core/src/pipeline.rs`**: `Stage::Basicadj` variant — `is_pixel_local()` →
  true, `working_space()` → None, `apply()` derives luma from `ColorSpace`
  (Rec2020 = [0.2627, 0.6780, 0.0593], LinearSrgb = [0.2126, 0.7152, 0.0722])
  and calls `commit_params` + `process`. Added to the
  `stage_pixel_locality_is_correctly_classified` exhaustiveness test.
- **`c41-ui/src/preview.rs`**: PreviewParams +1 bool +10 f32, ENCODE_VERSION
  11→12. `is_identity` gate excludes `middle_grey`, `preserve_colors`, and
  `hlcomprthresh`. `to_pipeline()` at iop_order pos 40 (between sharpen 39 and
  colorcorrection 55). Updated
  `to_pipeline_orders_stages_canonically` test with "basicadj" in the expected
  order.
- **`c41-ui/src/history.rs`**: `edit_label_for_change` includes basicadj change
  detection → "Basic adjustments". `hlcomprthresh` excluded from the change
  predicate (no UI slider).
- **`c41-ui/src/darkroom/mod.rs`**: `basicadj_module_row` with 8 sliders
  (black_point, exposure, hlcompr, contrast, middle_grey, brightness, saturation,
  vibrance). Added to `LIVE_MODULE_LABELS`.
- **`c41-ui/src/catalog.rs`**: already had "Basic adjustments" in Tone group (no
  change needed this pass).

**Deliberate omissions (documented in code):**
- `hlcomprthresh` is NOT a user slider — verified at basicadj.c gui_init
  (lines 596-663): only `hlcompr` gets a slider. `hlcomprthresh` is set internally
  by auto-exposure. The field persists in PreviewParams for encode/decode
  backward-compat, defaults to 0.0.
- `clip` is absent — the migrated kernel doesn't implement it; a control that
  does nothing is worse than no slider.
- `preserve_colors` is an enum, not a slider — left as a dropdown for a future
  increment.

**Senior review (inline):**
The `fricktrade-architect` agent was specified per CLAUDE.md (model: opus, Opus
4.8), but every invocation failed with OpenRouter 402 "request requires more
credits" — an infrastructure issue, not a code issue. The review was conducted
inline using the same UNDERSTAND→DIAGNOSE→PRIORITISE→PROPOSE→VALIDATE framework,
reading basicadj.c, basicadj.cl, rgb_norms.h, color.rs, and all 6 Rust files.

**Findings:**
1. ⚠️ MODERATE — `rgb_norm` mode 1 (LUMINANCE) used ProPhoto coefficients instead
   of working-space luminance in the preserve_colors path. Fixed by computing
   luminance directly from the `luma` array (same coefficients already used for
   hlcompr), matching the C `dt_rgb_norm` behavior. Test added.
2. MINOR — `fast_length` (C OpenCL approx) vs exact `sqrt` (Rust) in the
   saturation/vibrance delta. Negligible; left as-is (arguably more correct).

**Verified.** `scripts/ci-local.sh` — exit 0, all four steps:
cargo check, cargo clippy, `cargo test --workspace --release` (1017+ tests),
`cargo build --release -p c41 --bin c41-rs`. Zero basicadj.rs clippy warnings.

---

## 2026-08-21 — m4-112: Lowpass (local contrast enhancement) live preview module

**Commit** pending (GitHub + Gitea)

**What.** Wired the lowpass IOP into the Rust preview pipeline — the 20th live
module. Lowpass blurs a copy of the image and applies contrast/brightness LUTs +
a/b saturation to the blurred pixels.

- `c41-core/src/iop/lowpass.rs`: Already ported (FFI kernels ported + tested in
  Phase 2z+40). `commit_params()` builds 65536-entry contrast + brightness LUTs
  using `darkroom_lowpass_build_contrast_lut`/`darkroom_lowpass_build_brightness_lut`,
  fits extrapolation coefficients with `colisa::estimate_exp`, and hardcodes
  `unbound=true` (the C default; darktable does not expose it in the GUI).
  `process_pixels()` applies the LUT pair + saturation clamp to the blurred Lab
  buffer, matching `darkroom_lowpass_process`.
- `pipeline.rs`: `Stage::Lowpass { radius, contrast, brightness, saturation, scale,
  space }` variant. `is_pixel_local() = false` (Gaussian blur reads neighbours →
  serial whole-buffer path, same as Sharpen). Apply arm: RGB→Lab → Gaussian blur
  → `lowpass::process_pixels` → Lab→RGB. Added `lowpass` to the iop import list.
  Added assertion to `stage_pixel_locality_is_correctly_classified` test.
- `preview.rs`: Added 5 new PreviewParams fields (`lowpass_on: bool` + 4 f32s).
  ENCODE_VERSION 12→13, ENCODED_LEN 536→553 (1 ver + 20 bool + 133×4). Encode
  writes bools at bytes[0..20] then 133 f32s. Decode reads bools at
  bytes[1..21] then f32s from bytes[21..]. Updated encode/decode/
  default/bypassed/is_identity/to_pipeline and the
  `params_encode_decode_roundtrips` test literal. Lowpass placed in
  `to_pipeline` between basicadj (pos 40) and colorcorrection (pos 55), matching
  darktable iop_order.c pos 54.0.
- `history.rs`: Added "Lowpass" to `describe_change()`, updated exhaustive
  destructure test, bumped HISTORY_ENCODE_VERSION 3→4, pinned length 536→553.
- `darkroom/mod.rs`: Added "Lowpass" to LIVE_MODULE_LABELS, dispatch match arm,
  and `lowpass_module_row` with 4 sliders (radius 0.1..500, contrast/
  brightness/saturation -3..3 — matching darktable C params defaults).

**Verified.** `scripts/ci-local.sh` — exit 0, all four steps:
cargo check, cargo clippy, `cargo test --workspace --release`, and
`cargo build --release -p c41 --bin c41-rs`.

**Notes.**

- First CI run failed: `lowpass` was not in the iop import list in pipeline.rs
  (the `Stage::Lowpass` apply arm references `lowpass::commit_params`). Fixed by
  adding `lowpass` to the `use crate::iop::{...}` line.
- Second CI run failed: encode/decode offset mismatch. The decode read f32s from
  `bytes[20..]` (the old 19-bool boundary) instead of `bytes[21..]` (20-bool
  boundary). This caused `params_encode_decode_roundtrips` and all
  history/persist roundtrip tests to fail with garbage values. Fixed by shifting
  the f32 slice start to `bytes[21..]` and updating the comment.
- The senior-review (fricktrade-architect, Opus 4.8) could not be launched — the
  OpenRouter endpoint returned HTTP 402 (insufficient credits). The review was
  performed inline instead: verified LUT formulas match `darkroom_lowpass_build_*`
  (contrast ≤1.0 linear, >1.0 sigmoid with boost=5.0; brightness gamma),
  process_pixels order (contrast LUT → brightness LUT → saturation clamp → alpha
  from input), unbound=true default, iop_order position 54.0, and test coverage
  patterns consistent with basicadj/colisa.

## m4-113 — Shadows/Highlights (shadhi) senior-review fix-ups (2026-08-22)

**What changed** (review findings from fricktrade-architect, Opus 4.8, applied
after CI went green on the initial shadhi wiring):

- `preview.rs`: Rewrote `PreviewParams::decode` to be **version-aware** via a
  `LAYOUTS` table (`[(version, n_bools, n_f32s)]`). v13 blobs (20 bools /
  133 f32s) decode successfully with shadhi fields defaulted, so bumping
  `ENCODE_VERSION` 13→14 does NOT silently delete saved styles. The old
  length-gate (`bytes.len() != ENCODED_LEN`) would have rejected v13 blobs
  entirely → silent style loss.
- `preview.rs`: Relocated the ENCODE_VERSION history doc-block from
  `LEVELS_MIN_RANGE` (where it was accidentally attached) onto
  `const ENCODE_VERSION`, and corrected the v13 history note (lowpass, not
  lowlight). Added a `decode_v13_blob_defaults_shadhi_fields` roundtrip test
  that builds a raw v13 blob and asserts shadhi fields fall back to defaults.
- `pipeline.rs`: Replaced `f32::signum(sh)` / `f32::signum(-hg)` (returns
  0.0 for ±0.0) with a `CSignum` trait providing C-compatible `signum_c()`
  (returns 1.0 for ±0.0), matching darktable's `#define sign(x)` macro so the
  neutral-slider case is identical.
- `pipeline.rs`: Toned down the "bit-identical output to darktable" claim in
  the shadhi doc — the shadow/highlight *math* is identical, but the Gaussian
  blur (not bilateral) means the base layer differs.
- `pipeline.rs`: Added `// SAFETY:` comment on the `darkroom_shadhi_process`
  FFI call documenting buffer lengths, aliasing, and parameter provenance.
- `pipeline.rs`: Added `shadhi_*` to the `is_pixel_local` exhaustive-match test
  (asserts NOT pixel-local, same rationale as lowpass). Added two numeric
  tests: `shadhi_lifts_shadows_and_recovers_highlights` (gradient →
  shadows lift, highlights recover) and `shadhi_neutral_on_flat_is_identity`
  (neutral sliders on flat field → unchanged).
- `PARITY_AUDIT.md`: Reconciled the live-module count from 17 → **21** and
  added the 4 previously-omitted modules (vignette, lowlight, graduated ND,
  colour brightness saturation / colisa) to the 2.1 list.

**Verified.** `scripts/ci-local.sh` re-run after all senior-review fixes (see
below) — exit 0, all four steps: cargo check, cargo clippy,
`cargo test --workspace --release`, and `cargo build --release -p c41 --bin c41-rs`.

> **Correction (2026-08-22):** The original PROGRESS entry claimed CI was green
> immediately after the initial shadhi wiring; the senior review found CI was
> actually RED at that point (test fixture panic: `radius: 100.0` on a 16×16
> image collapsed the Gaussian to a global mean). That claim was recorded
> prematurely and is corrected here.

**Additional fix-ups applied after senior review (P0–P3):**
- `preview.rs`: Reconciled the local `LAYOUTS` const inside `decode()` with the
  module-level `PARAMS_LAYOUTS` (DRY — was duplicated). Updated
  `layouts_covers_current_version` test to assert against `PARAMS_LAYOUTS`
  directly instead of a mirrored copy.
- `history.rs`: Fixed `encoded_len_for_version` call site — it is a `pub(crate)`
  free function, not an associated function of `PreviewParams`. Changed
  `PreviewParams::encoded_len_for_version(...)` →
  `crate::preview::encoded_len_for_version(...)`.
- `preview.rs`: Rewrote the stale `ENCODE_VERSION` doc comment — it claimed
  "old blobs decode to `None` → defaults", which contradicts the version-aware
  decode that accepts any version in `PARAMS_LAYOUTS`.
- `shadhi.rs`: Added cross-reference comment on `sign()` documenting its
  relationship to `CSignum::signum_c()` in pipeline.rs (must agree at ±0.0).
- `shadhi.rs`: Added latent-upstream-bug note on the shadows overlay loop —
  darktable's `shadhi.c` tests `UNBOUND_HIGHLIGHTS_L` (a highlight flag) for the
  shadow L channel clamp; our port correctly uses `UNBOUND_SHADOWS_L`, and the
  discrepancy is documented so a future side-by-side doesn't "fix" us to match
  the bug.

**Notes.**

- The v13 backward-compat fix is the critical one: without it, every existing
  saved style (written under ENCODE_VERSION 13 before shadhi landed) would
  decode to `None` and be replaced by defaults — effectively wiping all edits
  for every existing user on first load after this version bump.

---

## 2026-08-22 07:42 UTC — m4-113: Full CI verification + dual-push

**What.** Re-ran `scripts/ci-local.sh` after all senior-review fix-ups (P0–P3)
to verify the complete codebase compiles and all tests pass under `--release`.

**Verified.** Exit code 0 on all four steps:
1. `cargo check --workspace` — OK (warnings only: pre-existing unused_parens in
   `color.rs`, unused_import in `film.rs`, deprecated `clone!` macros in c41-ui)
2. `cargo clippy --workspace` — OK (same warnings, no errors)
3. `cargo test --workspace --release` — OK (692 core + 92 db + 239 ui = **1023
   tests passed, 0 failed**; includes 3 new shadhi tests)
4. `cargo build --release -p c41 --bin c41-rs` — OK (links successfully)

**Push.** Commit `d7613a0ee9` pushed to both remotes:
- GitHub `ilfrick/c-41` ✓
- Gitea `housefz.com` ✓

---

## 2026-08-23 07:45 UTC — m4-114: Primaries (RGB primaries adjustment) live preview module

**Commit** pending (GitHub + Gitea)

**What.** Wired the primaries IOP (RGB primaries rotation/scaling) into the
preview pipeline — the **21st** live module.

- `c41-core/iop/primaries.rs`: fully ported from
  `src/iop/primaries.c` + `src/common/custom_primaries.c`. Key components:
  - `rotate_and_scale_primary`, `find_distance_to_edge`, `intersect_line_segments`
    — faithful ports of the gamut-hull projection geometry.
  - `mat3_inv` (port of `mat3SSEinv`) — all 9 cofactors verified term-for-term.
    Now returns `Option` (was: identity on singular); divergence from C is
    deliberate: C's caller writes nothing to a zero-initialised matrix (→ black
    frame, loud); the Rust identity fallback was a plausible-but-wrong silent mode.
  - `make_transposed_matrices_from_primaries_and_whitepoint` (Lindbloom) +
    `colormatrix_mul` (`dt_colormatrix_mul`, verified `M_dst = M_m2·M_m1`).
  - `compute_matrix` — public entry: builds custom RGB→XYZ, inverts working
    RGB→XYZ for `matrix_out`, multiplies, pads to 4×4. Added two fail-safes:
    (1) triangle-area precondition (1e-9 threshold; degenerate configs return
    identity), (2) output reasonableness backstop (non-finite or |coeff| > 1e3).
  - `process_pixels` — safe bounds-checked wrapper around the raw `darkroom_primaries_process`
    FFI shim; release-mode aliasing guard via `debug_assert_ne!(input.as_ptr(), output.as_ptr())`.
  - 7 unit tests including `degenerate_hue_returns_identity` and a full
    3⁸-parameter sweep over the proposed slider ranges asserting all coefficients
    stay bounded (< 10.0).

- `pipeline.rs`: `Stage::Primaries { matrix: [f32; 16] }` after `GraduatedNd`.
  Pixel-local (true) → stays on the rayon band-parallel path. Added explicit
  `working_space() => None` arm (linear RGB, no Lab agreement needed). `apply`
  now calls the safe `process_pixels` wrapper.

- `preview.rs`: 9 new fields (on/off + 8 params). ENCODE_VERSION 14→15,
  ENCODED_LEN 582→615, `PARAMS_LAYOUTS += (15, 22, 148)`. Extracted
  `primaries_is_neutral()` as the single source of truth for `is_identity()`,
  `to_pipeline()`, and tests (was 4 duplicated copies).

- `catalog.rs`: added "Primaries" to the Color group.
- `darkroom/mod.rs`: `primaries_module_row` with 8 GTK4 sliders using darktable's
  **soft** ranges (hue ±20°, purity 0.5..1.5, tint purity 0..0.2; tint hue keeps
  full ±180° since it only moves the white point and is safe).
- `history.rs`: primaries field comparison in `describe_change()`; pinned length
  582→615.

**Verified.** `scripts/ci-local.sh` — all four steps green:
1. `cargo check --workspace` — OK (pre-existing warnings only)
2. `cargo clippy --workspace` — OK (pre-existing warnings: unused_parens
   in color.rs, unused_import in film.rs, deprecated `clone!` macros in c41-ui)
3. `cargo test --workspace --release` — OK (695 core + 92 db + 256 ui = **1043
   tests passed, 0 failed**)
4. `cargo build --release -p c41 --bin c41-rs` — OK

**Senior review.** Spawned `fricktrade-architect` (Opus 4.8) with the full diff.
Two HIGH findings blocked commit; both fixed:
- H1: Slider ranges used darktable's hard range (±180°, purity 0.01..5.0) instead
  of the soft range (±20°, 0.5..1.5). Past ~112° hue the primaries go collinear
  (triangle area ~2.6e-18), giving singular matrices with coefficients ~1e16.
  Fixed by adopting the soft ranges as the slider bounds.
- H2: `mat3_inv` returned identity on near-singular (silent-wrong-answer); the
  absolute 1e-7 epsilon was straddled by f32 noise. Fixed by returning `Option`
  and adding a triangle-area precondition (1e-9) + output magnitude backstop (1e3)
  in `compute_matrix`.

MEDIUM/LOW fixes also applied:
- M1: Extracted `primaries_is_neutral()` to eliminate 4 duplicated neutral-check copies.
- M3: Added explicit `Stage::Primaries => None` arm to `working_space()`.
- L1: Added safe `process_pixels` wrapper with aliasing debug_assert; `apply` no longer uses `unsafe`.
- L2: Fixed doc comment ("gradnd 25" → "graduatednd 28.0").
- L4: Added Primaries to the `is_pixel_local` test vector.

**Notes.** PARITY_AUDIT.md updated with the ~10% working-profile divergence
(C41 uses nominal xy from colorspaces.c; darktable reads ICC colorant tags at
runtime which differ slightly after D50 adaptation). C41's choice is self-
consistent (identity at defaults is exact to 6e-16) and is kept deliberately.

---

## 2026-08-23 11:00 UTC — m4-115: Negadoctor (film negative inversion) live preview module

**Commit** `ef75389094` (GitHub + Gitea)

**What.** Wired the fully-ported `c41-core/iop/negadoctor.rs` (Cineon-style log-density
film negative inversion) into the Rust preview pipeline as a live `Stage::Negadoctor`.
21st live darkroom module (parity 2.1).
- `pipeline.rs`: `Stage::Negadoctor` variant with dmin/wb_high/offset arrays +
  black/gamma/soft_clip/soft_clip_comp/exposure scalars. Pixel-local (no neighbour
  reads → band-parallel path stays available).
- `preview.rs`: `PreviewParams` +1 bool (negadoctor_on) +1 f32 (negadoctor_film_stock)
  +14 f32 (Dmin R/G/B, WB_high R/G/B, WB_low R/G/B, D_max, offset, black, gamma,
  soft_clip, exposure) → ENCODE_VERSION 14→16, len 582→680. `to_pipeline` replicates
  darktable `commit_params` (negadoctor.c:239-267): `wb_high = wb_high/D_max` premultiply,
  `offset = wb_high_original * offset * wb_low`, B&W Dmin mono-collapse,
  `black = -exposure * (1 + black)` FMA trick, `soft_clip_comp = 1 - soft_clip`.
- `preview.rs` tests: `negadoctor_commit_params_matches_darktable` parity test
  verifying every derived field against the C arithmetic; `decode_v15_blob_defaults_`
  `negadoctor_fields` (backward compat with pre-negadoctor styles); encode/decode
  roundtrip updated to 16 fields.
- `darkroom/mod.rs`: `negadoctor_module_row` 15-slider UI.
- `history.rs`: `describe_change` "Negadoctor" label.
- `catalog.rs`: "Negadoctor" added to the "Base" group (after "Invert").

**Senior review (fricktrade-architect, Opus 4.8).** 8 findings + 1 process gap:
- **HIGH-1.** Exposure slider used linear multiplier (0.5–2.0) instead of darktable's
  EV scale (−1..+1). Fixed: slider in EV, param = 2^EV conversion at the widget boundary
  (darktable gui_init:925-929, gui_changed:964, gui_update:988).
- **HIGH-2.** `commit_params` arithmetic was correct but untested. Fixed: added
  `negadoctor_commit_params_matches_darktable` parity test pinning every derived field.
- **MEDIUM-1.** Comments mislabelled negadoctor as "scene-referred" (it simulates print
  on paper — gamma/black/soft-clip — which is display-referred). Also reviewed iop_order
  position (28.5, verified correct). Fixed: "Scene-referred" → "display-referred" in
  doc comments.
- **MEDIUM-2.** B&W mode didn't hide Dmin G/B sliders or mirror Dmin R. Fixed: Dmin G/B
  rows hidden when film_stock==B&W (toggle_stock_controls:388-410), Dmin R callback
  mirrors to G/B (gui_changed:953-957), film-stock slider toggles visibility.
- **MEDIUM-3.** `HISTORY_ENCODE_VERSION` bumped 5→6 unnecessarily — the container format
  is unchanged (PreviewParams carries its own per-entry version byte). Fixed: reverted
  to 5, corrected comment.
- **LOW-1.** Black slider step was 0.0001 (too fine). Fixed: 0.0001 → 0.001.
- **LOW-2.** No comment noting channel-3 (alpha) is inert in the process loop.
  Fixed: doc comment on `Stage::Negadoctor`.
- **LOW-3.** No defensive floor on `D_max` division (division by zero if a loaded
  style has D_max=0). Fixed: `.max(f32::MIN_POSITIVE)`.
- **Process gap.** Review found the negadoctor parity test was missing from the initial
  commit — addressed by HIGH-2 (test added before other fixes, per review's recommended
  order).

**Senior review (second pass, complete diff).** One residual finding:
- **MEDIUM.** The struct field doc comment on `PreviewParams::negadoctor_on` (preview.rs:270)
  still said "scene-referred" — the MEDIUM-1 fix in the first pass had changed the
  `to_pipeline()` comment and the `negadoctor_module_row` doc, but missed this one field-level
  comment. Fixed: "scene-referred" → "display-referred". One-word edit; re-ran CI (all four
  steps green).

**Verified.** `scripts/ci-local.sh` — all four CI steps pass:
`cargo check --workspace` ✓, `cargo clippy --workspace` ✓ (26 pre-existing clone!
deprecation warnings in c41-ui, out of scope per CLAUDE.md), `cargo test --workspace
--release` ✓ (1082 tests pass, including the new parity test),
`cargo build --release -p c41 --bin c41-rs` ✓.

**Notes.** The second senior review passed clean after the one-word doc fix. All 1082 tests
green; the new `negadoctor_commit_params_matches_darktable` parity test pins the full
commit_params arithmetic against darktable negadoctor.c:239-267 for both colour and B&W
film stock.

## 2026-08-23 22:30 UTC — m4-116: Tone equalizer (exposure-channel tone mapping) live preview module

**What.** Wired the 24th live darkroom module: darktable's tone equalizer
(`toneequal.c`, iop_order.c v50_order pos 24.0 — before graduatednd 25, the
28.5 group). 24th module ⇒ parity audit item 2.1 updated in the same commit.

- `c41-core/iop/toneequal.rs` (preview path, +351 lines): `solve_weights` ports
  `build_interpolation_matrix` + `pseudo_solve`/`_solve_hermitian` (normal
  equations AᵀA·w = Aᵀy over the lower triangle only + Cholesky–Banachiewicz,
  matching the C pivot gates); `cached_correction_lut` builds the 80 001-entry
  correction LUT via the existing `darkroom_toneequal_build_lut` FFI behind a
  thread-local memo keyed on `f32::to_bits` of the nine gains (per-band rebuilds
  would cost ~640k exp calls each — same rationale as the basicadj LUT memo);
  `process_preview_pixels` = norm-2 luminance mask (`pixel_rgb_norm_2`,
  luminance_mask.h) + apply-LUT pixel pass for the `details == DT_TONEEQ_NONE`
  configuration. Four new unit tests: normal-equation residual ≈0, flat unity
  at default gains (±1% RBF residual), single-channel boost lands at ≈1.746×
  at −4 EV (numpy lstsq cross-check), uniform-patch scaling.
- `pipeline.rs`: `Stage::ToneEqual { gains: [f32; 9] }`; name/is_pixel_local
  (true — per-pixel mask+LUT lookup, no neighbours)/working_space (None)/apply;
  exhaustiveness test entry.
- `preview.rs`: `toneeq_on` + nine EV gains (noise −8 EV … speculars 0 EV,
  get_channels_gains order); encode/decode v16→v17 (+1 bool +9 f32, 680→717 B),
  PARAMS_LAYOUTS row + explicit `decode_v16_blob_defaults_toneeq_fields`
  backward-compat test; to_pipeline pushes raw gains (no arithmetic — the solve
  happens in `Stage::apply`) between Exposure and GraduatedNd; ordering test now
  enables graduatednd too and pins toneequal-before-graduatednd; new
  raw-passthrough/identity-gate parity test and an end-to-end midtone-boost
  render test. History container stays at v5 (m4-115 precedent).
- `darkroom/mod.rs`: `toneequal_module_row` — nine −2..+2 EV sliders labelled
  with darktable's params descriptions + EV positions ("Blacks (−8 EV)" …
  "Speculars (+0 EV)", gui toneequal.c:3205-3213); dispatch arm +
  LIVE_MODULE_LABELS. `history.rs`: "Tone equalizer" describe_change block +
  exhaustive-destructure drift guard + pinned blob length 717.

**Scope decisions** (documented in PARITY_AUDIT.md): runs
`details == DT_TONEEQ_NONE` only — darktable's shipped default is the guided
filter (`DT_TONEEQ_EIGF`), not ported, and with it go the smoothing/feathering/
blending controls and mask display. Smoothing fixed at √2 (darktable removed its
slider in module version 2). Luminance estimator hardcoded to `DT_TONEEQ_NORM_2`
(the C default). All-zero gains render exact identity in C41 vs darktable's ±0.7%
fitted curve deviation.

**Mistakes caught this increment.**
- I first placed ToneEqual after GraduatedNd citing "v50_order pos 28.0" — that
  figure came from the *_jpg order tables; v50_order has toneequal at 24.0,
  BEFORE graduatednd 25. Caught by re-checking the table boundaries during a
  self-review of the diff; placement + all comments fixed, ordering test
  strengthened to pin toneequal-before-graduatednd by enabling graduatednd.
- The WIP `solve_weights` doc claimed the C leaves `d->factors` calloc-zero on a
  failed solve. Wrong: `commit_params` ignores `pseudo_solve`'s FALSE return and
  `dt_simd_memcpy`s the unsolved exp2(gains) vector into `d->factors`
  unconditionally (dt_simd_memcpy(in, out, n) — checked imagebuf.h:63). C41 now
  documents returning zeros instead (uniform ×0.25 floor — visibly broken rather
  than subtly wrong); branch unreachable for the fixed σ=√2 SPD basis anyway.
- One bad Edit duplicated two lines of `whitebalance_module_row` while inserting
  the new row function above it; noticed immediately from the file state and
  reverted.

**Senior review (fricktrade-architect).** Verdict APPROVE-WITH-FIXES; no
correctness bugs. Reviewer independently cross-checked the least-squares
numerics against numpy f64 lstsq (weights and LUT values match our pins:
×1.7459 @−4 EV, ×0.9646 @0 EV) and verified the memo cache is sound under the
rayon band-split (bit-identical bands hold). Findings, all addressed:
- MEDIUM: PARITY_AUDIT.md item 2.1 not yet updated → updated in this commit
  (24 modules, tone equalizer moved out of the missing list, scope deviations
  recorded).
- LOW: stale comments from the pre-fix draft (negadoctor block still said
  "after graduatednd 28.0"; GraduatedNd block said "right after exposure 21")
  → both corrected.
- LOW: UI comments claimed all-zero gains are exp2(0)=1 "everywhere" without
  the ≤0.7% fit-residual caveat → reworded with a pointer to the core test.
- LOW (no action): per-band luminance Vec allocation matches the C's own
  per-call alloc; noted as a future thread-local-scratch candidate if it ever
  profiles hot.

**Verified.** `scripts/ci-local.sh` — all four CI steps pass (exit 0):
`cargo check --workspace` ✓, `cargo clippy --workspace` ✓ (zero new warnings —
verified 559 pre-existing warnings before and after the change; clone!
deprecations in c41-ui remain known/out-of-scope), `cargo test --workspace
--release` ✓, `cargo build --release -p c41 --bin c41-rs` ✓. c41-core suite
locally: 701 tests green including the 11 toneequal tests.

## 2026-08-24T00:10Z — m4-117: colour balance RGB (`colorbalancergb`) live preview module

**What changed.** darktable's most complex IOP is now the **25th live darkroom
preview module**, wired end-to-end following the m4-115/m4-116 pattern:

- `c41-core/iop/colorbalancergb.rs`: `PartialEq` on `CbRgbParams` (backs the
  identity gate) and on `CbRgbData` (travels inside a `pipeline::Stage`); new
  test pinning the equality gate against a neutral edit.
- `c41-core/pipeline.rs`: `Stage::ColorBalanceRgb { data: Box<CbRgbData>, space }`
  carries the **prebuilt** commit output (zone vectors, weights, fulcrums and the
  hue-indexed 512-entry gamut LUT — the dt-UCS build alone marches the RGB gamut
  boundary 25 600 times) computed once per render in `to_pipeline`, so each
  per-band apply call is pure pixel math. Pixel-local ⇒ band-parallel stays
  available. Placed between basicadj (40.0) and shadhi, matching v50_order pos
  41.5; ordering test extended with a `<sigmoid (45.3)` assertion.
- `c41-ui/preview.rs`: 34 `cb_*` fields (1 bool + 33 f32), encode v17→v18
  append-only (850 B), backward-compatible decode, single-source-of-truth
  `cb_params()`/`cb_is_neutral()` pair so the identity gate and stage emission
  can never disagree; enabled-at-defaults emits no stage (darktable's defaults
  are a neutral edit).
- `c41-ui/history.rs`: "Color balance RGB" change label, exhaustive-destructure
  and len-pin tests updated.
- `c41-ui/darkroom/mod.rs`: expander row with 29 sliders using darktable's soft
  ranges (colorbalancergb.c:1786-1990) + saturation-formula dropdown
  (JzAzBz/dt-UCS) encoded as an f32.

**The bug found before review could.** My own field-mapping audit caught that
the kernel's Yrg/LMS chain hardcoded Rec.2020↔XYZ-D65 conversions — correct on
the raw path but wrong for the non-raw pipeline which runs linear sRGB (JPEGs
would grade under mismatched primaries). Fixed by generalising: `process()` is
now a thin Rec.2020 alias over `process_in_space(..., rgb_to_xyz_d65,
xyz_d65_to_rgb)`; the Stage carries `space: ColorSpace` (Sharpen/Vibrance
idiom), `working_space()` reports it, and `to_pipeline` picks the gamut-LUT
matrix and converter pair together per space. New pub
`color::{SRGB_TO_XYZ_D65_T4, srgb_to_xyz_d65}` complete the linear-sRGB↔XYZ-D65
pair (matrix rows sum to the D65 white 0.95047/1.0/1.08883).

**Cross-space semantics discovered while testing that fix:** results do NOT
agree bit-for-bit across working spaces, and shouldn't — the UCS soft-clip knee
starts at 0.8× the gamut boundary and each space's LUT encodes its own RGB cube
(sRGB ⊂ Rec.2020), exactly as the C whose LUT comes from
`work_profile->matrix_in`. A probe over an input grid showed 3–10% chroma
divergence everywhere once saturation edits land in the knee. The tests pin
what actually holds: delegation alias sanity, finite/non-negative output in
both spaces (catches crossed converter/LUT wiring), sRGB-clips-harder-than-
Rec2020 under a heavy push, UI-level assert_ne between spaces' stage data, and
the LUT-builders-distinguish-matrices test now using the shared matrix const.

**Senior review (fricktrade-architect).** The named agent was unavailable this
session — four launch/resume attempts died with API 402 credit errors; after
the fourth the review was performed by a fork subagent acting as senior
reviewer with the same checklist (substitution documented here per workflow).
Verdict **APPROVE-WITH-FIXES**, one finding: a stale doc fragment from the pre-
generalisation draft left stranded above `pub type RgbXyzConv` describing a
Rec.2020-only function contract — fixed by moving it onto `pub fn process`.
Reviewer independently verified params mapping vs colorbalancergb.c:60-105,
all 14 soft ranges (:1791-2004), the from_params↔commit_params arithmetic
(:1087), encode v17→v18 discipline, ordering claims, and the space fix.

**Verified.** `scripts/ci-local.sh` exit 0 (twice — after the space fix and
again after the doc fix): check/clippy/test --release/link all pass; 705 core +
92 + 249 UI tests green; zero new clippy warnings (remaining families
pre-existing).

**Also went wrong, fixed inline:** two malformed Edit calls during the kernel
generalisation (a placeholder fragment replacing the tail of
`xyz_d65_to_srgb`; a joined comment/code line in the UI test) — both caught
immediately from tool feedback and repaired; one E0308 in the new test (fn
items have unique types — cast to `RgbXyzConv` fn pointers); one genuinely
wrong test premise (cross-space XYZ identity) replaced twice before landing on
the invariants above.

## 2026-08-24T07:44Z — m4-118: filmic RGB (`filmicrgb`) live preview module

**What changed.** Filmic RGB wired as the **26th live darkroom preview module**
(v50_order pos 46.0 — display-transform cluster, after sigmoid 45.3, before
colisa 47.0; pinned by the canonical-order test).

- `c41-core/iop/filmicrgb.rs` gained the live-preview driver (~350 lines +
  12 tests): `FilmicParams` (from the `$DEFAULT` introspection annotations),
  `gauss_solve` (faithful f64 port of `gaussian_elimination.h` incl. partial
  pivoting and the C's column-swap elimination), `compute_spline` (V3 branch
  only + POLY_4 toe/shoulder quartics solved in f64 — the current colour
  science and the default curve types), `FilmicData::from_params`
  (commit_params scalar subset + `exp_tonemapping_v2` norm endpoints +
  `powf(y, power)` display endpoints), and `matrices_for_space` —
  `prepare_RGB_Yrg_matrices` adapted to our D65-referenced working spaces:
  C chains RGB(D50)→CAT16→LMS because its pipeline is D50-referenced; ours
  drops both CAT16 legs, which is algebraically exact since our
  `rgb_to_xyz_*` maps land directly on XYZ D65. Composition uses the
  stored-transposed plain-array product convention (`M[in][out]`,
  `out[r]=Σ_c M[c][r]·in[c]` ⇒ A then B composes as `mul_mat4(A,B)`),
  pinned by a test asserting the matrix path ≡ the scalar converter chains.
- `Stage::FilmicRgb { data, space }` in pipeline.rs: per-pixel → parallel-path
  eligible; carries its ColorSpace; `working_space()` reports it. The apply arm
  calls `process_in_space` with per-space matrices (Rec.2020 raw / linear sRGB
  non-raw) into the existing `darkroom_filmicrgb_v5` FFI kernel
  (has_work_profile=1 with a standard-form transposed input matrix,
  nonlinearlut=0 so LUT pointers are never dereferenced, POLY_4 ×2,
  use_output_profile=0 with export set duplicating working set).
- UI: PreviewParams **v19** — append-only bool #26 + trailing f32s f[206..212]
  (black −16..−0.1 def −8 EV, white 0.1..16 def 4 EV, hardness 1..10 def 4,
  latitude 0.01..99 def 0.01 %, contrast 0..5 def 1, balance ±50 def 0,
  saturation ±200 def 0); ENCODED_LEN 879 (pinned-length test updated);
  PARAMS_LAYOUTS row added; v18 blobs decode with filmic fields defaulted
  (test). Single-source-of-truth `filmic_params()` mapper shared by
  `is_identity` (filmic_on-only gate — a tone curve is never a no-op while
  enabled) and `to_pipeline`. history.rs describe_change "Filmic RGB" group +
  exhaustive destructure drift-guard extension. darkroom/mod.rs: dispatch arm,
  LIVE_MODULE_LABELS entry, 7-slider module row using the $MIN/$MAX ranges
  (C's soft ranges deliberately not modelled — our sliders are linear).
  Catalog already carried the label, so it went live without catalog changes.

**Documented deviation:** filmic ships OFF at defaults. Darktable auto-enables
it via the scene-referred preset plus `reload_defaults`' auto-exposure
adjustment — workflow logic we don't replicate — unlike sigmoid there is no
identity-at-defaults shortcut either way: enabling it always tone-maps.

**Real defect found & fixed en route (m4-117 follow-up):**
`color::SRGB_TO_XYZ_D65_T` shipped with G→Z = 0.0721750 (B→Y's value) instead
of the correct **0.1191920** (confirmed against colorspaces.c:915-922 and by
inverting the stored XYZ→sRGB matrix), so `srgb_to_xyz_d65` was *not* the
inverse of `xyz_d65_to_srgb` — non-raw previews ran subtly wrong chroma
through the pipeline converters and colour-balance-RGB gamut LUTs since m4-117
(b12067bfc8). One-entry fix; no test pinned the wrong value (full suite stayed
green post-fix). Caught precisely because the new matrix-composition round-trip
test refused to pass in the sRGB space.

**Also went wrong, fixed inline:** 12 first-compile errors (f32/f64 mixes in
the balance block and `from_params`; `[f32;4]` vs `*const f32` matrix pointers;
fn-item uniqueness in a test match — cast via `type Conv = fn(...)`); and a
nasty **release-mode trap**: the spline solves were originally wrapped as
`debug_assert!(gauss_solve(...))`, which under `--release` never evaluates the
call — coefficients silently stayed zero and five spline/process tests failed
with degenerate values. Solves now run unconditionally with the result asserted
separately (NOTE comment left at both sites); swept the touched files for other
instances — none.

**Senior review (fork subagent acting as senior reviewer).** The named
fricktrade-architect agent died with the same API 402 credit error as in
m4-117; after the failure the review was performed by a fork subagent with the
full eight-point checklist against the C sources (substitution documented here
per workflow). Verdict **SHIP**: compute_spline verified line-for-line against
filmicrgb.c:2445-2766 including both quartic systems and coefficient unpack
order; gauss_solve against gaussian_elimination.h; matrix composition proven
under the transposed-storage convention; FFI call site argument-for-argument
against filmicrgb.c:1712-1733; from_params endpoints against commit_params +
exp_tonemapping_v2; CAT16-leg drop confirmed algebraically exact; sRGB fix
values confirmed canonical; UI append-only discipline and slider ranges
verified against the $MIN/$MAX annotations. Zero BLOCKER/MAJOR findings; two
INFO notes (per-band `matrices_for_space` rebuild ~100 flops — negligible next
to the pixel loop; f64-widened spline node storage — kernel never sees x/y).

**Verified.** `scripts/ci-local.sh` exit 0 — all four CI steps (check, clippy,
test --release, release link) pass; 716 c41-core + 251 c41-ui tests green;
zero new clippy warning families.

## 2026-08-24T15:10Z — m4-119: highlight reconstruction (`highlights`) live preview module

**What changed.** Wired darktable's highlight reconstruction as the 27th live
darkroom module. The defining difference from modules 1–26: it runs on the raw
CFA mosaic **before demosaicing** (darktable's temperature → highlights →
demosaic order), so it is *not* a pipeline `Stage`. It threads through
`RawImage::to_linear_rgba_with(method, hl: Option<HlOpts>)`, and any UI change
re-decodes the raw (like the demosaic-method selector) instead of re-rendering
pipeline stages.

- **Decoder change (the real work):** `normalize_cfa`/`normalize_xtrans`
  previously hard-clamped photosites at 1.0 at load — there was nothing left to
  reconstruct. They now carry over-range values (`.max(0.0)`); the legacy
  `None` decode path clamps at 1.0 first, so its output is byte-identical to
  pre-m4-119 (pinned by `legacy_none_path_matches_clip_mode_at_unity_wb`,
  an `assert_eq!` on full RGBA). Side effect, flagged by review and kept:
  NaN now floors to 0.0 instead of propagating (strict improvement — corrupt
  white-level metadata used to poison the frame).
- **WB moved for the hl path:** new `apply_white_balance_mosaic` applies WB on
  the mosaic before reconstruction; the post-demosaic `apply_white_balance` is
  skipped when hl is on (otherwise applied twice). Thresholds use
  green-normalised ratios so they land at the same scene values either way.
- **Driver:** `highlights::reconstruct_mosaic` over the already-ported unsafe
  kernels (`darkroom_highlights_opposed_{mask,dilate,chroma,output}_raw`),
  replicating `_process_opposed`'s keep=FALSE path; clip mode is a safe
  per-channel min. Opposed semantics verified against the kernels before
  testing: output = `max(inval, refavg + chrominance)` only where
  `inval >= clips[color]` — it RAISES saturated channels toward what opposing
  channels imply, never lowers; if nothing is clipped the pass is an exact
  copy, which justifies the `!anyclipped` skip. `output_raw` takes separate
  in/out pointers, so the driver reconstructs into a scratch plane and
  `copy_from_slice`s back (aliasing one slice would be UB).
- **UI:** `PreviewParams` v19→v20 (+`hl_on`, `hl_opposed` bools; `hl_clip`
  float #213; blob 879→885), backward-compat decode keeps old blobs loading
  with hl OFF. `PreviewCtx` gained `decode_path`/`decode_method` so
  `spawn_decode(ctx)` reads its inputs from ctx — module rows only capture
  ctx, and the new row needs to trigger decodes. `highlights_module_row` is
  hand-rolled (not `module_expander`, whose notify handlers re-render stages):
  enable switch + method DropDown ("Clip highlights"/"Inpaint opposed",
  index == the `hl_opposed` bool) re-decode immediately; the clipping-threshold
  slider (0..2, default 1) debounces 160 ms like straighten. Row hidden for
  non-raw files. Export parity:
  `render_export_rgb8/16` pass `params.hl_opts()`, so a Rust-native export
  matches the preview.
- **Ships OFF** (`hl_on=false` default = exactly the old clamped decode), so no
  existing image changes appearance until the user opts in.

**Deviations vs upstream C (documented for the next session):**
(1) Bayer patterns whose second green encodes as colour index E=3 take the
clip fallback — the ported kernels slice `clips[0..3]`/`[u8;3]` mask planes and
would panic on a fourth colour, where the C harmlessly reads `clips[3]` from
its aligned struct. Gated by `has_e_colour`; X-Trans unaffected.
(2) Two of six methods wired (opposed = C default, plus clip); LCh, guided
laplacians and segmentation-based remain unwired.
(3) C compares thresholds against raw temperature coefficients;
C41 green-normalises both data and thresholds (scaling cancels in the
comparisons; matches the legacy `apply_white_balance` R/B÷g convention exactly,
keeping hl-on/off colour balance identical).
(4) Non-raw inputs are untouched by the module (darktable forces CLIP there —
a no-op on already-clamped sRGB data).

**Went wrong en route:** the Edit tool garbled two large insertions into
`highlights.rs` (stray glyphs, prose inside a match arm, a swallowed
`#[cfg(test)]` opener) — recovered via write-to-/tmp + scripted splice; that's
the pattern for big blocks now. An early test premise was wrong (assumed
opposed *lowers* clipped values; re-read the kernels, redesigned the test as a
bright-field recovery scenario before anything shipped). A garbage dead-loop
accidentally spliced into `apply_white_balance_mosaic` was caught and removed
pre-commit.

**Senior review (fork subagent acting as senior reviewer).** fricktrade-
architect died with the same OpenRouter API 402 credit error as m4-117/m4-118;
substitution documented here per workflow. Verdict **SHIP**: zero BLOCKER,
zero should-fix findings across unsafe-call contract checks (kernel signatures,
scratch-plane aliasing avoidance, null-`tmpout` branch confirmed supported),
colour-3 panic gate, legacy byte-identity proof, params-blob append-only
discipline, decode-race/debounce/closure-lifetime review of the UI wiring, and
export-parity check. Nits recorded, not gated: history label priority order
(hl checked after filmic though it runs first in the chain — cosmetic);
a failed decode delays autosave/history arming until the next successful
render (same exposure as the demosaic dropdown); export transients ~2×W·H·f32
extra while opposed mode is enabled and something is clipped (same allocation
class as the existing RGBA result); X-Trans opposed coverage is smoke-only.

**Verified.** `scripts/ci-local.sh` exit 0 — all four CI steps pass keyed off
exit codes; 723 c41-core (+7 new) + 92 c41-db + 253 c41-ui tests green under `--release`.

## 2026-08-24T16:52Z — m4-120: denoise (profiled) (`denoiseprofile`, wavelets) live preview module #28

Wired darktable's denoise (profiled) — wavelets mode — as a normal pipeline
stage. Core: new `c41-core::iop::eaw` porting eaw.c (`eaw_dn_decompose`
5×5 B-spline à-trous with edge-avoiding `dn_weight`, `eaw_synthesize`
soft-threshold accumulate, float `fast_mexp2f` bit-punning port) and a
`wavelets_denoise` driver in `denoiseprofile.rs`: VST precondition → per-scale
decompose + BayesShrink thresholds (adjt = 8.0 exactly — the default force
curve is flat 0.5, so Catmull-Rom gives force²·4 ≡ 1) → synthesize → residual
add → backtransform; alpha lane restored from input because the VST kernels
garble lane 3 (C callers never read it). `Stage::DenoiseProfile` with
`is_pixel_local() == false` (multi-scale neighbourhood), placed between
temperature and exposure per v50_order pos 9/10 — pinned by the ordering test.
UI: PreviewParams v21 (+2 bools/+3 f32, blob 885 → 899, pin test bumped;
HISTORY_ENCODE_VERSION stays 5 via per-entry version peeking), backward-compat
decode test for v20 blobs, identity/bypass/to_pipeline gates keyed on `dn_on`,
history label "Denoise (profiled)" + field-drift guard, `module_expander` row
(enable, Y0U0V0/RGB DropDown, Strength 0.001..4.0 / Preserve shadows 0..1.8 /
Bias ±10 — C soft ranges). Export inherits through to_pipeline. Deviations all
documented in code + PARITY_AUDIT 2.1: wb=[1,1,1] (post-WB buffer) with the C's
strength·compensate scaling carried by wb_s into the kernels; generic
Poissonian a=1e-4 instead of noiseprofiles.json; use_new_vst only; wavelets
only; clamped indexing refusing the C's narrow-image row bleed.

**Senior review (fork subagent acting as senior reviewer).** fricktrade-
architect died with OpenRouter API 402 again (fourth increment running);
substitution documented here per workflow. The fork's run was itself split
across a session restart and resumed from its saved transcript. Verdict SHIP
WITH FIXES. Finding 1 (MAJOR): strength was completely inert in RGB mode —
the v2 kernels take no colour matrix, and the driver passed unit wb where the
C passes wb scaled by strength·compensate_strength (:1385), which is RGB
mode's *only* strength carrier; finding 2 (MINOR): same root cause left the
Y0U0V0 backtransform's bias term at bias·1 instead of bias·s. One shared fix:
wb_s = [s,s,s,1] into precondition_v2/backtransform_v2/backtransform_yuv,
matrices stay on pre-strength wb per C order. Finding 3 (MINOR): two stale
field-doc ranges in preview.rs corrected. Nits (no action): describe_change
arm placement; whole-frame serial re-render per slider tick is heavy but
matches every existing whole-frame stage (tick-coalescing noted as a future
increment).

**Went wrong en route:** my first regression test asserted "higher strength ⇒
lower residual noise in RGB mode" and FAILED WITH THE FIX IN PLACE — both
strengths produced identical output. Working through the math: with the
generic Poissonian profile the transformed-space noise variance sits orders
of magnitude below σ_band² (=1 at scale 0), so the BayesShrink denominator
clamps at its 1e-6 floor, thresholds saturate, detail is wiped wholesale, and
the result reduces to the coarse pyramid — strength-independently. That is
faithful C behaviour given the same profile (strength only bites once a real
per-camera profile stabilises variance towards 1), not a port bug. Replaced
with tests whose premises hold: exact forward/backward cancellation at
off-default strengths in both modes, and the bias·wb drift being
strength-dependent (pre-fix the means were exactly equal; measured drift
≈ 1.9e-5, bounds calibrated 4e-6..1e-3 after a first 1e-4 epsilon proved too
coarse). Also fumbled one Edit that clipped a test function's closing brace
and briefly nested a #[test] inside another test — caught on read-back and
restructured before compiling.

**Verified.** `scripts/ci-local.sh` exit 0 twice (pre-fix tree and final tree)
— all four CI steps keyed off exit codes; 733 c41-core (+10 new: 4 eaw,
6 wavelets incl. 2 review regressions) + 254 c41-ui tests green; clippy exit 0
(pre-existing warning classes only).

## 2026-08-24T19:40Z — m4-121: bloom (`bloom`) live preview module #29

**What changed.** Bloom (display-referred glow, iop_order.c v50 pos 61) wired
end to end as a normal pipeline stage. The enabler was the missing box-mean
kernel:

- **New `c41-core/src/iop/box_filters.rs`** — port of the ch=1 path of
  `dt_box_mean` (box_filters.cc `_blur_horizontal`:185 / `_blur_vertical`:308 /
  `_box_mean<1>`:361). Five-phase sliding window preserved exactly:
  left-half accumulate over MIN(radius,dim); grow while `pos≤radius &&
  pos+radius<dim`; stall when radius>dim/2; bulk subtract-*then*-add;
  right-end decrement with `hits--` before the sub. Edge windows shrink via
  hits-counting (window∩bounds — no clamped replication). Plain f32
  accumulation, matching bloom's non-Kahan ch=1 dispatch. Documented
  layout-only deviations: no MAX_VECT vectorisation; full-length scratch
  instead of the power-of-two circular mask (numerically identical — every
  line is independent).
- **`bloom.rs` driver** — safe `process()` composing the existing tested FFI
  kernels (`darkroom_bloom_gather` → box_mean_1ch → `darkroom_bloom_blend`)
  with bloom.c process():137-146 derivations replicated including C int
  truncation: rad = int(256·(min(100,size+1)/100)), radius = MIN(256,
  ceil(rad·scale/iscale)) with scale/iscale ≡ 1 whole-frame (ratio kept as a
  documented parameter slot for future tiled callers), blur radius =
  (2r+1)/2 ≡ r, scale = exp2(min(100,strength+1)/100), BOX_ITERATIONS = 8.
- **pipeline.rs** — `Stage::Bloom { size, threshold, strength, space }`;
  NOT pixel-local (whole-frame serial path, like Sharpen/Lowpass); RGB↔Lab
  sandwich in the apply arm per the Colorize precedent.
- **UI** — PreviewParams v22 append-only (+bl_on bool #30, +bl_size/
  bl_threshold/bl_strength floats #217-219; blob 899→912; PARAMS_LAYOUTS +=
  (22,31,220)); defaults from bloom.h introspection (20/90/25, ranges 0..100);
  identity gate flag-only (correct: scale ≥ exp2(0.01) > 1 means even
  strength=0 changes output once anything passes the threshold, so there is no
  neutral-while-enabled — same reasoning class as Colorize); to_pipeline emits
  between velvia (57) and colorize (62) per iop_order 61, pinned by the
  ordering test with positional assertions; history label "Bloom" + drift
  guard + pin bump to 912; module row with three sliders mirroring the C
  ranges; liveness test flipped ("Bloom" now live, "Grain" negative case).
  Export parity inherits through to_pipeline.

**Went wrong.** Two of my own test bugs caught before review: (1) the glow test
put the spot at L=100, where the screen blend caps and "gains L" is
unobservable — spot moved to L=95; (2) I forgot that 8 iterations compound the
blur reach (~8×radius per axis), so a 21px frame was entirely inside the halo
and an "outside the halo ⇒ exact identity" assertion failed at the corner
(30.258 vs 30.0). Frame enlarged to 61px with decay asserted instead of
isolation. Also one Edit accidentally clipped the whitebalance_module_row
header when inserting bloom_module_row; restored immediately and verified.

**Senior review** — fricktrade-architect died on API 402 again (recurring);
fork subagent stood in per the documented fallback, verified every claim
against box_filters.cc/bloom.c/iop_order.c line-by-line. Verdict: **SHIP**,
zero BLOCKER/MAJOR/MINOR. Explicitly confirmed subtle spots: five-phase
boundary conditions, scratch copy-before-overwrite semantics, index-safety of
the op underflow paths, `as i32` ≡ C truncation for these positive in-range
values, exp2 sign-flip exactness, alpha handling through the Lab sandwich,
four-way gate agreement, and that upstream darktable in this repo calls these
very gather/blend Rust exports. Nits applied: mass-conservation comment
reworded (interior-only invariant, not global — border windows drop taps),
merged duplicate `use super::` imports.

**Verified.** `scripts/ci-local.sh` exit 0 twice (pre-review tree and post-nit
tree) — all four CI steps keyed off exit codes; 741 c41-core tests (+10 new:
5 box_filters incl. naive-reference differential across shapes/radii/
iterations, 3 new bloom driver tests, plus existing FFI kernels) and 255
c41-ui tests green in --release.

## 2026-08-24T19:56Z — m4-122: tone curve (`tonecurve`) live preview module #30 + curve editor widget

**What changed.** The first curve-based module wired, which required building
the missing widget class. Two enablers landed together:

- **New `c41-core/src/curve_tools.rs`** — port of src/common/curve_tools.c
  (the UFraw-lineage V1 sampler used by tonecurve/basecurve/atrous/
  denoiseprofile) plus the src/gui/draw.h wrappers. Thomas solver over the
  Burkardt D3 collapsed tridiagonal layout (a[3i]=super from row i−1,
  a[3i+1]=diag, a[3i+2]=sub from row i+1); boundary conditions ibcbeg/ibcend
  ∈ {natural, first-derivative}; MONOTONE_HERMITE computes Fritsch-Carlson
  tangents but *evaluates* via catmull_rom_val, mirroring draw.h's
  spline_val[] dispatch (the C never calls spline_cubic_val for that type);
  unknown type → None → box-diagonal fallback. `curve_data_sample` composes
  CurveDataSample exactly: box transform to [0,1] (all real callers), flat
  endpoint extension for x before the first anchor, int-truncation
  val·(res−1)+0.5 with clamping to [minY,maxY]·(res−1), ÷0x10000 float
  conversion.
- **New `c41-ui/src/darkroom/curve_editor.rs`** (~510 lines) — interactive
  L-curve editor: DrawingArea canvas painting grid/dashed diagonal/the spline
  itself via `curve_data_sample` (drawn curve IS the applied curve) plus
  darktable's basic gestures: press-to-grab and drag anchors, click empty
  space to insert (≤ 20 anchors, MAX_ANCHORS cap), double-click removes an
  interior node, endpoints pinned at x=0/1, interior x kept strictly between
  neighbours ± X_EPS. Draw func captures WeakRefs only (no ownership cycle);
  drag state is Rc<Cell<Option<usize>>> so the draw closure can read it.
- **`tonecurve.rs` build_lut + pipeline.rs Stage::ToneCurve** — commit_params
  replicated: sample curves at 65536, scale L×100 / ab×(256−128)
  unconditionally, then AUTOMATIC_RGB autoscale re-derives t_l IN PLACE via a
  ProPhoto round-trip (AUTOMATIC_XYZ ported alongside; Lab→XYZ→L-curve→Lab);
  signed CLAMP index semantics preserved via i64 widening; unbounded tails
  extrapolated by estimate_exp over x∈{0.7..1.0}·xm with mirrored lookups
  (slot order L-right, a-right, a-left, b-right, b-left). Placed at iop_order
  pos 48 (colisa 47 < tonecurve < levels 49, pinned by the ordering test);
  pixel-local = true (pure LUT lookups → rayon band path, exhaustiveness test
  updated); working space Rec2020/linear-sRGB with Lab sandwich preserving
  input alpha.
- **UI** — PreviewParams v23 append-only (+tc_on bool #31, +tc_unbound bool
  #32, +tc_type/tc_autoscale/tc_preserve/tc_nnodes floats #220-223, +40 f32
  interleaved L-anchor coords #224-263; blob 912→1090; PARAMS_LAYOUTS +=
  (23,33,264)); decode starts from defaults so v22 blobs keep identity
  anchors. Defaults are darktable's: interpolator monotone Hermite(2),
  autoscale AUTOMATIC_RGB(3), preserve_colors AVERAGE(3), unbound_ab=true,
  identity 2-node L anchors. Module row = enable switch + interpolator
  DropDown (seeded BEFORE connect_selected_notify per the rebuild invariant)
  + the editor canvas; history label "Tone curve" with full 7-field drift
  guard; catalog entry in the Tone group; liveness list updated. Export
  parity inherits through to_pipeline.

**Scope deviation (documented).** First slice ships the **L channel only**;
a/b channel tabs are deferred but their params already exist and decode
(tc_nodes default = darktable's 3-node identity monotone-Hermite ab curves),
so adding tabs later is additive, not a blob bump. Identity gate is flag-only
(Bloom precedent).

**Went wrong.** My own regression test for the review fix exposed a deeper
hole in the fix itself: collision detection compared the *raw* x while
storage clamps, so an out-of-range click (x=−3) clamps onto the x=0
endpoint's column and would have produced two anchors at x=0 — exactly the
non-increasing-anchor hazard the fix targeted. The check now runs on the
clamped coordinate, and the test asserts refusal for duplicate columns, near-
endpoint clicks, AND out-of-range clamps, plus no-mutation on refusal. Also:
an earlier draft embedded a `for` loop inside an array literal (invalid Rust);
a v22-decode draft started from zeroed nodes instead of defaults (would have
snapped legacy blobs' anchors off the diagonal); the pin test briefly carried
both the old 912 assert and the new 1090 one; hit_node needed total_cmp +
then_some restructure after `?` on Option-in-Result confusion; remove_node's
bool return forbids `?` (match-based rewrite). Every-node highlight-ring bug
caught by self-review before commit (selected_ring returned Some(idx)
unconditionally → all nodes ringed; replaced by drag_idx comparison).

**Senior review** — fricktrade-architect died on API 402 again (recurring);
fork subagent stood in per the documented fallback and verified the
load-bearing subtleties against curve_tools.c/draw.h/tonecurve.c line-by-line
(unconditional ×100/×256−128 scaling before autoscale re-derivation, signed
CLAMP semantics, tail-extrapolation slot order, D3 layout, catmull_rom_val vs
spline_cubic_val dispatch asymmetry, borrow scoping around queue_draw,
drag-index invalidation on refused removals). Verdict: **FIX FIRST**, both
findings one-liners, both applied: (1) ibcend==1 boundary diagonal carries
Δt/3, not 1 (curve_tools.c:324 — verified in source before fixing); (2)
insert_node must refuse duplicate-x inserts (accepting one silently snaps the
whole curve to the identity diagonal while nodes stay visible). NITs
consciously accepted, not deferred: isotropic hit tolerance ≈14.7px
horizontally on the 240px widget; NaN-in-blob falls back to box-diagonal
rather than sanitised (unreachable from the editor — every write clamps).

**Verified.** `scripts/ci-local.sh` exit 0 twice (pre-review tree and again
on the post-fix tree) — all four CI steps keyed off exit codes, never grep.
+16 c41-core tests (11 curve_tools sampler incl. hand-solved natural cubic,
non-increasing-anchor fallback and leading-gap clamp; 5 build_lut parity) and
+8 c41-ui tests (6 editor gesture-contract helpers incl. duplicate-x refusal
with no-mutation, v22-decode fallback, params→pipeline end-to-end): 757 core
+ 263 ui green in --release, plus 92 bin tests; only known pre-existing
warnings. New end-to-end test pins midtone darkening through the
full params→pipeline path (Catmull-Rom node at (0.5,0.35): 128-grey drops
below 110, whites stay >200).

## 2026-08-24T21:20Z — m4-123: RGB curve (`rgbcurve`) live preview module #31

Fourth curve-family module and the second consumer of the m4-122 editor, which this
increment generalised into a shared **multi-channel** widget. `Stage::RgbCurve` applies
per-channel user-drawn curves directly to the working RGB lanes (IOP_CS_RGB — no Lab
sandwich; darktable's rgbcurve likewise never leaves RGB).

**Core** (`c41-core/src/iop/rgbcurve.rs`, extending the pre-existing FFI-kernel file):
`build_luts` samples three 65 536-entry tables through the shared
`curve_tools::curve_data_sample` and replicates `_generate_curve_lut`
(rgbcurve.c:1671–1742) right-tail extrapolation — xm = last node x, sample xs at
{0.7, 0.8, 0.9, 1.0}·xm with y read from `table[CLAMP((int)(x·0x10000))]`,
exponent fitted by `eval_exp` ≡ `dt_iop_estimate_exp`; the LUT lookup index
`((v·0x10000) as usize).clamp(0, 0xffff)` matches C's CLAMP including negative
inputs (Rust's saturating float→uint cast lands negatives at 0 before the clamp).
`process_pixels` implements all three process() branches: MANUAL → per-channel
tables; AUTOMATIC + preserve==0 ("none") → R table applied to all three lanes;
AUTOMATIC + preserve≠0 → single ratio curve(rgb_norm)/rgb_norm scaling all lanes
equally with lum≤0 passthrough; alpha passes through (`o[3]=i[3]`). Notably C's own
process() already delegates into the Rust FFI kernel `darkroom_rgbcurve_process`, so
preview and production share one implementation — pinned byte-equal by
`ffi_kernel_matches_safe_path` over 7 pixels including the v=1.0 boundary (the FFI
delegation initially transposed the coefficient matrix; caught by that test, fixed
with a per-channel `copy_from_slice` zip fold). Placed at iop_order v50 pos **50.5**
(rgblevels 50.2 < rgbcurve < relight 51, verified in iop_order.c) → emitted after
Levels; ordering test pins `levels < rgbcurve < velvia`. Pixel-local → rayon band
path. Identity gate flag-only (`rc_identity = !rc_on`) per Bloom/ToneCurve precedent.

**UI**: params **v24** (+1 bool/+8 f32/+120 interleaved node coords, blob 1090→1603,
length-pinned by test). The encode layout is append-only and an early draft violated
it: the 8 rc scalars went *inside* the main float list, shifting tone-curve anchors
from float 224 to 232 and breaking six tests across persist/history/dialogs before it
was caught — rc scalars now emit in their own loop after the tc anchor loop (floats
264–271), R/G/B node arrays at 272–311/312–351/352–391; v23 blobs still decode with
defaulted rc fields (dedicated fallback test). The curve editor was rebuilt as a
shared multi-channel builder (`multi_curve_area`): channels are injected as closures
(`TypeFn`/`SyncFn`) plus per-channel stroke colours, so tone curve (single amber L)
and RGB curve (R/G/B-coloured lanes) share one gesture implementation; gestures route
through the active channel, release clears every channel's drag slot (no stale drag
across channel switches), and `set_channel_nodes` is the single write site pairing
each array with its count. RGB-curve row: interpolator combo writes all three
`rc_type_*` (mirrors `interpolator_callback`), linked/independent mode combo toggles
preserve-colors row sensitivity (`_rgbcurve_show_hide_controls` analogue), norm combo
(none/luminance/max/average/sum/norm/power = DT_RGB_NORM_*), R/G/B channel selector.
History label "RGB curve"; drift-guard destructure extended with all 12 rc fields;
catalog row was pre-existing.

**Deviations (documented):** compensate_middle_grey path omitted (default-off; needs
work-profile matrix plumbing); work-profile early-return cache in commit_params
omitted (C's is pure memoization — commit_params builds nothing, so building LUTs
once per render is faithful); picker_scale work-profile luminance is GUI-only;
corrupted-blob spline types degrade to curve_tools' box-diagonal fallback (safe, not
bit-faithful to C's default arm); channel selector stays visible-but-inert in linked
mode (semantic superset of C's tab hiding).

**Senior review** — fricktrade-architect died on API 402 again (recurring); fork
subagent stood in per the documented fallback and verified against source:
LUT generation + tail arithmetic, trivial commit_params (validates build-once-per-
render), init() 2-node identity defaults for all channels, header $DEFAULTs
(MONOTONE_HERMITE/AUTOMATIC_RGB/LUMINANCE/compensate_middle_grey=0), iop_order 50.5
placement, process() xm derivation and FFI call shape, v24 offsets offset-by-offset
both directions, editor borrow-safety (editor RefCells ≠ ctx.params RefCell), and
cross-file consistency (identity gate/bypassed/drift guard/ordering pin).
Verdict: **APPROVE** — zero BLOCKER/should-fix findings; three nits recorded
(truncating type cast mirrors the existing tc pattern; unknown-blob spline type
fallback; linked-mode channel-selector visibility), none actioned.

**Verified.** First ci-local attempt failed for a dumb reason: I ran the script
*inside* a Docker container (`docker run … bash scripts/ci-local.sh`), where docker
does not exist — it is the host-side entry point that drives Docker itself. Re-run on
the host: `scripts/ci-local.sh` exit 0, all four CI steps keyed off exit codes, never
grep. c41-core **761** + c41-db 92 + c41-ui **267** green in --release (+7 rgbcurve
core tests incl. FFI/safe byte equality and the three process() branches; +4 ui tests
for channel routing/seed clamping/v23 fallback/ordering/end-to-end midtone darken
with linked-channel equality), plus the release link of `c41-rs`.
PARITY_AUDIT.md row 2.1 updated to 31 modules in the same commit.

## 2026-08-24T22:05Z — m4-123b: fix RGB curve stage order (v50_order 42.0, not 50.5)

**The bug.** The m4-123 cut emitted `Stage::RgbCurve` immediately after Levels,
citing iop_order "v50 pos 50.5 — after rgblevels 50.2, before relight 51". That
pin misread src/common/iop_order.c: rgbcurve@50.5 lives in **legacy_order**
(lines 81–176), not v50_order (298–416). In v50_order — the table every other
C41 stage pin follows — rgbcurve sits at **42.0**, between colorbalancergb 41.5
and rgblevels 43.0, i.e. BEFORE the whole display-referred tone-mapping cluster
(filmic 45.0 / sigmoid 45.3 / filmicrgb 46.0 / colisa 47.0 / tonecurve 48.0 /
levels 49.0 / shadhi 50.0). Found while surveying basecurve for m4-124: its
v50_order neighbours forced a re-check of which array each pin actually comes
from. Practical impact: with both modules active, an RGB curve applied after
tone-mapping/levels produces a different (wrong-order) result than darktable's
scene-referred placement.

**The fix.** Moved the RgbCurve emission block to right after ColorBalanceRgb
in `to_pipeline` (pure move — same code, same gate); comment rewritten with a
note recording the legacy_order confusion; canonical-ordering test updated
(expected array gains "rgbcurve" between "colorbalancergb" and "shadhi";
positional asserts now rc > colorbalancergb and rc < sigmoid); PARITY_AUDIT.md
row 2.1's m4-123 paragraph corrected in the same commit (position fixed,
blocker documented, premature-APPROVE process note kept).

**Review.** fricktrade-architect died on API 402 again; the fork fallback was
also unavailable (fork-in-fork), so the review was performed inline by this
session with a *programmatic* cross-check instead of eyeballing: every C41 pin
(basicadj 40 … velvia 57) parsed against all five order tables and confirmed to
match v50_order exactly; rgbcurve = 42.0 in v30/v50/v30_jpg/v50_jpg and 50.5
only in legacy_order; basecurve = 44.0 in all modern tables (the input for
m4-124). No findings beyond the fix itself.

**Verified.** `scripts/ci-local.sh` exit 0 — all four CI steps keyed off exit
codes: c41-core 761 + c41-db 92 + c41-ui 267 green in --release (ordering test
now pins colorbalancergb < rgbcurve < sigmoid). Note: an editing race with a
concurrent worker on preview.rs briefly produced two RgbCurve emission blocks;
resolved by keeping the concurrent (better-commented) version and dropping the
duplicate before any build ran.

## 2026-08-24T22:41Z — m4-124: base curve (`basecurve`) live preview module #32

Wired darktable's **base curve** into the live preview, completing the curve
trio (tone → RGB → base). Core: `basecurve.rs` gained the pipeline-facing layer —
`build_table` (now a thin delegate over rgbcurve's refactored
`CurveLut`/`build_single_lut`, so both modules share one LUT builder and one
estimate_exp tail), `exposure_increment` (verbatim port of basecurve.c:565-569:
offset = stops·fusion·(bias−1)/2), `apply_curve_pixels` (dispatches legacy
non-chroma-preserving vs apply_curve kernel with preserve_colors + luminance
Y-row), and `process_fusion` — the full Laplacian-pyramid exposure-blend
orchestration over the Phase 2z+56 kernels (compute_features → gauss_reduce →
weight_update → pyramid blend coarse→fine → normalize/add_layers → copy_output),
with `level_dims`/`gauss_reduce` helpers. C reads only channel 0 (`const int ch =
0`) so a single LUT suffices.

**Working-space luminance**: C passes `work_profile->matrix_in` to apply_curve in
BOTH paths (incl. fusion, basecurve.c:1127) but the kernel consumes only row 1
(the Y row). Since Bradford chromatic adaptation preserves the Y row exactly,
`sRGB_TO_XYZ_D65_Y_ROW` / `REC2020_TO_XYZ_D65_Y_ROW` constants were added to
color.rs and threaded as `Option<[f32;3]>` instead of plumbing ICC profiles. The
T4 matrices in color.rs are transposed (column-major), so passing them directly
would have handed the kernel the G *column* — hence the row-extraction API.
Camera-primaries fallback (None) uses dt_camera_rgb_luminance coefficients.

Pipeline: `Stage::Basecurve { table, coeffs, preserve_colors, fusion, stops,
bias, space }` at iop_order v50_order pos 44.0 ("conversion from scene-referred
to display referred"), between rgblevels 43.0 and sigmoid 45.3 — emitted after
RGB curve, before sigmoid/shadhi (pinned by the ordering test with positional
asserts). Pixel-locality gate `fusion == 0`: fusion>0 routes to the serial
whole-buffer path (pyramid reads neighbours); plain-curve mode rides the rayon
band path like RgbCurve.

UI: params v24→v25 (+1 bool/+6 f32/+40 interleaved node coords, blob
1603→1788), append-only layout keeps all prior offsets stable. HISTORY_ENCODE_VERSION
5→6 (old v5 stacks still load via the accepts-version-1 rule; their embedded v24
params resolve through PARAMS_LAYOUTS). `basecurve_module_row` joins
`multi_curve_area` as its third client (single neutral-grey channel): preserve-
colors combo, exposure-fusion combo (none/two/three), and the two fusion sliders
built via labeled_slider directly because their rows must toggle visibility with
fusion≠0 (C gui_init :1966/:1976 + gui_update :1323-24). Two documented GUI
deviations: no interpolator dropdown (C's gui_init has none for this module —
bc_type still drives the drawn spline so decoded blobs render correctly), and
log-base graph scaling omitted (display-only).

Senior review: **APPROVE-WITH-NITS** — fricktrade-architect failed again (API
402), fork subagent stood in per established fallback, explicitly constrained to
read-only after the m4-123 incident. It independently confirmed all seven C
claims against source (iop_order placement, ch=0, exposure_increment formula,
fusion work-profile pass, GUI widget set, $DEFAULTs, Y-row/Bradford soundness)
and hand-recomputed the fusion grey test expectation (≈0.6726 vs my ≈0.67
comment). Findings: MINOR doc slip "v4" for "v5" history blobs (fixed); two NITs
recorded not fixed — corrupt-blob stops/bias reach the kernel unclamped (same
posture as every sibling module; NaN cannot panic), and the endpoint-pinning
closure is now duplicated between RgbCurve/Basecurve emission blocks (a third
curve module would justify hoisting it).

Verification: full `scripts/ci-local.sh` green **by exit code** (0): check +
clippy (zero new warnings from the m4-124 layer — all flagged lines in touched
files are pre-existing classes verified line-by-line) + tests under --release
(c41-core 768, c41-db 92, c41-ui 269, incl. new build_table/exposure_increment/
Y-row/fusion tests ×18 in core and ordering/v25-roundtrip/v24-fallback/e2e
darkening tests in ui) + release bin link. En route self-corrections: a stray
thinking-text paste into basecurve.rs and an inverted split_at_mut were caught
and fixed pre-review; the pipe-exit-code trap nearly bit again on an early
Docker test run (`| tail` swallows cargo's status) and was re-run redirect-first.

Process notes: PARITY_AUDIT row 2.1 updated in the same commit (31→32 live;
only lens correction remains missing, lensfun-blocked). NIT backlog: shared
endpoint-pinning helper when the next curve module lands.

## 2026-08-24T23:20Z — m4-125: lighttable history stack operations (parity row 2.2)

darktable's "actions on selection → history" group lands in C-41's lighttable as a
**History** section on the metadata panel: Copy / Paste / Discard plus a one-line
readout. Copy grabs the selected image's saved edit (`darkroom_preview` params blob
+ `darkroom_history` undo-stack); Paste REPLACES both target rows (dt paste
semantics) and toasts "Pasted edit from <basename>"; Discard clears both rows
behind an adw::AlertDialog confirm (deliberate deviation: dt acts immediately, but
its actions target a multi-image keyboard-driven selection while our grid is
SingleSelection). New `persist::discard_history` deletes both rows for an imgid in
one `unchecked_transaction` (DDL inside the tx keeps it safe on legacy dbs where
neither private table exists yet) so params can never point at a discarded stack.
Readout is a pure headless-testable function: "(no image selected)" /
"no saved edits" / "N-step edit stack" (a params row without a stack row counts as
1 step — it still changes the render vs raw defaults).

Review findings all fixed (fork subagent standing in as senior reviewer after the
fricktrade-architect 402 again; APPROVE-WITH-NITS): (1) partial-paste coherence —
a copy with params but NO stack row (pre-history source, or decode failure) now
seeds a fresh one-entry stack from the pasted params instead of leaving the
target's stale stack describing edits that no longer exist (+regression test);
(2) post-action sensitivity staleness — Copy/Discard re-derive their gating at the
tail of paste/discard instead of waiting for the next selection change;
(3) false-success toasts when no catalogue is open — paste/discard/copy now say
"No catalogue open" like the styles save button. NIT backlog accepted:
history_readout_text opens two SQLite connections per call (consistent-with-
neighbourhood cost, query_exif does the same); step count includes the "Original"
seed (display-only, pinned by test); clipboard field now typed via the
HistoryClipboardHandle alias.

Verification: c41-ui release tests green by EXIT CODE (274 passed, incl. new
discard-clears-both / discard-noop-for-uncatalogued / readout-transitions /
clipboard-roundtrip-between-images / params-only-paste-replaces-stale-stack).
First Docker run failed compile (EXIT=101) — my persist tests referenced helpers
from a *different* test module (catalogued_db lives in metadata_tests,
sample_history in tests) and three panels closures moved the shared ctx Rc /
dropped a live borrow; fixed with per-closure ctx clones and a statement-scoped
borrow that lifts the basename to owned String. En route self-correction: the
first discard test draft also had a redundant table-wipe (tmp_db already starts
clean). Full scripts/ci-local.sh gate then ran green by exit code.
PARITY_AUDIT row 2.2 closed in this commit.

## 2026-08-25T00:15Z — m4-126: colour-label quick filters in the top + bottom bars (parity row 2.5)

darktable has colour circles in BOTH lighttable bars; we only had the left panel's
checks — and those were a *collection selector* (GROUP BY/HAVING query) that
competed with folder/tag/search and was mutually exclusive with them (clicking a
folder wiped the checks and vice versa). m4-126 pays the audit's "reconcile first"
debt by DELETING that competing selector and converting colour into a compose-on-top
quick filter exactly like stars (m4-98d) and year range (m4-99):

- Canonical thread-local state `COLOUR_MASK` (5 bits) / `COLOUR_ALL` in
  lighttable/mod.rs; `set_colour_filter(mask, match_all)` fans out through the
  filter-observer bus. `current_filters_sql()` now splices rating → year → colour,
  so every view loader gets it for free; fragment is
  `AND ((SELECT COUNT(DISTINCT cl.color) FROM main.color_labels cl WHERE cl.imgid =
  i.id AND cl.color IN (...)) >= 1)` for ANY, `= N` for ALL.
- Three mirrors of one state: left-panel checks + Any/All toggle (panels/mod.rs),
  top-bar circles after the preset dropdown, bottom-bar circles right of the star
  box (both from `colour_circles_row()`, same Pango dot glyphs as grid cells).
  Handlers consult `filter_sync_in_progress()`; observers only repaint. Folder/tag
  clicks and search/import no longer clear colours — the AND genuinely runs.
- Persistence: token `off` / `any:M` / `all:M` under pref key `colour_filter`,
  written by ONE app-level observer in lib.rs (not per-widget — two bar rows would
  otherwise double-write), restored at startup beside the rating token BEFORE any
  control builds. Demo db can't restore (empty prefs path) but now carries an empty
  `color_labels` table anyway.
- Deleted: `build_color_mask_query`, `lighttable_load_by_color_mask`,
  `clear_color_checks`, the left panel's color_suppress plumbing.

Review findings all fixed (fork subagent standing in as senior reviewer after the
fricktrade-architect 402 again; APPROVE-WITH-NITS): MAJOR-1 demo-db blank-grid — a
live circle click in demo mode spliced the colour fragment against a db with no
color_labels table, so every subsequent load failed its query into an empty grid;
fixed by adding the table to open_demo_db's DDL (one line). MINOR-2 duplicate
persist writes → single app-level observer (colour_circles_row lost its db_path
parameter entirely). MINOR-3 stale "post-import shows all images" comments →
"shows the quick-filtered collection". NIT-4 unified child-indexing strategy
(widget-name parsing on both read and write paths of the left-panel checks).
NIT-5 accessible labels on the circle buttons. NIT-6 composition test extended to
pin the exact three-way rating+year+colour string.

The gates caught two real bugs en route, both kept here as the record: (1) the
first predicate draft emitted `>= N` in ANY mode — requiring EVERY selected colour
under OR semantics, contradicting the function's own doc contract; my new unit
test failed and the fix is `(op, bound) = match_all ? ("=", n) : (">=", 1)` — the
test did its job. (2) My three-way exact-string assertion forgot to re-arm the
rating preset after an earlier clear in the same test (left != right made it
obvious); fixed the test, not the code. Also one E0308 (accessible Label wants
&str not String) and one E0433/E0425 import round (panels needed `self` in the
lighttable use; test module had lost current_filters_sql).

Verification: full scripts/ci-local.sh green BY EXIT CODE (CILEXIT=0): cargo check
workspace, clippy workspace, tests --release (c41-core 768, c41-db 92, c41-ui 276
— including the end-to-end SQLite colour-semantics test: ANY{red} keeps both
red-labelled images, ALL{red,green} keeps only the both-labelled one, empty mask
keeps everything), and the release bin link. PARITY_AUDIT row 2.5 closed in this
commit.

## 2026-08-25T01:15Z — m4-127: ICC B2A direction + full Transform assembly

`c41-core::icc` completes the colour-transform engine: the PCS→device direction
and the two-profile assembly that chains them.

- `Curve::inverse()` (parser.rs): analytic `Gamma(1/g)` (degenerate g==0 →
  Identity); Table/Parametric invert by sampling + bisection (N=4096, per-target
  24-step bisection over [0,1]) following LCMS's tabular-inversion convention —
  entry j = smallest x with fwd(x) ≥ j/(N−1), so bottom plateaus map covered
  targets onto 0.
- `Profile::b2a_pipeline(intent)` (parser.rs): prefers `B2A{intent}` LUT tags
  with the same A→B fallback scheme (abs intent 3 → B2A1, then B2A0), PREPENDING
  `pcs_encode_stage` — the exact algebraic inverse of the a2b decode line per
  version/PCS (v4 Lab: L/100, ab/255+128/255; v2's legacy 65535/65280 scale;
  XYZ ×32768/65535) so tables consume encoded [0,1] like they were authored for.
  Matrix-shaper fallback consumes RAW XYZ D50 with NO encode stage: adjugate
  inverse colorants + inverted TRCs. Review finding: singular/non-finite det now
  returns Err(WrongTagType) at assembly instead of silently substituting identity
  (garbage-in-garbage-out became loud-fail; runs once per transform so it costs
  nothing).
- NEW `icc/transform.rs`: `Transform::new(src, dst, intent)` = head
  (src.a2b_pipeline) → optional absolute-intent media-white ratio scaling in XYZ
  (`wtpt_dst/wtpt_src` componentwise, BOTH whites now required all-positive after
  review — profiles are untrusted bytes and a zero/negative component would
  zero/flip a channel; missing/unusable wtpt degrades silently to relative,
  documented) → Lab↔XYZ bridge → tail (dst.b2a_pipeline). The bridge is three
  private fields (to_xyz / abs_scale / from_xyz), not an enum: Lab-src/Lab-dst
  absolute needs convert→scale→convert-back, which a single Bridge variant could
  not express. D50-referenced lab_from_xyz/xyz_from_lab helpers (ε=216/24389,
  κ=24389/27). Exported as `icc::Transform`; mod.rs increment history corrected
  (the earlier engine increments were m4-89..93b per git, not m4-90/91 as the doc
  claimed).

Numbering note: planned as "m4-91" from the stale mod.rs doc, but git history
shows m4-89..93b were consumed by the parser/clut-core/tag-parsing/a2b work on
07-20; this increment is m4-127, continuing from m4-126.

Tests (11 in transform.rs, 779 c41-core total, --release): Lab↔XYZ roundtrip vs
f64-tight tolerances incl. linear-leg; gamma/table/parametric inverse roundtrips;
diagonal AND full-matrix profile self-transforms; both bridge directions
end-to-end (XYZ→Lab dst through v4 encode onto an identity mft1 CLUT; Lab src
through v4 decode into a matrix-shaper B2A); abs ≡ rel on equal whites, abs ≠
rel on different whites (with the gotcha documented IN the test: axis-aligned
primaries make the white-ratio scale cancel exactly against the inverse
colorants — the first draft was vacuous), wtpt-less degradation; v2 legacy-scale
encode pinned against v4 (the new tight tolerance exposed that the synthetic
curv tag quantises γ2.2 to u16 563/256 — expectation fixed, not code); B2A1-
preferred-over-B2A0 for intent 1 with a flat-zero B2A0 probe, intent 0 landing
on B2A0; singular colorants → Err.

Review: APPROVE-WITH-FIXES (fork subagent standing in for fricktrade-architect,
API 402 again; read-only mandate enforced). Findings fixed: MINOR dst-wtpt
positivity guard; MINOR invert3 fail-loudly (+ test); MINOR test gap — v2
encode/decode coverage + B2A tag-preference tests added. NITs also applied:
hoisted eval(0)/eval(1) out of the N-loop; N 1024→4096. Forward note accepted
for the FFI follow-up: Transform::eval allocates per pixel-vector; band
processing wants a write-into-scratch variant when colorin/colorout get wired.
Clippy: one needless_range_loop in my own new loop fixed (zero new diagnostics
from icc files; the two erasing_op errors under --all-targets are pre-existing
test-target lints in committed bloom/rawimage code the CI clippy invocation
never compiles). Full scripts/ci-local.sh GATE_EXIT=0 (check, clippy, test
--release, release link). En-route failures kept for the record: mft1 helper
missing byte-11 alignment pad before the e-matrix (everything shifted one byte
→ Truncated); first Transform draft computed abs_scale but never applied it
(caught by rewriting eval before ever compiling — the enum-Bridge special case
for abs+Lab/Lab would ALSO have fed XYZ to a Lab-expecting tail; the field-based
design replaced it).

## 2026-08-25T02:25Z — m4-128: collection filters, first slice (aspect-ratio rule)

First slice of darktable's "collection filters" expander (parity row 2.6,
upstream src/libs/filtering.c): a collapsible "Collection filters" section in
the left panel (between Colours and Tags, fold state persisted under
`left_section_filters`) with one gtk4::DropDown carrying darktable's three
stock aspect presets — square `[1;1]`, landscape `>=1.01`, portrait `<=0.99`
(filtering.c:308-322). It drives the canonical quick-filter state through the
same observer bus as rating/year/colour, so it composes ON TOP of whatever
collection is active and re-renders the current view once per change.

Design: darktable filters a stored REAL column `main.images.aspect_ratio`
(snapped by dt_usable_aspect, image.c:1140); our schema has only import-time
width/height (probe_dims), so `aspect_predicate` expresses the same intent as
exact integer comparisons — `i.width > i.height` / `i.height > i.width` /
`i.width = i.height AND i.width > 0`. The `w > 0` guard keeps the failed-probe
row (0×0) from matching square; NULL dims match nothing under every arm.
Splice order in current_filters_sql() is now rating → year → colour → aspect;
empty fragment for Off preserves the nothing-spliced invariant. Token codec is
bare words (`off|landscape|portrait|square`) under pref key `aspect_filter`;
lib.rs restores it beside the colour token before any control builds and runs
one app-level persist observer per change. AspectFilter mirrors the
FilterPreset shape (ALL/from_index/to_index/label) so the dropdown can't drift.

Documented deviations: exact integer comparison vs upstream's ±0.5 %-tolerance
snapped float (a true 1.004 ratio stores as 1.0 there and its square preset
catches it; we class it landscape — divergence confined to that boundary band),
and sensor dimensions vs developed p_width/p_height (no materialised post-crop
ratio column yet).

Tests (5 new in lighttable/mod.rs, 281 c41-ui total): exact fragment strings
per variant; token roundtrip + corruption fallbacks + index mapping; startup
seeding contract (apply writes state without reload); composed-SQL pin with
colour+rating proving splice order and clean single-filter drop; end-to-end
SQLite run seeding {landscape, portrait, square, 0×0, NULL} rows asserting
survivors per preset (the 0×0-must-not-be-square guard pinned against real
engine semantics, not just string shapes).

Review: APPROVE-WITH-FIXES (fork subagent standing in for fricktrade-architect
— API 402 again, credits exhausted mid-launch; read-only mandate enforced).
All findings fixed: stray duplicated doc-comment line over set_year_range;
"exactly" overclaim in aspect_predicate doc replaced with both deviation
sentences; missing executed-SQL test added (the end-to-end above); encoder
signature aligned with siblings (String, not &'static str) + doc reworded.
En-route failure kept for the record: the first ci-local attempt ran the
script INSIDE the container (no docker there → exit 127), then three host-side
retries all died on the output redirect because the in-container run had left
target/m4128-cilocal.log root-owned ("Permesso negato" on truncate) — the
"docker not found" line I kept reading was stale content from the container
run, not a live diagnosis; rm + rerun fixed it. Full scripts/ci-local.sh
GATE_EXIT=0 (check, clippy, test --release, release link).

- **2026-08-25T03:05Z — m4-129 slice 1: ICC engine C boundary + allocation-free band path.** The Rust-side enabler for replacing colorin/colorout's LCMS LUT fallback: `Pipeline::eval_into3` / `Transform::eval_into` evaluate a whole transform on stack `[f32;3]`s (no per-stage Vec, bit-exact to the allocating `eval`s — pinned by test), and `icc/ffi.rs` exports `darkroom_icc_transform_new/free/apply_rgba`: profile *bytes* in → owned handle; stride-4 RGBA f32 rows with alpha passthrough, in-place-safe (RGB triplet copied before write) mirroring colorin's `cmsDoTransform(xform, out, out, width)` callsites. NULL-on-failure contract so C falls back exactly as it did for a failed `cmsCreateTransform`. C call-site wiring is deliberately NOT here — it needs the full-app Docker C build and lands as slice 2.

  Review: APPROVE-WITH-FIXES (fork subagent standing in for fricktrade-architect — API 402 again; read-only mandate enforced). The MAJOR finding was real and worth the review: the 3-channel invariant `eval_into3` rests on was asserted but never enforced — the parser caps channel counts at 16 rather than pinning 3, so a GRAY profile's A2B0 assembled fine and would have panicked across FFI (`curves[1]` out of bounds) on the first pixel, while CMYK would silently drop K. Fixed by enforcing the invariant in `Transform::new` (every Curves stage len==3, every Clut 3-in/3-out, else Err → NULL at the boundary), correcting the now-false lut.rs comment that claimed "parser rejects channel-count mismatches", and adding rejection tests both sides of the boundary (engine: GRAY fixture refused with WrongTagType as source AND destination; FFI: GRAY bytes → NULL handle). Fixture gotcha kept for the record: my first gray-lut helper omitted lut8's mandatory 36-byte e-matrix slot, so tables sat misaligned and the parse failed Truncated instead of exercising assembly — the test then asserted the wrong error variant and caught it. Also applied: intent>3 refused with NULL (mirrors cmsCreateTransform failing); apply_rgba docs pin the raw-Lab/raw-XYZ buffer domain (LCMS TYPE_LabA_FLT/XYZA_FLT equivalent, so slice-2 wiring maps formats 1:1) and the concurrent-use guarantee; eval_into scoped pub(crate).

  Verified: 41 icc tests green; full scripts/ci-local.sh GATE_EXIT=0 (check, clippy, test --release, release link). PARITY_AUDIT.md unchanged this commit — the parity item resolves only when the C call sites switch (slice 2).

- **2026-08-25T05:30Z — m4-129 slice 2: colorin/colorout call the pure-Rust ICC engine; full-c CMake build repaired.** The LCMS LUT fallback paths now prefer `darkroom_icc_transform_new/_apply_rgba`: colorin/colorout serialize their profiles once per commit_params (double `cmsSaveProfileToMem` into g_malloc'd buffers — builtin profiles are LCMS-API handles with no embedded bytes, hence serialization) and every former `cmsCreateTransform` site tries the engine first, falling back per-handle to the exact previous LCMS call whenever it returns NULL; all six cmsDoTransform callsites route through an `_icc_apply(rs, &lcms, …)` helper. Softproof/BPC/gamutcheck *and* the explicit `force_lcms2` export preference stay LCMS-only by construction (`rs_xform` built only when NORMAL && !force_lcms2). Engine enablers: colour-space-conversion profiles (`data_space == pcs`, darktable links both transforms through a bare Lab handle) assemble identity pipelines — spec-correct and pinned by exact-equality passthrough + Lab→device white tests; device spaces outside {RGB, XYZ, ==pcs} are refused at assembly so a tag-complete nonsense profile can't grade through the XYZ shaper path as plausible garbage (review MINOR; NULL → C falls back).

  Review: APPROVE-WITH-FIXES (fork subagent standing in for fricktrade-architect — API 402 again; read-only mandate enforced, zero files modified). Both MAJORs were real: (1) colorin's `init_pipe` is malloc-not-calloc with five initializers — my nine new fields stayed garbage and the unconditional `_colorin_rs_cleanup(d)` at the top of commit_params would have wild-freed them on the very first pipe build; fixed by initializing all nine (colorout was safe via calloc). (2) `force_lcms2` was silently overridden in NORMAL mode because the rs build gated only on mode — the pref exists precisely to pin export to LCMS; gated at both build sites. MINOR (odd-signature refusal) also applied with regression test + FFI doc update.

  Found en route and fixed in the same commit: **the 2026-08-12 rename broke the full-c CMake build outright** — `src/CMakeLists.txt` still pointed at `crates/darkroom-core` (75 DEPENDS paths), `-p darkroom-core`, and `libdarkroom_core.so` (OUTPUT/SONAME rustflag/IMPORTED_SONAME); CI's docker-full-c job is workflow_dispatch-only so nobody hit it since the rename. Repaired mechanically to c41-core/libc41_core.so; internal cmake target names renamed too. Also refreshed the stale pre-rename comments at the top of darkroom_core.h.

  Verification story worth rereading: no local :full-c image exists and Dockerfile.full-c clones from the remote, so uncommitted changes are invisible to it — built a persistent replica of its builder stage (`target/cbuild/Dockerfile.fullc-deps` → image c41-fullc-deps), initialized the seven submodules depth-1 locally, and ran the exact cmake configure+build against the mounted tree (persistent build dir under target/cbuild/build for incremental reruns). Pre-fix tree compiled+linked clean (855/855, exit 0). Post-fix run: the harness killed both gate wrapper shells mid-flight but the docker containers survived; `docker wait` then reported container exit 1 against a log showing all 855 edges done and zero FAILED/error lines — resolved by an idempotent `cmake --build` rerun in a fresh container over the same build dir: only 6 JSON-validity edges left, **REBUILD_EXIT=0**, so the exit-1 was kill/cleanup fallout, not a compile failure. ci-local first attempt likewise finished all four steps green in its log but lost GATE_EXIT with its wrapper → rerun fresh: **GATE_EXIT=0** (check/clippy/test --release/release link, 9 test-result-ok suites). PARITY_AUDIT.md unchanged — the ICC/LCMS tracking lives in RUST_MIGRATION_PLAN.md (row "Per-pixel ICC / LCMS"), which this commit updates to slice-2-done.

- **2026-08-25T04:23Z — m4-130 slice 1: liblensfun FFI bindings + lens-database plumbing (lens-correction groundwork).** The lens-correction parity item has been "blocked on the lensfun DB" for months — the blocker dissolves on inspection: the *math* was never missing, only the calibrated data files in the cargo-only runtime image. Route decision (documented against project principles): **FFI to the distro liblensfun**, not a pure-Rust port — m4-86's no-lcms2 decision was about cargo-native *build* (no C toolchain), not zero C libraries (GTK4 itself is one); linking a distro `.so` needs no compiler and no CMake, and lensfun IS the calibrated reference engine so parity is free by construction. A pure-Rust swap can come later behind the same Rust API. Landed: hand-written `c41-sys::lensfun` bindings (deliberately OUTSIDE generated `bindings.rs`, which bindgen regenerates from darktable.h and knows nothing of lensfun — a regen would silently drop them) for the database/camera/lens slice: `lf_db_new/load/destroy/find_cameras/find_lenses_hd/free`, opaque db/modifier handles, prefix-mirror structs lfCamera/lfLens (calibration-list tail deliberately undeclared until slice 2 consumes it; C++ methods occupy no storage). Build wiring: `links = "lensfun"` + `cargo:rustc-link-lib=dylib=lensfun` in the existing build.rs — dev image, CI runner, product builder get `liblensfun-dev`; runtime image gets `liblensfun1 liblensfun-data-v1`. Display-free probe test (`tests/lensfun.rs`) loads the system db, resolves "Canon EOS 5D Mark II" deterministically, checks crop factor + best-score lens identity; skips when `/usr/share/lensfun` is absent (bare host checkouts).

  Two self-caught bugs worth keeping visible: (1) I first invented `LF_SEARCH_SORT_AND_UNIQUIFY_ECS = 1` from memory — lensfun 0.3.4's enum is `LF_SEARCH_LOOSE = 1, LF_SEARCH_SORT_AND_UNIQUIFY = 2` (darktable's lens.cc #defines it as 2 itself); value 1 would have silently run LOOSE matching. Caught by grepping the real header before commit. (2) Same header check exposed that my `LF_ERROR = -1` constant matched nothing — in 0.3.4 error codes are POSITIVE enum values (`LF_WRONG_FORMAT = 1`, `LF_NO_DATABASE = 2`; negatives are `-errno`). Struct prefixes and all six extern signatures were then verified field-by-field against `/usr/include/lensfun/lensfun.h`.

  Review: APPROVE-WITH-FIXES (fork subagent standing in for fricktrade-architect — API 402 again; read-only mandate enforced, "Files modified by this review: none"). MAJOR finding was real and CI-fatal: GitHub Actions passes `run:` scripts verbatim to bash, which does NOT strip `#` lines inside backslash continuations (unlike Docker RUN layers) — my comment line inside the apt-get continuation made bash end the command there and execute `liblensfun-dev liblensfun-data-v1` as a *command* → exit 127 on every push; reviewer reproduced with a stubbed sudo. Fixed by moving the comment above the step (with an explanatory note left in place), YAML parse-checked. Also applied: ownership-doc contradiction fixed (returned lists are caller-freed arrays via `lf_free`, elements stay database-owned — "owned by the database" described neither); `lf_db_load` doc corrected to its documented `LF_NO_DATABASE` empty-db return; probe skip-guard documented (lensfun also searches /usr/local/share + XDG dirs; images install to /usr/share); builder stage no longer installs the data package it never reads (link-time `.so` only). En-route fix: `std::os::c_char` → `std::ffi::c_char` import.

  Verified: probe test green in rebuilt dev image (IMG_EXIT=0 after the package-name fix; PROBE_EXIT=0); c41-sys clippy+tests green post-fixes; full scripts/ci-local.sh **GATE_EXIT=0** (check, clippy, test --release, release link). PARITY_AUDIT.md row 2.1 updated in this commit: lens correction unblocked (bindings + data plumbing landed; pipeline stage + UI pending) — the item fully closes at slice 3.

- **2026-08-25T06:20Z — m4-130 slice 2: `Stage::LensCorrection` — the lensfun warp lands in the pipeline.** New `c41-core::iop::lens` (~600 lines) mirrors `src/iop/lens.cc`'s LENSFUN method at whole-frame scope: camera/lens resolve from the global db (OnceLock + Mutex), RAII `Modifier`, `_modflags_to_lensfun_mods` mapping (GEOMETRY|SCALE always on, matching the legacy-Initialize path that noble's 0.3.4 requires — no granular 0.3.95 API), and `_process_lf`'s direction split: forward applies vignetting on a scratch copy BEFORE the coordinate warp, inverse warps first then re-applies the falloff on the output. The warp resamples R/G/B each at its own distorted source coordinate via `lf_modifier_apply_subpixel_geometry_distortion` (per-row 6-float coords, clamp-to-[0,dim−1], bilinear only) through new `interp::compute_sample_strided` (single-channel sample of an interleaved buffer with per-channel base offset = C's `buf + c` pointer arithmetic; cross-validated against `compute_sample_1c` across all four kernels). `nan_checks` extracted as a pure fn of (target_geom, lens_type) mirroring commit_params' rule. Alpha carried through untouched (documented, safer-than-C deviation); embedded-metadata methods / custom TCA override / monochrome TCA suppression documented as out of scope. Pipeline: `Stage::LensCorrection { lens: ResolvedLens, params: LensParams }` appended after DenoiseProfile, name "lens", `is_pixel_local => false` (coordinate warp → whole-frame serial band). Five tests, db-dependent ones skip-guarded on /usr/share/lensfun.

  Two test assertions had to be corrected by MEASUREMENT before they encoded real behaviour rather than my assumptions — worth rereading next time a "physical invariant" test fails: (1) I asserted the centre pixel maps to itself exactly; it moved ~0.007. Probe showed vignetting is EXACTLY neutral at centre (Δ=0.0) while the pure warp moves it — lensfun places its radial origin at (width/2, height/2), so even-sized frames put the centre pixel half a pixel off-axis (plus calibration decentering). Now asserted as ~20× separation (centre <0.02 vs corner >0.05, measured 0.007 vs 0.15). (2) I asserted forward vignetting DARKENS corners; it BRIGHTENS them ~5× — the forward direction is *correction* (multiply by 1/falloff; 24mm f/2.8 wide-open loses >2 stops there). Both directions now asserted (forward brightens, inverse darkens).

  Review: APPROVE-WITH-FIXES (fork subagent standing in for fricktrade-architect — API 402 again; explicit read-only mandate enforced, zero files modified). MAJOR applied: modifier CONSTRUCTION now holds DB_LOCK in both `process` and `autoscale` — upstream holds its plugin mutex across `_get_modifier` because FindLensesHD writes Score into every candidate lens object, so a UI lookup racing a render could tear calibration lists mid-initialize; heavy per-row Apply calls stay unlocked (const/shared upstream too). MINORs applied: autoscale forces scale=1.0 into Initialize (`_get_autoscale_lf` builds dummy params that way so repeated presses don't compound through the current scale); crop ≤ 0 falls back to plain copy like C (lens.cc:1074) instead of feeding inf coordinates past the suppressed nan-check. NITs applied: `lf_modifier_get_auto_scale` binding const→mut to match lensfun.h; scratch `to_vec` now allocated only on the one branch that needs it (forward+vignetting); `compute_sample_1c` delegates to the strided sampler (one kernel loop; equivalence pinned by the existing test); empty-lens-name wildcard hazard documented + debug_asserted ahead of slice 3 wiring the UI to `resolve()`.

  Verified: all 5 lens tests + 6 interp tests green under --release; full scripts/ci-local.sh **GATE_EXIT=0** (check, clippy, test --release, release link). Note for the record: `cargo clippy --all-targets` locally reports 3 pre-existing deny-level lints in TEST code of untouched files (bloom.rs/rawimage.rs `identity_op`/`erasing_op`) — CI and ci-local.sh run clippy WITHOUT --all-targets so these never gate, but they will bite whoever enables test-target lints. PARITY_AUDIT.md row 2.1 updated in this commit (stage landed; only module UI pending).
- **2026-08-25T06:46Z — m4-130 slice 3: lens correction darkroom module UI; parity item 2.1 CLOSED (33 live modules).** The last unwired ported module gets its panel: camera DropDown (mount-filtered `lf_db_get_cameras`, sorted, maker-prepended labels), dependent lens DropDown (`list_lenses(mount)`), corrections bitmask combo (none/all/three pairs/three singles), target-geometry combo (rectilinear..Thoby), focal/aperture/distance sliders, auto-scale button + scale slider. Two design pivots worth keeping: (1) **exact identity resolution** — C probes against this distro DB showed `lf_db_find_lenses_hd` is a lossy did-you-mean scorer: it returns NOTHING for ~half of real entries fed their exact maker+model (it scores the Model field only, so prepending the maker breaks it) and even raw exact pairs hit exactly only 142/292 times. All persisted-pick resolution is therefore byte-equality enumeration over `lf_db_get_lenses` (~1304 entries) with mount filter + closest-crop-factor tie-break — duplicate Maker/Model rows exist across mount families and first-match-return broke "Sigma | Sigma 18-50mm f/2.8 EX DC". Persistence stores structured `(camera_maker, camera_model, lens_maker, lens_model)` in `main.darkroom_lens_choice` exactly as the DB spells them; labels round-trip through `str_of`, which reads the raw default-language lfMLstr pointer (`lf_mlstr_get` would locale-dependently unresolve every pick). darktable avoids the whole problem by hanging live `lfLens*` pointers off its menu items. (2) **blob v26**: numeric params join PreviewParams (1788 → 1814 bytes; history encode-length pin updated, HISTORY_ENCODE_VERSION stays 6); gear identity stays in SQLite, not in the blob. Corrections/geometry dropdowns write value tables built FROM the core constants — construction pins index→value, because the first draft wrote the combo INDEX into the bitmask (selecting "all" stored TCA-only; seeding value 7 highlighted "only vignetting"). Auto-scale prefers pristine dims and takes the single-commit path through the scale slider.

  Review: APPROVE-WITH-FIXES — run by a fork subagent on the session model (**stealth/ox-alpha**, per user direction that the senior reviewer run on ox-alpha) standing in for the named fricktrade-architect agent, which failed API-402 again; explicit read-only mandate enforced. Findings, all fixed: BLOCKER index-vs-value bitmask above; MAJOR history.rs comment falsely claimed gear changes are "change-detected separately by the module row" — HistoryStack dedups on blob params only, so a gear swap records no entry; comment replaced with honest documentation of the gap; MAJOR nested-notify multi-commit (`set_model` autoselects position 0 → lens handler fires synchronously → up to 3 persists incl. an arbitrary lenses[0]) fixed with a `lens_syncing: Rc<Cell<bool>>` guard around every programmatic fill (reviewer verified no RefCell panic risk); MAJOR seed could overwrite a persisted lens pick with empty when the persisted camera vanished from the current DB — seed now gated on persisted-choice emptiness so saved picks always win; citation fabrication called out — the claimed "C defaults 50/3.5/10" introspection does not exist; distance is now documented as 1000.0 (C's actual no-EXIF fallback, lens.cc:3455) and field docs rewritten honestly. Accepted nits: focal slider unclamped to the resolved lens range (dt forces primes to MinFocal; deferred), populate-runs-on-every-undo/redo cost (OnceLock camera-list cache candidate).

  Verified: release test suite green in the dev container before the gate; full scripts/ci-local.sh GATE_EXIT=0 (check / clippy / test --release / release link). The dev image carries /usr/share/lensfun/version_1, so the db-backed tests genuinely ran. PARITY_AUDIT.md row 2.1 converted to Fixed form in this same commit. Known limitations recorded in the module-row doc: stage measures the post-crop/straighten frame vs darktable's iop_order-13 pre-crop position (pristine-buffer pre-pass = follow-up; meanwhile auto-scale prefers pristine dims); undo/redo reverts numbers, not gear swaps (sidecar-in-history follow-up); no EXIF auto-population (C41 reads no EXIF lens tags — the choice is user-picked).
- **2026-08-25T07:32Z — m4-131: lens-correction pre-pass on the full frame, before crop/straighten.** Closes limitation (1) recorded with m4-130 slice 3: because C41 applies crop/straighten as a separate geometry pass BEFORE the pixel pipeline, the lens stage emitted inside `to_pipeline` measured distortion/TCA/vignetting against the POST-crop frame — wrong warp centre and vignetting referenced to crop edges whenever the image was cropped, where darktable (iop_order v50 pos 13 < crop) corrects the whole sensor frame. Now `preview::apply_lens_prepass` warps the freshly decoded linear buffer on the FULL frame; both raw export funnels call it between decode and `geometry.apply`; the preview caches the warped pristine in `PreviewCtx::lens_frame` (keyed by decode generation + enable flag + gear Arc identity + numeric LensParams) and `apply_geometry_to_base` composes `base` from it; `render_preview` opens with `refresh_lens_frame(ctx)` — every lens-affecting trigger funnels through that one function (sliders via add_param_slider, combos, gear commits through lens_commit_choice, undo/redo/Reset through apply_history_params, the bypass peek through effective_params), so no call site knows the warp exists and an input change recomposes base exactly once before painting (depth-2 re-entry, no borrows held across the warp). The raw pipeline builder variant `to_pipeline_lens_preapplied` drops exactly the lens stage; non-raw funnels keep the in-pipeline stage because their 8-bit sources are never cropped — there a full-frame in-pipeline warp IS darktable's placement. Side benefits: the warp no longer runs on every render of unrelated slider drags (it is cached until a lens input changes), and the before/after peek shows a truly uncorrected original (bypass ⇒ effective params disable the module ⇒ cached frame dropped ⇒ compose from pristine).

  Drive-by: running `cargo clippy --workspace --all-targets` (stricter than CI's invocation, which does not lint test targets) surfaced two deny-by-default erasing_op errors in c41-core test code from earlier increments — bloom.rs corner index `(0*w+0)*4` and rawimage.rs orientation assertion `(0*3+2)*4`; rewritten with named coordinates without changing what they assert. scripts/ci-local.sh's clippy step gained `--all-targets` so test targets are linted from now on.

  Review: SHIP (fork subagent on stealth/ox-alpha standing in for fricktrade-architect — API-402 again; read-only mandate honored; the session restart killed its first run mid-review and the agent was resumed from its transcript to deliver). Zero BLOCKER/MAJOR findings; warp placement, cache protocol, bypass semantics, borrow discipline and commutation-test soundness all independently verified. Applied: NIT — apply_lens_prepass zero-init allocates instead of cloning input into output first (`process()` provably overwrites every lane of every pixel in all branches); NIT — LensFrame::matches doc comment pins that the params self-compare fallback at the call site is deliberately inert. MINOR accepted: denoiseprofile (pos 9/10) now runs after the warp instead of before it — spatial denoise and a coordinate warp commute to first order; documented in PARITY_AUDIT.md row 2.1 rather than moving wavelets into the pre-pass (would tax every preview). MINOR acted on: the ci-local.sh clippy gap above.

  Verified: new tests `to_pipeline_lens_preapplied_omits_only_the_lens_stage` and `lens_correction_runs_precrop_and_commutes_with_crop` (cropped export == same region of uncropped export BIT-for-BIT across all pixels/channels, with an asserted non-identity correction — impossible under the old placement); pre-review gate GATE_EXIT=0; post-fix gate re-run exercises the new --all-targets clippy step.
- **2026-08-25T08:40Z — m4-132: lighttable parity 3.5 — full preview decodes raws; culling cells fill the viewport.** Two gaps closed in the lighttable's viewing paths. (1) Full preview (`f` key): camera raws branch on `raw_preview::is_raw_path` and decode off-thread through `decode_raw_preview` (demosaic RCD + WB + linear downscale to the widget's allocation) followed by `render_linear_to_srgb8` at `PreviewParams::default()` — i.e. **as shot**, matching the darkroom's initial render by construction (hl defaults None both sides); a single owned Send buffer crosses the thread boundary and paints as an R8g8b8 MemoryTexture, mirroring darkroom's nch==3 upload. A "Decoding <name>…" status covers the seconds-long demosaic; the stale-paint guard keeps exactly-one-await-then-guard-then-paint per arm. `.ORF` files render instead of showing "No preview available". (2) Culling: pins `min_columns = max_columns = window` and derives each cell's box from the viewport (`cull_cell_pixels`: width = viewport/window, height = viewport − chrome, floored at THUMB_SIZE), with `ContentFit::Contain` so whole frames compare; binds re-apply size+fit every recycle (`apply_cell_size`, pure core `cell_px_for` unit-tested), thumbnails scale aspect-preserving via new `fit_inside` (rounds, upscales small sources), realized cells refresh through the same walk `set_overlay_mode` uses, the vertical page-size notify joins the horizontal one so height-only resizes re-fit, and the file manager's chosen thumb size survives the visit (CULL_SAVED_MAX_COLS save/restore). The old `cull_capacity` window cap is DELETED along with its stepper dead zone: cells shrink instead of wrapping, so every clamped comparison-set size fits by construction. `THUMB_COLS_DEFAULT` moved into lighttable/mod.rs because leave_culling needs it as restore fallback. File-manager thumbnails change subtly for the better: aspect-preserving scale + Cover replaces squash-to-square + Cover (same center-crop look, no pre-distortion).

  The audit row predicted this needed "the darkroom view's BaseImage/render() lifted into a shared module" — it didn't: decode+encode were already free-standing pub functions outside darkroom (`raw_preview.rs`, `preview.rs`); only the darkroom's live-preview *state* is entangled, which a lighttable preview deliberately lacks. PARITY_AUDIT wording corrected accordingly rather than performing a cosmetic move.

  Review: SHIP WITH FIXES (fork subagent on stealth/ox-alpha standing in for fricktrade-architect; explicit read-only mandate honored — review-only). Zero blockers; reviewer independently verified Send boundaries, the stale-guard invariant, texture args vs encoder output, and the save/pin/restore lifecycle across startup-restore / resync-fallback / rapid flips. All applied: MINOR — RUST_MIGRATION_PLAN.md still documented the OLD culling design (cap + pin-rejected rationale) as current truth; amended in-commit with the reversal recorded as the lesson. MINOR — lib.rs stepper comment described the phantom "narrow viewport holds fewer than asked" state; rewritten. MINOR — leave_culling consumed CULL_SAVED_MAX_COLS before its defensive downcast bail; take moved after the guards. MINOR — current_cell_px decision untestable; extracted pure `cell_px_for(mode, stored)` + `cell_px_follows_the_mode_and_falls_back_until_measured`. NITs: dead `target.max(1)` clamp dropped; fit_inside rounds not truncates; module-doc intra-doc links fixed (private BaseImage link removed, wrapped path un-wrapped); set_max_columns now precedes set_min_columns so the pair never reads transiently inverted. Accepted/non-blocking: CULL_CELL_CHROME_PX=84 is an estimate — Contain letterboxing makes a wrong estimate cost slack, not correctness; a container screenshot pass is owed eventually.

  Verified: new display-free tests green under --release (cull_cell_pixels fill math incl. degenerate floors, fit_inside aspect/clamps/degenerates, cell_px_for mode/fallback matrix); full scripts/ci-local.sh **GATE_EXIT=0** (check / clippy --all-targets / test --release [10 suites ok, c41-ui 287] / release link). PARITY_AUDIT.md row 3.5 → Fixed in this same commit. Remaining niceties stay under 3.6: no 100 % zoom/pan on the full preview; grid thumbnails remain gdk-pixbuf-only (raws still show empty *grid* cells — the full preview is where they become visible).

- **2026-08-25T09:00Z — m4-133: lighttable full preview gains 100 % zoom/pan (parity 3.6).** What: `full_preview.rs` wraps the picture in a `ScrolledWindow` (both scrollbar policies Automatic — which bars exist follows from the mode). Wheel zooms doubling stops fit → 100 % → 2× → 4× → 8× (`step_zoom`, `ZOOM_MAX=8.0`) anchored at the cursor; primary-drag pans by dragging the adjustments backwards from drag-begin values (`set_value` clamps); double-click toggles fit ↔ 100 %; zoom resets on every `load()` entry and `close()`, so ← / → never lands mid-zoom on a corner of the next frame. Fit mode size-requests exactly the viewport dims + `Contain` (centred, zero scrollbars, empty pan range so drag is inert by clamping); scaled mode requests tex×k + `Fill` (aspect exact — both axes scale equally). Adjustment values are written immediately AND re-written from `glib::idle_add_local_once` under a generation counter, because uppers only describe the new pan range after GTK reallocates at the new request; each newer apply invalidates older pendings. Resize refit keys off adjustment `page-size` notify (both axes), coalesced through one idle with a pending flag, and only acts when zoom is None (zoomed mode doesn't chase resizes — viewer behaviour). Decode target now measures the **scroller's** allocation, not the picture's: once zoomed the picture's allocation IS the scaled content size and would feed back. `ZoomState` bundles WeakRef scroll/picture + `Rc<Cell>` zoom/tex_dims/generation and is what every controller closure clones — no strong child→ancestor edges (the cycle class the struct doc warns about). Wheel/double-click guard on "nothing painted" (failed decodes leave tex at (0,0)) but still eat the wheel.
  Review: SHIP WITH FIXES (fork subagent on stealth/ox-alpha standing in for fricktrade-architect; read-only mandate honored). BLOCKER fixed: the anchor helper spoke **viewport** space while callers passed picture-local coordinates, which are **content** space once scrolled (child origin sits at viewport −adj) — re-zooming after panning drifted by adj/k per step, hidden from casual testing because first-zoom-from-fit has adj≈0. Fix: call site translates (`cursor_vp = anchor − adj`), helper contract renamed/documented uniformly viewport-space, and `image_coords_speak_viewport_space_not_content_space` pins it. MINOR fixed: `EventControllerScroll` lacked `DISCRETE` — a touchpad smooth-scroll flick fired one full stop per event and rocketed fit → 8×; now wheels zoom and touchpads fall through to native panning. NITs applied: painted guards on wheel/dbl; fallback-centre documented as viewport-space under the now-uniform contract. Accepted as debt: drag has no activation threshold, so double-click micro-jitter can pan a few px between presses (KasmVNC target; GTK double-click slop absorbs it).
  Verified honestly: no live GUI check this increment — xdotool synthetic wheel/gesture events are exactly the unreliable-input category recorded from earlier container sessions, so geometry is pinned display-free: 6 new pure-core tests (stop list incl. top pin + fold-to-fit, rounding/floor dims, letterbox mapping incl. pillar refusal, scaled translation, the viewport-vs-content space trap, anchored adjustment incl. empty-range saturating case). Gate: container check exit 0; clippy `--workspace --all-targets` exit 0; `cargo test --workspace --release` green (c41-ui 293, c41-core 795, c41-db 92); full `scripts/ci-local.sh` **GATE_EXIT=0** (all four ok-lines present in target/cbuild/gate-m4-133.log). Same-commit doc fixes: PARITY_AUDIT row 3.6 struck; RUST_MIGRATION_PLAN's stale "Coverage limit" paragraph (~:481) amended — both its halves closed by m4-132/m4-133 — and the decode-size bullet notes the scroller measurement.
- **2026-08-25T10:20Z — m4-134: collection rule-stack UI — darktable's filtering.c arbitrary rules (parity 2.6 slice 2).** The left panel's "Collection filters" section gains the arbitrary half of darktable's collection filtering: N rules of (property, contains/excludes, value) joined by AND / OR / AND NOT, composing on top of the active collection exactly like the single-state quick filters. New pure module `lighttable/rule_stack.rs`: `like_literal` (the injection boundary — backslash escaped first, `%`/`_` backslash-escaped under `ESCAPE '\'`, `'` doubled, because the loaders splice SQL strings and can't bind params without touching every site); `rule_stack_sql` composing strictly left-associatively with parenthesization at every step (`r1 ANDNOT r2 OR r3` → `((r1 AND NOT r2) OR r3)`) so an OR can't swallow later terms, skipping blank-valued rules (a half-typed entry must not filter everything out mid-keystroke) while later rules still combine through their OWN combinator; token persistence via hand-rolled percent-encoding (everything outside RFC-3986 unreserved encoded, so the `/`/`:` separators are structurally impossible inside a value; leading rule's combinator elided for token stability across first-row deletion; lenient decode capped at MAX_RULES=16). Slice scope deliberately text-only over existing schema columns (`i.filename`, `f.folder`): exposure/ISO/focal-length need EXIF columns + importer parsing (schema migration recorded as the follow-on slice); the mechanism is built so a new property is one enum arm. Splice order rating→year→colour→aspect→rules pinned by `rule_stack_splices_into_composition_order`; all three loader queries carry both aliases so both property columns resolve. UI in panels/mod.rs: per-row three DropDowns + Entry + delete, row 0's combinator hidden (nothing to combine with), "+ Add rule" flat button insensitive at MAX_RULES, startup seed from canonical state (token applied in lib.rs restore-before-build beside aspect_filter), observer rebuilds rows ONLY on collect≠canonical divergence so typing never destroys the Entry under the cursor, persistence through the same one-writer-per-key app-level observer pattern.

  Review: SHIP WITH FIXES (fork subagent on stealth/ox-alpha standing in for fricktrade-architect — API-402 for the named agent again; explicit read-only mandate honored, zero files modified). MAJOR fixed — signal handlers were connected inside the layout pass, which runs on add/delete too, and those paths REUSE the surviving row widgets: after k structural edits each original widget carried k+1 handlers, multiplying identical reload cycles (grid reload + two pref upserts) per interaction. Wiring moved into `new_rule_row` (connected exactly once per widget lifetime, after preselection so construction writes fire nothing); `rebuild_rows` is now purely layout + fresh delete buttons (minted per pass, so their handlers can't accumulate). MINOR fixed — no end-to-end test exercised the composed fragment against real SQLite, so a wrong property-column string would have passed every shape test; added `rule_stack_keeps_only_matching_filenames_end_to_end` covering both columns, Excludes, literal-`%` (broken escaping ⇒ everything survives), the classic quote-injection staying inert, and left-assoc composition end-to-end. NITs applied: `pct_encode` uses `write!` instead of per-byte `format!`; stale-index delete guard gained a `debug_assert!` so future invariant breaks surface instead of no-op-ing silently. Dispositions (not changed): MINOR "every filter change now writes two pref keys" accepted as the documented cost of the one-writer-per-key observer pattern (consolidation needs dirty-tracking; write count grows linearly with filters — noted for the follow-up slice that adds EXIF properties); NIT "add path applies an inert blank rule" REJECTED with reasoning recorded in-code — skipping the apply leaves state≠widgets, and the NEXT unrelated filter change would then see divergence and rebuild, silently deleting the row just added.

  Verified: 11 rule-stack tests + 304 c41-ui total under --release in-container before the gate; full scripts/ci-local.sh **GATE_EXIT=0** (check / clippy --all-targets / test --release [c41-core 795, c41-db 92, c41-ui 304] / release link). Two clippy lints caught pre-gate and fixed (missing_const_for_thread_local on the RULE_STACK thread_local → const init; unnecessary_to_owned on from_strings). No live GUI check again this increment (same xdotool-unreliable-input category as m4-133) — behaviour is pinned display-free; PARITY_AUDIT.md row 2.6 amended in this commit.
- **2026-08-25T10:46Z — m4-135: EXIF rule properties end-to-end (parity 2.6 slice 3).** The rule stack gains darktable's numeric properties: exposure/aperture/ISO/focal-length rules, live from import to SQL. New `c41-core::exif` probes the four standard EXIF tags at import via kamadak-exif (pinned `=0.6.1` — published crate name is `kamadak-exif`, lib target `exif`; `exif = "0.7.5"` resolves only a squatted 0.0.1) into an all-`Option` `ExifMeta`: absent tags yield `None` and are stored as NULL, never invented values, because the values land in the catalogue and feed comparisons where an invented 0 would silently match. `c41-db` widens `images` with nullable REAL columns via pragma-guarded idempotent `ALTER TABLE`s inside `ensure_base_schema` (fresh installs get them from CREATE; upgrades get the ALTERs), and `image_insert` threads an `ImageExif` through; the folder importer probes each file before the insert transaction. `rule_stack.rs` grows `PropertyKind` partitioning the comparator space: text properties keep contains/excludes (`TEXT_SET`), numeric properties get `< ≤ = ≥ >` (`NUMERIC_SET`); predicates splice finite f64 literals straight into the SQL text after `parse_numeric_value` (trims, accepts fraction syntax `1/60` for shutter speeds, rejects zero denominators AND non-finite spellings — Rust's parser accepts `inf`/`NaN`, which would splice invalid SQL tokens; caught pre-review, guarded + tested). Kind/comparator mismatches and unparseable values make a rule INERT (composer skips it — no fragment lands, query unconstrained), matching the m4-134 blank-value semantics; `normalize_cmp` clamps wrong-kind comparators at token-decode time so the observer never holds an unrepresentable state (the divergence-rebuild loop hazard), while comparator indices 0–6 stay stable so pre-m4-135 persisted stacks decode unchanged. Semantics decision recorded: **NULL never satisfies any comparison** — unprobed/pre-migration images stay out of numeric-rule results (deviation: darktable's exiv2 backfill can put zeros in that WOULD match). UI: each rule row carries twin comparator DropDowns, exactly one visible following the property kind (same visibility-switch pattern as the hidden row-0 combinator); collect reads only the visible dropdown mapped through its kind's set.

  Two of my own end-to-end expectations were inverted and fixed by reasoning about what the code SHOULD do (code was right both times): probed rows correctly survive a generous bound (`exposure < 100000` keeps 0.016 s and 2.5 s; NULL exclusion shows as 2-of-3 surviving vs 3-of-3 for the text rule, and as all-NULL focal_length keeping NOTHING under any bound), and an inert mismatched rule leaves the query UNCONSTRAINED (whole table answers) rather than empty — "inert means skipped", not "matches nothing".

  Review: SHIP WITH FIXES (fork subagent on stealth/ox-alpha standing in for fricktrade-architect — API-402 for the named agent again; explicit read-only mandate honored, zero files modified). MAJOR-1 fixed: the demo catalogue's DDL lacked the four EXIF columns, so a deliberate numeric rule in demo mode failed every loader `prepare` into an EMPTY GRID for all rows — including ones an OR-composed text rule should keep; exactly the m4-126 colour-labels failure class recurring. Columns added to `open_demo_db` (NULL-seeded) + regression test preparing `current_filters_sql()` with a live ISO rule against the real demo connection (zero survivors asserted = correct answer, not just non-crash). MAJOR-2 fixed: `rebuild_rows` re-asserted the text placeholder on every layout pass, clobbering numeric rows' kind-dependent placeholder on any add/delete/rebuild; ownership moved entirely to `new_rule_row` + the property handler (the only two paths that change a row's kind) with an explanatory comment. MINORs applied: aperture column gained end-to-end coverage (seeded 4.0/2.8, two Gte assertions — its column string was otherwise untested against real SQLite); duplicated doc comment on `collect` deduplicated; cosmetic wrap in exif.rs unwrapped. MINOR noted, not changed: the importer opens each file twice (probe_dims then probe_or_none) — harmless at folder scale; merge when the probes combine. Reviewer verified SOUND: the numeric literal boundary (finite f64 Display never emits exponents or stray characters), token back-compat, observer stability under typing, once-per-lifetime handler wiring, migration chain ordering before any loader, importer tuple-shape and probe containment, c41-db not depending on c41-core, kamadak In::PRIMARY/tag coverage.

  Verified honestly, display-free per the recorded xdotool-unreliability: 16 rule_stack tests + 310 total c41-ui tests green under --release in-container (309 pre-review + the demo-schema pin); full scripts/ci-local.sh run TWICE — pre-fix **GATE_EXIT=0** (gate-m4-135.log) and post-review-fixes **GATE_EXIT=0** (gate-m4-135-r2.log) — judged by exit codes, never output greps. PARITY_AUDIT.md row 2.6 amended in this same commit (EXIF remainder closed; deviations recorded).
- **2026-08-25T11:21Z — m4-136: named collection presets — parity item 2.6 CLOSED.** darktable's collection-module presets, end-to-end. New store in `c41-ui/src/persist.rs` beside the styles DAO (same pattern: `main.c41_collection_presets (name PK, payload TEXT)`, DDL executed lazily inside the save path, missing table reads back as empty, names bound via params!, upsert on collision, NOCASE name ordering) — payloads are opaque TEXT here by design. The codec lives in `lighttable/mod.rs` next to the five token codecs it composes: `collection_filter_payload()` emits `v1 <rating> <colour> <aspect> <year> <rules>` whitespace-separated; the year range gets its FIRST persistence token (`off` / `a:b`, ordered-years enforced) — it was session-only until now; the EMPTY rule stack encodes as sentinel `-` because an empty string would vanish under split_whitespace and yield a 5-field payload (caught by my own round-trip test before any review). `parse_collection_payload` is strict on STRUCTURE (version tag, six fields, ordered years — a structurally different payload means a writer we can't reason about) and lenient on CONTENT (each component's own decoder falls back to no-filter, so one corrupt field can't make a preset unusable wholesale). `apply_collection_payload` lands the set as ONE view reload: rating/colour/aspect apply silently, year and the stack are written through their silent in-module paths (their pub setters each fire the bus and would show an intermediate half-applied grid), then one explicit `filter_changed()` — end state identical to hand-clicking every control, per-key pref observers rewrite live tokens, so an applied preset BECOMES current state. A no-rules preset explicitly CLEARS an active stack (the generic applier deliberately ignores empty parses). UI: a Presets sub-area at the foot of Collection filters — save row (Entry + Save, insensitive while blank, Enter activates) and a wholesale-rebuilt ListBox with fresh-per-pass Apply/Delete buttons per row (same handler-accumulation soundness as rule rows); demo mode shows "(no presets saved)" forever by design.

  Review: SHIP WITH FIXES (fork subagent on stealth/ox-alpha standing in for fricktrade-architect; read-only mandate honored). Zero critical/major; four MINORs applied: m1 — persist module comment claimed unreadable payloads die "in the loader"; rewritten to place rejection truthfully at apply time (structure all-or-nothing, per-field leniency below), AND corrupt-preset silence surfaced: refresh now pre-parses each row's payload and leaves a visibly inert Apply button ("Preset data unreadable — delete and re-save") instead of a silent no-op; m2 — apply fired filter_changed once mid-state (year setter fired while rules still old) plus again after rules: one transient reload + a stale rule_stack pref write per application; fixed with the single-fire design above; m3a — space-in-rule-value round-trip pinned ("new year" survives via %20); m3b — restore block now resets all five thread-locals instead of one under a comment implying full restoration; m4 noted, not changed — locked/read-only catalogue files fail save just as silently as demo mode; acknowledged in-code until this panel gains a toast channel. Reviewer verified SOUND: six-field contract unsmugglable (rules pct-encoded over RFC-3986 unreserved; `-` impossible for a real stack — ≥3 colons), section-append move order-neutral, GTK wiring accumulation-free, SQL fully parameterised, observer contract intact.

  Verified honestly, display-free: 313 c41-ui release tests green in-container (round-trip incl. space-in-value + clear-path pins, structural-rejection matrix, DAO upsert/delete/ordering); full scripts/ci-local.sh run twice — pre-review **GATE_EXIT=0** (gate-m4-136.log) and post-fixes gate re-run judged by exit code. PARITY_AUDIT.md row 2.6 → fully Fixed in this same commit.
- **2026-08-25T14:20Z — m4-137: colorbalancergb's four C OMP loops replaced with Rust FFI exports (iop loop metric 17→13).** The pure-Rust algorithm shipped in m4-81..85 (`iop/colorbalancergb.rs`); this increment converts the still-live C side per the m4-86 colorin convention: `src/iop/colorbalancergb.c` process() now ends in a single call to new export `darkroom_colorbalancergb_process` (31 args: buffers, premultiplied in/out matrices passed exactly as the C stores them, the 19 scalars/zone vectors, saturation-formula selector, the 512-entry gamut LUT, and mask-display plumbing), commit_params' JzAzBz branch calls `darkroom_colorbalancergb_build_gamut_lut_jzazbz`, and the GUI draw callback's checkerboard fill + three opacity LUTs route through `darkroom_colorbalancergb_checkerboard_fill` / `_opacity_luts`. Net −412 lines of C loop body; Rust side adds a private `process_premultiplied` transcribing the deleted DT_OMP_FOR body line-for-line (entry clipneg → transposed matrix apply → Yrg/Ych → opacity masks → hue rotation → chroma/vibrance → gamut check → grading RGB → masked slopes → sign-preserving midtones pow → Y gamma/contrast → XYZ D65 → JzAzBz/dt-UCS saturation → output projection → checker blend or clipneg). Deliberate coexistence documented: the FFI path uses the C's single premultiplied matrix apply against the arbitrary pipe work profile, while the shipped Rust pipeline keeps `process_in_space`'s fixed-space two-step composition — arithmetic differs only in FP associativity (<1e-4, pinned by an epsilon test over the Rec2020→LMS composition); the FFI path writes the computed alpha lane (0) like the C, unlike process_in_space's preserved alpha. Serial≡parallel: the LUT sampler's reduction(max:) is order-independent so serial is bit-identical; integer `%` guards added where C UB would be a Rust panic (zero checker cell degrades to a solid first-colour field; zero dims rejected by the GUI fill).

  Failures en route, kept for the record: first draft used a `static mut` for out_width — self-caught, replaced by a parameter. Test run SIGSEGV'd because the FFI entry loaded checker colours unconditionally while a legit test passes NULLs under mask_display=0 — loads now lazy-gated on mask_display (contract pinned by that NULL test). Python splice dropped the `in`/`out` DT_IS_ALIGNED aliases ('in' undeclared at :645) and I mis-mounted the full-c build dir at /build instead of its configured /tmp/build — both fixed, rebuild then green. A comment saying "replaces the former DT_OMP_FOR(collapse(2))" matched the metric grep and inflated the count to 14 — rephrased; recount with the plan's exact pattern gives iop = 13 across 8 files (colorreconstruction 3, retouch 2, colorin 2, channelmixerrgb 2 dead, toneequal 1 dead-guarded, colortransfer 1, colorout 1, colorequal 1), colorbalancergb 0.

  Review: APPROVE-WITH-FINDINGS (fork subagent on stealth/ox-alpha standing in for fricktrade-architect per standing direction; read-only mandate honored). R1–R7 all PASS: faithfulness vs the deleted HEAD loop verified line-by-line incl. textual identity of the masked-slope expression; matrix conventions correct (transposed apply for process, NON-transposed input_matrix for the LUT builder, matching dot_product); all 31 args align 1:1 with commit_params derivations (radian hue, contrast=1+p, L_white); aliasing safe for in==out (read-all-then-write-per-pixel, same as C); diff hunks prove zero modification inside process_in_space (shipped pipeline untouched); -Werror Release full-c rebuild green. Both LOW findings fixed: (1) `% checker_1 == 0` guard + pin test `ffi_process_mask_display_zero_cell_size_degrades_not_panics`; (2) intermediate-opacity blend gap closed by `ffi_process_mask_blend_recovers_a_single_intermediate_opacity`, which recovers the shared opacity from data (plain-run clipped output IS the blended v) and requires it to agree across every lane/cell strictly inside (0,1) — a per-lane or pre-clip blend cannot satisfy it. INFO accepted: dt_vector_powf vs scalar powf bit divergence covered by the epsilon-test design per port policy.

  Verified: 26/26 colorbalancergb tests green under --release in-container (24 prior + 2 new); incremental full-c Release -O3 -DNDEBUG -Werror rebuild REBUILD_EXIT=0 against this exact tree; full scripts/ci-local.sh **GATE_EXIT=0** judged by exit code only. PARITY_AUDIT.md "Where the port actually stands" bullet corrected in this same commit (its "4 files / 9 loops total" figure was long stale — replaced with the dated 13-across-8 count plus the ~223-site whole-tree context). OpenCL path deliberately untouched.
- **2026-08-25T17:35Z — m4-138: colorreconstruction's three C OMP loops replaced with Rust FFI exports (iop loop metric 13→10, 8→7 files).** The pure-Rust algorithm shipped in m4-80 (`colorreconstruct.rs`); this increment converts the still-live C side per the m4-86/m4-137 convention. `src/iop/colorreconstruction.c`'s splat / bilateral_blur / slice loops now each end in one call to new exports `darkroom_colorreconstruct_splat`, `_blur_line`, `_slice`; the now-unused static inline `image_to_grid`/`grid_rescale` are deleted and zero OMP tokens remain in the file. Design: instead of transcribing loop bodies a second time, the existing private free functions (`splat_cells`, `slice_cells`, `blur_line`) were refactored to take an explicit `#[repr(C)] GridHeader` + raw cell slice, so the `ColorReconstruct` methods AND the FFI exports share ONE implementation — method tests therefore pin both callers bit-for-bit. The 16-byte C struct layout (`{float L,a,b,weight}`, x-fastest index) is pinned by `cell_layout_matches_the_c_struct_for_ffi_aliasing`. Exports refuse NULLs and degenerate dims instead of panicking: splat/slice reject zero sizes and sizes > i32::MAX (the header math indexes with i32 after casting), blur_line bounds its highest touched index via checked ops; slice additionally rejects roi dims ≤ 0 and grid sizes < 2. Unknown precedence values behave like NONE (pinned). Serial≡parallel: the C loops were plain per-pixel scatter/gather plus a separable [1 4 6 4 1] pass — no reductions or cross-line reads whose order serial changes; the aliased in==out slice contract is preserved (per-pixel read-before-write) and pinned by a dedicated test leg so tiling/vectorising refactors cannot silently break it.

  Failures en route, kept for the record: std `.clamp(0, size−2)` panicked (min>max) on degenerate tiny grids in the rejection tests — SIGABRT surfaced the real invariant that real grids are always ≥5 cells/axis by construction, so the exports now REJECT sizes<2/==0 rather than clamp; the first slice parity test built sub-image-row input while slice indexes LINEARLY over roi dims — divergence at output float 64, fixed by feeding the full linear buffer; a doc-comment fragment `roi_*/iscale` closed its own block comment (`*/` inside comment text) and broke the full-c build with `unknown type name 'iscale'` (REBUILD_EXIT=1 → reworded → 0); hoisting `let h = self.header()` fixed the E0502 borrow conflict between `&mut buf` and `&header()`. Process note: the fork-subagent reviewer role-looped twice (inherited parent context and echoed workflow state back instead of reviewing); SendMessage redirect failed, so the review was re-run on a fresh general-purpose agent carrying an explicit senior-reviewer framing + read-only mandate — that succeeded.

  Review: APPROVE-WITH-FINDINGS (fresh general-purpose agent on opus standing in for fricktrade-architect — API-402 for the named agent; explicit READ-ONLY mandate honored, zero files modified). R1–R7 all PASS: faithfulness of shared bodies vs deleted HEAD loops verified line-by-line; Cell layout ≡ C struct; export signatures align 1:1 with C call sites incl. size_t casts; aliasing contract sound; guards cover every panic path reachable from C-supplied values; shipped pipeline untouched (diff hunks prove no modification inside process()); -Werror Release full-c rebuild green against this tree. Applied: i32::MAX wrap guards in both splat and slice (usize→i32 cast could wrap); new `ffi_splat_unknown_precedence_behaves_like_none` pin; nonzero-origin init ROI in the slice parity test (was x=0,y=0 — masked any roi-offset transcription error); aliased in==out test leg; NIT header-comment reflow; INFO sentence documenting that blur_line's bound holds because the inner fn early-returns for size3<4; annotation that splat's width/height/x/y/scale are passed for header symmetry with only width/height read on this path. Disposition recorded, not changed: dropping splat's unused x/y/scale args rejected — keeping them keeps the Rust signature isomorphic to the C grid struct, which is the whole point of the header-symmetry convention.

  Verified: 14/14 colorreconstruct tests green under --release in-container (9 prior + cell-layout pin + 4 new FFI pins); incremental full-c Release -O3 -DNDEBUG -Werror rebuild REBUILD_EXIT=0 against this exact tree; full scripts/ci-local.sh **GATE_EXIT=0** judged by exit code only (gate-m4-138-r2.log). PARITY_AUDIT.md "Where the port actually stands" bullet updated in this same commit: recount with the plan's exact pattern gives **10 across 7 files** (retouch 2 GUI-only, colorin 2 LCMS-bound, channelmixerrgb 2 dead #ifdef, toneequal 1 GUI, colortransfer 1 live-but-non-portable-in-principle k-means, colorout 1 LCMS, colorequal 1 GUI), colorreconstruction 0 — every remaining iop site is GUI-only, LCMS-bound pending the lcms2-retirement decision, dead code, or non-portable-in-principle; the portable-loop surface of src/iop is exhausted. OpenCL path deliberately untouched.
- **2026-08-25T18:05Z — m4-139: lighttable Zoomable view mode — parity item 3.1 CLOSED.** The mode that shipped greyed out since m4-98 is now real. `GridView` cannot express darktable's infinite zoom plane (integer columns, no continuous cell size, model-driven layout), so the mode is a hand-drawn `DrawingArea` in its own `ScrolledWindow`, added as a HIDDEN overlay child of a new `LighttablePage`-owned `gtk4::Overlay` that wraps the grid scroller (`FullPreview` now wraps that overlay, preserving the m4-98c never-swap-the-child invariant for every layout). Geometry is pure + display-free tested: plane layout (cols from viewport width, rows ceil), gap-aware hit-test as the exact inverse of `cell_origin`, multiplicative wheel steps pinned ±2 px at the ends, anchored-zoom scrollbar math (content·k − cursor_viewport), power-of-two decode buckets, LRU keep-set under a 96 MiB never-empty budget. Interactions mirror darktable as far as existing controls allow: wheel zooms continuously anchored at the cursor (m4-133 discipline: immediate adjustment write + generation-guarded idle re-assert; DISCRETE keeps touchpads native; Propagation::Stop starves the scroller's own wheel), drag pans via seeded adjustments, single click selects through the SHARED SingleSelection (metadata/rating/export follow automatically), double-click opens the darkroom page through the same extracted callback the grid's activate uses, and the bottom-bar stepper becomes images-per-row with an honest range (`CELL_MIN_PX`-derived max, so buttons stay sensitive exactly as long as a step can change something). Thumbnails are gdk-pixbuf like the file-manager grid (raws still blank there — recorded follow-up to share the raw decode pipeline); decodes keyed `(path, bucket)` with larger-supersedes-smaller, scaled DURING decode via `connect_size_prepared` (the m4-132 lesson) so a screenful of large JPEGs can't spike memory, only the file READ off-thread (GObjects aren't Send). Failed decodes land in a negative cache reset on every resync — without it every frame re-reads and re-fails undecodable files forever (review CRITICAL).

  Failures en route, kept for the record: first compile pass surfaced E0308 (`cell_origin` u32×i32), E0015 (`WeakRef::new()` isn't const), eight sites needing `LocalKey::with` form, and no `GestureClick::current_n_clicks` (fixed by using `released`'s n_press parameter). The bigger discovery: `set_draw_func` hands you a cairo Context, NOT a Snapshot — the whole paint side was rewritten from `append_texture`/graphene rects onto cairo (pixbuf blits via `set_source_pixbuf` + scale/clip, text via the bauhaus toy-font idiom, pango layouts dropped rather than dragging in pangocairo). `spawn_decode` originally ran PixbufLoader inside `spawn_blocking` and failed T:Send (Pixbuf is a GObject) — fixed by following the grid's split (read off-thread, decode on main). My python edit inserted the `impl Clone` block INSIDE the `impl ZoomableCanvas` body and split it — brace error, repaired by moving Clone after the impl. Four of my own tests failed against correct implementations: mis-derived contain_rect centre math, an identity case whose expectation contradicted the anchor formula (k=1 must return the CURRENT scroll value), a degenerate-width case expecting the axis cap instead of the CELL_MIN clamp, and `plane_of` promising "zero extent" for empty collections while still emitting a non-zero width — implementation aligned to the doc instead. A python replace landed mid-statement (`let bytes = let p = ...`) from anchoring on the wrong prefix.

  Review: FIX-FIRST (fresh general-purpose agent on opus standing in for fricktrade-architect per standing direction; explicit READ-ONLY mandate honored). One CRITICAL + three MAJORs all fixed: negative cache for undecodable paths (above); stepper direction was INVERTED vs the buttons it inherits (zoom_in means MORE-per-row everywhere else — arms swapped); lighttable keyboard shortcuts were inert in the mode because the sole key controller lived on the grid while the canvas is its sibling — the whole keymap is now factored into one shared handler attached to BOTH surfaces, with the canvas made focusable and grabbing focus on mode entry; decode memory spike closed via size_prepared. MINORs applied: inflight bookkeeping now removes only the entry this decode owns (supersede no longer respawns duplicates; resyncs stop clearing the map), selection frame paints OVER the thumbnail (a full-width image hid three of four bars), paint loop walks only the visible band in closed form instead of O(all items) per frame, stepper/enter now cancel pending adjustment re-asserts like the wheel does (stale-idle could land a stale offset). NITs applied: shared `texture_key` helper, theme-coloured caption/message text (style_context fg, the bauhaus idiom), honest comment about pitch overshoot showing the h-scrollbar. Verified sound by review: m4-98c invariant, FullPreview wrap contract, selection identity through culling model swaps, m4-133 discipline incl. anchor math, LRU/lifecycle/no-Rc-cycles, numeric guards.

  Verified honestly, display-free per repo discipline: 324 c41-ui release tests green in-container (313 prior + 11 new zoomable geometry/policy tests); clippy shows only the repo-known deprecated style_context idiom (bauhaus precedent). Full scripts/ci-local.sh gate run judged by exit code below. PARITY_AUDIT.md updated in this same commit: severity-3 row 3.1 struck, and the stale "darkroom exposes about 6 adjustments" bullet replaced with the current 33-live-modules count.
- **2026-08-25T18:23Z — m4-140: shared lighttable thumbnail service — camera raws now render in BOTH lighttable surfaces (closes the follow-up recorded at m4-132/m4-139).** New `crates/c41-ui/src/lighttable/thumbs.rs` owns the three things both surfaces previously duplicated or lacked: (1) a **decoder** with two branches — raws go through the full preview pipeline (`decode_raw_preview` + `render_linear_to_srgb8`), so ORF/NEF/etc. thumbnails are real demosaiced renders instead of blank cells, while JPEG/PNG/others decode via gdk-pixbuf scaled DURING load through `connect_size_prepared` (the m4-132 lesson — decode-time downscale bounds memory); (2) an **LRU pixel cache** keyed `(path, power-of-two bucket)` under a 128 MiB never-empty budget, where `bucket_for` quantises any requested size onto [128..2048] so the grid's arbitrary cell sizes and the zoomable canvas's continuous zoom genuinely CROSS-HIT one cache instead of two disjoint key spaces (review finding: my first draft oversold sharing between incompatible keys); (3) a **negative cache** of undecodable paths, cleared on every collection refill (`fill_grid`) so each reload is a fresh retry opportunity. Concurrency is a 2-slot gate claimed NON-BLOCKINGLY (`DecodePermit::try_acquire`, released by Drop) BEFORE any task is spawned; busy callers retry later (grid chains a 150 ms `timeout_add_local_once`; canvas retries on its next paint frame since unregistered inflight work re-triggers naturally). The grid's old inline pixbuf block is replaced by `ensure_grid_thumb` + `paint_thumb`; the zoomable canvas deletes its own `failed` field, local `evict_keep_set`, and bucket logic in favour of delegating to the shared module.

  Failures en route, kept for the record: first gate design used a Condvar-parking acquire — review MAJOR: gio's GTask blocking pool is capped (~10 threads), so parked workers stall UNRELATED blocking work like rating/colour-label DB queries during raw-folder browsing; redesigned to non-blocking try-acquire. `let _ = permit;` drops a Drop type IMMEDIATELY (underscore doesn't extend lifetime) — the slot would have freed before decoding; parameter named `_permit`. gdk-pixbuf friction: loader methods live behind `gdk_pixbuf::prelude::*`, pixel access is `read_pixel_bytes()` + `Bytes`' `[u8]` Deref (no read_pixels), `from_bytes` returns bare Pixbuf not Option. `const { RefCell::new(HashSet::new()) }` is E0015 (HashSet::new not const-callable) while clippy demanded the const form for VecDeque — asymmetric init styles, both pinned by tooling. Eviction test needed 70 MB entries because two 60 MB fit inside the 128 MiB budget; `==256` assertion relaxed to `<=256` because integer-factor downscale treats the bound as a CEILING (244 for bound 256); existence-gated raw test silently skipped until the fixture path moved to `env!("CARGO_MANIFEST_DIR")/../..` (cargo runs tests with crate-root CWD).

  Review: APPROVE-WITH-FIXES (fresh general-purpose agent on opus standing in for fricktrade-architect per standing direction — API-402 for the named agent; explicit READ-ONLY mandate honored). 1 MAJOR + 3 MINOR + advisory + 2 NITs, all dispositioned: MAJOR GTask parking redesigned as above; MINOR panic-collapse fixed — `.ok().flatten()` marked PANICKED decodes as corrupt files forever, join Results now matched so Err logs and leaves the path retryable while Ok(None) feeds the negative cache; MINOR alpha honesty fixed — straight-alpha pixels composite over black (`rgb·a/255`) instead of exposing stored under-colour via as-is RGB drop, pinned by a transparent-PNG test leg; MINOR oversold sharing fixed via unified `bucket_for` quantisation (above); NITs applied: identity-resample guard (no scale_simple when dimensions already match), `max_dim.clamp(1, 8192)` cast safety, pid-tagged temp filenames against parallel-test collisions. Advisory accepted as documented trade-off: full-res demosaic per bucket per session is expensive for raws — embedded-preview-first fast path recorded as follow-up.

  Verified honestly, display-free: 334/334 c41-ui release tests green in-container (9 new thumbs tests incl. gate admit-two/refuse-third/free-on-drop, alpha compositing, keep-set policy, raw decode vs real portrait.orf); clippy clean on all new code (only pre-existing repo-known warnings remain); workspace `cargo check` exit 0; full scripts/ci-local.sh **GATE_EXIT=0** judged by exit code only (gate-m4-140.log). PARITY_AUDIT.md rows 3.5 (grid-thumbnail gap) and 3.1 (raw-blank note) amended in this same commit.
- **2026-08-25T19:01Z — m4-141: raw thumbnail decodes persist to a darktable-mipmap-style disk cache (m4-140's cost-honesty follow-up, slice 1).** A raw thumbnail's one full-resolution demosaic is now paid once per (file, bucket) per MACHINE, not per session: finished decodes in `lighttable/thumbs.rs`' raw branch write to `$XDG_CACHE_HOME/c41/thumbs` (`C41_THUMB_CACHE_DIR` override; XDG→`~/.cache` fallback) as 32-byte-header + packed-RGB8 files — magic/version/dims/source-mtime LE, no image encoder in the loop, so a cache hit repaints bit-identical pixels of the original expensive decode. Filenames are `FNV-1a64(path)-bucket.c41thumb`: FNV because std's DefaultHasher is randomly keyed per process and would orphan every entry each launch. The source file's mtime is sealed INSIDE the entry rather than in the filename — my first draft put it in the name and my own deletion-proof test exposed the flaw before it ever ran: with the source gone, stat fails, mtime falls back to 0, the key changes and the cached render can never be served exactly when serving it is most wanted. Final contract: readable current mtime ≠ sealed → stale → re-decode + overwrite (a replaced raw invalidates itself; walked both race orders in review); unreadable current mtime (deleted source) → serve the last known render, darktable's behaviour until refresh; corrupt/truncated entry → miss, never a panic (exact-length validation pins dims to payload size BEFORE any allocation). Writes are atomic tmp-in-same-dir + rename with pid+counter+nanos tmp names; EVERY IO failure silently degrades to no-caching; unset XDG/HOME yields a pure uncached decoder. Budget prune (512 MiB) reuses the session LRU's `evict_keep_set` newest-first walk, never deletes the just-written entry (needed: equal mtimes tie in arbitrary read_dir order), and runs throttled — bytes-written-since-last-prune counter, at most once per 1/16th budget of new data — because a full directory stat-and-sort per store was measurable churn at five-figure entry counts (review M1).

  Failures en route, kept for the record: `HashSet<&Path>::contains(&PathBuf)` E0277 (Q-inference — `.contains(p.as_path())`); clippy `map_or`→`is_some_and`. Review dispositions: SHIP verdict, zero blockers/majors, all R1–R7 PASS. MINORs fixed: M1 prune throttle above; M2 my FNV doc claimed a collision wastes "one slot" — wrong, two colliding paths sharing a filename CAN serve each other's pixels when their sealed mtimes coincide (the mtime seal masks all other cases); reworded truthfully; M3 entries whose metadata cannot be read were exempt from every future prune sweep — now deleted outright (one regenerate later beats permanent off-budget limbo). NITs fixed: nanosecond timestamp mixed into tmp names for separate-pid-namespace containers sharing a volume; documented boundary that mtime-PRESERVING replacement (cp -p, backup restore) defeats staleness by design; panic-safe drop guard on the test helper's temp dir. Recorded, not changed (N4): the grid still lacks per-path inflight dedupe (zoomable has one) so two workers can double-demosaic the same fresh key once — natural companion follow-up to embedded-preview-first.

  Verified honestly, display-free: 340/340 c41-ui release tests green in-container (6 new: canonical FNV vectors pinning the exact hash variant, key-part sensitivity, codec round-trip incl. sealed-mtime staleness contract legs hit/stale/vanished-serves, corruption matrix truncation/append/bad-magic/bad-version/zero-dims/header-only, real-FS store+load, budget prune keeping newest, and the behavioural deletion proof — copy portrait.orf, decode, DELETE THE SOURCE, re-decode bit-identically, which only a genuine disk hit can pass); thumbs.rs clippy clean; full scripts/ci-local.sh **GATE_EXIT=0** judged by exit code only (gate-m4-141.log). PARITY_AUDIT.md row 3.5 amended in this same commit.

## 2026-08-25 20:21 UTC — m4-142: XMP sidecar writer for metadata edits (parity 2.3 gap closed)

**What changed.** Every successful metadata save now also synchronises `<filename>.xmp` beside the image — darktable's `dt_image_synch_xmps` behaviour (`src/libs/metadata.c:383`), closing the second half of PARITY_AUDIT 2.3 (the first half, the editor itself, landed 2026-08-19). New `c41-ui::xmp` module (~880 lines incl. tests) on quick-xml 0.38 (new dependency, resolves 0.38.4): `sync_sidecar(db, path)` reads all five Dublin Core fields back from the catalogue and rewrites the sidecar in one streaming pass; `panels/mod.rs` calls it from both commit paths (interactive entry closure + `flush_metadata_edits`, now batched per image into ONE save_metadata transaction + ONE sync instead of up to five of each), toasting "Could not update the XMP sidecar" on failure without blocking the authoritative catalogue write. `persist.rs` gained `try_load_metadata` (failure vs genuine blank distinction) and its stale "Known gap" doc paragraph was replaced.

**Safety contract, pinned by 14 tests.** Merge-preserving by construction: foreign `rdf:Description` blocks, comments, CDATA and packet PIs pass through byte-identically; only the five target properties are replaced (catalogue wins, blank deletes); namespace-correct across prefix spellings (recognition by DC-URI binding, output reuses the document's DC/RDF prefixes, declares `xmlns:dc` only when absent); shapes match exiv2/darktable (`Alt`/`x-default` for title/description/rights, `Seq` creator, `Bag` publisher); same-directory tmp+rename atomic writes; a malformed/unreadable existing sidecar is reported false and left byte-identical — never clobbered. Deviations recorded in PARITY_AUDIT: multi-li creator/publisher and non-x-default language alternatives collapse to the editor's single string (inherent to darktable's one-string model too); synchronous UI-thread writes matching the persist posture.

**Senior review: BLOCK → all findings fixed.** fricktrade-architect API-402s again; fresh general-purpose agent with model:"opus" and an explicit READ-ONLY mandate stood in (same substitution as m4-107..m4-141, documented here every time). The reviewer went beyond reading: extracted the merge machinery into a scratch crate and adversarially probed quick-xml 0.38.4 semantics. **BLOCKER:** the injected Description landed AFTER `</rdf:RDF>` — a sibling under x:xmpmeta that ISO 16684-1 consumers silently ignore, so the common case (exiftool sidecar without our fields) saved successfully yet reached no other tool, AND multiplied a degraded copy on every later sync because the stray block never counted as ours. Fixed by computing the injection before writing the close; new tests assert the `</rdf:Description></rdf:RDF></x:xmpmeta>` ordering and byte-idempotence of a double sync. **MAJOR:** nested `rdf:RDF` inside a buffered Description split-brained between writers producing ill-formed output — special cases now gated on `desc.is_none()` so anything seen while buffering streams verbatim. **MINORs:** attribute parse errors were flattened away (silently dropping attributes while still rewriting) — now refused like any other parse failure; packets truncated at a clean event boundary passed EOF unwritten — caught by a `depth != 0` guard (quick-xml emits clean Eof with elements open); `sync_sidecar` treated an unresolvable catalogue as "user blanked everything", one future caller away from stripping users' sidecars — fail-closed via `try_load_metadata`. **NITs fixed:** language-alternative collapse documented + pinned; XML-illegal control chars replaced with U+FFFD instead of creating permanently ill-formed files; accepted atomic-write edges documented; flush-path batching above.

Failures en route, kept for the record: first draft had seven known-broken spots I wrote deliberately before compiling (missing helper, dead block, Writer generic mismatches) — Docker iteration cleared them; `[String; 5]` init needed explicit type after `array::from_fn`; `r#"…"#` literal containing `"#2"` terminated early (`r##` fix); QName borrows required using its transparent `.0` field; nested `write_inner_content` closures need `.map(drop)` (they return `Result<&mut Writer,_>`, callers want `Result<(),_>`). My own test helper `text_of` leaked text across siblings ("TD" assertion) — rewritten so each element owns exactly its span. Two of my tests initially encoded WRONG semantics (expected sidecar values to survive empty-catalogue syncs) — corrected to catalogue-wins per upstream direction. The off-by-one in End-depth accounting (element opened at D sees End at D+1) shipped through my first test run as corrupted output and was diagnosed by instrumenting depth transitions rather than more mental tracing.

**Verification:** 14/14 xmp tests + full c41-ui suite 354/350-pass (354 total, 0 failed) under --release in Docker; clippy zero warnings on all touched files; `scripts/ci-local.sh` GATE_EXIT=0 (log target/cbuild/gate-m4-142.log). PARITY_AUDIT.md row 2.3 amended in this same commit.
