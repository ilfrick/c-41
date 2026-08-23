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
