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
