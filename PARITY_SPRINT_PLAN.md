# C-41 — Parity Sprint Plan

> What "full parity" means here, what is still missing, and the ordered sprint
> backlog to close it. Working agreement: one increment per sprint through the
> full loop (delegated development → independent senior review → `scripts/ci-local.sh`
> green → PROGRESS.md + PARITY_AUDIT.md updated in the same commit → dual push
> (GitHub + Gitea) → GitHub checks green → running `c-41` container rebuilt and
> verified). State lives in the repo: a fresh session must be able to resume
> from this file + PROGRESS.md + PARITY_AUDIT.md alone.

---

## 1. Current state (2026-11-10)

- Catalogue: **48 of 52 module rows active** (43 live + 5 elsewhere-driven).
  **4 inert rows:** `Color calibration` · `Retouch` · `Liquify` · `Watermark`.
- Tests: 1 960 `c41-core` / 560 `c41-ui` (Docker release). PreviewParams blob
  v35 (1977 B); geometry blob v2.
- G2 colour: input profiling (embedded + per-image assumption) live; output
  profile choice (sRGB/Adobe RGB) live for all Rust exports; raw Adobe exports
  transform directly from linear Rec.2020. **Still open: soft-proofing and the
  fixed-sRGB render seam** (wide-gamut render/display).
- G3 masks: kernels fully ported (9 shape modules, 33 FFI exports, used by the C
  app); **zero product surface** — no mask UI, no per-instance module stack.
- G4 Lua: not started; G5 GPU: deliberately CPU-only (documented product
  decision, compose profiles passthrough-only); G6: open decisions below.

## 2. Definition of parity (the target)

1. **All 52 catalogue rows live** and honest — either a real control, an
   elsewhere-driven link to a real control, or an explicitly recorded refusal
   (no inert rows at the end).
2. G2 closed: soft-proofing path + a user-selectable render/display space for
   raw previews/renders (wide-gamut seam), with the existing byte-identity
   guarantees kept for default output.
3. G3 closed to a **product-minimal mask surface**: parametric/draw masks
   (brush, circle, gradient, path) on the main live modules, backed by the
   ported kernels, via a **per-instance module stack**.
4. G6 decisions closed (below): each item either shipped, withdrawn, or
   explicitly refused with a rationale recorded in PARITY_AUDIT.
5. Declared non-goals for parity (documented, not "inert silently"):
   Lua scripting (G4), GPU OpenCL execution (G5), Foveon X3F, `darktablerc`
   adoption, ROADMAP.md Phases 1–8 (future tier) and full pixel-per-pixel
   bit-parity with the C reference where a documented deviation exists.

Exit criterion for the whole plan: §1's counters read **52/52**, G2/G3 incomings
closed, G6 rows all "done | withdrawn | refused with rationale", and
PARITY_AUDIT's "Not on the table" section rewritten to say what the product is,
not what it is not.

## 3. Backlog (ordered by value / risk)

### G1 — the six inert rows (kernel maths ✘ done, product wiring ✘ to do)

| Row | C module | Kernel status | Size | Notes |
|---|---|---|---|---|
| `Framing` | `borders.c` | ported (4 exports / 21 tests incl. geometry) | **M** | Effect-group border pass: per-side widths, colour, opacity, blend. New `Stage` or geometry post-pass; preview + all export funnels. |
| `Chromatic aberrations` | `cacorrectrgb.c` | ported (7 exports / 11 tests) | **M** | Per-pixel RGBA stage: red/blue transverse shift sliders, RC/GC/BC factor. Straightforward `Stage` + slider row. |
| `Watermark` | `watermark.c` | ported (`darkroom_watermark_blend`, tested) | **M** | Needs an asset-path picker (reuse `watermarks/` templates question) + compositing row (image, position, scale, opacity). First slice: single embedded PNG asset, one blend mode. |
| `Color calibration` | `channelmixerrgb.c` | ported (2 exports / 6 tests) | **L** | 12 coeff sliders + RGB/Lab mode + auto-WB; best implemented on a proven matrix-stage precedent (Primaries). Watch the documented primaries-vs-ICC coefficient divergence class. |
| `Liquify` | `liquify.c` | ported + hardened (6 exports / 10 tests, FFI safety fixes) | **L/XL** | Requires a warp-grid UI (drag lattice over the preview) + driver over the grid-math kernels. Param set is vector-shaped; the first slice can be a simple point-deform table like the C's stamp. |
| `Retouch` | `retouch.c` | ported kernels (7 exports / 12 tests) plus GUI-only auto-levels | **XL** | Clone/blur stamp needs a paint-UI; slice 1 = single-round brush stamp with radius/opacity, slice 2 = wavelet preview + feathering. Biggest G1 row. |

Suggested order: **Framing → Chromatic aberrations → Watermark** (three M
sprints, fast wins, all Effect/Correct rows), then **Color calibration** (L),
then **Liquify** (L), then **Retouch** (XL, planned as two slices).

### G2 — colour management (soft-proofing + render seam)

- **S-A: soft-proofing.** `c41-core::icc` already has the transforms (A2B/B2A,
  absolute intent, Lab bridge). Product: proof preview toggle + proof profile
  picker (embedded/file/Adobe/sRGB), gamut-check overlay (the C's
  `dt_dev_draw_softproof` semantics); export keeps the normal path. Risk: the
  preview pipeline currently renders to sRGB bytes — proofing needs the seam
  work first (S-B) or a separate proof render path fed from the linear buffer.
- **S-B: render/display space.** Stop hardcoding the sRGB display seam:
  `ColorSpace::DisplaySrgb` → user-selectable working→display transform
  (sRGB today, Adobe, wider Rec.2020 display). Non-raw preview/thumbs follow
  raw. Must keep today's byte-identity when the default is chosen (pinned).
- **S-C (after S-B): per-image Output color profile?** Decision recorded
  2026-11-08: the catalog row stays a global Export-dialog link. Revisit only
  if soft-proofing surfaces a per-image need; otherwise leave and say so.

### G3 — masks (product surface on ported kernels)

- **S-D: per-instance module stack.** darktable allows N independent instances
  of one module; nothing in the current stack models that (audit: "instance
  plumbing touches every module"). Infrastructure slice: instance id in
  `PreviewParams`/blob/history, duplicate/remove in the module panel, per-
  instance state on the render path. This is the structural prerequisite for
  per-module masks and must land first. Highest-risk slice in the plan.
- **S-E: minimal mask surface.** Draw + parametric masks (brush first, then
  circle/gradient/path) applied through the blend of live modules, backed by
  `c41-core::masks` FFI kernels. No raster/passthrough masks in v1; mask-banding
  UI on a subset of modules (exposure, invert, levels, colorin…). If S-E proves
  large, split mask-editor UI (draw/erase, feathering) from module wiring.

### G6 — decisions to close (choose per item: ship, withdraw, refuse)

- `heic/heif/avif`: **decide** at `RUST_MIGRATION_PLAN.md:2050` — either
  remove the extensions from the cargo-only product (catalogue then never
  shows 0×0 rows for them) or pull a Rust codec (rare/heif crate) for decode +
  probe. Recommendation: **dynamic probe → refuse-with-rationale** for parity
  (heic encoding documented as darkroom-cli-only via full-c), unless usage
  data says decode matters.
- Highlights LCh/laplacian/segmentation methods (m4-119 deviation): wire the
  remaining two user-visible methods or explicitly refuse (clip + opposed
  already ship).
- Denoise (profiled) nlmeans mode: refuse or port (wavelets-only ships).
- Tone equalizer guided filter / surface blur (`details != NONE`): refuse or
  port; the shipped global-curve mode is documented.
- Filmic auto-enable default preset: consider matching darktable's
  out-of-box filmic-on (auto-exposure) or keep OFF; decide and record.
- `watermarks/` + `styles/` + `darktablerc` template data in `data/`: decide
  keep/drop per G4-style product honesty.
- Slider % formatting (detail/highlights/shadows etc.): cosmetic pass to match
  darktable display units (0-100 % etc.).
- XMP sidecar sync: already partial (rating/labels/etc.); close the audit's
  remaining open items or record refusals.
- History jump/undo/redo for decode-level modules (hotpixels/rawprepare/
  highlights): re-decode on jump (one seam, shared fix, currently documented
  gap).
- Build: clippy warning cleanup (clone! deprecation, reference-fn dead code),
  decide on rustfmt adoption as a separate no-behaviour PR.

## 4. Sprint plan

Each sprint = one increment from §3, full work-agreement loop. Sizes are
increments; "M" ≈ the Grain/Soften (1-2 session-days), "L" ≈ Orientation +
RAW prepare combined, "XL" splits into two.

| Sprint | Backlog item | Deliverable (DoD) | Risk |
|---|---|---|---|
| ✅ **S1 2026-11-10** | `Framing` (G1 M) | **SHIPPED** +1 live (46→47/52) — constant-border compose + frame line, geometry-family, raw+non-raw preview & all export funnels, v34 blob, 3 review passes, no blockers | geometry interaction (crop/framing order) — verified |
| ✅ **S2 2026-11-10** | `Chromatic aberrations` (G1 M) | **SHIPPED** +1 live (47→48/52) — full manifold pipeline over the 7 pre-ported FFI kernels (downsample→get_manifolds→upsample→apply_correction→reduce_artifacts); Stage at v50 13.5 after lens in both funnels; v35 blob (bools 46/47, floats 478–481); 5-control Correct-group row; preview/export parity; NaN-strength hardening; review APPROVE | stage serial (non-pixel-local) — expected |
| S3 | `Watermark` (G1 M) | +1 live; asset picker + compositing row + seeds | asset resolution/persistence |
| S4 | `Color calibration` (G1 L) | +1 live; 12-coeff/auto-WB row on matrix-stage precedent | parity-of-coefficients claim ≠ C (document) |
| S5 | `Liquify` (G1 L) | +1 live; lattice/point-deform UI slice 1 | gesture plumbing heavy |
| S6 | `Retouch` slice 1 (G1 XL₁) | +1 live; brush stamp, radius/opacity, painting on preview | paint-canvas + blend wiring |
| S7 | `Retouch` slice 2 (G1 XL₂) | feathering/wavelet preview or refuse | — |
| S8 | G2 S-B render/display space | user-selectable render seam, default byte-identical | regression risk on the pinned default path |
| S9 | G2 S-A soft-proofing | proof toggle + proof profile + gamut check; docs | proof-of-sRGB-only display limits |
| S10 | G3 S-D per-instance module stack | duplicate/remove instances end-to-end | touches every module's state |
| S11 | G3 S-E masks v1 | draw/parametric mask editor on ≥ 4 modules | biggest; splits further if needed |
| S12 | G6 decision batch | each G6 row: ship/withdraw/refuse recorded; cosmetic + clippy pass | — |
| S13 | Closure | 52/52 counters, audit "Not on the table" rewrite, final CI + deploy | — |

> Sprint ledger: **S1 ✅ 2026-11-10** (Framing shipped). **S2 ✅ 2026-11-10**
> (Chromatic aberrations shipped). Remaining G1 rows for S3–S7: `Watermark` →
> `Color calibration` → `Liquify` → `Retouch` ×2.
| S12 | G6 decision batch | each G6 row: ship/withdraw/refuse recorded; cosmetic + clippy pass | — |
| S13 | Closure | 52/52 counters, audit "Not on the table" rewrite, final CI + deploy | — |

Counter bookkeeping: after S1–S6 the inert list is empty (46→52); G2/G3/G6
work does not change the counter but closes PARITY_AUDIT sections.

## 5. Verification gates per sprint (fixed, not negotiable)

1. Delegated development by a fresh subagent with a precise spec (files,
   conventions, read-only-everything-else).
2. Independent senior review — different agent, same model, full diff; findings
   applied before the gate.
3. `scripts/ci-local.sh` — all four steps green, exit-code only; release-mode
   tests (debug-only runs do not count).
4. `git diff --check`; PROGRESS.md entry + PARITY_AUDIT.md updates in the SAME
   commit (no stale-audit drift).
5. Push to `origin` (dual: GitHub + Gitea), verify both refs; GitHub checks
   green on both `check + test + clippy` and `Build & push Docker image`.
6. Rebuild the running `c-41` container with `CACHEBUST=<sha>`, smoke + swap,
   logs clean; previous image kept as `c-41-rollback-<sha>`.

## 6. Constraints (unchanged)

- No changes to deployment services beyond the `c-41` container itself without
  explicit confirmation.
- `install-debuntu.sh` + `.sha256sum` stay untracked and untouched.
- Credentials stay out of commits/diffs/PROGRESS/task-specs (`.netrc` only).
- C changes gate through the full-c image build; cargo-only product stays
  CMake-free.

## 7. Tracking

- This plan: keep §3/§4 ticked per sprint (strike items, record actual dates).
- PROGRESS.md: newest-last per-sprint entries (what changed, verification,
  failures — the rereadable part).
- PARITY_AUDIT.md: counters + section status updated in the same commit as the
  code that changes them.
- A fresh session resumes from: this file → PROGRESS.md tail → PARITY_AUDIT.md.