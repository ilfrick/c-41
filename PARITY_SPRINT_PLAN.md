# C-41 — Parity Sprint Plan

> What "full parity" means here, what is still missing, and the ordered sprint
> backlog to close it. Working agreement: one increment per sprint through the
> full loop (delegated development → independent senior review → `scripts/ci-local.sh`
> green → PROGRESS.md + PARITY_AUDIT.md updated in the same commit → dual push
> (GitHub + Gitea) → GitHub checks green → running `c-41` container rebuilt and
> verified). State lives in the repo: a fresh session must be able to resume
> from this file + PROGRESS.md + PARITY_AUDIT.md alone.

---

## 1. Current state (2026-11-12)

- Catalogue: **52 of 52 module rows wired** (47 live + 5 elsewhere-driven).
  No inert rows remain.
- Tests: 1,992 `c41-core` / 578 `c41-ui` (Docker release, after S7 CI).
  PreviewParams blob v40 (2353 B); geometry blob v2.
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

### G1 — formerly inert catalogue rows (kernel and product wiring status)

| Row | C module | Kernel status | Size | Notes |
|---|---|---|---|---|
| ~~`Framing`~~ | `borders.c` | wired S1 (4 exports / 21 tests incl. geometry) | **M** | Constant-border compose and frame line shipped in the geometry family; basis/aspect controls remain deferred. |
| ~~`Chromatic aberrations`~~ | `cacorrectrgb.c` | wired S2 (7 exports / 11 tests) | **M** | Full-frame manifold pipeline and Correct-group controls shipped. |
| ~~`Watermark`~~ | `watermark.c` | wired S3 (`darkroom_watermark_blend`, tested) | **M** | Embedded PNG compositor shipped; asset picker, SVG/text assets and additional blend modes remain deferred. |
| ~~`Color calibration`~~ | `channelmixerrgb.c` | wired S4 (2 exports / 6 tests) | **L** | Matrix chain and darkroom controls shipped; known CAT16 bridge divergence is documented in the audit. |
| ~~`Liquify`~~ | `liquify.c` | wired S5 slice 1 (6 exports / 12 kernel tests; lattice driver + UI) | **L/XL** | Fixed 5×5 normalized dx/dy lattice, editable in the module panel, bilinear map interpolation through the existing bicubic C sampler. Not the C freehand Bezier/stamp editor; no selectable interpolation mode or pixel-golden parity claim. Stage at v50 pos 28.0 before Exposure. |
| `Retouch` | `retouch.c` | wired S6 slice 1 + S7 feather (7 exports / clone-stamp driver and tests) plus GUI-only auto-levels | **XL** | Single non-destructive clone stamp with normalized source/destination, radius, opacity and smooth radial feather. Multi-stamp painting, heal, masks and wavelet preview remain deferred. |

S1–S7 delivery order and outcomes are recorded in the sprint ledger below.
All seven increments are shipped; remaining G1 scope is limited to deferred feature
parity (including broader Liquify and Retouch editors), not inert catalogue rows.

### G2 — colour management (soft-proofing + render seam)

- **S-A: soft-proofing.** `c41-core::icc` already has the transforms (A2B/B2A,
  absolute intent, Lab bridge). Product: proof preview toggle + proof profile
  picker (embedded/file/Adobe/sRGB), gamut-check overlay (the C's
  `dt_dev_draw_softproof` semantics); export keeps the normal path. Risk: the
  preview pipeline currently renders to sRGB bytes — proofing needs the seam
  work first (S-B) or a separate proof render path fed from the linear buffer.
- **S-B: wide-gamut rendering, sRGB presentation.** The GTK4/KasmVNC
  `GdkTexture`/Cairo byte surfaces do not carry a selected ICC profile, so an
  Adobe RGB or Rec.2020 encoded buffer cannot honestly be called a wide-gamut
  display. Keep screen-visible previews sRGB; widen non-raw darkroom processing
  and export to linear Rec.2020, with Adobe RGB exports transformed directly
  from that buffer. S8a adds the unclamped ICC transform foundation; S8b must
  carry the working buffer through darkroom decode/geometry and the non-raw
  export funnel. Thumbnails/full-preview/print stay out of S8 and require their
  own parity slice. Preserve current byte identity for sRGB/no-edit sources.
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
| ✅ **S3 2026-11-10** | `Watermark` (G1 M) | **SHIPPED** +1 live (48→49/52) — full `watermark.c` PNG-branch compositor (scale/bbox/alignment/rot/opacity), embedded deterministic 160×160 asset, Stage at v50 pos 69 (last), v36 blob (2006 B), 3×3 grid row; slice 1: built-in PNG only (no picker/SVG/text) — see G6 notes | asset persistence deferred to serialized string table; C-render golden recorded follow-up |
| ✅ **S4 2026-11-10** | `Color calibration` (G1 L) | **SHIPPED** +1 live (49→50/52) — commit_params matrix chain (CAT16 D65↔D50 bridge, 5 adaptation modes, temperature/illuminant polys, mix+sat/light/grey+normalize, gamut, clip) over the pre-ported loop_switch kernel; v50 28.5 pin both funnels; v37 blob (2122 B); 25-control row; review: algebra verified, 1 MAJOR fixed (bypassed() now clears the S1–S4 flag-only stage flags) | CAT16-bridge ≠ ICC-profile D50 (~1–3%) — goldens need bridged reference |
| ✅ **S5 2026-11-11** | `Liquify` (G1 L) | **LIVE, slice 1** +1 (50→51/52) — fixed 5×5 drag lattice in module panel, bilinear displacement map through C's bicubic sampler, stage v50 pos 28.0 before Exposure, PreviewParams v38 (2323 B); backward decode v37 verified; review APPROVE after fixing stage-order finding | freehand Bezier/stamp controls and selectable interpolation deferred; no C golden render parity claim |
| ✅ **S6 2026-11-12** | `Retouch` slice 1 (G1 XL₁) | **SHIPPED** +1 live (51→52/52) — single clone stamp, normalized source/destination capture, hard brush radius/opacity, preview/export stage, PreviewParams v39 (2349 B), backward v38 decode; review APPROVE, Docker CI green | multi-stroke painting, heal, masks and wavelets deferred |
| ✅ **S7 2026-11-12** | `Retouch` slice 2 (G1 XL₂) | **SHIPPED** — optional radial smoothstep feather with default hard edge, v40 blob (2353 B), v39 default decode, full-buffer hard-edge regression; senior review APPROVE; Docker CI 1,992 core / 578 UI passed; commit `6472ccd434d3ea16cbcdc0befd5d1841ec35af58` deployed as `c-41:6472ccd434`, healthy, zero restarts, HTTP/HTTPS 200; GitHub checks green; both remotes verified | multi-stroke painting, heal, masks and wavelets remain deferred |
| **S8a 2026-11-13** | G2 S-B ICC foundation | direct embedded/assumed RGB → linear Rec.2020 transforms; unclamped linear TRCs; sRGB screen path unchanged | curve semantics and gamut preservation |
| S8b | G2 S-B non-raw integration | darkroom preview + non-raw export process in Rec.2020; GTK bytes remain sRGB; Adobe export stays direct | base-image, geometry, fallback, and export ordering |
| S9 | G2 S-A soft-proofing | proof toggle + proof profile + gamut check; docs | proof-of-sRGB-only display limits |
| S10 | G3 S-D per-instance module stack | duplicate/remove instances end-to-end | touches every module's state |
| S11 | G3 S-E masks v1 | draw/parametric mask editor on ≥ 4 modules | biggest; splits further if needed |
| S12 | G6 decision batch | each G6 row: ship/withdraw/refuse recorded; cosmetic + clippy pass | — |
| S13 | Closure | 52/52 counters, audit "Not on the table" rewrite, final CI + deploy | — |

> Sprint ledger: **S1 ✅ 2026-11-10** (Framing). **S2 ✅ 2026-11-10**
> (Chromatic aberrations). **S3 ✅ 2026-11-10** (Watermark). **S4 ✅ 2026-11-10**
> (Color calibration). **S5 ✅ 2026-11-11** (Liquify). **S6 ✅ 2026-11-12** (Retouch slice 1). **S7 ✅ 2026-11-12** (Retouch feather control). Broader editor parity remains deferred.
| S12 | G6 decision batch | each G6 row: ship/withdraw/refuse recorded; cosmetic + clippy pass | — |
| S13 | Closure | 52/52 counters, audit "Not on the table" rewrite, final CI + deploy | — |

Counter bookkeeping: S1–S6 brought the wired count from 46 to 52 (current breakdown: 47 live + 5 elsewhere-driven). S7 adds feathering without changing the catalogue counter; its Docker CI and deployment are verified; G2/G3/G6 remain open.

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