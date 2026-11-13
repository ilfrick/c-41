# C-41 — Parity Sprint Plan

> What "full parity" means here, what is still missing, and the ordered sprint
> backlog to close it. Working agreement: one increment per sprint through the
> full loop (delegated development → independent senior review → `scripts/ci-local.sh`
> green → PROGRESS.md + PARITY_AUDIT.md updated in the same commit → dual push
> (GitHub + Gitea) → GitHub checks green → running `c-41` container rebuilt and
> verified). State lives in the repo: a fresh session must be able to resume
> from this file + PROGRESS.md + PARITY_AUDIT.md alone.

---

## 1. Current state and upstream baseline (2026-11-13)

- **C-41 code:** HEAD `8640ff37798232c598289d9c1cc1caabcf6ff13f`.
  The current inventory has 52 catalogue rows wired (47 live + 5 elsewhere-driven),
  but that is a snapshot, **not the final target count**.
- **Upstream reference:** `darktable-org/darktable` master
  `97912518a5a9a9786aeee3c38116fbc999412ff8` (2026-11-13). The fork's
  merge-base is `2939f5949aa17405f247859bb4fc272fda3b45c0` (2026-05-21): since
  then upstream has 2,379 commits not in C-41 and C-41 has 789 commits not in
  upstream. These counts are a point-in-time comparison, not a proposed merge.
- **First-pass upstream delta review:** found at least three new processing
  modules absent from C-41 (`contrastntexture`, `satcurvergb`, `spektrafilm`),
  DNG OpcodeList3 GainMap support, RAW/white-balance and numerical correctness
  changes, mask/UI work, and many GTK3/OpenCL-only changes. The candidate ledger
  below is selective; it is not yet an exhaustive audit of all 2,379 commits.
- **G2:** input profiling, output-profile selection, raw Adobe export, and S8a/b
  non-raw Rec.2020 processing/export are shipped. Screen presentation remains
  sRGB; soft-proofing and profile-correct proof/display-profile behavior with final sRGB
  presentation remain open.
- **G3:** nine mask kernel families are ported for C callers, but C-41 still
  lacks per-instance modules and a user-facing mask editor.
- **G4/G5:** Lua and GPU execution remain unimplemented in the Rust product;
  both are now **in scope** for the full-parity target. Foveon X3F and
  `darktablerc` compatibility also move from prior refusals to open parity work.
- **AI-agent track:** Phase 9 in `ROADMAP.md` already specifies a local,
  model-agnostic API/MCP path; it is part of this overall product plan, not a
  substitute for darktable parity.

## 2. Definition of complete parity (owner decision, 2026-11-13)

Target the complete user-visible and processing capability set of the pinned
upstream baseline, then repeat the inventory on every baseline refresh. The only
product-level exclusions are **cloud sync and a mobile app**. In scope are all
upstream processing and utility modules, formats and RAW features, UI workflows,
parameter/default/history behavior, masks and multiple module instances, color
management/proofing, Lua automation, and actual GPU processing (OpenCL or an
equivalent C-41 backend). Also in scope is safe local use by AI agents through
ROADMAP Phase 9.

Completion requires: (1) a generated, versioned upstream capability inventory
with every item implemented; cloud sync and a mobile app are the only exclusions.
Any proposed additional exception requires an explicit, recorded owner-approved
scope change and is not an implicit parity carve-out; (2) preview/export/history
and persistence behavior matching the reference; (3) C-reference golden or
differential tests for numeric processing,
using explicit, reviewed tolerances where architecture/compiler differences make
bit identity impossible; (4) no inert catalogue controls or undocumented
behavioral deviations; and (5) the agent workflow can safely discover, preview,
commit, undo, and export with shared UI state and explicit authorization.

This is a **moving-baseline program**, not a promise that one merge or a finite
checklist of currently planned sprints exhausts upstream. Pin each review wave to a SHA, keep a
complete delta ledger, and re-run it as upstream advances. The checklist below
orders the already-known work; new deltas become explicit backlog items.

## 3. Backlog (ordered by value / risk)

### G1 — formerly inert catalogue rows (kernel and product wiring status)

| Row | C module | Kernel status | Size | Notes |
|---|---|---|---|---|
| ~~`Framing`~~ | `borders.c` | wired S1 (4 exports / 21 tests incl. geometry) | **M** | Constant-border compose and frame line shipped in the geometry family; basis/aspect controls remain open parity work. |
| ~~`Chromatic aberrations`~~ | `cacorrectrgb.c` | wired S2 (7 exports / 11 tests) | **M** | Full-frame manifold pipeline and Correct-group controls shipped. |
| ~~`Watermark`~~ | `watermark.c` | wired S3 (`darkroom_watermark_blend`, tested) | **M** | Embedded PNG compositor shipped; asset picker, SVG/text assets and additional blend modes are open parity work. |
| ~~`Color calibration`~~ | `channelmixerrgb.c` | wired S4 (2 exports / 6 tests) | **L** | Matrix chain and darkroom controls shipped; known CAT16 bridge divergence is documented in the audit. |
| ~~`Liquify`~~ | `liquify.c` | wired S5 slice 1 (6 exports / 12 kernel tests; lattice driver + UI) | **L/XL** | Fixed 5×5 normalized dx/dy lattice, editable in the module panel, bilinear map interpolation through the existing bicubic C sampler. Not the C freehand Bezier/stamp editor; canvas, direct-on-image gestures, interpolation modes and pixel-golden parity remain open. Stage at v50 pos 28.0 before Exposure. |
| `Retouch` | `retouch.c` | wired S6 slice 1 + S7 feather (7 exports / clone-stamp driver and tests) plus GUI-only auto-levels | **XL** | Single non-destructive clone stamp with normalized source/destination, radius, opacity and smooth radial feather. Multi-stamp painting, heal, masks and wavelet preview remain open parity work. |

S1–S8 delivery order and outcomes are recorded in the sprint ledger below.
No inert rows remain in the old 52-row inventory, but complete parity also
requires the newly identified upstream modules and the deferred full Liquify,
Retouch, mask, RAW, scripting, GPU, and workflow behavior.

### G2 — colour management (soft-proofing + render seam)

- **S-A: soft-proofing and display profile.** `c41-core::icc` already has the
  transforms (A2B/B2A, absolute intent, Lab bridge). Match upstream proof-profile
  and display-profile selection/transform behavior, including proof preview,
  gamut-check overlay (the C's `dt_dev_draw_softproof` semantics), and
  profile-correct presentation. Adapt the last presentation step so GTK/KasmVNC
  screen-visible bytes remain sRGB; do not treat that transport constraint as
  permission to omit upstream profile behavior. Export retains its separate
  output-profile path.
- **S-B: wide-gamut processing and proof/presentation.** **S8a + S8b delivered
  2026-11-13:** eligible sources now process/export in linear Rec.2020, with
  direct Adobe RGB export, while the GTK/KasmVNC preview remains sRGB. For full
  parity, implement upstream proof/display-profile selection and transforms,
  adapted to a final sRGB screen presentation; do not omit display-profile
  behavior or mislabel unprofiled RGB bytes as wide-gamut output. Extend the
  work to thumbnail, full-preview, and print paths where upstream uses color
  management, while preserving default sRGB behavior.
- **S-C (after S-B): per-image Output color profile?** Decision recorded
  2026-11-08: the catalog row stays a global Export-dialog link. Revisit only
  if soft-proofing surfaces a per-image need; otherwise leave and say so.

### G3 — multiple instances and complete masks

- **S-D: per-instance module stack.** Implement independent params, history,
  blend state, UI identity, serialization, style/preset behavior, and pipeline
  ordering for N instances of each module. Build migration/backward-compatibility
  and C-reference tests before masks depend on it.
- **S-E: full mask system.** Port and expose every applicable upstream mask
  family and interaction (drawn, parametric, raster/external, groups, feather,
  opacity, inversion, intersections, copy/paste, edit/display modes) across all
  modules that support masks. The existing nine core mask families are a start,
  not the parity endpoint; preserve mask history and serialization semantics.

### G4 — Lua compatibility

Port the supported darktable Lua API surface required by the pinned baseline,
including event hooks, storage/UI extensions and script lifecycle. Inventory
APIs individually, maintain compatibility tests, and sandbox capabilities; the
safe C-41 agent API is complementary, not a Lua replacement.

### G5 — GPU execution

Implement real GPU execution for supported pipeline stages (OpenCL compatibility
or a documented equivalent), runtime capability detection, CPU fallback, device
memory/tiling correctness, and CPU↔GPU differential tests. Device passthrough
alone is not GPU parity.

### G6 — resolve and implement every recorded deviation

- **Formats and sensors:** implement the upstream-supported HEIC/HEIF/AVIF and
  Foveon X3F decode/probe paths; retain each format's metadata, export and
  catalogue semantics. Foveon and `darktablerc` are no longer refusals.
- **RAW pipeline:** implement DNG OpcodeList3 GainMap, full DNG forward-matrix
  handling, all upstream demosaic/highlight/reconstruction modes, and current
  white-balance/reference semantics; include metadata/defaults and decode-history
  behavior.
- **Modules/defaults:** bring Highlights LCh/laplacian/segmentation, Denoise
  nlmeans, Tone Equalizer guided-filter/surface-blur, Filmic auto-enable/defaults,
  and all newly added upstream modules to feature parity.
- **Resources/config:** import compatible upstream presets, styles, watermarks,
  profile databases and `darktablerc` behavior through safe, versioned Rust
  loaders/migrations; do not silently reuse incompatible C-only storage.
- **UX/workflows:** complete slider units, localized aliases, module layouts,
  selection/history/XMP semantics, masks, export/print and keyboard behavior.
- **History:** decode-level edits (hotpixels/rawprepare/highlights and future raw
  controls) must re-decode on undo/redo/jump and survive restart exactly.
- Clippy warning cleanup and rustfmt adoption are maintenance workstreams, not
  substitutes for parity items.

### Initial upstream commit triage (baseline `2939f5949a` → `97912518a5`, 2026-11-13)

This is a prioritized sample from the 2,379 upstream-only commits, not a claim
that every commit has been individually assessed. Dispositions mean: **PORT** =
reimplement/adapt in Rust; **APPLY** = selective transplant to shared C/FFI code
if its exact call path is confirmed; **REFERENCE** = use behavior/tests as a spec,
not cherry-pick GTK3/OpenCL implementation; **REVIEW** = inspect in the next
baseline audit before deciding.

| Upstream change | Relevance and decision |
|---|---|
| `1067c653c0` (2026-08-08), `23d5544c8d` (2026-09-24): new `contrastntexture` module + detail levels | **PORT, high.** Not present in C-41; new real upstream module and a clear missing parity capability. Port parameter/default mapping and EIGF/detail stages with C-reference tests. |
| `92c0f79a83` (2026-09-18), `7076cab103` (2026-10-23): new `satcurvergb` + histogram fix | **PORT, high.** New module absent in C-41; include its later correctness fix and CPU/reference tests, not only the initial feature commit. |
| `d81ccfe23b` (2026-09-05), plus follow-ups including `46e5e53825` (2026-10-21), `37d12f6b98` (2026-10-25), `558c15a1a1` (2026-11-10), `366391a5cf` and `25be0c4ca8` (2026-11-11) | **PORT after algorithm/data/licence audit, high/XL.** New analog-film pipeline with profile data, blurs, halation, glare, image-specific randomness, caching, allocation/error fixes, and later UX/correctness changes. These are representative follow-ups, not a complete commit list; audit all follow-ups and data provenance first. |
| `c7d98b58ea` (2026-11-09), `dcb52f7ffe`, `26117d3114`: DNG OpcodeList3 GainMap + safety fixes | **PORT, high RAW priority.** Current C-41 RAW decode uses the Rust path, so cherry-picking darktable's C/OpenCL integration would not make the product support it. Port sensor-grid evaluation, stage ordering, bounds and degenerate-input guards into the Rust RAW pipeline. |
| `fd7ddaf7cc` (2026-11-04), `3b875937e5` (2026-11-10), `41db6dfd1a` (2026-11-05): white-balance/manual-calibration and late-correction semantics | **REVIEW → PORT where behavior differs.** These change per-image multipliers, camera reference, manual WB interaction and which stages receive correction. Compare against C-41's raw WB, Color Calibration and saved-history behavior; add per-camera/history reference tests before changing. |
| `56b0ca0081` (2026-11-02): rawprepare fallback when white point ≤ black point | **REVIEW, likely PORT.** Safety/correctness edge case; compare against `RawImage` normalization and C-41's current guards, then port the semantics and C-derived tests if absent. |
| `533582354b` (2026-11-11): CACorrect CPU without fast-math; `7f4c9078fc` (2026-11-13): lens/interpolator precision | **REFERENCE + compare; adapt only on evidence.** These change upstream CACorrect/lens/interpolator numerics. The CACorrect change is not assumed to affect C-41's distinct CACorrectRGB path, and no shared call path is yet confirmed. Compare C-41's Rust/FFI implementations and compiler flags against the upstream behavior; carry precision constraints and goldens only where the exact C-41 path or differential tests justify it. Do not import unrelated OpenCL changes. |
| `8dc85d38b1` (2026-10-29), `2732e39c5f` (2026-11-01): highlight opposed/segmentation fixes and linear RAW path | **PORT/REFERENCE.** Relevant to C-41's already-porting Highlight Reconstruction methods; compare each CPU algorithm and method against Rust, add goldens, and implement remaining methods. The OpenCL-only portions wait for G5. |
| `228cbffe88` (2026-10-20), `4b0878d14a` (2026-10-07): sigmoid HDR target-white and PQ/HLG transfer fixes | **REVIEW → PORT.** Potentially changes existing output semantics; compare upstream C with C-41's Sigmoid/HDR encoding and S8 color pipeline on HDR and extended-range fixtures. |
| `f30f439fd8` (2026-11-13): Hot Pixels count reporting; `8dc11444df`: darktable cache sizing; GTK3 mask panel/layout and general OpenCL-only commits | **REFERENCE or SKIP individually.** UI-only reporting/cache changes do not automatically map to C-41 behavior; mask interactions are design references for G3; OpenCL code is not directly usable in Rust but becomes input to G5. No blanket commit imports. |

**Import rule:** cherry-pick only self-contained, reviewed shared-C fixes after
proving C-41 calls that exact code and its ABI/build assumptions match. New
features, C-only RAW-pipeline behavior, GTK3 UI, and OpenCL paths require Rust/
GTK4/GPU adaptations. Every row in the full delta audit must end with a source
SHA, affected feature, decision, rationale, owner sprint, and verification.

## 4. Sprint plan

Each sprint = one increment from §3, full work-agreement loop. Sizes are
increments; "M" ≈ the Grain/Soften (1-2 session-days), "L" ≈ Orientation +
RAW prepare combined, "XL" splits into two.

| Sprint | Backlog item | Deliverable (DoD) | Risk |
|---|---|---|---|
| ✅ **S1 2026-11-10** | `Framing` (G1 M) | **SHIPPED** +1 live (46→47/52) — constant-border compose + frame line, geometry-family, raw+non-raw preview & all export funnels, v34 blob, 3 review passes, no blockers | geometry interaction (crop/framing order) — verified |
| ✅ **S2 2026-11-10** | `Chromatic aberrations` (G1 M) | **SHIPPED** +1 live (47→48/52) — full manifold pipeline over the 7 pre-ported FFI kernels (downsample→get_manifolds→upsample→apply_correction→reduce_artifacts); Stage at v50 13.5 after lens in both funnels; v35 blob (bools 46/47, floats 478–481); 5-control Correct-group row; preview/export parity; NaN-strength hardening; review APPROVE | stage serial (non-pixel-local) — expected |
| ✅ **S3 2026-11-10** | `Watermark` (G1 M) | **SHIPPED** +1 live (48→49/52) — full `watermark.c` PNG-branch compositor (scale/bbox/alignment/rot/opacity), embedded deterministic 160×160 asset, Stage at v50 pos 69 (last), v36 blob (2006 B), 3×3 grid row; slice 1: built-in PNG only (no picker/SVG/text) — see G6 notes | asset persistence remains open; C-render golden is a follow-up |
| ✅ **S4 2026-11-10** | `Color calibration` (G1 L) | **SHIPPED** +1 live (49→50/52) — commit_params matrix chain (CAT16 D65↔D50 bridge, 5 adaptation modes, temperature/illuminant polys, mix+sat/light/grey+normalize, gamut, clip) over the pre-ported loop_switch kernel; v50 28.5 pin both funnels; v37 blob (2122 B); 25-control row; review: algebra verified, 1 MAJOR fixed (bypassed() now clears the S1–S4 flag-only stage flags) | CAT16-bridge ≠ ICC-profile D50 (~1–3%) — goldens need bridged reference |
| ✅ **S5 2026-11-11** | `Liquify` (G1 L) | **LIVE, slice 1** +1 (50→51/52) — fixed 5×5 drag lattice in module panel, bilinear displacement map through C's bicubic sampler, stage v50 pos 28.0 before Exposure, PreviewParams v38 (2323 B); backward decode v37 verified; review APPROVE after fixing stage-order finding | freehand Bezier/stamp controls and selectable interpolation remain open; no C golden render parity claim |
| ✅ **S6 2026-11-12** | `Retouch` slice 1 (G1 XL₁) | **SHIPPED** +1 live (51→52/52) — single clone stamp, normalized source/destination capture, hard brush radius/opacity, preview/export stage, PreviewParams v39 (2349 B), backward v38 decode; review APPROVE, Docker CI green | multi-stroke painting, heal, masks and wavelets remain open parity work |
| ✅ **S7 2026-11-12** | `Retouch` slice 2 (G1 XL₂) | **SHIPPED** — optional radial smoothstep feather with default hard edge, v40 blob (2353 B), v39 default decode, full-buffer hard-edge regression; senior review APPROVE; Docker CI 1,992 core / 578 UI passed; commit `6472ccd434d3ea16cbcdc0befd5d1841ec35af58` deployed as `c-41:6472ccd434`, healthy, zero restarts, HTTP/HTTPS 200; GitHub checks green; both remotes verified | multi-stroke painting, heal, masks and wavelets remain open parity work |
| ✅ **S8a 2026-11-13** | G2 S-B ICC foundation | **SHIPPED** — direct embedded/assumed RGB → linear Rec.2020 transforms; unclamped linear TRCs; sRGB screen path unchanged | curve semantics and gamut preservation |
| ✅ **S8b 2026-11-13** | G2 S-B non-raw integration | **SHIPPED** — darkroom preview + non-raw export process in Rec.2020; GTK bytes remain sRGB; Adobe export stays direct; review APPROVE; Docker CI 1,995 core / 584 UI; GitHub checks green; deployed as `c-41:748a870ecd`, healthy, zero restarts, HTTP/HTTPS 200; both remotes verified | base-image, geometry, fallback, and export ordering |
| S9 | Upstream delta inventory & import wave 1 | Complete inventory for pinned SHA; triage every commit; port high-priority modules/fixes with references and tests | scope is larger than the old 52 rows |
| S10 | G2 complete color-management parity | proofing, gamut check, profile-aware presentation; carry profiles through preview, thumbnails, print and export | GTK/KasmVNC display seam |
| S11 | G3 per-instance module architecture | independent params/history/presets/serialization and pipeline semantics, with migration tests | cross-cuts every module |
| S12 | G3 complete masks | all applicable mask shapes, groups, operations, UI, blending and persisted history | depends on S11; largest UI scope |
| S13 | New upstream modules wave | port Contrast Equalizer, RGB Saturation Curve, Spektrafilm, and every new module in the baseline | Spektrafilm data/algorithm audit |
| S14 | RAW, formats and sensors | DNG gain maps/forward matrices, upstream demosaic/highlight behavior, codecs incl. X3F, complete import/export metadata | rawloader and C pipeline differ |
| S15 | G4 Lua compatibility | versioned supported darktable Lua API, event/plugin lifecycle and compatibility suite | security/sandbox boundary |
| S16 | G5 GPU execution | real GPU backend, runtime selection, fallback, memory/tiling and CPU/GPU differential tests | broad pipeline work |
| S17 | Workflow/UI/config parity | full layouts, selection/history/metadata/presets/shortcuts, `darktablerc`, print, units/localization | keep GTK4-native UX |
| S18 | Close deviation register | resolve every recorded behavior/default/format gap; only cloud sync/mobile remain excluded | many small audited items |
| S19 | Upstream refresh wave | advance pinned baseline, regenerate inventory, classify and schedule the new diff | recurring, never permanently done |
| S20 | Parity release gate | differential suite, task-based UX, agent end-to-end, full CI, dual push and deploy | repeat for each baseline release |

**Parallel AI-agent workstream (ROADMAP Phase 9):** 9a shared GTK-independent
editing/session API and versioned capability schema; 9b read-only local MCP/stdio;
9c validated plan/preview/commit with shared optimistic revisions and undo; 9d
confined derivative export; 9e visible attribution, explicit approval, persistence,
security and end-to-end tests. Agent and UI must use the same edit service; no
second writer or GTK automation.

> Delivered: S1–S8 (Framing, Chromatic Aberrations, Watermark, Color Calibration,
> Liquify slice 1, Retouch clone stamp + feather, ICC foundation, and non-RAW
> Rec.2020 integration). Broader feature parity remains open.

The old catalogue counter remains 52 rows (47 live + 5 elsewhere-driven) for its
historical inventory; S9 refreshes the count and capability list from the pinned
upstream source. It is not the full-parity completion metric.

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