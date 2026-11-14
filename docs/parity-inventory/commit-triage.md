# S9 upstream commit triage (verified subset)

Pinned upstream baseline: darktable `97912518a5a9a9786aeee3c38116fbc999412ff8`.
C-41 comparison point: current `master` at the time of each entry. This ledger
records a reviewed subset, not an exhaustive upstream-only commit classification;
unlisted candidates remain pending. Structural/module presence alone is not
behavioral parity.

## Decisions

- **PORT** — implement/adapt the user-visible capability in C-41, preserving
  relevant pinned-baseline behavior; not a literal C/FFI transplant.
- **APPLY** — port a specific upstream fix into an existing corresponding C-41
  feature after tracing its call path and compatibility impact.
- **REFERENCE** — use upstream behavior/source as an implementation or test
  reference without treating the commit as a standalone port.
- **NOT APPLICABLE** — explain why the commit does not apply to this product or
  baseline. Do not use this for an unreviewed or merely inconvenient change.

## Verified clusters

| Upstream commit | Feature and pinned-baseline evidence | C-41 comparison | Decision / priority |
|---|---|---|---|
| [`92c0f79a835125609efc68761922629b700d8c8d`](https://github.com/darktable-org/darktable/commit/92c0f79a835125609efc68761922629b700d8c8d) — `add satcurvergb` (2026-09-18) | Ancestor of the pinned SHA. Adds the `satcurvergb` IOP, its OpenCL kernel/build registration, module grouping, and pipeline placement. At the pinned baseline it is a scene-referred “saturation curve” with saturation and brilliance curves indexed by perceptual saturation; its source also contains gamut handling and related preview/histogram controls. Upstream paths: `src/iop/satcurvergb.c`, `data/kernels/satcurve.cl`, `src/common/iop_order.c`, `src/iop/CMakeLists.txt`, `data/kernels/programs.conf`, `src/libs/modulegroups.c`. | No C-41 `satcurvergb` stage or live module was found. `ColorBalanceRgb` has overlapping perceptual saturation/brilliance controls, while `ColorZones` curves use different domains and `Velvia` is a different operation; none supplies the defining saturation-indexed saturation/brilliance curves. Evidence: `crates/c41-core/src/iop/{colorbalancergb,colorzones,velvia}.rs`, `crates/c41-core/src/pipeline.rs`, `crates/c41-ui/src/{preview.rs,darkroom/mod.rs}`. | **PORT — S13.** Implement as a Rust pipeline/UI feature; compare/reuse color math only after algorithm-level verification. Do not count overlapping controls as parity or transplant the GTK3 IOP wholesale. |
| [`7076cab1033da4ba4e5e009bbca3b78f76f1c0e3`](https://github.com/darktable-org/darktable/commit/7076cab1033da4ba4e5e009bbca3b78f76f1c0e3) — `satcurvergb: fix histogram calculation` (2026-10-23) | Ancestor of the pinned SHA. Changes histogram sampling to the preview pipeline so its display is not coupled to main-pipeline zoom/pan state. Pinned `src/iop/satcurvergb.c` applies the preview-pipe condition in both CPU and OpenCL histogram paths. | The parent feature is absent in C-41, so there is no corresponding histogram path to patch independently. Include this pinned behavior when implementing `satcurvergb`; upstream diff is a reference for preview histogram semantics. | **PORT with `satcurvergb` — S13.** No standalone `APPLY`; preserve the preview-pipe sampling behavior in the port and test that main-view zoom/pan does not alter histogram sampling. |

## Spektrafilm: verified feature and selected follow-ups

The parent and each selected follow-up below are ancestors of the pinned SHA.
This is a representative subset, not the full Spektrafilm history; later pack,
CPU/GPU, spatial, robustness, and UI fixes remain to be inventoried.

| Upstream commit | Pinned-baseline change | Triage |
|---|---|---|
| [`d81ccfe23b36d687cb911b01790103798732f77d`](https://github.com/darktable-org/darktable/commit/d81ccfe23b36d687cb911b01790103798732f77d) — `Spektrafilm analog film simulation module` (2026-09-05) | Adds a spectral film/print simulation IOP, its data-pack support, CPU/OpenCL processing and spatial effects. Integration includes module registration and ordering. Upstream evidence: `src/iop/spektrafilm.c`, `src/common/spektra_{sim,core,fetch}.{c,h}`, `data/kernels/spektrafilm.cl`, `src/common/iop_order.c`, `src/iop/CMakeLists.txt`, `src/libs/modulegroups.c`, and `src/tests/unittests/common/test_spektra_{pack,sim}.c`. | **PORT — S13, high effort / gated.** This is a distinct scene-to-display film/print simulation, not just a negative inverter or tone curve. Adapt to Rust/GTK4; do not transplant the GTK3 IOP. Require data provenance/licensing, versioned pack identity and configured download/security policy, and a memory/performance plan before shipping packs or enabling the full workflow. |
| [`351669cab2681f220d63cc57a1b35830599ef1ea`](https://github.com/darktable-org/darktable/commit/351669cab2681f220d63cc57a1b35830599ef1ea) — `Fix compiling spektrafilm module without OpenCL` (2026-09-06) | Keeps the module buildable without OpenCL. | **REFERENCE** in the port's optional-backend build/CI matrix; not a standalone C-41 patch. |
| [`3d2473bab1f014b6c317e3ca5e4175bb64f02719`](https://github.com/darktable-org/darktable/commit/3d2473bab1f014b6c317e3ca5e4175bb64f02719) — `spektrafilm: fix cpu vs gpu discrepancies` (2026-09-07) | Corrects CPU/GPU output differences. | **REFERENCE** for differential fixtures and numeric agreement; any C-41 GPU implementation needs its own test. |
| [`6bfc8a1a1cd4f617d096738c2ef995f2a3eae6f6`](https://github.com/darktable-org/darktable/commit/6bfc8a1a1cd4f617d096738c2ef995f2a3eae6f6) — `spektrafilm: harden pack loading and the sim build against bad input` (2026-10-06) | Hardens untrusted/malformed pack and simulation inputs. | **REFERENCE** for input validation, bounds and fail-closed behavior; data parsing must have independent adversarial tests. |
| [`9ef307dad85c4aa9ae3ba60a63fd47ec30626ac2`](https://github.com/darktable-org/darktable/commit/9ef307dad85c4aa9ae3ba60a63fd47ec30626ac2) — `spektrafilm: cascade wide recursive blurs instead of capping them` (2026-10-06) | Removes a width cap by cascading wide recursive blurs. | **REFERENCE** for behavior and resource strategy; does not justify an arbitrary full-frame allocation in C-41. |
| [`6d92ef5814af2932119932047dc230ef03485c81`](https://github.com/darktable-org/darktable/commit/6d92ef5814af2932119932047dc230ef03485c81) — `spektrafilm: run the print diffusion filter on linear exposure` (2026-10-06) | Places the print diffusion filter in linear exposure. | **REFERENCE** for stage-domain/order semantics and differential tests. |
| [`1b16d045a0fe6c7ff891e9bf32cc5930e27a9f45`](https://github.com/darktable-org/darktable/commit/1b16d045a0fe6c7ff891e9bf32cc5930e27a9f45) — `spektrafilm: give each data pack an identity and match edits to it` (2026-10-01) | Persists pack identity and associates edits with the matching pack. | **REFERENCE** for reproducible versioned data selection; C-41 would need an append-only compatible persistence design. |
| [`c9047ad838b5a16d2188713929af7be877c916dc`](https://github.com/darktable-org/darktable/commit/c9047ad838b5a16d2188713929af7be877c916dc) — `spektra_sim: consume reflectance tables from pack_format 3 packs` (2026-10-01) | Adds simulation support for pack-format-3 reflectance tables. | **REFERENCE** for pack schema and simulation inputs; no corresponding C-41 data-pack path exists. |
| [`46e5e538256a097af825fe1db40100de03ef5c27`](https://github.com/darktable-org/darktable/commit/46e5e538256a097af825fe1db40100de03ef5c27) — `spektrafilm: reduce CPU peak memory` (2026-10-21) | Reduces CPU peak memory. | **REFERENCE** as a resource-risk signal and optimization target; measure C-41 allocations independently. |
| [`25124eb2e932a3610155d26134a197289178969c`](https://github.com/darktable-org/darktable/commit/25124eb2e932a3610155d26134a197289178969c) — `spektrafilm: tile the Young-van Vliet row pass through local memory` (2026-10-29) | Tiles an OpenCL spatial pass through local memory. | **REFERENCE** for GPU tiling; not an implementation in C-41. |
| [`84416b26a71f22e44a464bddc3d0e97e310cd342`](https://github.com/darktable-org/darktable/commit/84416b26a71f22e44a464bddc3d0e97e310cd342) — `spektrafilm: derive the density curves from the curve model throughout` (2026-10-25) | Aligns density curves with the model throughout. | **REFERENCE** for the pinned algorithm and tests; preserve in any port. |

At the pinned baseline, the feature performs a film/print/scan color transform
from spectral profiles and models optional grain, halation, scatter and
diffusion. The 12 MB spectral table and full-image spatial intermediates make
resource bounds material; upstream follow-ups explicitly reduce memory and
optimize broad blurs. Its default data-pack source is a separate repository;
this inventory does not establish the pack repository's redistribution license
or C-41's right to bundle it. Verify provenance, version pinning, license,
network/download behavior and local data handling before implementation or
release.

C-41 has no Spektrafilm Rust stage, UI module or matching data-pack/fetch path.
`Negadoctor` is negative-scan inversion, `ColorBalanceRgb` handles color grading,
and `FilmicRgb` is a display tone transform; these are adjacent workflows, not
spectral film/print simulation. Relevant C-41 evidence:
`crates/c41-core/src/iop/{negadoctor,channelmixerrgb,filmicrgb}.rs`,
`crates/c41-core/src/pipeline.rs`, and `crates/c41-ui/src/{preview.rs,darkroom/mod.rs}`.

## Remaining work

The module, RAW/formats, and workflow/runtime matrices remain conservative inventories.
Other candidate clusters named in `ROADMAP.md` §5.4—including Contrast & Texture,
Spektrafilm, DNG GainMap, white balance, highlights, precision and HDR—are not
classified here unless a row is added with pinned-source and C-41 call-path
evidence. S9 is incomplete until the remaining capabilities and selected commit
deltas are verified and prioritized.
