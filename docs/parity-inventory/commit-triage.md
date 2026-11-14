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

## Remaining work

The module, RAW/formats, and workflow/runtime matrices remain conservative inventories.
Other candidate clusters named in `ROADMAP.md` §5.4—including Contrast & Texture,
Spektrafilm, DNG GainMap, white balance, highlights, precision and HDR—are not
classified here unless a row is added with pinned-source and C-41 call-path
evidence. S9 is incomplete until the remaining capabilities and selected commit
deltas are verified and prioritized.
