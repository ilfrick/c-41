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

## DNG OpcodeList3 GainMap: pipeline feature and selected follow-ups

All commits below are ancestors of the pinned SHA. The authored and committed
dates differ for several commits because they were rebased together; both are
recorded to avoid presenting author time as landing order.

| Upstream commit | Authored / committed | Pinned-baseline change | Triage |
|---|---|---|---|
| [`c7d98b58ea4ff39a3f30bd1f16bbccc34191b927`](https://github.com/darktable-org/darktable/commit/c7d98b58ea4ff39a3f30bd1f16bbccc34191b927) — `dng_opcode: apply the GainMap from DNG OpcodeList3` | 2026-10-11 / 2026-11-09 | Parses and applies supported OpcodeList3 GainMap data through image metadata/cache, demosaic for CFA data and rawprepare for linear DNG; includes CPU/OpenCL paths and compatibility/default handling. | **PORT — S14.** A distinct RAW capability; implement after defining supported DNG/map shapes, metadata ownership, crop/active-area geometry, edit defaults and CPU-correct behavior. The existing OpcodeList2 helper is not this feature. |
| [`26117d31144737704283888644b1915fb219dd85`](https://github.com/darktable-org/darktable/commit/26117d31144737704283888644b1915fb219dd85) — `dng_opcode: reject GainMap spacings that cannot be reciprocated` | 2026-08-04 / 2026-11-09 | Rejects map spacing/origin values that cannot be safely used by the sampler. | **REFERENCE** for finite/range/reciprocal validation and malformed-input tests in the port. |
| [`2657ff3bc4515a8b500ceb4fd02e77ea3bfa8dbf`](https://github.com/darktable-org/darktable/commit/2657ff3bc4515a8b500ceb4fd02e77ea3bfa8dbf) — `dng_opcode: compare the GainMap rectangle against the active area` | 2026-08-30 / 2026-11-09 | Validates the map rectangle against active image dimensions rather than padded raw-buffer dimensions. | **REFERENCE** for active-area/crop geometry tests; C-41 currently has no corresponding RAW active-area model in its Rust decode path. |
| [`dcb52f7ffe16e1d255d43b8634c77c217cd52719`](https://github.com/darktable-org/darktable/commit/dcb52f7ffe16e1d255d43b8634c77c217cd52719) — `demosaic, rawprepare: do not read and write one image in the GainMap kernel` | 2026-09-05 / 2026-11-09 | Uses separate/scratch storage to avoid in-place OpenCL image read/write. | **REFERENCE** only if a C-41 GPU path is added; first establish the CPU path and CPU differential. |
| [`e43fbf3f7323775de78739257722d686c2d9da79`](https://github.com/darktable-org/darktable/commit/e43fbf3f7323775de78739257722d686c2d9da79) — `Validate DNG opcode payload sizes` | 2026-10-07 / 2026-10-12 | Adds parser payload-size validation for DNG opcodes. | **REFERENCE** for bounds-checked parsing, truncation/overflow tests and fail-closed behavior. |
| [`eebdfbfd403d3a416c3c43e98905088718999079`](https://github.com/darktable-org/darktable/commit/eebdfbfd403d3a416c3c43e98905088718999079) — `RELEASE_NOTES: DNG OpcodeList3 GainMap` | 2026-11-09 / 2026-11-09 | Release notes only; no additional implementation delta. | **NOT APPLICABLE** as a standalone code port; retain as evidence of the feature's upstream landing/release. |

The pinned upstream call path is broader than a kernel: `src/common/exif.cc`
locates the opcode payload; `src/common/dng_opcode.c` parses/validates and
stores the map; `src/iop/demosaic.c` and `src/iop/rawprepare.c` apply it to
supported CFA/demosaiced and linear-DNG paths; image-cache ownership and the
OpenCL code are in `src/common/{image,image_cache,opencl,dng_opcode}.*` and
`data/kernels/dng_opcode.cl`. The map is not applied by simply recognizing a
`.dng` extension. The supported shapes and documented no-clamp/headroom behavior
must be carried into the compatibility and reference tests.

C-41 has **partial adjacent code, not OpcodeList3 GainMap parity**. Its retained
C path in `src/common/dng_opcode.c` parses OpcodeList2 Bayer gain maps and some
OpcodeList3 warp/vignette opcodes; `src/iop/rawprepare.c` passes four Bayer
OpcodeList2 maps to `darkroom_rawprepare_apply_gainmaps` in
`crates/c41-core/src/iop/rawprepare.rs`. That helper is a per-pixel four-map
Bayer sampler, not the OpcodeList3 three-plane/general-geometry path. The Rust
RAW path used by preview/export calls `crates/c41-core/src/rawimage.rs::load_with`
through `crates/c41-ui/src/raw_preview.rs` and `dialogs/mod.rs`; no OpcodeList3
GainMap payload, active-area geometry, or application stage is wired there.
This distinction preserves the evidence for the legacy C/FFI path without
claiming it covers the current Rust RAW workflow.

Before changing status, test supported and rejected opcode shapes, malformed
and truncated payloads, finite/reciprocal spacing, active-area and crop offsets,
padded/unpadded sensor buffers, pixel-centre interpolation and edge behavior,
legacy-edit defaults, CFA and linear-DNG paths, and consistent preview/export
output. Decide and document the supported map forms before implementation;
exercise GPU parity only if a GPU path is later introduced.

## DNG ForwardMatrix and ProfileHueSatMap: color-profile feature and selected commits

All three candidates are ancestors of the pinned SHA and absent from current C-41
HEAD. Their authored and committed timestamps differ for the first two commits.

| Upstream commit | Authored / committed | Pinned-baseline change | Triage |
|---|---|---|---|
| [`525c68c5361a5967e5a4c7cfc3039513a6622287`](https://github.com/darktable-org/darktable/commit/525c68c5361a5967e5a4c7cfc3039513a6622287) — `First part of using DNG forward matrix` | 2026-09-30 / 2026-10-01 | Reads DNG ForwardMatrix tags and constructs the corresponding profile matrix using illuminant-associated ForwardMatrix, AnalogBalance and CameraCalibration rather than treating it as ColorMatrix. | **PORT — S14.** Implement explicit matrix semantics/profile choice in the Rust RAW path; preserve ordinary ColorMatrix and current defaults unless the product decision changes them. |
| [`f00505df6f13cbb58d32094ae7b730f92bb69f6f`](https://github.com/darktable-org/darktable/commit/f00505df6f13cbb58d32094ae7b730f92bb69f6f) — `Use forward matrix in dng files via a new cmsHPROFILE` | 2026-09-30 / 2026-10-01 | Adds a selectable DNG ForwardMatrix input profile using a constructed LittleCMS profile. | **PORT — S14.** This is user-visible behavior absent from C-41’s Rust RAW UI/render path; keep it distinct from the default embedded ColorMatrix profile. |
| [`47002f161c6915a8202a1a3a08bf764ca82e8633`](https://github.com/darktable-org/darktable/commit/47002f161c6915a8202a1a3a08bf764ca82e8633) — `Full support for dng forward matrix images` | 2026-10-07 / 2026-10-07 | Completes the profile path, including DNG ProfileHueSatMap processing and CPU/OpenCL application. | **PORT — S14 follow-up.** Treat HueSatMap parsing, table semantics, profile association and interpolation as a distinct extension; ForwardMatrix alone does not imply it. |

At the pinned upstream baseline, `src/common/exif.cc` reads the DNG color,
calibration, analog-balance and ForwardMatrix tags and selects the associated
illuminant. The profile path composes the ForwardMatrix with the inverse
AnalogBalance×CameraCalibration and adapts D50 to the working profile. It is a
selectable profile, not the fresh-import default; the pinned implementation
selects an illuminant rather than interpolating matrices. `470…` adds the
associated ProfileHueSatMap data and applies it after the matrix transform.
Relevant upstream paths include `src/iop/colorin.c`, `src/common/exif.cc`,
`src/common/image.h`, `src/common/image_cache.c`, and CPU/OpenCL colorin code.

C-41 has **partial adjacent matrix handling, not ForwardMatrix profile parity**.
The retained C `src/common/exif.cc` reads ForwardMatrix tags into temporary
arrays and logs/marks native DNGs with ForwardMatrix for missing-sample review,
but does not store or use them to build `d65_color_matrix`: that matrix is
computed from AnalogBalance×CameraCalibration×ColorMatrix. The current C
`src/iop/colorin.c` exposes the embedded matrix profile, not a ForwardMatrix or
DNG-look profile; ProfileHueSatMap tags in EXIF handling are not applied. The
Rust RAW path uses `rawloader`'s `xyz_to_cam` in
`crates/c41-core/src/rawimage.rs::rec2020_from_cam_matrix`; that does not carry
ForwardMatrix, CameraCalibration, AnalogBalance, illuminant-specific profile
selection or HueSatMap data into the Rust preview/export transform. Evidence:
`crates/c41-ui/src/raw_preview.rs` and `crates/c41-ui/src/dialogs/mod.rs` call
through the Rust `RawImage` rendering path. The tag being present or read is not
evidence that the ForwardMatrix profile transform is applied.

Before changing status, compare real DNG fixtures with ColorMatrix and multiple
ForwardMatrix/illuminant combinations, AnalogBalance/CameraCalibration,
malformed or singular matrices, default versus explicit profile selection, and
resulting neutral/profile values against pinned darktable. Test missing tags and
safe fallback. For ProfileHueSatMap, separately test dimensions/encoding,
illuminant association, hue wrap, saturation/value boundaries, table
interpolation and preview/export consistency; test CPU/GPU only if a GPU path is
introduced. Do not infer ForwardMatrix interpolation or default selection from
the existence of multiple matrices.

## RAW white balance and manual calibration: workflow boundary and selected commits

The commits below are ancestors of the pinned upstream SHA. C-41 source
comparison: `bb8465414ccec026b76c102d18dba9a6d1afdf77` (later commits in this
S9 documentation increment do not change code). Some early behavior is present
in the retained C IOP; that does not mean it is exposed by the Rust GTK4 RAW
workflow. This C-41 source includes the early commits but not the 2026
manual-calibration/image-specific follow-ups listed below. The DNG ForwardMatrix
commits are triaged separately above.

| Upstream commit | Authored / committed | Pinned-baseline change | Triage |
|---|---|---|---|
| [`651f7fe868bbaa196885478d67a08c8f4b47c5bc`](https://github.com/darktable-org/darktable/commit/651f7fe868bbaa196885478d67a08c8f4b47c5bc) — `Rework temperature and channelmixerrgb` | 2023-10-26 / 2023-10-27 | Introduces shared white-balance/chroma state coordinated between temperature and Color Calibration. This commit is also in current C-41 HEAD's retained C history. | **REFERENCE** for cross-module state and invariants; the Rust UI currently stores separate fixed WB sliders and Color Calibration parameters. Do not treat the retained C code as Rust UI parity. |
| [`1c5df700213ea71c88e9c08ee20748d845550fbc`](https://github.com/darktable-org/darktable/commit/1c5df700213ea71c88e9c08ee20748d845550fbc) — `Introduce as-shot to later D65 workflow` | 2024-02-06 / 2024-02-08 | Adds a late correction workflow that can apply as-shot WB early and correct toward the camera reference later, with distinct spot/picker behavior. Present in current C-41 HEAD's retained C IOP. | **REFERENCE** for stage ordering and mode semantics. The Rust RAW path applies decoded as-shot multipliers but exposes no corresponding mode/picker state. |
| [`9540e292199cc78114a19d0501563233412d365f`](https://github.com/darktable-org/darktable/commit/9540e292199cc78114a19d0501563233412d365f) — `Fix temperature defaults` | 2024-02-17 / 2024-02-18 | Corrects mode-dependent temperature defaults. Present in current C-41 HEAD's retained C history. | **REFERENCE** for the default/migration tests needed if the Rust UI gains equivalent modes; not a request to change today's as-shot default. |
| [`c73800e2654830ca99c2279c707d4312d2c58cb7`](https://github.com/darktable-org/darktable/commit/c73800e2654830ca99c2279c707d4312d2c58cb7) — `Add exif colorcalibration support` | 2024-04-15 / 2024-04-17 | Reads CameraCalibration metadata for the legacy C color-matrix calculation. Present in current C-41 HEAD's retained C path, but not carried by the Rust `RawImage` color transform. | **PORT to Rust RAW — S14** if pinned DNG calibration behavior is not otherwise covered. This is CameraCalibration/ColorMatrix behavior, not the later ForwardMatrix profile; see the ForwardMatrix section. |
| [`fd7ddaf7ccb164fa3fddc2263a7935eab87a0e06`](https://github.com/darktable-org/darktable/commit/fd7ddaf7ccb164fa3fddc2263a7935eab87a0e06) — `colorcalibration also for manual wb (#20304)` | 2026-11-04 / 2026-11-04 | Defines broad interaction semantics between manual WB and Color Calibration, including chroma state, picker/illuminant behavior and trouble conditions. Not present in current C-41 HEAD. | **PORT — S14**, adapted to the Rust edit/pipeline model; do not transplant the GTK3/shared-global state design. |
| [`3b875937e5556692f6b455e0375ca434a37ffdbb`](https://github.com/darktable-org/darktable/commit/3b875937e5556692f6b455e0375ca434a37ffdbb) — `temperature: as shot and camera reference use the image's multipliers` | 2026-11-10 / 2026-11-10 | Makes as-shot/camera-reference modes resolve from the image currently being edited, rather than multipliers stored by a copied history/style/preset. Not present in current C-41 HEAD. | **PORT / REFERENCE — S14.** If symbolic WB modes are introduced, persist the mode and resolve per-image coefficients at render time; test style/history transfer across different RAW files. |

Current Rust RAW evidence: `crates/c41-core/src/rawimage.rs::load_with` stores
decoder-provided WB multipliers and `RawImage` applies them around demosaic and
highlight reconstruction; `crates/c41-ui/src/raw_preview.rs` and
`dialogs/mod.rs` use that Rust path for preview/export. The GTK4 “White balance”
row in `crates/c41-ui/src/darkroom/mod.rs` exposes three manual RGB multiplier
sliders, while Color Calibration is a separate parameter group. The C-FFI
`darkroom_temperature_*` functions in `crates/c41-core/src/iop/temperature.rs`
replace pixel loops called by retained `src/iop/temperature.c`; the legacy C
module has its own older mode/picker UI. Neither the FFI pixel kernels nor
retained C UI prove Rust mode/picker/state parity.

Keep these concepts separate in further triage: decoded/as-shot WB coefficients;
a neutral picked from image pixels; camera-reference/D65 and late-correction
modes; and DNG AnalogBalance, CameraCalibration, ColorMatrix and ForwardMatrix
metadata. Verify mode/default migration, monochrome behavior, image-specific
resolution after style/history copy, WB-before/after-demosaic ordering, picked
neutral calculation, and consistent preview/export output against pinned
fixtures. The 2026 `fd7…` manual-calibration changes are broad and require a
Rust-native interaction design, not just more coefficient sliders.

## Remaining work

The module, RAW/formats, and workflow/runtime matrices remain conservative inventories.
Other candidate clusters named in `ROADMAP.md` §5.4—including Contrast & Texture,
highlights, precision and HDR—remain unclassified until rows are added with
pinned-source and C-41 call-path evidence. The Spektrafilm, DNG and WB follow-up
lists here are representative, not exhaustive; additional commits for all
reviewed features may still need triage. S9 is incomplete until the remaining
capabilities and selected commit deltas are verified and prioritized.
