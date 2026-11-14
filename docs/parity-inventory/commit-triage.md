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
| [`e21cd07382f762d7fd89d5fda8417bd64ac741c7`](https://github.com/darktable-org/darktable/commit/e21cd07382f762d7fd89d5fda8417bd64ac741c7) — `Less rawprepare CPU vs OpenCL diffs`, followed by [`f33906d8bd3578909b560a7d0a81bfcd150a6620`](https://github.com/darktable-org/darktable/commit/f33906d8bd3578909b560a7d0a81bfcd150a6620) — `Identical rawprepare output using fma()` | Both are ancestors of the pinned baseline; `f339…` follows `e21…`. The first replaces per-pixel division with a reciprocal multiply on CPU/OpenCL. The follow-up uses an FMA residual correction (`q = n*rdiv; rem = fma(-q, div, n); out = fma(rem, rdiv, q)`) to restore correctly rounded division results while keeping CPU/OpenCL output aligned. The final pinned `rawprepare.c` retains this compensated formula. | C-41's Rust `rawimage.rs::{normalize_cfa,normalize_xtrans}` uses the reciprocal-plus-FMA correction and bit-pattern regression tests; it does not use the intermediate `e21…` multiply-only formula. This is an active RAW path and a direct match to the final pinned arithmetic for the covered inputs, not to the intermediate optimization in isolation. | **REFERENCE — S14 numerical parity, final behavior already represented.** Preserve the compensated reciprocal/FMA form and exact-bit tests. Do not regress to direct reciprocal multiplication; expand input/level tests if later RAW-path work changes the arithmetic. The upstream C/OpenCL implementation is not a GPU RAW path in C-41. |
| [`f28c5fe49b9f4a293ec240156b1dc68adf261983`](https://github.com/darktable-org/darktable/commit/f28c5fe49b9f4a293ec240156b1dc68adf261983) — `Fix CPU vs OpenCL diffs for debinfo builds` | 2026-10-31 | In `math.h`, keeps fast vector `exp`/`exp2`/`powf` approximations for `_DARKTABLE_RELEASE` but uses `expf`/`exp2f`/`powf` in non-release builds. The change addresses debug-information CPU/OpenCL differences; it is an ancestor of the pinned baseline. | C-41's Rust CPU stages call `f32` math methods, while its one current OpenCL kernel uses OpenCL `pow`; it does not import the C vector approximations or have a Rust debug/release math branch. The ignored hardware differential test in `gpu/mod.rs` compares the sigmoid kernel with CPU output at a bounded tolerance and exercises the fused pipeline, but it is not a profile matrix. | **REFERENCE — S16 differential validation.** Keep debug/release and CPU/GPU configurations explicit in future GPU regression coverage; do not transplant darktable's C-only vector-math conditional into Rust without a measured discrepancy. The current hardware test is not evidence of cross-device or all-profile equivalence. |
| [`7f4c9078fcaef2d70fc6d0c95c88022abe868e24`](https://github.com/darktable-org/darktable/commit/7f4c9078fcaef2d70fc6d0c95c88022abe868e24) — `More precision work on lens and interpolators` | 2026-11-13 | The OpenCL Lanczos kernel changes division to `fp32_div`; CPU C changes include strict-FP compile options for interpolation/Lens, true `hypotf` in the shared vignette prepass and `hypotf`/FMA changes in Lens spline/coordinate paths. The commit is an ancestor of the pinned baseline. | C-41's active Rust `interp.rs` Lanczos expression `PI * t / n` matches the pinned CPU C operation order; the upstream `fp32_div` change is in `data/kernels/basic.cl`, not the CPU C interpolation implementation. Rust normalization already uses a reciprocal as in the pinned CPU code. C-41's active `iop/lens.rs` uses liblensfun and bilinear resampling by default, not darktable's `lens.cc` processing path; thus neither its shared-vignette precision edit nor its metadata-spline edits establish a Rust discrepancy. The OpenCL changes have no corresponding C-41 GPU interpolation path. | **REFERENCE — S16 numerical/GPU validation.** Preserve the pinned CPU/OpenCL precision distinctions as reference for future differential tests; no active Rust Lanczos operation-order fix is established by this commit. Treat Lens C/OpenCL edits as references rather than transplanting them into the liblensfun path. Reassess if C-41 adds a matching GPU interpolation path or evidence shows a CPU numerical mismatch. |
| [`1067c653c0026030883ca6b5766a2213425b2eb3`](https://github.com/darktable-org/darktable/commit/1067c653c0026030883ca6b5766a2213425b2eb3) — `Add new module contrastntexture` and [`23d5544c8d9b870c247fc0c3e61ce9bb316c6e0a`](https://github.com/darktable-org/darktable/commit/23d5544c8d9b870c247fc0c3e61ce9bb316c6e0a) — `Contrastntexture step 2 add mulitple detail levels` | Both are ancestors of the pinned baseline. The first adds a scene-referred, edge-aware log-luminance local-contrast IOP. The second changes its versioned params and expands it to coarse/medium/fine detail bands plus shadow/highlight slopes and low-pass/detail-mask previews. The pinned implementation also includes CPU/OpenCL processing, image-scale-dependent pyramid levels, and a custom tiling estimate. | No Rust `contrastntexture` core module, pipeline stage, or live GTK4 darkroom control exists at C-41 `dfbd3bf6ccc3bbc6a51dd1a019291b796c08513c`. `Stage::LocalContrast` and `locallaplacian.rs` implement the distinct `bilat.c` local-Laplacian mode in Lab; they are related in broad effect, not algorithmic or parameter parity. The previously mentioned four-million-pixel silent bypass and duplicate UI-label match arm were in an abandoned, uncommitted attempt, not this checkout; no current implementation exists to patch. | **PORT — S13, high effort/resource-gated.** Adapt the pinned algorithm to Rust/GTK4 and current persistence/pipeline conventions. Do not treat LocalContrast as parity. Before enabling, establish a measured peak-memory bound and correct large-image behavior (tiling or a safe, explicit failure path); do not silently bypass processing. Preserve params/default/version semantics and CPU fallback before considering a GPU backend. |
| [`8b8f06a4f96277cd70dd4898d115b96f1c17f065`](https://github.com/darktable-org/darktable/commit/8b8f06a4f96277cd70dd4898d115b96f1c17f065) — `contrastntexture: allocate piece->data by the data struct, not the params` | Ancestor of the pinned baseline. The original one-level IOP allocated per-pipe data using the params size even though its processing-data struct was larger, causing an out-of-bounds write/read of `noise_bias`; this commit adds struct-sized aligned allocation and matching cleanup. The later pinned implementation retains explicit `init_pipe`/`cleanup_pipe` for its larger processing struct. | C-41 has no corresponding C pipe-data allocation or Rust module at `49929b2d6da0042eb5738ce6beaa6ee1b629088f`; this is not a standalone fix to apply. The Rust port must keep persistent params and derived per-render data distinct, size each allocation from its actual type, and test resource/error paths rather than transliterating the faulty default allocator assumption. | **REFERENCE — S13 implementation safety.** Preserve the allocation/ownership lesson in Rust design; no C/FFI transplant. |
| [`7d7dac363d7492db76aa8fa1cbba5712d9ead3a8`](https://github.com/darktable-org/darktable/commit/7d7dac363d7492db76aa8fa1cbba5712d9ead3a8) — `contrastntexture: add OpenCL support` | Ancestor of the pinned baseline. Adds a separate GPU implementation for luminance, multilevel details, application/previews, and the EIGF surface-blur path; the pinned source also declares OpenCL tiling estimates. | C-41 has no Contrast & Texture stage. Its OpenCL backend currently accelerates only supported contiguous Exposure/Sigmoid spans; container ICD availability and that unrelated dispatch do not establish this operation's GPU parity. | **REFERENCE — S13 future GPU follow-up.** First establish CPU reference behavior and resource-safe large-image behavior; only then consider an OpenCL port with differential tests and CPU fallback. |
| [`8075f03504f43fc2e99fb310c4f549c503920382`](https://github.com/darktable-org/darktable/commit/8075f03504f43fc2e99fb310c4f549c503920382) — `contrastntexture: use the common.h helpers in the OpenCL kernels` | Ancestor of the pinned baseline. Replaces duplicated OpenCL helper definitions with shared `common.h` helpers in the Contrast & Texture kernels. | There is no C-41 Contrast & Texture kernel to patch; the Rust OpenCL source currently covers only its existing Exposure/Sigmoid GPU stages. | **REFERENCE — S13 GPU follow-up.** Use common helpers if a compatible kernel is later added; no standalone product port or current behavior change. |
| [`a33b119694e837ff35e26951890027ba61525133`](https://github.com/darktable-org/darktable/commit/a33b119694e837ff35e26951890027ba61525133) — `Various fixes for pipe gui_data access` and [`b1321bb8f8b5daf20063530f7aedc80c93bd66b5`](https://github.com/darktable-org/darktable/commit/b1321bb8f8b5daf20063530f7aedc80c93bd66b5) — `Fixing critical_section use for masks` | Both are ancestors of the pinned baseline. They add critical-section protection around mask-preview state initialization, toggle-callback writes and CPU/OpenCL processing reads, and constrain the read to a GUI-attached full pipe. The pinned `gui_focus()` loss-of-focus reset still writes `details_display` outside that critical section, so the upstream synchronization is not complete. | C-41 has no equivalent Contrast & Texture stage or GTK3 GUI-data sharing path. A future Rust implementation should keep preview-control state isolated/snapshotted and synchronized with rendering, and should not let a display-only mask mode race with processing. This is a concurrency/UI reference, not evidence of current parity. | **REFERENCE — S13 UI/concurrency.** Preserve race-free mask-preview state and clear temporary previews on focus loss if those optional previews are ported. |
| [`92c0f79a835125609efc68761922629b700d8c8d`](https://github.com/darktable-org/darktable/commit/92c0f79a835125609efc68761922629b700d8c8d) — `add satcurvergb` (2026-09-18) | Ancestor of the pinned SHA. Adds the `satcurvergb` IOP, its OpenCL kernel/build registration, module grouping, and pipeline placement. At the pinned baseline it is a scene-referred “saturation curve” with saturation and brilliance curves indexed by perceptual saturation; its source also contains gamut handling and related preview/histogram controls. Upstream paths: `src/iop/satcurvergb.c`, `data/kernels/satcurve.cl`, `src/common/iop_order.c`, `src/iop/CMakeLists.txt`, `data/kernels/programs.conf`, `src/libs/modulegroups.c`. | No C-41 `satcurvergb` stage or live module was found. `ColorBalanceRgb` has overlapping perceptual saturation/brilliance controls, while `ColorZones` curves use different domains and `Velvia` is a different operation; none supplies the defining saturation-indexed saturation/brilliance curves. Evidence: `crates/c41-core/src/iop/{colorbalancergb,colorzones,velvia}.rs`, `crates/c41-core/src/pipeline.rs`, `crates/c41-ui/src/{preview.rs,darkroom/mod.rs}`. | **PORT — S13.** Implement as a Rust pipeline/UI feature; compare/reuse color math only after algorithm-level verification. Do not count overlapping controls as parity or transplant the GTK3 IOP wholesale. |
| [`7076cab1033da4ba4e5e009bbca3b78f76f1c0e3`](https://github.com/darktable-org/darktable/commit/7076cab1033da4ba4e5e009bbca3b78f76f1c0e3) — `satcurvergb: fix histogram calculation` (2026-10-23) | Ancestor of the pinned SHA. Changes histogram sampling to the preview pipeline so its display is not coupled to main-pipeline zoom/pan state. Pinned `src/iop/satcurvergb.c` applies the preview-pipe condition in both CPU and OpenCL histogram paths. | The parent feature is absent in C-41, so there is no corresponding histogram path to patch independently. Include this pinned behavior when implementing `satcurvergb`; upstream diff is a reference for preview histogram semantics. | **PORT with `satcurvergb` — S13.** No standalone `APPLY`; preserve the preview-pipe sampling behavior in the port and test that main-view zoom/pan does not alter histogram sampling. |

**Contrast & Texture resource-error gate.** C-41 `Pipeline::process` currently returns `Vec<f32>` rather than a recoverable result; GPU stagewise errors retry the CPU pipeline. That fallback is valid only if it successfully runs the entire pipeline, including C&T. The module must never return the input unchanged or skip C&T to conceal a resource failure. Before exposing it, provide a fallible C&T stage/pipeline route and propagate failures through every caller that can run a C&T-enabled edit—currently the darkroom preview and Rust raw/non-raw export funnels. Print and lighttable full-preview paths currently use default parameters, so they do not establish active-edit C&T execution; include them if they are later changed to consume the saved edit. If an enabled path cannot surface a render/resource error, keep the kernel gated and do not call it product parity. This requirement supplements (not replaces) measured peak-memory and correct large-image/tiling validation.

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

## Highlight reconstruction: opposed follow-ups and RAW default workflow

All candidates are ancestors of the pinned upstream SHA and absent from the
C-41 source comparison at `fb48c23ce063b3bc1e76b9a51b89f525560f6b1c`. C-41's
Rust opposed/clip workflow was added earlier in `356c54af3a7240b71e46e9d25beb44bb12ec7b81`;
that implementation predates these upstream follow-ups.

| Upstream commit | Authored / committed | Pinned-baseline change | Triage |
|---|---|---|---|
| [`8482e4cff2108b5c64b0fd342896bcd0548feb61`](https://github.com/darktable-org/darktable/commit/8482e4cff2108b5c64b0fd342896bcd0548feb61) — `Code update for opposed highlights` | 2026-09-29 / 2026-09-30 | Refactors opposed-state sharing, changes mask padding/boundary handling and chroma reductions, and removes the lower-quality thumbnail path that produced small-thumbnail color artifacts. The pinned C implementation includes CPU/OpenCL paths; this is not just a parameter change. | **REFERENCE — S14 validation.** The Rust RAW port is CPU-only and always runs its full-frame reconstruction; it has distinct buffer rounding/border logic. Differentially test odd/small dimensions, Bayer/X-Trans edges, clipped-mask boundaries and numerical output before deciding whether any delta must be applied. Do not port OpenCL or the old thumbnail-quality switch as a standalone feature. |
| [`a42d70c8bcb0c8a93e992b41d45c679dd55dd4ee`](https://github.com/darktable-org/darktable/commit/a42d70c8bcb0c8a93e992b41d45c679dd55dd4ee) — `Auto-disable opposed highlights for unclipped raw images` | 2026-10-02 / 2026-10-02 | On a fresh, unaltered RAW with opposed selected, waits for the full-pipe clipped-pixel result, then disables the enabled module and records a history item if no photosites are clipped. The pinned behavior is coupled to upstream's default-enabled module/preset and GUI/history lifecycle. | **PORT — S14.** C-41's Rust `hl_on` default is false and there is no clipped-result/history auto-disable; this avoids work by default but differs in initial module state, clipped-RAW recovery and edit history. Adapt only with a Rust-native default/history design and tests for fresh vs edited RAWs, clipped/unclipped images and stable preview/export results. |
| [`93c8f6efaf6c7a316daf5f1f5b303bb23906930e`](https://github.com/darktable-org/darktable/commit/93c8f6efaf6c7a316daf5f1f5b303bb23906930e) — `Let's cleanup the highlights parameters` | 2026-09-05 / 2026-09-05 | Bumps the C IOP parameter version and removes unused legacy fields while preserving old-version migration. | **REFERENCE** for retained-C parameter/history migration only; the Rust `PreviewParams` blob is independently versioned and does not serialize this C struct. Do not port as a Rust feature. |

Pinned-source paths are `src/iop/highlights.c` and
`src/iop/hlreconstruct/opposed.c`; the auto-disable uses full-pipe results,
image edit state, GUI synchronization and history. C-41 Rust evidence is
`crates/c41-core/src/iop/highlights.rs::reconstruct_mosaic`,
`crates/c41-core/src/rawimage.rs::to_linear_rgba_with_opts`, and
`crates/c41-ui/src/preview.rs` / `darkroom/mod.rs`. Both Rust preview and export
pass `hl_opts`, but highlight reconstruction is disabled by default. The C-41
port has only Clip and Opposed and no raw-history auto-disable. Its source
comments note different Bayer mask/buffer handling; do not infer algorithmic
parity from the shared method name.

## HDR PQ/HLG transfer and display output: selected commits

The following selected commits are ancestors of the pinned SHA and absent from
C-41 source comparison `3c21e36d7dacace646a5b6db716e3e11235c6d5c` (code in that
snapshot is unchanged from earlier documentation-only commits). This is a
partial HDR cluster, not a complete input/output or display-profile inventory.

| Upstream commit | Authored / committed | Pinned-baseline change | Triage |
|---|---|---|---|
| [`4b0878d14a11d016ef37e08b1a7d3a42355319ed`](https://github.com/darktable-org/darktable/commit/4b0878d14a11d016ef37e08b1a7d3a42355319ed) — `color: Fix PQ luminance scaling and HLG display conversion` | 2026-10-05 / 2026-10-07 | Adds shared CPU/OpenCL HDR transfer handling for built-in PQ/HLG profiles: PQ reference white at 203 nits and RGB-dependent HLG OOTF for nominal 1000-nit display. Keeps gamut matrix/LittleCMS operations linear to avoid double transfer and preserves structured HDR signaling in PNG/JPEG XL; adds transfer/CLI tests. | **PORT — S14 candidate, gated.** C-41 has no general PQ/HLG input/output profile or HDR export path. Treat this as a coordinated color-management + format feature, not a transfer-function-only cherry-pick. First define absolute-luminance/reference-white and target-display semantics, HDR metadata preservation, supported codecs and CPU reference tests. |
| [`6a4830844be8f3a08712429dfec5c37bc49afdea`](https://github.com/darktable-org/darktable/commit/6a4830844be8f3a08712429dfec5c37bc49afdea) — `colorspaces: encode PQ and HLG thumbnails the way the pipe does` | 2026-10-21 / 2026-10-23 | Makes lighttable, filmstrip and tethering thumbnails use the same PQ/HLG encoding as the display pipe instead of relying on incompatible built-in ICC shapers. | **REFERENCE** for any future HDR screen/thumbnail support. C-41 deliberately presents through sRGB and has no PQ/HLG display-profile path; this is not a current standalone fix. |

C-41's linear Rec.2020 processing/export seam does not mean HDR transfer or
absolute-luminance support: eligible non-RAW inputs are converted to linear
working space; current SDR exports offer sRGB by default and Adobe RGB (1998)
as a selectable destination. `crates/c41-ui/src/export.rs` only exposes JPEG/TIFF/PNG;
its preview path displays sRGB. `crates/c41-core/src/color.rs` includes PQ
math inside JzAzBz, but that perceptual-coordinate transform is not a general
PQ/HLG profile decoder/encoder or HDR signal path. The retained C tree has
broader HDR paths, but its presence does not wire them into the Rust GTK4
product. See `raw-formats.md` for the product capability matrix.

## Demosaic precision: selected RCD OpenCL differential fix

The selected commit is an ancestor of the pinned SHA and absent from C-41 source
comparison `23c26a1728dd78a5f858bfcdc05449cf3ddf2bc3` (code in that snapshot is
unchanged from earlier documentation-only commits).

| Upstream commit | Authored / committed | Pinned-baseline change | Triage |
|---|---|---|---|
| [`748066d59b22e52a50bd06355a4f9ad52f708e9a`](https://github.com/darktable-org/darktable/commit/748066d59b22e52a50bd06355a4f9ad52f708e9a) — `Increase precision in rcd demosaicer` | 2026-10-18 / 2026-10-18 | Fixes the pinned RCD OpenCL path's CPU differential: initializes/guards edge data, enables contraction to match CPU FMA, and adds an FMA/Newton-refined `fp32_div` where OpenCL's relaxed default division caused larger numerical error. This commit changes `data/kernels/common.h` and `demosaic_rcd.cl`; it does not change the CPU RCD algorithm. | **REFERENCE — S16 GPU follow-up.** Rust `RawImage::demosaic_rcd` is the CPU product path; it has no OpenCL RCD backend, so do not apply an OpenCL helper as a CPU feature. Preserve this commit as the acceptance/reference test if Rust RCD is later dispatched to GPU; compare edge rows/columns and CPU/GPU error over representative CFA patterns. |

C-41's Rust raw decode calls `crates/c41-core/src/rawimage.rs::demosaic_rcd`
from `RawImage::to_linear_rgba_with_opts`; it uses CPU/Rayon and is upstream's
RCD family, but algorithm naming and method presence do not establish full
behavioral parity. The retained C `src/iop/demosaicing/rcd.c` and its OpenCL
kernels are separate from the Rust GTK4 RAW preview/export path. Broader
precision-related commits and CPU reference comparisons remain untriaged.

## Color-picker statistics precision: area-picker follow-up

The selected commit is an ancestor of the pinned SHA and absent from C-41 source
comparison `23c26a1728dd78a5f858bfcdc05449cf3ddf2bc3`.

| Upstream commit | Authored / committed | Pinned-baseline change | Triage |
|---|---|---|---|
| [`1d19780b9c81a42242a17b7da1499156eaaf6930`](https://github.com/darktable-org/darktable/commit/1d19780b9c81a42242a17b7da1499156eaaf6930) — `color picker: add up the rows of the statistics in double on the CPU` | 2026-11-11 / 2026-11-11 | Accumulates each row's float statistics into double totals for 4-channel, no-matrix and 1-channel CPU picker paths, reducing large-area mean error and CPU/OpenCL disagreement without changing per-pixel work. | **REFERENCE — S17 workflow/UI.** C-41's GTK4 picker samples one pixel from the displayed 8-bit render; it has no area ROI or mean/min/max statistics to which this precision fix could apply. If area sampling is implemented, use row-local float plus double row accumulation and test large boxes, low-value sums, hue wrap and CPU/reference agreement. |

C-41 evidence is `crates/c41-ui/src/darkroom/mod.rs` (image click path and
`CachedRender`) plus `crates/c41-ui/src/preview.rs::sample_pixel`; sampling
deliberately reuses the displayed sRGB bytes and
reports one RGB8 value. Upstream implementation is `src/common/color_picker.c`
with view/UI integration in `src/views/darkroom.c`. A shared label or click
handler is not full color-picker parity.

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
Other candidate clusters named in `ROADMAP.md` §5.4—including Contrast & Texture
and broader precision work—remain unclassified until rows are added with
pinned-source and C-41 call-path evidence. HDR has selected transfer/display
commits triaged, not a full feature/path audit. The highlights, Spektrafilm,
DNG and WB clusters are also partial; all follow-up lists are representative,
not exhaustive.
Additional commits for all reviewed features may still need triage. S9 is
incomplete until the remaining capabilities and selected commit deltas are
verified and prioritized.
