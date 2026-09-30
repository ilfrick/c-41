# C-41 vs darktable — parity audit

**First written 2026-08-08. Last refreshed 2026-09-30** (severity 1 re-verified
against the code on 2026-08-11 and not re-walked since; resolved items struck
through rather than deleted, so the record of what was fixed survives). The
2026-09-30 refresh added the **"Known gaps"** section — every severity row was
closed at that point, and the list had no way to describe what it had never
looked at.

> Keep this current. Between 08-08 and 08-11 it went stale enough to be
> actively misleading — it still listed three severity-1 items that had shipped
> on 08-08/08-09, which led to a proposal to rebuild finished work. If you fix
> something in here, mark it in the same commit.

Audit against darktable 5.6.0 (reference screenshot `Schermata_20260721_210438.png`)
and the running app in the KasmVNC container, plus a code sweep of
`crates/c41-ui`. Ordered by **severity**, not by roadmap position: severity 1
is "a user hits this and something is broken or blocked", severity 3 is "darktable
has it and we don't, but nothing misbehaves".

Deadline context: this is the list being worked top-down for **2026-08-20**.
Everything not reached by then stays here, written down, rather than being quietly
dropped. Full darktable parity is **not** reachable by that date and is not the
target — see the note at the bottom.

> **2026-08-31 scope decision:** severity-1.9 (geotagging/neural-restore
> panel) and severity-3.4 (map/print/tethering views) are **deferred for
> later evaluation**, not dropped — see the matching entry in
> `RUST_MIGRATION_PLAN.md`. Migration is prioritizing the A→B FFI port chain
> (src/common shared infra → unblock dependent src/iop + develop/masks loops);
> C is written down, pending an explicit post-infra decision.
>
> **2026-09-26 closure:** the deferred UI parity scope above is now DONE —
> geotagging (u2), print (u1), map list + slippy canvas (u3/u5), tethering
> shell + watch + live capture (u4/u6), neural-restore infra + denoise +
> detail/split + upscale + raw + linear (u7a–u7f). Items 2.7 and 3.4 closed.

> **2026-09-30 — read this before trusting the strikethrough.** Every severity
> row below is closed, and that is *not* a claim of parity: those rows audited
> the product as it stood when the list was written. See **"Known gaps"** further
> down for what they never covered. That section is itself a partial sweep, not
> an exhaustive feature-set audit, and says so.

> **2026-08-21 update:** shadows/highlights (shadhi) wired as the 21st live module,
> joining basicadj (item 2.1) as a parity-2.1 increment — full pipeline Stage +
> Gaussian blur pre-pass + FFI kernel reuse + 7-slider UI + history label.

**Status 2026-08-11 (9 days out):** severity 1 is clear. The remaining work is
severity 2, and within it 2.1 (attaching UI to already-ported processing) is
both the largest gap and the cheapest per unit — each module is an increment.
Per-increment progress is logged in `PROGRESS.md`.

## Where the port actually stands

- **Pipeline (Phase 1): effectively done.** The tracked migration metric — live
  `DT_OMP_FOR` loop sites in `src/iop/*.c` — stands at **10 across 7 files**
  (recounted 2026-08-25 after m4-137/m4-138 replaced colorbalancergb's four and
  colorreconstruction's three; see the table in `RUST_MIGRATION_PLAN.md`, which
  is kept current per increment). Each remaining iop site is GUI-only,
  LCMS-bound pending the lcms2-retirement decision, dead `#ifdef` code, or —
  for colortransfer's k-means — live but non-portable-in-principle (its
  xorshift-seeded atomic accumulation makes any serial port cluster
  differently than some C run; see the RUST_MIGRATION_PLAN row). The
  full `src/` tree holds ~223 further loop sites, but those sit outside the
  pipeline boundary (develop/masks, blend, common helpers, imageio, chart) and
  belong to later phases. **83 of 93** image-operation modules are ported to
  Rust. (An earlier version of this bullet claimed "4 files / 9 total" — that
  figure was long stale and understated the remainder; corrected here.)
- **UI (Phase 3): this is the whole remaining gap, and it is closing module by
  module.** The processing for ~80 modules exists in `c41-core`; as of
  m4-130 slice 3 **33 of them are live in the darkroom panel** (severity-2 row
  2.1 below carries the list), with the unwired remainder being niche or
  blocked-on-plumbing rather than the broad "no controls attached" wall this
  bullet used to describe.

That asymmetry drives the ordering below: the expensive half is built, so most
severity-2 items are "attach a panel to code that already works", not new maths.

---

## Severity 1 — broken or blocking

**All clear as of 2026-08-11.** Every item below is resolved; kept for the record.

| # | Finding | Status |
|---|---------|--------|
| 1.1 | ~~**Side panels cannot be resized.**~~ | **Fixed** `45f0b427` (08-08). Nested `gtk4::Paned`; widths persist to the `darkroom_ui_prefs` table (debounced). Verified 08-11: `lib.rs` uses `Paned`, `LEFT/RIGHT_PANEL_*_PREF_KEY`. |
| 1.2 | ~~**Side panels cannot be collapsed.**~~ | **Fixed** `9fc1e344` (08-09). Header toggle per side + `L`/`R` keys; collapsed state persisted (`*_PANEL_COLLAPSED_PREF_KEY`). A header button rather than darktable's edge triangle — the GNOME idiom, and findable, which was half the complaint. |
| 1.3 | ~~**~915px minimum content width overflows a narrow window.**~~ | **Resolved by 1.1 + 1.2** — panels now shrink and collapse, so the fixed floor is gone. Re-check if a new fixed-width widget lands. |
| 1.4 | ~~**Metadata panel empty for the image selected at startup.**~~ | **Fixed** (08-08 session). `SingleSelection` auto-selects index 0 without firing `selection-changed`; the panel is now seeded explicitly and a `follow_selection` observer keeps preview + metadata in step (`lib.rs:663`). |

## Severity 2 — missing, high value, processing already ported

| # | Finding | Notes |
|---|---------|-------|
| 2.1 | ~~**Most ported modules still have no UI — but the gap is closing.**~~ **Fixed 2026-08-25** — as of m4-130 slice 3 **33 modules are live** in the darkroom panel (exposure, velvia, split-toning, monochrome, sigmoid, sharpen, vibrance, colorize, color correction, color contrast, color zones, levels, white balance, invert, vignette, lowlight vision, graduated density, colour brightness saturation, **basic adjustments**, **shadows/highlights**, **lowpass**, **primaries**, **negadoctor**, **tone equalizer**, **colour balance RGB**, **filmic RGB**, **highlight reconstruction**, **denoise (profiled)**, **bloom**, **tone curve**, **RGB curve**, **base curve**) plus crop/rotate/straighten and the demosaic selector. The m4-11x series adds roughly one per increment. Closed by **lens correction**, landed across three increments: slice 1 hand-wrote the liblensfun FFI (`c41-sys::lensfun`, dev/CI/runtime images ship `liblensfun-dev`/`liblensfun-data-v1`), slice 2 added the `Stage::LensCorrection` pipeline stage, slice 3 (this commit) the darkroom module UI. | basic adjustments (basicadj) wired 2026-08-21: FFI kernel + commit_params + LUT memo + 8-slider UI. `hlcomprthresh` and `clip` deliberately not surfaced (no UI control in darktable / not implemented in kernel). `preserve_colors` LUMINANCE mode fixed from ProPhoto→working-space coefficients (see PROGRESS.md 2026-08-21). **Shadows/Highlights (shadhi) wired 2026-08-21**: Stage::Shadhi with Gaussian blur pre-pass + darkroom_shadhi_process FFI kernel (already ported), 7-slider UI, history label. Gaussian algorithm hardcoded (not bilateral — only recursive Gaussian implemented in c41-core::gaussian); flags hardcoded to UNBOUND_DEFAULT (127), low_approximation to C default 0.000001. `shadhi_algo` dropdown not surfaced. **Primaries wired 2026-08-23** (m4-114): Stage::Primaries with compute_matrix ported from `custom_primaries.c` + `darkroom_primaries_process` FFI kernel, 8-slider UI, history label. **Known divergence: ~10% coefficient difference** vs darktable for the same slider values — darktable reads ICC colorant tags at runtime (`dt_colorspaces_get_primaries_and_whitepoint_from_profile`), which are D50-adapted and differ slightly from the nominal xy tables C41 hardcodes (C41 uses the Lindbloom-derived nominals). C41's choice is kept: it is self-consistent (identity at defaults is exact to 6e-16, vs darktable's approximate ICC round-trip). Documented here; not a bug, a deliberate scope decision. **Slider ranges use darktable's soft ranges** (hue ±20°, purity 0.5..1.5) rather than the hard range (±180°, 0.01..5.0), because the algorithm projects onto the gamut hull — at |hue| > 112° the primaries go collinear and the matrix is singular (coeffs ~1e16). The hard range is still accepted by decode (existing blobs are not rejected). **Negadoctor wired 2026-08-23** (m4-115): Stage::Negadoctor with `process_pixels` ported from `negadoctor.c`; commit_params arithmetic replicated in `to_pipeline` (wb_high/D_max, offset=wb_high_raw*offset*wb_low, black FMA trick, soft_clip_comp=1-soft_clip) with a parity test pinning the derivation against darktable `commit_params` (src/iop/negadoctor.c:239-267). 15-slider UI in `darkroom/mod.rs` with EV-scale exposure slider (2^EV conversion at the widget boundary, darktable gui_changed:964 / gui_update:988), B&W Dmin G/B visibility gating (toggle_stock_controls:388-410), Dmin R→G/B mirroring in B&W mode (gui_changed:953-957), and Black slider step 0.001. HISTORY_encode_VERSION intentionally pinned at 5 (container unchanged; PreviewParams carries its own per-entry version byte). Note (updated 2026-08-24): the curve-based modules waited on a **curve-editor widget**, which now exists (`darkroom::curve_editor`, m4-122) — tone curve is live on it and RGB/base curve can reuse it; colorzones shipped with sliders only for that historical reason. **Tone equalizer wired 2026-08-24** (m4-116): Stage::ToneEqual carrying the nine raw EV gains; `solve_weights` ports `build_interpolation_matrix` + `pseudo_solve` (normal equations + Cholesky–Banachiewicz, choleski.h); thread-local memoised 80 001-entry correction LUT via the existing `darkroom_toneequal_build_lut` FFI; pixel pass = norm-2 luminance mask + apply-LUT. Nine −2..+2 EV sliders labelled with darktable's params descriptions + EV positions (gui toneequal.c:3205-3213). Placed at v50_order pos 24.0 (before graduatednd 25 — pinned by the ordering test). **Two documented scope deviations:** (1) runs `details == DT_TONEEQ_NONE` ("preserve details: no") whereas darktable's shipped default is `DT_TONEEQ_EIGF` (guided filter) — so out-of-box behaviour differs (global per-pixel exposure-channel curve vs spatially-blurred local mask); the guided-filter/surface-blur pre-passes are not ported, and with them go the smoothing/feathering/blending/quantization controls and mask display. (2) all-zero gains render exact identity in C41, where darktable applies the fitted RBF curve with up to ±0.7% deviation from unity (numpy-lstsq cross-check; pinned by `solve_weights_flat_unity_at_default_gains`). On a hypothetical failed least-squares solve C41 returns zero weights (uniform ×0.25 floor, visibly broken) rather than mirroring the C's silent mis-read of unsolved exp2(gains) as weights (`commit_params` ignores `pseudo_solve`'s FALSE return); unreachable for the fixed σ=√2 SPD basis anyway. Senior review: APPROVE-WITH-FIXES, all findings addressed (numpy cross-check of weights/LUT values; zero new clippy warnings; Docker CI green) | **Colour balance RGB wired 2026-08-24** (m4-117): Stage::ColorBalanceRgb carrying the prebuilt commit output (`CbRgbData::from_params`, port of `commit_params` colorbalancergb.c:1087, including the hue-indexed 512-entry gamut LUT built once per render — the dt-UCS build marches the gamut boundary 25 600 times), placed at v50_order pos 41.5 between basicadj and shadhi (before sigmoid). Runs in the buffer's working space via `process_in_space` with per-space RGB↔XYZ-D65 converter pairs and a matching gamut LUT (Rec.2020 raw path / linear sRGB non-raw path, new `color::srgb_to_xyz_d65` + `SRGB_TO_XYZ_D65_T4`) — the kernel originally hardcoded Rec.2020 and would have graded JPEGs under mismatched primaries; caught by self-audit before review. 29-slider UI with darktable's soft ranges (colorbalancergb.c:1786-1990) plus a saturation-formula dropdown (JzAzBz/dt-UCS). Identity gate: enabled-at-defaults emits no stage (darktable's defaults are a neutral edit; the C would run its near-no-op gamut map — same deviation class as other identity-gated modules). Documented behaviour: cross-space results legitimately differ near/over the UCS gamut-knee because each space's LUT encodes its own RGB cube (as in the C, LUT from `work_profile->matrix_in`); hue/chroma edits therefore grade slightly differently on raw vs non-raw previews by design. GUI-only C fields (mask checker display) omitted. Senior review APPROVE-WITH-FIXES (performed by fork subagent acting as senior reviewer after four API-402 failures of the named agent — substitution documented in PROGRESS.md); single doc-staleness finding fixed. **Filmic RGB wired 2026-08-24** (m4-118): Stage::FilmicRgb carrying `FilmicData::from_params` (port of `commit_params`' spline/targets arithmetic — scene grey 18.45 %, display black/white pinned at darktable's defaults) feeding the existing `darkroom_filmicrgb_v5` FFI kernel; V3 spline branch (contrast·dynamic-range slope, latitude interpolation, balance shift, min-contrast clamp) with POLY_4 quartic toe/shoulder solved in f64 via `gauss_solve` (gaussian_elimination.h port), METHOD_MAX_RGB=1 hardcoded per the v5 process code. Placed at iop_order v50 pos 46.0 (sigmoid 45.3 < filmicrgb < colisa 47.0 — pinned by the ordering test). Runs in the buffer's working space via `process_in_space`; because C41's pipeline is D65-referenced where C's is D50, both CAT16 legs drop out algebraically exactly (input = XYZ_D65→LMS ∘ rgb_to_xyz_d65, output = xyz_d65_to_rgb ∘ LMS→XYZ_D65). **Ships OFF** — documented deviation: darktable auto-enables filmic through its scene-referred default preset plus reload_defaults auto-exposure, neither of which C41 runs. 7-slider UI with darktable's hard ranges (soft ranges deliberately not modelled). Fixed en route: `SRGB_TO_XYZ_D65_T`'s G row had carried B→Z's value (0.0721750 vs canonical 0.1191920) since m4-117 — caught precisely because the new matrix-composition round-trip test refused to pass in linear sRGB; verified against colorspaces.c and by numpy-inverting the stored inverse. Also recorded: `debug_assert!(gauss_solve(..))` silently skips the solve under --release (arguments are unevaluated) — solves now run unconditionally with a separate assert on their result. Senior review: SHIP (fork subagent standing in for fricktrade-architect, API 402 again), zero BLOCKER/MAJOR findings across all eight checklist dimensions; two INFO notes (per-band ~100-flop matrix rebuild; f64-widened spline nodes) explicitly non-blocking. **Highlight reconstruction wired 2026-08-24** (m4-119): the first live module that is NOT a pipeline stage — it runs on the raw CFA mosaic before demosaicing (darktable's temperature → highlights → demosaic order) and threads through `RawImage::to_linear_rgba_with(method, hl)`; any change re-decodes the raw. Enabler: `normalize_cfa`/`normalize_xtrans` now carry over-range photosites (`.max(0.0)`, previously hard-clamped at load so there was nothing to reconstruct); the legacy `None` decode path clamps first and stays byte-identical (pinned by an exact-RGBA test), hl ships OFF, NaN now floors to 0.0 (improvement). Driver = `highlights::reconstruct_mosaic` over the ported unsafe opposed kernels (`mask/dilate/chroma/output_raw`) + safe clip pass; WB applied on the mosaic pre-demosaic with green-normalised ratios (deviation vs C's raw-coefficient comparison — scaling cancels; keeps hl-on/off balance identical); post-demosaic WB skipped on the hl path. UI: params v20 (+2 bools/+1 f32, blob 885), hand-rolled row (enable + method DropDown clip/opposed + debounced threshold slider) because changes need `spawn_decode` not `render_preview`; `PreviewCtx` gained decode_path/decode_method for that; row hidden for non-raw (darktable forces CLIP there — no-op on sRGB). Export parity via `params.hl_opts()` in both render_export paths. Documented deviations: E-colour (index-3 second green) Bayer patterns take the clip fallback (kernels slice clips[0..3]/[u8;3]; upstream C reads clips[3] harmlessly); only opposed (C default) + clip methods wired — LCh/laplacian/segmentation unwired. Senior review: SHIP (fork subagent standing in for fricktrade-architect, API 402 again), zero BLOCKER/should-fix findings; nits recorded in PROGRESS.md. **Denoise (profiled) wired 2026-08-24** (m4-120): Stage::DenoiseProfile running `wavelets_denoise` (denoiseprofile.c wavelets path) — VST precondition → 7-scale edge-avoiding à-trous decompose (new `c41-core::iop::eaw` port of eaw.c: `eaw_dn_decompose`/`eaw_synthesize`/`dn_weight`/float `fast_mexp2f`) with BayesShrink thresholds (`thrs = adjt·σ_band²/std_x`, adjt = 8.0 exactly under the flat default force curve since Catmull-Rom through collinear 0.5 nodes gives force²·4 ≡ 1) → soft-threshold synthesize → residual add → backtransform; placed at v50_order pos 9/10 between temperature and exposure (pinned by the ordering test); alpha lane restored from input because the VST kernels garble lane 3. **Documented deviations:** wb=[1,1,1] (buffer arrives post-WB so C's `wb[i] *= strength·compensate·in_scale` scaling is carried by our `wb_s=[s,s,s]` passed to the kernels — matrices stay on pre-strength wb per C order; senior-review finding: without wb_s Strength was completely inert in RGB mode where it is the *only* carrier, and the Y0U0V0 bias term ignored strength — both fixed with regression tests); generic Poissonian profile a=1e-4/b=0 instead of noiseprofiles.json (consequence: transformed-space variance stays ≪ σ_band², thresholds saturate at the 1e-6 floor, and strength alone does not modulate output — identical to darktable given the same profile; strength bites only with real per-camera profiles); use_new_vst=TRUE path only; wavelets mode only (no nlmeans); buffer-dims supp0; unified clamped indexing refusing the C's narrow-image row-bleed; natural row order vs dwt_interleave_rows (threshold-input summation order only). UI: params v21 (+2 bools/+3 f32, blob 899), module_expander row (enable + Y0U0V0/RGB DropDown + Strength 0.001..4.0 / Preserve shadows 0..1.8 / Bias ±10 matching C soft ranges), history label "Denoise (profiled)". Export inherits via to_pipeline. Senior review SHIP-WITH-FIXES (fork subagent standing in for fricktrade-architect, API 402 again): findings 1+2 (shared wb_s fix) applied with two regression tests, doc-range nit fixed; verified strength-inertness-under-generic-profile is faithful C behaviour, not a port bug. **Bloom wired 2026-08-24** (m4-121): Stage::Bloom running `bloom::process` — the bloom.c process() chain (gather Lab L·scale above threshold → `dt_box_mean` ch=1, 8 iterations → screen-blend into L) — with the enabler being the first port of the box-mean kernel: new `c41-core::iop::box_filters` porting `_blur_horizontal`/`_blur_vertical` five-phase sliding-window semantics exactly (window∩bounds shrinking edges via hits-counting; sub-before-add bulk order; plain f32 accumulation matching the non-Kahan ch=1 dispatch), verified against a naive O(n·r) reference across shapes/radii/iterations plus hand-computed edge rows. Derivation chain replicated with C int-truncation semantics: rad = int(256·(min(100,size+1)/100)), radius = MIN(256, ceil(rad)) (scale/iscale ≡ 1 whole-frame, kept as a documented parameter slot), blur radius = (2r+1)/2 ≡ r, scale = exp2(min(100,strength+1)/100). Placed at iop_order v50 pos 61 between velvia 57 and colorize 62 (pinned by the ordering test); NOT pixel-local (whole-frame serial, like Lowpass/Sharpen); RGB↔Lab sandwich in the stage per Colorize precedent. UI: params v22 (+1 bool/+3 f32, blob 912), three-slider row Size/Threshold/Strength 0..100 with darktable defaults 20/90/25, history label "Bloom". Identity gate is flag-only and *correctly* so: scale ≥ exp2(0.01) > 1 means even strength=0 alters output once anything passes the threshold, so there is no neutral-while-enabled. Senior review: SHIP (fork subagent standing in for fricktrade-architect, API 402 again) — zero BLOCKER/MAJOR/MINOR findings; two comment-level nits applied. **Tone curve wired 2026-08-24** (m4-122): Stage::ToneCurve applying a user-drawn Lab L curve — first slice ships the L channel only (a/b channel tabs deferred, with darktable's defaults carried in params: identity 3-node monotone-Hermite ab curves). Two enablers landed together: `c41-core::curve_tools`, a port of src/common/curve_tools.c (the UFraw-lineage V1 sampler used by tonecurve/basecurve/atrous/denoiseprofile — Thomas solver over the Burkardt D3 collapsed layout, natural/first-derivative boundary conditions, MONOTONE_HERMITE computing Fritsch-Carlson tangents but evaluating via catmull_rom_val per draw.h's spline_val dispatch, unknown type falling back to box-diagonal), and the interactive `darkroom::curve_editor` widget (DrawingArea canvas with darktable's basic gestures: press-to-grab/drag anchors, click empty space to insert up to 20 nodes, double-click interior removal, endpoints pinned at x=0/1, interior x kept strictly between neighbours). The drawn curve IS the applied curve — both go through `curve_data_sample`. `to_pipeline` replicates commit_params: sample at 65536, scale ×100 (L) / ×256−128 (ab) unconditionally, then AUTOMATIC_RGB autoscale re-derives t_l in place via the ProPhoto round-trip (AUTOMATIC_XYZ ported alongside), signed-CLAMP LUT-index semantics preserved via i64 widening, and unbounded tails extrapolated by estimate_exp over x∈{0.7..1.0}·xm with mirrored lookups. preserve_colors=AVERAGE(3), unbound_ab=true — C defaults. Placed at iop_order pos 48 (colisa 47 < tonecurve < levels 49, pinned by the ordering test); pixel-local (pure LUT lookups → rayon band path); working space Rec2020/linear-sRGB with Lab sandwich. UI: params v23 (+1 bool/+7 f32/+40 interleaved node coords, blob 912→1090), interpolator DropDown (cubic spline/Catmull-Rom/monotone Hermite), history label "Tone curve". Identity gate is flag-only like Bloom (a moved node alters output even before the flag flips is not possible here, but an enabled-at-defaults curve is exact identity only while untouched — gate matches Bloom precedent). Senior review FIX FIRST (fork subagent standing in for fricktrade-architect, API 402 again): both findings were one-liners and are fixed — (1) ibcend==1 boundary diagonal carries Δt/3, not 1 (curve_tools.c:324); (2) insert_node refuses clamped-duplicate-x inserts — the reviewer's hazard was exact duplicates, and my own regression test then exposed that collision checks on the *raw* x also miss out-of-range clicks that clamp onto an endpoint's column, so the check now runs on the clamped coordinate. NITs consciously accepted: hit tolerance is isotropic in curve units (≈14.7px horizontally on the 240px widget vs 10px vertically); NaN in a decoded blob falls back to the box-diagonal rather than being sanitised (unreachable from the editor, which clamps every write). **RGB curve wired 2026-08-24** (m4-123): Stage::RgbCurve applying per-channel user-drawn curves directly to the working RGB lanes (IOP_CS_RGB — no Lab sandwich; darktable's rgbcurve likewise never leaves RGB). Reuses `curve_tools::curve_data_sample` plus tail extrapolation (`_generate_curve_lut` rgbcurve.c:1671–1742: xm = last node x, samples at {0.7,0.8,0.9,1.0}·xm via `dt_iop_estimate_exp`; LUT index CLAMP((int)(x·0x10000)) semantics preserved incl. negative-saturating casts landing at index 0 like C's CLAMP). All three process() branches faithful: MANUAL → per-channel tables; AUTOMATIC+preserve==0 ("none") → R table applied to all three lanes; AUTOMATIC+preserve≠0 → single ratio curve(rgb_norm)/rgb_norm scaling all lanes equally with lum≤0 passthrough; alpha passes through. C's own process() already delegates into the Rust FFI kernel `darkroom_rgbcurve_process`, so preview and production share one implementation — pinned byte-equal by test including the v=1.0 boundary. Placed at iop_order v50_order pos 42.0 between colour balance RGB 41.5 and rgblevels 43.0 → emitted after Colour balance RGB (pinned by the ordering test); pixel-local (rayon band path); identity gate flag-only (`rc_identity = !rc_on`, Bloom/ToneCurve precedent). UI: params v24 (+1 bool/+8 f32/+120 interleaved node coords, blob 1090→1603) — append-only layout keeps tc anchors at float 224 (an early draft put the rc scalars inside the main float list, shifting them +8 and breaking six tests across persist/history/dialogs before it was caught — kept here as the cautionary tale); editor generalised into a shared multi-channel builder (`multi_curve_area`) now serving both tone curve (1 channel, amber) and RGB curve (3 channels, R/G/B-coloured strokes) with per-channel gesture routing, channel selector, interpolator combo writing all three types (mirrors `interpolator_callback`), and linked/independent mode combo toggling preserve-colors row sensitivity (`_rgbcurve_show_hide_controls` analogue, semantic superset: channel selector stays visible-but-inert in linked mode). History label "RGB curve". Documented deviations: compensate_middle_grey path omitted (default-off; needs work-profile matrix plumbing), work-profile early-return cache in commit_params omitted (C's is pure memoization — commit_params builds nothing), picker_scale work-profile luminance is GUI-only; corrupted-blob spline types degrade to curve_tools' box-diagonal fallback (safe, not bit-faithful to C's default arm). Senior review found ONE BLOCKER: the initial cut placed the stage after Levels citing iop_order pos 50.5 — but 50.5 lives in legacy_order (iop_order.c:152), not v50_order, which puts rgbcurve at 42.0; every other C41 pin matches v50_order, so the placement was wrong and is fixed in m4-123b. (Process note: the m4-123 commit was made prematurely by the review fork subagent before its own finding landed, and its PROGRESS entry claimed 'APPROVE, zero blockers' — both superseded here; published history was left unrewritten.) Three nits recorded (truncation cast mirrors the existing tc pattern; blob-type fallback; linked-mode channel selector visibility). **Base curve wired 2026-08-24** (m4-124): Stage::Basecurve at iop_order v50_order pos 44.0 (rgblevels 43.0 < basecurve 44.0 < sigmoid 45.3 — pinned by the ordering test), completing the curve trio. Single-LUT semantics per C's `const int ch = 0`; LUT building delegated to rgbcurve's refactored `CurveLut`/`build_single_lut` so tone/RGB/base share one sampler + estimate_exp tail. Both process paths wired: plain curve (pixel-local, rayon band path) and exposure fusion > 0 (`process_fusion` — full Laplacian-pyramid blend over the Phase 2z+56 kernels; NOT pixel-local, routed serial via the fusion==0 locality gate). Working-space luminance solved without ICC plumbing: C's kernel consumes only row 1 of matrix_in and Bradford adaptation preserves that Y row, so new sRGB/Rec2020 D65 Y-row constants substitute exactly (T4 matrices are transposed — passing them directly would have supplied the G column; caught pre-review). UI: params v25 (+1 bool/+6 f32/+40 node coords, blob 1788), HISTORY_ENCODE_VERSION 5→6 with old stacks still loading; third client of `multi_curve_area` (single grey channel) with preserve-colors + fusion combos and fusion sliders whose rows toggle visibility with the mode (C gui_init/gui_update). Documented deviations: no interpolator dropdown (C has none for this module; bc_type still drives drawing), log-base graph scale omitted (display-only), corrupt-blob stops/bias unclamped (sibling posture). Senior review APPROVE-WITH-NITS (fork standing in for fricktrade-architect, API 402 again; read-only mandate enforced): MINOR doc slip fixed; two NITs recorded (unclamped corrupt-blob scalars; endpoint-pinning closure duplicated across two emission blocks until a third user appears). **Lens correction wired 2026-08-25** (m4-130 slices 1–3) — closes this item: `Stage::LensCorrection` calls the distro liblensfun through hand-written FFI, and slice 3 adds the darkroom module UI. Gear picking: camera DropDown fed from `lf_db_get_cameras` (mount-filtered, sorted, maker-prepended labels via `lens_label`) and a dependent lens DropDown from `list_lenses(mount)`; both commit **structured (maker, model) pairs** into `main.darkroom_lens_choice(imgid PK, 4 TEXT columns)` exactly as the DB spells them. Resolution is exact byte-equality enumeration over the full DB because probes showed `lf_db_find_lenses_hd` is a lossy did-you-mean scorer against this database (returns nothing for ~half of real entries fed their exact names — it scores the Model field only, so prepended-maker labels break it — and raw exact pairs hit only 142/292; darktable sidesteps this by keeping live `lfLens*` pointers on its menu items). Duplicate Maker/Model rows across mount families resolved by mount filter + closest-crop-factor tie-break (first-match-return broke Sigma's duplicated entry). Numeric params (corrections bitmask combo, target geometry, focal/aperture/distance sliders, scale + auto-scale button measuring the pristine buffer dims like C's img->p_width/p_height) live in PreviewParams blob v26 (1788 → 1814 bytes); corrections/geometry dropdowns write value tables built FROM the core constants — construction pins index→value (review blocker: the first draft wrote the combo INDEX into the bitmask, so "all" stored TCA-only). Seed-from-metadata fills the camera only when nothing is persisted (a saved pick wins even when unresolved); a `lens_syncing` re-entrancy guard stops nested dropdown notifies multi-committing. Distance defaults to 1000.0 = C's actual no-EXIF fallback (lens.cc:3455); focal/aperture have no EXIF introspection in C41 and sit at neutral values. Documented deviations: (1) RESOLVED same day in **m4-131** — the warp runs as a pre-pass on the FULL decoded frame before crop/straighten (darktable's placement): export warps between decode and `geometry.apply`, the preview caches the warped frame (`PreviewCtx::lens_frame`, keyed by decode generation + enable flag + gear Arc identity + numeric params) and composes `base` over it, `render_preview` revalidates centrally so every trigger invalidates without per-site wiring, and the raw pipeline builders drop exactly the lens stage while non-raw keeps it (its sources are never cropped — there a full-frame in-pipeline warp IS darktable's placement). Cropping now commutes with the correction bit-exactly, pinned by an export test asserting a cropped export equals the same region of an uncropped export across all pixels/channels with a non-identity correction (precondition-guarded). Residual nuance: denoiseprofile (pos 9/10) consequently runs after the warp rather than before it — spatial denoise and a coordinate warp commute to first order; (2) gear identity lives outside the HistoryStack — undo/redo reverts numbers, not swaps, and Reset resurrects the persisted choice (dt keeps the strings in params; closing needs an optional sidecar per history entry); (3) focal slider not clamped to the resolved lens range (dt forces primes to MinFocal); (4) choice is purely user-picked — no EXIF auto-population (C41 reads no EXIF lens tags). Senior review APPROVE-WITH-FIXES (run by a fork subagent on the session model stealth/ox-alpha standing in for fricktrade-architect after API-402 failures, per user direction that the reviewer run on ox-alpha): blocker index-vs-value bitmask + majors (false history comment, nested-notify commits, seed overwrite) fixed; fabricated "C default" citations removed. **Local contrast wired 2026-08-26** (m4-152): Stage::LocalContrast calling the faithful locallaplacian.c port (slice 1, commit ae0bcd97d1, 0-ulp review-verified), placed at v50_order pos 54 between shadhi 50 and colorcorrection 55 (pinned by the ordering test). Only the local-laplacian engine is wired — it is the C $DEFAULT:1 mode and both shipped presets use it; the bilateral-grid mode is future work, so the mode combobox is not surfaced. Identity gate is flag-only and *correctly* so: curve_scalar is a bezier knee over ±2σ with wings of slope shadows/highlights outside, which has no neutral pass at any slider combination. Param mapping mirrors bilat.c process() positional call exactly: midtone→sigma, sigma_s→shadows, sigma_r→highlights, detail→clarity; like the C LL branch, pipeline scale is ignored and the engine runs untiled. Params blob v27 (+1 bool/+4 f32, 1814→1831). UI ranges = C introspection intersected with the LL-mode GUI overrides (shadows/highlights hard-max 2.0, midtone 0.001..1.0 step .001 per digits-3, detail −1..4); slider order matches C widget creation order. History label "Local contrast", style group registered (MODULE_GROUPS 34). Known cosmetic divergence: darktable formats Detail/Highlights/Shadows as % with offset/format (reads 0–500 %/0–200 %); C41 sliders show raw unitless values — noted for a future slider-format pass. Senior review: SHIP (fork subagent standing in for fricktrade-architect after API-402 failures), five P2 findings all addressed (memory-figure doc corrections measured via the memory_use function — ~90–120 B/px typical, ~141 B/px large; curve-mechanism wording; new-in-diff clippy instances accepted as house-style-consistent chunks_exact loops matching sibling tests; shadows/highlight asymmetry regression test added; % units recorded here).
| 2.2 | ~~**No history stack in the lighttable.**~~ **Fixed 2026-08-25** — a *History* section at the foot of the right panel's metadata page: **Copy** grabs the selected image's saved edit (params blob + undo-stack), **Paste** writes it onto the current image replacing BOTH its rows (darktable's replace-the-stack paste semantics; toast names the source file), **Discard** clears both rows in one transaction behind a confirm dialog, and a one-line readout ("no saved edits" / "N-step edit stack") tracks the selection. Backed by new `persist::discard_history`. | Limits/deviations: actions bind to the currently-selected image (the lighttable is `SingleSelection`) rather than darktable's multi-image "actions on selection"; confirm-before-discard is deliberate (irreversible, and upstream assumes a keyboard-driven multi-select flow). A copy whose source has params but no stack row seeds a fresh one-step stack on paste, so the target can never render under a stale foreign stack. The clipboard is process-local like darktable's. **Multi-select fan-out closed 2026-09-27 (u11):** Export (`win.export-selected` + toolbar button) and Print (`win.print-selected`) now resolve through the shared containment rule (`panels::edit_target_paths`, widened to `pub(crate)` — one rule, three consumers) — set-contains-cursor fans out over the whole set (Export batch and Print multipage already handled N>1; first-stem default PDF name pinned), else cursor alone; empty stays a no-op. Rating/colour keys already fanned out; copy/paste history stays single (deliberate — a clipboard paste onto N targets would silently multiply one edit; out of scope). Geotagging panel, neural Run, map/tether/slideshow, darkroom open stay single by design. Known nit (accepted): the step count includes the "Original" seed entry, so one recorded edit reads "2-step edit stack". |
| 2.3 | ~~**Metadata is read-only.**~~ **Fixed 2026-08-19** — a *Metadata editor* section in the right panel with darktable's five default fields (title, description, creator, publisher, rights, in its `display_order`). Writes darktable's own `main.meta_data` via `c41_db::metadata`, so values are visible to the C app. Edits commit on Enter/focus-out, Escape reverts, and a `close-request` flush catches the last field. | ~~Two gaps left~~ **One gap left** (XMP closed 2026-08-25, m4-142): every successful metadata save now also synchronises `<filename>.xmp` beside the image, darktable's `dt_image_synch_xmps` behaviour (`src/libs/metadata.c:393`) — new `c41-ui::xmp` module on quick-xml. **Merge-preserving by construction**: an existing sidecar is rewritten in one streaming pass that copies everything non-target through verbatim (foreign `rdf:Description` blocks, comments, CDATA, packet PIs), replaces only the five Dublin Core properties (catalogue wins, blank deletes — upstream's convention), and an existing file that fails to parse as RDF/XMP is reported as a failure and left byte-identical rather than overwritten. Namespace-correct across prefix spellings (targets recognised by DC-URI binding; output reuses the document's DC and RDF prefixes, declaring `xmlns:dc` only when absent); shapes match exiv2/darktable (`rdf:Alt`/`x-default` for title/description/rights, `Seq` creator, `Bag` publisher); writes are same-directory tmp+rename atomic. Deviations recorded: editing a multi-value creator/publisher sidecar collapses it to the editor's single string (inherent to darktable's one-string metadata model too); sync runs synchronously on the UI thread after each save (sidecars are small local files; matches the existing persist-write posture); only these five fields sync — darktable also mirrors other namespaces into sidecars through its full exiv2 stack, which stays out of scope. Remaining gap (narrowed 2026-08-25, m4-144): the lighttable now has a real multi-selection — ctrl-click toggles, shift-click ranges, ctrl+A over the collection, Esc clears, bottom-bar count — with upstream's active-image-vs-selection split kept (the `SingleSelection` cursor still drives preview, darkroom entry and this panel). Keyboard star-rating and colour-label actions fan out over the whole selection (per-image toggle semantics for labels, absolute set for ratings; one serialised DB write per image under its own per-path lock). Those keyboard/click rating/label paths were catalogue-only writes (**sidecar sync closed 2026-08-26, m4-153** — see end of this entry). The metadata editor panel now fans out over that selection too (**closed 2026-08-26, m4-145**): an edit commits to every selected image — `dt_metadata_set_list`-over-the-selection semantics, one transaction per image — when the displayed image is inside the set, else to the displayed image alone. That containment rule (senior-review MAJOR-2) keeps the panel from ever writing to a set that excludes the image on screen: the grid cursor can leave the selection via native keynav or preview stepping, and upstream avoids the same trap only because its `last_act_on` drives both display and write. A dim hint under the entries states the target count while more than one is bound; uncatalogued skips are reported ("Saved … for K of N images"), never dropped silently; XMP sidecar failures aggregate into one line per commit; and the close-request flush groups entries sharing an identical target list into ONE fan-out write + sidecar pass. Remaining caveat: keyboard rating/colour fan-out stays catalogue-only (no sidecar sync there yet). The other four upstream keys (notes, version name, image id, preserved filename) are internal or off-by-default and stay unexposed.

**Caveat closed 2026-08-26 (m4-153): rating/colour-label sidecar sync.** All four lighttable write paths — star click, keyboard rating, colour click, colour key — now mirror into `<filename>.xmp` in the same serialised closure that writes the catalogue, under the image's existing per-path lock so catalogue and sidecar mutate together atomically. Same merge-preserving `c41-ui::xmp` machinery via a second payload behind the one streaming rewrite: `Xmp.xmp.Rating` scalar text = `flags & DT_VIEW_RATINGS_MASK` (0–5; verified `image.c:600-604`, `exif.cc:5149-5151`; no reject action exists here), and `Xmp.darktable:colorlabels` as an `rdf:Seq` of decimal colour indices written only when at least one label is set (`exif.cc:5066-5087`) — mask 0 DELETES the property rather than emitting an empty Seq. Partial payloads leave the other property untouched (a star click must not delete labels); owned properties are also recognised and dropped in ATTRIBUTE form (exiftool's `xmp:Rating="3"`); prefix spellings follow the document's bindings including the RDF container names (`r:Seq`/`r:li` under an `r:`-bound document); a payload that emits nothing never creates a phantom packet or injects an empty Description; unparseable documents stay byte-identical with a failure reported, as before. Deviation recorded (same as m4-142's): darktable syncs these through its full exiv2 stack on every XMP-writing path; C-41 mirrors exactly the fields it manages. |
| 2.4 | ~~**No styles.**~~ **Fixed 2026-08-18** — data layer `7a6a9744c1` (`c41_styles` table + DAO), UI in the same-day follow-up: a Styles section in the right panel (save the selected image's edit under a name, apply it, delete it, with confirm-on-overwrite/delete). | No limits left in the lighttable panel. Per-module styles closed the last one (**2026-08-26, m4-149**): the save dialog lists checkboxes for all 33 module groups (`stylemodules::MODULE_GROUPS`, byte-identical to the history labels); an all-ticked save stores a NULL `modules` column and applies exactly as before (wholesale replace — also the shape of every pre-149 row), while unticking any group makes the style partial and applying MERGES those groups over each target's saved edit (darktable's per-item behaviour; a target with no saved edit merges onto defaults; unknown group names from newer builds are skipped), and partial rows show "N of M modules". The other earlier limit was closed (**2026-08-26, m4-146**): Apply fans out over the lighttable multi-selection under the containment rule (`edit_target_paths`), counts writes against attempts, and holds the crate-standard 3s `busy_timeout`. Styles can also be applied from inside the darkroom (**2026-08-26, m4-151**): a Styles section pinned below the module list (darktable keeps the same lib module in the right-centre panel at position 599, addable to darkroom since 4.9), dropdown + Apply mirroring the Reset handler — explicitly labelled history entry, module sliders rebuilt so they show the applied values, one autosave settle — through the same `stylemodules::apply_style` semantic the lighttable path calls, so the two surfaces cannot drift. The former clobber hazard was investigated (m4-150) and is unreachable under current wiring: darkroom pages are built fresh per activation and flush on hide, while the darkroom surface mutates its own live params directly; it would only re-arise if darkroom pages were cached across visits. |
| 2.5 | ~~**No colour-label quick filter in either bar.**~~ **Fixed 2026-08-25** — five colour circles in **both** the lighttable top bar (left of centre, beside the preset dropdown) and the bottom bar (right of the star filter), plus the left panel's checks + Any/All toggle — three mirrors of ONE canonical quick-filter state (`lighttable::COLOUR_MASK`/`COLOUR_ALL`), driven through the m4-97c filter-observer bus so a click anywhere re-dots everywhere and reloads the current view once. The old collection-selector semantics (`lighttable_load_by_color_mask` GROUP BY query with mutual-exclusion clearing) are deleted; colour now composes ON TOP of folder/tag/search like the star filter (`AND ((SELECT COUNT(DISTINCT cl.color) FROM main.color_labels …) >= 1` for any / `= N` for all). Token-persisted (`off`/`any:M`/`all:M`) under pref key `colour_filter`, restored at startup before any control builds. | Deviations: darktable's circles also *set/clear labels* in some views and drive its "select by colour" — ours only filter (the labelling UI remains the metadata panel's chips); the Any/All combine toggle lives in the left panel section rather than on the bars (darktable hides this choice entirely — its bar circles OR, its collection module ANDs). Persistence is one app-level observer, not per-widget, so N mirrors still write once per change. The demo db now carries an empty `color_labels` table so the fragment can never fail its query there (senior-review MAJOR-1). |
| 2.6 | **Fixed** (final slice 2026-08-25) — ~~no *import* module in the left panel~~ (Import section added at the top, and an Export section at the foot of the right panel; both activate the existing `win.import` / `win.export-selected` actions rather than duplicating the dialogs). **Collection filters expander, first slice 2026-08-25 (m4-128):** a "Collection filters" section in the left panel with darktable's three stock aspect presets (square / landscape / portrait, filtering.c:308-322) as one dropdown, driving a canonical compose-on-top quick filter through the observer bus like rating/year/colour; token-persisted under pref key `aspect_filter`, restored before any control builds. Deviations: exact integer comparisons on import-time width/height instead of darktable's ±0.5 %-tolerance snapped float column (`dt_usable_aspect`; 0×0 probe failures and NULL dims match no aspect), and sensor dimensions rather than developed `p_width`/`p_height`. **Arbitrary rule stack, second slice 2026-08-25 (m4-134):** N rules of (property, contains/excludes, value) joined by AND / OR / AND NOT, composing on top of the collection through the same splice point as every quick filter; injection-safe `LIKE` composition (`like_literal`: escaped backslash first, then `%`/`_` under `ESCAPE '\'`, quotes doubled) because loaders splice SQL strings; strictly left-associative parenthesized composition so OR can't swallow later terms; blank-valued rules stay visible state but are skipped by the composer; row-0 combinator hidden (nothing precedes it) and elided from tokens for stability across first-row deletion; persisted under pref key `rule_stack` (percent-encoded, MAX_RULES=16, lenient decode). **EXIF numeric rule properties, third slice 2026-08-25 (m4-135):** the four numeric properties darktable's collection rules filter on — exposure (fraction syntax `1/60` accepted), aperture, ISO, focal length — are probed at import (kamadak-exif 0.6.1 over TIFF-family containers; published crate name `kamadak-exif`, lib target `exif`) and stored as nullable REAL columns on `images` (pragma-guarded idempotent ALTERs; demo DDL carries them too), selectable in rule-stack rows with kind-partitioned comparators (text keeps contains/excludes, numeric gets < ≤ = ≥ >). Semantics: NULL never satisfies any comparison (unprobed/pre-migration images stay out of numeric-rule results); an unparseable value (incl. non-finite spellings like `inf`/`NaN`) or a wrong-kind comparator makes that rule inert rather than erroring or narrowing; comparator token indices stable across the slice so pre-m4-134 persisted stacks decode unchanged, wrong-kind comparators normalised at parse. Deviations: only files kamadak-exif parses get values (dt reads via exiv2 across its whole format matrix); dt's zero-backfilled EXIF can match where our NULLs stay excluded. **Named collection presets, final slice 2026-08-25 (m4-136):** save/recall/delete the WHOLE filter set under a name (`main.c41_collection_presets`; payload string `v1` + the five component tokens incl. a first-ever year-range token; empty rule stack as sentinel `-`; structure-strict/content-lenient parsing; apply lands as ONE view reload through silent state writes and becomes current state via the per-key pref observers; corrupt payloads show an inert Apply button rather than silently no-oping). Deviation: darktable scopes presets to its own preset store with per-collection op_params serialization; ours is a dedicated two-column table and the payload is version-tagged for extension. | |
| 2.7 | ~~**No geotagging, no neural-restore panel.**~~ **Closed 2026-09-26** — **Geotagging leg landed 2026-09-24 (u2)** (see above). **Neural-restore infra landed 2026-09-24 (u7a)** (see above). **Denoise task landed 2026-09-25 (u7b):** end-to-end RGB denoise — `c41-core/src/ai/infer.rs` (tiled ORT inference mirroring `restore_rgb`, NAFNet/NIND download-on-demand) + "Neural restore" sidebar section (`crates/c41-ui/src/neural.rs`: Denoise active with model picker + progress + Run → sRGB TIFF beside source → import + grid reload; raw denoise/upscale greyed). Item 2.7 closed except raw denoise and upscale (recorded u7d+
work). **Detail recovery + split landed 2026-09-25 (u7c):** wavelet
texture restoration (`apply_detail_recovery` over the m4-79 DWT port) behind
a Strength slider (0..100, re-blend without re-inference, same-TIFF
overwrite with thumbnail eviction) plus a draggable before/after split
preview. **Upscale task landed 2026-09-25 (u7d):** 2x/4x super-resolution
end to end — scaled tiled driver (O=16, `model_x2`/`model_x4` stem ladders,
output-as-is + inverse gamma), BSRGAN/RealPLKSR download-on-demand,
task+scale pickers, per-task model lists, `_upscale-2x`/`_upscale-4x` naming
with has_suffix skip + collision loop, scaled-dims TIFF + import + reload.
**Raw denoise landed 2026-09-25 (u7e):** Bayer raw-denoise end to end —
CFA pack (force-RGGB origin, per-site black/range/WB norm), tiled RawNIND
inference with PixelShuffle-aware reassembly, match_gain, re-mosaic to a
minimal CFA DNG writer (no-preview tag set, verified by rawloader decode
roundtrip), RawDenoise panel task with NIND download-on-demand.
**Linear raw variant landed 2026-09-26 (u7f):** X-Trans/non-Bayer sensors
route to demosaiced linear camRGB inference (WB/matrix/boost/match_gain/
inversion per `restore_raw_linear`) with LinearRaw DNG output.
**Deviation batch landed 2026-09-27 (u10):** Bayer WB-NONE mode
(`wb_norm: none` → passthrough), `ax·ay` seam-blended Bayer tiling,
antimeridian map fit. Item 2.7 neural-restore leg CLOSED (Foveon stays
refused — upstream rawloader X3F unimplemented; darktablerc override
permanently out — no darktablerc in this product). | low priority; listed for completeness |

## Severity 3 — parity polish

| # | Finding |
|---|---------|
| 3.1 | ~~Zoomable view mode is still greyed out (`ViewMode::Zoomable::is_available() == false`). Deferred deliberately: `GridView` can't express an infinite zoom plane.~~ **Fixed 2026-08-25 (m4-139):** the mode is a hand-drawn `DrawingArea` plane inside its own `ScrolledWindow`, shown/hidden as an overlay child of the lighttable's centre slot (the m4-98c never-swap-the-child invariant preserved — the grid stays live underneath), with continuous wheel zoom anchored at the cursor (m4-133 adjustment discipline), drag-to-pan, single-click select through the shared `SingleSelection`, double-click open, and the bottom-bar stepper repurposed as images-per-row (the culling precedent). Geometry/hit-test/zoom math lives in pure display-free unit tests; thumbnails come from the shared `lighttable::thumbs` service since **m4-140** (2026-08-25), so camera raws render here too through the full preview's decode pipeline instead of staying blank. |
| 3.2 | ~~Panel sections are flat; darktable uses collapsible expanders throughout both panels.~~ **Fixed 2026-08-13** — `collapsible_section()` in `panels/mod.rs`; the left panel's Collections / Colours / Tags now fold from their title rows. The *darkroom* module panel already used `adw::ExpanderRow`. Collapse state persists across sessions (`left_section_*` keys, 2026-08-14). |
| 3.3 | ~~Theme is libadwaita dark, not darktable's exact greys.~~ **Fixed 2026-08-12** — `c41-ui/src/theme.rs` installs darktable's own palette (`grey_NN` from `data/themes/darktable.css`), flattens the chrome (square corners, no blue accent) and sets the functional canvas greys. Not attempted: bauhaus controls (custom-drawn upstream, not CSS) and panel *layout* (2.2-2.6). |
| 3.4 | ~~Map / print / tethering views absent — the header's "Other" is a permanent placeholder.~~ **Closed 2026-09-26** — **Print leg landed 2026-09-24 (u1)** (see above). **Map list landed 2026-09-24 (u3)** (see above). **Slippy-map canvas landed 2026-09-24 (u5):** custom tile renderer on the Map page (`crates/c41-ui/src/tiles.rs` + canvas section in `map.rs`) — Web-Mercator math, integer zoom 0–19, drag-pan, OSM tiles via `ureq` (pure-Rust rustls, no system dep) with policy UA + zoom funnel + 4-concurrent cap + 4 MiB body cap, 256-entry memory LRU + 128 MiB disk cache with startup prune, session negative-cache, clickable markers opening the darkroom, always-visible OSM attribution. **Tethering shell landed 2026-09-24 (u4)** (see above). **Live capture landed 2026-09-24 (u6):** Detect + Capture on the Tether page via libgphoto2 (Detect lists models, Capture saves into the watch folder else `incoming/` and imports + reloads). Item 3.4 CLOSED (neural restore lived here only as a cross-reference; its own item 2.7 is now closed too). **Slideshow view landed 2026-09-27 (u8)** (see below — fourth view; the "Other" placeholder was only ever map/print/tethering). **Physical printing landed 2026-09-27 (u9):** native GTK print dialog beside Export-PDF (`PrintDialog` action, dialog-settled paper/orientation via `page_setup()`, CUPS hardware margins honored conservatively, cancel/apply/error states), `libcups2t64` runtime dep. No CUPS-direct code, no ICC profiles, no templates. |
| 3.5 | ~~Culling shows `THUMB_SIZE` cells rather than filling the viewport; full preview is gdk-pixbuf only, so raws it can't read (`.ORF`) show a message instead of an image. Both need the darkroom view's `BaseImage`/`render()` lifted into a shared module.~~ **Fixed 2026-08-25 (m4-132):** culling pins `min_columns = max_columns = window` and sizes cells from the viewport (`viewport/window` wide, viewport-minus-chrome tall, `ContentFit::Contain`) so every comparison-set size lays out as one filling row; the file manager keeps its own thumb size across the visit, and the old `cull_capacity` window cap (and its stepper dead zone) is gone. Full preview decodes camera raws through `raw_preview::decode_raw_preview` + `preview::render_linear_to_srgb8` at default ("as shot") params off-thread, uploading an R8g8b8 `MemoryTexture` — an `.ORF` now renders instead of showing a message. The audit's predicted enabler turned out unnecessary: no `BaseImage`/`render()` lift was needed because decode+encode were already free-standing functions in `raw_preview.rs`/`preview.rs`; the darkroom's live-preview state stays where it belongs. Remaining nicety (3.6 territory): no 100 % zoom/pan on the full preview. The grid-thumbnail gap closed 2026-08-25 (**m4-140**): both grid cells and the m4-139 zoomable canvas bind through the shared `lighttable::thumbs` service — camera raws decode through `decode_raw_preview` + `render_linear_to_srgb8` into bounded packed RGB8, one session LRU keyed `(path, power-of-two bucket)` under a byte budget cross-hits between surfaces, undecodable paths sit in a negative cache reset per collection load, concurrent decodes cap at two via pre-claimed non-blocking slots. Since **m4-141** (2026-08-25) finished raw decodes also persist to a darktable-mipmap-style on-disk store (`$XDG_CACHE_HOME/c41/thumbs`, 512 MiB budget, newest-first prune reusing the session LRU's keep-set policy): one full-resolution demosaic per (file, bucket) per *machine* instead of per session, entries keyed `(path, bucket)` by process-stable FNV-1a with the source's mtime sealed INSIDE each entry — a replaced or re-saved raw invalidates itself, while a deleted file keeps showing its last known render (darktable behaves the same until refresh); writes are atomic tmp-plus-rename and any IO failure degrades to no-caching. Remaining follow-ups: first-view latency (embedded-preview-first fast path; rawloader exposes no thumbnail parser) and grid-side per-path inflight dedupe (the zoomable canvas already has one). |
| 3.6 | ~~Full preview has no 100 % zoom/pan (darktable does).~~ **Fixed 2026-08-25 (m4-133):** the preview's picture lives in a `ScrolledWindow`; wheel zooms through doubling stops fit → 100 % → 8× (`step_zoom`), anchored at the cursor (an `EventControllerMotion` sidecar remembers the pointer — scroll controllers carry no coords), primary-drag pans by dragging the adjustments backwards, double-click toggles fit ↔ 100 %, and zoom resets on every image change/close so ← / → never lands mid-zoom. Fit requests exactly the viewport dims (`Contain`, no scrollbars); a zoom requests tex×k (`Fill`, aspect exact since both axes scale equally); decode target is re-measured from the *scroller* because a zoomed picture's allocation is the scaled content size. The trap review caught: controllers on a scrolled child report **content** space while the helpers speak **viewport** space — translated at the call site and pinned by test; touchpads are excluded via `DISCRETE` (a smooth flick would fire one stop per event) and fall through to native panning. |

---

## Known gaps (2026-09-30) — outside the severity scale, because never audited

Every item above is struck through, which reads as "parity closed". It is not.
Those rows audited **the product as it stood when the list was written**; the
items below are ones that list never covered. They surfaced on 2026-09-30 from
a *goal-status* sweep (is the Rust migration done? is parity done?), not from a
feature-by-feature comparison against darktable — which is the method this
document's severity tables use, and the method that would be needed to claim the
list below is complete. It is not claimed to be. Expect a real darktable
feature-set audit to find more.

One reconciliation with the rest of this file before reading G1: the header note
and rows 2.1 / 2.4 say **33** live modules, and that was correct when written
(m4-130 slice 3, 2026-08-25; row 2.4's "all 33 module groups" is a m4-149 figure
from 2026-08-26 07:20 UTC). `Local contrast` landed four hours later the same
day, as the 34th. `G1` counts today's 34 + the 2 elsewhere-driven rows = 36.
Those historical figures are left as written and not retroactively edited.

Three provenance classes, kept distinct below:

- **G1–G3** — never audited. Not on any list, in any document, at any point.
- **G4–G5** — named as *goals or decisions* in `RUST_MIGRATION_PLAN.md`, so they
  were always visible as intentions; what was never done is auditing them as
  gaps against the shipped product. They are unimplemented, not unrecorded.
- **G6** — written down as open items and deliberately left open.

Sizing is rough effort, not a schedule. Nothing here is a regression: each was
never built (or, for G6, deliberately deferred), not lost.

**House rule for this section.** Every count here was computed from source and
carries what is needed to recompute it rather than trust it. Usually that is a
`file:line`; where it is not, the section says so. Six counts are composed
across the `geom_iop_stub!` macro and cannot be a single citation: 74, 22, 96,
89, 76 and 90. For those, the anchor is the macro itself
(`iop/geometry.rs:13-26`, invoked at `:30-35`, `:39-43`, `:46-56`) plus the rule
that its own body at `:20` is excluded from the 74 — a naive
`grep 'fn process_cl'` returns 76, not 96, because it also picks up the trait
declaration at `iop/mod.rs:112` and the macro body, and a naive
`grep '"not implemented"'` returns 110, not 76, because each stub says it in
both `process` and `process_cl`. Most other numbers here are a plain grep or
`wc -l`; a few are one-line arithmetic (36 = 34 + 2, 16 = 52 − 34 − 2).
Extending this section means recomputing the numbers you add *and* re-checking
the ones already here: earlier drafts carried counts that review disproved. A
claim with no way to re-derive it does not go in.

### G1 — 16 catalogue rows render inert, and wiring them is plumbing, not panels

`catalog.rs` publishes 52 module rows across the five darktable groups
(Base 11, Tone 11, Color 11, Correct 7, Effect 12). `LIVE_MODULE_LABELS` has
34 entries (`darkroom/mod.rs:2592`); `ELSEWHERE_MODULE_LABELS` has 2 (`:2574` —
`Crop`, `Rotate & perspective`, driven from controls outside the module list).
The header counter is `is_live_module`-derived (`mod.rs:2268`, `:2565-2567`) and
reads **"36 of 52 active"**.

So 16 rows are in neither list, and **all 16 render through `inert_module_row`**
(`mod.rs:2325-2331` dispatches on `elsewhere_hint`, which returns `Some` only
for Crop and Rotate & perspective, and `inert_module_row` is the fall-through).
Inert means dimmed, tooltip "not yet wired", a status icon, and deliberately not
interactive — no switch at all, because the code comments that a disabled switch
still suggests "turn me on", and an earlier version that shipped a *working*
switch is exactly the "looks live, does nothing" bug the function was written to
remove:

`Raw black/white point` · `Demosaic` · `Orientation` · `Color calibration` ·
`Input color profile` · `Output color profile` · `Hot pixels` ·
`Chromatic aberrations` · `Defringe` · `Retouch` · `Liquify` · `Grain` ·
`Soften` · `Highpass` · `Framing` · `Watermark`

`Demosaic` is the one row that is inert *and* independently functional: the
algorithm dropdown at the top of the right panel (`mod.rs:1607-1617`, handler at
`:1624-1629`) is working software that the "36 of 52" counter does not count,
since `is_live_module` only knows the two label lists. Two caveats, so the count
is not oversold: the dropdown sits inside `if is_raw_path(file_path)`
(`:1595`) and is hidden for X-Trans, so it only counts on a Bayer raw; and it is
a free-standing section, not a control on the `Demosaic` row. By "reachable
rather than listed": 37, not 36.

`Orientation` deserves its own note because it looks covered and is not: the
only other "Orientation" in `c41-ui` is the *print* page-orientation combo
(`print.rs:516`). There is no image rotate/flip control — `Straighten` belongs to
`Rotate & perspective`, a different darktable module. Raw files do get automatic
EXIF orientation applied at decode (`rawimage.rs`), which is not a control and is
raw-only. So `Orientation` is genuinely inert.

**The maths is ported; the plumbing is not.** This is the part that is easy to
get wrong, so it is worth being precise about *what* exists. The m4-129…m4-244
chain ported darktable's per-pixel `DT_OMP_FOR` loops into `c41-core/src/iop/`
and exposed them as `#[no_mangle] extern "C"` kernels for the **C** pixelpipe to
call. All 16 rows have such kernels. Row by row — the label each darktable module
returns, so the mapping is checkable against `src/iop/*.c` rather than asserted:

| catalogue row | darktable module | Rust kernel | exports / tests |
|---|---|---|---|
| `Raw black/white point` | `rawprepare.c` | `iop/rawprepare.rs` | 6 / 6 |
| `Demosaic` | `demosaic.c` | `iop/demosaic.rs` | 27 / 31 |
| `Orientation` | `flip.c` | `iop/geometry.rs:111,142` | 2 / 9 |
| `Color calibration` | `channelmixerrgb.c` | `iop/channelmixerrgb.rs` | 2 / 6 |
| `Input color profile` | `colorin.c` | `iop/colorin.rs` | 5 / 15 |
| `Output color profile` | `colorout.c` | `iop/colorout.rs` | 4 / 14 |
| `Hot pixels` | `hotpixels.c` | `iop/hotpixels.rs` | 3 / 14 |
| `Chromatic aberrations` | `cacorrectrgb.c` | `iop/cacorrectrgb.rs` | 7 / 11 |
| `Defringe` | `defringe.c` | `iop/defringe.rs` | 1 / 4 |
| `Retouch` | `retouch.c` | `iop/retouch.rs` | 7 / 12 |
| `Liquify` | `liquify.c` | `iop/liquify.rs` | 6 / **0** |
| `Grain` | `grain.c` | `iop/grain.rs` | 1 / 4 |
| `Soften` | `soften.c` | `iop/soften.rs` | 1 / 4 |
| `Highpass` | `highpass.c` | `iop/highpass.rs` | 2 / 2 |
| `Framing` | `borders.c` | `borders.rs:330` + `iop/geometry.rs:65,87,182` | 4 / 21 |
| `Watermark` | `watermark.c` | `iop/watermark.rs` | 1 / 3 |

Two traps in that table, both of which cost a wrong claim in an earlier draft.
`Chromatic aberrations` is `cacorrectrgb`, **not** `ashift` — `ashift.c:114` is
*rotate and perspective*, one of the 2 elsewhere-driven rows that are otherwise
live; *lens correction* is `lens.cc:252`. And `Orientation` is `flip.c`, whose
own label is "orientation"; `rasterfile.c` is *external raster masks* and has
nothing to do with it. `Liquify` is the one row whose ported kernel ships with no
unit test at all. The last column is per-row, not additive: `Orientation` and
`Framing` both draw on `iop/geometry.rs`, so its 9 tests are counted in each.

But two further things follow, and both cut against reading "a module file
exists" as "the Rust product can run this module":

1. **`IopProcess::process` is a stub — on all 16 of these rows**, and on 89 of the
   96 implementors crate-wide (74 explicit `impl` blocks plus 22 `geom_iop_stub!`
   expansions; the two lists share 6 type *names* — `Ashift`, `Clipping`,
   `Colorharmonizer`, `Denoiseprofile`, `Liquify`, `Retouch` — which are
   distinct types in `iop/*.rs` and `iop/geometry.rs`, so counting by name gives
   90). It returns `Err(Pipeline(…))` unconditionally; the literal
   `"not implemented"` is on 76 of the 89; the other 13 all say "use C FFI path"
   or "use the C FFI entry point", with a parenthetical where the params
   hold LUTs that cannot be cast — `levels.rs:43`, `colisa.rs:41`,
   `lowpass.rs:20`, `lowlight.rs:35`, `tonecurve.rs:154` are the LUT cases.
   None of the 13 says anything else; the 15 "not yet ported" strings in
   `src/iop/` are all `Error::OpenCl` from `process_cl`, not from `process`.
   The driver was never ported, because C calls the kernels directly.
   `watermark.rs` is the clearest case: 91 lines, a stub `process()` at
   `:7-9`, and a fully written, unit-tested `darkroom_watermark_blend` at
   `:22-42` beside it. `iop/geometry.rs:13-26` makes the pattern explicit with
   its `geom_iop_stub!` macro.
2. **Nothing in the Rust product constructs any of them.** `IopProcess` is
   declared at `crates/c41-core/src/iop/mod.rs:101` and, outside `src/iop/`, its
   only other mention in the tree is the module doc at
   `crates/c41-core/src/lib.rs:3`.
   The Rust product drives its *own* module set through the `Stage` enum in
   `pipeline.rs`, whose import list (`pipeline.rs:31`) names 31 modules — none of
   which is among these 16. Outside `src/iop/`, the only references to
   `Watermark`, `Highpass`, `Retouch` or `Liquify` are label strings in
   `catalog.rs:105,98,87,88`.

So each of these rows needs a panel, its parameters, a `Stage` variant and a
driver — the ported kernels remove the *maths* risk, not the *plumbing* work.
That is a materially bigger job than the closed severity-2.1 rows, which attached
panels to modules the Rust pipeline already ran.

### G2 — colour management: engine built, called by C, absent from the Rust product

`c41-core::icc` is built and unit-tested — parser (m4-89), cLUT N-D interpolation
core (m4-90), LUT tag pipelines (m4-91/92), device↔PCS (m4-93a/b), PCS→device and
assembly (m4-127), FFI boundary (m4-129 slice 1) — from the pure-Rust no-LCMS
decision (`RUST_MIGRATION_PLAN.md:122`). It is **not** unused: the C darktable
calls it at 15 `darkroom_icc*` sites (11 in `src/iop/colorin.c`, 4 in
`src/iop/colorout.c`, wired in m4-129 slice 2). But **nothing in the Rust product
calls it** — which is the part that matters for what users run.

In `c41-rs` the working space is fixed, not chosen:
`enum ColorSpace { Rec2020, LinearSrgb }` (`pipeline.rs:45-48`) has two variants
and no picker. There is no preferences dialog in `c41-ui` at all, so no
input-profile assignment, no output profile and no soft-proofing; `export.rs`
contains no ICC reference whatsoever (638 lines, zero hits for `icc`/`profile`).
Non-raw decode does not touch ICC either way. Export goes through the `image`
crate — gated at `dialogs/mod.rs:408` (`RUST_IMAGE_EXTENSIONS` =
jpg/jpeg/png/tif/tiff), decoded at `:472` (`image::ImageReader::open`), reached
from the batch-export path at `:218`. The darkroom *preview* decodes non-raw
through gdk-pixbuf instead (`darkroom/mod.rs:1059-1079`), which likewise ignores
the embedded profile. So **a JPEG tagged with a non-sRGB profile is shown and
written as if it were sRGB, and every export is untagged.** The two
`Input/Output color profile` rows in G1 are inert for the same root cause.

This is the largest gap here whose expensive half is already paid. The engine
exists and is tested; what is missing is the product surface around it — and
unlike G1's rows, the wiring is not a mechanical repeat of a proven pattern.

### G3 — masks: kernels ported, product half missing

Partly false to call this "absent". `crates/c41-core/src/masks/` is **5,192 lines
across 9 shape modules** (brush, circle, detail, ellipse, gradient, group,
object, path, points — 5,338 including `mod.rs`), carrying 33 `#[no_mangle]` FFI
exports, ported from `src/develop/masks/` in m4-154…m4-160
(`PROGRESS.md:3129`). The C darktable calls them — 45 `darkroom_masks_*`
references across 8 of the 9 shape `.c` files — the same way it consumes the rest
of the ported loops. `masks/mod.rs:1-13` is explicit that the loops interleaving
with pixelpipe callbacks stay in C by design.

What is missing is the Rust product's half, and it is the larger half:

- no mask row in the catalogue — no draw, parametric or raster mask;
- no mask UI in `c41-ui` at all. Do not try to confirm that with a `mask` grep:
  148 lines in `crates/c41-ui/src` match it case-insensitively (125
  case-sensitively), across at least five unrelated families — a `cb_mask_*`
  module parameter, a colour-label bitmask, a **star**-rating bitmask
  (`DT_VIEW_RATINGS_MASK`, `lighttable/mod.rs:445`), 10 lines carrying 13 X11
  `ModifierType::*_MASK` sites across `lib.rs` and `lighttable/mod.rs`, and
  prose collisions (`xmp.rs:361` `umask`, `preview.rs:74` "unsharp mask", and a
  comment at `lib.rs:1755-1756` explaining that NumLock arrives as `MOD2_MASK`
  and CapsLock as `LOCK_MASK`).
  There is also one genuinely mask-*named* widget that is not part of darktable's
  mask system at all: the "Mask middle-grey fulcrum" slider at
  `darkroom/mod.rs:3691`, under the "masks & threshold" heading at `:3686`, which
  drives colorbalancergb's own luminance-range mask (`cb_mask_grey_fulcrum`);
- no per-instance module parameters, which is the structural part: darktable lets
  one module exist many times over with independent params, and nothing in the
  current module stack models that.

So this is a UI + module-stack job sitting on top of finished kernels, not a
ground-up port. It is still the biggest single item here, because instance
plumbing touches every module.

### G4 — Lua scripting: no implementation

No `mlua` (or any Lua) dependency in the workspace — zero hits in `Cargo.toml`,
the five member manifests, and `Cargo.lock` — and no Lua in `c41-ui`: two real
hits, both prose in comments citing darktable's `lua scripts installer`
(`panels/mod.rs:1331`, `:4323`), plus one grep false positive (the word
"evaluated", `preview.rs:2512`). darktable's automation surface is Lua 5.4, and
`RUST_MIGRATION_PLAN.md:2106` names "Keep existing Lua scripting API (via
`mlua`)" as an end-state goal, so this is an unmet goal rather than a declined
one. `data/lua/darktable/debug.lua` still ships, inherited from the fork — dead
weight that reads as evidence the feature exists.

### G5 — GPU: the Rust product is CPU-only; one image label over-promises

No `opencl3` dependency (zero hits in `Cargo.lock`). The OpenCL surface in
`c41-core` is a scaffold, not an implementation. `IopProcess::process_cl` has
96 implementors crate-wide (74 explicit + the 22 `geom_iop_stub!` expansions, as
in G1) and **every one of them is an unconditional `Err`** — not one contains
an `Ok(` path — so none has a GPU implementation to call. 15 say
`Error::OpenCl("… OpenCL path not yet ported")`, the rest `Error::Pipeline`.
Alongside them sit `Error::OpenCl`, `has_opencl()` — whose only definition is a
default returning a flat `false` (`iop/mod.rs:117-119`) — and a `ClBuffer` with
a single private zero-sized field (`iop/mod.rs:96-98`). The product is CPU-only
via `rayon`, consistent with the AI backend's own "is CPU only" note
(`ai/mod.rs:34`).

The GPU **compose profiles are not a bug**, contrary to what a first pass at this
suggested. `docker/docker-compose.yml:82,109,126` defines `nvidia`, `amd` and
`intel` profiles; all three build `x-full-build` → `docker/Dockerfile.full-c`,
and that image builds *both* the C app and `c41-rs` and launches `c41-rs`
(`kasmvnc-autostart.sh:48`). Choosing full-c for GPU is a recorded decision
("the GPU profiles (OpenCL is a C-core feature) build full-c",
`RUST_MIGRATION_PLAN.md:2044-2045`). What is left is one genuine overclaim: the
`full-c` image **LABEL** advertises "Lightroom-competitive photo editor with GPU
acceleration" (`docker/Dockerfile.full-c:103`) for a front-end that is CPU-only,
because the OpenCL kernels live in the C app that image also installs but never
launches.

Also a real, separate doc bug found while checking this: `ROADMAP.md:255` and
`:263-265` tell the reader the tree has a `gpu` and an `opencl` compose profile
and to run `docker compose --profile gpu up` / `--profile opencl up` — but
neither profile name exists. They are `nvidia` / `amd` / `intel`, as above. So
the GPU quick-start in the roadmap cannot work as written.

### G6 — recorded earlier, still unclosed

These were already on the record; they are listed so the section is not read as
"everything below is new", and so their status is explicit.

- **HEIC/HEIF/AVIF**: the codec is C-only, so these degrade in the production
  image. Precisely: `c41-core` *does* carry ported HEIF/AVIF kernels
  (`heif.rs` — `heif_u16_to_float`, `heif_float_to_u8`, `heif_float_to_u16` from
  m4-208; `avif.rs` — `avif_u16_to_float`, `avif_u8_to_float`,
  `avif_float_to_u16`, `avif_float_to_u8` from m4-209/210/216) but those are only
  the u16↔float normalize and pack loops; the container/codec decode is still
  libheif/libavif in C, and neither library is in `docker/Dockerfile`
  (`docker/Dockerfile:5` records that as deliberate — the only consumer was
  `darktable-cli`, removed in m4-72). So the extensions are listed in
  `RAW_EXTENSIONS` (`dialogs/mod.rs:527-531`) and pass the import filter
  (`:604`), and `probe_dims` then returns `None`, so the catalogue's width and
  height land as 0×0 (`:610`, `:698`, and the test comment at `:1053` notes the
  0×0 insert is tolerated rather than invented). That is not a HEIF/AVIF-specific
  defect: `probe_dims` (`:728-744`) is gdk-pixbuf-only, so *any* accepted format
  without a registered loader in the shipped image registers 0×0 — the same
  accepted limitation item 2.6 already records for aspect filtering. HEIC/AVIF
  just have no decoder at all, so they stay that way permanently. Fix is a Rust
  codec or removing the extensions; tracked as an open decision at
  `RUST_MIGRATION_PLAN.md:2050` ("drop vs keep full-c"). Note the contrast:
  `docker/Dockerfile.full-c:110-112` *does* ship `libheif1`/`libavif16`, so this
  is a property of the production image, not of the codec being unportable.
- **Foveon X3F**: refused; upstream `rawloader` has no X3F. A deliberate,
  documented refusal — the severity-1 deviation-batch row above, and
  `PROGRESS.md:7114-7115` inside the u10 section. Not an oversight.
- **`darktablerc`**: never loaded. Deliberate, per the same row and
  `PROGRESS.md:7116` ("no darktablerc in product"). It does mean upstream user
  presets/config never apply, and `data/` still carries the fork's config
  templates (`darktableconfig.xml.in`, `shortcutsrc`, `styles/`, `watermarks/`).
- **Build reproducibility**: was open — `Cargo.lock` was gitignored, so no build
  was pinned to a dependency set. **Closed by x1** (`a699cb917c`): lockfile
  tracked, `--locked` passed everywhere the workspace is built.

---

## Not on the table for 2026-08-20

Full darktable parity. darktable is ~93 processing modules, 5 views, mask
editing, tethering and a decade of interaction detail. The realistic target for
the deadline is **a photo editor that is genuinely usable end to end** — import,
cull, rate, edit with a real set of adjustments, export — with this document
covering what remains. Anything claiming more than that would be a claim, not a
plan.

That judgement still holds for the *2026-08-20* deadline it was written about.
Two things it could not have accounted for. First, "usable end to end" was
reached while G1–G5 sat outside the audit entirely — so the sentence above, "with
this document covering what remains", is no longer true, and is left in place
only as the historical record. Second, "Known gaps" is not a complete statement
of the distance to parity: it came from a goal-status sweep, and the
feature-by-feature comparison this document's own severity rows were built on has
not been redone as a *sweep* since 2026-08-11. (The 2026-09-26 closure note at the
top closed the rows that sweep had found; it was not itself a fresh sweep, which
is exactly why the next one turned up G1–G6.) Treat G1–G6 as the top of that
list, not its extent. G2 (colour management) and G3 (masks) are the two largest.

`ROADMAP.md`'s Phases 1–8 (smart previews, virtual copies, face detection and
the People view, AI auto-tagging, batch AI queue, cloud sync, mobile companion,
print templates) are a *different* and much larger tier, untouched; see that
document's own ~44-week estimate.
