# Darkroom -- Rust Migration Plan

Incremental rewrite of the Darkroom codebase (C/GTK3) into Rust + GTK4 via the
`gtk4-rs` bindings. The chosen strategy is **incremental FFI-boundary migration**:
each subsystem is replaced one at a time behind a stable C FFI layer, keeping the
application runnable throughout.

---

## Current status -- 2026-06-04

### Phase 0 -- Infrastructure complete

### Phase 1 -- Image pipeline at hard boundary

| Metric | Value |
|---|---|
| IOP Rust modules registered | **93 / 93** |
| Unit tests passing | **394** |
| IOP `.rs` files | 93 (one per C IOP) |
| Shared modules | `color`, `math`, `raw`, `geometry` |
| Last patch | `Phase 2z+72` (filmicrgb init_reconstruct + compute_ratios ported to Rust FFI) |
| CI status | `Rust` workflow green; `Fork CI` green |

**All 93 `src/iop/*.c` files have a corresponding Rust module.**
The migration has reached the hard boundary: every remaining `DT_OMP_FOR` loop
depends on shared infrastructure (color-space transforms, interpolation,
bilateral grid, NLM, perspective matrices) not yet in Rust. Those IOPs have
stub `IopProcess` impls registered; their loops will be ported once the
blocking infrastructure lands.

#### Fully migrated IOPs (all active OMP loops -> Rust, 0 remain in C)

`agx`, `atrous`, `basecurve`, `basicadj`, `bloom`, `cacorrect`, `cacorrectrgb`,
`censorize`, `channelmixer`, `clahe`, `clipping`,
`colorbalance`, `colorchecker`, `colorcontrast`, `colorcorrection`,
`colorize`, `colormapping`, `colorzones`, `defringe`, `denoiseprofile`,
`dither`, `exposure`,
`filmic`, `filmicrgb` (utility loops), `globaltonemap`, `graduatednd`,
`grain`, `hazeremoval`, `highpass`, `hotpixels` (all 3 variants),
`invert`, `levels`, `liquify`, `lowlight`, `lowpass`, `lut3d`, `monochrome`,
`negadoctor`, `overexposed` (all 4 modes), `overlay`, `primaries`,
`profile_gamma`, `rasterfile`, `rawdenoise`, `rawprepare`,
`relight`, `rgbcurve`, `rgblevels`, `shadhi`, `sharpen`, `sigmoid`, `soften`,
`splittoning`, `temperature`, `toneequal` (main process loop is
`#else`-guarded dead code since `DT_TONEEQ_USE_LUT=TRUE`),
`useless`, `velvia`, `vibrance`, `vignette`, `watermark`, `zonesystem`.

Geometric distort loops fully migrated in `geometry.rs`:
`borders`, `crop`, `enlargecanvas`, `flip`, `rotatepixels` (distort only).

Commit-params LUT builders migrated:
`colisa`, `lowpass`, `profile_gamma` (contrast/brightness LUT fills).

#### Partially migrated (some loops remain, blocked on infrastructure)

| IOP | C loops remaining | Blocking dependency |
|-----|------------------|---------------------|
| `colorequal` | 1 | GUI background renderer (intentionally deferred) |
| `highlights` | 4 | `interpolate_color_xtrans` / `interpolate_color` CFA inpaint |
| `diffuse` | 1 | anisotropic PDE solver (very complex) |
| `filmicrgb` | 3 | split + chroma v1/v2/v3 + v4/v5 gamut-mapping done; Yrg + gamut-mapping foundation in color.rs; highlights init_reconstruct + compute_ratios ported. Remaining 3: wavelets_reconstruct_RGB (~1081), wavelets_reconstruct_ratios (~1154), reconstruct_highlights (~1319) — the wavelet/diffusion path |
| `gamma` | 5 | `dt_Lab_to_XYZ`, `dt_HSL_2_RGB`, `dt_JzAzBz_*` |
| `channelmixerrgb` | 2 | B-spline local avg reduction (illuminant detection) |
| `colortransfer` | 2 | k-means with atomic accumulators |
| `colorout` | 1 | LCMS `cmsDoTransform` |
| `retouch` | 4 | `dt_linearRGB_to_XYZ` / `dt_XYZ_to_Lab` ICC paths |
| `colorbalancergb` | 4 | Filmlight Yrg / `work_profile` |
| `colorin` | 5 | ICC matrix + LCMS |
| `ashift` | 2 | `dt_Rec709_to_XYZ_D50` + `dt_XYZ_to_Lab` |

#### Stubs only -- fully blocked

`ashift` (11), `clipping` (4), `colorbalancergb` (4),
`colorin` (5), `colorreconstruction` (3), `denoiseprofile` (6),
`liquify` (6), `rawoverexposed` (2), `retouch` (9).

These depend on `dt_interpolation_*`, 3D bilateral grid, NLM/wavelet, perspective
matrices, or per-pixel ICC transforms not yet ported to Rust.

#### What blocks the remaining loops

| Infrastructure | Unblocked IOPs |
|---|---|
| `dt_interpolation_*` | scalepixels, rotatepixels process(), demosaic |
| 3D bilateral grid | colorreconstruction |
| NLM + wavelet | denoiseprofile, nlmeans |
| `dt_dev_distort_backtransform_plus` | rawoverexposed |
| Keystone / perspective 3x3 | clipping, ashift |
| Filmlight Yrg / `work_profile` callbacks | filmicrgb main loops, colorin |
| GUI-only loops | colorbalancergb (2), toneequal GUI LUT |

#### Shared darkroom-core modules

| Module | Purpose |
|--------|---------|
| `color` | RGB<->HSL, Lab<->XYZ, dt UCS 2.2 (JCH/HSB), CAT16 matrices, gamut helpers |
| `math` | `fastlog2`, `fastlog`, PRNG (`splitmix32`, `xoshiro128+`, all noise distributions) |
| `raw` | `fc_bayer`, `fc_xtrans`, `fcol` -- CFA Bayer/X-Trans primitives |
| `geometry` | Coord-shift, flip/swap, rotate (2x2 matrix), row-blit |

### Phase 2 -- Database complete

`darkroom-db` crate: full CRUD for tags, metadata, film, collection, image, history.
C FFI trampolines for tags. 61 DB tests passing.

### Phase 3 -- GTK4 UI shell bootstrapped

`crates/darkroom-ui` compiles against `gtk4 0.9` + `libadwaita 0.7`.
`darkroom_ui::run()` boots an `adw::Application` with three-column
lighttable layout backed by `darkroom-db`.

---

## Architecture overview

```
+-------------------------------------------+
|              GTK4 UI shell (Rust)         |  Phase 3  bootstrapped
|  lighttable . darkroom . panels . dialogs |
+-------------------------------------------+
|           Core services (Rust)            |  Phase 2  complete
|  collection . tags . history . metadata   |
+-------------------------------------------+
|          Image pipeline (Rust)            |  Phase 1  at boundary
|  pixelpipe . IOPs . demosaic . OpenCL    |
+-------------------------------------------+
|    C FFI shim (darkroom-sys)              |  Phase 0  complete
+-------------------------------------------+
```

---

## Goals

- Memory safety (eliminate buffer-overflow / use-after-free class of bugs)
- Modern UI toolkit: GTK4 + libadwaita via `gtk4-rs` 0.9+
- Cargo-native build: `cargo test`, `cargo bench`, `cargo clippy` in CI
- Keep existing Lua scripting API (via `mlua`)
- Keep OpenCL GPU pipeline (`opencl3` crate)
- End state: `cargo build --release` produces the full binary; CMake deleted
