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
| Unit tests passing | **441** |
| IOP `.rs` files | 94 (one per C IOP) |
| Shared modules | `color`, `math`, `raw`, `geometry` |
| Last patch | `Phase 2z+80` (rawoverexposed — row loops split around the C distort callback, marking/coord-fill in Rust) |
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
`filmic`, `filmicrgb`, `globaltonemap`, `graduatednd`,
`grain`, `hazeremoval`, `highlights` (incl. all 6 hlreconstruct/ backends),
`highpass`, `hotpixels` (all 3 variants),
`invert`, `levels`, `liquify`, `lowlight`, `lowpass`, `lut3d`, `monochrome`,
`negadoctor`, `overexposed` (all 4 modes), `overlay`, `primaries`,
`profile_gamma`, `rasterfile`, `rawdenoise`, `rawoverexposed`, `rawprepare`,
`relight`, `rgbcurve`, `rgblevels`, `shadhi`, `sharpen`, `sigmoid`, `soften`,
`splittoning`, `temperature`, `toneequal` (main process loop is
`#else`-guarded dead code since `DT_TONEEQ_USE_LUT=TRUE`),
`useless`, `velvia`, `vibrance`, `vignette`, `watermark`, `zonesystem`.

Geometric distort loops fully migrated in `geometry.rs`:
`borders`, `crop`, `enlargecanvas`, `flip`, `rotatepixels` (distort only).

Commit-params LUT builders migrated:
`colisa`, `lowpass`, `profile_gamma` (contrast/brightness LUT fills).

#### Partially migrated (some loops remain, blocked on infrastructure)

Loop counts verified 2026-06-12 (`grep -rcE 'DT_OMP_FOR(_SIMD)?\(' src/iop --include=*.c`):

| IOP | C loops remaining | Blocking dependency |
|-----|------------------|---------------------|
| `demosaicing/` | 33 | capture.c 15, basics.c 5, vng.c 4, xtrans/ppg/passthrough/dual 2 each, rcd.c 1 — `dt_interpolation_*` + demosaic algorithms |
| `colorbalancergb` | 4 | Filmlight Yrg / `work_profile` |
| `colorreconstruction` | 3 | 3D bilateral grid |
| `colorin` | 3 | ICC matrix + LCMS |
| `channelmixerrgb` | 2 | B-spline local avg reduction (illuminant detection) |
| `colortransfer` | 2 | k-means with atomic accumulators |
| `retouch` | 2 | `dt_linearRGB_to_XYZ` / `dt_XYZ_to_Lab` ICC paths |
| `colorequal` | 1 | GUI background renderer (intentionally deferred) |
| `colorout` | 1 | LCMS `cmsDoTransform` |
| `diffuse` | 1 | anisotropic PDE solver (very complex) |
| `toneequal` | 1 | GUI LUT |

(`ashift`, `clipping`, `denoiseprofile`, `gamma`, `liquify`, `rawoverexposed`
previously listed here/as stubs are at 0 loops — fully migrated.)

#### What blocks the remaining loops

| Infrastructure | Unblocked IOPs |
|---|---|
| `dt_interpolation_*` | demosaicing cluster |
| 3D bilateral grid | colorreconstruction |
| Filmlight Yrg / `work_profile` callbacks | colorbalancergb, colorin |
| Per-pixel ICC / LCMS | colorin, colorout, retouch |
| GUI-only loops | colorequal, toneequal GUI LUT |

Pattern note (Phase 2z+80, rawoverexposed): loops interleaved with C
pipeline callbacks (`dt_dev_distort_backtransform_plus`) are split into
serial C row loops calling Rust for the work before/after the callback —
no Rust→C callback plumbing needed.

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
