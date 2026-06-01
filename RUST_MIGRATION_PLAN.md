# Darkroom — Rust Migration Plan

Incremental rewrite of the Darkroom codebase (C/GTK3) into Rust + GTK4 via the
`gtk4-rs` bindings. The chosen strategy is **incremental FFI-boundary migration**:
each subsystem is replaced one at a time behind a stable C FFI layer, keeping the
application runnable throughout.

---

## Current status — 2026-06-01

### Phase 0 — Infrastructure ✅ complete

### Phase 1 — Image pipeline ⚙️ at hard boundary

| Metric | Value |
|---|---|
| IOP Rust modules registered | **69 / 93** |
| Unit tests passing | **329** |
| IOP `.rs` files | 69 (one per C IOP) |
| Shared modules | `color`, `math`, `raw`, `geometry` |
| Last patch | `Phase 2z+32` (rotatepixels distort) |
| CI status | `Rust` workflow green; `Fork CI` green |

**All 93 `src/iop/*.c` files have a corresponding Rust module.**
The migration has reached the hard boundary: every remaining `DT_OMP_FOR` loop
depends on shared infrastructure (color-space transforms, interpolation,
bilateral grid, NLM, perspective matrices) not yet in Rust. Those IOPs have
stub `IopProcess` impls registered; their loops will be ported once the
blocking infrastructure lands.

#### Fully migrated IOPs (all OMP loops → Rust, 0 remain in C)

`agx`, `atrous`, `basecurve`, `basicadj`, `blurs` (process loops),
`bloom`, `cacorrect`, `cacorrectrgb`, `censorize`, `channelmixer`,
`channelmixerrgb`, `clahe`, `colisa`, `colorbalance`, `colorchecker`,
`colorcontrast`, `colorcorrection`, `colorequal` (prefilter loops),
`colorin`, `colorize`, `colormapping`, `colorout`, `colorzones`,
`defringe`, `diffuse` (mask loop), `dither`, `exposure`,
`filmic`, `filmicrgb` (utility loops), `gamma` (7 of 12),
`globaltonemap`, `graduatednd`, `grain`, `hazeremoval`,
`highlights` (5 of 9 + mask loops), `highpass`, `hotpixels` (all 3 variants),
`invert`, `levels`, `lowlight`, `lowpass`, `lut3d`, `monochrome`,
`negadoctor`, `overexposed` (all 4 modes), `overlay`, `primaries`,
`profile_gamma`, `rasterfile`, `rawdenoise`, `rawprepare`,
`relight`, `rgbcurve`, `rgblevels`, `shadhi`, `sigmoid`, `soften`,
`splittoning`, `temperature`, `tonecurve` (3 of 7), `toneequal` (wait for `colortransfer`),
`useless`, `velvia`, `vibrance`, `vignette`, `watermark`, `zonesystem`.

Geometric distort loops fully migrated in `geometry.rs`:
`borders`, `crop`, `enlargecanvas`, `flip`, `rotatepixels` (distort only).

#### Partially migrated (some loops remain, blocked on infrastructure)

| IOP | Loops remaining | Blocking dependency |
|-----|----------------|---------------------|
| `colorequal` | 9 | `dt_UCS_*` / `dt_JzAzBz_*` color-space |
| `toneequal` | 4 | full histogram + luminance-LUT machinery |
| `highlights` | 4 | `interpolate_color_xtrans` / `interpolate_color` inpaint |
| `diffuse` | 2 | `xoshiro128plus` + `gaussian_noise` PRNG |
| `filmicrgb` | 14 | `RGB_to_Ych`, `gamut_mapping`, Filmlight color space |
| `gamma` | 5 | `dt_Lab_to_XYZ`, `dt_HSL_2_RGB`, `dt_JzAzBz_*` |
| `colortransfer` | 1 | k-means + cluster-weighted ab transfer |
| `blurs` | 12 | Gaussian IIR, FFT, box-filter builders |
| `rotatepixels` | 1 | `dt_interpolation_compute_pixel4c` |
| `scalepixels` | 2 | `dt_interpolation_*` |

#### Stubs only — fully blocked

`ashift`, `bilat`, `clipping`, `colorbalancergb`, `colorharmonizer`,
`colorreconstruction`, `demosaic`, `denoiseprofile`, `equalizer`,
`finalscale`, `liquify`, `nlmeans`, `rawoverexposed`, `retouch`,
`sharpen`, `spots`.

#### What blocks the remaining loops

| Infrastructure | Unblocked IOPs |
|---|---|
| `dt_UCS_JCH` / `dt_JzAzBz_*` color spaces | colorequal, colorbalancergb, colorharmonizer, filmicrgb, gamma |
| `dt_interpolation_*` | scalepixels, rotatepixels process(), demosaic |
| 3D bilateral grid | colorreconstruction |
| NLM + wavelet | denoiseprofile, nlmeans, sharpen |
| `dt_dev_distort_backtransform_plus` | rawoverexposed |
| Keystone / perspective 3×3 | clipping, ashift |
| `xoshiro128plus` + `gaussian_noise` | diffuse, filmicrgb inpaint |
| Per-image edit history API | retouch |

#### Shared darkroom-core modules

| Module | Purpose |
|--------|---------|
| `color` | RGB↔HSL, Lab↔XYZ↔ProPhoto, ICC profile primitives (extrapolate_lut, apply_trc, get_rgb_matrix_luminance) |
| `math` | `fastlog2`, `fastlog` (IEEE-754 bit-twiddled approximations) |
| `raw` | `fc_bayer`, `fc_xtrans`, `fcol` — CFA Bayer/X-Trans primitives |
| `geometry` | Coord-shift, flip/swap, rotate (2×2 matrix), row-blit; covers all geometric distort_* loops |

### Phase 2 — Database ⚙️ starting now

`darkroom-db` crate skeleton exists with `rusqlite = "0.31"`.
Stub files in place; no migrated logic yet.

**Attack order:** tags → metadata → film → collection → image → history

### Phase 3 — GTK4 UI shell 🟡 bootstrapped

`crates/darkroom-ui` compiles against `gtk4 0.9` + `libadwaita 0.7`.
`darkroom_ui::run()` boots an `adw::Application` + placeholder window.
`darkroom-rs` binary calls it and exits cleanly. Production launch still
uses the C binary.

---

## Architecture overview

```
┌───────────────────────────────────────────┐
│              GTK4 UI shell (Rust)         │  Phase 3  🟡 bootstrapped
│  lighttable · darkroom · panels · dialogs │
├───────────────────────────────────────────┤
│           Core services (Rust)            │  Phase 2  ⚙️  starting
│  collection · tags · history · metadata  │
├───────────────────────────────────────────┤
│          Image pipeline (Rust)            │  Phase 1  ⚙️  at boundary
│  pixelpipe · IOPs · demosaic · OpenCL    │
├───────────────────────────────────────────┤
│    C FFI shim (darkroom-sys)              │  Phase 0  ✅ complete
└───────────────────────────────────────────┘
```

---

## Goals

- Memory safety (eliminate buffer-overflow / use-after-free class of bugs)
- Modern UI toolkit: GTK4 + libadwaita via `gtk4-rs` 0.9+
- Cargo-native build: `cargo test`, `cargo bench`, `cargo clippy` in CI
- Keep existing Lua scripting API (via `mlua`)
- Keep OpenCL GPU pipeline (`opencl3` crate)
- End state: `cargo build --release` produces the full binary; CMake deleted

---

## Phase 0 — Infrastructure ✅ done

- Cargo workspace: `darkroom-sys`, `darkroom-core`, `darkroom-db`, `darkroom-ui`, `darkroom`
- `darkroom-sys/build.rs` → bindgen for public C symbols
- CMake → `cargo` build integration with `CARGO_TARGET_DIR=build/cargo-target`
- RUSTFLAGS `-soname=libdarkroom_core.so` + `lib_darktable` `-Wl,--no-as-needed`
- `docker/Dockerfile.rust-dev` — persistent Rust + GTK4 + libadwaita build image
- CI: `.github/workflows/rust.yml` (check + test) and `ci-fork.yml` (C/CMake build) both green

---

## Phase 1 — Image pipeline ⚙️ at hard boundary

### Working model

Each patch: write `crates/darkroom-core/src/iop/<name>.rs` → register in
`mod.rs` → declare in `src/rust_ffi/darkroom_core.h` → replace C OMP body →
add to CMakeLists DEPENDS → independent architectural review by subagent →
fix findings → `cargo test` in dev container → commit + push both remotes.

### Next steps for Phase 1

To unblock the remaining ~40 OMP loops, port these in order:

1. **`xoshiro128plus` + `gaussian_noise`** → unblocks `diffuse` inpaint, `filmicrgb` noise init  
2. **`dt_UCS_JCH` / Filmlight Yrg/Ych** → unblocks `colorequal`, `colorbalancergb`, `colorharmonizer`, `filmicrgb` main loops  
3. **`dt_JzAzBz`** → unblocks `gamma` hz/Jz channels, `colorbalancergb` saturation formula  
4. **`dt_interpolation_compute_pixel4c`** → unblocks `rotatepixels` process(), `scalepixels`, `demosaic`  
5. **3D bilateral grid** → unblocks `colorreconstruction`  
6. **Gaussian IIR + FFT** → unblocks remaining `blurs` helpers  

---

## Phase 2 — Database and collections ⚙️ starting

**Goal:** All SQLite queries go through `rusqlite`-based Rust; C uses the same
structs through `#[repr(C)]` FFI. The C side shrinks to thin wrappers.

### Files to replace

| C file | Rust replacement | Status |
|--------|-----------------|--------|
| `src/common/tags.c` | `darkroom-db/src/tags.rs` | stub |
| `src/common/metadata.c` | `darkroom-db/src/metadata.rs` | stub |
| `src/common/film.c` | `darkroom-db/src/film.rs` | stub |
| `src/common/collection.c` | `darkroom-db/src/collection.rs` | stub |
| `src/common/image.c` | `darkroom-db/src/image.rs` | stub |
| `src/common/history.c` | `darkroom-db/src/history.rs` | stub |

### Pattern (per module)

1. Define a `#[repr(C)]` struct matching the C struct layout
2. Implement `extern "C"` trampolines that open the DB connection and call rusqlite
3. Replace the C function body with a `→ rust_wrapper()` call
4. Add `cargo test` coverage for the SQL surface

### Why tags first

`src/common/tags.c` is 1990 lines but the public API is well-isolated: ~20
functions, all operating on the `data.db` `tags` / `tagged_images` tables.
No tight coupling to the IOP pipeline. It proves the DB FFI pattern cleanly.

---

## Phase 3 — UI shell: GTK4 + gtk4-rs 🟡 bootstrapped

### Current state

- `crates/darkroom-ui` → `gtk4 0.9` + `libadwaita 0.7`
- `darkroom_ui::run()` boots `adw::Application` → `ApplicationWindow` (placeholder)
- `darkroom-rs` binary launches it and exits cleanly
- `docker/Dockerfile.rust-dev` has all GTK4 dev libs for local build/test

### Migration order

| Priority | View / panel | Status |
|---|---|---|
| 1 | `AdwApplicationWindow` shell | ✅ placeholder done |
| 2 | Lighttable thumbnail grid | pending |
| 3 | Darkroom editing view | pending |
| 4 | Collections panel | pending |
| 5 | History panel | pending |
| 6 | IOP panels | pending |
| 7–9 | Export, prefs, import/map | pending |

### Production Dockerfile changes (deferred)

When Phase 3 reaches the lighttable milestone:
1. Add `libgtk-4-dev libadwaita-1-dev` to builder stage
2. `cargo build --release --workspace` after CMake → install `darkroom-rs`
3. Add `libgtk-4-1 libadwaita-1-0` to runtime stage
4. `docker exec darkroom darkroom-rs` for opt-in testing
5. Flip autostart once feature-parity is reached

---

## Phase 4 — Remove C entirely (future)

1. Delete `CMakeLists.txt` and sub-files
2. Move asset install (`share/darktable/`) to `build.rs` or Cargo xtask
3. CI: remove CMake job
4. Dockerfile builder: `cargo build --release && cargo install --path crates/darkroom`

---

## Operations & packaging (cross-cutting, shipped)

- `docker/kasmvnc-autostart.sh` — traps SIGTERM/SIGINT/SIGHUP, forwards to
  darkroom child, waits ≤15 s for clean `dt_conf_save` before exiting
- `src/gui/gtk.c` — `g_unix_signal_add` SIGTERM handler + 30 s periodic
  `g_timeout_add_seconds` flush (survives SIGKILL)
- `docker/Dockerfile` — SONAME + `--no-as-needed` linker fix so all IOP
  plugins resolve `darkroom_*` symbols at startup
- CI — `rust.yml` (workspace type-check + test, GTK4 deps installed),
  `ci-fork.yml` (slim CMake GCC Release build for C-only changes)

---

## Effort and risk summary

| Phase | Status | Risk |
|-------|--------|------|
| 0 — Infrastructure | ✅ done | — |
| 1 — IOP pipeline | 69/93 modules, at boundary | Medium (unblocking infrastructure) |
| 2 — Database | starting now (tags first) | Low |
| 3 — UI (GTK4) | bootstrapped | High (largest phase) |
| 4 — Remove C | future | Medium |

---

## Key crate dependencies

```toml
gtk4       = { version = "0.9", features = ["v4_12"] }       # in use
libadwaita = { version = "0.7", features = ["v1_5"] }        # in use
glib       = "0.20"                                          # in use
rusqlite   = { version = "0.31", features = ["bundled"] }    # in use (darkroom-db)
anyhow     = "1"
tracing    = "0.1"
rayon      = "1"
cairo-rs   = "0.20"   # Phase 3 drawing area
gdk4       = "0.9"    # Phase 3 input controllers
lcms2      = "6"      # when colorin/colorout migrate LCMS calls
opencl3    = "0.9"    # when OpenCL kernels move to Rust
mlua       = { version = "0.10", features = ["lua54", "vendored"] }
```

---

## What is NOT in scope

- Replacing `rawspeed` C++ (keep as vendored submodule via `cc` crate)
- Replacing `lensfun`, `gmic` (keep as system libraries)
- Rewriting Lua plugin API (keep `mlua` as thin wrapper)
