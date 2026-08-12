# C-41 — Changes vs. upstream darktable

This document lists all modifications made to the darktable codebase in the C-41 fork.
The upstream base is darktable `master` as of May 2026.

---

## Branding

All user-visible occurrences of "darktable" have been replaced with "Darkroom":

- Application name, window titles, about dialog, preferences dialog
- Help menu labels and homepage URL (now points to this repo)
- Welcome screen and import dialog messages
- CLI `--help` usage text and binary symlink (`darkroom`, `darkroom-cli`)
- Configuration and cache directories: `~/.config/darkroom/`, `~/.cache/darkroom/`

Internal C identifiers (`darktable.h`, `dt_` prefix, struct names, function names) are
intentionally left unchanged to minimize diff size and ease future upstream merges.

---

## New Features

### Smart Previews
**Files:** `src/common/smart_previews.h`, `src/common/smart_previews.c`

Generates compact JPEG proxies (default 2560 px long edge) that allow editing
images when the original files are offline (e.g. on a disconnected external drive).

- DB schema: `smart_previews` table in `library.db` (migration version 59)
- API: `dt_smart_preview_generate()`, `dt_smart_preview_exists()`, `dt_smart_preview_remove()`
- Lighttable badge: `[SP]` indicator on thumbnails with a smart preview
- CLI: `darkroom --smart-previews generate <imgid>` and `--smart-previews remove <imgid>`

### Virtual Copies
**Files:** `src/common/virtual_copies.h`, `src/common/virtual_copies.c`

Creates independent editable copies of an image that share the same source file.
Equivalent to Lightroom's "Create Virtual Copy".

- DB schema: `virtual_copy_of` column on `images` table (migration version 59)
- API: `dt_virtual_copy_create()`, `dt_virtual_copy_delete()`
- Lighttable UI: "Create Virtual Copy" button in the image actions panel
- Export: virtual copies export as independent images

### Print Layout Templates
**Files:** `src/common/print_layouts.h`, `src/common/print_layouts.c`,
`data/print_layouts/*.json`

Predefined multi-image print layouts selectable from the print view sidebar.
Templates are JSON files stored in `$DATADIR/print_layouts/` (system) or
`~/.config/darkroom/print_layouts/` (user-defined).

Bundled templates:
| File | Description |
|------|-------------|
| `single-fullpage.json` | One image filling the entire page |
| `grid-2x3.json` | 6-cell grid (2 columns × 3 rows) |
| `contact-sheet-4x5.json` | 20-cell contact sheet (4 × 5) |
| `triptych.json` | 3 equal panels side by side (landscape) |
| `photo-book-page.json` | Large feature photo + caption strip |

Cell coordinates use relative [0..1] fractions of the page area, so templates
work correctly at any paper size or orientation.

---

## Docker Infrastructure

**Files:** `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/kasmvnc-autostart.sh`

Multi-stage Docker build:
- **Builder stage**: Ubuntu 24.04, compiles Darkroom from source via `git clone --recurse-submodules`
- **Runtime stage**: `linuxserver/baseimage-kasmvnc:ubuntunoble`, provides browser-accessible GUI via KasmVNC

GPU support profiles in `docker-compose.yml`:
| Profile | GPU | Requirement |
|---------|-----|-------------|
| *(default)* | CPU only | None |
| `nvidia` | NVIDIA CUDA/OpenCL | nvidia-container-toolkit |
| `amd` | AMD ROCm/OpenCL | ROCm stack + /dev/kfd |
| `intel` | Intel OpenCL | intel-opencl-icd + /dev/dri |

Browser UI: `http://localhost:3000`

---

## Bug Fixes vs. Upstream

| Commit | Description |
|--------|-------------|
| `59cb161` | Replace Unicode smart quotes (U+201C/U+201D) with ASCII `"` in `darktable.c` and `imageio_tiff.c` — caused compilation failure with `-Wfatal-errors` |
| `6fa44b0` | Fix C syntax: `"luarc"` inside string literal (introduced by smart-quote sweep) replaced with `'luarc'` |
