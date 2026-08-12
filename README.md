# C-41

**C-41** is a professional photo editing and management application for Linux, forked from [darktable](https://www.darktable.org/) with additional features aimed at closing the gap with Adobe Lightroom.

> C-41 is open source software licensed under the GNU GPL v3.

---

## What's new vs. upstream darktable

See [CHANGES.md](CHANGES.md) for the full changelog.

| Feature | C-41 | darktable upstream |
|---------|----------|--------------------|
| Smart Previews (offline editing) | ✅ | ❌ |
| Virtual Copies | ✅ | ❌ |
| Print Layout Templates | ✅ | ❌ |
| GPU-accelerated Docker image | ✅ | ❌ |

---

## Quick start with Docker

The fastest way to try C-41 is via the pre-built Docker image, which provides a full GUI through your browser via KasmVNC.

```bash
# Pull and run (CPU, no GPU required)
docker run -d \
  --name darkroom \
  -p 3000:3000 \
  -v ~/Pictures:/photos \
  -v ~/.config/darkroom-docker:/config \
  ghcr.io/ilfrick/c-41:latest

# Open in browser
open http://localhost:3000
```

For GPU acceleration and docker-compose usage see [docker/docker-compose.yml](docker/docker-compose.yml) and the [Docker section](#docker) below.

---

## Docker

### Build from source

```bash
git clone https://www.housefz.com/git/ilfrick/c-41.git
cd darkroom
docker build -t darkroom -f docker/Dockerfile .
```

Build arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `DARKROOM_REPO` | Gitea URL | Git repo to clone inside Docker |
| `DARKROOM_BRANCH` | `master` | Branch to build |
| `CMAKE_BUILD_TYPE` | `Release` | `Release` or `Debug` |
| `USE_AI` | `OFF` | Enable ONNX AI features (requires separate runtime) |
| `JOBS` | `4` | Parallel compile jobs |

### Run with docker-compose

```bash
# CPU only
docker compose -f docker/docker-compose.yml up darkroom

# NVIDIA GPU (requires nvidia-container-toolkit)
docker compose -f docker/docker-compose.yml --profile nvidia up darkroom-nvidia

# AMD GPU (requires ROCm)
docker compose -f docker/docker-compose.yml --profile amd up darkroom-amd

# Intel GPU (requires intel-opencl-icd)
docker compose -f docker/docker-compose.yml --profile intel up darkroom-intel
```

The browser UI is available at `http://localhost:3000`.

Volumes:

| Container path | Purpose |
|----------------|---------|
| `/photos` | Mount your photo library here |
| `/config` | Persistent C-41 configuration and cache |

---

## Building from source (native)

### Dependencies (Ubuntu 24.04)

```bash
sudo apt-get install -y \
  build-essential cmake ninja-build git intltool gettext \
  libarchive-dev libavif-dev libcairo2-dev libcolord-dev libcolord-gtk-dev \
  libcups2-dev libcurl4-gnutls-dev libexiv2-dev libgdk-pixbuf-2.0-dev \
  libglib2.0-dev libgmic-dev libgphoto2-dev libgraphicsmagick1-dev \
  libgtk-3-dev libheif-dev libjpeg-dev libjson-glib-dev liblcms2-dev \
  liblensfun-dev liblua5.4-dev libopenexr-dev libopenjp2-7-dev \
  libosmgpsmap-1.0-dev libpng-dev libportmidi-dev libpotrace-dev \
  libpugixml-dev libraw-dev librsvg2-dev libsqlite3-dev libtiff5-dev \
  libwebp-dev libx11-dev libxml2-dev libxml2-utils zlib1g-dev \
  libsdl2-dev libsecret-1-dev opencl-headers ocl-icd-opencl-dev \
  po4a python3-jsonschema xsltproc
```

### Compile

```bash
git clone --recurse-submodules https://www.housefz.com/git/ilfrick/c-41.git
cd darkroom
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/darkroom \
  -DUSE_AI=OFF \
  -DUSE_OPENCL=ON
cmake --build build --parallel $(nproc)
cmake --install build
```

Run: `/opt/darkroom/bin/darktable` (the binary keeps the upstream name internally; the Docker autostart script symlinks it to `/usr/local/bin/darkroom`)

---

## Repositories

| Remote | URL |
|--------|-----|
| Primary (Gitea) | https://www.housefz.com/git/ilfrick/c-41 |
| Mirror (GitHub) | https://github.com/ilfrick/c-41 |
| Upstream | https://github.com/darktable-org/darktable |

---

## License

C-41 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
