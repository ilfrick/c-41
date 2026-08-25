# Darkroom — the production container image (cargo-native, CMake-free).
#
# Milestone 5 made this the default: it builds and ships ONLY the Rust/GTK4
# front-end (c41-rs) — NO CMake and NO C darktable. There is no
# `darktable-cli`, so exporting a heic/heif/avif SOURCE (the only cli user) is
# unavailable and fails gracefully with a logged message; everything else (all
# raws + JPEG/PNG/TIFF) develops through the pure-Rust pipeline. The former full
# C+Rust image is preserved as docker/Dockerfile.full-c (opt-in; needs CMake and
# the C toolchain) for anyone who wants the C darktable / darktable-cli.
#
# Build:
#   docker build -t darkroom -f docker/Dockerfile .
# From a fork/branch:
#   docker build -t darkroom -f docker/Dockerfile \
#     --build-arg DARKROOM_REPO=https://github.com/yourfork/darkroom.git \
#     --build-arg DARKROOM_BRANCH=mybranch .
# Run (CPU): docker run -p 3000:3000 -v /photos:/photos -v /config:/config darkroom

# ─── Stage 1: Build (cargo only — no cmake, no C toolchain beyond cc) ─────────
FROM ubuntu:24.04 AS builder

ARG DARKROOM_REPO=https://github.com/ilfrick/c-41.git
ARG DARKROOM_BRANCH=master

ENV DEBIAN_FRONTEND=noninteractive

# Only the Rust + GTK4 build deps. None of docker/Dockerfile's C-darktable dep
# list (exiv2, lcms2, libraw, gmic, …) or cmake/ninja — the Rust workspace links
# only system GTK4/libadwaita, and c41-sys uses committed bindings (no
# libclang, no C source tree).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config git curl ca-certificates \
    libgtk-4-dev libadwaita-1-dev libgraphene-1.0-dev \
    libcairo2-dev libpango1.0-dev libgdk-pixbuf-2.0-dev \
    # c41-sys links liblensfun (lens-correction stage, m4-130): the dev
    # package's .so is needed at link time only; the runtime stage below ships
    # the calibration database (liblensfun-data-v1).
    liblensfun-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the toolchain WITH the components rust-toolchain.toml pins (rustfmt,
# clippy) so the first `cargo` invocation in /src doesn't trigger a build-time
# component reconcile. That reconcile previously combined with the old
# `rm -rf .../share/doc` (which deleted files rustup still tracked) to corrupt
# the component manifest and fail with "failure removing component 'cargo…',
# directory does not exist: 'share/doc/cargo/README.md'" — reproducible only on a
# fresh rustup install (i.e. CI, not a cached local layer), which is why the
# Docker publish job failed on every run. The builder stage is discarded by the
# multi-stage build, so keeping the docs costs nothing in the shipped image.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal -c rustfmt -c clippy
ENV PATH="/root/.cargo/bin:${PATH}"

# CACHEBUST invalidates the git-clone layer whenever the source changes (CI feeds
# the commit SHA). No submodules: the C-only rawspeed/LibRaw submodules aren't
# used by the cargo build (raw decode is the pure-Rust `rawloader` crate).
ARG CACHEBUST=unknown
# Clone the branch then PIN to the exact CACHEBUST commit, so the image's
# org.opencontainers.image.revision label can't diverge from the shipped bits if
# master advances mid-build. Falls back to branch HEAD when CACHEBUST isn't a real
# commit (e.g. a local build that leaves it "unknown").
RUN git clone -b ${DARKROOM_BRANCH} ${DARKROOM_REPO} /src \
    && (git -C /src checkout --quiet ${CACHEBUST} 2>/dev/null \
        || echo "CACHEBUST=${CACHEBUST} is not a commit; using ${DARKROOM_BRANCH} HEAD")

WORKDIR /src
# (Reproducibility follow-up: add `--locked` once Cargo.lock is committed — it's
# currently gitignored, so --locked has no lockfile to honour and errors out.)
RUN cargo build --release -p c41 --bin c41-rs

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM lscr.io/linuxserver/baseimage-kasmvnc:ubuntunoble AS runtime

# (No org.opencontainers.image.revision label: the KasmVNC base image sets its
# OWN revision and overriding it here proved unreliable — a wrong value is worse
# than none. The source is pinned at build time via the CACHEBUST checkout above;
# add a non-colliding provenance label later if wanted.)
LABEL maintainer="darkroom-project" \
      org.opencontainers.image.title="Darkroom" \
      org.opencontainers.image.description="Rust/GTK4 photo editor — cargo-native (CMake-free) build" \
      org.opencontainers.image.source="https://github.com/ilfrick/c-41"

ENV DEBIAN_FRONTEND=noninteractive

# Runtime deps: GTK4 + libadwaita and the desktop bits the Rust UI needs. NONE of
# the C darktable runtime libraries — the C app isn't built or shipped here.
# adwaita-icon-theme: symbolic icons libadwaita widgets draw (else blank glyphs).
# gsettings-desktop-schemas: libadwaita reads org.gnome.desktop.interface; a
#   missing schema makes GLib abort, so ship it even though we force no theme.
# librsvg2-common: registers the gdk-pixbuf SVG loader (libpixbufloader-svg.so)
#   — the whole UI chrome is `*-symbolic.svg` icons, so without it they render
#   blank. The old image only got librsvg2-2 (the lib, not the loader) transitively
#   via the C deps; install the loader explicitly rather than rely on that.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-4-1 libadwaita-1-0 adwaita-icon-theme gsettings-desktop-schemas \
    librsvg2-common \
    libgdk-pixbuf2.0-bin shared-mime-info python3-xdg xkb-data \
    # liblensfun runtime + calibration database (lens-correction stage)
    liblensfun1 liblensfun-data-v1 \
    && rm -rf /var/lib/apt/lists/* \
    # Regenerate the GDK pixbuf loader cache so SVG icons load (binary not in PATH)
    && (/usr/lib/x86_64-linux-gnu/gdk-pixbuf-2.0/gdk-pixbuf-query-loaders --update-cache || true) \
    && update-mime-database /usr/share/mime

# The only artifact from the build stage: the standalone Rust GUI binary. No
# /opt/darkroom C install, no darktable / darktable-cli symlinks.
COPY --from=builder /src/target/release/c41-rs /usr/local/bin/c41-rs

# cont-init: create config dirs with the runtime user's ownership (runs as root).
COPY docker/cont-init-darkroom.sh /etc/cont-init.d/50-darkroom-dirs
RUN chmod +x /etc/cont-init.d/50-darkroom-dirs

# KasmVNC autostart: launch c41-rs in the desktop session.
COPY docker/kasmvnc-autostart.sh /defaults/autostart
RUN chmod +x /defaults/autostart

ENV DARKROOM_CONFIGDIR=/config/darkroom
ENV DARKROOM_CACHEDIR=/config/cache
# c41-rs reads its catalog from this SQLite file.
ENV DARKROOM_LIBRARY_DB=/config/darkroom/library.db
# Force GTK4's software (Cairo) renderer — KasmVNC has no real GPU, and GSK's
# default GL renderer draws a black window under it.
ENV GSK_RENDERER=cairo
# No accessibility bus in the container; disable a11y so GTK doesn't warn on every
# launch about the missing org.a11y.Bus (GTK's own suggested remedy).
ENV GTK_A11Y=none

VOLUME ["/photos", "/config"]

EXPOSE 3000 3001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1
