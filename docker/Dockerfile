# Darkroom – GPU-accelerated photo editor in a container
#
# Build:
#   docker build -t darkroom -f docker/Dockerfile .
#
# Build from a different fork/branch:
#   docker build -t darkroom -f docker/Dockerfile \
#     --build-arg DARKROOM_REPO=https://github.com/yourfork/darkroom.git \
#     --build-arg DARKROOM_BRANCH=mybranch .
#
# Run (CPU only):
#   docker run -p 3000:3000 -v /photos:/photos -v /config:/config darkroom
#
# Run (NVIDIA GPU):
#   docker run --gpus all -e PUID=1000 -e PGID=1000 \
#     -p 3000:3000 -p 3001:3001 \
#     -v /path/to/photos:/photos \
#     -v /path/to/config:/config \
#     darkroom

# ─── Stage 1: Build ──────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ARG DARKROOM_REPO=https://github.com/ilfrick/darkroom.git
ARG DARKROOM_BRANCH=master
ARG CMAKE_BUILD_TYPE=Release
ARG USE_AI=OFF
ARG JOBS=4

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git intltool gettext \
    appstream-util desktop-file-utils \
    libarchive-dev libatk1.0-dev libavif-dev libcairo2-dev \
    libcolord-dev libcolord-gtk-dev libcups2-dev libcurl4-gnutls-dev \
    libexiv2-dev libgdk-pixbuf-2.0-dev libglib2.0-dev libgmic-dev \
    libgphoto2-dev libgraphicsmagick1-dev libgtk-3-dev libheif-dev \
    libjpeg-dev libjson-glib-dev liblcms2-dev liblensfun-dev \
    liblua5.4-dev libopenexr-dev libopenjp2-7-dev \
    libosmgpsmap-1.0-dev libpango1.0-dev libpng-dev libportmidi-dev \
    libpotrace-dev libpugixml-dev libraw-dev librsvg2-dev \
    libsqlite3-dev libtiff5-dev libwebp-dev libx11-dev \
    libxml2-dev libxml2-utils zlib1g-dev libsdl2-dev \
    libsecret-1-dev po4a python3-jsonschema xsltproc perl \
    opencl-headers ocl-icd-opencl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable — needed so CMake can invoke cargo to build darkroom-core
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal \
    && rm -rf /root/.rustup/toolchains/*/share/doc
ENV PATH="/root/.cargo/bin:${PATH}"

# CACHEBUST invalidates the git-clone layer (and everything after it) whenever
# the source changes. Without it, `RUN git clone -b master` is cached by command
# string alone, so Docker keeps a stale checkout even after master advances —
# the image would be silently rebuilt from old source. CI passes the commit SHA.
ARG CACHEBUST=unknown
RUN echo "source revision: ${CACHEBUST}" \
    && git clone --recurse-submodules --depth 1 -b ${DARKROOM_BRANCH} ${DARKROOM_REPO} /src

WORKDIR /src

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=/opt/darkroom \
    -DUSE_AI=${USE_AI} \
    -DUSE_OPENCL=ON \
    -DDONT_USE_INTERNAL_LIBRAW=ON \
    -DRAWSPEED_ENABLE_LTO=ON \
    -DBINARY_PACKAGE_BUILD=ON \
    && cmake --build build --parallel ${JOBS} \
    && cmake --install build

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
# KasmVNC provides browser-accessible GUI (same approach as linuxserver/darktable)
FROM lscr.io/linuxserver/baseimage-kasmvnc:ubuntunoble AS runtime

LABEL maintainer="darkroom-project" \
      org.opencontainers.image.title="Darkroom" \
      org.opencontainers.image.description="Lightroom-competitive photo editor with GPU acceleration" \
      org.opencontainers.image.source="https://github.com/ilfrick/darkroom"

ENV DEBIAN_FRONTEND=noninteractive

# Runtime library dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libarchive13 libavif16 libcairo2 libcolord2 libcolord-gtk1 \
    libcurl3-gnutls libexiv2-27 libglib2.0-0 libgmic1 \
    libgphoto2-6 libgraphicsmagick-q16-3 libgtk-3-0 libheif1 \
    libjpeg8 libjson-glib-1.0-0 liblcms2-2 liblensfun1 \
    liblua5.4-0 libopenexr-3-1-30 libopenjp2-7 \
    libosmgpsmap-1.0-1 libpng16-16 libportmidi0 libpotrace0 \
    libpugixml1v5 libraw23 librsvg2-2 libsqlite3-0 libtiff6 \
    libwebp7 libx11-6 libxml2 zlib1g libsdl2-2.0-0 \
    libsecret-1-0 lua5.4 \
    # OpenCL runtime (CPU fallback via Portable CL)
    pocl-opencl-icd \
    # GTK pixbuf loaders + MIME database (fixes SVG theme icon loading)
    libgdk-pixbuf2.0-bin shared-mime-info \
    # openbox-xdg-autostart dependency
    python3-xdg \
    # newer xkb-data silences KasmVNC keysym warnings (XF86Camera*, XF86NextElement…)
    xkb-data \
    && rm -rf /var/lib/apt/lists/* \
    # Regenerate GDK pixbuf loaders cache so SVG icons load correctly
    # Binary is not in $PATH on Ubuntu 24.04 — use full path
    && (/usr/lib/x86_64-linux-gnu/gdk-pixbuf-2.0/gdk-pixbuf-query-loaders --update-cache || true) \
    # Regenerate shared-mime-info database
    && update-mime-database /usr/share/mime

# Copy compiled Darkroom from build stage
COPY --from=builder /opt/darkroom /opt/darkroom

# Make darkroom binaries available system-wide
RUN ln -s /opt/darkroom/bin/darktable /usr/local/bin/darkroom \
    && ln -s /opt/darkroom/bin/darktable-cli /usr/local/bin/darkroom-cli \
    && ldconfig

# cont-init: runs as root before the desktop session; creates config dirs
# with the correct ownership so the container user (PUID/PGID) can write them
COPY docker/cont-init-darkroom.sh /etc/cont-init.d/50-darkroom-dirs
RUN chmod +x /etc/cont-init.d/50-darkroom-dirs

# KasmVNC autostart: launch darkroom on container start
# /defaults/autostart is a script file in the base image, not a directory
COPY docker/kasmvnc-autostart.sh /defaults/autostart
RUN chmod +x /defaults/autostart

# XDG data dirs so darkroom finds its share files
ENV XDG_DATA_DIRS=/opt/darkroom/share:/usr/share:/usr/local/share
ENV DARKROOM_DATADIR=/opt/darkroom/share/darktable
ENV DARKROOM_CONFIGDIR=/config/darkroom
ENV DARKROOM_CACHEDIR=/config/cache

# Photos and config are mounted as volumes
VOLUME ["/photos", "/config"]

# KasmVNC ports (HTTP + HTTPS)
EXPOSE 3000 3001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1
