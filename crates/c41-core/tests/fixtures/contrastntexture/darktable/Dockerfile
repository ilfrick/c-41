FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential cmake desktop-file-utils docbook-xml docbook-xsl \
  gettext git intltool libatk1.0-dev libcairo2-dev libcolord-dev \
  libcolord-gtk-dev libcmocka-dev libcups2-dev libcurl4-openssl-dev libexiv2-dev \
  libgdk-pixbuf-2.0-dev libgphoto2-dev libgraphicsmagick1-dev libgtk-3-dev \
  libheif-dev libjpeg-dev libjson-glib-dev liblcms2-dev liblensfun-dev \
  liblua5.4-dev libomp-dev libopenexr-dev libopenjp2-7-dev libosmgpsmap-1.0-dev \
  libpango1.0-dev libpng-dev libpotrace-dev libpugixml-dev librsvg2-dev \
  libsecret-1-dev libsqlite3-dev libtiff-dev libwebp-dev libxml2-dev \
  libxml2-utils ninja-build pkg-config po4a python3-jsonschema \
  xsltproc zlib1g-dev libarchive-dev libgmic-dev && \
  rm -rf /var/lib/apt/lists/*
