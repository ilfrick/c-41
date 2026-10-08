#!/usr/bin/env bash
# Generate the raw fixture for the G2a2 raw-path corpus claim, into the same
# corpus directory the ICC corpus test reads.
#
#   scripts/make-raw-corpus.sh [directory]     # default /tmp/c41-icc-corpus
#   C41_ICC_CORPUS=<dir> cargo test -p c41-ui --lib raw_preview
#
# Why a raw fixture at all. G2a2's input profiling scans a *non-raw* image's
# container for an embedded ICC and otherwise assumes sRGB. A raw must never
# go through that seam: it has no embedded ICC to lift, and its input profile
# is the camera→Rec.2020 matrix (`rawimage::load` derives `cam_to_working`
# from the file's ColorMatrix). Before this script there was no file anywhere
# that exercised the raw branch end-to-end on real bytes — the decode glue was
# only ever run by hand on the developer's machine, never by CI. This fixture
# closes that gap the same way the ICC corpus does: generated, not committed,
# so no third-party camera data lands in the tree.
#
# What the fixture is. A minimal but *real* Adobe DNG: a little-endian TIFF
# whose IFD carries exactly the tags the engine's raw decoder (rawloader
# 0.37.1) requires — Make+Model (looked up unconditionally), DNGVersion (what
# routes a TIFF to the DNG decoder), Compression=1 (uncompressed), 16-bit
# little-endian CFA photosites, WhiteLevel, a 2x2 RGGB CFA pattern, a 3x3
# ColorMatrix1 and a D65 AsShotNeutral so colour is finite and the camera
# matrix is provably non-identity. The pixel plane is a deterministic RGGB
# gradient, every value in [0, WhiteLevel]. Roughly 1.3 KB — the whole thing
# is regenerable from this script alone.
#
# The assertions it backs (ALL env-gated, so CI stays hermetic):
#   * `is_raw_path` routes the file to the raw branch (never the ICC scan);
#   * `extract_embedded_file` has nothing to lift from it (a raw carries no
#     profile for the G2a2 seam to mislabel);
#   * `rawimage::load` decodes it end-to-end with a non-identity camera matrix;
#   * `decode_raw_preview_with` produces a finite linear preview.

set -euo pipefail

OUT="${1:-/tmp/c41-icc-corpus}"
mkdir -p "$OUT"
DNG="$OUT/synthetic.dng"

python3 - "$DNG" <<'PYEOF'
import struct
import sys

out = sys.argv[1]
W, H = 24, 20
WHITE = 4095

u16 = lambda v: struct.pack("<H", v)
u32 = lambda v: struct.pack("<I", v)

# (tag, type) ordered exactly as the manifest comment above documents it.
entries = []

# TIFF counts are in *elements* (elemsize 2 for SHORT, 4 for LONG, 8 for
# RATIONAL/SRATIONAL …), not bytes — a byte count would make the reader treat
# an inline value as an out-of-line offset. This packs payloads whose length
# divides cleanly and records the element count.
ELEMSIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 10: 8}

def add(tag, typ, payload: bytes):
    size = ELEMSIZE[typ]
    assert len(payload) % size == 0, f"tag {tag}: payload not whole elements"
    entries.append((tag, typ, len(payload) // size, payload))

# Basic TIFF-EP / DNG tags.
add(256, 4, u32(W))                       # ImageWidth
add(257, 4, u32(H))                       # ImageLength
add(258, 3, u16(16))                      # BitsPerSample
add(259, 3, u16(1))                       # Compression = none
add(262, 3, u16(32803))                   # PhotometricInterpretation = CFA
add(271, 2, b"synthetic\0")               # Make (rawloader requires it)
add(272, 2, b"opal-dng\0")                # Model (rawloader requires it)
add(273, 4, u32(0))                       # StripOffsets (patched below)
add(274, 3, u16(1))                       # Orientation = top-left
add(277, 3, u16(1))                       # SamplesPerPixel = 1 (mosaic)
add(278, 4, u32(H))                       # RowsPerStrip
add(279, 4, u32(W * H * 2))               # StripByteCounts
add(33421, 3, struct.pack("<HH", 2, 2))   # CFARepeatPatternDim
add(33422, 1, bytes([0, 1, 1, 2]))        # CFAPattern = RGGB
add(50706, 1, bytes([1, 4, 0, 0]))        # DNGVersion -> routes to DngDecoder
add(50717, 4, u32(WHITE))                 # WhiteLevel (fetch_tag!-required)
# Colour: a plausible camera->XYZ(D50) matrix, encoded as 9 SRATIONALs, and a
# D65-ish AsShotNeutral (3 RATIONALs) so white balance is finite. Both are
# optional in rawloader (defaults exist) but the point of the fixture is that
# the file's OWN camera matrix drives the decode.
cm = [
    0.6339, -0.2569, -0.1024,
    -0.5447, 1.3153, 0.1264,
    -0.0511, 0.1080, 0.9231,
]
cm_bytes = b"".join(
    struct.pack("<ii", round(v * 100000), 100000) for v in cm
)
add(50721, 10, cm_bytes)                  # ColorMatrix1
asn = [0.517, 1.0, 0.588]
asn_bytes = b"".join(
    struct.pack("<II", round(v * 100000), 100000) for v in asn
)
add(50728, 5, asn_bytes)                  # AsShotNeutral

# IFD lays out after the 8-byte header; all inline values fit in 4 bytes
# except the two strings, the matrix and the neutral — those go to a heap
# after the IFD, with the pixel strip last.
ifd_off = 8
ifd_size = 2 + len(entries) * 12 + 4
heap = ifd_off + ifd_size

def align4(x):
    return (x + 3) & ~3

heap_offsets = {}
heap_blobs = []
for tag, typ, count, payload in entries:
    if len(payload) > 4:
        heap = align4(heap)  # align the START; pass 2 pads identically
        heap_offsets[tag] = heap
        heap_blobs.append((tag, payload))
        heap += len(payload)

strip_off = align4(heap)
strip_len = W * H * 2

# Patch StripOffsets now that the layout is final.
for i, (tag, typ, count, payload) in enumerate(entries):
    if tag == 273:
        entries[i] = (tag, typ, count, u32(strip_off))

# The pixel plane: a deterministic RGGB gradient in [0, WhiteLevel].
pat = [[0, 1], [1, 2]]  # RGGB
base = (0.42, 0.55, 0.62)
strip = bytearray()
for y in range(H):
    for x in range(W):
        c = pat[y % 2][x % 2]
        v = base[c] * WHITE * (0.75 + 0.25 * (x / (W - 1)))
        v += float((x * 7 + y * 13) % 9)  # small deterministic texture
        val = int(round(v))
        val = max(0, min(WHITE, val))
        strip += u16(val)
assert len(strip) == strip_len

blob = bytearray()
blob += b"II" + u16(42) + u32(ifd_off)
blob += u16(len(entries))
for tag, typ, count, payload in entries:
    blob += u16(tag) + u16(typ) + u32(count)
    if len(payload) > 4:
        blob += u32(heap_offsets[tag])
    else:
        blob += payload + b"\x00" * (4 - len(payload))
blob += u32(0)  # no next IFD
for _, payload in heap_blobs:
    blob += b"\x00" * (align4(len(blob)) - len(blob))
    blob += payload

# The strip lands at strip_off — but after the heap is fully laid down the
# file length may be short of it only if the heap fit in less than planned,
# which it cannot (the heap is exactly the four blobs above).
while len(blob) < strip_off:
    blob += b"\x00"
assert len(blob) == strip_off, f"layout drift: {len(blob)} != {strip_off}"
blob += strip

with open(out, "wb") as f:
    f.write(blob)
print(f"wrote {out} ({len(blob)} bytes, {W}x{H} RGGB, rawloader 0.37.1-minimal DNG)")
PYEOF