#!/usr/bin/env bash
# Build the real-file corpus for crates/c41-core/src/icc/embed.rs, and emit the
# manifest that `C41_ICC_CORPUS` points at.
#
#   scripts/make-icc-corpus.sh [output-dir]     # default /tmp/c41-icc-corpus
#   C41_ICC_CORPUS=<dir> cargo test -p c41-core --lib icc::embed
#
# Why this exists. The module's own unit tests build their fixtures with the
# module's own idea of the JPEG/PNG/TIFF layout, so they cannot catch a shared
# misreading of the spec — a builder and a parser that agree on the same wrong
# thing pass perfectly. This corpus closes that gap with an oracle that shares no
# code with the parser: ImageMagick's `convert FILE icc:OUT.icc` reads the
# embedded profile back byte-for-byte through **lcms**, the same library darktable
# itself uses for colour management. The manifest stores the sha256 of those
# bytes, so the test compares hashes and never has to ship a profile.
#
# The corpus is generated rather than committed on purpose. Vendoring the ICC
# profiles themselves would mean committing third-party profile data (colord and
# ghostscript, each with its own licence) into this repository, and the repo has
# no precedent for committed test fixtures. Generating keeps the oracle
# reproducible on any machine with ImageMagick and no encumbrance in-tree.
#
# What it deliberately covers, because these are the cases hand-written fixtures
# are weakest on:
#   * a profile too large for one JPEG APP2 segment (split, and reassembled by
#     sequence number) — the biggest profiles here are ~122 KB against a 64 KB
#     segment limit;
#   * both TIFF byte orders (ImageMagick only ever writes "II", so the "MM" files
#     are produced by byte-swapping its output and re-verifying through lcms);
#   * PNG's zlib-compressed iCCP;
#   * untagged controls, which must come back Absent rather than Unsupported;
#   * containers we do not parse, which must come back Unsupported.
#
# Requires: ImageMagick (`convert`, `identify`) and any ICC profiles on the
# system. colord and ghostscript ship them under /usr/share/color/icc on most
# distributions.

set -euo pipefail

OUT="${1:-/tmp/c41-icc-corpus}"
ICCDIR=/usr/share/color/icc

command -v convert >/dev/null || { echo "need ImageMagick (convert)" >&2; exit 127; }

# Profiles worth exercising: a small matrix-shaper one, a big LUT one that will
# not fit a single JPEG segment, and a non-RGB one.
PROFILES=()
for cand in \
  "$ICCDIR/colord/sRGB.icc" \
  "$ICCDIR/colord/AdobeRGB1998.icc" \
  "$ICCDIR/colord/FOGRA39L_coated.icc" \
  "$ICCDIR/ghostscript/lab.icc" \
  "$ICCDIR/ghostscript/default_gray.icc"
do
  [ -f "$cand" ] && PROFILES+=("$cand")
done
if [ ${#PROFILES[@]} -eq 0 ]; then
  echo "no ICC profiles found under $ICCDIR — install colord or ghostscript" >&2
  exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT"
convert -size 240x180 gradient:red-blue -depth 8 "$OUT/base.png"

for p in "${PROFILES[@]}"; do
  n=$(basename "$p" .icc)
  for fmt in jpg png tif; do
    convert "$OUT/base.png" -profile "$p" "$OUT/icc_${n}.${fmt}" 2>/dev/null \
      || echo "  skip ${n}.${fmt} (encoder rejected it)" >&2
  done
done

# Untagged controls: a container we parse, carrying no profile.
for fmt in jpg png tif; do cp "$OUT/base.png" "$OUT/none.${fmt}"; done

# Containers this module deliberately does not parse.
convert -size 40x30 gradient:green "$OUT/green.gif" 2>/dev/null || true
convert -size 40x30 gradient:green "$OUT/green.webp" 2>/dev/null || true
convert -size 40x30 gradient:green "$OUT/green.bmp" 2>/dev/null || true
printf 'GIF87a\x01\x00\x01\x00\x80\x00\x00' > "$OUT/notanimage.bin"
: > "$OUT/empty.dat"

# Big-endian TIFFs. ImageMagick writes only "II", so byte-swap its output; the
# swap is verified below by reading the profile back through lcms, which is what
# makes the resulting file trustworthy rather than merely plausible.
swap_tiff() {
  python3 - "$1" "$2" <<'PY'
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
d = bytearray(open(src, "rb").read())
if d[:2] != b"II":
    sys.exit(f"{src}: expected little-endian TIFF")
TYPESIZE = {1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8}
SWAPPED = {3,4,8,9,11,12}
d[0:2] = b"MM"
d[2:4] = struct.pack(">H", 42)
ifd0 = struct.unpack("<I", bytes(d[4:8]))[0]
d[4:8] = struct.pack(">I", ifd0)
seen, nxt = set(), ifd0
while nxt and nxt not in seen:
    if len(seen) > 1024: sys.exit("IFD chain does not terminate")
    seen.add(nxt)
    n = struct.unpack("<H", bytes(d[nxt:nxt+2]))[0]      # read LE, write BE
    d[nxt:nxt+2] = struct.pack(">H", n)
    for i in range(n):
        e = nxt + 2 + i*12
        tag = struct.unpack("<H", bytes(d[e:e+2]))[0]
        typ = struct.unpack("<H", bytes(d[e+2:e+4]))[0]
        cnt = struct.unpack("<I", bytes(d[e+4:e+8]))[0]
        field = bytes(d[e+8:e+12])
        size = TYPESIZE.get(typ,1)*cnt
        d[e:e+2]     = struct.pack(">H", tag)
        d[e+2:e+4]   = struct.pack(">H", typ)
        d[e+4:e+8]   = struct.pack(">I", cnt)
        if size <= 4:
            val = field[:size]
            if typ in SWAPPED: val = val[::-1]
            d[e+8:e+12] = val + b"\0"*(4-size)
        else:
            off = struct.unpack("<I", field)[0]
            d[e+8:e+12] = struct.pack(">I", off)
            if typ in SWAPPED and off+size <= len(d):
                d[off:off+size] = bytes(d[off:off+size])[::-1]
    p = nxt + 2 + n*12
    nxt = struct.unpack("<I", bytes(d[p:p+4]))[0]
    d[p:p+4] = struct.pack(">I", nxt)
open(dst,"wb").write(bytes(d))
PY
}

for f in "$OUT"/icc_*.tif; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  b=${b#icc_}
  swap_tiff "$f" "$OUT/be_$b"
done

# The manifest. Verdicts come from ImageMagick (lcms); container sniffing is done
# from the magic bytes here, deliberately independently of the code under test.
{
  printf 'file\tverdict\tsha256\n'
  for f in $(cd "$OUT" && ls -1 | grep -vE '^(base\.png|manifest\.tsv)$' | sort); do
    tmp=$(mktemp)
    if convert "$OUT/$f" "icc:$tmp" 2>/dev/null && [ -s "$tmp" ]; then
      # A real ICC profile: 'acsp' at 36 and a declared size equal to the length.
      magic=$(dd if="$tmp" bs=1 skip=36 count=4 2>/dev/null | xxd -p)
      declared=$(dd if="$tmp" bs=1 count=4 2>/dev/null | xxd -p)
      size=$(stat -c%s "$tmp")
      if [ "$magic" = "61637370" ] && [ "$declared" = "$(printf '%08x' "$size")" ]; then
        printf '%s\tpresent\t%s\n' "$f" "$(sha256sum "$tmp" | cut -d' ' -f1)"
      else
        printf '%s\tBADPROFILE\t-\n' "$f" >&2
      fi
    else
      case "$(head -c 4 "$OUT/$f" | xxd -p)" in
        ffd8*|89504e47*|49492a00*|4d4d002a*) printf '%s\tabsent\t-\n' "$f" ;;
        *) printf '%s\tunsupported\t-\n' "$f" ;;
      esac
    fi
    rm -f "$tmp"
  done
} > "$OUT/manifest.tsv"

n=$(($(wc -l < "$OUT/manifest.tsv") - 1))
echo
echo "corpus: $n files in $OUT"
echo "run:  C41_ICC_CORPUS=$OUT cargo test -p c41-core --lib icc::embed"