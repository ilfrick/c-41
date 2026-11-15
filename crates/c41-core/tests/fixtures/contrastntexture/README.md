# Contrast & Texture golden samples

These are two **numerical samples**, not a full behavioral parity claim. Both
are 256×256 little-endian `f32` RGBA outputs from pinned darktable CPU code.

| Fixture | `piece.iscale` | SHA-256 |
| --- | ---: | --- |
| `darktable-97912518-full-256x256.rgba32` | 1 | `e3edf9c08ed6e8695185a212035d82673688fcb505ff18781bf1f4d85406c9e4` |
| `darktable-97912518-proxy2-256x256.rgba32` | 2 | `703d6e42ebbd92f898b211d0dfe49eb52328eed319405d9548820cea670efb51` |

## Provenance and regeneration

The reference is darktable commit
`97912518a5a9a9786aeee3c38116fbc999412ff8`. The adjacent
`darktable/test_contrastntexture_golden.c` is the complete CMocka generator: it
includes that commit's `src/iop/contrastntexture.c`, calls `init_pipe`,
`commit_params`, `modify_roi_in`, and `process`, then writes the output bytes.
The C and Rust generators use the same 256×256 RGBA input formula and C&T
parameters (`gain_coarse=0.7`, `gain_medium=0.4`, `gain_fine=0.2`,
`detail_level=5`, `halo_control=0`, `noise_bias=0.001`, shadow/highlight gains
0). C `piece.iscale` is the only difference between fixtures (1 or 2); both
use a full 256×256 ROI with scale 1. OpenMP is limited to one thread. The C
build is Release with `USE_OPENCL=OFF`; the Rust test compares RGB using
max-absolute and max-relative tolerances `1e-5` and RMS tolerance `1e-6`, and
requires alpha bit-exactly.

The commands below reproduce the source/build setup on Ubuntu 24.04 / Linux
amd64. Run them from the C-41 checkout root (or set `C41_ROOT` explicitly).
Docker makes the native build dependencies repeatable; the pinned darktable
checkout and its build submodules are fetched separately. The unrelated
`src/tests/integration` image corpus is intentionally not needed for this
standalone CMocka test:

```sh
set -eu
C41_ROOT="${C41_ROOT:-$(git rev-parse --show-toplevel)}"
REF="$C41_ROOT/crates/c41-core/tests/fixtures/contrastntexture"
DT="${DT:-/tmp/darktable-97912518}"
OUT="${OUT:-/tmp/c41-ct-golden-output}"

git clone https://github.com/darktable-org/darktable.git "$DT"
git -C "$DT" checkout 97912518a5a9a9786aeee3c38116fbc999412ff8
git -C "$DT" submodule update --init --recursive -- \
  src/external/LibRaw src/external/OpenCL src/external/libxcf \
  src/external/lua-scripts src/external/rawspeed src/external/whereami
cp "$REF/darktable/test_contrastntexture_golden.c" "$DT/src/tests/unittests/iop/"
git -C "$DT" apply "$REF/darktable/register-test.patch"
mkdir -p "$OUT"
docker build --platform linux/amd64 -f "$REF/darktable/Dockerfile" \
  -t c41-darktable-ct-reference "$REF/darktable"
docker run --rm --platform linux/amd64 -v "$DT:/src" -v "$OUT:/out" \
  -w /src c41-darktable-ct-reference bash -euc '
  cmake -S . -B build-c41-ct -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DUSE_OPENCL=OFF
  cmake --build build-c41-ct --target test_contrastntexture_golden --parallel 2
  OMP_NUM_THREADS=1 C41_GOLDEN_ISCALE=1 \
    C41_GOLDEN_OUTPUT=/out/darktable-97912518-full-256x256.rgba32 \
    build-c41-ct/bin/tests/unittests/iop/test_contrastntexture_golden
  OMP_NUM_THREADS=1 C41_GOLDEN_ISCALE=2 \
    C41_GOLDEN_OUTPUT=/out/darktable-97912518-proxy2-256x256.rgba32 \
    build-c41-ct/bin/tests/unittests/iop/test_contrastntexture_golden
'
sha256sum "$OUT"/*.rgba32
```

The command is expected to print the two SHA-256 values in the table above.
The Dockerfile lists the Ubuntu build dependencies used for the reference
build. `register-test.patch` registers the standalone generator with darktable's
CMocka test CMake directory. No change to the C-41 source checkout is needed
during regeneration.

`piece.iscale=2` tests scale-dependent C&T behavior on the **same synthetic
input**; it does not establish proxy-vs-resampled-original parity. These
fixtures also do not establish full behavioral parity, UI/export integration,
or deployment readiness.
