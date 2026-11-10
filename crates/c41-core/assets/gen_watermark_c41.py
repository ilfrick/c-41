#!/usr/bin/env python3
"""Regenerate crates/c41-core/assets/watermark_c41.png.

The c41 Watermark module's built-in asset (slice 1): a 160×160 white emblem on
transparency — an aperture-style ring with a diagonal band and a centre dot —
rendered with 4×4 supersampling (the anti-aliasing comes from averaging the
16 samples per pixel).

Run from the repository root:

    python3 crates/c41-core/assets/gen_watermark_c41.py

The output is deterministic (no randomness, no date stamp) so the committed
PNG is reproducible byte-for-byte; the Rust test `embedded_asset_decodes_and_
has_the_expected_shape` pins the artistic shape (size, opaque coverage bounds,
solid white centre) so an accidental re-generation that produced something
visibly different fails CI.

The PNG writer accepts only 8-bit RGBA rows and emits colour type 6 with
filter 0 and zlib level 9 — the subset `c41_core::watermark::decode_png_rgba8`
decodes (it also reads RGB/type-2 and all filters, but the asset generator
does not need them).
"""
import binascii
import math
import struct
import zlib
from pathlib import Path

WIDTH = HEIGHT = 160
SS = 4  # supersampling per axis
OUT = Path(__file__).with_name("watermark_c41.png")


def chunk(typ: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + typ
        + data
        + struct.pack(">I", binascii.crc32(typ + data) & 0xFFFFFFFF)
    )


def sample(px: float, py: float) -> float:
    """Coverage [0,1] of the emblem at a sample point (pixel space)."""
    c = WIDTH / 2.0
    d = math.hypot(px - c, py - c)
    ring = 1.0 if 54.0 <= d <= 66.0 else 0.0
    dot = 1.0 if d <= 14.0 else 0.0
    band = 1.0 if abs(px - py) <= 16.0 else 0.0
    return 1.0 if (ring or dot or band) else 0.0


def main() -> None:
    rows = []
    for j in range(HEIGHT):
        row = []
        for i in range(WIDTH):
            cov = 0
            for sy in range(SS):
                for sx in range(SS):
                    cov += sample(i + (sx + 0.5) / SS, j + (sy + 0.5) / SS)
            alpha = int(round(cov / (SS * SS) * 255))
            row += [255, 255, 255, alpha]
        rows.append(row)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)  # filter 0 per row
    ihdr = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 6, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    OUT.write_bytes(png)
    print(f"wrote {OUT} ({len(png)} bytes)")


if __name__ == "__main__":
    main()
