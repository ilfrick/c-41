//! Raw image decode + sensor normalisation (Phase 3 milestone 2: the real
//! scene-referred front end for `pipeline`).
//!
//! Uses the pure-Rust [`rawloader`] crate to decode a camera raw file into a
//! Bayer/X-Trans CFA mosaic, then normalises each photosite to linear [0,1] by
//! subtracting the per-colour black level and scaling by the per-colour range
//! (`white - black`). The result is a single-channel float mosaic ready to feed
//! the migrated demosaic (wired in a later increment) and then [`pipeline`].
//!
//! rawloader's `blacklevels`/`whitelevels`/`wb_coeffs` are in **RGBE order**
//! indexed by `CFA::color_at` (0=R, 1=G, 2=B, 3=E), which is exactly the index
//! [`normalize_cfa`] uses.
//!
//! Known v1 gaps (to address as the front end matures — none block Bayer):
//! - **Bayer only.** A non-2×2 CFA (Fuji X-Trans is 6×6) is rejected by [`load`]
//!   until the X-Trans demosaic is wired through; our [`RawImage`] models a 2×2
//!   pattern.
//! - **Black level from `blacklevels` only.** Cameras that report `0` and expect
//!   the black point from the masked optical-black border (`blackareas`) get no
//!   black subtraction yet (slightly lifted shadows).
//! - **Highlights are hard-clamped at white.** Over-white photosites are clipped
//!   here, so a future highlight-reconstruction stage would have nothing to
//!   reconstruct from; revisit (carry over-range float) when that lands.
//! - **White balance is stored, not applied** — `wb` belongs after demosaic.
//!
//! [`pipeline`]: crate::pipeline

use crate::{Error, Result};

/// A decoded, black/white-normalised CFA mosaic plus the metadata the demosaic
/// and white-balance steps need. `mosaic` is `width * height` linear [0,1]
/// photosites (single channel); `cfa` is the 2×2 colour pattern (indices into
/// the RGBE-ordered `wb`).
#[derive(Clone, Debug)]
pub struct RawImage {
    pub width: usize,
    pub height: usize,
    /// 2×2 CFA colour indices: `cfa[row % 2][col % 2]` (0=R, 1=G, 2=B, 3=E).
    pub cfa: [[usize; 2]; 2],
    /// White-balance multipliers in RGBE order (as encoded in the file).
    pub wb: [f32; 4],
    /// Display orientation as `(transpose, flip_x, flip_y)` from rawloader's
    /// `Orientation::to_flips()` — applied (after demosaic) by `to_linear_rgba`.
    pub orientation: (bool, bool, bool),
    /// Black/white-normalised photosites, row-major, `width * height` long.
    pub mosaic: Vec<f32>,
}

/// Normalise a raw CFA plane to linear [0,1]: per photosite, subtract the
/// per-colour black level and divide by the per-colour range (`white - black`),
/// clamped. `cfa` gives the colour index at `[row % 2][col % 2]`; `black`/`white`
/// are indexed by that colour (RGBE order). Pure — no decode dependency.
pub fn normalize_cfa(
    data: &[u16],
    width: usize,
    height: usize,
    cfa: [[usize; 2]; 2],
    black: [f32; 4],
    white: [f32; 4],
) -> Vec<f32> {
    let mut out = vec![0.0f32; width.saturating_mul(height)];
    if data.len() < out.len() {
        return out; // malformed: short plane ⇒ all-black rather than a panic
    }
    for row in 0..height {
        for col in 0..width {
            let color = cfa[row % 2][col % 2];
            let b = black[color];
            // Guard a degenerate level pair so we never divide by ~0.
            let range = (white[color] - b).max(1.0);
            let v = (data[row * width + col] as f32 - b) / range;
            out[row * width + col] = v.clamp(0.0, 1.0);
        }
    }
    out
}

/// Decode a camera raw file into a normalised [`RawImage`].
///
/// Only single-component CFA mosaics (`cpp == 1`, integer data) are supported
/// for now — the demosaic front end's input. Already-demosaiced (`cpp == 3`) or
/// float-encoded raws return [`Error::Raw`].
pub fn load(path: impl AsRef<std::path::Path>) -> Result<RawImage> {
    let raw = rawloader::decode_file(path).map_err(|e| Error::Raw(format!("{e:?}")))?;

    if raw.cpp != 1 {
        return Err(Error::Raw(format!(
            "only CFA mosaics (cpp=1) are supported yet, got cpp={}",
            raw.cpp
        )));
    }
    let data = match raw.data {
        rawloader::RawImageData::Integer(ref v) => v,
        rawloader::RawImageData::Float(_) => {
            return Err(Error::Raw("float-encoded raw not supported yet".into()))
        }
    };

    let cfa = [
        [raw.cfa.color_at(0, 0), raw.cfa.color_at(0, 1)],
        [raw.cfa.color_at(1, 0), raw.cfa.color_at(1, 1)],
    ];
    // We model only a 2×2 Bayer pattern. Reject anything that isn't truly
    // 2×2-periodic (e.g. Fuji X-Trans is 6×6) rather than silently snapshotting
    // a larger pattern into 2×2 and mis-normalising most photosites. 6 = lcm(2,6)
    // covers the common CFA periods.
    for row in 0..6 {
        for col in 0..6 {
            if raw.cfa.color_at(row, col) != cfa[row % 2][col % 2] {
                return Err(Error::Raw(
                    "non-2x2 CFA (e.g. Fuji X-Trans) not supported yet".into(),
                ));
            }
        }
    }
    // Trust-boundary guard: color_at must index the length-4 RGBE level arrays.
    if cfa.iter().flatten().any(|&c| c >= 4) {
        return Err(Error::Raw("CFA colour index outside RGBE range".into()));
    }
    let black: [f32; 4] = std::array::from_fn(|i| raw.blacklevels[i] as f32);
    let white: [f32; 4] = std::array::from_fn(|i| raw.whitelevels[i] as f32);
    let mosaic = normalize_cfa(data, raw.width, raw.height, cfa, black, white);

    Ok(RawImage {
        width: raw.width,
        height: raw.height,
        cfa,
        wb: raw.wb_coeffs,
        orientation: raw.orientation.to_flips(),
        mosaic,
    })
}

/// Reorient a packed RGBA `f32` image to display orientation, given rawloader's
/// `(transpose, flip_x, flip_y)` decomposition. The flips are applied in source
/// space *then* the transpose (rawloader's convention — verified against a real
/// EXIF-8 portrait raw). Returns the (possibly swapped) dimensions and the
/// reoriented buffer. `(false, false, false)` is a copy.
pub fn apply_orientation(
    rgba: &[f32],
    width: usize,
    height: usize,
    flips: (bool, bool, bool),
) -> (usize, usize, Vec<f32>) {
    let (transpose, flip_x, flip_y) = flips;
    if !transpose && !flip_x && !flip_y {
        return (width, height, rgba.to_vec());
    }
    let (ow, oh) = if transpose { (height, width) } else { (width, height) };
    let mut out = vec![0.0f32; ow * oh * 4];
    for oy in 0..oh {
        for ox in 0..ow {
            // Forward op order is flip_x, flip_y (in source space), THEN
            // transpose — rawloader's `to_flips` convention. (Verified against a
            // real EXIF-8/Rotate270 portrait raw: anything else came out
            // upside-down.) Map each output pixel back to its source.
            let (sx, sy) = if transpose {
                let sr = if flip_y { height - 1 - ox } else { ox };
                let sc = if flip_x { width - 1 - oy } else { oy };
                (sc, sr)
            } else {
                let sc = if flip_x { width - 1 - ox } else { ox };
                let sr = if flip_y { height - 1 - oy } else { oy };
                (sc, sr)
            };
            let sp = (sy * width + sx) * 4;
            let op = (oy * ow + ox) * 4;
            out[op..op + 4].copy_from_slice(&rgba[sp..sp + 4]);
        }
    }
    (ow, oh, out)
}

/// Build darktable's packed Bayer `filters` value from a 2×2 CFA pattern such
/// that `raw::fc_bayer(row, col, filters) == cfa[row % 2][col % 2]` for all
/// `row`/`col`. `fc_bayer`'s field shift is periodic-8 in the row, so setting
/// all eight row-phases from the 2-row pattern reconstructs it exactly.
pub fn filters_from_cfa(cfa: [[usize; 2]; 2]) -> u32 {
    let mut f = 0u32;
    for r in 0u32..8 {
        for c in 0u32..2 {
            let shift = (((r << 1) & 14) + (c & 1)) << 1;
            f |= (cfa[(r % 2) as usize][c as usize] as u32 & 3) << shift;
        }
    }
    f
}

/// Demosaic a normalised Bayer CFA mosaic to packed RGBA `f32` via the migrated
/// 3×3 box kernel (`iop::demosaic::darkroom_demosaic_box3`, rcd.c:86) — a fast,
/// low-quality baseline; the higher-quality PPG/RCD/VNG kernels are migrated and
/// can replace this later. Bayer only (`cfa` must be 2×2; X-Trans is rejected at
/// [`load`]).
pub fn demosaic_box(mosaic: &[f32], width: usize, height: usize, cfa: [[usize; 2]; 2]) -> Vec<f32> {
    let mut out = vec![0.0f32; width.saturating_mul(height).saturating_mul(4)];
    // Guards the malformed/short-plane and zero-dimension cases — the latter so
    // we never hand box3 a zero-length (dangling) pointer.
    if mosaic.len() < width.saturating_mul(height) || out.is_empty() {
        return out;
    }
    let filters = filters_from_cfa(cfa);
    let xtrans = [0u8; 36]; // unused for Bayer (filters != 9) but box3 reads 36 bytes
    // Safety: `mosaic` has ≥ width*height floats, `out` has width*height*4, and
    // `xtrans` is 36 bytes — exactly box3's documented contract.
    unsafe {
        crate::iop::demosaic::darkroom_demosaic_box3(
            out.as_mut_ptr(),
            mosaic.as_ptr(),
            width,
            height,
            filters,
            xtrans.as_ptr(),
        );
    }
    out
}

/// Demosaic a normalised Bayer CFA mosaic to packed RGBA `f32` via the migrated
/// **PPG** (Patterned Pixel Grouping) kernels — sharper, with fewer colour
/// artefacts than [`demosaic_box`]. Ports `demosaic_ppg`'s 3-pixel border
/// interpolation (ppg.c:28-55), then runs the migrated green + red/blue sweeps
/// (margin 0, no pre-median). Falls back to the box demosaic for images too
/// small for the PPG interior ring.
pub fn demosaic_ppg(mosaic: &[f32], width: usize, height: usize, cfa: [[usize; 2]; 2]) -> Vec<f32> {
    let n = width.saturating_mul(height);
    if mosaic.len() < n || n == 0 {
        return vec![0.0f32; n * 4];
    }
    // PPG's green sweep reaches ±3 rows/cols at R/B sites, so it needs a real
    // interior; below 16 (comfortably > 2·3) the box fallback avoids under-running
    // the kernels.
    if width < 16 || height < 16 {
        return demosaic_box(mosaic, width, height, cfa);
    }
    let filters = filters_from_cfa(cfa);
    let mut out = vec![0.0f32; n * 4];

    // Border interpolate the outer 3-pixel ring (ppg.c:28-55): each output
    // channel is the mean of the in-image 3×3 neighbours of that colour, else
    // the photosite's own value for its native colour.
    let (w, h) = (width as i32, height as i32);
    for j in 0..h {
        let mut i = 0i32;
        while i < w {
            if i == 3 && j >= 3 && j < h - 3 {
                i = w - 3; // skip the interior; the kernels fill it
            }
            if i >= w {
                break;
            }
            let mut sum = [0.0f32; 8];
            for y in (j - 1)..=(j + 1) {
                for x in (i - 1)..=(i + 1) {
                    if y >= 0 && x >= 0 && y < h && x < w {
                        let f = crate::raw::fc_bayer(y, x, filters);
                        sum[f] += mosaic[y as usize * width + x as usize];
                        sum[f + 4] += 1.0;
                    }
                }
            }
            let f = crate::raw::fc_bayer(j, i, filters);
            let op = 4 * (j as usize * width + i as usize);
            let own = mosaic[j as usize * width + i as usize].max(0.0);
            for c in 0..3 {
                out[op + c] = if c != f && sum[c + 4] > 0.0 {
                    (sum[c] / sum[c + 4]).max(0.0)
                } else {
                    own
                };
            }
            i += 1;
        }
    }

    // Green then red/blue interpolation via the migrated PPG kernels. `margin`
    // is the interior-skip cursor for tiled ROIs; the untiled full-image path
    // wants a value larger than any image so the skip never fires and the whole
    // interior is processed (demosaic.c:874 uses 100000). i32::MAX/2 makes that
    // hold for *any* representable image — no silent banding above some size —
    // and `margin+3` still can't overflow (the green kernel saturating-adds 3).
    const FULL_FRAME_MARGIN: i32 = i32::MAX / 2;
    // Safety: `out` is `n*4` floats and `mosaic` is `>= n` floats — the contracts
    // both kernels document. `input` and `in_orig` are the same (no pre-median).
    unsafe {
        crate::iop::demosaic::darkroom_demosaic_ppg_green(
            out.as_mut_ptr(), mosaic.as_ptr(), mosaic.as_ptr(),
            width, height, filters, FULL_FRAME_MARGIN,
        );
        crate::iop::demosaic::darkroom_demosaic_ppg_redblue(
            out.as_mut_ptr(), width, height, filters, FULL_FRAME_MARGIN,
        );
    }
    out
}

/// Apply green-normalised white balance to packed RGBA `f32` in place: R and B
/// are scaled by their camera-multiplier ratio to green so neutral scene tones
/// stay neutral. `wb` is the RGBE multipliers. No-op if green isn't a usable
/// positive multiplier (some files report all-ones / zeros).
pub fn apply_white_balance(rgba: &mut [f32], wb: [f32; 4]) {
    let g = wb[1];
    if g <= 0.0 || !g.is_finite() || !wb[0].is_finite() || !wb[2].is_finite() {
        return;
    }
    let (rm, bm) = (wb[0] / g, wb[2] / g);
    for px in rgba.chunks_exact_mut(4) {
        px[0] *= rm;
        px[2] *= bm;
    }
}

impl RawImage {
    /// Demosaic + white-balance this raw into a packed **linear RGBA** `f32`
    /// buffer ready for [`crate::pipeline`]. Returns `(width, height, pixels)`.
    pub fn to_linear_rgba(&self) -> (usize, usize, Vec<f32>) {
        let mut rgba = demosaic_ppg(&self.mosaic, self.width, self.height, self.cfa);
        apply_white_balance(&mut rgba, self.wb);
        // box3 leaves the 4th channel at 0 (it has no contributors); set it
        // opaque so a display upload that honours alpha doesn't render the
        // preview fully transparent.
        for px in rgba.chunks_exact_mut(4) {
            px[3] = 1.0;
        }
        // Reorient sensor → display (handles portrait shots etc.).
        apply_orientation(&rgba, self.width, self.height, self.orientation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_subtracts_black_and_scales_per_colour() {
        // 2x2 RGGB: R=0, G=1, B=2. blacks: R=10,G=20,B=30; whites all 1010 etc.
        let cfa = [[0usize, 1], [1, 2]];
        let black = [10.0, 20.0, 30.0, 0.0];
        let white = [1010.0, 1020.0, 1030.0, 1.0]; // range 1000 for each
        // data row-major: (0,0)=R=510, (0,1)=G=520, (1,0)=G=270, (1,1)=B=530
        let data = [510u16, 520, 270, 530];
        let out = normalize_cfa(&data, 2, 2, cfa, black, white);
        assert!((out[0] - 0.5).abs() < 1e-6); // (510-10)/1000
        assert!((out[1] - 0.5).abs() < 1e-6); // (520-20)/1000
        assert!((out[2] - 0.25).abs() < 1e-6); // (270-20)/1000
        assert!((out[3] - 0.5).abs() < 1e-6); // (530-30)/1000
    }

    #[test]
    fn normalize_tiles_pattern_across_larger_frame() {
        // 4x4 RGGB with distinct per-colour ranges; verify interior photosites
        // pick the right colour's levels via cfa[row%2][col%2].
        let cfa = [[0usize, 1], [1, 2]];
        let black = [0.0; 4];
        let white = [100.0, 200.0, 400.0, 1.0];
        let data = vec![100u16; 16];
        let out = normalize_cfa(&data, 4, 4, cfa, black, white);
        assert!((out[2 * 4 + 2] - 1.0).abs() < 1e-6); // (2,2) R → 100/100
        assert!((out[2 * 4 + 3] - 0.5).abs() < 1e-6); // (2,3) G → 100/200
        assert!((out[3 * 4 + 3] - 0.25).abs() < 1e-6); // (3,3) B → 100/400
        assert!((out[1 * 4 + 0] - 0.5).abs() < 1e-6); // (1,0) G → 100/200
    }

    #[test]
    fn normalize_clamps_and_guards_degenerate_range() {
        let cfa = [[0usize, 0], [0, 0]];
        let black = [100.0; 4];
        let white = [100.0; 4]; // degenerate: range guarded to 1.0
        let data = [50u16, 100, 150, 99];
        let out = normalize_cfa(&data, 2, 2, cfa, black, white);
        assert_eq!(out[0], 0.0); // (50-100)/1 → -50 → clamp 0
        assert_eq!(out[1], 0.0); // (100-100)/1 → 0
        assert_eq!(out[2], 1.0); // (150-100)/1 → 50 → clamp 1
        assert_eq!(out[3], 0.0); // (99-100)/1 → clamp 0
    }

    #[test]
    fn normalize_short_plane_is_all_black_not_panic() {
        let cfa = [[0usize, 1], [1, 2]];
        let out = normalize_cfa(&[5u16], 2, 2, cfa, [0.0; 4], [100.0; 4]);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn load_rejects_missing_file() {
        assert!(matches!(load("/no/such/raw.cr2"), Err(Error::Raw(_))));
    }

    // 2x2 image, channel 0 = row*2+col marker, alpha = 1.
    fn marker_2x2() -> Vec<f32> {
        let mut v = Vec::new();
        for r in 0..2 {
            for c in 0..2 {
                v.extend([(r * 2 + c) as f32, 0.0, 0.0, 1.0]);
            }
        }
        v
    }

    #[test]
    fn orientation_identity_flips_transpose() {
        let img = marker_2x2(); // markers [0,1 / 2,3]
        // identity
        let (w, h, o) = apply_orientation(&img, 2, 2, (false, false, false));
        assert_eq!((w, h), (2, 2));
        assert_eq!(o[0], 0.0);
        // flip_x → columns mirrored: [1,0 / 3,2]
        let (_, _, o) = apply_orientation(&img, 2, 2, (false, true, false));
        assert_eq!(o[0], 1.0);
        // flip_y → rows mirrored: [2,3 / 0,1]
        let (_, _, o) = apply_orientation(&img, 2, 2, (false, false, true));
        assert_eq!(o[0], 2.0);
        // transpose → [0,2 / 1,3] (out[0]=0, out at (col0,row1)=1)
        let (_, _, o) = apply_orientation(&img, 2, 2, (true, false, false));
        assert_eq!(o[0], 0.0);
        assert_eq!(o[(1 * 2 + 0) * 4], 1.0);
    }

    #[test]
    fn orientation_transpose_swaps_dims() {
        // 2 wide × 3 tall, ch0 marker = row*2+col.
        let mut img = Vec::new();
        for r in 0..3 {
            for c in 0..2 {
                img.extend([(r * 2 + c) as f32, 0.0, 0.0, 1.0]);
            }
        }
        let (w, h, o) = apply_orientation(&img, 2, 3, (true, false, false));
        assert_eq!((w, h), (3, 2)); // dims swapped
        assert_eq!(o[0], 0.0); // out(0,0) = src(0,0)
        assert_eq!(o[(0 * 3 + 2) * 4], 4.0); // out(2,0) = src(0,2) = row2col0 = 4

        // Rotate270 = (transpose, flip_x, flip_y) = (true, true, false): the
        // exact path of the verified EXIF-8 portrait ORF. flips run in source
        // space, then transpose.
        let (w, h, o) = apply_orientation(&img, 2, 3, (true, true, false));
        assert_eq!((w, h), (3, 2));
        assert_eq!(o[0], 1.0); // out(0,0) = src(row0, col1) = 1
        assert_eq!(o[(1 * 3 + 2) * 4], 4.0); // out(2,1) = src(row2, col0) = 4
    }

    #[test]
    fn filters_from_cfa_matches_fc_bayer() {
        // For each Bayer arrangement, the reconstructed `filters` must read back
        // through fc_bayer as the original 2×2 pattern, at every row/col phase.
        let patterns = [
            [[0usize, 1], [1, 2]], // RGGB
            [[1usize, 0], [2, 1]], // GRBG
            [[1usize, 2], [0, 1]], // GBRG
            [[2usize, 1], [1, 0]], // BGGR
        ];
        for cfa in patterns {
            let f = filters_from_cfa(cfa);
            for row in 0..9i32 {
                for col in 0..9i32 {
                    assert_eq!(
                        crate::raw::fc_bayer(row, col, f),
                        cfa[(row % 2) as usize][(col % 2) as usize],
                        "cfa {cfa:?} at ({row},{col})"
                    );
                }
            }
        }
    }

    #[test]
    fn demosaic_box_interpolates_rggb_corner() {
        // 2×2 RGGB: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B.
        let cfa = [[0usize, 1], [1, 2]];
        let mosaic = [0.4f32, 0.6, 0.2, 0.8];
        let out = demosaic_box(&mosaic, 2, 2, cfa);
        // pixel (0,0): R=0.4 (self), G=(0.6+0.2)/2=0.4, B=0.8 (the lone B).
        assert!((out[0] - 0.4).abs() < 1e-6, "R {}", out[0]);
        assert!((out[1] - 0.4).abs() < 1e-6, "G {}", out[1]);
        assert!((out[2] - 0.8).abs() < 1e-6, "B {}", out[2]);
    }

    #[test]
    fn to_linear_rgba_composes_demosaic_wb_and_opaque_alpha() {
        let img = RawImage {
            width: 2,
            height: 2,
            cfa: [[0, 1], [1, 2]], // RGGB
            wb: [2.0, 1.0, 4.0, 1.0],
            orientation: (false, false, false),
            mosaic: vec![0.4, 0.6, 0.2, 0.8],
        };
        let (w, h, rgba) = img.to_linear_rgba();
        assert_eq!((w, h), (2, 2));
        assert_eq!(rgba.len(), 16);
        for px in rgba.chunks_exact(4) {
            assert_eq!(px[3], 1.0, "alpha must be opaque");
        }
        // pixel (0,0): demosaic R=0.4,G=0.4,B=0.8 → WB green-normalised (g=1):
        // R=0.4*2=0.8, G=0.4 (unchanged), B=0.8*4=3.2.
        assert!((rgba[0] - 0.8).abs() < 1e-6, "R {}", rgba[0]);
        assert!((rgba[1] - 0.4).abs() < 1e-6, "G {}", rgba[1]);
        assert!((rgba[2] - 3.2).abs() < 1e-6, "B {}", rgba[2]);
    }

    #[test]
    fn ppg_constant_mosaic_is_neutral() {
        // A flat 16x16 RGGB mosaic must demosaic to a flat neutral grey
        // (interpolating a constant gives the constant), interior + border.
        let cfa = [[0usize, 1], [1, 2]];
        let mosaic = vec![0.5f32; 16 * 16];
        let out = demosaic_ppg(&mosaic, 16, 16, cfa);
        assert_eq!(out.len(), 16 * 16 * 4);
        for (idx, px) in out.chunks_exact(4).enumerate() {
            for c in 0..3 {
                assert!(
                    (px[c] - 0.5).abs() < 1e-4,
                    "pixel {idx} ch{c} = {} (not neutral 0.5)",
                    px[c]
                );
            }
        }
    }

    #[test]
    fn ppg_gradient_reconstructs_all_channels() {
        // A smooth horizontal gradient (value depends only on column) must
        // reconstruct to ~itself in EVERY channel at interior pixels. A flat
        // field can't catch a channel left at the vec-init 0.0 or a grossly wrong
        // directional guess (every guess collapses to one value); a gradient can.
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (16usize, 16usize);
        let mosaic: Vec<f32> = (0..w * h)
            .map(|idx| (idx % w) as f32 / (w as f32 - 1.0))
            .collect();
        let out = demosaic_ppg(&mosaic, w, h, cfa);

        // interior pixel (col 8, row 8): every channel ≈ the gradient value 8/15.
        let expect = 8.0 / 15.0;
        let op = (8 * w + 8) * 4;
        for c in 0..3 {
            assert!(
                (out[op + c] - expect).abs() < 0.08,
                "ch{c} = {} (want ~{expect})",
                out[op + c]
            );
        }
        // coverage: no interior pixel left all-zero where the input was non-zero.
        for j in 4..h - 4 {
            for i in 4..w - 4 {
                let p = (j * w + i) * 4;
                assert!(
                    out[p] != 0.0 || out[p + 1] != 0.0 || out[p + 2] != 0.0,
                    "pixel ({i},{j}) RGB all-zero"
                );
            }
        }
    }

    #[test]
    fn ppg_falls_back_to_box_for_tiny_images() {
        // Below the PPG ring size, demosaic_ppg must equal the box demosaic.
        let cfa = [[0usize, 1], [1, 2]];
        let mosaic = vec![0.4f32, 0.6, 0.2, 0.8]; // 2x2
        assert_eq!(
            demosaic_ppg(&mosaic, 2, 2, cfa),
            demosaic_box(&mosaic, 2, 2, cfa)
        );
    }

    #[test]
    fn white_balance_green_normalises_and_guards() {
        let mut px = vec![1.0f32, 1.0, 1.0, 1.0];
        apply_white_balance(&mut px, [2.0, 1.0, 4.0, 1.0]);
        assert_eq!(px, vec![2.0, 1.0, 4.0, 1.0]); // R*2, G*1, B*4
        // green ≤ 0 ⇒ no-op
        let mut p2 = vec![1.0f32, 1.0, 1.0, 1.0];
        apply_white_balance(&mut p2, [2.0, 0.0, 4.0, 1.0]);
        assert_eq!(p2, vec![1.0, 1.0, 1.0, 1.0]);
    }
}
