//! Bayer raw denoise end to end (u7e, parity audit 2.7 raw leg).
//!
//! Ports the batch path of `src/common/ai/restore_raw_bayer.c`
//! (`dt_restore_raw_bayer`) onto the u7b ONNX Runtime session idiom: CFA
//! mosaic in, CFA mosaic out, written to disk by the caller as a CFA DNG
//! (see [`build_cfa_dng`]). X-Trans, Foveon and non-Bayer mosaics are
//! refused with an honest error — there is no silent fallback to another
//! pipeline. The linear (demosaiced) raw variant stays future work.
//!
//! ## Bayer contract, as implemented (C line evidence each)
//!
//! * Origin rule (`restore_raw_bayer.c:42-57` `_bayer_origin`): the packed
//!   channels always start at the CFA's R origin, so non-RGGB sensors are
//!   packed as if they were RGGB. [`CfaPattern::origin`] returns the
//!   `(y0, x0)` in `{0,1}^2` whose site holds R: RGGB `(0,0)`, GRBG
//!   `(0,1)`, GBRG `(1,0)`, BGGR `(1,1)`. Every tile origin below is
//!   shifted by it (`:239-244` `_bayer_tile_geometry`, force_rggb branch —
//!   the default; the NATIVE orientation stays unwired).
//! * Normalize formula (`:289-325` `_pack_bayer_tile`): per site
//!   `(raw - black[site]) / range[site] * wb_norm[ch]`, where `site` is
//!   the position index `((r&1)<<1)|(c&1)` of the MIRRORED coords
//!   (`:317`), `ch = FC(r, c)` (`:319`), and the four planes are
//!   `[R, G1, G2, B]` with `dr = (k>>1)&1, dc = k&1` (`:312-313`).
//!   `range = white - black` floored at `1.0`
//!   (`restore_common.h:173-196` `_compute_cfa_black_range`).
//! * Mirror bounds (`:253-268`): reflections happen inside the
//!   effective-RGGB-cropped rectangle, i.e. the working region shifted by
//!   `(y0, x0)` with an even span — for the full-frame visible region this
//!   port drives, `[y0, y0+2*Hh)` x `[x0, x0+2*Wh)`. The reflection itself
//!   reuses [`super::infer::mirror_coord`] (the `_mirror` of
//!   `restore_common.h:198-210`) via [`mirror_in_range`]
//!   (`restore_common.h:217-221`).
//! * PixelShuffle reassembly (`:604-641`): the model returns 3ch
//!   `2T x 2T` camRGB (`restore.h:128-134`); output sample `(my, mx)` maps
//!   to sensor `(sr, sc)` with `my = 2*O + (sr - sensor_py_base)`
//!   (`:606`), `mx = 2*O + (sc - sensor_px_base)` (`:632`), and the
//!   channel is `FC(sr, sc)` (`:637-639`) — channel 0 is R, 1 is G, 2 is B.
//! * match_gain (`:176-212` `_bayer_gain_match`): `gain = in_mean /
//!   out_mean` over the 4ch packed input vs the 3ch model output, applied
//!   in place, guarded by `|out_mean| > 1e-8` else `1.0`. Per tile, always
//!   on (there is no ABSOLUTE-scale variant handling here).
//! * Remosaic (`:163-170` `_bayer_remosaic_raw`): `raw = model_val /
//!   wb_norm[ch] * range[site] + black[site]`, clipped to `[0, white]`
//!   (`:676-679`) and rounded to u16 (`clipped + 0.5f`, `:679`).
//!
//! ## Deliberate deviations from the C batch path
//!
//! * Seam blending: the C accumulates `ax*ay`-weighted overlap seams
//!   across tiles (`:598-736`). This port writes each tile's core valid
//!   strip only (the u7b strip discipline) — no blending weights.
//! * Visible region: the C crops to the metadata-reported visible region
//!   and copies the margins from the source (`:358-373`). This port drives
//!   full-frame only; the sub-working-region margins (at most one row/col
//!   on odd-sized sensors) are copied from the source the same way.
//! * Strength: the C blends `alpha * raw + (1-alpha) * cfa_in` (`:644-645`).
//!   This port always takes the full model output (`alpha = 1`).
//! * WB mode: the C keys daylight vs as-shot off `ctx->wb_mode` with a
//!   fallback chain (`:139-152`). This port resolves daylight-first with
//!   as-shot fallback (the C default) in [`resolve_wb`]; there is no NONE
//!   mode and no per-model override.
//! * Session: one session per call, no GPU/CPU fallback reload (`:515-526`
//!   has none of that either — it is batch policy, not math).
//!
//! ## CFA metadata sourcing (exact)
//!
//! [`load_bayer_source`] decodes through `rawloader` (the same decoder
//! `crate::rawimage::load` uses) and sources each field as follows:
//!
//! * pattern: `raw.cfa.color_at` classified by
//!   [`crate::rawimage::classify_cfa`]; the 2x2 must map to
//!   [`CfaPattern`] (an `E`-coloured site or anything else is refused).
//! * black: position-indexed `[f32; 4]` — `raw.blacklevels[colour]` read
//!   through the pattern, i.e. `black[((r&1)<<1)|(c&1)]`, because the C
//!   indexes its separate levels by sensor position
//!   (`restore_raw_bayer.c:317`, fed from `raw_black_level_separate`).
//!   `rawloader` resolves masked-border levels into `blacklevels` already
//!   (`RawImage::new`), so this inherits that behaviour.
//! * white: single value `max(whitelevels[0..3])`, falling back to `65535`
//!   when zero — the C reads one `raw_white_point` with the same fallback
//!   (`restore_common.h:178-180`).
//! * wb: [`resolve_wb`] over `raw.xyz_to_cam` (rows 0..2, the
//!   `adobe_XYZ_to_CAM` analogue) and `raw.wb_coeffs`.
//! * make/model/filename: `clean_make`, `clean_model`, and the source file
//!   name — these seed the DNG tags, never invented values.
//! * color matrix: `xyz_to_cam` rows 0..2 flattened row-major (camRGB from
//!   XYZ), else the XYZ-D65 to sRGB fallback the C uses when the source
//!   has no matrix (`src/imageio/imageio_dng.c:157-174`).
//! * as-shot neutral: `1/wb_coeffs` G-normalised... more precisely
//!   max-normalised (`src/imageio/imageio_dng.c:143-155`); omitted when
//!   any coefficient is non-positive.
//!
//! Nothing here reaches the network and no test downloads a model: the
//! fake-session seam ([`run_bayer_tiled`] takes `run_tile`) plus synthetic
//! `.dtmodel` fixtures cover the driver, and a rawloader round-trip covers
//! the DNG writer.

use std::path::{Path, PathBuf};

use super::infer::InferError;
use super::{download, package, registry};

/// Tile overlap in packed (half-res) pixels (`restore_raw_bayer.c:40`
/// `OVERLAP_PACKED`).
pub const O_PACKED: u32 = 32;
/// Manifest stem for the Bayer raw-denoise variant: the payload is
/// `model_bayer.onnx` per the `_load` stem rule, and the tile ladder is
/// `model_bayer.input_sizes` stem-first
/// (`src/common/ai/restore.h:100-104`, `restore.c:186-205`).
pub const MODEL_BAYER_STEM: &str = "model_bayer";
/// Payload filename inside a raw-denoise package (`<stem>.onnx`).
pub const MODEL_BAYER_ONNX_FILE: &str = "model_bayer.onnx";
/// Known raw-denoise release asset name, for the download-on-demand list.
pub const RAWDENOISE_NIND_ASSET: &str = "rawdenoise-nind.dtmodel";
/// D65 white in XYZ, from `colorspaces_inline_conversions.h:472`.
pub const D65_WHITE_XYZ: [f32; 3] = [0.9504, 1.0, 1.0889];

/// The four Bayer CFA layouts, in row-major 2x2 colour-index order
/// (0=R, 1=G, 2=B — the `FC` values `restore_raw_bayer.c:637` reads).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CfaPattern {
    Rggb,
    Bggr,
    Grbg,
    Gbrg,
}

impl CfaPattern {
    /// Row-major 2x2 colour indices (`cfa[row][col]`).
    pub fn layout(self) -> [[usize; 2]; 2] {
        match self {
            Self::Rggb => [[0, 1], [1, 2]],
            Self::Bggr => [[2, 1], [1, 0]],
            Self::Grbg => [[1, 0], [2, 1]],
            Self::Gbrg => [[1, 2], [0, 1]],
        }
    }

    /// The CFA R origin: `(y0, x0)` in `{0,1}^2` whose site holds R, i.e.
    /// the `_bayer_origin` answer (`restore_raw_bayer.c:45-57`) for this
    /// pattern. Tile origins shift by it so packed channel 0 always hits R
    /// (`:239-244`).
    pub fn origin(self) -> (i64, i64) {
        match self {
            Self::Rggb => (0, 0),
            Self::Grbg => (0, 1),
            Self::Gbrg => (1, 0),
            Self::Bggr => (1, 1),
        }
    }

    /// Colour index at sensor `(row, col)`: the `FC` lookup
    /// (`restore_raw_bayer.c:319,637`), 0=R, 1=G, 2=B.
    pub fn fc(self, row: i64, col: i64) -> usize {
        let l = self.layout();
        l[(row & 1) as usize][(col & 1) as usize]
    }

    /// Row-major CFAPattern bytes for the DNG writer (0=R, 1=G, 2=B —
    /// matches `_cfa_bytes_from_filters`, `imageio_dng.c:99-105`).
    pub fn cfa_bytes(self) -> [u8; 4] {
        let l = self.layout();
        [l[0][0] as u8, l[0][1] as u8, l[1][0] as u8, l[1][1] as u8]
    }
}

/// Map a 2x2 colour-index pattern to [`CfaPattern`]. Anything that is not
/// one of the four Bayer layouts — notably an `E` (3) site from a masked
/// or exotic sensor — is refused with an honest error, never guessed.
pub fn pattern_from_cfa2x2(cfa: [[usize; 2]; 2]) -> Result<CfaPattern, InferError> {
    for pat in [CfaPattern::Rggb, CfaPattern::Bggr, CfaPattern::Grbg, CfaPattern::Gbrg] {
        if pat.layout() == cfa {
            return Ok(pat);
        }
    }
    Err(InferError::InvalidArgument(format!(
        "raw denoise needs a Bayer mosaic, got 2x2 pattern {cfa:?}: \
         X-Trans, Foveon and other sensors are not supported"
    )))
}

/// Per-site ranges from position-indexed blacks and the single white:
/// `range[i] = max(white - black[i], 1.0)`
/// (`restore_common.h:191-195`). A non-positive white falls back to 65535
/// (`:178-180`).
pub fn black_ranges(black: [f32; 4], white: f32) -> [f32; 4] {
    let w = if white > 0.0 { white } else { 65535.0 };
    std::array::from_fn(|i| (w - black[i]).max(1.0))
}

/// Daylight WB multipliers (G normalised to 1) from the camera
/// XYZ-to-camRGB matrix: `resp[c] = row_c . D65`, `wb = [G/R, 1, G/B]`
/// (`restore_raw_bayer.c:81-100` `_bayer_wb_daylight`). `None` when the
/// matrix is missing or any response is non-positive (`:94-95`) — the
/// caller falls back to as-shot.
pub fn daylight_wb(xyz_to_cam: [[f32; 3]; 4]) -> Option<[f32; 3]> {
    let mut resp = [0.0f32; 3];
    let mut mag = 0.0f32;
    for c in 0..3 {
        resp[c] = xyz_to_cam[c][0] * D65_WHITE_XYZ[0]
            + xyz_to_cam[c][1] * D65_WHITE_XYZ[1]
            + xyz_to_cam[c][2] * D65_WHITE_XYZ[2];
        mag += xyz_to_cam[c][0].abs() + xyz_to_cam[c][1].abs() + xyz_to_cam[c][2].abs();
    }
    if mag <= 0.0 || resp[0] <= 0.0 || resp[1] <= 0.0 || resp[2] <= 0.0 {
        return None;
    }
    Some([resp[1] / resp[0], 1.0, resp[1] / resp[2]])
}

/// As-shot WB multipliers (G normalised to 1) from the file's wb
/// coefficients (`restore_raw_bayer.c:102-114` `_bayer_wb_as_shot`).
/// `None` when any coefficient is non-positive (`:105-108`).
pub fn as_shot_wb(wb_coeffs: [f32; 4]) -> Option<[f32; 3]> {
    if wb_coeffs[0] <= 0.0 || wb_coeffs[1] <= 0.0 || wb_coeffs[2] <= 0.0 {
        return None;
    }
    let g = wb_coeffs[1];
    Some([wb_coeffs[0] / g, 1.0, wb_coeffs[2] / g])
}

/// WB normalisation with the C default policy: daylight first, as-shot as
/// the fallback (`restore_raw_bayer.c:142-146`, `DT_RESTORE_WB_DAYLIGHT`
/// branch). `{1, 1, 1}` when both are unavailable (the C keeps its
/// `{1,1,1}` init, `:139`).
pub fn resolve_wb(xyz_to_cam: [[f32; 3]; 4], wb_coeffs: [f32; 4]) -> [f32; 3] {
    if let Some(wb) = daylight_wb(xyz_to_cam) {
        return wb;
    }
    if let Some(wb) = as_shot_wb(wb_coeffs) {
        return wb;
    }
    [1.0, 1.0, 1.0]
}

/// One site's model-input value: `(raw - black) / range * wb_norm[ch]`
/// (`restore_raw_bayer.c:318-321`).
#[inline]
pub fn normalize_site(raw: f32, black: f32, range: f32, wb: f32) -> f32 {
    (raw - black) / range * wb
}

/// Mirror-pad reflection within `[lo, hi)`: `lo + _mirror(i - lo, hi -
/// lo)` (`restore_common.h:217-221` `_mirror_in_range`), reusing the u7b
/// [`super::infer::mirror_coord`] instead of a second copy.
pub fn mirror_in_range(i: i64, lo: i64, hi: i64) -> i64 {
    lo + super::infer::mirror_coord(i - lo, hi - lo)
}

/// Pack one `T x T` half-res 4-channel tile from the full CFA buffer:
/// planar `[R, G1, G2, B]`, `4*T*T` f32
/// (`restore_raw_bayer.c:289-325` `_pack_bayer_tile`). `(sr0, sc0)` is the
/// sensor-space origin of the tile's top-left 2x2 block — already shifted
/// by `(y0, x0)` by the caller for force_rggb (`:242-244`) — and
/// `[mir_y_lo, mir_y_hi)` x `[mir_x_lo, mir_x_hi)` are the
/// mirror-reflection bounds. `black`/`range` are position-indexed,
/// `wb_norm` is `[R, G, B]`.
#[allow(clippy::too_many_arguments)]
pub fn pack_bayer_tile(
    cfa: &[u16],
    w: i64,
    pattern: CfaPattern,
    sr0_origin: i64,
    sc0_origin: i64,
    mir_y_lo: i64,
    mir_y_hi: i64,
    mir_x_lo: i64,
    mir_x_hi: i64,
    t: usize,
    black: [f32; 4],
    range: [f32; 4],
    wb_norm: [f32; 3],
) -> Vec<f32> {
    let plane = t * t;
    let mut tile = vec![0f32; 4 * plane];
    let layout = pattern.layout();
    for dy in 0..t {
        let sr0 = sr0_origin + 2 * dy as i64;
        for dx in 0..t {
            let sc0 = sc0_origin + 2 * dx as i64;
            for k in 0..4 {
                let dr = ((k >> 1) & 1) as i64;
                let dc = (k & 1) as i64;
                let r = mirror_in_range(sr0 + dr, mir_y_lo, mir_y_hi);
                let c = mirror_in_range(sc0 + dc, mir_x_lo, mir_x_hi);
                let val = cfa[(r * w + c) as usize] as f32;
                let site = (((r & 1) << 1) | (c & 1)) as usize;
                let ch = layout[(r & 1) as usize][(c & 1) as usize];
                tile[k * plane + dy * t + dx] =
                    normalize_site(val, black[site], range[site], wb_norm[ch]);
            }
        }
    }
    tile
}

/// Scalar match_gain over one tile: scale the 3ch `2T x 2T` model output
/// in place so its mean equals the 4ch packed input mean
/// (`restore_raw_bayer.c:176-212` `_bayer_gain_match`). Returns
/// `(in_mean, out_mean, gain)`; the guard is `|out_mean| > 1e-8` else
/// `1.0` (`:202-203`), and a `gain == 1.0` skips the multiply (`:204`).
pub fn match_gain_in_place(tile_in: &[f32], tile_out: &mut [f32], t: usize) -> (f64, f64, f32) {
    let in_plane = t * t;
    let out_w = 2 * t;
    let out_plane = out_w * out_w;
    let mut in_sum = 0.0f64;
    for k in 0..4 {
        for v in &tile_in[k * in_plane..(k + 1) * in_plane] {
            in_sum += *v as f64;
        }
    }
    let mut out_sum = 0.0f64;
    for k in 0..3 {
        for v in &tile_out[k * out_plane..(k + 1) * out_plane] {
            out_sum += *v as f64;
        }
    }
    let in_mean = in_sum / (4 * in_plane) as f64;
    let out_mean = out_sum / (3 * out_plane) as f64;
    let gain = if out_mean.abs() > 1e-8 {
        (in_mean / out_mean) as f32
    } else {
        1.0
    };
    if gain != 1.0 {
        for v in tile_out.iter_mut() {
            *v *= gain;
        }
    }
    (in_mean, out_mean, gain)
}

/// Shared re-mosaic pixel math: model camRGB value back to a raw ADC
/// value, reversing WB, normalisation and the black shift
/// (`restore_raw_bayer.c:163-170` `_bayer_remosaic_raw`). `ch` is the
/// `FC(sr, sc)` channel the caller read the model value from.
#[inline]
pub fn remosaic_value(
    model_val: f32,
    row: i64,
    col: i64,
    ch: usize,
    wb_norm: [f32; 3],
    black: [f32; 4],
    range: [f32; 4],
) -> f32 {
    let normalized = model_val / wb_norm[ch];
    let site = (((row & 1) << 1) | (col & 1)) as usize;
    normalized * range[site] + black[site]
}

/// The tiled Bayer driver behind [`rawdenoise_bayer`]: pack each tile via
/// [`pack_bayer_tile`], hand it to `run_tile` (which must return 3ch
/// `2T x 2T` f32 — the `tile_out` of `dt_restore_run_patch_bayer`),
/// [`match_gain_in_place`], then re-mosaic the tile's core valid strip
/// into the CFA output. Injecting `run_tile` is the fake-session seam
/// (same shape as the u7b driver): tests pass a pure function and never
/// touch ONNX Runtime. `progress(tile, total)` fires after each completed
/// tile, 1-based and monotonic.
///
/// Tiling is in packed half-res space: `Wh = (w-x0)/2`, `Hh = (h-y0)/2`
/// (`restore_raw_bayer.c:371-372`, full-frame visible region), overlap
/// `O_PACKED`, `step = T-2O`, `ceil` grid (`:379-386`). The output starts
/// as a copy of the source so sub-working-region margins keep original
/// sensor values (`:358-363`).
#[allow(clippy::too_many_arguments)]
pub fn run_bayer_tiled(
    raw: &[u16],
    w: u32,
    h: u32,
    pattern: CfaPattern,
    black: [f32; 4],
    white: f32,
    wb_norm: [f32; 3],
    tile_size: u32,
    mut run_tile: impl FnMut(&[f32], usize) -> Result<Vec<f32>, InferError>,
    mut progress: impl FnMut(u32, u32),
) -> Result<Vec<u16>, InferError> {
    let (wt, ht) = (w as usize, h as usize);
    if raw.len() != wt * ht {
        return Err(InferError::InvalidArgument(format!(
            "raw mosaic is {} u16, expected w*h = {}",
            raw.len(),
            wt * ht
        )));
    }
    let t = tile_size as usize;
    let o = O_PACKED as usize;
    if tile_size <= 2 * O_PACKED {
        return Err(InferError::InvalidArgument(format!(
            "tile size {tile_size} must exceed 2*overlap {}",
            2 * O_PACKED
        )));
    }
    let (y0, x0) = pattern.origin();
    let ww = (w as i64 - x0) / 2;
    let hh = (h as i64 - y0) / 2;
    if ww <= 0 || hh <= 0 {
        return Ok(raw.to_vec());
    }
    let range = black_ranges(black, white);
    let clip_max = if white > 0.0 { white } else { 65535.0 };
    // Mirror-cropped bounds for the full-frame visible region: the working
    // region shifted by (y0, x0) with an even span
    // (`restore_raw_bayer.c:253-268`; `2*Hh`/`2*Wh` are even by
    // construction, and the region already lies inside the buffer).
    let (mir_y_lo, mir_y_hi) = (y0, y0 + 2 * hh);
    let (mir_x_lo, mir_x_hi) = (x0, x0 + 2 * ww);
    let step = t as i64 - 2 * o as i64;
    let cols = (ww + step - 1) / step;
    let rows = (hh + step - 1) / step;
    let total = (cols * rows) as u32;
    let out_plane = 4 * t * t;
    // Output starts as the source clamped to [0, white] like the C
    // (`restore_raw_bayer.c:358-363` initializes `cfa_out[i] =
    // clamp(cfa_in[i]) + 0.5`): margins the tiles never cover keep a
    // clipped source value, never an over-range one. `white` is integral
    // (a u16 level widened exactly), so the `as u16` is exact and the
    // `+ 0.5` is a no-op on integers — same result as the C.
    let white_u16 = white as u16;
    let mut out: Vec<u16> = raw.iter().map(|&v| v.min(white_u16)).collect();
    let mut done = 0u32;
    for ty in 0..rows {
        let py_base = ty * step;
        let py_end = (py_base + step).min(hh);
        for tx in 0..cols {
            let px_base = tx * step;
            let px_end = (px_base + step).min(ww);
            // Tile origin in sensor coords, shifted by (y0, x0) so packed
            // channel 0 always hits R (`:242-244`, `:478-483`: base
            // `2*(p_base - O)` plus the origin shift).
            let sr0_origin = y0 + 2 * (py_base - o as i64);
            let sc0_origin = x0 + 2 * (px_base - o as i64);
            let tile_in = pack_bayer_tile(
                raw,
                w as i64,
                pattern,
                sr0_origin,
                sc0_origin,
                mir_y_lo,
                mir_y_hi,
                mir_x_lo,
                mir_x_hi,
                t,
                black,
                range,
                wb_norm,
            );
            let mut tile_out = run_tile(&tile_in, t)?;
            if tile_out.len() != 3 * out_plane {
                return Err(InferError::InvalidArgument(format!(
                    "session returned {} f32 for a {t}x{t} bayer tile, need {}",
                    tile_out.len(),
                    3 * out_plane
                )));
            }
            match_gain_in_place(&tile_in, &mut tile_out, t);
            // Core valid strip only (no seam blending — see module doc):
            // packed (py, px) owns sensor rows y0+2py..+2 and cols
            // x0+2px..+2; the model sample is at 2*O + offset from the
            // tile's core base (`:573-576`, `:606`, `:632`, `:637-641`).
            let sensor_py_base = y0 + 2 * py_base;
            let sensor_px_base = x0 + 2 * px_base;
            let two_t = 2 * t;
            for py in py_base..py_end {
                let sr = y0 + 2 * py;
                let my = 2 * o as i64 + (sr - sensor_py_base);
                for px in px_base..px_end {
                    let sc = x0 + 2 * px;
                    let mx = 2 * o as i64 + (sc - sensor_px_base);
                    let ch = pattern.fc(sr, sc);
                    let model_val =
                        tile_out[ch * out_plane + my as usize * two_t + mx as usize];
                    let raw_val =
                        remosaic_value(model_val, sr, sc, ch, wb_norm, black, range);
                    let clipped = raw_val.clamp(0.0, clip_max);
                    out[(sr * w as i64 + sc) as usize] = (clipped + 0.5) as u16;
                }
            }
            done += 1;
            progress(done, total);
        }
    }
    Ok(out)
}

// ── Session (4ch packed in, 3ch PixelShuffle out) ────────────────────────────

/// An owned ONNX Runtime session for one `model_bayer.onnx` payload: the
/// same call sequence as the u7b [`super::infer::RtSession`] (see its
/// module doc), adapted to the bayer_v1 shapes — input NCHW `{1,4,T,T}`
/// packed half-res, output `{1,3,2T,2T}` camRGB
/// (`src/common/ai/restore.h:106-134`). Single input, so there is no
/// noise map. Built once per call, used on one thread, then dropped.
pub struct BayerSession {
    session: ort::session::Session,
}

/// Build a session over one `model_bayer.onnx` payload.
pub fn bayer_session_from_onnx(onnx_path: &Path) -> Result<BayerSession, InferError> {
    if !onnx_path.is_file() {
        return Err(InferError::Io(format!(
            "model payload not found: {}",
            onnx_path.display()
        )));
    }
    let mut builder = match ort::session::Session::builder() {
        Ok(b) => b,
        Err(e) => return Err(InferError::Ort(e.message().to_string())),
    };
    let session = match builder.commit_from_file(onnx_path) {
        Ok(s) => s,
        Err(e) => return Err(InferError::Ort(e.message().to_string())),
    };
    Ok(BayerSession { session })
}

impl BayerSession {
    /// Run one planar `{1,4,T,T}` f32 tile and return the model's planar
    /// `{1,3,2T,2T}` output. Anything else is an error, not a panic.
    pub fn run_tile(&mut self, planar: &[f32], t: usize) -> Result<Vec<f32>, InferError> {
        if planar.len() != 4 * t * t {
            return Err(InferError::InvalidArgument(
                "tile planar buffer is not 4*T*T f32".to_string(),
            ));
        }
        let shape: [i64; 4] = [1i64, 4i64, t as i64, t as i64];
        let input = match ort::value::TensorRef::from_array_view((shape, planar)) {
            Ok(v) => v,
            Err(e) => return Err(InferError::Ort(e.message().to_string())),
        };
        let outputs = match self.session.run(ort::inputs![input]) {
            Ok(o) => o,
            Err(e) => return Err(InferError::Ort(e.message().to_string())),
        };
        if outputs.len() == 0 {
            return Err(InferError::Ort("model produced no outputs".to_string()));
        }
        let out0 = &outputs[0];
        let (shape_out, data) = match out0.try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(e) => return Err(InferError::Ort(e.message().to_string())),
        };
        let dims: &[i64] = shape_out;
        let two_t = 2 * t as i64;
        if dims.len() < 4 || dims[1] != 3 || dims[2] != two_t || dims[3] != two_t {
            return Err(InferError::Ort(format!(
                "bayer model returned spatial output {dims:?}; \
                 bayer_v1 models must return 3ch 2T x 2T"
            )));
        }
        if data.len() != 3 * two_t as usize * two_t as usize {
            return Err(InferError::Ort(format!(
                "bayer model output has {} f32, expected 3*{two_t}*{two_t}",
                data.len()
            )));
        }
        Ok(data.to_vec())
    }
}

/// Full Bayer raw denoise of one CFA mosaic (`w*h` u16 sensor values)
/// into a denoised CFA mosaic of the same layout and size, via
/// [`run_bayer_tiled`] over a [`BayerSession`]. `tile_size` is the
/// manifest-declared static input dim for the `model_bayer` stem;
/// `progress(tile, total)` fires after each completed tile.
#[allow(clippy::too_many_arguments)]
pub fn rawdenoise_bayer(
    raw: &[u16],
    w: u32,
    h: u32,
    cfa: CfaPattern,
    black: [f32; 4],
    white: f32,
    wb: [f32; 3],
    onnx_path: &Path,
    tile_size: u32,
    progress: impl FnMut(u32, u32),
) -> Result<Vec<u16>, InferError> {
    if raw.len() != w as usize * h as usize {
        return Err(InferError::InvalidArgument(format!(
            "raw mosaic is {} u16, expected w*h = {}",
            raw.len(),
            w as usize * h as usize
        )));
    }
    let mut session = bayer_session_from_onnx(onnx_path)?;
    run_bayer_tiled(
        raw,
        w,
        h,
        cfa,
        black,
        white,
        wb,
        tile_size,
        |planar, t| session.run_tile(planar, t),
        progress,
    )
}

// ── Model handling (download + unpack + manifest resolution) ─────────────────

/// One known raw-denoise release, for the download-on-demand picker rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnownRawModel {
    pub name: String,
    /// Approximate size in MiB — display only, never used for logic.
    pub size_mi_b: u32,
}

/// The known download-on-demand entries: the RawNIND Bayer raw-denoise
/// model. Size is the release asset `size` field, rounded (57,700,134 B on
/// release-5.6.0) — display only, never used for logic.
pub fn known_rawdenoise_models() -> Vec<KnownRawModel> {
    vec![KnownRawModel { name: RAWDENOISE_NIND_ASSET.to_string(), size_mi_b: 55 }]
}

/// `rawdenoise-*.dtmodel` archives present in the store, filename-sorted.
/// A missing/unreadable store yields an empty vec — the picker then shows
/// only the download-on-demand rows.
pub fn downloaded_rawdenoise_models() -> Vec<String> {
    let Some(dir) = download::models_dir() else {
        return Vec::new();
    };
    let Some(read) = std::fs::read_dir(&dir).ok() else {
        return Vec::new();
    };
    let mut out: Vec<String> = read
        .flatten()
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .filter_map(|e| e.path().file_name().map(|n| n.to_string_lossy().to_string()))
        .filter(|n| n.starts_with("rawdenoise-") && n.ends_with(".dtmodel"))
        .collect();
    out.sort();
    out
}

/// Download-or-verify one [`KnownRawModel`] into the store: the same
/// sha256-gated path as the u7b/u7d ensure fns, parameterised by name.
pub fn ensure_rawdenoise_model(
    model: &KnownRawModel,
    assets: &[registry::ModelAsset],
    progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, InferError> {
    let Some(asset) = registry::find_asset(assets, &model.name) else {
        return Err(InferError::Registry(format!(
            "release has no {} asset",
            model.name
        )));
    };
    let Some(sha) = asset.sha256_hex() else {
        return Err(InferError::Registry(format!(
            "release asset {} carries no sha256 digest",
            model.name
        )));
    };
    let Some(dest) = download::model_store_path(&model.name) else {
        return Err(InferError::Io("model store is unavailable".to_string()));
    };
    let dest_ref = dest.clone();
    match download::ensure_asset(&asset.download_url, &dest_ref, sha, asset.size, progress) {
        Ok(_) => Ok(dest),
        Err(e) => Err(InferError::Download(e.to_string())),
    }
}

/// The runnable form of a downloaded raw-denoise model.
#[derive(Clone, Debug)]
pub struct PreparedRawModel {
    pub onnx_path: PathBuf,
    pub tile_size: u32,
}

/// Turn a downloaded raw-denoise archive into something the session can
/// load: unpack it (only when the `model_bayer.onnx` payload is missing)
/// and resolve the manifest-declared tile size stem-first —
/// `model_bayer` `input_sizes` first entry, else the top-level fallback
/// (`restore.c:186-205` `_resolve_tile_size`). A package declaring
/// neither is refused. A declared `model_bayer.input_kind` that is not
/// `bayer_v1` is a hard error — a declared-but-unknown input kind must
/// never silently load (`restore.h:136-139`); a missing label keeps the
/// back-compat `bayer_v1` reading.
pub fn prepare_rawdenoise_model(asset_name: &str) -> Result<PreparedRawModel, InferError> {
    let Some(archive) = download::model_store_path(asset_name) else {
        return Err(InferError::Io("model store is unavailable".to_string()));
    };
    if !archive.is_file() {
        return Err(InferError::MissingModel(asset_name.to_string()));
    }
    let stem = asset_name.strip_suffix(".dtmodel").unwrap_or(asset_name);
    if stem.is_empty()
        || stem.contains('/')
        || stem.contains('\\')
        || stem == ".."
        || stem == "."
    {
        return Err(InferError::InvalidArgument(format!(
            "unsafe model name {asset_name:?}"
        )));
    }
    let Some(base) = download::models_dir() else {
        return Err(InferError::Io("model store is unavailable".to_string()));
    };
    let unpack_dir = base.join(stem);
    let payload = unpack_dir.join(MODEL_BAYER_ONNX_FILE);
    if !payload.is_file() {
        let _ = std::fs::remove_dir_all(&unpack_dir);
        let arc = archive.clone();
        let dir = unpack_dir.clone();
        match package::unpack_dtmodel(&arc, &dir) {
            Ok(_) => {}
            Err(e) => return Err(InferError::Package(e.to_string())),
        }
    }
    if !payload.is_file() {
        return Err(InferError::Package(format!(
            "archive {asset_name} has no {MODEL_BAYER_ONNX_FILE} payload"
        )));
    }
    let manifest = match package::load_manifest_from_dir(&unpack_dir) {
        Ok(m) => m,
        Err(e) => return Err(InferError::Package(e.to_string())),
    };
    if let Some(kind) = package::variant_string(&manifest, MODEL_BAYER_STEM, "input_kind") {
        if kind != "bayer_v1" {
            return Err(InferError::Package(format!(
                "model {asset_name} declares {MODEL_BAYER_STEM}.input_kind = {kind:?}, \
                 want \"bayer_v1\""
            )));
        }
    }
    let Some(tile_size) = package::variant_tile_size(&manifest, MODEL_BAYER_STEM)
        .and_then(|t| if t > 0 { Some(t as u32) } else { None })
    else {
        return Err(InferError::Package(format!(
            "model {asset_name} declares no input_sizes"
        )));
    };
    Ok(PreparedRawModel { onnx_path: payload, tile_size })
}

// ── Source loading (raw file to mosaic + metadata) ───────────────────────────

/// Everything [`rawdenoise_bayer`] and [`build_cfa_dng`] need for one
/// source file: the native-u16 CFA mosaic plus the metadata sourced per
/// the module doc. Built by [`load_bayer_source`].
#[derive(Clone, Debug)]
pub struct BayerSource {
    pub raw: Vec<u16>,
    pub w: u32,
    pub h: u32,
    pub pattern: CfaPattern,
    pub black: [f32; 4],
    pub white: f32,
    pub wb: [f32; 3],
    pub make: String,
    pub model: String,
    pub filename: String,
    /// camRGB-from-XYZ 3x3 row-major (the DNG ColorMatrix1); sRGB-D65
    /// fallback when the file carries no matrix.
    pub color_matrix: [f32; 9],
    /// Max-normalised as-shot neutral (the DNG AsShotNeutral); `None`
    /// when the file carries no usable coefficients.
    pub as_shot_neutral: Option<[f32; 3]>,
}

/// XYZ-D65 to sRGB fallback matrix, row-major — the matrix the C
/// substitutes when the source has none (`imageio_dng.c:157-174`, the
/// `xyz_to_srgb_d65` values from `colorspaces_inline_conversions.h:479`,
/// which is the direction DNG ColorMatrix1 requires). Do NOT use the
/// sRGB→XYZ direction here: those values describe the inverse transform
/// and would write wrong colours on reimport.
#[allow(clippy::excessive_precision)] // values verbatim from the C header;
// trimming digits to silence the lint would risk diverging from it.
pub const SRGB_FALLBACK_MATRIX: [f32; 9] = [
    3.2404542, -1.5371385, -0.4985314, //
    -0.9692660, 1.8760108, 0.0415560, //
    0.0556434, -0.2040259, 1.0572252,
];

/// Decode `path` to its native-u16 CFA mosaic plus per-image metadata.
/// Non-mosaic files (already-demosaiced, float-encoded), X-Trans sensors,
/// and non-Bayer patterns are refused with an honest error — the caller
/// surfaces the message as a status line, never a crash.
pub fn load_bayer_source(path: &Path) -> Result<BayerSource, InferError> {
    let raw = match rawloader::decode_file(path) {
        Ok(r) => r,
        Err(e) => {
            return Err(InferError::InvalidArgument(format!(
                "raw denoise cannot decode {}: {e:?}",
                path.display()
            )))
        }
    };
    if raw.cpp != 1 {
        return Err(InferError::InvalidArgument(format!(
            "raw denoise needs a Bayer CFA mosaic, {} is already demosaiced (cpp={})",
            path.display(),
            raw.cpp
        )));
    }
    let data = match raw.data {
        rawloader::RawImageData::Integer(ref v) => v.clone(),
        rawloader::RawImageData::Float(_) => {
            return Err(InferError::InvalidArgument(format!(
                "raw denoise needs a Bayer CFA mosaic, {} is float-encoded",
                path.display()
            )))
        }
    };
    let pattern = match crate::rawimage::classify_cfa(|r, c| raw.cfa.color_at(r, c)) {
        Ok(crate::rawimage::CfaKind::Bayer(cfa2)) => pattern_from_cfa2x2(cfa2)?,
        Ok(crate::rawimage::CfaKind::Xtrans(_)) => {
            return Err(InferError::InvalidArgument(format!(
                "raw denoise needs a Bayer mosaic, {} is X-Trans: \
                 X-Trans sensors are not supported",
                path.display()
            )))
        }
        Err(e) => {
            return Err(InferError::InvalidArgument(format!(
                "raw denoise needs a Bayer mosaic, {}: {e}",
                path.display()
            )))
        }
    };
    // Position-indexed blacks through the pattern: black[site] carries the
    // level of the colour living at that sensor position (see module doc).
    let layout = pattern.layout();
    let mut black = [0.0f32; 4];
    for r in 0..2 {
        for c in 0..2 {
            black[r * 2 + c] = raw.blacklevels[layout[r][c]] as f32;
        }
    }
    let mut white = raw.whitelevels[0..3].iter().fold(0u16, |m, v| m.max(*v)) as f32;
    if white <= 0.0 {
        white = 65535.0;
    }
    let wb = resolve_wb(raw.xyz_to_cam, raw.wb_coeffs);
    let mut mag = 0.0f32;
    for row in raw.xyz_to_cam.iter().take(3) {
        for v in row.iter().take(3) {
            mag += v.abs();
        }
    }
    let color_matrix = if mag > 0.0 {
        let mut m = [0.0f32; 9];
        for k in 0..3 {
            for i in 0..3 {
                m[k * 3 + i] = raw.xyz_to_cam[k][i];
            }
        }
        m
    } else {
        SRGB_FALLBACK_MATRIX
    };
    let as_shot_neutral = if raw.wb_coeffs[0] > 0.0
        && raw.wb_coeffs[1] > 0.0
        && raw.wb_coeffs[2] > 0.0
    {
        let mut n = [0.0f32; 3];
        for (v, &c) in n.iter_mut().zip(raw.wb_coeffs.iter()) {
            *v = 1.0 / c;
        }
        let m = n[0].max(n[1]).max(n[2]);
        if m > 0.0 {
            for v in n.iter_mut() {
                *v /= m;
            }
            Some(n)
        } else {
            None
        }
    } else {
        None
    };
    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    Ok(BayerSource {
        raw: data,
        w: raw.width as u32,
        h: raw.height as u32,
        pattern,
        black,
        white,
        wb,
        make: raw.clean_make.clone(),
        model: raw.clean_model.clone(),
        filename,
        color_matrix,
        as_shot_neutral,
    })
}

// ── Minimal CFA DNG writer (pure byte building, no new deps) ─────────────────
///
/// Single-IFD, uncompressed, little-endian TIFF mirroring EXACTLY the
/// required tags of the no-preview path of `dt_imageio_dng_write_cfa_bayer`
/// (`src/imageio/imageio_dng.c:227-380`):
///
/// * payload IFD (`:273-283`): SubFileType=0, ImageWidth, ImageLength,
///   BitsPerSample=16, SamplesPerPixel=1, PlanarConfig=contig,
///   Photometric=CFA (32803), SampleFormat=uint, Compression=none,
///   Orientation=topleft, RowsPerStrip, StripOffsets, StripByteCounts.
/// * CFA tags (`:300-315`): CFARepeatPatternDim={2,2}, CFAPattern (native
///   2x2 order — the crop-origin rotation `:297-310` is the identity when
///   ActiveArea starts at 0,0, which it always does here),
///   CFAPlaneColor={0,1,2}, CFALayout=rectangular.
/// * levels (`:317-342`): BlackLevelRepeatDim={2,2}, BlackLevel (4),
///   WhiteLevel (1).
/// * visible region (`:347-357`): `ActiveArea=[0,0,h,w]` (full buffer —
///   there are no optical-black margins in a denoised mosaic),
///   DefaultScale={1,1}, DefaultCropOrigin={0,0}, DefaultCropSize={w,h}.
/// * shared metadata (`:111-179` `_set_dng_shared_metadata`, written on
///   the single IFD when there is no preview IFD, `:285-286`):
///   X/YResolution=300, ResolutionUnit=inch, Software, Make/Model (always
///   written, possibly empty — rawloader *requires* both tags to fall
///   back from its camera table), DNGVersion={1,4,0,0},
///   DNGBackwardVersion={1,2,0,0}, BaselineExposure=0, AsShotNeutral
///   (omitted when missing, `:143-155`), ColorMatrix1 (+ sRGB fallback,
///   `:157-174`), CalibrationIlluminant1=D65 (21, `:175-178`).
/// * OriginalRawFileName (`:128-134`, UTF-8 bytes, no NUL) carries the
///   source filename when known.
///
/// Tag numbers are the DNG ones (verified against
/// `src/external/rawspeed/src/librawspeed/tiff/TiffTag.h`, darktable's
/// own reader): CFAPLANECOLOR 50710, CFALAYOUT 50711,
/// BLACKLEVELREPEATDIM 50713, BLACKLEVEL 50714, WHITELEVEL 50717,
/// DEFAULTSCALE 50718, DEFAULTCROPORIGIN 50719, DEFAULTCROPSIZE 50720,
/// ORIGINALRAWFILENAME 50827, ACTIVEAREA 50829.
///
/// Deviations from the C writer, all documented:
///
/// * RowsPerStrip covers the whole frame (one strip) instead of
///   `TIFFDefaultStripSize` — valid TIFF, simpler offsets.
/// * BlackLevel is SHORT (spec-legal alongside LONG/RATIONAL; both
///   rawloader's `get_f32` and rawspeed's `getFloat` read it).
/// * No preview IFD / SubIFD layout (the no-preview path only) and no
///   EXIF-blob embedding (`dt_exif_write_blob`, `:370-374`).
/// * Software advertises `c41 <crate version>` instead of
///   `darktable <version>`.
///
/// Tag numbers above are the DNG ones; the entry order below is ascending.
pub mod dng_tag {
    pub const SUBFILETYPE: u16 = 254;
    pub const IMAGEWIDTH: u16 = 256;
    pub const IMAGELENGTH: u16 = 257;
    pub const BITSPERSAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC: u16 = 262;
    pub const MAKE: u16 = 271;
    pub const MODEL: u16 = 272;
    pub const STRIPOFFSETS: u16 = 273;
    pub const ORIENTATION: u16 = 274;
    pub const SAMPLESPERPIXEL: u16 = 277;
    pub const ROWSPERSTRIP: u16 = 278;
    pub const STRIPBYTECOUNTS: u16 = 279;
    pub const XRESOLUTION: u16 = 282;
    pub const YRESOLUTION: u16 = 283;
    pub const PLANARCONFIG: u16 = 284;
    pub const RESOLUTIONUNIT: u16 = 296;
    pub const SOFTWARE: u16 = 305;
    pub const SAMPLEFORMAT: u16 = 339;
    pub const CFAREPEATPATTERNDIM: u16 = 33421;
    pub const CFAPATTERN: u16 = 33422;
    pub const CFAPLANECOLOR: u16 = 50710;
    pub const CFALAYOUT: u16 = 50711;
    pub const BLACKLEVELREPEATDIM: u16 = 50713;
    pub const BLACKLEVEL: u16 = 50714;
    pub const WHITELEVEL: u16 = 50717;
    pub const DEFAULTSCALE: u16 = 50718;
    pub const DEFAULTCROPORIGIN: u16 = 50719;
    pub const DEFAULTCROPSIZE: u16 = 50720;
    pub const COLORMATRIX1: u16 = 50721;
    pub const ASSHOTNEUTRAL: u16 = 50728;
    pub const BASELINEEXPOSURE: u16 = 50730;
    pub const CALIBRATIONILLUMINANT1: u16 = 50778;
    pub const ORIGINALRAWFILENAME: u16 = 50827;
    pub const ACTIVEAREA: u16 = 50829;
    pub const DNGVERSION: u16 = 50706;
    pub const DNGBACKWARDVERSION: u16 = 50707;
}

/// TIFF field types used by the writer.
mod tiff_type {
    pub const BYTE: u16 = 1;
    pub const ASCII: u16 = 2;
    pub const SHORT: u16 = 3;
    pub const LONG: u16 = 4;
    pub const RATIONAL: u16 = 5;
    pub const UNDEFINED: u16 = 7;
    pub const SRATIONAL: u16 = 10;
}

/// Per-image tag values for [`build_cfa_dng`]: sourced from the caller
/// (ultimately [`BayerSource`]) — never invented EXIF.
pub struct DngMeta<'a> {
    pub make: &'a str,
    pub model: &'a str,
    /// Source filename for OriginalRawFileName; empty omits the tag.
    pub filename: &'a str,
    pub color_matrix: [f32; 9],
    pub as_shot_neutral: Option<[f32; 3]>,
    /// Position-indexed black levels `[R, G1, G2, B]`-by-position
    /// (the C rotates its separate levels by the ActiveArea origin;
    /// ours starts at 0,0 so native order stands).
    pub black: [f32; 4],
    pub white: u32,
    pub pattern: CfaPattern,
}

impl<'a> DngMeta<'a> {
    /// Tag values straight from a decoded [`BayerSource`].
    pub fn from_source(src: &'a BayerSource) -> Self {
        Self {
            make: &src.make,
            model: &src.model,
            filename: &src.filename,
            color_matrix: src.color_matrix,
            as_shot_neutral: src.as_shot_neutral,
            black: src.black,
            white: src.white as u32,
            pattern: src.pattern,
        }
    }
}

fn push_u16_le(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Build a complete single-IFD CFA DNG in memory: TIFF header + IFD0 +
/// data area + the raw strip. Pure bytes, no I/O — [`write_cfa_dng`]
/// adds the atomic file write.
pub fn build_cfa_dng(cfa: &[u16], w: u32, h: u32, meta: &DngMeta) -> Result<Vec<u8>, InferError> {
    if w == 0 || h == 0 {
        return Err(InferError::InvalidArgument(
            "cannot write a DNG with zero width or height".to_string(),
        ));
    }
    // u64 arithmetic: w*h*2 in u32 wraps past ~2 GPix (debug-panic).
    let pixels = w as u64 * h as u64;
    if pixels > u32::MAX as u64 / 2 {
        return Err(InferError::InvalidArgument(format!(
            "cfa mosaic {w}x{h} exceeds the 32-bit strip range"
        )));
    }
    if cfa.len() != w as usize * h as usize {
        return Err(InferError::InvalidArgument(format!(
            "cfa mosaic is {} u16, expected w*h = {}",
            cfa.len(),
            w as usize * h as usize
        )));
    }
    let software = format!("c41 {}", env!("CARGO_PKG_VERSION"));
    let black_u16: [u16; 4] =
        std::array::from_fn(|i| (meta.black[i].clamp(0.0, 65535.0) + 0.5) as u16);
    let white = if meta.white == 0 { 65535 } else { meta.white };
    // One IFD entry: (tag, type, count, payload). Payloads longer than 4
    // bytes spill into the data area; the assembler below sorts by tag
    // (strict readers require ascending order) and patches offsets.
    let mut entries: Vec<(u16, u16, u32, Vec<u8>)> = Vec::new();
    let entry = |tag: u16, typ: u16, payload: Vec<u8>, count: u32| (tag, typ, count, payload);
    let short1 = |v: u16| v.to_le_bytes().to_vec();
    let short2 = |a: u16, b: u16| vec![a as u8, (a >> 8) as u8, b as u8, (b >> 8) as u8];
    let long1 = |v: u32| v.to_le_bytes().to_vec();
    entries.push(entry(dng_tag::SUBFILETYPE, tiff_type::LONG, long1(0), 1));
    entries.push(entry(dng_tag::IMAGEWIDTH, tiff_type::LONG, long1(w), 1));
    entries.push(entry(dng_tag::IMAGELENGTH, tiff_type::LONG, long1(h), 1));
    entries.push(entry(dng_tag::BITSPERSAMPLE, tiff_type::SHORT, short1(16), 1));
    entries.push(entry(dng_tag::COMPRESSION, tiff_type::SHORT, short1(1), 1));
    entries.push(entry(dng_tag::PHOTOMETRIC, tiff_type::SHORT, short1(32803), 1));
    let mut make = meta.make.as_bytes().to_vec();
    make.push(0);
    entries.push(entry(dng_tag::MAKE, tiff_type::ASCII, make, meta.make.len() as u32 + 1));
    // UniqueCameraModel (imageio_dng.c:125-126) is deliberately omitted:
    // optional per the DNG spec and irrelevant to decode, so the source
    // value is dropped rather than carried.
    let mut model = meta.model.as_bytes().to_vec();
    model.push(0);
    entries.push(entry(dng_tag::MODEL, tiff_type::ASCII, model, meta.model.len() as u32 + 1));
    // StripOffsets value patched after layout (strip lands last).
    entries.push(entry(dng_tag::STRIPOFFSETS, tiff_type::LONG, long1(0), 1));
    entries.push(entry(dng_tag::ORIENTATION, tiff_type::SHORT, short1(1), 1));
    entries.push(entry(dng_tag::SAMPLESPERPIXEL, tiff_type::SHORT, short1(1), 1));
    entries.push(entry(dng_tag::ROWSPERSTRIP, tiff_type::LONG, long1(h), 1));
    entries.push(entry(
        dng_tag::STRIPBYTECOUNTS,
        tiff_type::LONG,
        long1(w * h * 2),
        1,
    ));
    entries.push(entry(dng_tag::XRESOLUTION, tiff_type::RATIONAL, long1(300).into_iter().chain(long1(1)).collect::<Vec<u8>>(), 1));
    entries.push(entry(dng_tag::YRESOLUTION, tiff_type::RATIONAL, long1(300).into_iter().chain(long1(1)).collect::<Vec<u8>>(), 1));
    entries.push(entry(dng_tag::PLANARCONFIG, tiff_type::SHORT, short1(1), 1));
    entries.push(entry(dng_tag::RESOLUTIONUNIT, tiff_type::SHORT, short1(2), 1));
    let mut sw = software.into_bytes();
    sw.push(0);
    entries.push(entry(dng_tag::SOFTWARE, tiff_type::ASCII, sw.clone(), sw.len() as u32));
    entries.push(entry(dng_tag::SAMPLEFORMAT, tiff_type::SHORT, short1(1), 1));
    entries.push(entry(
        dng_tag::CFAREPEATPATTERNDIM,
        tiff_type::SHORT,
        short2(2, 2),
        2,
    ));
    let pat = meta.pattern.cfa_bytes();
    entries.push(entry(dng_tag::CFAPATTERN, tiff_type::BYTE, pat.to_vec(), 4));
    entries.push(entry(dng_tag::CFAPLANECOLOR, tiff_type::BYTE, vec![0, 1, 2], 3));
    entries.push(entry(dng_tag::CFALAYOUT, tiff_type::SHORT, short1(1), 1));
    entries.push(entry(
        dng_tag::BLACKLEVELREPEATDIM,
        tiff_type::SHORT,
        short2(2, 2),
        2,
    ));
    let mut bl = Vec::with_capacity(8);
    for b in black_u16 {
        bl.extend_from_slice(&b.to_le_bytes());
    }
    entries.push(entry(dng_tag::BLACKLEVEL, tiff_type::SHORT, bl, 4));
    entries.push(entry(dng_tag::WHITELEVEL, tiff_type::LONG, long1(white), 1));
    let mut aa = Vec::with_capacity(16);
    for v in [0u32, 0, h, w] {
        aa.extend_from_slice(&v.to_le_bytes());
    }
    entries.push(entry(dng_tag::ACTIVEAREA, tiff_type::LONG, aa, 4));
    let mut ds = Vec::with_capacity(16);
    for (n, d) in [(1u32, 1u32), (1, 1)] {
        ds.extend_from_slice(&n.to_le_bytes());
        ds.extend_from_slice(&d.to_le_bytes());
    }
    entries.push(entry(dng_tag::DEFAULTSCALE, tiff_type::RATIONAL, ds, 2));
    let mut dco = Vec::with_capacity(16);
    for _ in 0..2 {
        dco.extend_from_slice(&0u32.to_le_bytes());
        dco.extend_from_slice(&1u32.to_le_bytes());
    }
    entries.push(entry(dng_tag::DEFAULTCROPORIGIN, tiff_type::RATIONAL, dco, 2));
    let mut dcs = Vec::with_capacity(16);
    for v in [w, h] {
        dcs.extend_from_slice(&v.to_le_bytes());
        dcs.extend_from_slice(&1u32.to_le_bytes());
    }
    entries.push(entry(dng_tag::DEFAULTCROPSIZE, tiff_type::RATIONAL, dcs, 2));
    entries.push(entry(dng_tag::DNGVERSION, tiff_type::BYTE, vec![1, 4, 0, 0], 4));
    entries.push(entry(
        dng_tag::DNGBACKWARDVERSION,
        tiff_type::BYTE,
        vec![1, 2, 0, 0],
        4,
    ));
    entries.push(entry(
        dng_tag::BASELINEEXPOSURE,
        tiff_type::SRATIONAL,
        long1(0).into_iter().chain(long1(1)).collect::<Vec<u8>>(),
        1,
    ));
    if let Some(n) = meta.as_shot_neutral {
        let mut buf = Vec::with_capacity(24);
        for v in n {
            // RATIONAL with 1e6 denominator, mirroring the float DNG
            // writer (`imageio_dng.c:676-683`).
            let num = (v * 1_000_000.0).round() as u32;
            buf.extend_from_slice(&num.to_le_bytes());
            buf.extend_from_slice(&1_000_000u32.to_le_bytes());
        }
        entries.push(entry(dng_tag::ASSHOTNEUTRAL, tiff_type::RATIONAL, buf, 3));
    }
    {
        // SRATIONAL at 1e4 denominator, like the float writer's `den`
        // (`imageio_dng.c:665-673`).
        let mut buf = Vec::with_capacity(72);
        for v in meta.color_matrix {
            let num = (v * 10000.0).round() as i32;
            buf.extend_from_slice(&num.to_le_bytes());
            buf.extend_from_slice(&10000i32.to_le_bytes());
        }
        entries.push(entry(dng_tag::COLORMATRIX1, tiff_type::SRATIONAL, buf, 9));
    }
    entries.push(entry(
        dng_tag::CALIBRATIONILLUMINANT1,
        tiff_type::SHORT,
        short1(21),
        1,
    ));
    if !meta.filename.is_empty() {
        entries.push(entry(
            dng_tag::ORIGINALRAWFILENAME,
            tiff_type::UNDEFINED,
            meta.filename.as_bytes().to_vec(),
            meta.filename.len() as u32,
        ));
    }
    entries.sort_by_key(|e| e.0);
    // Layout: header (8) + count (2) + entries (12 each) + next-IFD (4),
    // then the spilled payloads in tag order, then the strip.
    let n = entries.len();
    let data_off = (8 + 2 + n * 12 + 4) as u32;
    let mut ifd: Vec<u8> = Vec::with_capacity(data_off as usize);
    ifd.extend_from_slice(b"II");
    push_u16_le(&mut ifd, 42);
    push_u32_le(&mut ifd, 8);
    push_u16_le(&mut ifd, n as u16);
    let mut spill: Vec<u8> = Vec::new();
    for (tag, typ, count, payload) in &entries {
        push_u16_le(&mut ifd, *tag);
        push_u16_le(&mut ifd, *typ);
        push_u32_le(&mut ifd, *count);
        if payload.len() <= 4 {
            let mut inline = [0u8; 4];
            inline[..payload.len()].copy_from_slice(payload);
            ifd.extend_from_slice(&inline);
        } else {
            push_u32_le(&mut ifd, data_off + spill.len() as u32);
            spill.extend_from_slice(payload);
        }
    }
    push_u32_le(&mut ifd, 0);
    // StripOffsets never spills (LONG count 1) — patch it to the strip
    // address directly.
    let strip_off = data_off + spill.len() as u32;
    let pos = 8 + 2 + entries.iter().position(|e| e.0 == dng_tag::STRIPOFFSETS).unwrap() * 12 + 8;
    ifd[pos..pos + 4].copy_from_slice(&strip_off.to_le_bytes());
    let mut out = ifd;
    out.extend_from_slice(&spill);
    debug_assert_eq!(out.len() as u32, strip_off);
    for v in cfa {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// Write a CFA mosaic as a DNG file. Atomic (tmp-in-same-dir + rename)
/// so readers never see a half-written strip.
pub fn write_cfa_dng(
    dest: &Path,
    cfa: &[u16],
    w: u32,
    h: u32,
    meta: &DngMeta,
) -> Result<(), InferError> {
    let bytes = build_cfa_dng(cfa, w, h, meta)?;
    let tmp = dest.with_extension("dng.tmp");
    match std::fs::write(&tmp, &bytes) {
        Ok(_) => {}
        Err(e) => {
            return Err(InferError::Io(format!(
                "cannot write DNG {}: {e}",
                dest.display()
            )))
        }
    }
    match std::fs::rename(&tmp, dest) {
        Ok(_) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(InferError::Io(format!(
                "cannot publish DNG {}: {e}",
                dest.display()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    /// Serializes the tests that read or write the process-global
    /// `C41_MODELS_DIR` (cargo runs tests in parallel threads).
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    static TEST_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    const EPS: f32 = 1e-5;

    // ── Pattern / origin ────────────────────────────────────────────────────

    #[test]
    fn origin_rule_puts_channel_zero_on_red() {
        // _bayer_origin (restore_raw_bayer.c:45-57) finds the (y0,x0) with
        // FC == 0 (R); the packed channel 0 must land on it for every
        // pattern, which these origins satisfy by construction.
        for pat in [CfaPattern::Rggb, CfaPattern::Bggr, CfaPattern::Grbg, CfaPattern::Gbrg] {
            let (y0, x0) = pat.origin();
            assert_eq!(pat.fc(y0, x0), 0, "{pat:?} origin holds R");
            assert_eq!(pat.fc(y0, x0 + 1), 1, "{pat:?} right of R is G");
            assert_eq!(pat.fc(y0 + 1, x0), 1, "{pat:?} below R is G");
            assert_eq!(pat.fc(y0 + 1, x0 + 1), 2, "{pat:?} diagonal of R is B");
        }
        assert_eq!(CfaPattern::Rggb.origin(), (0, 0));
        assert_eq!(CfaPattern::Grbg.origin(), (0, 1));
        assert_eq!(CfaPattern::Gbrg.origin(), (1, 0));
        assert_eq!(CfaPattern::Bggr.origin(), (1, 1));
    }

    #[test]
    fn pattern_from_cfa2x2_maps_all_bayer_layouts() {
        assert_eq!(
            pattern_from_cfa2x2([[0, 1], [1, 2]]).unwrap(),
            CfaPattern::Rggb
        );
        assert_eq!(
            pattern_from_cfa2x2([[2, 1], [1, 0]]).unwrap(),
            CfaPattern::Bggr
        );
        assert_eq!(
            pattern_from_cfa2x2([[1, 0], [2, 1]]).unwrap(),
            CfaPattern::Grbg
        );
        assert_eq!(
            pattern_from_cfa2x2([[1, 2], [0, 1]]).unwrap(),
            CfaPattern::Gbrg
        );
        // An E site (masked/exotic sensor) is refused, not guessed.
        let r = pattern_from_cfa2x2([[0, 1], [1, 3]]).unwrap_err();
        assert!(matches!(r, InferError::InvalidArgument(_)), "{r}");
        assert!(r.to_string().contains("Bayer"), "{r}");
        // X-Trans style colours outside RGB are refused too.
        assert!(pattern_from_cfa2x2([[9, 9], [9, 9]]).is_err());
    }

    #[test]
    fn cfa_bytes_match_the_c_filter_masks() {
        // imageio_dng.c:630-645 writes RGGB as bytes [0,1,1,2] and BGGR as
        // [2,1,1,0]; GRBG/GBRG follow the same row-major FC rule.
        assert_eq!(CfaPattern::Rggb.cfa_bytes(), [0, 1, 1, 2]);
        assert_eq!(CfaPattern::Bggr.cfa_bytes(), [2, 1, 1, 0]);
        assert_eq!(CfaPattern::Grbg.cfa_bytes(), [1, 0, 2, 1]);
        assert_eq!(CfaPattern::Gbrg.cfa_bytes(), [1, 2, 0, 1]);
    }

    // ── Black / range / WB ──────────────────────────────────────────────────

    #[test]
    fn black_ranges_mirror_compute_cfa_black_range() {
        // restore_common.h:191-195: range = white - black, floored at 1.
        assert_eq!(black_ranges([10.0, 20.0, 30.0, 40.0], 1000.0), [990.0, 980.0, 970.0, 960.0]);
        // Degenerate pair never divides by ~0.
        assert_eq!(black_ranges([1000.0, 0.0, 0.0, 0.0], 1000.0), [1.0, 1000.0, 1000.0, 1000.0]);
        // Non-positive white falls back to 65535 (:178-180).
        assert_eq!(black_ranges([0.0, 0.0, 0.0, 0.0], 0.0), [65535.0; 4]);
    }

    #[test]
    fn daylight_wb_matches_the_c_d65_math() {
        // Identity-ish matrix: resp == D65 white [0.9504, 1.0, 1.0889], so
        // wb = [1/0.9504, 1, 1/1.0889] (restore_raw_bayer.c:96-98).
        let xyz = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]];
        let wb = daylight_wb(xyz).unwrap();
        assert!((wb[0] - 1.0 / 0.9504).abs() < EPS, "{wb:?}");
        assert_eq!(wb[1], 1.0);
        assert!((wb[2] - 1.0 / 1.0889).abs() < EPS, "{wb:?}");
        // All-zero matrix: mag <= 0, refused (:94).
        assert!(daylight_wb([[0.0; 3]; 4]).is_none());
        // A row combination with non-positive response is refused.
        assert!(daylight_wb([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0; 3]]).is_none());
    }

    #[test]
    fn as_shot_wb_normalizes_green_and_rejects_garbage() {
        // wb_coeffs [2, 1, 4] -> [2, 1, 4] with G == 1 already
        // (restore_raw_bayer.c:109-113).
        assert_eq!(as_shot_wb([2.0, 1.0, 4.0, 0.0]), Some([2.0, 1.0, 4.0]));
        assert_eq!(as_shot_wb([1.0, 2.0, 1.0, 0.0]), Some([0.5, 1.0, 0.5]));
        assert!(as_shot_wb([0.0, 1.0, 1.0, 0.0]).is_none());
        assert!(as_shot_wb([1.0, -1.0, 1.0, 0.0]).is_none());
    }

    #[test]
    fn resolve_wb_prefers_daylight_with_as_shot_fallback() {
        // C default DT_RESTORE_WB_DAYLIGHT with as-shot fallback (:142-146).
        let xyz = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0; 3]];
        let wb = resolve_wb(xyz, [2.0, 1.0, 4.0, 0.0]);
        assert!((wb[0] - 1.0 / 0.9504).abs() < EPS, "{wb:?}");
        // Dead matrix -> as-shot wins.
        assert_eq!(resolve_wb([[0.0; 3]; 4], [2.0, 1.0, 4.0, 0.0]), [2.0, 1.0, 4.0]);
        // Both dead -> unity (the C keeps its {1,1,1} init, :139).
        assert_eq!(resolve_wb([[0.0; 3]; 4], [0.0; 4]), [1.0, 1.0, 1.0]);
    }

    // ── Pack / gain / remosaic ──────────────────────────────────────────────

    #[test]
    fn normalize_site_applies_black_range_and_wb() {
        // (raw - black) / range * wb (restore_raw_bayer.c:318-321).
        assert!((normalize_site(110.0, 10.0, 100.0, 2.0) - 2.0).abs() < EPS);
        assert!((normalize_site(10.0, 10.0, 100.0, 2.0)).abs() < EPS);
    }

    #[test]
    fn mirror_in_range_stays_inside_the_cropped_rectangle() {
        // _mirror_in_range (restore_common.h:217-221): lo + mirror(i-lo).
        // Bounds [4, 12): -1 reflects to 9, 12 wraps to 10, 7 is identity.
        assert_eq!(mirror_in_range(-1, 4, 12), 9);
        assert_eq!(mirror_in_range(12, 4, 12), 10);
        assert_eq!(mirror_in_range(7, 4, 12), 7);
        assert_eq!(mirror_in_range(4, 4, 12), 4);
        for i in -20..30 {
            let m = mirror_in_range(i, 4, 12);
            assert!((4..12).contains(&m), "mirror({i}) = {m}");
        }
    }

    #[test]
    fn pack_tile_channels_hold_their_own_colours() {
        // 4x4 RGGB frame, T=2: each packed plane must carry its own
        // colour's normalized value (restore_raw_bayer.c:304-324).
        // raw = black + range so normalized == 1, times wb [2, 1, 0.5].
        let black = [10.0, 20.0, 30.0, 40.0];
        let range = [100.0, 200.0, 300.0, 400.0];
        let wb = [2.0, 1.0, 0.5];
        // Site order per row pair: (0,0)->black[0], (0,1)->black[1],
        // (1,0)->black[2], (1,1)->black[3].
        let site_black = [[black[0], black[1]], [black[2], black[3]]];
        let site_range = [[range[0], range[1]], [range[2], range[3]]];
        let mut cfa = vec![0u16; 16];
        for r in 0..4 {
            for c in 0..4 {
                cfa[r * 4 + c] =
                    (site_black[r % 2][c % 2] + site_range[r % 2][c % 2]) as u16;
            }
        }
        let tile = pack_bayer_tile(
            &cfa, 4, CfaPattern::Rggb, 0, 0, 0, 4, 0, 4, 2, black, range, wb,
        );
        assert_eq!(tile.len(), 16);
        // R plane all 2.0, G1/G2 planes all 1.0, B plane all 0.5.
        for v in &tile[0..4] {
            assert!((v - 2.0).abs() < 1e-4, "R plane: {v}");
        }
        for v in &tile[4..12] {
            assert!((v - 1.0).abs() < 1e-4, "G planes: {v}");
        }
        for v in &tile[12..16] {
            assert!((v - 0.5).abs() < 1e-4, "B plane: {v}");
        }
    }

    #[test]
    fn pack_tile_origin_shift_packs_bggr_as_rggb() {
        // force_rggb (restore_raw_bayer.c:239-244): a BGGR sensor packs
        // channel 0 from its R site at (1,1), i.e. origin shift (1,1).
        // Uniform frame, black 0, range 100, wb R=4: plane 0 must be 4.0
        // everywhere even though (0,0) is blue.
        let cfa = vec![50u16; 64];
        for pat in [CfaPattern::Rggb, CfaPattern::Bggr, CfaPattern::Grbg, CfaPattern::Gbrg] {
            let (y0, x0) = pat.origin();
            let tile = pack_bayer_tile(
                &cfa, 8, pat, y0, x0, y0, 8, x0, 8, 2,
                [0.0; 4], [100.0; 4], [4.0, 1.0, 1.0],
            );
            for v in &tile[0..4] {
                assert!((v - 2.0).abs() < 1e-4, "{pat:?} R plane: {v}");
            }
        }
    }

    #[test]
    fn remosaic_inverts_normalize_per_site() {
        // remosaic(pack(x)) == x (restore_raw_bayer.c:163-170 inverts
        // :318-321): sweep all four patterns, sites and parv values.
        let black = [12.0, 34.0, 56.0, 78.0];
        let range = black_ranges(black, 4000.0);
        let wb = [1.7, 1.0, 2.3];
        for pat in [CfaPattern::Rggb, CfaPattern::Bggr, CfaPattern::Grbg, CfaPattern::Gbrg] {
            for r in 0..4i64 {
                for c in 0..4i64 {
                    let ch = pat.fc(r, c);
                    for raw in [100.0f32, 500.0, 1500.0, 3900.0] {
                        let site = (((r & 1) << 1) | (c & 1)) as usize;
                        let norm = normalize_site(raw, black[site], range[site], wb[ch]);
                        let back = remosaic_value(norm, r, c, ch, wb, black, range);
                        assert!((back - raw).abs() < 1e-3, "{pat:?} ({r},{c}) {raw}: {back}");
                    }
                }
            }
        }
    }

    #[test]
    fn match_gain_equalizes_input_and_output_means() {
        // _bayer_gain_match (restore_raw_bayer.c:176-212): input mean 2,
        // output mean 4 -> gain 0.5 applied in place.
        let tile_in = vec![2.0f32; 4 * 9];
        let mut tile_out = vec![4.0f32; 3 * 36];
        let (im, om, gain) = match_gain_in_place(&tile_in, &mut tile_out, 3);
        assert!((im - 2.0).abs() < 1e-9, "{im}");
        assert!((om - 4.0).abs() < 1e-9, "{om}");
        assert!((gain - 0.5).abs() < EPS, "{gain}");
        for v in &tile_out {
            assert!((v - 2.0).abs() < 1e-6, "{v}");
        }
        // Near-zero output mean: gain 1.0, buffer untouched (:202-203).
        let mut tiny = vec![1e-12f32; 3 * 36];
        let before = tiny.clone();
        let (_, _, g2) = match_gain_in_place(&tile_in, &mut tiny, 3);
        assert_eq!(g2, 1.0);
        assert_eq!(tiny, before);
    }

    // ── Tiled driver (fake-session seam) ────────────────────────────────────

    /// A broadcast "model": packed 4ch in, 3ch 2T x 2T out where R/G/B
    /// come from the R/Gavg/B packed planes. With unity blacks-offsets
    /// this is near-identity (gain ~= 1), so the driver must round-trip.
    fn broadcast_tile(planar: &[f32], t: usize) -> Result<Vec<f32>, InferError> {
        let plane = t * t;
        let two_t = 2 * t;
        let mut out = vec![0f32; 3 * two_t * two_t];
        for y in 0..two_t {
            for x in 0..two_t {
                let py = y / 2;
                let px = x / 2;
                out[y * two_t + x] = planar[py * t + px];
                out[two_t * two_t + y * two_t + x] =
                    (planar[plane + py * t + px] + planar[2 * plane + py * t + px]) / 2.0;
                out[2 * two_t * two_t + y * two_t + x] = planar[3 * plane + py * t + px];
            }
        }
        Ok(out)
    }

    #[test]
    fn constant_frame_roundtrips_through_tiles() {
        // Constant mosaic: every tile_in plane is constant, match_gain is
        // exactly 1, and remosaic inverts the pack — output == input.
        for pat in [CfaPattern::Rggb, CfaPattern::Bggr, CfaPattern::Grbg, CfaPattern::Gbrg] {
            let (w, h) = (130u32, 74u32);
            let raw = vec![1000u16; w as usize * h as usize];
            let mut seen = Vec::new();
            let out = run_bayer_tiled(
                &raw, w, h, pat, [0.0; 4], 65535.0, [1.0, 1.0, 1.0], 96,
                broadcast_tile,
                |d, t| seen.push((d, t)),
            )
            .unwrap();
            assert_eq!(out, raw, "{pat:?} constant frame must round-trip");
            assert!(!seen.is_empty());
            for wnd in seen.windows(2) {
                assert!(wnd[1].0 > wnd[0].0, "monotonic: {seen:?}");
            }
            let total = seen.last().unwrap().1;
            assert_eq!(seen.last().unwrap().0, total);
            // Packed 65x37 (RGGB), T=96 O=32 step 32: 3 cols x 2 rows.
            if pat == CfaPattern::Rggb {
                assert_eq!(total, 6, "{seen:?}");
            }
        }
    }

    #[test]
    fn gradient_frame_roundtrips_within_u16_rounding() {
        // Non-constant input exercises mirror padding + gain. Unity WB
        // keeps the 4ch-input vs 3ch-output means equal up to float dust
        // (gain ~= 1), so every pixel lands within u16 rounding of the
        // source. (With non-unity WB the means structurally differ and
        // match_gain rescales — correct per the C, pinned by the
        // match_gain unit test instead.)
        let (w, h) = (100u32, 60u32);
        let mut raw = vec![0u16; w as usize * h as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                raw[y * w as usize + x] = (500 + (x * 37 + y * 91) % 30000) as u16;
            }
        }
        let out = run_bayer_tiled(
            &raw, w, h, CfaPattern::Rggb, [64.0, 64.0, 64.0, 64.0], 16000.0,
            [1.0, 1.0, 1.0], 96, broadcast_tile, |_, _| {},
        )
        .unwrap();
        assert_eq!(out.len(), raw.len());
        for (i, (o, r)) in out.iter().zip(raw.iter()).enumerate() {
            let d = (*o as i32 - *r as i32).abs();
            assert!(d <= 2, "pixel {i}: {o} vs {r}");
        }
    }

    #[test]
    fn odd_dims_keep_margins_from_source() {
        // 7x5 RGGB: working region is 6x4 (margins = last row/col), copied
        // from the source like the C margin init (:358-363).
        let (w, h) = (7u32, 5u32);
        let raw = vec![2000u16; w as usize * h as usize];
        let out = run_bayer_tiled(
            &raw, w, h, CfaPattern::Rggb, [0.0; 4], 65535.0, [1.0, 1.0, 1.0], 96,
            broadcast_tile, |_, _| {},
        )
        .unwrap();
        assert_eq!(out, raw);
    }

    #[test]
    fn driver_rejects_bad_inputs_without_panicking() {
        let raw = vec![1u16; 100];
        // Short mosaic.
        assert!(run_bayer_tiled(&raw, 11, 10, CfaPattern::Rggb, [0.0; 4], 100.0, [1.0; 3], 64, broadcast_tile, |_, _| {}).is_err());
        // T <= 2O.
        assert!(run_bayer_tiled(&raw, 10, 10, CfaPattern::Rggb, [0.0; 4], 100.0, [1.0; 3], 64, broadcast_tile, |_, _| {}).is_err());
        // Wrong tile-runner output length.
        let r = run_bayer_tiled(
            &raw, 10, 10, CfaPattern::Rggb, [0.0; 4], 100.0, [1.0; 3], 96,
            |_, _| Ok(vec![0f32; 4]),
            |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(r, InferError::InvalidArgument(_)), "{r}");
        // rawdenoise_bayer validates the mosaic before touching the session.
        let r = rawdenoise_bayer(
            &raw, 11, 10, CfaPattern::Rggb, [0.0; 4], 100.0, [1.0; 3],
            Path::new("/nonexistent/model_bayer.onnx"), 96, |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(r, InferError::InvalidArgument(_)), "{r}");
        // Missing payload is Io, not a panic.
        let raw100 = [1u16; 100];
        let r = rawdenoise_bayer(
            &raw100, 10, 10, CfaPattern::Rggb, [0.0; 4], 100.0, [1.0; 3],
            Path::new("/nonexistent/model_bayer.onnx"), 96, |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(r, InferError::Io(_)), "{r}");
    }

    #[test]
    fn loader_refuses_garbage_honestly() {
        // Missing file: an InvalidArgument naming the path, not a panic.
        let r = load_bayer_source(Path::new("/nonexistent/shot.cr2")).unwrap_err();
        assert!(matches!(r, InferError::InvalidArgument(_)), "{r}");
        // A non-raw file decodes to nothing useful: refused as well.
        let dir = std::env::temp_dir().join(format!(
            "c41_raw_{}_{}",
            std::process::id(),
            TEST_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("note.txt");
        std::fs::write(&txt, b"not a raw").unwrap();
        let r = load_bayer_source(&txt).unwrap_err();
        assert!(matches!(r, InferError::InvalidArgument(_)), "{r}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── DNG writer ──────────────────────────────────────────────────────────

    /// Minimal IFD walk: (tag, type, count, raw value bytes) with offsets
    /// resolved. Enough to assert tag presence, values and strip layout.
    fn parse_ifd(bytes: &[u8]) -> Vec<(u16, u16, u32, Vec<u8>)> {
        assert_eq!(&bytes[0..2], b"II");
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 42);
        let ifd_off = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let n = u16::from_le_bytes([bytes[ifd_off], bytes[ifd_off + 1]]) as usize;
        let mut out = Vec::new();
        for i in 0..n {
            let e = ifd_off + 2 + i * 12;
            let tag = u16::from_le_bytes([bytes[e], bytes[e + 1]]);
            let typ = u16::from_le_bytes([bytes[e + 2], bytes[e + 3]]);
            let count = u32::from_le_bytes([bytes[e + 4], bytes[e + 5], bytes[e + 6], bytes[e + 7]]);
            let unit = match typ {
                1 | 2 | 7 => 1,
                3 => 2,
                4 => 4,
                5 | 10 => 8,
                _ => panic!("unexpected field type {typ}"),
            };
            let total = count as usize * unit;
            let val = &bytes[e + 8..e + 12];
            let data = if total <= 4 {
                val[..total].to_vec()
            } else {
                let off = u32::from_le_bytes([val[0], val[1], val[2], val[3]]) as usize;
                bytes[off..off + total].to_vec()
            };
            out.push((tag, typ, count, data));
        }
        out
    }

    fn tag(entries: &[(u16, u16, u32, Vec<u8>)], tag: u16) -> &(u16, u16, u32, Vec<u8>) {
        entries.iter().find(|e| e.0 == tag).unwrap_or_else(|| panic!("missing tag {tag}"))
    }

    fn test_meta<'a>() -> DngMeta<'a> {
        DngMeta {
            make: "Make",
            model: "Model",
            filename: "shot.cr2",
            color_matrix: SRGB_FALLBACK_MATRIX,
            as_shot_neutral: Some([0.5, 1.0, 0.75]),
            black: [64.0, 64.0, 64.0, 64.0],
            white: 16000,
            pattern: CfaPattern::Rggb,
        }
    }

    #[test]
    fn dng_carries_the_c_no_preview_tag_set() {
        // Every tag the no-preview path writes (imageio_dng.c:273-357 +
        // shared metadata :111-179) must be present with the C's values.
        let meta = test_meta();
        let (w, h) = (24u32, 16u32);
        let cfa: Vec<u16> = (0..w * h).map(|i| (i % 16000) as u16).collect();
        let bytes = build_cfa_dng(&cfa, w, h, &meta).unwrap();
        let entries = parse_ifd(&bytes);
        // Ascending tag order (strict readers require it).
        let tags: Vec<u16> = entries.iter().map(|e| e.0).collect();
        let mut sorted = tags.clone();
        sorted.sort();
        assert_eq!(tags, sorted);
        let u32v = |t: u16| {
            let e = tag(&entries, t);
            u32::from_le_bytes([e.3[0], e.3[1], e.3[2], e.3[3]])
        };
        let u16v = |t: u16| {
            let e = tag(&entries, t);
            u16::from_le_bytes([e.3[0], e.3[1]])
        };
        assert_eq!(u32v(dng_tag::SUBFILETYPE), 0);
        assert_eq!(u32v(dng_tag::IMAGEWIDTH), w);
        assert_eq!(u32v(dng_tag::IMAGELENGTH), h);
        assert_eq!(u16v(dng_tag::BITSPERSAMPLE), 16);
        assert_eq!(u16v(dng_tag::SAMPLESPERPIXEL), 1);
        assert_eq!(u16v(dng_tag::PLANARCONFIG), 1);
        assert_eq!(u16v(dng_tag::PHOTOMETRIC), 32803);
        assert_eq!(u16v(dng_tag::SAMPLEFORMAT), 1);
        assert_eq!(u16v(dng_tag::COMPRESSION), 1);
        assert_eq!(u16v(dng_tag::ORIENTATION), 1);
        assert_eq!(u32v(dng_tag::ROWSPERSTRIP), h);
        assert_eq!(u32v(dng_tag::STRIPBYTECOUNTS), w * h * 2);
        assert_eq!(u16v(dng_tag::CFALAYOUT), 1);
        assert_eq!(u16v(dng_tag::CALIBRATIONILLUMINANT1), 21);
        assert_eq!(u16v(dng_tag::RESOLUTIONUNIT), 2);
        // CFARepeatPatternDim + BlackLevelRepeatDim are {2,2}.
        for t in [dng_tag::CFAREPEATPATTERNDIM, dng_tag::BLACKLEVELREPEATDIM] {
            let e = tag(&entries, t);
            assert_eq!((e.1, e.2), (3, 2), "tag {t}");
            assert_eq!(e.3, vec![2, 0, 2, 0], "tag {t}");
        }
        // CFAPattern native RGGB order; CFAPlaneColor R,G,B.
        assert_eq!(tag(&entries, dng_tag::CFAPATTERN).3, vec![0, 1, 1, 2]);
        assert_eq!(tag(&entries, dng_tag::CFAPLANECOLOR).3, vec![0, 1, 2]);
        // BlackLevel x4 (SHORT), WhiteLevel x1 (LONG).
        let bl = tag(&entries, dng_tag::BLACKLEVEL);
        assert_eq!((bl.1, bl.2), (3, 4));
        assert_eq!(bl.3, vec![64, 0, 64, 0, 64, 0, 64, 0]);
        assert_eq!(u32v(dng_tag::WHITELEVEL), 16000);
        // ActiveArea covers the full buffer; default crop geometry matches.
        let aa = tag(&entries, dng_tag::ACTIVEAREA);
        assert_eq!(aa.2, 4);
        let words: Vec<u32> = aa.3.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(words, vec![0, 0, h, w]);
        // DNGVersion {1,4,0,0}, backward {1,2,0,0} (imageio_dng.c:136-137).
        assert_eq!(tag(&entries, dng_tag::DNGVERSION).3, vec![1, 4, 0, 0]);
        assert_eq!(tag(&entries, dng_tag::DNGBACKWARDVERSION).3, vec![1, 2, 0, 0]);
        // BaselineExposure 0/1.
        assert_eq!(tag(&entries, dng_tag::BASELINEEXPOSURE).3, vec![0, 0, 0, 0, 1, 0, 0, 0]);
        // AsShotNeutral present (Some) at 1e6 denominator.
        let asn = tag(&entries, dng_tag::ASSHOTNEUTRAL);
        assert_eq!((asn.1, asn.2), (5, 3));
        let nums: Vec<u32> = asn.3.chunks(8).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(nums, vec![500000, 1000000, 750000]);
        // ColorMatrix1: 9 SRATIONAL at 1e4 denominator (XYZ->sRGB fallback).
        let cm = tag(&entries, dng_tag::COLORMATRIX1);
        assert_eq!((cm.1, cm.2), (10, 9));
        let first = i32::from_le_bytes([cm.3[0], cm.3[1], cm.3[2], cm.3[3]]);
        assert_eq!(first, (3.2404542f32 * 10000.0).round() as i32);
        // Software + Make/Model + OriginalRawFileName.
        let sw = tag(&entries, dng_tag::SOFTWARE);
        assert!(sw.3.starts_with(b"c41 "), "{sw:?}");
        assert_eq!(tag(&entries, dng_tag::MAKE).3, b"Make\0".to_vec());
        assert_eq!(tag(&entries, dng_tag::MODEL).3, b"Model\0".to_vec());
        assert_eq!(tag(&entries, dng_tag::ORIGINALRAWFILENAME).3, b"shot.cr2".to_vec());
        // Strip integrity: bytes at StripOffsets decode to the input LE.
        let off = u32v(dng_tag::STRIPOFFSETS) as usize;
        assert_eq!(off + (w * h * 2) as usize, bytes.len());
        let mut back = Vec::with_capacity(cfa.len());
        for c in bytes[off..].chunks(2) {
            back.push(u16::from_le_bytes([c[0], c[1]]));
        }
        assert_eq!(back, cfa);
    }

    #[test]
    fn dng_patterns_and_missing_neutral() {
        // CFAPattern follows the sensor for every Bayer layout; a missing
        // neutral omits AsShotNeutral entirely (imageio_dng.c:143-155).
        for (pat, expect) in [
            (CfaPattern::Rggb, vec![0, 1, 1, 2]),
            (CfaPattern::Bggr, vec![2, 1, 1, 0]),
            (CfaPattern::Grbg, vec![1, 0, 2, 1]),
            (CfaPattern::Gbrg, vec![1, 2, 0, 1]),
        ] {
            let meta = DngMeta { pattern: pat, as_shot_neutral: None, ..test_meta() };
            let cfa = vec![7u16; 64];
            let bytes = build_cfa_dng(&cfa, 8, 8, &meta).unwrap();
            let entries = parse_ifd(&bytes);
            assert_eq!(tag(&entries, dng_tag::CFAPATTERN).3, expect, "{pat:?}");
            assert!(entries.iter().all(|e| e.0 != dng_tag::ASSHOTNEUTRAL), "{pat:?}");
            // Empty filename omits OriginalRawFileName.
            let meta2 = DngMeta { filename: "", ..test_meta() };
            let bytes2 = build_cfa_dng(&cfa, 8, 8, &meta2).unwrap();
            let entries2 = parse_ifd(&bytes2);
            assert!(entries2.iter().all(|e| e.0 != dng_tag::ORIGINALRAWFILENAME));
        }
    }

    #[test]
    fn dng_rejects_degenerate_inputs() {
        let meta = test_meta();
        assert!(build_cfa_dng(&[1u16; 4], 0, 2, &meta).is_err());
        assert!(build_cfa_dng(&[1u16; 3], 2, 2, &meta).is_err());
    }

    #[test]
    fn dng_roundtrips_through_rawloader() {
        // Our DNG must decode in the same rawloader our raw path uses:
        // dims, cpp, pattern, levels, WB and matrix all survive.
        let meta = test_meta();
        let (w, h) = (32u32, 24u32);
        let cfa: Vec<u16> = (0..w * h).map(|i| 64 + (i % 15000) as u16).collect();
        let bytes = build_cfa_dng(&cfa, w, h, &meta).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "c41_raw_dng_{}_{}",
            std::process::id(),
            TEST_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.dng");
        std::fs::write(&path, &bytes).unwrap();
        let raw = rawloader::decode_file(&path).expect("rawloader reads our DNG");
        assert_eq!((raw.width, raw.height, raw.cpp), (w as usize, h as usize, 1));
        let data = match raw.data {
            rawloader::RawImageData::Integer(ref v) => v.clone(),
            _ => panic!("expected integer strip"),
        };
        assert_eq!(data, cfa);
        assert_eq!(
            crate::rawimage::classify_cfa(|r, c| raw.cfa.color_at(r, c)).unwrap(),
            crate::rawimage::CfaKind::Bayer([[0, 1], [1, 2]])
        );
        assert_eq!(raw.blacklevels, [64, 64, 64, 64]);
        assert_eq!(raw.whitelevels, [16000, 16000, 16000, 16000]);
        // WB round-trips through AsShotNeutral (1/neutral, NaN 4th).
        assert!((raw.wb_coeffs[0] - 2.0).abs() < 1e-3, "{:?}", raw.wb_coeffs);
        assert!((raw.wb_coeffs[1] - 1.0).abs() < 1e-3, "{:?}", raw.wb_coeffs);
        assert!((raw.wb_coeffs[2] - 1.0 / 0.75).abs() < 1e-3, "{:?}", raw.wb_coeffs);
        // ColorMatrix1 survives at 1e4 quantization (XYZ->sRGB fallback).
        assert!((raw.xyz_to_cam[0][0] - 3.2404542).abs() < 1e-4, "{:?}", raw.xyz_to_cam);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Model handling (fixtures, no network) ────────────────────────────────

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn fresh(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "c41_ai_raw_{}_{}_{}",
                std::process::id(),
                name,
                TEST_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// A minimal raw-denoise `.dtmodel`: config.json declaring the
    /// `model_bayer` stem plus a fake `model_bayer.onnx` payload.
    fn synthetic_bayer_dtmodel(manifest: &str) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            w.start_file(package::MANIFEST_FILENAME, opts).unwrap();
            w.write_all(manifest.as_bytes()).unwrap();
            w.start_file(MODEL_BAYER_ONNX_FILE, opts).unwrap();
            w.write_all(b"fake-bayer-onnx").unwrap();
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    const BAYER_MANIFEST: &str = r#"{
        "name": "test rawdenoise",
        "attributes": {
            "model_bayer": {
                "input_sizes": [256],
                "input_kind": "bayer_v1"
            }
        }
    }"#;

    #[test]
    fn prepare_resolves_bayer_stem_tile_size() {
        let _env = ENV_LOCK.lock().unwrap();
        let dir = TempDir::fresh("prepare");
        std::env::set_var("C41_MODELS_DIR", &dir.path);
        std::fs::write(
            dir.path.join(RAWDENOISE_NIND_ASSET),
            synthetic_bayer_dtmodel(BAYER_MANIFEST),
        )
        .unwrap();
        let prep = prepare_rawdenoise_model(RAWDENOISE_NIND_ASSET).unwrap();
        assert_eq!(prep.tile_size, 256);
        assert_eq!(prep.onnx_path.file_name().unwrap(), MODEL_BAYER_ONNX_FILE);
        // Second call reuses the unpacked payload (no re-unpack, same path).
        let prep2 = prepare_rawdenoise_model(RAWDENOISE_NIND_ASSET).unwrap();
        assert_eq!(prep2.onnx_path, prep.onnx_path);
        std::env::remove_var("C41_MODELS_DIR");
    }

    #[test]
    fn prepare_refuses_unknown_input_kind_and_missing_sizes() {
        let _env = ENV_LOCK.lock().unwrap();
        let dir = TempDir::fresh("prepare_bad");
        std::env::set_var("C41_MODELS_DIR", &dir.path);
        // Declared-but-unknown input_kind is a hard error (restore.h:136).
        std::fs::write(
            dir.path.join(RAWDENOISE_NIND_ASSET),
            synthetic_bayer_dtmodel(
                r#"{"attributes": {"model_bayer": {"input_sizes": [256], "input_kind": "fancy_v9"}}}"#,
            ),
        )
        .unwrap();
        let r = prepare_rawdenoise_model(RAWDENOISE_NIND_ASSET).unwrap_err();
        assert!(matches!(r, InferError::Package(_)), "{r}");
        assert!(r.to_string().contains("fancy_v9"), "{r}");
        // No input_sizes anywhere is refused like the C (restore.c:301-320).
        std::fs::write(
            dir.path.join(RAWDENOISE_NIND_ASSET),
            synthetic_bayer_dtmodel(r#"{"attributes": {}}"#),
        )
        .unwrap();
        let r = prepare_rawdenoise_model(RAWDENOISE_NIND_ASSET).unwrap_err();
        assert!(matches!(r, InferError::Package(_)), "{r}");
        // Missing archive is MissingModel, not a panic.
        let r = prepare_rawdenoise_model("rawdenoise-nope.dtmodel").unwrap_err();
        assert!(matches!(r, InferError::MissingModel(_)), "{r}");
        std::env::remove_var("C41_MODELS_DIR");
    }

    #[test]
    fn store_scan_lists_only_rawdenoise_archives() {
        let _env = ENV_LOCK.lock().unwrap();
        let dir = TempDir::fresh("scan");
        std::env::set_var("C41_MODELS_DIR", &dir.path);
        for n in ["rawdenoise-nind.dtmodel", "denoise-nind.dtmodel", "upscale-bsrgan.dtmodel"] {
            std::fs::write(dir.path.join(n), b"x").unwrap();
        }
        assert_eq!(
            downloaded_rawdenoise_models(),
            vec!["rawdenoise-nind.dtmodel".to_string()]
        );
        std::env::remove_var("C41_MODELS_DIR");
    }
}
