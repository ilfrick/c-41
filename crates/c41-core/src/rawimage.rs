//! Raw image decode + sensor normalisation (Phase 3 milestone 2: the real
//! scene-referred front end for `pipeline`).
//!
//! Uses the pure-Rust [`rawloader`] crate to decode a camera raw file into a
//! Bayer **or Fuji X-Trans** CFA mosaic, then normalises each photosite to
//! linear [0,1] by subtracting the per-colour black level and scaling by the
//! per-colour range (`white - black`). The result is a single-channel float
//! mosaic ready to feed the migrated demosaic (Bayer → PPG; X-Trans →
//! Markesteijn) and then [`pipeline`].
//!
//! rawloader's `blacklevels`/`whitelevels`/`wb_coeffs` are in **RGBE order**
//! indexed by `CFA::color_at` (0=R, 1=G, 2=B, 3=E), which is exactly the index
//! [`normalize_cfa`] uses.
//!
//! Known v1 gaps (to address as the front end matures — none block Bayer):
//! - **Black level from `blacklevels` only.** Cameras that report `0` and expect
//!   the black point from the masked optical-black border (`blackareas`) get no
//!   black subtraction yet (slightly lifted shadows).
//! - **White balance is stored, not applied** — `wb` belongs after demosaic
//!   (or into the mosaic when highlight reconstruction runs first, m4-119:
//!   [`apply_white_balance_mosaic`] / [`iop::highlights::reconstruct_mosaic`]).
//!
//! Since m4-119 the normalisation carries over-range photosites (`> 1.0`)
//! instead of clamping at white, so highlight reconstruction has data to
//! rebuild from; consumers clamp ([`RawImage::to_linear_rgba_with`]'s legacy
//! path reproduces the old hard-clip byte-for-byte).
//!
//! [`pipeline`]: crate::pipeline

use crate::{Error, Result};

/// A decoded, black/white-normalised CFA mosaic plus the metadata the demosaic
/// and white-balance steps need. `mosaic` is `width * height` linear [0,1]
/// photosites (single channel); `cfa` is the 2×2 colour pattern (indices into
/// the RGBE-ordered `wb`). For a Fuji X-Trans sensor `xtrans` carries the full
/// 6×6 pattern and the demosaic routes to Markesteijn instead of PPG; `cfa` is
/// then just the (unused) 2×2 snapshot at the origin.
#[derive(Clone, Debug)]
pub struct RawImage {
    pub width: usize,
    pub height: usize,
    /// 2×2 CFA colour indices: `cfa[row % 2][col % 2]` (0=R, 1=G, 2=B, 3=E).
    pub cfa: [[usize; 2]; 2],
    /// `Some(6×6 pattern)` for a Fuji X-Trans sensor (colours 0=R,1=G,2=B),
    /// `None` for a 2×2 Bayer sensor. Selects the demosaic in `to_linear_rgba`.
    pub xtrans: Option<[[u8; 6]; 6]>,
    /// White-balance multipliers in RGBE order (as encoded in the file).
    pub wb: [f32; 4],
    /// Display orientation as `(transpose, flip_x, flip_y)` from rawloader's
    /// `Orientation::to_flips()` — applied (after demosaic) by `to_linear_rgba`.
    pub orientation: (bool, bool, bool),
    /// Camera-native RGB → linear-Rec.2020 (D65) 3×3, derived from the raw's
    /// XYZ→camera matrix by [`rec2020_from_cam_matrix`] and applied (after white
    /// balance) by `to_linear_rgba`. [`IDENTITY3`] when the file carries no usable
    /// matrix. Rec.2020 is the working space (m4-35); the display seam converts it
    /// to sRGB via [`REC2020_TO_SRGB`].
    pub cam_to_working: [[f32; 3]; 3],
    /// Black/white-normalised photosites, row-major, `width * height` long.
    pub mosaic: Vec<f32>,
}

/// Normalise a raw CFA plane to linear [0,1]: per photosite, subtract the
/// per-colour black level and divide by the per-colour range (`white - black`).
/// Values below 0 clamp there; **over-range photosites are carried (`> 1.0`)**
/// since m4-119 so highlight reconstruction has data to rebuild from — the
/// consumer decides where to clip ([`RawImage::to_linear_rgba_with`] clamps at
/// white when reconstruction is off). `cfa` gives the colour index at
/// `[row % 2][col % 2]`; `black`/`white` are indexed by that colour (RGBE
/// order). Pure — no decode dependency.
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
            out[row * width + col] = v.max(0.0);
        }
    }
    out
}

/// Normalise a Fuji **X-Trans** CFA plane to linear [0,1], the 6×6 analogue of
/// [`normalize_cfa`]: per photosite the colour is `xtrans[row % 6][col % 6]`
/// (0=R, 1=G, 2=B) and `black`/`white` are indexed by that colour (RGBE order).
/// Pure — no decode dependency.
pub fn normalize_xtrans(
    data: &[u16],
    width: usize,
    height: usize,
    xtrans: &[[u8; 6]; 6],
    black: [f32; 4],
    white: [f32; 4],
) -> Vec<f32> {
    let mut out = vec![0.0f32; width.saturating_mul(height)];
    if data.len() < out.len() {
        return out; // malformed: short plane ⇒ all-black rather than a panic
    }
    for row in 0..height {
        for col in 0..width {
            let color = xtrans[row % 6][col % 6] as usize;
            let b = black[color];
            let range = (white[color] - b).max(1.0);
            let v = (data[row * width + col] as f32 - b) / range;
            out[row * width + col] = v.max(0.0);
        }
    }
    out
}

/// The CFA layouts the front end recognises: a 2×2 Bayer pattern or a 6×6 Fuji
/// X-Trans pattern. Anything else is rejected at [`load`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CfaKind {
    /// 2×2 Bayer (`cfa[row % 2][col % 2]`, colours 0=R,1=G,2=B,3=E).
    Bayer([[usize; 2]; 2]),
    /// 6×6 Fuji X-Trans (`pat[row % 6][col % 6]`, colours 0=R,1=G,2=B).
    Xtrans([[u8; 6]; 6]),
}

/// Bayer demosaic algorithm choice for [`RawImage::to_linear_rgba_with`]. Only
/// affects Bayer sensors — X-Trans always uses Markesteijn regardless. [`Rcd`]
/// is the default (darktable's default; highest quality). Ordered/keep the
/// discriminants stable: they are persisted (see the UI preview params).
///
/// [`Rcd`]: DemosaicMethod::Rcd
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DemosaicMethod {
    /// Ratio Corrected Demosaicing ([`demosaic_rcd`]) — darktable's default.
    #[default]
    Rcd,
    /// Variable Number of Gradients ([`demosaic_vng`]).
    Vng,
    /// Patterned Pixel Grouping ([`demosaic_ppg`]) — fastest, lowest quality.
    Ppg,
}

impl DemosaicMethod {
    /// Stable persistence code (must not change — stored per image). See
    /// [`from_u8`](Self::from_u8).
    pub fn as_u8(self) -> u8 {
        match self {
            DemosaicMethod::Rcd => 0,
            DemosaicMethod::Vng => 1,
            DemosaicMethod::Ppg => 2,
        }
    }

    /// Decode a persisted [`as_u8`](Self::as_u8) code. Any unknown byte (older
    /// or corrupt data, or a future method this build doesn't know) falls back
    /// to the [default](Self::default) so persistence can never fail a load.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => DemosaicMethod::Vng,
            2 => DemosaicMethod::Ppg,
            _ => DemosaicMethod::Rcd,
        }
    }
}

/// Classify a CFA from a colour-lookup probe `color_at(row, col)` (0=R,1=G,2=B,
/// 3=E, as `rawloader::CFA::color_at` returns). A pattern that is 2×2-periodic
/// across the 6×6 window (6 = lcm(2,6)) is [`CfaKind::Bayer`]; otherwise one that
/// is 6×6-periodic across a 12×12 window is [`CfaKind::Xtrans`]. Validates the
/// colour indices fit the level arrays (RGBE for Bayer, RGB for X-Trans).
///
/// Pure (closure-driven) so the classification — which can't be exercised on a
/// real file in CI without a sample of every sensor — is fully unit-testable.
/// Fail-closed: an unrecognised period or out-of-range colour returns an error
/// rather than guessing.
pub fn classify_cfa(color_at: impl Fn(usize, usize) -> usize) -> Result<CfaKind> {
    let cfa = [
        [color_at(0, 0), color_at(0, 1)],
        [color_at(1, 0), color_at(1, 1)],
    ];
    let is_2x2 = (0..6).all(|r| (0..6).all(|c| color_at(r, c) == cfa[r % 2][c % 2]));
    if is_2x2 {
        // Trust-boundary guard: color_at must index the length-4 RGBE arrays.
        if cfa.iter().flatten().any(|&c| c >= 4) {
            return Err(Error::Raw("CFA colour index outside RGBE range".into()));
        }
        return Ok(CfaKind::Bayer(cfa));
    }
    // Snapshot the 6×6 pattern and confirm it really is 6×6-periodic (checked
    // over a 12×12 window) rather than some other CFA we don't model.
    let mut xt = [[0u8; 6]; 6];
    for (r, row) in xt.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = color_at(r, c) as u8;
        }
    }
    let is_6x6 = (0..12).all(|r| (0..12).all(|c| color_at(r, c) == xt[r % 6][c % 6] as usize));
    if !is_6x6 {
        return Err(Error::Raw("unsupported CFA period (not 2×2 or 6×6)".into()));
    }
    // X-Trans is RGB only (no 'E'); guard the level-array indexing.
    if xt.iter().flatten().any(|&c| c >= 3) {
        return Err(Error::Raw("X-Trans colour index outside RGB range".into()));
    }
    Ok(CfaKind::Xtrans(xt))
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
    let black: [f32; 4] = std::array::from_fn(|i| raw.blacklevels[i] as f32);
    let white: [f32; 4] = std::array::from_fn(|i| raw.whitelevels[i] as f32);

    // Classify the CFA period (Bayer 2×2 vs X-Trans 6×6) and normalise with the
    // matching layout. `cfa` (the origin 2×2 snapshot) is kept on the struct for
    // the Bayer path; for X-Trans it's an unused placeholder.
    let (xtrans, mosaic) = match classify_cfa(|r, c| raw.cfa.color_at(r, c))? {
        CfaKind::Bayer(cfa2) => {
            (None, normalize_cfa(data, raw.width, raw.height, cfa2, black, white))
        }
        CfaKind::Xtrans(xt) => (
            Some(xt),
            normalize_xtrans(data, raw.width, raw.height, &xt, black, white),
        ),
    };

    Ok(RawImage {
        width: raw.width,
        height: raw.height,
        cfa,
        xtrans,
        wb: raw.wb_coeffs,
        orientation: raw.orientation.to_flips(),
        cam_to_working: rec2020_from_cam_matrix(raw.xyz_to_cam),
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
/// can replace this later. Bayer only (`cfa` must be 2×2; X-Trans routes to
/// [`demosaic_xtrans`]).
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

/// Per-tile scratch for [`demosaic_rcd`], allocated once per rayon worker (via
/// `for_each_init`) and cleared per tile. Sizes are the RCD tile
/// (`DT_RCD_TILESIZE = 112`); the half-resolution buffers are TS²/2.
struct RcdScratch {
    cfa_b: Vec<f32>,
    rgb: Vec<f32>,
    vh_dir: Vec<f32>,
    pq_dir: Vec<f32>,
    lpf: Vec<f32>,
    p_cdiff: Vec<f32>,
    q_cdiff: Vec<f32>,
    vsq: Vec<f32>,
    hsq: Vec<f32>,
}

impl RcdScratch {
    fn new() -> Self {
        const TS: usize = 112;
        Self {
            cfa_b: vec![0.0; TS * TS],
            rgb: vec![0.0; 3 * TS * TS],
            vh_dir: vec![0.0; TS * TS],
            pq_dir: vec![0.0; TS * TS / 2],
            lpf: vec![0.0; TS * TS / 2],
            p_cdiff: vec![0.0; TS * TS / 2],
            q_cdiff: vec![0.0; TS * TS / 2],
            vsq: vec![0.0; TS * TS],
            hsq: vec![0.0; TS * TS],
        }
    }
}

/// A raw `*mut f32` promoted to `Send`+`Sync` so rayon tile tasks can write their
/// output regions in parallel. Sound ONLY because the RCD tiles' valid write
/// regions (`[first_v,last_v) × [first_h,last_h)` in image space) are pairwise
/// disjoint — no two tiles ever write the same output element.
#[derive(Clone, Copy)]
struct SyncMutPtr(*mut f32);
unsafe impl Send for SyncMutPtr {}
unsafe impl Sync for SyncMutPtr {}

/// Demosaic a normalised Bayer CFA mosaic to packed RGBA `f32` via the migrated
/// **RCD** (Ratio Corrected Demosaicing) algorithm — darktable's default, with
/// noticeably fewer maze/zipper artefacts on fine detail than [`demosaic_ppg`].
/// Faithful port of `rcd_demosaic` (`src/iop/demosaicing/rcd.c:98`).
///
/// A full-frame [`demosaic_ppg`] provides the base image; RCD then overwrites
/// each tile interior. Like the C source this tiles the frame into
/// `DT_RCD_TILESIZE`-square (112 px) tiles with a `RCD_TILEVALID` (92 px) valid
/// stride, so the per-tile scratch buffers stay small (~350 KB total) even at
/// full sensor resolution — the demosaic runs before the preview downscale, so
/// a whole-image scratch would be hundreds of MB. Tiles are processed serially
/// (the C `DT_OMP` parallelism is a future rayon follow-up). The tile-local
/// index space matches C exactly: `RCD_TILEVALID` is even, so tile-local (row,
/// col) parity equals absolute parity and `fc_bayer` is correct on either.
///
/// Falls back to the [`demosaic_ppg`] base for frames narrower/shorter than one
/// border pair (`2·RCD_BORDER`). Leaves the 4th channel at 0 like the other
/// demosaicers — [`RawImage::to_linear_rgba`] sets alpha opaque afterwards.
pub fn demosaic_rcd(mosaic: &[f32], width: usize, height: usize, cfa: [[usize; 2]; 2]) -> Vec<f32> {
    // Base image: full-frame PPG. RCD refines the interior on top (rcd.c:105).
    let mut out = demosaic_ppg(mosaic, width, height, cfa);
    const BORDER: usize = 10; // RCD_BORDER — inter-tile overlap discarded per tile
    const MARGIN: usize = 9; // RCD_MARGIN — smaller border for the outermost tiles
    const TS: usize = 112; // DT_RCD_TILESIZE
    const TILEVALID: usize = TS - 2 * BORDER; // 92 (even ⇒ parity-preserving)
    const W1: usize = TS;
    const W2: usize = 2 * TS;
    const W3: usize = 3 * TS;
    const W4: usize = 4 * TS;
    const EPS: f32 = 1e-5;
    const EPSSQ: f32 = 1e-10;
    if width < 2 * BORDER || height < 2 * BORDER || mosaic.len() < width * height {
        return out;
    }
    use crate::raw::fc_bayer;
    let filters = filters_from_cfa(cfa);
    let sqrf = |a: f32| a * a;
    let clip01 = |x: f32| if x >= 0.0 { x.min(1.0) } else { 0.0 };
    // interpolatef(a, b, c) = a*b + (1-a)*c  (math.h:141)
    let interp = |a: f32, b: f32, c: f32| a * (b - c) + c;

    let num_vertical = (height - 2 * BORDER).saturating_sub(1) / TILEVALID + 1;
    let num_horizontal = (width - 2 * BORDER).saturating_sub(1) / TILEVALID + 1;

    // Refine each tile in PARALLEL (rayon), restoring the C `DT_OMP` the port
    // dropped: the scratch is per-worker (`for_each_init`), and every tile writes
    // only its own disjoint valid region of `out` through a raw pointer. The
    // half-resolution `lpf`/`pq_dir`/`p_cdiff`/`q_cdiff` buffers keep C's
    // full-`W1`-stride indexing over a TS²/2 buffer (why `RCD_TILEVALID` is even);
    // `vsq`/`hsq` replace C's rolling 3-row window with full-tile buffers.
    use rayon::prelude::*;
    let plane = TS * TS; // green plane base = 1*plane
    let out_ptr = SyncMutPtr(out.as_mut_ptr());
    let tiles: Vec<(usize, usize)> = (0..num_vertical)
        .flat_map(|tv| (0..num_horizontal).map(move |th| (tv, th)))
        .collect();
    tiles.par_iter().for_each_init(RcdScratch::new, |scratch, &(tv, th)| {
        let RcdScratch { cfa_b, rgb, vh_dir, pq_dir, lpf, p_cdiff, q_cdiff, vsq, hsq } = scratch;
        // Force the closure to capture the whole `SyncMutPtr` (Send+Sync), not
        // the bare `*mut f32` field: Rust 2021 disjoint capture would otherwise
        // capture `out_ptr.0`, which is not Sync. The rebind is deliberate.
        #[allow(clippy::redundant_locals)]
        let out_ptr = out_ptr;
        let op_base = out_ptr.0;
        {
            let row_start = tv * TILEVALID;
            let row_end = (row_start + TS).min(height);
            let col_start = th * TILEVALID;
            let col_end = (col_start + TS).min(width);
            let tile_rows = (row_end - row_start).min(TS) as i32;
            let tile_cols = (col_end - col_start).min(TS) as i32;

            cfa_b.fill(0.0);
            rgb.fill(0.0);
            vh_dir.fill(0.0);
            vsq.fill(0.0);
            hsq.fill(0.0);
            pq_dir.fill(0.0);
            lpf.fill(0.0);
            p_cdiff.fill(0.0);
            q_cdiff.fill(0.0);

            // Step 0: fill cfa; seed both in-row colour planes with the raw value.
            for row in row_start..row_end {
                let c0 = fc_bayer(row as i32, col_start as i32, filters);
                let c1 = fc_bayer(row as i32, (col_start + 1) as i32, filters);
                let base = (row - row_start) * TS;
                for (k, col) in (col_start..col_end).enumerate() {
                    let indx = base + k;
                    let v = mosaic[row * width + col].max(0.0); // _safe_in, scaler = 1
                    cfa_b[indx] = v;
                    // Warm-start both of this CFA row's two colour planes with the
                    // raw value; steps 3+4 overwrite them with interpolated values.
                    rgb[c0 * plane + indx] = v;
                    rgb[c1 * plane + indx] = v;
                }
            }

            // Step 1.1: squared vertical/horizontal colour-difference HPF.
            for row in 3..(tile_rows - 3) {
                for col in 4..(tile_cols - 4) {
                    let i = (row * TS as i32 + col) as usize;
                    vsq[i] = sqrf(
                        (cfa_b[i - W3] - cfa_b[i - W1] - cfa_b[i + W1] + cfa_b[i + W3])
                            - 3.0 * (cfa_b[i - W2] + cfa_b[i + W2])
                            + 6.0 * cfa_b[i],
                    );
                }
            }
            for row in 4..(tile_rows - 4) {
                for col in 3..(tile_cols - 3) {
                    let i = (row * TS as i32 + col) as usize;
                    hsq[i] = sqrf(
                        (cfa_b[i - 3] - cfa_b[i - 1] - cfa_b[i + 1] + cfa_b[i + 3])
                            - 3.0 * (cfa_b[i - 2] + cfa_b[i + 2])
                            + 6.0 * cfa_b[i],
                    );
                }
            }
            // Step 1.2: vertical/horizontal directional discrimination strength.
            for row in 4..(tile_rows - 4) {
                for col in 4..(tile_cols - 4) {
                    let i = (row * TS as i32 + col) as usize;
                    let v_stat = EPSSQ.max(vsq[i - W1] + vsq[i] + vsq[i + W1]);
                    let h_stat = EPSSQ.max(hsq[i - 1] + hsq[i] + hsq[i + 1]);
                    vh_dir[i] = v_stat / (v_stat + h_stat);
                }
            }

            // Step 2: low pass filter (green/red/blue local samples).
            for row in 2..(tile_rows - 2) {
                let mut col = 2 + (fc_bayer(row, 0, filters) & 1) as i32;
                while col < tile_cols - 2 {
                    let i = (row * TS as i32 + col) as usize;
                    lpf[i / 2] = cfa_b[i]
                        + 0.5 * (cfa_b[i - W1] + cfa_b[i + W1] + cfa_b[i - 1] + cfa_b[i + 1])
                        + 0.25
                            * (cfa_b[i - W1 - 1]
                                + cfa_b[i - W1 + 1]
                                + cfa_b[i + W1 - 1]
                                + cfa_b[i + W1 + 1]);
                    col += 2;
                }
            }

            // Step 3: green channel at blue and red CFA positions.
            for row in 4..(tile_rows - 4) {
                let mut col = 4 + (fc_bayer(row, 0, filters) & 1) as i32;
                while col < tile_cols - 4 {
                    let i = (row * TS as i32 + col) as usize;
                    // Tightest usize-subtraction invariant in the whole port:
                    // `cfa_b[i - W4]` below. row ≥ 4 ∧ col ≥ 4 ⇒ i ≥ 452 > W4.
                    debug_assert!(i >= W4, "step-3 usize guard (i - W4)");
                    let lp = i / 2;
                    let cfai = cfa_b[i];
                    let n_grad = EPS
                        + (cfa_b[i - W1] - cfa_b[i + W1]).abs()
                        + (cfai - cfa_b[i - W2]).abs()
                        + (cfa_b[i - W1] - cfa_b[i - W3]).abs()
                        + (cfa_b[i - W2] - cfa_b[i - W4]).abs();
                    let s_grad = EPS
                        + (cfa_b[i + W1] - cfa_b[i - W1]).abs()
                        + (cfai - cfa_b[i + W2]).abs()
                        + (cfa_b[i + W1] - cfa_b[i + W3]).abs()
                        + (cfa_b[i + W2] - cfa_b[i + W4]).abs();
                    let w_grad = EPS
                        + (cfa_b[i - 1] - cfa_b[i + 1]).abs()
                        + (cfai - cfa_b[i - 2]).abs()
                        + (cfa_b[i - 1] - cfa_b[i - 3]).abs()
                        + (cfa_b[i - 2] - cfa_b[i - 4]).abs();
                    let e_grad = EPS
                        + (cfa_b[i + 1] - cfa_b[i - 1]).abs()
                        + (cfai - cfa_b[i + 2]).abs()
                        + (cfa_b[i + 1] - cfa_b[i + 3]).abs()
                        + (cfa_b[i + 2] - cfa_b[i + 4]).abs();
                    let lpfi = lpf[lp];
                    let n_est = cfa_b[i - W1] * (lpfi + lpfi) / (EPS + lpfi + lpf[lp - W1]);
                    let s_est = cfa_b[i + W1] * (lpfi + lpfi) / (EPS + lpfi + lpf[lp + W1]);
                    let w_est = cfa_b[i - 1] * (lpfi + lpfi) / (EPS + lpfi + lpf[lp - 1]);
                    let e_est = cfa_b[i + 1] * (lpfi + lpfi) / (EPS + lpfi + lpf[lp + 1]);
                    let v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad);
                    let h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad);
                    let vh_c = vh_dir[i];
                    let vh_n = 0.25
                        * (vh_dir[i - W1 - 1]
                            + vh_dir[i - W1 + 1]
                            + vh_dir[i + W1 - 1]
                            + vh_dir[i + W1 + 1]);
                    let vh_disc = if (0.5 - vh_c).abs() < (0.5 - vh_n).abs() { vh_n } else { vh_c };
                    rgb[plane + i] = interp(clip01(vh_disc), h_est, v_est);
                    col += 2;
                }
            }

            // Step 4.0: squared P/Q diagonal colour-difference HPF.
            for row in 3..(tile_rows - 3) {
                let mut col = 3;
                while col < tile_cols - 3 {
                    let i = (row * TS as i32 + col) as usize;
                    let i2 = i / 2;
                    p_cdiff[i2] = sqrf(
                        (cfa_b[i - W3 - 3] - cfa_b[i - W1 - 1] - cfa_b[i + W1 + 1] + cfa_b[i + W3 + 3])
                            - 3.0 * (cfa_b[i - W2 - 2] + cfa_b[i + W2 + 2])
                            + 6.0 * cfa_b[i],
                    );
                    q_cdiff[i2] = sqrf(
                        (cfa_b[i - W3 + 3] - cfa_b[i - W1 + 1] - cfa_b[i + W1 - 1] + cfa_b[i + W3 - 3])
                            - 3.0 * (cfa_b[i - W2 + 2] + cfa_b[i + W2 - 2])
                            + 6.0 * cfa_b[i],
                    );
                    col += 2;
                }
            }
            // Step 4.1: P/Q diagonal directional discrimination strength.
            for row in 4..(tile_rows - 4) {
                let mut col = 4 + (fc_bayer(row, 0, filters) & 1) as i32;
                while col < tile_cols - 4 {
                    let i = (row * TS as i32 + col) as usize;
                    let i2 = i / 2;
                    let i3 = (i - W1 - 1) / 2;
                    let i4 = (i + W1 - 1) / 2;
                    let p_stat = EPSSQ.max(p_cdiff[i3] + p_cdiff[i2] + p_cdiff[i4 + 1]);
                    let q_stat = EPSSQ.max(q_cdiff[i3 + 1] + q_cdiff[i2] + q_cdiff[i4]);
                    pq_dir[i2] = p_stat / (p_stat + q_stat);
                    col += 2;
                }
            }
            // Step 4.2: red and blue channels at blue and red CFA positions.
            for row in 4..(tile_rows - 4) {
                let start = 4 + (fc_bayer(row, 0, filters) & 1) as i32;
                // Every non-green site in a Bayer row is the same colour, so the
                // opposite chroma plane is constant across the row — hoist it.
                let c = 2 - fc_bayer(row, start, filters);
                let rc = c * plane;
                let mut col = start;
                while col < tile_cols - 4 {
                    let i = (row * TS as i32 + col) as usize;
                    let pq = i / 2;
                    let pq2 = (i - W1 - 1) / 2;
                    let pq3 = (i + W1 - 1) / 2;
                    let pq_c = pq_dir[pq];
                    let pq_n = 0.25 * (pq_dir[pq2] + pq_dir[pq2 + 1] + pq_dir[pq3] + pq_dir[pq3 + 1]);
                    let pq_disc = if (0.5 - pq_c).abs() < (0.5 - pq_n).abs() { pq_n } else { pq_c };
                    let nw_grad = EPS
                        + (rgb[rc + i - W1 - 1] - rgb[rc + i + W1 + 1]).abs()
                        + (rgb[rc + i - W1 - 1] - rgb[rc + i - W3 - 3]).abs()
                        + (rgb[plane + i] - rgb[plane + i - W2 - 2]).abs();
                    let ne_grad = EPS
                        + (rgb[rc + i - W1 + 1] - rgb[rc + i + W1 - 1]).abs()
                        + (rgb[rc + i - W1 + 1] - rgb[rc + i - W3 + 3]).abs()
                        + (rgb[plane + i] - rgb[plane + i - W2 + 2]).abs();
                    let sw_grad = EPS
                        + (rgb[rc + i - W1 + 1] - rgb[rc + i + W1 - 1]).abs()
                        + (rgb[rc + i + W1 - 1] - rgb[rc + i + W3 - 3]).abs()
                        + (rgb[plane + i] - rgb[plane + i + W2 - 2]).abs();
                    let se_grad = EPS
                        + (rgb[rc + i - W1 - 1] - rgb[rc + i + W1 + 1]).abs()
                        + (rgb[rc + i + W1 + 1] - rgb[rc + i + W3 + 3]).abs()
                        + (rgb[plane + i] - rgb[plane + i + W2 + 2]).abs();
                    let nw_est = rgb[rc + i - W1 - 1] - rgb[plane + i - W1 - 1];
                    let ne_est = rgb[rc + i - W1 + 1] - rgb[plane + i - W1 + 1];
                    let sw_est = rgb[rc + i + W1 - 1] - rgb[plane + i + W1 - 1];
                    let se_est = rgb[rc + i + W1 + 1] - rgb[plane + i + W1 + 1];
                    let p_est = (nw_grad * se_est + se_grad * nw_est) / (nw_grad + se_grad);
                    let q_est = (ne_grad * sw_est + sw_grad * ne_est) / (ne_grad + sw_grad);
                    rgb[rc + i] = rgb[plane + i] + interp(clip01(pq_disc), q_est, p_est);
                    col += 2;
                }
            }
            // Step 4.3: red and blue channels at green CFA positions.
            for row in 4..(tile_rows - 4) {
                let mut col = 4 + (fc_bayer(row, 1, filters) & 1) as i32;
                while col < tile_cols - 4 {
                    let i = (row * TS as i32 + col) as usize;
                    let vh_c = vh_dir[i];
                    let vh_n = 0.25
                        * (vh_dir[i - W1 - 1]
                            + vh_dir[i - W1 + 1]
                            + vh_dir[i + W1 - 1]
                            + vh_dir[i + W1 + 1]);
                    let vh_disc = if (0.5 - vh_c).abs() < (0.5 - vh_n).abs() { vh_n } else { vh_c };
                    let g = rgb[plane + i];
                    let n1 = EPS + (g - rgb[plane + i - W2]).abs();
                    let s1 = EPS + (g - rgb[plane + i + W2]).abs();
                    let w1v = EPS + (g - rgb[plane + i - 2]).abs();
                    let e1 = EPS + (g - rgb[plane + i + 2]).abs();
                    let g_mw1 = rgb[plane + i - W1];
                    let g_pw1 = rgb[plane + i + W1];
                    let g_m1 = rgb[plane + i - 1];
                    let g_p1 = rgb[plane + i + 1];
                    for c in [0usize, 2] {
                        let rc = c * plane;
                        let snabs = (rgb[rc + i - W1] - rgb[rc + i + W1]).abs();
                        let ewabs = (rgb[rc + i - 1] - rgb[rc + i + 1]).abs();
                        let n_grad = n1 + snabs + (rgb[rc + i - W1] - rgb[rc + i - W3]).abs();
                        let s_grad = s1 + snabs + (rgb[rc + i + W1] - rgb[rc + i + W3]).abs();
                        let w_grad = w1v + ewabs + (rgb[rc + i - 1] - rgb[rc + i - 3]).abs();
                        let e_grad = e1 + ewabs + (rgb[rc + i + 1] - rgb[rc + i + 3]).abs();
                        let n_est = rgb[rc + i - W1] - g_mw1;
                        let s_est = rgb[rc + i + W1] - g_pw1;
                        let w_est = rgb[rc + i - 1] - g_m1;
                        let e_est = rgb[rc + i + 1] - g_p1;
                        let v_est = (n_grad * s_est + s_grad * n_est) / (n_grad + s_grad);
                        let h_est = (e_grad * w_est + w_grad * e_est) / (e_grad + w_grad);
                        rgb[rc + i] = g + interp(clip01(vh_disc), h_est, v_est);
                    }
                    col += 2;
                }
            }

            // Write the valid region back to the RGBA output (image coordinates).
            // Outermost tiles use the smaller RCD_MARGIN, interior joins RCD_BORDER.
            let first_v = row_start + if tv == 0 { MARGIN } else { BORDER };
            let last_v = row_end - if tv == num_vertical - 1 { MARGIN } else { BORDER };
            let first_h = col_start + if th == 0 { MARGIN } else { BORDER };
            let last_h = col_end - if th == num_horizontal - 1 { MARGIN } else { BORDER };
            for row in first_v..last_v {
                for col in first_h..last_h {
                    let idx = (row - row_start) * TS + (col - col_start);
                    let o = (row * width + col) * 4;
                    // SAFETY: `o..o+4` lies in this tile's valid region, which is
                    // disjoint from every other tile's (see `SyncMutPtr`) and in
                    // bounds (`out` is width*height*4 from `demosaic_ppg`). The
                    // unchecked write has no bounds check, so fail fast in debug if
                    // a future tiling-formula change ever breaks that invariant.
                    debug_assert!(o + 3 < width * height * 4, "rcd write-back OOB: o={o}");
                    unsafe {
                        *op_base.add(o) = rgb[idx].max(0.0);
                        *op_base.add(o + 1) = rgb[plane + idx].max(0.0);
                        *op_base.add(o + 2) = rgb[2 * plane + idx].max(0.0);
                        *op_base.add(o + 3) = 0.0;
                    }
                }
            }
        }
    });
    out
}

// dcraw VNG gradient term table (64 terms × {y1,x1,y2,x2,weight,grad-bits}) plus
// the 8 clockwise gradient-direction offsets (chood), transcribed EXACTLY from
// vng.c:97-115 (extracted programmatically, not hand-typed). Grad bytes ≥ 0x80
// become negative `i8`, but only bits 0..7 are ever tested via
// `(v as i32) & (1 << g)`, which reproduces C's signed-char→int promotion.
#[rustfmt::skip]
static VNG_TERMS: [i8; 384] = [
    -2, -2, 0, -1, 1, 1,   -2, -2, 0, 0, 2, 1,   -2, -1, -1, 0, 1, 1,   -2, -1, 0, -1, 1, 2,
    -2, -1, 0, 0, 1, 3,    -2, -1, 0, 1, 2, 1,   -2, 0, 0, -1, 1, 6,    -2, 0, 0, 0, 2, 2,
    -2, 0, 0, 1, 1, 3,     -2, 1, -1, 0, 1, 4,   -2, 1, 0, -1, 2, 4,    -2, 1, 0, 0, 1, 6,
    -2, 1, 0, 1, 1, 2,     -2, 2, 0, 0, 2, 4,    -2, 2, 0, 1, 1, 4,     -1, -2, -1, 0, 1, -128,
    -1, -2, 0, -1, 1, 1,   -1, -2, 1, -1, 1, 1,  -1, -2, 1, 0, 2, 1,    -1, -1, -1, 1, 1, -120,
    -1, -1, 1, -2, 1, 64,  -1, -1, 1, -1, 1, 34, -1, -1, 1, 0, 1, 51,   -1, -1, 1, 1, 2, 17,
    -1, 0, -1, 2, 1, 8,    -1, 0, 0, -1, 1, 68,  -1, 0, 0, 1, 1, 17,    -1, 0, 1, -2, 2, 64,
    -1, 0, 1, -1, 1, 102,  -1, 0, 1, 0, 2, 34,   -1, 0, 1, 1, 1, 51,    -1, 0, 1, 2, 2, 16,
    -1, 1, 1, -1, 2, 68,   -1, 1, 1, 0, 1, 102,  -1, 1, 1, 1, 1, 34,    -1, 1, 1, 2, 1, 16,
    -1, 2, 0, 1, 1, 4,     -1, 2, 1, 0, 2, 4,    -1, 2, 1, 1, 1, 4,     0, -2, 0, 0, 2, -128,
    0, -1, 0, 1, 2, -120,  0, -1, 1, -2, 1, 64,  0, -1, 1, 0, 1, 17,    0, -1, 2, -2, 1, 64,
    0, -1, 2, -1, 1, 32,   0, -1, 2, 0, 1, 48,   0, -1, 2, 1, 2, 16,    0, 0, 0, 2, 2, 8,
    0, 0, 2, -2, 2, 64,    0, 0, 2, -1, 1, 96,   0, 0, 2, 0, 2, 32,     0, 0, 2, 1, 1, 48,
    0, 0, 2, 2, 2, 16,     0, 1, 1, 0, 1, 68,    0, 1, 1, 2, 1, 16,     0, 1, 2, -1, 2, 64,
    0, 1, 2, 0, 1, 96,     0, 1, 2, 1, 1, 32,    0, 1, 2, 2, 1, 16,     1, -2, 1, 0, 1, -128,
    1, -1, 1, 1, 1, -120,  1, 0, 1, 2, 1, 8,     1, 0, 2, -1, 1, 64,    1, 0, 2, 1, 1, 16,
];
#[rustfmt::skip]
static VNG_CHOOD: [i8; 16] = [
    -1, -1, -1, 0, -1, 1, 0, 1,   1, 1, 1, 0, 1, -1, 0, -1,
];

/// Build the VNG linear-interpolation lookup table (`_vng_lininterpolate`,
/// vng.c:46-75) as the flat `i32[16*16*32]` the `darkroom_demosaic_vng_lookup`
/// kernel consumes. Bayer-only (size 16). Neighbour offsets are in pixels and
/// depend on `width`, so this is rebuilt per call. Per cell `[row*16+col]`:
/// `[0]` = neighbour count `np`, then `np` (offset, weight, colour) triples,
/// then `colours-1` (colour, weight-sum) pairs, then the centre colour.
fn build_vng_lookup(width: usize, filters4: u32) -> Vec<i32> {
    use crate::raw::fcol;
    let xt = [[0u8; 6]; 6]; // unused for Bayer
    let mut lut = vec![0i32; 16 * 16 * 32];
    let w = width as i32;
    for row in 0..16i32 {
        for col in 0..16i32 {
            let cell = ((row * 16 + col) as usize) * 32;
            let f = fcol(row, col, filters4, &xt);
            let mut sum = [0i32; 4];
            let mut k = cell + 1;
            let mut np = 0i32;
            for y in -1..=1 {
                for x in -1..=1 {
                    let weight = 1i32 << (((y == 0) as i32) + ((x == 0) as i32));
                    let color = fcol(row + y, col + x, filters4, &xt);
                    if color == f {
                        continue;
                    }
                    lut[k] = w * y + x;
                    lut[k + 1] = weight;
                    lut[k + 2] = color as i32;
                    sum[color] += weight;
                    k += 3;
                    np += 1;
                }
            }
            lut[cell] = np;
            // `colors == sum.len()` for Bayer; emit a (colour, weight-sum) pair
            // for every colour except the centre's own.
            for (c, &s) in sum.iter().enumerate() {
                if c != f {
                    lut[k] = c as i32;
                    lut[k + 1] = s;
                    k += 2;
                }
            }
            lut[k] = f as i32;
        }
    }
    lut
}

/// Build the VNG gradient `code[prow][pcol]` streams (vng.c:161-197), one
/// `Vec<i32>` per `(row, col)`, row-major (`row*pcol + col`). Each stream is the
/// `INT_MAX`-terminated term list (per surviving term: two packed neighbour
/// offsets, a weight, the set gradient-direction indices, `-1`) followed by 8
/// chood (offset, colour-or-0) pairs. Offsets are baked to RGBA stride and
/// depend on `width`.
fn build_vng_code(prow: usize, pcol: usize, width: usize, filters4: u32) -> Vec<Vec<i32>> {
    use crate::raw::fcol;
    let xt = [[0u8; 6]; 6];
    let w = width as i32;
    let mut code = Vec::with_capacity(prow * pcol);
    for row in 0..prow as i32 {
        for col in 0..pcol as i32 {
            let mut ip: Vec<i32> = Vec::with_capacity(320);
            for t in 0..64usize {
                let b = t * 6;
                let (y1, x1) = (VNG_TERMS[b] as i32, VNG_TERMS[b + 1] as i32);
                let (y2, x2) = (VNG_TERMS[b + 2] as i32, VNG_TERMS[b + 3] as i32);
                let weight = VNG_TERMS[b + 4] as i32;
                let grads = VNG_TERMS[b + 5] as i32;
                let color = fcol(row + y1, col + x1, filters4, &xt);
                if fcol(row + y2, col + x2, filters4, &xt) != color {
                    continue;
                }
                let diag = if fcol(row, col + 1, filters4, &xt) == color
                    && fcol(row + 1, col, filters4, &xt) == color
                {
                    2
                } else {
                    1
                };
                if (y1 - y2).abs() == diag && (x1 - x2).abs() == diag {
                    continue;
                }
                ip.push((y1 * w + x1) * 4 + color as i32);
                ip.push((y2 * w + x2) * 4 + color as i32);
                ip.push(weight);
                for g in 0..8 {
                    if grads & (1 << g) != 0 {
                        ip.push(g);
                    }
                }
                ip.push(-1);
            }
            ip.push(i32::MAX);
            for g in 0..8usize {
                let (y, x) = (VNG_CHOOD[2 * g] as i32, VNG_CHOOD[2 * g + 1] as i32);
                ip.push((y * w + x) * 4);
                let color = fcol(row, col, filters4, &xt);
                if fcol(row + y, col + x, filters4, &xt) != color
                    && fcol(row + 2 * y, col + 2 * x, filters4, &xt) == color
                {
                    ip.push((y * w + x) * 8 + color as i32);
                } else {
                    ip.push(0);
                }
            }
            code.push(ip);
        }
    }
    code
}

/// Demosaic a normalised Bayer CFA mosaic via **VNG** (Variable Number of
/// Gradients) — dcraw's threshold-based interpolation, a high-quality
/// alternative to [`demosaic_rcd`]. Assembles darktable's `vng_interpolate`
/// (`src/iop/demosaicing/vng.c`) natively: the migrated per-pass kernels in
/// [`crate::iop::demosaic`] (`vng_border`, `vng_lookup`, `vng_gradient_row`,
/// `vng_finish`) driven by the table builders ported here ([`build_vng_lookup`],
/// [`build_vng_code`]) plus the C 3-row ring buffer with its 2-row-deferred
/// write-back (so the gradient kernel always reads the un-refined base image).
///
/// Bayer only (X-Trans routes to [`demosaic_xtrans`]). RGGB greens are split
/// into G1/G2 (`filters4`) for the 4-colour interpolation, then re-merged by the
/// finish pass. Frames too small for the gradient interior get the linear-only
/// VNG (border + lookup), itself a complete demosaic. Leaves the 4th channel at
/// 0 like the sibling demosaicers; [`RawImage::to_linear_rgba`] sets alpha.
pub fn demosaic_vng(mosaic: &[f32], width: usize, height: usize, cfa: [[usize; 2]; 2]) -> Vec<f32> {
    let n = width.saturating_mul(height);
    let mut out = vec![0.0f32; n * 4];
    if mosaic.len() < n || n == 0 {
        return out;
    }
    let filters = filters_from_cfa(cfa);
    // Separate the two Bayer greens (G1=1, G2=3) so the 4-colour VNG treats them
    // apart (vng.c:137-143); the finish pass averages them back into green.
    let filters4 = if (filters & 3) == 1 {
        filters | 0x0303_0303
    } else {
        filters | 0x0c0c_0c0c
    };
    let colors = 4usize; // Bayer
    let (prow, pcol) = (8usize, 2usize);
    let xtb = [0u8; 36]; // unused for Bayer, but the kernels read 36 bytes

    // Linear VNG: border ring interpolation + lookup-table interior. `border`
    // = 1_000_000 disables the tile ring-skip (full-frame path), matching C.
    // Safety: `out` is n*4 floats, `mosaic` is ≥ n, `xtb` is 36 bytes, `lookup`
    // is exactly 16*16*32 i32 — every kernel's documented contract.
    unsafe {
        crate::iop::demosaic::darkroom_demosaic_vng_border(
            out.as_mut_ptr(),
            mosaic.as_ptr(),
            width,
            height,
            filters4,
            xtb.as_ptr(),
        );
    }
    let lookup = build_vng_lookup(width, filters4);
    unsafe {
        crate::iop::demosaic::darkroom_demosaic_vng_lookup(
            out.as_mut_ptr(),
            mosaic.as_ptr(),
            width,
            height,
            filters4,
            1_000_000,
            lookup.as_ptr(),
        );
    }

    // Gradient VNG (C vng.c:199-213). The serial C keeps a 3-row ring buffer and
    // defers each row's write-back by two rows so a gradient never reads an
    // already-refined row: at the moment row R is computed, its ±2-row stencil
    // (rows R-2..R+2) still holds the border+lookup `out`, since the deferred
    // copy has only overwritten rows ≤ R-3. Freezing that border+lookup `out`
    // into a read-only `src` therefore feeds every row the exact same inputs the
    // serial sweep saw, making the per-row work embarrassingly parallel and
    // bit-identical. The kernel reads `src` read-only and writes only the
    // interior cols 2..width-2 of its own row, so the `par_chunks_mut` row
    // slices are disjoint and the edge cols keep their border/lookup values —
    // exactly the region the serial ring buffer left untouched (its copy started
    // at float offset 8 = col 2).
    if width >= 6 && height >= 6 {
        use rayon::prelude::*;
        let code = build_vng_code(prow, pcol, width, filters4);
        let row_len = width * 4;
        let src = out.clone(); // border+lookup values, frozen for every row
        out.par_chunks_mut(row_len)
            .enumerate()
            .filter(|&(row, _)| (2..height - 2).contains(&row))
            .for_each(|(row, out_row)| {
                let pr = (row % prow) * pcol;
                // `code`/`src` are read-only for the whole sweep, so the raw
                // pointers into their inner storage stay valid across threads.
                let cptrs: [*const i32; 2] = [code[pr].as_ptr(), code[pr + 1].as_ptr()];
                // Safety: `src` is n*4 floats (read-only), `out_row` is exactly
                // width*4 (the kernel writes only interior cols 2..width-2, so
                // the edge cols keep their border/lookup values), `xtb` 36 bytes,
                // `cptrs` holds `pcol` valid INT_MAX-terminated streams; column
                // reads are independent so this equals the C OMP sweep.
                unsafe {
                    crate::iop::demosaic::darkroom_demosaic_vng_gradient_row(
                        src.as_ptr(),
                        out_row.as_mut_ptr(),
                        width,
                        height,
                        row as i32,
                        filters4,
                        xtb.as_ptr(),
                        colors as i32,
                        cptrs.as_ptr(),
                        pcol as i32,
                    );
                }
            });
    }

    // Finish: average G1/G2 back into the green channel + clip negatives.
    // Safety: `out` is n*4 floats.
    unsafe {
        crate::iop::demosaic::darkroom_demosaic_vng_finish(out.as_mut_ptr(), n, 1);
    }
    out
}

/// Demosaic a normalised **X-Trans** CFA mosaic to packed RGBA `f32` via the
/// migrated single-pass **Markesteijn** interpolation
/// (`iop::markesteijn::darkroom_xtrans_markesteijn`, xtrans.c:45). The kernel
/// tiles the full frame internally, so this is one call over the whole image.
/// Returns a zeroed buffer for a malformed/short plane or an image too small for
/// one Markesteijn tile interior (real X-Trans frames are always large).
pub fn demosaic_xtrans(
    mosaic: &[f32],
    width: usize,
    height: usize,
    xtrans: &[[u8; 6]; 6],
) -> Vec<f32> {
    let n = width.saturating_mul(height);
    let mut out = vec![0.0f32; n.saturating_mul(4)];
    // Markesteijn reads ±pad_tile (12) around each tile; below ~16px a full
    // frame can't supply a written interior. Real raws are thousands of px, so
    // this only guards synthetic/degenerate inputs (left zeroed, not a panic).
    // The `i32::MAX` cap closes the unsafe path: the kernel takes `width`/`height`
    // as `i32` and rebuilds its slice lengths from them, so a dimension that
    // wraps negative on the cast would under-size the slice → OOB (no real raw
    // is anywhere near 2³¹px, but the cast must be sound regardless).
    if mosaic.len() < n
        || width < 16
        || height < 16
        || width > i32::MAX as usize
        || height > i32::MAX as usize
    {
        return out;
    }
    let xt: [u8; 36] = std::array::from_fn(|i| xtrans[i / 6][i % 6]);
    // darktable marks an X-Trans sensor with filters == 9; the kernel ignores
    // the value (it reads the 6×6 `xtrans` table) but we pass it for fidelity.
    const XTRANS_FILTERS: u32 = 9;
    const PASSES: i32 = 1; // single-pass Markesteijn (3-pass is slower, marginal)
    // Safety: `out` is `n*4` floats, `mosaic` is `>= n` floats, and `xt` is 36
    // bytes — exactly the kernel's documented contract.
    unsafe {
        crate::iop::markesteijn::darkroom_xtrans_markesteijn(
            out.as_mut_ptr(),
            mosaic.as_ptr(),
            width as i32,
            height as i32,
            xt.as_ptr(),
            PASSES,
            XTRANS_FILTERS,
        );
    }
    out
}

/// The identity 3×3: camera colours passed through unchanged (treated as already
/// Rec.2020, the working space — the display seam still applies to them). The
/// colour-matrix fallback when a file carries no usable XYZ→camera matrix, and
/// the neutral value for the demo/test `RawImage`.
pub const IDENTITY3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Linear Rec.2020 → CIE XYZ (D65), derived from the Rec.2020 primaries and the
/// D65 white (Lindbloom construction, so RGB white maps exactly to XYZ D65).
/// Composed with the raw's XYZ→camera matrix to build camera→Rec.2020 (the
/// working space, m4-35). Rec.2020 is wider than sRGB, so saturated camera
/// colours stay in-gamut through the tone/saturation stages instead of clipping.
const REC2020_XYZ: [[f64; 3]; 3] = [
    [0.63701019, 0.14461503, 0.16884478],
    [0.26272172, 0.67798928, 0.05928901],
    [0.00000000, 0.02807233, 1.06075767],
];

/// Linear Rec.2020 → linear sRGB (D65), for the display-encode seam: the preview
/// stages run in the Rec.2020 working space, then this maps to sRGB just before
/// the OETF (`preview::render_linear_to_srgb8`). Derived as `inv(sRGB→XYZ) ·
/// (Rec.2020→XYZ)` from the same D65 primaries construction as [`REC2020_XYZ`],
/// so every row sums to 1 — a neutral maps to a neutral EXACTLY (no grey tint).
/// Out-of-sRGB-gamut Rec.2020 colours produce negatives here, hard-clipped at the
/// OETF (the display can't show them). `pub` for the c41-ui render seam.
pub const REC2020_TO_SRGB: [[f32; 3]; 3] = [
    [1.66036266, -0.58754000, -0.07282266],
    [-0.12456355, 1.13291137, -0.00834783],
    [-0.01815661, -0.10060173, 1.11875834],
];

/// Invert a 3×3 matrix, or `None` when it is (near-)singular.
fn mat3_inverse(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let d = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d,
        ],
    ])
}

/// Derive the camera-native-RGB → linear-Rec.2020 (D65) 3×3 from the raw's
/// XYZ→camera matrix (`rawloader`'s `xyz_to_cam`; the top 3 rows are used — a
/// 4-colour CFA's 4th row is ignored, since our demosaic yields 3-channel RGB).
/// Follows dcraw's `cam_xyz_coeff`: form `cam_rgb = xyz_to_cam · (Rec.2020→XYZ)`
/// (a Rec.2020→camera map), row-normalise it so a neutral maps to a neutral (each
/// camera channel's response to Rec.2020 white is unity), then invert to get
/// camera→Rec.2020. Returns [`IDENTITY3`] when the matrix is absent (rawloader
/// reports all-zeros for an unknown camera) or singular, so the pipeline falls
/// back to treating camera colours as the working space untransformed.
pub fn rec2020_from_cam_matrix(xyz_to_cam: [[f32; 3]; 4]) -> [[f32; 3]; 3] {
    // cam_rgb = xyz_to_cam(3×3) · REC2020_XYZ  → maps Rec.2020 to camera native.
    let mut cam_rgb = [[0.0f64; 3]; 3];
    for (i, row) in cam_rgb.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3)
                .map(|k| xyz_to_cam[i][k] as f64 * REC2020_XYZ[k][j])
                .sum();
        }
    }
    // Row-normalise (dcraw's `num`): make each row sum to 1 so camera white ==
    // Rec.2020 white, keeping neutrals neutral. A zero row (no matrix) is left
    // as-is and makes the inverse fail below → identity fallback.
    for row in cam_rgb.iter_mut() {
        let num: f64 = row.iter().sum();
        if num.abs() > 1e-12 {
            for cell in row.iter_mut() {
                *cell /= num;
            }
        }
    }
    // Invert (camera→Rec.2020); identity fallback when the file gave no usable matrix.
    match mat3_inverse(cam_rgb) {
        Some(inv) => {
            let mut out = IDENTITY3;
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j] = inv[i][j] as f32;
                }
            }
            out
        }
        None => IDENTITY3,
    }
}

/// Apply a 3×3 colour matrix to packed RGBA `f32` in place (alpha untouched).
/// Used both for camera-native RGB → working space (Rec.2020) and for the
/// working space → sRGB display seam ([`REC2020_TO_SRGB`]). Exactly a no-op for [`IDENTITY3`]
/// (`x*1 + y*0 + z*0 == x`). Out-of-gamut results may go slightly negative; that
/// is left unclamped for the scene-linear pipeline (the tone map rolls it off),
/// matching darktable's input-profile behaviour.
pub fn apply_color_matrix(rgba: &mut [f32], m: [[f32; 3]; 3]) {
    for px in rgba.chunks_exact_mut(4) {
        let (r, g, b) = (px[0], px[1], px[2]);
        px[0] = m[0][0] * r + m[0][1] * g + m[0][2] * b;
        px[1] = m[1][0] * r + m[1][1] * g + m[1][2] * b;
        px[2] = m[2][0] * r + m[2][1] * g + m[2][2] * b;
    }
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

/// Apply white balance to a **CFA mosaic**: each photosite of colour `c` is
/// multiplied by `wb[c]/wb[grey]` (RGBE-ordered, normalised by green — the
/// mosaic-domain twin of [`apply_white_balance`]). Runs before demosaic when
/// highlight reconstruction is on, mirroring darktable's temperature-before-
/// highlights pipeline order. Degenerate `wb` leaves the mosaic untouched.
///
/// `color_at(row, col)` yields the photosite's RGBE colour index (see
/// [`classify_cfa`]). Pure — no decode dependency.
pub fn apply_white_balance_mosaic(
    mosaic: &mut [f32],
    width: usize,
    color_at: impl Fn(usize, usize) -> usize,
    wb: [f32; 4],
) {
    let g = wb[1];
    if g <= 0.0 || !g.is_finite() {
        return;
    }
    // Per-colour multipliers relative to green; an unfinitable coefficient
    // degrades to 1.0 so one bad file value cannot nuke a whole channel.
    let mult: [f32; 4] = core::array::from_fn(|c| {
        if wb[c].is_finite() { wb[c] / g } else { 1.0 }
    });
    for (i, v) in mosaic.iter_mut().enumerate() {
        let c = color_at(i / width, i % width);
        *v *= if c < 4 { mult[c] } else { 1.0 };
    }
}

impl RawImage {
    /// Demosaic + white-balance this raw into a packed **linear RGBA** `f32`
    /// buffer ready for [`crate::pipeline`], using the default Bayer demosaicer
    /// ([`DemosaicMethod::Rcd`]) and no highlight reconstruction. Returns
    /// `(width, height, pixels)`.
    pub fn to_linear_rgba(&self) -> (usize, usize, Vec<f32>) {
        self.to_linear_rgba_with(DemosaicMethod::default(), None)
    }

    /// Demosaic + white-balance with an explicit Bayer [`DemosaicMethod`] and
    /// optional highlight reconstruction.
    ///
    /// The method selects the Bayer demosaicer ([`demosaic_rcd`] / [`demosaic_vng`]
    /// / [`demosaic_ppg`]); **X-Trans ignores it** and always uses the Markesteijn
    /// [`demosaic_xtrans`]. RCD/PPG internally fall back to a simpler kernel for
    /// frames too small for their interior.
    ///
    /// `hl = Some(opts)` runs darktable-style highlight reconstruction
    /// ([`iop::highlights::reconstruct_mosaic`]): the stored mosaic carries
    /// over-range photosites (`> 1.0`, see [`normalize_cfa`]), which are
    /// white-balanced and reconstructed **before** demosaicing — darktable's
    /// temperature → highlights → demosaic order. Post-demosaic WB is then
    /// skipped (already in the mosaic). `hl = None` clamps over-range
    /// photosites at 1.0 first, reproducing the pre-m4-119 decoder exactly.
    ///
    /// Works on a copy of the mosaic either way (the stored one stays intact,
    /// so repeated calls with different options are safe); that transient
    /// buffer is the same allocation class as the RGBA result itself.
    pub fn to_linear_rgba_with(
        &self,
        method: DemosaicMethod,
        hl: Option<crate::iop::highlights::HlOpts>,
    ) -> (usize, usize, Vec<f32>) {
        let (w, h) = (self.width, self.height);
        let mut work = self.mosaic.clone();
        match hl {
            Some(opts) => {
                apply_white_balance_mosaic(&mut work, w, |row, col| match &self.xtrans {
                    Some(xt) => {
                        crate::raw::fcol(row as i32, col as i32, 9, xt)
                    }
                    None => self.cfa[row % 2][col % 2],
                }, self.wb);
                crate::iop::highlights::reconstruct_mosaic(
                    &mut work, w, h, self.cfa, self.xtrans.as_ref(), self.wb, opts,
                );
            }
            None => {
                // Pre-m4-119 behaviour: hard-clip photosites at sensor white.
                for v in work.iter_mut() {
                    *v = v.min(1.0);
                }
            }
        }
        let mut rgba = match &self.xtrans {
            Some(xt) => demosaic_xtrans(&work, w, h, xt),
            None => match method {
                DemosaicMethod::Rcd => demosaic_rcd(&work, w, h, self.cfa),
                DemosaicMethod::Vng => demosaic_vng(&work, w, h, self.cfa),
                DemosaicMethod::Ppg => demosaic_ppg(&work, w, h, self.cfa),
            },
        };
        if hl.is_none() {
            apply_white_balance(&mut rgba, self.wb);
        }
        // Camera-native RGB → linear Rec.2020 working space (no-op when the file
        // gave no matrix, so this is the identity for the synthetic/demo path).
        // After WB so the neutral-preserving, row-normalised matrix sees a
        // white-balanced neutral. The display seam later maps Rec.2020 → sRGB.
        apply_color_matrix(&mut rgba, self.cam_to_working);
        // the demosaic leaves the 4th channel at 0 (it has no contributors); set it
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
        assert_eq!(out[2], 50.0); // (150-100)/1 → carried over-range (m4-119)
        assert_eq!(out[3], 0.0); // (99-100)/1 → clamp 0
    }

    #[test]
    fn normalize_carries_over_range_for_reconstruction() {
        // m4-119: photosites above sensor white must survive normalisation so
        // highlight reconstruction has data; negatives still floor at 0.
        let cfa = [[0usize, 1], [1, 2]]; // RGGB
        let out = normalize_cfa(
            &[1100u16, 500, 600, 2100],
            2,
            2,
            cfa,
            [100.0, 100.0, 100.0, 100.0],
            [1000.0, 1000.0, 1000.0, 1000.0],
        );
        assert!((out[0] - 1000.0 / 900.0).abs() < 1e-5); // R over white: carried
        assert!((out[1] - 400.0 / 900.0).abs() < 1e-5);
        assert!((out[3] - 2000.0 / 900.0).abs() < 1e-4); // B way over white: carried
    }

    #[test]
    fn legacy_none_path_matches_clip_mode_at_unity_wb() {
        // m4-119 invariant: with unity white balance, clip-mode reconstruction
        // performs exactly the legacy decoder's operations (clamp at 1.0) in a
        // different order — so both entry points must agree bit-for-bit.
        use crate::iop::highlights::{HlMode, HlOpts};
        let img = RawImage {
            width: 12,
            height: 12,
            cfa: [[0usize, 1], [1, 2]],
            xtrans: None,
            wb: [1.0; 4],
            orientation: (false, false, false),
            cam_to_working: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            // Diagonal gradient with over-range photosites carried in.
            mosaic: (0..144).map(|i| (i % 29) as f32 / 20.0).collect(),
        };
        let a = img.to_linear_rgba_with(DemosaicMethod::Ppg, None);
        let b = img.to_linear_rgba_with(
            DemosaicMethod::Ppg,
            Some(HlOpts { mode: HlMode::Clip, clip: 1.0 }),
        );
        assert_eq!(a, b);
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
            xtrans: None,
            wb: [2.0, 1.0, 4.0, 1.0],
            orientation: (false, false, false),
            cam_to_working: IDENTITY3, // colour matrix is a no-op for this fixture
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
    fn to_linear_rgba_with_dispatches_by_method() {
        // 40×40 Bayer so every demosaicer runs its real interior (RCD needs
        // ≥ 2·RCD_BORDER = 20, PPG ≥ 16, VNG ≥ 6); a diagonal gradient gives both
        // H and V structure so the three algorithms' directional logic diverges.
        let (w, h) = (40usize, 40usize);
        let cfa = [[0usize, 1], [1, 2]];
        let mosaic: Vec<f32> = (0..w * h)
            .map(|i| ((i % w + i / w) as f32) / ((w + h) as f32))
            .collect();
        let img = RawImage {
            width: w,
            height: h,
            cfa,
            xtrans: None,
            wb: [1.0, 1.0, 1.0, 1.0],
            orientation: (false, false, false),
            cam_to_working: IDENTITY3,
            mosaic,
        };
        // The no-arg entry point delegates to RCD, byte-for-byte.
        assert_eq!(
            img.to_linear_rgba(),
            img.to_linear_rgba_with(DemosaicMethod::Rcd, None)
        );
        let (_, _, rcd) = img.to_linear_rgba_with(DemosaicMethod::Rcd, None);
        let (_, _, vng) = img.to_linear_rgba_with(DemosaicMethod::Vng, None);
        let (_, _, ppg) = img.to_linear_rgba_with(DemosaicMethod::Ppg, None);
        // Distinct algorithms ⇒ distinct output on a gradient — proves the match
        // arms actually reach different demosaicers, not one aliased path.
        assert_ne!(rcd, vng, "RCD and VNG output must differ");
        assert_ne!(rcd, ppg, "RCD and PPG output must differ");
        assert_ne!(vng, ppg, "VNG and PPG output must differ");
        for buf in [&rcd, &vng, &ppg] {
            assert_eq!(buf.len(), w * h * 4);
            assert!(buf.iter().all(|v| v.is_finite()), "non-finite output");
            for px in buf.chunks_exact(4) {
                assert_eq!(px[3], 1.0, "alpha must be opaque");
            }
        }
    }

    #[test]
    fn to_linear_rgba_with_method_ignored_for_xtrans() {
        // X-Trans always uses Markesteijn, so the Bayer method must not change
        // the result (guards the dispatch order: xtrans checked before method).
        let xt = XTRANS;
        let (w, h) = (24usize, 24usize); // multiple of the 6×6 pattern
        let mosaic: Vec<f32> = (0..w * h).map(|i| (i % 7) as f32 / 7.0).collect();
        let img = RawImage {
            width: w,
            height: h,
            cfa: [[0, 1], [1, 2]],
            xtrans: Some(xt),
            wb: [1.0, 1.0, 1.0, 1.0],
            orientation: (false, false, false),
            cam_to_working: IDENTITY3,
            mosaic,
        };
        let rcd = img.to_linear_rgba_with(DemosaicMethod::Rcd, None);
        assert_eq!(rcd, img.to_linear_rgba_with(DemosaicMethod::Vng, None));
        assert_eq!(rcd, img.to_linear_rgba_with(DemosaicMethod::Ppg, None));
    }

    #[test]
    fn demosaic_method_u8_round_trips_and_defaults() {
        for m in [DemosaicMethod::Rcd, DemosaicMethod::Vng, DemosaicMethod::Ppg] {
            assert_eq!(DemosaicMethod::from_u8(m.as_u8()), m);
        }
        assert_eq!(DemosaicMethod::Rcd.as_u8(), 0); // default code is 0
        // Unknown/corrupt codes fall back to the default, never panic.
        for v in [3u8, 7, 255] {
            assert_eq!(DemosaicMethod::from_u8(v), DemosaicMethod::default());
        }
    }

    #[test]
    fn rec2020_from_cam_matrix_zero_falls_back_to_identity() {
        // rawloader reports an all-zero matrix for an unknown camera → identity
        // (treat camera colours as the working space untransformed).
        assert_eq!(rec2020_from_cam_matrix([[0.0; 3]; 4]), IDENTITY3);
    }

    #[test]
    fn rec2020_from_cam_matrix_preserves_neutral_and_ignores_4th_row() {
        // For any non-singular matrix the row-normalise-then-invert construction
        // maps a camera neutral to a working-space neutral. The 4th row is ignored.
        let xyz_to_cam = [
            [0.6, 0.1, -0.1],
            [-0.2, 1.1, 0.1],
            [0.0, 0.1, 0.7],
            [9.9, 9.9, 9.9], // 4th row: must not affect the RGB result
        ];
        let m = rec2020_from_cam_matrix(xyz_to_cam);
        assert_ne!(m, IDENTITY3, "a real camera matrix must transform colour");
        // camera neutral (1,1,1) → working-space neutral (1,1,1)
        let mut px = vec![1.0f32, 1.0, 1.0, 1.0];
        apply_color_matrix(&mut px, m);
        for (c, v) in px[..3].iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-4, "channel {c} = {v}, expected neutral");
        }
        assert_eq!(px[3], 1.0, "alpha untouched");
    }

    #[test]
    fn rec2020_from_cam_matrix_matches_dcraw_golden_for_a_real_camera() {
        // Golden regression pinning the FULL construction (multiply order +
        // row-normalise + invert + constants), not just the by-construction
        // neutral invariant. `xyz_to_cam` is the Canon EOS 5D Mark III matrix from
        // dcraw's `adobe_coeff` (cam_xyz × 1e-4); the expected camera→Rec.2020 was
        // computed by an independent pure-Python implementation of dcraw's
        // `cam_xyz_coeff` (target primaries = Rec.2020). A transposed multiply, a
        // wrong primaries constant, or an inversion bug all diverge from these
        // numbers (a neutral-only test would not — grey is preserved for any
        // invertible row-normalised matrix).
        let xyz_to_cam = [
            [0.6722, -0.0635, -0.0963],
            [-0.4287, 1.2460, 0.2028],
            [-0.0908, 0.2162, 0.5668],
            [0.0, 0.0, 0.0],
        ];
        let expected = [
            [1.15376201, -0.17528879, 0.02152677],
            [-0.08583896, 1.45544251, -0.36960355],
            [0.02341822, -0.36338841, 1.33997019],
        ];
        let m = rec2020_from_cam_matrix(xyz_to_cam);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j] - expected[i][j]).abs() < 1e-4,
                    "m[{i}][{j}] = {}, expected {}",
                    m[i][j], expected[i][j]
                );
            }
        }
    }

    #[test]
    fn apply_color_matrix_identity_is_exact_noop() {
        let orig = vec![0.2f32, 0.5, 0.9, 0.3, 1.5, -0.4, 2.0, 1.0];
        let mut px = orig.clone();
        apply_color_matrix(&mut px, IDENTITY3);
        assert_eq!(px, orig);
    }

    #[test]
    fn apply_color_matrix_mixes_channels() {
        // A channel-swapping matrix (R↔B) proves the per-pixel matrix multiply.
        let swap = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let mut px = vec![0.1f32, 0.2, 0.3, 1.0];
        apply_color_matrix(&mut px, swap);
        assert_eq!(px, vec![0.3, 0.2, 0.1, 1.0]);
    }

    #[test]
    fn rec2020_to_srgb_preserves_neutral() {
        // The display-seam matrix must map grey→grey exactly (its rows sum to 1),
        // else a neutral would pick up a colour cast on the way to the screen.
        let mut px = vec![0.5f32, 0.5, 0.5, 1.0];
        apply_color_matrix(&mut px, REC2020_TO_SRGB);
        for (c, v) in px[..3].iter().enumerate() {
            assert!((v - 0.5).abs() < 1e-5, "channel {c} = {v}, expected 0.5");
        }
        assert_eq!(px[3], 1.0);
    }

    #[test]
    fn rec2020_to_srgb_matches_golden_for_rec2020_red() {
        // Non-grey golden for the display-seam constant. The neutral test above
        // passes for ANY row-normalised matrix (even identity or a transpose);
        // this pins the actual values: the Rec.2020 red primary maps to sRGB as
        // ~[1.660, -0.125, -0.018] (out of sRGB gamut, hence >1 and negatives).
        // Expected values verified against an independent exact-arithmetic
        // derivation from the CIE chromaticity coordinates (agreement ~2e-4;
        // the residual is D65 white rounding conventions). A transposed matrix
        // would yield [1.660, -0.588, -0.073] here and fail hard.
        let mut px = vec![1.0f32, 0.0, 0.0, 1.0];
        apply_color_matrix(&mut px, REC2020_TO_SRGB);
        let expected = [1.66036266, -0.12456355, -0.01815661];
        for (c, (v, e)) in px[..3].iter().zip(expected).enumerate() {
            assert!(
                (v - e).abs() < 1e-4,
                "channel {c} = {v}, expected {e}"
            );
        }
        assert_eq!(px[3], 1.0, "alpha untouched");
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

    /// Build an RGGB Bayer mosaic sampling a per-pixel true colour `f(col,row)
    /// -> [r,g,b]` at each site's native colour (0=R, 1=G, 2=B).
    fn rggb_mosaic_from(w: usize, h: usize, f: impl Fn(usize, usize) -> [f32; 3]) -> Vec<f32> {
        let cfa = [[0usize, 1], [1, 2]];
        (0..w * h)
            .map(|idx| {
                let (col, row) = (idx % w, idx / w);
                f(col, row)[cfa[row % 2][col % 2]]
            })
            .collect()
    }

    #[test]
    fn rcd_constant_mosaic_is_neutral() {
        // A flat field is the one input RCD reconstructs exactly: all gradients
        // collapse to eps, every directional estimate is the constant, and the
        // green→red/blue colour differences are zero. Interior must be a flat
        // grey. Frame > one 112px tile so the tile loop and joins are exercised.
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (200usize, 130usize);
        let out = demosaic_rcd(&vec![0.5f32; w * h], w, h, cfa);
        assert_eq!(out.len(), w * h * 4);
        for j in 12..h - 12 {
            for i in 12..w - 12 {
                let p = (j * w + i) * 4;
                for c in 0..3 {
                    assert!(
                        (out[p + c] - 0.5).abs() < 1e-4,
                        "pixel ({i},{j}) ch{c} = {} (not neutral 0.5)",
                        out[p + c]
                    );
                }
            }
        }
    }

    #[test]
    fn rcd_reconstructs_flat_colour_separating_channels() {
        // A flat, non-grey field: RCD must recover each channel's constant at
        // interior pixels (a flat mosaic can't catch a channel left at 0 or a
        // swapped plane — a coloured one can). Larger than one tile.
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (160usize, 120usize);
        let truth = [0.8f32, 0.4, 0.2];
        let mosaic = rggb_mosaic_from(w, h, |_, _| truth);
        let out = demosaic_rcd(&mosaic, w, h, cfa);
        let p = (60 * w + 80) * 4;
        for c in 0..3 {
            assert!(
                (out[p + c] - truth[c]).abs() < 1e-3,
                "ch{c} = {} (want {})",
                out[p + c],
                truth[c]
            );
        }
    }

    #[test]
    fn rcd_gradient_output_is_finite_and_covered() {
        // A smooth diagonal gradient across a multi-tile frame: every interior
        // pixel must be finite (no NaN from a divide-by-zero guard miss) and no
        // interior RGB left all-zero. Also pins that RCD refines a real interior
        // (differs from the PPG base at high-detail sites).
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (150usize, 140usize);
        let mosaic = rggb_mosaic_from(w, h, |c, r| {
            let v = (c + r) as f32 / (w + h) as f32;
            [v, v * 0.9, v * 0.8]
        });
        let out = demosaic_rcd(&mosaic, w, h, cfa);
        assert_eq!(out.len(), w * h * 4);
        for j in 12..h - 12 {
            for i in 12..w - 12 {
                let p = (j * w + i) * 4;
                for c in 0..3 {
                    assert!(out[p + c].is_finite(), "pixel ({i},{j}) ch{c} not finite");
                }
                assert!(
                    out[p] != 0.0 || out[p + 1] != 0.0 || out[p + 2] != 0.0,
                    "pixel ({i},{j}) RGB all-zero"
                );
            }
        }
    }

    #[test]
    fn rcd_small_image_falls_back_to_ppg_base() {
        // Below one border pair (2·RCD_BORDER = 20) there's no tile interior, so
        // RCD returns the PPG base unchanged.
        let cfa = [[0usize, 1], [1, 2]];
        let mosaic = rggb_mosaic_from(16, 16, |c, _| {
            let v = c as f32 / 15.0;
            [v, v, v]
        });
        assert_eq!(
            demosaic_rcd(&mosaic, 16, 16, cfa),
            demosaic_ppg(&mosaic, 16, 16, cfa)
        );
    }

    #[test]
    fn rcd_interior_tile_join_is_covered() {
        // The BORDER (not MARGIN) trim on BOTH sides of a tile only runs for a
        // tile that is neither first nor last on its axis — i.e. ≥ 3 tiles, which
        // needs ≥ 2·BORDER + 2·TILEVALID + 1 = 205 px. Smaller frames leave this
        // interior-join branch unexecuted. 250×250 ⇒ 3×3 tiles; the centre tile
        // trims BORDER on all four sides. Flat grey ⇒ exact neutral there.
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (250usize, 250usize);
        let out = demosaic_rcd(&vec![0.5f32; w * h], w, h, cfa);
        let p = (125 * w + 125) * 4;
        for c in 0..3 {
            assert!(
                (out[p + c] - 0.5).abs() < 1e-4,
                "interior-join ch{c} = {}",
                out[p + c]
            );
        }
    }

    #[test]
    fn rcd_single_tile_rggb_is_neutral() {
        // 50×50 passes the ≥ 2·BORDER guard and yields exactly one tile
        // (num_v == num_h == 1), so every axis takes the MARGIN-on-both-ends
        // branch — a path the 16×16 fallback test never reaches.
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (50usize, 50usize);
        let out = demosaic_rcd(&vec![0.5f32; w * h], w, h, cfa);
        let p = (25 * w + 25) * 4;
        for c in 0..3 {
            assert!(
                (out[p + c] - 0.5).abs() < 1e-4,
                "single-tile ch{c} = {}",
                out[p + c]
            );
        }
    }

    #[test]
    fn rcd_constant_bggr_is_neutral() {
        // Non-RGGB parity: BGGR flips which sites are R/B vs the RGGB tests, so
        // it catches any inversion in `fc_bayer(row,0)&1` gating or the
        // `c = 2 - fc_bayer(..)` chroma-plane assignment. Flat grey ⇒ neutral.
        let cfa = [[2usize, 1], [1, 0]]; // BGGR
        let (w, h) = (200usize, 130usize);
        let out = demosaic_rcd(&vec![0.5f32; w * h], w, h, cfa);
        let p = (65 * w + 100) * 4;
        for c in 0..3 {
            assert!(
                (out[p + c] - 0.5).abs() < 1e-4,
                "BGGR ch{c} = {}",
                out[p + c]
            );
        }
    }

    #[test]
    fn vng_constant_mosaic_is_neutral() {
        // Flat field through the full VNG path (border + lookup + gradient +
        // green-mix finish): interior must be flat neutral. A frame > a couple of
        // prow/pcol periods exercises the ring buffer and the deferred copies.
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (200usize, 130usize);
        let out = demosaic_vng(&vec![0.5f32; w * h], w, h, cfa);
        assert_eq!(out.len(), w * h * 4);
        for j in 6..h - 6 {
            for i in 6..w - 6 {
                let p = (j * w + i) * 4;
                for c in 0..3 {
                    assert!(
                        (out[p + c] - 0.5).abs() < 1e-4,
                        "pixel ({i},{j}) ch{c} = {} (not neutral 0.5)",
                        out[p + c]
                    );
                }
            }
        }
    }

    #[test]
    fn vng_reconstructs_flat_colour_separating_channels() {
        // Flat non-grey field: VNG must recover each channel's constant at an
        // interior pixel (catches a swapped/zeroed plane or a bad G1/G2 re-merge).
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (160usize, 120usize);
        let truth = [0.8f32, 0.4, 0.2];
        let mosaic = rggb_mosaic_from(w, h, |_, _| truth);
        let out = demosaic_vng(&mosaic, w, h, cfa);
        let p = (60 * w + 80) * 4;
        for c in 0..3 {
            assert!(
                (out[p + c] - truth[c]).abs() < 2e-2,
                "ch{c} = {} (want {})",
                out[p + c],
                truth[c]
            );
        }
    }

    #[test]
    fn vng_gradient_output_is_finite_and_covered() {
        // Smooth diagonal gradient: every interior pixel finite (no NaN leaking
        // from the lookup 0/0 or a gradient divide) and no RGB left all-zero.
        let cfa = [[0usize, 1], [1, 2]];
        let (w, h) = (150usize, 140usize);
        let mosaic = rggb_mosaic_from(w, h, |c, r| {
            let v = (c + r) as f32 / (w + h) as f32;
            [v, v * 0.9, v * 0.8]
        });
        let out = demosaic_vng(&mosaic, w, h, cfa);
        assert_eq!(out.len(), w * h * 4);
        for j in 6..h - 6 {
            for i in 6..w - 6 {
                let p = (j * w + i) * 4;
                for c in 0..3 {
                    assert!(out[p + c].is_finite(), "pixel ({i},{j}) ch{c} not finite");
                }
                assert!(
                    out[p] != 0.0 || out[p + 1] != 0.0 || out[p + 2] != 0.0,
                    "pixel ({i},{j}) RGB all-zero"
                );
            }
        }
    }

    #[test]
    fn vng_constant_bggr_is_neutral() {
        // Non-RGGB parity through VNG: BGGR flips G1/G2 site assignment; a flat
        // field must still come out neutral (guards the filters4 green-split and
        // the finish re-merge on the other phase).
        let cfa = [[2usize, 1], [1, 0]]; // BGGR
        let (w, h) = (200usize, 130usize);
        let out = demosaic_vng(&vec![0.5f32; w * h], w, h, cfa);
        let p = (65 * w + 100) * 4;
        for c in 0..3 {
            assert!(
                (out[p + c] - 0.5).abs() < 1e-4,
                "BGGR VNG ch{c} = {}",
                out[p + c]
            );
        }
    }

    // Standard Fuji X-Trans 6×6 CFA (0=R, 1=G, 2=B).
    const XTRANS: [[u8; 6]; 6] = [
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
        [0, 2, 1, 2, 0, 1],
    ];

    #[test]
    fn normalize_xtrans_subtracts_black_and_scales_per_colour() {
        // 6×6 X-Trans, per-colour black/white. A handful of photosites of each
        // colour must pick the right colour's levels via xtrans[row%6][col%6].
        let black = [10.0, 20.0, 30.0, 0.0];
        let white = [1010.0, 1020.0, 1030.0, 1.0]; // range 1000 each
        let data = vec![520u16; 36];
        let out = normalize_xtrans(&data, 6, 6, &XTRANS, black, white);
        // (0,0)=G → (520-20)/1000 = 0.5; (0,2)=R → (520-10)/1000 = 0.51;
        // (0,5)=B → (520-30)/1000 = 0.49.
        assert!((out[0] - 0.5).abs() < 1e-6, "G {}", out[0]);
        assert!((out[2] - 0.51).abs() < 1e-6, "R {}", out[2]);
        assert!((out[5] - 0.49).abs() < 1e-6, "B {}", out[5]);
    }

    #[test]
    fn xtrans_constant_mosaic_is_neutral() {
        // A flat X-Trans field must Markesteijn-demosaic to a flat neutral grey
        // at interior pixels (interpolating a constant gives the constant).
        let (w, h) = (66usize, 66usize);
        let mosaic = vec![0.5f32; w * h];
        let out = demosaic_xtrans(&mosaic, w, h, &XTRANS);
        assert_eq!(out.len(), w * h * 4);
        for j in 24..h - 24 {
            for i in 24..w - 24 {
                let p = (j * w + i) * 4;
                for c in 0..3 {
                    assert!(
                        (out[p + c] - 0.5).abs() < 0.03,
                        "pixel ({i},{j}) ch{c} = {} (not neutral 0.5)",
                        out[p + c]
                    );
                }
            }
        }
    }

    #[test]
    fn xtrans_gradient_reconstructs_all_channels() {
        // A smooth horizontal gradient (value depends only on column) must
        // reconstruct to ~itself in every channel at interior pixels, and leave
        // no interior pixel all-zero (the demosaic actually ran).
        let (w, h) = (66usize, 66usize);
        let mosaic: Vec<f32> = (0..w * h)
            .map(|idx| (idx % w) as f32 / (w as f32 - 1.0))
            .collect();
        let out = demosaic_xtrans(&mosaic, w, h, &XTRANS);

        for j in 28..h - 28 {
            for i in 28..w - 28 {
                let p = (j * w + i) * 4;
                let expect = i as f32 / (w as f32 - 1.0);
                for c in 0..3 {
                    assert!(
                        (out[p + c] - expect).abs() < 0.12,
                        "pixel ({i},{j}) ch{c} = {} (want ~{expect})",
                        out[p + c]
                    );
                }
            }
        }
    }

    #[test]
    fn classify_cfa_detects_bayer() {
        // An RGGB probe is 2×2-periodic ⇒ Bayer with the origin 2×2 snapshot.
        let rggb = [[0usize, 1], [1, 2]];
        let got = classify_cfa(|r, c| rggb[r % 2][c % 2]).unwrap();
        assert_eq!(got, CfaKind::Bayer(rggb));
    }

    #[test]
    fn classify_cfa_detects_xtrans() {
        // A genuine 6×6 X-Trans probe (not 2×2-periodic) ⇒ Xtrans with the table.
        let got = classify_cfa(|r, c| XTRANS[r % 6][c % 6] as usize).unwrap();
        assert_eq!(got, CfaKind::Xtrans(XTRANS));
    }

    #[test]
    fn classify_cfa_rejects_unsupported_period() {
        // A period-5 column pattern is neither 2×2- nor 6×6-periodic ⇒ rejected
        // rather than mis-snapshotted into either layout.
        let got = classify_cfa(|_, c| (c % 5) % 3);
        assert!(matches!(got, Err(Error::Raw(_))), "got {got:?}");
    }

    #[test]
    fn classify_cfa_rejects_bayer_colour_out_of_rgbe_range() {
        // 2×2-periodic but a colour index ≥ 4 would index past the RGBE arrays.
        let got = classify_cfa(|r, c| if r % 2 == 0 && c % 2 == 0 { 4 } else { 1 });
        assert!(matches!(got, Err(Error::Raw(_))), "got {got:?}");
    }

    #[test]
    fn classify_cfa_rejects_xtrans_colour_out_of_rgb_range() {
        // 6×6-periodic (X-Trans-shaped) but a colour index ≥ 3 (no 'E' in X-Trans)
        // would index past the per-colour levels we use.
        let got = classify_cfa(|r, c| {
            if (r % 6, c % 6) == (2, 2) {
                3
            } else {
                XTRANS[r % 6][c % 6] as usize
            }
        });
        assert!(matches!(got, Err(Error::Raw(_))), "got {got:?}");
    }

    #[test]
    fn xtrans_tiny_image_is_zeroed_not_panic() {
        // Below the Markesteijn tile interior we return zeros rather than risk an
        // under-run; real X-Trans frames are always far larger.
        let out = demosaic_xtrans(&vec![0.5f32; 8 * 8], 8, 8, &XTRANS);
        assert_eq!(out, vec![0.0f32; 8 * 8 * 4]);
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
