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
        mosaic,
    })
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
}
