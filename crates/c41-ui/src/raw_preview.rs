//! Raw-file preview source (Phase 3 milestone-2): decode a camera raw via
//! `c41_core::rawimage`, downscale it in linear light, and hand the darkroom
//! view a **linear scene-referred f32 RGBA** buffer ([`RawPreview`]) it drives
//! through the float preview pipeline (`gdk-pixbuf` can't decode raws).
//!
//! The decode + demosaic + white-balance live in `c41-core`; this module is
//! just the UI-side marshalling (downscale for responsiveness). No 8-bit
//! round-trip: the pipeline runs on the f32 buffer directly (`BaseImage::Linear`)
//! and only sRGB-encodes at display time, so the sigmoid tone-map sees the
//! unclipped >1.0 highlights. The pure helpers are unit-tested; the
//! `decode_raw_preview` glue is exercised by the `raw_preview_stats` example on
//! a real raw. The Bayer demosaicer is selectable ([`decode_raw_preview_with`]).
//!
//! Known limitations:
//! - **Preview-only resolution.** The buffer is downscaled to `PREVIEW_MAX_DIM`
//!   for slider responsiveness, so **export must read the raw fresh and run the
//!   full-res pipeline** (it does — export is driven by the file path, via the C
//!   `darktable-cli`, not this buffer).
//! - **`.dng` is a container**: linear/float/already-demosaiced DNGs are rejected
//!   by the CFA core decoder → `None` (the loader logs and shows nothing).
//! - **`.raw` is ambiguous** (several vendors + unrelated binary dumps); routed
//!   best-effort, failing gracefully to `None`.

/// Longest-side cap for the raw preview buffer; balances demosaic + per-tick
/// `apply_pipeline` cost against on-screen sharpness. The natural seam for a
/// future "fit to widget size" / HiDPI value.
pub const PREVIEW_MAX_DIM: usize = 2048;

/// A decoded, downscaled raw preview in **linear scene-referred RGBA `f32`**
/// (packed, `width*height*4`, values may exceed 1.0). The darkroom view runs the
/// pipeline on this directly (no 8-bit round-trip), so a tone-map stage sees the
/// unclipped highlights.
pub struct RawPreview {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
    /// True if the source sensor is X-Trans (demosaiced with fixed Markesteijn).
    /// The UI uses this to hide the Bayer demosaic-method selector, which is a
    /// no-op for X-Trans files.
    pub is_xtrans: bool,
    /// Cleaned-up camera maker/model from the decoder's table — seeds the
    /// lens-correction module's camera dropdown. Empty when unknown.
    pub clean_make: String,
    pub clean_model: String,
}

/// File extensions we route through the raw decoder rather than gdk-pixbuf.
const RAW_EXTENSIONS: &[&str] = &[
    "orf", "cr2", "cr3", "nef", "arw", "raf", "dng", "rw2", "pef", "srw", "raw",
    "3fr", "iiq", "mrw", "dcr", "kdc", "x3f", "nrw", "sr2",
];

/// True if `path`'s extension is a camera raw format (case-insensitive).
pub fn is_raw_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| RAW_EXTENSIONS.contains(&e.as_str()))
}

/// Box-average downscale a packed RGBA `f32` image by an integer `factor`
/// (averaging in linear light, which is correct). `factor <= 1` returns a copy.
/// Edge pixels that don't fill a whole `factor × factor` block are dropped.
pub fn downscale_rgba(
    rgba: &[f32],
    width: usize,
    height: usize,
    factor: usize,
) -> (usize, usize, Vec<f32>) {
    let f = factor.max(1);
    if f == 1 {
        return (width, height, rgba.to_vec());
    }
    let (ow, oh) = (width / f, height / f);
    let mut out = vec![0.0f32; ow * oh * 4];
    let inv = 1.0 / (f * f) as f32;
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = [0.0f32; 4];
            for by in 0..f {
                for bx in 0..f {
                    let p = ((oy * f + by) * width + (ox * f + bx)) * 4;
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += rgba[p + c];
                    }
                }
            }
            let op = (oy * ow + ox) * 4;
            for (c, a) in acc.iter().enumerate() {
                out[op + c] = a * inv;
            }
        }
    }
    (ow, oh, out)
}

/// Decode a raw file into a linear [`RawPreview`] using the default Bayer
/// demosaicer ([`DemosaicMethod::Rcd`](c41_core::rawimage::DemosaicMethod)) and
/// no highlight reconstruction. See [`decode_raw_preview_with`].
pub fn decode_raw_preview(path: &str, max_dim: usize) -> Option<RawPreview> {
    decode_raw_preview_with(path, max_dim, Default::default(), None)
}

/// Decode a raw file into a linear [`RawPreview`] with an explicit Bayer
/// [`DemosaicMethod`](c41_core::rawimage::DemosaicMethod) and optional
/// highlight reconstruction (`hl`, applied on the white-balanced mosaic
/// *before* demosaicing — darktable's temperature → highlights → demosaic
/// order), downscaled so its longest side is at most `max_dim` (for slider
/// responsiveness). `None` on decode failure or an unsupported raw (e.g. a CFA
/// period the core decoder rejects). Changing the method or the hl options
/// requires re-running this (it re-decodes the full raw), unlike the pipeline
/// sliders which reuse the downscaled buffer.
pub fn decode_raw_preview_with(
    path: &str,
    max_dim: usize,
    method: c41_core::rawimage::DemosaicMethod,
    hl: Option<c41_core::iop::highlights::HlOpts>,
) -> Option<RawPreview> {
    let img = c41_core::rawimage::load(path).ok()?;
    let is_xtrans = img.xtrans.is_some();
    let clean_make = img.clean_make.clone();
    let clean_model = img.clean_model.clone();
    let (w, h, rgba) = img.to_linear_rgba_with(method, hl);
    if w == 0 || h == 0 {
        return None;
    }
    let longest = w.max(h);
    let factor = longest.div_ceil(max_dim.max(1)).max(1);
    let (w2, h2, small) = downscale_rgba(&rgba, w, h, factor);
    Some(RawPreview {
        width: w2,
        height: h2,
        pixels: small,
        is_xtrans,
        clean_make,
        clean_model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_raw_path_detects_extensions_case_insensitively() {
        assert!(is_raw_path("/a/b/P3080001.ORF"));
        assert!(is_raw_path("shot.cr2"));
        assert!(is_raw_path("x.DnG"));
        assert!(!is_raw_path("photo.jpg"));
        assert!(!is_raw_path("noext"));
    }

    #[test]
    fn downscale_averages_blocks_in_linear() {
        // 2x2 RGBA, factor 2 ⇒ 1 pixel = mean of the four.
        let rgba = vec![
            0.0, 0.0, 0.0, 1.0, 0.2, 0.4, 0.6, 1.0, // row 0
            0.4, 0.4, 0.4, 1.0, 0.6, 0.8, 1.0, 1.0, // row 1
        ];
        let (w, h, out) = downscale_rgba(&rgba, 2, 2, 2);
        assert_eq!((w, h), (1, 1));
        assert!((out[0] - 0.3).abs() < 1e-6); // (0+0.2+0.4+0.6)/4
        assert!((out[1] - 0.4).abs() < 1e-6); // (0+0.4+0.4+0.8)/4
        assert!((out[2] - 0.5).abs() < 1e-6); // (0+0.6+0.4+1.0)/4
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn downscale_factor_one_is_copy() {
        let rgba = vec![0.1, 0.2, 0.3, 1.0];
        let (w, h, out) = downscale_rgba(&rgba, 1, 1, 1);
        assert_eq!((w, h), (1, 1));
        assert_eq!(out, rgba);
    }

    /// Env-gated raw end-to-end (see `scripts/make-raw-corpus.sh`), CI-hermetic:
    /// skips unless `C41_ICC_CORPUS` is set, exactly like the ICC corpus test in
    /// dialogs — the raw fixture lives in that same generated corpus directory.
    ///
    /// Closes the G2a2 raw hole with a real file: before this there was no raw
    /// anywhere the test suite could decode, and the raw branch's ICC **exclusion**
    /// was asserted only by reading code, not by running it. The four claims:
    /// routing, "nothing to lift", camera-matrix profiling, and a finite preview.
    #[test]
    fn synthetic_raw_corpus_decodes_without_the_icc_seam() {
        let Ok(dir) = std::env::var("C41_ICC_CORPUS") else {
            eprintln!("skipping raw corpus check: C41_ICC_CORPUS not set");
            return;
        };
        let dng = std::path::Path::new(&dir).join("synthetic.dng");
        let path = dng.display().to_string();

        // 1. Routing: `spawn_decode` branches on `is_raw_path` BEFORE the ICC
        // scan, so a raw never reaches `extract_embedded`/`input_profile`.
        assert!(is_raw_path(&path), "the fixture must route as raw");

        // 2. Even if it were scanned, the container carries no ICC to lift —
        // a raw's input profile is the camera matrix, and the G2a2 seam has
        // nothing to mislabel. A TIFF-based DNG must classify as `Absent`
        // (NOT `UnsupportedFormat`): tolerating that would mean a regression
        // where the extractor refuses TIFF wholesale could pass unnoticed —
        // which is exactly the classification the routing exists to stay
        // ahead of.
        let embedded = c41_core::icc::extract_embedded_file(&dng)
            .unwrap_or(c41_core::icc::Embedded::UnsupportedFormat);
        assert!(
            matches!(embedded, c41_core::icc::Embedded::Absent),
            "a DNG must lift nothing (Absent), got an {:?}",
            embedded
        );

        // 3. Core decode end-to-end from FILE bytes: dims, a finite RGGB
        // mosaic, and a non-identity camera matrix — the raw's input profile,
        // derived from the ColorMatrix tag the fixture carries.
        let img = c41_core::rawimage::load(&dng).unwrap_or_else(|e| panic!("rawimage::load: {e:?}"));
        assert_eq!((img.width, img.height), (24, 20), "fixture dims changed");
        assert_eq!(img.cfa, [[0, 1], [1, 2]], "fixture must stay RGGB");
        assert_eq!(img.wb[1], 1.0, "fixture white balance must be finite");
        let identity = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(
            img.cam_to_working != identity
                && img.cam_to_working.iter().flatten().all(|v| v.is_finite()),
            "raws must input-profile via the camera matrix, got {:?}",
            img.cam_to_working
        );
        assert!(
            img.mosaic.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "mosaic must be normalized finite photosites"
        );

        // 4. The full preview path (demosaic + downscale) runs on the file.
        let pv = decode_raw_preview_with(&path, 2048, Default::default(), None)
            .unwrap_or_else(|| panic!("raw preview decode failed for {path}"));
        assert!(pv.width <= 2048 && pv.height <= 2048);
        assert!(
            pv.pixels.iter().all(|v| v.is_finite()),
            "raw preview produced non-finite pixels"
        );
    }

    #[test]
    fn downscale_drops_partial_edge_blocks() {
        // 3x3 at factor 2 ⇒ one full 2x2 block ⇒ 1x1; the partial right/bottom
        // edge is dropped (documented behaviour).
        let rgba = vec![0.0f32; 3 * 3 * 4];
        let (w, h, out) = downscale_rgba(&rgba, 3, 3, 2);
        assert_eq!((w, h), (1, 1));
        assert_eq!(out.len(), 4);
    }
}
