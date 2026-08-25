//! Pure export model for the darkroom export panel (Phase 3 m4-7a): the output
//! format, JPEG quality, and resize box, plus the `darktable-cli` argument
//! construction and the fit-within resize math. Kept free of GTK so the path /
//! resize / argv logic is unit-testable headless (the established display-free
//! discipline); the GTK export panel (m4-7b) builds its widgets over this.
//!
//! `darkroom-cli` is the C `darktable-cli` (a symlink in the app image). Its
//! argument parser (`src/cli/main.c`) accepts `--out-ext`, `--style`,
//! `--apply-custom-presets`, `--width`, `--height`, `--upscale`, and a trailing
//! `--core …` that forwards to the darktable core. There is **no `--quality`
//! flag** — output quality is a core config key — so quality is passed via
//! `--core --conf plugins/imageio/format/<module>/quality=<n>` where `<module>` is
//! the imageio module name (`jpeg`/`tiff`/`png`), *not* the file extension (a
//! positional `--quality 95` is silently swallowed as the output-path argument,
//! and an extension-keyed conf is read by nothing: the two bugs this builder
//! replaces).

/// The export output format. `out_ext` is the file extension passed to
/// `darktable-cli --out-ext`; `module_name` is the imageio format module the CLI
/// resolves that extension to — and the two differ (`jpg`→`jpeg`, `tif`→`tiff`),
/// which is why the core config keys must use [`ExportFormat::module_name`], not
/// the extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    Jpeg,
    Tiff,
    Png,
}

impl ExportFormat {
    /// The file extension passed to `--out-ext` (what the output file is named).
    pub fn out_ext(self) -> &'static str {
        match self {
            ExportFormat::Jpeg => "jpg",
            ExportFormat::Tiff => "tif",
            ExportFormat::Png => "png",
        }
    }

    /// The imageio format module name the CLI resolves `out_ext` to. This is the
    /// namespace for the format's `plugins/imageio/format/<module>/…` conf keys
    /// (e.g. quality, bpp) and is distinct from the file extension: `jpg`→`jpeg`,
    /// `tif`→`tiff`. Using the extension instead silently no-ops the conf.
    pub fn module_name(self) -> &'static str {
        match self {
            ExportFormat::Jpeg => "jpeg",
            ExportFormat::Tiff => "tiff",
            ExportFormat::Png => "png",
        }
    }

    /// Human label for the format combo row.
    pub fn label(self) -> &'static str {
        match self {
            ExportFormat::Jpeg => "JPEG (sRGB)",
            // No bit-depth claim: the Rust export path writes 8-bit for JPEG
            // (inherent) and 16-bit for PNG/TIFF (via the `image` crate); keep the
            // label neutral in case the CLI path or a future path differs.
            ExportFormat::Tiff => "TIFF",
            ExportFormat::Png => "PNG",
        }
    }

    /// Map a combo-row index to a format (out-of-range → PNG, matching the
    /// combo's last entry). Keep in sync with [`ExportFormat::ALL`].
    pub fn from_index(i: u32) -> ExportFormat {
        match i {
            0 => ExportFormat::Jpeg,
            1 => ExportFormat::Tiff,
            _ => ExportFormat::Png,
        }
    }

    /// All formats in combo order (drives the format row's `StringList`).
    pub const ALL: [ExportFormat; 3] = [ExportFormat::Jpeg, ExportFormat::Tiff, ExportFormat::Png];

    /// Whether a JPEG quality setting applies (the quality conf is only honoured
    /// by the JPEG imageio module; TIFF/PNG ignore it).
    pub fn uses_quality(self) -> bool {
        matches!(self, ExportFormat::Jpeg)
    }
}

/// A maximum bounding box for the exported image. A `0` on either axis means that
/// axis is unconstrained (the `darktable-cli --width/--height` convention). The
/// image is scaled to fit inside the box, preserving aspect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resize {
    pub max_w: u32,
    pub max_h: u32,
    /// Permit enlarging an image smaller than the box (off by default, like
    /// darktable's "allow upscaling").
    pub allow_upscale: bool,
}

/// The full export configuration the panel collects.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportSettings {
    pub format: ExportFormat,
    /// JPEG quality 1..=100 (clamped when emitted; ignored for non-JPEG).
    pub quality: u32,
    /// `None` exports at original size; `Some` fits within the box.
    pub resize: Option<Resize>,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self { format: ExportFormat::Jpeg, quality: 95, resize: None }
    }
}

/// Fit `(orig_w, orig_h)` inside the resize box, preserving aspect. A zero max
/// axis is unconstrained; without `allow_upscale` the result never exceeds the
/// original. Returns the output pixel dimensions — for a UI preview of the export
/// size; `darktable-cli` performs the actual resampling from the same box. Zero
/// or degenerate input is returned unchanged.
pub fn fit_within(orig_w: u32, orig_h: u32, r: &Resize) -> (u32, u32) {
    if orig_w == 0 || orig_h == 0 {
        return (orig_w, orig_h);
    }
    // Per-axis scale to reach the box; an unconstrained (0) axis imposes no limit.
    let sw = if r.max_w == 0 { f64::INFINITY } else { r.max_w as f64 / orig_w as f64 };
    let sh = if r.max_h == 0 { f64::INFINITY } else { r.max_h as f64 / orig_h as f64 };
    let mut scale = sw.min(sh);
    if !scale.is_finite() {
        return (orig_w, orig_h); // both axes unconstrained → original size
    }
    if scale > 1.0 && !r.allow_upscale {
        scale = 1.0; // don't enlarge
    }
    let w = ((orig_w as f64 * scale).round() as u32).max(1);
    let h = ((orig_h as f64 * scale).round() as u32).max(1);
    (w, h)
}

/// The default output-path template: an `exports/` subfolder beside the source,
/// keeping the original base name. The output *extension* is **not** part of the
/// template — `--out-ext` appends it (the CLI strips a redundant trailing `.ext`
/// and the format module re-applies the real one), so a bare stem is correct.
pub const DEFAULT_OUTPUT_TEMPLATE: &str = "$(FILE_FOLDER)/exports/$(FILE_NAME)";

/// Expand the output-path `template` for one `input_path` into a concrete,
/// extension-less destination path to hand `darktable-cli` (which then appends the
/// `--out-ext` extension). We expand only the subset of darktable variables we own
/// — `$(FILE_FOLDER)` (input's parent dir, no trailing slash), `$(FILE_NAME)`
/// (input stem, no extension), and `$(SEQUENCE)` (4-digit zero-padded batch index)
/// — so the panel can preview the resolved path and batch exports don't collide.
/// Any other `$(…)` token is left **verbatim** so the CLI's own `dt_variables`
/// expansion (e.g. `$(YEAR)`, `$(EXIF_…)`) still applies at export time.
pub fn expand_output_template(template: &str, input_path: &str, sequence: u32) -> String {
    let p = std::path::Path::new(input_path);
    // Resolve the source folder. A bare/relative input has an empty parent — fall
    // back to "." (CWD-relative) rather than letting `$(FILE_FOLDER)/…` root the
    // dest at the filesystem root (`/exports/…`), a permissions trap especially in
    // the container. The filesystem-root case keeps "/".
    let folder = match p.parent().and_then(|d| d.to_str()) {
        None | Some("") => ".".to_string(),
        // The filesystem root "/" trims to "" so the template's own separator
        // (`$(FILE_FOLDER)/…`) yields a single leading slash, not "//".
        Some(d) => d.trim_end_matches('/').to_string(),
    };
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    template
        .replace("$(FILE_FOLDER)", &folder)
        .replace("$(FILE_NAME)", stem)
        .replace("$(SEQUENCE)", &format!("{sequence:04}"))
}

/// Make a `template` safe for a batch of `batch_len` images: if more than one
/// image is being exported and the template doesn't already disambiguate via
/// `$(SEQUENCE)`, append a `_$(SEQUENCE)` suffix so two sources that map to the
/// same stem can't silently overwrite each other — e.g. a RAW+JPEG pair
/// `IMG.CR2`/`IMG.JPG` in one folder, both expanding to `exports/IMG`. The CLI's
/// own overwrite-vs-rename behaviour is governed by the disk storage's
/// `onsave_action`, which `darktable-cli` never sets, so uniqueness is ours to
/// guarantee. Single-image exports keep clean, suffix-free names.
pub fn batch_output_template(template: &str, batch_len: usize) -> String {
    if batch_len > 1 && !template.contains("$(SEQUENCE)") {
        format!("{template}_$(SEQUENCE)")
    } else {
        template.to_string()
    }
}

/// Build the `darktable-cli` argument vector (everything after argv[0]) to export
/// `input` into `output_dest` (a directory or a `$(…)` filename pattern that the
/// CLI expands). `--style none --apply-custom-presets false` gives a neutral
/// render; resize maps to `--width/--height/--upscale`; JPEG quality goes through
/// the trailing `--core --conf` (the CLI has no `--quality` flag). `--core` must
/// stay last — the CLI forwards everything after it to the darktable core.
pub fn cli_args(input: &str, output_dest: &str, s: &ExportSettings) -> Vec<String> {
    let mut a = vec![
        input.to_string(),
        output_dest.to_string(),
        "--out-ext".to_string(),
        s.format.out_ext().to_string(),
        "--style".to_string(),
        "none".to_string(),
        "--apply-custom-presets".to_string(),
        "false".to_string(),
    ];
    if let Some(r) = &s.resize {
        a.push("--width".to_string());
        a.push(r.max_w.to_string());
        a.push("--height".to_string());
        a.push(r.max_h.to_string());
        a.push("--upscale".to_string());
        a.push(if r.allow_upscale { "true" } else { "false" }.to_string());
    }
    // Per-format core config keys, namespaced on the imageio *module* name (not the
    // file extension): JPEG honours a quality key, TIFF a bit-depth key (so the
    // "16-bit" label is truthful), PNG needs none. Keyed on out_ext they'd be read
    // by nothing and silently no-op.
    let confs: Vec<String> = match s.format {
        ExportFormat::Jpeg => vec![format!(
            "plugins/imageio/format/{}/quality={}",
            s.format.module_name(),
            s.quality.clamp(1, 100)
        )],
        ExportFormat::Tiff => vec![format!(
            "plugins/imageio/format/{}/bpp=16",
            s.format.module_name()
        )],
        ExportFormat::Png => vec![],
    };
    // Emit a single trailing `--core` block: it forwards everything after it to the
    // darktable core parser, so it must come last and appear at most once.
    if !confs.is_empty() {
        a.push("--core".to_string());
        for c in confs {
            a.push("--conf".to_string());
            a.push(c);
        }
    }
    a
}

/// The c41-ui edit to bake into a Rust-native export so the output matches
/// the preview: the Bayer demosaic method, the geometry (straighten + crop),
/// the colour-pipeline params, and the resolved lens-correction gear
/// ([`crate::preview::LensGear`] — `None` when the module is off or nothing is
/// selected). `Clone` but no longer `Copy` (the gear is an `Arc` of
/// pointer-backed database objects); still plain Rust data, safe to move to the
/// export thread. Passed as `Some` for the single-image darkroom export;
/// `None` for the lighttable multi-export, which resolves each image's edit
/// from the catalog instead.
#[derive(Clone)]
pub struct ExportEdit {
    pub method: c41_core::rawimage::DemosaicMethod,
    pub geometry: c41_core::geometry::Geometry,
    pub params: crate::preview::PreviewParams,
    /// Resolved `(camera, lens)` for `params.lens_on`, shared with the preview's
    /// cache when exporting from the darkroom view. `None` omits the stage —
    /// exactly what the preview does without gear.
    pub lens: Option<std::sync::Arc<crate::preview::LensGear>>,
}

/// Render a decoded [`RawImage`] to a packed 8-bit **sRGB RGB** buffer
/// (`width*height*3`) through the c41-ui pipeline — the full-resolution
/// twin of the darkroom preview, so a Rust-native export matches what the user
/// edited. Composition mirrors the preview exactly: highlight reconstruction +
/// demosaic + white-balance + camera→Rec.2020
/// ([`RawImage::to_linear_rgba_with`] with `params.hl_opts()`) → geometry
/// ([`Geometry::apply`], straighten then crop) → the colour pipeline + Rec.2020→
/// sRGB display seam + sRGB OETF ([`crate::preview::render_linear_to_srgb8`]).
/// Returns the geometry-adjusted `(width, height)` and the RGB bytes.
pub fn render_export_rgb8(
    img: &c41_core::rawimage::RawImage,
    method: c41_core::rawimage::DemosaicMethod,
    geometry: c41_core::geometry::Geometry,
    params: &crate::preview::PreviewParams,
) -> (usize, usize, Vec<u8>) {
    render_export_rgb8_gear(img, method, geometry, params, None)
}

/// [`Self::render_export_rgb8`] with resolved lens-correction gear — the form
/// the export loop calls with the edit's cached gear so a corrected preview
/// exports corrected. See [`crate::preview::LensGear`].
pub fn render_export_rgb8_gear(
    img: &c41_core::rawimage::RawImage,
    method: c41_core::rawimage::DemosaicMethod,
    geometry: c41_core::geometry::Geometry,
    params: &crate::preview::PreviewParams,
    lens_gear: Option<&crate::preview::LensGear>,
) -> (usize, usize, Vec<u8>) {
    let (w, h, linear) = img.to_linear_rgba_with(method, params.hl_opts());
    // Lens pre-pass on the FULL frame, before crop/straighten — darktable runs
    // lens at iop_order 13, before the geometry modules (m4-131).
    let linear = crate::preview::apply_lens_prepass(&linear, w, h, params, lens_gear);
    let (gw, gh, geom_linear) = geometry.apply(&linear, w, h);
    let rgb = crate::preview::render_linear_to_srgb8_gear(&geom_linear, gw, gh, params, lens_gear);
    (gw, gh, rgb)
}

/// 16-bit twin of [`render_export_rgb8`] for high-bit-depth export (PNG/TIFF):
/// identical pipeline, quantised to 16 bits. Returns the geometry-adjusted
/// `(width, height)` and packed RGB `u16`.
pub fn render_export_rgb16(
    img: &c41_core::rawimage::RawImage,
    method: c41_core::rawimage::DemosaicMethod,
    geometry: c41_core::geometry::Geometry,
    params: &crate::preview::PreviewParams,
) -> (usize, usize, Vec<u16>) {
    render_export_rgb16_gear(img, method, geometry, params, None)
}

/// 16-bit twin of [`render_export_rgb8_gear`]: identical pipeline (including
/// the lens-correction gear), quantised to 16 bits.
pub fn render_export_rgb16_gear(
    img: &c41_core::rawimage::RawImage,
    method: c41_core::rawimage::DemosaicMethod,
    geometry: c41_core::geometry::Geometry,
    params: &crate::preview::PreviewParams,
    lens_gear: Option<&crate::preview::LensGear>,
) -> (usize, usize, Vec<u16>) {
    let (w, h, linear) = img.to_linear_rgba_with(method, params.hl_opts());
    // Lens pre-pass on the FULL frame, before crop/straighten — darktable runs
    // lens at iop_order 13, before the geometry modules (m4-131).
    let linear = crate::preview::apply_lens_prepass(&linear, w, h, params, lens_gear);
    let (gw, gh, geom_linear) = geometry.apply(&linear, w, h);
    let rgb = crate::preview::render_linear_to_srgb16_gear(&geom_linear, gw, gh, params, lens_gear);
    (gw, gh, rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small synthetic Bayer raw (diagonal-gradient mosaic) for exercising the
    /// Rust export render path without a real raw file.
    fn synthetic_raw(w: usize, h: usize) -> c41_core::rawimage::RawImage {
        let mosaic: Vec<f32> = (0..w * h)
            .map(|i| ((i % w + i / w) as f32) / ((w + h) as f32))
            .collect();
        c41_core::rawimage::RawImage {
            width: w,
            height: h,
            cfa: [[0, 1], [1, 2]], // RGGB
            xtrans: None,
            wb: [1.0, 1.0, 1.0, 1.0],
            orientation: (false, false, false),
            cam_to_working: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            clean_make: String::new(),
            clean_model: String::new(),
            mosaic,
        }
    }

    #[test]
    fn render_export_rgb8_identity_keeps_source_dims() {
        use c41_core::geometry::Geometry;
        use c41_core::rawimage::DemosaicMethod;
        let img = synthetic_raw(40, 30);
        let (w, h, rgb) = render_export_rgb8(
            &img,
            DemosaicMethod::Rcd,
            Geometry::default(),
            &crate::preview::PreviewParams::default(),
        );
        assert_eq!((w, h), (40, 30));
        assert_eq!(rgb.len(), 40 * 30 * 3);
        assert!(rgb.iter().any(|&b| b != 0), "produced a non-blank image");
    }

    #[test]
    fn render_export_rgb8_applies_crop_dims() {
        use c41_core::geometry::{Crop, Geometry};
        use c41_core::rawimage::DemosaicMethod;
        let img = synthetic_raw(40, 30);
        // Horizontal crop 0.25..0.75 of 40 px = a 20-wide, full-height result.
        let geom = Geometry {
            crop: Crop { left: 0.25, top: 0.0, right: 0.75, bottom: 1.0 },
            angle: 0.0,
        };
        let (w, h, rgb) = render_export_rgb8(
            &img,
            DemosaicMethod::Rcd,
            geom,
            &crate::preview::PreviewParams::default(),
        );
        assert_eq!((w, h), (20, 30));
        assert_eq!(rgb.len(), 20 * 30 * 3);
    }

    #[test]
    fn lens_correction_runs_precrop_and_commutes_with_crop() {
        // m4-131: the warp + vignette run on the FULL frame before the geometry
        // pass (darktable's iop_order-13 placement). Because the correction is
        // purely coordinate-driven, cropping afterwards must commute exactly:
        // a cropped export equals the same region of the uncropped export,
        // bit-for-bit. Under the pre-m4-131 placement (stage inside the
        // pipeline, i.e. after crop) the cropped frame was corrected around its
        // own centre and this equality failed.
        use c41_core::geometry::{Crop, Geometry};
        use c41_core::rawimage::DemosaicMethod;
        let Some(gear) = c41_core::iop::lens::resolve(
            "Canon",
            "Canon EOS 5D Mark II",
            "Canon",
            "Canon EF 50mm f/1.4 USM",
        ) else {
            eprintln!("skip: lensfun database unavailable");
            return;
        };
        let img = synthetic_raw(40, 30);
        let mut p = crate::preview::PreviewParams::default();
        p.lens_on = true; // defaults carry MODFLAG_ALL, focal 50 mm, f/3.5

        let full = render_export_rgb16_gear(
            &img,
            DemosaicMethod::Rcd,
            Geometry::default(),
            &p,
            Some(&gear),
        );
        let cropped = render_export_rgb16_gear(
            &img,
            DemosaicMethod::Rcd,
            Geometry {
                crop: Crop { left: 0.25, top: 0.0, right: 0.75, bottom: 1.0 },
                angle: 0.0,
            },
            &p,
            Some(&gear),
        );
        assert_eq!((cropped.0, cropped.1), (20, 30));

        // Precondition: the gear's calibration actually alters pixels here —
        // otherwise the commutation below would hold trivially even for a
        // wrongly-placed correction.
        let plain = render_export_rgb16_gear(
            &img,
            DemosaicMethod::Rcd,
            Geometry::default(),
            &p,
            None,
        );
        assert!(
            full.2.iter().zip(plain.2.iter()).any(|(a, b)| a != b),
            "correction was a no-op at these params; the test proves nothing",
        );

        // The commutation itself, exact.
        for y in 0..30usize {
            for x in 0..20usize {
                for c in 0..3usize {
                    let a = cropped.2[(y * 20 + x) * 3 + c];
                    let b = full.2[(y * 40 + (x + 10)) * 3 + c];
                    assert_eq!(
                        a, b,
                        "crop/lens mismatch at ({x},{y}) ch{c}: cropped {a} vs full {b}",
                    );
                }
            }
        }
    }

    #[test]
    fn render_export_rgb16_dims_and_matches_8bit_scaled() {
        use c41_core::geometry::Geometry;
        use c41_core::rawimage::DemosaicMethod;
        let img = synthetic_raw(40, 30);
        let params = crate::preview::PreviewParams::default();
        let (w, h, rgb16) =
            render_export_rgb16(&img, DemosaicMethod::Rcd, Geometry::default(), &params);
        assert_eq!((w, h), (40, 30));
        assert_eq!(rgb16.len(), 40 * 30 * 3);
        // The 16-bit encode is the same sRGB values at higher precision: the top
        // 8 bits of each u16 must equal the 8-bit render (within ±1 from rounding).
        let (_, _, rgb8) =
            render_export_rgb8(&img, DemosaicMethod::Rcd, Geometry::default(), &params);
        for (i, (&hi, &lo)) in rgb16.iter().zip(rgb8.iter()).enumerate() {
            // 65535 = 255·257, so the top byte of the 16-bit encode is ~0.4% above
            // the 8-bit one; with independent rounding they agree within ±2.
            let hi8 = (hi >> 8) as i32;
            assert!((hi8 - lo as i32).abs() <= 2, "sample {i}: 16→8 {hi8} vs 8-bit {lo}");
        }
    }

    #[test]
    fn format_index_and_ext_roundtrip() {
        assert_eq!(ExportFormat::from_index(0), ExportFormat::Jpeg);
        assert_eq!(ExportFormat::from_index(1), ExportFormat::Tiff);
        assert_eq!(ExportFormat::from_index(2), ExportFormat::Png);
        assert_eq!(ExportFormat::from_index(99), ExportFormat::Png); // clamp to last
        assert_eq!(ExportFormat::Jpeg.out_ext(), "jpg");
        assert_eq!(ExportFormat::Tiff.out_ext(), "tif");
        assert_eq!(ExportFormat::Png.out_ext(), "png");
        assert!(ExportFormat::Jpeg.uses_quality());
        assert!(!ExportFormat::Png.uses_quality());
        // Module names drive the conf namespace and differ from the extension.
        assert_eq!(ExportFormat::Jpeg.module_name(), "jpeg");
        assert_eq!(ExportFormat::Tiff.module_name(), "tiff");
        assert_eq!(ExportFormat::Png.module_name(), "png");
        // ALL stays in index order so the combo dispatch matches from_index, and
        // covers every variant (drift guard if a format is added to the enum).
        assert_eq!(ExportFormat::ALL.len(), 3);
        for (i, f) in ExportFormat::ALL.iter().enumerate() {
            assert_eq!(ExportFormat::from_index(i as u32), *f);
        }
    }

    #[test]
    fn fit_within_downscales_preserving_aspect() {
        // 4000x3000 into 1000x1000 box → scale 0.25 (height-bound) → 1000x750.
        let r = Resize { max_w: 1000, max_h: 1000, allow_upscale: false };
        assert_eq!(fit_within(4000, 3000, &r), (1000, 750));
        // portrait 3000x4000 → width-bound → 750x1000.
        assert_eq!(fit_within(3000, 4000, &r), (750, 1000));
    }

    #[test]
    fn fit_within_respects_upscale_flag_and_unconstrained_axis() {
        // smaller than box: no upscaling by default → unchanged.
        let no_up = Resize { max_w: 4000, max_h: 4000, allow_upscale: false };
        assert_eq!(fit_within(800, 600, &no_up), (800, 600));
        // with upscaling: scale up to the box (4000/800 = 5x → 4000x3000).
        let up = Resize { max_w: 4000, max_h: 4000, allow_upscale: true };
        assert_eq!(fit_within(800, 600, &up), (4000, 3000));
        // only width constrained (height 0): 2000x1000 into width 1000 → 1000x500.
        let w_only = Resize { max_w: 1000, max_h: 0, allow_upscale: false };
        assert_eq!(fit_within(2000, 1000, &w_only), (1000, 500));
        // both axes unconstrained → original size.
        let none = Resize { max_w: 0, max_h: 0, allow_upscale: true };
        assert_eq!(fit_within(2000, 1000, &none), (2000, 1000));
        // degenerate input returned unchanged.
        assert_eq!(fit_within(0, 0, &no_up), (0, 0));
    }

    #[test]
    fn cli_args_jpeg_sets_quality_via_core_conf() {
        let s = ExportSettings { format: ExportFormat::Jpeg, quality: 90, resize: None };
        let a = cli_args("/in/a.raw", "/out", &s);
        assert_eq!(a[0], "/in/a.raw");
        assert_eq!(a[1], "/out");
        // out-ext + neutral render flags present.
        assert!(a.windows(2).any(|w| w == ["--out-ext", "jpg"]));
        assert!(a.windows(2).any(|w| w == ["--style", "none"]));
        assert!(a.windows(2).any(|w| w == ["--apply-custom-presets", "false"]));
        // quality via --core --conf (NOT a bare --quality flag), keyed on the
        // *module* name "jpeg" (not the extension "jpg") or the core ignores it.
        assert!(!a.iter().any(|x| x == "--quality"));
        assert_eq!(a[a.len() - 3], "--core");
        assert_eq!(a[a.len() - 2], "--conf");
        assert_eq!(a[a.len() - 1], "plugins/imageio/format/jpeg/quality=90");
        // `--core` forwards the rest to the core parser, so it appears at most once.
        assert_eq!(a.iter().filter(|x| *x == "--core").count(), 1);
    }

    #[test]
    fn cli_args_tiff_sets_bpp_16_via_module_name() {
        let s = ExportSettings { format: ExportFormat::Tiff, quality: 95, resize: None };
        let a = cli_args("/in/a.raw", "/out", &s);
        assert!(a.windows(2).any(|w| w == ["--out-ext", "tif"]));
        // 16-bit label is truthful: bpp conf keyed on module name "tiff".
        assert_eq!(a[a.len() - 1], "plugins/imageio/format/tiff/bpp=16");
        assert_eq!(a.iter().filter(|x| *x == "--core").count(), 1);
    }

    #[test]
    fn cli_args_png_omits_quality_and_no_resize_omits_size() {
        let s = ExportSettings { format: ExportFormat::Png, quality: 90, resize: None };
        let a = cli_args("/in/a.raw", "/out", &s);
        assert!(a.windows(2).any(|w| w == ["--out-ext", "png"]));
        // PNG needs no core config → no --core/--conf appended.
        assert!(!a.iter().any(|x| x == "--core"));
        assert!(!a.iter().any(|x| x == "--width"));
    }

    #[test]
    fn cli_args_resize_emits_width_height_upscale_before_core() {
        let s = ExportSettings {
            format: ExportFormat::Jpeg,
            quality: 100,
            resize: Some(Resize { max_w: 2048, max_h: 1536, allow_upscale: true }),
        };
        let a = cli_args("/in/a.raw", "/out", &s);
        assert!(a.windows(2).any(|w| w == ["--width", "2048"]));
        assert!(a.windows(2).any(|w| w == ["--height", "1536"]));
        assert!(a.windows(2).any(|w| w == ["--upscale", "true"]));
        // resize args precede the trailing --core block.
        let core_at = a.iter().position(|x| x == "--core").unwrap();
        let width_at = a.iter().position(|x| x == "--width").unwrap();
        assert!(width_at < core_at);
    }

    #[test]
    fn expand_template_default_puts_export_subfolder_beside_source() {
        // Default template → exports/ beside the source, stem kept, no extension.
        let out = expand_output_template(DEFAULT_OUTPUT_TEMPLATE, "/photos/raw/IMG_1234.CR2", 0);
        assert_eq!(out, "/photos/raw/exports/IMG_1234");
    }

    #[test]
    fn expand_template_sequence_and_unknown_tokens() {
        // SEQUENCE is zero-padded to 4 digits; unknown $(…) tokens pass through
        // verbatim for the CLI's own dt_variables expansion.
        let out = expand_output_template(
            "$(FILE_FOLDER)/$(FILE_NAME)_$(SEQUENCE)_$(YEAR)", "/a/b/pic.raw", 7);
        assert_eq!(out, "/a/b/pic_0007_$(YEAR)");
    }

    #[test]
    fn expand_template_handles_missing_parent_and_extension() {
        // A bare filename has no parent dir → FILE_FOLDER falls back to "." so the
        // dest stays CWD-relative instead of rooting at "/". Stem is the whole name
        // when there's no extension.
        assert_eq!(expand_output_template("$(FILE_FOLDER)/$(FILE_NAME)", "pic", 0), "./pic");
        assert_eq!(expand_output_template("$(FILE_NAME)", "/a/b/noext", 0), "noext");
        // A file at the filesystem root keeps "/".
        assert_eq!(expand_output_template("$(FILE_FOLDER)/$(FILE_NAME)", "/pic.jpg", 0), "/pic");
    }

    #[test]
    fn batch_template_adds_sequence_to_prevent_overwrite() {
        // Single image: template untouched (clean, suffix-free names).
        assert_eq!(batch_output_template(DEFAULT_OUTPUT_TEMPLATE, 1), DEFAULT_OUTPUT_TEMPLATE);
        // Batch without $(SEQUENCE): suffix appended so a RAW+JPEG pair in one
        // folder (IMG.CR2 / IMG.JPG → both exports/IMG) can't collide.
        let t = batch_output_template(DEFAULT_OUTPUT_TEMPLATE, 2);
        assert_eq!(t, "$(FILE_FOLDER)/exports/$(FILE_NAME)_$(SEQUENCE)");
        let a = expand_output_template(&t, "/f/IMG.CR2", 1);
        let b = expand_output_template(&t, "/f/IMG.JPG", 2);
        assert_eq!(a, "/f/exports/IMG_0001");
        assert_eq!(b, "/f/exports/IMG_0002");
        assert_ne!(a, b);
        // Batch already carrying $(SEQUENCE): left as-is (no double suffix).
        let custom = "$(FILE_NAME)_$(SEQUENCE)";
        assert_eq!(batch_output_template(custom, 5), custom);
    }

    #[test]
    fn cli_args_clamps_out_of_range_quality() {
        let s = ExportSettings { format: ExportFormat::Jpeg, quality: 999, resize: None };
        let a = cli_args("/in", "/out", &s);
        assert_eq!(a.last().unwrap(), "plugins/imageio/format/jpeg/quality=100");
    }
}
