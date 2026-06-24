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
            ExportFormat::Tiff => "TIFF 16-bit",
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn cli_args_clamps_out_of_range_quality() {
        let s = ExportSettings { format: ExportFormat::Jpeg, quality: 999, resize: None };
        let a = cli_args("/in", "/out", &s);
        assert_eq!(a.last().unwrap(), "plugins/imageio/format/jpeg/quality=100");
    }
}
