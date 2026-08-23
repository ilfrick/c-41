//! Canonical IOP module catalogue for the darkroom module panel, grouped by
//! darktable's standard module groups (base / tone / color / correct / effect).
//!
//! For now this is the UI-side source of truth (display labels + default
//! enabled state). A later milestone wires `default_on`/enable state through to
//! `c41-core` + the history stack; the labels here name the IOPs already
//! ported in `c41-core::iop`.

/// One IOP module entry in the panel.
pub struct ModuleInfo {
    /// Human-readable label shown in the panel.
    pub label: &'static str,
    /// Whether the module is on by default in a fresh edit.
    pub default_on: bool,
}

/// A darktable module group (a collapsible section in the panel).
pub struct ModuleGroup {
    pub name: &'static str,
    pub modules: &'static [ModuleInfo],
}

const fn m(label: &'static str, default_on: bool) -> ModuleInfo {
    ModuleInfo { label, default_on }
}

/// The default module catalogue, in pipeline-ish presentation order.
/// Defined as a `static` so the nested slices get `'static` lifetime (a
/// function body does not promote these const-fn-built arrays).
static CATALOG: &[ModuleGroup] = &[
        ModuleGroup {
            name: "Base",
            modules: &[
                m("Raw black/white point", true),
                m("Invert", false),
                m("Negadoctor", false),
                m("White balance", true),
                m("Highlight reconstruction", false),
                m("Demosaic", true),
                m("Exposure", true),
                m("Lens correction", false),
                m("Rotate & perspective", false),
                m("Crop", false),
                m("Orientation", true),
            ],
        },
        ModuleGroup {
            name: "Tone",
            modules: &[
                m("Sigmoid", false),
                m("Filmic RGB", true),
                m("Tone equalizer", false),
                m("Levels", false),
                m("Contrast brightness saturation", false),
                m("Basic adjustments", false),
                m("Shadows/Highlights", false),
                m("RGB curve", false),
                m("Base curve", false),
            ],
        },
        ModuleGroup {
            name: "Color",
            modules: &[
                m("Color calibration", true),
                m("Color balance RGB", false),
                m("Input color profile", true),
                m("Output color profile", true),
                m("Color zones", false),
                m("Velvia", false),
                m("Vibrance", false),
                m("Colorize", false),
                m("Color correction", false),
                m("Color contrast", false),
                m("Primaries", false),
            ],
        },
        ModuleGroup {
            name: "Correct",
            modules: &[
                m("Denoise (profiled)", false),
                m("Sharpen", false),
                m("Hot pixels", false),
                m("Chromatic aberrations", false),
                m("Defringe", false),
                m("Retouch", false),
                m("Liquify", false),
            ],
        },
        ModuleGroup {
            name: "Effect",
            modules: &[
                m("Bloom", false),
                m("Grain", false),
                m("Vignetting", false),
                m("Soften", false),
                m("Highpass", false),
                m("Lowpass", false),
                m("Lowlight vision", false),
                m("Monochrome", false),
                m("Split-toning", false),
                m("Graduated density", false),
                m("Framing", false),
                m("Watermark", false),
            ],
        },
];

pub fn module_catalog() -> &'static [ModuleGroup] {
    CATALOG
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_well_formed() {
        let groups = module_catalog();
        assert_eq!(groups.len(), 5);
        // every group has modules and no empty labels
        for g in groups {
            assert!(!g.name.is_empty());
            assert!(!g.modules.is_empty(), "group {} is empty", g.name);
            for mi in g.modules {
                assert!(!mi.label.is_empty());
            }
        }
        // a sane total and at least one default-on module
        let total: usize = groups.iter().map(|g| g.modules.len()).sum();
        assert!(total >= 30, "only {total} modules");
        assert!(groups.iter().flat_map(|g| g.modules).any(|m| m.default_on));
    }
}
