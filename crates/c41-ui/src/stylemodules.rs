//! The per-module taxonomy behind partial styles (m4-149).
//!
//! darktable stores a style as one row per IOP, so a style can carry a subset
//! of modules and applying one merges those modules over the target's edit.
//! Our edits are a single [`PreviewParams`] blob (see `persist::STYLES_TABLE_DDL`
//! for why), so "which modules the style carries" is expressed as a list of
//! module-group names in `c41_styles.modules`; this module owns what such a
//! name means: which `PreviewParams` fields belong to it.
//!
//! Coupling, stated plainly — three places must know every field:
//! [`crate::preview::PreviewParams`]'s encode/decode, `history::describe_change`'s
//! group comparisons, and [`copy_module_group`] here. Adding a field without
//! extending all three compiles fine; the drift guards are
//! `params_encode_decode_roundtrips`, the exhaustive-destructure test in
//! `history.rs`, and `merging_every_group_equals_a_wholesale_copy` here.
//!
//! One deliberate divergence from `describe_change`: that function skips
//! `basicadj_hlcomprthresh` because auto-exposure is its only writer and it
//! never differs between history snapshots. A style copy is not change
//! detection — it must move every field the module owns, so the Basic
//! adjustments group includes it.

use crate::preview::PreviewParams;

/// Every module group a partial style can name, in pipeline / panel order.
/// Byte-identical to the labels `history::describe_change` produces, so a
/// history entry and a style row name the same module the same way.
pub const MODULE_GROUPS: &[&str] = &[
    "Exposure",
    "Velvia",
    "Split-toning",
    "Monochrome",
    "Sigmoid",
    "Sharpen",
    "Vibrance",
    "Color contrast",
    "Invert",
    "White balance",
    "Colorize",
    "Color correction",
    "Color zones",
    "Levels",
    "Vignetting",
    "Lowlight vision",
    "Graduated density",
    "Contrast brightness saturation",
    "Basic adjustments",
    "Lowpass",
    "Shadows/Highlights",
    "Primaries",
    "Negadoctor",
    "Tone equalizer",
    "Color balance RGB",
    "Filmic RGB",
    "Highlight reconstruction",
    "Denoise (profiled)",
    "Bloom",
    "Tone curve",
    "RGB curve",
    "Base curve",
    "Lens correction",
];

/// Copy every parameter of one module group from `src` into `target`,
/// leaving all other fields of `target` alone. Returns false if `group`
/// names no known module — the caller decides what an unknown name means.
pub fn copy_module_group(target: &mut PreviewParams, src: &PreviewParams, group: &str) -> bool {
    match group {
        "Exposure" => {
            target.exposure_on = src.exposure_on;
            target.black = src.black;
            target.ev = src.ev;
            true
        }
        "Velvia" => {
            target.velvia_on = src.velvia_on;
            target.velvia_strength = src.velvia_strength;
            target.velvia_bias = src.velvia_bias;
            true
        }
        "Split-toning" => {
            target.split_on = src.split_on;
            target.split_shadow_hue = src.split_shadow_hue;
            target.split_shadow_sat = src.split_shadow_sat;
            target.split_highlight_hue = src.split_highlight_hue;
            target.split_highlight_sat = src.split_highlight_sat;
            target.split_balance = src.split_balance;
            target.split_compress = src.split_compress;
            true
        }
        "Monochrome" => {
            target.mono_on = src.mono_on;
            target.mono_r = src.mono_r;
            target.mono_g = src.mono_g;
            target.mono_b = src.mono_b;
            true
        }
        "Sigmoid" => {
            target.sigmoid_on = src.sigmoid_on;
            target.sigmoid_contrast = src.sigmoid_contrast;
            target.sigmoid_skew = src.sigmoid_skew;
            true
        }
        "Sharpen" => {
            target.sharpen_on = src.sharpen_on;
            target.sharpen_radius = src.sharpen_radius;
            target.sharpen_amount = src.sharpen_amount;
            target.sharpen_threshold = src.sharpen_threshold;
            true
        }
        "Vibrance" => {
            target.vibrance_on = src.vibrance_on;
            target.vibrance_amount = src.vibrance_amount;
            true
        }
        "Color contrast" => {
            target.color_contrast_on = src.color_contrast_on;
            target.color_contrast_a_steepness = src.color_contrast_a_steepness;
            target.color_contrast_b_steepness = src.color_contrast_b_steepness;
            true
        }
        "Invert" => {
            target.invert_on = src.invert_on;
            target.invert_r = src.invert_r;
            target.invert_g = src.invert_g;
            target.invert_b = src.invert_b;
            true
        }
        "White balance" => {
            target.temperature_on = src.temperature_on;
            target.temperature_r = src.temperature_r;
            target.temperature_g = src.temperature_g;
            target.temperature_b = src.temperature_b;
            true
        }
        "Colorize" => {
            target.colorize_on = src.colorize_on;
            target.colorize_hue = src.colorize_hue;
            target.colorize_sat = src.colorize_sat;
            target.colorize_lightness = src.colorize_lightness;
            target.colorize_lightness_mix = src.colorize_lightness_mix;
            true
        }
        "Color correction" => {
            target.color_correction_on = src.color_correction_on;
            target.color_correction_loa = src.color_correction_loa;
            target.color_correction_hia = src.color_correction_hia;
            target.color_correction_lob = src.color_correction_lob;
            target.color_correction_hib = src.color_correction_hib;
            target.color_correction_saturation = src.color_correction_saturation;
            true
        }
        "Color zones" => {
            target.colorzones_on = src.colorzones_on;
            target.colorzones_strength = src.colorzones_strength;
            target.colorzones_channel = src.colorzones_channel;
            target.colorzones_mode = src.colorzones_mode;
            target.colorzones_num_nodes = src.colorzones_num_nodes;
            target.colorzones_curve_type = src.colorzones_curve_type;
            target.colorzones_curve_x = src.colorzones_curve_x;
            target.colorzones_curve_y = src.colorzones_curve_y;
            true
        }
        "Levels" => {
            target.levels_on = src.levels_on;
            target.levels_black = src.levels_black;
            target.levels_grey = src.levels_grey;
            target.levels_white = src.levels_white;
            true
        }
        "Vignetting" => {
            target.vignette_on = src.vignette_on;
            target.vignette_scale = src.vignette_scale;
            target.vignette_falloff = src.vignette_falloff;
            target.vignette_brightness = src.vignette_brightness;
            target.vignette_saturation = src.vignette_saturation;
            target.vignette_center_x = src.vignette_center_x;
            target.vignette_center_y = src.vignette_center_y;
            target.vignette_shape = src.vignette_shape;
            true
        }
        "Lowlight vision" => {
            target.lowlight_on = src.lowlight_on;
            target.lowlight_blueness = src.lowlight_blueness;
            target.lowlight_transition = src.lowlight_transition;
            true
        }
        "Graduated density" => {
            target.gradnd_on = src.gradnd_on;
            target.gradnd_density = src.gradnd_density;
            target.gradnd_hardness = src.gradnd_hardness;
            target.gradnd_rotation = src.gradnd_rotation;
            target.gradnd_offset = src.gradnd_offset;
            target.gradnd_hue = src.gradnd_hue;
            target.gradnd_saturation = src.gradnd_saturation;
            true
        }
        "Contrast brightness saturation" => {
            target.colisa_on = src.colisa_on;
            target.colisa_contrast = src.colisa_contrast;
            target.colisa_brightness = src.colisa_brightness;
            target.colisa_saturation = src.colisa_saturation;
            true
        }
        "Basic adjustments" => {
            target.basicadj_on = src.basicadj_on;
            target.basicadj_black_point = src.basicadj_black_point;
            target.basicadj_exposure = src.basicadj_exposure;
            target.basicadj_hlcompr = src.basicadj_hlcompr;
            // Included here but excluded from describe_change — see the module
            // comment. A copy moves everything the module owns.
            target.basicadj_hlcomprthresh = src.basicadj_hlcomprthresh;
            target.basicadj_contrast = src.basicadj_contrast;
            target.basicadj_preserve_colors = src.basicadj_preserve_colors;
            target.basicadj_middle_grey = src.basicadj_middle_grey;
            target.basicadj_brightness = src.basicadj_brightness;
            target.basicadj_saturation = src.basicadj_saturation;
            target.basicadj_vibrance = src.basicadj_vibrance;
            true
        }
        "Lowpass" => {
            target.lowpass_on = src.lowpass_on;
            target.lowpass_radius = src.lowpass_radius;
            target.lowpass_contrast = src.lowpass_contrast;
            target.lowpass_brightness = src.lowpass_brightness;
            target.lowpass_saturation = src.lowpass_saturation;
            true
        }
        "Shadows/Highlights" => {
            target.shadhi_on = src.shadhi_on;
            target.shadhi_shadows = src.shadhi_shadows;
            target.shadhi_highlights = src.shadhi_highlights;
            target.shadhi_whitepoint = src.shadhi_whitepoint;
            target.shadhi_radius = src.shadhi_radius;
            target.shadhi_compress = src.shadhi_compress;
            target.shadhi_shadows_ccorrect = src.shadhi_shadows_ccorrect;
            target.shadhi_highlights_ccorrect = src.shadhi_highlights_ccorrect;
            true
        }
        "Primaries" => {
            target.primaries_on = src.primaries_on;
            target.primaries_achromatic_tint_hue = src.primaries_achromatic_tint_hue;
            target.primaries_achromatic_tint_purity = src.primaries_achromatic_tint_purity;
            target.primaries_red_hue = src.primaries_red_hue;
            target.primaries_red_purity = src.primaries_red_purity;
            target.primaries_green_hue = src.primaries_green_hue;
            target.primaries_green_purity = src.primaries_green_purity;
            target.primaries_blue_hue = src.primaries_blue_hue;
            target.primaries_blue_purity = src.primaries_blue_purity;
            true
        }
        "Negadoctor" => {
            target.negadoctor_on = src.negadoctor_on;
            target.negadoctor_film_stock = src.negadoctor_film_stock;
            target.negadoctor_dmin_r = src.negadoctor_dmin_r;
            target.negadoctor_dmin_g = src.negadoctor_dmin_g;
            target.negadoctor_dmin_b = src.negadoctor_dmin_b;
            target.negadoctor_wb_high_r = src.negadoctor_wb_high_r;
            target.negadoctor_wb_high_g = src.negadoctor_wb_high_g;
            target.negadoctor_wb_high_b = src.negadoctor_wb_high_b;
            target.negadoctor_wb_low_r = src.negadoctor_wb_low_r;
            target.negadoctor_wb_low_g = src.negadoctor_wb_low_g;
            target.negadoctor_wb_low_b = src.negadoctor_wb_low_b;
            target.negadoctor_d_max = src.negadoctor_d_max;
            target.negadoctor_offset = src.negadoctor_offset;
            target.negadoctor_black = src.negadoctor_black;
            target.negadoctor_gamma = src.negadoctor_gamma;
            target.negadoctor_soft_clip = src.negadoctor_soft_clip;
            target.negadoctor_exposure = src.negadoctor_exposure;
            true
        }
        "Tone equalizer" => {
            target.toneeq_on = src.toneeq_on;
            target.toneeq_noise = src.toneeq_noise;
            target.toneeq_ultra_deep_blacks = src.toneeq_ultra_deep_blacks;
            target.toneeq_deep_blacks = src.toneeq_deep_blacks;
            target.toneeq_blacks = src.toneeq_blacks;
            target.toneeq_shadows = src.toneeq_shadows;
            target.toneeq_midtones = src.toneeq_midtones;
            target.toneeq_highlights = src.toneeq_highlights;
            target.toneeq_whites = src.toneeq_whites;
            target.toneeq_speculars = src.toneeq_speculars;
            true
        }
        "Color balance RGB" => {
            target.cb_on = src.cb_on;
            target.cb_shadows_y = src.cb_shadows_y;
            target.cb_shadows_c = src.cb_shadows_c;
            target.cb_shadows_h = src.cb_shadows_h;
            target.cb_midtones_y = src.cb_midtones_y;
            target.cb_midtones_c = src.cb_midtones_c;
            target.cb_midtones_h = src.cb_midtones_h;
            target.cb_highlights_y = src.cb_highlights_y;
            target.cb_highlights_c = src.cb_highlights_c;
            target.cb_highlights_h = src.cb_highlights_h;
            target.cb_global_y = src.cb_global_y;
            target.cb_global_c = src.cb_global_c;
            target.cb_global_h = src.cb_global_h;
            target.cb_shadows_weight = src.cb_shadows_weight;
            target.cb_white_fulcrum = src.cb_white_fulcrum;
            target.cb_highlights_weight = src.cb_highlights_weight;
            target.cb_chroma_shadows = src.cb_chroma_shadows;
            target.cb_chroma_highlights = src.cb_chroma_highlights;
            target.cb_chroma_global = src.cb_chroma_global;
            target.cb_chroma_midtones = src.cb_chroma_midtones;
            target.cb_saturation_global = src.cb_saturation_global;
            target.cb_saturation_highlights = src.cb_saturation_highlights;
            target.cb_saturation_midtones = src.cb_saturation_midtones;
            target.cb_saturation_shadows = src.cb_saturation_shadows;
            target.cb_hue_angle = src.cb_hue_angle;
            target.cb_brilliance_global = src.cb_brilliance_global;
            target.cb_brilliance_highlights = src.cb_brilliance_highlights;
            target.cb_brilliance_midtones = src.cb_brilliance_midtones;
            target.cb_brilliance_shadows = src.cb_brilliance_shadows;
            target.cb_mask_grey_fulcrum = src.cb_mask_grey_fulcrum;
            target.cb_vibrance = src.cb_vibrance;
            target.cb_grey_fulcrum = src.cb_grey_fulcrum;
            target.cb_contrast = src.cb_contrast;
            target.cb_formula = src.cb_formula;
            true
        }
        "Filmic RGB" => {
            target.filmic_on = src.filmic_on;
            target.filmic_black_point_source = src.filmic_black_point_source;
            target.filmic_white_point_source = src.filmic_white_point_source;
            target.filmic_output_power = src.filmic_output_power;
            target.filmic_latitude = src.filmic_latitude;
            target.filmic_contrast = src.filmic_contrast;
            target.filmic_balance = src.filmic_balance;
            target.filmic_saturation = src.filmic_saturation;
            true
        }
        "Highlight reconstruction" => {
            target.hl_on = src.hl_on;
            target.hl_opposed = src.hl_opposed;
            target.hl_clip = src.hl_clip;
            true
        }
        "Denoise (profiled)" => {
            target.dn_on = src.dn_on;
            target.dn_mode_y0u0v0 = src.dn_mode_y0u0v0;
            target.dn_strength = src.dn_strength;
            target.dn_shadows = src.dn_shadows;
            target.dn_bias = src.dn_bias;
            true
        }
        "Bloom" => {
            target.bl_on = src.bl_on;
            target.bl_size = src.bl_size;
            target.bl_threshold = src.bl_threshold;
            target.bl_strength = src.bl_strength;
            true
        }
        "Tone curve" => {
            target.tc_on = src.tc_on;
            target.tc_type = src.tc_type;
            target.tc_autoscale = src.tc_autoscale;
            target.tc_unbound = src.tc_unbound;
            target.tc_preserve = src.tc_preserve;
            target.tc_nnodes = src.tc_nnodes;
            target.tc_nodes_l = src.tc_nodes_l;
            true
        }
        "RGB curve" => {
            target.rc_on = src.rc_on;
            target.rc_type_r = src.rc_type_r;
            target.rc_type_g = src.rc_type_g;
            target.rc_type_b = src.rc_type_b;
            target.rc_autoscale = src.rc_autoscale;
            target.rc_preserve = src.rc_preserve;
            target.rc_nnodes_r = src.rc_nnodes_r;
            target.rc_nnodes_g = src.rc_nnodes_g;
            target.rc_nnodes_b = src.rc_nnodes_b;
            target.rc_nodes_r = src.rc_nodes_r;
            target.rc_nodes_g = src.rc_nodes_g;
            target.rc_nodes_b = src.rc_nodes_b;
            true
        }
        "Base curve" => {
            target.bc_on = src.bc_on;
            target.bc_type = src.bc_type;
            target.bc_preserve = src.bc_preserve;
            target.bc_nnodes = src.bc_nnodes;
            target.bc_exposure_fusion = src.bc_exposure_fusion;
            target.bc_exposure_stops = src.bc_exposure_stops;
            target.bc_exposure_bias = src.bc_exposure_bias;
            target.bc_nodes = src.bc_nodes;
            true
        }
        "Lens correction" => {
            target.lens_on = src.lens_on;
            target.lens_inverse = src.lens_inverse;
            target.lens_modify_flags = src.lens_modify_flags;
            target.lens_scale = src.lens_scale;
            target.lens_focal = src.lens_focal;
            target.lens_aperture = src.lens_aperture;
            target.lens_distance = src.lens_distance;
            target.lens_target_geom = src.lens_target_geom;
            true
        }
        _ => false,
    }
}

/// Merge the named groups of `overlay` onto `base`: start from `base` and copy
/// each listed group from `overlay`. Names not in [`MODULE_GROUPS`] are ignored,
/// so a style written by a newer build loads everywhere (it just carries less).
///
/// Applying a whole-edit style does NOT go through here — `None` modules means
/// replace wholesale with `overlay` untouched, which is what styles did before
/// partial ones existed (`persist::apply_style_to` branches on that).
pub fn merge_modules(base: &PreviewParams, overlay: &PreviewParams, groups: &[&str]) -> PreviewParams {
    let mut merged = *base;
    for group in groups {
        copy_module_group(&mut merged, overlay, group);
    }
    merged
}

/// The params an image takes when `style` is applied onto its current edit
/// `current`. Both apply surfaces call this — the lighttable's
/// [`crate::persist::apply_style_to`] and the darkroom Styles section — so
/// they cannot drift apart: a whole-edit style (`modules == None`) replaces
/// outright (`current` is irrelevant to that arm); a partial style merges only
/// its listed groups over what is already there. Unknown group names are
/// skipped (see [`merge_modules`]).
///
/// Note the whole-edit arm really is replace even though merging all 33 groups
/// would compute the same value — [`MODULE_GROUPS`] partitions every field —
/// because "replace" is the intent and the direct copy says so.
pub fn apply_style(current: &PreviewParams, style: &crate::persist::Style) -> PreviewParams {
    match &style.modules {
        None => style.params,
        Some(groups) => {
            let names: Vec<&str> = groups.iter().map(String::as_str).collect();
            merge_modules(current, &style.params, &names)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_groups_are_unique_and_complete_in_pipeline_order() {
        assert_eq!(MODULE_GROUPS.len(), 33);
        let mut sorted = MODULE_GROUPS.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "duplicate group name");
        assert_eq!(MODULE_GROUPS[0], "Exposure", "pipeline order starts at exposure");
        assert_eq!(
            MODULE_GROUPS[MODULE_GROUPS.len() - 1],
            "Lens correction",
            "pipeline order ends at lens correction"
        );
    }

    #[test]
    fn copying_one_group_moves_only_that_group() {
        // Overlay changes two groups' fields; copying Velvia must leave the
        // Exposure fields at the target's values.
        let mut target = PreviewParams { ev: -1.5, ..PreviewParams::default() };
        let overlay = PreviewParams {
            ev: 9.0,
            velvia_on: true,
            velvia_strength: 42.0,
            ..PreviewParams::default()
        };
        assert!(copy_module_group(&mut target, &overlay, "Velvia"));
        assert!(target.velvia_on);
        assert_eq!(target.velvia_strength, 42.0);
        assert_eq!(target.ev, -1.5, "another group's field must not move");
        assert_eq!(target.black, PreviewParams::default().black);
    }

    #[test]
    fn unknown_group_is_reported_and_ignored_by_merge() {
        let mut target = PreviewParams { ev: -1.5, ..PreviewParams::default() };
        assert!(!copy_module_group(&mut target, &PreviewParams::default(), "Not A Module"));
        assert_eq!(target.ev, -1.5);

        // Same for merge: an unknown token degrades to "carries nothing", it
        // does not panic or fall back to a wholesale copy.
        let merged = merge_modules(
            &PreviewParams { ev: -1.5, ..PreviewParams::default() },
            &PreviewParams { ev: 9.0, ..PreviewParams::default() },
            &["Not A Module"],
        );
        assert_eq!(merged.ev, -1.5);
    }

    #[test]
    fn merging_every_group_equals_a_wholesale_copy() {
        // The drift guard for this file: if a field exists but belongs to no
        // group, merging ALL groups cannot reproduce a wholesale copy and this
        // fails. `populated_params` names every field explicitly, so adding a
        // PreviewParams field breaks THIS test's compile (missing-field error)
        // until both the fixture and a group arm are extended — same discipline
        // as params_encode_decode_roundtrips.
        let populated = crate::preview::fully_populated_params();
        let base = PreviewParams::default();
        let merged = merge_modules(&base, &populated, MODULE_GROUPS);
        assert_eq!(merged, populated, "every field must belong to exactly one listed group");
    }

    #[test]
    fn basic_adjustments_group_moves_the_non_user_control_too() {
        // Pins the deliberate divergence from describe_change: hlcomprthresh
        // never differs between history snapshots, but a style copy must still
        // move it or a saved Basic adjustments style would silently drop it.
        let overlay = PreviewParams {
            basicadj_hlcomprthresh: 3.5,
            ..PreviewParams::default()
        };
        let mut target = PreviewParams::default();
        assert!(copy_module_group(&mut target, &overlay, "Basic adjustments"));
        assert_eq!(target.basicadj_hlcomprthresh, 3.5);
    }

    #[test]
    fn merge_moves_exactly_the_listed_groups() {
        // Populated base + default overlay: the listed groups take the
        // overlay's (default) values, and unlisted groups keep base's — spot-
        // checked across distant modules; full coverage is the wholesale-copy
        // guard above.
        let base = crate::preview::fully_populated_params();
        let merged = merge_modules(&base, &PreviewParams::default(), &["Velvia", "Levels"]);
        assert_eq!(merged.velvia_strength, PreviewParams::default().velvia_strength);
        assert_eq!(merged.levels_grey, PreviewParams::default().levels_grey);
        assert_eq!(merged.ev, base.ev, "Exposure is not listed");
        assert_eq!(merged.cb_contrast, base.cb_contrast, "Color balance RGB is not listed");
        assert_eq!(merged.lens_focal, base.lens_focal, "Lens correction is not listed");
    }

    fn style_with(params: PreviewParams, modules: Option<Vec<String>>) -> crate::persist::Style {
        crate::persist::Style {
            name: "test".into(),
            description: String::new(),
            params,
            modules,
        }
    }

    #[test]
    fn applying_a_whole_style_replaces_the_current_edit() {
        let current = crate::preview::fully_populated_params();
        let out = apply_style(&current, &style_with(PreviewParams::default(), None));
        assert_eq!(out, PreviewParams::default());
    }

    #[test]
    fn applying_a_partial_style_merges_over_the_current_edit() {
        let current = crate::preview::fully_populated_params();
        let out = apply_style(
            &current,
            &style_with(PreviewParams::default(), Some(vec!["Velvia".into(), "Garbage".into()])),
        );
        assert_eq!(out.velvia_strength, PreviewParams::default().velvia_strength);
        assert_eq!(out.ev, current.ev, "Exposure is not in the style");
        assert_eq!(out.lens_focal, current.lens_focal, "Lens correction is not in the style");
    }
}
