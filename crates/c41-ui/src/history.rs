//! Edit-history stack for the darkroom view (Phase 3 milestone-4): an ordered,
//! navigable list of [`PreviewParams`] snapshots with undo / redo / jump. This
//! is the pure model — the darkroom page records a (debounced) snapshot per
//! settled edit and lets the user step back to any earlier state. Kept free of
//! GTK so the navigation logic is fully unit-testable headless (the established
//! display-free discipline).
//!
//! Semantics mirror darktable's history stack: entry 0 is the seed ("original"),
//! recording a new state discards any redo tail (a fresh edit branches), and
//! identical consecutive states are de-duplicated so a slider that returns to its
//! value doesn't spam the list.

use crate::preview::PreviewParams;

/// One recorded edit state: a human label plus the full params at that point.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryEntry {
    pub label: String,
    pub params: PreviewParams,
}

/// Maximum retained entries. Beyond this the oldest are dropped so a long
/// editing session can't grow the stack without bound (each entry is small, so
/// this is generous — it's a guard, not a typical limit).
pub const HISTORY_CAP: usize = 100;

/// A linear undo/redo history of [`PreviewParams`] snapshots with a cursor.
///
/// Invariant: `entries` is never empty and `cursor < entries.len()` — there is
/// always a valid "current" state (the seed at minimum).
#[derive(Clone, Debug)]
pub struct HistoryStack {
    entries: Vec<HistoryEntry>,
    cursor: usize,
}

impl HistoryStack {
    /// New stack seeded with the initial state, which becomes entry 0 (e.g.
    /// "original"). The cursor starts on it.
    pub fn new(label: impl Into<String>, params: PreviewParams) -> Self {
        Self {
            entries: vec![HistoryEntry { label: label.into(), params }],
            cursor: 0,
        }
    }

    /// Record a new state after the cursor and move the cursor onto it.
    ///
    /// - No-op (returns `false`) if `params` equals the current entry — identical
    ///   consecutive states are de-duplicated.
    /// - Any redo tail (entries after the cursor) is discarded: recording from a
    ///   mid-history position branches a new line of edits, as in darktable.
    /// - Enforces [`HISTORY_CAP`] by dropping the oldest entries.
    pub fn record(&mut self, label: impl Into<String>, params: PreviewParams) -> bool {
        if self.entries[self.cursor].params == params {
            return false;
        }
        // Drop the redo tail, then append the new state.
        self.entries.truncate(self.cursor + 1);
        self.entries.push(HistoryEntry { label: label.into(), params });
        // Bound memory: drop the oldest *edits* past the cap, but keep entry 0
        // (the "Original" seed) pinned so the user can always jump back to the
        // unedited state — as darktable does.
        if self.entries.len() > HISTORY_CAP {
            let overflow = self.entries.len() - HISTORY_CAP;
            self.entries.drain(1..1 + overflow);
        }
        self.cursor = self.entries.len() - 1;
        true
    }

    /// True if there's an earlier state to step back to.
    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    /// True if there's a later state to step forward to.
    pub fn can_redo(&self) -> bool {
        self.cursor + 1 < self.entries.len()
    }

    /// Step the cursor back one and return the now-current params (`None` if
    /// already at the seed).
    pub fn undo(&mut self) -> Option<PreviewParams> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        Some(self.entries[self.cursor].params)
    }

    /// Step the cursor forward one and return the now-current params (`None` if
    /// already at the newest).
    pub fn redo(&mut self) -> Option<PreviewParams> {
        if self.cursor + 1 >= self.entries.len() {
            return None;
        }
        self.cursor += 1;
        Some(self.entries[self.cursor].params)
    }

    /// Move the cursor to `index` and return its params (`None` if out of range).
    pub fn jump_to(&mut self, index: usize) -> Option<PreviewParams> {
        let p = self.entries.get(index)?.params;
        self.cursor = index;
        Some(p)
    }

    /// The params at the cursor (the active edit state).
    pub fn current(&self) -> PreviewParams {
        self.entries[self.cursor].params
    }

    /// The cursor index (0 = seed).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Number of retained entries (always ≥ 1).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Always `false` — the stack holds at least the seed. Present so callers
    /// (and clippy) have the conventional companion to [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// All entries, oldest → newest, for rendering the panel list.
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Serialise the stack to a versioned little-endian blob for persistence:
    /// `[ver u8][cursor u32][count u32]` then per entry
    /// `[label_len u16][label utf8][params blob]`. The params blob is
    /// [`PreviewParams::encode`] (fixed length). Round-trips with [`decode`].
    ///
    /// [`decode`]: Self::decode
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::new();
        v.push(HISTORY_ENCODE_VERSION);
        v.extend_from_slice(&(self.cursor as u32).to_le_bytes());
        v.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            let label = e.label.as_bytes();
            // Labels are short module names; clamp defensively so the u16 length
            // prefix can't truncate-then-mismatch on decode.
            let len = label.len().min(u16::MAX as usize);
            v.extend_from_slice(&(len as u16).to_le_bytes());
            v.extend_from_slice(&label[..len]);
            v.extend_from_slice(&e.params.encode());
        }
        v
    }

    /// Parse a blob from [`encode`]. Returns `None` on any malformation (wrong
    /// version, truncation, a bad params blob, an out-of-range cursor, an empty
    /// or over-cap count) — the caller falls back to a fresh stack.
    ///
    /// [`encode`]: Self::encode
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut p = 0usize;
        let take = |p: &mut usize, n: usize| -> Option<()> {
            (*p + n <= bytes.len()).then(|| *p += n)
        };

        // Accept both the current and the previous history-container version.
        // The container (v5) wraps PreviewParams blobs that carry their own
        // per-entry version byte — PreviewParams::ENCODE_VERSION has drifted
        // from v13 (553 B) to v18 (850 B) as basicadj/shadhi/lowpass/negadoctor/
        // toneequalizer/colorbalancergb layers were added, but the history
        // *container* format is unchanged.
        // Rejecting the prior container version outright would wipe every saved
        // undo/redo stack on first load, so we decode leniently and let
        // PreviewParams::decode default any missing fields via the version byte.
        let version = *bytes.first()?;
        if version != HISTORY_ENCODE_VERSION && version != HISTORY_ENCODE_VERSION - 1 {
            return None;
        }
        p += 1;
        let cursor = read_u32(bytes, &mut p)? as usize;
        let count = read_u32(bytes, &mut p)? as usize;
        // A valid stack always holds ≥ 1 entry and never exceeds the cap; reject
        // anything else rather than allocate from a corrupt length.
        if count == 0 || count > HISTORY_CAP {
            return None;
        }

        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let label_len = read_u16(bytes, &mut p)? as usize;
            let lstart = p;
            take(&mut p, label_len)?;
            let label = std::str::from_utf8(&bytes[lstart..p]).ok()?.to_string();
            let pstart = p;
            // Peek at the params version byte to determine this entry's blob
            // length — older history stacks contain shorter (v13) blobs.
            let params_len = crate::preview::encoded_len_for_version(bytes[pstart])?;
            take(&mut p, params_len)?;
            let params = PreviewParams::decode(&bytes[pstart..p])?;
            entries.push(HistoryEntry { label, params });
        }
        // No trailing garbage, and the cursor must index a real entry.
        if p != bytes.len() || cursor >= entries.len() {
            return None;
        }
        Some(Self { entries, cursor })
    }
}

/// Version byte for [`HistoryStack::encode`]; bump on any layout change.
/// m4-124: the wrapped PreviewParams blob grew (v25, base curve) — old v5
/// history blobs decode fine (decode accepts version-1), but new writes carry 6.
const HISTORY_ENCODE_VERSION: u8 = 6;

fn read_u32(bytes: &[u8], p: &mut usize) -> Option<u32> {
    let end = p.checked_add(4)?;
    let slice = bytes.get(*p..end)?;
    *p = end;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

fn read_u16(bytes: &[u8], p: &mut usize) -> Option<u16> {
    let end = p.checked_add(2)?;
    let slice = bytes.get(*p..end)?;
    *p = end;
    Some(u16::from_le_bytes(slice.try_into().ok()?))
}

/// Name the module whose parameters changed between two states, for the history
/// entry label. Groups are checked in pipeline / panel order and the first that
/// differs wins (a single user gesture touches one module). Falls back to
/// `"Edit"` if nothing recognised differs. Pure, so it's unit-testable without a
/// live widget — the darkroom page calls it when recording a settled edit.
pub fn describe_change(old: &PreviewParams, new: &PreviewParams) -> &'static str {
    let exposure = old.exposure_on != new.exposure_on
        || old.black != new.black
        || old.ev != new.ev;
    if exposure {
        return "Exposure";
    }
    let velvia = old.velvia_on != new.velvia_on
        || old.velvia_strength != new.velvia_strength
        || old.velvia_bias != new.velvia_bias;
    if velvia {
        return "Velvia";
    }
    let split = old.split_on != new.split_on
        || old.split_shadow_hue != new.split_shadow_hue
        || old.split_shadow_sat != new.split_shadow_sat
        || old.split_highlight_hue != new.split_highlight_hue
        || old.split_highlight_sat != new.split_highlight_sat
        || old.split_balance != new.split_balance
        || old.split_compress != new.split_compress;
    if split {
        return "Split-toning";
    }
    let mono = old.mono_on != new.mono_on
        || old.mono_r != new.mono_r
        || old.mono_g != new.mono_g
        || old.mono_b != new.mono_b;
    if mono {
        return "Monochrome";
    }
    let sigmoid = old.sigmoid_on != new.sigmoid_on
        || old.sigmoid_contrast != new.sigmoid_contrast
        || old.sigmoid_skew != new.sigmoid_skew;
    if sigmoid {
        return "Sigmoid";
    }
    let sharpen = old.sharpen_on != new.sharpen_on
        || old.sharpen_radius != new.sharpen_radius
        || old.sharpen_amount != new.sharpen_amount
        || old.sharpen_threshold != new.sharpen_threshold;
    if sharpen {
        return "Sharpen";
    }
    let vibrance = old.vibrance_on != new.vibrance_on
        || old.vibrance_amount != new.vibrance_amount;
    if vibrance {
        return "Vibrance";
    }
    let color_contrast = old.color_contrast_on != new.color_contrast_on
        || old.color_contrast_a_steepness != new.color_contrast_a_steepness
        || old.color_contrast_b_steepness != new.color_contrast_b_steepness;
    if color_contrast {
        return "Color contrast";
    }
    let invert = old.invert_on != new.invert_on
        || old.invert_r != new.invert_r
        || old.invert_g != new.invert_g
        || old.invert_b != new.invert_b;
    if invert {
        return "Invert";
    }
    let temperature = old.temperature_on != new.temperature_on
        || old.temperature_r != new.temperature_r
        || old.temperature_g != new.temperature_g
        || old.temperature_b != new.temperature_b;
    if temperature {
        return "White balance";
    }
    let colorize = old.colorize_on != new.colorize_on
        || old.colorize_hue != new.colorize_hue
        || old.colorize_sat != new.colorize_sat
        || old.colorize_lightness != new.colorize_lightness
        || old.colorize_lightness_mix != new.colorize_lightness_mix;
    if colorize {
        return "Colorize";
    }
    let color_correction = old.color_correction_on != new.color_correction_on
        || old.color_correction_loa != new.color_correction_loa
        || old.color_correction_hia != new.color_correction_hia
        || old.color_correction_lob != new.color_correction_lob
        || old.color_correction_hib != new.color_correction_hib
        || old.color_correction_saturation != new.color_correction_saturation;
    if color_correction {
        return "Color correction";
    }
    let colorzones = old.colorzones_on != new.colorzones_on
        || old.colorzones_strength != new.colorzones_strength
        || old.colorzones_channel != new.colorzones_channel
        || old.colorzones_mode != new.colorzones_mode
        || old.colorzones_num_nodes != new.colorzones_num_nodes
        || old.colorzones_curve_type != new.colorzones_curve_type
        || old.colorzones_curve_x != new.colorzones_curve_x
        || old.colorzones_curve_y != new.colorzones_curve_y;
    if colorzones {
        return "Color zones";
    }
    let levels = old.levels_on != new.levels_on
        || old.levels_black != new.levels_black
        || old.levels_grey != new.levels_grey
        || old.levels_white != new.levels_white;
    if levels {
        return "Levels";
    }
    let vignette = old.vignette_on != new.vignette_on
        || old.vignette_scale != new.vignette_scale
        || old.vignette_falloff != new.vignette_falloff
        || old.vignette_brightness != new.vignette_brightness
        || old.vignette_saturation != new.vignette_saturation
        || old.vignette_center_x != new.vignette_center_x
        || old.vignette_center_y != new.vignette_center_y
        || old.vignette_shape != new.vignette_shape;
    if vignette {
        return "Vignetting";
    }
    let lowlight = old.lowlight_on != new.lowlight_on
        || old.lowlight_blueness != new.lowlight_blueness
        || old.lowlight_transition != new.lowlight_transition;
    if lowlight {
        return "Lowlight vision";
    }
    let gradnd = old.gradnd_on != new.gradnd_on
        || old.gradnd_density != new.gradnd_density
        || old.gradnd_hardness != new.gradnd_hardness
        || old.gradnd_rotation != new.gradnd_rotation
        || old.gradnd_offset != new.gradnd_offset
        || old.gradnd_hue != new.gradnd_hue
        || old.gradnd_saturation != new.gradnd_saturation;
    if gradnd {
        return "Graduated density";
    }
    let colisa = old.colisa_on != new.colisa_on
        || old.colisa_contrast != new.colisa_contrast
        || old.colisa_brightness != new.colisa_brightness
        || old.colisa_saturation != new.colisa_saturation;
    if colisa {
        return "Contrast brightness saturation";
    }
    // hlcomprthresh is not a user control (darktable only sets it via
    // auto-exposure), so it is excluded from change detection — it stays at
    // its 0.0 default and would never meaningfully differ between snapshots.
    let basicadj = old.basicadj_on != new.basicadj_on
        || old.basicadj_black_point != new.basicadj_black_point
        || old.basicadj_exposure != new.basicadj_exposure
        || old.basicadj_hlcompr != new.basicadj_hlcompr
        || old.basicadj_contrast != new.basicadj_contrast
        || old.basicadj_preserve_colors != new.basicadj_preserve_colors
        || old.basicadj_middle_grey != new.basicadj_middle_grey
        || old.basicadj_brightness != new.basicadj_brightness
        || old.basicadj_saturation != new.basicadj_saturation
        || old.basicadj_vibrance != new.basicadj_vibrance;
    if basicadj {
        return "Basic adjustments";
    }
    let lowpass = old.lowpass_on != new.lowpass_on
        || old.lowpass_radius != new.lowpass_radius
        || old.lowpass_contrast != new.lowpass_contrast
        || old.lowpass_brightness != new.lowpass_brightness
        || old.lowpass_saturation != new.lowpass_saturation;
    if lowpass {
        return "Lowpass";
    }
    let shadhi = old.shadhi_on != new.shadhi_on
        || old.shadhi_shadows != new.shadhi_shadows
        || old.shadhi_highlights != new.shadhi_highlights
        || old.shadhi_whitepoint != new.shadhi_whitepoint
        || old.shadhi_radius != new.shadhi_radius
        || old.shadhi_compress != new.shadhi_compress
        || old.shadhi_shadows_ccorrect != new.shadhi_shadows_ccorrect
        || old.shadhi_highlights_ccorrect != new.shadhi_highlights_ccorrect;
    if shadhi {
        return "Shadows/Highlights";
    }
    let primaries = old.primaries_on != new.primaries_on
        || old.primaries_achromatic_tint_hue != new.primaries_achromatic_tint_hue
        || old.primaries_achromatic_tint_purity != new.primaries_achromatic_tint_purity
        || old.primaries_red_hue != new.primaries_red_hue
        || old.primaries_red_purity != new.primaries_red_purity
        || old.primaries_green_hue != new.primaries_green_hue
        || old.primaries_green_purity != new.primaries_green_purity
        || old.primaries_blue_hue != new.primaries_blue_hue
        || old.primaries_blue_purity != new.primaries_blue_purity;
    if primaries {
        return "Primaries";
    }
    let negadoctor = old.negadoctor_on != new.negadoctor_on
        || old.negadoctor_film_stock != new.negadoctor_film_stock
        || old.negadoctor_dmin_r != new.negadoctor_dmin_r
        || old.negadoctor_dmin_g != new.negadoctor_dmin_g
        || old.negadoctor_dmin_b != new.negadoctor_dmin_b
        || old.negadoctor_wb_high_r != new.negadoctor_wb_high_r
        || old.negadoctor_wb_high_g != new.negadoctor_wb_high_g
        || old.negadoctor_wb_high_b != new.negadoctor_wb_high_b
        || old.negadoctor_wb_low_r != new.negadoctor_wb_low_r
        || old.negadoctor_wb_low_g != new.negadoctor_wb_low_g
        || old.negadoctor_wb_low_b != new.negadoctor_wb_low_b
        || old.negadoctor_d_max != new.negadoctor_d_max
        || old.negadoctor_offset != new.negadoctor_offset
        || old.negadoctor_black != new.negadoctor_black
        || old.negadoctor_gamma != new.negadoctor_gamma
        || old.negadoctor_soft_clip != new.negadoctor_soft_clip
        || old.negadoctor_exposure != new.negadoctor_exposure;
    if negadoctor {
        return "Negadoctor";
    }
    let toneeq = old.toneeq_on != new.toneeq_on
        || old.toneeq_noise != new.toneeq_noise
        || old.toneeq_ultra_deep_blacks != new.toneeq_ultra_deep_blacks
        || old.toneeq_deep_blacks != new.toneeq_deep_blacks
        || old.toneeq_blacks != new.toneeq_blacks
        || old.toneeq_shadows != new.toneeq_shadows
        || old.toneeq_midtones != new.toneeq_midtones
        || old.toneeq_highlights != new.toneeq_highlights
        || old.toneeq_whites != new.toneeq_whites
        || old.toneeq_speculars != new.toneeq_speculars;
    if toneeq {
        return "Tone equalizer";
    }
    let cb = old.cb_on != new.cb_on
        || old.cb_shadows_y != new.cb_shadows_y
        || old.cb_shadows_c != new.cb_shadows_c
        || old.cb_shadows_h != new.cb_shadows_h
        || old.cb_midtones_y != new.cb_midtones_y
        || old.cb_midtones_c != new.cb_midtones_c
        || old.cb_midtones_h != new.cb_midtones_h
        || old.cb_highlights_y != new.cb_highlights_y
        || old.cb_highlights_c != new.cb_highlights_c
        || old.cb_highlights_h != new.cb_highlights_h
        || old.cb_global_y != new.cb_global_y
        || old.cb_global_c != new.cb_global_c
        || old.cb_global_h != new.cb_global_h
        || old.cb_shadows_weight != new.cb_shadows_weight
        || old.cb_white_fulcrum != new.cb_white_fulcrum
        || old.cb_highlights_weight != new.cb_highlights_weight
        || old.cb_chroma_shadows != new.cb_chroma_shadows
        || old.cb_chroma_highlights != new.cb_chroma_highlights
        || old.cb_chroma_global != new.cb_chroma_global
        || old.cb_chroma_midtones != new.cb_chroma_midtones
        || old.cb_saturation_global != new.cb_saturation_global
        || old.cb_saturation_highlights != new.cb_saturation_highlights
        || old.cb_saturation_midtones != new.cb_saturation_midtones
        || old.cb_saturation_shadows != new.cb_saturation_shadows
        || old.cb_hue_angle != new.cb_hue_angle
        || old.cb_brilliance_global != new.cb_brilliance_global
        || old.cb_brilliance_highlights != new.cb_brilliance_highlights
        || old.cb_brilliance_midtones != new.cb_brilliance_midtones
        || old.cb_brilliance_shadows != new.cb_brilliance_shadows
        || old.cb_mask_grey_fulcrum != new.cb_mask_grey_fulcrum
        || old.cb_vibrance != new.cb_vibrance
        || old.cb_grey_fulcrum != new.cb_grey_fulcrum
        || old.cb_contrast != new.cb_contrast
        || old.cb_formula != new.cb_formula;
    if cb {
        return "Color balance RGB";
    }
    let filmic = old.filmic_on != new.filmic_on
        || old.filmic_black_point_source != new.filmic_black_point_source
        || old.filmic_white_point_source != new.filmic_white_point_source
        || old.filmic_output_power != new.filmic_output_power
        || old.filmic_latitude != new.filmic_latitude
        || old.filmic_contrast != new.filmic_contrast
        || old.filmic_balance != new.filmic_balance
        || old.filmic_saturation != new.filmic_saturation;
    if filmic {
        return "Filmic RGB";
    }
    // Highlight reconstruction (m4-119): runs pre-demosaic in the raw front
    // end, but the user still edits it as a module, so it needs its own
    // history label like any other.
    let hl = old.hl_on != new.hl_on
        || old.hl_opposed != new.hl_opposed
        || old.hl_clip != new.hl_clip;
    if hl {
        return "Highlight reconstruction";
    }
    let dn = old.dn_on != new.dn_on
        || old.dn_mode_y0u0v0 != new.dn_mode_y0u0v0
        || old.dn_strength != new.dn_strength
        || old.dn_shadows != new.dn_shadows
        || old.dn_bias != new.dn_bias;
    if dn {
        return "Denoise (profiled)";
    }
    let bl = old.bl_on != new.bl_on
        || old.bl_size != new.bl_size
        || old.bl_threshold != new.bl_threshold
        || old.bl_strength != new.bl_strength;
    if bl {
        return "Bloom";
    }
    // Tone curve (m4-122): the L anchors are compared pairwise (the array is a
    // plain [(f32, f32); 20], so `!=` does the right thing).
    let tc = old.tc_on != new.tc_on
        || old.tc_type != new.tc_type
        || old.tc_autoscale != new.tc_autoscale
        || old.tc_unbound != new.tc_unbound
        || old.tc_preserve != new.tc_preserve
        || old.tc_nnodes != new.tc_nnodes
        || old.tc_nodes_l != new.tc_nodes_l;
    if tc {
        return "Tone curve";
    }
    // RGB curve (m4-123): same pairwise-array comparison, now across all three
    // channels' anchors.
    let rc = old.rc_on != new.rc_on
        || old.rc_type_r != new.rc_type_r
        || old.rc_type_g != new.rc_type_g
        || old.rc_type_b != new.rc_type_b
        || old.rc_autoscale != new.rc_autoscale
        || old.rc_preserve != new.rc_preserve
        || old.rc_nnodes_r != new.rc_nnodes_r
        || old.rc_nnodes_g != new.rc_nnodes_g
        || old.rc_nnodes_b != new.rc_nnodes_b
        || old.rc_nodes_r != new.rc_nodes_r
        || old.rc_nodes_g != new.rc_nodes_g
        || old.rc_nodes_b != new.rc_nodes_b;
    if rc {
        return "RGB curve";
    }
    // Base curve (m4-124): single channel, but the exposure-fusion controls
    // (mode/stops/bias) are part of the module's effect, so they belong in the
    // group too.
    let bc = old.bc_on != new.bc_on
        || old.bc_type != new.bc_type
        || old.bc_preserve != new.bc_preserve
        || old.bc_nnodes != new.bc_nnodes
        || old.bc_exposure_fusion != new.bc_exposure_fusion
        || old.bc_exposure_stops != new.bc_exposure_stops
        || old.bc_exposure_bias != new.bc_exposure_bias
        || old.bc_nodes != new.bc_nodes;
    if bc {
        return "Base curve";
    }
    "Edit"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(ev: f32) -> PreviewParams {
        PreviewParams { ev, ..PreviewParams::default() }
    }

    #[test]
    fn new_stack_holds_only_the_seed() {
        let h = HistoryStack::new("original", params(0.0));
        assert_eq!(h.len(), 1);
        assert_eq!(h.cursor(), 0);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.current(), params(0.0));
        assert!(!h.is_empty());
    }

    #[test]
    fn record_appends_and_advances_cursor() {
        let mut h = HistoryStack::new("original", params(0.0));
        assert!(h.record("exposure", params(1.0)));
        assert_eq!(h.len(), 2);
        assert_eq!(h.cursor(), 1);
        assert!(h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.current(), params(1.0));
    }

    #[test]
    fn record_dedups_identical_consecutive_state() {
        let mut h = HistoryStack::new("original", params(0.0));
        h.record("exposure", params(1.0));
        // Same params again ⇒ no new entry.
        assert!(!h.record("exposure", params(1.0)));
        assert_eq!(h.len(), 2);
        assert_eq!(h.cursor(), 1);
    }

    #[test]
    fn undo_then_redo_round_trips() {
        let mut h = HistoryStack::new("original", params(0.0));
        h.record("a", params(1.0));
        h.record("b", params(2.0));
        assert_eq!(h.undo(), Some(params(1.0)));
        assert_eq!(h.cursor(), 1);
        assert_eq!(h.undo(), Some(params(0.0)));
        assert_eq!(h.cursor(), 0);
        assert_eq!(h.undo(), None); // at the seed
        assert_eq!(h.redo(), Some(params(1.0)));
        assert_eq!(h.redo(), Some(params(2.0)));
        assert_eq!(h.redo(), None); // at the newest
    }

    #[test]
    fn recording_after_undo_discards_redo_tail() {
        let mut h = HistoryStack::new("original", params(0.0));
        h.record("a", params(1.0));
        h.record("b", params(2.0));
        h.undo(); // back to a (cursor 1)
        assert!(h.can_redo());
        // A new edit from here branches: the old "b" tail is dropped.
        assert!(h.record("c", params(3.0)));
        assert_eq!(h.len(), 3); // seed, a, c
        assert_eq!(h.cursor(), 2);
        assert!(!h.can_redo());
        assert_eq!(h.current(), params(3.0));
        assert_eq!(h.entries()[2].label, "c");
    }

    #[test]
    fn jump_to_moves_cursor_and_bounds_check() {
        let mut h = HistoryStack::new("original", params(0.0));
        h.record("a", params(1.0));
        h.record("b", params(2.0));
        assert_eq!(h.jump_to(0), Some(params(0.0)));
        assert_eq!(h.cursor(), 0);
        assert_eq!(h.jump_to(2), Some(params(2.0)));
        assert_eq!(h.cursor(), 2);
        assert_eq!(h.jump_to(99), None); // out of range: cursor unchanged
        assert_eq!(h.cursor(), 2);
    }

    #[test]
    fn describe_change_names_the_first_differing_module() {
        let base = PreviewParams::default();
        let d = PreviewParams::default;

        assert_eq!(describe_change(&base, &PreviewParams { ev: 1.0, ..d() }), "Exposure");
        assert_eq!(
            describe_change(&base, &PreviewParams { exposure_on: !base.exposure_on, ..d() }),
            "Exposure"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { velvia_strength: 50.0, ..d() }),
            "Velvia"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { split_balance: 0.7, ..d() }),
            "Split-toning"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { mono_r: 0.9, ..d() }),
            "Monochrome"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { sigmoid_contrast: 2.0, ..d() }),
            "Sigmoid"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { sharpen_amount: 2.0, ..d() }),
            "Sharpen"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { vibrance_amount: 15.0, ..d() }),
            "Vibrance"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { color_contrast_a_steepness: 2.0, ..d() }),
            "Color contrast"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { invert_r: 0.8, ..d() }),
            "Invert"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { temperature_r: 1.5, ..d() }),
            "White balance"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { colorize_hue: 0.3, ..d() }),
            "Colorize"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { color_correction_saturation: 1.5, ..d() }),
            "Color correction"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { colorzones_strength: 50.0, ..d() }),
            "Color zones"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { levels_grey: 40.0, ..d() }),
            "Levels"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { vignette_scale: 60.0, ..d() }),
            "Vignetting"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { lowlight_blueness: 20.0, ..d() }),
            "Lowlight vision"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { gradnd_rotation: 30.0, ..d() }),
            "Graduated density"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { colisa_contrast: 0.4, ..d() }),
            "Contrast brightness saturation"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { lowpass_contrast: 0.6, ..d() }),
            "Lowpass"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { shadhi_shadows: 25.0, ..d() }),
            "Shadows/Highlights"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { primaries_red_hue: 10.0, ..d() }),
            "Primaries"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { toneeq_shadows: 0.5, ..d() }),
            "Tone equalizer"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { cb_saturation_global: 0.3, ..d() }),
            "Color balance RGB"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { tc_on: true, ..d() }),
            "Tone curve"
        );
        assert_eq!(
            describe_change(
                &base,
                &PreviewParams {
                    tc_nodes_l: {
                        let mut n = [(0.0f32, 0.0f32); 20];
                        n[1] = (0.75, 0.75);
                        n
                    },
                    ..d()
                }
            ),
            "Tone curve"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { rc_on: true, ..d() }),
            "RGB curve"
        );
        assert_eq!(
            describe_change(
                &base,
                &PreviewParams {
                    rc_nodes_b: {
                        let mut n = [(0.0f32, 0.0f32); 20];
                        n[1] = (0.6, 0.55);
                        n
                    },
                    ..d()
                }
            ),
            "RGB curve"
        );
        assert_eq!(
            describe_change(&base, &PreviewParams { bc_on: true, ..d() }),
            "Base curve"
        );
        assert_eq!(
            describe_change(
                &base,
                &PreviewParams {
                    bc_nodes: {
                        let mut n = [(0.0f32, 0.0f32); 20];
                        n[1] = (0.5, 0.4);
                        n
                    },
                    ..d()
                }
            ),
            "Base curve"
        );
        // No recognised difference ⇒ the generic fallback.
        assert_eq!(describe_change(&base, &base), "Edit");
    }

    #[test]
    fn describe_change_covers_every_previewparams_field() {
        // Drift guard: this exhaustive destructure (no `..`) fails to compile when
        // a field is added to PreviewParams, forcing whoever adds it to extend
        // `describe_change` with the new module group (otherwise edits to that
        // field would be silently mislabelled "Edit"). Pure compile-time check.
        let PreviewParams {
            exposure_on: _,
            black: _,
            ev: _,
            velvia_on: _,
            velvia_strength: _,
            velvia_bias: _,
            split_on: _,
            split_shadow_hue: _,
            split_shadow_sat: _,
            split_highlight_hue: _,
            split_highlight_sat: _,
            split_balance: _,
            split_compress: _,
            mono_on: _,
            mono_r: _,
            mono_g: _,
            mono_b: _,
            sigmoid_on: _,
            sigmoid_contrast: _,
            sigmoid_skew: _,
            sharpen_on: _,
            sharpen_radius: _,
            sharpen_amount: _,
            sharpen_threshold: _,
            vibrance_on: _,
            vibrance_amount: _,
            color_contrast_on: _,
            color_contrast_a_steepness: _,
            color_contrast_b_steepness: _,
            invert_on: _,
            invert_r: _,
            invert_g: _,
            invert_b: _,
            temperature_on: _,
            temperature_r: _,
            temperature_g: _,
            temperature_b: _,
            colorize_on: _,
            colorize_hue: _,
            colorize_sat: _,
            colorize_lightness: _,
            colorize_lightness_mix: _,
            color_correction_on: _,
            color_correction_loa: _,
            color_correction_hia: _,
            color_correction_lob: _,
            color_correction_hib: _,
            color_correction_saturation: _,
            colorzones_on: _,
            colorzones_strength: _,
            colorzones_channel: _,
            colorzones_mode: _,
            colorzones_num_nodes: _,
            colorzones_curve_type: _,
            colorzones_curve_x: _,
            colorzones_curve_y: _,
            levels_on: _,
            levels_black: _,
            levels_grey: _,
            levels_white: _,
            vignette_on: _,
            vignette_scale: _,
            vignette_falloff: _,
            vignette_brightness: _,
            vignette_saturation: _,
            vignette_center_x: _,
            vignette_center_y: _,
            vignette_shape: _,
            lowlight_on: _,
            lowlight_blueness: _,
            lowlight_transition: _,
            gradnd_on: _,
            gradnd_density: _,
            gradnd_hardness: _,
            gradnd_rotation: _,
            gradnd_offset: _,
            gradnd_hue: _,
            gradnd_saturation: _,
            colisa_on: _,
            colisa_contrast: _,
            colisa_brightness: _,
            colisa_saturation: _,
            basicadj_on: _,
            basicadj_black_point: _,
            basicadj_exposure: _,
            basicadj_hlcompr: _,
            basicadj_hlcomprthresh: _,
            basicadj_contrast: _,
            basicadj_preserve_colors: _,
            basicadj_middle_grey: _,
            basicadj_brightness: _,
            basicadj_saturation: _,
            basicadj_vibrance: _,
            lowpass_on: _,
            lowpass_radius: _,
            lowpass_contrast: _,
            lowpass_brightness: _,
            lowpass_saturation: _,
            shadhi_on: _,
            shadhi_shadows: _,
            shadhi_highlights: _,
            shadhi_whitepoint: _,
            shadhi_radius: _,
            shadhi_compress: _,
            shadhi_shadows_ccorrect: _,
            shadhi_highlights_ccorrect: _,
            primaries_on: _,
            primaries_achromatic_tint_hue: _,
            primaries_achromatic_tint_purity: _,
            primaries_red_hue: _,
            primaries_red_purity: _,
            primaries_green_hue: _,
            primaries_green_purity: _,
            primaries_blue_hue: _,
            primaries_blue_purity: _,
            negadoctor_on: _,
            negadoctor_film_stock: _,
            negadoctor_dmin_r: _,
            negadoctor_dmin_g: _,
            negadoctor_dmin_b: _,
            negadoctor_wb_high_r: _,
            negadoctor_wb_high_g: _,
            negadoctor_wb_high_b: _,
            negadoctor_wb_low_r: _,
            negadoctor_wb_low_g: _,
            negadoctor_wb_low_b: _,
            negadoctor_d_max: _,
            negadoctor_offset: _,
            negadoctor_black: _,
            negadoctor_gamma: _,
            negadoctor_soft_clip: _,
            negadoctor_exposure: _,
            toneeq_on: _,
            toneeq_noise: _,
            toneeq_ultra_deep_blacks: _,
            toneeq_deep_blacks: _,
            toneeq_blacks: _,
            toneeq_shadows: _,
            toneeq_midtones: _,
            toneeq_highlights: _,
            toneeq_whites: _,
            toneeq_speculars: _,
            cb_on: _,
            cb_shadows_y: _,
            cb_shadows_c: _,
            cb_shadows_h: _,
            cb_midtones_y: _,
            cb_midtones_c: _,
            cb_midtones_h: _,
            cb_highlights_y: _,
            cb_highlights_c: _,
            cb_highlights_h: _,
            cb_global_y: _,
            cb_global_c: _,
            cb_global_h: _,
            cb_shadows_weight: _,
            cb_white_fulcrum: _,
            cb_highlights_weight: _,
            cb_chroma_shadows: _,
            cb_chroma_highlights: _,
            cb_chroma_global: _,
            cb_chroma_midtones: _,
            cb_saturation_global: _,
            cb_saturation_highlights: _,
            cb_saturation_midtones: _,
            cb_saturation_shadows: _,
            cb_hue_angle: _,
            cb_brilliance_global: _,
            cb_brilliance_highlights: _,
            cb_brilliance_midtones: _,
            cb_brilliance_shadows: _,
            cb_mask_grey_fulcrum: _,
            cb_vibrance: _,
            cb_grey_fulcrum: _,
            cb_contrast: _,
            cb_formula: _,
            filmic_on: _,
            filmic_black_point_source: _,
            filmic_white_point_source: _,
            filmic_output_power: _,
            filmic_latitude: _,
            filmic_contrast: _,
            filmic_balance: _,
            filmic_saturation: _,
            hl_on: _,
            hl_opposed: _,
            hl_clip: _,
            dn_on: _,
            dn_mode_y0u0v0: _,
            dn_strength: _,
            dn_shadows: _,
            dn_bias: _,
            bl_on: _,
            bl_size: _,
            bl_threshold: _,
            bl_strength: _,
            tc_on: _,
            tc_type: _,
            tc_autoscale: _,
            tc_unbound: _,
            tc_preserve: _,
            tc_nnodes: _,
            tc_nodes_l: _,
            rc_on: _,
            rc_type_r: _,
            rc_type_g: _,
            rc_type_b: _,
            rc_autoscale: _,
            rc_preserve: _,
            rc_nnodes_r: _,
            rc_nnodes_g: _,
            rc_nnodes_b: _,
            rc_nodes_r: _,
            rc_nodes_g: _,
            rc_nodes_b: _,
            bc_on: _,
            bc_type: _,
            bc_preserve: _,
            bc_nnodes: _,
            bc_exposure_fusion: _,
            bc_exposure_stops: _,
            bc_exposure_bias: _,
            bc_nodes: _,
        } = PreviewParams::default();
    }

    #[test]
    fn describe_change_prefers_earliest_group_in_order() {
        // When two modules differ at once, the earlier pipeline group wins.
        let base = PreviewParams::default();
        let both = PreviewParams { ev: 1.0, mono_r: 0.9, ..PreviewParams::default() };
        assert_eq!(describe_change(&base, &both), "Exposure");
    }

    #[test]
    fn encode_decode_round_trips_with_cursor_and_labels() {
        let mut h = HistoryStack::new("Original", params(0.0));
        h.record("Exposure", params(1.0));
        h.record("Velvia", params(2.0));
        h.undo(); // cursor at index 1, with a redo tail present
        let blob = h.encode();
        let got = HistoryStack::decode(&blob).expect("decode");
        assert_eq!(got.len(), 3);
        assert_eq!(got.cursor(), 1);
        assert_eq!(got.entries()[0].label, "Original");
        assert_eq!(got.entries()[2].label, "Velvia");
        assert_eq!(got.current(), params(1.0));
        // Full structural equality of the entries.
        assert_eq!(got.entries(), h.entries());
    }

    #[test]
    fn decode_rejects_bad_version_truncation_and_bad_cursor() {
        let mut h = HistoryStack::new("Original", params(0.0));
        h.record("Exposure", params(1.0));
        let good = h.encode();

        // wrong version
        let mut bad = good.clone();
        bad[0] = 9;
        assert!(HistoryStack::decode(&bad).is_none());

        // truncated mid-blob
        assert!(HistoryStack::decode(&good[..good.len() - 3]).is_none());

        // trailing garbage
        let mut extra = good.clone();
        extra.push(0);
        assert!(HistoryStack::decode(&extra).is_none());

        // cursor past the end (cursor is bytes 1..5, little-endian)
        let mut bad_cursor = good.clone();
        bad_cursor[1..5].copy_from_slice(&99u32.to_le_bytes());
        assert!(HistoryStack::decode(&bad_cursor).is_none());

        // empty input
        assert!(HistoryStack::decode(&[]).is_none());
    }

    #[test]
    fn flush_style_record_after_undo_preserves_redo_tail() {
        // Pins the data-safety invariant the persistence flush depends on:
        // recording the current params while the cursor is mid-stack (params ==
        // current) must dedup and must NOT truncate the redo tail.
        let mut h = HistoryStack::new("Original", params(0.0));
        h.record("a", params(1.0));
        h.record("b", params(2.0));
        h.undo(); // cursor at index 1, redo tail = [b]
        let cur = h.current();
        assert!(!h.record(describe_change(&cur, &cur), cur)); // dedup ⇒ no-op
        assert_eq!(h.len(), 3);
        assert_eq!(h.cursor(), 1);
        assert!(h.can_redo());
    }

    #[test]
    fn previewparams_encode_len_is_pinned() {
        // Each history entry embeds a fixed-length PreviewParams::encode() blob.
        // If that length changes (a field added/removed), bump
        // HISTORY_ENCODE_VERSION (and PreviewParams' ENCODE_VERSION) so old
        // history blobs are rejected rather than mis-parsed. This pin forces the
        // deliberate decision when the length drifts.
        // m4-116 (toneequalizer): 680 → 717 (1 + 24 bools + 173 f32).
        // m4-117 (colorbalancergb): 717 → 850 (1 + 25 bools + 206 f32).
        // m4-118 (filmicrgb): 850 → 879 (1 + 26 bools + 213 f32).
        // m4-119 (highlight reconstruction): 879 → 885 (1 + 28 bools + 214 f32).
        // m4-120 (denoise profiled): 885 → 899 (1 + 30 bools + 217 f32).
        // m4-121 (bloom): 899 → 912 (1 + 31 bools + 220 f32).
        // m4-122 (tonecurve): 912 → 1090 (1 + 33 bools + 264 f32 — 4 scalars +
        // 40 interleaved L-anchor coordinates).
        // m4-123 (rgbcurve): 1090 → 1603 (1 + 34 bools + 392 f32 — 8 scalars +
        // 3×40 interleaved R/G/B anchor coordinates).
        // m4-124 (basecurve): 1603 → 1788 (1 + 35 bools + 438 f32 — 6 scalars +
        // 40 interleaved anchor coordinates).
        assert_eq!(PreviewParams::default().encode().len(), 1788);
    }

    #[test]
    fn decode_rejects_zero_and_over_cap_count() {
        let h = HistoryStack::new("Original", params(0.0));
        let good = h.encode();
        // count is bytes 5..9.
        let mut zero = good.clone();
        zero[5..9].copy_from_slice(&0u32.to_le_bytes());
        assert!(HistoryStack::decode(&zero).is_none());
        let mut huge = good.clone();
        huge[5..9].copy_from_slice(&(HISTORY_CAP as u32 + 1).to_le_bytes());
        assert!(HistoryStack::decode(&huge).is_none());
    }

    #[test]
    fn cap_drops_oldest_entries() {
        let mut h = HistoryStack::new("original", params(0.0));
        // Record well past the cap with all-distinct states.
        for i in 1..=(HISTORY_CAP as i32 + 10) {
            assert!(h.record(format!("e{i}"), params(i as f32)));
        }
        assert_eq!(h.len(), HISTORY_CAP);
        assert_eq!(h.cursor(), HISTORY_CAP - 1);
        // Cursor still points at the newest state.
        assert_eq!(h.current(), params((HISTORY_CAP as i32 + 10) as f32));
        // The "Original" seed is pinned at index 0 (the oldest *edits* are what
        // got dropped), so a jump-to-original is always possible.
        assert_eq!(h.entries()[0].label, "original");
        assert_eq!(h.entries()[0].params, params(0.0));
    }
}
