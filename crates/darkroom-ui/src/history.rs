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
        // Length of one params blob (fixed; computed from a default encode).
        let plen = PreviewParams::default().encode().len();
        let mut p = 0usize;
        let take = |p: &mut usize, n: usize| -> Option<()> {
            (*p + n <= bytes.len()).then(|| *p += n)
        };

        if *bytes.first()? != HISTORY_ENCODE_VERSION {
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
            take(&mut p, plen)?;
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
const HISTORY_ENCODE_VERSION: u8 = 2;

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
        assert_eq!(PreviewParams::default().encode().len(), 386);
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
