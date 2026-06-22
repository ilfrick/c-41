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
