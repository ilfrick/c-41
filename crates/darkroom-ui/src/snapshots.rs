//! Snapshot store for the darkroom view (Phase 3 milestone-4): a small, capped
//! set of captured renders the user can compare side-by-side with the live edit
//! (darktable's snapshots lib). This is the pure model — labelling, the cap, and
//! removal — kept free of GTK so it's unit-testable headless (the established
//! display-free discipline). The GTK layer instantiates it over the cached
//! render payload (`SnapshotStore<CachedRender>`).
//!
//! v1 comparison is **side-by-side** (snapshot in a second `Picture` beside the
//! live image, each `Contain`-fit in equal halves). It is approximate, not
//! pixel-aligned: a snapshot frozen at one preview size and the live image at
//! another letterbox independently, so a feature won't sit at the same panel-x
//! across the divider. darktable's scale-locked overlay with a draggable wipe
//! line is a future increment.

/// One captured snapshot: an auto-assigned label plus an opaque payload (the
/// frozen render in the GTK layer; any `P` in tests).
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot<P> {
    pub label: String,
    pub payload: P,
}

/// Default maximum retained snapshots (darktable keeps a small handful).
pub const SNAPSHOT_CAP: usize = 8;

/// A capped, auto-labelled collection of snapshots. Labels come from a monotonic
/// counter ("Snapshot 1", "Snapshot 2", …) so they stay stable as entries are
/// removed (the next capture never reuses a number); capturing past the cap
/// evicts the oldest.
#[derive(Clone, Debug)]
pub struct SnapshotStore<P> {
    entries: Vec<Snapshot<P>>,
    next_id: usize,
    cap: usize,
}

impl<P> SnapshotStore<P> {
    /// New empty store retaining at most `cap` snapshots (clamped to ≥ 1).
    pub fn new(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            cap: cap.max(1),
        }
    }

    /// Capture `payload` under the next auto label, evicting the oldest if that
    /// pushes past the cap. Returns the new snapshot's label.
    pub fn capture(&mut self, payload: P) -> String {
        let label = format!("Snapshot {}", self.next_id);
        self.next_id += 1;
        self.entries.push(Snapshot {
            label: label.clone(),
            payload,
        });
        if self.entries.len() > self.cap {
            let overflow = self.entries.len() - self.cap;
            self.entries.drain(0..overflow);
        }
        label
    }

    /// Remove the snapshot at `index`, returning it (`None` if out of range).
    pub fn remove(&mut self, index: usize) -> Option<Snapshot<P>> {
        (index < self.entries.len()).then(|| self.entries.remove(index))
    }

    /// Drop all snapshots (the auto-label counter keeps advancing).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn get(&self, index: usize) -> Option<&Snapshot<P>> {
        self.entries.get(index)
    }

    pub fn entries(&self) -> &[Snapshot<P>] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_auto_labels_monotonically() {
        let mut s = SnapshotStore::new(SNAPSHOT_CAP);
        assert_eq!(s.capture("a"), "Snapshot 1");
        assert_eq!(s.capture("b"), "Snapshot 2");
        assert_eq!(s.len(), 2);
        assert_eq!(s.entries()[0].payload, "a");
        assert_eq!(s.entries()[1].label, "Snapshot 2");
    }

    #[test]
    fn cap_evicts_oldest_keeping_newest() {
        let mut s = SnapshotStore::new(3);
        for i in 0..5 {
            s.capture(i);
        }
        assert_eq!(s.len(), 3);
        // Oldest two (0, 1) evicted; newest three remain in order.
        let payloads: Vec<_> = s.entries().iter().map(|e| e.payload).collect();
        assert_eq!(payloads, vec![2, 3, 4]);
        // Labels reflect the monotonic counter (no reuse after eviction).
        assert_eq!(s.entries()[0].label, "Snapshot 3");
        assert_eq!(s.entries()[2].label, "Snapshot 5");
    }

    #[test]
    fn remove_by_index_and_bounds() {
        let mut s = SnapshotStore::new(SNAPSHOT_CAP);
        s.capture("a");
        s.capture("b");
        s.capture("c");
        let removed = s.remove(1);
        assert_eq!(removed.map(|e| e.payload), Some("b"));
        assert_eq!(s.len(), 2);
        assert_eq!(s.entries()[0].payload, "a");
        assert_eq!(s.entries()[1].payload, "c");
        assert!(s.remove(99).is_none()); // out of range
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn labels_stay_stable_after_remove() {
        let mut s = SnapshotStore::new(SNAPSHOT_CAP);
        s.capture("a"); // Snapshot 1
        s.capture("b"); // Snapshot 2
        s.remove(0); // drop Snapshot 1
        // The next capture continues the counter — no reuse of "1".
        assert_eq!(s.capture("c"), "Snapshot 3");
        assert_eq!(s.entries()[0].label, "Snapshot 2");
        assert_eq!(s.entries()[1].label, "Snapshot 3");
    }

    #[test]
    fn clear_empties_but_keeps_counter() {
        let mut s = SnapshotStore::new(SNAPSHOT_CAP);
        s.capture("a");
        s.capture("b");
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.capture("c"), "Snapshot 3"); // counter advanced past 1,2
    }

    #[test]
    fn new_clamps_zero_cap_to_one() {
        let mut s = SnapshotStore::new(0);
        s.capture("a");
        s.capture("b");
        assert_eq!(s.len(), 1);
        assert_eq!(s.entries()[0].payload, "b"); // only the newest survives
    }
}
