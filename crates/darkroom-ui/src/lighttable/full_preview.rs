//! Lighttable **full preview** (m4-98c c) — darktable's `f`: the selected image
//! filling the centre view instead of the grid, stepped with ← / → and dismissed
//! with `f` again or Escape.
//!
//! Not a [`ViewMode`](super::ViewMode). Full preview is orthogonal to the layout:
//! it works from the file manager and from culling alike, and returns to whichever
//! one was underneath. Making it a fourth mode would have meant persisting it (a
//! session that opens straight into a single image is a lighttable that looks
//! broken) and teaching the switcher a state that isn't a layout.
//!
//! **Coverage limit:** images are decoded with gdk-pixbuf, so this shows exactly
//! what the grid's thumbnails show. Camera raws gdk-pixbuf can't read (`.ORF` and
//! friends, which render as empty grid cells today) get a "no preview available"
//! message rather than a blank page. Routing raws through the darkroom view's
//! pipeline is the follow-up; it would have made this increment un-shippable.
//!
//! **The grid is not unmapped.** The preview is an `Overlay` child *over* the
//! grid, not a `Stack` page beside it. That is load-bearing rather than
//! stylistic: a `Stack` unmaps the hidden page, GTK drops focus from an unmapped
//! widget, and the lighttable's key controller lives on the `GridView` — so the
//! preview became a **keyboard trap**, with `f`, Escape and the arrows all landing
//! on whatever else took focus. Found by pressing the keys in the container; every
//! unit test here passed throughout.

use adw::prelude::*;
use gtk4::glib;

/// Floor for the decode size, and the ceiling that bounds memory when the widget
/// reports something unreasonable. The actual target comes from the allocation
/// (see [`FullPreview::load`]): full preview exists to judge focus and detail, so
/// a fixed 2048 would have the user inspecting resampling artefacts on a 4K
/// display.
const FULL_PREVIEW_MIN_DIM: i32 = 512;
const FULL_PREVIEW_MAX_DIM: i32 = 4096;

/// What a key does to the full preview, given whether it is currently open. Pure,
/// so the mapping is testable with no display.
///
/// Everything except the toggle is gated on `open`, which is what keeps this from
/// stealing keys the lighttable needs: ← / → page the culling window when the
/// preview is closed. (Escape is gated too, though on the lighttable root it
/// belongs to nobody — `adw::NavigationView` can't pop its root page.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PreviewAction {
    /// Open on the selected image, or close if already open.
    Toggle,
    /// Leave the preview (Escape).
    Close,
    /// Step to the next/previous image without leaving the preview.
    Next,
    Prev,
    /// Swallow the key without acting on it. These are keys that would otherwise
    /// move the selection or page the culling window *underneath* a preview that
    /// wouldn't follow — a full-screen image next to another image's metadata.
    /// Better to do nothing visibly than something invisibly.
    Ignore,
}

pub fn preview_key_action(keyval: gtk4::gdk::Key, open: bool) -> Option<PreviewAction> {
    use gtk4::gdk::Key;
    match keyval {
        Key::f | Key::F => Some(PreviewAction::Toggle),
        Key::Escape if open => Some(PreviewAction::Close),
        Key::Right | Key::space | Key::Page_Down if open => Some(PreviewAction::Next),
        Key::Left | Key::Page_Up if open => Some(PreviewAction::Prev),
        // GridView's own bindings move the cursor; there is no within-image cursor
        // in a full preview, so they'd desync it from the selection.
        Key::Up | Key::Down | Key::Home | Key::End if open => Some(PreviewAction::Ignore),
        _ => None,
    }
}

/// What the preview should be showing for a given selection. Pure, so the rule
/// "no real image selected ⇒ get out of the way" is testable without a display —
/// it is the rule that keeps a reload from leaving a full-screen image the app no
/// longer considers selected.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PreviewTarget {
    Show(String),
    Close,
}

pub fn preview_target(selected: Option<&str>) -> PreviewTarget {
    match selected {
        Some(path) => PreviewTarget::Show(path.to_string()),
        None => PreviewTarget::Close,
    }
}

/// The full-preview layer over the lighttable's centre view.
///
/// Holds only the layer's own widgets — deliberately **not** the `Overlay` that
/// contains it. The key handler in `lib.rs` captures this, the grid owns that
/// handler, and the `Overlay` is the grid's ancestor: storing it here would close
/// a reference cycle that keeps the whole centre subtree (grid, model, every
/// cached texture) alive for the process's lifetime. GTK parents hold their
/// children, never the reverse, so keeping only children is cycle-free.
#[derive(Clone)]
pub struct FullPreview {
    /// The layer itself. Visibility *is* the open/closed state — one source of
    /// truth, so the state can't drift from what's on screen.
    layer: gtk4::Overlay,
    picture: gtk4::Picture,
    /// Shown instead of the image when the file can't be decoded, so a preview
    /// that fails is *visibly* a failure rather than a black rectangle the user
    /// has to guess about.
    status: gtk4::Label,
}

impl FullPreview {
    /// Wrap `content` (the grid's scroller) in an `Overlay` with the preview layer
    /// on top. Returns the container to pack where `content` used to go, plus the
    /// handle that drives it.
    pub fn wrap(content: &impl IsA<gtk4::Widget>) -> (gtk4::Overlay, Self) {
        let picture = gtk4::Picture::new();
        picture.set_content_fit(gtk4::ContentFit::Contain);
        picture.set_hexpand(true);
        picture.set_vexpand(true);

        let status = gtk4::Label::new(None);
        status.add_css_class("dim-label");
        status.set_halign(gtk4::Align::Center);
        status.set_valign(gtk4::Align::Center);
        status.set_visible(false);

        // An Overlay, not a Box: the status message belongs *over* the image area
        // (centred, where the image would have been), not stacked under it in its
        // own row at the bottom of the view.
        let layer = gtk4::Overlay::new();
        // Opaque on purpose: `ContentFit::Contain` letterboxes, and a transparent
        // letterbox would show the grid ghosting through the preview.
        layer.add_css_class("background");
        layer.set_hexpand(true);
        layer.set_vexpand(true);
        layer.set_visible(false);
        layer.set_child(Some(&picture));
        layer.add_overlay(&status);

        let overlay = gtk4::Overlay::new();
        overlay.set_child(Some(content));
        overlay.add_overlay(&layer);

        (overlay, Self { layer, picture, status })
    }

    pub fn is_open(&self) -> bool {
        self.layer.is_visible()
    }

    /// Show `path`, decoding it off the main thread.
    pub fn open(&self, path: &str) {
        self.layer.set_visible(true);
        self.load(path);
    }

    pub fn close(&self) {
        self.layer.set_visible(false);
        // Drop the texture: one full-size image is worth holding while it's on
        // screen, not for the rest of the session.
        self.picture.set_paintable(gtk4::gdk::Paintable::NONE);
        self.picture.set_widget_name("");
    }

    /// Apply a [`PreviewTarget`] — the selection-observer path. A no-op while the
    /// preview is closed, so an ordinary click in the grid doesn't open it.
    pub fn follow_selection(&self, target: &PreviewTarget) {
        if !self.is_open() {
            return;
        }
        match target {
            PreviewTarget::Show(path) => {
                // Already showing it: reloading would restart a decode for nothing.
                if self.picture.widget_name() != path.as_str() {
                    self.open(path);
                }
            }
            PreviewTarget::Close => self.close(),
        }
    }

    /// Decode and paint `path`. The path is stamped on the `Picture` before the
    /// await and re-checked after it, so a fast ← / → run paints the image the
    /// user actually stopped on — the same stale-paint guard the grid cells use,
    /// and for the same reason: decodes finish out of order.
    fn load(&self, path: &str) {
        let path = path.to_string();
        self.picture.set_widget_name(&path);
        self.status.set_visible(false);
        // Decode to what we'll actually paint into, in physical pixels. Measured
        // here (main thread, allocation known) rather than baked in as a constant.
        let target = self.decode_target();
        let picture = self.picture.clone();
        let status = self.status.clone();
        glib::spawn_future_local(async move {
            // Only the *read* goes off-thread: `Pixbuf` is not `Send`, so it can't
            // cross back from a worker (the grid's thumbnail loader splits the
            // work the same way). Decoding at a bounded size keeps the main-thread
            // half short.
            let p = path.clone();
            let bytes = gtk4::gio::spawn_blocking(move || std::fs::read(&p).ok())
                .await
                .ok()
                .flatten();
            // NOTE: the stale guard covers everything below only because there is
            // no further `await` — do not add one under this line without moving
            // the check with it.
            if picture.widget_name() != path {
                return; // the user moved on while this decoded
            }
            let pixbuf = bytes.and_then(|data| {
                let loader = gtk4::gdk_pixbuf::PixbufLoader::new();
                // Scale during decode rather than after, so a large JPEG never
                // materialises at full size just to be thrown away.
                loader.connect_size_prepared(move |loader, w, h| {
                    let longest = w.max(h);
                    if longest > target {
                        // One scale factor on both axes: `set_size` does NOT
                        // preserve aspect ratio for you.
                        let scale = f64::from(target) / f64::from(longest);
                        loader.set_size(
                            ((f64::from(w) * scale) as i32).max(1),
                            ((f64::from(h) * scale) as i32).max(1),
                        );
                    }
                });
                // Both unconditional: a loader finalized without `close()` emits a
                // g_warning, so an early return on a rejected header would print
                // one on every keypress that lands on that file.
                let _ = loader.write(&data);
                let _ = loader.close();
                loader.pixbuf()
            });
            match pixbuf {
                Some(pb) => {
                    picture.set_paintable(Some(&gtk4::gdk::Texture::for_pixbuf(&pb)));
                    status.set_visible(false);
                }
                None => {
                    // Raw formats gdk-pixbuf can't read (the grid shows an empty
                    // cell for these too) and unreadable files land here. Say so:
                    // a blank preview with no explanation is indistinguishable
                    // from a hang.
                    picture.set_paintable(gtk4::gdk::Paintable::NONE);
                    status.set_label(&format!(
                        "No preview available for {}",
                        std::path::Path::new(&path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(&path)
                    ));
                    status.set_visible(true);
                }
            }
        });
    }

    /// Longest side to decode to: the widget's allocation in physical pixels,
    /// bounded. Before the first allocation the picture reports 0, which the floor
    /// turns into a modest decode rather than a 1-pixel one.
    fn decode_target(&self) -> i32 {
        let logical = self.picture.width().max(self.picture.height());
        let physical = logical.saturating_mul(self.picture.scale_factor().max(1));
        physical.clamp(FULL_PREVIEW_MIN_DIM, FULL_PREVIEW_MAX_DIM)
    }
}

/// The index a ← / → step lands on, staying inside `0..n_items`. `None` when there
/// is nothing to move to: at either end of the collection (the preview does not
/// wrap around — running off the end of 2000 images to land back at the start is
/// disorienting) or when nothing is selected. Pure.
pub fn preview_step_index(current: u32, n_items: u32, forward: bool) -> Option<u32> {
    // GTK reports "no selection" as this sentinel; stepping from it is not a
    // clamped move but no move at all — clamping would silently select the
    // second-to-last image on a `←` with nothing selected.
    if n_items == 0 || current == gtk4::INVALID_LIST_POSITION {
        return None;
    }
    let last = n_items - 1;
    let current = current.min(last);
    let next = if forward {
        current.saturating_add(1).min(last)
    } else {
        current.saturating_sub(1)
    };
    (next != current).then_some(next)
}

#[cfg(test)]
mod tests {
    use super::{preview_key_action, preview_step_index, preview_target, PreviewAction,
                PreviewTarget};
    use gtk4::gdk::Key;

    #[test]
    fn preview_keys_are_gated_on_being_open() {
        // The toggle works from either state — that is how the preview is opened.
        assert_eq!(preview_key_action(Key::f, false), Some(PreviewAction::Toggle));
        assert_eq!(preview_key_action(Key::f, true), Some(PreviewAction::Toggle));
        // Everything else is inert while closed, so it can't steal keys the
        // lighttable owns: ← / → page the culling window, the grid moves its
        // cursor with the other arrows.
        for key in [Key::Escape, Key::Left, Key::Right, Key::space, Key::Page_Up,
                    Key::Page_Down, Key::Up, Key::Down, Key::Home, Key::End] {
            assert_eq!(preview_key_action(key, false), None, "{key:?} while closed");
            assert!(preview_key_action(key, true).is_some(), "{key:?} while open");
        }
        // Keys owned by the metadata shortcuts must never map, in either state.
        for key in [Key::F1, Key::F5, Key::_0, Key::_5] {
            assert_eq!(preview_key_action(key, true), None, "{key:?} must not map");
            assert_eq!(preview_key_action(key, false), None, "{key:?} must not map");
        }
    }

    #[test]
    fn keys_that_would_desync_the_preview_are_swallowed_not_forwarded() {
        // Page Up/Down page the culling window and the arrows move the grid
        // cursor. While the preview is up, forwarding either moves the collection
        // *underneath* a full-screen image — so they must map to something, and
        // that something must not be a silent fall-through.
        assert_eq!(preview_key_action(Key::Page_Down, true), Some(PreviewAction::Next));
        assert_eq!(preview_key_action(Key::Page_Up, true), Some(PreviewAction::Prev));
        for key in [Key::Up, Key::Down, Key::Home, Key::End] {
            assert_eq!(preview_key_action(key, true), Some(PreviewAction::Ignore));
        }
    }

    #[test]
    fn preview_target_closes_when_nothing_real_is_selected() {
        // A reload can drop the previewed image (a filter, a folder switch) or
        // land on a placeholder row, which `selected_path` reports as None. The
        // preview must get out of the way rather than keep showing an image the
        // app no longer considers selected.
        assert_eq!(preview_target(None), PreviewTarget::Close);
        assert_eq!(
            preview_target(Some("/a/b.jpg")),
            PreviewTarget::Show("/a/b.jpg".to_string())
        );
    }

    #[test]
    fn preview_step_stops_at_both_ends() {
        assert_eq!(preview_step_index(0, 5, true), Some(1));
        assert_eq!(preview_step_index(3, 5, true), Some(4));
        // No wrap-around: past the last image is a hold, not a jump to the first.
        assert_eq!(preview_step_index(4, 5, true), None);
        assert_eq!(preview_step_index(0, 5, false), None);
        assert_eq!(preview_step_index(4, 5, false), Some(3));
        // An empty collection has nowhere to step...
        assert_eq!(preview_step_index(0, 0, true), None);
        // ...and "nothing selected" is not a position to step from, in either
        // direction. (Clamping the sentinel would pick the second-to-last image.)
        assert_eq!(preview_step_index(gtk4::INVALID_LIST_POSITION, 5, true), None);
        assert_eq!(preview_step_index(gtk4::INVALID_LIST_POSITION, 5, false), None);
    }
}
