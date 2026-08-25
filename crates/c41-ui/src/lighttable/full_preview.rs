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
//! **Decode paths (m4-132):** raster files go through gdk-pixbuf — the same
//! source the grid's thumbnails show. Camera raws (anything
//! [`crate::raw_preview::is_raw_path`](crate::raw_preview) claims) decode instead
//! through the darkroom view's raw pipeline pieces — `raw_preview::
//! decode_raw_preview` (demosaic + white balance + linear downscale) followed by
//! [`render_linear_to_srgb8`](crate::preview::render_linear_to_srgb8) at *default*
//! params — so an `.ORF` renders as the image itself, matching what the darkroom
//! view shows before any edit ("as shot"), rather than the old "no preview
//! available" message. No shared-module lift was needed: those two stages were
//! already free-standing functions outside the darkroom module, unlike the
//! darkroom's live-preview state (`darkroom::BaseImage` and friends), which is
//! entangled with per-image editing state a lighttable preview deliberately
//! doesn't have.
//!
//! Still true: the *grid* cells stay gdk-pixbuf-only, so raws keep their empty
//! thumbnails there; the full preview is where they become visible.
//!
//! **Zoom/pan (m4-133, parity 3.6):** the picture lives inside a
//! `ScrolledWindow`; plain-wheel zooms through the stop list fit → 100 % → 200 %
//! → … → [`ZOOM_MAX`], anchored at the cursor (a `EventControllerMotion` sidecar
//! remembers the last pointer position, because the scroll controller carries no
//! coordinates), primary-button drag pans, and a double-click toggles fit ↔ 100 %.
//! 100 % means one texture pixel per *physical* screen pixel of the loaded
//! preview buffer — the buffer is capped by [`FULL_PREVIEW_MAX_DIM`], so beyond
//! its resolution this upsamples like any viewer; a full-res-on-demand pipeline
//! is darkroom territory, not a lighttable glance. Zoom resets to fit whenever
//! the image changes or the preview closes, so ← / → never lands mid-zoom on a
//! corner of the next frame.
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
use std::cell::Cell;
use std::rc::Rc;

/// Floor for the decode size, and the ceiling that bounds memory when the widget
/// reports something unreasonable. The actual target comes from the allocation
/// (see [`FullPreview::load`]): full preview exists to judge focus and detail, so
/// a fixed 2048 would have the user inspecting resampling artefacts on a 4K
/// display.
const FULL_PREVIEW_MIN_DIM: i32 = 512;
const FULL_PREVIEW_MAX_DIM: i32 = 4096;

/// Wheel zoom ceiling (× of the loaded buffer's native pixels). Four stops in,
/// and every stop doubles: enough to inspect demosaic detail, not enough to get
/// lost in upsampling mush.
const ZOOM_MAX: f64 = 8.0;

/// The next wheel-zoom state. `None` is fit-to-window; `Some(scale)` scales the
/// loaded buffer by `scale`, with `Some(1.0)` = 100 %. Stepping out below 100 %
/// lands on fit (there is no sub-100 % stop — shrinking below the window buys
/// nothing), stepping past the top clamps there. Pure.
fn step_zoom(current: Option<f64>, zoom_in: bool) -> Option<f64> {
    match (current, zoom_in) {
        (None, true) => Some(1.0),
        (None, false) => None, // already as small as it gets
        (Some(s), true) if s < ZOOM_MAX => Some((s * 2.0).min(ZOOM_MAX)),
        (Some(_), true) => Some(ZOOM_MAX), // pinned at the top: stay at max
        // Out-steps below the first real stop fold into fit.
        (Some(s), false) if s <= 1.0 => None,
        (Some(s), false) => Some((s / 2.0).max(1.0)),
    }
}

/// Pixel size of the picture widget for a zoom `scale` (`None` = fit): the scaled
/// texture dims rounded to whole pixels, never degenerate. Fit mode requests
/// exactly the viewport so the image centres via [`gtk4::ContentFit::Contain`]
/// and no scrollbar ever appears; scaled mode requests the scaled dims so the
/// scroller's adjustments describe the real pan range. Pure.
fn zoomed_dims(tex: (i32, i32), viewport: (i32, i32), scale: Option<f64>) -> (i32, i32) {
    match scale {
        None => (viewport.0.max(1), viewport.1.max(1)),
        Some(k) => (
            ((f64::from(tex.0) * k).round() as i32).max(1),
            ((f64::from(tex.1) * k).round() as i32).max(1),
        ),
    }
}

/// Where the cursor sits in *image* coordinates — the anchor a zoom keeps fixed.
///
/// `cursor_viewport` is in **viewport space** — what's on screen, i.e. a
/// scrolled child's widget-local coordinate minus the scroll offset. Controllers
/// on such a child report content space, so callers translate first (see
/// [`ZoomState::apply`]); the fallback centre is already viewport-space.
///
/// In fit mode the texture is letterboxed by `ContentFit::Contain`, so the
/// on-screen point maps through that rectangle first
/// ([`crate::preview::contain_rect`]); `None` when it falls outside the drawn
/// image (zooming then keeps the current view instead of jumping to the border).
/// In scaled mode there is no letterbox and content = viewport + adjustment, so
/// the adjustment translates back to content before dividing by the scale. Pure.
fn image_coords_under_cursor(
    cursor_viewport: (f64, f64),
    adj: (f64, f64),
    scale: Option<f64>,
    tex: (i32, i32),
    viewport: (i32, i32),
) -> Option<(f64, f64)> {
    let (tw, th) = (f64::from(tex.0.max(1)), f64::from(tex.1.max(1)));
    match scale {
        None => {
            let r = crate::preview::contain_rect(
                f64::from(viewport.0.max(1)),
                f64::from(viewport.1.max(1)),
                tex.0.max(1) as usize,
                tex.1.max(1) as usize,
            )?;
            let inside = cursor_viewport.0 >= r.off_x
                && cursor_viewport.1 >= r.off_y
                && cursor_viewport.0 <= r.off_x + r.disp_w
                && cursor_viewport.1 <= r.off_y + r.disp_h;
            inside.then(|| {
                (
                    (cursor_viewport.0 - r.off_x) / r.disp_w * tw,
                    (cursor_viewport.1 - r.off_y) / r.disp_h * th,
                )
            })
        }
        Some(k) if k > 0.0 => Some((
            (cursor_viewport.0 + adj.0) / k,
            (cursor_viewport.1 + adj.1) / k,
        )),
        Some(_) => None,
    }
}

/// The adjustment value that puts image coordinate `img` under screen offset
/// `cursor` at `scale`, clamped into the scrollable range. Pure.
fn anchored_adjustment(img: f64, cursor: f64, scale: f64, view: i32, content: i32) -> f64 {
    let raw = img * scale - cursor;
    let upper = f64::from(content.saturating_sub(view).max(0));
    raw.clamp(0.0, upper)
}

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
    /// The zoom/pan scroller around the [`Self::picture`] (m4-133). Its
    /// allocation — not the picture's — is the viewport: once zoomed, the
    /// picture's size is the scaled content size and would feed back into itself.
    scroll: gtk4::ScrolledWindow,
    picture: gtk4::Picture,
    /// Shown instead of the image when the file can't be decoded, so a preview
    /// that fails is *visibly* a failure rather than a black rectangle the user
    /// has to guess about.
    status: gtk4::Label,
    /// Shared zoom state — the cells the controllers mutate plus weak handles on
    /// the two widgets they lay out. One bundle so every closure clones the same
    /// thing instead of re-listing its parts.
    zs: ZoomState,
}

/// Everything the zoom machinery mutates or lays out, shared between the
/// controllers wired once in [`FullPreview::wrap`] and the per-image paint paths.
/// The widgets are held **weak**: these clones live inside closures attached to
/// the picture and the scroller, and a strong child→ancestor edge there would be
/// exactly the never-freed-subtree cycle the [`FullPreview`] doc warns about.
/// The `Rc` cells carry no widget edges and are safe to share freely.
#[derive(Clone)]
struct ZoomState {
    scroll_w: glib::WeakRef<gtk4::ScrolledWindow>,
    picture_w: glib::WeakRef<gtk4::Picture>,
    /// Texture dims of whatever is currently painted; `(0, 0)` until the first
    /// successful paint. Zoom math needs them but a `Picture` won't hand them back.
    tex_dims: Rc<Cell<(i32, i32)>>,
    /// Current wheel zoom — `None` = fit-to-window, `Some(k)` = k× native buffer
    /// pixels (see [`step_zoom`]). Reset whenever the image changes or the
    /// preview closes, so ← / → never lands mid-zoom on a corner of the next
    /// frame.
    zoom: Rc<Cell<Option<f64>>>,
    /// Generation counter guarding the deferred adjustment fix-ups (`apply`
    /// re-sets adjustment values once GTK has reallocated at the new request;
    /// each newer apply invalidates older pendings).
    zoom_gen: Rc<Cell<u64>>,
}

impl ZoomState {
    fn new(scroll: &gtk4::ScrolledWindow, picture: &gtk4::Picture) -> Self {
        Self {
            scroll_w: scroll.downgrade(),
            picture_w: picture.downgrade(),
            tex_dims: Rc::new(Cell::new((0, 0))),
            zoom: Rc::new(Cell::new(None)),
            zoom_gen: Rc::new(Cell::new(0)),
        }
    }

    /// Re-layout the picture for the current [`zoom`](Self::zoom) value: fit
    /// requests exactly the viewport so the image centres via
    /// [`gtk4::ContentFit::Contain`] and no scrollbar ever appears; a zoom
    /// requests the scaled texture dims so the adjustments describe the real pan
    /// range. Passing `old_scale` + an anchor point (the wheel/double-click
    /// paths) keeps the image point under that anchor fixed across the step;
    /// without them (resets, refits) the view simply goes to fit.
    ///
    /// Adjustment values are written twice — immediately (GTK clamps to the
    /// *old* content size until it reallocates) and once more from an idle, so
    /// the anchored position survives the allocation pass. Each apply stamps a
    /// fresh generation so superseded idles become no-ops.
    fn apply(&self, old_scale: Option<f64>, anchor_widget: Option<(f64, f64)>) {
        let Some(scroll) = self.scroll_w.upgrade() else { return };
        let Some(picture) = self.picture_w.upgrade() else { return };
        let scale = self.zoom.get();
        let vp = (scroll.width().max(1), scroll.height().max(1));
        let tex = self.tex_dims.get();
        // Nothing painted (yet): fit geometry regardless of `scale`, so a failed
        // decode can't inherit a zoomed request from the previous image.
        let painted = tex.0 > 0 && tex.1 > 0;
        let (rw, rh) = if painted { zoomed_dims(tex, vp, scale) } else { vp };
        picture.set_content_fit(if scale.is_none() {
            gtk4::ContentFit::Contain
        } else {
            gtk4::ContentFit::Fill
        });
        picture.set_size_request(rw.max(1), rh.max(1));

        // Every apply invalidates older deferred fix-ups — including this one's,
        // when the arms below don't schedule a replacement.
        let gen = self.zoom_gen.get().wrapping_add(1);
        self.zoom_gen.set(gen);

        let h = scroll.hadjustment();
        let v = scroll.vadjustment();
        let Some(k) = scale else {
            // Fit: the scrollable range is empty by construction; write it
            // anyway so a previous zoom's offset can't survive a refit.
            h.set_value(0.0);
            v.set_value(0.0);
            return;
        };
        if !painted {
            return;
        }
        // Anchor: the motion/click controllers report picture-local
        // coordinates, which is **content** space once scrolled — the current
        // adjustments turn those into **viewport** space, the uniform contract
        // of the helpers below (and of the fallback centre). Before any motion
        // has landed, fall back to the viewport centre — zooming on what the
        // user is looking at, for a wheel event straight after opening.
        let anchor =
            anchor_widget.unwrap_or((f64::from(vp.0) / 2.0, f64::from(vp.1) / 2.0));
        let (ah, av) = (h.value(), v.value());
        let cursor_vp = (anchor.0 - ah, anchor.1 - av);
        let Some((ix, iy)) =
            image_coords_under_cursor(cursor_vp, (ah, av), old_scale, tex, vp)
        else {
            // Anchor over the letterbox (fit mode only): keep the current view
            // instead of jumping wherever the maths would land.
            return;
        };
        let nh = anchored_adjustment(ix, cursor_vp.0, k, vp.0, rw);
        let nv = anchored_adjustment(iy, cursor_vp.1, k, vp.1, rh);
        h.set_value(nh);
        v.set_value(nv);
        // Deferred re-assert once the child is reallocated at the new request and
        // the uppers describe the new pan range (the immediate writes above were
        // clamped to the old one).
        let gens = self.zoom_gen.clone();
        let sw = scroll.downgrade();
        glib::idle_add_local_once(move || {
            if gens.get() != gen || sw.upgrade().is_none() {
                return;
            }
            h.set_value(nh);
            v.set_value(nv);
        });
    }

    /// Back to fit with no texture registered — the state a fresh image starts
    /// from and a closed preview leaves behind.
    fn reset(&self) {
        self.zoom.set(None);
        self.tex_dims.set((0, 0));
        self.apply(None, None);
    }
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

        // Zoom/pan host (m4-133): the picture's request IS the pan range — fit
        // requests exactly the viewport (no scrollbars), zoom requests the scaled
        // dims and the scroller does the rest. Both policies Automatic: which
        // bars exist follows from the mode instead of being pinned.
        let scroll = gtk4::ScrolledWindow::builder()
            .hscrollbar_policy(gtk4::PolicyType::Automatic)
            .vscrollbar_policy(gtk4::PolicyType::Automatic)
            .hexpand(true)
            .vexpand(true)
            .child(&picture)
            .build();

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
        layer.set_child(Some(&scroll));
        layer.add_overlay(&status);

        let zs = ZoomState::new(&scroll, &picture);
        Self::wire_zoom_controls(&picture, &zs);
        Self::wire_resize_refit(&scroll, &zs);

        let overlay = gtk4::Overlay::new();
        overlay.set_child(Some(content));
        overlay.add_overlay(&layer);

        (overlay, Self { layer, scroll, picture, status, zs })
    }

    /// Wheel / double-click / drag controllers, all on the **picture** so their
    /// handlers speak picture-local coordinates (content space, once scrolled)
    /// and so the wheel's [`glib::signal::Propagation::Stop`] starves the
    /// `ScrolledWindow`'s native wheel scrolling — the wheel belongs to the
    /// zoomer entirely. Closures capture [`ZoomState`] (weak widgets + `Rc`
    /// cells), never a strong ancestor.
    fn wire_zoom_controls(picture: &gtk4::Picture, zs: &ZoomState) {
        use glib::signal::Propagation;
        use gtk4::EventControllerScrollFlags as ScrollFlags;

        // Last pointer position over the picture, for cursor-anchored zoom — the
        // scroll controller carries no coordinates. `enter` seeds it too, so the
        // very first wheel after the pointer crosses onto the image anchors
        // correctly rather than at the fallback centre.
        let last_pos: Rc<Cell<Option<(f64, f64)>>> = Rc::new(Cell::new(None));
        let motion = gtk4::EventControllerMotion::new();
        {
            let lp = last_pos.clone();
            motion.connect_enter(move |_, x, y| lp.set(Some((x, y))));
            let lp = last_pos.clone();
            motion.connect_motion(move |_, x, y| lp.set(Some((x, y))));
        }
        picture.add_controller(motion);

        // Wheel → one stop along fit → 100 % → … → ZOOM_MAX. DISCRETE keeps
        // touchpad smooth-scroll out of the zoomer entirely: a two-finger flick
        // would otherwise fire a stop per event and rocket fit → 8×; instead
        // those events fall through to the ScrolledWindow's native panning.
        let wheel =
            gtk4::EventControllerScroll::new(ScrollFlags::VERTICAL | ScrollFlags::DISCRETE);
        {
            let zs = zs.clone();
            let zoom = zs.zoom.clone();
            let last_pos = last_pos.clone();
            wheel.connect_scroll(move |_, _dx, dy| {
                let tex = zs.tex_dims.get();
                if tex.0 <= 0 || tex.1 <= 0 {
                    // Nothing painted (failed decode): don't flip invisible
                    // state — but still eat the wheel.
                    return Propagation::Stop;
                }
                let old = zoom.get();
                let next = step_zoom(old, dy < 0.0);
                if next != old {
                    zoom.set(next);
                    zs.apply(old, last_pos.get());
                }
                // Stop even when the step was a no-op (pinned at a stop): the
                // wheel must never fall through to scrolling.
                Propagation::Stop
            });
        }
        picture.add_controller(wheel);

        // Double-click toggles fit ↔ 100 %, anchored at the click itself.
        let dbl = gtk4::GestureClick::new();
        dbl.set_button(gtk4::gdk::BUTTON_PRIMARY);
        {
            let zs = zs.clone();
            let zoom = zs.zoom.clone();
            dbl.connect_pressed(move |gesture, n, x, y| {
                if n != 2 {
                    return;
                }
                gesture.set_state(gtk4::EventSequenceState::Claimed);
                let tex = zs.tex_dims.get();
                if tex.0 <= 0 || tex.1 <= 0 {
                    return; // nothing painted: same guard as the wheel
                }
                let old = zoom.get();
                let next = if old.is_none() { Some(1.0) } else { None };
                zoom.set(next);
                zs.apply(old, Some((x, y)));
            });
        }
        picture.add_controller(dbl);

        // Primary-drag pans by dragging the adjustments backwards. In fit mode
        // the scrollable range is empty and the clamped writes become no-ops, so
        // there is deliberately no mode gate here.
        let drag = gtk4::GestureDrag::new();
        drag.set_button(gtk4::gdk::BUTTON_PRIMARY);
        {
            // Adjustment values captured at drag start; both axes move together
            // or not at all, hence one `Option<(f64, f64)>`.
            let start: Rc<Cell<Option<(f64, f64)>>> = Rc::new(Cell::new(None));
            drag.connect_drag_begin({
                let zs = zs.clone();
                let start = start.clone();
                move |_, _, _| {
                    if let Some(s) = zs.scroll_w.upgrade() {
                        start.set(Some((
                            s.hadjustment().value(),
                            s.vadjustment().value(),
                        )));
                    }
                }
            });
            drag.connect_drag_update({
                let zs = zs.clone();
                move |_, dx, dy| {
                    if let (Some(s), Some((sh, sv))) =
                        (zs.scroll_w.upgrade(), start.get())
                    {
                        // `set_value` clamps into the scrollable range.
                        s.hadjustment().set_value(sh - dx);
                        s.vadjustment().set_value(sv - dy);
                    }
                }
            });
        }
        picture.add_controller(drag);
    }

    /// Re-fit when the window resizes. The viewport size reaches us as the
    /// adjustments' `page-size` (the same signal the grid's column stepper keys
    /// off), coalesced through a single idle so a resize storm collapses into one
    /// re-apply. Zoomed mode deliberately doesn't chase resizes — the pan range
    /// just grows/shrinks around the content — which is what an image viewer does.
    fn wire_resize_refit(scroll: &gtk4::ScrolledWindow, zs: &ZoomState) {
        let pending = Rc::new(Cell::new(false));
        for adj in [scroll.hadjustment(), scroll.vadjustment()] {
            let zs = zs.clone();
            let pending = pending.clone();
            adj.connect_page_size_notify(move |_| {
                if zs.zoom.get().is_some() || pending.replace(true) {
                    return;
                }
                let zs = zs.clone();
                let pending = pending.clone();
                glib::idle_add_local_once(move || {
                    pending.set(false);
                    // Re-check inside the idle: the user may have wheel-zoomed
                    // between the notify and this running.
                    if zs.zoom.get().is_none() {
                        zs.apply(None, None);
                    }
                });
            });
        }
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
        self.zs.reset();
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
        // Fresh image, fresh viewport: any inherited zoom dies here, and sizing
        // falls back to fit until this decode actually paints (m4-133).
        self.zs.reset();
        // Decode to what we'll actually paint into, in physical pixels. Measured
        // here (main thread, allocation known) rather than baked in as a constant.
        let target = self.decode_target();
        let picture = self.picture.clone();
        let status = self.status.clone();
        let zs = self.zs.clone();

        if crate::raw_preview::is_raw_path(&path) {
            // Raw branch (m4-132): demosaic + white-balance + linear downscale,
            // then sRGB-encode at default ("as shot") params — everything
            // off-thread because every intermediate (`RawPreview`, the packed
            // RGB bytes) is an owned `Send` buffer. A raw demosaic can take
            // seconds; say so instead of leaving a silent letterbox. Both awaits
            // below are followed by the same stale guard the pixbuf path uses,
            // and each arm returns before the other runs, so exactly one decode
            // can ever paint.
            self.show_status(&format!(
                "Decoding {}…",
                file_display_name(&path)
            ));
            glib::spawn_future_local(async move {
                let p = path.clone();
                let frame = gtk4::gio::spawn_blocking(move || {
                    // `target` is already clamped ≥ FULL_PREVIEW_MIN_DIM by
                    // `decode_target`, so it converts to the decoder's `max_dim`
                    // as-is.
                    crate::raw_preview::decode_raw_preview(&p, target as usize).map(|rp| {
                        let bytes = crate::preview::render_linear_to_srgb8(
                            &rp.pixels,
                            rp.width,
                            rp.height,
                            &crate::preview::PreviewParams::default(),
                        );
                        (rp.width, rp.height, bytes)
                    })
                })
                .await
                .ok()
                .flatten();
                // NOTE: the stale guard covers everything below only because
                // there is no further `await` — do not add one under this line
                // without moving the check with it.
                if picture.widget_name() != path {
                    return; // the user moved on while this decoded
                }
                match frame {
                    Some((w, h, bytes)) => {
                        // Same 3-channel upload the darkroom preview uses (see
                        // `darkroom::cached_render_texture`): tightly-packed RGB8
                        // straight out of the sRGB encode.
                        let tex = gtk4::gdk::MemoryTexture::new(
                            w as i32,
                            h as i32,
                            gtk4::gdk::MemoryFormat::R8g8b8,
                            &glib::Bytes::from_owned(bytes),
                            w * 3,
                        );
                        picture.set_paintable(Some(&tex));
                        status.set_visible(false);
                        // Register the real buffer size, then land on fit
                        // (m4-133): a new frame never opens mid-zoom on a corner.
                        zs.tex_dims.set((w as i32, h as i32));
                        zs.apply(None, None);
                    }
                    None => show_unavailable(&picture, &status, &path),
                }
            });
            return;
        }

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
                    // Same registration the raw branch does (m4-133).
                    zs.tex_dims.set((pb.width(), pb.height()));
                    zs.apply(None, None);
                }
                None => show_unavailable(&picture, &status, &path),
            }
        });
    }

    /// Longest side to decode to: the scroller's allocation in physical pixels
    /// (m4-133: the *scroller*, not the picture — once zoomed, the picture's
    /// allocation is the scaled content size, and measuring it would feed the
    /// previous session's zoom into the next decode), bounded. Before the first
    /// allocation the widget reports 0, which the floor turns into a modest
    /// decode rather than a 1-pixel one.
    fn decode_target(&self) -> i32 {
        let logical = self.scroll.width().max(self.scroll.height());
        let physical = logical.saturating_mul(self.scroll.scale_factor().max(1));
        physical.clamp(FULL_PREVIEW_MIN_DIM, FULL_PREVIEW_MAX_DIM)
    }

    /// Show a transient message over the (still-empty) image area — used while a
    /// raw demosaic runs, which can take seconds on a 20MP file.
    fn show_status(&self, text: &str) {
        self.status.set_label(text);
        self.status.set_visible(true);
    }
}

/// The failure paint shared by both decode branches: clear the picture and say
/// what couldn't be shown. A blank preview with no explanation is
/// indistinguishable from a hang.
fn show_unavailable(picture: &gtk4::Picture, status: &gtk4::Label, path: &str) {
    picture.set_paintable(gtk4::gdk::Paintable::NONE);
    status.set_label(&format!("No preview available for {}", file_display_name(path)));
    status.set_visible(true);
}

/// The file's bare name for user-facing messages, falling back to the whole path
/// when it isn't valid UTF-8.
fn file_display_name(path: &str) -> &str {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
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
    use super::{anchored_adjustment, image_coords_under_cursor, preview_key_action,
                preview_step_index, preview_target, step_zoom, zoomed_dims,
                PreviewAction, PreviewTarget};
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

    #[test]
    fn zoom_steps_walk_the_stop_list_and_clamp_both_ends() {
        // Fit zooms in to 100 %; zooming out of fit has nowhere to go.
        assert_eq!(step_zoom(None, true), Some(1.0));
        assert_eq!(step_zoom(None, false), None);
        // Doubling stops up to the ceiling…
        assert_eq!(step_zoom(Some(1.0), true), Some(2.0));
        assert_eq!(step_zoom(Some(4.0), true), Some(8.0));
        // …where further in-steps pin rather than wrap or grow.
        assert_eq!(step_zoom(Some(8.0), true), Some(8.0));
        // Halving stops back down, folding into fit below 100 %.
        assert_eq!(step_zoom(Some(8.0), false), Some(4.0));
        assert_eq!(step_zoom(Some(2.0), false), Some(1.0));
        assert_eq!(step_zoom(Some(1.0), false), None);
    }

    #[test]
    fn zoomed_dims_round_and_never_reach_zero() {
        // Fit passes the viewport straight through — that's what makes the image
        // centre via Contain with an exactly-empty scroll range.
        assert_eq!(zoomed_dims((6000, 4000), (1280, 800), None), (1280, 800));
        // Scaling rounds to whole pixels…
        assert_eq!(zoomed_dims((641, 483), (100, 100), Some(2.0)), (1282, 966));
        // …and never below one pixel per axis.
        assert_eq!(zoomed_dims((1, 1), (100, 100), Some(8.0)), (8, 8));
        assert_eq!(zoomed_dims((1, 1), (100, 100), Some(0.125)), (1, 1));
    }

    #[test]
    fn image_coords_map_through_the_letterbox_in_fit_mode() {
        // Square viewport, square image: the contain rect is the whole viewport,
        // so the mapping is pure proportion.
        let sq =
            image_coords_under_cursor((50.0, 50.0), (0.0, 0.0), None, (200, 200), (100, 100));
        assert_eq!(sq, Some((100.0, 100.0)));
        // Tall image in a square viewport is pillarboxed 50 px each side:
        // contain rect = x ∈ [50, 150], y ∈ [0, 200].
        let inside =
            image_coords_under_cursor((75.0, 100.0), (0.0, 0.0), None, (100, 200), (200, 200));
        assert_eq!(inside, Some((25.0, 100.0)));
        // A cursor on the pillar is not on the image — zooming there must refuse
        // to anchor rather than divide by a zero-width display.
        let outside =
            image_coords_under_cursor((20.0, 50.0), (0.0, 0.0), None, (100, 200), (200, 200));
        assert_eq!(outside, None);
    }

    #[test]
    fn image_coords_translate_by_the_adjustment_when_scaled() {
        // Scaled mode has no letterbox and content = viewport + scroll offset,
        // divided by the scale — the cursor arrives in *viewport* space.
        let p = image_coords_under_cursor(
            (30.0, 40.0),
            (10.0, 20.0),
            Some(2.0),
            (400, 400),
            (100, 100),
        );
        assert_eq!(p, Some((20.0, 30.0)));
        // A degenerate scale refuses to divide instead of producing infinities.
        assert_eq!(
            image_coords_under_cursor(
                (30.0, 40.0),
                (0.0, 0.0),
                Some(0.0),
                (400, 400),
                (100, 100)
            ),
            None
        );
    }

    #[test]
    fn image_coords_speak_viewport_space_not_content_space() {
        // The bug class review caught on m4-133: controllers on a scrolled
        // picture report content space, but this helper's contract is viewport
        // space. Panned 100 px right / 60 down at 2×, a cursor at viewport
        // (50, 30) sits over content (150, 90) = image (75, 45). Passing the raw
        // controller coordinate here would compute (50+100)/2 = 75 for x by
        // accident but double-count y — so the caller MUST translate first
        // (`ZoomState::apply` does); this test pins which space that is.
        let p = image_coords_under_cursor(
            (50.0, 30.0),
            (100.0, 60.0),
            Some(2.0),
            (400, 400),
            (200, 200),
        );
        assert_eq!(p, Some((75.0, 45.0)));
    }

    #[test]
    fn anchored_adjustment_keeps_the_anchor_fixed_and_clamps() {
        // Image point 100 under screen offset 50 at 2×: value = 2·100 − 50.
        assert_eq!(anchored_adjustment(100.0, 50.0, 2.0, 100, 500), 150.0);
        // Clamped into [0, content − view]: never scrolls past either end…
        assert_eq!(anchored_adjustment(0.0, 500.0, 2.0, 100, 500), 0.0);
        assert_eq!(anchored_adjustment(400.0, 10.0, 2.0, 100, 500), 400.0);
        // …and an empty range (content smaller than view) pins to 0 without
        // saturating-subtraction surprises.
        assert_eq!(anchored_adjustment(100.0, 0.0, 2.0, 500, 100), 0.0);
    }
}
