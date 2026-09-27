//! Slideshow view (u8 — parity audit 3.4, slideshow leg): darktable's
//! `src/views/slideshow.c` as a pushed `NavigationPage`.
//!
//! The page shows one collection image at a time, fullscreen, with an
//! auto-advance timer, manual stepping, a persisted per-second delay, and
//! chrome that hides itself after a few idle seconds.
//!
//! Rendering reuses the lighttable full preview wholesale: the page embeds a
//! [`FullPreview`](crate::lighttable::full_preview::FullPreview) via
//! `FullPreview::wrap` over an empty placeholder and drives it with
//! `open(path)` per index. That is the exact decode path the lighttable `f`
//! preview uses — gdk-pixbuf at the allocation-sized target for raster
//! files, the darkroom raw pipeline (`decode_raw_preview` + default-params
//! sRGB encode) for camera raws — so no decode code is duplicated here. The
//! wheel-zoom and drag-pan controllers come along with it, which is a
//! harmless bonus, not a feature this page owns.
//!
//! Behaviour mirrored from the C (with line evidence in the notes below):
//! delay clamped 1–60 s and persisted under the same `slideshow_delay` key
//! (`_set_delay`, slideshow.c:186-191); stepping clamps at both ends with no
//! wrap — the C logs "end of images" and clears `auto_advance`
//! (slideshow.c:357-361 forward, :378-382 back); keyboard stepping stops the
//! advance first (`_step_forward_callback`, :714-719); Escape exits the view
//! (`_exit_callback`, :721-727); mouse motion re-shows the cursor and re-arms
//! a 1 s hide (`mouse_moved`, :624-637).
//!
//! Two deliberate adaptations, both documented at the use site: the page
//! starts paused (the C enters with `auto_advance = FALSE`, :503, and waits
//! for space — same here, with a visible Play control instead of a log
//! line); and chrome auto-hide covers the header plus the bottom bar after
//! ~3 s idle rather than the 1 s cursor hide (GTK gives no cursor API with
//! the same shape, and hiding the bar is what frees the pixels).
//!
//! Deviation: a mouse click only stops the timer. The C steps forward on
//! primary click and back on secondary (`button_pressed`, :650-668); this
//! page keeps clicks side-effect-free apart from stopping the advance, per
//! the u8 scope (any key/click stops the timer, only a second Esc or the
//! back button pops).

use adw::prelude::*;
use gtk4::{gdk, glib};
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use crate::lighttable::full_preview::FullPreview;
use crate::persist::{load_ui_pref, save_ui_pref};

/// Navigation-page tag. Deliberately slash-free, so the lighttable's
/// `popped` cell re-sync (which only handles tags containing `/`) and the
/// view-switcher mirror (which lights Darkroom only for slash tags) both
/// ignore it — the same contract as the print, map and tether tags.
pub const SLIDESHOW_PAGE_TAG: &str = "slideshow";

/// `darkroom_ui_prefs` key for the advance delay. Same key NAME the C
/// persists under (`dt_conf_set_int("slideshow_delay", ...)`,
/// slideshow.c:190), but a different STORE (our prefs table, not
/// darktablerc) — values do not cross between the apps, the name just
/// stays recognisable.
pub const SLIDESHOW_DELAY_PREF_KEY: &str = "slideshow_delay";

/// Default advance delay in seconds. Mirrors the C default
/// (`data/darktableconfig.xml.in`, `slideshow_delay` entry: `<default>5`).
pub const SLIDESHOW_DELAY_DEFAULT: i32 = 5;

/// Delay bounds in seconds. Mirrors the C clamp
/// (`CLAMP(d->delay + value, 1, 60)`, slideshow.c:189).
pub const SLIDESHOW_DELAY_MIN: i32 = 1;
pub const SLIDESHOW_DELAY_MAX: i32 = 60;

/// Idle seconds before the chrome (header + bottom bar) hides itself. The C
/// hides the cursor after 1 s (`g_timeout_add_seconds(1, _hide_mouse, ...)`
/// in `mouse_moved`, slideshow.c:636); 3 s here because hiding interactive
/// chrome on a 1 s hair-trigger would yank the Play control away mid-reach.
pub const CHROME_IDLE_HIDE_SECS: u64 = 3;

// ── Pure model (GTK-free, headless-tested) ──────────────────────────────────

/// Clamp a delay into the C's 1–60 s range (slideshow.c:189). Pure.
pub fn clamp_delay(delay: i32) -> i32 {
    delay.clamp(SLIDESHOW_DELAY_MIN, SLIDESHOW_DELAY_MAX)
}

/// Read the persisted advance delay: unparseable or absent pref falls back
/// to the C default, and any out-of-range value is clamped rather than
/// trusted — a hand-edited pref must never schedule a 0 s busy loop or a
/// week-long stall. Mirrors the int-pref read shape used for panel widths
/// (`stored_panel_width` in `lib.rs`): parse, clamp, default. Pure over its
/// `Option<&str>` input; the DB read itself stays at the call site.
pub fn parse_delay(raw: Option<&str>) -> i32 {
    raw.and_then(|v| v.trim().parse::<i32>().ok())
        .map(clamp_delay)
        .unwrap_or(SLIDESHOW_DELAY_DEFAULT)
}

/// Load the persisted advance delay for `db_path`. Pure logic is
/// [`parse_delay`]; this only adds the DB read.
pub fn load_delay(db_path: &str) -> i32 {
    parse_delay(load_ui_pref(db_path, SLIDESHOW_DELAY_PREF_KEY).as_deref())
}

/// Persist the advance delay (clamped first, so the stored value is always
/// one this page would actually use).
pub fn save_delay(db_path: &str, delay: i32) {
    save_ui_pref(
        db_path,
        SLIDESHOW_DELAY_PREF_KEY,
        &clamp_delay(delay).to_string(),
    );
}

/// Bottom-bar counter text: 1-based position over the total, `"N / M"`.
/// Empty collection renders `"0 / 0"` (reachable only if the list empties
/// after construction — entry refuses an empty list). Pure.
pub fn counter_text(index: usize, n: usize) -> String {
    if n == 0 {
        return "0 / 0".to_string();
    }
    format!("{} / {}", index.saturating_add(1).min(n), n)
}

/// Delay readout next to the stepper, `"N s"`. Pure.
pub fn delay_text(delay: i32) -> String {
    format!("{} s", clamp_delay(delay))
}

/// The index a step lands on, clamped at both ends with no wrap: past the
/// last image (or before the first) is `None`, mirroring the C's "end of
/// images" stop (slideshow.c:357-361, :378-382) rather than the full
/// preview's hold-or-page behaviour. `None` also when the collection is
/// empty or `current` is already out of range. Pure.
pub fn step_index(current: usize, n: usize, forward: bool) -> Option<usize> {
    if n == 0 || current >= n {
        return None;
    }
    if forward {
        (current + 1 < n).then_some(current + 1)
    } else {
        current.checked_sub(1)
    }
}

/// Resolve the opening index for `paths` given the lighttable selection:
/// the selected image when it is in the list, else the first image. `None`
/// when there is nothing to show — the entry action treats that as a no-op
/// (mirroring `try_enter` refusing an empty collection, slideshow.c:417-429,
/// and the print action's empty guard). Deviation, documented: the C falls
/// back to the thumbtable scroll offset (slideshow.c:487-490); this port
/// always falls back to index 0, which is the sane choice without a scroll
/// offset to read. Pure.
pub fn resolve_start(paths: &[String], selected: Option<&str>) -> Option<usize> {
    if paths.is_empty() {
        return None;
    }
    let idx = selected
        .and_then(|s| paths.iter().position(|p| p == s))
        .unwrap_or(0);
    Some(idx.min(paths.len() - 1))
}

/// What a key does on the slideshow page, given whether the advance timer
/// is currently running. Pure, so the mapping is testable with no display.
///
/// Space toggles play/pause (the C's "start and stop", slideshow.c:731);
/// arrows step (C: Right/Left, :746-749) and stop the advance first — the
/// stopping itself happens in the controller for *every* key (see below),
/// matching the C step callbacks that clear `auto_advance` before stepping
/// (:705-719); Up slows down / Down speeds up (C: Up = delay +1, Down =
/// delay −1, :737-744); Escape stops a running advance, otherwise pops the
/// page (C exits the view outright, :721-727 — the two-stage form keeps a
/// stray Esc from throwing the user out mid-show).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlideshowAction {
    TogglePlay,
    Next,
    Prev,
    Slower,
    Faster,
    Stop,
    Pop,
}

pub fn slideshow_key_action(keyval: gdk::Key, playing: bool) -> Option<SlideshowAction> {
    use gdk::Key;
    match keyval {
        Key::Escape if playing => Some(SlideshowAction::Stop),
        Key::Escape => Some(SlideshowAction::Pop),
        Key::space => Some(SlideshowAction::TogglePlay),
        Key::Right => Some(SlideshowAction::Next),
        Key::Left => Some(SlideshowAction::Prev),
        Key::Up => Some(SlideshowAction::Slower),
        Key::Down => Some(SlideshowAction::Faster),
        // Keypad/numpad and shifted variants the C also binds
        // (slideshow.c:737-744): KP_Add/KP_Subtract plus literal `+`/`-`,
        // which is what most layouts produce with or without Shift.
        Key::KP_Add | Key::plus => Some(SlideshowAction::Slower),
        Key::KP_Subtract | Key::minus => Some(SlideshowAction::Faster),
        _ => None,
    }
}

// ── Page widget ─────────────────────────────────────────────────────────────

/// Live state for one pushed slideshow page. Held in an `Rc` shared by the
/// timer, the key/click/motion controllers and the bottom-bar controls.
/// Every closure below captures only a `Weak` — except the page's own
/// `unmap` handler, which holds the ONE strong `Rc` that keeps the `Show`
/// alive exactly as long as the page (page → closure → `Show` → descendant
/// widgets is a DAG; GTK child→parent refs are weak, so nothing cycles or
/// leaks past pop). The widgets are owned by `Show` while the full preview
/// handle and timers borrow weakly, same rule as the full preview's
/// child-only handle and the tether page's WeakRef-only widget touches.
struct Show {
    paths: Vec<String>,
    index: Cell<usize>,
    delay: Cell<i32>,
    playing: Cell<bool>,
    timer: RefCell<Option<glib::SourceId>>,
    idle_gen: Cell<u64>,
    db_path: String,
    preview: FullPreview,
    counter: gtk4::Label,
    status: gtk4::Label,
    play_btn: gtk4::ToggleButton,
    delay_spin: gtk4::SpinButton,
    chrome_top: adw::HeaderBar,
    chrome_bottom: gtk4::Box,
}

impl Show {
    fn stop_timer(&self) {
        if let Some(id) = self.timer.borrow_mut().take() {
            id.remove();
        }
        self.playing.set(false);
        self.sync_play_button();
    }

    fn sync_play_button(&self) {
        let playing = self.playing.get();
        if self.play_btn.is_active() != playing {
            self.play_btn.set_active(playing);
        }
        self.play_btn.set_icon_name(if playing {
            "media-playback-pause-symbolic"
        } else {
            "media-playback-start-symbolic"
        });
        self.play_btn.set_tooltip_text(Some(if playing {
            "Pause the slideshow (space)"
        } else {
            "Play the slideshow (space)"
        }));
    }

    fn paint_at(&self, index: usize) {
        self.index.set(index);
        if let Some(path) = self.paths.get(index) {
            self.preview.open(path);
        }
        self.counter
            .set_label(&counter_text(index, self.paths.len()));
        self.status.set_label("");
    }

    /// Advance one step inside the timer tick. Returns `false` when the
    /// show reached an end, so the tick stops the advance exactly like the
    /// C's "end of images" arm (slideshow.c:357-361).
    fn tick_advance(self: &Rc<Self>) -> bool {
        match step_index(self.index.get(), self.paths.len(), true) {
            Some(next) => {
                self.paint_at(next);
                true
            }
            None => {
                self.status.set_label("End of images");
                self.stop_timer();
                false
            }
        }
    }

    /// (Re)arm the single-shot advance chain at the current delay. Each
    /// tick re-arms itself while the show is still playing, mirroring the
    /// C's `_step_state` tail that schedules the next `_auto_advance` only
    /// `if (d->auto_advance)` (slideshow.c:388).
    fn arm_timer(self: &Rc<Self>) {
        self.stop_timer();
        self.playing.set(true);
        self.sync_play_button();
        let weak: Weak<Self> = Rc::downgrade(self);
        let secs = self.delay.get().max(1) as u32;
        let id = glib::timeout_add_seconds_local(secs, move || {
            let Some(show) = weak.upgrade() else {
                return glib::ControlFlow::Break;
            };
            if !show.playing.get() {
                return glib::ControlFlow::Break;
            }
            if show.tick_advance() {
                glib::ControlFlow::Continue
            } else {
                glib::ControlFlow::Break
            }
        });
        *self.timer.borrow_mut() = Some(id);
    }

    fn set_playing(self: &Rc<Self>, playing: bool) {
        if playing {
            self.status.set_label("");
            self.arm_timer();
        } else {
            self.stop_timer();
        }
    }

    /// Manual step: stops the advance first, mirroring the C step callbacks
    /// that clear `auto_advance` before `_step_state`
    /// (slideshow.c:705-719). At an end the position holds with no wrap —
    /// same clamp as [`step_index`] — and a forward step past the last
    /// image raises the "End of images" line.
    fn manual_step(self: &Rc<Self>, forward: bool) {
        let was_playing = self.playing.get();
        if was_playing {
            self.stop_timer();
        }
        if let Some(next) = step_index(self.index.get(), self.paths.len(), forward) {
            self.paint_at(next);
        } else if forward {
            // The C logs "end of images" on a forward step past the last
            // rank regardless of whether the advance was running
            // (slideshow.c:359); a backward step off the head simply holds.
            self.status.set_label("End of images");
        }
    }

    /// Store a new delay: clamp, persist (the C's `_set_delay`,
    /// slideshow.c:186-191), sync the stepper without looping back through
    /// its own handler, and re-arm a running advance so the new delay takes
    /// effect from now rather than from the already-scheduled tick.
    fn store_delay(self: &Rc<Self>, delay: i32) {
        let next = clamp_delay(delay);
        if next == self.delay.get() {
            return;
        }
        self.delay.set(next);
        save_delay(&self.db_path, next);
        if self.delay_spin.value_as_int() != next {
            self.delay_spin.set_value(f64::from(next));
        }
        if self.playing.get() {
            self.arm_timer();
        }
    }

    /// Show the chrome and re-arm the idle hide. Called on pointer motion
    /// and on handled keys — the analogue of the C's `mouse_moved`
    /// re-showing the cursor and re-arming the hide (slideshow.c:624-637).
    fn poke_chrome(&self) {
        self.chrome_top.set_visible(true);
        self.chrome_bottom.set_visible(true);
        let gen = self.idle_gen.get().wrapping_add(1);
        self.idle_gen.set(gen);
    }
}

/// Schedule the chrome auto-hide `CHROME_IDLE_HIDE_SECS` after the latest
/// poke. Generation-guarded: an older pending hide is a no-op once a newer
/// poke has moved the generation on (the same shape as the full preview's
/// deferred zoom fix-ups).
fn arm_chrome_hide(show: &Rc<Show>) {
    let gen = show.idle_gen.get();
    let weak: Weak<Show> = Rc::downgrade(show);
    glib::timeout_add_local_once(
        std::time::Duration::from_secs(CHROME_IDLE_HIDE_SECS),
        move || {
            let Some(show) = weak.upgrade() else { return };
            if show.idle_gen.get() != gen {
                return;
            }
            show.chrome_top.set_visible(false);
            show.chrome_bottom.set_visible(false);
        },
    );
}

/// Build the slideshow page for `paths`, opening on `start` (clamped into
/// range). An empty `paths` still builds — a labelled empty state, never a
/// panic — but the `win.slideshow` entry action never pushes one
/// (`resolve_start` returning `None` is its no-op).
pub fn slideshow_page(paths: Vec<String>, start: usize, db_path: String) -> adw::NavigationPage {
    let n = paths.len();
    let start = if n == 0 { 0 } else { start.min(n - 1) };
    let delay = load_delay(&db_path);

    // Image surface: an empty placeholder wrapped by the full preview, so
    // the preview layer paints over the whole centre slot. Driven with
    // `open(path)` per index — the lighttable `f` decode path verbatim.
    let placeholder = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    placeholder.set_hexpand(true);
    placeholder.set_vexpand(true);
    let (preview_overlay, preview) = FullPreview::wrap(&placeholder);

    // Bottom chrome: step controls, play/pause, counter, delay stepper.
    let chrome_bottom = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
    chrome_bottom.set_halign(gtk4::Align::Center);
    chrome_bottom.set_valign(gtk4::Align::End);
    chrome_bottom.set_margin_bottom(12);

    let prev_btn = gtk4::Button::builder()
        .icon_name("go-previous-symbolic")
        .tooltip_text("Previous image (Left)")
        .build();
    let play_btn = gtk4::ToggleButton::builder()
        .icon_name("media-playback-start-symbolic")
        .tooltip_text("Play the slideshow (space)")
        .build();
    let next_btn = gtk4::Button::builder()
        .icon_name("go-next-symbolic")
        .tooltip_text("Next image (Right)")
        .build();
    let counter = gtk4::Label::new(Some(&counter_text(start, n)));
    counter.add_css_class("dim-label");
    let delay_adj = gtk4::Adjustment::new(
        f64::from(delay),
        f64::from(SLIDESHOW_DELAY_MIN),
        f64::from(SLIDESHOW_DELAY_MAX),
        1.0,
        5.0,
        0.0,
    );
    let delay_spin = gtk4::SpinButton::new(Some(&delay_adj), 1.0, 0);
    delay_spin.set_tooltip_text(Some("Seconds per image (Up/Down)"));
    // Seed before the handler connects, so the initial fill is not mistaken
    // for a user edit (same ordering the lighttable filter dropdown relies
    // on).
    delay_spin.set_value(f64::from(delay));
    let delay_unit = gtk4::Label::new(Some("s / image"));
    delay_unit.add_css_class("dim-label");
    let status = gtk4::Label::new(None);
    status.add_css_class("dim-label");

    for w in [
        prev_btn.upcast_ref::<gtk4::Widget>(),
        play_btn.upcast_ref::<gtk4::Widget>(),
        next_btn.upcast_ref::<gtk4::Widget>(),
        counter.upcast_ref::<gtk4::Widget>(),
        delay_spin.upcast_ref::<gtk4::Widget>(),
        delay_unit.upcast_ref::<gtk4::Widget>(),
        status.upcast_ref::<gtk4::Widget>(),
    ] {
        chrome_bottom.append(w);
    }

    let chrome_top = adw::HeaderBar::new();
    chrome_top.set_title_widget(Some(&adw::WindowTitle::new("Slideshow", " ")));
    // Explicit Back: the page has no view-switcher, so without this the only
    // exits would be Esc and swipe. Pops via the NavigationView's built-in
    // action (no nav handle needed) — the darkroom view-switcher pattern.
    let back_btn = gtk4::Button::builder()
        .icon_name("go-previous-symbolic")
        .tooltip_text("Back to the lighttable")
        .build();
    back_btn.connect_clicked(|b| {
        if let Err(e) = b.activate_action("navigation.pop", None) {
            eprintln!("slideshow: back pop failed (page not in a NavigationView?): {e}");
        }
    });
    chrome_top.pack_start(&back_btn);

    let outer = gtk4::Overlay::new();
    outer.set_hexpand(true);
    outer.set_vexpand(true);
    outer.set_child(Some(&preview_overlay));
    outer.add_overlay(&chrome_bottom);
    // Key and motion controllers live on the overlay so they fire wherever
    // the pointer/focus is inside the page.
    outer.set_focusable(true);
    outer.set_can_target(true);

    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&chrome_top);
    toolbar.set_content(Some(&outer));
    toolbar.set_vexpand(true);

    let empty_note = gtk4::Label::new(Some("No images in this collection."));
    empty_note.set_halign(gtk4::Align::Center);
    empty_note.set_valign(gtk4::Align::Center);
    empty_note.set_visible(n == 0);

    let content = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    content.set_hexpand(true);
    content.set_vexpand(true);
    content.append(&toolbar);
    content.append(&empty_note);

    let page = adw::NavigationPage::builder()
        .title("Slideshow")
        .tag(SLIDESHOW_PAGE_TAG)
        .child(&content)
        .build();

    let show = Rc::new(Show {
        paths,
        index: Cell::new(start),
        delay: Cell::new(delay),
        playing: Cell::new(false),
        timer: RefCell::new(None),
        idle_gen: Cell::new(0),
        db_path,
        preview,
        counter,
        status,
        play_btn,
        delay_spin,
        chrome_top,
        chrome_bottom,
    });

    // First paint. Starts paused, mirroring the C entering with
    // `auto_advance = FALSE` (slideshow.c:503) — the user presses play.
    if n > 0 {
        show.paint_at(start);
    }
    show.sync_play_button();

    // Play/pause toggle.
    {
        let weak = Rc::downgrade(&show);
        show.play_btn.connect_toggled(move |b| {
            let Some(show) = weak.upgrade() else { return };
            // Guard the programmatic sync in `sync_play_button`: only a real
            // state change re-arms or stops the timer.
            if b.is_active() != show.playing.get() {
                show.set_playing(b.is_active());
            }
            show.poke_chrome();
            arm_chrome_hide(&show);
        });
    }
    // Step buttons: manual steps stop the advance first (C :705-719).
    {
        let weak = Rc::downgrade(&show);
        prev_btn.connect_clicked(move |_| {
            let Some(show) = weak.upgrade() else { return };
            show.manual_step(false);
            show.poke_chrome();
            arm_chrome_hide(&show);
        });
    }
    {
        let weak = Rc::downgrade(&show);
        next_btn.connect_clicked(move |_| {
            let Some(show) = weak.upgrade() else { return };
            show.manual_step(true);
            show.poke_chrome();
            arm_chrome_hide(&show);
        });
    }

    // Delay stepper. `value-changed` fires for button clicks and typed
    // entries alike; the guard below keeps programmatic syncs (from the
    // Up/Down keys) from looping back through the persist + re-arm.
    {
        let weak = Rc::downgrade(&show);
        show.delay_spin.connect_value_changed(move |spin| {
            let Some(show) = weak.upgrade() else { return };
            show.store_delay(spin.value_as_int());
            show.poke_chrome();
            arm_chrome_hide(&show);
        });
    }

    // Any key stops a running advance first (the u8 interaction rule); known
    // keys then act, unknown ones propagate so the page never traps keys it
    // does not own. Escape on a stopped show pops via the NavigationView's
    // built-in `navigation.pop` action — the same handle-free pop the
    // darkroom view-switcher uses.
    {
        let weak = Rc::downgrade(&show);
        let page_w = page.downgrade();
        let spin_w = show.delay_spin.downgrade();
        let keys = gtk4::EventControllerKey::new();
        keys.connect_key_pressed(move |_, keyval, _, _| {
            let Some(show) = weak.upgrade() else {
                return glib::Propagation::Proceed;
            };
            let was_playing = show.playing.get();
            if was_playing {
                show.stop_timer();
            }
            show.poke_chrome();
            arm_chrome_hide(&show);
            // Keys typed into the delay stepper belong to it: Up/Down step
            // the value natively (persisted via value-changed) and must not
            // double-handle as slideshow slower/faster.
            if spin_w.upgrade().map(|s| s.has_focus()).unwrap_or(false) {
                return glib::Propagation::Proceed;
            }
            match slideshow_key_action(keyval, was_playing) {
                None => glib::Propagation::Proceed,
                Some(SlideshowAction::Stop) => glib::Propagation::Stop,
                Some(SlideshowAction::Pop) => {
                    if let Some(page) = page_w.upgrade() {
                        let _ = page.activate_action("navigation.pop", None);
                    }
                    glib::Propagation::Stop
                }
                Some(SlideshowAction::TogglePlay) => {
                    show.set_playing(!was_playing);
                    glib::Propagation::Stop
                }
                Some(SlideshowAction::Next) => {
                    show.manual_step(true);
                    glib::Propagation::Stop
                }
                Some(SlideshowAction::Prev) => {
                    show.manual_step(false);
                    glib::Propagation::Stop
                }
                Some(SlideshowAction::Slower) => {
                    // The advance was already stopped above (any key stops
                    // the timer); storing must not restart it.
                    show.store_delay(show.delay.get().saturating_add(1));
                    glib::Propagation::Stop
                }
                Some(SlideshowAction::Faster) => {
                    show.store_delay(show.delay.get().saturating_sub(1));
                    glib::Propagation::Stop
                }
            }
        });
        outer.add_controller(keys);
    }

    // Clicks stop a running advance and keep the chrome up; they neither
    // step (the C does — see the module doc) nor pop.
    {
        let weak = Rc::downgrade(&show);
        let click = gtk4::GestureClick::new();
        click.connect_pressed(move |_, _, _, _| {
            let Some(show) = weak.upgrade() else { return };
            if show.playing.get() {
                show.stop_timer();
            }
            show.poke_chrome();
            arm_chrome_hide(&show);
        });
        outer.add_controller(click);
    }

    // Pointer motion re-shows the chrome and re-arms the hide (C
    // `mouse_moved`, slideshow.c:624-637).
    {
        let weak = Rc::downgrade(&show);
        let motion = gtk4::EventControllerMotion::new();
        motion.connect_motion(move |_, _, _| {
            let Some(show) = weak.upgrade() else { return };
            show.poke_chrome();
            arm_chrome_hide(&show);
        });
        outer.add_controller(motion);
    }

    // Leaving the page stops the advance — the analogue of the C's `leave`
    // clearing `auto_advance` (slideshow.c:523). `unmap` (not just pop)
    // covers minimize/hide too, mirroring the tether page's timer park.
    // Re-mapping only takes focus; a popped page is destroyed and never
    // maps again, so this can never resurrect a dead show.
    {
        // Strong capture: this is the ONE strong owner keeping `Show` alive
        // exactly as long as the page — every other closure holds only a
        // Weak. Page → closure → Show → descendant widgets is a DAG (GTK
        // child→parent refs are weak), so nothing cycles or leaks past pop.
        let show_life = show.clone();
        let map_weak = Rc::downgrade(&show);
        let outer_w = outer.downgrade();
        page.connect_unmap(move |_| {
            show_life.stop_timer();
        });
        page.connect_map(move |_| {
            let Some(show) = map_weak.upgrade() else {
                return;
            };
            if let Some(o) = outer_w.upgrade() {
                o.grab_focus();
            }
            show.poke_chrome();
            arm_chrome_hide(&show);
        });
    }
    arm_chrome_hide(&show);

    page
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_default_matches_the_c_config_default() {
        // Pinned against data/darktableconfig.xml.in (`slideshow_delay`
        // entry: <default>5</default>). If upstream ever changes it, this
        // test names the place to mirror.
        assert_eq!(SLIDESHOW_DELAY_DEFAULT, 5);
    }

    #[test]
    fn delay_clamp_pins_to_one_to_sixty() {
        // The C clamp: CLAMP(d->delay + value, 1, 60) (slideshow.c:189).
        assert_eq!(clamp_delay(5), 5);
        assert_eq!(clamp_delay(1), 1);
        assert_eq!(clamp_delay(60), 60);
        assert_eq!(clamp_delay(0), 1);
        assert_eq!(clamp_delay(-40), 1);
        assert_eq!(clamp_delay(61), 60);
        assert_eq!(clamp_delay(3600), 60);
    }

    #[test]
    fn parse_delay_falls_back_and_clamps() {
        assert_eq!(parse_delay(None), SLIDESHOW_DELAY_DEFAULT);
        assert_eq!(parse_delay(Some("")), SLIDESHOW_DELAY_DEFAULT);
        assert_eq!(parse_delay(Some("not-a-number")), SLIDESHOW_DELAY_DEFAULT);
        assert_eq!(parse_delay(Some("7")), 7);
        // A hand-edited pref is clamped, never trusted: 0 s would busy-loop
        // the advance, negative or huge values would stall it.
        assert_eq!(parse_delay(Some("0")), 1);
        assert_eq!(parse_delay(Some("-3")), 1);
        assert_eq!(parse_delay(Some("999")), 60);
        // Surrounding whitespace is trimmed, not rejected.
        assert_eq!(parse_delay(Some(" 10 ")), 10);
    }

    #[test]
    fn counter_text_is_one_based() {
        assert_eq!(counter_text(0, 12), "1 / 12");
        assert_eq!(counter_text(11, 12), "12 / 12");
        assert_eq!(counter_text(0, 1), "1 / 1");
        assert_eq!(counter_text(0, 0), "0 / 0");
    }

    #[test]
    fn delay_text_names_seconds() {
        assert_eq!(delay_text(5), "5 s");
        assert_eq!(delay_text(0), "1 s");
        assert_eq!(delay_text(99), "60 s");
    }

    #[test]
    fn step_clamps_at_both_ends_without_wrapping() {
        // Forward: the C logs "end of images" and stops at the last rank
        // (slideshow.c:357-361) — no wrap to the first image.
        assert_eq!(step_index(0, 5, true), Some(1));
        assert_eq!(step_index(3, 5, true), Some(4));
        assert_eq!(step_index(4, 5, true), None);
        // Back: same at the head (slideshow.c:378-382).
        assert_eq!(step_index(4, 5, false), Some(3));
        assert_eq!(step_index(0, 5, false), None);
        // Empty collection and out-of-range positions have nowhere to go.
        assert_eq!(step_index(0, 0, true), None);
        assert_eq!(step_index(0, 0, false), None);
        assert_eq!(step_index(9, 5, true), None);
        assert_eq!(step_index(9, 5, false), None);
    }

    #[test]
    fn resolve_start_empty_is_the_entry_no_op() {
        // The `win.slideshow` action pushes nothing when this is None —
        // the analogue of `try_enter` refusing an empty collection
        // (slideshow.c:417-429).
        assert_eq!(resolve_start(&[], None), None);
        assert_eq!(resolve_start(&[], Some("/a/b.jpg")), None);
    }

    #[test]
    fn resolve_start_prefers_the_selection_then_the_first_image() {
        let paths = vec!["/a/1.jpg".to_string(), "/a/2.jpg".to_string()];
        assert_eq!(resolve_start(&paths, Some("/a/2.jpg")), Some(1));
        // Selection outside the list (stale path): open on the first image,
        // the same fallback the C uses via the thumbtable offset.
        assert_eq!(resolve_start(&paths, Some("/gone.jpg")), Some(0));
        assert_eq!(resolve_start(&paths, None), Some(0));
    }

    #[test]
    fn slideshow_keys_match_the_c_bindings() {
        // C gui_init (slideshow.c:729-750): space start/stop, Esc exit,
        // Up slow down, Down speed up, Right/Left step.
        assert_eq!(
            slideshow_key_action(gdk::Key::space, false),
            Some(SlideshowAction::TogglePlay)
        );
        assert_eq!(
            slideshow_key_action(gdk::Key::Right, true),
            Some(SlideshowAction::Next)
        );
        assert_eq!(
            slideshow_key_action(gdk::Key::Left, true),
            Some(SlideshowAction::Prev)
        );
        assert_eq!(
            slideshow_key_action(gdk::Key::Up, false),
            Some(SlideshowAction::Slower)
        );
        assert_eq!(
            slideshow_key_action(gdk::Key::Down, false),
            Some(SlideshowAction::Faster)
        );
        // Keypad/numpad and shifted variants the C also binds
        // (slideshow.c:737-744): same polarity as Up/Down.
        assert_eq!(
            slideshow_key_action(gdk::Key::KP_Add, false),
            Some(SlideshowAction::Slower)
        );
        assert_eq!(
            slideshow_key_action(gdk::Key::plus, false),
            Some(SlideshowAction::Slower)
        );
        assert_eq!(
            slideshow_key_action(gdk::Key::KP_Subtract, false),
            Some(SlideshowAction::Faster)
        );
        assert_eq!(
            slideshow_key_action(gdk::Key::minus, false),
            Some(SlideshowAction::Faster)
        );
        // Two-stage Escape: stop a running advance first, pop only when
        // already stopped.
        assert_eq!(
            slideshow_key_action(gdk::Key::Escape, true),
            Some(SlideshowAction::Stop)
        );
        assert_eq!(
            slideshow_key_action(gdk::Key::Escape, false),
            Some(SlideshowAction::Pop)
        );
        // Unowned keys propagate — the page must not trap them.
        assert_eq!(slideshow_key_action(gdk::Key::a, false), None);
        assert_eq!(slideshow_key_action(gdk::Key::F5, true), None);
    }

    #[test]
    fn slideshow_tag_is_slash_free() {
        // Same contract as the print/map/tether tags: the lighttable
        // `popped` re-sync and the switcher mirror only handle slash tags.
        assert!(
            !SLIDESHOW_PAGE_TAG.contains('/'),
            "tag {SLIDESHOW_PAGE_TAG:?}"
        );
        assert_eq!(SLIDESHOW_PAGE_TAG, "slideshow");
    }
}
