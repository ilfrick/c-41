//! darktable-style "bauhaus" slider (parity audit 3.3, second half).
//!
//! darktable does not use stock GTK sliders. Its controls are custom-drawn: a
//! flat baseline bar spanning the row, a filled portion showing the value, a
//! small triangular indicator at the current position, and the **label drawn on
//! the left with the value on the right, inside the same rectangle** — no
//! separate label widget, no handle, no trough inset. The result is dense and
//! quiet, which is the point in a panel holding thirty of them.
//!
//! This is a faithful-enough port of that presentation over a
//! [`gtk4::DrawingArea`]. The geometry follows `src/bauhaus/bauhaus.c`:
//! `_draw_indicator_shape` draws a triangle of `sin = 0.866 * r`,
//! `cos = 0.5 * r` (an equilateral pointing down), and the baseline sits below
//! the text line with `INNER_PADDING` between elements.
//!
//! It deliberately does **not** reimplement bauhaus's popup editor, gradient
//! stops, soft/hard bounds or the right-hand "quad" button — those are a much
//! larger surface and none are needed by the current module rows.
//!
//! ## Why a custom widget and not CSS
//!
//! The shape is not stylable: GTK's `Scale` always renders trough + handle as
//! separate nodes and always reserves the handle's width. No stylesheet turns
//! that into "text and value inside a filled bar". Hence a `DrawingArea`.

use gtk4::glib;
use gtk4::prelude::*;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// Value-changed callbacks registered on a [`BauhausSlider`].
type Callbacks = Rc<RefCell<Vec<Box<dyn Fn(f64)>>>>;

/// Shared mutable state, so the draw function, gestures and the public setters
/// all see the same value.
struct State {
    label: String,
    min: f64,
    max: f64,
    step: f64,
    value: Cell<f64>,
    /// Decimal places in the read-out, derived from `step`.
    digits: usize,
}

impl State {
    /// Value as a 0..1 fraction of the range, for drawing.
    fn fraction(&self) -> f64 {
        if self.max <= self.min {
            return 0.0;
        }
        ((self.value.get() - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }

    /// Snap to `step` and clamp to the range — the invariant every entry point
    /// (drag, scroll, key, `set_value`) must go through.
    fn quantise(&self, v: f64) -> f64 {
        let v = v.clamp(self.min, self.max);
        if self.step > 0.0 {
            let snapped = (v / self.step).round() * self.step;
            snapped.clamp(self.min, self.max)
        } else {
            v
        }
    }
}

/// A darktable-style slider: label and value drawn inside a flat filled bar.
///
/// Mirrors the slice of `gtk4::Scale`'s API the module rows actually use
/// (`value`, `set_value`, `connect_value_changed`), so it drops into the
/// existing call sites.
#[derive(Clone)]
pub struct BauhausSlider {
    pub widget: gtk4::DrawingArea,
    state: Rc<State>,
    callbacks: Callbacks,
}

/// Row height. darktable's is `line_height + INNER_PADDING * 2`. 30px leaves
/// the text line clear of the baseline bar at our 11px font — at 26 the
/// descenders touched the bar.
const ROW_HEIGHT: i32 = 30;
/// Horizontal inset, mirroring bauhaus `INNER_PADDING` (4.0 uncondensed).
const PAD: f64 = 4.0;
/// Baseline bar thickness.
const BAR_H: f64 = 2.0;
/// Indicator triangle radius.
const MARKER_R: f64 = 4.0;

impl BauhausSlider {
    pub fn new(label: &str, min: f64, max: f64, step: f64, value: f64) -> Self {
        // Decimals from the step: 1.0 → 0, 0.1 → 1, 0.01 → 2 … capped at 3 so a
        // pathological step can't produce a 17-digit read-out.
        let digits = if step >= 1.0 {
            0
        } else if step > 0.0 {
            ((-step.log10().floor()) as usize).min(3)
        } else {
            2
        };
        let state = Rc::new(State {
            label: label.to_string(),
            min,
            max,
            step,
            value: Cell::new(value.clamp(min, max)),
            digits,
        });
        let callbacks: Callbacks = Rc::new(RefCell::new(Vec::new()));

        let widget = gtk4::DrawingArea::builder()
            .height_request(ROW_HEIGHT)
            .hexpand(true)
            .can_focus(true)
            .focusable(true)
            .build();
        widget.add_css_class("c41-bauhaus");

        // ── Drawing ──────────────────────────────────────────────────────
        {
            let st = state.clone();
            widget.set_draw_func(move |area, cr, w, h| {
                let w = w as f64;
                let h = h as f64;
                let frac = st.fraction();

                // Colours are read from the widget's style context so the theme
                // (crate::theme) stays the single source of the palette.
                let ctx = area.style_context();
                let fg = ctx.color();

                // Baseline: a full-width dim bar with the filled portion bright.
                let bar_y = h - PAD - BAR_H;
                cr.set_source_rgba(
                    fg.red() as f64,
                    fg.green() as f64,
                    fg.blue() as f64,
                    0.25,
                );
                cr.rectangle(PAD, bar_y, w - 2.0 * PAD, BAR_H);
                let _ = cr.fill();

                let fill_w = (w - 2.0 * PAD) * frac;
                cr.set_source_rgba(
                    fg.red() as f64,
                    fg.green() as f64,
                    fg.blue() as f64,
                    0.75,
                );
                cr.rectangle(PAD, bar_y, fill_w, BAR_H);
                let _ = cr.fill();

                // Indicator: equilateral triangle pointing down at the value,
                // per bauhaus.c `_draw_indicator_shape` (sin/cos of 60°).
                let cx = PAD + fill_w;
                let cy = bar_y;
                let sin = 0.866_025_4 * MARKER_R;
                let cos = 0.5 * MARKER_R;
                cr.move_to(cx, cy + MARKER_R);
                cr.line_to(cx - sin, cy - cos);
                cr.line_to(cx + sin, cy - cos);
                cr.close_path();
                let _ = cr.fill();

                // Label (left) and value (right) on the text line above the bar.
                cr.set_source_rgba(
                    fg.red() as f64,
                    fg.green() as f64,
                    fg.blue() as f64,
                    1.0,
                );
                cr.select_font_face(
                    "sans-serif",
                    gtk4::cairo::FontSlant::Normal,
                    gtk4::cairo::FontWeight::Normal,
                );
                cr.set_font_size(11.0);
                // Sit the baseline of the text a full padding above the bar so
                // descenders (g, y, p) clear it.
                let text_y = bar_y - PAD * 1.5;
                cr.move_to(PAD, text_y);
                let _ = cr.show_text(&st.label);

                let val = format!("{:.*}", st.digits, st.value.get());
                if let Ok(ext) = cr.text_extents(&val) {
                    cr.move_to(w - PAD - ext.width(), text_y);
                    let _ = cr.show_text(&val);
                }
            });
        }

        let slider = Self { widget, state, callbacks };

        // ── Interaction ──────────────────────────────────────────────────
        // Drag and click both map x → value, so a click jumps and a drag
        // scrubs, matching bauhaus.
        {
            let s = slider.clone();
            let drag = gtk4::GestureDrag::new();
            let start_x = Rc::new(Cell::new(0.0));
            {
                let s = s.clone();
                let start_x = start_x.clone();
                drag.connect_drag_begin(move |g, x, _| {
                    start_x.set(x);
                    let w = g.widget().map(|w| w.width()).unwrap_or(1) as f64;
                    s.set_from_x(x, w);
                });
            }
            {
                let s = s.clone();
                let start_x = start_x.clone();
                drag.connect_drag_update(move |g, dx, _| {
                    let w = g.widget().map(|w| w.width()).unwrap_or(1) as f64;
                    s.set_from_x(start_x.get() + dx, w);
                });
            }
            slider.widget.add_controller(drag);
        }
        {
            // Scroll: one step per notch, the fine-adjust bauhaus offers.
            let s = slider.clone();
            let scroll = gtk4::EventControllerScroll::new(
                gtk4::EventControllerScrollFlags::VERTICAL,
            );
            scroll.connect_scroll(move |_, _, dy| {
                let st = &s.state;
                let delta = if st.step > 0.0 { st.step } else { (st.max - st.min) / 100.0 };
                // dy > 0 is scroll-down, which should decrease.
                s.set_value(st.value.get() - dy * delta);
                glib::Propagation::Stop
            });
            slider.widget.add_controller(scroll);
        }
        {
            // Keyboard: arrows step, Home/End jump to the bounds.
            let s = slider.clone();
            let keys = gtk4::EventControllerKey::new();
            keys.connect_key_pressed(move |_, key, _, _| {
                let st = &s.state;
                let delta = if st.step > 0.0 { st.step } else { (st.max - st.min) / 100.0 };
                let v = st.value.get();
                match key {
                    gtk4::gdk::Key::Left | gtk4::gdk::Key::Down => s.set_value(v - delta),
                    gtk4::gdk::Key::Right | gtk4::gdk::Key::Up => s.set_value(v + delta),
                    gtk4::gdk::Key::Home => s.set_value(st.min),
                    gtk4::gdk::Key::End => s.set_value(st.max),
                    _ => return glib::Propagation::Proceed,
                }
                glib::Propagation::Stop
            });
            slider.widget.add_controller(keys);
        }

        slider
    }

    /// Map a pointer x within a widget of width `w` to a value.
    fn set_from_x(&self, x: f64, w: f64) {
        let usable = (w - 2.0 * PAD).max(1.0);
        let frac = ((x - PAD) / usable).clamp(0.0, 1.0);
        let st = &self.state;
        self.set_value(st.min + frac * (st.max - st.min));
    }

    /// The current value.
    pub fn value(&self) -> f64 {
        self.state.value.get()
    }

    /// Set the value (snapped and clamped), redraw, and notify listeners.
    ///
    /// No-ops when the quantised value is unchanged, so a drag that stays
    /// within one step does not spam `render_preview` — the same de-duplication
    /// `gtk4::Scale` gives for free.
    pub fn set_value(&self, v: f64) {
        let q = self.state.quantise(v);
        if q == self.state.value.get() {
            return;
        }
        self.state.value.set(q);
        self.widget.queue_draw();
        for cb in self.callbacks.borrow().iter() {
            cb(q);
        }
    }

    /// Register a value-changed callback.
    pub fn connect_value_changed<F: Fn(f64) + 'static>(&self, f: F) {
        self.callbacks.borrow_mut().push(Box::new(f));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise the pure value logic, which needs no display — the same
    // display-free discipline the rest of the UI crate follows.
    fn st(min: f64, max: f64, step: f64, value: f64) -> State {
        State {
            label: "x".into(),
            min,
            max,
            step,
            value: Cell::new(value),
            digits: 2,
        }
    }

    #[test]
    fn quantise_snaps_and_clamps() {
        let s = st(0.0, 10.0, 0.5, 0.0);
        assert_eq!(s.quantise(3.26), 3.5);
        assert_eq!(s.quantise(3.24), 3.0);
        assert_eq!(s.quantise(-5.0), 0.0, "clamps to min");
        assert_eq!(s.quantise(99.0), 10.0, "clamps to max");
    }

    #[test]
    fn quantise_never_escapes_the_range() {
        // Rounding at the top edge must not push past max: with max=1.0 and
        // step=0.3, round(1.0/0.3)*0.3 = 1.2, which would overshoot.
        let s = st(0.0, 1.0, 0.3, 0.0);
        assert!(s.quantise(1.0) <= 1.0, "snap overshot max: {}", s.quantise(1.0));
        let neg = st(-1.0, 1.0, 0.3, 0.0);
        assert!(neg.quantise(-1.0) >= -1.0, "snap undershot min");
    }

    #[test]
    fn fraction_maps_range_to_unit_interval() {
        assert_eq!(st(0.0, 10.0, 0.1, 0.0).fraction(), 0.0);
        assert_eq!(st(0.0, 10.0, 0.1, 10.0).fraction(), 1.0);
        assert_eq!(st(0.0, 10.0, 0.1, 5.0).fraction(), 0.5);
        // Negative ranges (e.g. EV -3..3) must map linearly too.
        assert_eq!(st(-3.0, 3.0, 0.1, 0.0).fraction(), 0.5);
    }

    #[test]
    fn fraction_survives_a_degenerate_range() {
        // min == max would divide by zero and put NaN into the draw path.
        let s = st(1.0, 1.0, 0.1, 1.0);
        assert_eq!(s.fraction(), 0.0);
        assert!(s.fraction().is_finite());
    }
}
