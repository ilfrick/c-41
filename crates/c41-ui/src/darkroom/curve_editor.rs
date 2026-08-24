//! Interactive L-curve editor shared by the Tone curve (m4-122), RGB curve
//! (m4-123) and Base curve (m4-124) modules.
//!
//! A `DrawingArea` painting the curve box — grid, dashed identity diagonal,
//! the spline itself and draggable anchor nodes — plus click/drag handlers
//! implementing darktable's basic curve-widget gestures:
//!
//! - **click a node** → grab it (drag to move; endpoints keep x pinned 0/1,
//!   interior nodes stay strictly between their neighbours)
//! - **click empty space** → insert an anchor (x-order preserved, ≤ 20 nodes)
//! - **double-click a node** → remove it (interior nodes only, ≥ 2 remain)
//!
//! One builder serves both modules because the interaction is identical; what
//! differs is the number of channels (1 vs 3), the stroke colours and which
//! params fields back each channel. Those differences are injected as closures
//! ([`TypeFn`] reads a channel's spline type, [`SyncFn`] writes a channel's
//! anchors back into params), so the gesture code exists exactly once.
//!
//! The drawn curve is sampled through [`c41_core::curve_tools::curve_data_sample`]
//! — the *same* sampler `PreviewParams::to_pipeline` uses to build the LUTs —
//! so what the user sees is exactly what the pipeline applies.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use adw::prelude::*;
use gtk4::{DrawingArea, EventControllerMotion, GestureClick};

use c41_core::curve_tools;

use super::PreviewCtx;
use crate::preview::PreviewParams;

/// Widget padding around the plot box, in px.
const PAD: f64 = 10.0;
/// Node grab radius in widget px.
const HIT_PX: f64 = 10.0;
/// Minimum x separation between interior neighbouring anchors while dragging.
const X_EPS: f32 = 1e-4;
/// Curve samples drawn per repaint.
const DRAW_SAMPLES: usize = 256;
/// Maximum anchors per channel (the encoded-blob slot count).
const MAX_ANCHORS: usize = 20;

/// Stroke + grab-ring colours for one channel's curve.
#[derive(Clone, Copy)]
struct ChannelStyle {
    line: (f64, f64, f64),
    ring: (f64, f64, f64),
}

/// Tone curve: the single Lab L channel, amber.
const TONECURVE_STYLE: ChannelStyle = ChannelStyle {
    line: (0.98, 0.72, 0.11),
    ring: (0.98, 0.72, 0.11),
};

/// RGB curve: per-channel colours matching the R/G/B lanes they remap.
const RGBCURVE_STYLES: [ChannelStyle; 3] = [
    ChannelStyle {
        line: (0.94, 0.25, 0.25),
        ring: (0.94, 0.25, 0.25),
    },
    ChannelStyle {
        line: (0.30, 0.85, 0.35),
        ring: (0.30, 0.85, 0.35),
    },
    ChannelStyle {
        line: (0.35, 0.55, 0.98),
        ring: (0.35, 0.55, 0.98),
    },
];

/// Base curve (m4-124): one neutral channel — C paints this module's curve in
/// flat grey rather than a lane colour.
const BASECURVE_STYLE: ChannelStyle = ChannelStyle {
    line: (0.9, 0.9, 0.9),
    ring: (0.9, 0.9, 0.9),
};

/// The plot rectangle inside the widget: `[x, x+w) × [y, y+h)`, with curve
/// coordinate (0,0) mapping to its bottom-left corner (y flipped).
#[derive(Clone, Copy, Debug, PartialEq)]
struct PlotRect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl PlotRect {
    fn from_widget_size(w: i32, h: i32) -> Self {
        Self {
            x: PAD,
            y: PAD,
            w: (w as f64 - 2.0 * PAD).max(1.0),
            h: (h as f64 - 2.0 * PAD).max(1.0),
        }
    }

    /// Curve coords [0,1]² → widget px (y flips: v=0 is the bottom edge).
    fn to_widget(self, u: f32, v: f32) -> (f64, f64) {
        (
            self.x + u as f64 * self.w,
            self.y + self.h - v as f64 * self.h,
        )
    }

    /// Widget px → unclamped curve coords.
    fn to_curve(self, wx: f64, wy: f64) -> (f64, f64) {
        (
            (wx - self.x) / self.w,
            (self.y + self.h - wy) / self.h,
        )
    }
}

/// Insert `(x, y)` keeping the list x-sorted. Returns the new index, or `None`
/// when the list already holds [`MAX_ANCHORS`] nodes or the (box-clamped) x
/// collides with a neighbour — every spline rejects non-increasing anchors, so
/// a duplicate x would silently snap the whole curve to the identity diagonal
/// while the nodes stay visible.
fn insert_node(nodes: &mut Vec<(f32, f32)>, x: f32, y: f32) -> Option<usize> {
    if nodes.len() >= MAX_ANCHORS {
        return None;
    }
    // Collide-check the CLAMPED x: that is what gets stored, and a raw x
    // outside [0,1] clamps onto an endpoint's column.
    let x = x.clamp(0.0, 1.0);
    let pos = nodes.partition_point(|&(nx, _)| nx <= x);
    let collides = (pos > 0 && (x - nodes[pos - 1].0).abs() <= X_EPS)
        || (pos < nodes.len() && (nodes[pos].0 - x).abs() <= X_EPS);
    if collides {
        return None;
    }
    nodes.insert(pos, (x, y.clamp(0.0, 1.0)));
    Some(pos)
}

/// Clamp a dragged node's x: endpoints are pinned to 0/1 exactly, interior
/// nodes stay strictly between their (current) neighbours.
fn clamp_drag_x(nodes: &[(f32, f32)], idx: usize, mut x: f32) -> f32 {
    if idx == 0 {
        return 0.0;
    }
    if idx == nodes.len() - 1 {
        return 1.0;
    }
    x = x.clamp(0.0, 1.0);
    x.max(nodes[idx - 1].0 + X_EPS).min(nodes[idx + 1].0 - X_EPS)
}

/// Nearest node within `tol` curve units of `(u, v)`, or `None`.
fn hit_node(nodes: &[(f32, f32)], u: f32, v: f32, tol: f32) -> Option<usize> {
    let d2 = |n: (f32, f32)| (n.0 - u).powi(2) + (n.1 - v).powi(2);
    let (best_idx, best_d2) = nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, n)| (i, d2(n)))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    (best_d2 <= tol * tol).then_some(best_idx)
}

/// Remove node `idx` if it is interior and at least 2 anchors would remain.
fn remove_node(nodes: &mut Vec<(f32, f32)>, idx: usize) -> bool {
    match nodes.len().checked_sub(1) {
        Some(last) if idx != 0 && idx != last && nodes.len() > 2 => {
            nodes.remove(idx);
            true
        }
        _ => false,
    }
}

/// Pack the live anchors into the fixed-width param array (tail zeroed), so
/// the encoded blob layout stays stable regardless of how many are in use.
fn nodes_to_array(nodes: &[(f32, f32)]) -> [(f32, f32); MAX_ANCHORS] {
    let mut arr = [(0.0f32, 0.0f32); MAX_ANCHORS];
    for (slot, n) in arr.iter_mut().zip(nodes.iter()) {
        *slot = *n;
    }
    arr
}

// ── RGB-curve param plumbing (pure, unit-tested) ────────────────────────────

/// Copy `nodes` into channel `ch`'s fixed-width array + count (0 = R, 1 = G,
/// 2 = B). Single write site for the editor's sync callback, so the array and
/// its count can never disagree.
fn set_channel_nodes(p: &mut PreviewParams, ch: usize, nodes: &[(f32, f32)]) {
    let arr = nodes_to_array(nodes);
    let count = nodes.len() as f32;
    match ch {
        0 => {
            p.rc_nodes_r = arr;
            p.rc_nnodes_r = count;
        }
        1 => {
            p.rc_nodes_g = arr;
            p.rc_nnodes_g = count;
        }
        _ => {
            p.rc_nodes_b = arr;
            p.rc_nnodes_b = count;
        }
    }
}

/// A channel's anchor array + live count, for seeding the editor.
fn channel_nodes(p: &PreviewParams, ch: usize) -> (&[(f32, f32); MAX_ANCHORS], f32) {
    match ch {
        0 => (&p.rc_nodes_r, p.rc_nnodes_r),
        1 => (&p.rc_nodes_g, p.rc_nnodes_g),
        _ => (&p.rc_nodes_b, p.rc_nnodes_b),
    }
}

// ── Shared multi-channel editor ──────────────────────────────────────────────

/// Reads a channel's spline type from the live params (the drawn curve must
/// follow the interpolator dropdown without a rebuild).
type TypeFn = Rc<dyn Fn(&PreviewParams, usize) -> u32>;
/// Writes a channel's anchors back into the shared params before re-rendering.
type SyncFn = Rc<dyn Fn(&PreviewCtx, usize, &[(f32, f32)])>;

/// Shared editor state: per-channel anchor lists plus which node (if any) each
/// channel's drag holds, and which channel the gestures edit / the canvas paints.
struct MultiCurveState {
    channels: Vec<(Rc<RefCell<Vec<(f32, f32)>>>, Rc<Cell<Option<usize>>>)>,
    /// Index into [`MultiCurveState::channels`] (bounded by the selector UI).
    active: Rc<Cell<usize>>,
}

impl MultiCurveState {
    /// The active channel, clamped so an out-of-range index can never panic.
    fn active_ch(&self) -> usize {
        self.active.get().min(self.channels.len().saturating_sub(1))
    }
}

/// Build the interactive curve canvas for `state`: draw func painting the
/// ACTIVE channel through its own style/spline-type, and click/motion/release
/// gestures routed at the active channel. Every mutation goes through `sync`
/// (which writes params) followed by a re-render.
fn multi_curve_area(
    ctx: &PreviewCtx,
    state: &Rc<MultiCurveState>,
    styles: Vec<ChannelStyle>,
    type_of: TypeFn,
    sync: SyncFn,
) -> DrawingArea {
    let area = DrawingArea::builder()
        .content_width(240)
        .content_height(170)
        .hexpand(true)
        .build();
    area.add_css_class("card");
    area.set_margin_top(6);
    area.set_margin_bottom(6);
    area.set_margin_start(6);
    area.set_margin_end(6);

    // Draw func: captures ctx (to read the live spline type), the shared
    // state and the styles. No reference back to `area`, so no ownership cycle.
    {
        let draw_ctx = ctx.clone();
        let draw_state = state.clone();
        let mut draw_styles = styles.clone();
        let type_of = type_of.clone();
        area.set_draw_func(move |_, cr, w, h| {
            let rect = PlotRect::from_widget_size(w, h);
            let ch = draw_state.active_ch();
            // Defensive against a styles slice shorter than the channel count
            // (all call sites ship matching lengths; this keeps indexing total).
            while draw_styles.len() < draw_state.channels.len() {
                draw_styles.push(*draw_styles.last().unwrap_or(&TONECURVE_STYLE));
            }
            let (nodes, drag) = &draw_state.channels[ch];
            let spline_type = type_of(&draw_ctx.params.borrow(), ch);
            let nodes = nodes.borrow();
            draw_curve(cr, rect, &nodes, spline_type, drag.get(), &draw_styles[ch]);
        });
    }

    // Gestures: press grabs/inserts/removes on the ACTIVE channel, motion moves
    // the grabbed node, release drops it. Every mutation syncs params + repaints.
    let click = GestureClick::new();
    {
        let st = state.clone();
        let edit_ctx = ctx.clone();
        let edit_area = area.downgrade();
        let sync = sync.clone();
        click.connect_pressed(move |_, n_press, wx, wy| {
            let rect = current_rect(&edit_area);
            let Some(rect) = rect else { return };
            let ch = st.active_ch();
            let (nodes_cell, drag_slot) = st.channels[ch].clone();
            let (u, v) = rect.to_curve(wx, wy);
            // Double-click on a node removes it (interior only).
            if n_press >= 2 {
                let hit = {
                    let nodes = nodes_cell.borrow();
                    hit_node(&nodes, u as f32, v as f32, hit_tol(rect))
                };
                if let Some(idx) = hit {
                    let removed = remove_node(&mut nodes_cell.borrow_mut(), idx);
                    drag_slot.set(None);
                    if removed {
                        let nodes = nodes_cell.borrow();
                        sync(&edit_ctx, ch, &nodes);
                        drop(nodes);
                        if let Some(a) = edit_area.upgrade() {
                            a.queue_draw();
                        }
                        super::render_preview(&edit_ctx);
                    }
                }
                return;
            }
            // Single press: grab a nearby node, or plant a new one here.
            let hit = {
                let nodes = nodes_cell.borrow();
                hit_node(&nodes, u as f32, v as f32, hit_tol(rect))
            };
            let target = match hit {
                Some(idx) => Some(idx),
                None => {
                    // Only accept inserts that land inside the plot box.
                    if (0.0..=1.0).contains(&(u as f32)) && (0.0..=1.0).contains(&(v as f32)) {
                        insert_node(&mut nodes_cell.borrow_mut(), u as f32, v as f32)
                    } else {
                        None
                    }
                }
            };
            drag_slot.set(target);
            if target.is_some() {
                let nodes = nodes_cell.borrow();
                sync(&edit_ctx, ch, &nodes);
                drop(nodes);
                if let Some(a) = edit_area.upgrade() {
                    a.queue_draw();
                }
                super::render_preview(&edit_ctx);
            }
        });
    }
    {
        let st = state.clone();
        let end_area = area.downgrade();
        click.connect_released(move |_, _, _, _| {
            for (_, drag_slot) in &st.channels {
                drag_slot.set(None);
            }
            if let Some(a) = end_area.upgrade() {
                a.queue_draw();
            }
        });
    }
    area.add_controller(click);

    let motion = EventControllerMotion::new();
    {
        let st = state.clone();
        let move_ctx = ctx.clone();
        let move_area = area.downgrade();
        motion.connect_motion(move |_, wx, wy| {
            let ch = st.active_ch();
            let (nodes_cell, drag_slot) = st.channels[ch].clone();
            let Some(_idx) = drag_slot.get() else { return };
            let Some(rect) = current_rect(&move_area) else { return };
            let (u, v) = rect.to_curve(wx, wy);
            {
                let mut nodes = nodes_cell.borrow_mut();
                let Some(idx) = drag_slot.get() else { return };
                let clamped_x = clamp_drag_x(&nodes, idx, u as f32);
                nodes[idx] = (
                    clamped_x,
                    (v as f32).clamp(0.0, 1.0),
                );
                sync(&move_ctx, ch, &nodes);
            }
            if let Some(a) = move_area.upgrade() {
                a.queue_draw();
            }
            super::render_preview(&move_ctx);
        });
    }
    area.add_controller(motion);

    area
}

/// Seed a channel's live anchor list from the params (first `count` slots).
fn seed_nodes(p: &PreviewParams, ch: usize) -> Vec<(f32, f32)> {
    let (arr, count) = channel_nodes(p, ch);
    arr[..(count.round() as usize).clamp(2, MAX_ANCHORS)].to_vec()
}

/// Build the Tone curve module row: enable switch, interpolator selector and
/// the interactive L-curve editor (single Lab L channel).
pub(crate) fn tonecurve_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    let state = Rc::new(MultiCurveState {
        channels: vec![(
            Rc::new(RefCell::new(seed_nodes(&p0, 0))),
            Rc::new(Cell::new(None)),
        )],
        active: Rc::new(Cell::new(0)),
    });

    // Spline type: one field drives the (only) channel.
    let type_of: TypeFn = Rc::new(|p, _| p.tc_type.round() as i32 as u32);
    // Anchors live in tc_nodes_l / tc_nnodes.
    let sync: SyncFn = Rc::new(|ctx, _, nodes| {
        let mut p = ctx.params.borrow_mut();
        p.tc_nodes_l = nodes_to_array(nodes);
        p.tc_nnodes = nodes.len() as f32;
    });
    let area = multi_curve_area(ctx, &state, vec![TONECURVE_STYLE], type_of, sync);

    // ── Module row assembly ────────────────────────────────────────────────
    let expander = super::module_expander(ctx, "Tone curve", "Lab L curve", p0.tc_on,
        |p, on| p.tc_on = on,
        |e, ctx| {
            // Interpolator dropdown mirrors CurveData.m_spline_type. Seed the
            // selection BEFORE connecting the handler (rebuild invariant).
            let labels = ["cubic spline", "Catmull-Rom", "monotone Hermite"];
            let p0 = *ctx.params.borrow();
            let interp = adw::ComboRow::builder()
                .title("Interpolator")
                .model(&gtk4::StringList::new(&labels))
                .selected((p0.tc_type.round() as usize).min(labels.len() - 1) as u32)
                .build();
            let type_ctx = ctx.clone();
            let type_area = area.downgrade();
            interp.connect_selected_notify(move |row| {
                type_ctx.params.borrow_mut().tc_type = row.selected() as f32;
                if let Some(a) = type_area.upgrade() {
                    a.queue_draw(); // the drawn spline shape changes too
                }
                super::render_preview(&type_ctx);
            });
            e.add_row(&interp);
            e.add_row(&area);
        });
    expander
}

/// Build the RGB curve module row: enable switch, one interpolator selector
/// (writing all three channel types, mirroring C's `interpolator_callback`),
/// channel-mode + colour-preservation selectors, an R/G/B channel picker and
/// the interactive editor.
pub(crate) fn rgbcurve_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    let state = Rc::new(MultiCurveState {
        channels: (0..3)
            .map(|ch| {
                (
                    Rc::new(RefCell::new(seed_nodes(&p0, ch))),
                    Rc::new(Cell::new(None)),
                )
            })
            .collect(),
        active: Rc::new(Cell::new(0)),
    });

    // Per-channel spline types; anchors route through set_channel_nodes.
    let type_of: TypeFn = Rc::new(|p, ch| match ch {
        0 => p.rc_type_r.round() as i32 as u32,
        1 => p.rc_type_g.round() as i32 as u32,
        _ => p.rc_type_b.round() as i32 as u32,
    });
    let sync: SyncFn = Rc::new(|ctx, ch, nodes| {
        set_channel_nodes(&mut ctx.params.borrow_mut(), ch, nodes);
    });
    let area = multi_curve_area(ctx, &state, RGBCURVE_STYLES.to_vec(), type_of.clone(), sync);

    let expander = super::module_expander(ctx, "RGB curve", "R/G/B curves", p0.rc_on,
        |p, on| p.rc_on = on,
        move |e, ctx| {
            let p0 = *ctx.params.borrow();

            // ONE interpolator dropdown for all three channels — C's
            // interpolator_callback sets every curve's m_spline_type together.
            let labels = ["cubic spline", "Catmull-Rom", "monotone Hermite"];
            let interp = adw::ComboRow::builder()
                .title("Interpolator")
                .model(&gtk4::StringList::new(&labels))
                .selected((p0.rc_type_r.round() as usize).min(labels.len() - 1) as u32)
                .build();
            {
                let type_ctx = ctx.clone();
                let type_area = area.downgrade();
                interp.connect_selected_notify(move |row| {
                    let t = row.selected() as f32;
                    {
                        let mut p = type_ctx.params.borrow_mut();
                        p.rc_type_r = t;
                        p.rc_type_g = t;
                        p.rc_type_b = t;
                    }
                    if let Some(a) = type_area.upgrade() {
                        a.queue_draw(); // the drawn spline shape changes too
                    }
                    super::render_preview(&type_ctx);
                });
            }

            // Channel linking: AUTOMATIC_RGB (the R curve drives all channels)
            // vs MANUAL_RGB (independent per-channel curves).
            let mode_labels = ["linked", "independent"];
            let mode = adw::ComboRow::builder()
                .title("Channel mode")
                .model(&gtk4::StringList::new(&mode_labels))
                .selected((p0.rc_autoscale.round() as usize).min(mode_labels.len() - 1) as u32)
                .build();

            // Colour-preservation norm for linked mode (DT_RGB_NORM_*). Meaningless
            // in independent mode, so it greys out there (C hides the combo too).
            let norm_labels =
                ["none", "luminance", "max", "average", "sum", "norm", "power"];
            let norm = adw::ComboRow::builder()
                .title("Preserve colors")
                .model(&gtk4::StringList::new(&norm_labels))
                .selected((p0.rc_preserve.round() as usize).min(norm_labels.len() - 1) as u32)
                .sensitive(p0.rc_autoscale.round() as i32 == 0)
                .build();

            // Which channel the canvas paints + the gestures edit. In linked
            // mode only R is editable — rgbcurve.c:855 ("if autoscale is on:
            // do not display g and b curves") greys/hides them there too.
            let chan_labels = ["R", "G", "B"];
            let manual0 = p0.rc_autoscale.round() as i32 == 1;
            let chan = adw::ComboRow::builder()
                .title("Channel")
                .model(&gtk4::StringList::new(&chan_labels))
                .selected(0)
                .sensitive(manual0)
                .build();
            {
                let st = state.clone();
                let chan_area = area.downgrade();
                chan.connect_selected_notify(move |row| {
                    st.active.set(row.selected() as usize);
                    if let Some(a) = chan_area.upgrade() {
                        a.queue_draw();
                    }
                });
            }

            {
                let mode_ctx = ctx.clone();
                let mode_area = area.downgrade();
                let norm_for_mode = norm.clone();
                let chan_for_mode = chan.clone();
                mode.connect_selected_notify(move |row| {
                    let manual = row.selected() == 1;
                    mode_ctx.params.borrow_mut().rc_autoscale = row.selected() as f32;
                    norm_for_mode.set_sensitive(!manual);
                    // Linked mode edits R only; snap the picker back so an
                    // inert G/B curve is never shown as if it were live.
                    if manual {
                        chan_for_mode.set_sensitive(true);
                    } else {
                        chan_for_mode.set_selected(0);
                        chan_for_mode.set_sensitive(false);
                    }
                    if let Some(a) = mode_area.upgrade() {
                        a.queue_draw();
                    }
                    super::render_preview(&mode_ctx);
                });
                let norm_ctx = ctx.clone();
                norm.connect_selected_notify(move |row| {
                    norm_ctx.params.borrow_mut().rc_preserve = row.selected() as f32;
                    super::render_preview(&norm_ctx);
                });
            }

            e.add_row(&interp);
            e.add_row(&mode);
            e.add_row(&norm);
            e.add_row(&chan);
            e.add_row(&area);
        });
    expander
}

/// Build the Base curve module row (m4-124): enable switch, colour-preservation
/// selector, exposure-fusion selector and the interactive single-channel
/// editor.
///
/// Mirrors basecurve.c's gui_init control order — preserve-colors (:1952), then
/// fusion (:1955-58) with its two dependent sliders. Those sliders exist but
/// stay hidden while fusion is "none" (gui_init :1966/:1976, gui_update
/// :1323-24); we toggle their row visibility the same way. Two deliberate
/// omissions: C's GUI has NO interpolator dropdown for this module (the spline
/// type rides in params, defaulting to monotone Hermite — we still *read*
/// bc_type so a decoded blob with another type draws correctly), and the
/// log-base graph-scale slider is display-only. The canvas goes last to match
/// the sibling curve modules' layout.
pub(crate) fn basecurve_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    // Single channel; channel_nodes/seed_nodes are rgbcurve-specific, so seed
    // from the bc_ fields directly (first `count` slots).
    let state = Rc::new(MultiCurveState {
        channels: vec![(
            Rc::new(RefCell::new(
                p0.bc_nodes[..(p0.bc_nnodes.round() as usize).clamp(2, MAX_ANCHORS)].to_vec(),
            )),
            Rc::new(Cell::new(None)),
        )],
        active: Rc::new(Cell::new(0)),
    });

    let type_of: TypeFn = Rc::new(|p, _| p.bc_type.round() as i32 as u32);
    let sync: SyncFn = Rc::new(|ctx, _, nodes| {
        let mut p = ctx.params.borrow_mut();
        p.bc_nodes = nodes_to_array(nodes);
        p.bc_nnodes = nodes.len() as f32;
    });
    let area = multi_curve_area(ctx, &state, vec![BASECURVE_STYLE], type_of, sync);

    let expander = super::module_expander(ctx, "Base curve", "scene→display curve", p0.bc_on,
        |p, on| p.bc_on = on,
        move |e, ctx| {
            let p0 = *ctx.params.borrow();

            // Colour-preservation norm (DT_RGB_NORM_*). Always sensitive here —
            // unlike rgbcurve it is not gated behind a channel-link mode.
            let norm_labels =
                ["none", "luminance", "max", "average", "sum", "norm", "power"];
            let norm = adw::ComboRow::builder()
                .title("Preserve colors")
                .model(&gtk4::StringList::new(&norm_labels))
                .selected((p0.bc_preserve.round() as usize).min(norm_labels.len() - 1) as u32)
                .build();
            {
                let norm_ctx = ctx.clone();
                norm.connect_selected_notify(move |row| {
                    norm_ctx.params.borrow_mut().bc_preserve = row.selected() as f32;
                    super::render_preview(&norm_ctx);
                });
            }

            // Exposure fusion: none / two / three exposures (basecurve.c:1956-58).
            // Non-zero switches the pipeline stage to the Laplacian-pyramid blend.
            let fusion_labels = ["none", "two exposures", "three exposures"];
            let fusion = adw::ComboRow::builder()
                .title("Exposure fusion")
                .model(&gtk4::StringList::new(&fusion_labels))
                .selected((p0.bc_exposure_fusion.round() as usize).min(fusion_labels.len() - 1)
                    as u32)
                .build();

            // The two fusion sliders (stops 0.01..4.0, bias -1..1 per the C
            // introspection ranges). Built via labeled_slider directly because
            // add_param_slider does not return the widget and their rows must
            // start hidden when fusion is "none".
            let stops = super::labeled_slider("Exposure shift", 0.01, 4.0, 0.01,
                p0.bc_exposure_stops as f64);
            {
                let stops_ctx = ctx.clone();
                stops.scale.connect_value_changed(move |v| {
                    stops_ctx.params.borrow_mut().bc_exposure_stops = v as f32;
                    super::render_preview(&stops_ctx);
                });
            }
            let bias = super::labeled_slider("Exposure bias", -1.0, 1.0, 0.01,
                p0.bc_exposure_bias as f64);
            {
                let bias_ctx = ctx.clone();
                bias.scale.connect_value_changed(move |v| {
                    bias_ctx.params.borrow_mut().bc_exposure_bias = v as f32;
                    super::render_preview(&bias_ctx);
                });
            }
            // gui_init shows them only when a fusion mode is active.
            let fusion_active = p0.bc_exposure_fusion.round() as i32 != 0;
            stops.row.set_visible(fusion_active);
            bias.row.set_visible(fusion_active);

            {
                let fusion_ctx = ctx.clone();
                let stops_row = stops.row.clone();
                let bias_row = bias.row.clone();
                fusion.connect_selected_notify(move |row| {
                    fusion_ctx.params.borrow_mut().bc_exposure_fusion = row.selected() as f32;
                    let show = row.selected() != 0;
                    stops_row.set_visible(show);
                    bias_row.set_visible(show);
                    super::render_preview(&fusion_ctx);
                });
            }

            e.add_row(&norm);
            e.add_row(&fusion);
            e.add_row(&stops.row);
            e.add_row(&bias.row);
            e.add_row(&area);
        });
    expander
}

/// Current plot rect for gesture callbacks (`None` while the widget is gone or
/// not yet allocated).
fn current_rect(area: &glib::WeakRef<DrawingArea>) -> Option<PlotRect> {
    let a = area.upgrade()?;
    let (w, h) = (a.width(), a.height());
    (w > 0 && h > 0).then(|| PlotRect::from_widget_size(w, h))
}

/// Grab tolerance converted to curve units along the wider axis.
fn hit_tol(rect: PlotRect) -> f32 {
    (HIT_PX / rect.w.min(rect.h)) as f32
}

/// Paint the curve box: backdrop, quarter grid, dashed identity diagonal, the
/// spline polyline (sampled through the pipeline's own sampler) and the anchors.
/// `drag_idx` — when set — marks the grabbed node with a highlight ring.
fn draw_curve(
    cr: &gtk4::cairo::Context,
    rect: PlotRect,
    nodes: &[(f32, f32)],
    spline_type: u32,
    drag_idx: Option<usize>,
    style: &ChannelStyle,
) {
    // Backdrop.
    cr.set_source_rgb(0.10, 0.10, 0.10);
    cr.rectangle(rect.x, rect.y, rect.w, rect.h);
    let _ = cr.fill();

    // Quarter grid.
    cr.set_line_width(1.0);
    cr.set_source_rgba(1.0, 1.0, 1.0, 0.10);
    for k in 1..4 {
        let t = k as f64 / 4.0;
        let (gx, _) = rect.to_widget(t as f32, 0.0);
        let (_, gy) = rect.to_widget(0.0, t as f32);
        cr.move_to(gx, rect.y);
        cr.line_to(gx, rect.y + rect.h);
        let _ = cr.stroke();
        cr.move_to(rect.x, gy);
        cr.line_to(rect.x + rect.w, gy);
        let _ = cr.stroke();
    }

    // Dashed identity diagonal.
    cr.set_source_rgba(1.0, 1.0, 1.0, 0.25);
    cr.set_dash(&[4.0, 4.0], 0.0);
    let (x0, y0) = rect.to_widget(0.0, 0.0);
    let (x1, y1) = rect.to_widget(1.0, 1.0);
    cr.move_to(x0, y0);
    cr.line_to(x1, y1);
    let _ = cr.stroke();
    cr.set_dash(&[], 0.0);

    // The curve itself — same sampler the pipeline builds its LUT with, so the
    // drawing IS the applied transfer function.
    let (lr, lg, lb) = style.line;
    cr.set_source_rgb(lr, lg, lb);
    cr.set_line_width(2.0);
    let mut lut = vec![0.0f32; DRAW_SAMPLES];
    curve_tools::curve_data_sample(nodes, spline_type, 0.0, 1.0, &mut lut);
    for (k, &y) in lut.iter().enumerate() {
        let u = k as f32 / (DRAW_SAMPLES - 1) as f32;
        let (wx, wy) = rect.to_widget(u, y.clamp(0.0, 1.0));
        if k == 0 {
            cr.move_to(wx, wy);
        } else {
            cr.line_to(wx, wy);
        }
    }
    let _ = cr.stroke();

    // Anchor nodes; the grabbed one gets a highlight ring.
    for (idx, &(u, v)) in nodes.iter().enumerate() {
        let (cx, cy) = rect.to_widget(u, v);
        let grabbed = drag_idx == Some(idx);
        cr.set_source_rgb(0.95, 0.95, 0.95);
        cr.arc(cx, cy, 3.5, 0.0, 2.0 * std::f64::consts::PI);
        let _ = cr.fill();
        if grabbed {
            let (rr, rg, rb) = style.ring;
            cr.set_source_rgb(rr, rg, rb);
            cr.set_line_width(1.5);
            cr.arc(cx, cy, 5.5, 0.0, 2.0 * std::f64::consts::PI);
            let _ = cr.stroke();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_rect_round_trips_curve_through_widget() {
        let rect = PlotRect { x: 10.0, y: 8.0, w: 200.0, h: 150.0 };
        // Corners: (0,0) bottom-left, (1,1) top-right.
        assert_eq!(rect.to_widget(0.0, 0.0), (10.0, 158.0));
        assert_eq!(rect.to_widget(1.0, 1.0), (210.0, 8.0));
        // Round-trip within float noise, including the mid-point.
        for &(u, v) in &[(0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (1.0, 1.0)] {
            let (wx, wy) = rect.to_widget(u, v);
            let (bu, bv) = rect.to_curve(wx, wy);
            assert!((bu - u as f64).abs() < 1e-9 && (bv - v as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn insert_node_keeps_x_order_and_reports_index() {
        let mut nodes = vec![(0.0, 0.0), (1.0, 1.0)];
        let i = insert_node(&mut nodes, 0.5, 0.35).expect("mid insert");
        assert_eq!(i, 1);
        assert_eq!(nodes, vec![(0.0, 0.0), (0.5, 0.35), (1.0, 1.0)]);
        // Inserting left of everything lands first.
        let j = insert_node(&mut nodes, 0.1, 0.2).expect("left insert");
        assert_eq!(j, 1);
        assert!(nodes.windows(2).all(|w| w[0].0 <= w[1].0), "x-sorted");
        // Coordinates clamp into the box even for wild clicks: x sorts by its
        // CLAMPED value and y clamps at 1.
        let k = insert_node(&mut nodes, 0.3, 9.0).expect("interior insert");
        assert_eq!(nodes[k], (0.3, 1.0));
        // A click whose clamped x lands on an occupied column is refused —
        // every spline rejects non-increasing anchors, so accepting it would
        // silently snap the curve to the diagonal. This covers exact duplicate
        // columns AND raw clicks outside [0,1] that clamp onto an endpoint.
        assert_eq!(insert_node(&mut nodes, -3.0, 0.4), None, "clamps onto x=0");
        assert_eq!(
            insert_node(&mut nodes, 1.0 + X_EPS / 2.0, 0.4),
            None,
            "clamps onto x=1"
        );
        assert_eq!(
            insert_node(&mut nodes, 0.5 + X_EPS / 2.0, 0.4),
            None,
            "within X_EPS of the mid node"
        );
        let len_before = nodes.len();
        let _ = insert_node(&mut nodes, -3.0, 0.4);
        assert_eq!(nodes.len(), len_before, "refused insert must not mutate");
    }

    #[test]
    fn insert_node_refuses_when_full() {
        let mut nodes: Vec<(f32, f32)> =
            (0..MAX_ANCHORS).map(|i| (i as f32 / 19.0, 0.5)).collect();
        nodes[0].0 = 0.0;
        nodes.last_mut().unwrap().0 = 1.0;
        assert_eq!(insert_node(&mut nodes, 0.5, 0.5), None, "20 anchors is the cap");
        assert_eq!(nodes.len(), MAX_ANCHORS);
    }

    #[test]
    fn clamp_drag_pins_endpoints_and_bounds_interior() {
        let nodes = vec![(0.0, 0.0), (0.4, 0.5), (0.7, 0.5), (1.0, 1.0)];
        // Endpoints pinned no matter where the pointer goes.
        assert_eq!(clamp_drag_x(&nodes, 0, -5.0), 0.0);
        assert_eq!(clamp_drag_x(&nodes, 3, 5.0), 1.0);
        // Interior stays strictly between neighbours (and in [0,1]).
        assert_eq!(clamp_drag_x(&nodes, 1, 0.2), 0.2);
        assert!(
            clamp_drag_x(&nodes, 1, -1.0) > nodes[0].0,
            "left neighbour bound"
        );
        assert!(
            clamp_drag_x(&nodes, 2, 0.99) < nodes[3].0,
            "right neighbour bound"
        );
    }

    #[test]
    fn hit_and_remove_behave_like_the_gesture_contract() {
        let mut nodes = vec![(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)];
        // Near the middle node.
        assert_eq!(hit_node(&nodes, 0.52, 0.49, 0.05), Some(1));
        // Far away from everything.
        assert_eq!(hit_node(&nodes, 0.9, 0.1, 0.05), None);
        // Interior removal works…
        let mut editable = nodes.clone();
        assert!(remove_node(&mut editable, 1));
        assert_eq!(editable.len(), 2);
        // …but endpoints and 2-node lists refuse.
        assert!(!remove_node(&mut editable, 0));
        assert!(!remove_node(&mut editable, 1));
        assert!(!remove_node(&mut nodes, 0));
        assert!(!remove_node(&mut nodes, 2));
    }

    #[test]
    fn nodes_to_array_packs_prefix_zeroes_tail() {
        // The param array is fixed-width for the encoded blob; the live list
        // packs from the front and the tail must stay (0,0) so encoding is
        // deterministic regardless of history.
        let nodes = vec![(0.0, 0.0), (0.3, 0.2), (1.0, 1.0)];
        let arr = nodes_to_array(&nodes);
        assert_eq!(arr[0], (0.0, 0.0));
        assert_eq!(arr[1], (0.3, 0.2));
        assert_eq!(arr[2], (1.0, 1.0));
        assert!(arr[3..].iter().all(|&n| n == (0.0, 0.0)));
    }

    #[test]
    fn set_channel_nodes_routes_to_the_right_channel() {
        let mut p = PreviewParams::default();
        let nodes = vec![(0.0, 0.0), (0.5, 0.35), (1.0, 1.0)];
        set_channel_nodes(&mut p, 1, &nodes);
        // G got the list AND its count…
        assert_eq!(p.rc_nnodes_g, 3.0);
        assert_eq!(p.rc_nodes_g[1], (0.5, 0.35));
        // …while R and B keep their defaults untouched.
        assert_eq!(p.rc_nnodes_r, 2.0);
        assert_eq!(p.rc_nodes_r[1], (1.0, 1.0));
        assert_eq!(p.rc_nnodes_b, 2.0);
        // Channel 2 (B) routes too; the catch-all must not swallow ch 0.
        set_channel_nodes(&mut p, 2, &nodes);
        assert_eq!(p.rc_nnodes_b, 3.0);
        set_channel_nodes(&mut p, 0, &nodes);
        assert_eq!(p.rc_nnodes_r, 3.0);
    }

    #[test]
    fn channel_nodes_seeds_from_the_matching_array_and_count() {
        let mut p = PreviewParams::default();
        p.rc_nnodes_b = 4.0;
        p.rc_nodes_b[1] = (0.25, 0.45);
        let (arr_b, count_b) = channel_nodes(&p, 2);
        assert_eq!(count_b, 4.0);
        assert_eq!(arr_b[1], (0.25, 0.45));
        // Out-of-range channel falls back to B (same catch-all as the setter).
        let (arr_x, count_x) = channel_nodes(&p, 7);
        assert_eq!(count_x, 4.0);
        assert_eq!(arr_x[1], (0.25, 0.45));

        // seed_nodes clamps the count into [2, 20] whatever the blob said.
        p.rc_nnodes_g = 99.0;
        assert_eq!(seed_nodes(&p, 1).len(), MAX_ANCHORS);
        p.rc_nnodes_g = 0.0;
        assert_eq!(seed_nodes(&p, 1).len(), 2, "minimum two anchors");
    }
}
