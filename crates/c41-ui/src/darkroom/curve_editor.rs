//! Interactive L-curve editor for the Tone curve module (m4-122).
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
//! The drawn curve is sampled through [`c41_core::curve_tools::curve_data_sample`]
//! — the *same* sampler `PreviewParams::to_pipeline` uses to build the LUT — so
//! what the user sees is exactly what the pipeline applies. Every mutation is
//! written back into `ctx.params.tc_nodes_l` / `tc_nnodes` before re-rendering.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use adw::prelude::*;
use gtk4::{DrawingArea, EventControllerMotion, GestureClick};

use c41_core::curve_tools;

use super::PreviewCtx;

/// Widget padding around the plot box, in px.
const PAD: f64 = 10.0;
/// Node grab radius in widget px.
const HIT_PX: f64 = 10.0;
/// Minimum x separation between interior neighbouring anchors while dragging.
const X_EPS: f32 = 1e-4;
/// Curve samples drawn per repaint.
const DRAW_SAMPLES: usize = 256;

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
/// when the list already holds [`curve_tools::MAX_ANCHORS`] nodes or the
/// (box-clamped) x collides with a neighbour — every spline rejects
/// non-increasing anchors, so a duplicate x would silently snap the whole
/// curve to the identity diagonal while the nodes stay visible.
fn insert_node(nodes: &mut Vec<(f32, f32)>, x: f32, y: f32) -> Option<usize> {
    if nodes.len() >= curve_tools::MAX_ANCHORS {
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
fn nodes_to_array(nodes: &[(f32, f32)]) -> [(f32, f32); curve_tools::MAX_ANCHORS] {
    let mut arr = [(0.0f32, 0.0f32); curve_tools::MAX_ANCHORS];
    for (slot, n) in arr.iter_mut().zip(nodes.iter()) {
        *slot = *n;
    }
    arr
}

/// Copy `nodes` into the shared params (fixed-width array + count).
fn sync_params(ctx: &PreviewCtx, nodes: &[(f32, f32)]) {
    let mut p = ctx.params.borrow_mut();
    p.tc_nodes_l = nodes_to_array(nodes);
    p.tc_nnodes = nodes.len() as f32;
}

/// Shared editor state: the anchor list plus which node (if any) a drag holds.
struct CurveState {
    /// Anchors in curve-box coordinates, x-sorted, endpoints pinned at x=0/1.
    nodes: Rc<RefCell<Vec<(f32, f32)>>>,
    drag: Rc<Cell<Option<usize>>>,
}

/// Build the Tone curve module row: enable switch, interpolator selector and
/// the interactive L-curve editor.
pub(crate) fn tonecurve_module_row(ctx: &PreviewCtx) -> adw::ExpanderRow {
    let p0 = *ctx.params.borrow();
    let initial: Vec<(f32, f32)> =
        p0.tc_nodes_l[..(p0.tc_nnodes.round() as usize).clamp(2, curve_tools::MAX_ANCHORS)]
            .to_vec();
    let state = Rc::new(CurveState {
        nodes: Rc::new(RefCell::new(initial)),
        drag: Rc::new(Cell::new(None)),
    });

    // ── The editor canvas ──────────────────────────────────────────────────
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

    // Draw func: captures ctx (to read tc_type live) + the shared nodes + the
    // drag slot (so the grabbed node gets a highlight ring). No reference back
    // to `area`, so no ownership cycle.
    {
        let draw_ctx = ctx.clone();
        let draw_nodes = state.nodes.clone();
        let draw_drag = state.drag.clone();
        area.set_draw_func(move |_, cr, w, h| {
            let rect = PlotRect::from_widget_size(w, h);
            let spline_type = draw_ctx.params.borrow().tc_type as i32 as u32;
            let nodes = draw_nodes.borrow();
            draw_curve(cr, rect, &nodes, spline_type, draw_drag.get());
        });
    }

    // ── Gestures ───────────────────────────────────────────────────────────
    // Press: grab / insert / double-click-remove. Motion: move the grabbed
    // node. Release: drop it. Every mutation syncs params + re-renders.
    let click = GestureClick::new();
    {
        let st = state.clone();
        let edit_ctx = ctx.clone();
        let edit_area = area.downgrade();
        click.connect_pressed(move |_, n_press, wx, wy| {
            let rect = current_rect(&edit_area);
            let Some(rect) = rect else { return };
            let (u, v) = rect.to_curve(wx, wy);
            // Double-click on a node removes it (interior only).
            if n_press >= 2 {
                let hit = {
                    let nodes = st.nodes.borrow();
                    hit_node(&nodes, u as f32, v as f32, hit_tol(rect))
                };
                if let Some(idx) = hit {
                    let removed = remove_node(&mut st.nodes.borrow_mut(), idx);
                    st.drag.set(None);
                    if removed {
                        let nodes = st.nodes.borrow();
                        sync_params(&edit_ctx, &nodes);
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
                let nodes = st.nodes.borrow();
                hit_node(&nodes, u as f32, v as f32, hit_tol(rect))
            };
            let target = match hit {
                Some(idx) => Some(idx),
                None => {
                    // Only accept inserts that land inside the plot box.
                    if (0.0..=1.0).contains(&(u as f32)) && (0.0..=1.0).contains(&(v as f32)) {
                        insert_node(&mut st.nodes.borrow_mut(), u as f32, v as f32)
                    } else {
                        None
                    }
                }
            };
            st.drag.set(target);
            if target.is_some() {
                let nodes = st.nodes.borrow();
                sync_params(&edit_ctx, &nodes);
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
            st.drag.set(None);
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
            let Some(_idx) = st.drag.get() else { return };
            let Some(rect) = current_rect(&move_area) else { return };
            let (u, v) = rect.to_curve(wx, wy);
            {
                let mut nodes = st.nodes.borrow_mut();
                let Some(idx) = st.drag.get() else { return };
                let clamped_x = clamp_drag_x(&nodes, idx, u as f32);
                nodes[idx] = (
                    clamped_x,
                    (v as f32).clamp(0.0, 1.0),
                );
                sync_params(&move_ctx, &nodes);
            }
            if let Some(a) = move_area.upgrade() {
                a.queue_draw();
            }
            super::render_preview(&move_ctx);
        });
    }
    area.add_controller(motion);

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
    cr.set_source_rgb(0.98, 0.72, 0.11);
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
            cr.set_source_rgb(0.98, 0.72, 0.11);
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
            (0..curve_tools::MAX_ANCHORS).map(|i| (i as f32 / 19.0, 0.5)).collect();
        nodes[0].0 = 0.0;
        nodes.last_mut().unwrap().0 = 1.0;
        assert_eq!(insert_node(&mut nodes, 0.5, 0.5), None, "20 anchors is the cap");
        assert_eq!(nodes.len(), curve_tools::MAX_ANCHORS);
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
}
