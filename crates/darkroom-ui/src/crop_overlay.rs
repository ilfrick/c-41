//! Pure interaction math for the darkroom crop overlay (m4-48a). The GTK
//! `DrawingArea` + gestures (m4-48b) are a thin shell over these headless-tested
//! helpers, which work entirely in **fraction space** `[0, 1]` of the displayed
//! (already-rotated) image — the same space as [`Crop`]'s edges — so a grabbed
//! handle maps straight onto `ctx.geometry`'s crop.
//!
//! Widget pixels are converted to fractions with [`widget_to_fraction`] via the
//! shared [`ContainRect`] (so the overlay letterboxes identically to the picker
//! and wipe overlay).

use crate::preview::ContainRect;
use darkroom_core::geometry::Crop;

/// The part of a crop rectangle a pointer is interacting with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropHandle {
    TopLeft,
    Top,
    TopRight,
    Right,
    BottomRight,
    Bottom,
    BottomLeft,
    Left,
    /// Inside the rectangle — a whole-rect move.
    Inside,
    /// Outside, no handle grabbed.
    None,
}

/// Smallest crop extent the overlay allows per axis (2% of the image) — keeps
/// opposite handles from crossing and matches/exceeds [`Crop`]'s own guard.
const MIN: f32 = 0.02;

/// Map a widget-space point `(x, y)` to a fraction `(fx, fy)` in `[0, 1]` of the
/// displayed image `rect`, clamped so a drag onto the letterbox bars pins to an
/// edge. Mirrors [`crate::preview::wipe_fraction`] in both axes.
pub fn widget_to_fraction(rect: &ContainRect, x: f64, y: f64) -> (f32, f32) {
    let fx = if rect.disp_w > 0.0 {
        ((x - rect.off_x) / rect.disp_w).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let fy = if rect.disp_h > 0.0 {
        ((y - rect.off_y) / rect.disp_h).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (fx as f32, fy as f32)
}

/// Which handle of `crop` (edges in `[0, 1]`) is under point `(px, py)` (also a
/// `[0, 1]` fraction), within `tol` (a fraction). Corners take priority over
/// edges, edges over the interior. `tol` should be a few pixels expressed as a
/// fraction of the displayed size (`px / disp_w`).
pub fn hit_test(crop: Crop, px: f32, py: f32, tol: f32) -> CropHandle {
    let (l, t, r, b) = (crop.left, crop.top, crop.right, crop.bottom);
    let near = |a: f32, edge: f32| (a - edge).abs() <= tol;
    let (on_l, on_r, on_t, on_b) = (near(px, l), near(px, r), near(py, t), near(py, b));
    // An edge grab also requires being within the span of the perpendicular axis
    // (plus tol slack), so a point far off the side of the top edge isn't "Top".
    let in_x = px >= l - tol && px <= r + tol;
    let in_y = py >= t - tol && py <= b + tol;

    if on_l && on_t {
        CropHandle::TopLeft
    } else if on_r && on_t {
        CropHandle::TopRight
    } else if on_l && on_b {
        CropHandle::BottomLeft
    } else if on_r && on_b {
        CropHandle::BottomRight
    } else if on_t && in_x {
        CropHandle::Top
    } else if on_b && in_x {
        CropHandle::Bottom
    } else if on_l && in_y {
        CropHandle::Left
    } else if on_r && in_y {
        CropHandle::Right
    } else if px > l && px < r && py > t && py < b {
        CropHandle::Inside
    } else {
        CropHandle::None
    }
}

/// Resize `crop` by dragging `handle` to the fraction `(px, py)`, keeping the
/// opposite edge(s) fixed. Each moved edge is clamped into `[0, 1]` and kept at
/// least [`MIN`] from its opposite, so the rectangle can never invert or
/// collapse. `Inside`/`None` return the crop unchanged (use [`translate`] to
/// move).
pub fn resize_to(crop: Crop, handle: CropHandle, px: f32, py: f32) -> Crop {
    let px = px.clamp(0.0, 1.0);
    let py = py.clamp(0.0, 1.0);
    let mut c = crop;
    // Clamp a moved low/high edge against 0/1 and the opposite edge.
    let left_to = |c: &Crop, x: f32| x.min(c.right - MIN).max(0.0);
    let right_to = |c: &Crop, x: f32| x.max(c.left + MIN).min(1.0);
    let top_to = |c: &Crop, y: f32| y.min(c.bottom - MIN).max(0.0);
    let bottom_to = |c: &Crop, y: f32| y.max(c.top + MIN).min(1.0);
    match handle {
        CropHandle::Left => c.left = left_to(&c, px),
        CropHandle::Right => c.right = right_to(&c, px),
        CropHandle::Top => c.top = top_to(&c, py),
        CropHandle::Bottom => c.bottom = bottom_to(&c, py),
        CropHandle::TopLeft => {
            c.left = left_to(&c, px);
            c.top = top_to(&c, py);
        }
        CropHandle::TopRight => {
            c.right = right_to(&c, px);
            c.top = top_to(&c, py);
        }
        CropHandle::BottomLeft => {
            c.left = left_to(&c, px);
            c.bottom = bottom_to(&c, py);
        }
        CropHandle::BottomRight => {
            c.right = right_to(&c, px);
            c.bottom = bottom_to(&c, py);
        }
        CropHandle::Inside | CropHandle::None => {}
    }
    c
}

/// Move the whole `crop` by `(dx, dy)` (fractions), shrinking the delta so the
/// rectangle stays fully inside `[0, 1]` (its size is preserved).
pub fn translate(crop: Crop, dx: f32, dy: f32) -> Crop {
    let dx = dx.clamp(-crop.left, 1.0 - crop.right);
    let dy = dy.clamp(-crop.top, 1.0 - crop.bottom);
    Crop {
        left: crop.left + dx,
        top: crop.top + dy,
        right: crop.right + dx,
        bottom: crop.bottom + dy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(l: f32, t: f32, r: f32, b: f32) -> Crop {
        Crop { left: l, top: t, right: r, bottom: b }
    }

    #[test]
    fn hit_test_corners_edges_inside_outside() {
        let cr = c(0.2, 0.2, 0.8, 0.8);
        let tol = 0.03;
        assert_eq!(hit_test(cr, 0.2, 0.2, tol), CropHandle::TopLeft);
        assert_eq!(hit_test(cr, 0.8, 0.2, tol), CropHandle::TopRight);
        assert_eq!(hit_test(cr, 0.2, 0.8, tol), CropHandle::BottomLeft);
        assert_eq!(hit_test(cr, 0.8, 0.8, tol), CropHandle::BottomRight);
        assert_eq!(hit_test(cr, 0.5, 0.2, tol), CropHandle::Top);
        assert_eq!(hit_test(cr, 0.5, 0.8, tol), CropHandle::Bottom);
        assert_eq!(hit_test(cr, 0.2, 0.5, tol), CropHandle::Left);
        assert_eq!(hit_test(cr, 0.8, 0.5, tol), CropHandle::Right);
        assert_eq!(hit_test(cr, 0.5, 0.5, tol), CropHandle::Inside);
        assert_eq!(hit_test(cr, 0.05, 0.05, tol), CropHandle::None);
        // Near the top EDGE line but far off the side ⇒ not Top (out of x-span).
        assert_eq!(hit_test(cr, 0.05, 0.2, tol), CropHandle::None);
    }

    #[test]
    fn resize_clamps_to_unit_and_keeps_min_extent() {
        let cr = c(0.2, 0.2, 0.8, 0.8);
        // Drag the left edge past the right one ⇒ pinned MIN left of right.
        let r = resize_to(cr, CropHandle::Left, 0.95, 0.5);
        assert!((r.left - (0.8 - MIN)).abs() < 1e-6, "left={}", r.left);
        assert_eq!(r.right, 0.8); // opposite edge fixed
        // Drag a corner out of range ⇒ clamped into [0,1].
        let r2 = resize_to(cr, CropHandle::TopLeft, -0.5, -0.5);
        assert_eq!((r2.left, r2.top), (0.0, 0.0));
        assert_eq!((r2.right, r2.bottom), (0.8, 0.8));
        // Bottom-right drag past 1 clamps to 1.
        let r3 = resize_to(cr, CropHandle::BottomRight, 1.4, 1.4);
        assert_eq!((r3.right, r3.bottom), (1.0, 1.0));
    }

    #[test]
    fn translate_preserves_size_and_stays_in_bounds() {
        let cr = c(0.2, 0.3, 0.6, 0.7); // 0.4 x 0.4
        let m = translate(cr, 0.1, -0.1);
        assert!((m.left - 0.3).abs() < 1e-6 && (m.right - 0.7).abs() < 1e-6);
        assert!((m.top - 0.2).abs() < 1e-6 && (m.bottom - 0.6).abs() < 1e-6);
        // Push hard right: delta shrinks so right pins at 1.0, size kept (0.4).
        let m2 = translate(cr, 0.9, 0.0);
        assert!((m2.right - 1.0).abs() < 1e-6);
        assert!((m2.right - m2.left - 0.4).abs() < 1e-6);
        // Full-width crop can't move horizontally.
        assert_eq!(translate(c(0.0, 0.2, 1.0, 0.8), 0.3, 0.0), c(0.0, 0.2, 1.0, 0.8));
    }

    #[test]
    fn widget_to_fraction_maps_and_clamps() {
        let rect = ContainRect { off_x: 10.0, off_y: 20.0, disp_w: 100.0, disp_h: 200.0, scale: 1.0 };
        assert_eq!(widget_to_fraction(&rect, 10.0, 20.0), (0.0, 0.0)); // top-left
        assert_eq!(widget_to_fraction(&rect, 60.0, 120.0), (0.5, 0.5)); // centre
        assert_eq!(widget_to_fraction(&rect, 110.0, 220.0), (1.0, 1.0)); // bottom-right
        // Onto the letterbox bar (past the image) ⇒ pinned to the edge.
        assert_eq!(widget_to_fraction(&rect, -50.0, 500.0), (0.0, 1.0));
    }
}
