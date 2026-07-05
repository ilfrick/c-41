//! Geometry operations on the linear RGBA preview buffer — **crop** first, with
//! rotate to follow. Kept deliberately separate from the per-pixel colour
//! [`crate::pipeline`]: every current pipeline stage is position-independent, so
//! a rectangular crop yields identical colour whether applied before or after
//! them. Geometry therefore runs as its own pass rather than as a
//! dimension-changing `Stage`, which keeps the ping-pong `Pipeline` size-agnostic.
//!
//! **Note:** this commutativity holds only while every pipeline stage is
//! position-independent (per-pixel). Once a spatially-varying IOP such as lens
//! correction, local contrast, or denoise is added, crop placement relative to
//! that stage becomes order-sensitive and this separation must be revisited (the
//! same caveat is recorded in [`crate::pipeline`]).

/// A rectangular crop expressed as fractions of the source image in `[0, 1]`
/// (resolution-independent, so the same `Crop` survives the preview downscale
/// and applies unchanged at export resolution). Invariant after
/// [`normalized`](Self::normalized): `0 ≤ left < right ≤ 1` and
/// `0 ≤ top < bottom ≤ 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crop {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl Default for Crop {
    /// The identity crop — the whole image.
    fn default() -> Self {
        Self { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 }
    }
}

impl Crop {
    /// Smallest crop extent per axis (1% of the image) — stops a degenerate or
    /// inverted rectangle from collapsing the buffer to zero pixels.
    const MIN_EXTENT: f32 = 0.01;

    /// Clamp every edge into `[0, 1]` (NaN → the identity edge: `left`/`top` → 0,
    /// `right`/`bottom` → 1) and guarantee each axis keeps at least
    /// [`MIN_EXTENT`](Self::MIN_EXTENT), so the result can never invert or make an
    /// empty image. A too-small or inverted axis falls back to that axis's full
    /// extent.
    pub fn normalized(self) -> Crop {
        fn edge(v: f32, fallback: f32) -> f32 {
            if v.is_nan() { fallback } else { v.clamp(0.0, 1.0) }
        }
        let mut left = edge(self.left, 0.0);
        let mut right = edge(self.right, 1.0);
        let mut top = edge(self.top, 0.0);
        let mut bottom = edge(self.bottom, 1.0);
        if right - left < Self::MIN_EXTENT {
            (left, right) = (0.0, 1.0);
        }
        if bottom - top < Self::MIN_EXTENT {
            (top, bottom) = (0.0, 1.0);
        }
        Crop { left, top, right, bottom }
    }

    /// True when this crop selects (within a sub-pixel tolerance) the whole
    /// image, so the geometry pass can be skipped.
    pub fn is_identity(self) -> bool {
        let c = self.normalized();
        c.left <= f32::EPSILON
            && c.top <= f32::EPSILON
            && c.right >= 1.0 - f32::EPSILON
            && c.bottom >= 1.0 - f32::EPSILON
    }

    /// The integer pixel rectangle `(x, y, width, height)` this crop selects from
    /// a `src_w × src_h` image. Always at least `1 × 1` and fully in bounds (for
    /// a non-empty source); a zero-dimension source yields a zero rectangle.
    pub fn pixel_rect(self, src_w: usize, src_h: usize) -> (usize, usize, usize, usize) {
        if src_w == 0 || src_h == 0 {
            return (0, 0, 0, 0);
        }
        let c = self.normalized();
        let axis = |lo: f32, hi: f32, len: usize| -> (usize, usize) {
            let n = len as f32;
            let a = (lo * n).round().clamp(0.0, (len - 1) as f32) as usize;
            let b = (hi * n).round().clamp((a + 1) as f32, len as f32) as usize;
            (a, b - a)
        };
        let (x, w) = axis(c.left, c.right, src_w);
        let (y, h) = axis(c.top, c.bottom, src_h);
        (x, y, w, h)
    }
}

/// Crop a packed linear RGBA `f32` buffer (`pixels`, `w × h`, `w*h*4` long) to
/// the sub-rectangle `crop` selects. Returns `(new_w, new_h, pixels)`.
///
/// The identity crop, a zero-dimension image, or a short/malformed buffer
/// returns the source dimensions and a verbatim copy — geometry never fails a
/// render, it just no-ops.
pub fn apply_crop(pixels: &[f32], w: usize, h: usize, crop: Crop) -> (usize, usize, Vec<f32>) {
    if w == 0 || h == 0 || pixels.len() < w * h * 4 || crop.is_identity() {
        return (w, h, pixels.to_vec());
    }
    let (x, y, cw, ch) = crop.pixel_rect(w, h);
    let mut out = vec![0.0f32; cw * ch * 4];
    for row in 0..ch {
        let src = ((y + row) * w + x) * 4;
        let dst = row * cw * 4;
        out[dst..dst + cw * 4].copy_from_slice(&pixels[src..src + cw * 4]);
    }
    (cw, ch, out)
}

/// Below this magnitude (radians, ~0.006°) a rotation is treated as identity —
/// avoids a pointless full-frame resample for a numerically-zero angle.
pub const MIN_ANGLE: f32 = 1e-4;

/// Rotate a packed linear RGBA `f32` buffer (`pixels`, `w × h`) by `angle`
/// **radians** about the image centre, **expanding the canvas** to contain the
/// whole rotated image; pixels that fall outside the source become transparent
/// black `(0,0,0,0)`. Positive `angle` rotates the image counter-clockwise.
/// Returns `(new_w, new_h, pixels)`.
///
/// Resampling is bilinear via [`crate::interp::compute_pixel4c`]. A near-zero
/// angle ([`MIN_ANGLE`]), a non-finite angle, or a zero-dim / short buffer
/// returns the source dimensions and a verbatim copy — geometry never fails a
/// render. The expanded canvas + black corners compose cleanly with
/// [`apply_crop`] (crop the rotated buffer to trim the corners), which is how
/// the UI will straighten-and-crop.
pub fn apply_rotate(pixels: &[f32], w: usize, h: usize, angle: f32) -> (usize, usize, Vec<f32>) {
    if w == 0 || h == 0 || pixels.len() < w * h * 4 || !angle.is_finite() || angle.abs() < MIN_ANGLE
    {
        return (w, h, pixels.to_vec());
    }
    let (sin, cos) = angle.sin_cos();
    let (wf, hf) = (w as f32, h as f32);
    // Axis-aligned bounding box of the rotated rectangle. Subtract a sub-pixel
    // epsilon before `ceil` so float sin/cos artefacts at near-axis angles (e.g.
    // cos(π/2) ≈ 4e-8 instead of 0) can't inflate the canvas by a spurious
    // row/column. Safe while w,h ≲ 22 000 px (worst-case artefact w·4.37e-8 <
    // 1e-3); for larger buffers shrink the epsilon or use f64 here. The cost is
    // ≤ 1e-3 px of extremal-corner content excluded — visually zero for a
    // preview (those corners are the least-reliable mirror-clamp samples anyway).
    let ceil_dim = |v: f32| (v - 1e-3).ceil().max(1.0) as usize;
    let w2 = ceil_dim((wf * cos.abs()) + (hf * sin.abs()));
    let h2 = ceil_dim((wf * sin.abs()) + (hf * cos.abs()));
    // Grid centres in pixel-index space (integer for odd extents).
    let (scx, scy) = ((wf - 1.0) * 0.5, (hf - 1.0) * 0.5);
    let (dcx, dcy) = ((w2 as f32 - 1.0) * 0.5, (h2 as f32 - 1.0) * 0.5);
    let ls = (w * 4) as i32;
    let mut out = vec![0.0f32; w2 * h2 * 4];
    for j in 0..h2 {
        let dy = j as f32 - dcy;
        for i in 0..w2 {
            let dx = i as f32 - dcx;
            // Inverse rotation (dest → source): rotate the dest offset by -angle.
            let sx = scx + dx * cos + dy * sin;
            let sy = scy - dx * sin + dy * cos;
            if sx >= 0.0 && sy >= 0.0 && sx < wf && sy < hf {
                let base = (j * w2 + i) * 4;
                let px = crate::interp::compute_pixel4c(pixels, sx, sy, w as i32, h as i32, ls, 0);
                out[base..base + 4].copy_from_slice(&px);
            }
            // else: leave transparent black (buffer is zero-initialised)
        }
    }
    (w2, h2, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `w×h` RGBA buffer whose red channel encodes `col`, green `row`
    /// (so a cropped pixel's provenance is checkable), alpha 1.
    fn grid(w: usize, h: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h * 4];
        for row in 0..h {
            for col in 0..w {
                let p = (row * w + col) * 4;
                v[p] = col as f32;
                v[p + 1] = row as f32;
                v[p + 3] = 1.0;
            }
        }
        v
    }

    #[test]
    fn identity_crop_returns_input_unchanged() {
        let px = grid(8, 5);
        let (w, h, out) = apply_crop(&px, 8, 5, Crop::default());
        assert_eq!((w, h), (8, 5));
        assert_eq!(out, px);
    }

    #[test]
    fn half_crop_selects_the_right_subrectangle() {
        // Crop the right-bottom quarter of a 10×10 grid.
        let px = grid(10, 10);
        let crop = Crop { left: 0.5, top: 0.5, right: 1.0, bottom: 1.0 };
        let (w, h, out) = apply_crop(&px, 10, 10, crop);
        assert_eq!((w, h), (5, 5));
        // Top-left of the crop is source pixel (col 5, row 5).
        assert_eq!(out[0], 5.0, "red = source col");
        assert_eq!(out[1], 5.0, "green = source row");
        // Bottom-right of the crop is source (col 9, row 9).
        let last = (4 * 5 + 4) * 4;
        assert_eq!(out[last], 9.0);
        assert_eq!(out[last + 1], 9.0);
    }

    #[test]
    fn normalized_clamps_and_repairs_inverted_or_out_of_range() {
        // Out of range clamps; inverted axis falls back to full extent.
        let c = Crop { left: -0.5, top: 2.0, right: 0.4, bottom: 0.3 }.normalized();
        assert_eq!(c.left, 0.0); // -0.5 → 0
        assert_eq!(c.right, 0.4); // in range, kept
        // top 2.0→1.0, bottom 0.3 ⇒ inverted (top>bottom) ⇒ full-height fallback
        assert_eq!((c.top, c.bottom), (0.0, 1.0));
        // A NaN edge collapses to its identity value.
        let n = Crop { left: f32::NAN, top: 0.0, right: 1.0, bottom: f32::NAN }.normalized();
        assert_eq!((n.left, n.bottom), (0.0, 1.0));
    }

    #[test]
    fn too_small_crop_falls_back_to_full_axis() {
        // 0.5% wide < MIN_EXTENT ⇒ horizontal fallback to full; vertical kept.
        let c = Crop { left: 0.500, top: 0.2, right: 0.505, bottom: 0.8 }.normalized();
        assert_eq!((c.left, c.right), (0.0, 1.0));
        assert_eq!((c.top, c.bottom), (0.2, 0.8));
    }

    #[test]
    fn pixel_rect_is_in_bounds_and_at_least_one_pixel() {
        // A tiny (but >= MIN_EXTENT) crop still yields ≥ 1×1, in bounds.
        let crop = Crop { left: 0.99, top: 0.99, right: 1.0, bottom: 1.0 };
        let (x, y, w, h) = crop.pixel_rect(100, 100);
        assert!(w >= 1 && h >= 1);
        assert!(x + w <= 100 && y + h <= 100);
        // Full crop covers the whole image exactly.
        assert_eq!(Crop::default().pixel_rect(100, 80), (0, 0, 100, 80));
        // Zero-dimension source ⇒ zero rect (no panic).
        assert_eq!(Crop::default().pixel_rect(0, 5), (0, 0, 0, 0));
        // 1-pixel source: identity is the single pixel; a partial crop is below
        // MIN_EXTENT (0.01 px) so it falls back to the full axis ⇒ still 1×1.
        assert_eq!(Crop::default().pixel_rect(1, 1), (0, 0, 1, 1));
        let (_, _, w1, h1) =
            Crop { left: 0.5, top: 0.0, right: 1.0, bottom: 1.0 }.pixel_rect(1, 1);
        assert!(w1 >= 1 && h1 >= 1);
    }

    #[test]
    fn pixel_rect_rounds_half_away_from_zero() {
        // right = 0.5 on a 3-px axis → 0.5*3 = 1.5 → rounds to 2 (half away from
        // zero), so the crop takes 2 of the 3 columns. Pins the .5 boundary.
        assert_eq!(
            Crop { left: 0.0, top: 0.0, right: 0.5, bottom: 1.0 }.pixel_rect(3, 1),
            (0, 0, 2, 1)
        );
    }

    #[test]
    fn short_buffer_no_ops() {
        let px = vec![0.0f32; 3 * 4]; // claims 4x4 but only 3 px
        let (w, h, out) = apply_crop(&px, 4, 4, Crop { left: 0.0, top: 0.0, right: 0.5, bottom: 0.5 });
        assert_eq!((w, h), (4, 4)); // returned source dims, verbatim copy
        assert_eq!(out, px);
    }

    #[test]
    fn identity_via_normalized_is_detected() {
        assert!(Crop::default().is_identity());
        assert!(Crop { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0 }.is_identity());
        assert!(!Crop { left: 0.1, top: 0.0, right: 1.0, bottom: 1.0 }.is_identity());
        // An inverted crop normalizes back to identity.
        assert!(Crop { left: 0.9, top: 0.0, right: 0.905, bottom: 1.0 }.is_identity());
    }

    /// A uniform `w×h` field of colour `v` (alpha 1) — the robust rotate fixture:
    /// bilinear sampling anywhere inside returns `v`, so only in-vs-out-of-source
    /// matters (independent of sub-pixel resampling error).
    fn uniform(w: usize, h: usize, v: f32) -> Vec<f32> {
        let mut px = vec![0.0f32; w * h * 4];
        for p in px.chunks_exact_mut(4) {
            p[0] = v;
            p[1] = v;
            p[2] = v;
            p[3] = 1.0;
        }
        px
    }

    #[test]
    fn rotate_zero_and_tiny_angle_are_identity() {
        let px = grid(6, 4);
        assert_eq!(apply_rotate(&px, 6, 4, 0.0), (6, 4, px.clone()));
        // Below MIN_ANGLE ⇒ identity (verbatim copy, no resample).
        assert_eq!(apply_rotate(&px, 6, 4, MIN_ANGLE * 0.5), (6, 4, px));
    }

    #[test]
    fn rotate_ninety_swaps_dimensions() {
        // 90°: bbox = (w·|cos90| + h·|sin90|) × (w·|sin90| + h·|cos90|) = h × w.
        let px = uniform(8, 5, 0.5);
        let (w2, h2, _) = apply_rotate(&px, 8, 5, std::f32::consts::FRAC_PI_2);
        assert_eq!((w2, h2), (5, 8));
    }

    #[test]
    fn rotate_fills_interior_and_blackens_new_corners() {
        // Rotate a uniform field: the centre stays in-source (→ v, opaque); the
        // expanded canvas corners fall outside the source (→ transparent black).
        let (w, h) = (40usize, 30usize);
        let px = uniform(w, h, 0.5);
        let (w2, h2, out) = apply_rotate(&px, w, h, 0.3);
        assert!(w2 > w && h2 > h, "canvas expands: {w2}x{h2}");
        let centre = ((h2 / 2) * w2 + w2 / 2) * 4;
        for c in 0..3 {
            assert!((out[centre + c] - 0.5).abs() < 1e-4, "centre ch{c}={}", out[centre + c]);
        }
        assert_eq!(out[centre + 3], 1.0, "centre opaque");
        // Top-left corner of the expanded canvas is outside the rotated source.
        assert_eq!(&out[0..4], &[0.0, 0.0, 0.0, 0.0], "corner transparent black");
    }

    #[test]
    fn rotate_ninety_moves_left_half_to_top() {
        // Direction pin: source is white on the LEFT half, black on the right.
        // A +90° (counter-clockwise) rotation sends the left edge to the TOP, so
        // the dest top must read white and the dest bottom black. Guards the
        // rotation sense + centring convention (the uniform tests can't).
        let (w, h) = (8usize, 8usize);
        let mut px = vec![0.0f32; w * h * 4];
        for row in 0..h {
            for col in 0..w {
                let v = if col < w / 2 { 1.0 } else { 0.0 };
                let p = (row * w + col) * 4;
                px[p] = v;
                px[p + 1] = v;
                px[p + 2] = v;
                px[p + 3] = 1.0;
            }
        }
        let (w2, h2, out) = apply_rotate(&px, w, h, std::f32::consts::FRAC_PI_2);
        assert_eq!((w2, h2), (8, 8));
        let top = (1 * w2 + 4) * 4; // near top, mid-column
        let bottom = (6 * w2 + 4) * 4; // near bottom, mid-column
        assert!(out[top] > 0.6, "top should be white, got {}", out[top]);
        assert!(out[bottom] < 0.4, "bottom should be black, got {}", out[bottom]);
    }

    #[test]
    fn rotate_neg_ninety_moves_left_half_to_bottom() {
        // Direction pin for the OTHER sign: −90° (clockwise) sends the left edge
        // to the bottom. A flipped sin sign would pass the +90° test (symmetric
        // halves) but fail here.
        let (w, h) = (8usize, 8usize);
        let mut px = vec![0.0f32; w * h * 4];
        for row in 0..h {
            for col in 0..w {
                let v = if col < w / 2 { 1.0 } else { 0.0 };
                let p = (row * w + col) * 4;
                px[p] = v;
                px[p + 1] = v;
                px[p + 2] = v;
                px[p + 3] = 1.0;
            }
        }
        let (w2, h2, out) = apply_rotate(&px, w, h, -std::f32::consts::FRAC_PI_2);
        assert_eq!((w2, h2), (8, 8));
        let top = (1 * w2 + 4) * 4;
        let bottom = (6 * w2 + 4) * 4;
        assert!(out[top] < 0.4, "top should be black, got {}", out[top]);
        assert!(out[bottom] > 0.6, "bottom should be white, got {}", out[bottom]);
    }

    #[test]
    fn rotate_bounding_box_dimensions() {
        let px = uniform(8, 5, 0.5);
        // Non-square at a general angle: w2 = ceil(8·cos.5 + 5·sin.5) = 10,
        // h2 = ceil(8·sin.5 + 5·cos.5) = 9.
        let (w2, h2, _) = apply_rotate(&px, 8, 5, 0.5);
        assert_eq!((w2, h2), (10, 9));
        // 180° must preserve dimensions exactly (the -1e-3 epsilon is what keeps
        // cos(π) ≈ -1 with a float sin(π) ≈ 8.7e-8 from inflating to (9, 6)).
        let (w3, h3, _) = apply_rotate(&px, 8, 5, std::f32::consts::PI);
        assert_eq!((w3, h3), (8, 5));
    }

    #[test]
    fn rotate_degenerate_inputs_no_op() {
        let px = uniform(4, 4, 0.5);
        assert_eq!(apply_rotate(&px, 4, 4, f32::NAN), (4, 4, px.clone()));
        assert_eq!(apply_rotate(&px, 4, 4, f32::INFINITY), (4, 4, px.clone()));
        assert_eq!(apply_rotate(&px, 4, 4, f32::NEG_INFINITY), (4, 4, px.clone()));
        assert_eq!(apply_rotate(&px, 0, 4, 0.3), (0, 4, px.clone()));
        let short = vec![0.0f32; 3 * 4];
        assert_eq!(apply_rotate(&short, 4, 4, 0.3), (4, 4, short));
    }
}
