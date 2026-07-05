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
}
