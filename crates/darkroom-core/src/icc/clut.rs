//! Multi-dimensional colour lookup table (cLUT) evaluation — the interpolation
//! core of ICC LUT profiles (`mft1`/`mft2`/`mAB `/`mBA `).
//!
//! Two interpolators:
//! - [`Clut::eval`] dispatches to **tetrahedral** interpolation for the common
//!   3-input case (what LCMS uses for RGB cLUTs — more accurate than trilinear
//!   for typical device profiles, and exactly matched here), and to general
//!   **N-linear** interpolation for any other input dimensionality.
//! - Everything is `f32` (LCMS's default path is 16-bit fixed point), so we are
//!   at least as accurate as LCMS on the interpolation itself.
//!
//! ICC CLUT data layout: the first input channel varies slowest, the last
//! fastest; the `output_channels` values for a node are contiguous. So the node
//! at grid coords `(i0,…,i_{n-1})` starts at
//! `((…(i0·g1 + i1)·g2 + i2)…)·output_channels`.

/// A parsed colour lookup table: an N-input → M-output grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Clut {
    /// Grid points per input channel (length = number of input channels).
    pub grid: Vec<usize>,
    /// Number of output channels per node.
    pub output_channels: usize,
    /// Node data, ICC order (first input slowest), `output_channels` per node.
    pub data: Vec<f32>,
}

impl Clut {
    /// Number of input channels.
    #[inline]
    pub fn input_channels(&self) -> usize {
        self.grid.len()
    }

    /// The flat data offset of grid node `coords` (first channel slowest).
    #[inline]
    fn node_offset(&self, coords: &[usize]) -> usize {
        let mut idx = 0usize;
        for (c, &g) in coords.iter().zip(self.grid.iter()) {
            idx = idx * g + *c;
        }
        idx * self.output_channels
    }

    /// Interpolate the table at `input` (each in `[0, 1]`), writing
    /// `output_channels` values into `out`. Dispatches to tetrahedral (3-in) or
    /// N-linear.
    pub fn eval(&self, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), self.input_channels());
        debug_assert!(out.len() >= self.output_channels);
        if self.input_channels() == 3 {
            self.eval_tetrahedral(input, out);
        } else {
            self.eval_nlinear(input, out);
        }
    }

    /// General N-linear interpolation over the `2^n` surrounding nodes.
    pub fn eval_nlinear(&self, input: &[f32], out: &mut [f32]) {
        let n = self.input_channels();
        let m = self.output_channels;
        for o in out.iter_mut().take(m) {
            *o = 0.0;
        }
        // per-dim base index + fraction
        let mut lo = vec![0usize; n];
        let mut frac = vec![0.0f32; n];
        for d in 0..n {
            let g = self.grid[d];
            if g <= 1 {
                lo[d] = 0;
                frac[d] = 0.0;
                continue;
            }
            let x = input[d].clamp(0.0, 1.0) * (g - 1) as f32;
            let l = (x.floor() as usize).min(g - 2);
            lo[d] = l;
            frac[d] = x - l as f32;
        }
        // sum over the 2^n corners
        let corners = 1usize << n;
        let mut coords = vec![0usize; n];
        for mask in 0..corners {
            let mut w = 1.0f32;
            for d in 0..n {
                let bit = (mask >> d) & 1;
                // dims with grid==1 have frac 0; bit 1 would read out of range, skip
                if self.grid[d] <= 1 && bit == 1 {
                    w = 0.0;
                    break;
                }
                coords[d] = lo[d] + bit;
                w *= if bit == 1 { frac[d] } else { 1.0 - frac[d] };
            }
            if w == 0.0 {
                continue;
            }
            let base = self.node_offset(&coords);
            for o in 0..m {
                out[o] += w * self.data[base + o];
            }
        }
    }

    /// Tetrahedral interpolation for exactly 3 input channels — a faithful port of
    /// LCMS's `TetrahedralInterp` (per output channel), the standard RGB cLUT path.
    pub fn eval_tetrahedral(&self, input: &[f32], out: &mut [f32]) {
        let m = self.output_channels;
        let (gx, gy, gz) = (self.grid[0], self.grid[1], self.grid[2]);

        let coord = |v: f32, g: usize| -> (usize, f32) {
            if g <= 1 {
                return (0, 0.0);
            }
            let x = v.clamp(0.0, 1.0) * (g - 1) as f32;
            let l = (x.floor() as usize).min(g - 2);
            (l, x - l as f32)
        };
        let (x0, rx) = coord(input[0], gx);
        let (y0, ry) = coord(input[1], gy);
        let (z0, rz) = coord(input[2], gz);
        let x1 = (x0 + 1).min(gx.saturating_sub(1));
        let y1 = (y0 + 1).min(gy.saturating_sub(1));
        let z1 = (z0 + 1).min(gz.saturating_sub(1));

        // node value fetch: output channel o at grid (a,b,c)
        let v = |a: usize, b: usize, c: usize, o: usize| -> f32 {
            self.data[self.node_offset(&[a, b, c]) + o]
        };

        for (o, slot) in out.iter_mut().enumerate().take(m) {
            let c0 = v(x0, y0, z0, o);
            let (c1, c2, c3) = if rx >= ry && ry >= rz {
                (v(x1, y0, z0, o) - c0, v(x1, y1, z0, o) - v(x1, y0, z0, o), v(x1, y1, z1, o) - v(x1, y1, z0, o))
            } else if rx >= rz && rz >= ry {
                (v(x1, y0, z0, o) - c0, v(x1, y1, z1, o) - v(x1, y0, z1, o), v(x1, y0, z1, o) - v(x1, y0, z0, o))
            } else if rz >= rx && rx >= ry {
                (v(x1, y0, z1, o) - v(x0, y0, z1, o), v(x1, y1, z1, o) - v(x1, y0, z1, o), v(x0, y0, z1, o) - c0)
            } else if ry >= rx && rx >= rz {
                (v(x1, y1, z0, o) - v(x0, y1, z0, o), v(x0, y1, z0, o) - c0, v(x1, y1, z1, o) - v(x1, y1, z0, o))
            } else if ry >= rz && rz >= rx {
                (v(x1, y1, z1, o) - v(x0, y1, z1, o), v(x0, y1, z0, o) - c0, v(x0, y1, z1, o) - v(x0, y1, z0, o))
            } else {
                // rz >= ry >= rx
                (v(x1, y1, z1, o) - v(x0, y1, z1, o), v(x0, y1, z1, o) - v(x0, y0, z1, o), v(x0, y0, z1, o) - c0)
            };
            *slot = c0 + c1 * rx + c2 * ry + c3 * rz;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-input identity cLUT: node (i,j,k) → (i/(g-1), j/(g-1), k/(g-1)).
    fn identity_3d(g: usize) -> Clut {
        let mut data = Vec::with_capacity(g * g * g * 3);
        for i in 0..g {
            for j in 0..g {
                for k in 0..g {
                    data.push(i as f32 / (g - 1) as f32);
                    data.push(j as f32 / (g - 1) as f32);
                    data.push(k as f32 / (g - 1) as f32);
                }
            }
        }
        Clut { grid: vec![g, g, g], output_channels: 3, data }
    }

    #[test]
    fn tetrahedral_identity_recovers_input() {
        let lut = identity_3d(9);
        for &(r, g, b) in &[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.3, 0.6, 0.9), (0.51, 0.12, 0.77)] {
            let mut out = [0.0f32; 3];
            lut.eval(&[r, g, b], &mut out);
            assert!((out[0] - r).abs() < 1e-4 && (out[1] - g).abs() < 1e-4 && (out[2] - b).abs() < 1e-4,
                    "in=({r},{g},{b}) out={out:?}");
        }
    }

    #[test]
    fn nlinear_identity_recovers_input() {
        let lut = identity_3d(9);
        for &(r, g, b) in &[(0.25, 0.5, 0.75), (0.9, 0.1, 0.4)] {
            let mut out = [0.0f32; 3];
            lut.eval_nlinear(&[r, g, b], &mut out);
            assert!((out[0] - r).abs() < 1e-4 && (out[1] - g).abs() < 1e-4 && (out[2] - b).abs() < 1e-4,
                    "out={out:?}");
        }
    }

    #[test]
    fn tetra_and_nlinear_agree_at_grid_nodes() {
        // at exact grid nodes both methods return the node value exactly, for an
        // arbitrary (non-identity) table.
        let g = 4;
        let mut data = Vec::new();
        for i in 0..g {
            for j in 0..g {
                for k in 0..g {
                    data.push((i * 7 + j * 3 + k) as f32 * 0.01);
                    data.push((i + j * j + k) as f32 * 0.02);
                }
            }
        }
        let lut = Clut { grid: vec![g, g, g], output_channels: 2, data };
        for &(i, j, k) in &[(0, 0, 0), (3, 3, 3), (1, 2, 3), (2, 0, 1)] {
            let inp = [i as f32 / 3.0, j as f32 / 3.0, k as f32 / 3.0];
            let (mut a, mut b) = ([0.0f32; 2], [0.0f32; 2]);
            lut.eval_tetrahedral(&inp, &mut a);
            lut.eval_nlinear(&inp, &mut b);
            let node = lut.node_offset(&[i, j, k]);
            assert!((a[0] - lut.data[node]).abs() < 1e-5 && (a[1] - lut.data[node + 1]).abs() < 1e-5, "tetra {a:?}");
            assert!((b[0] - lut.data[node]).abs() < 1e-5, "nlinear {b:?}");
        }
    }

    #[test]
    fn one_dim_lut_interpolates() {
        // a 1-input, 1-output ramp doubling the input
        let lut = Clut { grid: vec![3], output_channels: 1, data: vec![0.0, 1.0, 2.0] };
        let mut out = [0.0f32; 1];
        lut.eval(&[0.5], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5, "{out:?}");
        lut.eval(&[0.25], &mut out);
        assert!((out[0] - 0.5).abs() < 1e-5, "{out:?}");
    }

    #[test]
    fn four_dim_nlinear_runs() {
        // CMYK-like 4-input → 3-output, just exercise the general path (finite).
        let g = 2;
        let lut = Clut { grid: vec![g; 4], output_channels: 3, data: vec![0.5f32; g * g * g * g * 3] };
        let mut out = [0.0f32; 3];
        lut.eval(&[0.2, 0.4, 0.6, 0.8], &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!((out[0] - 0.5).abs() < 1e-6); // constant table → constant out
    }
}
