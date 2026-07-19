//! ICC LUT-type transforms as a sequential **pipeline** of stages, and parsers
//! for the v2 LUT tag types `mft1` (lut8Type) and `mft2` (lut16Type).
//!
//! A [`Pipeline`] threads a colour vector through an ordered list of [`Stage`]s
//! (per-channel curves, a 3×3(+offset) matrix, an N-D cLUT). Both the v2 LUT
//! tags here and the v4 `mAB `/`mBA ` tags (next increment) parse into the same
//! representation, so evaluation is shared.
//!
//! All lengths coming from the (untrusted) profile are validated against the
//! actual tag bytes before any allocation (carrying the m4-89 DoS rule forward).

use super::clut::Clut;
use super::parser::{Curve, IccError};

/// One stage of a colour transform pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// One tone curve per channel (applied component-wise; `len` must match).
    Curves(Vec<Curve>),
    /// A 3×3 matrix with a 3-element offset (`m[i][3]`), applied to a 3-vector:
    /// `out[i] = Σ_j m[i][j]·in[j] + m[i][3]`.
    Matrix([[f32; 4]; 3]),
    /// An N-input → M-output colour lookup table.
    Clut(Clut),
}

/// An ordered colour transform.
#[derive(Debug, Clone, PartialEq)]
pub struct Pipeline {
    pub stages: Vec<Stage>,
}

impl Pipeline {
    /// Evaluate the pipeline on `input`, returning the transformed vector.
    pub fn eval(&self, input: &[f32]) -> Vec<f32> {
        let mut v = input.to_vec();
        for stage in &self.stages {
            v = match stage {
                Stage::Curves(curves) => {
                    v.iter().zip(curves.iter()).map(|(&x, c)| c.eval(x)).collect()
                }
                Stage::Matrix(m) => {
                    if v.len() < 3 {
                        v
                    } else {
                        let mut o = [0.0f32; 3];
                        for (i, oi) in o.iter_mut().enumerate() {
                            *oi = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + m[i][3];
                        }
                        o.to_vec()
                    }
                }
                Stage::Clut(clut) => {
                    let mut out = vec![0.0f32; clut.output_channels];
                    clut.eval(&v, &mut out);
                    out
                }
            };
        }
        v
    }
}

#[inline]
fn be_u16(b: &[u8], o: usize) -> Result<u16, IccError> {
    b.get(o..o + 2).map(|s| u16::from_be_bytes([s[0], s[1]])).ok_or(IccError::Truncated)
}
#[inline]
fn s15f16(b: &[u8], o: usize) -> Result<f32, IccError> {
    b.get(o..o + 4)
        .map(|s| (i32::from_be_bytes([s[0], s[1], s[2], s[3]])) as f32 / 65536.0)
        .ok_or(IccError::Truncated)
}

/// `grid^in_ch`, guarding overflow (untrusted `grid`/`in_ch`).
fn clut_nodes(grid: usize, in_ch: usize) -> Option<usize> {
    let mut n = 1usize;
    for _ in 0..in_ch {
        n = n.checked_mul(grid)?;
    }
    Some(n)
}

/// The e-matrix (offsets 12..48, 9× s15Fixed16), as a [`Stage::Matrix`] with zero
/// offset — returned only when it is not the identity (it only applies to a
/// 3-channel XYZ input per spec).
fn ematrix_stage(d: &[u8]) -> Result<Option<Stage>, IccError> {
    let mut m = [[0.0f32; 4]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().take(3).enumerate() {
            *cell = s15f16(d, 12 + (i * 3 + j) * 4)?;
        }
    }
    let identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    Ok(if m == identity { None } else { Some(Stage::Matrix(m)) })
}

/// A per-channel sampled curve from `count` table entries starting at `off`.
/// `u16_entries` true → u16 (÷65535); false → u8 (÷255, stored as u16·257).
fn read_table_curve(d: &[u8], off: usize, count: usize, u16_entries: bool) -> Result<Curve, IccError> {
    let bytes = if u16_entries { 2 } else { 1 };
    let end = off.checked_add(count * bytes).ok_or(IccError::Truncated)?;
    if end > d.len() {
        return Err(IccError::Truncated);
    }
    let mut t = Vec::with_capacity(count);
    for i in 0..count {
        if u16_entries {
            t.push(be_u16(d, off + i * 2)?);
        } else {
            // u8 0..255 → 0..65535 (255·257 = 65535)
            t.push(d[off + i] as u16 * 257);
        }
    }
    Ok(Curve::Table(t))
}

/// Parse a v2 LUT tag (`mft1` lut8 or `mft2` lut16) into a [`Pipeline`]:
/// `[e-matrix?] → input curves → CLUT → output curves`.
pub fn parse_lut_v2(d: &[u8]) -> Result<Pipeline, IccError> {
    if d.len() < 48 {
        return Err(IccError::Truncated);
    }
    let ty = &d[0..4];
    let is16 = match ty {
        b"mft2" => true,
        b"mft1" => false,
        _ => return Err(IccError::WrongTagType),
    };
    let in_ch = d[8] as usize;
    let out_ch = d[9] as usize;
    let grid = d[10] as usize;
    if in_ch == 0 || out_ch == 0 || grid < 2 {
        return Err(IccError::Truncated);
    }
    let entry_bytes = if is16 { 2 } else { 1 };

    // input/output table entry counts: fixed 256 for lut8; header fields for lut16
    let (n_in, n_out, mut off) = if is16 {
        (be_u16(d, 48)? as usize, be_u16(d, 50)? as usize, 52usize)
    } else {
        (256usize, 256usize, 48usize)
    };
    if n_in < 2 || n_out < 2 {
        return Err(IccError::Truncated);
    }

    let mut stages = Vec::new();
    if in_ch == 3 {
        if let Some(m) = ematrix_stage(d)? {
            stages.push(m);
        }
    }

    // input tables: one curve per input channel
    let mut input_curves = Vec::with_capacity(in_ch);
    for _ in 0..in_ch {
        input_curves.push(read_table_curve(d, off, n_in, is16)?);
        off += n_in * entry_bytes;
    }
    stages.push(Stage::Curves(input_curves));

    // CLUT: grid^in_ch nodes × out_ch entries
    let nodes = clut_nodes(grid, in_ch).ok_or(IccError::Truncated)?;
    let clut_vals = nodes.checked_mul(out_ch).ok_or(IccError::Truncated)?;
    let clut_end = off.checked_add(clut_vals * entry_bytes).ok_or(IccError::Truncated)?;
    if clut_end > d.len() {
        return Err(IccError::Truncated);
    }
    let mut data = Vec::with_capacity(clut_vals);
    for i in 0..clut_vals {
        if is16 {
            data.push(be_u16(d, off + i * 2)? as f32 / 65535.0);
        } else {
            data.push(d[off + i] as f32 / 255.0);
        }
    }
    off = clut_end;
    stages.push(Stage::Clut(Clut { grid: vec![grid; in_ch], output_channels: out_ch, data }));

    // output tables: one curve per output channel
    let mut output_curves = Vec::with_capacity(out_ch);
    for _ in 0..out_ch {
        output_curves.push(read_table_curve(d, off, n_out, is16)?);
        off += n_out * entry_bytes;
    }
    stages.push(Stage::Curves(output_curves));

    Ok(Pipeline { stages })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_be_u16(v: &mut Vec<u8>, x: u16) {
        v.extend_from_slice(&x.to_be_bytes());
    }

    /// Build a minimal identity `mft2`: 3→3, grid `g`, identity e-matrix, ramp
    /// input/output tables, and an identity CLUT (node (i,j,k) → normalised ijk).
    fn identity_mft2(g: usize, ntab: usize) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"mft2");
        d.extend_from_slice(&[0, 0, 0, 0]); // reserved
        d.push(3); // in
        d.push(3); // out
        d.push(g as u8); // grid
        d.push(0); // reserved
        // identity e-matrix (9 s15Fixed16)
        for i in 0..3 {
            for j in 0..3 {
                let val: i32 = if i == j { 65536 } else { 0 };
                d.extend_from_slice(&val.to_be_bytes());
            }
        }
        push_be_u16(&mut d, ntab as u16); // n input entries
        push_be_u16(&mut d, ntab as u16); // n output entries
        // input tables: identity ramp per channel
        for _ in 0..3 {
            for e in 0..ntab {
                push_be_u16(&mut d, ((e * 65535) / (ntab - 1)) as u16);
            }
        }
        // CLUT: identity
        for i in 0..g {
            for j in 0..g {
                for k in 0..g {
                    push_be_u16(&mut d, ((i * 65535) / (g - 1)) as u16);
                    push_be_u16(&mut d, ((j * 65535) / (g - 1)) as u16);
                    push_be_u16(&mut d, ((k * 65535) / (g - 1)) as u16);
                }
            }
        }
        // output tables: identity ramp
        for _ in 0..3 {
            for e in 0..ntab {
                push_be_u16(&mut d, ((e * 65535) / (ntab - 1)) as u16);
            }
        }
        d
    }

    #[test]
    fn parses_and_evaluates_identity_mft2() {
        let d = identity_mft2(9, 256);
        let p = parse_lut_v2(&d).unwrap();
        // identity e-matrix dropped; stages = input curves, clut, output curves
        assert_eq!(p.stages.len(), 3);
        for &(r, g, b) in &[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.3, 0.55, 0.82)] {
            let out = p.eval(&[r, g, b]);
            assert!((out[0] - r).abs() < 2e-3 && (out[1] - g).abs() < 2e-3 && (out[2] - b).abs() < 2e-3,
                    "in=({r},{g},{b}) out={out:?}");
        }
    }

    #[test]
    fn rejects_truncated_and_bad_type() {
        assert_eq!(parse_lut_v2(&[0u8; 10]).unwrap_err(), IccError::Truncated);
        let mut d = identity_mft2(2, 2);
        d[0..4].copy_from_slice(b"XYZ ");
        assert_eq!(parse_lut_v2(&d).unwrap_err(), IccError::WrongTagType);
        // a header claiming a huge grid must be rejected before allocating
        let mut d2 = identity_mft2(9, 8);
        d2[10] = 255; // grid 255 → 255^3·3 values, far exceeds the tag bytes
        assert_eq!(parse_lut_v2(&d2).unwrap_err(), IccError::Truncated);
    }

    #[test]
    fn pipeline_matrix_stage_applies_offset() {
        let m = Stage::Matrix([[2.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.2]]);
        let p = Pipeline { stages: vec![m] };
        let out = p.eval(&[0.5, 0.3, 0.4]);
        assert!((out[0] - (2.0 * 0.5 + 0.1)).abs() < 1e-6);
        assert!((out[1] - 0.3).abs() < 1e-6);
        assert!((out[2] - (0.4 - 0.2)).abs() < 1e-6);
    }
}
