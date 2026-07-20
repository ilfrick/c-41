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
    // checked (same fragile pattern as the CLUT sizing; robust for reuse by the
    // v4 mAB/mBA parser where `count` comes from other fields).
    let span = count.checked_mul(bytes).ok_or(IccError::Truncated)?;
    let end = off.checked_add(span).ok_or(IccError::Truncated)?;
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
    // Cap channel counts (LCMS cmsMAXCHANNELS = 16) to kill the `grid^in_ch`
    // overflow class at the source, on top of the checked arithmetic below.
    if in_ch == 0 || out_ch == 0 || grid < 2 || in_ch > 16 || out_ch > 16 {
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
    // checked multiply: a crafted in_ch could make clut_vals·entry_bytes wrap and
    // slip past the length guard (→ huge with_capacity abort). See regression test.
    let clut_bytes = clut_vals.checked_mul(entry_bytes).ok_or(IccError::Truncated)?;
    let clut_end = off.checked_add(clut_bytes).ok_or(IccError::Truncated)?;
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

#[inline]
fn be_u32(b: &[u8], o: usize) -> Result<usize, IccError> {
    b.get(o..o + 4)
        .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
        .ok_or(IccError::Truncated)
}

/// Parse `count` consecutive inline tone curves starting at `off`, each padded to
/// a 4-byte boundary (the v4 `mAB `/`mBA ` curve-set layout).
fn parse_curves(d: &[u8], off: usize, count: usize) -> Result<Vec<Curve>, IccError> {
    let mut curves = Vec::with_capacity(count);
    let mut p = off;
    for _ in 0..count {
        let slice = d.get(p..).ok_or(IccError::Truncated)?;
        let (c, len) = super::parser::parse_curve(slice)?;
        curves.push(c);
        let padded = (len + 3) & !3; // 4-byte aligned
        p = p.checked_add(padded).ok_or(IccError::Truncated)?;
    }
    Ok(curves)
}

/// The v4 matrix element: 12 s15Fixed16 (3×3 `e00..e22` then offset `e30 e31 e32`).
fn parse_matrix_v4(d: &[u8], off: usize) -> Result<[[f32; 4]; 3], IccError> {
    let v = |i: usize| s15f16(d, off + i * 4);
    Ok([
        [v(0)?, v(1)?, v(2)?, v(9)?],
        [v(3)?, v(4)?, v(5)?, v(10)?],
        [v(6)?, v(7)?, v(8)?, v(11)?],
    ])
}

/// The v4 CLUT element: 16 grid-point bytes (one per input channel), precision
/// byte @16 (1=u8, 2=u16), reserved 3, then the data. Grids may differ per axis.
fn parse_clut_v4(d: &[u8], off: usize, in_ch: usize, out_ch: usize) -> Result<Clut, IccError> {
    if off.checked_add(20).ok_or(IccError::Truncated)? > d.len() {
        return Err(IccError::Truncated);
    }
    let mut grid = Vec::with_capacity(in_ch);
    let mut nodes = 1usize;
    for c in 0..in_ch {
        let g = d[off + c] as usize;
        if g < 2 {
            return Err(IccError::Truncated);
        }
        grid.push(g);
        nodes = nodes.checked_mul(g).ok_or(IccError::Truncated)?;
    }
    let entry_bytes = match d[off + 16] {
        1 => 1usize,
        2 => 2,
        _ => return Err(IccError::Truncated),
    };
    let vals = nodes.checked_mul(out_ch).ok_or(IccError::Truncated)?;
    let span = vals.checked_mul(entry_bytes).ok_or(IccError::Truncated)?;
    let data_off = off + 20;
    let end = data_off.checked_add(span).ok_or(IccError::Truncated)?;
    if end > d.len() {
        return Err(IccError::Truncated);
    }
    let mut data = Vec::with_capacity(vals);
    for i in 0..vals {
        if entry_bytes == 2 {
            data.push(be_u16(d, data_off + i * 2)? as f32 / 65535.0);
        } else {
            data.push(d[data_off + i] as f32 / 255.0);
        }
    }
    Ok(Clut { grid, output_channels: out_ch, data })
}

/// Parse a v4 LUT tag (`mAB ` lutAtoBType or `mBA ` lutBtoAType) into a
/// [`Pipeline`]. Elements are located via the offset table (offset 0 = absent):
/// - `mAB ` (device→PCS): A curves → CLUT → M curves → matrix → B curves.
/// - `mBA ` (PCS→device): B curves → matrix → M curves → CLUT → A curves.
///
/// Curve-set channel counts follow the dimensional flow (mAB: A=in, M/B=out;
/// mBA: B/M=in, A=out).
pub fn parse_lut_v4(d: &[u8]) -> Result<Pipeline, IccError> {
    if d.len() < 32 {
        return Err(IccError::Truncated);
    }
    let is_ab = match &d[0..4] {
        b"mAB " => true,
        b"mBA " => false,
        _ => return Err(IccError::WrongTagType),
    };
    let in_ch = d[8] as usize;
    let out_ch = d[9] as usize;
    if in_ch == 0 || out_ch == 0 || in_ch > 16 || out_ch > 16 {
        return Err(IccError::Truncated);
    }
    let (off_b, off_mat, off_m, off_clut, off_a) =
        (be_u32(d, 12)?, be_u32(d, 16)?, be_u32(d, 20)?, be_u32(d, 24)?, be_u32(d, 28)?);

    let mut stages = Vec::new();
    if is_ab {
        // mAB: A(in) → CLUT → M(out) → Matrix → B(out)
        if off_a != 0 {
            stages.push(Stage::Curves(parse_curves(d, off_a, in_ch)?));
        }
        if off_clut != 0 {
            stages.push(Stage::Clut(parse_clut_v4(d, off_clut, in_ch, out_ch)?));
        }
        if off_m != 0 {
            stages.push(Stage::Curves(parse_curves(d, off_m, out_ch)?));
        }
        if off_mat != 0 {
            // the 3×3 matrix only applies to a 3-channel (XYZ/Lab) PCS
            if out_ch != 3 {
                return Err(IccError::WrongTagType);
            }
            stages.push(Stage::Matrix(parse_matrix_v4(d, off_mat)?));
        }
        if off_b != 0 {
            stages.push(Stage::Curves(parse_curves(d, off_b, out_ch)?));
        }
    } else {
        // mBA: B(in) → Matrix → M(in) → CLUT → A(out)
        if off_b != 0 {
            stages.push(Stage::Curves(parse_curves(d, off_b, in_ch)?));
        }
        if off_mat != 0 {
            // the 3×3 matrix only applies to a 3-channel (XYZ/Lab) PCS
            if in_ch != 3 {
                return Err(IccError::WrongTagType);
            }
            stages.push(Stage::Matrix(parse_matrix_v4(d, off_mat)?));
        }
        if off_m != 0 {
            stages.push(Stage::Curves(parse_curves(d, off_m, in_ch)?));
        }
        if off_clut != 0 {
            stages.push(Stage::Clut(parse_clut_v4(d, off_clut, in_ch, out_ch)?));
        }
        if off_a != 0 {
            stages.push(Stage::Curves(parse_curves(d, off_a, out_ch)?));
        }
    }
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
    fn rejects_input_channel_count_overflow() {
        // grid=2, in_ch=63 → 2^63 nodes; ×entry_bytes(2) = 2^64 wraps usize and
        // would slip past the CLUT bounds check without checked math / the cap.
        let (in_ch, n) = (63usize, 2usize);
        let mut d = Vec::new();
        d.extend_from_slice(b"mft2");
        d.extend_from_slice(&[0, 0, 0, 0]);
        d.push(in_ch as u8);
        d.push(1);
        d.push(2);
        d.push(0);
        for i in 0..3 {
            for j in 0..3 {
                let v: i32 = if i == j { 65536 } else { 0 };
                d.extend_from_slice(&v.to_be_bytes());
            }
        }
        push_be_u16(&mut d, n as u16);
        push_be_u16(&mut d, n as u16);
        for _ in 0..in_ch {
            for e in 0..n {
                push_be_u16(&mut d, ((e * 65535) / (n - 1)) as u16);
            }
        }
        assert_eq!(parse_lut_v2(&d).unwrap_err(), IccError::Truncated);
    }

    /// Three identity `curv` curves (count 0), 12 bytes each.
    fn identity_curves3() -> Vec<u8> {
        let mut v = Vec::new();
        for _ in 0..3 {
            v.extend_from_slice(b"curv");
            v.extend_from_slice(&[0, 0, 0, 0]); // reserved
            v.extend_from_slice(&0u32.to_be_bytes()); // count = 0 → identity
        }
        v
    }

    #[test]
    fn parses_and_evaluates_identity_mab() {
        // mAB, 3→3, A curves (identity) → CLUT (identity, grid 2³) → B curves.
        let a_curves = identity_curves3(); // 36 bytes
        // CLUT: 16 grid bytes + precision(2=u16) + 3 reserved + 48 data bytes
        let mut clut = Vec::new();
        clut.extend_from_slice(&[2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // grids
        clut.push(2); // precision = u16
        clut.extend_from_slice(&[0, 0, 0]); // reserved
        for i in 0..2u16 {
            for j in 0..2u16 {
                for k in 0..2u16 {
                    push_be_u16(&mut clut, i * 65535);
                    push_be_u16(&mut clut, j * 65535);
                    push_be_u16(&mut clut, k * 65535);
                }
            }
        }
        let b_curves = identity_curves3();

        let off_a = 32usize;
        let off_clut = off_a + a_curves.len();
        let off_b = off_clut + clut.len();

        let mut d = Vec::new();
        d.extend_from_slice(b"mAB ");
        d.extend_from_slice(&[0, 0, 0, 0]);
        d.push(3); // in
        d.push(3); // out
        d.extend_from_slice(&[0, 0]); // reserved
        d.extend_from_slice(&(off_b as u32).to_be_bytes()); // B curves offset
        d.extend_from_slice(&0u32.to_be_bytes()); // matrix (none)
        d.extend_from_slice(&0u32.to_be_bytes()); // M curves (none)
        d.extend_from_slice(&(off_clut as u32).to_be_bytes()); // CLUT
        d.extend_from_slice(&(off_a as u32).to_be_bytes()); // A curves
        d.extend_from_slice(&a_curves);
        d.extend_from_slice(&clut);
        d.extend_from_slice(&b_curves);

        let p = parse_lut_v4(&d).unwrap();
        assert_eq!(p.stages.len(), 3); // A curves, CLUT, B curves
        for &(r, g, b) in &[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.2, 0.7, 0.45)] {
            let out = p.eval(&[r, g, b]);
            assert!((out[0] - r).abs() < 1e-3 && (out[1] - g).abs() < 1e-3 && (out[2] - b).abs() < 1e-3,
                    "in=({r},{g},{b}) out={out:?}");
        }
    }

    /// Three gamma `curv` curves (count 1, u8Fixed8), each padded to 16 bytes.
    fn gamma_curves3(gamma: f32) -> Vec<u8> {
        let fixed = (gamma * 256.0).round() as u16;
        let mut v = Vec::new();
        for _ in 0..3 {
            v.extend_from_slice(b"curv");
            v.extend_from_slice(&[0, 0, 0, 0]);
            v.extend_from_slice(&1u32.to_be_bytes()); // count = 1 → gamma
            push_be_u16(&mut v, fixed);
            v.extend_from_slice(&[0, 0]); // pad 14 → 16 (4-byte align)
        }
        v
    }

    fn build_mab(sig: &[u8; 4], a: &[u8], clut: &[u8], b: &[u8]) -> Vec<u8> {
        let off_a = 32usize;
        let off_clut = off_a + a.len();
        let off_b = off_clut + clut.len();
        let mut d = Vec::new();
        d.extend_from_slice(sig);
        d.extend_from_slice(&[0, 0, 0, 0]);
        d.push(3); // in
        d.push(3); // out
        d.extend_from_slice(&[0, 0]);
        // For mAB: off_b@12, mat@16, m@20, clut@24, a@28.
        d.extend_from_slice(&(off_b as u32).to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes()); // matrix
        d.extend_from_slice(&0u32.to_be_bytes()); // M curves
        d.extend_from_slice(&(off_clut as u32).to_be_bytes());
        d.extend_from_slice(&(off_a as u32).to_be_bytes());
        d.extend_from_slice(a);
        d.extend_from_slice(clut);
        d.extend_from_slice(b);
        d
    }

    fn identity_clut_2x2x2() -> Vec<u8> {
        let mut clut = Vec::new();
        clut.extend_from_slice(&[2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        clut.push(2); // u16
        clut.extend_from_slice(&[0, 0, 0]);
        for i in 0..2u16 {
            for j in 0..2u16 {
                for k in 0..2u16 {
                    push_be_u16(&mut clut, i * 65535);
                    push_be_u16(&mut clut, j * 65535);
                    push_be_u16(&mut clut, k * 65535);
                }
            }
        }
        clut
    }

    #[test]
    fn mab_stage_order_pins_a_before_b() {
        // Asymmetric A (gamma 2.0) vs B (gamma 3.0): the parse must place A first
        // and B last — an A/B swap would flip the gammas and fail this.
        let d = build_mab(b"mAB ", &gamma_curves3(2.0), &identity_clut_2x2x2(), &gamma_curves3(3.0));
        let p = parse_lut_v4(&d).unwrap();
        assert_eq!(p.stages.len(), 3, "A curves, CLUT, B curves");
        match &p.stages[0] {
            Stage::Curves(cs) => assert!(matches!(cs[0], Curve::Gamma(g) if (g - 2.0).abs() < 1e-2),
                "stage0 (A) should be gamma 2.0, got {:?}", cs[0]),
            s => panic!("stage 0 not curves: {s:?}"),
        }
        assert!(matches!(&p.stages[1], Stage::Clut(_)), "stage 1 should be the CLUT");
        match &p.stages[2] {
            Stage::Curves(cs) => assert!(matches!(cs[0], Curve::Gamma(g) if (g - 3.0).abs() < 1e-2),
                "stage2 (B) should be gamma 3.0, got {:?}", cs[0]),
            s => panic!("stage 2 not curves: {s:?}"),
        }
    }

    #[test]
    fn mba_identity_roundtrip_and_order() {
        // mBA (B → matrix → M → CLUT → A): identity B, CLUT, A (no matrix/M).
        // Offsets differ from mAB (B@12 ... A@28) — build directly.
        let b_curves = identity_curves3();
        let clut = identity_clut_2x2x2();
        let a_curves = identity_curves3();
        let off_b = 32usize;
        let off_clut = off_b + b_curves.len();
        let off_a = off_clut + clut.len();
        let mut d = Vec::new();
        d.extend_from_slice(b"mBA ");
        d.extend_from_slice(&[0, 0, 0, 0]);
        d.push(3);
        d.push(3);
        d.extend_from_slice(&[0, 0]);
        d.extend_from_slice(&(off_b as u32).to_be_bytes()); // B @12
        d.extend_from_slice(&0u32.to_be_bytes()); // matrix @16
        d.extend_from_slice(&0u32.to_be_bytes()); // M @20
        d.extend_from_slice(&(off_clut as u32).to_be_bytes()); // CLUT @24
        d.extend_from_slice(&(off_a as u32).to_be_bytes()); // A @28
        d.extend_from_slice(&b_curves);
        d.extend_from_slice(&clut);
        d.extend_from_slice(&a_curves);

        let p = parse_lut_v4(&d).unwrap();
        assert_eq!(p.stages.len(), 3);
        assert!(matches!(&p.stages[1], Stage::Clut(_)), "matrix/M absent → CLUT is the middle stage");
        for &(r, g, b) in &[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.4, 0.15, 0.85)] {
            let out = p.eval(&[r, g, b]);
            assert!((out[0] - r).abs() < 1e-3 && (out[1] - g).abs() < 1e-3 && (out[2] - b).abs() < 1e-3,
                    "mBA out={out:?}");
        }
    }

    #[test]
    fn v4_matrix_requires_three_channel_pcs() {
        // an mAB with ONLY a matrix but out_ch != 3 (matrix defined only for a
        // 3-channel PCS) must be rejected before building the stage.
        let mut d = vec![0u8; 80];
        d[0..4].copy_from_slice(b"mAB ");
        d[8] = 3; // in
        d[9] = 4; // out != 3
        d[16..20].copy_from_slice(&32u32.to_be_bytes()); // matrix offset (A/CLUT/M/B = 0)
        assert_eq!(parse_lut_v4(&d).unwrap_err(), IccError::WrongTagType);
    }

    #[test]
    fn v4_rejects_bad_type_and_truncation() {
        assert_eq!(parse_lut_v4(&[0u8; 20]).unwrap_err(), IccError::Truncated);
        let mut d = vec![0u8; 40];
        d[0..4].copy_from_slice(b"mft2");
        d[8] = 3;
        d[9] = 3;
        assert_eq!(parse_lut_v4(&d).unwrap_err(), IccError::WrongTagType);
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
