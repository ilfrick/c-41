use crate::{params::IopParams, roi::RoiIn, Result};
use super::IopProcess;

pub struct Grain;

impl IopProcess for Grain {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut super::ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "grain" }
}

// ----------------------------------------------------------------------------
// Stefan Gustavson's 3D simplex noise — ported from grain.c
// ----------------------------------------------------------------------------

#[rustfmt::skip]
static GRAD3: [[f64; 3]; 12] = [
    [ 1.0,  1.0,  0.0], [-1.0,  1.0,  0.0],
    [ 1.0, -1.0,  0.0], [-1.0, -1.0,  0.0],
    [ 1.0,  0.0,  1.0], [-1.0,  0.0,  1.0],
    [ 1.0,  0.0, -1.0], [-1.0,  0.0, -1.0],
    [ 0.0,  1.0,  1.0], [ 0.0, -1.0,  1.0],
    [ 0.0,  1.0, -1.0], [ 0.0, -1.0, -1.0],
];

#[rustfmt::skip]
static PERMUTATION: [usize; 256] = [
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
];

fn build_perm() -> ([usize; 512], [usize; 512]) {
    let mut perm = [0usize; 512];
    let mut perm_mod = [0usize; 512];
    for i in 0..512 {
        perm[i] = PERMUTATION[i & 255];
        perm_mod[i] = perm[i] % 12;
    }
    (perm, perm_mod)
}

#[inline(always)]
fn fastfloor(x: f64) -> i32 {
    if x > 0.0 { x as i32 } else { x as i32 - 1 }
}

#[inline(always)]
fn dot3(g: &[f64; 3], x: f64, y: f64, z: f64) -> f64 {
    g[0] * x + g[1] * y + g[2] * z
}

fn simplex_noise(xin: f64, yin: f64, zin: f64, perm: &[usize; 512], perm_mod: &[usize; 512]) -> f64 {
    let f3 = 1.0 / 3.0;
    let g3 = 1.0 / 6.0;
    let s = (xin + yin + zin) * f3;
    let i = fastfloor(xin + s);
    let j = fastfloor(yin + s);
    let k = fastfloor(zin + s);
    let t = (i + j + k) as f64 * g3;
    let x0 = xin - (i as f64 - t);
    let y0 = yin - (j as f64 - t);
    let z0 = zin - (k as f64 - t);

    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0      { (1,0,0, 1,1,0) }
        else if x0 >= z0 { (1,0,0, 1,0,1) }
        else             { (0,0,1, 1,0,1) }
    } else {
        if y0 < z0       { (0,0,1, 0,1,1) }
        else if x0 < z0  { (0,1,0, 0,1,1) }
        else             { (0,1,0, 1,1,0) }
    };

    let x1 = x0 - i1 as f64 + g3;
    let y1 = y0 - j1 as f64 + g3;
    let z1 = z0 - k1 as f64 + g3;
    let x2 = x0 - i2 as f64 + 2.0 * g3;
    let y2 = y0 - j2 as f64 + 2.0 * g3;
    let z2 = z0 - k2 as f64 + 2.0 * g3;
    let x3 = x0 - 1.0 + 3.0 * g3;
    let y3 = y0 - 1.0 + 3.0 * g3;
    let z3 = z0 - 1.0 + 3.0 * g3;

    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let kk = (k & 255) as usize;
    let gi0 = perm_mod[ii     + perm[jj     + perm[kk    ]]];
    let gi1 = perm_mod[ii+i1  + perm[jj+j1  + perm[kk+k1 ]]];
    let gi2 = perm_mod[ii+i2  + perm[jj+j2  + perm[kk+k2 ]]];
    let gi3 = perm_mod[ii+1   + perm[jj+1   + perm[kk+1  ]]];

    let corner = |gi: usize, x: f64, y: f64, z: f64| -> f64 {
        let t = 0.6 - x*x - y*y - z*z;
        if t < 0.0 { 0.0 } else { let t2 = t*t; t2*t2 * dot3(&GRAD3[gi], x, y, z) }
    };

    32.0 * (corner(gi0, x0, y0, z0)
          + corner(gi1, x1, y1, z1)
          + corner(gi2, x2, y2, z2)
          + corner(gi3, x3, y3, z3))
}

fn simplex_2d_noise(x: f64, y: f64, z: f64, perm: &[usize; 512], perm_mod: &[usize; 512]) -> f64 {
    const F: [f64; 3] = [0.4910, 0.9441, 1.7280];
    const A: [f64; 3] = [0.2340, 0.7850, 1.2150];
    let mut total = 0.0;
    for octave in 0..3 {
        total += simplex_noise(x * F[octave] / z, y * F[octave] / z, octave as f64, perm, perm_mod) * A[octave];
    }
    total
}

// ----------------------------------------------------------------------------
// Grain LUT (128×128 photographic paper response model)
// ----------------------------------------------------------------------------

/// Side length of the grain LUT (grain.c `GRAIN_LUT_SIZE`). Public because
/// [`grain_lut`]'s return type names it.
pub const GRAIN_LUT_SIZE: usize = 128;
const GRAIN_LUT_DELTA_MAX: f32 = 2.0;
const GRAIN_LUT_DELTA_MIN: f32 = 0.0001;
const GRAIN_LUT_PAPER_GAMMA: f32 = 1.0;

fn paper_resp(exposure: f32, mb: f32, gp: f32) -> f32 {
    let delta = GRAIN_LUT_DELTA_MAX * ((mb / 100.0) * GRAIN_LUT_DELTA_MIN.ln()).exp();
    (1.0 + 2.0 * delta) / (1.0 + ((4.0 * gp * (0.5 - exposure)) / (1.0 + 2.0 * delta)).exp()) - delta
}

fn paper_resp_inverse(density: f32, mb: f32, gp: f32) -> f32 {
    let delta = GRAIN_LUT_DELTA_MAX * ((mb / 100.0) * GRAIN_LUT_DELTA_MIN.ln()).exp();
    -((1.0 + 2.0 * delta) / (density + delta) - 1.0).ln() * (1.0 + 2.0 * delta) / (4.0 * gp) + 0.5
}

fn build_grain_lut(mb: f32) -> [f32; GRAIN_LUT_SIZE * GRAIN_LUT_SIZE] {
    let mut lut = [0f32; GRAIN_LUT_SIZE * GRAIN_LUT_SIZE];
    for i in 0..GRAIN_LUT_SIZE {
        for j in 0..GRAIN_LUT_SIZE {
            let gu = i as f32 / (GRAIN_LUT_SIZE - 1) as f32 - 0.5;
            let l  = j as f32 / (GRAIN_LUT_SIZE - 1) as f32;
            lut[j * GRAIN_LUT_SIZE + i] = 100.0
                * (paper_resp(gu + paper_resp_inverse(l, mb, GRAIN_LUT_PAPER_GAMMA), mb, GRAIN_LUT_PAPER_GAMMA) - l);
        }
    }
    lut
}

/// Memo of the last grain LUT built on this thread, keyed on `midtones_bias`.
///
/// `Pipeline::process` splits the image into ~64k-pixel bands and runs each stage
/// **once per band**, so an unmemoised builder would re-evaluate the paper
/// response (2 × 16 384 `exp`/`log` pairs) for a byte-identical table. Keyed on
/// the single input, so this is pure memoisation; thread-local rather than
/// shared, so it needs no lock and the `Rc` never crosses a thread — the same
/// pattern as `toneequal::CORRECTION_LUT_CACHE`.
type Cached = std::cell::RefCell<Option<(u32, std::rc::Rc<[f32; GRAIN_LUT_SIZE * GRAIN_LUT_SIZE]>)>>;

thread_local! {
    static GRAIN_LUT_CACHE: Cached = const { std::cell::RefCell::new(None) };
}

/// The 128×128 grain LUT for `midtones_bias` (`mb`), memoised per thread.
///
/// `commit_params` calls `evaluate_grain_lut(d->grain_lut, d->midtones_bias)`
/// once per commit; we do the same once per render (keyed on `mb`) and share
/// the table across the render's bands.
pub fn grain_lut(mb: f32) -> std::rc::Rc<[f32; GRAIN_LUT_SIZE * GRAIN_LUT_SIZE]> {
    let key = mb.to_bits();
    GRAIN_LUT_CACHE.with(|c| {
        if let Some((k, lut)) = c.borrow().as_ref() {
            if *k == key {
                return lut.clone();
            }
        }
        let rc: std::rc::Rc<[f32; GRAIN_LUT_SIZE * GRAIN_LUT_SIZE]> =
            std::rc::Rc::from(build_grain_lut(mb));
        *c.borrow_mut() = Some((key, rc.clone()));
        rc
    })
}

/// darktable's `_hash_string` (grain.c:201) — djb2 XOR-folded **backwards**.
///
/// The C walks the filename from the last character to the first on purpose
/// (comment at grain.c:196-200): image sequences differ in their trailing
/// digits, so hashing the tail first makes those digits dominate the result and
/// keeps a video's frames from sharing one static grain pattern. The initial
/// value 5381 and the `((hash << 5) + hash) ^ c` step are djb2.
///
/// Two C details are reproduced rather than approximated, because a filename
/// need not be ASCII:
/// * `char` is signed on x86, so `str[i]` **sign-extends** to `int` and the XOR
///   runs on that. For a non-ASCII UTF-8 byte (≥ 0x80) the C folds the sign-
///   extended value, not the raw byte — hence the `as i8 as i32` cast.
/// * `hash` is `unsigned int`, so the shift-and-add wraps at 32 bits (C leaves
///   that to `unsigned` arithmetic) — hence `wrapping_add` on the `<< 5`.
pub fn hash_string(name: &str) -> u32 {
    let bytes = name.as_bytes();
    let mut hash: u32 = 5381;
    for &b in bytes.iter().rev() {
        hash = (hash << 5).wrapping_add(hash) ^ (b as i8 as i32 as u32);
    }
    hash
}

fn lut_lookup_2d(grain_lut: &[f32], x: f32, y: f32) -> f32 {
    let sz = GRAIN_LUT_SIZE as f32;
    let _x = ((x + 0.5) * (sz - 1.0)).clamp(0.0, sz - 1.0);
    let _y = (y * (sz - 1.0)).clamp(0.0, sz - 1.0);
    let x0 = if _x < sz - 2.0 { _x as usize } else { GRAIN_LUT_SIZE - 2 };
    let y0 = if _y < sz - 2.0 { _y as usize } else { GRAIN_LUT_SIZE - 2 };
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let xd = _x - x0 as f32;
    let yd = _y - y0 as f32;
    let l00 = grain_lut[y0 * GRAIN_LUT_SIZE + x0];
    let l01 = grain_lut[y0 * GRAIN_LUT_SIZE + x1];
    let l10 = grain_lut[y1 * GRAIN_LUT_SIZE + x0];
    let l11 = grain_lut[y1 * GRAIN_LUT_SIZE + x1];
    let xy0 = (1.0 - yd) * l00 + l10 * yd;
    let xy1 = (1.0 - yd) * l01 + l11 * yd;
    xy0 * (1.0 - xd) + xy1 * xd
}

const GRAIN_LIGHTNESS_STRENGTH_SCALE: f32 = 0.15;

/// Grain IOP — simulate silver grain using simplex noise on the L channel.
///
/// Implements both the fast (non-filter) and downsampled (filter) paths from grain.c.
///
/// Caller pre-computes from C:
///   strength = data->strength / 100.0
///   zoom     = (1.0 + 8*data->scale/100) / 800.0
///   wd       = fminf(piece->buf_in.width, piece->buf_in.height)
///   scale    = roi_out->scale
///   hash     = _hash_string(filename) % max(roi->width*0.3, 1)
///   filter   = !fastmode && fabsf(roi_out->scale - 1.0f) > 0.01f
///   filtermul = piece->iscale / (roi_out->scale * wd)   [only used when filter != 0]
///   grain_lut = data->grain_lut (128×128 floats from commit_params)
#[no_mangle]
pub unsafe extern "C" fn darkroom_grain_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    roi_x: i32,
    roi_y: i32,
    width: i32,
    height: i32,
    strength: f32,
    zoom: f64,
    wd: f64,
    scale: f64,
    hash: i32,
    filter: i32,       // 0 = fast path; non-zero = rank-1 lattice downsampling
    filtermul: f64,
    grain_lut: *const f32, // 128×128 floats from data->grain_lut
) {
    const FIB1: f64 = 34.0;
    const FIB2: f64 = 21.0;
    const FIB1DIV2: f64 = FIB1 / FIB2;
    const FIB2INV: f64 = 1.0 / FIB2;

    let (perm, perm_mod) = build_perm();
    let w = width as usize;
    let h = height as usize;
    let inp = std::slice::from_raw_parts(in_buf, w * h * 4);
    let out = std::slice::from_raw_parts_mut(out_buf, w * h * 4);
    let lut = std::slice::from_raw_parts(grain_lut, GRAIN_LUT_SIZE * GRAIN_LUT_SIZE);

    for j in 0..h {
        let wy = (roi_y + j as i32) as f64 / scale;
        let y = wy / wd;
        for i in 0..w {
            let wx = (roi_x + i as i32) as f64 / scale;
            let x = wx / wd;

            let noise = if filter != 0 {
                let mut n = 0.0f64;
                for l in 0..FIB2 as usize {
                    let px = l as f64 / FIB2;
                    let mut py = l as f64 * FIB1DIV2;
                    py -= py as i64 as f64; // fmod 1
                    let dx = px * filtermul;
                    let dy = py * filtermul;
                    n += FIB2INV * simplex_2d_noise(x + dx + hash as f64, y + dy, zoom, &perm, &perm_mod);
                }
                n as f32
            } else {
                simplex_2d_noise(x + hash as f64, y, zoom, &perm, &perm_mod) as f32
            };

            let base = (j * w + i) * 4;
            out[base + 0] = inp[base + 0]
                + lut_lookup_2d(lut, noise * strength * GRAIN_LIGHTNESS_STRENGTH_SCALE, inp[base + 0] / 100.0);
            out[base + 1] = inp[base + 1];
            out[base + 2] = inp[base + 2];
            out[base + 3] = inp[base + 3];
        }
    }
}

/// Apply the grain IOP to a packed RGBA `f32` buffer, deriving every argument
/// [`darkroom_grain_process`] needs the way `grain.c::process` does.
///
/// `input`/`output` are the same length (a multiple of 4) and describe a whole
/// `width × height` frame — the C's `roi_out`, with `roi_x = roi_y = 0`.
///
/// * `coarseness` — the C's `data->scale`: 20/213.2 = 0.0938 … 6400/213.2 =
///   30.02 after the `GRAIN_SCALE_FACTOR` division (`grain.c:45, 66-68`).
/// * `strength` — the raw slider, 0..100; divided by 100 as in the C.
/// * `midtones_bias` — 0..100; selects the paper-response LUT (memoised here).
/// * `scale` — the buffer's resolution relative to the full image (darktable's
///   `roi_out->scale`). **Every current c41 funnel passes `1.0`**: preview and
///   export each hand the stage a whole frame at the resolution being rendered,
///   rather than darktable's fixed full-resolution buffer with a scaled ROI. The
///   kernel divides pixel indices by it, so a future downscaled caller gets the
///   *same* grain field as the full-res export instead of grain scaled to the
///   thumbnail. It must be `> 0` — the kernel itself divides by it.
/// * `hash_seed` — [`hash_string`] of the source filename. The C reduces it by
///   the ROI width here (`% (int)fmax(roi_out->width*0.3, 1.0)`), which is what
///   keeps consecutive frames of a sequence from repeating one pattern.
///
/// Two documented deviations, both from c41 having no equivalent concept:
///
/// * `dt_pipe_is_fast(pipe)` is always false here — c41 has no separate
///   fast/zoom-pan preview. `filter` is therefore `|scale - 1| > 0.01`, but
///   since every c41 funnel also passes `scale = 1.0` (see above), `filter` is
///   **0 in the product**: the rank-1 lattice downsampling branch in the kernel
///   is currently unreachable. It is kept — and exercised directly by a test —
///   because it is part of the ported kernel and a caller that does pass a
///   downscaled `scale` needs it; claiming it "runs on downscaled previews"
///   would be false, because c41 has no such preview.
/// * `piece->iscale` is the *input buffer's* scale, a genuinely different
///   quantity from `roi_out->scale`. Every c41 funnel hands the stage a whole
///   frame whose resolution is the scale it is asked to process at, so
///   `iscale == scale` here and `filtermul` collapses to `1/wd`. Both terms are
///   kept in the expression so the assumption stays visible: a future caller
///   with a differently-scaled input buffer must set them apart, and writing
///   the collapsed `1/wd` would hide that.
// Nine arguments because the C derives them one-for-one from its own locals;
// bundling them behind a struct would only re-obscure which line of `process`
// each one came from.
#[allow(clippy::too_many_arguments)]
pub fn process(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    scale: f32,
    coarseness: f32,
    strength: f32,
    midtones_bias: f32,
    hash_seed: u32,
) {
    // `process` is a safe `pub` fn: a short slice here would become an
    // out-of-bounds read/write inside the kernel's `from_raw_parts(in, w*h*4)`,
    // so the length contract is checked in release builds too, not just debug.
    // (Same trust boundary `Pipeline::process` asserts at its own entry.)
    assert!(
        input.len() >= width * height * 4,
        "grain: input has {} floats, need {} for a {width}×{height} RGBA frame",
        input.len(),
        width * height * 4
    );
    assert!(
        output.len() >= width * height * 4,
        "grain: output has {} floats, need {} for a {width}×{height} RGBA frame",
        output.len(),
        width * height * 4
    );
    // The kernel divides pixel indices by `scale`; a zero or negative scale is
    // a caller-contract violation even though f32 division would not trap.
    assert!(scale > 0.0, "grain: the kernel divides pixel indices by `scale`");
    let lut = grain_lut(midtones_bias);

    // grain.c:224 — `% (int)fmax(roi_out->width * 0.3, 1.0)`: `roi_out->width`
    // is an `int` and `0.3` a `double`, so the product is done in double and
    // the `as u32` reproduces the C's `(int)` truncation; `.max(1.0)` is the
    // `fmax`. `fmax` guarantees a modulus of at least 1, so the `%` below can't
    // divide by zero on a 1- or 2-pixel-wide preview.
    let hash_modulus = ((width as f64) * 0.3).max(1.0) as u32;
    let hash = (hash_seed % hash_modulus) as i32;

    let strength = strength / 100.0;
    let wd = width.min(height) as f64;
    // `grain.c:231`: `const double zoom = (1.0 + 8 * data->scale / 100) / 800.0;`
    // The inner `8 * data->scale / 100` is **float** arithmetic (`data->scale`
    // is `float`, so the `int` literals promote to `float`); only `1.0 + …` and
    // `… / 800.0` are `double` because of the `1.0`/`800.0` literals. Evaluating
    // the whole thing in f64 shifts the last ulp of `zoom`, and `zoom` scales
    // every sampled noise coordinate, so the inner term is computed in f32 to
    // match the C.
    let zoom_inner: f32 = 8.0 * coarseness / 100.0;
    let zoom: f64 = (1.0 + zoom_inner as f64) / 800.0;
    let filter = i32::from((scale - 1.0).abs() > 0.01);
    let iscale = scale; // see the `piece->iscale` note above
    let filtermul = iscale as f64 / (scale as f64 * wd);

    // SAFETY: `input`/`output` are caller-borrowed, distinct, packed RGBA
    // slices of `width * height * 4` floats (the same contract every other
    // stage in this crate takes), `lut` is the memoised 128×128 table
    // `grain_lut` documents, and `hash`/`filter`/`filtermul` are the values
    // derived above.
    unsafe {
        darkroom_grain_process(
            input.as_ptr(),
            output.as_mut_ptr(),
            0,
            0,
            width as i32,
            height as i32,
            strength,
            zoom,
            wd,
            scale as f64,
            hash,
            filter,
            filtermul,
            lut.as_ptr(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `_hash_string` (grain.c:201) is the djb2 seed 5381 with the fold run
    /// **backwards**. This pins both halves against values worked out from the C:
    /// the empty string must land on the untouched seed, and two filenames that
    /// differ only in their trailing characters must NOT collide — that
    /// backwards walk is the whole point of the function (see the C comment at
    /// grain.c:196-200), since the differing digits are folded first and so
    /// reach every one of the 32 bits instead of only the last few.
    /// The signed-`char` promotion the C relies on is pinned too: the fold is
    /// over `char`, which is signed on x86, so bytes ≥ 0x80 must sign-extend.
    #[test]
    fn hash_string_matches_c_fold() {
        let step = |h: u32, b: u32| (h << 5).wrapping_add(h) ^ b;
        assert_eq!(hash_string(""), 5381, "no bytes ⇒ the seed survives");
        // A one-character name has no fold-order ambiguity.
        assert_eq!(hash_string("A"), step(5381, b'A' as u32));
        // Backwards order: 'B' is folded first, then 'A' — not the other way
        // round, which is what makes trailing sequence digits dominate.
        assert_eq!(hash_string("AB"), step(step(5381, u32::from(b'B')), u32::from(b'A')));
        assert_ne!(hash_string("AB"), step(step(5381, u32::from(b'A')), u32::from(b'B')));
        // Two frames of a sequence must decorrelate (not merely "differ").
        assert_ne!(
            hash_string("/f/seq/img001.cr2"),
            hash_string("/f/seq/img002.cr2")
        );
        // A non-ASCII character goes through the C's *signed* `char` promotion
        // (grain.c:207 — `char *str`, and `char` is signed on x86): each byte
        // ≥ 0x80 sign-extends to `int`, so all 32 bits take part in the XOR.
        // Rust `"\u{e9}"` is two UTF-8 bytes, 0xC3 then 0xA9, and *both* are
        // ≥ 0x80 — which is convenient, because the naive "fold the raw byte"
        // version then differs in two places instead of one. Pinned value
        // 5862479 = 0x59_744F, worked out from the C by hand.
        let signed = |b: u8| b as i8 as i32 as u32;
        assert_eq!(
            hash_string("\u{e9}"),
            step(step(5381, signed(0xA9)), signed(0xC3))
        );
        assert_eq!(hash_string("\u{e9}"), 5_862_479, "value worked out from grain.c");
        assert_ne!(
            hash_string("\u{e9}"),
            step(step(5381, 0xA9), 0xC3),
            "the test must be able to tell the two foldings apart"
        );
    }

    /// The memo must hand back the *same* table for a repeated key (that is the
    /// whole point — one build per render, not one per band) and a *different*
    /// one when `midtones_bias` moves, or a bias edit would silently keep the
    /// old paper response.
    #[test]
    fn grain_lut_cache_is_keyed_on_midtones_bias() {
        use std::rc::Rc;
        let a1 = grain_lut(50.0);
        let a2 = grain_lut(50.0);
        assert!(Rc::ptr_eq(&a1, &a2), "same bias must reuse the table");
        let b = grain_lut(80.0);
        assert!(!Rc::ptr_eq(&a1, &b), "a changed bias must rebuild the table");
        assert_ne!(*a1, *b, "different bias must yield different paper response");
    }

    /// The driver must leave the buffer's alpha alone and only perturb L. With
    /// `strength = 0` the C still runs the kernel (noise × 0 = 0 samples the LUT
    /// at x = 0), so the result is near — not exactly — the input lightness.
    #[test]
    fn process_preserves_alpha_and_is_quiet_at_zero_strength() {
        let n = 8 * 8;
        let mut input = vec![0.0f32; n * 4];
        for p in 0..n {
            input[p * 4] = 50.0;
            input[p * 4 + 1] = 12.0;
            input[p * 4 + 2] = -7.0;
            input[p * 4 + 3] = 1.0;
        }
        let mut output = vec![0.0f32; n * 4];
        process(&input, &mut output, 8, 8, 1.0, 7.5047, 0.0, 100.0, 0);
        for p in 0..n {
            assert_eq!(output[p * 4 + 3], 1.0, "alpha must pass through");
            assert!(
                (output[p * 4] - 50.0).abs() < 2.0,
                "L moved {} at strength 0",
                output[p * 4] - 50.0
            );
        }
    }

    /// A different `hash_seed` must move the noise field. This is the property
    /// `_hash_string` exists for, and it is what a forgotten `hash_seed` (every
    /// caller hard-coding 0) would silently collapse. The two seeds are picked
    /// so they still differ *after* the C's reduction by the ROI width
    /// (`hash % 9` on this 32-px frame): 5381 → 8, 12345 → 6.
    #[test]
    fn hash_seed_changes_the_noise_field() {
        let (w, h) = (32usize, 32usize);
        let n = w * h;
        let input: Vec<f32> = (0..n)
            .flat_map(|_| [50.0f32, 0.0, 0.0, 1.0])
            .collect();
        let run = |seed: u32| {
            let mut out = vec![0.0f32; n * 4];
            process(&input, &mut out, w, h, 1.0, 7.5047, 60.0, 100.0, seed);
            out
        };
        assert_ne!(run(5381), run(12345), "the seed must decorrelate images");
    }

    #[test]
    fn simplex_noise_bounded() {
        let (perm, perm_mod) = build_perm();
        for &(x, y, z) in &[(0.1, 0.2, 1.0), (1.0, 2.0, 3.0), (-0.5, 0.7, 2.0)] {
            let n = simplex_noise(x, y, z, &perm, &perm_mod);
            assert!(n.abs() <= 1.0 + 1e-6, "noise={n} out of [-1,1]");
        }
    }

    #[test]
    fn grain_lut_midpoint() {
        // At the center x-index and mid-lightness, grain contribution should be near 0.
        let lut = build_grain_lut(50.0);
        let mid = GRAIN_LUT_SIZE / 2;
        let v = lut[mid * GRAIN_LUT_SIZE + mid];
        assert!(v.abs() < 5.0, "mid LUT value={v} too far from 0");
    }

    #[test]
    fn grain_channels_1_2_pass_through() {
        let lut = build_grain_lut(0.0);
        let inp = [50.0f32, 10.0, -5.0, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_grain_process(
                inp.as_ptr(), out.as_mut_ptr(),
                0, 0, 1, 1,
                0.5, 0.01, 1000.0, 1.0, 0,
                0, 0.0,
                lut.as_ptr(),
            )
        };
        assert_eq!(out[1], inp[1]);
        assert_eq!(out[2], inp[2]);
        assert_eq!(out[3], inp[3]);
    }

    #[test]
    fn zero_strength_is_passthrough() {
        let lut = build_grain_lut(0.0);
        let inp = [60.0f32, 5.0, -3.0, 1.0];
        let mut out = [0f32; 4];
        unsafe {
            darkroom_grain_process(
                inp.as_ptr(), out.as_mut_ptr(),
                0, 0, 1, 1,
                0.0, 0.01, 1000.0, 1.0, 0, // strength=0
                0, 0.0,
                lut.as_ptr(),
            )
        };
        // strength=0 → noise*strength=0 → lut_lookup(0, L/100)
        // lut at x=0 (center) is ~0, so out[0] ≈ in[0]
        assert!((out[0] - inp[0]).abs() < 2.0);
    }

    /// Golden vectors for the paper-response LUT, hand-derived from
    /// `evaluate_grain_lut` / `paper_resp` / `paper_resp_inverse`
    /// (`grain.c:150-161`) with an independent float32 model. These pin the
    /// *values*, not just "something non-zero": the previous midpoint check
    /// (`grain_lut_midpoint`) is satisfied by a sign flip, a swapped `gu`/`l`, or
    /// a wrong `delta`, all of which this table catches. Tolerance is loose
    /// enough for libm-vs-libm `expf`/`logf` ulp differences and tight enough
    /// that any structural error is orders of magnitude larger.
    #[test]
    fn grain_lut_matches_c_reference_vectors() {
        let at = |mb: f32, i: usize, j: usize| build_grain_lut(mb)[j * GRAIN_LUT_SIZE + i];
        // (mb, i, j, expected) — the corners show the near-linear density range,
        // the centre the flat mid-tone region, and the off-axis points break the
        // `gu`/`l` symmetry so a swap cannot pass.
        let cases: &[(f32, usize, usize, f32)] = &[
            (0.0, 64, 64, 0.393_700_6),
            (0.0, 0, 0, -45.571_148),
            (0.0, 127, 127, 45.571_136),
            (0.0, 32, 96, -24.713_194),
            (0.0, 96, 32, 25.508_654),
            (50.0, 64, 64, 0.393_647_0),
            (50.0, 0, 0, -1.702_806_9),
            (50.0, 127, 127, 1.702_809_3),
            (50.0, 32, 96, -22.376_978),
            (50.0, 96, 32, 23.402_288),
            (100.0, 64, 64, 0.393_641_0),
            (100.0, 0, 0, -0.017_290_7),
            (100.0, 127, 127, 0.017_285_3),
            (100.0, 32, 96, -22.142_719),
            (100.0, 96, 32, 23.193_008),
        ];
        for &(mb, i, j, want) in cases {
            let got = at(mb, i, j);
            assert!(
                (got - want).abs() < 1e-2,
                "LUT[{i},{j}] at mb={mb}: got {got}, want {want}"
            );
        }
        // The paper response is (near) antisymmetric about the mid density, so
        // the two extreme corners carry equal and opposite sign.
        let c00 = at(0.0, 0, 0);
        let c11 = at(0.0, 127, 127);
        assert!((c00 + c11).abs() < 1e-2, "corners not antisymmetric: {c00} / {c11}");
        // Mid-tones bias damps the response: at mb=100 the corner swing is a
        // few hundredths of the mb=0 swing, not merely smaller.
        assert!(
            at(100.0, 0, 0).abs() < 0.05 && at(0.0, 0, 0).abs() > 40.0,
            "midtones bias must collapse the density range"
        );
    }

    /// The rank-1 lattice branch (`filter != 0`, `grain.c:233`) is unreachable
    /// from the product — every c41 funnel passes `scale = 1.0` — so it is
    /// exercised directly here. Any `scale != 1.0` takes the branch
    /// (`|scale - 1| > 0.01`); the test asserts it runs, perturbs L, and leaves
    /// the other channels and alpha alone.
    #[test]
    fn process_filter_path_runs_and_preserves_alpha() {
        let (w, h) = (16usize, 16usize);
        let n = w * h;
        let input: Vec<f32> = (0..n).flat_map(|_| [50.0f32, 3.0, -2.0, 1.0]).collect();
        let mut out = vec![0.0f32; n * 4];
        // scale = 0.5 ⇒ filter = 1. `iscale == scale` here, so
        // filtermul = iscale/(scale·wd) = 1/wd.
        process(&input, &mut out, w, h, 0.5, 7.5047, 80.0, 100.0, 5381);
        let mut moved = false;
        for p in 0..n {
            assert_eq!(out[p * 4 + 1], 3.0);
            assert_eq!(out[p * 4 + 2], -2.0);
            assert_eq!(out[p * 4 + 3], 1.0, "alpha must pass through the filter path");
            if (out[p * 4] - 50.0).abs() > 1e-3 {
                moved = true;
            }
        }
        assert!(moved, "the filtered path produced a flat tint");
    }

    /// `process` is a safe `pub` fn; a short slice must fail loudly rather than
    /// become an out-of-bounds read/write inside the kernel's `from_raw_parts`.
    #[test]
    #[should_panic(expected = "input has")]
    fn process_rejects_a_short_input() {
        let mut out = vec![0.0f32; 4 * 4 * 4];
        let input = vec![0.0f32; 3 * 4]; // one row short of 4×4
        process(&input, &mut out, 4, 4, 1.0, 7.5047, 25.0, 100.0, 0);
    }

    /// A non-positive `scale` is a caller-contract violation: the kernel divides
    /// pixel indices by it.
    #[test]
    #[should_panic(expected = "divides pixel indices")]
    fn process_rejects_nonpositive_scale() {
        let n = 4 * 4;
        let input = vec![0.0f32; n * 4];
        let mut out = vec![0.0f32; n * 4];
        process(&input, &mut out, 4, 4, 0.0, 7.5047, 25.0, 100.0, 0);
    }
}
