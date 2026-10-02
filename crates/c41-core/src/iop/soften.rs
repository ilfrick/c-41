use crate::{imagebuf::linear_blend, params::IopParams, roi::RoiIn, Result};
use super::{box_filters::box_mean_4ch, ClBuffer, IopProcess};

pub struct Soften;

impl IopProcess for Soften {
    fn process(&self, _input: &[f32], _output: &mut [f32], _params: &IopParams, _roi: &RoiIn) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn process_cl(&self, _buf: &mut ClBuffer, _params: &IopParams) -> Result<()> {
        Err(crate::Error::Pipeline("not implemented".into()))
    }
    fn name(&self) -> &'static str { "soften" }
}

/// Converts linear RGB to HSL.  Ports darktable rgb2hsl() from colorspaces.h.
///
/// **Known parity deviation (pre-existing, tracked): float-only.** `rgb2hsl`
/// and `hsl2rgb` (colorspaces.h:294-362) write several intermediates with bare
/// **double** literals — `lv = (pmin + pmax) / 2.0`, `hv = 2.0 + …`, `hv /= 6.0`,
/// `m2 = l < 0.5 ? l * (1.0 + s) : …`, `m1 = 2.0 * l - m2` — so the C evaluates
/// them in double and rounds to float only at the store. This port is float
/// throughout. Measured worst case against the C across the soften golden
/// vectors is **4 ulp (5.96e-08)** on a whole 4×4 frame, so it is invisible at
/// any working precision; closing it means promoting five locals to `f64`, which
/// belongs in its own increment rather than smuggled into a wiring change.
/// Same treatment as the rest of the tree's colour math (see `crate::color`).
#[inline(always)]
fn rgb2hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    const EPS: f32 = 1.525_878_906_25e-5_f32;
    let pmax = r.max(g).max(b);
    let pmin = r.min(g).min(b);
    let delta = pmax - pmin;
    let l = (pmin + pmax) / 2.0;

    if delta == 0.0 {
        return (0.0, 0.0, l);
    }

    let s = if l < 0.5 {
        delta / (pmax + pmin).max(EPS)
    } else {
        delta / (2.0 - pmax - pmin).max(EPS)
    };

    let mut h = if pmax == r {
        (g - b) / delta
    } else if pmax == g {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };
    h /= 6.0;
    if h < 0.0 { h += 1.0; } else if h > 1.0 { h -= 1.0; }

    (h, s, l)
}

/// Converts one HSL channel to RGB.  Ports darktable hue2rgb() — hue is pre-scaled to [0, 6).
#[inline(always)]
fn hue2rgb(m1: f32, m2: f32, hue: f32) -> f32 {
    if hue < 1.0 { m1 + (m2 - m1) * hue }
    else if hue < 3.0 { m2 }
    else if hue < 4.0 { m1 + (m2 - m1) * (4.0 - hue) }
    else { m1 }
}

/// Converts HSL back to RGB.  Ports darktable hsl2rgb() from colorspaces.h.
///
/// Like [`rgb2hsl`], this is a float-only port of double-evaluating C code —
/// see the deviation note there (it applies to `m1`/`m2` here).
#[inline(always)]
fn hsl2rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    let m2 = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let m1 = 2.0 * l - m2;
    let h6 = h * 6.0;
    let r = hue2rgb(m1, m2, if h6 < 4.0 { h6 + 2.0 } else { h6 - 4.0 });
    let g = hue2rgb(m1, m2, h6);
    let b = hue2rgb(m1, m2, if h6 > 2.0 { h6 - 2.0 } else { h6 + 4.0 });
    (r, g, b)
}

/// Soften IOP initial pixel loop.
///
/// Converts each pixel to HSL, scales saturation and lightness, writes back RGB.
/// Matches the DT_OMP_FOR loop in src/iop/soften.c::process() (before the box-mean blur).
///
/// `brightness` = 1.0 / exp2f(-d->brightness)
/// `saturation` = d->saturation / 100.0
///
/// Safe core shared by the FFI entry point and the Rust driver. `input` and
/// `output` must each hold `npixels * 4` floats.
pub fn overexpose(input: &[f32], output: &mut [f32], npixels: usize, brightness: f32, saturation: f32) {
    debug_assert!(input.len() >= npixels * 4 && output.len() >= npixels * 4);
    for k in (0..npixels * 4).step_by(4) {
        let (h, s, l) = rgb2hsl(input[k], input[k + 1], input[k + 2]);
        let s = (s * saturation).clamp(0.0, 1.0);
        let l = (l * brightness).clamp(0.0, 1.0);
        let (r, g, b) = hsl2rgb(h, s, l);
        output[k]     = r;
        output[k + 1] = g;
        output[k + 2] = b;
        output[k + 3] = 0.0; // hsl2rgb sets alpha=0 in C
    }
}

/// # Safety
/// `in_buf` and `out_buf` must each be valid for `npixels * 4` f32 reads/writes.
#[no_mangle]
pub unsafe extern "C" fn darkroom_soften_process(
    in_buf: *const f32,
    out_buf: *mut f32,
    npixels: usize,
    brightness: f32,
    saturation: f32,
) {
    let input  = std::slice::from_raw_parts(in_buf,  npixels * 4);
    let output = std::slice::from_raw_parts_mut(out_buf, npixels * 4);
    overexpose(input, output, npixels, brightness, saturation);
}

/// Apply the Soften (Orton effect) IOP to a packed RGBA `f32` buffer.
///
/// The C chain (`soften.c::process`) is three steps:
/// 1. `darkroom_soften_process` — overexpose: HSL, scale saturation and
///    lightness, write back (alpha becomes 0, because the C's `hsl2rgb` sets
///    `rgb[3] = 0.0f` — colorspaces.h:352/361).
/// 2. `dt_box_mean(out, h, w, 4, radius, BOX_ITERATIONS)` — blur all four
///    lanes ([`box_mean_4ch`]); `BOX_ITERATIONS` is 8 (box_filters.h:25).
/// 3. `dt_iop_image_linear_blend(out, amt, in, …)` — `out = amt*blurred +
///    (1-amt)*original` ([`linear_blend`]).
///
/// `size`/`saturation`/`amount` are the raw 0..100 sliders, `brightness` the
/// −2..2 slider; the divisions and `1/exp2(-b)` happen here as the C does them.
///
/// **No `scale` parameter**, unlike `grain::process` or `Stage::Lowpass`: the
/// C's radius is `MIN(mrad, ceil(rad * roi_in->scale / piece->iscale))`, and in
/// c41 that ratio is *identically* 1 for every caller — a funnel hands the
/// pipeline one whole frame at the render resolution, so the ROI covers the
/// whole buffer and `roi_in->scale == piece->iscale`. A parameter that can only
/// ever be 1.0 would be a lie in the signature, so it is folded into
/// `radius_for` and documented there instead. That is also why there is **no
/// radius deviation to declare**: the C's `iwidth * iscale` *is* the buffer
/// width, so `radius_for`'s `hypot(width, height)` is the same quantity, not a
/// substitute for it.
///
/// Not pixel-local (the box mean reads a spatial neighbourhood), so the
/// pipeline routes it down the serial whole-frame path.
#[allow(clippy::too_many_arguments)]
pub fn process(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    size: f32,
    saturation: f32,
    brightness: f32,
    amount: f32,
) {
    // Safe `pub` fn: a short slice would become an out-of-bounds access, so the
    // length contract is a release assert (as `grain::process` and
    // `Pipeline::process` do).
    assert!(
        input.len() >= width * height * 4 && output.len() >= width * height * 4,
        "soften: need {} floats per buffer for a {width}×{height} RGBA frame",
        width * height * 4
    );

    let npixels = width * height;
    let brightness = 1.0 / (-brightness).exp2();
    let saturation = saturation / 100.0;

    let radius = radius_for(width, height, size);
    overexpose(input, output, npixels, brightness, saturation);
    // soften.c:137 — `dt_box_mean(out, h, w, 4, radius, BOX_ITERATIONS)`; the
    // constant is 8 (box_filters.h:25).
    box_mean_4ch(output, height, width, radius, 8);
    let amt = amount / 100.0;
    linear_blend(output, input, npixels * 4, amt);
}

/// The C's radius derivation (`soften.c:131-135`), split out so the truncations
/// can be pinned directly (the `as usize` casts are the C's `(int)`):
///
/// ```text
/// w = iwidth * iscale;  h = iheight * iscale
/// mrad = dt_fast_hypotf(w, h) * 0.01f
/// rad  = mrad * (fmin(100.0, size + 1.0f) / 100.0)
/// radius = MIN(mrad, ceilf(rad * roi_in->scale / iscale))
/// ```
///
/// Two pieces of the C's spelling matter and are reproduced exactly:
///
/// * `dt_fast_hypotf(w, h)` is `sqrtf(w*w + h*h)` under darktable's
///   fast-math build, and `iwidth * iscale` **is** the working-buffer width —
///   so `(width, height)` is the same quantity the C names `w, h`, not a
///   stand-in for it. No deviation.
/// * `d->size + 1.0f` is a float add, but C then promotes it to `double` for
///   `fmin`/`/100.0` and for the `mrad * …` product, so the window factor is
///   evaluated in f64 here. Doing it in f32 diverges from the C at sizes where
///   the truncation boundary bites (`radius_for(100, 100, 99.0)` is the case
///   pinned in `matches_the_goldens_without_the_blur` / `…_with_it`).
///
/// The `roi_in->scale / iscale` factor is dropped because it cannot be anything
/// but 1.0 here: c41 runs each stage over one whole frame, so the ROI *is* the
/// buffer and the two scales are equal. With the ratio 1, `ceilf(rad)` is the
/// identity on the already-truncated integer `rad`.
///
/// The `MIN(mrad, …)` is kept to mirror `soften.c:135` but is **provably
/// non-binding**: the window factor is at most 1, so `rad ≤ mrad` for every
/// input.
pub(crate) fn radius_for(width: usize, height: usize, size: f32) -> usize {
    let mrad = (((width * width + height * height) as f32).sqrt() * 0.01) as usize;
    // float add → widen to f64 → fmin → divide, as the C's promotions do.
    let factor = ((size + 1.0) as f64).min(100.0) / 100.0;
    let rad = (mrad as f64 * factor) as usize;
    mrad.min(rad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pixels: &[f32], brightness: f32, saturation: f32) -> Vec<f32> {
        let n = pixels.len() / 4;
        let mut out = vec![0f32; pixels.len()];
        unsafe { darkroom_soften_process(pixels.as_ptr(), out.as_mut_ptr(), n, brightness, saturation); }
        out
    }

    #[test]
    fn grey_pixel_stays_grey() {
        let input = vec![0.5, 0.5, 0.5, 1.0];
        let out = run(&input, 1.0, 1.0);
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
        assert!((out[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn zero_saturation_produces_grey() {
        let input = vec![0.8, 0.3, 0.1, 1.0];
        let out = run(&input, 1.0, 0.0);
        // s=0 → all channels equal to L
        assert!((out[0] - out[1]).abs() < 1e-5);
        assert!((out[1] - out[2]).abs() < 1e-5);
    }

    #[test]
    fn alpha_is_zero() {
        let input = vec![0.6, 0.4, 0.2, 0.9];
        let out = run(&input, 1.0, 1.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn roundtrip_hsl() {
        let r = 0.7_f32;
        let g = 0.3_f32;
        let b = 0.5_f32;
        let (h, s, l) = rgb2hsl(r, g, b);
        let (rr, gg, bb) = hsl2rgb(h, s, l);
        assert!((r - rr).abs() < 1e-5, "R roundtrip");
        assert!((g - gg).abs() < 1e-5, "G roundtrip");
        assert!((b - bb).abs() < 1e-5, "B roundtrip");
    }

    /// `amount = 0` makes the final blend `0*blurred + 1*original`, so the whole
    /// chain must return the input bit-for-bit — this pins that the overexpose
    /// pass and the blur are both overwritten by the mix, i.e. the driver's step
    /// order is the C's.
    #[test]
    fn amount_zero_is_an_exact_passthrough() {
        let (w, h) = (12usize, 9usize);
        let input: Vec<f32> = (0..w * h * 4)
            .map(|k| ((k * 37 % 100) as f32) / 100.0)
            .collect();
        let mut out = vec![0.0f32; input.len()];
        process(&input, &mut out, w, h, 50.0, 100.0, 0.33, 0.0);
        assert_eq!(out, input, "amount 0 must reproduce the input exactly");
    }

    /// `amount = 100` leaves the blurred buffer, which must be a genuine blur:
    /// the variance of the L-ish first channel drops on a high-frequency frame.
    #[test]
    fn amount_full_blurs_a_noisy_frame() {
        // 256×256 so the C radius derivation yields a nonzero radius
        // (`mrad = int(sqrt(2)·256·0.01) = 3`); a thumbnail would give 0.
        let (w, h) = (256usize, 256usize);
        let input: Vec<f32> = (0..w * h)
            .flat_map(|k| {
                let v = if (k + k / w) % 2 == 0 { 0.9f32 } else { 0.1 };
                [v, v, v, 1.0]
            })
            .collect();
        let mut out = vec![0.0f32; input.len()];
        process(&input, &mut out, w, h, 100.0, 100.0, 0.0, 100.0);
        let var = |buf: &[f32]| {
            let m: f32 = buf.iter().step_by(4).sum::<f32>() / (w * h) as f32;
            buf.iter().step_by(4).map(|v| (v - m) * (v - m)).sum::<f32>() / (w * h) as f32
        };
        assert!(
            var(&out) < var(&input) * 0.5,
            "full-amount soften must reduce variance: {} vs {}",
            var(&out),
            var(&input)
        );
        // Alpha was zeroed by the overexpose pass and blurred over a zero field,
        // so at amount=100 (pure blurred buffer) it stays zero — as in the C.
        for k in (3..w * h * 4).step_by(4) {
            assert_eq!(out[k], 0.0);
        }
    }

    /// Pin the C's radius derivation, including both `(int)` truncations and the
    /// `MIN(mrad, …)` clamp. A 1000×1000 frame has diagonal
    /// `sqrt(2)·1000 = 1414.2`, so `mrad = 14`.
    #[test]
    fn radius_matches_the_c_derivation() {
        // size 50 → window factor 51/100 = 0.51 → rad = 14*0.51 = 7.14 → 7.
        assert_eq!(radius_for(1000, 1000, 50.0), 7);
        // size 100 → fmin(100, 101) = 100 → factor 1.0 → rad = 14 → 14.
        assert_eq!(radius_for(1000, 1000, 100.0), 14);
        // size 0 → factor 0.01 → rad = 0.14 → 0.
        assert_eq!(radius_for(1000, 1000, 0.0), 0);
        // Tiny frames keep mrad at 0 rather than underflowing.
        assert_eq!(radius_for(1, 1, 100.0), 0);
        // The window factor is clamped at `size + 1`, so a size past 100 gives
        // the same radius as 100 rather than a larger one.
        assert_eq!(radius_for(1000, 1000, 250.0), radius_for(1000, 1000, 100.0));
        // The radius scales with the frame: 2× the linear size ⇒ ~2× the radius.
        assert_eq!(radius_for(2000, 2000, 50.0), 14);
    }

    /// Absolute budget for the golden comparisons below.
    ///
    /// The reference is the C transcribed from `soften.c::process`,
    /// `box_filters.cc`, `imagebuf.c` and `colorspaces.h`, compiled *without*
    /// box_filters' fast-math pragma so it yields the exact f32 semantics of
    /// the C as written. Against this port it is **bit-identical for 60 736 of
    /// 60 736 floats** on sets B and C; set A's worst case is 5.96e-08 (4 ulp).
    /// That residue is the float-only `rgb2hsl`/`hsl2rgb` deviation documented
    /// on those functions (the C evaluates `lv`/`hv`/`m2`/`m1` in double), so it
    /// is a known, measured quantity rather than slack. `1e-7` is ~1.7x that
    /// worst case and still six orders of magnitude below the errors these tests
    /// exist to catch — a reversed `linear_blend` moves pixels by O(1e-1).
    const GOLDEN_TOL: f32 = 1.0e-7;

    /// Set A's frame: 16 pixels of exact 1/64 multiples, chosen so every value
    /// is bit-exact in f32 and readable by eye in a diff.
    fn field_a() -> [f32; 64] {
        let mut v = [0.0f32; 64];
        for p in 0..16 {
            v[p * 4] = (2 + p) as f32 / 64.0;
            v[p * 4 + 1] = (50 - 2 * p) as f32 / 64.0;
            v[p * 4 + 2] = (26 + 2 * p) as f32 / 64.0;
            v[p * 4 + 3] = 1.0;
        }
        v
    }

    /// A 1/16-step saw pattern on all three channels. High-frequency enough that
    /// a radius-1 blur moves essentially every pixel (so a blur that silently
    /// did nothing cannot pass), and every value is `k/16` with `k ≤ 15`, hence
    /// exactly representable.
    fn field_b(w: usize, h: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                v.push(((x + 3 * y) % 16) as f32 / 16.0);
                v.push(((2 * x + y) % 16) as f32 / 16.0);
                v.push(((5 * x + 11 * y) % 16) as f32 / 16.0);
                v.push(1.0);
            }
        }
        v
    }

    /// Golden vector for the whole chain with the blur a no-op.
    ///
    /// 4×4 gives `mrad = int(hypot(4,4)·0.01) = int(0.0566) = 0`, so
    /// `radius = 0` and the box mean is the identity — which isolates exactly
    /// what a `radius == 0` blur must do to the constants. What this pins:
    /// `brightness = 1/exp2f(-0.33)`, `saturation = 100/100`, the overexpose
    /// HSL round-trip, the step order (overexpose *then* blend), and above all
    /// the blend direction — `out = amt·blurred + (1-amt)·original`, so the
    /// golden alpha is `0.5·0 + 0.5·1 = 0.5`, not `0.0`. Reversing that
    /// argument order is the single easiest mistake in this driver and it
    /// changes every float here.
    #[test]
    fn matches_the_goldens_without_the_blur() {
        let input = field_a();
        let mut out = [0.0f32; 64];
        process(&input, &mut out, 4, 4, 0.0, 100.0, 0.33, 50.0);
        assert_eq!(radius_for(4, 4, 0.0), 0, "the C derives radius 0 here");

        const SET_A_GOLDEN: [f32; 64] = [
            0.0451073944568634, 0.8718043565750122, 0.4584558606147766, 0.5,
            0.05364114046096802, 0.8456376791000366, 0.4936392307281494, 0.5,
            0.07053166627883911, 0.8111141920089722, 0.5289875268936157, 0.5,
            0.08816459774971008, 0.775848388671875, 0.5642533302307129, 0.5,
            0.10579749941825867, 0.7405825853347778, 0.5995191335678101, 0.5,
            0.12343043088912964, 0.7053166627883911, 0.6347850561141968, 0.5,
            0.14106333255767822, 0.670050859451294, 0.670050859451294, 0.5,
            0.1586962640285492, 0.6347850561141968, 0.7053166627883911, 0.5,
            0.18289020657539368, 0.5962386727333069, 0.7340215444564819, 0.5,
            0.21803587675094604, 0.5576877593994141, 0.7517746090888977, 0.5,
            0.2523857653141022, 0.5265880823135376, 0.7703233957290649, 0.5,
            0.2860572934150696, 0.5018401741981506, 0.7895506620407104, 0.5,
            0.31914591789245605, 0.4825509190559387, 0.8093608021736145, 0.5,
            0.3424404263496399, 0.4632166922092438, 0.8389650583267212, 0.5,
            0.36632075905799866, 0.44553065299987793, 0.8679834604263306, 0.5,
            0.39088255167007446, 0.4297623932361603, 0.8963204026222229, 0.5,
        ];
        for (i, (got, want)) in out.iter().zip(SET_A_GOLDEN).enumerate() {
            assert!(
                (got - want).abs() <= GOLDEN_TOL,
                "float {i} (px {}, ch {}): got {got:.9e}, C golden {want:.9e}",
                i / 4,
                i % 4
            );
        }
    }

    /// Golden vector for the blur composition: 72×72, `size = 100` ⇒
    /// `mrad = int(hypot(72,72)·0.01) = int(1.0182) = 1`, factor `1.0`,
    /// `radius = 1`, `BOX_ITERATIONS = 8`, `amount = 100` so the frame *is* the
    /// blurred buffer. Samples straddle the column boundary (idx 36/37, 2555/
    /// 2556) and the row boundary (idx 2591/2592, 2555/2628) where a separable
    /// pass that covered the wrong set of interleaved columns would show up.
    ///
    /// Every sampled alpha is `0.0`: overexpose writes alpha 0 and the blur
    /// averages a zero field, so at full amount alpha stays 0 exactly as in
    /// the C — a strong, cheap check that the blur really ran.
    #[test]
    fn matches_the_goldens_with_the_blur() {
        let input = field_b(72, 72);
        let mut out = vec![0.0f32; input.len()];
        process(&input, &mut out, 72, 72, 100.0, 100.0, 0.0, 100.0);
        assert_eq!(radius_for(72, 72, 100.0), 1, "the C derives radius 1 here");

        const SET_B_IDX: [usize; 16] = [
            0, 1, 36, 37, 71, 2520, 2591, 2592, 2663, 2555, 2556, 2628, 5041, 5112,
            5182, 5183,
        ];
        const SET_B_GOLDEN: [f32; 64] = [
            0.3756581246852875, 0.31099334359169006, 0.392112672328949, 0.0,
            0.384995698928833, 0.3349502384662628, 0.40371131896972656, 0.0,
            0.4669652581214905, 0.5033556222915649, 0.49580734968185425, 0.0,
            0.49263423681259155, 0.5160048007965088, 0.518545925617218, 0.0,
            0.5109946727752686, 0.6277086734771729, 0.5528004169464111, 0.0,
            0.4682875871658325, 0.4015507698059082, 0.4067800045013428, 0.0,
            0.4709506034851074, 0.5287347435951233, 0.49580785632133484, 0.0,
            0.4691035747528076, 0.4443124532699585, 0.40571796894073486, 0.0,
            0.46839630603790283, 0.4809335172176361, 0.4664696455001831, 0.0,
            0.4717283546924591, 0.48938870429992676, 0.4396398067474365, 0.0,
            0.4712749719619751, 0.5051872730255127, 0.4584030210971832, 0.0,
            0.4681575298309326, 0.5051810145378113, 0.4396395683288574, 0.0,
            0.40426480770111084, 0.5135992169380188, 0.40426528453826904, 0.0,
            0.3904600739479065, 0.5165159702301025, 0.40762433409690857, 0.0,
            0.439700722694397, 0.4167081117630005, 0.3917854428291321, 0.0,
            0.44448134303092957, 0.3946138620376587, 0.3921131491661072, 0.0,
        ];
        for (n, &px) in SET_B_IDX.iter().enumerate() {
            for ch in 0..4 {
                let got = out[px * 4 + ch];
                let want = SET_B_GOLDEN[n * 4 + ch];
                assert!(
                    (got - want).abs() <= GOLDEN_TOL,
                    "px {px} ({}, {}) ch {ch}: got {got:.9e}, C golden {want:.9e}",
                    px % 72,
                    px / 72
                );
            }
        }
    }

    /// Golden vector that pins the `+ 1` inside `fmin(100.0, size + 1.0)`.
    ///
    /// `size = 99` sits exactly on the C's clamp: `min(100, 99 + 1) = 100`
    /// ⇒ factor `1.0` ⇒ `rad = 1` ⇒ `radius = 1`, so the blur *runs*. Drop the
    /// `+ 1` and the factor becomes `0.99`, `rad = int(1·0.99) = 0`, `radius = 0`,
    /// and every sampled pixel below becomes the unblurred input — which differs
    /// in the first decimal. 100×100 gives `mrad = int(hypot(100,100)·0.01) =
    /// int(1.4142) = 1`, comfortably clear of the truncation boundary at 1.4142.
    #[test]
    fn matches_the_goldens_at_the_window_factor_clamp() {
        let input = field_b(100, 100);
        let mut out = vec![0.0f32; input.len()];
        process(&input, &mut out, 100, 100, 99.0, 100.0, 0.0, 100.0);
        // The whole point: radius must be 1, not the 0 that dropping `+ 1` gives.
        assert_eq!(radius_for(100, 100, 99.0), 1, "the `+ 1` must survive");
        assert_eq!(radius_for(100, 100, 98.0), 0, "…and it is the only `+ 1`");

        const SET_C_IDX: [usize; 14] = [
            0, 1, 49, 50, 51, 99, 4949, 4950, 4951, 5049, 5050, 9900, 9998, 9999,
        ];
        const SET_C_GOLDEN: [f32; 56] = [
            0.3756581246852875, 0.31099334359169006, 0.392112672328949, 0.0,
            0.384995698928833, 0.3349502384662628, 0.40371131896972656, 0.0,
            0.40268564224243164, 0.41783666610717773, 0.4147842526435852, 0.0,
            0.41738730669021606, 0.43396973609924316, 0.43715766072273254, 0.0,
            0.44032275676727295, 0.4693916440010071, 0.46646925806999207, 0.0,
            0.38924503326416016, 0.34520041942596436, 0.3953216075897217, 0.0,
            0.4657716751098633, 0.4323188066482544, 0.43963953852653503, 0.0,
            0.4662250280380249, 0.4378708600997925, 0.4584031403064728, 0.0,
            0.4670628309249878, 0.46150872111320496, 0.4790970981121063, 0.0,
            0.4681575298309326, 0.43231260776519775, 0.4261600971221924, 0.0,
            0.4693424701690674, 0.44811105728149414, 0.4396396577358246, 0.0,
            0.446878045797348, 0.3193461298942566, 0.4076119661331177, 0.0,
            0.43970024585723877, 0.3517919182777405, 0.3917846083641052, 0.0,
            0.4444808065891266, 0.342101126909256, 0.3921123147010803, 0.0,
        ];
        for (n, &px) in SET_C_IDX.iter().enumerate() {
            for ch in 0..4 {
                let got = out[px * 4 + ch];
                let want = SET_C_GOLDEN[n * 4 + ch];
                assert!(
                    (got - want).abs() <= GOLDEN_TOL,
                    "px {px} ({}, {}) ch {ch}: got {got:.9e}, C golden {want:.9e}",
                    px % 100,
                    px / 100
                );
            }
        }
    }

    /// A short buffer must panic rather than reach the kernel's slice indexing.
    #[test]
    #[should_panic(expected = "need")]
    fn process_rejects_a_short_input() {
        let mut out = vec![0.0f32; 4 * 4 * 4];
        let input = vec![0.0f32; 3 * 4];
        process(&input, &mut out, 4, 4, 50.0, 100.0, 0.0, 50.0);
    }
}
