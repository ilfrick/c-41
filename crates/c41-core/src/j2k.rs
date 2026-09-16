//! Kernels ported from `src/imageio/imageio_j2k.c` (`dt_imageio_open_j2k`,
//! m4-211): the grayscale (1-2 component) and RGB (3-4 component)
//! int-to-float normalize loops. Each kernel replaces its whole loop body:
//! every OpenJPEG component lane is shifted by its signed offset and scaled
//! by its divisor into the R/G/B lanes of the 4-channel float mipmap
//! buffer. The alpha lane is never written (as in C).
//!
//! What the C loops do, per pixel `index`:
//! - grey (`numcomps < 3`): `v = (float)(comps[0].data[index] +
//!   signed_offsets[0]) / float_divs[0]`; `buf[4*index + c] = v` for `c` in
//!   `0..2` (chained assignment, so one value lands in all three lanes).
//! - rgb (`numcomps >= 3`): `buf[4*index + c] =
//!   (float)(comps[c].data[index] + signed_offsets[c]) / float_divs[c]`
//!   for `c` in `0..2`, where `signed_offsets[c]` is `1 << (prec - 1)` for
//!   signed components (else 0) and `float_divs[c]` is `(1 << prec) - 1`,
//!   with `prec <= 16` enforced by the C sanity checks above the loops.
//!
//! Bit-exactness notes:
//! - The C expression converts the (`int` lane + `long` offset) sum to
//!   `float` first and then divides by the `int` divisor (promoted to
//!   `float`), so each kernel computes `((lane as i64 + offset) as f32) /
//!   (div as f32)` — the same operation order, bit for bit. A hoisted
//!   reciprocal multiply is NOT an acceptable substitute here (unlike the
//!   AVIF/HEIF kernels, whose C loops multiply by a parenthesised
//!   reciprocal); this kernel divides exactly as the C loop does.
//! - Realistic sums are tiny (lanes fit `prec <= 16` bits plus an offset
//!   below `2^15`), so the `i64` addition cannot overflow and the `as f32`
//!   cast is exact; the single division then rounds exactly as the C one.
//! - A zero divisor yields IEEE `inf`/`NaN` from the Rust `/` exactly as
//!   the C `/` does (neither language traps on float division by zero);
//!   real callers always pass `(1 << prec) - 1 >= 1`.
//! - The alpha lane (`buf[4*index + 3]`) keeps whatever the mipmap-cache
//!   allocation left there — the kernels never write it. This differs
//!   from the AVIF/HEIF/WebP read kernels, which zero it; the tests pin
//!   the preservation with a sentinel.
//! - Buffers: the C caller passes the mipmap-cache allocation (`buf`) and
//!   the OpenJPEG component planes (`comps[c].data`, `int` lanes). The
//!   kernel must not be called with overlapping buffers.
//!
//! The Rust kernels are single-threaded sequential; the C loops were
//! `DT_OMP_FOR` over the pixel index, but each output triple reads only
//! its own source lanes, so thread scheduling cannot change the result.
//!
//! sYCC 4:4:4 conversion (m4-212): `j2k_sycc444_to_rgb` replaces the loop
//! body of `sycc444_to_rgb()` in the same C file. Per pixel it runs the
//! scalar `sycc_to_rgb` arithmetic: subtract `offset` from the Cb/Cr
//! lanes, add the truncated products `1.402*cr`, `-(0.344*cb + 0.714*cr)`
//! and `1.772*cb` to Y, and clamp each lane to `[0, upb]` (glib `CLAMP`
//! order). The C constants are unsuffixed doubles, so each product runs
//! in f64 over the exact `(float)lane` promotion — the kernel spells the
//! same `as f32 as f64` chain rather than computing in f32. The 4:2:2 and
//! 4:2:0 subsampled variants stay in C (shared-subsample indexing, and
//! the 4:2:0 tail passes the base `*cr` where the pattern calls for the
//! local `curr_cr`); the calloc allocation, the NULL-alloc early return,
//! and the plane hand-off stay in C with the converted call.
//!
//! 12-bit export pack (m4-217): `j2k_float_to_12bit` replaces the loop
//! body of the `case 12` branch of `write_image()` in
//! `src/imageio/format/j2k.c` (the only live `DT_OMP_FOR` site in that
//! file; the 8/16-bit cases are commented out and `prec` is hardcoded to
//! 12). Per pixel it runs the `DOWNSAMPLE_FLOAT_TO_12BIT` macro: branch
//! on the raw lane (`<= 0.0` yields 0, `>= 1.0` yields 4095), else
//! `(int)roundf(4095.0 * lane)` in f32. `roundf` is round-half-away, so
//! the kernel uses `f32::round`; the `(int)` cast truncates, which is
//! exact on the integral `round` output (range-bounded to `[0, 4095]`).
//! The branch is spelled `<=`/`>=` exactly as the macro, so `-0.0` takes
//! the zero branch in both languages. A NaN lane falls to the scale
//! branch in both; there the languages diverge: Rust `as i32` saturates
//! NaN to 0 (defined), while the C `(int)` cast on x86 yields INT_MIN
//! via cvttss2si. The kernel keeps the defined saturating behaviour and
//! the divergence is pinned by a test; real export buffers hold
//! pipeline floats, so both spellings only ever see it on degenerate
//! input. The alpha lane (`src[4*i + 3]`) is never read, as in C.

/// Scale one signed OpenJPEG component lane into the R/G/B lanes.
///
/// Port of the former element-wise loop in `dt_imageio_open_j2k`
/// (`src/imageio/imageio_j2k.c`, the `numcomps < 3` grayscale branch):
/// `buf[4*i + c] = ((comp0[i] as i64 + offset0) as f32) / (div0 as f32)`
/// for `c` in `0..2`; lane 3 is never touched. `npixels` is the C loop
/// bound; short buffers iterate clamped (no panic, no out-of-bounds
/// access). For the well-formed buffers the C caller passes the clamps
/// never engage and the behaviour is exactly the C loop's. A zero
/// `npixels` is a no-op.
pub fn j2k_grey_to_float(
    buf: &mut [f32],
    comp0: &[i32],
    npixels: usize,
    offset0: i64,
    div0: i32,
) {
    let n = npixels.min(buf.len() / 4).min(comp0.len());
    let div = div0 as f32;
    for (i, &lane) in comp0.iter().enumerate().take(n) {
        // wrapping_add: lane+offset cannot overflow for real callers
        // (offsets <= 2^15), but a debug panic across FFI would be UB —
        // wrap instead of trapping on adversarial inputs.
        let v = (lane as i64).wrapping_add(offset0) as f32 / div;
        let o = 4 * i;
        buf[o] = v;
        buf[o + 1] = v;
        buf[o + 2] = v;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `j2k_grey_to_float`: walks the
/// output in 4-lane chunks zipped with the source lane iterator (the
/// kernel indexes both sides with `4*i`/`i`), so the sweep test
/// cross-checks indexing as well as values. The arithmetic is the same
/// convert-then-divide spelling by construction (see the module docs: a
/// reciprocal multiply must not be used), and lane 3 is never assigned.
/// Same well-formed-buffers precondition, enforced here by the same
/// clamping.
#[cfg(test)]
fn ref_j2k_grey_to_float(
    buf: &mut [f32],
    comp0: &[i32],
    npixels: usize,
    offset0: i64,
    div0: i32,
) {
    let n = npixels.min(buf.len() / 4).min(comp0.len());
    let div = div0 as f32;
    for (quad, &lane) in buf.chunks_exact_mut(4).zip(comp0.iter()).take(n) {
        let v = i64::from(lane).wrapping_add(offset0) as f32 / div;
        quad[0] = v;
        quad[1] = v;
        quad[2] = v;
    }
}

/// Scale three OpenJPEG component lanes into the R/G/B lanes.
///
/// Port of the former element-wise loop in `dt_imageio_open_j2k`
/// (`src/imageio/imageio_j2k.c`, the `numcomps >= 3` RGB branch):
/// `buf[4*i + c] = ((comp[c][i] as i64 + offsets[c]) as f32) /
/// (divs[c] as f32)` for `c` in `0..2`; lane 3 is never touched.
/// `offsets`/`divs` each need 3 entries (fewer is a no-op); `npixels` is
/// the C loop bound; short buffers iterate clamped (no panic, no
/// out-of-bounds access). For the well-formed buffers the C caller passes
/// the clamps never engage and the behaviour is exactly the C loop's. A
/// zero `npixels` is a no-op.
pub fn j2k_rgb_to_float(
    buf: &mut [f32],
    comp0: &[i32],
    comp1: &[i32],
    comp2: &[i32],
    npixels: usize,
    offsets: &[i64],
    divs: &[i32],
) {
    let [off0, off1, off2, ..] = offsets else {
        return;
    };
    let [div0, div1, div2, ..] = divs else {
        return;
    };
    let n = npixels
        .min(buf.len() / 4)
        .min(comp0.len())
        .min(comp1.len())
        .min(comp2.len());
    let fdiv0 = *div0 as f32;
    let fdiv1 = *div1 as f32;
    let fdiv2 = *div2 as f32;
    for (i, &a) in comp0.iter().enumerate().take(n) {
        let o = 4 * i;
        buf[o] = (a as i64).wrapping_add(*off0) as f32 / fdiv0;
        buf[o + 1] = (comp1[i] as i64).wrapping_add(*off1) as f32 / fdiv1;
        buf[o + 2] = (comp2[i] as i64).wrapping_add(*off2) as f32 / fdiv2;
    }
}

// ── Independent reference implementation for bit-exactness tests ─────────────

/// Structurally divergent reference for `j2k_rgb_to_float`: walks the
/// output in 4-lane chunks zipped with the three source lane iterators
/// (the kernel iterates comp0 and indexes the other three sides), so the
/// sweep test cross-checks indexing as well as values. The arithmetic is the same
/// convert-then-divide spelling by construction, and lane 3 is never
/// assigned. Same preconditions, enforced here by early return (short
/// offset/div tables) and the same clamping.
#[cfg(test)]
fn ref_j2k_rgb_to_float(
    buf: &mut [f32],
    comp0: &[i32],
    comp1: &[i32],
    comp2: &[i32],
    npixels: usize,
    offsets: &[i64],
    divs: &[i32],
) {
    if offsets.len() < 3 || divs.len() < 3 {
        return;
    }
    let n = npixels
        .min(buf.len() / 4)
        .min(comp0.len())
        .min(comp1.len())
        .min(comp2.len());
    let fdiv = [divs[0] as f32, divs[1] as f32, divs[2] as f32];
    let offs = [offsets[0], offsets[1], offsets[2]];
    for (((quad, &a), &b), &c) in buf
        .chunks_exact_mut(4)
        .zip(comp0.iter())
        .zip(comp1.iter())
        .zip(comp2.iter())
        .take(n)
    {
        quad[0] = i64::from(a).wrapping_add(offs[0]) as f32 / fdiv[0];
        quad[1] = i64::from(b).wrapping_add(offs[1]) as f32 / fdiv[1];
        quad[2] = i64::from(c).wrapping_add(offs[2]) as f32 / fdiv[2];
    }
}

// ── FFI exports ──────────────────────────────────────────────────────────────

/// # Safety
/// `buf` must hold at least `4 * npixels` floats (the C caller passes the
/// mipmap-cache allocation) and `comp0` at least `npixels` `int` lanes
/// (the C caller passes `image->comps[0].data`). The two buffers must not
/// overlap. `offset0`/`div0` are the C `signed_offsets[0]`/`float_divs[0]`
/// (`1 << (prec - 1)` for signed components else 0, and `(1 << prec) - 1`;
/// `prec <= 16` by the C sanity checks). The `i64`/`i32` mirror the C
/// `long`/`int` exactly on LP64. Lane 3 of every quad is preserved.
#[no_mangle]
pub unsafe extern "C" fn darkroom_j2k_grey_to_float(
    buf: *mut f32,
    comp0: *const i32,
    npixels: usize,
    offset0: i64,
    div0: i32,
) {
    if buf.is_null() || comp0.is_null() || npixels == 0 {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let out_len = match npixels.checked_mul(4) {
        Some(n) => n,
        None => return,
    };
    let out = std::slice::from_raw_parts_mut(buf, out_len);
    let src = std::slice::from_raw_parts(comp0, npixels);
    j2k_grey_to_float(out, src, npixels, offset0, div0);
}

/// # Safety
/// `buf` must hold at least `4 * npixels` floats (the C caller passes the
/// mipmap-cache allocation) and `comp0`/`comp1`/`comp2` at least `npixels`
/// `int` lanes each (the C caller passes `image->comps[0..2].data`).
/// `offsets`/`divs` each point to 3 entries (the C caller passes the
/// `signed_offsets`/`float_divs` arrays, of which entries `0..2` are
/// read). No buffer may overlap another. The `i64`/`i32` mirror the C
/// `long`/`int` exactly on LP64. Lane 3 of every quad is preserved.
#[no_mangle]
pub unsafe extern "C" fn darkroom_j2k_rgb_to_float(
    buf: *mut f32,
    comp0: *const i32,
    comp1: *const i32,
    comp2: *const i32,
    npixels: usize,
    offsets: *const i64,
    divs: *const i32,
) {
    if buf.is_null()
        || comp0.is_null()
        || comp1.is_null()
        || comp2.is_null()
        || offsets.is_null()
        || divs.is_null()
        || npixels == 0
    {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let out_len = match npixels.checked_mul(4) {
        Some(n) => n,
        None => return,
    };
    let out = std::slice::from_raw_parts_mut(buf, out_len);
    let src0 = std::slice::from_raw_parts(comp0, npixels);
    let src1 = std::slice::from_raw_parts(comp1, npixels);
    let src2 = std::slice::from_raw_parts(comp2, npixels);
    let offs = std::slice::from_raw_parts(offsets, 3);
    let divs = std::slice::from_raw_parts(divs, 3);
    j2k_rgb_to_float(out, src0, src1, src2, npixels, offs, divs);
}

// ── sYCC 4:4:4 conversion (m4-212) ───────────────────────────────────────────

/// Convert sYCC 4:4:4 planes to RGB planes, one `int` lane per pixel.
///
/// Port of the former element-wise loop in `sycc444_to_rgb()`
/// (`src/imageio/imageio_j2k.c`): per index `k`, with `cbs = cb[k] -
/// offset` and `crs = cr[k] - offset`,
/// `r[k] = CLAMP(y[k] + (int)(1.402 * (float)crs), 0, upb)` and likewise
/// `g[k] = CLAMP(y[k] - (int)(0.344 * (float)cbs + 0.714 * (float)crs),
/// 0, upb)`, `b[k] = CLAMP(y[k] + (int)(1.772 * (float)cbs), 0, upb)`.
/// `npixels` is the C loop bound (`maxw * maxh`); short buffers iterate
/// clamped (no panic, no out-of-bounds access). A zero `npixels` is a
/// no-op.
#[allow(clippy::too_many_arguments)]
pub fn j2k_sycc444_to_rgb(
    y: &[i32],
    cb: &[i32],
    cr: &[i32],
    r: &mut [i32],
    g: &mut [i32],
    b: &mut [i32],
    npixels: usize,
    offset: i32,
    upb: i32,
) {
    let n = npixels
        .min(y.len())
        .min(cb.len())
        .min(cr.len())
        .min(r.len())
        .min(g.len())
        .min(b.len());
    for k in 0..n {
        // wrapping_sub: real lanes/offsets are small (prec <= 16 bits),
        // but a debug panic across FFI would be UB — wrap instead of
        // trapping on adversarial inputs, as the grey/RGB kernels do.
        let cbs = cb[k].wrapping_sub(offset);
        let crs = cr[k].wrapping_sub(offset);
        let yv = y[k];
        // C product order, in f64 over the exact (float)lane promotion:
        // the constants are unsuffixed doubles, so `(float)cr` promotes
        // and the multiply runs in f64. `as i32` truncates toward zero
        // exactly like the C cast for these in-range values.
        let rv = yv.wrapping_add((1.402f64 * (crs as f32 as f64)) as i32);
        let gv = yv.wrapping_sub(
            (0.344f64 * (cbs as f32 as f64) + 0.714f64 * (crs as f32 as f64)) as i32,
        );
        let bv = yv.wrapping_add((1.772f64 * (cbs as f32 as f64)) as i32);
        // glib CLAMP(v, 0, upb) order, spelled branch-by-branch so an
        // adversarial negative upb (unreachable from the C caller, whose
        // upb is (1 << prec) - 1 >= 1) degrades instead of panicking the
        // way `i32::clamp` would on a reversed range.
        r[k] = if rv > upb { upb } else if rv < 0 { 0 } else { rv };
        g[k] = if gv > upb { upb } else if gv < 0 { 0 } else { gv };
        b[k] = if bv > upb { upb } else if bv < 0 { 0 } else { bv };
    }
}

/// Structurally divergent reference for `j2k_sycc444_to_rgb`: folds the
/// six planes through zipped iterators into a per-pixel scalar helper
/// (the kernel indexes all six sides with `k`), so the sweep test
/// cross-checks indexing as well as values. The arithmetic is the same
/// f64-product-then-truncate-then-clamp spelling by construction (see
/// the module docs: computing the products in f32 must not be used).
/// Same well-formed-buffers precondition, enforced here by the same
/// clamping.
#[cfg(test)]
fn sycc_pixel(yv: i32, cbv: i32, crv: i32, offset: i32, upb: i32) -> (i32, i32, i32) {
    let cbs = cbv.wrapping_sub(offset);
    let crs = crv.wrapping_sub(offset);
    let rv = yv.wrapping_add((1.402f64 * f64::from(crs as f32)) as i32);
    let gv = yv.wrapping_sub(
        (0.344f64 * f64::from(cbs as f32) + 0.714f64 * f64::from(crs as f32)) as i32,
    );
    let bv = yv.wrapping_add((1.772f64 * f64::from(cbs as f32)) as i32);
    let clamp = |v: i32| {
        if v > upb {
            upb
        } else if v < 0 {
            0
        } else {
            v
        }
    };
    (clamp(rv), clamp(gv), clamp(bv))
}

#[cfg(test)]
fn ref_j2k_sycc444_to_rgb(
    y: &[i32],
    cb: &[i32],
    cr: &[i32],
    r: &mut [i32],
    g: &mut [i32],
    b: &mut [i32],
    npixels: usize,
    offset: i32,
    upb: i32,
) {
    let n = npixels
        .min(y.len())
        .min(cb.len())
        .min(cr.len())
        .min(r.len())
        .min(g.len())
        .min(b.len());
    let ins = y.iter().zip(cb.iter()).zip(cr.iter());
    let outs = r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut());
    for (((&yv, &cbv), &crv), ((rv, gv), bv)) in ins.zip(outs).take(n) {
        let (rr, gg, bb) = sycc_pixel(yv, cbv, crv, offset, upb);
        *rv = rr;
        *gv = gg;
        *bv = bb;
    }
}

/// # Safety
/// `y`/`cb`/`cr` must each hold at least `npixels` `int` lanes (the C
/// caller passes the OpenJPEG component planes) and `r`/`g`/`b` at
/// least `npixels` `int` lanes each (the C caller passes the calloc
/// allocations that replace the planes). No buffer may overlap another.
/// `offset`/`upb` are the C `1 << (prec - 1)` / `(1 << prec) - 1`.
#[no_mangle]
pub unsafe extern "C" fn darkroom_j2k_sycc444_to_rgb(
    y: *const i32,
    cb: *const i32,
    cr: *const i32,
    r: *mut i32,
    g: *mut i32,
    b: *mut i32,
    npixels: usize,
    offset: i32,
    upb: i32,
) {
    if y.is_null()
        || cb.is_null()
        || cr.is_null()
        || r.is_null()
        || g.is_null()
        || b.is_null()
        || npixels == 0
    {
        return;
    }
    // No length product to validate here (one lane per pixel per plane),
    // but refuse lengths that cannot back a slice so the constructions
    // below stay inside the language model; the safe kernel re-checks
    // defensively via clamped iteration.
    if npixels > isize::MAX as usize {
        return;
    }
    let y = std::slice::from_raw_parts(y, npixels);
    let cb = std::slice::from_raw_parts(cb, npixels);
    let cr = std::slice::from_raw_parts(cr, npixels);
    let r = std::slice::from_raw_parts_mut(r, npixels);
    let g = std::slice::from_raw_parts_mut(g, npixels);
    let b = std::slice::from_raw_parts_mut(b, npixels);
    j2k_sycc444_to_rgb(y, cb, cr, r, g, b, npixels, offset, upb);
}

// ── 12-bit export pack (m4-217) ──────────────────────────────────────────────

/// 12-bit full-scale value of the `DOWNSAMPLE_FLOAT_TO_12BIT` macro
/// (`src/imageio/format/j2k.c`): lanes at or above 1.0 quantize to this.
pub const J2K_12BIT_MAX: i32 = 4095;

/// Quantize one float lane exactly as `DOWNSAMPLE_FLOAT_TO_12BIT` spells
/// it: at or below zero yields 0, at or above one yields 4095, otherwise
/// `(int)roundf(4095.0 * v)` in f32. Shared by the kernel and the
/// divergent reference below so the two cannot drift on the spelling
/// while still differing structurally. NaN falls to the scale branch
/// and saturates to 0 via `as i32` (see the module docs for the x86 C
/// divergence this deliberately accepts).
fn downsample_12bit_lane(v: f32) -> i32 {
    if v <= 0.0 {
        0
    } else if v >= 1.0 {
        J2K_12BIT_MAX
    } else {
        (4095.0f32 * v).round() as i32
    }
}

/// Quantize float RGB lanes into three 12-bit OpenJPEG component planes.
///
/// Port of the former element-wise loop in the `case 12` branch of
/// `write_image()` (`src/imageio/format/j2k.c`): per pixel `i`,
/// `c[k][i] = DOWNSAMPLE_FLOAT_TO_12BIT(src[4*i + k])` for `k` in
/// `0..2`; the alpha lane (`src[4*i + 3]`) is never read. `npixels` is
/// the C loop bound (`w * h`); short buffers iterate clamped (no panic,
/// no out-of-bounds access). A zero `npixels` is a no-op.
///
/// The Rust kernel is single-threaded sequential; the C loop was
/// `DT_OMP_FOR_SIMD(collapse(2))` over pixels and components, but each
/// output lane reads only its own source lane, so thread scheduling
/// cannot change the result.
pub fn j2k_float_to_12bit(
    src: &[f32],
    c0: &mut [i32],
    c1: &mut [i32],
    c2: &mut [i32],
    npixels: usize,
) {
    let n = npixels
        .min(src.len() / 4)
        .min(c0.len())
        .min(c1.len())
        .min(c2.len());
    for i in 0..n {
        let s = 4 * i;
        c0[i] = downsample_12bit_lane(src[s]);
        c1[i] = downsample_12bit_lane(src[s + 1]);
        c2[i] = downsample_12bit_lane(src[s + 2]);
    }
}

/// Structurally divergent reference for `j2k_float_to_12bit`: walks the
/// output in per-pixel triples zipped with 4-lane source chunks (the
/// kernel indexes the source quad and the three planes with `4*i`/`i`),
/// so the sweep test cross-checks indexing as well as values. The lane
/// arithmetic is the shared `downsample_12bit_lane` helper by
/// construction (see the module docs: a clamp-then-round or
/// divide-then-scale respelling must not be used). Same
/// well-formed-buffers precondition, enforced here by the same
/// clamping.
#[cfg(test)]
fn ref_j2k_float_to_12bit(
    src: &[f32],
    c0: &mut [i32],
    c1: &mut [i32],
    c2: &mut [i32],
    npixels: usize,
) {
    let n = npixels
        .min(src.len() / 4)
        .min(c0.len())
        .min(c1.len())
        .min(c2.len());
    let outs = c0
        .iter_mut()
        .zip(c1.iter_mut())
        .zip(c2.iter_mut())
        .zip(src.chunks_exact(4))
        .take(n);
    for (((o0, o1), o2), quad) in outs {
        *o0 = downsample_12bit_lane(quad[0]);
        *o1 = downsample_12bit_lane(quad[1]);
        *o2 = downsample_12bit_lane(quad[2]);
    }
}

/// # Safety
/// `in_data` must hold at least `4 * npixels` floats (the C caller
/// passes the export pipeline buffer, tightly packed RGBA whose alpha
/// lane is never read) and `comp0`/`comp1`/`comp2` at least `npixels`
/// `int` lanes each (the C caller passes `image->comps[0..2].data`,
/// the 12-bit `prec` planes). No buffer may overlap another.
#[no_mangle]
pub unsafe extern "C" fn darkroom_j2k_float_to_12bit(
    in_data: *const f32,
    comp0: *mut i32,
    comp1: *mut i32,
    comp2: *mut i32,
    npixels: usize,
) {
    if in_data.is_null()
        || comp0.is_null()
        || comp1.is_null()
        || comp2.is_null()
        || npixels == 0
    {
        return;
    }
    // validate the products BEFORE building the slices below (a misuse
    // caller could otherwise wrap a length; the safe kernel re-checks
    // defensively via clamped iteration)
    let src_len = match npixels.checked_mul(4) {
        Some(n) => n,
        None => return,
    };
    if npixels > isize::MAX as usize {
        return;
    }
    let src = std::slice::from_raw_parts(in_data, src_len);
    let c0 = std::slice::from_raw_parts_mut(comp0, npixels);
    let c1 = std::slice::from_raw_parts_mut(comp1, npixels);
    let c2 = std::slice::from_raw_parts_mut(comp2, npixels);
    j2k_float_to_12bit(src, c0, c1, c2, npixels);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SENTINEL: f32 = -7.25;

    // Grey rails: unsigned 8-bit (offset 0, div 255) maps 0 -> +0.0 and
    // 255 -> 1.0 by division (255/255, not a reciprocal multiply); signed
    // 8-bit (offset 128, div 255) maps -128 -> +0.0 and 127 -> 1.0. The
    // alpha lane keeps its sentinel in both cases.
    #[test]
    fn grey_rails_pin() {
        let mut out = vec![SENTINEL; 8];
        j2k_grey_to_float(&mut out, &[0, 255], 2, 0, 255);
        assert_eq!(out[0].to_bits(), 0x0000_0000);
        assert_eq!(out[1].to_bits(), 0x0000_0000);
        assert_eq!(out[2].to_bits(), 0x0000_0000);
        assert_eq!(out[3].to_bits(), SENTINEL.to_bits());
        assert_eq!(out[4].to_bits(), (255.0f32 / 255.0f32).to_bits());
        assert_eq!(out[5].to_bits(), (255.0f32 / 255.0f32).to_bits());
        assert_eq!(out[6].to_bits(), (255.0f32 / 255.0f32).to_bits());
        assert_eq!(out[7].to_bits(), SENTINEL.to_bits());

        let mut out = vec![SENTINEL; 8];
        j2k_grey_to_float(&mut out, &[-128, 127], 2, 128, 255);
        assert_eq!(out[0].to_bits(), 0x0000_0000);
        assert_eq!(out[4].to_bits(), (255.0f32 / 255.0f32).to_bits());
        assert_eq!(out[3].to_bits(), SENTINEL.to_bits());
        assert_eq!(out[7].to_bits(), SENTINEL.to_bits());
    }

    // RGB rails with mixed per-component precision: comp0 unsigned 8-bit,
    // comp1 unsigned 10-bit (div 1023), comp2 signed 8-bit (offset 128).
    // Each channel divides by its own divisor; alpha keeps its sentinel.
    #[test]
    fn rgb_rails_pin() {
        let mut out = vec![SENTINEL; 4];
        j2k_rgb_to_float(&mut out, &[0], &[1023], &[-128], 1, &[0, 0, 128], &[255, 1023, 255]);
        assert_eq!(out[0].to_bits(), 0x0000_0000);
        assert_eq!(out[1].to_bits(), (1023.0f32 / 1023.0f32).to_bits());
        assert_eq!(out[2].to_bits(), 0x0000_0000);
        assert_eq!(out[3].to_bits(), SENTINEL.to_bits());
    }

    // Pin the divide-per-lane spelling against a reciprocal multiply: find
    // a lane/divisor where `x/d` and `x*(1/d)` observably differ in f32
    // (the hoisted reciprocal is itself inexact, so double rounding can
    // land on the other side of a rounding boundary), then assert the
    // kernel takes the division side. Without this the sweep below would
    // pass even if both sides shared the multiply spelling.
    #[test]
    fn division_spelling_differs_from_reciprocal_multiply() {
        let mut probe: Option<(i32, i32)> = None;
        for &div in &[255i32, 1023, 4095, 100, 7] {
            let inv = 1.0f32 / div as f32;
            let mut n: i64 = 1;
            while probe.is_none() && n < (1i64 << 31) {
                let a = n as f32 / div as f32;
                let b = n as f32 * inv;
                if a.to_bits() != b.to_bits() {
                    probe = Some((n as i32, div));
                }
                n = if n < 1000 { n + 1 } else { n + n / 7 + 1 };
            }
        }
        let (lane, div) = probe.expect("no lane/divisor distinguishes / from *recip");
        let mut out = vec![SENTINEL; 4];
        j2k_grey_to_float(&mut out, &[lane], 1, 0, div);
        let divided = lane as f32 / div as f32;
        let multiplied = lane as f32 * (1.0f32 / div as f32);
        assert_ne!(divided.to_bits(), multiplied.to_bits());
        assert_eq!(out[0].to_bits(), divided.to_bits());
        assert_eq!(out[3].to_bits(), SENTINEL.to_bits());
    }

    // Grey kernel and reference must agree bit-exactly on every written
    // lane over several shapes, with full-range i32 lanes (including
    // negatives) and per-shape offsets; lane 3 must keep its sentinel
    // everywhere.
    #[test]
    fn grey_matches_reference_over_sweep() {
        let shapes = [1usize, 2, 7, 20, 64];
        for (s, n) in shapes.iter().enumerate() {
            let n = *n;
            let mut comp = vec![0i32; n];
            for (i, v) in comp.iter_mut().enumerate() {
                *v = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(s as u64 * 0x9E37) as u32 as i32;
            }
            for (offset, div) in [(0i64, 255i32), (128, 255), (2048, 4095), (0, 1)] {
                let mut direct = vec![SENTINEL; 4 * n];
                let mut reference = vec![SENTINEL; 4 * n];
                j2k_grey_to_float(&mut direct, &comp, n, offset, div);
                ref_j2k_grey_to_float(&mut reference, &comp, n, offset, div);
                assert_eq!(direct.len(), reference.len());
                for (k, (d, r)) in direct.iter().zip(reference.iter()).enumerate() {
                    assert_eq!(d.to_bits(), r.to_bits(), "shape {n} lane {k}");
                    if k % 4 == 3 {
                        assert_eq!(d.to_bits(), SENTINEL.to_bits(), "alpha lane {k} preserved");
                    }
                }
                // broadcast check: the three written lanes of each quad agree
                for q in 0..n {
                    assert_eq!(direct[4 * q].to_bits(), direct[4 * q + 1].to_bits());
                    assert_eq!(direct[4 * q].to_bits(), direct[4 * q + 2].to_bits());
                }
            }
        }
    }

    // RGB kernel and reference must agree bit-exactly on every written
    // lane over several shapes with mixed offsets/divisors; lane 3 must
    // keep its sentinel everywhere.
    #[test]
    fn rgb_matches_reference_over_sweep() {
        let shapes = [1usize, 2, 7, 20, 64];
        for (s, n) in shapes.iter().enumerate() {
            let n = *n;
            let mut comps = [vec![0i32; n], vec![0i32; n], vec![0i32; n]];
            for (c, comp) in comps.iter_mut().enumerate() {
                for (i, v) in comp.iter_mut().enumerate() {
                    *v = (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add((s * 3 + c) as u64 * 0x9E37) as u32 as i32;
                }
            }
            let offsets = [0i64, 512, 128];
            let divs = [255i32, 1023, 255];
            let mut direct = vec![SENTINEL; 4 * n];
            let mut reference = vec![SENTINEL; 4 * n];
            j2k_rgb_to_float(
                &mut direct,
                &comps[0],
                &comps[1],
                &comps[2],
                n,
                &offsets,
                &divs,
            );
            ref_j2k_rgb_to_float(
                &mut reference,
                &comps[0],
                &comps[1],
                &comps[2],
                n,
                &offsets,
                &divs,
            );
            assert_eq!(direct.len(), reference.len());
            for (k, (d, r)) in direct.iter().zip(reference.iter()).enumerate() {
                assert_eq!(d.to_bits(), r.to_bits(), "shape {n} lane {k}");
                if k % 4 == 3 {
                    assert_eq!(d.to_bits(), SENTINEL.to_bits(), "alpha lane {k} preserved");
                }
            }
        }
    }

    #[test]
    fn degenerate_guards_no_op() {
        let comp = vec![100i32; 4];
        // zero pixels: output untouched
        let mut out = vec![SENTINEL; 16];
        j2k_grey_to_float(&mut out, &comp, 0, 0, 255);
        assert_eq!(out, vec![SENTINEL; 16]);
        j2k_rgb_to_float(&mut out, &comp, &comp, &comp, 0, &[0, 0, 0], &[255, 255, 255]);
        assert_eq!(out, vec![SENTINEL; 16]);
        // empty buffers with live dims: no panic, no writes possible
        let empty: Vec<i32> = vec![];
        let mut out2 = vec![SENTINEL; 4];
        j2k_grey_to_float(&mut out2, &empty, 1, 0, 255);
        assert_eq!(out2, vec![SENTINEL; 4]);
        // truncated source: no panic, written quads convert fully,
        // unwritten quads stay untouched
        let short = vec![255i32; 1];
        let mut short_out = vec![SENTINEL; 16];
        j2k_grey_to_float(&mut short_out, &short, 4, 0, 255);
        assert_eq!(short_out[0].to_bits(), (255.0f32 / 255.0f32).to_bits());
        assert_eq!(short_out[3].to_bits(), SENTINEL.to_bits());
        assert_eq!(&short_out[4..16], &[SENTINEL; 12]);
        // truncated output: no panic, partial quads never partially written
        let mut narrow = vec![SENTINEL; 6];
        j2k_grey_to_float(&mut narrow, &[255i32; 4], 4, 0, 255);
        assert_eq!(narrow[0].to_bits(), (255.0f32 / 255.0f32).to_bits());
        assert_eq!(&narrow[4..6], &[SENTINEL; 2]);
        // short offset/div tables: rgb is a no-op
        let mut out3 = vec![SENTINEL; 4];
        j2k_rgb_to_float(&mut out3, &comp, &comp, &comp, 1, &[0, 0], &[255, 255, 255]);
        j2k_rgb_to_float(&mut out3, &comp, &comp, &comp, 1, &[0, 0, 0], &[255, 255]);
        assert_eq!(out3, vec![SENTINEL; 4]);
    }

    #[test]
    fn ffi_round_trip() {
        let n = 37usize;
        let mut comp = vec![0i32; n];
        for (i, v) in comp.iter_mut().enumerate() {
            *v = (i as u64).wrapping_mul(2_654_435_761) as u32 as i32;
        }
        let mut ffi_out = vec![SENTINEL; 4 * n];
        let mut direct_out = vec![SENTINEL; 4 * n];
        unsafe {
            darkroom_j2k_grey_to_float(ffi_out.as_mut_ptr(), comp.as_ptr(), n, 128, 255);
        }
        j2k_grey_to_float(&mut direct_out, &comp, n, 128, 255);
        assert_eq!(ffi_out, direct_out);

        let mut ffi_rgb = vec![SENTINEL; 4 * n];
        let mut direct_rgb = vec![SENTINEL; 4 * n];
        let offsets = [0i64, 0, 0];
        let divs = [255i32, 255, 255];
        unsafe {
            darkroom_j2k_rgb_to_float(
                ffi_rgb.as_mut_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                n,
                offsets.as_ptr(),
                divs.as_ptr(),
            );
        }
        j2k_rgb_to_float(&mut direct_rgb, &comp, &comp, &comp, n, &offsets, &divs);
        assert_eq!(ffi_rgb, direct_rgb);
    }

    #[test]
    fn ffi_guards() {
        let comp = vec![100i32; 4];
        let mut out = vec![SENTINEL; 16];
        let offsets = [0i64, 0, 0];
        let divs = [255i32, 255, 255];
        unsafe {
            // null pointers
            darkroom_j2k_grey_to_float(std::ptr::null_mut(), comp.as_ptr(), 1, 0, 255);
            darkroom_j2k_grey_to_float(out.as_mut_ptr(), std::ptr::null(), 1, 0, 255);
            darkroom_j2k_rgb_to_float(
                std::ptr::null_mut(),
                comp.as_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                1,
                offsets.as_ptr(),
                divs.as_ptr(),
            );
            darkroom_j2k_rgb_to_float(
                out.as_mut_ptr(),
                std::ptr::null(),
                comp.as_ptr(),
                comp.as_ptr(),
                1,
                offsets.as_ptr(),
                divs.as_ptr(),
            );
            darkroom_j2k_rgb_to_float(
                out.as_mut_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                1,
                std::ptr::null(),
                divs.as_ptr(),
            );
            darkroom_j2k_rgb_to_float(
                out.as_mut_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                1,
                offsets.as_ptr(),
                std::ptr::null(),
            );
            // zero pixels
            darkroom_j2k_grey_to_float(out.as_mut_ptr(), comp.as_ptr(), 0, 0, 255);
            darkroom_j2k_rgb_to_float(
                out.as_mut_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                0,
                offsets.as_ptr(),
                divs.as_ptr(),
            );
            // overflowing dims
            darkroom_j2k_grey_to_float(out.as_mut_ptr(), comp.as_ptr(), usize::MAX, 0, 255);
            darkroom_j2k_rgb_to_float(
                out.as_mut_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                comp.as_ptr(),
                usize::MAX,
                offsets.as_ptr(),
                divs.as_ptr(),
            );
        }
        assert_eq!(out, vec![SENTINEL; 16]); // untouched
    }

    // ── sYCC 4:4:4 (m4-212) ──────────────────────────────────────────────

    // Achromatic pixels pass through, rails clamp: 8-bit (offset 128,
    // upb 255) with Cb == Cr == 128 reproduces Y on all three lanes;
    // a hot chroma lane drives its channel into the rail.
    #[test]
    fn sycc444_neutral_and_rails_pin() {
        let (offset, upb) = (128, 255);
        let run = |y: &[i32], cb: &[i32], cr: &[i32]| {
            let n = y.len();
            let (mut r, mut g, mut b) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
            j2k_sycc444_to_rgb(y, cb, cr, &mut r, &mut g, &mut b, n, offset, upb);
            (r, g, b)
        };
        // neutral grey ramps straight through
        let (r, g, b) = run(&[0, 100, 255], &[128, 128, 128], &[128, 128, 128]);
        assert_eq!((r, g, b), (vec![0, 100, 255], vec![0, 100, 255], vec![0, 100, 255]));
        // hot blue-yellow lane: b rails at 255, g takes the partial hit
        // b = 200 + (int)(1.772 * 127) = 200 + 225; g = 200 - (int)(0.344 * 127)
        let (r, g, b) = run(&[200], &[255], &[128]);
        assert_eq!((r, g, b), (vec![200], vec![157], vec![255]));
        // cold red lane on dark Y: r rails at 0
        // r = 10 + (int)(1.402 * -128) = 10 - 179
        let (r, g, b) = run(&[10], &[0], &[0]);
        assert_eq!((r, g, b), (vec![0], vec![145], vec![0]));
    }

    // Pin the f64-product spelling against an f32-product variant: the
    // C constants are unsuffixed doubles, so a kernel that multiplied
    // in f32 would observably differ on large-magnitude lanes. The pin
    // below sits strictly inside the rails, so the asserted value is the
    // raw truncated product, not a clamp. (An earlier revision pinned
    // r=4435/b=3296, but both spellings truncate identically there, so
    // those pins passed under the wrong spelling too and were vacuous.)
    #[test]
    fn sycc444_precision_spelling_is_f64() {
        let (offset, upb) = (0, 65535);
        // r = 60000 + (int)(1.402 * -41500): the f64 product is exactly
        // -58183.0, truncating to -58183 (r = 1817), while an f32 product
        // rounds to -58182.99609375, truncating to -58182 (r = 1818).
        let (mut r, mut g, mut b) = (vec![0i32; 1], vec![0i32; 1], vec![0i32; 1]);
        j2k_sycc444_to_rgb(&[60000], &[0], &[-41500], &mut r, &mut g, &mut b, 1, offset, upb);
        assert_eq!(r[0], 1817);
        // The 1.772 term shows no f32/f64 truncation divergence anywhere in
        // the realistic lane range, so the r-pin above plus inspection is
        // the practical coverage for the spelling requirement.
    }

    // Kernel and reference must agree exactly over several shapes,
    // precisions (8/12/16-bit offsets and rails), and full-span lanes
    // including out-of-range values that must clamp rather than wrap.
    #[test]
    fn sycc444_matches_reference_over_sweep() {
        let shapes = [1usize, 2, 7, 20, 64];
        let params = [(128i32, 255i32), (2048, 4095), (32768, 65535)];
        for (s, n) in shapes.iter().enumerate() {
            let n = *n;
            let mut planes = [vec![0i32; n], vec![0i32; n], vec![0i32; n]];
            for (c, plane) in planes.iter_mut().enumerate() {
                for (i, v) in plane.iter_mut().enumerate() {
                    // lanes span roughly [-70000, 70000]: in-range,
                    // rails, and beyond-the-rails values
                    *v = ((i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add((s * 3 + c) as u64 * 0x9E37)
                        % 140_001) as i32
                        - 70_000;
                }
            }
            for &(offset, upb) in &params {
                let (mut dr, mut dg, mut db) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
                let (mut rr, mut rg, mut rb) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
                j2k_sycc444_to_rgb(
                    &planes[0], &planes[1], &planes[2],
                    &mut dr, &mut dg, &mut db, n, offset, upb,
                );
                ref_j2k_sycc444_to_rgb(
                    &planes[0], &planes[1], &planes[2],
                    &mut rr, &mut rg, &mut rb, n, offset, upb,
                );
                for k in 0..n {
                    assert_eq!(dr[k], rr[k], "shape {n} r lane {k}");
                    assert_eq!(dg[k], rg[k], "shape {n} g lane {k}");
                    assert_eq!(db[k], rb[k], "shape {n} b lane {k}");
                    assert!((0..=upb).contains(&dr[k]), "r clamped {k}");
                    assert!((0..=upb).contains(&dg[k]), "g clamped {k}");
                    assert!((0..=upb).contains(&db[k]), "b clamped {k}");
                }
            }
        }
    }

    #[test]
    fn sycc444_degenerate_guards_no_op() {
        let (offset, upb) = (128, 255);
        // zero pixels: outputs untouched
        let (mut r, mut g, mut b) = (vec![9i32; 4], vec![9i32; 4], vec![9i32; 4]);
        j2k_sycc444_to_rgb(&[1; 4], &[1; 4], &[1; 4], &mut r, &mut g, &mut b, 0, offset, upb);
        assert_eq!((r, g, b), (vec![9; 4], vec![9; 4], vec![9; 4]));
        // empty planes with live dims: no panic, no writes possible
        let empty: Vec<i32> = vec![];
        let (mut r, mut g, mut b) = (vec![9i32; 2], vec![9i32; 2], vec![9i32; 2]);
        j2k_sycc444_to_rgb(&empty, &empty, &empty, &mut r, &mut g, &mut b, 2, offset, upb);
        assert_eq!((r, g, b), (vec![9; 2], vec![9; 2], vec![9; 2]));
        // truncated source: written prefix converts fully, the unwritten
        // tail stays untouched
        let (mut r, mut g, mut b) = (vec![9i32; 4], vec![9i32; 4], vec![9i32; 4]);
        j2k_sycc444_to_rgb(&[100, 100], &[128, 128], &[128, 128], &mut r, &mut g, &mut b, 4, offset, upb);
        assert_eq!(&r[..2], &[100, 100]);
        assert_eq!(&r[2..], &[9, 9]);
        // truncated output: no panic, partial planes never partially written
        let (mut r, mut g, mut b) = (vec![9i32; 1], vec![9i32; 4], vec![9i32; 4]);
        j2k_sycc444_to_rgb(
            &[100, 100], &[128, 128], &[128, 128],
            &mut r, &mut g, &mut b, 2, offset, upb,
        );
        assert_eq!(r, vec![100]);
        assert_eq!(&g[..1], &[100]);
        assert_eq!(&g[2..], &[9, 9]);
    }

    #[test]
    fn sycc444_ffi_round_trip() {
        let n = 37usize;
        let (offset, upb) = (2048, 4095);
        let mut planes = [vec![0i32; n], vec![0i32; n], vec![0i32; n]];
        for (c, plane) in planes.iter_mut().enumerate() {
            for (i, v) in plane.iter_mut().enumerate() {
                *v = ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(c as u64 * 0x9E37)
                    % 6000) as i32
                    - 1000;
            }
        }
        let (mut fr, mut fg, mut fb) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        let (mut dr, mut dg, mut db) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        unsafe {
            darkroom_j2k_sycc444_to_rgb(
                planes[0].as_ptr(),
                planes[1].as_ptr(),
                planes[2].as_ptr(),
                fr.as_mut_ptr(),
                fg.as_mut_ptr(),
                fb.as_mut_ptr(),
                n,
                offset,
                upb,
            );
        }
        j2k_sycc444_to_rgb(
            &planes[0], &planes[1], &planes[2],
            &mut dr, &mut dg, &mut db, n, offset, upb,
        );
        assert_eq!((fr, fg, fb), (dr, dg, db));
    }

    #[test]
    fn sycc444_ffi_guards() {
        let src = vec![100i32; 4];
        let (mut r, mut g, mut b) = (vec![9i32; 4], vec![9i32; 4], vec![9i32; 4]);
        unsafe {
            // each null plane in turn
            darkroom_j2k_sycc444_to_rgb(
                std::ptr::null(), src.as_ptr(), src.as_ptr(),
                r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr(), 1, 128, 255,
            );
            darkroom_j2k_sycc444_to_rgb(
                src.as_ptr(), std::ptr::null(), src.as_ptr(),
                r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr(), 1, 128, 255,
            );
            darkroom_j2k_sycc444_to_rgb(
                src.as_ptr(), src.as_ptr(), std::ptr::null(),
                r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr(), 1, 128, 255,
            );
            darkroom_j2k_sycc444_to_rgb(
                src.as_ptr(), src.as_ptr(), src.as_ptr(),
                std::ptr::null_mut(), g.as_mut_ptr(), b.as_mut_ptr(), 1, 128, 255,
            );
            darkroom_j2k_sycc444_to_rgb(
                src.as_ptr(), src.as_ptr(), src.as_ptr(),
                r.as_mut_ptr(), std::ptr::null_mut(), b.as_mut_ptr(), 1, 128, 255,
            );
            darkroom_j2k_sycc444_to_rgb(
                src.as_ptr(), src.as_ptr(), src.as_ptr(),
                r.as_mut_ptr(), g.as_mut_ptr(), std::ptr::null_mut(), 1, 128, 255,
            );
            // zero pixels
            darkroom_j2k_sycc444_to_rgb(
                src.as_ptr(), src.as_ptr(), src.as_ptr(),
                r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr(), 0, 128, 255,
            );
            // unsliceable dims
            darkroom_j2k_sycc444_to_rgb(
                src.as_ptr(), src.as_ptr(), src.as_ptr(),
                r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr(), usize::MAX, 128, 255,
            );
        }
        assert_eq!((r, g, b), (vec![9; 4], vec![9; 4], vec![9; 4])); // untouched
    }

    // ── 12-bit export pack (m4-217) ──────────────────────────────────────

    // Rails of DOWNSAMPLE_FLOAT_TO_12BIT: 0.0 and -0.0 take the zero
    // branch (the macro spells `<= 0.0f`), 1.0 and above take the 4095
    // branch, 0.5 pins the scale-then-round spelling (0.5 * 4095 =
    // 2047.5 -> 2048, round-half-away). The alpha lane is never read:
    // garbage (including NaN) there must not affect any output lane.
    #[test]
    fn float12_rails_pin() {
        let src = [
            0.0f32, 0.5, 1.0, f32::NAN, // px0: alpha NaN unread
            -0.0, 2.0, -1.0, -3.25, // px1: alpha garbage unread
        ];
        let (mut c0, mut c1, mut c2) = (vec![-1i32; 2], vec![-1i32; 2], vec![-1i32; 2]);
        j2k_float_to_12bit(&src, &mut c0, &mut c1, &mut c2, 2);
        assert_eq!((c0[0], c1[0], c2[0]), (0, 2048, 4095));
        assert_eq!((c0[1], c1[1], c2[1]), (0, 4095, 0));
    }

    // Round-half-away (roundf) spelling: 0.5/4095 scales to exactly 0.5
    // and rounds to 1 (round-half-even would give 0), 1.5/4095 rounds to
    // 2. A tiny negative takes the `<= 0.0` branch to 0 (scaling first
    // would give round(-0.5) = -1). NaN falls to the scale branch and
    // saturates to 0 via `as i32` — the x86 C cast instead yields
    // INT_MIN there (see the module docs); the pin locks the defined
    // Rust behaviour.
    #[test]
    fn float12_rounding_and_branch_spelling() {
        let unit = 1.0f32 / 4095.0;
        let src = [0.5 * unit, 1.5 * unit, -0.5 * unit, 0.0];
        let (mut c0, mut c1, mut c2) = (vec![-1i32; 1], vec![-1i32; 1], vec![-1i32; 1]);
        j2k_float_to_12bit(&src, &mut c0, &mut c1, &mut c2, 1);
        assert_eq!((c0[0], c1[0], c2[0]), (1, 2, 0));
        let nan = [f32::NAN, 0.0, 0.0, 0.0];
        let (mut n0, mut n1, mut n2) = (vec![-1i32; 1], vec![-1i32; 1], vec![-1i32; 1]);
        j2k_float_to_12bit(&nan, &mut n0, &mut n1, &mut n2, 1);
        assert_eq!((n0[0], n1[0], n2[0]), (0, 0, 0));
    }

    // Kernel and reference must agree exactly over several shapes with
    // inputs spanning below 0 and above 1, so both rails and the
    // rounding are exercised on every lane.
    #[test]
    fn float12_matches_reference_over_sweep() {
        for n in [1usize, 2, 7, 20, 64] {
            let mut src = vec![0.0f32; 4 * n];
            for (i, v) in src.iter_mut().enumerate() {
                // deterministic sweep across [-0.5, 1.5]
                let k =
                    ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(0x9E37) % 2001) as f32;
                *v = k / 1000.0 - 0.5;
            }
            let (mut d0, mut d1, mut d2) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
            let (mut r0, mut r1, mut r2) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
            j2k_float_to_12bit(&src, &mut d0, &mut d1, &mut d2, n);
            ref_j2k_float_to_12bit(&src, &mut r0, &mut r1, &mut r2, n);
            assert_eq!((&d0, &d1, &d2), (&r0, &r1, &r2), "shape {n}");
            for k in 0..n {
                assert!((0..=J2K_12BIT_MAX).contains(&d0[k]), "c0 clamped {k}");
                assert!((0..=J2K_12BIT_MAX).contains(&d1[k]), "c1 clamped {k}");
                assert!((0..=J2K_12BIT_MAX).contains(&d2[k]), "c2 clamped {k}");
            }
        }
    }

    #[test]
    fn float12_degenerate_guards_no_op() {
        let src = vec![1.0f32; 8];
        // zero pixels: outputs untouched
        let (mut c0, mut c1, mut c2) = (vec![9i32; 2], vec![9i32; 2], vec![9i32; 2]);
        j2k_float_to_12bit(&src, &mut c0, &mut c1, &mut c2, 0);
        assert_eq!((c0, c1, c2), (vec![9; 2], vec![9; 2], vec![9; 2]));
        // empty source with live dims: no panic, no writes possible
        let empty_src: Vec<f32> = vec![];
        let (mut f0, mut f1, mut f2) = (vec![9i32; 2], vec![9i32; 2], vec![9i32; 2]);
        j2k_float_to_12bit(&empty_src, &mut f0, &mut f1, &mut f2, 2);
        assert_eq!((f0, f1, f2), (vec![9; 2], vec![9; 2], vec![9; 2]));
        // empty output plane with live dims: no panic, nothing written
        // anywhere (the clamped bound collapses to zero for all planes)
        let mut empty_plane: Vec<i32> = vec![];
        let (mut g1, mut g2) = (vec![9i32; 2], vec![9i32; 2]);
        j2k_float_to_12bit(&[1.0; 8], &mut empty_plane, &mut g1, &mut g2, 2);
        assert!(empty_plane.is_empty());
        assert_eq!((g1, g2), (vec![9; 2], vec![9; 2]));
        // truncated source: written prefix converts fully, the unwritten
        // tail stays untouched
        let (mut t0, mut t1, mut t2) = (vec![9i32; 4], vec![9i32; 4], vec![9i32; 4]);
        j2k_float_to_12bit(&[1.0, 1.0, 1.0, 0.0], &mut t0, &mut t1, &mut t2, 4);
        assert_eq!(&t0[..1], &[4095]);
        assert_eq!(&t0[1..], &[9, 9, 9]);
        // truncated output: no panic, partial planes never partially written
        let (mut s0, mut s1, mut s2) = (vec![9i32; 1], vec![9i32; 4], vec![9i32; 4]);
        j2k_float_to_12bit(&[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], &mut s0, &mut s1, &mut s2, 2);
        assert_eq!(s0, vec![4095]);
        assert_eq!(&s1[..1], &[4095]);
        assert_eq!(&s1[1..], &[9, 9, 9]);
    }

    #[test]
    fn float12_ffi_round_trip() {
        let n = 37usize;
        let mut src = vec![0.0f32; 4 * n];
        for (i, v) in src.iter_mut().enumerate() {
            let k = ((i as u64).wrapping_mul(2_654_435_761) % 2001) as f32;
            *v = k / 1000.0 - 0.5;
        }
        let (mut f0, mut f1, mut f2) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        let (mut d0, mut d1, mut d2) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        unsafe {
            darkroom_j2k_float_to_12bit(
                src.as_ptr(),
                f0.as_mut_ptr(),
                f1.as_mut_ptr(),
                f2.as_mut_ptr(),
                n,
            );
        }
        j2k_float_to_12bit(&src, &mut d0, &mut d1, &mut d2, n);
        assert_eq!((f0, f1, f2), (d0, d1, d2));
    }

    #[test]
    fn float12_ffi_guards() {
        let src = vec![1.0f32; 16];
        let (mut c0, mut c1, mut c2) = (vec![9i32; 4], vec![9i32; 4], vec![9i32; 4]);
        unsafe {
            // each null plane in turn
            darkroom_j2k_float_to_12bit(
                std::ptr::null(),
                c0.as_mut_ptr(),
                c1.as_mut_ptr(),
                c2.as_mut_ptr(),
                1,
            );
            darkroom_j2k_float_to_12bit(
                src.as_ptr(),
                std::ptr::null_mut(),
                c1.as_mut_ptr(),
                c2.as_mut_ptr(),
                1,
            );
            darkroom_j2k_float_to_12bit(
                src.as_ptr(),
                c0.as_mut_ptr(),
                std::ptr::null_mut(),
                c2.as_mut_ptr(),
                1,
            );
            darkroom_j2k_float_to_12bit(
                src.as_ptr(),
                c0.as_mut_ptr(),
                c1.as_mut_ptr(),
                std::ptr::null_mut(),
                1,
            );
            // zero pixels
            darkroom_j2k_float_to_12bit(
                src.as_ptr(),
                c0.as_mut_ptr(),
                c1.as_mut_ptr(),
                c2.as_mut_ptr(),
                0,
            );
            // unsliceable dims
            darkroom_j2k_float_to_12bit(
                src.as_ptr(),
                c0.as_mut_ptr(),
                c1.as_mut_ptr(),
                c2.as_mut_ptr(),
                usize::MAX,
            );
        }
        assert_eq!((c0, c1, c2), (vec![9; 4], vec![9; 4], vec![9; 4])); // untouched
    }
}
