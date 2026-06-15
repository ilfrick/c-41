//! Minimal live preview: applies a single migrated `darkroom-core` IOP to the
//! decoded 8-bit preview image so the darkroom view shows *processed* output
//! (not just the file). This is the first UI↔core processing seam — a
//! stepping-stone toward a full Rust pixelpipe (RUST_MIGRATION_PLAN.md Phase 3
//! milestone 2). For now the one IOP is exposure.

use darkroom_core::iop::exposure::process_pixels;

/// Apply an exposure (EV) adjustment to an 8-bit interleaved image buffer,
/// preserving layout (rowstride) and any alpha channel. The colour channels
/// (0..min(3,nch)) run through the migrated `exposure::process_pixels`
/// (`out = (in - black) * scale`, with black = 0 and scale = 2^ev), on values
/// normalised to [0,1]; alpha (channel 3, if present) is copied unchanged.
pub fn apply_exposure(
    base: &[u8],
    width: usize,
    height: usize,
    rowstride: usize,
    nch: usize,
    ev: f32,
) -> Vec<u8> {
    let scale = 2.0f32.powf(ev);
    let colour = nch.min(3);

    // gather colour samples → f32 [0,1]
    let mut inp = Vec::with_capacity(width * height * colour);
    for y in 0..height {
        let row = y * rowstride;
        for x in 0..width {
            let p = row + x * nch;
            for c in 0..colour {
                inp.push(base[p + c] as f32 / 255.0);
            }
        }
    }

    // process through the migrated core IOP
    let mut outp = vec![0.0f32; inp.len()];
    process_pixels(&inp, &mut outp, 0.0, scale);

    // write back, preserving alpha and rowstride padding
    let mut out = base.to_vec();
    let mut k = 0usize;
    for y in 0..height {
        let row = y * rowstride;
        for x in 0..width {
            let p = row + x * nch;
            for c in 0..colour {
                out[p + c] = (outp[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                k += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ev_zero_is_identity() {
        // 2x1 RGB image, rowstride==width*nch, ev=0 ⇒ scale 1 ⇒ unchanged
        let base = vec![10u8, 20, 30, 200, 100, 50];
        let out = apply_exposure(&base, 2, 1, 6, 3, 0.0);
        assert_eq!(out, base);
    }

    #[test]
    fn ev_plus_one_doubles_and_clamps_keeps_alpha() {
        // RGBA: ev=+1 ⇒ scale 2; colour doubles (clamped at 255), alpha kept.
        // 50/255*2 = 0.392 → *255 ≈ 100; 200/255*2 clamps to 255.
        let base = vec![50u8, 200, 25, 111];
        let out = apply_exposure(&base, 1, 1, 4, 4, 1.0);
        assert_eq!(out[0], 100);
        assert_eq!(out[1], 255); // clamped
        assert_eq!(out[2], 50);
        assert_eq!(out[3], 111); // alpha unchanged
    }

    #[test]
    fn respects_rowstride_padding() {
        // 1x2 RGB with 2 padding bytes per row; padding must be preserved.
        let base = vec![10u8, 20, 30, 0xAA, 0xBB, 40, 50, 60, 0xCC, 0xDD];
        let out = apply_exposure(&base, 1, 2, 5, 3, 0.0);
        assert_eq!(out, base); // ev=0 identity, padding intact
    }
}
