//! Two numerical C&T samples against pinned darktable CPU output; this is not a full parity claim.
//! Provenance: darktable `97912518a5a9a9786aeee3c38116fbc999412ff8`, CPU Release,
//! USE_OPENCL=OFF, OpenMP one thread, 256x256 full ROI, ROI scale 1, identical
//! generated input/parameters. White-box CMocka test included contrastntexture.c
//! and called init/commit/ROI/process; C piece.iscale was 1 (full) and 2 (proxy2).
//! These samples do not establish behavioral parity, proxy-vs-resampled-original
//! parity, UI/export integration, or deployment readiness.

use c41_core::iop::contrastntexture::{ContrastTextureParams, prepare, process_into};
use c41_core::resource_budget::ResourceBudget;

const WIDTH: usize = 256;
const HEIGHT: usize = 256;
const FULL: &[u8] =
    include_bytes!("fixtures/contrastntexture/darktable-97912518-full-256x256.rgba32");
const PROXY2: &[u8] =
    include_bytes!("fixtures/contrastntexture/darktable-97912518-proxy2-256x256.rgba32");

fn compare(golden: &[u8], piece_scale: f32) {
    let params = ContrastTextureParams {
        gain_coarse: 0.7,
        gain_medium: 0.4,
        gain_fine: 0.2,
        detail_level: 5.0,
        halo_control: 0.0,
        noise_bias: 0.001,
        gain_shadows: 0.0,
        gain_highlights: 0.0,
    };
    let mut input = vec![0.0_f32; WIDTH * HEIGHT * 4];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let i = (y * WIDTH + x) * 4;
            let fx = x as f32 / 255.0;
            let fy = y as f32 / 255.0;
            let texture = ((x * 17 + y * 31 + x * y * 7) % 251) as f32 / 250.0;
            let v = 0.08 + 0.72 * fx + 0.1 * texture;
            input[i] = v;
            input[i + 1] = v * (0.7 + 0.3 * fy);
            input[i + 2] = 0.15 + 0.75 * (0.5 * fx + 0.5 * texture);
            input[i + 3] = 0.25 + 0.75 * ((x + y) % 257) as f32 / 256.0;
        }
    }
    assert_eq!(golden.len(), input.len() * 4);
    let prepared = prepare(params, 256.0, piece_scale, 1.0, WIDTH, HEIGHT).unwrap();
    let budget = ResourceBudget {
        cpu_available_bytes: u64::MAX,
        cpu_single_buffer_bytes: u64::MAX,
        mipmap_bytes: 0,
        opencl_available_bytes: 0,
    };
    let mut output = vec![0.0_f32; input.len()];
    process_into(
        &input,
        &mut output,
        WIDTH,
        HEIGHT,
        params,
        &prepared,
        0,
        0,
        budget,
    )
    .unwrap();
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut squared_error = 0.0_f64;
    let mut rgb_samples = 0usize;
    for (i, actual) in output.iter().copied().enumerate() {
        let offset = i * 4;
        let expected = f32::from_le_bytes(golden[offset..offset + 4].try_into().unwrap());
        if i % 4 == 3 {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "alpha pixel {}",
                i / 4
            );
        } else {
            let abs = (actual - expected).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / expected.abs().max(1.0e-8));
            squared_error += f64::from(abs) * f64::from(abs);
            rgb_samples += 1;
        }
    }
    let rms = (squared_error / rgb_samples as f64).sqrt();
    assert!(max_abs <= 1.0e-5, "max_abs={max_abs:.9e}");
    assert!(max_rel <= 1.0e-5, "max_rel={max_rel:.9e}");
    assert!(rms <= 1.0e-6, "rms={rms:.9e}, piece_scale={piece_scale}");
}

#[test]
fn matches_darktable_full_scale_sample() {
    compare(FULL, 1.0);
}

#[test]
fn matches_darktable_proxy2_scale_sample() {
    compare(PROXY2, 2.0);
}
