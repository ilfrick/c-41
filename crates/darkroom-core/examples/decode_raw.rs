//! Decode a camera raw and print pipeline-ready stats — a dev/verification tool
//! for the `rawimage` front end (raw is not committed; pass a local path):
//!
//!   cargo run -p darkroom-core --example decode_raw -- path/to/file.orf

use darkroom_core::rawimage;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: decode_raw <raw-file>");

    let img = rawimage::load(&path).expect("decode failed");
    println!(
        "decoded: {}x{}  cfa={:?}  wb={:?}  orientation(t,fx,fy)={:?}",
        img.width, img.height, img.cfa, img.wb, img.orientation
    );
    let mosaic_max = img.mosaic.iter().copied().fold(0.0f32, f32::max);
    println!(
        "mosaic: {} photosites, normalised max {:.4}",
        img.mosaic.len(),
        mosaic_max
    );

    let (w, h, rgba) = img.to_linear_rgba();
    println!("linear RGBA: {w}x{h} = {} floats", rgba.len());

    let ci = ((h / 2) * w + w / 2) * 4;
    println!(
        "centre px RGBA: [{:.4}, {:.4}, {:.4}, {:.1}]",
        rgba[ci],
        rgba[ci + 1],
        rgba[ci + 2],
        rgba[ci + 3]
    );

    let (mn, mx) = rgba
        .iter()
        .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    println!("RGBA range: [{mn:.4}, {mx:.4}]");
}
