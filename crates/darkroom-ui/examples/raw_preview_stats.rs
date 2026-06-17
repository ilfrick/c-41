//! Verify the raw→8-bit-sRGB preview path on a real raw (not committed):
//!
//!   cargo run -p darkroom-ui --example raw_preview_stats -- path/to/file.orf

use darkroom_ui::raw_preview;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: raw_preview_stats <raw-file>");
    println!("is_raw_path: {}", raw_preview::is_raw_path(&path));

    let p = raw_preview::decode_raw_preview(&path, 2048).expect("raw preview failed");
    println!("preview: {}x{}  ({} linear f32)", p.width, p.height, p.pixels.len());
    let (mn, mx) = p
        .pixels
        .iter()
        .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    println!("linear value range: [{mn:.4}, {mx:.4}]  (>1.0 ⇒ unclipped highlights)");
}
