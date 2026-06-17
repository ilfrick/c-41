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
    println!(
        "preview: {}x{}  nch={}  rowstride={}  bytes={}",
        p.width, p.height, p.nch, p.rowstride, p.bytes.len()
    );
    let (mn, mx) = p
        .bytes
        .iter()
        .fold((255u8, 0u8), |(mn, mx), &b| (mn.min(b), mx.max(b)));
    let mean: f64 = p.bytes.iter().map(|&b| b as f64).sum::<f64>() / p.bytes.len() as f64;
    println!("8-bit byte range: [{mn}, {mx}], mean {mean:.1}");
}
