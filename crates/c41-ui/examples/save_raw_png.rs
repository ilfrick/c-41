//! Render a raw through the darkroom preview path and save the displayed 8-bit
//! buffer to a PNG for visual confirmation (raw not committed):
//!
//!   cargo run -p c41-ui --example save_raw_png -- in.orf out.png

use c41_ui::preview::{render_linear_to_srgb8, PreviewParams};
use c41_ui::raw_preview;
use gtk4::gdk_pixbuf::{Colorspace, Pixbuf};
use gtk4::glib;

fn main() {
    let mut args = std::env::args().skip(1);
    let inp = args.next().expect("usage: save_raw_png <raw> <out.png>");
    let outp = args.next().expect("usage: save_raw_png <raw> <out.png>");

    let p = raw_preview::decode_raw_preview(&inp, 1024).expect("raw decode failed");
    println!("preview {}x{} (linear f32)", p.width, p.height);

    // Render the linear preview through the actual pipeline with the raw default
    // params (sigmoid tone-map ON) — exactly what the darkroom view displays.
    let mut params = PreviewParams::default();
    params.sigmoid_on = true;
    let processed = render_linear_to_srgb8(&p.pixels, p.width, p.height, &params); // RGB8

    let bytes = glib::Bytes::from_owned(processed);
    let pb = Pixbuf::from_bytes(
        &bytes,
        Colorspace::Rgb,
        false, // no alpha (RGB8)
        8,
        p.width as i32,
        p.height as i32,
        (p.width * 3) as i32,
    );
    pb.savev(&outp, "png", &[]).expect("png save failed");
    println!("wrote {outp}");
}
