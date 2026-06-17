//! Render a raw through the darkroom preview path and save the displayed 8-bit
//! buffer to a PNG for visual confirmation (raw not committed):
//!
//!   cargo run -p darkroom-ui --example save_raw_png -- in.orf out.png

use darkroom_ui::preview::{apply_pipeline, PreviewParams};
use darkroom_ui::raw_preview;
use gtk4::gdk_pixbuf::{Colorspace, Pixbuf};
use gtk4::glib;

fn main() {
    let mut args = std::env::args().skip(1);
    let inp = args.next().expect("usage: save_raw_png <raw> <out.png>");
    let outp = args.next().expect("usage: save_raw_png <raw> <out.png>");

    let p = raw_preview::decode_raw_preview(&inp, 1024).expect("raw decode failed");
    println!("preview {}x{} nch={}", p.width, p.height, p.nch);

    // Render through the actual preview pipeline with the raw default params
    // (sigmoid tone-map ON), i.e. exactly what the darkroom view displays.
    let mut params = PreviewParams::default();
    params.sigmoid_on = true;
    let processed = apply_pipeline(
        &p.bytes,
        p.width as usize,
        p.height as usize,
        p.rowstride,
        p.nch,
        &params,
    );

    let bytes = glib::Bytes::from_owned(processed);
    let pb = Pixbuf::from_bytes(
        &bytes,
        Colorspace::Rgb,
        false, // no alpha (nch == 3)
        8,
        p.width,
        p.height,
        p.rowstride as i32,
    );
    pb.savev(&outp, "png", &[]).expect("png save failed");
    println!("wrote {outp}");
}
