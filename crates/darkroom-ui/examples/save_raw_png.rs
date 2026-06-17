//! Render a raw through the darkroom preview path and save the displayed 8-bit
//! buffer to a PNG for visual confirmation (raw not committed):
//!
//!   cargo run -p darkroom-ui --example save_raw_png -- in.orf out.png

use darkroom_ui::raw_preview;
use gtk4::gdk_pixbuf::{Colorspace, Pixbuf};
use gtk4::glib;

fn main() {
    let mut args = std::env::args().skip(1);
    let inp = args.next().expect("usage: save_raw_png <raw> <out.png>");
    let outp = args.next().expect("usage: save_raw_png <raw> <out.png>");

    let p = raw_preview::decode_raw_preview(&inp, 1024).expect("raw decode failed");
    println!("preview {}x{} nch={}", p.width, p.height, p.nch);

    let bytes = glib::Bytes::from_owned(p.bytes);
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
