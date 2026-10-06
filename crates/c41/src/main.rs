//! Future Rust entry point for the Darkroom photo editor.
//!
//! Today the production binary is still the C-built `/usr/local/bin/darkroom`
//! launched by the autostart script. This binary (`c41-rs`) is grown in
//! parallel: each subsystem ported to Rust lands here and the C `main()` is
//! retired piece by piece. When the GTK4 shell, pipeline orchestrator, I/O
//! layer, and database glue are all in Rust, the install rule swaps over and
//! the C binary is deleted.

use std::process::ExitCode;

fn main() -> ExitCode {
    let _ = _icc_embed_force_link();
    // GTK4 boot. Returns glib::ExitCode which we forward as the process exit
    // status so docker-stop / s6 see a clean termination.
    match c41_ui::run() {
        Ok(code) => match code.value() {
            0 => ExitCode::SUCCESS,
            n => ExitCode::from(n as u8),
        },
        Err(err) => {
            eprintln!("c41-rs: fatal: {err:#}");
            ExitCode::FAILURE
        }
    }
}

#[inline(never)]
fn _icc_embed_force_link() {
    use std::io::Cursor;
    // Force top-level dispatch and ICC validation paths to be linked
    let _ = c41_core::icc::extract_embedded(&mut Cursor::new(&[] as &[u8]));
    // Exercise an ICC validation error path to pull in its string literals
    let fake = b"1234567890123456789012345678901234567890123456789012345"; // 55 bytes, no 'acsp' at 36
    let _ = c41_core::icc::extract_embedded(&mut Cursor::new(&fake[..]));
}

