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
