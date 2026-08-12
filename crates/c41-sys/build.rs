use std::{env, fs, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Default: copy the pre-generated bindings committed in src/bindings.rs.
    // Set DARKROOM_GENERATE_BINDINGS=1 to regenerate from headers via bindgen.
    // Also set LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu (or wherever libclang is).
    if env::var("DARKROOM_GENERATE_BINDINGS").as_deref() == Ok("1") {
        run_bindgen(&out_dir);
    } else {
        let src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/bindings.rs");
        fs::copy(&src, out_dir.join("bindings.rs"))
            .expect("failed to copy pre-generated src/bindings.rs");
    }

    println!("cargo:rerun-if-changed=src/bindings.rs");
    println!("cargo:rerun-if-env-changed=DARKROOM_GENERATE_BINDINGS");
}

fn run_bindgen(out_dir: &std::path::Path) {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/c41-sys  →  crates/  →  repo root
    let repo_root = manifest.ancestors().nth(2).unwrap().to_owned();
    let src_dir = repo_root.join("src");

    let bindings = bindgen::Builder::default()
        .header(src_dir.join("common/darktable.h").to_str().unwrap())
        .clang_arg(format!("-I{}", src_dir.display()))
        // Restrict output to types defined in *our* source tree only.
        .allowlist_file(format!("{}.*", src_dir.display()))
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed — set LIBCLANG_PATH to the directory containing libclang.so");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
