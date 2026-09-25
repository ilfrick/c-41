//! AI model infrastructure (u7a, parity audit 2.7 neural-restore leg) plus the
//! u7b RGB-denoise inference on top of it.
//!
//! `registry` finds model releases, `download` streams them with sha256
//! verification into the store, `package` unpacks and interprets the
//! `.dtmodel` manifest, and `infer` builds ONNX Runtime sessions on the
//! extracted payload and runs the tiled denoise. There is NO UI here — the
//! neural-restore panel lives in c41-ui and drives `infer`.
//!
//! Store layout (all paths derived here, never string-built by callers):
//!
//! * Downloaded archives live under `$XDG_DATA_HOME/c41/models/<asset-name>`
//!   (`models_dir`, `model_store_path`; `C41_MODELS_DIR` overrides wholesale
//!   for container experiments, mirroring the tiles crate's
//!   `C41_TILE_CACHE_DIR` seam). Asset names are sanitized — no separators,
//!   no `..` — so a hostile registry response cannot escape the store.
//! * Unpacked model directories sit beside the archive (same parent); the
//!   manifest is the extracted `config.json` read by `package`.
//!
//! Digest policy (mirrors `src/common/ai_models.c`): every download is
//! verified against the release asset's `digest` field (`sha256:<hex>`,
//! prefix stripped at parse). A file without a usable checksum is REFUSED,
//! never stored; a mismatch deletes the partial file and errors, so a bad
//! byte can never sit in the store looking complete. An already-present
//! file whose sha matches is skipped without touching the network.
//!
//! Execution-provider story: `ort` is linked with default features, which
//! is CPU only — no CUDA/TensorRT/CoreML/DirectML providers are enabled.
//! That is deliberate for u7a (deterministic everywhere, no GPU in CI or
//! in the shipping image). GPU providers are a later increment and will
//! need Docker + CI changes; the `ort_version` smoke fn below is the
//! linkage proof until then.
//!
//! What u7b builds on this: `registry` gives the asset list for the panel,
//! `download` gives the verified archive path, `package` gives the manifest
//! (tile size per variant stem) plus the extracted `.onnx` payload path,
//! and the `ort` linkage proven here becomes session construction.

pub mod download;
pub mod infer;
pub mod package;
pub mod registry;

/// Proves the ONNX Runtime linkage: returns the ORT release line this
/// `ort` build targets, e.g. `"1.22.x"` from `ort::MINOR_VERSION`.
///
/// This is the ONLY `ort` item referenced in the tree until u7b, kept
/// minimal on purpose: a compile-time const read, no global environment
/// is created (ort's own docs forbid library crates from calling
/// `ort::init()` — that belongs to the downstream application, i.e. u7b
/// wiring). Verified against the vendored ort 2.0.0-rc.13 sources during
/// u7a (no `ort::version()` fn exists on the 2.x line; the Docker gate
/// re-verifies on lockfile regen).
pub fn ort_version() -> String {
    format!("1.{}.x", ort::MINOR_VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ort_linkage_reports_a_version() {
        let v = ort_version();
        assert!(
            v.starts_with("1.") && v.ends_with(".x"),
            "expected an ORT release line like 1.22.x, got {v:?}"
        );
    }
}
