//! `.dtmodel` package handling (u7a): zip listing, manifest parse, payload extract.
//!
//! Source-evidence schema: the extracted model directory holds a
//! `config.json` manifest whose `attributes` object carries per-stem
//! variant knobs as NESTED objects, e.g. `attributes.model_bayer` holding
//! `input_sizes`, `input_kind` (`src/common/ai/restore.c:186-259`, looked
//! up by dotted-path descent in `src/ai/backend_common.c:538-578` —
//! intermediate segments must be objects). The tile size is the FIRST entry
//! of `<stem>.input_sizes`, falling back to the top-level `input_sizes` for
//! single-model packages (`_resolve_tile_size`, restore.c:186-205). The
//! payload file is `<stem>.onnx` (`_load`, restore.c:207-234). Attribute
//! string lookups (`<stem>.input_kind`, `.input_colorspace`, `.wb_norm`,
//! `.output_scale`, `.bayer_orientation`, `.edge_pad`) go through the same
//! dotted paths.
//!
//! `parse_manifest` flattens nested objects to dotted paths on load, so the
//! map holds `model_bayer.input_sizes` either way a manifest spells it;
//! literal dotted keys in legacy manifests keep working, with the nested
//! spelling winning on collision (matching what the C descent would read).
//!
//! `.dtmodel` files are ZIP archives (extracted with libarchive upstream,
//! ai_models.c:1114+; here the pure-Rust `zip` crate, default features so
//! deflate payloads decode). Extraction guards mirror the C: entries with
//! `..` or absolute paths are skipped, never written outside `dest_dir`.
//!
//! u7b builds on this: `variant_tile_size` feeds the tiler, and
//! `extract_onnx` stages the exact payload the ORT session will load.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

/// Manifest filename inside an extracted model directory.
pub const MANIFEST_FILENAME: &str = "config.json";

/// Parsed `config.json`: display name plus the `attributes` map, flattened
/// to `<stem>.<field>` dotted paths (nested objects are descended exactly
/// like the C `_attribute_node` dotted-path lookup; a literal dotted key in
/// a legacy manifest also lands, with the nested spelling winning on
/// collision). Values kept as JSON.
#[derive(Clone, Debug, Default)]
pub struct PackageManifest {
    pub name: Option<String>,
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Package failures.
#[derive(Debug, PartialEq, Eq)]
pub enum PackageError {
    /// File IO failure (read manifest, write payload, ...).
    Io(String),
    /// The bytes are not a zip archive, or an entry is corrupt.
    BadArchive(String),
    /// Manifest JSON missing or wrong shape.
    BadManifest(String),
    /// Requested `<stem>.onnx` not present in the archive.
    OnnxNotFound(String),
}

impl std::fmt::Display for PackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "model package IO failed: {s}"),
            Self::BadArchive(s) => write!(f, "model archive unreadable: {s}"),
            Self::BadManifest(s) => write!(f, "model manifest unparseable: {s}"),
            Self::OnnxNotFound(stem) => write!(f, "archive has no {stem}.onnx payload"),
        }
    }
}

impl std::error::Error for PackageError {}

impl From<std::io::Error> for PackageError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// Parse a `config.json` body. `attributes` is optional (absent means no
/// variant knobs — legacy single-model shape); non-object `attributes`
/// is a manifest error, not a silent default. Nested objects are flattened
/// to dotted paths (`{"model_bayer": {"input_sizes": [...]}}` becomes
/// `model_bayer.input_sizes`), mirroring the C dotted-path descent; a
/// literal dotted key also lands, with the nested spelling winning.
pub fn parse_manifest(json: &str) -> Result<PackageManifest, PackageError> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| PackageError::BadManifest(e.to_string()))?;
    let obj = root
        .as_object()
        .ok_or_else(|| PackageError::BadManifest("top level is not an object".to_string()))?;
    let name = obj
        .get("name")
        .and_then(|n| n.as_str())
        .map(str::to_string);
    let mut attributes = HashMap::new();
    if let Some(attrs) = obj.get("attributes") {
        let map = attrs
            .as_object()
            .ok_or_else(|| PackageError::BadManifest("attributes is not an object".to_string()))?;
        for (k, v) in map {
            if v.is_object() {
                flatten_attributes(k, v, &mut attributes);
            } else {
                // Legacy literal dotted key — inserted first so a nested
                // spelling of the same path overwrites it below.
                attributes.insert(k.clone(), v.clone());
            }
        }
        // Second pass so nested paths win over same-spelled literals.
        for (k, v) in map {
            if v.is_object() {
                flatten_attributes(k, v, &mut attributes);
            }
        }
    }
    Ok(PackageManifest { name, attributes })
}

/// Recursively flatten a nested attributes object to dotted paths:
/// `flatten("model_bayer", {"input_sizes": [...]})` inserts
/// `model_bayer.input_sizes`. Arrays are leaves (the C descends through
/// objects only).
fn flatten_attributes(
    prefix: &str,
    value: &serde_json::Value,
    out: &mut HashMap<String, serde_json::Value>,
) {
    let Some(map) = value.as_object() else {
        return;
    };
    for (k, v) in map {
        let key = format!("{prefix}.{k}");
        if v.is_object() {
            flatten_attributes(&key, v, out);
        } else {
            out.insert(key, v.clone());
        }
    }
}

/// Read `<dir>/config.json` into a manifest.
pub fn load_manifest_from_dir(dir: &Path) -> Result<PackageManifest, PackageError> {
    let text = std::fs::read_to_string(dir.join(MANIFEST_FILENAME))?;
    parse_manifest(&text)
}

/// Dotted attribute lookup: `variant_string(m, "model_bayer", "input_kind")`
/// reads `attributes["model_bayer.input_kind"]` as a string.
pub fn variant_string(manifest: &PackageManifest, stem: &str, field: &str) -> Option<String> {
    manifest
        .attributes
        .get(&format!("{stem}.{field}"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// Static input dim for a variant stem: first entry of
/// `<stem>.input_sizes`, else first entry of top-level `input_sizes`
/// (restore.c `_resolve_tile_size`). Non-integer entries do not count.
pub fn variant_tile_size(manifest: &PackageManifest, stem: &str) -> Option<i64> {
    for key in [format!("{stem}.input_sizes"), "input_sizes".to_string()] {
        if let Some(first) = manifest
            .attributes
            .get(&key)
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|n| n.as_i64())
        {
            if first > 0 {
                return Some(first);
            }
        }
    }
    None
}

/// Payload filename for a variant stem: `<stem>.onnx` (restore.c `_load`).
pub fn variant_onnx_name(stem: &str) -> String {
    format!("{stem}.onnx")
}

/// Entry names of a `.dtmodel` archive, in archive order.
pub fn list_zip_entries(zip_bytes: &[u8]) -> Result<Vec<String>, PackageError> {
    let archive = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes))
        .map_err(|e| PackageError::BadArchive(e.to_string()))?;
    Ok(archive.file_names().map(str::to_string).collect())
}

/// True when an archive entry path is safe to extract under `dest_dir`:
/// relative, with no `..` (the C `_extract_zip` skips `..` entries and
/// re-checks the canonical prefix; the component walk here is the same
/// guarantee in Rust terms).
fn entry_is_safe(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let path = Path::new(name);
    if path.is_absolute() {
        return false;
    }
    !path
        .components()
        .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_) | Component::RootDir))
}

/// Write `contents` atomically (tmp-in-same-dir + rename, tiles.rs house
/// pattern) so a crashed unpack never leaves a half-written payload.
fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), PackageError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Extract every safe entry of a `.dtmodel` file into `dest_dir`.
/// Unsafe entries (`..`, absolute) are SKIPPED like the C does, never
/// written; directory entries materialize as directories. Returns the
/// paths actually written.
pub fn unpack_dtmodel(zip_path: &Path, dest_dir: &Path) -> Result<Vec<PathBuf>, PackageError> {
    let bytes = std::fs::read(zip_path)?;
    unpack_dtmodel_bytes(&bytes, dest_dir)
}

/// Same as `unpack_dtmodel` but from in-memory bytes (lets tests skip the
/// filesystem round-trip for the archive itself).
pub fn unpack_dtmodel_bytes(
    zip_bytes: &[u8],
    dest_dir: &Path,
) -> Result<Vec<PathBuf>, PackageError> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes))
        .map_err(|e| PackageError::BadArchive(e.to_string()))?;
    std::fs::create_dir_all(dest_dir)?;
    let mut written = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| PackageError::BadArchive(e.to_string()))?;
        let name = entry.name().to_string();
        if !entry_is_safe(&name) {
            continue;
        }
        let out_path = dest_dir.join(&name);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)?;
            continue;
        }
        // No preallocation from the header size: a bogus uncompressed_size
        // would abort/OOM before a byte is read. `read_to_end` grows as
        // actual bytes arrive.
        let mut contents = Vec::new();
        entry.read_to_end(&mut contents)?;
        atomic_write(&out_path, &contents)?;
        written.push(out_path);
    }
    Ok(written)
}

/// Extract just the `<stem>.onnx` payload for one variant into `out_path`
/// (atomic write). Errors `OnnxNotFound` when the stem is absent — the
/// caller picks the stem, so a missing payload is a packaging error, not
/// a silent fallback (restore.h contract).
pub fn extract_onnx(
    zip_path: &Path,
    stem: &str,
    out_path: &Path,
) -> Result<PathBuf, PackageError> {
    let bytes = std::fs::read(zip_path)?;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&bytes))
        .map_err(|e| PackageError::BadArchive(e.to_string()))?;
    let want = variant_onnx_name(stem);
    let mut entry = archive
        .by_name(&want)
        .map_err(|_| PackageError::OnnxNotFound(stem.to_string()))?;
    // See above: no header-size preallocation on untrusted input.
    let mut contents = Vec::new();
    entry.read_to_end(&mut contents)?;
    drop(entry);
    atomic_write(out_path, &contents)?;
    Ok(out_path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // Synthetic .dtmodel built in-test with the zip crate's own writer
    // (Stored = no compression, so no codec features are involved):
    // config.json with NESTED stem objects (the real-world shape the C
    // dotted-path descent reads) + two payloads + one traversal probe +
    // one junk file.
    const CONFIG_JSON: &str = r#"{
        "name": "test denoise",
        "attributes": {
            "model_bayer": {
                "input_sizes": [512],
                "input_kind": "bayer_v1",
                "wb_norm": "daylight"
            },
            "input_sizes": [256]
        }
    }"#;
    const BAYER_ONNX: &[u8] = b"fake-onnx-payload-bayer";
    const LINEAR_ONNX: &[u8] = b"fake-onnx-payload-linear";

    fn synthetic_dtmodel() -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            w.start_file(MANIFEST_FILENAME, opts).unwrap();
            w.write_all(CONFIG_JSON.as_bytes()).unwrap();
            w.start_file("model_bayer.onnx", opts).unwrap();
            w.write_all(BAYER_ONNX).unwrap();
            w.start_file("model_linear.onnx", opts).unwrap();
            w.write_all(LINEAR_ONNX).unwrap();
            w.start_file("notes.txt", opts).unwrap();
            w.write_all(b"junk").unwrap();
            w.start_file("../evil.txt", opts).unwrap();
            w.write_all(b"escape").unwrap();
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn fresh(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "c41_ai_pkg_{}_{}_{}",
                std::process::id(),
                name,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn manifest_parses_nested_stem_objects() {
        // Real-world shape: nested objects read by dotted-path descent.
        // This is the reviewer's B1 failing input verbatim.
        let nested = parse_manifest(
            r#"{"attributes": {"model_bayer": {"input_sizes": [512]}}}"#,
        )
        .unwrap();
        assert_eq!(variant_tile_size(&nested, "model_bayer"), Some(512));
        let m = parse_manifest(CONFIG_JSON).unwrap();
        assert_eq!(m.name.as_deref(), Some("test denoise"));
        assert_eq!(
            variant_string(&m, "model_bayer", "input_kind").as_deref(),
            Some("bayer_v1")
        );
        assert_eq!(
            variant_string(&m, "model_bayer", "wb_norm").as_deref(),
            Some("daylight")
        );
        assert_eq!(variant_string(&m, "model_bayer", "missing"), None);
        assert_eq!(variant_string(&m, "model_nope", "input_kind"), None);
    }

    #[test]
    fn manifest_legacy_flat_keys_still_land_with_nested_winning() {
        // A legacy manifest spelling dotted keys literally keeps working;
        // where both spellings exist the nested one wins (what C reads).
        let m = parse_manifest(
            r#"{"attributes": {
                "model_bayer.input_sizes": [111],
                "model_bayer": {"input_sizes": [512]}
            }}"#,
        )
        .unwrap();
        assert_eq!(variant_tile_size(&m, "model_bayer"), Some(512));
        let flat_only = parse_manifest(
            r#"{"attributes": {"model_bayer.input_sizes": [111]}}"#,
        )
        .unwrap();
        assert_eq!(variant_tile_size(&flat_only, "model_bayer"), Some(111));
    }

    #[test]
    fn tile_size_prefers_stem_then_top_level() {
        let m = parse_manifest(CONFIG_JSON).unwrap();
        assert_eq!(variant_tile_size(&m, "model_bayer"), Some(512));
        assert_eq!(
            variant_tile_size(&m, "model_other"),
            Some(256),
            "single-model fallback"
        );
        let bare = parse_manifest(r#"{"name":"x"}"#).unwrap();
        assert_eq!(variant_tile_size(&bare, "model_bayer"), None);
        let bad = parse_manifest(r#"{"attributes":{"input_sizes":["wide"]}}"#).unwrap();
        assert_eq!(variant_tile_size(&bad, "s"), None);
        let zero = parse_manifest(r#"{"attributes":{"input_sizes":[0]}}"#).unwrap();
        assert_eq!(variant_tile_size(&zero, "s"), None);
        assert_eq!(variant_onnx_name("model_bayer"), "model_bayer.onnx");
    }

    #[test]
    fn bad_manifests_error() {
        assert!(parse_manifest("not json").is_err());
        assert!(parse_manifest("[1,2]").is_err());
        assert!(parse_manifest(r#"{"attributes":[1]}"#).is_err());
    }

    #[test]
    fn zip_listing_sees_all_entries() {
        let names = list_zip_entries(&synthetic_dtmodel()).unwrap();
        assert!(names.contains(&"config.json".to_string()));
        assert!(names.contains(&"model_bayer.onnx".to_string()));
        assert!(list_zip_entries(b"not a zip").is_err());
    }

    #[test]
    fn unpack_writes_safe_entries_and_skips_traversal() {
        let dir = TempDir::fresh("unpack");
        let bytes = synthetic_dtmodel();
        let written = unpack_dtmodel_bytes(&bytes, &dir.path).unwrap();
        assert_eq!(
            std::fs::read(dir.path.join("config.json")).unwrap(),
            CONFIG_JSON.as_bytes()
        );
        assert_eq!(
            std::fs::read(dir.path.join("model_bayer.onnx")).unwrap(),
            BAYER_ONNX
        );
        assert!(!dir.path.join("evil.txt").exists());
        assert!(
            !dir.path
                .parent()
                .unwrap()
                .join("evil.txt")
                .exists(),
            "no escape to the parent dir"
        );
        assert!(written.iter().all(|p| p.starts_with(&dir.path)));
        assert_eq!(written.len(), 4, "evil skipped, rest written");
    }

    #[test]
    fn on_demand_onnx_extract_matches_bytes() {
        let dir = TempDir::fresh("onnx");
        let zip_path = dir.path.join("m.dtmodel");
        std::fs::write(&zip_path, synthetic_dtmodel()).unwrap();
        let out = dir.path.join("staged.onnx");
        let got = extract_onnx(&zip_path, "model_bayer", &out).unwrap();
        assert_eq!(got, out);
        assert_eq!(std::fs::read(&out).unwrap(), BAYER_ONNX);
        assert!(matches!(
            extract_onnx(&zip_path, "model_xtrans", &dir.path.join("x.onnx")),
            Err(PackageError::OnnxNotFound(_))
        ));
    }

    #[test]
    fn manifest_loads_from_extracted_dir() {
        let dir = TempDir::fresh("dir");
        std::fs::write(dir.path.join(MANIFEST_FILENAME), CONFIG_JSON).unwrap();
        let m = load_manifest_from_dir(&dir.path).unwrap();
        assert_eq!(variant_tile_size(&m, "model_bayer"), Some(512));
        assert!(load_manifest_from_dir(&dir.path.join("nope")).is_err());
    }
}
