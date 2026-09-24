//! Model release registry (u7a): GitHub releases API fetch + asset parse.
//!
//! Mirrors `src/common/ai_models.c`: the release object comes from
//! `GET https://api.github.com/repos/{repo}/releases/tags/{tag}` and each
//! entry of its `assets` array carries `name`, `browser_download_url`,
//! `size`, and `digest` (`sha256:<hex>`). The C side builds the download
//! URL as `https://github.com/{repo}/releases/download/{tag}/{asset}`
//! (ai_models.c:1468) rather than trusting the API URL; we prefer the
//! API's `browser_download_url` when present and fall back to the
//! constructed form otherwise.
//!
//! Task filtering is by filename prefix, matching the C task strings:
//! `denoise-`, `rawdenoise-`, `upscale-`, `mask-`.
//!
//! Network lives only in `fetch_release_assets` (ureq with the house
//! `c41-darkroom/<version>` UA, same call shape as `tiles.rs`). Every
//! parse fn is pure and headless-tested with inline JSON fixtures — no
//! test hits github.com.

use std::time::Duration;

/// GitHub owner/repo serving the model releases.
pub const DEFAULT_MODEL_REPO: &str = "darktable-org/darktable-ai";
/// Pinned stable release tag carrying the u7a model set.
pub const DEFAULT_RELEASE_TAG: &str = "release-5.6.0";
/// Filename prefixes selecting assets per task (`*_TASK_PREFIX` filters).
pub const TASK_PREFIXES: &[&str] = &["denoise-", "rawdenoise-", "upscale-", "mask-"];
/// Per-request API timeout: small JSON bodies, same class as tiles (15 s).
pub const REGISTRY_TIMEOUT_SECS: u64 = 15;

/// One release asset: a downloadable `.dtmodel` (or sidecar file).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelAsset {
    /// Asset filename, e.g. `denoise-nind.dtmodel`.
    pub name: String,
    /// Where to fetch the bytes (API URL, or the constructed fallback).
    pub download_url: String,
    /// `size` field of the asset record, when the API reports one.
    pub size: Option<u64>,
    /// Raw `digest` field of the asset record, e.g. `sha256:<hex>`.
    /// `None` when the record carries no digest — such an asset must
    /// never be downloaded (see `sha256_hex` + the download crate).
    pub digest: Option<String>,
}

impl ModelAsset {
    /// The verified hex sha256, or `None` when the record has no usable
    /// `sha256:<64-hex>` digest. Strict on purpose: wrong prefix or a
    /// non-hex/short tail yields `None`, which the downloader refuses.
    pub fn sha256_hex(&self) -> Option<&str> {
        let d = self.digest.as_deref()?;
        let hex = d.strip_prefix("sha256:")?;
        if hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            Some(hex)
        } else {
            None
        }
    }
}

/// Registry failures. Network and JSON errors collapse to message
/// strings so callers never match on library error shapes.
#[derive(Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// HTTP transport failure (DNS, connect, timeout, non-2xx status).
    Network(String),
    /// Body is not JSON, or not the expected shape.
    Parse(String),
    /// A required field is absent (`assets`, asset `name`, ...).
    MissingField(&'static str),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(s) => write!(f, "registry request failed: {s}"),
            Self::Parse(s) => write!(f, "registry response unparseable: {s}"),
            Self::MissingField(k) => write!(f, "registry response missing {k:?}"),
        }
    }
}

impl std::error::Error for RegistryError {}

impl From<ureq::Error> for RegistryError {
    fn from(e: ureq::Error) -> Self {
        Self::Network(e.to_string())
    }
}

impl From<serde_json::Error> for RegistryError {
    fn from(e: serde_json::Error) -> Self {
        Self::Parse(e.to_string())
    }
}

/// The `User-Agent` every registry request carries: identifies the
/// application instead of reading as a scraper (tiles.rs precedent).
pub fn registry_user_agent() -> String {
    format!("c41-darkroom/{}", env!("CARGO_PKG_VERSION"))
}

/// API URL for one release tag of a repo.
pub fn release_api_url(repo: &str, tag: &str) -> String {
    format!("https://api.github.com/repos/{repo}/releases/tags/{tag}")
}

/// Constructed download URL fallback (ai_models.c:1468-1472 shape).
pub fn constructed_download_url(repo: &str, tag: &str, asset: &str) -> String {
    format!("https://github.com/{repo}/releases/download/{tag}/{asset}")
}

/// Parse the assets of a `releases/tags/{tag}` response body.
///
/// Entries missing `name` are skipped (the C iterates and matches by
/// name, ignoring anything else); entries missing
/// `browser_download_url` fall back to the constructed URL so a
/// thin-but-named record still resolves. `size` and `digest` stay
/// optional here — the downloader, not the listing, enforces the
/// checksum requirement.
pub fn parse_release_assets(
    json: &str,
    repo: &str,
    tag: &str,
) -> Result<Vec<ModelAsset>, RegistryError> {
    let root: serde_json::Value = serde_json::from_str(json)?;
    let assets = root
        .get("assets")
        .and_then(|a| a.as_array())
        .ok_or(RegistryError::MissingField("assets"))?;
    let mut out = Vec::with_capacity(assets.len());
    for a in assets {
        let Some(name) = a.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let download_url = a
            .get("browser_download_url")
            .and_then(|u| u.as_str())
            .filter(|u| !u.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| constructed_download_url(repo, tag, name));
        let size = a.get("size").and_then(|s| s.as_u64());
        let digest = a
            .get("digest")
            .and_then(|d| d.as_str())
            .filter(|d| !d.is_empty())
            .map(str::to_string);
        out.push(ModelAsset {
            name: name.to_string(),
            download_url,
            size,
            digest,
        });
    }
    Ok(out)
}

/// Fetch and parse the asset list for one repo tag (the only network fn).
/// `repo`/`tag` default to `DEFAULT_MODEL_REPO`/`DEFAULT_RELEASE_TAG`;
/// pass explicit values to override (nightlies, forks).
pub fn fetch_release_assets(repo: &str, tag: &str) -> Result<Vec<ModelAsset>, RegistryError> {
    fetch_release_assets_from(&release_api_url(repo, tag), repo, tag)
}

/// Same as `fetch_release_assets` but against an explicit URL — the seam
/// that lets future tests point at a stub without touching globals.
pub fn fetch_release_assets_from(
    url: &str,
    repo: &str,
    tag: &str,
) -> Result<Vec<ModelAsset>, RegistryError> {
    let resp = ureq::get(url)
        .header("User-Agent", registry_user_agent())
        .header("Accept", "application/vnd.github+json")
        .config()
        .timeout_global(Some(Duration::from_secs(REGISTRY_TIMEOUT_SECS)))
        .build()
        .call()?;
    // `read_to_vec` (not `read_to_string`) is the proven Body method in
    // the vendored ureq 3.4.2 — see tiles.rs `tile_http_get`.
    let mut body = resp.into_body();
    let bytes = body.read_to_vec().map_err(RegistryError::from)?;
    let text = String::from_utf8(bytes)
        .map_err(|e| RegistryError::Parse(format!("response is not UTF-8: {e}")))?;
    parse_release_assets(&text, repo, tag)
}

/// Assets whose filename starts with `prefix` (one of `TASK_PREFIXES`).
pub fn assets_for_task<'a>(assets: &'a [ModelAsset], prefix: &str) -> Vec<&'a ModelAsset> {
    assets
        .iter()
        .filter(|a| a.name.starts_with(prefix))
        .collect()
}

/// First asset with exactly this filename.
pub fn find_asset<'a>(assets: &'a [ModelAsset], name: &str) -> Option<&'a ModelAsset> {
    assets.iter().find(|a| a.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small captured-shape fixture: real field names, real digest shape,
    // one asset per task prefix plus a digest-less sidecar.
    const FIXTURE: &str = r#"{
        "tag_name": "release-5.6.0",
        "assets": [
            {"name": "denoise-nind.dtmodel",
             "browser_download_url": "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/denoise-nind.dtmodel",
             "size": 57671680,
             "digest": "sha256:825b3657cbb5193a67432a2f0b44ab86531cc7337f89e8e6c17d93db9665708a"},
            {"name": "rawdenoise-nind.dtmodel",
             "browser_download_url": "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/rawdenoise-nind.dtmodel",
             "size": 59768832,
             "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"},
            {"name": "upscale-realplksr.dtmodel",
             "browser_download_url": "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/upscale-realplksr.dtmodel",
             "size": 57671680,
             "digest": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"},
            {"name": "mask-segnext.dtmodel",
             "size": 12345,
             "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
            {"name": "versions.json",
             "browser_download_url": "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/versions.json",
             "size": 512},
            {"size": 99}
        ]
    }"#;

    fn fixture() -> Vec<ModelAsset> {
        parse_release_assets(FIXTURE, DEFAULT_MODEL_REPO, DEFAULT_RELEASE_TAG).unwrap()
    }

    #[test]
    fn parses_names_urls_sizes_and_strips_digest() {
        let assets = fixture();
        assert_eq!(assets.len(), 5, "nameless entry skipped, rest kept");
        let d = find_asset(&assets, "denoise-nind.dtmodel").unwrap();
        assert_eq!(
            d.download_url,
            "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/denoise-nind.dtmodel"
        );
        assert_eq!(d.size, Some(57671680));
        assert_eq!(
            d.sha256_hex(),
            Some("825b3657cbb5193a67432a2f0b44ab86531cc7337f89e8e6c17d93db9665708a")
        );
    }

    #[test]
    fn missing_url_falls_back_to_constructed_form() {
        let assets = fixture();
        let m = find_asset(&assets, "mask-segnext.dtmodel").unwrap();
        assert_eq!(
            m.download_url,
            constructed_download_url(DEFAULT_MODEL_REPO, DEFAULT_RELEASE_TAG, "mask-segnext.dtmodel")
        );
    }

    #[test]
    fn digestless_asset_reports_no_sha() {
        let assets = fixture();
        let v = find_asset(&assets, "versions.json").unwrap();
        assert_eq!(v.digest, None);
        assert_eq!(v.sha256_hex(), None);
    }

    #[test]
    fn malformed_digests_yield_no_sha() {
        let mut a = ModelAsset {
            name: "x.dtmodel".to_string(),
            download_url: "http://example.invalid/x".to_string(),
            size: None,
            digest: Some("md5:abc".to_string()),
        };
        assert_eq!(a.sha256_hex(), None, "wrong prefix refused");
        a.digest = Some("sha256:zzzz".to_string());
        assert_eq!(a.sha256_hex(), None, "non-hex refused");
        a.digest = Some("sha256:abc".to_string());
        assert_eq!(a.sha256_hex(), None, "short tail refused");
        a.digest = Some(
            "sha256:825B3657CBB5193A67432A2F0B44AB86531CC7337F89E8E6C17D93DB9665708A".to_string(),
        );
        assert!(a.sha256_hex().is_some(), "uppercase hex accepted");
    }

    #[test]
    fn task_prefix_filters_select_each_family() {
        let assets = fixture();
        assert_eq!(assets_for_task(&assets, "denoise-").len(), 1);
        assert_eq!(assets_for_task(&assets, "rawdenoise-").len(), 1);
        assert_eq!(assets_for_task(&assets, "upscale-").len(), 1);
        assert_eq!(assets_for_task(&assets, "mask-").len(), 1);
        assert_eq!(
            assets_for_task(&assets, "denoise-")[0].name,
            "denoise-nind.dtmodel",
            "rawdenoise- must not leak into the denoise- family"
        );
        assert!(assets_for_task(&assets, "mask-")[0].name.starts_with("mask-"));
        assert!(find_asset(&assets, "nope.dtmodel").is_none());
    }

    #[test]
    fn url_builders_match_c_shapes() {
        assert_eq!(
            release_api_url("o/r", "t"),
            "https://api.github.com/repos/o/r/releases/tags/t"
        );
        assert_eq!(
            constructed_download_url("o/r", "t", "a.dtmodel"),
            "https://github.com/o/r/releases/download/t/a.dtmodel"
        );
        assert_eq!(DEFAULT_MODEL_REPO, "darktable-org/darktable-ai");
        assert_eq!(DEFAULT_RELEASE_TAG, "release-5.6.0");
    }

    #[test]
    fn bad_bodies_error() {
        assert!(parse_release_assets("not json", "o/r", "t").is_err());
        assert_eq!(
            parse_release_assets(r#"{"tag_name":"t"}"#, "o/r", "t"),
            Err(RegistryError::MissingField("assets"))
        );
        assert_eq!(
            parse_release_assets(r#"{"assets":{}}"#, "o/r", "t"),
            Err(RegistryError::MissingField("assets"))
        );
    }

    #[test]
    fn user_agent_identifies_the_application() {
        let ua = registry_user_agent();
        assert!(ua.starts_with("c41-darkroom/"), "{ua}");
        assert!(!ua.contains(' '), "no spaces: {ua}");
    }
}
