//! Verified model download (u7a): streaming fetch, sha256 gate, atomic store.
//!
//! Mirrors `src/common/ai_models.c` download + `_verify_checksum` policy:
//! stream the bytes to disk, hash them, and refuse anything whose sha256
//! does not match the release asset digest — on mismatch the partial file
//! is DELETED and an error returned, so the store never holds a bad file
//! that looks complete. Files without a usable checksum are refused
//! outright (the C: "refusing to download without integrity verification").
//!
//! Store layout: `$XDG_DATA_HOME/c41/models/<asset-name>` (`models_dir`,
//! `model_store_path`), `C41_MODELS_DIR` overrides wholesale. Writes are
//! atomic tmp-in-same-dir + rename with pid+counter names (tiles.rs house
//! pattern), so readers never see a half-written archive.
//!
//! Skip-if-present: `ensure_asset` hashes an existing destination FIRST
//! and skips the network entirely on a match (hashing ~100 MB locally is
//! cheaper than re-downloading it). A present-but-wrong file is replaced.
//!
//! Progress is `FnMut(downloaded_bytes, total_bytes_or_none)`; the total
//! comes from the registry asset `size`, never from response headers, so
//! the callback shape stays independent of HTTP header APIs.

use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Whole-transfer timeout: 55-124 MB models on a slow link need room.
/// Localhost stub tests finish in milliseconds against this.
pub const DOWNLOAD_TIMEOUT_SECS: u64 = 600;
/// Stream chunk: matches the C checksum loop's 64 KiB window.
const CHUNK_BYTES: usize = 65536;

/// Download failures. Payloads are message strings / owned data so
/// callers never match on library error shapes.
#[derive(Debug, PartialEq, Eq)]
pub enum DownloadError {
    /// HTTP transport failure (DNS, connect, timeout, non-2xx status).
    Network(String),
    /// Filesystem failure (mkdir, write, rename, ...).
    Io(String),
    /// No usable checksum supplied — refused, never stored.
    MissingChecksum,
    /// Bytes do not match the asset digest; partial file deleted.
    ChecksumMismatch { expected: String, actual: String },
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(s) => write!(f, "model download failed: {s}"),
            Self::Io(s) => write!(f, "model store IO failed: {s}"),
            Self::MissingChecksum => {
                write!(f, "refusing download without integrity verification")
            }
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "model checksum mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<ureq::Error> for DownloadError {
    fn from(e: ureq::Error) -> Self {
        Self::Network(e.to_string())
    }
}

impl From<std::io::Error> for DownloadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// `$XDG_DATA_HOME/c41/models`, with the usual `~/.local/share` fallback.
/// `C41_MODELS_DIR` overrides wholesale (container experiments).
pub fn models_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("C41_MODELS_DIR").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(base.join("c41").join("models"))
}

/// Destination path for one asset, or `None` when the name is unsafe
/// (separators, `..`, empty) or no base dir resolves. Sanitizing here —
/// not at each call site — keeps a hostile registry response inside the
/// store.
pub fn model_store_path(asset_name: &str) -> Option<PathBuf> {
    if asset_name.is_empty()
        || asset_name.contains('/')
        || asset_name.contains('\\')
        || asset_name.split(std::path::MAIN_SEPARATOR).any(|c| c == "..")
        || asset_name == ".."
        || asset_name == "."
    {
        return None;
    }
    Some(models_dir()?.join(asset_name))
}

/// Lowercase hex sha256 of in-memory bytes.
pub fn sha256_hex_of_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Lowercase hex sha256 of a file, streamed in 64 KiB chunks (the C
/// `_verify_checksum` shape — never loads the whole model into memory).
pub fn sha256_hex_of_file(path: &Path) -> Result<String, DownloadError> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Case-insensitive hex comparison (the C uses `g_ascii_strcasecmp`).
pub fn sha256_matches(actual_hex: &str, expected_hex: &str) -> bool {
    actual_hex.eq_ignore_ascii_case(expected_hex)
}

/// Uniqueness for tmp files: pid separates processes, the counter
/// separates threads within one (tiles.rs house pattern).
static DOWNLOAD_WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Stream `url` to `dest` with per-chunk progress, then verify sha256.
///
/// * `expected_sha256_hex` must be `Some`: `None` returns
///   `MissingChecksum` without touching the network.
/// * `expected_size` feeds the progress total; `None` reports running
///   bytes with no total.
/// * Bytes land in a tmp file in the SAME directory, renamed over `dest`
///   only after the hash matches; on mismatch the tmp is deleted and
///   `ChecksumMismatch` returned — never a partial file at `dest`.
pub fn download_to_path(
    url: &str,
    dest: &Path,
    expected_sha256_hex: Option<&str>,
    expected_size: Option<u64>,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<(), DownloadError> {
    let expected = expected_sha256_hex.ok_or(DownloadError::MissingChecksum)?;
    let parent = dest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&parent)?;
    let seq = DOWNLOAD_WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(".tmp-{}-{seq:x}.part", std::process::id()));

    let result = download_to_tmp(url, &tmp, expected_size, &mut progress);
    match result {
        Ok(()) => {
            let actual = sha256_hex_of_file(&tmp)?;
            if !sha256_matches(&actual, expected) {
                let _ = std::fs::remove_file(&tmp);
                return Err(DownloadError::ChecksumMismatch {
                    expected: expected.to_string(),
                    actual,
                });
            }
            if std::fs::rename(&tmp, dest).is_err() {
                let _ = std::fs::remove_file(&tmp);
                return Err(DownloadError::Io(format!(
                    "could not publish {}",
                    dest.display()
                )));
            }
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Fetch `url` when `dest` is missing or wrong; skip when present + sha
/// matches. Returns `true` when bytes came off the network, `false` on a
/// skip (progress is still reported once at full, so UI reaches 100%).
/// `sha256_hex` is required — no unverified assets, ever.
pub fn ensure_asset(
    url: &str,
    dest: &Path,
    sha256_hex: &str,
    size: Option<u64>,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<bool, DownloadError> {
    if let Ok(actual) = sha256_hex_of_file(dest) {
        if sha256_matches(&actual, sha256_hex) {
            let done = size.unwrap_or_else(|| dest.metadata().map(|m| m.len()).unwrap_or(0));
            progress(done, size.or(Some(done)));
            return Ok(false);
        }
    }
    download_to_path(url, dest, Some(sha256_hex), size, &mut progress)?;
    Ok(true)
}

/// The streaming GET: ureq call shape mirrors `tiles.rs tile_http_get`
/// (UA + global timeout + build + call), but the body is consumed chunk
/// by chunk through `Body::into_reader` (an owned `impl Read`) so
/// progress stays live on 100 MB files. Verified against the vendored
/// ureq 3.4.2 sources during u7a (see report).
fn download_to_tmp(
    url: &str,
    tmp: &Path,
    total: Option<u64>,
    progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<(), DownloadError> {
    let resp = ureq::get(url)
        .header("User-Agent", super::registry::registry_user_agent())
        .config()
        .timeout_global(Some(Duration::from_secs(DOWNLOAD_TIMEOUT_SECS)))
        .build()
        .call()?;
    let mut reader = resp.into_body().into_reader();
    let mut out = std::fs::File::create(tmp)?;
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut downloaded: u64 = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        downloaded += n as u64;
        progress(downloaded, total);
    }
    out.flush()?;
    progress(downloaded, total.or(Some(downloaded)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::registry::registry_user_agent;
    use super::*;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    /// Serializes the tests that read or write the process-global
    /// `C41_MODELS_DIR` (cargo runs tests in parallel threads).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // Localhost HTTP stub (pure std, no new test deps): serves one fixed
    // body with Content-Length to every connection until stopped. Tests
    // point download_to_path / ensure_asset at it, so nothing ever hits
    // github.com (flaky + 100 MB).
    struct StubServer {
        addr: std::net::SocketAddr,
        hits: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl StubServer {
        fn start(body: Vec<u8>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let hits = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let (hits_c, stop_c) = (Arc::clone(&hits), Arc::clone(&stop));
            let handle = std::thread::spawn(move || {
                while !stop_c.load(Ordering::Relaxed) {
                    let (mut s, _) = match listener.accept() {
                        Ok(v) => v,
                        Err(_) => {
                            std::thread::sleep(Duration::from_millis(5));
                            continue;
                        }
                    };
                    hits_c.fetch_add(1, Ordering::Relaxed);
                    s.set_nonblocking(false).ok();
                    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
                    let mut req = vec![0u8; 4096];
                    let _ = s.read(&mut req);
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = s.write_all(head.as_bytes());
                    let _ = s.write_all(&body);
                }
            });
            Self {
                addr,
                hits,
                stop,
                handle: Some(handle),
            }
        }

        fn url(&self, path: &str) -> String {
            format!("http://{}{}", self.addr, path)
        }

        fn hits(&self) -> usize {
            self.hits.load(Ordering::Relaxed)
        }
    }

    impl Drop for StubServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn fresh(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "c41_ai_dl_{}_{}_{}",
                std::process::id(),
                name,
                DOWNLOAD_WRITE_SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn file(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }

        fn leftovers(&self) -> Vec<PathBuf> {
            std::fs::read_dir(&self.path)
                .unwrap()
                .flatten()
                .map(|e| e.path())
                .collect()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex_of_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(sha256_matches("BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
        assert!(!sha256_matches("00", "ff"));
    }

    #[test]
    fn download_stores_exact_bytes_with_monotonic_progress() {
        let body: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let sha = sha256_hex_of_bytes(&body);
        let server = StubServer::start(body.clone());
        let dir = TempDir::fresh("ok");
        let dest = dir.file("model.dtmodel");
        let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
        download_to_path(
            &server.url("/model.dtmodel"),
            &dest,
            Some(&sha),
            Some(body.len() as u64),
            |d, t| seen.push((d, t)),
        )
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(server.hits(), 1);
        assert!(seen.len() >= 2, "streaming must report more than once");
        for w in seen.windows(2) {
            assert!(w[1].0 >= w[0].0, "progress monotonic: {seen:?}");
        }
        for (d, t) in &seen {
            assert_eq!(*t, Some(body.len() as u64));
            assert!(*d <= body.len() as u64);
        }
        assert_eq!(seen.last().unwrap().0, body.len() as u64);
        assert_eq!(dir.leftovers(), vec![dest], "no tmp orphan left behind");
    }

    #[test]
    fn sha_mismatch_deletes_partial_and_errors() {
        let body = b"predictable bytes, wrong hash".to_vec();
        let server = StubServer::start(body);
        let dir = TempDir::fresh("mismatch");
        let dest = dir.file("model.dtmodel");
        let mut calls = 0u32;
        let err = download_to_path(
            &server.url("/model.dtmodel"),
            &dest,
            Some("0000000000000000000000000000000000000000000000000000000000000000"),
            None,
            |_, _| calls += 1,
        )
        .unwrap_err();
        assert!(matches!(err, DownloadError::ChecksumMismatch { .. }), "{err}");
        assert!(!dest.exists(), "never a partial file at dest");
        assert!(dir.leftovers().is_empty(), "tmp deleted on mismatch");
        assert!(calls > 0, "progress still ran during the bad fetch");
    }

    #[test]
    fn missing_checksum_refuses_without_network() {
        let server = StubServer::start(b"bytes".to_vec());
        let dir = TempDir::fresh("nocheck");
        let dest = dir.file("model.dtmodel");
        let err = download_to_path(&server.url("/x"), &dest, None, None, |_, _| {}).unwrap_err();
        assert_eq!(err, DownloadError::MissingChecksum);
        assert_eq!(server.hits(), 0, "refused before any request");
        assert!(!dest.exists());
    }

    #[test]
    fn ensure_skips_present_matching_file() {
        let body = b"stable model bytes".to_vec();
        let sha = sha256_hex_of_bytes(&body);
        let server = StubServer::start(body.clone());
        let dir = TempDir::fresh("skip");
        let dest = dir.file("model.dtmodel");
        std::fs::write(&dest, &body).unwrap();
        let mut seen = Vec::new();
        let fetched = ensure_asset(
            &server.url("/model.dtmodel"),
            &dest,
            &sha,
            Some(body.len() as u64),
            |d, t| seen.push((d, t)),
        )
        .unwrap();
        assert!(!fetched, "present + matching sha skips the fetch");
        assert_eq!(server.hits(), 0, "no request on skip");
        assert_eq!(seen.last(), Some(&(body.len() as u64, Some(body.len() as u64))));
    }

    #[test]
    fn ensure_replaces_present_wrong_file() {
        let body = b"fresh server bytes".to_vec();
        let sha = sha256_hex_of_bytes(&body);
        let server = StubServer::start(body.clone());
        let dir = TempDir::fresh("replace");
        let dest = dir.file("model.dtmodel");
        std::fs::write(&dest, b"stale bytes").unwrap();
        let fetched = ensure_asset(&server.url("/model.dtmodel"), &dest, &sha, None, |_, _| {})
            .unwrap();
        assert!(fetched);
        assert_eq!(server.hits(), 1);
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[test]
    fn store_path_sanitizes_names() {
        // Serialized with the override test below: both read the
        // process-global C41_MODELS_DIR, and cargo runs tests in parallel.
        let _env = ENV_LOCK.lock().unwrap();
        assert!(model_store_path("../evil.dtmodel").is_none());
        assert!(model_store_path("sub/dir.dtmodel").is_none());
        assert!(model_store_path("").is_none());
        assert!(model_store_path(".").is_none());
        let p = model_store_path("denoise-nind.dtmodel").unwrap();
        assert!(p.ends_with("c41/models/denoise-nind.dtmodel"), "{p:?}");
        // UA shared with the registry fetch (tiles.rs precedent).
        assert!(registry_user_agent().starts_with("c41-darkroom/"));
    }

    #[test]
    fn models_dir_override_is_honoured() {
        // Serialized with every other C41_MODELS_DIR reader: env is
        // process-global and cargo runs tests in parallel.
        let _env = ENV_LOCK.lock().unwrap();
        let dir = TempDir::fresh("override");
        let prev = std::env::var_os("C41_MODELS_DIR");
        std::env::set_var("C41_MODELS_DIR", &dir.path);
        let got = models_dir();
        match prev {
            Some(v) => std::env::set_var("C41_MODELS_DIR", v),
            None => std::env::remove_var("C41_MODELS_DIR"),
        }
        assert_eq!(got, Some(dir.path.clone()));
    }
}
