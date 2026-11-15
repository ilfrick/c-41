//! Runtime memory limits for safety-sensitive future call paths.
//!
//! Reports total/effective memory limits, not currently free memory. On Linux,
//! the effective total accounts for cgroup limits as C-41 safety behavior; this
//! is not darktable parity. It provides no guarantee against overcommit. If
//! probing fails, the pinned darktable 2 GiB fallback is reported rather than an error.

#[cfg(target_os = "linux")]
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorySource {
    Host,
    CgroupV1,
    CgroupV2,
    Fallback,
}

/// Host-reported total and effective total (not free memory), with its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMemory {
    pub host_bytes: u64,
    pub effective_bytes: u64,
    pub source: MemorySource,
}

const FALLBACK_BYTES: u64 = 2 * 1024 * 1024 * 1024;
// Linux cgroup v1 represents unlimited memory by a page-aligned value near
// LONG_MAX; the kernel's documented threshold is LONG_MAX - PAGE_SIZE.
const V1_UNLIMITED_THRESHOLD: u64 = (i64::MAX as u64) - 4096;

pub fn detect() -> SystemMemory {
    #[cfg(target_os = "linux")]
    if let Some(host) = read_host_total() {
        return construct_memory(host, read_cgroup_limit(host));
    }
    construct_fallback()
}

fn construct_memory(host: u64, cgroup: Option<(u64, MemorySource)>) -> SystemMemory {
    if host == 0 {
        return construct_fallback();
    }
    if let Some((limit, source)) = cgroup.filter(|(limit, _)| *limit > 0 && *limit < host) {
        return SystemMemory {
            host_bytes: host,
            effective_bytes: limit,
            source,
        };
    }
    SystemMemory {
        host_bytes: host,
        effective_bytes: host,
        source: MemorySource::Host,
    }
}

fn construct_fallback() -> SystemMemory {
    SystemMemory {
        host_bytes: FALLBACK_BYTES,
        effective_bytes: FALLBACK_BYTES,
        source: MemorySource::Fallback,
    }
}

#[cfg(target_os = "linux")]
fn read_host_total() -> Option<u64> {
    parse_meminfo_total(&fs::read_to_string("/proc/meminfo").ok()?)
}

#[cfg(target_os = "linux")]
fn read_cgroup_limit(host: u64) -> Option<(u64, MemorySource)> {
    let memberships = parse_memberships(&fs::read_to_string("/proc/self/cgroup").ok()?);
    let mounts = parse_mounts(&fs::read_to_string("/proc/self/mountinfo").ok()?);
    select_limit(host, &memberships, &mounts, |path| {
        fs::read_to_string(path)
            .ok()
            .as_deref()
            .and_then(parse_cgroup_limit)
    })
}

fn parse_meminfo_total(contents: &str) -> Option<u64> {
    let line = contents
        .lines()
        .find(|line| line.split_whitespace().next() == Some("MemTotal:"))?;
    let mut fields = line.split_whitespace();
    fields.next()?;
    let kb: u64 = fields.next()?.parse().ok()?;
    if fields.next()? != "kB" {
        return None;
    }
    let bytes = kb.checked_mul(1024)?;
    (bytes > 0).then_some(bytes)
}

fn parse_cgroup_limit(contents: &str) -> Option<u64> {
    let text = contents.trim();
    if text == "max" {
        return None;
    }
    let n = text.parse::<u64>().ok()?;
    // Reject the v1 unlimited sentinel and v2 unlimited text, rather than
    // treating a large sentinel as a meaningful (possibly host-clamped) limit.
    (n > 0 && n < V1_UNLIMITED_THRESHOLD).then_some(n)
}

#[derive(Debug, PartialEq, Eq)]
struct Membership {
    v2: bool,
    path: String,
}

#[derive(Debug, PartialEq, Eq)]
struct Mount {
    v2: bool,
    root: String,
    point: String,
}

fn parse_memberships(s: &str) -> Vec<Membership> {
    s.lines()
        .filter_map(|line| {
            let mut p = line.splitn(3, ':');
            let _id = p.next()?;
            let controllers = p.next()?;
            let path = p.next()?;
            if controllers.is_empty() {
                (line.starts_with("0::")).then(|| Membership {
                    v2: true,
                    path: path.into(),
                })
            } else if controllers.split(',').any(|c| c == "memory") {
                Some(Membership {
                    v2: false,
                    path: path.into(),
                })
            } else {
                None
            }
        })
        .collect()
}

fn unescape_mount(s: &str) -> String {
    s.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

fn parse_mounts(s: &str) -> Vec<Mount> {
    s.lines()
        .filter_map(|line| {
            let (left, right) = line.split_once(" - ")?;
            let l: Vec<_> = left.split_whitespace().collect();
            let r: Vec<_> = right.split_whitespace().collect();
            if l.len() < 6 || r.len() < 3 {
                return None;
            }
            let fs = r[0];
            let v2 = fs == "cgroup2";
            let v1memory = fs == "cgroup"
                && (r[1].split(',').any(|x| x == "memory")
                    || r[2].split(',').any(|x| x == "memory"));
            (v2 || v1memory).then(|| Mount {
                v2,
                root: unescape_mount(l[3]),
                point: unescape_mount(l[4]),
            })
        })
        .collect()
}

fn resolved_file(m: &Membership, mounts: &[Mount]) -> Option<(PathBuf, PathBuf, bool)> {
    let membership = Path::new(&m.path);
    let (mount, relative) = mounts
        .iter()
        .filter(|mount| mount.v2 == m.v2)
        .filter_map(|mount| {
            let relative = membership.strip_prefix(Path::new(&mount.root)).ok()?;
            if relative.components().any(|c| c == Component::ParentDir) {
                return None;
            }
            Some((mount, relative))
        })
        .max_by_key(|(mount, _)| Path::new(&mount.root).components().count())?;
    Some((
        Path::new(&mount.point).join(relative),
        PathBuf::from(&mount.point),
        m.v2,
    ))
}

fn select_limit<F>(
    host: u64,
    memberships: &[Membership],
    mounts: &[Mount],
    mut read: F,
) -> Option<(u64, MemorySource)>
where
    F: FnMut(&Path) -> Option<u64>,
{
    let mut best: Option<(u64, MemorySource)> = None;
    for m in memberships {
        if let Some((mut dir, root, v2)) = resolved_file(m, mounts) {
            let source = if v2 {
                MemorySource::CgroupV2
            } else {
                MemorySource::CgroupV1
            };
            let file = if v2 {
                "memory.max"
            } else {
                "memory.limit_in_bytes"
            };
            loop {
                if let Some(n) = read(&dir.join(file)).filter(|&n| n > 0 && n < host) {
                    if best.map_or(true, |(current, _)| n < current) {
                        best = Some((n, source));
                    }
                }
                if dir == root {
                    break;
                }
                let Some(parent) = dir.parent() else { break };
                // Do not follow a malformed mount path beyond its mount root.
                if !parent.starts_with(&root) {
                    break;
                }
                dir = parent.to_path_buf();
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const V2M: &str = "0::/tenant/job";
    const V1M: &str = "5:cpu,memory:/tenant/job";
    const MOUNTS: &str = "29 23 0:26 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n30 23 0:27 /tenant /sys/fs/cgroup/mem rw - cgroup cgroup rw,memory";

    #[test]
    fn meminfo_validates_nonzero() {
        assert_eq!(parse_meminfo_total("MemTotal: 123 kB"), Some(125952));
        assert_eq!(parse_meminfo_total("MemTotal: 0 kB"), None);
        assert_eq!(parse_meminfo_total("MemTotal: nope kB"), None);
    }

    #[test]
    fn parses_membership_mount_root_and_resolves_nested_paths() {
        let ms = parse_memberships(V2M);
        let mounts = parse_mounts(MOUNTS);
        assert_eq!(
            resolved_file(&ms[0], &mounts).unwrap().0,
            PathBuf::from("/sys/fs/cgroup/job")
        );
        assert!(!parse_memberships(V1M)[0].v2);
    }

    #[test]
    fn chooses_applicable_most_specific_matching_mount() {
        let membership = parse_memberships("0::/tenant/job");
        let mounts = parse_mounts(
            "1 0 0:1 /unrelated /first rw - cgroup2 cgroup rw\n\
             2 0 0:2 /tenant /later rw - cgroup2 cgroup rw\n\
             3 0 0:3 /tenant/job /specific rw - cgroup2 cgroup rw",
        );
        assert_eq!(
            resolved_file(&membership[0], &mounts).unwrap().0,
            PathBuf::from("/specific")
        );
    }

    #[test]
    fn escaped_mountpoint_and_v2_precedence_then_v1_fallback() {
        let m = parse_mounts("1 0 0:1 / /sys/fs/cgroup/a\\040b rw - cgroup2 cgroup rw");
        let mem = parse_memberships(V2M);
        assert_eq!(
            resolved_file(&mem[0], &m).unwrap().0,
            PathBuf::from("/sys/fs/cgroup/a b/tenant/job")
        );
        let ms = parse_memberships("0::/tenant/job\n5:memory:/tenant/job");
        let mounts = parse_mounts("1 0 0:1 /tenant /v2 rw - cgroup2 cgroup rw\n2 0 0:2 /tenant /v1 rw - cgroup cgroup rw,memory");
        let chosen = select_limit(100, &ms, &mounts, |p| {
            if p.ends_with("memory.max") {
                Some(100)
            } else {
                Some(30)
            }
        });
        assert_eq!(chosen, Some((30, MemorySource::CgroupV1)));
    }

    #[test]
    fn rejects_unlimited_zero_malformed_and_host_or_larger() {
        assert_eq!(
            parse_cgroup_limit(&V1_UNLIMITED_THRESHOLD.to_string()),
            None
        );
        assert_eq!(parse_cgroup_limit("0"), None);
        assert_eq!(parse_cgroup_limit("bad"), None);
        let m = parse_memberships(V2M);
        let mounts = parse_mounts(MOUNTS);
        assert_eq!(select_limit(10, &m, &mounts, |_| Some(10)), None);
        assert_eq!(select_limit(10, &m, &mounts, |_| Some(20)), None);
    }

    #[test]
    fn walks_nested_ancestors_and_selects_strictest_v2_limit() {
        let memberships = parse_memberships("0::/tenant/team/job");
        let mounts = parse_mounts("1 0 0:1 /tenant /cg rw - cgroup2 cgroup rw");
        for leaf in [None, Some(80)] {
            let chosen = select_limit(100, &memberships, &mounts, |p| match p.to_str().unwrap() {
                "/cg/team/job/memory.max" => leaf,
                "/cg/team/memory.max" => Some(30),
                _ => None,
            });
            assert_eq!(chosen, Some((30, MemorySource::CgroupV2)));
        }
    }

    #[test]
    fn walks_v1_parent_and_stops_at_mount_root() {
        let memberships = parse_memberships("5:memory:/tenant/team/job");
        let mounts = parse_mounts("1 0 0:1 /tenant /cg rw - cgroup cgroup rw,memory");
        let mut seen = Vec::new();
        let chosen = select_limit(100, &memberships, &mounts, |p| {
            seen.push(p.to_path_buf());
            match p.to_str().unwrap() {
                "/cg/team/job/memory.limit_in_bytes" => Some(80),
                "/cg/team/memory.limit_in_bytes" => Some(V1_UNLIMITED_THRESHOLD),
                "/cg/memory.limit_in_bytes" => Some(25),
                _ => Some(1),
            }
        });
        assert_eq!(chosen, Some((25, MemorySource::CgroupV1)));
        assert_eq!(seen.last().unwrap(), Path::new("/cg/memory.limit_in_bytes"));
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn selects_minimum_across_applicable_memberships_and_mounts() {
        let memberships = parse_memberships("0::/tenant/job\n5:memory:/tenant/job");
        let mounts = parse_mounts("1 0 0:1 /tenant /v2 rw - cgroup2 cgroup rw\n2 0 0:2 /tenant /v1 rw - cgroup cgroup rw,memory");
        let chosen = select_limit(100, &memberships, &mounts, |p| {
            if p.starts_with("/v2") {
                Some(40)
            } else {
                Some(20)
            }
        });
        assert_eq!(chosen, Some((20, MemorySource::CgroupV1)));
    }

    #[test]
    fn invalid_host_falls_back_and_unusable_cgroup_uses_host_total() {
        let invalid_host = parse_meminfo_total("MemTotal: invalid kB");
        let result =
            invalid_host.map_or_else(construct_fallback, |host| construct_memory(host, None));
        assert_eq!(result, construct_fallback());
        assert_eq!(construct_memory(0, None), construct_fallback());

        let memberships = parse_memberships(V2M);
        let mounts = parse_mounts(MOUNTS);
        let cgroup = select_limit(10, &memberships, &mounts, |_| Some(20));
        assert_eq!(
            construct_memory(10, cgroup),
            SystemMemory {
                host_bytes: 10,
                effective_bytes: 10,
                source: MemorySource::Host,
            }
        );
    }
}
