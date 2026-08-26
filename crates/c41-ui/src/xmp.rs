//! XMP sidecar writer (m4-142) — the piece parity item 2.3 deliberately left
//! open: metadata edits lived only in the catalogue, so they did not travel
//! with the file. darktable follows every metadata write with
//! `dt_image_synch_xmps` (`src/libs/metadata.c:393`), maintaining a
//! `<filename>.xmp` beside the image; this module does the same for the five
//! writable Dublin Core fields.
//!
//! **Merge-preserving, never clobbering.** An existing sidecar may carry a
//! user's entire darktable edit history — rewriting it wholesale would be data
//! loss, the worst thing this code could do. An existing packet is therefore
//! rewritten in one streaming pass where everything not ours is copied
//! through unchanged (foreign `rdf:Description` blocks, comments, CDATA,
//! processing instructions, the packet wrapper itself), only the five target
//! properties are replaced, and a document that fails to parse as RDF/XMP
//! leaves the file UNTOUCHED with a failure reported rather than being
//! overwritten by a fresh packet.
//!
//! **Shapes.** Values follow what exiv2/darktable emit: `dc:title`,
//! `dc:description` and `dc:rights` are language alternatives
//! (`rdf:Alt` / `x-default`); `dc:creator` a sequence (`rdf:Seq`);
//! `dc:publisher` a bag (`rdf:Bag`). The editor edits each as one string, so
//! editing a multi-creator sidecar's creator field collapses it to that one
//! string — recorded deviation, inherent to darktable's single-string
//! metadata model too. The same collapse applies within a value: language
//! alternatives beyond `x-default` are dropped when the field is written.
//!
//! **Namespaces** are resolved properly: a property counts as ours when its
//! prefix is bound to the Dublin Core namespace URI in scope, whatever the
//! prefix spells; appended properties reuse the document's existing DC prefix,
//! adding an `xmlns:dc` declaration only when none exists; the document's own
//! RDF prefix is reused when injecting a new Description.
//!
//! **Rating and colour labels** (m4-153) ride the same merge machinery through
//! [`Payload::RatingLabels`], closing what the metadata-editor sync left open:
//! star-rating and colour-label writes used to live only in the catalogue.
//! darktable's sidecar shape (`src/common/exif.cc`): `Xmp.xmp.Rating` is plain
//! scalar text holding `flags & 7` (0–5; −1 for rejected — unreachable here,
//! C41 has no reject action, so the API takes `u8`), and
//! `Xmp.darktable.colorlabels` is an `rdf:Seq` of the colour indices as
//! decimal strings, written ONLY when at least one label is set — zero labels
//! means the property is absent, so that is what we write. Unlike the DC
//! fields there is no read-from-catalogue step here: the caller passes the
//! values it just committed, which under the per-image write lock are exactly
//! the catalogue state. A `None` field leaves its property untouched — a
//! rating-only sync must not delete colour labels, nor vice versa.
//!
//! **Locking** is part of the entry-point contract. [`sync_sidecar`] acquires
//! [`crate::lighttable::path_write_lock`] for the image, serialising the whole
//! read-modify-write against the lighttable's own sidecar writers; holders of
//! that lock must call [`sync_sidecar_unlocked`] instead. The rating/label
//! sync is deliberately exposed ONLY as
//! [`sync_rating_labels_sidecar_unlocked`]: every current caller writes from
//! inside a [`crate::lighttable::serialized_write`] closure that already
//! holds the lock, and `std::sync::Mutex` is not reentrant — a plain-named
//! twin would sit one careless nesting away from a deadlock. A future caller
//! off the lock path takes the lock itself (or reintroduces a locking
//! wrapper) explicitly.

use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};

/// The Dublin Core namespace — what makes a property one of "ours".
const DC_NS: &str = "http://purl.org/dc/elements/1.1/";
/// The XMP basic namespace, home of `xmp:Rating` (exiv2's built-in binding).
const XMP_NS: &str = "http://ns.adobe.com/xap/1.0/";
/// darktable's own namespace, home of `darktable:colorlabels`
/// (`src/common/exif.cc:6232`).
const DT_NS: &str = "http://darktable.sf.net/";
/// The RDF namespace, for recognising `rdf:RDF` / `rdf:Description` however
/// their prefixes are bound.
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
/// Local names of the five writable properties, canonical order.
const TARGET_LOCALS: [&str; 5] = ["title", "description", "creator", "publisher", "rights"];
/// The namespaces this module may write properties into, with the fallback
/// prefix each gets when the document declares none. Index order is load-bearing
/// for [`DescBuf::prefixes`] and [`Payload::uses_ns`].
const NS_TABLE: [(&str, &str); 3] = [(DC_NS, "dc"), (XMP_NS, "xmp"), (DT_NS, "darktable")];

/// The five values to write, in [`TARGET_LOCALS`] order.
struct XmpValues {
    title: String,
    description: String,
    creator: String,
    publisher: String,
    rights: String,
}

impl XmpValues {
    fn all_empty(&self) -> bool {
        self.title.is_empty()
            && self.description.is_empty()
            && self.creator.is_empty()
            && self.publisher.is_empty()
            && self.rights.is_empty()
    }
}

/// The property set one sync owns and rewrites. One enum, two payloads, ONE
/// streaming merge ([`rewrite_document`]) — a second hand-rolled pass over
/// scope stacks and balance counting would only drift from this one.
enum Payload {
    /// The five Dublin Core fields (`sync_sidecar`, m4-142). Every target is
    /// owned unconditionally; an empty string deletes its property.
    Metadata(XmpValues),
    /// Star rating and/or colour-label mask (m4-153). `None` leaves that
    /// property exactly as found — these syncs are partial by nature (a star
    /// click knows nothing about labels). `Some(mask)` replaces the Seq,
    /// with mask 0 emitting nothing (= deleted, darktable writes the
    /// property only when at least one label is set).
    RatingLabels {
        rating: Option<u8>,
        colors: Option<u8>,
    },
}

impl Payload {
    /// Does this sync REPLACE the property resolved to `(iri, local)`? Only
    /// owned properties are dropped from the document; everything else —
    /// including our own namespaces when the payload's field is `None` —
    /// streams through verbatim.
    fn owns(&self, iri: Option<&str>, local: &str) -> bool {
        match self {
            Payload::Metadata(_) => iri == Some(DC_NS) && TARGET_LOCALS.contains(&local),
            Payload::RatingLabels { rating, colors } => {
                (iri == Some(XMP_NS) && local == "Rating" && rating.is_some())
                    || (iri == Some(DT_NS) && local == "colorlabels" && colors.is_some())
            }
        }
    }

    /// Does this payload write anything into namespace table row `i`?
    /// Drives prefix discovery and `xmlns:` declaration on injection. A
    /// zero colour mask emits nothing, so it must not make us declare the
    /// darktable namespace on a fresh or injected Description.
    fn uses_ns(&self, i: usize) -> bool {
        match self {
            Payload::Metadata(_) => i == 0,
            Payload::RatingLabels { rating, colors } => match i {
                1 => rating.is_some(),
                2 => colors.is_some_and(|m| m != 0),
                _ => false,
            },
        }
    }

    /// True when the payload EMITS at least one property. Gates sidecar
    /// creation and Description injection only — a payload that owns a
    /// property without emitting (colour mask 0 = delete) still rewrites an
    /// EXISTING document, it just never conjures new XML for nothing.
    fn emits_anything(&self) -> bool {
        match self {
            Payload::Metadata(v) => !v.all_empty(),
            Payload::RatingLabels { rating, colors } => {
                rating.is_some() || colors.is_some_and(|m| m != 0)
            }
        }
    }

    /// Emit the owned properties under per-namespace prefixes (each either
    /// the document's own binding or [`NS_TABLE`]'s fallback). `rdf_prefix` is
    /// the RDF binding in scope where the properties land — containers
    /// (`rdf:Alt/Seq/Bag/li`) are RDF resources and follow it like any other
    /// prefix spelling. Deletion-shaped values emit nothing, per the module doc.
    fn emit<W: std::io::Write>(
        &self,
        w: &mut Writer<W>,
        rdf_prefix: &str,
        prefixes: &[Option<&str>],
    ) {
        let pfx = |i: usize| prefixes.get(i).copied().flatten().unwrap_or(NS_TABLE[i].1);
        match self {
            Payload::Metadata(v) => push_properties(w, rdf_prefix, pfx(0), v),
            Payload::RatingLabels { rating, colors } => {
                if let Some(r) = rating {
                    let _ = w
                        .create_element(format!("{}:Rating", pfx(1)))
                        .write_text_content(BytesText::new(&r.to_string()));
                }
                if let Some(m) = colors.filter(|m| *m != 0) {
                    let _ = w
                        .create_element(format!("{}:colorlabels", pfx(2)))
                        .write_inner_content(|prop| {
                            prop.create_element(format!("{rdf_prefix}:Seq"))
                                .write_inner_content(|list| {
                                    for i in 0..c41_db::colorlabels::COLOR_COUNT {
                                        if m & (1 << i) != 0 {
                                            let _ = list
                                                .create_element(format!("{rdf_prefix}:li"))
                                                .write_text_content(BytesText::new(
                                                    &i.to_string(),
                                                ));
                                        }
                                    }
                                    Ok::<(), std::io::Error>(())
                                })
                                .map(drop)
                        });
                }
            }
        }
    }
}

/// Full tag name of a Start/End event as UTF-8, prefix included.
/// `QName` is a transparent newtype over `&'a [u8]`, so this borrows the
/// reader's input directly.
fn qname_str<'a>(qname: quick_xml::name::QName<'a>) -> &'a str {
    std::str::from_utf8(qname.0).unwrap_or("")
}

/// Resolve a full tag name against the xmlns scope stack →
/// `(bound IRI or None, local part)`. An unprefixed name never resolves here:
/// our five targets are always prefixed in practice, and treating "no binding"
/// as "not ours" is the safe direction (we skip rather than rewrite).
fn resolve<'n>(
    name: &'n str,
    scope: &[std::collections::HashMap<String, String>],
) -> (Option<String>, &'n str) {
    match name.split_once(':') {
        Some((p, local)) => {
            let iri = scope.iter().rev().find_map(|m| m.get(p).cloned());
            (iri, local)
        }
        None => (None, name),
    }
}

/// Sidecar path for an image, darktable's convention: `.xmp` appended to the
/// FULL filename (`IMG_0001.ORF` → `IMG_0001.ORF.xmp`).
pub(crate) fn sidecar_path(image_path: &str) -> String {
    format!("{image_path}.xmp")
}

/// Synchronise one image's sidecar with its catalogue metadata. Called after
/// every successful metadata save. True also covers the quiet nothing-to-do
/// cases: no image file on disk to attach to (demo-catalogue entries, an
/// unplugged medium), or all fields empty with no sidecar to clean up. False
/// means real trouble worth surfacing — unwritable location, unreadable
/// sidecar, or an existing sidecar that failed to parse (left untouched).
///
/// Locking: acquires [`crate::lighttable::path_write_lock`] for `image_path`
/// across the whole catalogue-read + rewrite, so a concurrent rating/label
/// write cannot interleave between this sync's read and its file write. If you
/// ALREADY hold that lock — inside a [`crate::lighttable::serialized_write`]
/// closure, as every lighttable writer does — call [`sync_sidecar_unlocked`]
/// instead; `std::sync::Mutex` is not reentrant and nesting deadlocks.
pub(crate) fn sync_sidecar(db_path: &str, image_path: &str) -> bool {
    let lock = crate::lighttable::path_write_lock(image_path);
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    sync_sidecar_unlocked(db_path, image_path)
}

/// Body of [`sync_sidecar`]. Caller MUST already hold
/// [`crate::lighttable::path_write_lock`] for `image_path`.
fn sync_sidecar_unlocked(db_path: &str, image_path: &str) -> bool {
    if image_path.is_empty() || db_path.is_empty() {
        return false;
    }
    if !std::path::Path::new(image_path).exists() {
        return true;
    }
    // Fail closed on lookup trouble: `try_load_metadata` distinguishes a
    // genuinely all-blank catalogue from one we could not read. Treating a
    // failed read as "user blanked everything" would strip the five fields
    // from an existing sidecar — the one direction this module must never go
    // by accident. Runs under the lock, so the read reflects exactly what the
    // catalogue committed before this sync was requested.
    let fields = crate::persist::try_load_metadata(db_path, image_path);
    let Some(fields) = fields else {
        return false;
    };
    let mut v: [String; 5] = std::array::from_fn(|_| String::new());
    for (field, value) in fields {
        use crate::persist::MetaField as F;
        let slot = match field {
            F::Title => 0,
            F::Description => 1,
            F::Creator => 2,
            F::Publisher => 3,
            F::Rights => 4,
        };
        v[slot] = value;
    }
    let values = XmpValues {
        title: std::mem::take(&mut v[0]),
        description: std::mem::take(&mut v[1]),
        creator: std::mem::take(&mut v[2]),
        publisher: std::mem::take(&mut v[3]),
        rights: std::mem::take(&mut v[4]),
    };
    let sp = sidecar_path(image_path);
    let payload = Payload::Metadata(values);
    match std::fs::read_to_string(&sp) {
        Ok(existing) => match rewrite_document(&existing, &payload) {
            Some(out) => atomic_write(std::path::Path::new(&sp), &out),
            // Parse failure: overwriting with a fresh packet could destroy
            // history we merely failed to understand. Keep the file.
            None => false,
        },
        Err(_) if std::path::Path::new(&sp).exists() => false, // unreadable: never clobber
        Err(_) => {
            if !payload.emits_anything() {
                return true; // nothing worth creating a sidecar for
            }
            atomic_write(std::path::Path::new(&sp), &fresh_document(&payload))
        }
    }
}

/// Synchronise one image's sidecar with star rating and/or colour labels the
/// caller just committed to the catalogue (m4-153). Quiet success covers the
/// nothing-to-do cases (no image on disk, both fields `None`); false means
/// real trouble — unwritable location, unreadable or unparseable existing
/// sidecar (left byte-identical). A missing sidecar is created only when
/// something would be emitted; an existing one is still rewritten when the
/// payload owns a property without emitting (colour mask 0 deletes).
///
/// # Locking
///
/// Caller MUST already hold [`crate::lighttable::path_write_lock`] for
/// `image_path` — i.e. run from inside a
/// [`crate::lighttable::serialized_write`] closure, like the catalogue write
/// this sync mirrors. Taking the lock here instead would deadlock every
/// current caller (`std::sync::Mutex` is not reentrant); see the module doc.
pub(crate) fn sync_rating_labels_sidecar_unlocked(
    image_path: &str,
    rating: Option<u8>,
    colors: Option<u8>,
) -> bool {
    if image_path.is_empty() {
        return false;
    }
    if !std::path::Path::new(image_path).exists() {
        return true;
    }
    let payload = Payload::RatingLabels { rating, colors };
    if rating.is_none() && colors.is_none() {
        return true; // owns nothing at all — no document could change under it
    }
    let sp = sidecar_path(image_path);
    match std::fs::read_to_string(&sp) {
        Ok(existing) => match rewrite_document(&existing, &payload) {
            Some(out) => atomic_write(std::path::Path::new(&sp), &out),
            None => false, // parse failure: keep the file, report failure
        },
        Err(_) if std::path::Path::new(&sp).exists() => false,
        Err(_) => {
            if !payload.emits_anything() {
                return true; // nothing emitted → never create a packet for it
            }
            atomic_write(std::path::Path::new(&sp), &fresh_document(&payload))
        }
    }
}

/// Atomic same-directory tmp-plus-rename write. Any IO failure is reported
/// (false), never panicked; a failed write leaves any previous file intact.
/// Accepted edges, consistent with g_file_set_contents-class writers: no
/// fsync before the rename (power-loss window), a symlinked target is
/// replaced by a regular file rather than written through, and the mode
/// resets to umask defaults instead of inheriting the old file's.
fn atomic_write(target: &std::path::Path, bytes: &[u8]) -> bool {
    let dir = match target.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => std::path::PathBuf::from("."),
    };
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("sidecar");
    let tmp = dir.join(format!(".{stem}.tmp-{}-{nanos:x}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    if std::fs::rename(&tmp, target).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    true
}

// ── Property emitters ───────────────────────────────────────────────────────

/// XML 1.0 forbids most control characters in content; the editor cannot stop
/// a paste, quick-xml escapes only markup characters, so an invalid char would
/// otherwise create a permanently ill-formed sidecar that every later sync
/// "succeeds" rewriting. Replace with U+FFFD rather than drop, so corruption
/// is visible at the other end.
fn sanitize_text(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if matches!(c, '\t' | '\n' | '\r')
                || (' '..='~').contains(&c)
                || ('\u{a0}'..='\u{d7ff}').contains(&c)
                || ('\u{e000}'..='\u{fffd}').contains(&c)
                || ('\u{10000}'..='\u{10ffff}').contains(&c)
            {
                c
            } else {
                '\u{fffd}'
            }
        })
        .collect()
}

/// Append our non-empty properties under `prefix` (canonical shapes, see the
/// module doc). Blank values emit nothing — blank DELETES the property, the
/// upstream convention that keeps "no title" one state instead of two.
fn push_properties<W: std::io::Write>(
    w: &mut Writer<W>,
    rdf_prefix: &str,
    prefix: &str,
    v: &XmpValues,
) {
    let prop = |name: &str| format!("{prefix}:{name}");
    list_property(w, prop("title"), rdf_prefix, "Alt", &v.title, true);
    list_property(w, prop("description"), rdf_prefix, "Alt", &v.description, true);
    list_property(w, prop("creator"), rdf_prefix, "Seq", &v.creator, false);
    list_property(w, prop("publisher"), rdf_prefix, "Bag", &v.publisher, false);
    list_property(w, prop("rights"), rdf_prefix, "Alt", &v.rights, true);
}

/// One array-shaped property:
/// `<p:name><r:KIND><r:li [xml:lang="x-default"]>v</r:li></r:KIND></p:name>`
/// with the container names bound to the document's RDF prefix.
fn list_property<W: std::io::Write>(
    w: &mut Writer<W>,
    qname: String,
    rdf_prefix: &str,
    kind: &str,
    value: &str,
    lang_default: bool,
) {
    if value.is_empty() {
        return;
    }
    let kind_qname = format!("{rdf_prefix}:{kind}");
    let li_qname = format!("{rdf_prefix}:li");
    let _ = w.create_element(qname).write_inner_content(|prop| {
        prop.create_element(kind_qname)
            .write_inner_content(|list| {
                let mut li = list.create_element(&li_qname);
                if lang_default {
                    li = li.with_attribute(("xml:lang", "x-default"));
                }
                // `BytesText::new` escapes markup; sanitize first because it does
            // not touch control characters (see `sanitize_text`).
                li.write_text_content(BytesText::new(&sanitize_text(value)))
                    .map(drop)
            })
            .map(drop)
    });
}

/// A brand-new minimal packet in darktable/exiv2 shape (xpacket wrapper,
/// `x:xmpmeta` / `rdf:RDF` / one Description carrying the payload's
/// properties, each written namespace declared).
fn fresh_document(payload: &Payload) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(1024);
    // The begin="…" delimiter carries a literal U+FEFF per the XMP spec.
    out.extend_from_slice(
        "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>".as_bytes(),
    );
    {
        let mut w = Writer::new(&mut out);
        let _ = w
            .create_element("x:xmpmeta")
            .with_attribute(("xmlns:x", "adobe:ns:meta/"))
            .with_attribute(("x:xmptk", "C-41"))
            .write_inner_content(|meta| {
                meta.create_element("rdf:RDF")
                    .with_attribute(("xmlns:rdf", RDF_NS))
                    .write_inner_content(|rdf| {
                        let mut el = rdf
                            .create_element("rdf:Description")
                            .with_attribute(("rdf:about", ""));
                        for (i, (uri, dp)) in NS_TABLE.iter().enumerate() {
                            if payload.uses_ns(i) {
                                el = el.with_attribute((format!("xmlns:{dp}").as_str(), *uri));
                            }
                        }
                        el.write_inner_content(|d| {
                            let defaults: Vec<Option<&str>> =
                                NS_TABLE.iter().map(|(_, dp)| Some(*dp)).collect();
                            // A fresh packet declares its own rdf binding.
                            payload.emit(d, "rdf", &defaults);
                            Ok::<(), std::io::Error>(())
                        })
                        .map(drop)
                    })
                    .map(drop)
            });
    }
    out.extend_from_slice(br#"<?xpacket end="w"?>"#);
    out
}

// ── Streaming merge ─────────────────────────────────────────────────────────

/// An attribute kept exactly as parsed (raw key/value bytes), so a rewritten
/// Description re-emits the source's entity encoding untouched.
struct RawAttr(Vec<u8>, Vec<u8>);

/// A top-level `rdf:Description` being buffered: original Start identity +
/// attributes, children streamed into owned bytes with target-property
/// subtrees dropped.
struct DescBuf {
    /// Original tag name including its prefix (`rdf:Description`, `r:Description`, …).
    qname: String,
    attrs: Vec<RawAttr>,
    /// Prefix bound to each [`NS_TABLE`] namespace ON THIS ELEMENT, if declared
    /// (index-aligned with the table; `None` = fall back to the default prefix
    /// and declare it when we emit into that namespace here).
    prefixes: [Option<String>; NS_TABLE.len()],
    children: Writer<std::vec::Vec<u8>>,
    has_target: bool,
}

/// Collect xmlns declarations of a Start event (default `xmlns` included as
/// the empty prefix). A malformed declaration means the bindings that follow
/// are unknowable — `Err` makes the caller refuse the document rather than
/// silently drop an attribute and rewrite anyway.
fn xmlns_of(start: &BytesStart) -> Result<std::collections::HashMap<String, String>, ()> {
    let mut m = std::collections::HashMap::new();
    for a in start.attributes() {
        let a = a.map_err(|_| ())?;
        let key = String::from_utf8_lossy(a.key.as_ref()).to_string();
        if let Some(p) = key.strip_prefix("xmlns:") {
            m.insert(p.to_string(), String::from_utf8_lossy(&a.value).to_string());
        } else if key == "xmlns" {
            m.insert(String::new(), String::from_utf8_lossy(&a.value).to_string());
        }
    }
    Ok(m)
}

/// Advance past the subtree rooted at a just-seen Start whose scope entry is
/// already pushed, emitting nothing. The skipped element's own End is detected
/// by balance counting and leaves its scope entry pushed for the CALLER to pop
/// (mirroring how every other Start hands its binding to its own End).
/// Returns false on EOF/error (truncated input) so the caller can fail
/// conservatively.
fn skip_subtree(
    reader: &mut Reader<&[u8]>,
    scope: &mut Vec<std::collections::HashMap<String, String>>,
    depth: &mut usize,
) -> bool {
    let mut balance = 0usize;
    loop {
        match reader.read_event() {
            Ok(Event::Start(s)) => {
                let ns = match xmlns_of(&s) {
                    Ok(ns) => ns,
                    Err(_) => return false,
                };
                scope.push(ns);
                *depth += 1;
                balance += 1;
            }
            Ok(Event::End(_)) => {
                if balance == 0 {
                    return true; // this End closed the skipped element
                }
                scope.pop();
                *depth -= 1;
                balance -= 1;
            }
            Ok(Event::Eof) | Err(_) => return false,
            _ => {}
        }
    }
}

/// Rewrite an existing packet. Everything passes through unchanged except:
/// properties the [`Payload`] owns are dropped wherever found and re-emitted
/// ONCE — into the first Description that carried any owned property (reusing
/// the document's own prefix for each namespace we write, declaring one if
/// needed), otherwise into a new Description injected just before
/// `</rdf:RDF>` (reusing the document's RDF prefix). `None` = not a
/// well-formed RDF/XMP document; callers must NOT overwrite the file in that
/// case.
fn rewrite_document(src: &str, payload: &Payload) -> Option<Vec<u8>> {
    let mut reader = Reader::from_str(src);
    // <dc:title/> arrives as Start+End, so one code path handles empties.
    reader.config_mut().expand_empty_elements = true;
    let mut w = Writer::new(Vec::with_capacity(src.len() + 512));
    let mut scope: Vec<std::collections::HashMap<String, String>> = Vec::new();
    let mut depth: usize = 0;
    let mut saw_rdf = false;
    let mut rdf_depth: Option<usize> = None;
    let mut rdf_prefix = String::from("rdf");
    let mut desc: Option<DescBuf> = None;
    let mut desc_depth: usize = 0;
    let mut props_emitted = false;

    loop {
        let ev = match reader.read_event() {
            Ok(e) => e,
            Err(_) => return None, // malformed input — caller keeps the old file
        };
        match ev {
            Event::Eof => break,
            Event::Start(start) => {
                let ns = match xmlns_of(&start) {
                    Ok(ns) => ns,
                    Err(_) => return None,
                };
                scope.push(ns);
                let name = qname_str(start.name());
                let (iri, local) = resolve(name, &scope);
                // Special cases apply only OUTSIDE a buffered Description:
                // while one is open everything streams verbatim into its
                // buffer, so even pathological nesting round-trips instead of
                // being split between two writers.
                let special_ok = desc.is_none();
                let is_rdf_root = special_ok && iri.as_deref() == Some(RDF_NS) && local == "RDF";
                let is_desc = special_ok && iri.as_deref() == Some(RDF_NS) && local == "Description";
                if is_rdf_root {
                    saw_rdf = true;
                    rdf_depth = Some(depth);
                    rdf_prefix = name.split_once(':').map_or(name, |(p, _)| p).to_string();
                    let _ = w.write_event(Event::Start(start.to_owned()));
                    depth += 1;
                    continue;
                }
                if is_desc && desc.is_none() && rdf_depth == Some(depth.saturating_sub(1)) {
                    // Top-level Description directly under rdf:RDF: buffer it.
                    let mut attrs: Vec<RawAttr> = Vec::new();
                    for a in start.attributes() {
                        let a = a.ok()?;
                        attrs.push(RawAttr(a.key.as_ref().to_vec(), a.value.to_vec()));
                    }
                    let mut prefixes: [Option<String>; NS_TABLE.len()] =
                        std::array::from_fn(|_| None);
                    for RawAttr(k, v) in &attrs {
                        let (Ok(k), Ok(v)) =
                            (std::str::from_utf8(k), std::str::from_utf8(v))
                        else {
                            continue;
                        };
                        let Some(p) = k.strip_prefix("xmlns:") else {
                            continue;
                        };
                        for (i, (uri, _)) in NS_TABLE.iter().enumerate() {
                            if v == *uri && prefixes[i].is_none() {
                                prefixes[i] = Some(p.to_string());
                            }
                        }
                    }
                    // Owned properties also arrive in ATTRIBUTE form — exiftool
                    // writes `xmp:Rating="3"` as one, and leaving it beside our
                    // replacement element would publish two different ratings
                    // in one Description. Unprefixed attributes have no
                    // namespace in XML, so they can never be ours. Note the
                    // scan covers only a TOP-level Description directly under
                    // rdf:RDF — while one is buffered everything else streams
                    // verbatim into it (`special_ok`), so a pathological nested
                    // structure round-trips instead of being rewritten.
                    let mut owned_attr = false;
                    attrs.retain(|RawAttr(k, _)| {
                        let Ok(k) = std::str::from_utf8(k) else {
                            return true;
                        };
                        let Some((p, local)) = k.split_once(':') else {
                            return true;
                        };
                        if p == "xmlns" {
                            return true; // a declaration, not a property
                        }
                        let iri = scope.iter().rev().find_map(|m| m.get(p).cloned());
                        let owned = payload.owns(iri.as_deref(), local);
                        owned_attr |= owned;
                        !owned
                    });
                    desc = Some(DescBuf {
                        qname: name.to_string(),
                        attrs,
                        prefixes,
                        children: Writer::new(Vec::new()),
                        has_target: owned_attr,
                    });
                    desc_depth = depth;
                    depth += 1;
                    continue;
                }
                if let Some(buf) = desc.as_mut() {
                    if payload.owns(iri.as_deref(), local) {
                        buf.has_target = true;
                        if !skip_subtree(&mut reader, &mut scope, &mut depth) {
                            return None; // truncated input
                        }
                        scope.pop(); // the skipped Start's own binding
                        continue;
                    }
                    let _ = buf.children.write_event(Event::Start(start.to_owned()));
                } else {
                    let _ = w.write_event(Event::Start(start.to_owned()));
                }
                depth += 1;
            }
            Event::End(end) => {
                let name = qname_str(end.name());
                let (iri, local) = resolve(name, &scope);
                let my_depth = depth;
                depth = depth.saturating_sub(1);
                // An element opened when depth was D has its End arrive at
                // D+1 (the Start arm increments after recording).
                // The injection decision precedes writing this close so the
                // fresh Description lands INSIDE rdf:RDF. ISO 16684-1 allows
                // exactly one rdf:RDF under x:xmpmeta and consumers parse only
                // that subtree — a sibling block would be silently ignored by
                // every other tool, and on the next sync it would not count as
                // ours, multiplying copies forever.
                let inject = !desc.is_some()
                    && iri.as_deref() == Some(RDF_NS)
                    && local == "RDF"
                    && saw_rdf
                    && !props_emitted
                    && payload.emits_anything();
                if desc.is_some() && my_depth == desc_depth + 1 {
                    // Closes the buffered Description: emit it (rewritten).
                    let buf = desc.take().unwrap();
                    let owns_props = buf.has_target && !props_emitted;
                    if owns_props {
                        props_emitted = true;
                    }
                    let kids = buf.children.into_inner();
                    let mut rebuilt = BytesStart::new(buf.qname.clone());
                    for RawAttr(k, v) in &buf.attrs {
                        rebuilt.push_attribute((k.as_slice(), v.as_slice()));
                    }
                    // Declare every namespace we emit into that the element
                    // does not already bind. Collected first: `push_attribute`
                    // borrows its bytes.
                    let decls: Vec<(String, &str)> = if owns_props {
                        NS_TABLE
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| payload.uses_ns(*i) && buf.prefixes[*i].is_none())
                            .map(|(i, (_, dp))| (format!("xmlns:{dp}"), NS_TABLE[i].0))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    for (k, v) in &decls {
                        rebuilt.push_attribute((k.as_str(), *v));
                    }
                    let _ = w.write_event(Event::Start(rebuilt));
                    w.get_mut().extend_from_slice(&kids);
                    if owns_props {
                        let mut inner = Vec::new();
                        {
                            let mut iw = Writer::new(&mut inner);
                            let prefixes: Vec<Option<&str>> =
                                buf.prefixes.iter().map(|p| p.as_deref()).collect();
                            // Containers follow the document's own RDF prefix;
                            // the buffered Description sits inside rdf:RDF,
                            // whose start tag binds it.
                            payload.emit(&mut iw, &rdf_prefix, &prefixes);
                        }
                        w.get_mut().extend_from_slice(&inner);
                    }
                    let _ = w.write_event(Event::End(BytesEnd::new(buf.qname)));
                } else if let Some(buf) = desc.as_mut() {
                    let _ = buf.children.write_event(Event::End(end.to_owned()));
                } else {
                    if inject {
                        props_emitted = true;
                        let bytes = injected_description(&rdf_prefix, payload);
                        w.get_mut().extend_from_slice(&bytes);
                    }
                    let _ = w.write_event(Event::End(end.to_owned()));
                }
                scope.pop();
            }
            other => {
                let owned = other.into_owned();
                if let Some(buf) = desc.as_mut() {
                    let _ = buf.children.write_event(owned);
                } else {
                    let _ = w.write_event(owned);
                }
            }
        }
    }
    // Refuse anything we did not fully understand: no RDF root at all, a
    // Description still open, or elements still open at EOF. quick-xml emits
    // a CLEAN Eof for input truncated exactly at an event boundary, so depth
    // is what catches e.g. a packet missing its closing wrapper.
    if !saw_rdf || desc.is_some() || depth != 0 {
        return None;
    }
    Some(w.into_inner())
}

/// A standalone Description with the payload's properties, injected into an
/// existing document using THAT document's RDF prefix. Every namespace the
/// payload writes is declared with its default prefix — a fresh Description
/// cannot rely on bindings declared elsewhere in scope.
fn injected_description(rdf_prefix: &str, payload: &Payload) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(512);
    let mut w = Writer::new(&mut out);
    let about = format!("{rdf_prefix}:about");
    let mut el = w
        .create_element(format!("{rdf_prefix}:Description"))
        .with_attribute((about.as_str(), ""));
    for (i, (uri, dp)) in NS_TABLE.iter().enumerate() {
        if payload.uses_ns(i) {
            el = el.with_attribute((format!("xmlns:{dp}").as_str(), *uri));
        }
    }
    let _ = el.write_inner_content(|d| {
        let defaults: Vec<Option<&str>> = NS_TABLE.iter().map(|(_, dp)| Some(*dp)).collect();
        // The injection sits inside the document's rdf:RDF, whose start tag
        // binds `rdf_prefix` — containers may follow it.
        payload.emit(d, rdf_prefix, &defaults);
        Ok::<(), std::io::Error>(())
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(t: &str, d: &str, c: &str, p: &str, r: &str) -> Payload {
        Payload::Metadata(XmpValues {
            title: t.into(),
            description: d.into(),
            creator: c.into(),
            publisher: p.into(),
            rights: r.into(),
        })
    }

    /// Rating/colour-label payload shorthands for the m4-153 tests.
    fn rl(rating: Option<u8>, colors: Option<u8>) -> Payload {
        Payload::RatingLabels { rating, colors }
    }

    /// Parse output and pull the FULL TEXT of the first element whose LOCAL
    /// name is `prop_local` (namespace-blind — adequate for assertions). Each
    /// element owns exactly the text inside its own span, bubbled to the
    /// parent on close, so sibling properties cannot contaminate each other.
    fn text_of(doc: &[u8], prop_local: &str) -> Option<String> {
        let mut reader = Reader::from_reader(doc);
        reader.config_mut().expand_empty_elements = true;
        // (local name, text accumulated within this element's span)
        let mut stack: Vec<(String, String)> = Vec::new();
        let mut found: Option<String> = None;
        loop {
            match reader.read_event() {
                Ok(Event::Start(s)) => {
                    let name = qname_str(s.name()).rsplit(':').next().unwrap().to_string();
                    stack.push((name, String::new()));
                }
                Ok(Event::End(_)) => {
                    let (l, txt) = stack.pop()?;
                    if l == prop_local && found.is_none() {
                        found = Some(txt);
                    } else if let Some((_, parent)) = stack.last_mut() {
                        parent.push_str(&txt);
                    }
                }
                Ok(Event::Text(t)) => {
                    if let Some((_, top)) = stack.last_mut() {
                        top.push_str(&t.xml_content().ok()?);
                    }
                }
                Ok(Event::CData(t)) => {
                    if let Some((_, top)) = stack.last_mut() {
                        top.push_str(std::str::from_utf8(t.into_inner().as_ref()).ok()?);
                    }
                }
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(_) => return None,
            }
        }
        found.filter(|s| !s.is_empty())
    }

    /// An exiftool-shaped sidecar: foreign namespaces, attribute-valued props,
    /// two of our targets present, packet PIs around everything.
    const DT_LIKE: &str = concat!(
        r#"<?xpacket begin="""" id="W5M0MpCehiHzreSzNTczkc9d"?>"#,
        "\n",
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Image::ExifTool 12.76">"#,
        "\n",
        r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
        "\n\n ",
        r#"<rdf:Description rdf:about="""#,
        "\n",
        r#"  xmlns:dc="http://purl.org/dc/elements/1.1/""#,
        "\n",
        r#"  xmlns:exif="http://ns.adobe.com/exif/1.0/""#,
        "\n",
        r#"  exif:DateTimeOriginal="2024:07:01 10:00:00""#,
        "\n",
        r#"  xmp:Rating="3">"#,
        "\n  ",
        r#"<dc:title><rdf:Alt><rdf:li xml:lang="x-default">OLD TITLE</rdf:li></rdf:Alt></dc:title>"#,
        "\n  ",
        r#"<dc:description><rdf:Alt><rdf:li xml:lang="x-default">kept description</rdf:li></rdf:Alt></dc:description>"#,
        "\n  ",
        r#"<exif:Flash><exif:Fired>False</exif:Fired></exif:Flash>"#,
        "\n ",
        r#"</rdf:Description>"#,
        "\n",
        r#"</rdf:RDF>"#,
        "\n",
        r#"</x:xmpmeta>"#,
        "\n",
        r#"<?xpacket end="w"?>"#,
    );

    #[test]
    fn fresh_packet_carries_all_five_fields_in_dt_shapes() {
        let doc = fresh_document(&vals("T", "D", "C", "P", "R"));
        assert_eq!(text_of(&doc, "title").as_deref(), Some("T"));
        assert_eq!(text_of(&doc, "description").as_deref(), Some("D"));
        assert_eq!(text_of(&doc, "creator").as_deref(), Some("C"));
        assert_eq!(text_of(&doc, "publisher").as_deref(), Some("P"));
        assert_eq!(text_of(&doc, "rights").as_deref(), Some("R"));
        // Language alternative / sequence / bag shapes.
        let s = String::from_utf8_lossy(&doc);
        assert!(s.contains(r#"<dc:title><rdf:Alt><rdf:li xml:lang="x-default">"#));
        assert!(s.contains(r#"<dc:creator><rdf:Seq><rdf:li>"#));
        assert!(s.contains(r#"<dc:publisher><rdf:Bag><rdf:li>"#));
        // Values are XML-escaped.
        let esc = fresh_document(&vals("a<b>&c", "", "", "", ""));
        assert!(String::from_utf8_lossy(&esc).contains("a&lt;b&gt;&amp;c"));
    }

    #[test]
    fn merge_updates_targets_and_preserves_everything_else() {
        // Catalogue wins (`dt_image_synch_xmps` direction): every target
        // property is re-emitted from the given values, sidecar originals
        // dropped — including a target the sidecar had but the catalogue
        // leaves blank. Only genuinely foreign content must survive.
        let out =
            rewrite_document(DT_LIKE, &vals("NEW", "NEWDESC", "", "", "")).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(s.contains(">NEW</"), "title replaced");
        assert!(!s.contains("OLD TITLE"), "old title gone");
        assert_eq!(text_of(&out, "title").as_deref(), Some("NEW"));
        assert_eq!(text_of(&out, "description").as_deref(), Some("NEWDESC"));
        assert!(
            s.contains("exif:DateTimeOriginal"),
            "attribute-valued foreign props kept"
        );
        assert!(
            s.contains("<exif:Fired>False</exif:Fired>"),
            "foreign subtrees kept"
        );
        assert!(s.contains("x:xmptk=\"Image::ExifTool"), "wrapper attributes kept");
        assert!(s.contains("<?xpacket begin="), "packet PIs kept");
        assert!(s.contains("<?xpacket end=\"w\"?>"), "closing PI kept");
    }

    #[test]
    fn merge_blank_value_deletes_the_property() {
        let out = rewrite_document(DT_LIKE, &vals("", "", "", "", "")).expect("parses");
        assert!(text_of(&out, "title").is_none(), "blank deletes");
        assert!(text_of(&out, "description").is_none(), "blank deletes");
        assert!(
            text_of(&out, "publisher").is_none(),
            "never-present stays absent"
        );
        assert_eq!(
            text_of(&out, "Fired").as_deref(),
            Some("False"),
            "foreign leaf untouched"
        );
        assert!(
            String::from_utf8_lossy(&out).contains("exif:DateTimeOriginal"),
            "foreign content survives a full wipe"
        );
    }

    #[test]
    fn merge_adds_our_fields_when_no_block_has_them() {
        let bare = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"><photoshop:City>Lyon</photoshop:City></rdf:Description></rdf:RDF></x:xmpmeta>"#;
        let out = rewrite_document(bare, &vals("T2", "", "C2", "", "")).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(s.contains("<photoshop:City>Lyon</photoshop:City>"));
        assert_eq!(text_of(&out, "title").as_deref(), Some("T2"));
        assert_eq!(text_of(&out, "creator").as_deref(), Some("C2"));
        assert!(s.contains("xmlns:dc="), "DC namespace declared on the injection");
        // The injected Description must sit INSIDE rdf:RDF — ISO 16684-1
        // allows exactly one child there and consumers ignore anything else.
        assert!(
            s.contains("</rdf:Description></rdf:RDF></x:xmpmeta>"),
            "injection landed outside rdf:RDF: {s}"
        );
        // And a second sync must be byte-identical: the stray-sibling bug of
        // the first draft multiplied copies on every save.
        let again = rewrite_document(
            std::str::from_utf8(&out).unwrap(),
            &vals("T2", "", "C2", "", ""),
        )
        .expect("reparse");
        assert_eq!(again, out, "sync not idempotent");
    }

    #[test]
    fn malformed_attributes_are_refused_not_silently_dropped() {
        // A value-less attribute is a parse error; rewriting would quietly
        // normalise a file we did not fully understand.
        let bad = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:exif="http://ns.adobe.com/exif/1.0/"><rdf:Description rdf:about="" exif:Flash/></rdf:RDF></x:xmpmeta>"#;
        assert!(rewrite_document(bad, &vals("X", "", "", "", "")).is_none());
        // Same for a malformed xmlns declaration on the buffered path.
        let bad_ns = "<x:xmpmeta xmlns:x><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\"/></rdf:RDF></x:xmpmeta>";
        assert!(rewrite_document(bad_ns, &vals("X", "", "", "", "")).is_none());
    }

    #[test]
    fn packet_truncated_at_a_clean_boundary_is_refused() {
        // quick-xml reports Eof without error when elements remain open; only
        // the depth guard catches this class of damage.
        let full = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title><rdf:Alt xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:li xml:lang="x-default">A</rdf:li></rdf:Alt></dc:title></rdf:Description></rdf:RDF></x:xmpmeta>"#;
        assert!(rewrite_document(full, &vals("N", "", "", "", "")).is_some());
        let cut = full.strip_suffix("</x:xmpmeta>").unwrap();
        assert!(rewrite_document(cut, &vals("N", "", "", "", "")).is_none());
        let cut2 = cut.strip_suffix("</rdf:RDF>").unwrap();
        assert!(rewrite_document(cut2, &vals("N", "", "", "", "")).is_none());
    }

    #[test]
    fn language_alternatives_collapse_to_x_default() {
        let multi = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title><rdf:Alt xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:li xml:lang="x-default">EN</rdf:li><rdf:li xml:lang="de">DE</rdf:li><rdf:li xml:lang="fr">FR</rdf:li></rdf:Alt></dc:title></rdf:Description></rdf:RDF></x:xmpmeta>"#;
        let out = rewrite_document(multi, &vals("ONLY", "", "", "", "")).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(!s.contains(">DE<") && !s.contains(">FR<"), "translations dropped: {s}");
        assert_eq!(s.matches("xml:lang").count(), 1);
        assert_eq!(text_of(&out, "title").as_deref(), Some("ONLY"));
    }

    #[test]
    fn xml_illegal_control_characters_are_replaced_not_embedded() {
        let doc = fresh_document(&vals("a\u{1}b\u{7}c", "", "", "", ""));
        let s = String::from_utf8_lossy(&doc);
        assert!(
            !s.contains('\u{1}') && !s.contains('\u{7}'),
            "control bytes leaked: {s:?}"
        );
        assert!(
            s.contains(r#"a"#) && s.contains("\u{fffd}"),
            "replacement character expected: {s:?}"
        );
    }

    #[test]
    fn merge_reuses_foreign_prefixes_for_both_dc_and_rdf() {
        // Dublin Core bound to `dcq`, RDF bound to `r`: recognition and output
        // must follow the bindings, not literal spellings.
        let odd = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><r:RDF xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><r:Description r:about="" xmlns:dcq="http://purl.org/dc/elements/1.1/"><dcq:title><rdf:Alt xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:li xml:lang="x-default">OLD</rdf:li></rdf:Alt></dcq:title></r:Description></r:RDF></x:xmpmeta>"#;
        let out = rewrite_document(odd, &vals("FIX", "", "", "", "")).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(s.contains(">FIX</"), "value updated");
        assert!(
            !s.contains("<dc:title>"),
            "canonical prefix NOT introduced when a binding exists"
        );
        assert!(s.contains("<dcq:title>"), "document's own DC prefix reused");
        assert!(s.contains("<r:Description"), "document's own RDF prefix kept");
    }

    #[test]
    fn two_blocks_holding_targets_consolidate_without_duplication() {
        let split = r##"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title><rdf:Alt xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:li xml:lang="x-default">A</rdf:li></rdf:Alt></dc:title></rdf:Description><rdf:Description rdf:about="#2" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:rights><rdf:Alt xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:li xml:lang="x-default">(c) old</rdf:li></rdf:Alt></dc:rights></rdf:Description></rdf:RDF></x:xmpmeta>"##;
        let out = rewrite_document(split, &vals("NA", "", "", "", "NR")).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert_eq!(s.matches("<dc:title>").count(), 1, "one title total");
        assert_eq!(s.matches("<dc:rights>").count(), 1, "one rights total");
        assert_eq!(text_of(&out, "title").as_deref(), Some("NA"));
        assert_eq!(text_of(&out, "rights").as_deref(), Some("NR"));
    }

    #[test]
    fn malformed_existing_sidecars_are_rejected_not_rewritten() {
        let v = vals("X", "", "", "", "");
        assert!(
            rewrite_document("<not-rdf><dc:title>x</dc:title></not-rdf>", &v).is_none(),
            "no rdf:RDF at all"
        );
        assert!(rewrite_document("", &v).is_none(), "empty input");
        assert!(
            rewrite_document(
                "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\">",
                &v
            )
            .is_none(),
            "unclosed Description rejected"
        );
        assert!(
            rewrite_document(
                "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><broken",
                &v
            )
            .is_none(),
            "parse error rejected"
        );
    }

    #[test]
    fn comments_and_cdata_inside_foreign_blocks_survive() {
        let rich = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><!-- provenance note --><rdf:Description rdf:about="" xmlns:foo="urn:foo"><foo:blob><![CDATA[raw <stuff> & things]]></foo:blob></rdf:Description></rdf:RDF></x:xmpmeta>"#;
        let out = rewrite_document(rich, &vals("T", "", "", "", "")).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(s.contains("<!-- provenance note -->"));
        assert!(s.contains("<![CDATA[raw <stuff> & things]]>"));
    }

    #[test]
    fn empty_values_never_create_a_packet_and_sidecars_follow_darktable_naming() {
        assert_eq!(sidecar_path("/photos/IMG_0001.ORF"), "/photos/IMG_0001.ORF.xmp");
        assert_eq!(sidecar_path("a.JPG"), "a.JPG.xmp");
        // all-empty + no existing file → quiet success without creating one.
        let dir = std::env::temp_dir().join(format!("c41_xmpempty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let img = dir.join("e.jpg");
        std::fs::write(&img, b"x").unwrap();
        let db = dir.join("lib.db");
        let conn = rusqlite::Connection::open(&db).unwrap();
        // Folder must resolve for the fail-closed catalogue lookup (m4-142
        // review): a sync can no longer confuse "unreadable" with "blank".
        conn.execute_batch(&format!(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT);
             CREATE TABLE meta_data (id INTEGER, key INTEGER, value TEXT);
             INSERT INTO film_rolls (folder) VALUES ('{}');
             INSERT INTO images (film_id, filename) VALUES (1, 'e.jpg');",
            dir.to_str().unwrap().replace('\'', "''")
        ))
        .unwrap();
        drop(conn);
        let img_s = img.to_str().unwrap();
        let db_s = db.to_str().unwrap();
        assert!(sync_sidecar(db_s, img_s));
        assert!(!std::path::Path::new(&sidecar_path(img_s)).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sync_round_trips_through_the_real_fs_and_never_clobbers_malformed() {
        let dir = std::env::temp_dir().join(format!("c41_xmp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let img = dir.join("img.jpg");
        std::fs::write(&img, b"jpeg-bytes").unwrap();

        // A catalogue mirroring persist's schema (film_rolls/images/meta_data).
        let db = dir.join("library.db");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(&format!(
            "CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT);
             CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, filename TEXT);
             CREATE TABLE meta_data (id INTEGER, key INTEGER, value TEXT);
             INSERT INTO film_rolls (folder) VALUES ('{}');
             INSERT INTO images (film_id, filename) VALUES (1, 'img.jpg');",
            dir.to_str().unwrap().replace('\'', "''")
        ))
        .unwrap();
        conn.execute(
            "INSERT INTO meta_data (id, key, value) VALUES (1, 2, 'Real Title')",
            [],
        )
        .unwrap();
        let db_s = db.to_str().unwrap();
        let img_s = img.to_str().unwrap();
        let sp = dir.join("img.jpg.xmp");

        // Missing image on disk: quiet success, no sidecar anywhere.
        assert!(sync_sidecar(db_s, dir.join("gone.jpg").to_str().unwrap()));
        // First real sync CREATES the packet from catalogue contents.
        assert!(sync_sidecar(db_s, img_s));
        let first = std::fs::read_to_string(&sp).unwrap();
        assert!(first.contains(">Real Title<"), "{first}");
        // Second sync is idempotent.
        assert!(sync_sidecar(db_s, img_s));
        assert_eq!(std::fs::read_to_string(&sp).unwrap(), first);
        // Merge phase, catalogue-wins like dt_image_synch_xmps: an exiftool
        // sidecar with its own title/description gets rewritten from the
        // catalogue; only non-target content survives.
        std::fs::write(&sp, DT_LIKE).unwrap();
        conn.execute("UPDATE meta_data SET value='Edited' WHERE key=2", [])
            .unwrap();
        assert!(sync_sidecar(db_s, img_s));
        let merged = std::fs::read_to_string(&sp).unwrap();
        assert!(
            merged.contains("exif:DateTimeOriginal"),
            "foreign attributes preserved: {merged}"
        );
        assert!(merged.contains("<exif:Fired>False</exif:Fired>"));
        assert!(!merged.contains("OLD TITLE"), "sidecar title superseded");
        assert!(
            !merged.contains("kept description"),
            "blank catalogue field deletes sidecar value"
        );
        assert!(merged.contains(">Edited<"), "catalogue edit landed");
        // Malformed sidecar is left byte-identical and reported.
        std::fs::write(&sp, "<definitely not rdf").unwrap();
        let before = std::fs::read(&sp).unwrap();
        assert!(!sync_sidecar(db_s, img_s));
        assert_eq!(std::fs::read(&sp).unwrap(), before, "malformed sidecar untouched");

        drop(conn);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── m4-153: rating / colour-label sync ────────────────────────────────

    /// A darktable-shaped Description declaring the canonical xmp/darktable
    /// prefixes, carrying one rating and one colour label.
    const RL_DOC: &str = concat!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
        r#"<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:darktable="http://darktable.sf.net/">"#,
        r#"<xmp:Rating>2</xmp:Rating>"#,
        r#"<darktable:colorlabels><rdf:Seq><rdf:li>0</rdf:li></rdf:Seq></darktable:colorlabels>"#,
        r#"</rdf:Description></rdf:RDF></x:xmpmeta>"#
    );

    #[test]
    fn rating_sync_replaces_element_form_in_place() {
        let out = rewrite_document(RL_DOC, &rl(Some(4), None)).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(s.contains("<xmp:Rating>4</xmp:Rating>"), "{s}");
        assert!(!s.contains(">2<"), "old rating gone: {s}");
        assert!(
            s.contains(r#"<darktable:colorlabels><rdf:Seq><rdf:li>0</rdf:li></rdf:Seq></darktable:colorlabels>"#),
            "labels untouched by a rating-only sync: {s}"
        );
        assert_eq!(s.matches("<xmp:Rating>").count(), 1, "exactly one rating");
        // Idempotent: the second run must be byte-identical.
        let again =
            rewrite_document(std::str::from_utf8(&out).unwrap(), &rl(Some(4), None))
                .expect("reparse");
        assert_eq!(again, out);
    }

    #[test]
    fn rating_attribute_form_from_exiftool_is_dropped_not_duplicated() {
        // Real exiftool shape: Rating arrives as an ATTRIBUTE (`xmp:Rating="3"`)
        // with the prefix declared on the Description. Dropping it matters —
        // leaving it beside our replacement element would publish two
        // different ratings in one Description.
        let exiftoolish = concat!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Image::ExifTool 12.76">"#,
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
            r#"<rdf:Description rdf:about="" xmlns:exif="http://ns.adobe.com/exif/1.0/" xmlns:xmp="http://ns.adobe.com/xap/1.0/""#,
            r#" exif:DateTimeOriginal="2024:07:01 10:00:00" xmp:Rating="3">"#,
            r#"<exif:Flash><exif:Fired>False</exif:Fired></exif:Flash>"#,
            r#"</rdf:Description></rdf:RDF></x:xmpmeta>"#
        );
        let out = rewrite_document(exiftoolish, &rl(Some(4), None)).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(
            !s.contains("xmp:Rating="),
            "attribute-form rating dropped: {s}"
        );
        assert!(s.contains("<xmp:Rating>4</xmp:Rating>"), "{s}");
        assert!(
            s.contains("exif:DateTimeOriginal="),
            "foreign attributes survive"
        );
        assert!(s.contains("<exif:Fired>False</exif:Fired>"), "{s}");
        // The replacement lands INSIDE the Description that carried the
        // dropped attribute, not in a stray sibling.
        assert_eq!(s.matches("<xmp:Rating>").count(), 1);
        assert_eq!(s.matches("<rdf:Description").count(), 1);
        assert_eq!(text_of(&out, "Rating").as_deref(), Some("4"));
    }

    #[test]
    fn undeclared_prefix_is_not_provably_ours_and_is_left_alone() {
        // Same attribute but with NO xmlns binding anywhere: we cannot resolve
        // it to a namespace, so owning-by-local-name could delete a foreign
        // property. The safe direction (module doc) is skip-and-preserve —
        // our replacement goes into its own Description.
        let orphaned = concat!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">"#,
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
            r#"<rdf:Description rdf:about="" xmp:Rating="3"></rdf:Description>"#,
            r#"</rdf:RDF></x:xmpmeta>"#
        );
        let out = rewrite_document(orphaned, &rl(Some(4), None)).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(s.contains(r#"xmp:Rating="3""#), "{s}");
        assert!(s.contains("<xmp:Rating>4</xmp:Rating>"), "{s}");
    }

    #[test]
    fn colour_sync_replaces_seq_and_zero_mask_deletes() {
        let out = rewrite_document(RL_DOC, &rl(None, Some(0b0_1010))).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(
            s.contains("<xmp:Rating>2</xmp:Rating>"),
            "unowned rating untouched: {s}"
        );
        assert!(!s.contains("<rdf:li>0<"), "old label list replaced: {s}");
        assert!(s.contains("<rdf:li>1</rdf:li>") && s.contains("<rdf:li>3</rdf:li>"));
        assert_eq!(s.matches("<rdf:li>").count(), 2, "mask bits become items");
        // Idempotent: a second identical sync must be byte-identical.
        let again =
            rewrite_document(std::str::from_utf8(&out).unwrap(), &rl(None, Some(0b0_1010)))
                .expect("reparse");
        assert_eq!(again, out);

        // Mask 0 = "no labels", and darktable writes the property only when
        // at least one is set — so zero DELETES rather than emitting an
        // empty Seq.
        let out = rewrite_document(RL_DOC, &rl(None, Some(0))).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(!s.contains("colorlabels"), "zero mask deletes: {s}");
        assert!(s.contains("<xmp:Rating>2</xmp:Rating>"), "rating kept: {s}");
    }

    #[test]
    fn rating_label_prefix_bindings_are_honoured_like_dc_was() {
        let odd = concat!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
            r#"<rdf:Description rdf:about="" xmlns:ns1="http://ns.adobe.com/xap/1.0/" xmlns:dtx="http://darktable.sf.net/">"#,
            r#"<ns1:Rating>1</ns1:Rating><dtx:colorlabels><rdf:Seq><rdf:li>2</rdf:li></rdf:Seq></dtx:colorlabels>"#,
            r#"</rdf:Description></rdf:RDF></x:xmpmeta>"#
        );
        let out = rewrite_document(odd, &rl(Some(5), Some(0b0_0100))).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(s.contains("<ns1:Rating>5</ns1:Rating>"), "{s}");
        assert!(s.contains("<dtx:colorlabels>"), "{s}");
        assert!(
            !s.contains("<xmp:Rating") && !s.contains("<darktable:"),
            "canonical prefixes NOT introduced when bindings exist: {s}"
        );
    }

    #[test]
    fn rating_only_sync_preserves_dc_fields_in_a_combined_description() {
        // One Description carrying DC fields AND rating AND labels (what a
        // metadata-editor save followed by star/label clicks converges to):
        // the RL payload owns neither namespace's DC neighbours, so a
        // rating-only sync must leave every DC field exactly as found.
        let combo = concat!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
            r#"<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/""#,
            r#" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:darktable="http://darktable.sf.net/">"#,
            r#"<dc:title><rdf:Alt><rdf:li xml:lang="x-default">Kept</rdf:li></rdf:Alt></dc:title>"#,
            r#"<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">(c) me</rdf:li></rdf:Alt></dc:rights>"#,
            r#"<xmp:Rating>1</xmp:Rating>"#,
            r#"<darktable:colorlabels><rdf:Seq><rdf:li>2</rdf:li></rdf:Seq></darktable:colorlabels>"#,
            r#"</rdf:Description></rdf:RDF></x:xmpmeta>"#
        );
        let out = rewrite_document(combo, &rl(Some(5), None)).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert_eq!(text_of(&out, "title").as_deref(), Some("Kept"), "{s}");
        assert_eq!(text_of(&out, "rights").as_deref(), Some("(c) me"), "{s}");
        assert_eq!(text_of(&out, "Rating").as_deref(), Some("5"), "{s}");
        assert!(
            s.contains(r#"<darktable:colorlabels><rdf:Seq><rdf:li>2</rdf:li></rdf:Seq>"#),
            "labels untouched: {s}"
        );
        assert_eq!(s.matches("<xmp:Rating>").count(), 1);
    }

    #[test]
    fn injected_containers_follow_the_documents_rdf_prefix() {
        // RDF bound to `r`, never spelled `rdf` anywhere: an injection into
        // this document must emit `<r:Seq>/<r:li>` — a literal `rdf:` would
        // reference an undeclared prefix and produce ill-formed XML.
        let odd = concat!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><r:RDF xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
            r#"<r:Description r:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">"#,
            r#"<xmp:Rating>1</xmp:Rating></r:Description></r:RDF></x:xmpmeta>"#
        );
        let out = rewrite_document(odd, &rl(None, Some(0b0_0010))).expect("parses");
        let s = String::from_utf8_lossy(&out);
        assert!(
            s.contains("<darktable:colorlabels><r:Seq><r:li>1</r:li></r:Seq></darktable:colorlabels>"),
            "{s}"
        );
        assert!(
            !s.contains("rdf:"),
            "no undeclared rdf: prefix may leak: {s}"
        );
        assert!(s.contains("xmlns:darktable="), "injection declares what it writes: {s}");
        // And it round-trips: the second sync finds the injected block.
        let again =
            rewrite_document(std::str::from_utf8(&out).unwrap(), &rl(None, Some(0b0_0010)))
                .expect("reparse");
        assert_eq!(again, out);
    }

    #[test]
    fn non_emitting_payloads_leave_foreign_documents_alone() {
        let bare = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"><photoshop:City>Lyon</photoshop:City></rdf:Description></rdf:RDF></x:xmpmeta>"#;
        // Neither field, or a zero mask that owns the label property but emits
        // nothing for it: neither may inject an empty Description or declare
        // namespaces nothing uses (the "phantom inject" of the first draft).
        for payload in [&rl(None, None), &rl(None, Some(0))] {
            let out = rewrite_document(bare, payload).expect("parses");
            let s = String::from_utf8_lossy(&out);
            assert!(s.contains("<photoshop:City>Lyon</photoshop:City>"), "{s}");
            assert!(!s.contains("Rating") && !s.contains("colorlabels"), "{s}");
            assert!(
                !s.contains("xmlns:darktable") && !s.contains("xmlns:xmp"),
                "no unused declaration: {s}"
            );
            assert_eq!(
                s.matches("<rdf:Description").count(),
                1,
                "nothing injected: {s}"
            );
        }
    }

    #[test]
    fn fresh_packet_declares_only_namespaces_it_writes() {
        let doc = fresh_document(&rl(Some(3), Some(0b0_0010)));
        let s = String::from_utf8_lossy(&doc);
        assert!(s.contains("xmlns:xmp="), "{s}");
        assert!(s.contains("xmlns:darktable="), "{s}");
        assert!(!s.contains("xmlns:dc="), "unused DC must not be declared: {s}");
        assert!(s.contains("<xmp:Rating>3</xmp:Rating>"), "{s}");
        assert!(s.contains(r#"<darktable:colorlabels><rdf:Seq><rdf:li>1</rdf:li></rdf:Seq>"#));
    }

    #[test]
    fn rating_label_sidecar_round_trips_through_the_real_fs() {
        let dir = std::env::temp_dir().join(format!("c41_xmprl_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let img = dir.join("img.jpg");
        std::fs::write(&img, b"jpeg-bytes").unwrap();
        let img_s = img.to_str().unwrap();
        let sp = dir.join("img.jpg.xmp");

        // Missing image on disk: quiet success, nothing anywhere.
        assert!(sync_rating_labels_sidecar_unlocked(
            dir.join("gone.jpg").to_str().unwrap(),
            Some(3),
            None
        ));
        // Nothing to write: success WITHOUT creating a sidecar.
        assert!(sync_rating_labels_sidecar_unlocked(img_s, None, None));
        assert!(!sp.exists());
        // First rating sync CREATES the packet, declaring exactly what it writes.
        assert!(sync_rating_labels_sidecar_unlocked(img_s, Some(3), None));
        let first = std::fs::read_to_string(&sp).unwrap();
        assert!(first.contains("<xmp:Rating>3</xmp:Rating>"), "{first}");
        assert!(first.contains("xmlns:xmp="), "{first}");
        assert!(!first.contains("xmlns:darktable="), "{first}");
        // Second identical sync is idempotent.
        assert!(sync_rating_labels_sidecar_unlocked(img_s, Some(3), None));
        assert_eq!(std::fs::read_to_string(&sp).unwrap(), first);
        // Labels merge into the same document alongside the rating.
        assert!(sync_rating_labels_sidecar_unlocked(img_s, Some(4), Some(0b1_0001)));
        let merged = std::fs::read_to_string(&sp).unwrap();
        assert!(merged.contains("<xmp:Rating>4</xmp:Rating>"), "{merged}");
        assert!(merged.contains("<rdf:li>0</rdf:li>"), "{merged}");
        assert!(merged.contains("<rdf:li>4</rdf:li>"), "{merged}");
        assert!(merged.contains("xmlns:darktable="), "{merged}");
        // Clearing every label deletes the property but keeps the rating.
        assert!(sync_rating_labels_sidecar_unlocked(img_s, Some(4), Some(0)));
        let cleared = std::fs::read_to_string(&sp).unwrap();
        assert!(!cleared.contains("colorlabels"), "{cleared}");
        assert!(cleared.contains("<xmp:Rating>4</xmp:Rating>"), "{cleared}");
        // Last label off with NO sidecar present at all: quiet success and
        // never a phantom packet (the missing-file branch gates on emitting).
        std::fs::remove_file(&sp).unwrap();
        assert!(sync_rating_labels_sidecar_unlocked(img_s, None, Some(0)));
        assert!(!sp.exists());
        // Malformed sidecar: reported false, left byte-identical.
        std::fs::write(&sp, "<definitely not rdf").unwrap();
        let before = std::fs::read(&sp).unwrap();
        assert!(!sync_rating_labels_sidecar_unlocked(img_s, Some(1), None));
        assert_eq!(std::fs::read(&sp).unwrap(), before);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
