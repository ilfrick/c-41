//! Collection **rule stack** (m4-134, parity 2.6 slice 2; numeric properties
//! m4-135 slice 3) — darktable's filtering.c arbitrary rules: a list of rules
//! (property, comparator, value) joined by AND / OR / AND-NOT combinators,
//! composing on top of whatever collection is active exactly like the
//! single-state quick filters do.
//!
//! Properties span both families: textual (`Filename`, `Film roll`) matched by
//! substring, and numeric (`Exposure`/`Aperture`/`ISO`/`Focal length` — the
//! m4-135 nullable EXIF columns) matched by comparison. Numeric values accept
//! plain decimals or fractions (`1/60`); an unparseable numeric value makes the
//! rule inert rather than broken. NULL columns (no EXIF, pre-migration rows)
//! never satisfy a comparison, so unprobed images stay out of numeric-rule
//! results — recorded in PARITY_AUDIT as a deviation from darktable, whose
//! schema backfills zeros that would match.
//!
//! Everything decision-bearing is pure and unit-tested display-free:
//! [`rule_stack_sql`] (composition), [`like_literal`] (the injection boundary),
//! and the token round-trip ([`rule_stack_token_for`] /
//! [`parse_rule_stack_token`]).

/// Comparator for a rule. The first two serve textual properties, the rest
/// numeric ones; the global index space (what tokens persist) is stable across
/// the m4-135 extension — pre-existing stacks only ever stored 0/1 and decode
/// exactly as before.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuleCmp {
    Contains,
    Excludes,
    Lt,
    Lte,
    Eq,
    Gte,
    Gt,
}

impl RuleCmp {
    pub const ALL: [RuleCmp; 7] = [
        RuleCmp::Contains,
        RuleCmp::Excludes,
        RuleCmp::Lt,
        RuleCmp::Lte,
        RuleCmp::Eq,
        RuleCmp::Gte,
        RuleCmp::Gt,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RuleCmp::Contains => "contains",
            RuleCmp::Excludes => "excludes",
            RuleCmp::Lt => "<",
            RuleCmp::Lte => "≤",
            RuleCmp::Eq => "=",
            RuleCmp::Gte => "≥",
            RuleCmp::Gt => ">",
        }
    }

    /// Stable index for `DropDown` wiring / token persistence; unknown indices
    /// decode to [`RuleCmp::Contains`], the same lenient default the other
    /// tokens use.
    pub fn to_index(self) -> u32 {
        match self {
            RuleCmp::Contains => 0,
            RuleCmp::Excludes => 1,
            RuleCmp::Lt => 2,
            RuleCmp::Lte => 3,
            RuleCmp::Eq => 4,
            RuleCmp::Gte => 5,
            RuleCmp::Gt => 6,
        }
    }

    pub fn from_index(i: u32) -> RuleCmp {
        match i {
            1 => RuleCmp::Excludes,
            2 => RuleCmp::Lt,
            3 => RuleCmp::Lte,
            4 => RuleCmp::Eq,
            5 => RuleCmp::Gte,
            6 => RuleCmp::Gt,
            _ => RuleCmp::Contains,
        }
    }

    /// The comparator list one property KIND shows in its dropdown, in visible
    /// order. `collect` maps a dropdown position back through this slice, so a
    /// row's UI only ever offers comparators meaningful for its property.
    pub const TEXT_SET: [RuleCmp; 2] = [RuleCmp::Contains, RuleCmp::Excludes];
    pub const NUMERIC_SET: [RuleCmp; 5] = [
        RuleCmp::Lt,
        RuleCmp::Lte,
        RuleCmp::Eq,
        RuleCmp::Gte,
        RuleCmp::Gt,
    ];

    /// Position of this comparator within its kind's set (`None` if it belongs
    /// to the OTHER kind — used to sanitize state when a property flips kind
    /// under a stale comparator).
    pub fn position_in(self, set: &[RuleCmp]) -> Option<u32> {
        set.iter().position(|c| *c == self).map(|p| p as u32)
    }
}

/// Which comparator family a property's value space needs. Drives which
/// dropdown a rule row shows and how its predicate is built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PropertyKind {
    Text,
    Numeric,
}

/// The column (or pseudo-column) a rule matches. `FileName` is `i.filename`,
/// `FilmRoll` the roll's folder path `f.folder`; the m4-135 numeric four are
/// the nullable EXIF columns added by that increment — NULL (no EXIF / older
/// rows) simply never satisfies a comparison, so unprobed images stay out of
/// numeric-rule results rather than matching as 0.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuleProperty {
    FileName,
    FilmRoll,
    Exposure,
    Aperture,
    Iso,
    FocalLength,
}

impl RuleProperty {
    pub const ALL: [RuleProperty; 6] = [
        RuleProperty::FileName,
        RuleProperty::FilmRoll,
        RuleProperty::Exposure,
        RuleProperty::Aperture,
        RuleProperty::Iso,
        RuleProperty::FocalLength,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RuleProperty::FileName => "Filename",
            RuleProperty::FilmRoll => "Film roll",
            RuleProperty::Exposure => "Exposure",
            RuleProperty::Aperture => "Aperture",
            RuleProperty::Iso => "ISO",
            RuleProperty::FocalLength => "Focal length",
        }
    }

    /// Kind — selects the comparator dropdown + predicate shape.
    pub fn kind(self) -> PropertyKind {
        match self {
            RuleProperty::FileName | RuleProperty::FilmRoll => PropertyKind::Text,
            _ => PropertyKind::Numeric,
        }
    }

    /// Column reference valid inside any loader query (they all alias
    /// `images i JOIN film_rolls f`). Kept beside the variant so a new property
    /// can't forget its column.
    pub fn column(self) -> &'static str {
        match self {
            RuleProperty::FileName => "i.filename",
            RuleProperty::FilmRoll => "f.folder",
            RuleProperty::Exposure => "i.exposure",
            RuleProperty::Aperture => "i.aperture",
            RuleProperty::Iso => "i.iso",
            RuleProperty::FocalLength => "i.focal_length",
        }
    }

    pub fn to_index(self) -> u32 {
        match self {
            RuleProperty::FileName => 0,
            RuleProperty::FilmRoll => 1,
            RuleProperty::Exposure => 2,
            RuleProperty::Aperture => 3,
            RuleProperty::Iso => 4,
            RuleProperty::FocalLength => 5,
        }
    }

    pub fn from_index(i: u32) -> RuleProperty {
        match i {
            1 => RuleProperty::FilmRoll,
            2 => RuleProperty::Exposure,
            3 => RuleProperty::Aperture,
            4 => RuleProperty::Iso,
            5 => RuleProperty::FocalLength,
            _ => RuleProperty::FileName,
        }
    }
}

/// How a rule combines with the **previous kept** rule. The first rule in the
/// stack ignores its combinator. `AndNot` is darktable's third state — "and not
/// this" — which is why a plain two-state toggle never fits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Combinator {
    And,
    Or,
    AndNot,
}

impl Combinator {
    pub const ALL: [Combinator; 3] = [Combinator::And, Combinator::Or, Combinator::AndNot];

    pub fn label(self) -> &'static str {
        match self {
            Combinator::And => "AND",
            Combinator::Or => "OR",
            Combinator::AndNot => "AND NOT",
        }
    }

    pub fn to_index(self) -> u32 {
        match self {
            Combinator::And => 0,
            Combinator::Or => 1,
            Combinator::AndNot => 2,
        }
    }

    pub fn from_index(i: u32) -> Combinator {
        match i {
            1 => Combinator::Or,
            2 => Combinator::AndNot,
            _ => Combinator::And,
        }
    }
}

/// One rule: match `property` against `value` via comparator `cmp`; `comb`
/// says how it attaches to the previous *kept* rule (ignored for the first).
/// A rule whose value can't produce a predicate is kept in the stack — what
/// the controls show IS the state — but [`rule_stack_sql`] skips it when
/// composing (blank text values, unparseable numeric ones), and the next
/// producible rule still combines through its own `comb`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Rule {
    pub property: RuleProperty,
    pub cmp: RuleCmp,
    pub comb: Combinator,
    pub value: String,
}

impl Rule {
    pub fn new(comb: Combinator, property: RuleProperty, cmp: RuleCmp, value: impl Into<String>) -> Self {
        Self { property, cmp, comb, value: value.into() }
    }
}

/// Parse a numeric rule value: plain decimal, or a fraction `a/b` (how shutter
/// speeds are written — `1/60`). Whitespace-tolerant; denominator 0 and empty
/// parts fail. Returns the SQL-ready decimal literal.
///
/// Non-finite results (`inf`/`NaN` — Rust's parser accepts those spellings)
/// are rejected: the predicate splices the value straight into the SQL text as
/// a decimal literal, and there is no finite spelling of either that SQLite
/// would accept, so such a rule must be inert rather than break the query.
fn parse_numeric_value(raw: &str) -> Option<f64> {
    let v = raw.trim();
    let parsed = if let Some((num, den)) = v.split_once('/') {
        let n: f64 = num.trim().parse().ok()?;
        let d: f64 = den.trim().parse().ok()?;
        if d == 0.0 {
            return None;
        }
        n / d
    } else {
        v.parse().ok()?
    };
    parsed.is_finite().then_some(parsed)
}

/// Escape `value` for inlining into a SQL `LIKE` pattern — **the** injection
/// boundary of this module, because the loaders build SQL by string splicing
/// (values can't be bound without touching every loader).
///
/// Two independent hazards, both handled:
/// - SQL string-literal quoting: a `'` would let a value terminate the literal
///   and inject arbitrary SQL; doubled per SQLite convention.
/// - LIKE metacharacters: `%` and `_` are wildcards and `\` is our escape char;
///   each is backslash-escaped and the pattern carries `ESCAPE '\'` so a user
///   searching for `50%` matches `50%`, not `50anything`. Backslash itself is
///   escaped first so we don't double-escape our own insertions.
fn like_literal(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '%' | '_' => {
                out.push('\\');
                out.push(c);
            }
            '\'' => out.push_str("''"),
            _ => out.push(c),
        }
    }
    out
}

/// One rule as a parenthesized boolean over its property's column, or `None`
/// when the rule is inert (text property with a blank value; numeric property
/// with an unparseable value or a text-only comparator). `None` rules are
/// skipped by the composer exactly like blank-valued ones.
fn rule_predicate(rule: &Rule) -> Option<String> {
    match rule.property.kind() {
        PropertyKind::Text => {
            if rule.value.trim().is_empty() {
                return None;
            }
            // `%…%` only for Contains; Excludes wraps NOT around the same
            // pattern so both comparators agree on what a non-match means.
            let lit = format!("%{}%", like_literal(&rule.value));
            let col = rule.property.column();
            let pred = match rule.cmp {
                RuleCmp::Contains => format!("({col} LIKE '{lit}' ESCAPE '\\')"),
                RuleCmp::Excludes => format!("({col} NOT LIKE '{lit}' ESCAPE '\\')"),
                // Text property carrying a numeric comparator: no meaningful
                // predicate. Inert rather than wrong (defensive — the UI never
                // produces this combination).
                _ => return None,
            };
            Some(pred)
        }
        PropertyKind::Numeric => {
            // A numeric property with a text comparator is likewise inert.
            let op = match rule.cmp {
                RuleCmp::Lt => "<",
                RuleCmp::Lte => "<=",
                RuleCmp::Eq => "=",
                RuleCmp::Gte => ">=",
                RuleCmp::Gt => ">",
                _ => return None,
            };
            // Unparseable ⇒ inert, not broken SQL. NULL columns (no EXIF)
            // never satisfy any comparison, so unknowns stay out by design.
            let v = parse_numeric_value(&rule.value)?;
            Some(format!("({}{op}{v})", rule.property.column()))
        }
    }
}

/// Whole stack as ONE leading-`AND` fragment — the exact contract every loader
/// splices (see `current_filters_sql`): empty when nothing filters, ` AND …`
/// otherwise, with the stack's internal combinators parenthesized at every step
/// so they can't leak precedence into the surrounding query.
///
/// - Rules with no producible predicate are skipped: blank text values (a
///   half-typed entry must not silently filter everything out while the user
///   types) and unparseable numeric ones.
/// - Composition is strictly left-associative over the kept rules —
///   `r1 OR r2 AND NOT r3` builds `((r1 OR r2) AND NOT r3)` — because each
///   subsequent predicate wraps the accumulated expression; a flat join would
///   let OR swallow every later AND.
/// - An all-blank/empty stack yields `""`, never a tautology.
pub fn rule_stack_sql(stack: &[Rule]) -> String {
    let mut expr: Option<String> = None;
    for rule in stack {
        let pred = match rule_predicate(rule) {
            Some(p) => p,
            None => continue,
        };
        expr = Some(match expr {
            None => pred,
            Some(prev) => match rule.comb {
                Combinator::And => format!("({prev} AND {pred})"),
                Combinator::Or => format!("({prev} OR {pred})"),
                Combinator::AndNot => format!("({prev} AND NOT {pred})"),
            },
        });
    }
    expr.map(|e| format!(" AND {e}")).unwrap_or_default()
}

/// Upper bound on persisted rules — a corrupt/hostile token can't balloon the
/// left panel or the WHERE clause. darktable's own presets stay well under this.
pub const MAX_RULES: usize = 16;

const RULE_SEP: char = '/';
const FIELD_SEP: char = ':';

/// Percent-encode everything outside RFC-3986 unreserved, so neither separator
/// (`/`, `:`) nor anything else structural can survive inside a value. UTF-8 is
/// encoded byte-wise; decoding rebuilds the exact string or fails cleanly.
fn pct_encode(s: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char)
            }
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

/// Inverse of [`pct_encode`]; `None` on malformed hex or a non-UTF-8 result.
fn pct_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                let hex = bytes.get(i + 1..i + 3)?;
                let hi = (hex[0] as char).to_digit(16)?;
                let lo = (hex[1] as char).to_digit(16)?;
                out.push((hi * 16 + lo) as u8);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// Clamp a comparator into its property's family: a text property can't carry
/// `<`, a numeric one can't carry `contains`. Tokens are the only source of
/// such combinations (the UI produces valid ones by construction); without
/// this normalization a hand-edited token would leave canonical state the
/// controls can never show, and the divergence observer would rebuild rows on
/// every filter change forever.
fn normalize_cmp(property: RuleProperty, cmp: RuleCmp) -> RuleCmp {
    let set: &[RuleCmp] = match property.kind() {
        PropertyKind::Text => &RuleCmp::TEXT_SET,
        PropertyKind::Numeric => &RuleCmp::NUMERIC_SET,
    };
    match cmp.position_in(set) {
        Some(_) => cmp,
        None => set[0],
    }
}

/// Encode the stack for the `darkroom_ui_prefs` table: one rule per
/// `property:cmp:value` field triple (indices, value percent-encoded), rules
/// joined by `/`. No serde dependency exists in this crate; a hand-rolled
/// format is round-trip-safe here precisely because [`pct_encode`] guarantees
/// the separators can't occur raw inside a value.
///
/// The leading rule's `comb` is deliberately not encoded — it is meaningless
/// (nothing precedes it), and dropping it keeps tokens stable when the user
/// deletes the first row.
pub fn rule_stack_token_for(stack: &[Rule]) -> String {
    stack
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let comb = if i == 0 {
                String::new()
            } else {
                format!("{}", r.comb.to_index())
            };
            format!(
                "{}{}{}{}{}{}{}",
                r.property.to_index(),
                FIELD_SEP,
                r.cmp.to_index(),
                FIELD_SEP,
                comb,
                FIELD_SEP,
                pct_encode(&r.value)
            )
        })
        .collect::<Vec<_>>()
        .join(&RULE_SEP.to_string())
}

/// Decode a token back into the stack. Lenient the way the other tokens are:
/// malformed pieces are dropped rather than failing the whole token, and the
/// count is capped at [`MAX_RULES`]. An absent/malformed token decodes empty —
/// never a panic, never a partial filter the user didn't set.
pub fn parse_rule_stack_token(tok: &str) -> Vec<Rule> {
    tok.split(RULE_SEP)
        .filter_map(|piece| {
            let mut parts = piece.splitn(4, FIELD_SEP);
            let p = parts.next()?.parse::<u32>().ok()?;
            let c = parts.next()?.parse::<u32>().ok()?;
            // Empty combinator field = leading rule; anything else must parse.
            let comb_s = parts.next()?;
            let comb = if comb_s.is_empty() {
                Combinator::And
            } else {
                Combinator::from_index(comb_s.parse::<u32>().ok()?)
            };
            let v = parts.next()?;
            let value = pct_decode(v)?;
            let property = RuleProperty::from_index(p);
            Some(Rule {
                property,
                cmp: normalize_cmp(property, RuleCmp::from_index(c)),
                comb,
                value,
            })
        })
        .take(MAX_RULES)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn like_literal_neutralises_injection_and_wildcards() {
        // The classic injection attempt becomes an inert literal: quote doubled.
        assert_eq!(like_literal("x' OR 1=1 --"), "x'' OR 1=1 --");
        // LIKE wildcards are escaped, so "50%" finds the literal text…
        assert_eq!(like_literal("50%"), "50\\%");
        assert_eq!(like_literal("my_file"), "my\\_file");
        assert_eq!(like_literal("a\\b"), "a\\\\b");
        // …and all three escapes compose without double-escaping our own
        // insertions (backslash first).
        assert_eq!(like_literal("100%\\_done'"), "100\\%\\\\\\_done''");
        assert_eq!(like_literal(""), "");
    }

    #[test]
    fn rule_predicates_match_their_comparator_shape() {
        let contains = Rule::new(Combinator::And, RuleProperty::FileName, RuleCmp::Contains, "img");
        assert_eq!(
            rule_predicate(&contains),
            Some("(i.filename LIKE '%img%' ESCAPE '\\')".to_string())
        );
        let excludes =
            Rule::new(Combinator::Or, RuleProperty::FilmRoll, RuleCmp::Excludes, "raw");
        assert_eq!(
            rule_predicate(&excludes),
            Some("(f.folder NOT LIKE '%raw%' ESCAPE '\\')".to_string())
        );
    }

    #[test]
    fn empty_and_all_blank_stacks_splice_nothing() {
        assert_eq!(rule_stack_sql(&[]), "");
        let blank = vec![Rule::new(
            Combinator::And,
            RuleProperty::FileName,
            RuleCmp::Contains,
            "   ",
        )];
        assert_eq!(rule_stack_sql(&blank), "");
    }

    #[test]
    fn rule_stack_composes_left_associatively() {
        let stack = vec![
            Rule::new(Combinator::And, RuleProperty::FileName, RuleCmp::Contains, "img"),
            Rule::new(Combinator::AndNot, RuleProperty::FilmRoll, RuleCmp::Contains, "raw"),
            Rule::new(Combinator::Or, RuleProperty::FileName, RuleCmp::Contains, "2024"),
        ];
        // ((name-has-img AND NOT folder-has-raw) OR name-has-2024): left-assoc
        // nesting means the trailing OR binds to the GROUP — a file named 2024
        // inside the raw folder matches. A flat join would have let AND NOT
        // swallow it.
        assert_eq!(
            rule_stack_sql(&stack),
            " AND (((i.filename LIKE '%img%' ESCAPE '\\') \
             AND NOT (f.folder LIKE '%raw%' ESCAPE '\\')) \
             OR (i.filename LIKE '%2024%' ESCAPE '\\'))"
        );
    }

    #[test]
    fn blank_rules_are_skipped_but_combinators_survive_them() {
        let stack = vec![
            Rule::new(Combinator::And, RuleProperty::FileName, RuleCmp::Contains, "img"),
            // Blank middle rule: kept in state, skipped in SQL…
            Rule::new(Combinator::Or, RuleProperty::FilmRoll, RuleCmp::Contains, ""),
            // …and this one combines through its OWN Or to the last kept rule.
            Rule::new(Combinator::Or, RuleProperty::FileName, RuleCmp::Excludes, "2024"),
        ];
        assert_eq!(
            rule_stack_sql(&stack),
            " AND ((i.filename LIKE '%img%' ESCAPE '\\') \
             OR (i.filename NOT LIKE '%2024%' ESCAPE '\\'))"
        );
    }

    #[test]
    fn token_round_trips_arbitrary_values() {
        let stack = vec![
            Rule::new(Combinator::And, RuleProperty::FileName, RuleCmp::Contains, "50%/off"),
            Rule::new(
                Combinator::AndNot,
                RuleProperty::FilmRoll,
                RuleCmp::Excludes,
                "back\\slash'quote",
            ),
            Rule::new(
                Combinator::Or,
                RuleProperty::FileName,
                RuleCmp::Contains,
                "héllo wörld ✓:colon",
            ),
        ];
        let tok = rule_stack_token_for(&stack);
        // The structural separators can't appear raw — that's the guarantee the
        // hand-rolled format rests on.
        assert!(!tok.contains("%/"), "encoded values must hide '/'");
        assert_eq!(parse_rule_stack_token(&tok), stack);
    }

    #[test]
    fn leading_rule_encodes_no_combinator() {
        // The leading slot's combinator is meaningless (nothing precedes it), so
        // it is not encoded — an AndNot there decodes back as And, and deleting
        // the first row later keeps tokens stable.
        let and_first =
            vec![Rule::new(Combinator::And, RuleProperty::FileName, RuleCmp::Contains, "x")];
        let notfirst =
            vec![Rule::new(Combinator::AndNot, RuleProperty::FileName, RuleCmp::Contains, "x")];
        let tok = rule_stack_token_for(&notfirst);
        assert_eq!(rule_stack_token_for(&and_first), tok);
        assert_eq!(tok, format!("0:0::{}", pct_encode("x")));
        assert_eq!(parse_rule_stack_token(&tok), and_first);
    }

    #[test]
    fn malformed_tokens_decode_leniently_never_panic() {
        assert_eq!(parse_rule_stack_token(""), Vec::<Rule>::new());
        assert_eq!(parse_rule_stack_token("garbage"), Vec::<Rule>::new());
        // Bad hex and a missing field drop their pieces; the good piece stays.
        assert_eq!(
            parse_rule_stack_token(&format!(
                "0:0:0:%ZZ{RULE_SEP}0:1:1:ok{RULE_SEP}0:%"
            )),
            vec![Rule::new(
                Combinator::Or,
                RuleProperty::FileName,
                RuleCmp::Excludes,
                "ok"
            )]
        );
        // Count capped at MAX_RULES.
        let many = (0..40)
            .map(|i| {
                rule_stack_token_for(&[Rule::new(
                    Combinator::And,
                    RuleProperty::FileName,
                    RuleCmp::Contains,
                    format!("v{i}"),
                )])
            })
            .collect::<Vec<_>>()
            .join("/");
        assert_eq!(parse_rule_stack_token(&many).len(), MAX_RULES);
    }

    #[test]
    fn numeric_predicates_use_their_operator_shapes() {
        // Fractions parse how shutter speeds are written; the literal comes out
        // as the parsed decimal (expected built through the same parser so the
        // assertion pins operator placement, not float formatting).
        let shutter = Rule::new(
            Combinator::And,
            RuleProperty::Exposure,
            RuleCmp::Lt,
            "1/60",
        );
        assert_eq!(
            rule_predicate(&shutter),
            Some(format!("(i.exposure<{})", 1.0 / 60.0))
        );
        let iso = Rule::new(Combinator::Or, RuleProperty::Iso, RuleCmp::Gte, "1600");
        assert_eq!(rule_predicate(&iso), Some("(i.iso>=1600)".to_string()));
    }

    #[test]
    fn inert_numeric_rules_are_skipped_not_broken() {
        // Unparseable value…
        assert_eq!(
            rule_stack_sql(&[Rule::new(
                Combinator::And,
                RuleProperty::Aperture,
                RuleCmp::Lte,
                "f/2.8",
            )]),
            ""
        );
        // …zero denominator…
        assert_eq!(
            rule_stack_sql(&[Rule::new(
                Combinator::And,
                RuleProperty::Exposure,
                RuleCmp::Lt,
                "1/0",
            )]),
            ""
        );
        // …non-finite spellings Rust's parser accepts (they'd splice `inf`/
        // `NaN` into the SQL literal and break the query)…
        for raw in ["inf", "-inf", "NaN", "1e400"] {
            assert_eq!(
                rule_stack_sql(&[Rule::new(
                    Combinator::And,
                    RuleProperty::Exposure,
                    RuleCmp::Lt,
                    raw,
                )]),
                "",
                "value {raw} must be inert"
            );
        }
        // …and a text-only comparator on a numeric property are all inert.
        assert_eq!(
            rule_stack_sql(&[Rule::new(
                Combinator::And,
                RuleProperty::Iso,
                RuleCmp::Contains,
                "100",
            )]),
            ""
        );
        // Inert rules still don't break the chain: the next producible rule
        // combines straight to the last kept one.
        assert_eq!(
            rule_stack_sql(&[
                Rule::new(Combinator::And, RuleProperty::FileName, RuleCmp::Contains, "img"),
                Rule::new(Combinator::AndNot, RuleProperty::FocalLength, RuleCmp::Gt, "junk"),
                Rule::new(Combinator::Or, RuleProperty::FileName, RuleCmp::Contains, "raw"),
            ]),
            " AND ((i.filename LIKE '%img%' ESCAPE '\\') \
             OR (i.filename LIKE '%raw%' ESCAPE '\\'))"
        );
    }

    #[test]
    fn comparator_sets_partition_by_kind() {
        // No comparator belongs to both sets, and every member resolves within
        // its own set at its listed position.
        for c in RuleCmp::TEXT_SET {
            assert!(c.position_in(&RuleCmp::NUMERIC_SET).is_none());
            assert_eq!(
                RuleCmp::TEXT_SET[c.position_in(&RuleCmp::TEXT_SET).unwrap() as usize],
                c
            );
        }
        for c in RuleCmp::NUMERIC_SET {
            assert!(c.position_in(&RuleCmp::TEXT_SET).is_none());
            assert_eq!(
                RuleCmp::NUMERIC_SET[c.position_in(&RuleCmp::NUMERIC_SET).unwrap() as usize],
                c
            );
        }
        // Concretely: Contains is position 0 of TEXT_SET, Gt position 4 of
        // NUMERIC_SET — these positions are what dropdowns select.
        assert_eq!(RuleCmp::Contains.position_in(&RuleCmp::TEXT_SET), Some(0));
        assert_eq!(RuleCmp::Gt.position_in(&RuleCmp::NUMERIC_SET), Some(4));
    }

    #[test]
    fn numeric_rules_round_trip_through_tokens() {
        // The token format never special-cased kinds: comparator indices ride
        // the same global space, values stay percent-encoded strings.
        let stack = vec![
            Rule::new(Combinator::And, RuleProperty::Exposure, RuleCmp::Lt, "1/60"),
            Rule::new(Combinator::AndNot, RuleProperty::Iso, RuleCmp::Gte, "1600"),
            Rule::new(Combinator::Or, RuleProperty::Aperture, RuleCmp::Eq, "2.8"),
        ];
        let tok = rule_stack_token_for(&stack);
        assert_eq!(parse_rule_stack_token(&tok), stack);
    }

    #[test]
    fn tokens_normalize_kind_mismatched_comparators() {
        // FileName (text) carrying Gt clamps to TEXT_SET[0]…
        let parsed = parse_rule_stack_token(&format!("0:6:0:{}", pct_encode("x")));
        assert_eq!(parsed[0].property, RuleProperty::FileName);
        assert_eq!(parsed[0].cmp, RuleCmp::Contains);
        // …and ISO (numeric) carrying excludes clamps to NUMERIC_SET[0] — the
        // invariant that keeps canonical state representable by the controls.
        let parsed = parse_rule_stack_token(&format!("4:1:0:{}", pct_encode("1600")));
        assert_eq!(parsed[0].property, RuleProperty::Iso);
        assert_eq!(parsed[0].cmp, RuleCmp::Lt);
    }
}
