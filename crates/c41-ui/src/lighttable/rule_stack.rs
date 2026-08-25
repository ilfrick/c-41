//! Collection **rule stack** (m4-134, parity 2.6 slice 2) — darktable's
//! filtering.c arbitrary rules: a list of rules (property, comparator, value)
//! joined by AND / OR / AND-NOT combinators, composing on top of whatever
//! collection is active exactly like the single-state quick filters do.
//!
//! Slice scope: text rules over properties the schema already carries — the
//! filename and the film-roll path. darktable's exposure/ISO/focal-length rules
//! stay out until those EXIF columns exist (`images` here is the narrow subset;
//! adding them is a schema migration + importer change, recorded in
//! PARITY_AUDIT as the follow-on slice). The *mechanism* — N rules, three
//! combinators, persistence, injection-safe composition — is the point, and it
//! is built so new properties are one enum arm each.
//!
//! Everything decision-bearing is pure and unit-tested display-free:
//! [`rule_stack_sql`] (composition), [`like_literal`] (the injection boundary),
//! and the token round-trip ([`rule_stack_token_for`] /
//! [`parse_rule_stack_token`]).

/// Comparator for a text rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextCmp {
    Contains,
    Excludes,
}

impl TextCmp {
    pub const ALL: [TextCmp; 2] = [TextCmp::Contains, TextCmp::Excludes];

    pub fn label(self) -> &'static str {
        match self {
            TextCmp::Contains => "contains",
            TextCmp::Excludes => "excludes",
        }
    }

    /// Stable index for `DropDown` wiring; unknown indices decode to
    /// [`TextCmp::Contains`], the same lenient default the other tokens use.
    pub fn to_index(self) -> u32 {
        match self {
            TextCmp::Contains => 0,
            TextCmp::Excludes => 1,
        }
    }

    pub fn from_index(i: u32) -> TextCmp {
        match i {
            1 => TextCmp::Excludes,
            _ => TextCmp::Contains,
        }
    }
}

/// The row's own columns a rule can match. `FileName` is `i.filename`,
/// `FilmRoll` the roll's folder path `f.folder` — the aliases every loader's
/// query already uses (see the splice sites in `super`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuleProperty {
    FileName,
    FilmRoll,
}

impl RuleProperty {
    pub const ALL: [RuleProperty; 2] = [RuleProperty::FileName, RuleProperty::FilmRoll];

    pub fn label(self) -> &'static str {
        match self {
            RuleProperty::FileName => "Filename",
            RuleProperty::FilmRoll => "Film roll",
        }
    }

    /// Column reference valid inside any loader query (they all alias
    /// `images i JOIN film_rolls f`). Kept beside the variant so a new property
    /// can't forget its column.
    pub fn column(self) -> &'static str {
        match self {
            RuleProperty::FileName => "i.filename",
            RuleProperty::FilmRoll => "f.folder",
        }
    }

    pub fn to_index(self) -> u32 {
        match self {
            RuleProperty::FileName => 0,
            RuleProperty::FilmRoll => 1,
        }
    }

    pub fn from_index(i: u32) -> RuleProperty {
        match i {
            1 => RuleProperty::FilmRoll,
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

/// One rule: match `property` against `value`, negated iff `cmp` excludes;
/// `comb` says how it attaches to the previous *kept* rule (ignored for the
/// first). A rule with a blank value is kept in the stack — what the controls
/// show IS the state — but [`rule_stack_sql`] skips it when composing, and the
/// next non-blank rule still combines through its own `comb`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Rule {
    pub property: RuleProperty,
    pub cmp: TextCmp,
    pub comb: Combinator,
    pub value: String,
}

impl Rule {
    pub fn new(comb: Combinator, property: RuleProperty, cmp: TextCmp, value: impl Into<String>) -> Self {
        Self { property, cmp, comb, value: value.into() }
    }
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

/// One rule as a parenthesized boolean over its property's column.
fn rule_predicate(rule: &Rule) -> String {
    // `%…%` only for Contains; Excludes wraps NOT around the same pattern so
    // both comparators agree on what a blank/absent match means.
    let lit = format!("%{}%", like_literal(&rule.value));
    let col = rule.property.column();
    match rule.cmp {
        TextCmp::Contains => format!("({col} LIKE '{lit}' ESCAPE '\\')"),
        TextCmp::Excludes => format!("({col} NOT LIKE '{lit}' ESCAPE '\\')"),
    }
}

/// Whole stack as ONE leading-`AND` fragment — the exact contract every loader
/// splices (see `current_filters_sql`): empty when nothing filters, ` AND …`
/// otherwise, with the stack's internal combinators parenthesized at every step
/// so they can't leak precedence into the surrounding query.
///
/// - Blank-valued rules are skipped: a half-typed entry must not silently filter
///   everything out while the user types.
/// - Composition is strictly left-associative over the kept rules —
///   `r1 OR r2 AND NOT r3` builds `((r1 OR r2) AND NOT r3)` — because each
///   subsequent predicate wraps the accumulated expression; a flat join would
///   let OR swallow every later AND.
/// - An all-blank/empty stack yields `""`, never a tautology.
pub fn rule_stack_sql(stack: &[Rule]) -> String {
    let mut expr: Option<String> = None;
    for rule in stack {
        if rule.value.trim().is_empty() {
            continue;
        }
        let pred = rule_predicate(rule);
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
            Some(Rule {
                property: RuleProperty::from_index(p),
                cmp: TextCmp::from_index(c),
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
        let contains = Rule::new(Combinator::And, RuleProperty::FileName, TextCmp::Contains, "img");
        assert_eq!(
            rule_predicate(&contains),
            "(i.filename LIKE '%img%' ESCAPE '\\')"
        );
        let excludes =
            Rule::new(Combinator::Or, RuleProperty::FilmRoll, TextCmp::Excludes, "raw");
        assert_eq!(
            rule_predicate(&excludes),
            "(f.folder NOT LIKE '%raw%' ESCAPE '\\')"
        );
    }

    #[test]
    fn empty_and_all_blank_stacks_splice_nothing() {
        assert_eq!(rule_stack_sql(&[]), "");
        let blank = vec![Rule::new(
            Combinator::And,
            RuleProperty::FileName,
            TextCmp::Contains,
            "   ",
        )];
        assert_eq!(rule_stack_sql(&blank), "");
    }

    #[test]
    fn rule_stack_composes_left_associatively() {
        let stack = vec![
            Rule::new(Combinator::And, RuleProperty::FileName, TextCmp::Contains, "img"),
            Rule::new(Combinator::AndNot, RuleProperty::FilmRoll, TextCmp::Contains, "raw"),
            Rule::new(Combinator::Or, RuleProperty::FileName, TextCmp::Contains, "2024"),
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
            Rule::new(Combinator::And, RuleProperty::FileName, TextCmp::Contains, "img"),
            // Blank middle rule: kept in state, skipped in SQL…
            Rule::new(Combinator::Or, RuleProperty::FilmRoll, TextCmp::Contains, ""),
            // …and this one combines through its OWN Or to the last kept rule.
            Rule::new(Combinator::Or, RuleProperty::FileName, TextCmp::Excludes, "2024"),
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
            Rule::new(Combinator::And, RuleProperty::FileName, TextCmp::Contains, "50%/off"),
            Rule::new(
                Combinator::AndNot,
                RuleProperty::FilmRoll,
                TextCmp::Excludes,
                "back\\slash'quote",
            ),
            Rule::new(
                Combinator::Or,
                RuleProperty::FileName,
                TextCmp::Contains,
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
            vec![Rule::new(Combinator::And, RuleProperty::FileName, TextCmp::Contains, "x")];
        let notfirst =
            vec![Rule::new(Combinator::AndNot, RuleProperty::FileName, TextCmp::Contains, "x")];
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
                TextCmp::Excludes,
                "ok"
            )]
        );
        // Count capped at MAX_RULES.
        let many = (0..40)
            .map(|i| {
                rule_stack_token_for(&[Rule::new(
                    Combinator::And,
                    RuleProperty::FileName,
                    TextCmp::Contains,
                    format!("v{i}"),
                )])
            })
            .collect::<Vec<_>>()
            .join("/");
        assert_eq!(parse_rule_stack_token(&many).len(), MAX_RULES);
    }
}
