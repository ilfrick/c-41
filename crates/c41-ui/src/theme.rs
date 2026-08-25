//! darktable-matched GTK4 theme (parity audit 3.3).
//!
//! The app ships libadwaita's stock dark theme, which is recognisably GNOME:
//! blue accents, rounded corners, generous padding. darktable is flat, square
//! and entirely grey — the deliberate choice of an image editor, where any
//! saturated chrome biases your perception of the photo you are grading.
//!
//! This module installs a [`gtk4::CssProvider`] carrying darktable's own palette
//! and flattening its chrome. The values are taken verbatim from
//! `data/themes/darktable.css` in this repo (the upstream theme), so they are
//! the real thing rather than an approximation:
//!
//! | darktable token      | value     | used for                         |
//! |----------------------|-----------|----------------------------------|
//! | `grey_10`            | `#1b1b1b` | borders, scrollbar trough        |
//! | `grey_15`            | `#262626` | general background               |
//! | `grey_20`            | `#303030` | module (plugin) background       |
//! | `grey_25`            | `#3b3b3b` | collapsible header background    |
//! | `grey_35`            | `#525252` | selection                        |
//! | `grey_40`            | `#5e5e5e` | lighttable canvas, inactive bar  |
//! | `grey_50`            | `#777777` | darkroom canvas (mid grey)       |
//! | `grey_60`            | `#919191` | module labels, active scrollbar  |
//! | `grey_75`            | `#b9b9b9` | general text                     |
//!
//! **The canvas greys are load-bearing, not taste.** darktable puts the darkroom
//! canvas at a true mid grey (`grey_50`) precisely because the surround affects
//! how you judge tone and colour in the image on top of it; the upstream CSS
//! calls this out with "this need to be middle grey to correctly work on images.
//! And for all themes". Do not darken it for looks.
//!
//! Sliders are NOT styled here. darktable's "bauhaus" controls (flat fill-bar
//! sliders with the label and value drawn inline) are custom-drawn widgets, not
//! styled GTK ranges — so they were built as one: every slider in the UI is
//! [`crate::bauhaus`], a DrawingArea that paints itself. No `GtkScale` remains
//! in the crate, and the rules that used to style one were removed. What this
//! still does not attempt is the panel *layout*, which is parity 2.2-2.6.

/// darktable's grey ramp (`data/themes/darktable.css`, `@define-color grey_NN`).
/// Exposed so widget code can match the chrome without re-deriving hex values.
pub mod grey {
    pub const G05: &str = "#111111";
    pub const G10: &str = "#1b1b1b";
    pub const G15: &str = "#262626";
    pub const G20: &str = "#303030";
    pub const G25: &str = "#3b3b3b";
    pub const G30: &str = "#474747";
    pub const G35: &str = "#525252";
    pub const G40: &str = "#5e5e5e";
    pub const G45: &str = "#6a6a6a";
    pub const G50: &str = "#777777";
    pub const G60: &str = "#919191";
    pub const G75: &str = "#b9b9b9";
    pub const G80: &str = "#c6c6c6";
    pub const G90: &str = "#e2e2e2";
}

/// The stylesheet, built from [`grey`] so the palette has one definition.
///
/// Kept as a function rather than a `const` string so the colours interpolate
/// from the constants above — a mistyped hex in one rule would otherwise be
/// invisible.
fn css() -> String {
    format!(
        "
/* ── Base ────────────────────────────────────────────────────────────── */
/* darktable: bg_color = grey_15, fg_color = grey_75. Flat everywhere: the
   editor's chrome should never compete with the image. */
window, .background {{
  background-color: {g15};
  color: {g75};
}}

/* Kill libadwaita's blue. darktable has no accent colour — selection is a
   lighter grey (grey_35), which keeps the UI achromatic so it cannot bias
   colour judgement.

   These are @define-color, NOT CSS custom properties: GTK4 has no `--var`
   support and logs a no-such-property warning for each one, so a custom-property
   block silently does nothing. libadwaita reads these @define-color names. */
@define-color accent_bg_color {g35};
@define-color accent_fg_color {g90};
@define-color accent_color {g75};
@define-color theme_selected_bg_color {g35};
@define-color theme_selected_fg_color {g90};

/* ── Header / toolbars ───────────────────────────────────────────────── */
headerbar, .toolbar, actionbar > revealer > box {{
  background-color: {g15};
  background-image: none;
  border: none;
  box-shadow: none;
  min-height: 32px;
  color: {g75};
}}
headerbar:backdrop {{ background-color: {g15}; color: {g45}; }}

/* darktable's view switcher is plain text that brightens when active, not a
   pill. Square it off and drop the raised look. */
button, headerbar button {{
  background-image: none;
  background-color: {g20};
  border: 1px solid {g10};
  border-radius: 0;
  box-shadow: none;
  color: {g75};
  padding: 2px 8px;
  min-height: 24px;
}}
button:hover {{ background-color: {g30}; color: {g90}; }}
button:active, button:checked {{ background-color: {g35}; color: {g90}; }}
button:disabled {{ color: {g45}; background-color: {g20}; }}
button.flat {{ background-color: transparent; border-color: transparent; }}
button.flat:hover {{ background-color: {g30}; }}

/* ── Side panels and module rows ─────────────────────────────────────── */
/* plugin_bg_color = grey_20; collapsible header = grey_25. */
list, listview, .view {{
  background-color: {g15};
  color: {g75};
}}
row, .activatable {{
  background-color: {g20};
  border-radius: 0;
  min-height: 28px;
}}
row:hover {{ background-color: {g25}; }}
row:selected, row:checked {{ background-color: {g35}; color: {g90}; }}

/* libadwaita cards/boxed lists are rounded and inset; darktable's panels are
   flush rectangles butted against each other. */
.card, .boxed-list, preferencesgroup > box > box {{
  background-color: {g20};
  border-radius: 0;
  border: 1px solid {g10};
}}
expander-row, .expander-row {{ background-color: {g20}; border-radius: 0; }}

/* Section labels: plugin_label_color = grey_60, and darktable sets them in
   lowercase small caps. We keep the label text as authored but match weight. */
label.title-4, .heading {{
  color: {g60};
  font-weight: bold;
}}
label.dim-label {{ color: {g45}; }}

/* ── Sliders ─────────────────────────────────────────────────────────── */
/* Intentionally absent. Every slider in the UI is crate::bauhaus, a custom
   DrawingArea that paints itself from the style context's colour — there is no
   GtkScale anywhere in the crate, so the rules that used to live here styled
   nothing at all and were removed as dead weight.
   They were NOT the source of the negative-size gizmo warnings: those were
   measured again after this deletion and were unchanged. The `slider` in those
   warnings is the scrollbar's node, not a scale's — see the scrollbar block.
   If a stock GtkScale is ever reintroduced, style it here and check stderr. */

/* ── Entries, dropdowns, switches ────────────────────────────────────── */
entry, dropdown, combobox button, spinbutton {{
  background-color: {g20};
  color: {g75};
  border: 1px solid {g10};
  border-radius: 0;
  box-shadow: none;
}}
entry:focus, dropdown:focus {{ border-color: {g50}; box-shadow: none; }}
entry placeholder, entry text placeholder {{ color: {g45}; }}

switch {{
  background-color: {g25};
  border: 1px solid {g10};
  border-radius: 0;
  box-shadow: none;
}}
switch:checked {{ background-color: {g50}; }}
/* The knob: colour only. GTK names this node `slider` and derives its box from
   the switch's own allocation, so a size or border constraint here risks the
   same negative-gizmo trap as the scrollbar below. Dropping the constraints did
   not by itself silence the `reported min height -4` stream — the scrollbar was
   the actual source — but there is no reason to reintroduce the hazard.
   (Backticks, not quotes: this whole sheet is one format! string literal, and
   a double quote in here ends it — that is a compile error, not a CSS one.) */
switch > slider {{
  background-color: {g75};
}}

/* ── Scrollbars ──────────────────────────────────────────────────────── */
/* scroll_bar_bg = grey_10, inactive = grey_40, active = grey_60. */
scrollbar {{ background-color: {g10}; border: none; }}
/* Colour and corners only — no size or border constraint. Adwaita draws this
   node with a transparent border for its inset look, and GTK subtracts that
   border and padding from any min-* declared here: 8px minus 12px of inherited
   border+padding is the `reported min height -4` in the app's stderr, once per
   scrollbar per layout pass. Same trap as `switch > slider` above. */
scrollbar slider {{
  background-color: {g40};
  border-radius: 0;
}}
scrollbar slider:hover {{ background-color: {g60}; }}

/* ── Separators ──────────────────────────────────────────────────────── */
separator, .separator {{ background-color: {g10}; min-width: 1px; min-height: 1px; }}
paned > separator {{ background-color: {g10}; min-width: 2px; }}

/* ── Popovers / menus ────────────────────────────────────────────────── */
popover > contents, menu, .menu {{
  background-color: {g20};
  color: {g75};
  border: 1px solid {g10};
  border-radius: 0;
  box-shadow: none;
}}

/* ── Tooltips ────────────────────────────────────────────────────────── */
tooltip {{ background-color: {g10}; color: {g80}; border-radius: 0; }}

/* ── Multi-image selection frame (m4-144) ───────────────────────────── */
/* Inset outline on the selected cell's thumbnail — darktable's selection
   border, kept INSIDE the cell box so selecting never shifts the layout.
   g75 is the theme's accent (same family as the star/label highlights). */
.c41-cell-selected {{
  outline: 2px solid {g75};
  outline-offset: -2px;
}}

/* ── Canvas backgrounds ──────────────────────────────────────────────── */
/* THESE TWO ARE FUNCTIONAL, NOT DECORATIVE. darktable puts the darkroom canvas
   at a true middle grey so the surround does not skew your perception of the
   image's tone and colour; the lighttable sits slightly darker. Changing these
   for aesthetics changes how edits look.

   Declared last and with the child selectors GTK actually renders: a GridView
   paints through its own `child` nodes and sits inside a ScrolledWindow whose
   generic `.view` rule would otherwise win on specificity. */
.c41-darkroom-canvas {{ background-color: {g50}; }}
.c41-lighttable-canvas,
.c41-lighttable-canvas > child,
scrolledwindow > viewport > .c41-lighttable-canvas {{
  background-color: {g40};
}}
/* NOTE: this used to be a `scrolledwindow:has(.c41-lighttable-canvas)` pair.
   GTK4 does not implement :has() — and an unknown pseudo-class makes it discard
   the entire rule, silently — so those two rules never painted anything. The
   child selectors above are what actually paint the grid.
   (stylesheet_uses_no_selectors_gtk4_rejects enforces this.) */
",
        g10 = grey::G10,
        g15 = grey::G15,
        g20 = grey::G20,
        g25 = grey::G25,
        g30 = grey::G30,
        g35 = grey::G35,
        g40 = grey::G40,
        g45 = grey::G45,
        g50 = grey::G50,
        g60 = grey::G60,
        g75 = grey::G75,
        g80 = grey::G80,
        g90 = grey::G90,
    )
}

/// Install the darktable-matched theme on the default display.
///
/// Call once during startup, after the display is open. Uses
/// `PRIORITY_APPLICATION` so it overrides libadwaita's stock stylesheet while
/// still letting a user stylesheet win.
pub fn install() {
    let provider = gtk4::CssProvider::new();
    provider.load_from_string(&css());
    if let Some(display) = gtk4::gdk::Display::default() {
        gtk4::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_matches_darktable_css() {
        // Pinned against data/themes/darktable.css @define-color grey_NN. If
        // upstream ever re-tunes the ramp these are the values to re-check.
        assert_eq!(grey::G15, "#262626", "bg_color");
        assert_eq!(grey::G20, "#303030", "plugin_bg_color");
        assert_eq!(grey::G75, "#b9b9b9", "fg_color");
        assert_eq!(grey::G50, "#777777", "darkroom_bg_color — middle grey, functional");
        assert_eq!(grey::G40, "#5e5e5e", "lighttable_bg_color");
    }

    #[test]
    fn css_interpolates_every_placeholder() {
        // A missing format argument would leave a literal "{gNN}" in the sheet,
        // which GTK parses as garbage and silently drops the whole rule.
        let sheet = css();
        assert!(!sheet.contains("{g"), "un-interpolated placeholder left in the CSS");
        assert!(!sheet.contains("{{"), "escaped brace leaked into the output");
        // Spot-check that real values landed.
        assert!(sheet.contains("#262626"), "background grey missing");
        assert!(sheet.contains("#777777"), "darkroom canvas grey missing");
    }

    /// Strip `/* … */` so the selector ban applies to CSS, not to prose.
    ///
    /// The sheet is roughly 40% comments, and those comments need to *name* the
    /// constructs they warn about. Scanning them would mean a comment saying
    /// "don't use :has()" fails the build — the test dictating how the warning
    /// is phrased, which is backwards.
    fn selectors_only(sheet: &str) -> String {
        let mut out = String::new();
        let mut rest = sheet;
        while let Some(i) = rest.find("/*") {
            out.push_str(&rest[..i]);
            rest = match rest[i..].find("*/") {
                Some(j) => &rest[i + j + 2..],
                None => "",
            };
        }
        out.push_str(rest);
        out
    }

    #[test]
    fn stylesheet_uses_no_selectors_gtk4_rejects() {
        // GTK4's CSS engine is not a browser engine. An unknown pseudo-class
        // makes it discard the ENTIRE rule, silently — the theme looks mostly
        // right and one block is simply missing. Both of these shipped and were
        // only found by reading the app's stderr:
        //   `:root`   → "Unknown name of pseudo-class" (the accent override
        //               never applied, so libadwaita's blue survived)
        //   `:has()`  → "Unknown pseudoclass" (the grid background rule died)
        //
        // This is a DENYLIST, so it proves nothing in general — it pins the two
        // bugs already shipped plus the obvious neighbours, and cannot catch the
        // next novel one (GTK4 also rejects the `+` and `~` combinators, too
        // noisy to grep for textually). The only real check is
        // CssProvider::load_from_string with connect_parsing_error, which needs
        // GTK initialised and so cannot run in this display-free suite.
        //
        // EXPIRY: this ban is a property of the runtime, not of CSS. The app
        // ships on Ubuntu Noble → GTK 4.14 / libadwaita 1.5 (docker/Dockerfile),
        // and Cargo.toml gates to v4_12/v1_5. GTK gained `:root` and custom
        // properties in 4.16, and libadwaita ≥1.6 prefers them. When the runtime
        // moves past 4.16, revisit rather than mechanically keeping this green.
        let sheet = selectors_only(&css());
        assert!(!sheet.contains(":root"), ":root is not a GTK4 selector — use `window`");
        assert!(!sheet.contains(":has("), ":has() is unsupported; GTK4 drops the whole rule");
        // Other browser-isms that would fail the same way.
        for bad in [":is(", ":where(", "::before", "::after", "!important"] {
            assert!(!sheet.contains(bad), "unsupported CSS construct in the theme: {bad}");
        }
    }

    #[test]
    fn selectors_only_strips_comments_not_css() {
        let s = selectors_only("a {} /* :has( :root */ b {} /* unterminated");
        assert!(s.contains("a {}") && s.contains("b {}"), "CSS was dropped: {s:?}");
        assert!(!s.contains(":has("), "comment body survived: {s:?}");
    }

    #[test]
    fn no_accent_colour_survives() {
        // The point of the exercise: nothing saturated in the chrome, or it
        // biases colour judgement. Every colour literal must be an even grey.
        let sheet = css();
        for tok in sheet.split(|c: char| !(c.is_ascii_hexdigit() || c == '#')) {
            if let Some(hex) = tok.strip_prefix('#') {
                if hex.len() == 6 {
                    let (r, g, b) = (&hex[0..2], &hex[2..4], &hex[4..6]);
                    assert!(r == g && g == b, "non-grey colour {tok} in the theme");
                }
            }
        }
    }
}
