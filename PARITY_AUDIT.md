# C-41 vs darktable — parity audit

**First written 2026-08-08. Last refreshed 2026-08-11** (severity 1 re-verified
against the code; resolved items struck through rather than deleted, so the
record of what was fixed survives).

> Keep this current. Between 08-08 and 08-11 it went stale enough to be
> actively misleading — it still listed three severity-1 items that had shipped
> on 08-08/08-09, which led to a proposal to rebuild finished work. If you fix
> something in here, mark it in the same commit.

Audit against darktable 5.6.0 (reference screenshot `Schermata_20260721_210438.png`)
and the running app in the KasmVNC container, plus a code sweep of
`crates/c41-ui`. Ordered by **severity**, not by roadmap position: severity 1
is "a user hits this and something is broken or blocked", severity 3 is "darktable
has it and we don't, but nothing misbehaves".

Deadline context: this is the list being worked top-down for **2026-08-20**.
Everything not reached by then stays here, written down, rather than being quietly
dropped. Full darktable parity is **not** reachable by that date and is not the
target — see the note at the bottom.

**Status 2026-08-11 (9 days out):** severity 1 is clear. The remaining work is
severity 2, and within it 2.1 (attaching UI to already-ported processing) is
both the largest gap and the cheapest per unit — each module is an increment.
Per-increment progress is logged in `PROGRESS.md`.

## Where the port actually stands

- **Pipeline (Phase 1): effectively done.** 4 files still contain OpenMP loops
  (9 total), and 2 of those aren't pipeline code (`tests/cache.c`,
  `sidecar_jobs.c`). **83 of 93** image-operation modules are ported to Rust.
- **UI (Phase 3): this is the whole remaining gap.** The processing for ~80
  modules exists in `c41-core` with **no controls attached to it**. The
  darkroom view exposes about 6 adjustments; darktable exposes 93 modules.

That asymmetry drives the ordering below: the expensive half is built, so most
severity-2 items are "attach a panel to code that already works", not new maths.

---

## Severity 1 — broken or blocking

**All clear as of 2026-08-11.** Every item below is resolved; kept for the record.

| # | Finding | Status |
|---|---------|--------|
| 1.1 | ~~**Side panels cannot be resized.**~~ | **Fixed** `45f0b427` (08-08). Nested `gtk4::Paned`; widths persist to the `darkroom_ui_prefs` table (debounced). Verified 08-11: `lib.rs` uses `Paned`, `LEFT/RIGHT_PANEL_*_PREF_KEY`. |
| 1.2 | ~~**Side panels cannot be collapsed.**~~ | **Fixed** `9fc1e344` (08-09). Header toggle per side + `L`/`R` keys; collapsed state persisted (`*_PANEL_COLLAPSED_PREF_KEY`). A header button rather than darktable's edge triangle — the GNOME idiom, and findable, which was half the complaint. |
| 1.3 | ~~**~915px minimum content width overflows a narrow window.**~~ | **Resolved by 1.1 + 1.2** — panels now shrink and collapse, so the fixed floor is gone. Re-check if a new fixed-width widget lands. |
| 1.4 | ~~**Metadata panel empty for the image selected at startup.**~~ | **Fixed** (08-08 session). `SingleSelection` auto-selects index 0 without firing `selection-changed`; the panel is now seeded explicitly and a `follow_selection` observer keeps preview + metadata in step (`lib.rs:663`). |

## Severity 2 — missing, high value, processing already ported

| # | Finding | Notes |
|---|---------|-------|
| 2.1 | **Most ported modules still have no UI — but the gap is closing.** As of 2026-08-11 **14 modules are live** in the darkroom panel (exposure, velvia, split-toning, monochrome, sigmoid, sharpen, vibrance, colorize, color correction, color contrast, color zones, levels, white balance, invert) plus crop/rotate/straighten and the demosaic selector. The m4-10x series adds roughly one per increment. Still missing with Rust code already present: **tone curve, RGB curve, denoise, highlight reconstruction, vignette, lens correction, colour balance RGB, filmic RGB, tone equalizer**. | still the highest value per day of work on this list. Note the curve-based modules (tone curve, RGB curve, base curve) need a **curve-editor widget** that does not exist yet — they are not one-increment slider jobs like the rest, and colorzones shipped with sliders only for the same reason |
| 2.2 | **No history stack in the lighttable.** The darkroom view has one; darktable exposes it in both. | `history.rs` exists — it is a darkroom-view panel |
| 2.3 | **Metadata is read-only.** darktable has a *metadata editor* (title, creator, rights…). Ours displays EXIF and nothing is writable. | m4-100 shipped the read side |
| 2.4 | **No styles.** darktable's "styles" panel (save a set of edits, apply to many images) has no equivalent. | |
| 2.5 | **No colour-label quick filter in either bar.** darktable has colour circles in the top bar *and* the bottom bar. Ours filters colours only from the left panel. Deliberately deferred (would duplicate the left-panel selector — reconcile first). | |
| 2.6 | **No collection-filters module** (darktable's "collection filters" expander), and no *import* module in the left panel — import is a header button only. | |
| 2.7 | **No geotagging, no neural-restore panel.** | low priority; listed for completeness |

## Severity 3 — parity polish

| # | Finding |
|---|---------|
| 3.1 | Zoomable view mode is still greyed out (`ViewMode::Zoomable::is_available() == false`). Deferred deliberately: `GridView` can't express an infinite zoom plane. |
| 3.2 | Panel sections are flat; darktable uses collapsible expanders throughout both panels. |
| 3.3 | ~~Theme is libadwaita dark, not darktable's exact greys.~~ **Fixed 2026-08-12** — `c41-ui/src/theme.rs` installs darktable's own palette (`grey_NN` from `data/themes/darktable.css`), flattens the chrome (square corners, no blue accent) and sets the functional canvas greys. Not attempted: bauhaus controls (custom-drawn upstream, not CSS) and panel *layout* (2.2-2.6). |
| 3.4 | Map / print / tethering views absent — the header's "Other" is a permanent placeholder. |
| 3.5 | Culling shows `THUMB_SIZE` cells rather than filling the viewport; full preview is gdk-pixbuf only, so raws it can't read (`.ORF`) show a message instead of an image. Both need the darkroom view's `BaseImage`/`render()` lifted into a shared module. |
| 3.6 | Full preview has no 100 % zoom/pan (darktable does). |

---

## Not on the table for 2026-08-20

Full darktable parity. darktable is ~93 processing modules, 5 views, mask
editing, tethering and a decade of interaction detail. The realistic target for
the deadline is **a photo editor that is genuinely usable end to end** — import,
cull, rate, edit with a real set of adjustments, export — with this document
covering what remains. Anything claiming more than that would be a claim, not a
plan.
