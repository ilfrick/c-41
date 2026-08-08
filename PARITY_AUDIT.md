# Darkroom vs darktable — parity audit (2026-08-08)

Audit against darktable 5.6.0 (reference screenshot `Schermata_20260721_210438.png`)
and the running app in the KasmVNC container, plus a code sweep of
`crates/darkroom-ui`. Ordered by **severity**, not by roadmap position: severity 1
is "a user hits this and something is broken or blocked", severity 3 is "darktable
has it and we don't, but nothing misbehaves".

Deadline context: this is the list being worked top-down for **2026-08-20**.
Everything not reached by then stays here, written down, rather than being quietly
dropped. Full darktable parity is **not** reachable by that date and is not the
target — see the note at the bottom.

## Where the port actually stands

- **Pipeline (Phase 1): effectively done.** 4 files still contain OpenMP loops
  (9 total), and 2 of those aren't pipeline code (`tests/cache.c`,
  `sidecar_jobs.c`). **83 of 93** image-operation modules are ported to Rust.
- **UI (Phase 3): this is the whole remaining gap.** The processing for ~80
  modules exists in `darkroom-core` with **no controls attached to it**. The
  darkroom view exposes about 6 adjustments; darktable exposes 93 modules.

That asymmetry drives the ordering below: the expensive half is built, so most
severity-2 items are "attach a panel to code that already works", not new maths.

---

## Severity 1 — broken or blocking

| # | Finding | Evidence |
|---|---------|----------|
| 1.1 | **Side panels cannot be resized.** No `GtkPaned` anywhere in the UI crate; panel widths are hard-coded (`panels/mod.rs:85`, `:1026` → 210px; `lib.rs:305` → 200px). darktable lets you drag both panel edges. | user-reported; `grep -c Paned` = 0 |
| 1.2 | **Side panels cannot be collapsed.** darktable collapses each panel with the triangles at the screen edges. Ours are permanently on screen, which is also what makes 1.3 unavoidable. | reference screenshot, left/right edge triangles |
| 1.3 | **~915px minimum content width overflows a narrow window** — fixed panels + a 2-column grid + fixed metadata panel. Below that the header controls and the bottom bar's right end clip off-screen. Open since m4-98a; 1.1/1.2 are the real fix. | `RUST_MIGRATION_PLAN.md` m4-98a note |
| 1.4 | **Metadata panel is empty for the image selected at startup.** `SingleSelection` auto-selects index 0 on load, which fires no `selection-changed`, so the panel sits on "Select an image to view metadata" until the user clicks something else. | observed live 2026-08-07 |

## Severity 2 — missing, high value, processing already ported

| # | Finding | Notes |
|---|---------|-------|
| 2.1 | **~80 ported modules have no UI.** The darkroom view has exposure/black, contrast, velvia, split-toning, monochrome, crop/rotate/straighten, demosaic choice. Missing with the Rust code already present: **white balance, tone curve, sharpen, denoise, highlight reconstruction, vignette, lens correction, colour balance**, and more. | highest value per day of work of anything on this list |
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
| 3.3 | Theme is libadwaita dark, not darktable's exact greys. |
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
