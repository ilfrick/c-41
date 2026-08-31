# C-41 — working agreement for Claude Code

This repo is **C-41**, an incremental rewrite of darktable (C/GTK3) into Rust + GTK4,
behind a stable C FFI boundary. Status and roadmap live in
`RUST_MIGRATION_PLAN.md` (Phase 0 infra ✅, Phase 1 pipeline at boundary,
Phase 2 db ✅, Phase 3 GTK4 UI in progress). Rust crates are under `crates/`
(`c41-sys`, `c41-core`, `c41-db`, `c41-ui`, `c41`).

## Required per-change workflow (do every time — no shortcuts)

For **every** code change, run this full loop in order:

1. **Develop** the change.
2. **Senior review** — spawn an **independent agent** (never review in the same
   context that wrote the code) with the profile of a **senior software
   developer with 20+ years of experience**, running the **same model in use
   for the development steps**. Give it the full diff and context.
3. **Fix** the review findings.
4. **Test with Docker** — run **`scripts/ci-local.sh`**. It executes the same
   four steps CI runs, in the CI image, and keys off **exit codes**:
   `cargo check --workspace`, `cargo clippy --workspace`,
   `cargo test --workspace --release`, and the
   `cargo build --release -p c41 --bin c41-rs` link.
   - **Never conclude "clean" by grepping command output.** Clippy prints its
     diagnostic on the line *before* the `--> file:line` locator, so grepping
     locator lines for "error" silently reports success on a failing run — that
     is exactly how m4-107 shipped a red build. Trust the exit code.
   - Note CI runs tests under `--release`; a debug-only run is not sufficient.
   - `scripts/pre-push` enforces this on every push
     (`git config core.hooksPath scripts`; bypass with `--no-verify`).
   - C changes: also compile under Release `-Werror`. For the full app image use
     `--build-arg CACHEBUST=<sha>` or `--no-cache` (the `git clone` layer caches).
5. **Fix** anything the build/logs surface.
6. **Log it in `PROGRESS.md`** — append a timestamped entry (UTC, newest last)
   recording what changed, how it was verified, and anything that went wrong.
   Keep failures and corrections in; they are the part worth rereading. Also
   update `PARITY_AUDIT.md` in the *same* commit if the change resolves an item
   there — that file went stale once and misled the next session's planning.
7. **Commit + push to BOTH remotes**: `origin` (GitHub `ilfrick/c-41`) has
   dual push URLs so a single `git push origin master` pushes to GitHub **and**
   Gitea (`housefz.com`). Verify both refs update.
8. (Non-blocking) **Confirm CI green** via
   `gh api repos/ilfrick/c-41/commits/<sha>/check-runs` — the
   `check + test + clippy` and `Build & push Docker image` runs. CI logs aren't
   directly readable (403); the local Docker build is the authoritative check.
9. **Fix CI failures and repeat until green** — if CI reports a failure, fix
   the root cause (re-run the loop from step 4, since a fix is a new change)
   and push again. Do not leave the branch red: a change is only done once CI
   is green on both the `check + test + clippy` and `Build & push Docker image`
   runs.
10. **Commit and push to both repos** (credentials in `~/.netrc`). After CI
    goes green, make sure the commit is on both `origin` URLs (GitHub
    `ilfrick/c-41` and Gitea `housefz.com`) — a single
    `git push origin master` covers both via the dual push URLs; verify both
    refs actually update.

## Naming (the C-41 rename, 2026-08-12)

The project, crates, docs and both repos are **C-41**. Three uses of the old
name are deliberately kept — renaming any of them would break something real:

- **`darkroom_*` FFI symbols** (311 `#[no_mangle]` exports called from 105 C
  files). Renaming needs both sides changed atomically and buys nothing.
- **`darkroom_ui_prefs`** — a live SQLite table (`c41-ui/src/persist.rs`).
  Renaming it orphans every user's saved panel widths, filters and view modes.
- **`/config/darkroom/`, `DARKROOM_*` env vars** — the running container's data
  directory, holding `library.db`. Renaming strands the user's catalogue.
- **"darkroom view"** — darktable's own vocabulary for the edit view (as opposed
  to the lighttable). It names a concept in the app being mirrored, not us.

A future increment could migrate the table and config path with a real
migration; until then, leave them.

## Autonomy

Work **continuously toward the current goal** — do not stop between increments.
Only stop when: (a) the goal is fully reached, (b) you genuinely need the user's
input/decision, or (c) you run out of context/tokens. Chain Phase increments
back-to-back, each through the full workflow above. After finishing one
increment, pick the next step from `RUST_MIGRATION_PLAN.md` and continue.

## Context efficiency (delegated development)

Main-session context is the binding constraint on chaining increments — a
session that hand-writes each increment exhausts itself after ~1–2. So:

- **Delegate the heavy development of each increment** — reading the C
  source, writing the Rust kernels/tests/references, FFI exports, and
  C call-site edits — to a **fresh general-purpose subagent** with a precise
  task spec: target loops, the established module conventions (guard
  pattern, reference-implementation style, test style), the files to
  create/edit, and a read-only-everything-else mandate. Give it the relevant
  repo paths and let it read the code itself.
- The main session keeps, itself: increment scoping, the task spec,
  verifying the delegated work (tests/gates), the **independent senior
  review** (step 2 — that agent is separate by design and must never be the
  one that wrote the code), applying review fixes (a small fix may be done
  in-session; a large one re-delegated), the PROGRESS.md entry,
  commit/push, and CI confirmation.
- If delegated development comes back broken, prefer **re-delegating with a
  corrected spec** over hand-fixing at length — the fix usually costs less
  context than the diagnosis.
- State lives in the repo, not the session: `PROGRESS.md` (newest last),
  `RUST_MIGRATION_PLAN.md`, `PARITY_AUDIT.md`. Any fresh session must be
  able to resume from those alone; keep them current for exactly that
  reason.

## Notes

- **Never commit, or otherwise share, API keys or other secrets** — no keys,
  tokens, passwords or credentials in commits, diffs, `PROGRESS.md` entries,
  logs, task specs or subagent prompts. Stage files deliberately (never a
  blind `git add -A` over unknown state) and treat anything secret-shaped in
  a diff as a stop-before-commit.
- No local C build deps — build via Docker, never assume a host toolchain.
- `clippy` runs at default strictness in CI; the pre-existing `clone!`
  deprecation warnings in `c41-ui` are a known style item, out of scope.
