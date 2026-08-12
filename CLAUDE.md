# Darkroom — working agreement for Claude Code

This repo is an **incremental rewrite of darktable (C/GTK3) into Rust + GTK4**,
behind a stable C FFI boundary. Status and roadmap live in
`RUST_MIGRATION_PLAN.md` (Phase 0 infra ✅, Phase 1 pipeline at boundary,
Phase 2 db ✅, Phase 3 GTK4 UI in progress). Rust crates are under `crates/`
(`darkroom-sys`, `darkroom-core`, `darkroom-db`, `darkroom-ui`, `darkroom`).

## Required per-change workflow (do every time — no shortcuts)

For **every** code change, run this full loop in order:

1. **Develop** the change.
2. **Senior review** — spawn the `fricktrade-architect` agent with `model: "opus"`
   (Opus 4.8). It is nominally scoped to "Fricktrade" but is intentionally used
   for darkroom reviews too. Give it the full diff and context.
3. **Fix** the review findings.
4. **Test with Docker** — run **`scripts/ci-local.sh`**. It executes the same
   four steps CI runs, in the CI image, and keys off **exit codes**:
   `cargo check --workspace`, `cargo clippy --workspace`,
   `cargo test --workspace --release`, and the
   `cargo build --release -p darkroom --bin darkroom-rs` link.
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
7. **Commit + push to BOTH remotes**: `origin` (GitHub `ilfrick/darkroom`) has
   dual push URLs so a single `git push origin master` pushes to GitHub **and**
   Gitea (`housefz.com`). Verify both refs update.
8. (Non-blocking) **Confirm CI green** via
   `gh api repos/ilfrick/darkroom/commits/<sha>/check-runs` — the
   `check + test + clippy` and `Build & push Docker image` runs. CI logs aren't
   directly readable (403); the local Docker build is the authoritative check.

End commit messages with:
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

## Autonomy

Work **continuously toward the current goal** — do not stop between increments.
Only stop when: (a) the goal is fully reached, (b) you genuinely need the user's
input/decision, or (c) you run out of context/tokens. Chain Phase increments
back-to-back, each through the full workflow above. After finishing one
increment, pick the next step from `RUST_MIGRATION_PLAN.md` and continue.

## Notes

- No local C build deps — build via Docker, never assume a host toolchain.
- `clippy` runs at default strictness in CI; the pre-existing `clone!`
  deprecation warnings in `darkroom-ui` are a known style item, out of scope.
