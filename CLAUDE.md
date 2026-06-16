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
4. **Test with Docker** — build the image, run the container, **read the logs**
   (real runtime validation, not just `--version`/unit tests):
   - Rust crates: `docker build -t darkroom-rust-dev -f docker/Dockerfile.rust-dev .`
     then in the container run `cargo check --workspace`, `cargo clippy`, and
     `cargo test` and inspect the output.
   - C changes: also compile under Release `-Werror`. For the full app image use
     `--build-arg CACHEBUST=<sha>` or `--no-cache` (the `git clone` layer caches).
5. **Fix** anything the build/logs surface.
6. **Commit + push to BOTH remotes**: `origin` (GitHub `ilfrick/darkroom`) has
   dual push URLs so a single `git push origin master` pushes to GitHub **and**
   Gitea (`housefz.com`). Verify both refs update.
7. (Non-blocking) **Confirm CI green** via
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
