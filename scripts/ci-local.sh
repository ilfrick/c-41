#!/usr/bin/env bash
# Run the exact four steps CI runs (.github/workflows/rust.yml), in the same
# Ubuntu 24.04 image, against the working tree.
#
#   scripts/ci-local.sh
#
# Why this exists: m4-107 shipped a deny-by-default clippy error to CI because
# the local check grepped the `--> file:line` locator lines for "error".
# Clippy prints the diagnostic on the line BEFORE the locator, so that grep can
# never match — it reported success on a failing run. This script runs the real
# commands and trusts their exit codes, which is the only reliable signal.
#
# Steps, matching rust.yml exactly:
#   1. cargo check --workspace
#   2. cargo clippy --workspace          (default strictness, as CI)
#   3. cargo test --workspace --release  (CI uses --release; debug-only runs
#                                         have missed release-profile breakage)
#   4. cargo build --release -p darkroom --bin darkroom-rs   (the real link)
#
# Exit code is non-zero if any step fails, so it works as a pre-push gate.

set -uo pipefail

IMAGE=darkroom-rust-dev
DOCKERFILE=docker/Dockerfile.rust-dev
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RED=$'\033[31m'; GREEN=$'\033[32m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

if ! command -v docker >/dev/null 2>&1; then
  echo "${RED}docker not found${RESET} — this repo has no host toolchain (see CLAUDE.md)." >&2
  exit 127
fi

# Build the dev image if absent. Cheap when cached.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "${BOLD}Building $IMAGE …${RESET}"
  docker build -t "$IMAGE" -f "$DOCKERFILE" . || exit 1
fi

run_step() {
  local name="$1"; shift
  echo
  echo "${BOLD}── $name ──${RESET}"
  # Stream output so a failure is readable, but key the result off the exit
  # code, never off grepping the text.
  docker run --rm \
    -v "$REPO_ROOT":/src \
    -v "$REPO_ROOT/target":/cargo-target \
    -e CARGO_TARGET_DIR=/cargo-target \
    "$IMAGE" bash -c "$*"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "${RED}FAILED: $name (exit $rc)${RESET}" >&2
    return $rc
  fi
  echo "${GREEN}ok: $name${RESET}"
  return 0
}

failed=()
run_step "cargo check"   "cargo check --workspace"                             || failed+=("check")
run_step "cargo clippy"  "cargo clippy --workspace"                            || failed+=("clippy")
run_step "cargo test"    "cargo test --workspace --release"                    || failed+=("test")
run_step "release build" "cargo build --release -p darkroom --bin darkroom-rs" || failed+=("build")

echo
if [ ${#failed[@]} -ne 0 ]; then
  echo "${RED}${BOLD}CI-local FAILED:${RESET} ${failed[*]}" >&2
  exit 1
fi
echo "${GREEN}${BOLD}All four CI steps passed.${RESET}"
