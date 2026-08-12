#!/bin/bash
# Launch Darkroom inside the KasmVNC desktop session.
# Loops so Darkroom restarts automatically if it exits or crashes.
#
# Signal forwarding: `docker stop` (and s6 service shutdown) delivers
# SIGTERM to the openbox session, not to this script's children, so by
# default darkroom is reparented to PID 1 and eventually SIGKILL'd —
# its SIGTERM handler never fires and any pending state is never flushed.
# We install a trap that forwards SIGTERM/SIGINT/SIGHUP to the running
# darkroom child and waits for it to exit cleanly (GTK teardown / any
# autosave) before the session goes down.

# Belt-and-suspenders: ensure config/cache dirs exist as the desktop user.
# The cont-init script (50-darkroom-dirs) runs as root earlier, but if the
# bind-mount timing or permissions caused it to fail, this catches it.
mkdir -p "${DARKROOM_CONFIGDIR:-/config/darkroom}" \
         "${DARKROOM_CACHEDIR:-/config/cache}" 2>/dev/null

# These are set by the loop and inspected by the trap.
child_pid=""
shutdown_requested=0

_forward_signal() {
  shutdown_requested=1
  if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
    echo "[autostart] forwarding $1 to darkroom (pid $child_pid)"
    kill -"$1" "$child_pid" 2>/dev/null
    # Wait up to ~15s for darkroom to shut down on its own. Teardown can
    # take a few seconds (pipeline teardown, cache write-back, db close),
    # so don't escalate too quickly.
    for _ in $(seq 1 30); do
      kill -0 "$child_pid" 2>/dev/null || break
      sleep 0.5
    done
  fi
  exit 0
}

trap '_forward_signal TERM' TERM
trap '_forward_signal INT'  INT
trap '_forward_signal HUP'  HUP

while true; do
  # c41-rs is the Rust/GTK4 front-end. It takes no CLI args — it reads the
  # catalog path from DARKROOM_LIBRARY_DB (set in the image, defaulted here as a
  # belt-and-suspenders fallback so the loop still works if the env is cleared).
  DARKROOM_LIBRARY_DB="${DARKROOM_LIBRARY_DB:-${DARKROOM_CONFIGDIR:-/config/darkroom}/library.db}" \
    /usr/local/bin/c41-rs &
  child_pid=$!
  # `wait` is interruptible by signals, so the trap above can run while we
  # block here. If the signal came in, the trap calls exit() so the loop
  # never iterates again.
  wait "$child_pid"
  rc=$?
  child_pid=""
  if [ "$shutdown_requested" = "1" ]; then
    exit 0
  fi
  echo "[autostart] Darkroom exited (code $rc), restarting in 3s..."
  sleep 3
done
