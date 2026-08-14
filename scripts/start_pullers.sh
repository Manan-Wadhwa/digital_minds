#!/usr/bin/env bash
# Start (or restart) the sandbox mirror supervisors. Idempotent: a sandbox
# whose supervisor is already running is left alone. Installed as a cron
# @reboot entry on 2026-08-14 after a low-battery shutdown killed both
# pullers; also safe to run by hand any time.
#
# Tokens live OUTSIDE git in <repo>/../archive-logs/sandbox_tokens.env:
#   SB3_URL=... SB3_TOKEN=...   (map/32B box)
#   SB2_URL=... SB2_TOKEN=...   (ladder box)
# Each supervisor tags its command line with pull_supervisor_<name> so it
# can be found (pgrep -f pull_supervisor_sandbox3) or killed individually;
# `pkill -f pull_from_sandbox` still stops everything.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/../archive-logs"
ENVFILE="$LOGDIR/sandbox_tokens.env"
EXEC="$HOME/.claude/skills/marimo-pair/scripts/execute-code.sh"
[ -f "$ENVFILE" ] || { echo "missing $ENVFILE" >&2; exit 1; }
. "$ENVFILE"
mkdir -p "$LOGDIR"

start_one() {
  local name="$1" url="$2" token="$3"
  if pgrep -f "pull_supervisor_$name" > /dev/null 2>&1; then
    echo "$name: already running"
    return 0
  fi
  MARIMO_EXEC="$EXEC" URL="$url" TOKEN="$token" \
    setsid nohup bash -c "while :; do bash '$ROOT/scripts/pull_from_sandbox.sh' '$ROOT' --once; sleep 150; done # pull_supervisor_$name" \
    >> "$LOGDIR/$name.log" 2>&1 < /dev/null &
  echo "$name: started (log $LOGDIR/$name.log)"
}

start_one sandbox3 "${SB3_URL:?}" "${SB3_TOKEN:?}"
start_one sandbox2 "${SB2_URL:?}" "${SB2_TOKEN:?}"
