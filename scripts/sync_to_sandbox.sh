#!/usr/bin/env bash
# Push repo code into the live marimo sandbox.
#
# The sandbox holds the GPU; this repo holds the code and the git history. They
# share no filesystem, so the repo stays the single source of truth and this
# script ships a snapshot of it into the kernel before every run. Results come
# back the other way and are committed here.
#
# Two failures this script is built to avoid, both learned the hard way:
#
#   1. The payload used to be passed as an argv element. Once enough result JSON
#      accumulated, that hit "Argument list too long" -- and because the caller
#      had redirected stderr, the sync failed silently, `import run` fell through
#      to a stale experiment directory still on sys.path, and a previous
#      experiment re-ran under the new one's name. The results looked entirely
#      plausible. Payload now goes over stdin, and results/ is excluded because
#      it is output, not input.
#
#   2. A sync that fails must be loud. Every step is checked and the script
#      verifies what actually landed rather than assuming.
#
# Usage: scripts/sync_to_sandbox.sh [expected_file_relative_to_repo_root ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL="$REPO_ROOT/.agents/skills/marimo-pair/scripts/execute-code.sh"

: "${MARIMO_URL:?set MARIMO_URL}"
: "${MARIMO_TOKEN_FILE:?set MARIMO_TOKEN_FILE}"
[ -r "$MARIMO_TOKEN_FILE" ] || { echo "token file unreadable: $MARIMO_TOKEN_FILE" >&2; exit 1; }

SHA="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Mark the SHA when the SYNCED SOURCE differs from HEAD. This matters because
# `runner.git_sha()` falls back to CALIBRATION_GIT_SHA, which this script sets --
# so without the marker a run of uncommitted code records a clean SHA whose
# contents are not the code that ran. That is not hypothetical: E5's manifest
# claims 043bf8a while the code that produced its numbers landed one commit later
# as 3da25d8, and the discrepancy had to be caught and documented by hand.
#
# Scoped to what is actually shipped (src, experiments) and excluding results/,
# which is output and changes on every run.
DIRTY="$(cd "$REPO_ROOT" && git status --porcelain -- src experiments 2>/dev/null \
         | grep -v '/results/' || true)"
if [ -n "$DIRTY" ]; then
    SHA="${SHA}-dirty"
    echo "WARNING: synced source differs from HEAD; manifests will record ${SHA}" >&2
    echo "$DIRTY" | sed 's/^/    /' >&2
fi

TARBALL="$(mktemp)"
trap 'rm -f "$TARBALL"' EXIT

# Source only. results/ is output and __pycache__ is noise; shipping either just
# grows the payload that broke this script once already.
tar czf "$TARBALL" \
    --exclude='results' --exclude='__pycache__' --exclude='*.pyc' \
    -C "$REPO_ROOT" src experiments

SIZE=$(wc -c < "$TARBALL")
echo "payload: ${SIZE} bytes  git: ${SHA}"

python3 - "$SHA" "$SKILL" "$MARIMO_URL" "$MARIMO_TOKEN_FILE" "$TARBALL" "$@" <<'DRIVER'
import base64, pathlib, subprocess, sys

sha, skill, url, token_file, tarball = sys.argv[1:6]
expected = sys.argv[6:]

payload = base64.b64encode(pathlib.Path(tarball).read_bytes()).decode()

code = f'''
import base64, io, os, pathlib, sys, tarfile

DEST = pathlib.Path("/marimo/repo")
DEST.mkdir(parents=True, exist_ok=True)
blob = base64.b64decode(_PAYLOAD)
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
    tf.extractall(DEST)

os.environ["CALIBRATION_GIT_SHA"] = {sha!r}
src = str(DEST / "src")
if src not in sys.path:
    sys.path.insert(0, src)
for mod in [m for m in list(sys.modules) if m.startswith("calibration") or m == "run"]:
    del sys.modules[mod]

import calibration
print("SYNC_OK", calibration.__version__, {sha!r}, sum(1 for _ in DEST.rglob("*.py")), "py files")
for rel in {expected!r}:
    p = DEST / rel
    print("SYNC_CHECK", rel, "PRESENT" if p.exists() else "MISSING")
'''
# Bind the payload by name rather than interpolating it into the source, so the
# generated program stays small and readable regardless of repo size.
code = f"_PAYLOAD = {payload!r}\n" + code

with open(token_file) as fh:
    token = fh.read().strip()

proc = subprocess.run(
    ["bash", skill, "--url", url, "--token", token, "-"],
    input=code, text=True, capture_output=True,
)
out = proc.stdout + proc.stderr
for line in out.splitlines():
    if line.startswith(("SYNC_OK", "SYNC_CHECK")):
        print(line)

if proc.returncode != 0 or "SYNC_OK" not in out:
    print("SYNC FAILED", file=sys.stderr)
    print(out[-2000:], file=sys.stderr)
    sys.exit(1)
if "MISSING" in out:
    print("SYNC INCOMPLETE: an expected file did not land", file=sys.stderr)
    sys.exit(1)
DRIVER
