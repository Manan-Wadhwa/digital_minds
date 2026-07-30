#!/usr/bin/env bash
# Push repo code into the live marimo sandbox.
#
# The sandbox holds the GPU; this repo holds the code and the git history. They
# share no filesystem, so the repo stays the single source of truth and this
# script ships a snapshot of it into the kernel before every run. Results come
# back the other way and are committed here.
#
# Usage: scripts/sync_to_sandbox.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL="$REPO_ROOT/.agents/skills/marimo-pair/scripts/execute-code.sh"

: "${MARIMO_URL:?set MARIMO_URL}"
: "${MARIMO_SESSION:?set MARIMO_SESSION}"
: "${MARIMO_TOKEN_FILE:?set MARIMO_TOKEN_FILE}"

SHA="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
PAYLOAD="$(cd "$REPO_ROOT" && tar czf - src experiments | base64 -w0)"

python3 - "$PAYLOAD" "$SHA" "$SKILL" "$MARIMO_URL" "$MARIMO_SESSION" "$MARIMO_TOKEN_FILE" <<'DRIVER'
import subprocess, sys

payload, sha, skill, url, session, token_file = sys.argv[1:7]

code = f'''
import base64, io, os, pathlib, sys, tarfile

DEST = pathlib.Path("/marimo/repo")
DEST.mkdir(parents=True, exist_ok=True)
blob = base64.b64decode({payload!r})
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
    tf.extractall(DEST)

os.environ["CALIBRATION_GIT_SHA"] = {sha!r}
src = str(DEST / "src")
if src not in sys.path:
    sys.path.insert(0, src)

for mod in [m for m in list(sys.modules) if m.startswith("calibration")]:
    del sys.modules[mod]

import calibration
print("synced", calibration.__version__, "git", {sha!r})
print("files:", sum(1 for _ in DEST.rglob("*.py")))
'''

with open(token_file) as fh:
    token = fh.read().strip()

subprocess.run(
    ["bash", skill, "--url", url, "--session", session, "--token", token, "-"],
    input=code, text=True, check=True,
)
DRIVER
