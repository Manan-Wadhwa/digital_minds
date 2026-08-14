#!/usr/bin/env bash
# Mirror result artifacts out of an ephemeral marimo sandbox, chunked base64
# over the execute-code channel. Written 2026-08-14 because molab storage is
# unreliable and the 2026-08-03 "recover the completed run off the ephemeral
# sandbox" scramble must not repeat.
#
# Usage:
#   MARIMO_EXEC=<execute-code.sh> URL=<sandbox url> TOKEN=<token> \
#     ./pull_from_sandbox.sh /local/repo/root [--once]
#
# Mirrors, relative to /marimo/repo: experiments/**/results* trees plus
# *_status.txt and *_driver.log. Skips files already local with matching
# size. Prints one line per pulled file and "CYCLE pulled=N".
set -euo pipefail
: "${MARIMO_EXEC:?}" "${URL:?}" "${TOKEN:?}"
LOCAL_ROOT="${1:?local root}"
MODE="${2:-}"

list_remote() {
  bash "$MARIMO_EXEC" --url "$URL" --token "$TOKEN" 2>/dev/null <<'PY' | grep '^F ' || true
import os, glob
seen = set()
pats = ["/marimo/repo/experiments/**/results*/**",
        "/marimo/repo/*_status.txt", "/marimo/repo/*_driver.log"]
for pat in pats:
    for p in glob.glob(pat, recursive=True):
        if os.path.isfile(p) and p not in seen:
            seen.add(p)
            print("F", os.path.getsize(p), p)
PY
}

pull_file() {
  local size="$1" rp="$2"
  local lp="$LOCAL_ROOT/${rp#/marimo/repo/}"
  if [ -f "$lp" ] && [ "$(wc -c < "$lp")" = "$size" ]; then
    return 1
  fi
  mkdir -p "$(dirname "$lp")"
  # 1 MB chunks, base64 wrapped at 16k chars/line: single giant lines truncate
  # on the exec channel (measured 2026-08-14), wrapped lines stream reliably
  # at ~1 min/MB. Smallest files first (see caller), so bulk never delays
  # science.
  local off=0 chunk=1000000 tmp="$lp.part"
  : > "$tmp"
  while [ "$off" -lt "$size" ]; do
    bash "$MARIMO_EXEC" --url "$URL" --token "$TOKEN" 2>/dev/null <<PY | sed -n '/^CHUNKSTART$/,/^CHUNKEND$/p' | grep -v '^CHUNK' | tr -d '\n' | base64 -d >> "$tmp"
import base64
fh = open("$rp", "rb")
fh.seek($off)
b = base64.b64encode(fh.read($chunk)).decode()
print("CHUNKSTART")
for i in range(0, len(b), 16000):
    print(b[i:i + 16000])
print("CHUNKEND")
PY
    off=$((off + chunk))
  done
  if [ "$(wc -c < "$tmp")" = "$size" ]; then
    mv "$tmp" "$lp"
    echo "pulled ${rp#/marimo/repo/} ($size)"
    return 0
  fi
  rm -f "$tmp"
  echo "FAILED ${rp#/marimo/repo/} (size mismatch)" >&2
  return 2
}

# Freshness beats bulk: every cycle re-checks all small files (status
# markers, result JSONs), but pulls at most ONE >1MB file per cycle so a
# ~12-min adapter grind can never delay marker/JSON mirroring by hours
# (added 2026-08-14 after the map's adapters made cycles multi-hour).
while :; do
  n=0 bulk=0
  while read -r _f size rp; do
    if [ "$size" -gt 1000000 ] && [ "$bulk" -ge 1 ]; then continue; fi
    if pull_file "$size" "$rp"; then
      n=$((n + 1))
      [ "$size" -gt 1000000 ] && bulk=$((bulk + 1))
    fi
  done < <(list_remote | sort -k2 -n)
  echo "CYCLE pulled=$n"
  [ "$MODE" = "--once" ] && break
  sleep 180
done
