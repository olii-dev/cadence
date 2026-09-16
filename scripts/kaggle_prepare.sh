#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/kaggle/working/cadence-data}
ARCHIVE="$ROOT/lmd_full.tar.gz"
SOURCE="$ROOT/source"
SHARDS="$ROOT/shards"
mkdir -p "$ROOT" "$SOURCE" "$SHARDS"
if [[ ! -f "$ARCHIVE" ]]; then
  curl -fL --retry 4 --retry-delay 5 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz' -o "$ARCHIVE"
fi
printf '%s  %s\n' '6fcfe2ac49ca08f3f214cec86ab138d4fc4dabcd7f27f491a838dae6db45a12b' "$ARCHIVE" | sha256sum -c -
if [[ ! -d "$SOURCE/lmd_full" ]]; then tar -xzf "$ARCHIVE" -C "$SOURCE"; fi
count=$(find "$SOURCE/lmd_full" -type f -name '*.mid' | wc -l)
[[ "$count" -eq 178561 ]] || { echo "unexpected MIDI count: $count" >&2; exit 1; }
# Process in bounded shards. Existing stats files are completion markers, so notebook restarts resume safely.
for start in $(seq 0 5000 175000); do
  out=$(printf '%s/shard-%06d' "$SHARDS" "$start")
  [[ -f "$out/stats.json" ]] || cadence-tokenize "$SOURCE/lmd_full" "$out" --start "$start" --limit 5000
 done
out="$SHARDS/shard-175000"
[[ -f "$out/stats.json" ]] || cadence-tokenize "$SOURCE/lmd_full" "$out" --start 175000 --limit 3561
inputs=("$SHARDS"/shard-*)
cadence-merge-shards "$ROOT/merged" "${inputs[@]}"
echo "Prepared corpus at $ROOT/merged"
