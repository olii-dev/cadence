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
# Each worker owns one deterministic output directory. stats.json is written last and
# remains the completion marker; interrupted runs repeat only incomplete shards.
# Global deduplication stays a single ordered merge after every worker exits.
workers=${CADENCE_PREP_WORKERS:-$(nproc)}
(( workers > 4 )) && workers=4
(( workers < 1 )) && workers=1
jobs="$ROOT/shard-jobs.tsv"
: > "$jobs"
for start in $(seq 0 5000 175000); do printf '%s\t%s\n' "$start" 5000 >> "$jobs"; done
printf '%s\t%s\n' 175000 3561 >> "$jobs"
export SOURCE SHARDS
process_shard() {
  local start=$1 limit=$2 out
  out=$(printf '%s/shard-%06d' "$SHARDS" "$start")
  [[ -f "$out/stats.json" ]] || cadence-tokenize "$SOURCE/lmd_full" "$out" --start "$start" --limit "$limit"
}
export -f process_shard
printf 'Preparing %s MIDI files with %s worker(s)\n' "$count" "$workers"
xargs -r -P "$workers" -n 2 bash -c 'process_shard "$@"' _ < "$jobs"
inputs=("$SHARDS"/shard-*)
cadence-merge-shards "$ROOT/merged" "${inputs[@]}"
echo "Prepared corpus at $ROOT/merged"
