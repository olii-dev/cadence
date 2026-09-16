#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/kaggle/working/cadence-data}
ARCHIVE="$ROOT/lmd_full.tar.gz"
SOURCE="$ROOT/source"
SHARDS="$ROOT/shards"
PERSIST_DATASET=${CADENCE_KAGGLE_DATASET:-}
# Remove only interrupted shard directories. A completed stats.json marker is never deleted.
if [[ -d "$SHARDS" ]]; then
  while IFS= read -r -d '' shard; do
    [[ -f "$shard/stats.json" ]] || { echo "Removing incomplete shard: $shard"; rm -rf -- "$shard"; }
  done < <(find "$SHARDS" -mindepth 1 -maxdepth 1 -type d -name 'shard-*' -print0)
fi
mkdir -p "$ROOT" "$SOURCE" "$SHARDS"
if [[ ! -f "$ARCHIVE" ]]; then
  curl -fL --retry 4 --retry-delay 5 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz' -o "$ARCHIVE"
fi
printf '%s  %s\n' '6fcfe2ac49ca08f3f214cec86ab138d4fc4dabcd7f27f491a838dae6db45a12b' "$ARCHIVE" | sha256sum -c -
if [[ ! -d "$SOURCE/lmd_full" ]]; then tar -xzf "$ARCHIVE" -C "$SOURCE"; fi
count=$(find "$SOURCE/lmd_full" -type f -name '*.mid' | wc -l)
[[ "$count" -eq 178561 ]] || { echo "unexpected MIDI count: $count" >&2; exit 1; }
workers=${CADENCE_PREP_WORKERS:-$(nproc)}
(( workers > 4 )) && workers=4
(( workers < 1 )) && workers=1
jobs="$ROOT/shard-jobs.tsv"
: > "$jobs"
for start in $(seq 0 5000 170000); do printf '%s\t%s\n' "$start" 5000 >> "$jobs"; done
printf '%s\t%s\n' 175000 3561 >> "$jobs"
export SOURCE SHARDS
process_shard() {
  local start=$1 limit=$2 out
  out=$(printf '%s/shard-%06d' "$SHARDS" "$start")
  [[ -f "$out/stats.json" ]] || cadence-tokenize "$SOURCE/lmd_full" "$out" --start "$start" --limit "$limit"
}
export -f process_shard
persist_shards() {
  [[ -n "$PERSIST_DATASET" ]] || return 0
  command -v kaggle >/dev/null || { echo 'kaggle CLI is required for incremental persistence' >&2; return 1; }
  local completed message owner_slug=${PERSIST_DATASET#datasets/}
  completed=$(find "$SHARDS" -mindepth 2 -maxdepth 2 -type f -name stats.json | wc -l)
  message="Cadence corpus: ${completed} completed deterministic shards"
  python - "$SHARDS" "$owner_slug" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
metadata = {
    "id": sys.argv[2],
    "title": "Cadence Lakh MIDI Token Shards",
    "licenses": [{"name": "CC-BY-4.0"}],
}
(root / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
  if kaggle datasets files "$owner_slug" >/dev/null 2>&1; then
    kaggle datasets version -p "$SHARDS" -m "$message" --dir-mode zip -q
  else
    kaggle datasets create -p "$SHARDS" --private --dir-mode zip -q
  fi
  printf 'Persisted %s completed shards to %s\n' "$completed" "$owner_slug"
}
printf 'Preparing %s MIDI files with %s worker(s)\n' "$count" "$workers"
# Run one worker-width wave at a time. Persisting between waves bounds session-loss
# exposure to the currently running wave without changing deterministic shard offsets.
mapfile -t job_lines < "$jobs"
for ((i=0; i<${#job_lines[@]}; i+=workers)); do
  printf '%s\n' "${job_lines[@]:i:workers}" | xargs -r -P "$workers" -n 2 bash -c 'process_shard "$@"' _
  persist_shards
done
inputs=("$SHARDS"/shard-*)
cadence-merge-shards "$ROOT/merged" "${inputs[@]}"
echo "Prepared corpus at $ROOT/merged"
