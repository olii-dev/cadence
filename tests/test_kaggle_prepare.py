from pathlib import Path


def test_parallel_prep_keeps_per_shard_markers_and_ordered_merge():
    script = Path("scripts/kaggle_prepare.sh").read_text()
    assert '[[ -f "$out/stats.json" ]] || cadence-tokenize' in script
    assert 'xargs -r -P "$workers"' in script
    assert '(( workers > 4 )) && workers=4' in script
    assert 'inputs=("$SHARDS"/shard-*)' in script
    assert 'cadence-merge-shards "$ROOT/merged" "${inputs[@]}"' in script
