import json

import numpy as np

from scripts.compact_legacy_shards import compact_shard


def test_compacts_legacy_int64_npy_in_place(tmp_path):
    shard = tmp_path / "shard-000000"
    shard.mkdir()
    (shard / "stats.json").write_text("{}")
    expected = {}
    for split, values in {"train": [1, 2, 3], "validation": [], "test": [1758]}.items():
        np.save(shard / f"{split}.npy", np.asarray(values, dtype=np.int64))
        lines = [] if not values else [{"length": len(values) - 1}]
        (shard / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in lines))
        expected[split] = values
    compact_shard(shard)
    assert not list(shard.glob("*.npy"))
    assert json.loads((shard / "packed.json").read_text())["dtype"] == "uint16"
    for split, values in expected.items():
        assert np.fromfile(shard / f"{split}.bin", dtype=np.uint16).tolist() == values
